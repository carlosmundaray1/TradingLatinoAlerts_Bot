#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GRID SEARCH · Búsqueda de Parámetros Óptimos
================================================================================
Prueba combinaciones de SIGNAL_SCORE_THRESHOLD, HYBRID_CONFIDENCE_THRESHOLD,
y W_HMM_REGIME para maximizar detección sin perder win rate.

Estrategia:
  - Pre-calcula datos, indicadores y HMM una sola vez (son caros)
  - Varía solo los parámetros en las funciones de señal y alerta (son rápidos)
  - Prueba 75 combinaciones en ~5-10 minutos
================================================================================
"""
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTAR MÓDULO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util as _imp_util
_MODULE_PATH = SCRIPT_DIR / "tradinglatino_hmm_clean.py"
_SPEC = _imp_util.spec_from_file_location("hmm_clean", _MODULE_PATH)
_HMM = _imp_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HMM)

# Re-exportar funciones
compute_all_indicators = _HMM.compute_all_indicators
build_hmm_features = _HMM.build_hmm_features
fit_hmm_sliding = _HMM.fit_hmm_sliding
compute_signal = _HMM.compute_signal
_classify_regime_bias = _HMM._classify_regime_bias
_format_date = _HMM._format_date
verify_signals_historically = _HMM.verify_signals_historically
compute_hybrid_alert = _HMM.compute_hybrid_alert
compute_precursor_signals = _HMM.compute_precursor_signals
load_data = _HMM.load_data
HYBRID_LOOKBACK = _HMM.HYBRID_LOOKBACK

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
ASSET = "BTC-USD"
TIMEFRAMES = ["1d", "1wk"]
PERIODS = {"1d": "5y", "1wk": "10y"}

# Grid de parámetros a probar
GRID_SIGNAL_THRESHOLD = list(range(55, 76, 5))    # 55, 60, 65, 70, 75
GRID_HYBRID_THRESHOLD = list(range(40, 65, 5))     # 40, 45, 50, 55, 60
GRID_W_HMM = list(range(10, 25, 5))                # 10, 15, 20

# Función objetivo: peso de cada métrica en el scoring
W_DETECTION = 0.35     # Detección HMM
W_HYBRID = 0.25        # Detección Híbrida
W_WINRATE = 0.25       # Win Rate
W_FP_FN = 0.15         # Penalización por FP+FN (negativo: minimizar)

OUTPUT_FILE = "grid_search_results.json"
OUTPUT_SUMMARY = "grid_search_summary.txt"


def load_data_extended(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Carga datos con períodos extendidos."""
    _HMM.PERIOD_1D = PERIODS.get("1d", "2y")
    _HMM.PERIOD_1W = PERIODS.get("1wk", "4y")
    return load_data(asset, timeframe)


def find_signal_changes(df: pd.DataFrame) -> List[Dict]:
    """Encuentra cambios de señal LONG/SHORT/FLAT."""
    changes = []
    prev_signal = "FLAT"
    for i in range(len(df)):
        is_long = bool(df["signal_long"].iloc[i]) if "signal_long" in df.columns else False
        is_short = bool(df["signal_short"].iloc[i]) if "signal_short" in df.columns else False
        current_signal = "LONG" if is_long else ("SHORT" if is_short else "FLAT")
        if current_signal != prev_signal:
            changes.append({
                "idx": i, "date": df.index[i],
                "from_signal": prev_signal, "to_signal": current_signal,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
            })
            prev_signal = current_signal
    return changes


def find_regime_changes_detailed(states: np.ndarray, index, state_summary: pd.DataFrame) -> List[Dict]:
    """Encuentra cambios de régimen."""
    desc_map = {int(r["state"]): r["description"] for _, r in state_summary.iterrows()}
    changes = []
    prev_state = states[0]
    for i in range(1, len(states)):
        if states[i] != prev_state:
            from_desc = desc_map.get(int(prev_state), f"R{prev_state}")
            to_desc = desc_map.get(int(states[i]), f"R{states[i]}")
            changes.append({
                "idx": i, "date": index[i],
                "from_state": int(prev_state), "to_state": int(states[i]),
                "from_desc": from_desc, "to_desc": to_desc,
                "from_bias": _classify_regime_bias(from_desc),
                "to_bias": _classify_regime_bias(to_desc),
            })
            prev_state = states[i]
    return changes


def cross_reference(regime_changes: List[Dict], signal_changes: List[Dict], max_lag=5) -> Dict:
    """Cruza cambios de régimen vs señal."""
    results = []
    regime_used = set()
    real_sig = [s for s in signal_changes if s["from_signal"] != s["to_signal"]]
    
    for sc in real_sig:
        best = None
        best_lag = None
        best_ri = -1
        for ri, rc in enumerate(regime_changes):
            if ri in regime_used: continue
            if rc["idx"] <= sc["idx"] and rc["idx"] >= sc["idx"] - max_lag:
                lag = sc["idx"] - rc["idx"]
                if best is None or lag < best_lag:
                    best, best_lag, best_ri = rc, lag, ri
        if best is not None:
            regime_used.add(best_ri)
            results.append({"type": "correcta"})
        else:
            results.append({"type": "no_regime"})
    
    for ri in range(len(regime_changes)):
        if ri not in regime_used:
            results.append({"type": "no_signal"})
    
    detected = sum(1 for r in results if r["type"] == "correcta")
    false_pos = sum(1 for r in results if r["type"] == "no_signal")
    false_neg = sum(1 for r in results if r["type"] == "no_regime")
    total = len(real_sig)
    
    return {
        "detected": detected, "total": total,
        "detection_rate": round(detected / total * 100, 1) if total else 0,
        "false_positives": false_pos, "false_negatives": false_neg,
    }

def find_hybrid_active_count(df: pd.DataFrame) -> int:
    """Cuenta alertas híbridas activas."""
    if "hybrid_alert_active" not in df.columns:
        return 0
    # Contar inicios de períodos activos
    active = df["hybrid_alert_active"].astype(bool)
    starts = (active & ~active.shift(1).fillna(False)).sum()
    return int(starts)


def cross_reference_hybrid_simple(df: pd.DataFrame, signal_changes: List[Dict], max_lag=5) -> Dict:
    """Versión simplificada: cuenta cuántos cambios de señal tienen alerta híbrida previa."""
    if "hybrid_alert_active" not in df.columns:
        return {"detected": 0, "total": 0, "detection_rate": 0.0}
    
    real_sig = [s for s in signal_changes if s["from_signal"] != s["to_signal"]]
    detected = 0
    aligned = 0
    
    for sc in real_sig:
        idx_start = max(0, sc["idx"] - max_lag)
        window = df.iloc[idx_start:sc["idx"] + 1]
        if window["hybrid_alert_active"].any():
            detected += 1
            # Check direction alignment
            hybrid_long = window["hybrid_alert_long"].any()
            hybrid_short = window["hybrid_alert_short"].any()
            if (sc["to_signal"] == "LONG" and hybrid_long) or \
               (sc["to_signal"] == "SHORT" and hybrid_short):
                aligned += 1
    
    total = len(real_sig)
    return {
        "detected": detected, "total": total,
        "detection_rate": round(detected / total * 100, 1) if total else 0,
        "aligned": aligned,
    }


def compute_win_rate_summary(df: pd.DataFrame, timeframe: str) -> Dict:
    """Calcula win rate resumido para el timeframe."""
    try:
        ver = verify_signals_historically(df, timeframe)
        return {
            "overall_wr": ver["overall_win_rate"],
            "total_signals": ver["total_signals"],
            "long_wr": ver["stats"]["LONG"]["win_rate"],
            "short_wr": ver["stats"]["SHORT"]["win_rate"],
            "long_n": ver["stats"]["LONG"]["num_signals"],
            "short_n": ver["stats"]["SHORT"]["num_signals"],
        }
    except Exception as e:
        return {"overall_wr": 0, "total_signals": 0, "error": str(e)}


def score_combination(metrics: Dict) -> float:
    """Calcula score compuesto para una combinación de parámetros."""
    det = metrics.get("detection_rate_1d", 0) / 100.0
    hyb = metrics.get("hybrid_rate_1d", 0) / 100.0
    wr = metrics.get("wr_1d", 0) / 100.0
    fp = metrics.get("fp_1d", 0)
    fn = metrics.get("fn_1d", 0)
    total_sig = metrics.get("total_signals_1d", 1)
    
    fp_fn_ratio = 1.0 - min(1.0, (fp + fn) / max(total_sig, 1))
    
    score = (det * W_DETECTION + hyb * W_HYBRID + wr * W_WINRATE + fp_fn_ratio * W_FP_FN) * 100
    return round(score, 1)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL DE GRID SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def safeprint(msg: str, log_lines: List[str] = None) -> None:
    """Print seguro que maneja caracteres Unicode."""
    try:
        print(msg)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe)
    if log_lines is not None:
        log_lines.append(msg)

def run_grid_search() -> None:
    log_lines: List[str] = []
    safeprint("=" * 70, log_lines)
    safeprint("  GRID SEARCH - Busqueda de Parametros Optimos", log_lines)
    safeprint("=" * 70, log_lines)
    safeprint(f"  Activo: {ASSET}", log_lines)
    safeprint(f"  Timeframes: {TIMEFRAMES}", log_lines)
    safeprint(f"  Parametros a probar:", log_lines)
    safeprint(f"    SIGNAL_SCORE_THRESHOLD: {GRID_SIGNAL_THRESHOLD}", log_lines)
    safeprint(f"    HYBRID_CONFIDENCE_THRESHOLD: {GRID_HYBRID_THRESHOLD}", log_lines)
    safeprint(f"    W_HMM_REGIME: {GRID_W_HMM}", log_lines)
    total_combos = len(GRID_SIGNAL_THRESHOLD) * len(GRID_HYBRID_THRESHOLD) * len(GRID_W_HMM)
    safeprint(f"  Total combinaciones: {total_combos}", log_lines)
    safeprint("", log_lines)
    
    results = {}
    tf_data = {}  # Cache de datos por timeframe (pre-calculados)
    
    # ---- FASE 1: PRE-CALCULAR DATOS PESADOS (UNA SOLA VEZ) ----
    safeprint("-" * 70, log_lines)
    safeprint("  FASE 1: Pre-calculando datos, indicadores y HMM...", log_lines)
    safeprint("-" * 70, log_lines)
    
    for tf in TIMEFRAMES:
        safeprint(f"\n  -- Timeframe: {tf} --", log_lines)
        t0 = time.time()
        
        # Cargar datos
        df = load_data_extended(ASSET, tf)
        if df is None or len(df) < 100:
            print(f"    ERROR: Datos insuficientes ({len(df) if df is not None else 0})")
            continue
        
        safeprint(f"    Datos: {len(df)} velas ({time.time()-t0:.0f}s)", log_lines)
        t1 = time.time()
        
        # Calcular indicadores (incluye signal_scores base con threshold DEFAULT)
        df = compute_all_indicators(df)
        safeprint(f"    Indicadores: OK ({time.time()-t1:.0f}s)", log_lines)
        t2 = time.time()
        
        # Construir features HMM
        features = build_hmm_features(df)
        safeprint(f"    Features HMM: {features.shape[1]} columnas ({time.time()-t2:.0f}s)", log_lines)
        t3 = time.time()
        
        # Entrenar HMM con ventana deslizante
        window = 500 if tf == "1d" else 300
        model, states, state_summary, _, trans_mat, probas, _, _ = fit_hmm_sliding(
            features, window_size=window, stride=50
        )
        if model is None or len(states) == 0:
            print(f"    ERROR: HMM falló")
            continue
        safeprint(f"    HMM entrenado: {int(states.max())+1} estados ({time.time()-t3:.0f}s)", log_lines)
        
        # Alinear df con states
        df = df.iloc[:len(states)].copy()
        df["regime"] = states
        
        # Pre-calcular precursores (no dependen de parámetros)
        df = compute_precursor_signals(df)
        
        # Detectar cambios de régimen (no dependen de parámetros)
        regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
        safeprint(f"    Cambios de regimen: {len(regime_changes)}", log_lines)
        
        # Guardar en caché
        tf_data[tf] = {
            "df": df, "states": states, "state_summary": state_summary,
            "trans_mat": trans_mat, "probas": probas,
            "regime_changes": regime_changes,
        }
        safeprint(f"    Total: {time.time()-t0:.0f}s", log_lines)
    
    if not tf_data:
        safeprint("\n  ERROR: No hay datos para ningun timeframe.", log_lines)
        return
    
    # ---- FASE 2: PROBAR COMBINACIONES ----
    safeprint("\n" + "-" * 70, log_lines)
    safeprint("  FASE 2: Probando combinaciones de parametros...", log_lines)
    safeprint("-" * 70, log_lines)
    
    all_results = []
    
    for sig_th, hyb_th, w_hmm in itertools.product(
        GRID_SIGNAL_THRESHOLD, GRID_HYBRID_THRESHOLD, GRID_W_HMM
    ):
        combo_key = f"T{sig_th}_H{hyb_th}_W{w_hmm}"
        combo_start = time.time()
        
        # Modificar parametros en el modulo
        _HMM.SIGNAL_SCORE_THRESHOLD = sig_th
        _HMM.HYBRID_CONFIDENCE_THRESHOLD = hyb_th
        _HMM.W_HMM_REGIME = w_hmm
        
        metrics = {"signal_threshold": sig_th, "hybrid_threshold": hyb_th, "w_hmm": w_hmm}
        
        for tf in TIMEFRAMES:
            if tf not in tf_data:
                continue
            d = tf_data[tf]
            df = d["df"].copy()
            
            # Re-calcular signal_scores con HMM (usa W_HMM_REGIME)
            df = _HMM.compute_signal_scores_with_hmm(
                df, d["state_summary"], d["probas"], d["trans_mat"]
            )
            
            # Encontrar cambios de senal (dependen de SIGNAL_SCORE_THRESHOLD)
            signal_changes = find_signal_changes(df)
            
            # Cross-reference regimen vs senal
            cross_ref = cross_reference(d["regime_changes"], signal_changes, max_lag=5)
            metrics[f"detection_rate_{tf}"] = cross_ref["detection_rate"]
            metrics[f"fp_{tf}"] = cross_ref["false_positives"]
            metrics[f"fn_{tf}"] = cross_ref["false_negatives"]
            metrics[f"total_signals_{tf}"] = cross_ref["total"]
            
            # Calcular win rate
            wr_data = compute_win_rate_summary(df, tf)
            metrics[f"wr_{tf}"] = wr_data["overall_wr"]
            metrics[f"wr_signals_{tf}"] = wr_data["total_signals"]
            
            # Alerta hibrida (usa HYBRID_CONFIDENCE_THRESHOLD)
            df_hyb = _HMM.compute_hybrid_alert(
                df, d["states"], d["state_summary"], d["probas"], d["trans_mat"]
            )
            hybrid_cross = cross_reference_hybrid_simple(df_hyb, signal_changes, max_lag=5)
            metrics[f"hybrid_rate_{tf}"] = hybrid_cross["detection_rate"]
            metrics[f"hybrid_detected_{tf}"] = hybrid_cross["detected"]
            metrics[f"hybrid_aligned_{tf}"] = hybrid_cross["aligned"]
        
        # Calcular score compuesto (basado en 1d principalmente)
        score = score_combination(metrics)
        metrics["score"] = score
        all_results.append(metrics)
        
        elapsed = time.time() - combo_start
        det = metrics.get("detection_rate_1d", 0)
        hyb = metrics.get("hybrid_rate_1d", 0)
        wr = metrics.get("wr_1d", 0)
        safeprint(f"  [{combo_key}] Score={score:.1f}  Det={det:.1f}%  Hybrid={hyb:.1f}%  WR={wr:.1f}%  ({elapsed:.1f}s)", log_lines)
    
    # ── FASE 3: ENCONTRAR ÓPTIMO ──
    print("\n" + "━" * 70)
    print("  FASE 3: Analizando resultados...")
    print("━" * 70)
    
    all_results.sort(key=lambda r: r["score"], reverse=True)
    
    # Top 10
    print(f"\n  TOP 10 COMBINACIONES:\n")
    print(f"  {'#':>3} {'Threshold':>10} {'Hybrid':>8} {'W_HMM':>6} {'Score':>7} {'Det 1d':>8} {'Hyb 1d':>8} {'WR 1d':>7} {'FP 1d':>6} {'FN 1d':>6}")
    print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6}")
    
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1:>3} {r['signal_threshold']:>10} {r['hybrid_threshold']:>8} {r['w_hmm']:>6} "
              f"{r['score']:>7.1f} {r.get('detection_rate_1d',0):>7.1f}% "
              f"{r.get('hybrid_rate_1d',0):>7.1f}% {r.get('wr_1d',0):>6.1f}% "
              f"{r.get('fp_1d',0):>5} {r.get('fn_1d',0):>5}")
    
    best = all_results[0]
    print(f"\n  🏆 MEJOR COMBINACIÓN:")
    print(f"     SIGNAL_SCORE_THRESHOLD = {best['signal_threshold']}")
    print(f"     HYBRID_CONFIDENCE_THRESHOLD = {best['hybrid_threshold']}")
    print(f"     W_HMM_REGIME = {best['w_hmm']}")
    print(f"     Score: {best['score']:.1f}")
    print(f"     Detección 1d: {best.get('detection_rate_1d',0):.1f}%")
    print(f"     Híbrido 1d: {best.get('hybrid_rate_1d',0):.1f}%")
    print(f"     Win Rate 1d: {best.get('wr_1d',0):.1f}%")
    print(f"     FP/FN 1d: {best.get('fp_1d',0)}/{best.get('fn_1d',0)}")
    
    # Guardar resultados
    output = {
        "grid_config": {
            "signal_threshold": GRID_SIGNAL_THRESHOLD,
            "hybrid_threshold": GRID_HYBRID_THRESHOLD,
            "w_hmm": GRID_W_HMM,
        },
        "best": best,
        "top_10": all_results[:10],
        "all_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Resultados guardados: {OUTPUT_FILE}")
    
    # Guardar resumen
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(f"GRID SEARCH RESULTS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Asset: {ASSET}\n")
        f.write(f"Timeframes: {TIMEFRAMES}\n")
        f.write(f"Combinations: {len(all_results)}\n\n")
        f.write(f"TOP 10:\n")
        f.write(f"{'Rank':>5} {'Thresh':>8} {'Hybrid':>8} {'W_HMM':>6} {'Score':>7} {'Det1d':>7} {'Hyb1d':>7} {'WR1d':>6} {'FP1d':>5} {'FN1d':>5}\n")
        f.write(f"{'-'*65}\n")
        for i, r in enumerate(all_results[:10]):
            f.write(f"{i+1:>5} {r['signal_threshold']:>8} {r['hybrid_threshold']:>8} "
                    f"{r['w_hmm']:>6} {r['score']:>7.1f} {r.get('detection_rate_1d',0):>6.1f}% "
                    f"{r.get('hybrid_rate_1d',0):>6.1f}% {r.get('wr_1d',0):>5.1f}% "
                    f"{r.get('fp_1d',0):>4} {r.get('fn_1d',0):>4}\n")
        
        f.write(f"\n🏆 BEST: T={best['signal_threshold']} H={best['hybrid_threshold']} W={best['w_hmm']}\n")
        f.write(f"   Score: {best['score']:.1f}\n")
        f.write(f"   Det 1d: {best.get('detection_rate_1d',0):.1f}%\n")
        f.write(f"   Hybrid 1d: {best.get('hybrid_rate_1d',0):.1f}%\n")
        f.write(f"   WR 1d: {best.get('wr_1d',0):.1f}%\n")
        f.write(f"   FP/FN 1d: {best.get('fp_1d',0)}/{best.get('fn_1d',0)}\n")
    
    print(f"  Resumen guardado: {OUTPUT_SUMMARY}")
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH COMPLETADO")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_grid_search()
