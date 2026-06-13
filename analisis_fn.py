#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALISIS DE FALSOS NEGATIVOS (FN)
================================================================================
Determina POR QUE el HMM no detecta ciertos cambios de senal.
Compara las caracteristicas de senales DETECTADAS vs NO DETECTADAS (FN).
Incluye analisis de features individuales del HMM.
================================================================================
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Importar pipeline ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util as _imp_util
_MODULE_PATH = SCRIPT_DIR / "tradinglatino_hmm_clean.py"
_SPEC = _imp_util.spec_from_file_location("hmm_clean", _MODULE_PATH)
_HMM = _imp_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HMM)

compute_all_indicators = _HMM.compute_all_indicators
build_hmm_features = _HMM.build_hmm_features

# ── Importar funciones de simulacion ──
_SIM_PATH = SCRIPT_DIR / "simulacion_alertas_tendencia.py"
_SIM_SPEC = _imp_util.spec_from_file_location("simulacion", _SIM_PATH)
_SIM = _imp_util.module_from_spec(_SIM_SPEC)
_SIM_SPEC.loader.exec_module(_SIM)

load_data_extended = _SIM.load_data_extended
find_regime_changes_detailed = _SIM.find_regime_changes_detailed
find_signal_changes = _SIM.find_signal_changes
cross_reference_changes = _SIM.cross_reference_changes
_format_date = _SIM._format_date

ASSET = "BTC-USD"
TIMEFRAMES = ["1d", "1wk"]
PERIODS = {"1d": "5y", "1wk": "10y"}


def analyze_fn(timeframe: str) -> Dict[str, Any]:
    """Pipeline completo de analisis de FN para un timeframe."""
    print(f"\n{'='*70}")
    print(f"  ANALISIS FN: {timeframe}")
    print(f"{'='*70}")

    # 1) Cargar datos
    print(f"\n  Cargando datos ({PERIODS[timeframe]})...")
    df = load_data_extended(ASSET, timeframe)
    if df is None or len(df) < 100:
        print(f"  ERROR: Datos insuficientes")
        return {}

    # 2) Indicadores
    print(f"  Calculando indicadores...")
    df = compute_all_indicators(df)

    # 3) HMM sliding
    print(f"  Entrenando HMM sliding window...")
    features = build_hmm_features(df)
    model, states, state_summary, bic_df, trans_mat, probas, _, _ = _HMM.fit_hmm_sliding(
        features, window_size=500, stride=50
    )
    if model is None:
        print(f"  ERROR: HMM fallo")
        return {}

    df = df.iloc[:len(states)].copy()
    features = features.iloc[:len(states)].copy()
    df["regime"] = states

    # 4) Aplicar score HMM
    df = _HMM.compute_signal_scores_with_hmm(df, state_summary, probas, trans_mat)

    # 5) Cambios de regimen y senal
    regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
    signal_changes = find_signal_changes(df)
    signal_changes_real = [s for s in signal_changes if s["from_signal"] != s["to_signal"]]

    # 6) Cross reference (REUTILIZAMOS la funcion de la simulacion)
    cross_ref = cross_reference_changes(regime_changes, signal_changes, max_lag_bars=5)

    # 7) Clasificar usando los resultados del cross_reference
    fn_indices = set()
    det_indices = set()

    for r in cross_ref["results"]:
        if r["type"] == "no_regime":
            # FN: cambio de senal SIN cambio de regimen previo
            if r.get("signal_change"):
                fn_indices.add(r["signal_change"]["idx"])
        elif r["type"] == "correcta":
            # Detectado correctamente
            if r.get("signal_change"):
                det_indices.add(r["signal_change"]["idx"])

    # Verificar que no nos perdimos ningun cambio de senal
    all_classified = fn_indices | det_indices
    all_signal_idxs = set(sc["idx"] for sc in signal_changes_real)
    unclassified = all_signal_idxs - all_classified
    if unclassified:
        print(f"  WARNING: {len(unclassified)} cambios de senal no clasificados (lag>5?)")

    print(f"\n  Cambios de senal totales: {len(signal_changes_real)}")
    print(f"  Detectados: {len(det_indices)}")
    print(f"  FALSOS NEGATIVOS: {len(fn_indices)} ({len(fn_indices)/len(signal_changes_real)*100:.1f}%)")

    # ── 8) Extraer datos para FN y detectados ──
    fn_data = []
    for idx in sorted(fn_indices):
        d = _extract_case_data(idx, df, features, states, state_summary, probas, trans_mat, regime_changes)
        if d:
            fn_data.append(d)

    det_data = []
    for idx in sorted(det_indices):
        d = _extract_case_data(idx, df, features, states, state_summary, probas, trans_mat, regime_changes)
        if d:
            det_data.append(d)

    return {
        "timeframe": timeframe,
        "total_signals": len(signal_changes_real),
        "detected": len(det_indices),
        "fn": len(fn_indices),
        "fn_rate": round(len(fn_indices) / len(signal_changes_real) * 100, 1) if signal_changes_real else 0,
        "det_rate": round(len(det_indices) / len(signal_changes_real) * 100, 1) if signal_changes_real else 0,
        "fn_data": fn_data,
        "det_data": det_data,
        "state_summary": state_summary,
        "n_states": len(state_summary) if state_summary is not None else 0,
        "regime_changes_count": len(regime_changes),
        "cross_ref": cross_ref,
    }


def _extract_case_data(idx, df, features, states, state_summary, probas, trans_mat, regime_changes):
    """Extrae datos completos de un caso (FN o detectado) en el indice dado."""
    if idx < 5 or idx >= len(df) - 1:
        return None

    # Datos basicos
    current_state = int(states[idx-1]) if idx-1 < len(states) else -1
    current_bias = "unknown"
    if not state_summary.empty and current_state >= 0:
        row = state_summary[state_summary["state"] == current_state]
        if not row.empty:
            current_bias = _HMM._classify_regime_bias(row.iloc[0]["description"])

    # Senal en este punto
    is_long = bool(df["signal_long"].iloc[idx-1]) if "signal_long" in df.columns else False
    is_short = bool(df["signal_short"].iloc[idx-1]) if "signal_short" in df.columns else False
    current_signal = "LONG" if is_long else ("SHORT" if is_short else "FLAT")

    # Senal en idx (despues del cambio)
    is_long_after = bool(df["signal_long"].iloc[idx]) if "signal_long" in df.columns else False
    is_short_after = bool(df["signal_short"].iloc[idx]) if "signal_short" in df.columns else False
    after_signal = "LONG" if is_long_after else ("SHORT" if is_short_after else "FLAT")

    needed_bias = "bullish" if after_signal == "LONG" else ("bearish" if after_signal == "SHORT" else "neutral")

    # Probabilidades de estado (especificas por bias)
    prob_max = 0.0
    prob_bullish_val = 0.0
    prob_bearish_val = 0.0
    if len(probas) > idx and probas.ndim > 1:
        prob_max = float(probas[idx, :].max())
        n_states_p = probas.shape[1]
        # Clasificar estados por bias
        bullish_states = []
        bearish_states = []
        for _, r in state_summary.iterrows():
            s = int(r["state"])
            bias = _HMM._classify_regime_bias(r["description"])
            if bias == "bullish" and s < n_states_p:
                bullish_states.append(s)
            elif bias == "bearish" and s < n_states_p:
                bearish_states.append(s)
        if bullish_states:
            prob_bullish_val = float(sum(probas[idx, s] for s in bullish_states))
        if bearish_states:
            prob_bearish_val = float(sum(probas[idx, s] for s in bearish_states))

    # Scores
    score_long = float(df["signal_score_long"].iloc[idx-1]) if "signal_score_long" in df.columns else 0
    score_short = float(df["signal_score_short"].iloc[idx-1]) if "signal_score_short" in df.columns else 0
    score_diff = abs(score_long - score_short)

    # Volatilidad (features del HMM)
    vol_20_val = float(features["vol_20"].iloc[idx]) if "vol_20" in features.columns else 0
    atr_val = float(df["atr"].iloc[idx]) if "atr" in df.columns else 0
    atr_norm = float(features["atr_norm"].iloc[idx]) if "atr_norm" in features.columns else 0

    # Features direccionales del HMM
    di_spread = float(features["di_spread"].iloc[idx]) if "di_spread" in features.columns else 0
    score_net = float(features["score_net"].iloc[idx]) if "score_net" in features.columns else 0
    ema_spread = float(features["ema_spread_raw"].iloc[idx]) if "ema_spread_raw" in features.columns else 0
    cumret = float(features["cumret_20"].iloc[idx]) if "cumret_20" in features.columns else 0
    momentum = float(features["momentum_20"].iloc[idx]) if "momentum_20" in features.columns else 0
    rsi_norm = float(features["rsi_norm"].iloc[idx]) if "rsi_norm" in features.columns else 0
    pos_in_range = float(features["pos_in_range"].iloc[idx]) if "pos_in_range" in features.columns else 0
    adx = float(features["adx_scaled"].iloc[idx]) if "adx_scaled" in features.columns else 0
    vol_rel = float(features["vol_rel_20"].iloc[idx]) if "vol_rel_20" in features.columns else 0

    # Cambio de regime reciente
    recent_regime_change = False
    regime_change_lag = None
    regime_change_from = None
    regime_change_to = None
    for rc in reversed(regime_changes):
        if 0 <= idx - rc["idx"] <= 10:
            recent_regime_change = True
            regime_change_lag = idx - rc["idx"]
            regime_change_from = rc["from_desc"]
            regime_change_to = rc["to_desc"]
            break

    return {
        "idx": idx,
        "date": _format_date(df.index[idx]),
        "current_signal": current_signal,
        "after_signal": after_signal,
        "needed_bias": needed_bias,
        "current_state": current_state,
        "current_bias": current_bias,
        "prob_max": prob_max,
        "prob_bullish": prob_bullish_val,
        "prob_bearish": prob_bearish_val,
        "score_long_before": score_long,
        "score_short_before": score_short,
        "score_diff": score_diff,
        # Features del HMM
        "vol_20": vol_20_val,
        "atr": atr_val,
        "atr_norm": atr_norm,
        "di_spread": di_spread,
        "score_net": score_net,
        "ema_spread_raw": ema_spread,
        "cumret_20": cumret,
        "momentum_20": momentum,
        "rsi_norm": rsi_norm,
        "pos_in_range": pos_in_range,
        "adx_scaled": adx,
        "vol_rel_20": vol_rel,
        # Cambio de regimen
        "recent_regime_change": recent_regime_change,
        "regime_change_lag": regime_change_lag,
        "regime_change_from": regime_change_from,
        "regime_change_to": regime_change_to,
    }


def print_analysis(results: Dict[str, Any]) -> None:
    """Imprime el analisis de FN con todas las metricas y features."""
    tf = results["timeframe"]

    print(f"\n{'='*70}")
    print(f"  RESULTADOS FN: {tf}")
    print(f"{'='*70}")
    print(f"  Total cambios de senal:  {results['total_signals']}")
    print(f"  Detectados:              {results['detected']} ({results['det_rate']}%)")
    print(f"  FALSOS NEGATIVOS:        {results['fn']} ({results['fn_rate']}%)")
    print(f"  Cambios de regimen HMM:  {results['regime_changes_count']}")
    print(f"  Estados HMM:             {results['n_states']}")

    fn_data = results["fn_data"]
    det_data = results["det_data"]

    if not fn_data:
        print(f"\n  No hay FNs que analizar!")
        return

    # ── 1. BIAS DEL REGIMEN ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 1: BIAS DEL REGIMEN ACTUAL")
    print(f"  {'─'*60}")

    fn_bias_match = sum(1 for f in fn_data if f["current_bias"] == f["needed_bias"])
    fn_bias_mismatch = sum(1 for f in fn_data if f["current_bias"] != f["needed_bias"] and f["current_bias"] != "unknown")
    fn_bias_neutral = sum(1 for f in fn_data if f["current_bias"] == "neutral")
    fn_bias_unknown = sum(1 for f in fn_data if f["current_bias"] == "unknown")

    det_bias_match = sum(1 for d in det_data if d["current_bias"] == d["needed_bias"])
    det_bias_mismatch = sum(1 for d in det_data if d["current_bias"] != d["needed_bias"] and d["current_bias"] != "unknown")

    print(f"\n  FN  - Bias CORRECTO:   {fn_bias_match}/{len(fn_data)} ({fn_bias_match/len(fn_data)*100:.0f}%)")
    print(f"  FN  - Bias INCORRECTO: {fn_bias_mismatch}/{len(fn_data)} ({fn_bias_mismatch/len(fn_data)*100:.0f}%)")
    print(f"  FN  - Bias NEUTRAL:    {fn_bias_neutral}/{len(fn_data)} ({fn_bias_neutral/len(fn_data)*100:.0f}%)")
    print(f"  DET - Bias CORRECTO:   {det_bias_match}/{len(det_data)} ({det_bias_match/len(det_data)*100:.0f}%)")
    print(f"  DET - Bias INCORRECTO: {det_bias_mismatch}/{len(det_data)} ({det_bias_mismatch/len(det_data)*100:.0f}%)")

    # ── 2. PROBABILIDADES ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 2: PROBABILIDADES DE REGIMEN (predict_proba)")
    print(f"  {'─'*60}")

    for label, data in [("FN", fn_data), ("DET", det_data)]:
        p_max = [d["prob_max"] for d in data]
        p_bull = [d["prob_bullish"] for d in data if d["needed_bias"] == "bullish"]
        p_bear = [d["prob_bearish"] for d in data if d["needed_bias"] == "bearish"]

        print(f"\n  {label}:")
        print(f"    Prob MAX media:    {np.mean(p_max):.3f} (min={min(p_max):.3f}, max={max(p_max):.3f})")

        # Que tan seguros estamos del bias CORRECTO?
        p_correct_bias = []
        for d in data:
            if d["needed_bias"] == "bullish":
                p_correct_bias.append(d["prob_bullish"])
            elif d["needed_bias"] == "bearish":
                p_correct_bias.append(d["prob_bearish"])
        if p_correct_bias:
            print(f"    Prob del bias CORRECTO: media={np.mean(p_correct_bias):.3f}")

        # Que tan seguros estamos del bias CONTRARIO?
        p_wrong_bias = []
        for d in data:
            if d["needed_bias"] == "bullish":
                p_wrong_bias.append(d["prob_bearish"])
            elif d["needed_bias"] == "bearish":
                p_wrong_bias.append(d["prob_bullish"])
        if p_wrong_bias:
            print(f"    Prob del bias CONTRARIO: media={np.mean(p_wrong_bias):.3f}")

    # ── 3. SCORES ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 3: SCORE DE SENAL ANTES DEL CAMBIO")
    print(f"  {'─'*60}")

    for label, data in [("FN", fn_data), ("DET", det_data)]:
        scores_long = [d["score_long_before"] for d in data if d["after_signal"] == "LONG"]
        scores_short = [d["score_short_before"] for d in data if d["after_signal"] == "SHORT"]
        all_scores = [max(d["score_long_before"], d["score_short_before"]) for d in data]
        print(f"\n  {label} - Score medio: {np.mean(all_scores):.0f} (rango: {min(all_scores):.0f}-{max(all_scores):.0f})")
        if scores_long:
            print(f"    LONG: media={np.mean(scores_long):.0f}, min={min(scores_long):.0f}, max={max(scores_long):.0f}")
        if scores_short:
            print(f"    SHORT: media={np.mean(scores_short):.0f}, min={min(scores_short):.0f}, max={max(scores_short):.0f}")

    # ── 4. FEATURES DEL HMM ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 4: FEATURES DEL HMM (comparacion FN vs DET)")
    print(f"  {'─'*60}")

    feature_cols = [
        ("vol_20", "Volatilidad"),
        ("atr_norm", "ATR Normalizado"),
        ("di_spread", "DI Spread (dir.)"),
        ("score_net", "Score Neto (dir.)"),
        ("ema_spread_raw", "EMA Spread"),
        ("cumret_20", "Cum Return"),
        ("momentum_20", "Momentum"),
        ("rsi_norm", "RSI Norm"),
        ("pos_in_range", "Pos. en Rango"),
        ("adx_scaled", "ADX"),
        ("vol_rel_20", "Vol Relativo"),
    ]

    print(f"\n  {'Feature':<22} {'FN media':>10} {'DET media':>10} {'Dif.':>8} {'Direccion':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for col, label in feature_cols:
        fn_vals = [d[col] for d in fn_data if d[col] is not None]
        det_vals = [d[col] for d in det_data if d[col] is not None]
        if fn_vals and det_vals:
            fn_mean = np.mean(fn_vals)
            det_mean = np.mean(det_vals)
            diff = fn_mean - det_mean
            # Direccion: que favorece los FN?
            direction = ""
            if abs(diff) > 0.01 * abs(fn_mean):
                if diff > 0:
                    direction = "MAS en FN" if fn_mean != 0 else "↑"
                else:
                    direction = "MENOS en FN" if fn_mean != 0 else "↓"
            print(f"  {label:<22} {fn_mean:>10.4f} {det_mean:>10.4f} {diff:>+8.4f} {direction:>10}")

    # ── 5. SEPARABILIDAD ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 5: SEPARABILIDAD DE FEATURES DIRECCIONALES")
    print(f"  {'─'*60}")

    # Para cada FN, ver el score_net y di_spread - deberian ser positivos para LONG, negativos para SHORT
    fn_correct_direction = sum(1 for d in fn_data
                               if (d["after_signal"] == "LONG" and d["score_net"] > 0) or
                                  (d["after_signal"] == "SHORT" and d["score_net"] < 0))
    fn_wrong_direction = sum(1 for d in fn_data
                             if (d["after_signal"] == "LONG" and d["score_net"] < 0) or
                                (d["after_signal"] == "SHORT" and d["score_net"] > 0))
    det_correct_direction = sum(1 for d in det_data
                                if (d["after_signal"] == "LONG" and d["score_net"] > 0) or
                                   (d["after_signal"] == "SHORT" and d["score_net"] < 0))
    det_wrong_direction = sum(1 for d in det_data
                              if (d["after_signal"] == "LONG" and d["score_net"] < 0) or
                                 (d["after_signal"] == "SHORT" and d["score_net"] > 0))

    print(f"\n  Score Net direction:")
    print(f"    FN - direction CORRECTA:  {fn_correct_direction}/{len(fn_data)} ({fn_correct_direction/len(fn_data)*100:.0f}%)")
    print(f"    FN - direction INCORRECTA: {fn_wrong_direction}/{len(fn_data)} ({fn_wrong_direction/len(fn_data)*100:.0f}%)")
    print(f"    DET - direction CORRECTA: {det_correct_direction}/{len(det_data)} ({det_correct_direction/len(det_data)*100:.0f}%)")
    print(f"    DET - direction INCORRECTA: {det_wrong_direction}/{len(det_data)} ({det_wrong_direction/len(det_data)*100:.0f}%)")

    # DI Spread direction
    fn_di_correct = sum(1 for d in fn_data
                        if (d["after_signal"] == "LONG" and d["di_spread"] > 0) or
                           (d["after_signal"] == "SHORT" and d["di_spread"] < 0))
    det_di_correct = sum(1 for d in det_data
                         if (d["after_signal"] == "LONG" and d["di_spread"] > 0) or
                            (d["after_signal"] == "SHORT" and d["di_spread"] < 0))
    print(f"\n  DI Spread direction:")
    print(f"    FN - direction CORRECTA:  {fn_di_correct}/{len(fn_data)} ({fn_di_correct/len(fn_data)*100:.0f}%)")
    print(f"    DET - direction CORRECTA: {det_di_correct}/{len(det_data)} ({det_di_correct/len(det_data)*100:.0f}%)")

    # ── 6. CAMBIOS DE REGIMEN RECIENTES ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 6: CAMBIOS DE REGIMEN RECIENTES (ultimas 10 velas)")
    print(f"  {'─'*60}")

    fn_recent = sum(1 for d in fn_data if d["recent_regime_change"])
    det_recent = sum(1 for d in det_data if d.get("recent_regime_change"))
    print(f"\n  FN  - Con cambio reciente:  {fn_recent}/{len(fn_data)} ({fn_recent/len(fn_data)*100:.0f}%)")
    print(f"  DET - Con cambio reciente:  {det_recent}/{len(det_data)} ({det_recent/len(det_data)*100:.0f}%)")

    # ── 7. EJEMPLOS CONCRETOS ──
    print(f"\n  {'─'*60}")
    print(f"  ANALISIS 7: EJEMPLOS CONCRETOS DE FN")
    print(f"  {'─'*60}")

    fn_sorted = sorted(fn_data, key=lambda f: max(f["score_long_before"], f["score_short_before"]), reverse=True)

    for i, fn in enumerate(fn_sorted[:8]):
        max_score = max(fn["score_long_before"], fn["score_short_before"])
        print(f"\n  FN #{i+1}: {fn['date']}")
        print(f"    Senal: {fn['current_signal']} -> {fn['after_signal']}")
        print(f"    Regimen: estado {fn['current_state']}, bias={fn['current_bias']} (necesitaba: {fn['needed_bias']})")
        print(f"    Score: LONG={fn['score_long_before']:.0f} SHORT={fn['score_short_before']:.0f}")
        print(f"    Probs: MAX={fn['prob_max']:.3f} BULL={fn['prob_bullish']:.3f} BEAR={fn['prob_bearish']:.3f}")
        print(f"    Features: score_net={fn['score_net']:.3f} di_spread={fn['di_spread']:.3f} vol={fn['vol_20']:.4f}")
        if fn["recent_regime_change"]:
            print(f"    Cambio de regimen hace {fn['regime_change_lag']}v: {fn['regime_change_from']} -> {fn['regime_change_to']}")

    # ── 8. CONCLUSIONES ──
    print(f"\n  {'─'*60}")
    print(f"  CONCLUSIONES PARA {tf}")
    print(f"  {'─'*60}")

    pct_bias_mismatch = fn_bias_mismatch / len(fn_data) * 100 if fn_data else 0
    pct_bias_match = fn_bias_match / len(fn_data) * 100 if fn_data else 0
    pct_recent = fn_recent / len(fn_data) * 100 if fn_data else 0
    pct_fn_wrong_dir = fn_wrong_direction / len(fn_data) * 100 if fn_data else 0

    conclusions = []
    recommendations = []

    if pct_bias_mismatch > 30:
        conclusions.append(f"REGIMEN OPUESTO: {pct_bias_mismatch:.0f}% de FN tienen bias incorrecto")
        recommendations.append("Mejorar features direccionales del HMM (score_net, di_spread)")
        recommendations.append("Usar soft probability de cambio en vez de Viterbi duro")

    if pct_bias_match > 50 and pct_recent < 40:
        conclusions.append(f"SIN CAMBIO: {(1-pct_recent/100)*100:.0f}% de FN no tuvieron cambio de regimen reciente")
        recommendations.append("Hacer HMM mas sensible: ventana deslizante mas corta o stride menor")
        recommendations.append("Anadir feature de 'score_delta' como senal de cambio inminente")

    fn_p_correct = np.mean([d["prob_bullish"] if d["needed_bias"] == "bullish" else d["prob_bearish"] for d in fn_data]) if fn_data else 0
    det_p_correct = np.mean([d["prob_bullish"] if d["needed_bias"] == "bullish" else d["prob_bearish"] for d in det_data]) if det_data else 0

    if fn_p_correct < 0.3:
        conclusions.append(f"PROBAS BAJAS: Probabilidad del bias correcto es solo {fn_p_correct:.2f} en FN")
        recommendations.append("Ensemble HMM para estabilizar probabilidades")

    fn_score_net = np.mean([d["score_net"] for d in fn_data]) if fn_data else 0
    det_score_net = np.mean([d["score_net"] for d in det_data]) if det_data else 0

    if abs(fn_score_net) < abs(det_score_net):
        conclusions.append(f"SCORE NET DEBIL: score_net={fn_score_net:.3f} en FN vs {det_score_net:.3f} en detectados")
        recommendations.append("Anadir score_net como feature con mas peso en el HMM o usar threshold adaptativo")

    # Feature mas diferenciadora
    max_diff_feature = ""
    max_diff_val = 0
    for col, label in feature_cols:
        fn_vals = [d[col] for d in fn_data if d[col] is not None]
        det_vals = [d[col] for d in det_data if d[col] is not None]
        if fn_vals and det_vals:
            diff = abs(np.mean(fn_vals) - np.mean(det_vals))
            if diff > max_diff_val:
                max_diff_val = diff
                max_diff_feature = label

    if max_diff_feature:
        conclusions.append(f"Feature MAS DIFERENCIADORA: {max_diff_feature} (dif={max_diff_val:.4f})")

    print(f"\n  Hallazgos:")
    for c in conclusions:
        print(f"  • {c}")
    print(f"\n  Recomendaciones:")
    for i, r in enumerate(recommendations, 1):
        print(f"  {i}. {r}")


def main():
    """Ejecuta analisis para todos los timeframes y recomienda accion."""
    all_results = {}

    for tf in TIMEFRAMES:
        results = analyze_fn(tf)
        if results:
            all_results[tf] = results
            print_analysis(results)

    # ── Resumen global ──
    if not all_results:
        print("\n  No se obtuvieron resultados.")
        return

    print(f"\n{'='*70}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*70}")
    total_signals = sum(r["total_signals"] for r in all_results.values())
    total_fn = sum(r["fn"] for r in all_results.values())
    total_det = sum(r["detected"] for r in all_results.values())
    global_rate = round(total_det / total_signals * 100, 1) if total_signals else 0
    global_fn_rate = round(total_fn / total_signals * 100, 1) if total_signals else 0

    print(f"  Total cambios de senal: {total_signals}")
    print(f"  Total detectados:       {total_det} ({global_rate}%)")
    print(f"  Total FALSOS NEGATIVOS: {total_fn} ({global_fn_rate}%)")
    print(f"  Timeframes analizados:  {list(all_results.keys())}")

    # ── Recomendacion unificada ──
    print(f"\n{'='*70}")
    print(f"  RECOMENDACION DE MEJORA")
    print(f"{'='*70}")

    # Recopilar todos los FN
    all_fn = []
    for r in all_results.values():
        all_fn.extend(r["fn_data"])

    all_det = []
    for r in all_results.values():
        all_det.extend(r["det_data"])

    if not all_fn:
        print(f"\n  No se encontraron FN. El sistema funciona perfectamente!")
        return

    # Metricas globales
    bias_mismatch = sum(1 for f in all_fn if f["current_bias"] != f["needed_bias"] and f["current_bias"] != "unknown")
    bias_match = sum(1 for f in all_fn if f["current_bias"] == f["needed_bias"])
    no_recent_change = sum(1 for f in all_fn if not f["recent_regime_change"])
    fn_p_correct = np.mean([f["prob_bullish"] if f["needed_bias"] == "bullish" else f["prob_bearish"] for f in all_fn])
    det_p_correct = np.mean([d["prob_bullish"] if d["needed_bias"] == "bullish" else d["prob_bearish"] for d in all_det])

    fn_score = np.mean([max(f["score_long_before"], f["score_short_before"]) for f in all_fn])
    det_score = np.mean([max(d["score_long_before"], d["score_short_before"]) for d in all_det])

    fn_score_net = np.mean([f["score_net"] for f in all_fn])
    det_score_net = np.mean([d["score_net"] for d in all_det])

    # Feature diff global
    feature_cols = [
        ("vol_20", "Volatilidad"),
        ("di_spread", "DI Spread"),
        ("score_net", "Score Neto"),
        ("rsi_norm", "RSI Norm"),
        ("pos_in_range", "Pos. Rango"),
        ("cumret_20", "Cum Return"),
    ]
    best_feature = ""
    best_diff = 0
    for col, label in feature_cols:
        f_vals = [f[col] for f in all_fn if f[col] is not None]
        d_vals = [d[col] for d in all_det if d[col] is not None]
        if f_vals and d_vals:
            diff = abs(np.mean(f_vals) - np.mean(d_vals))
            if diff > best_diff:
                best_diff = diff
                best_feature = label

    # Problemas identificados
    issues = []
    if bias_mismatch > len(all_fn) * 0.3:
        issues.append({
            "name": "REGIMEN OPUESTO",
            "pct": bias_mismatch / len(all_fn) * 100,
            "desc": f"El HMM asigna regimen opuesto en {bias_mismatch}/{len(all_fn)} casos",
            "solution": "ANADIR FEATURES DIRECCIONALES al HMM (score_net, di_spread, momentum_20 con mas peso de normalizacion)",
        })
    if no_recent_change > len(all_fn) * 0.5:
        issues.append({
            "name": "SIN CAMBIO DE REGIMEN",
            "pct": no_recent_change / len(all_fn) * 100,
            "desc": f"{(1-no_recent_change/len(all_fn))*100:.0f}% de FN sin cambio de regimen en ultimas 10v",
            "solution": "HMM MAS SENSIBLE: reducir stride (30), reducir window (300), o usar forward filter probabilities",
        })
    if fn_p_correct < 0.4:
        issues.append({
            "name": "PROBAS BAJAS",
            "pct": 100,
            "desc": f"Probabilidad del bias correcto = {fn_p_correct:.2f} en FN (vs {det_p_correct:.2f} en detectados)",
            "solution": "ENSEMBLE HMM (bagging): promediar predict_proba de 5-10 HMMs con diferentes semillas",
        })
    if abs(fn_score_net) < abs(det_score_net) * 0.7:
        issues.append({
            "name": "SCORE NET DEBIL",
            "pct": 100,
            "desc": f"score_net en FN={fn_score_net:.3f} vs detectados={det_score_net:.3f}",
            "solution": "REFORZAR score_net como feature y anadirlo al score compuesto con peso extra en senales debiles",
        })
    if fn_score < 50:
        issues.append({
            "name": "SCORE BAJO",
            "pct": 100,
            "desc": f"Score promedio en FN = {fn_score:.0f}",
            "solution": "REDUCIR SIGNAL_SCORE_THRESHOLD (de 60 a 55) para capturar mas senales",
        })

    # Feature mas diferenciadora
    if best_feature:
        issues.append({
            "name": f"FEATURE CLAVE: {best_feature}",
            "pct": 100,
            "desc": f"La feature con mayor diferencia FN vs DET es {best_feature} (dif={best_diff:.4f})",
            "solution": f"MAXIMIZAR {best_feature.upper()}: normalizarla mejor, anadirle interacciones, o usarla para threshold adaptativo",
        })

    # Mostrar problemas ordenados por impacto
    issues.sort(key=lambda x: x["pct"], reverse=True)

    print(f"\n  Se identificaron {len(issues)} areas de mejora:")
    for i, iss in enumerate(issues, 1):
        print(f"\n  {i}. [{iss['name']}] ({iss['pct']:.0f}% de FN afectados)")
        print(f"     {iss['desc']}")
        print(f"     -> {iss['solution']}")

    # TOP RECOMENDACION
    if issues:
        print(f"\n  {'='*60}")
        print(f"  RECOMENDACION PRINCIPAL:")
        print(f"  {'='*60}")
        print(f"  {issues[0]['solution']}")
        print(f"  {'='*60}")


if __name__ == "__main__":
    main()
