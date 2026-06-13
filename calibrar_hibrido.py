#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALIBRACION AUTOMATICA: Grid search de parametros del sistema hibrido.
Prueba miles de combinaciones de threshold y pesos para encontrar la
configuracion que maximiza la deteccion de cambios de tendencia.

Uso:
    python calibrar_hibrido.py

Genera:
    calibracion_resultados.txt  → Tabla comparativa con todas las combinaciones
"""
import sys, os, importlib.util, json
import numpy as np
import pandas as pd
from datetime import datetime

sys.stderr = open(os.devnull, 'w')

spec = importlib.util.spec_from_file_location("hmm", "tradinglatino_hmm_clean.py")
hmm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hmm_mod)

ASSET = "BTC-USD"
hmm_mod.PERIOD_1D = "5y"
hmm_mod.PERIOD_1W = "10y"

load_data = hmm_mod.load_data
compute_all_indicators = hmm_mod.compute_all_indicators
build_hmm_features = hmm_mod.build_hmm_features
fit_hmm = hmm_mod.fit_hmm
compute_precursor_signals = hmm_mod.compute_precursor_signals
HYBRID_LOOKBACK = hmm_mod.HYBRID_LOOKBACK

# ── PARAMETROS A CALIBRAR ──
THRESHOLDS = [25, 30, 35, 40, 45, 50, 55]
W_HMM_VALUES = [25, 30, 35, 40, 45, 50]
W_PRECURSOR_VALUES = [25, 30, 35, 40]
W_VELOCITY_VALUES = [15, 20, 25, 30]
W_ALIGNMENT_VALUES = [10, 15, 20]

# ── TIMEFRAMES ──
TIMEFRAMES = ["1d", "1wk"]

def compute_hybrid_alert_custom(df, states, state_summary, params):
    """
    Version parametrizada de compute_hybrid_alert.
    Usa los mismos pesos y threshold configurados.
    """
    n = len(df)
    th = params["threshold"]
    w_hmm = params["w_hmm"]
    w_precursor = params["w_precursor"]
    w_velocity = params["w_velocity"]
    w_alignment = params["w_alignment"]
    
    df["hybrid_confidence_long"] = 0.0
    df["hybrid_confidence_short"] = 0.0
    df["hybrid_alert_long"] = False
    df["hybrid_alert_short"] = False
    df["hybrid_alert_active"] = False
    
    state_bias_map = {}
    for _, r in state_summary.iterrows():
        state_bias_map[int(r["state"])] = hmm_mod._classify_regime_bias(r["description"])
    
    m = min(len(states), n)
    change_bias = pd.Series(np.nan, index=range(m))
    if m > 1:
        changes_mask = np.zeros(m, dtype=bool)
        changes_mask[1:] = (states[1:] != states[:-1])
        for i in np.where(changes_mask)[0]:
            change_bias.iloc[i] = state_bias_map.get(int(states[i]), "neutral")
    
    regime_new_bias_s = change_bias.ffill(limit=HYBRID_LOOKBACK).fillna("neutral")
    regime_changed_s = change_bias.ffill(limit=HYBRID_LOOKBACK).notna()
    
    full_regime_new_bias = np.full(n, "neutral", dtype=object)
    full_regime_new_bias[:m] = regime_new_bias_s.values
    full_regime_changed = np.zeros(n, dtype=bool)
    full_regime_changed[:m] = regime_changed_s.values
    
    # HMM
    hmm_score = np.where(
        full_regime_changed,
        np.where(full_regime_new_bias == "neutral", w_hmm // 2, w_hmm),
        0,
    )
    hmm_bullish_or_neutral = (full_regime_new_bias == "bullish") | (full_regime_new_bias == "neutral")
    hmm_bearish_or_neutral = (full_regime_new_bias == "bearish") | (full_regime_new_bias == "neutral")
    hmm_contrib_long = np.where(hmm_bullish_or_neutral, hmm_score, 0)
    hmm_contrib_short = np.where(hmm_bearish_or_neutral, hmm_score, 0)
    
    # Precursor
    if "precursor_long" in df.columns:
        precursor_long_window = df["precursor_long"].rolling(HYBRID_LOOKBACK + 1, min_periods=1).max().fillna(False).astype(bool)
        precursor_short_window = df["precursor_short"].rolling(HYBRID_LOOKBACK + 1, min_periods=1).max().fillna(False).astype(bool)
    else:
        precursor_long_window = pd.Series(False, index=df.index)
        precursor_short_window = pd.Series(False, index=df.index)
    precursor_score_long = precursor_long_window.astype(float) * w_precursor
    precursor_score_short = precursor_short_window.astype(float) * w_precursor
    
    # Velocity
    if "signal_score_long" in df.columns:
        velocity_long = df["signal_score_long"].diff().fillna(0)
        velocity_short = df["signal_score_short"].diff().fillna(0)
    else:
        velocity_long = pd.Series(0.0, index=df.index)
        velocity_short = pd.Series(0.0, index=df.index)
    
    HYBRID_VELOCITY_THRESHOLD = 5.0
    velocity_score_long = np.where(
        velocity_long > HYBRID_VELOCITY_THRESHOLD,
        np.clip((w_velocity * velocity_long / 10.0).astype(int), 0, w_velocity), 0)
    velocity_score_short = np.where(
        velocity_short > HYBRID_VELOCITY_THRESHOLD,
        np.clip((w_velocity * velocity_short / 10.0).astype(int), 0, w_velocity), 0)
    
    # Alignment
    if "regime" in df.columns:
        regime_series = df["regime"].astype(int)
        regime_bias = regime_series.map(lambda s: state_bias_map.get(s, "neutral"))
        alignment_score_long = (regime_bias == "bullish").astype(float) * w_alignment
        alignment_score_short = (regime_bias == "bearish").astype(float) * w_alignment
    else:
        alignment_score_long = pd.Series(0.0, index=df.index)
        alignment_score_short = pd.Series(0.0, index=df.index)
    
    # Total
    conf_long = np.clip(hmm_contrib_long + precursor_score_long.values + velocity_score_long + alignment_score_long.values, 0, 100)
    conf_short = np.clip(hmm_contrib_short + precursor_score_short.values + velocity_score_short + alignment_score_short.values, 0, 100)
    
    df["hybrid_confidence_long"] = conf_long
    df["hybrid_confidence_short"] = conf_short
    df["hybrid_alert_long"] = conf_long >= th
    df["hybrid_alert_short"] = conf_short >= th
    df["hybrid_alert_active"] = df["hybrid_alert_long"] | df["hybrid_alert_short"]
    
    return df


def evaluate_params(df, states, state_summary, params):
    """Evalua una combinacion de parametros y retorna metricas."""
    df2 = df.copy()
    df2 = compute_hybrid_alert_custom(df2, states, state_summary, params)
    
    # Signal changes
    signal_changes = []
    prev_signal = "FLAT"
    for i in range(len(df2)):
        is_long = bool(df2["signal_long"].iloc[i])
        is_short = bool(df2["signal_short"].iloc[i])
        curr = "LONG" if is_long else ("SHORT" if is_short else "FLAT")
        if curr != prev_signal:
            signal_changes.append({"idx": i, "from": prev_signal, "to": curr})
            prev_signal = curr
    
    real_changes = [s for s in signal_changes if s["from"] != s["to"]]
    n_signals = len(real_changes)
    if n_signals == 0:
        return {"detection_rate": 0, "detected": 0, "total": 0, "fp": 0, "fn": 0, "direction_ok": 0}
    
    # Detectados: híbrido alerta en ventana antes del cambio
    MAX_LAG = 5
    detected = 0
    direction_ok = 0
    
    for sc in real_changes:
        idx_start = max(0, sc["idx"] - MAX_LAG)
        idx_end = sc["idx"]
        window = df2.iloc[idx_start:idx_end + 1]
        
        hybrid_active = window["hybrid_alert_active"].any()
        if hybrid_active:
            detected += 1
            to_sig = sc["to"]
            hybrid_long = window["hybrid_alert_long"].any()
            hybrid_short = window["hybrid_alert_short"].any()
            if (to_sig == "LONG" and hybrid_long) or (to_sig == "SHORT" and hybrid_short):
                direction_ok += 1
    
    false_neg = n_signals - detected
    
    # Falsos positivos: alertas activas sin cambio de señal cercano
    alert_starts = []
    prev_active = False
    for i in range(len(df2)):
        active = bool(df2["hybrid_alert_active"].iloc[i])
        if active and not prev_active:
            alert_starts.append(i)
        prev_active = active
    
    false_pos = 0
    for start in alert_starts:
        has_signal = False
        for sc in real_changes:
            if abs(sc["idx"] - start) <= MAX_LAG * 2:
                has_signal = True
                break
        if not has_signal:
            false_pos += 1
    
    return {
        "detection_rate": round(detected / n_signals * 100, 1),
        "detected": detected,
        "total": n_signals,
        "fp": false_pos,
        "fn": false_neg,
        "direction_ok": direction_ok,
        "alert_starts": len(alert_starts),
    }


# ── MAIN ──
print("=" * 80)
print("  CALIBRACION AUTOMATICA DEL SISTEMA HIBRIDO")
print("=" * 80)
print()
print("  Barriendo %d x %d x %d x %d x %d = %d combinaciones" % (
    len(THRESHOLDS), len(W_HMM_VALUES), len(W_PRECURSOR_VALUES),
    len(W_VELOCITY_VALUES), len(W_ALIGNMENT_VALUES),
    len(THRESHOLDS) * len(W_HMM_VALUES) * len(W_PRECURSOR_VALUES) *
    len(W_VELOCITY_VALUES) * len(W_ALIGNMENT_VALUES)
))
print()

all_results = {}

for tf in TIMEFRAMES:
    print("  [%s] Cargando datos..." % tf)
    df = load_data(ASSET, tf)
    if df is None or len(df) < 100:
        print("  ERROR: datos insuficientes")
        continue
    
    df = compute_all_indicators(df)
    features = build_hmm_features(df)
    model, states, state_summary, _, _ = fit_hmm(features)
    if model is None or len(states) == 0:
        continue
    
    df = df.iloc[:len(states)].copy()
    df["regime"] = states
    df = compute_precursor_signals(df)
    
    print("  [%s] Evaluando %d combinaciones..." % (tf, 
        len(THRESHOLDS) * len(W_HMM_VALUES) * len(W_PRECURSOR_VALUES) *
        len(W_VELOCITY_VALUES) * len(W_ALIGNMENT_VALUES)))
    
    results = []
    total_combos = (len(THRESHOLDS) * len(W_HMM_VALUES) * len(W_PRECURSOR_VALUES) *
                    len(W_VELOCITY_VALUES) * len(W_ALIGNMENT_VALUES))
    combo_count = 0
    
    for th in THRESHOLDS:
        for wh in W_HMM_VALUES:
            for wp in W_PRECURSOR_VALUES:
                for wv in W_VELOCITY_VALUES:
                    for wa in W_ALIGNMENT_VALUES:
                        params = {
                            "threshold": th, "w_hmm": wh,
                            "w_precursor": wp, "w_velocity": wv, "w_alignment": wa
                        }
                        metrics = evaluate_params(df, states, state_summary, params)
                        metrics["params"] = params
                        
                        # Score compuesto: detection rate - FP penalty
                        fp_penalty = metrics["fp"] / max(metrics["detected"], 1) * 10
                        metrics["score"] = round(metrics["detection_rate"] - fp_penalty, 1)
                        
                        results.append(metrics)
                        combo_count += 1
    
    results.sort(key=lambda r: r["score"], reverse=True)
    
    all_results[tf] = results[:50]  # Keep top 50
    
    print("  [%s] Mejores 10 combinaciones:" % tf)
    print()
    print("  %4s | %4s | %4s | %4s | %4s | %8s | %5s | %5s | %4s | %6s" % (
        "TH", "HMM", "PRE", "VEL", "ALI", "DETEC%", "DETECT", "FP", "FN", "SCORE"))
    print("  " + "-"*4 + "-+-" + "-"*4 + "-+-" + "-"*4 + "-+-" + "-"*4 + "-+-" + "-"*4 + "-+-" + "-"*8 + "-+-" + "-"*5 + "-+-" + "-"*5 + "-+-" + "-"*4 + "-+-" + "-"*6)
    
    for r in results[:10]:
        p = r["params"]
        print("  %4d | %4d | %4d | %4d | %4d | %7.1f%% | %3d/%-d | %4d | %4d | %+5.1f" % (
            p["threshold"], p["w_hmm"], p["w_precursor"], p["w_velocity"], p["w_alignment"],
            r["detection_rate"], r["detected"], r["total"], r["fp"], r["fn"], r["score"]))

print()
print("=" * 80)
print("  MEJORES COMBINACIONES GLOBALES")
print("=" * 80)
print()

# Show top config for each threshold level
for tf in TIMEFRAMES:
    print("  --- %s ---" % tf)
    best = all_results[tf][0]
    p = best["params"]
    print("  #1: th=%d HMM=%d PRE=%d VEL=%d ALI=%d -> DETECCION: %.1f%% (%d/%d) | FP:%d FN:%d SCORE:%.1f" % (
        p["threshold"], p["w_hmm"], p["w_precursor"], p["w_velocity"], p["w_alignment"],
        best["detection_rate"], best["detected"], best["total"], best["fp"], best["fn"], best["score"]))
    
    # Best at each threshold
    print()
    print("  Mejor por threshold:")
    for th in THRESHOLDS:
        best_at_th = [r for r in all_results[tf] if r["params"]["threshold"] == th]
        if best_at_th:
            b = best_at_th[0]
            p = b["params"]
            print("    th=%2d: HMM=%d PRE=%d VEL=%d ALI=%d -> %.1f%% (%d/%d) FP:%d SCORE:%.1f" % (
                th, p["w_hmm"], p["w_precursor"], p["w_velocity"], p["w_alignment"],
                b["detection_rate"], b["detected"], b["total"], b["fp"], b["score"]))
    print()

# SUMMARY
print("=" * 80)
print("  RECOMENDACIONES FINALES")
print("=" * 80)
print()

for tf in TIMEFRAMES:
    best = all_results[tf][0]
    p = best["params"]
    print("  %s: threshold=%d, HMM=%d, Precursor=%d, Velocity=%d, Alignment=%d" % (
        tf, p["threshold"], p["w_hmm"], p["w_precursor"], p["w_velocity"], p["w_alignment"]))
    print("     Detection: %.1f%% (%d/%d), FP: %d, FN: %d" % (
        best["detection_rate"], best["detected"], best["total"], best["fp"], best["fn"]))

# Find best config that works well for BOTH timeframes
print()
print("  Mejor configuracion COMUN (promedio de ambos timeframes):")
common_best = None
common_best_score = -999

# Combine results from both timeframes
for th in THRESHOLDS:
    for wh in W_HMM_VALUES:
        for wp in W_PRECURSOR_VALUES:
            for wv in W_VELOCITY_VALUES:
                for wa in W_ALIGNMENT_VALUES:
                    params_key = (th, wh, wp, wv, wa)
                    score_sum = 0
                    valid = True
                    for tf in TIMEFRAMES:
                        matches = [r for r in all_results[tf] if 
                                   r["params"]["threshold"] == th and
                                   r["params"]["w_hmm"] == wh and
                                   r["params"]["w_precursor"] == wp and
                                   r["params"]["w_velocity"] == wv and
                                   r["params"]["w_alignment"] == wa]
                        if matches:
                            score_sum += matches[0]["score"]
                        else:
                            valid = False
                            break
                    if valid and score_sum > common_best_score:
                        common_best_score = score_sum
                        common_best = (th, wh, wp, wv, wa)

if common_best:
    th, wh, wp, wv, wa = common_best
    print("  threshold=%d, HMM=%d, Precursor=%d, Velocity=%d, Alignment=%d" % (th, wh, wp, wv, wa))
    print("  Score combinado: %.1f" % common_best_score)
    for tf in TIMEFRAMES:
        matches = [r for r in all_results[tf] if 
                   r["params"]["threshold"] == th and
                   r["params"]["w_hmm"] == wh and
                   r["params"]["w_precursor"] == wp and
                   r["params"]["w_velocity"] == wv and
                   r["params"]["w_alignment"] == wa]
        if matches:
            m = matches[0]
            print("    %s: %.1f%% (%d/%d) FP:%d FN:%d" % (
                tf, m["detection_rate"], m["detected"], m["total"], m["fp"], m["fn"]))

print()
print("  Fecha: %s" % datetime.now().strftime("%d-%m-%Y %H:%M"))
print("=" * 80)
