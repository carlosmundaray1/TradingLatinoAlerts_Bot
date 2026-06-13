#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNOSTICO: Analisis de distribucion de confianza del sistema hibrido.
"""
import sys, os, importlib.util
import numpy as np
import pandas as pd

# Redirigir stderr a nul para evitar warnings
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
compute_hybrid_alert = hmm_mod.compute_hybrid_alert
compute_precursor_signals = hmm_mod.compute_precursor_signals
HYBRID_CONFIDENCE_THRESHOLD = hmm_mod.HYBRID_CONFIDENCE_THRESHOLD
HYBRID_LOOKBACK = hmm_mod.HYBRID_LOOKBACK
HYBRID_W_HMM = hmm_mod.HYBRID_W_HMM
HYBRID_W_PRECURSOR = hmm_mod.HYBRID_W_PRECURSOR
HYBRID_W_VELOCITY = hmm_mod.HYBRID_W_VELOCITY
HYBRID_W_ALIGNMENT = hmm_mod.HYBRID_W_ALIGNMENT
HYBRID_VELOCITY_THRESHOLD = hmm_mod.HYBRID_VELOCITY_THRESHOLD

results = {}

for tf in ["1d", "1wk"]:
    print("=" * 70)
    print("  TIMEFRAME: %s" % tf)
    print("=" * 70)
    
    df = load_data(ASSET, tf)
    if df is None or len(df) < 100:
        print("  ERROR: datos insuficientes")
        continue
    
    print("  %d velas" % len(df))
    df = compute_all_indicators(df)
    
    features = build_hmm_features(df)
    model, states, state_summary, _, _ = fit_hmm(features)
    if model is None or len(states) == 0:
        continue
    
    df = df.iloc[:len(states)].copy()
    df["regime"] = states
    df = compute_precursor_signals(df)
    df = compute_hybrid_alert(df, states, state_summary)
    
    # Signal changes
    signal_changes = []
    prev_signal = "FLAT"
    for i in range(len(df)):
        is_long = bool(df["signal_long"].iloc[i])
        is_short = bool(df["signal_short"].iloc[i])
        curr = "LONG" if is_long else ("SHORT" if is_short else "FLAT")
        if curr != prev_signal:
            signal_changes.append({"idx": i, "from": prev_signal, "to": curr})
            prev_signal = curr
    
    real_changes = [s for s in signal_changes if s["from"] != s["to"]]
    n_signals = len(real_changes)
    print("\n  Cambios de senal REALES: %d" % n_signals)
    
    # Collect confidence values for each signal change
    conf_values = []
    conf_details = []  # (conf, has_hmm, has_prec, has_vel, has_align)
    
    for sc in real_changes:
        idx_start = max(0, sc["idx"] - HYBRID_LOOKBACK)
        idx_end = sc["idx"]
        window = df.iloc[idx_start:idx_end + 1]
        
        to_sig = sc["to"]
        if to_sig == "LONG":
            conf = float(window["hybrid_confidence_long"].max())
        else:
            conf = float(window["hybrid_confidence_short"].max())
        
        debug_text = " ".join(window["hybrid_debug"].astype(str))
        
        has_hmm = "HMM_ALCISTA" in debug_text or "HMM_BAJISTA" in debug_text or "HMM_NEUTRAL" in debug_text
        has_prec = "PREC_LONG" in debug_text or "PREC_SHORT" in debug_text
        has_vel = "VEL_LONG" in debug_text or "VEL_SHORT" in debug_text
        has_align = "ALIGN_LONG" in debug_text or "ALIGN_SHORT" in debug_text
        
        conf_values.append(conf)
        conf_details.append((conf, has_hmm, has_prec, has_vel, has_align))
    
    # False positives: alert starts without signal change nearby
    alert_starts = []
    prev_active = False
    for i in range(len(df)):
        active = bool(df["hybrid_alert_active"].iloc[i])
        if active and not prev_active:
            alert_starts.append(i)
        prev_active = active
    
    fp_confs = []
    for start in alert_starts:
        has_signal_change = False
        for sc in real_changes:
            if abs(sc["idx"] - start) <= HYBRID_LOOKBACK * 2:
                has_signal_change = True
                break
        if not has_signal_change:
            c = float(df["hybrid_confidence_long"].iloc[start]) if df["hybrid_confidence_long"].iloc[start] >= df["hybrid_confidence_short"].iloc[start] else float(df["hybrid_confidence_short"].iloc[start])
            fp_confs.append(c)
    
    print("\n  ---- DISTRIBUCION DE CONFIANZA ----")
    print("\n  %10s | %10s | %8s | %6s | %10s" % ("Threshold", "Detectados", "Tasa", "FP", "Balance"))
    print("  " + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*8 + "-+-" + "-"*6 + "-+-" + "-"*10)
    
    best_th = HYBRID_CONFIDENCE_THRESHOLD
    best_bal = -999
    
    for th in range(10, 95, 5):
        detected = sum(1 for c in conf_values if c >= th)
        rate = detected / n_signals * 100 if n_signals else 0
        fp_at_th = sum(1 for c in fp_confs if c >= th) if fp_confs else 0
        fp_rate = fp_at_th / max(len(alert_starts), 1) * 100 if fp_confs else 0
        balance = rate - fp_rate * 0.5
        
        print("  %10d | %3d/%-4d | %6.1f%% | %4d | %+8.1f" % (th, detected, n_signals, rate, fp_at_th, balance))
        
        if balance > best_bal:
            best_bal = balance
            best_th = th
    
    print("\n  >> OPTIMO: threshold=%d (balance=%.1f)" % (best_th, best_bal))
    
    # Component contribution analysis
    print("\n  ---- CONTRIBUCION COMPONENTES (sobre %d cambios reales) ----" % min(200, n_signals))
    hmm_c = sum(1 for d in conf_details[:200] if d[1])
    prec_c = sum(1 for d in conf_details[:200] if d[2])
    vel_c = sum(1 for d in conf_details[:200] if d[3])
    align_c = sum(1 for d in conf_details[:200] if d[4])
    n_analyzed = min(200, n_signals)
    
    print("  HMM cambio regimen:   %d/%d (%.1f%%)" % (hmm_c, n_analyzed, hmm_c/n_analyzed*100 if n_analyzed else 0))
    print("  Precursor activo:     %d/%d (%.1f%%)" % (prec_c, n_analyzed, prec_c/n_analyzed*100 if n_analyzed else 0))
    print("  Velocity > threshold: %d/%d (%.1f%%)" % (vel_c, n_analyzed, vel_c/n_analyzed*100 if n_analyzed else 0))
    print("  Alignment correcto:   %d/%d (%.1f%%)" % (align_c, n_analyzed, align_c/n_analyzed*100 if n_analyzed else 0))
    
    # HMM without precursor
    hmm_no_prec = sum(1 for d in conf_details if d[1] and not d[2])
    print("\n  HMM detecta pero SIN precursor: %d/%d" % (hmm_no_prec, n_signals))
    
    # Percentiles
    if conf_values:
        print("\n  ---- PERCENTILES DE CONFIANZA EN CAMBIOS REALES ----")
        print("  Min:   %.1f" % np.min(conf_values))
        print("  P25:   %.1f" % np.percentile(conf_values, 25))
        print("  P50:   %.1f" % np.percentile(conf_values, 50))
        print("  Media: %.1f" % np.mean(conf_values))
        print("  P75:   %.1f" % np.percentile(conf_values, 75))
        print("  Max:   %.1f" % np.max(conf_values))
        
        # What makes confidence exceed threshold currently?
        print("\n  ---- SENALES CON CONFIANZA >= %d (ACTUAL) ----" % HYBRID_CONFIDENCE_THRESHOLD)
        detected_details = [d for d in conf_details if d[0] >= HYBRID_CONFIDENCE_THRESHOLD]
        if detected_details:
            print("  %d senales detectadas" % len(detected_details))
            # Component breakdown for detected signals
            det_hmm = sum(1 for d in detected_details if d[1])
            det_prec = sum(1 for d in detected_details if d[2])
            det_vel = sum(1 for d in detected_details if d[3])
            det_align = sum(1 for d in detected_details if d[4])
            print("    Con HMM:     %d/%d" % (det_hmm, len(detected_details)))
            print("    Con Precursor: %d/%d" % (det_prec, len(detected_details)))
            print("    Con Velocity:  %d/%d" % (det_vel, len(detected_details)))
            print("    Con Alignment: %d/%d" % (det_align, len(detected_details)))
            
            # How many have HMM+Precursor vs HMM alone?
            both = sum(1 for d in detected_details if d[1] and d[2])
            hmm_only = sum(1 for d in detected_details if d[1] and not d[2])
            prec_only = sum(1 for d in detected_details if not d[1] and d[2])
            print("    HMM + Precursor: %d" % both)
            print("    HMM alone:       %d" % hmm_only)
            print("    Precursor alone:  %d" % prec_only)
        else:
            print("  (ninguna)")
    
    # RECOMMENDATIONS
    print("\n  ---- RECOMENDACIONES ----")
    
    # Calculate what threshold would give ~70% detection
    for target in [50, 60, 70, 80]:
        th_for_target = None
        for th in range(5, 95, 1):
            detected = sum(1 for c in conf_values if c >= th)
            rate = detected / n_signals * 100 if n_signals else 0
            if rate >= target:
                th_for_target = th
                break
        if th_for_target is not None:
            fp_at_th = sum(1 for c in fp_confs if c >= th_for_target) if fp_confs else 0
            print("  Para %.0f%% deteccion -> threshold=%d (FP estimados: %d)" % (target, th_for_target, fp_at_th))
        else:
            print("  Para %.0f%% deteccion -> INALCANZABLE con pesos actuales" % target)
    
    # Calculate what HMM weight alone would need to be
    print("\n  Peso HMM actual: %d" % HYBRID_W_HMM)
    print("  HMM + Alignment = %d (necesita >= %d para activar)" % (HYBRID_W_HMM + HYBRID_W_ALIGNMENT, HYBRID_CONFIDENCE_THRESHOLD))
    print("  HMM debe aumentar a %d para activar solo + alignment" % (HYBRID_CONFIDENCE_THRESHOLD - HYBRID_W_ALIGNMENT))
    print("  O threshold debe bajar a %d para que HMM+Alignment active" % (HYBRID_W_HMM + HYBRID_W_ALIGNMENT))
    
    # Store results
    results[tf] = {
        "n_signals": n_signals,
        "best_threshold": best_th,
        "conf_values": conf_values,
        "fp_confs": fp_confs[:50] if fp_confs else [],
        "hmm_contrib": hmm_c,
        "prec_contrib": prec_c,
        "vel_contrib": vel_c,
        "align_contrib": align_c,
        "hmm_no_precursor": hmm_no_prec,
        "p50": np.percentile(conf_values, 50) if conf_values else 0,
        "mean_conf": np.mean(conf_values) if conf_values else 0,
    }
    print()

print("=" * 70)
print("  DIAGNOSTICO COMPLETADO")
print("=" * 70)
