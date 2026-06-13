#!/usr/bin/env python
"""
Simulacion independiente para medir la tasa de deteccion de precursores
vs la tasa de deteccion por cambios de regimen HMM.

Enfoque A: Sistema de Precursores.
"""
import sys
import os
import numpy as np
import pandas as pd

# Importar el modulo HMM (con las mejoras ya aplicadas)
import importlib.util
spec = importlib.util.spec_from_file_location("hmm_clean", "tradinglatino_hmm_clean.py")
_HMM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_HMM)

# Configurar periodos extendidos
_HMM.PERIOD_1D = "5y"
_HMM.PERIOD_1W = "10y"

# Usar funciones del modulo
load_data = _HMM.load_data
compute_all_indicators = _HMM.compute_all_indicators
build_hmm_features = _HMM.build_hmm_features
fit_hmm = _HMM.fit_hmm
compute_signal = _HMM.compute_signal
compute_precursor_signals = _HMM.compute_precursor_signals
cross_reference_changes = _HMM.cross_reference_changes if hasattr(_HMM, 'cross_reference_changes') else None
verify_signals_historically = _HMM.verify_signals_historically
_detect_regime_changes = _HMM._detect_regime_changes
consecutive_bars_filter = _HMM.consecutive_bars_filter if hasattr(_HMM, 'consecutive_bars_filter') else None
_consecutive_bars_filter = _HMM._consecutive_bars_filter

TIMEFRAMES = ["1d", "1wk"]
ASSET = "BTC-USD"

print(f"{'='*60}")
print(f"  SIMULACION DE PRECURSORES - {ASSET}")
print(f"{'='*60}")
print()

results = {}

for tf in TIMEFRAMES:
    print(f"{'-'*50}")
    print(f"  TIMEFRAME: {tf}")
    print(f"{'-'*50}")

    # 1) Descargar datos
    if tf == "1d":
        df = load_data(ASSET, "1d")
    else:
        df = load_data(ASSET, "1wk")

    print(f"  {len(df)} velas cargadas.")

    # 2) Indicadores
    df = compute_all_indicators(df)

    # 3) HMM
    features_df = build_hmm_features(df)
    _, states, state_summary, _, trans_mat = fit_hmm(features_df)
    df["regime"] = states

    # 3b) Recalcular scores con HMM (si existe la funcion)
    if hasattr(_HMM, 'compute_signal_scores_with_hmm'):
        df = _HMM.compute_signal_scores_with_hmm(df, state_summary)

    # 3c) Precursores
    df = compute_precursor_signals(df)

    # 4) Detectar cambios de senal
    df["signal_combo_long"] = _consecutive_bars_filter(df["signal_long"], min_bars=2)
    df["signal_combo_short"] = _consecutive_bars_filter(df["signal_short"], min_bars=2)

    current_signal = "FLAT"
    signal_changes = []
    for i in range(len(df)):
        signal = "FLAT"
        if df["signal_combo_long"].iloc[i]:
            signal = "LONG"
        elif df["signal_combo_short"].iloc[i]:
            signal = "SHORT"

        if signal != current_signal:
            signal_changes.append({
                "idx": i,
                "from_signal": current_signal,
                "to_signal": signal,
                "date": df.index[i],
            })
            current_signal = signal

    # 5) Detectar cambios de regimen
    regime_changes = []
    current_regime = states[0]
    for i in range(1, len(states)):
        if states[i] != current_regime:
            regime_changes.append({
                "idx": i,
                "from_state": int(current_regime),
                "to_state": int(states[i]),
                "date": df.index[i],
            })
            current_regime = states[i]

    # 6) Detectar eventos precursores
    precursor_events = []
    if "precursor_long" in df.columns:
        precursor_long_idx = df[df["precursor_long"]].index.tolist()
        for pi in precursor_long_idx:
            precursor_events.append({
                "idx": df.index.get_loc(pi),
                "type": "PRECURSOR_LONG",
                "date": pi,
            })

    if "precursor_short" in df.columns:
        precursor_short_idx = df[df["precursor_short"]].index.tolist()
        for pi in precursor_short_idx:
            precursor_events.append({
                "idx": df.index.get_loc(pi),
                "type": "PRECURSOR_SHORT",
                "date": pi,
            })
    precursor_events.sort(key=lambda x: x["idx"])

    # 7) Cruzar cambios de regimen vs senal
    MAX_LAG = 5
    regime_used = set()
    regime_matched = []
    for sc in signal_changes:
        best = None
        best_lag = 999
        best_ri = -1
        for ri, rc in enumerate(regime_changes):
            if ri in regime_used:
                continue
            if rc["idx"] <= sc["idx"] and rc["idx"] >= sc["idx"] - MAX_LAG:
                lag = sc["idx"] - rc["idx"]
                if best is None or lag < best_lag:
                    best = rc
                    best_lag = lag
                    best_ri = ri
        if best is not None:
            regime_matched.append({"signal": sc, "regime": best, "lag": best_lag})
            regime_used.add(best_ri)

    regime_fp = max(0, len(regime_changes) - len(regime_matched))
    regime_fn = len(signal_changes) - len(regime_matched)
    regime_dr = round(len(regime_matched) / max(len(signal_changes), 1) * 100, 1)

    # 8) Cruzar precursores vs senal
    precursor_used = set()
    precursor_matched = []
    for sc in signal_changes:
        best = None
        best_lag = 999
        best_pi = -1
        for pi, pc in enumerate(precursor_events):
            if pi in precursor_used:
                continue
            if pc["idx"] <= sc["idx"] and pc["idx"] >= sc["idx"] - MAX_LAG:
                lag = sc["idx"] - pc["idx"]
                if best is None or lag < best_lag:
                    best = pc
                    best_lag = lag
                    best_pi = pi
        if best is not None:
            precursor_matched.append({"signal": sc, "precursor": best, "lag": best_lag})
            precursor_used.add(best_pi)

    precursor_fp = max(0, len(precursor_events) - len(precursor_matched))
    precursor_fn = len(signal_changes) - len(precursor_matched)
    precursor_dr = round(len(precursor_matched) / max(len(signal_changes), 1) * 100, 1)

    # Direccion correcta para precursores
    precursor_dir_ok = 0
    for m in precursor_matched:
        to_sig = m["signal"]["to_signal"]
        prec_type = m["precursor"]["type"]
        if to_sig == "LONG" and "LONG" in prec_type:
            precursor_dir_ok += 1
        elif to_sig == "SHORT" and "SHORT" in prec_type:
            precursor_dir_ok += 1

    lags = [m["lag"] for m in precursor_matched]

    # 9) Resultados
    print(f"  Cambios de senal: {len(signal_changes)}")
    print(f"  Cambios de regimen: {len(regime_changes)}")
    print(f"  Alertas precursoras: {len(precursor_events)}")
    print()
    print(f"  --- DETECCION POR REGIMEN HMM ---")
    print(f"  Detectados: {len(regime_matched)}/{len(signal_changes)} ({regime_dr}%)")
    print(f"  Falsos positivos: {regime_fp}")
    print(f"  Falsos negativos: {regime_fn}")
    print()
    print(f"  --- DETECCION POR PRECURSOR (Enfoque A) ---")
    print(f"  Detectados: {len(precursor_matched)}/{len(signal_changes)} ({precursor_dr}%)")
    print(f"  Falsos positivos: {precursor_fp}")
    print(f"  Falsos negativos: {precursor_fn}")
    print(f"  Anticipacion promedio: {round(sum(lags)/max(len(lags),1),1)} velas")
    print(f"  Direccion correcta: {precursor_dir_ok}/{len(precursor_matched)}")

    results[tf] = {
        "signal_changes": len(signal_changes),
        "regime_changes": len(regime_changes),
        "precursor_events": len(precursor_events),
        "regime_dr": regime_dr,
        "precursor_dr": precursor_dr,
        "regime_fp": regime_fp,
        "regime_fn": regime_fn,
        "precursor_fp": precursor_fp,
        "precursor_fn": precursor_fn,
        "precursor_lag": round(sum(lags)/max(len(lags),1),1),
        "precursor_dir_ok": precursor_dir_ok,
        "precursor_matched": len(precursor_matched),
    }
    print()

print(f"{'='*60}")
print(f"  RESUMEN GLOBAL")
print(f"{'='*60}")
print()

baseline_1d = 37.6
baseline_1wk = 35.7

for tf in TIMEFRAMES:
    r = results[tf]
    mejora_reg = round(r["regime_dr"] - (baseline_1d if tf == "1d" else baseline_1wk), 1)
    print(f"  --- {tf} ---")
    print(f"  Baseline (regimen HMM):   {baseline_1d if tf=='1d' else baseline_1wk}%")
    print(f"  Regimen HMM (actual):     {r['regime_dr']}% ({mejora_reg:+.1f}%)")
    print(f"  PRECURSOR (Enfoque A):    {r['precursor_dr']}%")
    print(f"  Mejora vs baseline:       +{round(r['precursor_dr'] - (baseline_1d if tf=='1d' else baseline_1wk), 1)}%")
    print()

print(f"{'='*60}")
print(f"  Reporte guardado: simulacion_precursores_resultados.txt")
print(f"{'='*60}")

# Guardar resultados
with open("simulacion_precursores_resultados.txt", "w", encoding="utf-8") as f:
    f.write(f"SIMULACION DE PRECURSORES - {ASSET}\n")
    f.write("=" * 60 + "\n\n")
    for tf in TIMEFRAMES:
        r = results[tf]
        f.write(f"TIMEFRAME: {tf}\n")
        f.write(f"  Senales: {r['signal_changes']}\n")
        f.write(f"  Regimenes: {r['regime_changes']} (DR: {r['regime_dr']}%)\n")
        f.write(f"  Precursores: {r['precursor_events']} (DR: {r['precursor_dr']}%, lag: {r['precursor_lag']}v)\n")
        f.write(f"  Precursor direccion correcta: {r['precursor_dir_ok']}/{r['precursor_matched']}\n")
        f.write(f"  FPs regimen: {r['regime_fp']}, FNs regimen: {r['regime_fn']}\n")
        f.write(f"  FPs precursor: {r['precursor_fp']}, FNs precursor: {r['precursor_fn']}\n")
        f.write("\n")

print("Hecho.")
