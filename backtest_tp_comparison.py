#!/usr/bin/env python3
"""
Backtest Comparativo: TP FIJO vs TP DINAMICO (ATR)
=================================================
Compara win rates usando ambos metodos de take profit.
"""
import sys, os, warnings, importlib
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tradinglatino_hmm_clean as tl
importlib.reload(tl)

ASSET = "BTC-USD"
TIMEFRAMES_TO_TEST = ["1h", "4h", "1d"]
TP_FIXO = tl.TAKE_PROFIT_PCT
TP_ATR_MULT = tl.TP_ATR_MULT
MAX_BARS_BY_TF = tl.MAX_BARS_BY_TF

print("=" * 70)
print("BACKTEST COMPARATIVO: TP FIJO vs TP DINAMICO (ATR)")
print(f"Activo: {ASSET}")
print(f"Threshold: {tl.SIGNAL_SCORE_THRESHOLD} | Minimo dinamico: {tl.DYNAMIC_THRESHOLD_MIN}")
print(f"TP ATR Multiplicador: {TP_ATR_MULT}x ATR")
print("=" * 70)

results = {}

for tf in TIMEFRAMES_TO_TEST:
    print(f"\n--- TIMEFRAME: {tf} ---")
    
    # Download
    print(f"  Descargando {tf}...", end=" ", flush=True)
    df = tl.load_data(ASSET, tf)
    if df is None or len(df) < 200:
        print(f"INSUFICIENTE: {len(df) if df is not None else 0} velas")
        continue
    print(f"{len(df)} velas OK")
    
    # Indicators
    print(f"  Calculando indicadores...", end=" ", flush=True)
    df = tl.compute_all_indicators(df)
    print("OK")
    
    # HMM
    print(f"  Entrenando HMM...", end=" ", flush=True)
    features = tl.build_hmm_features(df)
    _, hmm_states, state_summary, _, _ = tl.fit_hmm(features)
    # Alinear estados: HMM puede devolver menos filas (NaN dropeados)
    df["regime"] = np.nan
    valid_len = min(len(hmm_states), len(df))
    df.iloc[-valid_len:, df.columns.get_loc("regime")] = hmm_states[:valid_len]
    print("OK")
    
    # Current signal
    signal_info = tl.compute_signal(df, timeframe=tf)
    print(f"  Senal actual: {signal_info['signal']} (Fuerza: {signal_info['strength']}%)")
    print(f"  Scores: LONG={signal_info['signal_score_long']:.1f} SHORT={signal_info['signal_score_short']:.1f}")
    
    # Parameters
    tp_fixed = TP_FIXO.get(tf, 2.0)
    max_bars = MAX_BARS_BY_TF.get(tf, 14)
    
    # === VERIFICATION: TP FIXED ===
    long_fixed, short_fixed = [], []
    for i in range(len(df) - max_bars - 1):
        # LONG
        if df["signal_long"].iloc[i]:
            entry = float(df["Close"].iloc[i])
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0: continue
            max_r = (window["High"].max() - entry) / entry * 100.0
            long_fixed.append({"won": max_r >= tp_fixed, "max_return": round(max_r, 2)})
        # SHORT
        if df["signal_short"].iloc[i]:
            entry = float(df["Close"].iloc[i])
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0: continue
            max_r = (entry - window["Low"].min()) / entry * 100.0
            short_fixed.append({"won": max_r >= tp_fixed, "max_return": round(max_r, 2)})
    
    # === VERIFICATION: TP ATR ===
    long_atr, short_atr = [], []
    for i in range(len(df) - max_bars - 1):
        # LONG
        if df["signal_long"].iloc[i]:
            entry = float(df["Close"].iloc[i])
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0: continue
            max_r = (window["High"].max() - entry) / entry * 100.0
            atr_e = float(df["atr"].iloc[i]) if "atr" in df.columns else 0
            tp_atr = (atr_e / entry * 100) * TP_ATR_MULT if atr_e > 0 else tp_fixed
            tp_used = max(tp_fixed, tp_atr)
            long_atr.append({"won": max_r >= tp_used, "max_return": round(max_r, 2), "tp_used": round(tp_used, 2)})
        # SHORT
        if df["signal_short"].iloc[i]:
            entry = float(df["Close"].iloc[i])
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0: continue
            max_r = (entry - window["Low"].min()) / entry * 100.0
            atr_e = float(df["atr"].iloc[i]) if "atr" in df.columns else 0
            tp_atr = (atr_e / entry * 100) * TP_ATR_MULT if atr_e > 0 else tp_fixed
            tp_used = max(tp_fixed, tp_atr)
            short_atr.append({"won": max_r >= tp_used, "max_return": round(max_r, 2), "tp_used": round(tp_used, 2)})
    
    # Stats helper
    def calc_stats(lst, label):
        n = len(lst)
        if n == 0: return {"label": label, "total": 0, "wins": 0, "wr": 0, "avg_ret": 0}
        wins = sum(1 for r in lst if r["won"])
        avg_ret = float(np.mean([r["max_return"] for r in lst]))
        return {"label": label, "total": n, "wins": wins, "wr": round(wins/n*100, 1), "avg_ret": round(avg_ret, 2)}
    
    # Aggregate
    def fmt_stats(stats_dict):
        lines = []
        for k, v in stats_dict.items():
            lines.append(f"  {v['label']:<20} Total:{v['total']:>4}  Wins:{v['wins']:>4}  WR:{v['wr']:>6.1f}%  AvgRet:{v['avg_ret']:>7.2f}%")
        return '\n'.join(lines)
    
    stats_f = {"LONG": calc_stats(long_fixed, "TP_FIJO_LONG"), "SHORT": calc_stats(short_fixed, "TP_FIJO_SHORT")}
    stats_a = {"LONG": calc_stats(long_atr, "TP_ATR_LONG"), "SHORT": calc_stats(short_atr, "TP_ATR_SHORT")}
    
    total_f = stats_f["LONG"]["total"] + stats_f["SHORT"]["total"]
    wins_f = stats_f["LONG"]["wins"] + stats_f["SHORT"]["wins"]
    wr_f = round(wins_f/total_f*100, 1) if total_f > 0 else 0
    
    total_a = stats_a["LONG"]["total"] + stats_a["SHORT"]["total"]
    wins_a = stats_a["LONG"]["wins"] + stats_a["SHORT"]["wins"]
    wr_a = round(wins_a/total_a*100, 1) if total_a > 0 else 0
    
    # Print results
    print(f"\n  RESULTADOS {tf} (TP base: {tp_fixed}%, ventana: {max_bars} velas):")
    print(f"  {'-'*65}")
    print(f"  {'Metodo':<22} {'Total':>6} {'Ganadas':>8} {'WR':>8} {'RetornoProm':>12}")
    print(f"  {'-'*65}")
    for lado in ["LONG", "SHORT"]:
        s_f = stats_f[lado]
        s_a = stats_a[lado]
        print(f"  {'TP_FIJO_'+lado:<22} {s_f['total']:>6} {s_f['wins']:>8} {s_f['wr']:>7.1f}% {s_f['avg_ret']:>11.2f}%")
        print(f"  {'TP_ATR_'+lado:<22} {s_a['total']:>6} {s_a['wins']:>8} {s_a['wr']:>7.1f}% {s_a['avg_ret']:>11.2f}%")
        if s_f['total'] > 0:
            tp_means = [r.get('tp_used', 0) for r in (long_atr if lado=='LONG' else short_atr)]
            avg_tp = round(float(np.mean(tp_means)), 2) if tp_means else 0
            print(f"  {'TP_ATR_promedio_usado':<22} -> {avg_tp}%")
    
    print(f"\n  GLOBAL {tf}:")
    print(f"    TP FIJO      -> WR: {wr_f}% ({wins_f}/{total_f})")
    print(f"    TP DINAMICO  -> WR: {wr_a}% ({wins_a}/{total_a})")
    print(f"    Diferencia   -> {round(wr_a - wr_f, 1)}%")
    if total_f > 0 and total_f != total_a:
        print(f"    Senales dif  -> {total_f - total_a} ({round((total_f-total_a)/total_f*100, 1)}%)")
    
    results[tf] = {"tp_fixed": tp_fixed, "wr_f": wr_f, "wr_a": wr_a, "total_f": total_f, "total_a": total_a}

# Global summary
print(f"\n{'='*70}")
print(f"RESUMEN GLOBAL - {ASSET}")
print(f"{'='*70}")
print(f"{'TF':<8} {'TP_Fijo%':<12} {'Senales':<10} {'WR_Fijo':<12} {'WR_ATR':<12} {'Diferencia':<12}")
print(f"{'-'*70}")
for tf, r in results.items():
    dif = round(r["wr_a"] - r["wr_f"], 1)
    print(f"{tf:<8} {r['tp_fixed']:<12} {r['total_a']:<10} {r['wr_f']:<12} {r['wr_a']:<12} {dif:<12}")

print(f"\nLeyenda:")
print(f"  TP FIJO = Take Profit fijo ({', '.join([f'{k}:{v}%' for k,v in TP_FIXO.items() if k != '1wk'])})")
print(f"  TP ATR = Take Profit dinamico: max(TP_fijo, ATR_entry * {TP_ATR_MULT} / precio * 100)")
print(f"  WR = % de senales que alcanzaron el TP dentro de la ventana")
print(f"{'='*70}")
