#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERIFICADOR DE SEÑALES XRP-USD
Verifica si senales SHORT en 1d y 1wk son consistentes.
Uso: python verificar_xrp.py
"""
import os, sys, warnings, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import tradinglatino_hmm_clean as H
import tradinglatino_regime_switching as MS

ASSET = "XRP-USD"
TFS = ["1d", "1wk"]
OUT = "verificacion_xrp_resultados.txt"
H.PERIOD_1D = "5y"
H.PERIOD_1W = "10y"

def log(m):
    print(m)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(m + "\n")

def sep(t=None):
    s = "=" * 70
    log("")
    log(s)
    if t:
        log(f"  {t}")
        log(s)

def analizar(asset, tf):
    sep(f"ANALISIS {asset} [{tf}]")
    log("\n[1/5] Cargando datos...")
    df = H.load_data(asset, tf)
    if df is None or len(df) < 100:
        log("  ERROR: datos insuficientes")
        return None
    log(f"  {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}, {len(df)} velas")
    
    log("\n[2/5] Indicadores...")
    df = H.compute_all_indicators(df)
    
    log("\n[3/5] HMM...")
    feat = H.build_hmm_features(df)
    m, st, ss, _, _ = H.fit_hmm(feat)
    if m is None or len(st) == 0:
        log("  HMM fallo")
        return None
    log(f"  Estados: {ss['state'].nunique()}")
    for _, r in ss.iterrows():
        log(f"    S{int(r['state'])}: {r['description']} ({r['pct_time']:.1f}%)")
    
    df2 = df.copy()
    df2["regime"] = np.nan
    v = min(len(st), len(df2))
    df2.iloc[-v:, df2.columns.get_loc("regime")] = st[-v:]
    
    log("\n[4/5] Scores...")
    df2 = H.compute_signal_scores_with_hmm(df2, ss)
    
    log("\n[5/5] Senal + verificacion...")
    si = H.compute_signal(df2, timeframe=tf)
    
    log(f"\n  Senal ACTUAL: {si['signal']}")
    log(f"  Precio: ${si['price']:.4f}")
    log(f"  Fuerza: {si['strength']}%")
    log(f"  Score LONG: {si['signal_score_long']:.0f}")
    log(f"  Score SHORT: {si['signal_score_short']:.0f}")
    log(f"  Threshold: {si['score_threshold']}")
    
    cs = int(st[-1])
    row = ss[ss["state"] == cs]
    desc = row.iloc[0]["description"] if not row.empty else ""
    log(f"  Regimen: Estado {cs} {desc}")
    
    if si["signal"] == "SHORT":
        log("\n  --- DESGLOSE SHORT ---")
        tot = 0
        for c, v in sorted(si["score_breakdown_short"].items(), key=lambda x: -x[1]):
            ok = "[OK]" if v > 0 else "[NO]"
            log(f"    {ok} {c}: {v:+d}")
            tot += v if v > 0 else 0
        log(f"    TOTAL: {tot} (threshold: {H.SIGNAL_SCORE_THRESHOLD})")
    
    # Condiciones
    log("\n  --- CONDICIONES ---")
    for c, info in si["conditions"].items():
        ok = "[OK]" if info["met"] else "[NO]"
        log(f"    {ok} {c}")
    
    # Verificacion
    ver = H.verify_signals_historically(df2, tf)
    vt = H.verify_with_trailing_stop(df2, tf, 50.0)
    
    if ver and ver["total_signals"] > 0:
        sd = ver["short"]
        ns = len(sd)
        ws = sum(1 for r in sd if r["won"])
        wr = ws/ns*100 if ns > 0 else 0
        
        log(f"\n  --- SHORT HISTORICAS ({ns}) ---")
        log(f"  WR (TP fijo): {ws}/{ns} = {wr:.1f}%")
        if ns > 0:
            fav = np.mean([r["max_return"] for r in sd])
            log(f"  Retorno favorable prom: +{fav:.2f}%")
            btw = [r["bars_to_win"] for r in sd if r["bars_to_win"] is not None]
            if btw:
                log(f"  Velas prom hasta ganar: {np.mean(btw):.1f}")
            
            log(f"\n  Ultimas {min(10, ns)} SHORT:")
            for r in sorted(sd, key=lambda r: r["entry_date"], reverse=True)[:10]:
                d = pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d")
                w = "[WIN]" if r["won"] else "[LOSS]"
                log(f"    {d}: ${r['entry_price']:.4f} max={r['max_return']:+.2f}% {w}")
            
            los = [r for r in sd if not r["won"]]
            if los:
                log(f"\n  Perdedoras (falsos +):")
                for r in los:
                    d = pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d")
                    log(f"    {d}: ${r['entry_price']:.4f} max={r['max_return']:+.2f}% fin={r['final_return']:+.2f}%")
        
        ld = ver["long"]
        nl = len(ld)
        wl = sum(1 for r in ld if r["won"])
        log(f"\n  LONG: {nl} senales, WR={wl/nl*100:.1f}%" if nl > 0 else "")
    
    if vt and vt["total_signals"] > 0:
        tsd = vt.get("short", [])
        if tsd:
            tw = sum(1 for r in tsd if r["won_combined"])
            log(f"\n  TRAILING SHORT: {tw}/{len(tsd)} = {tw/len(tsd)*100:.1f}%")
    
    return {"si": si, "ver": ver, "vt": vt}

def comparar_ms(asset, tf):
    sep(f"HMM vs MS - {asset} {tf}")
    df = H.load_data(asset, tf)
    if df is None or len(df) < 100:
        return
    df = H.compute_all_indicators(df)
    
    log("\n--- HMM ---")
    feat = H.build_hmm_features(df)
    m, st, ss, _, _ = H.fit_hmm(feat)
    if m and len(st) > 0:
        dh = df.iloc[:len(st)].copy()
        dh["regime"] = st
        dh = H.compute_signal_scores_with_hmm(dh, ss)
        si = H.compute_signal(dh, tf)
        v = H.verify_signals_historically(dh, tf)
        log(f"  Senal: {si['signal']} S={si['signal_score_short']:.0f} L={si['signal_score_long']:.0f}")
        if v:
            log(f"  WR: {v['overall_win_rate']:.1f}%  SHORT: {v['stats']['SHORT']['win_rate']:.1f}%({v['stats']['SHORT']['num_signals']}) LONG: {v['stats']['LONG']['win_rate']:.1f}%({v['stats']['LONG']['num_signals']})")
    
    log("\n--- MS ---")
    mf, ms_st, _, _, mss = MS.fit_markov_switching(df)
    if mf and len(ms_st) > 0:
        dm = df.iloc[:len(ms_st)].copy()
        dm = MS.compute_signal_scores_with_ms(dm, ms_st, mss, weight=MS.W_MS_REGIME, threshold=H.SIGNAL_SCORE_THRESHOLD)
        dm["regime"] = ms_st
        dm["signal_long"] = (dm["signal_score_long"] >= H.SIGNAL_SCORE_THRESHOLD).astype(int)
        dm["signal_short"] = (dm["signal_score_short"] >= H.SIGNAL_SCORE_THRESHOLD).astype(int)
        si_ms = H.compute_signal(dm, tf)
        v_ms = H.verify_signals_historically(dm, tf)
        log(f"  Senal: {si_ms['signal']} S={si_ms['signal_score_short']:.0f} L={si_ms['signal_score_long']:.0f}")
        if v_ms:
            log(f"  WR: {v_ms['overall_win_rate']:.1f}%  SHORT: {v_ms['stats']['SHORT']['win_rate']:.1f}%({v_ms['stats']['SHORT']['num_signals']}) LONG: {v_ms['stats']['LONG']['win_rate']:.1f}%({v_ms['stats']['LONG']['num_signals']})")

def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    sep(f"VERIFICACION {ASSET}")
    log(f"Ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    res = {}
    for tf in TFS:
        try:
            r = analizar(ASSET, tf)
            if r:
                res[tf] = r
        except Exception as e:
            log(f"\nERROR {tf}: {e}")
            import traceback
            traceback.print_exc()
    
    for tf in TFS:
        try:
            comparar_ms(ASSET, tf)
        except Exception as e:
            log(f"\nERROR comparacion {tf}: {e}")
    
    sep("VEREDICTO")
    for tf in TFS:
        if tf in res:
            si = res[tf]["si"]
            ver = res[tf]["ver"]
            log(f"\n  {tf}: {si['signal']} (S={si['signal_score_short']:.0f}/L={si['signal_score_long']:.0f})")
            if ver and ver["total_signals"] > 0:
                sd = ver["short"]
                ns = len(sd)
                ws = sum(1 for r in sd if r["won"])
                wr = ws/ns*100 if ns > 0 else 0
                log(f"  WR SHORT: {wr:.1f}% ({ns} senales)")
                if wr >= 70:
                    log(f"  -> CONFIABLE (WR>=70%)")
                elif wr >= 50:
                    log(f"  -> MODERADO (WR 50-70%)")
                else:
                    log(f"  -> POCO CONFIABLE - POSIBLE BUG")
    log(f"\nResultados: {OUT}")

if __name__ == "__main__":
    main()
