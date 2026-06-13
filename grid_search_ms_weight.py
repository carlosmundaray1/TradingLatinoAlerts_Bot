#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRID SEARCH MS WEIGHT - Busqueda del Peso Optimo de Markov Switching
Prueba combinaciones de W_MS_REGIME y SIGNAL_SCORE_THRESHOLD
para maximizar deteccion sin perder win rate.
"""
import json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util as _iu
_s = _iu.spec_from_file_location("hmm", SCRIPT_DIR / "tradinglatino_hmm_clean.py")
_H = _iu.module_from_spec(_s)
_s.loader.exec_module(_H)
ci = _H.compute_all_indicators
bf = _H.build_hmm_features
fh = _H.fit_hmm
cs = _H.compute_signal
cb = _H._classify_regime_bias
ld = _H.load_data
fd = _H._format_date
vh = _H.verify_signals_historically
import tradinglatino_regime_switching as MS

ASSET = "BTC-USD"
TIMEFRAMES = ["1d", "1wk"]
PERIODS = {"1d": "5y", "1wk": "10y"}

GRID_W_MS = list(range(3, 33, 3))
GRID_THRESHOLD = list(range(55, 81, 5))

W_DETECTION = 0.40
W_WINRATE = 0.35
W_FP_FN = 0.25

OUTPUT_FILE = "grid_search_ms_results.json"
OUTPUT_SUMMARY = "grid_search_ms_summary.txt"
line = "-" * 70
eq = "=" * 70
empty = ""

def safeprint(msg, log=None):
    try: print(msg)
    except: pass
    if log is not None: log.append(msg)

def lde(a, tf):
    _H.PERIOD_1D = PERIODS.get("1d", "2y")
    _H.PERIOD_1W = PERIODS.get("1wk", "4y")
    _H.PERIOD_4H = "2y"; _H.PERIOD_1H = "1y"
    return ld(a, tf)
def frc(st, idx, ss):
    dm = {}; ch = []
    for _, r in ss.iterrows(): dm[int(r["state"])] = r["description"]
    ps = st[0]; cso = 0
    for i in range(1, len(st)):
        if st[i] != ps:
            f2 = dm.get(int(ps), f"R{ps}")
            t2 = dm.get(int(st[i]), f"R{st[i]}")
            ch.append({"idx": i, "date": idx[i],
                "fr": int(ps), "to": int(st[i]),
                "fd": f2, "td": t2,
                "fb": cb(f2), "tb": cb(t2), "dur": i-cso})
            ps = st[i]; cso = i
    return ch

def fsc(df):
    ch = []; ps = "FLAT"
    for i in range(len(df)):
        il = (df["signal_long"].iloc[i] == 1) if "signal_long" in df.columns else False
        ish = (df["signal_short"].iloc[i] == 1) if "signal_short" in df.columns else False
        cur = "LONG" if il else ("SHORT" if ish else "FLAT")
        if cur != ps:
            ch.append({"idx": i, "fr": ps, "to": cur})
            ps = cur
    return ch

def crc(rc, sc, ml=5):
    res = []; used = set()
    for s2 in sc:
        if s2["fr"] == s2["to"]: continue
        best = None; bl = None; bri = -1
        for ri, r2 in enumerate(rc):
            if ri in used: continue
            if r2["idx"] <= s2["idx"] and r2["idx"] >= s2["idx"] - ml:
                lag = s2["idx"] - r2["idx"]
                if best is None or lag < bl: best = r2; bl = lag; bri = ri
        if best:
            used.add(bri)
            ts2 = s2["to"]; tb2 = best["tb"]; td3 = best["td"].upper()
            aligned = False
            if ts2 == "SHORT" and s2["fr"] != "SHORT":
                aligned = (tb2 == "bearish" or "BAJISTA" in td3)
            elif ts2 == "LONG" and s2["fr"] != "LONG":
                aligned = (tb2 == "bullish" or "ALCISTA" in td3)
            res.append({"tp": "ok", "al": aligned})
        else: res.append({"tp": "fn"})
    for ri in range(len(rc)):
        if ri not in used: res.append({"tp": "fp"})
    tot = len([s for s in sc if s["fr"] != s["to"]])
    det = sum(1 for r in res if r["tp"] == "ok")
    fp = sum(1 for r in res if r["tp"] == "fp")
    fn2 = sum(1 for r in res if r["tp"] == "fn")
    return {"det": det, "tot": tot, "fp": fp, "fn": fn2,
            "rate": round(det/tot*100,1) if tot else 0}

def score_combo(metrics):
    det = metrics.get("det_rate_d", 0) / 100.0
    wr = metrics.get("wr_d", 0) / 100.0
    fp = metrics.get("fp_d", 0)
    fn = metrics.get("fn_d", 0)
    tot = metrics.get("tot_d", 1)
    fp_fn_ratio = 1.0 - min(1.0, (fp+fn)/max(tot,1))
    score = (det*W_DETECTION + wr*W_WINRATE + fp_fn_ratio*W_FP_FN) * 100
    return round(score, 1)

def run():
    log = []
    safeprint(eq, log)
    safeprint("  GRID SEARCH MS - Peso Optimo de Markov Switching", log)
    safeprint(eq, log)
    safeprint(f"  Activo: {ASSET}", log)
    safeprint(f"  Timeframes: {TIMEFRAMES}", log)
    safeprint(f"  W_MS_REGIME: {GRID_W_MS}", log)
    safeprint(f"  SIGNAL_THRESHOLD: {GRID_THRESHOLD}", log)
    total = len(GRID_W_MS) * len(GRID_THRESHOLD)
    safeprint(f"  Total combinaciones: {total}", log)
    safeprint(empty, log)
    safeprint(line, log)
    safeprint("  FASE 1: Pre-calculando datos e indicadores...", log)
    safeprint(line, log)
    tf_cache = {}
    for tf in TIMEFRAMES:
        safeprint(f"\n  -- {tf} --", log)
        t0 = time.time()
        df = lde(ASSET, tf)
        if df is None or len(df) < 100:
            safeprint("  ERROR: datos insuficientes", log); continue
        safeprint(f"  {len(df)} velas ({time.time()-t0:.0f}s)", log)
        t1 = time.time()
        df = ci(df)
        safeprint(f"  Indicadores OK ({time.time()-t1:.0f}s)", log)
        t2 = time.time()
        feat = bf(df)
        safeprint(f"  Features: {feat.shape[1]} cols ({time.time()-t2:.0f}s)", log)
        t3 = time.time()
        m, st0, ss0, bic, tr0 = fh(feat)
        if m is None or len(st0) == 0:
            safeprint("  HMM fallo", log); continue
        safeprint(f"  HMM: {int(st0.max())+1} estados ({time.time()-t3:.0f}s)", log)
        dh = df.iloc[:len(st0)].copy()
        dh["regime"] = st0
        _ = cs(dh, tf)
        sig_changes = fsc(dh)
        rc0 = frc(st0, dh.index, ss0)
        cr0 = crc(rc0, sig_changes, 5)
        hv = vh(dh, tf)
        hw = hv["overall_win_rate"] if hv else 0
        safeprint(f"  HMM baseline: det={cr0['rate']}% WR={hw}%", log)
        mf, ms_st, mpr, mm, mss = MS.fit_markov_switching(df)
        if mf is None or len(ms_st) == 0:
            safeprint("  MS fallo", log); continue
        safeprint(f"  MS: {len(np.unique(ms_st))} estados", log)
        safeprint(f"  Total: {time.time()-t0:.0f}s", log)
        tf_cache[tf] = {
            "df": df, "sig_changes": sig_changes,
            "ms_st": ms_st, "mss": mss,
        }

    if not tf_cache:
        safeprint("  ERROR: No data", log); return

    safeprint(empty, log)
    safeprint(line, log)
    safeprint("  FASE 2: Probando combinaciones...", log)
    safeprint(line, log)
    all_results = []
    for w_ms, thr in itertools.product(GRID_W_MS, GRID_THRESHOLD):
        combo = f"W{w_ms}_T{thr}"
        t_start = time.time()
        metrics = {"w_ms": w_ms, "threshold": thr}
        for tf in TIMEFRAMES:
            if tf not in tf_cache: continue
            c = tf_cache[tf]
            dm2 = c["df"].iloc[:len(c["ms_st"])].copy()
            dm2 = MS.compute_signal_scores_with_ms( dm2, c["ms_st"], c["mss"], weight=w_ms, threshold=thr)
            dm2["regime"] = c["ms_st"]
            if "signal_long" not in dm2.columns:
                dm2["signal_long"] = 0; dm2["signal_short"] = 0
            sc2 = fsc(dm2)
            rc2 = MS.find_ms_regime_changes(c["ms_st"], dm2.index, c["mss"])
            cr2 = crc(rc2, sc2, 5)
            mv = vh(dm2, tf)
            tf_key = "d" if tf == "1d" else "wk"
            metrics[f"det_rate_{tf_key}"] = cr2["rate"]
            metrics[f"fp_{tf_key}"] = cr2["fp"]
            metrics[f"fn_{tf_key}"] = cr2["fn"]
            metrics[f"tot_{tf_key}"] = cr2["tot"]
            metrics[f"wr_{tf_key}"] = mv["overall_win_rate"] if mv else 0
        score = score_combo(metrics)
        metrics["score"] = score
        all_results.append(metrics)
        d1 = metrics.get("det_rate_d", 0)
        w1 = metrics.get("wr_d", 0)
        safeprint(f"  [{combo}] Score={score:.1f} Det_1d={d1:.1f}% WR_1d={w1:.1f}% ({time.time()-t_start:.1f}s)", log)

    safeprint(empty, log)
    safeprint(eq, log)
    safeprint("  FASE 3: Resultados", log)
    safeprint(eq, log)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    safeprint("\n  TOP 10:", log)
    hdr = "  #    W_MS   Thresh   Score   Det_1d    WR_1d   FP_1d   FN_1d"
    safeprint(hdr, log)
    safeprint("  " + "-" * (len(hdr.strip())), log)
    for i, r in enumerate(all_results[:10]):
        safeprint(f"  {i+1:>3} {r['w_ms']:>6} {r['threshold']:>8} {r['score']:>7.1f} "
            f"{r.get('det_rate_d',0):>7.1f}% {r.get('wr_d',0):>6.1f}% "
            f"{r.get('fp_d',0):>5} {r.get('fn_d',0):>5}", log)
    best = all_results[0]
    safeprint("\n  MEJOR: W_MS=" + str(best["w_ms"]) + " THRESHOLD=" + str(best["threshold"]), log)
    safeprint(f"  Score: {best['score']:.1f}", log)
    safeprint(f"  Det_1d: {best.get('det_rate_d',0):.1f}%", log)
    safeprint(f"  WR_1d: {best.get('wr_d',0):.1f}%", log)
    safeprint(f"  Det_1wk: {best.get('det_rate_wk',0):.1f}%", log)
    safeprint(f"  WR_1wk: {best.get('wr_wk',0):.1f}%", log)
    safeprint(f"  FP/FN_1d: {best.get('fp_d',0)}/{best.get('fn_d',0)}", log)
    output = {
        "grid": {"w_ms": GRID_W_MS, "threshold": GRID_THRESHOLD},
        "best": best, "top_10": all_results[:10],
        "all": all_results, "ts": datetime.now().isoformat(),
    }
    with open(OUTPUT_FILE, "w") as fout:
        json.dump(output, fout, indent=2, default=str)
    safeprint(f"\n  Resultados guardados: {OUTPUT_FILE}", log)
    with open(OUTPUT_SUMMARY, "w") as fout:
        fout.write("GRID SEARCH MS RESULTS\n")
        fout.write("-" * 60 + "\n")
        fout.write(f"Timestamp: {datetime.now().isoformat()}\n")
        fout.write(f"Asset: {ASSET}\n")
        fout.write(f"Combinations: {len(all_results)}\n\n")
        fout.write("TOP 10:\n")
        for i, r in enumerate(all_results[:10]):
            fout.write(f"{i+1:>3}. W_MS={r['w_ms']:>2} Th={r['threshold']:>2} "
                f"Score={r['score']:>6.1f} "
                f"Det_d={r.get('det_rate_d',0):>5.1f}% "
                f"WR_d={r.get('wr_d',0):>5.1f}% "
                f"FP={r.get('fp_d',0):>3} FN={r.get('fn_d',0):>3}\n")
        fout.write(f"\nBEST: W_MS={best['w_ms']} THRESHOLD={best['threshold']}\n")
        fout.write(f"Score: {best['score']:.1f}\n")
        fout.write(f"Det_1d: {best.get('det_rate_d',0):.1f}%\n")
        fout.write(f"WR_1d: {best.get('wr_d',0):.1f}%\n")
    safeprint(f"  Resumen: {OUTPUT_SUMMARY}", log)
    safeprint(empty, log)
    safeprint(eq, log)
    safeprint("  GRID SEARCH MS COMPLETADO", log)
    safeprint(eq, log)

if __name__ == "__main__":
    run()
