#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARADOR HMM puro vs HMM+MS
Uso: python comparar_hmmms_simulacion.py
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import importlib.util as _u
_HMM = _u.module_from_spec((s:=_u.spec_from_file_location("hmm", SRC/"tradinglatino_hmm_clean.py")))
s.loader.exec_module(_HMM)
import tradinglatino_regime_switching as _MS

ASSET = "BTC-USD"
TFS = ["1d", "1wk"]
PER = {"1d": "5y", "1wk": "10y"}
OUTPUT_LOG = "comparacion_hmmms_alertas_log.txt"

def pl(msg):
    try:
        print(msg)
    except:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

def load(asset, tf):
    _HMM.PERIOD_1D = PER["1d"]
    _HMM.PERIOD_1W = PER["1wk"]
    return _HMM.load_data(asset, tf)

def find_rc(states, ix, ss):
    dm = {int(r["state"]): r["description"] for _, r in ss.iterrows()}
    ch, ps, cs = [], states[0], 0
    for i in range(1, len(states)):
        if states[i] != ps:
            fd = dm.get(int(ps), "R%d" % ps)
            td = dm.get(int(states[i]), "R%d" % states[i])
            ch.append({"idx": i, "date": ix[i], "fd": fd, "td": td, "dur": i - cs})
            ps, cs = states[i], i
    return ch

def find_sc(df):
    ch, ps = [], "FLAT"
    for i in range(len(df)):
        il = bool(df["signal_long"].iloc[i]) if "signal_long" in df.columns else False
        ish = bool(df["signal_short"].iloc[i]) if "signal_short" in df.columns else False
        cs = "LONG" if il else ("SHORT" if ish else "FLAT")
        if cs != ps:
            ch.append({"idx": i, "fr": ps, "to": cs, "price": float(df["Close"].iloc[i])})
            ps = cs
    return ch

def cross_ref(rc, sc, ml=5):
    used = set()
    det = fp = fn = 0
    for s in sc:
        if s["fr"] == s["to"]: continue
        found = False
        for ri, rv in enumerate(rc):
            if ri in used: continue
            if rv["idx"] <= s["idx"] < rv["idx"] + ml:
                used.add(ri); found = True; break
        if found: det += 1
        else: fn += 1
    fp = len(rc) - len(used)
    ts = len([s for s in sc if s["fr"] != s["to"]])
    return {"det": det, "ts": ts, "fp": fp, "fn": fn}

pl("=" * 70)
pl("  COMPARACION HMM puro vs HMM+MS")
pl("  Activo: %s" % ASSET)
pl("  Timeframes: %s" % ", ".join(TFS))
pl("")

results = {}
for tf in TFS:
    pl("\n  --- %s ---" % tf)
    df = load(ASSET, tf)
    if df is None or len(df) < 100:
        pl("  Datos insuficientes")
        continue
    df = _HMM.compute_all_indicators(df)
    feats = _HMM.build_hmm_features(df)
    model, states, ss, _, _ = _HMM.fit_hmm(feats)
    if model is None or len(states) == 0:
        pl("  HMM fallo")
        continue
    df_h = df.iloc[:len(states)].copy()
    df_h["regime"] = states

    # A) HMM puro
    pl("  [A] HMM puro...")
    df_a = df_h.copy()
    df_a = _HMM.compute_signal_scores_with_hmm(df_a, ss)
    si_a = _HMM.compute_signal(df_a, timeframe=tf)
    rc_a = find_rc(states, df_a.index, ss)
    sc_a = find_sc(df_a)
    cr_a = cross_ref(rc_a, sc_a)
    vf_a = _HMM.verify_signals_historically(df_a, tf)
    wr_a = vf_a["overall_win_rate"] if vf_a else 0
    sig_a = si_a["signal"]
    det_a = cr_a["det"]; ts_a = cr_a["ts"]; fp_a = cr_a["fp"]; fn_a = cr_a["fn"]
    dr_a = round(det_a / ts_a * 100, 1) if ts_a else 0
    pl("    Senal: %s (L=%.0f/S=%.0f)" % (sig_a, si_a["signal_score_long"], si_a["signal_score_short"]))
    pl("    Det: %d/%d (%s%%) FP=%d FN=%d | WR=%.1f%%" % (det_a, ts_a, dr_a, fp_a, fn_a, wr_a))

    # B) HMM+MS
    ms_ok = False
    df_b = df_h.copy()
    try:
        ms_r = _MS.fit_markov_switching(df_b)
        if ms_r and len(ms_r) >= 5 and ms_r[1] is not None and len(ms_r[1]) > 0:
            ms_ok = True
            df_b = _HMM.compute_signal_scores_with_hmm_ms(df_b, ss, ms_r[1], ms_r[4])
    except Exception as e:
        pl("    MS fallo: %s" % str(e))

    if ms_ok:
        si_b = _HMM.compute_signal(df_b, timeframe=tf)
        rc_b = find_rc(states, df_b.index, ss)
        sc_b = find_sc(df_b)
        cr_b = cross_ref(rc_b, sc_b)
        vf_b = _HMM.verify_signals_historically(df_b, tf)
        wr_b = vf_b["overall_win_rate"] if vf_b else 0
        sig_b = si_b["signal"]
        det_b = cr_b["det"]; ts_b = cr_b["ts"]; fp_b = cr_b["fp"]; fn_b = cr_b["fn"]
        dr_b = round(det_b / ts_b * 100, 1) if ts_b else 0
        pl("    Senal: %s (L=%.0f/S=%.0f)" % (sig_b, si_b["signal_score_long"], si_b["signal_score_short"]))
        pl("    Det: %d/%d (%s%%) FP=%d FN=%d | WR=%.1f%%" % (det_b, ts_b, dr_b, fp_b, fn_b, wr_b))
        pl("    DELTA: Det=%+.1f%% WR=%+.1f%% FP=%+d FN=%+d" % (dr_b - dr_a, wr_b - wr_a, fp_a - fp_b, fn_a - fn_b))
    else:
        pl("  [B] HMM+MS: No disponible")

    results[tf] = {
        "hmm": {"sig": sig_a, "sl": si_a["signal_score_long"], "ss": si_a["signal_score_short"],
                "det": det_a, "ts": ts_a, "fp": fp_a, "fn": fn_a, "dr": dr_a, "wr": wr_a},
        "hmm_ms": {"sig": sig_b if ms_ok else "N/A", "sl": si_b["signal_score_long"] if ms_ok else 0,
                   "ss": si_b["signal_score_short"] if ms_ok else 0,
                   "det": det_b if ms_ok else 0, "ts": ts_b if ms_ok else 0,
                   "fp": fp_b if ms_ok else 0, "fn": fn_b if ms_ok else 0,
                   "dr": dr_b if ms_ok else 0, "wr": wr_b if ms_ok else 0, "ok": ms_ok}
    }

# Summary
pl("\n" + "=" * 60)
pl("  RESUMEN COMPARATIVO")
pl("=" * 60)
for tf, d in results.items():
    h = d["hmm"]
    m = d["hmm_ms"]
    pl("  %s:" % tf)
    pl("    HMM:    %s | Det: %d/%d (%.1f%%) | WR: %.1f%% | FP=%d FN=%d" % (h["sig"], h["det"], h["ts"], h["dr"], h["wr"], h["fp"], h["fn"]))
    if m["ok"]:
        pl("    HMM+MS: %s | Det: %d/%d (%.1f%%) | WR: %.1f%% | FP=%d FN=%d" % (m["sig"], m["det"], m["ts"], m["dr"], m["wr"], m["fp"], m["fn"]))
        pl("    DELTA:  Det=%+.1f%% | WR=%+.1f%% | FP=%+d | FN=%+d" % (m["dr"] - h["dr"], m["wr"] - h["wr"], h["fp"] - m["fp"], h["fn"] - m["fn"]))
    else:
        pl("    HMM+MS: No disponible")

# Save log
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    pass
pl("\nLog: %s" % OUTPUT_LOG)
