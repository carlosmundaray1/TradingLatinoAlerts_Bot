#!/usr/bin/env python3
import os, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util as _iu
_s = _iu.spec_from_file_location("hmm", SCRIPT_DIR / "tradinglatino_hmm_clean.py")
_H = _iu.module_from_spec(_s)
_s.loader.exec_module(_H)
ci=_H.compute_all_indicators; bf=_H.build_hmm_features; fh=_H.fit_hmm
cs=_H.compute_signal; cb=_H._classify_regime_bias; ld=_H.load_data
fd=_H._format_date; vh=_H.verify_signals_historically
ST=_H.SIGNAL_SCORE_THRESHOLD
import tradinglatino_regime_switching as MS

A=["BTC-USD","XRP-USD"]; T=["1d","1wk"]
P={"1d":"5y","1wk":"10y"}; ML=5; LOG=[]
def pr(m):
    try: print(m)
    except: pass
    LOG.append(m)

def lde(a,tf):
    _H.PERIOD_1D=P.get("1d","2y")
    _H.PERIOD_1W=P.get("1wk","4y")
    _H.PERIOD_4H="2y"; _H.PERIOD_1H="1y"
    return ld(a,tf)

def frc(st,idx,ss):
    dm={}; ch=[]
    for _,r in ss.iterrows(): dm[int(r["state"])]=r["description"]
    ps=st[0]; cso=0
    for i in range(1,len(st)):
        if st[i]!=ps:
            f2=dm.get(int(ps),f"R{ps}")
            t2=dm.get(int(st[i]),f"R{st[i]}")
            ch.append({"idx":i,"date":idx[i],"ds":fd(idx[i]),
                "fr":int(ps),"to":int(st[i]),
                "fd":f2,"td":t2,"fb":cb(f2),"tb":cb(t2),"dur":i-cso})
            ps=st[i]; cso=i
    return ch

def fsc(df):
    ch=[]; ps="FLAT"
    for i in range(len(df)):
        il=bool(df["signal_long"].iloc[i]) if "signal_long" in df.columns else False
        ish=bool(df["signal_short"].iloc[i]) if "signal_short" in df.columns else False
        cur="LONG" if il else ("SHORT" if ish else "FLAT")
        if cur!=ps:
            ch.append({"idx":i,"date":df.index[i],"ds":fd(df.index[i]),
                "fr":ps,"to":cur,"pr":float(df["Close"].iloc[i])})
            ps=cur
    return ch

def crc(rc,sc,ml=5):
    res=[]; used=set()
    for s2 in sc:
        if s2["fr"]==s2["to"]: continue
        best=None; bl=None; bri=-1
        for ri,r2 in enumerate(rc):
            if ri in used: continue
            if r2["idx"]<=s2["idx"] and r2["idx"]>=s2["idx"]-ml:
                lag=s2["idx"]-r2["idx"]
                if best is None or lag<bl: best=r2; bl=lag; bri=ri
        if best:
            used.add(bri)
            ba=False; ts2=s2["to"]
            tb2=best["tb"]; td3=best["td"].upper()
            if ts2=="SHORT" and s2["fr"]!="SHORT":
                ba=(tb2=="bearish" or "BAJISTA" in td3)
            elif ts2=="LONG" and s2["fr"]!="LONG":
                ba=(tb2=="bullish" or "ALCISTA" in td3)
            res.append({"tp":"ok","lag":bl,"al":ba})
        else: res.append({"tp":"fn"})
    for ri,r2 in enumerate(rc):
        if ri not in used: res.append({"tp":"fp"})
    tot=len([s for s in sc if s["fr"]!=s["to"]])
    det=sum(1 for r in res if r["tp"]=="ok")
    fp=sum(1 for r in res if r["tp"]=="fp")
    fn2=sum(1 for r in res if r["tp"]=="fn")
    lags=[r["lag"] for r in res if r.get("lag")]
    al=float(np.mean(lags)) if lags else 0
    return {"det":det,"tot":tot,"fp":fp,"fn":fn2,
        "al":sum(1 for r in res if r.get("al")),
        "rate":round(det/tot*100,1) if tot else 0,"avg":round(al,1)}

def run():
    pr("="*60); pr("  HMM vs MARKOV SWITCHING"); pr("="*60)
    R={}
    for a in A:
        pr(f"--- {a} ---")
        for tf in T:
            pr(f"  TF: {tf}")
            df=lde(a,tf)
            if df is None or len(df)<100: pr("  ERROR"); continue
            pr(f"  {len(df)} velas")
            df=ci(df)
            # HMM
            pr("  -- HMM --")
            feat=bf(df)
            m,st0,ss0,bic,tr0=fh(feat)
            if m is None or len(st0)==0: pr("  HMM fallo"); continue
            dh=df.iloc[:len(st0)].copy()
            dh["regime"]=st0
            si=cs(dh,tf)
            rc0=frc(st0,dh.index,ss0)
            sc0=fsc(dh)
            cr0=crc(rc0,sc0,ML)
            pr(f"  HMM det: {cr0["det"]}/{cr0["tot"]} ({cr0["rate"]}%) FP:{cr0["fp"]} FN:{cr0["fn"]}")
            hv=vh(dh,tf)
            if hv and hv["total_signals"]>0: pr(f"  HMM WR: {hv["overall_win_rate"]}%")
            # MS
            pr("  -- MS --")
            mf,ms_st,mpr,mm,mss=MS.fit_markov_switching(df)
            if mf is None or len(ms_st)==0: pr("  MS fallo"); continue
            dm2=df.iloc[:len(ms_st)].copy()
            dm2=MS.compute_signal_scores_with_ms(dm2,ms_st,mss,weight=MS.W_MS_REGIME,threshold=ST)
            dm2['regime']=ms_st  # for verify_signals_historically
            rc2=MS.find_ms_regime_changes(ms_st,dm2.index,mss)
            sc2=fsc(dm2)
            cr2=crc(rc2,sc2,ML)
            pr(f"  MS det: {cr2["det"]}/{cr2["tot"]} ({cr2["rate"]}%) FP:{cr2["fp"]} FN:{cr2["fn"]}")
            mv=vh(dm2,tf)
            if mv and mv["total_signals"]>0: pr(f"  MS WR: {mv["overall_win_rate"]}%")
            pr(f"  HMM det={cr0["rate"]}% vs MS det={cr2["rate"]}%")
            if hv and mv: pr(f"  HMM WR={hv['overall_win_rate']}% vs MS WR={mv['overall_win_rate']}%")
            R[(a,tf)]={"hm":{"dt":cr0,"st":st0,"ss":ss0,"n":len(np.unique(st0))},
                "ms":{"dt":cr2,"st":ms_st,"ss":mss,"n":len(np.unique(ms_st))},
                "hv":hv,"mv":mv,"si":si}
    pr("")
    pr("="*60); pr("RESUMEN"); pr("="*60)
    for (a,tf),d in R.items():
        hd=d["hm"]["dt"]; md2=d["ms"]["dt"]
        hw=(d.get("hv") or {}).get("overall_win_rate",0)
        mw2=(d.get("mv") or {}).get("overall_win_rate",0)
        pr(f"  {a} {tf}: HMM det={hd["rate"]}% WR={hw}% | MS det={md2["rate"]}% WR={mw2}%")
    pr("COMPLETADO.")

if __name__=="__main__": run()
