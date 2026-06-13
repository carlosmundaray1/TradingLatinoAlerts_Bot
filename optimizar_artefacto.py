#!/usr/bin/env python3
"""
OPTIMIZACION MULTI-OBJETIVO: TradingLatino HMM
Barre thresholds, consecutive bars y trailing stops para encontrar
la combinacion optima que maximiza el Win Rate combinado.

Uso:
    python optimizar_artefacto.py [--asset BTC-USD] [--tfs 1h,4h,1d,1w]
"""
import sys, os, time, warnings, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Forzar stdout a utf-8 para evitar errores de encoding con caracteres especiales
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

# ── Importar el modulo principal ──
import tradinglatino_hmm_clean as tl

# ── Parse args ──
ASSET = "BTC-USD"
TFS = ["1h", "4h"]
for i, a in enumerate(sys.argv[1:]):
    if a == "--asset" and i+1 < len(sys.argv):
        ASSET = sys.argv[i+2]
    if a == "--tfs" and i+1 < len(sys.argv):
        TFS = sys.argv[i+2].split(",")

# ── Grid de parametros ──
THRESHOLDS = [50, 55, 60, 65, 70, 75]
CONS_BARS = [1, 2, 3]
TRAIL_PCTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

N_COMBOS = len(THRESHOLDS) * len(CONS_BARS) * len(TRAIL_PCTS)
print(f"{'='*80}")
print(f"OPTIMIZACION MULTI-OBJETIVO: {ASSET}")
print(f"Timeframes: {', '.join(TFS)}")
print(f"Grid: {len(THRESHOLDS)} thresholds x {len(CONS_BARS)} cons x {len(TRAIL_PCTS)} trail = {N_COMBOS} combos")
print(f"{'='*80}")

for TF in TFS:
    print(f"\n{'-'*80}")
    print(f"  PROCESANDO {ASSET} [{TF}]")
    print(f"{'-'*80}")
    try:
        t0 = time.time()
        # ── Fase 1: Descarga (una sola vez) ──
        print(f"  Descargando datos...", end=" ")
        df = tl.load_data(ASSET, TF)
        if df is None or len(df) < 100:
            print(f"DATOS INSUFICIENTES")
            continue
        print(f"{len(df)} velas ({time.time()-t0:.1f}s)")

        # ── Fase 2: Indicadores (una sola vez) ──
        t1 = time.time()
        print(f"  Calculando indicadores...", end=" ")
        df = tl.compute_all_indicators(df)
        print(f"{time.time()-t1:.1f}s")

        # ── Fase 3: HMM (una sola vez) ──
        t2 = time.time()
        print(f"  Entrenando HMM...", end=" ")
        features_df = tl.build_hmm_features(df)
        _, states, state_summary, _, trans_mat = tl.fit_hmm(features_df)
        if len(states) == 0:
            print(f"HMM FALLIDO")
            continue
        print(f"{time.time()-t2:.1f}s")

        # ── Fase 4: Pre-computar mascaras de regimen ──
        # (las mascaras no cambian, solo las senales basadas en score)
        state_bias_map = {}
        for _, r in state_summary.iterrows():
            state_bias_map[int(r["state"])] = tl._classify_regime_bias(r["description"])
        df["regime_bias"] = "neutral"
        if "regime" in df.columns:
            df["regime_bias"] = df["regime"].map(state_bias_map).fillna("neutral")
        bearish_mask = df["regime_bias"] == "bearish"
        bullish_mask = df["regime_bias"] == "bullish"

        # Guardar scores originales (no cambian)
        orig_score_long = df["signal_score_long"].copy()
        orig_score_short = df["signal_score_short"].copy()

        # ── Fase 5: Grid search ──
        resultados = []
        total = len(THRESHOLDS) * len(CONS_BARS) * len(TRAIL_PCTS)
        idx = 0
        t_search = time.time()
        for th, cons in itertools.product(THRESHOLDS, CONS_BARS):
            # Re-generar senales con nuevo threshold
            df["signal_raw_long"] = orig_score_long >= th
            df["signal_raw_short"] = orig_score_short >= th
            df["signal_long"] = tl._consecutive_bars_filter(df["signal_raw_long"], cons)
            df["signal_short"] = tl._consecutive_bars_filter(df["signal_raw_short"], cons)

            # Re-aplicar filtro de regimen
            if bearish_mask.any():
                df.loc[bearish_mask, "signal_long"] = False
            if bullish_mask.any():
                df.loc[bullish_mask, "signal_short"] = False

            # Verificacion base (TP fijo)
            verification = tl.verify_signals_historically(df, TF)
            base_wr = verification["overall_win_rate"]
            n_signals = verification["total_signals"]

            for tp in TRAIL_PCTS:
                idx += 1
                # Trailing stop
                trailing = tl.verify_with_trailing_stop(df, TF, tp)
                comb_wr = trailing["overall_win_rate_combined"]
                trail_wr = trailing["overall_win_rate_ts"]

                resultados.append({
                    "threshold": th,
                    "cons_bars": cons,
                    "trail_pct": tp,
                    "base_wr": round(base_wr, 1),
                    "trail_wr": round(trail_wr, 1),
                    "comb_wr": round(comb_wr, 1),
                    "n_signals": n_signals,
                    "n_long": verification["stats"]["LONG"]["num_signals"],
                    "n_short": verification["stats"]["SHORT"]["num_signals"],
                    "wr_long": verification["stats"]["LONG"]["win_rate"],
                    "wr_short": verification["stats"]["SHORT"]["win_rate"],
                })

            # Progress
            pct = idx / total * 100
            elapsed = time.time() - t_search
            eta = (elapsed / max(idx, 1)) * (total - idx) if idx > 0 else 0
            print(f"  Progreso: {idx}/{total} ({pct:.0f}%) | "
                  f"th={th} cons={cons} | "
                  f"mejor combWR={max(r['comb_wr'] for r in resultados):.1f}% | "
                  f"ETA {eta:.0f}s")

        # ── Fase 6: Mostrar TOP 20 ──
        resultados.sort(key=lambda r: (-r["comb_wr"], -r["n_signals"]))
        print(f"\n  {'='*70}")
        print(f"  TOP 20 RESULTADOS: {ASSET} [{TF}]")
        print(f"  {'='*70}")
        print(f"  {'#':>3} {'Th':>5} {'Cons':>5} {'Trail':>7} {'BaseWR':>9} {'CombWR':>8} {'Senales':>8} {'LWR':>6} {'SWR':>6}")
        print(f"  {'-'*3} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
        for i, r in enumerate(resultados[:20]):
            print(f"  {i+1:>3} {r['threshold']:>5} {r['cons_bars']:>5} "
                  f"{r['trail_pct']:>5.1f}% {r['base_wr']:>7.1f}% "
                  f" {r['comb_wr']:>6.1f}% {r['n_signals']:>8} "
                  f"{r['wr_long']:>5.1f}% {r['wr_short']:>5.1f}%")

        # ── Mostrar mejor combo por threshold ──
        print(f"\n  Mejor combo por threshold:")
        print(f"  {'Th':>5} {'Cons':>5} {'Trail':>7} {'BaseWR':>9} {'CombWR':>8} {'Senales':>8}")
        for th in THRESHOLDS:
            best = max([r for r in resultados if r["threshold"] == th],
                       key=lambda r: (r["comb_wr"], r["n_signals"]))
            print(f"  {best['threshold']:>5} {best['cons_bars']:>5} "
                  f"{best['trail_pct']:>5.1f}% {best['base_wr']:>7.1f}% "
                  f" {best['comb_wr']:>6.1f}% {best['n_signals']:>8}")

        # ── Guardar resultados a CSV ──
        csv_path = f"optimizacion_{ASSET}_{TF}.csv"
        pd.DataFrame(resultados).to_csv(csv_path, index=False)
        print(f"\n  Resultados guardados en: {csv_path}")

        # ── Recomendacion final ──
        best = resultados[0]
        print(f"\n  {'*'*50}")
        print(f"  RECOMENDACION OPTIMA: {ASSET} [{TF}]")
        print(f"  {'*'*50}")
        print(f"    Threshold:           {best['threshold']}")
        print(f"    Consecutive bars:    {best['cons_bars']}")
        print(f"    Trailing stop:       {best['trail_pct']:.1f}%")
        print(f"    Base WR (TP Fijo):   {best['base_wr']:.1f}%")
        print(f"    Combined WR:         {best['comb_wr']:.1f}%")
        print(f"    Total senales:       {best['n_signals']}")

    except Exception as e:
        print(f"  ERROR en {TF}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"OPTIMIZACION COMPLETADA")
print(f"{'='*80}")
