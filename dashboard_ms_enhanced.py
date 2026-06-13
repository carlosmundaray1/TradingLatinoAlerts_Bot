#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_ms_enhanced.py
Dashboard mejorado con comparacion HMM vs Markov Switching (MS).
Diseno visual moderno tipo TradingView.

Uso:
    python dashboard_ms_enhanced.py [--asset BTC-USD] [--timeframes 1h,4h,1d]
"""
import os
import sys
import time
import warnings
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tradinglatino_hmm_clean as H
import tradinglatino_regime_switching as _MS

# ==============================================================================
# CONFIG
# ==============================================================================
ASSET: str = "BTC-USD"
TIMEFRAMES: List[str] = ["1h", "4h", "1d", "1wk"]
OUTPUT_HTML: str = "hmm_ms_dashboard_{ASSET}.html"
OPEN_BROWSER: bool = True

for arg in sys.argv[1:]:
    if arg.startswith("--asset="):
        ASSET = arg.split("=", 1)[1]
    elif arg.startswith("--timeframes="):
        TIMEFRAMES = [tf.strip() for tf in arg.split("=", 1)[1].split(",")]
    elif arg.startswith("--output="):
        OUTPUT_HTML = arg.split("=", 1)[1]
    elif arg == "--no-browser":
        OPEN_BROWSER = False
OUTPUT_HTML = OUTPUT_HTML.format(ASSET=ASSET)


# ==============================================================================
# HELPERS
# ==============================================================================# ==============================================================================
# HTML BUILDERS — VISUAL HELPERS
# ==============================================================================

def _signal_badge(signal: str) -> str:
    """Badge moderno para LONG/SHORT/FLAT con gradiente."""
    if signal == "LONG":
        return '<span class="sig-badge sig-long">\u25b2 LONG</span>'
    elif signal == "SHORT":
        return '<span class="sig-badge sig-short">\u25bc SHORT</span>'
    else:
        return '<span class="sig-badge sig-flat">\u2014 FLAT</span>'


def _winrate_bar(wr: float, width: int = 80) -> str:
    """Barra de win rate con gradiente de color y etiqueta."""
    if wr is None:
        return '<span class="wr-na">N/A</span>'
    color = "#3fb950" if wr >= 65 else ("#d29922" if wr >= 50 else "#f85149")
    bar = (
        f'<div class="wr-track" style="width:{width}px;">'
        f'<div class="wr-fill" style="width:{wr:.0f}%;background:{color};"></div>'
        f'</div>'
        f'<span class="wr-label" style="color:{color};">{wr:.1f}%</span>'
    )
    return bar


def _agreement_badge(hmm_signal: str, ms_signal: str) -> str:
    """Badge de acuerdo/divergencia entre modelos con icono."""
    if hmm_signal == ms_signal:
        if hmm_signal in ("LONG", "SHORT"):
            return '<span class="agree-badge agree-yes"><span class="agree-icon">\u2714</span> CONFIRMADO</span>'
        else:
            return '<span class="agree-badge agree-neutral"><span class="agree-icon">\u2795</span> AMBOS NEUTRAL</span>'
    else:
        return '<span class="agree-badge agree-no"><span class="agree-icon">\u26a0</span> DIVERGENCIA</span>'


def _delta_badge(delta: float, label: str) -> str:
    """Muestra la diferencia MS - HMM con color."""
    if delta > 0:
        color = "#3fb950"
        icon = "\u2191"
        text = f"+{delta:.1f}"
    elif delta < 0:
        color = "#f85149"
        icon = "\u2193"
        text = f"{delta:.1f}"
    else:
        color = "#8b949e"
        icon = "\u2014"
        text = "0.0"
    return (
        f'<div class="delta-card" style="border-left:3px solid {color};">'
        f'<div class="delta-label">{label}</div>'
        f'<div class="delta-val" style="color:{color};">{icon} {text}</div>'
        f'<div class="delta-sub">MS vs HMM</div>'
        f'</div>'
    )


# ==============================================================================
# PIPELINE
# ==============================================================================

def run_pipeline(asset: str = None, timeframes: list = None) -> Dict[str, Any]:
    """Ejecuta pipeline completo HMM + MS para todos los timeframes."""
    if asset is None:
        asset = ASSET
    if timeframes is None:
        timeframes = TIMEFRAMES
    H.ASSET = asset

    print("=" * 60)
    print(f"DASHBOARD HMM vs MS - {asset}")
    print("=" * 60)
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Output: {OUTPUT_HTML}")
    print("-" * 60)

    results: Dict[str, H.TimeframeData] = {}

    for tf in timeframes:
        print(f"\nProcesando {asset} [{tf}]...")
        try:
            t0 = time.time()
            df = H.load_data(asset, tf)
            if df is None or len(df) < 100:
                print(f"  Datos insuficientes para {tf}, saltando.")
                continue
            print(f"  Datos ({len(df)} velas, {time.time()-t0:.1f}s)")

            t0 = time.time()
            df = H.compute_all_indicators(df)
            print(f"  Indicadores ({time.time()-t0:.1f}s)")

            # HMM
            t0 = time.time()
            features_df = H.build_hmm_features(df)
            _, states, state_summary, _, trans_mat = H.fit_hmm(features_df)
            n_states = state_summary["state"].nunique()
            print(f"  HMM: {n_states} estados ({time.time()-t0:.1f}s)")

            df = df.copy()
            df["regime"] = np.nan
            vl = min(len(states), len(df))
            df.iloc[-vl:, df.columns.get_loc("regime")] = states[-vl:]

            df = H.compute_signal_scores_with_hmm(df, state_summary)

            # MS
            ms_ok = False
            ms_states = np.array([])
            ms_summary = pd.DataFrame()
            ms_signal_info = None
            ms_verification = None
            ms_changes = []
            try:
                print("  Entrenando MS...")
                ms_res = _MS.fit_markov_switching(df)
                if ms_res and len(ms_res) >= 5:
                    mf, mst, mpr, mm, mss = ms_res
                    if mf is not None and len(mst) > 0:
                        ms_ok = True
                        ms_states = mst
                        ms_summary = mss
                        df = H.compute_signal_scores_with_hmm_ms(df, state_summary, ms_states, ms_summary)
                        ms_signal_info = H.compute_signal(df, timeframe=tf)
                        ms_verification = H.verify_signals_historically(df, tf)
                        ms_changes = _MS.find_ms_regime_changes(ms_states, df.index, ms_summary)
                        ms_sig = ms_signal_info.get("signal", "N/A")
                        ms_wr_v = ms_verification.get("overall_win_rate", 0)
                        print(f"  MS OK: {ms_sig} WR={ms_wr_v:.1f}%")
            except Exception as e:
                print(f"  MS fallo: {e}")
            if not ms_ok:
                print("  MS: omitido")

            df = H.compute_precursor_signals(df)
            signal_info = H.compute_signal(df, timeframe=tf)
            print(f"  HMM: {signal_info.get('signal','N/A'):5s} "
                  f"Fuerza: {signal_info.get('strength',0):.0f}% "
                  f"Precio: ${signal_info.get('price',0):.2f}")

            verification = H.verify_signals_historically(df, tf)
            regime_changes = H._detect_regime_changes(states, df.index, state_summary, max_alerts=15)

            results[tf] = H.TimeframeData(
                df_full=df,
                states_full=states,
                state_summary=state_summary,
                trans_mat=trans_mat,
                signal_info=signal_info,
                regime_changes=regime_changes,
                verification=verification,
                ms_states=ms_states if ms_ok else None,
                ms_summary=ms_summary if ms_ok else None,
                ms_signal_info=ms_signal_info,
                ms_regime_changes=ms_changes if ms_ok else None,
                ms_verification=ms_verification,
            )
            print("  Listo.")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nNo se pudo procesar ningun timeframe.")
        sys.exit(1)
    return results


# ==============================================================================
# DASHBOARD HTML — COMPARISON PANELS
# ==============================================================================

def _generate_ms_comparison_html(data: H.TimeframeData, timeframe: str) -> str:
    """Panel de comparacion HMM vs MS con diseno moderno."""
    hmm_sig = data.signal_info
    ms_sig = data.ms_signal_info
    hmm_ver = data.verification
    ms_ver = data.ms_verification

    hmm_signal = hmm_sig.get("signal", "N/A") if hmm_sig else "N/A"
    ms_signal = ms_sig.get("signal", "N/A") if ms_sig else "N/A"

    hmm_score_long = hmm_sig.get("signal_score_long", 0) if hmm_sig else 0
    hmm_score_short = hmm_sig.get("signal_score_short", 0) if hmm_sig else 0
    ms_score_long = ms_sig.get("signal_score_long", 0) if ms_sig else 0
    ms_score_short = ms_sig.get("signal_score_short", 0) if ms_sig else 0

    hmm_wr = hmm_ver.get("overall_win_rate", None) if hmm_ver else None
    ms_wr = ms_ver.get("overall_win_rate", None) if ms_ver else None
    hmm_total = hmm_ver.get("total_signals", 0) if hmm_ver else 0
    ms_total = ms_ver.get("total_signals", 0) if ms_ver else 0

    hmm_state_desc = ""
    if data.state_summary is not None and not data.state_summary.empty:
        cs = int(data.states_full[-1]) if len(data.states_full) > 0 else -1
        row = data.state_summary[data.state_summary["state"] == cs]
        if not row.empty:
            hmm_state_desc = row.iloc[0]["description"]

    ms_state_desc = ""
    if data.ms_summary is not None and not data.ms_summary.empty and data.ms_states is not None:
        cs_ms = int(data.ms_states[-1]) if len(data.ms_states) > 0 else -1
        row = data.ms_summary[data.ms_summary["state"] == cs_ms]
        if not row.empty:
            ms_state_desc = row.iloc[0]["description"]

    hmm_strength = hmm_sig.get("strength", 0) if hmm_sig else 0
    ms_strength = ms_sig.get("strength", 0) if ms_sig else 0

    hmm_n_states = data.state_summary["state"].nunique() if data.state_summary is not None else 0
    ms_n_states = data.ms_summary["state"].nunique() if data.ms_summary is not None else 0
    hmm_changes = len(data.regime_changes) if data.regime_changes else 0
    ms_changes = len(data.ms_regime_changes) if data.ms_regime_changes else 0

    # Computed values
    agreement = _agreement_badge(hmm_signal, ms_signal)
    hmm_badge = _signal_badge(hmm_signal)
    ms_badge = _signal_badge(ms_signal) if data.ms_signal_info else '<span class="sig-badge sig-na">N/A</span>'
    hmm_wr_html = _winrate_bar(hmm_wr)
    ms_wr_html = _winrate_bar(ms_wr) if ms_wr is not None else '<span class="wr-na">N/A</span>'

    delta_long = round(ms_score_long - hmm_score_long, 1)
    delta_short = round(ms_score_short - hmm_score_short, 1)

    wr_winner = "HMM" if (hmm_wr or 0) > (ms_wr or 0) else ("MS" if (ms_wr or 0) > (hmm_wr or 0) else "EMPATE")
    wr_color = "#3fb950" if wr_winner == "HMM" else ("#58a6ff" if wr_winner == "MS" else "#8b949e")

    # Determine agreement type for banner styling
    agree_type = "yes" if "CONFIRMADO" in agreement else ("no" if "DIVERGENCIA" in agreement else "neutral")

    rows = []

    # ── SECTION HEADER ──
    rows.append(f'<div class="ms-section">')
    rows.append(f'  <div class="ms-header">')
    rows.append(f'    <div class="ms-header-left">')
    rows.append(f'      <span class="ms-header-icon">\U0001f916</span>')
    rows.append(f'      <div>')
    rows.append(f'        <div class="ms-header-title">Comparacion de Modelos</div>')
    rows.append(f'        <div class="ms-header-sub">HMM Hidden Markov Model vs Markov Switching &middot; {timeframe}</div>')
    rows.append(f'      </div>')
    rows.append(f'    </div>')
    rows.append(f'    <div class="ms-header-right">{agreement}</div>')
    rows.append(f'  </div>')

    # ── AGREEMENT BANNER ──
    if agree_type == "yes":
        banner_bg = "rgba(63,185,80,0.1)"
        banner_border = "#3fb950"
        banner_icon = "\U0001f44f"
        banner_msg = "Ambos modelos coinciden en la misma direccion"
    elif agree_type == "no":
        banner_bg = "rgba(248,81,73,0.1)"
        banner_border = "#f85149"
        banner_icon = "\u26a0\ufe0f"
        banner_msg = "Los modelos muestran senales opuestas - precaucion"
    else:
        banner_bg = "rgba(139,148,158,0.1)"
        banner_border = "#8b949e"
        banner_icon = "\u2139\ufe0f"
        banner_msg = "Ambos modelos muestran senal neutral"

    rows.append(f'  <div class="agree-banner" style="background:{banner_bg};border-color:{banner_border};">')
    rows.append(f'    <span class="agree-banner-icon">{banner_icon}</span>')
    rows.append(f'    <div>')
    rows.append(f'      <div class="agree-banner-title">{agreement}</div>')
    rows.append(f'      <div class="agree-banner-msg">{banner_msg}</div>')
    rows.append(f'    </div>')
    rows.append(f'  </div>')

    # ── TWO-COLUMN MODEL CARDS ──
    rows.append(f'  <div class="model-grid">')

    # HMM Card
    rows.append(f'    <div class="model-card model-hmm">')
    rows.append(f'      <div class="model-card-header">')
    rows.append(f'        <span class="model-icon">\U0001f9e0</span>')
    rows.append(f'        <span class="model-name">Hidden Markov Model</span>')
    rows.append(f'        <span class="model-badge-hmm">HMM</span>')
    rows.append(f'      </div>')
    rows.append(f'      <div class="model-signal-row">{hmm_badge} <span class="model-strength">{hmm_strength:.0f}% fuerza</span></div>')
    rows.append(f'      <div class="model-scores">')
    rows.append(f'        <div class="score-row">')
    rows.append(f'          <span class="score-label">Score LONG</span>')
    rows.append(f'          <span class="score-val" style="color:#3fb950;">{hmm_score_long:.0f}</span>')
    rows.append(f'        </div>')
    rows.append(f'        <div class="score-row">')
    rows.append(f'          <span class="score-label">Score SHORT</span>')
    rows.append(f'          <span class="score-val" style="color:#f85149;">{hmm_score_short:.0f}</span>')
    rows.append(f'        </div>')
    rows.append(f'      </div>')
    rows.append(f'      <div class="model-divider"></div>')
    rows.append(f'      <div class="model-metrics">')
    rows.append(f'        <div class="metric-item">')
    rows.append(f'          <span class="metric-label">Win Rate</span>')
    rows.append(f'          <span class="metric-val">{hmm_wr_html}</span>')
    rows.append(f'        </div>')
    rows.append(f'        <div class="metric-item">')
    rows.append(f'          <span class="metric-label">Senales</span>')
    rows.append(f'          <span class="metric-val">{hmm_total}</span>')
    rows.append(f'        </div>')
    rows.append(f'        <div class="metric-item">')
    rows.append(f'          <span class="metric-label">Estados</span>')
    rows.append(f'          <span class="metric-val">{hmm_n_states}</span>')
    rows.append(f'        </div>')
    rows.append(f'        <div class="metric-item">')
    rows.append(f'          <span class="metric-label">Regimen</span>')
    rows.append(f'          <span class="metric-val regimen-tag">{hmm_state_desc}</span>')
    rows.append(f'        </div>')
    rows.append(f'        <div class="metric-item">')
    rows.append(f'          <span class="metric-label">Cambios</span>')
    rows.append(f'          <span class="metric-val">{hmm_changes}</span>')
    rows.append(f'        </div>')
    rows.append(f'      </div>')
    rows.append(f'    </div>')

    # MS Card
    if data.ms_signal_info:
        rows.append(f'    <div class="model-card model-ms">')
        rows.append(f'      <div class="model-card-header">')
        rows.append(f'        <span class="model-icon">\U0001f52e</span>')
        rows.append(f'        <span class="model-name">Markov Switching</span>')
        rows.append(f'        <span class="model-badge-ms">MS</span>')
        rows.append(f'      </div>')
        rows.append(f'      <div class="model-signal-row">{ms_badge} <span class="model-strength">{ms_strength:.0f}% fuerza</span></div>')
        rows.append(f'      <div class="model-scores">')
        rows.append(f'        <div class="score-row">')
        rows.append(f'          <span class="score-label">Score LONG</span>')
        rows.append(f'          <span class="score-val" style="color:#3fb950;">{ms_score_long:.0f}</span>')
        rows.append(f'        </div>')
        rows.append(f'        <div class="score-row">')
        rows.append(f'          <span class="score-label">Score SHORT</span>')
        rows.append(f'          <span class="score-val" style="color:#f85149;">{ms_score_short:.0f}</span>')
        rows.append(f'        </div>')
        rows.append(f'      </div>')
        rows.append(f'      <div class="model-divider"></div>')
        rows.append(f'      <div class="model-metrics">')
        rows.append(f'        <div class="metric-item">')
        rows.append(f'          <span class="metric-label">Win Rate</span>')
        rows.append(f'          <span class="metric-val">{ms_wr_html}</span>')
        rows.append(f'        </div>')
        rows.append(f'        <div class="metric-item">')
        rows.append(f'          <span class="metric-label">Senales</span>')
        rows.append(f'          <span class="metric-val">{ms_total}</span>')
        rows.append(f'        </div>')
        rows.append(f'        <div class="metric-item">')
        rows.append(f'          <span class="metric-label">Estados</span>')
        rows.append(f'          <span class="metric-val">{ms_n_states}</span>')
        rows.append(f'        </div>')
        rows.append(f'        <div class="metric-item">')
        rows.append(f'          <span class="metric-label">Regimen</span>')
        rows.append(f'          <span class="metric-val regimen-tag">{ms_state_desc}</span>')
        rows.append(f'        </div>')
        rows.append(f'        <div class="metric-item">')
        rows.append(f'          <span class="metric-label">Cambios</span>')
        rows.append(f'          <span class="metric-val">{ms_changes}</span>')
        rows.append(f'        </div>')
        rows.append(f'      </div>')
        rows.append(f'    </div>')
    else:
        rows.append(f'    <div class="model-card model-ms model-na">')
        rows.append(f'      <div class="model-card-header">')
        rows.append(f'        <span class="model-icon">\U0001f52e</span>')
        rows.append(f'        <span class="model-name">Markov Switching</span>')
        rows.append(f'        <span class="model-badge-ms">MS</span>')
        rows.append(f'      </div>')
        rows.append(f'      <div class="model-na-msg">MS no disponible para este timeframe</div>')
        rows.append(f'    </div>')

    rows.append(f'  </div>')

    # ── DELTA SCORES ROW ──
    rows.append(f'  <div class="delta-row">')
    rows.append(f'    {_delta_badge(delta_long, "Diferencia MS vs HMM (LONG)")}')
    rows.append(f'    {_delta_badge(delta_short, "Diferencia MS vs HMM (SHORT)")}')
    rows.append(f'  </div>')

    # ── VERDICT ──
    rows.append(f'  <div class="verdict-bar">')
    rows.append(f'    <span class="verdict-icon">\U0001f3c6</span>')
    rows.append(f'    <div class="verdict-body">')
    rows.append(f'      <span class="verdict-title">Mejor Win Rate: <span style="color:{wr_color};font-weight:700;">{wr_winner}</span></span>')
    rows.append(f'      <span class="verdict-sub">HMM={hmm_signal} ({hmm_wr:.1f}%) vs MS={ms_signal} ({ms_wr:.1f}%)</span>')
    rows.append(f'    </div>')
    rows.append(f'  </div>')

    rows.append(f'</div>')

    return "\n".join(rows)


def _generate_overall_summary(results: Dict[str, H.TimeframeData]) -> str:
    """Resumen global con estilo moderno."""
    rows_html = ""
    acuerdos = 0
    divergencias = 0
    total_tf = 0

    for tf in results:
        total_tf += 1
        data = results[tf]
        hmm_sig = data.signal_info.get("signal", "N/A") if data.signal_info else "N/A"
        ms_sig = data.ms_signal_info.get("signal", "N/A") if data.ms_signal_info else "N/A"
        hmm_wr = data.verification.get("overall_win_rate", None) if data.verification else None
        ms_wr = data.ms_verification.get("overall_win_rate", None) if data.ms_verification else None

        if hmm_sig in ("LONG", "SHORT") and ms_sig in ("LONG", "SHORT"):
            if hmm_sig == ms_sig:
                acuerdos += 1
            else:
                divergencias += 1

        if hmm_sig == ms_sig:
            agree_icon = '<span class="agree-icon-sm agree-yes-sm">\u2714</span>' if hmm_sig in ("LONG", "SHORT") else '<span class="agree-icon-sm agree-nt-sm">\u2795</span>'
        else:
            agree_icon = '<span class="agree-icon-sm agree-no-sm">\u26a0</span>'

        hmm_wr_str = f"{hmm_wr:.1f}%" if hmm_wr is not None else "N/A"
        ms_wr_str = f"{ms_wr:.1f}%" if ms_wr is not None else "N/A"
        wr_icon = '\U0001f9e0' if (hmm_wr or 0) > (ms_wr or 0) + 2 else (
            '\U0001f52e' if (ms_wr or 0) > (hmm_wr or 0) + 2 else '\u2795')

        rows_html += f"""
        <tr>
            <td class="sum-tf">{tf}</td>
            <td class="sum-sig">{_signal_badge(hmm_sig)}</td>
            <td class="sum-sig">{_signal_badge(ms_sig)}</td>
            <td class="sum-agree">{agree_icon}</td>
            <td class="sum-wr">{hmm_wr_str}</td>
            <td class="sum-wr">{ms_wr_str}</td>
            <td class="sum-winner">{wr_icon}</td>
        </tr>"""

    all_hmm_wrs = []
    all_ms_wrs = []
    for tf, data in results.items():
        if data.verification:
            wr = data.verification.get("overall_win_rate", 0)
            if wr:
                all_hmm_wrs.append(wr)
        if data.ms_verification:
            wr = data.ms_verification.get("overall_win_rate", 0)
            if wr:
                all_ms_wrs.append(wr)
    best_model = "Datos insuficientes"
    if all_hmm_wrs and all_ms_wrs:
        avg_hmm = float(np.mean(all_hmm_wrs))
        avg_ms = float(np.mean(all_ms_wrs))
        if avg_hmm > avg_ms + 2:
            best_model = f"HMM lidera ({avg_hmm:.1f}% vs {avg_ms:.1f}%)"
        elif avg_ms > avg_hmm + 2:
            best_model = f"MS lidera ({avg_ms:.1f}% vs {avg_hmm:.1f}%)"
        else:
            best_model = f"Similar (HMM={avg_hmm:.1f}%, MS={avg_ms:.1f}%)"

    if not rows_html:
        return ""

    html = f"""
<div class="sum-section">
    <div class="sum-header">
        <span class="sum-header-icon">\U0001f4ca</span>
        <div>
            <div class="sum-header-title">Resumen Global</div>
            <div class="sum-header-sub">Comparacion agregada HMM vs Markov Switching</div>
        </div>
    </div>
    <div class="sum-cards">
        <div class="sum-card sum-card-agree">
            <div class="sum-card-num" style="color:#3fb950;">{acuerdos}</div>
            <div class="sum-card-label">Timeframes en acuerdo</div>
            <div class="sum-card-bar">
                <div class="sum-card-fill" style="width:{acuerdos/max(total_tf,1)*100:.0f}%;background:#3fb950;"></div>
            </div>
        </div>
        <div class="sum-card sum-card-diverge">
            <div class="sum-card-num" style="color:#f85149;">{divergencias}</div>
            <div class="sum-card-label">Timeframes en divergencia</div>
            <div class="sum-card-bar">
                <div class="sum-card-fill" style="width:{divergencias/max(total_tf,1)*100:.0f}%;background:#f85149;"></div>
            </div>
        </div>
        <div class="sum-card sum-card-winner">
            <div style="font-size:0.75rem;color:#8b949e;margin-bottom:4px;">Mejor WR promedio</div>
            <div style="font-size:1.2rem;font-weight:700;color:#58a6ff;">{best_model}</div>
            <div style="font-size:0.7rem;color:#8b949e;margin-top:6px;">Diferencia {'>' if all_hmm_wrs and all_ms_wrs else '-'}2% para definir ganador</div>
        </div>
    </div>
    <table class="sum-table">
        <thead>
            <tr>
                <th>TF</th>
                <th>HMM</th>
                <th>MS</th>
                <th>Acuerdo</th>
                <th>WR HMM</th>
                <th>WR MS</th>
                <th>Ganador</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>"""
    return html


def build_enhanced_dashboard(results: Dict[str, H.TimeframeData], asset: str) -> str:
    """Construye el dashboard HTML completo con pestanas por timeframe."""
    tf_keys = list(results.keys())

    tabs_buttons = ""
    tabs_content = ""
    for idx, tf in enumerate(tf_keys):
        data = results[tf]
        try:
            inner = H._generate_tf_inner(data, asset, tf)
            ms_html = _generate_ms_comparison_html(data, tf)
            active_class = " active" if idx == 0 else ""
            tabs_buttons += f'<button class="tf-tab{active_class}" data-tf="{tf}" onclick="switchTF(\'{tf}\')">{tf.upper()}</button>'
            tabs_content += f'<div class="tf-content{active_class}" id="tf-{tf}">{inner}{ms_html}</div>'
        except Exception as e:
            print(f"  Error panel {tf}: {e}")
            import traceback
            traceback.print_exc()
            continue

    overall = _generate_overall_summary(results)

    if not tabs_content:
        return "<html><body><h2>Error: Sin datos</h2></body></html>"

    tf_panels = f'<div class="tf-tabs">{tabs_buttons}</div>\n{tabs_content}\n{overall if overall else ""}'

    CSS = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradingLatino - HMM vs Markov Switching Dashboard</title>
<style>
/* ===== RESET & BASE ===== */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,sans-serif;
    background:#0d1117;color:#f0f6fc;line-height:1.6;padding:24px;
}
.container{max-width:1400px;margin:0 auto;}
a{color:#58a6ff;text-decoration:none;}

/* ===== HEADER ===== */
.gheader{
    background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
    border:1px solid #30363d;border-radius:12px;padding:24px 28px;
    margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;
}
.gheader h1{font-size:1.4rem;font-weight:700;background:linear-gradient(135deg,#3fb950,#58a6ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.gheader .gsub{font-size:0.8rem;color:#8b949e;margin-top:2px;}
.gheader .gright{text-align:right;font-size:0.8rem;color:#8b949e;}
.gheader .gright .gasset{font-weight:600;color:#f0f6fc;}

/* ===== SIGNAL BADGES ===== */
.sig-badge{display:inline-flex;align-items:center;gap:4px;padding:3px 12px;border-radius:6px;font-weight:700;font-size:0.8rem;letter-spacing:0.3px;}
.sig-long{background:rgba(63,185,80,0.15);color:#3fb950;border:1px solid rgba(63,185,80,0.3);}
.sig-short{background:rgba(248,81,73,0.15);color:#f85149;border:1px solid rgba(248,81,73,0.3);}
.sig-flat{background:rgba(139,148,158,0.15);color:#8b949e;border:1px solid rgba(139,148,158,0.3);}
.sig-na{background:rgba(139,148,158,0.1);color:#8b949e;border:1px dashed #30363d;}

/* ===== AGREEMENT BADGE ===== */
.agree-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:8px;font-weight:600;font-size:0.85rem;white-space:nowrap;}
.agree-icon{font-size:1rem;}
.agree-yes{background:rgba(63,185,80,0.12);color:#3fb950;border:1px solid rgba(63,185,80,0.3);}
.agree-no{background:rgba(248,81,73,0.12);color:#f85149;border:1px solid rgba(248,81,73,0.3);}
.agree-neutral{background:rgba(139,148,158,0.12);color:#8b949e;border:1px solid rgba(139,148,158,0.3);}

/* ===== WIN RATE BAR ===== */
.wr-track{display:inline-block;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;vertical-align:middle;overflow:hidden;}
.wr-fill{height:100%;border-radius:3px;transition:width 0.5s ease;}
.wr-label{font-weight:600;font-size:0.8rem;vertical-align:middle;margin-left:6px;}
.wr-na{color:#8b949e;font-size:0.8rem;}

/* ===== MS COMPARISON SECTION ===== */
.ms-section{
    background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:24px;margin-bottom:24px;margin-top:24px;
}
.ms-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;}
.ms-header-left{display:flex;align-items:center;gap:12px;}
.ms-header-icon{font-size:1.5rem;}
.ms-header-title{font-size:1.15rem;font-weight:700;color:#f0f6fc;}
.ms-header-sub{font-size:0.8rem;color:#8b949e;margin-top:2px;}

/* ===== AGREEMENT BANNER ===== */
.agree-banner{display:flex;align-items:center;gap:12px;padding:14px 18px;border-radius:10px;border:1px solid;margin-bottom:20px;}
.agree-banner-icon{font-size:1.5rem;}
.agree-banner-title{font-weight:600;font-size:0.95rem;margin-bottom:2px;}
.agree-banner-msg{font-size:0.8rem;opacity:0.8;}

/* ===== MODEL GRID ===== */
.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}

.model-card{
    border-radius:10px;padding:20px;position:relative;
    transition:transform 0.2s ease,box-shadow 0.2s ease;
}
.model-card:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(0,0,0,0.3);}
.model-hmm{background:linear-gradient(145deg,rgba(63,185,80,0.06),rgba(63,185,80,0.02));border:1px solid rgba(63,185,80,0.2);}
.model-ms{background:linear-gradient(145deg,rgba(88,166,255,0.06),rgba(88,166,255,0.02));border:1px solid rgba(88,166,255,0.2);}
.model-na{background:rgba(139,148,158,0.03);border:1px dashed #30363d;}

.model-card-header{display:flex;align-items:center;gap:8px;margin-bottom:14px;}
.model-icon{font-size:1.3rem;}
.model-name{font-weight:600;font-size:0.95rem;flex:1;}
.model-badge-hmm{font-size:0.65rem;font-weight:700;background:rgba(63,185,80,0.2);color:#3fb950;padding:2px 8px;border-radius:4px;letter-spacing:0.5px;}
.model-badge-ms{font-size:0.65rem;font-weight:700;background:rgba(88,166,255,0.2);color:#58a6ff;padding:2px 8px;border-radius:4px;letter-spacing:0.5px;}

.model-signal-row{display:flex;align-items:center;gap:12px;margin-bottom:14px;}
.model-strength{font-size:0.8rem;color:#8b949e;}

.model-scores{display:flex;flex-direction:column;gap:6px;margin-bottom:14px;}
.score-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;}
.score-label{font-size:0.8rem;color:#8b949e;}
.score-val{font-weight:700;font-size:1.1rem;}

.model-divider{height:1px;background:rgba(255,255,255,0.06);margin:12px 0;}

.model-metrics{display:flex;flex-direction:column;gap:8px;}
.metric-item{display:flex;justify-content:space-between;align-items:center;}
.metric-label{font-size:0.8rem;color:#8b949e;}
.metric-val{font-weight:600;font-size:0.85rem;}

.regimen-tag{font-size:0.75rem !important;color:#d29922 !important;}

.model-na-msg{text-align:center;padding:40px 20px;color:#8b949e;font-size:0.9rem;}

/* ===== DELTA ROW ===== */
.delta-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;}
.delta-card{
    background:rgba(255,255,255,0.02);border-radius:8px;padding:14px 16px;
    border:1px solid #30363d;transition:background 0.2s;
}
.delta-card:hover{background:rgba(255,255,255,0.04);}
.delta-label{font-size:0.75rem;color:#8b949e;margin-bottom:4px;}
.delta-val{font-size:1.2rem;font-weight:700;}
.delta-sub{font-size:0.7rem;color:#8b949e;margin-top:2px;}

/* ===== VERDICT ===== */
.verdict-bar{
    display:flex;align-items:center;gap:14px;
    background:rgba(255,255,255,0.02);border:1px solid #30363d;
    border-radius:10px;padding:16px 20px;
}
.verdict-icon{font-size:1.5rem;}
.verdict-body{display:flex;flex-direction:column;gap:2px;}
.verdict-title{font-size:0.95rem;color:#f0f6fc;}
.verdict-sub{font-size:0.8rem;color:#8b949e;}

/* ===== SUMMARY SECTION ===== */
.sum-section{
    background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:24px;margin-bottom:24px;margin-top:30px;
}
.sum-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;}
.sum-header-icon{font-size:1.5rem;}
.sum-header-title{font-size:1.15rem;font-weight:700;color:#f0f6fc;}
.sum-header-sub{font-size:0.8rem;color:#8b949e;margin-top:2px;}

.sum-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px;}
.sum-card{
    border-radius:10px;padding:18px;text-align:center;
    border:1px solid #30363d;
}
.sum-card-agree{background:rgba(63,185,80,0.06);border-color:rgba(63,185,80,0.2);}
.sum-card-diverge{background:rgba(248,81,73,0.06);border-color:rgba(248,81,73,0.2);}
.sum-card-winner{background:rgba(88,166,255,0.06);border-color:rgba(88,166,255,0.2);}
.sum-card-num{font-size:2rem;font-weight:800;margin-bottom:4px;}
.sum-card-label{font-size:0.8rem;color:#8b949e;margin-bottom:10px;}
.sum-card-bar{height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;}
.sum-card-fill{height:100%;border-radius:2px;transition:width 0.8s ease;}

/* ===== SUMMARY TABLE ===== */
.sum-table{width:100%;border-collapse:collapse;font-size:0.9rem;}
.sum-table thead th{
    padding:10px 12px;text-align:left;font-size:0.8rem;color:#8b949e;
    border-bottom:2px solid #30363d;text-transform:uppercase;letter-spacing:0.5px;
}
.sum-table tbody tr{
    border-bottom:1px solid rgba(255,255,255,0.04);
    transition:background 0.15s;
}
.sum-table tbody tr:hover{background:rgba(255,255,255,0.03);}
.sum-table tbody td{padding:12px;vertical-align:middle;}
.sum-tf{font-weight:700;color:#f0f6fc;}
.sum-sig{text-align:center;}
.sum-agree{text-align:center;font-size:1.2rem;}
.sum-wr{text-align:right;font-weight:600;font-size:0.85rem;}
.sum-winner{text-align:center;font-size:1.2rem;}

.agree-icon-sm{font-size:1.1rem;}
.agree-yes-sm{color:#3fb950;}
.agree-no-sm{color:#f85149;}
.agree-nt-sm{color:#8b949e;}

/* ===== RESPONSIVE ===== */
@media(max-width:900px){
    body{padding:12px;}
    .model-grid{grid-template-columns:1fr;}
    .sum-cards{grid-template-columns:1fr;}
    .gheader{flex-direction:column;gap:12px;align-items:flex-start;}
    .gheader .gright{text-align:left;width:100%;}
    .ms-header{flex-direction:column;gap:10px;}
}
@media(max-width:600px){
    .delta-row{grid-template-columns:1fr;}
    .sum-table{font-size:0.75rem;}
    .sum-table thead th,.sum-table tbody td{padding:8px 6px;}
}

/* ===== TIMEFRAME TABS ===== */
.tf-tabs{display:flex;gap:4px;margin-bottom:24px;flex-wrap:wrap;border-bottom:1px solid #30363d;padding:0 0 0 0;}
.tf-tab{
    background:transparent;color:#8b949e;border:none;
    padding:10px 20px;font-size:0.85rem;font-weight:600;
    cursor:pointer;border-radius:8px 8px 0 0;
    transition:all 0.2s ease;position:relative;
    font-family:inherit;letter-spacing:0.3px;
}
.tf-tab:hover{background:rgba(255,255,255,0.04);color:#f0f6fc;}
.tf-tab.active{
    background:rgba(88,166,255,0.1);color:#58a6ff;
    border-bottom:2px solid #58a6ff;
}
.tf-content{display:none;animation:fadeIn 0.3s ease;}
.tf-content.active{display:block;}

/* ===== ANIMATIONS ===== */
.ms-section,.sum-section{animation:fadeIn 0.4s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}

/* ===== LEGACY: HMM PANELS (from _generate_tf_inner) ===== */
.section{
    background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:20px;margin-bottom:24px;
}
.section-title{
    font-size:1.1rem;font-weight:600;color:#f0f6fc;
    margin-bottom:16px;padding-bottom:8px;
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.cards-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:20px;}
.card{
    background:rgba(255,255,255,0.02);border:1px solid #30363d;
    border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:8px;
}
.card-title{font-size:0.8rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;}
.price-value{font-size:1.6rem;font-weight:700;color:#f0f6fc;}
.price-change{font-size:0.85rem;font-weight:600;}
.price-date{font-size:0.75rem;color:#8b949e;}

.regime-indicator{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:6px;}
.regime-number{font-size:1.3rem;font-weight:700;color:#f0f6fc;vertical-align:middle;}
.regime-desc{font-size:0.9rem;color:#d29922;font-weight:600;}
.regime-stats{font-size:0.75rem;color:#8b949e;display:flex;gap:12px;}

.signal-badge-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.expiration-badge{font-size:0.75rem;font-weight:600;}

.signal-badge{display:inline-flex;align-items:center;gap:4px;padding:4px 14px;border-radius:6px;font-weight:700;font-size:0.85rem;}
.signal-long{background:rgba(8,153,129,0.2);color:#3fb950;}
.signal-short{background:rgba(242,54,69,0.2);color:#f85149;}
.signal-flat{background:rgba(136,136,136,0.15);color:#8b949e;}

.regime-alignment-badge{
    display:flex;align-items:center;gap:6px;
    padding:6px 10px;border-radius:6px;font-size:0.75rem;
    border:1px solid #30363d;margin-bottom:8px;
}
.alignment-neutral{background:rgba(210,153,34,0.08);border-color:rgba(210,153,34,0.2);}
.alignment-icon{font-size:0.9rem;}
.alignment-text{font-weight:600;color:#d29922;}
.alignment-sub{font-size:0.7rem;color:#8b949e;margin-left:auto;}

.signal-strength-bar{height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;margin-bottom:4px;}
.signal-strength-fill{height:100%;border-radius:3px;transition:width 0.5s ease;}
.signal-label{font-size:0.75rem;color:#8b949e;}

.signal-start-box{display:flex;align-items:center;gap:8px;padding:6px 0;}
.signal-start-icon{font-size:1rem;}
.signal-start-info{display:flex;flex-direction:column;}
.signal-start-label{font-size:0.7rem;color:#8b949e;}
.signal-start-date{font-size:0.85rem;font-weight:600;color:#f0f6fc;}
.signal-start-bars{font-size:0.7rem;color:#8b949e;}

.signal-score-box{display:flex;align-items:center;gap:4px;padding:6px 0;}
.signal-score-label{font-size:0.75rem;color:#8b949e;margin-right:auto;}
.signal-score-value{font-size:1.2rem;font-weight:800;}
.signal-score-sep{color:#8b949e;font-size:0.9rem;}
.signal-score-threshold{color:#8b949e;font-size:0.9rem;font-weight:600;}
.signal-score-check{font-size:1rem;margin-left:4px;}

.signal-filter-badge{display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#8b949e;padding:4px 0;}
.filter-icon{font-size:0.85rem;}
.filter-text{font-weight:500;}

.expiration-container{background:rgba(255,255,255,0.02);border:1px solid #30363d;border-radius:8px;padding:12px 14px;}
.expiration-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.expiration-label{font-size:0.8rem;color:#8b949e;}
.expiration-window{font-size:0.8rem;font-weight:600;color:#58a6ff;}
.expiration-bar-track{height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;margin-bottom:6px;}
.expiration-bar-fill{height:100%;border-radius:2px;transition:width 0.5s ease;}
.expiration-details{display:flex;justify-content:space-between;font-size:0.75rem;color:#8b949e;}

.signal-params{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;}
.signal-params span{
    font-size:0.7rem;padding:2px 8px;border-radius:4px;
    background:rgba(255,255,255,0.04);color:#8b949e;
    border:1px solid rgba(255,255,255,0.06);
}

.chart-container{width:100%;}
</style>
</head>
<body>
<div class="container">
    <div class="gheader">
        <div>
            <h1>\U0001f9e0 TradingLatino &mdash; HMM vs \U0001f52e Markov Switching</h1>
            <div class="gsub">
                Dashboard comparativo de modelos de regimenes de mercado
                &mdash; HMM Hidden Markov Model | MS Markov Switching
            </div>
        </div>
        <div class="gright">
            <div class="gasset">{asset}</div>
            <div id="update-time"></div>
        </div>
    </div>
    {tf_panels}
</div>
<script>
function switchTF(tf){
    document.querySelectorAll('.tf-tab').forEach(function(t){t.classList.remove('active');});
    document.querySelectorAll('.tf-content').forEach(function(c){c.classList.remove('active');});
    var tab = document.querySelector('.tf-tab[onclick*="'+tf+'"]');
    if(tab) tab.classList.add('active');
    var content = document.getElementById('tf-'+tf);
    if(content) content.classList.add('active');
}
document.getElementById('update-time').textContent='Actualizado: '+new Date().toLocaleString('es-ES',{timeZone:'UTC'})+' UTC';
</script>
</body>
</html>
"""

    css = CSS.replace("{asset}", asset).replace("{tf_panels}", tf_panels)
    return css


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t_start = time.time()
    results = run_pipeline(asset=ASSET, timeframes=TIMEFRAMES)

    print(f"\n{'='*60}")
    print("Generando dashboard HMM vs MS...")
    t0 = time.time()
    H.ASSET = ASSET

    html_content = build_enhanced_dashboard(results, ASSET)
    print(f"  Dashboard generado ({time.time()-t0:.1f}s)")

    output_path = Path(OUTPUT_HTML)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"  Guardado en: {output_path.resolve()}")

    # Divergencias
    for tf, data in results.items():
        if data.signal_info and data.ms_signal_info:
            hmm_s = data.signal_info.get("signal", "N/A")
            ms_s = data.ms_signal_info.get("signal", "N/A")
            if hmm_s in ("LONG", "SHORT") and ms_s in ("LONG", "SHORT") and hmm_s != ms_s:
                print(f"  [DIVERGENCIA] {tf}: HMM={hmm_s} vs MS={ms_s}")

    if OPEN_BROWSER:
        print("Abriendo navegador...")
        webbrowser.open(str(output_path.resolve()))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Completado en {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
