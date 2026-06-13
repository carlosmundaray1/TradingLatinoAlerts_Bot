#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DASHBOARD HTML · TradingLatino HMM Regime Dashboard
================================================================================
Generación del dashboard HTML con gráficos Plotly, health meter y más.
================================================================================
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradinglatino_hmm.config import (
    ADX_THRESHOLD, REGIME_COLORS, SIGNAL_LABELS,
    POPULAR_ASSETS, RELEASE_LOOKBACK,
)
from tradinglatino_hmm.hmm_model import _classify_regime_bias
from tradinglatino_hmm.signals import _format_date


@dataclass
class TimeframeData:
    """Resultados completos para una temporalidad."""
    df_full: pd.DataFrame
    states_full: np.ndarray
    state_summary: pd.DataFrame
    trans_mat: Optional[np.ndarray]
    signal_info: Dict[str, Any]
    regime_changes: List[Dict]
    verification: Optional[Dict[str, Any]] = None
    trailing_verification: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────


def _current_regime_index(states: np.ndarray) -> int:
    """Devuelve el régimen de la última vela."""
    return int(states[-1]) if len(states) > 0 else -1


def _detect_regime_changes(states: np.ndarray, index, state_summary: pd.DataFrame, max_alerts: int = 15):
    """Detecta cambios de régimen y devuelve los más recientes."""
    desc_map: Dict[int, str] = {}
    for _, r in state_summary.iterrows():
        desc_map[int(r["state"])] = r["description"]
    changes = []
    prev_state = states[0]
    change_start = 0
    for i in range(1, len(states)):
        if states[i] != prev_state:
            if prev_state >= 0 and states[i] >= 0:
                changes.append({
                    "from_state": int(prev_state),
                    "to_state": int(states[i]),
                    "from_desc": desc_map.get(int(prev_state), f"R{prev_state}"),
                    "to_desc": desc_map.get(int(states[i]), f"R{states[i]}"),
                    "date": _format_date(index[i]),
                    "duration_velas": i - change_start,
                    "from_color": REGIME_COLORS[int(prev_state) % len(REGIME_COLORS)],
                    "to_color": REGIME_COLORS[int(states[i]) % len(REGIME_COLORS)],
                })
            prev_state = states[i]
            change_start = i
    return list(reversed(changes[-max_alerts:]))


def _build_change_summary(regime_changes: Optional[List[Dict]], state_summary: pd.DataFrame) -> str:
    """Genera un resumen visual de estadisticas agregadas de cambios de regimen."""
    if not regime_changes or len(regime_changes) == 0:
        return ""
    total_cambios = len(regime_changes)
    duraciones = [c["duration_velas"] for c in regime_changes]
    avg_dur = float(np.mean(duraciones)) if duraciones else 0.0
    if not state_summary.empty:
        max_dur_row = state_summary.loc[state_summary["mean_duration_bars"].idxmax()]
        most_persistent = max_dur_row["description"]
        most_persistent_dur = max_dur_row["mean_duration_bars"]
        most_persistent_state = int(max_dur_row["state"])
        most_persistent_color = REGIME_COLORS[most_persistent_state % len(REGIME_COLORS)]
    else:
        most_persistent = "-"
        most_persistent_dur = 0
        most_persistent_color = "#888"
    ultimo = regime_changes[0]
    ultimo_date = _format_date(ultimo["date"])
    ultimo_from = ultimo["from_desc"]
    ultimo_to = ultimo["to_desc"]
    ultimo_from_color = ultimo["from_color"]
    ultimo_to_color = ultimo["to_color"]
    html = f"""
    <div class="change-summary">
        <div class="summary-stat">
            <span class="summary-stat-value">{total_cambios}</span>
            <span class="summary-stat-label">Cambios detectados</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value" style="color:{most_persistent_color}">{most_persistent}</span>
            <span class="summary-stat-label">Mas persistente ({most_persistent_dur:.0f}v promedio)</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value">{avg_dur:.0f}</span>
            <span class="summary-stat-label">Duracion promedio (velas)</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value" style="font-size:0.75rem;">
                <span style="color:{ultimo_from_color}">{ultimo_from}</span>
                <span style="color:#666;margin:0 4px;">→</span>
                <span style="color:{ultimo_to_color}">{ultimo_to}</span>
            </span>
            <span class="summary-stat-label">Ultimo cambio ({ultimo_date})</span>
        </div>
    </div>"""
    return html


def _regime_alignment_badge(alignment: str, signal: str) -> str:
    """Genera un badge HTML para alineamiento de regimen."""
    if signal not in ("LONG", "SHORT") or alignment == "no_signal":
        return ""
    if alignment == "favorable":
        return (
            f'<div class="regime-alignment-badge alignment-favorable">'
            f'<span class="alignment-icon">✅</span>'
            f'<span class="alignment-text">Regimen FAVORABLE para {signal}</span>'
            f'<span class="alignment-sub">El mercado esta alineado con tu trade — Mantener</span>'
            f'</div>'
        )
    elif alignment == "adverse":
        return (
            f'<div class="regime-alignment-badge alignment-adverse">'
            f'<span class="alignment-icon">🛑</span>'
            f'<span class="alignment-text">Regimen ADVERSO para {signal}</span>'
            f'<span class="alignment-sub">El mercado esta en contra de tu trade — Considerar salir</span>'
            f'</div>'
        )
    else:
        return (
            f'<div class="regime-alignment-badge alignment-neutral">'
            f'<span class="alignment-icon">⚠️</span>'
            f'<span class="alignment-text">Regimen NEUTRAL para {signal}</span>'
            f'<span class="alignment-sub">Mercado sin direccion clara — Gestionar riesgo</span>'
            f'</div>'
        )


def _calculate_regime_entropy(trans_mat: np.ndarray, state: int) -> float:
    """Calcula la confianza del HMM basada en entropia normalizada."""
    if trans_mat is None or state < 0 or state >= len(trans_mat):
        return 50.0
    probs = trans_mat[state]
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 50.0
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(trans_mat))
    if max_entropy == 0:
        return 100.0
    normalized_entropy = entropy / max_entropy
    confidence = (1.0 - normalized_entropy) * 100.0
    return round(confidence, 1)


def _calculate_regime_duration_ratio(states: np.ndarray, current_state: int, state_summary: pd.DataFrame) -> dict:
    """Calcula cuantas velas lleva el regimen actual vs su duracion media."""
    if len(states) == 0 or current_state < 0:
        return {"current_duration": 0, "mean_duration": 1, "ratio": 0.0, "score": 50}
    count = 0
    for i in range(len(states) - 1, -1, -1):
        if states[i] == current_state:
            count += 1
        else:
            break
    mean_dur = 1.0
    if not state_summary.empty:
        row = state_summary[state_summary["state"] == current_state]
        if not row.empty:
            mean_dur = float(row.iloc[0]["mean_duration_bars"])
            if mean_dur <= 0:
                mean_dur = 1.0
    ratio = count / mean_dur if mean_dur > 0 else 1.0
    if ratio <= 0.5:
        score = 100
    elif ratio <= 1.0:
        score = 80
    elif ratio <= 1.3:
        score = 50
    else:
        score = 20
    return {"current_duration": count, "mean_duration": round(mean_dur, 1), "ratio": round(ratio, 2), "score": score}


def _calculate_impact_score(regime_warnings: List[Dict], signal: str) -> dict:
    """Calcula el score de impacto neto de los ultimos cambios de regimen."""
    if not regime_warnings or signal not in ("LONG", "SHORT"):
        return {"net_score": 0, "favorable": 0, "adverse": 0, "neutral": 0, "total": 0, "score": 50}
    favorable = sum(1 for w in regime_warnings if w["impact"] == "favorable")
    adverse = sum(1 for w in regime_warnings if w["impact"] == "adverse")
    neutral = sum(1 for w in regime_warnings if w["impact"] == "neutral")
    total = len(regime_warnings)
    net = favorable - adverse
    max_possible = total
    min_possible = -total
    if max_possible == min_possible:
        score = 50
    else:
        score = ((net - min_possible) / (max_possible - min_possible)) * 100.0
    return {"net_score": net, "favorable": favorable, "adverse": adverse, "neutral": neutral, "total": total, "score": round(score, 1)}


def _calculate_wr_by_regime(verification: Optional[Dict], current_state: int, signal: str) -> dict:
    """Calcula el win rate historico de senales en el regimen actual."""
    if not verification or current_state < 0:
        return {"win_rate": None, "signals": 0, "score": 50}
    side = signal if signal in ("LONG", "SHORT") else None
    if not side:
        return {"win_rate": None, "signals": 0, "score": 50}
    results = verification.get(side.lower(), [])
    regime_signals = [r for r in results if r.get("regime", -1) == current_state]
    if not regime_signals:
        all_signals = verification.get(side.lower(), [])
        if len(all_signals) > 0:
            wr = sum(1 for r in all_signals if r["won"]) / len(all_signals) * 100
            return {"win_rate": round(wr, 1), "signals": len(all_signals), "score": wr, "note": "todos los regimenes"}
    n = len(regime_signals)
    if n == 0:
        return {"win_rate": None, "signals": 0, "score": 50}
    wins = sum(1 for r in regime_signals if r["won"])
    wr = wins / n * 100
    confidence_mult = min(1.0, n / 10)
    score = wr * confidence_mult + 50 * (1 - confidence_mult)
    return {"win_rate": round(wr, 1), "signals": n, "score": round(score, 1)}


def _build_trade_health_meter(
    states: np.ndarray,
    state_summary: pd.DataFrame,
    regime_warnings: List[Dict],
    trans_mat: Optional[np.ndarray],
    current_state: int,
    signal: str,
    regime_alignment: str,
    signal_info: Dict[str, Any],
    verification: Optional[Dict[str, Any]],
    df_full: pd.DataFrame,
) -> str:
    """Genera el TRADE HEALTH METER."""
    stability = _calculate_regime_duration_ratio(states, current_state, state_summary)
    impact = _calculate_impact_score(regime_warnings, signal)
    wr_data = _calculate_wr_by_regime(verification, current_state, signal)
    confidence = _calculate_regime_entropy(trans_mat, current_state)

    health_score = (
        stability["score"] * 0.25
        + impact["score"] * 0.25
        + wr_data["score"] * 0.25
        + confidence * 0.25
    )
    health_score = round(min(100, max(0, health_score)), 1)

    if health_score >= 70 and regime_alignment != "adverse":
        verdict, verdict_icon, verdict_color = "MANTENER", "🟢", "#089981"
        verdict_desc = "El trade esta saludable. Todos los indicadores respaldan la posicion."
        action_text = "✅ Mantener trade — Regimen favorable con alta confianza"
        action_color = "#089981"
    elif health_score >= 45 and regime_alignment != "adverse":
        verdict, verdict_icon, verdict_color = "PRECAUCION", "🟡", "#2962FF"
        verdict_desc = "Senales mixtas. Monitorea de cerca y ajusta stops."
        action_text = "⚠️ Monitorear — Algunos indicadores muestran cautela"
        action_color = "#2962FF"
    else:
        verdict, verdict_icon, verdict_color = "SALIR / NO ENTRAR", "🔴", "#F23645"
        verdict_desc = "Condiciones adversas detectadas. Considera salir o no abrir posicion."
        action_text = "🔴 Considerar salir — Factores en contra del trade"
        action_color = "#F23645"

    if signal == "FLAT":
        if health_score >= 70:
            verdict, verdict_icon, verdict_color = "FAVORABLE", "🟢", "#089981"
            verdict_desc = "Mercado en condiciones favorables. Preparado para la proxima senal."
            action_text = "✅ Esperar senal — Condiciones de mercado favorables"
            action_color = "#089981"
        elif health_score >= 45:
            verdict, verdict_icon, verdict_color = "NEUTRAL", "🟡", "#2962FF"
            verdict_desc = "Mercado sin direccion clara. Esperar confirmacion."
            action_text = "⏳ Esperar confirmacion — Mercado neutral"
            action_color = "#2962FF"
        else:
            verdict, verdict_icon, verdict_color = "DESFAVORABLE", "🔴", "#F23645"
            verdict_desc = "Mercado en condiciones adversas. Evitar operar."
            action_text = "🚫 Evitar operar — Condiciones adversas del mercado"
            action_color = "#F23645"

    alerts_list = []
    if stability["score"] < 40:
        alerts_list.append(f"⏳ Regimen alargado: {stability['current_duration']}v (media {stability['mean_duration']}v) — posible cambio")
    if impact["adverse"] >= 2:
        alerts_list.append(f"📉 {impact['adverse']} cambios adversos en ultimos {impact['total']} cambios de regimen")
    if wr_data["win_rate"] is not None and wr_data["win_rate"] < 50 and wr_data["signals"] >= 3:
        alerts_list.append(f"📊 Win Rate bajo en este regimen: {wr_data['win_rate']:.0f}% ({wr_data['signals']} senales)")
    if confidence < 40:
        alerts_list.append(f"🔮 Confianza HMM baja ({confidence:.0f}%) — prediccion no fiable")
    if stability["score"] > 80 and impact["score"] > 70 and confidence > 70:
        alerts_list.append(f"✅ Todos los indicadores positivos — regimen estable y favorable")
    alerts_html = ""
    if alerts_list:
        alerts_html = '<div class="health-alerts">'
        for alert in alerts_list:
            alerts_html += f'<div class="health-alert-row">{alert}</div>'
        alerts_html += "</div>"

    bar_color = "#089981" if health_score >= 70 else ("#2962FF" if health_score >= 45 else "#F23645")
    stability_icon, stability_label = ("✅", "Estable") if stability["score"] >= 70 else (("⚠️", "Alargado") if stability["score"] >= 40 else ("🔴", "Critico"))
    impact_icon, impact_label = ("✅", f"+{impact['net_score']}") if impact["net_score"] > 0 else (("➖", "0") if impact["net_score"] == 0 else ("🔴", f"{impact['net_score']}"))
    if wr_data["win_rate"] is not None:
        wr_display = f"{wr_data['win_rate']:.0f}%"
        wr_icon = "🏆" if wr_data['win_rate'] >= 60 else ("📊" if wr_data['win_rate'] >= 40 else "⚠️")
    else:
        wr_display = "—"
        wr_icon = "📊"
    wr_note = f" ({wr_data['signals']} sig.)" if wr_data['signals'] > 0 else ""
    conf_icon, conf_label = "🔮", f"{confidence:.0f}%"

    html = f"""
    <div class="section" style="margin-top:20px;">
        <div class="section-title">🏥 TRADE HEALTH METER &mdash; ¿Entrar, Mantener o Salir?</div>
        <div class="health-meter-container">
            <div class="health-verdict" style="border-left:4px solid {verdict_color};">
                <div class="health-verdict-row">
                    <span class="health-verdict-icon">{verdict_icon}</span>
                    <div class="health-verdict-info">
                        <span class="health-verdict-label" style="color:{verdict_color};">{verdict}</span>
                        <span class="health-verdict-desc">{verdict_desc}</span>
                    </div>
                    <div class="health-score-ring" style="border-color:{bar_color};">
                        <span class="health-score-value" style="color:{bar_color};">{health_score:.0f}</span>
                        <span class="health-score-label">/100</span>
                    </div>
                </div>
            </div>
            <div class="health-bar-container">
                <div class="health-bar-track">
                    <div class="health-bar-fill" style="width:{health_score}%;background:{bar_color};"></div>
                </div>
                <div class="health-bar-labels">
                    <span style="color:#FF4136;">🔴 Salir</span>
                    <span style="color:#f39c12;">🟡 Precaución</span>
                    <span style="color:#2ECC40;">🟢 Mantener</span>
                </div>
            </div>
            <div class="health-action" style="border-left-color:{action_color};">
                <span class="health-action-label">🎯 Accion sugerida:</span>
                <span class="health-action-text" style="color:{action_color};">{action_text}</span>
            </div>
            <div class="health-meters-grid">
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if stability['score'] >= 70 else ('#f39c12' if stability['score'] >= 40 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">⏳</span>
                        <span class="health-meter-title">Estabilidad</span>
                    </div>
                    <div class="health-meter-value">{stability['current_duration']}v</div>
                    <div class="health-meter-sub">Media: {stability['mean_duration']}v ({stability['ratio']:.1f}x)</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if stability['score'] >= 70 else ('#f39c12' if stability['score'] >= 40 else '#FF4136')};">{stability_icon} {stability_label}</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if impact['net_score'] > 0 else ('#888' if impact['net_score'] == 0 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">📊</span>
                        <span class="health-meter-title">Impacto Neto</span>
                    </div>
                    <div class="health-meter-value">{impact_icon} {impact_label}</div>
                    <div class="health-meter-sub">Fav {impact['favorable']} / Adv {impact['adverse']}</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if impact['net_score'] > 0 else ('#888' if impact['net_score'] == 0 else '#FF4136')};">Ult. {impact['total']} cambios</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if wr_data['win_rate'] and wr_data['win_rate'] >= 60 else ('#f39c12' if wr_data['win_rate'] and wr_data['win_rate'] >= 40 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">{wr_icon}</span>
                        <span class="health-meter-title">Win Rate</span>
                    </div>
                    <div class="health-meter-value">{wr_display}</div>
                    <div class="health-meter-sub">Regimen actual{wr_note}</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if wr_data['win_rate'] and wr_data['win_rate'] >= 60 else ('#f39c12' if wr_data['win_rate'] and wr_data['win_rate'] >= 40 else '#888')};">{wr_data.get('note', f'R{current_state}') if current_state >= 0 else '—'}</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if confidence >= 70 else ('#f39c12' if confidence >= 45 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">{conf_icon}</span>
                        <span class="health-meter-title">Confianza HMM</span>
                    </div>
                    <div class="health-meter-value">{conf_label}</div>
                    <div class="health-meter-sub">Entropía normalizada</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if confidence >= 70 else ('#f39c12' if confidence >= 45 else '#FF4136')};">{'Alta' if confidence >= 70 else ('Media' if confidence >= 45 else 'Baja')} confianza</div>
                </div>
            </div>
            {alerts_html}
        </div>
    </div>"""
    return html


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ──────────────────────────────────────────────────────────────────────────────


def _make_price_chart(df: pd.DataFrame, states: np.ndarray, asset: str, timeframe: str, signal_info: Dict, state_summary: pd.DataFrame = None) -> go.Figure:
    """Genera gráfico de precio con regímenes y señales."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.55, 0.45])
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=50, r=20, t=30, b=20),
                      hovermode="x unified", paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)))

    # Regime backgrounds — OPTIMIZADO: shapes en lote en lugar de add_vrect individual
    regime_shapes = []
    if state_summary is not None and not state_summary.empty and len(states) > 0:
        unique_states = np.unique(states)
        state_descriptions = {}
        for _, r in state_summary.iterrows():
            state_descriptions[int(r["state"])] = r["description"]
        for s in unique_states:
            mask = states == s
            if mask.sum() < 2:
                continue
            transitions = np.where(np.diff(mask.astype(int)) != 0)[0]
            starts = np.concatenate([[0], transitions + 1])
            ends = np.concatenate([transitions + 1, [len(states)]])
            for st, en in zip(starts, ends):
                if mask[st]:
                    color = REGIME_COLORS[int(s) % len(REGIME_COLORS)]
                    x0_str = df.index[st].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[st], 'strftime') else str(df.index[st])
                    x1_str = df.index[min(en, len(df)-1)].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[min(en, len(df)-1)], 'strftime') else str(df.index[min(en, len(df)-1)])
                    regime_shapes.append(dict(
                        type="rect", x0=x0_str, x1=x1_str, yref="paper", y0=0, y1=1,
                        fillcolor=color, opacity=0.08, layer="below", line_width=0,
                    ))
        if regime_shapes:
            fig.update_layout(shapes=regime_shapes)

    # Candlestick
    colors = ["#089981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#F23645" for i in range(len(df))]
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 increasing_line_color="#089981", decreasing_line_color="#F23645",
                                 name=asset, showlegend=False), row=1, col=1)

    # EMA 55
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], mode="lines",
                                 line=dict(color="white", width=1, dash="dash"), name="EMA 55"), row=1, col=1)

    # Signal markers
    marker_symbols = {"LONG": "triangle-up", "SHORT": "triangle-down"}
    marker_colors = {"LONG": "#089981", "SHORT": "#F23645"}
    for signal_type in ["LONG", "SHORT"]:
        col = f"signal_{signal_type.lower()}"
        if col not in df.columns:
            continue
        signal_mask = df[col].astype(bool) & (~df[col].shift(1).fillna(False).astype(bool))
        signal_dates = df.index[signal_mask]
        signal_prices = df["Low"][signal_mask] if signal_type == "LONG" else df["High"][signal_mask]
        if len(signal_dates) > 0:
            fig.add_trace(go.Scatter(x=signal_dates, y=signal_prices, mode="markers",
                                     marker=dict(symbol=marker_symbols[signal_type], size=12,
                                                 color=marker_colors[signal_type], line=dict(width=1, color="white")),
                                     name=f"Señal {signal_type}"), row=1, col=1)

    # Volume
    vol_max = df["Volume"].max() if df["Volume"].max() > 0 else 1
    vol_colors = ["#089981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#F23645" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"] / vol_max * 100, marker_color=vol_colors,
                         name="Volumen", opacity=0.5, showlegend=False), row=2, col=1)

    # Layout
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08), row=2, col=1)
    fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volumen (%)", row=2, col=1, range=[0, 110])

    rangeselector = dict(buttons=list([
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), bgcolor="#1e1e24", activecolor="#2962FF", font=dict(color="white"))
    fig.update_xaxes(rangeselector=rangeselector, row=2, col=1)
    return fig


def _make_tradingview_chart(df: pd.DataFrame, states: np.ndarray, asset: str, timeframe: str, signal_info: Dict, state_summary: pd.DataFrame = None) -> go.Figure:
    """
    Genera grafico estilo TRADINGVIEW:
    - Fila 1 (0.65): Precio + EMA55 (candlestick)
    - Fila 2 (0.35): Squeeze Momentum histogram + ADX superpuesto (eje secundario)
    Compartiendo el mismo eje X (zoom/pan sincronizado).
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )
    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=50, r=50, t=30, b=20),
        hovermode="x unified", paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
        legend=dict(orientation="h", y=1.01, x=0.5, xanchor="center", font=dict(size=9)),
    )

    # ── Regime backgrounds (fila 1, solo precio) ──
    regime_shapes = []
    if state_summary is not None and not state_summary.empty and len(states) > 0:
        unique_states = np.unique(states)
        for s in unique_states:
            mask = states == s
            if mask.sum() < 2:
                continue
            transitions = np.where(np.diff(mask.astype(int)) != 0)[0]
            starts = np.concatenate([[0], transitions + 1])
            ends = np.concatenate([transitions + 1, [len(states)]])
            for st, en in zip(starts, ends):
                if mask[st]:
                    color = REGIME_COLORS[int(s) % len(REGIME_COLORS)]
                    x0_str = df.index[st].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[st], 'strftime') else str(df.index[st])
                    x1_str = df.index[min(en, len(df)-1)].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[min(en, len(df)-1)], 'strftime') else str(df.index[min(en, len(df)-1)])
                    regime_shapes.append(dict(
                        type="rect", x0=x0_str, x1=x1_str, yref="paper", y0=0, y1=1,
                        fillcolor=color, opacity=0.06, layer="below", line_width=0,
                    ))
        if regime_shapes:
            fig.update_layout(shapes=regime_shapes)

    # ══════════════════════════════════════
    # FILA 1: PRECIO + EMA55 (candlestick)
    # ══════════════════════════════════════
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#089981", decreasing_line_color="#F23645",
            name=asset, showlegend=False,
        ), row=1, col=1
    )

    if "ema_slow" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_slow"], mode="lines",
                line=dict(color="#FFD700", width=1.5), name="EMA 55",
            ), row=1, col=1
        )

    # Senal markers (LONG / SHORT)
    marker_symbols = {"LONG": "triangle-up", "SHORT": "triangle-down"}
    marker_colors = {"LONG": "#089981", "SHORT": "#F23645"}
    for signal_type in ["LONG", "SHORT"]:
        col = f"signal_{signal_type.lower()}"
        if col not in df.columns:
            continue
        signal_mask = df[col].astype(bool) & (~df[col].shift(1).fillna(False).astype(bool))
        signal_dates = df.index[signal_mask]
        signal_prices = df["Low"][signal_mask] if signal_type == "LONG" else df["High"][signal_mask]
        if len(signal_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=signal_dates, y=signal_prices, mode="markers",
                    marker=dict(
                        symbol=marker_symbols[signal_type], size=12,
                        color=marker_colors[signal_type], line=dict(width=1, color="white"),
                    ),
                    name=f"Senal {signal_type}",
                ), row=1, col=1
            )

    # ══════════════════════════════════════
    # FILA 2: SQUEEZE MOMENTUM (histograma) + ADX (linea, eje secundario)
    # ══════════════════════════════════════
    max_abs_smi = 10.0
    if "smi_hist" in df.columns:
        smi_values = df["smi_hist"].values
        max_abs_smi = float(np.nanmax(np.abs(smi_values)))
        if max_abs_smi <= 0:
            max_abs_smi = 1.0

        # Colores estilo Pine Script LazyBear
        smi_prev_vals = np.roll(smi_values, 1)
        smi_prev_vals[0] = 0.0
        bar_colors = np.where(
            smi_values >= 0,
            np.where(smi_values >= smi_prev_vals, "#00FF00", "#008000"),
            np.where(smi_values < smi_prev_vals, "#FF0000", "#800000")
        )

        fig.add_trace(
            go.Bar(
                x=df.index, y=smi_values,
                name="SMI Momentum",
                marker=dict(color=list(bar_colors), line_width=0),
                hovertemplate="Momentum: %{y:.2f}<extra></extra>",
            ), row=2, col=1
        )

        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1), row=2, col=1)

        # Squeeze zones (fondo naranja tenue)
        squeeze_shapes = []
        if "squeeze_on" in df.columns:
            squeeze_mask = df["squeeze_on"]
            if squeeze_mask.any():
                squeeze_arr = squeeze_mask.values.astype(int)
                diffs = np.diff(np.concatenate([[0], squeeze_arr, [0]]))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                for st, en in zip(starts, ends):
                    if en - st > 1:
                        x0_str = df.index[st].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[st], 'strftime') else str(df.index[st])
                        x1_str = df.index[min(en, len(df) - 1)].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[min(en, len(df) - 1)], 'strftime') else str(df.index[min(en, len(df) - 1)])
                        squeeze_shapes.append(dict(
                            type="rect", x0=x0_str, x1=x1_str, yref="paper", y0=0, y1=1,
                            fillcolor="rgba(255,152,0,0.06)", layer="below", line_width=0,
                        ))
            if squeeze_shapes:
                current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                fig.update_layout(shapes=current_shapes + squeeze_shapes)

    # ADX superpuesto (eje secundario)
    if "adx" in df.columns:
        adx_offset = df["adx"] - ADX_THRESHOLD
        fig.add_trace(
            go.Scatter(
                x=df.index, y=adx_offset,
                name="ADX",
                line=dict(color="#FFFFFF", width=2.0),
                customdata=df["adx"].values,
                hovertemplate="ADX: %{customdata:.1f}<extra></extra>",
            ), row=2, col=1, secondary_y=True
        )

        # Umbral ADX (linea a 0)
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[0, 0],
                name=f"Umbral ADX {ADX_THRESHOLD:.0f}",
                line=dict(color="rgba(255,215,0,0.6)", width=1.5, dash="dot"),
                showlegend=True, hoverinfo="skip",
            ), row=2, col=1, secondary_y=True
        )

    # ── Layout & ejes ──
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="ALL"),
            ]),
            bgcolor="#1e1e24", activecolor="#2962FF", font=dict(color="white"),
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Precio ($)", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(
        title_text="SMI Momentum", secondary_y=False, row=2, col=1,
        title_font=dict(size=10),
        range=[-max_abs_smi * 1.2, max_abs_smi * 1.2],
    )
    fig.update_yaxes(
        title_text="ADX", secondary_y=True, row=2, col=1,
        title_font=dict(size=10),
        tickfont=dict(size=9, color="#FFFFFF"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# GENERACIÓN HTML
# ──────────────────────────────────────────────────────────────────────────────


def _generate_tf_inner(data: TimeframeData, asset: str, timeframe: str) -> str:
    """Genera el HTML interno (secciones del body) para UNA temporalidad."""
    df_full = data.df_full
    states = data.states_full
    state_summary = data.state_summary
    trans_mat = data.trans_mat
    signal_info = data.signal_info
    regime_changes = data.regime_changes
    current_state = _current_regime_index(states)
    signal = signal_info["signal"]
    strength = signal_info["strength"]
    price = signal_info["price"]
    date = signal_info["date"]

    price_change = df_full["Close"].pct_change().iloc[-1] * 100 if len(df_full) > 1 else 0.0
    price_color = "#089981" if price_change >= 0 else "#F23645"
    price_arrow = "▲" if price_change >= 0 else "▼"
    signal_color = "#089981" if signal == "LONG" else ("#F23645" if signal == "SHORT" else "#FF851B")

    signal_start = signal_info.get("signal_start")
    signal_start_html = ""
    if signal_start and signal in ("LONG", "SHORT"):
        bars_since = signal_info.get("expiration", {}).get("bars_since_start", 0)
        tf_unit = {"1h": "h", "4h": "h", "1d": "d", "1wk": "sem"}.get(timeframe, "v")
        signal_start_html = f'''
                <div class="signal-start-box">
                    <span class="signal-start-icon">📅</span>
                    <div class="signal-start-info">
                        <span class="signal-start-label">Activa desde</span>
                        <span class="signal-start-date">{signal_start}</span>
                        <span class="signal-start-bars">(hace {bars_since} {tf_unit})</span>
                    </div>
                </div>'''

    expiration = signal_info.get("expiration")
    expiration_html = ""
    expired_badge_html = ""
    if expiration and signal in ("LONG", "SHORT"):
        exp = expiration
        bars_since = exp["bars_since_start"]
        max_bars = exp["max_bars"]
        bars_rem = exp["bars_remaining"]
        expired = exp["expired"]
        progress = exp["progress_pct"]
        window_label = exp["window_label"]
        if expired:
            prog_color = "#F23645"
        elif progress >= 75:
            prog_color = "#FF851B"
        elif progress >= 50:
            prog_color = "#2962FF"
        else:
            prog_color = "#089981"
        expired_badge_html = (
            '<span class="expired-badge">EXPIRADA</span>'
            if expired else
            f'<span class="expiration-badge" style="color:{prog_color};">'
            f'{bars_rem}/{max_bars} restantes</span>'
        )
        expiration_html = f"""
        <div class="expiration-container">
            <div class="expiration-header">
                <span class="expiration-label">Ventana Jaime Merino</span>
                <span class="expiration-window">{window_label}</span>
            </div>
            <div class="expiration-bar-track">
                <div class="expiration-bar-fill" style="width:{progress}%;background:{prog_color};"></div>
            </div>
            <div class="expiration-details">
                <span>Vela {bars_since} de {max_bars}</span>
                <span class="{'expired-text' if expired else ''}">
                    {'⛔ EXPIRADA' if expired else f'{bars_rem} velas restantes'}
                </span>
            </div>
        </div>"""

    regime_desc = ""
    regime_duration = "-"
    regime_pct = "-"
    if not state_summary.empty and current_state >= 0:
        row = state_summary[state_summary["state"] == current_state]
        if not row.empty:
            regime_desc = row.iloc[0]["description"]
            regime_duration = f"{row.iloc[0]['mean_duration_bars']}"
            regime_pct = f"{row.iloc[0]['pct_time']}%"
    regime_color = REGIME_COLORS[current_state % len(REGIME_COLORS)] if current_state >= 0 else "#888"

    current_bias = _classify_regime_bias(regime_desc)
    if signal == "LONG":
        regime_alignment = "favorable" if current_bias == "bullish" else ("adverse" if current_bias == "bearish" else "neutral")
    elif signal == "SHORT":
        regime_alignment = "favorable" if current_bias == "bearish" else ("adverse" if current_bias == "bullish" else "neutral")
    else:
        regime_alignment = "no_signal"

    regime_warnings = []
    for change in regime_changes[:5]:
        to_bias = _classify_regime_bias(change["to_desc"])
        if signal == "LONG":
            impact = "adverse" if to_bias == "bearish" else ("favorable" if to_bias == "bullish" else "neutral")
        elif signal == "SHORT":
            impact = "adverse" if to_bias == "bullish" else ("favorable" if to_bias == "bearish" else "neutral")
        else:
            impact = "neutral"
        regime_warnings.append({**change, "impact": impact})

    plotly_config = dict(scrollZoom=True, displayModeBar=True, displaylogo=False,
                         doubleClick="reset", responsive=True,
                         modeBarButtonsToRemove=["lasso2d", "select2d", "sendDataToCloud"])

    # MAIN CHART: TradingView-style (Price+EMA55 on top, Squeeze+ADX below, shared x-axis)
    fig_tv = _make_tradingview_chart(df_full, states, asset, timeframe, signal_info, state_summary)
    tradingview_chart_html = fig_tv.to_html(full_html=False, include_plotlyjs=False, div_id=f"tv-{timeframe}", config=plotly_config)

    # Additional detail chart: Price with full regime backgrounds and signal markers
    fig = _make_price_chart(df_full, states, asset, timeframe, signal_info, state_summary)
    price_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"pc-{timeframe}", config=plotly_config)

    conditions_html = ""
    for label, info in signal_info["conditions"].items():
        icon = "✅" if info["met"] else "❌"
        met_class = "condition-met" if info["met"] else "condition-not-met"
        detail = info["detail"]
        conditions_html += f"""
        <div class="condition-row {met_class}">
            <span class="condition-icon">{icon}</span>
            <span class="condition-label">{label}</span>
            <span class="condition-detail">{detail}</span>
        </div>"""

    regime_cards = ""
    for _, row in state_summary.iterrows():
        s = int(row["state"])
        c = REGIME_COLORS[s % len(REGIME_COLORS)]
        desc = row["description"]
        ret = row["mean_return"]
        vol = row["volatility"]
        pct = row["pct_time"]
        dur = row["mean_duration_bars"]
        is_active = "active-regime-card" if s == current_state else ""
        active_tag = "<div class='active-tag'>ACTUAL</div>" if s == current_state else ""

        if "EXPANSION BAJISTA" in desc:
            expl = "Mercado en panico o capitulacion. Alta volatilidad con fuertes caidas. Peligro de liquidaciones en cadena. Evitar compras, esperar senal de agotamiento."
        elif "EXPANSION ALCISTA" in desc:
            expl = "Euforia del mercado. Subida violenta con volumen extremo. Momento de alta riesgo/recompensa. Ideal para tomar ganancias parciales."
        elif "TREND ALCISTA" in desc:
            expl = "Tendencia alcista fuerte y saludable. Volatilidad elevada pero direccion clara. Momento ideal para buscar entradas LONG en retrocesos."
        elif "TREND BAJISTA" in desc:
            expl = "Tendencia bajista con conviccion. Momentum negativo fuerte. Preferir estar en corto o en efectivo. No comprar hasta que cambie la estructura."
        elif "ALCISTA FUERTE" in desc:
            expl = "Tendencia alcista consolidada con buen volumen. Confianza del mercado alta. Momento para operar LONG con confianza."
        elif "BAJISTA FUERTE" in desc:
            expl = "Tendencia bajista consolidada. Presion vendedora sostenida. Evitar compras, buscar oportunidades SHORT."
        elif "ALCISTA SUAVE" in desc:
            expl = "Leve presion compradora. El mercado esta probando direccion alcista pero sin fuerza. Requiere confirmacion adicional."
        elif "BAJISTA SUAVE" in desc:
            expl = "Leve presion vendedora. Mercado debil pero no en caida libre. Esperar confirmacion antes de operar."
        elif "ALCISTA" in desc:
            expl = "Mercado con sesgo alcista y volatilidad controlada. Buen momento para operar LONG con gestion de riesgo moderada."
        elif "BAJISTA" in desc:
            expl = "Mercado con sesgo bajista. Se estan formando maximos y minimos decrecientes. Preferir operaciones SHORT o esperar."
        elif "ACUMULACION" in desc:
            expl = "Mercado lateral con baja volatilidad. Grandes jugadores estan acumulando posiciones. Preparandose para el proximo movimiento grande."
        elif "LATERAL" in desc or "VOLATILIDAD NEUTRA" in desc:
            expl = "Mercado sin direccion clara. Alta incertidumbre. Mejor esperar a que el precio salga del rango antes de operar."
        elif "ALTA VOLATILIDAD" in desc:
            expl = "Volatilidad anormal sin direccion clara. Movimientos bruscos en ambas direcciones. Reducir tamano de posiciones."
        else:
            expl = "Regimen de mercado neutro. Operar con cautela hasta que se defina una direccion."
        arrow_icon = "▲" if ret >= 0 else "▼"
        ret_color = "#089981" if ret >= 0 else "#F23645"
        regime_cards += f"""
        <div class="regime-card {is_active}">
            {active_tag}
            <div class="regime-card-header">
                <span class="regime-card-dot" style="background:{c}"></span>
                <span class="regime-card-name">{desc}</span>
                <span class="regime-card-id">R{s}</span>
            </div>
            <div class="regime-card-metrics">
                <div class="metric">
                    <span class="metric-value" style="color:{ret_color}">{arrow_icon} {ret:+.4f}%</span>
                    <span class="metric-label">Retorno Medio</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{vol:.2f}%</span>
                    <span class="metric-label">Volatilidad</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{pct}%</span>
                    <span class="metric-label">Tiempo en este regimen</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{dur}</span>
                    <span class="metric-label">Duracion media (velas)</span>
                </div>
            </div>
            <div class="regime-card-explanation">{expl}</div>
        </div>"""

    verification = data.verification
    verification_html = ""
    if verification and verification["total_signals"] > 0:
        stats = verification["stats"]
        wr = verification["overall_win_rate"]
        total = verification["total_signals"]
        wr_color = "#089981" if wr >= 60 else ("#2962FF" if wr >= 40 else "#F23645")
        side_cards = ""
        for side_name in ["LONG", "SHORT"]:
            s = stats[side_name]
            if s["num_signals"] == 0:
                side_cards += f"""
                <div class="verif-side-card">
                    <div class="verif-side-header" style="color:{'#2ECC40' if side_name == 'LONG' else '#FF4136'}">{side_name}</div>
                    <div class="verif-side-body">
                        <span style="color:#666;font-size:0.75rem;">Sin senales en el periodo</span>
                    </div>
                </div>"""
                continue
            wr_side = s["win_rate"]
            wr_s_color = "#089981" if wr_side >= 60 else ("#2962FF" if wr_side >= 40 else "#F23645")
            avg_ret = s["avg_return"]
            ret_color_side = "#089981" if avg_ret >= 0 else "#F23645"
            ret_arrow = "▲" if avg_ret >= 0 else "▼"
            recent_str = ""
            if s["recent_signals"] > 0 and s["recent_win_rate"] is not None:
                r_wr = s["recent_win_rate"]
                r_color = "#089981" if r_wr >= 60 else ("#2962FF" if r_wr >= 40 else "#F23645")
                recent_str = f"<div class='verif-recent'>Ultimos 30d: <b style='color:{r_color}'>{r_wr:.0f}%</b> ({s['recent_wins']}/{s['recent_signals']})</div>"
            side_cards += f"""
            <div class="verif-side-card">
                <div class="verif-side-header" style="color:{'#2ECC40' if side_name == 'LONG' else '#FF4136'}">
                    {side_name}
                    <span class="verif-side-count">{s['num_signals']} senales</span>
                </div>
                <div class="verif-side-body">
                    <div class="verif-side-row">
                        <span>Win Rate</span>
                        <b style="color:{wr_s_color}">{wr_side:.0f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Retorno medio</span>
                        <b style="color:{ret_color_side}">{ret_arrow} {avg_ret:+.2f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Max favorable medio</span>
                        <b style="color:#2ECC40">{s['avg_max_favorable']:+.2f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Max adverso medio</span>
                        <b style="color:#FF4136">{s['avg_max_adverse']:+.2f}%</b>
                    </div>
                    {recent_str}
                </div>
            </div>"""

        # Trailing stop section
        trailing_html = ""
        if data.trailing_verification:
            tv = data.trailing_verification
            ts = tv["stats"]
            for tside in ["LONG", "SHORT"]:
                if ts[tside]["num_signals"] > 0:
                    trailing_html += f"""
                    <div class="verif-side-card">
                        <div class="verif-side-header" style="color:{'#2ECC40' if tside == 'LONG' else '#FF4136'}">
                            {tside} (Trailing Stop {tv['trail_pct']:.0f}%)
                            <span class="verif-side-count">{ts[tside]['num_signals']} senales</span>
                        </div>
                        <div class="verif-side-body">
                            <div class="verif-side-row">
                                <span>WR TP Fijo</span>
                                <b>{ts[tside]['win_rate_tp']:.0f}%</b>
                            </div>
                            <div class="verif-side-row">
                                <span>WR Trailing Stop</span>
                                <b>{ts[tside]['win_rate_ts']:.0f}%</b>
                            </div>
                            <div class="verif-side-row">
                                <span>WR Combinado</span>
                                <b>{ts[tside]['win_rate_combined']:.0f}%</b>
                            </div>
                            <div class="verif-side-row">
                                <span>Trail Activado</span>
                                <b>{ts[tside]['trail_activated_pct']:.0f}%</b>
                            </div>
                        </div>
                    </div>"""

        verification_html = f"""
        <div class="section" style="margin-top:20px;">
            <div class="section-title">📊 VERIFICACIÓN HISTÓRICA — Regla de Jaime Merino</div>
            <div class="verification-container">
                <div class="verif-header">
                    <span class="verif-window">Ventana: {verification['window_str']}</span>
                    <span class="verif-tp">TP: {verification['tp_target']:.1f}%</span>
                    <span class="verif-total">Total señales: {verification['total_signals']}</span>
                    <span class="verif-wr" style="color:{wr_color};">WR Global: {verification['overall_win_rate']:.0f}%</span>
                </div>
                <div class="verif-cards">
                    {side_cards}
                    {trailing_html}
                </div>
            </div>
        </div>"""

    alignment_badge = _regime_alignment_badge(regime_alignment, signal)
    change_summary = _build_change_summary(regime_changes, state_summary)
    health_meter = _build_trade_health_meter(
        states, state_summary, regime_warnings, trans_mat,
        current_state, signal, regime_alignment, signal_info,
        verification, df_full
    )

    html = f"""
    <!-- TIMEFRAME: {timeframe} -->
    <div id="tf-{timeframe}" class="tf-content" style="display:none;">
        <!-- HEADER: Señal, precio, régimen -->
        <div class="signal-header">
            <div class="signal-main">
                <div class="signal-indicator" style="background:{signal_color};">
                    <span class="signal-label">{SIGNAL_LABELS.get(signal, signal)}</span>
                    <span class="signal-strength">{strength}%</span>
                </div>
                <div class="price-box">
                    <span class="price-value" style="color:{price_color};">${price:,.2f}</span>
                    <span class="price-change" style="color:{price_color};">
                        {price_arrow} {abs(price_change):.2f}%
                    </span>
                    <span class="price-date">{date}</span>
                </div>
                {signal_start_html}
            </div>
            <div class="signal-expiration">
                {expired_badge_html}
                {expiration_html}
            </div>
        </div>

        <!-- REGIME ALIGNMENT -->
        {alignment_badge}

        <!-- MAIN CHART: TradingView-style Price+EMA55 (top) + Squeeze+ADX (bottom) -->
        <div class="dashboard-grid">
            <div class="grid-card grid-chart-full">
                <div class="card-title">📈 PRECIO + EMA55 <span style="color:#666;font-weight:400;text-transform:none;">| Squeeze Momentum + ADX</span></div>
                {tradingview_chart_html}
            </div>
        </div>

        <!-- TRADE HEALTH METER -->
        {health_meter}

        <!-- CONDITIONS + REGIME CARDS -->
        <div class="dashboard-grid-2col">
            <div class="grid-card">
                <div class="card-title">🔍 Condiciones de la señal</div>
                <div class="conditions-container">
                    {conditions_html}
                </div>
            </div>
            <div class="grid-card">
                <div class="card-title">🏛️ Guía de Regímenes</div>
                <div class="regime-cards-container">
                    {regime_cards}
                </div>
            </div>
        </div>

        <!-- DETAIL CHART: Price with regime backgrounds (collapsible) -->
        <details class="grid-card detail-chart-container" style="margin-bottom:16px;">
            <summary class="card-title" style="cursor:pointer;">📉 Precio con Regímenes y Señales (detalle)</summary>
            <div style="margin-top:12px;">
                {price_chart_html}
            </div>
        </details>

        <!-- CHANGE SUMMARY -->
        {change_summary if change_summary else ''}

        <!-- VERIFICATION -->
        {verification_html}
    </div>"""
    return html


# ──────────────────────────────────────────────────────────────────────────────
# ASSET SELECTOR
# ──────────────────────────────────────────────────────────────────────────────


def _build_asset_selector(current_asset: str) -> str:
    """Genera el selector de activos en HTML."""
    options = "\n".join(
        f'            <option value="{a}"{" selected" if a == current_asset else ""}>{a}</option>'
        for a in POPULAR_ASSETS
    )
    return f"""
        <div class="asset-selector">
            <select id="asset-select" onchange="window.location.href='?asset='+this.value">
                {options}
            </select>
            <div class="cmd-box">
                <span id="cmd-text">python tradinglatino_hmm/main.py --asset {current_asset}</span>
                <button onclick="copyCommand()" title="Copiar comando">📋</button>
            </div>
        </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# BUILD MULTI-TF DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────


def build_multi_tf_dashboard(results: Dict[str, TimeframeData], asset: str) -> str:
    """Genera el dashboard HTML completo multi-timeframe."""
    # Tab headers
    tab_headers = ""
    tf_inner_html = ""
    all_timeframes = ["1h", "4h", "1d", "1wk"]
    for i, tf in enumerate(all_timeframes):
        active = "active" if i == 0 else ""
        tab_headers += f'<button class="tf-tab {active}" onclick="switchTF(\'{tf}\')">{tf}</button>'
        if tf in results:
            if tf == "1w" and "1wk" in results:
                continue  # skip duplicate
            data = results[tf]
            inner = _generate_tf_inner(data, asset, tf)
            tf_inner_html += inner
        elif tf == "1w" and "1wk" in results:
            data = results["1wk"]
            inner = _generate_tf_inner(data, asset, "1wk")
            tf_inner_html += inner

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingLatino · HMM Regime Dashboard — {asset}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f13; color: #e0e0e0; padding: 20px; }}
        .dashboard-container {{ max-width: 1400px; margin: 0 auto; }}
        .dashboard-header {{ display: flex; justify-content: space-between; align-items: center;
                            margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }}
        .dashboard-title {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}
        .dashboard-title span {{ color: #2962FF; }}
        .asset-selector select {{ background: #1e1e24; color: #fff; border: 1px solid #333;
                                 padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; }}
        .cmd-box {{ margin-top: 8px; display: flex; align-items: center; gap: 8px;
                    background: #1a1a20; padding: 6px 12px; border-radius: 6px; }}
        .cmd-box span {{ font-family: monospace; font-size: 0.75rem; color: #888; }}
        .cmd-box button {{ background: none; border: none; color: #888; cursor: pointer; font-size: 1rem; }}
        .cmd-box button:hover {{ color: #fff; }}
        .tf-tabs {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
        .tf-tab {{ background: #1e1e24; color: #888; border: 1px solid #333;
                  padding: 8px 20px; border-radius: 8px; cursor: pointer; font-size: 0.85rem;
                  font-weight: 600; transition: all 0.2s; }}
        .tf-tab:hover {{ background: #2a2a32; color: #fff; }}
        .tf-tab.active {{ background: #2962FF; color: #fff; border-color: #2962FF; }}
        .tf-content {{ animation: fadeIn 0.3s ease; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        /* Signal Header */
        .signal-header {{ display: flex; justify-content: space-between; align-items: flex-start;
                         background: linear-gradient(135deg, #1a1a24, #16161e);
                         border: 1px solid #2a2a34; border-radius: 12px; padding: 20px;
                         margin-bottom: 16px; flex-wrap: wrap; gap: 16px; }}
        .signal-main {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
        .signal-indicator {{ display: flex; flex-direction: column; align-items: center;
                           padding: 12px 24px; border-radius: 10px; min-width: 100px; }}
        .signal-label {{ font-size: 1.2rem; font-weight: 800; letter-spacing: 1px; color: #fff; }}
        .signal-strength {{ font-size: 1.5rem; font-weight: 700; color: rgba(255,255,255,0.9); }}
        .price-box {{ display: flex; flex-direction: column; }}
        .price-value {{ font-size: 1.6rem; font-weight: 700; }}
        .price-change {{ font-size: 1rem; font-weight: 600; }}
        .price-date {{ font-size: 0.75rem; color: #888; margin-top: 2px; }}
        .signal-start-box {{ display: flex; align-items: center; gap: 8px;
                           background: #1a1a24; padding: 8px 14px; border-radius: 8px;
                           border: 1px solid #2a2a34; }}
        .signal-start-icon {{ font-size: 1.2rem; }}
        .signal-start-info {{ display: flex; flex-direction: column; }}
        .signal-start-label {{ font-size: 0.65rem; color: #888; text-transform: uppercase; }}
        .signal-start-date {{ font-size: 0.9rem; font-weight: 600; color: #fff; }}
        .signal-start-bars {{ font-size: 0.7rem; color: #666; }}
        .signal-expiration {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }}
        .expiration-badge {{ font-size: 0.8rem; font-weight: 600; }}
        .expired-badge {{ background: #F23645; color: #fff; padding: 4px 12px;
                         border-radius: 4px; font-size: 0.75rem; font-weight: 700; }}
        .expiration-container {{ background: #1e1e28; border-radius: 8px; padding: 10px 14px;
                               min-width: 200px; }}
        .expiration-header {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
        .expiration-label {{ font-size: 0.7rem; color: #888; }}
        .expiration-window {{ font-size: 0.75rem; font-weight: 600; color: #2962FF; }}
        .expiration-bar-track {{ height: 6px; background: #2a2a34; border-radius: 3px;
                                overflow: hidden; margin-bottom: 4px; }}
        .expiration-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
        .expiration-details {{ display: flex; justify-content: space-between; font-size: 0.65rem; color: #888; }}
        .expired-text {{ color: #F23645; font-weight: 700; }}

        /* Grid */
        .dashboard-grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; margin-bottom: 16px; }}
        .dashboard-grid-2col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
        @media (max-width: 900px) {{ .dashboard-grid-2col {{ grid-template-columns: 1fr; }} }}
        .grid-card {{ background: #1a1a24; border: 1px solid #2a2a34; border-radius: 12px;
                     padding: 16px; overflow: hidden; }}
        .grid-chart-full {{ grid-column: 1 / -1; }}
        .card-title {{ font-size: 0.85rem; font-weight: 600; color: #aaa; margin-bottom: 12px;
                      text-transform: uppercase; letter-spacing: 0.5px; }}

        /* Conditions */
        .conditions-container {{ display: flex; flex-direction: column; gap: 6px; }}
        .condition-row {{ display: flex; align-items: center; gap: 8px; padding: 6px 10px;
                         border-radius: 6px; font-size: 0.75rem; }}
        .condition-met {{ background: rgba(8,153,129,0.08); }}
        .condition-not-met {{ background: rgba(242,54,69,0.08); opacity: 0.5; }}
        .condition-icon {{ font-size: 0.8rem; }}
        .condition-label {{ flex: 1; color: #ccc; }}
        .condition-detail {{ font-size: 0.65rem; color: #888; max-width: 200px;
                            text-align: right; overflow: hidden; text-overflow: ellipsis; }}

        /* Regime Cards */
        .regime-cards-container {{ display: flex; flex-direction: column; gap: 10px;
                                  max-height: 600px; overflow-y: auto; }}
        .regime-card {{ background: rgba(255,255,255,0.03); border: 1px solid #2a2a34;
                       border-radius: 8px; padding: 12px; position: relative; }}
        .regime-card.active-regime-card {{ border-color: #2962FF; background: rgba(41,98,255,0.06); }}
        .active-tag {{ position: absolute; top: 8px; right: 8px; background: #2962FF; color: #fff;
                      font-size: 0.6rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; }}
        .regime-card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
        .regime-card-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
        .regime-card-name {{ font-size: 0.8rem; font-weight: 600; color: #fff; flex: 1; }}
        .regime-card-id {{ font-size: 0.65rem; color: #666; }}
        .regime-card-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px; }}
        .metric {{ display: flex; flex-direction: column; }}
        .metric-value {{ font-size: 0.85rem; font-weight: 700; }}
        .metric-label {{ font-size: 0.6rem; color: #666; text-transform: uppercase; }}
        .regime-card-explanation {{ font-size: 0.7rem; color: #888; line-height: 1.4; }}

        /* Change Summary */
        .change-summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                          background: #1a1a24; border: 1px solid #2a2a34; border-radius: 12px;
                          padding: 16px; margin-bottom: 16px; }}
        @media (max-width: 600px) {{ .change-summary {{ grid-template-columns: repeat(2, 1fr); }} }}
        .summary-stat {{ display: flex; flex-direction: column; align-items: center; text-align: center; }}
        .summary-stat-value {{ font-size: 1rem; font-weight: 700; color: #fff; }}
        .summary-stat-label {{ font-size: 0.65rem; color: #888; }}

        /* Regime Alignment Badge */
        .regime-alignment-badge {{ display: flex; align-items: center; gap: 10px;
                                  padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; }}
        .alignment-favorable {{ background: rgba(8,153,129,0.1); border-left: 3px solid #089981; }}
        .alignment-adverse {{ background: rgba(242,54,69,0.1); border-left: 3px solid #F23645; }}
        .alignment-neutral {{ background: rgba(255,133,27,0.1); border-left: 3px solid #FF851B; }}
        .alignment-icon {{ font-size: 1.2rem; }}
        .alignment-text {{ font-weight: 600; font-size: 0.85rem; color: #fff; }}
        .alignment-sub {{ font-size: 0.7rem; color: #888; }}

        /* Health Meter */
        .health-meter-container {{ background: rgba(255,255,255,0.02); border-radius: 12px;
                                  padding: 16px; }}
        .health-verdict {{ background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px 16px;
                          margin-bottom: 12px; }}
        .health-verdict-row {{ display: flex; align-items: center; gap: 12px; }}
        .health-verdict-icon {{ font-size: 2rem; }}
        .health-verdict-info {{ flex: 1; }}
        .health-verdict-label {{ font-size: 1.1rem; font-weight: 700; display: block; }}
        .health-verdict-desc {{ font-size: 0.75rem; color: #888; }}
        .health-score-ring {{ display: flex; flex-direction: column; align-items: center;
                             border: 3px solid; border-radius: 50%; width: 64px; height: 64px;
                             justify-content: center; flex-shrink: 0; }}
        .health-score-value {{ font-size: 1.3rem; font-weight: 700; }}
        .health-score-label {{ font-size: 0.6rem; color: #888; }}
        .health-bar-container {{ margin-bottom: 12px; }}
        .health-bar-track {{ height: 8px; background: #2a2a34; border-radius: 4px; overflow: hidden; }}
        .health-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
        .health-bar-labels {{ display: flex; justify-content: space-between; font-size: 0.6rem;
                              margin-top: 4px; }}
        .health-action {{ border-left: 3px solid; padding: 8px 12px; margin-bottom: 12px; }}
        .health-action-label {{ font-size: 0.7rem; color: #888; margin-right: 8px; }}
        .health-action-text {{ font-size: 0.85rem; font-weight: 600; }}
        .health-meters-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
                              margin-bottom: 12px; }}
        @media (max-width: 700px) {{ .health-meters-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
        .health-meter-card {{ background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; }}
        .health-meter-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }}
        .health-meter-icon {{ font-size: 1rem; }}
        .health-meter-title {{ font-size: 0.7rem; font-weight: 600; color: #aaa; text-transform: uppercase; }}
        .health-meter-value {{ font-size: 1rem; font-weight: 700; color: #fff; }}
        .health-meter-sub {{ font-size: 0.6rem; color: #666; }}
        .health-meter-status {{ font-size: 0.65rem; font-weight: 600; margin-top: 4px; }}
        .health-alerts {{ background: rgba(242,54,69,0.08); border-radius: 8px; padding: 8px 12px; }}
        .health-alert-row {{ font-size: 0.7rem; color: #e0e0e0; padding: 2px 0; }}

        /* Verification */
        .verification-container {{ background: rgba(255,255,255,0.02); border-radius: 8px; padding: 12px; }}
        .verif-header {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;
                         font-size: 0.75rem; }}
        .verif-window {{ color: #888; }}
        .verif-tp {{ color: #2962FF; font-weight: 600; }}
        .verif-total {{ color: #888; }}
        .verif-wr {{ font-weight: 700; }}
        .verif-cards {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        @media (max-width: 600px) {{ .verif-cards {{ grid-template-columns: 1fr; }} }}
        .verif-side-card {{ background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden; }}
        .verif-side-header {{ padding: 8px 12px; font-weight: 700; font-size: 0.8rem;
                             display: flex; justify-content: space-between; }}
        .verif-side-count {{ font-weight: 400; font-size: 0.65rem; color: #888; }}
        .verif-side-body {{ padding: 8px 12px; }}
        .verif-side-row {{ display: flex; justify-content: space-between; font-size: 0.7rem;
                          padding: 3px 0; }}
        .verif-recent {{ font-size: 0.65rem; margin-top: 6px; padding-top: 6px;
                         border-top: 1px solid #2a2a34; }}

        /* Section */
        .section {{ margin-bottom: 16px; }}
        .section-title {{ font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 12px; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <div class="dashboard-title">🌐 <span>TradingLatino</span> · HMM Regime Dashboard</div>
            {_build_asset_selector(asset)}
        </div>
        <div class="tf-tabs">
            {tab_headers}
        </div>
        {tf_inner_html}
    </div>
    <script>
        function switchTF(tf) {{
            document.querySelectorAll('.tf-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.tf-tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tf-' + tf).style.display = 'block';
            document.querySelector(`button[onclick="switchTF('${{tf}}')"]`).classList.add('active');
            // Resize Plotly charts in the active tab
            const tab = document.getElementById('tf-' + tf);
            if (tab) {{
                const plots = tab.querySelectorAll('.js-plotly-plot');
                plots.forEach(p => {{ try {{ Plotly.Plots.resize(p); }} catch(e) {{}} }});
            }}
        }}
        function copyCommand() {{
            const cmd = document.getElementById('cmd-text');
            navigator.clipboard.writeText(cmd.textContent);
            const btn = cmd.nextElementSibling;
            btn.textContent = '✅';
            setTimeout(() => btn.textContent = '📋', 2000);
        }}
        // Activate first tab
        document.addEventListener('DOMContentLoaded', () => {{
            const firstTab = document.querySelector('.tf-tab');
            if (firstTab) firstTab.click();
        }});
        // Synchronize chart zoom/pan across charts of the same timeframe
        document.addEventListener('plotly_relayout', function(event) {{
            if (!event.detail || !event.target) return;
            const divId = event.target.id;
            if (!divId) return;
            const parts = divId.split('-');
            const prefix = parts[0];
            const tf = parts.slice(1).join('-');
            if (['tv', 'pc'].includes(prefix) && tf) {{
                const xRange = event.detail['xaxis.range[0]'] ? [event.detail['xaxis.range[0]'], event.detail['xaxis.range[1]']] : null;
                if (!xRange) return;
                ['tv-' + tf, 'pc-' + tf].forEach(id => {{
                    if (id !== divId) {{
                        const otherDiv = document.getElementById(id);
                        if (otherDiv) {{
                            try {{ Plotly.relayout(otherDiv, {{ 'xaxis.range[0]': xRange[0], 'xaxis.range[1]': xRange[1] }}); }} catch(e) {{}}
                        }}
                    }}
                }});
            }}
        }});

        // Collapsible detail charts: smooth open/close
        document.querySelectorAll('details.detail-chart-container').forEach(detail => {{
            detail.addEventListener('toggle', function() {{
                const plots = this.querySelectorAll('.js-plotly-plot');
                setTimeout(() => {{
                    plots.forEach(p => {{ try {{ Plotly.Plots.resize(p); }} catch(e) {{}} }});
                }}, 100);
            }});
        }});
    </script>
</body>
</html>"""
