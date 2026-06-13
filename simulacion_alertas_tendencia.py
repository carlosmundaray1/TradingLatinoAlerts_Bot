#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SIMULACIÓN · VALIDACIÓN DE ALERTAS DE CAMBIO DE TENDENCIA
================================================================================
Analiza en profundidad si las alertas de cambio de régimen HMM detectan
correctamente los cambios de tendencia LONG ↔ SHORT en un histórico extenso
de BTC-USD.

Ejecución:
    python simulacion_alertas_tendencia.py

Genera:
    simulacion_alertas_BTC-USD.html  → Reporte HTML interactivo
    simulacion_alertas_log.txt        → Log detallado en consola

Preguntas que responde:
    1. ¿Cada cambio de régimen HMM va seguido de un cambio de señal?
    2. ¿Hay cambios de señal SIN cambio de régimen (falsos negativos)?
    3. ¿Hay cambios de régimen SIN cambio de señal (falsos positivos)?
    4. ¿Funciona tanto para LONG→SHORT como SHORT→LONG?
    5. ¿Con cuántas velas de antelación alerta el régimen el cambio?
================================================================================
"""
import contextlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTAR TODO EL PIPELINE DESDE tradinglatino_hmm_clean.py
# ──────────────────────────────────────────────────────────────────────────────
# Importamos el archivo como módulo para reutilizar todas las funciones
# sin duplicar código.

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import dinámico del módulo principal
import importlib.util as _imp_util

_MODULE_PATH = SCRIPT_DIR / "tradinglatino_hmm_clean.py"
_SPEC = _imp_util.spec_from_file_location("hmm_clean", _MODULE_PATH)
_HMM = _imp_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HMM)

# Re-exportar las funciones y configuraciones que necesitamos
compute_all_indicators = _HMM.compute_all_indicators
build_hmm_features = _HMM.build_hmm_features
fit_hmm = _HMM.fit_hmm
compute_signal = _HMM.compute_signal
_detect_regime_changes = _HMM._detect_regime_changes
_classify_regime_bias = _HMM._classify_regime_bias
_describe_regime = _HMM._describe_regime
_load_data_original = _HMM.load_data
_format_date = _HMM._format_date
verify_signals_historically = _HMM.verify_signals_historically

# Configuración
ASSET = "BTC-USD"
TIMEFRAMES = ["1d", "1wk"]  # Daily + Weekly son los mejores para ver tendencias largas

# Períodos extendidos para capturar ciclos completos de mercado
PERIODS = {
    "1d": "5y",    # 5 años de datos diarios
    "1wk": "10y",  # 10 años de datos semanales
}

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS EXTENDIDA (períodos más largos que el original)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_HTML = f"simulacion_alertas_{ASSET}.html"
OUTPUT_LOG = "simulacion_alertas_log.txt"

def _log(msg: str, log_lines: List[str]) -> None:
    """Imprime y guarda en log."""
    try:
        print(msg)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe)
    log_lines.append(msg)

def load_data_extended(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Carga datos con períodos extendidos reutilizando la función original.
    Para 1d usa '5y', para 1wk usa '10y'.
    """
    # Sobrescribir las constantes de período en el módulo importado
    _HMM.PERIOD_1D = PERIODS.get("1d", "2y")
    _HMM.PERIOD_1W = PERIODS.get("1wk", "4y")
    _HMM.PERIOD_4H = "2y"
    _HMM.PERIOD_1H = "1y"
    return _load_data_original(asset, timeframe)

# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS CRUZADO: CAMBIOS DE RÉGIMEN vs CAMBIOS DE SEÑAL
# ──────────────────────────────────────────────────────────────────────────────

def find_regime_changes_detailed(states: np.ndarray, index, state_summary: pd.DataFrame) -> List[Dict]:
    """
    Versión mejorada de _detect_regime_changes que también guarda
    la duración exacta y el estado numérico para análisis posterior.
    """
    desc_map: Dict[int, str] = {}
    for _, r in state_summary.iterrows():
        desc_map[int(r["state"])] = r["description"]
    changes = []
    prev_state = states[0]
    change_start = 0
    for i in range(1, len(states)):
        if states[i] != prev_state:
            if prev_state >= 0 and states[i] >= 0:
                from_desc = desc_map.get(int(prev_state), f"R{prev_state}")
                to_desc = desc_map.get(int(states[i]), f"R{states[i]}")
                changes.append({
                    "idx": i,
                    "date": index[i],
                    "date_str": _format_date(index[i]),
                    "from_state": int(prev_state),
                    "to_state": int(states[i]),
                    "from_desc": from_desc,
                    "to_desc": to_desc,
                    "from_bias": _classify_regime_bias(from_desc),
                    "to_bias": _classify_regime_bias(to_desc),
                    "duration_velas": i - change_start,
                })
            prev_state = states[i]
            change_start = i
    return changes

def find_signal_changes(df: pd.DataFrame) -> List[Dict]:
    """
    Encuentra todos los cambios de señal LONG/SHORT/FLAT en el DataFrame.
    Retorna lista con índices, fechas, señal anterior y nueva.
    """
    changes = []
    prev_signal = "FLAT"
    for i in range(len(df)):
        is_long = bool(df["signal_long"].iloc[i]) if "signal_long" in df.columns else False
        is_short = bool(df["signal_short"].iloc[i]) if "signal_short" in df.columns else False
        current_signal = "LONG" if is_long else ("SHORT" if is_short else "FLAT")

        if current_signal != prev_signal:
            changes.append({
                "idx": i,
                "date": df.index[i],
                "date_str": _format_date(df.index[i]),
                "from_signal": prev_signal,
                "to_signal": current_signal,
                "price": float(df["Close"].iloc[i]),
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
            })
            prev_signal = current_signal
    return changes

def cross_reference_changes(
    regime_changes: List[Dict],
    signal_changes: List[Dict],
    max_lag_bars: int = 5
) -> Dict[str, Any]:
    """
    Cruza los cambios de régimen con los cambios de señal para determinar:
    - Detecciones correctas: cambio de régimen → cambio de señal (en ≤ max_lag_bars)
    - Falsos positivos: cambio de régimen SIN cambio de señal posterior
    - Falsos negativos: cambio de señal SIN cambio de régimen previo
    - Detección anticipada: cuántas velas antes alertó el régimen
    """
    results = []
    regime_used = set()

    for sc in signal_changes:
        if sc["from_signal"] == sc["to_signal"]:
            continue
        # Buscar el cambio de régimen más cercano ANTES del cambio de señal
        best_match = None
        best_lag = None
        best_ri = -1
        for ri, rc in enumerate(regime_changes):
            if ri in regime_used:
                continue
            if rc["idx"] <= sc["idx"] and rc["idx"] >= sc["idx"] - max_lag_bars:
                lag = sc["idx"] - rc["idx"]
                if best_match is None or lag < best_lag:
                    best_match = rc
                    best_lag = lag
                    best_ri = ri

        if best_match is not None:
            regime_used.add(best_ri)
            # Determinar si el cambio de régimen fue en la dirección correcta
            bias_align = False
            to_sig = sc["to_signal"]
            from_sig = sc["from_signal"]
            to_bias = best_match["to_bias"]
            to_desc = best_match["to_desc"].upper()

            if to_sig == "SHORT" and from_sig != "SHORT":
                # Esperamos un cambio a bearish
                bias_align = (to_bias == "bearish" or "BAJISTA" in to_desc)
            elif to_sig == "LONG" and from_sig != "LONG":
                # Esperamos un cambio a bullish
                bias_align = (to_bias == "bullish" or "ALCISTA" in to_desc)

            results.append({
                "type": "correcta",
                "signal_change": sc,
                "regime_change": best_match,
                "lag_bars": best_lag,
                "bias_aligned": bias_align,
                "description": (
                    f"Régimen cambió {best_match['from_desc']}→{best_match['to_desc']} "
                    f"{best_lag} vela(s) antes que la señal {sc['from_signal']}→{sc['to_signal']}"
                ),
            })
        else:
            # Cambio de señal sin régimen previo = posible falso negativo
            results.append({
                "type": "no_regime",
                "signal_change": sc,
                "regime_change": None,
                "lag_bars": None,
                "bias_aligned": None,
                "description": (
                    f"Señal cambió {sc['from_signal']}→{sc['to_signal']} "
                    f"SIN cambio de régimen previo en {max_lag_bars} velas"
                ),
            })

    # Régimen changes no usados = falsos positivos (para señales)
    for ri, rc in enumerate(regime_changes):
        if ri not in regime_used:
            results.append({
                "type": "no_signal",
                "signal_change": None,
                "regime_change": rc,
                "lag_bars": None,
                "bias_aligned": None,
                "description": (
                    f"Régimen cambió {rc['from_desc']}→{rc['to_desc']} "
                    f"SIN cambio de señal posterior"
                ),
            })

    # Estadísticas
    total_regime = len(regime_changes)
    total_signal = len([s for s in signal_changes if s["from_signal"] != s["to_signal"]])
    detected = sum(1 for r in results if r["type"] == "correcta")
    false_pos = sum(1 for r in results if r["type"] == "no_signal")
    false_neg = sum(1 for r in results if r["type"] == "no_regime")
    bias_aligned = sum(1 for r in results if r.get("bias_aligned"))

    lags = [r["lag_bars"] for r in results if r["lag_bars"] is not None]
    avg_lag = float(np.mean(lags)) if lags else 0
    min_lag = min(lags) if lags else 0
    max_lag = max(lags) if lags else 0

    return {
        "results": results,
        "total_regime_changes": total_regime,
        "total_signal_changes": total_signal,
        "detected_correctly": detected,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "bias_aligned": bias_aligned,
        "detection_rate": round(detected / total_signal * 100, 1) if total_signal else 0,
        "avg_lag_bars": round(avg_lag, 1),
        "min_lag_bars": min_lag,
        "max_lag_bars": max_lag,
    }

# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE TRANSICIONES ENTRE REGÍMENES
# ──────────────────────────────────────────────────────────────────────────────

def analyze_regime_transitions(states: np.ndarray, state_summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza la matriz de transición empírica entre regímenes.
    Determina qué regímenes son "alcistas", "bajistas" o "neutrales"
    basado en sus descripciones.
    """
    unique_states = np.unique(states)
    desc_map = {}
    for _, r in state_summary.iterrows():
        desc_map[int(r["state"])] = {
            "desc": r["description"],
            "bias": _classify_regime_bias(r["description"]),
        }

    n = len(unique_states)
    trans_matrix = np.zeros((n, n))
    for i in range(1, len(states)):
        from_s = int(states[i - 1])
        to_s = int(states[i])
        if from_s < n and to_s < n:
            trans_matrix[from_s, to_s] += 1

    # Normalizar
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix_norm = np.divide(trans_matrix, row_sums, where=row_sums > 0)

    # Identificar regímenes de "cambio" (alta probabilidad de transición a otro estado)
    stability = np.diag(trans_matrix_norm) if n > 0 else np.array([])

    return {
        "transition_matrix": trans_matrix_norm.tolist(),
        "stability": stability.tolist(),
        "unique_states": n,
        "state_info": desc_map,
    }

# ──────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL REPORTE HTML
# ──────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    tf_results: Dict[str, Dict[str, Any]],
    log_lines: List[str],
) -> str:
    """Genera un reporte HTML completo con análisis de alertas de cambio de tendencia."""

    # Construir secciones por timeframe
    tf_sections = ""
    for tf, data in tf_results.items():
        cross = data["cross_ref"]
        trans = data["transitions"]
        df = data["df"]
        states = data["states"]
        state_summary = data["state_summary"]
        signal_info = data["signal_info"]
        regime_changes = data["regime_changes"]

        # ── Resumen de detección ──
        det_rate = cross["detection_rate"]
        det_color = "#089981" if det_rate >= 70 else ("#2962FF" if det_rate >= 50 else "#F23645")

        # ── Resultados tabla ──
        rows_html = ""
        for r in cross["results"]:
            if r["type"] == "correcta":
                icon = "✅"
                row_color = "rgba(8,153,129,0.08)"
                lag_str = f"{r['lag_bars']}v"
                dir_str = ""
                if r["bias_aligned"]:
                    dir_str = " <span style='color:#089981;'>✓ dirección correcta</span>"
                else:
                    dir_str = " <span style='color:#FF851B;'>⚠ dirección mixta</span>"
                desc = (
                    f"Régimen: {r['regime_change']['from_desc']} → {r['regime_change']['to_desc']} | "
                    f"Señal: {r['signal_change']['from_signal']} → {r['signal_change']['to_signal']}"
                    f"{dir_str}"
                )
            elif r["type"] == "no_regime":
                icon = "⚠️"
                row_color = "rgba(255,133,27,0.08)"
                lag_str = "—"
                desc = (
                    f"Señal: {r['signal_change']['from_signal']} → {r['signal_change']['to_signal']} "
                    f"(${r['signal_change']['price']:,.0f}) SIN cambio de régimen previo"
                )
            else:
                icon = "❌"
                row_color = "rgba(242,54,69,0.08)"
                lag_str = "—"
                desc = (
                    f"Régimen: {r['regime_change']['from_desc']} → {r['regime_change']['to_desc']} "
                    f"SIN cambio de señal"
                )

            date_str = ""
            if r.get("regime_change"):
                date_str = r["regime_change"]["date_str"]
            elif r.get("signal_change"):
                date_str = r["signal_change"]["date_str"]

            rows_html += f"""<tr style="background:{row_color}">
                <td style="text-align:center;font-size:1rem;">{icon}</td>
                <td style="font-size:0.75rem;color:#888;">{date_str}</td>
                <td style="font-size:0.75rem;">{desc}</td>
                <td style="text-align:center;font-size:0.75rem;">{lag_str}</td>
            </tr>"""

        # ── Tarjetas de régimen ──
        cards_html = ""
        for _, row in state_summary.iterrows():
            s = int(row["state"])
            bias = _classify_regime_bias(row["description"])
            bias_color = "#089981" if bias == "bullish" else ("#F23645" if bias == "bearish" else "#888")
            bias_label = "📈 Alcis." if bias == "bullish" else ("📉 Bajis." if bias == "bearish" else "➖ Neut.")
            cards_html += f"""<div class="stat-card" style="border-left:3px solid {bias_color};">
                <div class="stat-label" style="font-size:0.7rem;">{row['description']}</div>
                <div class="stat-value" style="font-size:0.85rem;color:{bias_color};">{bias_label}</div>
                <div style="font-size:0.65rem;color:#888;">
                    Ret: {row['mean_return']:+.4f}% | Vol: {row['volatility']:.2f}% | {row['pct_time']}% del tiempo
                </div>
            </div>"""

        # ── Señal actual ──
        sig = signal_info["signal"]
        sig_color = "#089981" if sig == "LONG" else ("#F23645" if sig == "SHORT" else "#FF851B")
        sig_strength = signal_info["strength"]

        # ── Gráfico de precio con regímenes (usando Plotly) ──
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               row_heights=[0.7, 0.3])
            fig.update_layout(template="plotly_dark", height=500,
                             margin=dict(l=40, r=20, t=20, b=20),
                             hovermode="x unified",
                             paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                             legend=dict(orientation="h", y=1.02, font=dict(size=9)))

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                increasing_line_color="#089981", decreasing_line_color="#F23645",
                name=ASSET, showlegend=False
            ), row=1, col=1)

            # EMA55
            if "ema_slow" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["ema_slow"], mode="lines",
                    line=dict(color="#FFD700", width=1.5, dash="dash"),
                    name="EMA 55"
                ), row=1, col=1)

            # Regímenes como fondos de color
            colors_map = {"bullish": "rgba(8,153,129,0.12)", "bearish": "rgba(242,54,69,0.12)",
                          "neutral": "rgba(136,136,136,0.08)"}
            if len(states) > 0:
                regime_shapes = []
                for _, r in state_summary.iterrows():
                    s = int(r["state"])
                    bias = _classify_regime_bias(r["description"])
                    mask = states == s
                    if mask.sum() < 2:
                        continue
                    transitions = np.where(np.diff(mask.astype(int)) != 0)[0]
                    starts = np.concatenate([[0], transitions + 1])
                    ends = np.concatenate([transitions + 1, [len(states)]])
                    for st, en in zip(starts, ends):
                        if mask[st]:
                            c = colors_map.get(bias, "rgba(136,136,136,0.08)")
                            x0 = str(df.index[st])
                            x1 = str(df.index[min(en, len(df)-1)])
                            regime_shapes.append(dict(
                                type="rect", x0=x0, x1=x1, yref="paper", y0=0, y1=1,
                                fillcolor=c, layer="below", line_width=0,
                            ))
                if regime_shapes:
                    fig.update_layout(shapes=regime_shapes)

            # Señales LONG/SHORT
            for sig_type, sym, col_sig in [("LONG", "triangle-up", "#089981"),
                                             ("SHORT", "triangle-down", "#F23645")]:
                col = f"signal_{sig_type.lower()}"
                if col not in df.columns:
                    continue
                mask = df[col].astype(bool) & (~df[col].shift(1).fillna(False).astype(bool))
                dates = df.index[mask]
                prices = df["Low"][mask] if sig_type == "LONG" else df["High"][mask]
                if len(dates) > 0:
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices, mode="markers",
                        marker=dict(symbol=sym, size=10, color=col_sig,
                                   line=dict(width=1, color="white")),
                        name=f"Señal {sig_type}"
                    ), row=1, col=1)

            # Volumen
            vol_max = df["Volume"].max() if df["Volume"].max() > 0 else 1
            vol_colors = ["#089981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#F23645"
                         for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"] / vol_max * 100,
                marker_color=vol_colors, name="Volumen", opacity=0.5, showlegend=False
            ), row=2, col=1)

            fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=2, col=1)
            fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volumen (%)", row=2, col=1, range=[0, 110])

            chart_html = fig.to_html(full_html=False, include_plotlyjs=False,
                                    div_id=f"chart-{tf}",
                                    config=dict(scrollZoom=True, displaylogo=False,
                                               modeBarButtonsToRemove=["lasso2d", "select2d"]))
        except Exception as e:
            chart_html = f'<div style="color:#F23645;padding:20px;">Error al generar gráfico: {e}</div>'

        # ── Matriz de transición (como tabla HTML) ──
        trans_html = ""
        if trans["unique_states"] > 0:
            trans_html = '<table class="trans-table">'
            trans_html += '<tr><th>Desde \\ Hacia</th>'
            for s in range(trans["unique_states"]):
                info = trans["state_info"].get(s, {})
                desc = info.get("desc", f"R{s}")
                trans_html += f'<th style="font-size:0.6rem;">{desc}</th>'
            trans_html += '</tr>'
            for i in range(trans["unique_states"]):
                info_i = trans["state_info"].get(i, {})
                desc_i = info_i.get("desc", f"R{i}")
                trans_html += f'<tr><td style="font-size:0.6rem;font-weight:600;">{desc_i}</td>'
                for j in range(trans["unique_states"]):
                    val = trans["transition_matrix"][i][j]
                    pct = val * 100
                    if i == j:
                        color = "#089981" if pct > 50 else ("#FF851B" if pct > 30 else "#888")
                    else:
                        color = "#F23645" if pct > 10 else "#666"
                    trans_html += f'<td style="text-align:center;font-size:0.7rem;color:{color};">{pct:.0f}%</td>'
                trans_html += '</tr>'
            trans_html += '</table>'

        # ── Cambios de régimen recientes ──
        regime_html = ""
        for rc in regime_changes[-10:]:
            from_bias = rc.get("from_bias", "neutral")
            to_bias = rc.get("to_bias", "neutral")
            arrow_color = "#089981" if to_bias == "bullish" else ("#F23645" if to_bias == "bearish" else "#888")
            regime_html += (
                f'<div style="font-size:0.7rem;padding:4px 8px;border-bottom:1px solid #2a2a34;">'
                f'<span style="color:#888;">{rc["date_str"]}</span> '
                f'{rc["from_desc"]} <span style="color:{arrow_color};">→</span> {rc["to_desc"]} '
                f'<span style="color:#666;">({rc["duration_velas"]}v)</span>'
                f'</div>'
            )

        # ── Sección del timeframe ──
        regime_changes_count = len(regime_changes)
        signal_changes_count = cross["total_signal_changes"]
        tf_sections += f"""
        <div class="tf-section" id="tf-{tf}">
            <div class="tf-header">
                <span class="tf-title">Timeframe: {tf}</span>
                <span class="tf-signal" style="background:{sig_color};">
                    {sig} {sig_strength}%
                </span>
            </div>

            <!-- KPI Cards -->
            <div class="kpi-grid">
                <div class="kpi-card" style="border-top:3px solid #2962FF;">
                    <div class="kpi-value">{regime_changes_count}</div>
                    <div class="kpi-label">Cambios de Régimen</div>
                </div>
                <div class="kpi-card" style="border-top:3px solid #FF851B;">
                    <div class="kpi-value">{signal_changes_count}</div>
                    <div class="kpi-label">Cambios de Señal</div>
                </div>
                <div class="kpi-card" style="border-top:3px solid {det_color};">
                    <div class="kpi-value" style="color:{det_color};">{cross['detected_correctly']}/{signal_changes_count}</div>
                    <div class="kpi-label">Detectados ({det_rate}%)</div>
                </div>
                <div class="kpi-card" style="border-top:3px solid {'#089981' if cross['false_positives'] == 0 else '#F23645'};">
                    <div class="kpi-value" style="color:{'#089981' if cross['false_positives'] == 0 else '#F23645'};">{cross['false_positives']}</div>
                    <div class="kpi-label">Falsos Positivos</div>
                </div>
                <div class="kpi-card" style="border-top:3px solid {'#089981' if cross['false_negatives'] == 0 else '#FF851B'};">
                    <div class="kpi-value" style="color:{'#089981' if cross['false_negatives'] == 0 else '#FF851B'};">{cross['false_negatives']}</div>
                    <div class="kpi-label">Falsos Negativos</div>
                </div>
                <div class="kpi-card" style="border-top:3px solid #2962FF;">
                    <div class="kpi-value">{cross['avg_lag_bars']}v</div>
                    <div class="kpi-label">Antelación Promedio</div>
                </div>
            </div>

            <!-- Detail grid -->
            <div class="detail-grid">
                <div class="detail-card">
                    <div class="card-title">📊 Gráfico de Precio con Regímenes y Señales</div>
                    <div style="color:#888;font-size:0.7rem;margin-bottom:8px;">
                        Fondos: 🟢 Alcista | 🔴 Bajista | ⬜ Neutral | ▼ Señales LONG/SHORT
                    </div>
                    {chart_html}
                </div>

                <div class="detail-card">
                    <div class="card-title">🏛️ Regímenes Detectados</div>
                    <div class="stats-grid">{cards_html}</div>
                </div>

                <div class="detail-card">
                    <div class="card-title">🔄 Matriz de Transición entre Regímenes</div>
                    <div style="color:#888;font-size:0.65rem;margin-bottom:8px;">
                        Probabilidad de pasar del régimen de la fila al de la columna
                    </div>
                    {trans_html}
                </div>

                <div class="detail-card">
                    <div class="card-title">📋 Últimos Cambios de Régimen</div>
                    <div style="max-height:400px;overflow-y:auto;">
                        {regime_html if regime_html else '<div style="color:#666;font-size:0.75rem;padding:12px;">Sin cambios detectados</div>'}
                    </div>
                </div>
            </div>

            <!-- Analysis Results Table -->
            <div class="detail-card" style="margin-top:16px;">
                <div class="card-title">🔍 Análisis Cruzado: Régimen vs Señal</div>
                <div style="overflow-x:auto;">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th style="width:40px;"></th>
                                <th style="width:100px;">Fecha</th>
                                <th>Descripción</th>
                                <th style="width:80px;">Lag</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html if rows_html else '<tr><td colspan="4" style="text-align:center;color:#666;padding:20px;">Sin resultados para mostrar</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """

    # ── Resumen global ──
    total_detected = sum(d["cross_ref"]["detected_correctly"] for d in tf_results.values())
    total_signals = sum(d["cross_ref"]["total_signal_changes"] for d in tf_results.values())
    total_false_pos = sum(d["cross_ref"]["false_positives"] for d in tf_results.values())
    total_false_neg = sum(d["cross_ref"]["false_negatives"] for d in tf_results.values())
    total_bias_aligned = sum(d["cross_ref"]["bias_aligned"] for d in tf_results.values())
    global_rate = round(total_detected / total_signals * 100, 1) if total_signals else 0

    # Tabla de navegación
    tabs = ""
    for i, (tf, _) in enumerate(tf_results.items()):
        active = "active" if i == 0 else ""
        tabs += f'<button class="tf-tab {active}" onclick="showTF(\'{tf}\')">{tf}</button>'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulación Alertas de Tendencia — {ASSET}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f0f13; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 24px; padding: 20px;
                  background: linear-gradient(135deg, #1a1a24, #16161e);
                  border: 1px solid #2a2a34; border-radius: 12px; }}
        .header h1 {{ font-size: 1.5rem; color: #fff; margin-bottom: 8px; }}
        .header h1 span {{ color: #2962FF; }}
        .header .subtitle {{ font-size: 0.85rem; color: #888; }}
        .header .date {{ font-size: 0.7rem; color: #666; margin-top: 4px; }}

        /* Global KPI */
        .global-kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                       gap: 12px; margin-bottom: 20px; }}
        .global-card {{ background: #1a1a24; border: 1px solid #2a2a34; border-radius: 10px;
                        padding: 16px; text-align: center; }}
        .global-value {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .global-label {{ font-size: 0.7rem; color: #888; text-transform: uppercase; margin-top: 4px; }}

        /* Tabs */
        .tf-tabs {{ display: flex; gap: 4px; margin-bottom: 16px; }}
        .tf-tab {{ background: #1e1e24; color: #888; border: 1px solid #333;
                  padding: 8px 24px; border-radius: 8px; cursor: pointer;
                  font-size: 0.85rem; font-weight: 600; transition: all 0.2s; }}
        .tf-tab:hover {{ background: #2a2a32; color: #fff; }}
        .tf-tab.active {{ background: #2962FF; color: #fff; border-color: #2962FF; }}
        .tf-section {{ display: none; animation: fadeIn 0.3s ease; }}
        .tf-section.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        /* TF Header */
        .tf-header {{ display: flex; justify-content: space-between; align-items: center;
                      background: #1a1a24; border: 1px solid #2a2a34; border-radius: 10px;
                      padding: 16px 20px; margin-bottom: 16px; }}
        .tf-title {{ font-size: 1.2rem; font-weight: 700; color: #fff; }}
        .tf-signal {{ padding: 6px 20px; border-radius: 6px; font-size: 1rem;
                     font-weight: 700; color: #fff; }}

        /* KPI Grid */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                     gap: 10px; margin-bottom: 16px; }}
        .kpi-card {{ background: #1a1a24; border: 1px solid #2a2a34; border-radius: 8px;
                     padding: 12px; }}
        .kpi-value {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}
        .kpi-label {{ font-size: 0.6rem; color: #888; text-transform: uppercase; margin-top: 2px; }}

        /* Detail Grid */
        .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 900px) {{ .detail-grid {{ grid-template-columns: 1fr; }} }}
        .detail-card {{ background: #1a1a24; border: 1px solid #2a2a34; border-radius: 10px;
                        padding: 16px; overflow: hidden; }}

        /* Stats */
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
        .stat-card {{ background: rgba(255,255,255,0.02); border-radius: 6px; padding: 8px 10px; }}
        .stat-label {{ color: #aaa; font-weight: 600; }}
        .stat-value {{ font-weight: 700; }}

        /* Table */
        .results-table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; }}
        .results-table th {{ background: #1e1e28; color: #888; padding: 8px 10px;
                             text-align: left; font-weight: 600; text-transform: uppercase;
                             font-size: 0.65rem; border-bottom: 1px solid #333; }}
        .results-table td {{ padding: 6px 10px; border-bottom: 1px solid #2a2a34; }}
        .results-table tr:hover td {{ background: rgba(255,255,255,0.03); }}

        .trans-table {{ width: 100%; border-collapse: collapse; }}
        .trans-table th, .trans-table td {{ border: 1px solid #2a2a34; padding: 6px 8px; }}

        .card-title {{ font-size: 0.8rem; font-weight: 600; color: #aaa; margin-bottom: 12px;
                       text-transform: uppercase; letter-spacing: 0.5px; }}

        /* Log */
        .log-section {{ background: #0a0a0e; border: 1px solid #2a2a34; border-radius: 8px;
                        padding: 16px; margin-top: 20px; max-height: 300px; overflow-y: auto;
                        font-family: monospace; font-size: 0.7rem; color: #888; }}
        .log-section .log-title {{ color: #aaa; font-weight: 600; font-size: 0.75rem;
                                   margin-bottom: 8px; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Simulación · <span>Alertas de Cambio de Tendencia</span></h1>
            <div class="subtitle">{ASSET} — Análisis de detección de cambios LONG ↔ SHORT mediante regímenes HMM</div>
            <div class="date">Generado: {datetime.now().strftime("%d-%m-%Y %H:%M")} | Períodos: {json.dumps(PERIODS)}</div>
        </div>

        <!-- Global KPIs -->
        <div class="global-kpi">
            <div class="global-card">
                <div class="global-value">{total_signals}</div>
                <div class="global-label">Cambios de Señal Totales</div>
            </div>
            <div class="global-card">
                <div class="global-value" style="color:{'#089981' if global_rate >= 70 else ('#2962FF' if global_rate >= 50 else '#F23645')};">{global_rate}%</div>
                <div class="global-label">Tasa de Detección Global</div>
            </div>
            <div class="global-card">
                <div class="global-value" style="color:{'#089981' if total_false_pos == 0 else '#F23645'};">{total_false_pos}</div>
                <div class="global-label">Falsos Positivos (régimen sin señal)</div>
            </div>
            <div class="global-card">
                <div class="global-value" style="color:{'#089981' if total_false_neg == 0 else '#FF851B'};">{total_false_neg}</div>
                <div class="global-label">Falsos Negativos (señal sin régimen)</div>
            </div>
            <div class="global-card">
                <div class="global-value">{total_bias_aligned}/{total_detected}</div>
                <div class="global-label">Dirección Correcta</div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tf-tabs">
            {tabs}
        </div>

        <!-- Timeframe Sections -->
        {tf_sections}

        <!-- Log -->
        <div class="log-section">
            <div class="log-title">📋 Log de Ejecución</div>
            {"<br>".join(log_lines[-50:])}
        </div>
    </div>

    <script>
        function showTF(tf) {{
            document.querySelectorAll('.tf-section').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tf-tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tf-' + tf).classList.add('active');
            document.querySelector(`button[onclick="showTF('${{tf}}')\"]`).classList.add('active');
            // Resize Plotly
            const tab = document.getElementById('tf-' + tf);
            if (tab) {{
                tab.querySelectorAll('.js-plotly-plot').forEach(p => {{
                    try {{ Plotly.Plots.resize(p); }} catch(e) {{}}
                }});
            }}
        }}
        // Activar primer tab
        document.addEventListener('DOMContentLoaded', () => {{
            const first = document.querySelector('.tf-tab');
            if (first) first.click();
        }});
    </script>
</body>
</html>"""
    return html

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Ejecuta el pipeline completo de simulación y genera el reporte."""
    log_lines: List[str] = []

    _log("=" * 70, log_lines)
    _log("  SIMULACIÓN · VALIDACIÓN DE ALERTAS DE CAMBIO DE TENDENCIA", log_lines)
    _log("=" * 70, log_lines)
    _log(f"  Activo: {ASSET}", log_lines)
    _log(f"  Timeframes: {TIMEFRAMES}", log_lines)
    _log(f"  Períodos: {json.dumps(PERIODS)}", log_lines)
    _log("", log_lines)

    tf_results: Dict[str, Dict[str, Any]] = {}

    for tf in TIMEFRAMES:
        _log(f"{'-' * 60}", log_lines)
        _log(f"  TIMEFRAME: {tf}", log_lines)
        _log(f"{'-' * 60}", log_lines)

        # ── 1) Cargar datos ──
        _log(f"  Descargando datos ({PERIODS[tf]})...", log_lines)
        df = load_data_extended(ASSET, tf)
        if df is None or len(df) < 100:
            _log(f"  ERROR: Datos insuficientes ({len(df) if df is not None else 0} velas).", log_lines)
            continue
        _log(f"  {len(df)} velas cargadas ({df.index[0].strftime('%d-%m-%Y')} → {df.index[-1].strftime('%d-%m-%Y')}).", log_lines)

        # ── 2) Calcular indicadores ──
        _log(f"  Calculando indicadores...", log_lines)
        df = compute_all_indicators(df)

        # ── 3) HMM ──
        _log(f"  Construyendo features y ajustando HMM...", log_lines)
        features = build_hmm_features(df)
        model, states, state_summary, bic_df, trans_mat = fit_hmm(features)

        if model is None or len(states) == 0:
            _log(f"  ERROR: HMM falló.", log_lines)
            continue

        df = df.iloc[:len(states)].copy()
        df["regime"] = states

        # ── 4) Señal actual ──
        signal_info = compute_signal(df, timeframe=tf)
        _log(f"  Señal actual: {signal_info['signal']} (fuerza: {signal_info['strength']}%)", log_lines)

        # ── 5) Cambios de régimen ──
        regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
        _log(f"  Cambios de régimen detectados: {len(regime_changes)}", log_lines)

        # ── 6) Cambios de señal ──
        signal_changes = find_signal_changes(df)
        signal_changes_real = [s for s in signal_changes if s["from_signal"] != s["to_signal"]]
        _log(f"  Cambios de señal: {len(signal_changes_real)}", log_lines)

        # ── 7) Análisis cruzado (régimen vs señal) ──
        cross_ref = cross_reference_changes(regime_changes, signal_changes, max_lag_bars=5)
        _log(f"  ✅ Detectados correctamente: {cross_ref['detected_correctly']}/{cross_ref['total_signal_changes']} ({cross_ref['detection_rate']}%)", log_lines)
        _log(f"  ❌ Falsos positivos (régimen sin señal): {cross_ref['false_positives']}", log_lines)
        _log(f"  ⚠️  Falsos negativos (señal sin régimen): {cross_ref['false_negatives']}", log_lines)
        _log(f"  📐 Antelación promedio: {cross_ref['avg_lag_bars']} velas", log_lines)
        _log(f"  🎯 Dirección correcta: {cross_ref['bias_aligned']}/{cross_ref['detected_correctly']}", log_lines)

        # ── 8) Transiciones ──
        transitions = analyze_regime_transitions(states, state_summary)

        # ── 9) Verificación histórica (win rate) ──
        verification = verify_signals_historically(df, tf)
        if verification and verification["total_signals"] > 0:
            _log(f"  📊 WR Global: {verification['overall_win_rate']:.1f}% ({verification['total_signals']} señales)", log_lines)
            for side in ["LONG", "SHORT"]:
                s = verification["stats"].get(side, {})
                if s.get("num_signals", 0) > 0:
                    _log(f"      {side}: {s['win_rate']:.1f}% WR ({s['num_signals']} señales, ret {s['avg_return']:+.2f}%)", log_lines)

        tf_results[tf] = {
            "df": df,
            "states": states,
            "state_summary": state_summary,
            "signal_info": signal_info,
            "regime_changes": regime_changes,
            "signal_changes": signal_changes,
            "cross_ref": cross_ref,
            "transitions": transitions,
            "verification": verification,
        }

    # ── Generar HTML ──
    _log(f"{'=' * 60}", log_lines)
    _log(f"  Generando reporte HTML...", log_lines)

    if not tf_results:
        _log("  ERROR: No hay resultados para ningún timeframe.", log_lines)
        return

    html = generate_html_report(tf_results, log_lines)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    _log(f"  Reporte guardado: {OUTPUT_HTML}", log_lines)

    # Guardar log
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    _log(f"  Log guardado: {OUTPUT_LOG}", log_lines)

    # Resumen final
    _log("", log_lines)
    _log("=" * 60, log_lines)
    _log("  RESUMEN GLOBAL", log_lines)
    _log("=" * 60, log_lines)
    for tf, data in tf_results.items():
        c = data["cross_ref"]
        _log(f"  {tf}: HMM={c['detected_correctly']}/{c['total_signal_changes']} ({c['detection_rate']}%) | "
             f"FP:{c['false_positives']} FN:{c['false_negatives']} | "
             f"Ant: {c['avg_lag_bars']}v | Dir: {c['bias_aligned']}/{c['detected_correctly']}", log_lines)

    total_det = sum(d["cross_ref"]["detected_correctly"] for d in tf_results.values())
    total_sig = sum(d["cross_ref"]["total_signal_changes"] for d in tf_results.values())
    total_rate = round(total_det / total_sig * 100, 1) if total_sig else 0
    _log("", log_lines)
    _log(f"  TASA DE DETECCIÓN GLOBAL: {total_rate}% ({total_det}/{total_sig})", log_lines)
    _log("", log_lines)
    _log(f"  Reporte: {OUTPUT_HTML}", log_lines)
    _log(f"  Log: {OUTPUT_LOG}", log_lines)

    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(OUTPUT_HTML)}")
        _log("  Reporte abierto en el navegador.", log_lines)
    except Exception:
        _log(f"  Abre manualmente: {OUTPUT_HTML}", log_lines)

    _log("=" * 60, log_lines)
    _log("  COMPLETADO", log_lines)
    _log("=" * 60, log_lines)

if __name__ == "__main__":
    run_pipeline()
