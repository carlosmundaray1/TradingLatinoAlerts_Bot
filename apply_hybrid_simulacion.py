#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica los cambios necesarios a simulacion_alertas_tendencia.py para incluir
la detección del sistema híbrido HMM+Precursor en el reporte HTML.
"""
import sys

FILE = "simulacion_alertas_tendencia.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ============================================================================
# CHANGE 1: Add compute_hybrid_alert to imports
# ============================================================================
old_import = """verify_signals_historically = _HMM.verify_signals_historically"""
new_import = """verify_signals_historically = _HMM.verify_signals_historically
compute_hybrid_alert = _HMM.compute_hybrid_alert
HYBRID_CONFIDENCE_THRESHOLD = _HMM.HYBRID_CONFIDENCE_THRESHOLD"""

if old_import in content:
    content = content.replace(old_import, new_import)
    changes += 1
    print(f"[OK] Change 1: Added compute_hybrid_alert import")
else:
    print(f"[FAIL] Change 1: Import section not found")

# ============================================================================
# CHANGE 2: Add cross_reference_hybrid() function after cross_reference_changes()
# ============================================================================
old_func_end = """    }


def analyze_regime_transitions(states: np.ndarray, state_summary: pd.DataFrame) -> Dict[str, Any]:"""

new_functions = """    }


# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS HÍBRIDO: ALERTAS HÍBRIDAS vs CAMBIOS DE SEÑAL
# ──────────────────────────────────────────────────────────────────────────────

def find_hybrid_alerts(df: pd.DataFrame) -> List[Dict]:
    \"\"\"
    Encuentra activaciones de alertas híbridas (hybrid_alert_active).
    Retorna lista con índices, fechas, dirección y confianza de cada alerta.
    \"\"\"
    alerts = []
    prev_active = False
    for i in range(len(df)):
        is_active = bool(df[\"hybrid_alert_active\"].iloc[i]) if \"hybrid_alert_active\" in df.columns else False
        is_long = bool(df[\"hybrid_alert_long\"].iloc[i]) if \"hybrid_alert_long\" in df.columns else False
        is_short = bool(df[\"hybrid_alert_short\"].iloc[i]) if \"hybrid_alert_short\" in df.columns else False
        conf_long = float(df[\"hybrid_confidence_long\"].iloc[i]) if \"hybrid_confidence_long\" in df.columns else 0
        conf_short = float(df[\"hybrid_confidence_short\"].iloc[i]) if \"hybrid_confidence_short\" in df.columns else 0

        if is_active and not prev_active:
            direction = \"LONG\" if is_long else \"SHORT\"
            confidence = conf_long if is_long else conf_short
            alerts.append({
                \"idx\": i,
                \"date\": df.index[i],
                \"date_str\": _format_date(df.index[i]),
                \"direction\": direction,
                \"confidence\": confidence,
                \"price\": float(df[\"Close\"].iloc[i]),
            })
        prev_active = is_active

    # También encontrar alertas que persisten (activas por N+ velas)
    return alerts


def find_hybrid_active_periods(df: pd.DataFrame) -> List[Dict]:
    \"\"\"
    Encuentra períodos donde hybrid_alert_long/short están activos.
    Retorna lista con inicio, fin, dirección, y confianza máxima del período.
    \"\"\"
    periods = []
    in_long = False
    in_short = False
    long_start = 0
    short_start = 0

    for i in range(len(df)):
        is_long = bool(df[\"hybrid_alert_long\"].iloc[i]) if \"hybrid_alert_long\" in df.columns else False
        is_short = bool(df[\"hybrid_alert_short\"].iloc[i]) if \"hybrid_alert_short\" in df.columns else False

        if is_long and not in_long:
            in_long = True
            long_start = i
        if not is_long and in_long:
            in_long = False
            conf_vals = df[\"hybrid_confidence_long\"].iloc[long_start:i].values
            max_conf = float(np.max(conf_vals)) if len(conf_vals) > 0 else 0
            periods.append({
                \"type\": \"LONG\",
                \"start_idx\": long_start,
                \"end_idx\": i - 1,
                \"start_date\": _format_date(df.index[long_start]),
                \"end_date\": _format_date(df.index[i - 1]),
                \"duration_bars\": i - long_start,
                \"max_confidence\": max_conf,
            })

        if is_short and not in_short:
            in_short = True
            short_start = i
        if not is_short and in_short:
            in_short = False
            conf_vals = df[\"hybrid_confidence_short\"].iloc[short_start:i].values
            max_conf = float(np.max(conf_vals)) if len(conf_vals) > 0 else 0
            periods.append({
                \"type\": \"SHORT\",
                \"start_idx\": short_start,
                \"end_idx\": i - 1,
                \"start_date\": _format_date(df.index[short_start]),
                \"end_date\": _format_date(df.index[i - 1]),
                \"duration_bars\": i - short_start,
                \"max_confidence\": max_conf,
            })

    return periods


def cross_reference_hybrid(
    df: pd.DataFrame,
    signal_changes: List[Dict],
    max_lag_bars: int = 5
) -> Dict[str, Any]:
    \"\"\"
    Cruza las alertas híbridas con los cambios de señal para determinar:
    - Detecciones correctas: alerta híbrida activa dentro de N velas antes del cambio de señal
    - Dirección alineada: la alerta fue en la misma dirección que el cambio de señal
    - Anticipación: cuántas velas antes se activó la alerta híbrida
    - Falsos negativos: cambios de señal SIN alerta híbrida previa
    \"\"\"
    if \"hybrid_alert_active\" not in df.columns:
        return {
            \"detected_correctly\": 0,
            \"total_signal_changes\": len(signal_changes),
            \"detection_rate\": 0.0,
            \"false_negatives\": len(signal_changes),
            \"bias_aligned\": 0,
            \"avg_lag_bars\": 0,
            \"avg_confidence\": 0.0,
            \"results\": [],
        }

    results = []
    signal_changes_real = [s for s in signal_changes if s[\"from_signal\"] != s[\"to_signal\"]]

    for sc in signal_changes_real:
        idx_start = max(0, sc[\"idx\"] - max_lag_bars)
        idx_end = sc[\"idx\"]

        # Buscar alerta híbrida en la ventana previa al cambio de señal
        window = df.iloc[idx_start:idx_end + 1]
        hybrid_active = window[\"hybrid_alert_active\"].any() if \"hybrid_alert_active\" in window.columns else False

        if hybrid_active:
            # Encontrar la alerta más cercana al cambio
            alert_indices = window.index[window[\"hybrid_alert_active\"]].tolist()
            closest_idx = alert_indices[-1]  # Más cercano al cambio
            lag = sc[\"idx\"] - list(df.index).index(closest_idx)

            # Verificar dirección
            to_sig = sc[\"to_signal\"]
            hybrid_long = window[\"hybrid_alert_long\"].any() if \"hybrid_alert_long\" in window.columns else False
            hybrid_short = window[\"hybrid_alert_short\"].any() if \"hybrid_alert_short\" in window.columns else False
            direction_aligned = (to_sig == \"LONG\" and hybrid_long) or (to_sig == \"SHORT\" and hybrid_short)

            # Confianza máxima en la ventana
            conf_long_vals = window[\"hybrid_confidence_long\"].values if \"hybrid_confidence_long\" in window.columns else [0]
            conf_short_vals = window[\"hybrid_confidence_short\"].values if \"hybrid_confidence_short\" in window.columns else [0]
            max_conf = max(float(np.max(conf_long_vals)) if len(conf_long_vals) > 0 else 0,
                          float(np.max(conf_short_vals)) if len(conf_short_vals) > 0 else 0)

            results.append({
                \"type\": \"correcta\",
                \"signal_change\": sc,
                \"lag_bars\": lag,
                \"bias_aligned\": direction_aligned,
                \"max_confidence\": max_conf,
                \"description\": (
                    f\"Híbrido alertó {lag} vela(s) antes: {sc['from_signal']}→{sc['to_signal']} \" + 
                    (\"✓ dirección correcta\" if direction_aligned else \"⚠ dirección mixta\")
                ),
            })
        else:
            results.append({
                \"type\": \"no_hybrid\",
                \"signal_change\": sc,
                \"lag_bars\": None,
                \"bias_aligned\": None,
                \"max_confidence\": 0,
                \"description\": (
                    f\"Señal cambió {sc['from_signal']}→{sc['to_signal']} \" +
                    f\"SIN alerta híbrida previa en {max_lag_bars} velas\"
                ),
            })

    # Estadísticas
    total_signal = len(signal_changes_real)
    detected = sum(1 for r in results if r[\"type\"] == \"correcta\")
    false_neg = sum(1 for r in results if r[\"type\"] == \"no_hybrid\")
    bias_aligned = sum(1 for r in results if r.get(\"bias_aligned\"))
    lags = [r[\"lag_bars\"] for r in results if r[\"lag_bars\"] is not None]
    avg_lag = float(np.mean(lags)) if lags else 0
    confs = [r[\"max_confidence\"] for r in results if r[\"max_confidence\"] > 0]
    avg_conf = float(np.mean(confs)) if confs else 0

    return {
        \"results\": results,
        \"total_signal_changes\": total_signal,
        \"detected_correctly\": detected,
        \"false_negatives\": false_neg,
        \"bias_aligned\": bias_aligned,
        \"detection_rate\": round(detected / total_signal * 100, 1) if total_signal else 0,
        \"avg_lag_bars\": round(avg_lag, 1),
        \"avg_confidence\": round(avg_conf, 1),
    }


def analyze_regime_transitions(states: np.ndarray, state_summary: pd.DataFrame) -> Dict[str, Any]:"""

if old_func_end in content:
    content = content.replace(old_func_end, new_functions)
    changes += 1
    print(f"[OK] Change 2: Added cross_reference_hybrid() and related functions")
else:
    print(f"[FAIL] Change 2: Function end marker not found")
    # Print context for debugging
    idx = content.find("def analyze_regime_transitions")
    if idx >= 0:
        print(f"  Context: {content[idx-200:idx]}")

# ============================================================================
# CHANGE 3: Update run_pipeline() to compute hybrid alerts
# ============================================================================
# Find the section after compute_signal and before signal_changes

old_pipeline_1 = """        # ── 5) Cambios de régimen ──
        regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
        _log(f\"  Cambios de régimen detectados: {len(regime_changes)}\", log_lines)

        # ── 6) Cambios de señal ──
        signal_changes = find_signal_changes(df)
        signal_changes_real = [s for s in signal_changes if s[\"from_signal\"] != s[\"to_signal\"]]
        _log(f\"  Cambios de señal: {len(signal_changes_real)}\", log_lines)

        # ── 7) Análisis cruzado ──"""

new_pipeline_1 = """        # ── 5) Computar alerta híbrida HMM+Precursor ──
        _log(f\"  Calculando alerta híbrida HMM+Precursor...\", log_lines)
        df = compute_hybrid_alert(df, states, state_summary)
        hybrid_alerts = find_hybrid_alerts(df)
        hybrid_periods = find_hybrid_active_periods(df)
        _log(f\"    Alertas híbridas activadas: {len(hybrid_alerts)}\", log_lines)
        _log(f\"    Períodos activos: {len(hybrid_periods)}\", log_lines)

        # ── 6) Cambios de régimen ──
        regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
        _log(f\"  Cambios de régimen detectados: {len(regime_changes)}\", log_lines)

        # ── 7) Cambios de señal ──
        signal_changes = find_signal_changes(df)
        signal_changes_real = [s for s in signal_changes if s[\"from_signal\"] != s[\"to_signal\"]]
        _log(f\"  Cambios de señal: {len(signal_changes_real)}\", log_lines)

        # ── 8) Análisis cruzado (régimen vs señal) ──"""

if old_pipeline_1 in content:
    content = content.replace(old_pipeline_1, new_pipeline_1)
    changes += 1
    print(f"[OK] Change 3: Added hybrid alert computation in pipeline")
else:
    print(f"[FAIL] Change 3: Pipeline section not found")

# ============================================================================
# CHANGE 4: Add hybrid cross_reference and hybrid data to tf_results
# ============================================================================
old_pipeline_2 = """        # ── 8) Transiciones ──
        transitions = analyze_regime_transitions(states, state_summary)"""

new_pipeline_2 = """        # ── 9) Análisis cruzado híbrido ──
        hybrid_cross_ref = cross_reference_hybrid(df, signal_changes, max_lag_bars=5)
        _log(f\"    Híbrido detectó: {hybrid_cross_ref['detected_correctly']}/{hybrid_cross_ref['total_signal_changes']} ({hybrid_cross_ref['detection_rate']}%)\", log_lines)
        _log(f\"    Dirección correcta: {hybrid_cross_ref['bias_aligned']}/{hybrid_cross_ref['detected_correctly']}\", log_lines)
        _log(f\"    Confianza promedio: {hybrid_cross_ref['avg_confidence']}%\", log_lines)
        _log(f\"    Antelación promedio: {hybrid_cross_ref['avg_lag_bars']} velas\", log_lines)

        # ── 10) Transiciones ──
        transitions = analyze_regime_transitions(states, state_summary)"""

if old_pipeline_2 in content:
    content = content.replace(old_pipeline_2, new_pipeline_2)
    changes += 1
    print(f"[OK] Change 4: Added hybrid cross-reference in pipeline")
else:
    print(f"[FAIL] Change 4: Transitions section not found")

# ============================================================================
# CHANGE 5: Add hybrid data to tf_results dict
# ============================================================================
old_tf_results = """        tf_results[tf] = {
            \"df\": df,
            \"states\": states,
            \"state_summary\": state_summary,
            \"signal_info\": signal_info,
            \"regime_changes\": regime_changes,
            \"signal_changes\": signal_changes,
            \"cross_ref\": cross_ref,
            \"transitions\": transitions,
            \"verification\": verification,
        }"""

new_tf_results = """        tf_results[tf] = {
            \"df\": df,
            \"states\": states,
            \"state_summary\": state_summary,
            \"signal_info\": signal_info,
            \"regime_changes\": regime_changes,
            \"signal_changes\": signal_changes,
            \"cross_ref\": cross_ref,
            \"hybrid_alerts\": hybrid_alerts,
            \"hybrid_periods\": hybrid_periods,
            \"hybrid_cross_ref\": hybrid_cross_ref,
            \"transitions\": transitions,
            \"verification\": verification,
        }"""

if old_tf_results in content:
    content = content.replace(old_tf_results, new_tf_results)
    changes += 1
    print(f"[OK] Change 5: Added hybrid data to tf_results")
else:
    print(f"[FAIL] Change 5: tf_results section not found")

# ============================================================================
# CHANGE 6: Add hybrid summary to global resumen
# ============================================================================
old_resumen = """    for tf, data in tf_results.items():
        c = data[\"cross_ref\"]
        _log(f\"  {tf}: {c['detected_correctly']}/{c['total_signal_changes']} detectados | \"
             f\"FP:{c['false_positives']} FN:{c['false_negatives']} | \"
             f\"Antelación: {c['avg_lag_bars']}v | Dir: {c['bias_aligned']}/{c['detected_correctly']}\", log_lines)"""

new_resumen = """    for tf, data in tf_results.items():
        c = data[\"cross_ref\"]
        hc = data.get(\"hybrid_cross_ref\", {})
        hybrid_rate = hc.get(\"detection_rate\", 0)
        _log(f\"  {tf}: HMM={c['detected_correctly']}/{c['total_signal_changes']} ({c['detection_rate']}%) | \"
             f\"Híbrido={hc.get('detected_correctly',0)}/{hc.get('total_signal_changes',0)} ({hybrid_rate}%) | \"
             f\"FP:{c['false_positives']} FN:{c['false_negatives']} | \"
             f\"Antelación: {c['avg_lag_bars']}v | Dir: {c['bias_aligned']}/{c['detected_correctly']}\", log_lines)"""

if old_resumen in content:
    content = content.replace(old_resumen, new_resumen)
    changes += 1
    print(f"[OK] Change 6: Added hybrid summary to resumen")
else:
    print(f"[FAIL] Change 6: Resumen section not found")

# ============================================================================
# CHANGE 7: Update generate_html_report() to include hybrid detection section
# ============================================================================
# Add hybrid KPI cards after existing KPIs in each timeframe section
old_kpi_end = """                <div class=\"kpi-card\" style=\"border-top:3px solid #2962FF;\">
                    <div class=\"kpi-value\">{cross['avg_lag_bars']}v</div>
                    <div class=\"kpi-label\">Antelación Promedio</div>
                </div>
            </div>"""

new_kpi_end = """                <div class=\"kpi-card\" style=\"border-top:3px solid #2962FF;\">
                    <div class=\"kpi-value\">{cross['avg_lag_bars']}v</div>
                    <div class=\"kpi-label\">Antelación Promedio</div>
                </div>
            </div>

            <!-- Hybrid Alert KPI Cards -->
            <div class=\"detail-card\" style=\"margin-top:16px;border-left:3px solid #9B59B6;\">
                <div class=\"card-title\" style=\"color:#9B59B6;\">🧬 SISTEMA HÍBRIDO: HMM + Precursores</div>
                <div class=\"kpi-grid\" style=\"margin-top:8px;\">
                    <div class=\"kpi-card\" style=\"border-top:3px solid #9B59B6;\">
                        <div class=\"kpi-value\" style=\"color:#9B59B6;\">{hc['detected_correctly']}/{hc['total_signal_changes']}</div>
                        <div class=\"kpi-label\">Detectados por Híbrido</div>
                    </div>
                    <div class=\"kpi-card\" style=\"border-top:3px solid {'#089981' if hc['detection_rate'] >= 70 else ('#2962FF' if hc['detection_rate'] >= 50 else '#F23645')};\">
                        <div class=\"kpi-value\" style=\"color:{'#089981' if hc['detection_rate'] >= 70 else ('#2962FF' if hc['detection_rate'] >= 50 else '#F23645')};\">{hc['detection_rate']}%</div>
                        <div class=\"kpi-label\">Tasa Detección Híbrida</div>
                    </div>
                    <div class=\"kpi-card\" style=\"border-top:3px solid #8E44AD;\">
                        <div class=\"kpi-value\" style=\"color:#8E44AD;\">{hc['avg_confidence']}%</div>
                        <div class=\"kpi-label\">Confianza Promedio</div>
                    </div>
                    <div class=\"kpi-card\" style=\"border-top:3px solid {'#089981' if hc['bias_aligned'] >= hc['detected_correctly'] * 0.6 else '#FF851B'};\">
                        <div class=\"kpi-value\" style=\"color:{'#089981' if hc['bias_aligned'] >= hc['detected_correctly'] * 0.6 else '#FF851B'};\">{hc['bias_aligned']}/{hc['detected_correctly']}</div>
                        <div class=\"kpi-label\">Dirección Correcta</div>
                    </div>
                    <div class=\"kpi-card\" style=\"border-top:3px solid #9B59B6;\">
                        <div class=\"kpi-value\">{hc['avg_lag_bars']}v</div>
                        <div class=\"kpi-label\">Antelación Híbrido</div>
                    </div>
                    <div class=\"kpi-card\" style=\"border-top:3px solid #8E44AD;\">
                        <div class=\"kpi-value\">{len(hybrid_alerts_list)}</div>
                        <div class=\"kpi-label\">Alertas Activadas</div>
                    </div>
                </div>

                <!-- Hybrid detection results table -->
                <div style=\"overflow-x:auto;margin-top:12px;\">
                    <table class=\"results-table\">
                        <thead>
                            <tr>
                                <th style=\"width:40px;\"></th>
                                <th style=\"width:100px;\">Fecha</th>
                                <th>Descripción</th>
                                <th style=\"width:60px;\">Lag</th>
                                <th style=\"width:60px;\">Conf</th>
                            </tr>
                        </thead>
                        <tbody>
                            {hybrid_rows_html if hybrid_rows_html else '<tr><td colspan=\"5\" style=\"text-align:center;color:#666;padding:20px;\">Sin alertas híbridas para mostrar</td></tr>'}
                        </tbody>
                    </table>
                </div>

                <!-- Hybrid periods timeline -->
                <div style=\"margin-top:12px;\">
                    <div style=\"color:#888;font-size:0.65rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;\">📅 Timeline de Alertas Híbridas</div>
                    <div style=\"max-height:200px;overflow-y:auto;\">
                        {hybrid_timeline_html if hybrid_timeline_html else '<div style=\"color:#666;font-size:0.75rem;padding:8px;\">Sin períodos activos</div>'}
                    </div>
                </div>
            </div>"""

if old_kpi_end in content:
    content = content.replace(old_kpi_end, new_kpi_end)
    changes += 1
    print(f"[OK] Change 7: Added hybrid KPI section in HTML report")
else:
    print(f"[FAIL] Change 7: KPI end section not found")

# ============================================================================
# CHANGE 8: Add hybrid variables in generate_html_report and fix the template
# ============================================================================
# We need to add hybrid data to the template variable initializations
old_template_init = """    for tf, data in tf_results.items():
        cross = data[\"cross_ref\"]
        trans = data[\"transitions\"]"""

new_template_init = """    for tf, data in tf_results.items():
        cross = data[\"cross_ref\"]
        trans = data[\"transitions\"]
        hc = data.get(\"hybrid_cross_ref\", {})
        hybrid_alerts_list = data.get(\"hybrid_alerts\", [])
        hybrid_periods_list = data.get(\"hybrid_periods\", [])"""

if old_template_init in content:
    content = content.replace(old_template_init, new_template_init)
    changes += 1
    print(f"[OK] Change 8: Added hybrid variables in template initialization")
else:
    print(f"[FAIL] Change 8: Template initialization not found")

# ============================================================================
# CHANGE 9: Add hybrid rows and timeline HTML generation in the template loop
# ============================================================================
# After the regime_changes_count line, add hybrid rows generation
old_hybrid_rows = """        regime_changes_count = len(regime_changes)
        signal_changes_count = cross[\"total_signal_changes\"]"""

new_hybrid_rows = """        # ── Híbrido: rows HTML ──
        hybrid_rows_html = ""
        for r in hc.get(\"results\", []):
            if r[\"type\"] == \"correcta\":
                icon = "🧬"
                row_color = \"rgba(155,89,182,0.08)\"
                lag_str = f\"{r['lag_bars']}v\"
                conf_str = f\"{r['max_confidence']:.0f}%\"
                if r.get(\"bias_aligned\"):
                    desc = r[\"description\"]
                else:
                    desc = r[\"description\"]
            else:
                icon = "⚠️"
                row_color = \"rgba(255,133,27,0.08)\"
                lag_str = "—"
                conf_str = "—"
                desc = r[\"description\"]

            date_str = \"\"
            if r.get(\"signal_change\"):
                date_str = r[\"signal_change\"][\"date_str\"]

            hybrid_rows_html += f\"\"\"<tr style=\"background:{row_color}\">
                <td style=\"text-align:center;font-size:1rem;\">{icon}</td>
                <td style=\"font-size:0.75rem;color:#888;\">{date_str}</td>
                <td style=\"font-size:0.75rem;\">{desc}</td>
                <td style=\"text-align:center;font-size:0.75rem;\">{lag_str}</td>
                <td style=\"text-align:center;font-size:0.75rem;color:#9B59B6;\">{conf_str}</td>
            </tr>\"\"\"

        # ── Híbrido: timeline HTML ──
        hybrid_timeline_html = ""
        for p in hybrid_periods_list[-15:]:
            p_color = \"#089981\" if p[\"type\"] == \"LONG\" else \"#F23645\"
            p_label = \"📈 LONG\" if p[\"type\"] == \"LONG\" else \"📉 SHORT\"
            bar_width = min(100, max(20, p[\"duration_bars\"] * 3))
            hybrid_timeline_html += (
                f\"<div style=\\\"display:flex;align-items:center;margin:4px 0;font-size:0.7rem;\\\">\"
                f\"<span style=\\\"width:80px;color:#888;\\\">{p['start_date']}</span>\"
                f\"<span style=\\\"width:55px;color:{p_color};font-weight:600;\\\">{p_label}</span>\"
                f\"<div style=\\\"flex:1;height:16px;background:#2a2a34;border-radius:4px;margin:0 8px;overflow:hidden;\\\">\"
                f\"<div style=\\\"height:100%;width:{bar_width}%;background:{p_color};border-radius:4px;opacity:0.7;\\\"></div>\"
                f\"</div>\"
                f\"<span style=\\\"width:40px;color:#888;text-align:right;\\\">{p['duration_bars']}v</span>\"
                f\"<span style=\\\"width:50px;color:#9B59B6;text-align:right;font-weight:600;\\\">{p['max_confidence']:.0f}%</span>\"
                f\"</div>\"
            )

        regime_changes_count = len(regime_changes)
        signal_changes_count = cross[\"total_signal_changes\"]"""

if old_hybrid_rows in content:
    content = content.replace(old_hybrid_rows, new_hybrid_rows)
    changes += 1
    print(f"[OK] Change 9: Added hybrid rows and timeline HTML generation")
else:
    print(f"[FAIL] Change 9: Regime changes/signal count line not found")

# ============================================================================
# Save
# ============================================================================
if changes > 0:
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[OK] Applied {changes} changes to {FILE}")
else:
    print(f"\n[FAIL] No changes were applied!")
    sys.exit(1)
