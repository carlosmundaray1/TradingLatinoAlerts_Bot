#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply hybrid detection changes to simulacion_alertas_tendencia.py"""
import sys

FILE = "simulacion_alertas_tendencia.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ============================================================================
# CHANGE 1: Import compute_hybrid_alert
# ============================================================================
old = "verify_signals_historically = _HMM.verify_signals_historically"
new = old + "\ncompute_hybrid_alert = _HMM.compute_hybrid_alert\nHYBRID_CONFIDENCE_THRESHOLD = _HMM.HYBRID_CONFIDENCE_THRESHOLD"
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 1: Added compute_hybrid_alert import")
else:
    print("FAIL 1: Import not found")

# ============================================================================
# CHANGE 2: Add cross_reference_hybrid() + helpers after cross_reference_changes
# ============================================================================
old = "}\n\n\n# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n# AN\u00c1LISIS DE TRANSICIONES ENTRE REG\u00cdMENES\n# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef analyze_regime_transitions"
# More reliable: find by function name
idx = content.find("\ndef analyze_regime_transitions(states: np.ndarray, state_summary: pd.DataFrame) -> Dict[str, Any]:")
if idx > 0:
    # Find the closing brace of cross_reference_changes + blank lines before analyze_regime_transitions
    prefix = content[:idx]
    # Find the last return statement in cross_reference_changes
    brace_idx = prefix.rfind("}")
    # Find the content from the closing brace to analyze_regime_transitions
    between = content[brace_idx:idx]
    
    new_funcs = """}


# ----------------------------------------------------------------------------
# ANALISIS HIBRIDO: ALERTAS HIBRIDAS vs CAMBIOS DE SENAL
# ----------------------------------------------------------------------------

def find_hybrid_alerts(df: pd.DataFrame) -> List[Dict]:
    \"\"\"Encuentra activaciones de alertas hibridas (hybrid_alert_active).\"\"\"
    alerts = []
    prev_active = False
    for i in range(len(df)):
        is_active = bool(df[\"hybrid_alert_active\"].iloc[i]) if \"hybrid_alert_active\" in df.columns else False
        if not is_active:
            prev_active = False
            continue
        is_long = bool(df[\"hybrid_alert_long\"].iloc[i]) if \"hybrid_alert_long\" in df.columns else False
        is_short = bool(df[\"hybrid_alert_short\"].iloc[i]) if \"hybrid_alert_short\" in df.columns else False
        conf_long = float(df[\"hybrid_confidence_long\"].iloc[i]) if \"hybrid_confidence_long\" in df.columns else 0
        conf_short = float(df[\"hybrid_confidence_short\"].iloc[i]) if \"hybrid_confidence_short\" in df.columns else 0
        if not prev_active:
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
        prev_active = True
    return alerts


def find_hybrid_active_periods(df: pd.DataFrame) -> List[Dict]:
    \"\"\"Encuentra periodos donde hybrid_alert_long/short estan activos.\"\"\"
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
    Cruza las alertas hibridas con los cambios de senal para determinar:
    - Detecciones correctas: alerta hibrida activa en ventana antes del cambio
    - Direccion alineada: alerta en la misma direccion que el cambio
    - Anticipacion: cuantas velas antes se activo la alerta
    - Falsos negativos: cambios de senal SIN alerta hibrida previa
    \"\"\"
    if \"hybrid_alert_active\" not in df.columns:
        return {
            \"detected_correctly\": 0, \"total_signal_changes\": 0,
            \"detection_rate\": 0.0, \"false_negatives\": 0,
            \"bias_aligned\": 0, \"avg_lag_bars\": 0, \"avg_confidence\": 0.0,
            \"results\": [],
        }
    results = []
    signal_changes_real = [s for s in signal_changes if s[\"from_signal\"] != s[\"to_signal\"]]
    for sc in signal_changes_real:
        idx_start = max(0, sc[\"idx\"] - max_lag_bars)
        idx_end = sc[\"idx\"]
        window = df.iloc[idx_start:idx_end + 1]
        hybrid_active = window[\"hybrid_alert_active\"].any() if \"hybrid_alert_active\" in window.columns else False
        if hybrid_active:
            found_indices = window.index[window[\"hybrid_alert_active\"]].tolist()
            closest_idx = found_indices[-1]
            lag = sc[\"idx\"] - list(df.index).index(closest_idx)
            hybrid_long = window[\"hybrid_alert_long\"].any() if \"hybrid_alert_long\" in window.columns else False
            hybrid_short = window[\"hybrid_alert_short\"].any() if \"hybrid_alert_short\" in window.columns else False
            to_sig = sc[\"to_signal\"]
            direction_aligned = (to_sig == \"LONG\" and hybrid_long) or (to_sig == \"SHORT\" and hybrid_short)
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
                \"description\": f\"Hibrido alerto {lag}v antes: {sc['from_signal']}->{sc['to_signal']} \" +
                    (\"dir correcta\" if direction_aligned else \"dir mixta\"),
            })
        else:
            results.append({
                \"type\": \"no_hybrid\",
                \"signal_change\": sc,
                \"lag_bars\": None,
                \"bias_aligned\": None,
                \"max_confidence\": 0,
                \"description\": f\"Senal cambio {sc['from_signal']}->{sc['to_signal']} SIN alerta hibrida previa en {max_lag_bars} velas\",
            })
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


# ----------------------------------------------------------------------------
# ANALISIS DE TRANSICIONES ENTRE REGIMENES
# ----------------------------------------------------------------------------

def analyze_regime_transitions"""
    
    content = content[:brace_idx] + new_funcs + content[idx:]
    changes += 1
    print("CHANGE 2: Added cross_reference_hybrid() and helper functions")
else:
    print("FAIL 2: analyze_regime_transitions not found")

# ============================================================================
# CHANGE 3: Add hybrid alert computation in run_pipeline() step 5
# ============================================================================
old = """        # \u2500\u2500 5) Cambios de r\u00e9gimen \u2500\u2500
        regime_changes = find_regime_changes_detailed(states, df.index, state_summary)
        _log(f\"  Cambios de r\u00e9gimen detectados: {len(regime_changes)}\", log_lines)

        # \u2500\u2500 6) Cambios de se\u00f1al \u2500\u2500
        signal_changes = find_signal_changes(df)
        signal_changes_real = [s for s in signal_changes if s[\"from_signal\"] != s[\"to_signal\"]]
        _log(f\"  Cambios de se\u00f1al: {len(signal_changes_real)}\", log_lines)

        # \u2500\u2500 7) An\u00e1lisis cruzado \u2500\u2500"""

# More reliable: just match the key lines
lines = content.split('\n')
new_lines = []
i = 0
hybrid_pipeline_added = False
hybrid_cross_added = False
hybrid_tf_added = False
hybrid_resumen_added = False
hybrid_html_init_added = False
hybrid_html_rows_added = False
hybrid_html_section_added = False
hybrid_global_kpi_added = False

while i < len(lines):
    line = lines[i]
    
    # CHANGE 3: Add hybrid computation after signal_info and before regime changes
    if not hybrid_pipeline_added and i > 0 and lines[i-1].strip() == '_log(f\"  Senal actual: {signal_info[\"signal\"]} (fuerza: {signal_info[\"strength\"]}%)\", log_lines)' and line.strip().startswith('# \u2500\u2500 5) Cambios de'):
        new_lines.append(line)
        i += 1
        # Add hybrid alert lines
        new_lines.append('        # -- 5) Computar alerta hibrida HMM+Precursor --')
        new_lines.append("        _log(f\"  Calculando alerta hibrida HMM+Precursor...\", log_lines)")
        new_lines.append("        df = compute_hybrid_alert(df, states, state_summary)")
        new_lines.append("        hybrid_alerts = find_hybrid_alerts(df)")
        new_lines.append("        hybrid_periods = find_hybrid_active_periods(df)")
        new_lines.append("        _log(f\"    Alertas hibridas activadas: {len(hybrid_alerts)}\", log_lines)")
        new_lines.append("        _log(f\"    Periodos activos: {len(hybrid_periods)}\", log_lines)")
        new_lines.append('')
        hybrid_pipeline_added = True
        print("CHANGE 3: Added hybrid alert computation")
        continue
    
    # CHANGE 4: Add hybrid cross-reference after cross_ref
    if not hybrid_cross_added and line.strip().startswith('_log(f\"  Direccion correcta: {cross_ref[\"bias_aligned\"]}/{cross_ref[\"detected_correctly\"]}\"'):
        new_lines.append(line)
        i += 1
        new_lines.append('')
        new_lines.append('        # -- 8) Analisis cruzado hibrido --')
        new_lines.append("        hybrid_cross_ref = cross_reference_hybrid(df, signal_changes, max_lag_bars=5)")
        new_lines.append("        _log(f\"    Hibrido detecto: {hybrid_cross_ref['detected_correctly']}/{hybrid_cross_ref['total_signal_changes']} ({hybrid_cross_ref['detection_rate']}%)\", log_lines)")
        new_lines.append("        _log(f\"    Direccion correcta: {hybrid_cross_ref['bias_aligned']}/{hybrid_cross_ref['detected_correctly']}\", log_lines)")
        new_lines.append("        _log(f\"    Confianza promedio: {hybrid_cross_ref['avg_confidence']}%\", log_lines)")
        new_lines.append("        _log(f\"    Anticipacion promedio: {hybrid_cross_ref['avg_lag_bars']} velas\", log_lines)")
        hybrid_cross_added = True
        print("CHANGE 4: Added hybrid cross-reference")
        continue
    
    # CHANGE 5: Renumber step 8->9, 9->10
    if line.strip().startswith('# \u2500\u2500 8) Transiciones'):
        new_lines.append('        # -- 9) Transiciones --')
        i += 1
        continue
    if line.strip().startswith('# \u2500\u2500 9) Verificacion'):
        new_lines.append('        # -- 10) Verificacion historica (win rate) --')
        i += 1
        continue
    
    # CHANGE 6: Add hybrid data to tf_results
    if not hybrid_tf_added and line.strip().startswith('"cross_ref": cross_ref,'):
        new_lines.append(line)
        i += 1
        new_lines.append('            "hybrid_alerts": hybrid_alerts,')
        new_lines.append('            "hybrid_periods": hybrid_periods,')
        new_lines.append('            "hybrid_cross_ref": hybrid_cross_ref,')
        hybrid_tf_added = True
        print("CHANGE 6: Added hybrid data to tf_results")
        continue
    
    # CHANGE 7: Update resumen to include hybrid
    if not hybrid_resumen_added and line.strip().startswith('_log(f\"  {tf}: {c[\"detected_correctly\"]}/{c[\"total_signal_changes\"]}'):
        new_lines.append("        hc = data.get(\"hybrid_cross_ref\", {})")
        new_lines.append("        hybrid_rate = hc.get(\"detection_rate\", 0)")
        new_lines.append("        _log(f\"  {tf}: HMM={c['detected_correctly']}/{c['total_signal_changes']} ({c['detection_rate']}%) | \"")
        new_lines.append("             f\"Hibrido={hc.get('detected_correctly',0)}/{hc.get('total_signal_changes',0)} ({hybrid_rate}%) | \"")
        new_lines.append("             f\"FP:{c['false_positives']} FN:{c['false_negatives']} | \"")
        new_lines.append("             f\"Ant: {c['avg_lag_bars']}v | Dir: {c['bias_aligned']}/{c['detected_correctly']}\", log_lines)")
        i += 1  # skip old line
        hybrid_resumen_added = True
        print("CHANGE 7: Updated resumen with hybrid")
        continue
    
    # CHANGE 8: Add hybrid variables in generate_html_report template init
    if not hybrid_html_init_added and line.strip() == 'cross = data["cross_ref"]' and i > 0 and 'for tf, data in' in lines[i-1]:
        new_lines.append(line)
        i += 1
        new_lines.append('        hc = data.get("hybrid_cross_ref", {})')
        new_lines.append('        hybrid_alerts_list = data.get("hybrid_alerts", [])')
        new_lines.append('        hybrid_periods_list = data.get("hybrid_periods", [])')
        hybrid_html_init_added = True
        print("CHANGE 8: Added hybrid vars in template init")
        continue
    
    # CHANGE 9: Add hybrid rows and timeline HTML generation
    if not hybrid_html_rows_added and line.strip() == 'regime_changes_count = len(regime_changes)' and 'signal_changes_count' in lines[i+1] if i+1 < len(lines) else False:
        # Insert hybrid rows/timeline generation before this
        new_lines.append('        # -- Hibrido: rows HTML --')
        new_lines.append('        hybrid_rows_html = ""')
        new_lines.append('        for r in hc.get("results", []):')
        new_lines.append('            if r["type"] == "correcta":')
        new_lines.append('                icon = "'''')
        new_lines.append('                row_color = "rgba(155,89,182,0.08)"')
        new_lines.append('                lag_str = f"{r[\'lag_bars\']}v"')
        new_lines.append('                conf_str = f"{r[\'max_confidence\']:.0f}%"')
        new_lines.append('                desc = r["description"]')
        new_lines.append('            else:')
        new_lines.append('                icon = "'''')
        new_lines.append('                row_color = "rgba(255,133,27,0.08)"')
        new_lines.append('                lag_str = "---"')
        new_lines.append('                conf_str = "---"')
        new_lines.append('                desc = r["description"]')
        new_lines.append('            date_str = r.get("signal_change", {}).get("date_str", "") if r.get("signal_change") else ""')
        new_lines.append('            hybrid_rows_html += f"<tr style=background:{row_color}>"')
        new_lines.append('            hybrid_rows_html += f"<td style=text-align:center;font-size:1rem;>{icon}</td>"')
        new_lines.append('            hybrid_rows_html += f"<td style=font-size:0.75rem;color:#888;>{date_str}</td>"')
        new_lines.append('            hybrid_rows_html += f"<td style=font-size:0.75rem;>{desc}</td>"')
        new_lines.append('            hybrid_rows_html += f"<td style=text-align:center;font-size:0.75rem;>{lag_str}</td>"')
        new_lines.append('            hybrid_rows_html += f"<td style=text-align:center;font-size:0.75rem;color:#9B59B6;>{conf_str}</td></tr>"')
        new_lines.append('')
        new_lines.append('        # -- Hibrido: timeline HTML --')
        new_lines.append('        hybrid_timeline_html = ""')
        new_lines.append('        for p in hybrid_periods_list[-15:]:')
        new_lines.append('            p_color = "#089981" if p["type"] == "LONG" else "#F23645"')
        new_lines.append('            p_label = "LONG" if p["type"] == "LONG" else "SHORT"')
        new_lines.append('            bar_width = min(100, max(20, p["duration_bars"] * 3))')
        new_lines.append('            hybrid_timeline_html += (')
        new_lines.append('                f"<div style=display:flex;align-items:center;margin:4px 0;font-size:0.7rem;>"')
        new_lines.append('                f"<span style=width:80px;color:#888;>{p[chr(39)+chr(115)+chr(116)+chr(97)+chr(114)+chr(116)+chr(95)+chr(100)+chr(97)+chr(116)+chr(101)+chr(39)]}</span>"')
        new_lines.append('                f"<span style=width:50px;color:{p_color};font-weight:600;>{p_label}</span>"')
        new_lines.append('                f"<div style=flex:1;height:16px;background:#2a2a34;border-radius:4px;margin:0 8px;overflow:hidden;>"')
        new_lines.append('                f"<div style=height:100%;width:{bar_width}%;background:{p_color};border-radius:4px;opacity:0.7;></div>"')
        new_lines.append('                f"</div>"')
        new_lines.append('                f"<span style=width:40px;color:#888;text-align:right;>{p[chr(39)+chr(100)+chr(117)+chr(114)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(98)+chr(97)+chr(114)+chr(115)+chr(39)]}v</span>"')
        new_lines.append('                f"<span style=width:50px;color:#9B59B6;text-align:right;font-weight:600;>{p[chr(39)+chr(109)+chr(97)+chr(120)+chr(95)+chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(100)+chr(101)+chr(110)+chr(99)+chr(101)+chr(39)]:.0f}%</span>"')
        new_lines.append('                f"</div>"')
        new_lines.append('            )')
        new_lines.append('')
        hybrid_html_rows_added = True
        print("CHANGE 9: Added hybrid rows and timeline HTML")
        # Don't skip - still add regime_changes_count line
        new_lines.append(line)
        i += 1
        continue
    
    # CHANGE 10: Add hybrid section in HTML template (after Antelacion Promedio KPI)
    if not hybrid_html_section_added and line.strip().startswith('<div class="kpi-value">{cross[\'avg_lag_bars\']}v</div>'):
        new_lines.append(line)
        i += 1
        new_lines.append(line)  # </div> closing kpi-card
        i += 1
        new_lines.append(line)  # </div> closing kpi-grid
        i += 1
        new_lines.append('')
        # Insert hybrid section
        new_lines.append('            <!-- Hybrid Alert Section -->')
        new_lines.append('            <div class="detail-card" style="margin-top:16px;border-left:3px solid #9B59B6;">')
        new_lines.append('                <div class="card-title" style="color:#9B59B6;">HYBRID: HMM + Precursores</div>')
        new_lines.append('                <div class="kpi-grid" style="margin-top:8px;">')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #9B59B6;">')
        new_lines.append('                        <div class="kpi-value" style="color:#9B59B6;">{hc[chr(39)+chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(101)+chr(100)+chr(95)+chr(99)+chr(111)+chr(114)+chr(114)+chr(101)+chr(99)+chr(116)+chr(108)+chr(121)+chr(39)]}/{hc[chr(39)+chr(116)+chr(111)+chr(116)+chr(97)+chr(108)+chr(95)+chr(115)+chr(105)+chr(103)+chr(110)+chr(97)+chr(108)+chr(95)+chr(99)+chr(104)+chr(97)+chr(110)+chr(103)+chr(101)+chr(115)+chr(39)]}</div>')
        new_lines.append('                        <div class="kpi-label">Detectados por Hibrido</div>')
        new_lines.append('                    </div>')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #8E44AD;">')
        new_lines.append('                        <div class="kpi-value" style="color:#8E44AD;">{hc[chr(39)+chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(114)+chr(97)+chr(116)+chr(101)+chr(39)]}%</div>')
        new_lines.append('                        <div class="kpi-label">Tasa Deteccion Hibrida</div>')
        new_lines.append('                    </div>')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #9B59B6;">')
        new_lines.append('                        <div class="kpi-value" style="color:#9B59B6;">{hc[chr(39)+chr(97)+chr(118)+chr(103)+chr(95)+chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(100)+chr(101)+chr(110)+chr(99)+chr(101)+chr(39)]}%</div>')
        new_lines.append('                        <div class="kpi-label">Confianza Promedio</div>')
        new_lines.append('                    </div>')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #8E44AD;">')
        new_lines.append('                        <div class="kpi-value" style="color:#8E44AD;">{hc[chr(39)+chr(98)+chr(105)+chr(97)+chr(115)+chr(95)+chr(97)+chr(108)+chr(105)+chr(103)+chr(110)+chr(101)+chr(100)+chr(39)]}/{hc[chr(39)+chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(101)+chr(100)+chr(95)+chr(99)+chr(111)+chr(114)+chr(114)+chr(101)+chr(99)+chr(116)+chr(108)+chr(121)+chr(39)]}</div>')
        new_lines.append('                        <div class="kpi-label">Direccion Correcta</div>')
        new_lines.append('                    </div>')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #9B59B6;">')
        new_lines.append('                        <div class="kpi-value">{hc[chr(39)+chr(97)+chr(118)+chr(103)+chr(95)+chr(108)+chr(97)+chr(103)+chr(95)+chr(98)+chr(97)+chr(114)+chr(115)+chr(39)]}v</div>')
        new_lines.append('                        <div class="kpi-label">Anticipacion Hibrido</div>')
        new_lines.append('                    </div>')
        new_lines.append('                    <div class="kpi-card" style="border-top:3px solid #8E44AD;">')
        new_lines.append('                        <div class="kpi-value" style="color:#8E44AD;">{len(hybrid_alerts_list)}</div>')
        new_lines.append('                        <div class="kpi-label">Alertas Activadas</div>')
        new_lines.append('                    </div>')
        new_lines.append('                </div>')
        new_lines.append('')
        new_lines.append('                <!-- Hybrid results table -->')
        new_lines.append('                <div style="overflow-x:auto;margin-top:12px;">')
        new_lines.append('                    <table class="results-table" style="font-size:0.7rem;">')
        new_lines.append('                        <thead><tr>')
        new_lines.append('                            <th style="width:30px;"></th>')
        new_lines.append('                            <th style="width:90px;">Fecha</th>')
        new_lines.append('                            <th>Descripcion</th>')
        new_lines.append('                            <th style="width:50px;">Lag</th>')
        new_lines.append('                            <th style="width:55px;">Conf</th>')
        new_lines.append('                        </tr></thead>')
        new_lines.append('                        <tbody>')
        new_lines.append('                            {hybrid_rows_html if hybrid_rows_html else "<tr><td colspan=5 style=text-align:center;color:#666;padding:20px;>Sin alertas hibridas</td></tr>"}')
        new_lines.append('                        </tbody>')
        new_lines.append('                    </table>')
        new_lines.append('                </div>')
        new_lines.append('')
        new_lines.append('                <!-- Hybrid timeline -->')
        new_lines.append('                <div style="margin-top:12px;">')
        new_lines.append('                    <div style="color:#888;font-size:0.65rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Timeline Alertas Hibridas</div>')
        new_lines.append('                    <div style="max-height:200px;overflow-y:auto;">')
        new_lines.append('                        {hybrid_timeline_html if hybrid_timeline_html else "<div style=color:#666;font-size:0.75rem;padding:8px;>Sin periodos activos</div>"}')
        new_lines.append('                    </div>')
        new_lines.append('                </div>')
        new_lines.append('            </div>')
        new_lines.append('')
        hybrid_html_section_added = True
        print("CHANGE 10: Added hybrid section in HTML template")
        continue
    
    new_lines.append(line)
    i += 1

content = '\n'.join(new_lines)

# ============================================================================
# Save
# ============================================================================
with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n[DONE] Applied changes to {FILE}")
