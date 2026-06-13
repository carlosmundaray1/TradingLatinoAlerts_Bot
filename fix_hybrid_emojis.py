#!/usr/bin/env python3
"""Fix the _build_hybrid_alert_html function to use emoji characters instead of text placeholders"""

with open("tradinglatino_hmm_dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the function
func_marker = "def _build_hybrid_alert_html(hybrid_data: Optional[Dict[str, Any]]) -> str:"
func_start = content.find(func_marker)
if func_start < 0:
    print("[FAIL] Could not find the function")
    exit(1)

# Find the return statement of the function (to locate the end)
return_stmt = func_start + content[func_start:].find("return html\n")
if return_stmt < func_start:
    print("[FAIL] Could not find return statement")
    exit(1)

# Now find the end of the function (next def or end of file)
func_end = content.find("\ndef ", func_start + 5)
if func_end < 0:
    func_end = len(content)

# Extract the function
old_func = content[func_start:func_end]

new_func = '''def _build_hybrid_alert_html(hybrid_data: Optional[Dict[str, Any]]) -> str:
    """Genera HTML con el estado del sistema de alerta hibrida HMM+Precursor."""
    if hybrid_data is None:
        return '<div class="hybrid-section"><p style="color: var(--text-muted);">Sistema hibrido no disponible.</p></div>'
    if "error" in hybrid_data:
        err_msg = hybrid_data["error"]
        s = '<div class="hybrid-section"><p style="color: #FF4136;">\u26a0\ufe0f Error: ' + str(err_msg) + '</p></div>'
        return s

    active = hybrid_data.get("alert_active", False)
    conf_long = hybrid_data.get("conf_long", 0.0)
    conf_short = hybrid_data.get("conf_short", 0.0)
    alert_long = hybrid_data.get("alert_long", False)
    alert_short = hybrid_data.get("alert_short", False)
    alerts_total = hybrid_data.get("alerts_total", 0)
    max_conf_long = hybrid_data.get("max_conf_long", 0.0)
    max_conf_short = hybrid_data.get("max_conf_short", 0.0)

    # Badge de estado
    if alert_long:
        badge = '<span class="hybrid-badge hybrid-badge-long">\U0001f7e2 ALERTA LONG ACTIVA</span>'
    elif alert_short:
        badge = '<span class="hybrid-badge hybrid-badge-short">\U0001f534 ALERTA SHORT ACTIVA</span>'
    elif active:
        badge = '<span class="hybrid-badge hybrid-badge-warn">\u26a0\ufe0f ALERTA HIBRIDA ACTIVA</span>'
    else:
        badge = '<span class="hybrid-badge hybrid-badge-off">\u23f8\ufe0f SIN ALERTA</span>'

    # Barras de confianza
    long_pct = min(conf_long / 100.0 * 100, 100)
    short_pct = min(conf_short / 100.0 * 100, 100)

    # Alert badges
    long_alert_badge = '<span class="hybrid-active-badge long">\U0001f7e2 ALERTA</span>' if alert_long else ''
    short_alert_badge = '<span class="hybrid-active-badge short">\U0001f534 ALERTA</span>' if alert_short else ''

    # Direction
    if conf_long > conf_short:
        direction = "LONG \U0001f7e2"
    elif conf_short > conf_long:
        direction = "SHORT \U0001f534"
    else:
        direction = "NEUTRAL \u26aa"
    max_conf = max(conf_long, conf_short)
    conf_color = "#2ECC40" if max_conf > 50 else "#FF851B"

    bars = ""
    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza LONG</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(long_pct) + '%;background:#2ECC40;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_long) + '/100</span>'
    bars += "</div>"
    bars += long_alert_badge
    bars += "</div>"

    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza SHORT</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(short_pct) + '%;background:#FF4136;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_short) + '/100</span>'
    bars += "</div>"
    bars += short_alert_badge
    bars += "</div>"

    html = ""
    html += '<div class="hybrid-section">'
    html += '<div class="hybrid-header">'
    html += '<span class="hybrid-icon">\U0001f9ec</span>'
    html += '<h3 style="margin:0;color:var(--text-primary);">SISTEMA HIBRIDO: HMM + Precursores</h3>'
    html += badge
    html += "</div>"
    html += '<div class="hybrid-kpi-grid">'
    html += bars
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Direccion</span>'
    html += '<span class="hybrid-kpi-value">' + direction + "</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Max Confianza</span>'
    html += '<span class="hybrid-kpi-value" style="color:' + conf_color + ';">' + str(max_conf) + "/100</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Alertas Historicas</span>'
    html += '<span class="hybrid-kpi-value">' + str(alerts_total) + "</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Mejor LONG / SHORT</span>'
    html += '<span class="hybrid-kpi-value" style="font-size:0.85rem;">\U0001f7e2' + str(max_conf_long) + " / \U0001f534" + str(max_conf_short) + "</span>"
    html += "</div>"
    html += "</div>"
    html += "</div>"
    return html
'''

content = content[:func_start] + new_func + content[func_end:]

with open("tradinglatino_hmm_dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Function rewritten with proper emoji characters")
