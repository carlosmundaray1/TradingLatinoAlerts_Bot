#!/usr/bin/env python3
"""Fix backslash issues in f-string expressions in _build_hybrid_alert_html"""

with open("tradinglatino_hmm_dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# Fix 1: Error return line - replace escaped quotes with single quotes and HTML entities
old_1 = 'return "<div class=\\"hybrid-section\\"><p style=\\"color: var(--text-muted);\\">Sistema hibrido no disponible.</p></div>"'
new_1 = "return '<div class=\"hybrid-section\"><p style=\"color: var(--text-muted);\">Sistema hibrido no disponible.</p></div>'"
if old_1 in content:
    content = content.replace(old_1, new_1)
    print("[OK] Fix 1: Error return path")
    changes += 1
else:
    print("[WARN] Fix 1: Pattern not found, trying different encoding")
    # Try with actual backslashes (as they appear in the file)
    old_1b = 'return "<div class=\\"hybrid-section\\"><p style=\\"color: var(--text-muted);\\">Sistema hibrido no disponible.</p></div>"'
    # Already literal backslashes - just replace with non-f-string version
    new_1b = "return '<div class=\"hybrid-section\"><p style=\"color: var(--text-muted);\">Sistema hibrido no disponible.</p></div>'"
    if old_1b in content:
        content = content.replace(old_1b, new_1b)
        print("[OK] Fix 1b: Error return path (literal backslashes)")
        changes += 1

# Fix 2: Error message with f-string
old_2 = 'return f"<div class=\\"hybrid-section\\"><p style=\\"color: #FF4136;\\">'
new_2 = "return '<div class=\"hybrid-section\"><p style=\"color: #FF4136;\">' + f\""
if old_2 in content:
    content = content.replace(old_2, new_2)
    print("[OK] Fix 2: Error message f-string")
    changes += 1
else:
    old_2b = 'return f"<div class=\\"hybrid-section\\"><p style=\\"color: #FF4136;\\">'
    if old_2b in content:
        content = content.replace(old_2b, new_2)
        print("[OK] Fix 2b: Error message (literal backslashes)")
        changes += 1

# Fix 3: The badge span with f-string - the alert_long/alert_short conditionals
# These have backslashes inside f-string {} expressions
# Strategy: replace with non-f-string approach using concatenation

# Find the whole function and rewrite it cleanly
import re

# Find the function boundaries
func_start = content.find("def _build_hybrid_alert_html")
if func_start >= 0:
    # Find end of function (next def or end of file)
    func_end = content.find("\ndef ", func_start + 5)
    if func_end < 0:
        func_end = len(content)
    
    old_func = content[func_start:func_end]
    
    # Rewrite the function without backslashes in f-string expressions
    new_func = '''def _build_hybrid_alert_html(hybrid_data: Optional[Dict[str, Any]]) -> str:
    """Genera HTML con el estado del sistema de alerta hibrida HMM+Precursor."""
    if hybrid_data is None:
        return '<div class="hybrid-section"><p style="color: var(--text-muted);">Sistema hibrido no disponible.</p></div>'
    if "error" in hybrid_data:
        err_msg = hybrid_data["error"]
        s = '<div class="hybrid-section"><p style="color: #FF4136;">WARNING Error: ' + str(err_msg) + '</p></div>'
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
        badge = '<span class="hybrid-badge hybrid-badge-long">GREEN ALERTA LONG ACTIVA</span>'
    elif alert_short:
        badge = '<span class="hybrid-badge hybrid-badge-short">RED ALERTA SHORT ACTIVA</span>'
    elif active:
        badge = '<span class="hybrid-badge hybrid-badge-warn">WARNING ALERTA HIBRIDA ACTIVA</span>'
    else:
        badge = '<span class="hybrid-badge hybrid-badge-off">PAUSE SIN ALERTA</span>'

    # Barras de confianza
    long_pct = min(conf_long / 100.0 * 100, 100)
    short_pct = min(conf_short / 100.0 * 100, 100)

    # Alert badges
    long_alert_badge = '<span class="hybrid-active-badge long">GREEN ALERTA</span>' if alert_long else ''
    short_alert_badge = '<span class="hybrid-active-badge short">RED ALERTA</span>' if alert_short else ''

    # Direction
    if conf_long > conf_short:
        direction = "LONG GREEN"
    elif conf_short > conf_long:
        direction = "SHORT RED"
    else:
        direction = "NEUTRAL WHITE"
    max_conf = max(conf_long, conf_short)
    conf_color = "#2ECC40" if max_conf > 50 else "#FF851B"

    bars = ''
    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza LONG</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(long_pct) + '%;background:#2ECC40;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_long) + '/100</span>'
    bars += '</div>'
    bars += long_alert_badge
    bars += '</div>'

    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza SHORT</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(short_pct) + '%;background:#FF4136;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_short) + '/100</span>'
    bars += '</div>'
    bars += short_alert_badge
    bars += '</div>'

    html = ''
    html += '<div class="hybrid-section">'
    html += '<div class="hybrid-header">'
    html += '<span class="hybrid-icon">DNA</span>'
    html += '<h3 style="margin:0;color:var(--text-primary);">SISTEMA HIBRIDO: HMM + Precursores</h3>'
    html += badge
    html += '</div>'
    html += '<div class="hybrid-kpi-grid">'
    html += bars
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Direccion</span>'
    html += '<span class="hybrid-kpi-value">' + direction + '</span>'
    html += '</div>'
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Max Confianza</span>'
    html += '<span class="hybrid-kpi-value" style="color:' + conf_color + ';">' + str(max_conf) + '/100</span>'
    html += '</div>'
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Alertas Historicas</span>'
    html += '<span class="hybrid-kpi-value">' + str(alerts_total) + '</span>'
    html += '</div>'
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Mejor LONG / SHORT</span>'
    html += '<span class="hybrid-kpi-value" style="font-size:0.85rem;">GREEN' + str(max_conf_long) + ' / RED' + str(max_conf_short) + '</span>'
    html += '</div>'
    html += '</div>'
    html += '</div>'
    return html
'''
    
    content = content[:func_start] + new_func + content[func_end:]
    print("[OK] Function completely rewritten without backslash issues")
    changes += 1

# Save
with open("tradinglatino_hmm_dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal fixes applied: {changes}")
