#!/usr/bin/env python3
"""Apply hybrid alert visual section to tradinglatino_hmm_dashboard.py"""

import re

with open("tradinglatino_hmm_dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ============================================================
# CHANGE 1: Add _build_hybrid_alert_html function
# ============================================================

HYBRID_FUNC = '''
def _build_hybrid_alert_html(hybrid_data: Optional[Dict[str, Any]]) -> str:
    """Genera HTML con el estado del sistema de alerta hibrida HMM+Precursor."""
    if hybrid_data is None:
        return "<div class=\\"hybrid-section\\"><p style=\\"color: var(--text-muted);\\">Sistema hibrido no disponible.</p></div>"
    if "error" in hybrid_data:
        return f"<div class=\\"hybrid-section\\"><p style=\\"color: #FF4136;\\">⚠️ Error: {hybrid_data['error']}</p></div>"

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
        badge = '<span class="hybrid-badge hybrid-badge-long">🟢 ALERTA LONG ACTIVA</span>'
    elif alert_short:
        badge = '<span class="hybrid-badge hybrid-badge-short">🔴 ALERTA SHORT ACTIVA</span>'
    elif active:
        badge = '<span class="hybrid-badge hybrid-badge-warn">⚠️ ALERTA HIBRIDA ACTIVA</span>'
    else:
        badge = '<span class="hybrid-badge hybrid-badge-off">⏸️ SIN ALERTA</span>'

    # Barras de confianza
    def confidence_bar(value, max_val=100.0, color="#3498DB"):
        pct = min(value / max_val * 100, 100)
        return f"""<div class="hybrid-bar-track">
            <div class="hybrid-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
            <span class="hybrid-bar-label">{value:.0f}/100</span>
        </div>"""

    long_bar = confidence_bar(conf_long, color="#2ECC40")
    short_bar = confidence_bar(conf_short, color="#FF4136")

    # KPI cards
    direction = "LONG 🟢" if conf_long > conf_short else "SHORT 🔴" if conf_short > conf_long else "NEUTRAL ⚪"
    max_conf = max(conf_long, conf_short)

    html = f"""
    <div class="hybrid-section">
        <div class="hybrid-header">
            <span class="hybrid-icon">🧬</span>
            <h3 style="margin:0;color:var(--text-primary);">SISTEMA HIBRIDO: HMM + Precursores</h3>
            {badge}
        </div>

        <div class="hybrid-kpi-grid">
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Confianza LONG</span>
                {long_bar}
                {"<span class=\\"hybrid-active-badge long\\">🟢 ALERTA</span>" if alert_long else ""}
            </div>
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Confianza SHORT</span>
                {short_bar}
                {"<span class=\\"hybrid-active-badge short\\">🔴 ALERTA</span>" if alert_short else ""}
            </div>
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Dirección</span>
                <span class="hybrid-kpi-value">{direction}</span>
            </div>
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Max Confianza</span>
                <span class="hybrid-kpi-value" style="color:{'#2ECC40' if max_conf > 50 else '#FF851B'};">{max_conf:.0f}/100</span>
            </div>
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Alertas Históricas</span>
                <span class="hybrid-kpi-value">{alerts_total}</span>
            </div>
            <div class="hybrid-kpi">
                <span class="hybrid-kpi-label">Mejor LONG / SHORT</span>
                <span class="hybrid-kpi-value" style="font-size:0.85rem;">🟢{max_conf_long:.0f} / 🔴{max_conf_short:.0f}</span>
            </div>
        </div>
    </div>
    """
    return html

'''

MARKER_1 = "\n\ndef _build_hmm_table(hmm_summary: pd.DataFrame) -> str:"
IDX_1 = content.find(MARKER_1)
if IDX_1 >= 0:
    content = content[:IDX_1] + HYBRID_FUNC + content[IDX_1:]
    print(f"[OK] Change 1: Added _build_hybrid_alert_html function")
    changes += 1
else:
    print("[FAIL] Change 1: Could not find marker for _build_hybrid_alert_html insertion")

# ============================================================
# CHANGE 2: Add hybrid alert section to entry loop template
# ============================================================

MARKER_2 = '            {trades_html}\n        </div>\n        """'
IDX_2 = content.find(MARKER_2)
if IDX_2 >= 0:
    REPLACEMENT = '''            {trades_html}
            <h3>🧬 Alerta Híbrida HMM+Precursor</h3>
            {_build_hybrid_alert_html(e.hybrid_alert_data)}
        </div>
        """'''
    content = content[:IDX_2] + REPLACEMENT + content[IDX_2 + len(MARKER_2):]
    print(f"[OK] Change 2: Added hybrid alert section to entry template")
    changes += 1
else:
    print("[FAIL] Change 2: Could not find entry template marker")

# ============================================================
# CHANGE 3: Add CSS for hybrid alert section
# ============================================================

CSS_HYBRID = '''
/* === HYBRID ALERT SYSTEM === */
.hybrid-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}
.hybrid-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.hybrid-icon {
    font-size: 1.5rem;
}
.hybrid-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.hybrid-badge-long {
    background: rgba(46, 204, 64, 0.2);
    color: #2ECC40;
    border: 1px solid rgba(46, 204, 64, 0.4);
    animation: pulse-green 1.5s ease-in-out infinite;
}
.hybrid-badge-short {
    background: rgba(255, 65, 54, 0.2);
    color: #FF4136;
    border: 1px solid rgba(255, 65, 54, 0.4);
    animation: pulse-red 1.5s ease-in-out infinite;
}
.hybrid-badge-warn {
    background: rgba(255, 133, 27, 0.15);
    color: #FF851B;
    border: 1px solid rgba(255, 133, 27, 0.3);
}
.hybrid-badge-off {
    background: rgba(150, 150, 150, 0.15);
    color: var(--text-muted);
    border: 1px solid rgba(150, 150, 150, 0.2);
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(46, 204, 64, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(46, 204, 64, 0); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 65, 54, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(255, 65, 54, 0); }
}
.hybrid-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
}
.hybrid-kpi {
    background: rgba(0,0,0,0.15);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
.hybrid-kpi-label {
    display: block;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}
.hybrid-kpi-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}
.hybrid-bar-track {
    position: relative;
    height: 20px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
    margin: 0.3rem 0;
}
.hybrid-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s ease;
    min-width: 4px;
}
.hybrid-bar-label {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.65rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 0 4px rgba(0,0,0,0.8);
}
.hybrid-active-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-top: 0.3rem;
}
.hybrid-active-badge.long {
    background: rgba(46, 204, 64, 0.3);
    color: #2ECC40;
}
.hybrid-active-badge.short {
    background: rgba(255, 65, 54, 0.3);
    color: #FF4136;
}
'''

MARKER_3 = "</style></style>"
IDX_3 = content.find(MARKER_3)
if IDX_3 >= 0:
    content = content[:IDX_3] + CSS_HYBRID + "\n" + content[IDX_3:]
    print(f"[OK] Change 3: Added hybrid alert CSS")
    changes += 1
else:
    print("[FAIL] Change 3: Could not find CSS closing tag")

# Save
with open("tradinglatino_hmm_dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal changes applied: {changes}")
