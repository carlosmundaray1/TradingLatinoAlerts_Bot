#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply hybrid alert integration to tradinglatino_hmm_dashboard.py"""
import sys

FILE = "tradinglatino_hmm_dashboard.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ============================================================================
# CHANGE 1: Import compute_hybrid_alert from the clean module
# ============================================================================
# Find the end of the _MISSING_DEPS check and add import
old = """if _MISSING_DEPS:
    print(\"=\" * 72)
    print(\"  ERROR: FALTAN DEPENDENCIAS\")
    print(\"=\" * 72)
    print(f\"\\n  Instala los siguientes paquetes:\\n\")
    print(f\"      pip install {' '.join(_MISSING_DEPS)}\\n\")
    print(\"  O usa:\\n\")
    print(f\"      pip install pandas numpy plotly hmmlearn yfinance\\n\")
    print(\"=\" * 72)
    sys.exit(1)"""

new = """if _MISSING_DEPS:
    print(\"=\" * 72)
    print(\"  ERROR: FALTAN DEPENDENCIAS\")
    print(\"=\" * 72)
    print(f\"\\n  Instala los siguientes paquetes:\\n\")
    print(f\"      pip install {' '.join(_MISSING_DEPS)}\\n\")
    print(\"  O usa:\\n\")
    print(f\"      pip install pandas numpy plotly hmmlearn yfinance\\n\")
    print(\"=\" * 72)
    sys.exit(1)

# --- Importar compute_hybrid_alert desde tradinglatino_hmm_clean.py ---
try:
    import importlib.util as _hybrid_imp
    _HYBRID_MODULE_PATH = Path(__file__).resolve().parent / \"tradinglatino_hmm_clean.py\"
    _HYBRID_SPEC = _hybrid_imp.spec_from_file_location(\"hmm_hybrid\", _HYBRID_MODULE_PATH)
    if _HYBRID_SPEC is not None:
        _HYBRID_MODULE = _hybrid_imp.module_from_spec(_HYBRID_SPEC)
        _HYBRID_SPEC.loader.exec_module(_HYBRID_MODULE)
        compute_hybrid_alert = _HYBRID_MODULE.compute_hybrid_alert
        compute_precursor_signals = _HYBRID_MODULE.compute_precursor_signals
        _compute_signal_scores = _HYBRID_MODULE._compute_signal_scores
    else:
        compute_hybrid_alert = None
        compute_precursor_signals = None
        _compute_signal_scores = None
except Exception as _hybrid_err:
    print(f\"  [WARN] No se pudo importar compute_hybrid_alert: {_hybrid_err}\")
    compute_hybrid_alert = None
    compute_precursor_signals = None
    _compute_signal_scores = None"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 1: Added hybrid alert import")
else:
    print("FAIL 1: MISSING_DEPS block not found")

# ============================================================================
# CHANGE 2: Add hybrid_data field to MultiAssetEntry dataclass
# ============================================================================
old = "@dataclass\nclass MultiAssetEntry:\n    asset: str\n    timeframe: str\n    df: pd.DataFrame\n    hmm_result: Optional[HMMResult]\n    hmm_summary: pd.DataFrame\n    sweep_results: List[BacktestResult]\n    classification: Dict[str, Any]\n    warnings_list: List[str]"

new = "@dataclass\nclass MultiAssetEntry:\n    asset: str\n    timeframe: str\n    df: pd.DataFrame\n    hmm_result: Optional[HMMResult]\n    hmm_summary: pd.DataFrame\n    sweep_results: List[BacktestResult]\n    classification: Dict[str, Any]\n    warnings_list: List[str]\n    hybrid_alert_data: Optional[Dict[str, Any]] = None"

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 2: Added hybrid_alert_data to MultiAssetEntry")
else:
    print("FAIL 2: MultiAssetEntry dataclass not found")

# ============================================================================
# CHANGE 3: Compute hybrid alert in main() pipeline after HMM states
# ============================================================================
old = """            print(\"  Resumen por estado:\")\n            print(state_summary.to_string(index=False))

            # 5) Extender df con estados HMM (rellenar para todas las filas)"""

new = """            print(\"  Resumen por estado:\")\n            print(state_summary.to_string(index=False))

            # 5a) Computar alerta hibrida HMM+Precursor
            hybrid_alert_data = None
            if _compute_signal_scores is not None and compute_precursor_signals is not None and compute_hybrid_alert is not None:
                try:
                    print(\"  Computando alerta hibrida HMM+Precursor...\")
                    df_hybrid = df.copy()
                    # Calcular signal scores (necesarios para los precursores)
                    df_hybrid = _compute_signal_scores(df_hybrid)
                    # Calcular EMA deviation, RSI14, volumen ratio (necesarios para signal scores)
                    if \"ema_slow\" in df_hybrid.columns:
                        df_hybrid[\"ema_dev_pct\"] = (df_hybrid[\"Close\"] - df_hybrid[\"ema_slow\"]) / df_hybrid[\"ema_slow\"] * 100
                    # Calcular precursores
                    df_hybrid = compute_precursor_signals(df_hybrid)
                    # Calcular alerta hibrida
                    df_hybrid = compute_hybrid_alert(df_hybrid, hmm_result.states, state_summary)
                    # Extraer datos relevantes
                    hybrid_alert_data = {
                        \"alert_active\": bool(df_hybrid[\"hybrid_alert_active\"].iloc[-1]) if \"hybrid_alert_active\" in df_hybrid.columns else False,
                        \"conf_long\": float(df_hybrid[\"hybrid_confidence_long\"].iloc[-1]) if \"hybrid_confidence_long\" in df_hybrid.columns else 0,
                        \"conf_short\": float(df_hybrid[\"hybrid_confidence_short\"].iloc[-1]) if \"hybrid_confidence_short\" in df_hybrid.columns else 0,
                        \"alert_long\": bool(df_hybrid[\"hybrid_alert_long\"].iloc[-1]) if \"hybrid_alert_long\" in df_hybrid.columns else False,
                        \"alert_short\": bool(df_hybrid[\"hybrid_alert_short\"].iloc[-1]) if \"hybrid_alert_short\" in df_hybrid.columns else False,
                        \"alerts_total\": int(df_hybrid[\"hybrid_alert_active\"].sum()) if \"hybrid_alert_active\" in df_hybrid.columns else 0,
                        \"max_conf_long\": float(df_hybrid[\"hybrid_confidence_long\"].max()) if \"hybrid_confidence_long\" in df_hybrid.columns else 0,
                        \"max_conf_short\": float(df_hybrid[\"hybrid_confidence_short\"].max()) if \"hybrid_confidence_short\" in df_hybrid.columns else 0,
                    }
                    print(f\"    Alerta hibrida activa: {hybrid_alert_data['alert_active']} (L={hybrid_alert_data['conf_long']:.0f} S={hybrid_alert_data['conf_short']:.0f})\")
                    print(f\"    Total alertas: {hybrid_alert_data['alerts_total']}\")
                except Exception as _hybrid_err:
                    print(f\"    [WARN] Error al computar alerta hibrida: {_hybrid_err}\")
                    hybrid_alert_data = {\"error\": str(_hybrid_err)}
            else:
                print(\"    [SKIP] compute_hybrid_alert no disponible (importacion fallo)\")

            # 5) Extender df con estados HMM (rellenar para todas las filas)"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 3: Added hybrid alert computation in pipeline")
else:
    print("FAIL 3: Pipeline section not found")

# ============================================================================
# CHANGE 4: Pass hybrid_alert_data to MultiAssetEntry
# ============================================================================
old = """            entries.append(MultiAssetEntry(
                asset=asset,
                timeframe=timeframe,
                df=df,
                hmm_result=hmm_result,
                hmm_summary=state_summary,
                sweep_results=sweep_results,
                classification=classification,
                warnings_list=entry_warnings,
            ))"""

new = """            entries.append(MultiAssetEntry(
                asset=asset,
                timeframe=timeframe,
                df=df,
                hmm_result=hmm_result,
                hmm_summary=state_summary,
                sweep_results=sweep_results,
                classification=classification,
                warnings_list=entry_warnings,
                hybrid_alert_data=hybrid_alert_data,
            ))"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 4: Passed hybrid_alert_data to MultiAssetEntry")
else:
    print("FAIL 4: MultiAssetEntry creation not found")

# ============================================================================
# CHANGE 5: Add hybrid alert HTML section in _generate_html (per-entry section)
# ============================================================================
# This is the trickiest part. I need to find the section where per-asset HTML is rendered.
# Let me search for where entries are iterated in _generate_html
old = """    html_parts: List[str] = [start_html]

    for e in entries:"""

new = """    html_parts: List[str] = [start_html]

    for e in entries:
        # Build hybrid alert HTML if available
        hybrid_html = \"\"
        if e.hybrid_alert_data and \"error\" not in e.hybrid_alert_data:
            ha = e.hybrid_alert_data
            if ha.get(\"alert_active\", False):
                direction = \"📈 LONG\" if ha.get(\"alert_long\") else (\"📉 SHORT\" if ha.get(\"alert_short\") else \"⚪ NEUTRAL\")
                direction_color = \"#089981\" if ha.get(\"alert_long\") else (\"#F23645\" if ha.get(\"alert_short\") else \"#888\")
                conf = ha[\"conf_long\"] if ha.get(\"alert_long\") else (ha[\"conf_short\"] if ha.get(\"alert_short\") else 0)
            else:
                direction = \"⚪ NO ALERTA\"
                direction_color = \"#888\"
                conf = 0
            max_conf = max(ha.get(\"max_conf_long\", 0), ha.get(\"max_conf_short\", 0))
            total = ha.get(\"alerts_total\", 0)
            hybrid_html = f\"\"\"
            <div class=\"hybrid-alert-card\" style=\"background:#1a1a24;border:1px solid #9B59B6;border-radius:10px;padding:16px;margin-bottom:16px;\">
                <div class=\"hybrid-header\" style=\"display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;\">
                    <h3 style=\"color:#9B59B6;font-size:1rem;margin:0;\">🧬 ALERTA HÍBRIDA HMM+PRECURSOR</h3>
                    <span style=\"background:#9B59B6;color:#fff;padding:4px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;\">{total} alertas</span>
                </div>
                <div style=\"display:flex;gap:16px;flex-wrap:wrap;\">
                    <div style=\"flex:1;min-width:120px;padding:12px;background:rgba(155,89,182,0.1);border-radius:8px;\">
                        <div style=\"font-size:0.65rem;color:#888;text-transform:uppercase;\">Estado Actual</div>
                        <div style=\"font-size:1.1rem;font-weight:700;color:{direction_color};\">{direction}</div>
                    </div>
                    <div style=\"flex:1;min-width:120px;padding:12px;background:rgba(155,89,182,0.1);border-radius:8px;\">
                        <div style=\"font-size:0.65rem;color:#888;text-transform:uppercase;\">Confianza Actual</div>
                        <div style=\"font-size:1.1rem;font-weight:700;color:#9B59B6;\">{conf:.0f}%</div>
                        <div style=\"height:4px;background:#333;border-radius:2px;margin-top:6px;\">
                            <div style=\"height:100%;width:{min(100, conf)}%;background:#9B59B6;border-radius:2px;transition:width 0.3s;\"></div>
                        </div>
                    </div>
                    <div style=\"flex:1;min-width:120px;padding:12px;background:rgba(155,89,182,0.1);border-radius:8px;\">
                        <div style=\"font-size:0.65rem;color:#888;text-transform:uppercase;\">Confianza Máxima</div>
                        <div style=\"font-size:1.1rem;font-weight:700;color:#9B59B6;\">{max_conf:.0f}%</div>
                    </div>
                    <div style=\"flex:1;min-width:120px;padding:12px;background:rgba(155,89,182,0.1);border-radius:8px;\">
                        <div style=\"font-size:0.65rem;color:#888;text-transform:uppercase;\">Alertas Activadas</div>
                        <div style=\"font-size:1.1rem;font-weight:700;color:#9B59B6;\">{total}</div>
                    </div>
                </div>
                {f\"\"\"<div style=\"margin-top:12px;padding:8px 12px;background:rgba(8,153,129,0.1);border:1px solid #08998133;border-radius:6px;font-size:0.75rem;color:#089981;\">
                    ✅ Alerta activa: {direction} con {conf:.0f}% de confianza
                </div>\"\"\" if ha.get(\"alert_active\") else \"\"}
            </div>
            \"\"\"
        elif e.hybrid_alert_data and \"error\" in e.hybrid_alert_data:
            hybrid_html = f\"\"\"<div class=\"hybrid-alert-card\" style=\"background:#1a1a24;border:1px solid #F23645;border-radius:10px;padding:12px;margin-bottom:16px;\">
                <h3 style=\"color:#F23645;font-size:0.85rem;margin:0;\">⚠️ Híbrido no disponible: {e.hybrid_alert_data['error']}</h3>
            </div>\"\"\"
        else:
            hybrid_html = \"\"\"

        # Insert hybrid HTML after the main content for this entry"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 5: Added hybrid alert HTML generation in _generate_html")
else:
    print("FAIL 5: _generate_html entry loop not found - finding alternative insertion point")
    # Try smaller exact match
    idx = content.find("html_parts: List[str] = [start_html]")
    if idx >= 0:
        print(f"  Found at index {idx}")
        # Print surrounding context
        start = max(0, idx - 50)
        end = min(len(content), idx + 200)
        print(f"  Context: {repr(content[start:end])}")

# ============================================================================
# CHANGE 6: Add hybrid alert CSS to the start_html
# ============================================================================
# Find the style section and add hybrid alert CSS
old = """        .small-table th {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; }}\n        .small-table td {{ font-size: 0.7rem; }}"""

new = """        .small-table th {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; }}\n        .small-table td {{ font-size: 0.7rem; }}

        /* Hybrid Alert Card */
        .hybrid-alert-card {{ animation: fadeInHybrid 0.5s ease; }}
        @keyframes fadeInHybrid {{ from {{ opacity: 0; transform: translateY(-10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .hybrid-alert-card:hover {{ border-color: #B07CD6; box-shadow: 0 0 20px rgba(155,89,182,0.2); }}"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 6: Added hybrid alert CSS styles")
else:
    print("FAIL 6: CSS section not found")

# ============================================================================
# CHANGE 7: Insert hybrid HTML after each entry's section in the HTML
# ============================================================================
# Find where each entry's HTML is added and insert hybrid_html after
# The section should be generated by _build_trades_table or similar
# Let me find where the entry content ends and append hybrid_html before closing the section

# Look for the pattern where an entry's regime/trade/cards section ends
# and insert the hybrid_html
old = """        # Append entry
        html_parts.append(entry_html)
    
    html_parts.append(multi_asset_html)
    html_parts.append(combined_equity_html)
    html_parts.append(overall_warnings_html)
    html_parts.append(end_html)
    html = \"\\n\".join(html_parts)"""

new = """        # Append entry with hybrid alert
        if hybrid_html:
            entry_html = entry_html.rstrip() + \"\\n\" + hybrid_html
        html_parts.append(entry_html)

    html_parts.append(multi_asset_html)
    html_parts.append(combined_equity_html)
    html_parts.append(overall_warnings_html)
    html_parts.append(end_html)
    html = \"\\n\".join(html_parts)"""

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("CHANGE 7: Inserted hybrid HTML into entry sections")
else:
    print("FAIL 7: HTML append section not found")

# ============================================================================
# Save
# ============================================================================
if changes > 0:
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[DONE] Applied {changes} changes to {FILE}")
else:
    print(f"\n[FAIL] No changes were applied!")
    sys.exit(1)
