#!/usr/bin/env python
"""
Actualiza simulacion_alertas_tendencia.py para medir tasa de deteccion
del sistema de precursores vs el sistema de cambios de regimen.
"""
import sys

# Leer usando binary mode para evitar encoding issues
with open('simulacion_alertas_tendencia.py', 'rb') as f:
    raw = f.read()

content = raw.decode('utf-8')

changes = 0

# ============================================================
# CHANGE 1: Add precursor cross-reference function
# Insert after cross_reference_changes function
# ============================================================
# Find the end of cross_reference_changes (look for its return statement and the next def)
target = 'def cross_reference_changes'
idx = content.find(target)
if idx >= 0:
    # Find the next def after this function
    rest = content[idx+50:]
    next_def = rest.find('\ndef ')
    if next_def > 0:
        func_body = content[idx:idx+50+next_def]
        insert_at = idx + 50 + next_def
        
        precursor_func = """


def cross_reference_precursor(precursor_events, signal_changes, max_lag_bars=5):
    """
    Mide cuantos cambios de senal son precedidos por alertas precursoras.
    Similar a cross_reference_changes pero usando precursores en vez de regimenes.
    """
    if not precursor_events or not signal_changes:
        return {
            "total_signal_changes": len(signal_changes),
            "detected": 0,
            "false_positives": len(precursor_events),
            "false_negatives": len(signal_changes),
            "detection_rate": 0.0,
            "avg_lag": 0,
            "min_lag": 0,
            "max_lag": 0,
            "direction_correct": 0,
            "total_matched": 0,
        }

    matched_precursors = []
    unmatched_signals = []
    precursor_used = set()

    # For each signal change, find closest preceding precursor
    for sci, sc in enumerate(signal_changes):
        best_match = None
        best_lag = 999
        best_pi = -1

        for pi, pc in enumerate(precursor_events):
            if pi in precursor_used:
                continue
            if pc["idx"] <= sc["idx"] and pc["idx"] >= sc["idx"] - max_lag_bars:
                lag = sc["idx"] - pc["idx"]
                if best_match is None or lag < best_lag:
                    best_match = pc
                    best_lag = lag
                    best_pi = pi

        if best_match is not None:
            matched_precursors.append({
                "signal": sc,
                "precursor": best_match,
                "lag": best_lag,
            })
            precursor_used.add(best_pi)
        else:
            unmatched_signals.append(sc)

    # Direction alignment check
    direction_correct = 0
    for m in matched_precursors:
        to_sig = m["signal"].get("to_signal", "")
        prec_type = m["precursor"].get("type", "")
        if to_sig == "LONG" and "LONG" in prec_type:
            direction_correct += 1
        elif to_sig == "SHORT" and "SHORT" in prec_type:
            direction_correct += 1

    lags = [m["lag"] for m in matched_precursors]
    false_positives = max(0, len(precursor_events) - len(matched_precursors))
    false_negatives = len(unmatched_signals)

    return {
        "total_signal_changes": len(signal_changes),
        "detected": len(matched_precursors),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detection_rate": round(len(matched_precursors) / max(len(signal_changes), 1) * 100, 1),
        "avg_lag": round(sum(lags) / max(len(lags), 1), 1),
        "min_lag": min(lags) if lags else 0,
        "max_lag": max(lags) if lags else 0,
        "direction_correct": direction_correct,
        "total_matched": len(matched_precursors),
    }

"""
        content = content[:insert_at] + precursor_func + content[insert_at:]
        changes += 1
        print(f"[OK] Change 1: Added cross_reference_precursor function")
    else:
        print(f"[FAIL] Change 1: Could not find next def")
else:
    print(f"[FAIL] Change 1: cross_reference_changes not found")


# ============================================================
# CHANGE 2: Add precursor detection to the simulation loop
# Find where signal changes and regime changes are computed
# ============================================================
# Look for where cross_reference_changes is called
target_call = 'cr = cross_reference_changes('
idx = content.find(target_call)
if idx >= 0:
    # Find the surrounding code block
    # Print context
    block_start = content.rfind('\n', 0, idx-200) + 1
    block_end = content.find('\n\n', idx)
    if block_end < idx:
        block_end = idx + 800
    
    # Find where regime_changes signal is extracted from df
    detect_call = content.find('detect_signal_changes(df)', idx-800, idx)
    if detect_call < 0:
        detect_call = content.find('signal_changes = detect', idx-800, idx)
    
    # Find where the results are printed
    print_section = content.find('print(f\"  Cambios de senal', idx)
    
    # Insert precursor detection between regime changes and signal changes
    # Find the line that prints regime changes
    regime_print = content.find('cambios de regimen detectados', idx-300, idx+500)
    if regime_print >= 0:
        # Find the next print section for signal changes
        signal_print_start = content.find('Cambios de senal', regime_print)
        if signal_print_start > 0:
            # Insert before signal print
            insert_line_start = content.rfind('\n', 0, signal_print_start) + 1
            
            # Build the precursor detection code
            precursor_code = """
    # Detectar alertas precursoras
    precursor_long = df[df["precursor_long"]].index.tolist() if "precursor_long" in df.columns else []
    precursor_short = df[df["precursor_short"]].index.tolist() if "precursor_short" in df.columns else []
    precursor_events = []
    for pl in precursor_long:
        precursor_events.append({"idx": df.index.get_loc(pl), "type": "PRECURSOR_LONG"})
    for ps in precursor_short:
        precursor_events.append({"idx": df.index.get_loc(ps), "type": "PRECURSOR_SHORT"})
    precursor_events.sort(key=lambda x: x["idx"])

    # Cruzar precursores vs cambios de senal
    precursor_cr = cross_reference_precursor(precursor_events, signal_changes, max_lag_bars=5)
    pcr = precursor_cr
    print(f\"  Alertas precursoras detectadas: {len(precursor_events)}\")
    print(f\"  Senales precedidas por precursor: {pcr['detected']}/{pcr['total_signal_changes']} ({pcr['detection_rate']}%)\")
    print(f\"  Falsos positivos (precursor sin senal): {pcr['false_positives']}\")
    print(f\"  Falsos negativos (senal sin precursor): {pcr['false_negatives']}\")
    print(f\"  Anticipacion promedio: {pcr['avg_lag']} velas\")
    print(f\"  Direccion correcta: {pcr['direction_correct']}/{pcr['detected']}\")

"""
            content = content[:insert_line_start] + precursor_code + content[insert_line_start:]
            changes += 1
            print(f"[OK] Change 2: Added precursor detection in simulation loop")
        else:
            print(f"[FAIL] Change 2: Could not find signal print section")
    else:
        print(f"[FAIL] Change 2: Could not find regime print")
else:
    print(f"[FAIL] Change 2: cross_reference_changes call not found")


# ============================================================
# CHANGE 3: Add precursor section to the HTML report generation
# ============================================================
target_html = 'TASA DE DETECCION POR REGIMEN'
idx = content.find(target_html)
if idx > 0:
    # Find where to insert precursor metric
    regime_rate_section = content.find('TASA DE DETECCION', idx-100)
    if regime_rate_section > 0:
        # Add precursor rate section after regime rate
        insert_after = content.find('\n\n', regime_rate_section)
        if insert_after > regime_rate_section:
            precursor_html = """

------------------------------------------------------------
  TASA DE DETECCION POR PRECURSOR
------------------------------------------------------------
  Timeframe: {tf}
  Alertas precursoras: {len(precursor_events)}
  Deteccion: {pcr['detected']}/{pcr['total_signal_changes']} ({pcr['detection_rate']}%)
  Falsos positivos: {pcr['false_positives']}
  Falsos negativos: {pcr['false_negatives']}
  Anticipacion: {pcr['avg_lag']} velas
  Direccion correcta: {pcr['direction_correct']}/{pcr['detected']}

"""
            # Add to the log accumulation section
            # Find the log lines accumulation
            log_append = content.find('.append(f"')
            if log_append > regime_rate_section:
                content = content[:log_append+8] + "\"\\n\" +\n            f" + precursor_html.replace('\n', '\\n\" +\n            f\"').rstrip(' +') + content[log_append+8:]
                changes += 1
                print(f"[OK] Change 3: Added precursor section to log")
            else:
                print(f"[FAIL] Change 3: Could not find log append")
        else:
            print(f"[FAIL] Change 3: Could not find section end")
    else:
        print(f"[FAIL] Change 3: Could not find TASA DETECCION")
else:
    print(f"[FAIL] Change 3: target_html not found")


# ============================================================
# Write back
# ============================================================
with open('simulacion_alertas_tendencia.py', 'wb') as f:
    f.write(content.encode('utf-8'))

print(f"\n{'='*50}")
print(f"Total changes applied: {changes}")
print(f"{'='*50}")
