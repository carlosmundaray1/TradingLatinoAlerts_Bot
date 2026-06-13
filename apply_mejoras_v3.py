#!/usr/bin/env python
"""
Apply remaining 3 improvements to tradinglatino_hmm_clean.py using corrected line numbers.
"""
import sys

# Read file
with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
changes = 0

# ============================================================
# CHANGE 3: Find _smooth_states end and add new functions after it
# ============================================================
# Find where _smooth_states starts and ends
smooth_start = None
smooth_return = None
for i in range(len(lines)):
    if 'def _smooth_states' in lines[i]:
        smooth_start = i
        print(f"_smooth_states starts at line {i+1}")
    if smooth_start is not None and i > smooth_start and 'return smoothed' in lines[i]:
        smooth_return = i
        print(f"_smooth_states ends at line {i+1}")
        break

if smooth_return is not None:
    # Find the last blank line after return smoothed
    insert_at = smooth_return + 1
    while insert_at < len(lines) and lines[insert_at].strip() == '':
        insert_at += 1
    
    new_function_lines = [
        "\n",
        "\n",
        "# -- MEJORA 1A: Recalcular scores incluyendo el regimen HMM --\n",
        "def compute_signal_scores_with_hmm(df, state_summary):\n",
        '    """Anade el peso W_HMM_REGIME a los scores LONG/SHORT segun el regimen HMM."""\n',
        "    state_bias = {}\n",
        "    for _, r in state_summary.iterrows():\n",
        '        state_bias[int(r["state"])] = _classify_regime_bias(r["description"])\n',
        "    hmm_bullish = pd.Series(False, index=df.index)\n",
        "    hmm_bearish = pd.Series(False, index=df.index)\n",
        '    if "regime" in df.columns:\n',
        "        for state, bias in state_bias.items():\n",
        '            mask = df["regime"] == state\n',
        '            if bias == "bullish":\n',
        "                hmm_bullish[mask] = True\n",
        '            elif bias == "bearish":\n',
        "                hmm_bearish[mask] = True\n",
        '    if "signal_score_long" in df.columns:\n',
        '        df["signal_score_long"] = df["signal_score_long"].astype(float) + hmm_bullish.astype(float) * W_HMM_REGIME\n',
        '    if "signal_score_short" in df.columns:\n',
        '        df["signal_score_short"] = df["signal_score_short"].astype(float) + hmm_bearish.astype(float) * W_HMM_REGIME\n',
        '    if "signal_score_long" in df.columns and "signal_score_short" in df.columns:\n',
        '        df["signal_long"] = (df["signal_score_long"] >= 65).astype(int)\n',
        '        df["signal_short"] = (df["signal_score_short"] >= 65).astype(int)\n',
        "    return df\n",
        "\n",
        "\n",
        "# -- MEJORA 2A: Detectar alertas tempranas de cambio de tendencia --\n",
        "def detect_early_alerts(df, states, state_summary):\n",
        '    """Detecta alertas tempranas usando regimen HMM + threshold reducido."""\n',
        "    state_bias_map = {}\n",
        "    for _, r in state_summary.iterrows():\n",
        '        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])\n',
        "    import numpy as np\n",
        "    regime_changed = np.zeros(len(df), dtype=bool)\n",
        "    for i in range(1, len(states)):\n",
        "        if states[i] != states[i-1]:\n",
        "            regime_changed[i] = True\n",
        "            for j in range(1, min(EARLY_WINDOW, len(states) - i)):\n",
        "                regime_changed[i + j] = True\n",
        '    df["alert_early_long"] = False\n',
        '    df["alert_early_short"] = False\n',
        "    for i in range(len(df)):\n",
        "        if not regime_changed[i]:\n",
        "            continue\n",
        "        current_state = states[i]\n",
        '        current_bias = state_bias_map.get(current_state, "neutral")\n',
        '        score_long = df["signal_score_long"].iloc[i] if "signal_score_long" in df.columns else 0\n',
        '        score_short = df["signal_score_short"].iloc[i] if "signal_score_short" in df.columns else 0\n',
        '        if current_bias == "bullish" and score_long >= EARLY_THRESHOLD:\n',
        '            df.loc[df.index[i], "alert_early_long"] = True\n',
        '        if current_bias == "bearish" and score_short >= EARLY_THRESHOLD:\n',
        '            df.loc[df.index[i], "alert_early_short"] = True\n',
        "    return df\n",
    ]
    lines = lines[:insert_at] + new_function_lines + lines[insert_at:]
    changes += 1
    print(f"[OK] Change 3: Added compute_signal_scores_with_hmm() and detect_early_alerts() after line {insert_at}")
else:
    print(f"[FAIL] Change 3: Could not find _smooth_states function")


# ============================================================
# CHANGE 4: Add smooth_states + HMM score + early alerts in main pipeline
# AFTER line: states, state_summary, _, trans_mat = fit_hmm(features_df)
# and AFTER: df["regime"] column assignment
# ============================================================
# Find the fit_hmm call in main
fit_line = None
for i in range(4900, min(len(lines), 5100)):
    if 'states, state_summary, _, trans_mat = fit_hmm(features_df)' in lines[i]:
        fit_line = i
        print(f"Found fit_hmm call at line {i+1}")
        break

# If not found with that exact string, try fuzzy
if fit_line is None:
    for i in range(4900, min(len(lines), 5100)):
        if 'fit_hmm(features_df)' in lines[i] and 'states' in lines[i]:
            fit_line = i
            print(f"Found fit_hmm variant at line {i+1}: {lines[i].strip()[:80]}")
            break

# Now find where "df["regime"] = ..." is assigned
regime_assign_line = None
for i in range(4900, min(len(lines), 5100)):
    if 'regime' in lines[i].lower() and '="' in lines[i] or "'" in lines[i] or '= ' in lines[i]:
        if "df[" in lines[i] and 'regime' in lines[i].lower():
            # Check it's an assignment
            stripped = lines[i].strip()
            if stripped.startswith('df[') and 'regime' in stripped.lower() and '=' in stripped:
                regime_assign_line = i
                print(f"Found df[regime] assignment at line {i+1}: {lines[i].strip()[:80]}")
                break

if fit_line is not None:
    # Insert the smoothing and HMM score logic AFTER the regime assignment (if found) or after fit_hmm call + 2 lines
    if regime_assign_line is not None:
        insert_at = regime_assign_line + 1
    else:
        insert_at = fit_line + 3
    
    new_main_lines = [
        "\n",
        "            # -- MEJORA 1B: Suavizar estados HMM (eliminar cambios espurios < 3 velas) --\n",
        "            states_smoothed = _smooth_states(states, min_duration=3)\n",
        "            df[\"regime\"] = states_smoothed\n",
        "\n",
        "            # -- MEJORA 1A: Recalcular scores con peso del regimen HMM --\n",
        "            df = compute_signal_scores_with_hmm(df, state_summary)\n",
        "\n",
        "            # -- MEJORA 2A: Detectar alertas tempranas --\n",
        "            df = detect_early_alerts(df, states_smoothed, state_summary)\n",
        "\n",
        "            # Actualizar states para el resto del pipeline\n",
        "            states = states_smoothed\n",
    ]
    lines = lines[:insert_at] + new_main_lines + lines[insert_at:]
    changes += 1
    print(f"[OK] Change 4: Added smooth_states + HMM score + early alerts after line {insert_at}")
else:
    print(f"[FAIL] Change 4: Could not find fit_hmm call in main()")


# ============================================================
# CHANGE 6: Remove the old regime_desc_col change (not needed anymore)
# The regime_desc approach is not needed because Change 5 already uses ATR proxy
# ============================================================
print(f"[INFO] Change 6: Skipped - not needed, Change 5 uses ATR proxy approach")


# ============================================================
# Write back
# ============================================================
with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n{'='*50}")
print(f"Total changes applied: {changes}")
print(f"{'='*50}")
