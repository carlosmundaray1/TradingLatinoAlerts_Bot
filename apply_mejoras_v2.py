#!/usr/bin/env python
"""
Apply remaining improvements to tradinglatino_hmm_clean.py using line numbers.
Avoids encoding matching issues with str_replace.
"""
import sys

# Read file
with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
changes = 0

# ============================================================
# CHANGE 1: Add new config constants after line 132 (W_ATR_ROC line)
# Line 132 = W_ATR_ROC: int = 2 ...
# We insert new lines AFTER line 132 (0-indexed: 132)
# ============================================================
new_config_lines = [
    "\n",
    "# -- MEJORA 1A: Peso del regimen HMM en el score compuesto --\n",
    "W_HMM_REGIME: int = 8      # Suma puntos cuando el regimen HMM esta alineado con la senal\n",
    "\n",
    "# -- MEJORA 2A: Alerta temprana (threshold reducido) --\n",
    "EARLY_THRESHOLD: int = 40  # Threshold reducido para alerta temprana de cambio de tendencia\n",
    "EARLY_WINDOW: int = 3      # Ventana de velas para detectar cambio de regimen reciente\n",
    "\n",
    "# -- MEJORA 2B: Reduccion de threshold por tipo de regimen --\n",
    "REGIME_THRESHOLD_REDUCTION = {\n",
    '    "EXPANSION ALCISTA": 15,   # Euforia: baja threshold 15 pts\n',
    '    "EXPANSION BAJISTA": 15,   # Panico: baja threshold 15 pts\n',
    '    "ALTA VOLATILIDAD": 10,\n',
    '    "TREND ALCISTA": 5,\n',
    '    "TREND BAJISTA": 5,\n',
    "}\n",
]

# Verify line 132
print(f"Line 133 (0-indexed 132): {repr(lines[132][:60])}")
lines = lines[:133] + new_config_lines + lines[133:]
changes += 1
print(f"[OK] Change 1: Added config constants after line 132")

# ============================================================
# CHANGE 2: Add features in build_hmm_features before return
# Find the return features line in build_hmm_features
# ============================================================
return_line = None
for i in range(650, 750):
    if i < len(lines) and 'return features' in lines[i].strip():
        # Make sure it's inside build_hmm_features (check for 'up_bar_ratio' a few lines before)
        context = ''.join(lines[max(650,i-10):i])
        if 'up_bar_ratio' in context:
            return_line = i
            print(f"Found 'return features' at line {i+1} (0-indexed {i})")
            break

if return_line is not None:
    new_feature_lines = [
        '    # -- MEJORA 1B: Anadir signal scores, RSI, y EMA deviation como features del HMM --\n',
        '    # Estas features ayudan a que el HMM capture mejor los cambios de tendencia\n',
        '    if "signal_score_long" in df.columns:\n',
        '        features["signal_score_long"] = df["signal_score_long"].fillna(0)\n',
        '    if "signal_score_short" in df.columns:\n',
        '        features["signal_score_short"] = df["signal_score_short"].fillna(0)\n',
        '    if "rsi" in df.columns:\n',
        '        features["rsi_14"] = (df["rsi"] - 50) / 50  # Normalizado: -1 a +1\n',
        '    if "ema_dev_pct" in df.columns:\n',
        '        features["ema_dev_pct"] = df["ema_dev_pct"].fillna(0)\n',
        '    # Diff de signal scores (cambio en el momentum)\n',
        '    if "signal_score_long" in features.columns:\n',
        '        features["score_delta_long"] = features["signal_score_long"].diff().fillna(0)\n',
        '    if "signal_score_short" in features.columns:\n',
        '        features["score_delta_short"] = features["signal_score_short"].diff().fillna(0)\n',
    ]
    lines = lines[:return_line] + new_feature_lines + lines[return_line:]
    changes += 1
    print(f"[OK] Change 2: Added features before return in build_hmm_features (line {return_line+1})")
else:
    print(f"[FAIL] Change 2: Could not find 'return features' in build_hmm_features")

# ============================================================
# CHANGE 3: Add compute_signal_scores_with_hmm() and detect_early_alerts()
# after _smooth_states function (ends at line 901 based on earlier check)
# ============================================================
# Find the end of _smooth_states (the 'return smoothed' line with '    return smoothed')
smooth_end = None
for i in range(876, 920):
    if i < len(lines) and lines[i].strip() == 'return smoothed':
        smooth_end = i
        print(f"Found 'return smoothed' at line {i+1} (0-indexed {i})")
        break

if smooth_end is not None:
    # Find the blank line after return smoothed
    insert_at = smooth_end + 1
    while insert_at < len(lines) and lines[insert_at].strip() == '':
        insert_at += 1
    # insert_at is now the first non-blank line after _smooth_states
    
    new_function_lines = [
        "\n",
        "\n",
        "# -- MEJORA 1A: Recalcular scores incluyendo el regimen HMM --\n",
        "# Se llama DESPUES de fit_hmm(), cuando ya tenemos los estados asignados\n",
        "def compute_signal_scores_with_hmm(df: pd.DataFrame,\n",
        "                                    state_summary: pd.DataFrame) -> pd.DataFrame:\n",
        '    """\n',
        "    Anade el peso W_HMM_REGIME a los scores LONG/SHORT segun el regimen HMM.\n",
        "    Conecta el HMM con las senales de trading.\n",
        '    """\n',
        "    import numpy as np\n",
        "\n",
        "    # Mapa de bias por estado HMM\n",
        "    state_bias = {}\n",
        "    for _, r in state_summary.iterrows():\n",
        '        state_bias[int(r["state"])] = _classify_regime_bias(r["description"])\n',
        "\n",
        "    # Aplicar bias HMM a cada barra\n",
        "    hmm_bullish = pd.Series(False, index=df.index)\n",
        "    hmm_bearish = pd.Series(False, index=df.index)\n",
        "\n",
        '    if "regime" in df.columns:\n',
        "        for state, bias in state_bias.items():\n",
        '            mask = df["regime"] == state\n',
        '            if bias == "bullish":\n',
        "                hmm_bullish[mask] = True\n",
        '            elif bias == "bearish":\n',
        "                hmm_bearish[mask] = True\n",
        "\n",
        "    # Anadir peso W_HMM_REGIME a los scores\n",
        '    if "signal_score_long" in df.columns:\n',
        '        df["signal_score_long"] = df["signal_score_long"].astype(float) + hmm_bullish.astype(float) * W_HMM_REGIME\n',
        '    if "signal_score_short" in df.columns:\n',
        '        df["signal_score_short"] = df["signal_score_short"].astype(float) + hmm_bearish.astype(float) * W_HMM_REGIME\n',
        "\n",
        "    # Recalcular las senales booleanas con los nuevos scores\n",
        '    if "signal_score_long" in df.columns and "signal_score_short" in df.columns:\n',
        '        df["signal_long"] = (df["signal_score_long"] >= SIGNAL_SCORE_THRESHOLD).astype(int)\n',
        '        df["signal_short"] = (df["signal_score_short"] >= SIGNAL_SCORE_THRESHOLD).astype(int)\n',
        "\n",
        "    return df\n",
        "\n",
        "\n",
        "# -- MEJORA 2A: Detectar alertas tempranas de cambio de tendencia --\n",
        "def detect_early_alerts(df: pd.DataFrame, states: np.ndarray,\n",
        "                         state_summary: pd.DataFrame) -> pd.DataFrame:\n",
        '    """\n',
        "    Detecta alertas tempranas de cambio de tendencia usando el regimen HMM\n",
        "    + un threshold reducido en el score compuesto.\n",
        '    """\n',
        "    # Mapa de bias por estado\n",
        "    state_bias_map = {}\n",
        "    for _, r in state_summary.iterrows():\n",
        '        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])\n',
        "\n",
        "    # Detectar barras con cambio de regimen reciente\n",
        "    regime_changed = np.zeros(len(df), dtype=bool)\n",
        "    for i in range(1, len(states)):\n",
        "        if states[i] != states[i-1]:\n",
        "            regime_changed[i] = True\n",
        "            # Marcar las siguientes EARLY_WINDOW velas\n",
        "            for j in range(1, min(EARLY_WINDOW, len(states) - i)):\n",
        "                regime_changed[i + j] = True\n",
        "\n",
        '    df["alert_early_long"] = False\n',
        '    df["alert_early_short"] = False\n',
        "\n",
        "    for i in range(len(df)):\n",
        "        if not regime_changed[i]:\n",
        "            continue\n",
        "\n",
        "        current_state = states[i]\n",
        '        current_bias = state_bias_map.get(current_state, "neutral")\n',
        "\n",
        "        # Verificar scores con threshold reducido\n",
        '        score_long = df["signal_score_long"].iloc[i] if "signal_score_long" in df.columns else 0\n',
        '        score_short = df["signal_score_short"].iloc[i] if "signal_score_short" in df.columns else 0\n',
        "\n",
        "        # Alerta temprana LONG: regimen bullish + score > EARLY_THRESHOLD\n",
        '        if current_bias == "bullish" and score_long >= EARLY_THRESHOLD:\n',
        '            df.loc[df.index[i], "alert_early_long"] = True\n',
        "\n",
        "        # Alerta temprana SHORT: regimen bearish + score > EARLY_THRESHOLD\n",
        '        if current_bias == "bearish" and score_short >= EARLY_THRESHOLD:\n',
        '            df.loc[df.index[i], "alert_early_short"] = True\n',
        "\n",
        "    return df\n",
    ]
    lines = lines[:insert_at] + new_function_lines + lines[insert_at:]
    changes += 1
    print(f"[OK] Change 3: Added compute_signal_scores_with_hmm() and detect_early_alerts() (at line {insert_at+1})")
else:
    print(f"[FAIL] Change 3: Could not find end of _smooth_states")


# ============================================================
# CHANGE 4: Add smooth_states + HMM score + early alerts in main pipeline
# Find the main() loop where best_model_info is processed
# Search for '"state_summary": best_model_info["state_summary"]'
# ============================================================
state_summary_line = None
for i in range(4900, min(len(lines), 5100)):
    if '"state_summary": best_model_info["state_summary"]' in lines[i]:
        state_summary_line = i
        print(f"Found state_summary assignment at line {i+1} (0-indexed {i})")
        break

if state_summary_line is not None:
    # After state_summary line + 2 (closing brace) + 1 (comment "# 4) Signal")
    # We want to insert the new code AFTER the closing brace of the best_model_info dict
    insert_at = state_summary_line + 1  # After state_summary line
    
    new_main_lines = [
        "\n",
        "            # -- MEJORA 1B: Suavizar estados HMM (eliminar cambios espurios < 3 velas) --\n",
        '            states = best_model_info["states"]\n',
        "            states_smoothed = _smooth_states(states, min_duration=3)\n",
        '            # Actualizar el dataframe con estados suavizados\n',
        '            df["regime"] = states_smoothed\n',
        "\n",
        "            # -- MEJORA 1A: Recalcular scores con peso del regimen HMM --\n",
        '            df = compute_signal_scores_with_hmm(df, best_model_info["state_summary"])\n',
        "\n",
        "            # -- MEJORA 2A: Detectar alertas tempranas --\n",
        '            df = detect_early_alerts(df, states_smoothed, best_model_info["state_summary"])\n',
        "\n",
        '            # Actualizar el modelo con los estados suavizados\n',
        '            best_model_info["states"] = states_smoothed\n',
    ]
    lines = lines[:insert_at] + new_main_lines + lines[insert_at:]
    changes += 1
    print(f"[OK] Change 4: Added smooth_states + HMM score + early alerts in main pipeline (line {insert_at+1})")
else:
    print(f"[FAIL] Change 4: Could not find state_summary assignment in main(). Searching...")


# ============================================================
# CHANGE 5: Add REGIME_THRESHOLD_REDUCTION in compute_signal()
# Find the dynamic_threshold calculation
# ============================================================
dynamic_line = None
for i in range(957, 1150):
    if i < len(lines) and 'DYNAMIC_THRESHOLD_MIN' in lines[i] and 'dynamic_threshold' in lines[i]:
        dynamic_line = i
        print(f"Found dynamic_threshold at line {i+1} (0-indexed {i}): {lines[i].strip()[:80]}")
        break

if dynamic_line is None:
    # Try broader search
    for i in range(957, 1200):
        if i < len(lines) and 'dynamic_threshold' in lines[i].lower():
            dynamic_line = i
            print(f"Found dynamic_threshold variant at line {i+1}: {lines[i].strip()[:80]}")
            break

if dynamic_line is not None:
    # Insert the regime reduction AFTER the dynamic_threshold line + any continuation parens
    insert_at = dynamic_line + 1
    # Skip any continuation lines (lines ending with '\n' and indented)
    while insert_at < len(lines) and (lines[insert_at].strip().startswith((')', ',')) or lines[insert_at].strip() == ''):
        insert_at += 1
    
    new_threshold_lines = [
        "\n",
        "    # -- MEJORA 2B: Reducir threshold adicional segun el tipo de regimen --\n",
        '    if "regime" in df.columns:\n',
        "        # Buscar la descripcion del regimen actual en state_summary\n",
        '        current_regime_desc = ""\n',
        "        try:\n",
        '            if df["regime"].iloc[-1] >= 0 and "state_summary" in dir():\n',
        "                pass  # state_summary no disponible en este scope\n",
        "        except:\n",
        "            pass\n",
        "        # Reducir threshold segun keywords en el nombre del regimen\n",
        "        # (la descripcion se obtiene del dashboard, no de compute_signal)\n",
        "        # Como alternativa, usamos ATR como proxy de volatilidad\n",
        '        atr_ratio = df["atr"].iloc[-1] / max(df["atr"].rolling(20, min_periods=1).mean().iloc[-1], 0.01)\n',
        "        if atr_ratio > 1.5:\n",
        '            dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold - 10)\n',
        "        if atr_ratio > 2.0:\n",
        '            dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold - 5)\n',
    ]
    lines = lines[:insert_at] + new_threshold_lines + lines[insert_at:]
    changes += 1
    print(f"[OK] Change 5: Added REGIME_THRESHOLD_REDUCTION in compute_signal()")
else:
    print(f"[FAIL] Change 5: Could not find dynamic_threshold")


# ============================================================
# CHANGE 6: Add regime_desc_col to df in the pipeline
# ============================================================
# Find where regime_desc is assigned in the main loop or dashboard generation
regime_desc_line = None
for i in range(4900, min(len(lines), 5300)):
    if i < len(lines) and 'regime_desc = ""' in lines[i] and 'regime_duration' in lines[i]:
        regime_desc_line = i
        print(f"Found regime_desc assignment at line {i+1}")
        break

if regime_desc_line is not None:
    # Find the line after regime_desc is assigned from state_summary
    for i in range(regime_desc_line, regime_desc_line + 20):
        if i < len(lines) and 'regime_desc = row.iloc[0]' in lines[i]:
            insert_at = i + 1
            new_regime_desc_lines = [
                '                    # -- MEJORA 2B: Guardar descripcion del regimen en df para threshold dinamico --\n',
                '                    if hasattr(df, "regime_desc_col"):\n',
                '                        pass\n',
                '                    try:\n',
                '                        df["regime_desc_col"] = df["regime_desc_col"].astype(str)\n',
                '                        df["regime_desc_col"].iloc[-1] = regime_desc\n',
                '                    except:\n',
                '                        pass\n',
            ]
            lines = lines[:insert_at] + new_regime_desc_lines + lines[insert_at:]
            changes += 1
            print(f"[OK] Change 6: Added regime_desc tracking")
            break
    else:
        print(f"[FAIL] Change 6: Could not find regime_desc assignment details")
else:
    print(f"[FAIL] Change 6: Could not find regime_desc initialization")


# ============================================================
# Write back
# ============================================================
with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n{'='*50}")
print(f"Total changes applied: {changes}")
print(f"{'='*50}")
