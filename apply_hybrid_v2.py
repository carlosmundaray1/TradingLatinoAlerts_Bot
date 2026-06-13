#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica el Sistema Híbrido Final: HMM + Precursores con Pesos
a tradinglatino_hmm_clean.py.
Usa line-by-line processing para evitar problemas de encoding y comillas.
"""
import sys

with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
changes = 0

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 1: Add hybrid config constants after PRECURSOR_MIN_COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════
for i, line in enumerate(lines):
    if 'PRECURSOR_MIN_COMPONENTS: int = 4' in line:
        insert = i + 1
        # The next line should be blank or the mejora 2B comment
        hybrid_constants = [
            "\n",
            "# -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --\n",
            "HYBRID_W_HMM: int = 30        # Peso del cambio de regimen HMM\n",
            "HYBRID_W_PRECURSOR: int = 35  # Peso del precursor de componentes\n",
            "HYBRID_W_VELOCITY: int = 20   # Peso de la velocidad del score hacia el threshold\n",
            "HYBRID_W_ALIGNMENT: int = 15  # Peso de la alineacion regimen-senal\n",
            "HYBRID_CONFIDENCE_THRESHOLD: int = 50  # Confianza minima (0-100) para activar alerta\n",
            "HYBRID_LOOKBACK: int = 3      # Ventana de deteccion (velas hacia atras)\n",
            "HYBRID_VELOCITY_THRESHOLD: float = 5.0  # Delta minimo de score para considerar velocidad\n",
            "\n",
        ]
        # Insert at position insert
        for j, hl in enumerate(hybrid_constants):
            lines.insert(insert + j, hl)
        print(f"[OK] Change 1: Added hybrid config constants after line {i+1}")
        changes += 1
        break
else:
    print("[FAIL] Change 1: PRECURSOR_MIN_COMPONENTS not found")

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 2: Add compute_hybrid_alert() function between compute_precursor_signals
# and _format_date
# ═══════════════════════════════════════════════════════════════════════════

# Find the "return df" line followed by "def _format_date"
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    stripped = line.rstrip('\n\r')
    if stripped == '    return df' and 'precursor_active' in ''.join(lines[max(0,i-3):i+1]):
        start_idx = i
    if start_idx is not None and i > start_idx and stripped.startswith('def _format_date'):
        end_idx = i
        break

if start_idx and end_idx:
    # Insert the hybrid function between the return df and _format_date
    hybrid_func_lines = [
        "\n",
        "# -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --\n",
        "\n",
        "def compute_hybrid_alert(df: pd.DataFrame, states: np.ndarray, state_summary: pd.DataFrame) -> pd.DataFrame:\n",
        '    """\n',
        "    Sistema hibrido final que combina HMM + Precursores con pesos ponderados\n",
        "    para detectar cambios de tendencia LONG <-> SHORT.\n",
        "\n",
        "    Para cada vela calcula un score de confianza (0-100) basado en:\n",
        "      - HYBRID_W_HMM: Cambio de regimen HMM reciente con bias alineado\n",
        "      - HYBRID_W_PRECURSOR: Precursor de componentes activo en la misma direccion\n",
        "      - HYBRID_W_VELOCITY: Velocidad del signal score hacia el threshold\n",
        "      - HYBRID_W_ALIGNMENT: Regimen actual alineado con la direccion senalada\n",
        '\n',
        "    Columnas anadidas:\n",
        "      - hybrid_confidence_long/short: confianza 0-100\n",
        "      - hybrid_alert_long/short: alerta activa (booleano)\n",
        "      - hybrid_alert_active: True si alguna alerta activa\n",
        '    """\n',
        '    df["hybrid_confidence_long"] = 0.0\n',
        '    df["hybrid_confidence_short"] = 0.0\n',
        '    df["hybrid_alert_long"] = False\n',
        '    df["hybrid_alert_short"] = False\n',
        '    df["hybrid_alert_active"] = False\n',
        '    df["hybrid_debug"] = ""\n',
        "\n",
        "    # Mapa de bias por estado\n",
        "    state_bias_map = {}\n",
        "    for _, r in state_summary.iterrows():\n",
        '        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])\n',
        "\n",
        "    # Detectar cambios de regimen\n",
        "    regime_changed = np.zeros(len(df), dtype=bool)\n",
        "    regime_new_bias = np.full(len(df), 'neutral', dtype=object)\n",
        "    for i in range(1, min(len(states), len(df))):\n",
        "        if states[i] != states[i-1]:\n",
        "            regime_changed[i] = True\n",
        "            regime_new_bias[i] = state_bias_map.get(int(states[i]), 'neutral')\n",
        "            # Marcar las siguientes velas como cambio reciente\n",
        "            lookahead = min(HYBRID_LOOKBACK, len(df) - i - 1)\n",
        "            for j in range(1, lookahead + 1):\n",
        "                regime_changed[i + j] = True\n",
        "                if regime_new_bias[i + j] == 'neutral':\n",
        "                    regime_new_bias[i + j] = state_bias_map.get(int(states[i]), 'neutral')\n",
        "\n",
        "    # Calcular componentes del score para cada vela\n",
        "    for i in range(len(df)):\n",
        "        debug_parts = []\n",
        "\n",
        "        # --- Componente 1: HMM Regime Change ---\n",
        "        hmm_score = 0\n",
        "        if regime_changed[i]:\n",
        "            nb = regime_new_bias[i]\n",
        '            if nb == "bullish":\n',
        "                hmm_score = HYBRID_W_HMM\n",
        "                debug_parts.append('HMM_ALCISTA+' + str(HYBRID_W_HMM))\n",
        '            elif nb == "bearish":\n',
        "                hmm_score = HYBRID_W_HMM\n",
        "                debug_parts.append('HMM_BAJISTA+' + str(HYBRID_W_HMM))\n",
        "            else:\n",
        "                hmm_score = HYBRID_W_HMM // 2\n",
        "                debug_parts.append('HMM_NEUTRAL+' + str(HYBRID_W_HMM // 2))\n",
        "\n",
        "        # --- Componente 2: Precursor activo ---\n",
        "        precursor_long_window = False\n",
        "        precursor_short_window = False\n",
        '        if "precursor_long" in df.columns:\n',
        "            lookback_start = max(0, i - HYBRID_LOOKBACK)\n",
        '            precursor_long_window = df["precursor_long"].iloc[lookback_start:i+1].any()\n',
        '            precursor_short_window = df["precursor_short"].iloc[lookback_start:i+1].any()\n',
        "\n",
        "        precursor_score_long = 0\n",
        "        precursor_score_short = 0\n",
        "        if precursor_long_window:\n",
        "            precursor_score_long = HYBRID_W_PRECURSOR\n",
        "            debug_parts.append('PREC_LONG+' + str(HYBRID_W_PRECURSOR))\n",
        "        if precursor_short_window:\n",
        "            precursor_score_short = HYBRID_W_PRECURSOR\n",
        "            debug_parts.append('PREC_SHORT+' + str(HYBRID_W_PRECURSOR))\n",
        "\n",
        "        # --- Componente 3: Velocity del signal score ---\n",
        "        velocity_long = 0.0\n",
        "        velocity_short = 0.0\n",
        '        if "signal_score_long" in df.columns and i > 0:\n',
        '            delta_l = float(df["signal_score_long"].iloc[i]) - float(df["signal_score_long"].iloc[i-1])\n',
        "            if delta_l > HYBRID_VELOCITY_THRESHOLD:\n",
        "                velocity_long = delta_l\n",
        '            delta_s = float(df["signal_score_short"].iloc[i]) - float(df["signal_score_short"].iloc[i-1])\n',
        "            if delta_s > HYBRID_VELOCITY_THRESHOLD:\n",
        "                velocity_short = delta_s\n",
        "\n",
        "        velocity_score_long = min(HYBRID_W_VELOCITY, int(HYBRID_W_VELOCITY * (velocity_long / 10.0))) if velocity_long > 0 else 0\n",
        "        velocity_score_short = min(HYBRID_W_VELOCITY, int(HYBRID_W_VELOCITY * (velocity_short / 10.0))) if velocity_short > 0 else 0\n",
        "        if velocity_score_long > 0:\n",
        "            debug_parts.append('VEL_LONG+' + str(velocity_score_long))\n",
        "        if velocity_score_short > 0:\n",
        "            debug_parts.append('VEL_SHORT+' + str(velocity_score_short))\n",
        "\n",
        "        # --- Componente 4: Alignment regimen-senal ---\n",
        "        alignment_score_long = 0\n",
        "        alignment_score_short = 0\n",
        '        if "regime" in df.columns:\n',
        '            current_state = int(df["regime"].iloc[i]) if "regime" in df.columns else -1\n',
        "            current_bias = state_bias_map.get(current_state, 'neutral')\n",
        '            if current_bias == "bullish":\n',
        "                alignment_score_long = HYBRID_W_ALIGNMENT\n",
        "                debug_parts.append('ALIGN_LONG+' + str(HYBRID_W_ALIGNMENT))\n",
        '            elif current_bias == "bearish":\n',
        "                alignment_score_short = HYBRID_W_ALIGNMENT\n",
        "                debug_parts.append('ALIGN_SHORT+' + str(HYBRID_W_ALIGNMENT))\n",
        "\n",
        "        # --- Confianza total ---\n",
        "        nb = regime_new_bias[i]\n",
        '        hmm_contrib_long = hmm_score if nb in ("bullish", "neutral") else 0\n',
        '        hmm_contrib_short = hmm_score if nb in ("bearish", "neutral") else 0\n',
        "\n",
        "        conf_long = hmm_contrib_long + precursor_score_long + velocity_score_long + alignment_score_long\n",
        "        conf_short = hmm_contrib_short + precursor_score_short + velocity_score_short + alignment_score_short\n",
        "\n",
        "        conf_long = min(100, conf_long)\n",
        "        conf_short = min(100, conf_short)\n",
        "\n",
        '        df.at[df.index[i], "hybrid_confidence_long"] = conf_long\n',
        '        df.at[df.index[i], "hybrid_confidence_short"] = conf_short\n',
        '        df.at[df.index[i], "hybrid_alert_long"] = conf_long >= HYBRID_CONFIDENCE_THRESHOLD\n',
        '        df.at[df.index[i], "hybrid_alert_short"] = conf_short >= HYBRID_CONFIDENCE_THRESHOLD\n',
        '        df.at[df.index[i], "hybrid_debug"] = " | ".join(debug_parts)\n',
        "\n",
        '    df["hybrid_alert_active"] = df["hybrid_alert_long"] | df["hybrid_alert_short"]\n',
        "    return df\n",
        "\n",
        "\n",
    ]
    # Insert the hybrid function BEFORE _format_date (at end_idx)
    for j, hl in enumerate(reversed(hybrid_func_lines)):
        lines.insert(end_idx, hl)
    print(f"[OK] Change 2: Added compute_hybrid_alert() function before _format_date (line {end_idx})")
    changes += 1
else:
    print(f"[FAIL] Change 2: Could not find end of compute_precursor_signals. start_idx={start_idx}, end_idx={end_idx}")
    # Debug
    for i, line in enumerate(lines):
        if 'precursor_active' in line:
            print(f"  precursor_active at line {i+1}: {line.rstrip()[:100]}")
        if 'def _format_date' in line:
            print(f"  _format_date at line {i+1}: {line.rstrip()[:80]}")

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 3: Wire compute_hybrid_alert into the main pipeline
# ═══════════════════════════════════════════════════════════════════════════
for i, line in enumerate(lines):
    stripped = line.rstrip('\n\r')
    if stripped == 'df = compute_precursor_signals(df)':
        # Insert after this line
        insert_at = i + 1
        pipeline_lines = [
            "\n",
            "    # -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --\n",
            "    df = compute_hybrid_alert(df, states, state_summary)\n",
        ]
        for j, pl in enumerate(reversed(pipeline_lines)):
            lines.insert(insert_at, pl)
        print(f"[OK] Change 3: Wired compute_hybrid_alert into pipeline after line {i+1}")
        changes += 1
        break
else:
    print("[FAIL] Change 3: Could not find 'df = compute_precursor_signals(df)'")

# ═══════════════════════════════════════════════════════════════════════════
# Write file
# ═══════════════════════════════════════════════════════════════════════════
if changes > 0:
    with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nTotal changes applied: {changes}/3")
else:
    print("\nNo changes applied!")
