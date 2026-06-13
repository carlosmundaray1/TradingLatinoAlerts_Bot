#!/usr/bin/env python
"""
Implementa el Enfoque A: Sistema de Precursores.
Anade compute_precursor_signals() y conecta en el pipeline.
"""
import sys

with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

changes = 0

# ============================================================
# CHANGE 1: Add PRECURSOR_THRESHOLD config constant
# Insert after EARLY_WINDOW config constant
# ============================================================
config_insert = None
for i, line in enumerate(lines):
    if 'EARLY_WINDOW: int = 3' in line:
        config_insert = i + 1
        print(f"Found EARLY_WINDOW at line {i+1}")
        break

if config_insert:
    new_config = [
        "\n",
        "# -- ENFOQUE A: Threshold para alertas precursoras (score antes del cruce) --\n",
        "PRECURSOR_THRESHOLD: int = 45  # Score minimo para activar alerta precursora\n",
        "PRECURSOR_VELOCITY_BARS: int = 5  # Ventana para calcular velocidad del score\n",
        "PRECURSOR_MIN_COMPONENTS: int = 4  # Componentes minimos activos para alerta\n",
    ]
    lines = lines[:config_insert] + new_config + lines[config_insert:]
    changes += 1
    print(f"[OK] Change 1: Added PRECURSOR config constants")
else:
    print(f"[FAIL] Change 1: Could not find insertion point")

# ============================================================
# CHANGE 2: Add compute_precursor_signals() function
# Insert after detect_early_alerts function (find its end)
# ============================================================
func_insert = None
for i, line in enumerate(lines):
    if 'def detect_early_alerts(df, states, state_summary):' in line:
        # Find the end of this function (next def)
        for j in range(i+1, min(i+100, len(lines))):
            if lines[j].startswith('def ') and j > i+5:
                func_insert = j
                print(f"Found detect_early_alerts end, next def at line {j+1}")
                break
        break

if func_insert:
    new_func_lines = [
        "\n",
        "\n",
        "# -- ENFOQUE A: Sistema de Precursores (detectar cambios de tendencia ANTES del cruce) --\n",
        "def compute_precursor_signals(df: pd.DataFrame) -> pd.DataFrame:\n",
        '    """\n',
        "    Sistema de alertas precursoras para cambios de tendencia.\n",
        "    Monitorea los COMPONENTES del score compuesto que se acercan al threshold (65).\n",
        "    Genera alertas cuando el score esta en zona de advertencia (45-64) y subiendo.\n",
        '    """\n',
        "    # Verificar que tenemos los datos necesarios\n",
        '    if "signal_score_long" not in df.columns or "signal_score_short" not in df.columns:\n',
        "        return df\n",
        "\n",
        "    # Inicializar columnas\n",
        '    df["precursor_long"] = False\n',
        '    df["precursor_short"] = False\n',
        '    df["precursor_confidence_long"] = 0.0\n',
        '    df["precursor_confidence_short"] = 0.0\n',
        '    df["precursor_active"] = False\n',
        "\n",
        "    for i in range(PRECURSOR_VELOCITY_BARS, len(df)):\n",
        "        # --- PROCESAR PRECURSOR LONG ---\n",
        '        score_long = df["signal_score_long"].iloc[i]\n',
        "\n",
        "        # Calcular velocidad del score (pendiente en los ultimos N velas)\n",
        '        slice_long = df["signal_score_long"].iloc[i-PRECURSOR_VELOCITY_BARS:i+1].values\n',
        "        if len(slice_long) >= 2:\n",
        "            x = list(range(len(slice_long)))\n",
        "            y = slice_long\n",
        "            n = len(x)\n",
        "            slope_long = (n * sum(x[j]*y[j] for j in range(n)) - sum(x)*sum(y)) / (n * sum(x[j]*x[j] for j in range(n)) - sum(x)*sum(x) + 0.001)\n",
        "        else:\n",
        "            slope_long = 0\n",
        "\n",
        "        # Componentes activos para LONG\n",
        "        components_long = 0\n",
        '        if "bull_bias" in df.columns and df["bull_bias"].iloc[i]:\n',
        "            components_long += 1\n",
        '        if "squeeze_off" in df.columns and df["squeeze_off"].iloc[i]:\n',
        "            components_long += 1\n",
        '        if "smi_hist" in df.columns and df["smi_hist"].iloc[i] > 0:\n',
        "            components_long += 1\n",
        '        if "smi_delta" in df.columns and df["smi_delta"].iloc[i] > 0:\n',
        "            components_long += 1\n",
        '        if "adx_delta" in df.columns and df["adx_delta"].iloc[i] > 0:\n',
        "            components_long += 1\n",
        '        if "plus_di" in df.columns and "minus_di" in df.columns and df["plus_di"].iloc[i] > df["minus_di"].iloc[i]:\n',
        "            components_long += 1\n",
        "\n",
        "        # Determinar alerta precursora LONG\n",
        "        if (PRECURSOR_THRESHOLD <= score_long < SIGNAL_SCORE_THRESHOLD\n",
        "                and slope_long > 0.5\n",
        "                and components_long >= PRECURSOR_MIN_COMPONENTS):\n",
        '            df.loc[df.index[i], "precursor_long"] = True\n',
        '            df.loc[df.index[i], "precursor_confidence_long"] = round(\n',
        "                min(100, (score_long / SIGNAL_SCORE_THRESHOLD) * 100 * (components_long / 6))\n",
        "            , 1)\n",
        '            df.loc[df.index[i], "precursor_active"] = True\n',
        "\n",
        "        # --- PROCESAR PRECURSOR SHORT ---\n",
        '        score_short = df["signal_score_short"].iloc[i]\n',
        "\n",
        "        # Calcular velocidad del score SHORT\n",
        '        slice_short = df["signal_score_short"].iloc[i-PRECURSOR_VELOCITY_BARS:i+1].values\n',
        "        if len(slice_short) >= 2:\n",
        "            x = list(range(len(slice_short)))\n",
        "            y = slice_short\n",
        "            n = len(x)\n",
        "            slope_short = (n * sum(x[j]*y[j] for j in range(n)) - sum(x)*sum(y)) / (n * sum(x[j]*x[j] for j in range(n)) - sum(x)*sum(x) + 0.001)\n",
        "        else:\n",
        "            slope_short = 0\n",
        "\n",
        "        # Componentes activos para SHORT\n",
        "        components_short = 0\n",
        '        if "bear_bias" in df.columns and df["bear_bias"].iloc[i]:\n',
        "            components_short += 1\n",
        '        if "squeeze_off" in df.columns and df["squeeze_off"].iloc[i]:\n',
        "            components_short += 1\n",
        '        if "smi_hist" in df.columns and df["smi_hist"].iloc[i] < 0:\n',
        "            components_short += 1\n",
        '        if "smi_delta" in df.columns and df["smi_delta"].iloc[i] < 0:\n',
        "            components_short += 1\n",
        '        if "adx_delta" in df.columns and df["adx_delta"].iloc[i] > 0:\n',
        "            components_short += 1\n",
        '        if "plus_di" in df.columns and "minus_di" in df.columns and df["minus_di"].iloc[i] > df["plus_di"].iloc[i]:\n',
        "            components_short += 1\n",
        "\n",
        "        # Determinar alerta precursora SHORT\n",
        "        if (PRECURSOR_THRESHOLD <= score_short < SIGNAL_SCORE_THRESHOLD\n",
        "                and slope_short > 0.5\n",
        "                and components_short >= PRECURSOR_MIN_COMPONENTS):\n",
        '            df.loc[df.index[i], "precursor_short"] = True\n',
        '            df.loc[df.index[i], "precursor_confidence_short"] = round(\n',
        "                min(100, (score_short / SIGNAL_SCORE_THRESHOLD) * 100 * (components_short / 6))\n",
        "            , 1)\n",
        '            df.loc[df.index[i], "precursor_active"] = True\n',
        "\n",
        "    return df\n",
    ]
    lines = lines[:func_insert] + new_func_lines + lines[func_insert:]
    changes += 1
    print(f"[OK] Change 2: Added compute_precursor_signals() function")
else:
    print(f"[FAIL] Change 2: Could not find insertion point")
    # Fallback: insert before compute_signal_scores_with_hmm
    for i, line in enumerate(lines):
        if 'def compute_signal_scores_with_hmm' in line:
            lines = lines[:i] + new_func_lines + lines[i:]
            changes += 1
            print(f"[OK] Change 2 (fallback): Inserted before compute_signal_scores_with_hmm")
            break


# ============================================================
# CHANGE 3: Wire compute_precursor_signals into the main pipeline
# Find where compute_signal is called and add precursor computation before it
# ============================================================
# Search for "df = compute_signal_scores_with_hmm" in main pipeline
pipeline_insert = None
for i, line in enumerate(lines):
    if 'df = compute_signal_scores_with_hmm(df, state_summary)' in line:
        pipeline_insert = i + 1
        print(f"Found compute_signal_scores_with_hmm call at line {i+1}")
        break

if pipeline_insert:
    new_pipeline_lines = [
        "\n",
        "            # -- ENFOQUE A: Detectar senales precursoras de cambios de tendencia --\n",
        "            df = compute_precursor_signals(df)\n",
    ]
    lines = lines[:pipeline_insert] + new_pipeline_lines + lines[pipeline_insert:]
    changes += 1
    print(f"[OK] Change 3: Wired compute_precursor_signals into pipeline")
else:
    print(f"[FAIL] Change 3: Could not find pipeline insertion point")


# ============================================================
# Write back
# ============================================================
with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n{'='*50}")
print(f"Total changes applied: {changes}")
print(f"{'='*50}")
