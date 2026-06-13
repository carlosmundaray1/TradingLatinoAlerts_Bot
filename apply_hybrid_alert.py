#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica el Sistema Híbrido Final: HMM + Precursores con Pesos
a tradinglatino_hmm_clean.py.
"""
import sys

with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 1: Add hybrid config constants after PRECURSOR_MIN_COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════
old_config_block = '''PRECURSOR_MIN_COMPONENTS: int = 4  # Componentes minimos activos para alerta

# -- MEJORA 2B: Reduccion de threshold por tipo de regimen --'''

new_config_block = '''PRECURSOR_MIN_COMPONENTS: int = 4  # Componentes minimos activos para alerta

# -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --
HYBRID_W_HMM: int = 30        # Peso del cambio de regimen HMM
HYBRID_W_PRECURSOR: int = 35  # Peso del precursor de componentes
HYBRID_W_VELOCITY: int = 20   # Peso de la velocidad del score hacia el threshold
HYBRID_W_ALIGNMENT: int = 15  # Peso de la alineacion regimen-senal
HYBRID_CONFIDENCE_THRESHOLD: int = 50  # Confianza minima (0-100) para activar alerta
HYBRID_LOOKBACK: int = 3      # Ventana de deteccion (velas hacia atras)
HYBRID_VELOCITY_THRESHOLD: float = 5.0  # Delta minimo de score para considerar velocidad

# -- MEJORA 2B: Reduccion de threshold por tipo de regimen --'''

if old_config_block in content:
    content = content.replace(old_config_block, new_config_block, 1)
    print("[OK] Change 1: Added hybrid config constants")
    changes += 1
else:
    print("[FAIL] Change 1: Could not find PRECURSOR_MIN_COMPONENTS block")

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 2: Add compute_hybrid_alert() function after compute_precursor_signals
# ═══════════════════════════════════════════════════════════════════════════

# Find the end of compute_precursor_signals - look for the line before _format_date
# The pattern is: the function ends with "df["precursor_active"] = ..." then "return df"
# followed by a blank line then "def _format_date"

hybrid_function = '''
# -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --

def compute_hybrid_alert(df: pd.DataFrame, states: np.ndarray, state_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Sistema hibrido final que combina HMM + Precursores con pesos ponderados
    para detectar cambios de tendencia LONG <-> SHORT.

    Para cada vela calcula un score de confianza (0-100) basado en:
      - HYBRID_W_HMM: Cambio de regimen HMM reciente con bias alineado a la nueva direccion
      - HYBRID_W_PRECURSOR: Precursor de componentes activo en la misma direccion
      - HYBRID_W_VELOCITY: Velocidad del signal score hacia el threshold
      - HYBRID_W_ALIGNMENT: Regimen actual alineado con la direccion senalada

    Columnas anadidas:
      - hybrid_confidence_long, hybrid_confidence_short: confianza 0-100
      - hybrid_alert_long, hybrid_alert_short: alerta activa (booleano)
      - hybrid_alert_active: True si alguna alerta activa
      - hybrid_debug: detalle de los componentes del score
    """
    df["hybrid_confidence_long"] = 0.0
    df["hybrid_confidence_short"] = 0.0
    df["hybrid_alert_long"] = False
    df["hybrid_alert_short"] = False
    df["hybrid_alert_active"] = False
    df["hybrid_debug"] = ""

    # Mapa de bias por estado
    state_bias_map = {}
    for _, r in state_summary.iterrows():
        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])

    # Detectar cambios de regimen
    regime_changed = np.zeros(len(df), dtype=bool)
    regime_new_bias = np.full(len(df), "neutral", dtype=object)
    for i in range(1, min(len(states), len(df))):
        if states[i] != states[i-1]:
            regime_changed[i] = True
            regime_new_bias[i] = state_bias_map.get(int(states[i]), "neutral")
            # Marcar las siguientes velas como "cambio reciente"
            lookahead = min(HYBRID_LOOKBACK, len(df) - i - 1)
            for j in range(1, lookahead + 1):
                regime_changed[i + j] = True
                if regime_new_bias[i + j] == "neutral":
                    regime_new_bias[i + j] = state_bias_map.get(int(states[i]), "neutral")

    # Calcular componentes del score para cada vela
    for i in range(len(df)):
        debug_parts = []

        # --- Componente 1: HMM Regime Change ---
        hmm_score = 0
        if regime_changed[i]:
            nb = regime_new_bias[i]
            if nb == "bullish":
                hmm_score = HYBRID_W_HMM
                debug_parts.append(f"HMM_ALCISTA+{HYBRID_W_HMM}")
            elif nb == "bearish":
                hmm_score = HYBRID_W_HMM
                debug_parts.append(f"HMM_BAJISTA+{HYBRID_W_HMM}")
            else:
                hmm_score = HYBRID_W_HMM // 2
                debug_parts.append(f"HMM_NEUTRAL+{HYBRID_W_HMM//2}")

        # --- Componente 2: Precursor activo ---
        precursor_long = False
        precursor_short = False
        precursor_long_window = False
        precursor_short_window = False

        if "precursor_long" in df.columns:
            # Mirar hacia atras HYBRID_LOOKBACK velas
            lookback_start = max(0, i - HYBRID_LOOKBACK)
            precursor_long_window = df["precursor_long"].iloc[lookback_start:i+1].any()
            precursor_short_window = df["precursor_short"].iloc[lookback_start:i+1].any()

        # Precursor en la vela actual
        precursor_long = bool(df["precursor_long"].iloc[i]) if "precursor_long" in df.columns else False
        precursor_short = bool(df["precursor_short"].iloc[i]) if "precursor_short" in df.columns else False

        precursor_score_long = 0
        precursor_score_short = 0
        if precursor_long_window:
            precursor_score_long = HYBRID_W_PRECURSOR
            debug_parts.append(f"PREC_LONG+{HYBRID_W_PRECURSOR}")
        if precursor_short_window:
            precursor_score_short = HYBRID_W_PRECURSOR
            debug_parts.append(f"PREC_SHORT+{HYBRID_W_PRECURSOR}")

        # --- Componente 3: Velocity del signal score ---
        velocity_long = 0.0
        velocity_short = 0.0
        if "signal_score_long" in df.columns and i > 0:
            if "signal_score_long" in df.columns:
                delta_l = float(df["signal_score_long"].iloc[i]) - float(df["signal_score_long"].iloc[i-1])
                if delta_l > HYBRID_VELOCITY_THRESHOLD:
                    velocity_long = delta_l
            if "signal_score_short" in df.columns:
                delta_s = float(df["signal_score_short"].iloc[i]) - float(df["signal_score_short"].iloc[i-1])
                if delta_s > HYBRID_VELOCITY_THRESHOLD:
                    velocity_short = delta_s

        velocity_score_long = min(HYBRID_W_VELOCITY, int(HYBRID_W_VELOCITY * (velocity_long / 10.0))) if velocity_long > 0 else 0
        velocity_score_short = min(HYBRID_W_VELOCITY, int(HYBRID_W_VELOCITY * (velocity_short / 10.0))) if velocity_short > 0 else 0
        if velocity_score_long > 0:
            debug_parts.append(f"VEL_LONG+{velocity_score_long}")
        if velocity_score_short > 0:
            debug_parts.append(f"VEL_SHORT+{velocity_score_short}")

        # --- Componente 4: Alignment regimen-senal ---
        alignment_score_long = 0
        alignment_score_short = 0
        if "regime" in df.columns and i < len(df):
            current_state = int(df["regime"].iloc[i]) if "regime" in df.columns else -1
            current_bias = state_bias_map.get(current_state, "neutral")
            if current_bias == "bullish":
                alignment_score_long = HYBRID_W_ALIGNMENT
                debug_parts.append(f"ALIGN_LONG+{HYBRID_W_ALIGNMENT}")
            elif current_bias == "bearish":
                alignment_score_short = HYBRID_W_ALIGNMENT
                debug_parts.append(f"ALIGN_SHORT+{HYBRID_W_ALIGNMENT}")

        # --- Confianza total ---
        # LONG: HMM alcista + precursor long + velocity long + alignment bullish
        # SHORT: HMM bajista + precursor short + velocity short + alignment bearish
        hmm_contrib_long = hmm_score if regime_new_bias[i] in ("bullish", "neutral") else 0
        hmm_contrib_short = hmm_score if regime_new_bias[i] in ("bearish", "neutral") else 0

        # Para HMM neutral, dividir el puntaje
        if regime_new_bias[i] == "neutral" and hmm_score > 0:
            hmm_contrib_long = hmm_score
            hmm_contrib_short = hmm_score

        conf_long = hmm_contrib_long + precursor_score_long + velocity_score_long + alignment_score_long
        conf_short = hmm_contrib_short + precursor_score_short + velocity_score_short + alignment_score_short

        # Normalizar a 0-100 (max possible = 100)
        conf_long = min(100, conf_long)
        conf_short = min(100, conf_short)

        df.at[df.index[i], "hybrid_confidence_long"] = conf_long
        df.at[df.index[i], "hybrid_confidence_short"] = conf_short
        df.at[df.index[i], "hybrid_alert_long"] = conf_long >= HYBRID_CONFIDENCE_THRESHOLD
        df.at[df.index[i], "hybrid_alert_short"] = conf_short >= HYBRID_CONFIDENCE_THRESHOLD
        df.at[df.index[i], "hybrid_debug"] = " | ".join(debug_parts)

    df["hybrid_alert_active"] = df["hybrid_alert_long"] | df["hybrid_alert_short"]
    return df


def _format_date(dt) -> str:
"""

# Find the pattern that ends compute_precursor_signals and starts _format_date
target = '''    df["precursor_active"] = df["precursor_long"] | df["precursor_short"]
    return df
def _format_date(dt) -> str:'''

replacement = '''    df["precursor_active"] = df["precursor_long"] | df["precursor_short"]
    return df
''' + hybrid_function

if target in content:
    content = content.replace(target, replacement, 1)
    print("[OK] Change 2: Added compute_hybrid_alert() function")
    changes += 1
else:
    print("[FAIL] Change 2: Could not find end of compute_precursor_signals")
    # Debug: find the pattern
    idx = content.find('precursor_active')
    if idx >= 0:
        print(f"  'precursor_active' found at {idx}")
        print(f"  Context: {repr(content[idx:idx+200])}")

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 3: Wire compute_hybrid_alert into the main pipeline
# ═══════════════════════════════════════════════════════════════════════════

# Find the call to compute_precursor_signals and add hybrid_alert after it
old_pipeline = '''    # -- ENFOQUE A: Detectar senales precursoras de cambios de tendencia --
    df = compute_precursor_signals(df)

    # 4) Senal actual'''

new_pipeline = '''    # -- ENFOQUE A: Detectar senales precursoras de cambios de tendencia --
    df = compute_precursor_signals(df)

    # -- SISTEMA HIBRIDO FINAL: HMM + Precursores con Pesos --
    df = compute_hybrid_alert(df, states, state_summary)

    # 4) Senal actual'''

if old_pipeline in content:
    content = content.replace(old_pipeline, new_pipeline, 1)
    print("[OK] Change 3: Wired compute_hybrid_alert into pipeline")
    changes += 1
else:
    print("[FAIL] Change 3: Could not find pipeline call to compute_precursor_signals")

# ═══════════════════════════════════════════════════════════════════════════
# Write the file back
# ═══════════════════════════════════════════════════════════════════════════
if changes > 0:
    with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nTotal changes applied: {changes}/3")
else:
    print("\nNo changes applied!")
