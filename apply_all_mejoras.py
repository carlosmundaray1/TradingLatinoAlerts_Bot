#!/usr/bin/env python
"""
Apply all remaining improvements to tradinglatino_hmm_clean.py
Changes: 1A, 1B, 2A, 2B (3A partially already applied)
"""

import sys

with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

changes = 0

# ============================================================
# CHANGE 1: Add new config constants after W_ATR_ROC (line ~131)
# ============================================================
target = 'W_ATR_ROC: int = 2        # ATR Rate of Change (reducido Opcion C)'
new_config = '''W_ATR_ROC: int = 2        # ATR Rate of Change (reducido Opcion C)

# -- MEJORA 1A: Peso del regimen HMM en el score compuesto --
W_HMM_REGIME: int = 8      # Suma puntos cuando el regimen HMM esta alineado con la senal

# -- MEJORA 2A: Alerta temprana (threshold reducido) --
EARLY_THRESHOLD: int = 40  # Threshold reducido para alerta temprana de cambio de tendencia
EARLY_WINDOW: int = 3      # Ventana de velas para detectar cambio de regimen reciente

# -- MEJORA 2B: Reduccion de threshold por tipo de regimen --
REGIME_THRESHOLD_REDUCTION = {
    "EXPANSION ALCISTA": 15,   # Euforia: baja threshold 15 pts
    "EXPANSION BAJISTA": 15,   # Panico: baja threshold 15 pts
    "ALTA VOLATILIDAD": 10,
    "TREND ALCISTA": 5,
    "TREND BAJISTA": 5,
}'''

if target in content:
    content = content.replace(target, new_config)
    changes += 1
    print(f"[OK] Change 1: Added new config constants (W_HMM_REGIME, EARLY_THRESHOLD, REGIME_THRESHOLD_REDUCTION)")
else:
    print(f"[FAIL] Change 1: Target not found: {target[:60]}...")

# ============================================================
# CHANGE 2: Add signal scores, RSI, EMA dev as HMM features in build_hmm_features
# ============================================================
# Find the return statement of build_hmm_features
# The function ends with 'return features'
# We need to add features before the return

target_feature_return = "return features"
# Find the build_hmm_features return by context
# We need to add features inside the function body

old_features_section = """    features["up_bar_ratio"] = (
        features["log_return_1"].gt(0).rolling(window=FEATURE_WINDOW).sum()
        / FEATURE_WINDOW
    )

    return features"""

new_features_section = """    features["up_bar_ratio"] = (
        features["log_return_1"].gt(0).rolling(window=FEATURE_WINDOW).sum()
        / FEATURE_WINDOW
    )

    # -- MEJORA 1B: Anadir signal scores, RSI, y EMA deviation como features del HMM --
    # Estas features ayudan a que el HMM capture mejor los cambios de tendencia
    if "signal_score_long" in df.columns:
        features["signal_score_long"] = df["signal_score_long"].fillna(0)
    if "signal_score_short" in df.columns:
        features["signal_score_short"] = df["signal_score_short"].fillna(0)
    if "rsi" in df.columns:
        features["rsi_14"] = (df["rsi"] - 50) / 50  # Normalizado: -1 a +1
    if "ema_dev_pct" in df.columns:
        features["ema_dev_pct"] = df["ema_dev_pct"].fillna(0)
    # Diff de signal scores (cambio en el momentum)
    if "signal_score_long" in features.columns:
        features["score_delta_long"] = features["signal_score_long"].diff().fillna(0)
    if "signal_score_short" in features.columns:
        features["score_delta_short"] = features["signal_score_short"].diff().fillna(0)

    return features"""

if old_features_section in content:
    content = content.replace(old_features_section, new_features_section)
    changes += 1
    print(f"[OK] Change 2: Added signal scores, RSI, EMA dev as HMM features")
else:
    print(f"[FAIL] Change 2: Target features return not found")

# ============================================================
# CHANGE 3: Add compute_signal_scores_with_hmm() function after _smooth_states
# ============================================================
target_smooth_end = '''def _smooth_states(states: np.ndarray, min_duration: int = 3) -> np.ndarray:
    """
    Suaviza la secuencia de estados HMM eliminando cambios espurios de regimen.
    Reduce falsos positivos por cambios espurios de regimen.
    Si un estado aparece por menos de min_duration velas consecutivas,
    se reemplaza por el estado anterior persistente.
    """
    if len(states) == 0:
        return states
    smoothed = states.copy()
    n = len(smoothed)
    for i in range(1, n):
        if smoothed[i] != smoothed[i-1]:
            lookback = max(0, i - min_duration)
            # Verificar cuantas velas atras fue el ultimo cambio
            if i - lookback < min_duration:
                smoothed[i] = smoothed[i-1]
            else:
                # Verificar si el nuevo estado es transitorio
                j = i
                while j < n and smoothed[j] == smoothed[i]:
                    j += 1
                if j - i < min_duration:
                    smoothed[i:j] = smoothed[i-1]
    return smoothed'''

target_smooth_new = '''def _smooth_states(states: np.ndarray, min_duration: int = 3) -> np.ndarray:
    """
    Suaviza la secuencia de estados HMM eliminando cambios espurios de regimen.
    Reduce falsos positivos por cambios espurios de regimen.
    Si un estado aparece por menos de min_duration velas consecutivas,
    se reemplaza por el estado anterior persistente.
    """
    if len(states) == 0:
        return states
    smoothed = states.copy()
    n = len(smoothed)
    for i in range(1, n):
        if smoothed[i] != smoothed[i-1]:
            lookback = max(0, i - min_duration)
            # Verificar cuantas velas atras fue el ultimo cambio
            if i - lookback < min_duration:
                smoothed[i] = smoothed[i-1]
            else:
                # Verificar si el nuevo estado es transitorio
                j = i
                while j < n and smoothed[j] == smoothed[i]:
                    j += 1
                if j - i < min_duration:
                    smoothed[i:j] = smoothed[i-1]
    return smoothed


# -- MEJORA 1A: Recalcular scores incluyendo el regimen HMM --
# Se llama DESPUES de fit_hmm(), cuando ya tenemos los estados asignados
def compute_signal_scores_with_hmm(df: pd.DataFrame,
                                    state_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Anade el peso W_HMM_REGIME a los scores LONG/SHORT segun el regimen HMM.
    Conecta el HMM con las senales de trading.
    """
    import numpy as np

    # Mapa de bias por estado HMM
    state_bias = {}
    for _, r in state_summary.iterrows():
        state_bias[int(r["state"])] = _classify_regime_bias(r["description"])

    # Aplicar bias HMM a cada barra
    hmm_bullish = pd.Series(False, index=df.index)
    hmm_bearish = pd.Series(False, index=df.index)

    if "regime" in df.columns:
        for state, bias in state_bias.items():
            mask = df["regime"] == state
            if bias == "bullish":
                hmm_bullish[mask] = True
            elif bias == "bearish":
                hmm_bearish[mask] = True

    # Anadir peso W_HMM_REGIME a los scores
    if "signal_score_long" in df.columns:
        df["signal_score_long"] = df["signal_score_long"].astype(float) + hmm_bullish.astype(float) * W_HMM_REGIME
    if "signal_score_short" in df.columns:
        df["signal_score_short"] = df["signal_score_short"].astype(float) + hmm_bearish.astype(float) * W_HMM_REGIME

    # Recalcular las senales booleanas con los nuevos scores
    if "signal_score_long" in df.columns and "signal_score_short" in df.columns:
        df["signal_long"] = (df["signal_score_long"] >= SIGNAL_SCORE_THRESHOLD).astype(int)
        df["signal_short"] = (df["signal_score_short"] >= SIGNAL_SCORE_THRESHOLD).astype(int)

    return df


# -- MEJORA 2A: Detectar alertas tempranas de cambio de tendencia --
def detect_early_alerts(df: pd.DataFrame, states: np.ndarray,
                         state_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta alertas tempranas de cambio de tendencia usando el regimen HMM
    + un threshold reducido en el score compuesto.
    """
    # Mapa de bias por estado
    state_bias_map = {}
    for _, r in state_summary.iterrows():
        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])

    # Detectar barras con cambio de regimen reciente
    regime_changed = np.zeros(len(df), dtype=bool)
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            regime_changed[i] = True
            # Marcar las siguientes EARLY_WINDOW velas
            for j in range(1, min(EARLY_WINDOW, len(states) - i)):
                regime_changed[i + j] = True

    df["alert_early_long"] = False
    df["alert_early_short"] = False

    for i in range(len(df)):
        if not regime_changed[i]:
            continue

        current_state = states[i]
        current_bias = state_bias_map.get(current_state, "neutral")

        # Verificar scores con threshold reducido
        score_long = df["signal_score_long"].iloc[i] if "signal_score_long" in df.columns else 0
        score_short = df["signal_score_short"].iloc[i] if "signal_score_short" in df.columns else 0

        # Alerta temprana LONG: regimen bullish + score > EARLY_THRESHOLD
        if current_bias == "bullish" and score_long >= EARLY_THRESHOLD:
            df.loc[df.index[i], "alert_early_long"] = True

        # Alerta temprana SHORT: regimen bearish + score > EARLY_THRESHOLD
        if current_bias == "bearish" and score_short >= EARLY_THRESHOLD:
            df.loc[df.index[i], "alert_early_short"] = True

    return df'''

if target_smooth_end in content:
    content = content.replace(target_smooth_end, target_smooth_new)
    changes += 1
    print(f"[OK] Change 3: Added compute_signal_scores_with_hmm() and detect_early_alerts()")
else:
    print(f"[FAIL] Change 3: _smooth_states function not found for extension")

# ============================================================
# CHANGE 4: Add smooth_states and HMM score + early alerts in main pipeline
# ============================================================
# Find the section where states = best_model_info["states"] and apply smoothing + HMM score + early alerts
# Looking for the block that processes HMM results in main()

# Target: After "state_summary = best_model_info["state_summary"]" and before signal computation
target_main = '''                "model": best_model_info["model"],
                "states": best_model_info["states"],
                "state_summary": best_model_info["state_summary"],
                "trans_mat": best_model_info["trans_mat"],
            }

            # 4) Senal'''

new_main = '''                "model": best_model_info["model"],
                "states": best_model_info["states"],
                "state_summary": best_model_info["state_summary"],
                "trans_mat": best_model_info["trans_mat"],
            }

            # -- MEJORA 1B: Suavizar estados HMM (eliminar cambios espurios < 3 velas) --
            states = best_model_info["states"]
            states_smoothed = _smooth_states(states, min_duration=3)
            # Actualizar el dataframe con estados suavizados
            df["regime"] = states_smoothed

            # -- MEJORA 1A: Recalcular scores con peso del regimen HMM --
            df = compute_signal_scores_with_hmm(df, best_model_info["state_summary"])

            # -- MEJORA 2A: Detectar alertas tempranas --
            df = detect_early_alerts(df, states_smoothed, best_model_info["state_summary"])

            # Actualizar el modelo con los estados suavizados
            best_model_info["states"] = states_smoothed

            # 4) Senal'''

if target_main in content:
    content = content.replace(target_main, new_main)
    changes += 1
    print(f"[OK] Change 4: Added smooth_states + HMM score + early alerts in main pipeline")
else:
    print(f"[FAIL] Change 4: Main pipeline target not found")
    # Try alternative: find 'state_summary' context
    for i, line in enumerate(lines):
        if '"state_summary": best_model_info["state_summary"],' in line:
            print(f"  Found alternative at line {i+1}: {line.strip()}")
            break

# ============================================================
# CHANGE 5: Add REGIME_THRESHOLD_REDUCTION in compute_signal()
# ============================================================
# Find the dynamic threshold calculation in compute_signal
target_dynamic = '''    # Umbral dinamico basado en volatilidad ATR (Opción B mejorada)
    dynamic_threshold = max(
        DYNAMIC_THRESHOLD_MIN,
        SIGNAL_SCORE_THRESHOLD - min(25, int((df['atr'].iloc[-1] / max(df['atr'].rolling(20, min_periods=1).mean().iloc[-1], 0.01)) * 5))
    )'''

new_dynamic = '''    # Umbral dinamico basado en volatilidad ATR (Opcion B mejorada)
    dynamic_threshold = max(
        DYNAMIC_THRESHOLD_MIN,
        SIGNAL_SCORE_THRESHOLD - min(25, int((df['atr'].iloc[-1] / max(df['atr'].rolling(20, min_periods=1).mean().iloc[-1], 0.01)) * 5))
    )

    # -- MEJORA 2B: Reducir threshold adicional segun el tipo de regimen --
    if "regime" in df.columns and "regime_desc_col" in df.columns:
        current_regime_desc = df["regime_desc_col"].iloc[-1] if "regime_desc_col" in df.columns else ""
        for keyword, reduction in REGIME_THRESHOLD_REDUCTION.items():
            if keyword in str(current_regime_desc):
                dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold - reduction)
                break'''

if target_dynamic in content:
    content = content.replace(target_dynamic, new_dynamic)
    changes += 1
    print(f"[OK] Change 5: Added REGIME_THRESHOLD_REDUCTION in compute_signal()")
else:
    print(f"[FAIL] Change 5: Dynamic threshold target not found")

# ============================================================
# CHANGE 6: Add regime_desc_col to df in main pipeline for change 5
# ============================================================
# Find where state_summary row is accessed to build regime_desc
target_regime_desc = '''            regime_desc = ""
            regime_duration = "-"
            regime_pct = "-"
            if state_summary is not None and len(state_summary) > 0 and "current_state" in dir():
                row = state_summary[state_summary["state"] == current_state]
                if len(row) > 0:
                    regime_desc = row.iloc[0]["description"]'''

new_regime_desc = '''            regime_desc = ""
            regime_duration = "-"
            regime_pct = "-"
            if state_summary is not None and len(state_summary) > 0 and "current_state" in dir():
                row = state_summary[state_summary["state"] == current_state]
                if len(row) > 0:
                    regime_desc = row.iloc[0]["description"]
                    # -- MEJORA 2B: Guardar descripcion del regimen en df para threshold dinamico --
                    if "regime_desc_col" not in df.columns:
                        df["regime_desc_col"] = ""
                    df["regime_desc_col"] = df["regime_desc_col"].astype(object)
                    df["regime_desc_col"].iloc[-1] = regime_desc'''

if target_regime_desc in content:
    content = content.replace(target_regime_desc, new_regime_desc)
    changes += 1
    print(f"[OK] Change 6: Added regime_desc_col to df for dynamic threshold")
else:
    print(f"[FAIL] Change 6: Regime desc assignment not found")

# ============================================================
# Write back
# ============================================================
with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n{'='*50}")
print(f"Total changes applied: {changes}")
print(f"{'='*50}")
