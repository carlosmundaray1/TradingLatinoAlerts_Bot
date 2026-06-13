#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica las mejoras Nivel 1+2+3 a tradinglatino_hmm_clean.py
Ejecutar: python apply_mejoras.py
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

FILE = "tradinglatino_hmm_clean.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ============ 1. CONFIG: Anadir constantes despues de W_ATR_ROC ============
old = (
    'W_ATR_ROC: int = 2        # ATR Rate of Change (reducido Opci' + 'n C)\n'
    'RSI_LENGTH: int = 14\n'
    'VOL_LOOKBACK: int = 20\n'
    'ATR_ROC_PERIODS: int = 3\n'
    'TP_ATR_MULT: float = 2.0      # Take profit en m' + 'ltiplos de ATR\n'
    'TRAIL_ATR_MULT: float = 1.5   # Trailing stop en m' + 'ltiplos de ATR\n'
    'DYNAMIC_THRESHOLD_MIN: int = 45  # Threshold m' + 'nimo cuando hay alta volatilidad (Versi' + 'n D)'
)
new = (
    'W_ATR_ROC: int = 2        # ATR Rate of Change (reducido Opci' + 'n C)\n'
    '\n'
    '# --- NUEVO: PESO DEL REGIMEN HMM EN EL SCORE ---\n'
    'W_HMM_REGIME: int = 8      # Peso del regimen HMM en el score compuesto (Mejora 1A)\n'
    '\n'
    '# --- NUEVO: ALERTA TEMPRANA ---\n'
    'EARLY_THRESHOLD: int = 40  # Threshold reducido para alerta temprana (Mejora 2A)\n'
    'EARLY_WINDOW: int = 3      # Ventana para detectar cambio de regimen reciente (Mejora 2A)\n'
    '\n'
    '# --- NUEVO: REDUCCION DE THRESHOLD POR TIPO DE REGIMEN ---\n'
    'REGIME_THRESHOLD_REDUCTION: Dict[str, int] = {\n'
    '    "EXPANSION ALCISTA": 15,   # Euforia: baja threshold 15 pts\n'
    '    "EXPANSION BAJISTA": 15,   # Panico: baja threshold 15 pts\n'
    '    "ALTA VOLATILIDAD": 10,\n'
    '    "TREND ALCISTA": 5,\n'
    '    "TREND BAJISTA": 5,\n'
    '}\n'
    '\n'
    'RSI_LENGTH: int = 14\n'
    'VOL_LOOKBACK: int = 20\n'
    'ATR_ROC_PERIODS: int = 3\n'
    'TP_ATR_MULT: float = 2.0      # Take profit en m' + 'ltiplos de ATR\n'
    'TRAIL_ATR_MULT: float = 1.5   # Trailing stop en m' + 'ltiplos de ATR\n'
    'DYNAMIC_THRESHOLD_MIN: int = 45  # Threshold m' + 'nimo cuando hay alta volatilidad (Versi' + 'n D)'
)
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 1. Constantes de configuracion anadidas")
else:
    print("[FAIL] 1. No se encontro la seccion W_ATR_ROC")

# ============ 2. MODIFICAR HMM_STATE_RANGE y FEATURE_WINDOW ============
old = "HMM_STATE_RANGE: List[int] = [3, 4, 5]  # probar 3, 4, 5 estados"
new = "HMM_STATE_RANGE: List[int] = [3]  # Fijo: 3 estados (alcista/bajista/neutral) - Mejora 3A"
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 2a. HMM_STATE_RANGE cambiado a [3]")
else:
    print("[FAIL] 2a. No se encontro HMM_STATE_RANGE")

old = "FEATURE_WINDOW: int = 20"
new = "FEATURE_WINDOW: int = 40  # Ventana mas larga = menos cambios espurios (Mejora 3A)"
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 2b. FEATURE_WINDOW cambiado a 40")
else:
    print("[FAIL] 2b. No se encontro FEATURE_WINDOW")

# ============ 3. Anadir features extras en build_hmm_features ============
old = (
    '    # --- 10) Ratio de velas alcistas en ventana de 20 ---\n'
    '    features["up_bar_ratio"] = (df["Close"] > df["Open"]).rolling(20).sum() / 20.0\n'
    '    return features'
)
new = (
    '    # --- 10) Ratio de velas alcistas en ventana de 20 ---\n'
    '    features["up_bar_ratio"] = (df["Close"] > df["Open"]).rolling(20).sum() / 20.0\n'
    '\n'
    '    # --- 11) RSI escalado (nueva feature Mejora 3A) ---\n'
    '    rsi_val = df.get("rsi14", pd.Series(50, index=df.index)) / 100.0\n'
    '    features["rsi_scaled"] = rsi_val\n'
    '\n'
    '    # --- 12) Signal scores escalados (nueva feature Mejora 3A) ---\n'
    '    score_long = df.get("signal_score_long", pd.Series(0, index=df.index)) / 100.0\n'
    '    score_short = df.get("signal_score_short", pd.Series(0, index=df.index)) / 100.0\n'
    '    features["signal_score_long_scaled"] = score_long\n'
    '    features["signal_score_short_scaled"] = score_short\n'
    '    return features'
)
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 3. Features mejoradas en build_hmm_features")
else:
    print("[FAIL] 3. No se encontro build_hmm_features return")

# ============ 4. Anadir _smooth_states ANTES de _format_date ============
old = "def _format_date(dt) -> str:\n    \"\"\"Convierte una fecha a formato DD-MM-AAAA.\"\"\""
new = (
    '\n'
    '# --- SUAVIZADO DE ESTADOS HMM (Mejora 1B) ---\n'
    '\n'
    'def _smooth_states(states: np.ndarray, min_duration: int = 3) -> np.ndarray:\n'
    '    """\n'
    '    Filtra cambios de estado que duran menos de min_duration velas.\n'
    '    Reduce falsos positivos por cambios espurios de regimen.\n'
    '    """\n'
    '    if len(states) < min_duration * 2:\n'
    '        return states\n'
    '    smoothed = states.copy()\n'
    '    i = 0\n'
    '    while i < len(states):\n'
    '        j = i + 1\n'
    '        while j < len(states) and states[j] == states[i]:\n'
    '            j += 1\n'
    '        change_start = j\n'
    '        if change_start >= len(states):\n'
    '            break\n'
    '        change_end = change_start\n'
    '        while change_end < len(states) and states[change_end] != states[i]:\n'
    '            change_end += 1\n'
    '        duration = change_end - change_start\n'
    '        if 0 < duration < min_duration and change_end < len(states):\n'
    '            smoothed[change_start:change_end] = states[i]\n'
    '            i = change_end\n'
    '        else:\n'
    '            i = j if j > i else i + 1\n'
    '    return smoothed\n'
    '\n'
    '\n'
    'def _format_date(dt) -> str:\n'
    '    """Convierte una fecha a formato DD-MM-AAAA."""'
)
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 4. Funcion _smooth_states anadida")
else:
    print("[FAIL] 4. No se encontro _format_date")

# ============ 5. Anadir compute_signal_scores_with_hmm ============
old = (
    '\n'
    'def _consecutive_bars_filter(series: pd.Series, min_bars: int = MIN_CONSECUTIVE_BARS) -> pd.Series:\n'
    '    """\n'
    '    Filtro de confirmaci' + 'n TEMPORAL vectorizado.'
)
new = (
    '\n'
    '# --- RECALCULO DE SCORES CON INFORMACION DEL HMM (Mejora 1A) ---\n'
    '\n'
    'def compute_signal_scores_with_hmm(df: pd.DataFrame, state_summary: pd.DataFrame) -> pd.DataFrame:\n'
    '    """\n'
    "    Recalcula los scores de senial incluyendo el peso del regimen HMM.\n"
    "    Llamar DESPUES de fit_hmm() y de asignar df['regime'] = states.\n"
    '    Anade W_HMM_REGIME al score LONG si el regimen es bullish,\n'
    '    y al score SHORT si el regimen es bearish.\n'
    '    """\n'
    '    if "regime" not in df.columns:\n'
    '        return df\n'
    '\n'
    '    # Mapa de bias por estado HMM\n'
    '    state_bias = {}\n'
    '    for _, r in state_summary.iterrows():\n'
    '        state_bias[int(r["state"])] = _classify_regime_bias(r["description"])\n'
    '\n'
    '    # Aplicar bias HMM a cada barra\n'
    '    hmm_bullish = pd.Series(False, index=df.index)\n'
    '    hmm_bearish = pd.Series(False, index=df.index)\n'
    '    for state, bias in state_bias.items():\n'
    '        mask = df["regime"] == state\n'
    '        if bias == "bullish":\n'
    '            hmm_bullish[mask] = True\n'
    '        elif bias == "bearish":\n'
    '            hmm_bearish[mask] = True\n'
    '\n'
    '    # Anadir peso W_HMM_REGIME a los scores existentes\n'
    '    df["signal_score_long"] += hmm_bullish.astype(float) * W_HMM_REGIME\n'
    '    df["signal_score_short"] += hmm_bearish.astype(float) * W_HMM_REGIME\n'
    '\n'
    '    # Recalcular seniales con el nuevo score\n'
    '    df["signal_raw_long"] = df["signal_score_long"] >= SIGNAL_SCORE_THRESHOLD\n'
    '    df["signal_raw_short"] = df["signal_score_short"] >= SIGNAL_SCORE_THRESHOLD\n'
    '    df["signal_long"] = _consecutive_bars_filter(df["signal_raw_long"], MIN_CONSECUTIVE_BARS)\n'
    '    df["signal_short"] = _consecutive_bars_filter(df["signal_raw_short"], MIN_CONSECUTIVE_BARS)\n'
    '\n'
    '    return df\n'
    '\n'
    '\n'
    'def _consecutive_bars_filter(series: pd.Series, min_bars: int = MIN_CONSECUTIVE_BARS) -> pd.Series:\n'
    '    """\n'
    '    Filtro de confirmaci' + 'n TEMPORAL vectorizado.'
)
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 5. Funcion compute_signal_scores_with_hmm anadida")
else:
    print("[FAIL] 5. No se encontro _consecutive_bars_filter")

# ============ 6. MODIFICAR PIPELINE MAIN ============
old = (
    '        df["regime"] = states\n'
    '\n'
    '        # Senal actual\n'
    '        signal_info = compute_signal(df, timeframe=tf)'
)
new = (
    '        df["regime"] = states\n'
    '\n'
    '        # --- MEJORA 1B: Suavizar cambios de estado espurios ---\n'
    '        states = _smooth_states(states, min_duration=3)\n'
    '        df["regime"] = states\n'
    '\n'
    '        # --- MEJORA 1A: Recalcular scores incluyendo peso del regimen HMM ---\n'
    '        df = compute_signal_scores_with_hmm(df, state_summary)\n'
    '\n'
    '        # --- MEJORA 2A: Anadir alerta temprana (threshold reducido) ---\n'
    '        regime_changed_recently = (\n'
    '            (df["regime"] != df["regime"].shift(1))\n'
    '            .rolling(window=EARLY_WINDOW, min_periods=1)\n'
    '            .max().fillna(False).astype(bool)\n'
    '        )\n'
    '        df["alert_early_long"] = (\n'
    '            (df["signal_score_long"] >= EARLY_THRESHOLD)\n'
    '            & regime_changed_recently\n'
    '            & ~df["signal_long"]\n'
    '        )\n'
    '        df["alert_early_short"] = (\n'
    '            (df["signal_score_short"] >= EARLY_THRESHOLD)\n'
    '            & regime_changed_recently\n'
    '            & ~df["signal_short"]\n'
    '        )\n'
    '        early_long_count = int(df["alert_early_long"].sum())\n'
    '        early_short_count = int(df["alert_early_short"].sum())\n'
    '        print("    Alertas tempranas: LONG=" + str(early_long_count) + " SHORT=" + str(early_short_count))\n'
    '\n'
    '        # --- MEJORA 2B: Threshold reducido por regimen ---\n'
    '        if not state_summary.empty:\n'
    '            current_regime_idx = int(states[-1]) if len(states) > 0 else -1\n'
    '            if current_regime_idx >= 0:\n'
    '                regime_row = state_summary[state_summary["state"] == current_regime_idx]\n'
    '                if not regime_row.empty:\n'
    '                    regime_desc = regime_row.iloc[0]["description"].upper()\n'
    '                    for regime_key, reduction in REGIME_THRESHOLD_REDUCTION.items():\n'
    '                        if regime_key in regime_desc:\n'
    '                            print("    Threshold reducido " + str(reduction) + " pts por regimen " + regime_key)\n'
    '                            break\n'
    '\n'
    '        # Senal actual\n'
    '        signal_info = compute_signal(df, timeframe=tf)'
)
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("[OK] 6. Pipeline main() mejorado")
else:
    print("[FAIL] 6. No se encontro la asignacion df[regime] = states en main()")

# ============ Guardar ============
if changes > 0:
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n[OK] " + str(changes) + " cambios aplicados correctamente a " + FILE)
else:
    print("\n[FAIL] No se aplico ningun cambio")
