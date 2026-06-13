#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRADINGLATINO · HMM REGIME DASHBOARD (SIMPLIFICADO)
================================================================================
Dashboard limpio y profesional para entender los regímenes de mercado HMM
con señales LONG/SHORT precisas de la estrategia TradingLatino.
Uso:
    python tradinglatino_hmm_clean.py
Dependencias:
    pip install pandas numpy plotly hmmlearn yfinance
================================================================================
"""
import contextlib
import os
import sys
import time
import warnings
import json
import webbrowser
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# DEPENDENCIAS

# ──────────────────────────────────────────────────────────────────────────────
_MISSING_DEPS: List[str] = []
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    _MISSING_DEPS.append("plotly")
try:
    from hmmlearn import hmm
except ImportError:
    _MISSING_DEPS.append("hmmlearn")
try:
    import yfinance as yf
except ImportError:
    _MISSING_DEPS.append("yfinance")
if _MISSING_DEPS:
    print("=" * 60)
    print("  ERROR: FALTAN DEPENDENCIAS")
    print("=" * 60)
    print(f"\n  pip install {' '.join(_MISSING_DEPS)}\n")
    sys.exit(1)
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN

# ──────────────────────────────────────────────────────────────────────────────
ASSET: str = "BTC-USD"
TIMEFRAMES: List[str] = ["1h", "4h", "1d", "1w"]
OUTPUT_HTML: str = "hmm_dashboard_{ASSET}.html"  # Se genera dinamicamente segun el activo
STATE_FILE: str = ".hmm_state_{ASSET}.json"  # Estado anterior entre ejecuciones (por activo)
OPEN_BROWSER: bool = True

# Parámetros fijos de la estrategia (basados en valores por defecto probados)
EMA_FAST: int = 10
EMA_SLOW: int = 55
ADX_THRESHOLD: float = 23.0
RELEASE_LOOKBACK: int = 3
ATR_STOP_MULT: float = 2.0
RR_TARGET: float = 2.0
USE_VOLUME_FILTER: bool = True

# Take-Profit automático para verificación histórica (por timeframe)
# En lugar de hold hasta expiración, la señal se considera ganadora si
# el precio alcanza el % objetivo en algún momento dentro de la ventana.
#
# 1H: TP=0.5%  (la mediana del movimiento máximo es 0.94% en 14h)
# 4H: TP=1.2%  (70.4% WR con 328 señales, threshold=68)
# 1D: TP=2.0%  (la mediana del movimiento máximo es 3.51% en 14d)
# 1W: TP=2.0%  (la mediana del movimiento máximo es ~5%+ en 12 sem)
TAKE_PROFIT_PCT: Dict[str, float] = {
    "1h": 0.5,
    "4h": 1.2,   # WR=70.4% con 328 senales (threshold=68, cons=2)
    "1d": 2.0,
    "1wk": 2.0,
}

# Trailing Stop para la Estrategia 3: salir cuando el precio retrocede X%
# desde su maximo favorable (trailing stop de retroceso).
# El trailing stop captura la ganancia antes de que el mercado revierta.
#
# Valores tipicos:
#   TRAIL_PCT = 30  -> Agresivo: sale con poco retroceso, captura casi todo el movimiento
#   TRAIL_PCT = 50  -> Balanceado: punto dulce entre WR y ganancia por trade
#   TRAIL_PCT = 70  -> Conservador: permite mas retroceso, busca trades mas grandes
#
# El trailing stop se COMBINA con el TP fijo (el que se active primero gana).
TRAILING_STOP_PCT: Dict[str, float] = {
    "1h": 2.0,    # Optimo multi-objetivo: WR 73.5% | PF 3.48 | Sharpe 13.48
    "4h": 2.0,    # Optimo multi-objetivo: WR 73.2% | PF 5.55 | Sharpe 10.65
    "1d": 1.0,    # Optimizado
    "1wk": 1.0,   # Optimizado
}

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETROS OPTIMIZADOS POR TIMEFRAME (Grid Search Multi-Objetivo)
# ──────────────────────────────────────────────────────────────────────────────
# Resultados en BTC-USD (Junio 2026):
#   TF    Thresh  Cons  Trail%  TP%   BaseWR  CombWR  Senales
#   ---   ------  ----  ------  ---   ------  ------  -------
#   1H      70      2    2.0    0.5   76.1%   78.3%    456
#   4H      75      3    0.5    1.2   76.3%   77.5%    173
#   1D      65      2    1.0    2.0   78.5%   80.4%    316
#   1WK     65      2    1.0    2.0   97.2%   97.2%     72
TRAILING_STOP_PCT_OPT: Dict[str, float] = {
    "1h": 2.0,    # Retroceso 2.0% desde maximo
    "4h": 0.5,    # Retroceso 0.5% desde maximo (tight)
    "1d": 1.0,
    "1wk": 1.0,
}
SIGNAL_THRESHOLDS: Dict[str, int] = {
    "1h": 70,
    "4h": 75,
    "1d": 65,
    "1wk": 65,
}
MIN_CONSECUTIVE_BY_TF: Dict[str, int] = {
    "1h": 2,
    "4h": 3,
    "1d": 2,
    "1wk": 2,
}

# ──────────────────────────────────────────────────────────────────────────────
# FILTRO MÍNIMO DE SEÑALES (Score Compuesto Ponderado + Confirmación Temporal)

# ──────────────────────────────────────────────────────────────────────────────
# El score compuesto reemplaza el AND binario con un sistema de puntuación.
# Cada condición contribuye con un peso específico. Si el score supera el
# umbral, la señal se activa. Esto permite señales con alta probabilidad de
# éxito sin exigir el 100% de las condiciones.
#
# Además, el filtro de velas consecutivas evita falsos positivos al requerir
# que la señal se mantenga activa por N velas antes de confirmarse.
SIGNAL_SCORE_THRESHOLD: int = 65   # Score mínimo (0-100) para activar la señal (Versión D: calibrado final)
MIN_CONSECUTIVE_BARS: int = 2      # Velas consecutivas para confirmar la señal

# Pesos del score compuesto (total base = 100, bonus = hasta 5)
W_BULL_BIAS: int = 16      # Tendencia direccional
W_SQUEEZE_OFF: int = 15    # Expansión del squeeze
W_SQUEEZE_REL: int = 8     # Compresión previa verificada (reducido: no relevante en crashes)
W_SMI_HIST: int = 10       # Momentum alineado con la dirección (reducido: nueva fórmula LazyBear más suave)
W_SMI_DELTA: int = 12      # Aceleración del momentum
W_ADX_THRESH: int = 5      # Fuerza de tendencia (reducido: ADX es rezagado para entradas)
W_ADX_DELTA: int = 2       # Tendencia fortaleciéndose (reducido: ADX es rezagado)
BONUS_ADX: int = 0         # Eliminado: bonus ADX no ayuda en entradas rápidas
# ── NUEVOS INDICADORES (Mejoras para capturar movimientos violentos) ──
W_EMA_DEV: int = 6        # Desviación del precio vs EMA55 (reducido Opción C)
W_RSI: int = 4            # RSI14 (reducido Opción C)
W_VOLUME: int = 5         # Volumen relativo a su media (reducido Opción C)
W_ATR_ROC: int = 2        # ATR Rate of Change (reducido Opción C)
RSI_LENGTH: int = 14
W_RSI_DIVERGENCE: int = 12  # Divergencias RSI-Precio (alcistas/bajistas, peso base)

# -- MEJORA 1A: Peso del regimen HMM en el score compuesto --
W_HMM_REGIME: int = 8

# -- INTEGRACION MARKOV SWITCHING: Peso del regimen MS en el score compuesto --
W_MS_REGIME: int = 10
MS_SIGNAL_THRESHOLD: int = 80      # Suma puntos cuando el regimen HMM esta alineado con la senal

# -- MEJORA 2A: Alerta temprana (threshold reducido) --
EARLY_THRESHOLD: int = 40  # Threshold reducido para alerta temprana de cambio de tendencia
EARLY_WINDOW: int = 3      # Ventana de velas para detectar cambio de regimen reciente

# -- ENFOQUE A: Threshold para alertas precursoras (score antes del cruce) --
PRECURSOR_THRESHOLD: int = 45  # Score minimo para activar alerta precursora
PRECURSOR_VELOCITY_BARS: int = 5  # Ventana para calcular velocidad del score
PRECURSOR_MIN_COMPONENTS: int = 4  # Componentes minimos activos para alerta

# -- MEJORA 2B: Reduccion de threshold por tipo de regimen --
REGIME_THRESHOLD_REDUCTION = {
    "EXPANSION ALCISTA": 15,   # Euforia: baja threshold 15 pts
    "EXPANSION BAJISTA": 15,   # Panico: baja threshold 15 pts
    "ALTA VOLATILIDAD": 10,
    "TREND ALCISTA": 5,
    "TREND BAJISTA": 5,
}
VOL_LOOKBACK: int = 20
ATR_ROC_PERIODS: int = 3
TP_ATR_MULT: float = 2.0      # Take profit en múltiplos de ATR
TRAIL_ATR_MULT: float = 1.5   # Trailing stop en múltiplos de ATR
DYNAMIC_THRESHOLD_MIN: int = 45  # Threshold mínimo cuando hay alta volatilidad (Versión D)

# -- TELEGRAM ALERTS --
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "8942505010:AAGM9ziP38U3flYxPzR11gnVGwkuDz9o4KQ")       # Token (desde env var)
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "536876820")        # Chat ID (desde env var)
ENABLE_TELEGRAM: bool = os.environ.get("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")     # Activar (desde env var)



# HMM
HMM_STATE_RANGE: List[int] = [3, 4, 5]  # Elegir optimo por BIC - Mejora 3A
HMM_COVARIANCE_TYPE: str = "diag"
RANDOM_STATE: int = 42
FEATURE_WINDOW: int = 20

# Indicadores
BB_LENGTH: int = 20
BB_STD: float = 2.0
KC_LENGTH: int = 20
KC_MULT: float = 2.0
ADX_LENGTH: int = 14
ATR_LENGTH: int = 14
VP_LOOKBACK: int = 75
VP_BINS: int = 24

# Periodos de descarga (personalizables por CLI: --period-1h, --period-4h, --period-1d, --period-1w)
PERIOD_1H: str = "90d"
PERIOD_4H: str = "180d"
PERIOD_1D: str = "2y"
PERIOD_1W: str = "4y"

# ──────────────────────────────────────────────────────────────────────────────
# REGLA DE EXPIRACIÓN DE JAIME MERINO

# ──────────────────────────────────────────────────────────────────────────────
# La señal debe materializarse en las siguientes 10-14 velas aproximadamente.
# Si no ocurre, la premisa se invalida y el trade se cierra por expiración.
MAX_BARS_BY_TF: Dict[str, int] = {
    "1h": 14,   # ~14 horas
    "4h": 14,   # ~56 horas (optimizado por experiencia del usuario)
    "1d": 14,   # ~14 días
    "1wk": 12,  # ~3 meses
}

# Assets populares para el selector en el Dashboard
POPULAR_ASSETS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "MATIC-USD", "ATOM-USD",
    "UNI-USD", "AAVE-USD", "APT-USD", "SUI-USD",
]

# Nombre del script (se detecta automaticamente)
SCRIPT_NAME: str = "tradinglatino_hmm_clean.py"

# ──────────────────────────────────────────────────────────────────────────────
# INDICADORES TÉCNICOS

# ──────────────────────────────────────────────────────────────────────────────


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def _std(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).std(ddof=0)


def _tr(high: pd.Series, low: pd.Series, close: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = _tr(high, low, close, prev_close)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def _bollinger_bands(close: pd.Series, length: int = 20, std_dev: float = 2.0):
    ma = _sma(close, length)
    sd = _std(close, length)
    return ma + std_dev * sd, ma, ma - std_dev * sd


def _keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 1.5):
    ma = _ema(close, length)
    atr = _atr_wilder(high, low, close, length)
    return ma + mult * atr, ma, ma - mult * atr



def _linreg(series: pd.Series, length: int) -> pd.Series:
    """Linear regression forecast (Pine Script linreg equivalent).
    Returns the value of the linear regression line at the current bar (offset 0).
    Equivalente a: linreg(source, length, 0) en Pine Script.
    """
    result = pd.Series(index=series.index, dtype=float, name="linreg")
    x = np.arange(length, dtype=float)
    for i in range(length - 1, len(series)):
        y = series.iloc[i - length + 1:i + 1].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        # Valor en x = length - 1 (barra actual)
        result.iloc[i] = slope * (length - 1) + intercept
    return result


def _adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    tr = _tr(high, low, close, prev_close)
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    plus_di = (plus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / atr.replace(0, np.nan)) * 100.0
    minus_di = (minus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / atr.replace(0, np.nan)) * 100.0
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1.0 / length, adjust=False).mean()
    return adx, plus_di, minus_di


def _squeeze_momentum(high, low, close, bb_length=20, bb_std=2.0, kc_length=20, kc_mult=1.5):
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, bb_length, bb_std)
    kc_upper, kc_mid, kc_lower = _keltner_channels(high, low, close, kc_length, kc_mult)
    squeeze_on = (bb_lower >= kc_lower) & (bb_upper <= kc_upper)
    squeeze_off = ~squeeze_on
    # LazyBear EXACTO: linreg(close - avg(avg(highest(high,KC), lowest(low,KC)), sma(close,KC)), KC, 0)
    highest_kc = high.rolling(kc_length).max()
    lowest_kc = low.rolling(kc_length).min()
    avg_hl = (highest_kc + lowest_kc) / 2  # avg(highest, lowest)
    sma_close = close.rolling(kc_length).mean()
    center = (avg_hl + sma_close) / 2       # avg(avg_hl, sma_close)
    diff = close - center                    # source - center
    smi_hist = _linreg(diff, kc_length)      # linreg(diff, KC, 0)
    smi_hist = smi_hist.fillna(0)
    return squeeze_on, squeeze_off, smi_hist, bb_upper, bb_mid, bb_lower, kc_upper, kc_mid, kc_lower

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calcula RSI (Relative Strength Index) usando Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _detect_rsi_divergence(df, lookback=30, swing_bars=3):
    """Detecta divergencias RSI-Precio."""
    df = df.copy()
    df["rsi_div_bullish"] = 0.0
    df["rsi_div_bearish"] = 0.0
    if "rsi14" not in df.columns or len(df) < lookback:
        return df
    low = df["Low"].values
    high = df["High"].values
    rsi = df["rsi14"].values
    n = len(df)
    swing_low_idx = []
    for i in range(swing_bars, n - swing_bars):
        if low[i] == min(low[i-swing_bars:i+swing_bars+1]):
            swing_low_idx.append(i)
    swing_high_idx = []
    for i in range(swing_bars, n - swing_bars):
        if high[i] == max(high[i-swing_bars:i+swing_bars+1]):
            swing_high_idx.append(i)
    for i in range(1, len(swing_low_idx)):
        p, q = swing_low_idx[i-1], swing_low_idx[i]
        if q - p > lookback:
            continue
        if low[q] < low[p] and rsi[q] > rsi[p]:
            strength = W_RSI_DIVERGENCE
            if rsi[p] < 30:
                strength += 4
            df.loc[df.index[q]:, "rsi_div_bullish"] = strength
    for i in range(1, len(swing_high_idx)):
        p, q = swing_high_idx[i-1], swing_high_idx[i]
        if q - p > lookback:
            continue
        if high[q] > high[p] and rsi[q] < rsi[p]:
            strength = W_RSI_DIVERGENCE
            if rsi[p] > 70:
                strength += 4
            df.loc[df.index[q]:, "rsi_div_bearish"] = strength
    return df


def _compute_signal_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa un score compuesto ponderado (0-110+) para señales LONG y SHORT.
    Incluye indicadores clásicos + nuevos: RSI14, desviación precio-EMA55,
    volumen relativo y ATR Rate of Change para capturar movimientos violentos.
    """
    # ADX strength bonus (eliminado: BONUS_ADX = 0)
    adx_strength = ((df["adx"] - ADX_THRESHOLD) / 20.0).clip(0, 1) * BONUS_ADX

    # Direccionalidad de ADX
    bull_directional = (df["plus_di"] > df["minus_di"]).astype(float)
    bear_directional = (df["minus_di"] > df["plus_di"]).astype(float)

    # ── NUEVOS INDICADORES ──

    # 1) Desviación del precio vs EMA55 (distancia porcentual)
    # SHORT: precio más de 3% bajo EMA55 = señal de caída violenta
    # LONG: precio más de 3% sobre EMA55 = señal de fuerza alcista
    ema_dev_pct = df.get("ema_deviation_pct", pd.Series(0, index=df.index))
    ema_dev_short = (ema_dev_pct < -3).astype(float) * W_EMA_DEV
    ema_dev_long  = (ema_dev_pct > 3).astype(float) * W_EMA_DEV

    # Bonus adicional: desviación extrema (>8% = capitulación/euforia)
    ema_dev_extreme_short = ((ema_dev_pct < -8).astype(float) * 5).clip(upper=5)
    ema_dev_extreme_long  = ((ema_dev_pct > 8).astype(float) * 5).clip(upper=5)

    # 2) RSI14
    rsi_val = df.get("rsi14", pd.Series(50, index=df.index))
    # SHORT: RSI < 45 (bearish momentum)
    rsi_short = (rsi_val < 45).astype(float) * W_RSI
    # LONG: RSI > 55 (bullish momentum)
    rsi_long  = (rsi_val > 55).astype(float) * W_RSI
    # Bonus RSI extremo: <25 capitulación (para LONG reversal), >80 euforia (para SHORT reversal)
    rsi_extreme_long  = ((rsi_val < 25).astype(float) * 4).clip(upper=4)
    rsi_extreme_short = ((rsi_val > 80).astype(float) * 4).clip(upper=4)

    # 3) Volumen relativo a su media
    vol_ratio = df.get("vol_ratio", pd.Series(1, index=df.index))
    vol_conf_short = ((vol_ratio > 1.5) & (ema_dev_pct < 0)).astype(float) * W_VOLUME
    vol_conf_long  = ((vol_ratio > 1.5) & (ema_dev_pct > 0)).astype(float) * W_VOLUME
    # Bonus volumen extremo (>3x media = capitulación/explosión)
    vol_extreme_short = ((vol_ratio > 3).astype(float) * 4).clip(upper=4)
    vol_extreme_long  = ((vol_ratio > 3).astype(float) * 4).clip(upper=4)

    # 4) ATR Rate of Change (expansión de volatilidad)
    atr_roc = df.get("atr_roc", pd.Series(0, index=df.index))
    # SHORT: volatilidad expandiéndose + sesgo bajista
    atr_roc_short = ((atr_roc > 0.2) & (ema_dev_pct < 0)).astype(float) * W_ATR_ROC
    # LONG: volatilidad expandiéndose + sesgo alcista
    atr_roc_long  = ((atr_roc > 0.2) & (ema_dev_pct > 0)).astype(float) * W_ATR_ROC

    # ── LONG SCORE ──
    df["signal_score_long"] = (
        df["bull_bias"].astype(float) * W_BULL_BIAS
        + df["squeeze_off"].astype(float) * W_SQUEEZE_OFF
        + df["squeeze_released"].astype(float) * W_SQUEEZE_REL
        + (df["smi_hist"] > 0).astype(float) * W_SMI_HIST
        + (df["smi_delta"] > 0).astype(float) * W_SMI_DELTA
        + (df["adx"] > ADX_THRESHOLD).astype(float) * W_ADX_THRESH
        + ((df["adx_delta"] > 0) & (df["plus_di"] > df["minus_di"])).astype(float) * W_ADX_DELTA
        + adx_strength * bull_directional
        + ema_dev_long + ema_dev_extreme_long
        + rsi_long + rsi_extreme_long
        + df["rsi_div_bullish"].values
        + vol_conf_long + vol_extreme_long
        + atr_roc_long
    ).round(1)

    # ── SHORT SCORE ──
    df["signal_score_short"] = (
        df["bear_bias"].astype(float) * W_BULL_BIAS
        + df["squeeze_off"].astype(float) * W_SQUEEZE_OFF
        + df["squeeze_released"].astype(float) * W_SQUEEZE_REL
        + (df["smi_hist"] < 0).astype(float) * W_SMI_HIST
        + (df["smi_delta"] < 0).astype(float) * W_SMI_DELTA
        + (df["adx"] > ADX_THRESHOLD).astype(float) * W_ADX_THRESH
        + ((df["adx_delta"] > 0) & (df["minus_di"] > df["plus_di"])).astype(float) * W_ADX_DELTA
        + adx_strength * bear_directional
        + ema_dev_short + ema_dev_extreme_short
        + rsi_short + rsi_extreme_short
        + df["rsi_div_bearish"].values
        + vol_conf_short + vol_extreme_short
        + atr_roc_short
    ).round(1)
    return df


def _consecutive_bars_filter(series: pd.Series, min_bars: int = MIN_CONSECUTIVE_BARS) -> pd.Series:
    """
    Filtro de confirmación TEMPORAL vectorizado.
    Una señal solo se activa si ha estado presente por al menos `min_bars`
    velas consecutivas. Esto elimina falsos positivos aislados.
    Ejemplo con min_bars=2:
        raw:     [F, T, T, T, F, T, F]
        result:  [F, F, T, T, F, F, F]
                         ^^ señal confirmada en vela 3
    """
    if min_bars <= 1:
        return series

    # Rolling sum: solo True si todas las últimas min_bars velas son True
    rolling_sum = series.rolling(window=min_bars, min_periods=min_bars).sum()
    return (rolling_sum >= min_bars) & series


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos necesarios."""
    df = df.copy()

    # EMAs
    df["ema_fast"] = _ema(df["Close"], EMA_FAST)
    df["ema_slow"] = _ema(df["Close"], EMA_SLOW)

    # ATR
    df["atr"] = _atr_wilder(df["High"], df["Low"], df["Close"], ATR_LENGTH)

    # Squeeze Momentum
    squeeze_on, squeeze_off, smi_hist, bb_upper, bb_mid, bb_lower, kc_upper, kc_mid, kc_lower = _squeeze_momentum(
        df["High"], df["Low"], df["Close"],
        bb_length=BB_LENGTH, bb_std=BB_STD,
        kc_length=KC_LENGTH, kc_mult=KC_MULT,
    )
    df["squeeze_on"] = squeeze_on.astype(bool)
    df["squeeze_off"] = squeeze_off.astype(bool)
    df["smi_hist"] = smi_hist
    df["smi_delta"] = df["smi_hist"].diff()

    # Store Bollinger Bands and Keltner Channels for visualization
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["kc_upper"] = kc_upper
    df["kc_mid"] = kc_mid
    df["kc_lower"] = kc_lower

    # ADX
    adx, plus_di, minus_di = _adx_wilder(df["High"], df["Low"], df["Close"], ADX_LENGTH)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx_delta"] = df["adx"].diff()

    # ── NUEVOS INDICADORES ──

    # RSI14
    df["rsi14"] = _rsi(df["Close"], RSI_LENGTH)
    # Divergencias RSI-Precio (Mejora #2)
    df = _detect_rsi_divergence(df)

    # Desviación del precio vs EMA55 (%)
    df["ema_deviation_pct"] = (df["Close"] - df["ema_slow"]) / df["ema_slow"] * 100

    # Volumen relativo a su media
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(VOL_LOOKBACK, min_periods=1).mean()

    # ATR Rate of Change (expansión de volatilidad en 3 velas)
    df["atr_roc"] = df["atr"].pct_change(periods=ATR_ROC_PERIODS)

    # Bias direccional (EMA trend)
    df["bull_bias"] = (df["Close"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"])
    df["bear_bias"] = (df["Close"] < df["ema_slow"]) & (df["ema_fast"] < df["ema_slow"])

    # Squeeze release
    df["squeeze_released"] = (
        df["squeeze_on"]
        .rolling(window=RELEASE_LOOKBACK, min_periods=1)
        .max()
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # ── FILTRO MÍNIMO DE SEÑALES ──
    # 1) Score compuesto ponderado (reemplaza el AND binario)
    df = _compute_signal_scores(df)

    # 2) Señales base por umbral de score
    df["signal_raw_long"] = df["signal_score_long"] >= SIGNAL_SCORE_THRESHOLD
    df["signal_raw_short"] = df["signal_score_short"] >= SIGNAL_SCORE_THRESHOLD

    # 3) Filtro de confirmación temporal (velas consecutivas)
    df["signal_long"] = _consecutive_bars_filter(df["signal_raw_long"], MIN_CONSECUTIVE_BARS)
    df["signal_short"] = _consecutive_bars_filter(df["signal_raw_short"], MIN_CONSECUTIVE_BARS)

    # Mantener también la señal binaria clásica (AND de todas las condiciones)
    # para comparación y referencia
    df["signal_long_classic"] = (
        df["bull_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] > 0)
        & (df["smi_delta"] > 0)
        & (df["adx"] > ADX_THRESHOLD)
        & (df["adx_delta"] > 0)
    )
    df["signal_short_classic"] = (
        df["bear_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] < 0)
        & (df["smi_delta"] < 0)
        & (df["adx"] > ADX_THRESHOLD)
        & (df["adx_delta"] > 0)
    )
    return df

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS

# ──────────────────────────────────────────────────────────────────────────────


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == 'open':    rename_map[col] = 'Open'
        elif cl == 'high':   rename_map[col] = 'High'
        elif cl == 'low':    rename_map[col] = 'Low'
        elif cl == 'close':  rename_map[col] = 'Close'
        elif cl == 'volume': rename_map[col] = 'Volume'
        elif cl in ('adj close', 'adj_close'): rename_map[col] = 'Adj Close'
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            similar = [c for c in df.columns if c.lower().strip() == col.lower()]
            if similar:
                df = df.rename(columns={similar[0]: col})
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resamplea datos OHLCV a una temporalidad superior (ej: 1h -> 4h)."""
    if df is None or df.empty:
        return df
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    return resampled.dropna()


def load_data(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Descarga datos OHLCV de Yahoo Finance con retry + User-Agent."""
    import requests as _requests

    # Mapa de periodos e intervalos nativos de yfinance
    # 4h no es intervalo nativo -> descargar 1h y resamplear
    native_intervals = {"1h": "60m", "1d": "1d", "1w": "1wk", "1wk": "1wk"}
    native_periods = {"1h": PERIOD_1H, "1d": PERIOD_1D, "1w": PERIOD_1W, "1wk": PERIOD_1W}
    if timeframe in native_intervals:
        interval = native_intervals[timeframe]
        period = native_periods[timeframe]
        print(f"  Descargando {asset} ({timeframe}, {period})...")
    elif timeframe == "4h":
        print(f"  Descargando {asset} (1h -> resample 4h)...")
        interval = "60m"
        period = PERIOD_4H
    else:
        print(f"  ERROR: Timeframe {timeframe} no soportado.")
        return None


    def _do_download() -> Optional[pd.DataFrame]:

        # Intentar 1: yf.download sin session (yfinance 1.2+ usa curl_cffi internamente)
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                df = yf.download(
                    asset, period=period, interval=interval,
                    progress=False
                )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # Intentar 2: Ticker.history
        try:
            ticker = yf.Ticker(asset)
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                df = ticker.history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # Intentar 3: requests directo a la API de Yahoo (fallback sin yfinance)
        try:
            session = _requests.Session()
            session.headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}"
                f"?range=2y&interval={interval}&includePrePost=False"
            )
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    timestamps = result[0].get("timestamp", [])
                    quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
                    if timestamps and quotes:
                        df_direct = pd.DataFrame({
                            "Open": quotes.get("open", []),
                            "High": quotes.get("high", []),
                            "Low": quotes.get("low", []),
                            "Close": quotes.get("close", []),
                            "Volume": quotes.get("volume", []),
                        }, index=pd.to_datetime(timestamps, unit="s"))
                        df_direct = df_direct.dropna()
                        df_direct["Adj Close"] = df_direct["Close"]
                        if not df_direct.empty:
                            print(f"      (descargado via API directa)")
                            return df_direct
        except Exception:
            pass
        return None

    # Retry con backoff
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        df = _do_download()
        if df is not None and not df.empty:
            df = _normalize_columns(df)
            if timeframe == "4h":
                df = _resample_ohlc(df, "4h")
            # Eliminar ultima vela (incompleta) para evitar datos inconsistentes
            df = df.iloc[:-1]
            print(f"    {len(df)} velas descargadas.")
            return df
        if attempt < max_attempts:
            wait = attempt * 4
            print(f"    Intento {attempt} fallo, reintentando en {wait}s...")
            time.sleep(wait)
    print(f"  ERROR: No se pudieron descargar datos para {asset} ({timeframe}).")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# HMM: CONSTRUCCIÓN DE FEATURES

# ──────────────────────────────────────────────────────────────────────────────


def _classify_regime_bias(description: str) -> str:
    """Clasifica un regimen como 'bullish', 'bearish', o 'neutral'.
    Usado para la advertencia de cambio de regimen."""
    d = description.upper()
    if any(x in d for x in ["ALCISTA", "EXPANSION ALCISTA", "TREND ALCISTA"]):
        return "bullish"
    if any(x in d for x in ["BAJISTA", "EXPANSION BAJISTA", "TREND BAJISTA"]):
        return "bearish"
    return "neutral"


def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features de mercado mejoradas para el HMM.
    Incluye:
      - Retornos multi-timeframe (1, 5 velas)
      - Volatilidad, ATR, momentum, ADX, squeeze
      - Volumen relativo a su media
      - Posicion relativa en el rango High-Low
      - Ratio de velas alcistas en ventana
    """
    features = pd.DataFrame(index=df.index)

    # ── 1) Retornos multi-timeframe ──
    features["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    features["log_return_5"] = np.log(df["Close"] / df["Close"].shift(5))

    # ── 2) Volatilidad ──
    features["vol_20"] = features["log_return_1"].rolling(window=FEATURE_WINDOW).std()

    # ── 3) Momentum / tendencia ──
    features["cumret_20"] = features["log_return_1"].rolling(window=FEATURE_WINDOW).sum()
    features["momentum_20"] = df["Close"].pct_change(FEATURE_WINDOW)

    # ── 4) ATR normalizado ──
    atr_col = df["atr"] if "atr" in df.columns else pd.Series(0.0, index=df.index)
    features["atr_norm"] = atr_col / df["Close"].replace(0, np.nan)

    # ── 5) Spread EMAs ──
    ema_fast = df.get("ema_fast", _ema(df["Close"], 10))
    ema_slow = df.get("ema_slow", _ema(df["Close"], 55))
    atr_safe = atr_col.replace(0, np.nan)
    features["ema_spread_atr"] = (ema_fast - ema_slow) / atr_safe

    # ── 6) Squeeze momentum ──
    smi_hist = df.get("smi_hist", pd.Series(0.0, index=df.index))
    features["smi_hist_norm"] = smi_hist / atr_safe.replace(0, np.nan)
    features["smi_hist_norm"] = features["smi_hist_norm"].fillna(0)

    # ── 7) ADX ──
    features["adx_scaled"] = df.get("adx", pd.Series(0.0, index=df.index)) / 100.0
    features["adx_delta_scaled"] = df.get("adx_delta", pd.Series(0.0, index=df.index)) / 100.0
    features["squeeze_flag"] = df.get("squeeze_on", pd.Series(False, index=df.index)).astype(int)

    # ── 8) Volumen relativo a su media movil de 20 ──
    vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
    features["vol_rel_20"] = df["Volume"] / vol_ma

    # ── 9) Posicion relativa en el rango High-Low de 20 velas ──
    high_20 = df["High"].rolling(20).max()
    low_20 = df["Low"].rolling(20).min()
    features["pos_in_range"] = (df["Close"] - low_20) / (high_20 - low_20 + 1e-10)

    # ── 10) Ratio de velas alcistas en ventana de 20 ──
    features["up_bar_ratio"] = (df["Close"] > df["Open"]).rolling(20).sum() / 20.0
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
    return features

# ──────────────────────────────────────────────────────────────────────────────
# HMM: FIT + RELABEL

# ──────────────────────────────────────────────────────────────────────────────


def fit_hmm(features_df: pd.DataFrame) -> Tuple[Any, np.ndarray, pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
    """Ajusta HMM probando varios estados, elige el mejor por BIC."""
    clean = features_df.dropna()
    if len(clean) < 100:
        print("  ERROR: Datos insuficientes para HMM.")
        return None, np.array([]), pd.DataFrame(), pd.DataFrame(), np.array([])
    X = clean.values
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds = np.where(stds < 1e-10, 1.0, stds)
    X_scaled = (X - means) / stds
    best_bic = np.inf
    best_model = None
    best_states = None
    bic_results = []

    # Multiples semillas para evitar optimos locales
    HMM_SEEDS: List[int] = [42, 7, 123]
    for n_states in HMM_STATE_RANGE:
        best_model_n = None
        best_ll_n = -np.inf
        best_states_n = None
        for seed in HMM_SEEDS:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    random_state=seed,
                    n_iter=300,
                    tol=1e-4,
                )
                model.fit(X_scaled)
                log_likelihood = model.score(X_scaled)
                if log_likelihood > best_ll_n:
                    best_ll_n = log_likelihood
                    best_model_n = model
                    best_states_n = model.predict(X_scaled).copy()
            except Exception as e:
                print(f"    HMM {n_states} estados (seed={seed}) falló: {e}")
                continue
        if best_model_n is None:
            print(f"    HMM {n_states} estados falló con todas las semillas.")
            continue

        # BIC con el mejor modelo de este n_states
        k = n_states * X_scaled.shape[1] * 2 + n_states * (n_states - 1) + (n_states - 1)
        bic = -2 * best_ll_n + k * np.log(len(X_scaled))
        bic_results.append({"n_states": n_states, "bic": bic})
        print(f"    {n_states} estados: BIC={bic:.1f}, LL={best_ll_n:.2f} (mejor de {len(HMM_SEEDS)} semillas)")
        if bic < best_bic:
            best_bic = bic
            best_model = best_model_n
            best_states = best_states_n.copy()
    if best_model is None:
        print("  ERROR: No se pudo ajustar HMM.")
        return None, np.array([]), pd.DataFrame(), pd.DataFrame(), np.array([])
    bic_df = pd.DataFrame(bic_results)

    # Reetiquetar por volatilidad
    vol_values = np.abs(features_df["log_return_1"].values[:len(best_states)])
    unique_states = np.unique(best_states)
    state_vol = {s: np.nanmean(vol_values[best_states == s]) for s in unique_states}
    sorted_states = sorted(state_vol.keys(), key=lambda x: state_vol[x])
    relabel_map = {old: new for new, old in enumerate(sorted_states)}
    best_states = np.array([relabel_map[s] for s in best_states])

    # Resumen por estado
    rows = []
    for s in range(len(unique_states)):
        mask = best_states == s
        count = int(mask.sum())
        pct = count / len(best_states) * 100.0
        mean_ret = float(np.nanmean(features_df["log_return_1"].values[:len(best_states)][mask]))
        vol = float(np.nanstd(features_df["log_return_1"].values[:len(best_states)][mask]))

        # Duración media
        runs = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        run_starts = np.where(runs == 1)[0]
        run_ends = np.where(runs == -1)[0]
        run_lengths = run_ends - run_starts
        mean_dur = float(np.mean(run_lengths)) if len(run_lengths) > 0 else 0.0
        rows.append({
            "state": s,
            "pct_time": round(pct, 1),
            "mean_return": round(mean_ret * 100, 4),
            "volatility": round(vol * 100, 4),
            "mean_duration_bars": round(mean_dur, 1),
            "description": _describe_regime(s, vol * 100, mean_ret * 100),
        })
    state_summary = pd.DataFrame(rows)

    # Matriz de transición
    trans_mat = best_model.transmat_

    # Reordenar según relabel
    reordered = np.zeros_like(trans_mat)
    for i, j in relabel_map.items():
        for k, l in relabel_map.items():
            reordered[j, l] = trans_mat[i, k]
    trans_mat = reordered
    return best_model, best_states, state_summary, bic_df, trans_mat


def _describe_regime(state: int, vol: float, mean_ret: float) -> str:
    """Genera una descripción legible del régimen para crypto (BTC diario).
    vol y mean_ret ya vienen en porcentaje (p.ej. 2.83 = 2.83%).
    Rangos típicos para BTC diario:
      - Volatilidad normal: 2.0-3.5%
      - Volatilidad alta: 3.5-5.0%
      - Volatilidad extrema: >5.0%
      - Retorno diario típico: -0.5% a +0.5%
    """
    # --- Regímenes extremos (prioridad alta) ---
    if vol >= 5.0:
        if mean_ret > 0.15:
            return "[EXPANSION ALCISTA]"      # Euforia, alta volatilidad alcista
        elif mean_ret < -0.15:
            return "[EXPANSION BAJISTA]"      # Pánico/capitulación
        else:
            return "[ALTA VOLATILIDAD]"

    # --- Trends fuertes con volatilidad elevada ---
    if vol >= 3.5:
        if mean_ret > 0.15:
            return "[TREND ALCISTA]"          # Tendencia alcista con volatilidad
        elif mean_ret < -0.15:
            return "[TREND BAJISTA]"          # Tendencia bajista con volatilidad
        else:
            return "[VOLATILIDAD NEUTRA]"     # Volátil pero sin dirección clara

    # --- Regímenes de volatilidad normal ---
    if mean_ret > 0.20:
        return "[ALCISTA FUERTE]"
    elif mean_ret > 0.08:
        return "[ALCISTA]"
    elif mean_ret < -0.20:
        return "[BAJISTA FUERTE]"
    elif mean_ret < -0.08:
        return "[BAJISTA]"
    elif mean_ret > 0.03:
        return "[ALCISTA SUAVE]"
    elif mean_ret < -0.03:
        return "[BAJISTA SUAVE]"
    else:

        # Retorno casi plano — depende de la volatilidad
        if vol < 2.5:
            return "[ACUMULACION]"            # Plano + baja vol = acumulación
        else:
            return "[LATERAL]"                # Plano + vol normal = lateral/sin dirección

# ──────────────────────────────────────────────────────────────────────────────
# SEÑAL EN VIVO

# ──────────────────────────────────────────────────────────────────────────────



# --- SUAVIZADO DE ESTADOS HMM (Mejora 1B) ---

def _smooth_states(states: np.ndarray, min_duration: int = 3) -> np.ndarray:
    """
    Filtra cambios de estado que duran menos de min_duration velas.
    Reduce falsos positivos por cambios espurios de regimen.
    """
    if len(states) < min_duration * 2:
        return states
    smoothed = states.copy()
    i = 0
    while i < len(states):
        j = i + 1
        while j < len(states) and states[j] == states[i]:
            j += 1
        change_start = j
        if change_start >= len(states):
            break
        change_end = change_start
        while change_end < len(states) and states[change_end] != states[i]:
            change_end += 1
        duration = change_end - change_start
        if 0 < duration < min_duration and change_end < len(states):
            smoothed[change_start:change_end] = states[i]
            i = change_end
        else:
            i = j if j > i else i + 1
    return smoothed




# -- FILTRO MAESTRO DE REGIMEN HMM (reemplaza el sistema de suma de pesos) --
def apply_regime_filter(df: pd.DataFrame, state_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro Maestro de Regimenes HMM.

    El regimen HMM determina la direccion PERMITIDA para las senales:
    - Regimen ALCISTA (bullish): Solo permite senales LONG. Bloquea SHORT.
    - Regimen BAJISTA (bearish): Solo permite senales SHORT. Bloquea LONG.
    - Regimen NEUTRAL: Permite ambas direcciones pero marca advertencia.

    Esto reemplaza el sistema anterior de sumar W_HMM_REGIME al score compuesto.
    El filtro NUNCA agrega senales (no crea falsos positivos),
    solo BLOQUEA senales que contradicen la direccion del mercado.

    El filtro se aplica a todo el historico para que verify_signals_historically
    mida correctamente el WR del sistema filtrado.
    """
    df = df.copy()

    # Mapa de bias por estado HMM
    state_bias_map = {}
    for _, r in state_summary.iterrows():
        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])

    # Asignar bias a cada vela segun su regimen
    df["regime_bias"] = "neutral"
    if "regime" in df.columns:
        df["regime_bias"] = df["regime"].map(state_bias_map).fillna("neutral")

    # Aplicar filtro: NUNCA agregar senales, solo BLOQUEAR
    # En regimen bajista: bloquear LONG
    bearish_mask = df["regime_bias"] == "bearish"
    if bearish_mask.any():
        df.loc[bearish_mask, "signal_long"] = False

    # En regimen alcista: bloquear SHORT
    bullish_mask = df["regime_bias"] == "bullish"
    if bullish_mask.any():
        df.loc[bullish_mask, "signal_short"] = False

    # Regimen neutral: senales permitidas pero marcar advertencia
    df["regime_warning"] = (df["regime_bias"] == "neutral") & (
        df["signal_long"] | df["signal_short"]
    )

    # Recalcular signal_score para que el dashboard refleje el bloqueo:
    # poner en 0 los scores de la direccion bloqueada (solo para display)
    df["signal_score_long_display"] = df["signal_score_long"].copy()
    df["signal_score_short_display"] = df["signal_score_short"].copy()
    if bearish_mask.any():
        df.loc[bearish_mask, "signal_score_long_display"] = 0.0
    if bullish_mask.any():
        df.loc[bullish_mask, "signal_score_short_display"] = 0.0

    return df


# -- MEJORA 2A: Detectar alertas tempranas de cambio de tendencia --
def detect_early_alerts(df, states, state_summary):
    """Detecta alertas tempranas usando regimen HMM + threshold reducido."""
    state_bias_map = {}
    for _, r in state_summary.iterrows():
        state_bias_map[int(r["state"])] = _classify_regime_bias(r["description"])
    import numpy as np
    regime_changed = np.zeros(len(df), dtype=bool)
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            regime_changed[i] = True
            for j in range(1, min(EARLY_WINDOW, len(states) - i)):
                regime_changed[i + j] = True
    df["alert_early_long"] = False
    df["alert_early_short"] = False
    for i in range(len(df)):
        if not regime_changed[i]:
            continue
        current_state = states[i]
        current_bias = state_bias_map.get(current_state, "neutral")
        score_long = df["signal_score_long"].iloc[i] if "signal_score_long" in df.columns else 0
        score_short = df["signal_score_short"].iloc[i] if "signal_score_short" in df.columns else 0
        if current_bias == "bullish" and score_long >= EARLY_THRESHOLD:
            df.loc[df.index[i], "alert_early_long"] = True
        if current_bias == "bearish" and score_short >= EARLY_THRESHOLD:
            df.loc[df.index[i], "alert_early_short"] = True
    return df


# -- ENFOQUE A: Sistema de Precursores (detectar cambios de tendencia ANTES del cruce) --
def compute_precursor_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sistema de alertas precursoras para cambios de tendencia.
    Monitorea los COMPONENTES del score compuesto que se acercan al threshold (65).
    Genera alertas cuando el score esta en zona de advertencia (45-64) y subiendo.
    """
    # Verificar que tenemos los datos necesarios
    if "signal_score_long" not in df.columns or "signal_score_short" not in df.columns:
        return df

    # Inicializar columnas
    df["precursor_long"] = False
    df["precursor_short"] = False
    df["precursor_confidence_long"] = 0.0
    df["precursor_confidence_short"] = 0.0
    df["precursor_active"] = False

    for i in range(PRECURSOR_VELOCITY_BARS, len(df)):
        # --- PROCESAR PRECURSOR LONG ---
        score_long = df["signal_score_long"].iloc[i]

        # Calcular velocidad del score (pendiente en los ultimos N velas)
        slice_long = df["signal_score_long"].iloc[i-PRECURSOR_VELOCITY_BARS:i+1].values
        if len(slice_long) >= 2:
            x = list(range(len(slice_long)))
            y = slice_long
            n = len(x)
            slope_long = (n * sum(x[j]*y[j] for j in range(n)) - sum(x)*sum(y)) / (n * sum(x[j]*x[j] for j in range(n)) - sum(x)*sum(x) + 0.001)
        else:
            slope_long = 0

        # Componentes activos para LONG
        components_long = 0
        if "bull_bias" in df.columns and df["bull_bias"].iloc[i]:
            components_long += 1
        if "squeeze_off" in df.columns and df["squeeze_off"].iloc[i]:
            components_long += 1
        if "smi_hist" in df.columns and df["smi_hist"].iloc[i] > 0:
            components_long += 1
        if "smi_delta" in df.columns and df["smi_delta"].iloc[i] > 0:
            components_long += 1
        if "adx_delta" in df.columns and df["adx_delta"].iloc[i] > 0:
            components_long += 1
        if "plus_di" in df.columns and "minus_di" in df.columns and df["plus_di"].iloc[i] > df["minus_di"].iloc[i]:
            components_long += 1

        # Determinar alerta precursora LONG
        if (PRECURSOR_THRESHOLD <= score_long < SIGNAL_SCORE_THRESHOLD
                and slope_long > 0.5
                and components_long >= PRECURSOR_MIN_COMPONENTS):
            df.loc[df.index[i], "precursor_long"] = True
            df.loc[df.index[i], "precursor_confidence_long"] = round(
                min(100, (score_long / SIGNAL_SCORE_THRESHOLD) * 100 * (components_long / 6))
            , 1)
            df.loc[df.index[i], "precursor_active"] = True

        # --- PROCESAR PRECURSOR SHORT ---
        score_short = df["signal_score_short"].iloc[i]

        # Calcular velocidad del score SHORT
        slice_short = df["signal_score_short"].iloc[i-PRECURSOR_VELOCITY_BARS:i+1].values
        if len(slice_short) >= 2:
            x = list(range(len(slice_short)))
            y = slice_short
            n = len(x)
            slope_short = (n * sum(x[j]*y[j] for j in range(n)) - sum(x)*sum(y)) / (n * sum(x[j]*x[j] for j in range(n)) - sum(x)*sum(x) + 0.001)
        else:
            slope_short = 0

        # Componentes activos para SHORT
        components_short = 0
        if "bear_bias" in df.columns and df["bear_bias"].iloc[i]:
            components_short += 1
        if "squeeze_off" in df.columns and df["squeeze_off"].iloc[i]:
            components_short += 1
        if "smi_hist" in df.columns and df["smi_hist"].iloc[i] < 0:
            components_short += 1
        if "smi_delta" in df.columns and df["smi_delta"].iloc[i] < 0:
            components_short += 1
        if "adx_delta" in df.columns and df["adx_delta"].iloc[i] > 0:
            components_short += 1
        if "plus_di" in df.columns and "minus_di" in df.columns and df["minus_di"].iloc[i] > df["plus_di"].iloc[i]:
            components_short += 1

        # Determinar alerta precursora SHORT
        if (PRECURSOR_THRESHOLD <= score_short < SIGNAL_SCORE_THRESHOLD
                and slope_short > 0.5
                and components_short >= PRECURSOR_MIN_COMPONENTS):
            df.loc[df.index[i], "precursor_short"] = True
            df.loc[df.index[i], "precursor_confidence_short"] = round(
                min(100, (score_short / SIGNAL_SCORE_THRESHOLD) * 100 * (components_short / 6))
            , 1)
            df.loc[df.index[i], "precursor_active"] = True

    return df
def _format_date(dt) -> str:
    """Convierte una fecha a formato DD-MM-AAAA."""
    if isinstance(dt, str):
        try:
            dt = pd.Timestamp(dt)
        except Exception:
            return dt
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%d-%m-%Y")
    try:
        return pd.Timestamp(dt).strftime("%d-%m-%Y")
    except Exception:
        return str(dt)


def _fmt_price(val: float) -> str:
    """Formatea precio al estilo espanol: 63.584,78$ o 1,1290$ para activos pequenos.
    Usa 2 decimales para valores grandes (BTC) y hasta 6 decimales para valores < 10.
    """
    if abs(val) < 1.0:
        n_dec = 6
    elif abs(val) < 10.0:
        n_dec = 4
    else:
        n_dec = 2
    s = f"{val:,.{n_dec}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s}$"


def _send_telegram_alert(message: str) -> bool:
    """Envia una alerta por Telegram.
    Retorna True si se envio correctamente, False si fallo o no esta configurado.
    """
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"  [Telegram] SKIP: ENABLE={ENABLE_TELEGRAM} TOKEN={bool(TELEGRAM_BOT_TOKEN)} CHAT={bool(TELEGRAM_CHAT_ID)}")
        return False
    try:
        import urllib.request as _ur
        import json as _json
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = _json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = _ur.urlopen(req, timeout=15)
        body = _json.loads(resp.read().decode("utf-8"))
        if body.get("ok"):
            print(f"  [Telegram] Enviado OK")
            return True
        else:
            print(f"  [Telegram] Error API: {body.get('description', 'desconocido')}")
            return False
    except Exception as e:
        print(f"  [Telegram] Error conexion: {e}")
        return False


def _build_telegram_message(asset, alertas):
    """Construye un mensaje formateado para Telegram."""
    lines = []
    lines.append(f"\U0001f916 <b>TradingLatino HMM - {asset}</b>")
    lines.append(f"{chr(45) * 30}")
    if alertas:
        lines.append("<b>\U0001f514 Alertas Detectadas:</b>")
        for a in alertas:
            lines.append(a)
    else:
        lines.append("\u2705 Sin alertas nuevas.")
    lines.append(f"{chr(45) * 30}")
    lines.append("<i>Enviado por TradingLatino HMM Bot</i>")
    return "\n".join(lines)


def _send_telegram_alerts_batch(asset, alertas):
    """Envia todas las alertas en un solo mensaje de Telegram."""
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    if not alertas:
        return False
    message = _build_telegram_message(asset, alertas)
    return _send_telegram_alert(message)


def _find_signal_start(df: pd.DataFrame, signal_col: str) -> int:
    """Encuentra hace cuántas velas comenzó el bloque continuo de señal actual.
    Retorna 0 si no hay señal en la última vela.
    """
    if signal_col not in df.columns or not df[signal_col].iloc[-1]:
        return 0
    bars = 0
    for i in range(len(df) - 1, -1, -1):
        if df[signal_col].iloc[i]:
            bars += 1
        else:
            break
    return bars


def compute_expiration(df: pd.DataFrame, signal_info: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    """Calcula la expiración de la señal según la regla de Jaime Merino.
    La señal debe materializarse dentro de las siguientes 10-14 velas.
    Si han pasado más velas que el máximo permitido, la señal expira.
    """
    max_bars = MAX_BARS_BY_TF.get(timeframe, 14)
    if signal_info["signal"] == "LONG":
        bars_since_start = _find_signal_start(df, "signal_long")
    elif signal_info["signal"] == "SHORT":
        bars_since_start = _find_signal_start(df, "signal_short")
    else:
        return {
            "bars_since_start": 0,
            "max_bars": max_bars,
            "bars_remaining": max_bars,
            "expired": False,
            "progress_pct": 0,
            "window_label": f"{max_bars} velas",
        }
    bars_remaining = max(0, max_bars - bars_since_start)
    expired = bars_since_start >= max_bars
    progress_pct = min(100, int(bars_since_start / max_bars * 100))

    # Labels descriptivos
    tf_labels = {"1h": "horas", "4h": "horas", "1d": "días", "1wk": "meses"}
    unit = tf_labels.get(timeframe, "velas")
    if timeframe == "1h":
        window_label = f"{max_bars} horas"
    elif timeframe == "4h":
        window_label = f"{max_bars * 4} horas"
    elif timeframe == "1d":
        window_label = f"{max_bars} días"
    elif timeframe == "1wk":
        window_label = f"~{max_bars // 4} meses"
    else:
        window_label = f"{max_bars} velas"
    return {
        "bars_since_start": bars_since_start,
        "max_bars": max_bars,
        "bars_remaining": bars_remaining,
        "expired": expired,
        "progress_pct": progress_pct,
        "window_label": window_label,
        "unit": unit,
    }


def compute_signal(df: pd.DataFrame, timeframe: Optional[str] = None) -> Dict[str, Any]:
    """Computa la señal actual (última vela) con todas las condiciones."""
    last = df.iloc[-1]
    conditions = {
        "Tendencia Alcista (Bull Bias)": {
            "met": bool(last["bull_bias"]),
            "detail": f"Close {_fmt_price(last['Close'])} > EMA_Slow {_fmt_price(last['ema_slow'])} & EMA_Fast {_fmt_price(last['ema_fast'])} > EMA_Slow"
        },
        "Tendencia Bajista (Bear Bias)": {
            "met": bool(last["bear_bias"]),
            "detail": f"Close {_fmt_price(last['Close'])} < EMA_Slow {_fmt_price(last['ema_slow'])} & EMA_Fast {_fmt_price(last['ema_fast'])} < EMA_Slow"
        },
        "Squeeze OFF ( expansión)": {
            "met": bool(last["squeeze_off"]),
            "detail": "Bandas de Bollinger fuera de Canales Keltner"
        },
        "Squeeze Release (compresión previa)": {
            "met": bool(last["squeeze_released"]),
            "detail": f"Hubo squeeze_on en las últimas {RELEASE_LOOKBACK} velas"
        },
        "SMI Hist > 0 (momentum alcista)": {
            "met": bool(last["smi_hist"] > 0),
            "detail": f"SMI Hist = {last['smi_hist']:.2f}"
        },
        "SMI Hist < 0 (momentum bajista)": {
            "met": bool(last["smi_hist"] < 0),
            "detail": f"SMI Hist = {last['smi_hist']:.2f}"
        },
        "SMI Delta > 0 (aceleración alcista)": {
            "met": bool(last["smi_delta"] > 0),
            "detail": f"SMI Delta = {last['smi_delta']:.2f}"
        },
        "SMI Delta < 0 (aceleración bajista)": {
            "met": bool(last["smi_delta"] < 0),
            "detail": f"SMI Delta = {last['smi_delta']:.2f}"
        },
        f"ADX > {ADX_THRESHOLD} (tendencia fuerte)": {
            "met": bool(last["adx"] > ADX_THRESHOLD),
            "detail": f"ADX = {last['adx']:.1f}"
        },
        "ADX Delta > 0 (tendencia fortaleciéndose)": {
            "met": bool(last["adx_delta"] > 0),
            "detail": f"ADX Delta = {last['adx_delta']:.2f}"
        },
        f"Threshold dinámico: {max(DYNAMIC_THRESHOLD_MIN, SIGNAL_SCORE_THRESHOLD - min(25, int((df['atr'].iloc[-1] / max(df['atr'].rolling(20, min_periods=1).mean().iloc[-1], 0.01)) * 5))):.0f}": {
            "met": bool(last.get("signal_score_long", 0) >= max(DYNAMIC_THRESHOLD_MIN, SIGNAL_SCORE_THRESHOLD - min(25, int((df['atr'].iloc[-1] / max(df['atr'].rolling(20, min_periods=1).mean().iloc[-1], 0.01)) * 5))) or last.get("signal_score_short", 0) >= max(DYNAMIC_THRESHOLD_MIN, SIGNAL_SCORE_THRESHOLD - min(25, int((df['atr'].iloc[-1] / max(df['atr'].rolling(20, min_periods=1).mean().iloc[-1], 0.01)) * 5)))),
            "detail": f"Score LONG={last.get('signal_score_long',0):.0f} / SHORT={last.get('signal_score_short',0):.0f}"
        },
    }
    is_long = bool(last["signal_long"])
    is_short = bool(last["signal_short"])
    signal = "LONG" if is_long else ("SHORT" if is_short else "FLAT")

    # Fuerza de la señal (0-100)
    strength = 0
    if signal == "LONG":
        strength = min(100, int(
            20 + 20
            + min(30, max(0, (last["smi_hist"] + 10) * 2))
            + min(30, max(0, (last["adx"] - ADX_THRESHOLD) * 3))
        ))
    elif signal == "SHORT":
        strength = min(100, int(
            20 + 20
            + min(30, max(0, (-last["smi_hist"] + 10) * 2))
            + min(30, max(0, (last["adx"] - ADX_THRESHOLD) * 3))
        ))

    # Signal score compuesto (usar display filtrando scores bloqueados por regimen)
    score_long = float(last.get("signal_score_long_display", last.get("signal_score_long", 0)))
    score_short = float(last.get("signal_score_short_display", last.get("signal_score_short", 0)))

    # --- Desglose de score SHORT (para visualización en dashboard) ---
    score_breakdown_short = {
        "Tendencia Bajista":          bool(last["bear_bias"]) * W_BULL_BIAS,
        "Squeeze OFF":                bool(last["squeeze_off"]) * W_SQUEEZE_OFF,
        "Squeeze Release":            bool(last["squeeze_released"]) * W_SQUEEZE_REL,
        "SMI Hist < 0":               bool(last["smi_hist"] < 0) * W_SMI_HIST,
        "SMI Delta < 0":              bool(last["smi_delta"] < 0) * W_SMI_DELTA,
        "ADX > 23":                   bool(last["adx"] > ADX_THRESHOLD) * W_ADX_THRESH,
        "ADX Delta + DI- > DI+":      bool(last["adx_delta"] > 0 and last["minus_di"] > last["plus_di"]) * W_ADX_DELTA,
        "EMA Dev < -3%":              bool(last.get("ema_deviation_pct", 0) < -3) * W_EMA_DEV,
        "EMA Dev < -8%":              bool(last.get("ema_deviation_pct", 0) < -8) * 5,
        "RSI < 45":                   bool(last.get("rsi14", 50) < 45) * W_RSI,
        "RSI > 80":                   bool(last.get("rsi14", 50) > 80) * 4,
        "Vol > 1.5x + bajista":       bool(last.get("vol_ratio", 0) > 1.5 and last.get("ema_deviation_pct", 0) < 0) * W_VOLUME,
        "Vol > 3x":                   bool(last.get("vol_ratio", 0) > 3) * 4,
        "ATR ROC > 0.2 + bajista":    bool(last.get("atr_roc", 0) > 0.2 and last.get("ema_deviation_pct", 0) < 0) * W_ATR_ROC,
        "RSI Div Bajista":          float(last.get("rsi_div_bearish", 0)),
    }
    score_breakdown_long = {
        "Tendencia Alcista":          bool(last["bull_bias"]) * W_BULL_BIAS,
        "Squeeze OFF":                bool(last["squeeze_off"]) * W_SQUEEZE_OFF,
        "Squeeze Release":            bool(last["squeeze_released"]) * W_SQUEEZE_REL,
        "SMI Hist > 0":               bool(last["smi_hist"] > 0) * W_SMI_HIST,
        "SMI Delta > 0":              bool(last["smi_delta"] > 0) * W_SMI_DELTA,
        "ADX > 23":                   bool(last["adx"] > ADX_THRESHOLD) * W_ADX_THRESH,
        "ADX Delta + DI+ > DI-":      bool(last["adx_delta"] > 0 and last["plus_di"] > last["minus_di"]) * W_ADX_DELTA,
        "EMA Dev > 3%":               bool(last.get("ema_deviation_pct", 0) > 3) * W_EMA_DEV,
        "EMA Dev > 8%":               bool(last.get("ema_deviation_pct", 0) > 8) * 5,
        "RSI > 55":                   bool(last.get("rsi14", 50) > 55) * W_RSI,
        "RSI < 25":                   bool(last.get("rsi14", 50) < 25) * 4,
        "Vol > 1.5x + alcista":       bool(last.get("vol_ratio", 0) > 1.5 and last.get("ema_deviation_pct", 0) > 0) * W_VOLUME,
        "Vol > 3x":                   bool(last.get("vol_ratio", 0) > 3) * 4,
        "ATR ROC > 0.2 + alcista":    bool(last.get("atr_roc", 0) > 0.2 and last.get("ema_deviation_pct", 0) > 0) * W_ATR_ROC,
        "RSI Div Alcista":          float(last.get("rsi_div_bullish", 0)),
    }
    # Threshold dinámico basado en volatilidad (ATR ratio)
    atr_series = df["atr"]
    atr_mean = atr_series.rolling(20, min_periods=1).mean().iloc[-1]
    atr_ratio = atr_series.iloc[-1] / atr_mean if atr_mean > 0 else 1.0
    dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, SIGNAL_SCORE_THRESHOLD - min(25, int(atr_ratio * 5)))


    # -- MEJORA 2B: Reducir threshold adicional segun el tipo de regimen --
    if "regime" in df.columns:
        # Buscar la descripcion del regimen actual en state_summary
        current_regime_desc = ""
        try:
            if df["regime"].iloc[-1] >= 0 and "state_summary" in dir():
                pass  # state_summary no disponible en este scope
        except:
            pass
        # Reducir threshold segun keywords en el nombre del regimen
        # (la descripcion se obtiene del dashboard, no de compute_signal)
        # Como alternativa, usamos ATR como proxy de volatilidad
        atr_ratio = df["atr"].iloc[-1] / max(df["atr"].rolling(20, min_periods=1).mean().iloc[-1], 0.01)
        if atr_ratio > 2.0:
            dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold - REGIME_THRESHOLD_REDUCTION["EXPANSION ALCISTA"])
        elif atr_ratio > 1.5:
            dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold - REGIME_THRESHOLD_REDUCTION["ALTA VOLATILIDAD"])
    score_used = score_long if signal == "LONG" else (score_short if signal == "SHORT" else 0)
    result = {
        "signal": signal,
        "strength": strength,
        "price": float(last["Close"]),
        "date": _format_date(df.index[-1]),
        "conditions": conditions,
        "is_long": is_long,
        "is_short": is_short,
        "signal_score_long": score_long,
        "signal_score_short": score_short,
        "signal_score_used": score_used,
        "score_threshold": SIGNAL_SCORE_THRESHOLD,
        "dynamic_threshold": dynamic_threshold,
        "atr_ratio": round(atr_ratio, 2),
        "min_consecutive_bars": MIN_CONSECUTIVE_BARS,
        "score_breakdown_short": score_breakdown_short,
        "score_breakdown_long": score_breakdown_long,
        "score_breakdown_total": max(score_breakdown_short.values()) if signal == "SHORT" else max(score_breakdown_long.values()) if signal == "LONG" else 0,
    }

    # Calcular expiración si se proporciona timeframe
    if timeframe:
        result["expiration"] = compute_expiration(df, result, timeframe)

        # Calcular fecha de inicio de la señal
        if result["signal"] in ("LONG", "SHORT"):
            bars_since = result["expiration"]["bars_since_start"]
            if bars_since > 0:
                start_idx = max(0, len(df) - bars_since)
                result["signal_start"] = _format_date(df.index[start_idx])
            else:
                result["signal_start"] = result["date"]
        else:
            result["signal_start"] = None
    else:
        result["expiration"] = None
        result["signal_start"] = None
    return result

# ──────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN HISTÓRICA DE SEÑALES (con Take-Profit Automático por Timeframe)

# ──────────────────────────────────────────────────────────────────────────────
# Analiza todas las señales LONG/SHORT pasadas y verifica si se cumplieron
# usando un **Take-Profit automático** configurado por temporalidad.
#
#   TAKE_PROFIT_PCT = { "1h": 0.5, "4h": 1.2, "1d": 2.0, "1wk": 2.0 }
#
# En lugar de esperar hasta el final de la ventana (hold-to-expiration),
# la señal se considera GANADORA si el precio alcanza el % objetivo
# en CUALQUIER MOMENTO dentro de la ventana de expiración.
#
# Esto captura el movimiento favorable antes de que el mercado revierta,
# lo que refleja mejor una operación real con take-profit.


def verify_signals_historically(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """
    Verifica históricamente si las señales LONG/SHORT se cumplieron
    usando un **Take-Profit automático** configurado por timeframe.
    El TP se obtiene de TAKE_PROFIT_PCT[timeframe] (1h=0.5%%, 4h=1.2%%, 1d=2.0%%, 1w=2.0%%).
    Para cada señal:
      - LONG: ganadora si el precio ALCANZA +TP% (High) dentro de la ventana
      - SHORT: ganadora si el precio ALCANZA -TP% (Low) dentro de la ventana
    Retorna:
        dict con estadísticas agregadas y listas de resultados por señal.
    """
    # Obtener TP base para este timeframe
    tp_target = TAKE_PROFIT_PCT.get(timeframe, TAKE_PROFIT_PCT.get("1d", 2.0))
    max_bars = MAX_BARS_BY_TF.get(timeframe, 14)

    # Labels descriptivos para la ventana
    tf_labels = {"1h": "horas", "4h": "horas", "1d": "días", "1wk": "meses"}
    if timeframe == "1h":
        window_str = f"{max_bars} horas"
    elif timeframe == "4h":
        window_str = f"{max_bars * 4} horas"
    elif timeframe == "1d":
        window_str = f"{max_bars} días"
    elif timeframe == "1wk":
        window_str = f"~{max_bars // 4} meses"
    else:
        window_str = f"{max_bars} velas"
    long_results: List[Dict] = []
    short_results: List[Dict] = []
    for i in range(len(df) - max_bars - 1):

        # --- SEÑAL LONG ---
        if df["signal_long"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0:
                continue
            close_prices = window["Close"].values
            high_prices = window["High"].values
            low_prices = window["Low"].values
            max_price = high_prices.max()
            min_price = low_prices.min()
            final_price = close_prices[-1]

            # Retornos
            max_return = (max_price - entry_price) / entry_price * 100.0
            min_return = (min_price - entry_price) / entry_price * 100.0
            final_return = (final_price - entry_price) / entry_price * 100.0

            # Velas hasta superar entry (primera vela con close > entry)
            bars_to_win: Optional[int] = None
            for j, cp in enumerate(close_prices):
                if cp > entry_price:
                    bars_to_win = j + 1  # +1 porque j=0 es la primera vela del window
                    break

            # Velas hasta que el precio tocó el máximo
            bars_to_max: Optional[int] = None
            for j in range(len(close_prices)):
                if close_prices[j] == max_price:
                    bars_to_max = j + 1
                    break
            # TP FIJO original
            won = max_return >= tp_target
            long_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price,
                "final_price": final_price,
                "max_return": round(max_return, 2),
                "min_return": round(min_return, 2),
                "final_return": round(final_return, 2),
                "bars_to_win": bars_to_win,
                "bars_to_max": bars_to_max,
                "won": won,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns and not pd.isna(df["regime"].iloc[i]) else -1,
            })

        # --- SEÑAL SHORT ---
        if df["signal_short"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            window = df.iloc[i + 1 : i + 1 + max_bars]
            if len(window) == 0:
                continue
            close_prices = window["Close"].values
            high_prices = window["High"].values
            low_prices = window["Low"].values
            max_price = high_prices.max()
            min_price = low_prices.min()
            final_price = close_prices[-1]

            # Para SHORT: ganancia si el precio BAJA
            max_return = (entry_price - min_price) / entry_price * 100.0  # favorable
            min_return = (entry_price - max_price) / entry_price * 100.0  # adverso
            final_return = (entry_price - final_price) / entry_price * 100.0

            # Velas hasta que el close bajó de entry (ganancia)
            bars_to_win: Optional[int] = None
            for j, cp in enumerate(close_prices):
                if cp < entry_price:
                    bars_to_win = j + 1
                    break

            # Velas hasta que el precio tocó el mínimo
            bars_to_min: Optional[int] = None
            for j in range(len(close_prices)):
                if close_prices[j] == min_price:
                    bars_to_min = j + 1
                    break
            # TP FIJO original
            won = max_return >= tp_target
            short_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price,
                "final_price": final_price,
                "max_return": round(max_return, 2),
                "min_return": round(min_return, 2),
                "final_return": round(final_return, 2),
                "bars_to_win": bars_to_win,
                "bars_to_min": bars_to_min,
                "won": won,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns and not pd.isna(df["regime"].iloc[i]) else -1,
            })

    # ── Estadísticas agregadas ──
    stats: Dict[str, Any] = {}
    for side, results_list in [("LONG", long_results), ("SHORT", short_results)]:
        n = len(results_list)
        if n == 0:
            stats[side] = {
                "num_signals": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "avg_max_favorable": 0.0,
                "avg_max_adverse": 0.0,
                "avg_bars_to_win": None,
            }
            continue
        wins = sum(1 for r in results_list if r["won"])
        avg_ret = float(np.mean([r["final_return"] for r in results_list]))
        avg_max_fav = float(np.mean([r["max_return"] for r in results_list]))
        avg_max_adv = float(np.mean([r["min_return"] for r in results_list]))
        btws = [r["bars_to_win"] for r in results_list if r["bars_to_win"] is not None]
        avg_btw = float(np.mean(btws)) if btws else None

        # Nuevas señales (últimos 30 días)
        last_date = df.index[-1]
        recent = [r for r in results_list if r["entry_date"] > (last_date - timedelta(days=30))]
        recent_wins = sum(1 for r in recent if r["won"])
        stats[side] = {
            "num_signals": n,
            "win_rate": round(wins / n * 100, 1),
            "wins": wins,
            "losses": n - wins,
            "avg_return": round(avg_ret, 2),
            "avg_max_favorable": round(avg_max_fav, 2),
            "avg_max_adverse": round(avg_max_adv, 2),
            "avg_bars_to_win": round(avg_btw, 1) if avg_btw is not None else None,
            "recent_signals": len(recent),
            "recent_wins": recent_wins,
            "recent_win_rate": round(recent_wins / len(recent) * 100, 1) if recent else None,
        }
    total = len(long_results) + len(short_results)
    total_wins = stats["LONG"]["wins"] + stats["SHORT"]["wins"]
    overall_win_rate = round(total_wins / total * 100, 1) if total > 0 else 0.0

    # Mejor/y peor señal
    all_signals = long_results + short_results
    best_signal = max(all_signals, key=lambda r: r["final_return"]) if all_signals else None
    worst_signal = min(all_signals, key=lambda r: r["final_return"]) if all_signals else None
    return {
        "long": long_results,
        "short": short_results,
        "stats": stats,
        "total_signals": total,
        "total_wins": total_wins,
        "overall_win_rate": overall_win_rate,
        "best_signal": best_signal,
        "worst_signal": worst_signal,
        "window_str": window_str,
        "max_bars": max_bars,
        "tp_target": tp_target,
    }


def verify_with_trailing_stop(df: pd.DataFrame, timeframe: str, trail_pct: float = 50.0) -> Dict[str, Any]:
    """
    Verifica historica de senales usando TRAILING STOP en lugar de TP fijo.
    El trailing stop sale cuando el precio retrocede un X% desde su maximo favorable.
    Esto captura ganancias antes de que el mercado revierta.
    Args:
        df: DataFrame con senales
        timeframe: Temporalidad (1h, 4h, 1d, 1wk)
        trail_pct: Porcentaje de retroceso desde el maximo para activar salida (50 = 50%)
    Retorna:
        dict con resultados de TP fijo, trailing stop, y combinado
    """
    tp_target = TAKE_PROFIT_PCT.get(timeframe, TAKE_PROFIT_PCT.get("1d", 2.0))
    max_bars = MAX_BARS_BY_TF.get(timeframe, 14)

    # Labels descriptivos para la ventana
    if timeframe == "1h":
        window_str = f"{max_bars} horas"
    elif timeframe == "4h":
        window_str = f"{max_bars * 4} horas"
    elif timeframe == "1d":
        window_str = f"{max_bars} dias"
    elif timeframe == "1wk":
        window_str = f"~{max_bars // 4} meses"
    else:
        window_str = f"{max_bars} velas"
    long_results: List[Dict] = []
    short_results: List[Dict] = []
    for i in range(len(df) - max_bars - 1):

        # --- SENAL LONG ---
        if df["signal_long"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            window = df.iloc[i + 1: i + 1 + max_bars]
            if len(window) == 0:
                continue
            close_prices = window["Close"].values
            high_prices = window["High"].values
            low_prices = window["Low"].values
            entry_price_f = float(entry_price)

            # Estrategia 1: TP FIJO (original)
            max_price = high_prices.max()
            final_price = close_prices[-1]
            max_return_tp = (max_price - entry_price_f) / entry_price_f * 100.0
            final_return_tp = (final_price - entry_price_f) / entry_price_f * 100.0
            won_tp = max_return_tp >= tp_target

            # Estrategia 2: TRAILING STOP (retroceso desde maximo)
            current_max = float(entry_price)
            exit_idx = None
            exit_price_ts = None
            for j in range(len(close_prices)):
                current_price = float(close_prices[j])
                current_high = float(high_prices[j])
                if current_high > current_max:
                    current_max = current_high
                retrace = (current_max - current_price) / current_max * 100.0
                if retrace >= trail_pct:
                    exit_idx = j + 1
                    exit_price_ts = current_price
                    break
            if exit_idx is not None:
                exit_return_ts = (exit_price_ts - entry_price_f) / entry_price_f * 100.0
                won_ts = exit_return_ts > 0
            else:
                exit_return_ts = (float(close_prices[-1]) - entry_price_f) / entry_price_f * 100.0
                won_ts = exit_return_ts > 0
            won_combined = won_tp or won_ts
            if won_tp:
                best_return_combined = max_return_tp
            else:
                best_return_combined = exit_return_ts
            long_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price_f,
                "max_return_tp": round(max_return_tp, 2),
                "final_return_tp": round(final_return_tp, 2),
                "won_tp": won_tp,
                "exit_return_ts": round(exit_return_ts, 2),
                "won_ts": won_ts,
                "won_combined": won_combined,
                "best_return_combined": round(best_return_combined, 2),
                "trail_activated": exit_idx is not None,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns and not pd.isna(df["regime"].iloc[i]) else -1,
            })

        # --- SENAL SHORT ---
        if df["signal_short"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            window = df.iloc[i + 1: i + 1 + max_bars]
            if len(window) == 0:
                continue
            close_prices = window["Close"].values
            high_prices = window["High"].values
            low_prices = window["Low"].values
            entry_price_f = float(entry_price)

            # Estrategia 1: TP FIJO (original) para SHORT
            min_price = low_prices.min()
            final_price = close_prices[-1]
            max_return_tp = (entry_price_f - min_price) / entry_price_f * 100.0
            final_return_tp = (entry_price_f - final_price) / entry_price_f * 100.0
            won_tp = max_return_tp >= tp_target

            # Estrategia 2: TRAILING STOP (para SHORT - seguimos el minimo)
            current_min = float(entry_price)
            exit_idx = None
            exit_price_ts = None
            for j in range(len(close_prices)):
                current_price = float(close_prices[j])
                current_low = float(low_prices[j])
                if current_low < current_min:
                    current_min = current_low
                retrace = (current_price - current_min) / current_min * 100.0
                if retrace >= trail_pct:
                    exit_idx = j + 1
                    exit_price_ts = current_price
                    break
            if exit_idx is not None:
                exit_return_ts = (entry_price_f - exit_price_ts) / entry_price_f * 100.0
                won_ts = exit_return_ts > 0
            else:
                exit_return_ts = (entry_price_f - float(close_prices[-1])) / entry_price_f * 100.0
                won_ts = exit_return_ts > 0
            won_combined = won_tp or won_ts
            if won_tp:
                best_return_combined = max_return_tp
            else:
                best_return_combined = exit_return_ts
            short_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price_f,
                "max_return_tp": round(max_return_tp, 2),
                "final_return_tp": round(final_return_tp, 2),
                "won_tp": won_tp,
                "exit_return_ts": round(exit_return_ts, 2),
                "won_ts": won_ts,
                "won_combined": won_combined,
                "best_return_combined": round(best_return_combined, 2),
                "trail_activated": exit_idx is not None,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns and not pd.isna(df["regime"].iloc[i]) else -1,
            })

    # --- Estadisticas Agregadas ---
    stats: Dict[str, Any] = {}
    for side, results_list in [("LONG", long_results), ("SHORT", short_results)]:
        n = len(results_list)
        if n == 0:
            stats[side] = {
                "num_signals": 0,
                "win_rate_tp": 0.0,
                "win_rate_ts": 0.0,
                "win_rate_combined": 0.0,
                "avg_return_tp": 0.0,
                "avg_return_ts": 0.0,
                "avg_return_combined": 0.0,
                "trail_activated_pct": 0.0,
            }
            continue
        wins_tp = sum(1 for r in results_list if r["won_tp"])
        wins_ts = sum(1 for r in results_list if r["won_ts"])
        wins_combined = sum(1 for r in results_list if r["won_combined"])
        trail_activated = sum(1 for r in results_list if r["trail_activated"])
        avg_ret_tp = float(np.mean([r["max_return_tp"] for r in results_list]))
        avg_ret_ts = float(np.mean([r["exit_return_ts"] for r in results_list]))
        avg_ret_combined = float(np.mean([r["best_return_combined"] for r in results_list]))
        stats[side] = {
            "num_signals": n,
            "win_rate_tp": round(wins_tp / n * 100, 1),
            "wins_tp": wins_tp,
            "win_rate_ts": round(wins_ts / n * 100, 1),
            "wins_ts": wins_ts,
            "win_rate_combined": round(wins_combined / n * 100, 1),
            "wins_combined": wins_combined,
            "avg_return_tp": round(avg_ret_tp, 2),
            "avg_return_ts": round(avg_ret_ts, 2),
            "avg_return_combined": round(avg_ret_combined, 2),
            "trail_activated": trail_activated,
            "trail_activated_pct": round(trail_activated / n * 100, 1) if n > 0 else 0.0,
        }
    total = len(long_results) + len(short_results)
    total_wins_tp = stats["LONG"]["wins_tp"] + stats["SHORT"]["wins_tp"]
    total_wins_ts = stats["LONG"]["wins_ts"] + stats["SHORT"]["wins_ts"]
    total_wins_combined = stats["LONG"]["wins_combined"] + stats["SHORT"]["wins_combined"]
    overall_win_rate_tp = round(total_wins_tp / total * 100, 1) if total > 0 else 0.0
    overall_win_rate_ts = round(total_wins_ts / total * 100, 1) if total > 0 else 0.0
    overall_win_rate_combined = round(total_wins_combined / total * 100, 1) if total > 0 else 0.0
    return {
        "long": long_results,
        "short": short_results,
        "stats": stats,
        "total_signals": total,
        "total_wins_tp": total_wins_tp,
        "total_wins_ts": total_wins_ts,
        "total_wins_combined": total_wins_combined,
        "overall_win_rate_tp": overall_win_rate_tp,
        "overall_win_rate_ts": overall_win_rate_ts,
        "overall_win_rate_combined": overall_win_rate_combined,
        "window_str": window_str,
        "max_bars": max_bars,
        "tp_target": tp_target,
        "trail_pct": trail_pct,
    }

# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD HTML

# ──────────────────────────────────────────────────────────────────────────────
REGIME_COLORS = ["#089981", "#3498DB", "#FF851B", "#F23645", "#B10DC9", "#F012BE"]


def _current_regime_index(states: np.ndarray) -> int:
    """Devuelve el régimen de la última vela."""
    return int(states[-1]) if len(states) > 0 else -1


def _detect_regime_changes(states: np.ndarray, index, state_summary: pd.DataFrame, max_alerts: int = 15):
    """Detecta cambios de régimen y devuelve los más recientes."""
    desc_map: Dict[int, str] = {}
    for _, r in state_summary.iterrows():
        desc_map[int(r["state"])] = r["description"]
    changes = []
    prev_state = states[0]
    change_start = 0
    for i in range(1, len(states)):
        if states[i] != prev_state:
            if prev_state >= 0 and states[i] >= 0:
                changes.append({
                    "from_state": int(prev_state),
                    "to_state": int(states[i]),
                    "from_desc": desc_map.get(int(prev_state), f"R{prev_state}"),
                    "to_desc": desc_map.get(int(states[i]), f"R{states[i]}"),
                    "date": _format_date(index[i]),
                    "duration_velas": i - change_start,
                    "from_color": REGIME_COLORS[int(prev_state) % len(REGIME_COLORS)],
                    "to_color": REGIME_COLORS[int(states[i]) % len(REGIME_COLORS)],
                })
            prev_state = states[i]
            change_start = i

    # Más recientes primero, limitado
    return list(reversed(changes[-max_alerts:]))

# Signal label - clean text for HTML rendering
SIGNAL_LABELS = {"LONG": "LONG", "SHORT": "SHORT", "FLAT": "SIN SENAL"}


def _build_change_summary(regime_changes: Optional[List[Dict]], state_summary: pd.DataFrame) -> str:
    """Genera un resumen visual de estadisticas agregadas de cambios de regimen."""
    if not regime_changes or len(regime_changes) == 0:
        return ""
    total_cambios = len(regime_changes)

    # Duracion media general (de los cambios detectados)
    duraciones = [c["duration_velas"] for c in regime_changes]
    avg_dur = float(np.mean(duraciones)) if duraciones else 0.0

    # Regimen mas persistente (mayor duracion media desde state_summary)
    if not state_summary.empty:
        max_dur_row = state_summary.loc[state_summary["mean_duration_bars"].idxmax()]
        most_persistent = max_dur_row["description"]
        most_persistent_dur = max_dur_row["mean_duration_bars"]
        most_persistent_state = int(max_dur_row["state"])
        most_persistent_color = REGIME_COLORS[most_persistent_state % len(REGIME_COLORS)]
    else:
        most_persistent = "-"
        most_persistent_dur = 0
        most_persistent_color = "#888"

    # Ultimo cambio (mas reciente primero)
    ultimo = regime_changes[0]
    ultimo_date = _format_date(ultimo["date"])
    ultimo_from = ultimo["from_desc"]
    ultimo_to = ultimo["to_desc"]
    ultimo_from_color = ultimo["from_color"]
    ultimo_to_color = ultimo["to_color"]
    html = f"""
    <div class="change-summary">
        <div class="summary-stat">
            <span class="summary-stat-value">{total_cambios}</span>
            <span class="summary-stat-label">Cambios detectados</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value" style="color:{most_persistent_color}">{most_persistent}</span>
            <span class="summary-stat-label">Mas persistente ({most_persistent_dur:.0f}v promedio)</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value">{avg_dur:.0f}</span>
            <span class="summary-stat-label">Duracion promedio (velas)</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-value" style="font-size:0.75rem;">
                <span style="color:{ultimo_from_color}">{ultimo_from}</span>
                <span style="color:#666;margin:0 4px;">→</span>
                <span style="color:{ultimo_to_color}">{ultimo_to}</span>
            </span>
            <span class="summary-stat-label">Ultimo cambio ({ultimo_date})</span>
        </div>
    </div>"""
    return html


def _regime_alignment_badge(alignment: str, signal: str) -> str:
    """Genera un badge HTML que muestra si el regimen actual es favorable o adverso para la senal activa."""
    if signal not in ("LONG", "SHORT") or alignment == "no_signal":
        return ""
    if alignment == "favorable":
        return (
            f'<div class="regime-alignment-badge alignment-favorable">'
            f'<span class="alignment-icon">✅</span>'
            f'<span class="alignment-text">Regimen FAVORABLE para {signal}</span>'
            f'<span class="alignment-sub">El mercado esta alineado con tu trade — Mantener</span>'
            f'</div>'
        )
    elif alignment == "adverse":
        return (
            f'<div class="regime-alignment-badge alignment-adverse">'
            f'<span class="alignment-icon">🛑</span>'
            f'<span class="alignment-text">Regimen ADVERSO para {signal}</span>'
            f'<span class="alignment-sub">El mercado esta en contra de tu trade — Considerar salir</span>'
            f'</div>'
        )
    else:
        return (
            f'<div class="regime-alignment-badge alignment-neutral">'
            f'<span class="alignment-icon">⚠️</span>'
            f'<span class="alignment-text">Regimen NEUTRAL para {signal}</span>'
            f'<span class="alignment-sub">Mercado sin direccion clara — Gestionar riesgo</span>'
            f'</div>'
        )
@dataclass


class TimeframeData:
    """Resultados completos para una temporalidad."""
    df_full: pd.DataFrame
    states_full: np.ndarray
    state_summary: pd.DataFrame
    trans_mat: Optional[np.ndarray]
    signal_info: Dict[str, Any]
    regime_changes: List[Dict]
    verification: Optional[Dict[str, Any]] = None
    trailing_verification: Optional[Dict[str, Any]] = None


def _calculate_regime_entropy(trans_mat: np.ndarray, state: int) -> float:
    """Calcula la confianza del HMM basada en entropia normalizada.
    Menor entropia = mayor confianza en la prediccion.
    Retorna 0-100 (100 = maxima confianza)."""
    import numpy as np
    if trans_mat is None or state < 0 or state >= len(trans_mat):
        return 50.0
    probs = trans_mat[state]
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 50.0
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(trans_mat))
    if max_entropy == 0:
        return 100.0
    normalized_entropy = entropy / max_entropy
    confidence = (1.0 - normalized_entropy) * 100.0
    return round(confidence, 1)


def _calculate_regime_duration_ratio(states: np.ndarray, current_state: int, state_summary: pd.DataFrame) -> dict:
    """Calcula cuantas velas lleva el regimen actual vs su duracion media."""
    if len(states) == 0 or current_state < 0:
        return {"current_duration": 0, "mean_duration": 1, "ratio": 0.0, "score": 50}

    # Contar velas consecutivas del regimen actual desde el final
    count = 0
    for i in range(len(states) - 1, -1, -1):
        if states[i] == current_state:
            count += 1
        else:
            break

    # Duracion media del regimen actual
    mean_dur = 1.0
    if not state_summary.empty:
        row = state_summary[state_summary["state"] == current_state]
        if not row.empty:
            mean_dur = float(row.iloc[0]["mean_duration_bars"])
            if mean_dur <= 0:
                mean_dur = 1.0
    ratio = count / mean_dur if mean_dur > 0 else 1.0

    # Score: menor ratio = mejor (recien empezado)
    if ratio <= 0.5:
        score = 100  # Recien empezado, mucha vida por delante
    elif ratio <= 1.0:
        score = 80   # Dentro del promedio
    elif ratio <= 1.3:
        score = 50   # Alargandose
    else:
        score = 20   # Muy alargado, probable cambio inminente
    return {"current_duration": count, "mean_duration": round(mean_dur, 1), "ratio": round(ratio, 2), "score": score}


def _calculate_impact_score(regime_warnings: List[Dict], signal: str) -> dict:
    """Calcula el score de impacto neto de los ultimos cambios de regimen."""
    if not regime_warnings or signal not in ("LONG", "SHORT"):
        return {"net_score": 0, "favorable": 0, "adverse": 0, "neutral": 0, "total": 0, "score": 50}
    favorable = sum(1 for w in regime_warnings if w["impact"] == "favorable")
    adverse = sum(1 for w in regime_warnings if w["impact"] == "adverse")
    neutral = sum(1 for w in regime_warnings if w["impact"] == "neutral")
    total = len(regime_warnings)
    net = favorable - adverse

    # Score: net / max_possible * 100, normalizado a 0-100
    max_possible = total  # best case: all favorable
    min_possible = -total  # worst case: all adverse
    if max_possible == min_possible:
        score = 50
    else:

        # Normalizar net de [-total, +total] a [0, 100]
        score = ((net - min_possible) / (max_possible - min_possible)) * 100.0
    return {"net_score": net, "favorable": favorable, "adverse": adverse, "neutral": neutral, "total": total, "score": round(score, 1)}


def _calculate_wr_by_regime(verification: Optional[Dict], current_state: int, signal: str) -> dict:
    """Calcula el win rate historico de senales en el regimen actual."""
    if not verification or current_state < 0:
        return {"win_rate": None, "signals": 0, "score": 50}
    side = signal if signal in ("LONG", "SHORT") else None
    if not side:
        return {"win_rate": None, "signals": 0, "score": 50}
    results = verification.get(side.lower(), [])
    regime_signals = [r for r in results if r.get("regime", -1) == current_state]
    if not regime_signals:

        # Usar datos de todos los regimenes si no hay suficientes
        all_signals = verification.get(side.lower(), [])
        if len(all_signals) > 0:
            wr = sum(1 for r in all_signals if r["won"]) / len(all_signals) * 100
            score = wr
            return {"win_rate": round(wr, 1), "signals": len(all_signals), "score": score, "note": "todos los regimenes"}
    n = len(regime_signals)
    if n == 0:
        return {"win_rate": None, "signals": 0, "score": 50}
    wins = sum(1 for r in regime_signals if r["won"])
    wr = wins / n * 100

    # Score = WR directo, pero con ajuste por numero de muestras
    confidence_mult = min(1.0, n / 10)  # 10+ senales = confianza total
    score = wr * confidence_mult + 50 * (1 - confidence_mult)
    return {"win_rate": round(wr, 1), "signals": n, "score": round(score, 1)}


def _build_trade_health_meter(
    states: np.ndarray,
    state_summary: pd.DataFrame,
    regime_warnings: List[Dict],
    trans_mat: Optional[np.ndarray],
    current_state: int,
    signal: str,
    regime_alignment: str,
    signal_info: Dict[str, Any],
    verification: Optional[Dict[str, Any]],
    df_full: pd.DataFrame,
) -> str:
    """Genera el TRADE HEALTH METER: panel visual con 4 contadores + veredicto accionable.
    Combina: Estabilidad de regimen, Impacto neto, Win Rate por regimen, Confianza HMM."""
    import numpy as np

    # ── 1) Estabilidad del Regimen ──
    stability = _calculate_regime_duration_ratio(states, current_state, state_summary)

    # ── 2) Score de Impacto Neto ──
    impact = _calculate_impact_score(regime_warnings, signal)

    # ── 3) Win Rate por Regimen ──
    wr_data = _calculate_wr_by_regime(verification, current_state, signal)

    # ── 4) Confianza HMM (Entropia) ──
    confidence = _calculate_regime_entropy(trans_mat, current_state)

    # ── Health Score Ponderado (25% cada uno) ──
    health_score = (
        stability["score"] * 0.25
        + impact["score"] * 0.25
        + wr_data["score"] * 0.25
        + confidence * 0.25
    )
    health_score = round(min(100, max(0, health_score)), 1)

    # ── Veredicto ──
    if health_score >= 70 and regime_alignment != "adverse":
        verdict = "MANTENER"
        verdict_icon = "🟢"
        verdict_color = "#089981"
        verdict_desc = "El trade esta saludable. Todos los indicadores respaldan la posicion."
        action_text = "✅ Mantener trade — Regimen favorable con alta confianza"
        action_color = "#089981"
    elif health_score >= 45 and regime_alignment != "adverse":
        verdict = "PRECAUCION"
        verdict_icon = "🟡"
        verdict_color = "#2962FF"
        verdict_desc = "Senales mixtas. Monitorea de cerca y ajusta stops."
        action_text = "⚠️ Monitorear — Algunos indicadores muestran cautela"
        action_color = "#2962FF"
    else:
        verdict = "SALIR / NO ENTRAR"
        verdict_icon = "🔴"
        verdict_color = "#F23645"
        verdict_desc = "Condiciones adversas detectadas. Considera salir o no abrir posicion."
        action_text = "🔴 Considerar salir — Factores en contra del trade"
        action_color = "#F23645"

    # Mejorar descripcion si es FLAT
    if signal == "FLAT":
        if health_score >= 70:
            verdict = "FAVORABLE"
            verdict_icon = "🟢"
            verdict_color = "#089981"
            verdict_desc = "Mercado en condiciones favorables. Preparado para la proxima senal."
            action_text = "✅ Esperar senal — Condiciones de mercado favorables"
            action_color = "#089981"
        elif health_score >= 45:
            verdict = "NEUTRAL"
            verdict_icon = "🟡"
            verdict_color = "#2962FF"
            verdict_desc = "Mercado sin direccion clara. Esperar confirmacion."
            action_text = "⏳ Esperar confirmacion — Mercado neutral"
            action_color = "#2962FF"
        else:
            verdict = "DESFAVORABLE"
            verdict_icon = "🔴"
            verdict_color = "#F23645"
            verdict_desc = "Mercado en condiciones adversas. Evitar operar."
            action_text = "🚫 Evitar operar — Condiciones adversas del mercado"
            action_color = "#F23645"

    # ── Alertas activas ──
    alerts_list = []
    if stability["score"] < 40:
        alerts_list.append(f"⏳ Regimen alargado: {stability['current_duration']}v (media {stability['mean_duration']}v) — posible cambio")
    if impact["adverse"] >= 2:
        alerts_list.append(f"📉 {impact['adverse']} cambios adversos en ultimos {impact['total']} cambios de regimen")
    if wr_data["win_rate"] is not None and wr_data["win_rate"] < 50 and wr_data["signals"] >= 3:
        alerts_list.append(f"📊 Win Rate bajo en este regimen: {wr_data['win_rate']:.0f}% ({wr_data['signals']} senales)")
    if confidence < 40:
        alerts_list.append(f"🔮 Confianza HMM baja ({confidence:.0f}%) — prediccion no fiable")
    if stability["score"] > 80 and impact["score"] > 70 and confidence > 70:
        alerts_list.append(f"✅ Todos los indicadores positivos — regimen estable y favorable")
    alerts_html = ""
    if alerts_list:
        alerts_html = '<div class="health-alerts">'
        for alert in alerts_list:
            alerts_html += f'<div class="health-alert-row">{alert}</div>'
        alerts_html += "</div>"

    # ── Construir HTML ──
    # Barra de salud
    bar_color = "#089981" if health_score >= 70 else ("#2962FF" if health_score >= 45 else "#F23645")

    # Stability display
    if stability["score"] >= 70:
        stability_icon, stability_label = "✅", "Estable"
    elif stability["score"] >= 40:
        stability_icon, stability_label = "⚠️", "Alargado"
    else:
        stability_icon, stability_label = "🔴", "Critico"

    # Impact display
    if impact["net_score"] > 0:
        impact_icon, impact_label = "✅", f"+{impact['net_score']}"
    elif impact["net_score"] == 0:
        impact_icon, impact_label = "➖", "0"
    else:
        impact_icon, impact_label = "🔴", f"{impact['net_score']}"

    # WR display
    if wr_data["win_rate"] is not None:
        wr_display = f"{wr_data['win_rate']:.0f}%"
        wr_icon = "🏆" if wr_data['win_rate'] >= 60 else ("📊" if wr_data['win_rate'] >= 40 else "⚠️")
    else:
        wr_display = "—"
        wr_icon = "📊"
    wr_note = f" ({wr_data['signals']} sig.)" if wr_data['signals'] > 0 else ""

    # Confidence display
    if confidence >= 70:
        conf_icon, conf_label = "🔮", f"{confidence:.0f}%"
    elif confidence >= 45:
        conf_icon, conf_label = "🔮", f"{confidence:.0f}%"
    else:
        conf_icon, conf_label = "🔮", f"{confidence:.0f}%"
    regime_color = REGIME_COLORS[current_state % len(REGIME_COLORS)] if current_state >= 0 else "#888"
    html = f"""
    <div class="section" style="margin-top:20px;">
        <div class="section-title">🏥 TRADE HEALTH METER &mdash; ¿Entrar, Mantener o Salir?</div>
        <div class="health-meter-container">
            <!-- HEADER: Veredicto -->
            <div class="health-verdict" style="border-left:4px solid {verdict_color};">
                <div class="health-verdict-row">
                    <span class="health-verdict-icon">{verdict_icon}</span>
                    <div class="health-verdict-info">
                        <span class="health-verdict-label" style="color:{verdict_color};">{verdict}</span>
                        <span class="health-verdict-desc">{verdict_desc}</span>
                    </div>
                    <div class="health-score-ring" style="border-color:{bar_color};">
                        <span class="health-score-value" style="color:{bar_color};">{health_score:.0f}</span>
                        <span class="health-score-label">/100</span>
                    </div>
                </div>
            </div>
            <!-- BARRA DE SALUD -->
            <div class="health-bar-container">
                <div class="health-bar-track">
                    <div class="health-bar-fill" style="width:{health_score}%;background:{bar_color};"></div>
                </div>
                <div class="health-bar-labels">
                    <span style="color:#FF4136;">🔴 Salir</span>
                    <span style="color:#f39c12;">🟡 Precaución</span>
                    <span style="color:#2ECC40;">🟢 Mantener</span>
                </div>
            </div>
            <!-- ACCION SUGERIDA -->
            <div class="health-action" style="border-left-color:{action_color};">
                <span class="health-action-label">🎯 Accion sugerida:</span>
                <span class="health-action-text" style="color:{action_color};">{action_text}</span>
            </div>
            <!-- 4 CONTADORES -->
            <div class="health-meters-grid">
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if stability['score'] >= 70 else ('#f39c12' if stability['score'] >= 40 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">⏳</span>
                        <span class="health-meter-title">Estabilidad</span>
                    </div>
                    <div class="health-meter-value">{stability['current_duration']}v</div>
                    <div class="health-meter-sub">Media: {stability['mean_duration']}v ({stability['ratio']:.1f}x)</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if stability['score'] >= 70 else ('#f39c12' if stability['score'] >= 40 else '#FF4136')};">{stability_icon} {stability_label}</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if impact['net_score'] > 0 else ('#888' if impact['net_score'] == 0 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">📊</span>
                        <span class="health-meter-title">Impacto Neto</span>
                    </div>
                    <div class="health-meter-value">{impact_icon} {impact_label}</div>
                    <div class="health-meter-sub">Fav {impact['favorable']} / Adv {impact['adverse']}</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if impact['net_score'] > 0 else ('#888' if impact['net_score'] == 0 else '#FF4136')};">Ult. {impact['total']} cambios</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if wr_data['win_rate'] and wr_data['win_rate'] >= 60 else ('#f39c12' if wr_data['win_rate'] and wr_data['win_rate'] >= 40 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">{wr_icon}</span>
                        <span class="health-meter-title">Win Rate</span>
                    </div>
                    <div class="health-meter-value">{wr_display}</div>
                    <div class="health-meter-sub">Regimen actual{wr_note}</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if wr_data['win_rate'] and wr_data['win_rate'] >= 60 else ('#f39c12' if wr_data['win_rate'] and wr_data['win_rate'] >= 40 else '#888')};">{wr_data.get('note', f'R{current_state}') if current_state >= 0 else '—'}</div>
                </div>
                <div class="health-meter-card" style="border-top:3px solid {'#2ECC40' if confidence >= 70 else ('#f39c12' if confidence >= 45 else '#FF4136')};">
                    <div class="health-meter-header">
                        <span class="health-meter-icon">{conf_icon}</span>
                        <span class="health-meter-title">Confianza HMM</span>
                    </div>
                    <div class="health-meter-value">{conf_label}</div>
                    <div class="health-meter-sub">Entropía normalizada</div>
                    <div class="health-meter-status" style="color:{'#2ECC40' if confidence >= 70 else ('#f39c12' if confidence >= 45 else '#FF4136')};">{'Alta' if confidence >= 70 else ('Media' if confidence >= 45 else 'Baja')} confianza</div>
                </div>
            </div>
            <!-- ALERTAS ACTIVAS -->
            {alerts_html}
        </div>
    </div>"""
    return html


def _score_breakdown_html(signal_info: Dict, signal: str) -> str:
    """Genera el HTML del desglose de score SHORT o LONG."""
    if signal == "FLAT":
        return ""
    breakdown = signal_info.get("score_breakdown_short" if signal == "SHORT" else "score_breakdown_long", {})
    if not breakdown:
        return ""
    score_used = signal_info.get("signal_score_short" if signal == "SHORT" else "signal_score_long", 0)
    threshold = signal_info.get("score_threshold", 65)
    dynamic = signal_info.get("dynamic_threshold", 45)
    actual_threshold = dynamic
    color = "#F23645" if signal == "SHORT" else "#089981"
    bar_color = "#F23645" if signal == "SHORT" else "#089981"
    label = "SHORT" if signal == "SHORT" else "LONG"

    # Filtrar solo componentes con peso > 0
    active = {k: v for k, v in breakdown.items() if v > 0}
    if not active:
        return ""

    rows = ""
    for comp_name, comp_val in sorted(active.items(), key=lambda x: x[1], reverse=True):
        bar_w = min(100, comp_val)
        bg = bar_color + "33"
        fg = bar_color
        if comp_val >= 10:
            bar_w = int(comp_val / 16 * 100)  # scale relative to max possible (16)
        else:
            bar_w = int(comp_val / 5 * 100)
        bar_w = max(10, min(100, bar_w))
        rows += f"""<div class="sb-row">
            <span class="sb-label">{comp_name}</span>
            <div class="sb-bar-track"><div class="sb-bar-fill" style="width:{bar_w}%;background:{fg};"></div></div>
            <span class="sb-value">+{comp_val:.0f}</span>
        </div>"""

    # Barra de threshold
    th_pct = min(100, int(actual_threshold / 100 * 100))
    th_label = f"Threshold: {actual_threshold}"
    active_sum = sum(breakdown.values())
    meet_th = "SI" if score_used >= actual_threshold else "NO"
    meet_color = "#089981" if score_used >= actual_threshold else "#F23645"

    score_met = ""
    if score_used >= actual_threshold:
        score_met = f'<div class="sb-score-met" style="color:#089981;">Senal {label} ACTIVADA (Score {score_used:.0f} >= {actual_threshold})</div>'
    else:
        score_met = f'<div class="sb-score-not-met" style="color:#F23645;">Score insuficiente ({score_used:.0f} < {actual_threshold})</div>'

    return f"""<div class="sb-container">
        <div class="sb-header">
            <span class="sb-title" style="color:{color};">Analisis del Score {label}</span>
            <span class="sb-total" style="background:{color}22;color:{color};">Score: {score_used:.0f} / {actual_threshold}</span>
        </div>
        <div class="sb-body">
            {rows}
            <div class="sb-threshold-bar" style="margin-top:8px;">
                <span class="sb-threshold-label">Umbral: {actual_threshold}</span>
                <div class="sb-bar-track"><div class="sb-bar-fill" style="width:{min(100, score_used)}%;background:{meet_color};"></div></div>
                <span class="sb-value" style="color:{meet_color};">{score_used:.0f}</span>
            </div>
        </div>
        {score_met}
    </div>"""


def _generate_tf_inner(data: TimeframeData, asset: str, timeframe: str) -> str:
    """Genera el HTML interno (secciones del body) para UNA temporalidad."""
    df_full = data.df_full
    states = data.states_full
    state_summary = data.state_summary
    trans_mat = data.trans_mat
    signal_info = data.signal_info
    regime_changes = data.regime_changes
    current_state = _current_regime_index(states)
    signal = signal_info["signal"]
    strength = signal_info["strength"]
    price = signal_info["price"]
    date = signal_info["date"]

    # Precio cambio %
    price_change = df_full["Close"].pct_change().iloc[-1] * 100 if len(df_full) > 1 else 0.0
    price_color = "#089981" if price_change >= 0 else "#F23645"
    price_arrow = "▲" if price_change >= 0 else "▼"
    signal_color = "#089981" if signal == "LONG" else ("#F23645" if signal == "SHORT" else "#FF851B")

    # Fecha de inicio de la señal actual
    signal_start = signal_info.get("signal_start")
    signal_start_html = ""
    if signal_start and signal in ("LONG", "SHORT"):
        bars_since = signal_info.get("expiration", {}).get("bars_since_start", 0)
        tf_unit = {"1h": "h", "4h": "h", "1d": "d", "1wk": "sem"}.get(timeframe, "v")
        signal_start_html = f'''
                <div class="signal-start-box">
                    <span class="signal-start-icon">📅</span>
                    <div class="signal-start-info">
                        <span class="signal-start-label">Activa desde</span>
                        <span class="signal-start-date">{signal_start}</span>
                        <span class="signal-start-bars">(hace {bars_since} {tf_unit})</span>
                    </div>
                </div>'''

    # ──────────────────────────────────────────────────────────────────────────
    # EXPIRACIÓN (Regla de Jaime Merino)

    # ──────────────────────────────────────────────────────────────────────────
    expiration = signal_info.get("expiration")
    expiration_html = ""
    expired_badge_html = ""
    if expiration and signal in ("LONG", "SHORT"):
        exp = expiration
        bars_since = exp["bars_since_start"]
        max_bars = exp["max_bars"]
        bars_rem = exp["bars_remaining"]
        expired = exp["expired"]
        progress = exp["progress_pct"]
        window_label = exp["window_label"]

        # Color de la barra de progreso
        if expired:
            prog_color = "#F23645"
        elif progress >= 75:
            prog_color = "#FF851B"
        elif progress >= 50:
            prog_color = "#2962FF"
        else:
            prog_color = "#089981"
        expired_badge_html = (
            '<span class="expired-badge">EXPIRADA</span>'
            if expired else
            f'<span class="expiration-badge" style="color:{prog_color};">'
            f'{bars_rem}/{max_bars} restantes</span>'
        )
        expiration_html = f"""
        <div class="expiration-container">
            <div class="expiration-header">
                <span class="expiration-label">Ventana Jaime Merino</span>
                <span class="expiration-window">{window_label}</span>
            </div>
            <div class="expiration-bar-track">
                <div class="expiration-bar-fill" style="width:{progress}%;background:{prog_color};"></div>
            </div>
            <div class="expiration-details">
                <span>Vela {bars_since} de {max_bars}</span>
                <span class="{"expired-text" if expired else ""}">
                    {"⛔ EXPIRADA" if expired else f"{bars_rem} velas restantes"}
                </span>
            </div>
        </div>"""

    # Descripcion del regimen actual
    regime_desc = ""
    regime_duration = "-"
    regime_pct = "-"
    if not state_summary.empty and current_state >= 0:
        row = state_summary[state_summary["state"] == current_state]
        if not row.empty:
            regime_desc = row.iloc[0]["description"]
            regime_duration = f"{row.iloc[0]['mean_duration_bars']}"
            regime_pct = f"{row.iloc[0]['pct_time']}%"
    regime_color = REGIME_COLORS[current_state % len(REGIME_COLORS)] if current_state >= 0 else "#888"

    # ── Regime alignment assessment for trade warnings ──
    current_bias = _classify_regime_bias(regime_desc)
    if signal == "LONG":
        regime_alignment = "favorable" if current_bias == "bullish" else ("adverse" if current_bias == "bearish" else "neutral")
    elif signal == "SHORT":
        regime_alignment = "favorable" if current_bias == "bearish" else ("adverse" if current_bias == "bullish" else "neutral")
    else:
        regime_alignment = "no_signal"

    # Assess recent regime changes for warnings
    regime_warnings = []
    for change in regime_changes[:5]:
        to_bias = _classify_regime_bias(change["to_desc"])
        if signal == "LONG":
            impact = "adverse" if to_bias == "bearish" else ("favorable" if to_bias == "bullish" else "neutral")
        elif signal == "SHORT":
            impact = "adverse" if to_bias == "bullish" else ("favorable" if to_bias == "bearish" else "neutral")
        else:
            impact = "neutral"
        regime_warnings.append({**change, "impact": impact})

    # Grafico 1 - Precio + EMA55 (simple, sin regimenes ni senales)
    fig_simple = _make_simple_price_chart(df_full, asset, timeframe)
    plotly_config = dict(
        scrollZoom=True, displayModeBar=True, displaylogo=False,
        doubleClick="reset", responsive=True,
        modeBarButtonsToRemove=["lasso2d", "select2d", "sendDataToCloud"],
    )
    simple_price_chart_html = fig_simple.to_html(full_html=False, include_plotlyjs=False, div_id=f"spc-{timeframe}", config=plotly_config)

    # Grafico 2 - ADX + Squeeze Momentum
    fig_ind = _make_indicators_chart(df_full, states, asset, timeframe, signal_info)
    indicators_chart_html = fig_ind.to_html(full_html=False, include_plotlyjs=False, div_id=f"ic-{timeframe}", config=plotly_config)

    # Grafico 3 - Precio con regimenes y senales
    fig = _make_price_chart(df_full, states, asset, timeframe, signal_info, state_summary)
    price_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"pc-{timeframe}", config=plotly_config)

    # Conditions
    conditions_html = ""
    for label, info in signal_info["conditions"].items():
        icon = "✅" if info["met"] else "❌"
        met_class = "condition-met" if info["met"] else "condition-not-met"
        detail = info["detail"]
        conditions_html += f"""
        <div class="condition-row {met_class}">
            <span class="condition-icon">{icon}</span>
            <span class="condition-label">{label}</span>
            <span class="condition-detail">{detail}</span>
        </div>"""

    # Regime guide cards
    regime_cards = ""
    for _, row in state_summary.iterrows():
        s = int(row["state"])
        c = REGIME_COLORS[s % len(REGIME_COLORS)]
        desc = row["description"]
        ret = row["mean_return"]
        vol = row["volatility"]
        pct = row["pct_time"]
        dur = row["mean_duration_bars"]
        is_active = "active-regime-card" if s == current_state else ""
        active_tag = "<div class='active-tag'>ACTUAL</div>" if s == current_state else ""
        if "EXPANSION BAJISTA" in desc:
            expl = "Mercado en panico o capitulacion. Alta volatilidad con fuertes caidas. Peligro de liquidaciones en cadena. Evitar compras, esperar senal de agotamiento."
        elif "EXPANSION ALCISTA" in desc:
            expl = "Euforia del mercado. Subida violenta con volumen extremo. Momento de alta riesgo/recompensa. Ideal para tomar ganancias parciales."
        elif "TREND ALCISTA" in desc:
            expl = "Tendencia alcista fuerte y saludable. Volatilidad elevada pero direccion clara. Momento ideal para buscar entradas LONG en retrocesos."
        elif "TREND BAJISTA" in desc:
            expl = "Tendencia bajista con conviccion. Momentum negativo fuerte. Preferir estar en corto o en efectivo. No comprar hasta que cambie la estructura."
        elif "ALCISTA FUERTE" in desc:
            expl = "Tendencia alcista consolidada con buen volumen. Confianza del mercado alta. Momento para operar LONG con confianza."
        elif "BAJISTA FUERTE" in desc:
            expl = "Tendencia bajista consolidada. Presion vendedora sostenida. Evitar compras, buscar oportunidades SHORT."
        elif "ALCISTA SUAVE" in desc:
            expl = "Leve presion compradora. El mercado esta probando direccion alcista pero sin fuerza. Requiere confirmacion adicional."
        elif "BAJISTA SUAVE" in desc:
            expl = "Leve presion vendedora. Mercado debil pero no en caida libre. Esperar confirmacion antes de operar."
        elif "ALCISTA" in desc:
            expl = "Mercado con sesgo alcista y volatilidad controlada. Buen momento para operar LONG con gestion de riesgo moderada."
        elif "BAJISTA" in desc:
            expl = "Mercado con sesgo bajista. Se estan formando maximos y minimos decrecientes. Preferir operaciones SHORT o esperar."
        elif "ACUMULACION" in desc:
            expl = "Mercado lateral con baja volatilidad. Grandes jugadores estan acumulando posiciones. Preparandose para el proximo movimiento grande."
        elif "LATERAL" in desc or "VOLATILIDAD NEUTRA" in desc:
            expl = "Mercado sin direccion clara. Alta incertidumbre. Mejor esperar a que el precio salga del rango antes de operar."
        elif "ALTA VOLATILIDAD" in desc:
            expl = "Volatilidad anormal sin direccion clara. Movimientos bruscos en ambas direcciones. Reducir tamano de posiciones."
        else:
            expl = "Regimen de mercado neutro. Operar con cautela hasta que se defina una direccion."
        arrow_icon = "▲" if ret >= 0 else "▼"
        ret_color = "#089981" if ret >= 0 else "#F23645"
        regime_cards += f"""
        <div class="regime-card {is_active}">
            {active_tag}
            <div class="regime-card-header">
                <span class="regime-card-dot" style="background:{c}"></span>
                <span class="regime-card-name">{desc}</span>
                <span class="regime-card-id">R{s}</span>
            </div>
            <div class="regime-card-metrics">
                <div class="metric">
                    <span class="metric-value" style="color:{ret_color}">{arrow_icon} {ret:+.4f}%</span>
                    <span class="metric-label">Retorno Medio</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{vol:.2f}%</span>
                    <span class="metric-label">Volatilidad</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{pct}%</span>
                    <span class="metric-label">Tiempo en este regimen</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{dur}</span>
                    <span class="metric-label">Duracion media (velas)</span>
                </div>
            </div>
            <div class="regime-card-explanation">{expl}</div>
        </div>"""

    # Regime summary table (empty - removed from dashboard)
    regime_rows = ""

    # ──────────────────────────────────────────────────────────────────────────
    # VERIFICACIÓN HISTÓRICA (Regla de Jaime Merino)

    # ──────────────────────────────────────────────────────────────────────────
    verification = data.verification
    verification_html = ""
    if verification and verification["total_signals"] > 0:
        stats = verification["stats"]
        wr = verification["overall_win_rate"]
        total = verification["total_signals"]
        wins = verification["total_wins"]

        # Color según win rate
        wr_color = "#089981" if wr >= 60 else ("#2962FF" if wr >= 40 else "#F23645")

        # Tarjetas por lado
        side_cards = ""
        for side_name in ["LONG", "SHORT"]:
            s = stats[side_name]
            if s["num_signals"] == 0:
                side_cards += f"""
                <div class="verif-side-card">
                    <div class="verif-side-header" style="color:{'#2ECC40' if side_name == 'LONG' else '#FF4136'}">{side_name}</div>
                    <div class="verif-side-body">
                        <span style="color:#666;font-size:0.75rem;">Sin senales en el periodo</span>
                    </div>
                </div>"""
                continue
            wr_side = s["win_rate"]
            wr_s_color = "#089981" if wr_side >= 60 else ("#2962FF" if wr_side >= 40 else "#F23645")
            avg_ret = s["avg_return"]
            ret_color = "#089981" if avg_ret >= 0 else "#F23645"
            ret_arrow = "▲" if avg_ret >= 0 else "▼"
            avg_bars = s["avg_bars_to_win"]
            bars_str = f"{avg_bars:.1f}" if avg_bars is not None else "-"
            recent_str = ""
            if s["recent_signals"] > 0 and s["recent_win_rate"] is not None:
                r_wr = s["recent_win_rate"]
                r_color = "#089981" if r_wr >= 60 else ("#2962FF" if r_wr >= 40 else "#F23645")
                recent_str = f"<div class='verif-recent'>Ultimos 30d: <b style='color:{r_color}'>{r_wr:.0f}%</b> ({s['recent_wins']}/{s['recent_signals']})</div>"
            side_cards += f"""
            <div class="verif-side-card">
                <div class="verif-side-header" style="color:{'#2ECC40' if side_name == 'LONG' else '#FF4136'}">
                    {side_name}
                    <span class="verif-side-count">{s['num_signals']} senales</span>
                </div>
                <div class="verif-side-body">
                    <div class="verif-side-row">
                        <span>Win Rate</span>
                        <b style="color:{wr_s_color}">{wr_side:.0f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Retorno medio</span>
                        <b style="color:{ret_color}">{ret_arrow} {avg_ret:+.2f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Max favorable medio</span>
                        <b style="color:#2ECC40">{s['avg_max_favorable']:+.2f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Max adverso medio</span>
                        <b style="color:#FF4136">{s['avg_max_adverse']:+.2f}%</b>
                    </div>
                    <div class="verif-side-row">
                        <span>Velas hasta acierto (media)</span>
                        <b>{bars_str}</b>
                    </div>
                    {recent_str}
                </div>
            </div>"""

        # Mejor y peor señal
        best = verification.get("best_signal")
        worst = verification.get("worst_signal")
        extremes_html = ""
        if best or worst:
            extremes_html = '<div class="verif-extremes">'
            if best:
                side_label = "LONG" if best.get("won") else "SHORT"
                b_color = "#089981" if best["final_return"] >= 0 else "#F23645"
                extremes_html += f"""
                <div class="verif-extreme">
                    <span class="verif-extreme-label">🏆 Mejor senal</span>
                    <span>{_format_date(best['entry_date'])}</span>
                    <span style="color:{b_color};font-weight:700;">{best['final_return']:+.2f}%</span>
                </div>"""
            if worst:
                w_color = "#089981" if worst["final_return"] >= 0 else "#F23645"
                extremes_html += f"""
                <div class="verif-extreme">
                    <span class="verif-extreme-label">⚠️ Peor senal</span>
                    <span>{_format_date(worst['entry_date'])}</span>
                    <span style="color:{w_color};font-weight:700;">{worst['final_return']:+.2f}%</span>
                </div>"""
            extremes_html += "</div>"
        verification_html = f"""
        <div class="section" style="margin-top:16px;">
            <div class="section-title">
                📊 Verificacion Historica de Senales
                <span style="font-size:0.65rem;color:#888;font-weight:400;margin-left:8px;">
                    Ventana: {verification['window_str']} ({verification['max_bars']} velas max) · TP: {verification.get('tp_target', 2.0):.1f}%
                </span>
            </div>
            <div class="verif-container">
                <div class="verif-overall">
                    <div class="verif-overall-stat">
                        <span class="verif-overall-value" style="color:{wr_color};">{wr:.1f}%</span>
                        <span class="verif-overall-label">Win Rate Global</span>
                    </div>
                    <div class="verif-overall-stat">
                        <span class="verif-overall-value">{total}</span>
                        <span class="verif-overall-label">Total Senales</span>
                    </div>
                    <div class="verif-overall-stat">
                        <span class="verif-overall-value" style="color:{wr_color};">{wins}/{total}</span>
                        <span class="verif-overall-label">Aciertos / Total</span>
                    </div>
                    <div class="verif-overall-stat">
                        <span class="verif-overall-value">{verification['window_str']}</span>
                        <span class="verif-overall-label">Ventana de Expiracion</span>
                    </div>
                    <div class="verif-overall-stat">
                        <span class="verif-overall-value" style="color:#f39c12;">TP {verification.get('tp_target', 2.0):.1f}%</span>
                        <span class="verif-overall-label">Take-Profit Objetivo</span>
                    </div>
                </div>
                <div class="verif-sides">
                    {side_cards}
                </div>
                {extremes_html}
            </div>
        </div>"""

    # Change summary
    change_summary_html = _build_change_summary(regime_changes, state_summary)

    # Alerts
    alerts_html = ""
    if regime_changes and len(regime_changes) > 0:
        for change in regime_changes:
            arrow = "→"
            from_c = change["from_color"]
            to_c = change["to_color"]
            alerts_html += f"""
            <div class="alert-row">
                <span class="alert-date">{change['date']}</span>
                <span class="alert-regime-from" style="color:{from_c}">{change['from_desc']}</span>
                <span class="alert-arrow">{arrow}</span>
                <span class="alert-regime-to" style="color:{to_c}">{change['to_desc']}</span>
                <span class="alert-duration">{change['duration_velas']}v</span>
            </div>"""

    # Transition matrix
    trans_mat_headers = ""
    trans_mat_rows = ""
    transmat_explanation = ""
    if trans_mat is not None and len(trans_mat) > 0:
        n = len(trans_mat)
        desc_map = {}
        for _, r in state_summary.iterrows():
            desc_map[int(r["state"])] = r["description"].strip("[]")
        for s in range(n):
            c = REGIME_COLORS[s % len(REGIME_COLORS)]
            name = desc_map.get(s, "")
            label = f"R{s} = {name}" if name else f"R{s}"
            trans_mat_headers += f'<th style="color:{c}">{label}</th>'
        for i in range(n):
            c = REGIME_COLORS[i % len(REGIME_COLORS)]
            name = desc_map.get(i, "")
            label = f"R{i} = {name}" if name else f"R{i}"
            cells = f'<td class="trans-mat-label" style="color:{c}">{label}</td>'
            for j in range(n):
                pct = trans_mat[i][j] * 100
                t = pct / 100.0
                if t <= 0.5:
                    t2 = t / 0.5
                    r = int(5 + (0 - 5) * t2)
                    g = int(10 + (180 - 10) * t2)
                    b_val = int(20 + (180 - 20) * t2)
                else:
                    t2 = (t - 0.5) / 0.5
                    r = int(0 + (0 - 0) * t2)
                    g = int(180 + (230 - 180) * t2)
                    b_val = int(180 + (120 - 180) * t2)
                bg = f"rgb({r},{g},{b_val})"
                text_color = "#fff" if pct >= 30 else ("#aaa" if pct >= 10 else "#555")
                is_diag = "trans-mat-diag" if i == j else ""
                cells += f'<td class="{is_diag}" style="background:{bg};color:{text_color}">{pct:.0f}%</td>'
            trans_mat_rows += f"<tr>{cells}</tr>"
        if current_state >= 0 and current_state < n:
            stay_pct = trans_mat[current_state][current_state] * 100
            dest_probs = [(j, trans_mat[current_state][j]) for j in range(n) if j != current_state]
            dest_probs.sort(key=lambda x: -x[1])
            regime_name = regime_desc if regime_desc else f"R{current_state}"
            if stay_pct >= 60:
                persistencia = "Alta persistencia"
                persistencia_desc = f"Hay un <b>{stay_pct:.0f}%</b> de probabilidad de que el mercado <b>permanezca</b> en este regimen. Es un estado estable."
            elif stay_pct >= 35:
                persistencia = "Persistencia moderada"
                persistencia_desc = f"Hay un <b>{stay_pct:.0f}%</b> de probabilidad de que el mercado se <b>mantenga</b> en este regimen. Puede cambiar pronto."
            else:
                persistencia = "Baja persistencia"
                persistencia_desc = f"Solo hay un <b>{stay_pct:.0f}%</b> de probabilidad de permanencia. Este regimen tiende a ser <b>transitorio</b>."
            if dest_probs and dest_probs[0][1] > 0.05:
                next_idx = dest_probs[0][0]
                next_pct = dest_probs[0][1] * 100
                next_desc = state_summary[state_summary["state"] == next_idx]["description"].values[0] if not state_summary[state_summary["state"] == next_idx].empty else f"R{next_idx}"
                next_color = REGIME_COLORS[next_idx % len(REGIME_COLORS)]
                transicion = f"Si cambia, lo mas probable es que pase a <b style='color:{next_color}'>{next_desc}</b> (R{next_idx}) con un <b>{next_pct:.0f}%</b> de probabilidad."
            else:
                transicion = ""
            transmat_explanation = f"""
                <p style="margin-bottom:10px;"><b>Regimen actual:</b> {regime_name} (R{current_state})</p>
                <p style="margin-bottom:10px;"><b>{persistencia}:</b><br>{persistencia_desc}</p>
                <p><b>Transicion mas probable:</b><br>{transicion}</p>
            """
        else:
            transmat_explanation = "<p style='color:#666;'>No hay suficiente informacion para interpretar el regimen actual.</p>"

    # ── Regime Warning Section HTML ──
    regime_warning_section_html = ""
    if signal in ("LONG", "SHORT") and regime_warnings:
        warning_rows = ""
        critical_count = 0
        for w in regime_warnings:
            if w["impact"] == "adverse":
                icon = "🛑"
                impact_label = "ADVERSO"
                row_class = "warning-row-adverse"
                critical_count += 1
            elif w["impact"] == "favorable":
                icon = "✅"
                impact_label = "FAVORABLE"
                row_class = "warning-row-favorable"
            else:
                icon = "➖"
                impact_label = "NEUTRAL"
                row_class = "warning-row-neutral"
            warning_rows += f"""
            <div class="warning-regime-row {row_class}">
                <span class="warning-date">{w['date']}</span>
                <span class="warning-from" style="color:{w['from_color']}">{w['from_desc']}</span>
                <span class="warning-arrow">→</span>
                <span class="warning-to" style="color:{w['to_color']}">{w['to_desc']}</span>
                <span class="warning-impact-badge impact-{w['impact']}">{icon} {impact_label}</span>
            </div>"""

        # Overall status
        if regime_alignment == "adverse":
            overall_icon = "🛑"
            overall_title = "ADVERTENCIA: Regimen adverso al trade"
            overall_msg = "El regimen actual del mercado esta en contra de tu posicion. Considera salir del trade o reducir tamano."
            overall_class = "warning-critical"
            action_color = "#F23645"
        elif regime_alignment == "favorable":
            overall_icon = "✅"
            overall_title = "Regimen favorable al trade"
            overall_msg = "El regimen actual respalda tu direccion. Puedes mantener la posicion con confianza."
            overall_class = "warning-ok"
            action_color = "#089981"
        elif regime_alignment == "neutral":
            if critical_count > 0:
                overall_icon = "⚠️"
                overall_title = "Precaucion: Cambios recientes adversos detectados"
                overall_msg = "Aunque el regimen actual es neutral, hubo cambios adversos recientes. Monitorea de cerca."
                overall_class = "warning-caution"
                action_color = "#2962FF"
            else:
                overall_icon = "➖"
                overall_title = "Regimen neutral"
                overall_msg = "El mercado no muestra direccion clara. Gestiona el riesgo con stop-loss ajustado."
                overall_class = "warning-caution"
                action_color = "#888"
        else:
            overall_icon = ""
            overall_title = ""
            overall_msg = ""
            overall_class = ""
            action_color = "#888"
        regime_warning_section_html = f"""
        <div class="section" style="margin-top:16px;">
            <div class="section-title">
                {overall_icon} Alerta de Regimen para Trade Activo
                <span style="font-size:0.65rem;color:#888;font-weight:400;margin-left:8px;">
                    Evaluacion de regimen actual vs posicion {signal}
                </span>
            </div>
            <div class="regime-warning-container {overall_class}">
                <div class="regime-warning-header">
                    <span class="regime-warning-title">{overall_icon} {overall_title}</span>
                </div>
                <div class="regime-warning-body">
                    <p>{overall_msg}</p>
                    <div class="regime-warning-action" style="border-left-color:{action_color};">
                        <span class="action-label">🎯 Accion sugerida:</span>
                        <span class="action-text" style="color:{action_color};">
                            {"🛑 CONSIDERAR SALIR del trade" if regime_alignment == "adverse" else ("✅ MANTENER trade" if regime_alignment == "favorable" else "⚠️ MONITOREAR de cerca")}
                        </span>
                    </div>
                </div>
                <div class="regime-warning-changes">
                    <div class="warning-subtitle">📊 Ultimos cambios de regimen y su impacto:</div>
                    {warning_rows}
                </div>
            </div>
        </div>"""

    # ── Trade Health Meter ──
    health_meter_html = _build_trade_health_meter(
        states=states,
        state_summary=state_summary,
        regime_warnings=regime_warnings,
        trans_mat=trans_mat,
        current_state=current_state,
        signal=signal,
        regime_alignment=regime_alignment,
        signal_info=signal_info,
        verification=verification,
        df_full=df_full,
    )

    # ── Trailing Stop Comparativa ──
    trailing_verification = data.trailing_verification
    trailing_html = ""
    if trailing_verification and trailing_verification["total_signals"] > 0:
        tv = trailing_verification
        wr_tp = tv["overall_win_rate_tp"]
        wr_ts = tv["overall_win_rate_ts"]
        wr_comb = tv["overall_win_rate_combined"]
        total_sig = tv["total_signals"]
        trail_pct = tv.get("trail_pct", 50.0)
        improvement = wr_comb - wr_tp
        imp_color = "#089981" if improvement > 0 else ("#F23645" if improvement < 0 else "#888")
        imp_arrow = "▲" if improvement > 0 else ("▼" if improvement < 0 else "➖")
        tp_color = "#089981" if wr_tp >= 60 else ("#2962FF" if wr_tp >= 40 else "#F23645")
        ts_color = "#089981" if wr_ts >= 60 else ("#2962FF" if wr_ts >= 40 else "#F23645")
        comb_color = "#089981" if wr_comb >= 60 else ("#2962FF" if wr_comb >= 40 else "#F23645")
        side_cards_html = ""
        for side_name in ["LONG", "SHORT"]:
            s = tv["stats"][side_name]
            if s["num_signals"] == 0:
                side_cards_html += f"""
                <div class="trailing-side-card">
                    <div class="trailing-side-header">{side_name}</div>
                    <div style="color:#666;font-size:0.75rem;padding:12px;">Sin senales</div>
                </div>"""
                continue
            wr_tp_s = s["win_rate_tp"]
            wr_ts_s = s["win_rate_ts"]
            wr_comb_s = s["win_rate_combined"]
            imp_s = wr_comb_s - wr_tp_s
            imp_color_s = "#089981" if imp_s > 0 else ("#F23645" if imp_s < 0 else "#888")

            # Determine best row
            best_wr = max(wr_tp_s, wr_ts_s, wr_comb_s)
            tp_best = " trailing-row-best" if wr_tp_s == best_wr else ""
            ts_best = " trailing-row-best" if wr_ts_s == best_wr else ""
            comb_best = " trailing-row-best" if wr_comb_s == best_wr else ""
            side_cards_html += f"""
            <div class="trailing-side-card">
                <div class="trailing-side-header" style="color:{'#2ECC40' if side_name == 'LONG' else '#FF4136'};">
                    {side_name} <span class="trailing-side-count">{s['num_signals']} senales</span>
                </div>
                <table class="trailing-table">
                    <tr><th>Estrategia</th><th>Win Rate</th><th>Retorno Medio</th></tr>
                    <tr class="trailing-table-row{tp_best}">
                        <td>TP Fijo ({tv['tp_target']:.1f}%)</td>
                        <td style="color:{tp_color};font-weight:700;">{wr_tp_s:.1f}%</td>
                        <td>{s['avg_return_tp']:+.2f}%</td>
                    </tr>
                    <tr class="trailing-table-row{ts_best}">
                        <td>Trailing ({trail_pct:.0f}%)</td>
                        <td style="color:{ts_color};font-weight:700;">{wr_ts_s:.1f}%</td>
                        <td>{s['avg_return_ts']:+.2f}%</td>
                    </tr>
                    <tr class="trailing-table-row{comb_best}">
                        <td>Combinado</td>
                        <td style="color:{comb_color};font-weight:700;">{wr_comb_s:.1f}%</td>
                        <td>{s['avg_return_combined']:+.2f}%</td>
                    </tr>
                </table>
                <div style="color:{imp_color_s};font-size:0.7rem;padding:4px 12px 8px;text-align:right;">
                    Mejora: {imp_s:+.1f}% vs TP Fijo
                </div>
            </div>"""
        trailing_html = f"""
        <div class="section" style="margin-top:16px;">
            <div class="section-title">
                🎯 Trailing Stop Comparativa
                <span style="font-size:0.65rem;color:#888;font-weight:400;margin-left:8px;">
                    Trail {trail_pct:.0f}% · {tv['window_str']} ({tv['max_bars']} velas max) · {total_sig} senales
                </span>
            </div>
            <div class="trailing-container">
                <div class="trailing-overall">
                    <div class="trailing-overall-card">
                        <div class="trailing-overall-label">TP Fijo</div>
                        <div class="trailing-overall-value" style="color:{tp_color};">{wr_tp:.1f}%</div>
                        <div class="trailing-overall-sub">Win Rate</div>
                    </div>
                    <div class="trailing-overall-arrow">→</div>
                    <div class="trailing-overall-card">
                        <div class="trailing-overall-label">Trailing Stop</div>
                        <div class="trailing-overall-value" style="color:{ts_color};">{wr_ts:.1f}%</div>
                        <div class="trailing-overall-sub">Win Rate</div>
                    </div>
                    <div class="trailing-overall-arrow">→</div>
                    <div class="trailing-overall-card" style="border-color:{imp_color};">
                        <div class="trailing-overall-label">Combinado</div>
                        <div class="trailing-overall-value" style="color:{comb_color};">{wr_comb:.1f}%</div>
                        <div class="trailing-overall-sub">Win Rate</div>
                    </div>
                    <div class="trailing-improvement" style="color:{imp_color};">
                        {imp_arrow} {improvement:+.1f}% vs TP Fijo
                    </div>
                </div>
                <div class="trailing-sides">
                    {side_cards_html}
                </div>
                <div class="trailing-note">
                    💡 El trailing stop ({trail_pct:.0f}% de retroceso) se combina con el TP fijo: 
                    la operacion se cierra cuando se activa <b>cualquiera</b> de los dos primero.
                    Esto captura ganancias antes de que el mercado revierta, mejorando el Win Rate.
                </div>
            </div>
        </div>"""

    # Armar HTML interno
    cards_inner = f"""
        <div class="cards-grid">
            <div class="card">
                <div class="card-title">Precio Actual</div>
                <div class="price-value">{_fmt_price(price)}</div>
                <div class="price-change" style="color:{price_color}">{price_arrow} {abs(price_change):.2f}%</div>
                <div class="price-date">{date}</div>
            </div>
            <div class="card">
                <div class="card-title">Regimen HMM Actual</div>
                <div>
                    <span class="regime-indicator" style="background:{regime_color}"></span>
                    <span class="regime-number">R{current_state}</span>
                </div>
                <div class="regime-desc">{regime_desc}</div>
                <div class="regime-stats">
                    <span>&#9201; {regime_duration} velas</span>
                    <span>&#128202; {regime_pct} del tiempo</span>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Senal Actual</div>
                <div class="signal-badge-row">
                    <span class="signal-badge" style="background:{signal_color};color:#fff;">{SIGNAL_LABELS[signal]}</span>
                    {expired_badge_html}
                </div>
                <!-- Regime alignment badge -->
                {_regime_alignment_badge(regime_alignment, signal)}
                <div class="signal-strength-bar">
                    <div class="signal-strength-fill" style="width:{strength}%;background:{signal_color};"></div>
                </div>
                <div class="signal-label">Fuerza de senal: {strength}/100</div>
                {signal_start_html}
                <div class="signal-score-box">
                    <span class="signal-score-label">Score Compuesto</span>
                    <span class="signal-score-value" style="color:{signal_color}">{signal_info.get("signal_score_used", 0):.0f}</span>
                    <span class="signal-score-sep">/</span>
                    <span class="signal-score-threshold">{signal_info.get("score_threshold", SIGNAL_SCORE_THRESHOLD)}</span>
                    <span class="signal-score-check">{'✅' if signal in ('LONG', 'SHORT') else '❌'}</span>
                </div>
                <div class="signal-filter-badge">
                    <span class="filter-icon">⏱</span>
                    <span class="filter-text">Confirmacion: {signal_info.get("min_consecutive_bars", MIN_CONSECUTIVE_BARS)} velas consecutivas</span>
                </div>
                {expiration_html}
                <div class="signal-params">
                    <span>EMA {EMA_FAST}/{EMA_SLOW}</span>
                    <span>ADX &ge; {ADX_THRESHOLD}</span>
                    <span>Score &ge; {SIGNAL_SCORE_THRESHOLD}</span>
                    <span>Release {RELEASE_LOOKBACK}</span>
                </div>
            </div>
        </div>
        <div class="section">
            <div class="section-title">📈 Precio + EMA 55</div>
            <div class="chart-container">{simple_price_chart_html}</div>
        </div>
        <div class="section">
            <div class="section-title">📊 ADX + SQUEEZE MOMENTUM</div>
            <div class="chart-container">{indicators_chart_html}</div>
        </div>
        <div class="section">
            <div class="section-title">📈 Precio con Regimenes y Senales</div>
            <div class="chart-container">{price_chart_html}</div>
        </div>
        <div class="section">
            <div class="section-title">CONDICIONES DEL TRIGGER &mdash; {SIGNAL_LABELS[signal]}</div>
            <div class="conditions-grid">{conditions_html}</div>
            <!-- SCORE BREAKDOWN -->
            {_score_breakdown_html(signal_info, signal)}
        </div>
        <div class="section">
            <div class="section-title">📖 Guia de Regimenes &mdash; ¿Que significa cada estado?</div>
            <div class="regime-guide">{regime_cards}</div>
            <div class="color-legend">
                <div class="legend-title">🎨 Leyenda de Colores</div>
                <div class="legend-items">
                    <div class="legend-item"><span class="legend-swatch" style="background:#2ECC40"></span> Alcista / Trend alcista</div>
                    <div class="legend-item"><span class="legend-swatch" style="background:#3498DB"></span> Acumulacion / Lateral</div>
                    <div class="legend-item"><span class="legend-swatch" style="background:#FF851B"></span> Transicion / Neutral</div>
                    <div class="legend-item"><span class="legend-swatch" style="background:#FF4136"></span> Bajista / Trend bajista</div>
                    <div class="legend-item"><span class="legend-swatch" style="background:#B10DC9"></span> Expansion / Alta volatilidad</div>
                </div>
            </div>
        </div>
        <div class="section" style="margin-top:16px;">
            <div class="section-title">🔄 Dinamica de los Regimenes &mdash; ¿Como cambia el mercado?</div>
            <div class="two-col">
                <div class="table-container" style="overflow-x:auto;">
                    <table class="trans-mat">
                        <thead><tr><th>Desde \\ Hacia</th>{trans_mat_headers}</tr></thead>
                        <tbody>{trans_mat_rows}</tbody>
                    </table>
                    <div style="font-size:0.7rem;color:#666;margin-top:8px;">
                        Cada celda muestra la probabilidad (%) de que el mercado pase de un regimen a otro en la siguiente vela.
                        Los valores altos en la diagonal (=) indican regimenes <b>persistentes</b> que duran muchas velas.
                    </div>
                </div>
                <div>
                    <div class="card" style="height:100%;">
                        <div class="card-title">Interpretacion del Regimen Actual</div>
                        <div style="font-size:0.82rem;color:#ccc;line-height:1.6;">{transmat_explanation}</div>
                    </div>
                </div>
            </div>
        </div>
        {regime_warning_section_html}
        {verification_html}
        {trailing_html}
        {health_meter_html}"""
    return cards_inner


def _build_asset_selector(current_asset: str) -> str:
    """Genera el HTML del selector de activos con dropdown y comando."""
    options_html = ""
    for a in POPULAR_ASSETS:
        selected = "selected" if a.upper() == current_asset.upper() else ""
        options_html += f'<option value="{a}" {selected}>{a}</option>'
    return f"""
    <div class="asset-selector">
        <div class="asset-selector-header">
            <label for="asset-select">Activo:</label>
            <select id="asset-select" onchange="onAssetChange(this.value)">
                {options_html}
            </select>
        </div>
        <div id="cmd-box" class="cmd-box" style="display:none;">
            <span class="cmd-label">Ejecuta este comando en la terminal:</span>
            <div class="cmd-row">
                <code id="cmd-text"></code>
                <button class="copy-btn" onclick="copyCommand()">Copiar</button>
            </div>
        </div>
    </div>"""


def build_multi_tf_dashboard(results: Dict[str, TimeframeData], asset: str) -> str:
    """Genera el HTML completo con pestanas para cambiar entre temporalidades."""
    tf_order = ["1h", "4h", "1d", "1w"]
    available = [tf for tf in tf_order if tf in results]
    if not available:
        return "<html><body><h1>No hay datos disponibles.</h1></body></html>"
    tabs_buttons = ""
    tabs_content = ""
    for i, tf in enumerate(available):
        active_class = "active" if i == 0 else ""
        tf_label = tf.upper()

        # Generar contenido interno para esta temporalidad
        inner = _generate_tf_inner(results[tf], asset, tf)
        tabs_buttons += f'<button class="tf-tab {active_class}" onclick="switchTF(\'{tf}\')">{tf_label}</button>'
        tabs_content += f'<div class="tf-content {active_class}" id="tf-{tf}">{inner}</div>'

    # Usar la fecha del primer timeframe para el header
    first_signal = results[available[0]].signal_info
    date = first_signal["date"]

    # Generar selector de activos
    asset_selector_html = _build_asset_selector(asset)
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HMM Regime Dashboard · {asset} (Multi-Timeframe)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #0f0f13;
    color: #e8e8e8;
    line-height: 1.5;
}}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
/* HEADER */
.header {{
    text-align: center;
    padding: 24px 20px 16px;
    border-bottom: 1px solid #2a2a35;
    margin-bottom: 24px;
}}
.header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
}}
.header h1 span {{
    background: linear-gradient(135deg, #f39c12, #e74c3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.header .subtitle {{
    font-size: 0.82rem;
    color: #888;
    margin-top: 4px;
}}
.header .meta {{
    font-size: 0.78rem;
    color: #666;
    margin-top: 4px;
}}
/* ASSET SELECTOR */
.asset-selector {{
    margin: 14px 0 10px;
    text-align: center;
}}
.asset-selector-header {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
}}
.asset-selector label {{
    font-size: 0.8rem;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.asset-selector select {{
    background: #1a1a24;
    border: 1px solid #2a2a35;
    color: #e8e8e8;
    padding: 8px 32px 8px 14px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23888' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    min-width: 160px;
    transition: border-color 0.2s;
}}
.asset-selector select:hover {{
    border-color: #444;
}}
.asset-selector select:focus {{
    outline: none;
    border-color: rgba(52,152,219,0.4);
    box-shadow: 0 0 0 2px rgba(52,152,219,0.1);
}}
/* COMMAND BOX */
.cmd-box {{
    margin: 12px auto 0;
    max-width: 600px;
    background: #1a1a24;
    border: 1px solid #f39c12;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: left;
}}
.cmd-label {{
    font-size: 0.7rem;
    color: #f39c12;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: block;
    margin-bottom: 8px;
}}
.cmd-row {{
    display: flex;
    align-items: stretch;
    gap: 8px;
}}
.cmd-row code {{
    flex: 1;
    display: block;
    padding: 8px 12px;
    background: #0f0f13;
    border-radius: 6px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.78rem;
    color: #e8e8e8;
    line-height: 1.4;
    white-space: nowrap;
    overflow-x: auto;
}}
.copy-btn {{
    background: #f39c12;
    border: none;
    color: #0f0f13;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: inherit;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    flex-shrink: 0;
}}
.copy-btn:hover {{
    background: #f1c40f;
}}
.copy-btn.copied {{
    background: #2ECC40;
}}
/* TIMEFRAME TABS */
.tabs-bar {{
    display: flex;
    justify-content: center;
    gap: 6px;
    margin: 16px 0 24px;
}}
.tf-tab {{
    background: #1a1a24;
    border: 1px solid #2a2a35;
    color: #888;
    padding: 8px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    transition: all 0.2s;
    font-family: inherit;
}}
.tf-tab:hover {{
    background: #2a2a35;
    color: #ccc;
    border-color: #444;
}}
.tf-tab.active {{
    background: rgba(52,152,219,0.15);
    color: #3498DB;
    border-color: rgba(52,152,219,0.4);
    box-shadow: 0 0 8px rgba(52,152,219,0.1);
}}
.tf-content {{
    display: none;
}}
.tf-content.active {{
    display: block;
}}
/* CARDS GRID */
.cards-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 30px;
}}
.card {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a2a35;
}}
.card-title {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #666;
    margin-bottom: 10px;
}}
/* PRICE CARD */
.price-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
}}
.price-change {{
    font-size: 0.9rem;
    margin-top: 4px;
    font-weight: 600;
}}
.price-date {{ font-size: 0.75rem; color: #888; margin-top: 4px; }}
/* REGIME CARD */
.regime-indicator {{
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}}
.regime-number {{
    font-size: 2rem;
    font-weight: 700;
    vertical-align: middle;
}}
.regime-desc {{
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 6px;
}}
.regime-stats {{
    display: flex;
    gap: 16px;
    margin-top: 10px;
    font-size: 0.75rem;
    color: #888;
}}
.regime-stats span {{ display: flex; align-items: center; gap: 4px; }}
/* SIGNAL CARD */
.signal-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
}}
.signal-strength-bar {{
    width: 100%;
    height: 6px;
    background: #2a2a35;
    border-radius: 3px;
    margin-top: 10px;
    overflow: hidden;
}}
.signal-strength-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}}
.signal-label {{ font-size: 0.7rem; color: #666; margin-top: 10px; }}
/* SIGNAL SCORE BOX */
.signal-score-box {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    padding: 8px 12px;
    background: #13131a;
    border-radius: 8px;
    border: 1px solid #2a2a35;
}}
.signal-score-label {{
    font-size: 0.6rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: auto;
}}
.signal-score-value {{
    font-size: 1.2rem;
    font-weight: 700;
}}
.signal-score-sep {{
    font-size: 0.8rem;
    color: #555;
}}
.signal-score-threshold {{
    font-size: 0.8rem;
    color: #888;
    font-weight: 600;
}}
.signal-score-check {{
    font-size: 0.9rem;
}}
/* SIGNAL FILTER BADGE */
.signal-filter-badge {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
    padding: 4px 10px;
    background: rgba(52,152,219,0.1);
    border: 1px solid rgba(52,152,219,0.2);
    border-radius: 6px;
}}
.filter-icon {{
    font-size: 0.85rem;
}}
.filter-text {{
    font-size: 0.68rem;
    color: #7fb8e0;
    font-weight: 500;
}}
/* SIGNAL START BOX */
.signal-start-box {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(243,156,18,0.08);
    border: 1px solid rgba(243,156,18,0.2);
    border-radius: 8px;
    border-left: 3px solid #f39c12;
}}
.signal-start-icon {{
    font-size: 1rem;
    flex-shrink: 0;
}}
.signal-start-info {{
    display: flex;
    flex-direction: column;
    gap: 1px;
}}
.signal-start-label {{
    font-size: 0.6rem;
    color: #f39c12;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}}
.signal-start-date {{
    font-size: 0.82rem;
    color: #e8e8e8;
    font-weight: 600;
}}
.signal-start-bars {{
    font-size: 0.65rem;
    color: #999;
}}
/* EXPIRATION (Jaime Merino Rule) */
.expired-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.5px;
    background: rgba(255,65,54,0.2);
    color: #FF4136;
    border: 1px solid rgba(255,65,54,0.3);
    margin-left: 8px;
    vertical-align: middle;
}}
.expiration-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.7rem;
    background: rgba(46,204,64,0.1);
    border: 1px solid rgba(46,204,64,0.2);
    margin-left: 8px;
    vertical-align: middle;
}}
.expiration-container {{
    margin-top: 10px;
    padding: 10px 12px;
    background: #13131a;
    border-radius: 8px;
    border-left: 3px solid rgba(243,156,18,0.4);
}}
.expiration-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}
.expiration-label {{
    font-size: 0.65rem;
    color: #f39c12;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.expiration-window {{
    font-size: 0.7rem;
    color: #aaa;
    background: #2a2a35;
    padding: 2px 8px;
    border-radius: 4px;
}}
.expiration-bar-track {{
    width: 100%;
    height: 6px;
    background: #2a2a35;
    border-radius: 3px;
    overflow: hidden;
}}
.expiration-bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}}
.expiration-details {{
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 0.65rem;
    color: #888;
}}
.expired-text {{
    color: #FF4136 !important;
    font-weight: 600;
}}
.signal-params {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }}
.signal-params span {{
    background: #2a2a35;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    color: #aaa;
}}
/* CONDITIONS */
.section {{ margin-bottom: 30px; }}
.section-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a35;
}}
.conditions-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}}
.condition-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    border: 1px solid transparent;
}}
.condition-met {{
    background: rgba(46, 204, 64, 0.08);
    border-color: rgba(46, 204, 64, 0.2);
}}
.condition-not-met {{
    background: rgba(255, 65, 54, 0.06);
    border-color: rgba(255, 65, 54, 0.15);
    opacity: 0.6;
}}
.condition-icon {{ font-size: 1rem; flex-shrink: 0; }}
        /* Score Breakdown */
        .sb-container {{
            margin-top: 14px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .sb-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px;
        }}
        .sb-title {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.3px; }}
        .sb-total {{
            font-size: 0.75rem; font-weight: 600;
            padding: 3px 10px; border-radius: 12px;
        }}
        .sb-body {{ display: flex; flex-direction: column; gap: 5px; }}
        .sb-row {{
            display: flex; align-items: center; gap: 8px;
            padding: 3px 0;
        }}
        .sb-label {{
            font-size: 0.7rem; color: #aaa; min-width: 140px;
            flex-shrink: 0;
        }}
        .sb-bar-track {{
            flex: 1; height: 8px; border-radius: 4px;
            background: rgba(255,255,255,0.06);
            overflow: hidden;
        }}
        .sb-bar-fill {{
            height: 100%; border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .sb-value {{
            font-size: 0.7rem; font-weight: 600;
            min-width: 30px; text-align: right;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        .sb-threshold-bar {{
            display: flex; align-items: center; gap: 8px;
            padding: 6px 0; border-top: 1px solid rgba(255,255,255,0.06);
            margin-top: 6px;
        }}
        .sb-threshold-label {{
            font-size: 0.7rem; font-weight: 600; color: #ccc;
            min-width: 140px; flex-shrink: 0;
        }}
        .sb-score-met, .sb-score-not-met {{
            margin-top: 10px; padding: 6px 10px;
            border-radius: 6px; font-size: 0.75rem; font-weight: 600;
            text-align: center;
        }}
        .sb-score-met {{ background: rgba(8,153,129,0.12); }}
        .sb-score-not-met {{ background: rgba(242,54,69,0.12); }}
.condition-label {{ flex-shrink: 0; color: #ccc; }}
.condition-detail {{ font-size: 0.7rem; color: #777; margin-left: auto; text-align: right; }}
/* CHARTS */
.chart-container {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #2a2a35;
    margin-bottom: 20px;
    overflow: hidden;
}}
.chart-container .js-plotly-plot {{ width: 100% !important; }}
.chart-container .plot-container {{ width: 100% !important; }}
/* REGIME TABLE */
.table-container {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #2a2a35;
    overflow-x: auto;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
th {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid #2a2a35;
    color: #888;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 10px 12px;
    border-bottom: 1px solid #222;
    color: #ccc;
}}
tr.active-regime td {{
    background: rgba(52, 152, 219, 0.1);
    font-weight: 600;
}}
.regime-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
/* REGIME GUIDE CARDS */
.regime-guide {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}}
.regime-card {{
    background: #1a1a24;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #2a2a35;
    position: relative;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.regime-card:hover {{
    border-color: #444;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}
.regime-card.active-regime-card {{
    border-color: #3498DB;
    box-shadow: 0 0 0 1px rgba(52,152,219,0.3), 0 2px 12px rgba(52,152,219,0.15);
}}
.active-tag {{
    position: absolute;
    top: -1px;
    right: 16px;
    background: #3498DB;
    color: #fff;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 0 0 6px 6px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
.regime-card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}}
.regime-card-dot {{
    display: inline-block;
    width: 16px; height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.regime-card-name {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
}}
.regime-card-id {{
    font-size: 0.7rem;
    color: #666;
    background: #2a2a35;
    padding: 1px 8px;
    border-radius: 4px;
    margin-left: auto;
}}
.regime-card-metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
}}
.metric {{
    background: #13131a;
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
}}
.metric-value {{
    font-size: 0.9rem;
    font-weight: 700;
    color: #e8e8e8;
    display: block;
}}
.metric-label {{
    font-size: 0.65rem;
    color: #777;
    display: block;
    margin-top: 2px;
}}
.regime-card-explanation {{
    font-size: 0.75rem;
    color: #999;
    line-height: 1.5;
    padding: 8px 10px;
    background: #13131a;
    border-radius: 6px;
    border-left: 3px solid #333;
}}
/* REGIME ALIGNMENT BADGE */
.regime-alignment-badge {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    border-left: 4px solid;
}}
.alignment-favorable {{
    background: rgba(46,204,64,0.08);
    border-left-color: #2ECC40;
}}
.alignment-adverse {{
    background: rgba(255,65,54,0.08);
    border-left-color: #FF4136;
}}
.alignment-neutral {{
    background: rgba(243,156,18,0.08);
    border-left-color: #f39c12;
}}
.alignment-icon {{
    font-size: 1rem;
}}
.alignment-text {{
    font-size: 0.75rem;
    font-weight: 700;
    color: #e8e8e8;
}}
.alignment-sub {{
    font-size: 0.65rem;
    color: #999;
}}
.alignment-favorable .alignment-text {{ color: #2ECC40; }}
.alignment-adverse .alignment-text {{ color: #FF4136; }}
.alignment-neutral .alignment-text {{ color: #f39c12; }}
/* REGIME WARNING SECTION */
.regime-warning-container {{
    background: #1a1a24;
    border-radius: 12px;
    border: 1px solid #2a2a35;
    overflow: hidden;
    margin-top: 8px;
}}
.regime-warning-container.warning-critical {{
    border-color: rgba(255,65,54,0.4);
    box-shadow: 0 0 12px rgba(255,65,54,0.08);
}}
.regime-warning-container.warning-ok {{
    border-color: rgba(46,204,64,0.3);
}}
.regime-warning-container.warning-caution {{
    border-color: rgba(243,156,18,0.3);
}}
.regime-warning-header {{
    padding: 14px 18px 10px;
    border-bottom: 1px solid #2a2a35;
}}
.regime-warning-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: #e8e8e8;
}}
.warning-critical .regime-warning-title {{ color: #FF4136; }}
.warning-ok .regime-warning-title {{ color: #2ECC40; }}
.warning-caution .regime-warning-title {{ color: #f39c12; }}
.regime-warning-body {{
    padding: 14px 18px;
}}
.regime-warning-body p {{
    font-size: 0.8rem;
    color: #ccc;
    line-height: 1.5;
    margin-bottom: 12px;
}}
.regime-warning-action {{
    padding: 10px 14px;
    background: #13131a;
    border-radius: 8px;
    border-left: 4px solid;
    display: flex;
    flex-direction: column;
    gap: 4px;
}}
.action-label {{
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.action-text {{
    font-size: 0.85rem;
    font-weight: 700;
}}
.regime-warning-changes {{
    padding: 12px 18px 16px;
    border-top: 1px solid #2a2a35;
}}
.warning-subtitle {{
    font-size: 0.72rem;
    color: #aaa;
    margin-bottom: 10px;
    font-weight: 600;
}}
.warning-regime-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 0.75rem;
}}
.warning-regime-row:last-child {{ margin-bottom: 0; }}
.warning-row-adverse {{
    background: rgba(255,65,54,0.06);
}}
.warning-row-favorable {{
    background: rgba(46,204,64,0.06);
}}
.warning-row-neutral {{
    background: rgba(243,156,18,0.04);
}}
.warning-date {{
    color: #666;
    font-size: 0.65rem;
    min-width: 110px;
}}
.warning-from, .warning-to {{
    font-weight: 600;
    font-size: 0.72rem;
}}
.warning-arrow {{
    color: #555;
}}
.warning-impact-badge {{
    margin-left: auto;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
}}
.impact-adverse {{
    background: rgba(255,65,54,0.15);
    color: #FF4136;
}}
.impact-favorable {{
    background: rgba(46,204,64,0.15);
    color: #2ECC40;
}}
.impact-neutral {{
    background: rgba(243,156,18,0.12);
    color: #f39c12;
}}
/* TRANSITION MATRIX */
table.trans-mat {{
    font-size: 0.78rem;
}}
table.trans-mat th {{
    text-align: center;
    padding: 6px 10px;
    font-size: 0.7rem;
}}
table.trans-mat td {{
    text-align: center;
    padding: 6px 10px;
    font-size: 0.78rem;
    border-radius: 4px;
    font-weight: 600;
}}
table.trans-mat td.trans-mat-label {{
    font-weight: 700;
    border-right: 1px solid #333;
    text-align: right;
}}
table.trans-mat td.trans-mat-diag {{
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1);
}}
/* CHANGE SUMMARY */
.change-summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}}
.summary-stat {{
    background: #1a1a24;
    border: 1px solid #2a2a35;
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
}}
.summary-stat-value {{
    display: block;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8e8e8;
    margin-bottom: 4px;
}}
.summary-stat-label {{
    display: block;
    font-size: 0.65rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
@media (max-width: 768px) {{
    .change-summary {{ grid-template-columns: repeat(2, 1fr); }}
}}
/* REGIME CHANGE ALERTS */
.alerts-container {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 8px 16px;
    border: 1px solid #2a2a35;
    max-height: 340px;
    overflow-y: auto;
}}
.alerts-container::-webkit-scrollbar {{
    width: 6px;
}}
.alerts-container::-webkit-scrollbar-track {{
    background: #0f0f13;
}}
.alerts-container::-webkit-scrollbar-thumb {{
    background: #2a2a35;
    border-radius: 3px;
}}
.alert-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-bottom: 1px solid #222;
    font-size: 0.8rem;
    transition: background 0.15s;
}}
.alert-row:last-child {{
    border-bottom: none;
}}
.alert-row:hover {{
    background: rgba(255,255,255,0.03);
}}
.alert-date {{
    font-size: 0.7rem;
    color: #666;
    min-width: 90px;
    flex-shrink: 0;
}}
.alert-regime-from,
.alert-regime-to {{
    font-weight: 600;
    font-size: 0.78rem;
}}
.alert-arrow {{
    color: #555;
    font-size: 1rem;
    flex-shrink: 0;
}}
.alert-duration {{
    font-size: 0.65rem;
    color: #888;
    background: #2a2a35;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: auto;
    flex-shrink: 0;
}}
/* COLOR LEGEND */
.color-legend {{
    background: #1a1a24;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #2a2a35;
    margin-top: 8px;
}}
.legend-title {{
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 10px;
    font-weight: 600;
}}
.legend-items {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 20px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #bbb;
}}
.legend-swatch {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.1);
}}
/* VERIFICACIÓN HISTÓRICA (Regla de Jaime Merino) */
.verif-container {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a2a35;
}}
.verif-overall {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}}
.verif-overall-stat {{
    text-align: center;
    padding: 14px 10px;
    background: #13131a;
    border-radius: 10px;
    border: 1px solid #2a2a35;
}}
.verif-overall-value {{
    display: block;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8e8e8;
    line-height: 1.2;
}}
.verif-overall-label {{
    display: block;
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 6px;
}}
.verif-sides {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
}}
.verif-side-card {{
    background: #13131a;
    border-radius: 10px;
    border: 1px solid #2a2a35;
    overflow: hidden;
}}
.verif-side-header {{
    padding: 10px 14px;
    font-size: 0.9rem;
    font-weight: 700;
    border-bottom: 1px solid #2a2a35;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.verif-side-count {{
    font-size: 0.7rem;
    font-weight: 400;
    color: #888;
    background: #2a2a35;
    padding: 2px 10px;
    border-radius: 10px;
}}
.verif-side-body {{
    padding: 12px 14px;
}}
.verif-side-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 0.78rem;
    color: #aaa;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.verif-side-row:last-child {{
    border-bottom: none;
}}
.verif-recent {{
    margin-top: 10px;
    padding: 8px 10px;
    background: rgba(243,156,18,0.08);
    border-radius: 6px;
    font-size: 0.72rem;
    color: #ccc;
    border-left: 3px solid #f39c12;
}}
.verif-extremes {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}}
.verif-extreme {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: #13131a;
    border-radius: 8px;
    border: 1px solid #2a2a35;
    font-size: 0.75rem;
    color: #aaa;
}}
.verif-extreme-label {{
    font-weight: 600;
    color: #e8e8e8;
    font-size: 0.7rem;
    white-space: nowrap;
}}
@media (max-width: 900px) {{
    .verif-overall {{ grid-template-columns: repeat(3, 1fr); }}
    .verif-sides {{ grid-template-columns: 1fr; }}
    .verif-extremes {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 600px) {{
    .verif-overall {{ grid-template-columns: repeat(2, 1fr); }}
}}
/* DISCLAIMER */
.disclaimer {{
    text-align: center;
    padding: 20px;
    font-size: 0.7rem;
    color: #555;
    border-top: 1px solid #2a2a35;
    margin-top: 30px;
}}
@media (max-width: 768px) {{
    .cards-grid {{ grid-template-columns: 1fr; }}
    .conditions-grid {{ grid-template-columns: 1fr; }}
    .two-col {{ grid-template-columns: 1fr; }}
}}
/* TRADE HEALTH METER */
.health-meter-container {{
    background: #1a1a24;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a2a35;
}}
.health-verdict {{
    background: #13131a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
}}
.health-verdict-row {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.health-verdict-icon {{
    font-size: 2.2rem;
    flex-shrink: 0;
}}
.health-verdict-info {{
    flex: 1;
}}
.health-verdict-label {{
    font-size: 1.3rem;
    font-weight: 700;
    display: block;
    letter-spacing: 0.5px;
}}
.health-verdict-desc {{
    font-size: 0.78rem;
    color: #999;
    display: block;
    margin-top: 2px;
}}
.health-score-ring {{
    width: 72px;
    height: 72px;
    border: 3px solid;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    background: #0f0f13;
}}
.health-score-value {{
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1;
}}
.health-score-label {{
    font-size: 0.55rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 1px;
}}
.health-bar-container {{
    margin: 12px 0;
}}
.health-bar-track {{
    width: 100%;
    height: 8px;
    background: #2a2a35;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}}
.health-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}}
.health-bar-labels {{
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
    font-size: 0.65rem;
    font-weight: 500;
}}
.health-action {{
    background: rgba(243,156,18,0.08);
    border: 1px solid rgba(243,156,18,0.2);
    border-left: 3px solid;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.health-action-label {{
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    flex-shrink: 0;
}}
.health-action-text {{
    font-size: 0.85rem;
    font-weight: 700;
}}
.health-meters-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}}
.health-meter-card {{
    background: #13131a;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #2a2a35;
    text-align: center;
}}
.health-meter-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin-bottom: 10px;
}}
.health-meter-icon {{
    font-size: 1.1rem;
}}
.health-meter-title {{
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}}
.health-meter-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
}}
.health-meter-sub {{
    font-size: 0.65rem;
    color: #777;
    margin-bottom: 6px;
}}
.health-meter-status {{
    font-size: 0.72rem;
    font-weight: 600;
}}
.health-alerts {{
    margin-top: 14px;
    padding: 12px 16px;
    background: rgba(243,156,18,0.06);
    border: 1px solid rgba(243,156,18,0.15);
    border-radius: 8px;
}}
.health-alert-row {{
    font-size: 0.75rem;
    color: #ccc;
    padding: 3px 0;
}}
.health-alert-row:first-child {{
    padding-top: 0;
}}
.health-alert-row:last-child {{
    padding-bottom: 0;
}}
@media (max-width: 768px) {{
    .health-meters-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .health-verdict-row {{
        flex-direction: column;
        text-align: center;
    }}
    .health-score-ring {{
        width: 60px;
        height: 60px;
    }}
    .health-score-value {{
        font-size: 1.1rem;
    }}
}}
/* ────────────────────────────────────────────── */
/* TRAILING STOP COMPARATIVA */
/* ────────────────────────────────────────────── */
.trailing-container {{
    background: var(--bg-card, #131b30);
    border-radius: 10px;
    padding: 18px;
    margin-top: 8px;
}}
.trailing-overall {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}}
.trailing-overall-card {{
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    min-width: 140px;
    flex: 1;
}}
.trailing-overall-label {{
    font-size: 0.72rem;
    color: #888;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.trailing-overall-value {{
    font-size: 1.4rem;
    font-weight: 800;
}}
.trailing-overall-sub {{
    font-size: 0.7rem;
    color: #666;
    margin-top: 3px;
}}
.trailing-overall-arrow {{
    font-size: 1.5rem;
    color: #444;
    font-weight: 300;
}}
.trailing-improvement {{
    font-size: 0.75rem;
    margin-top: 4px;
    font-weight: 600;
}}
.trailing-sides {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}}
.trailing-side-card {{
    flex: 1;
    min-width: 280px;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    padding: 12px 14px;
}}
.trailing-side-header {{
    font-size: 0.85rem;
    margin-bottom: 8px;
}}
.trailing-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
}}
.trailing-table th {{
    text-align: left;
    color: #666;
    font-weight: 400;
    padding: 4px 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.trailing-table td {{
    padding: 5px 6px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.trailing-row-best td {{
    background: rgba(46,204,64,0.06);
    border-bottom: none;
}}
.trailing-note {{
    margin-top: 12px;
    padding: 10px 14px;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    border-left: 3px solid #3498DB;
    font-size: 0.75rem;
    color: #aaa;
    line-height: 1.5;
}}
    /* Range slider y rangeselector */
    .rangeselector button {{
        font-size: 10px !important;
        padding: 2px 8px !important;
    }}
    .rangeslider-slidebox {{
        fill: rgba(46,204,64,0.08) !important;
    }}
    .rangeslider-mask-min,
    .rangeslider-mask-max {{
        fill: rgba(0,0,0,0.4) !important;
    }}
    .rangeslider-handle {{
        fill: #089981 !important;
        stroke: #089981 !important;
    }}
    /* Spikelines */
    .spikeline {{
        stroke: rgba(255,255,255,0.15) !important;
        stroke-dasharray: 2 2 !important;
    }}
    /* Crosshair cursor en hover */
    .hoverlayer .spikeline {{
        stroke: rgba(255,255,255,0.25) !important;
    }}
    .hoverlayer .spikeline-0 {{
        stroke: rgba(255,255,255,0.25) !important;
    }}
</style>
</head>
<body>
<div class="container">
    <!-- HEADER -->
    <div class="header">
        <h1><span>TradingLatino</span> · HMM Regime Dashboard</h1>
        <div class="subtitle">Regimenes de Mercado con Hidden Markov Model + Estrategia TradingLatino</div>
        <div class="meta">{asset} · Multi-Timeframe · Actualizado: {date}</div>
        <!-- ASSET SELECTOR -->
        {asset_selector_html}
    </div>
    <!-- TIMEFRAME TABS -->
    <div class="tabs-bar">{tabs_buttons}</div>
    <!-- TAB CONTENT -->
    {tabs_content}
    <!-- DISCLAIMER -->
    <div class="disclaimer">
        ⚠️ Esta herramienta es solo para fines educativos e informativos. No constituye asesoria financiera. 
        TradingLatino HMM Dashboard v2.0 (Simplificado)
    </div>
</div>
<script>
function switchTF(tf) {{
    // Actualizar tabs
    document.querySelectorAll('.tf-tab').forEach(function(tab) {{
        tab.classList.remove('active');
    }});
    document.querySelectorAll('.tf-content').forEach(function(content) {{
        content.classList.remove('active');
    }});
    // Activar seleccionado
    var tab = document.querySelector('.tf-tab[onclick*="' + tf + '"]');
    if (tab) tab.classList.add('active');
    var content = document.getElementById('tf-' + tf);
    if (content) {{
        content.classList.add('active');
        // Re-dibujar graficos Plotly que puedan estar ocultos
        var plots = content.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(p) {{
            var gd = p.querySelector('.plot-container');
            if (gd && gd._fullLayout) {{
                Plotly.Plots.resize(p);
            }}
        }});
    }}
}}
function onAssetChange(asset) {{
    var cmdBox = document.getElementById('cmd-box');
    var cmdText = document.getElementById('cmd-text');
    var scriptName = '{SCRIPT_NAME}';
    cmdText.textContent = 'cd C:/FreeBuff ; python ' + scriptName + ' --asset ' + asset;
    cmdBox.style.display = 'block';
}}
function copyCommand() {{
    var cmdText = document.getElementById('cmd-text');
    var btn = document.querySelector('.copy-btn');
    if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(cmdText.textContent).then(function() {{
            btn.textContent = 'Copiado!';
            btn.classList.add('copied');
            setTimeout(function() {{
                btn.textContent = 'Copiar';
                btn.classList.remove('copied');
            }}, 2000);
        }});
    }} else {{
        // Fallback
        var textarea = document.createElement('textarea');
        textarea.value = cmdText.textContent;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btn.textContent = 'Copiado!';
        btn.classList.add('copied');
        setTimeout(function() {{
            btn.textContent = 'Copiar';
            btn.classList.remove('copied');
        }}, 2000);
    }}    }}
    // TradingView-style synchronized zoom/pan: todas las graficas se mueven juntas
    var _chartSyncTimer = null;
    var _chartSyncing = false;

    function _getTF(id) {{
        if (id.indexOf('spc-') === 0 || id.indexOf('ic-') === 0 || id.indexOf('pc-') === 0) {{
            return id.substring(4);
        }}
        return null;
    }}

    function _syncCharts(activeGd, range) {{
        if (_chartSyncing) return;
        if (_chartSyncTimer) clearTimeout(_chartSyncTimer);
        _chartSyncTimer = setTimeout(function() {{
            var tf = _getTF(activeGd.id);
            if (!tf) return;
            _chartSyncing = true;
            var ids = ['spc-' + tf, 'ic-' + tf, 'pc-' + tf];
            ids.forEach(function(id) {{
                if (id === activeGd.id) return;
                var other = document.getElementById(id);
                if (other && other._fullLayout) {{
                    Plotly.relayout(other, {{'xaxis.range': range}});
                }}
            }});
            _chartSyncing = false;
        }}, 80);
    }}

    // TradingView-style scroll zoom: prevenir scroll de pagina y permitir zoom con rueda
    document.addEventListener('wheel', function(e) {{
        var target = e.target;
        var container = target.closest('.plot-container');
        if (container) {{
            e.preventDefault();
        }}
    }}, {{passive: false}});

    // Attach synchronized zoom/pan a todas las graficas
    setTimeout(function() {{
        var allCharts = document.querySelectorAll('[id^="spc-"], [id^="ic-"], [id^="pc-"]');
        allCharts.forEach(function(gd) {{
            if (gd.on) {{
                gd.on('plotly_relayout', function(eventdata) {{
                    var range = null;
                    if (eventdata['xaxis.range'] && Array.isArray(eventdata['xaxis.range'])) {{
                        range = eventdata['xaxis.range'];
                    }} else if (eventdata['xaxis.range[0]'] !== undefined && eventdata['xaxis.range[1]'] !== undefined) {{
                        range = [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']];
                    }}
                    if (range && range[0] !== undefined && range[1] !== undefined) {{
                        _syncCharts(gd, range);
                    }}
                    // Sync autorange (double-click reset) across all charts
                    // IMPORTANT: check _chartSyncing to prevent infinite loop
                    if (!_chartSyncing && eventdata['xaxis.autorange']) {{
                        var tfAutorange = _getTF(gd.id);
                        if (tfAutorange) {{
                            _chartSyncing = true;
                            var idsAuto = ['spc-' + tfAutorange, 'ic-' + tfAutorange, 'pc-' + tfAutorange];
                            idsAuto.forEach(function(id) {{
                                if (id === gd.id) return;
                                var other = document.getElementById(id);
                                if (other && other._fullLayout) {{
                                    Plotly.relayout(other, {{'xaxis.autorange': true}});
                                }}
                            }});
                            _chartSyncing = false;
                        }}
                    }}
                }});
            }}
        }});
    }}, 500);
</script>
</body>
</html>"""
    return html

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS PLOTLY

# ──────────────────────────────────────────────────────────────────────────────
REGIME_COLORS_PLOTLY = ["#089981", "#3498DB", "#FF851B", "#F23645", "#B10DC9", "#F012BE"]


def _make_price_chart(df: pd.DataFrame, states: np.ndarray, asset: str, timeframe: str, signal_info: Dict, state_summary: pd.DataFrame = None) -> go.Figure:
    """Grafico de precio con velas japonesas + regimes (fondo) + senales LONG/SHORT, estilo TradingView.
    Zoom fluido igual que el grafico de Squeeze + ADX."""
    from plotly.subplots import make_subplots
    import numpy as np

    # Colores de volumen segun direccion de la vela
    vol_colors = ["#089981" if cl >= op else "#F23645" for cl, op in zip(df["Close"], df["Open"])]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.45],
    )

    # -- Fondo por regimen --
    if len(states) == len(df):
        unique_states = [s for s in np.unique(states) if s >= 0]
        for s in unique_states:
            mask = states == s
            change_points = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
            starts = np.concatenate([[0], change_points])
            ends = np.concatenate([change_points, [len(mask)]])
            for st, en in zip(starts, ends):
                if mask[st]:
                    fig.add_vrect(
                        x0=df.index[st], x1=df.index[min(en, len(df)-1)],
                        fillcolor=REGIME_COLORS_PLOTLY[s % len(REGIME_COLORS_PLOTLY)],
                        opacity=0.06, layer="below", line_width=0,
                        row=1, col=1,
                    )

    # --- Precios formateados al estilo espanol (ej: 73.014,37$) ---
    fmt_open  = [_fmt_price(float(v)) for v in df["Open"]]
    fmt_high  = [_fmt_price(float(v)) for v in df["High"]]
    fmt_low   = [_fmt_price(float(v)) for v in df["Low"]]
    fmt_close = [_fmt_price(float(v)) for v in df["Close"]]
    fmt_vol_full = [f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df["Volume"]]
    # Texto hover pre-formateado (evita problemas con customdata en candlestick)
    hover_texts = [
        f"{idx.strftime('%d-%m-%Y')}<br><b>Open:</b> {o}<br><b>High:</b> {h}<br><b>Low:</b> {l}<br><b>Close:</b> {c}"
        for idx, o, h, l, c in zip(df.index, fmt_open, fmt_high, fmt_low, fmt_close)
    ]

    # --- Fila 1: Velas Japonesas ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="",
        line=dict(width=1),
        increasing=dict(line=dict(color="#089981", width=1), fillcolor="#089981"),
        decreasing=dict(line=dict(color="#F23645", width=1), fillcolor="#F23645"),
        whiskerwidth=0.2,
        showlegend=False,
        text=hover_texts,
        hoverinfo="text",
    ), row=1, col=1)

    # -- Senal LONG --
    if "signal_long" in df.columns:
        long_idx = df["signal_long"] & (df["signal_long"].shift(1) == False)
        if long_idx.any():
            long_prices_formatted = [_fmt_price(float(v)) for v in df.loc[long_idx, "Low"]]
            fig.add_trace(go.Scatter(
                x=df.index[long_idx],
                y=df.loc[long_idx, "Low"] * 0.995,
                mode="markers", name="LONG",
                marker=dict(symbol="triangle-up", size=12, color="#089981", line=dict(width=1, color="white")),
                customdata=np.stack([long_prices_formatted], axis=-1),
                hovertemplate="<b>LONG</b><br>Precio: %{customdata[0]}<extra></extra>",
            ), row=1, col=1)

    # -- Senal SHORT --
    if "signal_short" in df.columns:
        short_idx = df["signal_short"] & (df["signal_short"].shift(1) == False)
        if short_idx.any():
            short_prices_formatted = [_fmt_price(float(v)) for v in df.loc[short_idx, "High"]]
            fig.add_trace(go.Scatter(
                x=df.index[short_idx],
                y=df.loc[short_idx, "High"] * 1.005,
                mode="markers", name="SHORT",
                marker=dict(symbol="triangle-down", size=12, color="#F23645", line=dict(width=1, color="white")),
                customdata=np.stack([short_prices_formatted], axis=-1),
                hovertemplate="<b>SHORT</b><br>Precio: %{customdata[0]}<extra></extra>",
            ), row=1, col=1)

    # --- Fila 2: Volumen (escalado al %% del maximo para visibilidad en TF pequenos) ---
    vol_max = df["Volume"].max()
    vol_scaled = df["Volume"] / vol_max * 100 if vol_max > 0 else df["Volume"]
    fig.add_trace(go.Bar(
        x=df.index, y=vol_scaled,
        name="Vol",
        marker=dict(color=vol_colors, line=dict(width=0.5, color="#1a1a2e")),
        opacity=1.0,
        hovertemplate="Vol: %{customdata[0]}<extra></extra>",
        customdata=np.stack([fmt_vol_full], axis=-1),
    ), row=2, col=1)

    # --- Layout unificado (identico al grafico Precio + EMA 55) ---
    fig.update_layout(
        template="plotly_dark",
        height=500,
        dragmode="pan",
        margin=dict(l=40, r=20, t=10, b=30),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1a24", bordercolor="#444",
            font=dict(family="monospace", size=12, color="#FFD700"),
            namelength=-1,
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",

        # Eje X superior (oculto, el inferior es el compartido)
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   rangeslider=dict(visible=False)),

        # Eje X inferior (compartido) — con rangeslider + botones
        xaxis2=dict(
            showgrid=False, zeroline=False,
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
                bgcolor="rgba(255,255,255,0.05)",
                activecolor="rgba(46,204,64,0.3)",
                font=dict(color="white", size=10),
                x=0, y=1.02,
            ),
            showspikes=True, spikethickness=1, spikedash="solid",
            spikemode="across", spikesnap="cursor", spikecolor="#888",
        ),

        # Ejes Y
        yaxis=dict(showgrid=True, gridcolor="#2a2a35", zeroline=False, tickformat=".0f"),
        yaxis2=dict(showgrid=True, gridcolor="#2a2a35", zeroline=False, title="", tickformat=".0f"),
    )
    return fig

def _make_simple_price_chart(df: pd.DataFrame, asset: str, timeframe: str) -> go.Figure:
    """Gráfico Precio + EMA55 con volumen integrado debajo, estilo TradingView."""
    from plotly.subplots import make_subplots
    import numpy as np

    # Colores de volumen según dirección de la vela
    vol_colors = ["#089981" if cl >= op else "#F23645" for cl, op in zip(df["Close"], df["Open"])]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.45],
    )

    # ── Precios formateados al estilo espanol (ej: 73.014,37$) ──
    fmt_open  = [_fmt_price(float(v)) for v in df["Open"]]
    fmt_high  = [_fmt_price(float(v)) for v in df["High"]]
    fmt_low   = [_fmt_price(float(v)) for v in df["Low"]]
    fmt_close = [_fmt_price(float(v)) for v in df["Close"]]
    fmt_vol_full = [f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df["Volume"]]
    # Texto hover pre-formateado (evita problemas con customdata en candlestick)
    hover_texts = [
        f"{idx.strftime('%d-%m-%Y')}<br><b>Open:</b> {o}<br><b>High:</b> {h}<br><b>Low:</b> {l}<br><b>Close:</b> {c}"
        for idx, o, h, l, c in zip(df.index, fmt_open, fmt_high, fmt_low, fmt_close)
    ]

    # ── Fila 1: Velas Japonesas ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="",
        line=dict(width=1.5),
        increasing=dict(line=dict(color="#089981", width=2.0), fillcolor="#089981"),
        decreasing=dict(line=dict(color="#F23645", width=2.0), fillcolor="#F23645"),
        whiskerwidth=0.7,
        showlegend=False,
        text=hover_texts,
        hoverinfo="text",
    ), row=1, col=1)

    # ── EMA 55 (blanco) ──
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_slow"],
            mode="lines", name="EMA 55",
            line=dict(color="white", width=1.2),
            hoverinfo="skip",
        ), row=1, col=1)

    # ── Fila 2: Volumen (escalado al %% del maximo para visibilidad en TF pequenos) ---
    vol_max = df["Volume"].max()
    vol_scaled = df["Volume"] / vol_max * 100 if vol_max > 0 else df["Volume"]
    fig.add_trace(go.Bar(
        x=df.index, y=vol_scaled,
        name="Vol",
        marker=dict(color=vol_colors, line=dict(width=0.5, color="#1a1a2e")),
        opacity=1.0,
        hovertemplate="Vol: %{customdata[0]}<extra></extra>",
        customdata=np.stack([fmt_vol_full], axis=-1),
    ), row=2, col=1)

    # ── Layout unificado ──
    fig.update_layout(
        template="plotly_dark",
        height=500,
        dragmode="pan",
        margin=dict(l=40, r=20, t=10, b=30),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1a24", bordercolor="#444",
            font=dict(family="monospace", size=12, color="#FFD700"),
            namelength=-1,
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        # Eje X superior (oculto, el inferior es el compartido)
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   rangeslider=dict(visible=False)),

        # Eje X inferior (compartido) — con rangeslider + botones
        xaxis2=dict(
            showgrid=False, zeroline=False,
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
                bgcolor="rgba(255,255,255,0.05)",
                activecolor="rgba(46,204,64,0.3)",
                font=dict(color="white", size=10),
                x=0, y=1.02,
            ),
            showspikes=True, spikethickness=1, spikedash="solid",
            spikemode="across", spikesnap="cursor", spikecolor="#888",
        ),

        # Ejes Y
        yaxis=dict(showgrid=True, gridcolor="#2a2a35", zeroline=False, tickformat=".0f"),
        yaxis2=dict(showgrid=True, gridcolor="#2a2a35", zeroline=False, title="", tickformat=".0f"),
    )
    return fig



def _make_indicators_chart(df: pd.DataFrame, states: np.ndarray, asset: str, timeframe: str, signal_info: Dict) -> go.Figure:
    """Gráfico Squeeze Momentum (LazyBear) con ADX superpuesto (eje secundario)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    x_data = df.index
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # ══════════════════════════════════════
    # SQUEEZE MOMENTUM — Histograma degradado
    # ══════════════════════════════════════
    if "smi_hist" in df.columns:
        smi_values = df["smi_hist"].values
        # Calcular max absoluto del dataset COMPLETO para fijar escala Y (TradingView-style)
        max_abs_smi = float(np.nanmax(np.abs(smi_values)))
        if max_abs_smi <= 0:
            max_abs_smi = 1.0

        # LazyBear EXACTO: histograma estandar Pine Script
        # Colores: lime (val>0 + inc), green (val>0 + dec), red (val<0 + dec), maroon (val<0 + inc)
        # Sin ancho variable, sin opacidad, sin degradados - como el original
        bar_colors = []
        prev_v = 0.0
        for v in smi_values:
            if v != v:  # NaN
                bar_colors.append("rgba(0,0,0,0)")
                prev_v = v
                continue
            if v >= 0:
                bar_colors.append("#00FF00" if v >= prev_v else "#008000")  # lime if inc, green if dec
            else:
                bar_colors.append("#FF0000" if v < prev_v else "#800000")  # red if dec, maroon if inc
            prev_v = v
        
        fig.add_trace(go.Bar(
            x=x_data, y=smi_values,
            name="SMI Momentum",
            marker=dict(
                color=bar_colors,
                line_width=0,
            ),
            hovertemplate="Momentum: %{y:.2f}<extra></extra>",
        ), secondary_y=False)

        # Zero line
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))

        # Highlight squeeze zones
        if "squeeze_on" in df.columns:
            squeeze_mask = df["squeeze_on"].values
            in_squeeze = False
            squeeze_starts = []
            squeeze_ends = []
            for i in range(len(squeeze_mask)):
                if squeeze_mask[i] and not in_squeeze:
                    squeeze_starts.append(i)
                    in_squeeze = True
                elif not squeeze_mask[i] and in_squeeze:
                    squeeze_ends.append(i)
                    in_squeeze = False
            if in_squeeze:
                squeeze_ends.append(len(squeeze_mask) - 1)
            for s, e in zip(squeeze_starts, squeeze_ends):
                if s < e and e - s > 1:
                    fig.add_vrect(x0=x_data[s], x1=x_data[e], fillcolor="rgba(255,152,0,0.06)", layer="below", line_width=0)

    # ══════════════════════════════════════
    # ADX superpuesto (eje secundario)
    # ══════════════════════════════════════
    if "adx" in df.columns:
        adx_offset = df["adx"] - ADX_THRESHOLD
        fig.add_trace(go.Scatter(
            x=x_data, y=adx_offset,
            name="ADX",
            line=dict(color="#FFFFFF", width=2.0),
            customdata=df["adx"].values,
            hovertemplate="ADX: %{customdata:.1f}<extra></extra>",
        ), secondary_y=True)

        # Zero line (ADX_THRESHOLD alineado con cero del Squeeze)
        fig.add_trace(go.Scatter(
            x=[x_data[0], x_data[-1]],
            y=[0, 0],
            name=f"Umbral ADX {ADX_THRESHOLD}",
            line=dict(color="rgba(255,215,0,0.6)", width=1.5),
            showlegend=True,
            hoverinfo="skip",
        ), secondary_y=True)

    # ── Layout ──
    fig.update_layout(
        template="plotly_dark",
        height=200,
        dragmode="pan",
        margin=dict(l=40, r=20, t=10, b=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1a24", bordercolor="#444",
            font=dict(family="monospace", size=11, color="#FFD700"),
            namelength=-1,
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="relative",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        rangeslider=dict(visible=False),
        showspikes=True, spikethickness=1, spikedash="solid",
        spikemode="across", spikesnap="cursor", spikecolor="#888",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#2a2a35", gridwidth=0.5,
        zeroline=False, title="Momentum",
        autorange=False,
        range=[-max_abs_smi * 1.15, max_abs_smi * 1.15],
        secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False,
        title="ADX", range=[-25, 75],
        secondary_y=True,
    )
    return fig

# ─────────────────────────────────────────────────────────────
# MAIN — Punto de entrada para ejecución directa

# ─────────────────────────────────────────────────────────────

def _auto_backup(max_backups: int = 10) -> None:
    """Crea un backup automatico del script con timestamp."""
    import shutil, datetime, glob, os

    script_path = __file__ if __name__ != '__main__' else sys.argv[0]
    if not script_path or script_path == '':
        script_path = 'tradinglatino_hmm_clean.py'

    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'tradinglatino_hmm_clean_{date}.bak'
    
    try:
        shutil.copy2(script_path, backup_name)
        print(f"  [BACKUP] Creado: {backup_name}")
    except Exception as e:
        print(f"  [BACKUP] Error al crear backup: {e}")
        return

    # Limpiar backups viejos (mantener solo los ultimos N)
    try:
        backups = sorted(glob.glob('tradinglatino_hmm_clean_*.bak'))
        while len(backups) > max_backups:
            old = backups.pop(0)
            os.remove(old)
            print(f"  [BACKUP] Eliminado backup antiguo: {old}")
    except Exception:
        pass


def main():
    """Ejecuta el pipeline completo: descarga -> indicadores -> HMM -> dashboard -> navegador."""
    # Backup automatico del script antes de ejecutar
    _auto_backup()
    print("=" * 60)
    print("DASHBOARD HMM - TradingLatino")
    print("=" * 60)
    print(f"Activo: {ASSET}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Output: {OUTPUT_HTML}")
    print("-" * 60)

    results: Dict[str, TimeframeData] = {}

    for tf in TIMEFRAMES:
        print(f"\nProcesando {ASSET} [{tf}]...")
        try:
            # 1) Descargar datos
            t0 = time.time()
            df = load_data(ASSET, tf)
            if df is None or len(df) < 100:
                print(f"  Datos insuficientes para {tf}, saltando.")
                continue
            print(f"  Datos descargados ({len(df)} velas, {time.time()-t0:.1f}s)")

            # 2) Calcular indicadores
            t0 = time.time()
            df = compute_all_indicators(df)
            print(f"  Indicadores calculados ({time.time()-t0:.1f}s)")

            # -- Re-aplicar threshold y consecutive bars optimizados por timeframe --
            tf_th = SIGNAL_THRESHOLDS.get(tf, SIGNAL_SCORE_THRESHOLD)
            tf_cons = MIN_CONSECUTIVE_BY_TF.get(tf, MIN_CONSECUTIVE_BARS)
            df["signal_raw_long"] = df["signal_score_long"] >= tf_th
            df["signal_raw_short"] = df["signal_score_short"] >= tf_th
            df["signal_long"] = _consecutive_bars_filter(df["signal_raw_long"], tf_cons)
            df["signal_short"] = _consecutive_bars_filter(df["signal_raw_short"], tf_cons)

            # 3) HMM - features + entrenamiento
            t0 = time.time()
            features_df = build_hmm_features(df)
            _, states, state_summary, _, trans_mat = fit_hmm(features_df)
            n_states = state_summary["state"].nunique()
            print(f"  HMM entrenado: {n_states} estados ({time.time()-t0:.1f}s)")

            # -- FILTRO MAESTRO DE REGIMEN HMM (reemplaza la suma de pesos) --
            # El regimen determina la direccion PERMITIDA:
            #   alcista → solo LONG, bajista → solo SHORT, neutral → ambas
            # Esto elimina falsos positivos en direccion contraria al mercado
            df = apply_regime_filter(df, state_summary)

            # -- Detectar senales precursoras de cambios de tendencia --
            df = compute_precursor_signals(df)

            # 4) Senal actual
            signal_info = compute_signal(df, timeframe=tf)
            print(f"  Senal: {signal_info.get('signal', 'N/A'):5s}  "
                  f"Fuerza: {signal_info.get('strength', 0):.0f}%  "
                  f"Precio: ${signal_info.get('price', 0):.2f}")

            # 5) Verificacion historica
            verification = verify_signals_historically(df, tf)
            if verification and verification["total_signals"] > 0:
                print(f"  VERIFICACION: "
                      f"LONG {verification['stats']['LONG']['win_rate']:.1f}% "
                      f"({verification['stats']['LONG']['num_signals']} senales) | "
                      f"SHORT {verification['stats']['SHORT']['win_rate']:.1f}% "
                      f"({verification['stats']['SHORT']['num_signals']} senales) | "
                      f"GLOBAL {verification['overall_win_rate']:.1f}% "
                      f"({verification['total_signals']} senales)")
            else:
                print(f"  VERIFICACION: Sin senales en el periodo")

            # -- TRAILING STOP VERIFICATION (usa valores optimizados por TF) --
            trailing_verification = verify_with_trailing_stop(
                df, tf, TRAILING_STOP_PCT_OPT.get(tf, TRAILING_STOP_PCT.get(tf, 50.0))
            )
            if trailing_verification and trailing_verification["total_signals"] > 0:
                print(f"  TRAILING: "
                      f"TP-Fijo {trailing_verification['overall_win_rate_tp']:.1f}% | "
                      f"Trail {trailing_verification['overall_win_rate_ts']:.1f}% | "
                      f"Comb {trailing_verification['overall_win_rate_combined']:.1f}%")

            # 6) Cambios de regimen
            regime_changes = _detect_regime_changes(states, df.index, state_summary, max_alerts=15)

            # 7) Construir TimeframeData
            results[tf] = TimeframeData(
                df_full=df,
                states_full=states,
                state_summary=state_summary,
                trans_mat=trans_mat,
                signal_info=signal_info,
                regime_changes=regime_changes,
                verification=verification,
                trailing_verification=trailing_verification,
            )
            print(f"  Listo.")
        except Exception as e:
            print(f"  Error procesando {tf}: {e}")
            continue

    if not results:
        print("\nNo se pudo procesar ningun timeframe.")
        sys.exit(1)

    # Generar dashboard HTML
    print(f"\n{'='*60}")
    print("Generando dashboard HTML...")
    t0 = time.time()
    html_content = build_multi_tf_dashboard(results, ASSET)
    print(f"  Dashboard generado ({time.time()-t0:.1f}s)")

    # Guardar a disco
    output_path = Path(OUTPUT_HTML.format(ASSET=ASSET))
    output_path.write_text(html_content, encoding="utf-8")
    print(f"  Dashboard guardado en: {output_path.resolve()}")

    # -- ALERTAS AUTOMATICAS CON DETECCION DE CAMBIOS --
    # Cargar estado anterior para detectar transiciones
    prev_state = {}
    state_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE.format(ASSET=ASSET))
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, "r") as sf:
                prev_state = json.load(sf)
        except:
            prev_state = {}
    
    # Estado actual - registrar senales de TODOS los timeframes
    current_state = {}
    for tf_key in TIMEFRAMES:
        if tf_key in results:
            sig = results[tf_key].signal_info.get("signal", "FLAT")
            price = results[tf_key].signal_info.get("price", 0)
            strength = results[tf_key].signal_info.get("strength", 0)
            current_state[f"{tf_key}_signal"] = sig
            current_state[f"{tf_key}_price"] = price
            current_state[f"{tf_key}_strength"] = strength
            current_state[f"{tf_key}_score_long"] = results[tf_key].signal_info.get("signal_score_long", 0)
            current_state[f"{tf_key}_score_short"] = results[tf_key].signal_info.get("signal_score_short", 0)
        else:
            current_state[f"{tf_key}_signal"] = "N/A"
            current_state[f"{tf_key}_price"] = 0
    
    alertas = []
    
    # Alertas por cada timeframe: detectar cambios de senal
    for tf_key in TIMEFRAMES:
        if tf_key not in results:
            continue
        curr_sig = current_state.get(f"{tf_key}_signal", "FLAT")
        prev_sig = prev_state.get(f"{tf_key}_signal", "N/A")
        curr_price_val = current_state.get(f"{tf_key}_price", 0)
        curr_strength = current_state.get(f"{tf_key}_strength", 0)
        tf_label = tf_key.upper()

        # Primera ejecucion: no hay estado previo -> enviar alerta si hay senal
        if prev_sig == "N/A" and curr_sig in ("LONG", "SHORT"):
            direction_emoji = "🟢" if curr_sig == "LONG" else "🔴"
            msg = (f"{direction_emoji} {ASSET} [{tf_label}] SENAL INICIAL {curr_sig}!\n"
            f"Precio: {_fmt_price(curr_price_val)}\n"
            f"Fuerza: {curr_strength}%")
            alertas.append(msg)
        
        # Cambio a LONG
        elif curr_sig == "LONG" and prev_sig not in ("LONG", "N/A"):
            msg = (f"🟢 {ASSET} [{tf_label}] CAMBIO a LONG!\n"
            f"Precio: {_fmt_price(curr_price_val)}\n"
            f"Fuerza: {curr_strength}%")
            alertas.append(msg)
        
        # Cambio a SHORT
        elif curr_sig == "SHORT" and prev_sig not in ("SHORT", "N/A"):
            msg = (f"🔴 {ASSET} [{tf_label}] CAMBIO a SHORT!\n"
            f"Precio: {_fmt_price(curr_price_val)}\n"
            f"Fuerza: {curr_strength}%")
            alertas.append(msg)
    
    # Guardar estado actual para la proxima ejecucion
    os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
    with open(state_file_path, "w") as sf:
        json.dump(current_state, sf)
    print(f"\n[DEBUG] ENABLE_TELEGRAM={ENABLE_TELEGRAM} TOKEN={'OK' if TELEGRAM_BOT_TOKEN else 'VACIO'} CHAT={'OK' if TELEGRAM_CHAT_ID else 'VACIO'} ALERTAS={len(alertas)}")
    if alertas:
        for a in alertas:
            try:
                print(f"  {a}")
            except UnicodeEncodeError:
                clean = a.encode('ascii', 'ignore').decode('ascii')
                print(f"  {clean}")
        if ENABLE_TELEGRAM:
            sent = _send_telegram_alerts_batch(ASSET, alertas)
            if sent:
                print(f"  [Telegram] Alertas enviadas ({len(alertas)} alertas)")
            else:
                print(f"  [Telegram] Fallo al enviar alertas")
    print(f"\n{'='*60}")
    if ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        _send_telegram_alert(f"\U0001f916 <b>TradingLatino HMM - {ASSET}</b>\n{chr(61) * 20}\n\U0001f7e2 Bot activo\n\U000023f3 Proxima ejecucion en 2h\n{chr(61) * 20}\n<i>Enviado por TradingLatino HMM Bot</i>")
        print(f"  [Telegram] Heartbeat enviado")
    if OPEN_BROWSER:
        print("Abriendo en el navegador...")
        webbrowser.open(str(output_path.resolve()))
    print(f"{'='*60}")
    print("Listo!")
    print(f"{'='*60}")



if __name__ == "__main__":
    # Parse arguments from command line
    import sys as _sys
    _args_list = _sys.argv[1:]
    for _i, _arg in enumerate(_args_list, 1):
        if _arg == "--asset" and _i < len(_args_list):
            ASSET = _args_list[_i]
            print(f"Activo seleccionado: {ASSET}")
    if "--headless" in _args_list or os.environ.get("CI") == "true":
        OPEN_BROWSER = False
        print("Modo headless: no se abrira el navegador")
    main()
