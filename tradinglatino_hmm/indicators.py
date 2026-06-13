#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
INDICADORES TÉCNICOS · TradingLatino HMM Regime Dashboard
================================================================================
Funciones de indicadores técnicos: EMAs, RSI, ADX, Squeeze Momentum, etc.
================================================================================
"""
import numpy as np
import pandas as pd
from typing import Optional

from tradinglatino_hmm.config import (
    EMA_FAST, EMA_SLOW, ADX_THRESHOLD, RELEASE_LOOKBACK,
    SIGNAL_SCORE_THRESHOLD, MIN_CONSECUTIVE_BARS,
    W_BULL_BIAS, W_SQUEEZE_OFF, W_SQUEEZE_REL,
    W_SMI_HIST, W_SMI_DELTA, W_ADX_THRESH, W_ADX_DELTA, BONUS_ADX,
    W_EMA_DEV, W_RSI, W_VOLUME, W_ATR_ROC,
    RSI_LENGTH, VOL_LOOKBACK, ATR_ROC_PERIODS,
    BB_LENGTH, BB_STD, KC_LENGTH, KC_MULT,
    ADX_LENGTH, ATR_LENGTH,
)


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
    ema_dev_pct = df.get("ema_deviation_pct", pd.Series(0, index=df.index))
    ema_dev_short = (ema_dev_pct < -3).astype(float) * W_EMA_DEV
    ema_dev_long  = (ema_dev_pct > 3).astype(float) * W_EMA_DEV

    # Bonus adicional: desviación extrema (>8% = capitulación/euforia)
    ema_dev_extreme_short = ((ema_dev_pct < -8).astype(float) * 5).clip(upper=5)
    ema_dev_extreme_long  = ((ema_dev_pct > 8).astype(float) * 5).clip(upper=5)

    # 2) RSI14
    rsi_val = df.get("rsi14", pd.Series(50, index=df.index))
    rsi_short = (rsi_val < 45).astype(float) * W_RSI
    rsi_long  = (rsi_val > 55).astype(float) * W_RSI
    rsi_extreme_long  = ((rsi_val < 25).astype(float) * 4).clip(upper=4)
    rsi_extreme_short = ((rsi_val > 80).astype(float) * 4).clip(upper=4)

    # 3) Volumen relativo a su media
    vol_ratio = df.get("vol_ratio", pd.Series(1, index=df.index))
    vol_conf_short = ((vol_ratio > 1.5) & (ema_dev_pct < 0)).astype(float) * W_VOLUME
    vol_conf_long  = ((vol_ratio > 1.5) & (ema_dev_pct > 0)).astype(float) * W_VOLUME
    vol_extreme_short = ((vol_ratio > 3).astype(float) * 4).clip(upper=4)
    vol_extreme_long  = ((vol_ratio > 3).astype(float) * 4).clip(upper=4)

    # 4) ATR Rate of Change (expansión de volatilidad)
    atr_roc = df.get("atr_roc", pd.Series(0, index=df.index))
    atr_roc_short = ((atr_roc > 0.2) & (ema_dev_pct < 0)).astype(float) * W_ATR_ROC
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
        + vol_conf_short + vol_extreme_short
        + atr_roc_short
    ).round(1)
    return df


def _consecutive_bars_filter(series: pd.Series, min_bars: int = MIN_CONSECUTIVE_BARS) -> pd.Series:
    """
    Filtro de confirmación TEMPORAL vectorizado.
    Una señal solo se activa si ha estado presente por al menos `min_bars`
    velas consecutivas. Esto elimina falsos positivos aislados.
    """
    if min_bars <= 1:
        return series
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
