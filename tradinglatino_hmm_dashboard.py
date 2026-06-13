#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRADINGLATINO HMM DASHBOARD
================================================================================
Dashboard interactivo HTML para analizar regímenes de mercado con HMM aplicados
a una versión cuantificable de la estrategia pública de TradingLatino (Jaime Merino).

Soporta BTC-USD y XRP-USD por defecto, con temporalidades 1H, 4H, 1D y 1W.

Dependencias:
    pip install pandas numpy plotly hmmlearn yfinance

Uso:
    python tradinglatino_hmm_dashboard.py
================================================================================
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
import warnings
import webbrowser
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN INICIAL  (editable directamente aquí)
# ──────────────────────────────────────────────────────────────────────────────

ASSETS: List[str] = ["BTC-USD", "XRP-USD"]
TIMEFRAMES: List[str] = ["1h", "4h", "1d", "1wk"]  # permitir 1 o varias

USE_CSV: bool = False
CSV_PATHS: Dict[Tuple[str, str], str] = {}  # opcional: {(asset, timeframe): "ruta.csv"}

OUTPUT_HTML: str = "tradinglatino_hmm_dashboard.html"
OPEN_BROWSER: bool = True

RANDOM_STATE: int = 42
MIN_TRADES_FILTER: int = 30
HMM_STATE_RANGE: List[int] = [2, 3, 4, 5]
HMM_COVARIANCE_TYPE: str = "diag"
FEATURE_WINDOW: int = 20
COMMISSION_PCT: float = 0.0005
SLIPPAGE_PCT: float = 0.0005
INITIAL_CAPITAL: float = 100000.0

# ── Modo de grid ──
# Si USE_FULL_GRID = False, usa la grilla reducida (~108 combos válidos)
# Si USE_FULL_GRID = True, usa la grilla completa (~7,776 combos, ~5,184 válidos)
USE_FULL_GRID: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS POR DEFECTO PARA LA ESTRATEGIA
# ──────────────────────────────────────────────────────────────────────────────

# Dirección
EMA_FAST_DEFAULT: int = 10
EMA_SLOW_DEFAULT: int = 55

# Squeeze Momentum
BB_LENGTH: int = 20
BB_STD: float = 2.0
KC_LENGTH: int = 20
KC_MULT: float = 1.5

# ADX
ADX_LENGTH: int = 14
ADX_THRESHOLD_DEFAULT: float = 23.0

# Volumen / Volume Profile
VP_LOOKBACK_DEFAULT: int = 75
VP_BINS: int = 24

# ATR
ATR_LENGTH: int = 14
ATR_STOP_MULT_DEFAULT: float = 2.0
RR_TARGET_DEFAULT: float = 2.0

# Salida temprana
EARLY_EXIT_SMI_BARS: int = 2

# Release lookback (velas desde squeeze_on hasta squeeze_off)
RELEASE_LOOKBACK_DEFAULT: int = 3

# Filtro de volumen
USE_VOLUME_FILTER_DEFAULT: bool = True

# ── Timeframe-adaptive config ──
# KC multiplier por TF: más ancho en timeframes largos para que squeeze_on ocurra
KC_MULT_BY_TF: Dict[str, float] = {
    "1h": 1.5, "4h": 2.0, "1d": 2.0, "1wk": 3.5,
}
# Mínimo de trades por TF (diario/semanal tienen menos velas y menos squeezes)
MIN_TRADES_BY_TF: Dict[str, int] = {
    "1h": 30, "4h": 25, "1d": 15, "1wk": 5,
}

# ──────────────────────────────────────────────────────────────────────────────
# GRILLA DE OPTIMIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

PARAM_GRID_FULL: Dict[str, List[Any]] = {
    "ema_fast": [8, 10, 12],
    "ema_slow": [50, 55, 60],
    "adx_threshold": [20, 23, 25, 30],
    "release_lookback": [1, 3, 5],
    "atr_stop_mult": [1.5, 2.0, 2.5, 3.0],
    "rr_target": [1.5, 2.0, 3.0],
    "vp_lookback": [50, 75, 100],
    "use_volume_filter": [True, False],
}

# Grilla reducida (rápida) por defecto ~108 combinaciones válidas
PARAM_GRID: Dict[str, List[Any]] = (
    PARAM_GRID_FULL if USE_FULL_GRID else {
        "ema_fast": [8, 10, 12],
        "ema_slow": [50, 55, 60],
        "adx_threshold": [20, 25],
        "release_lookback": [3],
        "atr_stop_mult": [1.5, 2.0, 2.5],
        "rr_target": [2.0],
        "vp_lookback": [75],
        "use_volume_filter": [True, False],
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN DE DEPENDENCIAS
# ──────────────────────────────────────────────────────────────────────────────

_MISSING_DEPS: List[str] = []

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
    print("=" * 72)
    print("  ERROR: FALTAN DEPENDENCIAS")
    print("=" * 72)
    print(f"\n  Instala los siguientes paquetes:\n")
    print(f"      pip install {' '.join(_MISSING_DEPS)}\n")
    print("  O usa:\n")
    print(f"      pip install pandas numpy plotly hmmlearn yfinance\n")
    print("=" * 72)
    sys.exit(1)

# --- Importar compute_hybrid_alert desde tradinglatino_hmm_clean.py ---
try:
    import importlib.util as _hybrid_imp
    _HYBRID_MODULE_PATH = Path(__file__).resolve().parent / "tradinglatino_hmm_clean.py"
    _HYBRID_SPEC = _hybrid_imp.spec_from_file_location("hmm_hybrid", _HYBRID_MODULE_PATH)
    if _HYBRID_SPEC is not None:
        _HYBRID_MODULE = _hybrid_imp.module_from_spec(_HYBRID_SPEC)
        _HYBRID_SPEC.loader.exec_module(_HYBRID_MODULE)
        compute_hybrid_alert = _HYBRID_MODULE.compute_hybrid_alert
        compute_precursor_signals = _HYBRID_MODULE.compute_precursor_signals
        _compute_signal_scores = _HYBRID_MODULE._compute_signal_scores
    else:
        compute_hybrid_alert = None
        compute_precursor_signals = None
        _compute_signal_scores = None
except Exception as _hybrid_err:
    print(f"  [WARN] No se pudo importar compute_hybrid_alert: {_hybrid_err}")
    compute_hybrid_alert = None
    compute_precursor_signals = None
    _compute_signal_scores = None

# ──────────────────────────────────────────────────────────────────────────────
# TIPOS / DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    asset: str = ""
    timeframe: str = ""
    entry_date: pd.Timestamp = pd.Timestamp(0)
    exit_date: pd.Timestamp = pd.Timestamp(0)
    side: str = ""  # "LONG" | "SHORT"
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_in_trade: int = 0
    regime_at_entry: int = -1
    exit_reason: str = ""


@dataclass
class StrategyParams:
    ema_fast: int = EMA_FAST_DEFAULT
    ema_slow: int = EMA_SLOW_DEFAULT
    adx_threshold: float = ADX_THRESHOLD_DEFAULT
    release_lookback: int = RELEASE_LOOKBACK_DEFAULT
    atr_stop_mult: float = ATR_STOP_MULT_DEFAULT
    rr_target: float = RR_TARGET_DEFAULT
    vp_lookback: int = VP_LOOKBACK_DEFAULT
    use_volume_filter: bool = USE_VOLUME_FILTER_DEFAULT


@dataclass
class BacktestResult:
    params: StrategyParams = field(default_factory=StrategyParams)
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HMMResult:
    n_states: int = 0
    model: Any = None
    states: np.ndarray = field(default_factory=lambda: np.array([]))
    state_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    bic: float = 0.0
    log_likelihood: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 1) CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────


def _check_min_data(df: pd.DataFrame, asset: str, timeframe: str, min_required: int = 200) -> bool:
    if df is None or df.empty:
        print(f"  [WARN] {asset} @ {timeframe}: sin datos.")
        return False
    if len(df) < min_required:
        print(f"  [WARN] {asset} @ {timeframe}: solo {len(df)} velas (mínimo {min_required}).")
        return False
    return True


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas de yfinance a formato estándar:
    Open, High, Low, Close, Adj Close, Volume.
    Maneja MultiIndex y minúsculas.
    """
    if df is None or df.empty:
        return df
    # Aplanar MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Renombrar columnas a formato estándar
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
    # Asegurar columnas necesarias
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            similar = [c for c in df.columns if c.lower().strip() == col.lower()]
            if similar:
                df = df.rename(columns={similar[0]: col})
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def _try_download_yf(asset: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Intenta descargar datos con yf.download().
    Suprime stderr para ocultar mensajes internos de yfinance como 'Failed download:'.
    Retorna None si falla.
    """
    # Probar diferentes combinaciones de parámetros porque algunas
    # versiones de yfinance no soportan ciertos kwargs como multi_level_index
    param_sets = [
        {"auto_adjust": False, "multi_level_index": False},
        {"auto_adjust": False},
        {},
    ]
    for i, kwargs in enumerate(param_sets):
        try:
            # Silenciar stderr para evitar ruido de yfinance ('Failed download:')
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                df = yf.download(asset, period=period, interval=interval,
                                 progress=False, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        # Pequeña pausa entre intentos para evitar rate limiting
        if i < len(param_sets) - 1:
            time.sleep(0.3)
    return None

def _try_download_ticker(asset: str, period: str, interval: str,
                          retries: int = 2) -> Optional[pd.DataFrame]:
    """
    Intenta descargar datos con yf.Ticker().history().
    Reintenta con pequeña pausa para evitar rate limiting.
    Retorna None si falla.
    """
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(asset)
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                df = ticker.history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(0.5)
    return None


def load_data(
    asset: str,
    timeframe: str,
    use_csv: bool = False,
    csv_paths: Optional[Dict[Tuple[str, str], str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Carga datos para un activo y timeframe.
    Soporta CSV o descarga vía yfinance (con múltiples métodos de fallback).
    """
    if use_csv and csv_paths and (asset, timeframe) in csv_paths:
        path = csv_paths[(asset, timeframe)]
        print(f"  [CSV] Cargando {asset} @ {timeframe} desde {path} ...")
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # Normalizar nombres de columnas
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower().replace(" ", "_")
                if col_lower in ("open", "high", "low", "close", "volume", "adj_close"):
                    rename_map[col] = col_lower.capitalize() if col_lower != "adj_close" else "Adj Close"
                elif col_lower == "adj close":
                    rename_map[col] = "Adj Close"
            df = df.rename(columns=rename_map)
            # Asegurar columnas estándar
            for col, alt in [("Open", "open"), ("High", "high"), ("Low", "low"),
                             ("Close", "close"), ("Volume", "volume")]:
                if col not in df.columns and alt in df.columns:
                    df = df.rename(columns={alt: col})
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            if _check_min_data(df, asset, timeframe):
                return df
            return None
        except Exception as e:
            print(f"  [ERROR] Leyendo CSV {path}: {e}")
            return None

    print(f"  [YFINANCE] Descargando {asset} @ {timeframe} ...")

    # ── Estrategia de descarga (múltiples métodos) ──
    # 1) Primer intento: yf.download()
    # 2) Fallback: yf.Ticker().history()
    # 3) Para 4h: si falla 60m, probar descargar 1h y resamplear
    # 4) Para intradía si todo falla: probar 1d y resamplear (último recurso)

    df = None

    # ── Intento 1: yf.download() ──
    if timeframe == "1h":
        print(f"    [1/4] yf.download() - 60m...")
        df = _try_download_yf(asset, "730d", "60m")
    elif timeframe == "4h":
        print(f"    [1/4] yf.download() - 60m (resamplear a 4H)...")
        df_60m = _try_download_yf(asset, "730d", "60m")
        if df_60m is not None:
            df_60m = _normalize_columns(df_60m)
            df = resample_ohlcv(df_60m, "4h") if not df_60m.empty else None
    elif timeframe == "1d":
        print(f"    [1/4] yf.download() - 1d...")
        df = _try_download_yf(asset, "max", "1d")
    elif timeframe == "1wk":
        print(f"    [1/4] yf.download() - 1wk...")
        df = _try_download_yf(asset, "max", "1wk")
        if df is None or df.empty:
            df_daily = _try_download_yf(asset, "max", "1d")
            if df_daily is not None:
                df_daily = _normalize_columns(df_daily)
                df = resample_ohlcv(df_daily, "1W") if not df_daily.empty else None

    if df is not None:
        df = _normalize_columns(df)
        if not df.empty:
            print(f"      -> yf.download() exitoso: {len(df)} velas.")
        else:
            df = None

    # ── Intento 2: yf.Ticker().history() ──
    if df is None:
        time.sleep(0.5)  # cooldown antes de Ticker.history() para evitar rate limiting
        print(f"    [2/4] yf.Ticker().history()...")
        if timeframe == "1h":
            df_tmp = _try_download_ticker(asset, "730d", "60m")
            if df_tmp is not None:
                df = _normalize_columns(df_tmp)
        elif timeframe == "4h":
            df_60m = _try_download_ticker(asset, "730d", "60m")
            if df_60m is not None:
                df_60m = _normalize_columns(df_60m)
                if not df_60m.empty:
                    df = resample_ohlcv(df_60m, "4h")
        elif timeframe == "1d":
            df_tmp = _try_download_ticker(asset, "max", "1d")
            if df_tmp is not None:
                df = _normalize_columns(df_tmp)
        elif timeframe == "1wk":
            df_tmp = _try_download_ticker(asset, "max", "1wk")
            if df_tmp is not None:
                df = _normalize_columns(df_tmp)
            else:
                df_daily = _try_download_ticker(asset, "max", "1d")
                if df_daily is not None:
                    df_daily = _normalize_columns(df_daily)
                    df = resample_ohlcv(df_daily, "1W") if not df_daily.empty else None

        if df is not None and not df.empty:
            print(f"      -> Ticker.history() exitoso: {len(df)} velas.")
        else:
            df = None

    # ── Intento 3 (4h): probar 1h + resample ──
    if df is None and timeframe == "4h":
        time.sleep(0.5)  # cooldown antes del fallback 1h/1d para evitar rate limiting
        print(f"    [3/4] 60m falló, probando 1h data + resample...")
        # Probar Ticker.history con 1h
        df_1h = _try_download_ticker(asset, "730d", "1h")
        if df_1h is not None:
            df_1h = _normalize_columns(df_1h)
            if not df_1h.empty:
                print(f"      -> 1h descargado ({len(df_1h)} velas), resampleando a 4H...")
                df = resample_ohlcv(df_1h, "4h")
        # Si aún falla, probar 1d + resample
        if df is None or df.empty:
            df_1d = _try_download_ticker(asset, "max", "1d")
            if df_1d is not None:
                df_1d = _normalize_columns(df_1d)
                if not df_1d.empty:
                    print(f"      -> 1d descargado ({len(df_1d)} velas), resampleando a 4H (último recurso)...")
                    df = resample_ohlcv(df_1d, "4h")

    # ── Último recurso (intradía): probar 1d + resample ──
    if df is None and timeframe == "1h":
        print(f"    [3/3] 60m falló, probando 1d + resample (último recurso)...")
        df_1d = _try_download_ticker(asset, "max", "1d")
        if df_1d is not None:
            df_1d = _normalize_columns(df_1d)
            if not df_1d.empty:
                print(f"      -> 1d descargado ({len(df_1d)} velas), resampleando a 1H...")
                df = resample_ohlcv(df_1d, "1h")

    if df is None or df.empty:
        print(f"  [WARN] {asset} @ {timeframe}: no se pudieron descargar datos por ningún método.")
        return None

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Aviso si hay menos datos de los esperados para intradía
    if timeframe in ("1h", "4h") and len(df) < 500:
        print(f"    [WARN] {asset} @ {timeframe}: solo {len(df)} velas (yfinance limita intradía).")

    if not _check_min_data(df, asset, timeframe):
        return None

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2) RESAMPLE OHLCV
# ──────────────────────────────────────────────────────────────────────────────


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resamplea datos OHLCV a una temporalidad superior.
    - Open = first
    - High = max
    - Low = min
    - Close = last
    - Volume = sum
    """
    ohlc_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adj Close": "last",
        "Volume": "sum",
    }
    # Solo columnas existentes
    cols_to_use = {k: v for k, v in ohlc_dict.items() if k in df.columns}
    df_resampled = df.resample(rule).agg(cols_to_use)
    df_resampled = df_resampled.dropna(subset=["Open", "High", "Low", "Close"])
    return df_resampled


# ──────────────────────────────────────────────────────────────────────────────
# 3) INDICADORES TÉCNICOS
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
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    return atr


def _bollinger_bands(close: pd.Series, length: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = _sma(close, length)
    sd = _std(close, length)
    upper = ma + std_dev * sd
    lower = ma - std_dev * sd
    return upper, ma, lower


def _keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = _ema(close, length)
    atr = _atr_wilder(high, low, close, length)
    upper = ma + mult * atr
    lower = ma - mult * atr
    return upper, ma, lower


def _adx_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    tr = _tr(high, low, close, prev_close)

    # Wilder smoothing
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    plus_di_smooth = (plus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / atr) * 100.0
    minus_di_smooth = (minus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / atr) * 100.0

    dx = (plus_di_smooth - minus_di_smooth).abs() / (plus_di_smooth + minus_di_smooth).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1.0 / length, adjust=False).mean()

    return adx, plus_di_smooth, minus_di_smooth


def _squeeze_momentum(
    high: pd.Series, low: pd.Series, close: pd.Series, bb_length: int = 20, bb_std: float = 2.0,
    kc_length: int = 20, kc_mult: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Implementa squeeze-on/off y un histograma tipo Squeeze Momentum.
    squeeze_on: Bollinger completamente dentro de Keltner.
    smi_hist: basado en la diferencia entre el precio y la media de Bollinger,
              normalizado por el ancho de Bollinger (versión estable).
    """
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, bb_length, bb_std)
    kc_upper, kc_mid, kc_lower = _keltner_channels(high, low, close, kc_length, kc_mult)

    squeeze_on = (bb_lower >= kc_lower) & (bb_upper <= kc_upper)
    squeeze_off = ~squeeze_on

    # Histograma SMI: diferencia normalizada del precio respecto a la media de BB
    bb_width = bb_upper - bb_lower
    bb_width_safe = bb_width.replace(0, np.nan)
    # smi_hist: cuán lejos está el close del mid de BB, en desviaciones
    smi_hist = (close - bb_mid) / bb_width_safe * 100.0
    smi_hist = smi_hist.fillna(0)

    return squeeze_on, squeeze_off, smi_hist


def _volume_profile_bias(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    lookback: int = 75, bins: int = 24
) -> Tuple[pd.Series, pd.Series]:
    """
    Proxy rolling de Volume Profile.
    Calcula el POC (punto de control) como el centro del bin con mayor volumen
    acumulado en una ventana rolling.
    """
    typical_price = (high + low + close) / 3.0

    poc_series = pd.Series(np.nan, index=close.index)
    bias_bull = pd.Series(False, index=close.index)
    bias_bear = pd.Series(False, index=close.index)

    for i in range(lookback, len(close)):
        window_start = i - lookback
        tp_window = typical_price.iloc[window_start:i].values
        vol_window = volume.iloc[window_start:i].values

        if len(tp_window) < 2 or np.all(np.isnan(tp_window)) or np.all(vol_window == 0):
            continue

        tp_min, tp_max = np.nanmin(tp_window), np.nanmax(tp_window)
        if tp_max - tp_min < 1e-10:
            poc_series.iloc[i] = tp_window[~np.isnan(tp_window)][0] if np.any(~np.isnan(tp_window)) else close.iloc[i]
            bias_bull.iloc[i] = close.iloc[i] > poc_series.iloc[i]
            bias_bear.iloc[i] = close.iloc[i] < poc_series.iloc[i]
            continue

        bin_edges = np.linspace(tp_min, tp_max, bins + 1)
        bin_indices = np.digitize(tp_window, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        vol_by_bin = np.zeros(bins)
        for j, idx in enumerate(bin_indices):
            if not np.isnan(vol_window[j]):
                vol_by_bin[idx] += vol_window[j]

        max_bin = int(np.argmax(vol_by_bin))
        poc = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2.0
        poc_series.iloc[i] = poc
        bias_bull.iloc[i] = close.iloc[i] > poc
        bias_bear.iloc[i] = close.iloc[i] < poc

    return poc_series, bias_bull


# ──────────────────────────────────────────────────────────────────────────────
# 4) CÁLCULO COMPLETO DE INDICADORES
# ──────────────────────────────────────────────────────────────────────────────


def compute_base_indicators(df: pd.DataFrame, kc_mult: float = KC_MULT) -> pd.DataFrame:
    """
    Calcula indicadores que NO dependen de parámetros del sweep.
    Se ejecuta una sola vez por activo/timeframe.
    """
    df = df.copy()

    # ATR
    df["atr"] = _atr_wilder(df["High"], df["Low"], df["Close"], ATR_LENGTH)

    # Squeeze Momentum (usa kc_mult ajustable por TF)
    squeeze_on, squeeze_off, smi_hist = _squeeze_momentum(
        df["High"], df["Low"], df["Close"],
        bb_length=BB_LENGTH, bb_std=BB_STD,
        kc_length=KC_LENGTH, kc_mult=kc_mult,
    )
    df["squeeze_on"] = squeeze_on.astype(bool)
    df["squeeze_off"] = squeeze_off.astype(bool)
    df["smi_hist"] = smi_hist
    df["smi_delta"] = df["smi_hist"].diff()

    # ADX
    adx, plus_di, minus_di = _adx_wilder(df["High"], df["Low"], df["Close"], ADX_LENGTH)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx_delta"] = df["adx"].diff()

    return df


def compute_ema_indicators(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST_DEFAULT,
    ema_slow: int = EMA_SLOW_DEFAULT,
) -> pd.DataFrame:
    """
    Calcula exclusivamente EMAs (se ejecuta por cada combo del sweep,
    porque dependen de ema_fast/ema_slow).
    """
    df = df.copy()
    df["ema_fast"] = _ema(df["Close"], ema_fast)
    df["ema_slow"] = _ema(df["Close"], ema_slow)
    return df


def precompute_volume_profiles(
    df: pd.DataFrame,
    vp_lookbacks: List[int],
    vp_bins: int = VP_BINS,
) -> Dict[int, Tuple[pd.Series, pd.Series]]:
    """
    Precalcula volume profile para cada valor distinto de vp_lookback.
    Retorna dict {lookback: (poc_series, bias_bull_series)}.
    """
    result = {}
    for lb in sorted(set(vp_lookbacks)):
        print(f"      [VP] Precalculando volume profile con lookback={lb}...")
        poc, bias_bull = _volume_profile_bias(
            df["High"], df["Low"], df["Close"], df["Volume"],
            lookback=lb, bins=vp_bins,
        )
        result[lb] = (poc, bias_bull)
    return result


def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int = EMA_FAST_DEFAULT,
    ema_slow: int = EMA_SLOW_DEFAULT,
    adx_length: int = ADX_LENGTH,
    atr_length: int = ATR_LENGTH,
    bb_length: int = BB_LENGTH,
    bb_std: float = BB_STD,
    kc_length: int = KC_LENGTH,
    kc_mult: float = KC_MULT,
    vp_lookback: int = VP_LOOKBACK_DEFAULT,
    vp_bins: int = VP_BINS,
    vp_data: Optional[Tuple[pd.Series, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos de la estrategia.
    Si se pasa vp_data (poc, bias_bull), omite el costoso cálculo del volume profile.
    """
    df = df.copy()

    # Dirección
    df["ema_fast"] = _ema(df["Close"], ema_fast)
    df["ema_slow"] = _ema(df["Close"], ema_slow)

    # ATR
    df["atr"] = _atr_wilder(df["High"], df["Low"], df["Close"], atr_length)

    # Squeeze Momentum
    squeeze_on, squeeze_off, smi_hist = _squeeze_momentum(
        df["High"], df["Low"], df["Close"],
        bb_length=bb_length, bb_std=bb_std,
        kc_length=kc_length, kc_mult=kc_mult,
    )
    df["squeeze_on"] = squeeze_on.astype(bool)
    df["squeeze_off"] = squeeze_off.astype(bool)
    df["smi_hist"] = smi_hist
    df["smi_delta"] = df["smi_hist"].diff()

    # ADX
    adx, plus_di, minus_di = _adx_wilder(df["High"], df["Low"], df["Close"], adx_length)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx_delta"] = df["adx"].diff()

    # Volume Profile proxy (saltar si ya está precalculado)
    if vp_data is not None:
        poc, bias_bull = vp_data
        df["vp_poc"] = poc
        df["volume_bias_bull"] = bias_bull
    else:
        poc, bias_bull = _volume_profile_bias(
            df["High"], df["Low"], df["Close"], df["Volume"],
            lookback=vp_lookback, bins=vp_bins,
        )
        df["vp_poc"] = poc
        df["volume_bias_bull"] = bias_bull
    df["volume_bias_bear"] = ~df["volume_bias_bull"]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5) BACKTEST
# ──────────────────────────────────────────────────────────────────────────────


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    init_capital: float = INITIAL_CAPITAL,
    commission_pct: float = COMMISSION_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
    asset: str = "",
    timeframe: str = "",
    regimes: Optional[np.ndarray] = None,
    vp_data: Optional[Tuple[pd.Series, pd.Series]] = None,
    base_indicators_df: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Backtest de la estrategia TradingLatino cuantificable.
    OHLC real, stops/targets intrabar, una posición a la vez.

    Optimización: acepta base_indicators_df (con indicadores independientes del sweep
    ya calculados) y vp_data precalculado para evitar recomputar lo costoso.
    """
    if base_indicators_df is not None:
        # Partir de la base precalculada y solo añadir EMAs + VP
        df = compute_ema_indicators(base_indicators_df, ema_fast=params.ema_fast, ema_slow=params.ema_slow)
        df["vp_poc"] = vp_data[0] if vp_data is not None else base_indicators_df.get("vp_poc", np.nan)
        df["volume_bias_bull"] = vp_data[1] if vp_data is not None else base_indicators_df.get("volume_bias_bull", False)
        df["volume_bias_bear"] = ~df["volume_bias_bull"]
    else:
        df = df.copy()
        df = compute_indicators(
            df,
            ema_fast=params.ema_fast,
            ema_slow=params.ema_slow,
            vp_lookback=params.vp_lookback,
            vp_data=vp_data,
        )

    if regimes is not None and len(regimes) == len(df):
        df["regime"] = regimes
    else:
        df["regime"] = -1

    trades: List[TradeRecord] = []
    position: Optional[str] = None  # "LONG" | "SHORT" | None
    entry_price: float = 0.0
    entry_date: pd.Timestamp = pd.Timestamp(0)
    entry_idx: int = -1
    entry_atr: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    smi_delta_negative_count: int = 0
    smi_delta_positive_count: int = 0

    # Precalcular señales para evitar re-computar
    df["bull_bias"] = (
        (df["Close"] > df["ema_slow"])
        & (df["ema_fast"] > df["ema_slow"])
    )
    df["bear_bias"] = (
        (df["Close"] < df["ema_slow"])
        & (df["ema_fast"] < df["ema_slow"])
    )

    # Squeeze release flag: hubo squeeze_on en las últimas N velas
    df["squeeze_released"] = (
        df["squeeze_on"]
        .rolling(window=params.release_lookback, min_periods=1)
        .max()
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # Señal long
    df["signal_long"] = (
        df["bull_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] > 0)
        & (df["smi_delta"] > 0)
        & (df["adx"] > params.adx_threshold)
        & (df["adx_delta"] > 0)
    )

    # Señal short
    df["signal_short"] = (
        df["bear_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] < 0)
        & (df["smi_delta"] < 0)
        & (df["adx"] > params.adx_threshold)
        & (df["adx_delta"] > 0)
    )

    if params.use_volume_filter:
        df["signal_long"] = df["signal_long"] & df["volume_bias_bull"]
        df["signal_short"] = df["signal_short"] & df["volume_bias_bear"]

    # Equity curve pre-asignada
    equity = pd.Series(init_capital, index=df.index)
    cash = init_capital
    position_qty: float = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]

        # --- GESTIÓN DE POSICIÓN ABIERTA ---
        if position is not None:
            high, low = row["High"], row["Low"]
            exit_triggered = False
            exit_reason = ""
            exit_price_candidate = row["Open"]  # default: salir a open si se señal contraria

            if position == "LONG":
                # Stop: si low <= stop_price
                if low <= stop_price:
                    exit_price_candidate = min(stop_price * (1 + slippage_pct), high)
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                # Take profit: si high >= target_price
                if not exit_triggered and high >= target_price:
                    exit_price_candidate = max(target_price * (1 - slippage_pct), low)
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                # Señal contraria completa
                if not exit_triggered and df.iloc[i]["signal_short"]:
                    exit_price_candidate = row["Open"] * (1 - slippage_pct)
                    exit_triggered = True
                    exit_reason = "REVERSE_SIGNAL"
                # Salida temprana opcional
                if not exit_triggered:
                    if df.iloc[i]["smi_delta"] < 0:
                        smi_delta_negative_count += 1
                    else:
                        smi_delta_negative_count = 0
                    if smi_delta_negative_count >= EARLY_EXIT_SMI_BARS and df.iloc[i]["adx_delta"] < 0:
                        exit_price_candidate = row["Open"] * (1 - slippage_pct)
                        exit_triggered = True
                        exit_reason = "EARLY_EXIT"

            elif position == "SHORT":
                if high >= stop_price:
                    exit_price_candidate = max(stop_price * (1 - slippage_pct), low)
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                if not exit_triggered and low <= target_price:
                    exit_price_candidate = min(target_price * (1 + slippage_pct), high)
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                if not exit_triggered and df.iloc[i]["signal_long"]:
                    exit_price_candidate = row["Open"] * (1 + slippage_pct)
                    exit_triggered = True
                    exit_reason = "REVERSE_SIGNAL"
                if not exit_triggered:
                    if df.iloc[i]["smi_delta"] > 0:
                        smi_delta_positive_count += 1
                    else:
                        smi_delta_positive_count = 0
                    if smi_delta_positive_count >= EARLY_EXIT_SMI_BARS and df.iloc[i]["adx_delta"] < 0:
                        exit_price_candidate = row["Open"] * (1 + slippage_pct)
                        exit_triggered = True
                        exit_reason = "EARLY_EXIT"

            if exit_triggered:
                # Cerrar posición
                exit_date = df.index[i]
                entry_comm = commission_pct * abs(position_qty * entry_price)
                exit_comm = commission_pct * abs(position_qty * exit_price_candidate)
                if position == "LONG":
                    pnl = position_qty * (exit_price_candidate - entry_price) - entry_comm - exit_comm
                    pnl_pct = (exit_price_candidate - entry_price) / entry_price - commission_pct
                    # Vender shares: recibe exit_price, paga comisión de salida
                    cash += position_qty * exit_price_candidate * (1 - commission_pct)
                else:  # SHORT
                    pnl = position_qty * (entry_price - exit_price_candidate) - entry_comm - exit_comm
                    pnl_pct = (entry_price - exit_price_candidate) / entry_price - commission_pct
                    # Comprar para cubrir: paga exit_price + comisión de salida
                    cash -= position_qty * exit_price_candidate * (1 + commission_pct)

                trade = TradeRecord(
                    asset=asset,
                    timeframe=timeframe,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    side=position,
                    entry_price=entry_price,
                    exit_price=exit_price_candidate,
                    stop_price=stop_price,
                    target_price=target_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct * 100.0,
                    bars_in_trade=i - entry_idx,
                    regime_at_entry=int(df.iloc[entry_idx]["regime"]) if 0 <= entry_idx < len(df) else -1,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                position = None
                position_qty = 0.0
                smi_delta_negative_count = 0
                smi_delta_positive_count = 0

        # --- APERTURA DE NUEVA POSICIÓN ---
        if position is None:
            smi_delta_negative_count = 0
            smi_delta_positive_count = 0

            if df.iloc[i]["signal_long"]:
                position = "LONG"
                entry_price = row["Open"] * (1 + slippage_pct)
                entry_date = df.index[i]
                entry_idx = i
                entry_atr = row["atr"]
                stop_price = entry_price - entry_atr * params.atr_stop_mult
                target_price = entry_price + (entry_price - stop_price) * params.rr_target
                position_qty = cash / entry_price * (1 - commission_pct)
                cash -= position_qty * entry_price + commission_pct * (position_qty * entry_price)

            elif df.iloc[i]["signal_short"]:
                position = "SHORT"
                entry_price = row["Open"] * (1 - slippage_pct)
                entry_date = df.index[i]
                entry_idx = i
                entry_atr = row["atr"]
                stop_price = entry_price + entry_atr * params.atr_stop_mult
                target_price = entry_price - (stop_price - entry_price) * params.rr_target
                position_qty = cash / entry_price * (1 - commission_pct)
                cash += position_qty * entry_price - commission_pct * (position_qty * entry_price)

        # Valor del portfolio
        if position == "LONG":
            equity.iloc[i] = cash + position_qty * row["Close"]
        elif position == "SHORT":
            # Para short: cash incluye el efectivo recibido por la venta,
            # pero debemos qty*close para cubrir la posición
            equity.iloc[i] = cash - position_qty * row["Close"]
        else:
            equity.iloc[i] = cash

    result = BacktestResult(params=params, trades=trades, equity_curve=equity)
    result.metrics = compute_metrics(trades, equity, init_capital)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6) MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    trades: List[TradeRecord],
    equity: pd.Series,
    init_capital: float,
) -> Dict[str, float]:
    """Calcula métricas de rendimiento a partir de trades y equity curve."""
    metrics: Dict[str, float] = {}

    if not trades or len(trades) < 2:
        return {
            "net_profit": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "expectancy": 0.0,
            "num_trades": 0,
        }

    # Net Profit
    final_equity = equity.iloc[-1]
    net_profit = final_equity - init_capital
    metrics["net_profit"] = net_profit

    # CAGR
    days = (equity.index[-1] - equity.index[0]).total_seconds() / (60 * 60 * 24)
    years = days / 365.25
    if years > 0 and init_capital > 0 and final_equity > 0:
        cagr = (final_equity / init_capital) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0
    metrics["cagr"] = cagr

    # Returns diarios de la equity curve (usamos log returns)
    eq_returns = equity.pct_change().dropna()
    if len(eq_returns) == 0:
        metrics.update({"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "calmar": 0.0,
                        "profit_factor": 0.0, "win_rate": 0.0, "avg_trade": 0.0,
                        "expectancy": 0.0, "num_trades": len(trades)})
        return metrics

    # Sharpe
    avg_ret = eq_returns.mean()
    std_ret = eq_returns.std()
    if std_ret > 1e-10:
        sharpe = avg_ret / std_ret * np.sqrt(252)
    else:
        sharpe = 0.0
    metrics["sharpe"] = sharpe

    # Sortino
    downside = eq_returns[eq_returns < 0]
    downside_std = downside.std()
    if downside_std > 1e-10:
        sortino = avg_ret / downside_std * np.sqrt(252)
    else:
        sortino = 0.0
    metrics["sortino"] = sortino

    # Max Drawdown
    cumulative = (1 + eq_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    metrics["max_dd"] = max_dd if not np.isnan(max_dd) else 0.0

    # Calmar
    if abs(max_dd) > 1e-10:
        calmar = cagr / abs(max_dd)
    else:
        calmar = 0.0
    metrics["calmar"] = calmar

    # Profit Factor
    gross_profit = sum(max(t.pnl, 0) for t in trades)
    gross_loss = abs(sum(min(t.pnl, 0) for t in trades))
    if gross_loss > 1e-10:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = gross_profit if gross_profit > 0 else 0.0
    metrics["profit_factor"] = profit_factor

    # Win Rate
    wins = sum(1 for t in trades if t.pnl > 0)
    metrics["win_rate"] = wins / len(trades) if trades else 0.0

    # Avg Trade
    metrics["avg_trade"] = np.mean([t.pnl for t in trades]) if trades else 0.0

    # Expectancy
    if trades:
        avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0.0
        avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if (len(trades) - wins) > 0 else 0.0
        win_rate = metrics["win_rate"]
        if abs(avg_loss) > 1e-10:
            expectancy = (win_rate * avg_win / abs(avg_loss)) - ((1 - win_rate))
        else:
            expectancy = 0.0
        metrics["expectancy"] = expectancy
    else:
        metrics["expectancy"] = 0.0

    metrics["num_trades"] = len(trades)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 7) HMM: CONSTRUCCIÓN DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────


def build_hmm_features(df: pd.DataFrame, feature_window: int = FEATURE_WINDOW) -> pd.DataFrame:
    """
    Construye features de mercado + estrategia para el HMM.
    Devuelve un DataFrame con features y el índice original.
    """
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    # 1) log_return_1
    features["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2) vol_20
    features["vol_20"] = features["log_return_1"].rolling(window=feature_window).std()

    # 3) cumret_20
    features["cumret_20"] = features["log_return_1"].rolling(window=feature_window).sum()

    # 4) momentum_20
    features["momentum_20"] = df["Close"].pct_change(feature_window)

    # 5) atr_norm
    atr_col = df["atr"] if "atr" in df.columns else pd.Series(0.0, index=df.index)
    features["atr_norm"] = atr_col / df["Close"].replace(0, np.nan)

    # 6) ema_spread_atr
    ema_fast = df.get("ema_fast", _ema(df["Close"], 10))
    ema_slow = df.get("ema_slow", _ema(df["Close"], 55))
    atr_safe = atr_col.replace(0, np.nan)
    features["ema_spread_atr"] = (ema_fast - ema_slow) / atr_safe

    # 7) smi_hist_norm
    smi_hist = df.get("smi_hist", pd.Series(0.0, index=df.index))
    features["smi_hist_norm"] = smi_hist / atr_safe.replace(0, np.nan)
    features["smi_hist_norm"] = features["smi_hist_norm"].fillna(0)

    # 8) adx_scaled
    adx = df.get("adx", pd.Series(0.0, index=df.index))
    features["adx_scaled"] = adx / 100.0

    # 9) adx_delta_scaled
    features["adx_delta_scaled"] = df.get("adx_delta", pd.Series(0.0, index=df.index)) / 100.0

    # 10) squeeze_flag
    features["squeeze_flag"] = df.get("squeeze_on", pd.Series(False, index=df.index)).astype(int)

    # 11) poc_distance_atr
    vp_poc = df.get("vp_poc", df["Close"])
    features["poc_distance_atr"] = (df["Close"] - vp_poc) / atr_safe

    return features


# ──────────────────────────────────────────────────────────────────────────────
# 8) HMM: FIT + BIC + RELABEL
# ──────────────────────────────────────────────────────────────────────────────


def _compute_bic(hmm_model, log_likelihood: float, n_samples: int) -> float:
    """Calcula BIC manualmente para GaussianHMM diagonal."""
    n_states = hmm_model.n_components
    n_features = hmm_model.n_features
    # Para GaussianHMM con covariance_type="diag":
    #   - medias: n_states * n_features
    #   - covars: n_states * n_features (diagonal)
    #   - transmat: n_states * (n_states - 1) (última fila determinada)
    #   - startprob: n_states - 1
    k = n_states * n_features * 2 + n_states * (n_states - 1) + (n_states - 1)
    bic = -2 * log_likelihood + k * np.log(n_samples)
    return bic


def fit_hmm_select_bic(
    features_df: pd.DataFrame,
    state_range: List[int] = HMM_STATE_RANGE,
    covariance_type: str = HMM_COVARIANCE_TYPE,
    random_state: int = RANDOM_STATE,
    n_iter: int = 200,
) -> Tuple[Optional[HMMResult], pd.DataFrame]:
    """
    Prueba varios números de estados para GaussianHMM, elige por BIC.
    Retorna (mejor_resultado, df_summary).
    """
    # Eliminar NaN
    clean = features_df.dropna()
    if len(clean) < 100:
        print("    [WARN] Datos insuficientes para HMM después de eliminar NaN.")
        return None, pd.DataFrame()

    # Estandarizar (z-score)
    X = clean.values
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds = np.where(stds < 1e-10, 1.0, stds)
    X_scaled = (X - means) / stds

    results = []
    best_bic = np.inf
    best_result: Optional[HMMResult] = None

    for n_states in state_range:
        try:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                random_state=random_state,
                n_iter=n_iter,
                tol=1e-4,
            )
            model.fit(X_scaled)
            log_likelihood = model.score(X_scaled)
            bic = _compute_bic(model, log_likelihood, len(X_scaled))
            states = model.predict(X_scaled)
            results.append({
                "n_states": n_states,
                "log_likelihood": log_likelihood,
                "bic": bic,
            })
            if bic < best_bic:
                best_bic = bic
                hmm_res = HMMResult(
                    n_states=n_states,
                    model=model,
                    states=states,
                    bic=bic,
                    log_likelihood=log_likelihood,
                )
                best_result = hmm_res
        except Exception as e:
            print(f"    [WARN] HMM con {n_states} estados falló: {e}")
            results.append({"n_states": n_states, "log_likelihood": np.nan, "bic": np.nan})
            continue

    summary_df = pd.DataFrame(results)
    return best_result, summary_df


def relabel_states_by_vol(model, states: np.ndarray, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Reetiqueta estados por volatilidad ascendente (0 = menor volatilidad).
    Retorna (states_relabeled, state_summary).
    """
    unique_states = np.unique(states)
    # Usar vol_20 como proxy de volatilidad si existe
    vol_col = "vol_20"
    if vol_col in features_df.columns:
        vol_values = features_df[vol_col].values
    else:
        # fallback: usar log_return_1
        vol_values = np.abs(features_df.get("log_return_1", pd.Series(0.0, index=features_df.index))).values

    # Asignar a states
    state_vol_map = {}
    for s in unique_states:
        mask = states == s
        if np.any(mask):
            state_vol_map[s] = np.nanmean(vol_values[:len(states)][mask])
        else:
            state_vol_map[s] = 0.0

    # Ordenar por volatilidad
    sorted_states = sorted(state_vol_map.keys(), key=lambda x: state_vol_map[x])
    relabel_map = {old: new for new, old in enumerate(sorted_states)}
    states_relabeled = np.array([relabel_map[s] for s in states])

    # Calcular resumen por estado
    n_states = len(unique_states)
    total_len = len(states_relabeled)
    rows = []
    for s in range(n_states):
        mask = states_relabeled == s
        count = int(mask.sum())
        pct = count / total_len * 100.0
        mean_ret = np.nanmean(features_df["log_return_1"].values[:total_len][mask]) if "log_return_1" in features_df.columns else 0.0
        vol = np.nanstd(features_df["log_return_1"].values[:total_len][mask]) if "log_return_1" in features_df.columns else 0.0

        # Duración media de rachas
        runs = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        run_starts = np.where(runs == 1)[0]
        run_ends = np.where(runs == -1)[0]
        run_lengths = run_ends - run_starts
        mean_duration = np.mean(run_lengths) if len(run_lengths) > 0 else 0.0

        rows.append({
            "state": s,
            "pct_time": pct,
            "mean_return": mean_ret,
            "volatility": vol,
            "mean_duration_bars": mean_duration,
        })

    state_summary = pd.DataFrame(rows)
    return states_relabeled, state_summary


# ──────────────────────────────────────────────────────────────────────────────
# 9) AUTO-TUNE KC_MULT
# ──────────────────────────────────────────────────────────────────────────────


def auto_tune_kc_mult(
    df: pd.DataFrame,
    timeframe: str,
    target_sq_rate: float = 0.25,  # objetivo: 25% de velas en squeeze
    release_lookback: int = 3,
    min_candidates: int = 20,
) -> float:
    """
    Encuentra el kc_mult óptimo para un timeframe probando varios valores.
    Estrategia:
      1) Prueba kc_mult de 1.0 a 5.0 en pasos de 0.25
      2) Para cada uno, calcula squeeze_on rate y cota de señales candidatas
      3) Puntúa: máximo bonus si squeeze_on rate ≈ target, penaliza si demasiado
         bajo (nunca squeeze) o demasiado alto (siempre squeeze → nunca squeeze_off)
      4) Retorna el kc_mult con mejor puntuación
    """
    print(f"  [AUTO-TUNE] Buscando kc_mult óptimo para {timeframe}...")

    candidates = np.arange(1.0, 5.25, 0.25)
    best_kc = KC_MULT_BY_TF.get(timeframe, KC_MULT)
    best_score = -np.inf
    results_log = []

    df_base = df.copy()
    # Precalcular ATR y ADX (no dependen de kc_mult)
    df_base["atr"] = _atr_wilder(df_base["High"], df_base["Low"], df_base["Close"], ATR_LENGTH)

    best_result = None
    for kc in candidates:
        squeeze_on, squeeze_off, smi_hist = _squeeze_momentum(
            df_base["High"], df_base["Low"], df_base["Close"],
            bb_length=BB_LENGTH, bb_std=BB_STD,
            kc_length=KC_LENGTH, kc_mult=kc,
        )
        sq_on_rate = squeeze_on.mean()
        # Candidatos: squeeze_off + squeeze_released (release en las últimas N velas)
        squeeze_released = (
            squeeze_on.rolling(window=release_lookback, min_periods=1)
            .max()
            .shift(1)
            .fillna(False)
            .astype(bool)
        )
        candidates_signals = (squeeze_off & squeeze_released).sum()

        # Puntuación:
        # - Perfecto si sq_on_rate está entre 15% y 40%
        # - Bonus lineal entre 0% y 15%, y penalización > 40%
        if sq_on_rate < 0.01:
            score = -abs(kc - 3.0)  # castigo fuerte si nunca squeeze
        elif sq_on_rate < 0.15:
            # Bonus: cuanto más cerca de 0.15 mejor
            score = sq_on_rate / 0.15 * 10.0
        elif sq_on_rate <= 0.40:
            # Zona óptima: puntuación máxima (10) + bonus por candidatos
            score = 10.0 + min(candidates_signals / max(min_candidates, 1), 5.0)
        else:
            # Penalización: demasiado squeeze → squeeze_off raro
            penalty = (sq_on_rate - 0.40) / 0.40 * 5.0
            score = 10.0 - penalty + min(candidates_signals / max(min_candidates, 1), 2.0)

        results_log.append((kc, sq_on_rate, candidates_signals, score))

        if score > best_score:
            best_score = score
            best_kc = kc
            best_result = (kc, sq_on_rate, candidates_signals, score)

    # Mostrar resultados
    print(f"    kc_mult  squeeze_rate  candidates  score")
    for kc, sq, cand, sc in results_log:
        marker = " <<<" if abs(kc - best_kc) < 0.001 else ""
        print(f"    {kc:5.2f}  {sq*100:>6.1f}%  {cand:>6d}    {sc:5.1f}{marker}")

    if best_result is not None:
        bk, sq_r, cand_s, _ = best_result
        print(f"    -> Mejor kc_mult = {bk:.2f} (squeeze_rate={sq_r*100:.1f}%, candidates={cand_s})")

        if cand_s < min_candidates and timeframe in ("1d", "1wk"):
            print(f"    ⚠️  Incluso con kc_mult={bk:.2f}, solo {cand_s} velas candidatas.")
            print(f"       Para timeframes largos puede ser normal (pocas velas totales).")

    return best_kc


# ──────────────────────────────────────────────────────────────────────────────
# 10) PARAMETER SWEEP
# ──────────────────────────────────────────────────────────────────────────────


def sweep_parameters(
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    asset: str,
    timeframe: str,
    regimes: Optional[np.ndarray] = None,
    min_trades: int = MIN_TRADES_FILTER,
    kc_mult: float = KC_MULT,
) -> List[BacktestResult]:
    """
    Grid search sobre parámetros de la estrategia.
    Retorna lista de BacktestResult filtrados por min_trades.

    Optimización: precalcula indicadores base y volume profiles una sola vez.
    """
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Generar combinaciones
    all_combos = list(product(*param_values))
    valid_combos = []
    for combo in all_combos:
        combo_dict = dict(zip(param_keys, combo))
        # Validar ema_fast < ema_slow
        if combo_dict["ema_fast"] >= combo_dict["ema_slow"]:
            continue
        valid_combos.append(combo_dict)

    print(f"    -> {len(valid_combos)} combinaciones válidas (de {len(all_combos)} totales).")

    # ── Precalcular indicadores base (no dependen de sweep) ──
    print(f"    [SWEEP] Precalculando indicadores base (kc_mult={kc_mult})...")
    base_df = compute_base_indicators(df, kc_mult=kc_mult)

    # ── Precalcular volume profiles para cada vp_lookback distinto ──
    distinct_vp_lookbacks = sorted(set(c["vp_lookback"] for c in valid_combos))
    print(f"    [SWEEP] Precalculando volume profiles para {distinct_vp_lookbacks}...")
    vp_cache = precompute_volume_profiles(base_df, distinct_vp_lookbacks, vp_bins=VP_BINS)

    # ── Sweep loop ──
    results: List[BacktestResult] = []
    total = len(valid_combos)
    max_trades_found = 0
    best_combo_trades: Optional[Dict[str, Any]] = None
    print(f"    [SWEEP] Ejecutando {total} backtests...")
    t_start = time.time()

    for i, combo in enumerate(valid_combos):
        params = StrategyParams(**combo)
        vp_data = vp_cache.get(params.vp_lookback)
        bt_result = run_backtest(
            base_df, params,
            asset=asset, timeframe=timeframe, regimes=regimes,
            vp_data=vp_data, base_indicators_df=base_df,
        )
        num_trades = bt_result.metrics.get("num_trades", 0)
        if num_trades > max_trades_found:
            max_trades_found = num_trades
            best_combo_trades = combo
        if num_trades >= min_trades:
            results.append(bt_result)

        # Progreso cada 5% con tiempo estimado restante
        pct_step = max(1, total // 20)  # cada 5%
        if (i + 1) % pct_step == 0 or i == 0 or i == total - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.1)
            remaining = (total - i - 1) / max(rate, 0.1)
            print(f"      Progreso: {i + 1}/{total} ({100 * (i + 1) // total}%) - "
                  f"{len(results)} válidos - ETA: {remaining:.0f}s ({remaining/60:.1f}min)")

    total_time = time.time() - t_start
    if results:
        print(f"    [SWEEP] Completado en {total_time:.0f}s: {len(results)} combinaciones superan el filtro de {min_trades} trades.")
    else:
        print(f"    [SWEEP] Completado en {total_time:.0f}s: 0 combinaciones superan el filtro de {min_trades} trades.")
        if best_combo_trades is not None:
            print(f"    [SWEEP] Máximo de trades encontrados: {max_trades_found} (necesitaba {min_trades}).")
            print(f"    [SWEEP] Combinación con más trades: {best_combo_trades}")
            # Diagnosticar por qué no hay señales
            print(f"    [DIAGNÓSTICO] Posibles causas:")
            print(f"      - Pocas velas semanales en el periodo descargado")
            print(f"      - El kc_mult={kc_mult} no genera suficientes squeezes")
            print(f"      - La tendencia de XRP puede no alinearse con las EMAs")
            print(f"      - El filtro ADX puede ser demasiado restrictivo para semanal")
        else:
            print(f"    [SWEEP] No se encontraron trades en NINGUNA combinación.")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 10) CLASIFICACIÓN POR RÉGIMEN
# ──────────────────────────────────────────────────────────────────────────────


def classify_trades_by_regime(
    results: List[BacktestResult],
    n_states: int,
) -> Dict[str, Any]:
    """
    Clasifica trades por régimen de entrada y calcula métricas por régimen.
    Retorna dict con:
      - regime_metrics: métricas por régimen
      - best_global: mejor resultado global por Sharpe
      - best_by_regime: mejor resultado por régimen por Sharpe
      - top10: top 10 resultados por Sharpe
    """
    regime_metrics: Dict[int, List[Dict]] = {s: [] for s in range(n_states)}

    for result in results:
        trades_by_regime: Dict[int, List[TradeRecord]] = {s: [] for s in range(n_states)}
        for t in result.trades:
            r = t.regime_at_entry
            if r in trades_by_regime:
                trades_by_regime[r].append(t)

        for s in range(n_states):
            if trades_by_regime[s]:
                regime_metrics[s].append({
                    "params": result.params,
                    "trades": trades_by_regime[s],
                    "num_trades": len(trades_by_regime[s]),
                    "metrics": compute_metrics(trades_by_regime[s], result.equity_curve, INITIAL_CAPITAL),
                })

    # Construir tabla por régimen
    regime_summaries = {}
    for s in range(n_states):
        if regime_metrics[s]:
            df_r = pd.DataFrame([m["metrics"] for m in regime_metrics[s]])
            if not df_r.empty:
                best_idx = df_r["sharpe"].idxmax() if df_r["sharpe"].max() > -np.inf else df_r.index[0]
                regime_summaries[s] = {
                    "all_metrics": df_r,
                    "best": regime_metrics[s][best_idx],
                }

    # Mejor global
    all_flat = []
    for i, r in enumerate(results):
        all_flat.append({
            "index": i,
            "params": r.params,
            "trades": r.trades,
            "metrics": r.metrics,
        })
    df_all = pd.DataFrame([x["metrics"] for x in all_flat])
    if not df_all.empty and "sharpe" in df_all.columns:
        best_global_idx = df_all["sharpe"].idxmax() if df_all["sharpe"].max() > -np.inf else 0
        best_global = all_flat[best_global_idx]
        top10_idx = df_all["sharpe"].nlargest(min(10, len(df_all))).index.tolist()
        top10 = [all_flat[i] for i in top10_idx]
    else:
        best_global = all_flat[0] if all_flat else None
        top10 = all_flat[:10]

    return {
        "regime_metrics": regime_metrics,
        "regime_summaries": regime_summaries,
        "best_global": best_global,
        "top10": top10,
        "all_results": all_flat,
        "df_all": df_all,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 11) ANÁLISIS MULTI-ACTIVO / MULTI-TIMEFRAME
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MultiAssetEntry:
    asset: str
    timeframe: str
    df: pd.DataFrame
    hmm_result: Optional[HMMResult]
    hmm_summary: pd.DataFrame
    sweep_results: List[BacktestResult]
    classification: Dict[str, Any]
    warnings_list: List[str]
    hybrid_alert_data: Optional[Dict[str, Any]] = None


def build_multi_asset_summary(entries: List[MultiAssetEntry]) -> pd.DataFrame:
    """Construye tabla resumen multi-activo y multi-timeframe."""
    rows = []
    for e in entries:
        if e.classification and e.classification.get("best_global"):
            bg = e.classification["best_global"]
            m = bg["metrics"]
            n_states = e.hmm_result.n_states if e.hmm_result else 0
            rows.append({
                "Asset": e.asset,
                "Timeframe": e.timeframe,
                "HMM_States": n_states,
                "Best_Sharpe": round(m.get("sharpe", 0), 3),
                "Net_Profit": round(m.get("net_profit", 0), 0),
                "Profit_Factor": round(m.get("profit_factor", 0), 3),
                "Num_Trades": int(m.get("num_trades", 0)),
                "CAGR": round(m.get("cagr", 0) * 100, 2),
                "Max_DD": round(m.get("max_dd", 0) * 100, 2),
                "Win_Rate": round(m.get("win_rate", 0) * 100, 2),
            })
    return pd.DataFrame(rows)


def compute_combined_equity(
    entries: List[MultiAssetEntry],
    weights: Optional[List[float]] = None,
) -> pd.Series:
    """
    Combina equity curves de múltiples activos/timeframes con ponderación.
    Por defecto: equiponderada entre sleeves válidos.
    """
    valid_curves = []
    for e in entries:
        if e.sweep_results and e.classification.get("best_global"):
            bg = e.classification["best_global"]
            idx = bg["index"]
            if idx < len(e.sweep_results):
                valid_curves.append(e.sweep_results[idx].equity_curve)

    if not valid_curves:
        return pd.Series(dtype=float)

    if weights is None:
        weights = [1.0 / len(valid_curves)] * len(valid_curves)

    # Alinear por índice (estandarizar timezone para evitar tz-naive vs tz-aware)
    curves_tz_naive = []
    for c in valid_curves:
        if hasattr(c.index, 'tz') and c.index.tz is not None:
            curves_tz_naive.append(pd.Series(c.values, index=c.index.tz_localize(None)))
        else:
            curves_tz_naive.append(c)
    combined = pd.DataFrame({f"eq_{i}": c for i, c in enumerate(curves_tz_naive)})
    combined = combined.dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    weighted = combined.mul(weights, axis=1).sum(axis=1)
    return weighted


def compute_specialist_vs_universalist(entries: List[MultiAssetEntry]) -> pd.DataFrame:
    """
    Analiza qué activo/timeframe es más 'especialista' vs 'universalista'.
    Especialista = alta dispersión de rendimiento entre regímenes (se comporta
    muy diferente en cada régimen).
    Universalista = baja dispersión (similar en todos los regímenes).
    """
    rows = []
    for e in entries:
        if not e.classification or not e.classification.get("regime_summaries"):
            continue
        rs = e.classification["regime_summaries"]
        regime_sharpes = []
        for s, summ in rs.items():
            bm = summ["best"]["metrics"]
            regime_sharpes.append(bm.get("sharpe", 0))
        if regime_sharpes:
            dispersion = np.std(regime_sharpes)
            rows.append({
                "Asset": e.asset,
                "Timeframe": e.timeframe,
                "regime_sharpe_dispersion": dispersion,
                "mean_regime_sharpe": np.mean(regime_sharpes),
                "num_regimes": len(regime_sharpes),
                "type": "ESPECIALISTA" if dispersion > 0.5 else "UNIVERSALISTA",
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 12) FIGURAS PLOTLY
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = ["#2ECC40", "#3498DB", "#FF851B", "#FF4136", "#B10DC9"]

# Plantilla oscura personalizada
DARK_TEMPLATE = "plotly_dark"


def _add_warning_note(fig: go.Figure, text: str = "") -> None:
    """Añade anotación de advertencia."""
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        text=text,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="center",
    )


def fig_trade_signals(df: pd.DataFrame, trades: List[TradeRecord], asset: str, timeframe: str) -> go.Figure:
    """
    Gráfico de precio con marcadores de ENTRADAS y SALIDAS LONG/SHORT.
    - Triángulo verde ▲ = Entrada LONG
    - Triángulo rojo ▼ = Entrada SHORT
    - Círculo verde ● = Salida con ganancia
    - Círculo rojo ● = Salida con pérdida
    """
    fig = go.Figure()

    # Precio (línea) - más clara
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="rgba(255,255,255,0.5)", width=1.2),
        hovertemplate="Fecha: %{x}<br>Close: $%{y:.2f}<extra></extra>",
    ))

    # EMAs del mejor combo (si existen)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_fast"],
            mode="lines", name="EMA Fast",
            line=dict(color="rgba(46, 204, 64, 0.4)", width=1, dash="dash"),
            hovertemplate="EMA Fast: $%{y:.2f}<extra></extra>",
        ))
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_slow"],
            mode="lines", name="EMA Slow",
            line=dict(color="rgba(255, 65, 54, 0.4)", width=1, dash="dash"),
            hovertemplate="EMA Slow: $%{y:.2f}<extra></extra>",
        ))

    # Fondo por régimen HMM
    if "regime" in df.columns:
        y_range = df["Close"].max() - df["Close"].min()
        y_min = df["Close"].min()
        n_unique = df["regime"].nunique()
        for s in sorted(df["regime"].unique()):
            mask = df["regime"] == s
            idx = df.index[mask]
            if len(idx) > 0:
                fig.add_trace(go.Scatter(
                    x=idx,
                    y=[y_min + (s / max(n_unique, 1)) * y_range * 0.05] * len(idx),
                    mode="markers",
                    marker=dict(
                        color=REGIME_COLORS[s % len(REGIME_COLORS)],
                        size=3, opacity=0.3,
                        symbol="square",
                    ),
                    name=f"Régimen {s}",
                    showlegend=False,
                    hovertemplate=f"Régimen {s}<extra></extra>",
                ))

    # LONG entries (triángulo verde hacia arriba)
    long_trades = [t for t in trades if t.side == "LONG"]
    if long_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_date for t in long_trades],
            y=[t.entry_price for t in long_trades],
            mode="markers",
            name="🟢 LONG Entry",
            marker=dict(
                symbol="triangle-up",
                size=18,
                color="#2ECC40",
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                "<b>🟢 LONG ENTRY</b><br>"
                "Fecha: %{x}<br>"
                "Precio: $%{y:.2f}<br>"
                "Stop: $%{customdata[0]:.2f}<br>"
                "Target: $%{customdata[1]:.2f}<br>"
                "Régimen: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=[[t.stop_price, t.target_price, t.regime_at_entry] for t in long_trades],
        ))

    # SHORT entries (triángulo rojo hacia abajo)
    short_trades = [t for t in trades if t.side == "SHORT"]
    if short_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_date for t in short_trades],
            y=[t.entry_price for t in short_trades],
            mode="markers",
            name="🔴 SHORT Entry",
            marker=dict(
                symbol="triangle-down",
                size=18,
                color="#FF4136",
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                "<b>🔴 SHORT ENTRY</b><br>"
                "Fecha: %{x}<br>"
                "Precio: $%{y:.2f}<br>"
                "Stop: $%{customdata[0]:.2f}<br>"
                "Target: $%{customdata[1]:.2f}<br>"
                "Régimen: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=[[t.stop_price, t.target_price, t.regime_at_entry] for t in short_trades],
        ))

    # Exits - ganadores
    win_exits = [t for t in trades if t.pnl > 0]
    if win_exits:
        fig.add_trace(go.Scatter(
            x=[t.exit_date for t in win_exits],
            y=[t.exit_price for t in win_exits],
            mode="markers",
            name="✅ Exit Win",
            marker=dict(
                symbol="circle",
                size=14,
                color="#2ECC40",
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>✅ WIN EXIT</b><br>"
                "Fecha: %{x}<br>"
                "Precio: $%{y:.2f}<br>"
                "PnL: +$%{customdata[0]:.2f}<br>"
                "Motivo: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=[[t.pnl, t.exit_reason] for t in win_exits],
        ))

    # Exits - perdedores
    loss_exits = [t for t in trades if t.pnl <= 0]
    if loss_exits:
        fig.add_trace(go.Scatter(
            x=[t.exit_date for t in loss_exits],
            y=[t.exit_price for t in loss_exits],
            mode="markers",
            name="❌ Exit Loss",
            marker=dict(
                symbol="x",
                size=14,
                color="#FF4136",
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>❌ LOSS EXIT</b><br>"
                "Fecha: %{x}<br>"
                "Precio: $%{y:.2f}<br>"
                "PnL: $%{customdata[0]:.2f}<br>"
                "Motivo: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=[[t.pnl, t.exit_reason] for t in loss_exits],
        ))

    # Líneas de stop loss y target (solo para el primer trade de cada lado como ejemplo)
    example_trades = []
    seen_sides = set()
    for t in trades:
        if t.side not in seen_sides:
            example_trades.append(t)
            seen_sides.add(t.side)

    fig.update_layout(
        title=f"<b>Trade Signals - {asset} @ {timeframe}</b>",
        template=DARK_TEMPLATE,
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0.5)",
        ),
        margin=dict(t=80),
    )
    fig.update_yaxes(title_text="Precio ($)", tickprefix="$")
    fig.update_xaxes(title_text="Fecha", rangeslider_visible=False)

    return fig


def fig_trades_pnl(trades: List[TradeRecord], title: str = "PnL por Operación") -> go.Figure:
    """
    Gráfico de barras del PnL de cada operación en orden cronológico.
    Verde = ganancia, Rojo = pérdida.
    """
    if not trades:
        return go.Figure()

    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    colors = ["#2ECC40" if t.pnl > 0 else "#FF4136" for t in sorted_trades]
    total_pnl = sum(t.pnl for t in sorted_trades)
    wins = sum(1 for t in sorted_trades if t.pnl > 0)
    losses = sum(1 for t in sorted_trades if t.pnl <= 0)
    win_rate = wins / len(sorted_trades) * 100 if sorted_trades else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(sorted_trades) + 1)),
        y=[t.pnl for t in sorted_trades],
        marker_color=colors,
        marker_line=dict(width=1, color="white"),
        hovertemplate=(
            "<b>Trade #%{x}</b><br>"
            "Side: %{customdata[0]}<br>"
            "Entry: %{customdata[1]}<br>"
            "Exit: %{customdata[2]}<br>"
            "PnL: $%{y:.2f}<br>"
            "Return: %{customdata[3]:.2f}%<br>"
            "Motivo: %{customdata[4]}<br>"
            "<extra></extra>"
        ),
        customdata=[
            [t.side, t.entry_date.strftime("%Y-%m-%d"),
             t.exit_date.strftime("%Y-%m-%d"), t.pnl_pct, t.exit_reason]
            for t in sorted_trades
        ],
    ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.5)
    fig.add_hline(y=np.mean([t.pnl for t in sorted_trades]),
                  line_color="yellow", line_dash="dot", opacity=0.6,
                  annotation_text=f"Avg: ${np.mean([t.pnl for t in sorted_trades]):.0f}",
                  annotation_position="right")

    fig.update_layout(
        title=f"<b>{title}</b><br><sub>Total: ${total_pnl:,.0f} | Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)</sub>",
        template=DARK_TEMPLATE,
        height=350,
        xaxis_title="Trade #",
        yaxis_title="PnL ($)",
        hovermode="x",
        showlegend=False,
        margin=dict(t=80),
    )
    return fig


def fig_trades_cumulative(trades: List[TradeRecord]) -> go.Figure:
    """
    Gráfico de equity acumulada basada en trades (suma secuencial de PnL).
    """
    if not trades:
        return go.Figure()

    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    cumulative = np.cumsum([t.pnl for t in sorted_trades]) + INITIAL_CAPITAL
    dates = [t.exit_date for t in sorted_trades]

    colors = ["#2ECC40" if t.pnl > 0 else "#FF4136" for t in sorted_trades]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative,
        mode="lines+markers",
        name="Equity (Trades)",
        line=dict(color="#3498DB", width=2.5),
        marker=dict(
            color=colors,
            size=8,
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "Fecha: %{x}<br>"
            "Equity: $%{y:,.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Capital Inicial: ${INITIAL_CAPITAL:,.0f}",
                  annotation_position="right")

    fig.update_layout(
        title="<b>Equity Acumulada (Trades)</b>",
        template=DARK_TEMPLATE,
        height=350,
        yaxis_title="Capital ($)",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def fig_regime_timeline(df: pd.DataFrame, states: np.ndarray, asset: str, timeframe: str) -> go.Figure:
    """Timeline de regímenes con precio y banda de color."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    # Precio
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close",
                   line=dict(color="white", width=1.5)),
        row=1, col=1,
    )

    # Bandas de régimen
    state_colors = [REGIME_COLORS[s % len(REGIME_COLORS)] for s in states]
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[s * (df["Close"].max() - df["Close"].min()) / (max(states) + 1 if max(states) > 0 else 1) + df["Close"].min() for s in states],
            mode="markers",
            marker=dict(color=state_colors, size=4, opacity=0.6),
            name="Régimen",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Barras de régimen abajo
    for s in np.unique(states):
        mask = states == s
        idx = df.index[mask]
        if len(idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=[s] * len(idx),
                    mode="markers",
                    marker=dict(color=REGIME_COLORS[s % len(REGIME_COLORS)], size=6, opacity=0.8),
                    name=f"Estado {s}",
                ),
                row=2, col=1,
            )

    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="Estado", row=2, col=1, tickmode="linear", dtick=1)
    fig.update_layout(
        title=f"Regímenes de Mercado - {asset} @ {timeframe}",
        template=DARK_TEMPLATE,
        height=500,
        hovermode="x unified",
    )
    return fig


def fig_regime_pie(state_summary: pd.DataFrame) -> go.Figure:
    """Pie chart de distribución de regímenes."""
    colors = [REGIME_COLORS[s % len(REGIME_COLORS)] for s in state_summary["state"]]
    fig = go.Figure(
        go.Pie(
            labels=[f"Estado {s}" for s in state_summary["state"]],
            values=state_summary["pct_time"],
            marker=dict(colors=colors),
            textinfo="label+percent",
            hole=0.4,
        )
    )
    fig.update_layout(
        title="Distribución de Regímenes",
        template=DARK_TEMPLATE,
        height=400,
    )
    return fig


def fig_transition_heatmap(transition_matrix: np.ndarray) -> go.Figure:
    """Heatmap de matriz de transición."""
    n = transition_matrix.shape[0]
    labels = [f"Estado {i}" for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=transition_matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            text=np.round(transition_matrix, 3),
            texttemplate="%{text}",
            hovertemplate="Desde %{y} → Hasta %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz de Transición de Regímenes",
        template=DARK_TEMPLATE,
        height=400,
        xaxis_title="Hasta",
        yaxis_title="Desde",
    )
    return fig


def fig_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """Equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                             name="Equity", line=dict(color="#3498DB", width=2)))
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        template=DARK_TEMPLATE,
        height=400,
        yaxis_title="Capital",
        hovermode="x unified",
    )
    return fig


def fig_equity_by_regime(
    dfs_by_regime: Dict[int, pd.Series],
) -> go.Figure:
    """Equity curves separadas por régimen de entrada."""
    fig = go.Figure()
    for s, eq in sorted(dfs_by_regime.items()):
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            name=f"Régimen {s}",
            line=dict(color=REGIME_COLORS[s % len(REGIME_COLORS)], width=2),
        ))
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Equity Curves por Régimen de Entrada",
        template=DARK_TEMPLATE,
        height=400,
        yaxis_title="Capital",
        hovermode="x unified",
    )
    return fig


def fig_sharpe_heatmap(
    df_results: pd.DataFrame,
    x_param: str = "adx_threshold",
    y_param: str = "atr_stop_mult",
) -> go.Figure:
    """Heatmap de Sharpe para dos parámetros."""
    if df_results.empty or x_param not in df_results.columns or y_param not in df_results.columns:
        return go.Figure()

    pivot = df_results.pivot_table(index=y_param, columns=x_param, values="sharpe", aggfunc="mean")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            hovertemplate=f"{x_param}=%{{x}}, {y_param}=%{{y}}<br>Sharpe=%{{z:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Sharpe: {x_param} vs {y_param}",
        template=DARK_TEMPLATE,
        height=400,
        xaxis_title=x_param,
        yaxis_title=y_param,
    )
    return fig


def fig_scatter_combinations(df_results: pd.DataFrame) -> go.Figure:
    """Scatter de combinaciones: trades vs Sharpe, tamaño = Net Profit, color = Profit Factor."""
    if df_results.empty:
        return go.Figure()
    fig = go.Figure(
        go.Scatter(
            x=df_results["num_trades"],
            y=df_results["sharpe"],
            mode="markers",
            marker=dict(
                size=np.clip(np.abs(df_results["net_profit"]) / 1000, 5, 30),
                color=df_results["profit_factor"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Profit Factor"),
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=[f"Sharpe: {s:.2f}<br>Trades: {t}<br>Net: {n:.0f}<br>PF: {pf:.2f}"
                  for s, t, n, pf in zip(df_results["sharpe"], df_results["num_trades"],
                                         df_results["net_profit"], df_results["profit_factor"])],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Combinaciones: Trades vs Sharpe",
        template=DARK_TEMPLATE,
        height=500,
        xaxis_title="Nº Trades",
        yaxis_title="Sharpe",
    )
    return fig


def fig_regime_comparison_across_assets(entries: List[MultiAssetEntry]) -> go.Figure:
    """Comparativa de distribución de regímenes entre activos/timeframes."""
    rows = []
    for e in entries:
        if e.hmm_result is not None and not e.hmm_summary.empty:
            for _, r in e.hmm_summary.iterrows():
                rows.append({
                    "label": f"{e.asset} @ {e.timeframe}",
                    "state": int(r["state"]),
                    "pct": r["pct_time"],
                })
    if not rows:
        return go.Figure()

    df_plot = pd.DataFrame(rows)
    fig = px.bar(
        df_plot, x="label", y="pct", color="state",
        color_discrete_sequence=REGIME_COLORS,
        barmode="stack",
        labels={"label": "Activo @ Timeframe", "pct": "% del Tiempo", "state": "Régimen"},
    )
    fig.update_layout(
        title="Distribución de Regímenes por Activo/Timeframe",
        template=DARK_TEMPLATE,
        height=400,
        hovermode="x unified",
    )
    return fig


def fig_combined_equity(combined_eq: pd.Series) -> go.Figure:
    """Equity combinada multi-activo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_eq.index, y=combined_eq.values, mode="lines",
        name="Equity Combinada", line=dict(color="#2ECC40", width=2.5),
    ))
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Equity Combinada Multi-Activo (50/50 BTC/XRP por defecto)",
        template=DARK_TEMPLATE,
        height=400,
        yaxis_title="Capital",
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 13) TABLAS HTML
# ──────────────────────────────────────────────────────────────────────────────


def _df_to_html_table(df: pd.DataFrame, classes: str = "") -> str:
    """Convierte DataFrame a tabla HTML con formato."""
    return df.to_html(classes=f"table table-striped table-bordered {classes}", escape=False,
                      float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))


def _build_trades_table(trades: List[TradeRecord]) -> str:
    """
    Genera una tabla HTML con el detalle completo de todas las operaciones.
    """
    if not trades:
        return "<p>No hay trades para mostrar.</p>"

    rows = []
    for i, t in enumerate(trades, 1):
        pnl_class = "pnl-positive" if t.pnl > 0 else "pnl-negative"
        side_icon = "🟢" if t.side == "LONG" else "🔴"
        rows.append(f"""
        <tr class="{'trade-win' if t.pnl > 0 else 'trade-loss'}">
            <td>{i}</td>
            <td>{side_icon} {t.side}</td>
            <td>{t.entry_date.strftime('%Y-%m-%d %H:%M')}</td>
            <td>{t.exit_date.strftime('%Y-%m-%d %H:%M')}</td>
            <td>${t.entry_price:,.2f}</td>
            <td>${t.exit_price:,.2f}</td>
            <td>${t.stop_price:,.2f}</td>
            <td>${t.target_price:,.2f}</td>
            <td>{t.bars_in_trade}</td>
            <td class="{pnl_class}">${t.pnl:+,.2f}</td>
            <td class="{pnl_class}">{t.pnl_pct:+.2f}%</td>
            <td>R{t.regime_at_entry}</td>
            <td>{t.exit_reason}</td>
        </tr>
        """)

    # Métricas resumen arriba de la tabla
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if losses > 0 else 0
    best_trade = max(trades, key=lambda x: x.pnl)
    worst_trade = min(trades, key=lambda x: x.pnl)

    summary_html = f"""
    <div class="trades-summary">
        <div class="summary-card">
            <span class="summary-label">Total Trades</span>
            <span class="summary-value">{len(trades)}</span>
        </div>
        <div class="summary-card">
            <span class="summary-label">Win Rate</span>
            <span class="summary-value green">{wins/len(trades)*100:.1f}%</span>
            <span class="summary-sub">({wins}W / {losses}L)</span>
        </div>
        <div class="summary-card">
            <span class="summary-label">Total PnL</span>
            <span class="summary-value {'green' if total_pnl > 0 else 'red'}">${total_pnl:+,.0f}</span>
        </div>
        <div class="summary-card">
            <span class="summary-label">Avg Win / Avg Loss</span>
            <span class="summary-value">${avg_win:+,.0f} / ${avg_loss:+,.0f}</span>
        </div>
        <div class="summary-card">
            <span class="summary-label">Best Trade</span>
            <span class="summary-value green">${best_trade.pnl:+,.2f}</span>
            <span class="summary-sub">{best_trade.side} ({best_trade.exit_date.strftime('%Y-%m-%d')})</span>
        </div>
        <div class="summary-card">
            <span class="summary-label">Worst Trade</span>
            <span class="summary-value red">${worst_trade.pnl:+,.2f}</span>
            <span class="summary-sub">{worst_trade.side} ({worst_trade.exit_date.strftime('%Y-%m-%d')})</span>
        </div>
    </div>
    """

    # Desglose LONG vs SHORT
    long_trades = [t for t in trades if t.side == "LONG"]
    short_trades = [t for t in trades if t.side == "SHORT"]
    breakdown_rows = ""
    for label, side_trades in [("LONG", long_trades), ("SHORT", short_trades)]:
        if side_trades:
            side_pnl = sum(t.pnl for t in side_trades)
            side_wins = sum(1 for t in side_trades if t.pnl > 0)
            breakdown_rows += f"""
            <tr>
                <td>{'🟢' if label == 'LONG' else '🔴'} {label}</td>
                <td>{len(side_trades)}</td>
                <td>{side_wins/len(side_trades)*100:.1f}%</td>
                <td class="{'green' if side_pnl > 0 else 'red'}">${side_pnl:+,.0f}</td>
                <td>{sum(1 for t in side_trades if t.exit_reason == 'TAKE_PROFIT')} TP / {sum(1 for t in side_trades if t.exit_reason == 'STOP_LOSS')} SL</td>
            </tr>
            """

    breakdown_html = f"""
    <h4 style="margin-top:1rem;">Desglose LONG vs SHORT</h4>
    <table class="small-table">
        <thead><tr><th>Side</th><th>Trades</th><th>Win Rate</th><th>PnL</th><th>Salidas</th></tr></thead>
        <tbody>{breakdown_rows}</tbody>
    </table>
    """ if breakdown_rows else ""

    table_html = f"""
    <div style="overflow-x: auto;">
        {summary_html}
        {breakdown_html}
        <h4 style="margin-top:1rem;">Detalle de Operaciones</h4>
        <table class="table table-striped table-bordered small-table trades-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Side</th>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Stop</th>
                    <th>Target</th>
                    <th>Bars</th>
                    <th>PnL</th>
                    <th>Return</th>
                    <th>Régimen</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """
    return table_html

def _build_hybrid_alert_html(hybrid_data: Optional[Dict[str, Any]]) -> str:
    """Genera HTML con el estado del sistema de alerta hibrida HMM+Precursor."""
    if hybrid_data is None:
        return '<div class="hybrid-section"><p style="color: var(--text-muted);">Sistema hibrido no disponible.</p></div>'
    if "error" in hybrid_data:
        err_msg = hybrid_data["error"]
        s = '<div class="hybrid-section"><p style="color: #FF4136;">⚠️ Error: ' + str(err_msg) + '</p></div>'
        return s

    active = hybrid_data.get("alert_active", False)
    conf_long = hybrid_data.get("conf_long", 0.0)
    conf_short = hybrid_data.get("conf_short", 0.0)
    alert_long = hybrid_data.get("alert_long", False)
    alert_short = hybrid_data.get("alert_short", False)
    alerts_total = hybrid_data.get("alerts_total", 0)
    max_conf_long = hybrid_data.get("max_conf_long", 0.0)
    max_conf_short = hybrid_data.get("max_conf_short", 0.0)

    # Badge de estado
    if alert_long:
        badge = '<span class="hybrid-badge hybrid-badge-long">🟢 ALERTA LONG ACTIVA</span>'
    elif alert_short:
        badge = '<span class="hybrid-badge hybrid-badge-short">🔴 ALERTA SHORT ACTIVA</span>'
    elif active:
        badge = '<span class="hybrid-badge hybrid-badge-warn">⚠️ ALERTA HIBRIDA ACTIVA</span>'
    else:
        badge = '<span class="hybrid-badge hybrid-badge-off">⏸️ SIN ALERTA</span>'

    # Barras de confianza
    long_pct = min(conf_long / 100.0 * 100, 100)
    short_pct = min(conf_short / 100.0 * 100, 100)

    # Alert badges
    long_alert_badge = '<span class="hybrid-active-badge long">🟢 ALERTA</span>' if alert_long else ''
    short_alert_badge = '<span class="hybrid-active-badge short">🔴 ALERTA</span>' if alert_short else ''

    # Direction
    if conf_long > conf_short:
        direction = "LONG 🟢"
    elif conf_short > conf_long:
        direction = "SHORT 🔴"
    else:
        direction = "NEUTRAL ⚪"
    max_conf = max(conf_long, conf_short)
    conf_color = "#2ECC40" if max_conf > 50 else "#FF851B"

    bars = ""
    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza LONG</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(long_pct) + '%;background:#2ECC40;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_long) + '/100</span>'
    bars += "</div>"
    bars += long_alert_badge
    bars += "</div>"

    bars += '<div class="hybrid-kpi">'
    bars += '<span class="hybrid-kpi-label">Confianza SHORT</span>'
    bars += '<div class="hybrid-bar-track">'
    bars += '<div class="hybrid-bar-fill" style="width:' + str(short_pct) + '%;background:#FF4136;"></div>'
    bars += '<span class="hybrid-bar-label">' + str(conf_short) + '/100</span>'
    bars += "</div>"
    bars += short_alert_badge
    bars += "</div>"

    html = ""
    html += '<div class="hybrid-section">'
    html += '<div class="hybrid-header">'
    html += '<span class="hybrid-icon">🧬</span>'
    html += '<h3 style="margin:0;color:var(--text-primary);">SISTEMA HIBRIDO: HMM + Precursores</h3>'
    html += badge
    html += "</div>"
    html += '<div class="hybrid-kpi-grid">'
    html += bars
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Direccion</span>'
    html += '<span class="hybrid-kpi-value">' + direction + "</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Max Confianza</span>'
    html += '<span class="hybrid-kpi-value" style="color:' + conf_color + ';">' + str(max_conf) + "/100</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Alertas Historicas</span>'
    html += '<span class="hybrid-kpi-value">' + str(alerts_total) + "</span>"
    html += "</div>"
    html += '<div class="hybrid-kpi">'
    html += '<span class="hybrid-kpi-label">Mejor LONG / SHORT</span>'
    html += '<span class="hybrid-kpi-value" style="font-size:0.85rem;">🟢' + str(max_conf_long) + " / 🔴" + str(max_conf_short) + "</span>"
    html += "</div>"
    html += "</div>"
    html += "</div>"
    return html

def _build_hmm_table(hmm_summary: pd.DataFrame) -> str:
    """Tabla resumen HMM."""
    if hmm_summary.empty:
        return "<p>No hay datos HMM.</p>"
    return _df_to_html_table(hmm_summary.round(4))


def _build_best_params_table(
    classification: Dict[str, Any], n_states: int
) -> Tuple[str, str, str]:
    """Tablas de mejores parámetros."""
    best_global = classification.get("best_global")
    regime_summaries = classification.get("regime_summaries", {})
    top10 = classification.get("top10", [])

    # Global
    global_html = ""
    if best_global:
        params = best_global["params"]
        metrics = best_global["metrics"]
        rows = []
        for k, v in params.__dict__.items():
            rows.append({"Parámetro": k, "Valor": str(v)})
        for k, v in metrics.items():
            rows.append({"Parámetro": k, "Valor": f"{v:.4f}" if isinstance(v, float) else str(v)})
        global_html = _df_to_html_table(pd.DataFrame(rows), "small-table")

    # Por régimen
    regime_html = ""
    for s in range(n_states):
        if s in regime_summaries:
            best = regime_summaries[s]["best"]
            rows = []
            for k, v in best["params"].__dict__.items():
                rows.append({"Parámetro": k, "Valor": str(v)})
            for k, v in best["metrics"].items():
                rows.append({"Parámetro": k, "Valor": f"{v:.4f}" if isinstance(v, float) else str(v)})
            regime_html += f"<h4>Régimen {s}</h4>"
            regime_html += _df_to_html_table(pd.DataFrame(rows), "small-table")
            regime_html += f"<p>Trades en régimen: {best['num_trades']}</p><br>"

    # Top 10
    top10_html = ""
    if top10:
        rows = []
        for i, item in enumerate(top10):
            p = item["params"]
            m = item["metrics"]
            rows.append({
                "#": i + 1,
                "EMA Fast": p.ema_fast,
                "EMA Slow": p.ema_slow,
                "ADX Thresh": p.adx_threshold,
                "Release": p.release_lookback,
                "ATR Stop Mult": p.atr_stop_mult,
                "RR Target": p.rr_target,
                "VP Lookback": p.vp_lookback,
                "Vol Filter": p.use_volume_filter,
                "Sharpe": round(m.get("sharpe", 0), 3),
                "Net Profit": round(m.get("net_profit", 0), 0),
                "Trades": int(m.get("num_trades", 0)),
            })
        top10_html = _df_to_html_table(pd.DataFrame(rows), "small-table")

    return global_html, regime_html, top10_html


# ──────────────────────────────────────────────────────────────────────────────

# LIVE SIGNAL & REGIME CHARACTERIZATION


def compute_live_signal(
    df: pd.DataFrame,
    params: StrategyParams,
    use_volume_filter: bool = True,
) -> Dict[str, Any]:
    """
    Computa la señal LONG/SHORT/FLAT en la Última vela (la más reciente)
    usando los mejores parámetros del sweep.
    Retorna dict con todas las condiciones y la señal final.
    """
    if df is None or len(df) < 2:
        return {"signal": "FLAT", "reason": "Datos insuficientes", "components": {}}

    df = df.copy()
    
    # Calcular EMAs con los parámetros del sweep
    df["ema_fast"] = _ema(df["Close"], params.ema_fast)
    df["ema_slow"] = _ema(df["Close"], params.ema_slow)
    
    # Condiciones (vectoriales y luego tomar la última fila)
    df["bull_bias"] = (df["Close"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"])
    df["bear_bias"] = (df["Close"] < df["ema_slow"]) & (df["ema_fast"] < df["ema_slow"])
    
    df["squeeze_released"] = (
        df["squeeze_on"]
        .rolling(window=params.release_lookback, min_periods=1)
        .max()
        .shift(1)
        .fillna(False)
        .astype(bool)
    )
    
    signal_long = (
        df["bull_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] > 0)
        & (df["smi_delta"] > 0)
        & (df["adx"] > params.adx_threshold)
        & (df["adx_delta"] > 0)
    )
    signal_short = (
        df["bear_bias"]
        & df["squeeze_off"]
        & df["squeeze_released"]
        & (df["smi_hist"] < 0)
        & (df["smi_delta"] < 0)
        & (df["adx"] > params.adx_threshold)
        & (df["adx_delta"] > 0)
    )
    
    if use_volume_filter and "volume_bias_bull" in df.columns:
        signal_long = signal_long & df["volume_bias_bull"]
        signal_short = signal_short & ~df["volume_bias_bull"]
    
    # Última fila
    last = df.iloc[-1]
    
    components = {
        "close": float(last["Close"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "bull_bias": bool(last["bull_bias"]),
        "bear_bias": bool(last["bear_bias"]),
        "squeeze_off": bool(last["squeeze_off"]),
        "squeeze_released": bool(last["squeeze_released"]),
        "smi_hist": float(last["smi_hist"]),
        "smi_delta": float(last["smi_delta"]),
        "adx": float(last["adx"]),
        "adx_threshold": float(params.adx_threshold),
        "adx_delta": float(last["adx_delta"]),
        "volume_bias_bull": bool(last.get("volume_bias_bull", True)),
    }
    
    signal = signal_long.iloc[-1] or signal_short.iloc[-1]
    side = "FLAT"
    if signal_long.iloc[-1]:
        side = "LONG"
    elif signal_short.iloc[-1]:
        side = "SHORT"
    
    # Calcular fuerza de la señal (0-100)
    strength = 0
    if side == "LONG":
        # Más fuerte si smi_hist es muy positivo y adx muy alto
        strength = min(100, int(
            20  # bull_bias
            + 20  # squeeze_off + released
            + min(30, max(0, (components["smi_hist"] + 10) * 2))  # smi_hist positivo
            + min(30, max(0, (components["adx"] - components["adx_threshold"]) * 3))  # adx alto
        ))
    elif side == "SHORT":
        strength = min(100, int(
            20  # bear_bias
            + 20  # squeeze_off + released
            + min(30, max(0, (-components["smi_hist"] + 10) * 2))  # smi_hist negativo
            + min(30, max(0, (components["adx"] - components["adx_threshold"]) * 3))  # adx alto
        ))
    
    return {
        "signal": side,
        "strength": strength,
        "components": components,
        "latest_date": str(df.index[-1]),
        "params": {
            "ema_fast": params.ema_fast,
            "ema_slow": params.ema_slow,
            "adx_threshold": params.adx_threshold,
            "release_lookback": params.release_lookback,
            "atr_stop_mult": params.atr_stop_mult,
            "rr_target": params.rr_target,
            "use_volume_filter": params.use_volume_filter,
        },
    }




def _build_regime_characterization(
    hmm_summary: pd.DataFrame,
    classification: Dict[str, Any],
    n_states: int,
    df: pd.DataFrame,
) -> str:
    """
    Genera una sección HTML con la caracterización detallada de cada régimen.
    Incluye: nivel de volatilidad, retorno medio, bias direccional, win rate,
    trades, Sharpe, etiqueta descriptiva.
    """
    if hmm_summary.empty:
        return "<p>No hay datos HMM para caracterizar.</p>"
    
    # Obtener métricas por régimen desde classification
    regime_summaries = classification.get("regime_summaries", {})
    
    # Calcular bias direccional (Close > ema_slow) desde el df
    close = df["Close"] if not df.empty else pd.Series(dtype=float)
    
    cards = []
    for _, row in hmm_summary.iterrows():
        s = int(row["state"])
        vol = row.get("volatility", 0)
        mean_ret = row.get("mean_return", 0)
        pct_time = row.get("pct_time", 0)
        mean_dur = row.get("mean_duration_bars", 0)
        
        # Etiqueta descriptiva basada en volatilidad
        if vol < 0.005:
            vol_label = "🟢 Muy Baja"
            vol_desc = "Mercado quieto, spreads estrechos"
        elif vol < 0.01:
            vol_label = "🟡 Baja"
            vol_desc = "Movimientos moderados, tendencias suaves"
        elif vol < 0.02:
            vol_label = "🟠 Media"
            vol_desc = "Volatilidad normal, oportunidades medias"
        elif vol < 0.04:
            vol_label = "🔴 Alta"
            vol_desc = "Movimientos bruscos, riesgo elevado"
        else:
            vol_label = "💥 Muy Alta"
            vol_desc = "Picos de volatilidad, máximo riesgo"
        
        # Bias direccional
        if mean_ret > 0.0005:
            bias_label = "🟢 Alcista"
            bias_pct = min(100, max(0, (mean_ret / 0.005) * 100))
        elif mean_ret < -0.0005:
            bias_label = "🔴 Bajista"
            bias_pct = min(100, max(0, (-mean_ret / 0.005) * 100))
        else:
            bias_label = "🟡 Neutral"
            bias_pct = 50
        
        # Métricas de trading en este régimen
        if s in regime_summaries:
            best = regime_summaries[s]["best"]
            metrics = best["metrics"]
            trades_count = best["num_trades"]
            regime_win_rate = metrics.get("win_rate", 0) * 100
            regime_sharpe = metrics.get("sharpe", 0)
            regime_profit = metrics.get("net_profit", 0)
        else:
            trades_count = 0
            regime_win_rate = 0
            regime_sharpe = 0
            regime_profit = 0
        
        # Color de la card basado en Sharpe
        if regime_sharpe > 1.0:
        
            perf_color = "#2ECC40"
            perf_label = "🟢 EXCELENTE"
        elif regime_sharpe > 0.5:
            perf_color = "#3498DB"
            perf_label = "🔵 BUENA"
        elif regime_sharpe > 0:
            perf_color = "#FF851B"
            perf_label = "🟠 MODERADA"
        else:
            perf_color = "#FF4136"
            perf_label = "🔴 DEFICITARIA"
        
        card = f"""
        <div class="regime-card" style="border-left: 4px solid {REGIME_COLORS[s % len(REGIME_COLORS)]};">
            <div class="regime-card-header">
                <span class="regime-dot" style="background: {REGIME_COLORS[s % len(REGIME_COLORS)]};"></span>
                <span class="regime-title">Régimen {s}</span>
                <span class="regime-badge" style="background: {perf_color};">{perf_label}</span>
            </div>
            <div class="regime-card-body">
                <div class="regime-metrics-row">
                    <div class="regime-metric">
                        <span class="regime-metric-label">Volatilidad</span>
                        <span class="regime-metric-value">{vol_label}</span>
                        <span class="regime-metric-desc">{vol_desc}</span>
                    </div>
                    <div class="regime-metric">
                        <span class="regime-metric-label">Bias Direccional</span>
                        <span class="regime-metric-value">{bias_label}</span>
                        <div class="bias-bar">
                            <div class="bias-bar-fill" style="width: {bias_pct:.0f}%; background: {'#2ECC40' if mean_ret > 0 else '#FF4136'};"></div>
                        </div>
                    </div>
                    <div class="regime-metric">
                        <span class="regime-metric-label">% del Tiempo</span>
                        <span class="regime-metric-value">{pct_time:.1f}%</span>
                        <span class="regime-metric-desc">Duración media: {mean_dur:.0f} velas</span>
                    </div>
                </div>
                <div class="regime-metrics-row">
                    <div class="regime-metric">
                        <span class="regime-metric-label">Trades en Régimen</span>
                        <span class="regime-metric-value">{trades_count}</span>
                    </div>
                    <div class="regime-metric">
                        <span class="regime-metric-label">Win Rate</span>
                        <span class="regime-metric-value" style="color: {'#2ECC40' if regime_win_rate > 50 else '#FF4136'};">{regime_win_rate:.1f}%</span>
                    </div>
                    <div class="regime-metric">
                        <span class="regime-metric-label">Sharpe</span>
                        <span class="regime-metric-value" style="color: {perf_color};">{regime_sharpe:.2f}</span>
                    </div>
                    <div class="regime-metric">
                        <span class="regime-metric-label">Net Profit</span>
                        <span class="regime-metric-value" style="color: {'#2ECC40' if regime_profit > 0 else '#FF4136'};">
${regime_profit:+,.0f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        cards.append(card)
    
    return "".join(cards)



# 14) EXPORTACIÓN A HTML
# ──────────────────────────────────────────────────────────────────────────────


def _warning_banner_html(warnings_list: List[str]) -> str:
    """Genera bloque de warnings HTML."""
    if not warnings_list:
        return ""
    items = "".join(f"<li>{w}</li>" for w in warnings_list)
    return f"""
    <div class="warning-box">
        <h3>⚠️ Advertencias</h3>
        <ul>{items}</ul>
    </div>
    """


def _config_table_html() -> str:
    """Tabla de configuración usada."""
    config_items = [
        ("Assets", ", ".join(ASSETS)),
        ("Timeframes", ", ".join(TIMEFRAMES)),
        ("Initial Capital", f"${INITIAL_CAPITAL:,.0f}"),
        ("Commission", f"{COMMISSION_PCT:.2%}"),
        ("Slippage", f"{SLIPPAGE_PCT:.2%}"),
        ("HMM States Tested", ", ".join(str(s) for s in HMM_STATE_RANGE)),
        ("HMM Covariance", HMM_COVARIANCE_TYPE),
        ("Feature Window", str(FEATURE_WINDOW)),
        ("Min Trades Filter", str(MIN_TRADES_FILTER)),
        ("Random State", str(RANDOM_STATE)),
    ]
    rows = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in config_items)
    return f"<table class='config-table'>{rows}</table>"


def _build_live_signal_html(live_signals: Optional[Dict[str, Dict[str, Any]]]) -> str:
    '''Construye el HTML para el banner de senial en vivo.'''
    if not live_signals:
        return ""
    sections = []
    for key, signal in sorted(live_signals.items()):
        sig = signal.get("signal", "FLAT")
        strength = signal.get("strength", 0)
        reason = signal.get("reason", "")
        latest_date = signal.get("latest_date", "")
        components = signal.get("components", {})
        # Determinar clase CSS y emoji
        if sig == "LONG":
            cls = "LONG"
            emoji = "🟢"
            bar_color = "#2ECC40"
        elif sig == "SHORT":
            cls = "SHORT"
            emoji = "\U0001f534"
            bar_color = "#FF4136"
        else:
            cls = "FLAT"
            emoji = "\U0001f7e1"
            bar_color = "#FF851B"
        # Construir detalles de componentes
        comp_details = ""
        for k, v in components.items():
            if isinstance(v, float):
                comp_details += f'<span class="signal-params"><b>{k}:</b> {v:.2f}</span>'
            else:
                comp_details += f'<span class="signal-params"><b>{k}:</b> {v}</span>'
        sections.append(f'''
    <div class="live-signal-banner signal-{sig}">
        <div class="signal-indicator {cls}">{emoji} {sig}</div>
        <div class="signal-details">
            <div class="signal-detail-item">
                <span class="signal-detail-label">Activo / TF</span>
                <span class="signal-detail-value">{key}</span>
            </div>
            <div class="signal-detail-item">
                <span class="signal-detail-label">Senial</span>
                <span class="signal-detail-value">{sig}</span>
            </div>
            <div class="signal-detail-item">
                <span class="signal-detail-label">Confianza</span>
                <span class="signal-detail-value">{strength}%</span>
            </div>
            <div class="signal-detail-item">
                <span class="signal-detail-label">Ultima Vela</span>
                <span class="signal-detail-value">{latest_date}</span>
            </div>
        </div>
        <div style="flex:1;">
            <div class="signal-strength-bar">
                <div class="signal-strength-fill" style="width:{strength}%;background:{bar_color};"></div>
            </div>
            <div style="font-size:0.8rem;color:#aaa;text-align:center;">Confianza: {strength}%</div>
            <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{reason}</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-top:0.3rem;">{comp_details}</div>
        </div>
    </div>''')
    return "\n".join(sections)


def _build_regime_sections_html(regime_characterizations: Optional[Dict[str, str]]) -> str:
    '''Construye el HTML con la caracterizacion de regimenes por activo.'''
    if not regime_characterizations:
        return ""
    sections = []
    for key, regime_html in sorted(regime_characterizations.items()):
        # Extraer titulo del key
        sections.append(f'''
    <div style="margin-bottom:1.5rem;">
        <h3 style="color:#a0c4ff;margin-bottom:0.5rem;">📊 {key}</h3>
        {regime_html}
    </div>''')
    return "\n".join(sections)


def export_dashboard(
    entries: List[MultiAssetEntry],
    summary_df: pd.DataFrame,
    combined_equity: pd.Series,
    specialist_df: pd.DataFrame,
    global_warnings: List[str],
    output_path: str = OUTPUT_HTML,
    live_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    regime_characterizations: Optional[Dict[str, str]] = None,
) -> None:
    """
    Genera y exporta el dashboard HTML completo.
    """
    print(f"\n  [DASHBOARD] Generando {output_path} ...")

    # Generar figuras por cada entry
    all_figures: List[Tuple[str, str]] = []  # (html_div, title)

    for e in entries:
        if e.df is None or e.df.empty:
            continue

        asset_time_label = f"{e.asset} @ {e.timeframe}"

        # Sección HMM
        if e.hmm_result is not None:
            all_figures.append((fig_regime_timeline(e.df, e.hmm_result.states, e.asset, e.timeframe).to_html(full_html=False, include_plotlyjs=False),
                                f"Timeline - {asset_time_label}"))
            all_figures.append((fig_regime_pie(e.hmm_summary).to_html(full_html=False, include_plotlyjs=False),
                                f"Distribución - {asset_time_label}"))
            all_figures.append((fig_transition_heatmap(e.hmm_result.transition_matrix).to_html(full_html=False, include_plotlyjs=False),
                                f"Transición - {asset_time_label}"))

        # Trade Signals (ENTRADAS Y SALIDAS en el precio)
        if e.classification.get("best_global"):
            bg = e.classification["best_global"]
            idx = bg["index"]
            if idx < len(e.sweep_results):
                best_result = e.sweep_results[idx]
                trades = best_result.trades
                if trades:
                    # Gráfico de señales en el precio
                    all_figures.append((fig_trade_signals(e.df, trades, e.asset, e.timeframe).to_html(full_html=False, include_plotlyjs=False),
                                        f"Trade Signals - {asset_time_label}"))
                    # PnL por operación
                    all_figures.append((fig_trades_pnl(trades, f"PnL por Operación - {asset_time_label}").to_html(full_html=False, include_plotlyjs=False),
                                        f"PnL Trades - {asset_time_label}"))
                    # Equity acumulada por trades
                    all_figures.append((fig_trades_cumulative(trades).to_html(full_html=False, include_plotlyjs=False),
                                        f"Equity Trades - {asset_time_label}"))

        # Equity curves
        if e.classification.get("best_global"):
            bg = e.classification["best_global"]
            idx = bg["index"]
            if idx < len(e.sweep_results):
                eq = e.sweep_results[idx].equity_curve
                all_figures.append((fig_equity_curve(eq, f"Equity Curve (Backtest) - {asset_time_label}").to_html(full_html=False, include_plotlyjs=False),
                                    f"Equity - {asset_time_label}"))

        # Equity por régimen
        regime_eqs: Dict[int, pd.Series] = {}
        for s, summ in e.classification.get("regime_summaries", {}).items():
            # Construir equity por régimen combinando trades
            trades_by_r = [t for result in e.sweep_results for t in result.trades if t.regime_at_entry == s]
            if trades_by_r:
                # Equity sintética: sumar PnL de trades en orden
                sorted_trades = sorted(trades_by_r, key=lambda x: x.exit_date)
                eq_values = [INITIAL_CAPITAL]
                eq_dates = [pd.Timestamp(0)]
                for t in sorted_trades:
                    eq_values.append(eq_values[-1] + t.pnl)
                    eq_dates.append(t.exit_date)
                regime_eqs[s] = pd.Series(eq_values[1:], index=eq_dates[1:])

        if regime_eqs:
            all_figures.append((fig_equity_by_regime(regime_eqs).to_html(full_html=False, include_plotlyjs=False),
                                f"Equity por Régimen - {asset_time_label}"))

        # Heatmaps y scatter
        if e.classification.get("df_all") is not None and not e.classification["df_all"].empty:
            df_all = e.classification["df_all"]
            # Adjuntar parámetros
            params_list = [r.params for r in e.sweep_results]
            df_all["adx_threshold"] = [p.adx_threshold for p in params_list]
            df_all["atr_stop_mult"] = [p.atr_stop_mult for p in params_list]
            df_all["ema_fast"] = [p.ema_fast for p in params_list]
            df_all["ema_slow"] = [p.ema_slow for p in params_list]

            if len(df_all) > 1:
                all_figures.append((fig_sharpe_heatmap(df_all, "adx_threshold", "atr_stop_mult").to_html(full_html=False, include_plotlyjs=False),
                                    f"Sharpe ADX vs Stop - {asset_time_label}"))
                all_figures.append((fig_sharpe_heatmap(df_all, "ema_fast", "ema_slow").to_html(full_html=False, include_plotlyjs=False),
                                    f"Sharpe EMA - {asset_time_label}"))
                all_figures.append((fig_scatter_combinations(df_all).to_html(full_html=False, include_plotlyjs=False),
                                    f"Scatter - {asset_time_label}"))

    # Figuras globales
    if not combined_equity.empty:
        all_figures.append((fig_combined_equity(combined_equity).to_html(full_html=False, include_plotlyjs=False),
                            "Equity Combinada Multi-Activo"))
    if entries:
        all_figures.append((fig_regime_comparison_across_assets(entries).to_html(full_html=False, include_plotlyjs=False),
                            "Comparativa Regímenes"))

    # ── Construir HTML completo ──
    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.1.min.js" charset="utf-8"></script>'

    # Generar navegación
    nav_items = ""
    seen_labels = set()
    for _, title in all_figures:
        clean = title.lower().replace(" ", "-").replace(".", "-").replace("@", "")
        if clean not in seen_labels:
            seen_labels.add(clean)
            nav_items += f'<a href="#{clean}" class="nav-link">{title}</a>'

    # Secciones de figuras
    fig_sections = ""
    seen_sections = set()
    for fig_html, title in all_figures:
        clean = title.lower().replace(" ", "-").replace(".", "-").replace("@", "")
        if clean not in seen_sections:
            seen_sections.add(clean)
            fig_sections += f'''
            <div class="section" id="{clean}">
                <h2 class="section-title">{title}</h2>
                {fig_html}
            </div>
            '''

    # Tabla resumen multi-activo
    summary_table_html = _df_to_html_table(summary_df) if not summary_df.empty else "<p>No hay datos.</p>"

    # Tabla especialista/universalista
    specialist_table_html = _df_to_html_table(specialist_df) if not specialist_df.empty else "<p>No hay datos.</p>"

    # Tablas por activo/timeframe
    detail_sections = ""
    for e in entries:
        sid = f"details-{e.asset}-{e.timeframe}".replace(".", "-")
        n_states = e.hmm_result.n_states if e.hmm_result else 0
        glob_tbl, reg_tbl, top10_tbl = _build_best_params_table(e.classification, n_states)
        # Tabla de trades del mejor resultado
        trades_html = ""
        if e.classification.get("best_global"):
            bg = e.classification["best_global"]
            idx = bg["index"]
            if idx < len(e.sweep_results):
                trades_html = _build_trades_table(e.sweep_results[idx].trades)

        detail_sections += f"""
        <div class="section" id="{sid}">
            <h2 class="section-title">{e.asset} @ {e.timeframe} - Detalle</h2>
            <h3>Resumen HMM</h3>
            {_build_hmm_table(e.hmm_summary)}
            <h3>Mejor Combinación Global</h3>
            {glob_tbl}
            <h3>Mejores por Régimen</h3>
            {reg_tbl}
            <h3>Top 10 Global</h3>
            {top10_tbl}
            <h3>Operaciones del Mejor Resultado</h3>
            {trades_html}
            <h3>🧬 Alerta Híbrida HMM+Precursor</h3>
            {_build_hybrid_alert_html(e.hybrid_alert_data)}
        </div>
        """

    global_warnings_html = _warning_banner_html(global_warnings)

    # Generar HTML de senial en vivo y regimenes
    live_signal_html = ""
    regime_sections_html = ""

    if live_signals:
        live_signal_html = _build_live_signal_html(live_signals)
    if regime_characterizations:
        regime_sections_html = _build_regime_sections_html(regime_characterizations)

    # Advertencia obligatoria
    DISCLAIMER_HTML = """
    <div class="disclaimer-box">
        <h3>⚠️ Advertencia — Riesgo de Sobreoptimización</h3>
        <p>Este análisis es <strong>in-sample</strong> y susceptible a sobreoptimización.
        <strong>NO debe usarse para operar directamente.</strong></p>
        <p>El uso más correcto del HMM aquí es <strong>clasificar entornos de mercado</strong>
        y evaluar si la estrategia se comporta como <strong>universalista o especialista</strong>,
        para tomar decisiones a nivel de <strong>portfolio, activación/desactivación y sizing</strong>
        de estrategias, NO para asumir que optimizar parámetros por régimen generará ventaja
        robusta fuera de muestra.</p>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradingLatino HMM Dashboard</title>
{plotly_js}
<style>
    :root {{
        --bg-deep: #070b14;
        --bg-primary: #0a0f1e;
        --bg-secondary: #0f1629;
        --bg-card: #131b30;
        --bg-card-hover: #182240;
        --bg-nav: #0a0f1e;
        --bg-elevated: #1a2540;

        --border-color: #1e3050;
        --border-subtle: #16203a;
        --border-glow: rgba(59, 130, 246, 0.25);

        --text-primary: #f0f4f8;
        --text-secondary: #94a3b8;
        --text-muted: #546a8a;

        --blue: #3b82f6;
        --blue-light: #60a5fa;
        --blue-dark: #2563eb;
        --cyan: #06b6d4;
        --cyan-light: #22d3ee;
        --green: #22c55e;
        --green-light: #4ade80;
        --red: #ef4444;
        --red-light: #f87171;
        --orange: #f59e0b;
        --orange-light: #fbbf24;
        --purple: #8b5cf6;
        --pink: #ec4899;
        --teal: #14b8a6;

        --shadow-xs: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.35);
        --shadow-md: 0 4px 24px rgba(0,0,0,0.45);
        --shadow-lg: 0 8px 48px rgba(0,0,0,0.55);
        --shadow-glow-blue: 0 0 30px rgba(59, 130, 246, 0.1);
        --shadow-glow-green: 0 0 30px rgba(34, 197, 94, 0.1);
        --shadow-glow-red: 0 0 30px rgba(239, 68, 68, 0.1);

        --radius-xs: 4px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;

        --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
        background: var(--bg-deep);
        color: var(--text-primary);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        min-height: 100vh;
    }}

    /* ─── HEADER ─── */
    .header {{
        position: relative;
        padding: 2.5rem 1.5rem 1.5rem;
        text-align: center;
        background: linear-gradient(135deg, #080d1a 0%, #0f1629 30%, #0a1120 60%, #060a14 100%);
        border-bottom: 1px solid var(--border-color);
        overflow: hidden;
    }}
    .header::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 50%);
        pointer-events: none;
    }}
    .header::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 8%;
        right: 8%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--blue), var(--cyan), transparent);
        border-radius: 2px;
        opacity: 0.7;
    }}
    .header h1 {{
        position: relative;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.75px;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
    }}
    .header .subtitle {{
        position: relative;
        color: var(--text-secondary);
        font-size: 0.82rem;
        font-weight: 400;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.7;
    }}
    .header .subtitle strong {{
        color: var(--text-primary);
        font-weight: 500;
    }}
    .header-badge {{
        display: inline-block;
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(6,182,212,0.15));
        border: 1px solid rgba(59,130,246,0.2);
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.65rem;
        color: var(--blue-light);
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
        position: relative;
    }}

    /* ─── NAVIGATION ─── */
    .nav-bar {{
        background: linear-gradient(180deg, rgba(10, 15, 30, 0.98), rgba(10, 15, 30, 0.92));
        padding: 0.5rem 1.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.2rem;
        justify-content: center;
        border-bottom: 1px solid var(--border-subtle);
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    .nav-link {{
        color: var(--text-muted);
        text-decoration: none;
        padding: 0.4rem 0.9rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        font-weight: 500;
        transition: var(--transition);
        letter-spacing: 0.15px;
        position: relative;
    }}
    .nav-link:hover {{
        background: rgba(59, 130, 246, 0.08);
        color: var(--blue-light);
    }}
    .nav-link::after {{
        content: '';
        position: absolute;
        bottom: 2px;
        left: 50%;
        right: 50%;
        height: 2px;
        background: var(--blue);
        border-radius: 1px;
        transition: var(--transition);
        opacity: 0;
    }}
    .nav-link:hover::after {{
        left: 30%;
        right: 30%;
        opacity: 0.6;
    }}

    /* ─── CONTAINER ─── */
    .container {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem 1.8rem;
    }}

    /* ─── SECTIONS ─── */
    .section {{
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border-radius: var(--radius-lg);
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: var(--transition-slow);
        position: relative;
        overflow: hidden;
    }}
    .section::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
        opacity: 0;
        transition: var(--transition);
    }}
    .section:hover {{
        border-color: rgba(59, 130, 246, 0.25);
        box-shadow: var(--shadow-md), var(--shadow-glow-blue);
        transform: translateY(-1px);
    }}
    .section:hover::before {{
        opacity: 1;
    }}
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.2rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 0.6rem;
        letter-spacing: -0.2px;
    }}
    .section-subtitle {{
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }}

    /* ─── WARNING & DISCLAIMER BOXES ─── */
    .warning-box {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.06), rgba(239, 68, 68, 0.02));
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: var(--radius-md);
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        transition: var(--transition);
    }}
    .warning-box:hover {{
        border-color: rgba(239, 68, 68, 0.35);
        box-shadow: var(--shadow-glow-red);
    }}
    .warning-box h3 {{
        color: var(--red-light);
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .warning-box ul {{
        margin-left: 1.3rem;
        color: var(--text-secondary);
        font-size: 0.82rem;
    }}
    .disclaimer-box {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(245, 158, 11, 0.02));
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: var(--radius-md);
        padding: 1.2rem 1.5rem;
    }}
    .disclaimer-box:hover {{
        border-color: rgba(245, 158, 11, 0.3);
    }}
    .disclaimer-box h3 {{
        color: var(--orange-light);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}
    .disclaimer-box p {{
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.7;
        margin-bottom: 0.4rem;
    }}
    .disclaimer-box p:last-child {{ margin-bottom: 0; }}
    .disclaimer-box strong {{
        color: var(--text-primary);
    }}

    /* ─── TABLES ─── */
    table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 0.8rem 0;
        font-size: 0.8rem;
        border-radius: var(--radius-sm);
        overflow: hidden;
    }}
    th, td {{
        padding: 0.55rem 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
        text-align: left;
    }}
    th {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.03));
        color: var(--blue-light);
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        position: sticky;
        top: 0;
        z-index: 5;
        padding-top: 0.7rem;
        padding-bottom: 0.6rem;
    }}
    th:first-child {{ border-radius: var(--radius-sm) 0 0 0; }}
    th:last-child {{ border-radius: 0 var(--radius-sm) 0 0; }}
    tr:last-child td:first-child {{ border-radius: 0 0 0 var(--radius-sm); }}
    tr:last-child td:last-child {{ border-radius: 0 0 var(--radius-sm) 0; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{
        background: rgba(59, 130, 246, 0.04);
    }}
    tr:nth-child(even) td {{
        background: rgba(255,255,255,0.015);
    }}
    tr:nth-child(even):hover td {{
        background: rgba(59, 130, 246, 0.06);
    }}
    .small-table {{ font-size: 0.75rem; }}
    .config-table th, .config-table td {{
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
    }}

    /* ─── TYPOGRAPHY ─── */
    h2, h3, h4 {{
        color: var(--text-primary);
        margin: 1.2rem 0 0.5rem;
        font-weight: 600;
        letter-spacing: -0.2px;
    }}
    h2 {{ font-size: 1.1rem; }}
    h3 {{
        font-size: 0.92rem;
        background: linear-gradient(135deg, var(--cyan), var(--blue-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    h4 {{ font-size: 0.82rem; color: var(--text-secondary); }}

    /* ─── UTILITIES ─── */
    .text-warning {{ color: var(--orange-light); }}
    .text-success {{ color: var(--green-light); }}
    .text-danger {{ color: var(--red-light); }}
    .text-info {{ color: var(--cyan); }}
    .green {{ color: var(--green); }}
    .red {{ color: var(--red); }}
    .pnl-positive {{ color: var(--green); font-weight: 700; }}
    .pnl-negative {{ color: var(--red); font-weight: 700; }}

    /* ─── TRADES SUMMARY CARDS ─── */
    .trades-summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 1.2rem 0;
    }}
    .summary-card {{
        position: relative;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.04), rgba(6, 182, 212, 0.02));
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 0.9rem 0.8rem 0.7rem;
        text-align: center;
        transition: var(--transition);
        overflow: hidden;
    }}
    .summary-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--blue), transparent);
        opacity: 0;
        transition: var(--transition);
    }}
    .summary-card:hover {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(6, 182, 212, 0.04));
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: rgba(59, 130, 246, 0.2);
    }}
    .summary-card:hover::before {{ opacity: 0.6; }}
    .summary-label {{
        display: block;
        font-size: 0.58rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }}
    .summary-value {{
        display: block;
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}
    .summary-sub {{
        display: block;
        font-size: 0.62rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }}
    .trade-win {{ background: rgba(34, 197, 94, 0.06) !important; }}
    .trade-loss {{ background: rgba(239, 68, 68, 0.06) !important; }}
    .trades-table th {{
        white-space: nowrap;
        font-size: 0.62rem;
    }}

    /* ─── LIVE SIGNAL BANNER ─── */
    .live-signal-banner {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1.2rem 1.5rem;
        border-radius: var(--radius-lg);
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(10, 15, 30, 0.9), rgba(19, 27, 48, 0.9));
        border: 1.5px solid var(--border-color);
        transition: var(--transition-slow);
        position: relative;
        overflow: hidden;
    }}
    .live-signal-banner::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 50%, rgba(59,130,246,0.03), transparent);
        pointer-events: none;
    }}
    .live-signal-banner.signal-LONG {{
        border-color: rgba(34, 197, 94, 0.4);
        box-shadow: 0 0 40px rgba(34, 197, 94, 0.08), inset 0 0 40px rgba(34, 197, 94, 0.02);
    }}
    .live-signal-banner.signal-SHORT {{
        border-color: rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.08), inset 0 0 40px rgba(239, 68, 68, 0.02);
    }}
    .live-signal-banner.signal-FLAT {{
        border-color: rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 40px rgba(245, 158, 11, 0.06), inset 0 0 40px rgba(245, 158, 11, 0.02);
    }}

    .signal-indicator {{
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.7rem 1.5rem;
        border-radius: var(--radius-md);
        min-width: 130px;
        text-align: center;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
        flex-shrink: 0;
    }}
    .signal-indicator.LONG {{
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.03));
        color: var(--green-light);
        border: 1px solid rgba(34, 197, 94, 0.25);
        text-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
    }}
    .signal-indicator.SHORT {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.03));
        color: var(--red-light);
        border: 1px solid rgba(239, 68, 68, 0.25);
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }}
    .signal-indicator.FLAT {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.03));
        color: var(--orange-light);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }}

    .signal-indicator.LONG::after,
    .signal-indicator.SHORT::after {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        animation: spin-glow 3s linear infinite;
    }}
    .signal-indicator.LONG::after {{
        background: conic-gradient(transparent, rgba(34, 197, 94, 0.08), transparent 30%);
    }}
    .signal-indicator.SHORT::after {{
        background: conic-gradient(transparent, rgba(239, 68, 68, 0.08), transparent 30%);
    }}
    @keyframes spin-glow {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    .signal-details {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.3rem;
        flex: 1;
        position: relative;
        z-index: 1;
    }}
    .signal-detail-item {{
        text-align: center;
        padding: 0.3rem 0.4rem;
    }}
    .signal-detail-label {{
        font-size: 0.58rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.15rem;
    }}
    .signal-detail-value {{
        font-size: 0.88rem;
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.2px;
    }}
    .signal-strength-bar {{
        height: 4px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        margin: 0.5rem 0;
        overflow: hidden;
    }}
    .signal-strength-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 1s ease;
        background: linear-gradient(90deg, var(--orange), var(--green));
    }}
    .signal-strength-fill.short {{
        background: linear-gradient(90deg, var(--orange), var(--red));
    }}
    .signal-strength-fill.flat {{
        background: linear-gradient(90deg, var(--text-muted), var(--orange));
    }}
    .signal-params {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.2rem 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.68rem;
        color: var(--text-muted);
        justify-content: center;
        position: relative;
        z-index: 1;
    }}
    .signal-params span {{
        background: rgba(255,255,255,0.03);
        padding: 0.15rem 0.5rem;
        border-radius: var(--radius-xs);
    }}

    /* ─── REGIME CARDS ─── */
    .regime-cards-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 1rem;
        margin: 1.2rem 0;
    }}
    .regime-card {{
        background: linear-gradient(135deg, rgba(10, 15, 30, 0.8), rgba(19, 27, 48, 0.6));
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        overflow: hidden;
        transition: var(--transition-slow);
    }}
    .regime-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(59, 130, 246, 0.25);
    }}
    .regime-card-header {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.9rem 1rem;
        background: linear-gradient(135deg, rgba(0,0,0,0.3), rgba(0,0,0,0.15));
        border-bottom: 1px solid var(--border-subtle);
    }}
    .regime-dot {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
        box-shadow: 0 0 10px rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .regime-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        flex: 1;
    }}
    .regime-badge {{
        font-size: 0.58rem;
        font-weight: 600;
        color: #fff;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .regime-card-body {{
        padding: 0.9rem 1rem;
    }}
    .regime-metrics-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.5rem;
    }}
    .regime-metrics-row:last-child {{ margin-bottom: 0; }}
    .regime-metric {{
        flex: 1;
        min-width: 80px;
        padding: 0.35rem 0.45rem;
        background: rgba(0,0,0,0.15);
        border-radius: var(--radius-sm);
        text-align: center;
    }}
    .regime-metric-label {{
        display: block;
        font-size: 0.52rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.1rem;
    }}
    .regime-metric-value {{
        display: block;
        font-size: 0.92rem;
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.2px;
    }}
    .regime-metric-desc {{
        display: block;
        font-size: 0.6rem;
        color: var(--text-muted);
        margin-top: 0.05rem;
    }}
    .bias-bar {{
        height: 4px;
        background: rgba(255,255,255,0.04);
        border-radius: 2px;
        margin-top: 4px;
        overflow: hidden;
    }}
    .bias-bar-fill {{
        height: 100%;
        border-radius: 2px;
        transition: width 0.8s ease;
    }}

    /* ─── FOOTER ─── */
    .footer {{
        text-align: center;
        padding: 2rem 1.5rem 1.5rem;
        color: var(--text-muted);
        font-size: 0.72rem;
        border-top: 1px solid var(--border-subtle);
        line-height: 1.8;
        position: relative;
    }}
    .footer::before {{
        content: '';
        position: absolute;
        top: -1px;
        left: 20%;
        right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.15), transparent);
    }}
    .footer p {{ margin-bottom: 0.3rem; }}
    .footer p:last-child {{ margin-bottom: 0; }}

    /* ─── RESPONSIVE ─── */
    @media (max-width: 768px) {{
        .header {{ padding: 1.5rem 1rem 1.2rem; }}
        .header h1 {{ font-size: 1.3rem; }}
        .nav-bar {{
            flex-direction: column;
            align-items: center;
            gap: 0.15rem;
            padding: 0.4rem 1rem;
            position: static;
        }}
        .container {{ padding: 1rem; }}
        .section {{ padding: 1rem; border-radius: var(--radius-md); }}
        .trades-summary {{ grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }}
        .live-signal-banner {{ flex-direction: column; gap: 1rem; }}
        .signal-indicator {{ min-width: 100%; }}
        .regime-cards-container {{ grid-template-columns: 1fr; }}
        .signal-details {{ grid-template-columns: repeat(2, 1fr); }}
        .regime-metrics-row {{ flex-direction: column; }}
        table, th, td {{ font-size: 0.7rem; }}
        th, td {{ padding: 0.35rem 0.4rem; }}
    }}
    @media (max-width: 480px) {{
        .container {{ padding: 0.7rem; }}
        .trades-summary {{ grid-template-columns: 1fr; }}
        .header h1 {{ font-size: 1.1rem; }}
        .section-title {{ font-size: 0.95rem; }}
        .regime-cards-container {{ grid-template-columns: 1fr; }}
        .signal-details {{ grid-template-columns: 1fr; }}
    }}

    /* ─── SCROLLBAR ─── */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-deep); }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border-color);
        border-radius: 4px;
        border: 1px solid var(--bg-deep);
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

/* === HYBRID ALERT SYSTEM === */
.hybrid-section {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}}
.hybrid-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.hybrid-icon {{
    font-size: 1.5rem;
}}
.hybrid-badge {{
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.hybrid-badge-long {{
    background: rgba(46, 204, 64, 0.2);
    color: #2ECC40;
    border: 1px solid rgba(46, 204, 64, 0.4);
    animation: pulse-green 1.5s ease-in-out infinite;
}}
.hybrid-badge-short {{
    background: rgba(255, 65, 54, 0.2);
    color: #FF4136;
    border: 1px solid rgba(255, 65, 54, 0.4);
    animation: pulse-red 1.5s ease-in-out infinite;
}}
.hybrid-badge-warn {{
    background: rgba(255, 133, 27, 0.15);
    color: #FF851B;
    border: 1px solid rgba(255, 133, 27, 0.3);
}}
.hybrid-badge-off {{
    background: rgba(150, 150, 150, 0.15);
    color: var(--text-muted);
    border: 1px solid rgba(150, 150, 150, 0.2);
}}
@keyframes pulse-green {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(46, 204, 64, 0.4); }}
    50% {{ box-shadow: 0 0 0 6px rgba(46, 204, 64, 0); }}
}}
@keyframes pulse-red {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(255, 65, 54, 0.4); }}
    50% {{ box-shadow: 0 0 0 6px rgba(255, 65, 54, 0); }}
}}
.hybrid-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
}}
.hybrid-kpi {{
    background: rgba(0,0,0,0.15);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}}
.hybrid-kpi-label {{
    display: block;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}}
.hybrid-kpi-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}}
.hybrid-bar-track {{
    position: relative;
    height: 20px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
    margin: 0.3rem 0;
}}
.hybrid-bar-fill {{
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s ease;
    min-width: 4px;
}}
.hybrid-bar-label {{
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.65rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 0 4px rgba(0,0,0,0.8);
}}
.hybrid-active-badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-top: 0.3rem;
}}
.hybrid-active-badge.long {{
    background: rgba(46, 204, 64, 0.3);
    color: #2ECC40;
}}
.hybrid-active-badge.short {{
    background: rgba(255, 65, 54, 0.3);
    color: #FF4136;
}}

</style></style>
</head>
<body>
<div class="header">
    <div class="header-badge">⚡ Análisis Cuantitativo · Hidden Markov Model</div>
    <h1>📊 TradingLatino HMM Dashboard</h1>
    <div class="subtitle">
        <strong>Análisis de Regímenes de Mercado</strong> · Clasificación de entornos con HMM · Estrategia cuantificable<br>
        Generado: <strong>{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</strong>
    </div>
</div>

<div class="nav-bar">
    <a href="#resumen" class="nav-link">📋 Resumen</a>
    {nav_items}
    <a href="#live-signal" class="nav-link">🟢 Señal en Vivo</a>
    <a href="#regimenes" class="nav-link">📊 Regímenes</a>
    <a href="#disclaimer" class="nav-link">⚠️ Disclaimer</a>
</div>

<div class="container">

    <div class="section" id="resumen">
        <h2 class="section-title">📋 Configuración del Análisis</h2>
        {_config_table_html()}
        {global_warnings_html}
    </div>

    <div class="section">
        <h2 class="section-title">📊 Resumen Multi-Activo & Multi-Timeframe</h2>
        {summary_table_html}
    </div>

    <div class="section">
        <h2 class="section-title">🎯 Especialista vs Universalista</h2>
        <p class="section-subtitle">🎯 Dispersión del Índice de Sharpe entre regímenes. <strong style="color:var(--text-primary);">Mayor dispersión = estrategia más especialista</strong> (dependiente del régimen).</p>
        {specialist_table_html}
    </div>

    {fig_sections}

    {detail_sections}

    <!-- LIVE SIGNAL TRIGGER -->
    <div class="section" id="live-signal" style="display:{'' if live_signals else 'none'};">
        <h2 class="section-title">🟢 Señal en Vivo — Última Vela</h2>
        <p class="section-subtitle">Señal generada con los <strong style="color:var(--text-primary);">mejores parámetros del sweep</strong> × clasificación del régimen actual × condición direccional de la última vela.</p>
        {live_signal_html}
    </div>

    <!-- REGIME CHARACTERIZATION -->
    <div class="section" id="regimenes" style="display:{'' if regime_characterizations else 'none'};">
        <h2 class="section-title">📊 Caracterización de Regímenes HMM</h2>
        <p class="section-subtitle">Descripción detallada de cada régimen HMM: <strong style="color:var(--text-primary);">volatilidad, retornos, win rate, Índice de Sharpe y bias direccional</strong> por cada activo y timeframe.</p>
        {regime_sections_html}
    </div>

    <div class="section" id="disclaimer">
        {DISCLAIMER_HTML}
    </div>

    <div class="footer">
        <p>TradingLatino HMM Dashboard <span style="color:var(--border-color);">·</span> v2.0 <span style="color:var(--border-color);">·</span> Python <span style="color:var(--border-color);">·</span> pandas <span style="color:var(--border-color);">·</span> NumPy <span style="color:var(--border-color);">·</span> Plotly <span style="color:var(--border-color);">·</span> hmmlearn <span style="color:var(--border-color);">·</span> yfinance</p>
        <p>⚠️ Este software es solo para fines <strong style="color:var(--text-secondary);">educativos y de investigación</strong>. No constituye asesoramiento financiero ni recomendación de inversión.</p>
        <p style="margin-top:0.6rem;font-size:0.65rem;color:var(--border-color);">Built with ❤️ para la comunidad TradingLatino</p>
    </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [DASHBOARD] Exportado a: {output_path} ({len(html):,} bytes).")

    if OPEN_BROWSER:
        try:
            webbrowser.open(f"file://{Path(output_path).resolve()}")
            print(f"  [DASHBOARD] Abierto en navegador.")
        except Exception as e:
            print(f"  [DASHBOARD] No se pudo abrir navegador: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 15) MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Punto de entrada principal."""
    print("=" * 72)
    print("  TRADINGLATINO HMM DASHBOARD")
    print("  Análisis de Regímenes de Mercado con HMM")
    print("=" * 72)

    global_warnings: List[str] = []
    entries: List[MultiAssetEntry] = []

    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            print(f"\n{'─' * 72}")
            print(f"  PROCESANDO: {asset} @ {timeframe}")
            print(f"{'─' * 72}")

            # 1) Cargar datos
            df = load_data(asset, timeframe, USE_CSV, CSV_PATHS)
            if df is None or df.empty:
                msg = f"{asset} @ {timeframe}: sin datos suficientes. Se omite."
                print(f"  [SKIP] {msg}")
                global_warnings.append(msg)
                continue

            entry_warnings: List[str] = []
            if timeframe in ("1h", "4h") and len(df) < 500:
                msg = f"{asset} @ {timeframe}: solo {len(df)} velas intradía (yfinance limita a ~730 días)."
                print(f"  [WARN] {msg}")
                entry_warnings.append(msg)
                global_warnings.append(msg)

            print(f"  Datos: {len(df)} velas ({df.index[0].date()} → {df.index[-1].date()})")

            # 2) Auto-tune kc_mult (probar valores hasta encontrar el óptimo)
            tf_kc_mult = auto_tune_kc_mult(df, timeframe)
            print(f"\n  Calculando indicadores (kc_mult={tf_kc_mult})...")
            df = compute_indicators(df, kc_mult=tf_kc_mult)
            # Debug: contar squeeze, señales, etc.
            n_sq_on = int(df["squeeze_on"].sum())
            sq_rate = n_sq_on / max(len(df), 1) * 100
            print(f"  Squeeze ON: {n_sq_on}/{len(df)} ({sq_rate:.1f}% de velas)")

            # Debug: contar señales con emas por defecto
            if "signal_long" in df.columns and "signal_short" in df.columns:
                n_long = int(df["signal_long"].sum())
                n_short = int(df["signal_short"].sum())
                print(f"  Señales (EMAs por defecto): LONG={n_long}, SHORT={n_short}, total={n_long + n_short}")
            else:
                # Si compute_indicators no las generó, contarlas ahora con lógica equivalente
                bull = ((df["Close"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"])).sum()
                bear = ((df["Close"] < df["ema_slow"]) & (df["ema_fast"] < df["ema_slow"])).sum()
                print(f"  Bias: BULL={bull}, BEAR={bear} velas")

            # Debug: cuántas velas tendrían signal si solo exigieran squeeze_off + squeeze_released
            approx_released = (df["squeeze_off"] & df["squeeze_on"].rolling(3, min_periods=1).max().shift(1).fillna(False)).sum()
            print(f"  Cota superior (squeeze_off + squeeze_released): {approx_released} velas candidatas")
            if approx_released < 25:
                print(f"    -> Con este kc_mult, las señales candidatas son pocas.")

            # 3) HMM
            print("  Construyendo features para HMM ...")
            features_df = build_hmm_features(df)
            print(f"  Features: {features_df.shape[1]} variables, {features_df.dropna().shape[0]} filas sin NaN.")

            print("  Entrenando HMM (selección por BIC) ...")
            hmm_result, hmm_summary = fit_hmm_select_bic(features_df)
            if hmm_result is None:
                msg = f"{asset} @ {timeframe}: HMM no pudo entrenarse. Se omite backtest."
                print(f"  [SKIP] {msg}")
                global_warnings.append(msg)
                continue

            print(f"  Mejor modelo: {hmm_result.n_states} estados (BIC={hmm_result.bic:.2f}).")

            # 4) Reetiquetar estados
            print("  Reetiquetando estados por volatilidad ...")
            # Guardar estados originales ANTES de sobrescribirlos
            orig_model_states = hmm_result.states.copy()
            states_relabeled, state_summary = relabel_states_by_vol(
                hmm_result.model, hmm_result.states, features_df
            )
            hmm_result.states = states_relabeled
            hmm_result.state_summary = state_summary
            # Calcular matriz de transición reordenada según el relabel
            trans_mat = hmm_result.model.transmat_
            n = trans_mat.shape[0]
            # El relabel de relabel_states_by_vol ordena los estados por volatilidad.
            # Necesitamos saber qué estado ORIGINAL corresponde a cada estado NUEVO
            # para reordenar la matriz de transición.
            # Averiguamos el orden original de los estados según su volatilidad
            orig_vol_rank = np.argsort([
                np.nanstd(features_df["log_return_1"].values[:len(orig_model_states)][orig_model_states == s])
                if np.any(orig_model_states == s) else 0
                for s in range(n)
            ])
            # orig_vol_rank[i] = el estado original con el i-ésimo menor vol
            # Para reordenar: la nueva matriz T_new[i,j] = T_old[orig_vol_rank[i], orig_vol_rank[j]]
            trans_reordered = np.zeros_like(trans_mat)
            for i in range(n):
                for j in range(n):
                    trans_reordered[i, j] = trans_mat[orig_vol_rank[i], orig_vol_rank[j]]
            hmm_result.transition_matrix = trans_reordered

            print("  Resumen por estado:")
            print(state_summary.to_string(index=False))

            # 5a) Computar alerta hibrida HMM+Precursor
            hybrid_alert_data = None
            if _compute_signal_scores is not None and compute_precursor_signals is not None and compute_hybrid_alert is not None:
                try:
                    print("  Computando alerta hibrida HMM+Precursor...")
                    df_hybrid = df.copy()
                    # Calcular signal scores (necesarios para los precursores)
                    df_hybrid = _compute_signal_scores(df_hybrid)
                    # Calcular EMA deviation, RSI14, volumen ratio (necesarios para signal scores)
                    if "ema_slow" in df_hybrid.columns:
                        df_hybrid["ema_dev_pct"] = (df_hybrid["Close"] - df_hybrid["ema_slow"]) / df_hybrid["ema_slow"] * 100
                    # Calcular precursores
                    df_hybrid = compute_precursor_signals(df_hybrid)
                    # Calcular alerta hibrida
                    df_hybrid = compute_hybrid_alert(df_hybrid, hmm_result.states, state_summary)
                    # Extraer datos relevantes
                    hybrid_alert_data = {
                        "alert_active": bool(df_hybrid["hybrid_alert_active"].iloc[-1]) if "hybrid_alert_active" in df_hybrid.columns else False,
                        "conf_long": float(df_hybrid["hybrid_confidence_long"].iloc[-1]) if "hybrid_confidence_long" in df_hybrid.columns else 0,
                        "conf_short": float(df_hybrid["hybrid_confidence_short"].iloc[-1]) if "hybrid_confidence_short" in df_hybrid.columns else 0,
                        "alert_long": bool(df_hybrid["hybrid_alert_long"].iloc[-1]) if "hybrid_alert_long" in df_hybrid.columns else False,
                        "alert_short": bool(df_hybrid["hybrid_alert_short"].iloc[-1]) if "hybrid_alert_short" in df_hybrid.columns else False,
                        "alerts_total": int(df_hybrid["hybrid_alert_active"].sum()) if "hybrid_alert_active" in df_hybrid.columns else 0,
                        "max_conf_long": float(df_hybrid["hybrid_confidence_long"].max()) if "hybrid_confidence_long" in df_hybrid.columns else 0,
                        "max_conf_short": float(df_hybrid["hybrid_confidence_short"].max()) if "hybrid_confidence_short" in df_hybrid.columns else 0,
                    }
                    print(f"    Alerta hibrida activa: {hybrid_alert_data['alert_active']} (L={hybrid_alert_data['conf_long']:.0f} S={hybrid_alert_data['conf_short']:.0f})")
                    print(f"    Total alertas: {hybrid_alert_data['alerts_total']}")
                except Exception as _hybrid_err:
                    print(f"    [WARN] Error al computar alerta hibrida: {_hybrid_err}")
                    hybrid_alert_data = {"error": str(_hybrid_err)}
            else:
                print("    [SKIP] compute_hybrid_alert no disponible (importacion fallo)")

            # 5) Extender df con estados HMM (rellenar para todas las filas)
            full_states = np.full(len(df), -1, dtype=int)
            # features_df está alineado con df pero sin NaN
            clean_idx = features_df.dropna().index
            states_len = min(len(states_relabeled), len(clean_idx))
            for i, idx in enumerate(clean_idx[:states_len]):
                pos = df.index.get_loc(idx)
                if isinstance(pos, slice):
                    pos = pos.start
                if 0 <= pos < len(full_states):
                    full_states[pos] = states_relabeled[i]
            # Forward fill para NaN iniciales
            full_states_series = pd.Series(full_states).replace(-1, np.nan).ffill().fillna(0).astype(int).values
            # Almacenar el array completo (tamaño = len(df)) para que
            # fig_regime_timeline en export_dashboard tenga la misma longitud que df
            hmm_result.states = full_states_series

            # 6) Parameter sweep (con min_trades y kc_mult auto-tuneado)
            tf_min_trades = MIN_TRADES_BY_TF.get(timeframe, MIN_TRADES_FILTER)
            print(f"  Ejecutando parameter sweep (min_trades={tf_min_trades}, kc_mult={tf_kc_mult}) ...")
            sweep_results = sweep_parameters(df, PARAM_GRID, asset, timeframe,
                                              regimes=full_states_series,
                                              min_trades=tf_min_trades,
                                              kc_mult=tf_kc_mult)
            if not sweep_results:
                msg = f"{asset} @ {timeframe}: ninguna combinación supera el filtro de {tf_min_trades} trades. Se omite."
                print(f"  [SKIP] {msg}")
                global_warnings.append(msg)
                print(f"  [INFO] Revisa la salida del SWEEP para ver el máximo de trades encontrados y parámetros.")
                continue

            print(f"  Sweep completado: {len(sweep_results)} resultados.")

            # 7) Clasificar por régimen
            print("  Clasificando trades por régimen de entrada ...")
            classification = classify_trades_by_regime(sweep_results, hmm_result.n_states)
            print(f"  Clasificación completada.")

            # 8) Guardar entry
            entries.append(MultiAssetEntry(
                asset=asset,
                timeframe=timeframe,
                df=df,
                hmm_result=hmm_result,
                hmm_summary=state_summary,
                sweep_results=sweep_results,
                classification=classification,
                warnings_list=entry_warnings,
                hybrid_alert_data=hybrid_alert_data,
            ))
            print(f"  ✅ {asset} @ {timeframe} procesado correctamente.")

    if not entries:
        print("\n" + "=" * 72)
        print("  No se pudo procesar ningún activo/timeframe.")
        print("  Revisa los warnings anteriores.")
        print("=" * 72)
        sys.exit(1)

    # 9) Resumen multi-activo
    print(f"\n{'=' * 72}")
    print("  CONSTRUYENDO RESUMEN MULTI-ACTIVO ...")
    print(f"{'=' * 72}")

    summary_df = build_multi_asset_summary(entries)
    print("\n  Resumen Multi-Activo:")
    print(summary_df.to_string(index=False))

    # 10) Equity combinada
    print("\n  Calculando equity combinada ...")
    combined_eq = compute_combined_equity(entries)
    print(f"  Equity combinada: {len(combined_eq)} periodos.")

    # 11) Especialista vs Universalista
    print("  Analizando especialista vs universalista ...")
    specialist_df = compute_specialist_vs_universalist(entries)
    if not specialist_df.empty:
        print(specialist_df.to_string(index=False))

    # 12) Computar senial en vivo y caracterizacion de regimenes
    live_signals: Dict[str, Dict[str, Any]] = {}
    regime_characterizations: Dict[str, str] = {}
    for e_ in entries:
        if e_.df is None or e_.hmm_result is None:
            continue
        # Usar los mejores parametros del sweep
        best_params = StrategyParams()
        if e_.classification and e_.classification.get("best_global"):
            bg = e_.classification["best_global"]
            idx = bg.get("index", 0)
            if idx < len(e_.sweep_results):
                best_params = e_.sweep_results[idx].params
        try:
            signal = compute_live_signal(e_.df, best_params, use_volume_filter=True)
            live_signals[f"{e_.asset} @ {e_.timeframe}"] = signal
        except Exception as ex:
            print(f"  [WARN] No se pudo computar senial en vivo para {e_.asset} @ {e_.timeframe}: {ex}")
        try:
            if e_.classification:
                regime_html = _build_regime_characterization(
                    e_.hmm_result.hmm_summary if e_.hmm_result else pd.DataFrame(),
                    e_.classification,
                    e_.hmm_result.n_states if e_.hmm_result else 0,
                    e_.df,
                )
                if regime_html:
                    regime_characterizations[f"{e_.asset} @ {e_.timeframe}"] = regime_html
        except Exception as ex:
            print(f"  [WARN] No se pudo caracterizar regimenes para {e_.asset} @ {e_.timeframe}: {ex}")

    # 13) Exportar dashboard
    print(f"\n{'=' * 72}")
    print("  EXPORTANDO DASHBOARD ...")
    print(f"{'=' * 72}")

    export_dashboard(
        entries,
        summary_df,
        combined_eq,
        specialist_df,
        global_warnings,
        live_signals=live_signals or None,
        regime_characterizations=regime_characterizations or None,
    )

    # Disclaimer en consola
    print("\n" + "=" * 72)
    print("  ⚠️  ADVERTENCIA OBLIGATORIA")
    print("=" * 72)
    print("""
  Este análisis es in-sample y susceptible a sobreoptimización.
  NO debe usarse para operar directamente.

  El uso más correcto del HMM aquí es clasificar entornos de
  mercado y evaluar si la estrategia se comporta como universalista
  o especialista, para tomar decisiones a nivel de portfolio,
  activación/desactivación y sizing de estrategias, NO para asumir
  que optimizar parámetros por régimen generará ventaja robusta
  fuera de muestra.
    """)
    print("=" * 72)
    print("  ✅ Dashboard generado correctamente.")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
