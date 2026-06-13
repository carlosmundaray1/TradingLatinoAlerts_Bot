#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CARGA DE DATOS · TradingLatino HMM Regime Dashboard
================================================================================
Descarga de datos OHLCV desde Yahoo Finance con retry + fallback.
================================================================================
"""
import contextlib
import os
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from tradinglatino_hmm.config import PERIOD_1H, PERIOD_4H, PERIOD_1D, PERIOD_1W


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
        # Intentar 1: yf.download
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

        # Intentar 3: requests directo a la API de Yahoo
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
            print(f"    {len(df)} velas descargadas.")
            return df
        if attempt < max_attempts:
            wait = attempt * 4
            print(f"    Intento {attempt} fallo, reintentando en {wait}s...")
            time.sleep(wait)
    print(f"  ERROR: No se pudieron descargar datos para {asset} ({timeframe}).")
    return None
