#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CONFIGURACIÓN · TradingLatino HMM Regime Dashboard
================================================================================
Constantes y parámetros de la estrategia TradingLatino.
================================================================================
"""
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# ACTIVO Y TEMPORALIDADES
# ──────────────────────────────────────────────────────────────────────────────
ASSET: str = "BTC-USD"
TIMEFRAMES: List[str] = ["1h", "4h", "1d", "1wk"]
OUTPUT_HTML: str = "hmm_regime_dashboard.html"
STATE_FILE: str = ".hmm_last_state.json"  # Estado anterior entre ejecuciones
OPEN_BROWSER: bool = True

# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS FIJOS DE LA ESTRATEGIA
# ──────────────────────────────────────────────────────────────────────────────
EMA_FAST: int = 10
EMA_SLOW: int = 55
ADX_THRESHOLD: float = 23.0
RELEASE_LOOKBACK: int = 3
ATR_STOP_MULT: float = 2.0
RR_TARGET: float = 2.0
USE_VOLUME_FILTER: bool = True

# ──────────────────────────────────────────────────────────────────────────────
# TAKE-PROFIT AUTOMÁTICO POR TIMEFRAME
# ──────────────────────────────────────────────────────────────────────────────
TAKE_PROFIT_PCT: Dict[str, float] = {
    "1h": 0.5,
    "4h": 1.2,   # WR=70.4% con 328 senales (threshold=68, cons=2)
    "1d": 2.0,
    "1wk": 2.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# TRAILING STOP
# ──────────────────────────────────────────────────────────────────────────────
TRAILING_STOP_PCT: Dict[str, float] = {
    "1h": 2.0,    # Optimo multi-objetivo: WR 73.5% | PF 3.48 | Sharpe 13.48
    "4h": 2.0,    # Optimo multi-objetivo: WR 73.2% | PF 5.55 | Sharpe 10.65
    "1d": 1.0,    # Optimo multi-objetivo: WR 79.5% | PF 16.26 | Sharpe 13.43
    "1wk": 1.0,   # Optimo multi-objetivo: WR 96.8% | PF ∞ | Sharpe 7.95
}

# ──────────────────────────────────────────────────────────────────────────────
# FILTRO MÍNIMO DE SEÑALES (Score Compuesto Ponderado + Confirmación Temporal)
# ──────────────────────────────────────────────────────────────────────────────
SIGNAL_SCORE_THRESHOLD: int = 65
MIN_CONSECUTIVE_BARS: int = 2

# Pesos del score compuesto
W_BULL_BIAS: int = 16
W_SQUEEZE_OFF: int = 15
W_SQUEEZE_REL: int = 8
W_SMI_HIST: int = 10
W_SMI_DELTA: int = 12
W_ADX_THRESH: int = 5
W_ADX_DELTA: int = 2
BONUS_ADX: int = 0

# Nuevos indicadores
W_EMA_DEV: int = 6
W_RSI: int = 4
W_VOLUME: int = 5
W_ATR_ROC: int = 2
RSI_LENGTH: int = 14
VOL_LOOKBACK: int = 20
ATR_ROC_PERIODS: int = 3
TP_ATR_MULT: float = 2.0
TRAIL_ATR_MULT: float = 1.5
DYNAMIC_THRESHOLD_MIN: int = 45

# ──────────────────────────────────────────────────────────────────────────────
# HMM
# ──────────────────────────────────────────────────────────────────────────────
HMM_STATE_RANGE: List[int] = [3, 4, 5]
HMM_COVARIANCE_TYPE: str = "diag"
RANDOM_STATE: int = 42
FEATURE_WINDOW: int = 20

# ──────────────────────────────────────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ──────────────────────────────────────────────────────────────────────────────
BB_LENGTH: int = 20
BB_STD: float = 2.0
KC_LENGTH: int = 20
KC_MULT: float = 2.0
ADX_LENGTH: int = 14
ATR_LENGTH: int = 14
VP_LOOKBACK: int = 75
VP_BINS: int = 24

# ──────────────────────────────────────────────────────────────────────────────
# PERIODOS DE DESCARGA
# ──────────────────────────────────────────────────────────────────────────────
PERIOD_1H: str = "90d"
PERIOD_4H: str = "180d"
PERIOD_1D: str = "2y"
PERIOD_1W: str = "4y"

# ──────────────────────────────────────────────────────────────────────────────
# REGLA DE EXPIRACIÓN DE JAIME MERINO
# ──────────────────────────────────────────────────────────────────────────────
MAX_BARS_BY_TF: Dict[str, int] = {
    "1h": 14,   # ~14 horas
    "4h": 12,   # ~48 horas
    "1d": 14,   # ~14 días
    "1wk": 12,  # ~3 meses
}

# ──────────────────────────────────────────────────────────────────────────────
# ASSETS POPULARES
# ──────────────────────────────────────────────────────────────────────────────
POPULAR_ASSETS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "MATIC-USD", "ATOM-USD",
    "UNI-USD", "AAVE-USD", "APT-USD", "SUI-USD",
]

# ──────────────────────────────────────────────────────────────────────────────
# COLORES DE REGÍMENES
# ──────────────────────────────────────────────────────────────────────────────
REGIME_COLORS = ["#089981", "#3498DB", "#FF851B", "#F23645", "#B10DC9", "#F012BE"]

# ──────────────────────────────────────────────────────────────────────────────
# NOMBRE DEL SCRIPT
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_NAME: str = "tradinglatino_hmm_clean.py"

# Labels de señal
SIGNAL_LABELS = {"LONG": "LONG", "SHORT": "SHORT", "FLAT": "SIN SENAL"}
