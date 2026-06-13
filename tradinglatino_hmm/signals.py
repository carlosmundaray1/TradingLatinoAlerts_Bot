#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SEÑALES · TradingLatino HMM Regime Dashboard
================================================================================
Cálculo de señales en vivo, expiración (Jaime Merino) y verificación histórica.
================================================================================
"""
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tradinglatino_hmm.config import (
    ADX_THRESHOLD, DYNAMIC_THRESHOLD_MIN,
    SIGNAL_SCORE_THRESHOLD, MIN_CONSECUTIVE_BARS,
    MAX_BARS_BY_TF, TAKE_PROFIT_PCT, RELEASE_LOOKBACK,
)


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


def _find_signal_start(df: pd.DataFrame, signal_col: str) -> int:
    """Encuentra hace cuántas velas comenzó el bloque continuo de señal actual."""
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
    """Calcula la expiración de la señal según la regla de Jaime Merino."""
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
    }


def compute_signal(df: pd.DataFrame, timeframe: Optional[str] = None) -> Dict[str, Any]:
    """Computa la señal actual (última vela) con todas las condiciones."""
    last = df.iloc[-1]
    conditions = {
        "Tendencia Alcista (Bull Bias)": {
            "met": bool(last["bull_bias"]),
            "detail": f"Close ${last['Close']:.2f} > EMA_Slow ${last['ema_slow']:.2f} & EMA_Fast ${last['ema_fast']:.2f} > EMA_Slow"
        },
        "Tendencia Bajista (Bear Bias)": {
            "met": bool(last["bear_bias"]),
            "detail": f"Close ${last['Close']:.2f} < EMA_Slow ${last['ema_slow']:.2f} & EMA_Fast ${last['ema_fast']:.2f} < EMA_Slow"
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

    score_long = float(last.get("signal_score_long", 0))
    score_short = float(last.get("signal_score_short", 0))
    atr_series = df["atr"]
    atr_mean = atr_series.rolling(20, min_periods=1).mean().iloc[-1]
    atr_ratio = atr_series.iloc[-1] / atr_mean if atr_mean > 0 else 1.0
    dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, SIGNAL_SCORE_THRESHOLD - min(25, int(atr_ratio * 5)))

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
    }

    if timeframe:
        result["expiration"] = compute_expiration(df, result, timeframe)
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


def verify_signals_historically(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """
    Verifica históricamente si las señales LONG/SHORT se cumplieron
    usando un Take-Profit automático configurado por timeframe.
    """
    tp_target = TAKE_PROFIT_PCT.get(timeframe, TAKE_PROFIT_PCT.get("1d", 2.0))
    max_bars = MAX_BARS_BY_TF.get(timeframe, 14)

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
            high_prices = window["High"].values
            low_prices = window["Low"].values
            final_price = window["Close"].values[-1]
            max_price = high_prices.max()
            min_price = low_prices.min()
            max_return = (max_price - entry_price) / entry_price * 100.0
            min_return = (min_price - entry_price) / entry_price * 100.0
            final_return = (final_price - entry_price) / entry_price * 100.0
            bars_to_win: Optional[int] = None
            for j, cp in enumerate(window["Close"].values):
                if cp > entry_price:
                    bars_to_win = j + 1
                    break
            bars_to_max: Optional[int] = None
            for j in range(len(window["Close"].values)):
                if window["Close"].values[j] == max_price:
                    bars_to_max = j + 1
                    break
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
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
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
            max_return = (entry_price - min_price) / entry_price * 100.0
            min_return = (entry_price - max_price) / entry_price * 100.0
            final_return = (entry_price - final_price) / entry_price * 100.0
            bars_to_win: Optional[int] = None
            for j, cp in enumerate(close_prices):
                if cp < entry_price:
                    bars_to_win = j + 1
                    break
            bars_to_min: Optional[int] = None
            for j in range(len(close_prices)):
                if close_prices[j] == min_price:
                    bars_to_min = j + 1
                    break
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
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
            })

    # Estadísticas agregadas
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
                "wins": 0,
                "losses": 0,
                "recent_signals": 0,
                "recent_wins": 0,
                "recent_win_rate": None,
            }
            continue
        wins = sum(1 for r in results_list if r["won"])
        avg_ret = float(np.mean([r["final_return"] for r in results_list]))
        avg_max_fav = float(np.mean([r["max_return"] for r in results_list]))
        avg_max_adv = float(np.mean([r["min_return"] for r in results_list]))
        btws = [r["bars_to_win"] for r in results_list if r["bars_to_win"] is not None]
        avg_btw = float(np.mean(btws)) if btws else None
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
    """
    tp_target = TAKE_PROFIT_PCT.get(timeframe, TAKE_PROFIT_PCT.get("1d", 2.0))
    max_bars = MAX_BARS_BY_TF.get(timeframe, 14)

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

            max_price = high_prices.max()
            final_price = close_prices[-1]
            max_return_tp = (max_price - entry_price_f) / entry_price_f * 100.0
            won_tp = max_return_tp >= tp_target

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
            best_return_combined = max_return_tp if won_tp else exit_return_ts
            long_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price_f,
                "max_return_tp": round(max_return_tp, 2),
                "final_return_tp": round((final_price - entry_price_f) / entry_price_f * 100.0, 2),
                "won_tp": won_tp,
                "exit_return_ts": round(exit_return_ts, 2),
                "won_ts": won_ts,
                "won_combined": won_combined,
                "best_return_combined": round(best_return_combined, 2),
                "trail_activated": exit_idx is not None,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
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

            min_price = low_prices.min()
            final_price = close_prices[-1]
            max_return_tp = (entry_price_f - min_price) / entry_price_f * 100.0
            won_tp = max_return_tp >= tp_target

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
            best_return_combined = max_return_tp if won_tp else exit_return_ts
            short_results.append({
                "entry_date": df.index[i],
                "entry_price": entry_price_f,
                "max_return_tp": round(max_return_tp, 2),
                "final_return_tp": round((entry_price_f - final_price) / entry_price_f * 100.0, 2),
                "won_tp": won_tp,
                "exit_return_ts": round(exit_return_ts, 2),
                "won_ts": won_ts,
                "won_combined": won_combined,
                "best_return_combined": round(best_return_combined, 2),
                "trail_activated": exit_idx is not None,
                "regime": int(df["regime"].iloc[i]) if "regime" in df.columns else -1,
            })

    # Estadisticas Agregadas
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
                "wins_tp": 0,
                "wins_ts": 0,
                "wins_combined": 0,
                "trail_activated": 0,
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
