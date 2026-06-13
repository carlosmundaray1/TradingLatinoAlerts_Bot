#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HMM MODEL · TradingLatino HMM Regime Dashboard
================================================================================
Construcción de features para HMM, ajuste del modelo y descripción de regímenes.
================================================================================
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from hmmlearn import hmm

from tradinglatino_hmm.config import (
    HMM_STATE_RANGE, HMM_COVARIANCE_TYPE, FEATURE_WINDOW, REGIME_COLORS,
)
from tradinglatino_hmm.indicators import _ema


def _classify_regime_bias(description: str) -> str:
    """Clasifica un regimen como 'bullish', 'bearish', o 'neutral'."""
    d = description.upper()
    if any(x in d for x in ["ALCISTA", "EXPANSION ALCISTA", "TREND ALCISTA"]):
        return "bullish"
    if any(x in d for x in ["BAJISTA", "EXPANSION BAJISTA", "TREND BAJISTA"]):
        return "bearish"
    return "neutral"


def _describe_regime(state: int, vol: float, mean_ret: float) -> str:
    """Genera una descripción legible del régimen para crypto (BTC diario)."""
    # --- Regímenes extremos ---
    if vol >= 5.0:
        if mean_ret > 0.15:
            return "[EXPANSION ALCISTA]"
        elif mean_ret < -0.15:
            return "[EXPANSION BAJISTA]"
        else:
            return "[ALTA VOLATILIDAD]"

    # --- Trends fuertes con volatilidad elevada ---
    if vol >= 3.5:
        if mean_ret > 0.15:
            return "[TREND ALCISTA]"
        elif mean_ret < -0.15:
            return "[TREND BAJISTA]"
        else:
            return "[VOLATILIDAD NEUTRA]"

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
        if vol < 2.5:
            return "[ACUMULACION]"
        else:
            return "[LATERAL]"


def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features de mercado mejoradas para el HMM."""
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
    return features


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
