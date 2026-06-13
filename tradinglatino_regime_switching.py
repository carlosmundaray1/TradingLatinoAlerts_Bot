#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

MS_K_REGIMES = 2
MS_AR_ORDER = 1
MS_SWITCHING_VARIANCE = True
MS_SWITCHING_TREND = True
MS_MIN_OBS = 50
MS_MAX_ITER = 200
W_MS_REGIME = 10
MS_REGIME_COLORS = ['#F23645', '#089981', '#3498DB', '#FF851B']

def fit_markov_switching(df, k_regimes=2, order=1,
                          switching_variance=True, switching_trend=True,
                          max_iter=200):
    """Ajusta Markov Switching a los retornos del activo."""
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression
    )
    # Calcular retornos
    close = df['Close'] if 'Close' in df.columns else None
    if close is None:
        for c in df.columns:
            if 'close' in str(c).lower():
                close = df[c]; break
    if close is None:
        print('  [MS] ERROR: No se encuentra columna Close')
        return None, np.array([]), pd.DataFrame(), None, None

    returns = np.log(close / close.shift(1)).dropna().values
    if len(returns) < MS_MIN_OBS:
        print(f'  [MS] Datos insuficientes ({len(returns)} obs).')
        return None, np.array([]), pd.DataFrame(), None, None

    print(f'  [MS] Ajustando Markov Switching '
          f'(k={k_regimes}, AR({order}), N={len(returns)})...')

    best_fit = None
    best_states = None
    best_probs = None
    best_llf = -1e9
    best_model = None

    for sr in [5, 10, 3]:
        try:
            m = MarkovAutoregression(
                returns, k_regimes=k_regimes, order=order,
                switching_variance=switching_variance,
                switching_trend=switching_trend,
            )
            f = m.fit(maxiter=max_iter, search_reps=sr,
                      cov_type='robust', disp=False)
            if f.llf > best_llf:
                # Obtener probabilidades (puede ser DataFrame o array)
                raw_probs = f.smoothed_marginal_probabilities
                if hasattr(raw_probs, 'values'):
                    probs_array = raw_probs.values
                else:
                    probs_array = raw_probs
                if hasattr(probs_array, 'ndim') and probs_array.ndim == 2:
                    arr = np.argmax(probs_array, axis=1)
                elif hasattr(raw_probs, 'idxmax'):
                    arr = raw_probs.idxmax(axis=1).values
                else:
                    arr = np.argmax(np.array(raw_probs), axis=1)

                best_llf = f.llf
                best_fit = f
                best_model = m
                best_states = arr.copy()
                best_probs = raw_probs
                counts = np.bincount(arr.astype(int), minlength=k_regimes)
                print(f'  [MS] OK! LLF={f.llf:.2f}, '
                      f'dist: {counts.tolist()}')
        except Exception as e:
            print(f'  [MS] sr={sr}: {e}')
            continue

    if best_fit is None or best_states is None or len(best_states) == 0:
        print('  [MS] No se pudo ajustar.')
        return None, np.array([]), pd.DataFrame(), None, None

    # Reordenar: 0=bajista/retorno bajo, 1=alcista/retorno alto
    ra = returns[-len(best_states):]
    us = np.unique(best_states)
    smr = {}
    for s in us:
        mask = best_states == s
        smr[s] = float(np.nanmean(ra[mask])) if mask.sum() > 0 else 0.0
    ss = sorted(us, key=lambda x: smr.get(x, 0))
    rm = {o: n for n, o in enumerate(ss)}
    best_states = np.array([rm[s] for s in best_states])

    rows = []
    for s in range(len(us)):
        mask = best_states == s
        c = int(mask.sum())
        mr = float(np.nanmean(ra[mask])) if c > 0 else 0.0
        v = float(np.nanstd(ra[mask])) if c > 0 else 0.0
        runs = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        rst = np.where(runs == 1)[0]
        ren = np.where(runs == -1)[0]
        rl = ren - rst
        md = float(np.mean(rl)) if len(rl) > 0 else 0.0
        rows.append({
            'state': s,
            'pct_time': round(c / len(best_states) * 100, 1),
            'mean_return': round(mr * 100, 4),
            'volatility': round(v * 100, 4),
            'mean_duration_bars': round(md, 1),
            'description': _describe_ms(v * 100, mr * 100),
        })
    return best_fit, best_states, best_probs, best_model, pd.DataFrame(rows)


def _classify_ms(desc):
    d = desc.upper()
    if 'BAJISTA' in d or 'EXPANSION BAJISTA' in d or 'TREND BAJISTA' in d:
        return 'bearish'
    if 'ALCISTA' in d or 'EXPANSION ALCISTA' in d or 'TREND ALCISTA' in d:
        return 'bullish'
    return 'neutral'


def _describe_ms(vol, mr):
    if vol >= 5:
        if mr > 0.15: return '[MS EXPANSION ALCISTA]'
        elif mr < -0.15: return '[MS EXPANSION BAJISTA]'
        else: return '[MS ALTA VOLATILIDAD]'
    if vol >= 3.5:
        if mr > 0.15: return '[MS TREND ALCISTA]'
        elif mr < -0.15: return '[MS TREND BAJISTA]'
        else: return '[MS VOLATILIDAD]'
    if mr > 0.20: return '[MS ALCISTA FUERTE]'
    if mr > 0.08: return '[MS ALCISTA]'
    if mr < -0.20: return '[MS BAJISTA FUERTE]'
    if mr < -0.08: return '[MS BAJISTA]'
    if mr > 0.03: return '[MS ALCISTA SUAVE]'
    if mr < -0.03: return '[MS BAJISTA SUAVE]'
    return '[MS ACUMULACION]' if vol < 2.5 else '[MS LATERAL]'


def find_ms_regime_changes(states, index, state_summary):
    """Devuelve cambios de regimen con mismas keys que frc() del HMM.
    Keys: fr, to, fd, td, fb, tb, dur (shorthand consistent with HMM)."""
    dm = {}
    for _, r in state_summary.iterrows():
        dm[int(r['state'])] = r['description']
    changes = []
    ps = states[0]
    cs = 0
    for i in range(1, len(states)):
        if states[i] != ps:
            fd = dm.get(int(ps), f'R{ps}')
            td = dm.get(int(states[i]), f'R{states[i]}')
            ds = (index[i].strftime('%d-%m-%Y')
                  if hasattr(index[i], 'strftime') else str(index[i]))
            changes.append({
                'idx': i, 'date': index[i], 'ds': ds,
                'fr': int(ps), 'to': int(states[i]),
                'fd': fd, 'td': td,
                'fb': _classify_ms(fd), 'tb': _classify_ms(td),
                'dur': i - cs,
            })
            ps = states[i]
            cs = i
    return changes


def compute_signal_scores_with_ms(df, states, state_summary,
                                   weight=10, threshold=65):
    """Anade peso MS a los scores LONG/SHORT."""
    sb = {}
    for _, r in state_summary.iterrows():
        sb[int(r['state'])] = _classify_ms(r['description'])
    df = df.copy()
    vl = min(len(states), len(df))
    if vl == 0:
        return df
    mb = pd.Series(False, index=df.index)
    mbr = pd.Series(False, index=df.index)
    for s, b in sb.items():
        mk = pd.Series(False, index=df.index)
        mk.iloc[:vl] = (states[:vl] == s)
        if b == 'bullish':
            mb[mk] = True
        elif b == 'bearish':
            mbr[mk] = True
    if 'signal_score_long' in df.columns:
        df['signal_score_long'] = (
            df['signal_score_long'].astype(float) + mb.astype(float) * weight
        )
    if 'signal_score_short' in df.columns:
        df['signal_score_short'] = (
            df['signal_score_short'].astype(float) + mbr.astype(float) * weight
        )
    if ('signal_score_long' in df.columns
            and 'signal_score_short' in df.columns):
        df['signal_long'] = (df['signal_score_long'] >= threshold).astype(int)
        df['signal_short'] = (df['signal_score_short'] >= threshold).astype(int)
    return df


def detect_early_alerts_ms(df, states, state_summary, et=40, ew=3):
    """Detecta alertas tempranas usando MS."""
    sbm = {}
    for _, r in state_summary.iterrows():
        sbm[int(r['state'])] = _classify_ms(r['description'])
    df = df.copy()
    vl = min(len(states), len(df))
    rc = np.zeros(vl, dtype=bool)
    for i in range(1, vl):
        if states[i] != states[i - 1]:
            rc[i] = True
            for j in range(1, min(ew, vl - i)):
                rc[i + j] = True
    df['alert_early_long'] = False
    df['alert_early_short'] = False
    for i in range(vl):
        if not rc[i]:
            continue
        cb = sbm.get(states[i], 'neutral')
        sl = (df['signal_score_long'].iloc[i]
              if 'signal_score_long' in df.columns else 0)
        ss = (df['signal_score_short'].iloc[i]
              if 'signal_score_short' in df.columns else 0)
        if cb == 'bullish' and sl >= et:
            df.loc[df.index[i], 'alert_early_long'] = True
        if cb == 'bearish' and ss >= et:
            df.loc[df.index[i], 'alert_early_short'] = True
    return df
