# -*- coding: utf-8 -*-
with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The new function to insert
new_func = '''
# -- HMM DIRECCIONAL: Features puramente direccionales (sin volatilidad) --
def build_directional_features(df):
    """Construye features SOLO direccionales para el HMM direccional.
    Excluye: volatilidad (vol_20, atr_norm), ADX strength, volumen, squeeze flag.
    Incluye solo features que indican DIRECCION del precio.
    NOTA: log_return_1 se incluye por necesidad interna de fit_hmm().
    """
    import numpy as np
    import pandas as pd
    features = pd.DataFrame(index=df.index)

    # log_return_1: REQUERIDO por fit_hmm() para reetiquetar estados
    features["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))

    # 1) DI spread: (plus_di - minus_di) normalizado a [-1, 1]
    if "plus_di" in df.columns and "minus_di" in df.columns:
        di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
        features["di_spread"] = ((df["plus_di"] - df["minus_di"]) / di_sum).fillna(0)

    # 2) EMA fast/slow spread (% del precio)
    ema_fast = df.get("ema_fast", df["Close"].ewm(span=10, adjust=False).mean())
    ema_slow = df.get("ema_slow", df["Close"].ewm(span=55, adjust=False).mean())
    features["ema_spread_raw"] = (ema_fast - ema_slow) / df["Close"].replace(0, np.nan) * 100
    features["ema_spread_raw"] = features["ema_spread_raw"].fillna(0).clip(-5, 5)

    # 3) Momentum del precio (20 velas)
    features["momentum_20"] = df["Close"].pct_change(20)

    # 4) Retornos acumulados (20 velas)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    features["cumret_20"] = log_ret.rolling(window=20).sum()

    # 5) RSI normalizado (-1 a +1)
    rsi_val = df.get("rsi14", pd.Series(50, index=df.index))
    features["rsi_norm"] = (rsi_val - 50) / 50

    # 6) Desviacion del precio vs EMA55 (%)
    if "ema_deviation_pct" in df.columns:
        features["ema_dev_pct"] = df["ema_deviation_pct"].fillna(0)
    else:
        features["ema_dev_pct"] = (df["Close"] - ema_slow) / ema_slow.replace(0, np.nan) * 100
        features["ema_dev_pct"] = features["ema_dev_pct"].fillna(0)

    # 7) Squeeze momentum histogram
    smi_hist = df.get("smi_hist", pd.Series(0.0, index=df.index))
    features["smi_hist_norm"] = smi_hist.fillna(0)

    # 8) Posicion relativa en rango High-Low (20 velas)
    high_20 = df["High"].rolling(20).max()
    low_20 = df["Low"].rolling(20).min()
    features["pos_in_range"] = (df["Close"] - low_20) / (high_20 - low_20 + 1e-10)

    # 9) Ratio de velas alcistas en ventana de 20
    features["up_bar_ratio"] = (df["Close"] > df["Open"]).rolling(20).sum() / 20.0

    # 10) Signal scores
    if "signal_score_long" in df.columns:
        features["signal_score_long"] = df["signal_score_long"].fillna(0)
    if "signal_score_short" in df.columns:
        features["signal_score_short"] = df["signal_score_short"].fillna(0)

    # 11) Score neto: long - short (direccion neta)
    if "signal_score_long" in features.columns and "signal_score_short" in features.columns:
        features["score_net"] = (features["signal_score_long"] - features["signal_score_short"]) / 100.0

    # 12) Delta de scores
    if "signal_score_long" in features.columns:
        features["score_delta_long"] = features["signal_score_long"].diff().fillna(0)
    if "signal_score_short" in features.columns:
        features["score_delta_short"] = features["signal_score_short"].diff().fillna(0)

    return features

'''

# Find insertion point: after build_hmm_features ends, before HMM section
idx_return = content.rfind('return features')
idx_hmm = content.find('# HMM: FIT + RELABEL')

# Insert between them
before = content[:idx_return + len('return features')]
after = content[idx_return + len('return features'):]

new_content = before + new_func + after

with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f'Inserted directional features function. File size: {len(new_content)} chars')

# Verify
with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    verify = f.read()
if 'build_directional_features' in verify:
    print('OK: build_directional_features found in file!')
else:
    print('ERROR: Function not found after insertion!')
