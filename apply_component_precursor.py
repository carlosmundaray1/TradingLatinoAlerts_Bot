#!/usr/bin/env python
"""Update W_HMM_REGIME to 15 and replace compute_precursor_signals with component-level version."""
with open('tradinglatino_hmm_clean.py', 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1) Update W_HMM_REGIME to 15
old_w = 'W_HMM_REGIME: int = 8      # Suma puntos cuando el regimen HMM esta alineado con la senal'
new_w = 'W_HMM_REGIME: int = 15     # Optimo: 49.8% deteccion (vs 37.6% baseline)'
if old_w in content:
    content = content.replace(old_w, new_w)
    changes += 1
    print("[OK] Updated W_HMM_REGIME to 15")
else:
    print("[FAIL] Could not find W_HMM_REGIME")

# 2) Find and replace compute_precursor_signals
start = content.find('def compute_precursor_signals')
if start >= 0:
    end = content.find('\ndef ', start + 10)
    if end < 0:
        end = start + 3000
    
    old_func = content[start:end]
    
    new_func = '''def compute_precursor_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sistema de precursores a nivel de COMPONENTES.
    Monitorea smi_hist, bull_bias, squeeze_off, ema_dev_pct
    INDIVIDUALMENTE para detectar cambios inminentes de tendencia.
    """
    df["precursor_long"] = False
    df["precursor_short"] = False
    df["precursor_active"] = False
    df["precursor_confidence_long"] = 0.0
    df["precursor_confidence_short"] = 0.0

    # Precursor SMI: smi_hist subiendo hacia cero
    if "smi_hist" in df.columns:
        smi_rising = (df["smi_hist"] > df["smi_hist"].shift(1))
        smi_neg_rising = smi_rising & (df["smi_hist"] < 0) & (df["smi_hist"] > -0.5)
        smi_pos_falling = smi_rising & (df["smi_hist"] > 0) & (df["smi_hist"] < 0.5)
        # Actually for short: smi_hist falling from positive toward zero
        smi_falling = (df["smi_hist"] < df["smi_hist"].shift(1))
        smi_pos_falling = smi_falling & (df["smi_hist"] > 0) & (df["smi_hist"] < 0.5)
        df["precursor_smi_long"] = smi_neg_rising
        df["precursor_smi_short"] = smi_pos_falling

    # Precursor EMA: precio recuperandose hacia EMA
    if "ema_dev_pct" in df.columns:
        ema_improving = (df["ema_dev_pct"] < 0) & (df["ema_dev_pct"] > df["ema_dev_pct"].shift(1))
        df["precursor_ema_long"] = ema_improving & (df["ema_dev_pct"] > df["ema_dev_pct"].shift(2))
        ema_deteriorating = (df["ema_dev_pct"] > 0) & (df["ema_dev_pct"] < df["ema_dev_pct"].shift(1))
        df["precursor_ema_short"] = ema_deteriorating & (df["ema_dev_pct"] < df["ema_dev_pct"].shift(2))

    # Precursor ADX: tendencia fortaleciendose
    if "adx_delta" in df.columns and "adx" in df.columns:
        trend_up = df["adx_delta"] > 0
        adx_ok = df["adx"] > 20
        if "plus_di" in df.columns and "minus_di" in df.columns:
            bull_trend = trend_up & (df["plus_di"] > df["minus_di"])
            bear_trend = trend_up & (df["minus_di"] > df["plus_di"])
            df["precursor_adx_long"] = bull_trend & adx_ok & (df["adx_delta"] > df["adx_delta"].shift(1))
            df["precursor_adx_short"] = bear_trend & adx_ok & (df["adx_delta"] > df["adx_delta"].shift(1))
        else:
            df["precursor_adx_long"] = False
            df["precursor_adx_short"] = False

    # Precursor Squeeze Release
    if "squeeze_released" in df.columns and "adx" in df.columns and "smi_hist" in df.columns:
        recent_rel = df["squeeze_released"].rolling(3, min_periods=1).max() > 0
        df["precursor_sq_long"] = recent_rel & (df["adx"] > 20) & (df["smi_hist"] > -0.3)
        df["precursor_sq_short"] = recent_rel & (df["adx"] > 20) & (df["smi_hist"] < 0.3)

    # Precursor Bias: bull/bear bias activandose
    if "bull_bias" in df.columns and "bear_bias" in df.columns:
        bull_new = (df["bull_bias"] == 1) & (df["bull_bias"].shift(1) == 0)
        bear_new = (df["bear_bias"] == 1) & (df["bear_bias"].shift(1) == 0)
        df["precursor_bias_long"] = bull_new
        df["precursor_bias_short"] = bear_new

    # Combinar precursores
    long_cols = [c for c in df.columns if c.startswith("precursor_") and "long" in c.lower()]
    short_cols = [c for c in df.columns if c.startswith("precursor_") and "short" in c.lower()]

    if long_cols:
        df["precursor_long"] = df[long_cols].any(axis=1)
        df["precursor_confidence_long"] = df[long_cols].sum(axis=1) * 25
        df["precursor_confidence_long"] = df["precursor_confidence_long"].clip(0, 100)

    if short_cols:
        df["precursor_short"] = df[short_cols].any(axis=1)
        df["precursor_confidence_short"] = df[short_cols].sum(axis=1) * 25
        df["precursor_confidence_short"] = df["precursor_confidence_short"].clip(0, 100)

    df["precursor_active"] = df["precursor_long"] | df["precursor_short"]
    return df'''

    content = content.replace(old_func, new_func)
    changes += 1
    print("[OK] Replaced compute_precursor_signals with component-level version")
else:
    print("[FAIL] compute_precursor_signals not found")

with open('tradinglatino_hmm_clean.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal: {changes} changes applied")
