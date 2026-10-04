"""Bottom detectors for any bar timeframe (3m ... monthly). Everything at row t uses bars <= t.

Pre-registered families (fixed before the study ran):
  rsi2      RSI(2) < 10 while the timeframe's long trend is up (close > SMA(trend_len))
            — the Connors short-term dip in an uptrend
  boll      close below the lower 2-sigma Bollinger band (20) in an uptrend
  wick      capitulation "pin bar" (插针): lower wick >= 50% of the range, close in the upper
            half, after a 2-ATR drop from the 20-bar high, volume >= 1 sd above its mean
  panic     (daily and up) VIX spiked >= 30% above its 20-day mean and has started falling
            (close <= 90% of its 5-day high) while SPY is >= 7% below its 52-week high
  ath_dd    drawdown-from-all-time-high ladder: first close below -10%, -20%, -30% from the
            running high (each tier fires once per bear market); held until a new high
Exit rules per family are in ENGINE_DEFAULTS of spyq.dip.engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_LEN = {"1M": 10, "1w": 40, "1d": 200}  # ~10 months / 40 weeks / 200 days; intraday: 200 bars


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def features(df: pd.DataFrame, tf: str, vix: pd.Series | None = None) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    f = pd.DataFrame(index=df.index)
    tl = TREND_LEN.get(tf, 200)
    f["sma_trend"] = c.rolling(tl, min_periods=tl).mean()
    f["trend_up"] = c > f["sma_trend"]
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_z"] = (c - m) / s
    f["atr"] = atr(df)
    f["dd_atr"] = (c - h.rolling(20).max()) / f["atr"]
    rng = (h - l).replace(0, np.nan)
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["close_pos"] = (c - l) / rng
    if "volume" in df and df["volume"].fillna(0).sum() > 0:
        v = df["volume"]
        f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    else:
        f["vol_z"] = np.inf  # no volume (index data): don't require a volume spike
    f["sma5"] = c.rolling(5).mean()
    f["sma10"] = c.rolling(10).mean()
    f["sma20"] = m
    f["ath"] = c.cummax()
    f["dd_ath"] = c / f["ath"] - 1
    f["high_52w"] = c.rolling({"1d": 252, "1w": 52, "1M": 12}.get(tf, 252), min_periods=1).max()
    f["dd_52w"] = c / f["high_52w"] - 1
    if vix is not None:
        vx = vix.reindex(df.index, method="ffill")
        f["vix"] = vx
        f["vix_spike"] = vx / vx.rolling({"1d": 20, "1w": 4, "1M": 1}.get(tf, 20), min_periods=1).mean()
        f["vix_off_high"] = vx / vx.rolling({"1d": 5, "1w": 2, "1M": 1}.get(tf, 5), min_periods=1).max()
    return f


def signal(f: pd.DataFrame, family: str, tf: str) -> pd.Series:
    """Boolean entry signal at bar close (enter at the next bar's open)."""
    if family == "rsi2":
        return (f["rsi2"] < 10) & f["trend_up"]
    if family == "boll":
        return (f["boll_z"] < -2) & f["trend_up"]
    if family == "wick":
        return (f["lower_wick"] >= 0.5) & (f["close_pos"] >= 0.5) & (f["dd_atr"] <= -2) & (f["vol_z"] >= 1)
    if family == "panic":
        if "vix" not in f or tf not in ("1d", "1w", "1M"):
            return pd.Series(False, index=f.index)
        return (f["vix_spike"] >= 1.3) & (f["vix_off_high"] <= 0.90) & (f["dd_52w"] <= -0.07)
    if family == "ath_dd":
        out = pd.Series(False, index=f.index)
        armed = {0.10: True, 0.20: True, 0.30: True}
        tier = pd.Series(0.0, index=f.index)
        for t, d in f["dd_ath"].items():
            if d >= 0:
                armed = {k: True for k in armed}
                continue
            for k in sorted(armed):
                if armed[k] and d <= -k:
                    armed[k] = False
                    out[t] = True
                    tier[t] = k
        out.attrs["tier"] = tier
        return out
    raise ValueError(family)


FAMILIES = ("rsi2", "boll", "wick", "panic", "ath_dd")
