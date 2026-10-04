"""TradingView / moomoo-style indicators, support-resistance levels and candle patterns.

All functions are vectorized over a bar DataFrame (open/high/low/close/volume, DatetimeIndex
in ET) and only use data up to each row — nothing peeks ahead, so every value at row t is
what the chart showed when bar t closed. Defaults follow TradingView's built-ins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- basics

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (TradingView ta.rma)."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df):
    c = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - c).abs(), (df["low"] - c).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return rma(true_range(df), n)


# ----------------------------------------------------------------------------- oscillators

def rsi(c: pd.Series, n=14) -> pd.Series:
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def stoch(df, k=14, d=3, smooth=3):
    lo, hi = df["low"].rolling(k).min(), df["high"].rolling(k).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    kk = raw.rolling(smooth).mean()
    return kk, kk.rolling(d).mean()


def kdj(df, n=9, m1=3, m2=3):
    """KDJ as moomoo / 同花顺 draw it: K = SMA(RSV, 3), D = SMA(K, 3), J = 3K - 2D."""
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    rsv = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def williams_r(df, n=14):
    hi, lo = df["high"].rolling(n).max(), df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def macd(c, fast=12, slow=26, sig=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return m, s, m - s


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pd.Series(plus, index=df.index), n) / tr
    mdi = 100 * rma(pd.Series(minus, index=df.index), n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx, n), pdi, mdi


# ----------------------------------------------------------------------------- bands / trend

def bollinger(c, n=20, k=2.0):
    m, s = c.rolling(n).mean(), c.rolling(n).std(ddof=0)
    return m, m + k * s, m - k * s, (c - m) / s.replace(0, np.nan)


def keltner(df, n=20, k=1.5):
    m = ema(df["close"], n)
    a = atr(df, n)
    return m, m + k * a, m - k * a


def squeeze_on(df, n=20):
    """TTM squeeze: Bollinger bands inside Keltner channels."""
    _, bu, bl, _ = bollinger(df["close"], n, 2.0)
    _, ku, kl = keltner(df, n, 1.5)
    return (bu < ku) & (bl > kl)


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    up, dn = (hl2 - mult * a).values, (hl2 + mult * a).values
    c = df["close"].values
    trend = np.ones(len(df))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(df)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        trend[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else trend[i - 1]
    return pd.Series(trend, index=df.index), pd.Series(np.where(trend > 0, fu, fd), index=df.index)


def session_vwap(df):
    """Session VWAP with 1/2/3-sigma bands (volume-weighted std of typical price), reset daily."""
    day = df.index.normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"].clip(lower=0).replace(0, 1e-9)
    cv = v.groupby(day).cumsum()
    vw = (tp * v).groupby(day).cumsum() / cv
    var = (tp * tp * v).groupby(day).cumsum() / cv - vw * vw
    sd = np.sqrt(var.clip(lower=0))
    return vw, sd


def obv(df):
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


# ----------------------------------------------------------------------------- candles

def candle_parts(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    return rng, body, lower, upper


def hammer(df, wick=0.55, body_max=0.4):
    """Pin bar / hammer (下影线插针): long lower wick, small body, close in the upper part."""
    rng, body, lower, _ = candle_parts(df)
    return (lower / rng >= wick) & (body / rng <= body_max) & ((df["close"] - df["low"]) / rng >= 0.55)


def shooting_star(df, wick=0.55, body_max=0.4):
    rng, body, _, upper = candle_parts(df)
    return (upper / rng >= wick) & (body / rng <= body_max) & ((df["high"] - df["close"]) / rng >= 0.55)


def bull_engulf(df):
    o, c = df["open"], df["close"]
    return (c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())


def bear_engulf(df):
    o, c = df["open"], df["close"]
    return (c < o) & (c.shift() > o.shift()) & (c <= o.shift()) & (o >= c.shift())


# ----------------------------------------------------------------------------- swings / divergence

def swing_lows(low: pd.Series, k=2) -> pd.Series:
    """Confirmed fractal lows: the value appears k bars AFTER the low (no look-ahead)."""
    is_low = (low == low.rolling(2 * k + 1, center=True).min())
    return low.where(is_low).shift(k)


def swing_highs(high: pd.Series, k=2) -> pd.Series:
    is_high = (high == high.rolling(2 * k + 1, center=True).max())
    return high.where(is_high).shift(k)


def bullish_divergence(df, osc: pd.Series, lookback=30) -> pd.Series:
    """Price makes a lower low than the previous confirmed swing low while the oscillator
    makes a higher low (evaluated on the current bar's low vs the last swing low)."""
    sl = swing_lows(df["low"], 2)
    last_low = sl.ffill(limit=lookback)
    osc_at_low = osc.shift(2).where(sl.notna()).ffill(limit=lookback)  # oscillator AT the swing-low bar
    return (df["low"] < last_low) & (osc > osc_at_low)


# ----------------------------------------------------------------------------- daily levels for intraday bars

def daily_levels(bars_1m: pd.DataFrame, premarket: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-session support/resistance known BEFORE the session opens:
    prior-day high/low/close, classic pivots, prior-session volume profile POC / VAH / VAL,
    premarket high/low (if extended-hours bars are given)."""
    day = bars_1m.index.normalize()
    g = bars_1m.groupby(day)
    d = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last()})
    prev = d.shift()
    p = (prev["h"] + prev["l"] + prev["c"]) / 3
    lv = pd.DataFrame(index=d.index)
    lv["pdh"], lv["pdl"], lv["pdc"] = prev["h"], prev["l"], prev["c"]
    lv["pivot"], lv["s1"], lv["s2"] = p, 2 * p - prev["h"], p - (prev["h"] - prev["l"])
    lv["r1"], lv["r2"] = 2 * p - prev["l"], p + (prev["h"] - prev["l"])
    poc, vah, val = {}, {}, {}
    for dd, x in g:
        tp = ((x["high"] + x["low"] + x["close"]) / 3).round(1)
        prof = x["volume"].groupby(tp).sum().sort_index()
        if prof.empty or prof.sum() <= 0:
            continue
        poc[dd] = prof.idxmax()
        order = prof.sort_values(ascending=False)
        keep = order.cumsum() <= 0.7 * prof.sum()
        inside = order.index[keep] if keep.any() else order.index[:1]
        vah[dd], val[dd] = inside.max(), inside.min()
    lv["poc"] = pd.Series(poc).shift().reindex(lv.index) if poc else np.nan
    lv["vah"] = pd.Series(vah).shift().reindex(lv.index) if vah else np.nan
    lv["val"] = pd.Series(val).shift().reindex(lv.index) if val else np.nan
    if premarket is not None and not premarket.empty:
        pg = premarket.groupby(premarket.index.normalize())
        lv["pmh"], lv["pml"] = pg["high"].max().reindex(lv.index), pg["low"].min().reindex(lv.index)
    return lv


SUPPORT_LEVELS = ("pdl", "pdc", "s1", "s2", "pivot", "val", "poc", "pml")
RESIST_LEVELS = ("pdh", "pdc", "r1", "r2", "pivot", "vah", "poc", "pmh")


def nearest_support(price: pd.Series, lv_row_aligned: pd.DataFrame, extra: dict[str, pd.Series] | None = None):
    """Distance (fraction) from price down to the closest support at or below it, and its name."""
    cols = [c for c in SUPPORT_LEVELS if c in lv_row_aligned]
    cand = {c: lv_row_aligned[c] for c in cols}
    if extra:
        cand.update(extra)
    arr = np.vstack([v.reindex(price.index).values for v in cand.values()])
    names = np.array(list(cand.keys()))
    p = price.values
    below = np.where(arr <= p * 1.0005, arr, -np.inf)  # allow a hair above (the touch)
    idx = np.argmax(below, axis=0)
    lvl = below[idx, np.arange(len(p))]
    lvl = np.where(np.isfinite(lvl), lvl, np.nan)
    return pd.Series(lvl, index=price.index), pd.Series(np.where(np.isfinite(lvl), names[idx], ""), index=price.index)
