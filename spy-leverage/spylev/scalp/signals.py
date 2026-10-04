"""Multi-timeframe bottom / top detection for 1m-3m-5m SPY scalps.

For every timeframe bar we score six independent "bottom" facts (each 0/1):

  osc      >= 2 oscillators oversold: RSI14<=32, Stoch %K<=20, Williams %R<=-80, CCI<=-100,
           KDJ J<=0, MFI<=20                                    (TradingView / moomoo defaults)
  band     close <= lower Bollinger(20,2)  or  low <= session VWAP - 2 sigma
  support  the bar pierced or touched a support level and closed back above it (插针不下去):
           prior-day low/close, pivots S1/S2/P, prior-session value-area low / POC, premarket
           low, opening-range low, the session low before this bar, $5 round numbers,
           the last confirmed 5-bar swing low
  candle   hammer / pin bar or bullish engulfing
  climax   volume >= 1.5 sd above its 20-bar mean (selling climax)
  turn     MACD histogram rising, or bullish RSI divergence

The "top" score mirrors it (overbought, upper band / VWAP+2sd, rejection at resistance,
shooting star / bearish engulfing, climax, MACD histogram falling).

The 3m and 5m scores are attached to the 1m timeline only once their bar has CLOSED (merge on
the bar's last minute), so the 1m trigger never sees an unfinished higher-timeframe candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spylev import ta
from spylev.data.resample import resample

TOL = 0.0003  # 0.03% of price counts as touching a level


def _osc_counts(df):
    r = ta.rsi(df["close"], 14)
    k, _ = ta.stoch(df)
    wr = ta.williams_r(df)
    cc = ta.cci(df)
    _, _, j = ta.kdj(df)
    has_vol = df["volume"].fillna(0).sum() > 0
    mf = ta.mfi(df) if has_vol else pd.Series(50.0, index=df.index)
    low = (r <= 32).astype(int) + (k <= 20).astype(int) + (wr <= -80).astype(int) + (cc <= -100).astype(int) + (j <= 0).astype(int) + (mf <= 20).astype(int)
    high = (r >= 68).astype(int) + (k >= 80).astype(int) + (wr >= -20).astype(int) + (cc >= 100).astype(int) + (j >= 100).astype(int) + (mf >= 80).astype(int)
    return low, high, r, k, j


def level_frame(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Support / resistance candidates aligned to each bar (all known before the bar)."""
    day = df.index.normalize().tz_localize(None)
    lv = levels.copy()
    lv.index = pd.DatetimeIndex(lv.index).tz_localize(None) if getattr(lv.index, "tz", None) else pd.DatetimeIndex(lv.index)
    out = lv.reindex(day)
    out.index = df.index
    g = df.groupby(df.index.normalize())
    out["sess_low"] = g["low"].cummin().groupby(df.index.normalize()).shift()     # low BEFORE this bar
    out["sess_high"] = g["high"].cummax().groupby(df.index.normalize()).shift()
    mos = (df.index.hour * 60 + df.index.minute) - 570
    first15 = mos < 15
    orl = df["low"].where(first15).groupby(df.index.normalize()).transform("min")
    orh = df["high"].where(first15).groupby(df.index.normalize()).transform("max")
    out["or_low"] = orl.where(mos >= 15)
    out["or_high"] = orh.where(mos >= 15)
    out["swing_low"] = ta.swing_lows(df["low"], 2).ffill()
    out["swing_high"] = ta.swing_highs(df["high"], 2).ffill()
    return out


SUP = ("pdl", "pdc", "s1", "s2", "pivot", "val", "poc", "pml", "sess_low", "or_low", "swing_low")
RES = ("pdh", "pdc", "r1", "r2", "pivot", "vah", "poc", "pmh", "sess_high", "or_high", "swing_high")


def _rejection(df, lf, names, side):
    lo, hi, c = df["low"], df["high"], df["close"]
    hit = pd.Series(False, index=df.index)
    which = pd.Series("", index=df.index, dtype=object)
    cols = [n for n in names if n in lf]
    for n in cols:
        L = lf[n]
        if side == "sup":
            h = (lo <= L * (1 + TOL)) & (c >= L) & L.notna()
        else:
            h = (hi >= L * (1 - TOL)) & (c <= L) & L.notna()
        which = which.where(~(h & ~hit), n)
        hit |= h
    # $5 round numbers
    if side == "sup":
        r5 = np.floor(c / 5) * 5
        h = (lo <= r5 * (1 + TOL)) & (c >= r5)
    else:
        r5 = np.ceil(c / 5) * 5
        h = (hi >= r5 * (1 - TOL)) & (c <= r5)
    which = which.where(~(h & ~hit), "round5")
    hit |= h
    return hit, which


def tf_features(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    osc_lo, osc_hi, r, k, j = _osc_counts(df)
    mid, up, lo, z = ta.bollinger(df["close"])
    vw, sd = ta.session_vwap(df)
    _, _, hist = ta.macd(df["close"])
    v = df["volume"]
    vz = (v - v.rolling(20).mean()) / v.rolling(20).std()
    lf = level_frame(df, levels)
    sup_hit, sup_name = _rejection(df, lf, SUP, "sup")
    res_hit, res_name = _rejection(df, lf, RES, "res")
    div = ta.bullish_divergence(df, r)
    f["rsi"], f["stoch_k"], f["kdj_j"], f["boll_z"] = r, k, j, z
    f["vwap"], f["vwap_sd"], f["bb_mid"], f["bb_up"] = vw, sd, mid, up
    f["b_osc"] = osc_lo >= 2
    f["b_band"] = (df["close"] <= lo) | (df["low"] <= vw - 2 * sd)
    f["b_support"] = sup_hit
    f["b_candle"] = ta.hammer(df) | ta.bull_engulf(df)
    f["b_climax"] = vz >= 1.5
    f["b_turn"] = (hist > hist.shift()) | div
    f["support_name"] = sup_name
    f["t_osc"] = osc_hi >= 2
    f["t_band"] = (df["close"] >= up) | (df["high"] >= vw + 2 * sd)
    f["t_resist"] = res_hit
    f["t_candle"] = ta.shooting_star(df) | ta.bear_engulf(df)
    f["t_climax"] = vz >= 1.5
    f["t_turn"] = hist < hist.shift()
    bcols = ["b_osc", "b_band", "b_support", "b_candle", "b_climax", "b_turn"]
    tcols = ["t_osc", "t_band", "t_resist", "t_candle", "t_climax", "t_turn"]
    f["bottom"] = f[bcols].fillna(False).astype(int).sum(axis=1)
    f["top"] = f[tcols].fillna(False).astype(int).sum(axis=1)
    return f


def _attach(m1: pd.DataFrame, f_tf: pd.DataFrame, tf_minutes: int, prefix: str,
            keep=("bottom", "top", "b_osc", "b_band", "b_support", "b_candle", "b_climax", "b_turn",
                  "t_osc", "t_resist", "support_name")):
    """Attach a higher-timeframe bar's features to the 1m bar on which it closes, then carry
    forward. Also keeps the previous completed bar's bottom score for 'recently bottomed'."""
    x = f_tf[list(keep)].copy()
    x["prev_bottom"] = f_tf["bottom"].shift()
    x["prev_top"] = f_tf["top"].shift()
    x["prev_b_osc"] = f_tf["b_osc"].shift()
    x["prev_b_support"] = f_tf["b_support"].shift()
    x.index = x.index + pd.Timedelta(minutes=tf_minutes - 1)  # bar's last minute
    x = x.add_prefix(prefix)
    out = m1.join(x, how="left")
    cols = list(x.columns)
    day = m1.index.normalize()
    out[cols] = out[cols].groupby(day).ffill()
    return out


def build(m1_bars: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """1m timeline with 1m features and the latest CLOSED 3m / 5m features."""
    f1 = tf_features(m1_bars, levels)
    b3, b5 = resample(m1_bars, "3m"), resample(m1_bars, "5m")
    f3, f5 = tf_features(b3, levels), tf_features(b5, levels)
    base = m1_bars.join(f1.add_prefix("m1_"))
    base = _attach(base, f3, 3, "m3_")
    base = _attach(base, f5, 5, "m5_")
    base["confirm"] = (m1_bars["close"] > m1_bars["high"].shift()) | ta.bull_engulf(m1_bars) | ta.hammer(m1_bars)
    base["m1_bottom_recent"] = f1["bottom"].rolling(3).max()
    base["mos"] = (m1_bars.index.hour * 60 + m1_bars.index.minute) - 570
    # daily trend context known before the open: prior close above its 20-day average
    dc = m1_bars["close"].groupby(m1_bars.index.normalize()).last()
    up = (dc > dc.rolling(20).mean()).shift()
    base["daily_up"] = up.reindex(m1_bars.index.normalize()).values
    base["vwap"], base["vwap_sd"] = f1["vwap"], f1["vwap_sd"]
    base["atr1"] = ta.atr(m1_bars, 14)
    return base


def _flag(x, col):
    return x[col].fillna(False).astype(bool)


def tf_conditions(x: pd.DataFrame, thr5=3, thr3=3, thr1=2) -> pd.DataFrame:
    """Per-timeframe 'this timeframe has bottomed' flags used by buy_signal (and shown on screen)."""
    b5 = ((x["m5_bottom"] >= thr5) & _flag(x, "m5_b_osc") & _flag(x, "m5_b_support")) | (
        (x["m5_prev_bottom"] >= thr5) & _flag(x, "m5_prev_b_osc") & _flag(x, "m5_prev_b_support"))
    b3 = ((x["m3_bottom"] >= thr3) & _flag(x, "m3_b_osc")) | ((x["m3_prev_bottom"] >= thr3) & _flag(x, "m3_prev_b_osc"))
    b1 = (x["m1_bottom_recent"] >= thr1) & x["confirm"].fillna(False).astype(bool)
    return pd.DataFrame({"m5": b5, "m3": b3, "m1": b1}).fillna(False).astype(bool)


def buy_signal(x: pd.DataFrame, thr5=3, thr3=3, thr1=2, trend="none", first_min=5, last_min=360) -> pd.Series:
    """The user's setup, made precise:
      5m  (this or the previous closed bar): oscillators oversold AND rejected a support level
          (插针不下去) AND total bottom score >= thr5
      3m  (this or the previous closed bar): oscillators oversold AND bottom score >= thr3
      1m  bottom score >= thr1 within the last 3 minutes AND this minute confirms the turn
          (close above the prior minute's high, bullish engulfing or hammer)
    Entries only between `first_min` and `last_min` minutes after the open."""
    c = tf_conditions(x, thr5, thr3, thr1)
    ok = c["m5"] & c["m3"] & c["m1"] & (x["mos"] >= first_min) & (x["mos"] <= last_min)
    if trend == "up":
        ok &= x["daily_up"].fillna(False).astype(bool)
    return ok.fillna(False)


def quality(x: pd.DataFrame) -> pd.Series:
    """0..18: total bottom facts across 1m / 3m / 5m — used to bucket expected returns."""
    return x["m5_bottom"].fillna(0) + x["m3_bottom"].fillna(0) + x["m1_bottom"].fillna(0)
