"""Session-anchored resampling of 1-minute RTH bars to any intraday timeframe.

Bars are anchored at the 09:30 open of each session (the way TradingView and most US
brokers draw SPY): a 4h bar is 09:30-13:30 and 13:30-16:00, a 1h bar is 09:30-10:30 ...
15:30-16:00, a 3m bar is 09:30-09:33 ... Sessions are inferred from the data itself, so
half days (13:00 close) and missing minutes are handled without a holiday calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spylev.calendar import to_et

TIMEFRAMES = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "1d": 390,
}
DEFAULT_SET = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]


def parse_tf(tf: str) -> int:
    if tf in TIMEFRAMES:
        return TIMEFRAMES[tf]
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"unknown timeframe {tf!r}")


def resample(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate 1-minute RTH bars (start-time labelled) into `tf` bars anchored at 09:30.

    Returned bars are labelled by their start time. Volume-weighted price is carried when
    the input has a `vwap` column. Partial final buckets (e.g. the 15:30 hourly bar) are kept.
    """
    if bars.empty:
        return bars.copy()
    n = parse_tf(tf)
    idx = to_et(bars.index)
    df = bars.copy()
    df.index = idx
    day = idx.normalize()
    mos = (idx.hour * 60 + idx.minute) - (9 * 60 + 30)  # minute of session
    if tf == "1d":
        bucket = np.zeros(len(df), dtype=int)
    else:
        bucket = (np.asarray(mos) // n).astype(int)
    key_start = day + pd.to_timedelta(9 * 60 + 30 + bucket * (0 if tf == "1d" else n), unit="m")
    g = df.groupby(key_start)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    if "vwap" in df.columns:
        pv = (df["vwap"].fillna(df["close"]) * df["volume"]).groupby(key_start).sum()
        out["vwap"] = pv / out["volume"].replace(0, np.nan)
    out["n_bars"] = g["close"].size()
    out.index.name = "ts"
    return out


def resample_all(bars: pd.DataFrame, tfs=None) -> dict[str, pd.DataFrame]:
    return {tf: resample(bars, tf) for tf in (tfs or DEFAULT_SET)}


def integrity_report(bars: pd.DataFrame, max_jump: float = 0.02) -> pd.DataFrame:
    """Per-session data quality: bar count, missing minutes, OHLC violations, bad ticks.

    `max_jump` flags consecutive 1-minute closes that move more than 2% — almost always a
    bad print on SPY. A clean regular session has 390 bars (210 on a half day).
    """
    if bars.empty:
        return pd.DataFrame()
    idx = to_et(bars.index)
    df = bars.copy()
    df.index = idx
    day = idx.date
    viol = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | (
        df["low"] > df[["open", "close"]].min(axis=1) + 1e-9) | (df["low"] > df["high"])
    jump = df["close"].pct_change().abs() > max_jump
    jump[pd.Series(day, index=df.index) != pd.Series(day, index=df.index).shift()] = False
    rep = pd.DataFrame({
        "bars": df.groupby(day).size(),
        "first": pd.Series(idx.time, index=idx).groupby(day).min(),
        "last": pd.Series(idx.time, index=idx).groupby(day).max(),
        "ohlc_violations": viol.groupby(day).sum(),
        "bad_ticks": jump.groupby(day).sum(),
        "zero_volume": (df["volume"] <= 0).groupby(day).sum(),
    })
    expected = np.where(pd.Series(rep["last"]).astype(str) <= "13:00:00", 210, 390)
    rep["expected"] = expected
    rep["missing"] = (rep["expected"] - rep["bars"]).clip(lower=0)
    rep["ok"] = (rep["missing"] <= 5) & (rep["ohlc_violations"] == 0) & (rep["bad_ticks"] == 0)
    return rep
