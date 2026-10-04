"""Realized-volatility estimators and a leakage-free HAR forecast.

All functions return *daily variance* (decimal^2 per session) unless noted; annualize with
sqrt(252 * var). Forecasts at row t use information up to and including the close of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


def parkinson(h, l):
    return np.log(h / l) ** 2 / (4 * LN2)


def garman_klass(o, h, l, c):
    return 0.5 * np.log(h / l) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2


def rogers_satchell(o, h, l, c):
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang(df: pd.DataFrame, window=20) -> pd.Series:
    """Yang-Zhang daily variance over a rolling window (handles opening gaps)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    oc = np.log(o / c.shift())
    co = np.log(c / o)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    rs = rogers_satchell(o, h, l, c)
    return oc.rolling(window).var() + k * co.rolling(window).var() + (1 - k) * rs.rolling(window).mean()


def intraday_rv(bars_1m: pd.DataFrame, step="5min") -> pd.Series:
    """Session realized variance from minute bars (sum of squared `step` returns)."""
    c = bars_1m["close"]
    rs = c.groupby(c.index.date).apply(lambda x: x.resample(step).last().dropna())
    r = np.log(rs).groupby(level=0).diff().dropna()
    return (r**2).groupby(level=0).sum()


def har_forecast(rv: pd.Series, min_obs=250, refit_every=21) -> pd.Series:
    """One-step-ahead HAR-RV (Corsi 2009) on log variance, refit on an expanding window.

    Row t holds the forecast of rv[t+1] made with data through t. The log model is mapped
    back with Duan's smearing factor mean(exp(residual)) — the normal correction exp(s2/2)
    badly overshoots for noisy daily proxies (log chi-square residuals). Falls back to a
    20-day mean before `min_obs` observations are available.
    """
    x = np.log(rv.clip(lower=1e-10))
    feats = pd.DataFrame({"d": x, "w": x.rolling(5).mean(), "m": x.rolling(22).mean()})
    target = x.shift(-1)
    out = pd.Series(np.nan, index=rv.index)
    model = None
    vals = feats.values
    for i in range(len(rv)):
        if i >= min_obs and (model is None or i % refit_every == 0):
            X = vals[:i]
            y = target.values[:i]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            A = np.column_stack([np.ones(ok.sum()), X[ok]])
            beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            smear = float(np.mean(np.exp(y[ok] - A @ beta)))
            model = (beta, smear)
        if model is not None and np.isfinite(vals[i]).all():
            b, smear = model
            out.iloc[i] = np.exp(b[0] + vals[i] @ b[1:]) * smear
    fallback = rv.rolling(20, min_periods=5).mean()
    return out.fillna(fallback)


def ann(var):
    """Daily variance -> annualized vol in vol points."""
    return np.sqrt(252 * np.asarray(var)) * 100
