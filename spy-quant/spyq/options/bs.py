"""Vectorized Black (forward) pricing, greeks and implied vol.

Time is measured in *trading-day years* (1 session = 1/252): VIX1D is quoted that way —
its open print matches realized open->close SPY vol on 2023-2026 data (12.2 vs 12.3) —
and it keeps 0DTE math free of overnight/weekend calendar artefacts.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQRT2PI = np.sqrt(2 * np.pi)


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, sigma)))
    sT = np.maximum(sigma * np.sqrt(np.maximum(T, 0.0)), 1e-12)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    return d1, d1 - sT, sT


def price(F, K, T, sigma, is_call, df=1.0):
    """Undiscounted Black price times discount factor `df`. Exact intrinsic at T<=0."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2, _ = _d1d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    out = np.where(is_call, call, put)
    expired = np.asarray(T) <= 0
    if np.any(expired):
        intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
        out = np.where(expired, intrinsic, out)
    return df * np.maximum(out, 0.0)


def delta(F, K, T, sigma, is_call):
    d1, _, _ = _d1d2(F, K, T, sigma)
    return np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)


def gamma(F, K, T, sigma):
    d1, _, sT = _d1d2(F, K, T, sigma)
    return np.exp(-0.5 * d1 * d1) / SQRT2PI / (np.asarray(F) * sT)


def vega(F, K, T, sigma):
    d1, _, _ = _d1d2(F, K, T, sigma)
    return np.asarray(F) * np.exp(-0.5 * d1 * d1) / SQRT2PI * np.sqrt(np.maximum(T, 0.0))


def theta_per_day(F, K, T, sigma, is_call):
    """Price change for one trading day of time decay (negative for long options)."""
    dt = 1.0 / 252
    return price(F, K, np.maximum(np.asarray(T) - dt, 0), sigma, is_call) - price(F, K, T, sigma, is_call)


def implied_vol(px, F, K, T, is_call, lo=1e-4, hi=5.0, iters=100):
    """Vectorized bisection on [lo, hi] (robust for deep wings where Newton stalls)."""
    px, F, K, T, is_call = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (px, F, K, T, is_call)))
    is_call = is_call.astype(bool)
    a = np.full(px.shape, lo)
    b = np.full(px.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = price(F, K, T, m, is_call) > px
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    return np.where(px <= intrinsic + 1e-10, np.nan, iv)
