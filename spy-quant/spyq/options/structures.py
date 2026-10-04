"""Vectorized multi-leg option structures on SPY's $1 strike grid.

A structure is a list of legs (is_call, qty, strike_array): qty -1 = short, +1 = long, and
each strike array has one entry per trading day, so a whole backtest prices in one shot.
Strikes are placed in units of the 1-sigma expected move EM = F * sigma_atm * sqrt(T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRUCTURES = ("iron_condor", "put_spread", "call_spread", "iron_fly",
              "long_straddle", "bull_call_debit", "bear_put_debit")
CREDIT = {"iron_condor", "put_spread", "call_spread", "iron_fly"}


@dataclass
class Leg:
    is_call: bool
    qty: int
    strike: np.ndarray


def expected_move(F, atm, T):
    return np.asarray(F) * np.asarray(atm) * np.sqrt(T)


def build(name: str, F, atm, T, k_put=1.0, k_call=1.0, wing=0.5) -> list[Leg]:
    """Strikes: shorts at F -/+ k*EM (rounded away from spot), wings `wing`*EM further out
    (at least $1). Iron fly / straddle / debit spreads are centred on the nearest strike."""
    F = np.asarray(F, dtype=float)
    em = expected_move(F, atm, T)
    w = np.maximum(1.0, np.round(wing * em))
    atm_k = np.round(F)
    sp = np.floor(F - k_put * em)
    sc = np.ceil(F + k_call * em)
    if name == "iron_condor":
        return [Leg(False, 1, sp - w), Leg(False, -1, sp), Leg(True, -1, sc), Leg(True, 1, sc + w)]
    if name == "put_spread":
        return [Leg(False, 1, sp - w), Leg(False, -1, sp)]
    if name == "call_spread":
        return [Leg(True, -1, sc), Leg(True, 1, sc + w)]
    if name == "iron_fly":
        return [Leg(False, 1, atm_k - w), Leg(False, -1, atm_k), Leg(True, -1, atm_k), Leg(True, 1, atm_k + w)]
    if name == "long_straddle":
        return [Leg(False, 1, atm_k), Leg(True, 1, atm_k)]
    if name == "bull_call_debit":
        return [Leg(True, 1, atm_k), Leg(True, -1, atm_k + w)]
    if name == "bear_put_debit":
        return [Leg(False, 1, atm_k), Leg(False, -1, atm_k - w)]
    raise ValueError(name)


def payoff(legs: list[Leg], S):
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    for lg in legs:
        intr = np.maximum(S - lg.strike, 0) if lg.is_call else np.maximum(lg.strike - S, 0)
        out = out + lg.qty * intr
    return out


def worst_payoff(legs: list[Leg]):
    """Minimum expiry payoff (per share). Payoffs are piecewise linear, so checking every
    strike plus far-away spots is exact; long-only structures bottom out at 0."""
    ks = np.stack([lg.strike for lg in legs])
    probes = list(ks) + [ks.min(axis=0) * 0.5, ks.max(axis=0) * 1.5]
    return np.min(np.stack([payoff(legs, p) for p in probes]), axis=0)


def short_strikes(legs: list[Leg]):
    """(short put strike or nan, short call strike or nan) — used by touch stops."""
    n = len(legs[0].strike)
    sp = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    for lg in legs:
        if lg.qty < 0 and not lg.is_call:
            sp = lg.strike.astype(float)
        if lg.qty < 0 and lg.is_call:
            sc = lg.strike.astype(float)
    return sp, sc
