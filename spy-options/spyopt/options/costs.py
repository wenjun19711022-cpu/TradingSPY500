"""Execution costs for SPY options.

Spread model from the 2026-10-02 close snapshot: almost every SPY strike within ~4% of spot
quoted a $0.01 wide market, ATM 1-day options $0.01-0.02. We use

    full_spread = clip(0.01 + 0.004 * price, 0.01, 0.10) * max(1, VIX/20) * stress

which reproduces the calm-market quotes and widens with volatility (spreads blow out in
stress). Each leg fills at mid -/+ half spread (no combo price improvement assumed), plus a
per-contract commission. Defaults approximate moomoo US options pricing (platform fee plus
exchange/OCC/regulatory fees ~= $0.70 per contract per side); change for your broker.
Options that expire worthless incur no closing cost; ITM legs are closed at intrinsic
before the bell (SPY is physically settled — never let a short leg get assigned).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    commission: float = 0.70       # $ per contract per side
    min_spread: float = 0.01
    spread_coef: float = 0.004
    max_spread: float = 0.10
    vix_ref: float = 20.0
    stress: float = 1.0            # multiply spreads (2.0 = pessimistic)
    multiplier: int = 100

    def half_spread(self, mid, vix=None):
        mid = np.asarray(mid, dtype=float)
        sp = np.clip(self.min_spread + self.spread_coef * mid, self.min_spread, self.max_spread)
        if vix is not None:
            sp = sp * np.maximum(1.0, np.asarray(vix, dtype=float) / self.vix_ref)
        return 0.5 * sp * self.stress

    def fill(self, mid, qty, vix=None):
        """Per-share price paid (+) / received (-) for qty>0 buy / qty<0 sell, per leg."""
        hs = self.half_spread(mid, vix)
        return np.where(np.asarray(qty) > 0, mid + hs, np.maximum(mid - hs, 0.0))
