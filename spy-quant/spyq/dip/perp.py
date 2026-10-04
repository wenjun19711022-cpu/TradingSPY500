"""OKX-style USDT-margined perpetual: fees, funding and liquidation for a SPY long.

Defaults (change to your account's numbers — OKX shows both on the contract page):
  taker 0.05% / maker 0.02% per side (VIP0), slippage 0.01% per side
  funding 0.01% per 8h on notional, paid by longs when positive (3 settlements a day,
  every calendar day including weekends) -> 0.03%/day ~= 11%/yr of notional
  maintenance margin rate 0.5%

Liquidation of an isolated long opened at P with leverage L (initial margin P/L per unit):
  equity at price p = P/L + (p - P) - fees  -> liquidated when equity <= mmr * p
  => p_liq = P * (1 - 1/L) / (1 - mmr)   (ignoring accrued funding, which moves it up)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerpCosts:
    taker: float = 0.0005
    maker: float = 0.0002
    slippage: float = 0.0001
    funding_8h: float = 0.0001
    mmr: float = 0.005
    max_leverage: float = 20.0

    def round_trip(self, maker_entry: bool = False, maker_exit: bool = False) -> float:
        e = self.maker if maker_entry else self.taker + self.slippage
        x = self.maker if maker_exit else self.taker + self.slippage
        return e + x

    def funding(self, days: float) -> float:
        """Funding paid per unit notional for holding `days` calendar days."""
        return self.funding_8h * 3.0 * days

    def liq_drop(self, lev: float) -> float:
        """Fractional price drop from entry that liquidates an isolated long at `lev`."""
        if lev <= 1.0:
            return 1.0
        p_liq = (1 - 1 / lev) / (1 - self.mmr)
        return 1 - p_liq


ZERO = PerpCosts(taker=0, maker=0, slippage=0, funding_8h=0, mmr=0.005)
