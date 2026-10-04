"""The strategy this study ends up recommending, stated in one place.

VRP-20 ladder (the only family that cleared IS -> VAL -> TEST plus the robustness gate):
  * every session close, open one rung: a SPY iron condor expiring 20 sessions later
  * short strikes at F -/+ 2 expected moves (EM from the VIX1D/VIX9D/VIX term structure,
    roughly the 5-8 delta strikes), long wings 1 EM further out  -> defined risk
  * exit: take profit at 50% of the credit (smoother, smaller crash losses) or hold to expiry
    ("aggressive": more return, deeper crash drawdowns)
  * size: total max-loss of all open rungs = `budget` of equity (10% conservative, 20%
    standard), i.e. budget/20 per rung, with the drawdown brake from spyq.risk.sizing
  * never let a short leg finish in the money (SPY options are physically settled) — close
    or roll ITM spreads before 15:45 on expiry day

What it is not: 0DTE/1DTE premium selling did not survive costs in this study (VIX1D carries
almost no variance premium: realized/implied variance 1.02 open->close), so short-dated
options are used here for information (expected-move bands, event detection), not as trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spyq.backtest.daily import Spec
from spyq.calendar import next_session
from spyq.data.rates import forward
from spyq.options import bs
from spyq.options.structures import build, worst_payoff
from spyq.options.costs import CostModel
from spyq.options.smile import Smile

CORE = Spec("n20_close", "iron_condor", 2.0, 2.0, 1.0, "tp50", "none")
AGGRESSIVE = Spec("n20_close", "iron_condor", 2.0, 2.0, 1.0, "hold", "none")
BUDGETS = {"conservative": 0.10, "standard": 0.20}


@dataclass
class Rung:
    entry_date: str
    expiry_date: str
    sessions: int
    spot: float
    forward: float
    atm_iv: float
    expected_move: float
    strikes: dict
    model_mid: dict
    credit: float
    max_loss: float
    take_profit_debit: float
    sizing: list


def next_sessions(last_date: pd.Timestamp, n: int) -> pd.Timestamp:
    return next_session(last_date, n)


def plan_rung(panel: pd.DataFrame, spec: Spec = CORE, smile: Smile | None = None,
              costs: CostModel | None = None, equity: float = 100_000.0) -> Rung:
    """The rung to open at the latest close in `panel` (prices are model mids — check them
    against the live chain before trading)."""
    smile = smile or Smile()
    costs = costs or CostModel()
    last = panel.iloc[-1]
    d = panel.index[-1]
    n = int(spec.mode[1:-6])
    expiry = next_sessions(d, n)
    S = float(last["close"])
    F = float(forward([S], [d], [expiry])[0])
    atm = float(smile.atm_term(last.get("vix1d_close", np.nan), last["vix9d_close"], last["vix_close"], n))
    T = n / 252
    legs = build(spec.structure, np.array([F]), np.array([atm]), T, spec.k_put, spec.k_call, spec.wing)
    names = ["long_put", "short_put", "short_call", "long_call"][: len(legs)]
    mids, cash = {}, 0.0
    for nm, lg in zip(names, legs):
        m = float(smile.price(F, lg.strike, T, atm, lg.is_call)[0])
        mids[f"{nm} {lg.strike[0]:g}{'C' if lg.is_call else 'P'}"] = round(m, 3)
        cash += -lg.qty * float(costs.fill(m, lg.qty, last["vix_close"]))
    comm = len(legs) * costs.commission / costs.multiplier
    max_loss = float(-(worst_payoff(legs)[0] + cash) + 2 * comm)
    return Rung(
        entry_date=str(d.date()), expiry_date=str(expiry.date()), sessions=n, spot=S, forward=round(F, 2),
        atm_iv=round(atm, 4), expected_move=round(F * atm * np.sqrt(T), 2),
        strikes={nm: float(lg.strike[0]) for nm, lg in zip(names, legs)},
        model_mid=mids, credit=round(cash - comm, 3), max_loss=round(max_loss, 3),
        take_profit_debit=round(0.5 * cash, 3),
        sizing=sizing_table(max_loss * 100, n),
    )


def sizing_table(max_loss_per_contract: float, sessions: int,
                 equities=(25_000, 50_000, 100_000, 250_000, 500_000)) -> list:
    """Contracts per rung for each budget, laddering daily (N rungs) or weekly (N/5 rungs).
    Zero means the account is too small for that cadence at that budget (use weekly rungs,
    or narrower fixed-dollar wings, which cut max loss per contract)."""
    rows = []
    for eq in equities:
        for cadence, rungs in (("daily", sessions), ("weekly", max(1, sessions // 5))):
            row = {"equity": eq, "cadence": cadence, "open_rungs": rungs}
            for lab, b in BUDGETS.items():
                row[lab] = int(eq * b / rungs // max_loss_per_contract)
            rows.append(row)
    return rows


def expected_move_bands(panel: pd.DataFrame, smile: Smile | None = None) -> dict:
    """Next-session and next-week 1-sigma bands from VIX1D / VIX9D (for the K-line page)."""
    smile = smile or Smile()
    last = panel.iloc[-1]
    F = float(last["close"])
    out = {}
    for lab, days in (("1d", 1), ("5d", 5), ("20d", 20)):
        atm = float(smile.atm_term(last.get("vix1d_close", np.nan), last["vix9d_close"], last["vix_close"], days))
        em = F * atm * np.sqrt(days / 252)
        out[lab] = {"atm_iv": round(atm, 4), "em": round(em, 2), "lo": round(F - em, 2), "hi": round(F + em, 2),
                    "lo2": round(F - 2 * em, 2), "hi2": round(F + 2 * em, 2)}
    return out


def delta_of_strikes(rung: Rung, smile: Smile | None = None) -> dict:
    smile = smile or Smile()
    T = rung.sessions / 252
    out = {}
    for nm, k in rung.strikes.items():
        is_call = nm.endswith("call")
        iv = float(smile.iv(rung.forward, k, T, rung.atm_iv))
        out[nm] = round(float(bs.delta(rung.forward, k, T, iv, is_call)), 3)
    return out
