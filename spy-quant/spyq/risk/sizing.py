"""Position sizing for defined-risk option ladders.

Budget = total max-loss of all open positions as a fraction of equity. An N-session ladder
holds N rungs, so each new rung risks budget / N. On top of that:
  * drawdown brake  - scale new risk by 1.0 / 0.5 / 0.25 once equity is 10% / 20% below its peak
  * vol brake       - halve new risk while VIX is above its 1-year 90th percentile
  * Kelly check     - report f* = mean / var of return-on-risk; never run above half of it
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(ror: pd.Series) -> float:
    """Continuous-approximation Kelly fraction of equity to put at (max-loss) risk per bet."""
    r = ror.dropna()
    v = r.var()
    return float(r.mean() / v) if v > 0 else 0.0


def ladder_equity(trades: pd.DataFrame, budget: float, n_rungs: int, dd_brake: bool = True,
                  vol_brake: pd.Series | None = None, start: float = 1.0) -> pd.DataFrame:
    """Simulate equity for trades indexed by expiry (and carrying an `entry` column if they are
    multi-day). P&L is booked at expiry; risk is set at entry from the equity known then."""
    t = trades.sort_index()
    entry = pd.DatetimeIndex(t["entry"]) if "entry" in t else t.index
    days = sorted(set(entry) | set(t.index))
    eq = start
    peak = start
    pending: dict = {}
    hist = []
    by_entry = {}
    for i, e in enumerate(entry):
        by_entry.setdefault(e, []).append(i)
    by_exit = {}
    for i, x in enumerate(t.index):
        by_exit.setdefault(x, []).append(i)
    ror = t["ror"].values
    for d in days:
        # 1) book expiries first (they settle at this close)
        for i in by_exit.get(d, []):
            stake = pending.pop(i, None)
            if stake is not None:
                eq += stake * ror[i]
        peak = max(peak, eq)
        dd = eq / peak - 1
        # 2) open new rungs at this close
        scale = 1.0
        if dd_brake:
            scale = 1.0 if dd > -0.10 else 0.5 if dd > -0.20 else 0.25
        if vol_brake is not None and bool(vol_brake.get(d, False)):
            scale *= 0.5
        for i in by_entry.get(d, []):
            pending[i] = eq * budget / n_rungs * scale
        hist.append((d, eq, dd, sum(pending.values()) / eq if eq > 0 else np.nan))
    out = pd.DataFrame(hist, columns=["date", "equity", "drawdown", "open_risk"]).set_index("date")
    return out


def summarize_equity(eq: pd.DataFrame) -> dict:
    e = eq["equity"]
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    monthly = e.resample("ME").last().pct_change().dropna()
    cagr = e.iloc[-1] ** (1 / years) - 1
    mdd = float(eq["drawdown"].min())
    return {
        "cagr": float(cagr),
        "max_dd": mdd,
        "calmar": float(cagr / -mdd) if mdd < 0 else float("inf"),
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "pct_months_up": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "final": float(e.iloc[-1]),
    }
