"""Crash stress tests for option ladders.

The 2021-10..2026-10 sample has no crash of the 2020 / 2008 kind, and a short-premium
ladder's real risk is that *every open rung* loses at once. These templates are shaped on
well-known episodes (approximate magnitudes, not tick-exact replays):

  covid_2020        -34% over 23 sessions, VIX 15 -> 80   (Feb 19 - Mar 23, 2020)
  volmageddon_2018  -10% over 9 sessions,  VIX 11 -> 37   (Jan 26 - Feb 8, 2018)
  aug_2015          -11% over 4 sessions,  VIX 13 -> 40   (Aug 18 - Aug 24, 2015)
  q4_2018           -19% over 60 sessions, VIX 12 -> 36   (Sep 20 - Dec 24, 2018)
  gfc_2008          -40% over 30 sessions, VIX 25 -> 80   (late Sep - Nov 2008)

Each template is appended to a calm 60-session lead-in (so the ladder is fully built at
normal strikes), followed by a 20-session partial recovery. With real 2018-2020 minute data
(moomoo history) run the same ladder on the actual path instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spyq.backtest.daily import Spec, market_rows, simulate
from spyq.indicators.regime import features
from spyq.options.smile import Smile
from spyq.risk.sizing import ladder_equity, summarize_equity

SCENARIOS = {
    "covid_2020": dict(drop=-0.34, days=23, vix0=15, vix1=80),
    "volmageddon_2018": dict(drop=-0.10, days=9, vix0=11, vix1=37),
    "aug_2015": dict(drop=-0.11, days=4, vix0=13, vix1=40),
    "q4_2018": dict(drop=-0.19, days=60, vix0=12, vix1=36),
    "gfc_2008": dict(drop=-0.40, days=30, vix0=25, vix1=80),
}


def scenario_panel(drop, days, vix0, vix1, lead=60, tail=20, s0=500.0, start="2030-01-02"):
    n = lead + days + tail
    idx = pd.bdate_range(start, periods=n)
    calm_drift = 0.0004
    logp = np.r_[np.log(s0) + calm_drift * np.arange(lead),
                 np.log(s0) + calm_drift * (lead - 1) + np.linspace(0, np.log(1 + drop), days + 1)[1:]]
    trough = logp[-1]
    logp = np.r_[logp, trough + np.linspace(0, 0.4 * -np.log(1 + drop), tail + 1)[1:]]  # 40% retrace
    c = np.exp(logp)
    o = np.r_[c[0], c[:-1]]
    move = np.abs(np.diff(np.r_[logp[0], logp]))
    h = np.maximum(o, c) * np.exp(0.5 * move + 0.002)
    lo = np.minimum(o, c) * np.exp(-0.5 * move - 0.002)
    vix = np.r_[np.full(lead, vix0), np.exp(np.linspace(np.log(vix0), np.log(vix1), days + 1)[1:]),
                np.exp(np.linspace(np.log(vix1), np.log((vix0 + vix1) / 2), tail + 1)[1:])]
    p = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx)
    p["vix_close"] = vix
    p["vix9d_close"] = vix * np.where(np.arange(n) >= lead, 1.10, 0.92)  # inverted in stress
    p["vix3m_close"] = vix * np.where(np.arange(n) >= lead, 0.85, 1.12)
    p["vix1d_close"] = p["vix9d_close"] * 0.95
    p["vix1d_open"] = p["vix1d_close"]
    return p


def run_stress(spec: Spec, budget: float, smile: Smile | None = None, n_rungs: int | None = None) -> pd.DataFrame:
    smile = smile or Smile()
    n_rungs = n_rungs or (int(spec.mode[1:-6]) if spec.mode.startswith("n") else 1)
    out = {}
    for name, sc in SCENARIOS.items():
        p = scenario_panel(**sc)
        rows = market_rows(p, features(p), spec.mode, smile)
        tr = simulate(rows, spec, smile, panel=p)
        if "entry" in rows:
            tr["entry"] = rows.loc[tr.index, "entry"].values
        eq = ladder_equity(tr, budget, n_rungs, dd_brake=True)
        st = summarize_equity(eq)
        out[name] = {"max_dd": st["max_dd"], "trades": len(tr), "losers": int((tr["pnl"] < 0).sum()),
                     "worst_ror": float(tr["ror"].min()), **{k: sc[k] for k in ("drop", "days", "vix0", "vix1")}}
    return pd.DataFrame(out).T
