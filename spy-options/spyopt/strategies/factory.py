"""Pre-registered strategy grid and gates (fixed before looking at validation/test data).

Tenors: 0DTE at the open, 1DTE from the prior close, and 5-, 10- and 20-session trades opened
at every close (laddered, so each tail event is hit by N overlapping trades).
Structures: iron condor, put / call credit spreads, trend-conditioned credit spread,
iron fly, long straddle (buy vol), trend-conditioned debit spread (directional).
Strike distance k is in expected moves (EM = F * atm * sqrt(T)); wings in EM too.

Gates (same spirit as the earlier strategy-factory page):
  G1  in-sample   (start .. 2024-12-31): n>=30, mean return-on-risk > 0, t >= 2.0
                  (Newey-West for overlapping trades), profit factor >= 1.15, max drawdown at
                  2% risk per position (split across the N open ladder rungs) >= -20%
  G2  validation  (2025-01-01 .. 2025-09-30): mean RoR > 0 and PF >= 1.0  (n >= 10)
  G1b robustness  (amendment, added after the first full run showed "penny" winners and
                  model-level dependence): the in-sample edge must survive (a) doubled bid-ask
                  spreads and (b) a 5% lower implied-vol level, and earn >= $1 per contract.
  G3  holdout     (2025-10-01 .. 2026-10-02): reported once; pass = mean RoR > 0
Every row also carries the deflated Sharpe ratio for the full number of configurations.

Disclosure: a smoke test touched the full 2021-2026 sample before the grid was frozen, and
the ladder/robustness amendments were made after a first full run, so VAL and TEST are not
perfectly virgin. Treat survivors as candidates for paper trading, not as proven edges.
"""
from __future__ import annotations

from itertools import product

from spyopt.backtest.daily import Spec

SPLITS = {
    "IS": ("1900-01-01", "2024-12-31"),
    "VAL": ("2025-01-01", "2025-09-30"),
    "TEST": ("2025-10-01", "2100-01-01"),
}
GATES = {
    "G1": dict(n_min=30, t_min=2.0, pf_min=1.15, dd_min=-0.20),
    "G1b": dict(min_pnl_per_contract=1.0),
    "G2": dict(n_min=10, pf_min=1.0),
}
MODES = ["0dte_open", "1dte_close", "n5_close", "n10_close", "n20_close"]
SHORT_DATED = {"0dte_open", "1dte_close"}


def grid() -> list[Spec]:
    specs: list[Spec] = []
    for mode in MODES:
        filters = ["none", "vrp", "calm", "combo"]
        if mode in SHORT_DATED:
            filters += ["noevent"]
        if mode == "0dte_open":
            filters += ["gap"]
        exits_otm = ["hold", "touch"] + ([] if mode in SHORT_DATED else ["tp50"])
        otm = []
        for k, w in product((0.75, 1.0, 1.25, 1.5, 2.0), (0.5, 1.0)):
            otm.append(("iron_condor", k, w))
        for name in ("put_spread", "call_spread"):
            for k, w in product((0.5, 1.0, 1.5, 2.0), (0.5, 1.0)):
                otm.append((name, k, w))
        for k in (0.75, 1.0, 1.5):
            otm.append(("trend_spread", k, 0.5))
        for filt in filters:
            for (name, k, w), ex in product(otm, exits_otm):
                specs.append(Spec(mode, name, k, k, w, ex, filt))
            for w in (0.5, 1.0, 1.5):
                specs.append(Spec(mode, "iron_fly", 0.0, 0.0, w, "hold", filt))
            specs.append(Spec(mode, "long_straddle", 0.0, 0.0, 1.0, "hold", filt))
            for w in (0.5, 1.0):
                specs.append(Spec(mode, "trend_debit", 0.0, 0.0, w, "hold", filt))
    return specs


def passes_g1(st: dict) -> bool:
    g = GATES["G1"]
    return (st.get("n", 0) >= g["n_min"] and st["mean_ror"] > 0 and st["t_stat"] >= g["t_min"]
            and st["profit_factor"] >= g["pf_min"] and st["max_dd"] >= g["dd_min"])


def passes_g2(st: dict) -> bool:
    g = GATES["G2"]
    return st.get("n", 0) >= g["n_min"] and st["mean_ror"] > 0 and st["profit_factor"] >= g["pf_min"]


def overlap(mode: str) -> int:
    return int(mode[1:-6]) if mode.startswith("n") and mode.endswith("_close") else 1
