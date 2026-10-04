"""Run the pre-registered grid through IS -> VAL -> TEST and collect everything for reports."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from spyq.backtest.daily import market_rows, simulate
from spyq.backtest.metrics import trade_stats, yearly
from spyq.indicators.regime import features
from spyq.options.costs import CostModel
from spyq.options.smile import Smile
from spyq.strategies.factory import GATES, SPLITS, grid, overlap, passes_g1, passes_g2


def _cut(trades: pd.DataFrame, split: str) -> pd.DataFrame:
    a, b = SPLITS[split]
    return trades[(trades.index >= a) & (trades.index <= b)]


def run_factory(panel: pd.DataFrame, smile: Smile | None = None, costs: CostModel | None = None,
                specs=None, keep_trades: bool = False):
    """Returns (summary DataFrame one row per spec, dict key -> full trades if keep_trades)."""
    warnings.filterwarnings("ignore")
    smile = smile or Smile()
    costs = costs or CostModel()
    feats = features(panel)
    specs = specs or grid()
    rows_cache = {}
    out, trades_all = [], {}
    n_trials = len(specs)
    for sp in specs:
        if sp.mode not in rows_cache:
            rows_cache[sp.mode] = market_rows(panel, feats, sp.mode, smile)
        tr = simulate(rows_cache[sp.mode], sp, smile, costs, panel=panel)
        rec = {"key": sp.key, **sp.as_dict()}
        for split in ("IS", "VAL", "TEST"):
            st = trade_stats(_cut(tr, split), n_trials=n_trials if split == "IS" else 1, overlap=overlap(sp.mode))
            for k in ("n", "win_rate", "mean_ror", "t_stat", "profit_factor", "max_dd", "cagr",
                      "mean_pnl_per_contract", "avg_credit_per_contract", "avg_risk_per_contract",
                      "sharpe_ann", "dsr", "worst_ror", "cvar5_ror", "n_losses"):
                rec[f"{split}_{k}"] = st.get(k, np.nan)
            rec[f"{split}_ok"] = (passes_g1(st) if split == "IS" else passes_g2(st)) if st.get("n", 0) else False
        full = trade_stats(tr, n_trials=n_trials, overlap=overlap(sp.mode))
        rec.update({f"ALL_{k}": full.get(k, np.nan) for k in ("n", "win_rate", "mean_ror", "t_stat", "max_dd", "cagr")})
        out.append(rec)
        if keep_trades:
            trades_all[sp.key] = tr
    df = pd.DataFrame(out).set_index("key")
    df["G1"] = df["IS_ok"]
    # G1b: re-price every G1 survivor under doubled spreads and a 5% lower IV level (IS only)
    df["G1b"] = False
    stress = {
        "spread_x2": (smile, CostModel(**{**costs.__dict__, "stress": 2.0 * costs.stress})),
        "iv_-5%": (Smile(shapes=smile.shapes, level={k: v * 0.95 for k, v in smile.level.items()}), costs),
    }
    stress_rows = {}
    for key in df.index[df["G1"]]:
        sp = next(s for s in specs if s.key == key)
        ok = df.at[key, "IS_mean_pnl_per_contract"] >= GATES["G1b"]["min_pnl_per_contract"]
        for lab, (sm, cm) in stress.items():
            rk = (lab, sp.mode)
            if rk not in stress_rows:
                stress_rows[rk] = market_rows(panel, feats, sp.mode, sm)
            st = trade_stats(_cut(simulate(stress_rows[rk], sp, sm, cm, panel=panel), "IS"), overlap=overlap(sp.mode))
            df.at[key, f"IS_mean_ror_{lab}"] = st.get("mean_ror", np.nan)
            ok = ok and st.get("mean_ror", -1) > 0
        df.at[key, "G1b"] = bool(ok)
    df["G2"] = df["G1b"] & df["VAL_ok"]
    df["G3"] = df["G2"] & (df["TEST_mean_ror"] > 0)
    return df, trades_all


def best_by_family(df: pd.DataFrame, score="IS_t_stat") -> pd.DataFrame:
    """Best in-sample configuration per (mode, structure) — what the dashboard tabulates."""
    d = df[df["IS_n"] >= 20].copy()
    idx = d.groupby(["mode", "structure"])[score].idxmax()
    return d.loc[idx.values].sort_values(["mode", score], ascending=[True, False])


def sensitivity(panel, spec, smile: Smile, costs: CostModel):
    """Re-run one spec under level shifts (+/-5% ATM) and doubled spreads."""
    feats = features(panel)
    res = {}
    for lab, sm, cm in [
        ("base", smile, costs),
        ("iv_-5%", Smile(shapes=smile.shapes, level={k: v * 0.95 for k, v in smile.level.items()}), costs),
        ("iv_+5%", Smile(shapes=smile.shapes, level={k: v * 1.05 for k, v in smile.level.items()}), costs),
        ("spread_x2", smile, CostModel(**{**costs.__dict__, "stress": 2.0})),
        ("no_costs", smile, CostModel(commission=0, min_spread=0, spread_coef=0, max_spread=0)),
    ]:
        rows = market_rows(panel, feats, spec.mode, sm)
        tr = simulate(rows, spec, sm, cm, panel=panel)
        st = trade_stats(tr, overlap=overlap(spec.mode))
        res[lab] = {k: st.get(k) for k in ("n", "win_rate", "mean_ror", "t_stat", "profit_factor", "max_dd")}
    return pd.DataFrame(res).T


__all__ = ["run_factory", "best_by_family", "sensitivity", "yearly"]
