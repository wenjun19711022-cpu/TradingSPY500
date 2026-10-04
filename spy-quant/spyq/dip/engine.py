"""Bar-level backtest of leveraged dip-buying on SPY (perp economics), any timeframe.

Mechanics (no look-ahead): signal on bar t's close -> buy at bar t+1's open. Each later bar:
stop hit if its low <= stop (fill at the stop, or at the open when it gaps below); the exit rule
is checked on the bar's close and filled at the next open; a time stop does the same after
`max_bars`. Funding accrues on calendar time held (weekends included). Every trade records its
maximum adverse excursion (MAE, from the entry to the lowest low before exit) — that number,
not the win rate, decides how much leverage a signal can carry without being liquidated.

Leverage is applied afterwards (trade returns scale linearly with leverage; liquidation does
not): see `levered_equity` and `leverage_table`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from spyq.dip.perp import PerpCosts
from spyq.dip.signals import features, signal

# exit rule and time stop per family (fixed with the signal definitions)
EXITS = {
    "rsi2": ("sma5", 10),
    "boll": ("sma20", 15),
    "wick": ("sma5", 10),
    "panic": ("sma20", 40),
    "ath_dd": ("ath", 100_000),
}


@dataclass(frozen=True)
class DipSpec:
    tf: str
    family: str
    stop_atr: float | None = None   # stop this many ATR below the entry (None = no stop)

    @property
    def key(self):
        return f"{self.tf}|{self.family}|stop{'-' if self.stop_atr is None else self.stop_atr}"

    def as_dict(self):
        return asdict(self)


def _bar_days(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.values, dtype="datetime64[s]").astype(np.int64) / 86400.0


def run(df: pd.DataFrame, spec: DipSpec, costs: PerpCosts | None = None, vix: pd.Series | None = None,
        feats: pd.DataFrame | None = None) -> pd.DataFrame:
    costs = costs or PerpCosts()
    f = feats if feats is not None else features(df, spec.tf, vix)
    sig = signal(f, spec.family, spec.tf).fillna(False).values
    exit_rule, max_bars = EXITS[spec.family]
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    a = f["atr"].values
    ex_level = {"sma5": f["sma5"].values, "sma10": f["sma10"].values, "sma20": f["sma20"].values}.get(exit_rule)
    days = _bar_days(df.index)
    n = len(df)
    trades = []
    busy_until = -1
    tiers = signal(f, spec.family, spec.tf).attrs.get("tier") if spec.family == "ath_dd" else None
    for i in np.flatnonzero(sig):
        if i + 1 >= n:
            break
        if spec.family != "ath_dd" and i < busy_until:
            continue  # one position at a time per stream (ath_dd tiers stack)
        e = i + 1
        p0 = o[e]
        target_ath = f["ath"].values[i] if exit_rule == "ath" else None
        stop = p0 - spec.stop_atr * a[i] if spec.stop_atr else None
        low_min = p0
        hi_max = p0
        exit_px = None
        reason = None
        j = e
        while j < n:
            if stop is not None and l[j] <= stop:
                exit_px, reason = (min(o[j], stop) if j > e else min(o[j], stop)), "stop"
                low_min = min(low_min, exit_px)
                break
            low_min = min(low_min, l[j])
            hi_max = max(hi_max, h[j])
            done = False
            if exit_rule == "ath":
                done = c[j] >= target_ath
            else:
                done = c[j] > ex_level[j]
            if done or (j - e + 1) >= max_bars:
                reason = "target" if done else "time"
                if j + 1 < n:
                    exit_px, j = o[j + 1], j + 1
                else:
                    exit_px = c[j]
                break
            j += 1
        if exit_px is None:  # still open at the end of data: mark at the last close
            exit_px, reason, j = c[-1], "open", n - 1
        held_days = max(days[j] - days[e], 0.0)
        gross = exit_px / p0 - 1
        fee = costs.round_trip()
        fund = costs.funding(held_days)
        trades.append({
            "signal_time": df.index[i], "entry_time": df.index[e], "exit_time": df.index[j],
            "entry": p0, "exit": exit_px, "bars": j - e, "days": held_days,
            "gross": gross, "fees": fee, "funding": fund, "net": gross - fee - fund,
            "mae": low_min / p0 - 1, "mfe": hi_max / p0 - 1, "reason": reason,
            "stop_dist": (p0 - stop) / p0 if stop is not None else np.nan,
            "tier": float(tiers.iloc[i]) if tiers is not None else np.nan,
        })
        busy_until = j
    t = pd.DataFrame(trades)
    if not t.empty:
        t = t.set_index("entry_time")
    return t


# ----------------------------------------------------------------------------- leverage

def levered_equity(trades: pd.DataFrame, lev: float, costs: PerpCosts | None = None, start: float = 1.0) -> dict:
    """Compound trades at constant leverage on the whole stream's equity (isolated margin).

    Intra-trade drawdown uses the trade's MAE; a trade whose MAE reaches the liquidation drop
    loses the full margin (equity -> ~0 for that stream). Trades that overlap (ath_dd tiers)
    are compounded in entry order, which overstates their independence — acceptable for a
    read-out, not for sizing several tiers at once (the leverage table caps those separately).
    """
    costs = costs or PerpCosts()
    if trades is None or trades.empty:
        return {"final": start, "max_dd": 0.0, "liquidations": 0, "curve": []}
    liq = costs.liq_drop(lev)
    eq, peak, mdd, nliq = start, start, 0.0, 0
    curve = []
    for t, r in trades.sort_index().iterrows():
        if -r["mae"] >= liq:
            low_eq = 0.0
            eq = eq * 0.0
            nliq += 1
        else:
            low_eq = eq * (1 + lev * r["mae"])
            eq = eq * (1 + lev * r["net"])
        mdd = min(mdd, low_eq / peak - 1)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        curve.append((r["exit_time"], eq))
        if eq <= 0:
            break
    return {"final": eq, "max_dd": mdd, "liquidations": nliq, "curve": curve}


def leverage_table(trades: pd.DataFrame, costs: PerpCosts | None = None, risk_per_trade=0.02,
                   cap: float = 5.0, safety: float = 1.5) -> dict:
    """Leverage a stream can carry, from its own (in-sample) trades.

      L_liq    = largest L whose liquidation drop >= safety x worst MAE seen
      L_kelly  = half of the Kelly fraction mean/var of net per-trade returns
      L_risk   = risk_per_trade / typical loss size (stop distance, or 95th pct MAE w/o stop)
      L        = min(L_liq, L_kelly, L_risk, cap), floored at 0
    """
    costs = costs or PerpCosts()
    if trades is None or len(trades) < 5:
        return {"L": 0.0}
    net = trades["net"]
    worst = -trades["mae"].min()
    l_liq = 1.0 / (safety * worst + costs.mmr) if worst > 0 else costs.max_leverage
    var = net.var()
    l_kelly = 0.5 * net.mean() / var if var > 0 else 0.0
    loss_size = trades["stop_dist"].median() if trades["stop_dist"].notna().any() else -trades["mae"].quantile(0.05)
    l_risk = risk_per_trade / loss_size if loss_size > 0 else cap
    L = max(0.0, min(l_liq, l_kelly, l_risk, cap, costs.max_leverage))
    return {"L": float(L), "L_liq": float(l_liq), "L_kelly": float(l_kelly), "L_risk": float(l_risk),
            "worst_mae": float(-worst), "mae_p95": float(trades["mae"].quantile(0.05)), "binding":
            min((("liquidation", l_liq), ("half-Kelly", l_kelly), ("risk/trade", l_risk), ("cap", cap)), key=lambda x: x[1])[0]}


def stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n": 0}
    net = trades["net"]
    n = len(net)
    wins = net > 0
    se = net.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    gl = -net[~wins].sum()
    return {"n": int(n), "win_rate": float(wins.mean()), "mean_net": float(net.mean()), "median_net": float(net.median()),
            "mean_gross": float(trades["gross"].mean()), "mean_cost": float((trades["fees"] + trades["funding"]).mean()),
            "t_stat": float(net.mean() / se) if se and se > 0 else 0.0,
            "profit_factor": float(net[wins].sum() / gl) if gl > 0 else float("inf"),
            "avg_days": float(trades["days"].mean()), "worst_mae": float(trades["mae"].min()),
            "mae_p95": float(trades["mae"].quantile(0.05)), "worst_net": float(net.min()),
            "stop_rate": float((trades["reason"] == "stop").mean())}
