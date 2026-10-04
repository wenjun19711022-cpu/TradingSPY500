"""1-minute path backtest for multi-timeframe bottom -> top scalps on an OKX SPY perp.

Entry: the buy signal on 1m bar t's close -> fill at bar t+1's open.
Stop: below the lowest low of the last 5 one-minute bars minus 0.02% (at least 0.08% away);
      trades whose stop would sit more than 0.6% away are skipped (too wide for high leverage).
Exits (first that happens, checked every minute):
  stop      bar low <= stop -> fill at the stop (or at the open if it gapped through)
  top3m     the latest closed 3m bar shows a top (overbought + top score >= 3) -> next open
  top5m     same on the 5m bar
  tp1.5R    limit at entry + 1.5 x stop distance
  time      90 minutes in the trade -> next open;  eod  15:55 -> that bar's close
Costs: OKX taker 0.05% + 0.01% slippage per side, funding 0.01% if the position is open
through the 16:00 UTC settlement. One position at a time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from spylev.dip.perp import PerpCosts
from spylev.scalp.signals import buy_signal, quality


@dataclass(frozen=True)
class ScalpSpec:
    thr5: int = 4
    trend: str = "none"
    exit: str = "top3m"          # top3m | top5m | tp1.5R
    thr3: int = 3
    thr1: int = 2
    max_minutes: int = 90
    max_stop: float = 0.006
    min_stop: float = 0.0008

    @property
    def key(self):
        return f"thr5={self.thr5}|trend={self.trend}|exit={self.exit}"

    def as_dict(self):
        return asdict(self)


def _funding_crossed(t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
    u0, u1 = t0.tz_convert("UTC"), t1.tz_convert("UTC")
    s = u0.normalize() + pd.Timedelta(hours=16)
    return (u0 < s <= u1) or (u0 < s + pd.Timedelta(days=1) <= u1)


def run(x: pd.DataFrame, spec: ScalpSpec, costs: PerpCosts | None = None) -> pd.DataFrame:
    costs = costs or PerpCosts()
    sig = buy_signal(x, spec.thr5, spec.thr3, spec.thr1, spec.trend).values
    q = quality(x).values
    o, h, l, c = (x[k].values for k in ("open", "high", "low", "close"))
    mos = x["mos"].values
    day = x.index.normalize()
    day_id = pd.factorize(day)[0]
    top3 = ((x["m3_top"] >= 3) & x["m3_t_osc"].fillna(False).astype(bool)).values
    top5 = ((x["m5_top"] >= 3) & x["m5_t_osc"].fillna(False).astype(bool)).values
    low5 = pd.Series(l, index=x.index).rolling(5, min_periods=1).min().values
    idx = x.index
    n = len(x)
    out = []
    i = 0
    while i < n - 1:
        if not sig[i] or day_id[i + 1] != day_id[i]:
            i += 1
            continue
        e = i + 1
        p0 = o[e]
        stop = min(low5[i] * (1 - 0.0002), p0 * (1 - spec.min_stop))
        dist = (p0 - stop) / p0
        if dist > spec.max_stop or dist <= 0:
            i += 1
            continue
        tp = p0 * (1 + 1.5 * dist) if spec.exit == "tp1.5R" else None
        exit_px, reason, j = None, None, e
        mae, mfe = 0.0, 0.0
        while j < n and day_id[j] == day_id[e]:
            if l[j] <= stop:
                exit_px, reason = min(o[j], stop), "stop"
                mae = min(mae, exit_px / p0 - 1)
                break
            mae = min(mae, l[j] / p0 - 1)
            mfe = max(mfe, h[j] / p0 - 1)
            if tp is not None and h[j] >= tp:
                exit_px, reason = max(o[j], tp), "target"
                break
            if mos[j] >= 385:  # 15:55
                exit_px, reason = c[j], "eod"
                break
            top = (spec.exit == "top3m" and top3[j]) or (spec.exit == "top5m" and top5[j])
            if top or (j - e + 1) >= spec.max_minutes:
                reason = "top" if top else "time"
                if j + 1 < n and day_id[j + 1] == day_id[e]:
                    exit_px, j = o[j + 1], j + 1
                else:
                    exit_px = c[j]
                break
            j += 1
        if exit_px is None:
            j = min(j, n - 1)
            exit_px, reason = c[j], "eod"
        gross = exit_px / p0 - 1
        fund = costs.funding_8h if _funding_crossed(idx[e], idx[j]) else 0.0
        fee = costs.round_trip()
        out.append({"signal": idx[i], "entry_time": idx[e], "exit_time": idx[j], "entry": p0, "exit": exit_px,
                    "stop": stop, "stop_dist": dist, "minutes": int(j - e), "gross": gross, "fees": fee,
                    "funding": fund, "net": gross - fee - fund, "mae": mae, "mfe": mfe, "reason": reason,
                    "quality": float(q[i]), "support": x["m5_support_name"].iloc[i] if "m5_support_name" in x else ""})
        i = j + 1 if j > i else i + 1
    t = pd.DataFrame(out)
    return t.set_index("entry_time") if not t.empty else t


def stats(t: pd.DataFrame, lev: float = 10.0) -> dict:
    if t is None or t.empty:
        return {"n": 0}
    net = t["net"]
    n = len(net)
    days = t.index.normalize().nunique()
    wins = net > 0
    gl = -net[~wins].sum()
    se = net.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    eq = (1 + lev * net).clip(lower=0).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    daily = (lev * net).groupby(t.index.normalize()).sum()
    return {
        "n": int(n), "per_day": float(n / max(days, 1)), "win_rate": float(wins.mean()),
        "mean_gross_bp": float(t["gross"].mean() * 1e4), "mean_net_bp": float(net.mean() * 1e4),
        "median_net_bp": float(net.median() * 1e4), "t_stat": float(net.mean() / se) if se and se > 0 else 0.0,
        "profit_factor": float(net[wins].sum() / gl) if gl > 0 else float("inf"),
        "avg_minutes": float(t["minutes"].mean()), "avg_stop_bp": float(t["stop_dist"].mean() * 1e4),
        "stop_rate": float((t["reason"] == "stop").mean()),
        f"lev{lev:g}_per_trade": float(lev * net.mean()), f"lev{lev:g}_final": float(eq.iloc[-1]),
        f"lev{lev:g}_max_dd": float(dd), f"lev{lev:g}_worst_day": float(daily.min()),
        f"lev{lev:g}_best_day": float(daily.max()),
    }
