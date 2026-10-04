"""Backtest of the range idea: at 10:00 forecast the day's range, rest a limit buy near the low
end, stop just under the forecast low, take profit a bit higher. One trade a day, flat by 15:55.

Fills are conservative on 1-minute bars:
  limit buy     fills when a bar trades at least one cent below it (a touch is not a fill);
                at the bar's open if the bar opens below it
  stop          fills at the stop, or at the open if a bar opens below it; checked first
  take profit   fills when a bar trades at least one cent above it, never in the fill bar
  same bar      if a bar reaches both stop and target, the stop wins
Costs (OKX): limit entry and take-profit pay maker 0.02%; stop and the 15:55 exit pay taker
0.05% + 0.01% slippage; funding 0.01% when the position is open over 16:00 UTC.

Each trade also records the fair probability of reaching the target before the stop, from
the volatility forecast alone (no edge). The strategy has an edge only if it wins more often
than that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from spylev.dip.perp import PerpCosts
from spylev.range.model import RangeModel, p_bracket, p_touch_below
from spylev.scalp.engine import _funding_crossed

TICK = 0.01
SUPPORT_KEYS = ("pdl", "pdc", "s1", "s2", "val", "poc", "pml", "or_low")


@dataclass(frozen=True)
class RangeSpec:
    t0: int = 30              # decision minute after the open (30 = 10:00 ET)
    anchor: str = "sigma"     # sigma: entry a fixed number of forecast sigmas below the 10:00 price
                              # level: entry just above the highest support inside the forecast band
    entry: float = 0.5        # sigma anchor only
    stop_p: float = 0.8       # stop 0.02% under the forecast rest-of-day low that holds with prob p
    tp: str = "0.5s"          # 0.5s: entry + 0.5 forecast sigma ; mid: middle of the forecast band
    last_fill: int = 330      # no new fills after 15:00
    flat: int = 385           # flat at 15:55

    @property
    def key(self):
        e = f"entry={self.entry:g}s" if self.anchor == "sigma" else "entry=support"
        return f"t0={self.t0}|{e}|stop=p{self.stop_p:g}|tp={self.tp}"

    def as_dict(self):
        return asdict(self)


def plan(model: RangeModel, s0: float, sigma: float, spec: RangeSpec, supports: dict | None = None) -> dict | None:
    """The order ticket made at the decision minute, or None when the geometry doesn't work."""
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    lo, hi = model.band(s0, sigma, spec.stop_p)
    lo_mid, hi_mid = model.band(s0, sigma, 0.5)
    if spec.anchor == "sigma":
        entry = s0 * np.exp(-spec.entry * sigma)
        why = f"{spec.entry:g} 个波动单位下方"
    else:
        top = s0 * np.exp(-0.25 * sigma)
        cands = {k: v for k, v in (supports or {}).items() if v is not None and np.isfinite(v) and lo * 1.0005 < v < top}
        if not cands:
            return None
        name, lvl = max(cands.items(), key=lambda kv: kv[1])
        entry = lvl * 1.0003
        why = name
    stop = min(lo * (1 - 0.0002), entry * (1 - 0.0008))
    tp = entry * np.exp(0.5 * sigma) if spec.tp == "0.5s" else np.sqrt(lo_mid * hi_mid) if spec.tp == "mid" else None
    if tp is None or tp < entry * (1 + 0.0008) or stop >= entry:
        return None
    return {"entry": float(entry), "stop": float(stop), "tp": float(tp), "band_low": float(lo), "band_high": float(hi),
            "why": why}


def _var_left(model: RangeModel, sigma: float, mos: float) -> float:
    return model.sigma_between(sigma, mos) ** 2


def simulate(day: pd.DataFrame, tk: dict, model: RangeModel, sigma: float, spec: RangeSpec, costs: PerpCosts) -> dict:
    """Walk one session's 1m bars from the decision minute."""
    o, h, l, c = (day[k].values for k in ("open", "high", "low", "close"))
    mos = ((day.index.hour * 60 + day.index.minute) - 570).values
    entry, stop, tp = tk["entry"], tk["stop"], tk["tp"]
    start = np.searchsorted(mos, spec.t0)
    out = {**tk, "filled": False, "reason": "nofill",
           "p_fill": p_touch_below(c[start - 1] if start > 0 else o[0], entry, _var_left(model, sigma, spec.t0))}
    fill_i, px = None, None
    for i in range(start, len(day)):
        if mos[i] > spec.last_fill:
            break
        if o[i] <= entry:
            fill_i, px = i, o[i]
        elif l[i] <= entry - TICK:
            fill_i, px = i, entry
        if fill_i is not None:
            break
    if fill_i is None:
        return out
    up, dn, _ = p_bracket(px, stop, tp, _var_left(model, sigma, mos[fill_i]))
    out.update(filled=True, fill_time=day.index[fill_i], fill=float(px), p_fair=up, p_fair_stop=dn)
    exit_px, reason, j = None, None, fill_i
    if l[fill_i] <= stop:  # stopped inside the fill bar
        exit_px, reason = min(stop, px), "stop"
    else:
        for j in range(fill_i + 1, len(day)):
            if o[j] <= stop:
                exit_px, reason = o[j], "stop"
                break
            if l[j] <= stop:
                exit_px, reason = stop, "stop"
                break
            if h[j] >= tp + TICK:
                exit_px, reason = max(o[j], tp), "tp"
                break
            if mos[j] >= spec.flat:
                exit_px, reason = c[j], "eod"
                break
        if exit_px is None:
            j = len(day) - 1
            exit_px, reason = c[j], "eod"
    gross = exit_px / px - 1
    fee = costs.maker + (costs.maker if reason == "tp" else costs.taker + costs.slippage)
    fund = costs.funding_8h if _funding_crossed(day.index[fill_i], day.index[j]) else 0.0
    out.update(exit_time=day.index[j], exit=float(exit_px), reason=reason, gross=gross, fees=fee, funding=fund,
               net=gross - fee - fund, minutes=int(j - fill_i), stop_dist=(px - stop) / px, tp_dist=(tp - px) / px)
    return out


def run(m1: pd.DataFrame, fc: pd.DataFrame, model: RangeModel, spec: RangeSpec, levels: pd.DataFrame | None = None,
        costs: PerpCosts | None = None) -> pd.DataFrame:
    """fc: forecast table (spylev.range.model.forecast_table) for decision minute spec.t0."""
    costs = costs or PerpCosts()
    days = m1.index.normalize()
    by_day = {d.tz_localize(None): g for d, g in m1.groupby(days)}
    rows = []
    for date, r in fc.iterrows():
        g = by_day.get(date)
        if g is None or not np.isfinite(r.get("sigma", np.nan)):
            continue
        sup = None
        if spec.anchor == "level":
            sup = {k: (levels.loc[date, k] if levels is not None and date in levels.index and k in levels.columns else None)
                   for k in SUPPORT_KEYS if k != "or_low"}
            sup["or_low"] = r["l0"]
            fl = np.floor(r["s0"] / 5) * 5
            sup["round5"] = fl if fl < r["s0"] else fl - 5
        tk = plan(model, r["s0"], r["sigma"], spec, sup)
        if tk is None:
            rows.append({"date": date, "filled": False, "reason": "noplan", "sigma": r["sigma"], "s0": r["s0"]})
            continue
        res = simulate(g, tk, model, r["sigma"], spec, costs)
        rows.append({"date": date, "sigma": r["sigma"], "s0": r["s0"], **res})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def stats(t: pd.DataFrame, levs=(1, 5, 10, 20)) -> dict:
    if t is None or t.empty:
        return {"days": 0, "n": 0}
    f = t[t["filled"]] if "filled" in t else t
    out = {"days": int(len(t)), "plans": int((t["reason"] != "noplan").sum()), "n": int(len(f))}
    if not len(f):
        return out
    net = f["net"].astype(float)
    wins = f["reason"] == "tp"
    se = net.std(ddof=1) / np.sqrt(len(net)) if len(net) > 1 else np.nan
    gl = -net[net <= 0].sum()
    edge = (wins.astype(float) - f["p_fair"]).astype(float)
    out.update({
        "fill_rate": float(len(f) / max(out["plans"], 1)), "win_rate": float(wins.mean()),
        "fair_win_rate": float(f["p_fair"].mean()), "edge_vs_fair": float(edge.mean()),
        "edge_t": float(edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge)))) if len(edge) > 1 and edge.std() > 0 else 0.0,
        "stop_rate": float((f["reason"] == "stop").mean()), "eod_rate": float((f["reason"] == "eod").mean()),
        "mean_gross_bp": float(f["gross"].mean() * 1e4), "mean_net_bp": float(net.mean() * 1e4),
        "t_stat": float(net.mean() / se) if se and se > 0 else 0.0,
        "profit_factor": float(net[net > 0].sum() / gl) if gl > 0 else float("inf"),
        "avg_stop_bp": float(f["stop_dist"].mean() * 1e4), "avg_tp_bp": float(f["tp_dist"].mean() * 1e4),
        "avg_minutes": float(f["minutes"].mean()),
    })
    for L in levs:
        eq = (1 + L * net).clip(lower=0).cumprod()
        out[f"lev{L}_final"] = float(eq.iloc[-1])
        out[f"lev{L}_max_dd"] = float((eq / eq.cummax() - 1).min())
        out[f"lev{L}_worst_trade"] = float(L * net.min())
    return out
