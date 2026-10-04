"""1m/3m/5m confluence bottom -> 3m/5m top scalps with leverage -> results/scalp_study.json

    python -m spylev.data.minute_hist      # public 2022-10..2024-12 minute bars (once)
    python -m spylev.data.crawl --source futu --years 3     # your own moomoo history (optional)
    python scripts/run_scalp_study.py

Pre-registered grid (fixed before any P&L was looked at): 5m threshold {3,4} x daily trend
filter {none, up} x exit {3m top, 5m top, 1.5R target} = 12 configs. Splits on the public data:
IS 2022-10..2023-12, VAL 2024-01..06, TEST 2024-07..12; anything after 2024 in the store (your
moomoo data) is reported separately as an untouched holdout.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spylev.data import store  # noqa: E402
from spylev.data.history import spy_daily  # noqa: E402
from spylev.data.minute_hist import premarket_levels  # noqa: E402
from spylev.dip.engine import DipSpec  # noqa: E402
from spylev.dip.engine import run as dip_run  # noqa: E402
from spylev.dip.perp import PerpCosts  # noqa: E402
from spylev.scalp.engine import ScalpSpec, run, stats  # noqa: E402
from spylev.scalp.signals import build, buy_signal, quality  # noqa: E402
from spylev.ta import daily_levels  # noqa: E402

OUT = ROOT / "results"
SPLITS = {"IS": ("2022-10-01", "2023-12-31"), "VAL": ("2024-01-01", "2024-06-30"),
          "TEST": ("2024-07-01", "2024-12-31"), "HOLDOUT": ("2025-01-01", "2100-01-01")}
GRID = [ScalpSpec(t5, tr, ex) for t5 in (3, 4) for tr in ("none", "up") for ex in ("top3m", "top5m", "tp1.5R")]


def cut(t, a, b):
    if t is None or t.empty:
        return t
    d = t.index.tz_localize(None)
    return t[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b) + pd.Timedelta(hours=23))]


def forward_table(x: pd.DataFrame, mask_period) -> list:
    o, c, day = x["open"], x["close"], x.index.normalize()
    nxt = o.groupby(day).shift(-1)
    base = (x["mos"] >= 5) & (x["mos"] <= 330) & mask_period
    rows = []
    for spec in ((3, "none"), (4, "none"), (4, "up")):
        s = buy_signal(x, thr5=spec[0], trend=spec[1]) & mask_period
        first = s & ~s.shift(fill_value=False)
        row = {"thr5": spec[0], "trend": spec[1], "n": int(first.sum())}
        for H in (3, 5, 15, 30, 60, 120):
            f = c.groupby(day).shift(-H) / nxt - 1
            a, b = f[first].dropna(), f[base].dropna()
            row[f"{H}m"] = float(a.mean() * 1e4)
            row[f"{H}m_base"] = float(b.mean() * 1e4)
            row[f"{H}m_t"] = float((a.mean() - b.mean()) / (a.std() / np.sqrt(len(a)))) if len(a) > 2 else None
        rows.append(row)
    return rows


def random_baseline(x: pd.DataFrame, n_signals: int, seed=1) -> pd.DataFrame:
    """Same engine, same stops/exits, entries at random minutes (same count)."""
    rng = np.random.default_rng(seed)
    elig = np.flatnonzero(((x["mos"] >= 5) & (x["mos"] <= 360)).values)
    pick = rng.choice(elig, size=min(n_signals, len(elig)), replace=False)
    y = x.copy()
    for col in ("m5_bottom", "m5_prev_bottom", "m3_bottom", "m3_prev_bottom", "m1_bottom_recent"):
        y[col] = 0
    for col in ("m5_b_osc", "m5_b_support", "m3_b_osc", "confirm"):
        y[col] = False
    y.iloc[pick, y.columns.get_loc("m5_bottom")] = 6
    y.iloc[pick, y.columns.get_loc("m3_bottom")] = 6
    y.iloc[pick, y.columns.get_loc("m1_bottom_recent")] = 6
    for col in ("m5_b_osc", "m5_b_support", "m3_b_osc", "confirm"):
        y.iloc[pick, y.columns.get_loc(col)] = True
    return run(y, ScalpSpec(4, "none", "top5m"))


def daily_timing(x: pd.DataFrame) -> dict:
    spy, _ = spy_daily()
    t = dip_run(spy.assign(volume=0.0), DipSpec("1d", "rsi2", 2.5), PerpCosts())
    days = x.index.normalize().tz_localize(None)
    t = t[(t.index >= days.min()) & (t.index <= days.max())]
    sig = buy_signal(x, thr5=4)
    rows = []
    for ed in t.index:
        m = days == ed
        if not m.any():
            continue
        xd, sd = x[m], sig[m].values
        hits = np.flatnonzero(sd)
        if len(hits) and hits[0] + 1 < len(xd):
            px, when = xd["open"].iloc[hits[0] + 1], xd.index[hits[0] + 1].strftime("%H:%M")
        else:
            px, when = xd["close"].iloc[-1], "收盘"
        rows.append({"date": str(ed.date()), "open": float(xd["open"].iloc[0]), "mtf_entry": float(px), "when": when,
                     "improve_bp": float((xd["open"].iloc[0] / px - 1) * 1e4)})
    imp = pd.Series([r["improve_bp"] for r in rows])
    return {"n": len(rows), "mean_bp": float(imp.mean()) if len(imp) else None, "median_bp": float(imp.median()) if len(imp) else None,
            "better_share": float((imp > 0).mean()) if len(imp) else None,
            "se_bp": float(imp.std() / np.sqrt(len(imp))) if len(imp) > 1 else None, "rows": rows}


def expectation_table(t: pd.DataFrame) -> list:
    """What the alert pop-up quotes: by signal quality bucket, from IS+VAL trades only."""
    if t is None or t.empty:
        return []
    rows = []
    for lo, hi, lab in ((0, 11, "普通"), (11, 13, "较强"), (13, 19, "最强")):
        g = t[(t["quality"] >= lo) & (t["quality"] < hi)]
        if len(g) < 10:
            continue
        w = g[g["net"] > 0]
        rows.append({"bucket": lab, "q_lo": lo, "q_hi": hi, "n": len(g), "win_rate": float((g["net"] > 0).mean()),
                     "mean_net_bp": float(g["net"].mean() * 1e4), "median_win_bp": float(w["gross"].median() * 1e4) if len(w) else None,
                     "avg_minutes": float(g["minutes"].mean()), "avg_stop_bp": float(g["stop_dist"].mean() * 1e4)})
    return rows


def replay_day(x: pd.DataFrame, t: pd.DataFrame, date: str) -> dict:
    d = x[x.index.normalize().tz_localize(None) == pd.Timestamp(date)]
    sig = buy_signal(x, thr5=4).reindex(d.index)
    tt = t[t.index.normalize().tz_localize(None) == pd.Timestamp(date)] if t is not None and not t.empty else t
    bars = [[int((ts.tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds()), round(r.open, 2), round(r.high, 2), round(r.low, 2),
             round(r.close, 2), int(r.volume), int(r.m5_bottom if pd.notna(r.m5_bottom) else 0), int(r.m3_bottom if pd.notna(r.m3_bottom) else 0),
             int(r.m1_bottom), int(r.m5_top if pd.notna(r.m5_top) else 0), int(r.m3_top if pd.notna(r.m3_top) else 0), bool(s)]
            for (ts, r), s in zip(d.iterrows(), sig.values)]
    trades = [] if tt is None or tt.empty else [
        {"entry_time": str(i)[11:16], "entry": float(r.entry), "stop": float(r.stop), "exit_time": str(r.exit_time)[11:16],
         "exit": float(r.exit), "reason": r.reason, "net_bp": float(r.net * 1e4)} for i, r in tt.iterrows()]
    return {"date": date, "bars": bars, "trades": trades}


def main():
    warnings.filterwarnings("ignore")
    m1 = store.load("SPY", "1m")
    lv = daily_levels(m1).join(premarket_levels())
    x = build(m1, lv)
    nd = x.index.normalize().nunique()
    out = {"generated": str(pd.Timestamp.now(tz="UTC")), "data": {"start": str(x.index[0])[:10], "end": str(x.index[-1])[:10],
           "sessions": int(nd), "bars": len(x)}, "splits": SPLITS, "costs": PerpCosts().__dict__, "grid": []}
    period_isval = x.index.tz_localize(None) < pd.Timestamp("2024-07-01")
    trades_best = None
    for sp in GRID:
        t = run(x, sp)
        rec = {"key": sp.key, **sp.as_dict()}
        for k, (a, b) in SPLITS.items():
            rec[k] = stats(cut(t, a, b))
        rec["ALL"] = stats(t)
        g = rec["IS"]
        rec["G1"] = bool(g.get("n", 0) >= 50 and g["mean_net_bp"] > 0 and g["t_stat"] >= 2 and g["profit_factor"] >= 1.2)
        rec["G2"] = bool(rec["G1"] and rec["VAL"].get("mean_net_bp", -1) > 0)
        rec["G3"] = bool(rec["G2"] and rec["TEST"].get("mean_net_bp", -1) > 0)
        if len(t):
            rec["maker_net_bp"] = float((t["gross"] - PerpCosts().round_trip(True, True) - t["funding"]).mean() * 1e4)
            rec["zero_cost_bp"] = float(t["gross"].mean() * 1e4)
            rec["expectation"] = expectation_table(cut(t, "2022-10-01", "2024-06-30"))
        out["grid"].append(rec)
        print(f"{sp.key:<38} n={rec['ALL'].get('n', 0):>5} gross={rec['ALL'].get('mean_gross_bp', 0):+.2f}bp net={rec['ALL'].get('mean_net_bp', 0):+.2f}bp "
              f"win={rec['ALL'].get('win_rate', 0):.0%} 10x final={rec['ALL'].get('lev10_final', 0):.3f} G={int(rec['G1'])}{int(rec['G2'])}{int(rec['G3'])}")
        if sp.key == "thr5=4|trend=none|exit=top5m":
            trades_best = t
    out["forward"] = forward_table(x, period_isval)
    rb = random_baseline(x, len(trades_best))
    out["random_baseline"] = stats(rb)
    out["daily_timing"] = daily_timing(x)
    # a replay day for the demo UI: the IS day with the most trades of the reference config
    if trades_best is not None and len(trades_best):
        cnt = trades_best.groupby(trades_best.index.normalize()).size()
        days = [str(d.date()) for d in cnt.sort_values(ascending=False).index[:3]]
        out["replay"] = [replay_day(x, trades_best, d) for d in days]
    (OUT / "scalp_study.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda v: None))
    print("random baseline:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["random_baseline"].items() if k in ("n", "win_rate", "mean_gross_bp", "mean_net_bp")})
    print("daily timing:", {k: v for k, v in out["daily_timing"].items() if k != "rows"})
    print("wrote", OUT / "scalp_study.json")


if __name__ == "__main__":
    main()
