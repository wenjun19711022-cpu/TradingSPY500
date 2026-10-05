"""Is the playbook still working? Replays the noise-area and overnight sleeves on every minute
bar in the store after the backtest period, and compares them with the backtest.

    python -m spylev.data.crawl --source futu --years 2     # your moomoo minute bars into the store
    python -m spylev.edge.review --since 2025-01-01

For each sleeve it reports the trades since `since`, the net result on futures costs, and two
checks against the backtest (results/edge_study.json):
  vs_zero      t-statistic of the mean against 0 (is it still positive?)
  vs_backtest  t-statistic of the mean against the backtest mean (is it decaying?)
A sleeve is flagged "decaying" after at least 50 trades when the mean is more than 2 standard
errors below the backtest mean; the playbook then halves that sleeve until a re-test.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.data import store
from spylev.edge.evaluate import net_returns
from spylev.edge.strategies import noise, overnight, panel

ROOT = Path(__file__).resolve().parents[2]


def review(bars: pd.DataFrame, since: str, study: dict | None = None, min_trades: int = 50) -> dict:
    study = study or json.loads((ROOT / "results" / "edge_study.json").read_text())
    start = pd.Timestamp(since)
    # keep 40 calendar days before `since` so the 14-day noise band exists on day one
    p = panel(bars[bars.index.tz_localize(None) >= start - pd.Timedelta(days=40)])
    out = {"since": since, "sessions": int((p.days >= start).sum()), "sleeves": {}}
    for key, fn in (("noise", noise), ("overnight", overnight)):
        t = fn(p)
        t = t[pd.DatetimeIndex(t["date"]) >= start] if len(t) else t
        ref = study["strategies"][key]["regimes"]["mes"]["ALL"]
        rec = {"n": int(len(t)), "backtest_net_bp": ref["net_bp"]}
        if len(t) >= 2:
            net = net_returns(t, "mes")
            se = net.std(ddof=1) / np.sqrt(len(net))
            rec.update(net_bp=float(net.mean() * 1e4), win_rate=float((net > 0).mean()), total_pct=float(net.sum() * 100),
                       vs_zero=float(net.mean() / se) if se > 0 else 0.0,
                       vs_backtest=float((net.mean() - ref["net_bp"] / 1e4) / se) if se > 0 else 0.0)
            rec["status"] = ("decaying" if len(t) >= min_trades and rec["vs_backtest"] < -2 else
                             "ok" if len(t) >= min_trades else "too few trades")
        else:
            rec["status"] = "no data"
        out["sleeves"][key] = rec
    return out


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args(argv)
    bars = store.load(a.symbol, "1m")
    if bars.empty or bars.index[-1].tz_localize(None) < pd.Timestamp(a.since):
        print(f"store has no {a.symbol} minute bars after {a.since}. Pull them first:\n"
              f"  python -m spylev.data.crawl --source futu --years 2")
        return
    r = review(bars, a.since)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    (ROOT / "results" / "review_live.json").write_text(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
