"""Review round 2026-10-05: the hypotheses H43-H48 registered in research/REGISTRY.md
-> results/review.json

    python scripts/run_review.py

H43 robustness table for the noise-area breakout, H44 walk-forward parameter choice,
H45 volatility-managed overnight sleeve, H46 the overnight hold on daily data never used
before (2025-2026) and by decade since 1993, H47 daily RSI2 since 2025, H48 risk-parity weights.
All on futures (MES) costs; splits as in run_edge_study.py.
"""
from __future__ import annotations

import itertools
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
from spylev.edge.evaluate import RATES, daily, net_returns  # noqa: E402
from spylev.edge.strategies import noise, overnight, panel  # noqa: E402

OUT = ROOT / "results"
SPLITS = {"IS": ("2011-01-01", "2015-12-31"), "VAL": ("2016-01-01", "2018-12-31"), "TEST": ("2022-10-01", "2024-12-31")}
# rough yearly fed funds / S&P dividend yield (%) before 2011, for the futures carry of overnight holds
RATES_OLD = {1993: (3.0, 2.8), 1994: (4.2, 2.8), 1995: (5.8, 2.6), 1996: (5.3, 2.2), 1997: (5.5, 1.8), 1998: (5.4, 1.5),
             1999: (5.0, 1.2), 2000: (6.2, 1.2), 2001: (3.9, 1.3), 2002: (1.7, 1.6), 2003: (1.1, 1.7), 2004: (1.4, 1.6),
             2005: (3.2, 1.8), 2006: (5.0, 1.8), 2007: (5.0, 1.8), 2008: (1.9, 2.5), 2009: (0.16, 2.4), 2010: (0.18, 1.9)}
GRID = {"lookback": (7, 10, 14, 20, 28), "mult": (0.8, 1.0, 1.2), "every": (15, 30, 60)}
PUBLISHED = (14, 1.0, 30)


def cut(s, a, b):
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


def sharpe(d: pd.Series) -> float:
    return float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 2 and d.std() > 0 else 0.0


def split_stats(t: pd.DataFrame, net: np.ndarray, d: pd.Series) -> dict:
    out = {}
    dates = pd.DatetimeIndex(t["date"]) if len(t) else pd.DatetimeIndex([])
    for k, (a, b) in SPLITS.items():
        m = (dates >= a) & (dates <= b)
        x = net[m]
        out[k] = {"n": int(m.sum()), "net_bp": float(x.mean() * 1e4) if len(x) else None,
                  "net_t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0,
                  "sharpe": sharpe(cut(d, a, b))}
    return out


def h43_h44(panels, days) -> dict:
    cells = []
    for lb, m, ev in itertools.product(*GRID.values()):
        t = pd.concat([noise(p, lb, m, ev) for p in panels.values()], ignore_index=True)
        net = net_returns(t, "mes")
        d = daily(t, net, days)
        cells.append({"lookback": lb, "mult": m, "every": ev, **split_stats(t, net, d)})
    pub = next(c for c in cells if (c["lookback"], c["mult"], c["every"]) == PUBLISHED)
    i_lb, i_m, i_ev = GRID["lookback"].index(14), GRID["mult"].index(1.0), GRID["every"].index(30)
    neigh = []
    for c in cells:
        j = (GRID["lookback"].index(c["lookback"]), GRID["mult"].index(c["mult"]), GRID["every"].index(c["every"]))
        if sum(abs(a - b) for a, b in zip(j, (i_lb, i_m, i_ev))) == 1:
            neigh.append(c)
    is_net = [c["IS"]["net_bp"] for c in cells]
    best = max(cells, key=lambda c: c["IS"]["sharpe"])
    h43 = {"cells": cells, "published": pub, "neighbours": neigh,
           "neighbour_median_is_bp": float(np.median([c["IS"]["net_bp"] for c in neigh])),
           "share_cells_positive_is": float(np.mean([x > 0 for x in is_net])),
           "share_cells_positive_all_three": float(np.mean([all(c[k]["net_bp"] > 0 for k in SPLITS) for c in cells])),
           "passed": bool(np.median([c["IS"]["net_bp"] for c in neigh]) > 0 and np.mean([x > 0 for x in is_net]) > 1 / len(cells))}
    h44 = {"best_on_is": {k: best[k] for k in ("lookback", "mult", "every")}, "best": best, "published": pub,
           "better_val": bool(best["VAL"]["sharpe"] > pub["VAL"]["sharpe"]),
           "better_test": bool(best["TEST"]["sharpe"] > pub["TEST"]["sharpe"])}
    h44["passed"] = bool(h44["better_val"] and h44["better_test"])
    return {"H43": h43, "H44": h44}


def h45(panels, days, spy) -> dict:
    t = pd.concat([overnight(p) for p in panels.values()], ignore_index=True)
    net = net_returns(t, "mes")
    d = daily(t, net, days)
    sig = spy["close"].pct_change().rolling(14).std().shift().reindex(days)
    size = np.minimum(2.0, 0.009 / sig).fillna(1.0)
    dv = d * size
    rows = {}
    for k, (a, b) in SPLITS.items():
        rows[k] = {"sharpe_1x": sharpe(cut(d, a, b)), "sharpe_vol_managed": sharpe(cut(dv, a, b))}
    rows["ALL"] = {"sharpe_1x": sharpe(d), "sharpe_vol_managed": sharpe(dv), "avg_size": float(size.mean())}
    rows["passed"] = bool(all(rows[k]["sharpe_vol_managed"] > rows[k]["sharpe_1x"] for k in SPLITS))
    return rows


def h46(spy) -> dict:
    o, c = spy["open"], spy["close"]
    r = (o / c.shift() - 1).dropna()
    nights = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
    rq = pd.Series([(lambda y: (RATES.get(y) or RATES_OLD.get(y) or (3.0, 1.5)))(y) for y in r.index.year], index=r.index)
    carry = np.array([(a - b) / 100 / 365 for a, b in rq]) * nights.values
    net = r - 0.0001 - carry
    out = {"by_decade": {}}
    for a, b in (("1993", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2024"), ("2025", "2026")):
        g, n = r[a:b], net[a:b]
        out["by_decade"][f"{a}-{b}"] = {"nights": int(len(g)), "gross_bp": float(g.mean() * 1e4),
                                        "gross_t": float(g.mean() / (g.std() / np.sqrt(len(g)))), "mes_net_bp": float(n.mean() * 1e4),
                                        "win": float((g > 0).mean())}
    x = out["by_decade"]["2025-2026"]
    out["oos_2025_2026"] = x
    out["passed"] = bool(x["mes_net_bp"] > 0 and all(v["gross_bp"] > 0 for k, v in out["by_decade"].items() if k != "2025-2026"))
    return out


def h47() -> dict:
    d = json.loads((OUT / "dip_study.json").read_text())
    s = next(x for x in d["streams"] if x["key"] == "1d|rsi2|stop2.5")
    tr = pd.DataFrame(s["trades"], columns=["date", "net", "mae", "days", "reason"])
    tr["date"] = pd.to_datetime(tr["date"])
    new = tr[tr["date"] >= "2025-01-01"]
    return {"n": int(len(new)), "win_rate": float((new["net"] > 0).mean()) if len(new) else None,
            "mean_net_pct": float(new["net"].mean() * 100) if len(new) else None,
            "worst_pct": float(new["net"].min() * 100) if len(new) else None,
            "trades": [[str(r.date.date()), round(r.net * 100, 2), r.reason] for r in new.itertuples()],
            "passed": bool(len(new) and new["net"].mean() > 0)}


def h48(panels, days, spy) -> dict:
    sig14 = spy["close"].pct_change().rolling(14).std().shift().reindex(days)
    lev_vt = np.minimum(4.0, 0.02 / sig14).fillna(1.0)
    legs = {}
    t = pd.concat([noise(p) for p in panels.values()], ignore_index=True)
    legs["noise"] = daily(t, net_returns(t, "mes"), days) * lev_vt
    t = pd.concat([overnight(p) for p in panels.values()], ignore_index=True)
    legs["overnight"] = daily(t, net_returns(t, "mes"), days)
    d = json.loads((OUT / "dip_study.json").read_text())
    s = next(x for x in d["streams"] if x["key"] == "1d|rsi2|stop2.5")
    rs = pd.Series(0.0, index=spy.index)
    for r in pd.DataFrame(s["trades"], columns=["date", "net", "mae", "days", "reason"]).itertuples():
        i = spy.index.searchsorted(pd.Timestamp(r.date))
        seg = spy.index[i:i + max(int(r.days), 1)]
        rs.loc[seg] += r.net / max(len(seg), 1)
    legs["rsi2"] = rs.reindex(days).fillna(0.0)
    L = pd.DataFrame(legs)
    is_ = cut(L, *SPLITS["IS"])
    w_rp = (1 / is_.std()) / (1 / is_.std()).sum()
    w_now = pd.Series({"noise": 1.0, "overnight": 1.0, "rsi2": 2.0})
    out = {"weights_risk_parity": w_rp.round(4).to_dict()}
    for name, w in (("current", w_now), ("risk_parity", w_rp)):
        book = (L * w).sum(axis=1)
        k = 0.20 / (cut(book, *SPLITS["IS"]).std() * np.sqrt(252))  # scale fixed on IS only
        book = book * k
        out[name] = {"scale_from_is": float(k), **{sp: {"sharpe": sharpe(cut(book, a, b)), "ann": float(cut(book, a, b).mean() * 252)}
                                                  for sp, (a, b) in SPLITS.items()}, "ALL": {"sharpe": sharpe(book)}}
    out["passed"] = bool(all(out["risk_parity"][k]["sharpe"] > out["current"][k]["sharpe"] for k in ("VAL", "TEST")))
    return out


def main():
    warnings.filterwarnings("ignore")
    spy, _ = spy_daily()
    panels = {s: panel(store.load(s, "1m")) for s in ("SPXCFD", "SPY")}
    days = pd.DatetimeIndex(sorted(set(panels["SPXCFD"].days) | set(panels["SPY"].days)))
    out = {"generated": str(pd.Timestamp.now(tz="UTC")), "registry": "research/REGISTRY.md"}
    out.update(h43_h44(panels, days))
    out["H45"] = h45(panels, days, spy)
    out["H46"] = h46(spy)
    out["H47"] = h47()
    out["H48"] = h48(panels, days, spy)
    (OUT / "review.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda v: None))
    h = out["H43"]
    print("H43 published", {k: (h["published"][k]["net_bp"], round(h["published"][k]["sharpe"], 2)) for k in SPLITS},
          "| neighbour median IS bp", round(h["neighbour_median_is_bp"], 2), "| cells + in IS", round(h["share_cells_positive_is"], 2),
          "| + in all three", round(h["share_cells_positive_all_three"], 2), "| pass", h["passed"])
    for c in h["neighbours"]:
        print("    neighbour", c["lookback"], c["mult"], c["every"], {k: round(c[k]["net_bp"] or 0, 2) for k in SPLITS})
    b = out["H44"]
    print("H44 best on IS", b["best_on_is"], {k: round(b["best"][k]["sharpe"], 2) for k in SPLITS}, "vs published",
          {k: round(b["published"][k]["sharpe"], 2) for k in SPLITS}, "pass", b["passed"])
    print("H45", {k: v for k, v in out["H45"].items()})
    print("H46", {k: (v["nights"], round(v["gross_bp"], 2), round(v["gross_t"], 2), round(v["mes_net_bp"], 2)) for k, v in out["H46"]["by_decade"].items()}, "pass", out["H46"]["passed"])
    print("H47", {k: v for k, v in out["H47"].items() if k != "trades"})
    print("H48", json.dumps(out["H48"]))
    print("wrote", OUT / "review.json")


if __name__ == "__main__":
    main()
