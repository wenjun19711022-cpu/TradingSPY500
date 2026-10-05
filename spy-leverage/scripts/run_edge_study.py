"""Which published intraday long strategies survive real costs, and how much leverage do they
carry? -> results/edge_study.json

    python -m spylev.data.histdata        # S&P 500 minute bars 2011-2018 (once)
    python -m spylev.data.minute_hist     # SPY minute bars 2022-10..2024-12 (once)
    python scripts/run_edge_study.py

Pre-registered before looking at any result: the five published rules in spylev.edge.strategies
with their published parameters, plus two combinations fixed in advance:
  noise_trend      noise-area long only when SPY closed above its 200-day average the day before
  noise_overnight  noise-area long, and if still long at the close, hold to the next open
Splits: IS 2011-2015, VAL 2016-2018 (S&P 500 index minute bars, no volume -> equal-weight VWAP),
TEST 2022-10..2024-12 (SPY with volume). Cost regimes in spylev.edge.evaluate.
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
from spylev.edge.evaluate import LABELS, REGIMES, daily, kelly, leverage_table, net_returns, trade_stats  # noqa: E402
from spylev.edge.strategies import N, NAMES, STRATEGIES, Panel, noise, panel  # noqa: E402

OUT = ROOT / "results"
SPLITS = {"IS": ("2011-01-01", "2015-12-31"), "VAL": ("2016-01-01", "2018-12-31"), "TEST": ("2022-10-01", "2024-12-31"),
          "HOLDOUT": ("2025-01-01", "2100-01-01")}  # empty until your moomoo minute bars are in the store
NAMES = {**NAMES, "noise_trend": "噪声区突破，只在日线 200 日均线上方做（预先定的组合）",
         "noise_overnight": "噪声区突破，收盘仍持有就拿到次日开盘（预先定的组合）"}


def noise_trend(p: Panel, up: pd.Series) -> pd.DataFrame:
    t = noise(p)
    if t.empty:
        return t
    ok = up.reindex(pd.DatetimeIndex(t["date"])).fillna(False).values
    return t[ok].reset_index(drop=True)


def noise_overnight(p: Panel) -> pd.DataFrame:
    t = noise(p)
    if t.empty:
        return t
    rows = []
    idx = {d: i for i, d in enumerate(p.days)}
    for r in t.to_dict("records"):
        d = idx[r["date"]]
        if r["how_out"] == "close" and d + 1 < len(p.days) and p.consecutive[d + 1]:
            nxt = p.O[d + 1, 0]
            r = {**r, "exit": float(nxt), "gross": float(nxt / r["entry"] - 1), "how_out": "next_open",
                 "mae": float(min(r["mae"], nxt / r["entry"] - 1)), "nights": float(p.gap_days[d + 1])}
        rows.append(r)
    return pd.DataFrame(rows)


def _summ(d: pd.Series) -> dict:
    ann, vol = d.mean() * 252, d.std() * np.sqrt(252)
    eq = (1 + d).clip(lower=0).cumprod()
    out = {"ann_return": float(ann), "ann_vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else 0.0,
           "max_dd": float((eq / eq.cummax() - 1).min()), "worst_day": float(d.min())}
    out["by_split"] = {}
    for k, (a, b) in SPLITS.items():
        x = d[(d.index >= a) & (d.index <= b)]
        if len(x) < 2:
            continue
        out["by_split"][k] = {"ann_return": float(x.mean() * 252), "sharpe": float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0}
    out["by_year"] = {str(y): float((1 + x).prod() - 1) for y, x in d.groupby(d.index.year)}
    return out


def portfolio(panels: dict, days: pd.DatetimeIndex, spy: pd.DataFrame, rsi2: dict | None) -> dict:
    """The playbook on futures costs: noise-area long sized like the paper (2% daily vol target,
    at most 4x), the overnight hold at 1x, and the daily RSI2 dip at 2x. Then the whole book
    scaled to 20% annual volatility."""
    sig14 = spy["close"].pct_change().rolling(14).std().shift().reindex(days)
    lev_vt = np.minimum(4.0, 0.02 / sig14).fillna(1.0)
    legs = {}
    for key, fn in (("noise", noise), ("overnight", STRATEGIES["overnight"])):
        t = pd.concat([fn(p) for p in panels.values()], ignore_index=True)
        legs[key] = daily(t, net_returns(t, "mes"), days)
    legs["noise"] = legs["noise"] * lev_vt
    rs = pd.Series(0.0, index=spy.index)
    if rsi2:
        tr = pd.DataFrame(rsi2["trades"], columns=["date", "net", "mae", "days", "reason"])
        for r in tr.itertuples():
            i = spy.index.searchsorted(pd.Timestamp(r.date))
            seg = spy.index[i:i + max(int(r.days), 1)]
            rs.loc[seg] += r.net / max(len(seg), 1)
    legs["rsi2"] = 2 * rs.reindex(days).fillna(0.0)
    book = legs["noise"] + legs["overnight"] + legs["rsi2"]
    scale = float(0.20 / (book.std() * np.sqrt(252)))
    bh = spy["close"].pct_change().reindex(days).fillna(0.0)
    return {"legs": {k: _summ(v) for k, v in legs.items()}, "book_1x": _summ(book), "scale_to_20pct_vol": scale,
            "book_scaled": _summ(book * scale), "buy_and_hold": _summ(bh),
            "avg_leverage": {"noise": float(lev_vt.mean() * scale), "overnight": scale, "rsi2": 2 * scale},
            "corr": pd.DataFrame(legs).corr().round(3).to_dict()}


def split_of(dates: pd.Series) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    out = np.full(len(d), "", dtype=object)
    for k, (a, b) in SPLITS.items():
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = k
    return out


def main():
    warnings.filterwarnings("ignore")
    spy, _ = spy_daily()
    up = (spy["close"] > spy["close"].rolling(200).mean()).shift()  # known before the session
    panels = {}
    for sym in ("SPXCFD", "SPY"):
        b = store.load(sym, "1m")
        if b.empty:
            raise SystemExit(f"no {sym} minute bars: run python -m spylev.data.histdata / spylev.data.minute_hist")
        panels[sym] = panel(b)
    all_days = pd.DatetimeIndex(sorted(set(panels["SPXCFD"].days) | set(panels["SPY"].days)))
    out = {"generated": str(pd.Timestamp.now(tz="UTC")), "splits": SPLITS, "regimes": LABELS,
           "data": {k: {"start": str(p.days[0].date()), "end": str(p.days[-1].date()), "sessions": len(p.days)} for k, p in panels.items()},
           "strategies": {}}
    for key in list(STRATEGIES) + ["noise_trend", "noise_overnight"]:
        parts = []
        for sym, p in panels.items():
            t = STRATEGIES[key](p) if key in STRATEGIES else noise_trend(p, up) if key == "noise_trend" else noise_overnight(p)
            if not t.empty:
                parts.append(t.assign(data=sym))
        t = pd.concat(parts, ignore_index=True)
        t["split"] = split_of(t["date"])
        t = t[t["split"] != ""].reset_index(drop=True)
        rec = {"name": NAMES[key], "regimes": {}}
        for rg in REGIMES:
            net = net_returns(t, rg)
            r = {k: trade_stats(t[t["split"] == k], net[t["split"].values == k]) for k in SPLITS}
            r["ALL"] = trade_stats(t, net)
            g = r["IS"]
            r["G1"] = bool(g.get("n", 0) >= 100 and g["net_bp"] > 0 and g["net_t"] >= 2)
            r["G2"] = bool(r["G1"] and r["VAL"].get("net_bp", -1) > 0)
            r["G3"] = bool(r["G2"] and r["TEST"].get("net_bp", -1) > 0)
            d = daily(t, net, all_days)
            r["kelly"] = kelly(d)
            r["leverage"] = leverage_table(t, net, all_days)
            per_split_sharpe = {}
            for k, (a, b) in SPLITS.items():
                dd = d[(d.index >= a) & (d.index <= b)]
                if len(dd) > 1:
                    per_split_sharpe[k] = float(dd.mean() / dd.std() * np.sqrt(252)) if dd.std() > 0 else 0.0
            r["sharpe_by_split"] = per_split_sharpe
            rec["regimes"][rg] = r
        rec["avg_hold_min"] = float(np.where(t["t_in"] >= 0, t["t_out"] - t["t_in"], np.nan).mean()) if len(t) else None
        rec["trades_per_year"] = float(len(t) / (len(all_days) / 252))
        out["strategies"][key] = rec
        m = rec["regimes"]["mes"]
        print(f"{key:<16} n={m['ALL']['n']:>5} gross={m['ALL']['gross_bp']:+.2f}bp (t={m['ALL']['gross_t']:+.2f}) | "
              + " | ".join(f"{rg}: net {rec['regimes'][rg]['ALL']['net_bp']:+.2f} G={int(rec['regimes'][rg]['G1'])}{int(rec['regimes'][rg]['G2'])}{int(rec['regimes'][rg]['G3'])}"
                           for rg in REGIMES)
              + f" | kelly(mes)={m['kelly']['kelly']:.1f}x sharpe={m['kelly']['sharpe_1x']:.2f}")
    # the validated daily rule from the dip study, for the playbook
    dip = json.loads((OUT / "dip_study.json").read_text(encoding="utf-8"))
    s = next((x for x in dip.get("streams", []) if x["key"] == "1d|rsi2|stop2.5"), None)
    if s:
        out["daily_rsi2"] = {"ALL": s["ALL"], "IS": s["IS"], "VAL": s["VAL"], "TEST": s["TEST"], "leverage": s["leverage"], "oos": s["oos"]}
    out["portfolio"] = portfolio(panels, all_days, spy, s)
    (OUT / "edge_study.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda v: None), encoding="utf-8")
    print("wrote", OUT / "edge_study.json")


if __name__ == "__main__":
    main()
