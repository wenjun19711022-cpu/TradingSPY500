"""Multi-timeframe leveraged dip-buying study -> results/dip_study.json.

    python -m spyq.data.history          # once: long daily history (1993-) + VIX (1990-)
    python scripts/run_dip_study.py

Timeframes: 1d / 1w / 1M from SPY 1993-2026 daily OHLC; IBKR 4h (2024-04..) and 30m (last
~77 days) samples; and every intraday timeframe in the local minute store (3m ... 4h) when
`python -m spyq.data.crawl` has been run. Pre-registered grid: 5 bottom families x {no stop,
2.5-ATR stop}. Gates: IS n>=30, mean net > 0, t >= 2, PF >= 1.2; then VAL and TEST mean net > 0.
Leverage per stream is set from IS trades only (spyq.dip.engine.leverage_table) and then
applied unchanged to VAL and TEST.
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

from spyq.data import store  # noqa: E402
from spyq.data.history import spx_daily, spx_monthly, spy_daily, vix_daily  # noqa: E402
from spyq.data.resample import resample  # noqa: E402
from spyq.data.sources import load_ibkr_json  # noqa: E402
from spyq.dip.engine import DipSpec, leverage_table, levered_equity, run, stats  # noqa: E402
from spyq.dip.perp import PerpCosts  # noqa: E402
from spyq.dip.signals import FAMILIES, features, signal  # noqa: E402

OUT = ROOT / "results"
GATE = dict(n_min=30, t_min=2.0, pf_min=1.2)
DAILY_SPLITS = {"IS": ("1900", "2012-12-31"), "VAL": ("2013-01-01", "2019-12-31"), "TEST": ("2020-01-01", "2100")}
INTRA_SPLITS = {"IS": ("1900", "2022-12-31"), "VAL": ("2023-01-01", "2024-12-31"), "TEST": ("2025-01-01", "2100")}


def frac_splits(index: pd.DatetimeIndex) -> dict:
    a, b = index[int(len(index) * 0.6)], index[int(len(index) * 0.8)]
    return {"IS": ("1900", str(a)), "VAL": (str(a), str(b)), "TEST": (str(b), "2100")}


def cut(t: pd.DataFrame, rng) -> pd.DataFrame:
    if t is None or t.empty:
        return t
    idx = t.index.tz_localize(None) if getattr(t.index, "tz", None) is not None else t.index
    m = (idx >= pd.Timestamp(rng[0])) & (idx <= pd.Timestamp(rng[1]) + pd.Timedelta(hours=23, minutes=59))
    return t[m]


def datasets():
    spy, splice = spy_daily()
    vix = vix_daily()["close"]
    wk = spy.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    mo = spy.resample("ME").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    sets = {
        "1d": (spy.assign(volume=0.0), DAILY_SPLITS, "SPY 1993–2026 日线"),
        "1w": (wk.assign(volume=0.0), DAILY_SPLITS, "SPY 周线（由日线合成）"),
        "1M": (mo.assign(volume=0.0), DAILY_SPLITS, "SPY 月线（由日线合成）"),
    }
    full = store.load("SPY", "1m")
    if len(full) > 100_000:
        for tf in ("3m", "5m", "15m", "30m", "1h", "4h"):
            sets[tf] = (resample(full, tf).drop(columns=["n_bars"]), INTRA_SPLITS, f"本地 1 分钟库重采样 {tf}")
    else:
        raw = ROOT / "data" / "raw" / "ibkr"
        h4 = load_ibkr_json(raw / "SPY_4h_recent.json")
        h4.index = h4.index.tz_localize(None)
        m30 = load_ibkr_json(raw / "SPY_30m_recent.json")
        m30.index = m30.index.tz_localize(None)
        sets["4h"] = (h4, frac_splits(h4.index), "IBKR 4h 样本（2024-04 起，时钟对齐）")
        sets["30m"] = (m30, frac_splits(m30.index), "IBKR 30m 样本（约 77 天，仅冒烟测试）")
    return sets, vix, splice


def crash_section(costs: PerpCosts) -> dict:
    """Further fall after each drawdown tier, on 155 years of monthly S&P and 1980- daily S&P."""
    m = spx_monthly()
    dd = m / m.cummax() - 1
    rows = []
    for tier in (0.10, 0.20, 0.30, 0.40):
        armed, ev = True, []
        for t, d in dd.items():
            if d >= 0:
                armed = True
            elif armed and d <= -tier:
                armed = False
                px = m[t]
                fut = m[m.index >= t]
                back = fut[fut >= m[:t].max()]
                end = back.index[0] if len(back) else fut.index[-1]
                worst = fut[:end].min() / px - 1
                ev.append({"date": str(t.date()), "further": float(worst), "years_to_new_high": (end - t).days / 365.25 if len(back) else None})
        worst_ev = min(ev, key=lambda x: x["further"])
        rows.append({"tier": tier, "n": len(ev), "median_further": float(np.median([e["further"] for e in ev])),
                     "worst_further": worst_ev["further"], "worst_date": worst_ev["date"],
                     "max_years": max((e["years_to_new_high"] or 0) for e in ev),
                     "max_safe_lev": 1 / (1.5 * -worst_ev["further"] + costs.mmr) if worst_ev["further"] < 0 else None})
    s = spx_daily()
    r = s.pct_change().dropna()
    gaps = {f"{L}x": float((r <= -costs.liq_drop(L)).sum()) for L in (3, 5, 10, 20)}
    return {"spx_monthly_tiers": rows, "spx_worst_day": float(r.min()), "spx_worst_day_date": str(r.idxmin().date()),
            "days_that_liquidate_without_stop": gaps}


def overlay_sim(spy: pd.DataFrame, trades: pd.DataFrame, lev: float, costs: PerpCosts, base: float = 1.0) -> pd.Series:
    """Daily equity of `base` x SPY held as shares (no funding) plus a perp overlay of `lev` x
    notional during each dip trade (entries/exits at the open, fees on both legs, funding per
    calendar day). Price-only: SPY dividends (~1.3%/yr) are ignored for the share leg."""
    o, c = spy["open"], spy["close"]
    r = c.pct_change().fillna(0.0)
    ov = pd.Series(0.0, index=spy.index)
    if trades is not None and not trades.empty:
        days = spy.index
        for _, t in trades.iterrows():
            e, x = pd.Timestamp(t.name), pd.Timestamp(t["exit_time"])
            if e not in days.values or x not in days.values:
                continue
            i, j = days.get_loc(e), days.get_loc(x)
            if j <= i:
                ov.iloc[i] += lev * (c.iloc[i] / o.iloc[i] - 1)
                continue
            ov.iloc[i] += lev * (c.iloc[i] / o.iloc[i] - 1) - lev * costs.round_trip() / 2
            for k in range(i + 1, j):
                gap_days = (days[k] - days[k - 1]).days
                ov.iloc[k] += lev * (c.iloc[k] / c.iloc[k - 1] - 1) - lev * costs.funding(gap_days)
            gap_days = (days[j] - days[j - 1]).days
            ov.iloc[j] += lev * (o.iloc[j] / c.iloc[j - 1] - 1) - lev * costs.round_trip() / 2 - lev * costs.funding(gap_days)
    eq = (1 + base * r + ov).cumprod()
    return eq


def summarize(eq: pd.Series) -> dict:
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dd = eq / eq.cummax() - 1
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0, "max_dd": float(dd.min()),
            "final": float(eq.iloc[-1])}


def main():
    warnings.filterwarnings("ignore")
    costs = PerpCosts()
    sets, vix, splice = datasets()
    out = {"generated": str(pd.Timestamp.now(tz="UTC")), "costs": costs.__dict__, "splice": splice, "gate": GATE,
           "splits_daily": DAILY_SPLITS, "streams": [], "datasets": {}}
    for tf, (df, splits, label) in sets.items():
        out["datasets"][tf] = {"label": label, "start": str(df.index[0])[:16], "end": str(df.index[-1])[:16], "bars": len(df),
                               "splits": splits}
        f = features(df, tf, vix if tf in ("1d", "1w", "1M") else None)
        for fam in FAMILIES:
            if fam == "panic" and tf not in ("1d", "1w", "1M"):
                continue
            if fam == "ath_dd" and tf not in ("1d", "1w", "1M"):
                continue
            for stop in ((None,) if fam == "ath_dd" else (None, 2.5)):
                spec = DipSpec(tf, fam, stop)
                t = run(df, spec, costs, feats=f)
                rec = {"key": spec.key, **spec.as_dict(), "label": label}
                parts = {k: cut(t, v) for k, v in splits.items()}
                for k, tt in parts.items():
                    rec[k] = stats(tt)
                rec["ALL"] = stats(t)
                is_ = rec["IS"]
                g1 = is_.get("n", 0) >= GATE["n_min"] and is_["mean_net"] > 0 and is_["t_stat"] >= GATE["t_min"] and is_["profit_factor"] >= GATE["pf_min"]
                g2 = g1 and rec["VAL"].get("n", 0) > 0 and rec["VAL"]["mean_net"] > 0
                g3 = g2 and rec["TEST"].get("n", 0) > 0 and rec["TEST"]["mean_net"] > 0
                rec.update({"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)})
                lev = leverage_table(parts["IS"], costs) if parts["IS"] is not None and len(parts["IS"]) >= 5 else {"L": 0.0}
                rec["leverage"] = lev
                L = lev.get("L", 0.0) if g1 else 0.0
                rec["oos"] = {}
                for k in ("VAL", "TEST"):
                    tt = parts[k]
                    for lab, LL in (("rec", L), ("x1", 1.0), ("x3", 3.0), ("x10", 10.0)):
                        e = levered_equity(tt, LL, costs) if tt is not None and not tt.empty else {"final": 1.0, "max_dd": 0.0, "liquidations": 0}
                        rec["oos"][f"{k}_{lab}"] = {"L": LL, "final": e["final"], "max_dd": e["max_dd"], "liquidations": e["liquidations"]}
                if len(t):
                    worst = t.nsmallest(1, "mae").iloc[0]
                    rec["worst_trade"] = {"entry": str(t["mae"].idxmin())[:16], "mae": float(worst["mae"]), "net": float(worst["net"]),
                                          "days": float(worst["days"])}
                    # sensitivity on all trades: maker fills / no funding
                    rec["sens"] = {
                        "maker_rt": float((t["gross"] - PerpCosts().round_trip(True, True) - t["funding"]).mean()),
                        "no_funding": float((t["gross"] - t["fees"]).mean()),
                        "gross": float(t["gross"].mean()),
                    }
                    if g1:
                        rec["trades"] = [[str(i)[:16], round(r["net"], 5), round(r["mae"], 5), round(r["days"], 2), r["reason"]]
                                         for i, r in t.iterrows()][-400:]
                out["streams"].append(rec)
                print(f"{spec.key:<22} n={rec['ALL'].get('n', 0):>4} IS t={is_.get('t_stat', 0):+.2f} mean={is_.get('mean_net', 0):+.4%} "
                      f"VAL={rec['VAL'].get('mean_net', np.nan):+.4%} TEST={rec['TEST'].get('mean_net', np.nan):+.4%} "
                      f"G={int(g1)}{int(g2)}{int(g3)} L={lev.get('L', 0):.2f}({lev.get('binding', '-')})")
    out["crash"] = crash_section(costs)
    # funding sensitivity for the streams whose holding period makes funding matter
    spyd = sets["1d"][0]
    fs = {}
    for key in ("1d|rsi2|stop2.5", "1d|rsi2|stop-", "1w|rsi2|stop-", "1d|ath_dd|stop-", "1w|ath_dd|stop-", "1M|ath_dd|stop-"):
        tf, fam, st = key.split("|")
        stop = None if st == "stop-" else float(st[4:])
        df_ = sets[tf][0]
        row = {}
        for f8 in (0.0, 0.00005, 0.0001, 0.0003):
            c2 = PerpCosts(funding_8h=f8)
            t2 = run(df_, DipSpec(tf, fam, stop), c2, vix=vix)
            row[f"{f8 * 100:.3f}%"] = stats(t2).get("mean_net")
        fs[key] = row
    out["funding_sensitivity"] = fs
    # shares as the core position, perp leverage only during daily RSI2 dips (exploratory)
    rsi_t = run(spyd, DipSpec("1d", "rsi2", 2.5), costs, vix=vix)
    combos = {}
    for lab, base, lev in (("SPY 现货 1x（基准）", 1.0, 0.0), ("现货 1x + 抄底时永续 +1x", 1.0, 1.0),
                           ("现货 1x + 抄底时永续 +2x", 1.0, 2.0), ("现货 1x + 抄底时永续 +3x", 1.0, 3.0),
                           ("只做抄底 永续 3x", 0.0, 3.0), ("永续 1x 长期持有", 0.0, 0.0)):
        if lab == "永续 1x 长期持有":
            eq = (1 + spyd["close"].pct_change().fillna(0) - costs.funding(1.0) * spyd.index.to_series().diff().dt.days.fillna(1)).cumprod()
        else:
            eq = overlay_sim(spyd, rsi_t, lev, costs, base)
        row = {"all": summarize(eq)}
        for k, (a, b) in DAILY_SPLITS.items():
            seg = eq[(eq.index >= a) & (eq.index <= b)]
            row[k] = summarize(seg / seg.iloc[0])
        row["curve"] = [[str(d.date()), round(float(v), 4)] for d, v in eq.iloc[::5].items()]
        combos[lab] = row
    out["combos"] = combos
    # funding drag: what holding 1x SPY perp for a year costs vs SPY's average price return
    spy = sets["1d"][0]["close"]
    yrs = (spy.index[-1] - spy.index[0]).days / 365.25
    out["funding_vs_drift"] = {"spy_price_cagr": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / yrs) - 1),
                               "funding_per_year_1x": costs.funding(365.0)}
    (OUT / "dip_study.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda x: None if x is None else float(x)))
    print("wrote", OUT / "dip_study.json")


if __name__ == "__main__":
    main()
