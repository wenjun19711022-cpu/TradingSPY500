"""Run the full study and write results/study.json (+ CSVs) for the dashboard.

    python scripts/run_study.py            # IBKR 5y daily + VIX family bundled in data/raw
    python scripts/run_study.py --fast     # skip the 1,820-config factory (reuse the CSV)

Uses whatever is in data/: CBOE CSVs (data/raw/cboe) override the IBKR dumps, and a full
minute store (data/bars/SPY/1m) enables the minute-engine section and the 5-year K-lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spyopt.backtest.daily import Spec, market_rows, simulate  # noqa: E402
from spyopt.backtest.metrics import trade_stats, yearly  # noqa: E402
from spyopt.backtest.validation import _cut, best_by_family, run_factory, sensitivity  # noqa: E402
from spyopt.data import store  # noqa: E402
from spyopt.data.resample import resample  # noqa: E402
from spyopt.data.sources import load_ibkr_json  # noqa: E402
from spyopt.data.vix import load_panel  # noqa: E402
from spyopt.indicators import vol  # noqa: E402
from spyopt.indicators.regime import features  # noqa: E402
from spyopt.options.costs import CostModel  # noqa: E402
from spyopt.options.smile import Smile, fit  # noqa: E402
from spyopt.risk.sizing import kelly_fraction, ladder_equity, summarize_equity  # noqa: E402
from spyopt.risk.stress import run_stress  # noqa: E402
from spyopt.strategies.composite import AGGRESSIVE, BUDGETS, CORE, delta_of_strikes, expected_move_bands, plan_rung  # noqa: E402
from spyopt.strategies.factory import GATES, SPLITS, overlap  # noqa: E402

OUT = ROOT / "results"


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return _f(obj)


# ----------------------------------------------------------------------------- sections

def vrp_section(p: pd.DataFrame) -> dict:
    r_oc = np.log(p["close"] / p["open"])
    r_cc = np.log(p["close"] / p["close"].shift())
    rows = []
    for lab, iv, ret in [
        ("VIX1D 开盘 vs 当日开盘→收盘", p["vix1d_open"], r_oc),
        ("VIX1D 前收 vs 收盘→收盘", p["vix1d_close"].shift(), r_cc),
        ("VIX9D 前收 vs 收盘→收盘", p["vix9d_close"].shift(), r_cc),
        ("VIX 前收 vs 收盘→收盘", p["vix_close"].shift(), r_cc),
    ]:
        m = iv.notna() & ret.notna()
        impl = (iv[m] / 100) ** 2 / 252
        real = ret[m] ** 2
        z = ret[m].abs() / np.sqrt(impl)
        rows.append({"pair": lab, "n": int(m.sum()), "implied": float(np.sqrt(impl.mean() * 252) * 100),
                     "realized": float(np.sqrt(real.mean() * 252) * 100), "var_ratio": float(real.mean() / impl.mean()),
                     "mean_abs_z": float(z.mean()), "p_gt1": float((z > 1).mean()), "p_gt2": float((z > 2).mean())})
    by_year = []
    for y, g in p.groupby(p.index.year):
        rcc = np.log(g["close"] / g["close"].shift()).dropna()
        by_year.append({"year": int(y), "rv": float(np.sqrt((rcc**2).mean() * 252) * 100),
                        "vix9d": float(g["vix9d_close"].mean()), "vix": float(g["vix_close"].mean()),
                        "vix1d": float(g["vix1d_close"].mean()) if g["vix1d_close"].notna().any() else None})
    return {"pairs": rows, "by_year": by_year}


def zero_dte_section(p, feats, smile, costs) -> dict:
    rows = market_rows(p, feats, "0dte_open", smile)
    free = CostModel(commission=0, min_spread=0, spread_coef=0, max_spread=0)
    out = []
    for st, k, w in [("iron_condor", 1.0, 0.5), ("iron_condor", 1.5, 1.0), ("put_spread", 1.0, 0.5),
                     ("call_spread", 1.0, 0.5), ("iron_fly", 0.0, 1.0), ("long_straddle", 0.0, 1.0)]:
        spec = Spec("0dte_open", st, k, k, w, "hold", "none")
        net = simulate(rows, spec, smile, costs)
        gross = simulate(rows, spec, smile, free)
        be = None
        for mult in np.arange(0.90, 1.31, 0.01):
            sm = Smile(shapes=smile.shapes, level={kk: v * mult for kk, v in smile.level.items()})
            t = simulate(market_rows(p, feats, "0dte_open", sm), spec, sm, costs)
            if t["ror"].mean() > 0:
                be = float(mult)
                break
        out.append({"structure": st, "k": k, "wing": w, "n": len(net), "win_rate": float((net["pnl"] > 0).mean()),
                    "gross_ror": float(gross["ror"].mean()), "net_ror": float(net["ror"].mean()),
                    "cost_ror": float(gross["ror"].mean() - net["ror"].mean()),
                    "avg_credit": float(net["credit"].mean() * 100), "breakeven_iv_mult": be})
    return {"structures": out}


def core_section(p, feats, smile, costs) -> dict:
    res = {}
    for name, spec in (("standard", CORE), ("aggressive", AGGRESSIVE)):
        rows = market_rows(p, feats, spec.mode, smile)
        tr = simulate(rows, spec, smile, costs, panel=p)
        tr["entry"] = rows.loc[tr.index, "entry"].values
        N = overlap(spec.mode)
        stats = {sp: trade_stats(_cut(tr, sp), overlap=N) for sp in ("IS", "VAL", "TEST")}
        stats["ALL"] = trade_stats(tr, overlap=N)
        eqs = {}
        for lab, b in BUDGETS.items():
            eq = ladder_equity(tr, b, N)
            eqs[lab] = {"summary": summarize_equity(eq),
                        "curve": [[str(d.date()), round(float(v), 5)] for d, v in eq["equity"].items()]}
        res[name] = {
            "spec": spec.as_dict(), "key": spec.key, "stats": stats,
            "yearly": yearly(tr).reset_index().rename(columns={"index": "year", "ts": "year"}).to_dict("records"),
            "exits": tr["exit"].value_counts().to_dict(),
            "kelly_per_rung": kelly_fraction(tr["ror"]),
            "equity": eqs,
            "sensitivity": sensitivity(p, spec, smile, costs).reset_index().rename(columns={"index": "case"}).to_dict("records"),
            "stress": {lab: run_stress(spec, b, smile).reset_index().rename(columns={"index": "scenario"}).to_dict("records")
                       for lab, b in BUDGETS.items()},
            "losers": tr[tr["pnl"] < 0].assign(date=lambda x: x.index.strftime("%Y-%m-%d"))[
                ["date", "entry", "F", "credit", "pnl", "ror", "exit", "k0", "k1", "k2", "k3"]].astype({"entry": str}).to_dict("records"),
        }
    return res


def plan_section(p, smile) -> dict:
    r = plan_rung(p, CORE, smile)
    return {"rung": asdict(r), "deltas": delta_of_strikes(r, smile), "bands": expected_move_bands(p, smile),
            "regime": regime_now(p)}


def regime_now(p) -> dict:
    f = features(p)
    last, lf = p.iloc[-1], f.iloc[-1]
    from spyopt.backtest.daily import alarm_series
    return {"date": str(p.index[-1].date()), "close": float(last["close"]), "vix1d": float(last["vix1d_close"]),
            "vix9d": float(last["vix9d_close"]), "vix": float(last["vix_close"]), "vix3m": float(last["vix3m_close"]),
            "rv_forecast_cc": float(lf["rv_cc_ann"]), "rv_forecast_session": float(lf["rv_session_ann"]),
            "vrp_9d": float(lf["vrp_9d"]), "ts_9d_30d": float(lf["ts_9d_30d"]), "ts_30d_3m": float(lf["ts_30d_3m"]),
            "event_next": float(lf["event_next"]), "trend": int(lf["trend"]), "vix_pct_1y": float(lf["vix_pct"]),
            "alarm": bool(alarm_series(p).iloc[-1])}


# ----------------------------------------------------------------------------- bars for the K-line viewer

def _bars_json(df: pd.DataFrame, daily=False):
    out = []
    for t, r in df.iterrows():
        ts = int(pd.Timestamp(t).timestamp()) if not daily else str(pd.Timestamp(t).date())
        if not daily:
            # lightweight-charts has no time zones: shift so ET wall-clock shows on the axis
            ts = int((pd.Timestamp(t).tz_convert("America/New_York").tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds())
        out.append([ts, round(r["open"], 2), round(r["high"], 2), round(r["low"], 2), round(r["close"], 2), int(r.get("volume", 0) or 0)])
    return out


def bars_section(p) -> dict:
    raw = ROOT / "data" / "raw" / "ibkr"
    full = store.load("SPY", "1m")
    src = {}
    if len(full) > 50_000:  # local full history: everything comes from the 1m store
        for tf in ("1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h"):
            df = full if tf == "1m" else resample(full, tf)
            src[tf] = df.tail(1500 if tf in ("1m", "3m") else 5000)
        src["1d"] = resample(full, "1d")
        note = f"本地 1 分钟库：{full.index[0].date()} → {full.index[-1].date()}"
    else:
        m1 = load_ibkr_json(raw / "SPY_1m_recent.json")
        m5 = load_ibkr_json(raw / "SPY_5m_recent.json")
        m30 = load_ibkr_json(raw / "SPY_30m_recent.json")
        h4n = load_ibkr_json(raw / "SPY_4h_recent.json")
        src = {"1m": m1, "3m": resample(m1, "3m"), "5m": m5, "10m": resample(m5, "10m"), "15m": resample(m5, "15m"),
               "30m": m30, "1h": resample(m30, "1h"), "2h": resample(m30, "2h"), "4h": resample(m30, "4h"),
               "4h_ibkr": h4n}
        note = "云端样本：IBKR 最近 1000 根（1m≈2.5天, 5m≈13天, 30m≈77天）；完整 5 年请在本地运行 spyopt.data.crawl"
    out = {tf: _bars_json(df) for tf, df in src.items()}
    out["1d"] = _bars_json(p[["open", "high", "low", "close", "volume"]], daily=True)
    wk = p[["open", "high", "low", "close", "volume"]].resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    out["1w"] = _bars_json(wk, daily=True)
    # next-day expected-move channel for the daily chart (from the prior close's VIX1D, else VIX9D)
    sm = Smile()
    atm = sm.atm_term(p["vix1d_close"].values, p["vix9d_close"].values, p["vix_close"].values, 1)
    em = p["close"].values * atm * np.sqrt(1 / 252)
    nxt = list(p.index[1:]) + [p.index[-1] + pd.offsets.BDay(1)]
    out["em_1d"] = [[str(pd.Timestamp(d).date()), round(c - e, 2), round(c + e, 2)] for d, c, e in zip(nxt, p["close"].values, em)]
    return {"note": note, "series": out}


def tf_signals(p) -> list:
    """Latest-bar indicator stack per timeframe (trend / momentum / position)."""
    rows = []
    raw = ROOT / "data" / "raw" / "ibkr"
    m1 = load_ibkr_json(raw / "SPY_1m_recent.json")
    m5 = load_ibkr_json(raw / "SPY_5m_recent.json")
    m30 = load_ibkr_json(raw / "SPY_30m_recent.json")
    frames = {"1m": m1, "3m": resample(m1, "3m"), "5m": m5, "10m": resample(m5, "10m"), "15m": resample(m5, "15m"),
              "30m": m30, "1h": resample(m30, "1h"), "2h": resample(m30, "2h"), "4h": resample(m30, "4h"),
              "1d": p, "1w": p.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()}
    for tf, df in frames.items():
        c = df["close"]
        if len(c) < 30:
            continue
        e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
        d = c.diff()
        rsi = 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean())
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14).mean()
        hi20, lo20 = df["high"].rolling(20).max(), df["low"].rolling(20).min()
        trend = 1 if (c.iloc[-1] > e20.iloc[-1] > e50.iloc[-1]) else -1 if (c.iloc[-1] < e20.iloc[-1] < e50.iloc[-1]) else 0
        rows.append({"tf": tf, "close": float(c.iloc[-1]), "trend": trend, "ema20": float(e20.iloc[-1]), "ema50": float(e50.iloc[-1]),
                     "rsi14": float(rsi.iloc[-1]), "boll_pos": float((c.iloc[-1] - mid.iloc[-1]) / (2 * sd.iloc[-1])),
                     "atr_pct": float(atr.iloc[-1] / c.iloc[-1] * 100), "donchian_pos": float((c.iloc[-1] - lo20.iloc[-1]) / (hi20.iloc[-1] - lo20.iloc[-1]))})
    return rows


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reuse results/factory_all.csv")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    p = load_panel()
    feats = features(p)
    smile, info = fit()
    costs = CostModel()
    if a.fast and (OUT / "factory_all.csv").exists():
        fac = pd.read_csv(OUT / "factory_all.csv", index_col=0)
    else:
        fac, _ = run_factory(p, smile, costs)
        fac.to_csv(OUT / "factory_all.csv")
    best = best_by_family(fac)
    keep = ["mode", "structure", "k_put", "wing", "exit", "filt", "IS_n", "IS_win_rate", "IS_mean_ror", "IS_t_stat",
            "IS_profit_factor", "IS_dsr", "VAL_n", "VAL_win_rate", "VAL_mean_ror", "TEST_n", "TEST_win_rate", "TEST_mean_ror",
            "TEST_mean_pnl_per_contract", "G1", "G1b", "G2", "G3"]
    survivors = fac[fac["G3"]].sort_values("IS_t_stat", ascending=False)
    study = {
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "data": {"spy_daily": [str(p.index[0].date()), str(p.index[-1].date()), len(p)],
                 "vix1d_days": int(p["vix1d_close"].notna().sum()), "splits": SPLITS, "gates": GATES},
        "calibration": {"shapes": {str(k): v for k, v in smile.shapes.items()}, "level": {str(k): v for k, v in smile.level.items()},
                        "rmse_ratio": info["rmse_ratio"], "n_quotes": info["n"], "index": info["index"],
                        "points": {e: [{"K": q["K"], "type": "C" if q["call"] else "P", "z": q["z"], "ratio": q["ratio"],
                                        "model": float(smile.ratio(q["z"], d["days"])), "iv": q["iv"], "mid": q["mid"]}
                                       for q in d["points"]] for e, d in info["data"].items()},
                        "costs": costs.__dict__},
        "vrp": vrp_section(p),
        "factory": {"n_configs": int(len(fac)),
                    "by_mode": fac.groupby("mode").agg(n=("G1", "size"), G1=("G1", "sum"), G1b=("G1b", "sum"),
                                                       G2=("G2", "sum"), G3=("G3", "sum")).reset_index().to_dict("records"),
                    "best_by_family": best[keep].reset_index().to_dict("records"),
                    "survivors": survivors[keep].head(40).reset_index().to_dict("records")},
        "zero_dte": zero_dte_section(p, feats, smile, costs),
        "core": core_section(p, feats, smile, costs),
        "plan": plan_section(p, smile),
        "signals": tf_signals(p),
        "bars": bars_section(p),
    }
    (OUT / "study.json").write_text(json.dumps(clean(study), ensure_ascii=False))
    print("wrote", OUT / "study.json", f"{(OUT / 'study.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
