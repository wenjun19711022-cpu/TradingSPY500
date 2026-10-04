"""Range idea: at 10:00 forecast the rest of the day's range, buy near the low end, stop under
the forecast low, take profit a bit higher -> results/range_study.json, results/range_model.json

    python -m spylev.data.minute_hist          # public 2022-10..2024-12 minute bars (once)
    python scripts/run_range_study.py

Pre-registered (fixed before any P&L was looked at): decision time {10:00, 10:30} x entry
{0.5 sigma below, 1 sigma below, just above the highest support inside the band} x take profit
{entry + 0.5 sigma, middle of the band}; stop always 0.02% under the forecast low that holds 80%
of days = 12 configs. Splits: IS 2022-10..2023-12 (the model is fitted here only), VAL 2024H1,
TEST 2024H2. The last weeks of 2026 from IBKR (5-minute bars) are shown separately, too few
days to judge anything.
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
from spylev.data.minute_hist import premarket_levels  # noqa: E402
from spylev.data.sources import load_ibkr_json  # noqa: E402
from spylev.dip.perp import PerpCosts  # noqa: E402
from spylev.range.engine import RangeSpec, run, simulate, stats  # noqa: E402
from spylev.range.model import PROB_LEVELS, RangeModel, five_minute, forecast_table, p_bracket, p_touch_below, session_table  # noqa: E402
from spylev.ta import daily_levels  # noqa: E402

OUT = ROOT / "results"
IBKR = ROOT / "data" / "raw" / "ibkr"
SPLITS = {"IS": ("2022-10-01", "2023-12-31"), "VAL": ("2024-01-01", "2024-06-30"), "TEST": ("2024-07-01", "2024-12-31")}
FIT_END = pd.Timestamp("2024-01-01")
GRID = [RangeSpec(t0, "sigma", e, 0.8, tp) for t0 in (30, 60) for e in (0.5, 1.0) for tp in ("0.5s", "mid")] + \
       [RangeSpec(t0, "level", 0.5, 0.8, tp) for t0 in (30, 60) for tp in ("0.5s", "mid")]
REF = "t0=30|entry=0.5s|stop=p0.8|tp=0.5s"


def cut(t, a, b):
    return t[(t.index >= pd.Timestamp(a)) & (t.index <= pd.Timestamp(b))]


def coverage(fc: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in SPLITS.items():
        d = cut(fc, a, b)
        d = d[d["sigma"].notna() & d["full"]]
        real = np.sqrt(d["rv_rest"])
        rec = {"days": int(len(d)), "corr_log_sigma": float(np.corrcoef(np.log(d["sigma"]), np.log(real))[0, 1]),
               "realized_over_forecast": float((real / d["sigma"]).median()), "bands": {}}
        for p in PROB_LEVELS:
            k = f"{p:g}"
            hi_ok, lo_ok = d["rest_high"] <= d[f"high_{k}"], d["rest_low"] >= d[f"low_{k}"]
            rec["bands"][k] = {"high_holds": float(hi_ok.mean()), "low_holds": float(lo_ok.mean()), "both_hold": float((hi_ok & lo_ok).mean()),
                               "width_bp": float(((d[f"high_{k}"] / d[f"low_{k}"] - 1) * 1e4).mean())}
        out[name] = rec
    return out


def naive_width(fc: pd.DataFrame, p=0.8) -> dict:
    """Width of a constant-percentage band with the same in-sample coverage (sharpness check)."""
    d = fc[fc["sigma"].notna() & fc["full"]]
    isd = d[d.index < FIT_END]
    up = np.log(isd["rest_high"] / isd["s0"])
    dn = np.log(isd["s0"] / isd["rest_low"])
    qu, qd = np.quantile(up, p), np.quantile(dn, p)
    out = {}
    for name, (a, b) in SPLITS.items():
        x = cut(d, a, b)
        hi_ok = np.log(x["rest_high"] / x["s0"]) <= qu
        lo_ok = np.log(x["s0"] / x["rest_low"]) <= qd
        out[name] = {"width_bp": float((np.exp(qu + qd) - 1) * 1e4), "high_holds": float(hi_ok.mean()), "low_holds": float(lo_ok.mean())}
    return out


def reliability(t: pd.DataFrame) -> list:
    f = t[t["filled"]].copy()
    if f.empty:
        return []
    f["bucket"] = pd.cut(f["p_fair"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    g = f.groupby("bucket", observed=True)
    return [{"bucket": str(k), "n": int(len(v)), "p_fair": float(v["p_fair"].mean()), "won": float((v["reason"] == "tp").mean())}
            for k, v in g]


def regimes(t: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """Does the edge over the fair probability depend on the setting? (diagnostic, not a rule)"""
    f = t[t["filled"]].join(fc[["vix_prev", "open"]], how="left")
    f["edge"] = (f["reason"] == "tp").astype(float) - f["p_fair"]
    isv = fc.loc[fc.index < FIT_END, "vix_prev"].dropna()
    lo, hi = np.quantile(isv, [1 / 3, 2 / 3])
    groups = {"VIX 低": f["vix_prev"] < lo, "VIX 中": (f["vix_prev"] >= lo) & (f["vix_prev"] < hi), "VIX 高": f["vix_prev"] >= hi,
              "开盘后先跌": f["s0"] < f["open"], "开盘后先涨": f["s0"] >= f["open"]}
    out = {}
    for k, m in groups.items():
        e = f.loc[m, "edge"]
        out[k] = {"n": int(len(e)), "edge": float(e.mean()) if len(e) else None,
                  "t": float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))) if len(e) > 2 and e.std() > 0 else None,
                  "net_bp": float(f.loc[m, "net"].mean() * 1e4) if len(e) else None}
    out["_vix_cuts"] = [float(lo), float(hi)]
    return out


def bracket_math(entry, stop, tp, costs=PerpCosts(), lev=20.0) -> dict:
    """The user's kind of ticket: fair odds (no time limit) and what costs do to it."""
    p_up = np.log(entry / stop) / np.log(tp / entry * entry / stop)
    win, loss = tp / entry - 1, stop / entry - 1
    win_net, loss_net = win - 2 * costs.maker, loss - costs.maker - costs.taker - costs.slippage
    ev_net = p_up * win_net + (1 - p_up) * loss_net
    need = -loss_net / (win_net - loss_net)
    return {"entry": entry, "stop": stop, "tp": tp, "p_fair": float(p_up), "win_bp": float(win * 1e4), "loss_bp": float(loss * 1e4),
            "ev_gross_bp": float((p_up * win + (1 - p_up) * loss) * 1e4), "ev_net_bp": float(ev_net * 1e4),
            "breakeven_win_rate": float(need), "lev": lev, "ev_net_account_pct": float(ev_net * lev * 100),
            "win_account_pct": float(win_net * lev * 100), "loss_account_pct": float(loss_net * lev * 100),
            "liquidation_drop_pct": float(costs.liq_drop(lev) * 100)}


def recent_days(model: RangeModel, vix: pd.Series) -> dict:
    """2026 sessions from the IBKR files: the 10:00 forecast for 10/01 and 10/02 (the days in the
    user's screenshots) and the reference config on the last weeks of 5-minute bars."""
    b5 = load_ibkr_json(IBKR / "SPY_5m_recent.json")
    m1 = load_ibkr_json(IBKR / "SPY_1m_recent.json")
    tab = session_table(b5, vix, model.t0)
    fc = forecast_table(model, tab)
    days = []
    for d in ("2026-10-01", "2026-10-02"):
        if pd.Timestamp(d) not in fc.index:
            continue
        r = fc.loc[pd.Timestamp(d)]
        g = m1[m1.index.normalize().tz_localize(None) == pd.Timestamp(d)]
        rec = {"date": d, "open": float(r["open"]), "s0_1000": float(r["s0"]), "high_0930_1000": float(r["h0"]), "low_0930_1000": float(r["l0"]),
               "sigma_rest_pct": float(r["sigma"] * 100), "vix_prev": float(r["vix_prev"]),
               "day_high": float(g["high"].max()), "day_low": float(g["low"].min()), "close": float(g["close"].iloc[-1]),
               "rest_high": float(r["rest_high"]), "rest_low": float(r["rest_low"]), "bands": {}}
        for p in (0.8, 0.9):
            k = f"{p:g}"
            rec["bands"][k] = {"low": float(r[f"low_{k}"]), "high": float(r[f"high_{k}"]),
                               "low_held": bool(r["rest_low"] >= r[f"low_{k}"]), "high_held": bool(r["rest_high"] <= r[f"high_{k}"])}
        spec = RangeSpec()
        from spylev.range.engine import plan
        tk = plan(model, r["s0"], r["sigma"], spec)
        if tk is not None and len(g):
            res = simulate(g, tk, model, r["sigma"], spec, PerpCosts())
            rec["auto_plan"] = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in res.items()}
        days.append(rec)
    # the user's own ticket on 10/02: buy 768.5, stop 767, take profit 769.5
    user = None
    if pd.Timestamp("2026-10-02") in fc.index:
        r = fc.loc[pd.Timestamp("2026-10-02")]
        g = m1[m1.index.normalize().tz_localize(None) == pd.Timestamp("2026-10-02")]
        tk = {"entry": 768.5, "stop": 767.0, "tp": 769.5, "band_low": float(r["low_0.8"]), "band_high": float(r["high_0.8"]), "why": "用户的单子"}
        res = simulate(g, tk, model, r["sigma"], RangeSpec(), PerpCosts())
        var = model.sigma_between(r["sigma"], model.t0) ** 2
        user = {"ticket": tk, "at_1000": {"p_fill": p_touch_below(r["s0"], 768.5, var),
                                          "p_tp_after_fill": p_bracket(768.5, 767.0, 769.5, var)[0]},
                "result": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in res.items()},
                "math_20x": bracket_math(768.5, 767.0, 769.5)}
    # reference config on the 5-minute holdout (coarser fills)
    spec = next(s for s in GRID if s.key == REF)
    t = run(b5, fc[fc.index >= pd.Timestamp("2026-09-17")], model, spec)
    return {"days": days, "user_ticket": user, "holdout_5m": stats(t), "holdout_rows": [
        {"date": str(i.date()), "filled": bool(x["filled"]), "reason": x["reason"],
         "net_bp": float(x["net"] * 1e4) if x["filled"] else None} for i, x in t.iterrows()]}


def main():
    warnings.filterwarnings("ignore")
    m1 = store.load("SPY", "1m")
    if m1.empty:
        from spylev.data.minute_hist import build
        build()
        m1 = store.load("SPY", "1m")
    b5 = five_minute(m1)
    vix = load_ibkr_json(IBKR / "VIX_1d.json")["close"]
    lv = daily_levels(m1).join(premarket_levels())
    lv.index = lv.index.tz_localize(None)
    out = {"generated": str(pd.Timestamp.now(tz="UTC")), "data": {"start": str(m1.index[0])[:10], "end": str(m1.index[-1])[:10],
           "sessions": int(m1.index.normalize().nunique())}, "splits": SPLITS, "costs": PerpCosts().__dict__,
           "models": {}, "coverage": {}, "naive": {}, "grid": []}
    models, fcs = {}, {}
    for t0 in (30, 60):
        tab = session_table(b5, vix, t0)
        mdl = RangeModel.fit(tab[tab.index < FIT_END], b5[b5.index.tz_localize(None) < FIT_END], t0)
        models[t0], fcs[t0] = mdl, forecast_table(mdl, tab)
        out["models"][t0] = {"beta": mdl.beta, "features": ["const", "vix", "rv5", "rv1", "open"], "r2_is": mdl.r2,
                             "qu": mdl.qu, "qd": mdl.qd, "joint_is": mdl.joint, "fit_days": mdl.fit_days}
        out["coverage"][t0] = coverage(fcs[t0])
        out["naive"][t0] = naive_width(fcs[t0])
    models[30].save(OUT / "range_model.json")
    ref_trades = None
    pooled = []
    for sp in GRID:
        t = run(m1, fcs[sp.t0], models[sp.t0], sp, lv)
        rec = {"key": sp.key, **sp.as_dict()}
        for k, (a, b) in SPLITS.items():
            rec[k] = stats(cut(t, a, b))
        rec["ALL"] = stats(t)
        g = rec["IS"]
        rec["G1"] = bool(g.get("n", 0) >= 50 and g.get("mean_net_bp", -1) > 0 and g.get("t_stat", 0) >= 2 and g.get("profit_factor", 0) >= 1.2)
        rec["G2"] = bool(rec["G1"] and rec["VAL"].get("mean_net_bp", -1) > 0)
        rec["G3"] = bool(rec["G2"] and rec["TEST"].get("mean_net_bp", -1) > 0)
        f = t[t["filled"]]
        if len(f):
            rec["zero_cost_bp"] = float(f["gross"].mean() * 1e4)
        out["grid"].append(rec)
        pooled.append(t)
        a = rec["ALL"]
        print(f"{sp.key:<40} n={a.get('n', 0):>4} win={a.get('win_rate', 0):.0%} fair={a.get('fair_win_rate', 0):.0%} "
              f"gross={a.get('mean_gross_bp', 0):+.1f}bp net={a.get('mean_net_bp', 0):+.1f}bp 20x={a.get('lev20_final', 0):.3f} "
              f"G={int(rec['G1'])}{int(rec['G2'])}{int(rec['G3'])}")
        if sp.key == REF:
            ref_trades = t
    out["reliability_ref"] = reliability(ref_trades)
    out["reliability_pooled"] = reliability(pd.concat(pooled))
    out["regimes_ref"] = regimes(ref_trades, fcs[30])
    out["recent"] = recent_days(models[30], vix)
    out["examples"] = {"user_10_02_20x": bracket_math(768.5, 767.0, 769.5)}
    (OUT / "range_study.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda v: None if not isinstance(v, (pd.Timestamp,)) else str(v)))
    c = out["coverage"][30]
    print("coverage 80% one-sided (high/low/both):", {k: (round(v["bands"]["0.8"]["high_holds"], 2), round(v["bands"]["0.8"]["low_holds"], 2),
                                                          round(v["bands"]["0.8"]["both_hold"], 2)) for k, v in c.items()})
    print("reliability:", out["reliability_pooled"])
    print("recent:", json.dumps(out["recent"]["days"], ensure_ascii=False, default=str)[:1500])
    print("user ticket:", json.dumps(out["recent"]["user_ticket"], ensure_ascii=False, default=str)[:1500])
    print("wrote", OUT / "range_study.json")


if __name__ == "__main__":
    main()
