"""波段研究: catch a multi-day swing bottom and hold it to the swing top (the user's 759 -> 776 example, 2026-10-01..05).
Long only, OKX SPY-USDT-SWAP, 280 CNY isolated margin per trade at 50x (also 20x / 10x with the same 280 for reference).
Costs: 0.05% taker + 0.01% slippage per side (12 bp round trip), funding 0.01% per 8 h on notional for every calendar
8 h held (weekends included), maintenance margin 1% -> 50x liquidates 1.01% below the entry, 20x 4.04%, 10x 9.09%.
Path: SPY 1-minute bars (regular session) incl. the overnight gap at each open; OKX also trades overnight, so real
overnight dips can be deeper than the gap -> liquidation counts here are a LOWER bound.

Entries (fixed before looking at results):
  E1 日线RSI2<10 且在200日线上 -> next open          (TradingSPY500's validated rule)
  E2 日线RSI2<20 且在200日线上 -> next open
  E3 1小时 v6 底 (p>=0.55) + 200日线上 + 价格在5日线下 -> next 1m open
  E4 15分钟 v6 底 (p>=0.55) + 同样过滤
  E5 前一日 RSI2<20 且在200日线上的当天，第一个 15m/1h v6 底 -> next 1m open
Exits: every exit also has a protective stop 0.95% under the entry (just inside 50x liquidation) and a 10-day time stop.
  X1 收盘站上5日线 -> next open                       (TradingSPY500's rule)
  X2 日线RSI2>70 -> next open
  X3 “涨够了再看顶”: after the first close with RSI2>70 OR price >= entry + 1 daily ATR, exit at the first 15m v6 top
  X4 移动止盈: once up 0.5%, exit when price falls 0.8% from the highest high since entry
  X5 涨 0.5% 以后的第一个 1小时 v6 顶
v6 probabilities are out-of-sample only from 2024 (validation) on -> E3-E5 / X3 / X5 are judged on 2024-01..2026-10-05;
E1/E2 x X1/X2/X4 also on 2019-2023. Pick on VALIDATION (2024-01..2026-03-25), holdout 2026-03-26..10-05 reported once."""
import json, os, sys, itertools
import numpy as np, pandas as pd
import strat_lab as SL

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
MARGIN = 280.0; LEVS = (50, 20, 10); MMR = 0.01; FEE = 0.0012; FUND_8H = 0.0001; STOP = 0.0095; MAX_DAYS = 10
VAL0, VAL_END = "2024-01-01", "2026-03-25"


def daily():
    D = pd.read_parquet(os.path.join(DATA, "spy_1d.parquet")).set_index("t")
    c = D.c
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    D["rsi2"] = 100 - 100 / (1 + up / dn); D["sma5"] = c.rolling(5).mean(); D["sma200"] = c.rolling(200).mean()
    tr = pd.concat([D.h - D.l, (D.h - c.shift()).abs(), (D.l - c.shift()).abs()], axis=1).max(axis=1)
    D["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    D["trend"] = c > D.sma200
    return D


def v6_signals(side, tf, thr=0.55):
    """first-fire times (bar END) of out-of-sample v6 signals from the walk-forward cache + the replayed live days after it."""
    P = pd.read_parquet(os.path.join(HERE, "cache", "pred_%s.parquet" % side))
    P = P[(P.tf == tf) & (P.split != "train") & (P.p6 >= thr)].sort_values(["cand_i", "c"]).drop_duplicates("cand_i")
    b = pd.read_parquet(os.path.join(DATA, "spy_%s.parquet" % tf), columns=["t"])
    t = pd.to_datetime(b.t.to_numpy()[P.bar_j.to_numpy()])
    cp = "C:/Users/94868/spybt/watch/data/case_events.csv"
    if os.path.exists(cp):
        E = pd.read_csv(cp); E = E[(E.tf == tf) & (E.side == side) & (E.kind == "early") & (E.prob >= thr * 100)]
        t = t.append(pd.DatetimeIndex(pd.to_datetime(E.bar_t)))
    return np.sort(t.unique().values)


def simulate(M, D, entries, exit_rule, tops15, tops1h):
    """entries: sorted datetime64 entry-bar START times (we buy the open of the 1m bar starting then)."""
    days = list(D.index); dpos = {d: i for i, d in enumerate(days)}
    out = []; busy = np.datetime64("1970-01-01")
    for te in entries:
        if te < busy: continue
        k = int(np.searchsorted(M["t"], te + SL.ONE))
        if k >= len(M["t"]) or M["t"][k] - SL.ONE != te: continue
        e = M["o"][k]; d0 = M["d"][k]
        if d0 not in dpos: continue
        stop = e * (1 - STOP); peak = e; armed = False; i0 = dpos[d0]; exit_k = None; why = None; x = None
        hot = False
        kk = k
        while kk < len(M["t"]):
            day = M["d"][kk]; di = dpos.get(day)
            if di is None or di - i0 > MAX_DAYS:
                why = "time"; x = M["o"][kk]; exit_k = kk; break
            lo, hi, op = M["l"][kk], M["h"][kk], M["o"][kk]
            if lo <= stop:
                why = "stop"; x = min(stop, op); exit_k = kk; break
            peak = max(peak, hi)
            if exit_rule == "X4" and peak >= e * 1.005 and lo <= peak * (1 - 0.008):
                why = "trail"; x = min(peak * (1 - 0.008), op); exit_k = kk; break
            t_end = M["t"][kk]
            if exit_rule == "X3" and (hot or peak >= e + D.atr.iloc[i0 - 1] if i0 > 0 else False):
                hot = True
            if exit_rule == "X3" and hot and _hit(tops15, t_end):
                why = "top15"; exit_k = kk + 1; x = M["o"][exit_k] if exit_k < len(M["t"]) else M["c"][kk]; break
            if exit_rule == "X5" and peak >= e * 1.005 and _hit(tops1h, t_end):
                why = "top1h"; exit_k = kk + 1; x = M["o"][exit_k] if exit_k < len(M["t"]) else M["c"][kk]; break
            last_bar = kk + 1 >= len(M["t"]) or M["d"][kk + 1] != day
            if last_bar:                                                  # daily-close rules -> exit at the next open
                row = D.loc[day]
                if exit_rule == "X3" and row.rsi2 > 70: hot = True
                if (exit_rule == "X1" and row.c > row.sma5) or (exit_rule == "X2" and row.rsi2 > 70):
                    if kk + 1 < len(M["t"]):
                        why = "daily"; exit_k = kk + 1; x = M["o"][exit_k]
                        if x <= stop: why = "stop"; x = x               # gap through the stop at the open
                        break
            kk += 1
        if x is None: x = M["c"][-1]; exit_k = len(M["t"]) - 1; why = "open"
        lo_path = M["l"][k:exit_k + 1].min() if exit_k > k else min(M["l"][k], x)
        mae = min(lo_path, x) / e - 1
        held_h = (M["t"][min(exit_k, len(M["t"]) - 1)] - M["t"][k]) / np.timedelta64(1, "h")
        fund = FUND_8H * np.floor(max(held_h, 0) / 8.0 + 0.5)
        gross = x / e - 1
        out.append({"entry_t": str(M["t"][k] - SL.ONE)[:16], "exit_t": str(M["t"][min(exit_k, len(M["t"]) - 1)] - SL.ONE)[:16], "entry": e, "exit": x,
                    "gross": gross, "net": gross - FEE - fund, "fund": fund, "mae": mae, "why": why, "hours": held_h})
        busy = M["t"][min(exit_k, len(M["t"]) - 1)]
    return pd.DataFrame(out)


def _hit(times, t_end):
    i = np.searchsorted(times, t_end); return i < len(times) and times[i] == t_end


def money(T):
    r = {"n": int(len(T))}
    if not len(T): return r
    r.update({"win": round(float((T.net > 0).mean()), 3), "net_pct": round(float(T.net.mean() * 100), 3), "gross_pct": round(float(T.gross.mean() * 100), 3),
              "mae_p50": round(float(T.mae.median() * 100), 2), "hours": round(float(T.hours.mean()), 1), "stops": int((T.why == "stop").sum())})
    for L in LEVS:
        liq = 1 - (1 - 1 / L) / (1 - MMR)
        pnl = np.where(T.mae <= -liq, -MARGIN, MARGIN * L * T.net)        # liquidated -> the whole 280 is gone
        eq = np.cumsum(pnl); dd = float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min())
        r["x%d" % L] = {"liq": int((T.mae <= -liq).sum()), "yuan_per_trade": round(float(pnl.mean()), 1), "yuan_total": round(float(pnl.sum()), 0),
                        "max_dd_yuan": round(dd, 0), "worst_streak": int(_streak(pnl < 0))}
    return r


def _streak(b):
    m = c = 0
    for x in b: c = c + 1 if x else 0; m = max(m, c)
    return m


def split(T):
    d = T.entry_t.str[:10]
    return {"pre2024": T[d < VAL0], "val": T[(d >= VAL0) & (d <= VAL_END)], "hold": T[d > VAL_END], "oos": T[d >= VAL0]}


if __name__ == "__main__":
    M = SL.load_1m(); D = daily()
    days = list(D.index); nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    open_t = lambda day: np.datetime64(day + "T09:30")
    b15, b1h = v6_signals("bottom", "15m"), v6_signals("bottom", "1h")
    tops15, tops1h = v6_signals("top", "15m"), v6_signals("top", "1h")
    prev = D.shift(1)
    def filt(times):                                  # 200-day uptrend (yesterday) and price under yesterday's 5-day SMA
        keep = []
        cmap = dict(zip(M["t"], M["c"])) if False else None
        for t in times:
            day = str(t)[:10]
            if day not in D.index: continue
            p = prev.loc[day]
            if not (p.trend == True): continue
            k = int(np.searchsorted(M["t"], t)); px = M["c"][min(k, len(M["c"]) - 1)]
            if px < p.sma5: keep.append(t)
        return np.array(keep, dtype="datetime64[ns]")
    E = {}
    E["E1 日线RSI2<10"] = np.array([open_t(nxt[d]) for d, r in D.iterrows() if d in nxt and r.rsi2 < 10 and r.trend], dtype="datetime64[ns]")
    E["E2 日线RSI2<20"] = np.array([open_t(nxt[d]) for d, r in D.iterrows() if d in nxt and r.rsi2 < 20 and r.trend], dtype="datetime64[ns]")
    E["E3 1小时v6底+回调"] = filt(b1h)
    E["E4 15分钟v6底+回调"] = filt(b15)
    dip_days = {nxt[d] for d, r in D.iterrows() if d in nxt and r.rsi2 < 20 and r.trend}
    E["E5 RSI2<20次日的15m/1h底"] = np.array(sorted({t for t in np.r_[b15, b1h] if str(t)[:10] in dip_days and
                                                     t == min([u for u in np.r_[b15, b1h] if str(u)[:10] == str(t)[:10]])}), dtype="datetime64[ns]")
    res = []; trades = {}
    for (en, et), xr in itertools.product(E.items(), ("X1", "X2", "X3", "X4", "X5")):
        et = np.sort(et); et = et[et >= np.datetime64("2019-06-01")]
        T = simulate(M, D, et, xr, tops15, tops1h)
        if not len(T): continue
        name = "%s × %s" % (en, {"X1": "X1 站上5日线", "X2": "X2 RSI2>70", "X3": "X3 涨够后15m顶", "X4": "X4 移动止盈0.8%", "X5": "X5 涨0.5%后1h顶"}[xr])
        S = split(T); res.append({"name": name, "entry": en, "exit": xr, **{k: money(v) for k, v in S.items()}}); trades[name] = T
        print("%-44s val n%3d net%+.3f%% 50x %+7.1f元/笔 liq %d | hold n%3d %+7.1f元/笔 | pre24 n%3d %+7.1f元/笔" % (name, S["val"].shape[0],
              res[-1]["val"].get("net_pct", 0), res[-1]["val"].get("x50", {}).get("yuan_per_trade", 0), res[-1]["val"].get("x50", {}).get("liq", 0),
              S["hold"].shape[0], res[-1]["hold"].get("x50", {}).get("yuan_per_trade", 0), S["pre2024"].shape[0], res[-1]["pre2024"].get("x50", {}).get("yuan_per_trade", 0)), flush=True)
    # the user's case: what each method did around 2026-10-01 .. 10-05
    case = {}
    for n, T in trades.items():
        c = T[(T.entry_t >= "2026-09-30") & (T.entry_t <= "2026-10-05 23:59")]
        if len(c): case[n] = c.to_dict("records")
    json.dump({"rules": __doc__, "results": res, "case": case}, open(os.path.join(HERE, "results", "swing.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    pd.to_pickle(trades, os.path.join(HERE, "cache", "swing_trades.pkl"))
    print("\nCASE 2026-09-30..10-05:")
    for n, rows in case.items():
        for r in rows: print("  %-44s 买 %s @%.2f  卖 %s @%.2f  %s  净%+.2f%%  50x %+.0f元" % (n, r["entry_t"], r["entry"], r["exit_t"], r["exit"], r["why"], r["net"] * 100,
                              (-MARGIN if r["mae"] <= -(1 - (1 - 1 / 50) / (1 - MMR)) else MARGIN * 50 * r["net"])))
