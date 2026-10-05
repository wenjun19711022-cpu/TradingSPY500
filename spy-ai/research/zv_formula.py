"""SPY折线AI (moomoo main-chart formula) — numpy replica, fitted to the user's drawn pivots.
  BUY  = v5 bottom mark (FB) AND close is >= D% below the highest high of the last N bars AND no BUY since the last SELL
  SELL = v5 top mark (FT)    AND close is >= D% above the lowest low of the last N bars  AND no SELL since the last BUY
The alternation uses only raw marks (non-recursive, so the formula editor can compute it):
  BUY2 := BUY AND BARSLAST(SELL) < BARSLAST(REF(BUY,1))   (the latest raw SELL is newer than the previous raw BUY)
Fitted on the drawing (2026-08-05..10-05) — that is the brief: draw like the user. Trading results are reported separately."""
import itertools, json, math, os, sys
import numpy as np, pandas as pd
import overlap as O, strat_lab as SL, swing_study as S
sys.path.insert(0, "C:/Users/94868/spybt/watch"); import v5core

HERE = os.path.dirname(os.path.abspath(__file__))


def barslast(cond):
    out = np.full(len(cond), 10 ** 6, float); last = None
    for i, v in enumerate(cond):
        if v: last = i
        out[i] = (i - last) if last is not None else 10 ** 6
    return out


def marks(b, N, D):
    o, h, l, c = (b[k].to_numpy(float) for k in ("o", "h", "l", "c"))
    rb = v5core.side_signals(o, h, l, c, "bottom", 55.0); rt = v5core.side_signals(o, h, l, c, "top", 55.0)
    hh = pd.Series(h).rolling(N, min_periods=1).max().to_numpy(); ll = pd.Series(l).rolling(N, min_periods=1).min().to_numpy()
    buy = rb["fire"] & ((hh - c) / hh * 100 >= D); sell = rt["fire"] & ((c - ll) / ll * 100 >= D)
    prev_buy = np.r_[False, buy[:-1]]; prev_sell = np.r_[False, sell[:-1]]
    buy2 = buy & (barslast(sell) < barslast(prev_buy)); sell2 = sell & (barslast(buy) < barslast(prev_sell))
    return buy2, sell2


def overlap_rows(b, buy, sell, P, name):
    t = pd.to_datetime(b.t); c = b.c.to_numpy(float)
    sig = {"bottom": pd.DataFrame({"t": t[buy].values, "px": b.l.to_numpy(float)[buy]}), "top": pd.DataFrame({"t": t[sell].values, "px": b.h.to_numpy(float)[sell]})}
    return O.score(P, sig, name)


def trade(M, b, buy, sell, stop=0.025):
    """long only: buy the next 1m open after a BUY bar closes, sell the next 1m open after the next SELL bar (or stop / 10 days)"""
    t = pd.to_datetime(b.t).values; tb = np.sort(t[buy]); ts = np.sort(t[sell]); out = []; busy = np.datetime64("1970-01-01")
    for te in tb:
        if te < busy: continue
        k = int(np.searchsorted(M["t"], te + SL.ONE))
        if k >= len(M["t"]) or M["t"][k] != te + SL.ONE or M["em"][k] > 950: continue
        e = M["o"][k]; st = e * (1 - stop)
        i = np.searchsorted(ts, te, side="right"); tx = ts[i] if i < len(ts) else M["t"][-1]
        kx = min(int(np.searchsorted(M["t"], tx + SL.ONE)), len(M["t"]) - 1)
        kmax = min(kx, k + 10 * 390)
        seg = M["l"][k:kmax + 1]; js = np.flatnonzero(seg <= st)
        if len(js): kk = k + js[0]; x = min(st, M["o"][kk]); why = "stop"
        else: kk = kmax; x = M["o"][kk]; why = "sell" if kk == kx else "time"
        held = (M["t"][kk] - M["t"][k]) / np.timedelta64(1, "h")
        out.append({"entry_t": str(M["t"][k])[:16], "exit_t": str(M["t"][kk])[:16], "entry": e, "exit": x, "gross": x / e - 1,
                    "net": x / e - 1 - 0.0012 - 0.0001 * math.floor(max(held, 0) / 8 + 0.5), "mae": min(seg[:kk - k + 1].min(), x) / e - 1, "why": why, "hours": held})
        busy = M["t"][kk]
    return pd.DataFrame(out)


if __name__ == "__main__":
    P = O.pivots(); M = SL.load_1m(); res = []
    for tf, Ns in (("15m", (14, 28, 56)), ("1h", (7, 14, 21))):
        b = pd.read_parquet(os.path.join(HERE, "data", "spy_%s.parquet" % tf))
        for N, D in itertools.product(Ns, (0.0, 0.2, 0.3, 0.45, 0.6)):
            buy, sell = marks(b, N, D)
            R = overlap_rows(b, buy, sell, P, "%s N=%d D=%.2f" % (tf, N, D))
            f1 = []
            for r in R:
                rec, pre = r["召回率"], r["准确率(靠近你的点)"]; f1.append(2 * rec * pre / max(rec + pre, 1e-9))
            res.append({"tf": tf, "N": N, "D": D, "f1": round(float(np.mean(f1)), 3), "rows": R})
            print("%-4s N=%2d D=%.2f%%  底 召回%.2f 准确%.2f 距%s%% | 顶 召回%.2f 准确%.2f 距%s%% | F1 %.3f" % (tf, N, D, R[0]["召回率"], R[0]["准确率(靠近你的点)"],
                  R[0]["离你的点(中位%)"], R[1]["召回率"], R[1]["准确率(靠近你的点)"], R[1]["离你的点(中位%)"], np.mean(f1)), flush=True)
    best = {tf: max([r for r in res if r["tf"] == tf], key=lambda r: r["f1"]) for tf in ("15m", "1h")}
    out = {"fit": res, "best": {}}
    for tf, r in best.items():
        b = pd.read_parquet(os.path.join(HERE, "data", "spy_%s.parquet" % tf)); buy, sell = marks(b, r["N"], r["D"])
        T = trade(M, b, buy, sell); T = T[T.entry_t >= "2019-06-01"]
        d = T.entry_t.str[:10]
        sp = {"pre2024": T[d < "2024-01-01"], "2024+": T[d >= "2024-01-01"], "drawing": T[(d >= "2026-08-05") & (d <= "2026-10-05")]}
        out["best"][tf] = {"N": r["N"], "D": r["D"], "f1": r["f1"], "overlap": r["rows"], **{k: S.money(v) for k, v in sp.items()}}
        print("\nBEST %s N=%d D=%.2f%% (F1 %.3f)" % (tf, r["N"], r["D"], r["f1"]))
        for k, v in sp.items():
            m = S.money(v)
            if m.get("n"): print("  %-8s n%4d win %.0f%% net %+.3f%%/笔 | 50x %+.0f元/笔 爆仓%d | 20x %+.0f元/笔" % (k, m["n"], m["win"] * 100, m["net_pct"],
                                  m["x50"]["yuan_per_trade"], m["x50"]["liq"], m["x20"]["yuan_per_trade"]))
    json.dump(out, open(os.path.join(HERE, "results", "zv_formula.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
