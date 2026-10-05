"""折线模式 (ZV): trade the user's own drawing style in real time.
The user's lines = ZigZag 0.45% on 1h bars (zigfit.py). A zigzag pivot is only KNOWN after a 0.45% reversal (~0.5-0.6%
past the pivot), so every rule below tries to act EARLIER with v6 bottom/top signals, using the confirmed zigzag only as a
state filter (are we in a down-leg or an up-leg?).

  S1 确认折线: buy when a bottom is confirmed (+theta from the low), sell when a top is confirmed (-theta from the high)
  S2 v6+折线:  while the last confirmed pivot is a TOP and price is >= d below it, the first v6 bottom (tfs, p>=thr) -> buy;
              sell at the first v6 top (tfs, p>=thr) once price is >= u above the entry, or when the zigzag confirms a top;
              stop s under the entry; 10-day time stop
Long only, 280 CNY isolated margin, OKX costs (12 bp + funding 0.01%/8h), liquidation at 1/L - 1%.
Parameters chosen on VALIDATION 2024-01..2026-03-25, holdout 2026-03-26..2026-10-05 reported once (it contains the user's
drawing, 2026-08-05..10-05 -> also scored for how close the buys/sells land to the drawn bottoms/tops)."""
import itertools, json, math, os
import numpy as np, pandas as pd
import strat_lab as SL, swing_study as S
from zigfit import zigzag

HERE = os.path.dirname(os.path.abspath(__file__))
VAL0, VAL_END = "2024-01-01", "2026-03-25"


def zz_events(theta):
    b = pd.read_parquet(os.path.join(HERE, "data", "spy_1h.parquet"))
    z = zigzag(b.h.to_numpy(float), b.l.to_numpy(float), theta)
    t = pd.to_datetime(b.t).values
    return pd.DataFrame({"conf_t": [t[c] for _, _, _, c in z], "kind": [k for _, k, _, _ in z], "px": [p for _, _, p, _ in z], "piv_t": [t[i] for i, _, _, _ in z]})


def simulate(M, sig_b, sig_t, Z, d, u, s, rule):
    """sig_b / sig_t: sorted datetime64 bar-END times of v6 bottom / top signals; Z: confirmed zigzag events"""
    zc = Z.conf_t.values; zk = Z.kind.values; zp = Z.px.values
    tops_conf = zc[zk == "T"]; out = []; busy = np.datetime64("1970-01-01")
    if rule == "S1":
        entries = zc[zk == "B"]
    else:
        entries = sig_b
    for te in entries:
        if te < busy: continue
        k = int(np.searchsorted(M["t"], te + SL.ONE))
        if k >= len(M["t"]) or M["t"][k] != te + SL.ONE or M["em"][k] > 950: continue
        e = M["o"][k]
        if rule == "S2":
            i = np.searchsorted(zc, te, side="right") - 1          # last pivot confirmed at or before the signal
            if i < 0 or zk[i] != "T" or e > zp[i] * (1 - d): continue
        stop = e * (1 - s); kk = k; x = None; why = None; day0 = M["d"][k]; ndays = 0
        it = np.searchsorted(sig_t, te, side="right"); iz = np.searchsorted(tops_conf, te, side="right")
        while kk < len(M["t"]):
            if M["d"][kk] != M["d"][kk - 1] if kk > k else False:
                ndays += 1
                if ndays > 10: x, why = M["o"][kk], "time"; break
            if M["l"][kk] <= stop: x, why = min(stop, M["o"][kk]), "stop"; break
            tend = M["t"][kk]
            # top-confirmation fallback (S1's only exit)
            if iz < len(tops_conf) and tops_conf[iz] <= tend:
                if kk + 1 < len(M["t"]): x, why = M["o"][kk + 1], "zz_top"; kk += 1
                else: x, why = M["c"][kk], "zz_top"
                break
            if rule == "S2":
                while it < len(sig_t) and sig_t[it] < tend: it += 1
                if it < len(sig_t) and sig_t[it] == tend and M["c"][kk] >= e * (1 + u):
                    if kk + 1 < len(M["t"]): x, why = M["o"][kk + 1], "v6_top"; kk += 1
                    else: x, why = M["c"][kk], "v6_top"
                    break
            kk += 1
        if x is None: x, why, kk = M["c"][-1], "open", len(M["t"]) - 1
        mae = min(M["l"][k:kk + 1].min(), x) / e - 1
        held = (M["t"][kk] - M["t"][k]) / np.timedelta64(1, "h")
        fund = 0.0001 * math.floor(max(held, 0) / 8 + 0.5)
        out.append({"entry_t": str(M["t"][k] - SL.ONE)[:16], "exit_t": str(M["t"][kk] - SL.ONE)[:16], "entry": e, "exit": x,
                    "gross": x / e - 1, "net": x / e - 1 - 0.0012 - fund, "mae": mae, "why": why, "hours": held})
        busy = M["t"][kk]
    return pd.DataFrame(out)


def score(T):
    d = T.entry_t.str[:10]
    return {"val": S.money(T[(d >= VAL0) & (d <= VAL_END)]), "hold": S.money(T[d > VAL_END]), "oos": S.money(T[d >= VAL0])}


if __name__ == "__main__":
    M = SL.load_1m(); res = []; trades = {}
    for th in (0.0030, 0.0045, 0.0060, 0.0080):                    # S1: pure confirmed zigzag
        Z = zz_events(th); E0 = np.array([], dtype="datetime64[ns]"); T = simulate(M, E0, E0, Z, 0, 0, 0.025, "S1"); T = T[T.entry_t >= "2019-06-01"]
        r = {"name": "S1 确认折线 %.2f%%" % (th * 100), **score(T)}; res.append(r); trades[r["name"]] = T
        print("%-34s val n%3d net %+.3f%% | hold n%3d net %+.3f%%" % (r["name"], r["val"]["n"], r["val"].get("net_pct", 0), r["hold"]["n"], r["hold"].get("net_pct", 0)), flush=True)
    Z = zz_events(0.0045)
    SIG = {}
    for thr in (0.55, 0.65):
        b15, b1h, t15, t1h = (S.v6_signals(sd, tf, thr) for sd, tf in (("bottom", "15m"), ("bottom", "1h"), ("top", "15m"), ("top", "1h")))
        SIG[thr] = {"15m+1h": (np.sort(np.r_[b15, b1h]), np.sort(np.r_[t15, t1h])), "1h": (b1h, t1h), "15m": (b15, t15)}
    for thr, tfs, d, u, s in itertools.product((0.55, 0.65), ("15m+1h", "1h", "15m"), (0.003, 0.006), (0.003, 0.006, 0.010), (0.008, 0.015, 0.025)):
        sb, st = SIG[thr][tfs]
        T = simulate(M, sb, st, Z, d, u, s, "S2")
        if len(T) < 10: continue
        r = {"name": "S2 v6+折线 %s p≥%.2f 回调%.1f%% 涨%.1f%%后见顶 止损%.1f%%" % (tfs, thr, d * 100, u * 100, s * 100),
             "params": {"thr": thr, "tfs": tfs, "d": d, "u": u, "s": s}, **score(T)}
        res.append(r); trades[r["name"]] = T
    S2 = [r for r in res if r["name"].startswith("S2") and r["val"].get("n", 0) >= 30]
    best = max(S2, key=lambda r: r["val"]["net_pct"])
    top5 = sorted(S2, key=lambda r: r["val"]["net_pct"], reverse=True)[:8]
    print("\nS2 top by validation net/trade (1x):")
    for r in top5:
        v, h = r["val"], r["hold"]
        print("  %-62s val n%3d win%3.0f%% net%+.3f%% 50x%+5.0f元 liq%2d 20x%+5.0f元 | hold n%2d net%+.3f%% 50x%+5.0f元 20x%+5.0f元" % (
            r["name"][3:], v["n"], v["win"] * 100, v["net_pct"], v["x50"]["yuan_per_trade"], v["x50"]["liq"], v["x20"]["yuan_per_trade"],
            h.get("n", 0), h.get("net_pct", 0), h.get("x50", {}).get("yuan_per_trade", 0), h.get("x20", {}).get("yuan_per_trade", 0)))
    json.dump({"rules": __doc__, "results": res, "best": best["name"]}, open(os.path.join(HERE, "results", "zv.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    pd.to_pickle(trades, os.path.join(HERE, "cache", "zv_trades.pkl"))
    print("\nBEST (validation):", best["name"])
