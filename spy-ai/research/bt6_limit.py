"""Optimisation test: instead of buying the next open (by then ~60% of the move to target is gone), rest a LIMIT buy at
candidate low + x*ATR for `wait` bars after the signal (maker fee: ~7 bp round trip with a taker exit).
Fill = first 1m bar whose low <= limit (price = min(limit, bar open)); if that same bar also trades through the stop we count
it as stopped (conservative). Stop / target / 15:55 exit as in bt6. Config picked on VALIDATION net(7bp), holdout once."""
import json, os, itertools, time
import numpy as np, pandas as pd
import lib6, feat6, bt6

HERE = os.path.dirname(os.path.abspath(__file__))


def simulate_limit(sig, tfb, atr, M, x_atr, wait_bars, T_atr, tf_min):
    tend = pd.to_datetime(tfb.t).values; low = tfb.l.to_numpy(float)
    one = np.timedelta64(1, "m"); rows = []; busy_until = np.datetime64("1970-01-01")
    for r in sig.itertuples():
        te = tend[r.bar_j]
        if te < busy_until: continue
        k = np.searchsorted(M["t"], te + one)
        if k >= len(M["t"]) or M["t"][k] != te + one or M["em"][k] > 950: continue
        kend = M["kend"][k]
        if kend < k: continue
        a = atr[r.cand_i]; lo_c = low[r.cand_i]; stop = lo_c - 0.05 * a; lim = lo_c + x_atr * a; tgt = lo_c + T_atr * a
        kw = min(k + wait_bars * tf_min - 1, kend, k + np.searchsorted(M["em"][k:kend + 1], 950, side="right") - 1)
        fills = np.flatnonzero(M["l"][k:kw + 1] <= lim)
        if not len(fills): continue
        kf = k + fills[0]; e = min(lim, M["o"][kf])
        if e <= stop: e = M["o"][kf]                                         # gapped through the stop: filled at the open
        lo = M["l"][kf:kend + 1]; hi = M["h"][kf:kend + 1]; op = M["o"][kf:kend + 1]
        hit_s = np.flatnonzero(lo <= stop); hit_t = np.flatnonzero(hi >= tgt)
        hit_t = hit_t[hit_t > 0] if len(hit_t) and M["o"][kf] > lim else hit_t  # fill came from above: target can't be the fill bar
        cand = [(kend - kf, "time", M["c"][kend])]
        if len(hit_s): j = hit_s[0]; cand.append((j, "stop", min(stop, op[j]) if j > 0 else min(stop, e)))
        if len(hit_t): j = hit_t[0]; cand.append((j + 0.5, "target", max(tgt, op[j]) if j > 0 else tgt))
        j, why, xp = min(cand, key=lambda z: z[0]); jj = int(j)
        rows.append((str(te)[:16], M["d"][kf], r.split, e, xp, (xp - e) / e, (lo[:jj + 1].min() - e) / e, why, jj + 1 + (kf - k), (tgt - e) / e * 1e4, (e - stop) / e * 1e4))
        busy_until = M["t"][min(kf + jj, len(M["t"]) - 1)]
    return pd.DataFrame(rows, columns=["t", "day", "split", "entry", "exit", "ret", "mae", "why", "mins", "tgt_bp", "stop_bp"])


if __name__ == "__main__":
    t0 = time.time(); M = bt6.load_1m()
    predB = pd.read_parquet(os.path.join(HERE, "cache", "pred_bottom.parquet"))
    out = json.load(open(os.path.join(HERE, "results", "bt6.json"), encoding="utf-8")); out["limit"] = {}
    grid = list(itertools.product((0.55, 0.65), (0.1, 0.25, 0.5), (1, 3), (2.0, 3.0)))
    for tf in lib6.TFS:
        tfb = lib6.load(tf); atr = feat6._atr(tfb.h.to_numpy(float), tfb.l.to_numpy(float), tfb.c.to_numpy(float), 14)
        vdays = predB[(predB.tf == tf) & (predB.split == "val")].date.nunique(); hdays = predB[(predB.tf == tf) & (predB.split == "hold")].date.nunique()
        best = None; allv = []
        for thr, x, w, T in grid:
            sig = bt6.first_fire(predB, tf, thr, "p6"); sig = sig[sig.split == "val"]
            tr = simulate_limit(sig, tfb, atr, M, x, w, T, lib6.TF_MIN[tf])
            if len(tr) < 40: continue
            s = bt6.summarise(tr, vdays); s["fill_rate"] = round(len(tr) / max(len(sig), 1), 3)
            allv.append({"thr": thr, "x_atr": x, "wait_bars": w, "target_atr": T, "trades": s["trades"], "net_bp_7": s["net_bp_7"], "net_bp_12": s["net_bp_12"],
                         "gross_bp": s["gross_bp"], "win_7": s["win_7"]})
            if best is None or s["net_bp_7"] > best[1]["net_bp_7"]: best = ((thr, x, w, T), s)
        if not best: continue
        (thr, x, w, T), sv = best
        sig = bt6.first_fire(predB, tf, thr, "p6"); sig = sig[sig.split != "train"]
        tr = simulate_limit(sig, tfb, atr, M, x, w, T, lib6.TF_MIN[tf])
        hs = bt6.summarise(tr[tr.split == "hold"], hdays)
        out["limit"][tf] = {"config": {"thr": thr, "x_atr": x, "wait_bars": w, "target_atr": T}, "val": sv, "hold": hs, "grid": allv}
        out["curves"]["%s|v6|limit" % tf] = bt6.equity_curve(tr, 50, 7)
        print("%-3s LIMIT best %s | VAL n %d net7 %.2f gross %.2f win7 %.3f | HOLD n %s net7 %s gross %s  %.0fs" % (
            tf, out["limit"][tf]["config"], sv["trades"], sv["net_bp_7"], sv["gross_bp"], sv["win_7"], hs.get("trades"), hs.get("net_bp_7"),
            hs.get("gross_bp"), time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "bt6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved")
