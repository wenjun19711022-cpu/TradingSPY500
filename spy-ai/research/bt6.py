"""Long-only intraday trading backtest of the bottom signals (OKX SPY-USDT-SWAP stand-in, user's fixed rules):
enter at the next 1-minute open after the signal bar closes, stop 0.05 ATR under the candidate low, optional target
(T x ATR above the candidate low), optional exit on a top signal of the same timeframe, flat at 15:55 ET, one position
at a time per timeframe. Price path is walked on 1-minute bars (stop checked before target inside a bar = conservative).
Costs per round trip: 12 bp (taker + slippage) / 7 bp (maker entry). Account impact per trade = 1% margin x leverage x return;
a move of (1/L - 1% maintenance) against the entry before the stop = liquidation (whole 1% margin lost).
Config chosen on VALIDATION (2024-01 .. 2026-03-25) only; holdout (2026-03-26 .. 09-30) reported once."""
import json, os, sys, time, itertools
import numpy as np, pandas as pd
import lib6, feat6

HERE = os.path.dirname(os.path.abspath(__file__))
TFS = lib6.TFS; COSTS = (12, 7); LEVS = (20, 50); MMR = 0.01


def load_1m():
    b = pd.read_parquet(os.path.join(HERE, "data", "spy_1m.parquet"))
    t = pd.to_datetime(b.t).values
    em = (b.t.str[11:13].astype(int) * 60 + b.t.str[14:16].astype(int)).to_numpy(); d = b.t.str[:10].to_numpy()
    f = pd.DataFrame({"d": d, "i": np.arange(len(b))})
    kend = f.d.map(f[em <= 955].groupby("d").i.max()).fillna(-1).astype(int).to_numpy()     # last bar ending <= 15:55
    return dict(t=t, o=b.o.to_numpy(float), h=b.h.to_numpy(float), l=b.l.to_numpy(float), c=b.c.to_numpy(float), em=em, d=d, kend=kend)


def first_fire(pred, tf, thr, col):
    X = pred[(pred.tf == tf) & (pred[col] >= thr)].sort_values(["cand_i", "c"]).drop_duplicates("cand_i")
    return X.sort_values("bar_j")


def simulate(sig, tops_t, tfb, atr, M, T_atr, use_top, min_tgt_bp):
    """sig: fired bottom rows (bar_j, cand_i, split). Returns trades DataFrame."""
    tend = pd.to_datetime(tfb.t).values; low = tfb.l.to_numpy(float)
    one = np.timedelta64(1, "m"); rows = []; busy_until = np.datetime64("1970-01-01")
    for r in sig.itertuples():
        te = tend[r.bar_j]
        if te < busy_until: continue
        k = np.searchsorted(M["t"], te + one)                          # 1m bar that starts when the signal bar closes
        if k >= len(M["t"]) or M["t"][k] != te + one or M["em"][k] > 950: continue     # no entry after 15:50
        day = M["d"][k]; e = M["o"][k]
        a = atr[r.cand_i]; stop = low[r.cand_i] - 0.05 * a
        tgt = low[r.cand_i] + T_atr * a if T_atr else np.inf
        if e <= stop: continue
        if T_atr and (tgt - e) / e * 1e4 < min_tgt_bp: continue      # target too close to pay for costs
        kend = M["kend"][k]
        if kend < k: continue
        top_k = None
        if use_top and len(tops_t):
            i = np.searchsorted(tops_t, te, side="right")
            if i < len(tops_t) and tops_t[i] < M["t"][kend]:
                kk = np.searchsorted(M["t"], tops_t[i] + one)
                if kk <= kend and M["d"][kk] == day: top_k = kk
        lo = M["l"][k:kend + 1]; hi = M["h"][k:kend + 1]; op = M["o"][k:kend + 1]
        hit_s = np.flatnonzero(lo <= stop); hit_t = np.flatnonzero(hi >= tgt)
        cand = [(kend - k, "time", M["c"][kend])]
        if len(hit_s): j = hit_s[0]; cand.append((j, "stop", min(stop, op[j])))
        if len(hit_t): j = hit_t[0]; cand.append((j + 0.5, "target", max(tgt, op[j])))       # same bar: stop first
        if top_k is not None: cand.append((top_k - k - 0.25, "top", M["o"][top_k]))
        j, why, x = min(cand, key=lambda z: z[0]); jj = int(j)
        mae = (lo[:jj + 1].min() - e) / e
        rows.append((str(te)[:16], day, r.split, e, x, (x - e) / e, mae, why, jj + 1, (tgt - e) / e * 1e4 if T_atr else np.nan, (e - stop) / e * 1e4))
        busy_until = M["t"][min(k + jj, len(M["t"]) - 1)]
    return pd.DataFrame(rows, columns=["t", "day", "split", "entry", "exit", "ret", "mae", "why", "mins", "tgt_bp", "stop_bp"])


def summarise(tr, days):
    if len(tr) == 0: return {"trades": 0}
    out = {"trades": int(len(tr)), "per_day": round(len(tr) / max(days, 1), 2), "gross_bp": round(float(tr.ret.mean() * 1e4), 2),
           "hold_min": round(float(tr.mins.mean()), 1), "why": tr.why.value_counts(normalize=True).round(3).to_dict(),
           "avg_tgt_bp": round(float(tr.tgt_bp.mean()), 2) if tr.tgt_bp.notna().any() else None, "avg_stop_bp": round(float(tr.stop_bp.mean()), 2),
           "avg_win_bp": round(float(tr.ret[tr.ret > 0].mean() * 1e4), 2) if (tr.ret > 0).any() else None,
           "avg_loss_bp": round(float(tr.ret[tr.ret <= 0].mean() * 1e4), 2) if (tr.ret <= 0).any() else None,
           "gross_win": round(float((tr.ret > 0).mean()), 3)}
    for cst in COSTS:
        net = tr.ret - cst / 1e4
        out["net_bp_%d" % cst] = round(float(net.mean() * 1e4), 2); out["win_%d" % cst] = round(float((net > 0).mean()), 3)
        out["pf_%d" % cst] = round(float(net[net > 0].sum() / max(-net[net <= 0].sum(), 1e-12)), 3)
    for L in LEVS:
        liq = tr.mae <= -(1.0 / L - MMR)
        acct = np.where(liq, -0.01, 0.01 * L * (tr.ret - 12 / 1e4))
        eq = np.cumprod(1 + acct); dd = eq / np.maximum.accumulate(eq) - 1
        out["x%d" % L] = {"liquidations": int(liq.sum()), "acct_total_pct": round(float((eq[-1] - 1) * 100), 2),
                          "max_dd_pct": round(float(dd.min() * 100), 2)}
    return out


def equity_curve(tr, L=50, cost=12):
    if len(tr) == 0: return []
    liq = tr.mae <= -(1.0 / L - MMR); acct = np.where(liq, -0.01, 0.01 * L * (tr.ret - cost / 1e4))
    s = pd.Series(acct, index=tr.day).groupby(level=0).apply(lambda x: np.prod(1 + x) - 1)
    eq = np.cumprod(1 + s.to_numpy()); return [[d, round(float((v - 1) * 100), 3)] for d, v in zip(s.index, eq)]


if __name__ == "__main__":
    t0 = time.time(); M = load_1m()
    predB = pd.read_parquet(os.path.join(HERE, "cache", "pred_bottom.parquet")); predT = pd.read_parquet(os.path.join(HERE, "cache", "pred_top.parquet"))
    out = {"rules": __doc__, "default": {}, "grid_best": {}, "curves": {}}
    grid = list(itertools.product((0.55, 0.65, 0.75), (1.5, 2.0, 3.0, None), (True, False), (0, 24, 36)))
    for tf in TFS:
        tfb = lib6.load(tf); atr = feat6._atr(tfb.h.to_numpy(float), tfb.l.to_numpy(float), tfb.c.to_numpy(float), 14)
        tend = pd.to_datetime(tfb.t).values
        vdays = predB[(predB.tf == tf) & (predB.split == "val")].date.nunique(); hdays = predB[(predB.tf == tf) & (predB.split == "hold")].date.nunique()
        for model, col in (("v5重训", "p5"), ("v6", "p6")):
            # default rule (same for both models): thr 0.55, target 2 ATR (= the label), exit on top 0.65, no cost filter
            sig = first_fire(predB, tf, 0.55, col); sig = sig[sig.split != "train"]
            tops = np.sort(tend[first_fire(predT, tf, 0.65, col).bar_j.to_numpy()])
            tr = simulate(sig, tops, tfb, atr, M, 2.0, True, 0)
            out["default"]["%s|%s" % (tf, model)] = {"val": summarise(tr[tr.split == "val"], vdays), "hold": summarise(tr[tr.split == "hold"], hdays)}
            out["curves"]["%s|%s|default" % (tf, model)] = equity_curve(tr)
            d = out["default"]["%s|%s" % (tf, model)]
            print("%-3s %-6s default VAL n %5s net12 %6s gross %6s | HOLD n %5s net12 %6s  %.0fs" % (tf, model, d["val"].get("trades"), d["val"].get("net_bp_12"),
                  d["val"].get("gross_bp"), d["hold"].get("trades"), d["hold"].get("net_bp_12"), time.time() - t0), flush=True)
        # grid search on validation (v6 only), holdout of the chosen config reported once
        best = None; tops_cache = {}
        for thr, T, use_top, mtb in grid:
            sig = first_fire(predB, tf, thr, "p6"); sig = sig[sig.split == "val"]
            if use_top not in tops_cache: tops_cache[use_top] = np.sort(tend[first_fire(predT, tf, 0.65, "p6").bar_j.to_numpy()])
            tr = simulate(sig, tops_cache[use_top], tfb, atr, M, T, use_top, mtb if T else 0)
            if len(tr) < 60: continue
            s = summarise(tr, vdays)
            if best is None or s["net_bp_12"] > best[1]["net_bp_12"]: best = ((thr, T, use_top, mtb), s)
        if best:
            (thr, T, use_top, mtb), sv = best
            sig = first_fire(predB, tf, thr, "p6"); sig = sig[sig.split != "train"]
            tr = simulate(sig, tops_cache[use_top], tfb, atr, M, T, use_top, mtb if T else 0)
            out["grid_best"][tf] = {"config": {"thr": thr, "target_atr": T, "top_exit": use_top, "min_target_bp": mtb},
                                    "val": sv, "hold": summarise(tr[tr.split == "hold"], hdays)}
            out["curves"]["%s|v6|best" % tf] = equity_curve(tr)
            print("%-3s BEST %s VAL net12 %s | HOLD %s" % (tf, out["grid_best"][tf]["config"], sv["net_bp_12"],
                  out["grid_best"][tf]["hold"].get("net_bp_12")), flush=True)
        json.dump(out, open(os.path.join(HERE, "results", "bt6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done %.0fs" % (time.time() - t0))
