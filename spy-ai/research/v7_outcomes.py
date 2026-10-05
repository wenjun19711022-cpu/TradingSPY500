"""v7 step 1: for EVERY bottom-candidate stage row (3m/5m/15m/1h), simulate the real long trade on the 1-minute path:
entry = next 1m open after the decision bar closes, stop = candidate low - 0.05 ATR, R = entry - stop,
targets T = entry + k*R for k in K_LIST, exit at first stop / target touch (stop first inside a bar) or 15:55 ET close.
Saves cache/out7_<tf>.parquet: row index into S cache, entry, R_bp, gross return per k, time-exit flag per k."""
import os, time
import numpy as np, pandas as pd
import lib6, feat6, bt6

HERE = os.path.dirname(os.path.abspath(__file__))
TFS7 = ["3m", "5m", "15m", "1h"]; K_LIST = (1.0, 1.5, 2.0, 3.0)


def outcomes(tf, M):
    S = pd.read_parquet(os.path.join(HERE, "cache", "S_%s_bottom.parquet" % tf), columns=["cand_i", "bar_j", "c", "date"])
    b = lib6.load(tf); h, l, c = (b[k].to_numpy(float) for k in ("h", "l", "c"))
    atr = feat6._atr(h, l, c, 14); tend = pd.to_datetime(b.t).values; one = np.timedelta64(1, "m")
    n = len(S); E = np.full(n, np.nan); RB = np.full(n, np.nan); G = np.full((n, len(K_LIST)), np.nan); TX = np.zeros((n, len(K_LIST)), bool); XI = np.full((n, len(K_LIST)), -1, np.int64); KI = np.full(n, -1, np.int64)
    MAE = np.full(n, np.nan); MFE = np.full(n, np.nan)
    ks = np.searchsorted(M["t"], tend[S.bar_j.to_numpy()] + one)
    for r, (ci, k) in enumerate(zip(S.cand_i.to_numpy(), ks)):
        if k >= len(M["t"]) or M["t"][k] != tend[S.bar_j.iat[r]] + one or M["em"][k] > 950: continue
        kend = M["kend"][k]
        if kend < k: continue
        e = M["o"][k]; stop = l[ci] - 0.05 * atr[ci]
        if e <= stop: continue
        R = e - stop; lo = M["l"][k:kend + 1]; hi = M["h"][k:kend + 1]; op = M["o"][k:kend + 1]
        js = np.flatnonzero(lo <= stop); js = js[0] if len(js) else 10 ** 9
        E[r] = e; RB[r] = R / e * 1e4; KI[r] = k
        upto = min(js, kend - k); MFE[r] = (hi[:upto + 1].max() - e) / e * 1e4; MAE[r] = (lo[:upto + 1].min() - e) / e * 1e4
        for q, kk in enumerate(K_LIST):
            tg = e + kk * R; jt = np.flatnonzero(hi >= tg); jt = jt[0] if len(jt) else 10 ** 9
            if js <= jt and js < 10 ** 9: x = min(stop, op[js]); XI[r, q] = k + js
            elif jt < 10 ** 9: x = max(tg, op[jt]); XI[r, q] = k + jt
            else: x = M["c"][kend]; TX[r, q] = True; XI[r, q] = kend
            G[r, q] = (x - e) / e
    out = pd.DataFrame({"row": np.arange(n), "k_in": KI, "entry": E, "R_bp": RB, "mfe_bp": MFE, "mae_bp": MAE})
    for q, kk in enumerate(K_LIST): out["g%g" % kk] = G[:, q]; out["tx%g" % kk] = TX[:, q]; out["xi%g" % kk] = XI[:, q]
    return out


if __name__ == "__main__":
    t0 = time.time(); M = bt6.load_1m()
    for tf in TFS7:
        o = outcomes(tf, M); o.to_parquet(os.path.join(HERE, "cache", "out7_%s.parquet" % tf), index=False)
        v = o.dropna(subset=["g1"])
        print("%-3s rows %7d tradable %7d | R median %.1fbp | mean gross bp k1 %.2f k1.5 %.2f k2 %.2f k3 %.2f  %.0fs" % (
            tf, len(o), len(v), v.R_bp.median(), *(v["g%g" % k].mean() * 1e4 for k in K_LIST), time.time() - t0), flush=True)
