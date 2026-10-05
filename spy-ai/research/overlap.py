"""How well do real-time signals overlap the user's drawn pivots (2026-08-05..10-05, 1h ZigZag 0.45% reconstruction)?
For every drawn bottom/top: did a same-side signal fire within [pivot-2h, pivot+4h], and how far (in %) from the pivot price?
Also precision: share of signals in the window that sit near a drawn pivot. Sources:
  v6     live engine signals (15m / 1h, p >= 55%) — the watcher's v6 (replay of the holdout period + case days)
  v5     the moomoo-portable formula (v5core = the 副图 formula), 15m and 1h bars
  确认   zigzag confirmation (real time, by construction lagging)"""
import os, sys
import numpy as np, pandas as pd
import user_lines as UL, swing_study as S
from zigfit import zigzag
sys.path.insert(0, "C:/Users/94868/spybt/watch"); import v5core

HERE = os.path.dirname(os.path.abspath(__file__))
W0, W1 = "2026-08-05", "2026-10-05 23:59"


def pivots():
    b = UL.raw_1h(); w = b[(b.t >= W0) & (b.t <= W1)].reset_index(drop=True)
    z = zigzag(w.h.to_numpy(float), w.l.to_numpy(float), 0.0045)
    T = pd.to_datetime(w.t)
    P = pd.DataFrame([{"kind": k, "t": T[i], "px": p, "conf_t": T[c], "conf_px": float(w.c[c])} for i, k, p, c in z])
    # signals are on forward-adjusted bars; the drawing is unadjusted -> express pivots in forward-adjusted prices
    # (SPY went ex-dividend 2026-09-18: factor 0.997523 on everything before it, 1 after)
    f = lambda t: 0.997523 if t < pd.Timestamp("2026-09-18") else 1.0
    P["px"] = [p * f(t) for p, t in zip(P.px, P.t)]; P["conf_px"] = [p * f(t) for p, t in zip(P.conf_px, P.conf_t)]
    return P


def v5_signals(tf):
    b = pd.read_parquet(os.path.join(HERE, "data", "spy_%s.parquet" % tf))
    o, h, l, c = (b[k].to_numpy(float) for k in ("o", "h", "l", "c")); t = pd.to_datetime(b.t)
    out = {}
    for side in ("bottom", "top"):
        r = v5core.side_signals(o, h, l, c, side, 55.0)
        m = r["fire"] & (t >= W0).to_numpy()
        out[side] = pd.DataFrame({"t": t[m].values, "px": c[m]})
    return out


def score(P, sig, name):
    rows = []
    for kind, side in (("B", "bottom"), ("T", "top")):
        piv = P[P.kind == kind]; s = sig[side]; s = s[(s.t >= W0) & (s.t <= W1)]
        hits = []
        for p in piv.itertuples():
            m = s[(s.t >= p.t - pd.Timedelta(hours=2)) & (s.t <= p.t + pd.Timedelta(hours=4))]
            if len(m):
                d = (m.px / p.px - 1) * 100 * (1 if kind == "B" else -1)      # how far above a bottom / below a top
                hits.append(float(d.min()))
        near = 0
        for q in s.itertuples():
            pp = piv[(piv.t >= q.t - pd.Timedelta(hours=4)) & (piv.t <= q.t + pd.Timedelta(hours=2))]
            near += len(pp) > 0
        rows.append({"源": name, "方向": "底" if kind == "B" else "顶", "你画的点": len(piv), "被提示到": len(hits),
                     "召回率": round(len(hits) / max(len(piv), 1), 2), "离你的点(中位%)": round(float(np.median(hits)), 2) if hits else None,
                     "信号数": len(s), "准确率(靠近你的点)": round(near / max(len(s), 1), 2)})
    return rows


if __name__ == "__main__":
    P = pivots()
    v6 = {}
    for side in ("bottom", "top"):
        ts = np.sort(np.r_[S.v6_signals(side, "15m"), S.v6_signals(side, "1h")])
        b1 = pd.read_parquet(os.path.join(HERE, "data", "spy_1m.parquet"), columns=["t", "c"]); b1["T"] = pd.to_datetime(b1.t)
        px = b1.set_index("T").c.reindex(pd.to_datetime(ts)).to_numpy()
        v6[side] = pd.DataFrame({"t": pd.to_datetime(ts), "px": px})
    # v6/v5 prices are forward-adjusted; the drawing is unadjusted -> compare in % after shifting pre-2026-09-18 prices by the dividend factor
    rows = score(P, v6, "v6 实时（15m+1h）")
    v5 = {k: pd.concat([v5_signals("15m")[k], v5_signals("1h")[k]]).sort_values("t") for k in ("bottom", "top")}
    rows += score(P, v5, "v5 moomoo 公式（15m+1h）")
    conf = {"bottom": P[P.kind == "B"].rename(columns={"conf_t": "tt"})[["tt", "conf_px"]].rename(columns={"tt": "t", "conf_px": "px"}),
            "top": P[P.kind == "T"].rename(columns={"conf_t": "tt"})[["tt", "conf_px"]].rename(columns={"tt": "t", "conf_px": "px"})}
    rows += score(P, conf, "折线确认 0.45%")
    R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.to_string(index=False))
    R.to_csv(os.path.join(HERE, "results", "overlap.csv"), index=False, encoding="utf-8-sig")
    P.to_csv(os.path.join(HERE, "data", "user_pivots_zz.csv"), index=False)
