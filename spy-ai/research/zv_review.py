"""Data for the dashboard tab '你的折线': the user's drawing (1h, unadjusted) vs 折线AI marks vs the swing rule -> results/zigzag_review.json"""
import json, os, sys
import numpy as np, pandas as pd
import overlap as O, user_lines as UL, zv_formula as Z

HERE = os.path.dirname(os.path.abspath(__file__)); F = 0.997523          # forward-adjusted / unadjusted before 2026-09-18

if __name__ == "__main__":
    b = UL.raw_1h(); w = b[(b.t >= "2026-08-05") & (b.t <= "2026-10-05 23:59")]
    bars = [[t[5:16], round(o, 2), round(h, 2), round(l, 2), round(c, 2)] for t, o, h, l, c in zip(w.t, w.o, w.h, w.l, w.c)]
    P = O.pivots()
    unadj = lambda p, t: p / F if t < pd.Timestamp("2026-09-18") else p
    piv = [[str(r.t)[5:16], r.kind, round(unadj(r.px, r.t), 2)] for r in P.itertuples()]
    b15 = pd.read_parquet(os.path.join(HERE, "data", "spy_15m.parquet")); buy, sell = Z.marks(b15, 56, 0.3)
    t15 = pd.to_datetime(b15.t); m = (t15 >= "2026-08-05").to_numpy()
    mk = [[str(t)[5:16], "买", round(unadj(p, t), 2)] for t, p in zip(t15[m & buy], b15.l.to_numpy()[m & buy])] + \
         [[str(t)[5:16], "卖", round(unadj(p, t), 2)] for t, p in zip(t15[m & sell], b15.h.to_numpy()[m & sell])]
    fz = json.load(open(os.path.join(HERE, "results", "zv_formula.json"), encoding="utf-8"))
    zv = json.load(open(os.path.join(HERE, "results", "zv.json"), encoding="utf-8"))
    s1 = {r["name"]: {k: r[k] for k in ("val", "hold")} for r in zv["results"] if r["name"].startswith("S1")}
    best2 = next(r for r in zv["results"] if r["name"] == zv["best"])
    out = {"bars": bars, "pivots": piv, "marks": sorted(mk), "overlap": pd.read_csv(os.path.join(HERE, "results", "overlap.csv")).to_dict("records"),
           "formula": {tf: {k: v for k, v in d.items() if k != "fit"} for tf, d in fz["best"].items()},
           "s1": s1, "s2_best": {"name": best2["name"], "val": best2["val"], "hold": best2["hold"]},
           "theta": 0.45, "drawn_vertices": len(UL.V)}
    json.dump(out, open(os.path.join(HERE, "results", "zigzag_review.json"), "w", encoding="utf-8"), ensure_ascii=False, default=str)
    print("bars", len(bars), "pivots", len(piv), "marks", len(mk))
