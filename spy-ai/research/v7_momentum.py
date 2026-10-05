"""Alternative edge: intraday momentum (Gao, Han, Li & Zhou 2018, JFE "Market intraday momentum").
The first half hour's return (previous close -> 10:00, i.e. incl. the overnight gap) and the 12th half hour (15:00 -> 15:30)
predict the LAST half hour (15:30 -> close). Long only, intraday (enter 15:30, exit 15:55 or the 16:00 close), OKX costs.
Three variants fixed BEFORE looking: A first-half-hour > 0 | B 12th half hour > 0 | C both > 0. Plus an always-long baseline.
Pick on validation 2024-01..2026-03-25, holdout 2026-03-26..09-30 once; 2018-10..2023 shown as in-sample history."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_END = pd.Timestamp("2023-12-31"); VAL_END = pd.Timestamp("2026-03-25")

b = pd.read_parquet(os.path.join(HERE, "data", "spy_1m.parquet"))
b["d"] = pd.to_datetime(b.t.str[:10]); b["em"] = b.t.str[11:13].astype(int) * 60 + b.t.str[14:16].astype(int)
g = b.groupby("d")
def at(em, col="c"):                       # value of the 1m bar ENDING at minute em (e.g. 600 = 10:00)
    x = b[b.em == em].set_index("d")[col]; return x
D = pd.DataFrame({"close": g.c.last(), "full": g.em.max() == 960})
D["prev_close"] = D.close.shift(1)
D["c1000"] = at(600); D["c1500"] = at(900); D["o1530"] = at(931, "o"); D["c1530"] = at(930); D["c1555"] = at(955)
D = D[D.full & D.prev_close.notna()].dropna()
D["r1"] = D.c1000 / D.prev_close - 1; D["r12"] = D.c1530 / D.c1500 - 1
D["last_55"] = D.c1555 / D.o1530 - 1; D["last_close"] = D.close / D.o1530 - 1        # enter at the 15:30 open of the next bar
D["split"] = np.where(D.index <= TRAIN_END, "train", np.where(D.index <= VAL_END, "val", "hold"))
rules = {"A 前半小时涨": D.r1 > 0, "B 15:00-15:30 涨": D.r12 > 0, "C 两者都涨": (D.r1 > 0) & (D.r12 > 0), "基准 每天都买": D.r1 == D.r1}
out = {}
for name, sig in rules.items():
    for ex in ("last_55", "last_close"):
        for sp in ("train", "val", "hold"):
            r = D.loc[sig & (D.split == sp), ex] * 1e4
            out.setdefault(name, {}).setdefault(ex, {})[sp] = {"days": int(len(r)), "gross_bp": round(float(r.mean()), 2),
                "net12": round(float(r.mean() - 12), 2), "net7": round(float(r.mean() - 7), 2), "win_gross": round(float((r > 0).mean()), 3),
                "t_gross": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 else None}
for name, v in out.items():
    for ex, s in v.items():
        print("%-14s %-10s " % (name, ex) + " | ".join("%s n %4d gross %+6.2f t %+5.2f net12 %+6.2f" % (sp, s[sp]["days"], s[sp]["gross_bp"], s[sp]["t_gross"] or 0, s[sp]["net12"]) for sp in ("train", "val", "hold")))
# magnitude check: does a bigger first-half-hour move predict a bigger last-half-hour move? (deciles, all years)
D["dec"] = pd.qcut(D.r1, 10, labels=False)
print("\nlast-30min gross bp by decile of first-30min return (all years):"); print((D.groupby("dec").last_close.mean() * 1e4).round(2).to_dict())
json.dump(out, open(os.path.join(HERE, "results", "momentum.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
