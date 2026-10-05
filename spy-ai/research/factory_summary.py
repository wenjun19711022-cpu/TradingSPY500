"""Compact summary of the strategy factory for the dashboard: results/factory_summary.json"""
import json, math, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "results", "factory.json"), encoding="utf-8"))
T = pd.read_pickle(os.path.join(HERE, "cache", "factory_trades.pkl"))
R = F["results"]
fams = {}
for r in R:
    f = fams.setdefault(r["family"], {"configs": 0, "s1": 0, "s2": 0, "s3": 0, "best_net": None, "best_gross": None})
    f["configs"] += 1; f["s1"] += r["stage1"]; f["s2"] += r["stage2"]; f["s3"] += r["stage3"]
    tr = r["train"]
    if tr.get("n", 0) >= 30 and (f["best_net"] is None or tr["net12"] > f["best_net"]["train"]["net12"]): f["best_net"] = r
    if tr.get("n", 0) >= 30 and (f["best_gross"] is None or tr["gross"] > f["best_gross"]["train"]["gross"]): f["best_gross"] = r
keep = lambda r: {"name": r["name"], **{sp: {k: r[sp].get(k) for k in ("n", "gross", "net12", "net7", "t", "win")} for sp in ("train", "val", "hold")}}
fam_out = {k: {"configs": v["configs"], "s1": v["s1"], "s2": v["s2"], "s3": v["s3"], "best_net": keep(v["best_net"]), "best_gross": keep(v["best_gross"])}
           for k, v in fams.items() if v["best_net"]}
top = sorted([r for r in R if r["train"].get("n", 0) >= 60], key=lambda r: r["train"]["net12"], reverse=True)[:15]

star = "15分钟 突破26根高点 · 量比≥1.0 · 止损1.5ATR · 2ATR移动止损"
t = T[star].copy(); t["y"] = t.d.str[:4]; oos = t[t.split != "train"]
yearly = [[y, int(len(g)), round(float(g.gross.mean()), 2)] for y, g in t.groupby("y")]
costs = [0, 0.5, 1, 2, 3, 4, 5, 7, 9, 12]
sens = [[c, round(float((t.gross - c - t.fund).mean()), 2), round(float((oos.gross - c - oos.fund).mean()), 2)] for c in costs]
out = {"n_configs": F["n_configs"], "bonferroni_t": F["bonferroni_t"], "families": fam_out, "top": [keep(r) | {"family": r["family"]} for r in top],
       "star": {"name": star, "yearly": yearly, "cost_sens": sens, "oos_n": int(len(oos)), "oos_gross": round(float(oos.gross.mean()), 2),
                "oos_t": round(float(oos.gross.mean() / (oos.gross.std() / math.sqrt(len(oos)))), 2), "per_day": round(len(t) / 2015, 2),
                "hold_min": round(float(t.mins.mean())), "stop_share": round(float((t.why == "stop").mean()), 3)}}
json.dump(out, open(os.path.join(HERE, "results", "factory_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(fam_out), "families;", out["star"]["oos_gross"], out["star"]["oos_t"])
