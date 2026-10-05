"""Print a compact view of results/train6.json (+ bt6.json if present)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TFS = ["1m", "3m", "5m", "15m", "1h"]
r = json.load(open(os.path.join(HERE, "results", "train6.json"), encoding="utf-8"))
for side, R in r["sides"].items():
    ch = R["selection"]["chosen"]
    print("\n==== %s  chosen: %s  (n_train %d)" % (side, ch, R["n_train"]))
    print("tf    base  cand/d | v5 auc  v6 auc | v5 prec@.55 rec  /d  | v6 matched prec rec | v6 @.55 prec rec /d | HOLD v5 auc v6 auc  v5 prec  v6 matched")
    for tf in TFS:
        a, b, ha, hb = R["v5重训"][tf], R[ch][tf], R["holdout_v5重训"][tf], R["holdout_v6"][tf]
        print("%-4s %5.3f %6.1f | %.3f  %.3f | %.3f %.3f %5.2f | %.3f %.3f | %.3f %.3f %5.2f | %.3f %.3f  %.3f  %.3f" % (
            tf, a["base_rate"], a["cands_per_day"], a["auc_c0"], b["auc_c0"], a["thr0.55"]["precision"], a["thr0.55"]["recall"], a["thr0.55"]["per_day"],
            b["matched"]["precision"], b["matched"]["recall"], b["thr0.55"]["precision"], b["thr0.55"]["recall"], b["thr0.55"]["per_day"],
            ha["auc_c0"], hb["auc_c0"], ha["thr0.55"]["precision"] or 0, hb["matched"]["precision"] or 0))
    print("importance:", list(R["importance"].items())[:14])
    print("tail:", list(R["importance"].items())[-6:])
    print("auc by year v6", R["auc_by_year_val"], "v5", R["auc_by_year_val_v5"])
    print("ladder mean auc:", {k: round(sum(R["GBM " + k][tf]["auc_c0"] for tf in TFS) / 5, 4) for k in
          ["A_v5基础", "B_+价格类指标", "C_+量能类指标", "D_+日内位置/VIX", "E_+大周期状态"]})
p = os.path.join(HERE, "results", "bt6.json")
if os.path.exists(p) and "--bt" in sys.argv:
    b = json.load(open(p, encoding="utf-8"))
    print("\n==== trading default (val | hold)")
    for k, v in b["default"].items():
        f = lambda s: "n %5s /d %5s gross %6s net12 %6s net7 %6s win %5s pf %5s liq50 %s acct50 %s dd50 %s" % (
            s.get("trades"), s.get("per_day"), s.get("gross_bp"), s.get("net_bp_12"), s.get("net_bp_7"), s.get("win_12"), s.get("pf_12"),
            (s.get("x50") or {}).get("liquidations"), (s.get("x50") or {}).get("acct_total_pct"), (s.get("x50") or {}).get("max_dd_pct"))
        print("%-12s VAL %s\n%-12s HLD %s" % (k, f(v["val"]), "", f(v["hold"])))
    print("\n==== grid best")
    for tf, g in b["grid_best"].items():
        print(tf, g["config"], "VAL net12", g["val"]["net_bp_12"], "n", g["val"]["trades"], "| HOLD net12", g["hold"].get("net_bp_12"), "net7", g["hold"].get("net_bp_7"),
              "n", g["hold"].get("trades"), "acct50", (g["hold"].get("x50") or {}).get("acct_total_pct"))
