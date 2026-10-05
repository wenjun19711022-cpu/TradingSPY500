"""Which moomoo indicators matter? Train-only regularised GBM (all features), permutation importance on the VALIDATION
period, first look at the candidate (c = 0), EVERY timeframe weighted equally (1m would otherwise swamp the rest).
Score = mean over timeframes of the AUC drop when one indicator family is shuffled. Writes into results/train6.json."""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import lib6, train6
from models6 import G

HERE = os.path.dirname(os.path.abspath(__file__)); TFS = lib6.TFS; rng = np.random.default_rng(1)
res = json.load(open(os.path.join(HERE, "results", "train6.json"), encoding="utf-8"))
b0 = lib6.load("1h"); lib6.build("1h", "bottom", b=b0)                 # fills lib6.IND
sets = lib6.feature_sets(); fams = train6.families()
for side in ("bottom", "top"):
    P = pd.concat([pd.read_parquet(os.path.join(HERE, "cache", "S_%s_%s.parquet" % (tf, side))) for tf in TFS], ignore_index=True)
    tr = P.date <= lib6.TRAIN_END; va = (P.date > lib6.TRAIN_END) & (P.date <= lib6.VAL_END)
    T = train6.balanced(P, tr); m = G(sets["E_+大周期状态"], reg=True).fit(T, T.label)
    V = {tf: P[va & (P.tf == tf) & (P.c == 0)].sample(frac=1, random_state=0).head(25000).reset_index(drop=True) for tf in TFS}
    base = {tf: roc_auc_score(V[tf].label, m.predict_proba(V[tf])[:, 1]) for tf in TFS}
    out = {}
    for g, cols in fams.items():
        cols = [c for c in cols if c in m.cols]
        if not cols: continue
        d = []
        for tf in TFS:
            W = V[tf].copy(); W[cols] = V[tf][cols].to_numpy()[rng.permutation(len(W))]
            d.append(base[tf] - roc_auc_score(W.label, m.predict_proba(W)[:, 1]))
        out[g] = {"mean": round(float(np.mean(d)), 5), "by_tf": dict(zip(TFS, [round(float(x), 5) for x in d]))}
        print(side, g, out[g]["mean"], flush=True)
    res["sides"][side]["importance"] = {k: v["mean"] for k, v in sorted(out.items(), key=lambda kv: -kv[1]["mean"])}
    res["sides"][side]["importance_by_tf"] = {k: v["by_tf"] for k, v in out.items()}
    res["sides"][side]["importance_base_auc"] = {tf: round(float(x), 4) for tf, x in base.items()}
    res["sides"][side]["importance_note"] = "训练期模型；验证集；只看候选第一根（c=0）；5 个周期等权平均的 AUC 下降"
json.dump(res, open(os.path.join(HERE, "results", "train6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved")
