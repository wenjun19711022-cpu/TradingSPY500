"""v7 = cost-aware model. Instead of "will price reach low + 2 ATR", learn directly "is THIS long trade (enter next 1m open,
stop under the candidate low, target entry + k*R, flat 15:55) profitable after 12 bp?" -- classifier P(net > 0) and regressor E[net].
Features: all v6 features + trade geometry (R in bp / in ATR, cost / R). Train <= 2023 | choose k, model, threshold, timeframe on
VALIDATION 2024-01..2026-03-25 | holdout 2026-03-26..09-30 reported once (second look at that period, first for this model).
Trading sim: first qualifying stage per candidate, one position at a time per timeframe, 1-minute path (v7_outcomes.py)."""
import json, os, pickle, time, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
TFS7 = ["3m", "5m", "15m", "1h"]; K_LIST = (1.0, 1.5, 2.0, 3.0); COST = 12.0
TRAIN_END = pd.Timestamp("2023-12-31"); VAL_END = pd.Timestamp("2026-03-25")
QS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98); MIN_N = {"3m": 150, "5m": 120, "15m": 80, "1h": 40}
GEO = ["R_bp", "R_atr", "cost_R"]


def load():
    sets = json.load(open(os.path.join(HERE, "models", "v6_features.json"), encoding="utf-8"))["sets"]
    feats = list(dict.fromkeys(sets["E_+大周期状态"] + GEO))
    parts = []
    for tf in TFS7:
        S = pd.read_parquet(os.path.join(HERE, "cache", "S_%s_bottom.parquet" % tf))
        O = pd.read_parquet(os.path.join(HERE, "cache", "out7_%s.parquet" % tf))
        X = pd.concat([S.reset_index(drop=True), O.drop(columns=["row"])], axis=1)
        X = X[X.g1.notna()].copy()
        X["R_atr"] = X.R_bp / X.atr_bp; X["cost_R"] = COST / X.R_bp
        parts.append(X)
    P = pd.concat(parts, ignore_index=True)
    P["split"] = np.where(P.date <= TRAIN_END, "train", np.where(P.date <= VAL_END, "val", "hold"))
    for k in K_LIST: P["net%g" % k] = P["g%g" % k] * 1e4 - COST
    return P, feats


def weights(P, mask):
    n = P[mask].groupby("tf").size(); return (1.0 / P.loc[mask, "tf"].map(n)).to_numpy() * mask.sum() / len(n)


def models(P, feats, k):
    tr = (P.split == "train").to_numpy(); X = P.loc[tr, feats].to_numpy(np.float32); w = weights(P, P.split == "train")
    y = (P.loc[tr, "net%g" % k] > 0).astype(int).to_numpy(); yr = P.loc[tr, "net%g" % k].clip(-150, 300).to_numpy()
    kw = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=3.0, max_features=0.5, random_state=0)
    clf = HistGradientBoostingClassifier(**kw).fit(X, y, sample_weight=w)
    reg = HistGradientBoostingRegressor(**kw).fit(X, yr, sample_weight=w)
    return clf, reg


def trade(D, score, thr, k):
    """D: rows of one timeframe & split. Returns DataFrame of executed trades (first qualifying stage, one at a time)."""
    Q = D[score >= thr].sort_values(["bar_j", "c"]).drop_duplicates("cand_i")
    Q = Q.sort_values("k_in"); busy = -1; keep = []
    for ix, ki, xe in zip(Q.index, Q.k_in.to_numpy(), Q["xi%g" % k].to_numpy()):
        if ki <= busy: continue
        keep.append(ix); busy = xe
    return Q.loc[keep]


def stats(T, k, days):
    if len(T) == 0: return {"trades": 0}
    net = T["net%g" % k]; g = T["g%g" % k] * 1e4
    return {"trades": int(len(T)), "per_day": round(len(T) / max(days, 1), 2), "gross_bp": round(float(g.mean()), 2),
            "net_bp_12": round(float(net.mean()), 2), "net_bp_7": round(float(net.mean() + 5), 2), "win_12": round(float((net > 0).mean()), 3),
            "pf_12": round(float(net[net > 0].sum() / max(-net[net <= 0].sum(), 1e-9)), 3), "avg_R_bp": round(float(T.R_bp.mean()), 1),
            "t_stat": round(float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))), 2) if len(net) > 2 else None,
            "acct50_pct": round(float((np.prod(1 + 0.01 * 50 * net / 1e4) - 1) * 100), 2)}


if __name__ == "__main__":
    t0 = time.time(); P, feats = load(); print("rows", len(P), P.groupby(["tf", "split"]).size().unstack().to_dict(), flush=True)
    res = {"cost_bp": COST, "k_list": K_LIST, "configs": {}, "auc": {}, "best": {}}
    scores = {}
    for k in K_LIST:
        clf, reg = models(P, feats, k)
        Xa = P[feats].to_numpy(np.float32); scores[("clf", k)] = clf.predict_proba(Xa)[:, 1]; scores[("reg", k)] = reg.predict(Xa)
        va = (P.split == "val").to_numpy(); y = (P["net%g" % k] > 0).to_numpy()
        res["auc"]["k%g" % k] = {tf: round(roc_auc_score(y[va & (P.tf == tf).to_numpy()], scores[("clf", k)][va & (P.tf == tf).to_numpy()]), 4) for tf in TFS7}
        pickle.dump({"clf": clf, "reg": reg, "feats": feats}, open(os.path.join(HERE, "models", "v7_k%g.pkl" % k), "wb"))
        print("k=%g val AUC(net>0) %s  base win %.3f  %.0fs" % (k, res["auc"]["k%g" % k], y[va].mean(), time.time() - t0), flush=True)
    for tf in TFS7:
        m_tf = (P.tf == tf).to_numpy(); best = None
        V = P[m_tf & (P.split == "val").to_numpy()]; H = P[m_tf & (P.split == "hold").to_numpy()]
        vdays, hdays = V.date.nunique(), H.date.nunique()
        for (kind, k), sc in scores.items():
            sv = sc[m_tf & (P.split == "val").to_numpy()]
            for q in QS:
                thr = float(np.quantile(sv, q)); T = trade(V, sv, thr, k); s = stats(T, k, vdays)
                res["configs"]["%s|%s|k%g|q%g" % (tf, kind, k, q)] = dict(s, thr=thr)
                if s["trades"] >= MIN_N[tf] and (best is None or s["net_bp_12"] > best[1]["net_bp_12"]): best = ((kind, k, q, thr), s)
        (kind, k, q, thr), sv_ = best
        sh = scores[(kind, k)][m_tf & (P.split == "hold").to_numpy()]
        Th = trade(H, sh, thr, k); hs = stats(Th, k, hdays)
        # all trades of the chosen config on val+hold for the equity curve
        Tv = trade(V, scores[(kind, k)][m_tf & (P.split == "val").to_numpy()], thr, k)
        curve = pd.concat([Tv, Th]).sort_values("k_in")
        eq = (1 + 0.01 * 50 * curve["net%g" % k] / 1e4).groupby(curve.date.dt.strftime("%Y-%m-%d")).prod().cumprod()
        res["best"][tf] = {"config": {"model": kind, "k": k, "quantile": q, "thr": thr}, "val": sv_, "hold": hs,
                           "curve": [[d, round(float((v - 1) * 100), 3)] for d, v in eq.items()]}
        print("%-3s BEST %s k=%g q=%.2f | VAL n %d net12 %+.2f t %.2f win %.3f | HOLD n %s net12 %s t %s" % (tf, kind, k, q, sv_["trades"], sv_["net_bp_12"],
              sv_["t_stat"] or 0, sv_["win_12"], hs.get("trades"), hs.get("net_bp_12"), hs.get("t_stat")), flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "v7.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done %.0fs" % (time.time() - t0))
