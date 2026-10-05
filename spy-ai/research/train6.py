"""v6 training + honest evaluation.
  train  <= 2023-12-31 | validation 2024-01-01 .. 2026-03-25 (all choices made here) | holdout 2026-03-26 .. 2026-09-30 (looked at once)
Models: v5 architecture retrained on moomoo data (logistic, 18 features) vs gradient-boosted trees on an ablation ladder
(A v5 features -> B + price indicators -> C + volume indicators -> D + day/VIX context -> E + 15m/1h state) and a v6 logistic.
Outputs: cache/S_<tf>_<side>.parquet, cache/pred_<side>.parquet (walk-forward probabilities), results/train6.json, models/v6_<side>.pkl"""
import json, os, pickle, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lib6, ind, vixdata

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("cache", "results", "models"): os.makedirs(os.path.join(HERE, d), exist_ok=True)
TFS = os.environ.get("V6_TFS", ",".join(lib6.TFS)).split(","); CAP = 80000; SEED = 0


from models6 import gbm, LR, G, Ens, make


def balanced(P, mask):
    """equal weight per timeframe: subsample each TF to <= CAP rows"""
    idx = []
    for tf in TFS:
        ii = P.index[mask & (P.tf == tf)]
        idx.append(np.random.default_rng(SEED).choice(ii, min(len(ii), CAP), replace=False))
    return P.loc[np.sort(np.concatenate(idx))]


def cand_table(V, p):
    """one row per candidate: first stage where p >= thr is what fires; for ranking use max p over live stages."""
    X = V[["tf", "cand_i", "c", "label", "date"]].copy(); X["p"] = p
    g = X.groupby(["tf", "cand_i"]); return g.agg(label=("label", "first"), pmax=("p", "max"), date=("date", "first")).reset_index()


def fire_stats(C, thr, days):
    F = C[C.pmax >= thr]; ntrue = int(C.label.sum())
    return {"precision": round(float(F.label.mean()), 3) if len(F) else None, "recall": round(float(F.label.sum() / max(ntrue, 1)), 3),
            "per_day": round(len(F) / days, 2), "n": int(len(F))}


def matched(C, k):
    """precision / recall when taking the k highest-scoring candidates (same signal count as the baseline)."""
    if k <= 0: return {"precision": None, "recall": 0.0}
    T = C.sort_values("pmax", ascending=False).head(k)
    return {"precision": round(float(T.label.mean()), 3), "recall": round(float(T.label.sum() / max(C.label.sum(), 1)), 3)}


def evaluate(P, p, mask, base_counts=None):
    out = {}
    for tf in TFS:
        m = (mask & (P.tf == tf)).to_numpy(); V = P[m]; pv = p[m]; days = V.date.nunique()
        if len(V) == 0: continue
        c0 = (V.c == 0).to_numpy()
        C = cand_table(V, pv)
        r = {"auc_c0": round(roc_auc_score(V.label[c0], pv[c0]), 4), "auc_all": round(roc_auc_score(V.label, pv), 4),
             "base_rate": round(float(C.label.mean()), 3), "cands_per_day": round(len(C) / days, 2), "days": int(days)}
        for thr in (0.45, 0.55, 0.65): r["thr%.2f" % thr] = fire_stats(C, thr, days)
        if base_counts is not None: r["matched"] = matched(C, base_counts.get(tf, 0))
        out[tf] = r
    return out


def group_importance(model, P, mask, groups, rng):
    """drop in all-stage AUC when a whole indicator family is shuffled (validation rows)."""
    V = P[mask].reset_index(drop=True); y = V.label.to_numpy(); base = roc_auc_score(y, model.predict_proba(V)[:, 1]); res = {}
    for gname, cols in groups.items():
        cols = [c for c in cols if c in model.cols]
        if not cols: continue
        W = V.copy(); perm = rng.permutation(len(W))
        W[cols] = V[cols].to_numpy()[perm]                       # shuffle rows jointly for the family
        res[gname] = round(float(base - roc_auc_score(y, model.predict_proba(W)[:, 1])), 5)
    return dict(sorted(res.items(), key=lambda kv: -kv[1])), round(float(base), 4)


def families():
    g = {"v5 形态(跌幅/影线/RSI/均线距)": [c for c in lib6.V5 if c not in ("c1", "c2", "c3", "bounce", "reclaim", "clv_now")],
         "阶段(反弹/收复)": ["c1", "c2", "c3", "bounce", "reclaim", "clv_now"]}
    for k, fam in ind.FAMILY.items():
        if k in lib6.IND: g.setdefault(fam, []).append(k)
    g["日内位置(开盘/昨高低/时段)"] = ["daymove", "day_rng", "gap", "pdl_d", "pdh_d", "pdc_d", "tod", "dow", "regime_up"]
    g["VIX(前收/期限结构)"] = ["vix", "vix_term", "vix_9d"]
    g["15分钟大周期"] = [c for c in lib6.HTF if c.startswith("h15")]
    g["1小时大周期"] = [c for c in lib6.HTF if c.startswith("h60")]
    return g


def calib(y, p, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1); out = []
    for b in range(bins):
        m = q == b
        if m.sum() >= 30: out.append([round((b + 0.5) / bins, 2), round(float(y[m].mean()), 3), int(m.sum())])
    return out


if __name__ == "__main__":
    t0 = time.time()
    vix_prev = vixdata.prev_close_table(pd.read_parquet(os.path.join(HERE, "data", "vix_daily.parquet")))
    htf = {"h15": lib6.load("15m"), "h60": lib6.load("1h")}
    parts = {"bottom": [], "top": []}
    for tf in TFS:
        b = lib6.load(tf); BF = lib6.bar_features(b, vix_prev, htf)
        for side in ("bottom", "top"):
            S, _, _ = lib6.build(tf, side, b=b, vix_prev=vix_prev, htf_bars=htf, bars_feat=BF)
            S.to_parquet(os.path.join(HERE, "cache", "S_%s_%s.parquet" % (tf, side)), index=False)
            parts[side].append(S)
            print("built %-3s %-6s rows %7d cands %6d base %.3f  %.0fs" % (tf, side, len(S), (S.c == 0).sum(), S[S.c == 0].label.mean(), time.time() - t0), flush=True)
    sets = lib6.feature_sets(); res = {"feature_sets": {k: len(v) for k, v in sets.items()}, "sides": {}}
    json.dump({"IND": lib6.IND, "DAY": lib6.DAY, "HTF": lib6.HTF, "V5": lib6.V5, "FAMILY": ind.FAMILY, "sets": sets},
              open(os.path.join(HERE, "models", "v6_features.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rng = np.random.default_rng(SEED)
    for side in ("bottom", "top"):
        P = pd.concat(parts[side], ignore_index=True); parts[side] = None
        tr = P.date <= lib6.TRAIN_END; va = (P.date > lib6.TRAIN_END) & (P.date <= lib6.VAL_END); ho = P.date > lib6.VAL_END
        T = balanced(P, tr); R = {"n_train": int(len(T))}
        # baseline: v5 architecture retrained on the same data
        base = LR(lib6.V5).fit(T, T.label); pb = base.predict_proba(P)[:, 1]
        R["v5重训"] = evaluate(P, pb, va)
        base_counts = {tf: R["v5重训"][tf]["thr0.55"]["n"] for tf in TFS if tf in R["v5重训"]}
        R["v5重训"] = evaluate(P, pb, va, base_counts)
        ladder = {}
        for name, cols in sets.items():                   # ablation ladder (regularised GBM)
            m = G(cols, reg=True).fit(T, T.label); pp = m.predict_proba(P)[:, 1]
            R["GBM " + name] = evaluate(P, pp, va, base_counts); ladder[name] = (m, pp)
            print(side, "GBM", name, "val auc_c0", {tf: R["GBM " + name][tf]["auc_c0"] for tf in TFS}, "%.0fs" % (time.time() - t0), flush=True)
        print(side, "v5重训 val auc_c0", {tf: R["v5重训"][tf]["auc_c0"] for tf in TFS}, flush=True)
        cands = {"v5重训": (("lr", "v5"), pb), "GBM E_+大周期状态": (("gbmr", "E_+大周期状态"), ladder["E_+大周期状态"][1])}
        for name, spec in (("v6-LR(全特征)", ("lr", "E_+大周期状态")), ("GBM-默认参数(全特征)", ("gbm", "E_+大周期状态")),
                           ("集成LR+GBM(全特征)", ("ens", "E_+大周期状态")), ("集成LR+GBM(量价+日内)", ("ens", "D_+日内位置/VIX"))):
            m = make(spec, sets).fit(T, T.label); pp = m.predict_proba(P)[:, 1]
            R[name] = evaluate(P, pp, va, base_counts); cands[name] = (spec, pp)
            print(side, name, "val auc_c0", {tf: R[name][tf]["auc_c0"] for tf in TFS}, "%.0fs" % (time.time() - t0), flush=True)
        score = {n: (round(float(np.mean([R[n][tf]["matched"]["precision"] or 0 for tf in TFS])), 4),
                     round(float(np.mean([R[n][tf]["auc_c0"] for tf in TFS])), 4)) for n in cands}
        # rule (fixed before the full run): highest mean AUC -- the stabler statistic -- among models whose precision at the
        # v5 signal count is not worse than v5's by more than 0.005; otherwise stay with the v5 architecture
        ok = [n for n in cands if score[n][0] >= score["v5重训"][0] - 0.005]
        chosen = max(ok, key=lambda n: score[n][1]); R["selection"] = {"score": score, "chosen": chosen,
                                                                     "rule": "AUC最高、且同等信号数下准确率不低于v5重训0.5个百分点以上"}
        print(side, "SELECTION", score, "->", chosen, flush=True)
        spec, pE = cands[chosen]; final_name = chosen
        mE = ladder["E_+大周期状态"][0]
        R["importance"], R["importance_base_auc"] = group_importance(mE, P, va, families(), rng)
        R["calibration_val"] = calib(P.label[va].to_numpy(), pE[va.to_numpy()])
        R["calibration_val_v5"] = calib(P.label[va].to_numpy(), pb[va.to_numpy()])
        R["auc_by_year_val"] = {str(y): round(roc_auc_score(P.label[va & (P.date.dt.year == y)], pE[(va & (P.date.dt.year == y)).to_numpy()]), 4)
                                for y in sorted(P.date[va].dt.year.unique())}
        R["auc_by_year_val_v5"] = {str(y): round(roc_auc_score(P.label[va & (P.date.dt.year == y)], pb[(va & (P.date.dt.year == y)).to_numpy()]), 4)
                                   for y in sorted(P.date[va].dt.year.unique())}
        # final models: refit on train + validation, score the holdout ONCE
        TV = balanced(P, tr | va)
        fin = make(spec, sets).fit(TV, TV.label); fin5 = LR(lib6.V5).fit(TV, TV.label)
        ph = fin.predict_proba(P)[:, 1]; ph5 = fin5.predict_proba(P)[:, 1]
        hb = {tf: int((cand_table(P[(ho & (P.tf == tf)).to_numpy()], ph5[(ho & (P.tf == tf)).to_numpy()]).pmax >= 0.55).sum()) for tf in TFS}
        R["holdout_v6"] = evaluate(P, ph, ho, hb); R["holdout_v5重训"] = evaluate(P, ph5, ho, hb)
        R["calibration_holdout"] = calib(P.label[ho].to_numpy(), ph[ho.to_numpy()])
        pickle.dump({"v6": fin, "v6_name": chosen, "v6_spec": spec, "v5": fin5}, open(os.path.join(HERE, "models", "v6_%s.pkl" % side), "wb"))
        # walk-forward probabilities for the trading backtest: validation from train-only models, holdout from final models
        pred = P[["tf", "cand_i", "bar_j", "c", "label", "fb", "fh", "date", "bounce"]].copy()
        pred["p6"] = np.where(ho, ph, pE); pred["p5"] = np.where(ho, ph5, pb)
        pred["split"] = np.where(tr, "train", np.where(va, "val", "hold"))
        pred.to_parquet(os.path.join(HERE, "cache", "pred_%s.parquet" % side), index=False)
        res["sides"][side] = R
        json.dump(res, open(os.path.join(HERE, "results", "train6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(side, "done  %.0fs" % (time.time() - t0), flush=True)
    print("ALL DONE %.0fs" % (time.time() - t0))
