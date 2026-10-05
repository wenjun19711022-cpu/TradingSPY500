"""SPY 底顶AI v5 core -- identical math to the moomoo 副图 formula (standalone, numpy only)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "v5_model.json"), encoding="utf-8"))
BASE = ["drop_atr", "mom10", "rng_atr", "clv", "lwick", "body", "rsi14", "rsi3", "ndown", "ema50d", "ema200d", "atr_ratio"]
WARMUP = 300


def ema(x, n):                                    # moomoo EMA: Y0 = X0, Y = (2X + (N-1)Y')/(N+1)
    x = np.nan_to_num(np.asarray(x, float), nan=0.0); out = np.empty(len(x)); a = 2.0 / (n + 1)
    if len(x) == 0: return out
    out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ref(x, k):
    if k == 0: return x.copy()
    out = np.empty_like(x); out[:k] = x[0]; out[k:] = x[:-k]; return out


def rolling_max(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)): out[i] = x[i - n + 1:i + 1].max()
    return out


def lowest_prev(x, n):                            # REF(LLV(X,n),1)
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)): out[i] = x[i - n:i].min()
    return out


def barslast(cond):
    out = np.full(len(cond), 1e6); last = None
    for i, v in enumerate(cond):
        if v: last = i
        out[i] = (i - last) if last is not None else 1e6
    return out


def ref_var(x, n):
    idx = np.arange(len(x)) - np.minimum(n, np.arange(len(x))).astype(int)
    return x[idx]


def side_signals(o, h, l, c, side, TH):
    """Returns dict of arrays: p (probability %), live, fire (early 底/顶 mark), ok (√ confirmation), nb (bars since candidate)."""
    lc = ref(c, 1)
    tr = np.maximum(np.maximum(h - l, np.abs(h - lc)), np.abs(l - lc))
    at = ema(tr, 27); at2 = ema(tr, 199)
    up1 = ema(np.maximum(c - lc, 0), 27); ab1 = ema(np.abs(c - lc), 27)
    up2 = ema(np.maximum(c - lc, 0), 5); ab2 = ema(np.abs(c - lc), 5)
    r14 = np.where(ab1 > 0, up1 / np.where(ab1 > 0, ab1, 1), 0.5); r3 = np.where(ab2 > 0, up2 / np.where(ab2 > 0, ab2, 1), 0.5)
    e50 = ema(c, 50); e200 = ema(c, 200); rg = h - l
    hh20 = rolling_max(h, 20); ll20 = -rolling_max(-l, 20)
    mom = (c - ref(c, 10)) / at; arat = np.clip(at / at2, 0, 4)
    cl = lambda x, a, b: np.minimum(np.maximum(x, a), b)
    w = M["model"][side]["w"]; b0 = M["model"][side]["b0"]
    if side == "bottom":
        F = [cl((hh20 - l) / at, 0, 12), cl(mom, -12, 12), cl(rg / at, 0, 6), np.where(rg > 0, (c - l) / np.where(rg > 0, rg, 1), 0.5),
             cl((np.minimum(o, c) - l) / at, 0, 4), cl((c - o) / at, -4, 4), r14, r3, np.minimum(barslast(c >= lc), 8) / 8,
             cl((c - e50) / at, -15, 15), cl((c - e200) / at, -30, 30), arat]
        cand = (l <= lowest_prev(l, 4)) & ((hh20 - l) >= 1.5 * at); ext, oth = l, h
    else:
        F = [cl((h - ll20) / at, 0, 12), cl(-mom, -12, 12), cl(rg / at, 0, 6), np.where(rg > 0, (h - c) / np.where(rg > 0, rg, 1), 0.5),
             cl((h - np.maximum(o, c)) / at, 0, 4), cl((o - c) / at, -4, 4), 1 - r14, 1 - r3, np.minimum(barslast(c <= lc), 8) / 8,
             cl((e50 - c) / at, -15, 15), cl((e200 - c) / at, -30, 30), arat]
        cand = (h >= -lowest_prev(-h, 4)) & ((h - ll20) >= 1.5 * at); ext, oth = h, l
    cand = np.nan_to_num(cand.astype(float)) > 0
    zb = b0 + sum(w[k] * f for k, f in zip(BASE, F))
    nb = barslast(cand)
    pick = lambda x: np.where(nb == 0, x, np.where(nb == 1, ref(x, 1), np.where(nb == 2, ref(x, 2), ref(x, 3))))
    ce, co, ca, zc = pick(ext), pick(oth), pick(at), pick(zb)
    if side == "bottom":
        tg = ce + 2 * ca
        d1 = (l <= ce) | (h >= tg); d2 = (ref(l, 1) <= ce) | (ref(h, 1) >= tg); d3 = (ref(l, 2) <= ce) | (ref(h, 2) >= tg)
        bnc = cl((c - ce) / ca, -2, 6); rec = (c > co).astype(float)
    else:
        tg = ce - 2 * ca
        d1 = (h >= ce) | (l <= tg); d2 = (ref(h, 1) >= ce) | (ref(l, 1) <= tg); d3 = (ref(h, 2) >= ce) | (ref(l, 2) <= tg)
        bnc = cl((ce - c) / ca, -2, 6); rec = (c < co).astype(float)
    live = (nb == 0) | ((nb == 1) & ~d1) | ((nb == 2) & ~d1 & ~d2) | ((nb == 3) & ~d1 & ~d2 & ~d3)
    z = zc + w["c1"] * (nb == 1) + w["c2"] * (nb == 2) + w["c3"] * (nb == 3) + w["bounce"] * bnc + w["reclaim"] * rec + w["clv_now"] * F[3]
    p = 100 / (1 + np.power(2.718281828, -z))
    s = live & (p >= TH)
    fire = s & ~(((nb >= 1) & ref(s, 1)) | ((nb >= 2) & ref(s, 2)) | ((nb >= 3) & ref(s, 3)))
    cv = ref_var(ext, nb); av = ref_var(at, nb)
    if side == "bottom": ht = (nb >= 1) & (h >= cv + 2 * av); bk = (nb >= 1) & (l <= cv)
    else: ht = (nb >= 1) & (l <= cv - 2 * av); bk = (nb >= 1) & (h >= cv)
    ok = ht & ~bk & ((nb == 1) | ((ref(barslast(ht), 1) > nb - 2) & (ref(barslast(bk), 1) > nb - 2)))
    valid = np.arange(len(c)) >= WARMUP
    return dict(p=p, live=live & valid, fire=fire & valid, ok=ok & valid, nb=nb, ext=cv, atr=at)


def stats_text(tf, side, level="均衡"):
    v = M["validation"].get(tf + "_" + side, {}).get(level)
    return "" if not v else "历史准确率 %d%%（标出率 %d%%）" % (round(v["precision"] * 100), round(v["recall"] * 100))
