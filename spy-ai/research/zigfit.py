"""Which zigzag reproduces the user's hand-drawn lines best?
ZigZag(theta): a new pivot is confirmed when price reverses theta% from the running extreme (1h high/low, unadjusted).
Its pivot sequence is aligned to the user's drawn vertex prices (Needleman-Wunsch, same kind only, cost = |price diff|,
gap cost GAP dollars). The theta with the lowest cost = the rule that "draws like the user"."""
import numpy as np, pandas as pd
import user_lines as UL

GAP = 2.0


def zigzag(h, l, theta):
    """standard percent zigzag on highs/lows. returns [(bar, kind, price, confirm_bar)];
    confirm_bar = the bar on which the reversal of theta% made the pivot known (real time)."""
    n = len(h); piv = []; trend = 0
    hi_i, hi_p, lo_i, lo_p = 0, h[0], 0, l[0]
    for i in range(1, n):
        if trend == 0:
            if h[i] > hi_p: hi_i, hi_p = i, h[i]
            if l[i] < lo_p: lo_i, lo_p = i, l[i]
            if hi_p >= lo_p * (1 + theta) and lo_i < hi_i:
                piv.append((lo_i, "B", lo_p, i)); trend = 1
            elif lo_p <= hi_p * (1 - theta) and hi_i < lo_i:
                piv.append((hi_i, "T", hi_p, i)); trend = -1
        elif trend == 1:
            if h[i] > hi_p: hi_i, hi_p = i, h[i]
            elif l[i] <= hi_p * (1 - theta):
                piv.append((hi_i, "T", hi_p, i)); trend = -1; lo_i, lo_p = i, l[i]
        else:
            if l[i] < lo_p: lo_i, lo_p = i, l[i]
            elif h[i] >= lo_p * (1 + theta):
                piv.append((lo_i, "B", lo_p, i)); trend = 1; hi_i, hi_p = i, h[i]
    return piv


def align(D, Z):
    n, m = len(D), len(Z); C = np.zeros((n + 1, m + 1)); C[:, 0] = np.arange(n + 1) * GAP; C[0, :] = np.arange(m + 1) * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mt = C[i - 1, j - 1] + (abs(D[i - 1][1] - Z[j - 1][1]) if D[i - 1][0] == Z[j - 1][0] else 1e9)
            C[i, j] = min(mt, C[i - 1, j] + GAP, C[i, j - 1] + GAP)
    i, j, pairs = n, m, []                           # traceback
    while i > 0 and j > 0:
        if D[i - 1][0] == Z[j - 1][0] and abs(C[i, j] - (C[i - 1, j - 1] + abs(D[i - 1][1] - Z[j - 1][1]))) < 1e-9:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif abs(C[i, j] - (C[i - 1, j] + GAP)) < 1e-9: i -= 1
        else: j -= 1
    return C[n, m], pairs[::-1]


if __name__ == "__main__":
    b = UL.raw_1h(); w = b[(b.t >= "2026-08-05") & (b.t <= "2026-10-05 23:59")].reset_index(drop=True)
    h, l = w.h.to_numpy(float), w.l.to_numpy(float)
    P0, Y0, DY = UL.P0, UL.Y0, UL.DY
    D = [(k, P0 - (y - Y0) / DY) for x, y, k in UL.V]
    best = None
    for th in np.arange(0.20, 1.51, 0.05):
        Z = [(k, p) for _, k, p, _ in zigzag(h, l, th / 100)]
        cost, pairs = align(D, Z)
        md = np.mean([abs(D[a][1] - Z[c][1]) for a, c in pairs]) if pairs else 99
        good = sum(abs(D[a][1] - Z[c][1]) <= 1.0 for a, c in pairs); rec = good / len(D); prec = good / max(len(Z), 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        print("theta %.2f%%  pivots %3d  matched(<= $1) %2d/%d  recall %.2f precision %.2f F1 %.2f  mean |dprice| %.2f" % (th, len(Z), good, len(D), rec, prec, f1, md))
        if best is None or f1 > best[0]: best = (f1, th, pairs, Z)
    cost, th, pairs, Z = best
    zz = zigzag(h, l, th / 100)
    out = pd.DataFrame([{"kind": k, "t": w.t[i], "px": p, "confirm_t": w.t[c], "lag_bars": c - i,
                         "lag_pct": (abs(w.c[c] / p - 1) * 100)} for i, k, p, c in zz])
    out.to_csv("data/zigzag_best.csv", index=False)
    print("\nBEST theta = %.2f%%: %d of %d drawn vertices matched" % (th, len(pairs), len(D)))
    print(out.round(3).to_string(index=False))
