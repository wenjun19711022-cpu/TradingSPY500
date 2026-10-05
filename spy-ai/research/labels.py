"""Label scan for the bottom/top definition (copied verbatim from bt4/lib4.py so this folder is self-contained):
a candidate is a TRUE bottom if price reaches low + K x ATR within H bars before any bar trades below the low (same bar -> false)."""
import numpy as np

K, H = 2.0, 60


def outcomes(cidx, h, l, atr):
    """first offset (1..H) where low < cand low (break) / high >= low + K*atr (hit); H+1 = never"""
    n = len(l); fb = np.full(len(cidx), H + 1); fh = np.full(len(cidx), H + 1); tie = np.full(len(cidx), H + 1)
    off = np.arange(1, H + 1)
    for s in range(0, len(cidx), 40000):
        ci = cidx[s:s + 40000]; J = np.minimum(ci[:, None] + off[None, :], n - 1)
        valid = (ci[:, None] + off[None, :]) <= n - 1
        lo = l[ci][:, None]; tg = (l[ci] + K * atr[ci])[:, None]
        br = (l[J] < lo) & valid; hi = (h[J] >= tg) & valid; eq = (l[J] <= lo) & valid
        fb[s:s + 40000] = np.where(br.any(1), br.argmax(1) + 1, H + 1)
        fh[s:s + 40000] = np.where(hi.any(1), hi.argmax(1) + 1, H + 1)
        tie[s:s + 40000] = np.where(eq.any(1), eq.argmax(1) + 1, H + 1)
        endc = (ci + H) > n - 1
        fb[s:s + 40000] = np.where(endc & (fb[s:s + 40000] > H) & (fh[s:s + 40000] > H), -1, fb[s:s + 40000])  # unresolved at data end
    return fb, fh, tie
