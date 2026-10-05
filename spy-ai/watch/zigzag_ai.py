"""折线AI marks — exactly the moomoo main-chart formula 'SPY折线AI' (v6/zv_moo_gen.py), computed on the watcher's window:
  BUY  = v5 bottom mark AND close >= D% under the highest high of the last N bars AND the latest raw SELL is newer than the previous raw BUY
  SELL = v5 top mark    AND close >= D% over the lowest low of the last N bars   AND the latest raw BUY is newer than the previous raw SELL"""
import numpy as np, pandas as pd


def barslast(cond):
    out = np.full(len(cond), 10 ** 6, float); last = None
    for i, v in enumerate(cond):
        if v: last = i
        out[i] = (i - last) if last is not None else 10 ** 6
    return out


def marks(h, l, c, fire_b, fire_t, N=56, D=0.3):
    hh = pd.Series(h).rolling(N, min_periods=1).max().to_numpy(); ll = pd.Series(l).rolling(N, min_periods=1).min().to_numpy()
    buy = fire_b & ((hh - c) / hh * 100 >= D); sell = fire_t & ((c - ll) / ll * 100 >= D)
    buy2 = buy & (barslast(sell) < barslast(np.r_[False, buy[:-1]])); sell2 = sell & (barslast(buy) < barslast(np.r_[False, sell[:-1]]))
    return buy2, sell2, (hh - c) / hh * 100, (c - ll) / ll * 100
