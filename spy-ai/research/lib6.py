"""v6 sample builder: v5 candidates / labels / stages (unchanged definition) on moomoo's own K-lines (real volume),
with every moomoo built-in indicator + day / VIX context + higher-timeframe state, all measured at the decision bar.
Features come from feat6.py (shared with the live watcher); only the label scan (labels.outcomes) is research-side.

Candidate (bottom): low <= lowest low of previous 4 bars, after a drop >= 1.5 ATR from the 20-bar high.
Label: TRUE if price reaches low + 2 ATR within 60 bars before any bar trades below that low. Tops = mirror image.
Stages c = 0..3: the candidate is re-scored on each of the next bars while unresolved (same as the live watcher)."""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import labels                                        # outcome scanner only (label)
import ind, vixdata, feat6
from feat6 import V5, DAY, HTF, CMAX, decorate, bar_features

TFS = ["1m", "3m", "5m", "15m", "1h"]
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}
TRAIN_END = pd.Timestamp("2023-12-31"); VAL_END = pd.Timestamp("2026-03-25")      # holdout = after VAL_END
IND = None   # indicator column names (filled on first build)


def load(tf):
    return decorate(pd.read_parquet(os.path.join(HERE, "data", "spy_%s.parquet" % tf)))


def vix_prev_table():
    return vixdata.prev_close_table(pd.read_parquet(os.path.join(HERE, "data", "vix_daily.parquet")))


def build(tf, side, b=None, vix_prev=None, htf_bars=None, bars_feat=None):
    """Returns (S, b, px): one row per (candidate, stage) with v5 features + all bar features at the decision bar."""
    global IND
    b = load(tf) if b is None else b
    if vix_prev is None: vix_prev = vix_prev_table()
    if htf_bars is None: htf_bars = {"h15": load("15m"), "h60": load("1h")}
    BF = bar_features(b, vix_prev, htf_bars) if bars_feat is None else bars_feat
    IND = [k for k in BF.columns if k not in DAY and k not in HTF]
    f5, cand, px = feat6.base_features5(b, side)
    h, l, atr = px["h"], px["l"], px["atr"]
    cidx = np.flatnonzero(cand)
    fb, fh, tie = labels.outcomes(cidx, h, l, atr)
    keep = fb != -1; cidx, fb, fh, tie = cidx[keep], fb[keep], fh[keep], tie[keep]
    label = (fh < fb).astype(int); rows = []
    for cc in range(CMAX + 1):
        live = (cc < tie) & (cc < fh) if cc > 0 else np.ones(len(cidx), bool)
        ii = cidx[live]; jj = np.minimum(ii + cc, len(b) - 1)
        R = feat6.stage_rows(f5, BF, px, ii, jj, cc)
        R["cand_i"] = ii; R["bar_j"] = jj; R["label"] = label[live]; R["fb"] = fb[live]; R["fh"] = fh[live]
        rows.append(R)
    S = pd.concat(rows, ignore_index=True)
    S["date"] = b.date.to_numpy()[S.bar_j]; S["tf"] = tf; S["tf_min"] = np.log(TF_MIN[tf])
    return S, b, px


def feature_sets():
    """Ablation ladder: each block adds one kind of information."""
    vol_fams = {"VWAP", "VOL", "VR", "OBV", "EMV", "MFI"}
    ind_price = [k for k in IND if ind.FAMILY.get(k) not in vol_fams]
    ind_vol = [k for k in IND if ind.FAMILY.get(k) in vol_fams]
    A = V5 + ["tf_min"]; B = A + ind_price; C = B + ind_vol; D = C + DAY; E = D + HTF
    return {"A_v5基础": A, "B_+价格类指标": B, "C_+量能类指标": C, "D_+日内位置/VIX": D, "E_+大周期状态": E}
