"""v6 features -- ONE implementation shared by training (lib6.py) and the live watcher (watch/v6core.py).
No dependency on the research folders: v5 base features re-implemented with the exact v5 (moomoo) semantics."""
import numpy as np, pandas as pd
import ind

DROP_MIN, CMAX, K, WARMUP = 1.5, 3, 2.0, 300
BASE5 = ["drop_atr", "mom10", "rng_atr", "clv", "lwick", "body", "rsi14", "rsi3", "ndown", "ema50d", "ema200d", "atr_ratio"]
STAGE = ["c1", "c2", "c3", "bounce", "reclaim", "clv_now"]
V5 = BASE5 + STAGE
DAY = ["daymove", "day_rng", "gap", "pdl_d", "pdh_d", "pdc_d", "tod", "dow", "vix", "vix_term", "vix_9d", "regime_up"]
HTF = ["%s_%s" % (t, c) for t in ("h15", "h60") for c in ("rsi", "kdj_j", "boll_pb", "macd_h", "ema20_d", "sar_dir")]


# ------------------------------------------------------------------ bars
def decorate(b):
    """b: t (bar END time 'YYYY-MM-DD HH:MM:SS'), o,h,l,c,v,tv -> + date, end_min, day_id"""
    b = b.reset_index(drop=True).copy()
    b["date"] = pd.to_datetime(b.t.str[:10]); b["end_min"] = b.t.str[11:13].astype(int) * 60 + b.t.str[14:16].astype(int)
    b["day_id"] = (b.date - pd.Timestamp("2000-01-01")).dt.days.astype(int)
    return b


# ------------------------------------------------------------------ v5 base features (moomoo semantics, side-mirrored)
def _ema(x, n):
    x = np.nan_to_num(np.asarray(x, float)); out = np.empty(len(x)); a = 2.0 / (n + 1)
    if len(x): out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def _atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc)); return _ema(tr, 2 * n - 1)

def _rsi(c, n):
    lc = np.r_[c[0], c[:-1]]; d = c - lc
    up = _ema(np.maximum(d, 0), 2 * n - 1); ab = _ema(np.abs(d), 2 * n - 1)
    return np.where(ab > 0, up / np.where(ab > 0, ab, 1) * 100, 50.0)

def _ndown(c):
    out = np.zeros(len(c)); run = 0
    for i in range(len(c)):
        run = run + 1 if (i > 0 and c[i] < c[i - 1]) else 0; out[i] = run
    return out

def _highest(x, n): return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()
def _lowest_prev(x, n): return pd.Series(x).shift(1).rolling(n, min_periods=n).min().to_numpy()

def side_series(b, side):
    o, h, l, c = (b[k].to_numpy(float) for k in ("o", "h", "l", "c"))
    if side == "top": o, h, l, c = -o, -l, -h, -c
    return o, h, l, c

def base_features5(b, side):
    o, h, l, c = side_series(b, side)
    atr = _atr(h, l, c, 14); atr100 = _atr(h, l, c, 100)
    ema50 = _ema(c, 50); ema200 = _ema(c, 200); hh20 = _highest(h, 20); low4 = _lowest_prev(l, 4)
    rng = h - l; c10 = np.r_[np.full(10, np.nan), c[:-10]]
    f = pd.DataFrame({
        "drop_atr": np.clip((hh20 - l) / atr, 0, 12), "mom10": np.clip((c - c10) / atr, -12, 12),
        "rng_atr": np.clip(rng / atr, 0, 6), "clv": np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5),
        "lwick": np.clip((np.minimum(o, c) - l) / atr, 0, 4), "body": np.clip((c - o) / atr, -4, 4),
        "rsi14": _rsi(c, 14) / 100, "rsi3": _rsi(c, 3) / 100, "ndown": np.minimum(_ndown(c), 8) / 8,
        "ema50d": np.clip((c - ema50) / atr, -15, 15), "ema200d": np.clip((c - ema200) / atr, -30, 30),
        "atr_ratio": np.clip(atr / atr100, 0, 4)})
    warm = np.arange(len(b)) >= WARMUP
    cand = (l <= low4) & ((hh20 - l) >= DROP_MIN * atr) & warm
    return f, cand, dict(h=h, l=l, c=c, atr=atr)


# ------------------------------------------------------------------ day / VIX context
def daily_table(h1):
    """previous-session high / low / close and 50-day average from 1h bars (live 1h window ~140 days; a 1000-bar
    1m window only ~2.5 days -- so training and live both take these from 1h)."""
    d = h1.groupby("date").agg(pdh=("h", "max"), pdl=("l", "min"), pdc=("c", "last")).reset_index()
    d["ma50p"] = d.pdc.rolling(50, min_periods=20).mean()
    return d[["date", "pdh", "pdl", "pdc", "ma50p"]]          # keyed by the COMPLETED day; joined to later days below

def day_features(b, vix_prev, atr, daily):
    o, h, l, c = (b[k].to_numpy(float) for k in ("o", "h", "l", "c")); g = b.day_id.to_numpy()
    gi = np.cumsum(np.r_[True, g[1:] != g[:-1]]) - 1
    s = pd.DataFrame({"g": gi, "o": o, "h": h, "l": l})
    dopen = s.groupby("g").o.transform("first").to_numpy()
    dhi = s.groupby("g").h.cummax().to_numpy(); dlo = s.groupby("g").l.cummin().to_numpy()
    # last completed session strictly before each bar's date (works live before today's first 1h bar exists)
    dd = pd.merge_asof(b[["date"]].reset_index(), daily.sort_values("date"), on="date", direction="backward",
                       allow_exact_matches=False).set_index("index").sort_index()
    pdh, pdl, pdc, ma50 = (dd[k].to_numpy(float) for k in ("pdh", "pdl", "pdc", "ma50p"))
    a = np.where(atr > 0, atr, np.nan)
    ctx = b[["date"]].merge(vix_prev, on="date", how="left")
    return pd.DataFrame({
        "daymove": np.clip((c - dopen) / a, -30, 30), "day_rng": np.clip((dhi - dlo) / a, 0, 60),
        "gap": np.clip((dopen - pdc) / pdc * 100, -5, 5), "pdl_d": np.clip((l - pdl) / a, -40, 40),
        "pdh_d": np.clip((h - pdh) / a, -40, 40), "pdc_d": np.clip((c - pdc) / a, -40, 40),
        "tod": b.end_min.to_numpy() - 570, "dow": b.date.dt.dayofweek.to_numpy(),
        "vix": ctx.vix_prev.to_numpy(), "vix_term": (ctx.vix_prev / ctx.vix3m_prev).to_numpy(),
        "vix_9d": (ctx.vix9d_prev / ctx.vix_prev).to_numpy(), "regime_up": np.where(np.isnan(ma50), np.nan, (pdc > ma50).astype(float))})


# ------------------------------------------------------------------ higher-timeframe state (last CLOSED 15m / 1h bar)
def htf_block(hb):
    o, h, l, c = (hb[k].to_numpy(float) for k in ("o", "h", "l", "c"))
    atr = ind.ATR(h, l, c, 14); a = np.where(atr > 0, atr, np.nan)
    _, _, mac = ind.MACD(c); _, _, j = ind.KDJ(h, l, c); mid, up, lo = ind.BOLL(c); _, sd = ind.SAR(h, l)
    return pd.DataFrame({"t": hb.t.to_numpy(), "rsi": ind.RSI(c, 14), "kdj_j": j, "boll_pb": np.clip(ind.div(c - lo, up - lo, 0.5), -1, 2),
                         "macd_h": np.clip(mac / a, -10, 10), "ema20_d": np.clip((c - ind.EMA(c, 20)) / a, -20, 20), "sar_dir": sd})

def htf_features(b, htf_bars):
    out = pd.DataFrame(index=b.index); key = pd.to_datetime(b.t)
    for tag, hb in htf_bars.items():
        blk = htf_block(hb); blk["tk"] = pd.to_datetime(blk.t)
        m = pd.merge_asof(pd.DataFrame({"tk": key}).reset_index(), blk.drop(columns="t").sort_values("tk"), on="tk", direction="backward")
        m = m.set_index("index").sort_index()
        for col in ("rsi", "kdj_j", "boll_pb", "macd_h", "ema20_d", "sar_dir"): out["%s_%s" % (tag, col)] = m[col].to_numpy()
    return out


def bar_features(b, vix_prev, htf_bars):
    """every moomoo indicator + day/VIX + 15m/1h state, one row per bar (value known at that bar's close)."""
    o, h, l, c, v, tv = (b[k].to_numpy(float) for k in ("o", "h", "l", "c", "v", "tv"))
    I = ind.features(o, h, l, c, v, tv, b.day_id.to_numpy())
    D = day_features(b, vix_prev, ind.ATR(h, l, c, 14), daily_table(htf_bars["h60"]))
    return pd.concat([I, D, htf_features(b, htf_bars)], axis=1)


def stage_rows(f5, BF, px, ii, jj, cc):
    """model rows for candidates ii evaluated at bars jj = ii + cc (stage cc)."""
    h, l, c, atr = px["h"], px["l"], px["c"], px["atr"]
    R = pd.DataFrame(f5.to_numpy(float)[ii], columns=f5.columns)
    for k in (1, 2, 3): R["c%d" % k] = (np.asarray(cc) == k).astype(float) if np.ndim(cc) else float(cc == k)
    R["c"] = cc
    R["bounce"] = np.clip((c[jj] - l[ii]) / atr[ii], -2, 6)
    R["reclaim"] = (c[jj] > h[ii]).astype(float)
    R["clv_now"] = np.where((h[jj] - l[jj]) > 0, (c[jj] - l[jj]) / np.where((h[jj] - l[jj]) > 0, h[jj] - l[jj], 1), 0.5)
    R = pd.concat([R, pd.DataFrame(BF.to_numpy(np.float32)[jj], columns=BF.columns)], axis=1)
    R[V5] = R[V5].fillna(0.0)
    return R
