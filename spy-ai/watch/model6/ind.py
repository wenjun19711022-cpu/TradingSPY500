"""moomoo / 通达信 built-in technical indicators with the platform's own formula semantics
(EMA and SMA(X,N,M) seeded with the first value, MA / SUM / HHV / LLV over N bars, STD = sample std).
Every function takes numpy arrays and returns numpy arrays of the same length.
`features(b)` turns them into scale-free model inputs (price distances in ATR, oscillators in their native 0-100 range)."""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ primitives
def EMA(x, n):
    x = np.nan_to_num(np.asarray(x, float)); out = np.empty(len(x)); a = 2.0 / (n + 1); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def SMA(x, n, m):                     # 通达信 SMA(X,N,M): Y = (M*X + (N-M)*Y') / N
    x = np.nan_to_num(np.asarray(x, float)); out = np.empty(len(x)); out[0] = x[0]; a = m / n
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def MA(x, n): return pd.Series(x).rolling(n, min_periods=1).mean().to_numpy()
def SUM(x, n): return pd.Series(x).rolling(n, min_periods=1).sum().to_numpy()
def HHV(x, n): return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()
def LLV(x, n): return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()
def STD(x, n): return pd.Series(x).rolling(n, min_periods=2).std().fillna(0).to_numpy()
def REF(x, n=1):
    x = np.asarray(x, float); return np.r_[np.full(n, x[0]), x[:-n]] if n > 0 else x
def AVEDEV(x, n):
    s = pd.Series(x); return s.rolling(n, min_periods=1).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True).to_numpy()
def AVEDEV_fast(x, n):                # same as AVEDEV, vectorised with a strided window
    x = np.asarray(x, float); out = np.empty(len(x))
    if len(x) >= n:
        W = np.lib.stride_tricks.sliding_window_view(x, n); out[n - 1:] = np.abs(W - W.mean(1, keepdims=True)).mean(1)
    for i in range(min(n - 1, len(x))): w = x[:i + 1]; out[i] = np.abs(w - w.mean()).mean()
    return out
def div(a, b, fill=0.0):
    b = np.asarray(b, float); return np.where(np.abs(b) > 1e-12, np.asarray(a, float) / np.where(np.abs(b) > 1e-12, b, 1), fill)


# ------------------------------------------------------------------ indicators (default moomoo parameters)
def ATR(h, l, c, n=14):
    lc = REF(c); tr = np.maximum(np.maximum(h - l, np.abs(h - lc)), np.abs(l - lc)); return EMA(tr, 2 * n - 1)

def MACD(c, s=12, l=26, m=9):
    dif = EMA(c, s) - EMA(c, l); dea = EMA(dif, m); return dif, dea, 2 * (dif - dea)

def KDJ(h, l, c, n=9, m1=3, m2=3):
    rsv = div(c - LLV(l, n), HHV(h, n) - LLV(l, n), 0.5) * 100
    k = SMA(rsv, m1, 1); d = SMA(k, m2, 1); return k, d, 3 * k - 2 * d

def RSI(c, n):
    lc = REF(c); return div(SMA(np.maximum(c - lc, 0), n, 1), SMA(np.abs(c - lc), n, 1), 0.5) * 100

def WR(h, l, c, n): return div(HHV(h, n) - c, HHV(h, n) - LLV(l, n), 0.5) * 100

def BIAS(c, n): m = MA(c, n); return div(c - m, m) * 100

def CCI(h, l, c, n=14):
    typ = (h + l + c) / 3; return div(typ - MA(typ, n), 0.015 * AVEDEV_fast(typ, n))

def DMI(h, l, c, n=14, m=6):
    lc = REF(c); tr = SUM(np.maximum(np.maximum(h - l, np.abs(h - lc)), np.abs(l - lc)), n)
    hd = h - REF(h); ld = REF(l) - l
    dmp = SUM(np.where((hd > 0) & (hd > ld), hd, 0), n); dmm = SUM(np.where((ld > 0) & (ld > hd), ld, 0), n)
    pdi = div(dmp * 100, tr); mdi = div(dmm * 100, tr)
    adx = MA(div(np.abs(mdi - pdi), mdi + pdi) * 100, m); adxr = (adx + REF(adx, m)) / 2
    return pdi, mdi, adx, adxr

def TRIX(c, n=12, m=9):
    mtr = EMA(EMA(EMA(c, n), n), n); t = div(mtr - REF(mtr), REF(mtr)) * 100; return t, MA(t, m)

def ARBR(o, h, l, c, n=26):
    ar = div(SUM(h - o, n), SUM(o - l, n), 1) * 100
    lc = REF(c); br = div(SUM(np.maximum(0, h - lc), n), SUM(np.maximum(0, lc - l), n), 1) * 100
    return ar, br

def CR(h, l, n=26):
    mid = REF(h + l) / 2; return div(SUM(np.maximum(0, h - mid), n), SUM(np.maximum(0, mid - l), n), 1) * 100

def VR(c, v, n=26):
    lc = REF(c); th = SUM(np.where(c > lc, v, 0), n); tl = SUM(np.where(c < lc, v, 0), n); tq = SUM(np.where(c == lc, v, 0), n)
    return div(100 * (th * 2 + tq), tl * 2 + tq, 100)

def OBV(c, v):
    lc = REF(c); return np.cumsum(np.where(c > lc, v, np.where(c < lc, -v, 0)))

def PSY(c, n=12, m=6):
    p = SUM((c > REF(c)).astype(float), n) / n * 100; return p, MA(p, m)

def DMA(c, n1=10, n2=50, m=10):
    d = MA(c, n1) - MA(c, n2); return d, MA(d, m)

def EMV(h, l, v, n=14, m=9):
    vol = div(MA(v, n), v, 1); mid = div(100 * (h + l - REF(h + l)), h + l)
    e = MA(mid * vol * div(h - l, MA(h - l, n), 1), n); return e, MA(e, m)

def MTM(c, n=12, m=6): x = c - REF(c, n); return x, MA(x, m)

def ROC(c, n=12, m=6): x = div(c - REF(c, n), REF(c, n)) * 100; return x, MA(x, m)

def ASI(o, h, l, c, m1=26, m2=10):
    lc = REF(c); aa = np.abs(h - lc); bb = np.abs(l - lc); cc = np.abs(h - REF(l)); dd = np.abs(lc - REF(o))
    r = np.where((aa > bb) & (aa > cc), aa + bb / 2 + dd / 4, np.where((bb > cc) & (bb > aa), bb + aa / 2 + dd / 4, cc + dd / 4))
    x = c - lc + (c - o) / 2 + lc - REF(o); si = div(16 * x, r) * np.maximum(aa, bb)
    a = SUM(si, m1); return a, MA(a, m2)

def MFI(h, l, c, v, n=14):
    typ = (h + l + c) / 3; lt = REF(typ)
    pos = SUM(np.where(typ > lt, typ * v, 0), n); neg = SUM(np.where(typ < lt, typ * v, 0), n)
    return 100 - div(100.0, 1 + div(pos, neg, 1e3), 0)

def BOLL(c, n=20, k=2):
    mid = MA(c, n); sd = STD(c, n); return mid, mid + k * sd, mid - k * sd

def BBI(c): return (MA(c, 3) + MA(c, 6) + MA(c, 12) + MA(c, 24)) / 4

def SAR(h, l, step=0.02, mx=0.2, n=4):
    """Parabolic SAR (Wilder). Returns sar, direction (+1 long / -1 short)."""
    N = len(h); sar = np.empty(N); d = np.empty(N)
    up = True; af = step; ep = h[0]; s = l[0]
    for i in range(N):
        if i == 0:
            sar[i] = s; d[i] = 1; continue
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up = False; s = ep; ep = l[i]; af = step
            elif h[i] > ep: ep = h[i]; af = min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up = True; s = ep; ep = h[i]; af = step
            elif l[i] < ep: ep = l[i]; af = min(af + step, mx)
        sar[i] = s; d[i] = 1 if up else -1
    return sar, d

def STOCHRSI(c, n=14, m=14, k=3, d=3):
    r = RSI(c, n); s = div(r - LLV(r, m), HHV(r, m) - LLV(r, m), 0.5) * 100; kk = MA(s, k); return kk, MA(kk, d)

def barslast(cond):
    out = np.zeros(len(cond)); run = 0
    for i, x in enumerate(cond):
        run = 0 if x else run + 1; out[i] = run
    return out


# ------------------------------------------------------------------ feature block (side-independent, raw prices)
FAMILY = {}   # feature -> indicator family (for grouped importance)

def _put(F, fam, **cols):
    for k, v in cols.items(): F[k] = v; FAMILY[k] = fam

def features(o, h, l, c, v, tv, day_id):
    """All moomoo indicators as scale-free features. day_id: int per session (for VWAP / intraday resets)."""
    F = {}
    atr = ATR(h, l, c, 14); a = np.where(atr > 0, atr, np.nan)
    z = lambda x: np.clip(x / a, -30, 30)
    _put(F, "ATR", atr_bp=atr / c * 1e4, atr_rel100=np.clip(div(atr, ATR(h, l, c, 100), 1), 0, 4))
    for n in (5, 10, 20, 60): _put(F, "MA", **{"ma%d_d" % n: z(c - MA(c, n))})
    _put(F, "MA", ma_align=((MA(c, 5) > MA(c, 10)).astype(float) + (MA(c, 10) > MA(c, 20)) + (MA(c, 20) > MA(c, 60))))
    for n in (5, 10, 20, 60): _put(F, "EMA", **{"ema%d_d" % n: z(c - EMA(c, n))})
    mid, up, lo = BOLL(c); _put(F, "BOLL", boll_pb=np.clip(div(c - lo, up - lo, 0.5), -1, 2), boll_w=z(up - lo))
    sar, sd = SAR(h, l); _put(F, "SAR", sar_d=z(c - sar), sar_dir=sd, sar_age=np.minimum(barslast(np.r_[True, sd[1:] != sd[:-1]]), 50))
    _put(F, "BBI", bbi_d=z(c - BBI(c)))
    new_day = np.r_[True, day_id[1:] != day_id[:-1]]
    g = np.cumsum(new_day) - 1
    # intraday VWAP from typical price x volume (turnover/volume would mix unadjusted $ with forward-adjusted prices)
    cv = pd.Series(v).groupby(g).cumsum().to_numpy(); cpv = pd.Series((h + l + c) / 3 * v).groupby(g).cumsum().to_numpy()
    vwap = np.where(cv > 0, div(cpv, cv, np.nan), c); _put(F, "VWAP", vwap_d=z(c - vwap))
    _put(F, "VOL", vol_r5=np.clip(div(v, MA(v, 5), 1), 0, 10), vol_r20=np.clip(div(v, MA(v, 20), 1), 0, 10),
         vol_r100=np.clip(div(v, MA(v, 100), 1), 0, 10))
    dif, dea, mac = MACD(c); _put(F, "MACD", macd_dif=z(dif), macd_dea=z(dea), macd_h=z(mac), macd_hs=z(mac - REF(mac)),
                                  macd_x=np.sign(dif - dea) - np.sign(REF(dif) - REF(dea)))
    k, d, j = KDJ(h, l, c); _put(F, "KDJ", kdj_k=k, kdj_d=d, kdj_j=j, kdj_kd=k - d)
    for n in (6, 12, 24): _put(F, "RSI", **{"rsi%d" % n: RSI(c, n)})
    _put(F, "WR", wr10=WR(h, l, c, 10), wr6=WR(h, l, c, 6))
    for n in (6, 12, 24): _put(F, "BIAS", **{"bias%d" % n: BIAS(c, n)})
    _put(F, "CCI", cci=np.clip(CCI(h, l, c), -500, 500))
    pdi, mdi, adx, adxr = DMI(h, l, c); _put(F, "DMI", dmi_pdi=pdi, dmi_mdi=mdi, dmi_adx=adx, dmi_adxr=adxr, dmi_pm=pdi - mdi)
    t, tm = TRIX(c); _put(F, "TRIX", trix=t * 100, trix_m=(t - tm) * 100)
    ar, br = ARBR(o, h, l, c); _put(F, "ARBR", ar=np.clip(ar, 0, 1000), br=np.clip(br, 0, 1000))
    _put(F, "CR", cr=np.clip(CR(h, l), 0, 1000))
    _put(F, "VR", vr=np.clip(VR(c, v), 0, 1000))
    ob = OBV(c, v); _put(F, "OBV", obv10=div(ob - REF(ob, 10), SUM(v, 10)), obv30=div(ob - REF(ob, 30), SUM(v, 30)))
    p, pm = PSY(c); _put(F, "PSY", psy=p, psy_m=pm)
    dd, da = DMA(c); _put(F, "DMA", dma=z(dd), dma_a=z(dd - da))
    e, em = EMV(h, l, v); _put(F, "EMV", emv=np.clip(e * 100, -50, 50), emv_m=np.clip((e - em) * 100, -50, 50))
    m, mm = MTM(c); _put(F, "MTM", mtm=z(m), mtm_m=z(m - mm))
    r, rm = ROC(c); _put(F, "ROC", roc=r, roc_m=r - rm)
    asi, asit = ASI(o, h, l, c); s26 = 16 * 26 * a       # SI ~ 16 x ATR per bar, summed over 26 bars
    _put(F, "ASI", asi=np.clip(asi / s26, -3, 3), asi_t=np.clip((asi - asit) / s26, -3, 3))
    _put(F, "MFI", mfi=MFI(h, l, c, v))
    sk, sdd = STOCHRSI(c); _put(F, "StochRSI", srsi_k=sk, srsi_d=sdd)
    return pd.DataFrame(F)
