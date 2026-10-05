"""策略工厂 (strategy factory): many classic intraday long-only strategy families on SPY, one execution engine, one protocol.

Rules (user's, fixed): intraday only (flat by 15:55 ET), OKX SPY-USDT-SWAP long only, 1% margin per trade, 20-50x.
Execution: signal at a bar close -> buy the next 1-minute open; exits = stop / target / exit signal / trailing stop / 15:55,
walked on the 1-minute path (stop before target inside a bar); one position at a time per strategy.
Costs: 12 bp round trip (taker + slippage), 7 bp (maker entry) shown too, +1 bp funding when holding through 12:00 ET.
Leverage: account impact = 1% x L x return; adverse excursion beyond (1/L - 1%) before the exit = liquidation (lose the 1%).
Protocol (fixed BEFORE running): TRAIN 2018-10..2023-12 must show net(12bp) > 0 with t >= 2 and >= 60 trades;
survivors must stay > 0 on VALIDATION 2024-01..2026-03-25; HOLDOUT 2026-03-26..latest reported once.
A Bonferroni bar (one-sided 5% over all configs tried) is reported as the strict test."""
import itertools, json, math, os, sys, time
import numpy as np, pandas as pd
import ind

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
TRAIN_END = pd.Timestamp("2023-12-31"); VAL_END = pd.Timestamp("2026-03-25")
LEVS = (20, 50); MMR = 0.01; ONE = np.timedelta64(1, "m")


# ================================================================== data
def load_1m():
    b = pd.read_parquet(os.path.join(DATA, "spy_1m.parquet"))
    t = pd.to_datetime(b.t).values; em = (b.t.str[11:13].astype(int) * 60 + b.t.str[14:16].astype(int)).to_numpy(); d = b.t.str[:10].to_numpy()
    f = pd.DataFrame({"d": d, "i": np.arange(len(b))})
    kend = f.d.map(f[em <= 955].groupby("d").i.max()).fillna(-1).astype(int).to_numpy()
    return dict(t=t, o=b.o.to_numpy(float), h=b.h.to_numpy(float), l=b.l.to_numpy(float), c=b.c.to_numpy(float), em=em, d=d, kend=kend)


def bars(tf):
    b = pd.read_parquet(os.path.join(DATA, "spy_%s.parquet" % tf)).reset_index(drop=True)
    b["ts"] = pd.to_datetime(b.t); b["d"] = b.t.str[:10]; b["em"] = b.ts.dt.hour * 60 + b.ts.dt.minute
    o, h, l, c, v = (b[k].to_numpy(float) for k in ("o", "h", "l", "c", "v"))
    g = np.cumsum(np.r_[True, b.d.to_numpy()[1:] != b.d.to_numpy()[:-1]])
    b["atr"] = ind.ATR(h, l, c, 14); b["ema9"] = ind.EMA(c, 9); b["ema21"] = ind.EMA(c, 21); b["ema50"] = ind.EMA(c, 50)
    b["dif"], b["dea"], _ = ind.MACD(c); b["rsi2"] = ind.RSI(c, 2); b["k"], b["dd"], b["j"] = ind.KDJ(h, l, c)
    b["mid"], b["up"], b["lo"] = ind.BOLL(c); b["adx"] = ind.DMI(h, l, c)[2]; b["volr"] = ind.div(v, ind.MA(v, 20), 1)
    tp = (h + l + c) / 3
    b["vwap"] = pd.Series(tp * v).groupby(g).cumsum().to_numpy() / np.maximum(pd.Series(v).groupby(g).cumsum().to_numpy(), 1)
    b["dopen"] = pd.Series(o).groupby(g).transform("first").to_numpy()
    return b


def daily(M):
    df = pd.DataFrame({"d": M["d"], "o": M["o"], "c": M["c"], "h": M["h"], "l": M["l"], "em": M["em"]})
    D = df.groupby("d").agg(open=("o", "first"), close=("c", "last"), high=("h", "max"), low=("l", "min"), last_em=("em", "max"))
    D.index = pd.to_datetime(D.index)
    D["prev_close"] = D.close.shift(1); D["prev_ret"] = D.close.shift(1) / D.close.shift(2) - 1; D["gap"] = D.open / D.prev_close - 1
    D["ma200p"] = D.close.rolling(200, min_periods=120).mean().shift(1); D["trend"] = D.prev_close > D.ma200p
    D["ma20p"] = D.close.rolling(20).mean().shift(1)
    vx = pd.read_parquet(os.path.join(DATA, "vix_daily.parquet")).set_index("date").sort_index()
    vx = vx.shift(1).reindex(D.index, method="ffill")                    # previous session's close
    D["vix"] = vx.vix; D["vterm"] = vx.vix / vx.vix3m
    D["dow"] = D.index.dayofweek
    ym = D.index.to_period("M"); n = D.groupby(ym).cumcount(); m = D.groupby(ym).close.transform("size")
    D["tom"] = (n <= 2) | (n == m - 1)                                   # first 3 + last trading day of the month
    ev = pd.read_csv(os.path.join(DATA, "events.csv"), parse_dates=["date"])
    for k, pat in (("fomc", "FOMC"), ("cpi", "CPI"), ("nfp", "NFP")):
        D[k] = D.index.isin(ev[ev.event.str.contains(pat)].date)
    D["full"] = D.last_em == 960
    return D


def xasset():
    out = {}
    for s in ("QQQ", "IWM", "TLT"):
        p = os.path.join(DATA, "xa_%s_5m.parquet" % s)
        if not os.path.exists(p): return None
        b = pd.read_parquet(p); b["d"] = b.t.str[:10]; b["em"] = b.t.str[11:13].astype(int) * 60 + b.t.str[14:16].astype(int)
        b["ret_open"] = b.c / b.groupby("d").o.transform("first") - 1
        out[s] = b.set_index(["d", "em"]).ret_open
    return out


# ================================================================== execution
def simulate(M, E, exit_sig=None, trail=None, entry_cut=900):
    """E: DataFrame sorted by t with columns t (signal time), stop / stop_dist, tgt_R / tgt_abs, exit_em (optional).
    exit_sig: sorted datetime64 array of exit-signal bar END times. trail: column name holding a trailing distance (price)."""
    rows = []; busy = -1
    T = E.t.values; n = len(E)
    stop_abs = E.stop.to_numpy(float) if "stop" in E else np.full(n, np.nan)
    stop_d = E.stop_dist.to_numpy(float) if "stop_dist" in E else np.full(n, np.nan)
    tR = E.tgt_R.to_numpy(float) if "tgt_R" in E else np.full(n, np.nan)
    tA = E.tgt_abs.to_numpy(float) if "tgt_abs" in E else np.full(n, np.nan)
    xem = E.exit_em.to_numpy(float) if "exit_em" in E else np.full(n, 955.0)
    tr = E[trail].to_numpy(float) if trail else np.full(n, np.nan)
    ks = np.searchsorted(M["t"], T + ONE)
    for r in range(n):
        k = ks[r]
        if k >= len(M["t"]) or M["t"][k] != T[r] + ONE or k <= busy or M["em"][k] > entry_cut: continue
        kend = M["kend"][k]
        if kend < k: continue
        if xem[r] < 955: kend = min(kend, k + int(np.searchsorted(M["em"][k:kend + 1], xem[r], side="right")) - 1)
        if kend < k: continue
        e = M["o"][k]
        stop = stop_abs[r] if not np.isnan(stop_abs[r]) else (e - stop_d[r] if not np.isnan(stop_d[r]) else -np.inf)
        if stop >= e: continue
        tgt = tA[r] if not np.isnan(tA[r]) else (e + tR[r] * (e - stop) if not np.isnan(tR[r]) and np.isfinite(stop) else np.inf)
        if tgt <= e: continue
        lo = M["l"][k:kend + 1]; hi = M["h"][k:kend + 1]; op = M["o"][k:kend + 1]
        st = np.full(len(lo), stop)
        if not np.isnan(tr[r]):                                            # trailing stop from the highest high SO FAR (previous bars)
            st = np.maximum(st, np.r_[-np.inf, np.maximum.accumulate(hi)[:-1]] - tr[r])
        js = np.flatnonzero(lo <= st); jt = np.flatnonzero(hi >= tgt)
        cand = [(kend - k, "time", M["c"][kend])]
        if len(js): j = js[0]; cand.append((j, "stop", min(st[j], op[j]) if j > 0 else min(st[j], e)))
        if len(jt): j = jt[0]; cand.append((j + 0.5, "target", max(tgt, op[j]) if j > 0 else tgt))
        if exit_sig is not None and len(exit_sig):
            i = np.searchsorted(exit_sig, T[r], side="right")
            if i < len(exit_sig):
                kk = int(np.searchsorted(M["t"], exit_sig[i] + ONE))
                if k < kk <= kend: cand.append((kk - k - 0.25, "signal", M["o"][kk]))
        j, why, x = min(cand, key=lambda z: z[0]); jj = int(j)
        if why == "signal": jj = max(jj, 0)
        kx = k + jj
        fund = 1.0 if (M["em"][k] < 720 <= M["em"][min(kx, len(M["em"]) - 1)]) else 0.0
        rows.append((M["d"][k], int(M["em"][k]), (x - e) / e * 1e4, (lo[:jj + 1].min() - e) / e * 1e4, why, kx - k + 1, fund))
        busy = kx
    return pd.DataFrame(rows, columns=["d", "em", "gross", "mae", "why", "mins", "fund"])


def metrics(T, ndays):
    if len(T) == 0: return {"n": 0}
    net = T.gross - 12 - T.fund
    out = {"n": int(len(T)), "per_day": round(len(T) / max(ndays, 1), 3), "gross": round(float(T.gross.mean()), 2),
           "net12": round(float(net.mean()), 2), "net7": round(float(net.mean() + 5), 2), "win": round(float((net > 0).mean()), 3),
           "pf": round(float(net[net > 0].sum() / max(-net[net <= 0].sum(), 1e-9)), 3),
           "t": round(float(net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))), 2) if len(net) > 2 and net.std() > 0 else 0.0}
    for L in LEVS:
        liq = T.mae <= -(1.0 / L - MMR) * 1e4
        acct = np.where(liq, -0.01, 0.01 * L * net / 1e4); eq = np.cumprod(1 + acct)
        out["x%d" % L] = {"liq": int(liq.sum()), "acct": round(float((eq[-1] - 1) * 100), 2), "dd": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 2)}
    return out


def split_of(d):
    d = pd.to_datetime(d); return np.where(d <= TRAIN_END, "train", np.where(d <= VAL_END, "val", "hold"))


# ================================================================== strategy families
def at_time(D, em, mask):
    """signal timestamps em minutes after midnight on the days in mask"""
    days = D.index[mask & D.full]
    return pd.DataFrame({"t": days + pd.to_timedelta(em, unit="m")})


def fam_dayhold(D):
    filt = {"全部": D.full, "昨天跌>0.5%": D.prev_ret < -0.005, "昨天跌>1%": D.prev_ret < -0.01, "跳空低开>0.3%": D.gap < -0.003,
            "跳空高开>0.3%": D.gap > 0.003, "200日线上方": D.trend, "200日线下方": ~D.trend, "VIX升水(平静)": D.vterm < 0.9,
            "VIX倒挂(恐慌)": D.vterm > 1.0, "VIX>25": D.vix > 25, "月初月末": D.tom, "周一": D.dow == 0, "周五": D.dow == 4,
            "200日线上+昨天跌": D.trend & (D.prev_ret < -0.005), "VIX升水+200日线上": (D.vterm < 0.9) & D.trend}
    for (fn, m), em, sp in itertools.product(filt.items(), (570, 575, 600), (0.004, 0.008, 0.015)):
        E = at_time(D, em, m.fillna(False).to_numpy()); E["stop_pct"] = sp
        yield "日内持有", "%s · %s买 · 止损%.1f%%" % (fn, {570: "开盘", 575: "09:35", 600: "10:00"}[em], sp * 100), E, {}


def fam_orb(D, M):
    df = pd.DataFrame({"d": M["d"], "em": M["em"], "h": M["h"], "l": M["l"], "c": M["c"], "t": M["t"]})
    for rng in (5, 15, 30):
        R = df[df.em <= 570 + rng].groupby("d").agg(rh=("h", "max"), rl=("l", "min"))
        X = df[(df.em > 570 + rng) & (df.em <= 720)].join(R, on="d")
        X = X[X.c > X.rh].groupby("d").first().reset_index()          # first 1m close above the range high (before noon)
        X["trend"] = D.trend.reindex(pd.to_datetime(X.d)).to_numpy()
        for stop, tgt, fl in itertools.product(("区间低点", "区间中点"), (1.0, 2.0, np.nan), ("全部", "200日线上方")):
            E = X if fl == "全部" else X[X.trend == True]
            E = pd.DataFrame({"t": E.t.values, "stop": (E.rl if stop == "区间低点" else (E.rh + E.rl) / 2).to_numpy(), "tgt_R": tgt})
            yield "开盘区间突破", "%d分钟区间 · 止损%s · 目标%s · %s" % (rng, stop, "收盘" if np.isnan(tgt) else "%gR" % tgt, fl), E, {}


def fam_gap(D):
    for th, em, k in itertools.product((0.003, 0.005, 0.01), (575, 585), (0.5, 1.0)):
        m = (D.gap < -th).fillna(False).to_numpy()
        days = D.index[m & D.full]
        E = pd.DataFrame({"t": days + pd.to_timedelta(em, unit="m"), "tgt_abs": D.loc[days, "prev_close"].to_numpy(),
                          "stop_dist": (k * D.loc[days, "gap"].abs() * D.loc[days, "prev_close"]).to_numpy()})
        yield "跳空回补", "低开>%.1f%% · %s买 · 止损=%.1f×缺口" % (th * 100, {575: "09:35", 585: "09:45"}[em], k), E, {}


def fam_vwap(B5):
    above = (B5.c > B5.vwap).astype(int)
    for N, kk, tgt in itertools.product((6, 12), (0.5, 1.0), (1.5, 3.0, np.nan)):
        run = above.groupby(B5.d).transform(lambda s: s.rolling(N, min_periods=N).sum().shift(1))
        sig = (run == N) & (B5.ema21 > B5.ema21.shift(3)) & (B5.l <= B5.vwap) & (B5.c > B5.vwap) & (B5.em >= 600) & (B5.em <= 870)
        S = B5[sig]
        E = pd.DataFrame({"t": S.ts.values, "stop": (S.vwap - kk * S.atr).to_numpy(), "tgt_R": tgt})
        yield "VWAP回踩", "前%d根在VWAP上 · 止损VWAP-%.1fATR · 目标%s" % (N, kk, "收盘" if np.isnan(tgt) else "%gR" % tgt), E, {}


def fam_cross(B, tf, D):
    trend = D.trend.reindex(pd.to_datetime(B.d)).to_numpy() == True
    for sig, sa, fl in itertools.product(("EMA9/21", "MACD"), (1.5, 2.5), ("全部", "VWAP+200日线")):
        a, b_ = (B.ema9, B.ema21) if sig == "EMA9/21" else (B.dif, B.dea)
        up = (a > b_) & (a.shift(1) <= b_.shift(1)); dn = (a < b_) & (a.shift(1) >= b_.shift(1))
        m = up & (B.em >= 585) & (B.em <= 900)
        if fl != "全部": m &= (B.c > B.vwap) & trend
        S = B[m]
        E = pd.DataFrame({"t": S.ts.values, "stop_dist": (sa * S.atr).to_numpy()})
        yield "均线/MACD金叉", "%s %s金叉 · 止损%.1fATR · %s · 死叉平" % (tf, sig, sa, fl), E, {"exit_sig": np.sort(B.ts[dn].values)}


def fam_donchian(B15):
    for N, vr, sa, trail in itertools.product((8, 16, 26), (1.0, 1.5), (1.5, 2.5), (False, True)):
        hh = B15.h.rolling(N).max().shift(1)
        m = (B15.c > hh) & (B15.volr >= vr) & (B15.em >= 600) & (B15.em <= 885)
        S = B15[m]
        E = pd.DataFrame({"t": S.ts.values, "stop_dist": (sa * S.atr).to_numpy(), "trail": (2 * S.atr).to_numpy() if trail else np.nan})
        yield "唐奇安突破", "15分钟 突破%d根高点 · 量比≥%.1f · 止损%.1fATR%s" % (N, vr, sa, " · 2ATR移动止损" if trail else ""), E, {"trail": "trail" if trail else None}


def fam_rsi2(B, tf, D):
    trend = D.trend.reindex(pd.to_datetime(B.d)).to_numpy() == True
    for th, ex, sa in itertools.product((5, 10, 20), (60, 80), (1.5, 3.0)):
        m = (B.rsi2 < th) & (B.c > B.ema50) & trend & (B.em >= 600) & (B.em <= 900)
        S = B[m]
        E = pd.DataFrame({"t": S.ts.values, "stop_dist": (sa * S.atr).to_numpy()})
        yield "RSI2超卖回归", "%s RSI2<%d · 上升趋势 · RSI2>%d平 · 止损%.1fATR" % (tf, th, ex, sa), E, {"exit_sig": np.sort(B.ts[B.rsi2 > ex].values)}


def fam_boll(B, tf):
    for tgt, fl in itertools.product(("中轨", "上轨"), ("全部", "ADX<25")):
        m = (B.c.shift(1) < B.lo.shift(1)) & (B.c > B.lo) & (B.em >= 600) & (B.em <= 900)
        if fl != "全部": m &= B.adx < 25
        S = B[m]
        E = pd.DataFrame({"t": S.ts.values, "stop": (S.l - 0.5 * S.atr).to_numpy()})
        xs = B.ts[B.c >= (B.mid if tgt == "中轨" else B.up)].values
        yield "布林下轨回归", "%s 收回下轨 · 到%s平 · %s" % (tf, tgt, fl), E, {"exit_sig": np.sort(xs)}


def fam_kdj(B, tf, D):
    trend = D.trend.reindex(pd.to_datetime(B.d)).to_numpy() == True
    for sa, fl in itertools.product((1.5, 2.5), ("全部", "200日线上方")):
        m = (B.j.shift(1) < 0) & (B.k > B.dd) & (B.k.shift(1) <= B.dd.shift(1)) & (B.em >= 600) & (B.em <= 900)
        if fl != "全部": m &= trend
        S = B[m]
        E = pd.DataFrame({"t": S.ts.values, "stop_dist": (sa * S.atr).to_numpy()})
        yield "KDJ低位金叉", "%s J<0后金叉 · J>100平 · 止损%.1fATR · %s" % (tf, sa, fl), E, {"exit_sig": np.sort(B.ts[B.j > 100].values)}


def fam_afternoon(D, B5):
    for em, th, st in itertools.product((720, 780, 840), (0.002, 0.005), ("VWAP", "0.5%")):
        S = B5[(B5.em == em) & (B5.c / B5.dopen - 1 > th) & (B5.c > B5.vwap)]
        S = S[D.full.reindex(pd.to_datetime(S.d)).to_numpy() == True]
        E = pd.DataFrame({"t": S.ts.values})
        if st == "VWAP": E["stop"] = S.vwap.to_numpy()
        else: E["stop_pct"] = 0.005
        yield "午后趋势延续", "%02d:%02d 比开盘涨>%.1f%%且在VWAP上 · 止损%s" % (em // 60, em % 60, th * 100, st), E, {}


def fam_events(D, B5):
    for ev, em, xem, sp in (("fomc", 600, 835, 0.004), ("fomc", 600, 835, 0.008), ("fomc", 570, 835, 0.008),
                            ("cpi", 600, 955, 0.005), ("nfp", 600, 955, 0.005), ("cpi|nfp", 600, 955, 0.008)):
        m = np.zeros(len(D), bool)
        for e in ev.split("|"): m |= D[e].to_numpy()
        days = D.index[m & D.full.to_numpy()]
        if ev != "fomc":                                                   # data days: only if SPY is up from the open at 10:00
            S = B5[(B5.em == em) & (B5.c > B5.dopen)]; days = days[days.isin(pd.to_datetime(S.d))]
        E = pd.DataFrame({"t": days + pd.to_timedelta(em, unit="m"), "stop_pct": sp, "exit_em": xem})
        yield "事件日", "%s · %02d:%02d买 · %02d:%02d平 · 止损%.1f%%" % ({"fomc": "FOMC 议息日(公布前)", "cpi": "CPI 日且10点前上涨", "nfp": "非农日且10点前上涨",
              "cpi|nfp": "CPI/非农日且10点前上涨"}[ev], em // 60, em % 60, xem // 60, xem % 60, sp * 100), E, {}


def fam_xasset(D, B5, X):
    if X is None: return
    for em, st, rule in itertools.product((600, 630), ("VWAP", "0.5%"), ("risk-on", "SPY落后QQQ")):
        S = B5[B5.em == em].copy()
        key = list(zip(S.d, S.em))
        q, iw, tl = (X[s].reindex(key).to_numpy() for s in ("QQQ", "IWM", "TLT"))
        spy = (S.c / S.dopen - 1).to_numpy()
        m = (q > 0) & (iw > 0) & (tl < 0) & (S.c > S.vwap).to_numpy() if rule == "risk-on" else (q - spy > 0.002) & (S.c > S.vwap).to_numpy()
        S = S[m & (D.full.reindex(pd.to_datetime(S.d)).to_numpy() == True)]
        E = pd.DataFrame({"t": S.ts.values})
        if st == "VWAP": E["stop"] = S.vwap.to_numpy()
        else: E["stop_pct"] = 0.005
        yield "跨市场确认", "%s · %02d:%02d · 止损%s" % ({"risk-on": "QQQ/IWM涨+TLT跌(风险偏好)", "SPY落后QQQ": "SPY落后QQQ>0.2%"}[rule], em // 60, em % 60, st), E, {}


def fam_ai(B, tf):
    """v6 bottom signals held to the close -- out-of-sample probabilities exist only for validation/holdout."""
    P = pd.read_parquet(os.path.join(HERE, "cache", "pred_bottom.parquet"))
    P = P[(P.tf == tf) & (P.split != "train")]
    import feat6
    b = pd.read_parquet(os.path.join(DATA, "spy_%s.parquet" % tf)); low = b.l.to_numpy(float)
    atr = feat6._atr(b.h.to_numpy(float), low, b.c.to_numpy(float), 14); T = pd.to_datetime(b.t).values
    for thr, trail in itertools.product((0.55, 0.65), (False, True)):
        F = P[P.p6 >= thr].sort_values(["cand_i", "c"]).drop_duplicates("cand_i").sort_values("bar_j")
        ci = F.cand_i.to_numpy(); bj = F.bar_j.to_numpy()
        E = pd.DataFrame({"t": T[bj], "stop": low[ci] - 0.05 * atr[ci], "trail": 2 * atr[ci] if trail else np.nan})
        yield "底顶AI持有到收盘", "%s v6底 p≥%.2f · 持有到15:55%s（只有验证+保留期）" % (tf, thr, " · 2ATR移动止损" if trail else ""), E, {"trail": "trail" if trail else None}


# ================================================================== runner
def run_all():
    t0 = time.time(); M = load_1m(); D = daily(M); B5 = bars("5m"); B15 = bars("15m"); X = xasset()
    gens = [fam_dayhold(D), fam_orb(D, M), fam_gap(D), fam_vwap(B5), fam_cross(B5, "5分钟", D), fam_cross(B15, "15分钟", D),
            fam_donchian(B15), fam_rsi2(B5, "5分钟", D), fam_rsi2(B15, "15分钟", D), fam_boll(B5, "5分钟"), fam_boll(B15, "15分钟"),
            fam_kdj(B5, "5分钟", D), fam_kdj(B15, "15分钟", D), fam_afternoon(D, B5), fam_events(D, B5), fam_xasset(D, B5, X),
            fam_ai(B15, "15m"), fam_ai(None, "1h")]
    ndays = pd.Series(split_of(sorted(set(M["d"])))).value_counts().to_dict()
    res = []; trades = {}
    for g in gens:
        for fam, name, E, kw in g:
            if E is None or len(E) == 0: continue
            E = E.sort_values("t").reset_index(drop=True)
            if "stop_pct" in E:                                         # percentage stop -> distance at the signal bar's price level
                px = np.interp(E.t.values.astype("int64"), M["t"].astype("int64"), M["c"])
                E["stop_dist"] = E.stop_pct * px
            ex = kw.get("exit_sig"); trail = kw.get("trail")
            T = simulate(M, E, exit_sig=ex, trail=trail)
            if len(T) == 0: continue
            T["split"] = split_of(T.d)
            r = {"family": fam, "name": name}
            for sp in ("train", "val", "hold"): r[sp] = metrics(T[T.split == sp], ndays.get(sp, 1))
            res.append(r); trades[name] = T
        print("%-12s done, %d configs so far  %.0fs" % (fam, len(res), time.time() - t0), flush=True)
    return res, trades, ndays


if __name__ == "__main__":
    res, trades, ndays = run_all()
    N = len(res); z = float(__import__("statistics").NormalDist().inv_cdf(1 - 0.05 / N))
    for r in res:
        tr, va, ho = r["train"], r["val"], r["hold"]
        r["stage1"] = bool(tr.get("n", 0) >= 60 and tr.get("net12", -1) > 0 and tr.get("t", 0) >= 2.0)
        r["stage2"] = bool(r["stage1"] and va.get("n", 0) >= 20 and va.get("net12", -1) > 0)
        r["stage3"] = bool(r["stage2"] and ho.get("n", 0) >= 5 and ho.get("net12", -1) > 0)
        r["bonferroni"] = bool(tr.get("t", 0) >= z)
    out = {"n_configs": N, "bonferroni_t": round(z, 2), "ndays": ndays, "results": res}
    json.dump(out, open(os.path.join(HERE, "results", "factory.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.to_pickle(trades, os.path.join(HERE, "cache", "factory_trades.pkl"))
    fams = pd.DataFrame([{"family": r["family"], "s1": r["stage1"], "s2": r["stage2"], "s3": r["stage3"]} for r in res]).groupby("family").agg(["sum", "count"])
    print("\nconfigs %d | Bonferroni t >= %.2f" % (N, z)); print(fams)
    best = sorted(res, key=lambda r: r["train"].get("net12", -99) if r["train"].get("n", 0) >= 60 else -99, reverse=True)[:25]
    print("\nTOP 25 by TRAIN net12 (n>=60):")
    for r in best:
        f = lambda s: "n%4d %+6.2f t%+5.2f" % (s.get("n", 0), s.get("net12", 0), s.get("t", 0))
        print("%s %-10s %-48s | TR %s | VA %s | HO %s" % ("★" if r["stage3"] else ("☆" if r["stage2"] else " "), r["family"], r["name"][:48], f(r["train"]), f(r["val"]), f(r["hold"])))
