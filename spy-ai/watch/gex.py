"""Options analytics from an option snapshot table: call/put walls, zero-gamma level, ATM IV, put/call ratios.
Convention (same as most GEX dashboards incl. moomoo's Gamma 敞口): calls +, puts -, dealer GEX in $ per 1% move."""
import math
import numpy as np
import pandas as pd

SQ2PI = math.sqrt(2 * math.pi)


def _gamma(S, K, T, iv):
    iv = np.maximum(iv, 1e-4); T = np.maximum(T, 1 / (365 * 24))          # floor: 1 hour
    d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / SQ2PI / (S * iv * np.sqrt(T))


def normalise(df):
    """Expects columns: strike, type ('CALL'/'PUT'), expiry (date str), oi, iv, gamma, volume. IV in % or decimal."""
    d = df.copy()
    d["type"] = d["type"].astype(str).str.upper().str[0].map({"C": "CALL", "P": "PUT"})
    for c_ in ("strike", "oi", "iv", "gamma", "volume"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    if d["iv"].median() > 3: d["iv"] = d["iv"] / 100.0                        # 25.17 -> 0.2517
    return d.dropna(subset=["strike", "type"])


def analyse(df, spot, now_et, mult=100):
    """df = normalise()d option snapshot. Returns dict with walls/zero-gamma for 0DTE and all expiries in df."""
    out = {"spot": spot}
    df = df.copy()
    exp_dt = pd.to_datetime(df["expiry"]) + pd.Timedelta(hours=16)
    df["T"] = np.maximum((exp_dt - pd.Timestamp(now_et).tz_localize(None)).dt.total_seconds() / (365 * 24 * 3600), 1 / (365 * 24))
    sign = np.where(df["type"] == "CALL", 1.0, -1.0)
    g = df["gamma"].where(df["gamma"].notna() & (df["gamma"] > 0), _gamma(spot, df["strike"], df["T"], df["iv"].fillna(0.2)))
    df["gex"] = sign * g * df["oi"].fillna(0) * mult * spot ** 2 * 0.01
    today = pd.Timestamp(now_et).strftime("%Y-%m-%d")
    for tag, part in (("0dte", df[df["expiry"].astype(str).str[:10] == today]), ("all", df)):
        if part.empty:
            continue
        by = part.groupby(["strike", "type"])["gex"].sum().unstack(fill_value=0.0)
        calls = by.get("CALL", pd.Series(dtype=float)); puts = by.get("PUT", pd.Series(dtype=float))
        res = {"net_gex_bn": round(float(part["gex"].sum()) / 1e9, 3)}
        if len(calls) and calls.max() > 0: res["call_wall"] = float(calls.idxmax())
        if len(puts) and puts.min() < 0: res["put_wall"] = float(puts.idxmin())
        # zero gamma: total GEX re-computed on a spot grid with Black-Scholes gamma
        grid = spot * np.linspace(0.95, 1.05, 201)
        sgn = np.where(part["type"] == "CALL", 1.0, -1.0); K = part["strike"].to_numpy(); T = part["T"].to_numpy()
        iv = part["iv"].fillna(0.2).to_numpy(); oi = part["oi"].fillna(0).to_numpy()
        prof = np.array([(sgn * _gamma(s, K, T, iv) * oi * mult * s ** 2 * 0.01).sum() for s in grid])
        cross = np.flatnonzero(np.sign(prof[:-1]) != np.sign(prof[1:]))
        if len(cross):
            k = cross[np.argmin(np.abs(grid[cross] - spot))]
            x0, x1, y0, y1 = grid[k], grid[k + 1], prof[k], prof[k + 1]
            res["zero_gamma"] = round(float(x0 - y0 * (x1 - x0) / (y1 - y0)), 2)
        res["atm_iv"] = _atm_iv(part[part["expiry"] == part["expiry"].min()], spot)   # one expiry only (mixing expiries is noise)
        cv = part.loc[part["type"] == "CALL", "volume"].sum(); pv = part.loc[part["type"] == "PUT", "volume"].sum()
        co = part.loc[part["type"] == "CALL", "oi"].sum(); po = part.loc[part["type"] == "PUT", "oi"].sum()
        res["pc_volume"] = round(float(pv / cv), 2) if cv else None
        res["pc_oi"] = round(float(po / co), 2) if co else None
        out[tag] = res
    nxt = df[df["expiry"].astype(str).str[:10] > today]          # nearest expiry after today: steadier "VIX" proxy than 0DTE,
    if not nxt.empty:                                            # whose IV balloons into the close as T -> 0
        e = nxt["expiry"].min()
        out["iv_next"] = {"expiry": str(e)[:10], "atm_iv": _atm_iv(nxt[nxt["expiry"] == e], spot)}
    return out


def _atm_iv(part, spot):
    """Mean IV (%) of the call+put at the two strikes nearest to spot, within ONE expiry."""
    atm = part.iloc[(part["strike"] - spot).abs().argsort()[:4]]
    return round(float(atm["iv"].mean() * 100), 2) if atm["iv"].notna().any() else None


def context_line(ox, price, atr):
    """One-line human summary for pushes."""
    if not ox or "0dte" not in ox and "all" not in ox:
        return "期权数据：无"
    r = ox.get("0dte") or ox.get("all"); parts = []
    for k, name in (("put_wall", "看跌墙"), ("call_wall", "看涨墙")):
        if r.get(k) is not None:
            dist = (r[k] - price) / atr if atr else 0.0
            parts.append("%s %.0f（%s %.1f ATR）" % (name, r[k], "上方" if dist > 0 else "下方", abs(dist)))
    if r.get("zero_gamma"):
        parts.append("零Gamma %.2f，价格在%s（%s）" % (r["zero_gamma"], "上方" if price >= r["zero_gamma"] else "下方",
                                                   "正Gamma，偏震荡回归" if price >= r["zero_gamma"] else "负Gamma，趋势易延续"))
    nx = ox.get("iv_next") or {}
    if nx.get("atm_iv") is not None: parts.append("平值IV %.1f%%（%s 到期）" % (nx["atm_iv"], nx["expiry"][5:]))
    elif r.get("atm_iv") is not None: parts.append("平值IV %.1f%%" % r["atm_iv"])
    return "；".join(parts)
