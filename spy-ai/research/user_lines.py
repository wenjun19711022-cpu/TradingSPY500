"""The user's hand-drawn zigzag (moomoo 1h chart, 2026-08-05..10-05, UNADJUSTED prices) -> exact pivots.
Digitised from the screenshot (x pixel -> 1h bar, y pixel -> price), then each vertex is snapped to the real extreme
(highest high for a top / lowest low for a bottom) within +-4 bars. Output: data/user_pivots.csv"""
import os, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# screenshot geometry: 08/06 label x=197, 10/02 label x=1689 (280 one-hour bars apart); price 788 @ y=284, 736 @ y=980
X0, DX = 197.0, (1689 - 197) / 280.0
Y0, P0, DY = 284.0, 788.0, (980 - 284) / 52.0
# (x, y, kind) read off the blue line, left to right
V = [(178, 455, "T"), (227, 553, "B"), (285, 462, "T"), (305, 497, "B"), (332, 462, "T"), (342, 530, "B"), (369, 463, "T"), (384, 497, "B"),
     (439, 405, "T"), (451, 448, "B"), (474, 419, "T"), (541, 575, "B"), (554, 507, "T"), (603, 630, "B"), (641, 567, "T"), (668, 620, "B"),
     (699, 571, "T"), (731, 597, "B"), (814, 487, "T"), (911, 645, "B"), (968, 478, "T"), (1110, 688, "B"), (1149, 593, "T"), (1271, 773, "B"),
     (1297, 625, "T"), (1307, 650, "B"), (1322, 615, "T"), (1335, 673, "B"), (1433, 463, "T"), (1474, 620, "B"), (1525, 495, "T"), (1580, 608, "B"),
     (1590, 557, "T"), (1601, 618, "B"), (1630, 545, "T"), (1670, 665, "B"), (1701, 497, "T"), (1730, 548, "B"), (1747, 448, "T")]


def raw_1h():
    p = os.path.join(HERE, "data", "spy_1h_raw.parquet")
    if os.path.exists(p): return pd.read_parquet(p)
    import moomoo as F
    q = F.OpenQuoteContext(host="127.0.0.1", port=11111); parts, key = [], None
    try:
        while True:
            ret, d, key = q.request_history_kline("US.SPY", start="2018-09-01", end=time.strftime("%Y-%m-%d"), ktype=F.KLType.K_60M,
                                                  autype=F.AuType.NONE, max_count=1000, page_req_key=key)
            assert ret == F.RET_OK, d; parts.append(d)
            if key is None: break
            time.sleep(0.55)
    finally:
        q.close()
    b = pd.concat(parts).rename(columns={"time_key": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})[["t", "o", "h", "l", "c", "v"]]
    b["t"] = b.t.astype(str); b = b.drop_duplicates("t").sort_values("t").reset_index(drop=True); b.to_parquet(p, index=False); return b


if __name__ == "__main__":
    b = raw_1h(); w = b[(b.t >= "2026-08-05") & (b.t <= "2026-10-05 23:59")].reset_index()
    # calibrate x -> bar with the two exact labelled pivots: 779.370 (2026-08-13 11:30, x=439) and 749.600 (2026-09-16 15:30, x=1271)
    ia = int(np.flatnonzero(w.t.to_numpy() == "2026-08-13 11:30:00")[0]); ib = int(np.flatnonzero(w.t.to_numpy() == "2026-09-16 15:30:00")[0])
    dx = (1271 - 439) / (ib - ia)
    rows = []
    for x, y, k in V:
        i = int(round(ia + (x - 439) / dx)); i = min(max(i, 0), len(w) - 1)
        lo, hi = max(0, i - 3), min(len(w), i + 4); seg = w.iloc[lo:hi]
        j = seg.h.idxmax() if k == "T" else seg.l.idxmin()
        rows.append({"kind": k, "t": w.t[j], "px": float(w.h[j] if k == "T" else w.l[j]), "drawn_px": round(P0 - (y - Y0) / DY, 2), "bar": int(w["index"][j])})
    # keep the sequence strictly alternating and in time order (a snapped duplicate keeps the more extreme point)
    out = []
    for r in sorted(rows, key=lambda r: r["bar"]):
        if out and out[-1]["kind"] == r["kind"]:
            better = r["px"] > out[-1]["px"] if r["kind"] == "T" else r["px"] < out[-1]["px"]
            if better: out[-1] = r
            continue
        if out and out[-1]["bar"] == r["bar"]: continue
        out.append(r)
    rows = out
    P = pd.DataFrame(rows)
    P["leg_pct"] = (P.px / P.px.shift(1) - 1) * 100
    P.to_csv(os.path.join(HERE, "data", "user_pivots.csv"), index=False)
    pd.set_option("display.width", 200); print(P.round(3).to_string(index=False))
    up = P[(P.kind == "T") & P.leg_pct.notna()].leg_pct
    print("\npivots %d | up-legs %d: mean %.2f%% median %.2f%% min %.2f%% max %.2f%% | down-legs mean %.2f%%" % (
        len(P), len(up), up.mean(), up.median(), up.min(), up.max(), P[P.kind == "B"].leg_pct.mean()))
    print("perfect long-only capture (every drawn bottom -> next drawn top): %.2f%% gross over %d trades" % (up.sum(), len(up)))
