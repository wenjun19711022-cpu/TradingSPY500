"""Incremental data refresh (read-only quote API): append the newest SPY K-lines to data/spy_<tf>.parquet, refresh daily VIX,
and download 5-minute history for cross-asset symbols (QQQ, IWM, TLT) into data/xa_<SYM>_5m.parquet."""
import os, sys, time, datetime as dt
import pandas as pd
import moomoo as F
import vixdata

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
KT = {"1m": F.KLType.K_1M, "3m": F.KLType.K_3M, "5m": F.KLType.K_5M, "15m": F.KLType.K_15M, "1h": F.KLType.K_60M}
TODAY = dt.date.today().strftime("%Y-%m-%d")


def pull(q, code, ktype, start, end):
    parts, key = [], None
    while True:
        for attempt in range(6):
            ret, d, key2 = q.request_history_kline(code, start=start, end=end, ktype=ktype, max_count=1000, page_req_key=key, extended_time=False)
            if ret == F.RET_OK: break
            print(code, "retry", d, flush=True); time.sleep(5 + 5 * attempt)
        else:
            raise RuntimeError(d)
        parts.append(d[["time_key", "open", "high", "low", "close", "volume", "turnover"]])
        key = key2
        if key is None: break
        time.sleep(0.52)
    b = pd.concat(parts, ignore_index=True).rename(columns={"time_key": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v", "turnover": "tv"})
    b["t"] = b.t.astype(str); return b


if __name__ == "__main__":
    q = F.OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        for tf, kt in KT.items():
            p = os.path.join(DATA, "spy_%s.parquet" % tf); old = pd.read_parquet(p)
            last_day = old.t.str[:10].max()
            new = pull(q, "US.SPY", kt, last_day, TODAY)
            # forward-adjusted prices: if a dividend went ex since the last download, the whole history must be re-pulled
            ov = old[old.t.str[:10] == last_day].set_index("t").c; nv = new.set_index("t").c.reindex(ov.index)
            if (abs(ov - nv) > 1e-6).any(): print(tf, "WARNING: adjusted history changed (ex-dividend) -> re-run dl_moomoo.py", flush=True)
            b = pd.concat([old[old.t.str[:10] < last_day], new]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
            b.to_parquet(p, index=False); print("SPY", tf, len(old), "->", len(b), "last", b.t.iloc[-1], flush=True)
        for sym in (sys.argv[1:] or ["QQQ", "IWM", "TLT"]):
            p = os.path.join(DATA, "xa_%s_5m.parquet" % sym)
            if os.path.exists(p): print(sym, "exists"); continue
            b = pull(q, "US." + sym, F.KLType.K_5M, "2018-09-01", TODAY).drop_duplicates("t").sort_values("t")
            b.to_parquet(p, index=False); print(sym, len(b), b.t.iloc[0], b.t.iloc[-1], flush=True)
    finally:
        q.close()
    v = vixdata.fetch("10y"); v.to_parquet(os.path.join(DATA, "vix_daily.parquet"), index=False); print("VIX", v.date.max())
