"""Download SPY regular-session K-lines (forward-adjusted, like the live watcher's get_cur_kline) with REAL volume/turnover
from moomoo OpenD. Read-only quote API. Output: data/spy_<tf>.parquet with t (bar END time, ET), o,h,l,c,v,tv."""
import os, sys, time, datetime as dt
import pandas as pd
import moomoo as F

HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
KT = {"1m": F.KLType.K_1M, "3m": F.KLType.K_3M, "5m": F.KLType.K_5M, "15m": F.KLType.K_15M, "1h": F.KLType.K_60M}
START = "2017-01-01"; END = dt.date.today().strftime("%Y-%m-%d")

q = F.OpenQuoteContext(host="127.0.0.1", port=11111)
try:
    for tf in (sys.argv[1:] or list(KT)):
        out = os.path.join(HERE, "data", "spy_%s.parquet" % tf)
        parts = []; key = None; n = 0; t0 = time.time()
        while True:
            for attempt in range(6):
                ret, d, key2 = q.request_history_kline("US.SPY", start=START, end=END, ktype=KT[tf], max_count=1000,
                                                       page_req_key=key, extended_time=False)
                if ret == F.RET_OK: break
                print(tf, "retry", attempt, d, flush=True); time.sleep(5 + 5 * attempt)
            else:
                raise RuntimeError("%s failed: %s" % (tf, d))
            parts.append(d[["time_key", "open", "high", "low", "close", "volume", "turnover"]]); n += 1
            if n % 50 == 0: print(tf, "pages", n, "last", d.time_key.iloc[-1], "%.0fs" % (time.time() - t0), flush=True)
            key = key2
            if key is None: break
            time.sleep(0.52)                                   # stay under 60 requests / 30 s
        b = pd.concat(parts, ignore_index=True).rename(columns={"time_key": "t", "open": "o", "high": "h", "low": "l", "close": "c",
                                                                 "volume": "v", "turnover": "tv"})
        b["t"] = b["t"].astype(str); b = b.drop_duplicates("t").sort_values("t").reset_index(drop=True)
        b.to_parquet(out, index=False)
        print("DONE", tf, len(b), b.t.iloc[0], b.t.iloc[-1], "%.0fs" % (time.time() - t0), flush=True)
finally:
    q.close()
