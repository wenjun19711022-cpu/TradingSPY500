"""Daily VIX / VIX3M / VIX9D closes from Yahoo's chart API (the OpenAPI has no index quotes).
Model features only ever use the PREVIOUS session's close, so there is no look-ahead in backtest or live."""
import json, os, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = {"vix": "%5EVIX", "vix3m": "%5EVIX3M", "vix9d": "%5EVIX9D"}


def _one(sym, rng):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%s?range=%s&interval=1d" % (sym, rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.loads(urllib.request.urlopen(req, timeout=15).read())["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    return pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(ts.strftime("%Y-%m-%d"))).dropna()


def fetch(rng="10y"):
    cols = {}
    for k, s in SYMS.items():
        for attempt in range(3):
            try:
                cols[k] = _one(s, rng); break
            except Exception:
                time.sleep(2)
    d = pd.DataFrame(cols); d.index.name = "date"
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.reset_index()


def prev_close_table(d):
    """date -> previous session's vix/vix3m/vix9d (what is known before that session opens)."""
    d = d.sort_values("date").copy()
    for k in SYMS: d[k + "_prev"] = d[k].shift(1)
    return d[["date"] + [k + "_prev" for k in SYMS]]


if __name__ == "__main__":
    d = fetch("10y"); d.to_parquet(os.path.join(HERE, "data", "vix_daily.parquet"), index=False)
    print(len(d), d.date.min(), d.date.max()); print(d.tail(4).to_string())
