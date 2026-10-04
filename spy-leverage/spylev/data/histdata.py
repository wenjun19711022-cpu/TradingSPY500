"""S&P 500 index CFD minute bars 2011-2018 (HistData via github.com/FutureSharks/financial-data).

    python -m spylev.data.histdata          # download (~140 MB) and build data/bars/SPXCFD/1m

HistData documents its timestamps as EST without daylight saving, but checked against SPY's
daily open-to-close returns these files are New York local time (correlation 0.998 as local time,
0.88 if treated as fixed UTC-5), so they are read as America/New_York. We keep regular hours
(09:30-15:59) of NYSE sessions. There is no volume (an index CFD), so anything that
needs volume (VWAP, volume spikes) uses equal weights on this set. Prices are the S&P 500 index
(about 10x SPY); strategies here only use returns, so the scale does not matter. Missing minutes
(no quote change) are forward-filled within the session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spylev.calendar import nyse_holidays
from spylev.data import store

URL = "https://raw.githubusercontent.com/FutureSharks/financial-data/master/pyfinancialdata/data/stocks/histdata/SPXUSD/DAT_ASCII_SPXUSD_M1_{y}.csv"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "history" / "histdata"
YEARS = range(2011, 2019)
SYMBOL = "SPXCFD"


def download(years=YEARS, out: Path = RAW) -> list:
    import requests
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for y in years:
        f = out / f"SPXUSD_M1_{y}.csv"
        if not f.exists():
            r = requests.get(URL.format(y=y), timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        got.append(f)
    return got


def _read(f: Path) -> pd.DataFrame:
    d = pd.read_csv(f, sep=";", header=None, names=["ts", "open", "high", "low", "close", "volume"])
    ts = pd.DatetimeIndex(pd.to_datetime(d["ts"], format="%Y%m%d %H%M%S"))
    d.index = ts.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    d = d[d.index.notna()]
    return d[["open", "high", "low", "close"]].astype(float)


def build(years=YEARS) -> pd.DataFrame:
    files = download(years)
    raw = pd.concat([_read(f) for f in files]).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    mos = raw.index.hour * 60 + raw.index.minute - 570
    raw = raw[(mos >= 0) & (mos < 390) & (raw.index.dayofweek < 5)]
    hol = set()
    for y in years:
        hol |= nyse_holidays(y)
    days = []
    for day, g in raw.groupby(raw.index.normalize()):
        if day.strftime("%Y-%m-%d") in hol or len(g) < 300:
            continue
        full = pd.date_range(day + pd.Timedelta(minutes=570), periods=390, freq="1min")
        g = g.reindex(full)
        c = g["close"].ffill().bfill()
        g["close"] = c
        for k in ("open", "high", "low"):
            g[k] = g[k].fillna(c)
        g["volume"] = 1.0
        days.append(g)
    bars = pd.concat(days)
    bars.index.name = "ts"
    store.save(bars, SYMBOL, "1m")
    return bars


if __name__ == "__main__":
    b = build()
    print(len(b), "bars", b.index.normalize().nunique(), "sessions", b.index[0], b.index[-1])
