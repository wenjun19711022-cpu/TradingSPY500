"""Minute-bar sources. Every source returns start-time-labelled RTH bars in ET.

Pick whichever you have access to (the crawler tries them in the order you give):

  futu     moomoo / Futu OpenD (pip install futu-api), the source used for the 2018-2026 data
           in the original dashboard. Needs OpenD running locally and the US quote right.
  alpaca   Alpaca Market Data v2 (free account works for historical SIP minute bars).
           env: APCA_API_KEY_ID, APCA_API_SECRET_KEY
  polygon  Polygon.io aggregates (5y minute history needs a paid plan). env: POLYGON_API_KEY
  ibkr     Interactive Brokers TWS / IB Gateway via ib_async (pip install ib_async)
  yahoo    yfinance — only the last ~30 days of 1m bars; good for topping up
  csv      any CSV/parquet export (FirstRate Data, Kibot, an old moomoo export, ...)
  ibkr_json  JSON saved from the IBKR MCP connector (`get_price_history`)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from spyopt.calendar import rth_mask, to_et
from spyopt.data.store import normalize


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize(df)
    return df[rth_mask(df.index).values]


class Source:
    name = "base"
    chunk_days = 30  # crawler requests this many calendar days per call

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


class FutuSource(Source):
    """moomoo / Futu OpenD. Futu labels minute bars by their END time; we shift to start."""
    name = "futu"

    def __init__(self, host="127.0.0.1", port=11111):
        from futu import OpenQuoteContext  # noqa: F401  (import check)
        self.host, self.port = host, port

    def fetch(self, symbol, start, end):
        from futu import AuType, KLType, OpenQuoteContext, RET_OK
        ctx = OpenQuoteContext(host=self.host, port=self.port)
        frames, key = [], None
        try:
            while True:
                ret, data, key = ctx.request_history_kline(
                    f"US.{symbol}", start=str(start.date()), end=str(end.date()),
                    ktype=KLType.K_1M, autype=AuType.NONE, max_count=1000, page_req_key=key)
                if ret != RET_OK:
                    raise RuntimeError(f"futu: {data}")
                frames.append(data)
                if key is None:
                    break
                time.sleep(0.5)  # stay well under OpenD's history-kline rate limit
        finally:
            ctx.close()
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        ts = pd.to_datetime(df["time_key"]) - pd.Timedelta(minutes=1)
        df = df.assign(ts=ts.dt.tz_localize("America/New_York"))
        return _rth(df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts"))


class AlpacaSource(Source):
    name = "alpaca"
    url = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, key=None, secret=None, feed="sip"):
        self.key = key or os.environ["APCA_API_KEY_ID"]
        self.secret = secret or os.environ["APCA_API_SECRET_KEY"]
        self.feed = feed

    def fetch(self, symbol, start, end):
        import requests
        params = {"symbols": symbol, "timeframe": "1Min", "limit": 10000, "adjustment": "raw",
                  "feed": self.feed, "start": str(start.date()),
                  "end": str((end + pd.Timedelta(days=1)).date())}
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        rows = []
        while True:
            r = requests.get(self.url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            js = r.json()
            rows += js.get("bars", {}).get(symbol, [])
            if not js.get("next_page_token"):
                break
            params["page_token"] = js["next_page_token"]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume", "vw": "vwap", "n": "trades"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return _rth(df.set_index("ts"))


class PolygonSource(Source):
    name = "polygon"

    def __init__(self, key=None):
        self.key = key or os.environ["POLYGON_API_KEY"]

    def fetch(self, symbol, start, end):
        import requests
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
               f"{start.date()}/{end.date()}")
        params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": self.key}
        rows = []
        while url:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            rows += js.get("results", [])
            url = js.get("next_url")
            params = {"apiKey": self.key}
            time.sleep(12 if "free" in os.environ.get("POLYGON_PLAN", "") else 0.2)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume", "vw": "vwap", "n": "trades"})
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return _rth(df.set_index("ts"))


class IBKRSource(Source):
    """TWS / IB Gateway. IB paces historical requests (~60 per 10 min), so 5 years of
    1-minute bars takes roughly an hour; the crawler is resumable."""
    name = "ibkr"
    chunk_days = 7

    def __init__(self, host="127.0.0.1", port=7497, client_id=17):
        from ib_async import IB  # noqa: F401
        self.host, self.port, self.client_id = host, port, client_id

    def fetch(self, symbol, start, end):
        from ib_async import IB, Stock, util
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        try:
            contract = Stock(symbol, "SMART", "USD")
            end_str = (end + pd.Timedelta(days=1)).strftime("%Y%m%d 00:00:00") + " US/Eastern"
            days = max(1, (end - start).days + 1)
            bars = ib.reqHistoricalData(contract, endDateTime=end_str, durationStr=f"{days} D",
                                        barSizeSetting="1 min", whatToShow="TRADES",
                                        useRTH=True, formatDate=2)
            time.sleep(10)  # pacing
        finally:
            ib.disconnect()
        if not bars:
            return pd.DataFrame()
        df = util.df(bars).rename(columns={"date": "ts", "average": "vwap", "barCount": "trades"})
        return _rth(df.set_index("ts"))


class YahooSource(Source):
    name = "yahoo"
    chunk_days = 7

    def fetch(self, symbol, start, end):
        import yfinance as yf
        df = yf.download(symbol, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                         interval="1m", auto_adjust=False, prepost=False, progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _rth(df)


class CSVSource(Source):
    """Load an existing export. Column names are matched case-insensitively; set
    `end_labelled=True` for files that label minute bars by their close time (moomoo)."""
    name = "csv"

    def __init__(self, path, tz="America/New_York", end_labelled=False):
        self.path, self.tz, self.end_labelled = Path(path), tz, end_labelled

    def fetch(self, symbol=None, start=None, end=None):
        p = self.path
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df.columns = [c.lower().strip() for c in df.columns]
        tcol = next(c for c in ("ts", "time_key", "datetime", "timestamp", "time", "date") if c in df.columns)
        ts = pd.to_datetime(df[tcol])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(self.tz)
        if self.end_labelled:
            ts = ts - pd.Timedelta(minutes=1)
        df = df.assign(ts=ts).set_index("ts")
        df = _rth(df)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start).tz_localize("America/New_York")]
        if end is not None:
            df = df[df.index < pd.Timestamp(end).tz_localize("America/New_York") + pd.Timedelta(days=1)]
        return df


def load_ibkr_json(path) -> pd.DataFrame:
    """Bars saved from the IBKR connector: arrays `time` (UTC start), open/high/low/close[/volume]."""
    d = json.loads(Path(path).read_text())
    df = pd.DataFrame({k: d[k] for k in ("open", "high", "low", "close", "volume") if k in d},
                      index=pd.to_datetime(d["time"], utc=True))
    df.index = to_et(df.index)
    if d.get("chart_step", 0) >= 86400:  # daily bars: index by session date
        df.index = pd.DatetimeIndex(df.index.date)
        df.index.name = "date"
        return df.astype(float)
    return normalize(df)


SOURCES = {"futu": FutuSource, "alpaca": AlpacaSource, "polygon": PolygonSource,
           "ibkr": IBKRSource, "yahoo": YahooSource}
