"""Bar feeds for the live monitor: moomoo / Futu OpenD (real time) and a historical replay.

Both call `on_bar(df_row)` once per CLOSED regular-hours 1-minute bar (start-time labelled,
ET). OpenD pushes the forming bar repeatedly; a bar counts as closed when a newer minute
arrives (or about 70 s after its minute ends, for the last bar before a quiet spell).
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from spylev.calendar import rth_mask


class ReplayFeed:
    def __init__(self, bars: pd.DataFrame, date: str, speed: float = 30.0):
        d = pd.Timestamp(date)
        self.history = bars[bars.index.normalize().tz_localize(None) < d]
        self.today = bars[bars.index.normalize().tz_localize(None) == d]
        self.speed = speed
        self.name = f"回放 {date}（{speed:g} 倍速）"

    def connect(self):
        pass

    def start(self, on_bar):
        def loop():
            for ts, row in self.today.iterrows():
                on_bar(ts, row)
                time.sleep(60.0 / self.speed)
        threading.Thread(target=loop, daemon=True).start()


class FutuFeed:
    """moomoo OpenD. Needs OpenD running and logged in, with US stock quote permission.

    Extra data it collects when available (shown on screen as "参考", not used for orders):
      capital flow (get_capital_flow: net inflow today), order-book imbalance (ORDER_BOOK),
      last price / bid / ask (QUOTE), the nearest-expiry option chain (expected move, open-interest
      walls, dealer gamma; spylev.live.options) and daily VIX closes."""

    def __init__(self, host="127.0.0.1", port=11111, code="US.SPY", days=14):
        from futu import OpenQuoteContext  # noqa: F401
        self.host, self.port, self.code, self.days = host, port, code, days
        self.name = f"moomoo OpenD {host}:{port}"
        self.extra = {}
        self._last = None

    def _history(self, ctx):
        from futu import AuType, KLType, RET_OK
        end = pd.Timestamp.now(tz="America/New_York")
        start = end - pd.Timedelta(days=self.days)
        frames, key = [], None
        while True:
            ret, data, key = ctx.request_history_kline(self.code, start=str(start.date()), end=str(end.date()),
                                                       ktype=KLType.K_1M, autype=AuType.NONE, max_count=1000, page_req_key=key)
            if ret != RET_OK:
                raise RuntimeError(data)
            frames.append(data)
            if key is None:
                break
        df = pd.concat(frames)
        return self._norm(df)

    def _daily(self, ctx, days=420):
        """Unadjusted daily bars, so the daily RSI2 setup is current even if the repo's copy is old."""
        from futu import AuType, KLType, RET_OK
        end = pd.Timestamp.now(tz="America/New_York")
        frames, key = [], None
        while True:
            ret, data, key = ctx.request_history_kline(self.code, start=str((end - pd.Timedelta(days=days)).date()),
                                                       end=str(end.date()), ktype=KLType.K_DAY, autype=AuType.NONE,
                                                       max_count=1000, page_req_key=key)
            if ret != RET_OK:
                raise RuntimeError(data)
            frames.append(data)
            if key is None:
                break
        df = pd.concat(frames)
        df.index = pd.DatetimeIndex(pd.to_datetime(df["time_key"]).dt.normalize(), name="date")
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def connect(self):
        from futu import OpenQuoteContext
        self.ctx = OpenQuoteContext(host=self.host, port=self.port)
        self.history = self._history(self.ctx)
        self.daily = self._daily(self.ctx)
        self.vix = self._vix(self.ctx)

    def _vix(self, ctx):
        """Daily VIX closes (index code US..VIX). Needs US index quotes; None if not available,
        and the monitor then falls back to data/raw/ibkr/VIX_1d.json."""
        try:
            from futu import AuType, KLType, RET_OK
            end = pd.Timestamp.now(tz="America/New_York")
            ret, data, _ = ctx.request_history_kline("US..VIX", start=str((end - pd.Timedelta(days=20)).date()), end=str(end.date()),
                                                     ktype=KLType.K_DAY, autype=AuType.NONE, max_count=100)
            if ret != RET_OK or data.empty:
                self.extra["vix_note"] = "moomoo 没有 VIX 行情权限，用本地 VIX 文件"
                return None
            return pd.Series(data["close"].astype(float).values, index=pd.DatetimeIndex(pd.to_datetime(data["time_key"]).dt.normalize()))
        except Exception as e:  # optional
            self.extra["vix_note"] = str(e)[:120]
            return None

    @staticmethod
    def _norm(df):
        ts = pd.to_datetime(df["time_key"]) - pd.Timedelta(minutes=1)  # futu labels by bar END
        out = df.assign(ts=ts.dt.tz_localize("America/New_York")).set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float)
        return out[rth_mask(out.index).values]

    def start(self, on_bar):
        from futu import CurKlineHandlerBase, RET_OK, SubType
        if not hasattr(self, "ctx"):
            self.connect()
        ctx = self.ctx
        feed = self

        class Handler(CurKlineHandlerBase):
            def on_recv_rsp(self, rsp_pb):
                ret, data = super().on_recv_rsp(rsp_pb)
                if ret != RET_OK or data is None or data.empty:
                    return ret, data
                bars = feed._norm(data)
                for ts, row in bars.iterrows():
                    if feed._last is not None and ts > feed._last[0]:
                        on_bar(*feed._last)
                    feed._last = (ts, row)
                return ret, data

        ctx.set_handler(Handler())
        ret, err = ctx.subscribe([self.code], [SubType.K_1M, SubType.QUOTE, SubType.ORDER_BOOK], subscribe_push=True)
        if ret != RET_OK:
            raise RuntimeError(err)

        def extras():
            n = 0
            while True:
                if n % 15 == 0:  # option chain every ~5 minutes
                    try:
                        from spylev.live.options import fetch_futu
                        self.extra["options"] = fetch_futu(ctx, self.code)
                    except Exception as e:
                        self.extra["options"] = {"error": str(e)[:120]}
                n += 1
                try:
                    ret, cf = ctx.get_capital_flow(self.code)
                    if ret == RET_OK and not cf.empty:
                        self.extra["capital_flow_today"] = float(cf["in_flow"].iloc[-1])
                    ret, ob = ctx.get_order_book(self.code, num=10)
                    if ret == RET_OK:
                        bid = sum(v for _, v, *_ in ob["Bid"])
                        ask = sum(v for _, v, *_ in ob["Ask"])
                        self.extra["book_imbalance"] = (bid - ask) / max(bid + ask, 1)
                except Exception as e:  # extras are optional
                    self.extra["error"] = str(e)[:120]
                # flush the last bar if no newer minute has arrived 70 s after it ended
                if feed._last is not None:
                    age = pd.Timestamp.now(tz="America/New_York") - feed._last[0]
                    if age > pd.Timedelta(seconds=130):
                        on_bar(*feed._last)
                        feed._last = None
                time.sleep(20)
        threading.Thread(target=extras, daemon=True).start()
