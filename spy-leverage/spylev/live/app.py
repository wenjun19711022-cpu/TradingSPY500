"""Local alert server: one screen that says what to do.

    python -m spylev.live.app --source futu --open                 # live, moomoo OpenD on 127.0.0.1:11111
    python -m spylev.live.app --source replay --date 2023-08-17 --speed 30 --open
    python -m spylev.live.app --leverage 10                         # the leverage you want the math shown for

Open http://127.0.0.1:8765 . Endpoints: /state, /events (server-sent events), /bars,
POST /confirm (I bought), POST /close (I sold), POST /tv (TradingView alert webhook, optional).
Standard library only (no web framework to install).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.data import store
from spylev.data.history import spy_daily
from spylev.data.sources import load_ibkr_json
from spylev.live.engine import LiveEngine, screen_levels
from spylev.range.desk import RangeDesk, daily_rv
from spylev.scalp.signals import build
from spylev.ta import daily_levels

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "index.html"
LIVE_LOG = ROOT / "data" / "live"  # one CSV per day: what the screen said each minute + moomoo extras
PAGE_HEAD = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return str(o)


class Monitor:
    def __init__(self, feed, leverage: float, premarket: pd.DataFrame | None = None, log_dir: Path | None = None):
        self.feed = feed
        self.log_dir = log_dir
        daily, _ = spy_daily()
        fd = getattr(feed, "daily", None)  # OpenD daily bars keep the daily setup current
        if fd is not None and not fd.empty:
            daily = pd.concat([daily[daily.index < fd.index[0]], fd[["open", "high", "low", "close"]]])
        self.engine = LiveEngine(daily, leverage)
        self.track = self.engine.track_record()
        self.bars = getattr(feed, "history", pd.DataFrame())
        self.premarket = premarket
        self.state = {"status": "wait", "headline": "连接中…", "sub": feed.name, "alerts": [], "track": self.track,
                      "source": feed.name}
        self.subscribers: list = []
        self.lock = threading.Lock()
        self.tv_alerts: list = []
        # range desk: 10:00 forecast of the rest of the day + odds of a limit-buy ticket
        try:
            self.desk = RangeDesk.from_results(leverage)
        except FileNotFoundError:
            self.desk = None
        vix = load_ibkr_json(ROOT / "data" / "raw" / "ibkr" / "VIX_1d.json")["close"]
        fv = getattr(feed, "vix", None)
        if fv is not None and len(fv):
            vix = pd.concat([vix[vix.index < fv.index[0]], fv])
        self.vix = vix
        self.rv_extra = getattr(feed, "rv_extra", pd.Series(dtype=float))  # realized variance of days missing from the 1m history

    def on_bar(self, ts, row):
        with self.lock:
            row = row[["open", "high", "low", "close", "volume"]].astype(float)
            self.bars = pd.concat([self.bars, row.to_frame(ts).T]).sort_index()
            self.bars = self.bars[~self.bars.index.duplicated(keep="last")]
            self.refresh()
        self.broadcast()

    def refresh(self):
        """Re-evaluate the latest bar (called on every closed bar, and once at start-up)."""
        if self.bars.empty:
            return
        ts = self.bars.index[-1]
        window = self.bars[self.bars.index >= ts.normalize() - pd.Timedelta(days=12)]
        lv = daily_levels(window)
        if self.premarket is not None and not self.premarket.empty:
            lv = lv.join(self.premarket)
        x = build(window, lv)
        today = ts.normalize().tz_localize(None)
        setup = self.engine.daily_setup(pd.Timestamp(today))
        s = self.engine.evaluate(x, len(x) - 1, setup)
        s["levels"] = screen_levels(lv, x, len(x) - 1)
        s["alerts"] = (self.engine.alerts + self.tv_alerts)[-12:]
        s["source"] = self.feed.name
        s["extra"] = dict(getattr(self.feed, "extra", {}))
        s["user_leverage"] = self.engine.user_leverage
        s["track"] = self.track
        s["range"] = self.range_state(ts)
        self.state = s
        if self.log_dir is not None:
            self._log(s)

    def range_state(self, ts) -> dict | None:
        if self.desk is None:
            return None
        day = ts.normalize()
        today = self.bars[self.bars.index.normalize() == day]
        hist = self.bars[(self.bars.index.normalize() < day) & (self.bars.index >= day - pd.Timedelta(days=14))]
        rv = daily_rv(hist) if len(hist) else pd.Series(dtype=float)
        if len(self.rv_extra):
            rv = pd.concat([self.rv_extra[~self.rv_extra.index.isin(rv.index)], rv]).sort_index()
        d0 = day.tz_localize(None)
        prev = self.vix[self.vix.index < d0]
        vix_prev = float(prev.iloc[-1]) if len(prev) else float("nan")
        u = self.desk.update(today, rv, vix_prev)
        scr = self.desk.screen()
        return {"phase": u.get("phase"), "minutes_to_forecast": u.get("minutes_to_forecast"), "decision": u.get("decision"),
                "vix_stale_days": int((d0 - prev.index[-1]).days) if len(prev) else None, **(scr or {})}

    def _log(self, s: dict):
        """Keep the live record (capital flow and order-book imbalance have no history to backtest
        on; after a few months of these files they can be tested like everything else)."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            f = self.log_dir / f"{s['time'][:10]}.csv"
            c = s.get("checks", {})
            ok = s.get("tf_ok", {})
            ex = s.get("extra", {})
            row = [s["time"], f"{s['price']:.3f}", s["status"], f"{s.get('quality', 0):g}",
                   *(str(int(c.get(k, {}).get("score", 0))) for k in ("m5", "m3", "m1")),
                   *(str(int(bool(ok.get(k)))) for k in ("m5", "m3", "m1")),
                   str(ex.get("capital_flow_today", "")), str(ex.get("book_imbalance", ""))]
            op = ex.get("options") or {}
            rf = ((s.get("range") or {}).get("forecast") or {}).get("bands", {}).get("0.8", {})
            row += [str(op.get(k, "")) for k in ("expected_move", "call_wall", "put_wall", "gex_total", "flip", "pc_volume")]
            row += [str(rf.get("low", "")), str(rf.get("high", ""))]
            new_file = not f.exists()
            with f.open("a", encoding="utf-8") as fh:
                if new_file:
                    fh.write("time,price,status,quality,m5_score,m3_score,m1_score,m5_ok,m3_ok,m1_ok,capital_flow,book_imbalance,"
                             "opt_expected_move,call_wall,put_wall,gex_total,gamma_flip,pc_volume,range80_low,range80_high\n")
                fh.write(",".join(row) + "\n")
        except Exception as e:  # logging must never stop the monitor
            print("log error:", e)

    def broadcast(self):
        msg = ("data: " + json.dumps(self.state, ensure_ascii=False, default=_jsonable) + "\n\n").encode()
        for q in list(self.subscribers):
            try:
                q.append(msg)
            except Exception:
                pass

    def bars_json(self, n=240):
        b = self.bars[self.bars.index.normalize() == self.bars.index[-1].normalize()] if len(self.bars) else self.bars
        return [[int((t.tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds()), round(r.open, 3), round(r.high, 3),
                 round(r.low, 3), round(r.close, 3), int(r.volume)] for t, r in b.tail(n).iterrows()]


def make_handler(mon: Monitor):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE_HEAD + WEB.read_text(encoding="utf-8") + "</body></html>", "text/html; charset=utf-8")
            if self.path == "/state":
                return self._send(200, json.dumps(mon.state, ensure_ascii=False, default=_jsonable))
            if self.path.startswith("/bars"):
                return self._send(200, json.dumps(mon.bars_json(), default=_jsonable))
            if self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: list = []
                mon.subscribers.append(q)
                try:
                    self.wfile.write(("data: " + json.dumps(mon.state, ensure_ascii=False, default=_jsonable) + "\n\n").encode())
                    self.wfile.flush()
                    while True:
                        while q:
                            self.wfile.write(q.pop(0))
                            self.wfile.flush()
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        time.sleep(1)
                except Exception:
                    pass
                finally:
                    mon.subscribers.remove(q)
                return None
            return self._send(404, '{"error":"not found"}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            now = str(pd.Timestamp.now(tz="America/New_York"))[:16]
            with mon.lock:
                if self.path in ("/confirm", "/close"):
                    if self.path == "/confirm":
                        mon.engine.confirm_buy(float(body["price"]), float(body.get("leverage", 2)), float(body["stop"]),
                                               body.get("time", now))
                    else:
                        mon.engine.close_position()
                    mon.refresh()  # the screen switches to "hold" / back right away
                elif self.path == "/settings":
                    mon.engine.user_leverage = float(body.get("leverage", mon.engine.user_leverage))
                    mon.state["user_leverage"] = mon.engine.user_leverage
                elif self.path == "/tv":
                    mon.tv_alerts.append({"time": now, "type": "tv", "text": "TradingView: " + str(body.get("message", body))[:120]})
                    mon.state["alerts"] = (mon.engine.alerts + mon.tv_alerts)[-12:]
                else:
                    return self._send(404, '{"error":"not found"}')
            mon.broadcast()
            return self._send(200, '{"ok":true}')
    return H


OPEND_HELP = """
没连上 moomoo OpenD（{host}:{port}）。请按顺序检查：
  1. 下载并安装 OpenD：https://www.moomoo.com/download/OpenAPI （富途牛牛用户：https://www.futunn.com/download/OpenAPI）
  2. 打开 OpenD，用你的 moomoo 账号登录，保持窗口开着
  3. OpenD 的 API 端口是 11111（默认值）；如果改过，启动时加 --port 你的端口
  4. 需要美股行情权限（moomoo 里能看到 SPY 实时报价即可）
只想先看看效果：双击 START_REPLAY_DEMO，或直接用浏览器打开 web/demo.html。
"""


def opend_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["futu", "replay"], default="futu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11111, help="OpenD port")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--date", default="2023-08-17", help="replay date")
    ap.add_argument("--speed", type=float, default=30)
    ap.add_argument("--leverage", type=float, default=10)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    premarket = None
    if a.source == "replay":
        from spylev.data.minute_hist import build as build_minutes, premarket_levels
        bars = store.load("SPY", "1m")
        if bars.empty:
            print("首次运行：下载 2022-10..2024-12 分钟数据…")
            build_minutes()
            bars = store.load("SPY", "1m")
        premarket = premarket_levels()
        from spylev.live.feeds import ReplayFeed
        if pd.Timestamp(a.date) > bars.index[-1].tz_localize(None):  # recent days: the IBKR files
            bars = load_ibkr_json(ROOT / "data" / "raw" / "ibkr" / "SPY_1m_recent.json")
        feed = ReplayFeed(bars, a.date, a.speed)
        b5 = load_ibkr_json(ROOT / "data" / "raw" / "ibkr" / "SPY_5m_recent.json")
        feed.rv_extra = daily_rv(b5[b5.index.tz_localize(None) < pd.Timestamp(a.date)])
    else:
        if not opend_reachable(a.host, a.port):
            print(OPEND_HELP.format(host=a.host, port=a.port))
            raise SystemExit(2)
        from spylev.live.feeds import FutuFeed
        feed = FutuFeed(a.host, a.port)
        print(f"连接 moomoo OpenD {a.host}:{a.port} …")
        feed.connect()
    mon = Monitor(feed, a.leverage, premarket, log_dir=LIVE_LOG if a.source == "futu" else None)
    if a.source == "futu":
        with mon.lock:
            mon.refresh()
    srv = ThreadingHTTPServer(("127.0.0.1", a.web_port), make_handler(mon))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{a.web_port}"
    print(f"打开 {url}  （数据源：{feed.name}）")
    if a.open:
        webbrowser.open(url)
    feed.start(mon.on_bar)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
