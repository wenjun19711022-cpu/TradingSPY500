"""
SPY 盯盘机器人 —— moomoo OpenD 实时行情 + 底顶AI v5 + 期权墙/零Gamma + 资金分布 → 微信推送 + 日志 + 收盘复盘。
只调用行情接口（OpenQuoteContext），从不创建交易连接、从不下单。

运行：  启动盯盘.bat        （或  py -3 spy_watch.py）
测试：  py -3 replay_test.py （用历史数据模拟一整天，不需要 OpenD）
"""
import datetime as dt, json, os, sys, time, traceback
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v5core, gex, pushers, report, zigzag_ai
from store import Store

ET = ZoneInfo("America/New_York")
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}
TF_NAME = {"1m": "1分钟", "3m": "3分钟", "5m": "5分钟", "15m": "15分钟", "1h": "1小时"}


def load_cfg():
    return json.load(open(os.path.join(HERE, "watch_config.json"), encoding="utf-8"))


def opend_needs_disclaimer():
    """True if the newest OpenD gateway log says the API questionnaire/agreement is still pending."""
    import glob
    logs = sorted(glob.glob(os.path.expandvars(r"%APPDATA%\com.moomoo.OpenD\Log\GTWLog_*.log")), key=os.path.getmtime)
    if not logs: return False
    try:
        s = open(logs[-1], "rb").read().decode("utf-8", "ignore")
    except OSError:
        return False
    bad = s.rfind("UnAgreeDisclaimer")
    good = max(s.rfind("bAgreeDisclaimer  =  1"), s.rfind("ProgramStatusType_Ready"))
    return bad > good                                  # the most recent status wins


PROBE = r"""
import sys
try:
    import moomoo as F
except ImportError:
    import futu as F
q = F.OpenQuoteContext(host=sys.argv[1], port=int(sys.argv[2]))
ret, d = q.get_global_state()
print("PROBE_OK" if ret == F.RET_OK else "PROBE_ERR %s" % d, flush=True)
q.close()
"""


def probe_opend(host, port, timeout=45):
    """Try the real API handshake in a child process so a hung OpenD can never freeze the watcher."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-c", PROBE, host, str(port)], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="ignore",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = r.stdout + r.stderr
        return ("PROBE_OK" in out), (out.strip().splitlines() or [""])[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "握手 %d 秒无响应" % timeout


# ------------------------------------------------------------------------------------------ live data (moomoo OpenD)
class LiveSource:
    def __init__(self, cfg):
        try:
            import moomoo as F
        except ImportError:
            import futu as F
        self.F = F; self.cfg = cfg; self.code = cfg.get("code", "US.SPY"); self.q = None
        host, port = cfg.get("opend_host", "127.0.0.1"), int(cfg.get("opend_port", 11111))
        import socket                                   # the SDK retries forever if OpenD is down -> probe the port first
        try:
            socket.create_connection((host, port), timeout=2).close()
        except OSError:
            raise RuntimeError("OpenD 没有在 %s:%d 运行" % (host, port))
        if opend_needs_disclaimer():
            raise RuntimeError("OpenD 已登录，但需要先完成 moomoo API 问卷评估和协议确认")
        ok, why = probe_opend(host, port)
        if not ok:
            raise RuntimeError("OpenD 没有响应：%s" % why)
        self.q = F.OpenQuoteContext(host=host, port=port, is_async_connect=True)
        self.kl = {tf: self._const(F.KLType, tf) for tf in TF_MIN}
        subs = [self._const(F.SubType, tf) for tf in TF_MIN] + [F.SubType.QUOTE]
        t0 = time.time(); ret, msg = -1, ""
        while time.time() - t0 < 30:
            ret, msg = self.q.subscribe([self.code], subs, subscribe_push=False)
            if ret == F.RET_OK: break
            time.sleep(3)
        if ret != F.RET_OK:
            self.close()
            hint = "（OpenD 提示需先完成 moomoo API 问卷评估和协议确认）" if opend_needs_disclaimer() else "（检查 OpenD 是否已登录）"
            raise RuntimeError("OpenD 没有响应：%s%s" % (msg, hint))
        self._chain = None; self._chain_day = None; self._chain_t = 0

    @staticmethod
    def _const(enum, tf):
        m = TF_MIN[tf]
        for name in ("K_%dM" % m, "K_%dMIN" % m, "K_%dMin" % m):
            if hasattr(enum, name): return getattr(enum, name)
        raise AttributeError("%s 里找不到 %d 分钟常量" % (enum, m))

    def now_et(self):
        return dt.datetime.now(ET)

    def sleep(self, s):
        time.sleep(s)

    def klines(self, tf, n=1000):
        ret, d = self.q.get_cur_kline(self.code, n, self.kl[tf])
        if ret != self.F.RET_OK: raise RuntimeError("K线失败：%s" % d)
        return pd.DataFrame({"t": d["time_key"].astype(str), "o": d["open"], "h": d["high"], "l": d["low"], "c": d["close"], "v": d["volume"],
                             "tv": d["turnover"] if "turnover" in d else d["volume"] * d["close"]})

    def options(self, spot, now):
        oc = self.cfg.get("options", {}); F = self.F
        day = now.strftime("%Y-%m-%d")
        if self._chain is None or self._chain_day != day or time.time() - self._chain_t > 1800:
            end = (now + dt.timedelta(days=int(oc.get("max_days", 7)))).strftime("%Y-%m-%d")
            ret, ch = self.q.get_option_chain(code=self.code, start=day, end=end)
            if ret != F.RET_OK: raise RuntimeError("期权链失败：%s" % ch)
            self._chain, self._chain_day, self._chain_t = ch, day, time.time()
        ch = self._chain
        rng = float(oc.get("strike_range_pct", 5)) / 100
        ch = ch[(ch["strike_price"] >= spot * (1 - rng)) & (ch["strike_price"] <= spot * (1 + rng))]
        rows = []
        codes = ch["code"].tolist()
        for i in range(0, len(codes), 400):
            ret, s = self.q.get_market_snapshot(codes[i:i + 400])
            if ret != F.RET_OK: raise RuntimeError("期权快照失败（多半是没有美股期权行情权限）：%s" % s)
            rows.append(s)
        if not rows: return None
        s = pd.concat(rows)                              # the snapshot already carries type / strike / expiry
        df = pd.DataFrame({"strike": s["option_strike_price"], "type": s["option_type"], "expiry": s["strike_time"].astype(str).str[:10],
                           "oi": s["option_open_interest"], "iv": s["option_implied_volatility"],
                           "gamma": s["option_gamma"], "volume": s["volume"]})
        return gex.analyse(gex.normalise(df), spot, now)

    def flows(self):
        ret, d = self.q.get_capital_distribution(self.code)
        if ret != self.F.RET_OK: raise RuntimeError("资金分布失败：%s" % d)
        r = d.iloc[0]
        keys = ["capital_in_super", "capital_in_big", "capital_in_mid", "capital_in_small",
                "capital_out_super", "capital_out_big", "capital_out_mid", "capital_out_small"]
        out = {k: float(r[k]) for k in keys if k in d}
        out["update_time"] = str(r.get("update_time", ""))
        return out

    def daily(self, n_days=420):
        """completed daily bars (forward-adjusted, like the minute bars) for the swing rule"""
        now = dt.datetime.now(ET); start = (now - dt.timedelta(days=int(n_days * 1.5))).strftime("%Y-%m-%d")
        ret, d, _ = self.q.request_history_kline(self.code, start=start, end=now.strftime("%Y-%m-%d"), ktype=self.F.KLType.K_DAY, max_count=1000)
        if ret != self.F.RET_OK: raise RuntimeError("日线失败：%s" % d)
        D = pd.DataFrame({"o": d["open"].values, "h": d["high"].values, "l": d["low"].values, "c": d["close"].values},
                         index=d["time_key"].astype(str).str[:10].values)
        return D[D.index < now.strftime("%Y-%m-%d")]

    def close(self):
        try:
            if self.q is not None: self.q.close()
        except Exception: pass


# ------------------------------------------------------------------------------------------ helpers
def closed_rth(df, tf, now):
    """drop the forming bar, keep regular session only; works whether time_key is bar start or bar end."""
    t = pd.to_datetime(df["t"])
    hm = t.dt.hour * 60 + t.dt.minute
    end_labeled = (hm == 960).any() or (hm.min() >= 570 + TF_MIN[tf] and (hm == 570).sum() == 0)
    rth = (hm > 570) & (hm <= 960) if end_labeled else (hm >= 570) & (hm < 960)
    d = df[rth.to_numpy()].reset_index(drop=True)
    now_min = now.hour * 60 + now.minute
    session_over = now_min > 960 or (now_min == 960 and now.second >= 20)
    if len(d) and d["t"].iloc[-1][:10] == now.strftime("%Y-%m-%d") and not session_over:
        d = d.iloc[:-1]                                  # today's last row is still forming during the session
    return d


def flow_line(fl):
    if not fl: return ""
    sup = fl.get("capital_in_super", 0) - fl.get("capital_out_super", 0)
    big = fl.get("capital_in_big", 0) - fl.get("capital_out_big", 0)
    tot = sum(fl.get("capital_in_" + k, 0) - fl.get("capital_out_" + k, 0) for k in ("super", "big", "mid", "small"))
    f = lambda x: ("%+.2f亿" % (x / 1e8))
    return "资金：特大单 %s，大单 %s，合计 %s" % (f(sup), f(big), f(tot))


def handle_event(cfg, store, now, tf, side, kind, bar_t, r, j, c, ox, fl, push, verbose=True, p=None, extra=None, stats=""):
    if store.seen(tf, side, kind, bar_t): return
    p = float(r["p"][j]) if p is None else float(p)
    word = ("底" if side == "bottom" else "顶") if kind == "early" else ("√确认底" if side == "bottom" else "√确认顶")
    title = ("SPY %s %s %s" % (TF_NAME[tf], word, ("%d%%" % round(p)) if kind == "early" else "")).strip()
    late = now.hour * 60 + now.minute - (int(bar_t[11:13]) * 60 + int(bar_t[14:16])) if bar_t[:10] == now.strftime("%Y-%m-%d") else 0
    if late >= 2: title += "（补报，晚 %d 分钟）" % late          # caught up after a data outage / late start
    lines = ["**%s**　收 %.2f　%s %.2f　ATR %.2f" % (bar_t[11:16], c[j], "低点" if side == "bottom" else "高点", r["ext"][j], r["atr"][j]),
             gex.context_line(ox, c[j], r["atr"][j]), flow_line(fl), stats or v5core.stats_text(tf, side)]
    body = "\n\n".join(x for x in lines if x)
    pushed = pushers.send(cfg, title, body) if push else False
    ctx = {"options": ox, "flows": fl}; ctx.update(extra or {})
    store.event(now.strftime("%Y-%m-%d"), tf, side, kind, bar_t, p, float(c[j]), float(r["ext"][j]), float(r["atr"][j]), ctx, pushed)
    if verbose: print(now.strftime("%H:%M:%S"), title, "| 已推送" if pushed else "")


# ------------------------------------------------------------------------------------------ main loop
def run(cfg, src, store, poll=None, verbose=True, v6=None, swing=None):
    th = float(cfg.get("threshold", 55)); tfs = cfg.get("timeframes", list(TF_MIN))
    push_tfs = set(cfg.get("push_timeframes", ["5m", "15m"])); push_ok = bool(cfg.get("push_confirm", False))
    push_until = cfg.get("push_until_et", "15:55")              # bars ending at/after this can't be traded (flat by 15:55): log, don't push
    poll = poll or float(cfg.get("poll_sec", 5)); last = {}; ox = None; fl = None; t_ox = t_fl = -1e9
    first_pass = set(); spot = None
    down_since = None; down_warned = False; last_err = None    # data-outage tracking
    while True:
        now = src.now_et(); now_min = now.hour * 60 + now.minute
        if now.weekday() >= 5:
            print("周末，不开盘。"); return
        if now_min < 565:                                  # before 09:25
            src.sleep(min(60, (565 - now_min) * 60)); continue
        try:
            today = now.strftime("%Y-%m-%d")
            need = list(dict.fromkeys(list(tfs) + (["15m", "1h"] if v6 is not None else [])))
            D = {tf: closed_rth(src.klines(tf, 1000), tf, now) for tf in need}     # all windows first (v6 uses 15m/1h state)
            if v6 is not None: v6.set_day(today)
            if swing is not None and swing.day != today:
                try:
                    swing.on_day(today, src.daily())
                except Exception as e:
                    print(now.strftime("%H:%M:%S"), "波段：日线读取失败，今天不判断抄底日：", e); swing.day = today
            for tf in tfs:
                d = D[tf]
                if len(d) < v5core.WARMUP + 5: continue
                if last.get(tf) == d["t"].iloc[-1]: continue
                if tf not in first_pass:                     # first look: store today's bars, evaluate only the latest bar
                    store.bars(tf, d[d["t"].str[:10] == today][["t", "o", "h", "l", "c", "v"]].itertuples(index=False, name=None))
                    new_j = [len(d) - 1]
                else:
                    new_j = [j for j in range(len(d)) if d["t"].iloc[j] > last[tf]]    # every bar since last look (outage catch-up)
                    store.bars(tf, d.iloc[new_j][["t", "o", "h", "l", "c", "v"]].itertuples(index=False, name=None))
                last[tf] = d["t"].iloc[-1]
                o, h, l, c = (d[k].to_numpy(float) for k in ("o", "h", "l", "c"))
                if tf == tfs[0] or spot is None: spot = float(c[-1])
                sig = {side: v5core.side_signals(o, h, l, c, side, th) for side in ("bottom", "top")}
                js = [j for j in new_j if d["t"].iloc[j][:10] == today]
                s6 = {}
                if v6 is not None and js:
                    try:
                        s6 = {side: v6.signals(D, tf, side, th, js) for side in ("bottom", "top")}
                    except Exception as e:
                        print(now.strftime("%H:%M:%S"), "v6 计算失败，这一轮改用 v5：", e); s6 = {}
                for j in js:                                            # only today's bars produce events
                    for side, r in sig.items():
                        x = s6[side].get(j) if s6 else None
                        early, pe = (bool(x and x[1]), x[0] if x else None) if s6 else (bool(r["fire"][j]), float(r["p"][j]))
                        extra = {"model": "v6" if s6 else "v5", "p5": round(float(r["p"][j]), 2), "p6": round(x[0], 2) if x else None}
                        for kind, flag in (("early", early), ("confirm", r["ok"][j])):
                            if flag:
                                handle_event(cfg, store, now, tf, side, kind, d["t"].iloc[j], r, j, c, ox, fl,
                                             push=(tf in first_pass and tf in push_tfs and (kind == "early" or push_ok)
                                                   and d["t"].iloc[j][11:16] < push_until), verbose=verbose,
                                             p=pe if kind == "early" else None, extra=extra, stats=v6.stats_text(tf, side) if s6 else "")
                                if swing is not None and kind == "early" and _fresh(d["t"].iloc[j], now):
                                    swing.on_signal(tf, side, float(pe), float(c[j]), d["t"].iloc[j], now)
                zc = cfg.get("zigzag_push", {})
                if zc.get("enabled") and tf == zc.get("tf", "15m") and js and tf in first_pass:     # 折线AI = the moomoo main-chart marks
                    zb, zs, pull, rise = zigzag_ai.marks(h, l, c, sig["bottom"]["fire"], sig["top"]["fire"], int(zc.get("N", 56)), float(zc.get("D", 0.3)))
                    for j in js:
                        if not _fresh(d["t"].iloc[j], now) or d["t"].iloc[j][11:16] >= push_until: continue
                        if zb[j]:
                            e = float(c[j]); liq = lambda L: e * (1 - 1 / L) / 0.99
                            pushers.send(cfg, "SPY 折线买点 %.2f（%s）" % (e, TF_NAME[tf]), "%s 收 %.2f，低点 %.2f，比近 %d 根最高价回落 %.2f%%。\n\n爆仓价：50x %.2f · 20x %.2f。"
                                         "这是和你手画的线最像的买点标记（准确约一半，照着每个都做 50 倍历史上是亏的）；真正的进场以“波段：进场”为准。" % (
                                         d["t"].iloc[j][11:16], e, float(l[j]), int(zc.get("N", 56)), pull[j], liq(50), liq(20)))
                        if zs[j]:
                            pushers.send(cfg, "SPY 折线卖点 %.2f（%s）" % (float(c[j]), TF_NAME[tf]), "%s 收 %.2f，高点 %.2f，比近 %d 根最低价高 %.2f%%。和你手画的顶最像的卖点标记。" % (
                                         d["t"].iloc[j][11:16], float(c[j]), float(h[j]), int(zc.get("N", 56)), rise[j]))
                if swing is not None and tf == "1m" and js:            # stop / take-profit watch on every new minute
                    swing.on_bar(float(h[js].max()), float(l[js].min()), float(c[js[-1]]), d["t"].iloc[js[-1]])
                first_pass.add(tf)
            if down_since is not None:
                gap = int(now.timestamp() - down_since)
                print(now.strftime("%H:%M:%S"), "行情恢复（中断 %d 分 %02d 秒），漏掉的 K 线已补算" % divmod(gap, 60))
                if down_warned:
                    pushers.send(cfg, "SPY 盯盘：行情已恢复", "中断约 %d 分钟，漏掉的 K 线已补算；中断期间的信号标了“补报”。" % max(1, round(gap / 60)))
                down_since = None; down_warned = False; last_err = None
            ts = now.timestamp(); oc = cfg.get("options", {})
            if oc.get("enabled", True) and spot is not None and ts - t_ox >= float(oc.get("refresh_sec", 120)):
                try:
                    ox = src.options(spot, now); store.options(now.strftime("%Y-%m-%d %H:%M:%S"), spot, ox)
                except Exception as e:
                    print("期权数据暂不可用：", e); ox = None
                t_ox = ts
            if ts - t_fl >= float(cfg.get("flows_refresh_sec", 60)):
                try:
                    fl = src.flows(); store.flows(now.strftime("%Y-%m-%d %H:%M:%S"), fl)
                except Exception as e:
                    print("资金数据暂不可用：", e); fl = None
                t_fl = ts
        except Exception as e:                               # K-line fetch failed: OpenD lost its server link, etc.
            if down_since is None: down_since = now.timestamp()
            msg = "%s: %s" % (type(e).__name__, e)
            if msg != last_err:                              # log each distinct error once, not every 5 s
                if isinstance(e, RuntimeError): print(now.strftime("%H:%M:%S"), "行情中断：", e, "—— 自动重试中")
                else: traceback.print_exc()
                last_err = msg
            if not down_warned and now.timestamp() - down_since >= 180:
                pushers.send(cfg, "SPY 盯盘：行情中断", "从美东 %s 起拿不到 K 线（%s）。程序在自动重试，恢复后会补算漏掉的信号。"
                             % (dt.datetime.fromtimestamp(down_since, ET).strftime("%H:%M"), e))
                down_warned = True
        if now_min >= 965:                                   # 16:05 ET -> review + stop
            if swing is not None: swing.on_close()
            path, summary = report.build(store, now.strftime("%Y-%m-%d"), cfg)
            try:
                import dashboard
                dash = dashboard.build(); print("看板已更新：", dash)
            except Exception as e:
                dash = None; print("看板更新失败：", e)
            pushers.send(cfg, "SPY 今日复盘", summary + "\n\n报告：" + path + ("\n看板：" + dash if dash else ""))
            print("复盘已生成：", path); return
        src.sleep(poll)


def _fresh(bar_t, now, minutes=5):
    """a signal is actionable for the swing rule only if its bar closed in the last few minutes (not a catch-up after an outage)"""
    return bar_t[:10] == now.strftime("%Y-%m-%d") and now.hour * 60 + now.minute - (int(bar_t[11:13]) * 60 + int(bar_t[14:16])) <= minutes


def keep_awake(mode):
    """Ask Windows not to sleep while this process runs (same API video players use; nothing in the power settings changes,
    and the request ends automatically when the program exits). mode: "display" (screen stays on - most reliable against
    Modern Standby), "system" (screen may turn off), "off". Closing the laptop lid still sleeps."""
    if mode == "off" or not sys.platform.startswith("win"): return False
    import ctypes
    flags = 0x80000000 | 0x00000001 | (0x00000002 if mode == "display" else 0)   # ES_CONTINUOUS | SYSTEM | DISPLAY
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


class Tee:
    """print to the console AND to data/watch_<ET date>.log (so every run can be diagnosed afterwards)."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8"); self.c = sys.__stdout__
    def write(self, x):
        for s in (self.c, self.f):
            try: s.write(x); s.flush()
            except Exception: pass
    def flush(self):
        for s in (self.c, self.f):
            try: s.flush()
            except Exception: pass


if __name__ == "__main__":
    cfg = load_cfg()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(HERE, "data", "watch_%s.log" % dt.datetime.now(ET).strftime("%Y-%m-%d")))
    print("\n==== 启动 %s（美东 %s）====" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dt.datetime.now(ET).strftime("%H:%M")), flush=True)
    ka = cfg.get("keep_awake", "display")
    print("防休眠：%s" % ("已开启（%s），收盘退出后自动解除" % ka if keep_awake(ka) else "未开启"), flush=True)
    store = Store(os.path.join(HERE, "data", "spy_watch.sqlite"))
    src = None; attempt = 0
    while True:                                              # keep trying until today's close (16:05 ET)
        try:
            src = LiveSource(cfg); break
        except Exception as e:
            now = dt.datetime.now(ET)
            if now.weekday() >= 5 or now.hour * 60 + now.minute >= 965:
                pushers.send(cfg, "SPY 盯盘：今天没连上 OpenD", "一直到收盘都没连上 moomoo OpenD，今天不盯盘了。"); sys.exit(1)
            print("连不上 OpenD：%s —— 30 秒后重试（第 %d 次）" % (e, attempt + 1), flush=True)
            if attempt % 60 == 0:                            # remind every 30 minutes
                if opend_needs_disclaimer():
                    pushers.send(cfg, "SPY 盯盘：请完成 moomoo API 问卷", "OpenD 已登录，但 moomoo 要求先完成 API 问卷评估和协议确认（OpenD 日志里有链接）。完成后程序自动连上。")
                else:
                    pushers.send(cfg, "SPY 盯盘：请登录 OpenD", "moomoo OpenD 没开或还没登录（端口 11111）。登录后程序会自动连上，不用重启。")
            attempt += 1; time.sleep(30)
    print("已连接 OpenD，开始盯盘（Ctrl+C 退出）。", flush=True)
    v6 = None
    if cfg.get("model", "v6") == "v6":
        try:
            import v6core
            v6 = v6core.V6(); v6.set_day(dt.datetime.now(ET).strftime("%Y-%m-%d"))
            print("模型：v6（%s / %s），前一日 VIX %.2f" % (v6.name["bottom"], v6.name["top"], float(v6.vix_prev.vix_prev.iloc[0])), flush=True)
        except Exception as e:
            v6 = None; print("v6 加载失败，改用 v5：", e, flush=True)
    pushers.send(cfg, "SPY 盯盘已启动", "已连接 moomoo OpenD，模型 %s。美东 9:25 起盯 1/3/5/15/60 分钟，16:05 生成复盘和看板。" % ("v6" if v6 else "v5"))
    try:
        import swing as swing_mod
        sw = swing_mod.Swing(cfg, send=lambda t, b: pushers.send(cfg, t, b)) if cfg.get("swing", {}).get("enabled", True) else None
        run(cfg, src, store, v6=v6, swing=sw)
    finally:
        src.close()
