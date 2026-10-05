"""波段模式 — the multi-day swing rule found in the 2026-10-06 study (v6/swing_study.py, swing_final.py), as live alerts.

  抄底日:   previous session closed with daily RSI2 < 20 while above its 200-day SMA
  进场:     on a dip day, the first 15m or 1h v6 bottom (p >= threshold) -> buy (the next 1-minute open)
  止损:     2.5% under the entry (50x isolated liquidates ~1% under the entry -> the alert shows every leverage's liquidation price)
  止盈:     once up 1 daily ATR (or a session closes with RSI2 > 70) -> sell at the first 15m v6 top
  时间:     still open after 10 trading days -> sell at the open
Out-of-sample 2024-01..2026-10-05: 52 trades, 71% winners, +0.31% per trade at 1x after OKX costs (t = 1.66);
23 of 52 went >= 1% against the entry first (= liquidated at 50x), 12 went >= 2%.
The program only ALERTS. It keeps a paper position in data/swing_state.json so it knows when to tell you to exit."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "data", "swing_state.json")
MMR = 0.01


def daily_stats(D):
    """D: daily bars (columns o,h,l,c; index = date strings), COMPLETED sessions only."""
    c = D.c.astype(float)
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = 100 - 100 / (1 + up / dn)
    tr = pd.concat([D.h - D.l, (D.h - c.shift()).abs(), (D.l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    sma200 = c.rolling(200).mean(); sma5 = c.rolling(5).mean()
    return {"date": str(D.index[-1])[:10], "close": float(c.iloc[-1]), "rsi2": float(rsi2.iloc[-1]), "sma200": float(sma200.iloc[-1]),
            "sma5": float(sma5.iloc[-1]), "atr": float(atr.iloc[-1])}


def liq_price(entry, lev):
    return entry * (1 - 1 / lev) / (1 - MMR)


class Swing:
    def __init__(self, cfg, send, log=print, state_path=STATE):
        sc = cfg.get("swing", {})
        self.enabled = sc.get("enabled", True); self.margin = float(sc.get("margin_cny", 280)); self.lev = float(sc.get("leverage", 50))
        self.stop_pct = float(sc.get("stop_pct", 2.5)) / 100; self.max_days = int(sc.get("max_days", 10)); self.th = float(cfg.get("threshold", 55))
        self.send, self.log, self.path = send, log, state_path
        self.st = self._load(); self.day = None; self.dstat = None; self.dip = False

    # ---------------------------------------------------------------- state
    def _load(self):
        try:
            return json.load(open(self.path, encoding="utf-8"))
        except Exception:
            return {"status": "flat", "history": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.st, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------------------------------------------------------- once per session
    def on_day(self, today, D):
        if not self.enabled or self.day == today: return
        self.day = today; s = daily_stats(D); self.dstat = s
        self.dip = s["rsi2"] < 20 and s["close"] > s["sma200"]
        if self.st["status"] == "long":
            self.st["days_held"] = self.st.get("days_held", 0) + (1 if self.st.get("last_day") != today else 0); self.st["last_day"] = today
            if s["rsi2"] > 70 and not self.st.get("hot"):
                self.st["hot"] = True; self.log("波段：昨收 RSI2=%.0f > 70，进入止盈观察" % s["rsi2"])
            self._save()
            self.send("SPY 波段：持仓中（第 %d 天）" % self.st["days_held"], self._hold_text(s["close"]))
        elif self.dip:
            self.send("SPY 波段：今天是抄底日", "昨收 %.2f，日线 RSI2 = %.1f（< 20），在 200 日线 %.2f 上方。\n\n今天出现第一个 15 分钟或 1 小时“底”信号时提醒你进场；"
                      "止损 2.5%%（约 %.2f）。" % (s["close"], s["rsi2"], s["sma200"], s["close"] * (1 - self.stop_pct)))
        self.log("波段：%s RSI2=%.1f 200日线=%.2f %s" % (s["date"], s["rsi2"], s["sma200"], "抄底日" if self.dip else ("持仓中" if self.st["status"] == "long" else "非抄底日")))

    # ---------------------------------------------------------------- v6 signals (15m / 1h early marks)
    def on_signal(self, tf, side, prob, price, bar_t, now_et):
        if not self.enabled or tf not in ("15m", "1h") or prob < self.th: return
        today = now_et.strftime("%Y-%m-%d")
        if side == "bottom" and self.st["status"] == "flat" and self.dip and bar_t[:10] == today and bar_t[11:16] < "15:50":
            e = price; stop = e * (1 - self.stop_pct)
            self.st = {"status": "long", "entry": e, "entry_t": bar_t, "stop": stop, "peak": e, "hot": False, "atr": self.dstat["atr"],
                       "days_held": 0, "last_day": today, "tf": tf, "history": self.st.get("history", [])}
            self._save()
            liq = {L: liq_price(e, L) for L in (50, 25, 20, 10)}
            self.send("SPY 波段：进场 %.2f（%s 底 %d%%）" % (e, {"15m": "15分钟", "1h": "1小时"}[tf], round(prob)),
                      "抄底日里的第一个大周期底。参考买入价 %.2f（下一分钟开盘），止损 %.2f（−2.5%%）。\n\n"
                      "爆仓价：50x %.2f（−1.0%%）· 25x %.2f · 20x %.2f · 10x %.2f。止损 2.5%% 只有 25 倍以内来得及触发；"
                      "历史上这类单子 44%% 先逆向走 1%% 以上（= 50 倍爆仓）。\n\n"
                      "%.0f 元保证金 × %.0f 倍 = 仓位 %.0f 元。止盈规则：涨满 1 个日线 ATR（约 %.2f）后，第一个 15 分钟“顶”离场。" % (
                          e, stop, liq[50], liq[25], liq[20], liq[10], self.margin, self.lev, self.margin * self.lev, e + self.dstat["atr"]))
        elif side == "top" and tf == "15m" and self.st["status"] == "long" and self.st.get("hot"):
            self._exit(price, bar_t, "止盈：涨够后出现 15 分钟顶")

    # ---------------------------------------------------------------- every loop: latest closed 1m bar
    def on_bar(self, high, low, close, bar_t):
        if not self.enabled or self.st["status"] != "long": return
        st = self.st
        if low <= st["stop"]:
            self._exit(min(st["stop"], close), bar_t, "止损：跌破 −2.5%"); return
        if high > st["peak"]:
            st["peak"] = high
            if not st.get("hot") and st["peak"] >= st["entry"] + st["atr"]:
                st["hot"] = True; self.log("波段：已涨满 1 个日线 ATR，进入止盈观察")
                self.send("SPY 波段：进入止盈观察", "最高 %.2f，已比入场 %.2f 高出 1 个日线 ATR。接下来第一个 15 分钟“顶”提醒离场。" % (st["peak"], st["entry"]))
            self._save()

    def on_close(self):
        if self.enabled and self.st["status"] == "long" and self.st.get("days_held", 0) >= self.max_days:
            self._exit(None, None, "时间到：持有满 %d 个交易日，明天开盘离场" % self.max_days)

    # ---------------------------------------------------------------- helpers
    def _hold_text(self, px):
        st = self.st; r = px / st["entry"] - 1
        return ("入场 %.2f（%s），现价 %.2f（%+.2f%%），止损 %.2f，最高 %.2f。%s\n50 倍逐仓按 %.0f 元保证金：浮动 %+.0f 元（未计资金费）。" % (
            st["entry"], st["entry_t"], px, r * 100, st["stop"], st["peak"], "已进入止盈观察，等 15 分钟顶。" if st.get("hot") else "还没涨满 1 个日线 ATR。",
            self.margin, self.margin * self.lev * r))

    def _exit(self, px, bar_t, why):
        st = self.st; e = st["entry"]
        r = (px / e - 1) if px else None
        body = why + "。\n\n入场 %.2f（%s）" % (e, st["entry_t"]) + ("，参考离场 %.2f，%+.2f%%；%.0f 元 × %.0f 倍 ≈ %+.0f 元（扣 12bp 手续费前）。" % (
            px, r * 100, self.margin, self.lev, self.margin * self.lev * r) if px else "。")
        self.send("SPY 波段：离场" + (" %.2f" % px if px else ""), body)
        hist = st.get("history", []) + [{"entry_t": st["entry_t"], "entry": e, "exit_t": bar_t, "exit": px, "ret": r, "why": why}]
        self.st = {"status": "flat", "history": hist[-50:]}; self._save()
