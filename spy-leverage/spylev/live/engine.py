"""Turns bars into one plain instruction for the screen.

Inputs: 1-minute regular-hours bars (history + today, from moomoo OpenD or a replay) and the
daily history. Every time a 1m bar closes, `LiveEngine.on_bar()` returns a state dict:

  status   wait | watch | buy | hold | sell
  headline what to do, in one line (e.g. "做多 2x @ 523.40")
  action   leverage, price, stop, exit rule, expected result per trade (from the backtests)
  checks   the six bottom facts for 1m / 3m / 5m (the checklist the screen shows)
  levels   support / resistance levels in play
  daily    the validated daily RSI2 setup (the only one that earns leverage)

Rules (from results/dip_study.json and results/scalp_study.json):
* "buy" (green) only on days the validated daily RSI2 setup is active. The 1m/3m/5m
  confluence is used to time that entry; if it never fires, buy before the close.
* The intraday confluence on its own is shown as "watch" (yellow) with its real track record
  (1,152 trades, 24% wins, -11.8bp per trade after OKX costs; 10x lost the whole account over
  2022-10..2024-12). It is information, not an order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.dip.perp import PerpCosts
from spylev.dip.signals import rsi
from spylev.scalp.signals import build, buy_signal, quality, tf_conditions
from spylev.ta import daily_levels

ROOT = Path(__file__).resolve().parents[2]
FACT_NAMES = {"osc": "超卖指标≥2个", "band": "跌破布林下轨/VWAP−2σ", "support": "支撑位插针收回",
              "candle": "锤子线/看涨吞没", "climax": "放量", "turn": "MACD拐头/底背离"}
LEVEL_NAMES = {"pdl": "昨日低点", "pdh": "昨日高点", "pdc": "昨日收盘", "s1": "枢轴S1", "s2": "枢轴S2", "r1": "枢轴R1",
               "r2": "枢轴R2", "pivot": "枢轴P", "val": "昨日价值区下沿", "vah": "昨日价值区上沿", "poc": "昨日成交量峰",
               "pml": "盘前低点", "pmh": "盘前高点", "sess_low": "今日低点", "or_low": "开盘15分钟低点",
               "swing_low": "前一个波段低点", "round5": "5美元整数关口", "vwap": "今日VWAP", "day_low": "今日最低",
               "day_high": "今日最高"}


def _i(v) -> int:
    return int(v) if v is not None and pd.notna(v) else 0


def _b(v) -> bool:
    return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False


def _load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else {}


@dataclass
class Position:
    kind: str            # "daily" (validated) or "paper" (intraday watch, simulated)
    entry_time: str
    entry: float
    stop: float
    leverage: float
    note: str = ""


@dataclass
class LiveEngine:
    daily: pd.DataFrame                      # daily OHLC (history, unadjusted)
    user_leverage: float = 10.0              # what the user wants to see the math for
    costs: PerpCosts = field(default_factory=PerpCosts)
    position: Position | None = None
    paper: Position | None = None
    alerts: list = field(default_factory=list)

    def __post_init__(self):
        dip = _load("dip_study.json")
        scalp = _load("scalp_study.json")
        s = next((x for x in dip.get("streams", []) if x["key"] == "1d|rsi2|stop2.5"), None)
        self.daily_stats = s["ALL"] if s else {"win_rate": 0.67, "mean_net": 0.0024, "avg_days": 4.1}
        self.daily_lev = {"保守": 0.7, "标准": 2.0, "上限": 3.0}
        ref = next((g for g in scalp.get("grid", []) if g["key"] == "thr5=4|trend=none|exit=top5m"), None)
        self.scalp_all = ref["ALL"] if ref else {}
        self.scalp_exp = ref.get("expectation", []) if ref else []

    # ------------------------------------------------------------------ daily setup
    def daily_setup(self, today: pd.Timestamp) -> dict:
        d = self.daily[self.daily.index < today.normalize().tz_localize(None)]
        c = d["close"]
        r2 = float(rsi(c, 2).iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        sma5 = float(c.rolling(5).mean().iloc[-1])
        atr = float((pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1)
                     .max(axis=1)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        active = (r2 < 10) and (c.iloc[-1] > sma200)
        return {"rsi2": r2, "close": float(c.iloc[-1]), "sma200": sma200, "sma5": sma5, "atr": atr,
                "trend_ok": bool(c.iloc[-1] > sma200), "active": bool(active),
                "need": "RSI2<10 且收盘>200日均线" if not active else "今天是验证过的抄底日"}

    # ------------------------------------------------------------------ main step
    def evaluate(self, x: pd.DataFrame, i: int, setup: dict) -> dict:
        """State after bar i of the feature frame x (built by spylev.scalp.signals.build)."""
        row = x.iloc[i]
        t = x.index[i]
        price = float(row["close"])
        sig = bool(buy_signal(x.iloc[: i + 1].tail(3), thr5=4).iloc[-1]) if i >= 2 else False
        tf_ok = {k: bool(v) for k, v in tf_conditions(x.iloc[[i]], thr5=4).iloc[0].items()}
        q = float(quality(x.iloc[[i]]).iloc[0])
        checks = {}
        for tf in ("m5", "m3", "m1"):  # 3m / 5m: the latest CLOSED bar of that timeframe
            checks[tf] = {k: _b(row.get(f"{tf}_b_{k}")) for k in FACT_NAMES}
            checks[tf]["score"] = _i(row.get(f"{tf}_bottom"))
            checks[tf]["top"] = _i(row.get(f"{tf}_top"))
        mos = int(row["mos"])
        low5 = float(x["low"].iloc[max(0, i - 4): i + 1].min())
        stop_intraday = min(low5 * (1 - 0.0002), price * (1 - 0.0008))
        sup = row.get("m5_support_name")
        sup = sup if isinstance(sup, str) and sup else (row.get("m1_support_name") if isinstance(row.get("m1_support_name"), str) else "")
        state = {"time": t.strftime("%Y-%m-%d %H:%M"), "price": price, "quality": q, "checks": checks, "tf_ok": tf_ok,
                 "support": LEVEL_NAMES.get(sup, sup),
                 "daily": setup, "position": None, "paper": None}
        # ---- open validated position management
        if self.position is not None:
            p = self.position
            pnl = (price / p.entry - 1) * p.leverage
            state["position"] = {**p.__dict__, "pnl_pct": pnl}
            if float(row["low"]) <= p.stop:
                self.alerts.append({"time": state["time"], "type": "sell", "text": f"止损 {p.stop:.2f}"})
                self.position = None
                return {**state, "status": "sell", "headline": f"卖出！触及止损 {p.stop:.2f}",
                        "sub": "按规则离场，不要拖。", "action": {"side": "sell", "price": p.stop}}
            exit_rule = f"收盘价 > 5日均线 {setup['sma5']:.2f} 时，下一交易日开盘卖出（最多 10 个交易日）"
            top = _i(row.get("m5_top")) >= 3 and _b(row.get("m5_t_osc"))
            sub = exit_rule + ("。5分钟见顶信号出现：若当天就是卖出日，可在此时卖。" if top else "")
            return {**state, "status": "hold", "headline": f"持有中 {p.leverage:g}x · 浮动 {pnl:+.2%}", "sub": sub,
                    "action": {"side": "hold", "stop": p.stop, "exit_rule": exit_rule}}
        # ---- validated daily setup: time the entry
        if setup["active"]:
            lev = self.daily_lev["标准"]
            stop = price - 2.5 * setup["atr"]
            last_call = mos >= 375  # 15:45
            if sig or last_call:
                why = "1/3/5分钟共振见底" if sig else "收盘前没有共振，按规则收盘前买入"
                act = {"side": "buy", "leverage": lev, "leverage_range": self.daily_lev, "price": price, "stop": round(stop, 2),
                       "liquidation": round(price * (1 - self.costs.liq_drop(lev)), 2),
                       "exit_rule": "收盘价站上5日均线后，下一交易日开盘卖出；最多10个交易日",
                       "expected_pct": self.daily_stats["mean_net"] * lev, "win_rate": self.daily_stats["win_rate"],
                       "hold_days": self.daily_stats.get("avg_days", 4.1), "validated": True}
                self.alerts.append({"time": state["time"], "type": "buy", "text": f"做多 {lev:g}x @ {price:.2f}"})
                return {**state, "status": "buy", "headline": f"做多 {lev:g}x @ {price:.2f}", "sub": why, "action": act}
            return {**state, "status": "ready", "headline": "今天是抄底日：等 1/3/5 分钟共振见底再买",
                    "sub": f"最晚 15:45 收盘前买入。日线 RSI2 = {setup['rsi2']:.1f}", "action": None}
        # ---- intraday confluence without the daily setup: information only
        if self.paper is not None:
            p = self.paper
            pnl = (price / p.entry - 1) * p.leverage
            top = _i(row.get("m5_top")) >= 3 and _b(row.get("m5_t_osc"))
            done = float(row["low"]) <= p.stop or top or mos >= 385
            state["paper"] = {**p.__dict__, "pnl_pct": pnl}
            if done:
                why = "止损" if float(row["low"]) <= p.stop else "5分钟见顶" if top else "收盘"
                self.alerts.append({"time": state["time"], "type": "paper_exit", "text": f"模拟单结束（{why}）{pnl:+.2%}"})
                self.paper = None
        if sig and self.alerts and self.alerts[-1]["type"] == "watch" and \
                (t - pd.Timestamp(self.alerts[-1]["time"]).tz_localize(t.tz)).total_seconds() < 600:
            sig = False  # same setup still in play: one alert per 10 minutes
        if sig:
            bucket = next((b for b in self.scalp_exp if b["q_lo"] <= q < b["q_hi"]), None) or (self.scalp_exp[0] if self.scalp_exp else None)
            exp_bp = bucket["mean_net_bp"] if bucket else self.scalp_all.get("mean_net_bp", -11.8)
            win = bucket["win_rate"] if bucket else self.scalp_all.get("win_rate", 0.24)
            L = self.user_leverage
            act = {"side": "watch", "leverage": L, "price": price, "stop": round(stop_intraday, 2),
                   "exit_rule": "5分钟见顶（超买+见顶分≥3）时卖出，15:55 前必须平仓",
                   "expected_pct": exp_bp / 1e4 * L, "win_rate": win, "n_hist": bucket["n"] if bucket else self.scalp_all.get("n"),
                   "bucket": bucket["bucket"] if bucket else "", "validated": False,
                   "account_after_27m": self.scalp_all.get("lev10_final")}
            if self.paper is None:
                self.paper = Position("paper", state["time"], price, stop_intraday, L, "模拟跟踪，不下单")
            self.alerts.append({"time": state["time"], "type": "watch", "text": f"1/3/5分钟共振见底 @ {price:.2f}（历史期望为负）"})
            return {**state, "status": "watch", "headline": f"1/3/5 分钟共振见底 @ {price:.2f} — 不下单",
                    "sub": f"历史 {act['n_hist']} 次：胜率 {win:.0%}，每次 {exp_bp:+.1f} 个基点；{L:g} 倍杠杆每次约 {act['expected_pct']:+.2%} 账户",
                    "action": act}
        need = setup["need"]
        return {**state, "status": "wait", "headline": "等待", "sub": f"没有验证过的信号。日线 RSI2 = {setup['rsi2']:.1f}（需要 < 10）", "action": None}

    def track_record(self) -> dict:
        """The backtest numbers the screen quotes (from results/*.json)."""
        scalp = _load("scalp_study.json")
        dip = _load("dip_study.json")
        s = next((x for x in dip.get("streams", []) if x["key"] == "1d|rsi2|stop2.5"), {})
        keep = ("n", "win_rate", "mean_net_bp", "mean_gross_bp", "avg_minutes", "stop_rate", "lev10_final", "lev10_worst_day")
        return {"scalp": {k: self.scalp_all.get(k) for k in keep},
                "scalp_period": [scalp.get("data", {}).get("start"), scalp.get("data", {}).get("end")],
                "scalp_random_gross_bp": scalp.get("random_baseline", {}).get("mean_gross_bp"),
                "scalp_cost_bp": self.costs.round_trip() * 1e4,
                "daily": {k: self.daily_stats.get(k) for k in ("n", "win_rate", "mean_net", "avg_days", "worst_net")},
                "daily_label": s.get("label", "SPY 日线"), "daily_oos": s.get("oos", {}), "daily_lev": self.daily_lev}

    def confirm_buy(self, price: float, leverage: float, stop: float, when: str):
        self.position = Position("daily", when, price, stop, leverage)

    def close_position(self):
        self.position = None


def screen_levels(lv: pd.DataFrame, x: pd.DataFrame, i: int, n=6) -> list:
    """Nearest prior-day levels plus today's VWAP / high / low so far, for bar i of x."""
    t = x.index[i]
    day = t.normalize()
    hit = lv.index[lv.index.normalize() == day] if lv.index.tz is not None else lv.index[lv.index == day.tz_localize(None)]
    row = lv.loc[hit[0]].copy() if len(hit) else pd.Series(dtype=float)
    today = x.iloc[: i + 1]
    today = today[today.index.normalize() == day]
    row["vwap"] = x["vwap"].iloc[i] if "vwap" in x else np.nan
    row["day_low"], row["day_high"] = today["low"].min(), today["high"].max()
    return levels_for_screen(row, float(x["close"].iloc[i]), n)


def levels_for_screen(lv_row: pd.Series, price: float, n=6) -> list:
    out = []
    for k, v in lv_row.items():
        if k in LEVEL_NAMES and pd.notna(v):
            out.append({"name": LEVEL_NAMES[k], "price": round(float(v), 2), "dist_pct": float(v / price - 1)})
    out.sort(key=lambda r: abs(r["dist_pct"]))
    return out[:n]


def day_states(m1: pd.DataFrame, daily: pd.DataFrame, date: str, premarket: pd.DataFrame | None = None,
               user_leverage: float = 10.0, auto_follow: bool = True, levels: pd.DataFrame | None = None) -> list:
    """Replay a whole session through the engine (what the screen would have shown minute by
    minute). With auto_follow, a validated buy is assumed taken at the alert price."""
    lv = daily_levels(m1) if levels is None else levels
    if premarket is not None and not premarket.empty:
        lv = lv.join(premarket)
    d0 = pd.Timestamp(date)
    hist = m1[m1.index.normalize().tz_localize(None) >= d0 - pd.Timedelta(days=12)]
    hist = hist[hist.index.normalize().tz_localize(None) <= d0]
    x = build(hist, lv)
    eng = LiveEngine(daily[daily.index <= d0], user_leverage)
    setup = eng.daily_setup(pd.Timestamp(date))
    states = []
    idx = np.flatnonzero(x.index.normalize().tz_localize(None) == d0)
    for i in idx:
        s = eng.evaluate(x, i, setup)
        if auto_follow and s["status"] == "buy" and eng.position is None:
            eng.confirm_buy(s["price"], s["action"]["leverage"], s["action"]["stop"], s["time"])
        s["levels"] = screen_levels(lv, x, i)
        s["alerts"] = eng.alerts[-8:]
        states.append(s)
    return states
