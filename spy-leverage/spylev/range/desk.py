"""Live range desk: at 10:00 forecast the rest of the day, propose a limit-buy ticket, and every
minute re-price the odds of the ticket (like moomoo's profit probability for an option spread).

State it produces (shown on the screen, see web/index.html):
  bands      rest-of-day low / high holding with 50 / 80 / 90 % (from the 10:00 forecast)
  ticket     entry / stop / take-profit (the user's own, or the automatic one)
  odds       P(fill), P(take-profit first), P(stop first), P(neither by 15:55), fair EV,
             EV after OKX costs at the chosen leverage, the win rate needed to break even
  record     how this kind of ticket did in the backtest (results/range_study.json)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.dip.perp import PerpCosts
from spylev.range.engine import TICK, RangeSpec, plan
from spylev.range.model import RangeModel, five_minute, p_bracket, p_touch_below, slot_returns

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else {}


def daily_rv(bars: pd.DataFrame) -> pd.Series:
    """Realized variance per session from 5-minute returns (bars may be 1m or 5m)."""
    b5 = five_minute(bars)
    sr = slot_returns(b5)
    return (sr["r"] ** 2).groupby(sr["day"]).sum()


@dataclass
class RangeDesk:
    model: RangeModel
    costs: PerpCosts = field(default_factory=PerpCosts)
    leverage: float = 20.0
    spec: RangeSpec = field(default_factory=RangeSpec)
    user_ticket: dict | None = None        # {"entry":..,"stop":..,"tp":..} set from the screen
    day: object = None
    today: dict | None = None              # the 10:00 forecast
    fill: dict | None = None               # {"time","price"} once the limit buy traded
    done: dict | None = None               # {"time","price","reason"} once the ticket closed

    @classmethod
    def from_results(cls, leverage: float = 20.0) -> "RangeDesk":
        p = ROOT / "results" / "range_model.json"
        if not p.exists():
            raise FileNotFoundError("results/range_model.json missing: run scripts/run_range_study.py")
        d = cls(RangeModel.load(p), leverage=leverage)
        d.study = _load("range_study.json")
        return d

    # ------------------------------------------------------------------ helpers
    def _reset(self, day):
        self.day, self.today, self.fill, self.done = day, None, None, None

    def set_ticket(self, entry: float | None, stop: float | None, tp: float | None):
        self.user_ticket = None if entry is None else {"entry": float(entry), "stop": float(stop), "tp": float(tp)}
        self.fill, self.done = None, None

    def record(self) -> dict:
        st = getattr(self, "study", {}) or {}
        ref = next((g for g in st.get("grid", []) if g["key"] == self.spec.key), None)
        rel = st.get("reliability_pooled", [])
        return {"key": self.spec.key, "ALL": ref["ALL"] if ref else {}, "IS": ref["IS"] if ref else {},
                "VAL": ref["VAL"] if ref else {}, "TEST": ref["TEST"] if ref else {},
                "configs": len(st.get("grid", [])), "passed": sum(bool(g.get("G3")) for g in st.get("grid", [])),
                "reliability": rel, "period": st.get("data", {})}

    def forecast(self, today_bars: pd.DataFrame, rv_hist: pd.Series, vix_prev: float) -> dict | None:
        """The 10:00 forecast from today's first 30 minutes, the last days' realized variance
        (spylev.range.desk.daily_rv) and yesterday's VIX close."""
        t0 = self.model.t0
        mos = (today_bars.index.hour * 60 + today_bars.index.minute) - 570
        first = today_bars[mos < t0]
        if len(first) < t0 * 0.8 or not np.isfinite(vix_prev):
            return None
        day = today_bars.index[-1].normalize()
        rv_hist = rv_hist[rv_hist.index < day] if len(rv_hist) else rv_hist
        if len(rv_hist) < 5:
            return None
        sr = slot_returns(five_minute(first))
        row = {"vix_prev": vix_prev, "rv1": float(rv_hist.iloc[-1]), "rv5": float(rv_hist.iloc[-5:].mean()),
               "rv_open": float((sr["r"] ** 2).sum())}
        sigma = self.model.sigma_rest(row)
        s0 = float(first["close"].iloc[-1])
        bands = {}
        for p in (0.5, 0.8, 0.9):
            lo, hi = self.model.band(s0, sigma, p)
            bands[f"{p:g}"] = {"low": float(lo), "high": float(hi), "both": float(self.model.joint.get(f"{p:g}", np.nan))}
        auto = plan(self.model, s0, sigma, self.spec)
        return {"s0": s0, "sigma": float(sigma), "sigma_usd": float(s0 * sigma), "vix_prev": float(vix_prev),
                "open": float(first["open"].iloc[0]), "h0": float(first["high"].max()), "l0": float(first["low"].min()),
                "rv5_daily_pct": float(np.sqrt(row["rv5"]) * 100), "bands": bands, "auto": auto}

    # ------------------------------------------------------------------ main step
    def update(self, today_bars: pd.DataFrame, rv_hist: pd.Series, vix_prev: float) -> dict:
        """Call after every closed 1m bar. today_bars: today's 1m bars so far (ET index)."""
        if today_bars.empty:
            return {"phase": "closed"}
        day = today_bars.index[-1].normalize()
        if day != self.day:
            self._reset(day)
        ts = today_bars.index[-1]
        mos = int((ts.hour * 60 + ts.minute) - 570)
        price = float(today_bars["close"].iloc[-1])
        out = {"phase": "pre", "time": ts.strftime("%Y-%m-%d %H:%M"), "price": price, "leverage": self.leverage,
               "decision": f"{(570 + self.model.t0) // 60:02d}:{(570 + self.model.t0) % 60:02d}"}
        if mos + 1 < self.model.t0:
            out["minutes_to_forecast"] = self.model.t0 - mos - 1
            return out
        if self.today is None:
            self.today = self.forecast(today_bars, rv_hist, vix_prev)
            if self.today is None:
                return {**out, "phase": "nodata"}
        f = self.today
        tk = self.user_ticket or f["auto"]
        out.update(phase="live", forecast=f, ticket=tk, source="你的单子" if self.user_ticket else "自动计划")
        if tk is None:
            return out
        e, s, t = tk["entry"], tk["stop"], tk["tp"]
        # ---- fill / exit tracking on the bar that just closed
        bar = today_bars.iloc[-1]
        if self.fill is None and self.done is None and mos <= self.spec.last_fill and self.model.t0 <= mos:
            if bar["open"] <= e or bar["low"] <= e - TICK:
                self.fill = {"time": ts.strftime("%H:%M"), "price": float(min(bar["open"], e)) if bar["open"] <= e else e}
                if bar["low"] <= s:
                    self.done = {"time": ts.strftime("%H:%M"), "price": s, "reason": "stop"}
        elif self.fill is not None and self.done is None:
            if bar["low"] <= s:
                self.done = {"time": ts.strftime("%H:%M"), "price": float(min(bar["open"], s)), "reason": "stop"}
            elif bar["high"] >= t + TICK:
                self.done = {"time": ts.strftime("%H:%M"), "price": float(max(bar["open"], t)), "reason": "tp"}
            elif mos >= self.spec.flat:
                self.done = {"time": ts.strftime("%H:%M"), "price": float(bar["close"]), "reason": "eod"}
        # ---- odds from here (variance left until 15:55 on the intraday clock)
        var = self.model.sigma_between(f["sigma"], mos + 1, self.spec.flat) ** 2
        c = self.costs
        win, loss = t / e - 1, s / e - 1
        win_net, loss_net = win - 2 * c.maker, loss - c.maker - c.taker - c.slippage
        need = -loss_net / (win_net - loss_net)
        odds = {"win_bp": win * 1e4, "loss_bp": loss * 1e4, "win_net_bp": win_net * 1e4, "loss_net_bp": loss_net * 1e4,
                "breakeven": need, "liq_price": e * (1 - c.liq_drop(self.leverage)), "fair_no_time": np.log(e / s) / np.log(t / s)}
        if self.done is not None:
            px = self.done["price"]
            g = px / self.fill["price"] - 1
            fee = c.maker + (c.maker if self.done["reason"] == "tp" else c.taker + c.slippage)
            odds.update(state="done", result_bp=(g - fee) * 1e4, result_acct=(g - fee) * self.leverage)
        elif self.fill is not None:
            up, dn, none = p_bracket(price, s, t, var)
            pnl = price / self.fill["price"] - 1
            odds.update(state="in", p_tp=up, p_stop=dn, p_none=none, pnl_bp=pnl * 1e4, pnl_acct=pnl * self.leverage,
                        ev_from_here_acct=(-c.maker - (up * c.maker + (1 - up) * (c.taker + c.slippage))) * self.leverage)
        elif mos > self.spec.last_fill:
            odds.update(state="expired")
        else:
            pf = p_touch_below(price, e, var)
            up, dn, none = p_bracket(e, s, t, var)
            # no edge: the expected price move is zero, so the expected result is minus the fees
            ev_net = -(c.maker + up * c.maker + (1 - up) * (c.taker + c.slippage))
            odds.update(state="waiting", p_fill=pf, p_tp=up, p_stop=dn, p_none=none, p_success=pf * up,
                        ev_net_bp=ev_net * 1e4, ev_net_acct=ev_net * self.leverage)
        out.update(odds=odds, fill=self.fill, done=self.done, position_in_band=self._where(price, f))
        return out

    def screen(self) -> dict | None:
        """What the web page needs to price any ticket by itself (see web/index.html)."""
        if self.today is None:
            return None
        m = self.model
        return {"forecast": self.today, "t0": m.t0, "flat": self.spec.flat, "last_fill": self.spec.last_fill,
                "cum": [float(v) for v in np.r_[0, np.cumsum(m.profile)]], "costs": {"maker": self.costs.maker,
                "taker": self.costs.taker, "slippage": self.costs.slippage, "mmr": self.costs.mmr}, "record": self.record()}

    @staticmethod
    def _where(price, f):
        b = f["bands"]["0.8"]
        return float((price - b["low"]) / (b["high"] - b["low"])) if b["high"] > b["low"] else None
