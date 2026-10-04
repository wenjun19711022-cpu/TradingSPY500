"""Live state of the playbook from results/edge_study.json, for the monitor screen.

Three sleeves, all long, sized for CME Micro E-mini S&P 500 futures (MES):
  noise      noise-area breakout (Zarattini, Aziz & Barbon 2024): checked at 10:00 ... 15:30;
             long above max(open, prior close) x (1 + sigma(t)); out below max(upper bound, VWAP)
             at a check, or at the close. Size: min(4, 2% / 14-day daily vol) x book scale.
  overnight  buy into the close, sell at the next open. Size: 1 x book scale.
  rsi2       the validated daily dip (spylev.live.engine): 2 x book scale on setup days.
The book scale puts the whole book at 20% annual volatility in the backtest (about 1.4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.edge.strategies import CHECKS, N, panel

ROOT = Path(__file__).resolve().parents[2]
MES_MULT = 5.0           # dollars per index point
SPX_PER_SPY = 10.03      # SPY ~ S&P 500 / 10 (dividends make it drift slightly)


def _hm(t: int) -> str:
    m = 570 + t
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class Playbook:
    scale: float = 1.44
    cap: float = 4.0
    vol_target: float = 0.02
    lookback: int = 14
    record: dict = field(default_factory=dict)
    day: object = None
    sigma: np.ndarray | None = None      # noise sigma by minute for today
    prev_close: float = float("nan")
    sigma_daily: float = float("nan")

    @classmethod
    def from_results(cls) -> "Playbook":
        p = ROOT / "results" / "edge_study.json"
        if not p.exists():
            return cls()
        s = json.loads(p.read_text())
        pf = s.get("portfolio", {})
        st = s.get("strategies", {})
        rec = {"book": {k: pf.get("book_scaled", {}).get(k) for k in ("ann_return", "ann_vol", "sharpe", "max_dd")},
               "buy_and_hold": {k: pf.get("buy_and_hold", {}).get(k) for k in ("ann_return", "sharpe", "max_dd")},
               "by_split": {k: v.get("sharpe") for k, v in pf.get("book_scaled", {}).get("by_split", {}).items()},
               "noise_mes_bp": st.get("noise", {}).get("regimes", {}).get("mes", {}).get("ALL", {}).get("net_bp"),
               "noise_okx_bp": st.get("noise", {}).get("regimes", {}).get("okx_taker", {}).get("ALL", {}).get("net_bp"),
               "overnight_mes_bp": st.get("overnight", {}).get("regimes", {}).get("mes", {}).get("ALL", {}).get("net_bp"),
               "overnight_okx_bp": st.get("overnight", {}).get("regimes", {}).get("okx_taker", {}).get("ALL", {}).get("net_bp"),
               "data": s.get("data", {}),
               "dead": [k for k in ("orb5", "im_first30", "im_rod") if k in st]}
        return cls(scale=float(pf.get("scale_to_20pct_vol", 1.44)), record=rec)

    # ------------------------------------------------------------------ once a day
    def prepare(self, hist: pd.DataFrame, day) -> bool:
        """hist: 1-minute RTH bars of earlier sessions (at least 15 full ones)."""
        self.day, self.sigma = day, None
        if hist is None or hist.empty:
            return False
        p = panel(hist)
        if len(p.days) < self.lookback + 1:
            return False
        move = np.abs(p.C[-self.lookback:] / p.O[-self.lookback:, [0]] - 1)
        self.sigma = move.mean(axis=0)
        self.prev_close = float(p.C[-1, -1])
        closes = p.C[:, -1]
        r = np.diff(closes[-(self.lookback + 1):]) / closes[-(self.lookback + 1):-1]
        self.sigma_daily = float(np.std(r, ddof=1))
        return True

    def lev_noise(self) -> float:
        if not np.isfinite(self.sigma_daily) or self.sigma_daily <= 0:
            return self.scale
        return float(min(self.cap, self.vol_target / self.sigma_daily) * self.scale)

    # ------------------------------------------------------------------ every bar
    def update(self, today: pd.DataFrame, rsi2_active: bool = False) -> dict:
        out = {"scale": self.scale, "record": self.record,
               "lev": {"noise": self.lev_noise(), "overnight": self.scale, "rsi2": 2 * self.scale}}
        if today is None or today.empty:
            return out
        mos = (today.index.hour * 60 + today.index.minute - 570).values
        k = int(mos[-1]) + 1                       # bars closed so far
        price = float(today["close"].iloc[-1])
        out["price"] = price
        out["mes_point"] = price * SPX_PER_SPY
        out["mes_notional"] = price * SPX_PER_SPY * MES_MULT
        # ---- overnight sleeve (clock only)
        if k < 1:
            out["overnight"] = {"state": "pre"}
        elif k < 385:
            out["overnight"] = {"state": "sold" if k > 1 else "sell_open",
                                "text": "昨晚的隔夜仓 9:30 开盘卖出" if k <= 5 else f"15:55–15:59 买入隔夜仓（{self.scale:.1f}x），明早 9:30 卖出"}
        else:
            out["overnight"] = {"state": "buy_close", "text": f"现在买入隔夜仓 {self.scale:.1f}x，明早 9:30 开盘卖出"}
        out["rsi2"] = {"active": bool(rsi2_active)}
        # ---- noise sleeve
        if self.sigma is None or not np.isfinite(self.prev_close):
            out["noise"] = {"state": "nodata", "text": "需要最近 15 个交易日的 1 分钟数据"}
            return out
        idx = np.clip(mos, 0, N - 1)
        O = pd.Series(today["open"].values, index=idx)
        C = pd.Series(today["close"].values, index=idx)
        H = pd.Series(today["high"].values, index=idx)
        L = pd.Series(today["low"].values, index=idx)
        V = pd.Series(today["volume"].values if "volume" in today else np.ones(len(today)), index=idx)
        o0 = float(O.iloc[0])
        ub = max(o0, self.prev_close) * (1 + self.sigma)
        tp = (H + L + C) / 3
        vwap = (tp * V).cumsum() / V.cumsum()
        pos, trades, entry, t_in = False, [], None, None
        for t in CHECKS:
            if t >= k or t not in C.index:
                continue
            c = float(C[t])
            nxt = float(O[t + 1]) if (t + 1) in O.index else None
            if not pos and c > ub[t]:
                if nxt is None:
                    out["noise_signal"] = "buy"
                    pos, entry, t_in = True, c, t + 1
                    break
                pos, entry, t_in = True, nxt, t + 1
            elif pos and c < max(ub[t], float(vwap[t])):
                ex = nxt if nxt is not None else c
                trades.append({"in": _hm(t_in), "out": _hm(t + 1), "entry": entry, "exit": ex, "ret": ex / entry - 1})
                pos = False
                if nxt is None:
                    out["noise_signal"] = "sell"
        t_now = int(mos[-1])
        if pos and t_now >= N - 1:
            trades.append({"in": _hm(t_in), "out": "16:00", "entry": entry, "exit": price, "ret": price / entry - 1})
            pos = False
        nxt_check = next((t + 1 for t in CHECKS if t > t_now), None)
        stop_now = max(float(ub[min(t_now, N - 1)]), float(vwap.iloc[-1]))
        st = {"ub_now": float(ub[min(t_now, N - 1)]), "vwap": float(vwap.iloc[-1]), "stop_now": stop_now,
              "next_check": _hm(nxt_check) if nxt_check is not None else None, "trades": trades,
              "sigma_daily": self.sigma_daily, "prev_close": self.prev_close, "open": o0}
        if pos:
            st.update(state="long", entry=entry, entry_time=_hm(t_in), pnl=price / entry - 1,
                      text=f"做多中 @ {entry:.2f}（{_hm(t_in)} 进场）· 检查时低于 {stop_now:.2f} 就卖")
        elif t_now < CHECKS[0]:
            st.update(state="wait", text=f"10:00 第一次检查：价格要高于上沿 {ub[CHECKS[0]]:.2f}")
        elif nxt_check is None or t_now >= CHECKS[-1]:
            st.update(state="done", text="今天的检查结束" + ("，有交易" if trades else "，没有突破"))
        else:
            st.update(state="flat", text=f"没有突破。下次检查 {_hm(nxt_check)}：要高于上沿 {ub[nxt_check - 1]:.2f}")
        out["noise"] = st
        return out
