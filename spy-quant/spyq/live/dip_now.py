"""Scan every timeframe for a bottom right now and print a trade ticket for validated streams.

    python -m spyq.live.dip_now --equity 20000
    python -m spyq.live.dip_now --equity 20000 --funding 0.00005 --json ticket.json

Only streams that passed the pre-registered gates in results/dip_study.json get leverage;
every other timeframe is shown as an unvalidated read-out (0x until it passes on your own
minute data via scripts/run_dip_study.py).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spyq.data import store
from spyq.data.history import spy_daily, vix_daily
from spyq.data.resample import resample
from spyq.data.sources import load_ibkr_json
from spyq.dip.engine import EXITS
from spyq.dip.perp import PerpCosts
from spyq.dip.signals import FAMILIES, features, signal

ROOT = Path(__file__).resolve().parents[2]


def frames() -> dict:
    spy, _ = spy_daily()
    out = {
        "1d": spy.assign(volume=0.0),
        "1w": spy.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().assign(volume=0.0),
        "1M": spy.resample("ME").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().assign(volume=0.0),
    }
    full = store.load("SPY", "1m")
    if len(full) > 5_000:
        for tf in ("3m", "5m", "15m", "30m", "1h", "4h"):
            out[tf] = resample(full.tail(200_000), tf).drop(columns=["n_bars"])
    else:
        raw = ROOT / "data" / "raw" / "ibkr"
        m1 = load_ibkr_json(raw / "SPY_1m_recent.json")
        m5 = load_ibkr_json(raw / "SPY_5m_recent.json")
        m30 = load_ibkr_json(raw / "SPY_30m_recent.json")
        out.update({"3m": resample(m1, "3m").drop(columns=["n_bars"]), "5m": m5, "15m": resample(m5, "15m").drop(columns=["n_bars"]),
                    "30m": m30, "1h": resample(m30, "1h").drop(columns=["n_bars"]), "4h": resample(m30, "4h").drop(columns=["n_bars"])})
    return out


def scan(equity: float, costs: PerpCosts, lev_override: float | None = None) -> dict:
    study = json.loads((ROOT / "results" / "dip_study.json").read_text())
    validated = {s["key"]: s for s in study["streams"] if s.get("G3")}
    vix = vix_daily()["close"]
    rows, tickets = [], []
    for tf, df in frames().items():
        f = features(df, tf, vix if tf in ("1d", "1w", "1M") else None)
        last = f.iloc[-1]
        fired = {fam: bool(signal(f, fam, tf).iloc[-1]) for fam in FAMILIES
                 if not (fam in ("panic", "ath_dd") and tf not in ("1d", "1w", "1M"))}
        rows.append({"tf": tf, "time": str(df.index[-1])[:16], "close": float(df["close"].iloc[-1]),
                     "rsi2": float(last["rsi2"]), "rsi14": float(last["rsi14"]), "boll_z": float(last["boll_z"]),
                     "dd_atr": float(last["dd_atr"]), "lower_wick": float(last["lower_wick"]) if np.isfinite(last["lower_wick"]) else None,
                     "trend_up": bool(last["trend_up"]), "dd_ath": float(last["dd_ath"]), "fired": [k for k, v in fired.items() if v],
                     "validated": [k for k in validated if k.startswith(tf + "|")]})
        for fam, on in fired.items():
            for key, s in validated.items():
                t_, f_, st = key.split("|")
                if t_ != tf or f_ != fam or not on:
                    continue
                stop_atr = None if st == "stop-" else float(st[4:])
                px = float(df["close"].iloc[-1])
                L_rec = s["leverage"]["L"]
                L_max = min(s["leverage"]["L_liq"], s["leverage"]["L_kelly"], 5.0)
                L = lev_override or L_rec
                stop = px - stop_atr * float(last["atr"]) if stop_atr else None
                liq = px * (1 - costs.liq_drop(L)) if L > 1 else 0.0
                exit_rule, max_bars = EXITS[fam]
                avg_days = s["ALL"]["avg_days"]
                tickets.append({
                    "stream": key, "entry": "下一根 K 线开盘市价（或在收盘价附近挂限价单）", "ref_price": px,
                    "leverage_recommended": round(L_rec, 2), "leverage_max": round(L_max, 2), "leverage_used": round(L, 2),
                    "notional": round(equity * L, 2), "stop": round(stop, 2) if stop else None,
                    "liquidation_price": round(liq, 2), "exit": f"收盘价站上 {exit_rule.upper()} 后下一根开盘平仓；最多持有 {max_bars} 根",
                    "expected_net_per_trade": s["ALL"]["mean_net"], "win_rate": s["ALL"]["win_rate"],
                    "expected_funding": costs.funding(avg_days) * L * equity, "avg_days": avg_days,
                    "stop_above_liq": (stop or 0) > liq,
                })
    return {"as_of": str(pd.Timestamp.now(tz="UTC"))[:16], "equity": equity, "costs": costs.__dict__,
            "timeframes": rows, "tickets": tickets}


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--funding", type=float, default=0.0001, help="per 8h, e.g. 0.0001 = 0.01%%")
    ap.add_argument("--lev", type=float, help="override leverage for the ticket")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = scan(a.equity, PerpCosts(funding_8h=a.funding), a.lev)
    print(f"\n多周期抄底扫描  (账户 ${a.equity:,.0f}, 资金费 {a.funding * 100:.3f}%/8h)")
    for r in res["timeframes"]:
        flag = "触发 " + ",".join(r["fired"]) if r["fired"] else "—"
        print(f"  {r['tf']:>3} {r['time']}  收 {r['close']:.2f}  RSI2 {r['rsi2']:5.1f}  布林z {r['boll_z']:+.2f}  "
              f"距20高 {r['dd_atr']:+.1f}ATR  趋势{'上' if r['trend_up'] else '下'}  距历史高 {r['dd_ath']:+.1%}  {flag}"
              f"{'' if r['validated'] else '  (未验证周期，0x)'}")
    if not res["tickets"]:
        print("\n没有已验证的抄底信号：空仓等待。")
    for t in res["tickets"]:
        print(f"\n开仓单 {t['stream']}: 杠杆 {t['leverage_used']}x（建议 {t['leverage_recommended']}x，上限 {t['leverage_max']}x）"
              f" 名义 ${t['notional']:,.0f}  止损 {t['stop']}  强平 {t['liquidation_price']}  {t['exit']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


if __name__ == "__main__":
    main()
