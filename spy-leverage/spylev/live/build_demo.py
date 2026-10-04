"""Bake replay days into web/index.html -> a standalone demo page (no Python, no OpenD needed).

    python -m spylev.live.build_demo                       # -> web/demo.html
    python -m spylev.live.build_demo --out /tmp/demo.html --dates 2026-10-02 2023-08-17

Each day is run minute by minute through the same LiveEngine and RangeDesk the live monitor
uses, so the demo shows exactly what the screen would have said on that day. Days after 2024
come from the IBKR files in data/raw/ibkr (1-minute bars for the last three sessions).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spylev.data import store
from spylev.data.history import spy_daily
from spylev.data.minute_hist import premarket_levels
from spylev.data.sources import load_ibkr_json
from spylev.edge.live import Playbook
from spylev.live.engine import LiveEngine, day_states
from spylev.range.desk import RangeDesk, daily_rv
from spylev.ta import daily_levels

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
IBKR = ROOT / "data" / "raw" / "ibkr"
DAYS = {
    "2026-10-02": "你截图的那天：10:00 预测区间；你的单子 768.5 买 / 767 止损 / 769.5 止盈",
    "2026-10-01": "你截图的另一天：758.79 到 765.65",
    "2024-10-14": "噪声区突破：10:00 进场被跟踪止损扫掉 −0.09%，12:00 再进场拿到收盘 +0.14%",
    "2023-08-17": "日线抄底日：1/3/5 分钟共振时买入 2 倍",
    "2024-03-12": "普通日：只有 1/3/5 分钟共振（只看不买）",
}
TICKETS = {"2026-10-02": {"entry": 768.5, "stop": 767.0, "tp": 769.5}}
PLAYBOOK_DAYS = {"2024-10-14"}


def _round(o, nd=4):
    if isinstance(o, float):
        return round(o, nd) if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def _official_prior_day(lv: pd.DataFrame, m1: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Prior-day high/low/close and pivots from the daily bars when the minute file starts mid-day."""
    lv = lv.copy()
    counts = m1.groupby(m1.index.normalize()).size()
    for d in lv.index:
        prev = daily[daily.index < d.tz_localize(None)].iloc[-1]
        h, l, c = prev["high"], prev["low"], prev["close"]
        p = (h + l + c) / 3
        lv.loc[d, ["pdh", "pdl", "pdc", "pivot", "s1", "s2", "r1", "r2"]] = [h, l, c, p, 2 * p - h, p - (h - l), 2 * p - l, p + (h - l)]
        before = counts[counts.index < d]
        if not len(before) or before.iloc[-1] < 380:  # prior session incomplete in the minute file
            lv.loc[d, ["poc", "vah", "val"]] = np.nan
    return lv


def build_demo(dates: dict, leverage: float = 20.0) -> dict:
    store_m1 = store.load("SPY", "1m")
    if store_m1.empty:
        from spylev.data.minute_hist import build as build_minutes
        build_minutes()
        store_m1 = store.load("SPY", "1m")
    recent = load_ibkr_json(IBKR / "SPY_1m_recent.json")
    recent5 = load_ibkr_json(IBKR / "SPY_5m_recent.json")
    vix = load_ibkr_json(IBKR / "VIX_1d.json")["close"]
    daily, _ = spy_daily()
    pm = premarket_levels()
    days = []
    for d, label in dates.items():
        d0 = pd.Timestamp(d)
        new = d0 > store_m1.index[-1].tz_localize(None)
        m1 = recent if new else store_m1
        lv = _official_prior_day(daily_levels(m1), m1, daily) if new else None
        states = day_states(m1, daily, d, None if new else pm, user_leverage=leverage, levels=lv)
        bars = m1[m1.index.normalize().tz_localize(None) == d0]
        daily_setup = states[0]["daily"] if states else {}
        for s in states:
            s.pop("daily", None)
        # range desk: the 10:00 forecast for this day
        rv_src = recent5 if new else store_m1[store_m1.index.normalize().tz_localize(None) >= d0 - pd.Timedelta(days=14)]
        rv = daily_rv(rv_src[rv_src.index.normalize().tz_localize(None) < d0])
        desk = RangeDesk.from_results(leverage)
        vix_prev = float(vix[vix.index < d0].iloc[-1])
        desk.update(bars.iloc[: desk.model.t0], rv, vix_prev)
        rng = desk.screen()
        # playbook: noise-area breakout / overnight / RSI2 sleeves, minute by minute
        pb = Playbook.from_results()
        hist = m1[(m1.index.normalize().tz_localize(None) < d0) & (m1.index.normalize().tz_localize(None) >= d0 - pd.Timedelta(days=45))]
        pb.prepare(hist, d0)
        pb_record = pb.record
        for i, s in enumerate(states):
            st = pb.update(bars.iloc[: i + 1], rsi2_active=bool(daily_setup.get("active")))
            st.pop("record", None)
            s["playbook"] = st
        tk = TICKETS.get(d)
        record = rng.pop("record", None) if rng else None
        start = None if new else next((i for i, s in enumerate(states) if s["status"] in ("buy", "watch")), None)
        pb_in = next((i for i, s in enumerate(states) if (s["playbook"].get("noise") or {}).get("state") == "long"), None)
        focus = "alert" if start is not None else None
        if d in PLAYBOOK_DAYS and pb_in is not None:
            start, focus = pb_in, "playbook"
        if start is None:
            focus = "ticket" if TICKETS.get(d) else "range"
            start = _fill_index(bars, tk or (rng or {}).get("forecast", {}).get("auto"), desk.model.t0)
        days.append({
            "date": d, "label": label, "daily": daily_setup, "start": start, "focus": focus, "range": rng, "ticket": tk,
            "bars": [[int((t.tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds()),
                      round(r.open, 2), round(r.high, 2), round(r.low, 2), round(r.close, 2), int(r.volume)]
                     for t, r in bars.iterrows()],
            "states": states,
        })
    track = LiveEngine(daily, leverage).track_record()
    return _round({"track": track, "leverage": leverage, "range_record": record, "playbook_record": pb_record, "days": days})


def _fill_index(bars: pd.DataFrame, tk: dict | None, t0: int) -> int:
    """First minute a ticket's limit buy trades through, else the forecast minute."""
    if tk:
        mos = (bars.index.hour * 60 + bars.index.minute) - 570
        hit = np.flatnonzero((mos >= t0) & ((bars["open"].values <= tk["entry"]) | (bars["low"].values <= tk["entry"] - 0.01)))
        if len(hit):
            return int(hit[0])
    return t0 - 1


def write(demo: dict, out: Path) -> Path:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    blob = json.dumps(demo, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    assert "/*__DEMO__*/null" in page
    out.write_text(page.replace("/*__DEMO__*/null", blob), encoding="utf-8")
    return out


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WEB / "demo.html"))
    ap.add_argument("--dates", nargs="*", default=list(DAYS))
    ap.add_argument("--leverage", type=float, default=20.0)
    a = ap.parse_args(argv)
    dates = {d: DAYS.get(d, "回放日") for d in a.dates}
    out = write(build_demo(dates, a.leverage), Path(a.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
