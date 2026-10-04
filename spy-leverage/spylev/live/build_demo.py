"""Bake replay days into web/index.html -> a standalone demo page (no Python, no OpenD needed).

    python -m spylev.live.build_demo                       # -> web/demo.html
    python -m spylev.live.build_demo --out /tmp/demo.html --dates 2023-08-17 2024-03-12

Each day is run minute by minute through the same LiveEngine the live monitor uses, so the
demo shows exactly what the screen would have said on that day.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from spylev.data import store
from spylev.data.history import spy_daily
from spylev.data.minute_hist import premarket_levels
from spylev.live.engine import LiveEngine, day_states

WEB = Path(__file__).resolve().parents[2] / "web"
DAYS = {
    "2023-08-17": "抄底日：日线 RSI2<10，1/3/5 分钟共振时买入 2 倍",
    "2024-03-12": "普通日：只有 1/3/5 分钟共振（只看不买）",
}


def _round(o, nd=4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def build_demo(dates: dict, leverage: float = 10.0) -> dict:
    m1 = store.load("SPY", "1m")
    if m1.empty:
        from spylev.data.minute_hist import build as build_minutes
        build_minutes()
        m1 = store.load("SPY", "1m")
    daily, _ = spy_daily()
    pm = premarket_levels()
    days = []
    for d, label in dates.items():
        states = day_states(m1, daily, d, pm, user_leverage=leverage)
        bars = m1[m1.index.normalize().tz_localize(None) == pd.Timestamp(d)]
        daily_setup = states[0]["daily"] if states else {}
        for s in states:
            s.pop("daily", None)
        start = next((i for i, s in enumerate(states) if s["status"] in ("buy", "watch")), len(states) - 1)
        days.append({
            "date": d, "label": label, "daily": daily_setup, "start": start,
            "bars": [[int((t.tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds()),
                      round(r.open, 2), round(r.high, 2), round(r.low, 2), round(r.close, 2), int(r.volume)]
                     for t, r in bars.iterrows()],
            "states": states,
        })
    track = LiveEngine(daily, leverage).track_record()
    return _round({"track": track, "leverage": leverage, "days": days})


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
    ap.add_argument("--leverage", type=float, default=10.0)
    a = ap.parse_args(argv)
    dates = {d: DAYS.get(d, "回放日") for d in a.dates}
    out = write(build_demo(dates, a.leverage), Path(a.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
