"""Public SPY 1-minute history (2022-10-03 .. 2024-12-30, incl. extended hours) for the scalp study.

    python -m spylev.data.minute_hist     # download -> data/bars/SPY/1m/*.parquet + premarket levels

Source: GerardWu100/vol-forecast-benchmarks data/raw/SPY_minutes.parquet on GitHub (454,835
bars, Yahoo-style dividend-adjusted prices). Each session is rescaled to *raw* prices with
the factor raw_close(IBKR daily) / adjusted last regular-hours close, so levels (prior-day
low, round numbers, pivots) match what the chart showed that day. Regular-hours bars go to
the 1m store; 04:00-09:29 bars are reduced to per-day premarket high/low.

Your own moomoo history (python -m spylev.data.crawl --source futu) extends this to
2025-2026; the scalp study automatically uses whatever the store holds.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from spylev.data import store
from spylev.data.sources import load_ibkr_json

ROOT = Path(__file__).resolve().parents[2]
URL = "https://raw.githubusercontent.com/GerardWu100/vol-forecast-benchmarks/HEAD/data/raw/SPY_minutes.parquet"
CACHE = ROOT / "data" / "raw" / "history" / "SPY_minutes_2022_2024.parquet"
PREMARKET = ROOT / "data" / "bars" / "SPY" / "premarket_levels.parquet"


def download() -> Path:
    import requests
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        r = requests.get(URL, timeout=600)
        r.raise_for_status()
        CACHE.write_bytes(r.content)
    return CACHE


def build() -> dict:
    df = pd.read_parquet(download()).set_index("ts").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    mins = df.index.hour * 60 + df.index.minute
    rth = df[(mins >= 570) & (mins < 960)][["open", "high", "low", "close", "volume"]]
    pre = df[(mins >= 240) & (mins < 570)][["open", "high", "low", "close", "volume"]]
    last = rth["close"].groupby(rth.index.normalize().tz_localize(None)).last()
    raw = load_ibkr_json(ROOT / "data" / "raw" / "ibkr" / "SPY_1d.json")["close"]
    factor = (raw.reindex(last.index) / last).ffill().bfill()
    f_rth = factor.reindex(rth.index.normalize().tz_localize(None)).values
    for c in ("open", "high", "low", "close"):
        rth[c] = rth[c] * f_rth
    f_pre = factor.reindex(pre.index.normalize().tz_localize(None)).values
    pre = pre.assign(high=pre["high"] * f_pre, low=pre["low"] * f_pre)
    store.save(rth, "SPY", "1m")
    g = pre.groupby(pre.index.normalize())
    lv = pd.DataFrame({"pmh": g["high"].max(), "pml": g["low"].min()})
    PREMARKET.parent.mkdir(parents=True, exist_ok=True)
    lv.to_parquet(PREMARKET)
    return {"rth_bars": len(rth), "sessions": int(rth.index.normalize().nunique()),
            "factor_range": [float(factor.min()), float(factor.max())], "premarket_days": len(lv)}


def premarket_levels() -> pd.DataFrame:
    return pd.read_parquet(PREMARKET) if PREMARKET.exists() else pd.DataFrame(columns=["pmh", "pml"])


if __name__ == "__main__":
    print(build())
