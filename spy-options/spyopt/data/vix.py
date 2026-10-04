"""Daily market panel: SPY OHLC + VIX family (VIX1D, VIX9D, VIX, VIX3M).

Two loaders feed the same frame:
  * IBKR JSON dumps in data/raw/ibkr (what the bundled study uses), and
  * CBOE's free CSVs (https://cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv),
    fetched with `download_cboe()` on a machine that can reach cdn.cboe.com.

Column semantics (all are prints of the index, not tradable prices):
  vix1d_open   first VIX1D print after 09:30 = implied vol for the rest of today's session
  vix1d_close  VIX1D at the close = implied vol for close(t) -> close(t+1) (next-day options)
  vix9d/vix/vix3m_close   9-day / 30-day / 3-month implied vol at the close
VIX's own "open" from IBKR is the 03:15 global-trading-hours print, so only closes are used.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from spyopt.data.sources import load_ibkr_json

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"


def download_cboe(names=("VIX", "VIX9D", "VIX1D", "VIX3M"), out=RAW / "cboe"):
    import requests
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        r = requests.get(CBOE_URL.format(name=n), timeout=60)
        r.raise_for_status()
        (out / f"{n}_History.csv").write_bytes(r.content)


def _cboe_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.index = pd.to_datetime(df.pop("date"))
    return df.astype(float)


def load_panel(raw: Path = RAW) -> pd.DataFrame:
    """SPY daily OHLC joined with VIX-family closes (and VIX1D open). Index = session date."""
    ib = raw / "ibkr"
    cboe = raw / "cboe"
    spy = load_ibkr_json(ib / "SPY_1d.json")
    panel = spy[["open", "high", "low", "close", "volume"]].copy()
    for name in ("VIX1D", "VIX9D", "VIX", "VIX3M"):
        src = None
        if (cboe / f"{name}_History.csv").exists():
            src = _cboe_csv(cboe / f"{name}_History.csv")
        elif (ib / f"{name}_1d.json").exists():
            src = load_ibkr_json(ib / f"{name}_1d.json")
        if src is None:
            continue
        key = name.lower()
        panel[f"{key}_close"] = src["close"].reindex(panel.index)
        if name == "VIX1D":
            panel[f"{key}_open"] = src["open"].reindex(panel.index)
    panel.index.name = "date"
    return panel
