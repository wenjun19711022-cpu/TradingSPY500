"""Carry inputs for option forwards: short rate and SPY's quarterly dividends.

F = S * exp(r * calendar_days / 365) - (dividend if an ex-date falls in (entry, expiry]).

Ignoring carry overprices puts and underprices calls — on a 20-session 2-EM iron condor that
inflates the credit by ~20%, so it matters. The built-in rate table is an approximate monthly
3-month T-bill path (percent); replace it with FRED DTB3 by dropping a CSV with columns
DATE,DTB3 at data/raw/rates/DTB3.csv (download_fred() does it where fred.stlouisfed.org is
reachable). The 2026 level is set from SPY put-call parity on the 2026-10-02 snapshot
(~4.0% on the Oct-30 expiry, which has no ex-dividend date).
SPY goes ex-dividend on the third Friday of Mar/Jun/Sep/Dec; ~0.32% of price per quarter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "rates"

_TABLE = {
    "2021-10": 0.05, "2021-11": 0.05, "2021-12": 0.05,
    "2022-01": 0.15, "2022-02": 0.33, "2022-03": 0.44, "2022-04": 0.76, "2022-05": 1.05, "2022-06": 1.52,
    "2022-07": 2.23, "2022-08": 2.63, "2022-09": 3.13, "2022-10": 3.72, "2022-11": 4.15, "2022-12": 4.25,
    "2023-01": 4.54, "2023-02": 4.65, "2023-03": 4.69, "2023-04": 4.92, "2023-05": 5.14, "2023-06": 5.16,
    "2023-07": 5.25, "2023-08": 5.30, "2023-09": 5.32, "2023-10": 5.34, "2023-11": 5.27, "2023-12": 5.24,
    "2024-01": 5.22, "2024-02": 5.24, "2024-03": 5.24, "2024-04": 5.24, "2024-05": 5.25, "2024-06": 5.24,
    "2024-07": 5.20, "2024-08": 5.05, "2024-09": 4.72, "2024-10": 4.51, "2024-11": 4.42, "2024-12": 4.27,
    "2025-01": 4.21, "2025-02": 4.22, "2025-03": 4.20, "2025-04": 4.21, "2025-05": 4.25, "2025-06": 4.23,
    "2025-07": 4.25, "2025-08": 4.15, "2025-09": 3.95, "2025-10": 3.85, "2025-11": 3.80, "2025-12": 3.75,
}
_DEFAULT_LATE = 4.0
DIV_YIELD_Q = 0.0032


def download_fred(series="DTB3"):
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=60)
    r.raise_for_status()
    (RAW / f"{series}.csv").write_bytes(r.content)


def short_rate(dates) -> pd.Series:
    """Annualized continuously-compounded short rate (decimal) for each date."""
    idx = pd.DatetimeIndex(dates)
    f = RAW / "DTB3.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").ffill() / 100
        return s.reindex(idx, method="ffill").fillna(_DEFAULT_LATE / 100)
    keys = idx.strftime("%Y-%m")
    return pd.Series([_TABLE.get(k, _DEFAULT_LATE if k > "2025-12" else 0.05) / 100 for k in keys], index=idx)


def ex_dividend_dates(start="2015-01-01", end="2030-12-31") -> pd.DatetimeIndex:
    out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for m in (3, 6, 9, 12):
            fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
            fridays = [d for d in fridays if d.month == m and d.weekday() == 4]
            out.append(fridays[2])
    return pd.DatetimeIndex(out)


def forward(spot, entry_dates, expiry_dates) -> np.ndarray:
    """Forward for each (entry close -> expiry close) pair, with rate carry and dividends."""
    entry = pd.DatetimeIndex(entry_dates)
    expiry = pd.DatetimeIndex(expiry_dates)
    spot = np.asarray(spot, dtype=float)
    r = short_rate(entry).values
    cal = (expiry - entry).days.values.astype(float)
    exd = ex_dividend_dates().values
    has_div = np.searchsorted(exd, expiry.values, side="right") > np.searchsorted(exd, entry.values, side="right")
    return spot * np.exp(r * cal / 365.0) - np.where(has_div, spot * DIV_YIELD_Q, 0.0)
