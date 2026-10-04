"""Parquet bar store: data/bars/<SYMBOL>/<TF>/<YEAR>.parquet, index = bar start time (ET)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spyq import TZ
from spyq.calendar import to_et

ROOT = Path(__file__).resolve().parents[2] / "data" / "bars"
COLS = ["open", "high", "low", "close", "volume"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any bar frame to the canonical schema: ET tz-aware start-time index, OHLCV floats."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        for c in ("ts", "time", "time_key", "timestamp", "datetime", "date", "t"):
            if c in out.columns:
                out = out.set_index(c)
                break
    out.index = to_et(pd.to_datetime(out.index))
    out.index.name = "ts"
    for c in COLS:
        if c not in out.columns:
            out[c] = np.nan if c != "volume" else 0.0
    keep = COLS + [c for c in ("vwap", "trades") if c in out.columns]
    out = out[keep].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def path_for(symbol: str, tf: str, year: int, root: Path = ROOT) -> Path:
    return root / symbol.upper() / tf / f"{year}.parquet"


def save(df: pd.DataFrame, symbol: str, tf: str = "1m", root: Path = ROOT) -> list[Path]:
    """Merge bars into the yearly partitions (new rows win on duplicate timestamps)."""
    df = normalize(df)
    written = []
    for year, part in df.groupby(df.index.year):
        p = path_for(symbol, tf, int(year), root)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            old = pd.read_parquet(p)
            old.index = to_et(old.index)
            part = pd.concat([old, part])
            part = part[~part.index.duplicated(keep="last")].sort_index()
        part.to_parquet(p)
        written.append(p)
    return written


def load(symbol: str, tf: str = "1m", start=None, end=None, root: Path = ROOT) -> pd.DataFrame:
    folder = root / symbol.upper() / tf
    if not folder.exists():
        return pd.DataFrame(columns=COLS)
    years = sorted(int(p.stem) for p in folder.glob("*.parquet"))
    if start is not None:
        years = [y for y in years if y >= pd.Timestamp(start).year]
    if end is not None:
        years = [y for y in years if y <= pd.Timestamp(end).year]
    if not years:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_parquet(path_for(symbol, tf, y, root)) for y in years])
    df.index = to_et(df.index)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz=TZ)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)]
    return df.sort_index()


def coverage(symbol: str, tf: str = "1m", root: Path = ROOT) -> pd.Series:
    """Bars per session date — used by the crawler to skip months that are already complete."""
    df = load(symbol, tf, root=root)
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby(df.index.date).size()
