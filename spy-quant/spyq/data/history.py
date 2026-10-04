"""Long history for crash / leverage studies, from public GitHub datasets + the IBKR dumps.

    python -m spyq.data.history          # download, validate overlaps, write data/raw/history/

Sources (raw.githubusercontent.com; all Yahoo/CBOE-derived, research use only):
  SPY daily OHLC 1993-01-29 .. 2024-04  singhshubha/InvesterMaster  data/SPY.csv   (unadjusted)
  SPY daily OHLC 2021-10-04 .. latest   data/raw/ibkr/SPY_1d.json  (IBKR)        -> spliced
  S&P 500 daily close 1980 .. 2022-09   Louison22/M1_MBFA sp500.csv  (has Oct-1987)
  S&P 500 monthly 1871 .. latest        datasets/s-and-p-500 data/data.csv  (Shiller)
  VIX daily OHLC 1990 .. latest         datasets/finance-vix data/vix-daily.csv  (CBOE)

The splice keeps the older file up to the day before the IBKR series starts and checks that
closes agree on the overlap (they must match to the cent on unadjusted prices). Dividends are
NOT in these prices: SPY's ~1.3%/yr yield is ignored, which slightly understates a long-only
result held for months (and a perp position does not receive dividends either).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from spyq.data.sources import load_ibkr_json

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = RAW / "history"
URLS = {
    "spy_ohlc_1993": "https://raw.githubusercontent.com/singhshubha/InvesterMaster/HEAD/data/SPY.csv",
    "spx_daily_1980": "https://raw.githubusercontent.com/Louison22/M1_MBFA/HEAD/sp500.csv",
    "spx_monthly_1871": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "vix_daily_1990": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
}


def _get(url: str) -> pd.DataFrame:
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, url in URLS.items():
        df = _get(url)
        df.to_csv(out / f"{name}.csv", index=False)
        report[name] = len(df)
    return report


def _norm_daily(df, date_col, cols):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    d = pd.to_datetime(df[date_col.lower()].astype(str).str[:10])
    out = df[[c.lower() for c in cols]].astype(float)
    out.index = pd.DatetimeIndex(d)
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _ensure(out: Path = OUT):
    if not all((out / f"{k}.csv").exists() for k in URLS):
        download(out)


def spy_daily(out: Path = OUT) -> tuple[pd.DataFrame, dict]:
    """SPY unadjusted daily OHLC 1993 -> latest, with the splice check."""
    _ensure(out)
    old = _norm_daily(pd.read_csv(out / "spy_ohlc_1993.csv"), "date", ["open", "high", "low", "close"])
    ib = load_ibkr_json(RAW / "ibkr" / "SPY_1d.json")[["open", "high", "low", "close", "volume"]]
    ov = old.index.intersection(ib.index)
    diff = (old.loc[ov, "close"] - ib.loc[ov, "close"]).abs()
    check = {"overlap_days": int(len(ov)), "max_close_diff": float(diff.max()) if len(ov) else None,
             "days_over_1c": int((diff > 0.011).sum()) if len(ov) else None,
             "splice_at": str(ib.index[0].date())}
    merged = pd.concat([old[old.index < ib.index[0]], ib[["open", "high", "low", "close"]]])
    # repair rows where a source has open/high/low missing or inconsistent (very early SPY prints)
    hi = merged[["open", "high", "close"]].max(axis=1)
    lo = merged[["open", "low", "close"]].min(axis=1)
    merged["high"], merged["low"] = hi, lo
    return merged, check


def spx_daily(out: Path = OUT) -> pd.Series:
    _ensure(out)
    df = pd.read_csv(out / "spx_daily_1980.csv")
    s = pd.Series(df.iloc[:, 1].astype(float).values, index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])))
    return s.sort_index()


def spx_monthly(out: Path = OUT) -> pd.Series:
    _ensure(out)
    df = pd.read_csv(out / "spx_monthly_1871.csv")
    s = pd.Series(df["SP500"].astype(float).values, index=pd.DatetimeIndex(pd.to_datetime(df["Date"])))
    return s[s > 0].sort_index()


def vix_daily(out: Path = OUT) -> pd.DataFrame:
    _ensure(out)
    df = _norm_daily(pd.read_csv(out / "vix_daily_1990.csv"), "DATE", ["OPEN", "HIGH", "LOW", "CLOSE"])
    ib = load_ibkr_json(RAW / "ibkr" / "VIX_1d.json")["close"]
    df = df.reindex(df.index.union(ib.index))
    df["close"] = df["close"].fillna(ib)
    return df


def main():
    rep = download()
    print("downloaded", rep)
    spy, chk = spy_daily()
    print("SPY", spy.index[0].date(), "->", spy.index[-1].date(), len(spy), "splice", chk)
    vix = vix_daily()
    print("VIX", vix.index[0].date(), "->", vix.index[-1].date(), len(vix))
    print("SPX daily", spx_daily().index[[0, -1]].date, "monthly", spx_monthly().index[[0, -1]].date)
    spy.to_csv(OUT / "spy_daily_1993_latest.csv")
    print("missing business days (after splice):", int(np.sum(np.diff(spy.index.values).astype("timedelta64[D]").astype(int) > 4)))


if __name__ == "__main__":
    main()
