"""Resumable crawler: 5 years of SPY 1-minute RTH bars -> parquet -> every timeframe.

    python -m spyq.data.crawl --source futu --years 5            # moomoo OpenD
    python -m spyq.data.crawl --source alpaca --start 2021-10-01  # Alpaca SIP
    python -m spyq.data.crawl --csv old_moomoo_export.csv --end-labelled
    python -m spyq.data.crawl --rebuild-only                     # just re-resample

The crawler walks month by month, skips months whose sessions already have >= 385 bars,
merges new rows into data/bars/SPY/1m/<year>.parquet, then rebuilds 3m, 5m, 10m, 15m,
30m, 1h, 2h, 4h and 1d from the 1-minute store and writes an integrity report.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from spyq.data import store
from spyq.data.resample import DEFAULT_SET, integrity_report, resample
from spyq.data.sources import SOURCES, CSVSource


def month_windows(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int):
    cur = start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)


def crawl(source_names, symbol="SPY", start=None, end=None, min_bars=385, verbose=True):
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    start = pd.Timestamp(start or end - pd.DateOffset(years=5))
    have = store.coverage(symbol, "1m")
    complete = set(d for d, n in have.items() if n >= min_bars)
    for name in source_names:
        try:
            src = SOURCES[name]()
        except Exception as e:  # missing package / credentials
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        for a, b in month_windows(start, end, src.chunk_days):
            days = pd.bdate_range(a, b)
            if all(d.date() in complete for d in days):
                continue
            try:
                df = src.fetch(symbol, a, b)
            except Exception as e:
                print(f"[{name}] {a.date()}..{b.date()} failed: {e}", file=sys.stderr)
                continue
            if df is None or df.empty:
                continue
            store.save(df, symbol, "1m")
            complete |= set(d for d, n in df.groupby(df.index.date).size().items() if n >= min_bars)
            if verbose:
                print(f"[{name}] {a.date()}..{b.date()} +{len(df)} bars")
    return store.coverage(symbol, "1m")


def rebuild(symbol="SPY", tfs=None):
    bars = store.load(symbol, "1m")
    if bars.empty:
        print("no 1m bars in the store yet")
        return {}
    out = {}
    for tf in tfs or DEFAULT_SET:
        if tf == "1m":
            continue
        rs = resample(bars, tf)
        store.save(rs.drop(columns=["n_bars"]), symbol, tf)
        out[tf] = len(rs)
    rep = integrity_report(bars)
    rep_path = store.ROOT / symbol / "integrity_1m.csv"
    rep.to_csv(rep_path)
    bad = rep[~rep["ok"]]
    print(f"sessions={len(rep)} clean={int(rep['ok'].sum())} flagged={len(bad)} -> {rep_path}")
    for tf, n in out.items():
        print(f"  {tf:>4}: {n} bars")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], choices=sorted(SOURCES))
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--csv", help="import an existing CSV/parquet export instead of crawling")
    ap.add_argument("--end-labelled", action="store_true", help="CSV labels bars by close time (moomoo)")
    ap.add_argument("--rebuild-only", action="store_true")
    a = ap.parse_args(argv)
    if a.csv:
        df = CSVSource(a.csv, end_labelled=a.end_labelled).fetch(a.symbol, a.start, a.end)
        store.save(df, a.symbol, "1m")
        print(f"imported {len(df)} bars from {a.csv}")
    elif not a.rebuild_only:
        end = pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()
        start = pd.Timestamp(a.start) if a.start else end - pd.DateOffset(years=a.years)
        crawl(a.source or ["futu"], a.symbol, start, end)
    rebuild(a.symbol)


if __name__ == "__main__":
    main()
