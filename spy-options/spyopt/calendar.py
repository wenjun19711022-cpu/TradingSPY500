"""US equity session helpers (regular trading hours, early closes, SPY expiries)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from spyopt import TZ

RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)
RTH_MINUTES = 390

# NYSE scheduled 13:00 closes. Resampling never relies on this list (sessions are inferred
# from the bars themselves); it is used for live planning and minutes-to-close.
EARLY_CLOSES = {
    "2018-07-03", "2018-11-23", "2018-12-24",
    "2019-07-03", "2019-11-29", "2019-12-24",
    "2020-11-27", "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
}

# SPY added Tuesday/Thursday expiries in November 2022; from then on every session has a 0DTE.
SPY_DAILY_EXPIRY_START = pd.Timestamp("2022-11-14")


def session_close(day) -> dt.time:
    return EARLY_CLOSE if str(pd.Timestamp(day).date()) in EARLY_CLOSES else RTH_CLOSE


def session_minutes(day) -> int:
    return 210 if str(pd.Timestamp(day).date()) in EARLY_CLOSES else RTH_MINUTES


def has_0dte(day) -> bool:
    """True if SPY has an expiry on this session (Mon/Wed/Fri always; Tue/Thu since 2022-11)."""
    d = pd.Timestamp(day)
    if d >= SPY_DAILY_EXPIRY_START:
        return d.weekday() < 5
    return d.weekday() in (0, 2, 4)


def to_et(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(TZ)


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Bars whose start time lies inside 09:30 <= t < 16:00 ET."""
    idx = to_et(index)
    minutes = idx.hour * 60 + idx.minute
    return pd.Series((minutes >= 9 * 60 + 30) & (minutes < 16 * 60), index=index)


def _observed(d: pd.Timestamp) -> pd.Timestamp:
    return d - pd.Timedelta(days=1) if d.weekday() == 5 else d + pd.Timedelta(days=1) if d.weekday() == 6 else d


def _nth_weekday(year, month, weekday, n):
    days = [d for d in pd.date_range(f"{year}-{month:02d}-01", periods=31) if d.month == month and d.weekday() == weekday]
    return days[n] if n >= 0 else days[-1]


def nyse_holidays(year: int) -> set[str]:
    """Full-day NYSE closures for a year (rule-based; special closures like national days of
    mourning are not predictable and must be added by hand)."""
    from dateutil.easter import easter
    y = year
    hol = [
        _observed(pd.Timestamp(f"{y}-01-01")),
        _nth_weekday(y, 1, 0, 2),                        # MLK day
        _nth_weekday(y, 2, 0, 2),                        # Presidents' day
        pd.Timestamp(easter(y)) - pd.Timedelta(days=2),  # Good Friday
        _nth_weekday(y, 5, 0, -1),                       # Memorial day
        _observed(pd.Timestamp(f"{y}-07-04")),
        _nth_weekday(y, 9, 0, 0),                        # Labor day
        _nth_weekday(y, 11, 3, 3),                       # Thanksgiving
        _observed(pd.Timestamp(f"{y}-12-25")),
    ]
    if y >= 2022:
        hol.append(_observed(pd.Timestamp(f"{y}-06-19")))  # Juneteenth
    # New Year's Day on a Saturday is not observed on the prior Friday (Dec 31)
    return {str(d.date()) for d in hol if d.year == y}


def next_session(after, n: int) -> pd.Timestamp:
    """The n-th NYSE session strictly after `after`."""
    d = pd.Timestamp(after).normalize()
    count = 0
    while True:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5 and str(d.date()) not in nyse_holidays(d.year):
            count += 1
            if count == n:
                return d
