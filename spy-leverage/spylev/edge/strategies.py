"""Published intraday long strategies for SPY / the S&P 500, implemented exactly as specified by
their authors (no parameters fitted here), long side only.

  orb5        Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable?": if the first
              5-minute candle closes up, buy at the open of the second; stop at the first
              candle's low; target 10R; otherwise out at the close.
  noise       Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective Intraday Momentum
              Strategy for SPY": noise area = average absolute move from the open at each
              minute over the last 14 sessions; upper bound = max(open, prior close) x (1 + sigma).
              Checked every 30 minutes (10:00 ... 15:30): above the upper bound -> long;
              exit when the price is below max(upper bound, VWAP) at a check, or at the close.
  im_first30  Gao, Han, Li & Zhou (2018), "Market Intraday Momentum": if the return from the
              prior close to 10:00 is positive, hold the last half hour (15:30 -> close).
  im_rod      Baltussen, Da, Lammers & Martens (2021), "Hedging Demand and Market Intraday
              Momentum": if the return from the prior close to 15:30 is positive, hold the
              last half hour.
  overnight   Cliff, Cooper & Gulen (2008), Lou, Polk & Skouras (2019): hold from the close to the
              next open (needs a 24-hour instrument: futures or a perp).

Every function takes a Day panel (see `panel`) and returns a trade table with gross returns;
costs are applied in spylev.edge.evaluate. Fills: market orders at the next bar's open, stops at
the stop price (or the open if a bar opens through it), targets at the target price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spylev.calendar import nyse_holidays

N = 390
CHECKS = np.arange(29, 360, 30)  # bars that close at 10:00, 10:30, ..., 15:30


def _half_days(years) -> set:
    out = set()
    for y in years:
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-THU")
        out.add(str((nov[3] + pd.Timedelta(days=1)).date()))      # day after Thanksgiving
        for d in (pd.Timestamp(f"{y}-12-24"), pd.Timestamp(f"{y}-07-03")):
            if d.dayofweek < 5:
                out.add(str(d.date()))
    return out


@dataclass
class Panel:
    days: pd.DatetimeIndex     # session dates (naive)
    O: np.ndarray              # (D, 390) per-minute arrays
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    prev_close: np.ndarray     # (D,) previous session's close (NaN after a gap in the data)
    consecutive: np.ndarray    # (D,) previous row is the previous NYSE session
    gap_days: np.ndarray       # (D,) calendar days since the previous session


def panel(bars: pd.DataFrame) -> Panel:
    """1-minute RTH bars -> full-day matrices. Half days and days with < 300 bars are dropped."""
    day = bars.index.normalize()
    years = sorted(set(day.year))
    hol, half = set(), _half_days(years)
    for y in years:
        hol |= nyse_holidays(y)
    keep, mats = [], {k: [] for k in "OHLCV"}
    for d, g in bars.groupby(day):
        ds = str(d.date())
        if ds in half or ds in hol or len(g) < 300:
            continue
        mos = (g.index.hour * 60 + g.index.minute - 570).values
        ok = (mos >= 0) & (mos < N)
        g, mos = g[ok], mos[ok]
        c = pd.Series(np.nan, index=range(N))
        c.iloc[mos] = g["close"].values
        c = c.ffill().bfill().values
        row = {}
        for k, col in zip("OHLV", ("open", "high", "low", "volume")):
            a = np.full(N, np.nan)
            a[mos] = g[col].values
            row[k] = a
        o = np.where(np.isnan(row["O"]), c, row["O"])
        mats["O"].append(o)
        mats["H"].append(np.where(np.isnan(row["H"]), np.maximum(o, c), row["H"]))
        mats["L"].append(np.where(np.isnan(row["L"]), np.minimum(o, c), row["L"]))
        mats["C"].append(c)
        mats["V"].append(np.nan_to_num(row["V"], nan=0.0) + 1e-9)
        keep.append(d.tz_localize(None) if d.tzinfo else d)
    days = pd.DatetimeIndex(keep)
    if not keep:
        e = np.zeros((0, N))
        return Panel(days, e, e, e, e, e, np.zeros(0), np.zeros(0, bool), np.zeros(0))
    C = np.array(mats["C"])
    prev = np.r_[np.nan, C[:-1, -1]]
    # consecutive NYSE sessions (business days that are not holidays in between)
    bd = pd.bdate_range(days.min(), days.max())
    sessions = pd.DatetimeIndex([b for b in bd if str(b.date()) not in hol])
    pos = sessions.get_indexer(days)
    consec = np.r_[False, np.diff(pos) == 1]
    gap = np.r_[np.nan, np.diff(days.values).astype("timedelta64[D]").astype(float)]
    prev = np.where(consec, prev, np.nan)
    return Panel(days, np.array(mats["O"]), np.array(mats["H"]), np.array(mats["L"]), C, np.array(mats["V"]),
                 prev, consec, gap)


def _row(p: Panel, d, t_in, px_in, t_out, px_out, how_out, how_in="market", mae=np.nan, **extra) -> dict:
    return {"date": p.days[d], "t_in": int(t_in), "t_out": int(t_out), "entry": float(px_in), "exit": float(px_out),
            "gross": float(px_out / px_in - 1), "how_in": how_in, "how_out": how_out, "mae": float(mae), **extra}


def _walk(p: Panel, d, t0, entry, stop=None, target=None):
    """From bar t0 (entry at its open) to the close: first stop / target, else the close."""
    O, H, L, C = p.O[d], p.H[d], p.L[d], p.C[d]
    lo = entry
    for t in range(t0, N):
        if stop is not None:
            if O[t] <= stop and t > t0:
                return t, O[t], "stop", min(lo, O[t])
            if L[t] <= stop:
                return t, min(stop, O[t]) if t > t0 else stop, "stop", min(lo, stop)
        lo = min(lo, L[t])
        if target is not None and H[t] >= target and t > t0:
            return t, max(target, O[t]), "target", lo
    return N - 1, C[N - 1], "close", lo


def orb5(p: Panel, r_mult: float = 10.0) -> pd.DataFrame:
    rows = []
    for d in range(len(p.days)):
        o0, c4 = p.O[d, 0], p.C[d, 4]
        if not c4 > o0:
            continue
        stop = p.L[d, :5].min()
        entry = p.O[d, 5]
        risk = entry - stop
        if risk <= 0:
            continue
        t, px, how, lo = _walk(p, d, 5, entry, stop, entry + r_mult * risk)
        rows.append(_row(p, d, 5, entry, t, px, how, mae=lo / entry - 1, risk=risk / entry))
    return pd.DataFrame(rows)


def noise_sigma(p: Panel, lookback: int = 14) -> np.ndarray:
    move = np.abs(p.C / p.O[:, [0]] - 1)
    sig = pd.DataFrame(move).rolling(lookback).mean().shift(1).values  # previous 14 sessions only
    return sig


def noise(p: Panel, lookback: int = 14, mult: float = 1.0, every: int = 30) -> pd.DataFrame:
    """Published parameters: 14-day lookback, band = 1 x sigma, checks every 30 minutes from 10:00.
    The other values exist only for the robustness table (scripts/run_review.py)."""
    sig = noise_sigma(p, lookback) * mult
    checks = CHECKS if every == 30 else np.arange(29, 360, every)
    tp = (p.H + p.L + p.C) / 3
    vwap = np.cumsum(tp * p.V, axis=1) / np.cumsum(p.V, axis=1)
    rows = []
    for d in range(len(p.days)):
        if np.isnan(sig[d, 0]) or np.isnan(p.prev_close[d]):
            continue
        base = max(p.O[d, 0], p.prev_close[d])
        ub = base * (1 + sig[d])
        pos, t_in, px_in, lo = False, None, None, None
        for t in checks:
            c = p.C[d, t]
            if not pos and c > ub[t]:
                pos, t_in, px_in = True, t + 1, p.O[d, t + 1]
                lo = px_in
            elif pos:
                lo = min(lo, p.L[d, t_in:t + 1].min())
                if c < max(ub[t], vwap[d, t]):
                    rows.append(_row(p, d, t_in, px_in, t + 1, p.O[d, t + 1], "trail", mae=lo / px_in - 1))
                    pos = False
        if pos:
            lo = min(lo, p.L[d, t_in:].min())
            rows.append(_row(p, d, t_in, px_in, N - 1, p.C[d, N - 1], "close", mae=lo / px_in - 1))
    return pd.DataFrame(rows)


def _last_half_hour(p: Panel, signal_bar: int) -> pd.DataFrame:
    rows = []
    for d in range(len(p.days)):
        pc = p.prev_close[d]
        if np.isnan(pc) or not p.C[d, signal_bar] > pc:
            continue
        entry = p.O[d, 360]
        lo = p.L[d, 360:].min()
        rows.append(_row(p, d, 360, entry, N - 1, p.C[d, N - 1], "close", mae=lo / entry - 1))
    return pd.DataFrame(rows)


def im_first30(p: Panel) -> pd.DataFrame:
    return _last_half_hour(p, 29)


def im_rod(p: Panel) -> pd.DataFrame:
    return _last_half_hour(p, 359)


def overnight(p: Panel) -> pd.DataFrame:
    rows = []
    for d in range(1, len(p.days)):
        if not p.consecutive[d]:
            continue
        entry, exit_ = p.C[d - 1, N - 1], p.O[d, 0]
        rows.append({"date": p.days[d], "t_in": -1, "t_out": 0, "entry": float(entry), "exit": float(exit_),
                     "gross": float(exit_ / entry - 1), "how_in": "market", "how_out": "market",
                     "mae": float(min(exit_ / entry - 1, 0)), "nights": float(p.gap_days[d])})
    return pd.DataFrame(rows)


STRATEGIES = {"orb5": orb5, "noise": noise, "im_first30": im_first30, "im_rod": im_rod, "overnight": overnight}
NAMES = {"orb5": "5 分钟开盘区间突破（Zarattini & Aziz 2023）",
         "noise": "噪声区突破 + VWAP 跟踪止损（Zarattini, Aziz & Barbon 2024）",
         "im_first30": "日内动量：开盘半小时涨 → 买最后半小时（Gao 等 2018）",
         "im_rod": "日内动量：到 15:30 为涨 → 买最后半小时（Baltussen 等 2021）",
         "overnight": "隔夜持有：收盘买、次日开盘卖（Cliff 等 2008）"}
