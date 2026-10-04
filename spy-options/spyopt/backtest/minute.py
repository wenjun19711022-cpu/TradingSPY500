"""Path-exact intraday option engine on 1-minute SPY bars (for 0DTE research).

Run this on the full moomoo / Alpaca minute history (`python -m spyopt.data.crawl`); the daily
engine can only bound stops, this one walks every minute.

Model
-----
* entry at `entry` (HH:MM), spot = that minute's close; strikes from the expected move
* ATM vol for the rest of the session = VIX1D at the open (daily panel) mapped through the
  calibrated level ratio; time is a *variance clock*: the session's variance is spread over
  minutes by the empirical intraday profile (mean squared 1-minute return by minute, U-shaped)
  so 0DTE theta accrues faster through the open and close than through lunch
* sticky-strike marks: each leg keeps the IV it had at entry (puts the market falls into
  re-mark at their own higher skew IV)
* exits checked on every bar close: stop when the cost to close >= stop_mult x credit,
  take-profit when it <= (1 - tp) x credit, underlying touch of a short strike (bar high/low),
  time exit at `exit` (HH:MM); otherwise settle at the close (ITM legs closed at intrinsic)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from spyopt.calendar import to_et
from spyopt.options import bs
from spyopt.options.costs import CostModel
from spyopt.options.smile import Smile
from spyopt.options.structures import build, short_strikes, worst_payoff

DAY = 1.0 / 252


@dataclass(frozen=True)
class IntradaySpec:
    structure: str = "iron_condor"
    k_put: float = 1.0
    k_call: float = 1.0
    wing: float = 0.5
    entry: str = "09:35"
    exit: str | None = None          # "15:50" or None = settle at the close
    stop_mult: float | None = None   # e.g. 2.0: close when it costs 2x the credit
    tp: float | None = None          # e.g. 0.5: close at 50% of max profit
    touch: bool = False              # close when the underlying touches a short strike

    @property
    def key(self):
        return "|".join(f"{k}={v}" for k, v in asdict(self).items())


def minute_matrix(bars_1m: pd.DataFrame, field: str) -> pd.DataFrame:
    """days x 390 matrix (minute of session as columns), NaN where a bar is missing."""
    idx = to_et(bars_1m.index)
    mos = (idx.hour * 60 + idx.minute) - 570
    df = pd.DataFrame({"d": idx.normalize().tz_localize(None), "m": mos, "v": bars_1m[field].values})
    df = df[(df["m"] >= 0) & (df["m"] < 390)]
    return df.pivot_table(index="d", columns="m", values="v", aggfunc="last").reindex(columns=range(390))


def variance_profile(close: pd.DataFrame, min_days=20) -> np.ndarray:
    """Share of session variance per minute (sums to 1). U-shaped default with little data."""
    r = np.log(close).diff(axis=1)
    prof = (r**2).mean(axis=0).values
    if close.shape[0] < min_days or not np.isfinite(prof[1:]).all():
        m = np.arange(390)
        prof = 1 + 2.0 * np.exp(-m / 30) + 0.6 * np.exp(-(389 - m) / 30)
    prof = np.nan_to_num(prof, nan=np.nanmean(prof[1:]))
    prof[0] = prof[1:6].mean()
    return prof / prof.sum()


def _hhmm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m - 570


def simulate_intraday(bars_1m: pd.DataFrame, panel: pd.DataFrame, spec: IntradaySpec,
                      smile: Smile | None = None, costs: CostModel | None = None,
                      profile: np.ndarray | None = None) -> pd.DataFrame:
    smile = smile or Smile()
    costs = costs or CostModel()
    C = minute_matrix(bars_1m, "close")
    H = minute_matrix(bars_1m, "high").values
    L = minute_matrix(bars_1m, "low").values
    days = C.index
    prof = profile if profile is not None else variance_profile(C)
    remaining = np.r_[np.cumsum(prof[::-1])[::-1], 0.0]  # variance share left from minute m (bar start)
    e = _hhmm(spec.entry)
    x = _hhmm(spec.exit) if spec.exit else None
    pv = panel.reindex(days)
    v1 = pv["vix1d_open"].values if "vix1d_open" in pv else np.full(len(days), np.nan)
    atm = smile.atm_from_index(v1, 1)
    vix = (panel["vix_close"].shift(1).reindex(days).fillna(20.0).values if "vix_close" in panel
           else np.full(len(days), 20.0))
    S = C.values
    S0 = S[:, e]
    ok = np.isfinite(S0) & np.isfinite(atm)
    S, H, L, S0, atm, vix, days = S[ok], H[ok], L[ok], S0[ok], atm[ok], vix[ok], days[ok]
    # time left measured at the END of bar m (we act on bar closes)
    tau = lambda m: remaining[m + 1] * DAY  # noqa: E731
    T0 = tau(e)
    legs = build(spec.structure, S0, atm, T0, spec.k_put, spec.k_call, spec.wing)
    ivs = [smile.iv(S0, lg.strike, T0, atm) for lg in legs]
    mid0 = [bs.price(S0, lg.strike, T0, iv, lg.is_call) for lg, iv in zip(legs, ivs)]
    comm = len(legs) * costs.commission / costs.multiplier
    entry_cash = sum(-lg.qty * costs.fill(m, lg.qty, vix) for lg, m in zip(legs, mid0))
    credit_mid = sum(-lg.qty * m for lg, m in zip(legs, mid0))
    risk = np.maximum(-(worst_payoff(legs) + entry_cash) + 2 * comm, 0.01)
    sp, sc = short_strikes(legs)
    n = len(days)
    exit_cash = np.full(n, np.nan)
    exit_min = np.full(n, -1)
    reason = np.array(["close"] * n, dtype=object)
    open_ = np.ones(n, dtype=bool)
    last = 389
    for m in range(e + 1, 390):
        if not open_.any():
            break
        s = S[:, m]
        valid = open_ & np.isfinite(s)
        T = tau(m)
        mids = [bs.price(np.where(valid, s, S0), lg.strike, T, iv, lg.is_call) for lg, iv in zip(legs, ivs)]
        to_close = -sum(lg.qty * mm for lg, mm in zip(legs, mids))  # debit to buy back (credit structures)
        hit = np.zeros(n, dtype=bool)
        lab = np.array([""] * n, dtype=object)
        if spec.touch:
            t_hit = valid & (((L[:, m] <= sp) & np.isfinite(sp)) | ((H[:, m] >= sc) & np.isfinite(sc)))
            hit |= t_hit
            lab = np.where(t_hit, "touch", lab)
        if spec.stop_mult is not None:
            s_hit = valid & ~hit & (credit_mid > 0) & (to_close >= spec.stop_mult * credit_mid)
            hit |= s_hit
            lab = np.where(s_hit, "stop", lab)
        if spec.tp is not None:
            p_hit = valid & ~hit & (credit_mid > 0) & (to_close <= (1 - spec.tp) * credit_mid)
            hit |= p_hit
            lab = np.where(p_hit, "tp", lab)
        if x is not None and m >= x:
            t_x = valid & ~hit
            hit |= t_x
            lab = np.where(t_x, "time", lab)
        if hit.any() and m < last:
            cash = sum(lg.qty * costs.fill(mm, -lg.qty, vix) for lg, mm in zip(legs, mids)) - comm
            exit_cash = np.where(hit, cash, exit_cash)
            exit_min = np.where(hit, m, exit_min)
            reason = np.where(hit, lab, reason)
            open_ &= ~hit
    # settle the rest at the session's last close
    settle = np.array([row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.nan for row in S])
    fees = 0.0
    payoff_cash = 0.0
    for lg in legs:
        intr = np.maximum(settle - lg.strike, 0) if lg.is_call else np.maximum(lg.strike - settle, 0)
        itm = intr > 0
        payoff_cash = payoff_cash + lg.qty * intr - np.where(itm, costs.half_spread(intr, vix), 0.0)
        fees = fees + np.where(itm, costs.commission / costs.multiplier, 0.0)
    exit_cash = np.where(open_, payoff_cash - fees, exit_cash)
    pnl = entry_cash + exit_cash - comm
    return pd.DataFrame({"S0": S0, "atm": atm, "credit": entry_cash, "risk": risk, "pnl": pnl,
                         "ror": pnl / risk, "exit": reason,
                         "exit_time": [f"{(570 + m) // 60:02d}:{(570 + m) % 60:02d}" if m >= 0 else "16:00" for m in exit_min]},
                        index=pd.DatetimeIndex(days))
