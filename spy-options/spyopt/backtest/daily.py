"""Daily-resolution option backtester on real SPY OHLC + VIX-family data.

Modes (entry -> expiry):
  0dte_open     buy/sell at today's open, expires at today's close. IV = VIX1D first print.
  1dte_close    trade at yesterday's close, expires at today's close. IV = VIX1D close.
  weekly_close  trade at the last session of a week, expires at the last session of the
                next week (SPY always lists that expiry).
  n{N}_close    trade at every close, expires N sessions later (laddered; before Nov-2022
                SPY only listed Mon/Wed/Fri expiries, so early dates are an approximation).
                Multi-day IV = VIX1D/VIX9D/VIX term structure interpolated to N sessions.

Exactness: hold-to-expiry P&L is exact given the model entry price (settlement only needs
the closing print). Stops/targets need the intraday path, which daily bars only bound:
  touch stop   short strike touched (low <= short put / high >= short call) => close the
               whole structure with the underlying AT the strike, 0.5 session left for
               intraday touches (1 full session if the open already gapped through), the put
               side marked at entry IV x 1.10 (spot-down / vol-up), all legs paying the spread.
               Both sides touched => the worse outcome is booked.
  tp50         (multi-day) closes at a session close once the structure marks <= 50% of
               the credit.
  tp50a        tp50 plus a market-level alarm (alarm_series): close every open rung at the
               first alarm close and open nothing new while it is on.
Run spyopt.backtest.minute on 1-minute data for path-exact stops.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from spyopt.calendar import has_0dte
from spyopt.data.rates import forward
from spyopt.options import bs
from spyopt.options.costs import CostModel
from spyopt.options.smile import Smile
from spyopt.options.structures import CREDIT, Leg, build, payoff, short_strikes, worst_payoff

DAY = 1.0 / 252
PUT_TOUCH_IV_BUMP = 1.10


@dataclass(frozen=True)
class Spec:
    mode: str = "0dte_open"
    structure: str = "iron_condor"   # see options.structures, plus "trend_spread" (put spread in an uptrend,
                                     # call spread in a downtrend, IC otherwise) and "trend_debit"
    k_put: float = 1.0
    k_call: float = 1.0
    wing: float = 0.5
    exit: str = "hold"               # hold | touch | tp50 | tp50a
    filt: str = "none"               # none | vrp | calm | noevent | gap | combo

    @property
    def key(self) -> str:
        return (f"{self.mode}|{self.structure}|kp{self.k_put:g}|kc{self.k_call:g}|w{self.wing:g}"
                f"|{self.exit}|{self.filt}")

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- market rows

def alarm_series(panel: pd.DataFrame, inversion=1.05, spike=1.35) -> pd.Series:
    """Market-level short-vol alarm at each close (rule fixed before testing, not tuned):
    VIX9D/VIX above 1.05 (near-term fear) or VIX 35% above its 5-session low."""
    v9, v = panel["vix9d_close"], panel["vix_close"]
    return ((v9 / v) > inversion) | ((v / v.rolling(5, min_periods=1).min()) > spike)


def market_rows(panel: pd.DataFrame, feats: pd.DataFrame, mode: str, smile: Smile,
                iv_source: str = "vix1d", proxy_ratio: float | None = None, offset: int = 0,
                ladder: bool = True) -> pd.DataFrame:
    """One row per tradable expiry with everything known at entry (and the outcome path).

    Features are taken as of the entry decision: yesterday's close for 0dte_open (plus the
    opening gap and the VIX1D open print), the entry close otherwise.
    """
    p = panel
    prev = p.shift(1)
    fprev = feats.shift(1)
    if mode == "0dte_open":
        idx = [d for d in p.index if has_0dte(d)]
        r = pd.DataFrame(index=pd.DatetimeIndex(idx))
        r["F"] = p["open"]
        r["T"] = DAY
        iv_index = p["vix1d_open"] if iv_source == "vix1d" else prev["vix9d_close"] * proxy_ratio
        r["atm"] = smile.atm_from_index(iv_index.reindex(r.index), 1)
        r["settle"] = p["close"]
        r["hi"], r["lo"], r["open_px"] = p["high"], p["low"], p["open"]
        r["vix"] = prev["vix_close"]
        f = fprev.reindex(r.index)
        r["vrp"] = r["atm"] * 100 / f["rv_session_ann"]
        r["event"] = f.get("event_next")
        r["calm"] = (f["ts_9d_30d"] < 1.0) & (f["ts_30d_3m"] < 1.0)
        r["trend"] = f["trend"]
        gap = np.log(p["open"] / prev["close"]).reindex(r.index)
        r["gap_z"] = gap.abs() / (r["atm"] * np.sqrt(DAY))
        r["gapped"] = False
        return r.dropna(subset=["F", "atm", "settle"])
    if mode == "1dte_close":
        idx = [d for d in p.index[1:] if has_0dte(d)]
        r = pd.DataFrame(index=pd.DatetimeIndex(idx))
        prev_date = pd.Series(p.index, index=p.index).shift(1).reindex(r.index)
        r["F"] = forward(prev["close"].reindex(r.index).values, prev_date.values, r.index)
        r["T"] = DAY
        iv_index = prev["vix1d_close"] if iv_source == "vix1d" else prev["vix9d_close"] * proxy_ratio
        r["atm"] = smile.atm_from_index(iv_index.reindex(r.index), 1)
        r["settle"] = p["close"]
        r["hi"], r["lo"], r["open_px"] = p["high"], p["low"], p["open"]
        r["vix"] = prev["vix_close"]
        f = fprev.reindex(r.index)
        r["vrp"] = r["atm"] * 100 / f["rv_cc_ann"]
        r["event"] = f.get("event_next")
        r["calm"] = (f["ts_9d_30d"] < 1.0) & (f["ts_30d_3m"] < 1.0)
        r["trend"] = f["trend"]
        r["gap_z"] = 0.0
        r["gapped"] = True
        return r.dropna(subset=["F", "atm", "settle"])
    if mode == "weekly_close" or (mode.startswith("n") and mode.endswith("_close")):
        if mode == "weekly_close":
            wk = p.index.to_period("W-FRI")
            last = p.groupby(wk).apply(lambda x: x.index[-1])
            entries, expiries = list(last.values[:-1]), list(last.values[1:])
        else:
            # laddered: a new N-session trade every day, so every tail event in the sample is
            # hit by N overlapping trades (statistics use HAC errors, equity splits risk by N)
            N = int(mode[1:-6])
            locs = np.arange(offset, len(p) - N, 1 if ladder else N)
            entries, expiries = list(p.index[locs]), list(p.index[locs + N])
        r = pd.DataFrame(index=pd.DatetimeIndex(expiries))
        r["entry"] = pd.DatetimeIndex(entries)
        r["n"] = [p.index.get_loc(b) - p.index.get_loc(a) for a, b in zip(entries, expiries)]
        r["T"] = r["n"] * DAY
        e = p.reindex(r["entry"])
        r["F"] = forward(e["close"].values, r["entry"].values, r.index)
        r["atm"] = smile.atm_term(e.get("vix1d_close", pd.Series(np.nan, index=e.index)).values,
                                  e["vix9d_close"].values, e["vix_close"].values, r["n"].values)
        r["settle"] = p["close"].reindex(r.index).values
        r["vix"] = e["vix_close"].values
        f = feats.reindex(r["entry"])
        r["vrp"] = r["atm"].values * 100 / f["rv_cc_ann"].values
        r["event"] = np.nan
        r["calm"] = ((f["ts_9d_30d"] < 1.0) & (f["ts_30d_3m"] < 1.0)).values
        r["trend"] = f["trend"].values
        r["gap_z"] = 0.0
        r["gapped"] = True
        return r.dropna(subset=["F", "atm", "settle"])
    raise ValueError(mode)


def filter_mask(rows: pd.DataFrame, filt: str, vrp_min=1.0, event_max=1.15, gap_max=0.75) -> np.ndarray:
    ok = np.ones(len(rows), dtype=bool)
    vrp = rows["vrp"].values
    ev = rows["event"].values.astype(float)
    if filt in ("vrp", "combo"):
        ok &= np.nan_to_num(vrp, nan=0.0) >= vrp_min
    if filt in ("calm", "combo"):
        ok &= rows["calm"].fillna(False).values.astype(bool)
    if filt in ("noevent", "combo"):
        ok &= ~(np.nan_to_num(ev, nan=0.0) > event_max)
    if filt == "gap":
        ok &= rows["gap_z"].values < gap_max
    return ok


# ----------------------------------------------------------------------------- pricing

def _value(legs, F, T, atm, smile: Smile, iv_override=None):
    vals = []
    for i, lg in enumerate(legs):
        iv = smile.iv(F, lg.strike, T, atm) if iv_override is None else iv_override[i]
        vals.append(bs.price(F, lg.strike, T, iv, lg.is_call))
    return vals


def _close_cash(legs, mids, costs: CostModel, vix):
    """Cash from closing every leg (sell longs at bid, buy back shorts at ask), per share."""
    cash = 0.0
    for lg, m in zip(legs, mids):
        cash = cash + lg.qty * costs.fill(m, -lg.qty, vix)
    return cash


def _settle_cash(legs, S, costs: CostModel, vix):
    """Expiry: OTM legs lapse for free; ITM legs are closed at intrinsic minus the spread."""
    cash = 0.0
    fees = 0.0
    for lg in legs:
        intr = np.maximum(S - lg.strike, 0) if lg.is_call else np.maximum(lg.strike - S, 0)
        itm = intr > 0
        hs = costs.half_spread(intr, vix)
        cash = cash + lg.qty * intr - np.where(itm, hs, 0.0)
        fees = fees + np.where(itm, costs.commission / costs.multiplier, 0.0)
    return cash - fees


def _legs_for(spec: Spec, rows: pd.DataFrame):
    F, atm, T = rows["F"].values, rows["atm"].values, rows["T"].values
    if spec.structure == "trend_debit":
        # uptrend -> bull call debit spread, downtrend -> bear put debit spread
        tr = rows["trend"].fillna(0).values
        bull = build("bull_call_debit", F, atm, T, wing=spec.wing)
        bear = build("bear_put_debit", F, atm, T, wing=spec.wing)
        legs = [Leg(True, lg.qty, np.where(tr > 0, lg.strike, 1e6)) for lg in bull]
        legs += [Leg(False, lg.qty, np.where(tr < 0, lg.strike, 1.0)) for lg in bear]
        return legs, tr
    if spec.structure not in ("trend_spread", "signal_spread"):
        return build(spec.structure, F, atm, T, spec.k_put, spec.k_call, spec.wing), None
    # per-day choice: +1 -> put spread, -1 -> call spread, 0 -> iron condor. The direction is
    # the EMA trend, or an external signal column (e.g. bottom/top calls from another model)
    ic = build("iron_condor", F, atm, T, spec.k_put, spec.k_call, spec.wing)
    col = "trend" if spec.structure == "trend_spread" else "signal"
    tr = rows[col].fillna(0).values
    # a dropped side gets both legs at the same absurd strike: zero price, zero payoff
    legs = []
    for lg in ic:
        k = lg.strike.copy().astype(float)
        if lg.is_call:
            k = np.where(tr > 0, 1e6, k)
        else:
            k = np.where(tr < 0, 1.0, k)
        legs.append(Leg(lg.is_call, lg.qty, k))
    return legs, tr


def simulate(rows: pd.DataFrame, spec: Spec, smile: Smile | None = None, costs: CostModel | None = None,
             panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-trade results (per share; x100 for one contract). Columns:
    credit (entry cash, + = received), risk (max loss incl. fees), pnl, ror = pnl / risk,
    exit (expiry|touch_put|touch_call|gap_stop|tp50)."""
    smile = smile or Smile()
    costs = costs or CostModel()
    m = filter_mask(rows, spec.filt)
    if spec.exit == "tp50a" and panel is not None and "entry" in rows:
        m &= ~alarm_series(panel).reindex(pd.DatetimeIndex(rows["entry"])).fillna(False).values
    if spec.structure == "trend_debit":
        m &= rows["trend"].fillna(0).values != 0
    if spec.structure == "signal_spread":
        m &= rows["signal"].fillna(0).values != 0  # trade only on signal days
    r = rows[m]
    if r.empty:
        return pd.DataFrame(columns=["credit", "risk", "pnl", "ror", "exit"])
    F, T, atm, vix = r["F"].values, r["T"].values, r["atm"].values, r["vix"].values
    legs, _ = _legs_for(spec, r)
    iv0 = [smile.iv(F, lg.strike, T, atm) for lg in legs]
    mids0 = [bs.price(F, lg.strike, T, iv, lg.is_call) for lg, iv in zip(legs, iv0)]
    n_real = sum((lg.strike > 2.0) & (lg.strike < 1e5) for lg in legs)
    comm = n_real * costs.commission / costs.multiplier
    entry_cash = sum(-lg.qty * costs.fill(mid, lg.qty, vix) for lg, mid in zip(legs, mids0))
    entry_cash = np.asarray(entry_cash, dtype=float)
    # max loss: worst expiry payoff + entry cash, minus entry and closing commissions
    risk = -(worst_payoff(legs) + entry_cash) + 2 * comm
    risk = np.maximum(risk, 0.01)

    S = r["settle"].values
    exit_cash = _settle_cash(legs, S, costs, vix)
    reason = np.array(["expiry"] * len(r), dtype=object)

    if spec.exit == "touch" and spec.mode in ("0dte_open", "1dte_close"):
        sp, sc = short_strikes(legs)
        sp = np.where((sp > 2.0) & (sp < 1e5), sp, np.nan)
        sc = np.where((sc > 2.0) & (sc < 1e5), sc, np.nan)
        lo, hi, op = r["lo"].values, r["hi"].values, r["open_px"].values
        gapped = r["gapped"].values.astype(bool)
        stop_cash = np.full(len(r), np.inf)
        stop_lab = np.array([""] * len(r), dtype=object)
        for side, K in (("put", sp), ("call", sc)):
            if np.all(np.isnan(K)):
                continue
            hit = (lo <= K) if side == "put" else (hi >= K)
            gap_thru = gapped & ((op <= K) if side == "put" else (op >= K))
            S_x = np.where(gap_thru, op, K)
            T_x = np.where(gap_thru, 1.0, 0.5) * DAY
            bump = PUT_TOUCH_IV_BUMP if side == "put" else 1.0
            ivs = [iv * bump for iv in iv0]
            mids = [bs.price(S_x, lg.strike, T_x, iv, lg.is_call) for lg, iv in zip(legs, ivs)]
            cash = np.asarray(_close_cash(legs, mids, costs, vix), dtype=float) - comm
            hit = hit & ~np.isnan(K)
            take = hit & (cash < stop_cash)  # both sides touched -> book the worse one
            stop_cash = np.where(take, cash, stop_cash)
            stop_lab = np.where(take, np.where(gap_thru, "gap_stop", f"touch_{side}"), stop_lab)
        stopped = np.isfinite(stop_cash)
        exit_cash = np.where(stopped, stop_cash, exit_cash)
        reason = np.where(stopped, stop_lab, reason)

    if spec.mode not in ("0dte_open", "1dte_close") and spec.exit in ("touch", "tp50", "tp50a") and panel is not None:
        exit_cash, reason = _walk_multiday(r, legs, iv0, entry_cash, exit_cash, reason, spec, smile, costs, panel, comm)

    pnl = entry_cash + exit_cash - comm
    out = pd.DataFrame({"F": F, "atm": atm, "credit": entry_cash, "risk": risk, "pnl": pnl,
                        "ror": pnl / risk, "exit": reason}, index=r.index)
    for i, lg in enumerate(legs):
        out[f"k{i}"] = lg.strike
    return out


def _walk_multiday(r, legs, iv0, entry_cash, exit_cash, reason, spec, smile, costs, panel, comm):
    """Day-by-day management of multi-day trades using each session's OHLC and the VIX term structure."""
    exit_cash = exit_cash.copy()
    reason = reason.copy()
    done = np.zeros(len(r), dtype=bool)
    pos = panel.index
    sp, sc = short_strikes(legs)
    sp = np.where((sp > 2.0) & (sp < 1e5), sp, np.nan)
    sc = np.where((sc > 2.0) & (sc < 1e5), sc, np.nan)
    entry_locs = pos.get_indexer(pd.DatetimeIndex(r["entry"].values))
    n = r["n"].values.astype(int)
    for j in range(1, int(n.max()) + 1):
        live = ~done & (j <= n)
        if not live.any():
            break
        locs = np.minimum(entry_locs + j, len(pos) - 1)
        day = panel.iloc[locs]
        expiry_dates = r.index
        prev = panel.iloc[locs - 1]
        o, h, l, c = (day[k].values for k in ("open", "high", "low", "close"))
        vix = prev["vix_close"].values
        v1p = prev["vix1d_close"].values if "vix1d_close" in prev else np.full(len(prev), np.nan)
        atm_prev = smile.atm_term(v1p, prev["vix9d_close"].values, prev["vix_close"].values, np.maximum(n - j + 1, 1))
        t_open = (n - j + 1) * DAY
        last_day = j == n
        if spec.exit == "touch":
            for side, K in (("put", sp), ("call", sc)):
                hit = live & ~done & ~last_day & ~np.isnan(K) & ((l <= K) if side == "put" else (h >= K))
                if not hit.any():
                    continue
                gap_thru = (o <= K) if side == "put" else (o >= K)
                S_x = np.where(gap_thru, o, K)
                T_x = np.where(gap_thru, t_open, t_open - 0.5 * DAY)
                bump = PUT_TOUCH_IV_BUMP if side == "put" else 1.0
                mids = _value(legs, S_x, T_x, atm_prev * bump, smile)
                cash = np.asarray(_close_cash(legs, mids, costs, vix), dtype=float) - comm
                exit_cash = np.where(hit, cash, exit_cash)
                reason = np.where(hit, np.where(gap_thru, "gap_stop", f"touch_{side}"), reason)
                done |= hit
        if spec.exit in ("tp50", "tp50a"):
            T_c = (n - j) * DAY
            v1c = day["vix1d_close"].values if "vix1d_close" in day else np.full(len(day), np.nan)
            atm_c = smile.atm_term(v1c, day["vix9d_close"].values, day["vix_close"].values, np.maximum(n - j, 1))
            F_c = forward(c, day.index, expiry_dates)
            mids = _value(legs, F_c, np.maximum(T_c, 1e-6), atm_c, smile)
            mark = -np.asarray(sum(lg.qty * m for lg, m in zip(legs, mids)), dtype=float)  # cost to close at mid
            hit = live & ~done & ~last_day & (entry_cash > 0) & (mark <= 0.5 * entry_cash)
            lab = np.full(len(hit), "tp50", dtype=object)
            if spec.exit == "tp50a":
                alarm = alarm_series(panel).reindex(day.index).fillna(False).values
                al = live & ~done & ~last_day & alarm & ~hit
                hit = hit | al
                lab = np.where(al, "alarm", lab)
            if hit.any():
                cash = np.asarray(_close_cash(legs, mids, costs, day["vix_close"].values), dtype=float) - comm
                exit_cash = np.where(hit, cash, exit_cash)
                reason = np.where(hit, lab, reason)
                done |= hit
    return exit_cash, reason


def is_credit(spec: Spec) -> bool:
    return spec.structure in CREDIT or spec.structure in ("trend_spread", "signal_spread")


def attach_signal(rows: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """Add an external direction signal (+1 bottom/bullish, -1 top/bearish, 0 none), indexed by
    the decision date, to market rows. For 0dte_open the signal must be known before the open;
    for close-entry modes it must be known by the entry close (no look-ahead is checked here)."""
    out = rows.copy()
    key = out["entry"] if "entry" in out else out.index
    out["signal"] = signals.reindex(pd.DatetimeIndex(key)).values
    return out


__all__ = ["Spec", "market_rows", "simulate", "filter_mask", "payoff"]
