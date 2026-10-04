"""Costs, splits, gates and leverage for the edge study.

Cost regimes (per side unless noted):
  okx_taker   OKX SPY perp, market orders: 0.05% fee + 0.01% slippage; resting targets pay maker
              0.02%; funding 0.01% per 8-hour settlement crossed (OKX set the TradFi interest
              component to 0% in March 2026, so real funding may be lower; this is conservative)
  okx_maker   same, but entries rest as limit orders (maker 0.02%); stops and the close still market
  mes         CME Micro E-mini S&P 500 (MES) through a futures broker: about 0.5bp per side
              (commission ~0.2bp + half a tick); overnight holds pay the futures carry
              (r - q) per calendar day instead of funding
Leverage: account return of a day = L x (sum of that day's net trade returns). With the whole
account as margin, a trade whose worst intraday drop reaches the liquidation distance at L wipes
the account (isolated perp, maintenance margin 0.5%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spylev.dip.perp import PerpCosts

# yearly average fed funds and S&P 500 dividend yield (%), for the futures carry of overnight holds
RATES = {2011: (0.10, 2.0), 2012: (0.14, 2.1), 2013: (0.11, 2.0), 2014: (0.09, 1.9), 2015: (0.13, 2.1),
         2016: (0.40, 2.1), 2017: (1.00, 1.9), 2018: (1.83, 1.9), 2019: (2.16, 1.9), 2020: (0.38, 1.8),
         2021: (0.08, 1.4), 2022: (3.00, 1.6), 2023: (5.02, 1.6), 2024: (5.14, 1.3), 2025: (4.20, 1.3),
         2026: (3.60, 1.2)}
REGIMES = ("okx_taker", "okx_maker", "mes")
LABELS = {"okx_taker": "OKX 永续·市价", "okx_maker": "OKX 永续·挂单进场", "mes": "CME 微型标普期货 MES"}
LEVS = (1, 2, 3, 5, 10, 20)


def _funding_settlements(t: pd.DataFrame) -> np.ndarray:
    """Number of OKX funding settlements (00:00, 08:00, 16:00 UTC) inside each holding period."""
    day = pd.DatetimeIndex(t["date"]).tz_localize("America/New_York")
    t_in = np.where(t["t_in"].values < 0,
                    day - pd.to_timedelta(t.get("nights", pd.Series(1, index=t.index)).fillna(1).values, unit="D") + pd.Timedelta(hours=16),
                    day + pd.to_timedelta(570 + t["t_in"].values, unit="m"))
    t_out = day + pd.to_timedelta(570 + np.where(t["t_in"].values < 0, 0, t["t_out"].values + 1), unit="m")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    k_in = (pd.DatetimeIndex(t_in, tz="UTC") - epoch) // pd.Timedelta(hours=8)
    k_out = (pd.DatetimeIndex(t_out).tz_convert("UTC") - epoch) // pd.Timedelta(hours=8)
    return np.asarray(k_out - k_in, dtype=float)


def net_returns(t: pd.DataFrame, regime: str, costs: PerpCosts | None = None) -> np.ndarray:
    c = costs or PerpCosts()
    if t.empty:
        return np.array([])
    g = t["gross"].values
    if regime == "mes":
        fee = np.full(len(t), 2 * 0.00005)
        if "nights" in t:
            yr = pd.DatetimeIndex(t["date"]).year
            rq = np.array([(RATES.get(y, (3.0, 1.5))[0] - RATES.get(y, (3.0, 1.5))[1]) / 100 for y in yr])
            fee = fee + rq / 365 * t["nights"].fillna(0).values
        return g - fee
    taker = c.taker + c.slippage
    entry = np.where((regime == "okx_maker") & (t["how_in"].values == "market"), c.maker, taker)
    if regime == "okx_maker" and "nights" in t:
        entry = np.full(len(t), c.maker)
    exit_ = np.where(t["how_out"].values == "target", c.maker, taker)
    if regime == "okx_maker" and "nights" in t:
        exit_ = np.full(len(t), c.maker)  # both legs of the overnight hold can rest as limits at the auction prices
    fund = _funding_settlements(t) * c.funding_8h
    return g - entry - exit_ - fund


def daily(t: pd.DataFrame, net: np.ndarray, days: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(net, index=pd.DatetimeIndex(t["date"])).groupby(level=0).sum() if len(t) else pd.Series(dtype=float)
    return s.reindex(days, fill_value=0.0)


def trade_stats(t: pd.DataFrame, net: np.ndarray) -> dict:
    if t is None or t.empty:
        return {"n": 0}
    g = t["gross"].values
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan  # noqa: E731
    wins = net > 0
    loss = -net[~wins].sum()
    return {"n": int(len(t)), "win_rate": float(wins.mean()), "gross_bp": float(g.mean() * 1e4),
            "gross_t": float(g.mean() / se(g)) if len(g) > 1 else 0.0, "net_bp": float(net.mean() * 1e4),
            "net_t": float(net.mean() / se(net)) if len(net) > 1 else 0.0,
            "profit_factor": float(net[wins].sum() / loss) if loss > 0 else float("inf"),
            "worst_bp": float(net.min() * 1e4), "worst_mae_bp": float(t["mae"].min() * 1e4)}


def leverage_table(t: pd.DataFrame, net: np.ndarray, days: pd.DatetimeIndex, levs=LEVS, mmr: float = 0.005) -> dict:
    """Account curves at several leverages, whole account as margin, liquidation on the worst drop."""
    out = {}
    if t.empty:
        return out
    years = (days[-1] - days[0]).days / 365.25
    for L in levs:
        liq = 1.0 if L <= 1 else 1 - (1 - 1 / L) / (1 - mmr)
        r = L * net
        wiped = t["mae"].values <= -liq
        r = np.where(wiped, -1.0, np.maximum(r, -1.0))
        d = daily(t, r, days)
        eq = (1 + d).clip(lower=0).cumprod()
        final = float(eq.iloc[-1])
        out[str(L)] = {"final": final, "cagr": float(final ** (1 / years) - 1) if final > 0 else -1.0,
                       "max_dd": float((eq / eq.cummax() - 1).min()), "worst_day": float(d.min()),
                       "liquidations": int(wiped.sum()), "ruined": bool(final <= 0.01)}
    return out


def kelly(d: pd.Series) -> dict:
    """Growth-optimal leverage from the daily 1x return series (mean / variance)."""
    m, v = d.mean(), d.var()
    f = m / v if v > 0 else 0.0
    sharpe = m / d.std() * np.sqrt(252) if d.std() > 0 else 0.0
    return {"kelly": float(f), "half_kelly": float(f / 2), "sharpe_1x": float(sharpe),
            "ann_return_1x": float(m * 252), "ann_vol_1x": float(d.std() * np.sqrt(252))}
