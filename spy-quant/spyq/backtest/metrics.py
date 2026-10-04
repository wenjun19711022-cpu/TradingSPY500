"""Trade and equity statistics, including the deflated Sharpe ratio.

Equity is simulated with fixed-fraction risk: every trade risks `risk_frac` of current
equity *at its maximum loss* (defined-risk structures only), so returns per trade = ror x
risk_frac. That makes strategies with very different premia comparable and keeps the
drawdown numbers meaningful for position sizing.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649


def equity_curve(ror: pd.Series, risk_frac=0.02, start=1.0) -> pd.Series:
    r = ror.fillna(0.0).values * risk_frac
    return pd.Series(start * np.cumprod(1 + r), index=ror.index)


def max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1).min())


def hac_se(x: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the mean (Bartlett weights)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    d = x - x.mean()
    var = d @ d / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / n
    return math.sqrt(max(var, 0.0) / n)


def trade_stats(trades: pd.DataFrame, risk_frac=0.02, n_trials: int = 1, years: float | None = None,
                overlap: int = 1) -> dict:
    """`overlap` = number of concurrently open trades (N for an N-session ladder): t-stats use
    Newey-West errors with N-1 lags, equity risks risk_frac/N per trade, DSR uses n/N."""
    if trades is None or trades.empty:
        return {"n": 0}
    ror = trades["ror"].astype(float)
    pnl = trades["pnl"].astype(float)
    n = len(ror)
    wins = pnl > 0
    gross_win = pnl[wins].sum()
    gross_loss = -pnl[~wins].sum()
    mu, sd = ror.mean(), ror.std(ddof=1) if n > 1 else 0.0
    se = hac_se(ror.values, overlap - 1) if overlap > 1 else (sd / math.sqrt(n) if n > 1 else float("nan"))
    t = mu / se if se and se > 0 else 0.0
    eq = equity_curve(ror, risk_frac / overlap)
    if years is None:
        years = max((ror.index[-1] - ror.index[0]).days / 365.25, 1 / 12) if n > 1 else 1.0
    tpy = n / years / overlap  # independent bets per year
    sr_trade = mu / sd if sd > 0 else 0.0
    n_eff = max(int(n / overlap), 3)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    mdd = max_drawdown(eq)
    q = ror.quantile(0.05)
    return {
        "n": int(n),
        "trades_per_year": float(tpy),
        "win_rate": float(wins.mean()),
        "avg_win_ror": float(ror[ror > 0].mean()) if (ror > 0).any() else 0.0,
        "avg_loss_ror": float(ror[ror <= 0].mean()) if (ror <= 0).any() else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "mean_ror": float(mu),
        "median_ror": float(ror.median()),
        "t_stat": float(t),
        "mean_pnl_per_contract": float(pnl.mean() * 100),
        "avg_credit_per_contract": float(trades["credit"].mean() * 100),
        "avg_risk_per_contract": float(trades["risk"].mean() * 100),
        "sharpe_ann": float(sr_trade * math.sqrt(tpy)),
        "sortino_ann": float(mu / ror[ror < 0].std(ddof=1) * math.sqrt(tpy)) if (ror < 0).sum() > 1 else 0.0,
        "cagr": float(cagr),
        "max_dd": float(mdd),
        "calmar": float(cagr / -mdd) if mdd < 0 else float("inf"),
        "cvar5_ror": float(ror[ror <= q].mean()),
        "worst_ror": float(ror.min()),
        "skew": float(skew(ror)) if n > 2 else 0.0,
        "n_losses": int((~wins).sum()),
        "dsr": float(deflated_sharpe(sr_trade, n_eff, skew(ror) if n > 2 else 0.0,
                                     kurtosis(ror, fisher=False) if n > 3 else 3.0, n_trials)),
        "final_equity": float(eq.iloc[-1]),
    }


def expected_max_sharpe(n_trials: int, var_sr: float = 1.0) -> float:
    """Expected maximum of n_trials iid N(0, var_sr) Sharpe estimates (Bailey & Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr: float, n: int, sk: float, ku: float, n_trials: int = 1, var_sr: float | None = None) -> float:
    """Probability that the true per-trade Sharpe exceeds the best of `n_trials` lucky nulls.

    sr is per-trade (not annualized); var_sr defaults to 1/n (the sampling variance of a
    null Sharpe estimate), which is the conservative textbook choice.
    """
    if n < 3:
        return 0.0
    var_sr = var_sr if var_sr is not None else 1.0 / n
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    g = trades.groupby(trades.index.year)
    return pd.DataFrame({
        "n": g.size(),
        "win_rate": g["pnl"].apply(lambda x: (x > 0).mean()),
        "mean_ror": g["ror"].mean(),
        "sum_ror": g["ror"].sum(),
        "pnl_per_contract": g["pnl"].sum() * 100,
    })
