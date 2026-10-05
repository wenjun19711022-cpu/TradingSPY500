"""Forecast the rest of the session's range at a fixed time after the open, and price brackets.

At the decision minute (default 10:00 ET, 30 minutes after the open) we know the open, the
high and low so far, the price, the realized volatility of the first 30 minutes, the last days'
realized volatility and yesterday's VIX close. From these we forecast the volatility of the rest
of the session (log-linear fit on in-sample days) and turn it into a band with the in-sample
distribution of how far the rest of the day actually went up and down, measured in units of the
forecast. That band is the answer to "it won't break 772 and won't break 767" - with a stated
probability instead of hindsight.

Realized variance uses 5-minute log returns (the usual choice: less bid-ask noise than 1m).

First-passage probabilities treat log price as a driftless Brownian motion whose variance runs
on the intraday volatility clock (busy open, quiet lunch, busy close). They are what a
no-edge market implies - the same idea as moomoo's "profit probability" for an option spread.
Anything we claim above them has to show up in the backtest.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from spylev.data.resample import resample

SLOTS = 78                      # 5-minute bars in a full session
PROB_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FEATURES = ("vix", "rv5", "rv1", "open")


# ----------------------------------------------------------------------------- per-day table

def five_minute(bars: pd.DataFrame) -> pd.DataFrame:
    """1m -> 5m session-anchored bars; 5m input passes through."""
    step = pd.Series(bars.index).diff().dt.total_seconds().median()
    return bars if step and step >= 299 else resample(bars, "5m")


def slot_returns(b5: pd.DataFrame) -> pd.DataFrame:
    """5m log returns within each session (first slot = close / open: no overnight gap)."""
    day = b5.index.normalize()
    prev = b5["close"].groupby(day).shift()
    r = np.log(b5["close"] / prev.fillna(b5["open"]))
    slot = ((b5.index.hour * 60 + b5.index.minute) - 570) // 5
    return pd.DataFrame({"day": day, "slot": slot, "r": r.values}, index=b5.index)


def var_profile(b5: pd.DataFrame) -> np.ndarray:
    """Share of a full session's variance that falls in each 5-minute slot (sums to 1)."""
    sr = slot_returns(b5)
    full = sr.groupby("day")["slot"].transform("count") == SLOTS
    prof = (sr[full]["r"] ** 2).groupby(sr[full]["slot"]).mean().reindex(range(SLOTS)).fillna(0).values
    return prof / prof.sum()


def session_table(b5: pd.DataFrame, vix: pd.Series, t0: int = 30) -> pd.DataFrame:
    """One row per session: what was known at minute t0 and what the rest of the day did."""
    k0 = t0 // 5
    sr = slot_returns(b5)
    rows = []
    for day, g in b5.groupby(b5.index.normalize()):
        r = sr.loc[g.index, "r"].values
        if len(g) <= k0:
            continue
        rows.append({
            "date": day.tz_localize(None) if day.tzinfo else day, "n_slots": len(g),
            "open": g["open"].iloc[0], "s0": g["close"].iloc[k0 - 1],
            "h0": g["high"].iloc[:k0].max(), "l0": g["low"].iloc[:k0].min(),
            "rv_open": float((r[:k0] ** 2).sum()), "rv_rest": float((r[k0:] ** 2).sum()),
            "rv_day": float((r ** 2).sum()),
            "rest_high": g["high"].iloc[k0:].max(), "rest_low": g["low"].iloc[k0:].min(),
            "close": g["close"].iloc[-1],
        })
    t = pd.DataFrame(rows).set_index("date")
    t["rv1"] = t["rv_day"].shift()
    t["rv5"] = t["rv_day"].rolling(5).mean().shift()
    v = vix.copy()
    v.index = pd.DatetimeIndex(v.index).tz_localize(None).normalize()
    # VIX close of the previous trading day (the series may have dates the bars don't, and vice versa)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    t["vix_prev"] = v.reindex(v.index.union(t.index)).ffill().shift().reindex(t.index).values
    t["full"] = t["n_slots"] == SLOTS
    return t


# ----------------------------------------------------------------------------- the model

@dataclass
class RangeModel:
    t0: int = 30
    profile: list = field(default_factory=list)       # variance share per 5m slot
    beta: list = field(default_factory=list)          # intercept + FEATURES
    qu: dict = field(default_factory=dict)            # p -> quantile of up-move / forecast sigma
    qd: dict = field(default_factory=dict)            # p -> quantile of down-move / forecast sigma
    joint: dict = field(default_factory=dict)         # p -> share of days inside both sides
    fit_days: int = 0
    r2: float = 0.0

    # ---- shares of the variance clock
    def share(self, a_min: float, b_min: float = 390) -> float:
        """Share of a session's variance between minute a and minute b of the session."""
        prof = np.asarray(self.profile)
        edges = np.arange(SLOTS + 1) * 5.0
        cum = np.r_[0, np.cumsum(prof)]
        f = lambda m: float(np.interp(min(max(m, 0), 390), edges, cum))  # noqa: E731
        return max(f(b_min) - f(a_min), 0.0)

    def x(self, row) -> np.ndarray:
        w_rest, w_open = self.share(self.t0), self.share(0, self.t0)
        vix_var = (float(row["vix_prev"]) / 100) ** 2 / 252
        return np.array([1.0, np.log(vix_var * w_rest), np.log(float(row["rv5"]) * w_rest),
                         np.log(float(row["rv1"]) * w_rest), np.log(max(float(row["rv_open"]), 1e-12) / w_open * w_rest)])

    def sigma_rest(self, row) -> float:
        """Forecast standard deviation of the log return from t0 to the close."""
        return float(np.sqrt(np.exp(self.x(row) @ np.asarray(self.beta))))

    def band(self, s0: float, sigma: float, p: float) -> tuple[float, float]:
        """Rest-of-day low / high that each hold with probability p (one-sided, in-sample)."""
        k = f"{p:g}"
        return s0 * np.exp(-self.qd[k] * sigma), s0 * np.exp(self.qu[k] * sigma)

    def sigma_between(self, sigma_rest: float, a_min: float, b_min: float = 390) -> float:
        """Forecast sigma for a sub-window of the rest of the session."""
        w = self.share(self.t0)
        return sigma_rest * np.sqrt(self.share(a_min, b_min) / w) if w > 0 else 0.0

    # ---- fit / io
    @classmethod
    def fit(cls, table: pd.DataFrame, b5: pd.DataFrame, t0: int = 30) -> "RangeModel":
        m = cls(t0=t0, profile=list(var_profile(b5)))
        d = table[table["full"]].dropna(subset=["rv1", "rv5", "vix_prev"])
        d = d[(d["rv_rest"] > 0) & (d["rv_open"] > 0)]
        X = np.vstack([m.x(r) for _, r in d.iterrows()])
        y = np.log(d["rv_rest"].values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        m.beta = [float(b) for b in beta]
        m.r2 = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        sig = np.sqrt(np.exp(X @ beta))
        u = np.log(d["rest_high"].values / d["s0"].values) / sig
        dn = np.log(d["s0"].values / d["rest_low"].values) / sig
        for p in PROB_LEVELS:
            k = f"{p:g}"
            m.qu[k], m.qd[k] = float(np.quantile(u, p)), float(np.quantile(dn, p))
            m.joint[k] = float(((u <= m.qu[k]) & (dn <= m.qd[k])).mean())
        m.fit_days = int(len(d))
        return m

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RangeModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def forecast_table(model: RangeModel, table: pd.DataFrame) -> pd.DataFrame:
    """Add the forecast sigma and the bands to a session table (rows lacking inputs get NaN)."""
    t = table.copy()
    ok = t[["rv1", "rv5", "vix_prev"]].notna().all(axis=1) & (t["rv_open"] > 0)
    t["sigma"] = np.nan
    t.loc[ok, "sigma"] = [model.sigma_rest(r) for _, r in t[ok].iterrows()]
    for p in PROB_LEVELS:
        k = f"{p:g}"
        t[f"low_{k}"] = t["s0"] * np.exp(-model.qd[k] * t["sigma"])
        t[f"high_{k}"] = t["s0"] * np.exp(model.qu[k] * t["sigma"])
    return t


# ----------------------------------------------------------------------------- probabilities

def p_touch_below(s: float, level: float, var: float) -> float:
    """P(price trades at or below `level` before the horizon), driftless log-Brownian motion."""
    if level >= s:
        return 1.0
    if var <= 0:
        return 0.0
    return float(2 * norm.sf(np.log(s / level) / np.sqrt(var)))


def p_bracket(s: float, lo: float, hi: float, var: float, terms: int = 80) -> tuple[float, float, float]:
    """(P(hit hi first), P(hit lo first), P(neither before the horizon)) for a driftless
    log-Brownian motion with total variance `var` until the horizon. Eigen-series solution of
    the heat equation on [log lo, log hi]."""
    if s >= hi:
        return 1.0, 0.0, 0.0
    if s <= lo:
        return 0.0, 1.0, 0.0
    a, b, x = np.log(lo), np.log(hi), np.log(s)
    w, y = b - a, x - a
    if var <= 0:
        return 0.0, 0.0, 1.0
    need = int(np.ceil(np.sqrt(2 * 40 / var) * w / np.pi))  # terms until exp(-...) < e^-40
    if need > 20000:  # tiny time left: the barriers can't both matter, use one-sided touches
        up, dn = p_touch_below(1 / s, 1 / hi, var), p_touch_below(s, lo, var)
        return up, dn, max(0.0, 1.0 - up - dn)
    n = np.arange(1, max(terms, need) + 1)
    e = np.exp(-(n * np.pi / w) ** 2 * var / 2)
    c = 2 * (-1.0) ** n / (n * np.pi)
    up = y / w + float((c * np.sin(n * np.pi * y / w) * e).sum())
    dn = (w - y) / w + float((c * np.sin(n * np.pi * (w - y) / w) * e).sum())
    up, dn = min(max(up, 0.0), 1.0), min(max(dn, 0.0), 1.0)
    return up, dn, max(0.0, 1.0 - up - dn)
