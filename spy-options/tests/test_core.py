import numpy as np
import pandas as pd
import pytest

from spyopt.backtest.daily import Spec, market_rows, simulate
from spyopt.backtest.metrics import deflated_sharpe, max_drawdown, trade_stats
from spyopt.data.resample import integrity_report, resample
from spyopt.indicators.regime import features
from spyopt.options import bs
from spyopt.options.costs import CostModel
from spyopt.options.smile import Smile, fit
from spyopt.options.structures import build, payoff, worst_payoff


# --------------------------------------------------------------------------- Black-Scholes

def test_put_call_parity_and_iv_roundtrip():
    F, K, T, s = 770.0, np.array([740.0, 770.0, 800.0]), 5 / 252, 0.12
    c = bs.price(F, K, T, s, True)
    p = bs.price(F, K, T, s, False)
    assert np.allclose(c - p, F - K, atol=1e-9)
    assert np.allclose(bs.implied_vol(c, F, K, T, True), s, atol=1e-6)
    assert np.allclose(bs.implied_vol(p, F, K, T, False), s, atol=1e-6)


def test_expiry_is_intrinsic():
    assert bs.price(770.0, 760.0, 0.0, 0.2, True) == pytest.approx(10.0)
    assert bs.price(770.0, 760.0, 0.0, 0.2, False) == pytest.approx(0.0)


# --------------------------------------------------------------------------- resampling

def _minute_session(day="2024-03-04", n=390, start=500.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="1min", tz="America/New_York")
    c = start * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    o = np.r_[start, c[:-1]]
    h = np.maximum(o, c) * 1.0002
    lo = np.minimum(o, c) * 0.9998
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1000.0}, index=idx)


def test_resample_anchors_and_values():
    m1 = pd.concat([_minute_session("2024-03-04"), _minute_session("2024-11-29", n=210, seed=1)])
    h4 = resample(m1, "4h")
    d0 = h4.loc["2024-03-04"]
    assert [t.strftime("%H:%M") for t in d0.index] == ["09:30", "13:30"]
    assert list(d0["n_bars"]) == [240, 150]
    first = m1.loc["2024-03-04 09:30":"2024-03-04 13:29"]
    assert d0.iloc[0]["open"] == first["open"].iloc[0]
    assert d0.iloc[0]["high"] == first["high"].max()
    assert d0.iloc[0]["close"] == first["close"].iloc[-1]
    assert d0.iloc[0]["volume"] == first["volume"].sum()
    half = h4.loc["2024-11-29"]
    assert len(half) == 1 and half.iloc[0]["n_bars"] == 210  # 13:00 close -> single 4h bar
    m3 = resample(m1, "3m").loc["2024-03-04"]
    assert len(m3) == 130 and m3.index[1].strftime("%H:%M") == "09:33"
    d1 = resample(m1, "1d")
    assert len(d1) == 2


def test_integrity_report_flags_gaps():
    m1 = _minute_session().drop(index=_minute_session().index[100:120])
    rep = integrity_report(m1)
    assert rep.iloc[0]["missing"] == 20 and not rep.iloc[0]["ok"]


# --------------------------------------------------------------------------- smile

def test_smile_fits_real_quotes():
    sm, info = fit()
    assert info["rmse_ratio"] < 0.02  # < 2% of ATM vol across 29 real SPY quotes, 3 tenors
    for e in info["data"].values():
        idx = {1: "VIX1D", 5: "VIX9D", 20: "VIX"}[e["days"]]
        atm = sm.atm_from_index(info["index"][idx], e["days"])
        assert atm == pytest.approx(e["atm"], rel=1e-3)
    # puts richer than calls at equal distance (skew)
    assert sm.ratio(-2.0, 1) > 1.2 > sm.ratio(2.0, 1)


# --------------------------------------------------------------------------- structures

def test_structure_worst_case():
    F = np.array([770.0])
    legs = build("iron_condor", F, np.array([0.12]), 1 / 252, 1.0, 1.0, 0.5)
    widths = [legs[1].strike - legs[0].strike, legs[3].strike - legs[2].strike]
    assert worst_payoff(legs)[0] == pytest.approx(-max(w[0] for w in widths))
    assert payoff(legs, F)[0] == 0.0
    ls = build("long_straddle", F, np.array([0.12]), 1 / 252)
    assert worst_payoff(ls)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- null test

def _gbm_panel(n=2500, sigma=0.15, seed=7):
    """Synthetic world: implied vol == realized vol, no skew, no overnight moves."""
    rng = np.random.default_rng(seed)
    steps = 78
    dt = 1 / 252 / steps
    days = pd.bdate_range("2015-01-05", periods=n)
    rows = []
    s = 400.0
    for d in days:
        path = s * np.exp(np.cumsum(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(steps)))
        rows.append((s, max(s, path.max()), min(s, path.min()), path[-1]))
        s = path[-1]
    p = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=days)
    p["volume"] = 1.0
    flat = Smile(shapes={1: (0, 0, 0, 0), 5: (0, 0, 0, 0), 20: (0, 0, 0, 0)})
    idx = sigma * 100 / flat.level[1]
    p["vix1d_open"] = idx
    p["vix1d_close"] = idx
    p["vix9d_close"] = sigma * 100 / flat.level[5]
    p["vix_close"] = sigma * 100 / flat.level[20]
    p["vix3m_close"] = p["vix_close"]
    return p, flat


@pytest.mark.parametrize("structure,k", [("iron_condor", 1.0), ("put_spread", 1.0), ("iron_fly", 0.0),
                                         ("long_straddle", 0.0), ("call_spread", 0.5)])
def test_null_world_has_no_edge(structure, k):
    """With fair pricing and zero costs, hold-to-expiry P&L must average ~0 (|t| < 3)."""
    p, flat = _gbm_panel()
    rows = market_rows(p, features(p), "0dte_open", flat)
    free = CostModel(commission=0.0, min_spread=0.0, spread_coef=0.0, max_spread=0.0)
    t = simulate(rows, Spec("0dte_open", structure, k, k, 0.5, "hold", "none"), flat, free)
    se = t["pnl"].std() / np.sqrt(len(t))
    assert abs(t["pnl"].mean()) < 3 * se, (t["pnl"].mean(), se)
    # and costs make it strictly worse
    t2 = simulate(rows, Spec("0dte_open", structure, k, k, 0.5, "hold", "none"), flat, CostModel())
    assert t2["pnl"].mean() < t["pnl"].mean()


# --------------------------------------------------------------------------- metrics

def test_metrics_basic():
    idx = pd.bdate_range("2020-01-01", periods=200)
    tr = pd.DataFrame({"ror": np.r_[[0.1] * 150, [-0.3] * 50], "pnl": np.r_[[1.0] * 150, [-3.0] * 50],
                       "credit": 1.0, "risk": 10.0}, index=idx)
    st = trade_stats(tr)
    assert st["win_rate"] == pytest.approx(0.75)
    assert st["profit_factor"] == pytest.approx(1.0)
    assert max_drawdown(pd.Series([1, 2, 1, 3])) == pytest.approx(-0.5)
    assert deflated_sharpe(0.2, 500, 0, 3, 1) > deflated_sharpe(0.2, 500, 0, 3, 1000)


# --------------------------------------------------------------------------- minute engine

def _gbm_minutes(n_days=400, sigma=0.15, seed=11):
    from spyopt.options.smile import Smile as _S
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=n_days)
    dt = 1 / 252 / 390
    frames, s = [], 400.0
    for d in days:
        r = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(390)
        c = s * np.exp(np.cumsum(r))
        o = np.r_[s, c[:-1]]
        idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="1min", tz="America/New_York")
        frames.append(pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                                    "close": c, "volume": 1.0}, index=idx))
        s = c[-1]
    bars = pd.concat(frames)
    flat = _S(shapes={1: (0, 0, 0, 0), 5: (0, 0, 0, 0), 20: (0, 0, 0, 0)})
    panel = pd.DataFrame({"vix1d_open": sigma * 100 / flat.level[1], "vix_close": 15.0}, index=days)
    return bars, panel, flat


@pytest.mark.parametrize("kw", [dict(), dict(stop_mult=2.0), dict(tp=0.5), dict(touch=True),
                                dict(entry="11:00", exit="15:30")])
def test_minute_engine_null_world(kw):
    """Fair marks + any stopping rule => zero expected P&L (optional stopping theorem)."""
    from spyopt.backtest.minute import IntradaySpec, simulate_intraday
    bars, panel, flat = _gbm_minutes()
    free = CostModel(commission=0.0, min_spread=0.0, spread_coef=0.0, max_spread=0.0)
    prof = np.full(390, 1 / 390)
    t = simulate_intraday(bars, panel, IntradaySpec("iron_condor", 1.0, 1.0, 0.5, **kw), flat, free, profile=prof)
    se = t["pnl"].std() / np.sqrt(len(t))
    assert abs(t["pnl"].mean()) < 3 * se, (t["pnl"].mean(), se, t["exit"].value_counts().to_dict())


def test_nyse_calendar():
    from spyopt.calendar import next_session, nyse_holidays
    h25 = nyse_holidays(2025)
    assert {"2025-04-18", "2025-06-19", "2025-07-04", "2025-11-27", "2025-12-25"} <= h25
    assert "2027-06-18" in nyse_holidays(2027) and "2027-03-26" in nyse_holidays(2027)
    assert str(next_session("2026-10-02", 20).date()) == "2026-10-30"  # the snapshot's 20-day expiry
