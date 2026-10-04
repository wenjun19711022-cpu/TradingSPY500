"""Range forecast, first-passage probabilities, bracket backtest and the live desk."""
import numpy as np
import pandas as pd
import pytest

from spylev.dip.perp import PerpCosts
from spylev.range.desk import RangeDesk, daily_rv
from spylev.range.engine import RangeSpec, plan, run, simulate, stats
from spylev.range.model import RangeModel, five_minute, forecast_table, p_bracket, p_touch_below, session_table


def _minutes(days=160, seed=5):
    """Random-walk regular-hours sessions whose daily volatility wanders (so it is forecastable)."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2023-01-02", periods=days)
    vol = 0.0005 * np.exp(np.cumsum(0.15 * rng.standard_normal(days)) * 0.5)
    rows, px = [], 400.0
    for d, v in zip(sessions, vol):
        r = v * rng.standard_normal(390)
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px, c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(0.2 * v * rng.standard_normal(390)))
        lo = np.minimum(o, c) * (1 - np.abs(0.2 * v * rng.standard_normal(390)))
        idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30 + m) for m in range(390)]).tz_localize("America/New_York")
        rows.append(pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1e5}, index=idx))
        px = c[-1]
    vix = pd.Series(vol * np.sqrt(390 * 252) * 100, index=sessions)
    return pd.concat(rows), vix


@pytest.fixture(scope="module")
def world():
    m1, vix = _minutes()
    b5 = five_minute(m1)
    tab = session_table(b5, vix, 30)
    cut = tab.index[100]
    model = RangeModel.fit(tab[tab.index < cut], b5[b5.index.tz_localize(None) < cut], 30)
    return m1, b5, vix, tab, model, forecast_table(model, tab)


def _mc_bracket(s, lo, hi, var, n=40000, steps=400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.full(n, np.log(s))
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    sd = np.sqrt(var / steps)
    for _ in range(steps):
        live = ~(up | dn)
        x[live] += sd * rng.standard_normal(live.sum())
        up |= live & (x >= np.log(hi))
        dn |= live & (x <= np.log(lo))
    return up.mean(), dn.mean()


def test_bracket_probabilities_match_simulation():
    s, lo, hi, var = 100.0, 99.6, 100.3, (0.004) ** 2
    up, dn, none = p_bracket(s, lo, hi, var)
    mu, md = _mc_bracket(s, lo, hi, var)
    assert up == pytest.approx(mu, abs=0.02) and dn == pytest.approx(md, abs=0.02)
    assert up + dn + none == pytest.approx(1.0)
    # no time limit -> the classic gambler's-ruin ratio; no time -> nothing happens
    assert p_bracket(s, lo, hi, 10.0)[0] == pytest.approx(np.log(s / lo) / np.log(hi / lo), abs=1e-6)
    assert p_bracket(s, lo, hi, 1e-12)[2] == pytest.approx(1.0, abs=1e-6)
    assert p_bracket(s, lo, hi, 2e-7)[2] == pytest.approx(1 - p_touch_below(s, lo, 2e-7) - p_touch_below(1 / s, 1 / hi, 2e-7), abs=0.01)


def test_touch_probability_reflection_principle():
    rng = np.random.default_rng(1)
    var, lvl = (0.003) ** 2, 99.8
    sd = np.sqrt(var / 400)
    paths = np.log(100) + np.cumsum(sd * rng.standard_normal((20000, 400)), axis=1)
    mc = (paths.min(axis=1) <= np.log(lvl)).mean()
    # discrete monitoring misses touches between steps: Broadie-Glasserman barrier shift
    assert p_touch_below(100.0, lvl * np.exp(-0.5826 * sd), var) == pytest.approx(mc, abs=0.015)
    assert p_touch_below(100.0, lvl, var) > mc
    assert p_touch_below(100.0, 100.5, var) == 1.0


def test_variance_clock_shares(world):
    model = world[4]
    assert model.share(0, 390) == pytest.approx(1.0)
    assert model.share(0, 30) + model.share(30) == pytest.approx(1.0)
    assert model.sigma_between(0.01, model.t0) == pytest.approx(0.01)


def test_bands_are_calibrated_in_sample(world):
    *_, tab, model, fc = world
    d = fc[(fc.index < tab.index[100]) & fc["sigma"].notna()]
    for p in ("0.8", "0.9"):
        assert (d["rest_high"] <= d[f"high_{p}"]).mean() == pytest.approx(float(p), abs=0.02)
        assert (d["rest_low"] >= d[f"low_{p}"]).mean() == pytest.approx(float(p), abs=0.02)
    assert model.r2 > 0.2  # the wandering volatility is forecastable


def test_random_walk_bracket_has_no_edge(world):
    """On a martingale the plan wins as often as the fair probability says and grosses ~0."""
    m1, *_, model, fc = world
    t = run(m1, fc, model, RangeSpec(30, "sigma", 0.5, 0.8, "0.5s"), costs=PerpCosts())
    f = t[t["filled"]]
    assert len(f) > 60
    se = f["gross"].std() / np.sqrt(len(f))
    assert abs(f["gross"].mean()) < 3.5 * se
    s = stats(t)
    assert abs(s["edge_vs_fair"]) < 0.12
    assert s["mean_net_bp"] < s["mean_gross_bp"]


def _day(prices_low, n=60):
    idx = pd.date_range("2024-03-12 09:30", periods=n, freq="1min", tz="America/New_York")
    d = pd.DataFrame({"open": 100.0, "high": 100.05, "low": 99.95, "close": 100.0, "volume": 1.0}, index=idx)
    for i, (col, v) in prices_low.items():
        d.iloc[i, d.columns.get_loc(col)] = v
    return d


def test_fill_needs_a_trade_through_and_stop_wins_ties(world):
    model = world[4]
    spec = RangeSpec(t0=30, last_fill=330, flat=385)
    tk = {"entry": 99.90, "stop": 99.70, "tp": 100.20}
    touch = _day({35: ("low", 99.90)})
    assert not simulate(touch, tk, model, 0.005, spec, PerpCosts())["filled"]  # a touch is not a fill
    both = _day({35: ("low", 99.88), 40: ("high", 100.25)})
    both.iloc[40, both.columns.get_loc("low")] = 99.60
    r = simulate(both, tk, model, 0.005, spec, PerpCosts())
    assert r["filled"] and r["fill"] == 99.90 and r["reason"] == "stop" and r["exit"] == 99.70
    win = _day({35: ("low", 99.88), 41: ("high", 100.25)})
    r = simulate(win, tk, model, 0.005, spec, PerpCosts())
    c = PerpCosts()
    assert r["reason"] == "tp" and r["net"] == pytest.approx(100.20 / 99.90 - 1 - 2 * c.maker - r["funding"])
    assert 0 < r["p_fair"] < 1


def test_plan_geometry(world):
    model = world[4]
    tk = plan(model, 500.0, 0.006, RangeSpec())
    assert tk["stop"] < tk["entry"] < 500.0 < tk["tp"] or tk["stop"] < tk["entry"] < tk["tp"]
    assert tk["stop"] < tk["band_low"]


def test_desk_forecasts_at_ten_and_tracks_the_ticket(world):
    m1, b5, vix, tab, model, fc = world
    day = tab.index[120]
    today = m1[m1.index.normalize().tz_localize(None) == day]
    hist = m1[m1.index.normalize().tz_localize(None) < day]
    desk = RangeDesk(model, leverage=20)
    rv = daily_rv(hist)
    vp = float(vix[vix.index < day].iloc[-1])
    assert desk.update(today.iloc[:20], rv, vp)["phase"] == "pre"
    s = desk.update(today.iloc[:30], rv, vp)
    assert s["phase"] == "live" and s["forecast"]["s0"] == pytest.approx(today["close"].iloc[29])
    assert s["forecast"]["sigma"] == pytest.approx(fc.loc[day, "sigma"], rel=1e-6)
    b = s["forecast"]["bands"]
    assert b["0.9"]["low"] < b["0.8"]["low"] < b["0.5"]["low"] < s["forecast"]["s0"] < b["0.5"]["high"] < b["0.8"]["high"]
    for i in range(31, 390):
        s = desk.update(today.iloc[:i], rv, vp)
    assert s["odds"]["state"] in ("done", "expired")
    scr = desk.screen()
    assert len(scr["cum"]) == 79 and scr["cum"][-1] == pytest.approx(1.0)


def test_option_chain_summary():
    from spylev.live.options import summarize
    rows = []
    for k in range(760, 781):
        rows.append({"strike": k, "type": "CALL", "bid": max(770 - k, 0) + 1.0, "ask": max(770 - k, 0) + 1.1,
                     "oi": 5000 if k == 775 else 1000, "gamma": 0.05, "volume": 100, "iv": 0.15})
        rows.append({"strike": k, "type": "PUT", "bid": max(k - 770, 0) + 0.9, "ask": max(k - 770, 0) + 1.0,
                     "oi": 6000 if k == 765 else 800, "gamma": 0.05, "volume": 150, "iv": 0.16})
    out = summarize(pd.DataFrame(rows), 770.2)
    assert out["atm"] == 770 and out["expected_move"] == pytest.approx(1.05 + 0.95)
    assert out["call_wall"] == 775 and out["put_wall"] == 765
    assert out["pc_volume"] == pytest.approx(1.5)
    assert "gex_total" in out
