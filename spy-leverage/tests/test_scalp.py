"""Indicators, multi-timeframe features, the scalp backtest engine and the live engine."""
import json

import numpy as np
import pandas as pd
import pytest

from spylev import ta
from spylev.dip.perp import PerpCosts
from spylev.live.engine import LiveEngine
from spylev.scalp.engine import ScalpSpec, run
from spylev.scalp.signals import build, buy_signal, tf_conditions


def _minutes(days=40, seed=7, vol=0.0006):
    """Random-walk regular-hours 1m bars (09:30..15:59 ET), no drift."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2023-03-01", periods=days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30 + m) for d in sessions for m in range(390)]).tz_localize("America/New_York")
    n = len(idx)
    c = 400 * np.exp(np.cumsum(vol * rng.standard_normal(n)))
    o = np.r_[400, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(0.0003 * rng.standard_normal(n)))
    lo = np.minimum(o, c) * (1 - np.abs(0.0003 * rng.standard_normal(n)))
    v = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)


@pytest.fixture(scope="module")
def bars():
    return _minutes()


@pytest.fixture(scope="module")
def feats(bars):
    return build(bars, ta.daily_levels(bars))


def test_indicators_do_not_look_ahead(bars):
    k = 3000
    full, cut = bars, bars.iloc[:k]
    for f in (lambda d: ta.rsi(d["close"]), lambda d: ta.macd(d["close"])[2], lambda d: ta.bollinger(d["close"])[2],
              lambda d: ta.stoch(d)[0], lambda d: ta.cci(d), lambda d: ta.mfi(d), lambda d: ta.session_vwap(d)[0],
              lambda d: ta.swing_lows(d["low"]), lambda d: ta.kdj(d)[2], lambda d: ta.supertrend(d)[0]):
        a, b = f(full).iloc[:k], f(cut)
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_daily_levels_use_prior_day_only(bars):
    lv = ta.daily_levels(bars)
    day = bars.index.normalize()
    d1 = day.unique()[5]
    prev = bars[day == day.unique()[4]]
    assert lv.loc[d1, "pdh"] == pytest.approx(prev["high"].max())
    assert lv.loc[d1, "pdl"] == pytest.approx(prev["low"].min())
    assert lv.loc[d1, "pdc"] == pytest.approx(prev["close"].iloc[-1])


@pytest.mark.parametrize("k", [3 * 390 + 2, 3 * 390 + 4, 3 * 390 + 5, 5 * 390 + 101, 7 * 390 + 389])
def test_mtf_features_do_not_look_ahead(bars, feats, k):
    """Row k of the full feature frame must equal the last row built from bars[:k+1] only:
    the 3m / 5m values on a minute are those of the latest CLOSED 3m / 5m bar."""
    cut = bars.iloc[: k + 1]
    part = build(cut, ta.daily_levels(cut))
    cols = ["m1_bottom", "m3_bottom", "m5_bottom", "m3_prev_bottom", "m5_prev_bottom", "m5_b_support", "m3_b_osc", "confirm", "vwap"]
    a = feats[cols].iloc[k].astype(float).values
    b = part[cols].iloc[-1].astype(float).values
    assert np.allclose(a, b, equal_nan=True)


def test_buy_signal_is_the_three_timeframes_together(feats):
    c = tf_conditions(feats, thr5=3)
    s = buy_signal(feats, thr5=3)
    window = (feats["mos"] >= 5) & (feats["mos"] <= 360)
    assert (s == (c["m5"] & c["m3"] & c["m1"] & window)).all()


def test_scalp_engine_random_walk_has_no_edge():
    """On a martingale the gross result per trade is ~0 and net is gross minus the 12bp round trip."""
    m = _minutes(days=90, seed=11)
    x = build(m, ta.daily_levels(m))
    t = run(x, ScalpSpec(3, "none", "top3m"))
    assert len(t) > 100
    se = t["gross"].std() / np.sqrt(len(t))
    assert abs(t["gross"].mean()) < 3.5 * se + 1e-5
    assert (t["net"] - (t["gross"] - PerpCosts().round_trip() - t["funding"])).abs().max() < 1e-12
    assert t.index.is_monotonic_increasing and (t["exit_time"] >= t.index).all()


def _flat_frame(n=120):
    """A hand-built feature frame with one buy signal at minute 30."""
    idx = pd.date_range("2024-03-12 09:30", periods=n, freq="1min", tz="America/New_York")
    x = pd.DataFrame({"open": 100.0, "high": 100.02, "low": 99.98, "close": 100.0, "volume": 1e5}, index=idx)
    x["mos"] = np.arange(n)
    for col in ("m5_bottom", "m5_prev_bottom", "m3_bottom", "m3_prev_bottom", "m1_bottom", "m1_bottom_recent", "m3_top", "m5_top"):
        x[col] = 0
    for col in ("m5_b_osc", "m5_b_support", "m5_prev_b_osc", "m5_prev_b_support", "m3_b_osc", "m3_prev_b_osc", "confirm",
                "m3_t_osc", "m5_t_osc", "daily_up"):
        x[col] = False
    i = 30
    x.iloc[i, x.columns.get_loc("m5_bottom")] = 4
    x.iloc[i, x.columns.get_loc("m3_bottom")] = 3
    x.iloc[i, x.columns.get_loc("m1_bottom_recent")] = 2
    for col in ("m5_b_osc", "m5_b_support", "m3_b_osc", "confirm"):
        x.iloc[i, x.columns.get_loc(col)] = True
    return x


def test_entry_next_open_and_stop_fill():
    x = _flat_frame()
    x.iloc[40, x.columns.get_loc("low")] = 99.5  # trades through the stop
    t = run(x, ScalpSpec(4, "none", "top5m"))
    assert len(t) == 1
    r = t.iloc[0]
    assert t.index[0] == x.index[31] and r["entry"] == 100.0
    stop = min(99.98 * (1 - 0.0002), 100 * (1 - 0.0008))
    assert r["stop"] == pytest.approx(stop) and r["reason"] == "stop"
    assert r["exit"] == pytest.approx(stop)
    assert r["net"] == pytest.approx(stop / 100 - 1 - PerpCosts().round_trip())


def test_gap_through_stop_fills_at_open():
    x = _flat_frame()
    x.iloc[40, x.columns.get_loc("open")] = 99.7
    x.iloc[40, x.columns.get_loc("low")] = 99.6
    r = run(x, ScalpSpec(4, "none", "top5m")).iloc[0]
    assert r["reason"] == "stop" and r["exit"] == pytest.approx(99.7)


def test_top_exit_at_next_open():
    x = _flat_frame()
    x.iloc[50, x.columns.get_loc("m5_top")] = 3
    x.iloc[50, x.columns.get_loc("m5_t_osc")] = True
    x.iloc[51, x.columns.get_loc("open")] = 100.3
    r = run(x, ScalpSpec(4, "none", "top5m")).iloc[0]
    assert r["reason"] == "top" and r["exit"] == pytest.approx(100.3) and r["exit_time"] == x.index[51]


def _daily(active: bool):
    """300 rising days, then three down closes (RSI2 < 10) if active."""
    idx = pd.bdate_range("2022-01-03", periods=303)
    c = np.linspace(300, 420, 303)
    if active:
        c[-3:] = [415, 410, 404]
    else:
        c[-3:] = [421, 422, 423]
    return pd.DataFrame({"open": c, "high": c + 2, "low": c - 2, "close": c}, index=idx)


def test_live_engine_validated_buy_before_the_close(bars, feats):
    daily = _daily(True)
    today = daily.index[-1] + pd.offsets.BDay(1)
    eng = LiveEngine(daily, user_leverage=10)
    setup = eng.daily_setup(today)
    assert setup["active"] and setup["rsi2"] < 10
    i = int(np.flatnonzero(feats["mos"].values == 375)[-1])  # 15:45 on some day, no signal needed
    s = eng.evaluate(feats, i, setup)
    assert s["status"] == "buy" and s["action"]["leverage"] == 2.0 and s["action"]["validated"]
    assert s["action"]["stop"] == pytest.approx(s["price"] - 2.5 * setup["atr"], abs=0.01)
    assert s["action"]["liquidation"] < s["action"]["stop"]
    eng.confirm_buy(s["price"], 2.0, s["action"]["stop"], s["time"])
    s2 = eng.evaluate(feats, i + 1, setup)
    assert s2["status"] == "hold"


def test_live_engine_waits_without_the_daily_setup(feats):
    eng = LiveEngine(_daily(False))
    setup = eng.daily_setup(pd.Timestamp("2023-03-30"))
    assert not setup["active"]
    sig = buy_signal(feats, thr5=4).values
    statuses = {eng.evaluate(feats, i, setup)["status"] for i in range(len(feats) - 600, len(feats))}
    assert statuses <= {"wait", "watch"}
    assert "buy" not in statuses
    if sig.any():
        i = int(np.flatnonzero(sig)[0])
        s = LiveEngine(_daily(False)).evaluate(feats, i, setup)
        assert s["status"] == "watch" and not s["action"]["validated"]


def test_demo_page_has_the_data_slot(tmp_path):
    from spylev.live import build_demo
    page = (build_demo.WEB / "index.html").read_text(encoding="utf-8")
    assert "/*__DEMO__*/null" in page and "<title>" in page[:300]
    out = build_demo.write({"track": {}, "days": [], "x": "</script>"}, tmp_path / "d.html")
    html = out.read_text(encoding="utf-8")
    blob = html.split("window.SPY_DEMO = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["x"] == "</script>"
