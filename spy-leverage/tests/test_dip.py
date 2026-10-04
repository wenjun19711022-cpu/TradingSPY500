import numpy as np
import pandas as pd
import pytest

from spylev.dip.engine import DipSpec, leverage_table, levered_equity, run
from spylev.dip.perp import ZERO, PerpCosts
from spylev.dip.signals import features, signal


def test_liquidation_price():
    c = PerpCosts(mmr=0.005)
    assert c.liq_drop(10) == pytest.approx(1 - 0.9 / 0.995)
    assert c.liq_drop(1) == 1.0
    assert c.liq_drop(3) > c.liq_drop(5) > c.liq_drop(20)
    assert c.funding(365) == pytest.approx(0.1095)  # 0.01% x 3 x 365


def _rw(n=6000, seed=3, drift=0.0, vol=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    o = np.r_[100, c[:-1]] * np.exp(0.002 * rng.standard_normal(n))
    h = np.maximum(o, c) * np.exp(np.abs(0.004 * rng.standard_normal(n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(0.004 * rng.standard_normal(n)))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 0.0}, index=idx)


def test_entry_is_next_bar_open():
    df = _rw(800)
    f = features(df, "1d")
    sig = signal(f, "rsi2", "1d")
    t = run(df, DipSpec("1d", "rsi2", None), ZERO, feats=f)
    first_sig = sig[sig].index[0]
    nxt = df.index[df.index.get_loc(first_sig) + 1]
    assert t.index[0] == nxt and t["entry"].iloc[0] == df.loc[nxt, "open"]


@pytest.mark.parametrize("fam,stop", [("rsi2", None), ("rsi2", 2.5), ("boll", None), ("wick", 2.5)])
def test_random_walk_has_no_edge(fam, stop):
    """No drift, no costs: dip signals on a martingale must average ~0 per trade."""
    df = _rw(20000, seed=7)
    t = run(df, DipSpec("1d", fam, stop), ZERO)
    se = t["net"].std() / np.sqrt(len(t))
    assert len(t) > 100
    assert abs(t["net"].mean()) < 3 * se, (t["net"].mean(), se)


def test_levered_equity_liquidates():
    idx = pd.bdate_range("2020-01-01", periods=3)
    tr = pd.DataFrame({"net": [0.01, 0.01, 0.01], "mae": [-0.01, -0.12, -0.01],
                       "exit_time": idx, "stop_dist": np.nan, "days": 1.0}, index=idx)
    e = levered_equity(tr, 10.0, PerpCosts())
    assert e["liquidations"] == 1 and e["final"] == 0.0
    e3 = levered_equity(tr, 3.0, PerpCosts())
    assert e3["liquidations"] == 0 and e3["final"] > 1.0
    lt = leverage_table(pd.concat([tr] * 3), PerpCosts())
    assert lt["L_liq"] < 1 / 0.12  # 1.5x safety on the worst MAE
