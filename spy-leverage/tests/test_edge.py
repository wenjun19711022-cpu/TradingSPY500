"""Published intraday strategies, cost regimes and the live playbook."""
import numpy as np
import pandas as pd
import pytest

from spylev.edge.evaluate import _funding_settlements, net_returns
from spylev.edge.live import Playbook
from spylev.edge.strategies import N, noise, orb5, overnight, panel


def _sessions(n=260, seed=3, vol=0.0005):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2013-01-07", periods=n)
    days = [d for d in days if not (d.month == 1 and d.day == 1)]
    rows, px = [], 1500.0
    for d in days:
        px *= np.exp(0.003 * rng.standard_normal())  # overnight move
        r = vol * rng.standard_normal(N)
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px, c[:-1]]
        h = np.maximum(o, c) * (1 + 0.0001)
        lo = np.minimum(o, c) * (1 - 0.0001)
        idx = pd.date_range(d + pd.Timedelta(minutes=570), periods=N, freq="1min", tz="America/New_York")
        rows.append(pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx))
        px = c[-1]
    return pd.concat(rows)


@pytest.fixture(scope="module")
def walk():
    return _sessions()


def test_panel_shapes_and_gaps(walk):
    p = panel(walk)
    assert p.C.shape[1] == N and len(p.days) == p.C.shape[0]
    assert np.isnan(p.prev_close[0]) and np.allclose(p.prev_close[1:5], p.C[0:4, -1])
    gone = walk.index.normalize().unique()[5]  # 2013-01-14, a regular session
    q = panel(walk.drop(walk.index[walk.index.normalize() == gone]))
    i = int(np.searchsorted(q.days, gone.tz_localize(None)))
    assert np.isnan(q.prev_close[i]) and not q.consecutive[i]  # a missing session breaks the chain
    assert q.consecutive[i + 1]


@pytest.mark.parametrize("fn", [orb5, noise, overnight])
def test_random_walk_has_no_edge(walk, fn):
    t = fn(panel(walk))
    assert len(t) > 40
    g = t["gross"]
    assert abs(g.mean()) < 3.5 * g.std() / np.sqrt(len(g)) + 1e-6


def _one_day(path_close, prev_close=100.0, n_hist=15):
    """n_hist quiet sessions, then a day whose closes follow path_close (dict minute -> price)."""
    days = pd.bdate_range("2024-03-01", periods=n_hist + 1)
    out = []
    for i, d in enumerate(days):
        idx = pd.date_range(d + pd.Timedelta(minutes=570), periods=N, freq="1min", tz="America/New_York")
        if i < n_hist:
            c = np.full(N, prev_close) * (1 + 0.0005 * np.sin(np.arange(N) / 7))
        else:
            c = pd.Series(np.nan, index=range(N))
            for k, v in path_close.items():
                c[k] = v
            c = c.interpolate().ffill().bfill().values
        o = np.r_[prev_close if i == n_hist else c[0], c[:-1]]
        out.append(pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, index=idx))
    return pd.concat(out)


def test_orb_rules():
    bars = _one_day({0: 100.0, 4: 100.2, 30: 100.1, 389: 100.5})
    p = panel(bars)
    t = orb5(p)
    last = t.iloc[-1]
    assert last["t_in"] == 5 and last["entry"] == pytest.approx(p.O[-1, 5])
    assert last["how_out"] == "close" and last["exit"] == pytest.approx(p.C[-1, -1])
    down = _one_day({0: 100.0, 4: 99.8, 389: 100.5})
    assert orb5(panel(down)).empty or orb5(panel(down))["date"].iloc[-1] != panel(down).days[-1]


def test_noise_breakout_and_vwap_exit():
    # quiet history (sigma ~ 0.05%), then a clear break above the band by 10:00 and a fall by 10:30
    bars = _one_day({0: 100.0, 20: 100.1, 29: 100.6, 40: 100.7, 59: 99.9, 389: 99.9})
    p = panel(bars)
    t = noise(p)
    t = t[t["date"] == p.days[-1]]
    assert len(t) == 1
    r = t.iloc[0]
    assert r["t_in"] == 30 and r["entry"] == pytest.approx(p.O[-1, 30])
    assert r["t_out"] == 60 and r["how_out"] == "trail"


def test_live_playbook_matches_backtest():
    bars = _one_day({0: 100.0, 20: 100.1, 29: 100.6, 40: 100.7, 59: 99.9, 150: 100.2, 179: 100.9, 389: 101.2})
    p = panel(bars)
    bt = noise(p)
    bt = bt[bt["date"] == p.days[-1]]
    day = p.days[-1]
    hist = bars[bars.index.normalize().tz_localize(None) < day]
    today = bars[bars.index.normalize().tz_localize(None) == day]
    pb = Playbook(scale=1.0)
    assert pb.prepare(hist, day)
    s = pb.update(today)
    live = [(t["in"], round(t["ret"], 8)) for t in s["noise"]["trades"]]
    ref = [("%02d:%02d" % divmod(570 + int(r.t_in), 60), round(r.gross, 8)) for r in bt.itertuples()]
    assert live == ref and len(ref) == 2
    mid = pb.update(today.iloc[:40])
    assert mid["noise"]["state"] == "long" and mid["noise"]["next_check"] == "10:30"


def test_cost_regimes_and_funding():
    t = pd.DataFrame({"date": pd.to_datetime(["2024-03-12", "2024-03-12", "2024-01-08"]), "t_in": [5, 200, -1],
                      "t_out": [389, 389, 0], "gross": [0.001, 0.001, 0.001], "how_in": "market",
                      "how_out": ["close", "target", "market"], "mae": -0.001, "nights": [np.nan, np.nan, 3]})
    assert list(_funding_settlements(t)) == [1, 0, 8]  # 12:00 EDT is 16:00 UTC; a weekend has 8 settlements
    okx = net_returns(t, "okx_taker")
    assert okx[0] == pytest.approx(0.001 - 0.0012 - 0.0001)
    assert okx[1] == pytest.approx(0.001 - 0.0006 - 0.0002)
    mes = net_returns(t, "mes")
    assert mes[0] == pytest.approx(0.001 - 0.0001)
    assert mes[2] < 0.001 - 0.0001  # three nights of futures carry in 2024 (rates above the dividend yield)
