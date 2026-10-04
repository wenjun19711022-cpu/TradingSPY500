"""Daily regime features for the option strategies (all as of the close of each row).

vrp_*        implied / forecast realized vol (> 1 = options rich, the seller's edge)
ts_9d_30d    VIX9D / VIX   (< 1 = contango = calm; > 1 = near-term stress)
ts_30d_3m    VIX / VIX3M
event_next   VIX1D(close) / VIX9D(close): tomorrow's 1-day vol vs the 9-day average. Spikes
             mark scheduled events (FOMC, CPI, NFP) without needing an event calendar.
trend        +1 / -1 / 0 from close vs EMA20 and EMA20 slope
vix_pct      VIX percentile over the trailing year
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spyopt.indicators import vol


def features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel
    f = pd.DataFrame(index=p.index)
    o, h, l, c = p["open"], p["high"], p["low"], p["close"]
    r_cc = np.log(c / c.shift())
    r_co = np.log(o / c.shift())
    gk = vol.garman_klass(o, h, l, c).clip(lower=1e-10)
    f["rv_cc"] = r_cc**2
    f["rv_session"] = gk
    f["rv_overnight"] = r_co**2
    f["fc_session"] = vol.har_forecast(gk)                       # forecast for next session
    # close-to-close proxy: session range estimator + overnight return^2 (far less noisy than r_cc^2)
    f["fc_cc"] = vol.har_forecast(gk + (r_co**2).fillna(0.0))
    f["yz20"] = vol.yang_zhang(p, 20)
    f["rv_session_ann"] = vol.ann(f["fc_session"])
    f["rv_cc_ann"] = vol.ann(f["fc_cc"])
    if "vix9d_close" in p:
        f["vrp_9d"] = p["vix9d_close"] / vol.ann(f["fc_cc"])
        f["ts_9d_30d"] = p["vix9d_close"] / p["vix_close"]
    if "vix1d_close" in p:
        f["vrp_1d"] = p["vix1d_close"] / vol.ann(f["fc_cc"])
        f["event_next"] = p["vix1d_close"] / p["vix9d_close"]
    if "vix3m_close" in p:
        f["ts_30d_3m"] = p["vix_close"] / p["vix3m_close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    slope = ema20.diff(5)
    f["trend"] = np.where((c > ema20) & (slope > 0), 1, np.where((c < ema20) & (slope < 0), -1, 0))
    f["above_ema50"] = (c > ema50).astype(int)
    f["mom5"] = c / c.shift(5) - 1
    f["vix_pct"] = p["vix_close"].rolling(252, min_periods=60).rank(pct=True)
    return f
