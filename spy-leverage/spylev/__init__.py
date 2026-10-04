"""spylev — SPY long-only leverage research and live alerts.

spylev.data      crawl / store / resample SPY bars, long daily history (1993-)
spylev.ta        TradingView-style indicators, support/resistance levels, candle patterns
spylev.dip       daily/weekly/monthly dip-buying study (OKX perp costs, liquidation, leverage)
spylev.scalp     1m/3m/5m confluence bottom -> top scalps: signals, backtest, alert engine
spylev.live      moomoo OpenD feed, replay feed, alert server and the one-screen UI
"""

__version__ = "0.2.0"

TZ = "America/New_York"
