"""spyopt — SPY multi-timeframe data, options pricing and strategy research toolkit.

Layout
------
spyopt.data        crawl / store / resample SPY bars (1m -> 3m, 5m, 10m, ... 4h, 1d), VIX family
spyopt.options     Black-Scholes, smile model calibrated on real SPY quotes, costs, structures
spyopt.indicators  realized-vol estimators, VRP, regime / event / trend signals, intraday TA
spyopt.backtest    daily-resolution and minute-resolution option engines, metrics, validation
spyopt.strategies  pre-registered strategy factory and the composite strategy
spyopt.risk        position sizing and drawdown brakes
spyopt.live        today's trade plan from the latest data
spyopt.report      HTML dashboard (multi-timeframe K-line viewer + research pages)
"""

__version__ = "0.1.0"

TZ = "America/New_York"
