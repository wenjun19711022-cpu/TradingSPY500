"""spyq — SPY multi-timeframe data, options pricing and strategy research toolkit.

Layout
------
spyq.data        crawl / store / resample SPY bars (1m -> 3m, 5m, 10m, ... 4h, 1d), VIX family
spyq.options     Black-Scholes, smile model calibrated on real SPY quotes, costs, structures
spyq.indicators  realized-vol estimators, VRP, regime / event / trend signals, intraday TA
spyq.backtest    daily-resolution and minute-resolution option engines, metrics, validation
spyq.strategies  pre-registered strategy factory and the composite strategy
spyq.risk        position sizing and drawdown brakes
spyq.live        today's trade plan from the latest data
spyq.report      HTML dashboard (multi-timeframe K-line viewer + research pages)
"""

__version__ = "0.1.0"

TZ = "America/New_York"
