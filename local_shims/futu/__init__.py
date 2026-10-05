"""Local shim: TradingSPY500 imports `futu`; this machine has the API-identical `moomoo` package."""
from moomoo import *  # noqa: F401,F403
