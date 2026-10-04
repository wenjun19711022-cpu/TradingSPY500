"""Collect a SPY option-chain snapshot in the format spyopt.options.smile.fit() reads.

    python -m spyopt.live.snapshot --source futu        # run ~15:55 ET (or after the close)

Writes data/raw/snapshots/SPY_options_<date>.json with OTM quotes for the 1-, 5- and
20-trading-day expiries (strikes within +/-3 expected moves) and the VIX-family closes you
pass in (or that load_panel() has). Re-fitting on a few weeks of snapshots instead of one day
is the single biggest improvement you can make to the pricing model:

    from spyopt.options.smile import fit
    sm, info = fit("data/raw/snapshots/SPY_options_2026-10-09.json")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spyopt.data.vix import load_panel
from spyopt.calendar import next_session

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "snapshots"


def futu_snapshot(spot_hint=None, host="127.0.0.1", port=11111):
    from futu import OpenQuoteContext, RET_OK
    ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, s = ctx.get_market_snapshot(["US.SPY"])
        spot = float(s["last_price"].iloc[0]) if ret == RET_OK else float(spot_hint)
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        exp = {}
        for n in (1, 5, 20):
            e = next_session(today, n)
            ret, chain = ctx.get_option_chain("US.SPY", start=str(e.date()), end=str(e.date()))
            if ret != RET_OK or chain.empty:
                continue
            em = spot * 0.15 * np.sqrt(n / 252)
            chain = chain[(chain["strike_price"] > spot - 3 * em) & (chain["strike_price"] < spot + 3 * em)]
            ret, snap = ctx.get_market_snapshot(list(chain["code"]))
            if ret != RET_OK:
                continue
            m = chain.merge(snap[["code", "bid_price", "ask_price"]], on="code")
            exp[str(e.date())] = {
                "trading_days": n,
                "puts": {f"{k:g}": [b, a] for k, b, a in m[m["option_type"] == "PUT"][["strike_price", "bid_price", "ask_price"]].values},
                "calls": {f"{k:g}": [b, a] for k, b, a in m[m["option_type"] == "CALL"][["strike_price", "bid_price", "ask_price"]].values},
            }
        return spot, exp
    finally:
        ctx.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="futu", choices=["futu"])
    a = ap.parse_args(argv)
    p = load_panel()
    last = p.iloc[-1]
    spot, exp = futu_snapshot(spot_hint=last["close"])
    doc = {"source": f"{a.source} snapshot", "underlying": {"symbol": "SPY", "close_1600": spot},
           "index_close": {"VIX1D": float(last.get("vix1d_close", np.nan)), "VIX9D": float(last["vix9d_close"]),
                           "VIX": float(last["vix_close"]), "VIX3M": float(last["vix3m_close"])},
           "expiries": exp}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"SPY_options_{pd.Timestamp.now(tz='America/New_York').date()}.json"
    f.write_text(json.dumps(doc, indent=1))
    print("wrote", f, "— note: index_close uses the latest panel row; refresh VIX data first")


if __name__ == "__main__":
    main()
