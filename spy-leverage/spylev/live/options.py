"""SPY option-chain summary from moomoo OpenD (reference only: there is no history to backtest it).

What it reports for the nearest expiry (0DTE on most days):
  expected_move   at-the-money straddle mid price: the move the option market is pricing to expiry
  call_wall       strike above spot with the most call open interest
  put_wall        strike below spot with the most put open interest
  gex_by_strike   dealer gamma exposure per 1% move, assuming dealers are short puts and long calls
                  (the usual simplification). Positive total -> dealers sell rallies and buy dips,
                  days tend to stay in a range; negative -> moves get extended.
  flip            strike where the cumulative exposure changes sign (rough "zero gamma" level)
  pc_volume       put / call volume today
"""
from __future__ import annotations

import pandas as pd


def summarize(chain: pd.DataFrame, spot: float) -> dict:
    """chain columns: strike, type ('CALL'/'PUT'), bid, ask, oi, gamma, volume, iv"""
    if chain is None or chain.empty:
        return {}
    c = chain.copy()
    c["mid"] = (c["bid"].fillna(0) + c["ask"].fillna(0)) / 2
    calls, puts = c[c["type"] == "CALL"].set_index("strike"), c[c["type"] == "PUT"].set_index("strike")
    strikes = sorted(set(calls.index) & set(puts.index))
    out = {"spot": spot}
    if strikes:
        atm = min(strikes, key=lambda k: abs(k - spot))
        out["atm"] = float(atm)
        out["expected_move"] = float(calls.loc[atm, "mid"] + puts.loc[atm, "mid"])
        out["expected_range"] = [spot - out["expected_move"], spot + out["expected_move"]]
    up, dn = calls[calls.index > spot], puts[puts.index < spot]
    if len(up) and up["oi"].sum() > 0:
        out["call_wall"] = float(up["oi"].idxmax())
    if len(dn) and dn["oi"].sum() > 0:
        out["put_wall"] = float(dn["oi"].idxmax())
    g = (calls["gamma"] * calls["oi"]).reindex(strikes).fillna(0) - (puts["gamma"] * puts["oi"]).reindex(strikes).fillna(0)
    gex = g * 100 * spot * spot * 0.01  # dollars of delta per 1% move
    if len(gex):
        out["gex_total"] = float(gex.sum())
        out["gex_by_strike"] = {f"{k:g}": float(v) for k, v in gex.items() if abs(k - spot) / spot < 0.02}
        cum = gex.cumsum()
        flips = [k for k, a, b in zip(strikes[1:], cum.values[:-1], cum.values[1:]) if a * b < 0]
        if flips:
            out["flip"] = float(min(flips, key=lambda k: abs(k - spot)))
    vc, vp = calls["volume"].sum(), puts["volume"].sum()
    if vc > 0:
        out["pc_volume"] = float(vp / vc)
    return out


def fetch_futu(ctx, code: str = "US.SPY", spot: float | None = None, width: float = 0.03) -> dict:
    """Nearest-expiry chain within +-3% of spot from OpenD, summarized."""
    from futu import RET_OK
    ret, exp = ctx.get_option_expiration_date(code)
    if ret != RET_OK or exp.empty:
        return {"error": str(exp)[:120]}
    exp = exp[exp["option_expiry_date_distance"] >= 0].sort_values("option_expiry_date_distance")
    day = str(exp["strike_time"].iloc[0])[:10]
    ret, ch = ctx.get_option_chain(code, start=day, end=day)
    if ret != RET_OK or ch.empty:
        return {"error": str(ch)[:120]}
    if spot is None:
        ret, q = ctx.get_market_snapshot([code])
        spot = float(q["last_price"].iloc[0]) if ret == RET_OK else float(ch["strike_price"].median())
    ch = ch[(ch["strike_price"] - spot).abs() / spot <= width]
    rows = []
    codes = list(ch["code"])
    for i in range(0, len(codes), 300):
        ret, snap = ctx.get_market_snapshot(codes[i:i + 300])
        if ret != RET_OK:
            return {"error": str(snap)[:120]}
        rows.append(snap)
    s = pd.concat(rows)
    chain = pd.DataFrame({"strike": s["option_strike_price"].astype(float), "type": s["option_type"].astype(str).str.upper(),
                          "bid": s["bid_price"].astype(float), "ask": s["ask_price"].astype(float),
                          "oi": s["option_open_interest"].astype(float), "gamma": s["option_gamma"].astype(float),
                          "volume": s["volume"].astype(float), "iv": s["option_implied_volatility"].astype(float)})
    out = summarize(chain, spot)
    out["expiry"] = day
    return out
