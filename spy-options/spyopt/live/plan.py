"""Tonight's plan: the next VRP-20 rung, expected-move bands and the regime read-out.

    python -m spyopt.live.plan                 # from the data in data/raw (IBKR dumps / CBOE CSVs)
    python -m spyopt.live.plan --futu          # also price the legs off the live moomoo chain
    python -m spyopt.live.plan --equity 150000 --json plan.json

Prices printed are model mids; with --futu the live bid/ask/mid of each leg is shown next to
them. If they differ by more than ~10%, refresh the calibration (spyopt.live.snapshot, then
spyopt.options.smile.fit) before trusting the backtest numbers.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict

from spyopt.data.vix import load_panel
from spyopt.options.smile import Smile
from spyopt.strategies.composite import CORE, delta_of_strikes, expected_move_bands, plan_rung, sizing_table


def live_quotes(rung, host="127.0.0.1", port=11111):
    """Bid/ask for the rung's four legs from moomoo OpenD (needs US options quote rights)."""
    from futu import OpenQuoteContext, RET_OK
    ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, chain = ctx.get_option_chain("US.SPY", start=rung.expiry_date, end=rung.expiry_date)
        if ret != RET_OK:
            raise RuntimeError(chain)
        want = {(float(k), "CALL" if "call" in name else "PUT"): name for name, k in rung.strikes.items()}
        codes = {}
        for _, row in chain.iterrows():
            key = (float(row["strike_price"]), row["option_type"])
            if key in want:
                codes[row["code"]] = want[key]
        ret, snap = ctx.get_market_snapshot(list(codes))
        if ret != RET_OK:
            raise RuntimeError(snap)
        out = {}
        for _, r in snap.iterrows():
            out[codes[r["code"]]] = {"bid": float(r["bid_price"]), "ask": float(r["ask_price"]),
                                     "mid": (float(r["bid_price"]) + float(r["ask_price"])) / 2,
                                     "iv": float(r.get("option_implied_volatility", float("nan")))}
        return out
    finally:
        ctx.close()


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=100_000)
    ap.add_argument("--futu", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    p = load_panel()
    sm = Smile()
    r = plan_rung(p, CORE, sm, equity=a.equity)
    out = {"rung": asdict(r), "deltas": delta_of_strikes(r, sm), "bands": expected_move_bands(p, sm),
           "sizing_for_equity": [row for row in sizing_table(r.max_loss * 100, r.sessions, (a.equity,))]}
    print(f"\n下一档  {r.entry_date} 收盘开仓 → {r.expiry_date} 到期（{r.sessions} 个交易日）")
    print(f"现价 {r.spot:.2f}  远期 {r.forward:.2f}  20日平值隐波 {r.atm_iv:.1%}  1倍预期波动 ±{r.expected_move:.2f}")
    for (name, k), (lab, mid) in zip(r.strikes.items(), r.model_mid.items()):
        print(f"  {name:<11} {k:>7.0f}  delta {out['deltas'][name]:+.3f}  模型中价 {mid:.3f}")
    print(f"净收 ${r.credit * 100:.0f}/张  最大亏损 ${r.max_loss * 100:.0f}/张  50%止盈买回价 <= {r.take_profit_debit:.2f}")
    for row in out["sizing_for_equity"]:
        print(f"  账户 ${row['equity']:,.0f} {row['cadence']:<6} 在场 {row['open_rungs']:>2} 档: 保守 {row['conservative']} 张 / 标准 {row['standard']} 张")
    if a.futu:
        q = live_quotes(r)
        out["live"] = q
        for name, v in q.items():
            print(f"  实时 {name:<11} bid {v['bid']:.2f} ask {v['ask']:.2f} mid {v['mid']:.3f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
