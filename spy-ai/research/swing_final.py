"""Freeze the swing results used by the watcher's 波段模式 and the dashboard: results/swing_final.json.
  主规则 E5×X3: previous close RSI2<20 and above the 200-day SMA -> the first 15m/1h v6 bottom of the day -> buy the next 1m open;
               stop 2.5% under the entry; once up 1 daily ATR (or a close with RSI2>70) exit at the first 15m v6 top; 10-day time stop.
  对照 E1×X1: TradingSPY500's daily RSI2<10 rule (buy next open, sell the open after a close above the 5-day SMA), same 2.5% stop.
  50x 版本: the same entry/exit with a 0.95% stop (the widest stop 50x isolated survives)."""
import json, math, os
import numpy as np, pandas as pd
import swing_study as S, strat_lab as SL

HERE = os.path.dirname(os.path.abspath(__file__))


def tstat(x):
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


if __name__ == "__main__":
    M = SL.load_1m(); D = S.daily(); days = list(D.index); nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    tops15, tops1h = S.v6_signals("top", "15m"), S.v6_signals("top", "1h"); b15, b1h = S.v6_signals("bottom", "15m"), S.v6_signals("bottom", "1h")
    dip = {nxt[d] for d, r in D.iterrows() if d in nxt and r.rsi2 < 20 and r.trend}
    allb = np.sort(np.r_[b15, b1h]); E5 = []
    for day in sorted(dip):
        x = allb[np.array([str(t)[:10] == day for t in allb])]
        if len(x): E5.append(x[0])
    E5 = np.array(E5, dtype="datetime64[ns]")
    E1 = np.array([np.datetime64(nxt[d] + "T09:30") for d, r in D.iterrows() if d in nxt and r.rsi2 < 10 and r.trend and d >= "2019-06-01"], dtype="datetime64[ns]")
    out = {"rules": __doc__, "variants": {}}
    for key, en, xr, stop, fund in (("主规则 E5×X3 止损2.5%", E5, "X3", 0.025, 0.0001), ("对照 E1×X1 原规则 止损2.5%", E1, "X1", 0.025, 0.0001), ("50倍版 E5×X3 止损0.95%", E5, "X3", 0.0095, 0.0001), ("主规则·按OKX实际资金费率", E5, "X3", 0.025, 0.000009)):
        S.FUND_8H = fund
        S.STOP = stop
        T = S.simulate(M, D, en, xr, tops15, tops1h); sp = S.split(T)
        v = {"stop": stop}
        for k, t in sp.items():
            m = S.money(t)
            if m.get("n"): m["t_net"] = round(tstat(t.net), 2)
            v[k] = m
        v["trades_oos"] = sp["oos"][["entry_t", "exit_t", "entry", "exit", "net", "mae", "why", "hours"]].round(5).to_dict("records")
        out["variants"][key] = v
        o = v["oos"]; print("%-26s OOS n%d win %.0f%% net %+.3f%% t %.2f | 50x liq %d %+.0f元/笔 | 20x %+.0f元/笔 | 10x %+.0f元/笔" % (
            key, o["n"], o["win"] * 100, o["net_pct"], o["t_net"], o["x50"]["liq"], o["x50"]["yuan_per_trade"], o["x20"]["yuan_per_trade"], o["x10"]["yuan_per_trade"]))
    # MAE distribution of the main rule (how far trades go against you before working) -> what leverage survives
    T = pd.DataFrame(out["variants"]["主规则 E5×X3 止损2.5%"]["trades_oos"])
    out["mae_hist"] = [[b, int(((-T.mae * 100) >= b).sum())] for b in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)]
    out["n_oos"] = int(len(T))
    json.dump(out, open(os.path.join(HERE, "results", "swing_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("MAE >= x%:", out["mae_hist"])
