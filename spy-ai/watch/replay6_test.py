"""v6 end-to-end parity: replay a HOLDOUT day minute by minute through the real main loop (spy_watch.run) with the v6 engine,
fed by moomoo's own historical K-lines (v6/data), and check every early 底/顶 against the offline walk-forward predictions
(v6/cache/pred_*.parquet, same final model). Usage: py -3 replay6_test.py 2026-08-03"""
import datetime as dt, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spy_watch as W, v6core
from store import Store

V6DIR = "C:/Users/94868/spybt/v6"
DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-08-03"


class ReplaySource6:
    def __init__(self, day):
        self.day = pd.Timestamp(day); self.now = dt.datetime.combine(self.day.date(), dt.time(9, 24), W.ET); self.b = {}
        for tf in W.TF_MIN:
            d = pd.read_parquet(os.path.join(V6DIR, "data", "spy_%s.parquet" % tf))
            d = d[d.t.str[:10] <= day].tail(1500).reset_index(drop=True)
            end = pd.to_datetime(d.t); start = end - pd.to_timedelta(W.TF_MIN[tf], unit="m")
            prev = end.shift(1); same = prev.dt.date == end.dt.date
            start = start.where(~same | (prev <= start), prev)       # 1h 16:00 bar starts at 15:30
            d["start"] = start; self.b[tf] = d
        self.rng = np.random.default_rng(0)

    def now_et(self): return self.now
    def sleep(self, s): self.now += dt.timedelta(seconds=max(s, 60))

    def klines(self, tf, n=1000):
        d = self.b[tf]; nw = pd.Timestamp(self.now).tz_localize(None)
        vis = d[d.start < nw].tail(n)                          # closed bars + the one forming now
        return vis[["t", "o", "h", "l", "c", "v", "tv"]].reset_index(drop=True)

    def daily(self):
        D = pd.read_parquet(os.path.join(V6DIR, "data", "spy_1d.parquet")).set_index("t")
        return D[D.index < self.day.strftime("%Y-%m-%d")][["o", "h", "l", "c"]]

    def options(self, spot, now): return None
    def flows(self):
        keys = ["capital_in_super", "capital_in_big", "capital_in_mid", "capital_in_small", "capital_out_super", "capital_out_big", "capital_out_mid", "capital_out_small"]
        return dict(zip(keys, map(float, np.abs(self.rng.normal(0, 1, 8)) * 2e8 + 8e8)), update_time=self.now.strftime("%H:%M:%S"))


if __name__ == "__main__":
    cfg = W.load_cfg(); cfg["push"] = {"provider": "console"}; cfg["options"] = {"enabled": False}
    dbp = os.path.join(HERE, "data", "replay6_%s.sqlite" % DAY)
    if os.path.exists(dbp): os.remove(dbp)
    vix = pd.read_parquet(os.path.join(V6DIR, "data", "vix_daily.parquet")).sort_values("date")
    for k in ("vix", "vix3m", "vix9d"): vix[k + "_prev"] = vix[k].shift(1)
    v6 = v6core.V6(vix_prev=vix[["date", "vix_prev", "vix3m_prev", "vix9d_prev"]])
    store = Store(dbp); W.run(cfg, ReplaySource6(DAY), store, poll=60, verbose=False, v6=v6)
    ev = pd.read_sql("SELECT tf, side, kind, bar_t, prob FROM events WHERE kind='early'", store.db)
    th = float(cfg["threshold"]); tot = ok = 0
    for tf in W.TF_MIN:
        bars = pd.read_parquet(os.path.join(V6DIR, "data", "spy_%s.parquet" % tf))
        for side in ("bottom", "top"):
            P = pd.read_parquet(os.path.join(V6DIR, "cache", "pred_%s.parquet" % side))
            P = P[(P.tf == tf) & (P.date == pd.Timestamp(DAY)) & (P.p6 * 100 >= th)].sort_values(["cand_i", "c"]).drop_duplicates("cand_i")
            off = set(bars.t.to_numpy()[P.bar_j.to_numpy()])
            got = set(ev[(ev.tf == tf) & (ev.side == side)].bar_t)
            both = len(off & got); tot += len(off | got); ok += both
            print("%-3s %-6s offline %3d  loop %3d  match %3d%s" % (tf, side, len(off), len(got), both, "" if off == got else "  diff: " + str(sorted(off ^ got)[:4])))
    print("OVERALL v6 match %.1f%%" % (100 * ok / max(tot, 1)))
