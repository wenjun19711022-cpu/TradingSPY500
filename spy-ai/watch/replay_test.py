"""Offline end-to-end test: replay a historical day minute by minute through the REAL main loop (spy_watch.run),
with a fake OpenD (K-lines from the 9-year archive, synthetic option chain / capital distribution).
Checks that every early 底/顶 the loop catches equals the offline full-history computation."""
import datetime as dt, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spy_watch as W, v5core, gex
from store import Store

ARCH = "C:/Users/94868/spybt/muse/sealed_full/data/spy_%s.parquet"
DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-04-09"
OUTAGE = sys.argv[2] if len(sys.argv) > 2 else None        # e.g. "11:00-11:12": fake OpenD loses its server link (K-lines fail)


class ReplaySource:
    def __init__(self, day):
        self.day = pd.Timestamp(day); self.now = dt.datetime.combine(self.day.date(), dt.time(9, 24), W.ET)
        self.b = {}
        for tf in W.TF_MIN:
            d = pd.read_parquet(ARCH % tf)
            d = d[d.date <= self.day].tail(1500).reset_index(drop=True)
            start = d.ts.dt.tz_localize(None)
            end = start + pd.to_timedelta(W.TF_MIN[tf], unit="m")
            end = end.where(end.dt.hour * 60 + end.dt.minute <= 960, end.dt.normalize() + pd.Timedelta(hours=16))
            d["start"] = start; d["end"] = end
            self.b[tf] = d
        self.rng = np.random.default_rng(0)

    def now_et(self): return self.now
    def sleep(self, s): self.now += dt.timedelta(seconds=max(s, 60))       # advance one simulated minute per poll

    def klines(self, tf, n=1000):
        if OUTAGE and OUTAGE[:5] <= self.now.strftime("%H:%M") < OUTAGE[6:]: raise RuntimeError("K线失败：网络中断")
        d = self.b[tf]; nw = pd.Timestamp(self.now).tz_localize(None)
        vis = d[d.start <= nw].tail(n)                        # closed bars + the bar currently forming
        return pd.DataFrame({"t": vis.end.dt.strftime("%Y-%m-%d %H:%M:%S"), "o": vis.open, "h": vis.high, "l": vis.low, "c": vis.close, "v": vis.duka_vol})

    def options(self, spot, now):
        strikes = np.arange(np.floor(spot * 0.95), np.ceil(spot * 1.05) + 1, 1.0)
        rows = []
        for exp in (now.strftime("%Y-%m-%d"), (now + dt.timedelta(days=1)).strftime("%Y-%m-%d")):
            for k in strikes:
                base = 3000 * np.exp(-((k - spot) / (spot * 0.01)) ** 2) + (4000 if k % 5 == 0 else 0)
                rows.append((k, "CALL", exp, base * (1.3 if k > spot else 0.6), 16 + 6 * abs(k / spot - 1) * 100 / 5, np.nan, base / 3))
                rows.append((k, "PUT", exp, base * (1.4 if k < spot else 0.5), 18 + 8 * abs(k / spot - 1) * 100 / 5, np.nan, base / 2.5))
        df = pd.DataFrame(rows, columns=["strike", "type", "expiry", "oi", "iv", "gamma", "volume"])
        return gex.analyse(gex.normalise(df), spot, now)

    def flows(self):
        x = self.rng.normal(0, 1, 8) * 2e8 + 8e8
        keys = ["capital_in_super", "capital_in_big", "capital_in_mid", "capital_in_small", "capital_out_super", "capital_out_big", "capital_out_mid", "capital_out_small"]
        return dict(zip(keys, map(float, np.abs(x))), update_time=self.now.strftime("%H:%M:%S"))


if __name__ == "__main__":
    cfg = W.load_cfg(); cfg["push"] = {"provider": "console"}
    dbp = os.path.join(HERE, "data", "replay_%s.sqlite" % DAY)
    if os.path.exists(dbp): os.remove(dbp)
    store = Store(dbp); src = ReplaySource(DAY)
    W.run(cfg, src, store, poll=60, verbose=False)
    # ---- parity: loop-detected early marks vs offline computation on the full archive
    ev = pd.read_sql("SELECT tf, side, kind, bar_t FROM events", store.db)
    tot = ok = 0
    for tf in W.TF_MIN:
        d = pd.read_parquet(ARCH % tf); d = d[d.date <= pd.Timestamp(DAY)].reset_index(drop=True)
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        start = d.ts.dt.tz_localize(None); end = start + pd.to_timedelta(W.TF_MIN[tf], unit="m")
        end = end.where(end.dt.hour * 60 + end.dt.minute <= 960, end.dt.normalize() + pd.Timedelta(hours=16))
        lab = end.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(); today = (d.date == pd.Timestamp(DAY)).to_numpy()
        for side in ("bottom", "top"):
            r = v5core.side_signals(o, h, l, c, side, float(cfg["threshold"]))
            for kind, key in (("early", "fire"), ("confirm", "ok")):
                off = set(lab[today & r[key]])
                got = set(ev[(ev.tf == tf) & (ev.side == side) & (ev.kind == kind)].bar_t)
                first_bar = sorted(off | got)[:0]
                both = len(off & got); tot += len(off | got); ok += both
                print("%-3s %-6s %-7s offline %3d  loop %3d  match %3d%s" % (tf, side, kind, len(off), len(got), both,
                      "" if off == got else "  diff: " + str(sorted(off ^ got)[:3])))
    print("OVERALL match %.1f%%" % (100 * ok / max(tot, 1)))
    print("report:", os.path.join(HERE, "reports", "%s.html" % DAY))
