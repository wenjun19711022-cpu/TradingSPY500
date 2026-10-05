"""底顶AI v6 live engine: all moomoo indicators + day/VIX + 15m/1h state, same code as training (model6/feat6.py).
Semantics identical to v5core: the latest candidate is re-scored on each of its stages c = 0..3 while unresolved;
an early mark fires the first time p >= threshold for that candidate."""
import json, os, pickle, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); M6 = os.path.join(HERE, "model6")
sys.path.insert(0, M6)
import feat6, models6                              # models6: classes needed to unpickle the models

TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}


def fetch_vix_prev(today):
    """previous-session VIX / VIX3M / VIX9D close (Yahoo chart API). Missing -> NaN (the model handles it)."""
    out = {"date": pd.Timestamp(today)}
    for k, s in (("vix", "%5EVIX"), ("vix3m", "%5EVIX3M"), ("vix9d", "%5EVIX9D")):
        v = np.nan
        for attempt in range(2):
            try:
                req = urllib.request.Request("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=1mo&interval=1d" % s,
                                             headers={"User-Agent": "Mozilla/5.0"})
                r = json.loads(urllib.request.urlopen(req, timeout=10).read())["chart"]["result"][0]
                d = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d")
                ser = pd.Series(r["indicators"]["quote"][0]["close"], index=d).dropna()
                ser = ser[ser.index < pd.Timestamp(today).strftime("%Y-%m-%d")]
                if len(ser): v = float(ser.iloc[-1])
                break
            except Exception:
                time.sleep(1)
        out[k + "_prev"] = v
    return pd.DataFrame([out])


class V6:
    def __init__(self, vix_prev=None):
        self.m = {s: pickle.load(open(os.path.join(M6, "v6_%s.pkl" % s), "rb")) for s in ("bottom", "top")}
        self.vix_prev = vix_prev                   # DataFrame date, vix_prev, vix3m_prev, vix9d_prev
        self.name = {s: self.m[s]["v6_name"] for s in self.m}

    def stats_text(self, tf, side):
        try:
            R = json.load(open(os.path.join(M6, "train6.json"), encoding="utf-8"))["sides"][side]
            x = R[R["selection"]["chosen"]][tf]["thr0.55"]
            return "v6 历史准确率 %d%%（标出率 %d%%，2024–2026 验证集）" % (round(x["precision"] * 100), round(x["recall"] * 100))
        except Exception:
            return ""

    def set_day(self, today):
        if self.vix_prev is None or pd.Timestamp(today) not in set(self.vix_prev.date):
            self.vix_prev = fetch_vix_prev(today)

    def signals(self, bars, tf, side, th, js):
        """bars: {tf: closed RTH bars t,o,h,l,c,v,tv}. js: bar indices (in bars[tf]) to evaluate.
        Returns {j: (p6 %, fire, nb)} for js where a live candidate exists."""
        b = feat6.decorate(bars[tf]); htf = {"h15": feat6.decorate(bars["15m"]), "h60": feat6.decorate(bars["1h"])}
        f5, cand, px = feat6.base_features5(b, side)
        l, h, atr = px["l"], px["h"], px["atr"]
        BF = None; out = {}; model = self.m[side]["v6"]
        cidx = np.flatnonzero(cand)
        for j in js:
            prior = cidx[cidx <= j]
            if not len(prior): continue
            i = int(prior[-1]); nb = j - i
            if nb > feat6.CMAX: continue
            tg = l[i] + feat6.K * atr[i]
            seg = slice(i + 1, j + 1)
            if nb > 0 and ((l[seg] <= l[i]).any() or (h[seg] >= tg).any()): continue      # resolved -> not live
            if BF is None: BF = feat6.bar_features(b, self.vix_prev, htf)
            cc = np.arange(nb + 1); ii = np.full(nb + 1, i); jj = i + cc
            R = feat6.stage_rows(f5, BF, px, ii, jj, cc); R["tf_min"] = np.log(TF_MIN[tf])
            p = model.predict_proba(R)[:, 1] * 100
            out[j] = (float(p[-1]), bool(p[-1] >= th and not (p[:-1] >= th).any()), int(nb))
        return out
