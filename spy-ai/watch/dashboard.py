"""复盘看板：把每天的实盘复盘（data/spy_watch.sqlite）和 v6 回测结果（model6/*.json）汇总成一个离线单文件网页
reports/看板.html。收盘复盘后自动重建；也可以双击 打开看板.bat 手动重建并打开。"""
import datetime as dt, json, os, sqlite3, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import report

TF_ORDER = ["1m", "3m", "5m", "15m", "1h"]


def _f(x, nd=2):
    try:
        x = float(x); return None if np.isnan(x) else round(x, nd)
    except (TypeError, ValueError):
        return None


def live_payload(db_path):
    if not os.path.exists(db_path): return {"days": [], "events": [], "intraday": {}}
    db = sqlite3.connect(db_path)
    ev = pd.read_sql("SELECT * FROM events ORDER BY bar_t", db)
    bars = pd.read_sql("SELECT * FROM bars ORDER BY t", db); bars["day"] = bars.t.str[:10]
    ox = pd.read_sql("SELECT * FROM options ORDER BY t", db); fl = pd.read_sql("SELECT * FROM flows ORDER BY t", db)
    ox["day"] = ox.t.str[:10]; fl["day"] = fl.t.str[:10]
    B = {k: g for k, g in bars.groupby(["tf", "day"])}
    events = []
    for e in ev.itertuples():
        day = e.bar_t[:10]
        res, mfe, mae = report.outcome(B.get((e.tf, day), pd.DataFrame(columns=["t", "h", "l"])), e) if e.kind == "early" else ("", None, None)
        ctx = json.loads(e.ctx or "{}"); o = ctx.get("options") or {}; z = o.get("0dte") or o.get("all") or {}
        f = ctx.get("flows") or {}
        big = (f.get("capital_in_super", 0) + f.get("capital_in_big", 0) - f.get("capital_out_super", 0) - f.get("capital_out_big", 0)) / 1e8 if f else None
        zg = z.get("zero_gamma")
        events.append({"day": day, "t": e.bar_t[11:16], "tf": e.tf, "side": e.side, "kind": e.kind, "p": _f(e.prob, 1), "price": _f(e.price),
                       "res": "ok" if res.startswith("✓") else "bad" if res.startswith("✗") else ("open" if e.kind == "early" else ""),
                       "mfe": _f(mfe, 1), "mae": _f(mae, 1), "pushed": int(e.pushed),
                       "zg": None if zg is None else ("above" if e.price >= zg else "below"), "gex": _f(z.get("net_gex_bn"), 3),
                       "iv": _f((o.get("iv_next") or {}).get("atm_iv") or z.get("atm_iv"), 1), "big": _f(big, 2),
                       "model": ctx.get("model", "v5"), "p5": _f(ctx.get("p5"), 1), "p6": _f(ctx.get("p6"), 1)})
    days = sorted(set(bars.day) | set(e["day"] for e in events))
    intraday = {}
    for d in days:
        b1 = B.get(("1m", d))
        if b1 is None or b1.empty: continue
        k = [[int(t[11:13]) * 60 + int(t[14:16]), _f(o_), _f(h), _f(l), _f(c)] for t, o_, h, l, c in zip(b1.t, b1.o, b1.h, b1.l, b1.c)]
        O = []
        for r in ox[ox.day == d].itertuples():
            x = json.loads(r.data or "{}"); z = x.get("0dte") or x.get("all") or {}
            O.append([int(r.t[11:13]) * 60 + int(r.t[14:16]), _f(z.get("zero_gamma")), _f(z.get("call_wall")), _f(z.get("put_wall")),
                      _f(z.get("net_gex_bn"), 3), _f((x.get("iv_next") or {}).get("atm_iv"), 1), _f(z.get("atm_iv"), 1)])
        Fw = []
        for r in fl[fl.day == d].itertuples():
            x = json.loads(r.data or "{}")
            net = lambda k: (x.get("capital_in_" + k, 0) - x.get("capital_out_" + k, 0)) / 1e8
            Fw.append([int(r.t[11:13]) * 60 + int(r.t[14:16]), _f(net("super") + net("big")), _f(net("mid") + net("small"))])
        intraday[d] = {"k": k, "o": O, "f": Fw}
    return {"days": days, "events": events, "intraday": intraday}


def bt_payload(model_dir):
    out = {}
    for name in ("train6", "bt6", "plan", "v7", "momentum", "factory_summary"):
        p = os.path.join(model_dir, name + ".json")
        if os.path.exists(p): out[name] = json.load(open(p, encoding="utf-8"))
    if "v7" in out: out["v7"].pop("configs", None)
    if "bt6" in out:
        out["bt6"].pop("rules", None)
        for g in out["bt6"].get("limit", {}).values(): g.pop("grid", None)
    p = os.path.join(model_dir, "v6_features.json")
    if os.path.exists(p):                                   # indicator families actually fed to the model
        fam = json.load(open(p, encoding="utf-8")).get("FAMILY", {})
        out["families"] = sorted(set(fam.values()), key=list(dict.fromkeys(fam.values())).index)
    return out


def swing_payload(model_dir):
    """swing rule results + the 2026-09-30..10-05 case on 15-minute bars + the live paper position"""
    out = {}
    p = os.path.join(model_dir, "swing_final.json")
    if os.path.exists(p):
        F = json.load(open(p, encoding="utf-8")); F.pop("rules", None); out["final"] = F
    p = os.path.join(HERE, "data", "swing_state.json")
    if os.path.exists(p): out["state"] = json.load(open(p, encoding="utf-8"))
    p = "C:/Users/94868/spybt/v6/data/spy_15m.parquet"
    if os.path.exists(p):
        b = pd.read_parquet(p); b = b[(b.t >= "2026-09-29") & (b.t <= "2026-10-05 23:59")]
        out["case_bars"] = [[t[5:16], _f(o_), _f(h), _f(l), _f(c)] for t, o_, h, l, c in zip(b.t, b.o, b.h, b.l, b.c)]
    return out


def build(db_path=None, out_path=None):
    db_path = db_path or os.path.join(HERE, "data", "spy_watch.sqlite")
    out_path = out_path or os.path.join(HERE, "reports", "看板.html")
    data = {"built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "live": live_payload(db_path), "bt": bt_payload(os.path.join(HERE, "model6")), "swing": swing_payload(os.path.join(HERE, "model6"))}
    tpl = open(os.path.join(HERE, "dashboard_tpl.html"), encoding="utf-8").read()
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", js))
    return out_path


if __name__ == "__main__":
    p = build(*sys.argv[1:3]); print("看板：", p)
