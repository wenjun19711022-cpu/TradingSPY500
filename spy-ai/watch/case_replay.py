"""Replay several days through the real watcher loop with the v6 engine (offline) and list every early 底/顶 signal.
Usage: py -3 case_replay.py 2026-10-01 2026-10-02 2026-10-05  -> data/case_<day>.sqlite + printed table"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spy_watch as W, v6core, swing as swing_mod
from store import Store
from replay6_test import ReplaySource6, V6DIR

if __name__ == "__main__":
    cfg = W.load_cfg(); cfg["push"] = {"provider": "console"}; cfg["options"] = {"enabled": False}
    vix = pd.read_parquet(os.path.join(V6DIR, "data", "vix_daily.parquet")).sort_values("date")
    for k in ("vix", "vix3m", "vix9d"): vix[k + "_prev"] = vix[k].shift(1)
    rows = []; sp = os.path.join(HERE, "data", "case_swing_state.json")
    if os.path.exists(sp): os.remove(sp)
    sw = swing_mod.Swing(cfg, send=lambda t, b: print("【提醒】", t, "|", b.replace(chr(10), " ")[:160]), state_path=sp)
    for day in sys.argv[1:]:
        dbp = os.path.join(HERE, "data", "case_%s.sqlite" % day)
        if os.path.exists(dbp): os.remove(dbp)
        store = Store(dbp)
        W.run(cfg, ReplaySource6(day), store, poll=60, verbose=False, v6=v6core.V6(vix_prev=vix[["date", "vix_prev", "vix3m_prev", "vix9d_prev"]]), swing=sw)
        ev = pd.read_sql("SELECT tf, side, kind, bar_t, prob, price, ext FROM events WHERE kind='early' ORDER BY bar_t", store.db)
        rows.append(ev)
    E = pd.concat(rows); E.to_csv(os.path.join(HERE, "data", "case_events.csv"), index=False)
    print(E.to_string(index=False))
