"""First real OpenD connection check: K-lines (and time_key convention), option chain + snapshot (permission), capital distribution.
Read-only quotes; closes the connection at the end."""
import datetime as dt, sys, time
import spy_watch as W, gex, v5core

cfg = W.load_cfg()
t0 = time.time()
src = W.LiveSource(cfg)
print("连接+订阅 OK（%.1fs）" % (time.time() - t0))
now = src.now_et(); print("美东时间", now.strftime("%Y-%m-%d %H:%M:%S"))
try:
    for tf in W.TF_MIN:
        k = src.klines(tf, 1000)
        d = W.closed_rth(k, tf, now)
        print("%-3s K线 %4d 根（收盘的常规时段 %4d 根）首 %s 末 %s" % (tf, len(k), len(d), k.t.iloc[0], k.t.iloc[-1]))
        if tf == "1m":
            print("    最近 3 根：", k.tail(3)[["t", "o", "h", "l", "c", "v"]].to_dict("records"))
            o, h, l, c = (d[x].to_numpy(float) for x in ("o", "h", "l", "c"))
            for side in ("bottom", "top"):
                r = v5core.side_signals(o, h, l, c, side, 55.0)
                print("    %s 当前概率 %.0f%%（候选存活=%s），今日早报 %d 个" % (side, r["p"][-1], bool(r["live"][-1]),
                      int(sum(r["fire"][i] for i in range(len(d)) if d.t.iloc[i][:10] == now.strftime("%Y-%m-%d")))))
    spot = float(W.closed_rth(src.klines("1m", 5), "1m", now).c.iloc[-1]) if len(src.klines("1m", 5)) > 1 else None
    print("现价", spot)
    try:
        ox = src.options(spot, now)
        print("期权 OK：", {k: v for k, v in (ox or {}).items() if k != "spot"})
        print("   提示行：", gex.context_line(ox, spot, 0.8))
    except Exception as e:
        print("期权 失败：", e)
    try:
        print("资金分布 OK：", src.flows())
    except Exception as e:
        print("资金分布 失败：", e)
finally:
    src.close()
    print("已断开。")
