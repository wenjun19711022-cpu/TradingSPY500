"""Daily review (复盘): every signal with its real outcome, options levels through the day, capital flow. Offline HTML."""
import html, json, os
import numpy as np
import pandas as pd

TF_NAME = {"1m": "1分钟", "3m": "3分钟", "5m": "5分钟", "15m": "15分钟", "1h": "1小时"}
HERE = os.path.dirname(os.path.abspath(__file__))


def outcome(bars, ev):
    """True bottom/top by the v5 definition, judged on the bars after the signal: +2 ATR before breaking the extreme."""
    after = bars[bars.t > ev.bar_t]
    if after.empty: return "未完", np.nan, np.nan
    side = ev.side; ext, atr, px = ev.ext, ev.atr, ev.price
    res = "未定"
    for r in after.itertuples():
        if side == "bottom":
            if r.l <= ext: res = "✗ 跌破"; break
            if r.h >= ext + 2 * atr: res = "✓ 真底"; break
        else:
            if r.h >= ext: res = "✗ 突破"; break
            if r.l <= ext - 2 * atr: res = "✓ 真顶"; break
    nxt = after.head(30)
    if side == "bottom":
        mfe, mae = (nxt.h.max() / px - 1) * 1e4, (nxt.l.min() / px - 1) * 1e4
    else:
        mfe, mae = (1 - nxt.l.min() / px) * 1e4, (1 - nxt.h.max() / px) * 1e4
    return res, round(mfe, 1), round(mae, 1)


def build(store, day, cfg):
    db = store.db
    ev = pd.read_sql("SELECT * FROM events WHERE day=? ORDER BY bar_t", db, params=(day,))
    bars = {tf: pd.read_sql("SELECT * FROM bars WHERE tf=? AND substr(t,1,10)=? ORDER BY t", db, params=(tf, day)) for tf in TF_NAME}
    ox = pd.read_sql("SELECT * FROM options WHERE substr(t,1,10)=? ORDER BY t", db, params=(day,))
    fl = pd.read_sql("SELECT * FROM flows WHERE substr(t,1,10)=? ORDER BY t", db, params=(day,))
    rows, summ = [], {}
    for e in ev[ev.kind == "early"].itertuples():
        res, mfe, mae = outcome(bars.get(e.tf, pd.DataFrame(columns=["t", "h", "l"])), e)
        k = (e.tf, e.side); s = summ.setdefault(k, [0, 0]); s[0] += 1; s[1] += res.startswith("✓")
        ctx = json.loads(e.ctx or "{}"); o = (ctx.get("options") or {}).get("0dte") or (ctx.get("options") or {}).get("all") or {}
        rows.append((e.bar_t[11:16], TF_NAME.get(e.tf, e.tf), "底" if e.side == "bottom" else "顶", "%.0f%%" % e.prob, "%.2f" % e.price,
                     res, mfe, mae, o.get("put_wall", ""), o.get("call_wall", ""), o.get("zero_gamma", ""), "是" if e.pushed else ""))
    n_ok = len(ev[ev.kind == "confirm"])
    lines = []
    for (tf, side), (n, k) in sorted(summ.items(), key=lambda x: list(TF_NAME).index(x[0][0])):
        lines.append("%s %s：%d 个，事后证实 %d 个" % (TF_NAME[tf], "底" if side == "bottom" else "顶", n, k))
    summary = "；".join(lines) if lines else "今天没有早报信号"
    summary += "。√确认 %d 个。" % n_ok
    # html
    th = "".join("<th>%s</th>" % h for h in ("时间", "周期", "类型", "概率", "价格", "结果", "30根内最大顺向bp", "最大逆向bp", "看跌墙", "看涨墙", "零Gamma", "已推送"))
    tr = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(str(x)) for x in r) for r in rows)
    otab = ""
    if len(ox):
        pick = ox.iloc[::max(1, len(ox) // 14)]
        orows = []
        for r in pick.itertuples():
            d = json.loads(r.data or "{}"); z = d.get("0dte") or d.get("all") or {}
            orows.append((r.t[11:16], "%.2f" % r.spot, z.get("put_wall", ""), z.get("call_wall", ""), z.get("zero_gamma", ""), z.get("atm_iv", ""),
                          (d.get("iv_next") or {}).get("atm_iv", ""), z.get("net_gex_bn", ""), z.get("pc_volume", "")))
        otab = "<h2>期权（当日到期）</h2><table><tr>%s</tr>%s</table>" % ("".join("<th>%s</th>" % h for h in ("时间", "现价", "看跌墙", "看涨墙", "零Gamma", "当日平值IV%", "次近到期平值IV%", "净GEX(十亿$)", "看跌/看涨成交比")),
                                                                     "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % x for x in r) for r in orows))
    ftab = ""
    if len(fl):
        d = json.loads(fl.data.iloc[-1]); f = lambda k: d.get(k, 0) / 1e8
        ftab = "<h2>资金分布（收盘前最后一次，亿美元）</h2><table><tr><th></th><th>特大</th><th>大</th><th>中</th><th>小</th></tr>" + \
               "<tr><td>流入</td>%s</tr><tr><td>流出</td>%s</tr></table>" % ("".join("<td>%.2f</td>" % f("capital_in_" + k) for k in ("super", "big", "mid", "small")),
                                                                            "".join("<td>%.2f</td>" % f("capital_out_" + k) for k in ("super", "big", "mid", "small")))
    page = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPY 复盘 %s</title><style>body{font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:16px;background:#fff;color:#1b1f24}
@media (prefers-color-scheme:dark){body{background:#131722;color:#d1d4dc}td,th{border-color:#2a2f3d!important}}
table{border-collapse:collapse;margin:8px 0 20px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}td,th{border:1px solid #e3e6eb;padding:4px 8px;white-space:nowrap}
h1{font-size:18px}h2{font-size:15px;margin-top:18px}</style></head><body><h1>SPY 盯盘复盘 · %s</h1><p>%s</p>
<h2>早报信号与结果</h2><table><tr>%s</tr>%s</table>%s%s<p style="color:#888">结果按 v5 定义判定：底之后先涨 2×ATR 且没跌破低点 = ✓；顶反之。顺向/逆向 = 信号后 30 根K线内的最大有利/不利波动（bp）。</p></body></html>""" % (
        day, day, html.escape(summary), th, tr, otab, ftab)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    path = os.path.join(HERE, "reports", "%s.html" % day)
    open(path, "w", encoding="utf-8").write(page)
    return path, summary
