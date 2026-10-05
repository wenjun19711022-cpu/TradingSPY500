"""Fold the strategy-factory result into results/plan.json (headline, verdict, a new section, next steps)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "results", "plan.json"); P = json.load(open(p, encoding="utf-8"))
fs = json.load(open(os.path.join(HERE, "results", "factory_summary.json"), encoding="utf-8")); n = str(fs["n_configs"]); st = fs["star"]
P["headline"] = ("认得出底和顶（实盘两天 85 个信号证实 74%，回测 64–76%），但在“日内、OKX、只做多、12bp 成本”这组规则下，"
                 "我测过的 " + n + " 组策略配置（13 类常见量化策略）加上 6 种买底方法，没有一个扣费后能稳定赚钱。")
P["verdicts"][2] = {"status": "bad", "title": "自动交易：没有找到能赚钱的",
                    "text": "13 类常见日内做多策略、" + n + " 组参数，连 2018–2023 年（挑参数的那段）都没有一组扣 12bp 后显著为正。根本原因：SPY 开盘到收盘平均每天只涨约 2bp，"
                            "而一来回成本 12bp——任何择时都得多赚 6 倍于大盘漂移才能打平。"}
sec = {"title": "4b. 策略工厂：13 类常见量化策略（2026-10-04）", "tag": "已排除", "body": [
    "测了：日内持有 + 15 种过滤（VIX 期限结构、200 日线、昨天涨跌、跳空、月初月末、星期几）、开盘区间突破、跳空回补、VWAP 回踩、EMA/MACD 金叉、唐奇安突破、RSI2 超卖、"
    "布林下轨回归、KDJ 低位金叉、午后趋势延续、事件日（FOMC / CPI / 非农）、跨市场确认（QQQ / IWM / TLT）、底顶 AI 持有到收盘，共 " + n + " 组配置。",
    "结果：0 组通过第一关（2018–2023 年扣费后 t ≥ 2）。按毛利看，唯一三段都为正的是 15 分钟唐奇安突破（突破 26 根高点 + 2ATR 移动止损）：每笔毛利 2018–23 年约 +6bp、"
    "样本外 +" + str(st["oos_gross"]) + "bp（t=" + str(st["oos_t"]) + "），8 个完整年份 7 个为正。盈亏平衡成本约 4bp：一来回 1bp 时每笔约 +3.8bp（t≈2），OKX 吃单 12bp 时每笔 −7bp。",
    "其它家族要么毛利接近 0，要么只在挑参数的那几年有效（例如 15 分钟 EMA 金叉 2018–23 年毛利 +8.8bp，2024–26 年 −0.7bp）。"],
    "ev": "factory.json / factory_summary.json；看板“策略工厂”页"}
P["sections"] = [x for x in P["sections"] if not x["title"].startswith("4b.")]
P["sections"].insert(4, sec)
nx = P["sections"][-1]
nx["body"][2] = "③ 技术指标和常见量化策略这条路已经走到头：28 类 moomoo 指标、13 类策略、" + n + " 组参数都试过，再调只会过拟合。"
nx["body"][3] = ("④ 数据显示卡住的是成本：唯一有真信号的唐奇安突破，毛利每笔约 +3–6bp，要在一来回成本 ≤ 3bp 的渠道才可能为正（OKX 吃单 12bp、挂单 7bp 都不够）。"
                 "在现有规则不变的前提下，我不建议再投入回测。")
json.dump(P, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("plan ok")
