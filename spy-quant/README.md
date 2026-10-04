# spy-quant · SPY 多周期 K 线与期权策略研究

围绕 SPY 的一套可复现工具：爬取 5 年 1 分钟 K 线并重采样到 3 分钟 … 4 小时，用真实期权报价校准的定价模型，回测从 0DTE 到 20 个交易日的期权策略，按预先设定的关卡做样本内 → 验证 → 留出检验，最后生成看板和每日开仓计划。

看板：`dashboard/index.html`（单文件，直接用浏览器打开）。

## 结论

| 问题 | 结果（2021-10 → 2026-10 真实数据） |
|---|---|
| 0DTE / 1DTE 卖期权能赚钱吗？ | 704 组配置 0 组通过。1 天期隐含波动（VIX1D）和随后实际波动几乎相等（方差比 1.02），没有可收的保险费；0DTE 铁鹰扣成本前每笔 +1.7% 风险收益，扣成本后 −2.2%。 |
| 哪里有优势？ | 10–20 个交易日期限。VIX9D/VIX 比随后实际波动贵（方差比 0.79 / 0.75）。1,820 组配置里 44 组通过全部关卡，全部是这一类：短腿在 ±2 倍预期波动之外的卖方价差/铁鹰。 |
| 推荐策略 | **VRP-20 梯子**：每个收盘开一档 20 日 SPY 铁鹰，短腿 ±2 倍预期波动（约 5–8 delta），翅膀再往外 1 倍，权利金赚到 50% 平仓。 |
| 表现 | 1,235 笔，胜率 99.0%，每笔风险收益 +0.97%（样本内 +0.91% / 验证 +0.71% / 留出 +1.35%），每年都为正，含 2022 熊市。点差翻倍或隐波下调 5% 后仍为正。 |
| 风险 | 20% 风险预算下 5 年年化 +2.4%、最大回撤 −0.9%，但样本里没有崩盘。按 2020 新冠（23 天 −34%）构造的情景回撤 −16%，2008 式情景 −19%。仓位上限看压力测试，不看历史回撤。 |

一句话：短期权（0DTE）这条路被成本卡住，和上一轮“策略工厂”在正股上的结论一样；能留下来的是 2–4 周期限的波动率风险溢价，胜率高、单笔薄、尾部风险真实存在。不要把 20% 以上的账户放进最大亏损预算。

## 目录

```
spyq/
  data/        crawl.py  多数据源 5 年 1 分钟爬虫（moomoo/富途 OpenD、Alpaca、Polygon、IBKR、yfinance、CSV）
               resample.py  9:30 开盘锚定重采样 1m→3m/5m/10m/15m/30m/1h/2h/4h/1d，数据完整性报告
               store.py  按年分区的 parquet 库    vix.py  VIX1D/VIX9D/VIX/VIX3M    rates.py  利率与 SPY 除息
  options/     bs.py  向量化 Black 定价/希腊值/隐波    smile.py  按期限拟合的波动率微笑 + VIX→平值隐波
               structures.py  铁鹰/价差/铁蝶/跨式    costs.py  点差与佣金
  indicators/  vol.py  Parkinson/Garman-Klass/Yang-Zhang、HAR-RV 预测    regime.py  VRP、期限结构、事件、趋势
  backtest/    daily.py  日线引擎（0DTE 开盘、1DTE、N 日梯子）    minute.py  1 分钟路径精确引擎（0DTE 研究）
               metrics.py  胜率、盈亏比、Newey-West t、DSR    validation.py  因子工厂 + 关卡 + 敏感性
  strategies/  factory.py  预注册网格与关卡    composite.py  最终策略与开仓计划
  risk/        sizing.py  梯子资金曲线、回撤刹车、Kelly    stress.py  崩盘情景压力测试
  live/        plan.py  今晚开哪一档    snapshot.py  采集期权链快照用于重新校准
  report/      dashboard.py + template.html  看板
scripts/run_study.py   一键跑完全部研究，写 results/study.json
tests/                 单元测试，含两个“零优势检验”
docs/STRATEGY.md       策略说明书（规则、仓位、出场、到期日操作）
docs/RESEARCH.md       GitHub 工具与策略调研，以及借鉴了什么
```

## 快速开始

```bash
cd spy-quant
pip install -e ".[dev]"          # numpy pandas scipy pyarrow pytest
pytest -q                         # 17 个测试
python scripts/run_study.py       # 用仓库自带的 IBKR 数据跑完整研究（约 2 分钟）
python -m spyq.report.dashboard   # 生成 dashboard/index.html
python -m spyq.live.plan --equity 100000
```

### 爬 5 年 1 分钟 K 线（在你自己的电脑上）

```bash
pip install futu-api               # moomoo / 富途 OpenD，先启动 OpenD
python -m spyq.data.crawl --source futu --years 5
# 其他来源：--source alpaca（APCA_API_KEY_ID/APCA_API_SECRET_KEY）、polygon（POLYGON_API_KEY）、ibkr（TWS/Gateway）
# 已有导出文件：python -m spyq.data.crawl --csv 旧数据.csv --end-labelled   （moomoo 按收盘时间标记分钟线）
```

爬虫按月续传，写入 `data/bars/SPY/1m/<年>.parquet`，再生成 3m、5m、10m、15m、30m、1h、2h、4h、1d，并输出 `data/bars/SPY/integrity_1m.csv`（每天缺几根、坏 tick、OHLC 矛盾）。重采样以 9:30 为锚：4 小时线是 9:30–13:30、13:30–16:00，半日市自动处理；已和 IBKR 原生 5m/30m 逐根比对一致。

有了本地 1 分钟库后，`run_study.py` 和看板会自动改用完整 5 年分钟数据，`spyq.backtest.minute` 可以对 0DTE 的盘中止损/止盈做逐分钟回测。

VIX 历史可以从 CBOE 免费下载（`spyq.data.vix.download_cboe()`），放到 `data/raw/cboe/` 后会覆盖仓库自带的 IBKR 数据；利率可用 `spyq.data.rates.download_fred()`。

## 方法要点

- **定价**：没有免费的历史期权数据，所以期权价格由模型给出，但模型的每个参数都来自真实数据。微笑形状用 2026-10-02 收盘 SPY 三个到期日（1/5/20 个交易日）的 29 个真实报价拟合，误差 0.9%（相对平值隐波）；平值隐波水平 = VIX1D×0.94 / VIX9D×0.84 / VIX×0.81，中间期限按总方差插值；远期计入利率和 SPY 季度除息。10/30 到期的 715P、740P、800C 模型价与实盘中价相差 0.5–3%。
- **成本**：每张每边 $0.70；点差 0.01 + 0.4%×价格（上限 $0.10），VIX 高于 20 时同比放大。到期作废的腿不收平仓费；实值腿按内在价值减点差平仓。
- **零优势检验**：在“隐波 = 实际波动、无偏斜、零成本”的模拟世界里，日线引擎 5 种结构、分钟引擎 5 种止损/止盈规则的平均盈亏都与 0 无显著差异。引擎本身不制造优势。
- **关卡**：G1 样本内（t ≥ 2、盈亏比 ≥ 1.15）→ G1b 稳健性（点差×2、隐波−5% 仍盈利）→ G2 验证期 → G3 留出期。多日梯子每天开仓，t 值用 Newey-West 处理重叠。
- **披露**：冒烟测试碰过全样本，梯子和稳健性关卡是第一次全量运行后加的；验证期和留出期不是完全干净。上线前请模拟盘 2–3 个月。

## 局限

- 日线引擎里“触价止损”和“50% 止盈”只在收盘价上检查；盘中路径请用分钟引擎。
- 校准只用了一个交易日的报价。每天收盘跑 `python -m spyq.live.snapshot` 积累快照后重新拟合，是改进模型最有效的一步。
- SPY 期权实物交割：到期日 15:45 前必须平掉或移仓实值短腿。若做 0DTE，SPX/XSP 现金交割更合适。
- 这里没有任何收益保证。
