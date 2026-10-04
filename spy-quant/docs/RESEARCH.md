# GitHub 量化工具与 0DTE 策略调研

2026-10-04 在 GitHub 上检索期权回测、0DTE、波动率曲面、GEX、过拟合检验和看板相关的项目，下面是读过或看过说明的部分，以及本项目从中拿了什么。星数为检索当天的数字。

## 回测与交易框架

| 项目 | 是什么 | 本项目的取舍 |
|---|---|---|
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) ★29.6k | Rust 内核的事件驱动交易引擎，回测与实盘同一套代码 | 对这次的研究太重；以后要做自动下单时是首选的执行层 |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) ★9.3k | 向量化回测，一次跑成千上万组参数 | 借鉴“整段历史一次性数组计算”：日线引擎把 1,820 组配置一次跑完（约 2 分钟），分钟引擎把所有交易日的每一分钟同时定价 |
| [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) ★7.7k | 组合绩效报表 | 只需要其中一小部分，自己实现在 `backtest/metrics.py`，并加了 Newey-West t 值和 DSR |
| [Lumiwealth/lumibot](https://github.com/Lumiwealth/lumibot) ★2.1k | 多券商（含 IBKR、Alpaca）的交易机器人框架，支持期权 | 实盘接入的备选 |
| [goldspanlabs/optopsy](https://github.com/goldspanlabs/optopsy) ★1.5k | 基于真实期权链（EOD）的期权策略回测库 | 结构命名和“按 delta/距离选腿”的思路；它需要真实历史期权链，如果你购买了 OptionsDX / ThetaData 数据，可以直接拿它交叉验证本项目的模型价结果 |
| [wangzhe3224/awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading) ★5.2k | 系统化交易资源清单 | 用作检索地图 |

## 0DTE 专项研究

| 项目 | 结论或特点 | 对本项目的影响 |
|---|---|---|
| [M-man2591/0dte-research-platform](https://github.com/M-man2591/0dte-research-platform) | 0DTE 研究平台，带公开的代码审查。作者自己指出：没有波动率微笑的合成 Black-Scholes 期权链会让铁鹰“跑出” Sharpe 8，属于模拟器假象；修正年化错误前 Sharpe 曾被高估 20 倍 | 本项目的定价不用平坦隐波：微笑形状和水平都用真实报价校准，并加了“零优势检验”（公平定价 + 零成本时任何规则都应赚 0） |
| [emlama/gex-backtesting](https://github.com/emlama/gex-backtesting) | SPX 0DTE 逐笔成交（Polygon）计算方向性 GEX，预注册假设 + FDR 校正 | 预注册关卡的做法一致；GEX 需要真实持仓/成交数据，本次没有，列为下一步 |
| [severin-spagnola/iv-surf-classifier](https://github.com/severin-spagnola/iv-surf-classifier) | 用 SPY 0DTE 隐波曲面特征训练 LightGBM，滚动回测显示没有优势，附泄漏审计 | 与本项目“0DTE 无优势”的结论方向一致 |
| [KIBA0993/Alpaca-Hackathon](https://github.com/KIBA0993/Alpaca-Hackathon) | 0DTE 交易代理，作者自己的一年回测显示没有优势，于是只交付风控架构 | 同上 |
| [toomanyforks/ema-vwap-0dte-study](https://github.com/toomanyforks/ema-vwap-0dte-study) | 检验 5 分钟 EMA9×VWAP 0DTE 策略，2024–2025 年 38 个标的 28,335 笔 | 分钟引擎支持同类检验（任意进场时间、止损、止盈） |
| [jefrnc/ibkr-odte-strategies](https://github.com/jefrnc/ibkr-odte-strategies) | IBKR 上开发和回测 0DTE 的脚本 | IBKR 接入参考 |
| [klou23/0DTE-SPY-Backtesting](https://github.com/klou23/0DTE-SPY-Backtesting) | 简单的 SPY 0DTE 买 Call / 铁鹰回测 | — |
| [DivyamBanga/OptionHarvest](https://github.com/DivyamBanga/OptionHarvest) | QQQ 0DTE 卖权回测引擎 | — |

多个独立项目对 0DTE 得出“扣成本后没有稳定优势”的结论，本项目用真实 VIX1D 与实际波动的对比给出了原因：1 天期几乎没有波动率风险溢价。

## 波动率曲面、GEX 与过拟合检验

| 项目 | 用途 | 取舍 |
|---|---|---|
| [XanderRobbins/Arbitrage-Free-Volatility-Surface](https://github.com/XanderRobbins/Arbitrage-Free-Volatility-Surface) | SVI 与 Heston 校准、无套利检查 | 现在只有 29 个报价，用“标准化距离的分段二次式”已经拟合到 0.9% 误差；积累几周快照后升级为 SVI |
| [vollib/vollib](https://github.com/vollib/vollib) ★1.0k | Jäckel “Let's be rational” 隐波求解 | 本项目用向量化二分法（稳健、够快）；要更快可换成它 |
| [Matteo-Ferrara/gex-tracker](https://github.com/Matteo-Ferrara/gex-tracker) ★218 | 由 CBOE 期权链计算做市商 GEX | 可作为 regime 特征接入 `indicators/regime.py` |
| [quantskills/skill-backtest-overfit](https://github.com/quantskills/skill-backtest-overfit) | DSR、PBO（CSCV）、Harvey-Liu 折扣 | 实现了 DSR；PBO 是下一步 |
| [DaruFinance/deflated-sharpe](https://github.com/DaruFinance/deflated-sharpe) | 按原论文校验过的 DSR/PSR 实现 | 对照公式 |

## 看板与数据接入

| 项目 | 用途 |
|---|---|
| TradingView lightweight-charts（[louisnw01/lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python) ★2.2k 是其 Python 封装） | 看板的多周期 K 线 |
| futu-api（moomoo / 富途 OpenAPI 官方 SDK，PyPI） | 5 年 1 分钟 K 线、实时期权链与快照（`FutuSource`、`live/plan.py --futu`、`live/snapshot.py`） |
| ib_async（ib_insync 的后继） | IBKR TWS/Gateway 历史数据（`IBKRSource`） |
| Alpaca Market Data v2 / Polygon REST | 备选分钟数据源 |

## 结合后的设计

1. 数据：多源爬虫 + 9:30 锚定重采样 + 完整性报告，任何一个源都能填同一个 parquet 库。
2. 定价：Black 远期模型 + 按期限拟合的微笑 + VIX 家族给水平 + 利率/除息。不用真实历史期权链也能回测，但每个参数都可追溯到真实报价。
3. 验证：预注册网格（1,820 组）→ 样本内 → 稳健性（点差×2、隐波−5%）→ 验证 → 留出；重叠交易用 Newey-West；报告 DSR；零优势检验。
4. 风险：定义风险结构、按最大亏损分摊的梯子预算、回撤刹车、崩盘情景压力测试。
5. 产出：看板（多周期 K 线 + 指标面板 + 研究页）和每日开仓计划。
