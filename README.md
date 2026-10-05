# TradingSPY500

SPY（标普 500 ETF）交易研究与实盘工具。只做多，按真实成本回测，所有结论都有数据和代码可以复现。

| 子项目 | 做什么 | 入口 |
|---|---|---|
| [`spy-leverage/`](spy-leverage/) | 做多杠杆：日线 RSI2 抄底、超短线组合（噪声区突破 + 隔夜持有）、10:00 区间预测与盈利概率、moomoo 实时监控屏 | [README](spy-leverage/README.md) · 双击 `spy-leverage/START_WINDOWS.bat` |
| [`spy-options/`](spy-options/) | 期权：0DTE 到 20 个交易日的 SPY 期权策略，波动率风险溢价 | [README](spy-options/README.md) · `spy-options/dashboard/index.html` |
| [`spy-ai/`](spy-ai/) | moomoo 实时盯盘：底顶 AI v6（28 类 moomoo 指标 + 真实成交量）、期权墙 / 零 Gamma / 资金分布、**波段模式**（抄底到顶）、桌面弹窗、每日复盘和 8 页看板；研究：策略工厂 309 组、波段研究 | [README](spy-ai/README.md) · 双击 `START_ALL_WINDOWS.bat` |

## 结论

| 问题 | 回测结果 | 文件 |
|---|---|---|
| 日线 RSI2 抄底（收盘 > 200 日线且 RSI2 < 10） | 1993–2026 共 282 笔，胜率 67%，每笔扣费后 +0.24%；2025 年后 21 笔，胜率 76%。唯一过全部关卡的抄底规则 | `spy-leverage/results/dip_study.json` |
| 超短线做多：噪声区突破、隔夜持有 | CME 微型期货成本下每次 +1.5bp、+2.5bp；在 OKX 永续（一买一卖 12bp）上全部亏钱 | `spy-leverage/results/edge_study.json` |
| 三条腿的组合（年波动 20%） | 年化 +19.4%，夏普 0.97（买入持有 0.76），最大回撤 −34% | 同上 |
| 1/3/5 分钟共振见底 → 10 倍做多 | 扣费前 ≈ 0，扣费后每次 −12bp，10 倍账户归零 | `spy-leverage/results/scalp_study.json` |
| 10:00 预测今天的区间 | 预测准（80% 线实际守住 77–87%），但区间下部挂单扣费后每次 −7 到 −9bp | `spy-leverage/results/range_study.json` |
| 20 倍做多 | 整个账户开 20 倍，一次 −5.6% 的隔夜跳空就爆仓；用张数控制有效杠杆，约 1.4–4 倍 | `spy-leverage/agent/SPY_20X_TRADER_PROMPT.md` |
| 0DTE 卖期权 | 704 种做法 0 种通过；10–20 天期限的 VRP 梯子通过 | `spy-options/results/study.json` |
| 日内认底认顶（v6，28 类 moomoo 指标） | 准确率 64–76%（随便挑 18–29%），实盘 3 天证实 70%；但一出信号就做多，扣 12bp 后每笔 −11 到 −13bp | `spy-ai/research/results/train6.json`、`bt6.json` |
| 13 类常见日内做多策略（策略工厂） | 309 组参数，0 组在 2018–2023 扣费后显著为正；唯一真信号是 15 分钟唐奇安突破（扣费前 +3.5bp/笔） | `spy-ai/research/results/factory_summary.json` |
| **波段：抄底日 + v6 择时进场，涨满 1 ATR 后 15 分钟顶离场** | 2024-01 至 2026-10-05 样本外 52 笔，胜率 71–79%，每笔扣费后 +0.31% 到 +0.42%（1 倍）；52 笔里 23 笔先逆向 ≥1%，所以 **50 倍不行（每笔 −43 元/280 元），20 倍可以（+17 到 +23 元/280 元，0 次爆仓）**。抓到了 2026-10-01 759 → 10-05 776：761.42 进、775.58 出 | `spy-ai/research/results/swing_final.json` |

研究登记簿 [`spy-leverage/research/REGISTRY.md`](spy-leverage/research/REGISTRY.md)：每个想法都先登记规则再回测，至今约 3,350 种配置（含 spy-ai 的 400 组）。

**总提示词**：[`MASTER_PROMPT.md`](MASTER_PROMPT.md)（技能 `spy-master-trader`）——整合两套系统的每日流程、交易手册、杠杆规则和研究纪律。

## 快速开始

**只想看效果：** 用浏览器打开 `spy-leverage/web/demo.html`。里面是几个真实交易日的逐分钟回放，不用安装任何东西。

**一键全开（这台电脑）：** 双击根目录 `START_ALL_WINDOWS.bat`——启动 OpenD（如果没开）、spy-ai 盯盘机器人（弹窗 + 波段模式 + 收盘复盘看板）和抄底灯网页 http://127.0.0.1:8766。不建新的 Python 环境：用本机已有的包和 `local_shims/futu`（把 `futu` 指向已装的 `moomoo` 包，两者接口相同）。

**实盘监控（Windows）：**
1. 安装 moomoo OpenD 并登录：https://www.moomoo.com/download/OpenAPI
2. 双击 `spy-leverage/START_WINDOWS.bat`（Mac 用 `START_MAC.command`）。第一次会自动安装依赖。
3. 浏览器打开 http://127.0.0.1:8765，点“开启声音和弹窗”。

**开发：**
```bash
cd spy-leverage && pip install -e ".[futu,dev]" && python -m pytest -q
cd ../spy-options && pip install -e ".[dev]" && python -m pytest -q
```

## AI 交易员技能

用 Claude Code 打开这个仓库，会自动加载 `.claude/skills/` 下的技能：

| 技能 | 用途 |
|---|---|
| `spy-master-trader` | **总操盘手（合并版）**：统领 spy-ai 实时盯盘、波段模式和下面所有技能 |
| `spy-quant-lab` | 总负责人：每天执行、每周复盘、每月重跑研究、研究纪律 |
| `spy-20x-trader` | 超短线做多组合、MES 张数、20 倍怎么用 |
| `spy-dip-trader` | 日线到月线抄底、OKX 永续成本与强平 |
| `spy-range-trader` | 10:00 区间预测、单子的盈利概率 |
| `spy-scalp-trader` | 1/3/5 分钟共振（已淘汰，只看不买）和监控屏 |

每个技能的拷贝也在 `spy-leverage/agent/`，可以直接复制给其他 AI 使用。

## 目录

```
spy-leverage/      做多杠杆（Python 包 spylev）：数据、指标、四项研究、实时监控、网页、一键启动脚本
spy-options/       期权研究（Python 包 spyopt）：定价、波动率、回测、看板
.claude/skills/    AI 交易员技能
.github/workflows/ 两个子项目的测试；spy-options 的看板可发布到 GitHub Pages
```

## 说明

- 数据来源：
  - IBKR（日线、近期分钟线、期权报价）；
  - GitHub 上的公开数据集（SPY 2022–24 分钟线、标普 2011–18 分钟线、长历史日线）；
  - 你自己的 moomoo 数据（`python -m spylev.data.crawl --source futu`）。
  - 大文件不进仓库，第一次运行时自动下载。
- 发布期权看板到 GitHub Pages：先在 Settings → Pages 选 “GitHub Actions”，再在 Settings → Secrets and variables → Actions → Variables 加 `ENABLE_PAGES = true`。
- 只用于研究和提示。回测不是承诺，杠杆会放大亏损，程序不会替你下单。
- 本仓库于 2026-10-05 从 `wenjun19711022-cpu/human-api` 的 `claude/relaxed-tesla-wjwgyf` 分支迁移过来，保留了全部提交历史。
