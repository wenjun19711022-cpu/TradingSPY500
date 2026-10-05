# spy-ai · moomoo 实时盯盘 + 底顶 AI v6 + 波段模式 + 复盘看板

和 `spy-leverage/`（日线 RSI2、超短线组合、区间面板）互补：这里负责**实时数据、信号、提醒、复盘和研究**。
程序只提醒、不下单。总提示词见仓库根目录 `MASTER_PROMPT.md`（技能 `spy-master-trader`）。

## 一句话结论（2026-10-06）

| 问题 | 结论 |
|---|---|
| 认得出 SPY 日内的底和顶吗 | 能。v6 准确率 64–76%（随便挑只有 18–29%），实盘 3 天证实 70% |
| 一出信号就在 OKX 做多（日内）能赚吗 | 不能。13 类常见日内策略、309 组参数、6 种买底方法，扣 12bp 后全部亏 |
| **抄底拿到顶（波段，可以过夜）** | **能。** 抄底日的第一个 15m/1h v6 底进场，涨满 1 个日线 ATR 后第一个 15m 顶离场：样本外 52 笔，胜率 71–79%，1 倍每笔 +0.31% 到 +0.42%（t 1.66 到 2.27，取决于资金费按假设还是按 OKX 实际） |
| 波段能开 50 倍吗 | 不能。52 笔里 23 笔先逆向 ≥1%，50 倍逐仓（OKX 第一档 mmr 1%）跌 1.0% 就爆仓；280 元/笔：50 倍每笔 −43 到 −49 元，20 倍 +17 到 +23 元、0 次爆仓 |
| 2026-10-01 759 → 10-05 776 那一波 | 规则在 10-01 10:45 15 分钟底 761.42 进场，10-05 15:45 15 分钟顶 775.58 离场，+1.6%；280 元 × 50 倍 ≈ +225 元（理想的 759 → 776 ≈ +293 元，已扣手续费和实际资金费） |

## 目录

```
watch/                 实盘程序（本机运行副本在 C:\Users\94868\spybt\watch，计划任务每天北京时间 21:15 启动）
  spy_watch.py         主循环：moomoo OpenD → 1/3/5/15/60 分钟 → v6 底顶；期权墙/零 Gamma；资金分布；断线补算；防休眠
  v6core.py            v6 实时引擎（和训练共用 model6/feat6.py，回放逐根 100% 一致）
  swing.py             波段模式：抄底日判定 → 进场 → 止损 / 止盈观察 / 15m 顶离场 / 时间止损；状态在 data/swing_state.json
  gex.py pushers.py store.py report.py   期权分析、桌面弹窗、SQLite、每日复盘
  dashboard.py + dashboard_tpl.html      看板（离线单文件）：总览、波段、实盘复盘、单日回看、模型、交易回测、策略工厂、优化路线
  replay_test.py replay6_test.py case_replay.py   回放测试（v5 / v6 一致性、10-01..10-05 波段案例）
  model6/              冻结的 v6 模型、特征代码和全部回测结果 JSON
research/              研究代码（本机副本在 C:\Users\94868\spybt\v6）
  dl_moomoo.py update_data.py vixdata.py      数据：moomoo 前复权分钟线 2018-09 起、日线、QQQ/IWM/TLT、VIX
  ind.py feat6.py lib6.py labels.py models6.py train6.py   v6：28 类 moomoo 指标、训练与选择
  bt6.py bt6_limit.py v7_*.py                  日内交易回测、挂单、成本感知模型、日内动量
  strat_lab.py factory_summary.py              策略工厂：13 类日内策略 309 组
  swing_study.py swing_final.py                波段研究（5 种进场 × 5 种出场 × 3 种杠杆）
  results/                                     所有结果 JSON
sync_from_local.py     把本机运行中的系统同步进仓库（不含实盘数据和行情数据）
```

## 怎么跑

- **实盘**：本机已配置好（计划任务）。手动：`watch\启动盯盘.bat`；看板：`watch\打开看板.bat`。
- **两套一起开**：仓库根目录 `START_ALL_WINDOWS.bat`（盯盘机器人 + spy-leverage 抄底灯网页 http://127.0.0.1:8766）。
- **研究重跑**：在 `research/` 里先 `py -3 update_data.py`（需要 OpenD 已登录），再 `train6.py` → `bt6.py` → `strat_lab.py` → `swing_study.py` → `swing_final.py`。系统 pandas 3.x 在这台电脑上会崩，脚本前设 `PYTHONPATH=C:\Users\94868\spybt\pylib`（pandas 2.2.3）。

## 规则和成本口径

- 信号 K 线收盘后，下一分钟开盘成交；止损在 K 线内先于止盈判断；15:55 后的日内信号只记录。
- OKX：吃单 0.05% + 滑点 0.01% 每边；资金费保守按 0.01%/8h，另报 OKX 近 3 个月实际（平均约 0.003%/8h，84% 时段为 0）；维持保证金 1%（第一档，最高 50 倍）。
- 隔夜：按 SPY 开盘跳空计算逆向幅度；OKX 夜间也在交易，真实的夜间低点可能更低，所以爆仓次数是下限。
- 切分：2018-10..2023 训练，2024-01..2026-03-25 验证（所有选择在这里），2026-03-26 以后保留期只看一次。
