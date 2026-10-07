# orarla 十题求解档案（供审核）

生成 2026-10-07 20:31（round r11）

## 目录
- `problems/`            — 10 题题面 + 参考答案（item_N.json）与 task-check 规约（check_N.json）
- `solutions/`           — 本轮(r11) agent 产物 orarla_N/：task.json(含 model)、candidates.json、cir.json、solve.py、result.json、check.json
- `SOLUTIONS_OVERVIEW.md`— 逐题：类别/难度、参考、本轮 obj、task-check、采用方法
- `bank_*.json`          — 记忆库各高层视图（experience/strategic/actions/snapshots/predictions/texts/evaluations/capability/archive/retention）
- `bank_db/`             — **原始 sqlite 表逐表导出**（executions/strategic_entries/belief_snapshots/action_records/contract_predictions/episode_closeouts/task_texts/prediction_contexts/meta/…）
- `or_harness.db`        — 记忆库数据库原件（完整快照）
- `calibration.json`     — 世界模型校准摘要（wm-calib/2）
- `run_logs/`            — 逐题 solve/induct/budget/聚合 日志
- `MANIFEST.json`        — 本次导出清单 + 各层计数
- `RESULTS_all10_r11.txt` / `results_r11.json` — 结果
- `REVIEW.md`            — 记忆库可读汇总（含策略库 + 校准）

## 本轮结果（round r11）
| 题 | 参考 | 本轮 obj | check | 正确 |
|---|---|---|---|---|
| orarla_1 | 10000.0 | 10000.0 | passed | Y |
| orarla_2 | 8350.0 | 8350.0 | passed | Y |
| orarla_3 | 168.0 | 34 | failed | N |
| orarla_4 | 1200.0 | 1200.0 | passed | Y |
| orarla_5 | 450000.0 | 450000.0 | passed | Y |
| orarla_6 | 10000.0 | 10000 | passed | Y |
| orarla_7 | 3000.0 | 3000.0 | passed | Y |
| orarla_8 | 50000.0 | 50000.0 | passed | Y |
| orarla_9 | 150.0 | 150.0 | passed | Y |
| orarla_10 | 40020.0 | 40020.0 | passed | Y |

准确率（obj==ref）：见 `RESULTS_all10_r11.txt`；

