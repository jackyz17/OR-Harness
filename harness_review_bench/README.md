# orarla 十题求解档案（供审核）

生成 2026-10-06 18:20（round r9，champions: induction back to the agent）

## 目录
- `problems/`            — 10 题题面 + 参考答案（item_N.json）与 task-check 规约（check_N.json）
- `solutions/`           — 本轮(r9) agent 产物 orarla_N/：task.json(含 model)、candidates.json、cir.json、solve.py、result.json、check.json
- `SOLUTIONS_OVERVIEW.md`— 逐题：类别/难度、参考、本轮 obj、task-check、采用方法
- `bank_*.json`          — 本轮 bench 记忆库各层原始导出
- `calibration.json`     — 世界模型校准摘要（wm-calib/2）
- `RESULTS_all10_r9.txt` / `results_r9.json` — 结果
- `REVIEW.md`            — 记忆库可读汇总（含策略库）

## 本轮结果（round r9）
| 题 | 参考 | 本轮 obj | check | 正确 |
|---|---|---|---|---|
| orarla_1 | 10000.0 | 10000.0 | passed | Y |
| orarla_2 | 8350.0 | None | insufficient | N |
| orarla_3 | 168.0 | None | unchecked | N |
| orarla_4 | 1200.0 | 1200 | passed | Y |
| orarla_5 | 450000.0 | 3000.0 | failed | N |
| orarla_6 | 10000.0 | None | none | N |
| orarla_7 | 3000.0 | None | none | N |
| orarla_8 | 50000.0 | 50000 | passed | Y |
| orarla_9 | 150.0 | 150.0 | passed | Y |
| orarla_10 | 40020.0 | 40020 | passed | Y |

准确率（obj==ref）：见 `RESULTS_all10_r9.txt`；

