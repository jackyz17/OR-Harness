# orarla 十题求解档案（供审核）

生成 2026-10-05 16:29（round r6，champions: induction back to the agent）

## 目录
- `problems/`            — 10 题题面 + 参考答案（item_N.json）与 task-check 规约（check_N.json）
- `solutions/`           — 本轮(r6) agent 产物 orarla_N/：task.json(含 model)、candidates.json、cir.json、solve.py、result.json、check.json
- `SOLUTIONS_OVERVIEW.md`— 逐题：类别/难度、参考、本轮 obj、task-check、采用方法
- `bank_*.json`          — 本轮 bench 记忆库各层原始导出
- `calibration.json`     — 世界模型校准摘要（wm-calib/2）
- `RESULTS_all10_r6.txt` / `results_r6.json` — 结果
- `REVIEW.md`            — 记忆库可读汇总（含策略库）

## 本轮结果（round r6）
| 题 | 参考 | 本轮 obj | check | 正确 |
|---|---|---|---|---|
| orarla_1 | 219816.0 | 130119.99999999965 | unchecked | N |
| orarla_2 | 125.0 | 50.0 | passed | N |
| orarla_3 | 10349920.0 | 10349920.0 | passed | Y |
| orarla_4 | 30400.0 | 30400.0 | passed | Y |
| orarla_5 | 18.6943 | 18.7 | failed | N |
| orarla_6 | 10755.0 | 13255.0 | passed | N |
| orarla_7 | 904590.0 | 904590.0 | passed | Y |
| orarla_8 | 14.0 | 14 | passed | Y |
| orarla_9 | 623.0 | 623.0 | passed | Y |
| orarla_10 | 3.0 | None | none | N |

准确率（obj==ref）：见 `RESULTS_all10_r6.txt`；

