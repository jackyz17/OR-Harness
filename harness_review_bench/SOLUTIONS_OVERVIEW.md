# orarla 十题求解总览（round r5）

- 生成：2026-10-04 17:33
- 记忆库：`/home/ubuntu/.openclaw/workspace/bench_orarla/home`

| 题 | 类别 | 难度 | 参考 | 本轮 obj | check | 正确 | execs | 采用方法 |
|---|---|---|---|---|---|---|---|---|
| orarla_1 | planning | Medium | 219816.0 | None | unchecked | N | 7 | milp_labor_training |
| orarla_2 | generic | Medium | 125.0 | 125 | passed | Y | 1 | direct_calculation |
| orarla_3 | inventory | Hard | 10349920.0 | 10349920.0 | passed | Y | 6 | milp_aggregate_production_v2 |
| orarla_4 | generic | Easy | 30400.0 | 30400.0 | passed | Y | 5 | milp_animal_farm |
| orarla_5 | planning | Easy | 18.6943 | 16.0 | insufficient | N | 2 | milp_meal_fiber |
| orarla_6 | inventory | Hard | 10755.0 | 10755.0 | passed | Y | 2 | milp_lot_sizing |
| orarla_7 | inventory | Medium | 904590.0 | 904590.0 | passed | Y | 1 | network_flow_transport |
| orarla_8 | scheduling | Medium | 14.0 | 14 | passed | Y | 1 | enumerate_assignment |
| orarla_9 | planning | Hard | 623.0 | 623.0 | passed | Y | 2 | milp_binary_indicator |
| orarla_10 | inventory | Easy | 3.0 | 3.0 | passed | Y | 3 | milp_set_cover |

> check 口径来自 `RESULTS_all10_r5.txt`；`correct` = (obj == ref)。
> 逐题产物见 `solutions/orarla_N/`。
