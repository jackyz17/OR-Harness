# orarla 十题求解总览（round r6）

- 生成：2026-10-05 16:29
- 记忆库：`/home/ubuntu/.openclaw/workspace/bench_orarla/home`

| 题 | 类别 | 难度 | 参考 | 本轮 obj | check | 正确 | execs | 采用方法 |
|---|---|---|---|---|---|---|---|---|
| orarla_1 | planning | Medium | 219816.0 | 130119.99999999965 | unchecked | N | 5 | milp_time_indexed |
| orarla_2 | generic | Medium | 125.0 | 50.0 | passed | N | 5 | milp_training_allocation |
| orarla_3 | inventory | Hard | 10349920.0 | 10349920.0 | passed | Y | 4 | milp_hiring_firing_decoupled |
| orarla_4 | generic | Easy | 30400.0 | 30400.0 | passed | Y | 5 | milp_direct_integer |
| orarla_5 | planning | Easy | 18.6943 | 18.7 | failed | N | 4 | milp_direct_integer |
| orarla_6 | inventory | Hard | 10755.0 | 13255.0 | passed | N | 2 | milp_production_inventory_backlog |
| orarla_7 | inventory | Medium | 904590.0 | 904590.0 | passed | Y | 3 | milp_transport_integer |
| orarla_8 | scheduling | Medium | 14.0 | 14 | passed | Y | 1 | enumerate_exclude_one |
| orarla_9 | planning | Hard | 623.0 | 623.0 | passed | Y | 2 | milp_direct_integer |
| orarla_10 | inventory | Easy | 3.0 | None | none | N | 0 |  |

> check 口径来自 `RESULTS_all10_r6.txt`；`correct` = (obj == ref)。
> 逐题产物见 `solutions/orarla_N/`。
