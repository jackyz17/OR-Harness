# orarla 十题求解总览（round r9）

- 生成：2026-10-06 18:20
- 记忆库：`/home/ubuntu/.openclaw/workspace/bench_orarla/home`

| 题 | 类别 | 难度 | 参考 | 本轮 obj | check | 正确 | execs | 采用方法 |
|---|---|---|---|---|---|---|---|---|
| orarla_1 | planning | None | 10000.0 | 10000.0 | passed | Y | 2 | PuLP_CBC_ILP |
| orarla_2 | planning | None | 8350.0 | None | insufficient | N | 3 | ilp_cbc |
| orarla_3 | planning | None | 168.0 | None | unchecked | N | 6 | ilp_scipy_milp_v2 |
| orarla_4 | planning | None | 1200.0 | 1200 | passed | Y | 4 | enumeration_with_constraints |
| orarla_5 | planning | None | 450000.0 | 3000.0 | failed | N | 2 | LP_ilp_pulp_cbc |
| orarla_6 | planning | None | 10000.0 | None | none | N | 0 |  |
| orarla_7 | planning | None | 3000.0 | None | none | N | 0 |  |
| orarla_8 | planning | None | 50000.0 | 50000 | passed | Y | 1 | enumeration_with_constraints |
| orarla_9 | planning | None | 150.0 | 150.0 | passed | Y | 1 | milp_highs_integer |
| orarla_10 | planning | None | 40020.0 | 40020 | passed | Y | 1 | enumeration_with_constraints |

> check 口径来自 `RESULTS_all10_r9.txt`；`correct` = (obj == ref)。
> 逐题产物见 `solutions/orarla_N/`。
