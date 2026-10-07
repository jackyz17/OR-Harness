# orarla 十题求解总览（round r10）

- 生成：2026-10-07 10:24
- 记忆库：`/home/ubuntu/.openclaw/workspace/bench_orarla/home`

| 题 | 类别 | 难度 | 参考 | 本轮 obj | check | 正确 | execs | 采用方法 |
|---|---|---|---|---|---|---|---|---|
| orarla_1 | planning | None | 10000.0 | 10000 | passed | Y | 1 | enumeration_based |
| orarla_2 | planning | None | 8350.0 | 167.0 | failed | N | 1 | direct_milp_2var |
| orarla_3 | planning | None | 168.0 | 34.0 | failed | N | 4 | milp_highs_integer |
| orarla_4 | planning | None | 1200.0 | 1200.0 | passed | Y | 1 | enumeration_2var_ilp_direct |
| orarla_5 | planning | None | 450000.0 | 450000 | passed | Y | 1 | enumeration_budget_2var |
| orarla_6 | planning | None | 10000.0 | 10000.0 | passed | Y | 1 | enumeration_2var_budget_effectiveness |
| orarla_7 | planning | None | 3000.0 | 3000.0 | passed | Y | 2 | milp_highs_integer |
| orarla_8 | planning | None | 50000.0 | 50000.0 | passed | Y | 3 | enumeration_2var_budget_effectiveness |
| orarla_9 | planning | None | 150.0 | 150 | passed | Y | 1 | enumeration_2var_analytic |
| orarla_10 | planning | None | 40020.0 | 40020 | passed | Y | 1 | enumeration_2var_budget_effectiveness |

> check 口径来自 `RESULTS_all10_r10.txt`；`correct` = (obj == ref)。
> 逐题产物见 `solutions/orarla_N/`；原始库表见 `bank_db/`。
