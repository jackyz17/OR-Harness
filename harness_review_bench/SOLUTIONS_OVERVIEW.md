# orarla 十题求解总览（round r11）

- 生成：2026-10-07 20:31
- 记忆库：`/home/ubuntu/.openclaw/workspace/bench_orarla/home`

| 题 | 类别 | 难度 | 参考 | 本轮 obj | check | 正确 | execs | 采用方法 |
|---|---|---|---|---|---|---|---|---|
| orarla_1 | planning | None | 10000.0 | 10000.0 | passed | Y | 3 | milp_pulp_cbc |
| orarla_2 | planning | None | 8350.0 | 8350.0 | passed | Y | 1 | milp_pulp_cbc |
| orarla_3 | planning | None | 168.0 | 34 | failed | N | 2 | orarla_3_pure_python_enum |
| orarla_4 | planning | None | 1200.0 | 1200.0 | passed | Y | 1 | milp_pulp_cbc |
| orarla_5 | planning | None | 450000.0 | 450000.0 | passed | Y | 1 | milp_pulp_cbc |
| orarla_6 | planning | None | 10000.0 | 10000 | passed | Y | 2 | orarla_6_milp_pulp_cbc |
| orarla_7 | planning | None | 3000.0 | 3000.0 | passed | Y | 1 | milp_pulp_cbc |
| orarla_8 | planning | None | 50000.0 | 50000.0 | passed | Y | 5 | milp_pulp_cbc |
| orarla_9 | planning | None | 150.0 | 150.0 | passed | Y | 1 | milp_pulp_cbc |
| orarla_10 | planning | None | 40020.0 | 40020.0 | passed | Y | 1 | milp_pulp_cbc |

> check 口径来自 `RESULTS_all10_r11.txt`；`correct` = (obj == ref)。
> 逐题产物见 `solutions/orarla_N/`；原始库表见 `bank_db/`。
