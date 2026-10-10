# harness_review_bench — binding `config.solver` mismatch cases

**同一根因**：候选把 solver **重复**写进 `config`，写法与执行侧规范值不一致 →
绑定时按 config 键不一致判 **`identity_mismatch`** → `benefit/interval/risk` 三维被 block，
**只剩 cost** 进校准。（顶层 `candidate.solver` 与执行 solver 其实是一致的。）

样本来自**实际已存储的、benefit 被判 identity_mismatch 的评估**，非构造。

| # | task | 预测 config.solver | 顶层 candidate.solver | bound-action params.solver | 执行 solver.name | 被 block |
|---|---|---|---|---|---|---|
| 1 | nl4opt_0135 | `pulp_cbc` | `pulp` | `pulp` | `pulp` | benefit, interval, risk |
| 2 | nl4opt_0140 | `pulp CBC` | `pulp` | `pulp` | `pulp` | benefit, interval, risk |
| 3 | nl4opt_0141 | `CBC` | `pulp_cbc` | `pulp_cbc` | `pulp_cbc` | benefit, interval, risk |
| 4 | nl4opt_0159 | `pulp_CBC` | `pulp` | `pulp` | `pulp` | benefit, interval, risk |
| 5 | nl4opt_0184 | `CBC_CMD` | `pulp` | `pulp` | `pulp` | benefit, interval, risk |

逐案明细：`cases/<task>.json`；机读：`cases.jsonl`。

## 复核要点
1. **三处 solver 值**：`candidate.solver`(规范) / `config.solver`(重复且常不同) / `action.params.solver`(binder 实际比对的对象) / `execution.solver.name`。看它们如何互相不一致。
2. binder 取 `params.get('solver')` 当 actual；若该值与 `config.solver` 不等 → mismatch；若为 None → unknown。
3. `evaluation.attribution` 把 `config.solver` 标进 benefit/interval/risk 三维。
4. config 里除 solver 外的键：`formulation`/`integer`/`warm_start` 属 approach 键（同样挡 benefit）；`msg`/`variable_type`/`variables`/`objective` 等属软键（不挡）。
