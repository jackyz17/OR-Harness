---
name: or-harness-wm
description: >
  Formulate, solve, check, retry, and learn from large or coupled industrial
  optimization tasks, including LP, MILP, scheduling, routing, allocation,
  and supply chains. Use for repeated optimization work, strategy selection
  by predicted quality/cost/risk, task-level answer checking, execution-cost
  recording, offline strategy learning, or explicit OR-Harness requests.
  Skip ordinary isolated math questions and pure proofs unless the user
  explicitly requests the Harness workflow; exclude non-optimization work.
---

# OR-Harness: one scored solving workflow

Use one memory home (`--home` or `OR_HARNESS_HOME`) throughout the task. The agent proposes methods; the framework predicts G/C/R/H+, computes U, selects a candidate, executes it and retains attributable facts. Configure the world-model provider (`--world-model URL::MODEL` or `ORHarness(world_model=...)`) before planning.

## Prepare and propose

Create task JSON with a stable `task_id`, actual original text/data, and `family`. Keep the task ID across repairs; use a different ID for a different original problem. Prepare CIR when useful and inspect `profile` / `recall`. Read admitted strategic entries first, then relevant executed cases. Retrieval similarity discovers evidence; it does not select a solving method or prove applicability. Missing/degraded recall remains unknown.

Propose one or more methods in a JSON list. One candidate is sufficient; do not invent alternatives just to fill a menu. Each candidate identifies its steps, solver and relevant execution settings, for example:

```json
[{"name":"joint factory assignment","steps":["Preserve BE/CFC same-factory coupling in one integer model","Solve and check the original task"],"solver":"highs","config":{"time_limit":60}}]
```

## Plan, execute, check and record

The normal path is `profile / recall → plan-next → execute --prediction → check-task → record → close-episode → consume frozen H+`.

```bash
orx profile --task task.json
orx recall --task task.json --top 5
orx plan-next --task task.json --episode E --candidates candidates.json
# P is result.selected_prediction_id from the decision above.
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
# EX is result.execution_id; CHECK uses this task's existing acceptance evidence.
orx check-task EX --check CHECK
orx record --from-staged EX
orx close-episode --task T --episode E --terminal completed
```

`plan-next` freezes one context, calls the world model once per candidate and saves `U = αG − βC − γR`. Read `result.selected_prediction_id`, `result.selected_score` and `result.plan.candidates[]`. The score reports raw signals, actual terms, common resource dimensions, scale sources and missing-data reasons. Stable ties keep proposal order. A single candidate is scored too. Existing budget/execution constraints apply; no universal U-positive threshold is imposed.

If no executable prediction is returned, inspect the reason and revise candidates or repair the forecast before planning again. View an existing forecast with `orx inspect --bank predictions --prediction P`. Bare strategy/solver arguments, independent Python forecasts, manual choices and later binding cannot authorize an execution. Shadow scoring creates no attempt. `execute` checks the saved U selection, identity, known configuration and unused claim before solver startup; it records one actual adoption and binds the original prediction. No routine `choose-next` or second forecast is needed.

Implement the selected steps faithfully. Return protocol-valid solver status, objective/bound/gap, actual variables and available method/config receipts. The expected method is a plan; the performed method comes from actual evidence. A failed attempt is still real evidence: check when evaluable and save it with `record --from-staged EX`. A repair/retry proposes a fresh candidate and uses `plan-next` again. Keep the same original problem; do not change its checker or reference to fit an answer.

Task checking covers only the declared bases and reports unchecked scope and provenance. A passing partial checker does not establish complete original-task acceptance. Declare actual rule adoption with `--used-entry-ids ENTRY@VERSION` (empty string for adopted none), separate from recall/adaptation citations. See [modeling.md](references/modeling.md) and [episode_closeout.md](references/episode_closeout.md).

## Interpret the signals and costs

G is `P(original-task acceptance | faithful execution of candidate steps under agreed solving conditions)`, `effective_completion/task_check_passed_given_faithful_execution`, probability, `wm-g-faithful/1`. A method's own omissions remain omissions under faithful execution. `benefit.notes` gives one grounded condition → step → expected consequence sentence; the main agent chooses revisions. G observations require attributable faithful execution and adequate declared acceptance evidence; crashes, implementation deviations and unclear attribution remain unknown.

R is one attempt's technical implementation/execution failure probability, read only from `implementation_failure.probability`. It covers code/API/solver exit/result-protocol problems. Ordinary timeouts, environment blocks, infeasibility and business checks retain separate diagnostics. Finite probabilities in [0,1], including 0, are valid. If any ranked candidate lacks R, every candidate omits R for that comparison.

C predicts raw resources from the shared decision anchor to real task termination, including modelling, solves, checking, repair and retry. It uses common same-scope valid dimensions and component deduplication. Each dimension uses `clip(log1p(predicted) / log1p(peak), 0, 1)` with a positive peak from applicable pre-decision measured history or explicit configuration. Peaks never come from the current candidate maximum. The remaining dimensions use the same normalized resource weights for every candidate. Missing values or reliable scales omit that dimension for all candidates. With comparable G, a partial U remains usable; absent comparable G returns no decision. H+ never enters U.

The runner supplies its known execution host/model/interface and decision phase through `execution_context` / `OR_EXECUTION_CONTEXT`. Use measured host reports, full input+output tokens and deduplicated calls when available. Attempt totals keep their scope; a last-action time is a lower bound, not end-to-end completion. Contradictory dimensions remain unknown. Global C/G residuals are diagnostics; conditional facts and relevant cases inform future predictions. Details and explicit scale configuration: [strategy_outcome.md](references/strategy_outcome.md), [commands.md](references/commands.md).

## Close and settle the frozen induction jobs

### Frozen H+ and the building handoff

Choose `completed`, `failed`, `aborted` or `budget_exhausted` to match the real terminal state. `close-episode` publishes calibration and returns `result.induction_schedule.pending_ids`; re-closing shows any jobs still pending. Finish each current task's required reviews before starting the next training task. Require `induction-status.completed:true` before declaring library building complete.

H+ is frozen in the same world-model prediction inside `plan-next`: `capability_gain.decision` is JSON 0 or 1. A positive includes a learning hypothesis and concrete verification; hand that check to the solving agent. Actual execution uses its original prediction ID. Only actual adopted attempts receive scheduling. A failed or insufficiently checked H+=1 still requires one review. H+=0 skips new induction while preserving execution, existing-rule use, checks and utility management. Missing/illegal signals schedule 0 with their separate diagnostics; unexecuted candidates stay unobserved.

Solving alone is not a maintenance fact. Completed induction automatically binds the REAL induction action and its rule versions. Predicted-but-unexecuted candidates stay unobserved. Do not declare library building finished with pending jobs. `process_induction_jobs` is the Python callback entry for the same outer-agent handoff. Keep hypothesis truth separate from the agent-supported substantive increment; legacy `evaluate-capability` reports later transfer evidence. It never changes G/C/R strategy ranking.

The outer building agent consumes every pending J with `induction-material --induction-id J` and one submission. Resume the same job/session after interruption; repeated close and identical submit reuse the action and receipt. The framework does not invent a method or make a hidden induction model call. Material pages preserve the real trigger and advance controls within the total budget. Cite real artifact/check/fact references; minimal direct-support facts survive eviction. Later corrections affect future supported labels, preserving original inputs and checks.

The agent writes reusable `when / do / expect`, chooses comparison cases, derivation, boundary checks and publication scope. Existing cases plus checked derivation or small constructed tests can support scoped publication before any third real task. Keep entry IDs across revisions and actual version adoption; act on versioned entries through the shared lifecycle. An explicit publication choice is the rule's use decision; the learning label depends on the real review and its supporting facts. Prediction H+=0 means no new induction; learning label 0 means an actual supported no-increment review.

### 有据的新增或修订

先执行 `orx induction-material --induction-id J`。随后同时提交短规则和学习结论：

```bash
orx induce --induction-id J \
  --relation '{
    "when":"Y可有限枚举，固定Y后整数X的非空可行域是有限区间，且目标对X单调。",
    "do":"跳过没有可行X的Y。对其余Y按目标方向选相应区间端点，比较所有Y。目标对X不单调时，完整枚举可行X。",
    "expect":"在所述条件下保留原问题的最优目标值，并减少对X的枚举。",
    "evidence":["EX_A","EX_B","EX_BOUNDARY"],
    "publication":{"state":"published","reason":"端点推导和记录的小规模枚举支持这一范围。","scope":["有限整数区间、固定Y时目标对X单调"]}
  }' \
  --review '{
    "outcome":"new_rule",
    "conclusion":"明确了端点选择方法和非单调目标下的回退条件。",
    "comparison":{
      "sufficient":true,"substantive":true,"basis":"derivation",
      "source":"实际A、B题的输入和枚举记录，以及小规模非单调边界试验。",
      "assessment":"两种目标方向都由端点推导解释，边界试验核对了完整枚举回退。",
      "evidence_refs":["EX_A","EX_B","EX_BOUNDARY"]
    }
  }'
```

`--relation` 保存规则、证据和发布选择；`--review` 保存真实学习结论。实际新版本与有据的实质增量可以得到学习标签 1。修订已有条目时在 relation 中增加实际 `entry_id` 和 `operation:"revise"`，保留该编号及版本链；outcome 可用 `revised`。发布本身不自动产生学习正例。

### 已覆盖或没有增量

可以完成空审阅，也可以在同一提交中给实际已有版本补证据。空审阅示例：

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"already_covered",
    "conclusion":"实际执行采用的端点方向及回退条件已由当前规则版本覆盖，没有可复用的新增量。",
    "comparison":{
      "sufficient":true,"substantive":false,
      "source":"已对照当前规则版本、实际执行步骤及原题检查记录。",
      "assessment":"实际方法落在已有适用条件内，对照没有发现需要新增或修订的步骤或边界。",
      "evidence_refs":["EX_A","EX_B"]
    }
  }'
```

充分对照且引用有效检查事实时可得到学习标签 0；没有增量也可用 `no_increment`。需要补证据时加 `--relation '{"entry_id":"实际编号","operation":"add_evidence","evidence":["EX_A"]}'`，与上述 review 一起结清原 J。材料不足、仅未召回旧规则或只改写措辞不能自动充当充分对照。

### 证据不足

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"insufficient_material",
    "conclusion":"已审阅实际输入与执行记录，但缺少约束覆盖检查，无法核对端点回退是否保持原题可行性；本轮不新增规则。"
  }'
```

该命令完成本轮审阅并保留回执，学习标签为未知。不能把未知写成无增量。后续有效补检、更正、引用失效按现有证据规则进入之后的反馈；已经发出的预测仍保留当时输入。


## References

- [world_model_contract.md](references/world_model_contract.md): versioned G/C/R/H+ contracts and historical reading.
- [commands.md](references/commands.md): result keys, explicit configuration, inspection and recovery commands.
- [induction.md](references/induction.md), [hplus.md](references/hplus.md): review receipts, immutable versions, existing-entry maintenance and evidence corrections.
- [prediction_context.md](references/prediction_context.md): frozen input, provenance and feedback budgets.
- [snapshot_lifecycle.md](references/snapshot_lifecycle.md): historical views, cleanup and existing freeze operations.
- [end_to_end.py](references/examples/end_to_end.py): runnable workflow with a substitute predictor.
