Normal solving: `profile / recall → plan-next → execute --prediction → check-task → record → close-episode → consume frozen H+`. A standalone forecast or manual choice does not authorize execution.

# Frozen H+ and evidence-backed learning receipts

H+ is the pre-execution binary investment switch in each candidate's SAME world-model answer inside `plan-next`. G/C/R still selects the solving strategy. Use `capability_gain.decision`
as the JSON number 0 or 1 (`wm-hplus-binary/1`). With 1, provide a short learning
hypothesis in `claim` and a concrete `verification_conditions` check; with 0,
state a reason in `basis`. Missing/illegal decisions schedule 0 with distinct
diagnostics. Old three-state predictions retain their original protocol.

The question is what reusable method, applicability boundary or verifiable
effect this execution and one review can add relative to the current library,
which fact can be checked, and which future step would change. Empty recall
alone does not establish novelty. A valid 1 with missing detail still requires
one honest review; actual evidence decides the learning receipt.

## The normal handoff

```bash
orx plan-next --task task.json --episode E --candidates candidates.json
# P = result.selected_prediction_id; even one candidate is scored.
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
orx check-task EX --check '{"reference_objective":90,"reference_source":"bench_declared"}'
orx record --from-staged EX
orx close-episode --task T --episode E
orx induction-material --induction-id J
orx induce --induction-id J --relation '{"when":"...","do":"...","expect":"...","evidence":["EX"],"publication":{"state":"published","reason":"Agent checked the stated mechanism.","scope":["listed evidence only"]}}'
```

`J` comes from `induction_schedule.pending_ids`. The outer building agent must
settle each positive executed job with a short submission or a reasoned empty
review. Failure and insufficient material do not cancel 1. H+=0 stores the
execution and existing-entry use, skips new induction, and may still supply
another review's contrast or utility evidence. Unexecuted predictions remain
unobserved. `process_induction_jobs(reviewer, ...)` runs the same callback handoff;
the framework does not author a rule or make an extra hidden model call.

Give the solving agent the chosen frozen verification condition. Actual method,
artifacts and `check-task` provide the evidence. Optional `hplus_verification`
records its completion with the original prediction ID, condition index, source
and actual references. Absence stays unknown and does not cancel the review.

Every material call is a current query. Purpose/query/case IDs/cursor may change;
the triggering execution stays fixed while controls paginate independently.
Jobs keep trigger/ownership, necessary evidence/session references and final
submission/receipt, not cached full packs. Full pending evidence survives normal
window eviction/restart, and is released after completion while minimum audit
facts remain. Old packs/receipts stay readable. The runner passes the existing
job/material references automatically; ambiguous cases choose an existing object.
A shared historical evidence ID cannot dispatch an old identical submission to
a new job: an exact retry returns its original receipt. Explicit conflicts name
the differing payload fields and the receipt to inspect.

Receipt comparison references also retain minimal facts, even for historical
executions absent from material pages, rule evidence and other jobs. Both
`review.comparison.evidence_refs` and rule `mechanism_comparison.evidence_refs`
are checked before completion. New missing references are correctable errors;
an unrecoverable old receipt becomes unknown individually, without blocking
other feedback or status queries. Ordinary execution rows remain window bounded.

A relation's existing `induction_id` or attached `material_id` identifies formal
review before the evidence-only maintenance branch. Even an `add_evidence`
submission then settles that job. Conflicting job/session ownership is refused
before writes. Without an induction association, existing-entry maintenance
keeps its lightweight path. Frozen test libraries reject either kind of write.

Repeated close reuses the actually bound prediction/action/episode identity.
Omitted episode auto-resolves only one real round; `--execution EX` selects an
existing run when needed. Two connections reread state inside transactions;
a late material write cannot reset a completed job. Before declaring building
complete, require `induction-status.completed:true`.

## Separate publication and learning

The induction agent chooses publication, candidate status, revision, merge or
only evidence addition. There is no transfer/boundary/regression prerequisite.
Derivation and small executable checks can support a scoped cold-start rule;
later independent tasks add transfer observations. See [induction.md](induction.md).

Learning receipts (`wm-hplus-review/2`) preserve original prediction/hypothesis,
actual induction action, conclusion/reason, valid fact references and exact rule
versions. The agent supplies `review.outcome` and `review.comparison`, or a
relation's `mechanism_comparison`. For example:

```json
{"outcome":"new_rule","hypothesis_outcome":"supported",
 "conclusion":"Endpoint direction and a disable boundary are supported in the stated scope.",
 "comparison":{"sufficient":true,"substantive":true,"basis":"derivation",
   "source":"inducer checked endpoint algebra and captured actual enumerations",
   "assessment":"At fixed Y, monotonicity and direction select the interval endpoint. A nonmonotone constructed case uses full enumeration.",
   "evidence_refs":["ex_A","ex_B","ex_constructed"]}}
```

- **1:** an actual new/revised body plus the agent's sourced, sufficient
  substantive-increment conclusion and valid support. Derivation or small
  runnable checks can confirm only their checked scope; a third real task and
  publication are not prerequisites.
- **0:** a sourced, supported `already_covered` or `no_increment` conclusion.
  Adding evidence may accompany this conclusion without changing the body.
- **Unknown:** insufficient support, missing actual review, unchecked required
  facts, or invalid/excluded sole support. An empty review can finish a job
  while leaving its learning observation unknown.

The default comparison basis uses existing checks; a late checker may make that
same recorded conclusion usable in a later observation. `basis:"derivation"`
names the agent's retained derivation over cited actual inputs;
`basis:"artifact_inspection"` uses actual captured artifacts. References must
exist. A context/library reference alone is not an executed result. Numerical
checks report their exact declared coverage; the framework does not certify
the agent's mathematical or causal judgment.

`artifacts.*` references must name a captured, readable artifact belonging to
the execution. `task_check.*` must resolve an actual report field or check item
(for example `task_check.checks.0`). The receipt freezes the observed check,
version and coverage supporting its conclusion. Repeating the same check keeps
support; withdrawing its result or scope makes later support unknown. A first
late check can supply support, with its correction history preserved. A real
failed check can support an agent's counterexample/boundary assessment. Existing
prediction inputs and observations at their original cutoff remain unchanged.

Rule versions, actions, reference validation, receipts and index repair intents
commit together. External vector writes follow commit. A refused receipt leaves
the previous rule and vector view intact. `committed:true` with
`index_sync.state:deferred` means the rule/receipt succeeded and indexing needs
retry; repeat the same submission or use `rebuild-index`. Retry returns the same
receipt/action and attempts the pending index repair.

Hash changes, no recalled old rule, current publication and later retirement do
not infer increments. Safe exact/formatting dedup is retained. Synonym rewriting
without supported review is unknown. Truth of the original hypothesis is separate
from the useful method/boundary actually learned; a refuted hypothesis can still
support new knowledge.

An empty negative review can use existing job references:

```bash
orx induce --induction-id J --note 'Agent compared the actual checked workflow.' --review '{"outcome":"already_covered","conclusion":"No reusable increment beyond the existing mechanism.","comparison":{"sufficient":true,"substantive":false,"source":"agent compared frozen rules and actual checks","assessment":"The performed mechanism is already covered in the stated scope.","evidence_refs":["SOURCE_FACT_REF_FROM_JOB","BASELINE_SNAPSHOT_REF_FROM_JOB"]}}'
```

## Later facts, utility and prediction feedback

Later corrections affect only later observations. `exclude-execution` removes
support from current material/feedback; restoring reads valid sources again.
The entry's published status remains the agent's management decision, and its
previously supported learning result is not revoked merely because the agent
retires the method for utility reasons. Original receipts/check versions and
already-issued prediction inputs remain traceable.

Existing-entry utility management is independent of new induction. H+=0 can
record use/check/effects and `promote`, `demote`, `keep`, or `retire` with real
adoption references and a reason. Cost baselines and causal benefit judgments
belong to the agent; raw per-dimension observations are provided separately.
No empty induction receipt is needed for pure maintenance.

Within the existing feedback budget, select related settled positive/negative
examples before unknowns. Send a short hypothesis, conclusion, effective fact
references/check identifiers, reason and exact versions. Long comparisons stay
behind receipt/value references. If a compact row still does not fit, try the
next candidate rather than ending selection. Feedback uses only observations
recorded before the new prediction's cutoff.

## Historical/control paths

Historical receipts and three-state capability records remain inspectable;
legacy `evaluate-capability` later-performance attribution has a separate
meaning from this induction conclusion. Historical full material packs can be
read for audit, but new pending calls always query current evidence.

Existing H+/NoH+/random build-and-freeze experiment commands remain available
in separate homes. This change does not redesign their frozen-library lifecycle,
experiment splits/missed-opportunity statistics, or `evaluate-capability`
cross-execution attribution (the deferred P1-9 through P1-11). Old experiment
field names are not evidence of independent verification under this new label
protocol. Use real attributable samples to assess discrimination; substitute
providers validate the chain.

Runnable normal-flow demonstration:

```bash
PYTHONPATH=src python docs/validation/run_review_eight_demo.py
```

It publishes from A/B plus derivation and a constructed boundary before real C,
then records C use and performs lightweight `keep`, with an actual next provider
request fragment in its JSON output.


## Complete review submissions

Use actual delivered J and execution IDs. A prediction H+=0 skips new induction; a learning label 0 follows a real supported review. Publication and learning labels remain separate.

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

