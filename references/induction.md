Normal solving: `profile / recall → plan-next → execute --prediction → check-task → record → close-episode → consume frozen H+`. A standalone forecast or manual choice does not authorize execution.

# Agent-led induction and existing-entry management

The agent chooses the method, comparison purpose, evidence scope and publication.
The framework keeps execution facts, immutable rule versions, references and
retry state. G/C/R still selects solving strategies; the frozen H+ only decides
whether to spend a new induction round.

## Close and consume H+

Use `plan-next` → execute --prediction → check → record → close. A real bound H+=1
must complete one review, including after failure or missing checks. H+=0 skips
new induction but keeps execution evidence and ordinary existing-rule use.
A prediction never executed remains unobserved. Missing/invalid H+ schedules
0 with its separate diagnostic; historical three-state records keep their meaning.

```bash
orx close-episode --task T --episode E
orx induction-status --task T --episode E
orx induction-material --induction-id J
orx induce --induction-id J --review '{"outcome":"insufficient_material","conclusion":"Actual facts reviewed; support is insufficient."}' --note "Reviewed available evidence."
```

`J` is returned in `induction_schedule.pending_ids`. The outer building agent
consumes each pending job with a short submission or reasoned empty review.
`process_induction_jobs(reviewer, ...)` provides the same callback handoff.
Close-out never invents a rule or calls a second hidden model. A resumed runner
reuses the job, actual induction action and final receipt. Omitted episode
resolves only one unique actual round; ambiguity returns readable existing
execution choices (`close-episode --task T --execution EX`).

An empty review is submitted without `--relation`, with a reason or structured
review receipt. Placeholder claims such as "no new knowledge" are not reusable
rules and are rejected rather than stored as method bodies.

## Query current material

```bash
orx induction-material --induction-id J --purpose modelling
orx induction-material --induction-id J --purpose implementation --query "endpoint direction and fallback" --related-top-k 3
orx induction-material --induction-id J --executions ex_A,ex_B --limit 2
orx induction-material --induction-id J --cursor 'CURSOR_FROM_BUDGET'
```

Every call reads current evidence. There is no query-key cache or saved full
response to replay. The primary case is J's actual triggering execution;
newer same-task attempts remain selectable controls. In a formal job, the
anchor stays first while `limit`/cursor paginate controls, so an older anchor
cannot hide newer attempts. Manual batches retain ordinary newest-first paging.

The default `induction-material/2` pack contains actual methods and their sources,
plans separately, original checks/unchecked scope, costs and provenance, failures,
related cases and short existing rules. Source links and exact original-input
matches are advisory overlap information. The agent decides what they establish;
text-number heuristics, file paths, family labels and mother-task fields do not
rule on transfer or publication. Repeated references to an execution are not new
physical uses. Similarity only finds candidates.

The 32000-character default counts the entire result and CLI envelope, controlled
by `OR_HARNESS_INDUCTION_MATERIAL_CHARS`. Oversized content is a whole-value
reference. `budget.next_cursor` reports omitted controls. Use:

When controls remain, their next row or valid short detail reference gets space
before old rules and optional neighbors. The cursor advances only over delivered
controls, including references; the actual triggering execution stays first.

```bash
orx inspect --bank experience --execution ex_A
orx inspect --bank strategic --entry 3 --rule-version candidate
orx inspect --bank values --value 'rule_fact:...'
```

Pending jobs temporarily hold their trigger and selected/delivered necessary
cases through normal window eviction and restart. Sessions retain IDs and fact
references rather than full packs. Completion releases full temporary execution
copies; minimal original facts/checker references remain for audit. Ordinary
`check-task` does not freeze a second full record. Other raw executions continue
through the normal evidence window. Historical saved packs remain readable.

## Submit a short rule and choose publication

```bash
orx induce --induction-id J --relation '{"when":"For fixed Y, feasible integer X forms an interval and the objective is monotone in X.","do":"Enumerate Y, derive both X endpoints and select by coefficient sign and objective direction. Disable this reduction for nonmonotone objectives and use full enumeration.","expect":"Preserve the stated integer optimum and constraints.","evidence":["ex_A","ex_B","ex_constructed"],"publication":{"state":"published","reason":"Agent checked the endpoint derivation, enumerations and nonmonotone fallback.","scope":["stated integer-interval mechanism and listed checks"]}}'
```

`publication` is one small author choice: `state:"published"` or `"candidate"`,
a short `reason`, and optional declared `scope`. Omission leaves a new body as
candidate; it does not trigger a framework content judgment. Publishing offers
that exact version in normal retrieval immediately. Actual verification scope
is reported beside the author's judgment. A solver status, a numerical reference
comparison and a method proof cover different claims; the report keeps them apart.

Derivation, independent benchmarks, exhaustive constructed cases and counterexamples
can support their stated conclusions. The agent can publish an endpoint mechanism
from A/B plus derivation and a constructed boundary without waiting for a third
real task. A later real C can then be recorded as transfer. One unchecked
observation can remain candidate. Transfer/boundary/regression are optional
review purposes, not a uniform publication checklist.

| Operation | Behavior |
|---|---|
| New paragraphs and evidence | New candidate body; explicit publication may offer it now |
| `entry_id`, `operation:"revise"`, paragraphs | New immutable version under the same ID; the active version remains until chosen otherwise |
| `entry_id`, `operation:"add_evidence"` | Reuse the body/version; append evidence or optional validation |
| `entry_id`, `operation:"merge"`, `merge_entry_ids`, reason | Agent selects and confirms the target; confirmed sources leave normal recall when the target is offered |

Safe formatting equivalence reuses an existing body. More complex equivalence is
an agent judgment. There is no automatic list of old successes that must be
re-run. Review a few discriminating cases, re-run where needed, or explain why
recorded derivation/facts suffice. Original checker versions remain traceable.
A reference-only `check-task --check '{"reference_objective":2699}'` establishes
that numerical comparison, with its declared source; it does not fabricate
constraint coverage. Missing acceptance is unknown.

Check that each declared assertion matches the claim it supports. A failed
assertion is not automatically a refutation of the whole rule. Record what the
check actually establishes; the agent decides attribution and utility management.
Existing assertion kinds such as `probe`, `status`, `comparison`, `code_changed`
and `code_unchanged` retain their declared scope; a changed code hash is not proof
that the method improved or that the original task was accepted.

## Existing-entry validation and utility need no positive job

At execution declare the actual adopted version with `--used-entry-ids 3@2`.
Omitted adoption is unknown; `''` declares none. The frozen reference preserves
that exact version. Historical v1 facts may support an agent's v2 compatibility
review, but only an actual v2 adoption generates a v2 use event.

Normal check/record/close saves use and scope. Optional `rule_assessments` can
record the agent's verdict and purpose:

```json
{"entry_id":"3","version":2,"purpose":"transfer","verdict":"supported",
 "source":"agent inspected the actual script, original inputs and declared checks",
 "assessment":"The chosen endpoint matches the objective direction; only the listed checks are claimed.",
 "evidence_refs":["artifacts.submitted_script","artifacts.workspace_result"]}
```

Existing-body evidence addition, validation or publication selection can use
`induce --relation` without a new job, provided it selects an existing entry and
contains no new/changed body. The existing `--attribute-effect` is the utility
management path:

```bash
orx induce --attribute-effect '{"entry_id":"3","execution_id":"ex_USED","verdict":"helped","lifecycle_action":"keep","note":"Agent reviewed the actual adopted use and declared acceptance."}'
orx retire --entry 3 --reason 'Agent concludes this method should leave normal recall.'
```

`promote`, `demote`, `keep` use the same lifecycle for legacy and versioned
entries. Management names real adopted evidence and a short reason. The version
comes from frozen adoption; multiple versions require choosing an existing one.
Management of an old version does not silently demote a different current version.
When `execution_id` is supplied, its frozen adoption determines the version;
auxiliary `evidence_execution_ids` cannot override it. Auxiliary-only evidence
retains the existing unique-version/explicit-selection behavior. A candidate
merged target leaves its sources usable until that exact target version is
published. Confirmed merged sources retain history but cannot be promoted back
into ordinary recall; management names their selected merge target explicitly.
Pure maintenance returns `maintenance_only:true`, state differences and index
results without a new induction action, positive job or empty review receipt.
New/changed bodies still need authorized induction. Final retirement uses the
existing cold archive.
Frozen libraries reject maintenance writes, publication and effect events too;
solving, recorded use and task checking continue through their existing paths.

## Cost and corrections are factual inputs

The agent selects a cost baseline and decides comparability and attribution.
Optional `comparator_execution_id` and `cost_dimensions` on assessment/validation
return both actual methods, measurement scopes, execution conditions, acceptance
rules, values and provenance. No baseline search, dimension substitution or
framework advantage verdict occurs. `cost_eligibility` distinguishes measured,
estimated, partial and unknown values. For 9→10 calls and 100→25 tokens, the
report gives +1 call and −75 tokens separately; it does not say calls decreased.

If an optional selected baseline expires, is excluded or does not exist,
ordinary task checking/close-out returns `cost_comparison.state:unavailable`
with the baseline ID and reason. Task checks, actual adoption and frozen H+
scheduling still finish. Explicit independent validation with an invalid
comparator reports a correctable input error. A readable comparison reports
`state:available` and both actual sources without choosing a winner.

Omitted optional assessments on unchanged facts preserve the prior assessment.
Explicit correction/withdrawal remains visible. `exclude-execution` and
`restore-execution` change evidence validity; current material omits excluded
facts and future H+ support reflects their validity. They do not trigger global
rule revalidation, automatic pause/recovery or publication. The agent evaluates
counterexamples and manages the entry. Valid failed executions remain useful
facts, not records to exclude merely for failure. Historical inputs, original
checks and exact adopted versions remain auditable.

For learning conclusions and the frozen switch see [hplus.md](hplus.md).
Runnable A/B publication and later C management:
`PYTHONPATH=src python docs/validation/run_review_eight_demo.py`.


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

