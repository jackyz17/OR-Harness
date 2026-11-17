# Offline induction and existing-entry management

Use this guide to turn completed executions into reusable methods and manage existing rules. The agent compares cases and chooses the rule's publication scope. Follow [SKILL.md](../SKILL.md) to solve and close the task.

## When to start induction

`plan-next` freezes a candidate's H+ decision before execution. The same world-model response carries `capability_gain.decision` as the JSON number 0 or 1 (`wm-hplus-binary/1`). G/C/R determine U. With H+ = 1, use `claim` to name a reusable increment or boundary and `verification_conditions` to name a concrete fact that can be checked. With 0, explain the reason in `basis`. Judge the increment relative to the prediction-time library, including adaptations of existing methods.

Carry the frozen condition into the solve and collect actual method receipts, artifacts, and task checks. Optional `check-task.hplus_verification` records the original prediction ID, condition index, source, and evidence references. An unfinished check leaves the condition unknown.

When building the library, close-out schedules one review per bound executed positive prediction and episode, including failed attempts and incomplete verification. Executed H+ = 0 attempts remain available as evidence for other reviews. Unexecuted candidates remain unobserved. Missing and invalid signals schedule 0 with distinct diagnostics. A valid positive still requires review when its explanatory fields are incomplete.

## Take a required review

`close-episode` returns each required review's ID J in `result.induction_schedule.pending_ids`. Use J when reading material and submitting the review.

```bash
orx induction-status --task T --episode EP
orx induction-material --induction-id J
```

Recover pending IDs with `induction-status` after interruption. Repeated close returns the still-pending IDs without creating new jobs. Python building runners can use `process_induction_jobs(reviewer, ...)`. Finish required reviews before starting the next building task.

## Read material and choose comparisons

Start with the default material pack, then narrow or page only when needed:

```bash
orx induction-material --induction-id J --purpose implementation --query "endpoint direction and fallback" --related-top-k 3
orx induction-material --induction-id J --executions EX_A,EX_B --limit 2
orx induction-material --induction-id J --cursor CURSOR_FROM_BUDGET
```

| Result field | Read it for |
|---|---|
| `material[]` | The fixed triggering execution first, followed by delivered current-task controls; actual `method`, its source, outcome, task checks, measured costs and references |
| `related_history.executions[]` | Candidate cross-task/same-source contrasts, including failures; discovery similarity does not certify comparability or transfer |
| `existing_knowledge` | Relevant short rule/version views; compare their conditions and steps with the proposed increment |
| `comparison_table` | Execution IDs, relationship labels, actual-method references and check state; this is a factual index, not a framework verdict on generalization |
| `induction_gate` | Frozen trigger, hypothesis, verification, `source_fact_ref` and `baseline_snapshot_ref` |
| `budget.next_cursor` | Controls still to read; a bounded page is not the entire evidence bank |

Every pending material call queries current evidence. Purpose/query/case IDs may change while the trigger and J remain fixed. The default whole result/CLI envelope is bounded at 32,000 characters (`OR_HARNESS_INDUCTION_MATERIAL_CHARS`). A large value may become a whole-value reference; follow its inspect hint. Source overlap and original-input identity are advisory. The agent decides which methods, assumptions, checks and measurement conditions can be compared.

For closed cases, use available reference answers, checker results, and user feedback as post-answer evidence. Identify their source and the part of the outcome they establish. Pair a numerical comparison with the task checks or derivation needed to justify a broader claim.

```bash
orx inspect --bank experience --execution EX_A
orx inspect --bank strategic --entry ENTRY --rule-version candidate
orx inspect --bank values --value VALUE_REF
```

Pending jobs hold necessary execution facts across ordinary eviction/restart. Completed jobs release full temporary copies while retaining minimum audit references. Expired/unavailable evidence is reported; it is not silently reconstructed from today's library. A same-task retry can be a repair control without being an independent transfer task.

## Complete review submissions

Submit a new or revised rule, a supported conclusion that the library already covers the method, or an insufficient-material review according to the evidence.

### A/B setup and ID sources

Use actual recorded cases. The runnable [run_review_eight_demo.py](../docs/validation/run_review_eight_demo.py) supplies small integer enumerations and a scoped publication example. [unified_solving.py](examples/unified_solving.py) exercises the normal CLI handoff with one or more candidates.

| Case | Actual original problem and checked result | Role |
|---|---|---|
| A | Nonnegative integers X,Y; X+Y<=1000, X>=2Y+300; maximize 2X+5Y; optimum 2699 | Increasing objective with maximization: choose the upper feasible X endpoint |
| B | Nonnegative integers X,Y; X+Y<=641, X>=2Y+130; minimize 2X-5Y; optimum 90; always taking the upper endpoint gives 92 | Increasing objective with minimization: choose the lower endpoint |
| Constructed boundary | Nonnegative integers X,Y; X+Y<=12, X>=2Y; minimize (X-3)^2+Y; optimum 0 at X=3,Y=0 | Nonmonotonicity defeats endpoint-only minimization; full enumeration is the fallback |

EX_A, EX_B and EX_BOUNDARY below are `result.execution_id` values from those actual `execute` calls, then checked and recorded. J is the positive executed trigger's `close-episode` `result.induction_schedule.pending_ids` item. Retain the same memory home. A prior H+=0 execution can be a comparison for this positive job. A small boundary computation is useful support when actually run and referenced; it is not a compulsory checklist item. Do not paste these placeholder IDs as if they were recorded facts.

### New or substantively revised method

Read `induction-material --induction-id J`, inspect the actual evidence and derivation, then submit the reusable rule and learning conclusion together:

```bash
orx induce --induction-id J \
  --relation '{
    "when":"Y is finitely enumerable; at fixed Y, the nonempty feasible integer X set is a finite interval and the objective is monotone in X.",
    "do":"Skip Y values with no feasible X. For each remaining Y, choose the interval endpoint required by coefficient sign and objective direction, then compare objectives across Y. If the objective is nonmonotone in X, enumerate all feasible X instead.",
    "expect":"Preserve the original optimum and constraints within this scope while reducing enumeration over X.",
    "evidence":["EX_A","EX_B","EX_BOUNDARY"],
    "publication":{
      "state":"published",
      "reason":"The endpoint derivation and captured enumerations support the monotone scope and the tested nonmonotone fallback.",
      "scope":["Finite integer intervals with a fixed-Y monotone objective; full enumeration when that premise fails"]
    }
  }' \
  --review '{
    "outcome":"new_rule",
    "conclusion":"Objective direction selects the endpoint; nonmonotonicity requires the stated fallback.",
    "comparison":{
      "sufficient":true,
      "substantive":true,
      "basis":"derivation",
      "source":"Actual A/B inputs and captured integer enumerations, endpoint algebra, and the recorded small nonmonotone check.",
      "assessment":"A and B require opposite endpoints under the same fixed-Y increasing objective. The constructed case checks why endpoint-only minimization must be disabled outside monotonicity.",
      "evidence_refs":["EX_A","EX_B","EX_BOUNDARY"]
    }
  }'
```

`--relation` retains `when / do / expect`, evidence and the author's publication choice. `--review` retains the real comparison and knowledge increment for training. A substantive new/revised version with valid support can yield learning label 1. Publishing alone does not produce that label.

For an explicit revision, keep the existing number by adding `"entry_id":"ENTRY","operation":"revise"` to the relation and `"outcome":"revised"` to the review. The existing active version stays available while the revision is a candidate. `operation:"merge"` names the target, `merge_entry_ids`, and `merge_reason`. Source rules stay available until that target version is published. Exact repeats and safe formatting changes reuse the existing body and version.

### Already covered or no reusable increment

Compare the actual method/checks with the library that was available when the prediction was frozen. `SOURCE_FACT_REF` is `result.induction_gate.source_fact_ref`; `BASELINE_SNAPSHOT_REF` is `result.induction_gate.baseline_snapshot_ref` from material. Inspect these real references and the relevant rule versions before claiming coverage. Today's new rule or an empty recall alone cannot establish past coverage.

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"already_covered",
    "conclusion":"The checked execution uses endpoint direction and fallback conditions already present in the prediction-time rule version.",
    "comparison":{
      "sufficient":true,
      "substantive":false,
      "source":"Actual executed method and check report, compared with the available frozen rule version.",
      "assessment":"The recorded steps and applicable conditions add no reusable method or boundary beyond that version.",
      "evidence_refs":["SOURCE_FACT_REF","BASELINE_SNAPSHOT_REF"]
    }
  }'
```

A supported sufficient comparison yields learning label 0; `outcome:"no_increment"` is also accepted. For optional evidence addition in that same job, include `--relation '{"entry_id":"ENTRY","operation":"add_evidence","evidence":["EX_B"]}'` with this review. It retains the body/version and settles the same J. Rewording alone, unavailable prior rules or insufficient checks do not establish a negative learning label.

### Insufficient material

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"insufficient_material",
    "conclusion":"The actual method and outputs were reviewed, but constraint coverage is missing; the proposed endpoint preservation claim cannot be checked from the available evidence. No rule is added."
  }'
```

This finishes the required review with a receipt and an unknown learning observation. No placeholder rule is written. State the real gap; insufficient evidence is not a no-increment conclusion.

### Read publication, completion and learning separately

| Question | Field | Meaning |
|---|---|---|
| Is this submitted version offered for use? | `induce`: `result.relations[].publication.published` | The author's explicit publication choice for that version; omitted publication leaves a new body candidate |
| Did the actual review commit? | `induce`: `result.induction_receipt` | Original prediction, real induction action, exact rule versions, evidence and conclusion |
| Are required jobs settled? | `induction-status`: `result.completed` / `result.pending_ids` | No pending reviews; neither mathematical correctness nor a positive learning label is implied |
| What was learned with support? | `induction-status`: `result.observation_counts` | `positive`, `negative`, `unknown`, and `bound_attempts`; these are review observations, not counts of H+=0 predictions |

A `wm-hplus-review/2` receipt retains the original hypothesis, actual induction action, conclusion, supporting evidence, and exact rule versions. Its learning observation comes from the review's `outcome` and `comparison` or the rule's `mechanism_comparison`.

| Learning label | Support |
|---|---|
| `1` | A substantive new or revised rule body, a sourced conclusion, and valid supporting evidence |
| `0` | A sufficient, sourced comparison showing `already_covered` or `no_increment` relative to the prediction-time library |
| Unknown | Insufficient material, unavailable support, unexecuted work, or a review without a supported conclusion |

`result.signal_counts` tracks prediction-time decisions and diagnostics separately from these observations. Publication and the hypothesis verdict also have their own fields. A refuted hypothesis can yield a supported boundary rule. Use recorded checks by default, `basis:"derivation"` for a derivation over cited inputs, or `basis:"artifact_inspection"` for captured artifacts. Use resolving references such as `task_check.*` or `artifacts.*`. Existing cases with a reliable derivation or small executable checks can support a scoped cold-start rule. Record the scope of later transfer, boundary, and regression checks separately.

The relation's `induction_id` or material session associates it with J, including for `add_evidence`. Ownership is checked before writing. Rules, action links, receipt references, and pending index work commit together. For `index_sync.state:deferred`, repeat the identical submission to recover the same action and receipt and retry indexing. A different submission against a completed job is a conflict.

## Existing-entry use and utility

Explicitly adopt a version you have read with `execute --used-entry-ids ENTRY@VERSION`. Structural recall exposes `recommendations[].knowledge.rule`; a semantic hit may omit version details. Resolve it with `orx inspect --bank strategic --entry ENTRY`, reading `result.entries[0].rule.view_version`. Omitted adoption is unknown; `--used-entry-ids ''` means adopted none. A citation, recalled item, historical compatibility review and actual version adoption are separate facts.

Normal check/record/close retains use. Add an optional item in `check-task`'s `rule_assessments` only when making a sourced transfer/boundary/counterexample judgment:

```json
{
  "entry_id":"ENTRY", "version":2, "purpose":"transfer", "verdict":"supported",
  "source":"Agent inspected the actual adopted script, original input and declared checks.",
  "assessment":"The endpoint follows the objective direction within the listed interval conditions; only the declared acceptance coverage is claimed.",
  "evidence_refs":["artifacts.submitted_script","artifacts.workspace_result"]
}
```

Existing-body evidence addition, validation and publication selection can use `induce --relation` without a new positive job if they select an existing entry and introduce no new body. Utility/status judgments use the existing separate interface:

```bash
orx induce --attribute-effect '{"entry_id":"ENTRY","execution_id":"EX_USED","verdict":"helped","lifecycle_action":"keep","note":"Actual adopted use meets its declared checks; no comparative cost advantage is claimed."}'
orx retire --entry ENTRY --reason 'The agent concludes this method should leave ordinary recall.'
```

| Action | Current behavior |
|---|---|
| `keep` | Leaves the selected version/status unchanged; records the reasoned use judgment |
| `demote` | Suspends the selected version; if it is the active adopted version, the entry becomes paused/suspect and leaves ordinary recall |
| `promote` | Explicitly activates the selected version, subject to existing identity/merge checks |
| `retire` | Separate command moving the entry to the cold archive |
| `verdict:"refuting"` with no `lifecycle_action` | The current implementation defaults to demotion of the adopted version; it is not annotation-only |

Use a real use reference and reason. A primary `execution_id` fixes the version from frozen adoption; auxiliary evidence cannot substitute another version. Managing an older version does not silently demote a different current version. A `rule_assessments` observation is separate from utility-effect management. Pure maintenance returns `maintenance_only:true` and changes/index results without a new induction action or empty receipt. Frozen libraries reject these writes too.

## Scoped checks, costs and corrections

A solver status, a reference objective comparison and a method derivation answer different questions. Objective agreement alone does not prove original-constraint coverage. Legacy assertions (`probe`, `status`, `comparison`, `code_changed`, `code_unchanged`) retain their scopes; a code hash change is not proof of improvement. An assertion failure is not automatically a refutation of an entire method.

The agent chooses cost baselines, comparability and causal interpretation. Optional `comparator_execution_id` / `cost_dimensions` return actual methods, conditions, acceptance, values and provenance, without searching for a baseline or declaring a winner. Measured, estimated, partial and unknown costs remain distinct. An expired optional baseline returns `cost_comparison.state:unavailable` and its reason without blocking ordinary close-out; explicit validation with an invalid comparator is a correctable input error.

`exclude-execution` / `restore-execution` change evidence validity and which facts support a later observation. Retain failed attempts as contrast. A late check can supply missing support, and withdrawn or excluded sole support makes the observation unknown. Original checks, versions, and issued prediction inputs remain auditable. Rule publication and retirement are explicit author decisions.

## Learning feedback and experiment modes

The next `plan-next` in the same memory home selects applicable settled receipts from before its decision cutoff. See [prediction input audit](prediction_context.md#audit-what-the-next-provider-actually-received) to trace which feedback reached the provider.

New ordinary prediction homes use H+ automatically. Use global `--induction-mode hplus` or `OR_INDUCTION_MODE=hplus` to set the building mode explicitly. H+, NoH+, and random controls use separate homes. NoH+ opportunity reviews use the prediction-time library. Drain required jobs before freezing the library. For execution isolation in fixed-memory tests, follow [the per-question copy procedure](snapshot_lifecycle.md#fixed-memory-testing-with-per-question-copies).

Historical capability and effect records retain their original contracts. `evaluate-capability` measures later performance through its separate protocol. The review labels above measure a supported learning increment from the actual induction.
