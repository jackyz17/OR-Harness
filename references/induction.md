# Offline induction and existing-entry management

Compare recorded executions to extract a method, test its scope, and submit the review. Use [SKILL.md](../SKILL.md) for solving and close-out.

## When to start induction

`plan-next` freezes `capability_gain.decision` as the JSON number `0` or `1` (`wm-hplus-binary/1`). G, C and R determine U. For H+ = 1, use `claim` to state what could improve or extend the prediction-time library, and `verification_conditions` to name a concrete fact to check. For 0, give the basis. The increment can be modelling knowledge as well: how the task statement is translated into decision variables, objective and constraints, together with the assumptions and applicability boundaries of that translation. A standard solving algorithm can still support 1 when this run and its review can test such an increment or boundary that the published strategy library does not cover.

### Predict H+ at cold start

Compare the proposed increment with published strategy rules that apply to the candidate in the frozen prediction input. Similar executions, matching structural cells and structural recommendations help find material and checks; they do not establish rule coverage by themselves. The **when** paragraph is the ONE semantic applicability statement: the framework checks only deterministic structural conditions (`predicates`, optional; empty means no machine-checkable condition, never unconditional applicability) and never decides whether `when` holds. A different `family` label alone is not a contradiction — a shared mathematical mechanism may transfer. When a recommendation contains a published rule, read its `when`/`do`/`expect` and decide applicability; a retrieval hit is not adoption. For an already-covered 0, cite its visible ID, version and applicable conditions in `basis`.

An empty library or missing historical reviews does not prevent a hypothesis grounded in the task, candidate steps and evidence obtainable during this run. Even a known method can add a reusable step or boundary to this library. For example, take a small finite task whose full enumeration fits this run's budget. With no published applicable rule, the candidate enumerates Y and selects the lower endpoint of each finite integer X interval because the minimization objective increases with X. Its `capability_gain` can be:

```json
{
  "decision": 1,
  "claim": "For fixed Y and a minimization objective increasing in X, selecting the lower feasible integer endpoint can replace full X enumeration.",
  "basis": ["The original task has finite integer bounds; the candidate uses lower-endpoint enumeration and the strategy-rule view is empty."],
  "verification_conditions": [{
    "condition": "Compare endpoint enumeration with full enumeration on this task's finite domain using the same constraints and objective.",
    "check_basis": "The original task inputs, candidate implementation and a captured full-enumeration check from this run."
  }]
}
```

For a covered control, suppose the frozen input actually shows published rule `12@3`, including its finite-interval premises, endpoint choice by objective direction and fallback for nonmonotone objectives. If the candidate repeats that method under those conditions, return 0 and omit `claim`:

```json
{
  "decision": 0,
  "basis": ["Published rule 12@3 already covers finite integer X intervals at fixed Y, lower-endpoint selection for this increasing minimization objective and full-enumeration fallback when monotonicity fails; this candidate adds no step or boundary."]
}
```

Use only rule IDs and versions actually present in the input; the control ID above is illustrative. If no concrete reusable increment with a check feasible during this run can be identified, return 0 with that specific reason and omit `claim`. Missing review history alone is not that reason. A 1 schedules a factual review; the review decides whether anything was learned and whether a rule can be published.

When a condition is checked, add an item to the `hplus_verification` list in the `check-task --check` JSON. Supply the original `prediction_id`, the zero-based `condition_index`, `state:"checked"`, a `source`, and locatable `evidence_refs` such as `artifacts.submitted_script` and `task_check.checks`. Use `finding` to record what the check showed. An omitted state or missing support leaves the condition unknown.

During H+ building, close-out schedules one required review per bound executed positive prediction, including failed attempts and incomplete verification. Missing or invalid decisions schedule as 0 with their own diagnostics. Executed H+ = 0 attempts remain available as comparisons.

## Take a required review

Read each J from `close-episode` under `result.induction_schedule.pending_ids`. Use `induction-status` to resume interrupted work. Settle the returned jobs before continuing the building stream.

```bash
orx induction-status --task T --episode EP
orx induction-material --induction-id J
```

## Minimal submissions

Compare cases, form a supported conclusion, then submit it with the same J.
J comes from `close-episode` under `result.induction_schedule.pending_ids`.
Replace EX below with an actual `execution_id` from the material you read;
evidence references must resolve in this memory. Use the short rule only when
the execution and your derivation support its stated conditions.

```bash
orx induce --induction-id J \
  --relation '{
    "when":"Y is finitely enumerable; at fixed Y, feasible integer X forms a finite interval and the objective is monotone in X.",
    "do":"Skip empty intervals. Choose the lower or upper X endpoint according to objective direction and coefficient sign, then compare all Y. Enumerate X fully when the objective is not monotone.",
    "expect":"Preserve the optimum under these conditions while reducing X enumeration.",
    "evidence":["EX"],
    "publication":{"state":"published","reason":"The recorded enumeration and the endpoint derivation support this monotone finite-interval scope."}
  }' \
  --review '{
    "outcome":"new_rule",
    "conclusion":"Endpoint choice by objective direction is reusable within the stated monotonicity boundary.",
    "comparison":{
      "sufficient":true,"substantive":true,"basis":"derivation",
      "source":"The recorded task inputs, executed enumeration and endpoint derivation.",
      "assessment":"For each feasible fixed Y, monotonicity puts the best X at the selected endpoint; comparing all Y preserves the optimum.",
      "evidence_refs":["EX"]
    }
  }'
```

One rule can use one `review.comparison`; there is no need to repeat it as a
`mechanism_comparison` inside the relation. A relation may supply its own
`mechanism_comparison` when its learning assessment differs. Choose `basis`
from the evidence actually used: `derivation` for an argument tied to real
execution inputs, `recorded_checks` for sourced checker results, or
`artifact_inspection` for inspected captured artifacts. Publication is the
agent's scoped usage decision; the learning label depends on the substantive
increment and its supported comparison.

When material is insufficient, settle J with only a short review:

```bash
orx induce --induction-id J \
  --review '{"outcome":"insufficient_material","conclusion":"The captured run does not establish the proposed boundary; a comparable checked case is missing."}'
```

This writes a receipt without a rule; its learning label remains unknown.
It requires neither `comparison` nor `--note`.

Optionally add `--dry-run` to either submission command when unsure about its
input. It checks the review fields, evidence references, optional per-rule
comparison, and publication choice without writing rules, receipts, events or
index entries. Fix a reported field or use a real reference from the material,
then remove `--dry-run` to submit with the same J. A passing preview leaves J
pending. `would_create`, `would_update` and `would_complete` describe expected
work; `saved` and `published` count actual writes and remain zero in a preview.
Insufficient learning support is reported separately from invalid input and
does not prevent a factual review from completing. After submission, read
`result.induction_receipt` and use `induction-status` to confirm completion.

## Read material and choose comparisons

Read the default pack, then follow the question raised by the evidence. `--purpose` sets the search focus for that read. You can change the focus, request known execution IDs, or page through attempts while keeping the same trigger and J.

```bash
orx induction-material --induction-id J --purpose modelling --query "MECHANISM_KEYWORDS"
orx induction-material --induction-id J --purpose implementation --query "ERROR_OR_API_KEYWORDS"
orx induction-material --induction-id J --executions EX1,EX2
orx induction-material --induction-id J --cursor CURSOR_FROM_BUDGET
```

| Field | How to use it |
|---|---|
| `material[]` | The trigger first, then current-task controls. Read `method`, `method_source`, outcomes, checks, and measured costs. |
| `related_history.executions[]` | Candidate comparisons from history, including failures. Compare the original constraints, actual methods, and check coverage. |
| `existing_knowledge` | Short views of relevant rules in the current library during ordinary H+ building. Use them to find possible revisions or duplicates. |
| `comparison_table` | An index of `execution_id`, `relationship`, and `check`. Read the case itself for its method. |
| `induction_gate` | The frozen hypothesis, verification conditions, `source_fact_ref`, and prediction-time `baseline_snapshot_ref`. |
| `budget.next_cursor` | The next page of current-task controls. |

Use `--purpose modelling` for model structure or `--purpose implementation` for code and repairs. Compare retries to diagnose a failure and independent cases to test how the method travels. Pending reviews retain delivered execution facts. The default response budget is 32,000 characters (`OR_HARNESS_INDUCTION_MATERIAL_CHARS`). Follow `inspect_hint` to open a compact or reference-only case.

```bash
orx inspect --bank experience --execution EX1
orx inspect --bank strategic --entry ENTRY --rule-version candidate
orx inspect --bank values --value VALUE_REF
```

After answering, cite available reference answers, checker results, or user feedback with their source and coverage. Use original-constraint checks or a derivation to support claims about feasibility and why a step transfers.

## Complete review submissions

### Generalize the method from cases

Start from the frozen H+ hypothesis and what the execution showed. Identify a specific decision in the model or implementation and the effect it might have on feasibility, objective value, runtime, or reliability. Choose the next comparison by what would clarify that claim.

Let the evidence determine what to record. A review can yield a modeling or solving method, an implementation technique, or a rule that combines steps that work together. Submit separate rules for independently useful findings with their own evidence. When the library already covers the method or evidence is insufficient, complete the review with that conclusion.

- **Find what changed the result.** A successful run gives a method worth inspecting. A failure and its repair on the same input can identify the step responsible. Compare what the model expressed, what the code actually did, and what the checks observed.
- **Align the roles across problems.** Look for another task that poses the same decision under different domain details. Map the decision variables, the constraints that connect them, and the quantity or guarantee they affect. Keep the relationship that the cases share and express case-specific capacities or coefficients as parameters.
- **Explain why the step works.** Derive the effect from the original constraints and objective. Keep the assumptions the argument uses and phrase them as facts the next agent can check. A derivation can support a scoped rule when only one case is available.
- **Try to break the claim.** Use a similar failed execution, an independent task, or a small constructed case that changes a condition relevant to the explanation. For an exact reformulation, check which feasible decisions and objective values it preserves. For an empirical speed or quality claim, compare measured outcomes under comparable conditions. If the claim fails, narrow its scope or add a fallback.
- **Record the reusable decision.** Put the applicability conditions and any relevant failure boundary in **when**. Put the method and the checks needed to apply it in **do**. State the guarantee or measured effect the evidence supports in **expect**. Check the relevant existing rule before choosing to add, revise, or supplement it. If the evidence leaves a decisive question open, record what would resolve it.

### Induce implementation methods

Use `--purpose implementation` when the reusable change concerns code or a solver interface, such as data parsing, index mapping, API calls, or result delivery. Compare the submitted scripts, error or protocol receipts, and actual results. A failed attempt and a repair on the same problem, model, and settings can isolate the step that mattered. Look for the same failure mechanism in another task or implementation. Generalize the reusable code step, the interface assumptions it depends on, and the check that detects failure.

Match validation to the claim. Use `execution_features.technical_outcome` and captured artifacts to check technical delivery. For a claim about the computed answer, check the original task. For a cost claim, compare measured values under matching conditions.

#### Implementation example · Deliver solver results

Use the computed objectives from the container and transshipment examples as inputs to a controlled script test. Change the delivery code between attempts.

| Case | Failure | Repair and observed result |
|---|---|---|
| Container production | The script prints 43,200 and exits without writing `result.json`. The executor records a missing-result protocol failure. | Write `{"status":"optimal","objective_value":43200}` in `result.json` in the execution workspace. The executor reads a valid result. |
| Transshipment | The script writes `{"status":"optimal","objective":685}`. The executor reports that the objective field is missing. | Use `objective_value` for the numeric objective. The executor reads a valid result with value 685. |

The shared result writer can be a short rule.

- **when** The solve script hands results to the executor through the workspace file.
- **do** Map the solver's actual status and available solution in one place. Write a fresh `result.json`. For `optimal` or `feasible`, include a finite numeric `objective_value`. When the solver establishes infeasibility, deliver `{"status":"infeasible"}` and any available diagnostics.
- **expect** The executor can read both the solution and infeasibility branches under the result protocol.

For this claim, use `effect_scope:"technical"` in the relation. For the learning comparison, use `basis:"artifact_inspection"` with the actual before/after executions and their captured artifacts.

### Modeling example · Fixed setup costs in SIRL IndustryOR

These teaching problems are IndustryOR cases 12, 44, and 77 from SIRL. They illustrate one route from recorded cases to a transferable rule. Read the problem statements, then compare the checked outcomes and the constraints that explain them.

#### Example 1 · Container production with substitution

A factory must meet its orders for six container sizes. Each size has its own production equipment. Producing any positive quantity of a size triggers a one-time setup fee of 1,200 yuan for that equipment.

| Size ID | Volume (cm³) | Orders (units) | Variable cost (yuan/unit) |
|---|---:|---:|---:|
| 1 | 1,500 | 500 | 5 |
| 2 | 2,500 | 550 | 8 |
| 3 | 4,000 | 700 | 10 |
| 4 | 6,000 | 900 | 12 |
| 5 | 9,000 | 400 | 16 |
| 6 | 12,000 | 300 | 18 |

One container can fill one order of the same or a smaller required volume. Choose the production quantities and their allocation to orders. Meet all orders and minimize variable production costs plus setup fees.

#### Example 2 · Transshipment through stations with fixed fees

Send material from two origins to two destinations. Each shipment passes through exactly one station. Destination 1 requires 80 units and destination 2 requires 120. The tables give supply limits, station capacity, and the unit shipping costs on both legs.

| Origin | Available supply | To station 1 | To station 2 |
|---|---:|---:|---:|
| 1 | 100 | 2 | 3 |
| 2 | 150 | 4 | 1 |

| Station | Throughput capacity | One-time fee if used | To destination 1 | To destination 2 |
|---|---:|---:|---:|---:|
| 1 | 100 | 10 | 3 | 2 |
| 2 | 100 | 15 | 1 | 4 |

Choose route flows that meet demand, conserve flow at each station, and respect supply and capacity. Minimize the costs of both shipping legs plus the fees of used stations.

#### Example 3 · Part assignment with mandatory machines

Process one part of each type from 1 to 10. Assign each part to one machine. Using a machine incurs its setup fee once. The per-part processing costs and setup fees are in yuan.

| Machine | Setup fee | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 100 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| B | 135 | 15 | 25 | 35 | 45 | 55 | 65 | 75 | 85 | 95 | 105 |
| C | 200 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |

Exactly one of parts 1 and 2 must use A. The other uses B or C. Parts 3, 4 and 5 are assigned to A, B and C respectively. Machine C can handle at most three parts. Formulate an integer model that minimizes total processing and setup costs.

#### Compare the cases and test the boundary

| Evidence | Inference |
|---|---|
| **Containers.** Selecting a plan by variable cost alone, then scoring it with all fees, costs 43,700 yuan. Joint optimization costs 43,200 yuan by producing size 3 for size 2 orders and size 6 for size 5 orders. | The choice is whether to pay a setup fee to avoid more expensive substitution. Link the setup flag to all production of that size. Bound production by the total demand the size can serve. |
| **Transshipment.** The optimum costs 685. Each station carries 100 units. In every feasible plan, demand of 200 fills the two capacities of 100. | The quantity controlled by the flag is total station throughput. Use physical capacity for its bound and keep inbound equal to outbound. The network constraints already force both stations open. |
| **Part assignment.** The optimum costs 1,005 yuan, of which 435 is setup. The three mandatory assignments force all machines to be used in every feasible plan. | A different constraint proves the same simplification. Fix the three flags to one and carry the fees as constants. Keep the part assignment and machine limits. |

The common model uses total nonnegative use `q_j`, a valid upper bound `U_j`, a binary flag `y_j`, and a one-time positive fee `F_j`. The link `q_j <= U_j*y_j` makes any positive use require activation, and `F_j*y_j` charges once. With that fee and no other need to activate an unused resource, cost minimization permits zero activation at zero use. Fixing a flag to one is justified when the original constraints force positive use in every feasible plan.

Probe that last premise by raising station 2's capacity from 100 to 200 in the transshipment example. Route 50 units from origin 1 and 150 from origin 2 through station 2, then send 80 and 120 to the destinations. This satisfies the modified constraints with station 1 closed, so the original proof for fixing both flags no longer applies. The bounded-use linkage still applies.

For the submission below, `EX_CONTAINERS`, `EX_TRANSFER`, and `EX_PARTS` stand for real `result.execution_id` values from running and checking those problems in the same memory home. J is the review ID from the positive trigger. Choose the publication scope from the comparisons and derivation you can support.

### New or substantively revised method

This submission uses the fixed-setup-cost modeling example.

```bash
orx induce --induction-id J \
  --relation '{
    "when":"In a cost minimization model, a resource incurs a positive one-time fee when used. Its total use is bounded and nonnegative, and the setup flag governs that use and fee.",
    "do":"Use a binary activation variable y_j for each optional resource j. Charge F_j*y_j once and constrain its total use q_j by q_j <= U_j*y_j. Set U_j from capacity or reachable demand. Check total use, original flow balances, and assignment constraints. Fix y_j to one and keep F_j as a constant when the original constraints force positive use in every feasible plan.",
    "expect":"Choose resource use jointly with variable cost and account for each setup fee once. Fixing a mandatory activation preserves feasibility and total cost.",
    "evidence":["EX_CONTAINERS","EX_TRANSFER","EX_PARTS"],
    "publication":{
      "state":"published",
      "reason":"Substitution and station flow support the use-to-activation link. Capacity bounds and mandatory assignments show how to prove a flag can be fixed.",
      "scope":["Positive fixed charges on bounded resource use; forced activation only when implied by the original constraints"]
    }
  }' \
  --review '{
    "outcome":"new_rule",
    "conclusion":"Use a setup flag for each optional resource. Fix the flag only when the original constraints imply it.",
    "comparison":{
      "sufficient":true,
      "substantive":true,
      "basis":"derivation",
      "source":"Checked container allocations, station flows and part assignments, together with the activation constraints.",
      "assessment":"The container comparison isolates the effect of joint setup and variable costs. Transshipment maps total use to throughput. Capacity in transshipment and designated parts in assignment independently show how original constraints can force activation.",
      "evidence_refs":["EX_CONTAINERS","EX_TRANSFER","EX_PARTS"]
    }
  }'
```

`--relation` saves **when / do / expect**, evidence and publication choice. `--review` saves the comparison and the learning conclusion. Omit `publication` to leave a new body as a candidate.

To revise, add `"entry_id":"ENTRY","operation":"revise"` and use `outcome:"revised"`. The active version stays available while the revision is a candidate. To merge, set `operation:"merge"` with a target, `merge_entry_ids`, and `merge_reason`. Source rules stay available until the target version is published. Identical bodies reuse their version.

### Already covered or no reusable increment

Inspect `source_fact_ref` and `baseline_snapshot_ref` from `result.induction_gate`. For `already_covered`, identify the rule version in the prediction-time snapshot that covers the executed method and cite both references. If the available evidence leaves prior coverage unresolved, complete the review as `insufficient_material`.

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"already_covered",
    "conclusion":"The prediction-time rule already covers bounded activation and the mandatory-activation case.",
    "comparison":{
      "sufficient":true,
      "substantive":false,
      "source":"The checked execution and the rule version in the frozen snapshot.",
      "assessment":"The executed steps and assumptions match that version.",
      "evidence_refs":["SOURCE_FACT_REF","BASELINE_SNAPSHOT_REF"]
    }
  }'
```

Use `no_increment` for a supported review that finds no reusable addition. To attach evidence to an unchanged rule in the same J, add `--relation '{"entry_id":"ENTRY","operation":"add_evidence","evidence":["EX_TRANSFER"]}'`.

### Insufficient material

```bash
orx induce --induction-id J \
  --review '{"outcome":"insufficient_material","conclusion":"The objective is available, but the execution record lacks resource-level flows needed to check the activation constraints."}'
```

This completes the review with an unknown learning label. Use it when evidence is insufficient for either a rule or a no-increment conclusion.

### Read publication, completion and learning separately

| Result | Field |
|---|---|
| Version offered for use | `induce` → `result.relations[].publication.published` |
| Committed review, evidence and rule versions | `induce` → `result.induction_receipt` |
| Required jobs remaining | `induction-status` → `result.completed` and `result.pending_ids` |
| Prediction-time decisions and diagnostics | `induction-status` → `result.signal_counts` |
| Learning observations | `induction-status` → `result.observation_counts` |

A `wm-hplus-review/2` receipt supports label 1 for an actual new version with a substantive, sourced comparison. Label 0 requires a supported no-increment conclusion. Source your comparison with `source`, `assessment` and `evidence_refs`. Use recorded checks, `basis:"derivation"` for reasoning over actual inputs, or `basis:"artifact_inspection"` for captured artifacts. A refuted hypothesis can support a new boundary rule.

Read material for J before submitting. If `index_sync.state` is `deferred`, retry the identical submission to resume indexing with the saved receipt. A completed J accepts only the same submission on retry.

## Existing-entry use and utility

For an adopted rule, pass `execute --used-entry-ids ENTRY@VERSION`. An omitted
declaration leaves use unknown; pass `--used-entry-ids ''` to declare no
adoption. The version is resolved from the frozen `used_entries` declaration,
so you do not re-type it for every judgement.

Read the entry's UTILITY summary with `orx inspect --bank strategic --entry N`
(`result.utility`, and the same block inside `rule_history.utility`):

- `totals.adoptions` — distinct `(entry, version, execution_id)` adoptions;
- `totals.judged` — adoptions with a CURRENT VALID effect judgement;
- `totals.unjudged` — adoptions with no judgement at all;
- `totals.{helped,neutral,unrelated,refuting}` — your own latest judgements;
- `totals.invalidated` — judgements whose supporting execution was excluded/
  withdrawn (the adoption and the historical judgement stay recorded);
- `unattributed_judgments` — judgements that cannot be tied to an adopted
  version; they are never guessed onto the current version;
- `latest_management`, `status`, and per-version `evidence` references.

Three facts stay separate in `status_view`: publication (is it offered?),
verification (how far was the CLAIM checked?) and usage (how did actual use
turn out?). `validated`/`legacy_status` is the historical maturity label, not
a proof of usefulness; a passing check or a publication does not prove that
using the entry helps, and the forward track's `hit_rate`/`n_predictions`
belongs to `forward_calibration` — `n_predictions = 0` means "not estimated",
not a measured zero effect. A failed task check is not `refuting`; a technical
check failure is evidence, not a utility verdict.

At `close-episode`, `result.entry_usage.effect_review.pending` lists this
episode's adoptions that still lack a judgement. Record the review in the SAME
close-out (works for H+ = 0):

```bash
orx close-episode --task T --episode E --terminal completed \
  --attribute-effect '{"entry_id":"2","execution_id":"EX_USED","verdict":"helped","note":"The rule fixed the price unit before the objective was written; the recorded check then matched the reference scale."}'
```

A `refuting` judgement keeps its existing default (pause that version); a
confirmed in-scope error can be paused immediately, without waiting for a
fixed number of failures. Use `keep`/`demote`/`promote` for an explicit
management decision. Leave the judgement unevaluated when the evidence does
not support one — missing or low-sample use is NOT evidence of uselessness.
If several judgements are neutral/unrelated/refuting, or another rule or the
documented procedure now covers the case, consider revising or retiring the
entry — your decision, not an automatic threshold.

Add a sourced transfer, boundary or counterexample observation in `check-task` through `rule_assessments`. Each item identifies the `entry_id`, `version`, `purpose`, `verdict`, `source`, `assessment` and `evidence_refs`.

To add evidence, validate, or change publication for an existing body, select its entry and version with `induce --relation`. For utility judgments use the actual adoption execution.

```bash
orx induce --attribute-effect '{"entry_id":"ENTRY","execution_id":"EX_USED","verdict":"helped","lifecycle_action":"keep","note":"The adopted activation rule matched the recorded constraints and checks."}'
orx retire --entry ENTRY --reason 'The rule should leave ordinary recall under current operating conditions.'
```

| Action | Effect |
|---|---|
| `keep` | Records the judgment and keeps the selected version and status. |
| `demote` | Suspends the selected version and removes it from ordinary recall if active. |
| `promote` | Activates the selected version after identity and merge checks. |
| `retire` | Moves the entry to the cold archive. |

A management decision applies to the version the cited execution ACTUALLY
adopted: a problem found in v1 does not demote a corrected v2 that is now
active. A changed statistic never restores a paused version by itself —
restore is an explicit `promote` (or a new version with an explicit
publication decision). `retire` accepts `--evidence-execution-id` (repeatable)
so the cold-archive card keeps the stated reason, the cited facts and the
read-time usage summary; retire/restore maintain the recall index through the
same synchronization path.

`verdict:"refuting"` defaults to demotion when `lifecycle_action` is omitted. Specify `keep` to record the judgment while retaining the version. The primary `execution_id` identifies the frozen adopted version. Existing-body maintenance returns `maintenance_only:true`.

## Scoped checks, costs and corrections

Use task checks and derivations that cover the actual effect claimed. For cost comparisons, supply `comparator_execution_id` and optional `cost_dimensions`. Compare the observed settings, acceptance criteria, measured dimensions and their sources. A missing optional baseline is reported as `cost_comparison.state:unavailable`.

Keep failed attempts as contrasts. Use `exclude-execution` and `restore-execution` to correct evidence validity. Later checks can supply missing support. A learning label can return to unknown if its supporting facts are withdrawn.

## Learning feedback and experiment modes

Later `plan-next` calls in the same memory home select applicable receipts recorded before the decision cutoff. Read [prediction input audit](world_model_contract.md#4-inspect-a-prediction-and-its-input) to see which feedback reached the provider.

Ordinary prediction homes enable H+ by default. To set it explicitly, use global `--induction-mode hplus` or `OR_INDUCTION_MODE=hplus`. Settle required reviews before freezing the library. Use separate homes for [H+, NoH+ and random controls](../docs/validation/hplus_binary.md) and follow the [fixed-memory test procedure](snapshot_lifecycle.md#fixed-memory-testing-with-per-question-copies).
