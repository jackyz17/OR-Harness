# Command reference

Follow [SKILL.md](../SKILL.md) for the solving workflow. This reference covers CLI syntax, required inputs and the results needed for the next action. Read the relevant detailed contract only when needed.

Commands emit one JSON line containing `result` and `summary`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. Select the memory directory with `--home DIR` (default `$OR_HARNESS_HOME` or `./or_harness_home`).

## 1. Common paths and command index

Routine solving: `profile` → `recall` → `plan-next` → `execute` → `check-task` → `record` → commit the answer → `close-episode`. Run required H+ reviews before the next task. Use the same [task input](modeling.md#start-from-the-shared-original-input) and episode across attempts.

| Need | Command |
|---|---|
| Prepare and retrieve | `profile`, `recall` |
| Select and execute | `plan-next`, `execute` |
| Verify and save | `check-task`, `record`, `close-episode` |
| Review a learning opportunity | `induction-material`, `induce`, `induction-status` |
| Inspect or diagnose | `inspect`, `doctor`, `context`, `snapshot`, `contract`, `calibration`, `budget` |
| Report or recover | `action`, `amend-cost`, `choose-next`, `bind-strategy` |
| Maintain the store | `exclude-execution`, `restore-execution`, `retire`, `rebuild-index`, `archive-calibration`, `enforce-window` |

Detailed host reporting and recovery: [host_operations.md](host_operations.md). Store and fixed-memory operations: [snapshot_lifecycle.md](snapshot_lifecycle.md).

## 2. IDs and shared rules

| ID | Read from | Use |
|---|---|---|
| T / E | Task ID / episode | Reuse across attempts |
| D | `plan-next`: `result.decision_action_id` | Optional `choose-next` |
| P | `plan-next`: `result.selected_prediction_id` | `execute --prediction P`; implementation reports |
| EX | `execute`: `result.execution_id` | `check-task`, `record --from-staged`, evidence references |
| Execution action | `execute`: `result.action_id` | Historical binding and action cost reports |
| J | `close-episode`: `result.induction_schedule.pending_ids[]` | `induction-material`, `induce` |
| Evidence execution | `induction-material`: `result.material[].execution_id` | Cite in a review |

Compare costs only on matching scopes, units and measured dimensions. A solver status describes the written model; task checks assess the original requirements.

## 3. Understand and retrieve

### `orx profile --task t.json [--cir cir.json] [--allow-empty-cir]`

Read `result.profile`, `result.derivation`, and `result.coupling.{health,modeling_guidance}`. CIR comes from `--cir` or `task.coupling`. For `cir_format` errors, correct the field named in `result.error.hint`. Use `--allow-empty-cir` only for an intentional empty CIR. See [CIR schema and examples](modeling.md).

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--candidate ID ...] [--memory-mode M] [--include-unverified]`

Find related execution evidence and strategic knowledge. `--top` caps returned hits; `--candidate` filters strategy IDs. Structural modes: `none` (empty), `cases` (statistics), `strategic` or `cost-aware` (adds knowledge). Vector retrieval remains available in all modes. `--include-unverified` exposes unpublished candidates, not admitted rules.

| Result field | What to read |
|---|---|
| `profile` | Problem structure |
| `recommendations[]` | Applicable rules or conditional statistics |
| `recommendations_basis`, `recall_budget` | Matching basis and coverage |
| `vector_recall.execution_evidence[]` | Actual methods, checks, costs and failures |
| `vector_recall.strategic_knowledge[]` | Rules, applicability and verification |
| `evidence_candidates[]`, `vector_recall.suggested_candidates[]` | Past methods to consider |
| `available_solver_families`, `solver_advisories` | Solver families and observed advisories |

Check a rule's premises and `rule.view_version` before adoption. `structural_match` is `applies|conflicts|unknown`; similarity is cosine similarity. Empty or unknown predicates are not structural matches. Semantic-only rules require vector retrieval. Adapt `method.actual` rather than merely `method.planned`; cite the cases with `--adapted-from`. Use `inspect_hint` for missing details.

If retrieval is degraded, read `degraded.reason` and `recall_budget.omitted_without_machine_support`. Restore the backend/index when needed; do not treat withheld rules as structural matches. See [retrieval modes and configuration](world_model_contract.md#recall-and-embedding-configuration).

### Embedding configuration

See [embedding backend setup and thresholds](world_model_contract.md#embedding-configuration). No per-task configuration change is needed for ordinary recall.

### `orx context --task t.json [--episode ep1] [--top 3] [--cir cir.json] [--math JSON] [--include-unverified] [--no-persist]`

Build and save a frozen prediction input for inspection or explicit reuse. `plan-next` builds its own shared context. Use `--no-persist` to preview or `orx context --context-id CTX_ID` to read an existing context. Read `context_id`, `joint.math`, `joint.unknowns`, `missing` and `degraded[]`. See [input identity and context reuse](world_model_contract.md#1-prediction-inputs).

## 4. Plan and inspect predictions

### Candidate format

Pass a JSON list or a file to `--candidates`. A minimal method has a name, concrete steps and solver; run settings belong in `config`.

```json
[{
  "name": "Direct MILP",
  "steps": ["Formulate facility-opening and allocation decisions", "Solve the coupled capacity and demand model"],
  "solver": "highspy",
  "config": {"time_limit": 60}
}]
```

Use `strategy_id` for a recorded method. Without one, the method receives a `cand_<hash8>` ID. Different solver or configuration choices are distinct candidates. See [candidate contracts](strategy_outcome.md#2-propose-candidates-and-inspect-saved-forecasts).

### Stored forecasts and their conventions

`orx inspect --bank predictions --prediction P` returns the forecast at `result.prediction.prediction` and the kind at `result.prediction.kind`. If no candidate has eligible U, read the refusal, `failure.kind` and notes; repair the input/provider settings before planning again. See [forecast diagnostics](world_model_contract.md#stored-forecasts-and-their-conventions).

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode E] --candidates specs.json [--cost-scales JSON_OR_FILE] [--horizon 1] [--max-calls N] [--time-budget S]`

Predict G/C/R/H+ once per candidate using a shared context; select highest eligible U. One candidate is valid. Read `result.decision_action_id`, `result.selected_prediction_id`, `result.selected_score` and `result.plan.candidates[]`, including the selected `capability_gain.verification_conditions`.

Decision completion establishes the implementation observation anchor. Write the selected method's code, then execute using P. No separate selection call is required. A new formal attempt requires a fresh decision in the same episode.

`--cost-scales` declares resource scales. `--execution-context runner.json` overrides `OR_EXECUTION_CONTEXT`; a legacy phase does not enter C. See [scoring](strategy_outcome.md#3-score-one-or-more-candidates).

### `orx choose-next --decision ACTION_ID (--prediction P | --chosen spec.json | --rejected) [--note "..."]`

Optional confirmation or rejection of the saved decision. Read `result.action_id`; repeated confirmation reuses the observation window. It neither consumes P nor grants another execution. Normal solving goes directly to `execute`. See [selection compatibility](strategy_outcome.md#optional-cli-selection-confirmation-and-historical-binding).

## 5. Execute and check

### `orx execute --task t.json --prediction PREDICTION_ID --code solve.py --workspace DIR [--episode ep1] [--method JSON] [--adapted-from ex_a,ex_b] [--adaptation TEXT] [--used-entry-ids ENTRY@VERSION]`

Run the selected method using an unused P for the same task and episode. The saved candidate supplies strategy, solver and config. Use the selected candidate's configuration in the actual implementation and report the settings actually applied. Keep the script inside the workspace. A fresh isolated `/work` exposes read-only `solve.py` and declared inputs; write `result.json` and temporary `scratch/` (64 MiB total). Only this attempt's result is exported. Use in-process solver APIs; network and subprocess launches are blocked. The script wall-clock limit is separate from solver settings.

Keep real-instance solving inside formal execute. Before planning, only inspect structure and use tiny artificial API probes. See [file input and isolation](modeling.md#file-input-and-isolation).

**Result file** — write actual values from this run, including solver status, finite objective for `optimal|feasible`, available bound/gap/runtime and variables for task checks. Other statuses: `infeasible|unbounded|timeout|error`.

```python
import json, os

# Replace example values with this run's observed result and read-back settings.
result = {
    "status": "optimal", "objective_value": 10755,
    "objective_bound": 10755, "mip_gap": 0.0, "runtime_seconds": 4.1,
    "variables": {"x1": 2, "x2": 3},
    "config": {"action_id": os.environ.get("OR_ACTION_ID"),
               "time_limit": 60, "seed": 42},
    "method_performed": {"action_id": os.environ.get("OR_ACTION_ID"),
                         "name": "rolling-horizon decomposition",
                         "steps": ["solve each window", "carry state forward"]}
}
json.dump(result, open('result.json', 'w'))
```

Stamp `config` and `method_performed` with the current `OR_ACTION_ID`. The performed steps must describe what ran. Executor observations take precedence over script read-back and configured settings. Declare adopted rule versions with `--used-entry-ids 3@2,5@1`; use `--used-entry-ids ''` to declare none. Cite past cases with `--adapted-from` and `--adaptation`.

Read `result.execution_id`, `action_id`, `execution`, `prediction_binding`, `method_planned`/`method_actual`, quality and failures. `solution_variables` retains up to 2000 variables. An exit-2 precondition refusal before starting is not a spent attempt; fix the named cause. Check and record every real attempt, including failures. A repaired formal submission needs a fresh `plan-next`.

#### File input and isolation

For file tasks declare only public files in `task.input_resources` and read them at `inputs/<path>`. Text-only tasks omit the list. Do not scan the workspace or benchmark directory for undeclared inputs. See [file declarations, input hashes and isolation](modeling.md#file-input-and-isolation).

#### Submission and runtime diagnostics

Read `execution.execution_features.technical_outcome` for failure stage, cause, `process_started`, `executed`, `result_received` and `runtime_eligible`. Distinguish precondition refusal, submission rejection, environment failure, implementation failure and formal timeout. Solver time-limit results are not automatically Harness timeouts. See [technical receipt table](host_operations.md#submission-and-runtime-diagnostics).

### `orx check-task <execution_id> [--check JSON] [--verification JSON_OR_FILE] [--episode ep1]`

Check a staged or recorded execution without rerunning the solver or consuming another attempt. Normal order: execute → minimum sufficient self-check → final attributed check → record → commit → close. Each call writes a check revision; a later check without attribution does not inherit the previous assessment. Recheck only changed facts or a corrected check.

| Check key | Use |
|---|---|
| `reference_objective` | Compare a sourced value; include `reference_source`, `reference_version`, optional `tolerance` |
| `reference_status` | Check reported solver status |
| `integer` | Check named `variables` or recorded variables; default tolerance `1e-6`; only if task requires integrality |
| `recompute_objective` | Recompute with task-derived coefficients |
| `semantic_probe` | Apply `{path, equals\|min\|max\|in}` to recorded fields |
| `intent` | Mark a `relaxation` or `intermediate` result |
| `g_attribution` | Sourced assessment, separate from the acceptance/check verdict |

Check original-task fidelity and feasibility before trusting a solver optimum. Add enhanced verification only for a concrete unresolved doubt that could change the conclusion. Stop when the available evidence is sufficient. Declare only applicable checks; the tool cannot infer uncovered task requirements. See [self-check principles](self_check.md) and [G attribution schema](episode_closeout.md#g-attribution-evidence).

Read `result.state`, `report.checks[]`, `report.diffs[]`, `report.scope.{basis,unchecked,note}`, `reflection_material` and `result.g_assessment_ref`.

- `passed`: declared checks held; not a proof of model fidelity.
- `failed`: locate the cause before acting.
- `insufficient`: the check could not decide; not a failure or a reason by itself to re-solve.

Correct check/evidence errors on the SAME execution. A real model or implementation change requires new planning and execution. Reuse the returned original task, method, solution, artifacts and prior-attempt references; do not reread an entire chain unnecessarily. An empty `unchecked` list is not complete task coverage.

Use `--verification` to attach independent verification (JSON or file, including script/output/result). It is agent-reported evidence on the same execution, not a new attempt or framework verdict. Script errors and assertion violations are not passes. See [verification payload](self_check.md#independent-verification-payload).

For H+=1, `hplus_verification` may record the frozen knowledge hypothesis's verification conditions. Reuse sufficient evidence or report `not_checked`/`unavailable`. Do not add a full solve just to complete these conditions.

Submit G attribution only through this interface, not `action --report verify`. Honest Unknown is valid; do not repeat checks or solves to force a numeric label. Invalid declarations are rejected before writing. External benchmark reference comparisons are after commitment, outside the retry loop. See [check and reference contracts](self_check.md#cli-check-contract).

### `orx bind-strategy --prediction ID --action ACTION_ID`

Recover a historical association only. Inspect `binding_mismatch` versus `binding_unknown`; close-out decides eligibility by signal. This command grants no execution permission. See [binding compatibility](strategy_outcome.md#optional-cli-selection-confirmation-and-historical-binding).

## 6. Record facts and cost

### `orx record`

```bash
orx record --from-staged EX [--usage-file JSON_OR_PATH] [--used-entry-ids ENTRY@VERSION]
```

Save the original attempt with its check and receipts, including failures. `--method`, `--method-actual` and `--used-entry-ids` fill missing fields, not existing declarations. Supply missing adoption metadata through `record --from-staged EX --used-entry-ids ENTRY@VERSION`; do not repeat an otherwise valid solve merely to fill it. An explicit declaration of no adoption is not a missing field. Save attempts even if cost measurements are incomplete. Use `amend-cost` for late observations.

Read `recorded`, `prediction_checks[]`, `cost_completeness`, `cost_feedback`, `index_sync` and `unrecorded_staged_executions[]`. Only declared adoptions support entry prediction checks; an observation without an entry prediction has `hit=null`.

For imports, discarding staged items, overrides and host-confirmed termination (`record --session`), see [record and host usage](host_operations.md#record-and-host-usage). Do not fabricate an execution for a host stop.

### Usage reports and overrides

Attach actual usage for the corresponding scope with `--usage-file` or `--usage-host`. Missing measurements stay unknown. Token totals are prompt plus completion; reasoning and cached tokens are breakdowns. See [usage reports and overrides](host_operations.md#usage-reports-and-overrides).

### `orx amend-cost <execution_id> [--override llm_tokens=1840,tool_calls=9] [--usage-file JSON|PATH] [--usage-host HOST] [--usage-source host] [--mode replace|increment] [--source agent_observed|agent_estimate|provider_usage] [--force]`

Update a recorded cost with matching scope and provenance. Default `replace`; `increment` adds an observed amount. Read `cost`, `cost_measured`, `still_missing` and `usage` if supplied. See [cost corrections](host_operations.md#cost-corrections).

### `orx snapshot --task t.json [--episode ep1]`

Inspect progress and budget; save a compact condition receipt. For reusable prediction inputs use `context`. See [snapshot lifecycle](snapshot_lifecycle.md).

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Report actual external actions (`model|select_strategy|verify|finish_task`) with task JSON, not bare task ID. `--params`, `--outcome`, `--cost` are JSON objects/files. An action report does not submit G attribution. See [host action reports](host_operations.md#action-reports).

### Implementation and correction reports

The host reports `implementation_facts`, `implementation_coverage` and `correction_coverage` with the selected P, actual event/action times and evidence references. Unknown or incomplete coverage must remain explicit. See [report schemas](host_operations.md#implementation-and-correction-reports).

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

Read `status`, `exceeded_dims`, `episode_exhausted`, `attempt_limited`; `--declare` saves limits. Unknown usage leaves the budget unconfirmed. Exhaustion or a confirmed host stop blocks new planning/execution. Formal attempt increments (0/1/1 for three attempts) differ from C's started corrective-round observation. See [budget details](host_operations.md#budget-and-retry-observations).

## 7. Close out and inspect calibration

### `orx close-episode --task ID [--episode ep1] [--terminal completed|failed|aborted|budget_exhausted] [--finish-action ACTION_ID] [--attribute-effect JSON ...] [--min-samples N]`

Close with the actual terminal state after commitment. Evaluate bound predictions, publish calibration and schedule H+ reviews. Link a recorded finish with `--finish-action`.

Read `closeout`, `task_checks`, `cost_completeness_warnings`, `entry_usage`, and `induction_schedule.pending_ids`. Resolve `unfinished_actions` before closing again. Review actual adoptions, including H+=0, through `entry_usage.entries[]` and `entry_usage.effect_review.pending`.

`--attribute-effect '{"entry_id":"N","execution_id":"EX_USED","verdict":"helped|neutral|unrelated|refuting","note":"...","lifecycle_action":"keep|demote|promote"}'` is repeatable. Judgements require evidence; `refuting` keeps its default pause unless explicitly managed. Publication, adoption and effect are distinct. See [existing-entry utility](induction.md#existing-entry-use-and-utility).

### `orx calibration [--min-samples N] [--rebuild]`

Read published `wm-calib/3` statistics by model, metric, unit and scope. Read each reported denominator; low support is `insufficient_evidence` with `reliability:null`. `--rebuild` is for repair/migration, not a routine solve step. See [calibration and retention](episode_closeout.md).

## 8. Offline induction

### `orx induction-material [--induction-id J] [--strategy S] [--task T] [--episode E] [--limit N] [--cursor C] [--related-top-k K] [--purpose modelling|implementation] [--query TEXT] [--executions EX1,EX2] [--view compact|detail]`

Read the scheduled review and comparable evidence. Choose modelling or implementation purpose; use `--query`/`--executions` to select comparisons. Read `material[]`, `related_history.executions[]`, `existing_knowledge`, `comparison_table` and `budget.next_cursor`. Use `inspect_hint` for full cases and `--cursor` for more material. See [choosing comparisons](induction.md#read-material-and-choose-comparisons).

### `orx induce --induction-id J --submission JSON_OR_FILE [--dry-run] [--force]`

Submit one review object: `result` (`new_rule|revision|evidence_added|already_covered|no_new_knowledge|insufficient_evidence`), `rule` (when/do/expect, when writing a rule), `basis` (kind/refs/analysis), optional `publication`. Read `induction_receipt`, `relations[].publication`, `committed`, `index_sync`.

`--dry-run` previews without settling J; remove it for actual submission. `--force` is for a justified override of the cold archive veto. The legacy advanced interface cannot be mixed with `--submission`. See [minimal submissions](induction.md#minimal-submissions) and [legacy interface](induction.md#legacy-advanced-interface).

### `orx induction-status [--task T] [--episode E]`

Read `result.completed`, `result.pending_ids`, `result.signal_counts`, `result.observation_counts`. Resume pending IDs with `induction-material`. H+=1 requires one review, not publication; H+=0 adds no review. See [review outcomes](induction.md#read-publication-completion-and-learning-separately).

## 9. Diagnostics and inspection

### `orx doctor`

Read `solvers[].api_hint`, `executor`, `memory` and `retrieval_index` as needed. Module discovery (`available=true`) does not prove runtime/sandbox usability or license access. Use the API and limit hints; no real-instance probe is required.

### `orx contract [--kind strategy_outcome] [--payload JSON]`

Construct or inspect a strategy-outcome contract; not itself a prediction. Read `status` and `prediction_made`. See [contract CLI example](world_model_contract.md#contract-cli).

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts|evaluations|retention [--task ID] [--strategy S] [--status candidate] [--episode EP] [--evaluation ID] [--prediction ID]`

Prefer specific IDs rather than whole-bank queries.

| Query | Material |
|---|---|
| `--bank experience --execution EX` | Complete execution |
| `--bank strategic --entry ENTRY [--rule-version candidate]` | Rule/version, `utility`, `status_view`, `forward_calibration` |
| `--bank predictions --prediction P` | `result.prediction.kind`, `result.prediction.prediction` |
| `--bank evaluations --evaluation ID` | One evaluation |
| `--bank texts --task T` | Task versions, match `task_text_digest` |
| `--bank values --value REF` | One saved value/body/fact/assessment |
| `--bank actions --task T --episode E` | Action history |
| `--bank snapshots`, `--bank archive`, `--bank retention` | Snapshots, retired cards, windows |

Forward calibration, publication/verification and actual use/effect are separate facts. `validated` or an adoption count is not evidence that a rule helped. Use `--status suspect|dormant` for retirement review. Read a numeric `--rule-version` when a particular published version is needed.

## 10. Corrections and store maintenance

These are explicit recovery/maintenance operations, not routine solving steps. See [full maintenance contracts](snapshot_lifecycle.md#store-corrections-and-maintenance).

### `orx exclude-execution --execution EXECUTION_ID --reason "..." [--superseded-by EXECUTION_ID]`

Withdraw incorrect facts while retaining the correction record. Do not exclude genuine failed solves.

### `orx restore-execution --execution EXECUTION_ID --reason "..."`

Restore active evidence; rebuild the execution index and refresh derived feedback when needed.

### `orx retire --entry NUMBER --reason "..."`

Explicitly retire a rule to the cold archive after inspecting its use and evidence. `--evidence-execution-id EX` (repeatable) cites supporting facts.

### `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

Rebuild after backend setup/change, deferred indexing or restored evidence. Preview with `--dry-run`.

### `orx archive-calibration [--dry-run]`

Archive out-of-window calibration detail; close-out also performs maintenance.

### `orx enforce-window [--window-episodes N] [--open-grace-days D] [--dry-run]`

Enforce the evidence window while protecting calibration and late-check scopes. Preview first when inspecting impact.

## Input lifecycle and fixed-memory maintenance

`orx context --context-id CTX --audit` reads the saved input manifest. `orx --home BASE fork-memory --destination NEW_HOME` creates an isolated per-question copy; reuse the copy across retries. `orx --home COPY maintain-snapshots` previews maintenance. See [snapshot lifecycle](snapshot_lifecycle.md) for apply, backup, expiry and vacuum options.
