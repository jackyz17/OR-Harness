# Command reference

Follow [SKILL.md](../SKILL.md) for solving and [induction.md](induction.md) for reviewing and maintaining rules.

Commands print one JSON line with `result` and `summary`. Exit codes are `0` for success, `2` for a usage or precondition error, and `1` for a crash. Use `--home DIR` to select the memory directory. The default is `$OR_HARNESS_HOME` or `./or_harness_home`.

## 1. Common paths and command index

Use the same [task input](modeling.md#start-from-the-shared-original-input) for profiling, planning and execution.

| Goal | Commands |
|---|---|
| Understand the task and find relevant memory | `profile`, `recall` |
| Predict outcomes, score U and select a candidate | `plan-next` |
| Run, check and record an attempt | `execute`, `check-task`, `record` |
| Publish calibration and schedule H+ reviews | `close-episode` |
| Take a pending review, compare evidence and submit a result | `induction-status`, `induction-material`, `induce` |
| Inspect an input or resource budget | `context`, `snapshot`, `budget` |
| Inspect stored records or diagnose the environment | `inspect`, `doctor`, `contract`, `calibration` |
| Repair or report records | `bind-strategy`, `amend-cost`, `action`, `exclude-execution`, `restore-execution` |
| Maintain knowledge, indexes and retention | `retire`, `rebuild-index`, `archive-calibration`, `enforce-window` |

## 2. IDs and shared rules

| ID | Where to read it | Where to use it |
|---|---|---|
| Task / episode | One task ID per problem and one episode per solving effort | Reuse both for retries |
| Decision action | `plan-next` → `result.decision_action_id` | Framework observation anchor; optional `choose-next --decision` |
| Selected prediction P | `plan-next` → `result.selected_prediction_id` | Implement/debug, then `execute --prediction P`; `implementation_facts[].prediction_id` |
| Execution EX | `execute` → `result.execution_id` | `check-task`, `record --from-staged`, evidence corrections |
| Execution action | `execute` → `result.action_id` | `bind-strategy --action`, `action --amend-cost` |
| Induction job J | `close-episode` → `result.induction_schedule.pending_ids[]` | `induction-material --induction-id`, `induce --induction-id` |
| Evidence execution | `induction-material` → `result.material[].execution_id` | Cite in an induction submission |

Use solver diagnostics to assess the solved model and task checks to assess the original requirements. Compare costs on matching scopes, units and measured dimensions.

## 3. Understand and retrieve

### `orx profile --task t.json [--cir cir.json] [--allow-empty-cir]`

Read the pre-strategy structure and modelling guidance. CIR comes from `--cir` or `task.coupling`. Use lists under `entities`, `decisions`, `constraints`, `relations`, `coupling_groups` and `issues`. Name entities and decisions, give constraints IDs, and give coupling groups a `type`. Relation endpoints refer to declared nodes.

```json
{"coupling": {
  "entities": [{"name": "R1", "kind": "resource", "attrs": {}}],
  "decisions": [{"name": "x", "kind": "production", "indexes": ["i", "t"]}],
  "constraints": [{"id": "C1", "kind": "capacity", "expr": "sum(x) <= limit"}],
  "relations": [{"source": "x", "target": "R1", "type": "uses_resource",
                 "evidence": "semantic", "detail": "shared capacity"}],
  "coupling_groups": []
}}
```

Set relation evidence to `structural` for co-occurrence, `semantic` for a supported relationship, or `declared` for an agent assertion. Store scalar annotations in `annotations.coupling`.

**Read** `result.profile`, `result.derivation` and `result.coupling.{health,modeling_guidance}`. Derivation uses CIR, structured `spec`, then annotations. A supplied model's coupling appears in `derivation.model_coupling`.

**Fix input** For a `cir_format` error, read `result.error.hint` and correct the indicated field. Use `--allow-empty-cir` for an intentional empty CIR. See [CIR shape and strictness](modeling.md).

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--candidate ID ...] [--memory-mode M] [--include-unverified]`

Find text matches and related structural evidence. `--top` caps returned hits and `--candidate` filters by strategy ID. `--memory-mode` controls the structural channel. `none` leaves it empty, `cases` returns statistics, and `strategic` or `cost-aware` adds knowledge. Structural recommendations include only rules with confirmed or partial machine conditions; a rule with empty predicates (`not_specified`) or unmeasured conditions (`unknown`) is not a structural match and is reachable only through vector retrieval. Vector retrieval remains available in all modes. Use `--include-unverified` to inspect unpublished candidates.

Runtime boundary: when no embedding backend or usable index exists, semantic-only rules cannot enter `recommendations`. `degraded.path=profile_only` explains the missing channel, and `recall_budget.omitted_without_machine_support` reports how many such rules were withheld. Restore the backend/index (`rebuild-index`) or read the entry directly with `inspect --bank strategic`; this is not a reason to let empty predicates occupy structural slots.

| Result field | What to read |
|---|---|
| `profile` | Task structure |
| `recommendations[]` | Published rules or conditional statistics, with basis and evidence |
| `recommendations_basis`, `recall_budget` | Matching basis and counts |
| `vector_recall.execution_evidence[]` | Past methods, task checks, costs and failures |
| `vector_recall.strategic_knowledge[]` | Rules, applicability and verification |
| `evidence_candidates[]`, `vector_recall.suggested_candidates[]` | Methods found in prior executions |
| `available_solver_families`, `solver_advisories` | Available solvers and observed environment failures |

Check the rule's conditions against the task. Read `recommendations[].knowledge.rule` or use `inspect --bank strategic --entry ENTRY` to get `rule.view_version` before declaring adoption. `similarity` is cosine similarity. For a rule, `structural_match` reports `applies|conflicts|unknown` from its predicates. For an execution it compares structural cells.

Use `method.actual` to adapt a method that was performed. `method.planned` records intent. Read its task check or failure details and use `inspect_hint` to open the full record. Cite adapted cases with `execute --adapted-from`. Propose a method from the task when the retrieved material is sparse.

**Recover text retrieval** Read `degraded.reason` if retrieval falls back to the profile. Supply task text, configure the backend, or run `rebuild-index` when the index is missing or the embedding model changes.

### Embedding configuration

| Variable | Value |
|---|---|
| `OR_EMBEDDING_BASE_URL` | OpenAI-compatible embedding endpoint |
| `OR_EMBEDDING_MODEL` | Embedding model |
| `OR_EMBEDDING_API_KEY` | Embedding credential |
| `OR_EMBEDDING_BACKEND` | `auto`, `none` or `local-hashing` |

`auto` needs the URL, model and key. Use `local-hashing` with its own index for offline tests.

Historical-retrieval thresholds are selected from the **actual backend**,
including when it is wrapped by `EmbeddingIndex`. Model identity is not
inferred from vector width. Local hashing retains its existing profile:
problem 0.62, candidate method 0.60, conditions 0.30, method relevance 0.62.
An uncalibrated remote model, including `GLM-Embedding-3`, uses the existing
`openai-compatible` provisional profile: 0.45, 0.45, 0.40, 0.45 respectively,
with `threshold_verified=false`. These numbers are not GLM calibration.

An unknown injected or missing backend never inherits verified hashing
thresholds. Its configuration explicitly reports an unverified generic
provisional fallback, its actual model id (or null when unavailable),
`backend_source` and `threshold_diagnostics`. A missing backend or unusable
index continues to report degraded local retrieval and retains global
statistics; no keyword substitute is introduced.

| Independent threshold override | Configuration field |
|---|---|
| `OR_HARNESS_HISTORY_THRESHOLD` | `similarity_thresholds.problem` |
| `OR_HARNESS_HISTORY_METHOD_THRESHOLD` | `similarity_thresholds.candidate_method` |
| `OR_HARNESS_HISTORY_CONDITIONS_THRESHOLD` | `similarity_thresholds.conditions` |
| `OR_HARNESS_HISTORY_METHOD_RELEVANCE_THRESHOLD` | `method_similarity_threshold` |

`threshold_sources` records all four sources: `profile:<name>`, an explicit
`fallback:<name>`, or `env:<variable>`. Manual overrides are independent and
are not calibration evidence: even one override, or an override equal to a
default, makes the effective configuration unverified. Zero is preserved.
Non-finite or malformed overrides are ignored with a diagnostic and also
cannot certify verification. `threshold_basis` identifies the actual model,
profile and verification limits. Pool limits, per-view admission, reranking
and task-diversification behavior are unchanged.

Real-model calibration is a separate experiment. Supply the exact model and
version, embedding preprocessing/document-builder version, production-shaped
problem/method/conditions queries, independently labelled relevant and
irrelevant historical executions (including paraphrased cross-task positives),
per-view cosine scores and sample counts, and held-out precision/recall and
admission results. Record method relevance separately from admission. Synthetic
vectors and a matching embedding dimension cannot establish a real model's
score distribution or a verified threshold.

### `orx context --task t.json [--episode ep1] [--top 3] [--cir cir.json] [--math JSON] [--include-unverified] [--no-persist]`

Build and save a frozen prediction input for inspection or explicit reuse. `plan-next` builds its own shared context. Use `--no-persist` for a preview and `orx context --context-id CTX_ID` to read a saved context.

`--cir` sets the effective CIR and `--math` supplies known attributes, such as `'{"linearity":"linear","objective_kind":"min"}'`. Read `context_id`, `joint.math`, `joint.unknowns`, `missing` and `degraded[]`. For Python reuse pass the context and the same effective task, episode and CIR to `ORHarness.predict_strategy_outcome`. See [prediction inputs and identity](world_model_contract.md#1-prediction-inputs).

## 4. Plan and inspect predictions

### Candidate format

Provide an inline JSON list or a file to `--candidates`. Give a new method a name and concrete steps. Use `strategy_id` when referring to a recorded method.

```json
[
  {"name":"bounded_enumeration",
   "steps":["derive a finite range covering every feasible integer Y",
            "derive the feasible X interval for each Y",
            "choose the best X boundary after checking objective monotonicity in X"],
   "solver":"python","config":{"time_limit":60}}
]
```

`method` describes the modelling and solving steps. Put the solver in top-level `solver` and put run parameters such as `time_limit`, `seed`, `mip_gap` and `warm_start` in `config`. Top-level `name` and `steps` form a minimal method. Without an explicit `strategy_id` the method receives a `cand_<hash8>` ID. Different solver or configuration choices are distinct execution candidates.

### Stored forecasts and their conventions

```bash
orx inspect --bank predictions --prediction P
```

Read the stored forecast in `result.prediction.prediction` and its kind in `result.prediction.kind`. Use [contract fields](world_model_contract.md) when declaring a benefit metric or inspecting a forecast.

If the plan has no eligible U, read its reason and the saved prediction's `failure.kind` and notes. Fix the input or provider settings and run `plan-next` again.

| `failure.kind` | Action |
|---|---|
| `truncated` | Raise `--wm-max-tokens` or `OR_WM_MAX_TOKENS`. Check the shared output budget when `OR_WM_ENABLE_THINKING` is enabled |
| `wrong_top_level` | Check the provider response for a JSON object |
| `empty_response`, `unparsable` | Inspect the response and output format |
| `timeout`, `network_error` | Check the endpoint, `--wm-timeout` / `OR_WM_TIMEOUT` and the planning time budget |
| `unusable_payload` | Fix or review the fields named in the validation notes |

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode E] --candidates specs.json [--cost-scales JSON_OR_FILE] [--horizon 1] [--max-calls N] [--prediction-mode M] [--time-budget S]`

Predict G/C/R/H+ once per candidate under one shared context. Save the scores and select the eligible candidate with highest U. Read `result.decision_action_id` (D), `result.selected_prediction_id` (P), `result.selected_score` and `result.plan.candidates[]`.

Solve flow: `plan-next → implement/debug → execute → check/finish/closeout`.
The framework establishes the selected candidate's implementation anchor at decision
completion. Report implementation facts and coverage with P; execution binds to
the same anchor. For a new attempt, obtain a fresh decision in the same episode.

Use `--cost-scales '{"tool_calls":30}'` to declare resource scales. `--execution-context runner.json` sets the solver interface, taking precedence over `OR_EXECUTION_CONTEXT`; a legacy `phase` key is accepted but C no longer uses a modelling phase. See [score calculation](strategy_outcome.md#3-score-one-or-more-candidates).

### `orx choose-next --decision ACTION_ID (--prediction P | --chosen spec.json | --rejected) [--note "..."]`

Optionally confirm a saved decision's scored winner or record its rejection.
Use `--rejected --note TEXT` to record why the plan was rejected. Normal solving
continues directly from `plan-next` to implementation and `execute`.

`result.action_id` identifies the framework's selection action. Repeated
confirmation reuses that action and observation window. Existing
`selection_action_id` reports remain supported. Confirmation does not consume
the prediction or change execution authorization.

## 5. Execute and check

### `orx execute --task t.json --prediction PREDICTION_ID --code solve.py --workspace DIR [--episode ep1] [--method JSON] [--adapted-from ex_a,ex_b] [--adaptation TEXT] [--used-entry-ids ENTRY@VERSION]`

Run the selected code and stage the resulting attempt. Pass an unused P from `plan-next` with the same task and episode. The saved candidate supplies the strategy, solver and configuration.

Execute binds the submission to the implementation anchor already established
by `plan-next`. It also retains automatic selection for older saved decisions;
their observations retain their original boundaries.

**Workspace** Keep the submitted script inside `--workspace`. Execution uses a
fresh isolated `/work`, with declared files at `inputs/<path>`, read-only
`solve.py`, writable `result.json` and `scratch/`. Existing host workspace
files are not exposed. Only this attempt's `result.json` is exported to
`--workspace/result.json`; scratch is temporary (64 MiB total). Use in-process
solver APIs. Network and subprocess launches remain blocked; PuLP/CBC and
Pyomo backends that launch a binary require an in-process alternative.
`pathlib` and dynamic paths are supported. The script has a wall-clock limit
in addition to its solver settings. Use labels `C1`, `C2` and so on for L2 checks.

#### File input and isolation

Declare explicit regular files in the same task used for plan-next and execute:

```json
{
  "task_id":"network-example", "family":"routing",
  "text":"Solve the network specified by the declared nodes and edges.",
  "input_resources":[
    {"source":"data/nodes.csv","path":"network/nodes.csv"},
    {"source":"data/edges.csv","path":"network/edges.csv"},
    {"source":"data/config.json","path":"config.json"}
  ]
}
```

`source` is relative to the CLI task JSON file, or to the caller's cwd in the
Python API. Explicit absolute sources are also supported. `path` is a unique
relative POSIX file name under `inputs/`, without `..`, symlinks or directory
declarations. An optional full `sha256` pins expected content. The framework
streams each file into the existing content-addressed artifact store and
freezes its source, relative path, bytes and hash with the task/prediction.
Execution rechecks content and exposes only the matching files. Changed input
declarations require a new plan; missing/changed bytes at submission are saved
as a failed bound attempt. No file contents enter the model prompt or script.
The trusted Runner must supply the public-input list: no automatic attachment,
workspace or benchmark-directory scan includes reference answers, checkers,
other tasks or memory banks.

```python
from pathlib import Path
import csv, json

filename = "nodes.csv"
path = Path("inputs") / "network" / filename
with path.open() as f:
    nodes = list(csv.DictReader(f))
with open(Path("inputs") / "config.json") as f:
    config = json.load(f)
# Installed pandas.read_csv(path), numpy.loadtxt(path), and native solver
# readers can use the same authorized files. Write results from THIS solve.
```

CSV, JSON, TXT and arbitrary regular instance formats use this same interface;
the corresponding parser must be installed. Text-only tasks omit
`input_resources`. The Python API is unchanged:
`h.execute(task, prediction_id=P, code_path="ws/solve.py", workspace="ws")`.

Linux user/mount/network namespaces, `unshare` and `libseccomp.so.2` are required.
Inputs and runtime files are read-only mounts; the remaining jail is read-only
except `result.json`, bounded `scratch/` and log descriptors. Python auditing
adds path diagnostics, while mounts/chroot and syscall filters also constrain
native readers, aliases and third-party libraries. Runtime library directories,
the current process's maps/executable link and CPU metadata are exposed for
numerical runtimes; host `/proc`, fd directories, credentials and cwd are not.
Network, process launches, namespace changes and cross-process access are
blocked. CPU/address-space/file limits and whole-process-group timeout cleanup
remain active. Missing isolation fails closed; there is no AST-only fallback.
The host kernel and installed runtime are trusted. This is not a VM or metadata
secrecy guarantee, and concurrent host filesystem mutation is outside this
boundary. Run under a dedicated account/container for hostile multi-tenant use.
Keep task files, submission workspaces and Harness memory outside runtime
library directories; detected overlaps are rejected before launch. The Runner
must also keep other tasks' materials and secrets out of trusted distributions.

One selected method can include reading, LP relaxation, feasible-bound
construction, certified reduction and final MILP in the same execute. Keep
real-instance solving and code trials in formal attempts. Before planning,
inspect data/structure and use only tiny artificial API probes. See
[self-check](self_check.md) for post-execution validation and its cost/evidence.

#### Submission and runtime diagnostics

Existing execution statuses stay `error`, `timeout` or the script's result
status. Read `execution.execution_features.technical_outcome`:

| Case | `failure_stage` | `process_started` / `executed` | Risk evidence |
|---|---|---|---|
| Syntax or documented script-contract rejection | `submission` | false / false | implementation failure |
| Missing, changed or invalid input | `submission` | false / false | cause Unknown |
| Process cannot launch | `launch` | false / false | environment failure |
| Required isolation setup fails | `sandbox_setup` | true / false | environment failure |
| Python/API exception or missing dependency | `runtime` | true / true | code error → implementation; dependency/license unavailable → environment |
| Invalid delivered JSON/protocol | `result` | true / true | implementation failure |
| Harness interrupts script | `runtime`, status `timeout` | true / true | formal timeout; code cause remains Unknown without evidence |
| Harness deadline interrupts isolation setup | `sandbox_setup`, status `timeout` | true / false | formal timeout; algorithm runtime unobserved |
| Solver's own time limit returns readable result | no failure stage | true / true | no Harness timeout |

The receipt also keeps `failure_cause`, `submission_rejected`, `result_received`
and `runtime_eligible`. A rejected attempt has formal timeout=0 because it was
not interrupted, but `runtime_eligible=false`: it is no algorithm runtime
evidence. Formal timeout labels/Brier retain their existing definition;
`technical_r.runtime_coverage` reports this distinction separately. Actual
submission/launch latency is recorded; solver runtime stays unmeasured before
script startup. Prediction, action, script, input refs, reason and spent cost
survive failure. Record it, then obtain a fresh plan-next before a repaired
submission; no implicit solve retry is introduced. Historical receipts and
probabilities are not rewritten by this diagnosis.

**Result file** Write `result.json` with `open('result.json', 'w')`. Set the solver's `status` to `optimal|feasible|infeasible|unbounded|timeout|error`. For `optimal` or `feasible`, include a finite `objective_value`. Include available bounds, gap, runtime and the values needed for task checks.

```python
import json, os

# Populate these values from this run's solver result and read-back parameters.
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

Use the numbers and method steps from the actual run. Stamp `config` and `method_performed` with the current `OR_ACTION_ID`. Read performed steps in `method_actual` and the plan in `method_planned`. `execution_config` identifies executor observations, script read-back and configured settings. Executor observations take precedence. `script_timeout_s` gives the whole-script limit.

Use `--adapted-from` and `--adaptation` to cite past cases. Use `--used-entry-ids 3@2,5@1` to declare versions actually adopted. Pass `--used-entry-ids ''` to report no adoption.

**Read and continue** Read `result.execution_id`, `action_id`, `execution` and `prediction_binding`. `solution_variables` retains up to 2000 variables. For an exit-2 precondition error, check the saved P and task identity. Run `check-task` and record every real attempt, including a failed one. Use a new `plan-next` selection for a repaired attempt.

### `orx check-task <execution_id> [--check JSON] [--verification JSON_OR_FILE] [--episode ep1]`

Check original task requirements against a staged or recorded execution. `check-task` writes a `verify` action and `execution_features.task_check`. It never runs the solver again, so a corrected check or a late check on the same execution does not create a solving attempt or consume a retry.

Normal order is execute → necessary self-check/verification → final `check-task` with supported `g_attribution` (or honest Unknown) → `record` → `close-episode`. Each call makes a new check revision. A later check without attribution does not inherit the old assessment, even if its verdict is the same. Avoid rechecking unchanged evidence; a necessary changed check must carry its own current attribution or remain Unknown.

**Before answering** The self-check is the agent's, not the tool's: check the model against the original task, then check the result against both. A solver's `optimal` proves only that the written model was solved. Use the base check on every task; add targeted enhanced checks only for concrete reasons, continue while an unresolved doubt can still change the conclusion, and stop when the evidence is sufficient. Do not default to a complete independent re-solve, and check the model-to-statement correspondence before trusting a second algorithm or solver. See [self-check detail](self_check.md) for the two layers, failure handling and examples. `--check` takes a JSON object or JSON file containing the computable bases and optional G/H+ evidence annotations; it is not the whole self-check.

| Check key | Use |
|---|---|
| `reference_objective` | Compare the reported objective with a value you source (`reference_source`, `reference_version`, optional `tolerance`) |
| `reference_status` | Check the reported solver status |
| `integer` | Check named `variables` or the recorded variables, with optional tolerance (default `1e-6`). Declare it only if the TASK requires integer values; the framework never infers integrality, and a continuous task is not "missing" this check |
| `recompute_objective` | Compute the objective from the variables with coefficients the agent declares |
| `semantic_probe` | Apply `{path, equals\|min\|max\|in}` probes to recorded fields |
| `intent` | Mark a `relaxation` or `intermediate` result |
| `g_attribution` | Optional sourced G assessment; saved separately from the generic acceptance rule and checker verdict |

A useful deterministic check is not a proof that the model represents the task: `recompute_objective` with the coefficient you already used in the code can reproduce a unit error (see [the B018 example](self_check.md#b018--a-consistent-answer-with-the-wrong-unit)). The declared bases must come from the task's units and constraints.

Read:

- `result.state` — `passed`: the declared checks held on the recorded values; `failed`: a declared check found a violation; `insufficient`: the check could not decide. None of them decides the next action. A pass does not block a new `plan-next`; a failure does not prove the model is wrong, and an `insufficient` result is neither a pass nor a reason to force a full re-solve. On a failure, analyse whether the task reading, the model, the implementation or the check itself is wrong, then act on that finding.
- `report.checks[]` — what the framework actually executed, with observed values, tolerance and `ok`.
- `report.diffs[]` — the numeric differences for a failure.
- `report.scope.basis` — the declared bases evaluated; `report.scope.unchecked` — only declared checks that could not run, with the reason; `report.scope.note` — the fixed scope statement. The framework does not scan the task text for uncovered constraints, so an empty `unchecked` is not a claim that nothing else needed checking.
- `reflection_material` — the self-check material: `task_text`/`task_ref`, the candidate and performed method, `artifacts` references (script, result.json, error clips), the captured solution with `solution_ref`, the check report, `verification_evidence`, and prior attempts as one-line references. Read it before deciding, and prefer the references over re-reading the whole attempt chain.
- `result.g_assessment_ref` — the saved G assessment, also referenced by the verify action's `outcome.g_assessment_ref` and the Evaluation's `benefit.attribution_ref`. A generic check without G attribution has no such reference. Invalid declarations return an actionable error before writing; valid Unknown attribution is accepted.

**Independent verification** When the agent runs its own verification (a separate script, an enumeration, a hand computation), attach the evidence to this same execution:

```json
{
  "script": "ws/verify.py",
  "output": "ws/verify.out",
  "result": {"ran": true, "exit_code": 0,
             "assertions": [
               {"name": "B018 order coefficient equals 15*50",
                "ok": false, "observed": 750, "expected": 50,
                "note": "code used the per-chair price per order"}]}
}
```

`--verification` accepts that JSON or a file path; `result_path` can name a JSON file instead of inline `result`. The script, output and result are captured as immutable artifacts and the normalised block is labelled `source=agent_reported` under `execution_features.verification_evidence` — a different source than the framework's `task_check`, and a separate file from the solve `result.json` (which is never overwritten). `state` distinguishes a `script_error` (the script did not run / raised) from `assertion_violations` (the script ran and an assertion was not satisfied); `assertions_*` and `evidence_only` cover the other cases. A satisfied assertion is the agent's claim, not a framework verdict, and creates no solving attempt. A `script_error` or `assertion_violations` result is NOT a pass: the self-check is not done until the cause is handled.

When the selected prediction has H+=1, the same `--check` JSON may include `hplus_verification` items for its frozen verification conditions (see [the H+ review handoff](induction.md#when-to-start-induction)). They check the predicted knowledge hypothesis, not an extra task requirement. Reuse self-check evidence where it applies; if the condition cannot be checked, record `not_checked`/`unavailable`. A complete re-solve is not the default, but an affordable check that actually verifies the mechanism is allowed — never one run merely to make the state show as checked.

A changed fact or incorrect check can be corrected by re-running `check-task` on the same execution; a real model or implementation change requires a fresh `plan-next` selection and a new execution. Submit a sourced G assessment once through this same interface. Unknown does not require repeated checks or extra solving to force a numeric label. See [G attribution evidence](episode_closeout.md#g-attribution-evidence), including unresolved reference conflicts.

**After answering** Benchmark evaluators compare the locked answer with reference evidence outside the retry loop; reference comparisons are for evaluation and offline analysis, never a reason to return to solving. For offline checks use `reference_objective` or `reference_status` with `reference_source` (`bench_declared\|independent\|self_derived`) and an available `reference_version`. The default objective tolerance is `1e-6 * max(1, abs(reference))`. Omitted source is treated as `self_derived`. See [offline use of reference evidence](induction.md#read-material-and-choose-comparisons).

### `orx bind-strategy --prediction ID --action ACTION_ID`

Recover the association between a saved prediction and an action already run. Compare their identities and observed configuration. Read `binding_mismatch` for a known mismatch and `binding_unknown` for missing evidence. [Close-out](episode_closeout.md) determines eligibility by signal. Execution permission comes from `plan-next`.

## 6. Record facts and cost

### `orx record`

```text
orx record (--execution JSON|PATH | --from-staged ID)
  [--usage-file JSON|PATH] [--usage-host HOST]
  [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment]
  [--override-source agent_estimate|agent_observed|provider_usage] [--force]
  [--prediction JSON] [--method JSON] [--method-actual JSON]
  [--used-entry-ids ENTRY@VERSION]
orx record --discard-staged ID
orx record --session --task T --reason TEXT [--episode E] [--cancelled]
  [--attempted-execution EX_ID] [--override ...]
```

Use `--from-staged EX` to save the original attempt, including its check and receipts. `--execution` imports a supplied record. `--discard-staged` drops a pending item. `--method`, `--method-actual` and `--used-entry-ids` fill missing plan, performed-method and adoption fields.

Read `recorded`, `prediction_checks[]`, `cost_completeness`, `cost_feedback` and `index_sync`. Quality feedback uses the versions declared adopted. An entry's declared prediction supports a hit/miss. A recorded observation without a prediction has `hit=null`. Attach available usage and save attempts with incomplete costs. Use `amend-cost` for a late report. Use `index_sync` and `unrecorded_staged_executions[]` to find follow-up work.

**Host stop** Use `--session` once the host confirms termination or cancellation. Keep the host's stated reason. Attach `--attempted-execution EX` when a staged attempt exists. A running action with no execution fact is recorded as interrupted. With no attempt, the command records `finish_task` for close-out. See [host stops and interrupted runs](strategy_outcome.md#4-budgets-stops-and-archiving-an-interrupted-run).

### Usage reports and overrides

The executor records available latency and solver runtime. Pass whole-attempt host usage with `--usage-file` as inline JSON or a path.

```json
{"schema":"or-host-usage/1","host":"generic","model":"your-model",
 "scope":{"task_id":"t1","episode_id":"ep1","attempt_id":"ex_..."},
 "tokens":{"prompt_tokens":700,"completion_tokens":300,
           "reasoning_tokens":120,"cached_tokens":40,"calls":3},
 "tool_calls":9,"tool_calls_lower_bound":1,
 "measured":["prompt_tokens","completion_tokens","tool_calls"],
 "provenance":{"llm_tokens":"provider_usage","tool_calls":"agent_observed"},
 "notes":"whole-attempt host report"}
```

| Item | Rule |
|---|---|
| Token totals | Prompt plus completion. Reasoning and cached tokens are breakdowns |
| Partial token totals | One observed side is a lower bound with `prompt_only` or `completion_only` |
| Measured dimensions | Use the report's `measured` list and observed values |
| Tool calls | Report all calls in scope, at or above the sandbox lower bound |
| Report plus override | Use overrides to fill different dimensions |
| Override provenance | Bare overrides are `agent_estimate`. Use `agent_observed` for observed reports and `provider_usage` for provider measurements |

`--usage-host openclaw` looks for `<home>/host_usage/openclaw-usage.<execution_id>.json`, then `openclaw-usage.json`. `generic` uses `host-usage.json`. `OR_HOST_USAGE_FILE` is the fallback. Missing reports leave the affected measurements unknown.

Reports can include window, counter or call evidence for [C observations](episode_closeout.md#c-observations). A wider scope is stored alongside attempt costs. Submit later reports through `amend-cost --usage-file`.

### `orx amend-cost <execution_id> [--override llm_tokens=1840,tool_calls=9] [--usage-file JSON|PATH] [--usage-host HOST] [--usage-source host] [--mode replace|increment] [--source agent_observed|agent_estimate|provider_usage] [--force]`

Update recorded costs for EX. `replace` is the default. Use `increment` for an additional measured amount in the same scope. Use `--source` to state provenance and `--force` to replace framework measurements. Read `cost`, `cost_measured`, `still_missing` and normalized `usage` when supplied.

### `orx snapshot --task t.json [--episode ep1]`

Inspect current problem, progress and budget. The command returns the current view and saves a compact condition receipt. Use `context` to save a reusable prediction input. See [snapshot lifecycle](snapshot_lifecycle.md).

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Record an action performed outside `execute`. Choose `model|select_strategy|verify|finish_task` and supply observed parameters, outcome and cost. Supplied cost dimensions are marked measured. Read the action with `inspect --bank actions`. Use `--amend-cost` to replace its cost.

`--task` requires a task JSON object or JSON file containing `task_id`, not a bare task ID. `--params`, `--outcome` and `--cost` also require JSON objects or JSON files; malformed input returns a named argument error. `action --report verify` records an agent-reported action and does not submit G attribution. Use only `check-task EX --check check.json` for that; putting `g_attribution` in action params/outcome is rejected with the correct interface.

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

Read episode usage or save limits with `--declare`. Tokens, tool calls, solver runtime and retries accumulate over the episode. Latency limits apply per attempt. Read `status`, `exceeded_dims`, `episode_exhausted` and `attempt_limited`. Unknown usage for a declared dimension leaves the budget `unconfirmed`. An exhausted episode or confirmed host stop blocks new planning and execution.

`execute` automatically observes each real attempt's retry increment within
the task/episode, including failures and changes of strategy or solver.
Three attempts record 0/1/1 and consume 2 retries. The corresponding
current world-model `retries` predicts started corrective/retry rounds from
its own decision through task termination (`wm-retry-corrective-rounds/1`).
It needs separate coverage and must not be backfilled as a formal increment.
Use [action outcome reports](build64_implementation.md#minimal-host-reports) for
local defects, corrections and coverage; formal execution limits are unchanged.

## 7. Close out and inspect calibration

### `orx close-episode --task ID [--episode ep1] [--terminal completed|failed|aborted|budget_exhausted] [--finish-action ACTION_ID] [--attribute-effect JSON ...] [--min-samples N]`

Close the episode with its actual terminal state. The command evaluates bound predictions, publishes calibration and schedules H+ reviews. Use `--finish-action` to link a recorded user-facing finish.

Read `closeout`, `task_checks`, `cost_completeness_warnings`, `induction_schedule.pending_ids` and `entry_usage`. Resolve any `unfinished_actions` before closing again. Use the pending IDs to [start or resume reviews](induction.md#take-a-required-review).

`entry_usage.entries[]` lists the entries this episode actually adopted, with the version resolved from their frozen `used_entries`, the read-time utility totals (`adoptions`, `judged`, `unjudged`, verdict counts) and `inspect_ref`. `entry_usage.effect_review.pending` names adoptions that still have no current valid effect judgement. Review them in the same call (H+ = 0 included): `--attribute-effect '{"entry_id":"N","execution_id":"EX_USED","verdict":"helped|neutral|unrelated|refuting","note":"...","lifecycle_action":"keep|demote|promote"}'` (repeatable). A `refuting` verdict keeps the existing default pause; an explicit management action is the agent's decision. Retiring stays `orx retire`.

### `orx calibration [--min-samples N] [--rebuild]`

Read published `wm-calib/3` statistics by model, metric, unit and scope. The window defaults to the newest 50 closed task-episodes (`OR_CALIBRATION_WINDOW`). `--rebuild` recomputes and publishes the summary for repair or migration.

Read `insufficient_evidence` and `reliability:null` below the minimum support. Use the denominator reported with each `occurrence` rate. See [calibration fields](episode_closeout.md).

## 8. Offline induction

### `orx induction-material [--induction-id J] [--strategy S] [--task T] [--episode E] [--limit N] [--cursor C] [--related-top-k K] [--purpose modelling|implementation] [--query TEXT] [--executions EX1,EX2] [--view compact|detail]`

Read material for an H+ review J. Choose `--purpose modelling` for constraint relationships or `implementation` for code and repairs. Use `--query` or `--executions` to choose comparisons.

Read `material[]`, `related_history.executions[]`, `existing_knowledge`, `comparison_table` and `budget.next_cursor`. Use `--cursor` for more cases. For full case details from an H+ material pack, follow the case's `inspect_hint`. See [choosing material](induction.md#read-material-and-choose-comparisons).

### `orx induce --induction-id J --submission JSON_OR_FILE [--dry-run] [--force]`

Submit the whole review as one compact JSON: `result`
(`new_rule|revision|evidence_added|already_covered|no_new_knowledge|insufficient_evidence`),
`rule` in **when / do / expect** when a rule is written, one `basis`
(`kind`, `refs`, `analysis`), and an optional `publication` choice. The
framework resolves the cited evidence identities, writes through the existing
versioned rule path and completes the H+ receipt. Read
`induction_receipt`, `relations[].publication`, `committed` and `index_sync`.
Later predictions can use the completed receipt as H+ feedback.

Use `--dry-run` to preview and `--force` to override the cold-archive veto
when conditions have changed. See [minimal submissions](induction.md#minimal-submissions)
for the field table.

The legacy advanced form
`orx induce [--induction-id J] [--review JSON_OR_FILE] [--relation JSON] [--strategy S] [--note TEXT] [--verify JSON] [--attribute-effect JSON]`
remains accepted for compatibility (merge, assertion audits and existing-entry
utility management) but cannot be mixed with `--submission` in one call.
Assertions in `--verify` take precedence over assertions embedded in
`--relation`; `--attribute-effect` records a sourced judgment about an adopted
rule. See [version use and utility](induction.md#existing-entry-use-and-utility).

### `orx induction-status [--task T] [--episode E]`

Read `result.completed`, `result.pending_ids`, `result.signal_counts` and `result.observation_counts`. Resume pending IDs through `induction-material`. See [review result fields](induction.md#read-publication-completion-and-learning-separately).

## 9. Diagnostics and inspection

### `orx doctor`

Check installed solver module specs, executor isolation, memory, staged attempts
and retrieval indexes. `solvers[].api_hint` names the basic entry point, domains,
constraint family, time-limit setting and process requirements.
`available=true` means module discovery only: `runtime_usable` and
`sandbox_usable` remain unknown until confirmed; no license/service probe is
performed automatically. `executor` gives both time limits, paths, resource
limits and a cached tiny isolation probe. The same compact capability facts
reach plan-next in `execution_constraints`, without a per-task solver tutorial
or real-instance probe. Read `retrieval_index` for missing/stale vectors.

### `orx contract [--kind strategy_outcome] [--payload JSON]`

Build a strategy-outcome contract with `--task`, `--spec` and optional estimates. Use `--payload` to read a saved contract.

```bash
orx contract --kind strategy_outcome --task t.json \
  --spec '{"action_type":"execute_strategy","strategy_id":"S01","scope":"attempt"}' \
  --benefit '{"kind":"solution_quality","metric":"normalized_objective_gap","unit":"1-gap","value":0.8,"baseline":{"kind":"conditional_stats","value":0.7}}' \
  --cost '{"expected":{"llm_tokens":1200},"expected_measured":["llm_tokens"]}' \
  --risk '{"events":[{"event":"task_failure","probability":0.2}]}'
orx contract --payload prediction.json
```

Read `status` and `prediction_made` to see whether the payload contains a prediction. A constructed contract reports `contract_only`. See [field definitions and stored payloads](world_model_contract.md).

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts|evaluations|retention [--task ID] [--strategy S] [--status candidate] [--episode EP] [--evaluation ID] [--prediction ID]`

| Bank | What to read |
|---|---|
| `experience`, `strategic`, `archive` | Execution evidence, current rules and retired knowledge |
| `actions`, `snapshots` | Action logs and execution context |
| `predictions`, `evaluations` | Saved forecasts and close-out evaluations |
| `retention` | Calibration and evidence windows, grace periods and archive counts |
| `texts`, `values` | Task-text versions and stored values referenced by ID |

Use `--prediction ID` or `--evaluation ID` for one record. For specific evidence and rules, use `inspect --bank experience --execution EX` or `inspect --bank strategic --entry ENTRY [--rule-version candidate|2]`. A single-entry read also returns `utility` (read-time adoption/effect/check/management summary with per-version evidence references), `status_view` (publication vs verification vs usage, kept separate) and, for versioned rules, `forward_calibration` (a calibration diagnostic, not a utility statistic). `validated`, a passed check or an adoption count is not proof that use helps. Use `--status suspect|dormant` to find rules to review for retirement. `inspect --bank values --value REF` resolves a saved body, fact or assessment. Use `--bank texts --task ID` and the execution's `task_text_digest` to locate its task version.

## 10. Corrections and store maintenance

### `orx exclude-execution --execution EXECUTION_ID --reason "..." [--superseded-by EXECUTION_ID]`

Withdraw a stored execution with incorrect facts. This removes it from active evidence, retrieval and calibration while retaining the correction record. Use `--superseded-by` to link a corrected rerun. For a real failed solve, keep its record and check as failure evidence. Within-window calibration is republished.

### `orx restore-execution --execution EXECUTION_ID --reason "..."`

Restore an excluded execution to active evidence. Use `rebuild-index --layer execution` to restore text retrieval. Run relevant induction maintenance to refresh derived feedback.

### `orx retire --entry NUMBER --reason "..."`

Move a rule to the cold archive and remove it from recall. Inspect its status and use evidence before retiring it. Pass `--evidence-execution-id EX` (repeatable) for the facts the decision relies on; the cold-archive card keeps the stated reason, the cited executions and the read-time usage summary (the old forward-calibration numbers are labelled as calibration, never as proof of usefulness). Retirement is permanent and the archive retains its entry number.

### `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

Rebuild embedding indexes after setup, an embedding-model change, deferred indexing or restored evidence. Read `layers` for item counts, model, dimensions and `unindexable` items. `--dry-run` previews the work. Configure an embedding backend for a real rebuild.

### `orx archive-calibration [--dry-run]`

Archive out-of-window calibration detail under `<home>/archive/calibration/`. Close-out also runs archive maintenance. Read `pending_migration` in a preview and use `calibration --rebuild` or an archive pass when migration is needed. See [archive limits and late-check grace](episode_closeout.md#6-retention-three-separate-scopes).

### `orx enforce-window [--window-episodes N] [--open-grace-days D] [--dry-run]`

Enforce the whole-episode execution evidence window. Defaults are 800 episodes (`OR_EVIDENCE_WINDOW_EPISODES`) and 30 days of unclosed grace (`OR_EVIDENCE_OPEN_GRACE_DAYS`). Calibration-window episodes and episodes in late-check grace are protected. Use `--dry-run` to preview evictions. See [evidence retention](episode_closeout.md#the-evidence-window-bounding-the-raw-evidence-bank).

## Input lifecycle and fixed-memory maintenance

- `orx context --context-id CTX --audit` reads the saved input manifest, including after expiry.
- `orx --home BASE fork-memory --destination NEW_HOME` creates one test-question copy. Use a fresh copy from BASE for each question and keep the same copy for its retries.
- `orx --home COPY maintain-snapshots` previews compaction. Use `--apply --backup NEW_FILE` to apply it, `--vacuum` to reclaim file space, and `--expire-open-days N` to review inactive legacy views.

See [snapshot lifecycle](snapshot_lifecycle.md) for protections and copy-first commands.
