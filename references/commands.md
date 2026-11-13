New `--induction-mode hplus` (or `OR_INDUCTION_MODE`) fixes a memory home's binary building mode; new ordinary prediction homes enable it automatically. Explicit NoH+ and random controls use separate homes, and frozen testing refuses new induction. The outer agent drains close-out's required jobs; a task close alone is not building completion. See [hplus.md](hplus.md).

# Command reference

For a cold first proposal, the normal runner can pass known execution conditions before any action is recorded:

```bash
export OR_EXECUTION_CONTEXT='{"host":"my-runner","model":"actual-executor-model","agent_id":"runner/v1","solver_interface":"highs","phase":"initial_modelling","source":"runner_configuration"}'
orx predict-strategy --task task.json --candidate candidate.json --episode ep1
# Or: orx --execution-context runner.json predict-strategy ...
```

`OR_EXECUTION_CONTEXT` accepts inline JSON or a file (`@file` also works). The global CLI flag overrides the environment; Python uses `ORHarness(execution_context={...})`. Supply the actual execution model, independently of `--world-model`, and update the declaration when runner conditions change. Host/model/optional agent ID, known solver interface, phase and source are frozen with the snapshot and remain fixed for reused contexts. Missing information remains unknown. Applicable technical R history uses the execution agent and the interface actually recorded by past executions.

C host-window evidence is optional on the existing reporting paths. `record --usage-file` and `amend-cost --usage-file` accept `or-host-usage/1` reports with explicit window/counter/call evidence (see [episode_closeout.md](episode_closeout.md)); wider scopes are stored as evidence beside the execution, preserving its attempt cost. Through the API use `report_action(..., host_usage=report)` or include the raw report in `action --report ... --outcome '{"host_usage": ...}'`. A `finish_task` report supplies an actual terminal event; `close-episode` publishes recorded evidence and supplies no substitute timestamp. Use `cancelled` action status with an `aborted` episode for a confirmed abort. The next prediction receives applicable raw `cost_evidence`; no additional prediction or feedback command is required.

Use this page to look up command syntax, important outputs, side effects, and recovery steps. Follow [../SKILL.md](../SKILL.md) for the workflow, [concepts.md](concepts.md) for the design, and [induction.md](induction.md) for forming reusable knowledge. The command index is not a checklist: optional diagnostics, repairs, and maintenance do not belong in every solve.

Commands print one JSON line to stdout: `{"result": {...}, "summary": "..."}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. Read the result state as well as the exit code. All commands accept `--home DIR`, defaulting to `$OR_HARNESS_HOME`, then `./or_harness_home`.

## 1. Common paths and command index

For a new task, understand its structure with `profile`, inspect memory with `recall`, and propose candidate methods. Choose one prediction path:

| Decision | Calls | Reuse |
|---|---|---|
| One candidate | `predict-strategy` | Pass its prediction ID directly to `execute` |
| Compare candidates | `plan-next` → `choose-next` | Pass the chosen candidate's existing prediction ID to `execute` |

Then write the chosen method's code, `execute`, `check-task`, and `record --from-staged`. Record failures too; inspect the reflection material and make a new decision when a repair is needed. At episode end, `close-episode` evaluates predictions and publishes calibration. Offline, `induction-material` supplies facts from which you form and submit a claim with `induce --relation`.

Avoid duplicate work:

- `predict-strategy` builds a prediction context by default; `plan-next` shares one across its candidates. Call `context` separately when you need to inspect or explicitly reuse a frozen input.
- `plan-next` replaces separate `predict-strategy` calls for that comparison. Selecting a candidate does not require predicting it again.
- `execute --prediction` establishes the prediction link before the run. Use `bind-strategy` only for an attempt that needs a manual or recovery binding.
- `predict-cost` is an optional historical cost baseline, separate from the world-model forecast.
- `close-episode` publishes calibration and performs retention maintenance. Extra reads, rebuilds, or retention passes are for inspection and repair.
- Direct `induce` and forecast-assisted maintenance are alternative paths. `accept-capability` already runs and binds the accepted operation; do not run it again through `induce`, `retire`, or `bind-capability`.

| Purpose | Commands | Effects |
|---|---|---|
| Understand and discover | `profile`, `recall` | Read; recall may call an embedding provider |
| Predict and select | `predict-strategy`, `plan-next`, `choose-next` | Prediction/planning calls the world model and logs spend; choice writes a decision |
| Execute, check, record | `execute`, `check-task`, `record` | Runs code; logs checks; stages and records facts |
| Close the episode | `close-episode` | Writes evaluations/calibration; performs retention |
| Optional input/baseline | `context`, `predict-cost`, `snapshot`, `budget` | Context/snapshot persist; budget declaration writes; cost baseline reads |
| Offline knowledge | `induction-material`, `induce`, `retire` | Material reads; induction/retirement change knowledge |
| Optional maintenance forecasts | `predict-capability`, `compare-capability`, `accept-capability`, `reject-capability` | Forecast, compare, decide |
| H+ building handoff | `induction-status`, `induction-material`, `induce` | Consume every frozen positive job after task close-out; its real induction receipt binds automatically. Legacy `evaluate-capability` remains a separate later-performance check |
| Inspect and diagnose | `doctor`, `contract`, `inspect`, `calibration` | Read/build views; `calibration --rebuild` writes |
| Repair/report | `bind-strategy`, `amend-cost`, `action`, `exclude-execution`, `restore-execution` | Writes links, costs, actions, or correction history |
| Store/index maintenance | `rebuild-index`, `archive-calibration`, `enforce-window` | Explicit maintenance; `--dry-run` writes nothing |

## 2. IDs and shared rules

| ID or payload | Source | Consumer |
|---|---|---|
| Task / episode | Choose one task ID per logical problem and one episode label per solving effort; retries retain them | Commands accepting `--task` / `--episode` |
| Decision action | `plan-next`: `result.decision_action_id` | `choose-next --decision` |
| Candidate prediction | `predict-strategy`: `result.prediction_id`; `plan-next`: `result.plan.candidates[].prediction_id` | `choose-next --prediction`, `execute --prediction`, manual `bind-strategy` |
| Execution | `execute`: `result.execution_id` | `check-task`, `record --from-staged`, correction commands |
| Execution action | `execute`: `result.action_id` | `bind-strategy --action`, `action --amend-cost` |
| Evidence IDs | `induction-material`: `result.material[].execution_id` | Relation evidence; `predict-capability --bundle` |
| Capability prediction | `predict-capability`: `result.prediction_id` | Compare, accept/reject, bind, evaluate |
| Recommendation | `compare-capability`: `result.recommendation` | `accept-capability` / `reject-capability` |

Keep these boundaries throughout the calls:

- A proposed method is a plan; an execution receipt reports what happened. A retrieval hit or a reuse citation does not establish successful transfer.
- Missing values are unknown. Cost comparisons require compatible units, scopes, and measured dimensions; estimates and lower bounds are not complete observations.
- Solver quality measures how well the model was solved. Task checks assess declared requirements of the original task. Knowledge verification covers the submitted assertions. These are separate judgments.
- Freeze problem identity before choosing a strategy. CIR, structured `spec`, and declared annotations supply the structural profile; a later model or solve script is a diagnostic/artifact, not a new identity source.
- In the conceptual notation, `X=(Q,C,R)` is the outcome. Legacy serialized snapshot fields named `X` and `B` hold execution progress and budget respectively; they are runtime field names, not that outcome definition.

## 3. Understand and retrieve

### `orx profile --task t.json [--cir cir.json] [--allow-empty-cir]`

**Purpose / effects:** Read-only analysis of CIR, modeling implications, and the pre-strategy structural profile. A task without a `model` is normal; model coupling, when present, is reported separately in `derivation.model_coupling` and does not change the key. Solve scripts are not profile inputs.

**Input:** CIR comes from `--cir` or `task.coupling`. Allowed keys are `entities`, `decisions`, `constraints`, `relations`, `coupling_groups`, `issues`; list keys must contain lists. Members need entity/decision `name`, constraint `id`, relation `source`/`target`, and coupling-group `type`. Relation endpoints must name declared nodes. Types/kinds are free-form strings.

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

Relation evidence is `structural` (co-occurrence), `semantic` (supported meaning), or `declared` (agent assertion). Co-occurrence alone gives generic dependency, not a semantic resource claim. Scalar annotations belong in `annotations.coupling`, not `task.coupling`.

**Read:** `result.coupling` contains validated CIR, `health`, and `modeling_guidance`; `result.profile` / `result.derivation` show values, origins, and warnings. Derivation priority is CIR → structured `spec` → annotations; `semantic_coupling` is not derived. Guidance and retrieval identity use the same structure.

**Recover / next:** Malformed CIR exits 2 with `result.error={kind:"cir_format",cause,detail,hint,...}`. Causes include `not_object`, `unknown_keys`, `bad_type`, `empty_cir`; missing members identify their exact path. Fill an empty CIR, omit it, or explicitly use `--allow-empty-cir`. `OR_CIR_STRICT=0` downgrades unknown-key/empty-CIR policy checks to `shape_lints`, never structural parse errors. Read `health.parsed` and `contributes_scalars` rather than assuming presence implies usable structure. Proceed to recall and candidate prediction.

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--candidate ID ...] [--memory-mode M] [--include-unverified]`

**Purpose / effects:** Discover relevant memory, without utility ranking or writes. Text embedding is the main discovery route; the structural channel supplies auxiliary evidence and statistics. Both remain visible when one is empty or unavailable.

**Input:** `--candidate` filters by existing method IDs; it does not create evidence or prohibit a fresh method. `--top` caps returned material, not decisions. Modes: `none` disables memory; `cases` reports statistics; `strategic` adds entries; `cost-aware` retains that view for experiment settings, without recall-side cost ranking. `--include-unverified` exposes unpublished candidates in both channels with their state labelled. A host that calls `profile` and `recall` in ONE request can pass the already-built profile object (Python `recall(task, profile=h.profile(task))`) so the task's CIR/structure is parsed ONCE instead of twice; a profile whose `problem_id` is a different task is refused (a cheap id check that does not re-parse).

**Read:**

| Result | Meaning |
|---|---|
| `recommendations[]` | Structural memory report: published entry first, otherwise conditional statistics; no `score` |
| `recommendations_basis` / `recall_budget` | Why rows exist or are empty; matched, returned, omitted counts |
| `candidates_without_evidence` | Proposed IDs without backing memory |
| `held_claims[]` | Claims outside the ordinary eligibility path, exposed by the inspection option |
| `vector_recall.execution_evidence[]` | Text hits with execution/task/strategy IDs, text digest/excerpt, observed quality/cost, failures, task check, method, solution/reuse information |
| `vector_recall.strategic_knowledge[]` | Text hits with claim, expected effects, verification/support, predicates, applicability and reuse judgment |
| `evidence_candidates[]` / `vector_recall.suggested_candidates[]` | Methods actually recorded by retrieved cases, offered for consideration when admitted knowledge is unavailable |
| `available_solver_families` / `solver_advisories` | Available concrete solvers and remembered environment failures |

Structural rows carry `strategy_id`, `expected{quality,cost,failure_prob}`, evidence/refs, confidence, basis, cost-known/basis dimensions, and risk warnings. `knowledge` carries the entry's method/actions, applicability, support, and verification information where available. For `conditional_stats`, expected fields are observed recounts, not a verified knowledge commitment.

Check admitted knowledge's conditions first. If none applies, inspect execution hits and adapt their methods. Text search has no structural pre-filter; `similarity` is raw cosine, never predicted quality or utility. For executions, `structural_match` compares cells; for entries, it evaluates predicates (`applies|conflicts|unknown`). Neither substitutes for checking semantic premises.

Method evidence is `performed`, `planned_only`, or `none`: `method.actual` comes from the execution receipt; `method.planned` is intent. A failed task check makes a case repair material; a null check is unchecked. Observed solver quality alone does not establish task success. Follow the returned `inspect_hint` when you need the full record, and cite adapted cases through `execute --adapted-from`.

**Recover / next:** Empty memory is a normal cold start: propose your own methods. There is no built-in candidate menu. A `degraded={path:"profile_only",reason:...}` means text search could not run, distinct from empty hits after a successful query:

| Cause | Action |
|---|---|
| No embedding backend | Configure it, or use the reported structural fallback |
| No task text | Supply the actual problem text |
| Index missing / embedding model changed | `rebuild-index` |
| Embedding backend error | Inspect the provider failure; structural results remain available |

`stale_indexed` counts outdated vectors excluded from search; `unindexed` counts memories without vectors, still inspectable through banks/profile. Ineligible entries are filtered before the text hit cap. Retrieval text comes from `text`, `description`, `objective`, `requirements`, `business_rules`, `constraints`, `spec`, in that order. Predict the candidate methods you choose to consider.

### Embedding configuration

| Variable | Meaning |
|---|---|
| `OR_EMBEDDING_BASE_URL` | OpenAI-compatible embedding endpoint base |
| `OR_EMBEDDING_MODEL` | Embedding model |
| `OR_EMBEDDING_API_KEY` | Credential, not persisted |
| `OR_EMBEDDING_BACKEND` | `auto` (default), `none`, or explicit `local-hashing` |

`auto` requires URL, model, and key; otherwise text retrieval is unavailable. It does not substitute hashing. `local-hashing` is for offline tests, not semantic retrieval; its `model_id=local-hashing-embedding-v1` vectors cannot mix with real-model vectors. These settings are separate from `OR_WM_*`. Python callers may inject `ORHarness(home=..., embedding=MyBackend())`.

### `orx context --task t.json [--episode ep1] [--top 3] [--cir cir.json] [--math JSON] [--include-unverified] [--no-persist]`

**Purpose / effects:** Optional explicit construction of a frozen prediction input. It gathers a snapshot and recall once, may call the embedding provider, and persists by default; no world-model call, solver run, or induction. Read an existing context with `orx context --context-id CTX_ID`, which reruns nothing.

**Input / read:** Freezes task text/payload, CIR relations, math attributes with origins, profile/derivation, execution progress, budget, deduplicated retrieval, capability evidence/version, and execution constraints. Read `result.context_id`, `task_digest`, `joint.math`, `joint.unknowns`, `missing`, `degraded[]`, `evidence_classes`, and `capability_version`. Evidence categories stay distinct: execution facts, verified/unverified/legacy knowledge, and structural recommendations.

`--math` declares known attributes, e.g. `'{"linearity":"linear","objective_kind":"min"}'`. A supplied `--cir` replaces task coupling consistently for representation, profile, and retrieval; `joint.sources.cir` reports its origin. Missing model/CIR/text is reported, not guessed. Retrieval deduplicates `layer:id`; content digests cover consulted knowledge, not read timestamps.

**Recover / next:** Reuse the context with `predict-strategy --context CTX_ID` and the same effective CIR. Task/version/episode or known structural mismatches are refused; unknown structure is not a mismatch. `--no-persist` returns content without creating a later lookup target. Historical `--snapshot` builds use frozen knowledge/retrieval, exclude later-created memories, and mark unavailable reliability/evidence blocks missing; they cannot reconstruct history from today's banks. See [prediction_context.md](prediction_context.md).

### `orx predict-cost --task t.json --strategy S`

**Purpose / effects:** Optional read-only historical attempt-cost baseline; no provider call. It does not predict benefit, risk, or uncertainty.

**Read / next:** `source=entry|stats|unknown`, support counts, per-dimension support, and expected cost. Unknown or incompatible evidence yields null, not free execution. Pass the printed cost snapshot to `record --prediction` if you used it, so feedback compares with that frozen baseline. Use `predict-strategy` or `plan-next` for world-model consequences.

## 4. Predict and choose

### Candidate format

`predict-strategy --candidate` accepts inline JSON or a path; `plan-next --candidates` takes a list. A minimal described candidate is:

```json
{"name":"bounded_enumeration",
 "steps":["derive a finite range covering every feasible integer Y",
          "derive the feasible X interval for each Y",
          "choose the best X boundary after checking objective monotonicity in X"],
 "solver":"python","config":{"time_limit":60}}
```

Top-level `name`/`steps` normalize into `method={name,steps,why?,fallback?}`; string steps become one step. Without an explicit strategy ID, method content allocates `cand_<hash8>`. Solver/config do not enter that method ID, but do distinguish execution candidates. Allocation is not a claim that historical methods are equivalent; use an existing `strategy_id` when referring to one.

The full shape supports `action_type:"execute_strategy"`, task/episode identity, `strategy_id`, `method`, `solver`, `config`, and `scope:"attempt"`. Legacy `ActionSpec` preserves execution params and budget hints; an unmappable scope such as `task` is refused. Method-like content in config is moved to method; unknown execution-parameter keys are refused. Conflicting top-level and nested method declarations, a named method without steps, or neither ID nor method content are rejected before a model call. A bare ID alone does not describe a cold-start method: supply the actual steps.

**Field responsibilities — do not duplicate.** `solver` (top level) is the declared solver / calling interface. `method` is the modelling form, variable domains, objective, constraints and the method steps. `config` is the execution parameters: time limit, random seed, solve precision, and comparable knobs. Put the solver ONLY at the top level: a `config.solver` copies it, and the two spellings need not agree — so a duplicate is neither matched nor mistaken for a mismatch (it does not block the outcome). Genuine execution params (`warm_start`, `msg`, `time_limit`, `seed`, `mip_gap`, …) stay in `config`; only keys that name no execution parameter at all are refused.

### `orx [--world-model URL::MODEL] predict-strategy --task t.json --candidate c.json [--episode ep1] [--context CTX_ID] [--cir cir.json]`

**Purpose / effects:** Make and persist one `wm-so/1` candidate forecast and its real call usage. Builds a fresh context by default; an explicit context is checked against task/version/episode and effective CIR. Provider input contains the frozen content, not just IDs.

**Read:** `result.prediction_id`, `result.prediction.candidate.strategy_id`, forecast status and failure. The forecast covers benefit with metric/unit/baseline, five-dimensional cost with a predicted mask, named risk events, and explicitly uncalibrated self-reported uncertainty. Normalized solution quality must be in `[0,1]`; a raw objective is refused, not clamped. Capability-gain traces, when present, remain predictions for later evaluation.

Declare the benefit meaning before observing the result:

| Kind / metric | Observation |
|---|---|
| `solution_quality` / `normalized_objective_gap` | Solver gap; optimal has gap zero |
| `effective_completion` / `task_check_passed_given_faithful_execution` (**new default**) | `probability`, `wm-g-faithful/1`: faithful+original acceptance passed=1, faithful+confirmed method-caused failure=0, otherwise unknown |
| legacy `effective_completion` / `task_result_check_passed` | Declared task check: passed=1, confirmed failed=0, unchecked=unknown |
| `valid_progress` | No observation channel in this build; reported without scoring |

New G is P(original-task acceptance | faithful execution of the candidate steps under the agreed solving conditions); method omissions remain omitted, while script implementation risk belongs in R. The request fixes convention and scope before the call. A different metric or non-finite/out-of-range numeric G makes G unavailable with a reason, preserving C/R. `benefit.notes` keeps one grounded requirement → candidate step → expected-consequence sentence even when `value` is absent. Explicit old API/planning conventions and stored forecasts keep their original meanings.


**Recover / next:** No usable prediction is `invalid` with `result.failure`; configuration/availability and actual forecast production are separate facts. One call, no automatic retries/default forecasts. Failures are saved with usage consumed:

| `failure.kind` | Recovery |
|---|---|
| `truncated` | Increase `--wm-max-tokens` / `OR_WM_MAX_TOKENS`; consider thinking's shared output budget (`OR_WM_ENABLE_THINKING`) |
| `wrong_top_level` | Fix the payload/prompt; expected JSON object |
| `empty_response` / `unparsable` | Inspect endpoint output/format |
| `timeout` / `network_error` | Retry deliberately or adjust `--wm-timeout` / `OR_WM_TIMEOUT` and planning time budget |
| `unusable_payload` | Fix fields named in validation notes |

Unknown `finish_reason` remains unknown. Pass the chosen prediction ID to `execute`; do not also plan the same decision. See [strategy_outcome.md](strategy_outcome.md).

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode ep1] --candidates specs.json [--horizon 1] [--max-calls N] [--delta W] [--prediction-mode M] [--time-budget S]`

**Purpose / effects:** Optional bounded comparison replacing individual prediction calls. Requires caller-proposed `execute_strategy` candidates and a configured provider. Freezes one root snapshot/context, predicts candidates against it, logs planning spend, and suggests a first step. It neither selects nor executes.

**Limits / scoring:** Horizon is exactly 1; other values exit 2. Default bounds are at most 3 candidates, 6 model calls, and 120 seconds for the whole decision. Remaining wall time limits each next call; exhaustion is explicit. Among candidates with comparable G, utility uses common terms of `alpha*G - beta*C - gamma*R`. C uses their valid same-scope `expected_measured` dimension intersection followed by common component deduplication; empty intersection omits C for everyone. R reads only `implementation_failure.probability` and is omitted for everyone if any ranked candidate lacks it. Explicit zeros are valid. Raw inputs, omissions and shared scales/weights remain in the score audit. Incompatible G produces no utility. The knowledge bonus is off; planning spend is charged once outside candidate scores. Existing scales, weights and gates are unchanged.

**Read / next:** `result.decision_action_id`, `plan.candidates[].prediction_id`, suggestion, `planning_cost`, `effective_parameters`, per-candidate failures/rejections, `score.incomparable`, and `budget_confirmation`. Choose with `choose-next`, then execute using the same prediction ID. Missing budget evidence is `unknown/unconfirmed`, not an assurance of affordability.

**Recover:** Known exceeded budget stops before model calls (`fallback`). No usable forecast gives `no_valid_predictions` and no suggestion; inspect memory or choose explicitly. A malformed/failed candidate does not prevent scoring the others. No imagined rollout or extra prediction is needed. See [strategy_outcome.md](strategy_outcome.md).

### `orx choose-next --decision ACTION_ID (--prediction PREDICTION_ID | --chosen spec.json | --rejected) [--note "..."]`

**Purpose / effects:** Record acceptance, deviation, or rejection of a plan. Writes the selected-plan execution-context field, not execution quality; no provider or solver call.

**Input / next:** Prefer a prediction ID from this decision's `plan.candidates[]`. It identifies the stored candidate without retyping JSON. `--chosen` accepts its full spec; if both are supplied they must agree. Deviation compares action type, strategy, solver, and configuration, not cosmetic plan fields. Execute with the chosen prediction ID.

**Recover:** Unknown/non-selection decisions and prediction IDs outside this comparison are refused. Rejection records `last_rejected_suggestion` without replacing the selected plan.

## 5. Execute and check

### `orx execute --task t.json [--prediction PREDICTION_ID | --strategy S04 --solver NAME] --code solve.py --workspace DIR [--episode ep1] [--method JSON] [--adapted-from ex_a,ex_b] [--adaptation TEXT] [--used-entry-ids 3,5]`

**Purpose / effects:** Run the chosen code, log an action, capture task text/version and observed configuration, and stage every resulting attempt. Staging is not recording; use `record --from-staged` afterward. No automatic second solve is started.

**Input / binding:** With `--prediction`, the frozen candidate supplies strategy, solver, configuration identity, and episode; contradictory explicit values, changed problem identity, or an already-claimed prediction are refused before the run. The action/link is persisted before execution. Adding a post-strategy model does not change problem identity. Without a prediction, supply strategy/solver; this is an explicitly unpredicted attempt, eligible for a later manual binding. `--method` describes its plan.

**Sandbox:** Code must live inside the workspace, which becomes its working directory. Blocked imports include `subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil`, and `pathlib`; only `os` / `os.path` are allowed from `os`, with operations such as `chdir`, `walk`, and `remove` refused. In-process solvers such as `highspy` and `ortools` are allowed; subprocess-based PuLP/CBC cannot run. Constraint labels must be `C1`, `C2`, … for L2 cross-reference. POSIX resource limits and a whole-script wall-clock timeout apply.

**Result file:** Write with literal `open('result.json', 'w')` inside the workspace. Prior output is removed; stale output is refused. Required fields are `status`, `objective_value`, `objective_bound`, `mip_gap`, and `runtime_seconds`; report `variables` whenever task checks need solution values. Legal statuses are `optimal|feasible|infeasible|unbounded|timeout|error` (case-normalized). Use the actual solver status: a feasible incumbent does not establish optimality. Boolean statuses are errors.

For a run that actually proved the reported optimum, the result shape is:

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

This illustrates the reporting shape, not a solve or a reference answer. Replace all values and steps with the run's observations. Stamp optional configuration/method receipts with `OR_ACTION_ID`; only matching receipts are accepted, including on failure paths. Without a receipt, performed method/configuration remains unknown rather than copied from the plan.

**Read:** `result.execution_id`, `action_id`, `execution`, and `prediction_binding`. The execution contains quality diagnostics, raw cost and measured mask, runtime provenance, CIR snapshot, text digest, and `execution_features`. Variables are stored under `solution_variables`, capped at 2000 with a truncation report; checks needing omitted variables stay unchecked.

`execution_config` distinguishes `executor` observations, `script_reported` read-back values, and `executor_configured` instructions. Instructions are not evidence that a solver honored them. Executor observations prevail over contradictory receipts, with conflicts retained. Whole-script `script_timeout_s` differs from solver time limit. Method receipts populate `method_actual`, trajectory, and `method_receipt`; candidate intent remains `method_planned`.

Optional `--adapted-from` and `--adaptation` store `reuse_trace`. Citation records the cases considered, not benefit; unknown cited IDs are retained without validating existence.

`--used-entry-ids` declares the knowledge ENTRY NUMBERS this attempt ADOPTED (e.g. `3,5`; `''` records "adopted none"). This is a DECLARATION, distinct from recall (`last_consulted_at`) and from `--adapted-from` (cases you read): only declared entries have a forward check computed, so a mere recall never counts as a successful use. The adoption is stored under `execution_features.used_entries`; an unknown number is kept as a citation, and the flag never rewrites a value the record already carries.

**Cost:** The executor observes latency and solver runtime with provenance. Supply whole-attempt tokens/tool calls from a host report; hand-typed defaults are estimates. Tool calls have a provable sandbox lower bound. Retries are measured zero only when no earlier attempt exists for this task/episode/strategy; otherwise remain unknown until supplied. A static-policy rejection has no measured solving cost. See recording/cost rules below.

**Recover / next:** Run `check-task`, then record the staged attempt, including failures. Prediction precondition errors exit 2 before execution. If the executor fails after starting, it stages its own failure/cost and releases the prediction claim for a separate retry; the failed attempt remains evidence. Inspect `prediction_binding` (`bound`, `config_observed`, `config_unknown`) and use manual binding only when needed.

### `orx check-task <execution_id> [--check JSON] [--episode ep1]`

**Purpose / effects:** Verify declared requirements of the original task against a staged or recorded result. Writes a `verify` action and `execution_features.task_check`; no solver or prediction call. This is separate from solver-side model quality.

**Input:** Declare only checks supported by real task requirements and a truthful reference source:

| Check key | Meaning |
|---|---|
| `reference_objective`, `tolerance` | Compare objective; default tolerance `1e-6 * max(1,abs(reference))` |
| `reference_status` | Require the stated solver status |
| `integer` | Optional named `variables`, optional tolerance (default `1e-6`); otherwise checks recorded variables |
| `recompute_objective` | Coefficients, optional constant/tolerance; recompute from solution variables |
| `semantic_probe` | One or more `{path, equals|min|max|in}` probes over record fields |
| `reference_source`, `reference_version` | `bench_declared|independent|self_derived`; omitted source is self-derived |
| `intent` | `relaxation|intermediate`: intentionally not the final task answer |

**Read:** `result.state`, `report.conclusion`, `report.checks[]`, `report.scope.unchecked[]`, and optional `reflection_material` / `next`.

- `passed`: declared checks held; undeclared constraints/model fidelity remain outside the scope. A self-derived reference gives a consistency check, not independent confirmation.
- `failed`: a declared check ran and failed. Keep the execution, solver measurements, and real cost; it cannot support task-success claims. For prediction calibration, solver quality stays its own measurement and task completion uses the separate check channel.
- `insufficient`: a needed value/basis/path is missing or the result is unusable; neither pass nor refutation. Missing probe paths name available fields.

**Recover / next:** Record either way. On failure, inspect task text, solution, code hash, check report, and earlier attempts in `reflection_material`; locate interpretation/model/code/reference issues, repair, and make a new prediction. Never change a benchmark reference to match your answer. On insufficient evidence, supply an available basis or report required variables; do not invent a reference. Natural-language constraints are not parsed automatically, and objective agreement alone does not prove model fidelity. A staged annotation carries through recording; late checks are handled by close-out correction rules.

### `orx bind-strategy --prediction ID --action ACTION_ID`

**Purpose / effects:** Manual/recovery association for an action already run; same identity rules as automatic binding, no model call or re-billing. Idempotent re-binding re-evaluates comparability.

**Read / recover:** Checks action/task/episode/strategy/solver and observed execution config. Known `binding_mismatch` and missing `binding_unknown` are distinct. Missing receipts are caveats, not automatic whole-sample exclusions: an unreported time limit blocks no comparison by itself; a different solver blocks outcome dimensions while keeping actual cost. Read `config_observed` / `config_unknown`; predictions never fill missing observations. Close-out determines eligibility field by field. See [episode_closeout.md](episode_closeout.md).

## 6. Record facts and cost

### `orx record`

```text
orx record (--execution JSON|PATH | --from-staged ID)
  [--usage-file JSON|PATH] [--usage-host HOST]
  [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment]
  [--override-source agent_estimate|agent_observed|provider_usage] [--force]
  [--prediction JSON] [--method JSON] [--method-actual JSON]
  [--used-entry-ids 3,5]
orx record --discard-staged ID
orx record --session --task T --reason TEXT [--episode E] [--cancelled]
  [--attempted-execution EX_ID] [--override ...]
```

**Purpose / effects:** Append an execution fact, or record a confirmed host stop. Prefer `--from-staged` to retain the original payload. `--execution` accepts a bare record, execute envelope, or legacy one-level form. Discarding pending material is a separate explicit mode, not a substitute for recording a real failed attempt.

**Execution record:** Applies cost backfill (replace by default), freezes quality checks against the entries this run **DECLARED it adopted** (`--used-entry-ids`) onto `execution_features.quality_feedback`, and compares measured cost with the frozen historical prediction under `cost_feedback`. A check exists ONLY for a declared adoption, and a hit/miss is computed ONLY against a DECLARED prediction (an entry with no declared prediction records the observation with `hit=null`, never a default-interval hit). Recording neither interprets the method nor promotes/demotes knowledge; later induction replays feedback.

`--method` declares the plan; `--method-actual` declares observed performance. Existing record values, including script receipts, are not overwritten. `--used-entry-ids` declares the knowledge ENTRY NUMBERS this attempt ADOPTED (e.g. `3,5`; `''` records "adopted none"), tying them to this attempt's result; a value the record already carries is never overwritten. Explicit `execution_features.contrast` marks contrast evidence; there is no per-record retention label.

**Read / next:** `result.{execution_id,recorded,prediction_checks[],cost_completeness?,cost_feedback?}`, optional `unrecorded_staged_executions[]`, and `index_sync`. Missing cost is reported with dimensions, lower bounds, and an amend hint; it does not block recording. Task text resolves from a supplied digest, otherwise the most recent real task snapshot. If unavailable it stays missing. Index sync is best effort: `synced|deferred|skipped`; a failure never rolls back a durable fact. Record remaining staged attempts, then retry deliberately or close the episode.

**Host stop:** `--session` requires host confirmation that the process ended/cancelled. Keep its stated reason verbatim. With `--attempted-execution`, annotate the staged fact; with a running action but no fact, record an interruption with unobservable result and real cost; with no attempt, record `finish_task`, never invent an execution. This idempotent finish operation marks the episode terminated, permits close-out, and is not blocked by the budget it records. See [strategy_outcome.md](strategy_outcome.md#4-budgets-stops-and-archiving-an-interrupted-run).

### Usage reports and overrides

`--usage-file` accepts inline JSON or a path. Prefer the versioned whole-attempt report:

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

| Rule | Handling |
|---|---|
| Token total | Prompt + completion; reasoning/cached are sub-facts, not added again |
| Only one token side known | Lower bound with `prompt_only` / `completion_only` basis |
| Measured whitelist | Only declared measured dimensions enter `cost_measured` |
| Tool calls | All invocations in attempt scope; below the provable lower bound is refused |
| Host report + override | Same-dimension conflict is refused; override may fill other dimensions |
| Bare override | Defaults to `agent_estimate`; displayed, excluded from calibration actuals and measured cost claims |
| Observed override | Use `agent_observed` for a real report read by the agent, `provider_usage` for provider/host measurements |

Legacy reports accept optional prompt/completion/reasoning/cached totals, model/source, `report_id`, and per-call `calls[{id,...}]`. Per-call values sum only when totals are absent; repeated report/call IDs deduplicate. A known call marked `late:true` replaces its figure rather than adding it. No numbers means no measurement.

`--usage-host openclaw` checks `<home>/host_usage/openclaw-usage.<execution_id>.json`, then `openclaw-usage.json`; `generic` uses `host-usage.json`. `OR_HOST_USAGE_FILE` is the last fallback after explicit file/host selection. Missing reports leave dimensions unknown and recording succeeds; an unknown adapter exits 2 with known names. The host owns the usage report: request provider usage (streaming may need `stream_options.include_usage=true` / `compat.supportsUsageInStreaming:true`), aggregate one report per attempt, and deliver it to the adapter or `--usage-file`.

### `orx amend-cost <execution_id> [--override llm_tokens=1840,tool_calls=9] [--usage-file JSON|PATH] [--usage-host HOST] [--usage-source host] [--mode replace|increment] [--source agent_observed|agent_estimate|provider_usage] [--force]`

**Purpose / effects:** Repair recorded cost in place without rerunning or appending an execution. Uses the report/measurement rules above and recomputes feedback against the frozen prediction.

**Input / read:** Replace is default and idempotent; increment adds a separately measured amount within scope. `--source` defaults to `agent_estimate`. Framework measurements, including latency/runtime and provider-usage dimensions, require `--force` to overwrite. Mixed complete/lower-bound token bases are reported as `token_basis_mixed`. Result includes `execution_id`, `mode`, `cost`, `cost_measured`, `still_missing`, and normalized `usage` when supplied.

**Recover / next:** Tool calls below the sandbox floor exit 2. A late host report can fill both tokens and tool calls at once. Remaining unknown dimensions support no measured cost claim. Task attempt costs sum recorded attempts; retries mean new retries per attempt. End-to-end latency needs explicit task timing, never a sum/max inference. Re-read a historical cost baseline only when a later decision needs the corrected evidence.

### `orx snapshot --task t.json [--episode ep1]`

**Purpose / effects:** Return the current temporary belief view and persist a lightweight condition receipt with shared problem/progress values and compact budget facts. Full coverage and knowledge bodies are not added to historical snapshot storage. Progress includes selected plan, model, solution, and verification, each labelled by provenance and epistemic status. Later writes do not change the snapshot.

**Read / next:** Use the returned temporary object to inspect current state. Reading a new receipt later explicitly reports that full coverage was not retained; complete legacy snapshots remain readable. Use `context` for cross-process prediction-input reuse. The default prediction/planning path captures its own inputs; a separate snapshot is unnecessary there. A task missing `task_id` / `family` is refused. Legacy `X` is progress here, not conceptual Q/C/R; coverage is not an aggregate capability score.

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

**Purpose / effects:** Report an action performed outside framework execution: `TYPE=model|select_strategy|verify|finish_task`. Writes `source=agent_reported`; missing pre-state is `pre_snapshot_missing`, not reconstructed. Explicit cost dimensions, including explicit zero, are tagged measured by this reporting interface; provide only observations you can substantiate.

**Next:** Inspect through `--bank actions`. `--amend-cost` replaces the reported action cost idempotently. Do not report the same automatically logged action again.

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

**Purpose / effects:** Read consumption, or persist a declaration. Covers real own-cost actions plus recorded/staged executions, deduplicating execution IDs and macro reference costs. Unlinked executions are `unattributed`, not charged to a new episode; hypothetical actions are excluded.

**Read:** `exceeded|ok|unconfirmed|no_budget_declared`, `exceeded_dims`, `episode_exhausted`, `attempt_limited`. Tokens, tool calls, solver runtime, and retries accumulate per episode; latency limit applies per attempt. An episode overrun or recorded host termination stops new execute/prediction work. A prior attempt's latency overrun is retained but permits a retry. Unknown declared dimensions mean unconfirmed affordability.

Execute actions are logged automatically; induction logs maintenance under `__maintenance__` / `maint_<ts>` with pre/post state, verification, costs, and knowledge delta. Created-unverified entries are not confirmed capability growth. Failed induction leaves a failed action; dry-run writes nothing. Close the episode when finished; do not add a budget read after every action unless needed for a decision.

## 7. Close out and inspect calibration

### `orx close-episode --task ID [--episode ep1] [--terminal completed|failed|aborted|budget_exhausted] [--finish-action ACTION_ID] [--min-samples N]`

**Purpose / effects:** Close one episode, evaluate bound strategy predictions against real outcomes, publish calibration for later contexts, then archive/enforce retention. It also registers existing versioned rule uses and saves an actual induction start judgement; no solver/model call or new-rule induction. Idempotent re-close returns the stored record without counting twice; interrupted publication can recover (`recovered_publication:true`).

**Read:** Per-field benefit errors under the declared metric/baseline, same-scope measured cost errors, labelled-risk Brier scores, interval coverage, pending/excluded reasons, `task_checks`, and `cost_completeness_warnings`. Warnings identify execution IDs with unknown token/tool-call dimensions; use late usage/amendment if available. Honest incomplete costs do not block closing.

Solver quality remains the solver's measurement. Task failure is a separate `benefit.task_check`; new G is observed through `effective_completion/task_check_passed_given_faithful_execution` only with sourced fidelity and original-acceptance evidence; explicit legacy completion retains its old rules. Closing is not answer certification, and unchecked validity remains unknown.

For new G, the optional `g_attribution` object belongs inside the existing `--check` JSON. Declare `method_fidelity`, `failure_source`, `acceptance_scope`, a versioned `acceptance_id`, `source`, `assessment` and `evidence_refs`. It cites code AND result evidence on the execution; it does not alter the ordinary checker verdict. See the [complete evidence schema and example](episode_closeout.md#g-attribution-evidence). Crashes/timeouts with no evaluable result, method changes, implementation omissions, inadequate acceptance or unclear causes leave G unknown. Same step text or a checker `failed` alone is insufficient.

`close-episode` publishes conditional G facts and same-task method comparisons through the existing feedback blocks. Global G residuals remain diagnostic; G correction reminders are removed, including old cached reminders at the provider boundary. Conditional counts deduplicate repeated execution/ranges; Brier/bins require both numeric G and an attributable label. Two methods passing the same binary checker tie.

**Late corrections:** Stored evaluations retain what was known at close-out. Live re-derivation updates later calibration use, reported in `validity_corrections[]`; within-window corrections republish calibration (`calibration_republished`). A late failed check changes task-check/completion facts while retaining measured solver quality/cost; withdrawn checks restore the ungated state; excluded executions remove their sample. Corrections are scoped to the executions actually compared, not sibling evaluations.

Close-out reads the pre-execution binary `capability_gain.decision` and returns durable `induction_schedule.pending_ids` / `next_actions`. The outer building agent must settle all positive jobs. Failed executions and missing learning checks still trigger 1. `--induction-decision allow|skip|pending` is historical compatibility and cannot override a new switch.

**Recover / next:** Running actions produce a pending refusal; finish them honestly, including confirmed host stops, and close again. Read check coverage, then undertake offline induction when there is something worth examining. See [episode_closeout.md](episode_closeout.md).

### `orx calibration [--min-samples N] [--rebuild]`

**Purpose / effects:** Read published `wm-calib/3`; `--rebuild` explicitly recomputes and republishes for migration/repair. Default window: newest 50 closed task-episodes (`OR_CALIBRATION_WINDOW`). New prediction contexts receive the published summary automatically.

**Read:** Groups separate model identity, metric, unit, and scope. Statistics include signed benefit error, cost log ratio, interval coverage/width, per-event Brier, and predicted probability versus occurrence over the same scored denominator. Separate `occurrence` reports its own observation-unit counts/denominator; do not mix those rates. Below minimum support: `insufficient_evidence`, `reliability:null`. `applicability:"global_diagnostic"` does not imply per-strategy reliability or guaranteed prediction improvement. Legacy knowledge-prediction statistics are separate.

## 8. Offline knowledge and capability feedback

### `orx induction-material [--induction-id J] [--strategy S] [--task T] [--episode E] [--limit N] [--cursor C] [--related-top-k K] [--purpose modelling|implementation] [--query TEXT] [--executions EX1,EX2] [--view compact|detail]`

**Purpose / effects:** After execute/check/record/close, authorize the positive executed job identified by `--induction-id J` (from close-out pending IDs), before retrieval. H+=0/missing/invalid cannot authorize material. One existing index lookup supplies cross-source mechanism candidates and changed-premise neighbors. Each read queries current evidence and records references, without caching or replaying a full response. The actual trigger remains fixed; newer attempts paginate as controls. Pure old/manual libraries retain their historical transition; new prediction homes use binary H+ automatically.

**Read:** The default `induction-material/2` pack contains small execution comparisons, actual/planned methods and sources, source-problem identities, checks/failures, measured resources/masks/scope, and short old rules/state. `modelling` prioritizes constraint relationships across base problems; `implementation` emphasizes implementation/repair contrasts. Similarity/family is discovery. Known source relationships and exact input overlap are advisory facts; the agent chooses validation purpose.

**Budget / detail:** The entire default compact JSON, including history, knowledge, H+, diagnostics and CLI envelope, is bounded at 32000 characters (`OR_HARNESS_INDUCTION_MATERIAL_CHARS`). `budget.chars_used` counts the actual serialization. Values are complete or replaced by detail references; method prefixes are never evidence. Use `budget.next_cursor` for more. Delivered cases are available review references; the agent selects which support its conclusion. Pending necessary evidence is temporarily retained and full copies are released after the receipt. `--view detail` is an expanded legacy diagnostic; inspect by ID for targeted full evidence.

### `orx induce [--induction-id J] [--review JSON_OR_FILE] [--strategy S] [--relation JSON] [--dry-run] [--force] [--note TEXT] [--verify JSON] [--attribute-effect JSON]`

**Purpose / effects:** Complete the same positive job with an agent-written when/do/expect rule plus real evidence IDs, or an empty review receipt. Both material and writing enforce job authorization. H+=0 cannot authorize a new/changed body. Existing-body validation/evidence, explicit publication selection and utility management use a maintenance-only path without a positive job. Publication is the agent's explicit choice over its evidence scope. The framework assigns identity, resolves scope/evidence, freezes versions, appends checks/events and synchronizes affected documents. It does not infer a technique from means. Normal interaction is one review round, with current material queries as needed, followed by one idempotent submission.

```bash
orx induce --relation '{"when":"For fixed Y, feasible integer X is an interval and the objective is monotone.","do":"Enumerate Y, derive both X endpoints and select using coefficient sign and objective direction. Disable reduction when monotonicity fails.","expect":"Preserve the original optimum and constraints.","evidence":["ex_REAL"]}'
orx induce --relation '{"entry_id":"3","operation":"add_evidence","evidence":["ex_NEW"]}'
# Revision: entry_id, operation=revise, complete when/do/expect.
# Merge: entry_id, operation=merge, merge_entry_ids and merge_reason.
```

**Versioning / publication:** Exact and safe formatting repeats reuse the body/version. `publication:{"state":"published"|"candidate","reason":"...","scope":[...]}` selects the offered version; omitting it leaves a new body candidate. An active version remains while its revision is candidate. Agent-confirmed merges use the selected target, with no compulsory old-success regression list. Derivation, numerical checks, exhaustive constructions, boundary tests and future independent transfer report their actual scopes; none is a uniform prerequisite. `--used-entry-ids 3@2` freezes actual adoption. `validation` can record a historical compatibility review; this is not a real v2 use. Full fields: [induction.md](induction.md).

**Compatibility:** Historical claim/evidence/check input remains accepted. Embedded assertions and `--verify` remain audit records (`--verify` wins a reported conflict); their `verified`/`fact_checked`/`insufficient_evidence`/`refuted` states do not replace transfer checks. True unversioned old rows retain their original historical semantics; new writes use v2 candidate publication.

**Effect / recovery:** `--attribute-effect` records the agent's judgment and actual adopted version. `lifecycle_action:promote|demote|keep` works for legacy and versioned entries with a real use reference and reason. Pure maintenance returns `maintenance_only`, state differences and index results, without an induction action or empty receipt. Corrections update evidence validity, not publication; the agent manages utility. Explicit cost comparisons return selected executions' methods, conditions, acceptance and dimension values/provenance without finding a baseline or declaring savings. Read `changed`/`idempotent`, version, publication authority/scope, validation, execution references, knowledge delta and index sync. `--dry-run` writes nothing; `--force` only lifts the cold-archive veto. Deferred indexing can be retried; `rebuild-index` remains explicit repair. `--task` belongs to material, not induce.

### `orx induction-status [--task T] [--episode E]`

Read/recover required jobs, immutable receipt references, separate explicit-0/missing/invalid/legacy counts and known/unknown learning observations. `completed:true` means there are no pending required reviews; it does not assert that all hypotheses or rules were verified. The building runner consumes each `pending_id` with the SAME material/submission ID. See [hplus.md](hplus.md) for CLI/API callbacks, empty reviews and separate H+/NoH+/equal-budget random frozen-library experiments.

### `orx [--world-model URL::MODEL] predict-capability --operation JSON [--task t.json] [--bundle bundle.json] [--horizon TEXT] [--horizon-tasks N] [--budget JSON] [--task-id ID] [--episode ep1] [--timeout S]`

**Purpose / effects:** Optional `wm-ce/1` forecast of a proposed maintenance operation's later performance effects. Logs forecast/failure and real call usage. Operation specifies `induce|revise|reverify|retire`, strategy/target and config; prediction support does not guarantee an execution path for every type.

**Input / read:** Bundle supplies frozen recorded `execution_ids`; framework fixes identity, scope, targeting, horizon, and per-metric baselines. Model supplies expected changes (metric/unit/direction/value/baseline), learning cost, degradation risk, uncertainty, and verification conditions. Cost baselines are per dimension/unit; unlike currencies cannot be converted. No provider gives `contract_only`; a validated actual forecast is required for `valid`.

**Next:** Compare multiple operation predictions when choosing among them, or retain the forecast for later binding. This is unnecessary for direct induction or an online H+ already forecast by `predict-strategy`.

### `orx compare-capability --predictions ID[,ID...] [--horizon-tasks N] [--allow-quality-loss]`

**Purpose / effects:** Read-only comparison; no operation runs. Among compatible quantified savings without quality loss, recommends greatest cumulative saving over the declared horizon minus that operation's one-time predicted maintenance cost, in the same unit.

**Read / next:** Recommendation, `defer`, `insufficient_evidence`, or incomparable reasons. Missing horizon, incompatible units, no executable operation path, or no payback prevents ordinary ranking. `--allow-quality-loss` is an explicit comparison option, not evidence of benefit. Accept/reject explicitly; a recommendation is not execution.

### `orx accept-capability --recommendation JSON [--prediction ID] [--verify JSON] [--note TEXT] [--force]`

**Purpose / effects:** Execute the accepted offline operation on its own frozen scope and automatically bind the maintenance fact. `induce|revise` delegate to induction; `retire` retires the named target; unsupported types are refused.

**Input / next:** Induction/revision operations must carry the exact relation payloads in `operation.config.relations`; no relations means no statistical fallback. Retirement uses its named target. The operation never widens scope by rereading today's bank. Read the knowledge delta and `result.maintenance_binding`; do not repeat the operation or its binding. Evaluate future effects when qualified later tasks exist.

### `orx reject-capability --recommendation JSON [--prediction ID] [--reason TEXT]`

**Purpose / effects:** Record decline/defer and its reason. No maintenance operation runs or knowledge changes; the decision log is a write. Prediction remains unbound. Nothing further is required for that decision.

### `orx bind-capability --prediction ID [--adoption-action ACTION_ID]`

**Purpose / effects:** Bind a real maintenance fact when the operation ran outside `accept-capability`. Captures actual operation/end time, created/revised/retired entries, admission verdict, cost, and scope agreement. Idempotent; no model call. It cannot set `effect_verified`.

**When (stage-1):** after the REAL induction completes. New binary jobs bind automatically; explicit binding is a recovery path. An executed solve alone is not an induction fact; unexecuted claims stay pending.

**Online H+:** `--prediction sp_...` binds the REAL induction action from its original frozen gain trace and completed job (`prediction_source:"online_trace"`); no separate capability prediction is required. Stored maintenance forecasts report source `stored`. Evaluate later useful performance, not merely entry creation.

### `orx evaluate-capability --prediction ID [--tasks ID,...] [--paired JSON] [--allow-descriptive]`

**Purpose / effects:** Persist later-effect evaluation. Uses closed task-episodes whose actual work ran after the operation, under produced knowledge, within frozen targeting and outside its source evidence. The unit is task-episode, not prediction count; repeated predictions for one execution do not add independent tasks.

**When (stage-2):** triggered **later, on a rolling basis**, once a qualified later task-episode has closed (an unreached horizon stays `pending` — not a failure). Keep it bounded: evaluate the oldest ≤K claims per episode, or sweep at checkpoints.

**Read / next:** Horizon not reached stays `pending`. Paired treated/reference evidence must cite real tasks and compatible metric/unit; it supplies the observed difference. Only `observed_improvement` sets `effect_verified`; descriptive movement, insufficient evidence and refutation are separate. `--allow-descriptive` explicitly limits the conclusion. Final verdicts short-circuit repeats; pending results may be evaluated later. Online `sp_...` traces use the original claim/conditions, never a hindsight forecast. No extra call is needed after a final verdict.

## 9. Diagnostics and inspection

### `orx doctor`

**Purpose / effects:** Read-only solver (7 adapters), memory, pending staging, home, and retrieval-index health checks. `retrieval_index` reports configured backend, current/index counts, `missing`, `stale`, and `orphaned`; it builds nothing.

**Next:** Record unrecorded real attempts. Rebuild an unhealthy configured index when needed. No backend means structural fallback, not a missing semantic match.

### `orx contract [--kind strategy_outcome|capability_evolution] [--payload JSON]`

**Purpose / effects:** Build/inspect a versioned contract without provider calls. Strategy-outcome construction supports `--task`, `--spec`, `--benefit`, `--cost`, `--risk`; capability construction supports `--operation`, `--horizon`, `--horizon-tasks`, `--verification`.

```bash
orx contract --kind strategy_outcome --task t.json \
  --spec '{"action_type":"execute_strategy","strategy_id":"S01","scope":"attempt"}' \
  --benefit '{"kind":"solution_quality","metric":"normalized_objective_gap","unit":"1-gap","value":0.8,"baseline":{"kind":"conditional_stats","value":0.7}}' \
  --cost '{"expected":{"llm_tokens":1200},"expected_measured":["llm_tokens"]}' \
  --risk '{"events":[{"event":"task_failure","probability":0.2}]}'
orx contract --payload prediction.json
```

**Read:** Contract version and separate `provider_configured`, `service_available`, `prediction_made`. Building returns `contract_only` even with a provider; only a produced, validated forecast is `valid`. Unsupported versions exit 2; unversioned payloads use `legacy_view`, without inventing gains/risk/measurements. Legacy specs preserve params/budget hints and refuse unmappable task scope. See [world_model_contract.md](world_model_contract.md).

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts|evaluations|retention|capability [--task ID] [--strategy S] [--status candidate] [--episode EP] [--evaluation ID] [--prediction ID]`

**Purpose / effects:** Read stored layers. No solving, prediction, retirement, or migration. Use filters/key lookup instead of repeatedly listing history.

| Bank | Read |
|---|---|
| `experience`, `strategic`, `archive` | Facts, active knowledge, archived knowledge; strategic status/verification/provenance/prediction track |
| `actions`, `snapshots` | Logged actions and frozen execution context |
| `predictions` | Separate legacy, strategy, and capability generations; only strategy channel feeds strategy calibration |
| `evaluations` | Stored close-out evaluations; pending waits for scope, excluded is neither hit nor miss |
| `retention` | Calibration window, grace periods, archive caps and counts; evidence window is a separate scope |
| `capability` | Maintenance-fact binding and later-effect verification, separately |
| `texts` | Retained retrieval documents keyed by task ID/text digest; not claims or a learning bank |

`predictions --prediction ID` returns one record and resolved generation, not a history listing; `capability --prediction ID` returns its forecast/binding/effect state. `--evaluation ID` selects an evaluation. `--status suspect|dormant` finds retirement candidates without deleting them. `texts --task ID` lists that task's versions; no task filter lists all retained versions, including sources for unindexed records. Use each execution's `task_text_digest` to identify its version.

Targeted reads: `inspect --bank strategic --entry 3 [--rule-version candidate|2]` returns body versions and check/use/effect references; `inspect --bank experience --execution ex_ID` reads a full fact. `inspect --bank values --value REF` resolves a frozen body/fact/assessment after its source execution is evicted.

## 10. Corrections and store maintenance

### `orx exclude-execution --execution EXECUTION_ID --reason "..." [--superseded-by EXECUTION_ID]`

**Purpose / effects:** Explicitly withdraw an incorrect execution fact, retaining the row and correction audit. `source=excluded` removes it from evidence statistics, induction, retrieval, and calibration use; its vector is removed. `--superseded-by` is an explicit link to a corrected rerun.

**Next:** Derived knowledge feedback refreshes through induction; within-window calibration is republished automatically. Use `calibration --rebuild` for explicit repair outside that correction path. A real failed solve is still valid failure evidence: record/check it, rather than withdrawing it merely because the answer failed. Withdrawal differs from reference expiry under retention.

### `orx restore-execution --execution EXECUTION_ID --reason "..."`

**Purpose / effects:** Reverse an exclusion explicitly, restoring `source=executed` with both decisions retained. Calibration follows current facts under the same republish rule.

**Next:** Re-index with `rebuild-index --layer execution` when text retrieval is needed, and run induction maintenance to refresh derived feedback. Restoration does not automatically recreate its vector.

### `orx retire --entry NUMBER --reason "..."`

**Purpose / effects:** Irreversibly move the numbered knowledge entry to the cold archive and remove its vector. Recall cannot surface retired advice. Inspect suspect/dormant entries first; retirement is explicit, not automatic deletion on a miss. The number is kept on the archive card and is never reused. There is no undo command for this retirement.

### `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

**Purpose / effects:** Explicit bulk embedding/index repair after a first build, embedding-model change, deferred item sync, or restored execution. Ordinary writes refresh their own items; a rebuild is not a routine solve step and never changes source facts/claims.

**Read / recover:** Reports `dry_run`, `layers.{execution_evidence|strategic_knowledge}.{items,model_id,dimension,unindexable}`, and backend. A changed model invalidates the old vector space wholesale. Missing source text is `unindexable`, never invented. Without a backend, a real rebuild exits 2. Dry-run counts only: no embeddings, directories, or index writes. Recheck recall/doctor afterward when needed.

### `orx archive-calibration [--dry-run]`

**Purpose / effects:** Move out-of-window evaluation/prediction/context detail to `<home>/archive/calibration/calibration-NNNN.jsonl`. Registry tombstones remain online for idempotence/window lookup. Unchecked episodes stay online during late-check grace (`OR_CALIBRATION_LATE_CHECK_GRACE_DAYS`, default 30).

**Bounds / recovery:** Oldest archive files expire under per-file bytes (64 MB), total bytes (1 GB), or age (365 days): `OR_CALIBRATION_ARCHIVE_MAX_FILE_BYTES`, `OR_CALIBRATION_ARCHIVE_MAX_TOTAL_BYTES`, `OR_CALIBRATION_ARCHIVE_RETENTION_DAYS`. Restored payloads do not automatically rejoin calibration. Dry-run never migrates a legacy registry; it reports `pending_migration` with the explicit path (`calibration --rebuild` or a real archive pass). Routine close-out already invokes archive maintenance.

### `orx enforce-window [--window-episodes N] [--open-grace-days D] [--dry-run]`

**Purpose / effects:** Bound evidence by whole episodes (default 800, `OR_EVIDENCE_WINDOW_EPISODES`), ordered by registry close time. Evicts obsolete vectors and unreferenced text versions with facts; never solve sources. Idempotent/recoverable.

**Bounds / next:** Protects calibration-window episodes, unchecked episodes in late-check grace, and young unclosed episodes. Unclosed protection expires after default 30 days (`OR_EVIDENCE_OPEN_GRACE_DAYS`), reported as `evicted_unclosed`. Keeps an oversized single episode intact; episode count is not a byte guarantee. Expired evidence references do not refute knowledge. Close-out runs this last after publication/archive with a cheap count guard; force a pass after import when needed. Dry-run writes nothing; zero evictions is success.


## Input lifecycle and fixed-memory maintenance

- `orx context --context-id CTX --audit` reads the stored manifest, including after consumption/expiry. Ordinary reuse of an expired input returns exit code 2, never an empty substitute.
- `orx --home BASE fork-memory --destination NEW_HOME` creates one isolated test-question copy using SQLite backup and copies its fixed index/archive/artifacts. Repeat from BASE for the next question; retries keep the same question home.
- `orx --home COPY maintain-snapshots` is a read-only preview. `--apply --backup NEW_FILE` explicitly compacts recoverable legacy views; `--vacuum` separately requests physical reclamation. `--expire-open-days N` explicitly previews/applies inactive legacy-view expiry without deleting pending attempts or completing actions.

Exact lifecycle, protection rules, limits and copy-first commands: [snapshot_lifecycle.md](snapshot_lifecycle.md). Main-bank application is a separate user-initiated maintenance operation.
