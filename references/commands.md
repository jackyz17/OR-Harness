# Command reference

Use this page to look up syntax, outputs, and recovery steps. Follow [SKILL.md](../SKILL.md) for the solving workflow and [induction.md](induction.md) for review and rule management.

Commands print one JSON line to stdout: `{"result": {...}, "summary": "..."}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. Read the result state as well as the exit code. All commands accept `--home DIR`, defaulting to `$OR_HARNESS_HOME`, then `./or_harness_home`.

## 1. Common paths and command index

Use the same task file through preparation, planning, and execution. See [task inputs](modeling.md#start-from-the-shared-original-input) for the minimum fields and optional model or CIR.

| Purpose | Commands | Effects |
|---|---|---|
| Understand and discover | `profile`, `recall` | Read; recall may call an embedding provider |
| Score and select | `plan-next` | One shared context, one forecast per candidate, saved U selection; inspect reads existing forecasts |
| Execute, check, record | `execute`, `check-task`, `record` | Runs code; logs checks; stages and records facts |
| Close the episode | `close-episode` | Writes evaluations/calibration; performs retention |
| Optional inputs | `context`, `snapshot`, `budget` | Save prediction inputs or snapshots, declare a budget |
| Offline knowledge | `induction-material`, `induce`, `retire` | Material reads; induction/retirement change knowledge |
| H+ building handoff | `induction-status`, `induction-material`, `induce` | Review each pending H+ job after close-out. `induce` records the result in an induction receipt |
| Inspect and diagnose | `doctor`, `contract`, `inspect`, `calibration` | Read/build views; `calibration --rebuild` writes |
| Repair/report | `bind-strategy`, `amend-cost`, `action`, `exclude-execution`, `restore-execution` | Writes links, costs, actions, or correction history |
| Store/index maintenance | `rebuild-index`, `archive-calibration`, `enforce-window` | Explicit maintenance; `--dry-run` writes nothing |

## 2. IDs and shared rules

| ID or payload | Source | Consumer |
|---|---|---|
| Task / episode | Choose one task ID per logical problem and one episode label per solving effort; retries retain them | Commands accepting `--task` / `--episode` |
| Decision action | `plan-next`: `result.decision_action_id` | saved authorization; optional compatibility rejection |
| Candidate prediction | `plan-next`: `result.selected_prediction_id` / `result.plan.candidates[].prediction_id` | `execute --prediction` for the scored winner; independent IDs are inspection only |
| Execution | `execute`: `result.execution_id` | `check-task`, `record --from-staged`, correction commands |
| Execution action | `execute`: `result.action_id` | `bind-strategy --action`, `action --amend-cost` |
| Evidence IDs | `induction-material`: `result.material[].execution_id` | Evidence for the induction submission |

Keep these boundaries throughout the calls:

- A proposed method is a plan; an execution receipt reports what happened. A retrieval hit or a reuse citation does not establish successful transfer.
- Missing values are unknown. Cost comparisons require compatible units, scopes, and measured dimensions; estimates and lower bounds are not complete observations.
- Solver quality measures how well the model was solved. Task checks assess declared requirements of the original task. Knowledge verification covers the submitted assertions. These are separate judgments.
- See [prediction input identity](prediction_context.md#one-cir-for-the-whole-request-and-one-effective-input-identity) for the fields that must agree between planning and execution.

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

**Input:** `--candidate` filters by existing method IDs; it does not create evidence or prohibit a fresh method. `--top` caps returned material, not decisions. Modes apply to the structural recommendations: `none` clears that channel; `cases` reports statistics; `strategic` adds entries; `cost-aware` retains that view for experiment settings, without recall-side cost ranking. **Vector retrieval still runs with `--memory-mode none`**, so this flag is not a no-memory isolation switch. `--include-unverified` exposes unpublished candidates in both channels with their state labelled. A host that calls `profile` and `recall` in ONE request can pass the already-built profile object (Python `recall(task, profile=h.profile(task))`) so the task's CIR/structure is parsed ONCE instead of twice; a profile whose `problem_id` is a different task is refused (a cheap id check that does not re-parse).

**Read:** `recall` includes `result.profile`; a separate profile call is optional for structure/guidance inspection.

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

Structural rows carry `strategy_id`, `expected{quality,cost,failure_prob}`, evidence/refs, confidence, basis, cost-known/basis dimensions, and risk warnings. `knowledge` carries the entry's method/actions, applicability, support, and verification information where available. `recommendations[].knowledge.rule` exposes a short rule and its version. If a vector hit lacks the version, use `inspect --bank strategic --entry ENTRY` and read `result.entries[0].rule.view_version` before declaring adoption. For `conditional_stats`, expected fields are observed recounts, not a verified knowledge commitment.

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

**Recover / next:** Explicit reuse uses `ORHarness.predict_strategy_outcome(..., context=ctx, cir=cir)` with the same effective CIR. CLI `plan-next` builds its own shared context. Task/version/episode or known structural mismatches are refused; unknown structure is not a mismatch. `--no-persist` returns content without creating a later lookup target. Historical `--snapshot` builds use frozen knowledge/retrieval, exclude later-created memories, and mark unavailable reliability/evidence blocks missing; they cannot reconstruct history from today's banks. See [prediction_context.md](prediction_context.md).

## 4. Plan and inspect predictions

### Candidate format

`plan-next --candidates` accepts an inline JSON list or a path to one. The list may contain one or multiple candidates. A minimal described candidate is:

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

### Stored forecasts and their conventions

`plan-next` saves one `wm-so/1` forecast per candidate together with the score
and execution permission. Read `result.selected_prediction_id` and execute that
original P. View an existing candidate forecast with:

```bash
orx inspect --bank predictions --prediction P
```

Inspection makes no provider call. `result.prediction.kind` names the generation;
`result.prediction.prediction` contains its stored payload. It exposes status, candidate,
G/C/R/H+, notes and trace. The Python service `ORHarness.predict_strategy_outcome`
remains available for explicit context reuse and contract inspection; standalone
API forecasts do not authorize execution.

Set any required benefit convention before planning. See [contract fields](world_model_contract.md) for metric names and [G attribution evidence](episode_closeout.md#g-attribution-evidence) for observations.

**Recover / next:** A failed or unsupported forecast retains its status,
notes, failure details and measured call usage. Inspect that saved prediction;
repair the candidate or provider response and call `plan-next` again when no
eligible U is returned. A candidate is predicted once per plan without hidden
retries or default forecasts. See [strategy_outcome.md](strategy_outcome.md).

Saved forecast diagnostics retain these recovery hints:

| `failure.kind` | Recovery |
|---|---|
| `truncated` | Increase `--wm-max-tokens` / `OR_WM_MAX_TOKENS`; consider thinking's shared output budget (`OR_WM_ENABLE_THINKING`) |
| `wrong_top_level` | Fix the payload/prompt; expected JSON object |
| `empty_response` / `unparsable` | Inspect endpoint output/format |
| `timeout` / `network_error` | Retry deliberately or adjust `--wm-timeout` / `OR_WM_TIMEOUT` and planning time budget |
| `unusable_payload` | Fix fields named in validation notes |

Unknown `finish_reason` remains unknown.

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode E] --candidates specs.json [--cost-scales JSON_OR_FILE] [--horizon 1] [--max-calls N] [--prediction-mode M] [--time-budget S]`

**Purpose / effects:** The normal decision entry accepts one or more candidates,
freezes one shared context, calls the world model once per candidate, saves
G/C/R/H+ and U, and selects the eligible highest-U candidate with stable ties.
It does not execute the solver. Read `result.selected_prediction_id`,
`result.selected_score`, `result.plan.candidates[]`, limits and planning spend.
When no selection is available these keys are null and the reason is explicit.

**Settings:** `--cost-scales '{"tool_calls":30}'` supplies a declared resource scale. The global `--execution-context runner.json` takes precedence over `OR_EXECUTION_CONTEXT` (inline JSON, path, or `@path`) for declared execution host, model, interface, and phase. See [score calculation](strategy_outcome.md#3-score-one-or-more-candidates) and [prediction inputs](prediction_context.md) for how these settings apply.

**Next:** Pass the selected P to `execute --prediction P`. For a repaired attempt, obtain a fresh scored selection.

### `orx choose-next --decision ACTION_ID (--prediction P | --chosen spec.json | --rejected) [--note "..."]`

Compatibility/rejection record, with no solver or model call. Accept only the
scored winner from this decision; execute reuses the choice record for its one
actual adoption. A manual choice cannot override saved U permission. Rejection
records the reason and prevents executing that decision. Historical choices
remain readable.

## 5. Execute and check

### `orx execute --task t.json --prediction PREDICTION_ID --code solve.py --workspace DIR [--episode ep1] [--method JSON] [--adapted-from ex_a,ex_b] [--adaptation TEXT] [--used-entry-ids ENTRY@VERSION]`

**Purpose / effects:** Run the chosen code, log an action, capture task text/version and observed configuration, and stage every resulting attempt. Staging is not recording; use `record --from-staged` afterward. No automatic second solve is started.

**Input / binding:** A selected, unused, finite-U prediction from `plan-next` is required. Bare execution and independent forecasts are refused before execution. With `--prediction`, the frozen candidate supplies strategy, solver, configuration identity, and episode; contradictory explicit values, changed problem identity, or an already-claimed prediction are refused before the run. The action/link is persisted before execution. Adding a post-strategy model does not change problem identity. Bare strategy/solver arguments and an independent API forecast do not authorize formal solving. `--method` describes candidate intent; it cannot replace a scored selection.

**Sandbox:** Code must live inside the workspace, which becomes its working directory. Blocked imports include `subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil`, and `pathlib`; only `os` / `os.path` are allowed from `os`, with operations such as `chdir`, `walk`, and `remove` refused. In-process solvers such as `highspy` and `ortools` are allowed; subprocess-based PuLP/CBC cannot run. Constraint labels must be `C1`, `C2`, … for L2 cross-reference. POSIX resource limits and a whole-script wall-clock timeout apply.

**Result file:** Write with literal `open('result.json', 'w')` inside the workspace. Prior output is removed; stale output is refused. `status` is required; `optimal`/`feasible` require a finite `objective_value`. Report actual `objective_bound`, `mip_gap` and `runtime_seconds` when available, and `variables` whenever task checks need solution values. Missing measurements remain unknown. Legal statuses are `optimal|feasible|infeasible|unbounded|timeout|error` (case-normalized). Use the actual solver status: a feasible incumbent does not establish optimality. Boolean statuses are errors.

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

`--used-entry-ids` declares the actual knowledge versions this attempt ADOPTED (e.g. `3@2,5@1`; `''` records "adopted none"). Bare entry numbers remain accepted for compatibility; read and name the version explicitly in ordinary use. This is a DECLARATION, distinct from recall (`last_consulted_at`) and from `--adapted-from` (cases you read): only declared entries have a forward check computed, so a mere recall never counts as a successful use. The adoption is stored under `execution_features.used_entries`; an unknown number is kept as a citation, and the flag never rewrites a value the record already carries.

**Cost:** The executor observes latency and solver runtime with provenance. Supply whole-attempt tokens/tool calls from a host report; hand-typed defaults are estimates. Tool calls have a provable sandbox lower bound. Retries are measured zero only when no earlier attempt exists for this task/episode/strategy; otherwise remain unknown until supplied. A static-policy rejection has no measured solving cost. See recording/cost rules below.

**Recover / next:** Run `check-task`, then record the staged attempt, including failures. Prediction precondition errors exit 2 before execution. If the executor fails after starting, retain the staged failure/cost. A repair proposes a fresh candidate through `plan-next` in the same episode. Inspect `prediction_binding` (`bound`, `config_observed`, `config_unknown`) as evidence; recovery association cannot replace saved execution permission.

### `orx check-task <execution_id> [--check JSON] [--episode ep1]`

**Purpose / effects:** Verify declared requirements of the original task against a staged or recorded result. Writes a `verify` action and `execution_features.task_check`; no solver or prediction call. This is separate from solver-side model quality.

**Before submitting the answer:** Build `--check` from the original requirements, inputs, and execution. Evaluate constraints from the returned solution. Use these checks and the execution outcome to decide whether to retry.

| Check key | Meaning |
|---|---|
| `integer` | Optional named `variables`, optional tolerance (default `1e-6`); otherwise checks recorded variables |
| `recompute_objective` | Coefficients, optional constant/tolerance; recompute from solution variables |
| `semantic_probe` | One or more `{path, equals|min|max|in}` probes over record fields |
| `intent` | `relaxation|intermediate`: intentionally not the final task answer |

**Read:** `result.state`, `report.conclusion`, `report.checks[]`, `report.scope.unchecked[]`, and optional `reflection_material` / `next`.

- `passed`: the declared checks held. Read `scope.unchecked[]` for requirements outside that verdict.
- `failed`: a declared check ran and failed. Keep the execution, solver measurements, and real cost; it cannot support task-success claims. For prediction calibration, solver quality stays its own measurement and task completion uses the separate check channel.
- `insufficient`: a needed value/basis/path is missing or the result is unusable; neither pass nor refutation. Missing probe paths name available fields.

**Recover / next:** Record every attempt. Use `reflection_material` to inspect failures and `result.report.scope.unchecked[]` to identify missing coverage. Recheck the same EX when you can fix the check or supply missing evidence. For a solve retry, use the same episode and obtain a new selection via `plan-next`. To record a sourced G assessment in `--check`, see [G attribution evidence](episode_closeout.md#g-attribution-evidence).

**After submitting the answer:** Benchmark evaluators compare the locked result against the reference outside the solve and retry loop. For offline reference evidence, `reference_objective` compares a numeric objective (default `tolerance` is `1e-6 * max(1, abs(reference))`), and `reference_status` compares a solver status. Declare `reference_source` (`bench_declared`, `independent`, or `self_derived`) and `reference_version` when available. Unspecified provenance is treated as self-derived. For using post-answer evidence in a review, see [induction materials](induction.md#read-material-and-choose-comparisons).

### `orx bind-strategy --prediction ID --action ACTION_ID`

**Purpose / effects:** Manual/recovery association for an action already run; same identity rules as automatic binding, no model call or re-billing. Idempotent re-binding re-evaluates comparability. This is a compatibility/recovery entry, not a normal solve step: it cannot grant U permission, select a losing prediction or create an execution that did not happen.

**Read / recover:** Checks action/task/episode/strategy/solver and observed execution config. Known `binding_mismatch` and missing `binding_unknown` are distinct. Missing receipts are caveats, not automatic whole-sample exclusions: an unreported time limit blocks no comparison by itself; a different solver blocks outcome dimensions while keeping actual cost. Read `config_observed` / `config_unknown`; predictions never fill missing observations. Close-out determines eligibility field by field. See [episode_closeout.md](episode_closeout.md).

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

**Purpose / effects:** Append an execution fact, or record a confirmed host stop. Prefer `--from-staged` to retain the original payload. `--execution` accepts a bare record, execute envelope, or legacy one-level form. Discarding pending material is a separate explicit mode, not a substitute for recording a real failed attempt.

**Execution record:** Applies cost backfill (replace by default), freezes quality checks against the entries this run **DECLARED it adopted** (`--used-entry-ids`) onto `execution_features.quality_feedback`, and compares measured cost with the frozen historical prediction under `cost_feedback`. A check exists ONLY for a declared adoption, and a hit/miss is computed ONLY against a DECLARED prediction (an entry with no declared prediction records the observation with `hit=null`, never a default-interval hit). Recording neither interprets the method nor promotes/demotes knowledge; later induction replays feedback.

`--method` declares the plan; `--method-actual` declares observed performance. Existing record values, including script receipts, are not overwritten. `--used-entry-ids` declares the actual knowledge versions this attempt ADOPTED (e.g. `3@2,5@1`; `''` records "adopted none"), tying them to this attempt's result; a value the record already carries is never overwritten. Explicit `execution_features.contrast` marks contrast evidence; there is no per-record retention label.

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

`record --usage-file` and `amend-cost --usage-file` also accept `or-host-usage/1` reports with window/counter/call evidence. They store wider scopes beside the execution, preserving its attempt cost. With the API, pass `host_usage=report` to `report_action(...)` or submit it in `action --report ... --outcome '{"host_usage": ...}'`. See [C observations](episode_closeout.md#c-observations) for supported measurements and their scopes.

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

**Purpose / effects:** Close the episode with its actual terminal state, evaluate bound predictions, publish calibration, and return scheduled induction IDs. Pass `--finish-action` to link a recorded user-facing task end. See [close-out and calibration](episode_closeout.md#2-closing-an-episode) for evidence and retention.

**Read:** `closeout`, per-field evaluations and eligibility, `task_checks`, `cost_completeness_warnings`, and `result.induction_schedule.pending_ids`. Finish scheduled reviews with [the returned IDs](induction.md#take-a-required-review).

**Recover / next:** A pending close lists `unfinished_actions`. Resolve them, then close again. A repeated close returns the stored record and current review IDs. `recovered_publication:true` identifies a repaired publication.

### `orx calibration [--min-samples N] [--rebuild]`

**Purpose / effects:** Read published `wm-calib/3`; `--rebuild` explicitly recomputes and republishes for migration/repair. Default window: newest 50 closed task-episodes (`OR_CALIBRATION_WINDOW`). New prediction contexts receive the published summary automatically.

**Read:** Groups separate model identity, metric, unit, and scope. Statistics include signed benefit error, cost log ratio, interval coverage/width, per-event Brier, and predicted probability versus occurrence over the same scored denominator. Separate `occurrence` reports its own observation-unit counts/denominator; do not mix those rates. Below minimum support: `insufficient_evidence`, `reliability:null`. `applicability:"global_diagnostic"` does not imply per-strategy reliability or guaranteed prediction improvement. Legacy knowledge-prediction statistics are separate.

## 8. Offline induction

### `orx induction-material [--induction-id J] [--strategy S] [--task T] [--episode E] [--limit N] [--cursor C] [--related-top-k K] [--purpose modelling|implementation] [--query TEXT] [--executions EX1,EX2] [--view compact|detail]`

**Purpose / effects:** Read current material for a required review J. `--purpose modelling` prioritizes constraint relationships, and `--purpose implementation` emphasizes implementation and repairs. Use `--query` or `--executions` to select comparisons.

**Read:** `material[]`, `related_history.executions[]`, `existing_knowledge`, `comparison_table`, and `budget.next_cursor`. Use `--cursor` for more controls or `--view detail` for expanded diagnostics. See [choosing material](induction.md#read-material-and-choose-comparisons) for evidence scope and budget.

### `orx induce [--induction-id J] [--review JSON_OR_FILE] [--strategy S] [--relation JSON] [--dry-run] [--force] [--note TEXT] [--verify JSON] [--attribute-effect JSON]`

**Purpose / effects:** Submit a review with `--review`. For a new or revised rule, include `--relation` and the same J. See [full review examples](induction.md#complete-review-submissions). Use `--relation` for existing-body maintenance and `--attribute-effect` for actual adopted use. See [version use and utility](induction.md#existing-entry-use-and-utility).

**Read:** `induction_receipt`, `relations[].publication`, `changed`/`idempotent`, `committed`, and `index_sync`. Maintenance-only calls return `maintenance_only` and the affected version or status.

**Options:** `--dry-run` previews without writing, and `--force` lifts the cold-archive veto. `--verify` records scoped assertions and takes precedence over embedded verification on a reported conflict. Use `--task` on `induction-material` and J on the review submission.

### `orx induction-status [--task T] [--episode E]`

Read `result.completed`, `result.pending_ids`, `result.signal_counts`, and `result.observation_counts`. Use each pending ID to [resume its review](induction.md#take-a-required-review). See [review result fields](induction.md#read-publication-completion-and-learning-separately) for how completion and learning are reported.

## 9. Diagnostics and inspection

### `orx doctor`

**Purpose / effects:** Read-only solver (7 adapters), memory, pending staging, home, and retrieval-index health checks. `retrieval_index` reports configured backend, current/index counts, `missing`, `stale`, and `orphaned`; it builds nothing.

**Next:** Record unrecorded real attempts. Rebuild an unhealthy configured index when needed. No backend means structural fallback, not a missing semantic match.

### `orx contract [--kind strategy_outcome] [--payload JSON]`

**Purpose / effects:** Build a strategy-outcome contract locally with `--task`, `--spec` and optional `--benefit`, `--cost` and `--risk`. Use `--payload` to inspect a saved prediction payload.

```bash
orx contract --kind strategy_outcome --task t.json \
  --spec '{"action_type":"execute_strategy","strategy_id":"S01","scope":"attempt"}' \
  --benefit '{"kind":"solution_quality","metric":"normalized_objective_gap","unit":"1-gap","value":0.8,"baseline":{"kind":"conditional_stats","value":0.7}}' \
  --cost '{"expected":{"llm_tokens":1200},"expected_measured":["llm_tokens"]}' \
  --risk '{"events":[{"event":"task_failure","probability":0.2}]}'
orx contract --payload prediction.json
```

**Read:** Contract version and separate `provider_configured`, `service_available`, `prediction_made`. Building returns `contract_only` even with a provider; only a produced, validated forecast is `valid`. Unsupported versions exit 2; unversioned payloads use `legacy_view`, without inventing gains/risk/measurements. Legacy specs preserve params/budget hints and refuse unmappable task scope. See [world_model_contract.md](world_model_contract.md).

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts|evaluations|retention [--task ID] [--strategy S] [--status candidate] [--episode EP] [--evaluation ID] [--prediction ID]`

**Purpose / effects:** Read stored layers. No solving, prediction, retirement, or migration. Use filters/key lookup instead of repeatedly listing history.

| Bank | Read |
|---|---|
| `experience`, `strategic`, `archive` | Facts, active knowledge, archived knowledge; strategic status/verification/provenance/prediction track |
| `actions`, `snapshots` | Logged actions and frozen execution context |
| `predictions` | Stored strategy predictions and historical prediction records |
| `evaluations` | Stored close-out evaluations; pending waits for scope, excluded is neither hit nor miss |
| `retention` | Calibration window, grace periods, archive caps and counts; evidence window is a separate scope |
| `texts` | Retained retrieval documents keyed by task ID/text digest; not claims or a learning bank |

`predictions --prediction ID` returns one stored record and its prediction kind. `--evaluation ID` selects an evaluation. `--status suspect|dormant` finds retirement candidates without deleting them. `texts --task ID` lists that task's versions; no task filter lists all retained versions, including sources for unindexed records. Use each execution's `task_text_digest` to identify its version.

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
