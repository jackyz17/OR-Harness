# Command reference

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

## `orx profile --task t.json [--code solve.py] [--cir cir.json]`

The single analysis entry: CIR validation + modeling guidance + problem profile + derivation report, in one call. It serves two purposes: it helps you understand the problem structure, and it produces the structural key that retrieves comparable evidence at `recall` time. A task **without** a `model` field is a normal state — strategy selection needs the task text, the CIR, and the profile; the `model` (an intermediate representation) is written only AFTER the strategy is chosen, and re-running `profile` then adds the CIR ↔ model cross-check.

**CIR side** (`result.coupling`): when the task carries a `coupling` field (or `--cir` is given), the CIR (Coupling-Aware Intermediate Representation — an explicit, inspectable set of entities, decisions, constraints, and relations) is validated and returns `modeling_guidance` with concrete coupling implications for the model you will write. When both CIR and `model` are present, the framework cross-checks them and surfaces `cir_warnings`. Without a `coupling` field, `coupling.cir=null` with a prompt message (not an error).

CIR schema (all `kind`/`type` fields are free-form strings — domain-general, no hard-coded vocabulary):

```json
{
  "coupling": {
    "entities":     [{"name": "R1", "kind": "resource", "attrs": {}}],
    "decisions":    [{"name": "x", "kind": "production", "indexes": ["i", "t"]}],
    "constraints":  [{"id": "C1", "kind": "capacity", "expr": "sum(x) <= limit"}],
    "relations":    [{"source": "x", "target": "R1", "type": "uses_resource",
                       "evidence": "semantic", "detail": "..."}],
    "coupling_groups": []
  }
}
```

Relation `evidence` levels: `structural` (from constraint-variable co-occurrence only — never a semantic claim), `semantic` (supported by entity/constraint semantics), `declared` (agent-asserted). Co-occurrence alone produces generic `depends_on` edges; a semantic relation like `uses_resource` requires additional evidence (resource-kind entity + capacity constraint).

The `modeling_guidance` entries each carry `{type, members, resource, implication}` — e.g. `shared_bottleneck`: "Multiple decisions (x, y) consume the same resource (R1); ensure one aggregate capacity constraint covers all relevant decisions."

**Profile side** (`result.profile` + `result.derivation`): coupling derivation priority — CIR (task JSON `coupling` field, best — derived from its relations/indexes) > task JSON `model` field (measured from declared constraints) > structured `spec` fields > your `annotations.coupling` supply. `semantic_coupling` is never derived. The `derivation` report carries per-dimension value/origin/notes, `model_verification` (L1+L2 issues) when a model is given, `coupling_warnings` when a supplied value contradicts the structural derivation across a bin boundary, and `cir_warnings` when both CIR and model are present. The retrieval signature and the modeling guidance come from the SAME structure, not two parallel lines.

## `orx recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--code solve.py]`

Recalls accumulated experience for the task's problem signature. Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: **published** Strategic Knowledge entry → conditional statistics → **no evidence** (`evidence="no_memory"`, `score=-inf`, `confidence=0`). An entry that is not published (an unverified candidate, or one whose admission check was `refuted` / `insufficient_evidence`) is skipped, so recall falls back to the statistics — which is what "we have a candidate but no knowledge yet" should look like. `--include-unverified` is the offline/inspection view that returns such an entry with a warning.

When no experience exists for any strategy, all candidates return with `evidence="no_memory"` — the catalog still provides the strategy vocabulary (applicability, actions, fallback, solver family) but makes no quality/cost/risk claims. Pick based on structural fit and your own judgment.

Result: `result.recommendations[]`, each `{strategy_id, name, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before). Note: when `evidence="conditional_stats"`, the `expected` fields report OBSERVED means (a recount from the Evidence Bank), not a knowledge commitment — no interval, no calibration track, no lifecycle.

`--memory-mode`: `none` (no memory consulted; all candidates return `no_memory`) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

## `orx execute --task t.json --strategy S04 --code solve.py --workspace DIR --solver NAME [--verification basic]`

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector with `llm_tokens=0` UNMEASURED — that dimension is yours to backfill, `cost_measured` mask, `solver_runtime_provenance`, `execution_features` with solver diagnostics if the solver reported any, and `cir_snapshot` = the CIR that was actually solved when the task carries a `coupling` field). **Nothing is recorded yet.**

## `orx predict --task t.json --strategy S [--code solve.py]`

Pre-execution cost expectation for one (task, strategy), with explicit provenance: `source=entry` (a matching Strategic Knowledge entry), `stats` (conditional statistics over attempt-scope evidence — a recount, with `support_n` and per-dimension `support_per_dim`), or `unknown` (no usable evidence: no data, task-scope data only, or clearly different scale — expected cost is `null`, NEVER a default zero presented as cheap). Pass the printed snapshot to `orx record --prediction` so feedback compares against the prediction actually used.

## `orx record --execution <json|path> | --from-staged <id> [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment] [--prediction <json>] [--retain-reason contrast] | --discard-staged <id>`

Appends the fact to the Execution Evidence Bank, then runs the automatic chain: cost backfill (replace by default — idempotent, never double-counts) → quality checks against matching entries, frozen ONTO THE FACT (`execution_features.quality_feedback`: the interval in force, the observed quality, hit/miss — only the strategy that actually ran is checked, attempt scope only) → cost feedback against the record's frozen pre-execution prediction snapshot (same strategy, same scope, both sides measured; written into `execution_features.cost_feedback`) → C1–C6 induction-hint checks (C4 detects cross-execution recovery chains automatically: a failed attempt under one solver followed by success under another).

**Recording never changes knowledge.** No entry is promoted, demoted, or woken here — the frozen checks are replayed by the next `orx induce`.

`--retain-reason <free text>` explicitly marks this episode as representative evidence, reserved for future compaction policies. When omitted, any mark the record already carries is preserved — no automatic marking is performed.

Result: `result.{execution_id, recorded, prediction_checks[], cost_feedback?, induction_hints[]}` and, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand). Always backfill `llm_tokens` here — it is invisible to the sandbox. A task's full cost is the sum of its recorded attempt-scope records (`api.task_cost_summary`): every attempt charged to the strategy that actually ran it; retries = sum of per-attempt NEW retries; end-to-end latency is unknown unless you supply explicit task timing (never inferred by max or sum).

## `orx induce [--strategy S | --all] [--rebuild] [--dry-run] [--force] [--note TEXT] [--verify JSON]`

Consolidates facts into Strategic Knowledge (your explicit call — hints never auto-induce). This is the ONLY place knowledge changes: it (i) builds or refreshes the claim of each (family, structural cell, strategy) evidence set — its applicability is read off the supporting records (the cell its evidence occupies, never a cross-sample span that could stretch across incomparable regions); and (ii) replays the frozen checks recorded on the facts, reporting what it did under `result.revisions` (promotion at ≥5 checks / ≥70% hits **with a verified claim**, demotion at 3 consecutive misses, dormancy wakeup). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). One evidence set owns exactly one claim: re-inducing refreshes it rather than restating it (dormant entries included, so a woken claim keeps its id). `--rebuild` re-induces the whole Strategic Knowledge Bank from currently retained evidence (cold archive preserved); the result may differ from the previous bank — exact reconstruction is not a requirement. New entries inherit `strategy_type`, `actions` and `fallback_strategy_id` from the catalog vocabulary (harness-supplied values are never overwritten). `--dry-run` never writes — including under `--force` (a rehearsal does not lift a cold-archive veto).

**Independent evidence gate (creation only).** Creating a claim takes at least 2 supporting executions from ≥2 distinct `task_id`s: repeating one task is repetition, not reproduction. Every count comes from the same facts — executed, attempt-scope, in the target's structural cell — so a task-scope total can never stand in for an independent attempt. When only one task is behind the evidence the call creates nothing and returns `verification {tasks, required_tasks}` plus a `skipped` reason — the evidence is NOT discarded (recall still answers from it as `conditional_stats`), and refreshing an entry that already exists is never blocked, no matter how repetitive the new evidence is.

**`--verify JSON` publishes the candidate.** Admission and creation are separate: the gate above decides whether evidence may become a *candidate*; this decides whether the candidate's claim may become *published knowledge*. See [induction.md](induction.md#admission-verification-offline-what-makes-a-claim-published) for the payload and the three checks (`rule` / `repair` / `cost_saving`). The framework computes the verdict from the executions you supply; without it the entry is `unverified` and **is not published** — recall answers from `conditional_stats` and `predict_cost` will not quote it.

What the verdict requires, concretely: a declared criterion the framework can evaluate itself (`reference_status`, `reference_objective`, `semantic_probe`) or a genuinely independent comparison — **feasibility alone is a precondition, not a check**, and a bare `semantic_ok` boolean is recorded as `agent-declared` but cannot carry a verdict. Every execution you pass is evaluated (a counterexample anywhere in the batch refutes, whatever the order), the evidence must be for THIS strategy and family, and a comparison must be like-for-like (same task, same measurement scope for costs; a repeated execution id is not an independent comparison). Failed executions yield `insufficient_evidence` (not `refuted`): "we could not check it" is not "we checked it and it failed".

Each element of `result.results` reports `created` (entry id) / `updated` (entry id) / `skipped` (human-readable reason: fewer than 2 executions, needs independent evidence, cold-archive veto, restatement, or `recorded as an unverified candidate`), plus `verification` when a check ran — read the reason, it tells you what the memory is still missing.

`--note "TEXT"` (optional, repeatable, you phrase it): free-text applicability notes attached to the entries this call creates or refreshes. They are kept for the reader and shown by `inspect`; they never enter scoring — the framework does not pretend to verify a sentence.

## `orx inspect --bank experience|strategic|archive|actions|snapshots [--task ID] [--strategy S] [--status candidate] [--episode EP]`

Queries a memory layer. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids). `actions` / `snapshots` query the world-model substrate (below): the unified action log and the frozen belief snapshots.

## `orx snapshot --task t.json [--episode ep1]`

Freeze and persist the current **belief state** for a task (world-model M1 substrate): H (value-copied verified/legacy/unverified knowledge refs + experience counts + tool config), P (profile + task digest + CIR/model digests + the problem-relevant task payload when no `task_ref` exists), X (the episode's accumulated progress — selected plan, model artifact, current solution, verification evidence — each field labelled with `provenance` (observed vs agent_reported) AND `epistemic` (fact/inferred/unknown), two orthogonal dimensions), B (declared budget + consumption), and a coverage view (cell statistics + layered knowledge, no composite capability score). Snapshots are frozen by deep-copy isolation: later bank writes never change what a snapshot meant. Within an episode, each action's post state merges into the accumulated progress, so the next action inherits what earlier actions established.

## `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Report an action **you** performed (TYPE ∈ model/select_strategy/verify/finish_task) — the library did not execute it; the record is labelled `source=agent_reported` and a missing pre-state is marked `pre_snapshot_missing`, never fabricated. Explicitly supplied cost dimensions (including an explicit zero) are marked measured. `--amend-cost` backfills an action's cost (replace semantics, idempotent).

## `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

The honest budget view for a task/episode: consumption over ALL real action costs — recorded AND staged-but-unrecorded executions (deduplicated by execution_id) plus own-cost actions (a macro's reference cost is never re-counted; its own additional spend is). Status: `exceeded` (a measured dimension over the limit; a declared latency budget is judged per-attempt), `ok` (every declared dimension measured and within), `unconfirmed` (known spend within limits but a declared dimension is unknown — NOT confirmed within budget), `no_budget_declared`. Executions of the task with no episode linkage (pre-M1 records) are reported separately as `unattributed` and never charged to a fresh episode. Declarations persist in the store, so separate CLI invocations see the same budget. Hypothetical actions are excluded from every real consumption view.

`orx execute --episode ep1` and `orx induce` automatically record their actions in the unified log: execute as a macro (`pre` snapshot before execution, `post` after, `rollup=reference`, linked to the execution id); induce in the **maintenance scope** (`__maintenance__` / `maint_<ts>`) with a real pre/post knowledge state, verification results, a knowledge delta, and a business result that separates `created` (verified) from `created_unverified` (a candidate is NOT knowledge growth), `updated` / `revised` / `refused` / `unchanged`. A crashed induction still leaves a `failed` action with its pre state. Dry-run persists nothing.

## `orx [--world-model URL::MODEL] predict-outcome --task t.json --action-spec spec.json [--episode ep1] [--parent-action ACTION_ID]`

**World-model outcome prediction (M2, shadow mode).** Ask a configured world model for a structured prediction of ONE candidate action's consequences, from the frozen pre-action state. The candidate is an `ActionSpec` JSON (`{"action_type": "execute_strategy", "task_id": ..., "strategy_id": "S01", "solver": "highs", ...}`) — a hypothesis, NOT a recorded action. The prediction carries: expected execution status / feasibility / quality / failure risk / per-dimension cost, predicted successor state changes (hypothetical — never written to real state), the model's self-reported confidence (**uncalibrated**), its claimed evidence basis, and fields it explicitly declined to predict.

- **Configuration boundary**: `--world-model BASE_URL::MODEL` (OpenAI-compatible endpoint; API key from `$OR_WM_API_KEY`). Credentials never persist. Without the flag, the command returns an explicit `not_configured` error — and NO other command ever invokes a model.
- **Shadow discipline**: the prediction changes NOTHING. `recall`, `predict`, `execute`, `record` behave identically whether or not you predict. You remain the decision-maker.
- **Call cost**: the model call's own spend (tokens/latency from provider usage) is recorded on the prediction and, with `--parent-action`, charged to that action's own cost — separate from the PREDICTED cost of the target action.
- **Timing**: predict BEFORE executing. The input snapshot is frozen at prediction time; later bank changes never rewrite it.

## `orx bind-outcome --prediction ID --action ACTION_ID`

Bind a prediction to the real action that ran, then compare. Type/strategy/solver are checked: a mismatch (you predicted strategy A, executed B) is recorded and NOT scored — no counterfactual truth is fabricated. The comparison covers only fields both sides define: status category, feasibility, quality (when the execution produced a solution), and cost per dimension (both sides measured — the same both-sides-measured discipline as cost feedback). Missing comparisons are listed with reasons. The feedback is APPENDED to the frozen prediction — the original is never modified, and re-running the comparison is idempotent (the model is never re-invoked). Online comparison records facts and errors only; knowledge updates still go through explicit offline induction with verification.

Query predictions with `orx inspect --bank predictions [--task ID]`.

## `orx [--world-model URL::MODEL] plan-next --task t.json [--episode ep1] [--candidates specs.json] [--horizon 1|2] [--max-calls N]`

**Bounded next-step planning (M3).** Compare a small set of candidate actions by their PREDICTED consequences and get a suggested first step. The decision:

1. freezes ONE root snapshot for the whole comparison (all candidates see the same state);
2. predicts each root candidate's first-step consequences (default ≤3 candidates, ≤6 model calls total);
3. with `--horizon 2`, builds a HYPOTHETICAL successor state from each first prediction's `state_changes` and predicts the continuation FROM that successor (a genuine state-conditioned two-step rollout — never two independent root predictions);
4. scores each path as `U = alpha*Q_terminal − beta*C_path − gamma*R_terminal` (terminal quality / incremental predicted cost on the common measured dimensions / terminal failure risk — longer paths never win by accumulating quality terms; step risks are never summed or multiplied);
5. suggests the FIRST step of the best path.

- **Candidates**: `--candidates` (your own ActionSpec list, recommended when you have domain hypotheses), or the catalog vocabulary filtered by applicability and available solvers. Without memory, candidates carry no fabricated performance claims — consequences come from the world model.
- **Hard bounds**: candidate count, horizon (1–2), `--max-calls`, and a wall-clock budget. Exhaustion truncates with an explicit reason — never a silent partial answer. The real planning spend (the model calls) is charged ONCE to the decision action and reported in `planning_cost` — sunk, never part of any path's score.
- **A suggestion is not a selection**: `plan-next` never writes `X.selected_plan` and never executes. Only `choose-next` does.
- **Honesty**: missing quality/cost/risk predictions are reported per path under `incomparable` (unknown never auto-wins); a second step the first prediction cannot support (no incumbent) is truncated and marked `conditional_unsupported`; with an undeclared or partially-unknown budget, `budget_confirmation` is `unknown`/`unconfirmed` — never claimed "within budget". An already-exceeded real budget stops planning before any model call (`status=fallback`).
- **Requirements**: a configured `--world-model` provider. Without one every path is `not_configured` and no suggestion is made. Planning supports `execute_strategy` candidates; other action types are reported as not plannable.

## `orx choose-next --decision ACTION_ID [--chosen spec.json | --rejected] [--note "..."]`

Record YOUR explicit choice after a plan: accept the suggestion, pick another candidate (a deviation, recorded with its reason), or reject all. Only this call writes `X.selected_plan`; the choice itself produces no execution quality. Then execute the step with `orx execute`, record the result with `orx record`, and bind the executed step's prediction with `orx bind-outcome`. Re-plan from the new real state afterwards — the old plan stays as the suggestion of its time.

## `orx [--world-model URL::MODEL] assess-induction [--bundle bundle.json | --candidates-only] [--workload forecast.json]`

**M4 offline maintenance assessment.** Scans the banks for induction candidate bundles (frozen evidence: exact execution IDs, tasks, statistics, trigger reasons — new claims need ≥2 executions from ≥2 tasks; revisions reference the existing entry and its state at bundle time), then asks the world model to predict the INDUCTION action's consequences: candidate formation probability, expected reuse benefit, generalization risk, and the resulting claim's quality/cost. The recommendation (`induce_new` / `revise` / `defer` / `insufficient_evidence`) comes with a value decomposition (`net_value = α·benefit − γ·risk`, workload forecasts scale the benefit term) and the real assessment cost, charged once to a maintenance-scope action (`__maintenance__`) — never to a business task's budget. An assessment NEVER induces: it only records. Accepting it (`accept_induction` API) runs the existing `induce` strictly on the bundle's execution IDs (no silent scope widening) with your admission check; rejecting it records the rejection and touches nothing. The model's confidence never bypasses evidence gates, budgets, or verification. `induction_assessment="shadow"` evaluates and records but withholds advice; `"disabled"` returns explicitly without model calls.

## `orx gc [--mode compact|purge] [--dry-run]`

Disposes only of the derived layer. `compact` is **deferred**: lossy evidence compaction is paused until the summary consumption contract exists (statistics and induction currently ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics — e.g. 90 successes + 10 failures would read as 100% failure rate). The command still runs and honestly reports the deferral; raw facts are never touched. `purge` lists retirement candidates (suspect/dormant entries) but never retires them itself.

## `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive.

## `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path.
