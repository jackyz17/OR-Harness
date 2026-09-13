---
name: or-harness
description: >
  Formulate, solve, debug, validate, and improve operations research and
  optimization problems — LP, MILP, scheduling, routing, assignment,
  network flow, resource allocation, supply-chain, and other highly coupled
  industrial OR scenarios. Provides coupling-aware problem understanding
  (CIR), a sandboxed executor with seven solver adapters, and a two-layer
  strategy memory (Execution Evidence facts + Strategic Knowledge
  commitments) that learns which solving strategies fit which problem
  structures at what execution cost. Use when the user asks to formulate, solve, retry,
  decompose, validate, or debug an optimization model, or to query/manage
  accumulated OR strategy experience. Do not use for one-off optimization
  questions with no repetition, generic math proofs, or non-optimization tasks.
---

# OR-Harness: Strategy Learning for Optimization Agents

You are the orchestrator. This capability layer only advises and executes — you retain full control: you may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to induce and when to collect garbage.

OR-Harness never calls an LLM, never runs autonomously, and keeps no hidden state: every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`).

## Core workflow (the loop to run for every optimization task)

1. **Understand coupling** — before writing any model, extract the coupling structure from the natural-language task and submit it as the task JSON's optional `coupling` field (a CIR — Coupling-Aware Intermediate Representation). Run `orx understand --task t.json` to validate it and receive `modeling_guidance`: explicit, inspectable coupling groups (e.g. "multiple decisions share one resource → ensure an aggregate capacity constraint"). The CIR is a **pre-model** artifact — it does not require a `model` field. See the `understand` command below for the CIR schema.
2. **Model the problem** — carrying the modeling guidance from step 1, write a GAMS-style model representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; see [references/modeling.md](references/modeling.md)) as the task JSON's top-level `model` field. The framework verifies it (L1 format + L2 symbol cross-reference) and derives exact structural coupling from the declared constraints. The model may later cross-check the CIR (step 3), but is never required to create one.
3. **Profile** — `orx profile --task t.json [--cir cir.json]`. Read the `derivation` report: each coupling dimension shows its value and origin (`model` > `code` > `spec` > `supplied` > `null`). If you see a `coupling_warnings` entry (supplied value contradicts structural derivation across a bin boundary), fix your understanding before proceeding. When a CIR is provided via `--cir`, the profile also runs a CIR ↔ model cross-check and surfaces `cir_warnings` (e.g. a CIR decision not declared as a model variable).
4. **Recall** — `orx recall --task t.json --top 3`. Read each candidate's `evidence`, `confidence`, and `risk_warnings`; check `solver_advisories` for solvers that failed in this environment before. You may `--exclude` any candidate and re-recall. When there is no experience yet, candidates return with `evidence="no_memory"`, `score=-inf`, `confidence=0` — the catalog is a structural vocabulary (applicability, actions, fallback, solver family), not a source of fabricated priors. Choose based on structural fit alone.
5. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories (e.g. an in-process solver when a subprocess-based one failed the sandbox). Freeze the pre-execution cost expectation with `orx predict --task t.json --strategy S` and pass the returned snapshot back at record time (`--prediction`) — feedback compares against the prediction actually used, never a post-hoc estimate. Unknown cost is reported as unknown, never as zero.
6. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The canonical model from step 2 is your blueprint — the code is a translation, not a re-derivation. Strategies such as decomposition, rolling horizon, or Lagrangian relaxation may alter the final formulation and execution plan; this is strategy-conditioned modeling, separate from the coupling-guided canonical modeling in step 2. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
7. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording.
8. **Verify** (see Verification below) — do not record an execution you have not checked.
9. **Record** — `orx record --execution <json> --override llm_tokens=<your actual token count> [--override-mode replace|increment] [--prediction <predict-output.json>]`. `--override` default is `replace` (idempotent — the value IS the measurement; re-applying never double-counts); use `increment` only for an additional measured amount within the same attempt. The response lists `unrecorded_staged_executions` for this task — if a failed first attempt is sitting there, backfill it with `orx record --from-staged <id>` (verbatim, no re-typing). Read the returned `induction_hints`, `prediction_checks`, and `cost_feedback` (computed only against the prediction snapshot you passed).
10. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.
11. **On failure** — follow Recovery below before retrying.

## Coupling dimensions (operational definitions)

Structural grouping — the foundation of all memory — keys on these. Supply them accurately or let the framework derive them (priority: CIR structure > model > spec):

| Dimension | Measures | Derivable? |
|---|---|---|
| resource_coupling | fraction of decisions involved in a resource relation (CIR) / fraction of decision variables appearing in MORE THAN ONE constraint (model) | yes — cir > model > spec |
| temporal_coupling | fraction of decisions/variables indexed by a temporal set (time/period/stage/...) | yes — cir > model > spec |
| route_complexity | fraction of decisions/variables indexed by a network set (arc/edge/link/...) | yes — cir > model > spec |
| semantic_coupling | business-semantic relatedness — invisible to structure | NO — always your call |

With a CIR present, `resource_coupling` = fraction of decisions that are the source of ≥1 `uses_resource`/`shares_resource`/`competes_for` relation; `temporal_coupling`/`route_complexity` = fraction of decisions with time-like/network-like indexes. Without a CIR, the model-based definitions apply.

Do not guess these from the problem's *name* ("it's a resource allocation problem, so rc must be high") — measure from the constraint structure. Two independent resource constraints means rc≈0, however resource-flavored the problem sounds.

## Verification (before every record)

### CIR verification (after `understand`, before writing the model)

Check `result.cir` and `result.modeling_guidance`:
- **Entity coverage**: every resource, product, site, period, or route mentioned in the task description appears as an entity or is reachable via a decision's indexes.
- **Relation plausibility**: every `uses_resource` / `shares_resource` / `competes_for` relation connects a decision to a resource-like entity — not two decisions with no shared resource.
- **Coupling group alignment**: the detected `shared_bottleneck` or agent-declared coupling groups correspond to coupling patterns the task description actually implies (e.g. "all modes use the same downstream capacity" → `shared_bottleneck` on that capacity).
- **No unresolved issues**: `result.cir.issues` is empty — non-empty means L1/L2 validation found structural problems (dangling references, duplicate names, bad evidence levels). Fix the CIR and re-run `understand` before proceeding.
- **cir_warnings** (when a `model` is also present): each warning flags a CIR ↔ model inconsistency (e.g. a CIR decision not declared as a model variable). Reconcile the CIR or the model before profiling.

If the CIR looks wrong, do not proceed to modeling — re-extract from the task description and re-run `understand`.

### Execution verification (after `execute`, before `record`)

Check `result.execution`:
- `quality.status` is one of optimal/feasible/infeasible/unbounded/timeout/error — treat anything else as a broken script, not a solver result.
- `quality.feasible` is true and `quality.objective` is finite when you expect a solution.
- `quality.gap` is recorded (derived from the bound when the solver omits it).
- `quality.problems` is empty — non-empty means verification caught something (illegal status, missing objective, non-finite value).
- The objective value is plausible for the problem (right order of magnitude, correct min/max direction) — the sandbox checks structure, not semantics; only you know what the number should mean.

Never record an execution whose `problems` is non-empty without noting why.

## Recovery (when execution fails)

- **status=error, normalized_error mentions security policy** — your script used a blocked construct (network/shell/pathlib/dynamic `open()` paths). Rewrite using only stdlib and literal `open('result.json', 'w')`. Note: subprocess-based solvers (e.g. PuLP's CBC backend) cannot run in the sandbox — switch to an in-process solver (ortools GLOP/CP-SAT, highspy). The failed execution is staged automatically; record it (`--from-staged`) so the memory learns this too.
- **status=error, traceback in normalized_error** — re-read the error, fix the model or script, re-execute. Every attempt is a separate record; a first failure is `retries=0` (an observed zero). When THIS attempt follows earlier ones inside one retry loop, declare it: `--override retries=N` (replace, absolute count of NEW retries this attempt adds). Do not hide failed attempts by not recording them.
- **status=infeasible** — do not fabricate a feasible answer. Check variable bounds and conflicting constraints; if the task itself is infeasible, record the execution with its status (infeasible outcomes are valuable induction evidence — criterion C4).
- **status=timeout** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>` and try the next candidate; record the timeout (it is a fact worth remembering).
- **recall returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, relax your exclusions or reconsider the coupling values.
- **Verification fails after a successful solve** (wrong magnitude, wrong direction) — re-derive the model, do not adjust the answer to match expectations.
- **A coupling_warnings entry at profile time** — your supplied value contradicts the structural derivation. Trust the structure (it is measured, not guessed); the derived value is what gets used for grouping anyway.
- **understand returns cir=null** — no `coupling` field was found in the task. For highly coupled problems (shared resources, cross-stage dependencies, temporal propagation), you must extract and submit a CIR before modeling. For simple problems with a single independent constraint and no shared resources, you may skip CIR and go directly to `model` — but state this decision explicitly.
- **cir_warnings at profile time** — the CIR and the model disagree (e.g. a CIR decision is not declared as a model variable, or a CIR relation has no co-occurrence in the model). Reconcile: either fix the model to match the CIR, or fix the CIR to match the model, then re-run `profile --cir`.
- **CIR validation issues (L1/L2)** — `understand` returned non-empty `cir.issues` (dangling references, duplicate names, bad evidence levels). Fix the CIR JSON and re-run `understand` — do not proceed to modeling with a broken CIR.

## Core concepts (terminology is strict)

- **Execution Evidence Bank** — append-only episodic facts ("what actually happened": the strategy actually used, the quality/cost actually observed, failures/recovery, implementation artifacts). Never stores generalizations. The single source of truth. Mutability: append-first, fact-preserving — only cost dimensions may be backfilled (`llm_tokens`), historical facts are never rewritten. An explicit `retention_reason` mark (harness-supplied) reserves representative episodes for future compaction policies; lossy GC compaction itself is currently deferred.
- **Strategic Knowledge Bank** — induced commitments ("what to do next time": expected quality, expected cost, expected failure risk): prediction intervals, calibration tracking, applicability predicates read off the supporting evidence (family + the structural cell it covered). Mutation happens at INDUCTION time only: recording collects evidence, the next `induce` creates/refreshes/revises entries. Creating one takes ≥2 supporting executions from ≥2 distinct tasks (repetitions of one task are not reproduction), and **publishing** a candidate takes a passed admission check (`induce --verify`). Admission never depends on the survival of the original evidence rows, and `induce --rebuild` re-induces from currently retained evidence (exact reconstruction is not a requirement).
- **Conditional statistics** — on-the-fly aggregation over the Evidence Bank per (strategy × structural group). Arithmetic, not knowledge; never persisted. A recount of observations, not a commitment.
- **group / evidence set** — one (family, structural cell, strategy) triple: the observations that may be aggregated together. A cell is the measurable coupling dims quantized to `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]` (unmeasured = its own `[unknown]` cell). Structurally different regions of one family are never pooled — doing so once averaged a region scoring 1.0 and a region scoring 0.1 into a single "0.55" claim.
- **CostVector** — five dimensions, stored raw, never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. `retries` counts only *extra* attempts beyond the first (a first failure is not a retry). Unknown ≠ zero: each record carries a measured-dimension mask (`cost_measured`) and a solver-runtime provenance (`reported` vs `wall_proxy` — wall-clock used when the script did not report runtime, an explicit proxy, never a precise solver runtime). Unmeasured dimensions are excluded from means, comparisons, and prediction errors — they are never treated as cheap. (In Chinese documentation: 代价, not 成本 — it is the price paid at decision time, not bookkeeping.)
- **Cold archive** — cards for retired entries. A card vetoes re-induction of the same failed generalization; `induce --force` LIFTS that veto (removes the card) when you judge the environment has genuinely drifted.

## Commands

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

### `orx understand --task t.json`

Pre-model coupling-aware understanding. Submits and validates a CIR (Coupling-Aware Intermediate Representation) — an explicit, inspectable set of entities, decisions, constraints, and relations — and returns `modeling_guidance` with concrete coupling implications for your canonical model.

The CIR lives in the task JSON's optional `coupling` field. It is a **pre-model** artifact: it does not require a `model` field and is produced before the canonical model is written. When both CIR and `model` are present, the framework cross-checks them and surfaces `cir_warnings`.

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

When no `coupling` field is present, returns `cir=null` with a message prompting you to submit coupling understanding before modeling.

Result: `result.{cir, modeling_guidance[], cir_warnings[]}`. The `modeling_guidance` entries each carry `{type, members, resource, implication}` — e.g. `shared_bottleneck`: "Multiple decisions (x, y) consume the same resource (R1); ensure one aggregate capacity constraint covers all relevant decisions."

The CIR also feeds the scalar ProblemSignature: `profile` / `recall` / `execute` derive the three structural coupling dims from the CIR structure whenever the task carries a `coupling` field (priority: CIR > model > spec > supplied) — so the retrieval signature and the modeling guidance come from the SAME structure, not two parallel lines.

### `orx profile --task t.json [--code solve.py] [--cir cir.json]`

Builds a `ProblemProfile`. Coupling derivation priority: CIR (task JSON `coupling` field, best — derived from its relations/indexes) > task JSON `model` field (measured from declared constraints) > structured `spec` fields > your `annotations.coupling` supply. `semantic_coupling` is never derived. The response includes a `derivation` report (per-dimension value/origin/notes), `model_verification` (L1+L2 issues) when a model is given, `coupling_warnings` when a supplied value contradicts the structural derivation across a bin boundary, and `cir_warnings` when a CIR is provided (via `--cir` or the task's `coupling` field) and inconsistencies with the model are found.

Result: `result.profile` = `{problem_id, family, scale_features, <four coupling dims>, risk_features, source, annotations}` plus `result.derivation`.

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--code solve.py]`

Recalls accumulated experience for the task's problem signature. Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: **published** Strategic Knowledge entry → conditional statistics → **no evidence** (`evidence="no_memory"`, `score=-inf`, `confidence=0`). An entry that is not published (an unverified candidate, or one whose admission check was `refuted` / `insufficient_evidence`) is skipped, so recall falls back to the statistics — which is what "we have a candidate but no knowledge yet" should look like. `--include-unverified` is the offline/inspection view that returns such an entry with a warning.

When no experience exists for any strategy, all candidates return with `evidence="no_memory"` — the catalog still provides the strategy vocabulary (applicability, actions, fallback, solver family) but makes no quality/cost/risk claims. Pick based on structural fit and your own judgment.

Result: `result.recommendations[]`, each `{strategy_id, name, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before). Note: when `evidence="conditional_stats"`, the `expected` fields report OBSERVED means (a recount from the Evidence Bank), not a knowledge commitment — no interval, no calibration track, no lifecycle.

`--memory-mode`: `none` (no memory consulted; all candidates return `no_memory`) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

### `orx execute --task t.json --strategy S04 --code solve.py --workspace DIR --solver NAME [--verification basic]`

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector with `llm_tokens=0` UNMEASURED — that dimension is yours to backfill, `cost_measured` mask, `solver_runtime_provenance`, `execution_features` with solver diagnostics if the solver reported any, and `cir_snapshot` = the CIR that was actually solved when the task carries a `coupling` field). **Nothing is recorded yet.**

### `orx predict --task t.json --strategy S [--code solve.py]`

Pre-execution cost expectation for one (task, strategy), with explicit provenance: `source=entry` (a matching Strategic Knowledge entry), `stats` (conditional statistics over attempt-scope evidence — a recount, with `support_n` and per-dimension `support_per_dim`), or `unknown` (no usable evidence: no data, task-scope data only, or clearly different scale — expected cost is `null`, NEVER a default zero presented as cheap). Pass the printed snapshot to `orx record --prediction` so feedback compares against the prediction actually used.

### `orx record --execution <json|path> | --from-staged <id> [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment] [--prediction <json>] [--retain-reason contrast] | --discard-staged <id>`

Appends the fact to the Execution Evidence Bank, then runs the automatic chain: cost backfill (replace by default — idempotent, never double-counts) → quality checks against matching entries, frozen ONTO THE FACT (`execution_features.quality_feedback`: the interval in force, the observed quality, hit/miss — only the strategy that actually ran is checked, attempt scope only) → cost feedback against the record's frozen pre-execution prediction snapshot (same strategy, same scope, both sides measured; written into `execution_features.cost_feedback`) → C1–C6 induction-hint checks (C4 detects cross-execution recovery chains automatically: a failed attempt under one solver followed by success under another).

**Recording never changes knowledge.** No entry is promoted, demoted, or woken here — the frozen checks are replayed by the next `orx induce` (see below).

`--retain-reason <free text>` explicitly marks this episode as representative evidence, reserved for future compaction policies. When omitted, any mark the record already carries is preserved — no automatic marking is performed.

Result: `result.{execution_id, recorded, prediction_checks[], cost_feedback?, induction_hints[]}` and, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand). Always backfill `llm_tokens` here — it is invisible to the sandbox. A task's full cost is the sum of its recorded attempt-scope records (`api.task_cost_summary`): every attempt charged to the strategy that actually ran it; retries = sum of per-attempt NEW retries; end-to-end latency is unknown unless you supply explicit task timing (never inferred by max or sum).

### `orx induce [--strategy S | --all] [--rebuild] [--dry-run] [--force] [--note TEXT] [--verify JSON]`

Consolidates facts into Strategic Knowledge (your explicit call — hints never auto-induce). This is the ONLY place knowledge changes: it (i) builds or refreshes the claim of each (family, structural cell, strategy) evidence set — its applicability is read off the supporting records (the cell its evidence occupies, never a cross-sample span that could stretch across incomparable regions); and (ii) replays the frozen checks recorded on the facts, reporting what it did under `result.revisions` (promotion at ≥5 checks / ≥70% hits **with a verified claim**, demotion at 3 consecutive misses, dormancy wakeup). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). One evidence set owns exactly one claim: re-inducing refreshes it rather than restating it (dormant entries included, so a woken claim keeps its id). `--rebuild` re-induces the whole Strategic Knowledge Bank from currently retained evidence (cold archive preserved); the result may differ from the previous bank — exact reconstruction is not a requirement. New entries inherit `strategy_type`, `actions` and `fallback_strategy_id` from the catalog vocabulary (harness-supplied values are never overwritten). `--dry-run` never writes — including under `--force` (a rehearsal does not lift a cold-archive veto).

**Independent evidence gate (creation only).** Creating a claim takes at least 2 supporting executions from ≥2 distinct `task_id`s: repeating one task is repetition, not reproduction. Every count comes from the same facts — executed, attempt-scope, in the target's structural cell — so a task-scope total can never stand in for an independent attempt. When only one task is behind the evidence the call creates nothing and returns `verification {tasks, required_tasks}` plus a `skipped` reason — the evidence is NOT discarded (recall still answers from it as `conditional_stats`), and refreshing an entry that already exists is never blocked, no matter how repetitive the new evidence is.

**`--verify JSON` publishes the candidate.** Admission and creation are separate: the gate above decides whether evidence may become a *candidate*; this decides whether the candidate's claim may become *published knowledge*. See [references/induction.md](references/induction.md#admission-verification-offline-what-makes-a-claim-published) for the payload and the three checks (`rule` / `repair` / `cost_saving`). The framework computes the verdict from the executions you supply; without it the entry is `unverified` and **is not published** — recall answers from `conditional_stats` and `predict_cost` will not quote it.

What the verdict requires, concretely: a declared criterion the framework can evaluate itself (`reference_status`, `reference_objective`, `semantic_probe`) or a genuinely independent comparison — **feasibility alone is a precondition, not a check**, and a bare `semantic_ok` boolean is recorded as `agent-declared` but cannot carry a verdict. Every execution you pass is evaluated (a counterexample anywhere in the batch refutes, whatever the order), the evidence must be for THIS strategy and family, and a comparison must be like-for-like (same task, same measurement scope for costs; a repeated execution id is not an independent comparison). Failed executions yield `insufficient_evidence` (not `refuted`): "we could not check it" is not "we checked it and it failed".

Each element of `result.results` reports `created` (entry id) / `updated` (entry id) / `skipped` (human-readable reason: fewer than 2 executions, needs independent evidence, cold-archive veto, restatement, or `recorded as an unverified candidate`), plus `verification` when a check ran — read the reason, it tells you what the memory is still missing.

`--note "TEXT"` (optional, repeatable, you phrase it): free-text applicability notes attached to the entries this call creates or refreshes. They are kept for the reader and shown by `inspect`; they never enter scoring — the framework does not pretend to verify a sentence.

### `orx inspect --bank experience|strategic|archive [--task ID] [--strategy S] [--status candidate]`

Queries a memory layer. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids).

### `orx gc [--mode compact|purge] [--dry-run]`

Disposes only of the derived layer. `compact` is **deferred**: lossy evidence compaction is paused until the summary consumption contract exists (statistics and induction currently ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics — e.g. 90 successes + 10 failures would read as 100% failure rate). The command still runs and honestly reports the deferral; raw facts are never touched. `purge` lists retirement candidates (suspect/dormant entries) but never retires them itself.

### `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive.

### `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path.

## Decision guidance

- **induction_hints after record**: hints are evidence, not orders. C1 (significant contrast between ≥2 strategies in the same structural cell, judged separately for quality and cost), C2 (extreme high/low performance with n≥2), C3 (in-group quality drift), C4 (fallback exercised), C5 (same-direction performance reproduced in ≥2 families at the same structure), C6 (all-feasible stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all. Hints never verify knowledge.
- **status vs verification**: `status` tracks how the entry has behaved (`candidate` = plausible, unproven; `validated` = ≥5 frozen checks with ≥70% hits **and a passed admission check**; `suspect` = 3 consecutive content misses, downweighted ×0.5 — treat its estimates as warnings, not facts; `dormant` = not consulted for 10 tasks, excluded from matching, and the next induction wakes it if new matching evidence arrived). `verification.state` tracks whether the CLAIM was checked (`unverified` / `verified` / `insufficient_evidence` / `refuted`). They cannot contradict: `validated` requires `verified`, and a `refuted` claim can never be `validated`. Forward calibration never substitutes for admission. Both are applied by `orx induce`, never by `record`. Cost deviations never demote or re-scope an entry — they stay on the execution as `cost_feedback` evidence.
- **confidence & cross_family**: a family-free pattern (only harness-authored entries are) is discounted and labelled when it is applied outside its provenance families — weigh it accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Always `--dry-run` first and review the plan.

## Anti-patterns (do not do these)

- Do not skip coupling understanding and jump to the model for highly coupled problems — run `understand` first. If the problem has a single independent constraint and no shared resources, you may skip CIR, but state this decision explicitly.
- Do not infer `shares_resource` or `competes_for` relations from variable name similarity alone — co-occurrence in a constraint is structural evidence only; a semantic relation requires entity/constraint semantics (resource-kind entity + capacity constraint). When in doubt, use `depends_on` with `evidence="structural"`.
- Do not skip the model representation and jump to solver code — the model is your canonical blueprint; solve.py is a translation, not a re-derivation.
- Do not write solve.py before strategy selection (`recall`) — coupling-guided canonical modeling (step 2) and strategy-conditioned modeling (step 6) are separate phases.
- Do not guess coupling values from the problem's name; measure them from constraint structure (or let the framework do it from your model).
- Do not induce just because a hint appeared.
- Do not treat repeated attempts of one task as independent verification — a claim is created only from ≥2 distinct tasks (the framework refuses otherwise; keep recording, a second task is what is missing).
- Do not publish a claim on the strength of a program's own printed verdict, or on the candidate's own natural-language summary — admission runs a framework-side check on real executions (`induce --verify`).
- Do not treat applicability notes as fact — they are your own phrasing, stored for the reader and never scored.
- Do not ignore `risk_warnings` in recommendations or `solver_advisories`.
- Do not skip the `llm_tokens` backfill on record — cost learning silently degrades without it.
- Do not skip recording failed executions — failures are the most valuable induction raw material (C4).
- Do not hand-craft an execution JSON to backfill a failure — use `record --from-staged` (verbatim, no drift).
- Do not revive cold-archive vetoes without strong evidence of environment drift.
- Do not adjust a model's constraints merely to match a reference value; re-derive instead.

## References (read on demand)

- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers, and the Coupling-Aware Intermediate Representation (CIR) schema. Read before writing your first model or CIR.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), how applicability is read off evidence, the offline lifecycle. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, a claim meeting its counterexample). Read when unsure how the pieces fit together in practice.
