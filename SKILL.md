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

1. **Profile (the single analysis entry)** — submit your coupling understanding as the task JSON's optional `coupling` field (a CIR — Coupling-Aware Intermediate Representation) and run `orx profile --task t.json`. One call returns: the validated CIR with `modeling_guidance` (explicit, inspectable coupling groups), the problem profile, and the derivation report (each coupling dimension's value and origin). The profile serves two purposes: it helps YOU understand the problem structure, and it is the structural key that retrieves comparable evidence in the next step. **A task without a `model` field is a normal state** — strategy selection needs the task text, the CIR, and the profile, never a finished formulation.
2. **Recall and compare strategies** — `orx recall --task t.json --top 3`. Recall returns **two independent channels** (details under Commands):
   - `recommendations` (structural — the REUSE key): the profile decides which evidence is comparable. Each candidate's expected quality/cost/risk (`score = α·Q̂ − β·Ĉ − γ·R̂`) comes from `evidence`, `confidence`, `risk_warnings`; check `solver_advisories`. With no experience yet, candidates return `evidence="no_memory"`, `score=-inf`, `confidence=0` — the catalog is a vocabulary, not fabricated priors.
   - `vector_recall` (text similarity — the DISCOVERY channel): unfiltered by structural cell, so a near-identical problem in a different bucket is still visible. `similarity` is **never** a quality/cost/risk number; each hit carries the memory's own text and a `structural_match` label. **A cross-cell hit is never pooled into your cell's statistics**, and `unknown` applicability is not a yes.

   When the text channel cannot run, `degraded` says why and the structural result comes back alone — never a silent empty result; `vector_recall.unindexed` counts memories with no vector and says where to find them. You may `--exclude` any candidate and re-recall; `--include-unverified` is the offline view. **Optionally (M3)**: with a world model configured, `orx plan-next` compares candidates by their PREDICTED consequences (same quality/cost/risk yardstick) and suggests a first step.
3. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories. Freeze the pre-execution cost expectation with `orx predict --task t.json --strategy S` and pass the returned snapshot back at record time (`--prediction`) — feedback compares against the prediction actually used, never a post-hoc estimate. Unknown cost is reported as unknown, never as zero. If you planned with `plan-next`, record your explicit choice with `orx choose-next` (accept, deviate, or reject).
4. **Model the problem (intermediate representation, after the strategy is chosen)** — NOW write the GAMS-style model representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; see [references/modeling.md](references/modeling.md)) as the task JSON's top-level `model` field. The model is the blueprint for solve.py, written under the chosen strategy: decomposition, rolling horizon, or relaxation strategies may alter the formulation and execution plan. The framework verifies it (L1 format + L2 symbol cross-reference) and derives exact structural coupling from the declared constraints. Re-run `orx profile` to get the CIR ↔ model cross-check (`cir_warnings`, e.g. a CIR decision not declared as a model variable) — this is how you catch "the model missed a coupling the CIR declared" before coding.
5. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The model from step 4 is your blueprint — the code is a translation, not a re-derivation. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
6. **Predict cost (optional world-model shadow)** — with a world model configured (`--world-model`), you may ask for a structured outcome prediction of the candidate execution (`orx predict-outcome`) BEFORE executing — it never changes your choice, and you compare it afterwards (`orx bind-outcome`) to accumulate calibration evidence.
7. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording. If you made a shadow prediction, bind it now (`orx bind-outcome --prediction <id> --action <action_id from the execute output>`).
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

### CIR verification (at `orx profile`, before choosing a strategy)

Check `result.coupling.cir` and `result.coupling.modeling_guidance`:
- **Entity coverage**: every resource, product, site, period, or route mentioned in the task description appears as an entity or is reachable via a decision's indexes.
- **Relation plausibility**: every `uses_resource` / `shares_resource` / `competes_for` relation connects a decision to a resource-like entity — not two decisions with no shared resource.
- **Coupling group alignment**: the detected `shared_bottleneck` or agent-declared coupling groups correspond to coupling patterns the task description actually implies (e.g. "all modes use the same downstream capacity" → `shared_bottleneck` on that capacity).
- **No unresolved issues**: `result.coupling.cir.issues` is empty — non-empty means L1/L2 validation found structural problems (dangling references, duplicate names, bad evidence levels). Fix the CIR and re-run `orx profile` before proceeding.
- **cir_warnings** (when a `model` is also present — i.e. after you wrote the model in step 4): each warning flags a CIR ↔ model inconsistency (e.g. a CIR decision not declared as a model variable). Reconcile the CIR or the model before writing solve.py.

If the CIR looks wrong, do not proceed — re-extract from the task description and re-run `orx profile`.

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
- **recall returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, relax your exclusions or reconsider the coupling values. (This concerns the STRUCTURAL channel: a `vector_recall` hit for a similar problem whose structure differed often means your coupling values are off.)
- **recall reports `degraded`** — the text channel was skipped; only the structural channel ran. Read the `reason` (missing backend / no task text / index missing → `orx rebuild-index` / model changed / backend error). Never read it as "no similar memories exist". `vector_recall.unindexed` lists what text search could not see — look those up with `orx inspect --bank experience|strategic|texts`.
- **Verification fails after a successful solve** (wrong magnitude, wrong direction) — re-derive the model, do not adjust the answer to match expectations.
- **A coupling_warnings entry at profile time** — your supplied value contradicts the structural derivation. Trust the structure (it is measured, not guessed); the derived value is what gets used for grouping anyway.
- **profile returns coupling.cir=null** — no `coupling` field was found in the task. For highly coupled problems (shared resources, cross-stage dependencies, temporal propagation), extract and submit a CIR before choosing a strategy. For simple problems with a single independent constraint and no shared resources, you may skip CIR — but state this decision explicitly.
- **cir_warnings at profile time** — the CIR and the model disagree (e.g. a CIR decision is not declared as a model variable, or a CIR relation has no co-occurrence in the model). Reconcile: either fix the model to match the CIR, or fix the CIR to match the model, then re-run `profile`.
- **CIR validation issues (L1/L2)** — `profile` returned non-empty `coupling.cir.issues` (dangling references, duplicate names, bad evidence levels). Fix the CIR JSON and re-run `orx profile` — do not proceed with a broken CIR.

## Core concepts (terminology is strict)

- **Execution Evidence Bank** — append-only episodic facts ("what actually happened": the strategy actually used, the quality/cost actually observed, failures/recovery, implementation artifacts). Never stores generalizations. The single source of truth. Mutability: append-first, fact-preserving — only cost dimensions may be backfilled (`llm_tokens`), historical facts are never rewritten. An explicit `retention_reason` mark (harness-supplied) reserves representative episodes for future compaction policies; lossy GC compaction itself is currently deferred.
- **Strategic Knowledge Bank** — induced commitments ("what to do next time": expected quality, expected cost, expected failure risk): prediction intervals, calibration tracking, applicability predicates read off the supporting evidence (family + the structural cell it covered). Mutation happens at INDUCTION time only: recording collects evidence, the next `induce` creates/refreshes/revises entries. Creating one takes ≥2 supporting executions from ≥2 distinct tasks (repetitions of one task are not reproduction), and **publishing** a candidate takes a passed admission check (`induce --verify`). Admission never depends on the survival of the original evidence rows, and `induce --rebuild` re-induces from currently retained evidence (exact reconstruction is not a requirement).
- **Conditional statistics** — on-the-fly aggregation over the Evidence Bank per (strategy × structural group). Arithmetic, not knowledge; never persisted. A recount of observations, not a commitment.
- **group / evidence set** — one (family, structural cell, strategy) triple: the observations that may be aggregated together. A cell is the measurable coupling dims quantized to `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]` (unmeasured = its own `[unknown]` cell). Structurally different regions of one family are never pooled — doing so once averaged a region scoring 1.0 and a region scoring 0.1 into a single "0.55" claim.
- **CostVector** — five dimensions, stored raw, never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. `retries` counts only *extra* attempts beyond the first (a first failure is not a retry). Unknown ≠ zero: each record carries a measured-dimension mask (`cost_measured`) and a solver-runtime provenance (`reported` vs `wall_proxy` — wall-clock used when the script did not report runtime, an explicit proxy, never a precise solver runtime). Unmeasured dimensions are excluded from means, comparisons, and prediction errors — they are never treated as cheap. (In Chinese documentation: 代价, not 成本 — it is the price paid at decision time, not bookkeeping.)
- **Cold archive** — cards for retired entries. A card vetoes re-induction of the same failed generalization; `induce --force` LIFTS that veto (removes the card) when you judge the environment has genuinely drifted.

## Commands

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

### `orx profile --task t.json [--code solve.py] [--cir cir.json]`

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

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--include-unverified] [--code solve.py]`

Recalls experience through **two independent channels**, never blended (full field list and JSON shape: [references/commands.md](references/commands.md)):

**1. Structural (`recommendations[]`) — REUSE.** Score = `α·Q̂ − β·C_scalar − γ·R̂`. Evidence precedence: **published** Strategic Knowledge entry → conditional statistics → **no evidence** (`evidence="no_memory"`, `score=-inf`, `confidence=0`). With `evidence="conditional_stats"` the `expected` fields are OBSERVED means (a recount), not a commitment. Also returns `available_solver_families` and `solver_advisories`.

**2. Text similarity (`vector_recall`) — DISCOVERY.** The task text is embedded and compared with every indexed memory **with no structural pre-filter**, so a near-identical problem in a different cell is still surfaced. `similarity` is a raw text cosine — **never** a quality/cost/risk estimate, never in any score. Hits carry the memory's own text and a structural label: `same_cell`/`different_cell`/`unknown` (executions), `applies`/`conflicts`/`unknown` + `reusable` + `reason` (knowledge). A `different_cell` hit is **discovery only** — it never joins your cell's statistics. `unknown` means a needed value is missing: **missing information is not applicability**, so do not reuse it. Dormant/retired/(by default) unpublished entries are excluded *before* the `top_k` cut.

**Degradation is always reported, never silent.** When the text channel cannot run, `degraded = {"path": "profile_only", "reason": ...}` names the cause (no embedding backend / no task text / index missing → `orx rebuild-index` / embedding model changed / backend error) and the structural result comes back intact. `vector_recall.unindexed` counts memories with no vector (legacy text, deferred sync) and says where to find them: `orx inspect --bank experience|strategic|texts`. An empty or degraded `vector_recall` is **not** evidence that no similar memory exists.

`--include-unverified` is the offline/inspection view (unpublished candidates appear in both channels; the structural path labels them `UNPUBLISHED`). `--memory-mode`: `none` | `cases` | `strategic` | `cost-aware`. When no experience exists, candidates return `evidence="no_memory"` — the catalog provides vocabulary, never fabricated priors.

### Embedding configuration (the text channel)

The text channel is **off unless a real embedding model is configured** — deliberately: a lexical hash is not semantic retrieval, so the framework never substitutes one silently. Set `OR_EMBEDDING_BASE_URL` + `OR_EMBEDDING_MODEL` + `OR_EMBEDDING_API_KEY` (any OpenAI-compatible `/embeddings` endpoint; **not** the `OR_WM_*` chat variables — different endpoint, different dimension), or inject a backend (`ORHarness(embedding=...)`). Code, CLI flags, degradation reasons, and index health: [references/commands.md](references/commands.md).

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

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts [--task ID] [--strategy S] [--status candidate] [--episode EP]`

Queries a memory layer. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids). `actions` / `snapshots` query the world-model substrate (below): the unified action log and the frozen belief snapshots.

### `orx snapshot --task t.json [--episode ep1]`

Freeze and persist the current **belief state** for a task (world-model M1 substrate): H (value-copied verified/legacy/unverified knowledge refs + experience counts + tool config), P (profile + task digest + CIR/model digests + the problem-relevant task payload when no `task_ref` exists), X (the episode's accumulated progress — selected plan, model artifact, current solution, verification evidence — each field labelled with `provenance` (observed vs agent_reported) AND `epistemic` (fact/inferred/unknown), two orthogonal dimensions), B (declared budget + consumption), and a coverage view (cell statistics + layered knowledge, no composite capability score). Snapshots are frozen by deep-copy isolation: later bank writes never change what a snapshot meant. Within an episode, each action's post state merges into the accumulated progress, so the next action inherits what earlier actions established.

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Report an action **you** performed (TYPE ∈ model/select_strategy/verify/finish_task) — the library did not execute it; the record is labelled `source=agent_reported` and a missing pre-state is marked `pre_snapshot_missing`, never fabricated. Explicitly supplied cost dimensions (including an explicit zero) are marked measured. `--amend-cost` backfills an action's cost (replace semantics, idempotent).

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

The honest budget view for a task/episode: consumption over ALL real action costs — recorded AND staged-but-unrecorded executions (deduplicated by execution_id) plus own-cost actions (a macro's reference cost is never re-counted; its own additional spend is). Status: `exceeded` (a measured dimension over the limit; a declared latency budget is judged per-attempt), `ok` (every declared dimension measured and within), `unconfirmed` (known spend within limits but a declared dimension is unknown — NOT confirmed within budget), `no_budget_declared`. Executions of the task with no episode linkage (pre-M1 records) are reported separately as `unattributed` and never charged to a fresh episode. Declarations persist in the store, so separate CLI invocations see the same budget. Hypothetical actions are excluded from every real consumption view.

`orx execute --episode ep1` and `orx induce` automatically record their actions in the unified log: execute as a macro (`pre` snapshot before execution, `post` after, `rollup=reference`, linked to the execution id); induce in the **maintenance scope** (`__maintenance__` / `maint_<ts>`) with a real pre/post knowledge state, verification results, a knowledge delta, and a business result that separates `created` (verified) from `created_unverified` (a candidate is NOT knowledge growth), `updated` / `revised` / `refused` / `unchanged`. A crashed induction still leaves a `failed` action with its pre state. Dry-run persists nothing.

### `orx [--world-model URL::MODEL] predict-outcome --task t.json --action-spec spec.json [--episode ep1] [--parent-action ACTION_ID]`

**World-model outcome prediction (M2, shadow mode).** Ask a configured world model for a structured prediction of ONE candidate action's consequences, from the frozen pre-action state. The candidate is an `ActionSpec` JSON (`{"action_type": "execute_strategy", "task_id": ..., "strategy_id": "S01", "solver": "highs", ...}`) — a hypothesis, NOT a recorded action. The prediction carries: expected execution status / feasibility / quality / failure risk / per-dimension cost, predicted successor state changes (hypothetical — never written to real state), the model's self-reported confidence (**uncalibrated**), its claimed evidence basis, and fields it explicitly declined to predict.

- **Configuration boundary**: `--world-model BASE_URL::MODEL` (OpenAI-compatible endpoint; API key from `$OR_WM_API_KEY`). Credentials never persist. Without the flag, the command returns an explicit `not_configured` error — and NO other command ever invokes a model.
- **Shadow discipline**: the prediction changes NOTHING. `recall`, `predict`, `execute`, `record` behave identically whether or not you predict. You remain the decision-maker.
- **Call cost**: the model call's own spend (tokens/latency from provider usage) is recorded on the prediction and, with `--parent-action`, charged to that action's own cost — separate from the PREDICTED cost of the target action.
- **Timing**: predict BEFORE executing. The input snapshot is frozen at prediction time; later bank changes never rewrite it.

### `orx bind-outcome --prediction ID --action ACTION_ID`

Bind a prediction to the real action that ran, then compare. Type/strategy/solver are checked: a mismatch (you predicted strategy A, executed B) is recorded and NOT scored — no counterfactual truth is fabricated. The comparison covers only fields both sides define: status category, feasibility, quality (when the execution produced a solution), and cost per dimension (both sides measured — the same both-sides-measured discipline as cost feedback). Missing comparisons are listed with reasons. The feedback is APPENDED to the frozen prediction — the original is never modified, and re-running the comparison is idempotent (the model is never re-invoked). Online comparison records facts and errors only; knowledge updates still go through explicit offline induction with verification.

Query predictions with `orx inspect --bank predictions [--task ID]`.

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode ep1] [--candidates specs.json] [--horizon 1|2] [--max-calls N]`

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

### `orx choose-next --decision ACTION_ID [--chosen spec.json | --rejected] [--note "..."]`

Record YOUR explicit choice after a plan: accept the suggestion, pick another candidate (a deviation, recorded with its reason), or reject all. Only this call writes `X.selected_plan`; the choice itself produces no execution quality. Then execute the step with `orx execute`, record the result with `orx record`, and bind the executed step's prediction with `orx bind-outcome`. Re-plan from the new real state afterwards — the old plan stays as the suggestion of its time.

### `orx [--world-model URL::MODEL] assess-induction [--bundle bundle.json | --candidates-only] [--workload forecast.json]`

**M4 offline maintenance assessment.** Scans the banks for induction candidate bundles (frozen evidence: exact execution IDs, tasks, statistics, trigger reasons — new claims need ≥2 executions from ≥2 tasks; revisions reference the existing entry and its state at bundle time), then asks the world model to predict the INDUCTION action's consequences: candidate formation probability, expected reuse benefit, generalization risk, and the resulting claim's quality/cost. The recommendation (`induce_new` / `revise` / `defer` / `insufficient_evidence`) comes with a value decomposition (`net_value = α·benefit − γ·risk`, workload forecasts scale the benefit term) and the real assessment cost, charged once to a maintenance-scope action (`__maintenance__`) — never to a business task's budget. An assessment NEVER induces: it only records. Accepting it (`accept_induction` API) runs the existing `induce` strictly on the bundle's execution IDs (no silent scope widening) with your admission check; rejecting it records the rejection and touches nothing. The model's confidence never bypasses evidence gates, budgets, or verification. `induction_assessment="shadow"` evaluates and records but withholds advice; `"disabled"` returns explicitly without model calls.

### `orx gc [--mode compact|purge] [--dry-run]`

Disposes only of the derived layer. `compact` is **deferred**: lossy evidence compaction is paused until the summary consumption contract exists (statistics and induction currently ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics — e.g. 90 successes + 10 failures would read as 100% failure rate). The command still runs and honestly reports the deferral; raw facts are never touched. `purge` lists retirement candidates (suspect/dormant entries) but never retires them itself.

### `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive. Its vector leaves the retrieval index with it, so recall can never surface the retired advice.

### `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

Explicit retrieval-index maintenance — the only bulk-embedding path. Needed on FIRST build (until then recall reports `degraded`), after an embedding-model change (a foreign index is refused wholesale, never partially reused), and to settle a sync deferred by an embedding failure. Not routine: `record`/`induce` keep the index current incrementally. `--dry-run` writes nothing at all. The index is derived data — rebuilding never changes a fact or an entry.

### `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path, and **retrieval-index health** (`result.retrieval_index`): backend/model id and per layer index count vs current documents, `stale` / `missing` / `orphaned`. Read-only — `doctor` builds nothing; a missing index is reported with the `rebuild-index` hint.

## Decision guidance

- **induction_hints after record**: hints are evidence, not orders. C1 (significant contrast between ≥2 strategies in the same structural cell, judged separately for quality and cost), C2 (extreme high/low performance with n≥2), C3 (in-group quality drift), C4 (fallback exercised), C5 (same-direction performance reproduced in ≥2 families at the same structure), C6 (all-feasible stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all. Hints never verify knowledge.
- **status vs verification**: `status` tracks how the entry has behaved (`candidate` = plausible, unproven; `validated` = ≥5 frozen checks with ≥70% hits **and a passed admission check**; `suspect` = 3 consecutive content misses, downweighted ×0.5 — treat its estimates as warnings, not facts; `dormant` = not consulted for 10 tasks, excluded from matching, and the next induction wakes it if new matching evidence arrived). `verification.state` tracks whether the CLAIM was checked (`unverified` / `verified` / `insufficient_evidence` / `refuted`). They cannot contradict: `validated` requires `verified`, and a `refuted` claim can never be `validated`. Forward calibration never substitutes for admission. Both are applied by `orx induce`, never by `record`. Cost deviations never demote or re-scope an entry — they stay on the execution as `cost_feedback` evidence.
- **confidence & cross_family**: a family-free pattern (only harness-authored entries are) is discounted and labelled when it is applied outside its provenance families — weigh it accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Always `--dry-run` first and review the plan.
- **How to read `vector_recall`**: `similarity` says a memory is worth LOOKING AT, never that it is good, cheap, or safe. Let the structural verdict decide reuse: `different_cell` means the same text was solved with a different structure, so its numbers describe THAT structure — informative, not transferable. `applies` may be reused; `conflicts` names why not; `unknown` means a value needed to decide is missing — go measure it instead of assuming fit.

## Anti-patterns (do not do these)

- Do not treat a world-model prediction as a decision — it is a shadow hypothesis. You choose the strategy; the prediction only gets compared afterwards.
- Do not treat a `plan-next` suggestion as a selection or an execution — it changes nothing until you `choose-next` and `execute`. Do not re-plan from a hypothetical state: the second step of a rollout is a conditional outlook, never real feedback; after executing the first step, re-plan from the NEW real state.
- Do not treat an `assess-induction` recommendation as performed induction — it only records an evaluation and its cost. Acceptance is your explicit call, and the actual induction still goes through the same admission verification as a direct `induce --verify`. A high model confidence is not evidence: a bundle whose verification fails never publishes.
- Do not predict after executing and call it a forecast — the input snapshot must be frozen BEFORE the action. A post-hoc "prediction" is not evidence.
- Do not compare a prediction against a different strategy's or configuration's execution — bind the action that actually matches the candidate; a mismatch is recorded, not scored.
- Do not treat the model's self-reported `confidence` as a calibrated probability, or a prediction with no evidence basis as knowledge.
- Do not infer `shares_resource` or `competes_for` relations from variable name similarity alone — co-occurrence in a constraint is structural evidence only; a semantic relation requires entity/constraint semantics (resource-kind entity + capacity constraint). When in doubt, use `depends_on` with `evidence="structural"`.
- Do not write the `model` field or solve.py before choosing a strategy — the model is an intermediate representation written AFTER the strategy is chosen and BEFORE solve.py; strategy selection relies on the task text, the CIR, the profile, and the evidence's expected quality/cost/risk, never on a finished formulation. Conversely, do not skip the model once the strategy is chosen — it is your canonical blueprint; solve.py is a translation, not a re-derivation.
- Do not skip coupling understanding for highly coupled problems — submit a CIR via `orx profile` before choosing a strategy. If the problem has a single independent constraint and no shared resources, you may skip CIR, but state this decision explicitly.
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
- Do not read a similarity score as a quality/cost/risk estimate, and do not apply a `different_cell` hit's statistics to your problem — cross-cell evidence is DISCOVERY, never reuse. Its observed numbers belong to the structure it was measured in.
- Do not treat `unknown` applicability as a yes, and do not apply an entry whose `reusable` is false — a missing measurement is not a satisfied condition.
- Do not read a `degraded` recall (or an empty `vector_recall`) as "no similar memories exist" — the channel may simply have been skipped; the `reason` says which, and `unindexed` says what the text search could not see.
- Do not hand-edit or commit `{home}/index/*.embedding.json` — it is derived data; rebuild it with `orx rebuild-index` instead.

## References (read on demand)

- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers, and the Coupling-Aware Intermediate Representation (CIR) schema. Read before writing your first model or CIR.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), how applicability is read off evidence, the offline lifecycle. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, a claim meeting its counterexample). Read when unsure how the pieces fit together in practice.
