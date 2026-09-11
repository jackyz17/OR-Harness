---
name: or-harness
description: >
  Formulate, solve, debug, validate, and improve operations research and
  optimization problems — LP, MILP, scheduling, routing, assignment,
  network flow, resource allocation, supply-chain, and other highly coupled
  industrial OR scenarios. Provides coupling-aware problem understanding
  (CIR), a sandboxed executor with seven solver adapters, and a two-layer
  strategy memory (Experience Bank facts + Strategic Bank commitments) that
  learns which solving strategies fit which problem structures at what
  execution cost. Use when the user asks to formulate, solve, retry,
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
5. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories (e.g. an in-process solver when a subprocess-based one failed the sandbox).
6. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The canonical model from step 2 is your blueprint — the code is a translation, not a re-derivation. Strategies such as decomposition, rolling horizon, or Lagrangian relaxation may alter the final formulation and execution plan; this is strategy-conditioned modeling, separate from the coupling-guided canonical modeling in step 2. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
7. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording.
8. **Verify** (see Verification below) — do not record an execution you have not checked.
9. **Record** — `orx record --execution <json> --override llm_tokens=<your actual token count>`. The response lists `unrecorded_staged_executions` for this task — if a failed first attempt is sitting there, backfill it with `orx record --from-staged <id>` (verbatim, no re-typing). Read the returned `induction_hints` and `prediction_checks`.
10. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.
11. **On failure** — follow Recovery below before retrying.

## Coupling dimensions (operational definitions)

Structural grouping — the foundation of all memory — keys on these. Supply them accurately or let the framework derive them (it will, from your model representation):

| Dimension | Measures | Derivable? |
|---|---|---|
| resource_coupling | fraction of decision variables appearing in MORE THAN ONE constraint (0 = constraints independent; 1 = fully coupled) | yes — model > spec |
| temporal_coupling | fraction of variables indexed by a temporal set (time/period/stage/...) | yes — model > spec |
| route_complexity | fraction of variables indexed by a network set (arc/edge/link/...) | yes — model > spec |
| semantic_coupling | business-semantic relatedness — invisible to structure | NO — always your call |

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
- **status=error, traceback in normalized_error** — re-read the error, fix the model or script, re-execute. Each rerun costs `retries +1` in the CostVector — that is by design; do not hide failed attempts by not recording them.
- **status=infeasible** — do not fabricate a feasible answer. Check variable bounds and conflicting constraints; if the task itself is infeasible, record the execution with its status (infeasible outcomes are valuable induction evidence — criterion C4).
- **status=timeout** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>` and try the next candidate; record the timeout (it is a fact worth remembering).
- **recall returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, relax your exclusions or reconsider the coupling values.
- **Verification fails after a successful solve** (wrong magnitude, wrong direction) — re-derive the model, do not adjust the answer to match expectations.
- **A coupling_warnings entry at profile time** — your supplied value contradicts the structural derivation. Trust the structure (it is measured, not guessed); the derived value is what gets used for grouping anyway.
- **understand returns cir=null** — no `coupling` field was found in the task. For highly coupled problems (shared resources, cross-stage dependencies, temporal propagation), you must extract and submit a CIR before modeling. For simple problems with a single independent constraint and no shared resources, you may skip CIR and go directly to `model` — but state this decision explicitly.
- **cir_warnings at profile time** — the CIR and the model disagree (e.g. a CIR decision is not declared as a model variable, or a CIR relation has no co-occurrence in the model). Reconcile: either fix the model to match the CIR, or fix the CIR to match the model, then re-run `profile --cir`.
- **CIR validation issues (L1/L2)** — `understand` returned non-empty `cir.issues` (dangling references, duplicate names, bad evidence levels). Fix the CIR JSON and re-run `understand` — do not proceed to modeling with a broken CIR.

## Core concepts (terminology is strict)

- **Experience Bank** — append-only episodic facts ("what happened"). Never stores generalizations. The single source of truth.
- **Strategic Bank** — induced commitments ("what will happen"): prediction intervals, calibration tracking, feature predicates. Fully rebuildable from facts (`induce --rebuild`).
- **Conditional statistics** — on-the-fly aggregation over the Experience Bank per (strategy × structural group). Arithmetic, not knowledge; never persisted.
- **Structural group** — problem family + coupling-feature bins (e.g. `resource_coupling ∈ [0.75, 1.0]`).
- **CostVector** — five dimensions, stored raw, never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. Retries are a cost: "wrong model → repair → rerun" must cost more than getting it right. (In Chinese documentation: 代价, not 成本 — it is the price paid at decision time, not bookkeeping.)
- **Cold archive** — tombstones of retired entries. Vetoes re-induction of the same failed generalization; `--force` revives only under genuine environment drift.

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

### `orx profile --task t.json [--code solve.py] [--cir cir.json]`

Builds a `ProblemProfile`. Coupling derivation priority: task JSON `model` field (best — measured from declared constraints) > structured `spec` fields > your `annotations.coupling` supply. `semantic_coupling` is never derived. The response includes a `derivation` report (per-dimension value/origin/notes), `model_verification` (L1+L2 issues) when a model is given, `coupling_warnings` when a supplied value contradicts the structural derivation across a bin boundary, and `cir_warnings` when a CIR is provided via `--cir` and inconsistencies with the model are found.

Result: `result.profile` = `{problem_id, family, scale_features, <four coupling dims>, risk_features, source, annotations}` plus `result.derivation`.

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--code solve.py]`

Recalls accumulated experience for the task's problem signature. Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: matching Strategic entry → conditional statistics → **no evidence** (`evidence="no_memory"`, `score=-inf`, `confidence=0`).

When no experience exists for any strategy, all candidates return with `evidence="no_memory"` — the catalog still provides the strategy vocabulary (applicability, actions, fallback, solver family) but makes no quality/cost/risk claims. Pick based on structural fit and your own judgment.

Result: `result.recommendations[]`, each `{strategy_id, name, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before).

`--memory-mode`: `none` (no memory consulted; all candidates return `no_memory`) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

### `orx execute --task t.json --strategy S04 --code solve.py --workspace DIR --solver NAME [--verification basic]`

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector with `llm_tokens=0` — that dimension is yours to backfill, `execution_features` with solver diagnostics if the solver reported any). **Nothing is recorded yet.**

### `orx record --execution <json|path> | --from-staged <id> [--override llm_tokens=1840,tool_calls=9] | --discard-staged <id>`

Appends the fact to the Experience Bank, then runs the automatic chain: cost backfill → prediction checks against matching entries (hits/misses feed calibration; 3 consecutive misses demote an entry to `suspect`; cross-family misses tighten an L2/L3 entry's scope) → C1–C6 induction-hint checks (C4 detects cross-execution recovery chains automatically: a failed attempt under one solver followed by success under another).

Result: `result.{execution_id, recorded, prediction_checks[], induction_hints[]}` and, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand). Always backfill `llm_tokens` here — it is invisible to the sandbox.

### `orx induce [--strategy S | --all] [--rebuild] [--widen ID] [--tighten ID] [--dry-run] [--force] [--llm-conditions <json>]`

Consolidates facts into a Strategic entry (your explicit call — hints never auto-induce). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). A pattern already covered by an existing entry is refused (restatement-only entries are forbidden). `--rebuild` regenerates the whole Strategic Bank from facts (cold archive preserved). `--widen/--tighten` move an entry along the scope ladder L1 (family + fine bins) → L2 (fine bins) → L3 (coarse bins); widening is falsifiable — a cross-family miss auto-tightens.

`--llm-conditions` (optional, you phrase it): `[{"text": "...", "supporting_execution_ids": ["ex_..."]}]`. Citations are verified — fake ids or numeric claims disagreeing with cited records are rejected; accepted conditions stay `verified: false` and never enter scoring until re-confirmed by future executions.

### `orx inspect --bank experience|strategic|archive [--task ID] [--strategy S] [--status candidate]`

Queries a memory layer. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids).

### `orx gc [--mode compact|purge] [--dry-run]`

Disposes only of the derived layer. `compact`: groups covered by entries and beyond retention limits (provenance references, recent 50 tasks, exploratory groups with n<5) collapse into ledger lines — statistics preserved, trajectory detail dropped. `purge` mode lists retirement candidates but never retires them itself.

### `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive.

### `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path.

## Decision guidance

- **induction_hints after record**: hints are evidence, not orders. C1 (significant contrast between ≥2 strategies), C2 (extreme high/low performance with n≥2), C3 (in-group drift), C4 (fallback exercised), C5 (cross-family reproduction — suggests L2), C6 (stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all.
- **status**: `candidate` = plausible, unproven. `validated` = ≥5 predictions, ≥70% hit rate. `suspect` = 3 consecutive QUALITY misses, downweighted ×0.5 — treat its estimates as warnings, not facts. `dormant` = not consulted for 10 tasks, excluded from matching (wakes on a future hit). Cost misses never demote — they feed `cost_hit_rate` and surface as "uncalibrated cost estimate" warnings.
- **confidence & cross_family**: cross-family generalizations (L2/L3 entry matching a family absent from its provenance) are discounted and labelled — weigh them accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Always `--dry-run` first and review the plan.

## Anti-patterns (do not do these)

- Do not skip coupling understanding and jump to the model for highly coupled problems — run `understand` first. If the problem has a single independent constraint and no shared resources, you may skip CIR, but state this decision explicitly.
- Do not infer `shares_resource` or `competes_for` relations from variable name similarity alone — co-occurrence in a constraint is structural evidence only; a semantic relation requires entity/constraint semantics (resource-kind entity + capacity constraint). When in doubt, use `depends_on` with `evidence="structural"`.
- Do not skip the model representation and jump to solver code — the model is your canonical blueprint; solve.py is a translation, not a re-derivation.
- Do not write solve.py before strategy selection (`recall`) — coupling-guided canonical modeling (step 2) and strategy-conditioned modeling (step 6) are separate phases.
- Do not guess coupling values from the problem's name; measure them from constraint structure (or let the framework do it from your model).
- Do not induce just because a hint appeared.
- Do not treat `verified: false` applicability text as fact.
- Do not ignore `risk_warnings` in recommendations or `solver_advisories`.
- Do not skip the `llm_tokens` backfill on record — cost learning silently degrades without it.
- Do not skip recording failed executions — failures are the most valuable induction raw material (C4).
- Do not hand-craft an execution JSON to backfill a failure — use `record --from-staged` (verbatim, no drift).
- Do not revive cold-archive vetoes without strong evidence of environment drift.
- Do not adjust a model's constraints merely to match a reference value; re-derive instead.

## References (read on demand)

- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers, and the Coupling-Aware Intermediate Representation (CIR) schema. Read before writing your first model or CIR.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), the scope ladder, forward validation, citation binding. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, cross-family tighten). Read when unsure how the pieces fit together in practice.
