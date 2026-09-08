---
name: or-harness
description: Learn which solving strategies fit which optimization problem structures, at what execution cost, by recording and consolidating your own solve executions — a two-layer memory (Experience Bank facts + Strategic Bank commitments) with a sandboxed executor and seven solver adapters. Use when the user asks to solve, retry, or improve at LP/MILP/scheduling/routing/assignment tasks over a session, or to query/manage accumulated OR strategy experience. Do not use for one-off optimization questions with no repetition, generic math proofs, or non-optimization tasks.
---

# OR-Harness: Strategy Learning for Optimization Agents

You are the orchestrator. This capability layer only advises and executes — you retain full control: you may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to induce and when to collect garbage.

OR-Harness never calls an LLM, never runs autonomously, and keeps no hidden state: every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`).

## Core workflow (the loop to run for every optimization task)

1. **Model the problem** — before any solver code, write a GAMS-style model representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; see [references/modeling.md](references/modeling.md)) and carry it as the task JSON's top-level `model` field. The framework verifies it (L1 format + L2 symbol cross-reference) and derives exact structural coupling from the declared constraints — the single best coupling source. Skipping this makes every later step weaker.
2. **Profile** — `orx profile --task t.json`. Read the `derivation` report: each coupling dimension shows its value and origin (`model` > `code` > `spec` > `supplied` > `null`). If you see a `coupling_warnings` entry (supplied value contradicts structural derivation across a bin boundary), fix your understanding before proceeding.
3. **Recommend** — `orx recommend --task t.json --top 3`. Read each candidate's `evidence`, `confidence`, and `risk_warnings`; check `solver_advisories` for solvers that failed in this environment before. You may `--exclude` any candidate and re-recommend.
4. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories (e.g. an in-process solver when a subprocess-based one failed the sandbox).
5. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The model representation from step 1 is your blueprint — the code is a translation, not a re-derivation. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
6. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording.
7. **Verify** (see Verification below) — do not record an execution you have not checked.
8. **Record** — `orx record --execution <json> --override llm_tokens=<your actual token count>`. The response lists `unrecorded_staged_executions` for this task — if a failed first attempt is sitting there, backfill it with `orx record --from-staged <id>` (verbatim, no re-typing). Read the returned `induction_hints` and `prediction_checks`.
9. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.
10. **On failure** — follow Recovery below before retrying.

## Coupling dimensions (operational definitions)

Structural grouping — the foundation of all memory — keys on these. Supply them accurately or let the framework derive them (it will, from your model representation):

| Dimension | Measures | Derivable? |
|---|---|---|
| resource_coupling | fraction of decision variables appearing in MORE THAN ONE constraint (0 = constraints independent; 1 = fully coupled) | yes — model > code |
| temporal_coupling | fraction of variables indexed by a temporal set (time/period/stage/...) | yes — model > code |
| route_complexity | fraction of variables indexed by a network set (arc/edge/link/...) | yes — model > code |
| semantic_coupling | business-semantic relatedness — invisible to structure | NO — always your call |

Do not guess these from the problem's *name* ("it's a resource allocation problem, so rc must be high") — measure from the constraint structure. Two independent resource constraints means rc≈0, however resource-flavored the problem sounds.

## Verification (before every record)

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
- **status=timeout** — the strategy may be too heavy for this scale. Re-recommend with `--exclude <strategy>` and try the next candidate; record the timeout (it is a fact worth remembering).
- **recommend returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, fall back to `--memory-mode none` (default strategy) or relax your exclusions.
- **Verification fails after a successful solve** (wrong magnitude, wrong direction) — re-derive the model, do not adjust the answer to match expectations.
- **A coupling_warnings entry at profile time** — your supplied value contradicts the structural derivation. Trust the structure (it is measured, not guessed); the derived value is what gets used for grouping anyway.

## Core concepts (terminology is strict)

- **Experience Bank** — append-only episodic facts ("what happened"). Never stores generalizations. The single source of truth.
- **Strategic Bank** — induced commitments ("what will happen"): prediction intervals, calibration tracking, feature predicates. Fully rebuildable from facts (`induce --rebuild`).
- **Conditional statistics** — on-the-fly aggregation over the Experience Bank per (strategy × structural group). Arithmetic, not knowledge; never persisted.
- **Structural group** — problem family + coupling-feature bins (e.g. `resource_coupling ∈ [0.75, 1.0]`).
- **CostVector** — five dimensions, stored raw, never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. Retries are a cost: "wrong model → repair → rerun" must cost more than getting it right.
- **Cold archive** — tombstones of retired entries. Vetoes re-induction of the same failed generalization; `--force` revives only under genuine environment drift.

## Commands

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

### `orx profile --task t.json [--code solve.py]`

Builds a `ProblemProfile`. Coupling derivation priority: task JSON `model` field (best — measured from declared constraints) > solve-script AST (`--code`) > structured `spec` fields > your `annotations.coupling` supply. `semantic_coupling` is never derived. The response includes a `derivation` report (per-dimension value/origin/notes), `model_verification` (L1+L2 issues) when a model was given, and `coupling_warnings` when a supplied value contradicts the structural derivation across a bin boundary.

Result: `result.profile` = `{problem_id, family, scale_features, <four coupling dims>, risk_features, source, annotations}` plus `result.derivation`.

### `orx recommend --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--code solve.py]`

Ranks applicable strategies. Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: matching Strategic entry → conditional statistics → catalog prior.

Result: `result.recommendations[]`, each `{strategy_id, name, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before).

`--memory-mode`: `none` (default strategy only) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

### `orx execute --task t.json --strategy S04 --code solve.py --workspace DIR --solver NAME [--verification basic]`

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector with `llm_tokens=0` — that dimension is yours to backfill). **Nothing is recorded yet.**

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

- **induction_hints after record**: hints are evidence, not orders. C1 (strategy contrast contradicting priors), C2 (prior divergence), C3 (in-group drift), C4 (fallback exercised), C5 (cross-family reproduction — suggests L2), C6 (stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all.
- **status**: `candidate` = plausible, unproven. `validated` = ≥5 predictions, ≥70% hit rate. `suspect` = 3 consecutive misses, downweighted ×0.5 — treat its estimates as warnings, not facts. `dormant` = not consulted for 10 tasks, excluded from matching (wakes on a future hit).
- **confidence & cross_family**: cross-family generalizations (L2/L3 entry matching a family absent from its provenance) are discounted and labelled — weigh them accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Always `--dry-run` first and review the plan.

## Anti-patterns (do not do these)

- Do not skip the model representation and jump to solver code — especially for highly coupled problems, where modeling errors are most expensive.
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

- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers. Read before writing your first model.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), the scope ladder, forward validation, citation binding. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, cross-family tighten). Read when unsure how the pieces fit together in practice.
