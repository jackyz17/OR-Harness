# Concepts: why OR-Harness is built this way

For agents that need the "why" behind the mechanisms. Everyone else: SKILL.md is sufficient.

## The research question

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

The conceptual loop:

```
P_t → Problem Profiling → Strategy Selection → Strategy Execution
    → Outcome + CostVector → Execution Evidence Bank (E_t)
    → Induction/Consolidation → Strategic Knowledge Bank (M_strategic)
    → S_{t+1}
```

E_t is factual execution evidence (what actually happened); M_strategic is derived strategic knowledge (what to do next time). Both layers are necessary — neither alone is memory.

## Structural grouping and coupling derivation

Structural groups — the similarity keys for all memory — are one (family,
structural cell, strategy) triple each: the observations that may be
aggregated together. A cell is the measurable coupling dims quantized to the
four intervals `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, with an
unmeasured dimension in its own `[unknown]` cell. Structure conditions the
statistics but not the ladder: there are no generalization levels, no
`widen`/`tighten`, and no automatic re-scoping. Coupling values are derived by
priority: the task's CIR (the `coupling` field — relations/indexes measured
from the pre-model coupling understanding; the cleanest source) > the `model`
representation (measured from declared constraints) > structured spec fields >
harness-supplied values. `semantic_coupling` is never derived as a scalar.
When a supplied value contradicts the winning structural derivation across a
bin boundary, the profile carries a warning — and the derived value wins for
grouping, because an append-only fact filed in the wrong group would pollute
conditional statistics permanently.

**The signature is frozen before strategy selection.** `execute` uses the same profile as `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only. The solve-script AST path (`profile --code solve.py`) remains available as a diagnostic, but `execute` does not use it.

Post-strategy information (solver diagnostics, model diagnostics) is stored separately in `ExecutionRecord.execution_features` — never in `profile_snapshot`. This keeps task identity stable across executions while preserving execution-time observations for offline induction.

## Coupling-Aware Intermediate Representation (CIR)

The four scalar coupling dimensions are **derived summaries**, not the primary representation. The primary representation is the CIR — an explicit, inspectable set of entities, decisions, constraints, and relations that captures *how* the problem's components interact. The CIR is produced **before** the canonical model, from the natural-language task description, and provides modeling guidance that improves the correctness of the canonical representation (which is written only after the strategy is chosen).

The architectural shift:

```
Old:  Task → scalar coupling scores → strategy matching → model
New:  Task → profile (CIR: entities/relations → understanding + retrieval key)
                → strategy comparison on expected quality/cost/risk
                → canonical model (after the strategy is chosen) → solve.py
```

Key principles:

- **CIR before model**: the CIR does not require a `model` field. The model is an intermediate representation written AFTER the strategy is chosen; re-running `orx profile` then cross-checks CIR ↔ model, but the model is never required to create a CIR.
- **The profile retrieves, evidence decides**: the profile (including the CIR-derived scalar signature) is only the retrieval key — it decides WHICH historical evidence is comparable to this problem. The strategy choice itself rests on the expected quality/cost/risk carried by that evidence (entries, conditional statistics, or world-model predictions), never on the profile alone.
- **Structure first, scalars second**: scalar coupling scores (rc, tc, rx) are derived from the structure; they are summaries, never substitutes for the structure itself.
- **Co-occurrence is structural evidence only**: constraint-variable co-occurrence produces generic `depends_on` edges. A semantic relation (`uses_resource`, `shares_resource`, `competes_for`) requires additional entity/constraint semantics.
- **Domain-general**: all `kind`/`type` fields in the CIR are free-form strings — the schema never hard-codes supply-chain-specific vocabulary.
- **Dual downstream**: CIR feeds both modeling guidance and strategy retrieval via the single `orx profile` entry (rc/tc/rx derive from the CIR structure with priority CIR > model > spec > supplied). Coupling-aware understanding works even when Strategic Memory is empty.

See [references/modeling.md](modeling.md) for the CIR schema, evidence levels, and validation rules.

## Two-layer memory: facts vs derived knowledge

| | Execution Evidence Bank | Strategic Knowledge Bank |
|---|---|---|
| Question answered | "What actually happened?" | "What should we do next time?" |
| Storage unit | ExecutionRecord (episodic fact) | StrategicEntry (commitment) |
| Strategy role | the strategy ACTUALLY used | the strategy RECOMMENDED for this structure |
| Quality / cost | actual (`quality`, `cost`; alias properties `actual_quality` / `actual_cost`) | expected (`expected_quality_hat`, `expected_cost_hat`, `failure_prob`) |
| Artifacts | solver output, diagnostics — artifacts are evidence | none (only future abstracted patterns) |
| Mutation | append-first, fact-preserving; only cost backfill | CRUD, lifecycle, disposal — beliefs may be revised |
| Validation | it *is* the truth — the factual grounding layer | recorded, never rewritten: applicability and intervals are read off the supporting evidence at induction time, and the frozen checks on the facts are what promote, demote, or wake an entry |
| Re-induction | delete it and the factual basis is gone | re-inducible from currently retained evidence (`induce --rebuild`) — exact reconstruction of past entries is NOT a requirement |

A **conditional statistic** ("S04 averaged 0.91 quality over 6 runs in this group") is a query result — a recount. A **Strategic entry** ("S04 will land in [0.75, 0.95] for routing problems with resource coupling ≥ 0.75") is a claim about the future: it carries a prediction interval, a calibration track, and feature predicates that can match across groups. An entry that only restates statistics is redundant and refused at creation.

Strict Evidence ↔ Knowledge linkage is an **induction-time** requirement **by target design**: evidence must support and validate a candidate before admission. This is a migration target — today entries are born `candidate` and forward prediction checks still run online; moving admission validation offline is the next Induction round's job. Once admitted (and also under the current regime), an entry's continued validity does NOT depend on preserving its original supporting evidence — compacting or deleting old evidence never invalidates an entry. New evidence accumulates normally and influences knowledge again at the **next induction cycle**.

When the task carries a CIR, the evidence record preserves a `cir_snapshot` (the coupling representation actually solved), so future induction can re-bin evidence by structural context (`structural_context`) beyond the four scalar coupling features — an extension slot reserved for the strategy-cost phase, not implemented yet.

## The catalog: structural vocabulary, not prior knowledge

The strategy catalog (S01–S10) is a **cold-start vocabulary**: it carries structural knowledge — applicability conditions (which coupling profiles a strategy suits), modeling actions, fallback chains, and solver-family hints — but **no prior quality/cost/risk scores**.

Without accumulated experience, `recall` honestly returns `evidence="no_memory"` with `score=-inf` and `confidence=0`. The system does not fabricate priors to fill the gap. This is deliberate: fabricated priors would prejudice the learning loop toward whatever numbers were guessed, rather than letting evidence accumulate from real executions. The catalog vocabulary is still useful at cold start — applicability filtering narrows the candidate menu — but the selection is your call, not a score ranking.

## CostVector: five dimensions, never folded at rest

`llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`.

Retries and rework are costs. A "wrong model → repair → rerun" trajectory must be more expensive than getting it right the first time, even when solver runtime is similar — otherwise the memory cannot learn that one-shot strategies are worth preferring. Scalarization (`C_scalar = Σ wᵢ·norm(cᵢ)`) happens only inside the selector, with configurable weights; the stored record always keeps the raw five dimensions.

**Unknown ≠ zero.** Each record carries a measured-dimension mask (`cost_measured`) and a solver-runtime provenance (`reported` vs `wall_proxy` — wall-clock used when the script did not report runtime is an explicit proxy, never a precise solver runtime). `retries` counts only *extra* attempts beyond the first: the executor records a measured zero for every single attempt (a first failure is not a retry), and the harness declares retry relationships via `--override retries=...`. Unmeasured dimensions are excluded from means, cost comparisons, and prediction errors — a placeholder zero is never evidence of cheapness. Legacy payloads without a mask keep the value-level inference: non-zero values (including already-backfilled tokens) stay usable, unconfirmable zeros stay unknown.

`llm_tokens` is invisible to the execution sandbox (the harness owns the LLM), so it is backfilled at record time via `--override` (replace by default — idempotent, never double-counted; `--override-mode increment` exists for additional measured amounts within the same attempt).

Cost predictions are validated like quality predictions — but separately. The pre-execution expectation is frozen as a prediction snapshot (`orx predict`: strategy, expected cost with its measured mask, source, scope, per-dimension support). Feedback compares the actual attempt against that snapshot only — same strategy, same scope, and only on dimensions measured on BOTH sides; A's execution never audits B's prediction, an attempt never audits a task-scope prediction, and an unmeasured predicted zero never fabricates an error. Feedback is written into the Execution Evidence (`cost_feedback`) for the next offline induction round — it NEVER demotes, retires, or re-scopes an entry, and never updates calibration online. No snapshot, no fabricated error.

## The disposal ladder (derived layer only)

Facts are permanently neutral — no disposal ever touches an ExecutionRecord's meaning. The derived layer carries the ladder:

| State / action | Trigger | Effect | Reversible? |
|---|---|---|---|
| suspect | 3 consecutive prediction misses (auto) | score ×0.5, warnings attached | yes |
| dormant | 10 tasks unconsulted (auto) | excluded from matching | yes (wakes on hit) |
| retired | your explicit confirmation | moved to cold archive | no (leaves hot store) |
| cold archive | default forever | vetoes re-induction of the same pattern | `--force` only |
| compaction (deferred) | — | lossy evidence compaction is PAUSED: statistics and induction ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics (90 successes + 10 failures would read as 100% failure rate). Re-enabled once the summary consumption contract exists (Cost/Induction rounds) | — |

## Evidence retention (inside the Evidence layer — no new Banks)

- **raw facts** — every execution stays a full, append-only `ExecutionRecord`; GC compaction never touches them today.
- **explicit retention mark** — a harness-supplied `retention_reason` (free text, e.g. `contrast`) reserves representative episodes for future compaction policies. No automatic marking is performed.

Lossy compaction and the summary consumption contract are left to the Cost/Induction rounds.

## Verification philosophy: forward, not backward

An induced entry is a prediction hypothesis. Two separate things must hold
before it counts as published knowledge: its CLAIM must pass an admission
check at induction time (`induce --verify` — a rule holding, a repair working,
or a quality-preserving cost saving, judged by the framework from real
executions), and its PREDICTIONS are then validated by *future* executions
checking its interval — never by self-testing on the training data. This is
why entries are born `candidate`, why intervals are floored by sample size
(n=2 may not claim [0.95, 1.0]), and why promotion requires ≥5 predictions
with ≥70% hit rate ON TOP of the passed admission check.

The only LLM involvement is *phrasing*: you may attach applicability notes at induce time (`--note`). They are stored for the reader and never enter scoring — the framework cannot verify a sentence, so it does not pretend to.

## World-model outcome predictions: shadow hypotheses, never decisions

A *world-model prediction* is a structured hypothesis about what ONE
candidate action would do from the current frozen state: expected status,
feasibility, quality, failure risk, per-dimension cost, and successor state
changes. It is produced by an explicitly configured provider
(`--world-model URL::MODEL`, OpenAI-compatible; credentials from the
environment, never persisted) and lives in its own log table — it is not a
third knowledge bank, not an ExecutionRecord, and not a StrategicEntry.

The discipline that makes these predictions useful rather than corrosive:

- **Shadow mode.** A prediction never changes a recommendation, a score, or
  a route. You decide; the prediction is compared afterwards.
- **Frozen before the act.** The input snapshot is frozen at prediction
  time. Predicting after executing and calling it a forecast is not
  evidence — it is hindsight wearing a costume.
- **Both sides defined, or not compared.** Comparison covers only fields
  the prediction defined AND the execution measured (the same
  both-sides-measured rule as cost feedback). A predicted-but-unmeasured
  cost dimension is listed as not-compared, never scored as zero error.
- **No counterfactuals.** A candidate that never ran has no result. A
  prediction for strategy A is never scored against strategy B's
  execution — the mismatch is recorded and the comparison is skipped.
- **Two costs, never confused.** The PREDICTED cost of the action is a
  hypothesis inside the prediction record; the model call's OWN cost
  (tokens, latency) is real spend, recorded on the prediction and charged
  to the calling action when one is named.
- **Uncalibrated confidence.** The model's self-reported confidence is
  data about the model, not a probability you may bank on. Calibration is
  what the accumulated prediction-vs-outcome record is FOR — that is the
  entire point of the shadow loop.
- **Knowledge stays gated.** Prediction feedback records facts and errors
  online; it never promotes, revises, or publishes anything. Knowledge
  changes only through explicit offline induction with admission
  verification, exactly as before.

## Applicability: the structural cell, not a declared ladder

There is no ladder of generalization levels and no `widen`/`tighten` command.
Each (family, structural cell, strategy) triple is one evidence set, and the
claim induced from it states its own applicability: that family and that cell
(e.g. `rc[0.75,1.00]`). Structurally different regions of one family are
separate evidence sets — one strategy scoring 1.0 at low coupling and 0.1 at
high coupling yields two claims, not one averaged "0.55 everywhere".

Unmeasured structure is never similarity: an `[unknown]` claim matches only a
task whose value is also unmeasured. The cost of the cell rule is honest:
evidence scattered across cells may be too thin to form a claim, in which case
only the statistics remain and no knowledge is invented.

Cross-family transfer is therefore a judgment, not a mechanism: a claim
speaks for the family it was induced from, and applying a routing lesson to
packing is your call. Record those executions and packing earns its own
claim. The frozen per-fact checks (same strategy, attempt scope) are what
make the lifecycle revision evidence-based rather than self-referential.
