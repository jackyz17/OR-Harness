# Concepts: why OR-Harness is built this way

For agents that need the "why" behind the mechanisms. Everyone else: SKILL.md is sufficient.

## The research question

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

The conceptual loop:

```
P_t → Problem Profiling → Strategy Selection → Strategy Execution
    → Outcome + CostVector → Experience Bank (E_t)
    → Induction/Consolidation → Strategic Bank (M_strategic)
    → S_{t+1}
```

E_t is episodic experience; M_strategic is generalized strategy knowledge. Both layers are necessary — neither alone is memory.

## Structural grouping and coupling derivation

Structural groups — the similarity keys for all memory — are built from problem family plus coupling-feature bins. Coupling values are derived by priority: the task's `model` representation (measured from declared constraints; the cleanest source) > solve-script AST > structured spec fields > harness-supplied values. `semantic_coupling` is never derived. When a supplied value contradicts the structural derivation across a bin boundary, the profile carries a warning — and the derived value wins for grouping, because an append-only fact filed in the wrong group would pollute conditional statistics permanently.

## Two-layer memory: commitments vs recounts

| | Experience Bank | Strategic Bank |
|---|---|---|
| Question answered | "What happened?" | "What will happen?" |
| Storage unit | ExecutionRecord (fact) | StrategicEntry (commitment) |
| Mutation | append-only (+ cost backfill) | CRUD, lifecycle, disposal |
| Rebuildable? | it *is* the truth | fully, via `induce --rebuild` |

A **conditional statistic** ("S04 averaged 0.91 quality over 6 runs in this group") is a query result — a recount. A **Strategic entry** ("S04 will land in [0.75, 0.95] for routing problems with resource coupling ≥ 0.75") is a claim about the future: it carries a prediction interval, a calibration track, and feature predicates that can match across groups. An entry that only restates statistics is redundant and refused at creation.

## CostVector: five dimensions, never folded at rest

`llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`.

Retries and rework are costs. A "wrong model → repair → rerun" trajectory must be more expensive than getting it right the first time, even when solver runtime is similar — otherwise the memory cannot learn that one-shot strategies are worth preferring. Scalarization (`C_scalar = Σ wᵢ·norm(cᵢ)`) happens only inside the selector, with configurable weights; the stored record always keeps the raw five dimensions.

`llm_tokens` is invisible to the execution sandbox (the harness owns the LLM), so it is backfilled at record time via `--override`.

## The disposal ladder (derived layer only)

Facts are permanently neutral — no disposal ever touches an ExecutionRecord's meaning. The derived layer carries the ladder:

| State / action | Trigger | Effect | Reversible? |
|---|---|---|---|
| suspect | 3 consecutive prediction misses (auto) | score ×0.5, warnings attached | yes |
| dormant | 10 tasks unconsulted (auto) | excluded from matching | yes (wakes on hit) |
| retired | your explicit confirmation | moved to cold archive | no (leaves hot store) |
| cold archive | default forever | vetoes re-induction of the same pattern | `--force` only |
| compacted ledger line | gc, groups covered by entries | raw rows → (n, mean Q, mean C, failures) | statistics survive; trajectories do not |

## Verification philosophy: forward, not backward

An induced entry is a prediction hypothesis. It is validated by *future* executions checking its interval — never by self-testing on the training data. This is why entries are born `candidate`, why intervals are floored by sample size (n=2 may not claim [0.95, 1.0]), and why promotion requires ≥5 predictions with ≥70% hit rate.

The only LLM involvement is *phrasing*: you may write applicability text at induce time, but citations are verified against real records, and unverified text never enters scoring.

## Scope ladder: generalization as a mechanism

Patterns live on L1 (family + fine bins) → L2 (fine bins) → L3 (coarse bins). Induction asks "which rung does the evidence support?" Evidence from one family supports only L1; independent reproduction across families (criterion C5) justifies proposing L2. Widening is falsifiable: a wide entry makes riskier predictions, and a cross-family miss tightens the pattern back down — the generalization range is itself a continuously tested hypothesis.
