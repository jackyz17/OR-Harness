# Induction: evidence, methods, and verification

Use this reference to review completed episodes and create or revise reusable strategies.

Contents: [Read evidence](#1-read-evidence) · [Form a strategy](#2-form-or-revise-a-strategy) · [Scope and verification](#3-match-verification-to-the-claim) · [Submit](#4-submit-and-check-publication) · [Reuse and revise](#5-reuse-and-revise)

## 1. Read evidence

Read a bounded batch from the evidence bank:

```bash
orx induction-material [--strategy S] [--task T] [--limit N] [--cursor C] [--related-top-k K]
```

Inspect task semantics and constraint relationships, the actual method, outcomes, task checks, failures, cost, and the chronological attempt chain. Inspect associated code/model/trajectory when the summary lacks the method or structure needed for reasoning. Review `existing_knowledge` before deciding whether to add, revise, merge, or leave it unchanged.

**One task at a time, without losing cross-task material.** For a fast per-task review, read THIS task's chain with `--task T`. The response then fills `related_history` with a SMALL semantically related set from other tasks, so the narrowing does not hide a comparable method, a failure, or a boundary case:

- `query_basis` is built from the batch's OWN recorded method (performed preferred, else the plan marked `planned_only`) plus the family — **no model call**, and the outcome / task number / solver name are deliberately excluded so the search is not biased toward successes. You may rewrite the query when researching a specific tool-chain fix.
- `executions[]` / `knowledge[]` are UNFILTERED: a failed or cross-cell execution and an unpublished entry are exactly the material a boundary check needs, so they are not removed by the online admission filter.
- Three facts stay separate: `no_hits: true` (ran, matched nothing — **not** proof that no counterexample exists), `failure` (could not run at all — reported with a hint, never as "nothing similar exists"), and `degraded_layers`.
- `--related-top-k K` sets the budget (default 5; `0` disables the channel and restores the plain whole-batch behaviour). A `similarity` value is a DISCOVERY signal, never support strength; a hit does not raise a claim's support, and the reviewer still predicts, executes and checks what it picks.

- `execute --method` records a plan. A matching `method_performed` receipt or inspected execution artifacts support what actually ran. Keep the source explicit; a plan alone is not performed-method evidence.
- A solver status and an independent `task_check` are separate facts. An absent task check is unobserved, not passed. An `error` does not establish model infeasibility.
- Several attempts with one `task_id` form one task chain. Preserve failed attempts and their costs; do not count retries as independent tasks.
- Successes, failures, different method names, and different structural cells may all inform an abstraction. Sample counts and observation hints organize facts; they do not determine which material may be reviewed.
- Read missing fields as unknown. Follow `budget.next_cursor` for omitted material when it is needed; do not enlarge the batch merely to collect more records. A cursor is a READ POSITION, never a record that the earlier material was already induced.

A structural cell organizes statistics and assists retrieval. Similar coupling scores do not establish the semantic premises of a method; two unknown profiles do not establish structural similarity. Compare the actual relationships before pooling evidence or transferring a technique.

## 2. Form or revise a strategy

Work from the execution content, using these steps:

1. **Compare relationships.** Identify how constraints, variables, and decisions interact. Shared solver names or successful outcomes do not establish a common mechanism.
2. **Explain the operative step.** Identify the modeling, decomposition, search, checking, or repair step that matters. Explain why it works and state the premises it requires. A before/after difference alone does not establish causality.
3. **Check the boundary.** Vary a necessary premise in your reasoning and examine what fails. Distinguish loss of correctness, feasibility, and efficiency. A hypothetical counterexample guides reasoning; it is not an observed execution.
4. **Separate support.** Distinguish mathematical reasoning, observed task results, and measured quality/cost/risk effects. State which source supports each part and what remains unchecked.
5. **Compare existing knowledge.** Add a distinct mechanism, revise an existing strategy, merge duplicates, narrow an unsupported condition, or produce nothing. Entry count is not the objective.

Write a compact strategy:

**condition → operation → reason → expected effect → boundary**

Cite representative executions and relevant counterexamples. Include measured cost and risk when available; leave unmeasured effects unknown. A strategy may be a reusable step rather than an entire solve plan or a solver choice.

## 3. Match verification to the claim

Separate the following conclusions, even when one execution suggests all three:

| Conclusion | Required support | Permitted scope |
|---|---|---|
| Execution observation | Recorded method, outcome, and any independent task check | What happened on the cited task |
| Conditional method correctness | An inspectable argument covering the premises, transformation, and solution preservation | Tasks satisfying the checked premises; empirical performance remains unestablished |
| Empirical advantage | Comparable measured executions and checks for the claimed quality, cost, or risk effect | The tested tasks and conditions; broader performance remains uncertain |

One execution can reveal a conditional method. Check its argument rather than requiring a second task to substitute for reasoning. Independent tasks are needed to support broader empirical effects; two tasks are a minimum runtime support threshold, not a proof of general performance.

### Method example: bound and monotone reduction

For an integer problem with `Y >= 0`, `X + Y <= 1000`, and `X >= 2Y + 300`, these constraints imply `3Y <= 700`, hence `Y <= 233`.

To justify reducing two-dimensional search to one-dimensional enumeration, check:

1. **Coverage:** the enumerated finite range contains every feasible integer `Y`.
2. **Subproblem feasibility:** for each `Y`, derive the feasible integer `X` interval using all applicable constraints and variable bounds; skip empty intervals.
3. **Boundary optimality:** with `Y` fixed, establish monotonicity of the objective in **X** and choose its minimizing or maximizing feasible boundary as appropriate.
4. **Global preservation:** examine every feasible `Y` and compare its best `X`; this covers an optimum of the original problem.

Inspect the actual model and code to confirm these premises and steps. Record the argument and its source in the claim or applicability note. Merely checking `Y >= 0`, a feasible solution, or an `optimal` status does not verify this reduction.

A loose but valid outer bound can preserve correctness while increasing cost. A bound that excludes feasible solutions, omitted feasibility constraints, or an unjustified boundary choice can destroy solution preservation. Extending the method beyond an interval-valued subproblem requires a new argument.

### Empirical example: lower cost at comparable quality

Compare baseline and proposed-method executions on the same tasks, with compatible settings, the same measurement scope, and measured values on both sides. Check the quality condition and the claimed cost dimension separately. Count independent tasks actually covered by those checks.

Use paired comparisons for task-level advantages. An unpaired group comparison can describe those groups, but does not isolate a method effect when task difficulty or other conditions differ. Do not convert a crashed baseline or a missing metric into a fabricated numerical value.

### Framework checks

The framework computes declared assertions over recorded facts. Choose assertions that cover the claim, not merely ones likely to pass.

| Assertion | What it checks |
|---|---|
| `probe` with `roles`, `path`, and `equals` / `min` / `max` / `in` | The referenced record values satisfy the declared condition |
| `status` with `roles` and `status` | The referenced execution statuses match |
| `comparison` with `metric`, `roles_a`, `roles_b`, `direction`, and `min_gap` | The measured difference between the specified sides |
| `code_unchanged` with `roles` | The referenced records share a code hash |

For a comparison, `direction` describes side A relative to side B. `mode: "paired"` uses same-task pairs; unmatched records do not establish paired support. `aggregation: "all"` requires every pair to meet the condition; `"mean"` checks the batch average. Match the wording to that scope.

Report the verdict accurately:

- `verified`: the declared computable assertions passed within their recorded scope.
- `fact_checked`: execution facts were read; the method argument and causal or transferable claims were not thereby certified.
- `insufficient_evidence`: a required fact, metric, probe path, or comparison counterpart was unavailable.
- `refuted`: a computable assertion failed on relevant evidence.

Keep agent-checked derivations and premises distinct from framework-computed checks. The current runtime has no automatic mathematical-proof verdict. A natural-language argument must be inspected on its merits; do not represent it as framework-certified correctness. Identify unchecked claim parts even when other assertions pass.

## 4. Submit and check publication

Submit one strategy per entry with `orx induce --relation '<json>'`. Use actual execution IDs and role names that explain their part in the claim. The framework reads task identities from the records; do not create task IDs to satisfy a publication count.

The following is a template. Replace the execution ID and example family with recorded values, and include only a method argument you have actually checked:

```json
{
  "subject": "method:bound_then_monotone_enumerate",
  "kind": "conditional_fact",
  "claim": "For a two-variable integer problem with a valid finite Y range and an interval-valued feasible X subproblem, enumerate all Y and choose the objective-best X boundary after establishing monotonicity in X. This preserves an optimum; it does not establish a cost advantage.",
  "method": {
    "name": "bound_then_monotone_enumerate",
    "steps": [
      "derive a valid finite integer Y range",
      "derive the feasible integer X interval for each Y",
      "choose the objective-best X boundary and compare all Y"
    ]
  },
  "evidence": [
    {"execution_id": "ex_REAL_EXECUTION_ID", "role": "observed_application"}
  ],
  "conditions": {
    "predicates": {"family": "planning"},
    "note": "Check range coverage, per-Y feasibility, monotonicity in X, and exhaustive enumeration against the model and executed code. Record the actual argument and observed result; report any unchecked premise."
  }
}
```

Keep the reusable method and its conditional argument distinct from the task-specific observation. `conditional_fact` is the existing runtime route for a one-task finding; its `single_observation` / `unproven` provenance is not a correctness verdict. Reuse requires checking the argument and premises against the new task.

For a measured cost advantage, add checks inside the empirical claim's `check` block, for example:

```json
{
  "assertions": [
    {
      "kind": "comparison", "metric": "quality",
      "roles_a": ["proposed"], "roles_b": ["baseline"],
      "direction": "higher", "min_gap": 0.0,
      "mode": "paired", "aggregation": "all"
    },
    {
      "kind": "comparison", "metric": "cost:llm_tokens",
      "roles_a": ["proposed"], "roles_b": ["baseline"],
      "direction": "lower", "min_gap": 1.0,
      "mode": "paired", "aggregation": "all"
    }
  ]
}
```

Cite both roles on at least two independent tasks for that empirical claim. These comparisons check non-degraded recorded quality and lower token cost; they do not establish method correctness or independently validate task answers. Match metrics, quality requirements, tolerances, and wording to the actual evidence.

Use one check source: the embedded `check`, or the supported standalone `--verify` argument. Inspect the returned verification scope and `publication.reasons`.

**Current runtime publication:**

| Entry kind | Publication requirement |
|---|---|
| `conditional_fact` | `fact_checked` or `verified`, not stale; one task may suffice |
| Other kinds | `verified`, not stale, with at least two distinct tasks in the verification scope |

These are current interface rules. They do not define scientific validity, certify a mathematical argument, or guarantee generalization. Do not overstate a claim to fit a kind or treat a published fact as a validated performance rule.

Use `conditions.predicates` for the supported profile keys: `family`, `resource_coupling`, `temporal_coupling`, and `route_complexity`. Put semantic premises in the claim or `conditions.note`; check them before use. Unsupported predicate keys produce unknown applicability, not a satisfied condition.

## 5. Reuse and revise

- Read published entries through ordinary recall recommendations, including their claim, verification scope, provenance, and semantic premises. Treat unknown applicability as requiring inspection.
- Inspect held claims or use `--include-unverified` during offline review. Unverified or refuted entries may inform investigation; they do not establish a usable advantage.
- Revise an existing entry with its identity or `target_entry_id`. A substantive edit requires fresh checks; without them, the previous verdict becomes stale. Merge duplicates by revising one entry and retiring the other.
- Assess a new miss against the full claimed conditions. A task violating a necessary premise is outside the method's scope; a relevant failure within scope may require narrowing, refutation, or repair. Structural-cell membership alone does not settle this.
- Forward calibration adjusts confidence from later observations; it does not replace claim-specific verification. Review lifecycle changes returned by induction.
- Retired patterns remain in the cold archive. Use `--force` only when new evidence or a changed environment justifies reopening the claim; record the reason.
