# Induction: evidence, methods, and verification

Use this reference to review completed episodes and create or revise reusable strategies.

Contents: [Read evidence](#1-read-evidence) · [Form a strategy](#2-form-or-revise-a-strategy) · [Scope and verification](#3-match-verification-to-the-claim) · [Submit](#4-submit) · [Reuse, declare adoption, and manage](#5-reuse-declare-adoption-and-manage)

## 1. Read evidence

Read a bounded batch from the evidence bank:

```bash
orx induction-material [--strategy S] [--task T] [--limit N] [--cursor C] [--related-top-k K]
```

Inspect task semantics and constraint relationships, the actual method, outcomes, task checks, failures, cost, and the chronological attempt chain. Inspect associated code/model/trajectory when the summary lacks the method or structure needed for reasoning. Optionally declare that no new knowledge is needed (`orx induce` with no relations): the review still runs the utility lifecycle and is recorded. Review `existing_knowledge` before deciding whether to add or leave it unchanged.

**One task at a time, without losing cross-task material.** For a fast per-task review, read THIS task's chain with `--task T`. The response then fills `related_history` with a SMALL semantically related set from other tasks, so the narrowing does not hide a comparable method, a failure, or a boundary case:

- `query_basis` is built from the batch's OWN recorded method (performed preferred, else the plan marked `planned_only`) plus the family — **no model call**, and the outcome / task number / solver name are deliberately excluded so the search is not biased toward successes. You may rewrite the query when researching a specific tool-chain fix.
- `executions[]` / `knowledge[]` are UNFILTERED: a failed or cross-cell execution and an unpublished entry are exactly the material a boundary check needs, so they are not removed by the online admission filter.
- Three facts stay separate: `no_hits: true` (ran, matched nothing — **not** proof that no counterexample exists), `failure` (could not run at all — reported with a hint, never as "nothing similar exists"), and `degraded_layers`.
- `--related-top-k K` sets the budget (default 5; `0` disables the channel and restores the plain whole-batch behaviour). A `similarity` value is a DISCOVERY signal, never support strength; a hit does not raise a claim's support, and the reviewer still predicts, executes and checks what it picks.

**Read WHOLE records, not a summary.** The material returns each SELECTED record's full content: the WHOLE task text (by version), the problem's retained CIR structure (constraint expressions/kinds, decision/entity names, relations, coupling groups) beside its counts, the method's full `steps`/`why`/`fallback`, the task check's checked fields and conclusion, the failures and the trajectory. Reading fewer, complete records beats reading many summaries that dropped the decisive bound or reason.

- **Task texts are VERSIONED.** `task_texts` holds each retained version ONCE for the response, keyed `task_id|task_text_digest`; an entry references its version by `task_text_ref`. A version not retained is reported (`unknown`) and is **never** replaced by another version's text.
- **The full CIR is NOT recovered.** What you get is the EXISTING retained structure plus the counts — cost coefficients and bounds are read from the WHOLE task text and the method, and are not required to appear in `problem.cir`.
- **Reading volume is bounded by selection, not by clipping.** `--limit`, `--related-top-k` and the character budget control how many records come back; when the budget is reached, WHOLE records are omitted (reported with `omitted_execution_ids` + `next_cursor`), never silently truncated field-by-field.
- **The batch's saved joint H+ is included** under `joint_hplus`: the H+ stances from the SAME strategy predictions that were made for these executions, with their `assessment` kept apart (`expected` / `none` / `insufficient_basis` / unstated). It is EXPLANATORY — weigh it yourself; it does not gate induction, does not require publication, and is not a post-hoc effect proof.

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
5. **Compare existing knowledge.** Add a distinct mechanism, or produce nothing when existing knowledge covers the finding. Knowledge is ADDITIVE: a revision or a refinement is a NEW entry, not an in-place edit or a merge. Entry count is not the objective.

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

One execution can reveal a conditional method. Check its argument rather than requiring a second task to substitute for reasoning. More independent tasks strengthen empirical claims about broader quality, cost, or risk effects, but the task count is reported as a fact — two tasks are not a proof, and no count substitutes for an inspectable derivation.

### Method example: bound and monotone reduction

For an integer problem with `Y >= 0`, `X + Y <= 1000`, and `X >= 2Y + 300`, these constraints imply `3Y <= 700`, hence `Y <= 233`.

To justify reducing two-dimensional search to one-dimensional enumeration, check:

1. **Coverage:** the enumerated finite range contains every feasible integer `Y`.
2. **Subproblem feasibility:** for each `Y`, derive the feasible integer `X` interval using all applicable constraints and variable bounds; skip empty intervals.
3. **Boundary optimality:** with `Y` fixed, establish monotonicity of the objective in **X** and choose its minimizing or maximizing feasible boundary as appropriate.
4. **Global preservation:** examine every feasible `Y` and compare its best `X`; this covers an optimum of the original problem.

Inspect the actual model and code to confirm these premises and steps. Record the argument and its source in the claim or applicability note. Merely checking `Y >= 0`, a feasible solution, or an `optimal` status does not verify this reduction.

A loose but valid outer bound can preserve correctness while increasing cost. A bound that excludes feasible solutions, omitted feasibility constraints, or an unjustified boundary choice can destroy solution preservation. Extending the method beyond an interval-valued subproblem requires a new argument.

**Counter-check example (a derived argument, not a built-in rule).** For a minimise problem with `X >= Y + 200`, `Y >= 0`, and positive costs `50X + 30Y`, rewrite the objective as `50X + 30Y >= 50(Y + 200) + 30Y = 10000 + 80Y >= 10000`; then verify `(X, Y) = (200, 0)` satisfies the REMAINING constraints to conclude the bound is attained. Note that "X is more expensive than Y" is NOT a necessary condition here — swapping the two cost coefficients still yields the same boundary solution, so a claim that rests on the cost ORDER would be a false generalisation. This is an example of how to DERIVE and counter-check a bound; it is not a framework rule. Derive a technique from the TASK like this AFTER the fact is fine as an argument, but it must be kept distinct from the method the run ACTUALLY performed: do not claim the run used this derivation, or that it measured a speed-up, unless a recorded `method_actual` receipt says so.

**Recording a method (`method_actual` / `--method`).** Write the ACTUAL modelling, transformation, bound, search or verification treatment AND the reason it works — not "set up variables -> call MILP -> read result". For example:

```json
{
  "name": "bound-then-monotone-enumeration",
  "steps": [
    "derive Y <= (1000 - 300) / 3 = 233 from X >= 2Y + 300 and X + Y <= 1000",
    "enumerate integer Y in [0, 233]",
    "for each Y, solve the X subproblem over its feasible interval",
    "keep the best objective across Y (monotone boundary in X)"
  ],
  "why": "the two constraints bound Y independently of the objective, so a finite one-dimensional scan covers every feasible (X, Y)",
  "fallback": "run the full two-variable MILP when the derived bound fails to hold"
}
```

A plan is INTENT, not a performed method: the framework keeps `method_planned` and `method_actual` separate and never copies the plan into the actual, so record the performed method through the run's own `method_performed` receipt (or `--method-actual`) when you have it.

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

## 4. Submit

Submit one strategy with `orx induce --relation '<json>'`. Use actual execution IDs and role names that explain their part in the claim. The framework reads task identities from the records; do not create task IDs to satisfy a publication count.

**Knowledge is ADDITIVE.** Each submission creates a NEW entry with a NUMBER the framework assigns (`1`, `2`, …). The number is the entry's identity: you do not invent it, a re-submission is a SEPARATE entry (never an in-place rewrite), and a revision or a contradiction is its own entry. A retired number is never reused. To point at an existing entry, cite its number in `--used-entry-ids` when you adopt it.

**The framework checks only administrative matters at submission:**

- the claim payload is well-formed and storable;
- every cited execution id exists and is real `executed` evidence;
- the assigned number is valid;
- the write (and the later index sync) succeeds.

**Publication is YOUR decision.** A submitted claim is offered within the scope you declared — the framework does NOT gate on the content and does NOT certify the conclusion. Any `check` block (or a standalone `--verify`) you supply is EVALUATED and RECORDED as your own audit trail on the entry's `verification` block (`verified` / `fact_checked` / `insufficient_evidence` / `refuted`); it is never required to publish. With no declared check, the framework reads the cited facts (`fact_checked`) and says so. A claim it cannot decide is stored as `insufficient_evidence`, honestly labelled.

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

Keep the reusable method and its conditional argument distinct from the task-specific observation. A single task can reveal a conditional method; its `single_observation` / `unproven` provenance is not a correctness verdict, and more independent tasks widen the scope (the distinct-task count is reported as a FACT, never a threshold).

**Declared prediction (optional).** Add `"prediction": {"value": q, "interval": [lo, hi]}` ONLY when you are stating a quality prediction the runs can be calibrated against. It sets `quality_estimated=True` and is what makes the interval checkable: later, a run that DECLARED it adopted this entry is compared against this interval. Omit it and the entry carries no prediction — no hit/miss is ever computed against its defaults, and it is never promoted on them.

A `check` block you include is your audit trail, for example a measured cost comparison:

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

These comparisons check non-degraded recorded quality and lower token cost; they do not establish method correctness or independently validate task answers. Match metrics, quality requirements, tolerances, and wording to the actual evidence.

Use one check source: the embedded `check`, or the supported standalone `--verify` argument. Inspect the returned verification scope in `verification` (the `not_covered` wording states that a passing check covers only the declared checks over the listed samples).

`induce` with NO relations still runs utility maintenance: the offline lifecycle (`revise`) replays the frozen forward checks on existing entries, and the review is recorded as a maintenance action. Knowledge creation and utility maintenance are decoupled.

Use `conditions.predicates` for the supported profile keys: `family`, `resource_coupling`, `temporal_coupling`, and `route_complexity`. Put semantic premises in the claim or `conditions.note`; check them before use. Unsupported predicate keys produce unknown applicability, not a satisfied condition.

## 5. Reuse, declare adoption, and manage

- Read offered entries through ordinary recall recommendations, including their claim, verification state, provenance, and semantic premises. Treat unknown applicability as requiring inspection.
- **Declare adoption**: when a run uses an entry, pass its number in `execute --used-entry-ids` / `record --used-entry-ids` (e.g. `3,5`). This is a DECLARATION, distinct from being recalled (`--adapted-from` cites cases you READ). Passing an empty list (`''`) records "adopted no prior knowledge" as a fact. **Attribution is by NUMBER, never by a shared name**: an entry called `method:monotone_reduction` adopted by a run called `milp_pulp_cbc` is checked normally. A run that does not declare its adoption produces no forward check.
- **Three things are kept apart**: (1) the **adoption record** (which entry numbers a run relied on), (2) the run's **outcome** (answer check, cost, failures), and (3) **prediction calibration** (does the run's observed quality fall in a DECLARED interval). A missing prediction closes (3) only — the adoption record and its outcome still accumulate. An entry that declared no prediction is never scored hit/miss, and **an arbitrary execution failure is NOT automatically a counterexample to the knowledge** — that attribution is your analysis.
- **A fourth, agent-owned judgement: the USE effect.** Verification state records how far the CLAIM was checked, NOT whether USING the entry helped. Record that with `induce --attribute-effect` (repeatable): `{"entry_id": NUMBER, "verdict": helped|neutral|unrelated|refuting, "execution_id"?: ID, "note"?: TEXT}`. The framework never writes one of these itself. A `refuting` verdict drives the existing demotion (the same transition a miss streak uses); the other verdicts are reported only — none promotes, and none proves a mathematical result. This is the channel that lets a **qualitative technique** (one with no numeric prediction) carry a real judgement of its effect without a second scoring system: such an entry still accumulates **adoptions** (a usage fact, counted separately from `n_predictions`), so its real use is visible in the maintenance report even though it can never be calibrated.
- A claim that contradicts or refines an existing one is a NEW entry: do not rewrite the earlier one. Assess a new miss against the full claimed conditions; a task violating a necessary premise is outside the method's scope, while a relevant failure within scope may warrant a new, narrower claim.
- Forward calibration of a DECLARED prediction adjusts an entry's confidence from later observations — it is what promotes (≥5 checks, ≥70% hits) and demotes (3 consecutive misses of a declared prediction) an entry. It does NOT depend on, nor replace, the agent's verification state or the agent's use-effect attribution; the three are separate records. **Adoptions never promote on their own.**
- Retired patterns remain in the cold archive. Use `--force` only when new evidence or a changed environment justifies reopening the claim; record the reason.
