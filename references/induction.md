# Induction: criteria, scope, and validation

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## Trigger criteria C1–C6

After every `record`, cheap checks run automatically. Any hit produces an `induction_hint` carrying a concrete evidence structure (never a bare counter). The criteria are OR-ed — there is no "all satisfied" state machine, and **hints never induce by themselves**: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Divergence judgments require **n ≥ 2** supporting executions. A single observation never counts as divergence — that restraint is deliberate (see worked example 1 in examples.md).

| Criterion | Fires when | Key refusal condition |
|---|---|---|
| C1 strategy contrast | ≥2 strategies in one group differ significantly (quality or cost) **and** the difference contradicts priors/entries | a difference the prior already encodes is not news |
| C2 prior divergence | one strategy systematically departs from its catalog prior | n < 2, or departure within noise |
| C3 in-group drift | same strategy, same group, n ≥ 3, trending quality | flat series |
| C4 failure-recovery | a fallback was exercised — within one execution (`failures[].recovery_action`) or across executions (a same-task attempt failed under one solver, then succeeded under another) | failures without recovery; retrying the same solver is not a chain |
| C5 cross-family reproduction | same strategy, same-direction advantage in ≥2 families with similar structure | single-family evidence |
| C6 stable success | same group, n ≥ 4, zero failures, zero retries | any retry breaks stability |

C1's cost dimension is the interesting one: when quality is tied and one strategy is markedly cheaper, the *only* thing memory can learn is the cost structure — and that is decision-changing evidence (equivalent choices should consistently favor the cheap side).

C4's cross-execution path deserves emphasis: the recovery chain ("solver A failed, switched to solver B, succeeded") is detected from two independent facts — you never need to narrate it into a record. This is why failed executions must be recorded: the chain is invisible if the failure was dropped. The pending staging area guarantees the failure is at least never lost, and `record --from-staged` backfills it verbatim.

## The scope ladder

An entry's pattern lives on a ladder:

- **L1** — family + fine bins (e.g. routing × rc∈[0.75,1.0]). Narrowest, safest.
- **L2** — fine bins, any family. Cross-family generalization.
- **L3** — coarse bins, any family. Widest, riskiest.

`induce` starts at L1. C5 evidence (independent reproduction across families) justifies `--widen`. Widening is falsifiable: a wide entry makes riskier predictions, and when a cross-family execution misses, the pattern **auto-tightens back down** — the range was wrong, not necessarily the content. A scope miss never demotes an entry to `suspect`; that state is reserved for prediction-content misses.

Cross-problem alignment needs no LLM role schema: the profiler's coupling dimensions are the quantitative version of abstract roles. Predicate intervals merge when the strategy's evidence points the same direction in each family — the merge replaces what an older design spent 500 lines of LLM alignment pipeline on.

## Forward validation and lifecycle

Every `record` automatically checks each matching entry's interval against the observation:

- **hit** → `n_hits += 1`, calibration error updated
- **miss** → `consecutive_misses += 1`
- **promotion** (auto): n ≥ 5 and hit rate ≥ 0.7 → `candidate` becomes `validated`
- **demotion** (auto): 3 consecutive misses → `suspect` (score ×0.5, warnings attached)
- **retirement** (never auto): your explicit `orx retire` moves a suspect entry to the cold archive

Cost predictions run a PARALLEL, warning-only loop: observed cost vs the entry's multiplicative interval → `cost_hit_rate` + per-dimension log-error calibration. Cost misses never touch `consecutive_misses` or the lifecycle — an entry whose quality predictions are perfect but whose costs are volatile stays validated, with an "uncalibrated cost" warning attached for you to weigh.

Induced entries also carry a **mechanism annotation**: the mechanism features of the supporting evidence, aggregated automatically from the provenance records' profiles — measured structure transferred from facts, never narrated. An optional mechanism explanation (your phrasing) is citation-bound like applicability text. This is mechanism ANNOTATION, not causal discovery: the framework moves measured structure; it never infers causality.

Prediction intervals are honest to sample size: with n=2 the floor width is 0.50 — you may not pretend to more certainty than the data supports.

## LLM phrasing and citation binding

The single optional LLM injection point: at `induce --llm-conditions`, you supply applicability/risk text. The framework checks every condition:

1. every `supporting_execution_ids` entry must reference a real record;
2. numeric claims in the text (e.g. "achieves 90%") must agree with the cited records' quality range;
3. accepted conditions are stored `verified: false` and **never enter scoring** — only `R̂` from statistics does — until the described situation recurs and the condition is confirmed.

A condition failing either check is rejected with the reason; the entry itself is still created from the statistical evidence.

## Cold archive (anti-resurrection)

Retired entries leave tombstones (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `--force` lifts a veto — reserve it for genuine environment drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data.
