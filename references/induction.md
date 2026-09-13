# Induction: criteria, scope, and validation

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## Trigger criteria C1–C6

After every `record`, cheap checks run automatically. Any hit produces an `induction_hint` carrying a concrete evidence structure (never a bare counter). The criteria are OR-ed — there is no "all satisfied" state machine, and **hints never induce by themselves**: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Divergence judgments require **n ≥ 2** supporting executions. A single observation never counts as divergence — that restraint is deliberate (see worked example 1 in examples.md).

| Criterion | Fires when | Key refusal condition |
|---|---|---|
| C1 strategy contrast | ≥2 strategies in one group differ significantly (quality or cost) **and** existing entries don't already encode the contrast | a difference existing entries already capture is not news |
| C2 extreme performance | one strategy's observed mean quality is extreme (high ≥0.75 or low ≤0.35) with n ≥ 2, and no existing entry captures it | n < 2, moderate quality, or already encoded |
| C3 in-group drift | same strategy, same group, n ≥ 3, trending quality | flat series |
| C4 failure-recovery | a fallback was exercised — within one execution (`failures[].recovery_action`) or across executions (a same-task attempt failed under one solver, then succeeded under another) | failures without recovery; retrying the same solver is not a chain |
| C5 cross-family reproduction | same strategy, same-direction extreme performance in ≥2 families with similar structure | single-family evidence, or mixed directions |
| C6 stable success | same group, n ≥ 4, zero failures, zero retries | any retry breaks stability |

Note: trigger criteria no longer reference catalog priors (which have been removed). All criteria are purely statistical — they detect patterns in observed data (strategy contrasts, extreme performance, cross-family reproduction), not divergence from fabricated baselines. For C4, the recovery chain ("solver A failed, switched to solver B, succeeded") is detected from two independent facts — you never need to narrate it into a record. This is why failed executions must be recorded: the chain is invisible if the failure was dropped. The pending staging area guarantees the failure is at least never lost, and `record --from-staged` backfills it verbatim.

Hints count **executions**, not tasks: three runs of one instance can legitimately fire C2 or C6. That is not a bug and not a contradiction — a hint says the numbers look patterned, while the admission gate below decides whether the evidence may become a claim.

## Applicability is read off the evidence

A claim's applicability is not declared on a ladder and not quantized into
bins. It is the evidence itself:

- **family** — the evidence set the claim came from: one (family, strategy)
  cell is one evidence set, and the claim speaks for that family only;
- **intervals** — for every structural dimension the supporting executions
  measured, the span they actually covered (`resource_coupling ∈ [0.62, 0.94]`).

Both move with the evidence, because `induce` re-reads the claim's predicates
from its supporting records every time it refreshes them. Run the strategy on
a task at another coupling magnitude and the interval follows; a task anywhere
inside the demonstrated span matches the claim; a task outside it gets no
answer (and the statistics view, `evidence: "conditional_stats"`, is still
there).

Widening across families is **your** call, not the framework's: the evidence
says "in routing, at these couplings, S04 held", and nothing about packing. If
you decide a routing lesson transfers to packing, act on it yourself — and
record the executions, which is what will give packing its own claim.

Counterexamples need no special machinery either: a miss inside the claimed
range is a miss *of the claim* (the claim covered that task when it ran), and
three consecutive ones demote the entry to `suspect`. Records outside the
range or from another family simply do not match, so they are neither checks
nor counterexamples.

## What it takes to become a claim (admission)

Creating an entry is cheap and reversible, but it is not free of evidence
requirements. `induce` refuses to create one unless all of this holds:

1. **≥2 supporting executions** in the (family, strategy) evidence set;
2. **≥2 distinct `task_id`s** — repeating one task is repetition, not
   reproduction: five runs of the same instance prove something about that
   instance, not about the strategy in this family;
3. **no cold-archive card** for the same (strategy, predicates) pattern.

A refusal is a report, not a loss: the call returns
`verification {tasks, required_tasks}` and a `skipped` reason, the executions
stay in the Evidence Bank, and recall keeps answering from them as
`conditional_stats` until a second task arrives. A standing cold-archive card
is reported **before** this gate, because that is the real blocker (collecting
a second task would not help while the card stands). The gate guards
**creation only** — once a claim exists, any new matching evidence refreshes
it, however repetitive, because the claim's value there is calibration, not
admission. (This is also why `--dry-run` obeys the gate: it reports the same
`skipped` reason it would have produced for real.)

Task identity is taken from the recorded `task_id`, so one logical problem
solved several times (retry under another solver, larger time limit) is one
task. If you legitimately consider two runs independent — different instance
drawn from the same distribution — give them distinct `task_id`s; that
decision is yours to make and to record.

## Forward validation and lifecycle (applied offline)

Recording never changes knowledge. Each `record` freezes one check per matching entry (same strategy, attempt scope) onto the fact: the interval in force at that moment, the observed quality, and whether it fell inside. The next `orx induce` replays those frozen checks:

- **promotion**: n ≥ 5 checks and hit rate ≥ 0.7 → `candidate` becomes `validated`
- **demotion**: 3 consecutive misses → `suspect` (score ×0.5, warnings attached)
- **wakeup**: a dormant entry with evidence newer than its last consultation returns to `candidate`
- **retirement** (never automatic): your explicit `orx retire` moves an entry to the cold archive

The report of what changed comes back as `result.revisions` (`forward` counters, `misses`, `transitions`). Counters are recomputed from the whole chronological series, so `consecutive_misses` counts the misses at the END of the series — a later hit clears the streak, and the state transition is applied once per induction rather than once per execution.

Prediction intervals are honest to sample size: with n=2 the floor width is 0.50 — you may not pretend to more certainty than the data supports.

## Applicability notes

`induce --note "TEXT"` (repeatable) attaches free text to the entries that call creates or refreshes. Notes are stored verbatim, shown by `inspect`, and **never enter scoring** — they are your phrasing for your own future reading, not a verified fact. The framework deliberately does not pretend to validate a sentence.

## Cold archive (anti-resurrection)

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every induction. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data.
