# Induction: criteria, scope, and validation

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## Trigger criteria C1–C6

After every `record`, cheap checks run automatically. Any hit produces an `induction_hint` carrying a concrete evidence structure (never a bare counter). The criteria are OR-ed — there is no "all satisfied" state machine, and **hints never induce by themselves**: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Divergence judgments require **n ≥ 2** supporting executions. A single observation never counts as divergence — that restraint is deliberate (see worked example 1 in examples.md).

| Criterion | Fires when | Key refusal condition |
|---|---|---|
| C1 strategy contrast | ≥2 strategies in the same structural cell differ significantly in quality **or** in cost, and the contrast is not already encoded | a difference existing entries already capture is not news (quality and cost are judged separately: an entry explaining the quality gap does not explain the cost gap) |
| C2 extreme performance | one strategy's observed mean quality in the cell is extreme (high ≥0.75 or low ≤0.35) with n ≥ 2, and no existing entry captures it | n < 2, moderate quality, or already encoded |
| C3 in-group drift | same strategy, same cell, n ≥ 3, trending quality | flat series |
| C4 failure-recovery | a fallback was exercised — within one execution (`failures[].recovery_action`) or across executions (a same-task attempt failed under one solver, then succeeded under another) | failures without recovery; retrying the same solver is not a chain |
| C5 cross-family reproduction | same strategy, same-direction extreme performance in ≥2 families **at the same structure** (the reference dimensions must all be measured) | single-family evidence, mixed directions, or an unmeasured structure |
| C6 stable success | same cell, n ≥ 4, **every record feasible**, zero failures, zero retries (retries measured on all) | any retry, or any infeasible record — an empty `failures` list is not proof of success |

All criteria are scoped to the target's **structural cell**, not the whole
family: evidence from a structurally different region cannot create, dilute,
or veto a contrast. C5 is the only cross-family criterion, and it compares
each family's evidence in the same cell.

Note: trigger criteria no longer reference catalog priors (which have been removed). All criteria are purely statistical — they detect patterns in observed data (strategy contrasts, extreme performance, cross-family reproduction), not divergence from fabricated baselines. For C4, the recovery chain ("solver A failed, switched to solver B, succeeded") is detected from two independent facts — you never need to narrate it into a record. This is why failed executions must be recorded: the chain is invisible if the failure was dropped. The pending staging area guarantees the failure is at least never lost, and `record --from-staged` backfills it verbatim.

Hints count **executions**, not tasks: three runs of one instance can legitimately fire C2 or C6. That is not a bug and not a contradiction — a hint says the numbers look patterned, while the admission gate below decides whether the evidence may become a claim.

## Applicability: family + structural cell

Applicability is not declared on a ladder and no widening command exists, but
it IS structurally quantized — the same four intervals the framework used
before the ladder was retired:

- **family** — the claim speaks for that family only;
- **cell** — each measurable coupling dimension falls in one of
  `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, and the claim's
  applicability is that cell (e.g. `rc[0.75,1.00]`). Unmeasured dimensions
  get their own `[unknown]` cell.

Why cell rather than observed min/max span: a min/max span is a range in
which samples happened to be observed, not a demonstrated region. One family
can hold a low-coupling region where a strategy scores 1.0 and a
high-coupling region where it scores 0.1; a single span across both would
claim "0.55 everywhere in [0.10, 0.90]", erasing the very relation between
structure and performance. Cells keep those regions separate — each gets its
own claim.

**Unknown is never similarity.** `[unknown]` matches only a task whose value
is also unmeasured: "neither side measured it" is a shared absence of
evidence, not evidence that the structures agree. This is stricter than the
pre-2026 behaviour (which simply omitted the dimension and therefore matched
everything) and old entries are NOT rewritten — their wording stays as
written; only newly induced claims use the strict reading.

**The trade-off, stated plainly:** evidence scattered across cells may be too
thin to form a claim. Four successful runs in one family at four different
coupling magnitudes produce four cells holding one observation each, so no
claim is created — the statistics remain visible and recall answers from
them. The framework does not merge neighbouring cells automatically; run the
strategy again at a similar structure, or judge the transfer yourself.

Widening across families is **your** call: the evidence says "in routing, in
this cell, S04 held", and nothing about packing. If you decide a routing
lesson transfers, act on it yourself — and record the executions, which is
what will give packing its own claim.

Counterexamples need no special machinery: a miss inside the claimed cell is
a miss *of the claim* (the claim covered that task when it ran), and three
consecutive ones demote the entry to `suspect`. Records outside the cell or
from another family do not match, so they are neither checks nor
counterexamples.

## What it takes to become a claim (admission)

Creating an entry is cheap and reversible, but it is not free of evidence
requirements. `induce` refuses to create one unless all of this holds:

1. **≥2 supporting executions** in the (family, cell, strategy) evidence set,
   all of them executed, attempt-scope facts;
2. **≥2 distinct `task_id`s** — repeating one task is repetition, not
   reproduction: five runs of the same instance prove something about that
   instance, not about the strategy in this family. A task-scope total is not
   attempt evidence and never counts here or in the statistics;
3. **no cold-archive card** for the same (strategy, predicates) pattern.

A refusal is a report, not a loss: the call returns
`verification {tasks, required_tasks}` and a `skipped` reason, the executions
stay in the Evidence Bank, and recall keeps answering from them as
`conditional_stats` until a second task arrives. A standing cold-archive card
is reported **before** this gate, because that is the real blocker (collecting
a second task would not help while the card stands). The gate guards
**creation only** — once a claim exists, any new matching evidence refreshes
it, however repetitive, because the claim's value there is calibration, not
admission. Dedup also considers **dormant** entries: new evidence refreshes
the dormant claim's id (and waking it stays an offline decision) instead of
creating a second entry for the same knowledge object. (This is also why
`--dry-run` obeys the gate: it reports the same `skipped` reason it would have
produced for real.)

Task identity is taken from the recorded `task_id`, so one logical problem
solved several times (retry under another solver, larger time limit) is one
task. If you legitimately consider two runs independent — different instance
drawn from the same distribution — give them distinct `task_id`s; that
decision is yours to make and to record.

## Admission verification (offline): what makes a claim published

"Two tasks" is a **support threshold**, not a verification. Once a candidate is
formed, its actual claim must be checked before it counts as published
strategic knowledge:

    offline candidate -> verify the claim -> publish -> accumulate evidence
    online -> revise at the next offline induction

`orx induce --strategy S --verify '<json>'` runs the check. The payload is:

```json
{"purpose": "rule | repair | cost_saving",
 "claim": "the sentence being asserted (audit trail only)",
 "check": {"reference_status": "optimal",
           "reference_objective": 100.0, "tolerance": 1e-6,
           "semantic_probe": {"path": "quality.objective", "max": 250.0},
           "dimension": "llm_tokens", "quality_floor": 0.9},
 "executions": [ ... ], "supporting": [ ... ]}
```

The **framework** computes the verdict from those executions — it does not
execute anything itself, and it does not take the candidate's word for
anything. `purpose` selects the check family:

- `rule` — the produced result satisfies a declared criterion. At least one
  must be declared; feasibility alone is a PRECONDITION, not a criterion.
  Available criteria: `reference_status` (the recorded status must match),
  `reference_objective` (finite objective within `tolerance`), `semantic_probe`
  (the framework reads a dotted path in the record — e.g.
  `quality.objective`, `execution_features.solver_diagnostics.X` — and
  compares it to `equals` / `min` / `max` / `in`), `semantic_ok` (labelled
  `agent-declared`: it is recorded, but since the framework cannot re-derive a
  bare boolean it cannot carry a verdict on its own), and a comparison set.
- `repair` — the fix turned a recorded failure into a usable success **on the
  same task**. Both sides are required; a failure on one problem and a success
  on another is two unrelated facts, not a repair.
- `cost_saving` — quality meets `quality_floor`, results are comparable
  (within `tolerance`), and the declared cost dimension is measurably LOWER on
  BOTH sides, measured on the **same task** under the **same measurement
  scope**, with the dimension actually measured on both sides. An attempt cost
  is never compared against a task-scope total (different quantities), and an
  unmeasured dimension never proves a saving.

Every supplied execution is evaluated, so a counterexample anywhere in the
batch refutes the claim regardless of the order you list records in. A
comparison set that repeats the inducing tasks (or re-passes an execution id
already used on the other side) is not independent and reports as such.

The evidence must correspond to the candidate: the executions have to be for
this strategy and family. The same payload also cannot be reused across
induction targets — evidence for another candidate reports
`insufficient_evidence` instead of publishing this one by accident.

Three outcomes, deliberately distinct:

- `verified` — at least one substantive check was computed by the framework
  and every declared criterion held on all supplied evidence;
- `insufficient_evidence` — the check could not decide: an execution did not
  identify itself, nothing checkable was declared, the evidence does not
  correspond to this candidate, the two sides are not comparable, or the
  execution itself failed. **A crash is not a refutation.**;
- `refuted` — the check ran on real evidence and did not hold.

Two things that are NOT verification, however convenient they look:

- a program's own printed verdict (`print('{"principle_failed": false}')`
  proves the program ran, nothing about the claim);
- a candidate's natural-language summary of itself, or a single unchecked
  boolean (`semantic_ok` with nothing else declared reports
  `insufficient_evidence` — use `semantic_probe` when the framework should
  evaluate the condition itself).

`surviving forward calibration` cannot substitute either: five frozen hits
raise an entry's calibration confidence, but a `candidate` reaches
`validated` **only** if its claim was verified. (A `refuted` claim can never
be `validated`; the bank refuses the write.)

**Unverified candidates are recorded, not published.** Recall does not present
them as strategic knowledge (it falls back to `conditional_stats`), and they
cannot serve a cost prediction either. The catalog remains available, so this
gates *publishing*, never *trying*. Entries written before this mechanism
existed have no verdict recorded: they stay usable (discarding accumulated
knowledge would be worse) and their provenance stays visible in `inspect`.
Use `recall --include-unverified` (offline/inspection) to see the candidate
with an explicit warning.

## Forward validation and lifecycle (applied offline)

Recording never changes knowledge. Each `record` freezes one check per matching entry (same strategy, attempt scope) onto the fact: the interval in force at that moment, the observed quality, and whether it fell inside. The next `orx induce` replays those frozen checks:

- **promotion**: n ≥ 5 checks and hit rate ≥ 0.7 **and a verified claim** → `candidate` becomes `validated`
- **demotion**: 3 consecutive misses → `suspect` (score ×0.5, warnings attached)
- **wakeup**: a dormant entry with evidence newer than its last consultation returns to `candidate`
- **retirement** (never automatic): your explicit `orx retire` moves an entry to the cold archive

The report of what changed comes back as `result.revisions` (`forward` counters, `misses`, `transitions`). Counters are recomputed from the whole chronological series, so `consecutive_misses` counts the misses at the END of the series — a later hit clears the streak, and the state transition is applied once per induction rather than once per execution.

Prediction intervals are honest to sample size: with n=2 the floor width is 0.50 — you may not pretend to more certainty than the data supports.

## Ability boundaries (be honest about these)

- **C1/C2/C3/C6** read the target's structural cell only — never the whole
  family, so a different region's behaviour cannot drive or dilute a contrast.
- **C1** treats quality and cost contrasts independently: entries explaining
  the quality gap do not explain the cost gap.
- **C3** detects quality drift; it does not yet cover cost drift.
- **C4** detects a recorded recovery within one execution, or a cross-execution
  chain when the solver changed. Retrying the same solver is not a chain, and
  other automatic detection is not attempted this round — if you fixed
  something, say so in the record.
- **C5** is a hint that reproduction happened at one structure across families.
  It never verifies knowledge and never widens applicability.
- All of C1–C6 are hints: they never satisfy the admission gate.

## Applicability notes

`induce --note "TEXT"` (repeatable) attaches free text to the entries that call creates or refreshes. Notes are stored verbatim, shown by `inspect`, and **never enter scoring** — they are your phrasing for your own future reading, not a verified fact. The framework deliberately does not pretend to validate a sentence.

## Cold archive (anti-resurrection)

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every induction. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data.
