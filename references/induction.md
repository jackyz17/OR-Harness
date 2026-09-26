# Induction: patterns, scope, and validation

Read this page when you are deciding whether to induce, working out why a candidate was skipped, or submitting a structured relation claim.

## The offline flow

```text
completed episodes
  induction-candidates            freeze the evidence packages (no model call)
        |
        v   nothing reported = the evidence is below the bar
  predict-capability              what would this operation change?
  compare-capability              one recommendation, or defer
  accept-capability               EXPLICIT accept — runs the operation
        |                         and binds the maintenance fact itself
        v
  induce                          the only place knowledge changes:
        |                         creates/refreshes the claim, replays the
        |                         frozen checks, publishes what was verified
        v
  later real tasks accumulate
        v
  evaluate-capability             did it help? (TASK-EPISODES, work after it)
```

**Entry conditions.** Creating an entry needs ≥2 executions from ≥2 distinct `task_id`s in the same structural cell; publishing one needs a passed admission check as well. `induction-candidates` reports only what clears the bar, so an empty answer means "keep solving and recording" — there is nothing to decide yet. Use `--relation` for a lesson that is not one strategy's statistics.

**What `record` tells you.** After every `record`, cheap detectors may return an `induction_hint`. A hint is a reason to LOOK, never an induction: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## Induction-worthy patterns

After every `record`, cheap detectors run automatically. Any hit produces an `induction_hint` carrying a concrete evidence structure (never a bare counter). The detectors are OR-ed — there is no "all satisfied" state machine, and **hints never induce by themselves**: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Four patterns are worth generalizing. They are named for what they are — no historical criterion numbers are used anywhere.

| Pattern | Fires when | Key refusal condition |
|---|---|---|
| `strategy_contrast` | ≥2 strategies in the same structural cell differ significantly in quality **or** in cost, and the contrast is not already encoded | a difference existing entries already capture is not news (quality and cost are judged separately: an entry explaining the quality gap does not explain the cost gap) |
| `intervention_recovery` | a real result changed after an intervention — within one execution (`failures[].recovery_action`) or across executions (a same-task attempt failed under one solver, then succeeded under another) | failures without an intervention; retrying the same solver is not an intervention |
| `structural_reproduction` | the same strategy shows the same-direction behaviour in ≥2 families **at the same structure** (the reference dimensions must all be measured) | single-family evidence, mixed directions, or an unmeasured structure |
| `advantage_reversal` | the same strategy performs high (≥0.75) in one structural cell and low (≤0.35) in another cell **of the same family**, each with n ≥ 2 | consistent advantage across cells, a single cell, a thin cell, or a cross-family difference |

All detectors are scoped to the target's **structural cell**, not the whole family: evidence from a structurally different region cannot create, dilute, or veto a relation. `structural_reproduction` is the only cross-family pattern and it compares each family's evidence in the same cell; `advantage_reversal` is the only cross-cell pattern and it stays inside one family.

These four are **what draws your attention**, not a classification your final claim must fit. An `intervention_recovery` observation may end up as a modeling rule; a `structural_reproduction` may occur within one family. When you submit a structured relation (below), `kind` is an optional note about the prompt, and the verdict is decided by the assertions you declare — never by the pattern name.

**The lesson is the relation, not the win.** `strategy_contrast` reports a comparison between two strategies under one structural condition; `advantage_reversal` reports where an advantage weakens or flips — an applicability boundary or a counterexample. Neither is a success count. `intervention_recovery` names the change in real outcome; a success after an intervention is **evidence, not proof of causation by itself**, so the hint carries both sides and you draw the conclusion.

All detectors are purely statistical — they detect patterns in observed data, not divergence from fabricated baselines. For `intervention_recovery`, the recovery chain ("solver A failed, switched to solver B, succeeded") is detected from two independent facts — you never need to narrate it into a record. This is why failed executions must be recorded: the chain is invisible if the failure was dropped. The pending staging area guarantees the failure is at least never lost, and `record --from-staged` backfills it verbatim.

Every detector requires **n ≥ 2** supporting executions: a single observation never counts as a pattern — that restraint is deliberate (see worked example 1 in examples.md). Hints count **executions**, not tasks: three runs of one instance can legitimately fire a hint. That is not a bug and not a contradiction — a hint says the numbers look patterned, while the admission gate below decides whether the evidence may become a claim.

## Applicability: family + structural cell

Applicability is not declared on a ladder and no widening command exists, but it IS structurally quantized — the same four intervals the framework used before the ladder was retired:

- **family** — the claim speaks for that family only;
- **cell** — each measurable coupling dimension falls in one of `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, and the claim's applicability is that cell (e.g. `rc[0.75,1.00]`). Unmeasured dimensions get their own `[unknown]` cell.

Why cell rather than observed min/max span: a min/max span is a range in which samples happened to be observed, not a demonstrated region. One family can hold a low-coupling region where a strategy scores 1.0 and a high-coupling region where it scores 0.1; a single span across both would claim "0.55 everywhere in [0.10, 0.90]", erasing the very relation between structure and performance. Cells keep those regions separate — each gets its own claim.

**Unknown is never similarity.** `[unknown]` matches only a task whose value is also unmeasured: "neither side measured it" is a shared absence of evidence, not evidence that the structures agree. This is stricter than the pre-2026 behaviour (which simply omitted the dimension and therefore matched everything) and old entries are NOT rewritten — their wording stays as written; only newly induced claims use the strict reading.

**The trade-off, stated plainly:** evidence scattered across cells may be too thin to form a claim. Four successful runs in one family at four different coupling magnitudes produce four cells holding one observation each, so no claim is created — the statistics remain visible and recall answers from them. The framework does not merge neighbouring cells automatically; run the strategy again at a similar structure, or judge the transfer yourself.

Widening across families is **your** call: the evidence says "in routing, in this cell, S04 held", and nothing about packing. If you decide a routing lesson transfers, act on it yourself — and record the executions, which is what will give packing its own claim.

Counterexamples need no special machinery: a miss inside the claimed cell is a miss *of the claim* (the claim covered that task when it ran), and three consecutive ones demote the entry to `suspect`. Records outside the cell or from another family do not match, so they are neither checks nor counterexamples.

## What it takes to become a claim (admission)

Creating an entry is cheap and reversible, but it is not free of evidence requirements. `induce` refuses to create one unless all of this holds:

1. **≥2 supporting executions** in the (family, cell, strategy) evidence set, all of them executed, attempt-scope facts;
2. **≥2 distinct `task_id`s** — repeating one task is repetition, not reproduction: five runs of the same instance prove something about that instance, not about the strategy in this family. A task-scope total is not attempt evidence and never counts here or in the statistics;
3. **no cold-archive card** for the same (strategy, predicates) pattern.

A refusal is a report, not a loss: the call returns `verification {tasks, required_tasks}` and a `skipped` reason, the executions stay in the Evidence Bank, and recall keeps answering from them as `conditional_stats` until a second task arrives. A standing cold-archive card is reported **before** this gate, because that is the real blocker (collecting a second task would not help while the card stands). The gate guards **creation only** — once a claim exists, any new matching evidence refreshes it, however repetitive, because the claim's value there is calibration, not admission. Dedup also considers **dormant** entries: new evidence refreshes the dormant claim's id (and waking it stays an offline decision) instead of creating a second entry for the same knowledge object. (This is also why `--dry-run` obeys the gate: it reports the same `skipped` reason it would have produced for real.)

Task identity is taken from the recorded `task_id`, so one logical problem solved several times (retry under another solver, larger time limit) is one task. If you legitimately consider two runs independent — different instance drawn from the same distribution — give them distinct `task_id`s; that decision is yours to make and to record.

## Admission verification (offline): what makes a claim published

"Two tasks" is a **support threshold**, not a verification. Once a candidate is formed, its actual claim must be checked before it counts as published strategic knowledge:

offline candidate -> verify the claim -> publish -> accumulate evidence online -> revise at the next offline induction

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

The **framework** computes the verdict from those executions — it does not execute anything itself, and it does not take the candidate's word for anything. `purpose` selects the check family:

- `rule` — the produced result satisfies a declared criterion. At least one must be declared; feasibility alone is a PRECONDITION, not a criterion. Available criteria: `reference_status` (the recorded status must match), `reference_objective` (finite objective within `tolerance`), `semantic_probe` (the framework reads a dotted path in the record — e.g. `quality.objective`, `execution_features.solver_diagnostics.X` — and compares it to `equals` / `min` / `max` / `in`), `semantic_ok` (labelled `agent-declared`: it is recorded, but since the framework cannot re-derive a bare boolean it cannot carry a verdict on its own), and a comparison set.
- `repair` — the fix turned a recorded failure into a usable success **on the same task**. Both sides are required; a failure on one problem and a success on another is two unrelated facts, not a repair.
- `cost_saving` — quality meets `quality_floor`, results are comparable (within `tolerance`), and the declared cost dimension is measurably LOWER on BOTH sides, measured on the **same task** under the **same measurement scope**, with the dimension actually measured on both sides. An attempt cost is never compared against a task-scope total (different quantities), and an unmeasured dimension never proves a saving.

Every supplied execution is evaluated, so a counterexample anywhere in the batch refutes the claim regardless of the order you list records in. A comparison set that repeats the inducing tasks (or re-passes an execution id already used on the other side) is not independent and reports as such.

The evidence must correspond to the candidate: the executions have to be for this strategy and family. The same payload also cannot be reused across induction targets — evidence for another candidate reports `insufficient_evidence` instead of publishing this one by accident.

Three outcomes, deliberately distinct:

- `verified` — at least one substantive check was computed by the framework and every declared criterion held on all supplied evidence;
- `insufficient_evidence` — the check could not decide: an execution did not identify itself, nothing checkable was declared, the evidence does not correspond to this candidate, the two sides are not comparable, or the execution itself failed. **A crash is not a refutation.**;
- `refuted` — the check ran on real evidence and did not hold.

Two things that are NOT verification:

- a program's own printed verdict (`print('{"principle_failed": false}')` proves the program ran, nothing about the claim);
- a candidate's natural-language summary of itself, or a single unchecked boolean (`semantic_ok` with nothing else declared reports `insufficient_evidence` — use `semantic_probe` when the framework should evaluate the condition itself).

Forward calibration cannot substitute either: five frozen hits raise an entry's calibration confidence, but a `candidate` reaches `validated` **only** if its claim was verified. (A `refuted` claim can never be `validated`; the bank refuses the write.)

**Unverified candidates are recorded, not published.** Recall does not present them as strategic knowledge (it falls back to `conditional_stats`), and they cannot serve a cost prediction either. This gates *publishing*, never *trying* — and nothing else fills the gap: there is no built-in directory of methods, so an unpublished claim is simply not presented as knowledge until a verdict exists. Entries written before this mechanism existed have no verdict recorded: they stay usable (discarding accumulated knowledge would be worse) and their provenance stays visible in `inspect`. Use `recall --include-unverified` (offline/inspection) to see the candidate with an explicit warning.

## Forward validation and lifecycle (applied offline)

Recording never changes knowledge. Each `record` freezes one check per matching entry (same strategy, attempt scope) onto the fact: the interval in force at that moment, the observed quality, and whether it fell inside. The next `orx induce` replays those frozen checks:

- **promotion**: n ≥ 5 checks and hit rate ≥ 0.7 **and a verified claim** → `candidate` becomes `validated`
- **demotion**: 3 consecutive misses → `suspect` (score ×0.5, warnings attached)
- **wakeup**: a dormant entry with evidence newer than its last consultation returns to `candidate`
- **retirement** (never automatic): your explicit `orx retire` moves an entry to the cold archive

The report of what changed comes back as `result.revisions` (`forward` counters, `misses`, `transitions`). Counters are recomputed from the whole chronological series, so `consecutive_misses` counts the misses at the END of the series — a later hit clears the streak, and the state transition is applied once per induction rather than once per execution.

Prediction intervals are honest to sample size: with n=2 the floor width is 0.50 — you may not pretend to more certainty than the data supports.

## Ability boundaries

- **strategy_contrast / intervention_recovery / structural_reproduction** read the target's structural cell only — never the whole family, so a different region's behaviour cannot drive or dilute a relation. `structural_reproduction` is the one cross-family pattern, and it compares each family in the SAME cell. `advantage_reversal` is the one cross-cell pattern, and it stays inside one family.
- **strategy_contrast** treats quality and cost contrasts independently: entries explaining the quality gap do not explain the cost gap.
- **intervention_recovery** detects a recorded intervention within one execution, or a cross-execution chain when the solver changed. Retrying the same solver is not an intervention, and other automatic detection is not attempted this round — if you fixed something, say so in the record.
- **structural_reproduction** is a hint that reproduction happened at one structure across families. It never verifies knowledge and never widens applicability. One task's observation is not transferable knowledge.
- **advantage_reversal** detects a boundary between two cells of one family. It does not claim WHY the advantage flips, and it never merges the cells into one applicability range.
- All four patterns are hints: they never satisfy the admission gate, and they never decide how a submitted relation is verified.

## Structured relation claims (`induce --relation`)

Not every lesson is one strategy's statistics. "When temporal coupling is high, a temporal decomposition must keep its cross-period state" is knowledge about a **structural condition paired with a choice and its consequence** — it may belong to no strategy id at all, and a mean plus a free-text remark cannot carry it.

`induce --relation '<json>'` submits such a claim. The JSON is:

```json
{"subject": "principle:cross_period_state",
 "claim": "the sentence being asserted (yours to write)",
 "evidence": [{"execution_id": "ex_...", "role": "dropped"},
              {"execution_id": "ex_...", "role": "preserved"}],
 "conditions": {"predicates": {"family": "scheduling",
                               "temporal_coupling": [0.5, 1.0]},
                "note": "optional"},
 "check": {"assertions": [ ... ]},
 "kind": "intervention_recovery"}
```

- **`evidence` + `role` is the anchor.** Every reference names a recorded execution and the part it plays in *this* claim. Roles are free strings (`dropped`/`preserved`, `before`/`after`, `strategy_a`, `violation`/`satisfying`, …) — they are not a taxonomy, and nothing forces your claim into one of the four trigger patterns. `kind` is an optional note about what prompted the claim.
- **Identity is derived, never submitted.** Tasks, family, structural cell and strategy ids are read off the recorded facts. You supply ids and roles only, so the same evidence cannot acquire two contradictory identities.
- **`subject` is optional** and matters only for knowledge that does not belong to one strategy. With no host entry, the claim creates a **relation-only entry** whose `strategy_id` is your subject and which carries no statistical claim (`support_n = 0`). With a strategy host (the evidence's own strategy, or a subject naming an existing entry), the relation is appended to that entry's `relations` — the peer evidence still never enters the host's statistics.
- **`conditions`** are the applicability predicates; when omitted they are read off the evidence's own structural cell.

### What the framework checks (`--verify` with `purpose: relation`)

The framework computes **only what your structured declaration makes computable** — it does not parse the claim sentence and does not promise to notice that a sentence overreaches its evidence. Each assertion carries its own semantics:

| Assertion | Checks | Applies to |
|---|---|---|
| `{"kind": "probe", "roles": [...], "path": "quality.feasible", "equals"/"min"/"max"/"in": ...}` | every record of those roles satisfies the probe (the framework reads the dotted path itself) | any claim |
| `{"kind": "status", "roles": [...], "status": "optimal"}` | every record of those roles finished with that status | any claim |
| `{"kind": "comparison", "metric": "quality"\|"cost:<dim>"\|"<dotted.path>", "roles_a": [...], "roles_b": [...], "direction": "higher"\|"lower", "min_gap": 0.1, "mode": "paired"\|"group", "aggregation": "all"\|"mean"}` | the declared difference between the two role groups | numerical claims |

**Pairing is a property of the assertion, not of the batch.** `mode: "paired"` compares only records that share a `task_id`, one per side; records with no counterpart are listed as `unpaired_execution_ids` in the scope and are **not** mixed into the statistic. `mode: "group"` compares the two sides' means over a metric every referenced record measured (a metric measured on only some records is `insufficient_evidence`, never a subset mean dressed up as a claim). A claim may declare one paired and one group assertion — each is evaluated over its own scope.

**`aggregation` decides how an unfavourable sample is treated:**
- `"all"` — every pair must meet the direction and gap. One comparable counterexample refutes the claim. Use it when you are asserting "on every such task".
- `"mean"` — the batch mean must meet the direction and gap. A single negative pair does not refute it. Use it when you are asserting "on average across this batch".

**Only the declared parts are covered.** "Quality is higher **and** tokens are lower" needs a `quality` assertion *and* a `cost:llm_tokens` assertion; declare only the first and the verification scope records the cost part as unchecked. The verdict is never extended to parts you did not declare.

**What `verified` means.** Every declared assertion held over the referenced evidence — "no violation was found **within this scope**", not "true for every future task". The `verification.scope` block names the executions, tasks, roles, and which assertions were checked and unchecked. Verdicts: `verified` / `insufficient_evidence` (nothing computable declared, a metric unmeasured, no counterpart, evidence unusable — **not** a refutation) / `refuted` (an assertion ran on real evidence and failed).

### Publication is per relation, on two conditions

1. its **own** verdict is `verified` and not stale;
2. its verification scope covers **≥2 distinct tasks**.

Condition 2 is the same independence rule the statistical gate uses: a single-task repair is a verified **fact about that task**; transferring it to future tasks is a knowledge claim and needs independent evidence. A single-task relation is still **saved** (and verifiable as that fact) — it is simply not published, and `publication.reasons` says exactly why.

Neither condition touches the host entry's statistical claim, and the host's admission never grants the relation anything: **a verified relation does not publish the entry's statistics, and a stale or refuted relation does not invalidate another relation.**

### Revision

Re-submitting the same `subject`+`kind` **revises** that relation (the id is derived from the two, so a revision never appends a duplicate). Give distinct relations under one subject distinct `kind`s.

- A **substantive** change (claim text, conditions, evidence set, check) with no fresh verification marks the previous verdict `stale_after_revision` — the old check no longer covers the new claim.
- A **fresh** verification wins outright: it was computed over the incoming evidence, so it is neither kept nor marked stale.
- An **identical** re-submission keeps the verdict.

`recall` reports `newer_evidence_since_verification` on each relation: matching executions recorded after the verdict. It is a visibility annotation, not a lifecycle state — a frozen batch remains a true historical fact, and this count simply tells you the world has moved on. Decide whether to re-verify.

### Reading relations back

`orx recall` carries them in a **`knowledge` section**, separate from `recommendations`. That separation is deliberate: `recommendations` is keyed on the strategy ids memory holds and filtered by `is_publishable`, which speaks about the **statistical** claim — so a verified relation whose host statistics were never verified would otherwise be filtered out, and a relation-only entry names no strategy id at all. `knowledge[]` items carry the claim, conditions, evidence, verification state and scope, `published`, and the newer-evidence count. `--include-unverified` also returns refuted/stale relations, clearly labelled.

In a world-model context the same content reaches `provider_view` through the structural hit, and the entry's boundary text (`risk_conditions`) travels with it — verified knowledge's conditions, edges and verification scope are part of what the knowledge *says*.

### Relation vs statistical admission (do not blur)

| | Statistical claim | Relation claim |
|---|---|---|
| Verification | entry's `verification` block | the relation's own `verification` |
| Checks | `rule` / `repair` / `cost_saving` over one strategy's executions | `relation` assertions over explicitly referenced evidence with roles |
| Gate | ≥2 tasks + ≥2 executions in the cell | own verdict + ≥2 tasks in the relation's scope |
| Publication | `is_publishable(entry)` | `relation_is_published(relation)`, per relation |
| Failure scope | the entry | that one relation |

## Peer evidence as phrasing (`--peer-strategy` / `--peer-cell`)

The older, narrower path: naming another strategy (same cell) or another cell (same strategy) writes one contrast line under the entry's `risk_conditions` and reports it under `peer_relations`. It remains a **phrasing** mechanism — read only to state what the claim was induced against, never a statistic, never satisfying the admission gate, never creating an entry.

When the relation itself is the knowledge you want (verified, scoped, revisable, recallable), use `--relation` instead: it is the structured form of exactly this observation. A plain induction with no peer flags and no relations is unchanged.

## Applicability notes

`induce --note "TEXT"` (repeatable) attaches free text to the entries that call creates or refreshes. Notes are stored verbatim, shown by `inspect`, and sit outside scoring — they are your phrasing for your own future reading, not a validated fact. Peer relations use the same free-text discipline (`risk_conditions` rather than `applicability`, because a relation is a boundary the reader must respect).

## Cold archive (anti-resurrection)

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every induction. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data.
