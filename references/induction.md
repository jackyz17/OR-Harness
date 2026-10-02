# Induction: patterns, scope, and validation

Read this page when you are deciding whether to induce, working out why a candidate was skipped, or submitting a knowledge claim.

## The offline flow

```text
completed episodes
  induction-candidates            freeze the evidence packages (no model call);
        |                         detector-derived candidates carry both sides
        |                         of a contrast, cell candidates carry the gate
  induction-material              read the METHODS, the change, what followed
        |                         and the verification state (no model call)
        v   material=insufficient or unavailable = record how the work was
        |   done / re-check the evidence, do not invent
  (you form the claim: condition -> how -> consequence -> boundary)
        |
  induce --relation               submit it; the framework checks what you wrote
        |
  predict-capability              (optional) what would this operation change?
        |                         declare the relations it forms in the
        |                         operation's config — accept runs them
  compare-capability              one recommendation, or defer
  accept-capability               EXPLICIT accept — runs the operation it
        |                         DECLARED (relations, or a statistical
        |                         refresh when none were declared) and binds
        |                         the maintenance fact itself
        v
  induce                          the only place knowledge changes:
        |                         creates/refreshes the claim, replays the
        |                         frozen checks, publishes what was verified
        v
  later real tasks accumulate
        v
  evaluate-capability             did it help? (TASK-EPISODES, work after it)
```

**Entry conditions.** Creating a statistical entry needs ≥2 executions from ≥2 distinct `task_id`s in the same structural cell; publishing one needs a passed admission check as well. `induction-candidates` reports only what clears the bar, so an empty answer means "keep solving and recording" — there is nothing to decide yet. Use `--relation` for a lesson that is not one strategy's statistics.

**What `record` tells you.** After every `record`, cheap detectors may return an `induction_hint`. A hint is a reason to LOOK, never an induction: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all. Each hint is also **persisted onto the fact that produced it** (`execution_features.induction_hints`), so `induction-candidates` can reuse the detector's own cross-execution evidence — both sides of a contrast, the failed/recovered pair — instead of re-deriving the pattern from bare counts.

**The method is the missing half of the evidence.** `execute --method '<json>'` (or a candidate's `method`) records the PLAN; the solve script's optional `method_performed` receipt (stamped with the attempt's `OR_ACTION_ID`) records what ACTUALLY ran, and the receipt's steps appear in the record's trajectory. Nothing promotes the plan to a fact: an unobserved performance stays `None`. `induction-material` is what you read before writing a claim.

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## Induction-worthy patterns

After every `record`, cheap detectors run automatically. Any hit produces an `induction_hint` carrying a concrete evidence structure (never a bare counter). The detectors are OR-ed — there is no "all satisfied" state machine, and **hints never induce by themselves**: `induce` is your explicit call, and you may induct from your own business knowledge with no hint at all.

Four patterns are worth generalizing. They are named for what they are — no historical criterion numbers are used anywhere.

| Pattern | Fires when | Key refusal condition |
|---|---|---|
| `strategy_contrast` | ≥2 strategies in the same structural cell differ significantly in quality **or** in cost, and the contrast is not already encoded | a difference existing entries already capture is not news (quality and cost are judged separately: an entry explaining the quality gap does not explain the cost gap) |
| `intervention_recovery` | a real result changed after an intervention — within one execution (`failures[].recovery_action`), across executions under the SAME solver when the PERFORMED method changed (a differing `method_actual`, or a declared `intervention`), or across executions when the solver changed. A prior attempt counts as failed when the solver refused OR the answer failed its task check | failures without an intervention; retrying the same solver with NO evidence of a change (a plain retry is not a demonstrated recovery); only a PLAN differing while neither attempt reports what it ran |
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
- **intervention_recovery** detects a recorded intervention within one execution, a same-solver fix when the PERFORMED method changed (a differing `method_actual` or a declared `intervention`), or a cross-execution chain when the solver changed. A prior attempt is a failure when the solver REFUSED **or** the answer FAILED its task check: a model written wrong and solved to a legal optimum is the main modeling error to summarize, so it counts even though `quality.feasible` was true. Only a PLAN differing (neither attempt reports what it ran) does NOT fire — a plan is intent, not an intervention. The hint names WHAT changed and links the two executions; it never claims causation.
- **structural_reproduction** is a hint that reproduction happened at one structure across families. It never verifies knowledge and never widens applicability. One task's observation is not transferable knowledge.
- **advantage_reversal** detects a boundary between two cells of one family. It does not claim WHY the advantage flips, and it never merges the cells into one applicability range.
- All four patterns are hints: they never satisfy the admission gate, and they never decide how a submitted relation is verified.

## Reading the material (`induction-material`)

Semantic induction is a division of labour: the framework ORGANS element the material and checks what you submit; YOU read the material and write the claim.

`orx induction-material [--bundle BUNDLE_ID] [--pattern P] [--strategy S]` returns, for each candidate, the frozen evidence a claim can rest on:

- `methods[]` — per execution, its `planned` method, the method it reports as actually `performed`, and the trajectory steps that really happened;
- `comparisons[]` — the detector's own evidence blocks (both sides of a contrast, the failed/recovered pair), so a claim can cite both sides rather than one;
- `outcome` / `task_check` / `failures` per execution — what followed, and what was actually checked;
- `material_state` — one of four, and the distinction is about WHAT the claim may assert:
  - `sufficient` — at least one PERFORMED method (`method_actual`) is on record: the strongest basis.
  - `sufficient_limited` — a STATISTICAL (conditional-fact) candidate whose evidence reports only PLANNED methods but has at least one PASSED task check. It grounds a CONDITIONAL FACT ("under condition C, method M produced a checked-correct answer"), **not** a performed technique; the state carries `basis: "planned_only"` and the reason says a claim from it must not assert the method was observed running.
  - `insufficient` — nothing to abstract: no method content at all, OR a HOW-TO (method_induction) candidate whose evidence reports only PLANNED methods (a plan is intent, not an observation), OR a planned-only candidate with no passed check (an unverified plan grounds nothing).
  - `unavailable` — the comparison lost a whole side to an exclusion.
- `purpose` — what an induction from this candidate is FOR: `method_induction` (a detector candidate — a comparison or recovery worth abstracting into a technique) or `statistical_refresh` (a structural cell that cleared the sample-count gate, whose claim is its quality/cost/failure estimate). A cell clearing the count gate is NOT by itself a reason to abstract a method. The purpose is what splits `insufficient` from `sufficient_limited` above: a plan is fatal to a how-to claim, but a plan PLUS a passed check can ground a conditional fact.

**`insufficient` is the honest answer to "there is nothing to abstract from".** The evidence holding a strategy name and a mean, or (for a how-to claim) ONLY a plan, means there is nothing observed to abstract — so `induction-material` says so, and the right response is to record how the work was actually done (the script's `method_performed` receipt is the observation; `execute --method` is only the plan). The framework itself never derives a how-to from numbers: `induce`'s statistical path writes no method prose at all. But a plan is not worthless: with a PASSED task check it is `sufficient_limited`, and a LIMITED claim that does not overreach ("the method the agent declared, applied to condition C, produced an answer the declared bases accepted") is honest and allowed.

**`unavailable` means the premise no longer holds.** A candidate is RE-CHECKED against the current evidence before it becomes a claim: an execution excluded since the hint fired drops out, and when that removes a whole side (a contrast's other strategy, a recovery's failed attempt) — or when a later `check-task` has REFUTED the side the hint called a success — the candidate is reported with the reason rather than presented carrying a half-excluded comparison. A hint that merely loses one of several records on a side stays, with the dead id pruned and the prune noted in `trigger_reasons`. The method material is read from the LIVE records, so a later method or cost correction is reflected. **An execution cited by a hint can also leave the evidence window entirely**; when that removes a required side the candidate is reported `unavailable` (honest, not silently upgraded), so a hint whose evidence has aged out simply produces nothing rather than a claim on a stale premise.

Submitting a relation whose evidence reports no method (and whose own claim declares none) still saves the claim — you may legitimately state the method in your own words — but the outcome carries a `material` warning saying the framework did not and will not derive a technique from the numbers, so the claim's basis is visible as numbers-based.

You may cite a candidate directly: an evidence entry `{"bundle_id": "cb_...", "role": "..."}` is expanded into that bundle's frozen execution set. Bundle ids are content-addressed, so the id `induction-candidates` printed still names the same candidate on the next call.

## Knowledge claims (`induce --relation`)

Not every lesson is one strategy's statistics. "When temporal coupling is high, a temporal decomposition must keep its cross-period state" is knowledge about a **structural condition paired with a choice and its consequence** — it may belong to no strategy id at all, and a mean plus a free-text remark cannot carry it.

**The unit of knowledge is the ENTRY, and one entry is ONE claim.** A second independent claim is a second entry; there is no host lookup, no embedded claim list, and no shared verdict. `induce --relation '<json>'` submits one claim:

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

An evidence entry may instead be `{"bundle_id": "cb_...", "role": "..."}` — see `induction-material`.

A claim submission is a knowledge WRITE like any other: it records a maintenance action with the pre state, a knowledge delta (`entries_created` / `entry_changes`), the index result, and the same knowledge feedback. Re-submitting the same claim revises that entry, not a duplicate.

- **`evidence` + `role` is the anchor.** Every reference names a recorded execution and the part it plays in *this* claim. Roles are free strings (`dropped`/`preserved`, `before`/`after`, `strategy_a`, `violation`/`satisfying`, …) — they are not a taxonomy, and nothing forces your claim into one of the four trigger patterns. `kind` is an optional note about what prompted the claim.
- **Identity is derived, never submitted.** Tasks, family, structural cell and strategy ids are read off the recorded facts. You supply ids and roles only, so the same evidence cannot acquire two contradictory identities.
- **`subject` is optional.** With no subject, the evidence's single strategy names the entry; with a subject, that free-form name is the entry's `strategy_id` (e.g. `principle:cross_period_state`). An entry that states a claim but has no statistical support (`support_n = 0`) is a **claim-only entry**: it makes no quality/cost/failure claim, and its publication is decided by its single `verification` block alone.
- **`conditions`** are the applicability predicates; when omitted they are read off the evidence's own structural cell.
- **One entry, one identity.** The entry a submission revises is found by `strategy_id` + structural cell + `kind`. Two independent claims under one subject (a different cell, or a different kind) are separate entries, and neither inherits the other's verification. A claim and a statistical claim under the same strategy id are also separate: the claim never attaches to the statistical entry.

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

**Only the declared parts are covered, and the block says so.** "Quality is higher **and** tokens are lower" needs a `quality` assertion *and* a `cost:llm_tokens` assertion; declare only the first and the verification scope records the cost part as unchecked. The verdict is never extended to parts you did not declare.

**What `verified` means — and what it does not.** Every declared assertion held over the referenced evidence: "no violation was found **within this scope**". It is NOT a proof that the natural-language claim is correct, causal, or broadly general. The verification block carries both `checked` (the assertions, executions and tasks the verdict covered) and `not_covered` (that explicit limitation), plus `scope` naming the executions, tasks, roles, and which assertions were checked and unchecked. Verdicts: `verified` / `insufficient_evidence` (nothing computable declared, a metric unmeasured, no counterpart, evidence unusable — **not** a refutation) / `refuted` (an assertion ran on real evidence and failed).

### Publication: two conditions

1. the entry's verdict is `verified` and not stale;
2. its verification scope covers **≥2 distinct tasks**.

Condition 2 is the independence rule the statistical gate uses: a single-task repair is a verified **fact about that task**; transferring it to future tasks is a knowledge claim and needs independent evidence. A single-task claim is still **saved** (and verifiable as that fact) — it is simply not published, and `publication.reasons` says exactly why.

### Revision

Re-submitting the same identity (`strategy_id` + cell + `kind`) **revises** that entry.

- A **substantive** change (claim text, conditions, evidence set, method) with no fresh verification marks the previous verdict `stale_after_revision` — the old check no longer covers the new claim.
- A **fresh** verification wins outright: it was computed over the incoming evidence, so it is neither kept nor marked stale.
- An **identical** re-submission keeps the verdict.

A claim must still be checked against CURRENT real evidence when it is created or substantively revised: an old source id expiring (the evidence window) never blocks a later update, and never by itself backs a revised claim.

### Reading claims back

A published claim reaches you through the **ordinary recommendation path** (`recall().recommendations[]`): its entry is an admitted entry, so it appears with its expected effect and, under `knowledge`, its verification state and stated claim. There is no separate knowledge section — one claim, one retrieval channel.

Unpublished claims are the ones that would not survive that filter: `recall().held_claims[]` carries them (a claim-only entry whose `strategy_id` is a free-form subject, or an entry not yet verified), each with its verification state, plus `newer_evidence_since_verification` — matching executions recorded after the verdict, a visibility annotation rather than a lifecycle state. `--include-unverified` fills this section with refuted/stale claims too, clearly labelled.

> Legacy note: a store written before knowledge was unified may still carry per-entry `relations` lists. `orx migrate-relations` converts each relation into its OWN claim entry, with its verification copied verbatim; it is idempotent and a `--dry-run` writes nothing. `recall` reads migrated claims through the ordinary path.

## Relation vs statistical admission (do not blur)

| | Statistical claim | Knowledge claim |
|---|---|---|
| Where it lives | the entry's interval fields + its `verification` block | the entry's `claim` block + the SAME `verification` block |
| Checks | `rule` / `repair` / `cost_saving` over one strategy's executions | `relation` assertions over explicitly referenced evidence with roles |
| Gate | ≥2 tasks + ≥2 executions in the cell | own verdict + ≥2 tasks in the claim's verification scope |
| Publication | `is_publishable(entry)` | the same single rule (verified + not stale + ≥2 tasks) |
| Independence | one evidence set owns one entry | one claim owns one entry |

## Peer evidence as phrasing (`--peer-strategy` / `--peer-cell`)

REMOVED. Naming another strategy or cell used to write one contrast line under the entry's `risk_conditions`. That was a sentence the framework could not check, and the observable form of the same observation is a knowledge claim — submit it with `--relation` (below), citing the executions on both sides. Existing entries that already carry such text keep it; nothing writes new ones.

## Applicability notes

`induce --note "TEXT"` (repeatable) attaches free text to the entries that call creates or refreshes. Notes are stored verbatim, shown by `inspect`, and sit outside scoring — they are your phrasing for your own future reading, not a validated fact.

## Cold archive (anti-resurrection)

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every induction. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data.
