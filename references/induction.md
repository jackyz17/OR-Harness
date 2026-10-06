# Induction: scope, material, and validation

Read this page when you are deciding whether to induce, working out why a submission was skipped, or submitting a new strategy.

## The offline flow

```text
completed episodes
  induction-material              read a BATCH straight from the evidence bank
        |                         (no model call, no candidate, no count gate):
        |                         success / failure / cross-cell / cross-name
        |                         all visible, retries marked non-independent
        |                         -- the ONLY material entry point
        v   method.basis / missing   a plan, a performed method, or nothing:
        |   (no material_report)     record how the work was done; the
        |                            framework does not abstract it for you
  (you form the new strategy: condition -> how -> consequence -> boundary)
        |
  induce --relation               submit it; the framework checks what you wrote
        |
  predict-capability              (optional) what would this operation change?
        |                         declare the relations it forms in the
        |                         operation's config — accept runs them
  compare-capability              one recommendation, or defer
  accept-capability               EXPLICIT accept — runs the relations the
        |                         operation DECLARED and binds the
        |                         maintenance fact itself. There is NO
        |                         statistical fallback: a candidate that
        |                         declares no relations is refused
        v
  induce --relation               the ONLY place knowledge changes: creates
        |                         or revises the strategy entry, replays the
        |                         frozen checks, publishes what was verified
        v
  later real tasks accumulate
        v
  evaluate-capability             did it help? (TASK-EPISODES, work after it)
```

**Read first, gate later.** The material you review comes from ONE place: `orx induction-material` reads a BATCH of completed tasks straight from the evidence bank — no candidate and no sample-count gate. Success, failure, cross-cell and cross-method-name material all reach you, and a repeated run of one task is marked `independent_task: false` so repetition is never mistaken for cross-task support. There is no separate "lead" command: the batch IS the material.

**"At least two tasks" gates PUBLICATION, not visibility.** A transferable strategy needs the same mechanism on ≥2 distinct `task_id`s, and it must also pass its declared check to be published. But the material is VISIBLE from the first execution: the batch reports every attempt with its facts, and the distinct-task count (`cross_task_hint`) is a FACT you weigh, never a bar on what you may read or cite. Different tasks, strategy ids and cells may be cited TOGETHER — the framework no longer requires a shared method name or a shared cell.

**What `record` tells you.** `record` does NOT interpret the fact it stored: it emits no induction labels and no hints. Deciding what a fact MEANS is your job — read the material with `orx induction-material`; `induce --relation` is your explicit call, and you may induce from your own business knowledge with no evidence of your own at all. (Old records may still carry an `execution_features.induction_hints` key left by the retired detectors; nothing reads it any more.)

**The method is the missing half of the evidence.** `execute --method '<json>'` (or a candidate's `method`) records the PLAN; the solve script's optional `method_performed` receipt (stamped with the attempt's `OR_ACTION_ID`) records what ACTUALLY ran, and the receipt's steps appear in the record's trajectory. Nothing promotes the plan to a fact: an unobserved performance stays `None`. Method evidence is not limited to `method_performed` either — the code, model or trajectory associated with an execution can be inspected too, keeping its source and determinism; but a PLAN alone never counts as a performed method. `induction-material` is what you read before writing a strategy.

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## The material you read (no framework interpretation)

The framework organizes the recorded facts; it does not decide what they mean. `orx induction-material` is the entry point: per attempt it gives you the problem SHAPE (a count-based CIR/profile summary — the full representation stays on the record), the method (`planned`/`actual` with `basis`: `performed` / `planned_only` / `none`), the `changes` against the previous same-task attempt (code hash, planned method), the outcome (status / objective / gap / executed `code_hash`), the task check with its `reference_source`, the failures, the trajectory tail and the cost. Every attempt of a task is kept (each failure with its own cost) and grouped under `task_chains`, so a same-task retry is never read as cross-task support.

**Key questions to answer yourself:** what STRUCTURE recurred, what METHOD was applied, WHY it helped, and where its cost and failure boundary lie. The framework does not fit your answer into a fixed pattern catalogue, and it does not name "patterns" for you. The four angles below are yours to use, not detectors the framework runs.

**The method is the missing half of the evidence.** `execute --method '<json>'` (or a candidate's `method`) records the PLAN; the solve script's optional `method_performed` receipt records what ACTUALLY ran. Nothing promotes the plan to a fact: an unobserved performance stays `None`. `induction-material` is what you read before writing a strategy.

**Nothing is admitted for you.** A cell clearing a sample count, two runs sharing a `strategy_id` — none of these is a precondition for reading material or a bar on what you may abstract. They shape how strong a strategy its evidence can later support ("at least two tasks" is a PUBLICATION bar for TRANSFER, not a gate on material). Where a strategy asserts a benefit — faster, higher quality, cheaper — the matching COMPARISON evidence is required at VERIFICATION, not as a precondition for review.

**A strategy must match its evidence.** An assertion whose probe PATH does not resolve is reported `insufficient` — a typo is not a refutation. `code_unchanged` (every cited record shares one `solver.code_hash`) backs a "the code was not changed" strategy; an `optimal` status alone never does. A strategy whose text names a method its evidence does not carry is reported.

## Applicability: problem class + structural cell

Applicability is not declared on a ladder and no widening command exists, but it IS structurally quantized:

- **problem class** — the grouping anchor, DERIVED from what the model IS (integer-vs-continuous structure and index shape): `pure_lp`, `milp`, `network`, `scheduling`, `mixed`, or `unknown`. It is NOT the free-text `family` label. The label used to be the anchor, and because callers typed it freely ("inventory", "generic", "planning" for structurally identical tasks) it scattered comparable evidence across incomparable groups — the direct cause of a cold-start run yielding zero claims. Two tasks with different labels but the same class now share a cell, which is what makes them comparable.
- **cell** — each measurable coupling dimension falls in one of `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, and the claim's applicability is that cell (e.g. `rc[0.75,1.00]`). Unmeasured dimensions get their own `[unknown]` cell.

`family` is still recorded and a claim still NAMES the family its evidence came from (it appears in the predicate set), but it no longer SPLITS the statistics. A profile with no structural signal at all classifies as `unknown`, which is its own cell — never pooled with a measured class.

Why cell rather than observed min/max span: a min/max span is a range in which samples happened to be observed, not a demonstrated region. One region can score 1.0 and another 0.1; a single span across both would claim "0.55 everywhere", erasing the relation between structure and performance. Cells keep those regions separate — each gets its own claim.

**Unknown is never similarity.** `[unknown]` matches only a task whose value is also unmeasured: "neither side measured it" is a shared absence of evidence, not evidence that the structures agree.

**Only evaluable predicates can decide applicability.** `profile_matches` / `classify_applicability` evaluate exactly four keys: `family`, `resource_coupling`, `temporal_coupling`, `route_complexity`. A predicate carrying ANY OTHER key (`problem_class`, `task_family`, `structure`, a free-text `description`, …) is a SEMANTIC condition code cannot decide, and it is never defaulted to satisfied:

- `profile_matches` returns `False` for such a predicate set (an unverifiable condition is not a match);
- recall's `classify_applicability` returns `("unknown", reusable=False, reason)` naming the exact unsupported keys — distinct from `("conflicts", …)` where a KNOWN value contradicts the claim.

Read `unknown` as "the framework could not check this — TEST it", never as "it applies". A claim's structural predicates are what the code can verify; a semantic condition belongs in the claim TEXT for a human/agent reader, not in `conditions.predicates` pretending to be checkable.

Counterexamples need no special machinery: a miss inside the claimed cell is a miss *of the claim* (the claim covered that task when it ran), and three consecutive ones demote the entry to `suspect`. Records outside the cell or from another family do not match, so they are neither checks nor counterexamples.

## What it takes to become a claim (admission)

The framework does NOT decide whether evidence is "enough to create a claim": that is your judgment, made from the material. `induce` creates or refreshes a DRAFT entry from whatever attempt-scope evidence a structural cell holds, and the entry's `verification` (not a sample count) decides whether it is PUBLISHED.

**A single task is not transferable.** An entry citing fewer than 2 distinct `task_id`s is SAVED but its `publication` reports `published: false` until it is verified over ≥2 independent tasks. The evidence is never refused — recall still answers from it — but repetition of one task never becomes transferable knowledge.

**Only a cold-archive card blocks creation.** A standing card for the same (strategy, predicates) pattern is reported before anything else (lift it with `--force` when the environment has genuinely drifted). Dedup considers **dormant** entries, and once a claim exists any new matching evidence refreshes it, however repetitive.

**Publication splits by KIND, and `conditional_fact` is a FACT, not a shortcut.** A `conditional_fact` claim publishes with ONE verified observation (it makes no transfer claim); every OTHER kind is a TRANSFERABLE claim that still needs ≥2 DISTINCT tasks. There is deliberately no third "transferable-from-one-task" kind.

**Do not reach for `conditional_fact` to keep a single-task habit.** It is for a claim that genuinely has NO transfer intent — "under this structure, THIS task's answer was checked and holds". If what you learned feels reusable, it is by definition a transferable claim and needs ≥2 tasks: read more batches (`--cursor`) and look for the SAME mechanism recurring, rather than restating one task and labelling it a fact. A review where every claim is a one-task `conditional_fact` is a symptom the sample set was never widened — the framework reports `cross_task_hint` precisely so you can see the task span before writing.

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

- The framework organizes recorded FACTS and computes declared checks. It does not interpret what a fact means, does not name patterns, and does not derive a technique from numbers or a strategy name.
- The material reports what the evidence CARRIES (`method.basis`: `performed` / `planned_only` / `none`) and what is `missing`; you decide what to abstract. There is no `material_report` object and no admission verdict.
- A strategy must match its evidence: a probe PATH that does not resolve is `insufficient` (never a refutation); `code_unchanged` backs a "the code was not changed" strategy from the recorded `solver.code_hash`; and a text-vs-evidence mismatch is reported.

## Reading the material

Semantic induction is a division of labour: the framework organizes the material and checks what you submit; YOU read the material and write the new strategy.

**`orx induction-material [--strategy S] [--task T] [--limit N] [--cursor C]` is the ONLY material entry point.** It reads a BATCH of completed tasks straight from the evidence bank and returns, per attempt:

- `task_text` — an excerpt of the task semantics (or an explicit `unknown` marker when the version is not retained — never a fabricated summary);
- `problem.profile` / `problem.cir` — the identity measured BEFORE modeling, and the coupling structure if one was built (frozen, never rewritten after the solve);
- `method` — `planned` vs `actual` with a `basis` of `performed` > `planned_only` > `none`, plus `trajectory`;
- `outcome`, `task_check` (a SEPARATE fact from the solver's own status; absent reads as `never_checked`, never a pass), `failures`, `cost`;
- `attempts_of_task` / `independent_task` — how many attempts share this `task_id`, so a same-task RETRY is never read as cross-task support;
- `existing_knowledge` — the entries that COULD relate to this batch (by strategy id, by cited evidence, or by overlapping family), each with its stated claim, predicates and verification state, so you can see whether to add, revise, merge or leave alone;
- `task_chains` — the full chronological attempt chain per task;
- a `budget` block (chars used / limit / omitted ids / `next_cursor`) — eviction is REPORTED, never silent. Pass `budget.next_cursor` back as `--cursor` to read the next (older) page.

Missing fields are marked `unknown` individually; one unknown field never drops the rest of the fact. Scope filters and `--limit` narrow the batch, and `OR_HARNESS_INDUCTION_MATERIAL_CHARS` bounds its size — growth trends toward batching and PAGING, never toward material that is permanently invisible, and never toward a bigger one-shot default.

Submitting a strategy whose evidence reports no method (and which declares none) still saves the strategy — you may legitimately state the method in your own words — but the outcome carries a `material` warning saying the framework did not and will not derive a technique from the numbers, so the strategy's basis is visible as numbers-based.

Evidence is cited by explicit `execution_id`; there is no `bundle_id` citation to expand (there is no candidate generator).

## New strategies (`induce --relation`)

Not every lesson is one strategy's statistics. "When temporal coupling is high, a temporal decomposition must keep its cross-period state" is knowledge about a **structural condition paired with a choice and its consequence** — it may belong to no strategy id at all, and a mean plus a free-text remark cannot carry it.

A new strategy is a reusable modeling, decomposition, search, checking or repair technique — not necessarily the whole plan of one execution and not necessarily one solver. It carries: the structure and conditions it applies to; the concrete method; a grounded explanation and expected effect; its cost, risks and boundary; and its supporting evidence, counterexamples and anything still unverified.

**The unit of knowledge is the ENTRY, and one entry is ONE strategy.** A second independent strategy is a second entry; there is no host lookup, no embedded claim list, and no shared verdict. `induce --relation '<json>'` submits one strategy or strategy revision:

```json
{"subject": "principle:cross_period_state",
 "claim": "the sentence being asserted (yours to write)",
 "evidence": [{"execution_id": "ex_...", "role": "dropped"},
              {"execution_id": "ex_...", "role": "preserved"}],
 "conditions": {"predicates": {"family": "scheduling",
                               "temporal_coupling": [0.5, 1.0]},
                "note": "optional"},
 "check": {"assertions": [ ... ]},
 "kind": "rule"}
```

**Where the checks live.** The declared checks sit INSIDE the relation (the `"check"` block above). A standalone `--verify` payload (`{"claim": "...", "check": {"assertions": [...]}}`) is the SAME mechanism and is accepted for callers that keep the check separate; an embedded `check` is NEVER silently ignored. If you supply BOTH they are a CONFLICT: `--verify` wins and the outcome records `check_note.check_source: "verify_arg"` plus a note that the embedded block was overridden — never merged. Verify a strategy before trusting it: one whose checks were not read is not a verified strategy.

**Different tasks, strategy ids and cells may be cited TOGETHER.** There is no "same method name" or "same cell" requirement: whether a shared mechanism exists is YOUR judgment from the content. The framework reports the distinct-task count and lets the publication gate speak — it does not splice evidence to reach a count, and a single-task observation is not dressed up as a rule.

**Only evaluable predicates decide applicability.** `conditions.predicates` may carry keys the framework can EVALUATE (`family`, `resource_coupling`, `temporal_coupling`, `route_complexity`) and keys it cannot (`problem_class`, `task_family`, `structure`, any free-text key). A key the framework cannot evaluate is NEVER treated as satisfied: recall reports the hit as `unknown` (undecided, with the exact unsupported keys) rather than `applies`. Put what matters in the structural predicates the framework can check, and read `unknown` as "test this", not "this applies".

A submission is a knowledge WRITE like any other: it records a maintenance action with the pre state, a knowledge delta (`entries_created` / `entry_changes`), the index result, and the same knowledge feedback. Re-submitting the same strategy revises that entry, not a duplicate.

- **`evidence` + `role` is the anchor.** Every reference names a recorded execution and the part it plays in *this* strategy. Roles are free strings (`dropped`/`preserved`, `before`/`after`, `strategy_a`, `violation`/`satisfying`, …) — they are not a taxonomy, and nothing forces your strategy into one of the four observation angles. `kind` is an optional note about what prompted it.
- **Identity is derived, never submitted.** Tasks, family, structural cell and strategy ids are read off the recorded facts. You supply ids and roles only, so the same evidence cannot acquire two contradictory identities.
- **`subject` is optional.** With no subject, the evidence's single strategy names the entry; with a subject, that free-form name is the entry's `strategy_id` (e.g. `principle:cross_period_state`). An entry that states a strategy but has no statistical support (`support_n = 0`) is a **strategy-only entry**: it makes no quality/cost/failure claim, and its publication is decided by its single `verification` block alone.
- **`conditions`** are the applicability predicates; when omitted they are read off the evidence's own structural cell.
- **One entry, one identity.** The entry a submission revises is found by `strategy_id` + structural cell + `kind`. Two independent strategies under one subject (a different cell, or a different kind) are separate entries, and neither inherits the other's verification. A strategy and a statistical claim under the same strategy id are also separate: the strategy never attaches to the statistical entry.

### What the framework checks (the `check` block)

The framework computes **only what your structured declaration makes computable** — it does not parse the strategy sentence and does not promise to notice that a sentence overreaches its evidence. Each assertion carries its own semantics:

| Assertion | Checks | Applies to |
|---|---|---|
| `{"kind": "probe", "roles": [...], "path": "quality.feasible", "equals"/"min"/"max"/"in": ...}` | every record of those roles satisfies the probe (the framework reads the dotted path itself) | any strategy |
| `{"kind": "status", "roles": [...], "status": "optimal"}` | every record of those roles finished with that status | any strategy |
| `{"kind": "comparison", "metric": "quality"\|"cost:<dim>"\|"<dotted.path>", "roles_a": [...], "roles_b": [...], "direction": "higher"\|"lower", "min_gap": 0.1, "mode": "paired"\|"group", "aggregation": "all"\|"mean"}` | the declared difference between the two role groups | numerical strategies |
| `{"kind": "code_unchanged", "roles": [...]}` | every cited record shares one `solver.code_hash` (an `optimal` status never backs it) | "the code was not changed" |

**Pairing is a property of the assertion, not of the batch.** `mode: "paired"` compares only records that share a `task_id`, one per side; records with no counterpart are listed as `unpaired_execution_ids` in the scope and are **not** mixed into the statistic. `mode: "group"` compares the two sides' means over a metric every referenced record measured (a metric measured on only some records is `insufficient_evidence`, never a subset mean dressed up as a strategy). A strategy may declare one paired and one group assertion — each is evaluated over its own scope.

**`aggregation` decides how an unfavourable sample is treated:**
- `"all"` — every pair must meet the direction and gap. One comparable counterexample refutes the strategy. Use it when you are asserting "on every such task".
- `"mean"` — the batch mean must meet the direction and gap. A single negative pair does not refute it. Use it when you are asserting "on average across this batch".

**Only the declared parts are covered, and the block says so.** "Quality is higher **and** tokens are lower" needs a `quality` assertion *and* a `cost:llm_tokens` assertion; declare only the first and the verification scope records the cost part as unchecked. The verdict is never extended to parts you did not declare.

**What `verified` means — and what it does not.** Every declared assertion held over the referenced evidence: "no violation was found **within this scope**". It is NOT a proof that the natural-language strategy is correct, causal, or broadly general. The verification block carries both `checked` (the assertions, executions and tasks the verdict covered) and `not_covered` (that explicit limitation), plus `scope` naming the executions, tasks, roles, and which assertions were checked and unchecked. Verdicts: `verified` / `insufficient_evidence` (nothing computable declared, a metric unmeasured, no counterpart, evidence unusable — **not** a refutation) / `refuted` (an assertion ran on real evidence and failed). An `optimal` status or a passed task check alone proves NOTHING about the whole strategy.

### Publication: two conditions

1. the entry's verdict is `verified` and not stale;
2. its verification scope covers **≥2 distinct tasks**.

Condition 2 is the independence rule: a single-task repair is a verified **fact about that task**; transferring it to future tasks is a knowledge strategy and needs independent evidence. A single-task strategy is still **saved** (and verifiable as that fact) — it is simply not published, and `publication.reasons` says exactly why. The distinct-task count is computed from the evidence the strategy ACTUALLY cites — never padded by same-name or same-cell conditions.

### Revision

Re-submitting the same identity (`strategy_id` + cell + `kind`) **revises** that entry. To target a specific existing entry unambiguously, name its id with `"target_entry_id": "se_..."` — a substantive edit then does not have to re-derive the target from subject + cell + kind. An unknown id is REFUSED rather than silently revising a different entry (or creating a new one).

- A **substantive** change (strategy text, conditions, evidence set, method) with no fresh verification marks the previous verdict `stale_after_revision` — the old check no longer covers the new strategy.
- A **fresh** verification wins outright: it was computed over the incoming evidence, so it is neither kept nor marked stale.
- An **identical** re-submission keeps the verdict.

When two existing strategies are near-duplicates, MERGE them (revise one to cover both and retire the other) rather than creating a third near-identical entry. A strategy must still be checked against CURRENT real evidence when it is created or substantively revised: an old source id expiring (the evidence window) never blocks a later update, and never by itself backs a revised strategy.

### Reading strategies back

A published strategy reaches you through the **ordinary recommendation path** (`recall().recommendations[]`): its entry is an admitted entry, so it appears with its expected effect and, under `knowledge`, its verification state and stated strategy. There is no separate knowledge section — one strategy, one retrieval channel.

Unpublished strategies are the ones that would not survive that filter: `recall().held_claims[]` carries them (a strategy-only entry whose `strategy_id` is a free-form subject, or an entry not yet verified), each with its verification state, plus `newer_evidence_since_verification` — matching executions recorded after the verdict, a visibility annotation rather than a lifecycle state. `--include-unverified` fills this section with refuted/stale strategies too, clearly labelled.

> Legacy note: a store written before knowledge was unified may still carry per-entry `relations` lists. `orx migrate-relations` converts each relation into its OWN entry, with its verification copied verbatim; it is idempotent and a `--dry-run` writes nothing. `recall` reads migrated strategies through the ordinary path.

## Statistical evidence vs a submitted strategy (do not blur)

| | Statistical evidence (computed) | Submitted strategy (agent-formed) |
|---|---|---|
| Where it lives | the fact layer; read via `ConditionalStats` (recall, world model) | the entry's `claim` block + its `verification` block |
| Who writes it | NOBODY — the framework never writes a technique from it | the agent, via `induce --relation` |
| Checks | — | the `check` block's assertions over explicitly referenced evidence with roles |
| Gate | — | own verdict + ≥2 tasks in the strategy's verification scope |
| Publication | — | verified + not stale + ≥2 tasks |
| Independence | one evidence cell supports a prediction | one strategy owns one entry |

## Peer evidence as phrasing (`--peer-strategy` / `--peer-cell`)

REMOVED. Naming another strategy or cell used to write one contrast line under the entry's `risk_conditions`. That was a sentence the framework could not check, and the observable form of the same observation is a submitted strategy — submit it with `--relation`, citing the executions on both sides. Existing entries that already carry such text keep it; nothing writes new ones.

## Applicability notes

`induce --relation ... --note "TEXT"` (repeatable) attaches free text to the entry that call creates or revises. Notes are stored verbatim, shown by `inspect` and in a recall hit's `knowledge.applicability`, and sit outside scoring — they are your phrasing for your own future reading, not a validated fact.

## Cold archive (anti-resurrection)

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --relation --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every submission. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data. New SUPPORT or a counterexample is likewise a reason to re-review (with `--force` when a card stands): the card blocks the SAME failed generalization, not new evidence.

## Worked contrasts

Three short contrasts between what NOT to submit and what to submit, all from the same kind of batch `induction-material` returns.

**1. Execution summary → reusable technique.** A summary reads:

> "`milp_lot_sizing` solved `orarla_9` to optimal (`objective=14.0`, reference match)."

That is a fact about one run, not transferable knowledge: it names a task, not a condition→how→consequence. The reusable version states the MECHANISM, the CONDITIONS, the EXPECTED EFFECT and the BOUNDARY, and cites the executions on both sides:

> "On multi-product lot-sizing with a terminal stock target and backlog (this structural cell), formulating inventory AND backlog as integer per-period variables with a final-period stock/backlog equality keeps the model feasible where a continuous-flow relaxation reports `optimal` at a fractional answer; scope: terminal constraints present, integer lots required."

Submit it with `kind="rule"` (or `conditional_fact` ONLY if you truly mean a fact about the one task, not a rule), `evidence` naming the dropped and preserved runs, and a `check` whose assertions state the comparison.

**2. When to REVISE old knowledge.** An existing entry claims a method is faster in this cell. A new `induction-material` batch shows a later task where it was slower (`independent_task: true`). Do NOT just add another entry: re-submit the SAME identity (`subject` + cell + `kind`, or the entry id) with the widened or narrowed condition and a FRESH `check`. A substantive change marks the old verdict `stale_after_revision` until the new one lands; a fresh verdict replaces it outright. If a near-duplicate entry already exists, MERGE rather than create a third.

**3. When NOT to induce.** The batch carries no method content (a name and a mean), or the only "evidence" is two runs of ONE task (repetition — `independent_task: false`), or the strategy would rest on a semantic predicate the framework cannot evaluate. In all three the honest outcome is NO new entry: record how the work was actually done (or make the retry a distinct task if it truly is one), and wait for independent evidence. Deleting a task number or renaming a label is not abstraction — the CONTENT has to earn the strategy.

**4. A strategy must not overreach its evidence.** A text that says "the code was unchanged" needs a `code_unchanged` assertion over the cited records (an `optimal` status never backs it); a text that names one method needs evidence carrying THAT method. Submit the strategy you can back, and report the part you cannot instead of leaning on a disclaimer.

These examples prove the FLOW (read → form → verify → publish → recall); they make no claim about generalization or solving quality beyond the conditions the cited evidence covers.
