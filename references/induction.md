# Induction: scope, material, and validation

Read this page when you are deciding whether to induce, working out why a candidate was skipped, or submitting a knowledge claim.

## The offline flow

```text
completed episodes
  review-material                 read a BATCH straight from the evidence bank
        |                         (no model call, no detector, no count gate):
        |                         success / failure / cross-cell / cross-name
        |                         all visible, retries marked non-independent
        |                         -- the DEFAULT material entry point
  (optional lead)                 induction-candidates freeze cell packages;
        |                         induction-material reads one
        v   material_report.missing non-empty = record how the work was
        |   done; the framework does not abstract it for you
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

**Read first, gate later.** The material you review comes from ONE place: `orx review-material` reads a BATCH of completed tasks straight from the evidence bank — no detector candidate and no sample-count gate. Success, failure, cross-cell and cross-method-name material all reach you, and a repeated run of one task is marked `independent_task: false` so repetition is never mistaken for cross-task support. `induction-candidates` / `induction-material` are an OPTIONAL cell LEAD worth a second look — they never decide what you may read.

**Entry conditions gate PUBLICATION, not visibility.** Creating a statistical entry needs ≥2 executions from ≥2 distinct `task_id`s in the same structural cell; publishing one needs a passed admission check as well. A thin cell (fewer than 2 executions) or a single task's repeated runs still APPEAR — as `kind="cell_observation"` with an `admission_note` — because its material is reviewable even though a transferable claim is not yet admissible from it. There is NO separate "single observation" trigger category: a lone verified execution is simply a thin cell, and the same principle covers every case — induction may come from one or many executions, and the evidence COUNT constrains the claim's STRENGTH, never whether the material may be seen.

**What `record` tells you.** `record` does NOT interpret the fact it stored: it emits no induction labels and no hints. Deciding what a fact MEANS is your job — read the material with `orx review-material`; `induce` is your explicit call, and you may induct from your own business knowledge with no lead at all. (Old records may still carry an `execution_features.induction_hints` key left by the retired detectors; nothing reads it any more.)

**The method is the missing half of the evidence.** `execute --method '<json>'` (or a candidate's `method`) records the PLAN; the solve script's optional `method_performed` receipt (stamped with the attempt's `OR_ACTION_ID`) records what ACTUALLY ran, and the receipt's steps appear in the record's trajectory. Nothing promotes the plan to a fact: an unobserved performance stays `None`. Method evidence is not limited to `method_performed` either — the code, model or trajectory associated with an execution can be inspected too, keeping its source and determinism; but a PLAN alone never counts as a performed method. `review-material` (or, for a lead, `induction-material`) is what you read before writing a claim.

Induction is the part of OR-Harness most worth understanding correctly. It answers: "given the facts accumulated so far, which generalizations am I entitled to commit to?"

## The material you read (no framework interpretation)

The framework organizes the recorded facts; it does not decide what they mean. `orx review-material` is the entry point: per attempt it gives you the problem SHAPE (a count-based CIR/profile summary — the full representation stays on the record), the method (`planned`/`actual` with `basis`), the `changes` against the previous same-task attempt (code hash, planned method), the outcome (status / objective / gap / executed `code_hash`), the task check with its `reference_source`, the failures, the trajectory tail and the cost. Every attempt of a task is kept (each failure with its own cost) and grouped under `task_chains`, so a same-task retry is never read as cross-task support.

**Key questions to answer yourself:** what STRUCTURE recurred, what METHOD was applied, WHY it helped, and where its cost and failure boundary lie. The framework does not fit your answer into a fixed pattern catalogue, and it does not name "patterns" for you.

**The method is the missing half of the evidence.** `execute --method '<json>'` (or a candidate's `method`) records the PLAN; the solve script's optional `method_performed` receipt records what ACTUALLY ran. Nothing promotes the plan to a fact: an unobserved performance stays `None`. `review-material` is what you read before writing a claim.

**Leads point; they do not admit.** A cell clearing a sample count, two runs sharing a `strategy_id` — none of these is a precondition for reading material or a bar on what you may abstract. They shape how strong a claim its evidence can later support ("at least two tasks" is a PUBLICATION bar for TRANSFER, not a gate on material). Where a claim asserts a benefit — faster, higher quality, cheaper — the matching COMPARISON evidence is required at VERIFICATION, not as a precondition for review.

**A claim must match its evidence.** An assertion whose probe PATH does not resolve is reported `insufficient` — a typo is not a refutation. `code_unchanged` (every cited record shares one `solver.code_hash`) backs a "the code was not changed" claim; an `optimal` status never does. A claim whose text names a strategy its evidence does not carry is reported.

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

**A single task is not transferable.** An entry created from fewer than 2 distinct `task_id`s is a DRAFT: the outcome carries a `single_task_note`, and its `publication` reports `published: false` until it is verified over ≥2 independent tasks. The evidence is never refused — it is kept as a candidate and recall still answers from it — but repetition of one task never becomes transferable knowledge.

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
- The material report (`material_report`) is a report, not an admission verdict: it names what the evidence CARRIES (`performed` / `planned_only` / `none`) and what is `missing`; you decide what to abstract.
- A claim must match its evidence: a probe PATH that does not resolve is `insufficient` (never a refutation); `code_unchanged` backs a "the code was not changed" claim from the recorded `solver.code_hash`; and a text-vs-evidence mismatch is reported.

## Reading the material

Semantic induction is a division of labour: the framework organizes the material and checks what you submit; YOU read the material and write the claim.

**`orx review-material [--strategy S] [--task T] [--limit N]` is the default.** It reads a BATCH of completed tasks straight from the evidence bank and returns, per attempt:

- `task_text` — an excerpt of the task semantics (or an explicit `unknown` marker when the version is not retained — never a fabricated summary);
- `profile` / `cir` — the identity measured BEFORE modeling, and the coupling structure if one was built (frozen, never rewritten after the solve);
- `method` — `planned` vs `actual` with a `basis` of `performed` > `planned_only` > `none`, plus `trajectory`;
- `outcome`, `task_check` (a SEPARATE fact from the solver's own status; absent reads as `never_checked`, never a pass), `failures`, `cost`;
- `attempts_of_task` / `independent_task` — how many attempts share this `task_id`, so a same-task RETRY is never read as cross-task support;
- `existing_knowledge` — the entries this batch's strategies already have, so you can see whether to add, revise, merge or leave alone;
- a `budget` block (chars used / limit / omitted ids) — eviction is REPORTED, never silent.

Missing fields are marked `unknown` individually; one unknown field never drops the rest of the fact. Scope filters and `--limit` narrow the batch, and `OR_HARNESS_REVIEW_MATERIAL_CHARS` bounds its size — growth trends toward batching, never toward material that is permanently invisible.

**`orx induction-material [--bundle BUNDLE_ID] [--strategy S]` is the optional LEAD** for a structural-cell candidate. It returns, for each candidate, the recorded evidence a claim can rest on:

- `methods[]` — per execution, its `planned` method, the method it reports as actually `performed`, and the trajectory steps that really happened;
- `outcome` / `task_check` / `failures` per execution — what followed, and what was actually checked;
- `material_report` — a REPORT, not an admission verdict: `basis` names the strongest method content present (`performed` / `planned_only` / `none`), the counts are stated, and `missing` lists the absent content (`method_performed`, `task_check`). The framework does not decide whether "there is enough to abstract" — you read the material and decide. It never derives a technique from numbers: record how the work was actually done (the script's `method_performed` receipt is the observation; `execute --method` is only the plan).

**`unavailable` means the premise no longer holds.** A candidate whose evidence lost a required side (an exclusion, a later `check-task` that refuted it) is reported with the reason rather than presented. An execution cited by a lead can also leave the evidence window entirely; the candidate is reported `unavailable` (honest, not silently upgraded) rather than a claim on a stale premise.

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
 "kind": "rule"}
```

**Where the checks live — one of two equivalent spellings.** The declared checks may sit INSIDE the relation (the `"check"` block above, as shown) OR in a standalone `--verify` payload (`{"purpose": "relation", "claim": "...", "check": {"assertions": [...]}}`). Both are read; an embedded `check` is NEVER silently ignored. If you supply BOTH, `--verify` wins and the outcome records `check_note.check_source: "verify_arg"` plus a note that the embedded block was overridden — a conflict is reported, never swallowed. Verify a claim before trusting it: a claim whose checks were not read is not a verified claim.

**Only evaluable predicates decide applicability.** `conditions.predicates` may carry keys the framework can EVALUATE (`family`, `resource_coupling`, `temporal_coupling`, `route_complexity`) and keys it cannot (`problem_class`, `task_family`, `structure`, any free-text key). A key the framework cannot evaluate is NEVER treated as satisfied: recall reports the hit as `unknown` (undecided, with the exact unsupported keys) rather than `applies`. Put what matters in the structural predicates the framework can check, and read `unknown` as "test this", not "this applies".

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

Retired entries leave cards in the cold archive (~200 B: pattern hash, predicates, outcome, reason, evidence summary). Before creating any entry, induction checks the archive: the same (strategy, predicates) evidence cannot resurrect the same failed generalization. `induce --force` LIFTS the veto — it removes the card and then proceeds, so your environment-drift judgment is made once rather than repeated on every induction. Reserve it for genuine drift (new solver version, changed problem distribution), which is exactly the situation where yesterday's failure is today's stale data. New SUPPORT or a counterexample is likewise a reason to re-review (with `--force` when a card stands): the card blocks the SAME failed generalization, not new evidence.

## Worked contrasts

Three short contrasts between what NOT to submit and what to submit, all from the same kind of batch `review-material` returns.

**1. Execution summary → reusable technique.** A summary reads:

> "`milp_lot_sizing` solved `orarla_9` to optimal (`objective=14.0`, reference match)."

That is a fact about one run, not transferable knowledge: it names a task, not a condition→how→consequence. The reusable version states the MECHANISM, the CONDITIONS, the EXPECTED EFFECT and the BOUNDARY, and cites the executions on both sides:

> "On multi-product lot-sizing with a terminal stock target and backlog (this structural cell), formulating inventory AND backlog as integer per-period variables with a final-period stock/backlog equality keeps the model feasible where a continuous-flow relaxation reports `optimal` at a fractional answer; scope: terminal constraints present, integer lots required."

Submit it with `kind="rule"` (or `conditional_fact` ONLY if you truly mean a fact about the one task, not a rule), `evidence` naming the dropped and preserved runs, and a `check` whose assertions state the comparison.

**2. When to REVISE old knowledge.** An existing entry claims a method is faster in this cell. A new `review-material` batch shows a later task where it was slower (`independent_task: true`). Do NOT just add another entry: re-submit the SAME identity (`subject` + cell + `kind`) with the widened or narrowed condition and a FRESH `check`. A substantive change marks the old verdict `stale_after_revision` until the new one lands; a fresh verdict replaces it outright.

**3. When NOT to induce.** The batch carries no method content (a name and a mean), or the only "evidence" is two runs of ONE task (repetition — `independent_task: false`), or the claim would rest on a semantic predicate the framework cannot evaluate. In all three the honest outcome is NO new entry: record how the work was actually done (or make the retry a distinct task if it truly is one), and wait for independent evidence. Deleting a task number or renaming a label is not abstraction — the CONTENT has to earn the claim.

**4. A claim must not overreach its evidence.** A text that says "the code was unchanged" needs a `code_unchanged` assertion over the cited records (an `optimal` status never backs it); a text that names one strategy needs evidence carrying THAT strategy. Submit the claim you can back, and report the part you cannot (`unsupported_fields` / a narrower `scope_note`) instead of leaning on a disclaimer.

These examples prove the FLOW (read → form → verify → publish → recall); they make no claim about generalization or solving quality beyond the conditions the cited evidence covers.
