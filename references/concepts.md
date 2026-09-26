# Concepts: why OR-Harness is built this way

Read this page when you want the reasoning behind a mechanism rather than its syntax: why coupling is derived structurally, why memory is two layers, why cost is never folded at rest, and how the world model relates to the loop. For the calls themselves see [commands.md](commands.md), for formats see [modeling.md](modeling.md), and for the contract fields see [world_model_contract.md](world_model_contract.md).

For agents that need the "why" behind the mechanisms. Everyone else: SKILL.md is sufficient.

## The research question

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

The conceptual loop:

```
P_t → Problem Profiling → Strategy Selection → Strategy Execution
    → Outcome + CostVector → Execution Evidence Bank (E_t)
    → Induction/Consolidation → Strategic Knowledge Bank (M_strategic)
    → S_{t+1}
```

E_t is factual execution evidence (what actually happened); M_strategic is derived strategic knowledge (what to do next time). Both layers are necessary — neither alone is memory.

## Structural grouping and coupling derivation

Structural groups — the similarity keys for all memory — are one (family, structural cell, strategy) triple each: the observations that may be aggregated together. A cell is the measurable coupling dims quantized to the four intervals `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, with an unmeasured dimension in its own `[unknown]` cell. Structure conditions the statistics but not the ladder: there are no generalization levels, no `widen`/`tighten`, and no automatic re-scoping. Coupling values are derived by priority: the task's CIR (the `coupling` field — relations/indexes measured from the pre-model coupling understanding; the cleanest source) > the `model` representation (measured from declared constraints) > structured spec fields > harness-supplied values. `semantic_coupling` is never derived as a scalar. When a supplied value contradicts the winning structural derivation across a bin boundary, the profile carries a warning — and the derived value wins for grouping, because an append-only fact filed in the wrong group would pollute conditional statistics permanently.

**The signature is frozen before strategy selection.** `execute` uses the same profile as `recall` — derived from `coupling` (CIR) / `spec` / `annotations` only. The `model` field and a solve script are POST-strategy artifacts and never enter the key; no command accepts a solve script as a coupling source.

Post-strategy information (solver diagnostics, model diagnostics) is stored separately in `ExecutionRecord.execution_features` — never in `profile_snapshot`. This keeps task identity stable across executions while preserving execution-time observations for offline induction.

## Coupling-Aware Intermediate Representation (CIR)

The four scalar coupling dimensions are **derived summaries**, not the primary representation. The primary representation is the CIR — an explicit, inspectable set of entities, decisions, constraints, and relations that captures *how* the problem's components interact. The CIR is produced **before** the canonical model, from the natural-language task description, and provides modeling guidance that improves the correctness of the canonical representation (which is written only after the strategy is chosen).

The architectural shift:

```
Old:  Task → scalar coupling scores → strategy matching → model
New:  Task → profile (CIR: entities/relations → understanding + retrieval key)
                → strategy comparison on expected quality/cost/risk
                → canonical model (after the strategy is chosen) → solve.py
```

Key principles:

- **CIR before model**: the CIR does not require a `model` field. The model is an intermediate representation written AFTER the strategy is chosen; it is verified and its coupling is reported as a diagnostic, but it never moves the structural key.
- **The profile retrieves, evidence decides**: the profile (including the CIR-derived scalar signature) is only the retrieval key — it decides WHICH historical evidence is comparable to this problem. The strategy choice itself rests on the expected quality/cost/risk carried by that evidence (entries, conditional statistics, or world-model predictions), never on the profile alone.
- **Discovery and reuse are separate operations**: text similarity (embedding) decides which memories are SEEN; structural keys and applicability predicates decide whether a seen memory may be APPLIED. Two nearly identical tasks whose coupling lands in different cells are invisible to each other through the structural channel — that is the aggregation rule working as intended, not a defect — so the text channel exists to surface them. What must never happen is the reverse leak: a `different_cell` hit's observed numbers joining the target cell's statistics, or a text similarity being read as a quality/cost/risk estimate. The score formula, `for_profile`, `evidence_predicates` and `group_key` are therefore untouched by retrieval, and `similarity` is reported as its own field, never folded into a score.
- **Structure first, scalars second**: scalar coupling scores (rc, tc, rx) are derived from the structure; they are summaries, never substitutes for the structure itself.
- **Co-occurrence is structural evidence only**: constraint-variable co-occurrence produces generic `depends_on` edges. A semantic relation (`uses_resource`, `shares_resource`, `competes_for`) requires additional entity/constraint semantics.
- **Domain-general**: all `kind`/`type` fields in the CIR are free-form strings — the schema never hard-codes supply-chain-specific vocabulary.
- **Dual downstream**: CIR feeds both modeling guidance and strategy retrieval via the single `orx profile` entry (rc/tc/rx derive from the CIR structure with priority CIR > spec > supplied). Coupling-aware understanding works even when Strategic Memory is empty.

See [references/modeling.md](modeling.md) for the CIR schema, evidence levels, and validation rules.

## Two-layer memory: facts vs derived knowledge

| | Execution Evidence Bank | Strategic Knowledge Bank |
|---|---|---|
| Question answered | "What actually happened?" | "What should we do next time?" |
| Storage unit | ExecutionRecord (episodic fact) | StrategicEntry (commitment) |
| Strategy role | the strategy ACTUALLY used | the strategy RECOMMENDED for this structure |
| Quality / cost | actual (`quality`, `cost`; alias properties `actual_quality` / `actual_cost`) | expected (`expected_quality_hat`, `expected_cost_hat`, `failure_prob`) |
| Artifacts | solver output, diagnostics — artifacts are evidence | none (only future abstracted patterns) |
| Mutation | append-first, fact-preserving; only cost backfill | CRUD, lifecycle, disposal — beliefs may be revised |
| Validation | it *is* the truth — the factual grounding layer | recorded, never rewritten: applicability and intervals are read off the supporting evidence at induction time, and the frozen checks on the facts are what promote, demote, or wake an entry |
| Re-induction | delete it and the factual basis is gone | re-inducible from currently retained evidence (`induce --rebuild`) — exact reconstruction of past entries is NOT a requirement |

A **conditional statistic** ("S04 averaged 0.91 quality over 6 runs in this group") is a query result — a recount. A **Strategic entry** ("S04 will land in [0.75, 0.95] for routing problems with resource coupling ≥ 0.75") is a claim about the future: it carries a prediction interval, a calibration track, and feature predicates that can match across groups. An entry that only restates statistics is redundant and refused at creation.

An entry may also carry **structured relation claims** (`induce --relation`): a structural condition paired with a modeling/solving choice and its consequence, anchored to explicitly referenced executions with roles. They are not a third memory — they live on the StrategicEntry, and each carries its **own** verification, independent of the entry's statistical admission: a verified relation does not publish the entry's statistics, and a stale or refuted relation does not invalidate another relation. A relation may also exist with no statistical claim at all (its `subject` names the principle). See [induction.md](induction.md#structured-relation-claims-induce---relation).

Strict Evidence ↔ Knowledge linkage is an **induction-time** requirement **by target design**: evidence must support and validate a candidate before admission. This is a migration target — today entries are born `candidate` and forward prediction checks still run online; moving admission validation offline is the next Induction round's job. Once admitted (and also under the current regime), an entry's continued validity does NOT depend on preserving its original supporting evidence — compacting or deleting old evidence never invalidates an entry. New evidence accumulates normally and influences knowledge again at the **next induction cycle**.

When the task carries a CIR, the evidence record preserves a `cir_snapshot` (the coupling representation actually solved), so future induction can re-bin evidence by structural context (`structural_context`) beyond the four scalar coupling features — an extension slot reserved for the strategy-cost phase, not implemented yet. The snapshot is the PARSED CIR, never the caller's raw `coupling` payload: a payload of the wrong shape is rejected before the execution runs, so a record can never carry a structure that looks present while parsing to zero entities.

## Task texts: a retrieval attachment, not a third bank

The task-text versions captured on the write paths (`task_texts`, keyed by `(task_id, text_digest)`) are neither facts about outcomes nor derived beliefs. They are the **source documents** of text retrieval: a task's own text, keyed by a digest of the whole task payload (`task_digest`), referenced from each `ExecutionRecord` via `task_text_digest`. Consequences of that distinction:

- They enter no statistic, carry no lifecycle, and cannot be recalled as evidence — they are input to the discovery channel, never an answer.
- One `task_id` may have SEVERAL versions: solving the same problem with changed requirements produces different digests, so each execution stays bound to the text that was actually in force. Re-capturing the same version is idempotent (the primary key is `(task_id, text_digest)`).
- A legacy record whose text was never captured keeps `task_text_digest = None` and is simply not vector-indexed. It is labelled (counted under `vector_recall.unindexed`) and remains fully visible through the profile channel and `orx inspect --bank texts`, and the text is never reconstructed from a guess.
- The index built from them (`{home}/index/*.embedding.json`) stores `{id, doc_digest, vector}` only — never a snapshot of the record. Recall re-reads the current record by id, so validity is always current and a changed document is detected by digest mismatch (reported as `stale`, excluded) rather than being served as a current similarity.

## No candidate menu: memory IS the candidate set

There is **no built-in strategy directory**. Nothing in the codebase ships a list of method names, descriptions, applicability rules, action lists or fallback chains, and nothing loads one.

A strategy is a **candidate** when, and only when, the memory really holds something about it in this structural cell:

| Candidate because | Evidence kind |
|---|---|
| it was really executed and recorded here | `conditional_stats` (a recount) |
| a claim about it here passed admission | `strategic_entry` (a commitment) |

With nothing in either, `recall` returns an **empty list** plus `recommendations_basis.reason` saying so. It does not return a `no_memory` row at score `-inf`, a zero quality, or a zero risk — a fabricated row is indistinguishable downstream from a measured one, and the whole point of the memory layer is that a number means "this was observed".

**The outer agent proposes the candidates.** You name the methods you are considering (`recall --candidate X --candidate Y`, `plan-next --candidates`, `predict-cost --strategy X`, `execute --strategy X`). Supplying them to `recall` only *filters* the real memory: the ones with no memory come back under `candidates_without_evidence` rather than being given a score.

**No directory membership is required anywhere.** `predict-cost` and `execute` accept any method id. An unknown cost basis is reported as `source="unknown"` with `expected_cost=null` — never a default zero, and never a refusal. The checks that remain are about *legality and safety*, not identity: the strategy id must be non-empty, the solver must be named, `code_path` must live inside `workspace`, and the sandbox's timeout/rlimits/budget rules are unchanged.

**Content comes from evidence, never from a directory.** A method's recorded type, actions and fallback live on the `StrategicEntry` that the harness wrote (or migrated); induction fills in **none** of them, because inferring a method's actions from its id would be fabrication. An entry with no recorded actions reports `[]` — "the memory does not record this", which is a fact, not a gap to paper over. Two different methods that happen to share an id are **not merged**: their entries are reported separately, each with its own content, and their executions stay separate samples.

**Retrieval documents are composed from what already exists** — the entry's own applicability notes, its own relation claims, its own recorded actions, and a readable transcription of its predicates. A method's name and description are not evidence about what ran, so they are not part of the document either.

**Historical ids stay readable.** Records and entries written under an earlier naming scheme (S01, S04, …) are ordinary strings to this build: they recall, predict, induce and inspect exactly as before. Nothing is rewritten and nothing is back-filled.

## CostVector: five dimensions, never folded at rest

`llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`.

Retries and rework are costs. A "wrong model → repair → rerun" trajectory must be more expensive than getting it right the first time, even when solver runtime is similar — otherwise the memory cannot learn that one-shot strategies are worth preferring. Scalarization (`C_scalar = Σ wᵢ·norm(cᵢ)`) happens only inside the selector, with configurable weights; the stored record always keeps the raw five dimensions.

**Unknown ≠ zero.** Each record carries a measured-dimension mask (`cost_measured`) meaning "this value is a real observation". Only what the framework can actually observe enters it: `latency_s` (the attempt's whole wall-clock span, from `time.monotonic()` — sandbox check, spawn, interpreter start-up, imports, solve, verify) and `solver_runtime_s`. A constant is never masked in: `tool_calls`, `retries` and `llm_tokens` are **declarations**, not measurements, and stay UNKNOWN until you supply them.

`solver_runtime_s` carries a provenance: `reported` (the script's `result.json:runtime_seconds` — the inner solve) or `wall_proxy` (the whole subprocess wall clock, imports included). They are **different quantities under one name**, so conditional statistics keep them apart and refuse to average a mixture; a runtime that is not a usable number is rejected and downgraded to the proxy with a recorded note.

`tool_calls` counts **every tool invocation in the attempt's scope** — shell commands, file reads/writes, sandbox runs, solver calls — not just sandbox executions. The sandbox sees exactly one of them and records that as a provable lower bound (`execution_features.tool_calls_lower_bound`); declaring a total below it is refused. `retries` counts only *extra* attempts beyond the first, and the framework claims the zero only when it can PROVE there was nothing to retry (no earlier attempt of the same task, episode and strategy exists); otherwise the count stays unknown until you declare it via `--override retries=...`. `llm_tokens` is invisible to the sandbox (you own the LLM), so it is declared at record time via `--override` (replace by default — idempotent, never double-counted; `--override-mode increment` exists for additional measured amounts within the same attempt).

Unmeasured dimensions are excluded from means, cost comparisons, and prediction errors — a placeholder zero is never evidence of cheapness. Legacy payloads without a mask keep the value-level inference: non-zero values (including already-backfilled tokens) stay usable, unconfirmable zeros stay unknown.

**A cost claim is complete-or-silent.** When `induce` consolidates a cell into a Strategic Entry, a dimension enters the entry's cost claim only when EVERY supporting record measured it. A mean over a subset ("3 of 4 records reported tokens") is a partial observation dressed up as a full claim, and downstream it would feed strategy selection and world-model prediction as if it were the whole truth. The entry is still created and its complete dimensions are still published; the incomplete dimension reads as unknown, and the gap is reported (`cost_claim_withheld`). Close it with `orx amend-cost <execution_id> --override <dim>=<value>` — the fact is amended in place, nothing is re-run — and the next `induce` restores the claim.

Cost predictions are validated like quality predictions — but separately. The pre-execution expectation is frozen as a prediction snapshot (`orx predict-cost`: strategy, expected cost with its measured mask, source, scope, per-dimension support). Feedback compares the actual attempt against that snapshot only — same strategy, same scope, and only on dimensions measured on BOTH sides; A's execution never audits B's prediction, an attempt never audits a task-scope prediction, and an unmeasured predicted zero never fabricates an error. Feedback is written into the Execution Evidence (`cost_feedback`) for the next offline induction round — it NEVER demotes, retires, or re-scopes an entry, and never updates calibration online. No snapshot, no fabricated error.

## The disposal ladder (derived layer only)

Facts are permanently neutral — no disposal ever touches an ExecutionRecord's meaning. The derived layer carries the ladder:

| State / action | Trigger | Effect | Reversible? |
|---|---|---|---|
| suspect | 3 consecutive prediction misses (auto) | score ×0.5, warnings attached | yes |
| dormant | 10 tasks unconsulted (auto) | excluded from matching | yes (wakes on hit) |
| retired | your explicit confirmation | moved to cold archive | no (leaves hot store) |
| cold archive | default forever | vetoes re-induction of the same pattern | `--force` only |
| compaction (deferred) | — | lossy evidence compaction is PAUSED: statistics and induction ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics (90 successes + 10 failures would read as 100% failure rate). Re-enabled once the summary consumption contract exists (Cost/Induction rounds) | — |

## Evidence retention (inside the Evidence layer — no new Banks)

- **raw facts** — every execution stays a full, append-only `ExecutionRecord`; GC compaction never touches them today.
- **explicit retention mark** — a harness-supplied `retention_reason` (free text, e.g. `contrast`) reserves representative episodes for future compaction policies. No automatic marking is performed.

Lossy compaction and the summary consumption contract are left to the Cost/Induction rounds.

## Verification philosophy: forward, not backward

An induced entry is a prediction hypothesis. Two separate things must hold before it counts as published knowledge: its CLAIM must pass an admission check at induction time (`induce --verify` — a rule holding, a repair working, or a quality-preserving cost saving, judged by the framework from real executions), and its PREDICTIONS are then validated by *future* executions checking its interval — never by self-testing on the training data. This is why entries are born `candidate`, why intervals are floored by sample size (n=2 may not claim [0.95, 1.0]), and why promotion requires ≥5 predictions with ≥70% hit rate ON TOP of the passed admission check.

The only text a model writes into the knowledge layer is *phrasing*: you may attach applicability notes at induce time (`--note`). They are stored for the reader and sit outside scoring — a sentence cannot be verified, so it is not scored. (A configured world model may separately produce *predictions*, but those live in their own log and never enter an entry, a statistic, or a verdict; see [world_model_contract.md](world_model_contract.md).)

### Three different questions, three different layers

The framework keeps apart three things that are easy to conflate, and conflating them is how a wrong answer becomes training data:

| Question | Layer | Who answers |
|---|---|---|
| Did the solver solve the MODEL it was given? | the execution's own `quality` | the executor (`status`, finite objective, gap) |
| Is this a valid answer to the TASK? | `execution_features.task_check` | `orx check-task`, from the check bases YOU declare |
| Does the strategic CLAIM hold? | the entry's `verification` | `induce --verify`, offline |

A relaxed LP answered with fractional values passes the first (it is `optimal` with `gap=0`), fails the second (the units cannot be shipped), and says nothing about the third. **Solver optimality is a property of the model as written, not of the task**: a model with the wrong variable domain, the wrong objective or a missing constraint is optimally wrong. That is why a task-level verdict is a separate fact attached to the execution rather than an edit of its quality — the observation must survive so a correction can be judged against it.

**A confirmed-wrong answer stays in the evidence set.** Its cost is real and its failure is raw material, so it is never deleted, excluded, or rewritten. What changes is that it may no longer count as a success sample: every quality consumer reads `0.0` for it, while its cost still counts in the task total. Three states, never merged:

- `passed` — every DECLARED basis held. This covers the declared bases only: the model's fidelity to the task and any undeclared constraint remain UNCHECKED, and the report says so.
- `failed` — a declared basis ran and did not hold.
- `insufficient` — no basis declared, no solution vector recorded, a needed variable missing, or the execution produced no usable result. **Not a pass, not a failure.** An unchecked answer's validity is UNKNOWN, and the framework never demands a reference value to fill the gap.

The framework does not parse natural-language constraints: a constraint is checkable only through a probe or a recomputation you declare. A matching objective never proves the model correct — which is why the old "no gold ⇒ matched" shortcut is gone: with nothing declared, the honest verdict is `insufficient`.

**A task check passing is not a knowledge claim passing.** The admission gate (`induce --verify`) is untouched by this layer: an execution whose answer was validated still has to earn any entry's verification separately, and an entry's verification says nothing about a particular execution's answer.

## World-model outcome predictions: shadow hypotheses, never decisions

A *world-model prediction* is a structured hypothesis about what ONE candidate action would do from the current frozen state: expected status, feasibility, quality, failure risk, per-dimension cost, and successor state changes. It is produced by an explicitly configured provider (`--world-model URL::MODEL`, OpenAI-compatible; credentials from the environment, never persisted) and lives in its own log table — it is not a third knowledge bank, not an ExecutionRecord, and not a StrategicEntry.

The discipline that makes these predictions useful rather than corrosive:

- **Shadow mode.** A prediction never changes a recommendation, a score, or a route. You decide; the prediction is compared afterwards.
- **Frozen before the act.** The input snapshot is frozen at prediction time. Predicting after executing and calling it a forecast is not evidence — it is hindsight wearing a costume.
- **Both sides defined, or not compared.** Comparison covers only fields the prediction defined AND the execution measured (the same both-sides-measured rule as cost feedback). A predicted-but-unmeasured cost dimension is listed as not-compared, never scored as zero error.
- **No counterfactuals.** A candidate that never ran has no result. A prediction for strategy A is never scored against strategy B's execution — the mismatch is recorded and the comparison is skipped.
- **Two costs, never confused.** The PREDICTED cost of the action is a hypothesis inside the prediction record; the model call's OWN cost (tokens, latency) is real spend, recorded on the prediction and charged to the calling action when one is named.
- **Uncalibrated confidence.** The model's self-reported confidence is data about the model, not a probability you may bank on. Calibration is what the accumulated prediction-vs-outcome record is FOR — that is the entire point of the shadow loop.
- **Knowledge stays gated.** Prediction feedback records facts and errors online; it never promotes, revises, or publishes anything. Knowledge changes only through explicit offline induction with admission verification, exactly as before.

### H is a prediction subject, not only a condition

The state the model conditions on is H/P/X/B, and what it predicts covers X, B **and** the capability side — how the action changes accumulated experience and strategic knowledge. Predicting only X/B and then updating H after the fact would make the harness unable to reason about *which action is worth taking for what it teaches*, which is the capability this layer exists to provide.

**The unified contract for this is defined in [world_model_contract.md](world_model_contract.md)** — that page is the single authority for the two prediction modules (`StrategyOutcomePrediction` and `CapabilityEvolutionPrediction`), the status vocabulary, the attempt-vs-strategy-window scope rule, and the migration table. What follows here is the *why* behind the design.

`H = F(M, W_OR, Pi, R, T)` — five INTERACTING capability sources, not five score dimensions and not a sum:

| Source | What it covers |
|---|---|
| M | Execution Evidence / Strategic Knowledge **and how usable it actually is** |
| W_OR | consequence prediction and reliability judgement for OR solving |
| Pi | strategy generation, composition, comparison, selection, switching, stopping |
| R | knowledge retrieval, applicability matching, context adaptation (**not** the final decision) |
| T | tool/solver selection, composition, invocation, result handling |

Three consequences that are easy to get wrong:

- **H is not its evidence.** `harness_state` knowledge refs, experience counts and tool configuration are *evidence about* H. They map to M and T as `indirect_evidence` only; nothing in the old state observed W_OR, Pi or R, so those stay `no_evidence` rather than being filled from an unrelated count. There is no composite H score in this build, and the validator rejects a payload that tries to add one.
- **There is no `E_hist` capability term.** Historical experience is not a sixth component — it is the substrate the sources are evidenced from. The H-evolution predictor's own quality is assessed separately; it is not part of its own claim to have got stronger.
- **B is not a predicted world-state object.** Budget declarations, execution limits and the real ledger stay in `BudgetLedger`. Cost appears twice in the contracts and the two are never summed: the *predicted* cost of the candidate, and the *measured* spend of the prediction call itself.

Three facts about a prediction service that are routinely collapsed into one, and must not be: `provider_configured` (a provider object is attached), `service_available` (this build IMPLEMENTS that kind and a provider is configured) and `prediction_made` (a prediction really was produced and validated). Only the third makes a contract `valid`, and building a contract makes no model call at all — so a configured provider with zero calls and empty benefit/cost/risk is not a forecast. The field-level rules (status values, the attempt-vs-window scope rule, the legacy migration table) are in [world_model_contract.md](world_model_contract.md).

A knowledge change is predicted against a target that is either an existing entry (it must really exist — a model cannot invent knowledge) or a hypothesis (allowed, but it must state what would be observed and how that observation would be judged). Each item names a change, a horizon, and any standing preconditions, because the timescales genuinely differ: evidence lands when the execution is recorded, while a claim only forms, moves or narrows after an offline induction.

**Prediction, fact binding and verified effect are three different things.** Adding knowledge entries, accumulating evidence, or a model asserting an improvement confirms none of them; only an observed improvement in future task performance does. The contract records all three flags separately, and a capability prediction is judged through those observable consequences — never through a latent vector.

**Unknown upside is not rewarded.** The knowledge term is `δ·K` with `δ` defaulting to 0, and a K that cannot be justified contributes exactly zero. This is the deliberate mirror image of how unknown *risk* is treated: an unknown downside is charged in full, but an unknown upside is paid nothing — otherwise the system would prefer whichever action it understands least. `K = None` means "no justified value", never "worth nothing".

**The model cannot raise its own value.** K's magnitude comes from framework-side quantities (how thin the support is, how much room the claim's interval still has above its honest floor, how much *independent* cross-task reuse the cell has). The model contributes a direction and any preconditions. Its `uncertainty` and its self-scored `expected_knowledge_value` are recorded for later calibration but never consumed as value.

**Feedback is judged in stages, and pending is not failure.** A prediction's verdicts are partitioned by stage, so an already-compared X/B prediction can still receive its knowledge verdict later. A precondition that never arrived leaves the item `pending` — it is not counted as a miss. Only a class with enough resolved samples earns a measured reliability, and that measured reliability (never the model's own confidence) is what later predictions may draw on.

### The prediction input context: what a prediction is conditioned on

The state, candidates and predictions are expressed in one place; what a prediction actually uses, and how it gets there, is the FROZEN INPUT side. The mechanism is one frozen bundle, `PredictionContext` (see [references/prediction_context.md](prediction_context.md)), built once per decision by `orx context` / `build_prediction_context`.

**The problem's name is not evidence about its mathematics.** The joint representation carries math attributes (`integrality`, `linearity`, `objective_kind`, `constraint_kinds`) with an explicit ORIGIN each, taken from a declaration, the declared model, the structured spec, or the CIR. A task whose only content is the word "routing" gets `unknown` — not "MILP". The same discipline governs the CIR: its relations stay relations, because compressing them into three coupling numbers and discarding the structure loses exactly the information a prediction needs.

**A model is not a precondition for a prediction input.** A task with no `model` field, no CIR and no solve.py still builds a context; the absent parts are listed with what they mean. This is the same principle as the snapshot's "unknown ≠ zero", applied to input assembly.

**"Retrieved" is not "verified".** Every retrieved item carries an evidence CLASS — an execution fact was observed, verified knowledge was admitted, an unverified candidate was neither, a structural recommendation may be backed by nothing at all. Retrieval never upgrades one class into another, and a candidate does not become knowledge by being surfaced.

**Two channels, never one number.** The structural channel answers "what may I reuse?" (applicability) and the text channel answers "what should I look at?" (discovery). They are reported side by side with their own statuses; blending them into a single score would hide which question was answered. A near-identical problem in a different structural cell stays VISIBLE and labelled `different_cell`, and its numbers never join the target cell's statistics. One hit found by both channels is ONE piece of evidence, identified as `layer:id`, and when the two channels report different versions of one id that disagreement is recorded rather than silently resolved.

**Reuse must be provable.** A supplied recall result or context carries the task VERSION it was produced for; a mismatch is refused with a named reason, and a result with no recorded version cannot be confirmed either way and is also refused. An external result of unknown provenance must not masquerade as aligned frozen evidence — the same rule as "an unmeasured condition is not a satisfied one".

**Freezing is content, not a pointer.** A built context is stored WITH the content it was built from, so replaying it reads nothing from today's banks, and reuse replays its frozen X/B, its frozen knowledge targets and its frozen reliability. A stored id alone would not reproduce the input, because the bank it points at may have moved. The BUDGET is deliberately the exception — an external limit on whether a call may be made, not a prediction condition — and when it has moved the difference is REPORTED rather than silently substituted.

**One CIR per request.** An explicit CIR must drive the joint representation, the profile (hence the snapshot's cell) and the retrieval alike; otherwise a single request carries two structural judgments and may retrieve knowledge from the wrong cell. A supplied CIR replaces the task's own rather than being quietly overridden by it. The structural consistency check therefore runs against the EFFECTIVE input, with a conflict refused rather than carried — an unmeasured dimension is not a conflict.

**A historical input must not read today's memory.** Building from a frozen snapshot bounds the retrieval to that moment: evidence created afterwards is excluded and reported, and an item whose creation time cannot be established is kept but counted unbounded — neither silently dropped nor silently trusted. Creation-time filtering is NOT historical reconstruction, because it cannot tell that an entry which already existed was later REVISED; reconstruction therefore reads what the snapshot SAVED and reports everything it did not save (reliability, cell evidence, the retrieval) as missing. The snapshot itself is the saved history, so no separate historical database is needed.

**Degradation is per part, and "did not run" ≠ "found nothing".** No backend, no task text, a missing index and a failed backend call are four different facts with four different reasons, and all four differ from a healthy channel that ran and matched nothing. Collapsing them would make an unavailable channel look like an empty memory.

**Capability evidence is evidence, not a level.** Which retrieval channels ran, which strategies were recorded, what a prediction track record measured — all of these are `indirect_evidence`, and a source nothing observed stays `no_evidence` rather than being filled in to complete a set of five. The capability VERSION block (config / model / prompt / tools / memory content) is an identity whose digest covers the decision-relevant CONTENT of what was consulted, with read timestamps excluded — so a revision moves it, a re-read does not — and it says which memories were read, never how capable the harness is. Building an input is not predicting: context assembly may read the embedding index and does nothing else.

## Applicability: the structural cell, not a declared ladder

There is no ladder of generalization levels and no `widen`/`tighten` command. Each (family, structural cell, strategy) triple is one evidence set, and the claim induced from it states its own applicability: that family and that cell (e.g. `rc[0.75,1.00]`). Structurally different regions of one family are separate evidence sets — one strategy scoring 1.0 at low coupling and 0.1 at high coupling yields two claims, not one averaged "0.55 everywhere".

Unmeasured structure is never similarity: an `[unknown]` claim matches only a task whose value is also unmeasured. The cost of the cell rule is honest: evidence scattered across cells may be too thin to form a claim, in which case only the statistics remain and no knowledge is invented.

Cross-family transfer is therefore a judgment, not a mechanism: a claim speaks for the family it was induced from, and applying a routing lesson to packing is your call. Record those executions and packing earns its own claim. The frozen per-fact checks (same strategy, attempt scope) are what make the lifecycle revision evidence-based rather than self-referential.
