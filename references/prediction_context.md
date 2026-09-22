# Prediction input context (`wm-context/1`)

**The strategy-outcome prediction service consumes this input** (see [`references/strategy_outcome.md`](strategy_outcome.md)). The legacy `predict_outcome` protocol is unchanged and remains the compatibility path.

This document defines **what information a prediction actually uses**, and how that information reaches the model consistently, completely and traceably.

- Context version: `wm-context/1`
- Joint representation version: `joint/1`
- Python module: `or_harness.world_model.context`
- Entry point: `ORHarness.build_prediction_context` / `orx context`
- Tests: `tests/harness/test_prediction_context.py`
- Runnable example: [`references/examples/prediction_context.py`](examples/prediction_context.py)

This document is the authority for the INPUT side. The contract side (statuses, scopes, capability sources) stays in [`references/world_model_contract.md`](world_model_contract.md).

---

## 1. What you must provide, and what you get

| You want to… | Use | What you must supply | What you get |
|---|---|---|---|
| Build the frozen prediction input | `orx context --task t.json` / `ORHarness.build_prediction_context` | a task JSON (nothing else is required) | a `PredictionContext` |
| Read a stored input back | `orx context --context-id CTX` / `ORHarness.get_prediction_context` | the context id | the frozen context, **without re-running anything** |
| Predict with a shared input | `orx predict-outcome --context CTX` / `predict_outcome(..., context=ctx)` | a task + an action spec | an `OutcomePrediction` conditioned on that input |
| Predict without any context | `orx predict-outcome --no-context` / `predict_outcome(..., context=False)` | as before this phase | the pre-phase-2 request shape, exactly |

**The minimum task fields** for a useful context:

| Field | Needed for | Notes |
|---|---|---|
| `task_id` | identity and version | required by the framework generally |
| `family` | structural cell | a grouping label, not prose |
| `description` / `objective` / `requirements` / `business_rules` / `constraints` / `spec` | the text channel | at least one of these gives the retrieval document its content. **With none of them the semantic channel cannot run** — that is reported, not silently ignored |
| `coupling` (CIR) | relational structure | optional. Absent means the structural dimensions come from the profile alone — *not* evidence of weak coupling |
| `model` | math attributes from structure | optional. A task with no model is a normal input; the math attributes then come from the spec/CIR or stay `unknown` |
| `spec.objective_sense`, `spec.constraint_kinds`, `spec.n_vars`, `spec.n_int_vars` | math attributes | optional; a source for `integrality` / `objective_kind` |

Everything is optional beyond `task_id`. **You do not need a mathematical model, and you do not need a solve.py.** There is no separate "understand" step: `orx context` profiles the task itself.

---

## 2. What the context carries

| Block | Content |
|---|---|
| `joint` | the joint problem representation (see §3) |
| `solving_context` | **X** — the episode's accumulated progress + the budget state, frozen from one snapshot |
| `retrieval` | the evidence from the two existing channels, deduplicated (see §4) |
| `capability` | `HarnessCapabilityEvidence` — evidence ABOUT H with an explicit status each (see §5) |
| `capability_version` | the version IDENTITY of config / model / prompt / tools / memory content (see §5) |
| `execution_constraints` | declared budget + consumption status, available solver families, executor limits |
| `knowledge_targets` | the framework's structural proposal set, **frozen** (see §6) |
| `reliability` | the measured reliability of past predictions, **frozen** |
| `snapshot` | the frozen condition blocks (X/B + coverage + harness condition) of the snapshot this context was built from |
| `cell_evidence` | per strategy, the distinct-task count behind the cell's statistics, **frozen** |
| `sources` | where each part came from (snapshot id, task version, channel list, CIR origin) |
| `degraded` | parts that did NOT run, each with its reason |
| `missing` | parts that are absent, each with what it means |
| `notes` | what the numbers above are, and are not |

Identity: `context_id`, `task_id`, `task_digest` (the task VERSION), `episode_id`, `snapshot_id`, `created_at`, `context_version`.

---

## 3. The joint problem representation (available BEFORE modelling)

P is described by three kinds of information, all organised together:

| Dimension | Where it comes from | Rules |
|---|---|---|
| **Math attributes** (`integrality`, `linearity`, `objective_kind`, `constraint_kinds`) | an explicit `--math` declaration > the declared `model` > the structured `spec` > the CIR | every attribute carries an **origin** (`declared` / `model` / `spec` / `cir` / `unknown`). **Nothing is derived from `family`**: the problem's name is not evidence about its mathematics |
| **Scenario semantics** (task text, objective, business rules, constraints, entities) | the task JSON's own textual fields | the text is kept as TEXT, not reduced to a hash. An absent text is reported as absent |
| **Structural features** | the CIR (entities, decisions, constraints, relations, coupling groups) + the profile + the derivation report | the CIR's **relations stay relations** — they are never compressed into `rc`/`tc`/`rx` and then discarded |

`joint.unknowns` lists the attributes nothing established. `joint.missing` lists what is absent and what that means. `joint.sources` maps each part to where it came from.

`linearity` deserves a caveat that is written into `joint.notes`: it is a **syntactic candidate** read off the declared model's expressions (a product of two declared decision variables). It is not a proof of convexity.

---

## 4. Retrieval evidence

The two channels that already existed are read as they are — **no new world-model-specific retrieval system was added**.

| Channel | Decides | Its signal is |
|---|---|---|
| `structural` | which strategies are APPLICABLE to this profile, with evidence-based scores | an applicability verdict |
| `semantic` | which memories are textually CLOSE, regardless of cell | a discovery signal **only** |

Rules enforced in the context:

- **Content, not just ids.** A hit carries the task excerpt, the strategy, the observed quality/cost, the failure count, the applicability label and the reuse verdict — a surfaced memory without its content is a number without a subject.
- **Cross-cell hits are visible and labelled.** A near-identical problem in a different structural cell is still surfaced (`structural_match: "different_cell"`), and its numbers **never enter the target cell's statistics**. The two channels are never blended into one score.
- **Deduplication by identity.** One memory hit by both channels is ONE piece of evidence (`identity = layer:id`) carrying both channel names and a `duplicate_count`. Support is never inflated by counting channels. Two channels reporting *different versions* of one id is a `version_conflict`, reported rather than silently resolved.
- **Distinct-task counting.** `deduplication.distinct_tasks_in_execution_hits` counts tasks, because repeat runs of one `task_id` are repetition, not reproduction.
- **"Retrieved" ≠ "verified".** Every hit carries an `evidence_class`: `execution_fact`, `verified_knowledge`, `unverified_knowledge`, `legacy_knowledge` or `structural_recommendation`. Retrieval never upgrades one class into another, and unverified candidates are **not surfaced by default** (`--include-unverified` is the inspection view; even then they stay labelled).
- **Bounded.** Hits are capped (top-k per channel, `MAX_HITS_PER_CHANNEL`). Truncation is recorded in `notes`, never silent.

### Degradation is reported per part

These are **different facts** and the context keeps them apart:

| Situation | Where it shows up |
|---|---|
| No embedding backend configured | `degraded[].part = "retrieval.semantic"`, reason `no embedding backend configured` |
| The task JSON carries no text | same part, reason `no task text in task JSON` |
| The index is missing | same part, reason `index missing; run \`orx rebuild-index\`` |
| A single layer's index is unusable | `degraded[].part = "retrieval.semantic.<layer>"` |
| The backend call failed | same part, reason naming the exception |
| The channel ran and found nothing | **no `degraded` entry**; `channels_run` contains `semantic` and `hits` may still be empty |

An empty evidence set with a degraded channel is **not** "nothing comparable exists" — the context says so in `retrieval.notes`.

---

### One CIR for the whole request, and ONE effective-input identity

An explicit `--cir` (or `cir=`) is resolved ONCE and drives **all** of: the joint representation, the profile (hence the snapshot's structural cell and its coverage view) and the retrieval channels. They already read `task["coupling"]`, so the effective CIR is routed to them through the task (`task_with_effective_cir`), and a supplied CIR **replaces** the task's own rather than being silently overridden by it. `joint.sources["cir"]` records which one won (`caller_supplied` / `task_coupling` / `none`).

Without this, one request could carry two structural judgments — e.g. the joint representation seeing a supplied CIR while the snapshot and the retrieval fell back to the task's own weak coupling — and could retrieve knowledge from the wrong structural cell.

**The effective-input version.** A context carries TWO digests: `task_digest` (the caller-supplied task JSON's version, kept for provenance) and `effective_input_digest` (the version of the task **with the resolved CIR merged in** — `effective_input_version(task, cir)`). Reuse checks compare against the EFFECTIVE one, because it follows the problem the prediction actually conditions on: a context built with an explicit CIR is reusable against the same task **when you pass the same CIR again** (`predict_outcome(..., context=ctx, cir=...)` / `predict_strategy_outcome(..., context=ctx, cir=...)`), while the plain task digests differ only because the caller-supplied JSON differs. Two different CIRs that happen to produce the same rc/tc/rx still produce different effective-input digests — the RELATIONS are part of the digested content, so three coupling numbers agreeing is not relations agreeing.

### Retrieval is bounded to the frozen moment

A context built from a **historical snapshot** describes the state as it was. A memory created *after* that moment belongs to a later state, so the retrieval is bounded to the snapshot's own time: postdating items are dropped and reported (`execution_constraints.retrieval_bounding`, `retrieval.structural.retrieval_bounding`, plus a `retrieval.bounding` entry in `degraded`). A memory whose creation time cannot be established is KEPT but counted under `unbounded_kept` — dropping it would discard real evidence, and keeping it silently would claim a bound that was never verified.

### Historical reconstruction vs current gathering

The two are kept apart by an explicit `historical` flag, NOT by the mere presence of a snapshot — a caller that freezes a snapshot for THIS decision (`plan_next`, `predict_outcome`) passes `historical=False` and the banks are read NOW and frozen; an EXTERNAL snapshot (the default when one is supplied without the flag) fixes a historical moment and everything comes from what it SAVED. Creation-time filtering is **not** historical reconstruction: it cannot tell that an entry which already existed was later *revised*, so the old entry would still be read at its NEW value.

| | Current gathering (`historical=False`, the default with no snapshot) | Historical reconstruction (an external snapshot) |
|---|---|---|
| knowledge | `verified_knowledge_view(profile, sbank)` — today's bank | the snapshot's **frozen** `coverage.knowledge_layers` (`frozen_knowledge_view`) |
| reliability | today's `prediction_reliability_table()` | **not saved** with a snapshot → empty, reported missing |
| cell evidence | today's statistics | **not saved** → empty, reported missing |
| knowledge targets | proposed from the snapshot + today's cell records | **not saved** → none proposed, reported missing |
| retrieval | the live channels, bounded to the snapshot's time | only what was saved with the snapshot — nothing is rebuilt from today's index |
| supplied recall result | accepted, bounded to the snapshot's moment | **REFUSED** — a result gathered now cannot be proven to belong to the frozen moment |
| recorded choices | read from the action log | not read |
| execution constraints | today's declared budget, consumption, tools | the snapshot's frozen `budget_state`; today's budget/tools reported missing |

A gap is **reported as missing** (`ctx.missing`), never filled from today's banks: reading today's bank to "complete" an earlier state would put after-the-fact information into a historical prediction input. Every historical build carries a note saying so, and a snapshot that carries no frozen knowledge view is reported as `MISSING` rather than silently consulted.

There is deliberately **no historical database**: the snapshot itself is the saved history, and what it did not save is reported as absent.

### Structure consistency is checked AFTER the effective input is resolved

The snapshot identity check runs *after* the explicit CIR is merged, so a snapshot taken under one structure may not be combined with a different effective input. Supplying the original task's snapshot together with a new CIR is **refused**:

```
the supplied snapshot cannot be combined with this problem input: the
supplied snapshot describes a different structure: resource_coupling=0.3 but
the effective problem input has resource_coupling=1.0. Build a new context
from the current input instead of reusing a snapshot taken under a different
structure.
```

The same check guards reuse at the prediction entry point: a context built under another structure is refused with `does not describe this prediction`. A dimension that is unknown on either side is **not** a mismatch — "cannot be compared" is different from "disagrees", and refusing on an unmeasured value would reject legitimate inputs (`structure_problems`).

### Reuse sends the FROZEN conditions, never re-derived ones

`predict_outcome(..., context=ctx)` conditions the request entirely on the frozen content:

| Condition | Where a reused context takes it from |
|---|---|
| `state` (X/B, coverage, harness condition) | `snapshot_from_context(ctx)` — the context's own frozen blocks and snapshot id |
| `candidate_knowledge_targets` | `ctx.knowledge_targets` (proposed and frozen at build time) |
| `prediction_reliability` | `ctx.reliability` |
| the recorded `input_snapshot_id` | `ctx.snapshot_id` |

Consequences worth knowing:

- progress established *after* the context was built never appears in the request, and the prediction's snapshot id equals the context's;
- a strategy whose targets were not frozen gets **none** sent, with the omission stated in `missing` — the proposal is never re-derived from today's bank;
- `model_info.conditions_source` says `frozen_context` when a reused context supplied the conditions.

The **budget** is the one deliberate exception: it is an external limit on whether a call may be made, not a prediction condition, so the pre-call check uses the CURRENT ledger. A difference is REPORTED (`model_info.budget_checked_at_call_time`) and the frozen constraint is left untouched — a move is never silently substituted.

### The memory version digests content, not counts

`capability_version.knowledge_content_digest` covers the decision-relevant **content** of the memory actually consulted: the layered knowledge entries (including the `applicability` notes and strategy `actions` the `KnowledgeRef` subset omits), the retrieved hits and the executions. Editing an entry's expected quality, its interval, its predicates or its actions therefore **moves** the digest, while a re-read does not — read timestamps (`snapshot_at`, `created_at`, `last_consulted_at`, …) are excluded and listed under `excluded_keys`.

`capability_version.knowledge_content` carries the digest and its composition (`knowledge_by_layer`, counts) — never the digested content, so the version cannot become a second unbounded copy of the knowledge view. It is bounded by construction (the scoped knowledge view + the carried hits), so no global snapshot mechanism is introduced.

---

## 5. Harness capability evidence (honest evidence strength)

The phase-1 contract is reused unchanged. This phase only adds observations that genuinely exist:

| Source | What this phase can add | Status | What it does NOT mean |
|---|---|---|---|
| `m` | knowledge references, experience volume, cells with evidence | `indirect_evidence` | more entries is not more capability; being referenced is not being used |
| `w_or` | the measured reliability of PAST predictions, when resolved samples exist | `indirect_evidence` | a track record is not a claim of present accuracy; too few samples reports no value at all |
| `pi` | recorded strategy selections/executions for this task | `indirect_evidence` | a recorded choice is not evidence that the choice was optimal |
| `r` | which channels ran, with their applicability labels | `indirect_evidence` | a hit count or a similarity is not evidence that a transfer succeeded |
| `t` | available solver families, executor limits | `indirect_evidence` | having a tool installed is not the ability to orchestrate it |

`capability.score_scheme` is fixed at `no_composite_score` and no composite number exists. A source nothing observed stays `no_evidence` — it is never filled in to make the five look complete. Nothing in this phase sets `direct_evidence`.

`capability_version` is an **identity, not a level**: it separates the harness configuration digest, the provider description, the prompt template version, the tool list and the memory-content digest. A content digest says *which* memories were read; it is not an ability measurement, and a fresh read timestamp is not a capability change.

---

## 6. One build, consistent reuse

This is the phase's most important engineering boundary.

1. **One call, one freeze.** `build_prediction_context` freezes ONE snapshot and runs ONE recall, then freezes everything into the context.
2. **No hidden side effects.** Building performs **no prediction-model call, no solver execution, no induction, no capability re-measurement**. It may call the embedding backend through the existing retrieval path — that is a read, and it is the only external call.
3. **Shared across candidates.** `plan_next` builds ONE context from the root snapshot and passes it to every root candidate (and to the horizon-2 continuation). Only each candidate's own `config`/`scope` differs. Building per candidate would let a bank that moved mid-decision contaminate the comparison, and would re-embed the same query once per candidate.
4. **Frozen against later changes.** Editing the task object, the banks, the knowledge entries or the tool configuration afterwards never changes a built context. `get_prediction_context` reads the stored content — it re-runs nothing.
5. **Reuse is version-verified.** A supplied recall result or context that disagrees with the current `task_digest`, `task_id` or `episode_id` is **refused** with a named reason (`retrieval_reuse_problems`, `context_identity_problems`). A result with *no* recorded version cannot be confirmed either way and is also refused — an unaligned external result must not masquerade as frozen evidence. A *known* disagreement is always a refusal; an *unestablishable* version is reported, not guessed.
6. **Replay, don't reconstruct.** A historical context replays from its own stored content. Reading today's bank to "fill in" what a stored context lacks is not done: absent parts are reported absent, and a genuinely new input gets a new context (a new `context_id`).
7. **The prediction records which input it used.** The provider request carries `prediction_context`; the stored prediction carries `model_info.prediction_context_id` / `_version` / `_task_digest`, so a later reader can resolve the exact content.

### What calls what

| Entry point | Embedding call | Prediction model call | Solver |
|---|---|---|---|
| `build_prediction_context` | **yes** (existing retrieval path) | no | no |
| `orx context --task` | yes | no | no |
| `get_prediction_context` / `orx context --context-id` | no | no | no |
| `predict_outcome` (context defaults to building one) | yes, once | **yes**, once | no |
| `predict_outcome(..., context=ctx)` | no | **yes**, once | no |
| `predict_outcome(..., context=False)` | no | yes, once | no |
| `plan_next` | yes, once per decision | yes, per candidate | no |
| `recall` | yes | no | no |

---

## 7. Compatibility

- The legacy `predict_outcome` request is unchanged when no context is passed (`context=False`): the request still carries `action_spec`, `state` and the pre-existing `prediction_reliability` block. The `x-b-only` ablation's byte-compatibility rests on this.
- Stored predictions written before this phase remain readable: they simply have no `prediction_context_id`.
- `recall` gained a `task_digest` field. It is additive.
- The new `prediction_contexts` table is created idempotently alongside the others; no existing table is rewritten and no schema version token moved.
- The phase-1 fixes are untouched: a configured provider with no predicted content is still `contract_only`; a legacy `ActionSpec` still carries its execution `params` verbatim and still refuses an unmappable `measurement_scope`; an unfinished window is still not `comparable`.

---

## 8. What is NOT here

- **The legacy `predict_outcome` protocol is unchanged.** The provider still receives and returns the existing payload shape when that path is used; the new context travels as an additional request key. The strategy-outcome protocol is a SEPARATE path — see [`references/strategy_outcome.md`](strategy_outcome.md).
- **Capability evolution is a separate service.** Its contract and service live in [`references/world_model_contract.md`](world_model_contract.md); the input context described here is shared with it.
- **No task-closing scheduler, no offline learning schedule.**
- **No capability evaluation system, no H score, no latent vectors.**
- **No training** of a model, a prediction head or a retriever.
- **No new semantic extractor, no retrieval rework**: the two existing channels are wired in, not replaced, and no composite retrieval score is introduced.
- **No change** to the structural bins, the unknown-matching rule, the selector's scoring, the induction-pattern detectors or the induction publication gate.
- **No new autonomous agent and no background loop.**

---

## 9. Verification checklist for a consuming agent

- [ ] Do I need a mathematical model to build a context? **No.** Check `joint.has_model` and read `joint.missing` to see what was absent.
- [ ] Is `retrieval.semantic.status` `degraded`? Then the text channel did not run — read the reason. An empty hit list is **not** evidence that no similar memory exists.
- [ ] Am I about to treat a `similarity` as a quality/cost estimate? It is a discovery signal. Read `evidence_class` and the applicability label.
- [ ] Is a hit `unverified_knowledge`? Then it is not knowledge yet; it never becomes publishable by being retrieved.
- [ ] Am I reusing a recall result or a context? Check `task_digest` matches. A mismatch is refused for a reason.
- [ ] Am I reusing a context and expecting the CURRENT state? A reused context replays its frozen X/B, its frozen knowledge targets and its frozen reliability. The budget is the only condition re-checked live, and a difference is reported rather than substituted.
- [ ] Did I pass `--cir`? Then check `joint.sources["cir"]` is `caller_supplied` and that the snapshot/retrieval agreed with it — one request must not carry two structural judgments.
- [ ] Built from a historical snapshot? Read `execution_constraints.retrieval_bounding` to see what was excluded as postdating it, and `unbounded_kept` for items whose creation time could not be established.
- [ ] Does `capability_version.knowledge_content_digest` look unchanged after a knowledge revision? It should have moved. It digests content, and read timestamps are excluded on purpose.
- [ ] Does a capability source read `direct_evidence`? In this phase it should not. `no_evidence` is the honest state when nothing observed a source.
- [ ] Am I reading `capability_version.knowledge_content_digest` as a capability measure? It is a content identity.
- [ ] Am I expecting several candidates to have different contexts? They share one — that is deliberate.
- [ ] Am I expecting the new context to change what the model PREDICTS? Not in this phase: it changes what the model is GIVEN.
