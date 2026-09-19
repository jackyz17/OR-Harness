# World-model contract

**Status: the CONTRACT is implemented. The prediction SERVICE is not
attached.** Read that sentence twice before using anything on this page.

This document is the single authority for the unified world-model contract.
Everything else (SKILL.md, `references/commands.md`, README) links here
rather than restating it.

- Contract version: `wm-contract/1`
- Legacy version label: `legacy/unversioned` (read, never upgraded)
- Python module: `or_harness.world_model.contracts`
- CLI entry: `orx contract`
- Tests: `tests/harness/test_world_model_contract.py`
- Runnable example: [`references/examples/contract_roundtrip.py`](examples/contract_roundtrip.py)

---

## 1. When to call this, and what you must provide

| You want to… | Use | What you must supply | What you get |
|---|---|---|---|
| Ask what a candidate strategy would produce | `orx contract --kind strategy_outcome` / `ORHarness.build_strategy_outcome_contract` | a task JSON + a candidate (strategy id, optional solver/config/scope) | a `StrategyOutcomePrediction` |
| Ask what a learning operation would change | `orx contract --kind capability_evolution` / `ORHarness.build_capability_evolution_contract` | a learning operation (`induce` / `revise` / `reverify` / `retire`) + a horizon | a `CapabilityEvolutionPrediction` |
| Read a stored prediction | `orx contract --payload <json>` / `ORHarness.read_prediction_payload` | the payload | a version-identified view (current / legacy / unsupported) |
| See what is known about harness capability | `ORHarness.capability_evidence` | nothing (task optional) | `HarnessCapabilityEvidence` — evidence, **no score** |
| See the real scope a strategy occupied | `ORHarness.strategy_execution_window` | task id (+ episode, strategy) | `StrategyExecutionWindow` with a `comparable` flag |

**You do not need a complete mathematical model.** A task without a `model`
field is a normal state — the contract is built from the task text, the CIR,
and the profile. You also do not need to fill everything in: every field is
optional, and an absent field stays absent (never a placeholder zero).

**Nothing here calls a model.** Building a contract performs no network
call. Three DIFFERENT facts are kept apart, because conflating them is how
an empty object came to be read as a forecast:

| Fact | Meaning |
|---|---|
| `provider_configured` | a provider object is attached to this instance |
| `service_available` | this build **implements** the service for this kind **and** a provider is configured |
| `prediction_made` | a prediction really was produced and passed validation |

Only the third yields `status="valid"`. Merely constructing a contract
returns `contract_only` **even when a provider is configured** — a
configured provider with zero model calls is not a prediction, and an
empty benefit/cost/risk is not a forecast.

`capability_evolution` has a **contract and no service**: this build does
not implement capability-evolution prediction at all, so configuring a
provider does NOT make that kind available. Check
`ORHarness.prediction_service_status(kind)` (or the `service_implemented`
field) rather than assuming "a provider exists" means "this works".

---

## 2. The two prediction modules

Both live over one shared frame, and both are *predictions*, not facts.

### 2.1 `StrategyOutcomePrediction` — OR strategy consequence prediction

Answers: *under the current problem (P), solving context (X) and harness
capability condition, what would this candidate strategy produce?*

| Field | Meaning |
|---|---|
| `candidate` | `CandidateRef`: strategy id, solver, config, preconditions, expected scope, stop conditions, and `scope` |
| `benefit` | `BenefitEstimate` — the gain, **with its metric, unit and baseline** |
| `cost` | `ExpectedCost` — the `CostVector` consequence of the candidate, with its measured mask |
| `risk` | `RiskStatement` — named loss events, **separate from cost** |
| `uncertainty` | `UncertaintyStatement` — execution randomness vs knowledge gap |
| `trace` | `PredictionTrace` — version, input, evidence, unsupported fields, call cost |

**Benefit is not one arbitrary 0–1 score.** `benefit.kind` says which
currency you are in:

| `kind` | Meaning | Typical metric |
|---|---|---|
| `effective_completion` | the task actually got completed | completion (0/1) |
| `solution_quality` | a qualified solution's quality | normalized objective gap, `1 - mip_gap` |
| `valid_progress` | a real intermediate step forward | progress ratio |
| `correct_infeasibility_diagnosis` | a *correct* infeasibility verdict | diagnosis (0/1) |

`benefit.value` requires a `baseline` — a gain with nothing to measure it
against is not a prediction. `benefit.value = None` means **unknown**, and
that is a first-class state: it is not zero, and it is not "no gain".

### 2.2 `CapabilityEvolutionPrediction` — harness capability evolution

Answers: *from the current capability evidence, the real experience and a
candidate learning operation `u`, how would future task performance
change — at what learning cost, with what degradation risk, and under what
verification conditions?*

| Field | Meaning |
|---|---|
| `current_evidence` | `HarnessCapabilityEvidence` — M / W_OR / Pi / R / T with an evidence status each |
| `candidate_operation` | `LearningOperation` — what is being learned, over which experience scope |
| `experience_scope` | `ExperienceScope` — the real executions/tasks used (independent-task count is derived) |
| `task_targeting` | which task types the claim is about |
| `baseline` / `horizon` / `horizon_tasks` | against what, over what window |
| `expected_changes` | `ExpectedChange` list — metric, direction, optional value |
| `learning_cost` | the cost of learning (separate from any execution cost) |
| `degradation_risk` | `RiskStatement` — what could get worse |
| `verification_conditions` | `VerificationCondition` list — what would confirm it |
| `trace` | shared traceability block |

**H is judged through observable consequences, never a latent vector.** The
contract defines no "H vector", no latent transition network, and no
composite capability score. It is training-free by design: a frozen model
may later fill these fields through two interfaces, and nothing here needs
fine-tuning.

---

## 3. The capability sources of `H`

`H = F(M, W_OR, Pi, R, T)`. Five **interacting** sources — not five fixed
score dimensions, and not a sum.

| Source | What it covers |
|---|---|
| `m` | Execution Evidence / Strategic Knowledge **and how usable it actually is** |
| `w_or` | consequence prediction and reliability judgement for OR solving |
| `pi` | strategy generation, composition, comparison, selection, switching, stopping |
| `r` | knowledge retrieval, applicability matching, context adaptation (**not** the final strategy decision) |
| `t` | tool/solver selection, composition, invocation, result handling |

Each source carries an evidence **status**, deliberately ordinal:

- `no_evidence` — nothing observed this source;
- `indirect_evidence` — a proxy exists (a knowledge reference, a tool
  listing, an execution count);
- `direct_evidence` — a measured outcome that speaks to this source.

`HarnessCapabilityEvidence.score_scheme` is fixed at
`no_composite_score`, and `validate_capability_evidence` rejects any
payload that tries to smuggle in a composite number. **This build has no
measured H**, and manufacturing one from counts is exactly the mistake the
contract exists to prevent.

`harness_state.knowledge` (knowledge refs), `experience` counts and
`tool_config` are **evidence about H**, not measured H. They map to `m`
and `t` as `indirect_evidence` only. `w_or`, `pi` and `r` stay
`no_evidence` — nothing in the legacy state observed them, so nothing is
inferred.

**There is no `E_hist` capability term.** Historical experience is not a
capability component; it is the substrate the sources are evidenced from.
And the H-evolution predictor's own quality is assessed separately — it is
not part of its own claim to have "got stronger".

---

## 4. Scope: attempt vs strategy execution window

One `execute_strategy` call is **one execution attempt** — a single solve
invocation with its own ExecutionRecord. A **strategy execution window** is
the larger real thing: writing the model, solving (possibly more than
once), repairing, verifying.

| | Attempt | Strategy window |
|---|---|---|
| Unit | one solve invocation | the whole strategy episode |
| In scope by default | `execute_strategy` | the types you declare |
| Auxiliary by default | — | `model`, `select_strategy`, `verify` |
| `CandidateRef.scope` | `"attempt"` | `"strategy_window"` (+ a real `window_id`) |

Rules the framework enforces:

- A window-scope prediction **must** reference a real `window_id`
  (`validate_strategy_outcome` rejects one without it).
- **Predicting before executing is legitimate; scoring is what waits.** A
  window-scope prediction about an UNEXECUTED (or still-running) window is a
  valid forecast with `trace.comparable=False` and the reasons — the window
  has no final numbers yet. What is refused is a contradiction: a
  `comparable=True` trace carrying `not_comparable_reasons`. A window is
  `comparable` **only when it is a completed real scope**. Every one of
  these makes it `comparable=False` with the reasons listed:
  - no in-scope executed attempt at all;
  - an in-scope attempt with **no linked execution**;
  - an in-scope attempt that has **not ended** (status `running`) — a
    window that is still moving has no final numbers;
  - the window's task / episode / strategy does **not** match the candidate
    it is used for.
  **Only a comparable window may be scored.**
- Window identity is checked, not trusted. A `candidate.window_id` naming
  another task / episode / strategy is refused, and so is a `window=`
  object handed in for a different candidate. Use
  `window_identity_problems(window, task_id=…, episode_id=…,
  strategy_id=…)` to check one yourself; `parse_window_id(wid)` splits an id
  back into its identity.
- Auxiliary actions are reported separately (`auxiliary_cost`) and are
  counted in the budget ledger — "the strategy was cheap" can never be
  claimed by omitting the modeling work that made it possible.
- A `rollup="reference"` action is never summed again: its cost lives on
  its child.
- Cost totals report `total` / `n_measured` / `n_items` / `complete` /
  `partial` per dimension. A partial total is labelled, never presented as
  the whole picture, and a dimension nothing measured stays `null`.

**B is not a predicted world-state object.** Budget declarations, execution
limits and the real resource ledger stay in `BudgetLedger`. Cost appears in
these contracts twice, and the two are never summed:

1. `cost` / `learning_cost` — the *predicted* cost of the candidate;
2. `trace.call_cost` — the *measured* spend of the prediction call itself.

---

## 5. Statuses: prediction ≠ fact ≠ verified effect

`status` on either contract:

| Value | Means |
|---|---|
| `draft` | being built, not yet validated |
| `contract_only` | **the contract is implemented, no prediction was made** — this build's state, and also the state whenever a provider is configured but produced nothing |
| `valid` | a real prediction, produced and validated (requires `service_available=true` **and** predicted content) |
| `unsupported` | the request is outside the supported prediction path; nothing was predicted |
| `invalid` | a prediction was attempted and failed validation |

Three things that are routinely confused, and are kept apart:

1. **prediction made** — the object exists;
2. **fact bound** — the real evidence landed and can be bound to it;
3. **effect verified** — the predicted improvement was *observed*.

Only (3) supports a claim that the harness got stronger. Adding knowledge
entries, accumulating evidence, or a model asserting an improvement is
none of the three. `VerificationCondition` records all three flags
separately.

---

## 6. Runnable example

See [`references/examples/contract_roundtrip.py`](examples/contract_roundtrip.py)
— it runs with no model, no network, and no solver:

```bash
PYTHONPATH=src python3 references/examples/contract_roundtrip.py
```

It builds a strategy-outcome contract for a task with **no `model` field
and several unknowns**, round-trips it through JSON, builds a
capability-evolution contract, and reads a legacy payload through the
legacy view. The script asserts its own invariants and prints the key
facts.

---

## 7. Migration table

How old shapes are preserved or interpreted. **Nothing is auto-migrated,
no table is rewritten, no old API is removed.**

| Old shape | New home | How it is read |
|---|---|---|
| `harness_state.knowledge` (refs) | `HarnessCapabilityEvidence.sources.m` | `indirect_evidence` — **never** a measured capability level or increment |
| `harness_state.experience` (counts) | `HarnessCapabilityEvidence.sources.m` | `indirect_evidence` — evidence *volume*, not capability |
| `harness_state.tool_config` | `HarnessCapabilityEvidence.sources.t` | `indirect_evidence` — availability is a precondition for `T`, not a measured `T` |
| `budget_state` | unchanged | stays in `BudgetLedger`; `B` is not a predicted object |
| `state_changes` (fine-grained) | `BenefitEstimate.progress_shape` | read as an **X progress shape only**; per-field successor values are not carried forward |
| `knowledge_changes` | `CapabilityEvolutionPrediction.expected_changes` | raw hypotheses for reference; **not** a measured capability increment |
| `predicted.quality` | `benefit.value` (`kind=solution_quality`) | value preserved; `metric`/`unit`/`baseline` are **not derivable** — reported as gaps |
| `predicted.failure_prob` | `risk.events[event=task_failure].probability` | probability preserved; **severity is not invented** |
| `predicted.cost` / `cost_measured` | `cost.expected` / `cost.expected_measured` | preserved verbatim, mask included |
| `confidence` | `uncertainty` (`source=model_self_report`) | recorded as **uncalibrated**, never as a probability |
| `call_cost` | `trace.call_cost` | the prediction call's own spend, unchanged |
| `evidence_basis` / `unsupported_fields` | `trace.*` | preserved verbatim |
| `action_spec` | `CandidateRef` | mechanical mapping; the spec's execution `params` and `budget_hint` are carried through **verbatim**. `measurement_scope="attempt"` → `scope="attempt"` (recorded as `scope_basis="legacy_attempt"`); a legacy `"task"` scope is **refused**, never silently shrunk — pass `scope=` to declare the narrowing yourself |
| `OutcomePrediction`, snapshots, config, logs | unchanged | still readable; the old path still returns `not_configured` with no provider |

**Not convertible, on purpose** (the full list is `LEGACY_UNMAPPABLE` in
`contracts.py`, and it is returned by every legacy read): knowledge
references into a capability level, experience counts into capability,
tool availability into a `T` level, `budget_state` into a predicted state,
and fine-grained `state_changes` into per-field predictions. No capability
increment, risk severity or measurement is derived from any of them.

---

## 8. Legacy prediction modes

The experiment ablation switches keep their names — they are NOT renamed.
This is what they mean in the new vocabulary:

| Legacy mode | Contract kinds covered | Knowledge term |
|---|---|---|
| `x-b-only` | `strategy_outcome` | off |
| `h-x-b` | `strategy_outcome` + `capability_evolution` | forced off |
| `h-x-b-value` (default) | `strategy_outcome` + `capability_evolution` | live at `--delta` (default **0.0**) |

Use `prediction_kinds_for_mode(mode)` to get this mapping programmatically.

---

## 9. What is NOT in this phase

The contract is one phase of a larger reconstruction. **Not implemented
here**, and not to be described as done:

- no new semantic extractor and no retrieval rework;
- no switch away from the legacy `predict_outcome` shadow path (it is
  unchanged; the wm-so/1 service is a separate path — see
  [strategy_outcome.md](strategy_outcome.md));
- no task-closing scheduler, no automatic offline learning schedule;
- no H evaluation system;
- no multi-step latent rollouts;
- no model fine-tuning or training.

Also deliberately out of scope: a universal H score, renaming the existing
action vocabulary (the six action types stay), reviving the retired
`understand` action, and replacing `BudgetLedger`.

> **Phase 2 note.** The INPUT side of a prediction is now wired: see
> [references/prediction_context.md](prediction_context.md). A prediction is
> conditioned on a frozen `PredictionContext` (joint problem representation,
> X/B, retrieval evidence, capability evidence, execution constraints).
>
> **Phase 3 (M3) note.** The strategy-outcome prediction SERVICE is now
> implemented under the `wm-so/1` protocol: see
> [references/strategy_outcome.md](strategy_outcome.md). `predict-strategy`
> fills a `StrategyOutcomePrediction` from a frozen context and one
> candidate; `plan-next --protocol strategy-outcome` compares candidates on
> it; `bind-strategy` links the real execution. The legacy
> `predict_outcome` path is unchanged.

## 10. Verification checklist for a consuming agent

- [ ] Did I get `status="contract_only"`? Then **no prediction was made** —
      do not read the object as a forecast. Check `prediction_made` and
      `provider_configured` separately: a configured provider is not a
      prediction.
- [ ] Is `service_available` true for `capability_evolution`? It should not
      be — this build implements no capability-evolution service, and a
      configured provider must not make it look available.
- [ ] Is `benefit.value` present? Then a `baseline` must be present too.
- [ ] Is `scope="strategy_window"`? Then check `trace.comparable` — if it
      is `False`, the prediction may not be scored. Also confirm the window
      really belongs to this task / episode / strategy.
- [ ] Am I about to treat a knowledge entry appearing as a capability gain?
      That is `prediction_made`, not `effect_verified`.
- [ ] Am I reading a legacy payload? Check `legacy_view.gaps` and
      `legacy_view.unmappable` for what was **not** derived.
