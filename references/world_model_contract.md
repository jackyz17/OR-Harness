# World-model contract

Read this page when you need an exact field definition, a status value, or the mapping of a legacy shape — it is the single authority for the unified contract. For what a prediction is conditioned on see [prediction_context.md](prediction_context.md), for the wm-so/1 loop see [strategy_outcome.md](strategy_outcome.md), and for how outcomes are scored see [episode_closeout.md](episode_closeout.md).

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
| Inspect a candidate's prediction contract | `orx contract --kind strategy_outcome` / `ORHarness.build_strategy_outcome_contract` | a task JSON + a candidate (strategy id, optional solver/config/scope) | a `StrategyOutcomePrediction` contract |
| Read a stored prediction | `orx contract --payload <json>` / `ORHarness.read_prediction_payload` | the payload | a version-identified view (current / legacy / unsupported) |
| See what is known about harness capability | `ORHarness.capability_evidence` | nothing (task optional) | `HarnessCapabilityEvidence` — evidence, **no score** |
| See the real scope a strategy occupied | `ORHarness.strategy_execution_window` | task id (+ episode, strategy) | `StrategyExecutionWindow` with a `comparable` flag |

**You do not need a complete mathematical model.** A task without a `model` field is a normal state — the contract is built from the task text, the CIR, and the profile. You also do not need to fill everything in: every field is optional, and an absent field stays absent (never a placeholder zero).

**Nothing here calls a model.** Building a contract performs no network call. Three DIFFERENT facts are kept apart, because conflating them is how an empty object came to be read as a forecast:

| Fact | Meaning |
|---|---|
| `provider_configured` | a provider object is attached to this instance |
| `service_available` | this build **implements** the service for this kind **and** a provider is configured |
| `prediction_made` | a prediction really was produced and passed validation |

Only the third yields `status="valid"`. Merely constructing a contract returns `contract_only` **even when a provider is configured** — a configured provider with zero model calls is not a prediction, and an empty benefit/cost/risk is not a forecast.

---

## 2. Strategy consequence prediction

Answers: *under the current problem (P), solving context (X) and harness capability condition, what would this candidate strategy produce?*

| Field | Meaning |
|---|---|
| `candidate` | `CandidateRef`: strategy id, solver, config, preconditions, expected scope, stop conditions, and `scope` |
| `benefit` | `BenefitEstimate` — the gain, **with its metric, unit and baseline** |
| `cost` | `ExpectedCost` — raw `CostVector` resource predictions. `expected_measured` marks supplied predictions; observed costs come from execution evidence |
| `risk` | `RiskStatement` — named events; scored R is only `implementation_failure.probability`, separate from cost |
| `uncertainty` | `UncertaintyStatement` — execution randomness and knowledge gaps. `model_self_report` records the model's uncalibrated assessment |
| `capability_gain` | Same-call binary `decision` (JSON 0 or 1), `claim`, `verification_conditions`, `basis`, `applies_to`, and per-metric expected changes with their units. Scheduling reads the frozen `trace.model_info.hplus`. See [H+ and review handoff](induction.md#when-to-start-induction) |
| `trace` | `PredictionTrace` — version, input, evidence, unsupported fields, call cost |

**Benefit is not one arbitrary 0–1 score.** `benefit.kind` says which currency you are in:

| `kind` | Meaning | Typical metric |
|---|---|---|
| `effective_completion` | the task actually got completed | completion (0/1) |
| `solution_quality` | a qualified solution's quality | normalized objective gap, `1 - mip_gap` |
| `valid_progress` | a real intermediate step forward | progress ratio |
| `correct_infeasibility_diagnosis` | a *correct* infeasibility verdict | diagnosis (0/1) |

`benefit.value` requires a `baseline` — a gain with nothing to measure it against is not a prediction. `benefit.value = None` means **unknown**, and that is a first-class state: it is not zero, and it is not "no gain".

Supply a normalized value in `[0,1]` for `solution_quality`. Business objective values belong to task outcomes. For an interval, `interval_kind` distinguishes an `outcome` for one run from a `mean` over runs, and `interval_coverage` gives its nominal coverage when available. Missing kinds remain unstated. An effective-completion probability describes a mean over runs.

New strategy requests default to `effective_completion/task_check_passed_given_faithful_execution`, unit `probability`, with `BenefitEstimate.observation_rule_version="wm-g-faithful/1"`. This is P(original-task acceptance | faithfully implement the candidate steps, under the agreed solving conditions). Method omissions remain omissions. The finite JSON numeric value must be in [0,1]; bool/string values are unavailable. The framework fixes convention/scope before the call; a response mismatch makes G unavailable with a reason while preserving valid C/R and the explanatory sentence in `benefit.notes`. This sentence identifies a decisive task condition, candidate operation and expected consequence, or missing evidence, and survives a missing numeric value.

Use the [G attribution evidence](episode_closeout.md#g-attribution-evidence) to connect the recorded code and original-task checks to the G observation.

Old forecasts and explicit old conventions retain their original meanings, units and observation channels. An absent `observation_rule_version` is never silently upgraded. In particular `normalized_objective_gap` still measures the solver's own `max(0,1-gap)`, and `task_result_check_passed` retains the legacy completion rules.

**R: one-attempt technical failure.** New requests fix `risk_convention` to `implementation_failure`, `unit=execution`, `scope=attempt`, `observation_rule_version=wm-r-technical/1`, saved in `trace.model_info`. Only that event's finite JSON numeric probability in [0,1] participates in scoring. Missing, boolean, string, duplicate or invalid probabilities become unavailable with an unsupported reason; explicit 0 remains valid and valid G/C survive. Timeout, environment failure, budget exhaustion, solver infeasibility and task-check events remain diagnostics and never replace R.

See [technical R observations](episode_closeout.md#technical-r-observations) for executor evidence and [candidate scoring](strategy_outcome.md#3-score-one-or-more-candidates) for the shared U basis.

---

## 3. The capability sources of `H`

`H = F(M, W_OR, Pi, R, T)`. Five **interacting** sources — not five fixed score dimensions, and not a sum.

| Source | What it covers |
|---|---|
| `m` | Execution Evidence / Strategic Knowledge **and how usable it actually is** |
| `w_or` | consequence prediction and reliability judgement for OR solving |
| `pi` | strategy generation, composition, comparison, selection, switching, stopping |
| `r` | knowledge retrieval, applicability matching, context adaptation (**not** the final strategy decision) |
| `t` | tool/solver selection, composition, invocation, result handling |

Each source carries an evidence **status**, deliberately ordinal:

- `no_evidence` — nothing observed this source;
- `indirect_evidence` — a proxy exists (a knowledge reference, a tool listing, an execution count);
- `direct_evidence` — a measured outcome that speaks to this source.

`HarnessCapabilityEvidence.score_scheme` is fixed at `no_composite_score`, and `validate_capability_evidence` rejects any payload that tries to smuggle in a composite number. **This build has no measured H**, and manufacturing one from counts is exactly the mistake the contract exists to prevent.

`harness_state.knowledge` (knowledge refs), `experience` counts and `tool_config` are **evidence about H**, not measured H. They map to `m` and `t` as `indirect_evidence` only. `w_or`, `pi` and `r` stay `no_evidence` — nothing in the legacy state observed them, so nothing is inferred.

Historical experience supplies evidence for the capability sources.

---

## 4. Scope: attempt vs strategy execution window

One `execute_strategy` call is **one execution attempt** — a single solve invocation with its own ExecutionRecord. A **strategy execution window** is the larger real thing: writing the model, solving (possibly more than once), repairing, verifying.

| | Attempt | Strategy window |
|---|---|---|
| Unit | one solve invocation | the whole strategy episode |
| In scope by default | `execute_strategy` | the types you declare |
| Auxiliary by default | — | `model`, `select_strategy`, `verify` |
| `CandidateRef.scope` | `"attempt"` | `"strategy_window"` (+ a real `window_id`) |

Rules the framework enforces:

- A window-scope prediction **must** reference a real `window_id` (`validate_strategy_outcome` rejects one without it).
- **Predicting before executing is legitimate; scoring is what waits.** A window-scope prediction about an UNEXECUTED (or still-running) window is a valid forecast with `trace.comparable=False` and the reasons — the window has no final numbers yet. What is refused is a contradiction: a `comparable=True` trace carrying `not_comparable_reasons`. A window is `comparable` **only when it is a completed real scope**. Every one of these makes it `comparable=False` with the reasons listed:
  - no in-scope executed attempt at all;
  - an in-scope attempt with **no linked execution**;
  - an in-scope attempt that has **not ended** (status `running`) — a window that is still moving has no final numbers;
  - the window's task / episode / strategy does **not** match the candidate it is used for. **Only a comparable window may be scored.**
- Window identity is checked, not trusted. A `candidate.window_id` naming another task / episode / strategy is refused, and so is a `window=` object handed in for a different candidate. Use `window_identity_problems(window, task_id=…, episode_id=…, strategy_id=…)` to check one yourself; `parse_window_id(wid)` splits an id back into its identity.
- Auxiliary actions are reported separately (`auxiliary_cost`) and are counted in the budget ledger — "the strategy was cheap" can never be claimed by omitting the modeling work that made it possible.
- A `rollup="reference"` action is never summed again: its cost lives on its child.
- Cost totals report `total` / `n_measured` / `n_items` / `complete` / `partial` per dimension. A partial total is labelled, never presented as the whole picture, and a dimension nothing measured stays `null`.

**B is not a predicted world-state object.** Budget declarations, execution limits and the real resource ledger stay in `BudgetLedger`. The prediction records two costs with different scopes.

1. `cost` gives the predicted cost of running the candidate.
2. `trace.call_cost` gives the measured spend of the prediction call.

For strategy-outcome C, the default measurement window is the shared decision anchor through real task termination. `cost.expected` and `expected_measured` keep their raw-value/mask meanings. `cost_evidence` is request evidence, not an extra cost dimension. `wm-c-window/1` observations include complete host-window counts or counter deltas and timestamped call ledgers; they do not relabel legacy attempt totals. A full host total already includes modelling/checking/repair work, so auxiliary action totals are not added again. Per-dimension error diagnostics keep both `solver_runtime_s` and `remaining_latency_s`; utility's existing component charging rule is unchanged. A script-wall proxy is not precise inner solver runtime. Missing C leaves real observations intact and supplies no prediction-error sample.

---

## 5. Prediction status and H+ review

| `status` | Meaning |
|---|---|
| `draft` | Contract in preparation |
| `contract_only` | An assembled contract |
| `valid` | A prediction produced by the configured service and validated |
| `unsupported` | Request outside the supported prediction path |
| `invalid` | Prediction attempt failed validation |

Check `prediction_made` for a completed forecast. Scoring a `strategy_window` prediction requires `trace.comparable=True` and a window for the same task, episode and strategy.

The frozen H+ decision controls review after a bound execution. A completed review records the original hypothesis, the actual evidence and the supported learning result in its induction receipt. Read the [learning labels](induction.md#read-publication-completion-and-learning-separately) and [feedback input](prediction_context.md#6d-the-h-training-receipt-block).

---

## 6. Runnable example

See [`references/examples/contract_roundtrip.py`](examples/contract_roundtrip.py) — it runs with no model, no network, and no solver:

```bash
PYTHONPATH=src python3 references/examples/contract_roundtrip.py
```

It builds a strategy-outcome contract for a task with several unknown fields, round-trips it through JSON, and reads a legacy payload through the legacy view. The script asserts its own invariants and prints the key facts.

---

## 7. Migration table

Stored predictions retain their original version, scope and fields. These mappings explain the legacy read view.

| Old shape | New home | How it is read |
|---|---|---|
| `harness_state.knowledge` (refs) | `HarnessCapabilityEvidence.sources.m` | `indirect_evidence` — **never** a measured capability level or increment |
| `harness_state.experience` (counts) | `HarnessCapabilityEvidence.sources.m` | `indirect_evidence` — evidence *volume*, not capability |
| `harness_state.tool_config` | `HarnessCapabilityEvidence.sources.t` | `indirect_evidence` — availability is a precondition for `T`, not a measured `T` |
| `budget_state` | unchanged | stays in `BudgetLedger`; `B` is not a predicted object |
| `state_changes` (fine-grained) | `BenefitEstimate.progress_shape` | read as an **X progress shape only**; per-field successor values are not carried forward |
| `predicted.quality` | `benefit.value` (`kind=solution_quality`) | value preserved; `metric`/`unit`/`baseline` are **not derivable** — reported as gaps |
| `predicted.failure_prob` | `risk.events[event=task_failure].probability` | probability preserved; **severity is not invented** |
| `predicted.cost` / `cost_measured` | `cost.expected` / `cost.expected_measured` | preserved verbatim, mask included |
| `confidence` | `uncertainty` (`source=model_self_report`) | recorded as **uncalibrated**, never as a probability |
| `call_cost` | `trace.call_cost` | the prediction call's own spend, unchanged |
| `evidence_basis` / `unsupported_fields` | `trace.*` | preserved verbatim |
| `action_spec` | `CandidateRef` | mechanical mapping; the spec's execution `params` and `budget_hint` are carried through **verbatim**. `measurement_scope="attempt"` → `scope="attempt"` (recorded as `scope_basis="legacy_attempt"`); a legacy `"task"` scope is **refused**, never silently shrunk — pass `scope=` to declare the narrowing yourself |
| `OutcomePrediction`, snapshots, config, logs | unchanged | still readable; the old path returns `not_configured` with no provider — a BLOCKER, since the world model is mandatory |

**Not convertible, on purpose** (the full list is `LEGACY_UNMAPPABLE` in `contracts.py`, and it is returned by every legacy read): knowledge references into a capability level, experience counts into capability, tool availability into a `T` level, `budget_state` into a predicted state, and fine-grained `state_changes` into per-field predictions. No capability increment, risk severity or measurement is derived from any of them. A legacy payload's own report names what was NOT derived under `legacy_view.gaps` and `legacy_view.unmappable`.

---

## Versioned strategic rules (`or-rule/2`)

New root rows store active/candidate pointers and state; immutable condition values hold canonical when/do/expect bodies, assessments and decisive facts. Separate `rule_versions`, `rule_checks` and `rule_events` tables associate validation, use, effects and explicit aliases with exact versions. Numeric predictions retain their contract and absence semantics. Maintenance deltas name resolved evidence and committed updates. Adoption snapshots freeze immutable knowledge references rather than live entry lookups.

`induction-material/2` and recall expose short views and whole-value detail references. See [induction material fields](induction.md#read-material-and-choose-comparisons). Historical unversioned rows retain their scope, and new submissions in the old relation format become v2 candidates. `execution-evidence/2` stores shared descriptions and deduplicates the performed-method receipt. Audit decoding preserves runtime values, provenance, prediction and cost coverage, and explicit zeroes.
