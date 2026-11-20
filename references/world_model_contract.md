# World-model prediction inputs and outputs

Use this reference to read a `plan-next` forecast and inspect the evidence behind it. See [strategy selection and execution binding](strategy_outcome.md) for how the planner scores and acts on the forecast.

## 1. Prediction inputs

`plan-next` builds one frozen `wm-context/4` context for a decision and shares it across the candidates. Each `wm-so/1` request supplies that context's provider view, the candidate's method, solver, configuration and scope, and the decision's benefit convention and cost window.

| Material | Where to read it | What it contributes |
|---|---|---|
| Problem | `joint` in storage, `joint_problem` in the request | Original task text, mathematical attributes, structural profile and their sources |
| Solving conditions | `solving_context` and `execution_constraints` | Current progress, declared limits and usage, available tools, known modelling phase and solver interface |
| Retrieved evidence | `retrieval` in storage, `retrieval_evidence` in the request | Relevant executions, structural recommendations and applicable rule versions |
| Capability evidence | `capability` in storage, `harness_capability` in the request | Evidence about memory, prediction, planning, retrieval and tools. Each source reports `no_evidence`, `indirect_evidence` or `direct_evidence`. `capability_version` identifies the configuration and memory content |
| Published feedback | `strategy_outcome_calibration`, `prediction_execution_pairs`, `prediction_reminders` and `hplus_feedback` in the request | Prior observations, selected cases and completed induction reviews |

Start with the [shared task input](modeling.md#start-from-the-shared-original-input). Mathematical attributes use `--math`, the declared model, structured `spec` and CIR in that order. `joint.math` records attribute origins and `joint.unknowns` lists unresolved attributes. The effective CIR drives profiling and retrieval. The saved context retains its relations, while the provider gets the task text, derived attributes and profile with the CIR marked as omitted.

Structural recommendations explain applicability. Semantic hits identify textually similar cases, including cases from other cells. The request deduplicates hits by `layer:id`, preserves their evidence classes and points both channels to the same body. Read `degraded[]` for a channel that could not run and the channel's hit list for a successful search with no matches. See [recall and index recovery](commands.md#3-understand-and-retrieve).

Use `orx context --task task.json --episode E` to inspect a separately assembled input. `--no-persist` previews it. `orx context --context-id CTX` reads a saved working input during its reuse lifetime. In Python, pass it as `context=ctx` to `ORHarness.predict_strategy_outcome` with the same task, episode and effective CIR. `task_digest` identifies the original input. `effective_input_digest` covers the task with the resolved CIR and is checked for reuse. See [reuse and expiry](snapshot_lifecycle.md#context-reuse-and-expiry).

## 2. Published prediction feedback

A new decision can draw on published calibration and completed reviews available before its cutoff. It freezes the selected evidence with the context. Per-candidate request selection further matches that evidence to the candidate's metric, scope and execution conditions.

| Request block | What the model can use |
|---|---|
| `strategy_outcome_calibration.conditional_g` | Attributable passes, failures and unknowns on comparable tasks. Match the observation rule, metric, scope and known structure and execution settings |
| `cost_evidence` | Raw cost ranges and executed cases that match the measurement scope, known phase and solving conditions |
| `implementation_r_evidence` | Technical outcomes from closed attempts with matching solver interface, with independent episode counts and source references |
| `prediction_execution_pairs` | Prior predictions paired with performed methods, checks, measured results, failures and comparability notes |
| `prediction_reminders` | Applicable risk and interval observations. Global G and C residuals remain available in calibration diagnostics |
| `hplus_feedback` | Relevant settled receipts with the original learning hypothesis, actual facts, supported conclusion and exact rule versions |

`--execution-context` or `OR_EXECUTION_CONTEXT` can declare the solver interface and modelling phase on the first decision, for example `{"solver_interface":"highs","phase":"initial_modelling"}`. Execution-agent consistency is managed by the caller. The prediction provider's identity continues to track forecast calibration.

Request selection preserves missing support as unknown and marks background or partial evidence. The input budget (`OR_HARNESS_INPUT_BUDGET_TOKENS`) can trim selected evidence further. Inspect the saved request diagnostics to see what survived. See [published feedback and calibration](episode_closeout.md#published-feedback) and [H+ learning labels](induction.md#read-publication-completion-and-learning-separately) for the observation rules.

## 3. Prediction fields and observation scopes

| Field | Meaning |
|---|---|
| `candidate` | The proposed strategy, method, solver, config and scope |
| `benefit` (G) | Expected benefit under the decision's metric, unit and baseline |
| `cost` (C) | Raw resource estimates in `expected`. `expected_measured` marks supplied dimensions |
| `risk` (R) | Predicted event probabilities. U uses `implementation_failure.probability` |
| `capability_gain` (H+) | `decision` (`0` or `1`), learning hypothesis, basis, applicability and checkable conditions |
| `uncertainty` | Reported uncertainty about the forecast, with any interval kind and coverage supplied by the contract |
| `trace` | Input and protocol identity, evidence basis, unsupported fields and measured forecast-call cost |

### G and acceptance

The default G convention is `effective_completion/task_check_passed_given_faithful_execution` in `probability` under `wm-g-faithful/1`. It asks whether the original task would pass its acceptance check if the candidate's steps were faithfully implemented. The framework fixes this convention before the call. Use a finite JSON number in `[0,1]` for that probability and retain the baseline. When G is unavailable, read the reason and `benefit.notes`, which can still explain how the candidate addresses a decisive task requirement.

Other registered benefit kinds include `solution_quality` (normalized quality), `valid_progress` and `correct_infeasibility_diagnosis`. Use the declared metric and unit when comparing results. For intervals, `interval_kind` identifies an individual outcome or a mean. See [G attribution](episode_closeout.md#g-attribution-evidence) for how an executed attempt supplies a label.

### C and R

C's default measurement window runs from the shared decision anchor to task termination. Compare forecasts with matching actual dimensions and time scopes. Use `trace.call_cost` for the measured spend of making the forecast itself. See [C observations](episode_closeout.md#c-observations) for supported host reports and cost coverage.

The token forecast is `cost.output_tokens`: execution-model tokens generated
during the remaining modelling, coding, self-checks and expected rework. The
task text and its length are known inputs. Missing output is unknown, not zero.
The host's trustworthy complete `completion_tokens` in that same window supply
the actual value; an initial decision includes later retries, and a retry's
decision starts its own window. Raw usage, budgets and historical `llm_tokens`
still mean input plus output. They cannot substitute for output-token evidence
or its normalization scale. See [generated-token observations](strategy_outcome.md#4-budgets-stops-and-archiving-an-interrupted-run).

R uses the one-attempt technical-failure convention `wm-r-technical/1`. A finite numeric `implementation_failure.probability` in `[0,1]` supplies R. Other event predictions remain useful for diagnosis. U treats an explicit zero as a supplied value and records unavailable terms. See [technical R observations](episode_closeout.md#technical-r-observations) and [U scoring](strategy_outcome.md#3-score-one-or-more-candidates).

The candidate can cover an `attempt` or a `strategy_window` spanning the declared modelling, execution, repair and verification activities. Window scoring requires a matching task, episode and strategy, ended linked attempts, and `trace.comparable=True`. Read coverage and auxiliary cost when inspecting a window total.

For C, `cost.retries` under `remaining_to_task_end` predicts the expected
additional solve attempts AFTER this prediction's bound execution. It may be
a non-negative fraction or zero. The frozen convention is
`cost_scope.retries_convention.observation_rule_version` =
`wm-retry-additional-attempts/1`. Execution records separately store 0/1
increments within the same task/episode: A/B/C gives 0/1/1, an episode total
of 2, and per-decision future actuals of 2/1/0. See [retry observations](episode_closeout.md#4-the-per-field-evaluation)
for unknown evidence and historical conventions.

### H+ and induction

Read the assigned switch in `trace.model_info.hplus`. During library building, close-out schedules a review for an executed, bound positive signal. The review keeps the prediction, actual evidence and supported learning result in an induction receipt that can inform later forecasts. Read the [induction handoff](induction.md#when-to-start-induction) for what to do with its ID.

## 4. Inspect a prediction and its input

```bash
orx inspect --bank predictions --prediction P
orx context --context-id CTX --audit
```

Read `result.prediction.prediction` for the saved forecast. Its `trace.model_info.prediction_context_id` gives CTX. The context audit gives the stored input manifest and version references. Read `trace.model_info.request_omissions` on the same forecast to see omissions from the request. For example, a settled H+ receipt can appear in the frozen context and be trimmed before the provider call. The audit remains readable after the working input is released, subject to [retention](snapshot_lifecycle.md#context-reuse-and-expiry).

| Prediction status | Meaning |
|---|---|
| `draft` | Contract in preparation |
| `contract_only` | Assembled contract |
| `valid` | Produced and validated forecast |
| `unsupported` | Input outside the supported prediction path |
| `invalid` | Attempted forecast with failure or invalid fields |

Use `prediction_made` to confirm that a forecast exists. Read `trace.unsupported_fields` for unavailable values. `provider_configured` and `service_available` describe the setup. `orx contract` constructs or reads the contract. Use the [command reference](commands.md#stored-forecasts-and-their-conventions) for provider failures, and [historical records](snapshot_lifecycle.md#read-historical-records) for old payloads.

Runnable examples are available for [context assembly](examples/prediction_context.py) and the [contract round trip](examples/contract_roundtrip.py).
