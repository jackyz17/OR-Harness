# World-model prediction inputs and outputs

Use this reference to read a `plan-next` forecast and inspect the evidence behind it. See [strategy selection and execution binding](strategy_outcome.md) for how the planner scores and acts on the forecast.

## 1. Prediction inputs

`plan-next` builds one frozen `wm-context/4` context for a decision and shares it across the candidates. Each `wm-so/1` request supplies that context's provider view, the candidate's method, solver, configuration and scope, and the decision's benefit convention and cost window.

| Material | Where to read it | What it contributes |
|---|---|---|
| Problem | `joint` in storage, `joint_problem` in the request | Original task text, mathematical attributes, structural profile and their sources |
| Solving conditions | `solving_context` and `execution_constraints` | Current progress, declared limits and usage, available tools and solver interface. A legacy stored `phase` is readable for audit but no longer a prediction condition |
| Retrieved evidence | `retrieval` in storage, `retrieval_evidence` in the request | Relevant executions, structural recommendations and rule versions offered for the agent's applicability assessment |
| Capability evidence | `capability` in storage, `harness_capability` in the request | Evidence about memory, prediction, planning, retrieval and tools. Each source reports `no_evidence`, `indirect_evidence` or `direct_evidence`. `capability_version` identifies the configuration and memory content |
| Published feedback | `strategy_outcome_calibration`, `prediction_execution_pairs`, `prediction_reminders` and `hplus_feedback` in the request | Prior observations, selected cases and completed induction reviews |

Start with the [shared task input](modeling.md#start-from-the-shared-original-input). Mathematical attributes use `--math`, the declared model, structured `spec` and CIR in that order. `joint.math` records attribute origins and `joint.unknowns` lists unresolved attributes. The effective CIR drives profiling and retrieval. The saved context retains its relations, while the provider gets the task text, derived attributes and profile with the CIR marked as omitted.

Structural recommendations report only deterministic MACHINE compatibility with the profile (the `family` label and the coupling-cell intervals). Semantic hits identify textually similar cases, including cases from other cells. Neither is adoption and neither is full applicability: a rule's applicability is its own **when** paragraph, and only the agent can judge it. A different `family` LABEL is not by itself a contradiction (a shared mathematical mechanism may transfer); only a measured structural value outside the rule's claimed cell is a confirmed conflict. An empty predicate set means "no machine-checkable structural condition", never "applies to everything". Knowledge hits carry `rule_body` (`when`/`do`/`expect`), `machine_applicability` (`state`, checks, unknown fields, conflicts, family-label relation), `semantic_applicability` (`state: requires_agent_assessment`, the `when` text) and `adoption_state: not_adopted`; adoption is recorded separately through the execution's `used_entries` and effect attribution. The request deduplicates hits by `layer:id`, preserves their evidence classes and points both channels to the same body. Read `degraded[]` for a channel that could not run and the channel's hit list for a successful search with no matches. See [recall and index recovery](commands.md#3-understand-and-retrieve).

Use `orx context --task task.json --episode E` to inspect a separately assembled input. `--no-persist` previews it. `orx context --context-id CTX` reads a saved working input during its reuse lifetime. In Python, pass it as `context=ctx` to `ORHarness.predict_strategy_outcome` with the same task, episode and effective CIR. `task_digest` identifies the original input. `effective_input_digest` covers the task with the resolved CIR and is checked for reuse. See [reuse and expiry](snapshot_lifecycle.md#context-reuse-and-expiry).

## 2. Published prediction feedback

A new decision can draw on published calibration and completed reviews available before its cutoff. It freezes the selected evidence with the context. Per-candidate request selection further matches that evidence to the candidate's metric, scope and execution conditions.

| Request block | What the model can use |
|---|---|
| `strategy_outcome_calibration.conditional_g` | Attributable passes, failures and unknowns on comparable tasks. Match the observation rule, metric, scope and known structure and execution settings |
| `cost_evidence` | LOCAL similar executions with comparable remaining-task cost (four dimensions, missing dimensions named) plus the GLOBAL four-dimensional pre-decision distribution and prediction-vs-actual error |
| `implementation_r_evidence` | LOCAL technical cases selected by embedding similarity and reranked by solver interface/method similarity, each with versioned selected-window implementation facts and bound-formal-attempt timeout/environment facts, plus the separate GLOBAL event base rates and this model's per-event Brier performance |
| `prediction_execution_pairs` | Prior predictions paired with performed methods, checks, measured results, failures and comparability notes |
| `history_retrieval` | The frozen retrieval record: embedding backend, per-view similarity floors, the query views, admission/pool counts and degradation (when local history is unavailable) |
| `prediction_reminders` | Applicable risk and interval observations. Global G and C residuals remain available in calibration diagnostics |
| `hplus_feedback` | Relevant settled receipts with the original learning hypothesis, actual facts, supported conclusion and exact rule versions |

`--execution-context` or `OR_EXECUTION_CONTEXT` can declare the solver interface on the first decision, for example `{"solver_interface":"highs"}`. A stored legacy `phase` field is still readable for audit but C no longer classifies, groups or predicts with a modelling phase; a missing phase is no longer a gap. Execution-agent consistency is managed by the caller. The prediction provider's identity continues to track forecast calibration.

Request selection preserves missing support as unknown and marks background or partial evidence.

The authoritative history-retrieval configuration is frozen at
`PredictionContext.history_retrieval.config` and retained with the original
active runtime input. Its actual provider projection is flat under the
request's `prediction_context.history_retrieval`. The saved prediction's
`trace.model_info.input_selection.history_selection.retrieval.config` copies
the fields **actually sent**: backend model id, local-case/pool limits, the
three admission thresholds, the separate method-relevance threshold,
verification flag and all four threshold sources. It is a compact projection,
not a second configuration computed from live environment variables;
`config_source` identifies its source. The full `threshold_basis`, backend
source and fallback diagnostics remain in the frozen configuration. After
runtime-input expiry the saved prediction still retains the sent projection.

The same audit retains retrieval status and admission/pool accounting. Its
G/R selection counts and C reference coverage copy the final request after
budget trimming. Use these together to distinguish unverified thresholds,
an unavailable index, admissions, and delivered/omitted cases. No case bodies
are duplicated in the configuration audit, and request budgets are unchanged.

The history block (`prediction_execution_pairs`) is delivered through a provider-only compact view derived on every read; the stored block and calibration database are unchanged. Each pair keeps its task conditions, the recorded plan/actual method difference, the original prediction, the real observation, the error, the recorded attribution, the G/C/R metric and observation-rule versions, eligibility/unknown/comparability facts, and a `detail_ref` that restores every reduced field. A crash keeps no observed G (never zero) and an unknown risk label stays unknown. Free-text and audit scaffolding are replaced by references before any case is dropped; cases are dropped only from the tail of the relevance order, and every dropped case is reported with its inspect reference.

Three independent provider-only ceilings apply: history feedback (`OR_HARNESS_HISTORY_FEEDBACK_CHARS`, default 30000), G/C/R evidence (`OR_HARNESS_GCR_EVIDENCE_CHARS`, default 24000 after the local-case ceiling rose to 10 per signal) and other non-core material (`OR_HARNESS_NONCORE_CHARS`, default 20000). The WHOLE request (system prompt + JSON body) is then measured against `OR_HARNESS_INPUT_BUDGET_TOKENS`; the provider's own `count_tokens` is used when exposed, otherwise a labelled conservative estimate counts CJK/JSON-dense text more heavily than `chars/4`. On overflow the history feedback is reduced first, then other non-core blocks, then G/C/R evidence; the current task, candidate, execution constraints and prediction contract are never truncated. If that protected core alone exceeds the bound the request is NOT sent: the prediction is recorded as invalid with `failure.kind=request_over_budget` and its size diagnostics. Inspect `request_size` (`tokens`, `token_basis`, `trimming`, `sendable`, `over_budget`) and `request_omissions` on the saved prediction. See [published feedback and calibration](episode_closeout.md#published-feedback) and [H+ learning labels](induction.md#read-publication-completion-and-learning-separately) for the observation rules.

## 3. Prediction fields and observation scopes

| Field | Meaning |
|---|---|
| `candidate` | The proposed strategy, method, solver, config and scope |
| `benefit` (G) | Expected benefit under the decision's metric, unit and baseline |
| `cost` (C) | Raw resource estimates in `expected`. `expected_measured` marks supplied dimensions |
| `risk` (R) | Predicted event probabilities. U uses the MAXIMUM of the events all ranked candidates supplied validly: `implementation_failure`, `timeout`, `environment_failure` |
| `capability_gain` (H+) | `decision` (`0` or `1`), a concrete learning hypothesis, basis, applicability and checkable conditions. It is not a difficulty or success rating |
| `uncertainty` | Reported uncertainty about the forecast, with any interval kind and coverage supplied by the contract |
| `trace` | Input and protocol identity, evidence basis, unsupported fields and measured forecast-call cost |

### G and acceptance

The default G convention is `effective_completion/task_check_passed_given_faithful_execution` in `probability` under `wm-g-faithful/2`. It measures whether the current Agent and Harness can faithfully execute the candidate, effectively complete the original task and sufficiently verify the result. Task understanding, modelling, solving and verification ability all matter. The framework fixes the convention before the call. Use a finite JSON number in `[0,1]` and retain the baseline and evidence-grounded `benefit.notes`.

Pure code/API errors and Harness interruption belong primarily to R, without
lowering G again for that same technical cause. A wrong model, unsafe reduction,
approximation that changes the task or absent quality guarantee affects G.
An explicitly time-limited approximation can affect G through the quality it
delivers on normal completion. A mathematically correct model does not alone
force G=1. With faithful execution, an evaluable result and valid sourced
attribution, insufficient verification, circular evidence, confirmed failure
with an unknown cause and unresolved trusted-reference conflicts yield G=0.
G=1 requires sufficient credible original-task verification. Unknown covers
missing opportunity, unestablished fidelity, invalid binding, lost key facts,
external disruption or an unresolved defective task/reference standard.
The checker's passed/failed/insufficient verdict remains a separate fact.
Historical `/1` observations keep their original attribution rules, and numeric
feedback groups and current global G diagnostics keep the versions separate.

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

R uses `wm-r-max-event/2` with implementation observations under
`wm-r-implementation-window/1`. Implementation_failure means an evidenced
substantive defect during the selected candidate's implementation/execution/
necessary technical verification, including repaired local and recovered internal
defects. Its window starts at automatic selection when the decision completes and ends at candidate completion or the next formal decision. Zero requires actual implementation and complete
coverage; final execute success alone is insufficient. Timeout and
`environment_failure` retain their bound formal execution scopes. R still takes
the maximum valid per-event probability common to every ranked candidate,
never a joint probability or a sum. Probabilities remain finite JSON numbers in
[0,1]; missing, duplicate or invalid values stay Unknown. Original
`wm-r-technical/1` receipts and `wm-r-max-event/1` predictions keep their archived
formal definitions. No old/new labels or numeric feedback are pooled. See [observation and correction counting](episode_closeout.md#implementation-observation-and-correction-counting).

The candidate can cover an `attempt` or a `strategy_window` spanning the declared modelling, execution, repair and verification activities. Window scoring requires a matching task, episode and strategy, ended linked attempts, and `trace.comparable=True`. Read coverage and auxiliary cost when inspecting a window total.

The candidate must describe the actual formal solve, including LP relaxations,
reductions and final optimization when they form one composite strategy.
`execution_constraints` carries solver module/API hints, unknown runtime/license
usability and executor isolation/limits. `joint_problem.task_payload.input_resources`
carries only the explicit input manifest (source, path, bytes, hash and artifact
reference), never the large file bodies. The same content versions bind the
prediction and execution. A source/manifest change is not a reusable forecast.

Current structured receipts distinguish a documented script-contract violation
(implementation failure), unavailable process/isolation/dependency/license
(environment failure), and input/framework rejections with uncertain
responsibility (Unknown). Legacy receipts retain their recorded classifications.
Formal R.timeout still asks whether the bound attempt was interrupted;
`runtime_eligible=false` on a pre-runtime rejection prevents using its zero label
as evidence that an algorithm ran without timing out. See [diagnostics](commands.md#submission-and-runtime-diagnostics).

For C, `cost.retries` under `remaining_to_task_end` predicts additional
corrective/retry rounds actually started from this prediction's decision anchor
through task termination, including later strategy rework. Initial coding,
probes and multiple edits within one round do not add rounds. Continuous formal
failure → local repair → resubmission counts once. The convention is
`wm-retry-corrective-rounds/1`; complete host coverage is required for a numeric
actual, otherwise a lower bound/Unknown is retained. Expected fractions and zero
are valid; there is no 0.1 default or floor. Formal execution increments and
retry permissions remain separate under `or-retry-increment/1`. Archived
`wm-retry-additional-attempts/1` predictions retain their original target.

### H+ and induction

Read the assigned switch in `trace.model_info.hplus`. During library building, close-out schedules a review for an executed, bound positive signal. First generalize to a future task from the Harness's visible published strategic knowledge and ask what concrete decision could improve if the same Agent additionally received the potential rule. Then assess its incremental value (new mechanism, broader scope, correction or stronger evidence) against actual published rule coverage and feasible evidence from this candidate, not novelty to OR research or pretrained knowledge. Missing retrieval is not proof of global absence. Prior executions or forecasts do not establish coverage. The positive switch records that a concrete, checkable increment hypothesis was proposed; it does not assert that knowledge was gained. The review compares the hypothesis, actual evidence and existing rules and may settle as a new rule, a revision, added evidence, already covered, no new knowledge, or insufficient evidence. The receipt keeps the prediction, actual evidence and supported learning result and can inform later forecasts. Read the [induction handoff](induction.md#when-to-start-induction) for what to do with its ID.

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
