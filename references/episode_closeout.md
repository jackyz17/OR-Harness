# Episode close-out and experience calibration

Use this reference to close a task, read the resulting evaluations, and interpret the evidence published for later predictions. For planning and prediction inputs, see [scored strategy selection](strategy_outcome.md) and the [world-model contract](world_model_contract.md).

## 1. Units of observation

| Unit | What it covers |
|---|---|
| Attempt | One recorded `execute_strategy` action and its execution. The script may make multiple solver calls. |
| Strategy execution window | One selection round, including its modeling, attempts, repairs, and checks. Each selection gets a `round_index`. |
| Episode | The task interaction through its terminal state. |

## 2. Closing an episode

```bash
orx close-episode --task t1 --episode ep1 [--terminal completed|failed|aborted|budget_exhausted] \
    [--finish-action ACTION_ID] [--min-samples N]
```

Finish [task-derived checks and retries](../SKILL.md#3b-check-the-original-task-and-decide-whether-to-retry), record each attempt, and commit the answer. Pass the actual terminal state. Use `--finish-action` to link the recorded `finish_task` event and its end time. In benchmarks, the evaluator scores the committed answer against the reference.

Close-out evaluates bound predictions, publishes calibration, and records rule use. Read `task_checks` for counts by verdict and for unchecked attempts. If the result is `state="pending"` with `closeout: null` (exit 2), finish the listed `unfinished_actions` and retry. Repeated close calls return the saved record.

During library building, take the review IDs from `result.induction_schedule.pending_ids` and follow the [induction handoff](induction.md#take-a-required-review).

## 3. The real outcome summary

The action log and execution records supply the observed results for each bound prediction.

- **Benefit** uses the prediction's declared metric, unit, and scope. A strategy window uses the last qualified in-scope attempt for its solution observation. The original-task check is reported under `evaluations[].benefit.task_check` alongside solver quality.
- **Cost** uses `ExpectedCost.scope`. Add cumulative resource measurements once per attempt, including failed attempts. New remaining-scope `retries` counts additional attempts after the bound baseline, rather than summing execution increments. Read `n_measured` and `complete` before comparing a value.
- **Risk** uses the observation unit for each [risk event](#the-risk-event-vocabulary-wm-events3). Technical R reads the specific bound attempt.

| Cost scope | Measurements |
|---|---|
| `attempt` | The bound execution. |
| `strategy_window` | The matching selection round. Two in-scope executions with 11 and 17 seconds of solver time contribute 28 seconds. |
| `remaining_to_task_end` (default) | The shared `select_strategy.started_at` anchor through task termination. Include measured modeling, checks, failed attempts, and repairs in that span. |

For `remaining_latency_s`, use a matching explicit span when available, or derive it from the decision anchor and `finish_task.ended_at`. The last in-scope execute or verify action supplies a lower bound (`basis="last_action_not_task_end"`). Without a measured span or usable timestamps, the value is unknown. Compare one explicit wall-clock span or consistent repeats. Keep distinct spans separate.

## 4. The per-field evaluation

For new `retries` forecasts (`wm-retry-additional-attempts/1`), three executions
record increments 0/1/1, but predictions before A, B and C observe future
additional attempts 2/1/0. The episode retry budget totals 2. The real bound
execution establishes each baseline, even when strategies or solvers change.
A confirmed episode close-out or task termination seals the discrete attempt
chain; it does not establish an end-to-end wall-clock duration. Missing
receipts, unresolved chronology or an unconfirmed ending leave this count
unknown. Failures and staged interrupted attempts remain in the chain;
duplicate records and checks do not add attempts. Legacy forecasts keep their
original convention and are excluded from the new retry reference ranges.
`predicted=0, actual=1` has absolute and signed error 1; `0` versus `0` has
error 0. Both are evaluable, with no manufactured log-ratio.

| Field | Evidence for comparison |
|---|---|
| Solver quality (`solution_quality`) | The same measured metric, unit, and scope. A feasible solution without a measured gap or bound stays `unverified`. |
| Faithful-execution benefit (G) | In-scope faithful execution and original-task acceptance → 1. Faithful execution and confirmed failure caused by the candidate method → 0. Missing attribution or acceptance evidence → unknown. See [G attribution](#g-attribution-evidence). |
| Completion (`task_result_check_passed`) | The execution's own confirmed `passed` or `failed` verdict → 1 or 0. In attempt scope, an attempt that ran and failed without a usable result also supplies 0 (`confirmed_failed_no_result`). Otherwise an unchecked or insufficient result stays unobserved. A window is evaluated at its end. |
| Cost | A complete measured value for the predicted dimension and scope. Estimates and one-sided token counts stay out of the measured total. See [C observations](#c-observations). |
| Risk | The original probability and an attributable label for the same event and observation rule yield a Brier score. |
| Interval | An `outcome` interval uses the comparable observed value for coverage and width. `mean` intervals are marked `unobservable` in this build. Read coverage by `interval_by_kind`. |

Read `binding_mismatch` for recorded disagreements and `binding_unknown` for unreported fields. A wrong task, episode, strategy, or round, or a prediction made after the action started, blocks the comparison. A reported solver or config deviation blocks outcome comparisons and keeps measured cost. Missing config receipts and differences in method wording are reported for review. Check the action's `execution_config` for observed values and sources.

If the executor raises before returning a record, close-out has a linked failure fact for the interrupted attempt. Read that failure and its measurements separately from the retry. For each evaluation, `bound_action_id` locates the action, `eligibility` explains which fields can be compared, and `calibratable` is true when at least one field was evaluated. Read `exclusion_reasons` when none can be compared. Running scopes stay `pending`.

## 5. The experience calibration summary

```bash
orx calibration [--min-samples N] [--rebuild]
```

Calibration uses the newest closed task-episodes (default 50, `OR_CALIBRATION_WINDOW`). `groups` are keyed by predicting model identity, benefit metric, unit, scope, and observation-rule version. If comparing different call settings for the same model identity, give the arms distinct model identities or separate `--home` directories.

| Summary fields | How to read them |
|---|---|
| `n_samples`, `n_distinct_episodes`, `correlated_predictions` | Paired samples, independent task-episodes, and repeated comparisons of the same observation. |
| `mean_benefit_abs_error`, `mean_benefit_signed_error` | Magnitude and direction (`actual - predicted`). Positive signed error means underprediction. |
| `mean_cost_log_error`, `mean_cost_log_ratio` | Absolute and signed log ratio by dimension. Positive log ratio means underprediction. |
| `interval_coverage`, `mean_interval_width` | Coverage and width by interval kind. |
| `mean_brier_by_event`, `n_brier_samples_by_event` | Error and paired sample count per event and applicable rule. |
| `mean_predicted_probability`, `scored_occurrence_rate` | Predicted and observed rates over the same scored pairs. |

### Observation rules

The current summary uses `wm-obs/4`. Each evaluation retains its own rule, including `wm-g-faithful/1` or `wm-r-technical/1` where applicable. Read historical groups by the version they report. An absent rule version is grouped under `(unknown)`. For solver quality under `wm-obs/2` and later, `normalized_objective_gap` uses `1.0` for `optimal` or `1 - gap` when observed. Check verdicts travel separately.

### Technical R observations

R before execution is the probability that one attempt will fail at code construction, execution, or result delivery. The only R input to U is one `implementation_failure` event with a finite numeric `probability` in `[0,1]`. Zero is valid. If any scored candidate lacks a valid R, the decision omits R for all participants. See [U scoring](strategy_outcome.md#3-score-one-or-more-candidates).

For each recorded attempt, `execution_features.technical_outcome` holds the executor's receipt for `wm-r-technical/1`. Close-out reads the one bound execution, including when G covers a strategy window. Read `evaluations[].risk.implementation_observation` for its label and reason. An attempt and its repair get separate labels.

| Label | Executor evidence |
|---|---|
| `1` | Confirmed code construction failure, such as a syntax error before launch. A failed attempt with a code/API exception, a result-protocol failure, or an abnormal script/solver exit without evidence of external blockage also supplies 1. For example, `optimal` with a required objective missing is a protocol failure. |
| `0` | The process exits with code 0 and delivers a readable, protocol-valid result with a status other than `error` or `timeout`. A valid `infeasible` result qualifies, as does a valid result delivered after a caught or logged exception. |
| Unknown | No attributable execution, external blockage such as a missing dependency, an ordinary time limit, or insufficient evidence of the cause. |

For an attempt that ran and did not time out, the executor checks for successful delivery before classifying a logged exception. On the failure path, a specific code exception such as `ValueError: invalid network flow indices` supplies code evidence. Timeout, reported infeasibility, and original-task check failure have their own event records.

For a timed-out execution, a sourced cause assessment in `technical_failure_evidence` can supply label 1. Use `kind="code_caused_timeout"`, nonempty `source` and `assessment`, and `evidence_refs` to this execution's `artifacts.submitted_script` plus `artifacts.stderr_clip` or `artifacts.workspace_result`. Each cited artifact must have a saved hash. The assessment supplies the causal judgment and the framework checks its fields and references.

For earlier execution records, the framework can derive technical facts from captured error, result, or protocol evidence. Historical predictions without the technical R convention still use the `error_class == 'model'` event rule for their own comparisons. Current-rule Brier is grouped under `implementation_failure@wm-r-technical/1`.

`technical_r` reports `n_actual_attempts`, `n_known_labels`, `n_reported_probabilities`, and `n_scorable`. Attempt counts cover recorded executions in the closed window, including attempts whose R was omitted or never predicted. Each execution is counted once per task-episode. Brier uses the original probability of each comparable prediction with a known same-rule label. Groups use actual execution-agent identity, solver interface, family, and problem class. Later predictions can receive up to three applicable groups in `implementation_r_evidence`, with counts, source IDs, and available script references. Label and Brier support count independent episodes separately.

### C observations

For `wm-c-window/1`, supply `or-host-usage/1` at attempt scope or declare `scope="remaining_to_task_end"`. Describe the span using `window.anchor_action_id`, `started_at`, `ended_at`, `complete`, and `end_event` where available. A complete remaining window needs the decision anchor and a terminal `finish_task` or a complete host span ending in `finish_task`, `final_failure`, or `confirmed_abort`.

- **Counters**. Supply `counters.start` and `counters.end` for `prompt_tokens`, `completion_tokens`, `tool_calls`, and optional model `calls`. Nonnegative differences measure usage. A missing side or reset leaves the dimension unknown or partial.
- **Call ledger**. Supply `id`, `kind` (`tool`, `model`/`llm`), call timestamps, and model input/output token counts in `call_details`. Use `call_details_complete` to declare complete dimensions such as `tool_calls` and `llm_tokens`. A complete declaration can support an explicit zero for an empty kind. A partial ledger supplies lower bounds. The framework deduplicates call IDs and uses calls inside the measured span. Crossing calls need evidence that allocates usage to the span.
- **Totals**. Same-window aggregates, counter differences, and complete ledgers must agree and cover known execution/call lower bounds. Conflicted or incomplete dimensions stay out of whole-window comparisons. Repeated references to the same report are counted once. Attempt reports retain their own scope. Use solver-reported time for `solver_runtime_s` and a measured span for latency.

`evaluations[].cost.observed_fact` can exist with no forecast of C. Read `conditional_cost.coverage` for actual windows, predictions, error pairs, and positive-positive log pairs. An explicit zero participates in signed and absolute errors, and log ratios require both values to be positive. `cost_evidence` in later requests selects applicable raw ranges and support from comparable execution settings. [C measurement examples](../docs/c_signal_improvement.md)

### Published feedback

New prediction contexts carry the selected publication under `strategy_outcome_calibration`. Read `reminders` for risk and interval diagnostics and `groups` for global benefit and cost errors. See [provider feedback](world_model_contract.md#2-published-prediction-feedback) for request filtering.

`conditional_g` groups sourced observations by metric, rule, scope, structure, model, observed solving conditions, and declared budget. Real execution facts can contribute pass/fail/unknown counts when G was omitted. Each execution unit uses the earliest attributable prediction. Its Brier score and probability bin require both a numeric G and a known label.

`same_task_comparisons` pairs different methods with the same frozen task, acceptance rule, observed solving conditions, scope, and version. Read actual solver and limits from `ExecutionRecord.solver` and `execution_features.execution_config.values/sources`, including recorded executor or action-stamped script receipts. Missing or conflicting settings keep observations as individual cases. The `check-task` report's `acceptance_rule.fingerprint` covers the full declared check, versions, thresholds, and reference provenance. Use it with `acceptance_id` when comparing cases. Two passes tie on binary acceptance, with a `verified_objective` retained when available.

### G attribution evidence

Provide optional `g_attribution` inside `check-task --check` JSON or via `check_task_result` when the execution and checks support the attribution.

```json
{
  "semantic_probe": [{"path":"execution_features.solution_variables.CFC", "equals":1}],
  "g_attribution": {
    "method_fidelity":"faithful",
    "failure_source":"candidate_method",
    "acceptance_scope":"original_task",
    "acceptance_id":"factory-coupling/v1",
    "source":"agent audit of the captured implementation and result",
    "assessment":"The task requires BE and CFC at one factory; the candidate and code select separately, and the recorded assignments violate that condition.",
    "evidence_refs":["artifacts.submitted_script", "artifacts.workspace_result"]
  }
}
```

- Use `method_fidelity=faithful|deviated|unknown`, `failure_source=candidate_method|implementation|environment|unknown` (use `unknown` on a pass), and `acceptance_scope=original_task|partial|unknown`. Name the stable acceptance standard in `acceptance_id` and record the source and grounds of the assessment.
- Cite code evidence (`artifacts.submitted_script` or `solver.code_hash`) and result evidence (`artifacts.workspace_result` or `execution_features.solution_variables`). Inspect `method_actual` and match the recorded plan to the candidate.
- For G, use checks that actually ran and cover the original task. A substantive semantic, domain, or recomputation check can support the label. A self-derived or unsourced reference alone leaves G unknown. Use a trusted failed check and evidence that the faithful method caused the failure to support label 0. The ordinary task-check verdict is recorded in either case.

### Two counting units

- `groups` evaluates eligible prediction-observation pairs. A probability requires a known label to enter Brier or the paired occurrence rate.
- `occurrence` counts actual observations from current window facts, once per execution or once per episode for `budget_exhausted`. Read `n_observation_units`, `n_occurred`, `n_not_occurred`, and `n_unknown`. `unit_occurrence_rate` divides by labelled observation units, including labelled executions without predictions. An excluded execution can still contribute an observed event while leaving the prediction-pair sample.

### The risk-event vocabulary (`wm-events/3`)

| Event | Unit | Evidence |
|---|---|---|
| `implementation_failure` | One bound attempt | [Technical receipt](#technical-r-observations) |
| `environment_failure` | Execution | `error_class == 'environment'` |
| `timeout` | Execution | `status == 'timeout'` |
| `solver_reported_infeasible` | Execution | `status == 'infeasible'` |
| `task_check_failed` | Execution | `task_check.state` |
| `budget_exhausted` | Episode | Ledger from decision anchor to task end |

For `budget_exhausted`, a measured episode limit overrun or an explicit budget-termination receipt supplies `occurred`. A finished episode with all declared episode dimensions measured within limits supplies `not_occurred`. Other cases stay unknown. A change to the budget declaration after prediction makes this event non-comparable.

### Per-statistic thresholds

`--min-samples` (default 5) counts distinct `(task_id, episode_id)` pairs separately for each statistic. Inspect `benefit_evidence`, `cost_evidence[dim]`, `interval_evidence`, and `brier_evidence_by_event[event]` for support. A statistic below the minimum reports `insufficient_evidence`. `under_sampled_statistics` identifies gaps in partially supported groups.

## 6. Retention: three separate scopes

```bash
orx inspect --bank retention
orx archive-calibration [--dry-run]
```

| Scope | What it retains | Default | Setting |
|---|---|---|---|
| Calibration window | Closed task-episodes used as samples | 50 | `OR_CALIBRATION_WINDOW` |
| Late-check grace | Online details for out-of-window episodes with executions awaiting a task verdict | 30 days | `OR_CALIBRATION_LATE_CHECK_GRACE_DAYS` |
| Archive caps | Per-file size, total size, and file age. The oldest files expire first. | 64 MB / 1 GB / 365 days | `OR_CALIBRATION_ARCHIVE_*` |

Archive removes online prediction, evaluation, and context details after writing them to `{home}/archive/calibration/calibration-NNNN.jsonl`. The close-out registry keeps the episode identity and close time for repeat close calls and window selection. Restored archive details are available for audit. The registry still determines the window.

Close-out automatically checks whether archival is due (`OR_CALIBRATION_AUTO_ARCHIVE_THRESHOLD`, default 200). A preview can report `pending_migration` for a store that needs a close-out registry. Use `orx calibration --rebuild` or a real archive pass to backfill it from the stored close times.

### The evidence window (bounding the RAW evidence bank)

```bash
orx enforce-window [--window-episodes 800] [--open-grace-days 30] [--dry-run]
```

| Scope | Default | Setting |
|---|---|---|
| Retained closed episodes of raw execution evidence | 800 | `OR_EVIDENCE_WINDOW_EPISODES` |
| Grace for unclosed episodes | 30 days | `OR_EVIDENCE_OPEN_GRACE_DAYS` |

The evidence pass runs after publication and archive maintenance. It evicts whole episodes, oldest first, along with their vectors and task-text versions no remaining execution uses. Calibration-window episodes, episodes in late-check grace, and unclosed episodes in open grace are protected. Use `--dry-run` to inspect eviction candidates. Published rules and archived evaluations retain their own records when source executions expire.

## 7. Window rounds (the identity extension)

Select one round with `ORHarness.strategy_execution_window(..., round_index=N)` or `win::task::episode::strategy::rN`. The window starts at that selection of the strategy and ends at the next recorded selection of any strategy. A second selection of the same strategy gets a separate window. A nonexistent round is non-comparable. A window ID without a round maps to `round_index=None`.

## 8. Late evidence and corrections

A later `check-task` adds evidence to the execution it cites. Read `validity_corrections[]` for observations re-derived under their original metrics and attribution rules. A late check, exclusion, or restore inside the window can republish calibration (`calibration_republished`). The stored evaluation preserves the close-out facts. Use `orx calibration --rebuild` to regenerate the publication from current records without rerunning the solver or predictor.

Use the check's recorded coverage when interpreting acceptance. Later reference comparisons can supply [offline induction evidence](induction.md#read-material-and-choose-comparisons) within their documented scope. Once evaluation detail is archived, later checks can annotate retained execution facts while the archived evaluation stays final.
