# Strategy-outcome prediction service (wm-so/1)

Use this page for scored candidate selection, utility, and execution binding. Follow [SKILL.md](../SKILL.md) for the full solving flow.

- Protocol: `wm-so/1`
- APIs: `ORHarness.predict_strategy_outcome` and `bind_strategy_outcome`
- [Prediction inputs and output fields](world_model_contract.md)
- Runnable examples: [scored CLI](examples/unified_solving.py), [Python forecasts](examples/strategy_outcome.py), and [binding attribution](examples/linkage_attribution.py)

## 1. Plan and select

`plan-next` freezes one shared context and makes one forecast per candidate, including when only one is proposed. It records the score and permission. Stable ties preserve proposal order.

Read `result.selected_prediction_id`, `result.selected_score`, and the candidates' `prediction_status`, `execution_allowed`, `execution_refusal_reason`, and `score`. If no candidate is selected, use `result.plan.status`, `truncation_reason`, and the refusal details to adjust the proposal or configuration.

Use the selected P in `execute --prediction P`. It claims the scored prediction and records adoption before starting the solver. A repair requires a fresh decision on the same task and episode. See [the H+ review handoff](induction.md#when-to-start-induction) for the separate induction decision.

## 2. Propose candidates and inspect saved forecasts

```bash
orx --world-model URL::MODEL plan-next --task t.json --episode ep1 \
  --candidates '[{"name":"joint assignment","steps":["Build the coupled model","Solve and check original constraints"],"solver":"highs","config":{"time_limit":60}}]'
# P = result.selected_prediction_id
orx inspect --bank predictions --prediction P
```

The same candidate list format handles one or more proposals. Inspection reads
the saved forecast without a new model call. The Python service
`ORHarness.predict_strategy_outcome()` remains the planner's prediction entry;
advanced Python callers can reuse an explicit frozen context. Independent API
forecasts remain readable and need a scored plan before formal execution.

**Candidates.** A candidate is a `CandidateRef` (or a legacy `ActionSpec`, whose execution params and budget hint are preserved verbatim and whose unmappable scopes are refused). A candidate may cover modelling / decomposition / solving / repair — it is not just one solver call. The service does NOT generate candidates — they come from YOU (there is no built-in strategy directory to enumerate, and `recall` is a report on real memory rather than a menu) — and it never generates solve.py.

**Candidate identity.** The same `strategy_id` under a different solver, `time_limit`, `mip_gap`, `seed` or step scope is a DIFFERENT candidate: the config travels with the candidate into the prediction, the score and selection (a contradictory execution config is refused) and the execution association, so a result is never attributed to a configuration that was not the one predicted.

The provider receives the frozen context, the candidate, and the decision's benefit convention. In `benefit.notes`, connect a decisive task requirement to the proposed step and its expected consequence. For example, choosing factories separately for two products can violate an original requirement to produce them at the same factory. When numeric G is unavailable, retain the explanation.

G (`wm-g-faithful/2`) predicts effective completion and sufficient verification
of the original task by the current Agent and Harness, given faithful execution.
Understanding, modelling, solving and verification limitations affect G. An
evaluable faithful run that fails or lacks sufficient credible verification is
G=0, including when the cause is unknown; G=1 needs verified original-task
completion. Unknown denotes missing opportunity, fidelity or key evidence,
external disruption or a defective evaluation standard. Pure technical failure
without an evaluable result belongs to R. `/1` forecasts retain their original
labels and calibration groups. See [G attribution](episode_closeout.md#g-attribution-evidence).

Use `trace.unsupported_fields` and the saved call diagnostics to inspect unusable predictions. [Forecast recovery](commands.md#stored-forecasts-and-their-conventions) describes the CLI outputs.

## 3. Score one or more candidates

```bash
orx plan-next --task task.json --episode E --candidates candidates.json \
  --cost-scales '{"tool_calls":30}'
# Read result.selected_prediction_id P and result.selected_score.
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
```

One JSON candidate list works for single and multiple strategies. Horizon is 1;
default bounds are 3 candidates, 6 calls and 120 seconds. Known budget limits,
provider errors and truncation retain their existing rules. There is no universal
`U > 0` threshold. With comparable G the score is `alpha*G - beta*C - gamma*R`;
H+ never changes this score. When no candidate has comparable G there is no
executable selection. Eligible partial U is allowed and records its omissions.

R is `max(p(event))` over the events `implementation_failure`, `timeout` and
`environment_failure` that EVERY participating candidate supplies as a finite
number in [0,1]. It is a maximum single-item risk score, NOT a joint failure
probability. An unknown event is dropped for the whole comparison: when the common
event set is empty (or participants were archived under different R conventions),
R is omitted for everyone. An explicit zero is a supplied value; a duplicate event
or an illegal probability makes that event unknown. Other event probabilities
remain diagnostics. `scoring_basis` reports the participating events, the per-
candidate probabilities, the omitted events with reasons, and the maximum's source.
C first intersects finite nonnegative predictions of the FOUR active
remaining-task dimensions with identical measurement scopes, then removes
dimensions without a reliable independent scale. The active C targets are
`remaining_latency_s`, `output_tokens`, `tool_calls` and `retries`; every one
covers the decision-to-task-end window. Execution-only `latency_s` (one
attempt's sandbox wall-clock), `solver_runtime_s` (the solver's inner runtime)
and raw `llm_tokens` (input plus output) remain recorded observations for
diagnostics/audit, but they are NOT prediction targets, not U terms and not
C-calibration dimensions. Explicit zero is a value; unknown is never imputed as
zero or peak cost.

The token term in U is `output_tokens`: tokens the execution model is expected
to generate from this decision until task termination. The known task text and
its length inform that forecast but are not generated-token cost. Later
modelling, code, self-checks and expected repairs/retries are included. Raw
`llm_tokens` remains input plus output for usage and budgets; old predictions
keep that meaning and cannot supply the new token term. `output_tokens` has
its own weight; an old `llm_tokens` weight does not supply it. Its scale must
come from output-token history or an explicit
`cost_scales.output_tokens`, never a total-token scale.

Each retained dimension uses `clip(log1p(predicted_d)/log1p(peak_d),0,1)`.
Positive `peak_d` comes from explicit `limits.cost_scales` / CLI `--cost-scales`,
or the global pre-decision remaining-task distribution frozen on the context (the p90
of each dimension in `history_retrieval.global.c`). That population is the same
for every participating candidate and uses no phase classification. The current
candidate costs and current planning spend never set the peak. A scalar explicit peak is a declared resource scale; a structured
one can declare scope and a source reference, for example
`{"tool_calls":{"peak":30,"scope":"remaining_to_task_end","reference":"training-v1"}}`.
History provenance records its observation version, support and source IDs.
An unavailable, invalid, nonpositive or mismatched scale removes that dimension
and explains why. Existing positive dimension weights are normalized over the
retained set. Scores preserve raw resources, normalized C, scales, sources,
weights, actual terms and omissions. Candidate costs do not change another
candidate's same-dimension normalized value.

### C resource weights and explicit overrides

The active default is `DEFAULT_PREDICTED_COST_WEIGHTS` in `core/schema.py`.
These are **provisional engineering resource preferences**, not weights fitted
to execution data and not an experimentally optimal configuration. Remaining
time and generated tokens receive the largest shares; tool calls retain an
independent charge, and retries retain a smaller share because their resource
use can also appear in the other dimensions.

| Dimension | Previous raw weight | Current raw weight | Current share with all four valid |
|---|---:|---:|---:|
| `remaining_latency_s` | 0.1 | 1.0 | 1/3 |
| `output_tokens` | 1.0 | 1.0 | 1/3 |
| `tool_calls` | 0.2 | 0.5 | 1/6 |
| `retries` | 2.0 | 0.5 | 1/6 |

For the common valid dimension set D, `w_d = raw_weight_d / sum(raw_weight_j
for j in D)` and `C = sum(w_d * normalized_d for d in D)`. C stays in [0,1].
Removing an unknown, invalid, differently scoped, unscaled or nonpositively
weighted dimension removes it for every ranked candidate and renormalizes the
remaining positive weights. No common valid dimensions means `C = null` and
the C term is omitted from U. Clipping and six-decimal output rounding can
produce ties; componentwise lower costs cannot produce a higher C on the same
valid dimensions, scales and nonnegative weights.

Precedence is an explicit `PlanLimits.cost_weights` / `plan_next(limits=...)`
map, then `ORHarness(cost_weights=...)` / global CLI `--cost-weights`, then the
four-dimensional default. Direct `score_strategy_outcome_predictions` calls
use the same default when `PlanLimits.cost_weights is None`. An explicit map
**replaces** the default: unspecified dimensions get no positive weight, a
zero disables that dimension, and an explicit `{}` disables C. No output-token
weight is silently added to a supplied map. These settings change only C's
contribution to U; alpha/beta/gamma and H+ keep their existing meanings.

Before an experiment, check the actual launch script and Python configuration
for old explicit overrides. Remove the override (or use `cost_weights=None`)
to use the current default; `{}` disables C and is not a request for defaults.
For example, `--cost-weights llm_tokens=1,retries=2` still gives retries all of
the positive active weight when retries has valid predictions and a scale.
It does not activate the new four-dimensional preference. A map containing
only `llm_tokens`, `solver_runtime_s` or `latency_s` gives no active C weight.

Such legacy keys now produce a **nonblocking**
`scoring_basis.cost_weight_diagnostic` and a note, also shown once in the CLI
summary. It names ignored legacy keys, positive active **configured**
dimensions and active dimensions without a positive configured weight. It
does not migrate, merge, reject or replace the supplied map. Intentional
active-only partial maps and explicit empty maps keep their existing behavior.

For example, switch between the default and the previous preference using the
existing CLI option (put global options before `plan-next`):

```bash
orx --cost-weights remaining_latency_s=1,output_tokens=1,tool_calls=0.5,retries=0.5 \
  plan-next --task task.json --episode E --candidates candidates.json
orx --cost-weights remaining_latency_s=0.1,output_tokens=1,tool_calls=0.2,retries=2 \
  plan-next --task task.json --episode E --candidates candidates.json
```

Legacy `DEFAULT_COST_WEIGHTS`, `CostVector.scalarize()` and the selector's
legacy cost-reporting defaults are unchanged. The active planner inherits
`ORHarness.cost_weights` independently of `Selector.cost_weights`. Historical
prediction/observation formats and measurements are unchanged; no old records
are rewritten by changing a preference.

### Scoring audit and synthetic arithmetic checks

Each `candidates[].score` and `selected_score` carries the following audit.
The CLI nests candidates under `result.plan`; the Python plan returns them
directly. `selected_prediction_id` and `execution_allowed` identify the chosen
candidate, while `execution_refusal_reason` explains the others.

| Field in a score | Meaning |
|---|---|
| `cost_reported.values` | Raw declared active C predictions; absent/invalid dimensions remain unknown |
| `scoring_basis.cost_dimensions` | Common valid dimensions actually charged |
| `scoring_basis.cost_normalized_values` | Per-dimension clipped log1p values, at full calculation precision; omitted dimensions have no value |
| `scoring_basis.cost_scales`, `cost_scale_sources` | Peaks and their explicit or pre-decision provenance |
| `scoring_basis.cost_weights` | Positive weights renormalized over the charged dimensions |
| `scoring_basis.omitted_cost_dimensions`, `incomparable` | Reasons that reported dimensions or whole terms could not be used |
| `scoring_basis.cost_weight_diagnostic`, `notes` | Nonblocking migration warning when legacy weight keys are present; configured dimensions are distinct from charged dimensions |
| `cost_normalized` | Aggregated C, rounded to six decimals; null when unavailable |
| `benefit_value`, `risk_effective`, `scoring_basis.utility_weights`, `utility` | G, R, alpha/beta/gamma and final U |

Recompute C from the full-precision normalized values and weights. Recompute U
from that **unrounded** C and the reported G/R and external coefficients,
omitting unavailable terms; round C and U separately to six decimals at the
end. No new persistent log or database schema is needed.

For each decision in the experiment report, retain `selected_prediction_id`,
the actual `plan.limits.cost_weights` and each candidate's existing score
audit. Record the effective `cost_dimensions`, `omitted_cost_dimensions`,
`cost_weights`, `cost_scale_sources` and whether `terms` contains C. A retained
scale with `source=pre_decision_measured_history` uses the frozen history p90;
`source=explicit_plan_limits` uses an explicit scale. Inspect these sources
per dimension, since one decision can mix them.

An empty history with no explicit scales gives `cost_dimensions=[]`,
`cost_scale_sources={}`, `cost_weights={}` and `cost_normalized=null`; C is
absent from `terms` and U uses the remaining comparable terms. Even positive
configured weights cannot affect that decision. With only some valid scales,
the effective dimensions and renormalized weights can differ from the full
default profile. Report this coverage before attributing any early-task
selection to the new weights. No cold-start scale or candidate-derived peak
is introduced.

The following are **synthetic fixtures only**, from
`tests/harness/test_c_aggregation_weights.py`. Fixed peaks in the dimension
order above are `(99, 9999, 15, 3)`; G=0.8, R=0.2 and alpha=beta=gamma=1.
These deliberately constructed resource combinations test arithmetic, not
whether any real solving workflow has these costs.

| Synthetic candidate | Raw four-dimensional costs | Normalized costs | Previous C | Current C | Previous U | Current U |
|---|---|---|---:|---:|---:|---:|
| A | `(9, 99, 3, 0)` | `(0.5, 0.5, 0.5, 0)` | 0.196970 | 0.416667 | 0.403030 | 0.183333 |
| B | `(0, 0, 0, 3)` | `(0, 0, 0, 1)` | 0.606061 | 0.166667 | -0.006061 | 0.433333 |

The previous preference selects A; the current preference selects B. Neither
dominates the other in all dimensions. With beta=0 both have U=0.6 under
either preference, and the existing stable tie rule applies. Missing retries
on one candidate removes retries for both and gives the current remaining
weights `(0.4, 0.4, 0.2)`; absent scales omit C rather than deriving peaks from
these predictions. Tests also cover per-dimension monotonicity, dominance,
explicit empty/partial/zero overrides, single/multiple candidate agreement,
audit recomputation and unchanged legacy reads.

Real data must still validate resource preferences, sensitivity of selection
to weights/scales and overlap among retries, time, tokens and tool calls. The
synthetic checks establish no success-rate or measured resource advantage;
the separate experiment environment performs the real-data and eight-task
acceptance work.

`result.selected_prediction_id` and `result.selected_score` are nullable.
`result.plan.candidates[]` includes each original P, score, frozen candidate
reference, `execution_allowed` and refusal reason. The completed selection action
stores the same audit. `execute` checks ownership, task content, candidate,
method, solver, config, finite U, selection and unused consumption. Bare strategy
arguments, independent predictions, manual `choose-next --chosen`, or later
`bind-strategy` cannot grant permission. `choose-next --rejected` remains useful
for recording a declined plan; compatibility acceptance may reference only the
winner and is reused at actual adoption.

Runnable CLI checks with an explicit local substitute provider:

```bash
PYTHONPATH=src python3 references/examples/unified_solving.py --mode single
PYTHONPATH=src python3 references/examples/unified_solving.py --mode multi
```

These verify one-call-per-candidate wiring, U, actual adoption and H+ receipts.
Predictor discrimination still requires comparable attributable real executions.

## 4. Budgets, stops, and archiving an interrupted run

**A declared budget caps SOLVING, and the stop is the HOST's.** `orx budget --task T [--episode E] --declare 'dim=value,...'` declares (and replaces) a task/episode budget. It is a cap over real consumption, not a promise: the framework REFUSES to START new solving work once a budget is spent, but it does not itself kill a running process.

**Dimensions and scope.** OBSERVATION dimensions are `llm_tokens`, `output_tokens`, `tool_calls`, `solver_runtime_s`, `retries`, `latency_s`, `remaining_latency_s`. A wm-so/1 C PREDICTION may name only the four active remaining-task targets: `remaining_latency_s`, `output_tokens`, `tool_calls`, `retries` (contract marker `wm-c-predict/4`). Raw attempt usage accumulates over the EPISODE (recorded + staged executions, own-cost actions, unparented prediction calls, deduplicated by execution id). `llm_tokens` retains the input-plus-output budget total. The two WALL-CLOCK dimensions are NEVER summed — attempts may overlap, and overlapping prediction ranges would double-count the same span — so they are reported per item. Full decision-window reports are observations, not extra attempt charges.

**Generated-token observations.** C predicts `cost.output_tokens`; saved
forecasts expose it under `cost.expected.output_tokens` and declare it in
`expected_measured`. Omit an unknown value (or use JSON `null` with a reason in
`unsupported_fields`); an explicit zero remains valid. For example, a trusted
complete same-window host report with 10,000 `prompt_tokens` and 400
`completion_tokens` gives C an actual `output_tokens=400`, while raw usage and
the `llm_tokens` budget still count 10,400. Reasoning tokens already included
in completion are counted once. The first decision includes later retries;
a retry starts a new window at its own decision anchor. Complete timestamped
call ledgers are sliced at those anchors and deduplicated by call ID; same-window
counter deltas can also establish output. Missing completion counts, partial
attempt totals or unclear boundaries leave output unknown. A raw total alone
does not establish output. Older trustworthy same-window reports can reuse
their recorded completion sub-fact without rewriting historical data.

**The PRIMARY latency is `remaining_latency_s`.** It measures the wall-clock from the DECISION's anchor to the END of the task, covering the later modelling, coding, tool calls, solving, checking, repair and retries. Every candidate of one decision shares ONE anchor (the decision point) and ONE end (the end of the task) — stated in the request's `cost_scope` block. `solver_runtime_s` (the solver's own inner runtime) and attempt `latency_s` (one sandbox attempt) are EXECUTION OBSERVATIONS, not C prediction targets: they are never renamed into the remaining span, never added to it, and a new prediction containing either is rejected. When no real start/end boundary is available the framework leaves `remaining_latency_s` UNKNOWN and says so — it never downgrades to the script-execution cost.

**C evidence (`wm-c-history/1`).** `prediction_context.cost_evidence` carries LOCAL similar historical executions recalled through the shared embedding retrieval (up to 10, fewer when fewer pass the similarity floor) with the four actual remaining-task dimensions, a `missing_dimensions` list, the observed scope/window and source references, plus a `global_cost` block with the pre-decision n/p50/p90 per dimension, prediction-vs-actual error direction and rank correlation. The ONE hard applicability constraint is the measurement window: every case covers the same remaining task, never one sandbox attempt or an inner solver runtime. There is NO phase classification: `current_conditions.phase_used` is false, and a current model artifact is ordinary context in `reusable_artifacts`. The ONLY hard exclusion is an incomparable measurement window (different scope or incomplete coverage). Strict means the same known family and NO known structure/solver/config difference; a same-scope case with any known difference (a different solver such as HiGHS vs CBC, a different config, a different structural field) or an unknown condition is `broader_reference` with explicit `differences`/`unknown_conditions` and a lower `relevance_weight` — it is NOT discarded. Missing dimensions stay missing, never zero. `history_retrieval` records the embedding backend, the per-view similarity floors, admission counts and any degradation. The complete audit population and its condition groups stay in the published summary under `conditional_cost`.

First compare the candidate steps against the reference workflow: what model can be reused, what must be rebuilt, and what additional subproblem solves, merges, acceptance checks or evidenced repairs occur. Estimate effects in raw resource units, using generated-token evidence for `output_tokens`; do not add steps already included in a historical total. Close estimates are valid when supported. Cite range/case references in `evidence_basis`; use `unsupported_fields` for unavailable dimensions. `cost` contains resource fields only, and `cost.expected`/`expected_measured` retain their declaration semantics. Host rounds explain costs but are not a new cost dimension. Neither global nor single-case C residuals generate directional instructions; old cached C reminders are filtered at both provider boundaries.

**Retry counts.** Each real execution records only its own retry increment
in `cost.retries`: the first attempt of a task/episode adds 0, each later
attempt adds 1. Switching strategy, solver or candidate does not reset that
chain. Failed and interrupted attempts count; replanning and rechecking an
existing execution do not. Recording the same execution again does not add
another attempt.

In a current `remaining_to_task_end` forecast, `cost.retries` predicts the
expected number of additional corrective/retry rounds started from that
prediction's decision through task termination, including later strategy rework.
Initial implementation is excluded; a failure, local repair and resubmission
in one continuous round count once. Use nonnegative fractions, explicit zero or
Unknown, without a 0.1 default/floor. The request freezes
`wm-retry-corrective-rounds/1` under `cost_scope.retries_convention`; the trace
keeps `cost_observation_rule_versions.retries`. Complete host coverage is needed
for a comparable actual; partial observations retain a lower bound.

Formal execution increments/quotas still use their original 0/1 accounting and
do not count local repair events. Archived `wm-retry-additional-attempts/1`
forecasts keep their original formal-attempt definition without mixing with new
corrective counts. Other C dimensions/normalization/weights stay unchanged.
See [host reports](commands.md#implementation-and-correction-reports). Absolute and
signed errors accept zeros; log errors require positive values on both sides.

**Four honest states** (`orx budget` prints `result.status`):

| State | Meaning |
|---|---|
| `no_budget_declared` | consumption is aggregated but not judged |
| `ok` | every declared dimension measured and within the limit |
| `unconfirmed` | known spend within the limit, but a declared dimension is UNKNOWN — actual remaining budget is NOT confirmed |
| `exceeded` | a measured dimension is over the limit |

`unconfirmed` is **not** "within budget": a declared dimension that no record measured leaves the true remaining budget unknown. Never read it as headroom.

**Two kinds of overrun, two different responses.** The view reports `exceeded_dims` (each with its `scope`) plus two booleans:

- **`episode_exhausted`** — an EPISODE-scoped cumulative dimension is over the limit. This STOPS the task: `execute` and `plan-next` refuse to start new work (they raise `BudgetExhausted` before any action, prediction or sandbox touch).
- **`attempt_limited`** — only `latency_s` on one attempt is over. This is a FAILURE to keep (the attempt stays recorded with its cost) and a RETRY is allowed: one slow attempt must never permanently block the rest of the task.

The gate reads the SCOPE, never the bare `exceeded` status.

**Reaching a cap is a stop even when the number is AT the limit.** A 100-tool-call cap refuses the 101st call, so the recorded count can be exactly 100 — never `> 100`. The stop is therefore carried by the HOST's own reason, recorded as an episode-termination mark; the ledger is not asked to infer it from a numeric comparison.

**The host cancels; the framework archives.** When the host has confirmed the process is gone (it cancelled it, or it exited), it records the stop:

```bash
orx record --session --task T --episode E \
    --reason "tool_calls reached 100" --cancelled \
    [--attempted-execution EX_ID] [--override llm_tokens=...,tool_calls=...]
```

This archives the REAL termination, in one of three honest shapes, never fabricating a solve result:

1. an already-STAGED attempt (named with `--attempted-execution`) is ANNOTATED (`session_interrupted`) and, when the host supplied the real cost (`--override`), that cost is BACKFILLED onto the staged fact as its measured receipt — its observed result is not rewritten. Without the receipt a stopped attempt's real spend is invisible to the budget ledger, so supply it: the numbers are the host's measurement, never fabricated and never split from a session total;
2. a `running` execute action with no staged fact (a kill that landed BEFORE staging) is ended as an honest interruption (`status="failed"`, `outcome.interrupted=true`, result marked `unobservable`) — the cost incurred is kept;
3. NO execute attempt at all (the session spent its budget editing code / calling the world model) is recorded as a non-execution `finish_task` action — no `execute_strategy` is invented.

In every case the reason is the HOST's stated cause, verbatim, and an episode-termination mark is written so a later `execute`/`plan-next` refuses new work for the task. Re-recording the SAME stop is IDEMPOTENT (nothing is written twice).

**Finishing is never gated.** `record`, `check-task`, `record --session`, archiving and `close-episode` do NOT go through the solve gate — the host's finish path must complete even after a stop. After the stop is recorded the interrupted action is no longer `running`, so `close-episode --terminal budget_exhausted` evaluates instead of returning `pending`. The stop does not rewrite observed quality/cost and does not turn a benefit into 0: what could be observed is kept, what could not is marked interrupted/unknown.

**A budget does not replace the solver's/executor's own limits.** The executor's per-attempt wall clock (`timeout_seconds`) and a script's own solver time limit still apply and are separate: one bounds a single attempt's EXECUTION, the other caps the whole solution's SPEND. They are not the same default and must not be copied from one to the other. `latency_s` (what is MEASURED) is likewise not the same as `timeout_seconds` (what is ALLOWED).

The host enforces its session caps. For the OpenClaw deployment, these are 100 tool calls and 1,800 seconds. Follow [session stop reporting](commands.md#6-record-facts-and-cost) after the host confirms a stop.

## 5. Associating the real execution (before it runs)

```bash
# the ordinary flow: the association is established BEFORE the run
orx execute --task t.json --prediction PREDICTION_ID --code ws/solve.py --workspace ws
# Historical binding inspection is described under compatibility below.
```

**The association is established before the execution, and the observation is added after it.** `execute --prediction` reads the FROZEN candidate, checks it, and takes the strategy/solver/episode FROM it — the caller does not re-type what the prediction already fixed: the**problem identity** must match (a changed task/data/CIR is a different problem; writing the `model` after predicting is NOT a change), the prediction's status must be usable, and a prediction already claimed by another attempt is refused (one attempt per selected prediction). A path error, a contradictory explicit argument or a claimed prediction is reported BEFORE anything runs, so no execution evidence or solver spend is produced; the action and its link are persisted before the sandbox starts, so an interrupted execution still shows which attempt the prediction was testing. What the run then adds is the RESULT and the OBSERVED configuration — never a re-interpretation of the candidate.

The action's `execution_config` contains what the executor observed and any script receipt stamped for this attempt. Unreported values appear in `binding_unknown`; reported deviations appear in `binding_mismatch`. The `bound_action_id` identifies the action. `trace.comparable` also requires a complete scope and matching identity. Read [per-field evaluation](episode_closeout.md#4-the-per-field-evaluation) for which comparisons the observations support.

`bind-strategy` supports historical pairings and idempotent binding of an authorized adoption. The scored selection is the execution permission.

## 6. Compatibility

Historical prediction records retain their original scope and metrics. See [historical records](snapshot_lifecycle.md#read-historical-records). `HttpChatProvider` selects its prompt using the request protocol, with `wm-so/1` selecting the current strategy-outcome prompt.

Current executions require a saved `wm-u-decision/1` selection. An interrupted adopted attempt retains its failure evidence and consumes the prediction. Repair by planning again.
