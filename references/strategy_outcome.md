# Strategy-outcome prediction service (wm-so/1)

Use this page for scored candidate selection, utility, and execution binding. Follow [SKILL.md](../SKILL.md) for the full solving flow.

- Protocol: `wm-so/1`
- APIs: `ORHarness.predict_strategy_outcome` and `bind_strategy_outcome`
- [Prediction inputs](prediction_context.md) and [output fields](world_model_contract.md)
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

The provider receives [frozen inputs and a candidate](prediction_context.md), along with the [fixed benefit convention and output contract](world_model_contract.md). In `benefit.notes`, connect a decisive task requirement to the proposed step and its expected consequence. For example, choosing factories separately for two products can violate an original requirement to produce them at the same factory. When numeric G is unavailable, retain the explanation.

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

R reads only finite numeric `implementation_failure.probability` in [0,1].
If any participating candidate lacks it, R is omitted for everyone. Other event
probabilities remain diagnostics. C first intersects finite nonnegative resource
predictions with identical measurement scopes, then removes dimensions without a
reliable independent scale, then performs common component deduplication.
`solver_runtime_s`, attempt `latency_s` and `remaining_latency_s` retain their own
scopes. Explicit zero is a value; unknown is never imputed as zero or peak cost.

Each retained dimension uses `clip(log1p(predicted_d)/log1p(peak_d),0,1)`.
Positive `peak_d` comes from explicit `limits.cost_scales` / CLI `--cost-scales`,
or the maximum trustworthy actual in pre-decision published conditional-cost
groups applicable to every participating candidate's host, phase, structure and
solving conditions. The current candidate costs and current planning spend never
set the peak. A scalar explicit peak is a declared resource scale; a structured
one can declare scope and a source reference, for example
`{"tool_calls":{"peak":30,"scope":"remaining_to_task_end","reference":"training-v1"}}`.
History provenance records its observation version, support and source IDs.
An unavailable, invalid, nonpositive or mismatched scale removes that dimension
and explains why. Existing positive dimension weights are normalized over the
retained set. Scores preserve raw resources, normalized C, scales, sources,
weights, actual terms and omissions. Candidate costs do not change another
candidate's same-dimension normalized value.

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

**Dimensions and scope.** Supported dimensions are `llm_tokens`, `tool_calls`, `solver_runtime_s`, `retries`, `latency_s`, `remaining_latency_s`. The first four accumulate over the EPISODE (recorded + staged executions, own-cost actions, unparented prediction calls, deduplicated by execution id). The two WALL-CLOCK dimensions are NEVER summed — attempts may overlap, and overlapping prediction ranges would double-count the same span — so they are reported per item.

**The PRIMARY latency is `remaining_latency_s` (r12).** It measures the wall-clock from the DECISION's anchor to the END of the task, covering the later modelling, coding, tool calls, solving, checking, repair and retries. Every candidate of one decision shares ONE anchor (the decision point) and ONE end (the end of the task) — stated in the request's `cost_scope` block. `solver_runtime_s` is a COMPONENT of that span (the solver's own inner solve time): it is kept for diagnosis but is NOT added on top of `remaining_latency_s` (the part and the whole are never billed together — see `chargeable_dimensions`). The old attempt-level `latency_s` keeps its own meaning (one attempt's sandbox wall-clock) and is NEVER renamed into the remaining span. When no real start/end boundary is available the framework leaves `remaining_latency_s` UNKNOWN and says so — it never downgrades to the script-execution cost.

**C evidence (`wm-c-window/1`).** `prediction_context.cost_evidence` contains the execution host/model actually recorded before the decision, known modelling phase, reusable artifacts, up to four applicable P50/P75 ranges and three executed cases with raw costs, work facts and source references. Select history before request-size trimming. Host/model, cost window, known phase, family, structure and solving restrictions constrain applicability; unknown conditions are stated. Actual receipt configurations remain separate groups even when candidates omitted limits. Prediction-token consumption is not a grouping key. The full population stays in the published audit summary under `conditional_cost`.

First compare the candidate steps against the reference workflow: what model can be reused, what must be rebuilt, and what additional subproblem solves, merges, acceptance checks or evidenced repairs occur. Estimate effects in raw resource units; do not add steps already included in a historical total. Close estimates are valid when supported. Cite range/case references in `evidence_basis`; use `unsupported_fields` for unavailable dimensions. `cost` still contains only existing numeric dimensions, and `cost.expected`/`expected_measured` retain their declaration semantics. Host rounds explain costs but are not a new cost dimension. Neither global nor single-case C residuals generate directional instructions; old cached C reminders are filtered at both provider boundaries.

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

Historical prediction records retain their original scope and metrics. See [contract migration](world_model_contract.md#7-migration-table) and [stored input lifecycle](snapshot_lifecycle.md). `HttpChatProvider` selects its prompt using the request protocol, with `wm-so/1` selecting the current strategy-outcome prompt.

Current executions require a saved `wm-u-decision/1` selection. An interrupted adopted attempt retains its failure evidence and consumes the prediction. Repair by planning again.
