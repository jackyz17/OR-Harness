# Strategy-outcome prediction service (wm-so/1)

Read this page when you are predicting one candidate, comparing several, or binding a real execution to a prediction — the wm-so/1 loop in detail: the yardstick, the comparison rules and the binding identity. For the frozen input a prediction is conditioned on, see [prediction_context.md](prediction_context.md); for every field and status, [world_model_contract.md](world_model_contract.md).

The solving entry is `plan-next` for one or more candidates. Read existing forecasts with `orx inspect --bank predictions --prediction P`.

- Protocol version: `wm-so/1`
- Python module: `or_harness.world_model.strategy_prediction`
- API: `ORHarness.predict_strategy_outcome` / `bind_strategy_outcome`
- CLI: `plan-next` → `execute --prediction` → `check-task` → `record` → `close-episode` → frozen H+ jobs. `choose-next` is a compatibility/rejection record; later binding cannot authorize solving.
- Tests: `tests/harness/test_strategy_outcome.py`
- Runnable example: [`references/examples/strategy_outcome.py`](examples/strategy_outcome.py)
- Bounded attribution (a missing receipt is a caveat, not a discard): [`references/examples/linkage_attribution.py`](examples/linkage_attribution.py)

The INPUT side (what a prediction is conditioned on) is documented in [`references/prediction_context.md`](prediction_context.md); the CONTRACT (the objects a prediction parses into) in [`references/world_model_contract.md`](world_model_contract.md). This document is the authority for the SERVICE: the protocol, the decision loop and the execution binding.

---

## 1. Normal solving loop

```
profile / recall → propose one or more candidates → plan-next
  → execute --prediction P → check-task → record --from-staged EX
  → close-episode → consume induction_schedule.pending_ids
```

The framework freezes one shared context and makes one prediction per candidate.
It saves G/C/R/H+, finite U, the winning prediction ID and its permission in the
plan action. Stable ties preserve proposal order. Single candidates use the same
score calculation. The next execute call uses the original selected P, records
one actual adoption and atomically claims it before starting the solver. It makes
no new forecast. Invalid/absent U, losing candidates, shadow decisions and rejected
plans do not authorize execution or create actual outcomes or induction jobs.

The outer agent writes and checks the chosen method. A failed attempt is recorded;
a repaired attempt needs a fresh `plan-next`. H+ is independent of U. Only actual
bound attempts with frozen H+=1 schedule a review, including failed attempts.
Unexecuted forecasts stay unobserved. Close-out publishes factual feedback; the
outer building agent settles the pending IDs before the next training task.

## 2. Propose candidates and inspect saved forecasts

```bash
orx plan-next --task t.json --episode ep1 \
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

**What the provider receives.** The full frozen context content — the joint problem representation (text, math attributes with origins; **CIR is internal and never sent**), the retrieval evidence (hits with their content, not ids), the capability evidence, the execution constraints — plus the candidate, an explicit output contract AND a `benefit_convention` block. The framework fixes the task, the candidate, the scope, the evidence sources AND the benefit convention; the model fills prediction content only.

**Default G contract.** Before the call, the framework fixes `kind=effective_completion`, `metric=task_check_passed_given_faithful_execution`, `unit=probability`, `observation_rule_version=wm-g-faithful/1` and the candidate's execution scope. G is the probability of passing the original task's acceptance under the agreed solving conditions, GIVEN faithful implementation of the candidate steps. Constraints omitted by the method remain omitted; whether the script can be implemented successfully belongs to implementation risk. Explicit registered legacy conventions remain available; saved forecasts retain their own metric.

The response must use the required convention. A different metric/unit/scope or a non-finite/out-of-range G makes **G unavailable** (`trace.unsupported_fields.benefit`), preserving valid C/R and `benefit.notes`. No rejected number is converted to the required metric. JSON booleans and numeric strings are not G numbers. In this same call, `benefit.notes` carries one grounded sentence linking the decisive requirement, candidate steps and expected consequence, or naming the missing condition. Omit `value` when unsupported; keep the sentence. The main agent decides how to revise.

> 原题要求 BE 和 CFC 同厂，候选分别选厂后直接合并，预计可能产生跨厂解。

**Output semantics.**

| Field | Contract object | Rules |
|---|---|---|
| benefit G | `BenefitEstimate` | metric, unit and baseline required with a value; `solution_quality` must be a NORMALIZED value in [0,1] (e.g. `1-gap`) — a raw objective value is refused, never clamped. Business production/transport cost is OR objective quality, NOT a harness resource cost. An optional `interval` states WHAT it is about via `interval_kind` (`outcome` = ONE run's value, `mean` = the average over runs) and an optional `interval_coverage` (the nominal level claimed). A completion PROBABILITY is a mean over runs — an `effective_completion` interval with no kind is FLAGGED, never read as a single 0/1 label's interval. Omitting the kind leaves the range's meaning unstated (legacy) and the close-out groups it under `(unstated)` |
| cost c | `ExpectedCost` / `CostVector` | the five resource dimensions; the measured mask marks PREDICTED dimensions (a prediction, not a measurement); an omitted dimension is unknown, never zero |
| risk R | `RiskStatement` / `RiskEvent` | only `implementation_failure.probability` is scored: one attempt's technical failure probability under the actual execution agent, solver interface and running conditions; other events remain diagnostics |
| uncertainty | `UncertaintyStatement` | execution randomness vs knowledge gap; a model self-report is recorded as `model_self_report` with the numbers in its notes, explicitly UNCALIBRATED — never relabeled as measured |
| capability gain H+ | `CapabilityGain` | `capability_gain.decision` is the JSON number 0 or 1, the frozen induction switch (`wm-hplus-binary/1`). With 1, provide one learning hypothesis in `claim` and one concrete `verification_conditions` check; with 0, one reason in `basis`. Reuse `applies_to`, per-metric `expected_changes` and evidence fields. Missing/illegal values schedule 0 with separate diagnostics; valid 1 still requires review if explanatory detail is missing. Legacy `assessment` retains its old meaning and does not supply the new switch. H+ never enters strategy utility |
| trace | `PredictionTrace` | context/candidate/task/episode binding, protocol and prompt version, evidence refs, unsupported fields, call cost, error reasons |

**H+ decides induction investment.** The model sees the candidate, related executions,
current rule short views and applicable settled feedback in this SAME call.
It asks what reusable increment/boundary the current library lacks, which fact
this execution can check, and which later step would change. A known method's
structural adaptation or verifiable failure boundary can justify 1; already
covered, duplicate or insufficiently grounded work gets 0 with a reason.
G/C/R comparison is unchanged.

The framework freezes the authoritative switch in `trace.model_info.hplus`,
with the original prediction and input references. Model-supplied verification
flags cannot establish a fact. On selection, the solving agent receives the
check. `close-episode` creates a unique durable job for each bound executed 1;
material and writes require that job. A failure or missing check cannot
retrospectively cancel it. A short submission or an empty review receipt
completes the job, linking the actual induction action and exact rule versions.
Solving alone is never a maintenance binding. See [hplus.md](hplus.md).

`hplus_feedback` contains a few relevant pre-decision settled training receipts.
Independent validation of a substantive increment confirms a positive; a
sourced sufficient no-increment comparison confirms a negative; unavailable,
unexecuted or unvalidated evidence remains unknown. Hypothesis truth and rule
increment are separate. Legacy stance/effect records remain readable under
their original protocol, without mapping `expected` to 1 or silence to an
explicit 0. The legacy performance-effect evaluator remains distinct from
these knowledge-increment labels.

**Honest failure states** — each distinguishable, each persisted with whatever usage the call consumed:

| Situation | Result |
|---|---|
| No provider configured | `contract_only`, `provider_configured=false` |
| Provider error / timeout | `invalid` with the error; the call cost is kept |
| Empty payload / all fields omitted | a recorded non-prediction (`contract_only`-shaped, "no predicted content") |
| Invalid JSON or malformed contract | `invalid` with the specific problems |
| Invalid `implementation_failure.probability` | R unavailable with a reason; valid G/C remain readable |
| A `solution_quality` value outside [0,1] | `invalid` — never silently clamped |

One provider call per prediction. No retries, no default success values.

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

**The two session caps of this deployment live in the HOST, not here.** This repo provides the gate, the interruption archive and the close-out; the host owns the hard cut and the cancel. For the OpenClaw deployment (100 tool calls, 1800 s session wall clock, and the cancel→record→close sequence) see [`docs/openclaw_budget_deployment.md`](../docs/openclaw_budget_deployment.md).

Unexecuted candidates keep `unexecuted`/`unassociated` semantics: no counterfactual truth is fabricated from the winner's result. After the real execution you may build a NEW current context and re-plan — that updates facts and decision inputs; it is not in-task parameter learning, calibration or knowledge induction. A STOPPED run is a recorded interruption, never a completed outcome: its truth enters calibration only where the field was actually observed.

## 5. Associating the real execution (before it runs)

```bash
# the ordinary flow: the association is established BEFORE the run
orx execute --task t.json --prediction PREDICTION_ID --code ws/solve.py --workspace ws
# Historical binding inspection is described under compatibility below.
```

**The association is established before the execution, and the observation is added after it.** `execute --prediction` reads the FROZEN candidate, checks it, and takes the strategy/solver/episode FROM it — the caller does not re-type what the prediction already fixed: the**problem identity** must match (a changed task/data/CIR is a different problem; writing the `model` after predicting is NOT a change), the prediction's status must be usable, and a prediction already claimed by another attempt is refused (one attempt per selected prediction). A path error, a contradictory explicit argument or a claimed prediction is reported BEFORE anything runs, so no execution evidence or solver spend is produced; the action and its link are persisted before the sandbox starts, so an interrupted execution still shows which attempt the prediction was testing. What the run then adds is the RESULT and the OBSERVED configuration — never a re-interpretation of the candidate.

**A missing receipt is a caveat, not a discard.** The config comparison reads the configuration that really took effect, from the action's own `execution_config` (the executor's report plus the solve script's `config` receipt, accepted only when stamped with this attempt's action id so a leftover `result.json` is never read as this run's config). A key the run never reported is recorded under `binding_unknown` — separately from a known `binding_mismatch` — but it does NOT erase the sample: the close-out blocks only the comparison the problem really invalidates. An unreported knob — effort (`time_limit`, `seed`) or approach (`method`, `relaxation`) — blocks nothing: it is reported as a caveat, and identity is decided by the structured fields, not by a knob the run failed to echo. A different solver or a DIFFERENT REPORTED config value blocks the outcome dimensions and keeps the cost; only a wrong task, a wrong strategy, a wrong round or a post-hoc prediction excludes the sample entirely. The per-dimension rule is one small framework table (`world_model/attribution.py`), never a dependency engine over field names.

**`bound` is not `calibratable`.** Being associated (`bound_action_id` set) says the prediction is linked to a real action. Whether ANYTHING may be calibrated is a separate question the CLOSE-OUT answers, field by field; the two are never conflated. `trace.comparable` becomes true only when the associated action is a completed real scope AND no mismatch exists (an attempt-scope prediction needs a completed linked execution; a window-scope prediction needs the window itself complete).

`bind-strategy` retains historical pairing and idempotent re-binding of an already authorized adoption. For new predictions, the action must carry the original scored adoption. It cannot turn an independent or unselected forecast into execution permission.

**What is deliberately out of scope here:**

- capability-evolution prediction, offline learning decisions and their effect verification — see [`references/world_model_contract.md`](world_model_contract.md) and the `*-capability` commands in [`references/commands.md`](commands.md);
- evaluating strategy discrimination on comparable attributable executions. Common-term scores expose missing evidence; the close-out's field eligibility remains independent of utility and suggestion.

The window-level error aggregation and the episode close-out are **implemented**: after the real execution, `orx close-episode` evaluates every ASSOCIATED prediction field by field and publishes the experience calibration — see [`references/episode_closeout.md`](episode_closeout.md).

Unexecuted candidates keep `unexecuted`/`unassociated` semantics: no counterfactual truth is fabricated from the winner's result. After the real execution you may build a NEW current context and re-plan — that updates facts and decision inputs; it is not in-task parameter learning, calibration or knowledge induction.

## 6. Compatibility

- The legacy `predict_outcome` Python API is kept for READING existing records (the contract reader, the record-time knowledge verdicts, `inspect --bank predictions`). No agent-facing entry point uses it, and the horizon=2 rollout is gone.
- Old `OutcomePrediction` records remain readable; the new predictions live in their own `contract_predictions` log table (created idempotently, no schema-version move).
- The `capability_evolution` kind has its own contract AND service (`wm-ce/1`): `predict-capability` / `compare-capability` / `accept-capability` / `bind-capability` / `evaluate-capability`. Its records live in a SEPARATE table, so neither generation is ever read as the other.
- `HttpChatProvider` selects the system prompt by the REQUEST's protocol: a `wm-so/1` request gets the strategy-outcome prompt; everything else keeps the legacy prompts. The wire format stays the OpenAI chat shape.


Prediction input storage and compatibility are defined in [snapshot_lifecycle.md](snapshot_lifecycle.md). New predictions retain identity, real/hypothetical source, frozen budget conditions, actual selection/version references, prompt version and request/omission diagnostics across success, failure and interruption. Binding and budget evaluation do not require a full snapshot. Individual evaluations and published corrections remain independently auditable; model input uses correction counts/field summaries instead of full correction details. H+ availability remains separate from actual adoption and effectiveness.

Historical records retain their original conventions and bindings. New predictions
require the saved `wm-u-decision/1` selection; a post-hoc binding cannot replace it.
An interrupted adopted attempt stays consumed and retains its failure evidence;
retry with a fresh decision. Large frozen evaluation protocol changes are outside
this workflow change.
