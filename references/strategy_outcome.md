# Strategy-outcome prediction service (wm-so/1)

Read this page when you are predicting one candidate, comparing several, or binding a real execution to a prediction — the wm-so/1 loop in detail: the yardstick, the comparison rules and the binding identity. For the frozen input a prediction is conditioned on, see [prediction_context.md](prediction_context.md); for every field and status, [world_model_contract.md](world_model_contract.md).

**Status: the service is implemented and wired into planning.** This is the third phase of the reconstruction: the unified contract (phase 1) and the frozen prediction input context (phase 2) now carry a real, training-free OR strategy-consequence prediction service.

- Protocol version: `wm-so/1`
- Python module: `or_harness.world_model.strategy_prediction`
- API: `ORHarness.predict_strategy_outcome` / `bind_strategy_outcome`
- CLI: `orx predict-strategy` / `orx plan-next` (the only planning protocol) / `orx choose-next --prediction` / `orx execute --prediction` (associates BEFORE the run) / `orx bind-strategy` (manual recovery)
- Tests: `tests/harness/test_strategy_outcome.py`
- Runnable example: [`references/examples/strategy_outcome.py`](examples/strategy_outcome.py)
- Bounded attribution (a missing receipt is a caveat, not a discard): [`references/examples/linkage_attribution.py`](examples/linkage_attribution.py)

The INPUT side (what a prediction is conditioned on) is documented in [`references/prediction_context.md`](prediction_context.md); the CONTRACT (the objects a prediction parses into) in [`references/world_model_contract.md`](world_model_contract.md). This document is the authority for the SERVICE: the protocol, the decision loop and the execution binding.

---

## 1. The loop this phase implements

```
problem understanding / profile + memory retrieval
        -> ONE frozen PredictionContext
        -> a few candidate strategies
        -> predicted benefit / cost / risk / uncertainty per candidate
        -> compared, one suggested
        -> the outer agent EXPLICITLY chooses (choose-next --prediction <id>)
        -> modelling, solving, verification (the agent's work)
        -> execute --prediction: the association is established BEFORE the run,
           and the real result + observed config are added AFTER it
        -> re-plan from the real observation if warranted
```

- **P is exogenous**: the joint problem representation (math attributes, scenario semantics, structural relations) describes the problem; the service never predicts "the next problem P".
- **X is the solving context the decision needs** — not a full mathematical model, not solve.py, not every internal state.
- **H stays a latent capability**: capability evidence CONDITIONS the prediction; no H score, vector or capability-improvement claim is produced.
- **Training-free**: a frozen, externally configured model fills the prediction content; nothing is fine-tuned, calibrated online, or trained inside a task.
- **One macro comparison, then re-planning from the real observation.** The new protocol plans at horizon=1 only; no strategy-branch trial runs, no snapshot-rollback platform, no factory simulator, no latent-space dynamics. The legacy `plan-next` horizon=2 rollout remains available under the legacy protocol and its old boundaries.

## 2. Predicting one candidate

```bash
# build the frozen input once (optional: predict-strategy builds its own)
orx context --task t.json --episode ep1

# predict ONE candidate under wm-so/1
orx predict-strategy --task t.json --episode ep1 \
  --candidate '{"action_type":"execute_strategy","strategy_id":"S01",
                "solver":"highs","config":{"time_limit":60}}'

# reuse a frozen context (pass the SAME --cir you built it with)
orx predict-strategy --task t.json --episode ep1 --context CTX_ID --cir cir.json \
  --candidate '{"action_type":"execute_strategy","strategy_id":"S02"}'
```

**Candidates.** A candidate is a `CandidateRef` (or a legacy `ActionSpec`, whose execution params and budget hint are preserved verbatim and whose unmappable scopes are refused). A candidate may cover modelling / decomposition / solving / repair — it is not just one solver call. The service does NOT generate candidates — they come from YOU (there is no built-in strategy directory to enumerate, and `recall` is a report on real memory rather than a menu) — and it never generates solve.py.

**Candidate identity.** The same `strategy_id` under a different solver, `time_limit`, `mip_gap`, `seed` or step scope is a DIFFERENT candidate: the config travels with the candidate into the prediction, the choice (a different config is a deviation) and the execution association, so a result is never attributed to a configuration that was not the one predicted.

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

## 3. Comparing candidates and choosing

```bash
orx plan-next --task t.json --episode ep1 \
  [--candidates specs.json] [--max-calls N] \
  [--benefit-kind KIND --benefit-metric METRIC]
```

`plan-next` (horizon fixed at 1):

1. freezes ONE context for the whole decision (every candidate is conditioned on the same problem representation, X/B, retrieval evidence, capability evidence and constraints — one embedding call, one snapshot);
2. predicts each candidate under wm-so/1 — under ONE shared `benefit_convention`, so every candidate is predicted on the SAME currency (a failed prediction does not drag the others down; the model-call count, wall clock and the REAL budget are checked before every call);
3. determines a shared scoring basis from the candidates actually eligible for this ranking, then applies the existing `U = alpha*G - beta*C - gamma*R` weights and cost scales:
   - **G**: the candidate's own usable value under the decision's benefit metric/unit/version. A candidate with unavailable or incompatible G has no utility and does not alter the ranked candidates' common basis;
   - **C**: intersect the candidates' valid `expected_measured` dimensions with matching measurement scopes, then apply cost-component deduplication once on that intersection. Every candidate uses those same dimensions, component weights and existing peak scales. Explicit zero is valid. With no common dimension, mark C incomparable and omit it for the whole comparison. `latency_s` (attempt sandbox) and `remaining_latency_s` (end to end) retain separate meanings;
   - **R**: read only `implementation_failure.probability`. All ranked candidates must supply a finite JSON number in [0,1] to include R. If any lacks it, omit R for all while preserving each original probability. Zero is valid; timeout, environment, budget, infeasibility and checker probabilities never substitute for R;
   - the knowledge term is OFF (delta=0) for this protocol: an old knowledge-gain score is not an H improvement and does not leak into the new comparison;
4. suggests the best candidate's first step. **A suggestion is not a selection**: only `choose-next` writes `X.selected_plan`, and it records accept / deviate / reject explicitly. Accept by ID (`choose-next --decision <id> --prediction <the chosen candidate's id>`): the candidate is read from this decision's own recorded comparison, and the deviation test compares the candidate REFERENCE (action type/strategy/solver/config), never the whole plan JSON.

**Fallback.** When NO candidate carries a usable prediction (provider down, every payload invalid), the plan reports `status="no_valid_predictions"` with the reasons and NO suggestion — fall back to `recall`/Selector ordering or choose yourself. The fallback is reported as what it is; it is never dressed up as a completed world-model comparison, and the calls that did happen keep their recorded cost.

**One convention per decision, resolved and reported.** The comparison reads ONE benefit currency, and where it came from is recorded in `result.plan.benefit_convention` (`kind`/`metric`/`unit`/`source`/`reason`). Three sources, in order:

1. **declared** — the caller names it for this decision: `--benefit-kind` / `--benefit-metric` (optionally `--benefit-unit`). It is then pushed into EVERY candidate's request as a REQUIRED convention, so the candidates are PREDICTED in that currency, not merely compared in it. Declare this to compare **solver quality** instead of completion: `--benefit-kind solution_quality --benefit-metric normalized_objective_gap`.
2. **agreed** — nothing declared, and every candidate carrying a benefit declared the SAME `kind`/`metric`. This is what makes two candidates comparable without ceremony.
3. **legacy fallback** — the existing comparison fallback stays available for old/mixed inputs. New provider requests already have the required faithful-execution convention, so valid new forecasts agree on it; unavailable G never enters utility. Candidate entry, method forwarding, comparison and gating changes are deferred.

A declaration this build has no scale for (`valid_progress`, or an unknown pair) resolves to `source="unknown_declaration"` with `comparable=false`: no ranking is produced. A candidate whose own `kind`/`metric` differs from the decision's convention is reported in `score.incomparable["benefit"]` and produces no utility. Its raw cost and risk remain inspectable.

**Inspect partial scores.** `score.scoring_basis` records the ranked prediction IDs, actual terms, shared cost dimensions/scopes/weights/scales and omission reasons. `risk_probability`, `risk_reported` and `cost_reported` preserve original supported values. A=.50 before risk with R=.20 and B=.50 with unknown R both remain .50 in this comparison. A=500k tokens/120s and B=700k tokens/unknown time both use tokens alone. Missing inputs incur no candidate-specific penalty. Normalization, utility weights and gate thresholds are unchanged.

**R has a fixed observation contract.** Requests and saved traces carry `risk_convention={event:implementation_failure,unit:execution,scope:attempt,observation_rule_version:wm-r-technical/1}`. R concerns constructing solver code, calling APIs, running it and delivering a protocol-valid result in one bound attempt. A code/API exception, abnormal solver exit or invalid delivery is 1; a normal valid delivery, including legal `infeasible`, is 0; external blockage, ordinary limits and insufficient attribution are unknown. Code-caused timeout requires sourced investigation. Business acceptance remains G, and solve/repair spend remains C. Historical predictions without this version retain their original observation rules. See [episode_closeout.md](episode_closeout.md#technical-r-observations).

The same provider call receives `implementation_r_evidence`: a few closed-attempt facts for the actual execution agent and recorded solver interface, including counts and source references. A captured, previously successful script may be cited as available evidence, never as proof that a new candidate will succeed. Shared base rates and similar candidate probabilities are acceptable when supported by these facts.

**Declare the benefit's meaning explicitly.** The `kind`/`metric` pair is the prediction's own yardstick, and the close-out observes the SAME one (the convention in the request says so):

| You are predicting | Declare | Observed from |
|---|---|---|
| whether faithful execution passes original-task acceptance (**new request default**) | `kind=effective_completion`, `metric=task_check_passed_given_faithful_execution`, unit `probability` | sourced fidelity + full declared acceptance: passed=1, confirmed method-caused failure=0, otherwise unknown; `wm-g-faithful/1` |
| legacy task completion | `kind=effective_completion`, `metric=task_result_check_passed` | the execution's own `check-task` verdict — 1.0 `passed`, 0.0 confirmed `failed`, UNKNOWN when unchecked or `insufficient` |
| how well the solver solved the MODEL | `kind=solution_quality`, `metric=normalized_objective_gap` | the solver's gap (an `optimal` status is a gap of 0) |

Candidates compared on completion are compared **with each other**, never against a quality candidate (and the reverse): the convention has to match before two numbers are ranked, because `1 - mip_gap` is not a completion rate and a `passed` check is not a quality level, even though both land in [0,1]. `valid_progress` has no observation channel in this build, so it is reported and never scored — and never quietly replaced by a completion rate.

**Judge new G from the original task, the candidate steps and the expected result.** Faithful execution is an assumption of G, not a forecast that the script will be implemented correctly. Keep implementation risk, feasibility, solver quality and optimality separate. A method that omits a decisive constraint may faithfully produce an invalid answer.

**Observation scope and legacy recovery.** New G uses the terminal in-scope attempt and its sourced method/acceptance assessment. A terminal unknown never borrows an earlier pass. Switching method attributes the later result to the actual method; a later code-only repair may supply a positive example for the original method. The old `task_result_check_passed` adapter retains its last-verdict/recovery and attempt-with-no-result rules; those rules do not apply to new G.

**Feedback.** Global G residuals remain diagnostic. They produce neither aggregate nor single-case G correction instructions, and both cached reminder locations are filtered before provider calls. `conditional_g` reports comparable executed outcomes grouped by metric, observation version, scope, family, structural cell, predictor and solving conditions, with explicit sparse support. Same-task comparisons require the same frozen problem and acceptance standard and actual different methods; one executed method yields one factual case, and two passes tie. See [episode_closeout.md](episode_closeout.md).

**Describe the candidate's `method`.** A candidate carries `{"name": str, "steps": [str, ...], "why": str?, "fallback": str?}` and the model predicts FROM it: a bare strategy id or solver name describes nothing about what would happen. The caller supplies it (a cold start has no memory to read it from), the solver/config stay separate fields, and an empty value means the caller did not describe it — not that there is nothing to describe.

**Failure has ONE status and a structured reason.** A call that produced no usable prediction is `status="invalid"` (there is no separate `provider_error` status), with `trace.model_info["failure"]`:

| `failure.kind` | Evidence |
|---|---|
| `truncated` | `finish_reason` is `length`/`max_tokens` — the endpoint stopped for length |
| `wrong_top_level` | valid JSON whose top level is not an object (the field case of `[]`) |
| `empty_response` | content was `""` or `null` |
| `unparsable` | content is not JSON |
| `timeout` / `network_error` | the call did not complete |
| `unusable_payload` | a JSON object that failed contract validation |

A MISSING `finish_reason` is recorded as `null` with a note saying how the answer ended is UNKNOWN — it is never read as a clean stop. An OLD record written under the retired `provider_error` word still READS: it is normalized to `invalid` and its original status and error are preserved under `trace.model_info["legacy_status"]` / `["error"]`, so no failure is lost and no corrupt record is silently accepted either.

**Call parameters.** Four knobs, each with a flag and an environment variable, and the values actually in force come back under `result.effective_parameters` / `trace.model_info["effective_parameters"]`:

| Knob | Precedence | Default |
|---|---|---|
| Output budget | `--wm-max-tokens` > `$OR_WM_MAX_TOKENS` > adapter default | 2048 |
| Socket timeout | `--wm-timeout` > `$OR_WM_TIMEOUT` > adapter default | 300 |
| JSON-mode hint | `$OR_WM_NO_RESPONSE_FORMAT=1` removes it | sent |
| Thinking | `$OR_WM_ENABLE_THINKING=1` | OFF |

A socket timeout bounds ONE blocking operation, not the whole request, so the caller's own clock check is what bounds the call in aggregate.

**The two wire-format knobs exist because an endpoint's defaults are not ours.** `OR_WM_NO_RESPONSE_FORMAT=1` omits `response_format` from the body (never sends `null`), for endpoints that reject JSON mode. `OR_WM_ENABLE_THINKING=1` omits `enable_thinking`, letting the endpoint's own default apply; unset, the request carries `"enable_thinking": false`. Both are read ONCE at provider construction, so the body and `describe()` cannot disagree about the same call. Measured on one endpoint (Qwen3.8-27B via `llmapi.paratera.com/v1`, same request): thinking ON cost 36.3 s and 1291 reasoning tokens against 4.5 s and 0, and `reasoning_tokens` is counted INSIDE `completion_tokens` (325 total, 291 of them reasoning) — so thinking draws on the SAME output budget as the answer. Which arm a call used is reported in `describe()` and folded into `capability_version.provider`; it is NOT in `trace.model_info`, whose keys stay a fixed identity set.

**Do not mix the two arms in one store.** Calibration groups by `strategy_outcome|<model identity>|<metric>|<unit>|<scope>`, and the model identity is the NAME, not the call configuration — so predictions made with thinking ON and OFF are the SAME group and would be pooled. Keep them in separate `--home` directories (or give them different `--world-model` model names) to compare them.

**Cost accounting.** The model calls are REAL spend: charged once to the decision action as own cost (failed calls included), reported in `planning_cost`, never part of any candidate's utility. Candidate execution costs are PREDICTED values. Missing usage stays unknown — never free.

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

- **`episode_exhausted`** — an EPISODE-scoped cumulative dimension is over the limit. This STOPS the task: `execute` and `predict-strategy` refuse to start new work (they raise `BudgetExhausted` before any action, prediction or sandbox touch).
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

In every case the reason is the HOST's stated cause, verbatim, and an episode-termination mark is written so a later `execute`/`predict-strategy` refuses new work for the task. Re-recording the SAME stop is IDEMPOTENT (nothing is written twice).

**Finishing is never gated.** `record`, `check-task`, `record --session`, archiving and `close-episode` do NOT go through the solve gate — the host's finish path must complete even after a stop. After the stop is recorded the interrupted action is no longer `running`, so `close-episode --terminal budget_exhausted` evaluates instead of returning `pending`. The stop does not rewrite observed quality/cost and does not turn a benefit into 0: what could be observed is kept, what could not is marked interrupted/unknown.

**A budget does not replace the solver's/executor's own limits.** The executor's per-attempt wall clock (`timeout_seconds`) and a script's own solver time limit still apply and are separate: one bounds a single attempt's EXECUTION, the other caps the whole solution's SPEND. They are not the same default and must not be copied from one to the other. `latency_s` (what is MEASURED) is likewise not the same as `timeout_seconds` (what is ALLOWED).

**The two session caps of this deployment live in the HOST, not here.** This repo provides the gate, the interruption archive and the close-out; the host owns the hard cut and the cancel. For the OpenClaw deployment (100 tool calls, 1800 s session wall clock, and the cancel→record→close sequence) see [`docs/openclaw_budget_deployment.md`](../docs/openclaw_budget_deployment.md).

Unexecuted candidates keep `unexecuted`/`unassociated` semantics: no counterfactual truth is fabricated from the winner's result. After the real execution you may build a NEW current context and re-plan — that updates facts and decision inputs; it is not in-task parameter learning, calibration or knowledge induction. A STOPPED run is a recorded interruption, never a completed outcome: its truth enters calibration only where the field was actually observed.

## 5. Associating the real execution (before it runs)

```bash
# the ordinary flow: the association is established BEFORE the run
orx execute --task t.json --prediction PREDICTION_ID --code solve.py --workspace ws
# the manual/recovery path, for an action that already ran
orx bind-strategy --prediction PREDICTION_ID --action ACTION_ID
```

**The association is established before the execution, and the observation is added after it.** `execute --prediction` reads the FROZEN candidate, checks it, and takes the strategy/solver/episode FROM it — the caller does not re-type what the prediction already fixed: the**problem identity** must match (a changed task/data/CIR is a different problem; writing the `model` after predicting is NOT a change), the prediction's status must be usable, and a prediction already claimed by another attempt is refused (one attempt per candidate). A path error, a contradictory explicit argument or a claimed prediction is reported BEFORE anything runs, so the prediction is not spent and no cost is incurred; the action and its link are persisted before the sandbox starts, so an interrupted execution still shows which attempt the prediction was testing. What the run then adds is the RESULT and the OBSERVED configuration — never a re-interpretation of the candidate.

**A missing receipt is a caveat, not a discard.** The config comparison reads the configuration that really took effect, from the action's own `execution_config` (the executor's report plus the solve script's `config` receipt, accepted only when stamped with this attempt's action id so a leftover `result.json` is never read as this run's config). A key the run never reported is recorded under `binding_unknown` — separately from a known `binding_mismatch` — but it does NOT erase the sample: the close-out blocks only the comparison the problem really invalidates. An unreported knob — effort (`time_limit`, `seed`) or approach (`method`, `relaxation`) — blocks nothing: it is reported as a caveat, and identity is decided by the structured fields, not by a knob the run failed to echo. A different solver or a DIFFERENT REPORTED config value blocks the outcome dimensions and keeps the cost; only a wrong task, a wrong strategy, a wrong round or a post-hoc prediction excludes the sample entirely. The per-dimension rule is one small framework table (`world_model/attribution.py`), never a dependency engine over field names.

**`bound` is not `calibratable`.** Being associated (`bound_action_id` set) says the prediction is linked to a real action. Whether ANYTHING may be calibrated is a separate question the CLOSE-OUT answers, field by field; the two are never conflated. `trace.comparable` becomes true only when the associated action is a completed real scope AND no mismatch exists (an attempt-scope prediction needs a completed linked execution; a window-scope prediction needs the window itself complete).

`bind-strategy` is the MANUAL/recovery path: it applies the SAME identity rules to an action that already ran, so a manual bind behaves exactly like an automatic one. Re-binding the same action is idempotent and re-bills nothing.

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
