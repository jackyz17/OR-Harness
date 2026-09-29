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

**What the provider receives.** The full frozen context content — the joint problem representation (text, CIR relations, math attributes with origins), the retrieval evidence (hits with their content, not ids), the capability evidence, the execution constraints — plus the candidate, an explicit output contract AND a `benefit_convention` block. The framework fixes the task, the candidate, the scope, the evidence sources AND the benefit convention; the model fills prediction content only.

**Output semantics.**

| Field | Contract object | Rules |
|---|---|---|
| benefit G | `BenefitEstimate` | metric, unit and baseline required with a value; `solution_quality` must be a NORMALIZED value in [0,1] (e.g. `1-gap`) — a raw objective value is refused, never clamped. Business production/transport cost is OR objective quality, NOT a harness resource cost |
| cost c | `ExpectedCost` / `CostVector` | the five resource dimensions; the measured mask marks PREDICTED dimensions (a prediction, not a measurement); an omitted dimension is unknown, never zero |
| risk L | `RiskStatement` / `RiskEvent` | named events with optional probability/severity; no basis → the fields stay absent; rework already in c is not repeated |
| uncertainty | `UncertaintyStatement` | execution randomness vs knowledge gap; a model self-report is recorded as `model_self_report` with the numbers in its notes, explicitly UNCALIBRATED — never relabeled as measured |
| trace | `PredictionTrace` | context/candidate/task/episode binding, protocol and prompt version, evidence refs, unsupported fields, call cost, error reasons |

**Honest failure states** — each distinguishable, each persisted with whatever usage the call consumed:

| Situation | Result |
|---|---|
| No provider configured | `contract_only`, `provider_configured=false` |
| Provider error / timeout | `invalid` with the error; the call cost is kept |
| Empty payload / all fields omitted | a recorded non-prediction (`contract_only`-shaped, "no predicted content") |
| Invalid JSON, NaN/Infinity, out-of-range probability, wrong type | `invalid` with the specific problems |
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
3. scores each candidate on ONE conservative yardstick: `U = alpha*G - beta*C - gamma*R` with the SAME weights the harness scores everything else with:
   - **G**: the candidate's own benefit value, only when BOTH its kind AND its metric are on the comparison's yardstick (`solution_quality` / `normalized_objective_gap` — the SAME pair the close-out observes). A benefit of another kind or metric is reported, and the candidate produces **no utility at all** — an upside in a currency the comparison cannot read is never scored as zero (which would let it be ranked as if it were a real, worthless benefit);
   - **C**: the predicted cost over the common basis (the union of the dimensions any candidate predicted). A candidate missing a basis dimension is charged that dimension's PEAK normalized share — unknown cost is never free;
   - **R**: the MAXIMUM event probability (one explicit risk-evaluation rule; events are never assumed independent, so probabilities are neither summed nor multiplied). An event with no probability basis — or no risk prediction at all — charges the full gamma weight as a deficit;
   - the knowledge term is OFF (delta=0) for this protocol: an old knowledge-gain score is not an H improvement and does not leak into the new comparison;
4. suggests the best candidate's first step. **A suggestion is not a selection**: only `choose-next` writes `X.selected_plan`, and it records accept / deviate / reject explicitly. Accept by ID (`choose-next --decision <id> --prediction <the chosen candidate's id>`): the candidate is read from this decision's own recorded comparison, and the deviation test compares the candidate REFERENCE (action type/strategy/solver/config), never the whole plan JSON.

**Fallback.** When NO candidate carries a usable prediction (provider down, every payload invalid), the plan reports `status="no_valid_predictions"` with the reasons and NO suggestion — fall back to `recall`/Selector ordering or choose yourself. The fallback is reported as what it is; it is never dressed up as a completed world-model comparison, and the calls that did happen keep their recorded cost.

**One convention per decision, resolved and reported.** The comparison reads ONE benefit currency, and where it came from is recorded in `result.plan.benefit_convention` (`kind`/`metric`/`unit`/`source`/`reason`). Three sources, in order:

1. **declared** — the caller names it for this decision: `--benefit-kind` / `--benefit-metric` (optionally `--benefit-unit`). It is then pushed into EVERY candidate's request as a REQUIRED convention, so the candidates are PREDICTED in that currency, not merely compared in it. Use it to compare completion: `--benefit-kind effective_completion --benefit-metric task_result_check_passed`.
2. **agreed** — nothing declared, and every candidate carrying a benefit declared the SAME `kind`/`metric`. This is what makes two completion candidates comparable without ceremony.
3. **default** — otherwise the build's own yardstick (`solution_quality` / `normalized_objective_gap`); the disagreement is named in `reason`.

A declaration this build has no scale for (`valid_progress`, or an unknown pair) resolves to `source="unknown_declaration"` with `comparable=false`: **no ranking is produced at all**, because ranking by an invented scale is worse than saying there is none. A candidate whose own `kind`/`metric` differs from the decision's convention is reported in `score.incomparable["benefit"]` and produces **no utility** — so it can never silently win or lose the ranking on an upside the comparison could not read. It keeps its cost and risk picture (still reported, still charged).

**Declare the benefit's meaning explicitly.** The `kind`/`metric` pair is the prediction's own yardstick, and the close-out observes the SAME one (the convention in the request says so):

| You are predicting | Declare | Observed from |
|---|---|---|
| how well the solver solved the model | `kind=solution_quality`, `metric=normalized_objective_gap` | the solver's gap (an `optimal` status is a gap of 0) |
| whether the ANSWER satisfies the TASK | `kind=effective_completion`, `metric=task_result_check_passed` | the execution's own `check-task` verdict — 1.0 `passed`, 0.0 confirmed `failed`, UNKNOWN when unchecked or `insufficient` |

Candidates compared on completion are compared **with each other**, never against a quality candidate (and the reverse): the convention has to match before two numbers are ranked, because `1 - mip_gap` is not a completion rate and a `passed` check is not a quality level, even though both land in [0,1]. `valid_progress` has no observation channel in this build, so it is reported and never scored — and never quietly replaced by a completion rate.

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

## 4. Associating the real execution (before it runs)

```bash
# the ordinary flow: the association is established BEFORE the run
orx execute --task t.json --prediction PREDICTION_ID --code solve.py --workspace ws

# the manual/recovery path, for an action that already ran
orx bind-strategy --prediction PREDICTION_ID --action ACTION_ID
```

**The association is established before the execution, and the observation is added after it.** `execute --prediction` reads the FROZEN candidate, checks it, and takes the strategy/solver/episode FROM it — the caller does not re-type what the prediction already fixed: the**problem identity** must match (a changed task/data/CIR is a different problem; writing the `model` after predicting is NOT a change), the prediction's status must be usable, and a prediction already claimed by another attempt is refused (one attempt per candidate). A path error, a contradictory explicit argument or a claimed prediction is reported BEFORE anything runs, so the prediction is not spent and no cost is incurred; the action and its link are persisted before the sandbox starts, so an interrupted execution still shows which attempt the prediction was testing. What the run then adds is the RESULT and the OBSERVED configuration — never a re-interpretation of the candidate.

**A missing receipt is a caveat, not a discard.** The config comparison reads the configuration that really took effect, from the action's own `execution_config` (the executor's report plus the solve script's `config` receipt, accepted only when stamped with this attempt's action id so a leftover `result.json` is never read as this run's config). A key the run never reported is recorded under `binding_unknown` — separately from a known `binding_mismatch` — but it does NOT erase the sample: the close-out blocks only the comparison the problem really invalidates. An unreported effort knob (`time_limit`, `seed`) blocks nothing; an unreported APPROACH knob (`method`, `relaxation`) blocks the benefit and keeps the measured cost; a different solver or a different config value blocks the outcome dimensions and keeps the cost; only a wrong task, a wrong strategy, a wrong round or a post-hoc prediction excludes the sample entirely. The per-dimension rule is one small framework table (`world_model/attribution.py`), never a dependency engine over field names.

**`bound` is not `calibratable`.** Being associated (`bound_action_id` set) says the prediction is linked to a real action. Whether ANYTHING may be calibrated is a separate question the CLOSE-OUT answers, field by field; the two are never conflated. `trace.comparable` becomes true only when the associated action is a completed real scope AND no mismatch exists (an attempt-scope prediction needs a completed linked execution; a window-scope prediction needs the window itself complete).

`bind-strategy` is the MANUAL/recovery path: it applies the SAME identity rules to an action that already ran, so a manual bind behaves exactly like an automatic one. Re-binding the same action is idempotent and re-bills nothing.

**What is deliberately out of scope here:**

- capability-evolution prediction, offline learning decisions and their effect verification — see [`references/world_model_contract.md`](world_model_contract.md) and the `*-capability` commands in [`references/commands.md`](commands.md);
- closing the two known planner gaps (a candidate with no comparable benefit can still be suggested; the unknown-cost/unknown-risk charges are decision rules, not measurements). The close-out's evaluation does NOT depend on the planner's utility or its suggestion: eligibility is decided by the prediction–outcome match and the measurement basis only.

The window-level error aggregation and the episode close-out are **implemented**: after the real execution, `orx close-episode` evaluates every ASSOCIATED prediction field by field and publishes the experience calibration — see [`references/episode_closeout.md`](episode_closeout.md).

Unexecuted candidates keep `unexecuted`/`unassociated` semantics: no counterfactual truth is fabricated from the winner's result. After the real execution you may build a NEW current context and re-plan — that updates facts and decision inputs; it is not in-task parameter learning, calibration or knowledge induction.

## 5. Compatibility

- The legacy `predict_outcome` Python API is kept for READING existing records (the contract reader, the record-time knowledge verdicts, `inspect --bank predictions`). No agent-facing entry point uses it, and the horizon=2 rollout is gone.
- Old `OutcomePrediction` records remain readable; the new predictions live in their own `contract_predictions` log table (created idempotently, no schema-version move).
- The `capability_evolution` kind has its own contract AND service (`wm-ce/1`): `predict-capability` / `compare-capability` / `accept-capability` / `bind-capability` / `evaluate-capability`. Its records live in a SEPARATE table, so neither generation is ever read as the other.
- `HttpChatProvider` selects the system prompt by the REQUEST's protocol: a `wm-so/1` request gets the strategy-outcome prompt; everything else keeps the legacy prompts. The wire format stays the OpenAI chat shape.
