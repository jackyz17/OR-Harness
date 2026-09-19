# Strategy-outcome prediction service (world-model M3)

**Status: the service is implemented and wired into planning.** This is the
third phase of the reconstruction: the unified contract (phase 1) and the
frozen prediction input context (phase 2) now carry a real, training-free
OR strategy-consequence prediction service.

- Protocol version: `wm-so/1`
- Python module: `or_harness.world_model.strategy_prediction`
- API: `ORHarness.predict_strategy_outcome` / `bind_strategy_outcome`
- CLI: `orx predict-strategy` / `orx bind-strategy` / `orx plan-next
  --protocol strategy-outcome`
- Tests: `tests/harness/test_strategy_outcome.py`
- Runnable example: [`references/examples/strategy_outcome.py`](examples/strategy_outcome.py)

The INPUT side (what a prediction is conditioned on) is documented in
[`references/prediction_context.md`](prediction_context.md); the CONTRACT
(the objects a prediction parses into) in
[`references/world_model_contract.md`](world_model_contract.md). This
document is the authority for the SERVICE: the protocol, the decision loop
and the execution binding.

---

## 1. The loop this phase implements

```
problem understanding / profile + memory retrieval
        -> ONE frozen PredictionContext
        -> a few candidate strategies
        -> predicted benefit / cost / risk / uncertainty per candidate
        -> compared, one suggested
        -> the outer agent EXPLICITLY chooses
        -> modelling, solving, verification (the agent's work)
        -> the real execution is BOUND to the chosen candidate's prediction
        -> re-plan from the real observation if warranted
```

- **P is exogenous**: the joint problem representation (math attributes,
  scenario semantics, structural relations) describes the problem; the
  service never predicts "the next problem P".
- **X is the solving context the decision needs** — not a full mathematical
  model, not solve.py, not every internal state.
- **H stays a latent capability**: capability evidence CONDITIONS the
  prediction; no H score, vector or capability-improvement claim is
  produced.
- **Training-free**: a frozen, externally configured model fills the
  prediction content; nothing is fine-tuned, calibrated online, or trained
  inside a task.
- **One macro comparison, then re-planning from the real observation.** The
  new protocol plans at horizon=1 only; no strategy-branch trial runs, no
  snapshot-rollback platform, no factory simulator, no latent-space
  dynamics. The legacy `plan-next` horizon=2 rollout remains available
  under the legacy protocol and its old boundaries.

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

**Candidates.** A candidate is a `CandidateRef` (or a legacy `ActionSpec`,
whose execution params and budget hint are preserved verbatim and whose
unmappable scopes are refused). A candidate may cover modelling /
decomposition / solving / repair — it is not just one solver call. The
service does NOT generate candidates (they come from you or the existing
catalog/recall) and never generates solve.py.

**Candidate identity.** The same `strategy_id` under a different solver,
`time_limit`, `mip_gap`, `seed` or step scope is a DIFFERENT candidate: the
config travels with the candidate into the prediction, the choice and the
execution binding, so a result is never bound to a configuration that was
not the one predicted.

**What the provider receives.** The full frozen context content — the joint
problem representation (text, CIR relations, math attributes with origins),
the retrieval evidence (hits with their content, not ids), the capability
evidence, the execution constraints — plus the candidate and an explicit
output contract. The framework fixes the task, the candidate, the scope and
the evidence sources; the model fills prediction content only.

**Output semantics.**

| Field | Contract object | Rules |
|---|---|---|
| benefit G | `BenefitEstimate` | metric, unit and baseline required with a value; `solution_quality` must be a NORMALIZED value in [0,1] (e.g. `1-gap`) — a raw objective value is refused, never clamped. Business production/transport cost is OR objective quality, NOT a harness resource cost |
| cost c | `ExpectedCost` / `CostVector` | the five resource dimensions; the measured mask marks PREDICTED dimensions (a prediction, not a measurement); an omitted dimension is unknown, never zero |
| risk L | `RiskStatement` / `RiskEvent` | named events with optional probability/severity; no basis → the fields stay absent; rework already in c is not repeated |
| uncertainty | `UncertaintyStatement` | execution randomness vs knowledge gap; a model self-report is recorded as `model_self_report` with the numbers in its notes, explicitly UNCALIBRATED — never relabeled as measured |
| trace | `PredictionTrace` | context/candidate/task/episode binding, protocol and prompt version, evidence refs, unsupported fields, call cost, error reasons |

**Honest failure states** — each distinguishable, each persisted with
whatever usage the call consumed:

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
orx plan-next --task t.json --episode ep1 --protocol strategy-outcome \
  [--candidates specs.json] [--max-calls N]
```

`plan-next --protocol strategy-outcome` (horizon fixed at 1):

1. freezes ONE context for the whole decision (every candidate is
   conditioned on the same problem representation, X/B, retrieval evidence,
   capability evidence and constraints — one embedding call, one snapshot);
2. predicts each candidate under wm-so/1 (a failed prediction does not drag
   the others down; the model-call count, wall clock and the REAL budget
   are checked before every call);
3. scores each candidate on ONE conservative yardstick:
   `U = alpha*G - beta*C - gamma*R` with the SAME weights the harness
   scores everything else with:
   - **G**: the candidate's own benefit value, when its kind is on the
     comparison's yardstick (normalized solution quality). A benefit of a
     different currency, or none, contributes NOTHING — unknown upside is
     never rewarded;
   - **C**: the predicted cost over the common basis (the union of the
     dimensions any candidate predicted). A candidate missing a basis
     dimension is charged that dimension's PEAK normalized share — unknown
     cost is never free;
   - **R**: the MAXIMUM event probability (one explicit risk-evaluation
     rule; events are never assumed independent, so probabilities are
     neither summed nor multiplied). An event with no probability basis —
     or no risk prediction at all — charges the full gamma weight as a
     deficit;
   - the knowledge term is OFF (delta=0) for this protocol: an old
     knowledge-gain score is not an H improvement and does not leak into
     the new comparison;
4. suggests the best candidate's first step. **A suggestion is not a
   selection**: only `choose-next` writes `X.selected_plan`, and it records
   accept / deviate / reject explicitly.

**Fallback.** When NO candidate carries a usable prediction (provider down,
every payload invalid), the plan reports `status="no_valid_predictions"`
with the reasons and NO suggestion — fall back to `recall`/Selector
ordering or choose yourself. The fallback is reported as what it is; it is
never dressed up as a completed world-model comparison, and the calls that
did happen keep their recorded cost.

**Cost accounting.** The model calls are REAL spend: charged once to the
decision action as own cost (failed calls included), reported in
`planning_cost`, never part of any candidate's utility. Candidate execution
costs are PREDICTED values. Missing usage stays unknown — never free.

## 4. Binding the real execution

```bash
orx execute ... --strategy S02 ...      # your modelling + solving + verify
orx record --from-staged <id>
orx bind-strategy --prediction PREDICTION_ID --action ACTION_ID
```

`bind-strategy` checks request identity — task, episode, strategy, solver
and the candidate's execution config — and records a mismatch rather than
silently comparing: a result produced under another configuration is not
this prediction's truth. The prediction's `trace.comparable` becomes true
only when the bound action is a completed real scope (an attempt-scope
prediction needs a completed linked execution; a window-scope prediction
needs the window itself complete). Re-binding the same action is idempotent
and re-bills nothing.

**What is deliberately left to M5/M6:**

- capability-evolution prediction, offline learning decisions and their
  effect verification (M5);
- closing the two known planner gaps (a candidate with no comparable
  benefit can still be suggested; the unknown-cost/unknown-risk charges
  are decision rules, not measurements) — M6, together with the
  integration acceptance. The M4 close-out's evaluation does NOT depend
  on the planner's utility or its suggestion: eligibility is decided by
  the prediction–outcome match and the measurement basis only.

The window-level error aggregation and the episode close-out are
**implemented in M4**: after the real execution, `orx close-episode`
evaluates every bound prediction field by field and publishes the
experience calibration — see
[`references/episode_closeout.md`](episode_closeout.md).

Unexecuted candidates keep `unexecuted`/`unbound` semantics: no
counterfactual truth is fabricated from the winner's result. After the real
execution you may build a NEW current context and re-plan — that updates
facts and decision inputs; it is not in-task parameter learning, calibration
or knowledge induction.

## 5. Compatibility

- The legacy `predict_outcome` / `plan-next` (default protocol) paths are
  unchanged, including `--no-context` byte-compatibility and the horizon=2
  rollout under the old protocol.
- Old `OutcomePrediction` records remain readable; the new predictions live
  in their own `contract_predictions` log table (created idempotently, no
  schema-version move).
- The `capability_evolution` kind still has a contract and NO service.
- `HttpChatProvider` selects the system prompt by the REQUEST's protocol:
  a `wm-so/1` request gets the strategy-outcome prompt; everything else
  keeps the legacy prompts. The wire format stays the OpenAI chat shape.

## 6. Verification checklist for a consuming agent

- [ ] Did the prediction come back `valid`? Only then is it a forecast.
      `contract_only` means nothing was predicted — check
      `provider_configured` and the notes for which failure state it is.
- [ ] Is `benefit.value` present? Then a `baseline` must be too, and a
      `solution_quality` value must be normalized — a raw objective value
      was refused, not clamped.
- [ ] Am I about to read `uncertainty` as a probability? A model
      self-report is UNCALIBRATED; the numbers live in its notes and are
      never used as measured probabilities.
- [ ] Am I comparing candidates? Unknown cost is charged the peak share,
      unknown risk the full weight, unknown benefit nothing — unknown never
      auto-wins, and that is a DECISION RULE, not a measured probability.
- [ ] Did the suggestion change the selection? No — only `choose-next`
      writes `X.selected_plan`.
- [ ] Am I binding an execution to a prediction of a DIFFERENT config?
      The mismatch is recorded and the prediction is not comparable. An
      UNKNOWN identity field (no episode on the action, a config key the
      log never carries) is recorded separately and never counts as a
      match either.
- [ ] Do I want window-level error numbers? Run `orx close-episode` when
      the episode ends: every bound prediction is evaluated field by
      field and the experience calibration is published.
