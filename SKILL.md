---
name: or-harness-wm
description: >
  Formulate, solve, verify, retry and learn from large-scale industrial
  optimization problems (LP, MILP, scheduling, routing, assignment, network
  flow, capacity planning, resource allocation, supply chain) whose decisions
  are coupled through shared resources, cross-stage dependencies or temporal
  propagation. Use when the user asks to build or fix an optimization model
  or solve.py; to choose between candidate solving strategies ("which
  approach should I use", "compare these formulations", "is this method worth
  trying") by predicted quality/cost/risk; to decide whether a solver's
  answer really satisfies the stated task rather than only its own model
  ("the solver says optimal, is the result actually right"); to record an
  attempt's real cost and outcome; to close an episode and read calibration;
  or to consolidate completed tasks offline (induction, capability
  prediction, adoption). Also use for follow-up turns on a task it already
  knows, where prior attempts and their costs should inform the next
  decision. Do NOT use for a one-off optimization question with no
  repetition, a pure mathematical proof, or non-optimization work — this
  layer's value is the memory built across repeated tasks.
---

# OR-Harness: strategy learning for optimization agents

You are the orchestrator; this layer advises, executes and remembers. Every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`) — no hidden state, no background work. Every command prints ONE JSON line: `{"result": {...}, "summary": "..."}` (exit `0` ok, `2` usage/precondition, `1` crash). You may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to consolidate.

**The world model is REQUIRED** on both timescales. Configure a provider (`--world-model URL::MODEL`, API key from `$OR_WM_API_KEY`, or `ORHarness(world_model=...)`). With none, `predict-strategy`/`plan-next` exit `2` with `result.error` naming the missing provider, and the persisted prediction status is `contract_only` — that is a configuration error, not a degraded mode.

## 1. What you deliver

A solved AND validated optimization task, plus the evidence that makes the next one cheaper: the answer, its task-level verdict, the real cost of every attempt, and the recorded facts that calibration and (optionally) the knowledge layer read back.

The task is done when every attempt you want counted has been checked against the TASK (`check-task`), recorded with its measured cost, and its episode closed. Two endings are legitimate: a verified answer, or a correctly identified infeasibility.

## 2. The whole loop

```text
  UNDERSTAND + RECALL   profile, recall  -> CIR + profile + memory report
            |
  PROPOSE      YOU name the candidate methods; there is no menu to enumerate
            |
  PREDICT + CHOOSE     ONE branch, never both:
            |            A) predict-strategy per candidate -> keep the chosen id
            |            B) plan-next -> choose-next --prediction <id> -> same id
            v  (a failed task check returns here, with the candidate fixed)
  MODEL -> EXECUTE --prediction <id> -> CHECK TASK -> RECORD
            |     (the association is established BEFORE the run;         (facts + real cost;
            |      the real result and config are added AFTER it)           failed attempts included)
            v
  FINISH       close-episode --task T --episode E --terminal STATE
            |
      +-----+----------------------------------------------------+
      v                                                          v
  FEEDBACK 1: CALIBRATION                     FEEDBACK 2 (OPTIONAL): INDUCTION
  re-score predictions against real           accumulate cross-task evidence, then
  outcomes; the result enters LATER           predict-capability -> compare-capability
  predictions                                 -> accept-capability (runs it and
                                              publishes knowledge for later solving)
                                                            |
                                              later real tasks accumulate
                                                            v
                                              evaluate-capability: did it help?
```

Stage responsibilities:

| Stage | Job | Output |
|---|---|---|
| Understand + recall | validate the CIR, fix the structural key, report real memory | `result.coupling`, `result.profile`, `result.recommendations` |
| Propose | name the candidate methods | a candidate list you write yourself |
| Predict + choose | one prediction per candidate, then one explicit decision | `prediction_id` |
| Model + execute | the CHOSEN strategy only: strategy first, then the formulation and `solve.py`; the prediction is associated BEFORE the run | `execution_id` |
| Check + record | did the ANSWER satisfy the TASK; the fact and its real cost | a verdict, a recorded row |
| Finish + feedback | calibration (automatic); induction (optional, offline) | a published summary; knowledge |

The two feedback paths answer different questions and run on different timescales.

| Feedback | What it updates | Where it lands |
|---|---|---|
| Calibration — automatic, at `close-episode` | prediction-error statistics over real outcomes | the input of LATER predictions |
| Induction — optional, offline, after the episode | strategic knowledge entries | what later solving can REUSE; its effect is judged separately by `evaluate-capability` on later tasks |

## 3. Before you start

- **Task file.** A JSON object with `task_id` and `family` at minimum. Add a `text` field so the text channel can index it, and a `coupling` field (a CIR) when the decisions are coupled. A task with no `model` field is a normal state — the model is written AFTER the strategy is chosen. **`task_id` is yours to choose**: one logical problem keeps ONE id however many times you solve it (a retry under another solver, a larger time limit), and an episode id (`--episode`, any stable label such as `ep1`) groups the attempts of one solving effort.

  ```json
  {"task_id": "delivery_q3", "family": "routing",
   "text": "3 depots, 40 stores, whole pallets only, 12h driving limit per truck",
   "spec": {"n_vars": 120, "n_constraints": 90, "n_int_vars": 120},
   "annotations": {"coupling": {"resource_coupling": 0.6,
                                "temporal_coupling": 0.7,
                                "route_complexity": 0.8,
                                "semantic_coupling": 0.5}}}
  ```

`family` is a label you choose; it groups statistics, so use the same word for the same kind of problem. `spec` numbers are what you can count (`n_vars`, `n_constraints`, `n_int_vars` at least — see [references/modeling.md](references/modeling.md)). **The README-level facts of the request ("12h driving limit") belong in `text` and in YOUR model — no field makes the framework enforce them** (see step 6 on declaring check bases).
- **Memory directory.** `--home DIR`, else `$OR_HARNESS_HOME`, else `./or_harness_home`. Every fact, entry, prediction and calibration lives there, so pass the same directory to every call in one episode.
- **World model.** Required (see the top of this file). Both `predict-strategy` and `plan-next` charge their model calls to the episode budget. With no provider they exit `2` (`result.error`, persisted status `contract_only`): `execute` still solves without `--prediction`, but nothing is then forecast, bound or calibrated — a configuration gap, not a normal path.

Two settings are worth setting EXPLICITLY, because their defaults are conservative and a silent fallback would make a truncated answer look like a model failure:

  | Setting | Precedence | Default |
  |---|---|---|
  | Output budget | `--wm-max-tokens` > `$OR_WM_MAX_TOKENS` > adapter default | 2048 |
  | Call timeout | `--wm-timeout` > `$OR_WM_TIMEOUT` > adapter default | 300s |

  ```bash
  # a request of ~10k input tokens; the default 2048 output budget cut it short
  OR_WM_MAX_TOKENS=8192 orx --world-model $URL::MODEL --home ./mem \
      predict-strategy --task t.json --candidate c.json --episode ep1
  ```

The BUDGET is what the endpoint is asked to generate; the TIMEOUT bounds one blocking socket operation (not a hard deadline on the whole request). `plan-next` additionally takes `--time-budget` (whole decision, default 120s) and passes what remains to each call. The values actually in force come back on the result (`effective_parameters`), so read them rather than assuming.
- **Solver.** Name the concrete solver on `execute` (`--solver`) for an UNPREDICTED attempt; when you pass `--prediction` the solver (and strategy, and episode) come from the candidate, so you do not re-type them — an explicit value that contradicts the candidate is refused before the run. `recall` returns `available_solver_families` and `solver_advisories`. Subprocess solvers (PuLP's CBC) cannot run in the sandbox — use an in-process solver (ortools, highspy).
- **Embedding (optional).** The text channel is OFF unless a real embedding model is configured; with none, `recall` returns `degraded` plus the structural channel alone, which is a different fact from "ran and matched nothing". After configuring one, run `orx rebuild-index` once; `record`/`induce`/`retire` keep it current.
- **Cold start.** An empty `recall` is the normal state, not an error: no memory matched, so YOU propose and describe the candidate methods yourself. Predicting compares them; you then execute the ONE you chose. Several candidates being predicted never means several are executed — a second attempt is triggered by a real failure, by an explicit uncertainty you cannot resolve, or by your own decision, and it is not automatic.

## 4. Online operations

Run these in order. Steps 3a and 3b are alternatives — doing both predicts the same candidate twice. Every `result.*` path below is the TOP-LEVEL key of the command's JSON line; nested paths are written out in full.

| # | Action | Input | Read from the result | Next |
|---|---|---|---|---|
| 1 | `orx profile --task t.json` | task JSON, optional `--cir` | `result.coupling`, `result.profile`, `result.derivation` | 2 |
| 2 | `orx recall --task t.json --top 3` | task | `result.recommendations[]`, `result.held_claims[]`, `result.recommendations_basis`, and `result.vector_recall` **only when the text channel is configured** (otherwise only `result.degraded`) | 3a or 3b |
| — | `orx context --task t.json` (optional) | task, for a stored frozen input | `result.context_id` | pass it to 3a as `--context`; `predict-strategy` builds its own when omitted |
| 3a | `orx predict-strategy --task t.json --candidate c.json [--episode ep1]` | task + one candidate, repeated per candidate | `result.prediction_id`, `result.prediction.status` | 4, keeping the chosen candidate's id |
| 3b | `orx plan-next --task t.json --episode ep1 --candidates cs.json` | task + candidate list | `result.decision_action_id`, `result.plan.candidates[].prediction_id`, `result.plan.suggested`, `result.plan.status` | 3b′ |
| 3b′ | `orx choose-next --decision <decision_action_id> --prediction <prediction_id>` | the decision id + the chosen candidate's prediction id from 3b | `result.selected` (the chosen spec), `result.deviation` (set only when you chose against the suggestion) | 4, executing with that same prediction id |
| 4 | Write the model, then `solve.py` | the chosen strategy | your script's `result.json` (`status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and `variables` if a check will need the answer) | 5 |
| 5 | `orx execute --task t.json --prediction <prediction_id> --code solve.py --workspace ws` | task + the prediction id from 3a or 3b (strategy/solver come from the candidate) | `result.execution_id`, `result.prediction_binding` | 6 |
| 6 | `orx check-task <execution_id> --check '{...}'` | the execution id from 5 | `result.state`, `result.report.conclusion`, `result.report.scope.unchecked`, `result.next` | 7, or the retry loop |
| 7 | `orx record --from-staged <execution_id> [--usage-file usage.json] [--override llm_tokens=<count>]` | the execution id from 5; the host's usage report if you have one | `result.recorded`, `result.cost_completeness`, `result.induction_hints` | 8, or back to 3a/3b |
| 8 | `orx close-episode --task T --episode E --terminal STATE` | task + episode | `result.task_checks`, `result.evaluations[]`, `result.calibration_summary`, `result.closeout.terminal_state` | offline operations |

- **1 — understand.** Put your coupling understanding in the task's `coupling` field. A malformed CIR is a precondition error (exit 2), never a silent empty one; `--allow-empty-cir` accepts one with no structure. A `coupling_warnings` entry means your supplied value contradicts the derivation, and the derived value wins for grouping. **If the task has no `model` yet, that is normal** — the model is written at step 4, after the strategy is chosen.
- **2 — recall is a report, not a menu.** `recommendations` says what may be REUSED; `held_claims` names applicable claims NOT yet published. An empty list with `recommendations_basis.reason` is a valid answer (cold start), not an error. `vector_recall` is the text-similarity channel: **if that key is absent, the channel is OFF** (see `result.degraded`) — that is a different fact from "ran and matched nothing". **If a recommendation exists but you cannot see why it applies** (empty `knowledge.applicability`, low `support_n`, or no verified claim), treat it as a hint to test, not a decision, and read [references/induction.md](references/induction.md) before trusting it.
- **3 — propose, then predict with ONE branch.** You name the candidates; no directory exists to enumerate. Three rules, then the branch choice:
  - **Write the METHOD, not just an id.** Put the approach in the candidate's `method` (`{"name": ..., "steps": [...]}`) and the execution parameters (`time_limit`, `mip_gap`, `seed`, …) in `config`. A bare strategy id or solver name tells the model nothing, so an undescribed method is an unanswerable request. A `method` key mistakenly placed in `config` is moved to `method` and recorded; a key that names no execution parameter is refused before an attempt is spent.
  - **One benefit convention per decision.** Candidates are compared in ONE `kind`/`metric`/`unit` (`--benefit-kind`/`--benefit-metric` on `plan-next`), else the candidates' own agreement, else `solution_quality`/`normalized_objective_gap`. `solution_quality` measures how well the SOLVER solved the model; `effective_completion`/`task_result_check_passed` measures whether the ANSWER satisfies the TASK — different measurements, comparable only within their own kind. A candidate whose benefit is in another currency is reported and does not enter the full ranking (an unread upside is never scored as zero).
  - **Pick ONE branch — never both.** Branch A (ordinary): `predict-strategy` once per candidate; keep the chosen candidate's `result.prediction_id`. Branch B (candidates close, or the decision is high-stakes): `plan-next`, then `choose-next --decision <result.decision_action_id> --prediction <the chosen candidate's prediction_id>`. `choose-next` belongs to branch B only — branch A writes no planning decision. Planning is horizon 1; a `--horizon` other than 1 is refused (re-plan from the REAL observation of the chosen step instead).
  - **If a prediction comes back not `valid`, read `result.failure.kind` before retrying**: `truncated` = the output budget was too small (`finish_reason: "length"`; raise `--wm-max-tokens`), `wrong_top_level` = the model answered with an array, `empty_response` = no text at all, `timeout`/`network_error` = the call did not complete. `finish_reason: null` means the endpoint reported nothing, so how it ended is UNKNOWN. **If `unsupported_fields` is non-empty**, the model could not judge those fields — that is a reported gap, not a zero; do not read it as "no cost" or "no risk".
- **4 — model, then code (your judgement).** Write the representation into the task's `model` field AFTER the strategy is chosen, then write `solve.py` — the framework generates neither, and the formulation is yours to decide (see [references/modeling.md](references/modeling.md)). Two hard sandbox rules: every constraint label must be `C1`, `C2`, `C3`, … (a different naming is the most common failure), and the sandbox blocks `subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil`, `pathlib` while restricting `os` to `os`/`os.path` — importing a solver library (ortools, highspy, …) IS allowed. The answer goes to a literal `open('result.json', 'w')`, which resolves inside `--workspace`.
- **5 — execute with the prediction.** `execute --prediction <prediction_id>` establishes the association BEFORE the run: the frozen candidate supplies the strategy/solver/episode (do not re-type them — a conflicting explicit argument is refused before anything executes), and a prediction already consumed by another attempt is refused. After the run the real result and the observed configuration are added to the same association. `result.prediction_binding` reports `bound`, `config_observed` (keys the run really confirmed) and `config_unknown` (keys the run never reported — a missing receipt is a caveat, not a discard). Every attempt is staged automatically and a prediction never becomes a fact. **If the executor is interrupted before producing a record**, the attempt is staged as its OWN failure fact and the association is RELEASED, so the same prediction can be tested against the retry. Omit `--prediction` only when you never predicted the candidate; bind later with `bind-strategy`.
- **6 — check the ANSWER.** `check-task <execution_id>` covers the TASK; `execute`'s own verdict covers only the solver's MODEL. Read `result.state`: **`passed`** = the DECLARED bases held, never proof the model is right (declare what you can support: `reference_objective`, `reference_status`, `integer`, `recompute_objective`, `semantic_probe`, `intent`); **`failed`** = the answer is confirmed wrong — record this attempt with its real cost, fix the model, and go back to **step 3** (a retry of a CONSUMED attempt — one that produced a real record — needs its OWN prediction and its OWN check); **`insufficient`** = validity is UNKNOWN, not a pass — declare the missing basis or have `solve.py` report `variables`, and `result.report.scope.unchecked` names what was not checked. An undeclared basis is simply not checked, and you never change the task to match a reference value. The verdict is also an OBSERVATION: a prediction that declared `effective_completion / task_result_check_passed` is scored against it at close-out (1.0 `passed`, 0.0 confirmed `failed`; unchecked/`insufficient` stays UNKNOWN, never counted as a failure).
- **7 — record the fact and the real cost (exact rules).** `record --from-staged <execution_id>` — also for timeouts, infeasibilities and failures. The sandbox measures only `latency_s` and `solver_runtime_s`; `llm_tokens`, `tool_calls` and `retries` are invisible to it. Therefore:
  - **Prefer the REAL token figure**: `record --from-staged <id> --usage-file usage.json` ingests the host's attempt-level usage report — the full口径 total (prompt + completion), marked as a host observation and usable as a calibration truth. The accepted shape is in [references/commands.md](references/commands.md).
  - **A bare `--override llm_tokens=<count>` is a DECLARATION** (`agent_estimate`): kept and shown, but **never** used as a calibration actual, a cost claim or a learning mean. Pass `--override-source agent_observed` (or `provider_usage`) when the number really came from a report. A single-side (completion-only) figure is a LOWER bound, never a complete total.
  - **`retries` means the NEW retries THAT attempt represents**: a first attempt records `retries=0` as an observed fact; the attempt that retried it declares `--override retries=N`.
  - Amend later with `orx amend-cost <id> --override <dim>=<value>` or `--usage-file`; the framework refuses to overwrite a dimension it measured unless you pass `--force`.
- **8 — finish the episode.** `close-episode` is idempotent, reads what was recorded, publishes the calibration and reports `task_checks` coverage — a close-out is not a certification of the answer. Pick `--terminal` to match what happened: `completed` (a passed check, or a correctly identified infeasibility), `failed` (the answer was confirmed wrong), `aborted` (you stopped), `budget_exhausted`. Read `result.calibration_summary` for the published numbers and `result.evaluations[]` for the per-prediction scoring. **If the result reports `state: "pending"`**, an action is still `running` — end it first.

### Who does what

| The FRAMEWORK does this automatically | YOU decide this |
|---|---|
| Stages every attempt (success and failure), measures `latency_s`/`solver_runtime_s`, proves a first attempt's `retries=0` | the candidate methods, their `method` description, and which one to run |
| Binds the prediction to the real run, records the observed config, keeps `config_observed`/`config_unknown` apart | the real token/tool-call figures and their SOURCE |
| Evaluates each bound prediction field by field at `close-episode` and publishes the calibration | the check bases you can honestly support, and the episode's `--terminal` state |
| Re-derives a late task check, and excludes/includes samples by evidence eligibility | whether to consolidate offline, and which knowledge claim (if any) to submit |
| Keeps the evidence window bounded (`enforce-window`) — eviction is a reference expiring, never a refutation | whether a result is "good enough", whether to retry, and on what grounds |

### When you get stuck

| Symptom | Read | Do next |
|---|---|---|
| `recall` is empty, or nothing usable matched | this file §3 (cold start) | propose and describe the candidate methods yourself; go to step 3 |
| A recommendation matched but you do not trust it | [references/induction.md](references/induction.md) | test it as one candidate among several; do not adopt it blindly |
| The candidates are close, or the choice is costly | this file §4 step 3 (branch B) | `plan-next` → `choose-next`; read `result.candidates[].score.incomparable` |
| `execute` or `check-task` failed / returned `insufficient` | step 6 above, then [references/commands.md](references/commands.md) | record the attempt honestly, fix the model, re-predict (a new prediction) |
| Prediction is `not valid`, or fields are missing | step 3 above, then [references/strategy_outcome.md](references/strategy_outcome.md) | read `result.failure.kind` / `unsupported_fields`; a missing field is a gap, not a zero |
| You want to know what a prediction was conditioned on | [references/prediction_context.md](references/prediction_context.md) | inspect the frozen context, then decide whether to re-predict |
| The episode is done and you want the loop to learn | [references/induction.md](references/induction.md) | `close-episode` first, then the offline operations below |

## 5. Offline operations

Induction never runs automatically and never inside an episode. `induce` is the only place knowledge changes, and `accept-capability` is the only command that runs a maintenance operation; the `induction_hints` you get from `record` are attention signals, not inductions.

| # | Action | When | Output (JSON path) |
|---|---|---|---|
| 1 | `orx induction-candidates` | you are considering a maintenance decision | `result.candidates[]` — frozen evidence packages (detector-derived and cell-derived), no model call |
| 1′ | `orx induction-material [--bundle ID] [--pattern P]` | a candidate exists | the methods, the comparison evidence and what followed — read it, then form the claim yourself |
| 2 | `orx induce --relation '{...}' [--verify ...]` | you have read the material and formed a claim | `result.relations[].publication` — the framework checks what you submitted; ONE entry is ONE claim |
| 3 | `orx predict-capability --operation op.json --bundle <bundle>` | a bundle exists and you want to price the operation | `result.prediction_id` |
| 4 | `orx compare-capability --predictions hp_1,hp_2 --horizon-tasks 10` | two or more predictions exist | `result.recommendation` (`accept`/`defer`), `result.selected_prediction_id`, `result.incomparable[]` |
| 5 | `orx accept-capability --recommendation <json>` or `orx reject-capability ...` | you have decided | `result.binding.knowledge_delta`; accept also runs the operation |
| 6 | `orx evaluate-capability --prediction hp_1` | qualified LATER tasks have accumulated | `result.evaluation.state`, `result.evaluation.effect_verified` |
| — | `orx induce [--verify '<json>']` | you are writing statistical knowledge directly, without the capability-prediction path | `result.results[]` — created / updated / skipped |

Steps 3–6 are the PREDICTION route; step 2 (and the last row) are the DIRECT route. **Use one or the other for the same decision, never both** — accepting a recommendation already ran the operation, so calling `induce` again for it would be a second write.

**Method material first, technique second.** `induction-candidates` freezes the evidence; `induction-material` tells you what it says — the methods actually used, the change (for an intervention), both sides of a comparison, and what followed. A candidate reports `material=insufficient` when there is nothing to abstract (no performed method at all, or only PLANNED methods — a plan is intent, not a performed method) and `material=unavailable` when the comparison lost a whole side to an exclusion or a later task check refuted the premise. The framework will not turn a strategy name and a mean into a technique; the right response is to record how the work was really done (the script's `method_performed` receipt stamped with `$OR_ACTION_ID`) or re-check the evidence. `purpose` says what the candidate is FOR: `method_induction` (a comparison/recovery worth abstracting) vs `statistical_refresh` (a cell that cleared the count gate — its claim is its statistics, not a technique). Form the claim yourself (condition → how → consequence → boundary) and submit it with `induce --relation`, citing the executions or the `bundle_id` you were shown.

**Entry condition, and what to do when evidence is thin.** Creating a statistical entry needs ≥2 executions from ≥2 DISTINCT tasks; publishing it needs a passed admission check (`induce --verify`). When `induction-candidates` reports none, either keep solving and recording until the evidence exists, or submit a knowledge CLAIM instead (`induce --relation ... --verify '{"purpose":"relation",...}'`) for knowledge that is not one strategy's statistics. Patterns, thresholds and the claim format: [references/induction.md](references/induction.md).

**Storage and repair.** `inspect --bank retention` shows the calibration window, late-check grace period and archive caps; `archive-calibration --dry-run` previews without writing; `calibration --rebuild` repairs after a late correction. The RAW evidence is bounded separately by the **evidence window** (`enforce-window`, default 800 complete episodes, run last after a close-out): eviction is a source reference EXPIRING, never a refutation and never a withdrawal — `exclude-execution`/`restore-execution` are the explicit correction channels.

## 6. Reference index

Read on demand, one hop, no chains: the workflow above is complete on its own.

| Read it when | Reference |
|---|---|
| You are about to call a command and need its flags, return states or failure handling | [references/commands.md](references/commands.md) |
| You are writing your first model or CIR | [references/modeling.md](references/modeling.md) |
| You want the "why": coupling-aware understanding, the two-layer memory, strategy cost, how the world model fits in | [references/concepts.md](references/concepts.md) |
| You are deciding whether to induce, or submitting a knowledge claim | [references/induction.md](references/induction.md) |
| You are debugging what a prediction was conditioned on | [references/prediction_context.md](references/prediction_context.md) |
| You are predicting or comparing candidates under wm-so/1 | [references/strategy_outcome.md](references/strategy_outcome.md) |
| You need a protocol field definition, a status value, or a legacy mapping | [references/world_model_contract.md](references/world_model_contract.md) |
| You are closing an episode, reading the calibration, or handling a late verdict | [references/episode_closeout.md](references/episode_closeout.md) |
| You want a runnable script to read | `references/examples/`: `no_catalog.py` (cold start), `strategy_outcome.py` (predict two candidates and bind the real execution), `task_check.py` (check → diagnose → repair → re-check), `method_evidence.py` (record the method → detect a same-solver fix → read the material → submit a relation → recall it), `episode_closeout.py` (evaluate → calibrate → retain), `capability_evolution.py` (the whole offline path), `capability_followup.py` (an online H+ followed up: token eligibility, a preserved failure, and the trace→bound→evaluate chain), `usage_accounting.py` (token口径 and cost provenance), `linkage_attribution.py` (which identity problem blocks which comparison), `prediction_context.py` and `contract_roundtrip.py` (frozen inputs and contract shapes) |

Every example runs with no model, no network and no credentials by injecting a fixed-output stub provider, so what each one demonstrates is the PROCESS, never model accuracy. The ids they produce are the same ones the table above passes between commands. **If you read only one, read `end_to_end.py`**: it runs a cold start through `close-episode`, including a `failed` check and the repair.
