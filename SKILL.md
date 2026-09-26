---
name: or-harness-wm
description: >
  Formulate, solve, retry, validate and debug large-scale industrial
  optimization problems (LP, MILP, scheduling, routing, assignment, network
  flow, capacity planning, resource allocation, supply chain) whose decisions
  are coupled through shared resources, cross-stage dependencies or temporal
  propagation, and run the learning loop around them. Use when asked to
  formulate, solve, retry, validate or debug an optimization model; to compare
  candidate solving strategies by predicted quality/cost/risk before
  committing; to check whether a solver's answer satisfies the ORIGINAL task;
  to close out a finished episode and read its calibration; or to run offline
  consolidation (induction, capability prediction, adoption) over completed
  tasks. Do not use for one-off optimization questions with no repetition,
  generic math proofs, or non-optimization tasks.
---

# OR-Harness: strategy learning for optimization agents

You are the orchestrator; this layer advises, executes and remembers. Every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`) — no hidden state, no background work. Every command prints ONE JSON line: `{"result": {...}, "summary": "..."}` (exit `0` ok, `2` usage/precondition, `1` crash). You may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to consolidate.

**The world model is REQUIRED** on both timescales. Configure a provider (`--world-model URL::MODEL`, API key from `$OR_WM_API_KEY`, or `ORHarness(world_model=...)`). With none, prediction returns `not_configured` and the loop cannot run to spec — that is a configuration error, not a degraded mode.

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
            |            B) plan-next -> choose-next       -> reuse the planner's id
            v  (a failed task check returns here, with the candidate fixed)
  MODEL -> EXECUTE --prediction <id> -> CHECK TASK -> RECORD
            |                                          (facts + real cost;
            |                            failed attempts included)
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
| Model + execute | strategy first, then the formulation and `solve.py` | `execution_id` |
| Check + record | did the ANSWER satisfy the TASK; the fact and its real cost | a verdict, a recorded row |
| Finish + feedback | calibration (automatic); induction (optional, offline) | a published summary; knowledge |

The two feedback paths answer different questions and run on different timescales.

| Feedback | What it updates | Where it lands |
|---|---|---|
| Calibration — automatic, at `close-episode` | prediction-error statistics over real outcomes | the input of LATER predictions |
| Induction — optional, offline, after the episode | strategic knowledge entries | what later solving can REUSE; its effect is judged separately by `evaluate-capability` on later tasks |

## 3. Before you start

- **Task file.** A JSON object with `task_id` and `family` at minimum. Add a `text` field so the text channel can index it, and a `coupling` field (a CIR) when the decisions are coupled. A task with no `model` field is a normal state — the model is written AFTER the strategy is chosen. **`task_id` is yours to choose**: one logical problem keeps ONE id however many times you solve it (a retry under another solver, a larger time limit), and an episode id (`--episode`, any stable label such as `ep1`) groups the attempts of one solving effort.
- **Memory directory.** `--home DIR`, else `$OR_HARNESS_HOME`, else `./or_harness_home`. Every fact, entry, prediction and calibration lives there, so pass the same directory to every call in one episode.
- **World model.** Required (see the top of this file). Both `predict-strategy` and `plan-next` charge their model calls to the episode budget. If no provider is available, prediction returns `not_configured` and you cannot run the loop to spec: `execute` still solves without `--prediction`, but nothing is then forecast, bound or calibrated — treat that as a configuration gap, not a normal path.
- **Solver.** Name the concrete solver on `execute` (`--solver`); `recall` returns `available_solver_families` and `solver_advisories`. Subprocess solvers (PuLP's CBC) cannot run in the sandbox — use an in-process solver (ortools, highspy).
- **Embedding (optional).** The text channel is OFF unless a real embedding model is configured; with none, `recall` returns `degraded` plus the structural channel alone, which is a different fact from "ran and matched nothing". After configuring one, run `orx rebuild-index` once; `record`/`induce`/`retire` keep it current.
- **Cold start.** An empty `recall` is the normal state, not an error: propose the methods you want and run them.

## 4. Online operations

Run these in order. Steps 3a and 3b are alternatives — doing both predicts the same candidate twice.

| # | Action | Input | Output (JSON path) | Next |
|---|---|---|---|---|
| 1 | `orx profile --task t.json` | task JSON, optional `--cir` | `result.coupling`, `result.profile`, `result.derivation` | 2 |
| 2 | `orx recall --task t.json --top 3` | task | `result.recommendations[]`, `result.knowledge[]`, `result.vector_recall` | 3a or 3b |
| — | `orx context --task t.json` (optional) | task, for a stored frozen input | `result.context_id` | pass it to 3a as `--context`; `predict-strategy` builds its own when omitted |
| 3a | `orx predict-strategy --task t.json --candidate c.json [--episode ep1]` | task + one candidate, repeated per candidate | `result.prediction_id` | 4, keeping the chosen candidate's id |
| 3b | `orx plan-next --task t.json --episode ep1 --candidates cs.json` | task + candidate list | `result.decision_action_id`, `result.plan.candidates[].prediction_id` | 3b′ |
| 3b′ | `orx choose-next --decision <decision_action_id> --chosen c.json` | the decision action id from 3b | `result.selected_plan` | 4, keeping the chosen candidate's id |
| 4 | Write the model, then `solve.py` | the chosen strategy | your script's `result.json` (`status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and `variables` if a check will need the answer) | 5 |
| 5 | `orx execute ... --prediction <prediction_id>` | the prediction id from 3a or 3b | `result.execution_id`, `result.prediction_binding` | 6 |
| 6 | `orx check-task <execution_id> --check '{...}'` | the execution id from 5 | `result.verdict` | 7, or the retry loop |
| 7 | `orx record --from-staged <execution_id> --override llm_tokens=<actual>` | the execution id from 5 | `result.recorded`, `result.induction_hints` | 8, or back to 3a/3b |
| 8 | `orx close-episode --task T --episode E --terminal STATE` | task + episode | `result.task_checks` | offline operations |

- **1 — understand.** Put your coupling understanding in the task's `coupling` field. A malformed CIR is a precondition error (exit 2), never a silent empty one; `--allow-empty-cir` accepts one with no structure. A `coupling_warnings` entry means your supplied value contradicts the derivation, and the derived value wins for grouping.
- **2 — recall is a report, not a menu.** `recommendations` says what may be REUSED, `vector_recall` what should be LOOKED AT, and a `similarity` is a discovery signal rather than an estimate. An empty result with `recommendations_basis.reason` is a valid answer.
- **3 — propose, then predict with ONE branch.** You name the candidates; no directory exists to enumerate. Branch A: `predict-strategy` once per candidate, keep the chosen candidate's `result.prediction_id`. Branch B: `plan-next`, then `choose-next --decision <result.decision_action_id> --chosen c.json`, then reuse the chosen candidate's id from `result.plan.candidates[].prediction_id` (matched on the full identity, solver and config included). `choose-next` belongs to branch B only — branch A writes no planning decision. Use branch B when the candidates are close or the decision is high-stakes; predicting them yourself is the ordinary path. Planning is horizon 1: it compares the candidates once, and you re-plan from the REAL observation of the chosen step.
- **4 — model, then code.** Write the representation into the task's `model` field AFTER the strategy is chosen, then write `solve.py` — the framework generates neither. Every constraint label must be `C1`, `C2`, `C3`, … (a different naming is the most common L2 failure). The sandbox blocks `subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil` and `pathlib`, and restricts `os` to `os`/`os.path` — but **importing a solver library (ortools, highspy, …) is allowed**, and the answer goes to a literal `open('result.json', 'w')`, which resolves inside `--workspace`.
- **5 — execute with the prediction.** `execute --prediction <prediction_id>` binds the action to that prediction automatically once it ends, checking task/episode/strategy/solver/config and recording a mismatch on `result.prediction_binding` instead of scoring it. Omit `--prediction` only when you never predicted the candidate, then bind later with `bind-strategy`. Every attempt is staged automatically, success or failure, and a prediction never becomes a fact.
- **6 — check the ANSWER.** `check-task <execution_id>` covers the TASK, while `execute`'s own verdict covers the solver's MODEL; `passed` covers the DECLARED bases only, never proof that the model is right. Declare the bases you can support (`reference_objective`, `reference_status`, `integer`, `recompute_objective`, `semantic_probe`, `intent`) — a wrong reference is not detected for you, and an undeclared basis is simply not checked. On `failed`, record this attempt with its real cost and go back to **step 3**: predict the corrected candidate, because a retry needs its OWN prediction and its OWN check and the earlier failure does not carry over. On `insufficient`, the validity is UNKNOWN and not a pass: declare the missing basis, or have `solve.py` report `variables`. Either way the attempt is evidence — never dropped, and you never change the task to match a reference value.
- **7 — record the fact and the real cost.** `record --from-staged <execution_id> --override llm_tokens=<actual>`, and the same for timeouts, infeasibilities and failures. `llm_tokens`, `tool_calls` and `retries` are invisible to the sandbox, which measures only `latency_s` and `solver_runtime_s`: declare them here or with `amend-cost`, or that dimension stays unmeasured. `retries` means the NEW retries THAT attempt represents — a first attempt records `retries=0` as an observed fact, and the attempt that retried it declares `--override retries=N`.
- **8 — finish the episode.** `close-episode` is idempotent, reads what was recorded, publishes the calibration and reports `task_checks` coverage — a close-out is not a certification of the answer. Pick `--terminal` to match what happened: `completed` after a passed check or a correctly identified infeasibility, `failed` when the answer was confirmed not to satisfy the task, `aborted` when you stopped, `budget_exhausted` when the budget ran out. A `state: "pending"` refusal means an action is still `running` — end it first.

## 5. Offline operations

Induction never runs automatically and never inside an episode. `induce` is the only place knowledge changes, and `accept-capability` is the only command that runs a maintenance operation; the `induction_hints` you get from `record` are attention signals, not inductions.

| # | Action | When | Output (JSON path) |
|---|---|---|---|
| 1 | `orx induction-candidates` | you are considering a maintenance decision | `result.bundles[]` — frozen evidence packages, no model call |
| 2 | `orx predict-capability --operation op.json --bundle <bundle>` | a bundle exists | `result.prediction_id` |
| 3 | `orx compare-capability --predictions hp_1,hp_2 --horizon-tasks 10` | two or more predictions exist | `result.recommended` |
| 4 | `orx accept-capability --recommendation <json>` or `orx reject-capability ...` | you have decided | `result.maintenance_binding`; accept also runs the operation |
| 5 | `orx evaluate-capability --prediction hp_1` | qualified LATER tasks have accumulated | `result.effect_verified` |
| — | `orx induce [--verify '<json>']` | you are writing knowledge directly, without the capability-prediction path | `result.results[]` — created / updated / skipped |

**Entry condition, and what to do when evidence is thin.** Creating an entry needs ≥2 executions from ≥2 DISTINCT tasks, and publishing it needs a passed admission check (`induce --verify`). When `induction-candidates` reports none, keep solving and recording until the evidence exists, or submit a structured relation claim instead — `orx induce --relation '{...}' --verify '{"purpose":"relation",...}'`, for knowledge that is not one strategy's statistics. Patterns, thresholds and the relation format: [references/induction.md](references/induction.md).

**Accepting, and when the effect is judged.** `accept-capability` already ran the operation and bound the maintenance fact on `result.maintenance_binding`, so do not call `induce` or `retire` again for the same decision — `induce` is for writing knowledge directly (with `--verify` to publish it), and the two are alternative paths to the same write. `retire` is irreversible and `induce --force` lifts a cold-archive veto once. A knowledge entry appearing, and a passed verification, are not capability improvement — only `evaluate-capability` over real later tasks can support that, counted in TASK-EPISODES with work that really ran AFTER the operation.

**Storage and repair.** `inspect --bank retention` shows the window, grace period and archive caps; `archive-calibration --dry-run` previews without writing; `calibration --rebuild` repairs after a late correction. `exclude-execution` / `restore-execution` withdraw and reinstate a fact without deleting it — the row survives and stops counting, and a late task check re-derives the benefit while keeping the measured cost.

## 6. Reference index

Read on demand, one hop, no chains: the workflow above is complete on its own.

| Read it when | Reference |
|---|---|
| You are about to call a command and need its flags, return states or failure handling | [references/commands.md](references/commands.md) |
| You are writing your first model or CIR | [references/modeling.md](references/modeling.md) |
| You want the "why": coupling-aware understanding, the two-layer memory, strategy cost, how the world model fits in | [references/concepts.md](references/concepts.md) |
| You are deciding whether to induce, or submitting a structured relation claim | [references/induction.md](references/induction.md) |
| You are debugging what a prediction was conditioned on | [references/prediction_context.md](references/prediction_context.md) |
| You are predicting or comparing candidates under wm-so/1 | [references/strategy_outcome.md](references/strategy_outcome.md) |
| You need a protocol field definition, a status value, or a legacy mapping | [references/world_model_contract.md](references/world_model_contract.md) |
| You are closing an episode, reading the calibration, or handling a late verdict | [references/episode_closeout.md](references/episode_closeout.md) |
| You want a runnable script to read | `references/examples/`: `no_catalog.py` (cold start), `strategy_outcome.py` (predict two candidates and bind the real execution), `task_check.py` (check → diagnose → repair → re-check), `episode_closeout.py` (evaluate → calibrate → retain), `capability_evolution.py` (the whole offline path), `prediction_context.py` and `contract_roundtrip.py` (frozen inputs and contract shapes) |

Every example runs with no model, no network and no credentials by injecting a fixed-output stub provider, so what each one demonstrates is the PROCESS, never model accuracy. The ids they produce are the same ones the table above passes between commands.
