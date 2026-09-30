---
name: or-harness-wm
description: >
  Formulate, solve, verify, retry and learn from large-scale coupled
  industrial optimization problems (LP, MILP, scheduling, routing,
  assignment, network flow, capacity planning, resource allocation, supply
  chain). Use when the user asks which solving strategy or formulation to
  use; to compare candidates before committing on predicted quality, cost or
  risk; to find out why a solver that reports "optimal" still gives an answer
  the task forbids; to record an attempt's real cost; or to learn from
  completed tasks. Built for repeated or follow-up optimization work on the
  same task family, where earlier attempts, their real costs and their
  outcomes are worth carrying forward. Do NOT use for a one-off optimization
  question, a pure mathematical proof, or non-optimization work.
---

# OR-Harness: strategy learning for optimization agents

You are the orchestrator; this layer advises, executes and remembers. Each command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`): no hidden state, no background work. Every command prints ONE JSON line — `{"result": {...}, "summary": "..."}` — and exits `0` ok, `2` usage/precondition, `1` crash. You may refuse any recommendation, execute without recording, and you alone decide when to consolidate.

**A world-model provider is REQUIRED** (`--world-model URL::MODEL`, key from `$OR_WM_API_KEY`, or `ORHarness(world_model=...)`). Without one, prediction is unavailable and the loop cannot run to spec: the prediction commands exit `2` and nothing is forecast, bound or calibrated. That is a configuration error, never a quiet fallback to prediction-free operation.

## 1. What you deliver

A solved AND validated task, plus the evidence that makes the next one cheaper: the answer, its task-level verdict, the real cost of every attempt, and the recorded facts later calibration and induction read back. The task is done when every attempt you want counted has been checked against the TASK, recorded with its measured cost, and its episode closed. Two endings are legitimate: a verified answer, or a correctly identified infeasibility.

## 2. The whole loop

```text
  UNDERSTAND + RECALL   profile, recall -> CIR + profile + memory report
            |
  PROPOSE      YOU name the candidate methods; there is no menu to enumerate
            |
  PREDICT + CHOOSE     ONE branch, never both:
            |            A) predict-strategy per candidate -> keep that id
            |            B) plan-next -> choose-next --prediction <id>
            v (a failed task check returns here with the candidate fixed)
  MODEL -> EXECUTE --prediction <id> -> CHECK TASK -> RECORD
            |   association made BEFORE the run;                      (facts + real cost,
            |   result and observed config added AFTER it              failures included)
            v
  FINISH       close-episode --task T --episode E --terminal STATE
            |
      +-----+-----------------------------------------------------+
      v                                                           v
  CALIBRATION (automatic)                     INDUCTION (optional, offline)
  re-scores predictions against real          abstracts knowledge from cross-task
  outcomes; feeds LATER predictions           evidence; its effect is judged later
                                              by evaluate-capability
```

**Your decisions, in order.** The workflow is one pass of this list; only step 6 may send you back.

1. **Understand and recall.** Validate the task's structural key, then read what memory really holds. Recall is a report, not a menu.
2. **Propose.** Write the candidate methods yourself — name and steps. There is no built-in catalog, and there never will be.
3. **Predict, then choose.** ONE branch (A or B below). A candidate's method is described before it is predicted.
4. **Model, then code.** Only after the strategy is chosen. The formulation is your judgement.
5. **Execute with the prediction, then check the ANSWER.** The prediction is bound to the attempt before the run.
6. **Record, then close the episode** — including failures. If the task check failed, fix and go back to step 3 with a NEW prediction.
7. **Consolidate offline** if the evidence is worth abstracting. Never inside an episode.

## 3. Before you start

- **Task identity.** A JSON object with `task_id` and `family` at minimum; `text` (so the text channel can index it) and `coupling` (a CIR) when the decisions are coupled. **`task_id` is yours to choose and must be stable**: one logical problem keeps ONE id however often you solve it (a retry, another solver, a bigger time limit), while an episode id (`--episode`, e.g. `ep1`) groups the attempts of one solving effort. **A DIFFERENT problem is a different `task_id`, even when it looks similar** — that is what makes the second task's evidence independent, which is what induction and calibration need. A task with no `model` field is normal — the model is written at step 4, after the strategy is chosen. Required fields, the CIR shape and the spec numbers: [references/modeling.md](references/modeling.md).
- **One memory directory.** Pass the same `--home` to every call in an episode. Everything — facts, entries, predictions, calibration — lives there.
- **Recall may be empty, and that is normal.** No memory matched, so you propose and describe the candidates yourself. Distinguish that from the text channel being OFF (no embedding model configured): with none, `recall` reports `degraded` and only the structural channel ran — a different fact from "ran and matched nothing". Configuring one and running `orx rebuild-index` once enables it.

## 4. Online operations

Steps 3a and 3b are alternatives — doing both predicts the same candidate twice. Every `result.*` path below is a TOP-LEVEL key of the command's JSON line.

| # | Action | Input | Read from the result | Next |
|---|---|---|---|---|
| 1 | `orx profile --task t.json` | task JSON, optional `--cir` | `result.coupling`, `result.profile`, `result.derivation` | 2 |
| 2 | `orx recall --task t.json --top 3` | task | `result.recommendations[]`, `result.recommendations_basis`, `result.held_claims[]`, and `result.vector_recall` **only when the text channel is configured** | 3a or 3b |
| 3a | `orx predict-strategy --task t.json --candidate c.json [--episode ep1]` | task + one candidate, repeated per candidate | `result.prediction_id`, `result.prediction.status` | 4, keeping the chosen candidate's id |
| 3b | `orx plan-next --task t.json --episode ep1 --candidates cs.json` | task + candidate list | `result.decision_action_id`, `result.plan.candidates[].prediction_id`, `result.plan.suggested` | 3b′ |
| 3b′ | `orx choose-next --decision <id> --prediction <id>` | the decision id + the chosen candidate's prediction id | `result.selected`, `result.deviation` | 4, with that same prediction id |
| 4 | Write the model, then `solve.py` | the chosen strategy | your script's `result.json`: `status`, `objective_value`, `objective_bound`, `runtime_seconds`, plus `variables` if a check will need the answer | 5 |
| 5 | `orx execute --task t.json --prediction <prediction_id> --code solve.py --workspace ws` | task + the prediction id (strategy/solver come from the candidate) | `result.execution_id`, `result.prediction_binding` | 6 |
| 6 | `orx check-task <execution_id> --check '{...}'` | the execution id | `result.state`, `result.report.conclusion`, `result.report.scope.unchecked`, `result.next` | 7, or back to 3 |
| 7 | `orx record --from-staged <execution_id> [--usage-file usage.json]` | the execution id; the host's usage report if you have one | `result.recorded`, `result.cost_completeness`, `result.induction_hints` | 8, or back to 3a/3b |
| 8 | `orx close-episode --task T --episode E --terminal STATE` | task + episode | `result.task_checks`, `result.evaluations[]`, `result.calibration_summary` | offline operations |

Three rules govern the whole pass. Everything else per step is a pointer.

- **The candidate is a PLAN, the prediction is an EXPECTATION, the execution is the FACT.** A method must be DESCRIBED (`method: {"name", "steps"}`) before it is predicted — a bare strategy id tells the model nothing, so an undescribed candidate is unanswerable. Put execution parameters (`time_limit`, `mip_gap`, `seed`) in `config`, not in `method`. A prediction never becomes a fact, and every attempt is staged automatically, success or failure.
- **One benefit convention per decision, and never score an unread field as zero.** The candidates are compared in ONE `kind`/`metric`/`unit`, else the candidates' own agreement, else `solution_quality`/`normalized_objective_gap`. `solution_quality` measures how well the SOLVER solved the model; `effective_completion`/`task_result_check_passed` measures whether the ANSWER satisfies the TASK — different measurements, comparable only within their own kind. A missing or unsupported prediction field is UNKNOWN, never zero. Protocol and failure kinds: [references/strategy_outcome.md](references/strategy_outcome.md).
- **A solver verdict is not a task verdict.** `execute`'s own status covers the solver's MODEL: an LP relaxation answered with fractional values is `optimal` with `gap=0` and still wrong. `check-task` covers the TASK, and it only evaluates the bases YOU declare — an undeclared basis is not checked, and `result.report.scope.unchecked` names what was left out. **The basis you declare is how a real-world rule (a per-vehicle limit, whole units, a reference objective) becomes checkable at all**; the basis list and its JSON are in [references/commands.md](references/commands.md), and how a verdict scores into the calibration is in [references/episode_closeout.md](references/episode_closeout.md).

Per-step rules you must not skip:

- **2 — recall.** A recommendation is a hint to TEST, not a decision. If you cannot see why it applies (no applicability, thin support, unverified claim), treat it as one candidate among several. What recall reads and how the channels differ: [references/concepts.md](references/concepts.md).
- **3 — pick ONE branch, and keep the id.** Branch A (ordinary): `predict-strategy` once per candidate, then keep the chosen candidate's `result.prediction_id`. Branch B (candidates close, or the decision is costly): `plan-next` then `choose-next`. `choose-next` belongs to branch B only — branch A writes no planning decision, and planning is horizon 1 (a different `--horizon` is refused; re-plan from the REAL observation instead). If a prediction comes back not `valid`, read `result.failure.kind` before retrying.
- **5 — execute, and the binding is checked.** `--prediction` supplies the strategy, solver and episode: do not re-type them, and a prediction already consumed by another attempt is refused. **Every attempt needs a prediction**: if you never predicted the candidate, do not invent a binding later — predict it, or run without `--prediction` only when you accept the attempt will not be calibrated. `result.prediction_binding` reports `bound`, `config_observed` (keys the run confirmed) and `config_unknown` (keys it never reported — a missing receipt is a caveat, not a discard). If the executor is interrupted before producing a record, the attempt is staged as its OWN failure fact and the association is RELEASED, so the SAME prediction can be tested against the retry.
- **6 — the three verdicts, and what each means for you.** `passed` = every DECLARED basis held (never proof the model is right). `failed` = the answer is confirmed wrong: record this attempt with its real cost, fix the model, and go back to step 3 — a retry of a CONSUMED attempt needs its OWN prediction and its OWN check. `insufficient` = validity is UNKNOWN, **not a pass**: declare the missing basis, or have `solve.py` report `variables`. Never change the task to match a reference value. On `failed`, `result.reflection_material` carries what you need to locate the cause, and locating it is your job.
- **7 — record every attempt, failures included.** The sandbox measures only `latency_s` and `solver_runtime_s`; `llm_tokens`, `tool_calls` and `retries` are yours to supply. **Prefer the REAL observation**: a host usage report (`--usage-file`) is a real measurement and may stand as a calibration truth; a bare `--override llm_tokens=<count>` is a DECLARATION that is shown but **never** used as a calibration actual, a cost claim or a learning mean — pass `--override-source agent_observed`/`provider_usage` when the number really came from a report, and treat a single-side (completion-only) figure as a lower bound, never a complete total. A first attempt's `retries=0` is an observed fact; the attempt that retried it declares `--override retries=N`. The accepted report shape, `amend-cost`, and the overwrite guard: [references/commands.md](references/commands.md).
- **8 — close the episode.** `--terminal` must match what happened (`completed`/`failed`/`aborted`/`budget_exhausted`). The close-out reads what was RECORDED — a close-out is not a certification of the answer; only `check-task` speaks to that. A `state: "pending"` refusal means an action is still `running`. Late verdicts and re-publishing: [references/episode_closeout.md](references/episode_closeout.md).

**Episode boundary.** Nothing about the world model changes inside an episode: predictions and facts accumulate, and calibration runs at `close-episode` from the real, complete outcomes, serving LATER tasks. A capability gain a prediction mentions is explanatory — never an immediate training signal and never a verified improvement. Every strategy-outcome prediction's H+ block must STATE a stance (`assessment`: `expected`/`none`/`insufficient_basis`); "no gain" and "cannot judge" are honest answers, silence is not, and `inspect --bank capability` counts each stance so a silent model is visible.

## 5. Offline operations

Induction never runs automatically and never inside an episode. `induce` is the only place knowledge changes, and `accept-capability` is the only command that runs a maintenance operation; the `induction_hints` from `record` are attention signals, not inductions.

**Direct route** — abstract the knowledge yourself:

```text
orx induction-candidates  ->  orx induction-material  ->  YOU write the claim
                                                          ->  orx induce --relation '{...}'
```

`induction-candidates` freezes the evidence packages; `induction-material` says what they show (the methods actually used, both sides of a comparison, what followed). **You form the claim** (condition → how → consequence → boundary) and submit it; the framework checks what you wrote, derives the identity, and returns `result.relations[].publication`. A planned method can never masquerade as a performed one — without a `method_performed` receipt the material is `insufficient` and there is nothing to abstract. Creating a statistical entry needs ≥2 executions from ≥2 DISTINCT tasks, and publishing needs a passed admission check.

**Prediction route** — price the operation first, when the decision is worth the model call:

```text
orx predict-capability -> orx compare-capability -> accept- / reject-capability
                                                          -> (later) evaluate-capability
```

`compare-capability` returns `result.recommendation` (`accept`/`defer`) plus every candidate it could not rank. **Use ONE route per decision, never both** — accepting already ran the operation, so calling `induce` again for the same decision is a second write. `evaluate-capability` is a separate, later judgement over REAL later tasks: it is the only thing that can support "the harness got stronger", and a knowledge entry appearing or passing its verification is not that.

Patterns, thresholds, material states and the claim format: [references/induction.md](references/induction.md).

## 6. Who decides what

| The framework does automatically | You decide |
|---|---|
| Stages every attempt (success and failure), measures `latency_s`/`solver_runtime_s`, proves a first attempt's `retries=0` | the candidate methods, their `method` description, and which one to run |
| Binds the prediction to the real run; keeps `config_observed` and `config_unknown` apart | the real token/tool-call figures and their SOURCE |
| Evaluates every bound prediction at `close-episode` and publishes the calibration | the check bases you can honestly support, and the `--terminal` state |
| Re-derives a late task check; excludes or includes samples by evidence eligibility | whether to consolidate offline, and which knowledge claim to submit |
| Bounds the evidence window — eviction is a reference expiring, never a refutation | whether a result is good enough, whether to retry, and on what grounds |

## 7. Read when needed

One hop, no chains: the workflow above is complete on its own, and the rest is at the moment you need it.

| Read it when | Reference |
|---|---|
| `recall` is empty, or a recommendation matched but you cannot see why it applies | [references/concepts.md](references/concepts.md) |
| You are writing or fixing a model, a CIR, or `solve.py` | [references/modeling.md](references/modeling.md) |
| A prediction is `not valid`, a field is missing, or you are comparing candidates | [references/strategy_outcome.md](references/strategy_outcome.md) |
| You are closing an episode, reading the calibration, or handling a late verdict | [references/episode_closeout.md](references/episode_closeout.md) |
| You are deciding whether to induce, or submitting a knowledge claim | [references/induction.md](references/induction.md) |
| You want to know what a prediction was conditioned on | [references/prediction_context.md](references/prediction_context.md) |
| You need a protocol field, a status value, or a legacy mapping | [references/world_model_contract.md](references/world_model_contract.md) |
| You are about to call a command and need its flags, return states or failure handling | [references/commands.md](references/commands.md) |
| You want a runnable script | [references/examples/end_to_end.py](references/examples/end_to_end.py) — cold start through `close-episode`, including a `failed` check and the repair. Others in `references/examples/` cover one path each. If the same command fails twice, read the reference above rather than retrying variations. |
