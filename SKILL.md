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
- **Memory directory.** `--home DIR`, else `$OR_HARNESS_HOME`, else `./or_harness_home`. Every fact, entry, prediction and calibration lives there, so pass the same directory to every call in one episode.
- **World model.** Required (see the top of this file). Both `predict-strategy` and `plan-next` charge their model calls to the episode budget. If no provider is available, prediction returns `not_configured` and you cannot run the loop to spec: `execute` still solves without `--prediction`, but nothing is then forecast, bound or calibrated — treat that as a configuration gap, not a normal path.

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

Run these in order. Steps 3a and 3b are alternatives — doing both predicts the same candidate twice.

| # | Action | Input | Output (JSON path) | Next |
|---|---|---|---|---|
| 1 | `orx profile --task t.json` | task JSON, optional `--cir` | `result.coupling`, `result.profile`, `result.derivation` | 2 |
| 2 | `orx recall --task t.json --top 3` | task | `result.recommendations[]`, `result.held_claims[]`, `result.vector_recall` | 3a or 3b |
| — | `orx context --task t.json` (optional) | task, for a stored frozen input | `result.context_id` | pass it to 3a as `--context`; `predict-strategy` builds its own when omitted |
| 3a | `orx predict-strategy --task t.json --candidate c.json [--episode ep1]` | task + one candidate, repeated per candidate | `result.prediction_id` | 4, keeping the chosen candidate's id |
| 3b | `orx plan-next --task t.json --episode ep1 --candidates cs.json` | task + candidate list | `result.decision_action_id`, `result.plan.candidates[].prediction_id` | 3b′ |
| 3b′ | `orx choose-next --decision <decision_action_id> --prediction <prediction_id>` | the decision id + the chosen candidate's prediction id from 3b | `result.selected_plan` | 4, executing with that same prediction id |
| 4 | Write the model, then `solve.py` | the chosen strategy | your script's `result.json` (`status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and `variables` if a check will need the answer) | 5 |
| 5 | `orx execute --task t.json --prediction <prediction_id> --code solve.py --workspace ws` | task + the prediction id from 3a or 3b (strategy/solver come from the candidate) | `result.execution_id`, `result.prediction_binding` | 6 |
| 6 | `orx check-task <execution_id> --check '{...}'` | the execution id from 5 | `result.verdict` | 7, or the retry loop |
| 7 | `orx record --from-staged <execution_id> --override llm_tokens=<actual>` | the execution id from 5 | `result.recorded`, `result.induction_hints` | 8, or back to 3a/3b |
| 8 | `orx close-episode --task T --episode E --terminal STATE` | task + episode | `result.task_checks` | offline operations |

- **1 — understand.** Put your coupling understanding in the task's `coupling` field. A malformed CIR is a precondition error (exit 2), never a silent empty one; `--allow-empty-cir` accepts one with no structure. A `coupling_warnings` entry means your supplied value contradicts the derivation, and the derived value wins for grouping.
- **2 — recall is a report, not a menu.** `recommendations` says what may be REUSED (a published knowledge claim reaches you here), `held_claims` names applicable claims NOT yet published (candidates in waiting), `vector_recall` what should be LOOKED AT, and a `similarity` is a discovery signal rather than an estimate. An empty result with `recommendations_basis.reason` is a valid answer.
- **3 — propose, then predict with ONE branch.** You name the candidates; no directory exists to enumerate. Describe each candidate's METHOD — a short name and its steps — because the model predicts from that description, and a bare strategy id or solver name tells it nothing. Put the approach in the candidate's `method` field and the execution parameters (`time_limit`, `mip_gap`, `seed`, ...) in `config`: a `method` mistakenly placed in `config` is moved to `method` and recorded, and a key that names no execution parameter is refused before an attempt is spent. Every candidate is predicted under ONE shared benefit convention (`kind`/`metric`/`unit`/`scope`), so the comparison reads one currency. The convention is **declared per decision** (`--benefit-kind`/`--benefit-metric`), otherwise it is taken from the candidates' own agreement, else it defaults to `solution_quality`/`normalized_objective_gap`. `solution_quality` measures how well the SOLVER solved the model; `effective_completion`/`task_result_check_passed` measures whether the ANSWER satisfies the TASK — a different measurement, comparable only with others of its own kind. A candidate whose benefit is in a currency the decision did not adopt is reported and **does not enter the full ranking** (an unread upside is never scored as zero); a declared convention this build has no scale for ranks nothing at all rather than inventing one. Branch A: `predict-strategy` once per candidate, keep the chosen candidate's `result.prediction_id`. Branch B: `plan-next`, then `choose-next --decision <result.decision_action_id> --prediction <the chosen candidate's prediction_id>` (the candidate is read from the plan's own comparison — you do not re-type its JSON). `choose-next` belongs to branch B only — branch A writes no planning decision. Use branch B when the candidates are close or the decision is high-stakes; predicting them yourself is the ordinary path. Planning is horizon 1: it compares the candidates once, and you re-plan from the REAL observation of the chosen step. If a prediction comes back `invalid`, read `result.failure.kind` before retrying: `truncated` means the output budget was too small (`finish_reason` says `length`), `wrong_top_level` means the model answered with an array instead of an object, `empty_response` means it produced no text, and `timeout`/`network_error` mean the call did not complete — each calls for a different fix, and `finish_reason: null` means the endpoint reported nothing, so how it ended is UNKNOWN.
- **4 — model, then code.** Write the representation into the task's `model` field AFTER the strategy is chosen, then write `solve.py` — the framework generates neither. Every constraint label must be `C1`, `C2`, `C3`, … (a different naming is the most common L2 failure). The sandbox blocks `subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil` and `pathlib`, and restricts `os` to `os`/`os.path` — but **importing a solver library (ortools, highspy, …) is allowed**, and the answer goes to a literal `open('result.json', 'w')`, which resolves inside `--workspace`.
- **5 — execute with the prediction.** `execute --prediction <prediction_id>` establishes the association BEFORE the run: the frozen candidate supplies the strategy/solver/episode (do not re-type them; a conflicting explicit argument is refused before anything executes), the prediction's status, problem identity and claim are checked, and a prediction already consumed by another attempt is refused. After the run the real result AND the observed configuration are added to the same association. `result.prediction_binding` reports `bound`, `config_observed` (candidate config keys the run really confirmed) and `config_unknown` (keys the run never reported — a missing receipt is a caveat, not a discard). If the executor is interrupted before producing any record, the association is released so the same prediction can be tested against a real attempt. Omit `--prediction` only when you never predicted the candidate, then bind later with `bind-strategy`. Every attempt is staged automatically, success or failure, and a prediction never becomes a fact. **Record the METHOD, not just the outcome**: the candidate's `method` (or `execute --method '{...}'`) is the PLAN; have `solve.py` write a `method_performed` object stamped with `$OR_ACTION_ID` to report what ACTUALLY ran. Without it the performed method stays unknown — a plan is never promoted to a fact, and `induction-material` will report the evidence as method-less.
- **6 — check the ANSWER.** `check-task <execution_id>` covers the TASK, while `execute`'s own verdict covers the solver's MODEL; `passed` covers the DECLARED bases only, never proof that the model is right. Declare the bases you can support (`reference_objective`, `reference_status`, `integer`, `recompute_objective`, `semantic_probe`, `intent`) — a wrong reference is not detected for you, and an undeclared basis is simply not checked. The verdict also becomes an OBSERVATION: a prediction that declared `effective_completion / task_result_check_passed` is scored against it at close-out (1.0 `passed`, 0.0 confirmed `failed`, and an unchecked or `insufficient` execution stays UNKNOWN rather than being counted as a failure). A `failed` verdict is also a **prior failure** for the repair detector: a model written wrong and solved to a legal optimum is the main modeling error to summarize, so "answer wrong → fix the model → answer right" is detected even though `quality.feasible` stayed true. On `failed`, record this attempt with its real cost and go back to **step 3**: predict the corrected candidate, because a retry needs its OWN prediction and its OWN check and the earlier failure does not carry over. On `insufficient`, the validity is UNKNOWN and not a pass: declare the missing basis, or have `solve.py` report `variables`. Either way the attempt is evidence — never dropped, and you never change the task to match a reference value.
- **7 — record the fact and the real cost.** `record --from-staged <execution_id> --override llm_tokens=<actual>`, and the same for timeouts, infeasibilities and failures. `llm_tokens`, `tool_calls` and `retries` are invisible to the sandbox, which measures only `latency_s` and `solver_runtime_s`: declare them here or with `amend-cost`, or that dimension stays unmeasured. `retries` means the NEW retries THAT attempt represents — a first attempt records `retries=0` as an observed fact, and the attempt that retried it declares `--override retries=N`. A record you assemble yourself may declare a method with `--method` (the plan) and `--method-actual` (what you really observed); a value the record already carries is never overwritten.
- **8 — finish the episode.** `close-episode` is idempotent, reads what was recorded, publishes the calibration and reports `task_checks` coverage — a close-out is not a certification of the answer. Pick `--terminal` to match what happened: `completed` after a passed check or a correctly identified infeasibility, `failed` when the answer was confirmed not to satisfy the task, `aborted` when you stopped, `budget_exhausted` when the budget ran out. A `state: "pending"` refusal means an action is still `running` — end it first.

## 5. Offline operations

Induction never runs automatically and never inside an episode. `induce` is the only place knowledge changes, and `accept-capability` is the only command that runs a maintenance operation; the `induction_hints` you get from `record` are attention signals, not inductions.

| # | Action | When | Output (JSON path) |
|---|---|---|---|
| 1 | `orx induction-candidates` | you are considering a maintenance decision | `result.candidates[]` — frozen evidence packages (detector-derived and cell-derived), no model call |
| 1′ | `orx induction-material [--bundle ID] [--pattern P]` | a candidate exists | the methods, the comparison evidence and what followed — read it, then form the claim yourself |
| 2 | `orx induce --relation '{...}' [--verify ...]` | you have read the material and formed a claim | `result.relations[].publication` — the framework checks what you submitted; ONE entry is ONE claim |
| 3 | `orx predict-capability --operation op.json --bundle <bundle>` | a bundle exists and you want to price the operation | `result.prediction_id` |
| 4 | `orx compare-capability --predictions hp_1,hp_2 --horizon-tasks 10` | two or more predictions exist | `result.recommended` |
| 5 | `orx accept-capability --recommendation <json>` or `orx reject-capability ...` | you have decided | `result.maintenance_binding`; accept also runs the operation |
| 6 | `orx evaluate-capability --prediction hp_1` | qualified LATER tasks have accumulated | `result.effect_verified` |
| — | `orx induce [--verify '<json>']` | you are writing statistical knowledge directly, without the capability-prediction path | `result.results[]` — created / updated / skipped |

**Method material first, technique second.** `induction-candidates` freezes the evidence; `induction-material` tells you what it says — the methods actually used, the change (for an intervention), both sides of a comparison, and what followed. A candidate reports `material=insufficient` when there is nothing to abstract (no method at all, or only PLANNED methods — a plan is intent, not a performed method) and `material=unavailable` when the comparison lost a whole side to an exclusion or a later task check refuted the premise: the framework will not turn a strategy name and a mean into a technique, and the right response is to record how the work was really done (the script's `method_performed` receipt; `execute --method` is only the plan) or re-check the evidence. `purpose` says what the candidate is FOR: `method_induction` (a comparison/recovery worth abstracting) vs `statistical_refresh` (a cell that cleared the count gate — its claim is its statistics, not a technique). Form the claim yourself (condition → how → consequence → boundary), then submit it with `induce --relation`, citing the executions (or the `bundle_id`) you were shown. The framework organizes the material and checks what YOU write; it never writes the technique for you and never calls a model.

**Accepting a maintenance operation runs what it DECLARED.** When you go the prediction route (`predict-capability` → `compare-capability` → `accept-capability`), put the relations the operation forms in the operation's `config["relations"]` (each an `induce --relation` payload). Acceptance then submits exactly those claims — the technique reaches the bank instead of only a cell mean. With none declared, the statistical refresh runs as before. A real modeling repair is detected even when the solver was happy: a prior attempt counts as failed when the answer FAILED its task check, and the fix is recognised from a differing PERFORMED method (a plan alone is not an intervention).

**Entry condition, and what to do when evidence is thin.** Creating a statistical entry needs ≥2 executions from ≥2 DISTINCT tasks, and publishing it needs a passed admission check (`induce --verify`). When `induction-candidates` reports none, keep solving and recording until the evidence exists, or submit a knowledge CLAIM instead — `orx induce --relation '{...}' --verify '{"purpose":"relation",...}'`, for knowledge that is not one strategy's statistics. A claim submission is a knowledge write like any other: it records a maintenance action, a knowledge delta and the index result, so it is as auditable as a statistical induction and re-submitting the same claim revises that entry rather than duplicating it. ONE entry is ONE claim: two independent claims never share an entry or a verdict. Patterns, thresholds and the claim format: [references/induction.md](references/induction.md).

**Accepting, and when the effect is judged.** `accept-capability` already ran the operation and bound the maintenance fact on `result.maintenance_binding`, so do not call `induce` or `retire` again for the same decision — `induce` is for writing knowledge directly (with `--verify` to publish it), and the two are alternative paths to the same write. `retire` is irreversible and `induce --force` lifts a cold-archive veto once. A knowledge entry appearing, and a passed verification, are not capability improvement — only `evaluate-capability` over real later tasks can support that, counted in TASK-EPISODES with work that really ran AFTER the operation.

**Storage and repair.** `inspect --bank retention` shows the calibration window, late-check grace period and archive caps; `archive-calibration --dry-run` previews without writing; `calibration --rebuild` repairs after a late correction. The RAW evidence is bounded separately by the **evidence window**: `enforce-window` keeps a recent window of complete episodes (default 800) and runs LAST after a close-out, so calibration and induction have already had their chance. eviction is a source reference EXPIRING, never a refutation and never a withdrawal — `exclude-execution` / `restore-execution` are the explicit correction channels, and a late task check re-derives the benefit while keeping the measured cost. `migrate-relations` is the one-way, idempotent upgrade of a store written before claims were unified.

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
| You want a runnable script to read | `references/examples/`: `no_catalog.py` (cold start), `strategy_outcome.py` (predict two candidates and bind the real execution), `task_check.py` (check → diagnose → repair → re-check), `method_evidence.py` (record the method → detect a same-solver fix → read the material → submit a relation → recall it), `episode_closeout.py` (evaluate → calibrate → retain), `capability_evolution.py` (the whole offline path), `prediction_context.py` and `contract_roundtrip.py` (frozen inputs and contract shapes) |

Every example runs with no model, no network and no credentials by injecting a fixed-output stub provider, so what each one demonstrates is the PROCESS, never model accuracy. The ids they produce are the same ones the table above passes between commands.
