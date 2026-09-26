---
name: or-harness-wm
description: >
  Formulate, solve, retry, decompose, validate and debug large-scale industrial
  optimization problems (LP, MILP, scheduling, routing, assignment, network
  flow, capacity planning, resource allocation, supply chain) whose decisions
  are highly coupled through shared resources, cross-stage dependencies or
  temporal propagation; and run the learning loop around them. Use when the
  user asks to formulate, solve, retry, decompose, validate or debug an
  optimization model; to compare candidate solving strategies by predicted
  quality/cost/risk before committing; to retrieve or manage accumulated
  execution evidence and reusable strategic knowledge; to check whether a
  solver's answer satisfies the ORIGINAL task; to close out a finished episode
  and read the resulting calibration; or to run offline consolidation
  (induction, capability prediction, adoption) over completed tasks. Provides
  coupling-aware problem understanding, a sandboxed executor with seven solver
  adapters, a two-layer strategy memory, and a world-model consequence
  predictor wired into both the online loop and offline maintenance. Do not use
  for one-off optimization questions with no repetition, generic math proofs,
  or non-optimization tasks.
---

# OR-Harness: strategy learning for optimization agents

You are the orchestrator; this layer advises, executes and remembers. Every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`) — no hidden state, no background work. Every command prints ONE JSON line: `{"result": {...}, "summary": "..."}` (exit `0` ok, `2` usage/precondition, `1` crash). You may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to consolidate.

**The world model is REQUIRED** on both timescales. Configure a provider (`--world-model URL::MODEL`, API key from `$OR_WM_API_KEY`, or `ORHarness(world_model=...)`). With none, prediction returns `not_configured` and the loop cannot run to spec — that is a configuration error, not a degraded mode.

## The ONE prediction path

```
predict-strategy  →  execute --prediction <id>  →  close-episode
```

There is exactly ONE agent-facing prediction protocol (`wm-so/1`) and ONE planning protocol (the same one). `plan-next` calls it for you; when you plan, **reuse the prediction id it already produced** instead of predicting the chosen candidate again. The legacy `predict_outcome` Python API survives only to read old records — no CLI/agent flow uses it.

## Invariants (do not reinterpret these)

```
profile / coupling-aware understanding  BEFORE  detailed formulation
strategy selection                      BEFORE  detailed modelling
recommendation ≠ selection ≠ execution
prediction ≠ fact        hypothetical ≠ observed        unknown ≠ 0
online solving ≠ offline consolidation
no world-model parameter update inside an episode
planning is horizon 1, then re-plan from the REAL observation
Execution Evidence ≠ Strategic Knowledge
knowledge growth ≠ demonstrated capability improvement
predicted cost  ≠  measured cost       online bounds ≠ disk bounds
```

## Online solving (the main path)

```
Problem P
  → profile                     CIR validated + modeling guidance + profile
  → recall                      what memory REALLY holds (structural + text)
  → YOU propose the candidates  no built-in menu exists
  → EITHER predict-strategy per candidate   (you compare and pick)
    OR     plan-next                        (it predicts for you and suggests)
  → choose-next                 ONLY after plan-next: record your selection
  → model the problem           the GAMS-style representation
  → write solve.py  → execute   pass --prediction <id>; staged automatically
  → check-task                  does the ANSWER satisfy the TASK?
  → record                      persist the fact + cost backfill
  → replan / retry / finish
  → close-episode               evaluate the predictions, publish calibration
```

**Step 3 is a fork, not a sequence.** Both branches predict each candidate exactly once; doing both predicts the same candidate twice and produces two competing samples for one decision.

| | **Branch A — compare it yourself** | **Branch B — let the planner compare** |
|---|---|---|
| Predict | `predict-strategy` once per candidate | `plan-next` predicts them all internally |
| Decide | you weigh quality/cost/risk; no framework record is written | `plan-next` suggests; `choose-next` records your explicit choice |
| Prediction id to hand over | the id `predict-strategy` printed | the chosen entry's `result.plan.candidates[].prediction_id` |
| Cost | one call per candidate, charged where you like | one call per candidate, charged to the decision action |

1. **Understand (one step).** `orx profile --task t.json`. Put your coupling understanding in the task JSON's `coupling` field (a CIR). One call returns the validated CIR with `modeling_guidance`, the problem profile, and a per-dimension derivation report. A task with no `model` field is normal: strategy selection needs the task text, the CIR and the profile — never a finished formulation.
2. **Recall.** `orx recall --task t.json --top 3`. This is a REPORT on real memory, **not a menu**. An empty result with `recommendations_basis.reason` is a valid, expected answer (a cold bank says so). Two independent channels: `recommendations` (structural) and `vector_recall` (text). Neither substitutes for the other; a `different_cell` hit is context to READ, never a statistic to apply.
3. **Propose, then predict — pick ONE branch (see above).** **You name the candidates** — there is no directory to enumerate and the framework will not invent one. Branch A: `orx predict-strategy --task t.json --candidate c.json [--episode ep1]` for each, and keep each printed `prediction_id`. Branch B: `orx plan-next --task t.json --episode ep1 --candidates candidates.json`, then `orx choose-next --decision <the action id plan-next returned> --chosen chosen.json`. **`choose-next` belongs to branch B**: it records a choice against a planning decision action, and branch A has no such action to name.
4. **Model the problem** (AFTER the strategy is chosen). Write the GAMS-style representation into the task's `model` field; the framework verifies L1 shape and L2 symbol cross-reference and reports `derivation.model_coupling` as a DIAGNOSTIC. The model is a post-strategy artifact — the profile you retrieved with stays the structural key, and re-running `profile` will not move it.
5. **Write `solve.py`.** The framework never generates code and never supplies a method's action list (there is none to supply — if memory recorded one, `recall` shows it under `recommendation.knowledge.actions`). `result.json` must carry `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds` — and `variables` whenever a task check will need the answer.
6. **Execute, with the prediction.** `orx execute ... --prediction <id>`. The executed action is BOUND to that prediction automatically once it ends (identity checked: task/episode/strategy/**solver/config**; a mismatch is recorded, never scored). Omit `--prediction` only when you never predicted the candidate; `orx bind-strategy` can bind later. Every attempt is staged automatically, success or failure.
7. **Check the ANSWER.** `orx check-task <execution_id> --check '{...}'`. `execute`'s verdict covers the solver's own MODEL; this covers the TASK. Declare the bases that apply (`reference_objective`, `reference_status`, `integer`, `recompute_objective`, `semantic_probe`, `intent`).
   - `passed` covers the DECLARED bases only — not proof the model is right.
   - `failed` → diagnose (misread task? mis-specified model? implementation bug? wrong reference?) and re-solve in the SAME episode. The attempt stays recorded with its real cost.
   - `insufficient` → the validity is UNKNOWN, never a pass. Declare the missing basis, or make `solve.py` report `variables`.
8. **Record.** `orx record --execution <json> --override llm_tokens=<actual>`. `llm_tokens`, `tool_calls` and `retries` are invisible to the sandbox — the executor measures only `latency_s` and `solver_runtime_s`. Supply them here or with `amend-cost`, or the dimension stays unmeasured and no cost claim can be published for it. A failed attempt is backfilled verbatim: `orx record --from-staged <id>`.
9. **Close the episode** — `orx close-episode --task ID --episode EP --terminal STATE`. Idempotent. It reads what was recorded (no solver, no model call) and publishes the calibration. **A close-out is not a certification**: read its `task_checks` coverage.

## When something goes wrong

- **`status=error`, security policy** — the script used a blocked construct (network/shell/pathlib/dynamic `open()`). Use only stdlib and a literal `open('result.json', 'w')`. Subprocess solvers (PuLP's CBC) cannot run in the sandbox — switch to an in-process solver (ortools, highspy). Record the failure with `--from-staged`.
- **`status=error`, traceback** — fix the model or the script and re-execute. Each attempt is its own record, and its `retries` counts only the NEW retries THAT attempt represents: a first failure is `retries=0` (an observed zero, written by the framework); the attempt that RETRIED it declares `--override retries=N`. Never write the retry count onto the attempt that failed.
- **`status=timeout`** — the strategy may be too heavy at this scale. Re-`recall` with `--exclude <strategy>`, try the next candidate, record it.
- **`quality.problems` non-empty** — fix the script and re-execute; the record waits for the check.
- **`check-task` `failed`** — see step 7. Never change the task to match a reference value, and do not assume every mismatch is a modelling error.
- **`close-episode` refuses with `state: "pending"`** — an action of the episode is still `running`. A genuinely stranded one means the process died mid-call.
- **`recall` empty but you expected memory** — read `candidates_with_memory`: an id absent from it was never executed or induced in this cell.
- **A `coupling_warnings` entry at profile time** — your supplied value contradicts the derivation. The derived value wins for grouping.

## What the framework does for you vs what only you can do

| The framework does this automatically | You must do this explicitly |
|---|---|
| Stages every execution (success and failure) | Propose the candidate methods (no menu exists) |
| Binds `execute --prediction` to the real action | Declare the check bases for `check-task` |
| Evaluates bound predictions at close-out | Record the fact (`orx record`) and the terminal state |
| Maintains the retrieval index on `record`/`induce`/`retire` | Make the selection (`choose-next`, when you used the planner) |
| Binds the maintenance fact on `accept-capability` | Diagnose a failed check and re-solve |
| Publishes/republishes the calibration summary | Decide when to consolidate offline |
| Reports `unknown` instead of guessing | Never write a prediction off as a fact |

## Offline consolidation (a separate timescale — never inside an episode)

```
completed episodes
  → close-episode                         evaluate + publish calibration
  → archive-calibration                   move out-of-window DETAIL to the archive
  → (optional) induction-candidates       freeze the evidence package
  → predict-capability                    what would this operation change?
  → compare-capability                    recommend one, or defer
  → accept-capability / reject-capability EXPLICIT choice (accept runs it)
  → evaluate-capability                   judge the EFFECT on LATER real tasks
```

- **Induction never runs automatically.** `orx induce` is the only place knowledge changes; creating an entry needs ≥2 executions from ≥2 DISTINCT tasks, and PUBLISHING it needs a passed admission check (`induce --verify`). An empty `strategy_type`/`actions` field means memory does not record it — record it yourself when you know it.
- **`accept-capability` already ran the operation.** Do NOT call `induce` or `retire` again for the same decision: it dispatches the operation the prediction was about, and it binds the maintenance fact itself.
- **`evaluate-capability` is not optional if you want to claim an improvement.** A knowledge change and a passed verification are NOT capability improvement — only real later tasks (counted in TASK-EPISODES, work must have run AFTER the operation) can support that.
- **Structured relation claims** are for knowledge that is NOT one strategy's statistics (a modelling principle, a necessary condition, a repair pattern): `orx induce --relation '{...}' --verify '{"purpose":"relation",...}'`. See [references/induction.md](references/induction.md).

## Two complete walkthroughs

**A. Online solve with a failure and a retry**

Branch B (planner) is shown; branch A replaces the first three lines with one `predict-strategy` per candidate and keeps each printed id instead.

```bash
orx profile --task t.json                     # CIR + profile + guidance
orx recall --task t.json --top 3              # real memory, maybe empty
orx plan-next --task t.json --episode ep1 \
    --candidates candidates.json              # → result.plan.candidates[].prediction_id
orx choose-next --decision <action_id from plan-next> --chosen chosen.json

# attempt 1: a prediction exists, so hand it over
orx execute --task t.json --strategy S01 --code solve.py \
    --workspace . --solver highs --episode ep1 --prediction sp_...
orx check-task ex_1 --check '{"reference_objective": 10755}'
#   -> failed: diagnose, fix solve.py, re-solve in the SAME episode
orx record --from-staged ex_1 --override llm_tokens=1840
#   attempt 1 is retries=0 — the FIRST failure is an observed zero, not a
#   retry. Do not write retries=1 here: ex_1 never retried anything.

# attempt 2: predict the CORRECTED candidate, then execute it with that id
orx predict-strategy --task t.json --episode ep1 \
    --candidate corrected.json                 # → sp_retry_...
orx execute --task t.json --strategy S01 --code solve.py \
    --workspace . --solver highs --episode ep1 --prediction sp_retry_...
orx check-task ex_2 --check '{"reference_objective": 10755, "integer": {"variables": ["x1","x2"]}}'
#   -> passed: the retry needs ITS OWN verdict; attempt 1's failure does not
#      carry over, and an unchecked retry is not a confirmed success.
orx record --from-staged ex_2 --override llm_tokens=1810,retries=1
#   retries=1 belongs to the ATTEMPT THAT RETRIED (ex_2), and it is the
#   absolute number of NEW retries this attempt represents.
orx close-episode --task t1 --episode ep1 --terminal completed
```

**B. Post-episode calibration, then optional offline induction**

```bash
orx inspect --bank retention                  # window / grace / archive caps
orx inspect --bank evaluations --task t1      # the stored per-prediction verdicts
orx calibration                               # the published wm-calib/2 summary
orx archive-calibration --dry-run             # what would leave the online set

orx induction-candidates                      # frozen evidence packages (no model call)
orx predict-capability --operation op.json --bundle <bundle>
orx compare-capability --predictions hp_1,hp_2 --horizon-tasks 10
orx accept-capability --recommendation <json> # runs it AND binds the fact
orx inspect --bank capability                 # fact bound vs effect verified
orx evaluate-capability --prediction hp_1     # needs qualified LATER tasks
```

## Tool policy

| Situation | Action |
|---|---|
| No `--world-model` configured | STOP — `not_configured` blocks both loops. Configure a provider first |
| Any task, before executing | `predict-strategy` per candidate (or `plan-next` once for all of them), then `execute --prediction`. One prediction per candidate — never two for the same decision |
| After a `plan-next` suggestion | `choose-next`, then `execute --prediction <that candidate's id>` — do NOT call `predict-strategy` again |
| After predicting candidates yourself (no planner) | Execute the candidate you chose with `--prediction <its id>`. There is no planning decision, so there is no `choose-next` to call |
| High-stakes or tied candidates | Plan instead of predicting one at a time; that is the only reason `plan-next` exists |
| Before treating an answer as a success | `check-task`. A relaxed LP is `optimal` with `gap=0` and still wrong |
| Facing an induction decision | `induction-candidates` → `predict-capability` → `compare-capability`, then accept or reject explicitly |
| No embedding backend configured | The text channel is off: `recall` returns `degraded` plus the structural channel alone. That is a different fact from "ran and matched nothing" |
| Embedding configured, no index yet | `orx rebuild-index` once; `record`/`induce`/`retire` keep it current |
| Simple problem, one independent constraint, no shared resources | CIR optional — state the skip decision explicitly |
| Cold start (empty bank) | `recall` returns `[]` with a reason. That is the NORMAL state: propose the methods you want and run them |
| You distrust a memory hit | `recall --exclude <id>` and re-read |
| A method the framework has never seen | Just name it: `predict-cost` reports `source: "unknown"` (cost UNKNOWN, not zero) and `execute` runs it |
| An execution fact is WRONG | `exclude-execution --execution EX --reason "..."` (append-only: the row survives, stops counting). Reverse with `restore-execution` |
| An entry looks dead | `inspect --bank strategic --status suspect` (or `dormant`) lists the candidates; `retire --entry ID --reason "..."` is the explicit, irreversible act |
| Storage growth | `inspect --bank retention` shows the three scopes; `archive-calibration` moves out-of-window detail into a BOUNDED archive. Online capacity and total disk are separate numbers |

## Decision rules

- **A signal is not a decision.** A hint, a `similarity`, a model's self-reported `confidence` and a plan suggestion are inputs. The choice and the admission verdict remain yours or the framework's.
- **A prediction is a shadow.** `plan-next` suggests, `choose-next` decides, `execute` acts. A prediction never becomes a fact.
- **Similarity discovers; structure decides reuse.** `applies` may be reused; `conflicts` and `unknown` may not — an unmeasured condition is not a satisfied one.
- **Measure coupling from structure.** Two independent resource constraints mean rc≈0, however resource-flavoured the task sounds.
- **The model comes after the strategy, before the code.**
- **An attempt is not a strategy window.** One `execute` is one attempt; modelling/repair/verify is auxiliary overhead.
- **Failures are raw material.** Record every attempt, timeouts and infeasibilities included; backfill from staging rather than re-typing.
- **A solved model is not a solved task.** Task validity and knowledge validity are separate gates, and a confirmed-wrong answer stops being a success sample without leaving the evidence set (its cost is real).
- **A failed check is not a modelling error.** `task_check_failed` reports that the DECLARED bases did not hold; the cause is YOUR diagnosis. The framework deliberately has no `model_invalid` label.
- **A reported infeasibility is a verdict, not a failure.** Correctly identifying an infeasible original problem is a valid outcome.
- **Two units, never conflated.** Occurrence rates count real OBSERVATION UNITS (an execution once, however many predictions were bound to it), and labels are PER EXECUTION. Brier statistics count prediction–observation PAIRS; only `scored_occurrence_rate` shares a denominator with `mean_predicted_probability`. The occurrence tally is derived from CURRENT facts, not frozen labels.
- **A correction is field-scoped.** A confirmed-wrong answer re-derives the benefit to `0.0` and KEEPS the measured cost; a withdrawn execution removes the sample; the correction applies only to the execution it names.
- **A threshold is per statistic.** A group of many episodes can still have ONE sample for a risk only one prediction mentioned — read `brier_evidence_by_event` / `under_sampled_statistics`. A statistic nobody predicted is ABSENT, not under-sampled.
- **The calibration is a window and a global diagnostic.** It reads the newest N closed episodes, carries no per-strategy breakdown, and must never be presented as the current candidate's conditional bias. It is a measured record of past errors — NOT trained model parameters and NOT a promise that future predictions improve.
- **Three different things, three different bounds.** Calibration STATISTICS enter the prompt; the online input is BOUNDED (window/grace/archive caps); raw archive detail is separate. Bounded online input does not by itself bound total disk.
- **A knowledge entry appearing is not a capability gain.** Predicting, binding the fact and verifying the effect are three different things.
- **The sample is counted in TASK-EPISODES.** Ten predictions bound to one execution are ONE independent truth.
- **Retirement is deliberate.** `retire` is irreversible; `induce --force` lifts a cold-archive veto (one-time). Reserve both for genuine drift and dead ends.
- **Cost learning needs the backfill.** `llm_tokens`/`tool_calls`/`retries` are your declarations — unknown ≠ zero, and an unmeasured dimension supports no cost claim.

## References

Read on demand — one hop, no chains. You do NOT need to read them all before starting; the workflow above is complete on its own.

| Read it when | Reference |
|---|---|
| You are about to call a command whose flags you do not remember, or you need the exact output semantics / return states | [references/commands.md](references/commands.md) |
| You want the "why": two-layer memory, structural grouping, CIR, CostVector, the disposal ladder, verification philosophy | [references/concepts.md](references/concepts.md) |
| You are writing your FIRST model or CIR (syntax, constraint labels, verification layers) | [references/modeling.md](references/modeling.md) |
| You are forming an induction decision, or submitting a structured relation claim (patterns, gate, verification semantics) | [references/induction.md](references/induction.md) |
| You need the unified prediction contracts, `contract_only`, attempt vs strategy window, or the legacy migration table | [references/world_model_contract.md](references/world_model_contract.md) |
| You are debugging what a prediction was conditioned on (the frozen input context, math-attribute origins, the two retrieval channels) | [references/prediction_context.md](references/prediction_context.md) |
| You want the wm-so/1 protocol in detail (yardstick, binding identity, comparison rules) | [references/strategy_outcome.md](references/strategy_outcome.md) |
| You are closing an episode, reading the calibration, or handling a late task-result verdict | [references/episode_closeout.md](references/episode_closeout.md) |
| You are unsure how the pieces fit together and want a runnable script to read | [references/examples/task_check.py](references/examples/task_check.py) (check → diagnose → repair → record → close) and [references/examples/episode_closeout.py](references/examples/episode_closeout.py) (predict → execute → bind → close → calibrate) |
