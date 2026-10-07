---
name: or-harness-wm
description: >
  Formulate, solve, check, retry, and learn from large or coupled industrial
  optimization tasks, including LP, MILP, scheduling, routing, allocation,
  and supply chains. Use for repeated optimization work, strategy selection
  by predicted quality/cost/risk, task-level answer checking, execution-cost
  recording, offline strategy learning, or explicit OR-Harness requests.
  Skip ordinary isolated math questions and pure proofs unless the user
  explicitly requests the Harness workflow; exclude non-optimization work.
---

# OR-Harness: strategy learning for optimization agents

Orchestrate the task: understand its structure, choose a method before coding, predict its consequences, execute, check the answer, and retain the facts that support later decisions. You choose methods and judge applicability; the framework records executions, checks declared assertions, and maintains feedback.

Use one explicit memory directory (`--home` or `OR_HARNESS_HOME`) throughout the episode. Commands return `{"result": {...}, "summary": "..."}`; exit codes are `0` success, `2` usage/precondition error, `1` crash. Read the result state before deciding the next action.

Configure a world-model provider for the prediction-based solving loop (`--world-model URL::MODEL`, key in `OR_WM_API_KEY`, or `ORHarness(world_model=...)`). If unavailable, resolve configuration or report the blocked forecast; do not silently present an unpredicted run as the full loop. Inspection and direct offline review do not themselves produce predictions.

## 1. Prepare the task

- Create task JSON with stable `task_id`, `family`, and the actual problem `text`. Keep the same task ID across retries/solver changes; use a different ID for a different problem. Choose one episode label for the solving effort and carry it through prediction, execution, and close-out.
- Represent coupled entities, decisions, constraints, and relations in CIR. Use `profile` to understand them and freeze the pre-strategy profile. A missing `model` is normal: write it after choosing the strategy; later model/script diagnostics do not redefine problem identity.
- Use the same task/CIR version across a prediction and its execution. Inspect missing values and disagreements rather than filling them with guesses. Task/CIR and script requirements are in [modeling.md](references/modeling.md).

## 2. Run the online loop

```text
UNDERSTAND + RECALL
        |
        v
PROPOSE -> PREDICT -> CHOOSE <-------------------------+
        |                                            |
        v                                            |
MODEL -> CODE -> EXECUTE                              |
        |                                            |
        v                                            |
CHECK -> RECORD (every attempt, including failures)   |
        |                                            |
        +-- repair needed + budget permits ----------+
        |
        +-- episode ends --> CLOSE EPISODE
                                   |
                                   v
                          PUBLISH CALIBRATION
                                   |
                                   v
                          REVIEW OFFLINE
                          Consolidate when useful
```

| Step | Action | Keep / decide |
|---|---|---|
| Understand | `orx profile --task t.json` | CIR guidance, profile, derivation; optional `--cir` |
| Retrieve | `orx recall --task t.json --top 5` | Applicable knowledge, execution cases, unknowns, degradation |
| Propose | Describe method `name` and `steps`, concrete solver, execution `config` | Reuse an existing strategy ID when appropriate; propose your own methods at cold start |
| Predict and choose | Select one of the two paths below | Chosen candidate's prediction ID |
| Implement | Write the chosen formulation and `solve.py` | Actual constraints, variable domains, solution reporting |
| Execute | `orx execute --task t.json --prediction ID --code solve.py --workspace ws` | `result.execution_id`, `prediction_binding`; attempt is staged |
| Check | `orx check-task EXECUTION_ID --check '{...}'` | State, checks/unchecked scope, reflection material |
| Record | `orx record --from-staged EXECUTION_ID [--usage-file usage.json]` | Always record; attach an available usage file or supported host adapter |
| Continue or finish | Repair and make a new decision, or `orx close-episode --task T --episode E --terminal STATE` | Real terminal state, check coverage, evaluations, calibration |

### Predict once per candidate decision

**Direct path:** `predict-strategy --task t.json --episode E --candidate c.json`; choose from the valid forecasts and pass the chosen `result.prediction_id` to execution. One candidate is enough when the decision warrants only one; do not invent alternatives to fill a quota.

**Comparison path:** `plan-next --task t.json --episode E --candidates cs.json`, then `choose-next --decision ACTION_ID --prediction ID`. Use a prediction ID returned in that decision's `result.plan.candidates[]`, then execute with that same ID. Planning has horizon 1; re-plan after real observations when needed.

Use one path for a decision. Default prediction/planning already builds its context; explicit `context`, `snapshot`, historical `predict-cost`, and manual binding are optional inspection/reuse/recovery tools. Do not call them all in sequence. For an invalid forecast, inspect `result.failure.kind` and correct the cause before another call. Command formats and failures: [commands.md](references/commands.md).

### Read memory as evidence

Use embedding discovery with the structural profile as auxiliary evidence. Inspect applicable admitted knowledge first; otherwise adapt relevant execution cases. Recall reports memory, not utility-ranked decisions, and there is no built-in method catalog. An empty search is a cold start; a `degraded` text channel means it could not search, not that no related memory exists.

Read the two channels in that order and by field: an admitted entry appears as `recommendations[]` with `evidence: "strategic_entry"` (or `vector_recall.strategic_knowledge[]`); a `conditional_stats` row is only a recount over the evidence bank, NOT admitted knowledge. When no admitted entry applies, `vector_recall.execution_evidence[]` (and the `evidence_candidates[]` methods it suggests) is your main source — a past attempt is usable experience without being promoted to knowledge, and `inspect_hint` names the call that fetches the full record when the excerpt is not enough.

Check the actual semantic premises before transfer. `method.actual` / `performed` reports execution; `planned_only` is intent and `none` is unknown. Failed checks supply repair evidence; absent checks do not establish success. Inspect a full record when the excerpt omits a necessary premise. Cite adapted executions with `--adapted-from` and optional `--adaptation`; a citation does not prove reuse helped.

### Check, record, then retry

Report real solver status and result fields: `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and solution `variables` required by checks. A feasible incumbent is not a proved optimum. Report observed method/configuration receipts when available; do not copy the candidate plan into performed facts.

Keep solver quality, task completion, and knowledge validity separate. Declare comparable benefit metrics before candidate comparison. Solver `optimal` can still answer the wrong model; `check-task` evaluates only declared task requirements and reports unchecked scope. Preserve an independent reference's source; objective agreement alone does not prove fidelity. Do not change the task or reference merely to fit a result.

| Task-check state | Next action |
|---|---|
| `passed` | Record the attempt; report which requirements were checked and which remain unchecked |
| `failed` | Record the attempt and its real cost first; inspect reflection material, fix the interpretation/model/code/reference issue, then predict the revised candidate |
| `insufficient` | Record with validity unknown; supply an available missing basis or solution value, without fabricating a reference |

Record all real attempts, including errors and deliberate relaxations. A failed task check does not erase measured solver quality or cost; completion remains a separate observation. Do not withdraw valid failure evidence merely because the answer is wrong.

Automatic `execute --prediction` binding happens before the run. A consumed prediction needs a new forecast for a new attempt. If an executor interruption releases the claim, the failed attempt remains staged and a retry may use the released ID. Missing receipts are unknown, distinct from known mismatches; inspect field-level eligibility instead of excluding everything.

## 3. Account for cost and close honestly

Collect the host's whole-attempt usage report for tokens and all tool invocations; pass it with `record --usage-file` or a supported `--usage-host`. Keep measured dimensions, source, units, and scope explicit. Missing reports leave costs unknown; bare estimates and token lower bounds are not calibration actuals. Record and close honestly even when some measurements are unavailable; amend later from real reports.

Match terminal state to what happened: `completed`, `failed`, `aborted`, or `budget_exhausted`. Success requires task-supported validation or supported infeasibility; ending an episode does not certify either. Establish essential requirements through checks and inspected model/solution evidence. If essential validity remains unresolved, report it as unknown; a partial `passed` verdict alone does not establish task success. Return the answer/status, check coverage and limits, available real costs, and any unresolved issue.

If the host confirms cancellation, use `record --session --task T --episode E --reason "..."` to archive the stop before close-out. Episode budget exhaustion stops new solving/prediction work; a per-attempt latency overrun permits a retry if the episode still allows it. Checking, recording, and close-out remain finishing operations.

State the knowledge this attempt ADOPTED with `--used-entry-ids 3,5` (or `''` for "adopted none"): this is a DECLARATION, distinct from what recall surfaced (`--adapted-from` cites cases you read). The framework ties the entry numbers to this attempt's result and keeps recall, adoption and outcome apart.

Within an episode, predictor parameters and published calibration stay fixed. Real attempts/checks may inform the next decision, whose input is frozen at that prediction's moment. At close-out, actual outcomes calibrate eligible forecasts for later episodes; unexecuted forecasts are not outcomes. H+ states an anticipated learning gain (`expected|none|insufficient_basis`), not verified improvement or an immediate training signal. Detailed feedback and late corrections: [prediction_context.md](references/prediction_context.md), [episode_closeout.md](references/episode_closeout.md).

## 4. Offline induction

After close-out, review when a method, repair, boundary, or new evidence may add value. Start with `orx induction-material --task T`: it reports the banks' `memory_state`, the task chain and bounded `related_history` (the query uses the recorded method, including failures and cross-cell cases). `--related-top-k` controls the budget (default 5; 0 disables); read wider via `budget.next_cursor` only when useful. An EMPTY bank is a reported state, not a gate — with no prior knowledge, solve the task and its own evidence is the first material; you are never required to publish a first entry.

Form knowledge from the actual method and relationships:

**condition → operation → reason → expected effect → boundary**

Inspect premises and existing knowledge; you may **add, or submit nothing**. Knowledge is ADDITIVE: each `induce --relation` creates a NEW numbered entry (the framework assigns the number); a revision or a contradiction is its own entry, and an existing entry is never rewritten. The framework checks only administrative matters (well-formed, cited executions exist, valid number, successful write) — publication is YOUR decision, and any `check`/`--verify` you supply is recorded as your own audit trail, not a gate. A single task can reveal a conditional method; a claimed quality/cost/risk advantage needs comparable measurements; retries are not independent tasks, similarity is not support. Runtime formats: [induction.md](references/induction.md).

Induce only when worth it; `induce` with no relations still runs utility maintenance so existing knowledge's lifecycle keeps working.

Later, reuse an entry by stating it in `--used-entry-ids`; an entry's `verification` block says what was checked. Recording and close-out do not automatically induce strategies.

## 5. Read references when needed

| Need | Reference |
|---|---|
| Design, two memory layers, transfer and capability meaning | [concepts.md](references/concepts.md) |
| Task/CIR/model/script construction | [modeling.md](references/modeling.md) |
| Flags, result paths, usage reports, diagnostics and recovery | [commands.md](references/commands.md) |
| Candidate consequences, comparison protocol and budgets | [strategy_outcome.md](references/strategy_outcome.md) |
| Frozen prediction input and feedback context | [prediction_context.md](references/prediction_context.md) |
| Close-out, calibration and late checks | [episode_closeout.md](references/episode_closeout.md) |
| Method abstraction, assertions, publication and revision | [induction.md](references/induction.md) |
| Versioned fields and legacy contract mapping | [world_model_contract.md](references/world_model_contract.md) |
| Runnable cold-start/check/repair example | [end_to_end.py](references/examples/end_to_end.py) |

Load only the reference needed for the current decision. Read it after a repeated unexplained command failure instead of trying more flag variations. Diagnostics, index rebuilds, maintenance forecasts, and later capability evaluations are conditional operations, not extra steps in every solve.
