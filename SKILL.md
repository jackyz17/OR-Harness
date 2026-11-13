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
                          CONSUME FROZEN H+
                          Settle every positive job
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

Full prediction views are temporary and shared per decision; execution facts and light frozen conditions retain the history. `choose-next` or successful episode close releases the working input. An expired context needs a new decision; inspect its manifest with `context --context-id CTX --audit`. Fixed-memory tests fork a fresh home per question (`fork-memory`); old-view cleanup is explicit and preview-first (`maintain-snapshots`). See [lifecycle and operations](references/snapshot_lifecycle.md).

Use one path for a decision. Default prediction/planning already builds its context; explicit `context`, `snapshot`, historical `predict-cost`, and manual binding are optional inspection/reuse/recovery tools. Do not call them all in sequence. For an invalid forecast, inspect `result.failure.kind` and correct the cause before another call. Command formats and failures: [commands.md](references/commands.md).

### Read memory as evidence

Use embedding discovery with the structural profile as auxiliary evidence. Inspect applicable admitted knowledge first; otherwise adapt relevant execution cases. Recall reports memory, not utility-ranked decisions, and there is no built-in method catalog. An empty search is a cold start; a `degraded` text channel means it could not search, not that no related memory exists.

Read the two channels in that order and by field: an admitted entry appears as `recommendations[]` with `evidence: "strategic_entry"` (or `vector_recall.strategic_knowledge[]`); a `conditional_stats` row is only a recount over the evidence bank, NOT admitted knowledge. When no admitted entry applies, `vector_recall.execution_evidence[]` (and the `evidence_candidates[]` methods it suggests) is your main source — a past attempt is usable experience without being promoted to knowledge, and `inspect_hint` names the call that fetches the full record when the excerpt is not enough.

Check the actual semantic premises before transfer. `method.actual` / `performed` reports execution; `planned_only` is intent and `none` is unknown. Failed checks supply repair evidence; absent checks do not establish success. Inspect a full record when the excerpt omits a necessary premise. Cite adapted executions with `--adapted-from` and optional `--adaptation`; a citation does not prove reuse helped.

### Check, record, then retry

Report real solver status and result fields: `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and solution `variables` required by checks. A feasible incumbent is not a proved optimum. Report observed method/configuration receipts when available; do not copy the candidate plan into performed facts.

New requests use G = `P(original-task acceptance | faithful execution of the candidate steps under the agreed solving conditions)`, `effective_completion/task_check_passed_given_faithful_execution`, unit `probability`, observation rule `wm-g-faithful/1`. Method omissions remain omissions under faithful execution; script implementation risk belongs in R. In the same prediction, put one grounded sentence in `benefit.notes`: decisive task condition → candidate steps → expected consequence (or the missing condition), even without a numeric G. The main agent decides the revision. Example: 原题要求 BE 和 CFC 同厂，候选分别选厂后直接合并，预计可能产生跨厂解。

New G labels require a sourced `check-task` `g_attribution` assessment of method fidelity and original-task acceptance coverage, with references to actual code/result evidence. Faithful+passed=1; faithful+confirmed method-caused failure=0; implementation deviation, crash, timeout without a result, changed method, inadequate checking or unclear attribution=unknown. Matching step text and a checker verdict alone do not establish fidelity. Preserve explicit legacy metrics and old forecasts. Global G errors are diagnostic; never use their sign or a single residual to raise/lower G. Use conditional executed facts and same-task, same-acceptance method comparisons; two binary passes tie. See [episode_closeout.md](references/episode_closeout.md) for evidence fields and denominators.

R is one implementation/execution attempt's technical failure probability under the actual execution agent and solver interface. Predict only `implementation_failure.probability` as the scored R: a finite JSON number in [0,1], with 0 valid and missing unknown. Other risk events remain diagnostics. New `wm-r-technical/1` labels use executor/protocol evidence: code/API/abnormal-exit/protocol failure=1; normal protocol-valid result, including `infeasible`=0; external blockage or ordinary timeout=unknown. Failed and repaired attempts are separate. Use the request's applicable `implementation_r_evidence`; absent or sparse history does not establish a low risk.

`plan-next` uses a common scoring basis among ranked candidates. If any lacks R, all omit R while keeping raw probabilities visible. C uses the intersection of valid same-scope `expected_measured` dimensions, then component deduplication; no common dimension omits C for everyone and marks it incomparable. Explicit zero is valid. Read `score.scoring_basis` for actual terms, dimensions and omission reasons; partial utility carries no candidate-specific missing-value penalty.

Keep solver quality, task completion, and knowledge validity separate. Declare comparable benefit metrics before candidate comparison. Solver `optimal` can still answer the wrong model; `check-task` evaluates only declared task requirements and reports unchecked scope. Preserve an independent reference's source; objective agreement alone does not prove fidelity. Do not change the task or reference merely to fit a result.

| Task-check state | Next action |
|---|---|
| `passed` | Record the attempt; report which requirements were checked and which remain unchecked |
| `failed` | Record the attempt and its real cost first; inspect reflection material, fix the interpretation/model/code/reference issue, then predict the revised candidate |
| `insufficient` | Record with validity unknown; supply an available missing basis or solution value, without fabricating a reference |

Record all real attempts, including errors and deliberate relaxations. A failed task check does not erase measured solver quality or cost; completion remains a separate observation. Do not withdraw valid failure evidence merely because the answer is wrong.

Automatic `execute --prediction` binding happens before the run. A consumed prediction needs a new forecast for a new attempt. If an executor interruption releases the claim, the failed attempt remains staged and a retry may use the released ID. Missing receipts are unknown, distinct from known mismatches; inspect field-level eligibility instead of excluding everything.

## 3. Account for cost and close honestly

C predicts raw resources from the shared decision anchor to real task termination, including modelling, solves, checking, repair and retries. Use `cost_evidence` to compare candidate work with applicable executed workflows: model reuse, rebuilding, extra subproblems/checks and observed repairs. Count historical workflow steps once; cite sources in `evidence_basis` and put unsupported dimensions in `unsupported_fields`, outside the numeric `cost` object. Global C residuals are audit statistics, never instructions to multiply or raise/lower C. Conditional ranges require the execution host's recorded model, matching window and a known decision phase.

The runner should supply its known `execution_context` before the first prediction: actual execution `host`/`model`, optional `agent_id`, known `solver_interface`, and `phase` (`initial_modelling`, `existing_model`, or `unknown`). Use the Python constructor, `OR_EXECUTION_CONTEXT` (JSON/file), or the global `--execution-context` option. Ordinary `predict-strategy` freezes it into the snapshot; no preliminary model/usage action is needed. This is separate from `--world-model`. Missing runner facts remain unknown. Actual model-artifact progress advances the phase; reused/historical contexts retain their saved inputs.

When available, report full-window host counter deltas or a complete timestamped invocation ledger with IDs, the decision anchor and real termination. Tokens include input plus output; duplicate call references count once. `finish_task` or a trustworthy complete host span establishes end-to-end time for success, final failure or confirmed abort. An attempt total keeps its own scope, and the last execution/check end supplies only a lower bound. Missing counts, host identity or phase stay unknown. See [episode_closeout.md](references/episode_closeout.md) for the optional host-window evidence.

Summaries and complete call details must agree per dimension and respect in-window proven bounds, including execution receipts. Set `call_details_complete` to the dimensions the ledger fully covers; `[]` means partial details. A missing model-call ledger is not zero tokens. Conflicting dimensions stay unknown with source/reason and do not enter actual-cost ranges or errors; consistent dimensions remain usable.

Collect the host's whole-attempt usage report for tokens and all tool invocations; pass it with `record --usage-file` or a supported `--usage-host`. Keep measured dimensions, source, units, and scope explicit. Missing reports leave costs unknown; bare estimates and token lower bounds are not calibration actuals. Record and close honestly even when some measurements are unavailable; amend later from real reports.

Match terminal state to what happened: `completed`, `failed`, `aborted`, or `budget_exhausted`. Success requires task-supported validation or supported infeasibility; ending an episode does not certify either. Establish essential requirements through checks and inspected model/solution evidence. If essential validity remains unresolved, report it as unknown; a partial `passed` verdict alone does not establish task success. Return the answer/status, check coverage and limits, available real costs, and any unresolved issue.

If the host confirms cancellation, use `record --session --task T --episode E --reason "..."` to archive the stop before close-out. Episode budget exhaustion stops new solving/prediction work; a per-attempt latency overrun permits a retry if the episode still allows it. Checking, recording, and close-out remain finishing operations.

State the knowledge this attempt ADOPTED with `--used-entry-ids 3,5` (or `''` for "adopted none"): a DECLARATION, distinct from what recall surfaced (`--adapted-from` cites cases you read). Attribution is by NUMBER, never by a shared method name. Adoption records, the run's outcome and prediction calibration are kept apart: a missing prediction disables only the hit/miss, never the adoption record; an execution failure is not by itself a counterexample to the knowledge — that attribution is your analysis.

Within an episode, predictor parameters and published calibration stay fixed. Real attempts/checks may inform the next decision, whose input is frozen at that prediction's moment. At close-out, actual outcomes calibrate eligible forecasts for later episodes; unexecuted forecasts are not outcomes. New `capability_gain.decision` is the frozen binary induction switch, separate from G/C/R utility. A hypothesis and a verified knowledge increment are separate facts. Detailed feedback and late corrections: [prediction_context.md](references/prediction_context.md), [episode_closeout.md](references/episode_closeout.md).

### Frozen H+ and the building handoff

In the SAME `predict-strategy` answer, read `capability_gain.decision` as the JSON number **0 or 1** (`wm-hplus-binary/1`). With **1**, the model supplies one learning hypothesis in `claim` and one concrete check in `verification_conditions`; with **0**, it gives a reason in `basis`. H+ asks what reusable rule or boundary the current library lacks, what THIS execution can verify, and which future step would change. It never changes G/C/R strategy ranking.

When choosing a candidate, hand its frozen check to the solving agent. Use the actual method receipt, captured artifacts and `check-task`; optional `hplus_verification` records its sourced completion. An omitted/illegal decision schedules 0 with a distinct missing/invalid diagnostic. Never map a historical three-state stance to a binary decision. A valid 1 with missing hypothesis/check detail still requires one honest review.

**The outer building agent consumes the close-out handoff.** After execute/check/record/close, iterate every `result.induction_schedule.pending_ids` (also visible in `induction-status`). For each ID, read material once and submit a short rule or a sourced empty review using that SAME `--induction-id`. Failures and unfinished checks do not cancel a frozen 1. H+=0 saves the task/usage facts and skips formal material/writes; its execution may still serve as another positive job's contrast. Predicted-but-unexecuted candidates stay unobserved.

Do not declare library building finished with pending jobs. Check `induction-status` and require `completed:true` before freezing. A restarted agent resumes those IDs; repeated closes/material/submissions reuse the job/action/receipt. Python runners call `process_induction_jobs(reviewer, task_id=..., episode_id=...)`; the callback is the same outer inducer, not a second hidden framework model. The framework does not invent a rule or invoke a model for the agent.

Completed induction automatically binds the REAL induction action to the original `prediction_id`. Solving alone is not a maintenance fact. `hplus_feedback` shows relevant settled training receipts: independently validated substantive increment=positive; sourced sufficient comparison establishing coverage/no increment=negative; insufficient/unvalidated/unexecuted=unknown. Keep hypothesis truth separate: a refuted hypothesis can still yield a useful boundary rule. The separate legacy `evaluate-capability` path still measures later performance effects and requires comparable observations, especially for cost claims. See [hplus.md](references/hplus.md).

## 4. Offline induction

Complete execute → check → record → close before reading material. Close-out freezes existing rule-use checks and actual source facts before retention, then schedules from the PRE-EXECUTION H+. New calls automatically enable H+ building; an explicit `--induction-mode hplus` fixes it from the first command. Post-close `--induction-decision` is historical compatibility and cannot override a binary switch. Standalone old/manual libraries retain their explicit transition; do not use that mode to bypass an H+ training branch.

Normally read once with `orx induction-material --task T` and submit once. The default small comparison pack includes this task, related source problems/changed-premise neighbors and short existing rules. `--purpose modelling` compares constraint relationships across source problems; `--purpose implementation` emphasizes actual implementation and repair contrasts. Actual method/source, original checks and measured cost stay separate from plans and unknowns. Same-base retries/seeds/branches are controls; unknown lineage is marked. Similarity and family find candidates, never establish premises or semantic equivalence.

The whole default JSON response, including related history, old rules, diagnostics, H+ and CLI envelope, is bounded by 32000 characters and counted in `budget.chars_used`. Selection precedes expansion. Oversized methods/text are whole-value references, never clipped prefixes. Follow `budget.next_cursor` only when needed; use `inspect --bank experience --execution ID` or `inspect --bank strategic --entry ID` for full detail. Displayed cases remain derivation material, including earlier pages.

Submit a short **when / do / expect** rule with real execution evidence IDs. Put decisive premises in `when`, executable steps/reason/boundary or fallback in `do`, and the specific checked effect in `expect`. The framework assigns IDs, resolves evidence, freezes the body and manages checks. Compare existing rules by their actual premises, operation and effect. Use `entry_id` with `operation=add_evidence|revise|merge` for the same mechanism; select merge sources explicitly. Exact repeats are idempotent. Body changes create a new candidate version of the same ID; adding evidence/uses never inflates its body. If the library already covers the mechanism, complete the positive job with an empty review (`induce --induction-id J --review '{"outcome":"already_covered","conclusion":"..."}' --note "..."`), without creating a rule. A negative training label additionally needs a sourced sufficient comparison referencing the frozen baseline and actual fact.

```bash
orx induction-material --task T --induction-id J
orx induce --induction-id J --relation '{"when":"For fixed Y, feasible X is an integer interval and the objective is monotone in X.","do":"Enumerate Y and derive both X endpoints. Select the endpoint using coefficient sign and objective direction. Disable this reduction for a nonmonotone objective and use full enumeration.","expect":"Preserve the original integer optimum and constraints.","evidence":["ex_REAL_ID"]}'
# Targeted trial of a pending version, before it enters default recall:
orx execute --task next.json --strategy endpoint --code solve.py --workspace ws --used-entry-ids 3@1
```

A new rule stays candidate until an **independent unseen source problem** and a **changed-premise boundary** have been checked; a revision also rechecks affected old successes. Declare the exact adopted version, actual-method evidence and a sourced `check-task.rule_assessments` record covering premises, fidelity and the claimed effect. A plan, a solver status or checker pass alone is insufficient. Missing adoption/method/coverage stays unknown. Only an attributable in-domain counterexample can pause an active rule. Two successful tasks do not prove a cost advantage: it requires same-task, same-scope measured comparisons.

Only one version is normally active; the old active version remains while a revision is pending. Close-out and late checks append independent versioned check/use records. Actual uses retain their frozen versions; audits retain decisive facts after evidence eviction. Effect attributions remain separate facts and cannot bypass v2 publication. True legacy rows retain their original semantics and are marked historical; old `--relation` input is accepted but new writes follow the v2 lifecycle. Full examples, assessment fields and recovery: [induction.md](references/induction.md).

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
