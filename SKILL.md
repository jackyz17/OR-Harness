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

Orchestrate the task: understand its structure, propose methods before coding, score their predicted consequences, execute the permitted selection, check the answer, and retain the facts that support later decisions. You propose methods and judge applicability; the framework selects by U, records executions, checks declared assertions, and maintains feedback.

Use one explicit memory directory (`--home` or `OR_HARNESS_HOME`) throughout the episode. Commands return `{"result": {...}, "summary": "..."}`; exit codes are `0` success, `2` usage/precondition error, `1` crash. Read the result state before deciding the next action.

Configure a world-model provider for the prediction-based solving loop (`--world-model URL::MODEL`, key in `OR_WM_API_KEY`, or `ORHarness(world_model=...)`). If unavailable, resolve configuration or report the blocked forecast; do not silently present an unpredicted run as the full loop. Inspection and direct offline review do not themselves produce predictions.

## 1. Prepare the task

- Create task JSON with stable `task_id`, `family`, and the actual problem `text`. Keep the same task ID across retries/solver changes; use a different ID for a different problem. Choose one episode label for the solving effort and carry it through prediction, execution, and close-out.
- Represent coupled entities, decisions, constraints, and relations in CIR. Use `profile` to understand them and freeze the pre-strategy profile. A missing `model` is normal: write it after choosing the strategy; later model/script diagnostics do not redefine problem identity.
- Use the same task/CIR version across a prediction and its execution. Inspect missing values and disagreements rather than filling them with guesses. Task/CIR and script requirements are in [modeling.md](references/modeling.md).

Create or reuse one `task.json` for both interactive and batch commands, and pass that same file through profile, recall, planning and execution. A minimal input is:

```json
{"task_id":"endpoint-B","family":"assignment","text":"Minimize 2*X-5*Y for nonnegative integers X,Y, subject to X+Y<=641 and X>=2*Y+130. Return the optimum and X,Y."}
```

T below means `task.json.task_id`; E is the episode label chosen for this solving effort. Add structured `spec` or CIR when useful. Put a CIR needed by later commands in `task.json.coupling`; a CIR supplied only to `profile --cir` is not implicitly reused. `profile` reports structure, modeling guidance and CIR health; `recall` already includes a task profile.

## 2. Run the online loop

```text
UNDERSTAND + RECALL
        |
        v
PROPOSE -> PLAN-NEXT (PREDICT + U + SELECT) <----------+
        |                                            |
        v                                            |
MODEL -> CODE -> EXECUTE --prediction P               |
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
                                   |
                                   v
                          NEXT TASK (same home)
                          RECALL + PLAN-NEXT
```

| Step | Action | Keep / decide |
|---|---|---|
| Understand | `orx profile --task task.json` | CIR guidance, profile, derivation; optional `--cir` |
| Retrieve | `orx recall --task task.json --top 5` | Applicable knowledge, execution cases, unknowns, degradation |
| Propose | Describe method `name` and `steps`, concrete solver, execution `config` | Reuse an existing strategy ID when appropriate; propose your own methods at cold start |
| Score and select | `orx plan-next --task task.json --episode E --candidates candidates.json` | `result.selected_prediction_id`, `result.selected_score`; only the permitted winner can execute |
| Implement | Write the chosen formulation and `ws/solve.py` | Actual constraints, variable domains, solution reporting |
| Execute | `orx execute --task task.json --prediction ID --code ws/solve.py --workspace ws` | `result.execution_id`, `prediction_binding`; attempt is staged |
| Check | `orx check-task EXECUTION_ID --check check.json` | `result.state`, `result.report.checks[]`, `result.report.scope.unchecked[]`; use sourced original-task checks |
| Record | `orx record --from-staged EXECUTION_ID [--usage-file usage.json]` | Always record; attach an available usage file or supported host adapter |
| Continue or finish | Repair and make a new decision, or `orx close-episode --task T --episode E --terminal STATE` | Real terminal state, check coverage, evaluations, calibration |

### Predict once per candidate decision

Write `candidates.json` as a list of one or more substantive methods. For the task above, one candidate can be:

```json
[{"name":"Lower endpoint enumeration","steps":["Enumerate integer Y from 0 through 170 and derive 2*Y+130 <= X <= 641-Y.","Skip empty intervals; choose the lower X endpoint because the minimization objective increases with X at fixed Y.","Compare objectives across Y and check integrality and both original inequalities."],"solver":"python","config":{"time_limit":60}}]
```

Use `plan-next` for one or multiple candidates. It shares one frozen context, predicts each candidate once and saves `U = alpha*G - beta*C - gamma*R`. A valid forecast is separate from execution permission. Read the selected ID and score; if none is permitted, inspect the plan/candidate reasons and revise before planning again.

```bash
orx plan-next --task task.json --episode E --candidates candidates.json
# P = result.selected_prediction_id; use the original permitted prediction.
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
```

Single candidates are scored too. No routine manual choice or second forecast is needed. Independent API forecasts, unselected candidates, bare strategy arguments and later binding cannot authorize formal execution. Compatibility and recovery entries are documented in [commands.md](references/commands.md).

Full prediction views are temporary and shared per decision; execution facts and light frozen conditions retain the history. Successful episode close releases the working input; a compatibility choice/rejection can release it earlier. An expired context needs a new decision. Optional lifetime audits, historical cleanup and existing experiment operations are in [lifecycle and operations](references/snapshot_lifecycle.md).

Use one scored planning decision for a proposed execution. Default planning already builds its context and retrieval; explicit `context`, `snapshot`, historical `predict-cost`, and manual binding are optional inspection/reuse/recovery tools. Do not call them all in sequence. Read `result.selected_prediction_id`, `result.selected_score` and `result.plan.candidates[]`. If no ID is permitted, inspect `result.plan.status`, `result.plan.truncation_reason`, candidate refusals and any input rejections. Inspect a saved forecast with `orx inspect --bank predictions --prediction P`; its payload is `result.prediction.prediction`, with detailed failure information in `trace.model_info.failure`. Correct the cause before planning again. Command formats and failures: [commands.md](references/commands.md).

### Read memory as evidence

Use embedding discovery with the structural profile as auxiliary evidence. Inspect applicable admitted knowledge first; otherwise adapt relevant execution cases. Recall reports memory, not utility-ranked decisions, and there is no built-in method catalog. An empty search is a cold start; a `degraded` text channel means it could not search, not that no related memory exists.

Read `result.recommendations_basis`, `result.vector_recall` and `result.degraded` separately to distinguish no structural recommendations, no semantic matches and unavailable search. In this implementation `--memory-mode none` clears structural recommendations but still runs vector retrieval; it is not a no-memory isolation switch.

Read the two channels in that order and by field: an admitted entry appears as `recommendations[]` with `evidence: "strategic_entry"` (or `vector_recall.strategic_knowledge[]`); a `conditional_stats` row is only a recount over the evidence bank, NOT admitted knowledge. When no admitted entry applies, `vector_recall.execution_evidence[]` (and the `evidence_candidates[]` methods it suggests) is your main source — a past attempt is usable experience without being promoted to knowledge, and `inspect_hint` names the call that fetches the full record when the excerpt is not enough.

Check the actual semantic premises before transfer. `method.actual` / `performed` reports execution; `planned_only` is intent and `none` is unknown. Failed checks supply repair evidence; absent checks do not establish success. Inspect a full record when the excerpt omits a necessary premise. Cite adapted executions with `--adapted-from` and optional `--adaptation`; a citation does not prove reuse helped.

### Check, record, then retry

Report real solver status and result fields: `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and solution `variables` required by checks. A feasible incumbent is not a proved optimum. Report observed method/configuration receipts when available; do not copy the candidate plan into performed facts.

New requests use G = `P(original-task acceptance | faithful execution of the candidate steps under the agreed solving conditions)`, `effective_completion/task_check_passed_given_faithful_execution`, unit `probability`, observation rule `wm-g-faithful/1`. Method omissions remain omissions under faithful execution; script implementation risk belongs in R. In the same prediction, put one grounded sentence in `benefit.notes`: decisive task condition → candidate steps → expected consequence (or the missing condition), even without a numeric G. The main agent decides the revision. Example: the original task requires BE and CFC at one factory; the candidate selects their factories separately and merges the results, so a cross-factory answer may result.

New G labels require a sourced `check-task` `g_attribution` assessment of method fidelity and original-task acceptance coverage, with references to actual code/result evidence. Faithful+passed=1; faithful+confirmed method-caused failure=0; implementation deviation, crash, timeout without a result, changed method, inadequate checking or unclear attribution=unknown. Matching step text and a checker verdict alone do not establish fidelity. Preserve explicit legacy metrics and old forecasts. Global G errors are diagnostic; never use their sign or a single residual to raise/lower G. Use conditional executed facts and same-task, same-acceptance method comparisons; two binary passes tie. See [episode_closeout.md](references/episode_closeout.md) for evidence fields and denominators.

R is one implementation/execution attempt's technical failure probability under the actual execution agent and solver interface. Predict only `implementation_failure.probability` as the scored R: a finite JSON number in [0,1], with 0 valid and missing unknown. Other risk events remain diagnostics. New `wm-r-technical/1` labels use executor/protocol evidence: code/API/abnormal-exit/protocol failure=1; normal protocol-valid result, including `infeasible`=0; external blockage or ordinary timeout=unknown. Failed and repaired attempts are separate. Use the request's applicable `implementation_r_evidence`; absent or sparse history does not establish a low risk.

`plan-next` uses a common scoring basis among ranked candidates. If any lacks R, all omit R while keeping raw probabilities visible. C uses the intersection of valid same-scope `expected_measured` dimensions, then component deduplication; no common dimension omits C for everyone and marks it incomparable. Explicit zero is valid. Cost scales come from applicable pre-decision measurements or explicit configuration, independently of the candidate set. Read `score.scoring_basis` for actual terms, scales, dimensions and omission reasons; partial utility carries no candidate-specific missing-value penalty.

Keep solver quality, task completion, and knowledge validity separate. Declare comparable benefit metrics before candidate comparison. Solver `optimal` can still answer the wrong model; `check-task` evaluates only declared task requirements and reports unchecked scope. Preserve an independent reference's source; objective agreement alone does not prove fidelity. Do not change the task or reference merely to fit a result.

EX is `execute`'s `result.execution_id`. Build `check.json` from the actual original requirements and sourced evidence; read `result.state`, `result.report.checks[]` and `result.report.scope.unchecked[]` from `check-task`, then save EX with `record --from-staged EX`. A repair keeps the same original task file and episode, obtains a fresh U-selected prediction and preserves earlier failed attempts and measured costs.

| Task-check state | Next action |
|---|---|
| `passed` | Record the attempt; report which requirements were checked and which remain unchecked |
| `failed` | Record the attempt and its real cost first; inspect reflection material, fix the interpretation/model/code/reference issue, then predict the revised candidate |
| `insufficient` | Record with validity unknown; supply an available missing basis or solution value, without fabricating a reference |

Record all real attempts, including errors and deliberate relaxations. A failed task check does not erase measured solver quality or cost; completion remains a separate observation. Do not withdraw valid failure evidence merely because the answer is wrong.

Automatic `execute --prediction` binding happens before the run. A consumed prediction needs a new forecast for a new attempt. An interrupted real attempt remains staged with its measured costs; a repair obtains a new permitted prediction through `plan-next` in the same episode. Missing receipts are unknown, distinct from known mismatches; inspect field-level eligibility instead of excluding everything.

## 3. Account for cost and close honestly

C predicts raw resources from the shared decision anchor to real task termination, including modelling, solves, checking, repair and retries. Use `cost_evidence` to compare candidate work with applicable executed workflows: model reuse, rebuilding, extra subproblems/checks and observed repairs. Count historical workflow steps once; cite sources in `evidence_basis` and put unsupported dimensions in `unsupported_fields`, outside the numeric `cost` object. Global C residuals are audit statistics, never instructions to multiply or raise/lower C. Conditional ranges require the execution host's recorded model, matching window and a known decision phase.

The runner should supply its known `execution_context` before the first prediction: actual execution `host`/`model`, optional `agent_id`, known `solver_interface`, and `phase` (`initial_modelling`, `existing_model`, or `unknown`). Use the Python constructor, `OR_EXECUTION_CONTEXT` (JSON/file), or the global `--execution-context` option. Ordinary `plan-next` freezes it into the snapshot; no preliminary model/usage action is needed. This is separate from `--world-model`. Missing runner facts remain unknown. Actual model-artifact progress advances the phase; reused/historical contexts retain their saved inputs.

When available, report full-window host counter deltas or a complete timestamped invocation ledger with IDs, the decision anchor and real termination. Tokens include input plus output; duplicate call references count once. `finish_task` or a trustworthy complete host span establishes end-to-end time for success, final failure or confirmed abort. An attempt total keeps its own scope, and the last execution/check end supplies only a lower bound. Missing counts, host identity or phase stay unknown. See [episode_closeout.md](references/episode_closeout.md) for the optional host-window evidence.

Summaries and complete call details must agree per dimension and respect in-window proven bounds, including execution receipts. Set `call_details_complete` to the dimensions the ledger fully covers; `[]` means partial details. A missing model-call ledger is not zero tokens. Conflicting dimensions stay unknown with source/reason and do not enter actual-cost ranges or errors; consistent dimensions remain usable.

Collect the host's whole-attempt usage report for tokens and all tool invocations; pass it with `record --usage-file` or a supported `--usage-host`. Keep measured dimensions, source, units, and scope explicit. Missing reports leave costs unknown; bare estimates and token lower bounds are not calibration actuals. Record and close honestly even when some measurements are unavailable; amend later from real reports.

Match terminal state to what happened: `completed`, `failed`, `aborted`, or `budget_exhausted`. Success requires task-supported validation or supported infeasibility; ending an episode does not certify either. Establish essential requirements through checks and inspected model/solution evidence. If essential validity remains unresolved, report it as unknown; a partial `passed` verdict alone does not establish task success. Return the answer/status, check coverage and limits, available real costs, and any unresolved issue.

If the host confirms cancellation, use `record --session --task T --episode E --reason "..."` to archive the stop before close-out. Episode budget exhaustion stops new solving/prediction work; a per-attempt latency overrun permits a retry if the episode still allows it. Checking, recording, and close-out remain finishing operations.

State the knowledge this attempt ADOPTED with `--used-entry-ids ENTRY@VERSION` (or `''` for "adopted none"): a DECLARATION, distinct from what recall surfaced (`--adapted-from` cites cases you read). Attribution is by NUMBER, never by a shared method name. Adoption records, the run's outcome and prediction calibration are kept apart: a missing prediction disables only the hit/miss, never the adoption record; an execution failure is not by itself a counterexample to the knowledge — that attribution is your analysis.

Adopt the actual version you have read. Structural recall exposes it in `result.recommendations[].knowledge.rule`. If a semantic hit lacks a version, run `orx inspect --bank strategic --entry ENTRY` and read `result.entries[0].rule.view_version`. Normal check/record/close retains the use; optional sourced `rule_assessments` record transfer, boundary or counterexample judgments. Utility/status management uses the separate `--attribute-effect` path described below.

Within an episode, predictor parameters and published calibration stay fixed. Real attempts/checks may inform the next decision, whose input is frozen at that prediction's moment. At close-out, actual outcomes calibrate eligible forecasts for later episodes; unexecuted forecasts are not outcomes. New `capability_gain.decision` is the frozen binary induction switch, separate from G/C/R utility. A hypothesis and a verified knowledge increment are separate facts. Detailed feedback and late corrections: [prediction_context.md](references/prediction_context.md), [episode_closeout.md](references/episode_closeout.md).

### Frozen H+ and the building handoff

In the SAME candidate prediction inside `plan-next`, read `capability_gain.decision` as the JSON number **0 or 1** (`wm-hplus-binary/1`). With **1**, the model supplies one learning hypothesis in `claim` and one concrete check in `verification_conditions`; with **0**, it gives a reason in `basis`. H+ asks what reusable rule or boundary the current library lacks, what THIS execution can verify, and which future step would change. It never changes G/C/R strategy ranking.

For the U-selected candidate, hand its frozen check to the solving agent. Use the actual method receipt, captured artifacts and `check-task`; optional `hplus_verification` records its sourced completion. An omitted/illegal decision schedules 0 with a distinct missing/invalid diagnostic. Never map a historical three-state stance to a binary decision. A valid 1 with missing hypothesis/check detail still requires one honest review.

**The outer building agent consumes the close-out handoff.** After execute/check/record/close, iterate every `result.induction_schedule.pending_ids` (also visible in `induction-status`). For each ID, perform one bounded review round and submit a short rule or a sourced empty review using that SAME `--induction-id`. Material starts from the bound execution. Adjust `--purpose`, `--query`, `--related-top-k` or `--executions` when useful; every delivered session belongs to this round. Failures and unfinished checks do not cancel a frozen 1. H+=0 saves the task/usage facts and skips formal material/writes; its execution may still serve as another positive job's contrast or validate an existing rule. Predicted-but-unexecuted candidates stay unobserved.

Do not declare library building finished with pending jobs. Check `induction-status` and require `completed:true` before freezing. A restarted agent resumes those IDs; repeated closes/material/submissions reuse the job/action/receipt. Python runners call `process_induction_jobs(reviewer, task_id=..., episode_id=...)`; the callback is the same outer inducer, not a second hidden framework model. The framework does not invent a rule or invoke a model for the agent.

Close-out dispatches the jobs; the agent consumes them; `induction-status --task T --episode E` queries settlement. Resolve actual `unfinished_actions` before retrying a refused close. Finish the current task's required reviews before starting the next training task. H+=0 and no pending jobs require no empty review.

Completed induction automatically binds the REAL induction action to the original `prediction_id`. Solving alone is not a maintenance fact. `hplus_feedback` shows relevant settled training receipts: agent-supported substantive increment=positive (including checked derivation or small runnable tests); sourced coverage/no-increment conclusion=negative; insufficient or unobserved review=unknown. Publication, empty recall and later retirement do not determine this label. Keep hypothesis truth separate: a refuted hypothesis can still yield a useful boundary rule. The separate legacy `evaluate-capability` path still measures later performance effects and requires comparable observations, especially for cost claims. See [hplus.md](references/hplus.md).

## 4. Offline induction

Complete execute → check → record → close before reading material. Close-out freezes existing rule-use checks and actual source facts before retention, then schedules from the PRE-EXECUTION H+. New calls automatically enable H+ building; an explicit `--induction-mode hplus` fixes it from the first command. Post-close `--induction-decision` is historical compatibility and cannot override a binary switch. Standalone old/manual libraries retain their explicit transition; do not use that mode to bypass an H+ training branch.

Normally read with `orx induction-material --task T --induction-id J`. Every call queries current evidence; change `--purpose`, `--query`, `--related-top-k`, `--executions` or `--cursor` as useful. The primary case remains J's actual bound execution, even if a newer attempt ran the same task with H+=0. Control pagination never skips those newer attempts behind the older anchor. Known source links and exact overlap are advisory facts; the agent decides comparison purpose and transfer scope. No topic, mother-task field, path or text heuristic decides publication.

Read `result.material[]` for the trigger/current attempts, `result.related_history.executions[]` for possible contrasts, `result.existing_knowledge` for relevant short rules and `result.comparison_table` for case relationships/check references. Compare actual methods, original requirements, declared acceptance coverage and existing rule conditions before deciding what can be generalized.

The default JSON pack remains bounded by 32000 characters, including history, old rules and the CLI envelope. Oversized content is referenced whole. Pending reviews temporarily retain their triggering and selected necessary evidence through eviction/restart. Completing the receipt releases full temporary copies and keeps minimal audit facts. Ordinary `check-task` does not create a separate full execution snapshot. Historical packs/receipts remain readable.

The next control or its detail reference has budget priority over old rules. Cite concrete captured artifacts and real check items. Every directly cited comparison execution keeps minimum audit facts after receipt completion. A later checker correction can remove support from future H+ feedback; old prediction inputs stay traceable. Missing new references reject completion, while an unrecoverable historical receipt is individually unknown.

Submit **when / do / expect**, chosen real evidence IDs, and an explicit publication choice when ready. `publication:{"state":"published","reason":"...","scope":[...]}` offers this version immediately within the reported scope. Choose `candidate` to defer; omission leaves a new body candidate. The agent can use derivation, benchmark/objective checks, exhaustive constructed instances and boundary tests before any third real task exists. Independent future tasks can expand the observed scope. The framework checks references and records scope; it does not require transfer, boundary or regression as publication prerequisites.

```bash
orx induction-material --task T --induction-id J
```

Use the [complete relation-and-review submission](references/induction.md#complete-review-submissions) with the same J. Submit `--relation` and `--review` together for a new/revised method. That reference is the single full example and includes supported `already_covered` / `no_increment` and `insufficient_material` branches. A coverage conclusion needs actual execution and available prediction-time rules. Read `result.relations[].publication.published` separately from `induction-status`'s `result.completed` and `result.observation_counts`. A frozen H+=0 skips new induction; learning label 0 requires a real supported no-increment review, while insufficient support remains unknown.


Keep the same entry number for `add_evidence`, `revise` or an explicitly confirmed `merge`. Body versions are immutable; safe formatting equivalence reuses the version. A candidate revision leaves the active version in normal recall until the agent selects another. Historical v1 facts may be reviewed for v2 compatibility; only an execution that actually adopted v2 is a v2 use. Optional old-case reviews/replays inform the decision; no automatic regression todo list is generated. A check of `reference_objective:2699` establishes that numerical comparison only. Missing acceptance remains unknown.

Existing-entry use, validation, evidence addition, publication selection and utility management are separate from a new induction round. With H+=0, use `check-task`, or `induce --relation` selecting an existing entry without body edits, or the existing `--attribute-effect` input. These maintenance-only calls need no positive job and create no empty induction receipt. New/changed bodies still need authorized induction. At execution declare `--used-entry-ids ENTRY@VERSION`; utility management resolves that actual version and needs a real adopted use plus the agent's reason.

```bash
orx induce --attribute-effect '{"entry_id":"3","execution_id":"ex_USED","verdict":"helped","lifecycle_action":"keep","note":"Agent inspected the actual use and declared checks."}'
```

`promote`, `demote` and `keep` act on versioned entries through the shared lifecycle; `retire` uses the existing cold archive. `keep` leaves status unchanged; `demote` suspends the selected version. An effect verdict `refuting` without `lifecycle_action` currently defaults to demotion of the adopted version. Cost comparability and benefit attribution are agent decisions. To compare costs explicitly select `comparator_execution_id` and resource dimensions in the assessment/validation. The framework returns both methods, actual conditions, acceptance and per-dimension values, masks/scopes and provenance. It neither finds a baseline nor substitutes another dimension or declares an advantage. Estimates remain estimates.

Frozen libraries reject maintenance writes as well as new induction. The main execution's frozen adoption fixes an effect's version. A candidate merge preserves its sources until that exact target version is published; confirmed merged sources are managed through their target. An attached induction/material ID makes evidence addition complete that formal review. Rules and receipts commit before external vector writes; a deferred index sync can retry the same submission with the same receipt. An unavailable optional cost baseline is reported separately and does not block task checks or H+ scheduling.

Omitting optional assessments on unchanged facts preserves them. `exclude-execution`/`restore-execution` correct evidence validity and future H+ support without automatically revalidating, pausing, restoring or publishing entries. The agent manages an entry from real use/counterexamples. Issued prediction inputs and original checker versions remain traceable.

For H+ training, `review.outcome` and a sourced `review.comparison` (or relation `mechanism_comparison`) describe what the agent learned. State `sufficient`, `substantive`, `source`, `assessment`, and valid `evidence_refs`; `basis:"derivation"` names a checked derivation over the cited actual inputs, while the default uses recorded checks. A substantive new/revised rule with supported review can yield 1 within that scope; supported coverage/no increment can yield 0 alongside evidence addition. Insufficient facts, missing review or withdrawn sole support remain unknown. Empty review can complete a job. No hash change, empty recall, publication or retirement alone supplies a label. See [induction.md](references/induction.md) and [hplus.md](references/hplus.md).

### Continue with settled evidence

Start the next task's `plan-next` in the same memory home. It automatically selects applicable published solving calibration, recalled rules and settled H+ receipts recorded before that decision. No extra feedback or prediction command is a normal prerequisite. For optional diagnosis, inspect a saved P and read `result.prediction.prediction.trace.model_info.request_omissions`; `orx context --context-id CTX --audit` exposes its frozen feedback references. A receipt retained in context may have been omitted from the actual provider request for budget. See [prediction_context.md](references/prediction_context.md) and [snapshot_lifecycle.md](references/snapshot_lifecycle.md).

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
