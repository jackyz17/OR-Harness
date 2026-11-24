---
name: or-harness-wm
description: >
  Formulate, solve, and verify operations research problems with OR-Harness.
  Use for LP, MILP, scheduling, routing, allocation, and coupled optimization
  tasks that benefit from comparing strategies, checking solutions, or
  reusing execution experience.
---

# OR-Harness

Use one memory directory (`--home` or `OR_HARNESS_HOME`) throughout the task stream. Configure `--world-model URL::MODEL` and `OR_WM_API_KEY` for planning.

## Workflow

```text
1. Prepare the task and recall experience
   |
   v
2. Propose candidates and select by U (plan-next)
   |
   v
3a. Build and execute the selected method
   |
   v
3b. Self-check the answer against the original task and record the attempt
   |-- Model/implementation wrong --> return to 2 for a fresh prediction
   |-- Check or evidence wrong ----> correct it on the same execution
   |
   v
4. Commit the answer and close the episode
   |
   +-- H+ = 1 review scheduled ---------> 5. Run offline induction
   |                                      |
   |                                      v
   +-- Otherwise -----------------------> 6. Next task (same memory)
```

### 1. Prepare and recall

Create `task.json` with `task_id`, `family`, and the original requirements in `text`. Keep that file and one episode label E across repairs. Add `coupling` (CIR) to represent relationships that affect the solution. Use [modeling.md](references/modeling.md) for task inputs and CIR.

```bash
orx profile --task task.json
orx recall --task task.json --top 5
```

Read the task profile, applicable rules, and related executions. Check their premises before adopting a method.

### 2. Select a method

Put one or more concrete methods in `candidates.json` with `name`, `steps`, `solver`, and any execution `config`. `plan-next` predicts G, C, R, and H+ in one call per candidate and selects by `U = alpha*G - beta*C - gamma*R`. Single candidates go through the same decision.

```bash
orx --world-model URL::MODEL plan-next --task task.json --episode E --candidates candidates.json
```

Read `result.selected_prediction_id` (P) and `result.selected_score`. Carry any `capability_gain.verification_conditions` from the selected row into the solve. If planning returns a refusal, resolve its cause and plan again. See [strategy_outcome.md](references/strategy_outcome.md) for scoring and settings.

### 3a. Build and execute

Implement the selected model and method in `ws/solve.py`. Write the observed solver status, objective, solution variables, and available bound and gap to `result.json`. Follow the [output format](references/commands.md#5-execute-and-check) to capture the method actually run. Include `--used-entry-ids ENTRY@VERSION` when adopting a rule.

```bash
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
```

### 3b. Self-check the answer, then decide to repair, re-solve or commit

After `execute`, return to the ORIGINAL task and check your own model, code and
solution. A self-check answers two questions, in this order:

1. Does the model correctly express the original task?
2. Does the actual result correctly satisfy that model and the task?

The solver's `optimal` status only says the model you wrote was solved. It does
not prove that the model is the task, and adding more verification methods does
not by itself make the check reliable. Choose the checks the task's real
difficulty and uncertainty require, in two layers.

**Layer 1 — necessary base check (every task).** Re-read the original statement
and inputs and check, where they apply:

- the objective and the quantity it actually optimises;
- the meaning and domain of each decision variable (integer, binary, continuous, sign);
- units — quantity, price, capacity, time — and the billing or cost range;
- completeness of the key constraints, especially coupling or aggregation
  relations that are easy to misread;
- feasibility of the returned solution against the original constraints;
- agreement of the objective value, constraint residuals and solver status;
- for a task that needs an optimality argument: whether the available bound,
  gap, dual or other evidence is enough.

These are check dimensions, not a fixed checklist. A genuinely continuous
problem needs no integer check, and a task with no time dimension needs no
constructed time-constraint check. Base checks must produce real evidence —
code review, recomputation of constraints, a short calculation, or the solver's
own report — but they do not require a new verification script per item.

```text
candidate answer
   |
   v
re-read the ORIGINAL task; run the base check on model meaning, domains,
units, key constraints, feasibility and objective/residual/status agreement
   |
   v
is there a concrete reason for enhanced verification?
   |-- no --> read the evidence and decide
   |-- yes
   v
run ONE targeted independent check (enumeration, independent script,
bound/dual, second exact algorithm or solver) and read its ACTUAL evidence
   |
   v
locate the finding: the task reading? the model? the implementation?
the recording? or the check itself?
   |
   +-- check or evidence wrong/incomplete --> correct it and re-run
   |                                          `check-task` on the SAME EX
   +-- model/implementation wrong -----------> new `plan-next`, then execute
   |
   v
self-check done --> commit (record the attempt, close the episode)
   |
   v
the benchmark/reference comparison happens AFTER the commitment
```

**Layer 2 — enhanced verification only when needed.** Options include a small
enumeration, an independent verification script, a dual or bound argument, a
second exact algorithm, a second solver, or another independent check suited to
the structure. Use one only when there is a concrete reason, such as: optimality
cannot be confirmed from the available evidence; the solution or objective looks
anomalous; a constraint interpretation is genuinely ambiguous; the solver's
return disagrees with an independent computation; the structure is complex
enough that the base check cannot cover a key error; or a specific modelling
assumption needs independent validation.

Do NOT run a complete independent re-solve as a default step for every task.
This does not forbid independent verification: when a second algorithm is
genuinely how a complex problem gets a reliable answer, run it. It forbids the
automatic "another full solve per task" habit.

**Verify the model, not only the solver.** Verification of the model is not
verification of the task:

- a solver's `optimal` proves optimality for the model as written, not that the
  model matches the statement;
- a second solver solving the same wrong model can return the same wrong answer;
- a dual, enumeration or independent algorithm can prove a mathematical result
  while sharing a wrong objective or constraint interpretation.

Check the model-to-statement correspondence BEFORE relying on an independent
numerical verification. No extra semantic-review tool or world-model call is
needed for this.

**H+ verification conditions are a separate check.** When the selected
prediction has H+=1, its `verification_conditions` test the predicted knowledge
hypothesis. They are not a second task requirement. Reuse the same evidence
when it supports both. If the condition cannot be checked within a reasonable
scope, record `not_checked`, `unavailable` or the protocol's unknown state; do
not run a second complete solve just to make the condition show as complete.
The offline review decides whether the evidence is sufficient.

`check.json` declares only the deterministic checks the framework can compute.
It is not the whole self-check, and a basis that does not apply to the task must
NOT be declared merely for form: a genuinely continuous problem needs no
`integer` check, and the framework does not infer that a missing declaration is
an uncovered requirement. EX is `result.execution_id` from `execute`.

```bash
orx check-task EX --check check.json [--verification verify.json]
orx record --from-staged EX
```

Read `result.state`:

- `passed`: the DECLARED checks held on the recorded values. It covers those
  bases only, it is not a proof that the model matches the task, and it does
  NOT stop you from planning a different candidate when your own review finds a
  problem.
- `failed`: a declared check found a violation. The framework does not decide
  whether the task reading, the model, the code or the check itself is wrong.
  Analyse which one it is. A bad check, a broken verification script or missing
  evidence is corrected and re-run on the SAME execution; a new attempt is
  planned only when the model or implementation is the real problem.
- `insufficient`: the declared check could not decide. It is neither a pass nor
  a failure, and it does not by itself require a complete re-solve.

**Never mark the self-check complete or passed when a verification script
raised or an assertion failed without handling it.** A failed verification is
evidence about the model, the implementation, the task reading or the script;
decide which, and act on that decision.

`report.scope.note` states the checker's fixed scope; `report.checks` and
`report.diffs` hold what actually ran and the numeric differences;
`report.scope.unchecked` names only declared checks that could not run.
`reflection_material` gives the task text/version reference, the candidate and
performed method, the captured artifacts and solution (with references), the
check report, any agent-run verification, and prior attempts as references.

A verification you ran yourself can be attached to the same execution with
`--verification`: the framework captures the script, output and result as
artifacts and labels them `agent_reported`. They are kept separate from the
framework checks — a satisfied assertion is your claim, not a framework verdict
— and this path creates no solving attempt and overwrites no `result.json`.

Record every real attempt and attach available host usage with
`record --usage-file`. See [task-check fields](references/commands.md#5-execute-and-check)
and [the self-check examples](#self-check-examples).

Start at most three retries after the first execution, within the episode
budget. Keep the same task and episode and obtain a fresh scored selection from
`plan-next` before each NEW execution. A re-check or a corrected check on the
same execution is not a new attempt and does not consume a retry.

### 4. Commit the answer and close the episode

Commit the final result and state what the self-check covered: the check verdict, any verification you attached, and anything you could not check. Close with the actual terminal state (`completed`, `failed`, `aborted`, or `budget_exhausted`). T is the task ID.

```bash
orx close-episode --task T --episode E --terminal STATE
```

In a benchmark, the evaluator scores the committed answer against any reference outside the solving and retry loop.

Close-out also returns `result.entry_usage`: the strategy entries THIS episode
actually adopted (the version is resolved from their frozen `used_entries`),
each with its read-time usage summary and the adoptions that still have no
effect judgement. Review that list BEFORE moving on: how did each adoption
turn out — did it help a specific step, prevent an error, improve modelling or
implementation, or just repeat interface prose? Record the judgement with the
same close-out call:

```bash
orx close-episode --task T --episode E --terminal STATE \
  --attribute-effect '{"entry_id":"2","execution_id":"EX_USED","verdict":"helped","note":"The adopted rule exposed the unit first; the objective then reproduced the reference scale."}'
```

This works when H+ = 0 as well: it is management of existing knowledge, not a
new induction job. A `refuting` verdict keeps the existing default (the
version is paused); `keep`/`demote`/`promote` is your explicit management
decision. Retiring stays the explicit `orx retire --entry N --reason ...`
(with `--evidence-execution-id` for the cited facts). A version must not be
hurt by evidence about an older version, and a paused version is restored only
by an explicit management decision.

Read the status WITHOUT reading it as usefulness: `validated`/`published`/
`fact_checked` say what checks the entry passed, an adoption count is a usage
fact, and only the agent's `helped|neutral|unrelated|refuting` judgements say
how the use turned out. `orx inspect --bank strategic --entry N` shows
`status_view` (publication / verification / usage separately), `utility` per
version and `forward_calibration` (a calibration diagnostic, not utility).
See [existing-entry use and utility](references/induction.md#existing-entry-use-and-utility).

Close-out publishes calibration and returns scheduled induction IDs in `result.induction_schedule.pending_ids`. When building the library, review each executed prediction with H+ = 1. Continue to the next task when no review is scheduled.

### 5. Run offline induction

Fetch evidence for each induction ID J.

```bash
orx induction-material --task T --induction-id J
```

Compare the actual methods, task constraints, checked outcomes, and relevant existing rules. During this post-answer review, available reference answers, checker verdicts, or user correctness feedback can supply evidence; cite their sources and coverage. Form a conclusion about the reusable method and its supported scope, using a contrast, derivation, or small executable check as needed.

A positive trigger schedules the review; it does not assert that knowledge was gained, and writing a rule is never required. A normal modelling/solving run, a method already covered by a published rule, or missing verification supports `no_new_knowledge`, `already_covered` or `insufficient_evidence`. Record what happened to the original hypothesis separately from the learning result with the existing `hypothesis_outcome` and `deviation_reason`; a later unrelated finding does not turn a refuted hypothesis into a supported one. Keep a durable rule decidable and executable at a future decision point: no benchmark answer, hidden-reference difference or post-hoc evaluation result may be part of its `when`.

Compare the frozen prediction with the actual execution and identify the affected premise, step, or execution condition when evidence supports it. Separate confirmed causes from hypotheses and unknowns. Keep attribution brief and sourced; see [G attribution evidence](references/episode_closeout.md#g-attribution-evidence). Post-answer reference feedback supports this review, not a return to solving or changing the committed answer.

Submit that conclusion as ONE compact JSON object with
`orx induce --induction-id J --submission ...`. It contains `result`, the
`rule` in **when / do / expect** (no separate predicates needed), one `basis`
with `kind`, real evidence `refs` and a short `analysis`, plus an optional
`publication` choice. Choose `result` as `new_rule`, `revision`,
`evidence_added`, `already_covered`, `no_new_knowledge` or
`insufficient_evidence`; use that single field instead of restating the
conclusion under several names. If material is insufficient, a short
`insufficient_evidence` submission can settle J without a rule. See
[minimal submissions](references/induction.md#minimal-submissions) for the
field table and [detailed comparisons](references/induction.md#complete-review-submissions)
for examples. The legacy `--relation`/`--review` spelling remains only for
advanced compatibility and cannot be mixed with `--submission`.

When unsure about the input, optionally add `--dry-run` to the same submission command. Fix any reported field or evidence reference, then remove `--dry-run` to submit. A successful preview leaves J pending; check `induction-status` after the real submission.

### 6. Continue with the next task

After completing the required reviews, start the next task in the same memory directory. The next `plan-next` can use applicable calibration, recalled rules, and completed reviews.

## NL4Opt example · `nl4opt_0184`

Save the rice transport problem as `task.json`.

```json
{
  "task_id": "example-nl4opt-0184",
  "family": "transportation",
  "text": "Choose integer counts M and L of medium and large carts to maximize rice transported. Medium carts require 2 horses and carry 30 kg each. Large carts require 4 horses and carry 70 kg each. Use at most 60 horses, at least 5 carts of each size, and exactly 3 medium carts for every large cart."
}
```

Submit this method in `candidates.json`.

```json
[{
  "name": "Substitute the cart ratio and enumerate one count",
  "steps": [
    "Substitute M = 3*L into the horse limit to bound L. Enumerate integer L and keep pairs that satisfy both cart minima.",
    "Compare 30*M + 70*L across feasible pairs and report both counts and the best weight."
  ],
  "solver": "python"
}]
```

After execution, self-check before committing: recompute the cart ratio,
horse usage and objective from the returned counts, verify both cart minima,
and use `integer` on M and L plus `recompute_objective` with coefficients
`{"M": 30, "L": 70}` only because the task states integer carts. If the
declared checks pass but your own review finds the model wrong, still plan a
new candidate; if a check/evidence is wrong, correct it on the same execution.
This small task needs only the base check: the returned counts are the whole
decision and recomputation covers the stated constraints, so do not add a
second solver run.

## Self-check examples

These examples show the judgment the framework leaves to the agent: the tool
gives numbers and references, not the decision.

### B018 — a consistent answer with the wrong unit

The task prices each chair at 50 and each order at 15 chairs, so one order in
the objective is 15 × 50 = 750. Suppose the script used 50 per order and the
solver reported a matching optimum. Recomputing the objective with the same
wrong coefficient reproduces the same number: agreement proves the model is
solved consistently, NOT that its unit is right. The self-check is to read the
unit from the task, compute one order by hand (750), and compare it with the
coefficient in the code. The framework's `recompute_objective` uses the
coefficients the agent declares, so declaring the code's coefficient would
just repeat the bug; either the model is corrected through a new `plan-next`,
or the task's coefficient is declared and the difference is reported.

### 0087 — a claimed optimum the evidence rejects

A continuous LP was reported optimal at 20, while a grid search found feasible
points near 17.725 and a dual bound near 17.714. The size of that difference is
enough to review the corner calculation: with the task as stated (continuous,
no integrality requirement), 17.714 has an optimality argument because the
bound is achievable. Do not add an integer constraint because the reference
value happens to be integral — the task is continuous, so the finding is a
calculation/verification problem, not a missing domain. Record the bound or
derivation as the optimality evidence; the agent decides that, not the tool.

### 0312 — two solvers agree on the same wrong model

A task asks to minimise total cost, but the script maximises revenue and then
negates the number. A second solver reproduces the same value. The agreement
proves the model is solved consistently; it proves nothing about the objective
semantics. The base check is to read the objective from the task (what is
minimised or maximised, in which unit) and compare it with the code before
trusting any independent verification. If the objective is wrong, plan a new
candidate; a more thorough verification of the same wrong model cannot fix it.

### A failed verification script is not a pass

If a verification script raises or an assertion fails, the self-check is not
done. Decide whether the model, the implementation, the task reading or the
verification script is wrong. A broken verification script or a bad check
declaration is fixed and re-run on the SAME execution with `check-task`; it
does not create a new solving attempt and it cannot be recorded as a pass.

## References

- [commands.md](references/commands.md) for CLI syntax, host usage, and recovery.
- [episode_closeout.md](references/episode_closeout.md) for acceptance evidence, G feedback, and calibration.
- [induction.md](references/induction.md) for H+ review dispatch, induction, publication, and existing rule use.
- [world_model_contract.md](references/world_model_contract.md) for prediction inputs, forecast fields, and request audit.
- [snapshot_lifecycle.md](references/snapshot_lifecycle.md) for saved inputs, fixed-memory testing, and maintenance.
