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

In formal experiments, cross-task memory comes only through the configured Harness banks. Per-task scratch files are allowed; do not persist extra solving notes for reuse across tasks.

## Workflow

```text
1. Prepare the task and recall experience
   |
   v
2. Propose candidates and select by U (plan-next)
   |
   v
3a. Write the solve code for the selected method, then execute
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

For file tasks, declare only public inputs in `input_resources`:
`[{"source":"data/nodes.csv","path":"network/nodes.csv"}]`; read `inputs/network/nodes.csv` in the solve. Sources are relative to the CLI task file or Python API caller's cwd. Text-only tasks omit the list. See [file input and isolation](references/commands.md#file-input-and-isolation).

```bash
orx profile --task task.json
orx recall --task task.json --top 5
```

Read the returned profile, applicable rules and related executions once; reuse those fields and inspect only a missing detail needed for the next decision. Check rule premises before adoption.

### 2. Select a method

Before planning, inspect data/schema/scale and mathematical structure; read `doctor` or `execution_constraints` for APIs and limits. Discovered modules have unknown backend/license usability. Use tiny artificial instances for API checks and report analysis/probing/coding costs through host usage.
Original-task LP relaxations, subproblems, search/enumeration, column generation, optimization-based reductions, solver comparisons and real-instance code trials belong inside formal execute.

Put one or more concrete methods in `candidates.json` with `name`, `steps`, `solver`, and any execution `config`. `plan-next` predicts G, C, R, and H+ in one call per candidate and selects by `U = alpha*G - beta*C - gamma*R`. Single candidates go through the same decision.

```bash
orx --world-model URL::MODEL plan-next --task task.json --episode E --candidates candidates.json
```

Read `result.selected_prediction_id` (P) and `result.selected_score`, then write
the solve code for the selected candidate. The framework establishes P's
implementation observation anchor when the decision completes, so `execute`
binds to the same anchor. Follow `plan-next → write code → execute → check →
fix/retry → settlement → H+ induction`.

Carry any `capability_gain.verification_conditions` from the selected row into
the solve. If planning returns a refusal, resolve its cause and plan again. See
[strategy_outcome.md](references/strategy_outcome.md) for scoring and settings.

### 3a. Write the solve code and execute

Write the selected model and method in `ws/solve.py`. Write the observed solver status, objective, solution variables, and available bound and gap to `result.json`. Follow the [output format](references/commands.md#5-execute-and-check) to capture the method actually run. Include `--used-entry-ids ENTRY@VERSION` when adopting a rule.

Report evidenced defects, actually started correction rounds and honest coverage
through existing action outcomes using `prediction_id: P`; the framework resolves
the candidate's selection anchor. Preserve failure
versions, repair diffs and checks by reference; see
[build6.4 host reports](references/build64_implementation.md). Local correction
reports do not consume or increase formal execution retry permission.

One candidate/script can read files, solve an LP, derive a safe reduction and solve the final model. Predict/execute it once, without separate algorithm-step predictions. Carry out the actual solve here; do not pre-solve the instance in the host terminal and replay its answer in execute.
Dynamic `open`, `pathlib` and installed readers share the input boundary. Inputs are read-only; write `result.json` and temporary files in `scratch/`.

```bash
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
```

### 3b. Self-check the answer, then decide to repair, re-solve or commit

After `execute`, return to the ORIGINAL task and answer two questions, in
order: does the model correctly express the task, and does the actual result
satisfy that model and the task? A solver's `optimal` proves only that the
written model was solved; it does not prove the model is the task.

Every task needs the base check on the objective, variable meaning and domain,
units, key constraints, feasibility, and the agreement of objective value,
constraint residuals and solver status. Add an enhanced check (enumeration,
independent script, bound/dual, second exact algorithm or solver) only when
there is a concrete reason: uncertain optimality, an anomalous value, an
ambiguous constraint reading, a solver/independent disagreement, or a structure
the base check cannot cover. Use the minimum sufficient verification: keep
adding targeted checks while an unresolved doubt can still change the
conclusion, and stop once the evidence is sufficient. Do not run a complete
independent re-solve as a default step for every task, and do not treat a fixed
number of checks as an automatic limit.

```text
candidate answer
   |
   v
re-read the ORIGINAL task; base check the model, units, constraints, solution
and solver report
   |
   v
concrete reason for enhanced verification? -- no --> read the evidence
   |-- yes
   v
run the cheapest targeted check for the open doubt; read the ACTUAL evidence
   |
   v
does a substantive doubt still block the conclusion? -- yes --> another
   |                                                        targeted check
   no
   v
locate the finding: task reading? model? implementation? recording? the check?
   |
   +-- check or evidence wrong/incomplete --> fix it and re-run on the SAME EX
   +-- model/implementation wrong ----------> new `plan-next`, then execute
   |
   v
self-check done --> final check-task + supported G attribution (or Unknown)
   |
   v
record the attempt --> commit the answer --> close the episode
```

A second solver on the same wrong model still returns the wrong answer: check
the model-to-statement correspondence before trusting any independent
verification. When the prediction has H+=1, its `verification_conditions` check
the knowledge hypothesis, not a second task requirement; reuse evidence, record
`not_checked`/`unavailable` when it cannot be checked, and never run a large
complete re-solve merely to make it show complete — an affordable small-scale
check that actually verifies the mechanism is allowed. See [self-check detail and
examples](references/self_check.md) for the layer criteria, failure handling and
worked cases.

`check.json` declares the deterministic checks the framework can compute; it is
not the whole self-check, and an inapplicable basis must not be declared for
form. After reviewing the captured method/code/result, include one sourced
`g_attribution` in the same JSON through the [G attribution interface](references/episode_closeout.md#g-attribution-evidence).
Use `method_fidelity` and `acceptance_scope` honestly; when the evidence cannot
support a judgement, record `unknown` with the available references. EX is
`result.execution_id` from `execute`.

```bash
orx check-task EX --check check.json [--verification verify.json]
orx record --from-staged EX
```

Submit this final check before `record` and `close-episode`; do not repeat a
completed check on unchanged evidence. `result.g_assessment_ref` acknowledges
the saved attribution for that check revision. Invalid or
misplaced attribution is rejected before the check is written. `check-task` is
the sole G submission interface; `action --report verify` only reports an
external action. The framework derives G from the saved facts at close-out.
Unknown is a normal outcome: do not repeat checks or run a full independent
solve merely to obtain G=1 or G=0; re-check only for changed facts or a corrected
check.

Read `result.state`:

- `passed`: the declared checks held; not proof that the model matches the task,
  and not a block on planning a different candidate.
- `failed`: a declared check found a violation. Decide whether the task reading,
  model, implementation or check is wrong; fix a bad check on the SAME
  execution and re-plan only for a real model/implementation problem.
- `insufficient`: neither a pass nor a failure, and not by itself a reason for a
  complete re-solve.

A failed or errored verification is evidence, not a pass. Attach your own
verification with `--verification`: it is captured as `agent_reported` evidence
on the same execution, separate from the framework checks, and creates no
solving attempt.

Record verification artifacts and actual cost through the existing verify action or host usage report. A new solving method or substantially revised answer returns to plan-next and execute; verification cannot bypass binding/retry limits.

`reflection_material` carries the task reference, method, artifacts, solution,
check report and prior attempts. Reuse the returned evidence; batch necessary
reads rather than issuing an inspect per field or rereading unchanged material. Record every real
attempt with `record --usage-file`. Start at most three retries after the first
execution; a re-check on the same execution is not a new attempt. See
[self-check detail and examples](references/self_check.md) and
[task-check fields](references/commands.md#5-execute-and-check).

### 4. Commit the answer and close the episode

After the final check (including supported G attribution or honest Unknown) and record, commit the result and state the self-check's coverage. Then close with the actual terminal state (`completed`, `failed`, `aborted`, or `budget_exhausted`). T is the task ID. Do not make closeout the normal point for collecting G attribution; do not run another plain `check-task` after the final attributed check unless facts or the acceptance basis changed.

```bash
orx close-episode --task T --episode E --terminal STATE
```

In a benchmark, the evaluator scores the committed answer against any reference outside the solving and retry loop.

Close-out returns `result.entry_usage`: this episode's adopted rule versions,
their read-time usage summary, and the adoptions that still need an effect
judgement. Review them before moving on — did an adoption help a specific step,
prevent an error, change modelling or implementation, or just repeat interface
prose? Record a judgement on the same call when the evidence supports one:

```bash
orx close-episode --task T --episode E --terminal STATE \
  --attribute-effect '{"entry_id":"2","execution_id":"EX_USED","verdict":"helped","note":"The adopted rule exposed the unit first; the objective then reproduced the reference scale."}'
```

This also works when H+ = 0: it manages existing knowledge, not a new induction
job. `validated`/`published`/`fact_checked` are check status, adoption counts are
usage facts, and only your own `helped|neutral|unrelated|refuting` verdicts are
effect. See [existing-entry use and utility](references/induction.md#existing-entry-use-and-utility)
for entry status, utility, management actions and version rules.

Close-out publishes calibration and returns scheduled induction IDs in `result.induction_schedule.pending_ids`. When building the library, review each executed prediction with H+ = 1. Continue to the next task when no review is scheduled.

### 5. Run offline induction

H+=1 requires one review; H+=0 adds no induction job. Fetch evidence for each
scheduled induction ID J.

```bash
orx induction-material --task T --induction-id J
```

Assess the frozen H+ claim first against the actual method, task constraints and checked evidence. Extract the mechanism that changes a future decision: reductions, monotonicity, dominance, valid bounds, analytic certificates, applicability boundaries or implementation repairs. Avoid turning the execution steps into a repeated-solve recipe. Compare existing rules by When/Do/Expect; past success or similar terms do not prove coverage. Post-answer references and checker feedback can supply evidence with explicit sources and scope. Reuse a sufficient cheap certificate; further checks need an unresolved claim.

A positive trigger schedules the review, not a rule. The review may settle as `already_covered`, `no_new_knowledge` or `insufficient_evidence`, and it records the original hypothesis outcome separately from the learning result; durable rules must be decidable at a future decision point. See [review decisions](references/induction.md#review-decisions) and [durable knowledge](references/induction.md#durable-knowledge-must-be-usable-by-a-future-decision).

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

When unsure about the input, optionally add `--dry-run` to the same submission command. Fix any reported field or evidence reference, then remove `--dry-run` to submit. A successful preview leaves J pending; run `orx induction-status` after the real submission.

### 6. Continue with the next task

After completing the required reviews, start the next task in the same memory directory. The next `plan-next` can use applicable calibration, recalled rules, and completed reviews.

The external Runner enforces hard stops, session lifetime/write locks, public-input authorization, host-terminal pre-solving audits and cross-task scratch/note isolation. OR-Harness observes submitted executions and reported usage; it cannot monitor arbitrary host-terminal solving.

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
This small task needs only the base check; do not add a second solver run.

## References

- [commands.md](references/commands.md) for CLI syntax, host usage, and recovery.
- [self_check.md](references/self_check.md) for self-check layers, failure handling, and worked examples.
- [episode_closeout.md](references/episode_closeout.md) for acceptance evidence, G feedback, and calibration.
- [induction.md](references/induction.md) for H+ review dispatch, induction, publication, and existing rule use.
- [world_model_contract.md](references/world_model_contract.md) for prediction inputs, forecast fields, and request audit.
- [snapshot_lifecycle.md](references/snapshot_lifecycle.md) for saved inputs, fixed-memory testing, and maintenance.
