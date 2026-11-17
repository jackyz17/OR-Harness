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
3b. Check against the original task and record the attempt
   |-- Retry justified --> return to 2 for a fresh prediction
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

### 3b. Check the original task and decide whether to retry

Before committing the answer, use only the task requirements, original inputs, and execution evidence for verification and retry decisions. Match the model to the requirements. Check variable domains and independently recompute constraints and objective from the returned solution. Use a bound, gap, or independent argument to support optimality when required. A checker based solely on the task and its inputs can perform these checks.

Put applicable deterministic checks in `check.json`, such as `integer`, `recompute_objective`, or `semantic_probe` on recorded values. EX is `result.execution_id` from `execute`.

```bash
orx check-task EX --check check.json
orx record --from-staged EX
```

Read `result.state` and the declared coverage. Record every attempt and attach available host usage with `record --usage-file`. Correct a check or supplement its evidence on the same EX when possible. See [task-check fields](references/commands.md#5-execute-and-check).

Retry when a specific change can address a technical failure, an unmet task requirement, or missing evidence that requires a new solve. Start at most three retries after the first execution, within the episode budget. Keep the same task and episode and obtain a fresh scored selection from `plan-next` before each new execution.

### 4. Commit the answer and close the episode

Commit the final result and state what was verified. Close with the actual terminal state (`completed`, `failed`, `aborted`, or `budget_exhausted`). T is the task ID.

```bash
orx close-episode --task T --episode E --terminal STATE
```

In a benchmark, the evaluator scores the committed answer against any reference outside the solving and retry loop.

Close-out publishes calibration and returns scheduled induction IDs in `result.induction_schedule.pending_ids`. When building the library, review each executed prediction with H+ = 1. Continue to the next task when no review is scheduled.

### 5. Run offline induction

Fetch evidence for each induction ID J.

```bash
orx induction-material --task T --induction-id J
```

Compare the actual method, task constraints, and checked outcomes across cases. During this post-answer review, use available reference answers, checker verdicts, or user correctness feedback as evidence. Cite their sources and what they check. Test the proposed scope with an available contrast, a derivation, or a small executable case. Submit the conclusion with `--review` using the same J, including when the available evidence supports no rule. For a new or revised rule, add **when / do / expect**, evidence IDs, and a publication choice in `--relation`. Follow the [complete submission examples](references/induction.md#complete-review-submissions).

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

After execution, check integrality and both cart minima. Recompute the cart ratio, horse usage, and objective from the returned counts. Use `integer` on M and L and `recompute_objective` with coefficients `{"M": 30, "L": 70}`.

## References

- [commands.md](references/commands.md) for CLI syntax, host usage, and recovery.
- [episode_closeout.md](references/episode_closeout.md) for acceptance evidence, G feedback, and calibration.
- [induction.md](references/induction.md) for H+ review dispatch, induction, publication, and existing rule use.
- [world_model_contract.md](references/world_model_contract.md) for prediction inputs, forecast fields, and request audit.
- [snapshot_lifecycle.md](references/snapshot_lifecycle.md) for saved inputs, fixed-memory testing, and maintenance.
