# OR-Harness ACTIVE memory bank -- /home/ubuntu/.openclaw/workspace/or_harness_home

- generated: 2026-10-08 20:30 (active-bank)
- DB: `/home/ubuntu/.openclaw/workspace/or_harness_home/or_harness.db`
- bank layers: experience=12 | strategic=5 | archive=1 | actions=41 | snapshots=52 | predictions=18 | texts=10 | evaluations=11 | retention=None | capability=0
- DB tables:   action_records=41 | belief_snapshots=52 | capability_predictions=0 | cold_archive=1 | contract_predictions=18 | episode_closeouts=8 | executions=12 | meta=42 | pending_executions=0 | prediction_contexts=13 | strategic_entries=5 | task_texts=10 | world_model_predictions=0

## Strategic entries

- `1` method:direct_integer_program_small_linear_allocation [candidate] support_n=0
    - For a SMALL linear resource-allocation MILP (bounded integer decision variables, a single shared capacity cap plus linear bound/difference constraints, linear cost objective), a single direct integer-
- `2` method:direct_exact_solve_small_linear_allocation [candidate] support_n=0
    - For a SMALL linear allocation MILP - two integer decision variables, one shared budget/capacity cap, at most one linear difference (effort/precedence) constraint, linear cost objective (structural cel
- `3` highspy_status_string_fix [candidate] support_n=0
    - When using Highspy solver in a solve.py that writes to result.json for the OR-Harness framework, the status field must be written as a canonical string (optimal/feasible/infeasible/unbounded/timeout/e
- `4` effectiveness_constraint_coefficients_are_literal [candidate] support_n=0
    - When a task defines effectiveness as 'twice advertising + three times promotion' and also separately states per-unit contribution rates (4 and 5), the constraint coefficients are the LITERAL twice/thr
- `6` method:direct_exact_solve_small_covering_allocation [candidate] support_n=0
    - Condition: a TINY (two-decision) linear allocation MILP that minimises a positive linear cost cX*X + cY*Y subject to (i) a shared cap X + Y <= C and (ii) a weighted-sum COVERING requirement aX*X + aY*

## World-model calibration

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt|wm-obs/3`: basis=insufficient_evidence n_distinct_episodes=1 benefit_MAE=0.0 signed=0.0 coverage=1.0 width=0.0
- `strategy_outcome|Qwen3.8-27B|task_result_check_passed|boolean|attempt|wm-obs/3`: basis=partially_measured n_distinct_episodes=6 benefit_MAE=0.396667 signed=-0.185556 coverage=None width=None

- n_evaluations_total=11 n_window_episodes=8

