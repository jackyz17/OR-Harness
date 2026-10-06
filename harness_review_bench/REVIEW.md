# OR-Harness 记忆库 —— bench_orarla/home（round r9 十题跑）

- 生成：2026-10-06 18:20
- DB：`/home/ubuntu/.openclaw/workspace/bench_orarla/home/or_harness.db`
- experience 20 | predictions 28 | evaluations 3 | actions 61 | strategic 4 | archive 0

## 策略库 (Strategic)

- `se_878ae9607535` ilp_cbc infeasibility handling [candidate] support_n=0
    - When an ILP model returns infeasible status for a task with a known feasible answer, retrying with the same code_hash does not resolve the error; only code chan
- `se_4856cc1078e3` scipy_milp_constraint_matrix_failure_and_enumeration_recovery [candidate] support_n=0
    - For 2-variable integer linear planning tasks in the planning family, when scipy.optimize.milp is given a constraint that reformulates a variable lower bound as 
- `se_7beeea57e621` enumeration_with_constraints optimal 2-variable planning [candidate] support_n=0
    - For 2-variable integer linear planning tasks in the planning family, enumeration_with_constraints analytically derives variable bounds from combined constraints
- `se_033df2481b38` scipy_milp_integer vs enumeration_with_constraints recovery [candidate] support_n=0
    - For 2-variable integer linear planning tasks in the planning family, when scipy.optimize.milp consistently fails across multiple attempts (ValueError/TypeError 

## 世界模型校准 (wm-calib/2)

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt|wm-obs/2`: basis=insufficient_evidence n_distinct_episodes=2 benefit_MAE=0.0 signed=0.0 coverage=1.0 width=0.025
- `strategy_outcome|Qwen3.8-27B|task_result_check_passed|boolean|attempt|wm-obs/2`: basis=insufficient_evidence n_distinct_episodes=1 benefit_MAE=0.0 signed=0.0 coverage=1.0 width=0.15

- n_evaluations_total=3 n_window_episodes=10

