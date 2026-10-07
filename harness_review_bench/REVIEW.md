# OR-Harness 记忆库 —— bench_orarla/home（round r11 十题跑）

- 生成：2026-10-07 20:31
- DB：`/home/ubuntu/.openclaw/workspace/bench_orarla/home/or_harness.db`
- bank 层： experience=18 | strategic=14 | archive=0 | actions=81 | snapshots=83 | predictions=27 | texts=10 | evaluations=18 | retention=None | capability=0

- DB 表：   action_records=81 | belief_snapshots=83 | capability_predictions=0 | cold_archive=0 | contract_predictions=27 | episode_closeouts=10 | executions=18 | meta=77 | pending_executions=0 | prediction_contexts=19 | strategic_entries=14 | task_texts=10 | world_model_predictions=0

## 策略库 (Strategic)

- `se_dab6d0c168a0` orarla_1_pulp_cbc_recovery [candidate] support_n=0
    - When an ILP formulation for a small planning problem (2 integer decisions, ~4 constraints) runs in PuLP but fails with HiGHS backend (solver not found or API mi
- `se_d8fe85d9c17c` orarla_1_pulp_cbc_fallback [candidate] support_n=0
    - On small planning ILPs (2 integer decisions, ~4 constraints) where PuLP HiGHS backend fails (solver-not-found or API error), the same ILP formulation with PuLP 
- `se_e0b1a8f5d4e6` orarla_1_pulp_cbc_optimal [candidate] support_n=0
    - On the orarla_1 planning problem (2 integer decisions, 4 constraints, family=planning), PuLP+CBC (pulp.PULP_CBC_CMD) achieves optimal status=10000.0 and passes 
- `se_667383892293` orarla_2_pulp_cbc_optimal [candidate] support_n=0
    - On small budget-allocation planning problems (2 integer decisions, 6 constraints, family=planning) with constraints of the form X+Y<=MAX, a*X+b*Y>=TARGET, X,Y i
- `se_ae6e9d118007` pure_python_enumeration_optimal_on_small_planning [candidate] support_n=0
    - On small planning problems (family=planning, 2 integer decisions, ≤5 linear constraints, bounded variable domains) where the feasible space is enumerable within
- `se_fa3d7c2ee405` PuLP_CBC_optimal_on_two_campaign_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with resource-total constraint (X+Y<=MAX) and a multiplicative difference constra
- `se_7a2af95c4664` orarla_5_pulp_cbc_optimal_on_three_constraint_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with three constraints: resource-total (X+Y<=MAX), effectiveness lower-bound (a*X
- `se_8752540fffc4` PuLP_CBC_optimal_on_two_constraint_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with exactly two constraints: resource-total (X+Y<=MAX) and effectiveness lower-b
- `se_81f2aaba75f1` PuLP_CBC_optimal_on_two_constraint_balance_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with exactly two constraints: resource-total (X+Y>=MIN) and a balance/difference 
- `se_6ab7192b458b` orarla_7_task_check_undeclared_reference [candidate] support_n=0
    - The orarla_7 task_check passed (status=optimal, objective=3000.0) but the caller did not declare reference_source as bench_declared/independent/self_derived. Pe
- `se_aa8fdfdc3a76` PuLP_CBC_optimal_on_two_constraint_resource_effectiveness_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with exactly two constraints: resource-total (X+Y<=MAX) and effectiveness lower-b
- `se_166144327da7` task_check_passed_with_undeclared_reference_source [candidate] support_n=0
    - When a task_check reports state=passed for an optimal solution, the caller MUST declare reference_source as bench_declared/independent/self_derived. If referenc
- `se_b9cf043e821a` PuLP_CBC_optimal_on_two_constraint_lower_and_upper_bounded_planning [candidate] support_n=0
    - On small two-campaign planning problems (family=planning, 2 integer decisions) with exactly two constraints: a lower-bound on the combined resource (Ad1 + Ad2 >
- `se_7e7d5ac16ce3` PuLP_CBC_optimal_on_energy_investment_two_constraint_planning [candidate] support_n=0
    - On the orarla_10 energy company investment problem (family=planning, 2 integer decisions) with two constraints: resource-total (X+Y<=5000) and effectiveness low

## 世界模型校准 (wm-calib/2)

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt|wm-obs/2`: basis=partially_measured n_distinct_episodes=10 benefit_MAE=0.0 signed=0.0 coverage=1.0 width=0.0
    - cost_log_ratio={'latency_s': -0.830882, 'llm_tokens': 0.943225, 'solver_runtime_s': -0.068118, 'tool_calls': 0.810776}
    - brier_by_event={'environment_failure': 0.00385, 'implementation_failure': 0.360147, 'task_check_failed': 0.0025}

- n_evaluations_total=18 n_window_episodes=10

