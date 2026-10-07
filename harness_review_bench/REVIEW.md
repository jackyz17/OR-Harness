# OR-Harness 记忆库 —— bench_orarla/home（round r10 十题跑）

- 生成：2026-10-07 10:24
- DB：`/home/ubuntu/.openclaw/workspace/bench_orarla/home/or_harness.db`
- bank 层： experience=16 | strategic=14 | archive=0 | actions=82 | snapshots=76 | predictions=36 | texts=10 | evaluations=16 | retention=None | capability=0

- DB 表：   action_records=82 | belief_snapshots=76 | capability_predictions=0 | cold_archive=0 | contract_predictions=36 | episode_closeouts=10 | executions=16 | meta=75 | pending_executions=0 | prediction_contexts=17 | strategic_entries=14 | task_texts=10 | world_model_predictions=0

## 策略库 (Strategic)

- `se_ff271cc93174` enumeration_finds_optimal_corner_for_bounded_2var_lp [candidate] support_n=0
    - For a 2-variable linear minimization LP with non-negative variables, a minimum-excess constraint X - Y >= D (D > 0), a total-resource constraint X + Y <= B (B >
- `se_6b5c9eea4c5b` direct_milp_2var_produces_unchecked_trivial_solution [candidate] support_n=0
    - Direct MILP with PuLP on a 2-variable planning task with one total-budget constraint (X+Y <= B) and one reach/excess constraint (a*X - b*Y >= D) can return a tr
- `se_f66d4d7d52bc` direct_milp_2var_task_check_fails_despite_optimal_solver_status [candidate] support_n=0
    - Direct MILP with PuLP on a 2-variable planning task can return solver status=optimal while task_check.state=failed, because the reference objective was self-der
- `se_3ef7a0c5784e` enumeration_2var_bounded_finds_optimal_orarla_3 [candidate] support_n=0
    - For orarla_3, the analytical enumeration method (iterating A from 0 to 50, computing minimum P=ceil((100-2A)/3) for each, picking min A+P) finds the true optimu
- `se_ce6a083b96ad` highspy_milp_api_errors_on_2var_planning [candidate] support_n=0
    - highspy 1.15 has two distinct API errors when used for 2-variable MILP planning: (1) AttributeError: Highs has no setObjectiveSense (should be changeObjectiveSe
- `se_a63df0fe1e29` enumeration_and_milp_converge_same_solution_orarla_3 [candidate] support_n=0
    - For orarla_3, both the analytical enumeration method (ex_27ea23e58213) and the MILP method (ex_dbb149ab1846) converge on the identical solution A=0, P=34 with o
- `se_cc06f01a6a9a` enumeration_2var_ilp_direct_finds_optimal_for_orarla_4 [candidate] support_n=0
    - For orarla_4 (2-var planning: minimize 4X+3Y subject to X+2Y<=1000 and X-2Y>=300, X,Y>=0 integers), enumeration of Y from 0 to floor((1000-300)/3)=233 with X=ma
- `se_6e252be9c30b` enumeration_budget_2var: coupling_min_feasible_within_budget_implies_binding_coupling [candidate] support_n=0
    - For a 2-variable linear minimization with non-negative X,Y, budget ceiling X+Y<=B, coupling floor Y>=ceil((C-a*X)/b) (from a*X+b*Y<=C form), and a second linear
- `se_b6905c693cab` enumeration_2var_budget_effectiveness_prefers_cheaper_channel_full_allocation [candidate] support_n=0
    - For a 2-variable budget/effectiveness planning task (minimize c_X*X + c_Y*Y subject to X+Y<=B and a*X+b*Y>=D, X,Y>=0 integers), if channel X is cheaper per unit
- `se_c4a8c68388c4` highspy_koptimal_status_validation_error_causes_error_outcome [candidate] support_n=0
    - When highspy returns status koptimal (the C++ enum value) instead of the string optimal, the verification layer raises illegal status error and the execution re
- `se_945f18b67794` enumeration_2var_budget_effectiveness_finds_optimal_for_budget_effectiveness_task_where_cheaper_satisfies_alone [candidate] support_n=0
    - For a 2-variable integer budget/effectiveness planning task (minimize c_X*X + c_Y*Y subject to X+Y<=B and a*X+b*Y>=D, X,Y>=0 integers) where the cheaper channel
- `se_8529ae684e3f` highspy_1p15_milp_api_attributeerror_str_object_has_no_attribute_bounds_on_2var_planning [candidate] support_n=0
    - highspy 1.15 for a 2-variable MILP planning task (minimize c_X*X+c_Y*Y subject to X+Y<=B and a*X+b*Y>=D, X,Y>=0 integers) can raise AttributeError: str object h
- `se_a9a742d7553b` enumeration_2var_analytic_finds_optimal_when_coupling_binding_budget_nonbinding_orarla_9 [candidate] support_n=0
    - For a 2-variable integer linear minimization planning task (minimize c_x*x + c_y*y) with: (1) a coupling floor constraint binding at optimum (y = D - a*x from a
- `se_a801f9b3fc75` enumeration_2var_budget_effectiveness_finds_optimal_when_expensive_channel_has_better_effectiveness_ratio_orarla_10 [candidate] support_n=0
    - For a 2-variable integer budget/effectiveness planning task (minimize c_X*X + c_Y*Y s.t. X+Y<=B, a_X*X + a_Y*Y >= D, X,Y>=0 integers) where the nominal-cheap ch

## 世界模型校准 (wm-calib/2)

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt|wm-obs/2`: basis=partially_measured n_distinct_episodes=10 benefit_MAE=0.0 signed=0.0 coverage=1.0 width=0.0
    - cost_log_ratio={'latency_s': -1.131607, 'llm_tokens': 1.372353, 'solver_runtime_s': -0.60875, 'tool_calls': 0.880193}
    - brier_by_event={'implementation_failure': 0.140208, 'solver_reported_infeasible': 0.9025, 'task_check_failed': 0.154722}

- n_evaluations_total=16 n_window_episodes=10

