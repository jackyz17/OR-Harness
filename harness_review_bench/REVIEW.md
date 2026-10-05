# OR-Harness 记忆库 —— bench_orarla/home（round r6 十题跑）

- 生成：2026-10-05 16:29
- DB：`/home/ubuntu/.openclaw/workspace/bench_orarla/home/or_harness.db`
- experience 31 | predictions 57 | evaluations 22 | actions 105 | strategic 9 | archive 0

## 策略库 (Strategic)

- `se_926041a1a149` milp_training_allocation on orarla_2 [candidate] support_n=0
    - When a temporal production-allocation MILP (integer t1,t2 with multi-year pilot training dynamics) uses highspy 1.15, environment errors and model API errors do
- `se_3be154292829` milp_hiring_firing_decoupled on orarla_3 [candidate] support_n=0
    - When milp_hiring_firing_decoupled throws AttributeError from HighsModelStatus.to_string, retrying the same code without modification can recover to optimal solu
- `se_1b3c8b970bde` highspy_API_recovery [candidate] support_n=0
    - When LLM-generated HiGHS/highspy MILP code uses incorrect API symbols (ObjSense, ModelStatus, vartype kwarg, setMipRelGap) for orarla_4, correcting to the prope
- `se_2de4ec75579d` orarla_5 [candidate] support_n=0
    - When milp_direct_integer (or variant strategies on same task) reports solver-optimal but task_check fails on reference_objective, it indicates model-task mismat
- `se_5ff13b763846` milp_production_inventory_backlog on orarla_6 [candidate] support_n=0
    - When a production/inventory/backlog MILP solve fails with an environment error (ModuleNotFoundError), retrying the exact same strategy after the environment is 
- `se_96bb132168e5` orarla_7 / milp_transport_integer [candidate] support_n=0
    - When highs solver raises ModuleNotFoundError for a transport-integer task, switching the solver backend to ortools (CP-SAT) resolves the environment failure and
- `se_cefac66cf855` orarla_7 / milp_transport_integer via ortools [candidate] support_n=0
    - On a balanced transportation-integer problem (supply=demand), the same strategy and solver (ortools CP-SAT) can report infeasible on one attempt and optimal on 
- `se_541cf3070d11` enumerate_exclude_one on cardinality-constrained assignment (pick n-1 of n workers) [candidate] support_n=0
    - For a worker-task assignment problem with binary x[worker,task] where exactly (n-1) workers must be assigned to (n-1) tasks (one task per worker, each task cove
- `se_9baddc1c0fbf` highspy API compatibility in milp_direct_integer [candidate] support_n=0
    - When milp_direct_integer uses highspy >= 1.15, calling setObjectiveSense raises AttributeError; the correct API is changeObjectiveSense with ObjSense.kMaximize/

## 世界模型校准 (wm-calib/2)

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt|wm-obs/2`: basis=partially_measured n_distinct_episodes=7 benefit_MAE=0.775 signed=0.775 coverage=0.5 width=0.31875

- n_evaluations_total=22 n_window_episodes=10

