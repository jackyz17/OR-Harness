# OR-Harness 记忆库 —— bench_orarla/home（round r5 十题跑）

- 生成：2026-10-04 17:33
- DB：`/home/ubuntu/.openclaw/workspace/bench_orarla/home/or_harness.db`
- experience 30 | predictions 64 | evaluations 24 | actions 116 | strategic 7 | archive 0

## 策略库 (Strategic)

- `se_d7b30fd6f261` milp_labor_training [candidate] support_n=0
    - The milp_labor_training MILP for the 8-week labor training planning problem (50-trainee target, dedicated trainers producing 3 trainees per 2-week period, integ
- `se_3088f84ef612` direct_calculation [candidate] support_n=0
    - The direct_calculation arithmetic approach solves the orarla_2 task to optimal (objective=125.0, reference match) when executed without errors.
- `se_72e9b7a98d44` milp_aggregate_production_v2 [candidate] support_n=0
    - The milp_aggregate_production_v2 cumulative supply/demand MILP formulation (integer produce/outsource/hire/fire/workforce/cum_supply/cum_demand, continuous inv/
- `se_bd647a6c22ab` milp_lot_sizing [candidate] support_n=0
    - The milp_lot_sizing multi-product lot-sizing formulation (integer production lots, inventory, backlog per product per quarter; flow-balance constraints; final-s
- `se_16ae57b92d61` network_flow_transport [candidate] support_n=0
    - The network_flow_transport min-cost transshipment formulation (source->warehouse->port->sink arcs, integer flow variables, arc cost = 30 * distance[w][p], capac
- `se_621929fbd0ea` enumerate_assignment [candidate] support_n=0
    - The enumerate_assignment strategy (enumerate all feasible assignments, evaluate objective for each, select best) solves the orarla_8 task to optimal (objective=
- `se_161bf535d0fa` network_flow_transport on orarla_7 [candidate] support_n=0
    - When a transportation problem with supply-constrained warehouses and demand ports is formulated as a min-cost transshipment network (source->warehouse->port->si

## 世界模型校准 (wm-calib/2)

- `strategy_outcome|Qwen3.8-27B|normalized_objective_gap|1-gap|attempt`: basis=partially_measured n_distinct_episodes=9 benefit_MAE=0.596154 signed=0.45 coverage=0.384615 width=0.088462

- n_evaluations_total=24 n_window_episodes=10

