"""
ORClaw task orarla_1: Workforce & Production Planning (FIXED v2)
MILP formulation using HiGHS high-level API.
Selected strategy: milp_time_indexed

Key fix: workforce constraint should be sp + st + ow + tr <= N0,
where tr = trained workers available (cumulated from 2 weeks ago).
The original model had sp + st + ow <= N0, which excluded trained workers
from the workforce count, causing infeasibility in weeks 3-8.

Also fix: trained workers' wage should be 240/week, but the training cost
(nt[t] * 120 * 2 = nt[t] * 240) is correct.
"""

import json
import time
import highspy

def solve():
    h = highspy.Highs()
    h.setOptionValue("log_to_console", True)
    h.setOptionValue("time_limit", 60.0)
    h.setOptionValue("mip_gap", 0.0)
    h.setOptionValue("presolve", "on")

    # --- Parameters ---
    T = 8  # weeks 0..7
    DEMAND_I  = [10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000]
    DEMAND_II = [6000,  7200,  8400,  10800, 10800, 12000, 12000, 12000]
    RATE_I  = 10.0   # kg per worker-hour
    RATE_II =  6.0   # kg per worker-hour
    STD_HRS = 40
    OT_HRS  = 60
    W_SKILL = 360.0  # skilled worker weekly wage
    W_TRAIN = 120.0  # trainee weekly wage during training
    W_TR    = 240.0  # trained worker weekly wage after training
    W_OT    = 540.0  # overtime worker weekly wage (60h)
    PEN_I   =   0.5  # yuan per kg per week backlog
    PEN_II  =   0.6
    N0      = 50     # initial skilled workers

    # --- Variables ---
    # sp[t]: skilled workers producing (integer)
    # st[t]: skilled workers in training (2-week commitment, integer)
    # ow[t]: overtime workers (integer, each works 60h)
    # nt[t]: new trainees starting (integer, 2 weeks to become trained)
    # tr[t]: trained workers available for production (integer, cumulated from nt[t-2])
    # pI[t], pII[t]: production (continuous)
    # bI[t], bII[t]: backlog at end of week (continuous)

    sp  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kInteger, f"sp_{t}")  for t in range(T)]
    st  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kInteger, f"st_{t}")  for t in range(T)]
    ow  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kInteger, f"ow_{t}")  for t in range(T)]
    nt  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kInteger, f"nt_{t}")  for t in range(T)]
    tr  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kInteger, f"tr_{t}")  for t in range(T)]
    pI  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kContinuous, f"pI_{t}")  for t in range(T)]
    pII = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kContinuous, f"pII_{t}") for t in range(T)]
    bI  = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kContinuous, f"bI_{t}")  for t in range(T)]
    bII = [h.addVariable(0.0, float('inf'), 0.0,
              highspy.HighsVarType.kContinuous, f"bII_{t}") for t in range(T)]

    # --- Constraints ---

    # C1: Workforce balance (FIXED: includes tr)
    # All workers (producing + training + overtime + trained) come from initial N0
    for t in range(T):
        h.addConstr(sp[t] + st[t] + ow[t] + tr[t] <= N0, f"wbalance_{t}")

    # C2: Training capacity: nt[t] <= 3 * st[t]
    # (1 trainer handles up to 3 trainees in 2-week batch)
    for t in range(T):
        h.addConstr(nt[t] <= 3.0 * st[t], f"tcap_{t}")

    # C3: Trained worker accumulation: tr[t] = nt[t-2]
    # Workers starting training at t-2 finish at t and become available
    h.addConstr(tr[0] == 0.0, "tr_0")
    if T > 1:
        h.addConstr(tr[1] == 0.0, "tr_1")
    for t in range(2, T):
        h.addConstr(tr[t] - nt[t-2] == 0.0, f"tr_def_{t}")

    # C4: Total trained by week 8 >= 50
    h.addConstr(sum(nt[t] for t in range(T)) >= 50.0, "total_trainees")

    # C5/C6: Production capacity (total worker-hours)
    # Workers in training (st[t]) produce 0; trained workers (tr[t]) produce full 40h
    # overtime workers (ow[t]) work 60h
    for t in range(T):
        hours = (sp[t] + tr[t]) * STD_HRS + ow[t] * OT_HRS
        h.addConstr(pI[t]  <= hours * RATE_I,  f"cap_I_{t}")
        h.addConstr(pII[t] <= hours * RATE_II, f"cap_II_{t}")

    # C7/C8: Backlog balance for product I
    h.addConstr(bI[0] == DEMAND_I[0] - pI[0], "bI_0")
    for t in range(1, T):
        h.addConstr(bI[t] == bI[t-1] + DEMAND_I[t] - pI[t], f"bI_{t}")

    # C9/C10: Backlog balance for product II
    h.addConstr(bII[0] == DEMAND_II[0] - pII[0], "bII_0")
    for t in range(1, T):
        h.addConstr(bII[t] == bII[t-1] + DEMAND_II[t] - pII[t], f"bII_{t}")

    # --- Objective ---
    # Weekly costs:
    #   Skilled producing:   sp[t] * W_SKILL
    #   Skilled training:     st[t] * W_SKILL (trainers paid skilled wage)
    #   Trained:             tr[t] * W_TR
    #   Trainees (2 weeks):  nt[t] * W_TRAIN * 2  (= nt[t] * 240)
    #   Overtime:            ow[t] * W_OT
    #   Backlog penalty:     sum over t of bI[t]*PEN_I + bII[t]*PEN_II

    obj_expr = sum(
        sp[t]  * W_SKILL +
        st[t]  * W_SKILL +
        tr[t]  * W_TR   +
        nt[t]  * W_TRAIN * 2.0 +
        ow[t]  * W_OT   +
        bI[t]  * PEN_I  +
        bII[t] * PEN_II
        for t in range(T)
    )
    h.setObjective(obj_expr, highspy.ObjSense.kMinimize)

    # --- Solve ---
    solve_start = time.time()
    status = h.run()
    runtime = time.time() - solve_start
    print(f"Solver run() returned: {status}")

    model_status = h.getModelStatus()
    info = h.getInfo()

    mip_gap  = float(info.mip_gap) if info.mip_gap < float('inf') else 0.0
    obj_val   = h.getObjectiveValue()
    obj_bound = float(info.mip_dual_bound) if info.mip_dual_bound > -float('inf') else obj_val

    print(f"Model status: {model_status}")
    print(f"Objective: {obj_val:.4f}, Bound: {obj_bound:.4f}, Gap: {mip_gap:.6f}")

    status_map = {
        highspy.HighsModelStatus.kOptimal:     "optimal",
        highspy.HighsModelStatus.kInfeasible:  "infeasible",
        highspy.HighsModelStatus.kUnbounded:   "unbounded",
        highspy.HighsModelStatus.kTimeLimit:    "timeout",
    }
    status_str = status_map.get(model_status, str(model_status))

    sol = h.getSolution()

    vars_dict = {}
    for t in range(T):
        vars_dict[f"sp_{t}"]  = sol.col_value[sp[t].index]
        vars_dict[f"st_{t}"]  = sol.col_value[st[t].index]
        vars_dict[f"ow_{t}"]  = sol.col_value[ow[t].index]
        vars_dict[f"nt_{t}"]  = sol.col_value[nt[t].index]
        vars_dict[f"tr_{t}"]  = sol.col_value[tr[t].index]
        vars_dict[f"pI_{t}"]  = sol.col_value[pI[t].index]
        vars_dict[f"pII_{t}"] = sol.col_value[pII[t].index]
        vars_dict[f"bI_{t}"]  = sol.col_value[bI[t].index]
        vars_dict[f"bII_{t}"] = sol.col_value[bII[t].index]

    result = {
        "status":          status_str,
        "objective_value":  obj_val,
        "objective_bound": obj_bound,
        "mip_gap":         mip_gap,
        "runtime_seconds":  runtime,
        "variables":        vars_dict,
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Result written to result.json")
    return result

if __name__ == "__main__":
    solve()
