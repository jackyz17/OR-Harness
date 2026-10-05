#!/usr/bin/env python3
"""
ORClaw solve script for orarla_1: workforce training and production planning
Method: milp_binary_week_selection
Uses HiGHS high-level API (addVariable/addConstr)

Key modeling decisions:
- Backlog balance: backlog[t] >= backlog[t-1] + DEMAND[t] - prod[t]/RATE
  (backlog accumulates when demand > supply; >= allows backlog to grow without bound)
- All workers paid wages regardless of production (wages are fixed overhead)
- Trainee wages: 120/wk for 2-week training, then 240/wk after
- Overtime premium: 540/wk (extra 180 over base 360)
- Penalty: backlog * penalty_rate per week (accumulates with backlog)
"""

import json
import time
import highspy

# ── Data ──────────────────────────────────────────────────────────────────────
WEEKS = list(range(1, 9))
N = len(WEEKS)

INITIAL_SKILLED = 50
MAX_TRAIN_CAPACITY = 150
TARGET_TRAINEES = 50

WAGE_SKILLED         = 360.0
WAGE_OVERTIME        = 540.0
WAGE_TRAINEE_DURING  = 120.0
WAGE_TRAINEE_AFTER   = 240.0

HOURS_PER_WEEK  = 40.0
HOURS_OVERTIME  = 60.0
PROD_RATE_I     = 10.0
PROD_RATE_II    = 6.0

PENALTY_I  = 0.5
PENALTY_II = 0.6

DEMAND_I  = [10000.0, 10000.0, 12000.0, 12000.0, 16000.0, 16000.0, 20000.0, 20000.0]
DEMAND_II = [6000.0, 7200.0, 8400.0, 10800.0, 10800.0, 12000.0, 12000.0, 12000.0]

# ── Model ─────────────────────────────────────────────────────────────────────
m = highspy.Highs()
m.setOptionValue("output_flag", True)
m.setOptionValue("log_to_console", True)

def idx(t): return t - 1

# ── Variables ─────────────────────────────────────────────────────────────────
train          = []
ot             = []
prod_I         = []
prod_II        = []
backlog_I      = []
backlog_II     = []
trained_active = []
u_b            = []
v_b            = []

for t in WEEKS:
    i = idx(t)
    train.append(         m.addVariable(0.0, float(MAX_TRAIN_CAPACITY), 0.0, highspy.HighsVarType.kInteger,   f"train_{t}"))
    ot.append(            m.addVariable(0.0, float(INITIAL_SKILLED),    0.0, highspy.HighsVarType.kInteger,   f"ot_{t}"))
    prod_I.append(        m.addVariable(0.0, float('inf'),              0.0, highspy.HighsVarType.kContinuous, f"prod_I_{t}"))
    prod_II.append(       m.addVariable(0.0, float('inf'),              0.0, highspy.HighsVarType.kContinuous, f"prod_II_{t}"))
    backlog_I.append(     m.addVariable(0.0, float('inf'),              0.0, highspy.HighsVarType.kContinuous, f"backlog_I_{t}"))
    backlog_II.append(    m.addVariable(0.0, float('inf'),              0.0, highspy.HighsVarType.kContinuous, f"backlog_II_{t}"))
    trained_active.append(m.addVariable(0.0, float(TARGET_TRAINEES),   0.0, highspy.HighsVarType.kInteger,   f"trained_active_{t}"))
    u_b.append(           m.addVariable(0.0, 1.0,                      0.0, highspy.HighsVarType.kInteger,   f"u_{t}"))
    v_b.append(           m.addVariable(0.0, 1.0,                      0.0, highspy.HighsVarType.kInteger,   f"v_{t}"))

# ── Constraints ───────────────────────────────────────────────────────────────

# C1: Capacity
# Available hours = 40*(50 - train - ot) + 60*ot + 40*trained
# kg produced = prod_I/10 + prod_II/6
for t in WEEKS:
    i = idx(t)
    avail = (HOURS_PER_WEEK * (INITIAL_SKILLED - train[i] - ot[i])
             + HOURS_OVERTIME * ot[i]
             + HOURS_PER_WEEK * trained_active[i])
    m.addConstr(prod_I[i] / PROD_RATE_I + prod_II[i] / PROD_RATE_II <= avail, f"C1_{t}")

# C2: Cannot train more than 50 skilled workers per week
for t in WEEKS:
    i = idx(t)
    m.addConstr(train[i] <= INITIAL_SKILLED, f"C2_{t}")

# C3: Overtime only for non-training skilled workers
for t in WEEKS:
    i = idx(t)
    m.addConstr(ot[i] <= INITIAL_SKILLED - train[i], f"C3_{t}")

# C4: Training flow
# trained_active[t] = sum_{s=1}^{t-2} train[s]
for t in WEEKS:
    i = idx(t)
    if t >= 3:
        inflow = sum(train[idx(s)] for s in range(1, t - 1))
        m.addConstr(trained_active[i] == inflow, f"C4_{t}")
    else:
        m.addConstr(trained_active[i] == 0, f"C4_{t}")

# C5: Demand balance for food I
# backlog[t] >= backlog[t-1] + DEMAND_I[t] - prod_I[t]/10
# backlog[0] = 0
m.addConstr(backlog_I[0] == 0, "C5_init")
for t in WEEKS:
    i = idx(t)
    if i == 0:
        m.addConstr(backlog_I[0] >= DEMAND_I[0] - prod_I[0] / PROD_RATE_I, f"C5_I_{t}")
    else:
        m.addConstr(backlog_I[i] >= backlog_I[i - 1] + DEMAND_I[i] - prod_I[i] / PROD_RATE_I, f"C5_I_{t}")

# C6: Demand balance for food II
m.addConstr(backlog_II[0] == 0, "C6_init")
for t in WEEKS:
    i = idx(t)
    if i == 0:
        m.addConstr(backlog_II[0] >= DEMAND_II[0] - prod_II[0] / PROD_RATE_II, f"C6_II_{t}")
    else:
        m.addConstr(backlog_II[i] >= backlog_II[i - 1] + DEMAND_II[i] - prod_II[i] / PROD_RATE_II, f"C6_II_{t}")

# C7: Training target
m.addConstr(sum(train) == TARGET_TRAINEES, "C7_target")

# C8: Training capacity per 2-week window
for t in range(1, N):
    m.addConstr(train[idx(t)] + train[idx(t + 1)] <= MAX_TRAIN_CAPACITY, f"C8_{t}")

# C9/C10: Binary linking (optional - helps MIP solver)
for t in WEEKS:
    i = idx(t)
    m.addConstr(train[i] <= float(INITIAL_SKILLED) * u_b[i], f"C9_{t}")
    m.addConstr(ot[i] <= float(INITIAL_SKILLED) * v_b[i], f"C10_{t}")

# ── Objective ─────────────────────────────────────────────────────────────────
# Total cost = wages (fixed) + trainee wages + penalty
obj = 0.0

# Wage costs (fixed overhead, all workers paid regardless of production)
for t in WEEKS:
    i = idx(t)
    # Skilled workers not training or overtime
    obj += WAGE_SKILLED * (INITIAL_SKILLED - train[i] - ot[i])
    # Overtime premium
    obj += (WAGE_OVERTIME - WAGE_SKILLED) * ot[i]
    # Trained workers
    obj += WAGE_TRAINEE_AFTER * trained_active[i]

# Trainee during-training wages (120/wk for 2 weeks per trainee)
for t in WEEKS:
    i = idx(t)
    obj += WAGE_TRAINEE_DURING * train[i]         # new trainees this week
    if i > 0:
        obj += WAGE_TRAINEE_DURING * train[i - 1]  # still in 2nd training week

# Backlog penalty (accumulates per week)
for t in WEEKS:
    i = idx(t)
    obj += PENALTY_I * backlog_I[i]
    obj += PENALTY_II * backlog_II[i]

m.setObjective(obj, sense=highspy.ObjSense.kMinimize)

# ── Solve ─────────────────────────────────────────────────────────────────────
start_time = time.time()
m.run()
runtime = time.time() - start_time

# ── Extract solution ──────────────────────────────────────────────────────────
sol = m.getSolution()
cv = sol.col_value
ms = m.getModelStatus()

result = {
    "status": None,
    "objective_value": None,
    "objective_bound": None,
    "runtime_seconds": runtime,
    "variables": {}
}

if ms == highspy.HighsModelStatus.kOptimal or ms == highspy.HighsModelStatus.kFeasible:
    result["status"] = "optimal" if ms == highspy.HighsModelStatus.kOptimal else "feasible"
    result["objective_value"] = m.getObjectiveValue()
    info = m.getInfo()
    result["objective_bound"] = info.objective_function_value

    for t in WEEKS:
        i = idx(t)
        result["variables"][f"train_{t}"]          = round(cv[train[i].index])
        result["variables"][f"ot_{t}"]             = round(cv[ot[i].index])
        result["variables"][f"prod_I_{t}"]         = round(cv[prod_I[i].index], 2)
        result["variables"][f"prod_II_{t}"]        = round(cv[prod_II[i].index], 2)
        result["variables"][f"backlog_I_{t}"]      = round(cv[backlog_I[i].index], 2)
        result["variables"][f"backlog_II_{t}"]     = round(cv[backlog_II[i].index], 2)
        result["variables"][f"trained_active_{t}"] = round(cv[trained_active[i].index])
        result["variables"][f"u_{t}"]              = round(cv[u_b[i].index])
        result["variables"][f"v_{t}"]              = round(cv[v_b[i].index])
else:
    result["status"] = "infeasible" if ms == highspy.HighsModelStatus.kInfeasible else "unknown"

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResult: {result['status']}, objective: {result['objective_value']}, runtime: {runtime:.2f}s")
