"""
orarla_3: Foldable table production & workforce planning (6 months)
Method: milp_aggregate_production_v2
- Cumulative supply/demand formulation
- HiGHS high-level API (5-arg addVariable)
"""

import json
import highspy

# ── Parameters ──────────────────────────────────────────────────────────────
MONTHS = list(range(1, 7))
DEMAND = {1: 20000, 2: 40000, 3: 42000, 4: 35000, 5: 19000, 6: 18500}
INITIAL_WORKFORCE = 1000
INITIAL_INVENTORY = 15000

SELL_PRICE = 300
RAW_MATERIAL_COST = 90
OUTSOURCE_COST = 200
HOLDING_COST = 15
BACKORDER_COST = 35
HIRE_COST = 5000
FIRE_COST = 8000
REGULAR_WAGE = 30
OVERTIME_WAGE = 40
REGULAR_HOURS_PER_WORKER = 160
OVERTIME_LIMIT_PER_WORKER = 20
LABOR_HOURS_PER_UNIT = 5

total_demand = sum(DEMAND.values())

# ── Model setup ─────────────────────────────────────────────────────────────
m = highspy.Highs()
m.setOptionValue("log_to_console", True)
m.setOptionValue("mip_gap", 0.0)
m.setOptionValue("mip_abs_gap", 0.0)
m.changeObjectiveSense(highspy.ObjSense.kMaximize)
m.changeObjectiveOffset(SELL_PRICE * total_demand)

# ── Decision variables ──────────────────────────────────────────────────────
produce     = {t: m.addVariable(0.0, float('inf'), -RAW_MATERIAL_COST, highspy.HighsVarType.kInteger,   f"produce_{t}")     for t in MONTHS}
outsource   = {t: m.addVariable(0.0, float('inf'), -OUTSOURCE_COST,    highspy.HighsVarType.kInteger,   f"outsource_{t}")   for t in MONTHS}
hire        = {t: m.addVariable(0.0, float('inf'), -HIRE_COST,         highspy.HighsVarType.kInteger,   f"hire_{t}")        for t in MONTHS}
fire        = {t: m.addVariable(0.0, float('inf'), -FIRE_COST,         highspy.HighsVarType.kInteger,   f"fire_{t}")        for t in MONTHS}
workforce   = {t: m.addVariable(0.0, float('inf'), 0.0,                highspy.HighsVarType.kInteger,   f"workforce_{t}")   for t in MONTHS}
cum_supply  = {t: m.addVariable(0.0, float('inf'), 0.0,                highspy.HighsVarType.kInteger,   f"cum_supply_{t}")  for t in MONTHS}
cum_demand  = {t: m.addVariable(0.0, float('inf'), 0.0,                highspy.HighsVarType.kInteger,   f"cum_demand_{t}")  for t in MONTHS}
inv         = {t: m.addVariable(0.0, float('inf'), -HOLDING_COST,      highspy.HighsVarType.kContinuous, f"inv_{t}")        for t in MONTHS}
bo_         = {t: m.addVariable(0.0, float('inf'), -(SELL_PRICE + BACKORDER_COST), highspy.HighsVarType.kContinuous, f"bo_{t}") for t in MONTHS}
regular_hrs = {t: m.addVariable(0.0, float('inf'), -REGULAR_WAGE,      highspy.HighsVarType.kContinuous, f"rh_{t}")         for t in MONTHS}
overtime_hrs= {t: m.addVariable(0.0, float('inf'), -OVERTIME_WAGE,     highspy.HighsVarType.kContinuous, f"ot_{t}")         for t in MONTHS}

# ── Constraints ─────────────────────────────────────────────────────────────
m.addConstr(workforce[1] == INITIAL_WORKFORCE + hire[1] - fire[1], "C1_m1")
for t in MONTHS[1:]:
    m.addConstr(workforce[t] == workforce[t-1] + hire[t] - fire[t], f"C1_m{t}")

for t in MONTHS:
    m.addConstr(regular_hrs[t] == REGULAR_HOURS_PER_WORKER * workforce[t], f"C2_m{t}")

for t in MONTHS:
    m.addConstr(overtime_hrs[t] <= OVERTIME_LIMIT_PER_WORKER * workforce[t], f"C3_m{t}")

for t in MONTHS:
    m.addConstr(LABOR_HOURS_PER_UNIT * produce[t] <= regular_hrs[t] + overtime_hrs[t], f"C4_m{t}")

m.addConstr(cum_supply[1] == produce[1] + outsource[1], "C5_m1")
for t in MONTHS[1:]:
    m.addConstr(cum_supply[t] == cum_supply[t-1] + produce[t] + outsource[t], f"C5_m{t}")

m.addConstr(cum_demand[1] == DEMAND[1], "C6_m1")
for t in MONTHS[1:]:
    m.addConstr(cum_demand[t] == cum_demand[t-1] + DEMAND[t], f"C6_m{t}")

for t in MONTHS:
    m.addConstr(INITIAL_INVENTORY + cum_supply[t] - cum_demand[t] == inv[t] - bo_[t], f"C7_m{t}")

m.addConstr(inv[6] >= 10000, "C8_terminal_inv")
m.addConstr(bo_[6] == 0,      "C8_terminal_bo")

# ── Solve ───────────────────────────────────────────────────────────────────
m.setOptionValue("time_limit", 300)
m.run()

# Use named constants instead of magic numbers
OPT_STATUS = m.getModelStatus()
K_OPTIMAL = highspy.HighsModelStatus.kOptimal.value
K_INFEASIBLE = highspy.HighsModelStatus.kInfeasible.value
K_UNBOUNDED = highspy.HighsModelStatus.kUnbounded.value

opt_status = OPT_STATUS.value

if opt_status == K_OPTIMAL:
    status_str = "optimal"
    obj_val = m.getInfo().objective_function_value
    obj_bound = m.getInfo().mip_dual_bound
elif opt_status == K_INFEASIBLE:
    status_str = "infeasible"
    obj_val = None
    obj_bound = None
elif opt_status == K_UNBOUNDED:
    status_str = "unbounded"
    obj_val = None
    obj_bound = None
else:
    status_str = "feasible"  # any other feasible status
    obj_val = m.getInfo().objective_function_value
    obj_bound = m.getInfo().mip_dual_bound

print(f"Status value: {opt_status} -> {status_str}")
print(f"Objective: {obj_val}")

col_vals = m.getSolution().col_value

solution = {}
for t in MONTHS:
    solution[f"produce_{t}"]      = col_vals[produce[t].index]
    solution[f"outsource_{t}"]    = col_vals[outsource[t].index]
    solution[f"hire_{t}"]         = col_vals[hire[t].index]
    solution[f"fire_{t}"]         = col_vals[fire[t].index]
    solution[f"workforce_{t}"]    = col_vals[workforce[t].index]
    solution[f"rh_{t}"]           = col_vals[regular_hrs[t].index]
    solution[f"ot_{t}"]           = col_vals[overtime_hrs[t].index]
    solution[f"cum_supply_{t}"]   = col_vals[cum_supply[t].index]
    solution[f"cum_demand_{t}"]   = col_vals[cum_demand[t].index]
    solution[f"inv_{t}"]          = col_vals[inv[t].index]
    solution[f"bo_{t}"]           = col_vals[bo_[t].index]

print("\n=== SOLUTION ===")
for t in MONTHS:
    print(f"Month {t}: produce={solution[f'produce_{t}']:.1f}, "
          f"outsource={solution[f'outsource_{t}']:.1f}, "
          f"hire={solution[f'hire_{t}']:.1f}, fire={solution[f'fire_{t}']:.1f}, "
          f"workforce={solution[f'workforce_{t}']:.1f}, "
          f"inv={solution[f'inv_{t}']:.1f}, bo={solution[f'bo_{t}']:.1f}")

# ── Runtime (safe access) ───────────────────────────────────────────────────
info = m.getInfo()
runtime = 0.0
for attr in ['mip_runtime', 'simplex_iteration_time', 'ipm_iteration_time',
             'pdlp_iteration_count', 'qp_iteration_count']:
    val = getattr(info, attr, None)
    if val is not None and val >= 0:
        runtime = float(val)
        break

# ── Write result.json FIRST (before any other API calls) ────────────────────
result = {
    "status": status_str,
    "objective_value": obj_val,
    "objective_bound": obj_bound,
    "mip_gap": getattr(info, 'mip_gap', None),
    "runtime_seconds": runtime,
    "variables": solution,
    "method_performed": {
        "name": "milp_aggregate_production_v2",
        "steps": [
            "Cumulative supply/demand formulation",
            "Integer: produce, outsource, hire, fire, workforce, cum_supply, cum_demand",
            "Continuous: regular_hours, overtime_hours, inv, bo",
            "inv[t] - bo[t] = 15000 + cum_supply[t] - cum_demand[t]",
            "Terminal: inv[6]>=10000, bo[6]=0"
        ]
    }
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nresult.json written.")
