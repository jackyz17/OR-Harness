"""
Farm animal allocation: maximize profit using milp_direct_integer.
Direct MILP solve with all integer variables.
"""
import json
import time
from highspy import Highs, ObjSense, HighsVarType, HighsModelStatus

start = time.time()

m = Highs()
m.changeObjectiveSense(ObjSense.kMaximize)

# Decision variables: n_cows, n_sheep, n_chickens (all integer)
n_cows = m.addVariable(lb=10.0, ub=float('inf'), obj=400.0,
                       type=HighsVarType.kInteger, name="n_cows")
n_sheep = m.addVariable(lb=20.0, ub=float('inf'), obj=120.0,
                        type=HighsVarType.kInteger, name="n_sheep")
n_chickens = m.addVariable(lb=0.0, ub=50.0, obj=3.0,
                          type=HighsVarType.kInteger, name="n_chickens")

# Constraints
m.addConstr(10*n_cows + 5*n_sheep + 3*n_chickens <= 800, "manure_limit")
m.addConstr(n_cows + n_sheep + n_chickens <= 100, "total_limit")

# Tight solver settings
m.setOptionValue('mip_rel_gap', 0.0)
m.setOptionValue('mip_abs_gap', 0.0)

status = m.run()
model_status = m.getModelStatus()
sol = m.getSolution()

c_val = sol.col_value[n_cows.index]
s_val = sol.col_value[n_sheep.index]
k_val = sol.col_value[n_chickens.index]

obj_val = 400*c_val + 120*s_val + 3*k_val
elapsed = time.time() - start

# Determine status
if model_status == HighsModelStatus.kOptimal:
    final_status = "optimal"
elif model_status == HighsModelStatus.kMipSolutionFeasible:
    final_status = "feasible"
else:
    final_status = "suboptimal"

result = {
    "status": final_status,
    "objective_value": obj_val,
    "objective_bound": obj_val,
    "runtime_seconds": elapsed,
    "solution": {
        "n_cows": int(c_val),
        "n_sheep": int(s_val),
        "n_chickens": int(k_val)
    }
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Status: {final_status}, Objective: {obj_val}")
print(f"Solution: cows={c_val}, sheep={s_val}, chickens={k_val}")
