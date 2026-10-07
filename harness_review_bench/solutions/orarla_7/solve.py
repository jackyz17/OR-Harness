#!/usr/bin/env python3
"""
solve.py for orarla_7: marketing campaign planning
Method: MILP with Highs solver (highspy)
Selected by: orx choose-next --prediction sp_4ec0dde4b9c7
"""
import os
import json
import highspy

# Record method_performed for traceability
method_performed = {
    "strategy_id": "milp_highs_integer",
    "solver": "highspy",
    "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
    "task_id": "orarla_7",
    "episode": "ep1",
    "planned_steps": [
        "Create Highs model",
        "Add integer variable X >= 0, Y >= 0",
        "Add constraint: X + Y >= 1000",
        "Add constraint: X - 2*Y <= 500",
        "Set objective: minimize 5*X + 3*Y",
        "Set mip_rel_gap=0, mip_abs_gap=0",
        "Solve and extract solution"
    ]
}

# Create Highs model
h = highspy.Highs()
h.setOptionValue("log_to_console", False)

# Add integer variables X, Y >= 0
# addVariable(lb, ub, obj, type, name)
x_var = h.addVariable(0.0, highspy.kHighsInf, 5.0, highspy.HighsVarType.kInteger, "X")
y_var = h.addVariable(0.0, highspy.kHighsInf, 3.0, highspy.HighsVarType.kInteger, "Y")

# Add constraints
# C1: X + Y >= 1000  =>  -X - Y <= -1000
h.addConstr(-x_var - y_var <= -1000.0, "effectiveness")

# C2: X - 2*Y <= 500
h.addConstr(x_var - 2*y_var <= 500.0, "balance")

# Set sense to minimize (default is minimize, but be explicit)
h.changeObjectiveSense(highspy.ObjSense.kMinimize)

# Set MIP gap to 0 for exact solution
h.setOptionValue("mip_rel_gap", 0.0)
h.setOptionValue("mip_abs_gap", 0.0)

# Solve
status = h.run()
model_status = h.getModelStatus()

# Map Highs model status to canonical status string
def canonical_status(ms):
    s = ms.name.lower() if ms else "unknown"
    # kOptimal -> optimal, kInfeasible -> infeasible, etc.
    return s.replace("k", "")

# Extract solution
sol = h.getSolution()
col_val = sol.col_value

x_val = int(round(col_val[x_var.index]))
y_val = int(round(col_val[y_var.index]))
objective_value = h.getInfo().objective_function_value

# Write result
result = {
    "status": canonical_status(model_status),
    "objective_value": objective_value,
    "objective_bound": h.getInfo().mip_global_bound if hasattr(h.getInfo(), "mip_global_bound") else None,
    "mip_gap": h.getInfo().mip_gap if hasattr(h.getInfo(), "mip_gap") else None,
    "runtime_seconds": h.getInfo().mip_runtime if hasattr(h.getInfo(), "mip_runtime") else None,
    "variables": {"X": x_val, "Y": y_val},
    "method_performed": method_performed,
    "solver": "highspy"
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"X={x_val}, Y={y_val}, objective={objective_value}, status={model_status.name}")
