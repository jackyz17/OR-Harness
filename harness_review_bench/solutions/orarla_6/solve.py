#!/usr/bin/env python3
"""
solve.py — orarla_6
ILP: min 10*X + 20*Y s.t. X + Y <= 1000, 2*X + 3*Y >= 2000, X,Y >= 0 integers
Solver: PuLP + CBC
"""
import json
import pulp
import os

action_id = os.environ.get("OR_ACTION_ID", "unknown")

# --- ILP formulation ---
prob = pulp.LpProblem("orarla_6", pulp.LpMinimize)

X = pulp.LpVariable("X", lowBound=0, cat="Integer")
Y = pulp.LpVariable("Y", lowBound=0, cat="Integer")

# Objective: minimize total cost
prob += 10 * X + 20 * Y, "TotalCost"

# Budget constraint
prob += X + Y <= 1000, "BudgetLimit"

# Effectiveness requirement
prob += 2 * X + 3 * Y >= 2000, "EffectivenessReq"

# Solve
solver = pulp.PULP_CBC_CMD(msg=0)
status = prob.solve(solver)

# --- Extract results ---
status_str = pulp.LpStatus[status]
obj_val = pulp.value(prob.objective)

# Get MIP gap and bound from solver model
mip_gap = None
objective_bound = None
runtime = None
try:
    solver_model = prob.solverModel
    mip_gap = getattr(solver_model, "mipGap", None) or getattr(solver_model, "MIPGap", None)
    # CBC stores best bound differently
    if hasattr(solver_model, "objBound"):
        objective_bound = solver_model.objBound
    elif hasattr(solver_model, "bestBound"):
        objective_bound = solver_model.bestBound
    runtime = getattr(solver_model, "time", None) or getattr(prob, "solveTime", None)
except Exception:
    pass

X_star = pulp.value(X) if status == pulp.LpStatusOptimal else None
Y_star = pulp.value(Y) if status == pulp.LpStatusOptimal else None

result = {
    "status": status_str,
    "objective_value": round(obj_val) if obj_val is not None else None,
    "objective_bound": round(objective_bound) if objective_bound is not None else None,
    "mip_gap": mip_gap,
    "runtime_seconds": runtime,
    "variables": {"X": X_star, "Y": Y_star},
    "method_performed": {
        "action_id": action_id,
        "solver": "pulp.PULP_CBC_CMD",
        "model": "ILP: min 10*X + 20*Y s.t. X + Y <= 1000, 2*X + 3*Y >= 2000, X,Y integer",
        "notes": "X=dollars invested in channel X; Y=dollars invested in channel Y"
    }
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Status: {status_str}")
print(f"X={X_star}, Y={Y_star}")
print(f"Objective: {obj_val}")
