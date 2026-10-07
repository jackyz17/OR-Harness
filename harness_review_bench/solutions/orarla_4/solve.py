#!/usr/bin/env python3
"""
solve.py — orarla_4
ILP: Min 4X + 3Y s.t. X + Y <= 1000, X - 2*Y >= 300, X,Y >= 0 integers.
Method: PuLP + CBC (pulp.PULP_CBC_CMD)
"""
import json
import time
import pulp
import os

start = time.time()

# Create the problem
prob = pulp.LpProblem("orarla_4", pulp.LpMinimize)

# Decision variables
X = pulp.LpVariable("X", lowBound=0, cat=pulp.LpInteger)
Y = pulp.LpVariable("Y", lowBound=0, cat=pulp.LpInteger)

# Objective: minimize 4*X + 3*Y
prob += 4 * X + 3 * Y, "Total_Cost"

# Constraints
prob += X + Y <= 1000, "total_resources"
prob += X - 2 * Y >= 300, "x_minimum"

# Solve with CBC
solver = pulp.PULP_CBC_CMD(msg=0)
result = prob.solve(solver)

elapsed = time.time() - start

status = pulp.LpStatus[result]
objective_value = pulp.value(prob.objective) if status in ("Optimal", "Feasible") else None

# Gap
mip_gap = None
if status == "Optimal":
    mip_gap = 0.0
elif hasattr(prob, 'MIP'):
    try:
        mip_gap = abs(pulp.value(prob.objective) - (pulp.value(prob.objective) or 0))
    except Exception:
        mip_gap = None

solution_vars = None
if status in ("Optimal", "Feasible"):
    solution_vars = {
        "X": pulp.value(X),
        "Y": pulp.value(Y)
    }

result_data = {
    "status": status,
    "objective_value": objective_value,
    "objective_bound": None,
    "mip_gap": mip_gap,
    "runtime_seconds": round(elapsed, 6),
    "variables": solution_vars,
    "method_performed": {
        "name": "Integer Linear Programming with PuLP + CBC solver",
        "solver": "pulp.PULP_CBC_CMD",
        "action_id": os.environ.get("OR_ACTION_ID", None),
        "formulation": "Min 4X + 3Y s.t. X + Y <= 1000, X - 2*Y >= 300, X,Y >= 0 integer"
    }
}

with open("result.json", "w") as f:
    json.dump(result_data, f, indent=2)

print(f"Status: {status}")
print(f"Objective: {objective_value}")
print(f"X={pulp.value(X)}, Y={pulp.value(Y)}")
print(f"Runtime: {elapsed:.4f}s")
