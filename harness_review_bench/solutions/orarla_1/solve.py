#!/usr/bin/env python3
"""
ORClaw solve for orarla_1: resource allocation ILP with PuLP+CBC
Minimize 50*X + 30*Y subject to:
  X + Y <= 1000
  X - Y >= 200
  X <= 700
  Y <= 500
  X, Y integer >= 0
"""
import json
import time
import os
import pulp

def solve():
    start_time = time.time()

    # Create the LP problem
    prob = pulp.LpProblem("orarla_1_resource_allocation", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, upBound=700, cat='Integer')
    Y = pulp.LpVariable("Y", lowBound=0, upBound=500, cat='Integer')

    # Objective: minimize 50*X + 30*Y
    prob += 50*X + 30*Y, "Total_Cost"

    # Constraints
    prob += X + Y <= 1000, "total_budget"
    prob += X - Y >= 200, "minimum_excess"
    prob += X <= 700, "x_max"
    prob += Y <= 500, "y_max"

    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    elapsed = time.time() - start_time

    # Extract results
    obj_value = pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None
    x_val = pulp.value(X) if status == pulp.LpStatusOptimal else None
    y_val = pulp.value(Y) if status == pulp.LpStatusOptimal else None

    # Get bound and gap if available
    mip_gap = None
    if hasattr(prob, 'MIP'):
        try:
            mip_gap = prob.MIP.get('gap') if isinstance(prob.MIP, dict) else None
        except:
            pass

    result = {
        "status": "optimal" if status == pulp.LpStatusOptimal else "infeasible" if status == pulp.LpStatusInfeasible else "other",
        "objective_value": round(obj_value, 2) if obj_value is not None else None,
        "objective_bound": None,  # CBC doesn't easily expose this
        "mip_gap": mip_gap,
        "runtime_seconds": round(elapsed, 3),
        "variables": {
            "X": int(x_val) if x_val is not None else None,
            "Y": int(y_val) if y_val is not None else None
        },
        "method_performed": {
            "solver": "pulp",
            "solver_config": {"solver": "PULP_CBC_CMD"},
            "formulation": "standard_linear",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }

    # Write result.json (relative path for security policy)
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    solve()
