#!/usr/bin/env python3
"""
solve.py — orarla_7
Marketing campaign channel allocation (X, Y integers)
Minimize 5X + 3Y
Subject to: X + Y >= 1000, X - 2Y <= 500, X, Y >= 0 integers
"""
import json
import pulp
import os

def main():
    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    # Create the problem
    prob = pulp.LpProblem("orarla_7", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, cat="Integer")
    Y = pulp.LpVariable("Y", lowBound=0, cat="Integer")

    # Objective: minimize 5X + 3Y
    prob += 5 * X + 3 * Y, "TotalCost"

    # Constraints
    # Effectiveness: X + Y >= 1000
    prob += X + Y >= 1000, "Effectiveness"

    # Balance: X - 2Y <= 500
    prob += X - 2 * Y <= 500, "Balance"

    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    # Extract results
    obj_val = pulp.value(prob.objective)
    X_val = pulp.value(X)
    Y_val = pulp.value(Y)
    status_str = pulp.LpStatus[status]

    # MIP gap (CBC doesn't compute it directly, set to None for optimal)
    mip_gap = None
    if status == pulp.LpStatusOptimal:
        mip_gap = 0.0

    result = {
        "status": status_str,
        "objective_value": obj_val,
        "objective_bound": obj_val if status == pulp.LpStatusOptimal else None,
        "mip_gap": mip_gap,
        "runtime_seconds": None,  # CBC doesn't expose this cleanly
        "variables": {"X": X_val, "Y": Y_val},
        "method_performed": {
            "action_id": action_id,
            "name": "Integer Linear Programming with PuLP + CBC solver",
            "solver": "pulp_CBC",
            "formulation": "Min 5X+3Y s.t. X+Y>=1000, X-2Y<=500, X,Y>=0 integers"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {status_str}")
    print(f"X = {X_val}, Y = {Y_val}")
    print(f"Objective (Total Cost) = {obj_val}")

if __name__ == "__main__":
    main()
