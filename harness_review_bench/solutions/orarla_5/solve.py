#!/usr/bin/env python3
"""
ORClaw orarla_5: Marketing budget allocation (planning, integer ILP)

Formulation:
  Minimize: 200*X + 150*Y
  Subject to:
    X + Y <= 5000          (total budget)
    3*X + 4*Y >= 12000     (minimum effectiveness)
    5*X - 2*Y <= 10000     (balance constraint)
    X, Y >= 0, integer

Solver: PuLP + CBC
"""

import json
import os
import pulp

def solve():
    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, cat=pulp.LpInteger)
    Y = pulp.LpVariable("Y", lowBound=0, cat=pulp.LpInteger)

    # Problem
    prob = pulp.LpProblem("marketing_budget_allocation", pulp.LpMinimize)

    # Objective: minimize total cost (200 per unit effectiveness for X, 150 for Y)
    prob += 200 * X + 150 * Y, "total_cost"

    # Constraints
    prob += X + Y <= 5000, "total_budget"
    prob += 3 * X + 4 * Y >= 12000, "effectiveness"
    prob += 5 * X - 2 * Y <= 10000, "balance"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    # Extract results
    status_str = pulp.LpStatus[status]
    X_val = pulp.value(X)
    Y_val = pulp.value(Y)
    obj_val = pulp.value(prob.objective)

    result = {
        "status": status_str,
        "objective_value": obj_val,
        "objective_bound": pulp.value(prob.objective),
        "variables": {"X": X_val, "Y": Y_val},
        "method_performed": {
            "action_id": action_id,
            "solver": "PuLP_CBC",
            "formulation": "Min 200*X + 150*Y s.t. X+Y<=5000, 3X+4Y>=12000, 5X-2Y<=10000, X,Y integer>=0"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {status_str}")
    print(f"X = {X_val}, Y = {Y_val}")
    print(f"Objective (total cost) = {obj_val}")

if __name__ == "__main__":
    solve()
