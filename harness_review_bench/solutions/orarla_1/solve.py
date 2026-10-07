#!/usr/bin/env python3
"""
solve.py for orarla_1: Resource allocation planning (MILP)
Method: PuLP + CBC solver (integer variables)

Mathematical model:
  Minimize: 50*X + 30*Y
  Subject to:
    X + Y <= 1000   (budget)
    X - Y >= 200    (X excess over Y)
    X <= 700        (X max)
    Y <= 500        (Y max)
    X, Y >= 0, integer
"""

import json
import os
import pulp

def main():
    # Create the MILP problem
    prob = pulp.LpProblem("orarla_1_resource_allocation", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, upBound=700, cat="Integer")
    Y = pulp.LpVariable("Y", lowBound=0, upBound=500, cat="Integer")

    # Objective: minimize 50*X + 30*Y
    prob += 50 * X + 30 * Y, "Total_Cost"

    # Constraints
    prob += X + Y <= 1000, "budget"
    prob += X - Y >= 200, "excess"
    prob += X <= 700, "x_max"
    prob += Y <= 500, "y_max"

    # Solve with CBC (default solver bundled with PuLP)
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    # Extract results
    result = {
        "status": pulp.LpStatus[status],
        "objective_value": pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None,
        "objective_bound": None,
        "mip_gap": None,
        "runtime_seconds": None,
        "variables": {
            "X": pulp.value(X) if status == pulp.LpStatusOptimal else None,
            "Y": pulp.value(Y) if status == pulp.LpStatusOptimal else None
        },
        "method_performed": {
            "name": "Integer Linear Programming with PuLP + CBC solver",
            "solver": "pulp",
            "strategy_id": "milp_pulp_highs",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }

    # Write result.json
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {result['status']}")
    print(f"X = {result['variables']['X']}")
    print(f"Y = {result['variables']['Y']}")
    print(f"Objective Value = {result['objective_value']}")

if __name__ == "__main__":
    main()
