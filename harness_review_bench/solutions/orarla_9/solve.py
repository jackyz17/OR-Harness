#!/usr/bin/env python3
"""
solve.py — orarla_9
Method: Integer Linear Programming with PuLP + CBC solver

Minimize: 5*Ad1 + 3*Ad2
s.t.     Ad1 + Ad2 >= 50        (minimum effectiveness score)
         4*Ad1 + 6*Ad2 <= 300  (audience fatigue limit)
         Ad1, Ad2 >= 0, integer
"""

import json
import os
import pulp

def main():
    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    # Create the LP problem
    prob = pulp.LpProblem("orarla_9_advertising_budget", pulp.LpMinimize)

    # Decision variables: budget for Ad1 and Ad2 (integer, >= 0)
    Ad1 = pulp.LpVariable("Ad1", lowBound=0, cat="Integer")
    Ad2 = pulp.LpVariable("Ad2", lowBound=0, cat="Integer")

    # Objective: minimize total cost 5*Ad1 + 3*Ad2
    prob += 5 * Ad1 + 3 * Ad2, "Total_Cost"

    # Constraints
    # Minimum effectiveness score: Ad1 + Ad2 >= 50
    prob += Ad1 + Ad2 >= 50, "effectiveness_min"

    # Audience fatigue limit: 4*Ad1 + 6*Ad2 <= 300
    prob += 4 * Ad1 + 6 * Ad2 <= 300, "effort_limit"

    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    # Extract results
    status_str = pulp.LpStatus[status]
    if status == pulp.LpStatusOptimal:
        obj_val = pulp.value(prob.objective)
        ad1_val = pulp.value(Ad1)
        ad2_val = pulp.value(Ad2)
        mip_gap = None  # optimal, no gap
    else:
        obj_val = None
        ad1_val = None
        ad2_val = None
        mip_gap = None

    result = {
        "status": status_str,
        "objective_value": obj_val,
        "objective_bound": None,
        "mip_gap": mip_gap,
        "runtime_seconds": None,
        "variables": {
            "Ad1": ad1_val,
            "Ad2": ad2_val
        },
        "method_performed": {
            "action_id": action_id,
            "strategy_id": "milp_pulp_cbc",
            "solver": "pulp",
            "method": "Integer Linear Programming with PuLP + CBC solver",
            "steps": [
                "Define Ad1, Ad2 as integer LpVariables with lower bound 0",
                "Set objective: minimize 5*Ad1 + 3*Ad2",
                "Add constraint: Ad1 + Ad2 >= 50  (minimum effectiveness score)",
                "Add constraint: 4*Ad1 + 6*Ad2 <= 300  (audience fatigue limit)",
                "Solve with pulp.PULP_CBC_CMD(msg=0)",
                "Extract optimal solution (Ad1*, Ad2*) and total cost"
            ]
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {status_str}")
    print(f"Objective: {obj_val}")
    print(f"Ad1={ad1_val}, Ad2={ad2_val}")

if __name__ == "__main__":
    main()
