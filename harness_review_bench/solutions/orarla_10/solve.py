#!/usr/bin/env python3
"""solve.py for orarla_10: 2-var budget/effectiveness ILP via enumeration"""
import json
import math
import os

# Parameters
BUDGET = 5000
COST_X = 50
COST_Y = 60
EFF_X = 10  # output per unit X
EFF_Y = 30  # output per unit Y
MIN_OUTPUT = 20000

def solve():
    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Iterate X from 0 to BUDGET (bounded integer domain)
    for X in range(BUDGET + 1):
        # Minimum Y to meet output requirement: 10*X + 30*Y >= 20000
        # => Y >= ceil((20000 - 10*X) / 30)
        if 10 * X >= MIN_OUTPUT:
            min_Y = 0
        else:
            min_Y = math.ceil((MIN_OUTPUT - 10 * X) / EFF_Y)

        # Budget feasibility: X + Y <= BUDGET
        if min_Y > BUDGET - X:
            continue  # infeasible

        cost = COST_X * X + COST_Y * min_Y
        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = min_Y

    # Also check Y-dominant corner: maximize Y (cheaper per output)
    # Y can go up to min(BUDGET, floor(20000/30)) = 666
    for Y in range(BUDGET + 1):
        if 30 * Y >= MIN_OUTPUT:
            min_X = 0
        else:
            min_X = math.ceil((MIN_OUTPUT - 30 * Y) / EFF_X)

        if min_X > BUDGET - Y:
            continue

        cost = COST_X * min_X + COST_Y * Y
        if cost < best_cost:
            best_cost = cost
            best_X = min_X
            best_Y = Y

    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "mip_gap": 0.0,
        "runtime_seconds": 0.0,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "strategy_id": "enumeration_2var_budget_effectiveness",
            "solver": "python_enum",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
            "steps": [
                "Iterate X from 0 to 5000",
                "Compute min feasible Y = ceil(max(0, (20000-10*X)/30))",
                "Check budget Y <= 5000-X",
                "Track minimum 50*X + 60*Y",
                "Also iterate Y-dominant scan",
                "Return optimal (X,Y) and total cost"
            ]
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={best_cost}")
    return result

if __name__ == "__main__":
    solve()
