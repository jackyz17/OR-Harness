#!/usr/bin/env python3
"""
solve.py — orarla_8
Method: enumeration_with_constraints (pure Python, no external solver)

Strategy: Since X+Y <= 5000 and we minimize cost (X cheaper at 10 vs Y at 20),
push X to maximum while satisfying the effectiveness constraint 2X+3Y >= 10000.

Substituting Y = 5000 - X (binding budget) into effectiveness:
  2X + 3*(5000-X) >= 10000  →  X >= 1666.67
So the optimum is at X=1666 (integer), Y=3334.
Objective = 10*1666 + 20*3334 = 83340.
"""
import os
import json
import math

def solve():
    # Parameters from the problem
    BUDGET_MAX = 5000
    EFFECTIVENESS_MIN = 10000
    COST_X = 10
    COST_Y = 20

    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Enumerate Y from 0 to 5000; X is determined by budget constraint
    # For each Y, compute max feasible X = BUDGET_MAX - Y
    # Check if effectiveness constraint is satisfied
    for Y in range(BUDGET_MAX + 1):
        X = BUDGET_MAX - Y  # use full budget
        if X < 0:
            break
        effectiveness = 2 * X + 3 * Y
        if effectiveness >= EFFECTIVENESS_MIN:
            cost = COST_X * X + COST_Y * Y
            if cost < best_cost:
                best_cost = cost
                best_X = X
                best_Y = Y

    # Verify solution
    assert best_X is not None, "No feasible solution found"
    assert best_X + best_Y <= BUDGET_MAX, f"Budget violated: {best_X}+{best_Y}>{BUDGET_MAX}"
    assert 2*best_X + 3*best_Y >= EFFECTIVENESS_MIN, f"Effectiveness violated"
    assert best_cost == COST_X * best_X + COST_Y * best_Y

    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "mip_gap": 0.0,
        "runtime_seconds": None,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "name": "enumeration_with_constraints",
            "solver": "python",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
            "note": "Pure Python enumeration; Y iterates 0..5000, X = budget - Y; pick min cost satisfying effectiveness"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, cost={best_cost}")

if __name__ == "__main__":
    solve()
