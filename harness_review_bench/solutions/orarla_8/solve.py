#!/usr/bin/env python3
"""solve.py for orarla_8: enumeration method for bounded 2-var budget/effectiveness ILP."""
import json
import math
import time

def solve():
    start = time.time()
    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Iterate X from 0 to 5000 (bounded by budget)
    for X in range(0, 5001):
        # Minimum feasible Y to satisfy effectiveness: 2X + 3Y >= 10000
        # => Y >= ceil((10000 - 2X) / 3)
        if 2 * X >= 10000:
            min_Y = 0
        else:
            min_Y = math.ceil((10000 - 2 * X) / 3.0)

        # Budget constraint: Y <= 5000 - X
        max_Y = 5000 - X
        if min_Y > max_Y:
            continue  # infeasible for this X

        Y = min_Y  # smallest Y is cheapest (Y costs 20 per unit vs X costs 10)
        cost = 10 * X + 20 * Y
        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = Y

    elapsed = time.time() - start

    # Verify solution
    assert best_X is not None, "No feasible solution found"
    assert best_X + best_Y <= 5000, f"Budget violated: {best_X}+{best_Y}={best_X+best_Y}>5000"
    assert 2*best_X + 3*best_Y >= 10000, f"Effectiveness violated: 2*{best_X}+3*{best_Y}={2*best_X+3*best_Y}<10000"

    result = {
        "status": "optimal",
        "objective_value": float(best_cost),
        "objective_bound": float(best_cost),
        "mip_gap": 0.0,
        "runtime_seconds": elapsed,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "name": "Analytical enumeration for bounded 2-variable budget/effectiveness ILP. Iterate X from 0 to 5000; for each X compute minimum feasible integer Y = ceil(max(0, (10000-2X)/3)); check budget constraint Y <= 5000-X; track minimum cost. Uses cost-per-effectiveness insight (X=$5/pt < Y=$6.67/pt) but verifies via scan.",
            "strategy_id": "enumeration_2var_budget_effectiveness",
            "solver": "python_enum",
            "steps": [
                "Iterate X from 0 to 5000 (integer)",
                "For each X, compute minimum feasible Y = max(0, ceil((10000-2X)/3))",
                "Check budget feasibility: Y <= 5000 - X",
                "Track minimum 10X + 20Y across all feasible (X,Y)",
                "Return optimal (X,Y) and total cost"
            ],
            "action_id": "ac_ea18c3f90eca",
            "prediction_id": "sp_867bd705b597"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={best_cost}")
    return result

if __name__ == "__main__":
    solve()
