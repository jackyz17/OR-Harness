#!/usr/bin/env python3
"""
solve.py for orarla_9
Method: Analytical enumeration for bounded 2-var ILP

Problem: Minimize 5x + 3y subject to:
  x + y >= 50  (effectiveness)
  4x + 6y <= 300  (audience fatigue) => 2x + 3y <= 150
  x, y >= 0 integers
"""

import math
import json
import os

def solve():
    best_cost = float('inf')
    best_x = None
    best_y = None

    # x in [0, 75] from 4x <= 300
    for x in range(0, 76):
        # y >= ceil(50 - x) from effectiveness constraint
        y_min = max(0, math.ceil(50 - x))
        # y upper bound from fatigue: 2x + 3y <= 150 => y <= (150 - 2x) / 3
        y_max = (150 - 2 * x) // 3

        for y in range(y_min, y_max + 1):
            # Verify fatigue constraint: 2x + 3y <= 150
            if 2 * x + 3 * y <= 150:
                cost = 5 * x + 3 * y
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_y = y

    # Also check corner cases
    # x=0: y >= 50, y <= 50 -> y=50, cost=150
    # x=50: y >= 0, y <= 16 -> no y >= 0 satisfies 2*50+3y<=150 (100+3y<=150 => y<=16)
    #        but x+y>=50 => 50+y>=50 => y>=0, so (50,0) is infeasible due to fatigue

    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "mip_gap": 0.0,
        "runtime_seconds": 0.0,
        "variables": {"x": best_x, "y": best_y},
        "method_performed": {
            "strategy_id": os.environ.get("OR_ACTION_ID", "enumeration_2var_analytic"),
            "solver": "python_enum",
            "approach": "Analytical enumeration for bounded 2-var ILP"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal solution: x={best_x}, y={best_y}, cost={best_cost}")
    return result

if __name__ == "__main__":
    solve()
