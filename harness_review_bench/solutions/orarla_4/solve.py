#!/usr/bin/env python3
"""Solve orarla_4: minimum cost resource allocation between campaigns X and Y.

Formulation (MILP):
  min 4*X + 3*Y
  s.t. X + Y <= 1000
       X >= 2*Y + 300
       X >= 0, Y >= 0, integers

Approach: direct enumeration since Y <= 233 (derived bound) and X is then determined.
"""

import json
import time
import os

def solve():
    start = time.time()

    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Y <= (1000 - 300) / 3 = 233.33 → Y <= 233
    max_Y = (1000 - 300) // 3

    for Y in range(max_Y + 1):
        # Feasibility: X >= 2*Y + 300 AND X + Y <= 1000
        X_min = 2 * Y + 300
        X_max = 1000 - Y
        if X_min > X_max:
            continue  # infeasible for this Y
        # For minimum cost with cost coefficients 4 (X) > 3 (Y),
        # we want smallest X that is feasible
        X = X_min  # smallest X satisfies both constraints
        cost = 4 * X + 3 * Y
        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = Y

    elapsed = time.time() - start

    objective_rounded = round(best_cost)

    result = {
        "status": "optimal",
        "objective_value": objective_rounded,
        "objective_exact": best_cost,
        "variables": {"X": int(best_X), "Y": int(best_Y)},
        "mip_gap": 0.0,
        "runtime_seconds": round(elapsed, 6),
        "method_performed": {
            "name": "enumeration_with_constraints",
            "solver": "brute_force",
            "steps": [
                "From X >= 2Y + 300 and X + Y <= 1000, derive Y <= 233",
                "Enumerate Y from 0 to 233, compute minimal feasible X = max(2Y+300, 0)",
                "Verify X + Y <= 1000; pick minimum cost = 4X + 3Y",
                "Write result to result.json with status/objective_value/variables"
            ],
            "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
            "deviation": "Switched from scipy.optimize.milp (res.x=None, solver failed in sandbox) to direct enumeration; problem is small enough (Y<=233) for O(n) scan"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={objective_rounded}")

if __name__ == "__main__":
    solve()
