#!/usr/bin/env python3
"""
solve.py for orarla_4
Task: Minimize 4*X + 3*Y subject to X >= 2*Y + 300, X + Y <= 1000, X,Y integers >= 0.

Method: Analytical enumeration - exploit structure of 2-var bounded ILP.
  - Coupling constraint X >= 2Y + 300 is binding at optimum (cost coeff of X > Y).
  - Enumerate Y from 0 to floor((1000-300)/3)=233, compute minimum feasible X for each Y.
  - Pick (X,Y) with minimum objective 4X+3Y.
"""

import json
import os

def solve():
    # Parameters
    cX, cY = 4, 3  # cost per unit
    B = 1000       # budget cap
    D = 300        # demand offset

    best_obj = None
    best_X = None
    best_Y = None

    # Y can be at most floor((B - D) / 3) from budget + coupling:
    # X >= 2Y + D and X + Y <= B  =>  2Y + D + Y <= B  =>  Y <= (B - D) / 3
    Y_max = (B - D) // 3
    if Y_max < 0:
        Y_max = 0

    for Y in range(Y_max + 1):
        # Minimum X to satisfy coupling: X >= 2Y + 300
        X_min = 2 * Y + D
        # Check budget: need X_min + Y <= B
        if X_min + Y > B:
            continue
        X = X_min  # binding coupling, minimal X for this Y
        obj = cX * X + cY * Y
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_X = X
            best_Y = Y

    # Verify solution
    assert best_X is not None, "No feasible solution found"
    assert best_X + best_Y <= B, f"Budget violated: {best_X}+{best_Y}={best_X+best_Y} > {B}"
    assert best_X >= 2 * best_Y + D, f"Coupling violated: {best_X} < 2*{best_Y}+{D}"
    assert best_X >= 0 and best_Y >= 0, "Non-negativity violated"

    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    result = {
        "status": "optimal",
        "objective_value": float(best_obj),
        "objective_bound": float(best_obj),
        "mip_gap": 0.0,
        "runtime_seconds": 0.0,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "action_id": action_id,
            "strategy_id": "enumeration_2var_ilp_direct",
            "solver": "builtin",
            "method": "Analytical enumeration for bounded 2-variable ILP",
            "steps": [
                "Enumerate Y from 0 to floor((1000-300)/3)=233",
                "For each Y, set X = max(0, 2Y+300) to satisfy coupling binding",
                "Pick (X,Y) minimizing 4X+3Y",
                f"Solution: X={best_X}, Y={best_Y}, obj={best_obj}"
            ]
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, objective={best_obj}")
    return result

if __name__ == "__main__":
    solve()
