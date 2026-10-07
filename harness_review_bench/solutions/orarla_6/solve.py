#!/usr/bin/env python3
"""
orarla_6: Marketing budget allocation (enumeration method)
Minimize 10*X + 20*Y subject to:
  X + Y <= 1000  (budget)
  2*X + 3*Y >= 2000  (effectiveness)
  X, Y >= 0 integers
"""
import math
import json
import time

def solve():
    start = time.time()
    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Effective lower bound on X: need 2*X >= 2000 - 3*Y >= 2000 - 3*1000 = -1000, so X >= 0
    # But to satisfy effectiveness: 2*X + 3*Y >= 2000 with Y <= 1000 - X
    # Max effectiveness with all budget on Y: 3*1000 = 3000 >= 2000 ✓
    # Min X to satisfy effectiveness with Y=1000-X: 2X + 3(1000-X) >= 2000 → X >= 334 (integer)
    # So X ranges from 334 to 1000

    for X in range(0, 1001):
        # Minimum Y to satisfy effectiveness: ceil(max(0, 2000 - 2*X) / 3)
        if 2*X >= 2000:
            min_Y = 0
        else:
            min_Y = math.ceil((2000 - 2*X) / 3)
        
        # Feasibility: Y must satisfy budget constraint
        if min_Y > 1000 - X:
            continue  # Not feasible
        
        # Y can be larger than minimum (we want minimum cost, so use minimum)
        Y = min_Y
        cost = 10*X + 20*Y
        
        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = Y

    elapsed = time.time() - start

    result = {
        "status": "optimal",
        "objective_value": float(best_cost),
        "objective_bound": float(best_cost),
        "mip_gap": 0.0,
        "runtime_seconds": elapsed,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "strategy_id": "enumeration_2var_budget_effectiveness",
            "solver": "python enumeration",
            "action_id": None,  # will be filled by framework
            "note": "Enumerated X from 0 to 1000, computed min feasible Y=ceil(max(0,2000-2X)/3), checked budget feasibility, minimized 10X+20Y"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={best_cost}")

if __name__ == "__main__":
    solve()
