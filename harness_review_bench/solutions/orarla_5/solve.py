#!/usr/bin/env python3
"""
solve.py — orarla_5
Method: Direct enumeration for 2-variable bounded ILP
Strategy: enumeration_budget_2var (chosen via plan-next)
"""

import math
import json
import os

def solve():
    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Enumerate X from 0 to 5000 (bounded by total budget constraint)
    for X in range(0, 5001):
        # From effectiveness: 3*X + 4*Y >= 12000 => Y >= ceil((12000 - 3*X) / 4)
        remaining = 12000 - 3 * X
        if remaining <= 0:
            Y_min = 0
        else:
            Y_min = math.ceil(remaining / 4)

        # Y must also satisfy budget: Y <= 5000 - X
        if Y_min > 5000 - X:
            continue  # no feasible Y for this X

        # Balance constraint: 5*X - 2*Y <= 10000 => Y >= (5*X - 10000) / 2
        # But we want the minimum Y (lowest cost), so only check upper bound
        if Y_min < 0:
            Y_min = 0

        # Check balance constraint at Y = Y_min: 5*X - 2*Y_min <= 10000
        if 5 * X - 2 * Y_min > 10000:
            # Need larger Y to satisfy balance; find minimum Y that works
            # 5*X - 2*Y <= 10000 => Y >= (5*X - 10000) / 2
            Y_balance_min = math.ceil((5 * X - 10000) / 2)
            Y_min = max(Y_min, Y_balance_min)
            # Recheck budget
            if Y_min > 5000 - X:
                continue

        Y = Y_min
        cost = 200 * X + 150 * Y

        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = Y

    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "mip_gap": 0.0,
        "runtime_seconds": None,
        "variables": {
            "budget_X": best_X,
            "budget_Y": best_Y
        },
        "method_performed": {
            "action_id": action_id,
            "strategy_id": "enumeration_budget_2var",
            "solver": "custom",
            "note": "Direct enumeration: iterate X 0..5000, compute min feasible Y from effectiveness constraint, verify balance and budget, pick min cost"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={best_cost}")
    return result

if __name__ == "__main__":
    solve()
