#!/usr/bin/env python3
"""
orarla_10: Energy investment planning (2-variable MILP)
Method: enumeration_with_constraints — solve analytically + verify by enumeration
"""

import json
import os
import time
from typing import Dict, Any

def solve():
    start = time.time()

    # ===== Analytical solution =====
    # Minimize: 50*X + 60*Y
    # Subject to:
    #   X + Y <= 5000
    #   10*X + 30*Y >= 20000
    #   X >= 0, Y >= 0, integers

    # Output constraint requires: 10X + 30Y >= 20000  =>  X + 3Y >= 2000
    # With X + Y <= 5000, we have: Y >= 2000 - X, so combined: Y >= max(0, 2000-X)
    # Also from output: X >= 2000 - 3Y, combined with X >= 0

    # To minimize cost (50X + 60Y), prefer X (cheaper) over Y (60 > 50)
    # But must satisfy output: with X=5000, Y=0: output=10*5000=50000 >= 20000 ✓
    # Cost = 50*5000 + 60*0 = 250000

    # Check corner points analytically:
    candidates = []

    # Vertex analysis of the feasible LP region:
    # Intersection of X+Y=5000 and 10X+30Y=20000:
    # From X+Y=5000: X = 5000-Y
    # 10(5000-Y) + 30Y = 20000 => 50000 - 10Y + 30Y = 20000 => 20Y = -30000 => Y=-1500 (infeasible)
    # So the constraints don't bind together at a feasible point.

    # Vertices of feasible polygon:
    # 1. X=0: 30Y >= 20000 => Y >= 667 (ceil), X+Y <= 5000 => Y <= 5000 => Y=667
    #    Cost = 60*667 = 40020
    # 2. Y=0: 10X >= 20000 => X >= 2000, X <= 5000 => X=2000
    #    Cost = 50*2000 = 100000
    # 3. X+Y=5000 with 10X+30Y >= 20000:
    #    Substitute X=5000-Y: 10(5000-Y)+30Y >= 20000 => 50000+20Y >= 20000 => always true for Y>=0
    #    So corner is (X=5000, Y=0) but that violates X+Y<=5000 (it's on boundary)
    #    Actually (5000,0) gives X+Y=5000 ✓, output=50000 ✓
    #    Cost = 250000

    # The minimum is at (X=0, Y=667) with cost 40020... but wait, need to check integrality.
    # Actually X=0, Y=667 gives output=30*667=20010 >= 20000 ✓, cost=40020
    # X=2000, Y=0 gives cost=100000
    # X=5000, Y=0 gives cost=250000

    # But wait - we need to re-examine. The constraint is X+Y <= 5000.
    # If X=0, Y=667: cost=40020 ✓
    # If X=2000, Y=0: cost=100000 ✓
    # If X=5000, Y=0: cost=250000 ✓

    # But there's a problem: X=0, Y=667 gives output=20010 >= 20000 ✓
    # X=2000, Y=0 gives output=20000 >= 20000 ✓

    # Cheapest is X=0, Y=667 with cost 40020. But wait, is Y=667 integer? Yes.
    # Let me verify: X=0, Y=667: budget=667 <= 5000 ✓, output=20010 >= 20000 ✓

    # But let me re-think. The output constraint is: 10X + 30Y >= 20000
    # With X=0: 30Y >= 20000 => Y >= 666.67 => Y >= 667 (integer)
    # Cost = 60Y, so minimize Y. Y=667 gives cost=40020.

    # With Y=0: 10X >= 20000 => X >= 2000
    # Cost = 50X, so minimize X. X=2000 gives cost=100000.

    # With some X and Y:
    # Since 50X + 60Y, for each unit of Y we replace, we save 50 on X but add 60, net +10.
    # So Y is more expensive per unit of investment. Prefer X.
    # But X requires 1 unit of investment to get 10 output (ratio 0.1 cost/output)
    # Y requires 1 unit to get 30 output (ratio 0.05 cost/output) -- BETTER ratio!
    # So Y is actually MORE efficient for output, but since we want to minimize COST
    # and have a budget constraint (not output constraint), we want cheap investment.

    # Actually: Cost per unit of output:
    # X: 50/10 = 5 cost per output
    # Y: 60/30 = 2 cost per output
    # So Y is more cost-efficient for producing output!

    # But we have a BUDGET constraint, not an output constraint in the objective.
    # We want to minimize total cost, not cost per output.
    # So we want the cheapest investment: X at 50 per unit.

    # The output constraint is a MINIMUM requirement. We just need to meet it.
    # So: use X as much as possible (cheapest), then add Y only if needed.

    # Check: max X alone, Y=0: output=10*5000=50000 >= 20000 ✓, cost=250000
    # But we can do better with a mix.
    # If we use X=2000, Y=0: cost=100000, output=20000 ✓
    # If we use X=0, Y=667: cost=40020, output=20010 ✓
    # If we use X=1000, Y=333: cost=50*1000+60*333=50000+19980=69980, output=10*1000+30*333=10000+9990=19990 ✓

    # The cheapest is X=0, Y=667 with cost 40020.
    # But is there a cheaper solution with both X and Y?

    # Let's do a smarter search. Since we want to minimize 50X+60Y:
    # For a given total investment I = X+Y (budget used), output = 10X+30Y = 10(I-Y)+30Y = 10I+20Y
    # To satisfy output >= 20000: 10I + 20Y >= 20000 => I + 2Y >= 2000
    # Since I <= 5000, we need I >= 2000 - 2Y
    # And I = X+Y, so cost = 50X + 60Y = 50(I-Y) + 60Y = 50I + 10Y

    # For a fixed Y, minimize I (which is X+Y) subject to I >= max(Y, 2000-2Y)
    # Cost = 50*I + 10Y, so we want smallest I and smallest Y
    # Try Y from 0 to 667 (since 10*0 + 30*667 = 20010 >= 20000)
    #   For Y=0: I >= max(0, 2000) = 2000, cost = 50*2000 + 0 = 100000
    #   For Y=667: I >= max(667, 2000-1334=666) = 667, cost = 50*667 + 10*667 = 60*667 = 40020
    #   For Y=500: I >= max(500, 2000-1000=1000) = 1000, cost = 50*1000 + 10*500 = 50000+5000=55000
    #   For Y=600: I >= max(600, 2000-1200=800) = 800, cost = 50*800 + 10*600 = 40000+6000=46000

    # Minimum at Y=667: cost = 60*667 = 40020
    # But wait, is Y=667 the ONLY minimum? Let's check Y=666:
    #   I >= max(666, 2000-1332=668) = 668
    #   Cost = 50*668 + 10*666 = 33400 + 6660 = 40060
    #   Output check: X = I - Y = 668-666=2, output = 10*2 + 30*666 = 20 + 19980 = 20000 ✓
    #   So (X=2, Y=666) costs 100 + 39960 = 40060

    # So Y=667 gives the minimum: cost = 60*667 = 40020
    # Verified: X=0, Y=667: budget=667<=5000, output=20010>=20000, cost=40020 ✓

    # But wait - can we do even better? What about Y=668?
    #   I >= max(668, 2000-1336=664) = 668, cost = 50*668 + 10*668 = 60*668 = 40080 > 40020

    # So minimum cost = 40020 at X=0, Y=667.

    # But wait, I need to verify this is actually the minimum by enumeration to be safe.

    best_cost = float('inf')
    best_X = None
    best_Y = None

    # Enumerate Y from 0 to 667 (since 10*0+30*667=20010>=20000, and Y=666 gives X>=2)
    for Y in range(0, 668):
        # From output: 10X + 30Y >= 20000 => X >= max(0, ceil((20000-30Y)/10))
        min_X_from_output = max(0, (20000 - 30*Y + 9) // 10) if 20000 - 30*Y > 0 else 0
        # From budget: X + Y <= 5000 => X <= 5000 - Y
        max_X_from_budget = 5000 - Y
        if min_X_from_output > max_X_from_budget:
            continue  # infeasible
        # For fixed Y, cost = 50X + 60Y increases with X, so pick minimum X
        X = min_X_from_output
        cost = 50*X + 60*Y
        if cost < best_cost:
            best_cost = cost
            best_X = X
            best_Y = Y

    elapsed = time.time() - start

    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "mip_gap": 0.0,
        "runtime_seconds": elapsed,
        "variables": {"X": best_X, "Y": best_Y},
        "method_performed": {
            "name": "enumeration_with_constraints",
            "planned": "LP relaxation analysis + constrained enumeration over Y in [0,667]",
            "actual": "Analytical + enumeration: Y in [0,667], X = max(0, ceil((20000-30Y)/10)), pick minimum cost",
            "action_id": os.environ.get("OR_ACTION_ID"),
            "basis": "chosen from plan-next sp_15df0ba220ac"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: X={best_X}, Y={best_Y}, Cost={best_cost}")

if __name__ == "__main__":
    solve()
