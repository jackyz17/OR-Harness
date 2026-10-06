"""
solve.py for orarla_7
Method: MILP via PuLP/COIN-CBC (corrected)
- Variables: X, Y integers >= 0
- Objective: minimize 5*X + 3*Y
- Constraints:
    X + Y >= 1000        (effectiveness)
    |X - 2*Y| <= 500     (balance, two-sided)
  Reformulated as:
    X - 2*Y <= 500
    2*Y - X <= 500
"""

import json
import os
import pulp
import time

def solve():
    start = time.time()

    # Create the LP problem
    prob = pulp.LpProblem("marketing_campaign", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, cat="Integer")
    Y = pulp.LpVariable("Y", lowBound=0, cat="Integer")

    # Objective: minimize 5*X + 3*Y
    prob += 5 * X + 3 * Y, "total_cost"

    # Constraints
    prob += X + Y >= 1000, "effectiveness"
    # Balance: |X - 2Y| <= 500  =>  -500 <= X - 2Y <= 500
    prob += X - 2 * Y <= 500, "balance_upper"   # X - 2Y <= 500
    prob += 2 * Y - X <= 500, "balance_lower"   # 2Y - X <= 500  =>  X >= 2Y - 500

    # Solve with CBC (default)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=1))

    elapsed = time.time() - start

    # Extract results
    obj_val = None
    mip_gap = None

    if status == pulp.LpStatusOptimal:
        obj_val = pulp.value(prob.objective)
        mip_gap = 0.0

    X_val = int(X.value()) if hasattr(X.value(), '__int__') else int(X.value())
    Y_val = int(Y.value()) if hasattr(Y.value(), '__int__') else int(Y.value())

    # Collect result
    result = {
        "status": "optimal" if status == pulp.LpStatusOptimal else "unknown",
        "objective_value": obj_val,
        "objective_bound": obj_val,  # optimal => bound == value
        "mip_gap": mip_gap,
        "runtime_seconds": elapsed,
        "variables": {
            "X": X_val,
            "Y": Y_val
        },
        "method_performed": {
            "name": "MILP via PuLP/COIN-CBC",
            "approach": "ILP with integer X,Y >= 0; objective 5X+3Y; constraints: X+Y>=1000, |X-2Y|<=500 (reformulated as X-2Y<=500 AND 2Y-X<=500)",
            "solver": "COIN-CBC (default PULP_CBC_CMD)",
            "action_id": os.environ.get("OR_ACTION_ID", "none"),
            "note": "Balance constraint interpreted as two-sided |X-2Y|<=500 based on 'balanced strategy' phrasing"
        }
    }

    with open("/home/ubuntu/.openclaw/workspace/bench_orarla/ws/orarla_7/result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult: status={result['status']}, objective={result['objective_value']}, "
          f"X={X_val}, Y={Y_val}, runtime={elapsed:.3f}s")

    return result

if __name__ == "__main__":
    solve()
