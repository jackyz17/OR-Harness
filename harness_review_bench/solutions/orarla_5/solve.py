#!/usr/bin/env python3
"""
ORClaw solve for orarla_5: Marketing Budget Allocation

Minimize total cost x + y subject to:
- 3x + 4y >= 12000 (effectiveness)
- 5x - 2y <= 10000 (balance)
- x + y <= 5000 (budget)
- x, y >= 0, integers (whole dollars)

Expected optimal solution: x=0, y=3000, objective=3000
"""

import json
import os
import pulp

def solve():
    # Create the LP problem
    prob = pulp.LpProblem("Marketing_Budget_Allocation", pulp.LpMinimize)
    
    # Decision variables
    x = pulp.LpVariable("x", lowBound=0, cat=pulp.LpInteger)
    y = pulp.LpVariable("y", lowBound=0, cat=pulp.LpInteger)
    
    # Objective: minimize total cost
    prob += x + y, "Total_Cost"
    
    # Constraints
    # Effectiveness: 3x + 4y >= 12000
    prob += 3*x + 4*y >= 12000, "Effectiveness"
    
    # Balance: 5x - 2y <= 10000
    prob += 5*x - 2*y <= 10000, "Balance"
    
    # Budget: x + y <= 5000
    prob += x + y <= 5000, "Budget"
    
    # Solve with CBC
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract results
    obj_val = pulp.value(prob.objective)
    x_val = pulp.value(x)
    y_val = pulp.value(y)
    
    # Get MIP gap if available
    mip_gap = None
    if hasattr(prob, 'MIP_GAP'):
        mip_gap = prob.MIP_GAP
    
    result = {
        "status": str(pulp.LpStatus[status]),
        "objective_value": obj_val,
        "objective_bound": None,  # PuLP doesn't expose this directly
        "mip_gap": mip_gap,
        "runtime_seconds": None,
        "variables": {
            "x": x_val,
            "y": y_val
        },
        "method_performed": {
            "name": "LP_ilp_pulp_cbc",
            "source": "PuLP CBC solver",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }
    
    # Write result.json
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Status: {result['status']}")
    print(f"Objective: {result['objective_value']}")
    print(f"x = {x_val}, y = {y_val}")
    
    return result

if __name__ == "__main__":
    solve()
