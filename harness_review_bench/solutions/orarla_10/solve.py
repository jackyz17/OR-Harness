#!/usr/bin/env python3
"""
orarla_10: Investment planning - minimize cost with budget and output constraints.
Method: ILP via PuLP + CBC solver.
"""
import json
import pulp
import os

def solve():
    # Create the LP problem
    prob = pulp.LpProblem("orarla_10", pulp.LpMinimize)
    
    # Decision variables: X, Y as non-negative integers
    X = pulp.LpVariable("X", lowBound=0, upBound=5000, cat='Integer')
    Y = pulp.LpVariable("Y", lowBound=0, upBound=5000, cat='Integer')
    
    # Objective: minimize 50*X + 60*Y
    prob += 50*X + 60*Y, "Total_Cost"
    
    # Constraints
    # Budget: X + Y <= 5000
    prob += X + Y <= 5000, "budget_limit"
    # Output: 10*X + 30*Y >= 20000
    prob += 10*X + 30*Y >= 20000, "output_req"
    
    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)
    
    # Collect results
    result = {
        "status": pulp.LpStatus[status],
        "objective_value": pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None,
        "objective_bound": pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None,
        "mip_gap": None,
        "runtime_seconds": None,
        "variables": {
            "X": pulp.value(X) if status == pulp.LpStatusOptimal else None,
            "Y": pulp.value(Y) if status == pulp.LpStatusOptimal else None
        },
        "method_performed": {
            "name": "Integer Linear Programming with PuLP + CBC solver",
            "solver": "pulp.PULP_CBC_CMD",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }
    
    # Write result.json
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Status: {result['status']}")
    print(f"X = {result['variables']['X']}")
    print(f"Y = {result['variables']['Y']}")
    print(f"Objective (Total Cost) = {result['objective_value']}")

if __name__ == "__main__":
    solve()
