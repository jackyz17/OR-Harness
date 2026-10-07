#!/usr/bin/env python3
"""
solve.py for orarla_2
Method: Direct MILP with PuLP
- Variables: budget_X, budget_Y (integer, >= 0)
- Constraint 1: budget_X + budget_Y <= 2000
- Constraint 2 (reach): 3*budget_X - 2*budget_Y >= 500
- Objective: minimize budget_X + budget_Y
  (actual cost = (budget_X/50)*50 + (budget_Y/100)*100 = budget_X + budget_Y)
"""

import json
import os
import pulp

def main():
    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    # Create the MILP problem
    prob = pulp.LpProblem("orarla_2_marketing_budget", pulp.LpMinimize)

    # Decision variables: budget allocations (integers, non-negative)
    budget_X = pulp.LpVariable("budget_X", lowBound=0, cat="Integer")
    budget_Y = pulp.LpVariable("budget_Y", lowBound=0, cat="Integer")

    # Objective: minimize total cost (budget_X + budget_Y)
    prob += budget_X + budget_Y, "Total_Cost"

    # Constraint 1: total budget <= 2000
    prob += budget_X + budget_Y <= 2000, "total_budget_limit"

    # Constraint 2: reach constraint
    # 3*budget_X - 2*budget_Y >= 500
    prob += 3*budget_X - 2*budget_Y >= 500, "reach_constraint"

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results
    if status == pulp.LpStatusOptimal:
        status_str = "optimal"
        obj_val = pulp.value(prob.objective)
        mip_gap = 0.0
    elif status == pulp.LpStatusInfeasible:
        status_str = "infeasible"
        obj_val = None
        mip_gap = None
    elif status == pulp.LpStatusUndefined:
        status_str = "undefined"
        obj_val = None
        mip_gap = None
    else:
        status_str = str(status)
        obj_val = None
        mip_gap = None

    # Collect variable values
    variables = {}
    if status == pulp.LpStatusOptimal:
        variables = {
            "budget_X": pulp.value(budget_X),
            "budget_Y": pulp.value(budget_Y)
        }

    # Write result.json
    result = {
        "status": status_str,
        "objective_value": obj_val,
        "objective_bound": None,
        "mip_gap": mip_gap,
        "runtime_seconds": None,
        "variables": variables,
        "method_performed": {
            "action_id": action_id,
            "solver": "pulp",
            "approach": "direct_milp_2var",
            "note": "MILP with integer budget_X, budget_Y; constraint 3*budget_X - 2*budget_Y >= 500 from reach requirement"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
