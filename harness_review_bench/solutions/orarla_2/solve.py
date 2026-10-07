#!/usr/bin/env python3
"""
solve.py for orarla_2
Method: ILP via PuLP + CBC solver
Selected by: orx choose-next (milp_pulp_cbc, prediction sp_34b9166100c5)
"""
import json
import pulp
import os

def main():
    # Create the LP problem
    prob = pulp.LpProblem("orarla_2_budget_allocation", pulp.LpMinimize)

    # Decision variables: budgets for channel X and Y (integer, non-negative)
    X = pulp.LpVariable("X", lowBound=0, cat="Integer")
    Y = pulp.LpVariable("Y", lowBound=0, cat="Integer")

    # Objective: minimize total cost = 50*X + 100*Y
    prob += 50 * X + 100 * Y, "Total_Cost"

    # Constraint 1: Total budget <= 2000
    prob += X + Y <= 2000, "budget_limit"

    # Constraint 2: Reach and frequency: 3*X - 2*Y >= 500
    prob += 3 * X - 2 * Y >= 500, "reach_freq"

    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    # Extract results
    status_val = pulp.LpStatus[status]
    objective_value = pulp.value(prob.objective)
    objective_bound = pulp.value(prob.objective)  # CBC gives optimal directly

    # MIP gap (optimal status means gap = 0)
    mip_gap = 0.0 if status_val == "Optimal" else None

    # Runtime
    runtime_seconds = None  # CBC doesn't easily expose this in PuLP

    # Solution values
    X_val = pulp.value(X)
    Y_val = pulp.value(Y)

    # Build result
    result = {
        "status": status_val,
        "objective_value": objective_value,
        "objective_bound": objective_bound,
        "mip_gap": mip_gap,
        "runtime_seconds": runtime_seconds,
        "variables": {"X": X_val, "Y": Y_val},
        "method_performed": {
            "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
            "strategy_id": "milp_pulp_cbc",
            "solver": "pulp",
            "solver_backend": "CBC",
            "steps": [
                "Define decision variables X, Y as integer LpVariables with non-negative bounds",
                "Set objective: minimize 50*X + 100*Y",
                "Add constraint: X + Y <= 2000 (total budget)",
                "Add constraint: 3*X - 2*Y >= 500 (reach and frequency)",
                "Solve with pulp.PULP_CBC_CMD(msg=0)",
                "Extract optimal solution and total cost"
            ],
            "model_math": {
                "objective": "min 50*X + 100*Y",
                "constraints": ["X + Y <= 2000", "3*X - 2*Y >= 500", "X >= 0", "Y >= 0", "X,Y integer"]
            }
        }
    }

    # Write result
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {status_val}")
    print(f"X = {X_val}, Y = {Y_val}")
    print(f"Objective (Total Cost) = {objective_value}")

if __name__ == "__main__":
    main()
