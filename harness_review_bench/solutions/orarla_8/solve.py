#!/usr/bin/env python3
"""
ORClaw orarla_8: Budget allocation planning problem.
Minimize 10*X + 20*Y subject to X + Y <= 5000, 2*X + 3*Y >= 10000, X,Y integers.
"""
import json
import os
import time
import traceback

ACTION_ID = os.environ.get("OR_ACTION_ID", "unknown")
start_time = time.time()

def write_result(data):
    with open("result.json", "w") as f:
        json.dump(data, f, indent=2)

try:
    import pulp

    # Create the problem
    prob = pulp.LpProblem("orarla_8", pulp.LpMinimize)

    # Decision variables
    X = pulp.LpVariable("X", lowBound=0, upBound=5000, cat="Integer")
    Y = pulp.LpVariable("Y", lowBound=0, upBound=5000, cat="Integer")

    # Objective: minimize 10*X + 20*Y
    prob += 10 * X + 20 * Y, "Total_Cost"

    # Constraints
    prob += X + Y <= 5000, "resource_limit"
    prob += 2 * X + 3 * Y >= 10000, "effectiveness_min"

    # Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    result = prob.solve(solver)

    runtime = time.time() - start_time

    status_map = {
        pulp.LpStatusOptimal: "optimal",
        pulp.LpStatusNotSolved: "not_solved",
        pulp.LpStatusInfeasible: "infeasible",
        pulp.LpStatusUnbounded: "unbounded",
        pulp.LpStatusUndefined: "undefined",
    }
    status_str = status_map.get(result, f"unknown_{result}")

    obj_val = pulp.value(prob.objective) if result == pulp.LpStatusOptimal else None

    result_data = {
        "status": status_str,
        "objective_value": obj_val,
        "objective_bound": obj_val,
        "mip_gap": 0.0,
        "runtime_seconds": runtime,
        "variables": {"X": pulp.value(X), "Y": pulp.value(Y)} if result == pulp.LpStatusOptimal else None,
        "method_performed": {
            "action_id": ACTION_ID,
            "name": "Integer Linear Programming with PuLP + CBC solver",
            "solver": "pulp.PULP_CBC_CMD",
            "steps": [
                "Define X, Y as integer LpVariables with bounds [0, 5000]",
                "Set objective: minimize 10*X + 20*Y",
                "Add constraint: X + Y <= 5000",
                "Add constraint: 2*X + 3*Y >= 10000",
                "Solve with pulp.PULP_CBC_CMD(msg=0)",
                "Extract optimal solution"
            ]
        }
    }

    write_result(result_data)

except Exception as e:
    runtime = time.time() - start_time
    err_data = {
        "status": "error",
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc(),
        "runtime_seconds": runtime,
        "method_performed": {
            "action_id": ACTION_ID,
            "name": "Integer Linear Programming with PuLP + CBC solver",
            "error": str(e)
        }
    }
    write_result(err_data)
    raise SystemExit(1)
