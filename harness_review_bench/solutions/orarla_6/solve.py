#!/usr/bin/env python3
"""
solve.py for orarla_6 - Marketing Budget Allocation (ILP)
Method: HiGHS MILP (highspy)

min 10*X + 20*Y
s.t. X + Y <= 1000       (total budget)
     2*X + 3*Y >= 2000   (effectiveness constraint)
     X >= 0, Y >= 0
     X, Y integer (budget units are indivisible)
"""
import json
import time
import os
import highspy

def solve():
    t0 = time.time()
    
    # Create HiGHS model
    h = highspy.Highs()
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("log_to_console", False)
    
    # Add variables: X (budget for channel X), Y (budget for channel Y)
    # Both are non-negative integers
    # Objective: min 10*X + 20*Y
    x = h.addVariable(lb=0.0, ub=float('inf'), obj=10.0, type=highspy.HighsVarType.kInteger, name="X")
    y = h.addVariable(lb=0.0, ub=float('inf'), obj=20.0, type=highspy.HighsVarType.kInteger, name="Y")
    
    # Constraint 1: X + Y <= 1000 (total budget limit)
    h.addConstr(x + y <= 1000.0, "budget_limit")
    
    # Constraint 2: 2*X + 3*Y >= 2000 (effectiveness requirement)
    h.addConstr(2*x + 3*y >= 2000.0, "effectiveness")
    
    # Set minimize (default is minimize for Highs)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    
    # Solve
    status = h.run()
    runtime = time.time() - t0
    
    # Extract solution
    solution = h.getSolution()
    model_status = h.getModelStatus()
    
    # Get variable values
    X_val = solution.col_value[x.index]
    Y_val = solution.col_value[y.index]
    
    # Get objective value
    objective_value = h.getObjectiveValue()
    
    # Get info
    info = h.getInfo()
    objective_bound = info.mip_dual_bound if hasattr(info, 'mip_dual_bound') and info.mip_dual_bound is not None else None
    mip_gap = info.mip_gap if hasattr(info, 'mip_gap') and info.mip_gap is not None else None
    
    # Determine status
    if model_status == highspy.HighsModelStatus.kOptimal:
        status_str = "optimal"
    elif model_status == highspy.HighsModelStatus.kInfeasible:
        status_str = "infeasible"
    elif model_status == highspy.HighsModelStatus.kUnbounded:
        status_str = "unbounded"
    else:
        status_str = f"model_status_{model_status}"
    
    result = {
        "status": status_str,
        "objective_value": round(objective_value) if objective_value is not None else None,
        "objective_bound": objective_bound,
        "mip_gap": mip_gap,
        "runtime_seconds": round(runtime, 4),
        "variables": {
            "X": X_val,
            "Y": Y_val
        },
        "method_performed": {
            "name": "HiGHS_MILP",
            "solver": "highspy",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }
    
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Status: {status_str}")
    print(f"Objective: {objective_value}")
    print(f"X={X_val}, Y={Y_val}")
    print(f"Runtime: {runtime:.4f}s")
    
    return result

if __name__ == "__main__":
    solve()
