#!/usr/bin/env python3
"""
orarla_9: Advertising budget allocation (integer MILP with HiGHS highspy API)

Problem:
  Variables: Ad1, Ad2 (integer, >= 0)
  Minimize:  5*Ad1 + 3*Ad2
  Subject to:
    C1: Ad1 + Ad2 >= 50          (effectiveness constraint)
    C2: 4*Ad1 + 6*Ad2 <= 300     (audience fatigue constraint)

Method: Integer MILP solved by HiGHS via highspy 1.15 native C-style API.
  - addVariable() for each variable
  - addRow(lower, upper, num_nz, idx_array, val_array) for each constraint
  - mip_rel_gap/mip_abs_gap set to ~0 for exact optimal solution
  - getSolution().col_value to read results
"""

import json
import os
import time
import numpy as np
import highspy

def solve():
    action_id = os.environ.get("OR_ACTION_ID", "unknown")

    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("time_limit", 60.0)
    h.setOptionValue("mip_rel_gap", 1e-9)
    h.setOptionValue("mip_abs_gap", 1e-9)

    # Add variables: integer >= 0, obj coefficients 5 and 3
    ad1_var = h.addVariable(lb=0.0, ub=highspy.kHighsInf, obj=5.0,
                            type=highspy.HighsVarType.kInteger, name="Ad1")
    ad2_var = h.addVariable(lb=0.0, ub=highspy.kHighsInf, obj=3.0,
                            type=highspy.HighsVarType.kInteger, name="Ad2")

    v1 = int(ad1_var)  # column index
    v2 = int(ad2_var)

    # C1: Ad1 + Ad2 >= 50  =>  lower=50, upper=+inf
    status_c1 = h.addRow(
        50.0, highspy.kHighsInf, 2,
        np.array([v1, v2], dtype=np.int32),
        np.array([1.0, 1.0], dtype=np.float64)
    )

    # C2: 4*Ad1 + 6*Ad2 <= 300  =>  lower=-inf, upper=300
    status_c2 = h.addRow(
        -highspy.kHighsInf, 300.0, 2,
        np.array([v1, v2], dtype=np.int32),
        np.array([4.0, 6.0], dtype=np.float64)
    )

    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    start = time.time()
    status = h.solve()
    elapsed = time.time() - start

    model_status = h.getModelStatus()
    sol = h.getSolution()
    info = h.getInfo()

    model_status_map = {
        highspy.HighsModelStatus.kOptimal: "optimal",
        highspy.HighsModelStatus.kInfeasible: "infeasible",
        highspy.HighsModelStatus.kUnbounded: "unbounded",
        highspy.HighsModelStatus.kTimeLimit: "time_limit",
        highspy.HighsModelStatus.kObjectiveBound: "objective_bound",
        highspy.HighsModelStatus.kObjectiveTarget: "objective_target",
    }

    solver_str = str(status)
    model_str = model_status_map.get(model_status, str(model_status))

    col_values = sol.col_value
    ad1_val = float(col_values[v1]) if v1 < len(col_values) else None
    ad2_val = float(col_values[v2]) if v2 < len(col_values) else None

    # Objective value
    obj_value = None
    try:
        obj_value = float(info.objective_function_value)
    except Exception:
        pass

    # Dual bound (MIP lower bound)
    dual_bound = None
    try:
        dual_bound = float(info.mip_node_logical_lower_bound)
    except Exception:
        pass

    # MIP gap
    mip_gap = None
    try:
        mip_gap = float(info.mip_gap)
    except Exception:
        pass

    result = {
        "status": model_str,
        "solver_status": solver_str,
        "objective_value": obj_value,
        "objective_bound": dual_bound,
        "mip_gap": mip_gap,
        "runtime_seconds": round(elapsed, 6),
        "variables": {
            "Ad1": round(ad1_val, 10) if ad1_val is not None else None,
            "Ad2": round(ad2_val, 10) if ad2_val is not None else None
        },
        "method_performed": {
            "action_id": action_id,
            "name": "Integer MILP with HiGHS (highspy native API)",
            "solver": "highspy",
            "solver_version": f"{highspy.HIGHS_VERSION_MAJOR}.{highspy.HIGHS_VERSION_MINOR}.{highspy.HIGHS_VERSION_PATCH}",
            "model": "minimize 5*Ad1 + 3*Ad2 | Ad1+Ad2>=50, 4*Ad1+6*Ad2<=300 | Ad1,Ad2 integer>=0"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    solve()
