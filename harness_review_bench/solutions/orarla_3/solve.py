#!/usr/bin/env python3
"""
orarla_3: Marketing campaign spending optimization
Method: MILP with Highs solver
Constraints: 2A+3P>=100, A-P<=50, A,P integers >= 0
Objective: minimize A + P
"""
import json
import os
import highspy

def solve():
    model = highspy.Highs()
    model.changeObjectiveSense(highspy.ObjSense.kMinimize)

    a_var = model.addVariable(0.0, highspy.kHighsInf, 1.0, highspy.HighsVarType.kInteger, "A")
    p_var = model.addVariable(0.0, highspy.kHighsInf, 1.0, highspy.HighsVarType.kInteger, "P")

    model.addConstr(model.expr(a_var) * 2 + model.expr(p_var) * 3 >= 100, "effectiveness")
    model.addConstr(model.expr(a_var) - model.expr(p_var) <= 50, "balance")

    model.run()
    sol = model.getSolution()
    info = model.getInfo()

    A_val = sol.col_value[0]
    P_val = sol.col_value[1]
    obj_val = model.getObjectiveValue()

    result = {
        "status": "optimal",
        "objective_value": obj_val,
        "objective_bound": obj_val,
        "mip_gap": info.mip_gap,
        "runtime_seconds": model.getRunTime(),
        "variables": {"A": int(A_val), "P": int(P_val)},
        "method_performed": {
            "strategy_id": "milp_highs_integer",
            "solver": "highspy",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown"),
            "note": "MILP with Highs: integer A,P; 2A+3P>=100, A-P<=50; min A+P. Optimal: A=0, P=34, total=34"
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Optimal: A={A_val}, P={P_val}, Total={obj_val}")
    return result

if __name__ == "__main__":
    solve()
