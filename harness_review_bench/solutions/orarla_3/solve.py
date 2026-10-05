"""
orarla_3: Foldable table production & HR planning (Jan-Jun)
Method: milp_hiring_firing_decoupled
Maximize total net profit over 6 months.
"""

import json
import highspy

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
INITIAL_WORKFORCE = 1000
INITIAL_INVENTORY = 15000
INITIAL_BACKORDER = 0
DEMAND = [20000, 40000, 42000, 35000, 19000, 18500]

SELL_PRICE = 300
RAW_MATERIAL_COST = 90
OUTSOURCE_COST = 200
HOLDING_COST = 15
BACKORDER_COST = 35
REGULAR_WAGE = 30
OVERTIME_WAGE = 40
REGULAR_HOURS = 160
MAX_OT_PER_WORKER = 20
LABOR_HOURS_PER_UNIT = 5
HIRE_COST = 5000
FIRE_COST = 8000

# Variable kinds (for indexing)
PRODUCE = 0; OUTSOURCE = 1; HIRE = 2; FIRE = 3
WORKFORCE = 4; INV = 5; BO = 6; SELL = 7; OT = 8
VAR_PER_MONTH = 9

def var_idx(t, kind):
    return t * VAR_PER_MONTH + kind

def solve():
    h = highspy.Highs()
    h.setOptionValue("time_limit", 120.0)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("log_to_console", True)

    num_months = len(MONTHS)

    # Objective coefficients per variable kind
    obj_coeffs = {
        PRODUCE:   -RAW_MATERIAL_COST,
        OUTSOURCE: -OUTSOURCE_COST,
        HIRE:      -HIRE_COST,
        FIRE:      -FIRE_COST,
        WORKFORCE: -REGULAR_WAGE * REGULAR_HOURS,
        INV:       -HOLDING_COST,
        BO:        -BACKORDER_COST,
        SELL:       SELL_PRICE,
        OT:        -OVERTIME_WAGE,
    }

    # Track highs_var objects by (t, kind)
    var_obj = {}

    # Add all variables
    for t in range(num_months):
        for kind, name in enumerate(["produce","outsource","hire","fire","workforce","inv","bo","sell","ot"]):
            var_type = (highspy.HighsVarType.kContinuous if kind == OT
                        else highspy.HighsVarType.kInteger)
            result = h.addVariable(0.0, 1e9, obj_coeffs[kind], var_type, f"{name}_{MONTHS[t]}")
            var_obj[(t, kind)] = result

    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    # Variable shortcut - returns highs_var object
    def v(t, kind):
        return var_obj[(t, kind)]

    # C1: Workforce dynamics
    for t in range(num_months):
        if t == 0:
            # workforce[0] - hire[0] + fire[0] = 1000
            h.addConstr(v(0,WORKFORCE) - v(0,HIRE) + v(0,FIRE) == float(INITIAL_WORKFORCE), "C1_Jan")
        else:
            h.addConstr(v(t,WORKFORCE) - v(t-1,WORKFORCE) - v(t,HIRE) + v(t,FIRE) == 0.0, f"C1_{MONTHS[t]}")

    # C2: Labor capacity: produce*5 <= workforce*160 + ot
    for t in range(num_months):
        h.addConstr(
            v(t,PRODUCE) * LABOR_HOURS_PER_UNIT <= v(t,WORKFORCE) * REGULAR_HOURS + v(t,OT),
            f"C2a_{MONTHS[t]}"
        )
        h.addConstr(v(t,OT) <= v(t,WORKFORCE) * MAX_OT_PER_WORKER, f"C2b_{MONTHS[t]}")

    # C3: Inventory balance
    for t in range(num_months):
        if t == 0:
            h.addConstr(
                v(0,INV) - v(0,PRODUCE) - v(0,OUTSOURCE) + v(0,SELL) == float(INITIAL_INVENTORY),
                "C3_Jan"
            )
        else:
            h.addConstr(
                v(t,INV) - v(t-1,INV) - v(t,PRODUCE) - v(t,OUTSOURCE) + v(t,SELL) == 0.0,
                f"C3_{MONTHS[t]}"
            )

    # C4: Backorder balance
    for t in range(num_months):
        if t == 0:
            h.addConstr(v(0,BO) + v(0,SELL) == float(DEMAND[t]), f"C4_Jan")
        else:
            h.addConstr(
                v(t,BO) - v(t-1,BO) + v(t,SELL) == float(DEMAND[t]),
                f"C4_{MONTHS[t]}"
            )

    # C5: Sales availability
    for t in range(num_months):
        if t == 0:
            h.addConstr(
                v(0,SELL) <= float(INITIAL_INVENTORY) + v(0,PRODUCE) + v(0,OUTSOURCE),
                f"C5_Jan"
            )
        else:
            h.addConstr(
                v(t,SELL) <= v(t-1,INV) + v(t,PRODUCE) + v(t,OUTSOURCE) + v(t-1,BO),
                f"C5_{MONTHS[t]}"
            )

    # C6: Terminal inventory >= 10000
    h.addConstr(v(num_months-1,INV) >= 10000.0, "C6")

    # C7: Terminal backorder = 0
    h.addConstr(v(num_months-1,BO) == 0.0, "C7")

    # Solve
    h.run()
    status_str = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    runtime = h.getRunTime()

    result = {
        "status": status_str,
        "objective_value": None,
        "objective_bound": None,
        "runtime_seconds": runtime,
        "variables": {}
    }

    if status_str in ("Optimal", "Feasible"):
        sol = h.getSolution()
        col_vals = sol.col_value
        result["objective_value"] = h.getObjectiveValue()
        result["objective_bound"] = info.mip_dual_bound
        for t in range(num_months):
            for kind, name in enumerate(["produce","outsource","hire","fire","workforce","inv","bo","sell","ot"]):
                result["variables"][f"{name}_{MONTHS[t]}"] = col_vals[var_obj[(t,kind)].index]

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nStatus: {status_str}")
    print(f"Objective: {result['objective_value']}")
    print(f"Bound: {result['objective_bound']}")
    print(f"Runtime: {runtime:.4f}s")

if __name__ == "__main__":
    solve()
