"""
ORClaw solve for orarla_9: Haus Toys production planning (MILP with binary indicators).
Strategy: milp_binary_indicator
Solver: HiGHS (highspy 1.15+ high-level API)
"""
import json
import highspy

def solve():
    # Profits per unit
    PROFIT = {"truck": 5, "airplane": 10, "boat": 8, "train": 7}

    # Resource consumption
    WOOD = {"truck": 12, "airplane": 20, "boat": 15, "train": 10}
    STEEL = {"truck": 6, "airplane": 3, "boat": 5, "train": 4}

    # Resource availability
    WOOD_AVAIL = 890
    STEEL_AVAIL = 500

    # Tightened big-M values (min over wood-bound and steel-bound maxima)
    # Truck: wood→74, steel→83 → M_t=74
    # Airplane: wood→44, steel→166 → M_a=44
    # Boat: wood→59, steel→100 → M_b=59
    # Train: wood→89, steel→125 → M_n=89
    M = {"truck": 74, "airplane": 44, "boat": 59, "train": 89}

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)

    # Integer production variables
    t = h.addVariable(lb=0.0, ub=float(M["truck"]), obj=PROFIT["truck"],
                      type=highspy.HighsVarType.kInteger, name="truck")
    a = h.addVariable(lb=0.0, ub=float(M["airplane"]), obj=PROFIT["airplane"],
                      type=highspy.HighsVarType.kInteger, name="airplane")
    b = h.addVariable(lb=0.0, ub=float(M["boat"]), obj=PROFIT["boat"],
                      type=highspy.HighsVarType.kInteger, name="boat")
    n = h.addVariable(lb=0.0, ub=float(M["train"]), obj=PROFIT["train"],
                      type=highspy.HighsVarType.kInteger, name="train")

    # Binary indicator variables (1 if corresponding product is manufactured)
    # Note: highspy has no kBinary; use kInteger with bounds [0,1]
    m_t = h.addVariable(lb=0.0, ub=1.0, obj=0.0,
                        type=highspy.HighsVarType.kInteger, name="m_truck")
    m_a = h.addVariable(lb=0.0, ub=1.0, obj=0.0,
                        type=highspy.HighsVarType.kInteger, name="m_airplane")
    m_b = h.addVariable(lb=0.0, ub=1.0, obj=0.0,
                        type=highspy.HighsVarType.kInteger, name="m_boat")
    m_n = h.addVariable(lb=0.0, ub=1.0, obj=0.0,
                        type=highspy.HighsVarType.kInteger, name="m_train")

    # Maximize profit
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    # Resource constraints
    h.addConstr(t * WOOD["truck"] + a * WOOD["airplane"] + b * WOOD["boat"] + n * WOOD["train"]
                <= WOOD_AVAIL, name="wood")
    h.addConstr(t * STEEL["truck"] + a * STEEL["airplane"] + b * STEEL["boat"] + n * STEEL["train"]
                <= STEEL_AVAIL, name="steel")

    # Linking constraints: quantity <= M * indicator
    h.addConstr(t <= M["truck"] * m_t, name="link_truck_ub")
    h.addConstr(a <= M["airplane"] * m_a, name="link_airplane_ub")
    h.addConstr(b <= M["boat"] * m_b, name="link_boat_ub")
    h.addConstr(n <= M["train"] * m_n, name="link_train_ub")

    # Linking constraints: if indicator=1, quantity>=1 (minimum one unit)
    h.addConstr(t >= m_t, name="link_truck_lb")
    h.addConstr(a >= m_a, name="link_airplane_lb")
    h.addConstr(b >= m_b, name="link_boat_lb")
    h.addConstr(n >= m_n, name="link_train_lb")

    # Mutual exclusion: truck + train cannot both be manufactured
    h.addConstr(m_t + m_n <= 1, name="mutual_exclusion_truck_train")

    # Conditional: if boat>0 then airplane>0 (m_b <= m_a)
    h.addConstr(m_b <= m_a, name="boat_requires_airplane")

    # Boats cannot exceed trains
    h.addConstr(b <= n, name="boats_le_trains")

    # Solve
    h.run()
    status = h.getModelStatus()
    sol = h.getSolution()

    col_val = sol.col_value
    col_dual = sol.col_dual

    # Extract variable values
    def get_var(var):
        return col_val[var.index]

    trucks_val = get_var(t)
    airplanes_val = get_var(a)
    boats_val = get_var(b)
    trains_val = get_var(n)
    m_truck_val = get_var(m_t)
    m_airplane_val = get_var(m_a)
    m_boat_val = get_var(m_b)
    m_train_val = get_var(m_n)

    # Compute objective value
    obj_val = (PROFIT["truck"] * trucks_val + PROFIT["airplane"] * airplanes_val +
               PROFIT["boat"] * boats_val + PROFIT["train"] * trains_val)

    # Get bound and runtime
    mip_info = h.getInfo()
    runtime = h.getRunTime()
    objective_value = h.getObjectiveValue()
    objective_bound = mip_info.mip_dual_bound

    # Determine status
    if status == highspy.HighsModelStatus.kOptimal:
        status_str = "optimal"
    elif status == highspy.HighsModelStatus.kMipBetter:
        status_str = "mip_better"
    elif status == highspy.HighsModelStatus.kMipOptimal:
        status_str = "mip_optimal"
    elif status == highspy.HighsModelStatus.kInfeasible:
        status_str = "infeasible"
    else:
        status_str = f"model_status_{status}"

    result = {
        "status": status_str,
        "objective_value": objective_value,
        "objective_bound": objective_bound,
        "runtime_seconds": runtime,
        "variables": {
            "trucks": trucks_val,
            "airplanes": airplanes_val,
            "boats": boats_val,
            "trains": trains_val,
            "m_truck": m_truck_val,
            "m_airplane": m_airplane_val,
            "m_boat": m_boat_val,
            "m_train": m_train_val
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {status_str}")
    print(f"Objective: {objective_value}")
    print(f"Bound: {objective_bound}")
    print(f"Runtime: {runtime}s")
    print(f"Solution: trucks={trucks_val}, airplanes={airplanes_val}, boats={boats_val}, trains={trains_val}")
    print(f"Indicators: m_truck={m_truck_val}, m_airplane={m_airplane_val}, m_boat={m_boat_val}, m_train={m_train_val}")

if __name__ == "__main__":
    solve()
