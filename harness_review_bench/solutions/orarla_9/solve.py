"""
Haus Toys profit maximization - MILP formulation
Method: milp_direct_integer

Variables:
  x_truck, x_airplane, x_boat, x_train: nonnegative integers
  y: binary (1=make trucks, 0=make trains)

Objective: maximize 5*x_truck + 10*x_airplane + 8*x_boat + 7*x_train

Constraints:
  Wood:   12*x_truck + 20*x_airplane + 15*x_boat + 10*x_train <= 890
  Steel:   6*x_truck +  3*x_airplane +  5*x_boat +  4*x_train <= 500
  Truck+Train mutually exclusive via big-M:
    x_truck <= 100*y
    x_train <= 100*(1-y)
  Boat requires airplane: x_boat <= x_airplane
  Boat <= Train: x_boat <= x_train
"""

import json
import highspy
import time

def solve():
    t_start = time.time()

    m = highspy.Highs()
    m.changeObjectiveSense(highspy.ObjSense.kMaximize)

    INF = highspy.kHighsInf

    x_truck = m.addVariable(lb=0.0, ub=INF, obj=5.0, type=highspy.HighsVarType.kInteger, name="x_truck")
    x_airplane = m.addVariable(lb=0.0, ub=INF, obj=10.0, type=highspy.HighsVarType.kInteger, name="x_airplane")
    x_boat = m.addVariable(lb=0.0, ub=INF, obj=8.0, type=highspy.HighsVarType.kInteger, name="x_boat")
    x_train = m.addVariable(lb=0.0, ub=INF, obj=7.0, type=highspy.HighsVarType.kInteger, name="x_train")
    y = m.addVariable(lb=0.0, ub=1.0, obj=0.0, type=highspy.HighsVarType.kInteger, name="y")

    # Wood: 12*x_truck + 20*x_airplane + 15*x_boat + 10*x_train <= 890
    m.addConstr(12*x_truck + 20*x_airplane + 15*x_boat + 10*x_train <= 890.0, name="wood")

    # Steel: 6*x_truck + 3*x_airplane + 5*x_boat + 4*x_train <= 500
    m.addConstr(6*x_truck + 3*x_airplane + 5*x_boat + 4*x_train <= 500.0, name="steel")

    # Mutual exclusion: x_truck <= 100*y  => x_truck - 100*y <= 0
    m.addConstr(x_truck + (-100.0)*y <= 0.0, name="truck_y")

    # x_train <= 100*(1-y) => x_train + 100*y <= 100
    m.addConstr(x_train + 100.0*y <= 100.0, name="train_y")

    # Boat requires airplane: x_boat <= x_airplane => x_boat - x_airplane <= 0
    m.addConstr(x_boat + (-1.0)*x_airplane <= 0.0, name="boat_airplane")

    # Boat <= Train: x_boat <= x_train => x_boat - x_train <= 0
    m.addConstr(x_boat + (-1.0)*x_train <= 0.0, name="boat_train")

    status = m.run()
    t_solve = time.time() - t_start

    sol = m.getSolution()
    model_status = m.getModelStatus()

    # Get solution values (use index-based access to be safe)
    col_value = sol.col_value

    x_truck_val = int(round(col_value[0]))
    x_airplane_val = int(round(col_value[1]))
    x_boat_val = int(round(col_value[2]))
    x_train_val = int(round(col_value[3]))
    y_val = int(round(col_value[4]))

    objective_value = m.getObjectiveValue()

    if model_status == highspy.HighsModelStatus.kOptimal:
        status_str = "optimal"
    elif model_status == highspy.HighsModelStatus.kMipOptimal:
        status_str = "optimal"
    elif model_status == highspy.HighsModelStatus.kInfeasible:
        status_str = "infeasible"
    else:
        status_str = f"status_{model_status.value}"

    result = {
        "status": status_str,
        "objective_value": objective_value,
        "objective_bound": objective_value if status_str == "optimal" else None,
        "runtime_seconds": t_solve,
        "solution_variables": {
            "x_truck": x_truck_val,
            "x_airplane": x_airplane_val,
            "x_boat": x_boat_val,
            "x_train": x_train_val,
            "y": y_val
        }
    }

    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    solve()
