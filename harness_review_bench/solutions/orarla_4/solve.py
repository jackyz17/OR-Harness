"""
orarla_4: Animal Farm Profit Maximization
MILP with integer variables using HiGHS高层API
"""
import json
import time
import highspy

STATUS_MAP = {
    highspy.HighsModelStatus.kOptimal: "optimal",
    highspy.HighsModelStatus.kInfeasible: "infeasible",
    highspy.HighsModelStatus.kUnbounded: "unbounded",
    highspy.HighsModelStatus.kUnboundedOrInfeasible: "unbounded_or_infeasible",
}

start = time.time()

h = highspy.Highs()
h.setOptionValue("output_flag", False)

x_cow = h.addVariable(lb=10, ub=highspy.kHighsInf, obj=400.0, type=highspy.HighsVarType.kInteger, name="x_cow")
x_sheep = h.addVariable(lb=20, ub=highspy.kHighsInf, obj=120.0, type=highspy.HighsVarType.kInteger, name="x_sheep")
x_chicken = h.addVariable(lb=0, ub=50, obj=3.0, type=highspy.HighsVarType.kInteger, name="x_chicken")

h.addConstr(10*x_cow + 5*x_sheep + 3*x_chicken <= 800, "manure")
h.addConstr(x_cow + x_sheep + x_chicken <= 100, "total_animals")

h.changeObjectiveSense(highspy.ObjSense.kMaximize)
h.run()

status_enum = h.getModelStatus()
solution = h.getSolution()
info = h.getInfo()
elapsed = time.time() - start

status_str = STATUS_MAP.get(status_enum, str(status_enum.name))

result = {
    "status": status_str,
    "objective_value": info.objective_function_value,
    "objective_bound": None,
    "runtime_seconds": elapsed,
    "solution": {
        "x_cow": solution.col_value[x_cow.index],
        "x_sheep": solution.col_value[x_sheep.index],
        "x_chicken": solution.col_value[x_chicken.index]
    }
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Status: {status_str}")
print(f"Objective: {info.objective_function_value}")
print(f"x_cow={solution.col_value[x_cow.index]}, x_sheep={solution.col_value[x_sheep.index]}, x_chicken={solution.col_value[x_chicken.index]}")
