"""
orarla_6: Production planning with inventory and backlog (multi-product, multi-period)
Method: milp_inventory_backlog_multi
Solver: HiGHS (high-level API)
"""

import json
import highspy

# ── Parameters ──────────────────────────────────────────────────────────────
products = ["I", "II", "III"]
quarters = [1, 2, 3, 4]

demand = {
    "I":   {1: 1500, 2: 1000, 3: 2000, 4: 1200},
    "II":  {1: 1500, 2: 1500, 3: 1200, 4: 1500},
    "III": {1: 1000, 2: 2000, 3: 1500, 4: 2500},
}

prod_hours = {"I": 2, "II": 4, "III": 3}
backlog_cost = {"I": 20, "II": 20, "III": 10}
inv_cost = 5
capacity = 15000          # production hours per quarter
final_stock = 150         # required ending inventory per product
no_prod_I_q2 = True      # product I cannot be produced in Q2

# ── HiGHS model ────────────────────────────────────────────────────────────
m = highspy.Highs()
m.setOptionValue("output_flag", True)
m.setOptionValue("log_to_console", True)

# Decision variables: production, inventory, backlog per product per quarter
# x[p][q]  – production
# inv[p][q] – inventory (stock on hand)
# back[p][q] – backlog (unfulfilled demand carried forward)

x_var = {}    # production
inv_var = {}  # inventory
back_var = {} # backlog

# Add production variables (integer, >= 0)
for p in products:
    for q in quarters:
        name = f"x_{p}_{q}"
        x_var[(p, q)] = m.addVariable(
            lb=0.0,
            ub=1e9,
            obj=0.0,
            type=highspy.HighsVarType.kInteger,
            name=name,
        )

# Add inventory variables (integer, >= 0)
for p in products:
    for q in quarters:
        name = f"inv_{p}_{q}"
        inv_var[(p, q)] = m.addVariable(
            lb=0.0,
            ub=1e9,
            obj=inv_cost,
            type=highspy.HighsVarType.kInteger,
            name=name,
        )

# Add backlog variables (integer, >= 0)
for p in products:
    for q in quarters:
        name = f"back_{p}_{q}"
        back_var[(p, q)] = m.addVariable(
            lb=0.0,
            ub=1e9,
            obj=backlog_cost[p],
            type=highspy.HighsVarType.kInteger,
            name=name,
        )

# ── Constraints ────────────────────────────────────────────────────────────

# 1. Flow balance: inv - back at q = inv - back at q-1 + x - demand
#    Rearranged: inv[p][q] - back[p][q] - inv[p][q-1] + back[p][q-1] - x[p][q] = -demand[p][q]
for p in products:
    for q in quarters:
        if q == 1:
            # inv[p][1] - back[p][1] - x[p][1] = -demand[p][1]
            expr = (inv_var[(p, 1)]
                    - back_var[(p, 1)]
                    - x_var[(p, 1)])
            rhs = -demand[p][1]
        else:
            # inv[p][q] - back[p][q] - inv[p][q-1] + back[p][q-1] - x[p][q] = -demand[p][q]
            expr = (inv_var[(p, q)]
                    - back_var[(p, q)]
                    - inv_var[(p, q-1)]
                    + back_var[(p, q-1)]
                    - x_var[(p, q)])
            rhs = -demand[p][q]
        m.addConstr(expr == rhs, name=f"flow_{p}_{q}")

# 2. Final inventory = 150, final backlog = 0
for p in products:
    m.addConstr(inv_var[(p, 4)] == final_stock, name=f"final_inv_{p}")
    m.addConstr(back_var[(p, 4)] == 0.0, name=f"final_back_{p}")

# 3. Production capacity: sum_p x[p][q] * prod_hours[p] <= capacity
for q in quarters:
    expr = sum(x_var[(p, q)] * prod_hours[p] for p in products)
    m.addConstr(expr <= capacity, name=f"capacity_{q}")

# 4. No product I in Q2
if no_prod_I_q2:
    m.addConstr(x_var[("I", 2)] == 0.0, name="no_I_Q2")

# ── Objective ──────────────────────────────────────────────────────────────
# Already set obj on inv and back variables; production cost = 0
m.changeObjectiveSense(highspy.ObjSense.kMinimize)

# ── Solve ──────────────────────────────────────────────────────────────────
status = m.run()
optimal = m.getModelStatus() == highspy.HighsModelStatus.kOptimal

# ── Extract solution ────────────────────────────────────────────────────────
solution = m.getSolution()
col_vals = solution.col_value

def get_var_val(v):
    return col_vals[v.index]

result = {
    "status": m.modelStatusToString(m.getModelStatus()),
    "optimal": optimal,
    "objective_value": m.getObjectiveValue(),
    "objective_bound": None,
    "runtime_seconds": m.getRunTime(),
    "production": {p: {str(q): get_var_val(x_var[(p, q)]) for q in quarters} for p in products},
    "inventory": {p: {str(q): get_var_val(inv_var[(p, q)]) for q in quarters} for p in products},
    "backlog": {p: {str(q): get_var_val(back_var[(p, q)]) for q in quarters} for p in products},
    "variables": {},
}

# Build variables dict for check
for p in products:
    for q in quarters:
        result["variables"][f"x_{p}_{q}"] = get_var_val(x_var[(p, q)])
        result["variables"][f"inv_{p}_{q}"] = get_var_val(inv_var[(p, q)])
        result["variables"][f"back_{p}_{q}"] = get_var_val(back_var[(p, q)])

# Objective bound = objective value when optimal
result["objective_bound"] = result["objective_value"] if optimal else None

# ── Write result ────────────────────────────────────────────────────────────
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=int)

print(f"\n=== RESULT ===")
print(f"Status:     {result['status']}")
print(f"Objective:  {result['objective_value']}")
print(f"Bound:      {result['objective_bound']}")
print(f"Runtime:    {result['runtime_seconds']:.3f}s")
print("\nProduction (units/quarter):")
for p in products:
    print(f"  {p}: {[int(result['production'][p][str(q)]) for q in quarters]}")
print("\nInventory (units/quarter-end):")
for p in products:
    print(f"  {p}: {[int(result['inventory'][p][str(q)]) for q in quarters]}")
print("\nBacklog (units/quarter-end):")
for p in products:
    print(f"  {p}: {[int(result['backlog'][p][str(q)]) for q in quarters]}")
