"""
Mary dinner planning: maximize fiber intake with MILP (HiGHS high-level API).
Protein: exactly 1 of salmon/beef/pork (binary vars)
Vegetables: at least 2 of okra/carrots/celery/cabbage (continuous grams + binary selection)
Budget: $15, Total weight: 600g
"""
import json
import highspy

def solve():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)

    # Add protein binaries (in 100g units, 0 or 1)
    salmon = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "salmon")
    beef   = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "beef")
    pork   = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "pork")

    # Add vegetable binaries (0 or 1)
    okra_b     = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "okra_b")
    carrots_b  = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "carrots_b")
    celery_b   = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "celery_b")
    cabbage_b  = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "cabbage_b")

    # Vegetable amounts in 100g units (0 to 6, since 600g = 6 * 100g)
    # Objective coefficients = fiber per 100g
    okra_amt    = h.addVariable(0.0, 6.0, 3.2, highspy.HighsVarType.kContinuous, "okra_amt")
    carrots_amt = h.addVariable(0.0, 6.0, 2.7, highspy.HighsVarType.kContinuous, "carrots_amt")
    celery_amt  = h.addVariable(0.0, 6.0, 1.6, highspy.HighsVarType.kContinuous, "celery_amt")
    cabbage_amt = h.addVariable(0.0, 6.0, 2.0, highspy.HighsVarType.kContinuous, "cabbage_amt")

    # Maximize fiber
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    # C1: exactly one protein
    h.addConstr(salmon + beef + pork == 1, "one_protein")

    # C2: total weight = 600g = 6 (in 100g units)
    h.addConstr(salmon + beef + pork + okra_amt + carrots_amt + celery_amt + cabbage_amt == 6, "total_weight")

    # C3: budget <= 15
    # Price per 100g: salmon=4, beef=3.6, pork=1.8, okra=2.6, carrots=1.2, celery=1.6, cabbage=2.3
    h.addConstr(
        4.0*salmon + 3.6*beef + 1.8*pork +
        2.6*okra_amt + 1.2*carrots_amt + 1.6*celery_amt + 2.3*cabbage_amt <= 15,
        "budget"
    )

    # C4: at least 2 vegetables selected
    h.addConstr(okra_b + carrots_b + celery_b + cabbage_b >= 2, "at_least_2_veg")

    # C5-C8: vegetable amount <= 6 * vegetable_binary
    h.addConstr(okra_amt <= 6.0 * okra_b, "link_okra")
    h.addConstr(carrots_amt <= 6.0 * carrots_b, "link_carrots")
    h.addConstr(celery_amt <= 6.0 * celery_b, "link_celery")
    h.addConstr(cabbage_amt <= 6.0 * cabbage_b, "link_cabbage")

    # Solve
    h.run()

    sol = h.getSolution()
    info = h.getInfo()
    model_status = h.getModelStatus().name.lower()

    # Extract solution
    col_names = h.allVariableNames()
    sol_dict = {col_names[i]: sol.col_value[i] for i in range(len(col_names))}

    # Determine protein type
    protein_map = {0: "salmon", 1: "beef", 2: "pork"}
    protein_val = sol_dict["salmon"] + 2*sol_dict["beef"] + 3*sol_dict["pork"]
    if sol_dict["salmon"] > 0.5:
        protein_type = "salmon"
    elif sol_dict["beef"] > 0.5:
        protein_type = "beef"
    elif sol_dict["pork"] > 0.5:
        protein_type = "pork"
    else:
        protein_type = "none"

    result = {
        "status": model_status,
        "objective_value": info.objective_function_value,
        "objective_bound": info.mip_dual_bound if info.mip_gap != float('inf') else info.objective_function_value,
        "runtime_seconds": h.getRunTime(),
        "solution_variables": sol_dict,
        "protein_type": protein_type,
        "vegetables_selected": [
            name for name in ["okra", "carrots", "celery", "cabbage"]
            if sol_dict.get(name + "_b", 0) > 0.5
        ],
        "vegetable_grams": {
            "okra": round(sol_dict.get("okra_amt", 0) * 100, 1),
            "carrots": round(sol_dict.get("carrots_amt", 0) * 100, 1),
            "celery": round(sol_dict.get("celery_amt", 0) * 100, 1),
            "cabbage": round(sol_dict.get("cabbage_amt", 0) * 100, 1),
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    solve()
