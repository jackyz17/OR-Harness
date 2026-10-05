#!/usr/bin/env python3
"""
Mary Dinner Planning - MILP (milp_direct_integer)
Maximize fiber. Protein source is optional (0 or 100g via binary selection).
Vegetables in 100g discrete portions (integer), at least 2 types selected.
"""
import json
import highspy

def solve():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    INF = highspy.kHighsInf

    # --- Binary variables: protein source (optional, 0 or 100g) ---
    y_salmon = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_salmon")
    y_beef   = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_beef")
    y_pork   = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_pork")

    # --- Integer variables: vegetable portions (100g each, 0-6 portions) ---
    okra_p    = h.addVariable(0.0, 6.0, 0.0, highspy.HighsVarType.kInteger, "okra_p")
    carrots_p = h.addVariable(0.0, 6.0, 0.0, highspy.HighsVarType.kInteger, "carrots_p")
    celery_p  = h.addVariable(0.0, 6.0, 0.0, highspy.HighsVarType.kInteger, "celery_p")
    cabbage_p = h.addVariable(0.0, 6.0, 0.0, highspy.HighsVarType.kInteger, "cabbage_p")

    # --- Continuous protein grams (0 or 100g) ---
    salmon_g = h.addVariable(0.0, INF, 0.0, highspy.HighsVarType.kContinuous, "salmon_g")
    beef_g   = h.addVariable(0.0, INF, 0.0, highspy.HighsVarType.kContinuous, "beef_g")
    pork_g   = h.addVariable(0.0, INF, 0.0, highspy.HighsVarType.kContinuous, "pork_g")

    # --- Objective: maximize fiber (grams) ---
    # Fiber per 100g: okra=3.2, carrots=2.7, celery=1.6, cabbage=2.0
    # So fiber = 3.2*okra_p + 2.7*carrots_p + 1.6*celery_p + 2.0*cabbage_p (portions * 100g * fiber/100g = fiber in g)
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    h.changeColCost(okra_p.index,    3.2)
    h.changeColCost(carrots_p.index, 2.7)
    h.changeColCost(celery_p.index,  1.6)
    h.changeColCost(cabbage_p.index, 2.0)

    # --- Constraints ---

    # 1. At most one protein source (0 or 1)
    h.addConstr(y_salmon + y_beef + y_pork <= 1.0, "at_most_one_protein")

    # 2. At least 2 vegetable types selected (sum of portions >= 2 ensures at least 2 types... but
    #    wait: sum>=2 could be 2+ portions of ONE vegetable. Need to check at least 2 TYPES)
    #    Actually: if okra_p >= 2, that counts as 2 portions but only 1 TYPE.
    #    The problem says "at least two kinds of vegetables among okra, carrots, celery, and cabbage"
    #    So we need at least 2 types selected. Use binary for type selection.
    y_okra    = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_okra")
    y_carrots = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_carrots")
    y_celery  = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_celery")
    y_cabbage = h.addVariable(0.0, 1.0, 0.0, highspy.HighsVarType.kInteger, "y_cabbage")
    h.addConstr(y_okra + y_carrots + y_celery + y_cabbage >= 2.0, "at_least_2_veg_types")

    # 3. Total weight = 600g
    h.addConstr(salmon_g + beef_g + pork_g + 100*(okra_p + carrots_p + celery_p + cabbage_p) == 600.0, "total_weight")

    # 4. Budget <= $15
    # Prices per 100g: salmon=4, beef=3.6, pork=1.8, okra=2.6, carrots=1.2, celery=1.6, cabbage=2.3
    h.addConstr(
        0.04*salmon_g + 0.036*beef_g + 0.018*pork_g +
        2.6*okra_p + 1.2*carrots_p + 1.6*celery_p + 2.3*cabbage_p <= 15.0,
        "budget"
    )

    # 5. Protein linking: if selected (binary=1), protein_g = 100; if not, protein_g = 0
    h.addConstr(salmon_g >= 100.0 * y_salmon, "link_salmon_lo")
    h.addConstr(salmon_g <= 100.0 * y_salmon, "link_salmon_hi")
    h.addConstr(beef_g   >= 100.0 * y_beef,   "link_beef_lo")
    h.addConstr(beef_g   <= 100.0 * y_beef,   "link_beef_hi")
    h.addConstr(pork_g   >= 100.0 * y_pork,   "link_pork_lo")
    h.addConstr(pork_g   <= 100.0 * y_pork,   "link_pork_hi")

    # 6. Vegetable portion linking: if type selected (binary=1), portions >= 1; if not, 0
    h.addConstr(okra_p    >= 1.0 * y_okra,    "link_okra")
    h.addConstr(carrots_p >= 1.0 * y_carrots, "link_carrots")
    h.addConstr(celery_p  >= 1.0 * y_celery,  "link_celery")
    h.addConstr(cabbage_p >= 1.0 * y_cabbage, "link_cabbage")
    # Upper bound on portions
    h.addConstr(okra_p    <= 6.0 * y_okra,    "link_okra_ub")
    h.addConstr(carrots_p <= 6.0 * y_carrots, "link_carrots_ub")
    h.addConstr(celery_p  <= 6.0 * y_celery,  "link_celery_ub")
    h.addConstr(cabbage_p <= 6.0 * y_cabbage, "link_cabbage_ub")

    # --- Solve ---
    h.run()
    status_enum = h.getModelStatus()
    sol = h.getSolution()
    col = sol.col_value

    protein_src = (
        "salmon" if col[y_salmon.index] > 0.5 else
        "beef"   if col[y_beef.index]   > 0.5 else
        "pork"   if col[y_pork.index]   > 0.5 else "none"
    )

    vegs = [n for n, v in [("okra", okra_p), ("carrots", carrots_p), ("celery", celery_p), ("cabbage", cabbage_p)] if col[v.index] > 0]

    result = {
        "status": h.modelStatusToString(status_enum),
        "objective_value": h.getObjectiveValue(),
        "objective_bound": None,
        "runtime_seconds": 0.0,
        "solution": {
            "protein_source": protein_src,
            "salmon_g": round(col[salmon_g.index], 2),
            "beef_g":   round(col[beef_g.index],   2),
            "pork_g":   round(col[pork_g.index],   2),
            "okra_g":    round(col[okra_p.index] * 100, 2),
            "carrots_g": round(col[carrots_p.index] * 100, 2),
            "celery_g":  round(col[celery_p.index] * 100, 2),
            "cabbage_g": round(col[cabbage_p.index] * 100, 2),
            "vegetables_selected": vegs,
            "total_fiber_g": round(h.getObjectiveValue(), 4)
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    solve()
