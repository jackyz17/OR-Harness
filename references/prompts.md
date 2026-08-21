# Prompt Templates

## Contents

1. Normalization (framework-internal)
2. Structured modeling (think / model)
3. Structural-signature extraction
4. Semantic judge (L3)
5. Solver implementation
6. Sequential repair
7. Comparative synthesis (success vs failure)
8. Admission judge
9. Outer reflection (gold mismatch)
10. Induction: alignment
11. Induction: hypothesis generation
12. Induction: counterexample search

All structured outputs must be **raw JSON** (no markdown fences, no prefix text). The framework parses once; on failure it retries with a correction hint. Never write unparsed output to the bank. Under the harness the agent's LLM produces these outputs; the framework supplies the templates and validates/parses.

---

## 1. Normalization (framework-internal)

**Who**: `[framework]` only. The agent does not participate.

The framework normalizes the problem into `{normalized_description, problem_family, objective, entities, constraints, scale}` by keyword matching. The agent never sees this prompt.

---

## 2. Structured modeling (think / model)

**Who**: `[agent]` generates, `[framework]` validates.

**Input** (what the framework injects into the prompt):
- The original problem text
- Planning priors (if wired): `[E1]...` past modeling experiences (unified, all peers)
- On retry: the failed issues from the previous round

**Agent output**: `generate_text` → ` <think> ` + `<model>` in GAMS-style DSL.

**Example input (abridged)**:
```
You are an OR modeling expert. Analyze the problem, then formalize it
as a mathematical model using the GAMS-style blocks SETS / PARAMETERS /
VARIABLES / OBJECTIVE / CONSTRAINTS with symbolic indexing (e.g. x[i,t]).
Output EXACTLY two blocks:
 <think> your analysis
<model>the five blocks</model>

=== Validated optimization principles (offline induction) ===
[E1] When allocating integer quantities under a shared scarce resource
constraint, rank decision variables by profit-per-unit-of-resource-consumed...

=== Similar past modeling realizations (reference) ===
[E1] Profit-per-unit-of-scarce-resource ranking...

PROBLEM:
A farmer needs to decide how many cows, sheep, and chickens to raise...
```

**Example output**:
```
 <think> 
This is a resource-constrained profit maximization problem with three
integer decision variables. Profit per animal = sell price - feed cost.
The manure constraint is the binding scarce resource. I apply [E1]:
rank by profit/manure_rate (cow=40, sheep=24, chicken=1) and maximize cows.
[uses E1]
<model>
SETS:
 a in Animals = {cow, sheep, chicken}

PARAMETERS:
 sell_price[a]
 feed_cost[a]
 manure_rate[a]
 manure_limit
 max_chickens
 min_cows
 min_sheep
 max_total

VARIABLES:
 x[a] integer >= 0

OBJECTIVE:
 maximize sum(a, (sell_price[a] - feed_cost[a]) * x[a])

CONSTRAINTS:
 C1: sum(a, manure_rate[a] * x[a]) <= manure_limit
 C2: x[chicken] <= max_chickens
 C3: x[cow] >= min_cows
 C4: x[sheep] >= min_sheep
 C5: sum(a, x[a]) <= max_total
</model>
```

See [modeling-contract.md](modeling-contract.md) for DSL rules and [structural-signature.md](structural-signature.md) for the vocabulary.

---

## 3. Structural-signature extraction

**Who**: `[agent]` generates JSON, `[framework]` validates against controlled vocabulary.

**Input**: the verified `<model>` text + allowed vocabularies.

**Agent output**: `generate_object` → JSON object.

**Example output**:
```json
{
 "objective": "linear",
 "decision": ["integer_batch"],
 "constraint": ["capacity", "covering"],
 "interaction": "shared_resource_coupled",
 "features": {"resource": "shared_scarce"}
}
```

Out-of-vocabulary core values are rejected and retried with a correction hint listing the allowed values.

---

## 4. Semantic judge (L3)

**Who**: `[agent]` generates JSON array, `[framework]` feeds issues back if non-empty.

**Input**: the original problem + the verified model.

**Agent output**: `generate_object` → JSON array.

**Example output (no defects)**:
```json
[]
```

**Example output (missing constraint)**:
```json
[
 {"type": "missing_constraint", "detail": "TSP requires subtour elimination constraints, which are absent from the model"}
]
```

L3 is a no-op (returns passed) when no LLM client is injected.

---

## 5. Solver implementation

**Who**: `[agent]` generates Python code, `[framework]` executes in sandbox.

**Input** (what the framework injects):
- Original problem text
- Verified mathematical model
- Normalized problem spec (JSON)
- Solver context (solver name, family, API, attempt number)
- Result contract (write `result.json`)
- Execution limits (no network, no shell, no parent paths)
- Modeling experience hits (from Modeling Bank)
- Implementation experience hits (from Implementation Bank)

**Agent output**: `generate_text` → complete Python code. Markdown fences are OK (the framework strips them).

**Example output (Gurobi branch)**:
```python
import json
import gurobipy as gp
from gurobipy import GRB

model = gp.Model("farmer")
animals = ["cow", "sheep", "chicken"]
sell_price = {"cow": 500, "sheep": 200, "chicken": 8}
feed_cost = {"cow": 100, "sheep": 80, "chicken": 5}
manure_rate = {"cow": 10, "sheep": 5, "chicken": 3}

x = {}
for a in animals:
 x[a] = model.addVar(vtype=GRB.INTEGER, name=f"x_{a}")

model.setObjective(
 gp.quicksum((sell_price[a] - feed_cost[a]) * x[a] for a in animals),
 GRB.MAXIMIZE
)

model.addConstr(gp.quicksum(manure_rate[a] * x[a] for a in animals) <= 800, "C1")
model.addConstr(x["chicken"] <= 50, "C2")
model.addConstr(x["cow"] >= 10, "C3")
model.addConstr(x["sheep"] >= 20, "C4")
model.addConstr(gp.quicksum(x[a] for a in animals) <= 100, "C5")

model.optimize()

result = {
 "status": "optimal" if model.status == GRB.OPTIMAL else "unknown",
 "solver": "gurobi",
 "objective_sense": "maximize",
 "objective_value": model.objval if model.status == GRB.OPTIMAL else None,
 "variables": {a: int(x[a].x) for a in animals},
 "runtime_seconds": model.runtime,
}
with open("result.json", "w") as f:
 json.dump(result, f)
```

The framework reads `result.json`, not stdout. An exit without this file is an execution failure even if stdout claims success.

See [solver-adapters.md](solver-adapters.md) for the result contract and solver-specific API notes.

---

## 6. Sequential repair

**Who**: `[agent]` generates corrected code, `[framework]` executes.

**Input** (what the framework injects on attempt 2+):
- Everything from solver implementation (step 5)
- `This is sequential repair attempt N.`
- Latest branch state (current code summary, latest feedback, resolved/unresolved issues)
- Repair-graph guidance (ranked actions, pitfalls, repair path)
- Repair experience hits (from Repair Bank)
- `Return the complete latest code, not a patch.`

**Agent output**: `generate_text` → complete corrected Python code (not a diff/patch).

The framework always runs the full code, never applies a patch. On repeated error or unchanged code, the branch terminates.

---

## 7. Comparative synthesis (success vs failure)

**Who**: `[agent]` generates experience candidates, `[framework]` routes to judge.

**Runs only after a gold-matched success.** The framework builds one of two prompts:
- **Failures present**: contrast prompt (success vs all buffered failures)
- **No failures**: success-only prompt

**Input**: the problem, the successful outcome (verified model + per-branch results), and ALL buffered failures (if any).

**Agent output**: `generate_object` → JSON array of experience candidates.

**Example output**:
```json
[
 {
 "layer": "modeling",
 "title": "Profit-per-unit-of-scarce-resource ranking for integer allocation",
 "retrieval_text": "When allocating integer quantities under a shared scarce resource constraint, rank decision variables by profit-per-unit-of-resource-consumed and greedily favor the highest-ranked item until a constraint binds.",
 "polarity": "positive",
 "diagnosis": "The manure constraint is the binding scarce resource. Computing profit density (profit / resource_consumption) reveals which animal gives the most profit per unit of limited resource.",
 "action": "Calculate profit_per_unit = (sell_price - feed_cost) / resource_consumption for each variable. Sort descending. Allocate maximum feasible quantity of the highest-ranked variable first.",
 "rationale": "Greedy fractional-knapsack insight adapted to ILP: when one linear constraint is binding and variables have different profit density, the LP relaxation optimum aligns with the greedy ranking."
 }
]
```

Self-classify each lesson by WHERE it applies:
- `modeling` — formulation semantics
- `implementation` — solver/API mechanics
- `repair` — error→fix path
- `solving` — performance/solver-choice

Failure records are never appended on their own.

---

## 8. Admission judge

**Who**: `[agent]` generates verdict, `[framework]` appends if accepted.

**Input**: one synthesis candidate.

**Agent output**: `generate_object` → JSON object.

**Example output**:
```json
{"accept": true, "reason": "Actionable, accurate, and valuable for future resource-constrained allocation problems."}
```

Only accepted candidates are appended (routed by layer: `modeling` → ModelingStore, others → flat store). When no LLM is available, the judge is a no-op (accepts everything that passed structural validation).

---

## 9. Outer reflection (gold mismatch)

**Who**: `[agent]` generates new modeling direction, `[framework]` feeds back.

**Input**: problem, gold answer, selected objective, per-branch outcomes.

**Agent output**: `generate_text` → free text analyzing why the **modeling direction** was wrong (not the code) and how the model should change.

**Example output**:
```
The gold objective is 30400 but the model produced 28000. The model is
correct in structure but the solver returned a suboptimal result because
the objective coefficients were computed as selling price only, without
subtracting feed cost. The model's OBJECTIVE should be:
 maximize sum(a, (sell_price[a] - feed_cost[a]) * x[a])
The current model has:
 maximize sum(a, sell_price[a] * x[a])
Fix: subtract feed_cost[a] from each term in the objective.
```

This drives a fresh Structured Modeling round (≤3 outer rounds). The reflection returns to modeling, not to code generation.

---

## 10. Induction: alignment

**Who**: `[agent]` generates role mappings, `[framework]` checks grounding.

**Input**: a candidate cluster (list of realizations with their structural signatures and models).

**Agent output**: `generate_object` → JSON with `roles` and `bindings`.

**Example output**:
```json
{
 "roles": [
 {"name": "resource_pool", "description": "The scarce shared resource being allocated"},
 {"name": "competing_decisions", "description": "Integer quantities competing for the resource"},
 {"name": "objective_contribution", "description": "Profit or cost per unit of decision"}
 ],
 "bindings": [
 {"role": "resource_pool", "source": "exp_abc", "concrete": "manure capacity (800 units)"},
 {"role": "competing_decisions", "source": "exp_def", "concrete": "cows, sheep, chickens (integer quantities)"},
 {"role": "objective_contribution", "source": "exp_ghi", "concrete": "profit per animal (400, 120, 3)"}
 ]
}
```

Every binding **must cite its source `realization_id`** (grounding). Ungrounded bindings are dropped. The canonical role names are: `resource_pool`, `capacity_limit`, `competing_decisions`, `objective_contribution`, `demand_requirement`, `coupling_constraint`, `time_period`, `flow_balance`.

---

## 11. Induction: hypothesis generation

**Who**: `[agent]` generates candidate principle(s), `[framework]` stamps `status=hypothesis`.

**Input**: the cluster + the alignment map.

**Agent output**: `generate_object` → JSON object (or array of objects for multiple hypotheses).

**Example output**:
```json
{
 "statement": "When allocating integer quantities under a shared scarce resource constraint, rank decision variables by profit-per-unit-of-resource-consumed and greedily favor the highest-ranked item until a constraint binds. The binding constraint is the one with the lowest capacity-to-consumption ratio.",
 "rationale": "Across inventory (storage capacity), production (machine hours), and workforce (labor hours), the same greedy density ranking produces optimal or near-optimal integer allocations because the LP relaxation optimum aligns with the ranking.",
 "complexity": 2
}
```

The principle is grounded in the alignment, not a free-text summary. `complexity` feeds the scoring penalty (Minimum-Explanation preference). The principle is stamped `status=hypothesis` — it is NOT knowledge yet.

---

## 12. Induction: counterexample search

**Who**: `[agent]` proposes failure conditions + refutation code, `[framework]` **executes** the code and uses the executor verdict.

**Input**: the hypothesis statement + the alignment.

**Agent output**: `generate_object` → JSON with conditions and refutation code.

**Example output**:
```json
{
 "conditions": [
 "When the resource consumption rates are non-monotone with respect to profit density",
 "When there are multiple binding constraints with different consumption profiles"
 ],
 "refutation_code": "import json\n\nprofits = [100, 80, 60]\nconsumption_a = [10, 2, 1]\nconsumption_b = [1, 3, 10]\ncap_a = 100\ncap_b = 100\n\ndensity_a = [p/c for p, c in zip(profits, consumption_a)]\ndensity_b = [p/c for p, c in zip(profits, consumption_b)]\n\nif density_a != sorted(density_a, reverse=True) or density_b != sorted(density_b, reverse=True):\n result = {'refuted': True, 'reason': 'density ranking conflicts between constraints'}\nelse:\n result = {'refuted': False}\n\nwith open('result.json', 'w') as f:\n json.dump(result, f)"
}
```

**Critical rule**: the LLM proposes conditions and writes code, but **only the `SafePythonExecutor` execution verdict decides** whether the principle breaks. A crashed or unexecuted refutation is NOT a counterexample (anti self-judgment). Confirmed counterexamples shrink `applicability_conditions` and are recorded; they never delete the hypothesis.
