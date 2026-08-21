# Solver Adapters

## Contents

1. Interface
2. Availability
3. Result contract
4. Execution controls
5. Solver differences

## Interface

`SolverAdapter` exposes `name`, `solver_family`, `api`, `is_available`, `validate_environment`, `build_generation_context`, `execute`, `normalize_feedback`, and `parse_result`. `SolverRegistry` owns adapter discovery and skips unavailable adapters without failing other branches.

## Availability

Seven solver branches are registered by default:

| Branch | Package | Family | API | Notes |
|---|---|---|---|---|
| `gurobi` | `gurobipy` | milp | gurobipy | checks module + silent env to distinguish missing license |
| `scip` | `pyscipopt` | milp | pyscipopt | module check |
| `highs` | `highspy` | milp | highspy | module check (open source) |
| `copt` | `coptpy` | milp | coptpy | module check |
| `ortools` | `ortools` | cp_sat | ortools.cp_model | CP-SAT; requires integer coefficients |
| `pulp` | `pulp` | milp | pulp | modeling framework branch, default CBC backend |
| `pyomo` | `pyomo` | milp | pyomo | modeling framework branch; requires a backend solver (highspy/gurobipy/pyscipopt/coptpy/pulp) |

- Mock is registered only by explicit test/demo injection.
- Missing modules map to `solver_unavailable`; Gurobi license failures map to `license_error`. Neither becomes a modeling experience.
- `pulp` and `pyomo` are modeling-framework branches (per project Option 1): they test code generation quality for that API. Their underlying engine may overlap with another branch, which is intended — the comparison is at the modeling-API level.

## Result contract

Generated code must write `result.json` in its branch directory:

```json
{
  "status": "optimal|feasible|infeasible|unbounded|timeout|error|unknown",
  "solver": "...",
  "objective_sense": "minimize|maximize|feasibility|unknown",
  "objective_value": null,
  "objective_bound": null,
  "mip_gap": null,
  "runtime_seconds": null,
  "variables": {},
  "diagnostics": {},
  "message": ""
}
```

An exit without this file is an execution failure even if stdout claims success.

## Execution controls

Execution uses `sys.executable`, no shell, branch-local cwd, wall timeout, solver timeout environment hint, environment allowlist, bounded/redacted output, and Unix CPU/address-space/file-size limits. API keys and full environment variables are excluded. This process wrapper cannot guarantee network isolation on every OS; use a container or host sandbox for production isolation.

## Solver differences

Gurobi and SCIP are MILP-oriented and accept continuous coefficients. CP-SAT requires integer variables and coefficients, so scale rational data deliberately and document the scale. Do not send a solver-specific record to another solver merely because its natural-language text is similar.

### MISOCP: SOC constraint + Big-M linking (critical formulation pattern)

When combining a second-order cone constraint (`dist² ≥ ΣΔ²`) with a Big-M edge indicator (`dist ≤ M·e`), **never use a single variable for both**. A single `dist` variable with SOC (always active) and Big-M (`dist ≤ M·e`) causes **infeasibility**: when `e=0`, Big-M forces `dist=0` but SOC requires `dist ≥ ‖Δ‖ > 0` (unless the Steiner point coincides with the terminal, which is not guaranteed).

**Correct two-variable formulation:**
- `dist[i,j] ≥ 0` — Euclidean distance, SOC always active, upper bound `M` (max diagonal).
- `w[i,j] ≥ 0` — objective contribution, Big-M linked: `w ≤ M·e` and `w ≥ dist − M·(1−e)`.
- Objective: `min Σ w`.
- When `e=1`: `w = dist = ‖Δ‖` (minimized). When `e=0`: `w = 0`, `dist` unconstrained in objective.

Big-M = √(dimensions) for unit hypercube. K = N−2 Steiner points from the handshake lemma (each Steiner degree 3, each terminal degree 1 → edges = 2N−3 = N+K−1).

### Solver-specific API notes

**HiGHS (highspy):**
- Import: `from highspy import Highs, HighsStatus, HighsModelStatus`.
- Create model: `h = Highs()`, then `h.addCols(...)`, `h.addRows(...)`.
- **Status check**: use `h.getModelStatus()` (NOT `h.status`). Returns `HighsModelStatus.kOptimal`, `HighsModelStatus.kInfeasible`, `HighsModelStatus.kUnbounded`.
- `h.getInfinity()` for infinity value.
- **Row count**: `h.getNumRow()` (singular, NOT `getNumRows`).
- **Column cost**: `h.changeColsCost(n, np_int32_array, np_float64_array)` — requires numpy arrays, not plain lists. For older API, `h.changeColsCost(range(n), list)` may work.
- Objective value: `h.getObjectiveValue()` (not `h.objval`).
- Variable values: `h.getSolution().col_value`.
- Set time limit: `h.setOptionValue("time_limit", 60.0)`.
- MIP gap: `h.getDoubleInfoValue("mip_gap")`.
- **Version note**: API changed significantly between versions. Test with your installed `highspy` version.

**PuLP (pulp):**
- Status: `pulp.LpStatus[problem.status]` returns **capitalized** string like `"Optimal"`. **Always `.lower()` the status before writing to result.json**: `status = pulp.LpStatus[problem.status].lower()`.
- Variable values: `var.varValue` (not `var.value()`).
- Sense: `pulp.LpMinimize` or `pulp.LpMaximize`.
- Default backend is CBC (open source).

**COPT (coptpy):**
- MIP gap parameter: `cp.COPT.Param.RelGap` (NOT `MIPGap`).
- `model.status` is an **integer**: 1=OPTIMAL, 8=TIMEOUT, 2=INFEASIBLE, 5=UNBOUNDED.
- `model.objval` works directly (no `hasprimal` attribute).
- Quadratic constraints: `model.addQConstr(lhs, sense, rhs, name)`. Use `var * var` for products (NOT `addTerm`).
- COPT struggles to close MIP gaps on MIQCP/MISOCP problems (observed 49% gap after 120s on a 33-variable instance).

**SCIP (pyscipopt):**
- Import only `Model` — `Term` does not exist in newer versions.
- `model.getStatus()` returns lowercase string: `'optimal'`, `'timelimit'`, `'infeasible'`.
- Quadratic constraints via `model.addCons(lhs >= 0)` where `lhs = var*var - sum(v*v)` works natively.

**Gurobi (gurobipy):**
- `model.addQConstr(dist*dist >= gp.quicksum(v*v for v in diffs))` is recognized as SOC automatically.
- Status constants: `GRB.OPTIMAL=2`, `GRB.TIME_LIMIT=9`, `GRB.INFEASIBLE=3`.
- Restricted (free) license: 2000-variable limit, supports SOC.

**OR-Tools (ortools.cp_model):**
- CP-SAT requires **integer** variables and coefficients. Scale rational data deliberately.
- `model.ModelStatus` → use `cp_model.OPTIMAL`, `cp_model.FEASIBLE`, `cp_model.INFEASIBLE`.
- `solver.ObjectiveValue()` returns the objective (scaled).

**Pyomo (pyomo):**
- Requires a backend solver: `opt = pyomo.SolverFactory("highs")` (or "gurobi", "scip", "cbc").
- `results = opt.solve(model)`.
- Status is in `results.solver.termination_condition`.
