# Worked Examples

Three end-to-end examples showing expected behavior at the boundaries.
Read this when you are unsure how to handle a specific situation type -
each example shows the exact tool sequence and the reasoning at each step.

**Note:** The `[THINK]`/`[MODEL]` markers below are literal square-bracket markers (plain text). Angle-bracket `<think>` tags are also accepted by the parser, but square brackets are preferred — some harnesses strip `<think>` before it reaches the file.

## Example 1: Direct application (normal solve, gold match)

**Input:**
> "A factory produces two products. Product A yields \$3 profit per unit and
> requires 2 hours of machine time; product B yields \$4 per unit and requires
> 3 hours. 100 machine hours are available. Maximize profit. Gold answer: 150."

**Expected behavior:**

```
0. orx doctor -> solvers_available: [gurobi, highs, ortools, pulp]
   (pick THREE from DIFFERENT families for stronger cross-validation; this run
   picks highs + ortools + pulp — two milp solvers plus a cp_sat solver. Rotate
   the trio across runs: the bank only accumulates API knowledge for solvers
   you actually use.)

1. orx recall --problem-file problem.txt
   -> priors_count=1, [E1] "shared-resource capacity: single linear row"

2. [THINK]
   Objective: maximize 3x_A + 4x_B. Decisions: continuous production quantities.
   Machine time is a shared scarce capacity - same structure as [E1].
   [uses E1]
   [/THINK]
   [MODEL]
   SETS
     p in Products = {A, B}
   PARAMETERS
     profit[p]
     machine_hours[p]
     capacity
   VARIABLES
     x[p] >= 0, continuous
   OBJECTIVE
     maximize sum_p profit[p] * x[p]
   CONSTRAINTS
     C1: sum_p machine_hours[p] * x[p] <= capacity
   [/MODEL]

3. orx validate -> passed=true, stamps/model.json
4. orx signature: {objective: "linear", decision: ["continuous_flow"],
   constraint: ["capacity"], interaction: "shared_resource_coupled"} -> passed
5. orx solve --solver highs,ortools,pulp (all branches run CONCURRENTLY):
   highs first attempt hits AttributeError on HighsInfo.run_time
   → read repair_hints in branches/highs/result.json → fix code (use
   h.getRunTime()) → re-run orx solve --solver highs (single-branch retry):
   optimal, 150.0; ortools: optimal, 150.0; pulp: optimal, 150.0
   (NOTE: the solver trio here is an EXAMPLE, not a recommendation — pick
   from what `orx doctor` reports as available, prefer different families,
   and rotate across runs)
6. orx cross-validate -> consistent=true, best_objective=150.0
7. gold 150.0 == 150.0 -> orx gold --answer 150 -> matched
8. orx append ×3 (one file per bankable lesson, BEFORE orx episode):
   a. (layer=modeling, modeling_aspect=constraint) "shared linear capacity row
      covers multi-product machine time"
   b. (layer=implementation) "HiGHS HighsInfo has no run_time attribute — use
      h.getRunTime() for actual wall-clock"
   c. (layer=repair) "AttributeError: 'HighsInfo' has no attribute 'run_time'
      → replace with h.getRunTime() (error in result.json construction, not
      solver logic)"
   (No solving lesson — no timeout/gap/numerical issues occurred.)
9. orx episode
    -> utility_credited=1 (E1 was cited and helped)
    -> induction_check: {"should_induce": false,
                         "reason": "no heterogeneous isomorphic cluster (candidate gate)",
                         "instruction": "Induction not due yet... No action needed; keep solving."}
   (READ this field every time. When it flips to should_induce: true — after
   enough cross-family realizations accumulate — run the Offline Induction
   Workflow BEFORE starting the next solve. See SKILL.md.)
```

**Report to user:** objective 150.0, all three solvers agree, model verified,
3 experiences appended (modeling + implementation + repair), prior [E1] credited,
induction not yet due.

## Example 2: Ambiguous input (missing information)

**Input:**
> "There are 50 workers to assign to shifts. Minimize cost."

**Expected behavior:**
- Identify what is missing: cost structure (per-worker? per-shift? overtime?),
  shift count and length, per-worker availability, hard vs soft constraints.
- **State the assumptions explicitly** or ask the user - do NOT invent hidden data.
- If proceeding with stated assumptions, note them in the `[THINK]` block so the model
  is auditable.

**Do NOT:**
- Silently assume uniform costs.
- Build a model from imagined shift structures without declaring the assumption.
- Proceed to `orx validate` with a model you cannot justify line-by-line
  from the problem text plus stated assumptions.

**Gold answer missing:** if the user does not provide a gold answer, you have two options:
1. **Ask the user** for the gold answer before proceeding past `orx cross-validate`.
2. If the user confirms no gold is available, proceed with consistency-only
   validation: `orx gold` (no --answer) records consistency-only, then
   `orx append` + `orx episode` — but explicitly tell the user that
   cross-solver consistency does NOT prove correctness.

**Do NOT** use your own solver output as the gold answer. Gold comes ONLY from
the user or the problem statement.

## Example 3: Negative example (modeling direction wrong -> reflection)

**Input:**
> "A courier must visit 5 cities and return home. Minimize total distance.
> Gold answer: 230."

**First attempt (WRONG direction):**
```
Model: assignment-style x[i,j] binary with degree constraints only.
orx solve --solver highs,ortools -> both solvers: optimal, 185.0
orx cross-validate -> consistent=true (both agree - but both are WRONG)
orx gold --answer 230 -> gold_matched=false
```

**What went wrong:** the model has no subtour elimination. Both solvers
"agree" on 185.0 because they solved the same wrong relaxation. Cross-solver
consistency does NOT prove correctness - it only proves the implementations
match.

**Recovery (reflection loop):**
```
1. Analyze WHY the direction was wrong (formulation, not code):
   "Degree constraints alone permit subtours. 185 < 230 because disjoint
   cycles are cheaper than one Hamiltonian cycle."
2. orx new-round (archives the failed round's artifacts)
3. New model: add subtour elimination (MTZ or exponential lazy constraints).
4. orx validate -> orx hints --solver highs,ortools
   -> write both branches' solve.py -> orx solve --solver highs,ortools
   -> orx cross-validate
5. Both solvers: 230.0 -> orx gold --answer 230 -> matched
6. orx append (the contrast lesson):
   title: "Degree constraints alone under-count TSP cost (subtours)"
   polarity: "negative"
   diagnosis: "assignment relaxation permits disjoint cycles"
   action: "add subtour elimination (MTZ) when routing requires a single tour"
7. orx episode -> read induction_check (should_induce?)
```

**Lesson:** cross-solver consistency catches implementation bugs, not
formulation bugs. Gold mismatch + consistent solvers almost always means
the MODEL is wrong, not the code.

## Do-not-do-this summary

| ❌ Do not | Because |
|---|---|
| Adjust the answer to match gold without re-deriving the model | Matching-by-fiat produces no transferable knowledge and fails on any variation |
| Infer an exact plan from the gold number alone | The gold validates a model; it is not a substitute for one |
| Treat cross-solver consistency as proof of correctness | Consistent solvers can agree on the same wrong relaxation (Example 3) |
| Cite a prior you did not apply | Citation is a utility credit; false credits corrupt the ranking |
| Skip reflection after a gold mismatch and just re-run the same model | The same model will produce the same wrong answer |
| Only append a modeling experience, ignoring API gotchas and error→fix lessons | The Implementation and Repair banks exist to capture solver-specific knowledge. If you hit and fixed an API issue during `orx solve`, write it via `orx append` (layer="implementation"|"repair") BEFORE `orx episode`. |
| Invent `[uses E7]` when only [E1]-[E3] were injected | Unknown tags are dropped; the citation is silently lost |
