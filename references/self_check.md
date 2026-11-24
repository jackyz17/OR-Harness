# Self-check detail and examples

The runtime flow is in [SKILL.md §3b](../SKILL.md#3b-self-check-the-answer-then-decide-to-repair-re-solve-or-commit). This reference expands the judgement behind it: the two layers, the model-versus-task distinction, failed-check handling, H+ verification conditions, and worked examples. The command contract (`check-task`, `--verification`, `hplus_verification`) is in [commands.md §5](commands.md#5-execute-and-check).

## The two layers

### Layer 1 — necessary base check (every task)

Re-read the original statement and inputs and check, where they apply:

- the objective and the quantity it actually optimises;
- the meaning and domain of each decision variable (integer, binary, continuous, sign);
- units — quantity, price, capacity, time — and the billing or cost range;
- completeness of the key constraints, especially coupling or aggregation relations that are easy to misread;
- feasibility of the returned solution against the original constraints;
- agreement of the objective value, constraint residuals and solver status;
- for a task that needs an optimality argument: whether the available bound, gap, dual or other evidence is enough.

These are check dimensions, not a fixed checklist. A genuinely continuous problem needs no integer check, and a task with no time dimension needs no constructed time-constraint check. Base checks must produce real evidence — code review, recomputation of constraints, a short calculation, or the solver's own report — but they do not require a new verification script per item.

### Layer 2 — enhanced verification only when needed

Options include a small enumeration, an independent verification script, a dual or bound argument, a second exact algorithm, a second solver, or another independent check suited to the structure. Use one only when there is a concrete reason, such as: optimality cannot be confirmed from the available evidence; the solution or objective looks anomalous; a constraint interpretation is genuinely ambiguous; the solver's return disagrees with an independent computation; the structure is complex enough that the base check cannot cover a key error; or a specific modelling assumption needs independent validation.

Do NOT run a complete independent re-solve as a default step for every task. This does not forbid independent verification: when a second algorithm is genuinely how a complex problem gets a reliable answer, run it. It forbids the automatic "another full solve per task" habit.

## Verify the model, not only the solver

Verification of the model is not verification of the task:

- a solver's `optimal` proves optimality for the model as written, not that the model matches the statement;
- a second solver solving the same wrong model can return the same wrong answer;
- a dual, enumeration or independent algorithm can prove a mathematical result while sharing a wrong objective or constraint interpretation.

Check the model-to-statement correspondence BEFORE relying on an independent numerical verification. No extra semantic-review tool or world-model call is needed for this.

## H+ verification conditions

When the selected prediction has H+=1, its `verification_conditions` test the predicted knowledge hypothesis. They are not a second task requirement. Reuse the same evidence when it supports both. If the condition cannot be checked within a reasonable scope, record `not_checked`, `unavailable` or the protocol's unknown state; do not run a second complete solve just to make the condition show as complete. The offline review decides whether the evidence is sufficient. See [the H+ review handoff](induction.md#when-to-start-induction) for where the check is declared.

## Failed, insufficient and erroneous checks

- `failed`: a declared check found a violation. The framework does not decide whether the task reading, the model, the code or the check itself is wrong. Analyse which one it is. A bad check, a broken verification script or missing evidence is corrected and re-run on the SAME execution; a new attempt is planned only when the model or implementation is the real problem.
- `insufficient`: the declared check could not decide. It is neither a pass nor a failure, and it does not by itself require a complete re-solve.

Never mark the self-check complete or passed when a verification script raised or an assertion failed without handling it. A failed verification is evidence about the model, the implementation, the task reading or the script; decide which, and act on that decision.

An agent-run verification attached with `--verification` is captured as immutable artifacts and labelled `agent_reported`. It is a different source than the framework's `task_check`, creates no solving attempt, and never overwrites the solve `result.json`. `state` distinguishes `script_error` (the script did not run or raised) from `assertion_violations` (it ran and an assertion failed); neither is a pass.

## Read the evidence

`check.json` declares only the deterministic checks the framework can compute. It is not the whole self-check, and a basis that does not apply to the task must NOT be declared merely for form: a genuinely continuous problem needs no `integer` check, and the framework does not infer that a missing declaration is an uncovered requirement. EX is `result.execution_id` from `execute`.

`report.scope.note` states the checker's fixed scope; `report.checks` and `report.diffs` hold what actually ran and the numeric differences; `report.scope.unchecked` names only declared checks that could not run. `reflection_material` gives the task text/version reference, the candidate and performed method, the captured artifacts and solution (with references), the check report, any agent-run verification, and prior attempts as references.

## Examples

The examples show the judgment the framework leaves to the agent: the tool gives numbers and references, not the decision.

### B018 — a consistent answer with the wrong unit

The task prices each chair at 50 and each order at 15 chairs, so one order in the objective is 15 × 50 = 750. Suppose the script used 50 per order and the solver reported a matching optimum. Recomputing the objective with the same wrong coefficient reproduces the same number: agreement proves the model is solved consistently, NOT that its unit is right. The self-check is to read the unit from the task, compute one order by hand (750), and compare it with the coefficient in the code. The framework's `recompute_objective` uses the coefficients the agent declares, so declaring the code's coefficient would just repeat the bug; either the model is corrected through a new `plan-next`, or the task's coefficient is declared and the difference is reported.

### 0087 — a claimed optimum the evidence rejects

A continuous LP was reported optimal at 20, while a grid search found feasible points near 17.725 and a dual bound near 17.714. The size of that difference is enough to review the corner calculation: with the task as stated (continuous, no integrality requirement), 17.714 has an optimality argument because the bound is achievable. Do not add an integer constraint because the reference value happens to be integral — the task is continuous, so the finding is a calculation/verification problem, not a missing domain. Record the bound or derivation as the optimality evidence; the agent decides that, not the tool.

### 0312 — two solvers agree on the same wrong model

A task asks to minimise total cost, but the script maximises revenue and then negates the number. A second solver reproduces the same value. The agreement proves the model is solved consistently; it proves nothing about the objective semantics. The base check is to read the objective from the task (what is minimised or maximised, in which unit) and compare it with the code before trusting any independent verification. If the objective is wrong, plan a new candidate; a more thorough verification of the same wrong model cannot fix it.

### A failed verification script is not a pass

If a verification script raises or an assertion fails, the self-check is not done. Decide whether the model, the implementation, the task reading or the verification script is wrong. A broken verification script or a bad check declaration is fixed and re-run on the SAME execution with `check-task`; it does not create a new solving attempt and it cannot be recorded as a pass.
