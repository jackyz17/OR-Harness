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

Options include a small enumeration, an independent verification script, a dual or bound argument, a second exact algorithm, a second solver, or another independent check suited to the structure. Add one only when there is a concrete reason, such as: optimality cannot be confirmed from the available evidence; the solution or objective looks anomalous; a constraint interpretation is genuinely ambiguous; the solver's return disagrees with an independent computation; the structure is complex enough that the base check cannot cover a key error; or a specific modelling assumption needs independent validation.

Verification is sufficiency-driven, not count-limited: pick the cheapest check for a named unresolved doubt; address independent doubts separately and reuse established results. Stop when sufficient evidence supports the conclusion. A complete independent re-solve is not the default; use it only when it reasonably resolves a specific open doubt. Neither H+=1 nor a missing G label requires another solve. Prefer an already sufficient analytic, dual or other certificate over repeating a complete solve. Equal optimal objectives may have different optimal solution points; check feasibility and the claimed guarantee rather than forcing identical vectors.

Validation follows the bound formal solve. Record important independent scripts
and results with `check-task --verification` on that execution, plus their
actual cost in the existing verify action/host usage report. A new method or a
substantially revised answer requires a fresh plan-next and execute. LP
relaxations, reduction searches, solver comparisons and real-instance code
trials that produce the original answer are part of the formal strategy, even
when an intermediate calculation is called a check. The Runner audits host
terminal activity; the framework does not monitor that terminal.

## Verify the model, not only the solver

Verification of the model is not verification of the task:

- a solver's `optimal` proves optimality for the model as written, not that the model matches the statement;
- a second solver solving the same wrong model can return the same wrong answer;
- a dual, enumeration or independent algorithm can prove a mathematical result while sharing a wrong objective or constraint interpretation.

Check the model-to-statement correspondence BEFORE relying on an independent numerical verification. No extra semantic-review tool or world-model call is needed for this.

## H+ verification conditions

When the selected prediction has H+=1, its `verification_conditions` test the predicted knowledge hypothesis. They are not a second task requirement. Reuse the same evidence when it supports both. If the condition cannot be checked within a reasonable scope, record `not_checked`, `unavailable` or the protocol's unknown state; never run a large task's complete re-solve merely to make the condition show as complete. Proportionate evidence is expected: an affordable small-scale full enumeration, an original-versus-reduced-method comparison, or a targeted reproduction is allowed when it verifies the key mechanism. The offline review decides whether the evidence is sufficient. See [the H+ review handoff](induction.md#when-to-start-induction) for where the check is declared.

## Failed, insufficient and erroneous checks

- `failed`: a declared check found a violation. The framework does not decide whether the task reading, the model, the code or the check itself is wrong. Analyse which one it is. A bad check, a broken verification script or missing evidence is corrected and re-run on the SAME execution; a new attempt is planned only when the model or implementation is the real problem.
- `insufficient`: the declared check could not decide. It is neither a pass nor a failure, and it does not by itself require a complete re-solve.

G separately measures effective completion and verification by the current
Agent. With faithful execution and an evaluable result, substantive coverage
gaps, circular evidence or Agent-supplied insufficient checks yield G=0; locating
the exact failure cause is not required. Record that cause as unknown when
needed. Missing key external evidence or a defective task/reference standard
remains Unknown; declare the distinction through `g_attribution.evidence_status`.
Verification sufficiency depends on the original requirements and evidence,
not the number of scripts, checks or solvers. Keep the minimum sufficient check.

Never mark the self-check complete or passed when a verification script raised or an assertion failed without handling it. A failed verification is evidence about the model, the implementation, the task reading or the script; decide which, and act on that decision.

An agent-run verification attached with `--verification` is captured as immutable artifacts and labelled `agent_reported`. It is a different source than the framework's `task_check`, creates no solving attempt, and never overwrites the solve `result.json`. `state` distinguishes `script_error` (the script did not run or raised) from `assertion_violations` (it ran and an assertion failed); neither is a pass.

## Read the evidence

Use the task, captured method/code/result and returned reflection material together. Inspect only a missing detail that can change the conclusion, and batch related reads; do not reconstruct the same evidence through many field-by-field calls. Single-task scratch work is allowed, but formal experiments must not persist extra solving notes as cross-task memory outside the Harness banks.

`check.json` declares computable bases and optional G/H+ evidence, not the whole self-check. A basis that does not apply must NOT be declared merely for form: a genuinely continuous task needs no `integer` check. Complete necessary validation, then submit the final check with supported G attribution or honest Unknown before record and closeout. Repeating a completed check on unchanged evidence creates a new revision without improving support. EX is `result.execution_id` from `execute`.

`report.scope.note` states the checker's fixed scope; `report.checks` and `report.diffs` hold what actually ran and the numeric differences; `report.scope.unchecked` names only declared checks that could not run. `reflection_material` gives the task text/version reference, the candidate and performed method, the captured artifacts and solution (with references), the check report, any agent-run verification, and prior attempts as references.

## Examples

The examples show the judgment the framework leaves to the agent: the tool gives numbers and references, not the decision. They use the original task, the recorded execution and standard mathematics; a reference answer or checker result is used only AFTER commitment as offline evidence, never as a modelling premise.

### B018 — a consistent answer with the wrong unit

The task prices each chair at 50 and each order at 15 chairs, so one order in the objective is 15 × 50 = 750. Suppose the script used 50 per order and the solver reported a matching optimum. Recomputing the objective with the same wrong coefficient reproduces the same number: agreement proves the model is solved consistently, NOT that its unit is right. The self-check is to read the unit from the task, compute one order by hand (750), and compare it with the coefficient in the code. The framework's `recompute_objective` uses the coefficients the agent declares, so declaring the code's coefficient would just repeat the bug; either the model is corrected through a new `plan-next`, or the task's coefficient is declared and the difference is reported.

### 0087 — a claimed optimum the evidence rejects

A continuous LP was reported optimal at 20, while a grid search found feasible points near 17.725 and a dual bound near 17.714. The size of that difference is enough to review the corner calculation: the task states continuous variables (no integrality requirement), and 17.714 has an optimality argument because the bound is achievable. Judge the variable domain from the task statement, not from any external answer: do not add an integer constraint because some outside value happens to be integral — that would change the task. Record the bound or derivation as the optimality evidence; the agent decides that, not the tool.

### 0312 — two solvers agree on the same wrong model

A task asks to minimise total cost, but the script maximises revenue and then negates the number. A second solver reproduces the same value. The agreement proves the model is solved consistently; it proves nothing about the objective semantics. The base check is to read the objective from the task (what is minimised or maximised, in which unit) and compare it with the code before trusting any independent verification. If the objective is wrong, plan a new candidate; a more thorough verification of the same wrong model cannot fix it.

### A failed verification script is not a pass

If a verification script raises or an assertion fails, the self-check is not done. Decide whether the model, the implementation, the task reading or the verification script is wrong. A broken verification script or a bad check declaration is fixed and re-run on the SAME execution with `check-task`; it does not create a new solving attempt and it cannot be recorded as a pass.

## CLI check contract

Check original task requirements against a staged or recorded execution. `check-task` writes a `verify` action and `execution_features.task_check`. It never runs the solver again, so a corrected check or a late check on the same execution does not create a solving attempt or consume a retry.

Normal order is execute → necessary self-check/verification → final `check-task` with supported `g_attribution` (or honest Unknown) → `record` → `close-episode`. Each call makes a new check revision. A later check without attribution does not inherit the old assessment, even if its verdict is the same. Avoid rechecking unchanged evidence; a necessary changed check must carry its own current attribution or remain Unknown.

**Before answering** The self-check is the agent's, not the tool's: check the model against the original task, then check the result against both. A solver's `optimal` proves only that the written model was solved. Use the base check on every task; add targeted enhanced checks only for concrete reasons, continue while an unresolved doubt can still change the conclusion, and stop when the evidence is sufficient. Do not default to a complete independent re-solve, and check the model-to-statement correspondence before trusting a second algorithm or solver. See [self-check detail](self_check.md) for the two layers, failure handling and examples. `--check` takes a JSON object or JSON file containing the computable bases and optional G/H+ evidence annotations; it is not the whole self-check.

| Check key | Use |
|---|---|
| `reference_objective` | Compare the reported objective with a value you source (`reference_source`, `reference_version`, optional `tolerance`) |
| `reference_status` | Check the reported solver status |
| `integer` | Check named `variables` or the recorded variables, with optional tolerance (default `1e-6`). Declare it only if the TASK requires integer values; the framework never infers integrality, and a continuous task is not "missing" this check |
| `recompute_objective` | Compute the objective from the variables with coefficients the agent declares |
| `semantic_probe` | Apply `{path, equals\|min\|max\|in}` probes to recorded fields |
| `intent` | Mark a `relaxation` or `intermediate` result |
| `g_attribution` | Optional sourced G assessment; saved separately from the generic acceptance rule and checker verdict |

A useful deterministic check is not a proof that the model represents the task: `recompute_objective` with the coefficient you already used in the code can reproduce a unit error (see [the B018 example](self_check.md#b018--a-consistent-answer-with-the-wrong-unit)). The declared bases must come from the task's units and constraints.

Read:

- `result.state` — `passed`: the declared checks held on the recorded values; `failed`: a declared check found a violation; `insufficient`: the check could not decide. None of them decides the next action. A pass does not block a new `plan-next`; a failure does not prove the model is wrong, and an `insufficient` result is neither a pass nor a reason to force a full re-solve. On a failure, analyse whether the task reading, the model, the implementation or the check itself is wrong, then act on that finding.
- `report.checks[]` — what the framework actually executed, with observed values, tolerance and `ok`.
- `report.diffs[]` — the numeric differences for a failure.
- `report.scope.basis` — the declared bases evaluated; `report.scope.unchecked` — only declared checks that could not run, with the reason; `report.scope.note` — the fixed scope statement. The framework does not scan the task text for uncovered constraints, so an empty `unchecked` is not a claim that nothing else needed checking.
- `reflection_material` — the self-check material: `task_text`/`task_ref`, the candidate and performed method, `artifacts` references (script, result.json, error clips), the captured solution with `solution_ref`, the check report, `verification_evidence`, and prior attempts as one-line references. Read it before deciding, and prefer the references over re-reading the whole attempt chain.
- `result.g_assessment_ref` — the saved G assessment, also referenced by the verify action's `outcome.g_assessment_ref` and the Evaluation's `benefit.attribution_ref`. A generic check without G attribution has no such reference. Invalid declarations return an actionable error before writing; valid Unknown attribution is accepted.

### Independent verification payload

**Independent verification** When the agent runs its own verification (a separate script, an enumeration, a hand computation), attach the evidence to this same execution:

```json
{
  "script": "ws/verify.py",
  "output": "ws/verify.out",
  "result": {"ran": true, "exit_code": 0,
             "assertions": [
               {"name": "B018 order coefficient equals 15*50",
                "ok": false, "observed": 750, "expected": 50,
                "note": "code used the per-chair price per order"}]}
}
```

`--verification` accepts that JSON or a file path; `result_path` can name a JSON file instead of inline `result`. The script, output and result are captured as immutable artifacts and the normalised block is labelled `source=agent_reported` under `execution_features.verification_evidence` — a different source than the framework's `task_check`, and a separate file from the solve `result.json` (which is never overwritten). `state` distinguishes a `script_error` (the script did not run / raised) from `assertion_violations` (the script ran and an assertion was not satisfied); `assertions_*` and `evidence_only` cover the other cases. A satisfied assertion is the agent's claim, not a framework verdict, and creates no solving attempt. A `script_error` or `assertion_violations` result is NOT a pass: the self-check is not done until the cause is handled.

When the selected prediction has H+=1, the same `--check` JSON may include `hplus_verification` items for its frozen verification conditions (see [the H+ review handoff](induction.md#when-to-start-induction)). They check the predicted knowledge hypothesis, not an extra task requirement. Reuse self-check evidence where it applies; if the condition cannot be checked, record `not_checked`/`unavailable`. A complete re-solve is not the default, but an affordable check that actually verifies the mechanism is allowed — never one run merely to make the state show as checked.

A changed fact or incorrect check can be corrected by re-running `check-task` on the same execution; a real model or implementation change requires a fresh `plan-next` selection and a new execution. Submit a sourced G assessment once through this same interface. Unknown does not require repeated checks or extra solving to force a numeric label. See [G attribution evidence](episode_closeout.md#g-attribution-evidence), including unresolved reference conflicts.

**After answering** Benchmark evaluators compare the locked answer with reference evidence outside the retry loop; reference comparisons are for evaluation and offline analysis, never a reason to return to solving. For offline checks use `reference_objective` or `reference_status` with `reference_source` (`bench_declared\|independent\|self_derived`) and an available `reference_version`. The default objective tolerance is `1e-6 * max(1, abs(reference))`. Omitted source is treated as `self_derived`. See [offline use of reference evidence](induction.md#read-material-and-choose-comparisons).
