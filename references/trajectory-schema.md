# Trajectory and Result Schemas

## Contents

1. AttemptRecord
2. BranchResult
3. SolveResult
4. Episode (problem-level)
5. Termination and validation

## AttemptRecord

Each attempt is appended to `trajectories/<problem_id>/<branch_id>/attempts.jsonl`. It records identity and timestamps, retrieved IDs grouped by layer, problem/formulation summaries, relative code path and SHA-256, bounded stdout/stderr, normalized error, solver status/objective/bound/gap/runtime, validator report, repair action, validation level, and termination reason.

Long code lives in `runs/<problem_id>/<branch_id>/attempt_N.py`; trajectory records store its path and hash.

## BranchResult

A branch result contains branch/solver/workspace identity, all attempts, final `SolverExecutionResult`, final `ValidationReport`, and termination reason. Branch workspaces are unique and created with `exist_ok=False`.

## SolveResult

The aggregate contains selected branch and reason, every branch result, retrieved/appended/duplicate experience IDs, validation level, warnings, event timeline, objective-comparability flag, and discrepancies. Under `defer_extraction=True` (harness Option A) the returned SolveResult has EMPTY appended/duplicate IDs; appending happens later in `evaluate_with_gold` after the gold verdict.

## Episode (problem-level)

An Episode is the problem-level snapshot of one solve run, stored at `episodes/episodes.jsonl`, distinct from the attempt-level trajectory. Two-phase append-only:

- `record_kind="base"`: written right after solving. Carries `episode_id`, `problem_id`, original `problem`, `normalized_spec` (family, objective, `verified_model`, status, failure_count), `structural_signature`, per-branch `branches` summaries, `final_objective`, and empty `produced_realization_ids`.
- `record_kind="gold_supplement"`: written after `evaluate_with_gold`. Carries `problem_id`, `gold_answer`, `matched`, and `produced_realization_ids`. Links to the base by `problem_id`; neither line is edited.

Episodes are the raw material for offline structural induction and the provenance target of Modeling Bank records (`evidence.source_episodes`).

## Termination and validation

Termination values include `optimal`, `feasible`, `max_attempts`, `repeated_error`, `unchanged_code`, `timeout`, `infeasible`, `unbounded`, `numerical_issue`, `unsupported_solver`, `solver_unavailable`, `license_error`, `validation_failed`, `execution_error`, and `unknown`.

Validation levels progress from `unverified` to `runtime_only`, `solver_feasible`, `semantic_checked`, or `cross_solver_consistent`. Equal objectives alone do not prove semantic correctness. Cross-solver consistency requires at least two isolated feasible branches with comparable objective senses and values inside tolerance.
