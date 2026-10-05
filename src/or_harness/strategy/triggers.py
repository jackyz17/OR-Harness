"""Solver failure facts: classification and environment advisories.

The framework does NOT decide what a recorded execution MEANS. It no longer
manufactures induction labels from online pattern detectors
(``strategy_contrast`` / ``intervention_recovery`` /
``structural_reproduction`` / ``advantage_reversal``) — the outer agent reads
the recorded material (``orx review-material``) and abstracts the method
itself. What remains here is purely FACTUAL and recomputable:

- :func:`classify_failure` — is a failed run an ENVIRONMENT problem (the
  solver/stack cannot run here) or a MODEL problem (the harness's own code)?
  This is a mechanical split of an error string, not an induction judgment.
- :func:`solver_advisories` — a live aggregation of environment-class
  failures per solver, so the harness can pick a solver informed by its own
  history. Like conditional statistics, this is arithmetic over facts, not
  knowledge, and it is never persisted.

A memo on the removed detectors: they used to fire after every ``record`` and
persist ``induction_hints`` onto the fact. Those keys may still sit on old
records; nothing reads them any more — the hints were a framework-side
interpretation, and interpretation is the agent's job.
"""

from __future__ import annotations

from typing import Any, Dict, List


def classify_failure(record: Any) -> str:
    """Classify a failed execution's error.

    ``environment`` — the solver/stack cannot run here (sandbox security
    policy, missing module, import error). These feed solver advisories.
    ``model`` — the harness's own code failed (traceback). These do not.
    """
    error = _first_error(record) or ""
    lowered = error.lower()
    if ("security policy" in lowered or "importerror" in lowered
            or "modulenotfounderror" in lowered
            or "no module named" in lowered):
        return "environment"
    return "model"


def _first_error(record: Any) -> str:
    failures = getattr(record, "failures", None)
    if failures:
        return failures[0].error
    quality = getattr(record, "quality", None) or {}
    return str(quality.get("status", ""))


def solver_advisories(bank) -> List[Dict[str, Any]]:
    """On-the-fly environment-level solver failure view (never persisted).

    Aggregates environment-class failures per solver from the Experience
    Bank — "pulp failed once in this environment (security_policy)" — so the
    harness picks solvers informed by its own history. Model-class failures
    (the harness's own code bugs) are excluded: they say nothing about the
    solver. Like conditional statistics, this is arithmetic over facts, not
    knowledge; compaction naturally retires it."""
    per_solver: Dict[str, Dict[str, Any]] = {}
    for rec in bank.all():
        if rec.source != "executed" or rec.quality.get("feasible", False):
            continue
        solver = str((rec.solver or {}).get("name", ""))
        if not solver:
            continue
        error_class = classify_failure(rec)
        if error_class != "environment":
            continue
        entry = per_solver.setdefault(
            solver, {"solver": solver, "environment_failures": 0,
                     "error_classes": [], "last_execution_id": None,
                     "last_error": None})
        entry["environment_failures"] += 1
        if error_class not in entry["error_classes"]:
            entry["error_classes"].append(error_class)
        entry["last_execution_id"] = rec.execution_id
        entry["last_error"] = (_first_error(rec) or "")[:200]
    return sorted(per_solver.values(), key=lambda e: e["solver"])
