"""Solver failure facts: classification and environment advisories.

The framework does NOT decide what a recorded execution MEANS. It no longer
manufactures induction labels from online pattern detectors
(``strategy_contrast`` / ``intervention_recovery`` /
``structural_reproduction`` / ``advantage_reversal``) — the outer agent reads
the recorded material (``orx induction-material``) and forms the strategy
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

import re
from typing import Any, Dict, List


def classify_failure(record: Any) -> str:
    """Classify a failed execution's error.

    Three outcomes, and the third is the honest one:

    - ``environment`` — the solver/stack cannot run here (sandbox security
      policy, missing module, unavailable backend or licence, a blocked
      network/API). These feed solver advisories and the
      ``environment_failure`` risk event.
    - ``model`` — the harness's OWN code failed with a Python-level error
      (a traceback / a typical exception type). Recorded as
      ``implementation_failure``. This is an IMPLEMENTATION problem, never
      a statement that the mathematical MODEL was wrong.
    - ``unknown`` — the error does not match a reliable pattern. It is
      NEVER forced into ``model``: a stranger error is not evidence about
      the harness's code, and mislabelling it would corrupt the risk
      calibration. Unknown stays unknown.

    This is a mechanical split of an error string, not an induction
    judgment, and it adds no API failure tree — just the patterns that can
    be told apart reliably.
    """
    error = _first_error(record) or ""
    lowered = error.lower()
    if any(marker in lowered for marker in _ENVIRONMENT_MARKERS):
        return "environment"
    if any(marker in lowered for marker in _IMPLEMENTATION_MARKERS):
        return "model"
    # A Python exception TYPE name (``FooError`` / ``FooException``, in the
    # original casing) is a reliable structural signal that the harness's
    # OWN code raised — the error string begins with the class name the
    # interpreter printed. It is checked AFTER the environment markers so a
    # ``ModuleNotFoundError`` still classifies as environment.
    if _PYTHON_EXCEPTION_RE.search(error):
        return "model"
    return "unknown"


#: Matches a Python exception class name as it appears at the start of an
#: error string (``FileExistsError: [Errno 17] ...``).
_PYTHON_EXCEPTION_RE = re.compile(r"\b[A-Z][A-Za-z]*"
                                  r"(?:Error|Exception|Warning)\b")


#: Substrings that reliably indicate the ENVIRONMENT could not run the
#: solver/script (as opposed to the script itself being wrong).
_ENVIRONMENT_MARKERS = (
    "security policy",
    "importerror",
    "modulenotfounderror",
    "no module named",
    "not installed",
    "is not available",
    "not available in this environment",
    "unsupported solver",
    "solver not supported",
    "unrecognized solver",
    "no solver",
    "could not load",
    "cannot load",
    "failed to load",
    "license",
    "not licensed",
    "permission denied",
    "connection refused",
    "connection error",
    "network",
    "timed out connecting",
    "api key",
    "authentication",
    "read-only file system",
)

#: Substrings that reliably indicate the harness's OWN code failed — an
#: implementation failure, NOT a modelling verdict.
_IMPLEMENTATION_MARKERS = (
    "traceback",
    "typeerror",
    "valueerror",
    "nameerror",
    "keyerror",
    "attributeerror",
    "indexerror",
    "zerodivisionerror",
    "syntaxerror",
    "indentationerror",
    "assertionerror",
    "runtimeerror",
    "shape",  # numpy shape mismatch, a common coding bug
    "not subscriptable",
    "not iterable",
    "object has no attribute",
)


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
