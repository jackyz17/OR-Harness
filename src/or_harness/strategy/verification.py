"""Offline admission verification: does the candidate claim actually hold?

The framework's job here is narrow and mechanical. The harness (outer agent)
proposes the claim, writes the verification tasks/programs and states which
check applies; THIS module runs the checks against actual executions and
decides:

  verified              — the declared check passed on real execution evidence
  insufficient_evidence — the check ran but cannot decide (missing evidence,
                          no comparison basis, or the execution itself failed)
  refuted               — the check ran on real evidence and did NOT hold

Two red lines:

1. **A program's own verdict is not a verification.** "It printed
   ``{"principle_failed": false}``" proves nothing about the claim — the
   program merely executed. Verdicts are computed from framework-side,
   checkable facts (statuses, objective values, measured costs, explicit
   tolerances).
2. **A failed execution is not a refutation.** "We could not check it" and
   "we checked it and it failed" are different results, and only the second
   refutes.

Purpose tags (``rule`` / ``repair`` / ``cost_saving``) select the check; they
are NOT an enumeration of what strategic knowledge may ever be, and the tags
are not validated as a closed set.

Every report carries the audit trail the harness needs to read back: which
claim, which executions, which check, and why that conclusion followed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

#: Purpose tags (open vocabulary — a hint, not a closed set).
PURPOSE_RULE = "rule"
PURPOSE_REPAIR = "repair"
PURPOSE_COST_SAVING = "cost_saving"

#: Objective values must agree within this relative tolerance for a
#: "results are close enough to compare cost" judgment.
RESULT_SIMILARITY_TOLERANCE = 0.02

#: Verification outcomes.
VERIFIED = "verified"
INSUFFICIENT = "insufficient_evidence"
REFUTED = "refuted"


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _report(state: str, *, purpose: Optional[str], claim: str,
            checks: List[Dict[str, Any]], evidence: List[str],
            conclusion: str) -> Dict[str, Any]:
    return {
        "state": state,
        "purpose": purpose,
        "claim": claim,
        "checks": checks,
        "evidence": evidence,
        "conclusion": conclusion,
        "verified_at": time.time(),
    }


def verify_candidate(purpose: Optional[str], claim: str,
                     *, check: Optional[Dict[str, Any]] = None,
                     executions: Sequence[Any] = (),
                     supporting: Sequence[Any] = ()) -> Dict[str, Any]:
    """Run the declared check and decide the candidate claim's fate.

    ``purpose`` selects the check family (``rule`` / ``repair`` /
    ``cost_saving``; anything else is treated as a plain ``rule`` check).
    ``claim`` is the harness's statement of what is being asserted (kept for
    the audit trail, never used as evidence). ``check`` describes the
    framework-side check:

    - ``{"kind": "result"|"comparison", "reference_objective": float,
       "tolerance": float, "semantic_ok": bool}``
    - ``{"kind": "cost_saving", "baseline": {...}, "candidate": {...},
       "quality_floor": float, "tolerance": float}``

    ``executions`` / ``supporting`` are ExecutionRecord objects (or dicts
    with ``quality`` / ``cost``). The verdict is computed from those facts —
    never from text the candidate wrote about itself.
    """
    check = dict(check or {})
    records = [_as_fact(r) for r in executions]
    supports = [_as_fact(r) for r in supporting]
    kind = str(check.get("kind") or ("cost_saving"
                                     if purpose == PURPOSE_COST_SAVING
                                     else "result"))
    if purpose == PURPOSE_COST_SAVING or kind == "cost_saving":
        return _verify_cost_saving(purpose, claim, check, records, supports)
    if purpose == PURPOSE_REPAIR:
        return _verify_repair(purpose, claim, check, records, supports)
    return _verify_rule(purpose, claim, check, records, supports)


def _as_fact(record: Any) -> Dict[str, Any]:
    """Normalize a record-like object into the fields checks read."""
    if isinstance(record, dict):
        quality = dict(record.get("quality") or {})
        cost = record.get("cost")
        if cost is None:
            measured: List[str] = []
        else:
            measured = sorted(record.get("cost_measured")
                              or getattr(cost, "measured", None) or [])
        return {"execution_id": str(record.get("execution_id", "")),
                "quality": quality, "cost": cost, "measured": measured}
    cost = getattr(record, "cost", None)
    return {
        "execution_id": str(getattr(record, "execution_id", "")),
        "quality": dict(getattr(record, "quality", {}) or {}),
        "cost": cost,
        "measured": sorted(cost.measured_dims()) if cost is not None else [],
    }


def _usable(fact: Dict[str, Any]) -> bool:
    """A fact whose execution actually produced a verdict-bearing result."""
    q = fact["quality"]
    return q.get("status") not in ("error", "timeout")


class _ScoreInput:
    """Minimal adapter so the shared ``quality_score`` definition is used for
    the cost/quality comparison instead of a second quality scale."""

    def __init__(self, fact: Dict[str, Any]):
        self.quality = fact["quality"]


def _as_score_input(fact: Dict[str, Any]) -> Any:
    return _ScoreInput(fact)


def _verify_rule(purpose: Optional[str], claim: str,
                 check: Dict[str, Any], records: List[Dict[str, Any]],
                 supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A rule claim: does the produced result satisfy the declared check?

    Framework-side checks only:
    - the execution completed with a feasible status;
    - the objective is finite and matches the reference within tolerance,
      when a reference is declared;
    - a problem-specific semantic check, when the harness supplied one
      (a boolean the FRAMEWORK evaluated, not a boolean the program printed);
    - the claim holds on more than just the inducing executions, when the
      harness supplied a comparison set (the with/without evidence).
    """
    if not records or not any(_usable(r) for r in records):
        return _report(
            INSUFFICIENT, purpose=purpose, claim=claim,
            checks=[{"check": "execution_completed"}],
            evidence=[r["execution_id"] for r in records],
            conclusion=("no usable execution evidence: the run(s) failed or "
                        "produced no result, so the claim was not checked — "
                        "not refuted"))
    fact = next(r for r in records if _usable(r))
    checks: List[Dict[str, Any]] = []
    evidence = [r["execution_id"] for r in records]
    q = fact["quality"]
    feasible = bool(q.get("feasible", False))
    checks.append({"check": "feasible", "observed": feasible})
    if not feasible:
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion="the declared rule did not produce a feasible result")
    obj = _finite(q.get("objective"))
    reference = _finite(check.get("reference_objective"))
    tolerance = _finite(check.get("tolerance"))
    if reference is not None:
        tol = tolerance if tolerance is not None else 1e-6 * max(1.0, abs(reference))
        gap = None if obj is None else abs(obj - reference)
        checks.append({"check": "objective_within_tolerance",
                       "observed": obj, "reference": reference,
                       "tolerance": tol, "gap": gap})
        if obj is None or gap > tol:
            return _report(REFUTED, purpose=purpose, claim=claim,
                           checks=checks, evidence=evidence,
                           conclusion=("the declared rule's result does not "
                                       "match the check's reference value"))
    semantic_ok = check.get("semantic_ok")
    if semantic_ok is not None:
        checks.append({"check": "problem_semantic_check",
                       "observed": bool(semantic_ok)})
        if not semantic_ok:
            return _report(REFUTED, purpose=purpose, claim=claim,
                           checks=checks, evidence=evidence,
                           conclusion="the problem-specific semantic check failed")
    if supports and not any(_usable(s) for s in supports):
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks + [{"check": "independent_comparison"}],
                       evidence=evidence,
                       conclusion=("the rule held where it was applied, but "
                                   "the independent comparison produced no "
                                   "usable execution — not yet verified"))
    if supports:
        checks.append({"check": "independent_comparison",
                       "n": len(supports),
                       "execution_ids": [s["execution_id"] for s in supports]})
        evidence += [s["execution_id"] for s in supports]
    return _report(VERIFIED, purpose=purpose, claim=claim, checks=checks,
                   evidence=evidence,
                   conclusion=("the declared check passed on real execution "
                               "evidence" + (" and an independent comparison"
                                             if supports else "")))


def _verify_repair(purpose: Optional[str], claim: str,
                   check: Dict[str, Any], records: List[Dict[str, Any]],
                   supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A repair claim: did the fix turn a failing execution into a success?

    Requires BOTH sides: the failed attempt (``supporting``) and the repaired
    attempt (``executions``). Only a usable repaired execution that succeeded
    where the original failed supports the claim."""
    evidence = [r["execution_id"] for r in records]
    checks: List[Dict[str, Any]] = []
    failed = [s for s in supports if not s["quality"].get("feasible", False)]
    succeeded = [r for r in records if _usable(r) and r["quality"].get("feasible")]
    checks.append({"check": "repair_succeeded_where_original_failed",
                   "failures": [s["execution_id"] for s in failed],
                   "successes": [r["execution_id"] for r in succeeded]})
    if not failed:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("no recorded failure to repair, so the "
                                   "repair claim cannot be checked"))
    if not succeeded:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence + [s["execution_id"] for s in failed],
                       conclusion=("the repaired execution produced no usable "
                                   "success — not refuted, simply unproven"))
    semantic_ok = check.get("semantic_ok")
    if semantic_ok is not None:
        checks.append({"check": "problem_semantic_check",
                       "observed": bool(semantic_ok)})
        if not semantic_ok:
            return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                           evidence=evidence,
                           conclusion="the repaired result fails the semantic check")
    return _report(VERIFIED, purpose=purpose, claim=claim, checks=checks,
                   evidence=evidence + [s["execution_id"] for s in failed],
                   conclusion="the repair produced a usable success where the original failed")


def _verify_cost_saving(purpose: Optional[str], claim: str,
                        check: Dict[str, Any], records: List[Dict[str, Any]],
                        supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A cost-saving claim: quality met, results close, cost ACTUALLY lower.

    Three conditions, all framework-side:
    1. the candidate's quality meets the declared floor (results may be
       equivalent rather than better — this is not an "improve the objective"
       test);
    2. the two results are close (within tolerance) so the comparison is
       like-for-like;
    3. the declared cost dimension is LOWER on the candidate, measured on
       BOTH sides under the same scope. A missing measurement proves nothing,
       so unknown never counts as cheap.
    """
    dim = str(check.get("dimension") or "llm_tokens")
    tolerance = _finite(check.get("tolerance"))
    tolerance = RESULT_SIMILARITY_TOLERANCE if tolerance is None else tolerance
    floor = _finite(check.get("quality_floor"))
    checks: List[Dict[str, Any]] = []
    evidence = [r["execution_id"] for r in records] + \
        [s["execution_id"] for s in supports]

    cand = [r for r in records if _usable(r)]
    base = [s for s in supports if _usable(s)]
    if not cand or not base:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "both_sides_executed",
                                "candidate": len(cand), "baseline": len(base)}],
                       evidence=evidence,
                       conclusion=("one side has no usable execution, so the "
                                   "cost comparison cannot be made — not "
                                   "refuted, simply unproven"))
    cand_fact = cand[-1]
    base_fact = base[-1]
    cand_q = _finite(cand_fact["quality"].get("objective"))
    base_q = _finite(base_fact["quality"].get("objective"))
    # The floor is expressed in the same [0, 1] quality space conditional
    # statistics use (infeasible = 0, else 1 - clamped gap), so a caller can
    # compare a "quality at least as good" claim against the observed result
    # without inventing a second quality definition.
    from or_harness.strategy.stats import quality_score
    cand_quality = quality_score(_as_score_input(cand_fact))
    checks.append({"check": "quality_floor", "floor": floor,
                   "observed": round(cand_quality, 4)})
    if not cand_fact["quality"].get("feasible", False):
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion="the cheaper candidate is not feasible")
    if floor is not None:
        if floor > 0.0 and cand_quality < floor:
            return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                           evidence=evidence,
                           conclusion="the candidate's quality is below the declared floor")
    if cand_q is not None and base_q is not None:
        scale = max(abs(base_q), 1e-9)
        delta = abs(cand_q - base_q) / scale
        checks.append({"check": "results_comparable", "candidate": cand_q,
                       "baseline": base_q, "relative_gap": delta,
                       "tolerance": tolerance})
        if delta > tolerance:
            return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                           evidence=evidence,
                           conclusion=("the results are not comparable "
                                       "(outside tolerance), so a lower cost "
                                       "would not be like-for-like"))
    else:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks + [{"check": "results_comparable"}],
                       evidence=evidence,
                       conclusion=("one side reported no objective, so the "
                                   "results cannot be shown comparable"))
    if dim not in cand_fact["measured"] or dim not in base_fact["measured"]:
        checks.append({"check": "cost_measured_both_sides", "dimension": dim,
                       "candidate_measured": dim in cand_fact["measured"],
                       "baseline_measured": dim in base_fact["measured"]})
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=(f"{dim} is not measured on both sides — "
                                   "an unmeasured dimension can never prove "
                                   "a saving"))
    cand_cost = _finite(getattr(cand_fact["cost"], dim, None))
    base_cost = _finite(getattr(base_fact["cost"], dim, None))
    checks.append({"check": "cost_lower", "dimension": dim,
                   "candidate": cand_cost, "baseline": base_cost})
    if cand_cost is None or base_cost is None:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion="the cost dimension could not be read")
    if not cand_cost < base_cost:
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion=(f"measured {dim} is not lower "
                                   f"({cand_cost} vs {base_cost})"))
    return _report(VERIFIED, purpose=purpose, claim=claim, checks=checks,
                   evidence=evidence,
                   conclusion=(f"quality met the floor with comparable results "
                               f"while measured {dim} fell from {base_cost} "
                               f"to {cand_cost}"))


__all__ = ["verify_candidate", "VERIFIED", "INSUFFICIENT", "REFUTED",
           "PURPOSE_RULE", "PURPOSE_REPAIR", "PURPOSE_COST_SAVING"]
