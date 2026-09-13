"""Offline admission verification: does the candidate claim actually hold?

The framework's job here is narrow and mechanical. The harness (outer agent)
proposes the claim, supplies the executions and states which check applies;
THIS module evaluates those checks against the actual execution facts and
decides:

  verified              — every declared check holds on real execution evidence
  insufficient_evidence — the check cannot be run or cannot decide (no
                          identification, no check basis, missing/duplicate/
                          non-corresponding evidence, or the execution failed)
  refuted               — the check ran on real evidence and did NOT hold

Red lines:

1. **A program's own verdict is not a verification.** "It printed
   ``{"principle_failed": false}``" proves the program ran. Verdicts are
   computed by the framework from checkable facts.
2. **A failed execution is not a refutation.** "We could not check it" and
   "we checked it and it failed" are different results; only the second
   refutes.
3. **Evidence must correspond to the candidate.** The same payload must not be
   reused across induction targets: the executions have to be for this
   strategy/family, the two sides of a comparison have to be the same kind of
   thing (same task, same measurement scope), and a record repeated on both
   sides is not an independent comparison.
4. **No check basis, no verdict.** Feasibility alone is a precondition, not a
   check: with nothing declared to check against, the honest answer is
   ``insufficient_evidence``.

Purpose tags (``rule`` / ``repair`` / ``cost_saving``) select the check; they
are NOT an enumeration of what strategic knowledge may ever be.

Every report carries the audit trail the harness needs to read back: which
claim, which executions, which checks, and why that conclusion followed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from or_harness.core.schema import COST_DIMENSIONS, CostVector

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

#: Provenance of a check. Only a FRAMEWORK check can carry a `verified`
#: verdict: a bare boolean the harness asserts cannot be re-derived by the
#: framework, so it is recorded and labelled, never sufficient alone.
AGENT_DECLARED = "agent-declared"
FRAMEWORK = "framework"


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
        "evidence": [e for e in evidence if e],
        "conclusion": conclusion,
        "verified_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Fact normalization — ONE representation, whichever door the data came in
# ---------------------------------------------------------------------------

def _as_cost(raw: Any) -> Optional[CostVector]:
    """Normalize a cost payload.

    The same verification must behave identically whether the caller passes
    ``ExecutionRecord`` objects (Python API) or the JSON a CLI user typed — a
    dict cost once made the CLI report "the cost dimension could not be read"
    for evidence the object path verified happily."""
    if raw is None:
        return None
    if isinstance(raw, CostVector):
        return raw
    if isinstance(raw, dict):
        return CostVector.from_dict(raw)
    return None


def _as_fact(record: Any) -> Dict[str, Any]:
    """Normalize a record-like object into the fields every check reads.

    Identification is preserved deliberately (id, task, strategy, family,
    measurement scope) and so is the raw payload: dropping them is how a
    verification payload for one candidate once verified a different one, and
    how a task-scope total was compared against an attempt cost."""
    if isinstance(record, dict):
        profile = record.get("profile_snapshot")
        family = record.get("family")
        if family is None and isinstance(profile, dict):
            family = profile.get("family")
        cost = _as_cost(record.get("cost"))
        raw_measured = record.get("cost_measured")
        if isinstance(raw_measured, list):
            measured = sorted(str(d) for d in raw_measured
                              if d in COST_DIMENSIONS)
        elif cost is not None:
            measured = sorted(cost.measured_dims())
        else:
            measured = []
        return {
            "payload": dict(record),
            "execution_id": str(record.get("execution_id") or ""),
            "task_id": str(record.get("task_id") or ""),
            "strategy_id": str(record.get("strategy_id") or ""),
            "family": str(family) if family is not None else "",
            "measurement_scope": str(record.get("measurement_scope") or "attempt"),
            "quality": dict(record.get("quality") or {}),
            "cost": cost,
            "measured": measured,
        }
    cost = _as_cost(getattr(record, "cost", None))
    profile = getattr(record, "profile_snapshot", None)
    if isinstance(profile, dict):
        family = profile.get("family", "")
    else:
        family = getattr(profile, "family", "") if profile is not None else ""
    return {
        "payload": (record.to_dict() if hasattr(record, "to_dict") else {}),
        "execution_id": str(getattr(record, "execution_id", "") or ""),
        "task_id": str(getattr(record, "task_id", "") or ""),
        "strategy_id": str(getattr(record, "strategy_id", "") or ""),
        "family": str(family or ""),
        "measurement_scope": str(getattr(record, "measurement_scope", "attempt")),
        "quality": dict(getattr(record, "quality", {}) or {}),
        "cost": cost,
        "measured": sorted(cost.measured_dims()) if cost is not None else [],
    }


def _usable(fact: Dict[str, Any]) -> bool:
    """A fact whose execution actually produced a verdict-bearing result."""
    return fact["quality"].get("status") not in ("error", "timeout")


def _identified(fact: Dict[str, Any]) -> bool:
    """A fact the audit trail can point at. An evidence list of anonymous
    records cannot be checked back against anything, so it is not evidence."""
    return bool(fact["execution_id"])


def _copy_for_score(fact: Dict[str, Any]):
    class _ScoreInput:
        quality = fact["quality"]
    return _ScoreInput()


# ---------------------------------------------------------------------------
# Framework-side probes (a declared boolean is not a check)
# ---------------------------------------------------------------------------

def _resolve(payload: Any, path: str) -> Tuple[bool, Any]:
    """Resolve a dotted path inside a record payload."""
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _evaluate_probe(fact: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ONE framework-side probe against a real record.

    Supported probes all read actual observed values:
    ``{"path": "quality.objective", "equals"|"min"|"max"|"in": ...}``.

    This is what makes a problem-specific semantic check a *check* rather than
    an assertion: the framework reads the value and compares it."""
    path = str(probe.get("path", ""))
    found, observed = _resolve(fact.get("payload") or {}, path)
    result: Dict[str, Any] = {"check": "semantic_probe", "source": FRAMEWORK,
                              "path": path, "found": found,
                              "observed": observed}
    if "equals" in probe:
        ok = found and observed == probe["equals"]
        result.update({"expected": probe["equals"], "mode": "equals", "ok": ok})
    elif "min" in probe or "max" in probe:
        value = _finite(observed)
        lo = _finite(probe.get("min"))
        hi = _finite(probe.get("max"))
        ok = (value is not None
              and (lo is None or value >= lo)
              and (hi is None or value <= hi))
        result.update({"min": probe.get("min"), "max": probe.get("max"),
                       "mode": "range", "ok": ok})
    elif "in" in probe:
        options = probe.get("in") or []
        ok = found and observed in options
        result.update({"expected": list(options), "mode": "in", "ok": ok})
    else:
        result.update({"ok": None,
                       "error": "probe declares no comparison "
                                "(equals / min / max / in)"})
    return result


def _declared_semantic(flag: Any) -> Dict[str, Any]:
    """A bare boolean the harness asserts: kept and labelled, never able to
    carry a verdict on its own (the framework cannot re-derive it)."""
    return {"check": "problem_semantic_check", "source": AGENT_DECLARED,
            "observed": bool(flag), "ok": bool(flag)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def verify_candidate(purpose: Optional[str], claim: str,
                     *, check: Optional[Dict[str, Any]] = None,
                     executions: Sequence[Any] = (),
                     supporting: Sequence[Any] = (),
                     strategy_id: Optional[str] = None,
                     family: Optional[str] = None) -> Dict[str, Any]:
    """Run the declared check and decide the candidate claim's fate.

    ``purpose`` selects the check family (``rule`` / ``repair`` /
    ``cost_saving``; anything else is treated as ``rule``). ``claim`` is the
    harness's statement of what is asserted (kept for the audit trail, never
    used as evidence). ``strategy_id`` / ``family`` identify the candidate
    this evidence is offered FOR — evidence about another strategy or family
    does not verify this one.
    """
    check = dict(check or {})
    records = [_as_fact(r) for r in executions]
    supports = [_as_fact(r) for r in supporting]
    mismatch = _correspondence_problem(records + supports, strategy_id, family)
    if mismatch is not None:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "evidence_corresponds_to_candidate",
                                "source": FRAMEWORK,
                                "expected": {"strategy_id": strategy_id,
                                             "family": family},
                                "problem": mismatch}],
                       evidence=[f["execution_id"] for f in records + supports],
                       conclusion=("the supplied evidence does not correspond "
                                   "to this candidate: " + mismatch))
    kind = str(check.get("kind") or ("cost_saving"
                                     if purpose == PURPOSE_COST_SAVING
                                     else "result"))
    if purpose == PURPOSE_COST_SAVING or kind == "cost_saving":
        return _verify_cost_saving(purpose, claim, check, records, supports)
    if purpose == PURPOSE_REPAIR:
        return _verify_repair(purpose, claim, check, records, supports)
    return _verify_rule(purpose, claim, check, records, supports)


def _correspondence_problem(facts: List[Dict[str, Any]],
                            strategy_id: Optional[str],
                            family: Optional[str]) -> Optional[str]:
    for fact in facts:
        if not _identified(fact):
            continue
        if strategy_id and fact["strategy_id"] and fact["strategy_id"] != strategy_id:
            return (f"execution {fact['execution_id']} ran strategy "
                    f"{fact['strategy_id']}, not {strategy_id}")
        if family and fact["family"] and fact["family"] != family:
            return (f"execution {fact['execution_id']} belongs to family "
                    f"{fact['family']}, not {family}")
    return None


def _duplicate_ids(facts: List[Dict[str, Any]]) -> Set[str]:
    seen: Set[str] = set()
    dupes: Set[str] = set()
    for fact in facts:
        eid = fact["execution_id"]
        if not eid:
            continue
        if eid in seen:
            dupes.add(eid)
        seen.add(eid)
    return dupes


def _declared_basis(check: Dict[str, Any], has_comparison: bool) -> List[str]:
    declared: List[str] = []
    if _finite(check.get("reference_objective")) is not None:
        declared.append("reference_objective")
    if check.get("reference_status") is not None:
        declared.append("reference_status")
    if check.get("semantic_probe") is not None:
        declared.append("semantic_probe")
    if check.get("semantic_ok") is not None:
        declared.append("semantic_ok")
    if has_comparison:
        declared.append("comparison")
    return declared


#: Checks that are PRECONDITIONS rather than a declared criterion. Feasibility
#: (and "the run produced a result") must hold for any check to run, but
#: neither is a basis for a verdict on its own — that is how a payload of
#: ``{"quality": {"feasible": true}}`` once got verified with nothing declared.
PRECONDITION_CHECKS = {"feasible", "execution_completed",
                       "both_sides_required", "both_sides_executed",
                       "evidence_is_distinct", "evidence_is_identified",
                       "check_basis"}


def _framework_checks(checks: List[Dict[str, Any]]) -> int:
    """How many SUBSTANTIVE framework checks were evaluated.

    Agent-declared results do not count (the framework cannot re-derive
    them), and neither do preconditions — only a criterion the framework
    itself computed from observed values."""
    return sum(1 for c in checks
               if c.get("source") == FRAMEWORK
               and c.get("check") not in PRECONDITION_CHECKS)


def _apply_declared_checks(fact: Dict[str, Any], check: Dict[str, Any],
                           checks: List[Dict[str, Any]]
                           ) -> Tuple[bool, Optional[str]]:
    """Evaluate every declared criterion on ONE record.

    Returns ``(passed, failure_reason)``. EVERY supplied record is checked —
    evaluating only the first usable one let record ORDER decide the verdict
    and let a counterexample later in the same batch go unnoticed."""
    q = fact["quality"]
    feasible = bool(q.get("feasible", False))
    checks.append({"check": "feasible", "source": FRAMEWORK,
                   "execution_id": fact["execution_id"], "observed": feasible})
    if not feasible:
        return False, f"{fact['execution_id']} is not feasible"

    reference_status = check.get("reference_status")
    if reference_status is not None:
        status = str(q.get("status", ""))
        ok = status == str(reference_status)
        checks.append({"check": "status_matches", "source": FRAMEWORK,
                       "execution_id": fact["execution_id"],
                       "observed": status, "expected": reference_status,
                       "ok": ok})
        if not ok:
            return False, (f"{fact['execution_id']} finished with status "
                           f"{status!r}, not {reference_status!r}")

    reference = _finite(check.get("reference_objective"))
    if reference is not None:
        obj = _finite(q.get("objective"))
        tolerance = _finite(check.get("tolerance"))
        tol = tolerance if tolerance is not None else 1e-6 * max(1.0, abs(reference))
        gap = None if obj is None else abs(obj - reference)
        ok = obj is not None and gap <= tol
        checks.append({"check": "objective_within_tolerance", "source": FRAMEWORK,
                       "execution_id": fact["execution_id"], "observed": obj,
                       "reference": reference, "tolerance": tol, "gap": gap,
                       "ok": ok})
        if not ok:
            return False, (f"{fact['execution_id']} reports objective {obj}, "
                           f"outside {tol} of the reference {reference}")

    probe = check.get("semantic_probe")
    if probe is not None:
        for item in (probe if isinstance(probe, list) else [probe]):
            result = _evaluate_probe(fact, dict(item or {}))
            result["execution_id"] = fact["execution_id"]
            checks.append(result)
            if result.get("ok") is None:
                return False, (f"semantic probe {result.get('path')!r} "
                               "declares no comparison")
            if not result["ok"]:
                return False, (f"semantic probe {result.get('path')!r} failed "
                               "on real values")

    if check.get("semantic_ok") is not None:
        result = _declared_semantic(check.get("semantic_ok"))
        result["execution_id"] = fact["execution_id"]
        checks.append(result)
        if not result["ok"]:
            return False, "the problem-specific semantic check failed"
    return True, None


def _decision(purpose: Optional[str], claim: str, checks: List[Dict[str, Any]],
              evidence: List[str], verified_conclusion: str) -> Dict[str, Any]:
    """Only a FRAMEWORK check can carry a `verified` verdict: a bare
    agent-declared boolean is recorded, but it cannot do the job alone."""
    if _framework_checks(checks) == 0:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("only agent-declared results were supplied; "
                                   "the framework found nothing it could check "
                                   "by itself"))
    return _report(VERIFIED, purpose=purpose, claim=claim, checks=checks,
                   evidence=evidence, conclusion=verified_conclusion)


def _guards(purpose: Optional[str], claim: str, records: List[Dict[str, Any]],
            supports: List[Dict[str, Any]], side_names: Tuple[str, str],
            *, require_both: bool = True) -> Optional[Dict[str, Any]]:
    """Shared entry guards: distinct, identified, present evidence."""
    evidence = [f["execution_id"] for f in records + supports]
    dupes = _duplicate_ids(records + supports)
    if dupes:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "evidence_is_distinct",
                                "source": FRAMEWORK,
                                "duplicates": sorted(dupes)}],
                       evidence=evidence,
                       conclusion=("the same execution was supplied more than "
                                   "once, so it is not independent evidence: "
                                   + ", ".join(sorted(dupes))))
    left, right = side_names
    missing = (not records) if not require_both else (not records or not supports)
    if missing:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "both_sides_required",
                                "source": FRAMEWORK,
                                left: len(records), right: len(supports),
                                "both_required": require_both}],
                       evidence=evidence,
                       conclusion=(f"this claim needs the '{left}' side"
                                   + (f" and the '{right}' side" if require_both
                                      else "")))
    if not records:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "both_sides_required",
                                "source": FRAMEWORK,
                                left: len(records), right: len(supports),
                                "both_required": require_both}],
                       evidence=evidence,
                       conclusion=f"this claim needs the '{left}' side")
    if not all(_identified(f) for f in records + supports):
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "evidence_is_identified",
                                "source": FRAMEWORK}],
                       evidence=evidence,
                       conclusion=("an execution carries no execution_id, so "
                                   "the evidence cannot be retraced and is not "
                                   "a check — supply the recorded id"))
    return None


def _verify_rule(purpose: Optional[str], claim: str,
                 check: Dict[str, Any], records: List[Dict[str, Any]],
                 supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A rule claim: does the produced result satisfy the declared check?

    Every supplied execution is evaluated, so a counterexample anywhere in
    the batch refutes the claim regardless of ordering. The comparison set
    must be genuinely independent — not the same executions re-passed, and
    not the same tasks that produced the claim."""
    evidence = [f["execution_id"] for f in records + supports]
    guard = _guards(purpose, claim, records, supports,
                    ("executions", "supporting"), require_both=False)
    if guard is not None:
        return guard
    usable = [f for f in records if _usable(f)]
    if not usable:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "execution_completed",
                                "source": FRAMEWORK, "observed": False}],
                       evidence=evidence,
                       conclusion=("no usable execution evidence: the run(s) "
                                   "failed or produced no result, so the claim "
                                   "was not checked — not refuted"))
    declared = _declared_basis(check, bool(supports))
    if not declared:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "check_basis", "source": FRAMEWORK,
                                "declared": []}],
                       evidence=evidence,
                       conclusion=("no check basis: declare reference_status, "
                                   "reference_objective, semantic_probe or "
                                   "semantic_ok, or supply a comparison set — "
                                   "feasibility alone is a precondition, not a "
                                   "check"))

    checks: List[Dict[str, Any]] = []
    for fact in records:
        if not _usable(fact):
            checks.append({"check": "execution_completed", "source": FRAMEWORK,
                           "execution_id": fact["execution_id"],
                           "observed": False})
            return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                           checks=checks, evidence=evidence,
                           conclusion=(f"{fact['execution_id']} produced no "
                                       "usable result, so the check could not "
                                       "run on every supplied execution"))
        passed, reason = _apply_declared_checks(fact, check, checks)
        if not passed:
            return _report(REFUTED, purpose=purpose, claim=claim,
                           checks=checks, evidence=evidence,
                           conclusion=f"the declared check failed: {reason}")

    if supports:
        primary_tasks = {f["task_id"] for f in records if f["task_id"]}
        fresh = [f for f in supports
                 if f["task_id"] and f["task_id"] not in primary_tasks]
        checks.append({"check": "independent_comparison", "source": FRAMEWORK,
                       "n": len(supports),
                       "independent_from_the_inducing_tasks": len(fresh),
                       "execution_ids": [f["execution_id"] for f in supports]})
        if not fresh:
            return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                           checks=checks, evidence=evidence,
                           conclusion=("the comparison set repeats the "
                                       "inducing tasks, so it is not an "
                                       "independent check"))
        for fact in fresh:
            if not _usable(fact):
                return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                               checks=checks, evidence=evidence,
                               conclusion=(f"the comparison execution "
                                           f"{fact['execution_id']} produced no "
                                           "usable result — not yet verified"))
            passed, reason = _apply_declared_checks(fact, check, checks)
            if not passed:
                return _report(REFUTED, purpose=purpose, claim=claim,
                               checks=checks, evidence=evidence,
                               conclusion=("the claim did not hold on the "
                                           f"independent comparison: {reason}"))
    return _decision(purpose, claim, checks, evidence,
                     ("the declared check passed on real execution evidence"
                      + (" and an independent comparison" if supports else "")))


def _verify_repair(purpose: Optional[str], claim: str,
                   check: Dict[str, Any], records: List[Dict[str, Any]],
                   supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A repair claim: did the fix turn a failing execution into a success?

    Both sides must describe the SAME task: a failure on one problem and a
    success on an unrelated one is not a repair, it is two unrelated facts."""
    evidence = [f["execution_id"] for f in records + supports]
    guard = _guards(purpose, claim, records, supports,
                    ("executions", "supporting"))
    if guard is not None:
        return guard
    checks: List[Dict[str, Any]] = []
    failed = [f for f in supports if not f["quality"].get("feasible", False)]
    succeeded = [f for f in records if _usable(f) and f["quality"].get("feasible")]
    repair_tasks = {f["task_id"] for f in failed if f["task_id"]}
    fixed_tasks = {f["task_id"] for f in succeeded if f["task_id"]}
    shared = repair_tasks & fixed_tasks
    checks.append({"check": "repair_succeeded_where_original_failed",
                   "source": FRAMEWORK,
                   "failures": [f["execution_id"] for f in failed],
                   "successes": [f["execution_id"] for f in succeeded],
                   "failure_tasks": sorted(repair_tasks),
                   "success_tasks": sorted(fixed_tasks),
                   "shared_tasks": sorted(shared)})
    if not failed:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("no recorded failure to repair, so the "
                                   "repair claim cannot be checked"))
    if not succeeded:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("the repaired execution produced no usable "
                                   "success — not refuted, simply unproven"))
    if not shared:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("the failure and the success belong to "
                                   "different tasks, so this is not a repair "
                                   f"of one problem: {sorted(repair_tasks)} vs "
                                   f"{sorted(fixed_tasks)}"))
    passed, reason = _apply_declared_checks(succeeded[-1], check, checks)
    if not passed:
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion=f"the repaired result fails the check: {reason}")
    return _decision(purpose, claim, checks, evidence,
                     "the repair produced a usable success on the same task "
                     "where the original failed")


def _verify_cost_saving(purpose: Optional[str], claim: str,
                        check: Dict[str, Any], records: List[Dict[str, Any]],
                        supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A cost-saving claim: quality met, results close, cost ACTUALLY lower.

    Comparability is part of the claim, so the framework enforces it:
    - the two sides describe the SAME task (a cheaper run of a different
      problem is not a cheaper answer to this one);
    - the two sides share a measurement SCOPE (an attempt cost is never
      compared against a task total — different quantities);
    - the same execution id may not appear on both sides;
    - the declared dimension is measured on BOTH sides (unknown is not cheap).
    """
    dim = str(check.get("dimension") or "llm_tokens")
    tolerance = _finite(check.get("tolerance"))
    tolerance = RESULT_SIMILARITY_TOLERANCE if tolerance is None else tolerance
    floor = _finite(check.get("quality_floor"))
    evidence = [f["execution_id"] for f in records + supports]
    guard = _guards(purpose, claim, records, supports,
                    ("executions", "supporting"))
    if guard is not None:
        return guard
    checks: List[Dict[str, Any]] = []

    cand = [f for f in records if _usable(f)]
    base = [f for f in supports if _usable(f)]
    if not cand or not base:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=[{"check": "both_sides_executed",
                                "source": FRAMEWORK,
                                "candidate": len(cand), "baseline": len(base)}],
                       evidence=evidence,
                       conclusion=("one side has no usable execution, so the "
                                   "cost comparison cannot be made — not "
                                   "refuted, simply unproven"))

    cand_tasks = {f["task_id"] for f in cand if f["task_id"]}
    base_tasks = {f["task_id"] for f in base if f["task_id"]}
    shared_tasks = cand_tasks & base_tasks
    cand_scopes = {f["measurement_scope"] for f in cand}
    base_scopes = {f["measurement_scope"] for f in base}
    shared_scopes = cand_scopes & base_scopes
    checks.append({"check": "comparison_is_like_for_like", "source": FRAMEWORK,
                   "candidate_tasks": sorted(cand_tasks),
                   "baseline_tasks": sorted(base_tasks),
                   "shared_tasks": sorted(shared_tasks),
                   "candidate_scopes": sorted(cand_scopes),
                   "baseline_scopes": sorted(base_scopes),
                   "shared_scopes": sorted(shared_scopes)})
    if not shared_tasks:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("the two sides describe different tasks "
                                   f"({sorted(cand_tasks)} vs "
                                   f"{sorted(base_tasks)}), so the costs are "
                                   "not comparable"))
    if not shared_scopes:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=("the two sides measure different scopes "
                                   f"({sorted(cand_scopes)} vs "
                                   f"{sorted(base_scopes)}): an attempt cost "
                                   "and a task total are different quantities"))

    # Compare WITHIN one shared task+scope pair, never across mixed batches
    # (comparing a cheap attempt against an expensive task total "proved" a
    # saving that does not exist).
    pair_task = sorted(shared_tasks)[0]
    pair_scope = sorted(shared_scopes)[0]
    cand_fact = next(f for f in reversed(cand) if f["task_id"] == pair_task
                     and f["measurement_scope"] == pair_scope)
    base_fact = next(f for f in reversed(base) if f["task_id"] == pair_task
                     and f["measurement_scope"] == pair_scope)

    cand_q = _finite(cand_fact["quality"].get("objective"))
    base_q = _finite(base_fact["quality"].get("objective"))
    from or_harness.strategy.stats import quality_score
    cand_quality = quality_score(_copy_for_score(cand_fact))
    checks.append({"check": "quality_floor", "source": FRAMEWORK, "floor": floor,
                   "observed": round(cand_quality, 4),
                   "ok": floor is None or cand_quality >= floor})
    if not cand_fact["quality"].get("feasible", False):
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion="the cheaper candidate is not feasible")
    if floor is not None and cand_quality < floor:
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion="the candidate's quality is below the declared floor")
    if cand_q is not None and base_q is not None:
        scale = max(abs(base_q), 1e-9)
        delta = abs(cand_q - base_q) / scale
        checks.append({"check": "results_comparable", "source": FRAMEWORK,
                       "candidate": cand_q, "baseline": base_q,
                       "relative_gap": delta, "tolerance": tolerance})
        if delta > tolerance:
            return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                           evidence=evidence,
                           conclusion=("the results are not comparable "
                                       "(outside tolerance), so a lower cost "
                                       "would not be like-for-like"))
    else:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks + [{"check": "results_comparable",
                                         "source": FRAMEWORK}],
                       evidence=evidence,
                       conclusion=("one side reported no objective, so the "
                                   "results cannot be shown comparable"))
    if dim not in cand_fact["measured"] or dim not in base_fact["measured"]:
        checks.append({"check": "cost_measured_both_sides", "source": FRAMEWORK,
                       "dimension": dim,
                       "candidate_measured": dim in cand_fact["measured"],
                       "baseline_measured": dim in base_fact["measured"]})
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion=(f"{dim} is not measured on both sides — "
                                   "an unmeasured dimension can never prove "
                                   "a saving"))
    cand_cost = _finite(getattr(cand_fact["cost"], dim, None))
    base_cost = _finite(getattr(base_fact["cost"], dim, None))
    checks.append({"check": "cost_lower", "source": FRAMEWORK,
                   "dimension": dim, "candidate": cand_cost,
                   "baseline": base_cost, "task": pair_task,
                   "scope": pair_scope})
    if cand_cost is None or base_cost is None:
        return _report(INSUFFICIENT, purpose=purpose, claim=claim,
                       checks=checks, evidence=evidence,
                       conclusion="the cost dimension could not be read")
    if not cand_cost < base_cost:
        return _report(REFUTED, purpose=purpose, claim=claim, checks=checks,
                       evidence=evidence,
                       conclusion=(f"measured {dim} is not lower "
                                   f"({cand_cost} vs {base_cost})"))
    return _decision(purpose, claim, checks, evidence,
                     (f"quality met the floor with comparable results on the "
                      f"same task ({pair_task}, {pair_scope}) while measured "
                      f"{dim} fell from {base_cost} to {cand_cost}"))


__all__ = ["verify_candidate", "VERIFIED", "INSUFFICIENT", "REFUTED",
           "PURPOSE_RULE", "PURPOSE_REPAIR", "PURPOSE_COST_SAVING"]
