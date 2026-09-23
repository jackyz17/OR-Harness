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
#: A STRUCTURED RELATION claim: assertions over explicitly referenced
#: evidence with roles (see :func:`verify_relation`). Not a statistic about
#: one strategy — the check is the set of computable assertions the claim
#: declares, evaluated over the referenced evidence.
PURPOSE_RELATION = "relation"

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


# ---------------------------------------------------------------------------
# Relation claims: assertions over explicitly referenced evidence
# ---------------------------------------------------------------------------

#: Assertion kinds the framework can actually COMPUTE. Anything else is
#: reported as an unsupported assertion (never silently ignored): the
#: framework checks what a structured declaration makes computable, and the
#: outer agent owns interpretation and phrasing.
ASSERTION_PROBE = "probe"
ASSERTION_STATUS = "status"
ASSERTION_COMPARISON = "comparison"

#: Aggregation modes for a comparison assertion.
AGGREGATION_ALL = "all"
AGGREGATION_MEAN = "mean"

#: Pairing modes for a comparison assertion. ``paired`` compares only records
#: that share a task (one per side); ``group`` compares the two sides' means.
#: Pairing is a property of the ASSERTION, never of the whole batch: a claim
#: may carry a paired assertion and a group assertion side by side, each
#: evaluated over its own scope.
MODE_PAIRED = "paired"
MODE_GROUP = "group"


def _metric_value(fact: Dict[str, Any], metric: str) -> Optional[float]:
    """The numeric value of a declared comparison metric on one fact."""
    metric = str(metric or "")
    if metric in ("quality", "quality_score"):
        from or_harness.strategy.stats import quality_score
        return quality_score(_copy_for_score(fact))
    if metric.startswith("cost:"):
        dim = metric.split(":", 1)[1]
        if dim not in fact["measured"] or fact["cost"] is None:
            return None
        return _finite(getattr(fact["cost"], dim, None))
    if metric in COST_DIMENSIONS:
        if metric not in fact["measured"] or fact["cost"] is None:
            return None
        return _finite(getattr(fact["cost"], metric, None))
    if metric.startswith("quality."):
        found, observed = _resolve(fact.get("payload") or {}, metric)
        return _finite(observed) if found else None
    found, observed = _resolve(fact.get("payload") or {}, metric)
    return _finite(observed) if found else None


def _direction_ok(delta: float, direction: str, min_gap: float) -> bool:
    """Whether a signed difference satisfies the declared direction/gap.

    ``delta`` = (side_a - side_b). ``higher`` requires a positive advantage of
    at least ``min_gap``; ``lower`` requires a negative one of at least
    ``min_gap``."""
    gap = abs(delta)
    if gap < min_gap:
        return False
    return delta > 0 if direction == "higher" else delta < 0


def _assertion_checks(assertion: Dict[str, Any],
                      by_role: Dict[str, List[Dict[str, Any]]],
                      checks: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """Evaluate ONE assertion over the role-partitioned evidence.

    Returns ``(state, failure_reason)`` where state is ``VERIFIED`` /
    ``REFUTED`` / ``INSUFFICIENT``. Every record of every named role is
    evaluated, so a counterexample anywhere in the scope is found regardless
    of ordering."""
    kind = str(assertion.get("kind") or "")
    roles = [str(r) for r in (assertion.get("roles") or [])]

    def _role_records(role_list: List[str]) -> Optional[List[Dict[str, Any]]]:
        out: List[Dict[str, Any]] = []
        for role in role_list:
            records = by_role.get(role)
            if not records:
                return None
            out.extend(records)
        return out

    if kind == ASSERTION_PROBE:
        records = _role_records(roles)
        if records is None:
            checks.append({"check": "assertion_scope", "source": FRAMEWORK,
                           "kind": kind, "roles": roles,
                           "problem": "a named role has no evidence"})
            return INSUFFICIENT, ("a named role has no referenced evidence, "
                                  "so the assertion cannot run")
        probe = {k: v for k, v in assertion.items()
                 if k in ("path", "equals", "min", "max", "in")}
        for fact in records:
            result = _evaluate_probe(fact, probe)
            result["check"] = "assertion_probe"
            result["role_records"] = True
            result["execution_id"] = fact["execution_id"]
            checks.append(result)
            if result.get("ok") is None:
                return INSUFFICIENT, (f"probe {probe.get('path')!r} declares "
                                      "no comparison")
            if not result["ok"]:
                return REFUTED, (f"probe {probe.get('path')!r} failed on "
                                 f"{fact['execution_id']}")
        return VERIFIED, None

    if kind == ASSERTION_STATUS:
        records = _role_records(roles)
        if records is None:
            checks.append({"check": "assertion_scope", "source": FRAMEWORK,
                           "kind": kind, "roles": roles,
                           "problem": "a named role has no evidence"})
            return INSUFFICIENT, ("a named role has no referenced evidence, "
                                  "so the assertion cannot run")
        expected = str(assertion.get("status"))
        for fact in records:
            status = str(fact["quality"].get("status", ""))
            ok = status == expected
            checks.append({"check": "assertion_status", "source": FRAMEWORK,
                           "execution_id": fact["execution_id"],
                           "observed": status, "expected": expected, "ok": ok})
            if not ok:
                return REFUTED, (f"{fact['execution_id']} finished with status "
                                 f"{status!r}, not {expected!r}")
        return VERIFIED, None

    if kind == ASSERTION_COMPARISON:
        side_a = _role_records([str(r) for r in
                                (assertion.get("roles_a") or [])])
        side_b = _role_records([str(r) for r in
                                (assertion.get("roles_b") or [])])
        if side_a is None or side_b is None:
            checks.append({"check": "assertion_scope", "source": FRAMEWORK,
                           "kind": kind,
                           "roles_a": assertion.get("roles_a"),
                           "roles_b": assertion.get("roles_b"),
                           "problem": "a named role has no evidence"})
            return INSUFFICIENT, ("a named role has no referenced evidence, "
                                  "so the comparison cannot run")
        metric = str(assertion.get("metric") or "")
        direction = str(assertion.get("direction") or "higher")
        min_gap = _finite(assertion.get("min_gap"))
        min_gap = 0.0 if min_gap is None else min_gap
        mode = str(assertion.get("mode") or MODE_GROUP)
        aggregation = str(assertion.get("aggregation") or AGGREGATION_ALL)
        # Requirement declared by the claim itself: when the assertion names
        # two strategies, the framework can check each side is internally
        # consistent about which strategy ran.
        if assertion.get("require_strategy_ids"):
            for label, side in (("roles_a", side_a), ("roles_b", side_b)):
                ids = {f["strategy_id"] for f in side if f["strategy_id"]}
                if len(ids) > 1:
                    checks.append({"check": "assertion_strategy_consistent",
                                   "source": FRAMEWORK, "side": label,
                                   "strategy_ids": sorted(ids), "ok": False})
                    return INSUFFICIENT, (f"{label} mixes strategies "
                                          f"{sorted(ids)}: the comparison has "
                                          "no single strategy per side")
        if mode == MODE_PAIRED:
            pairs, unpaired = _pair_by_task(side_a, side_b)
            checks.append({"check": "assertion_paired_scope",
                           "source": FRAMEWORK, "metric": metric,
                           "n_pairs": len(pairs),
                           "unpaired_execution_ids": [
                               f["execution_id"] for f in unpaired],
                           "note": ("only same-task pairs participate; "
                                    "records without a counterpart on the "
                                    "other side are recorded but not counted")})
            if not pairs:
                return INSUFFICIENT, ("no task has comparable records on both "
                                      "sides, so a paired comparison cannot "
                                      "be made")
            deltas: List[float] = []
            for task_id, fact_a, fact_b in pairs:
                value_a = _metric_value(fact_a, metric)
                value_b = _metric_value(fact_b, metric)
                checks.append({"check": "assertion_pair", "source": FRAMEWORK,
                               "task_id": task_id, "metric": metric,
                               "a": {"execution_id": fact_a["execution_id"],
                                     "value": value_a},
                               "b": {"execution_id": fact_b["execution_id"],
                                     "value": value_b}})
                if value_a is None or value_b is None:
                    return INSUFFICIENT, (f"{metric} is not measurable on both "
                                          f"sides of task {task_id} — an "
                                          "unmeasured metric cannot be "
                                          "compared")
                deltas.append(value_a - value_b)
            if aggregation == AGGREGATION_MEAN:
                mean_delta = sum(deltas) / len(deltas)
                ok = _direction_ok(mean_delta, direction, min_gap)
                checks.append({"check": "assertion_mean_delta",
                               "source": FRAMEWORK, "metric": metric,
                               "mean_delta": round(mean_delta, 6),
                               "direction": direction, "min_gap": min_gap,
                               "ok": ok})
                if not ok:
                    return REFUTED, (f"the mean paired advantage "
                                     f"({mean_delta:.4g}) does not meet "
                                     f"{direction} {min_gap}")
            else:
                for delta in deltas:
                    ok = _direction_ok(delta, direction, min_gap)
                    if not ok:
                        return REFUTED, (f"a paired comparison failed the "
                                         f"declared direction/gap "
                                         f"(delta {delta:.4g})")
                checks.append({"check": "assertion_all_pairs",
                               "source": FRAMEWORK, "metric": metric,
                               "n_pairs": len(deltas),
                               "direction": direction, "min_gap": min_gap,
                               "ok": True})
            return VERIFIED, None
        # group mode: compare the two sides' means over complete metrics
        values_a = [_metric_value(f, metric) for f in side_a]
        values_b = [_metric_value(f, metric) for f in side_b]
        if any(v is None for v in values_a) or any(v is None for v in values_b):
            checks.append({"check": "assertion_metric_complete",
                           "source": FRAMEWORK, "metric": metric,
                           "measured_a": [v is not None for v in values_a],
                           "measured_b": [v is not None for v in values_b]})
            return INSUFFICIENT, (f"{metric} is not measured on every "
                                  "referenced record — a mean over a subset "
                                  "is a partial observation, not a claim")
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)
        delta = mean_a - mean_b
        ok = _direction_ok(delta, direction, min_gap)
        checks.append({"check": "assertion_group_means", "source": FRAMEWORK,
                       "metric": metric, "mean_a": round(mean_a, 6),
                       "mean_b": round(mean_b, 6), "delta": round(delta, 6),
                       "direction": direction, "min_gap": min_gap, "ok": ok})
        if not ok:
            return REFUTED, (f"the group mean difference ({delta:.4g}) does "
                             f"not meet {direction} {min_gap}")
        return VERIFIED, None

    checks.append({"check": "assertion_supported", "source": FRAMEWORK,
                   "kind": kind, "problem": "unsupported assertion kind"})
    return INSUFFICIENT, (f"assertion kind {kind!r} is not computable by the "
                          "framework; only probe/status/comparison are "
                          "evaluated")


def _pair_by_task(side_a: List[Dict[str, Any]],
                  side_b: List[Dict[str, Any]]
                  ) -> Tuple[List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
                             List[Dict[str, Any]]]:
    """Pair records that share a task, one per side (most recent wins).

    Records without a counterpart on the other side are returned separately:
    they are NOT silently mixed into a paired statistic."""
    by_task_b: Dict[str, List[Dict[str, Any]]] = {}
    for fact in side_b:
        by_task_b.setdefault(fact["task_id"], []).append(fact)
    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    used_b: Set[str] = set()
    unpaired: List[Dict[str, Any]] = []
    for fact_a in side_a:
        candidates = [f for f in by_task_b.get(fact_a["task_id"], [])
                      if f["execution_id"] not in used_b]
        if not candidates:
            unpaired.append(fact_a)
            continue
        fact_b = candidates[-1]
        used_b.add(fact_b["execution_id"])
        pairs.append((fact_a["task_id"], fact_a, fact_b))
    for fact_b in side_b:
        if fact_b["execution_id"] not in used_b:
            unpaired.append(fact_b)
    return pairs, unpaired


def verify_relation(claim: str,
                    *, evidence: Sequence[Any] = (),
                    roles: Optional[Sequence[Dict[str, str]]] = None,
                    assertions: Optional[Sequence[Dict[str, Any]]] = None,
                    purpose: str = PURPOSE_RELATION
                    ) -> Dict[str, Any]:
    """Evaluate a structured relation claim's declared assertions.

    The framework computes ONLY what the claim declares as checkable; it does
    not parse natural language and does not promise to detect that a sentence
    overreaches its evidence. ``roles`` names the part each referenced
    execution plays in THIS claim (free strings), and every assertion reads
    evidence through those roles.

    Reused machinery (never a second verifier): :func:`_as_fact`
    normalization, the distinct/identified guards, the precondition discipline
    (:data:`PRECONDITION_CHECKS`), and the complete-metric rule. The result
    carries a ``scope`` block naming exactly which executions and assertions
    the verdict covered — ``verified`` means "no violation was found within
    this scope", never "true for every future task".
    """
    records = [_as_fact(r) for r in evidence]
    role_list = [dict(r) for r in (roles or [])]
    assertion_list = [dict(a) for a in (assertions or [])]
    declared = [str(a.get("kind")) for a in assertion_list]

    if not records:
        return _relation_report(INSUFFICIENT, claim, [],
                                assertion_list, scope={},
                                conclusion=("no evidence was referenced, so "
                                            "no assertion could run"))
    dupes = _duplicate_ids(records)
    if dupes:
        return _relation_report(
            INSUFFICIENT, claim, [], assertion_list,
            scope={"evidence": [f["execution_id"] for f in records]},
            conclusion=("the same execution was referenced more than once, so "
                        "it is not independent evidence: "
                        + ", ".join(sorted(dupes))))
    if not all(_identified(f) for f in records):
        return _relation_report(
            INSUFFICIENT, claim, [], assertion_list,
            scope={"evidence": [f["execution_id"] for f in records]},
            conclusion=("an execution carries no execution_id, so the evidence "
                        "cannot be retraced and is not a check"))
    # Role partition: every referenced execution must have a role, and every
    # assertion reads through those roles.
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    role_of = {str(r.get("execution_id")): str(r.get("role"))
               for r in role_list}
    for fact in records:
        role = role_of.get(fact["execution_id"])
        if not role:
            return _relation_report(
                INSUFFICIENT, claim, [], assertion_list,
                scope={"evidence": [f["execution_id"] for f in records]},
                conclusion=(f"execution {fact['execution_id']} has no role "
                            "declared, so the framework cannot know what part "
                            "it plays in the claim"))
        by_role.setdefault(role, []).append(fact)
    scope = {
        "evidence": [{"execution_id": f["execution_id"],
                      "role": role_of.get(f["execution_id"]),
                      "task_id": f["task_id"],
                      "strategy_id": f["strategy_id"]}
                     for f in records],
        "tasks": sorted({f["task_id"] for f in records if f["task_id"]}),
        "roles": sorted(by_role),
        "assertions_declared": declared,
    }
    if not assertion_list:
        return _relation_report(
            INSUFFICIENT, claim, [], assertion_list, scope=scope,
            conclusion=("no assertion was declared: the framework has nothing "
                        "it can compute, so the claim stays unverified — "
                        "declare probe/status/comparison assertions for the "
                        "parts of the claim that are checkable"))

    checks: List[Dict[str, Any]] = []
    usable = [f for f in records if _usable(f)]
    if not usable:
        checks.append({"check": "execution_completed", "source": FRAMEWORK,
                       "observed": False})
        return _relation_report(
            INSUFFICIENT, claim, checks, assertion_list, scope=scope,
            conclusion=("no referenced execution produced a usable result, so "
                        "the assertions could not run — not refuted"))
    verified_assertions: List[int] = []
    for index, assertion in enumerate(assertion_list):
        state, reason = _assertion_checks(assertion, by_role, checks)
        if state == INSUFFICIENT:
            scope["assertions_checked"] = verified_assertions
            scope["assertions_unchecked"] = [
                i for i in range(len(assertion_list))
                if i not in verified_assertions and i != index]
            return _relation_report(INSUFFICIENT, claim, checks,
                                    assertion_list, scope=scope,
                                    conclusion=("an assertion could not be "
                                                "decided: " + str(reason)))
        if state == REFUTED:
            scope["assertions_checked"] = verified_assertions
            scope["assertions_unchecked"] = [
                i for i in range(len(assertion_list))
                if i not in verified_assertions and i != index]
            return _relation_report(REFUTED, claim, checks, assertion_list,
                                    scope=scope,
                                    conclusion=("the claim did not hold on "
                                                "real evidence: " + str(reason)))
        verified_assertions.append(index)
    scope["assertions_checked"] = verified_assertions
    scope["assertions_unchecked"] = []
    return _relation_report(
        VERIFIED, claim, checks, assertion_list, scope=scope,
        conclusion=("every declared assertion held over the referenced "
                    "evidence; this is a verified claim WITHIN THIS SCOPE — "
                    "no violation was found among the referenced executions, "
                    "not a guarantee about every future task"))


def _relation_report(state: str, claim: str, checks: List[Dict[str, Any]],
                     assertions: List[Dict[str, Any]], *,
                     scope: Dict[str, Any],
                     conclusion: str) -> Dict[str, Any]:
    """The relation verdict, carrying its own scope and audit trail."""
    return {
        "state": state,
        "purpose": PURPOSE_RELATION,
        "claim": claim,
        "assertions": assertions,
        "checks": checks,
        "evidence": [str(e.get("execution_id")) for e in
                     (scope.get("evidence") or [])
                     if isinstance(e, dict) and e.get("execution_id")],
        "scope": scope,
        "conclusion": conclusion,
        "verified_at": time.time(),
    }


# ---------------------------------------------------------------------------
# TASK-RESULT checks: does the answer satisfy the ORIGINAL task?
# ---------------------------------------------------------------------------
#
# A different question from everything above. Admission verification asks
# "does this strategic claim hold"; the executor's own check asks "did the
# solver reach a legal status with a finite objective". NEITHER asks "is this
# answer actually a valid answer to the task", and that gap is how a relaxed
# LP solution (an optimal fractional answer to an integer problem) becomes a
# positive quality sample and poisons recall, statistics, world-model
# feedback and offline induction.
#
# Scope discipline (the same red lines, restated for this layer):
#
# 1. **A solver's optimality is not task correctness.** `optimal` + `gap=0`
#    says the answer is best for the MODEL AS WRITTEN. A model with the wrong
#    variable domain, the wrong objective or a missing constraint is
#    optimally wrong. Only a check the harness declares is evaluated here.
# 2. **No gold is not a pass.** With no check basis the honest verdict is
#    `insufficient` — never "matched", never "probably fine", and never a
#    demand that the user supply a reference.
# 3. **The framework does not understand natural language constraints.** It
#    evaluates the check bases that are computable (a reference value, a
#    status, declared integer domains, an objective recomputation, explicit
#    value probes). Anything not declared is listed as UNCHECKED in the
#    report's scope, so "passed" is never read as "fully validated".

#: Task-result verdicts. Deliberately NOT the admission vocabulary: a task
#: check answers "is this answer acceptable", not "is this claim true".
TASK_CHECK_PASSED = "passed"
TASK_CHECK_FAILED = "failed"
TASK_CHECK_INSUFFICIENT = "insufficient"

#: Declared intent of an execution that must not be read as a plain success.
INTENT_RELAXATION = "relaxation"
INTENT_INTERMEDIATE = "intermediate"
TASK_INTENTS = (INTENT_RELAXATION, INTENT_INTERMEDIATE)

#: What a task check can never establish, whatever passes. Always reported in
#: ``scope.unchecked`` so a `passed` verdict is not over-read.
_TASK_CHECK_UNCHECKED_ALWAYS = (
    "whether the MODEL represents the task (a correct answer to a wrong "
    "model is still wrong)",
    "constraint satisfaction not declared as a probe or a recomputation",
)


def _solution_variables(fact: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """The recorded solution vector of a fact: ``(present, values)``.

    Read from ``execution_features.solution_variables`` — the executor stores
    the script-reported ``result.json`` ``variables`` map there. Absent means
    absent: a check needing a variable reports it as missing rather than
    treating the solution as empty (an empty solution would silently satisfy
    an "all variables are integers" test)."""
    payload = fact.get("payload") or {}
    features = payload.get("execution_features") or {}
    raw = features.get("solution_variables")
    if not isinstance(raw, dict) or not raw:
        return False, {}
    return True, dict(raw)


def _variable_value(values: Dict[str, Any],
                    name: str) -> Tuple[bool, Any]:
    """Resolve one variable name, supporting dotted paths into a nested map."""
    return _resolve(values, str(name))


def _task_check_report(state: str, *, execution_id: str, checks: List[Dict[str, Any]],
                       diffs: List[Dict[str, Any]], basis: List[str],
                       unchecked: List[str], intent: Optional[str],
                       conclusion: str) -> Dict[str, Any]:
    """The task-result verdict: identity, scope, checks, diffs, unchecked."""
    return {
        "state": state,
        "execution_id": execution_id,
        "checks": checks,
        "diffs": diffs,
        "scope": {
            "basis": basis,
            "unchecked": unchecked,
        },
        "intent": intent,
        "conclusion": conclusion,
        "checked_at": time.time(),
    }


def _task_check_declared(check: Dict[str, Any]) -> List[str]:
    """Which check bases the harness actually declared."""
    declared: List[str] = []
    if _finite(check.get("reference_objective")) is not None:
        declared.append("reference_objective")
    if check.get("reference_status") is not None:
        declared.append("reference_status")
    if check.get("integer") is not None:
        declared.append("integer")
    if check.get("recompute_objective") is not None:
        declared.append("recompute_objective")
    if check.get("semantic_probe") is not None:
        declared.append("semantic_probe")
    return declared


def verify_task_result(execution: Any,
                       check: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """Check whether ONE execution's answer satisfies the original task.

    ``check`` is the harness's declaration of what is checkable. Every base is
    optional; the framework evaluates ONLY what is declared and reports the
    rest as unchecked:

    - ``reference_objective`` (+ optional ``tolerance``): the reported
      objective must agree with the reference. The tolerance rule is the SAME
      one admission verification uses (``1e-6 * max(1, |reference|)`` unless
      overridden) — one rule, one place.
    - ``reference_status``: the reported solver status must equal it.
    - ``integer``: ``{"variables": [names] | omitted, "tolerance": t}`` —
      every named variable (or every recorded variable) must be integral
      within ``t``. This is the check that catches an LP relaxation answered
      with fractional values.
    - ``recompute_objective``: ``{"coefficients": {name: c}, "constant": k,
      "tolerance": t}`` — the objective is RECOMPUTED from the recorded
      solution vector and compared with the reported one.
    - ``semantic_probe``: one or more ``{"path", equals|min|max|in}`` probes
      over the record payload (the same probe evaluator admission uses).

    A ``failed`` verdict means a declared check ran on real values and did not
    hold. ``insufficient`` means the check could not be decided (no basis
    declared, no solution vector, a needed variable missing, the execution
    produced no usable result) — which is NOT a pass and NOT a failure. A
    ``passed`` verdict covers only the declared bases: the report always names
    what it did not check.
    """
    fact = _as_fact(execution)
    execution_id = fact["execution_id"]
    check = dict(check or {})
    intent = check.get("intent")
    if intent is not None and str(intent) not in TASK_INTENTS:
        raise ValueError(
            f"intent must be one of {TASK_INTENTS} or omitted (got {intent!r})")
    intent = str(intent) if intent is not None else None
    unchecked = list(_TASK_CHECK_UNCHECKED_ALWAYS)
    if intent == INTENT_RELAXATION:
        unchecked.append(
            "the harness declared this a deliberate relaxation: its answer is "
            "not the task's answer, so a pass here does not make it one")
    elif intent == INTENT_INTERMEDIATE:
        unchecked.append(
            "the harness declared this an intermediate solve: it is a step "
            "toward the answer, not the answer")

    if not execution_id:
        return _task_check_report(
            TASK_CHECK_INSUFFICIENT, execution_id="", checks=[], diffs=[],
            basis=[], unchecked=unchecked, intent=intent,
            conclusion=("the execution carries no execution_id, so the check "
                        "cannot be attached to a fact and is not a check"))

    declared = _task_check_declared(check)
    if not declared:
        return _task_check_report(
            TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
            checks=[{"check": "task_check_basis", "source": FRAMEWORK,
                     "declared": []}], diffs=[], basis=[],
            unchecked=unchecked + ["every check base (none was declared)"],
            intent=intent,
            conclusion=("no check basis was declared: the framework has "
                        "nothing it can compute, so the answer's validity is "
                        "UNKNOWN — feasibility and optimality are properties "
                        "of the solver's own model, not of the task. Declare "
                        "reference_objective / reference_status / integer / "
                        "recompute_objective / semantic_probe as applicable"))

    if not _usable(fact):
        return _task_check_report(
            TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
            checks=[{"check": "execution_completed", "source": FRAMEWORK,
                     "observed": False,
                     "status": fact["quality"].get("status")}],
            diffs=[], basis=declared, unchecked=unchecked, intent=intent,
            conclusion=("the execution produced no usable result, so the "
                        "declared check could not run — this is not a failure "
                        "of the answer"))

    checks: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []
    ran = 0

    # -- reference status -----------------------------------------------------
    reference_status = check.get("reference_status")
    if reference_status is not None:
        ran += 1
        status = str(fact["quality"].get("status", ""))
        ok = status == str(reference_status)
        checks.append({"check": "reference_status", "source": FRAMEWORK,
                       "observed": status, "expected": reference_status,
                       "ok": ok})
        if not ok:
            diffs.append({"basis": "reference_status", "observed": status,
                          "expected": reference_status,
                          "reason": (f"the solver finished {status!r}, not the "
                                     f"declared {reference_status!r}")})
            return _task_check_report(
                TASK_CHECK_FAILED, execution_id=execution_id, checks=checks,
                diffs=diffs, basis=declared, unchecked=unchecked,
                intent=intent,
                conclusion=(f"the answer's status {status!r} does not match "
                            f"the declared {reference_status!r}"))

    # -- reference objective (ONE tolerance rule) ------------------------------
    reference = _finite(check.get("reference_objective"))
    if reference is not None:
        ran += 1
        objective = _finite(fact["quality"].get("objective"))
        tolerance = _finite(check.get("tolerance"))
        tol = (tolerance if tolerance is not None
               else 1e-6 * max(1.0, abs(reference)))
        gap = None if objective is None else abs(objective - reference)
        ok = objective is not None and gap <= tol
        checks.append({"check": "reference_objective", "source": FRAMEWORK,
                       "observed": objective, "reference": reference,
                       "tolerance": tol, "abs_diff": gap, "ok": ok})
        if not ok:
            diffs.append({"basis": "reference_objective", "observed": objective,
                          "expected": reference, "tolerance": tol,
                          "abs_diff": gap,
                          "reason": ("no objective value was reported"
                                     if objective is None else
                                     f"the objective differs from the reference "
                                     f"by {gap:.6g}, outside the tolerance "
                                     f"{tol:.6g}")})
            return _task_check_report(
                TASK_CHECK_FAILED, execution_id=execution_id, checks=checks,
                diffs=diffs, basis=declared, unchecked=unchecked,
                intent=intent,
                conclusion=(f"the objective does not match the reference "
                            f"within {tol:.6g}"))

    # -- integer domains (the LP-relaxation trap) ------------------------------
    integer_spec = check.get("integer")
    if integer_spec is not None:
        ran += 1
        spec = integer_spec if isinstance(integer_spec, dict) else {}
        present, values = _solution_variables(fact)
        if not present:
            checks.append({"check": "integer_domains", "source": FRAMEWORK,
                           "ran": False,
                           "problem": "no solution vector was recorded"})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + ["integer domains (no solution vector "
                                       "was recorded)"],
                intent=intent,
                conclusion=("the integer-domain check needs the recorded "
                            "solution vector and this execution carries none "
                            "(the script must report `variables` in "
                            "result.json) — the answer's domain validity is "
                            "UNKNOWN, not confirmed"))
        names = spec.get("variables")
        if names is None:
            names = sorted(values)
            scope_note = "every recorded variable"
        else:
            names = [str(n) for n in names]
            scope_note = "the declared variables"
        tol = _finite(spec.get("tolerance"))
        tol = 1e-6 if tol is None else tol
        missing = [n for n in names if not _variable_value(values, n)[0]]
        if missing:
            checks.append({"check": "integer_domains", "source": FRAMEWORK,
                           "ran": False, "missing_variables": missing})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + [
                    "integer domains for " + ", ".join(missing)
                    + " (absent from the recorded solution vector)"],
                intent=intent,
                conclusion=("the integer-domain check names variables the "
                            "solution vector does not contain ("
                            + ", ".join(missing)
                            + "): the answer's domain validity is UNKNOWN"))
        non_integer: List[Dict[str, Any]] = []
        for name in names:
            found, raw = _variable_value(values, name)
            value = _finite(raw)
            if not found or value is None:
                # A non-numeric value (a string label, a nested structure) is
                # not integral and not comparable: recorded, never coerced.
                non_integer.append({"variable": name, "value": raw,
                                    "fractional_part": None,
                                    "reason": "the value is not numeric"})
                continue
            fractional = abs(value - round(value))
            if fractional > tol:
                non_integer.append({"variable": name, "value": value,
                                    "fractional_part": round(fractional, 12),
                                    "reason": (f"{value} is not an integer "
                                               f"(fractional part "
                                               f"{fractional:.6g})")})
        checks.append({"check": "integer_domains", "source": FRAMEWORK,
                       "ran": True, "scope": scope_note,
                       "n_variables": len(names),
                       "n_non_integer": len(non_integer),
                       "tolerance": tol,
                       "ok": not non_integer})
        if non_integer:
            diffs.extend({"basis": "integer_domains", **item}
                         for item in non_integer)
            return _task_check_report(
                TASK_CHECK_FAILED, execution_id=execution_id, checks=checks,
                diffs=diffs, basis=declared, unchecked=unchecked,
                intent=intent,
                conclusion=(f"{len(non_integer)} of {len(names)} "
                            f"{scope_note} are not integral: this is not a "
                            "valid answer to an integer task, whatever the "
                            "solver's own optimality says"))

    # -- objective recomputation from the solution -----------------------------
    recompute = check.get("recompute_objective")
    if recompute is not None:
        ran += 1
        spec = recompute if isinstance(recompute, dict) else {}
        coefficients = spec.get("coefficients") or {}
        if not isinstance(coefficients, dict) or not coefficients:
            checks.append({"check": "recompute_objective", "source": FRAMEWORK,
                           "ran": False,
                           "problem": "no coefficients were declared"})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + ["objective recomputation (no "
                                       "coefficients declared)"],
                intent=intent,
                conclusion=("the recomputation declares no coefficients, so "
                            "the objective could not be recomputed"))
        present, values = _solution_variables(fact)
        if not present:
            checks.append({"check": "recompute_objective", "source": FRAMEWORK,
                           "ran": False,
                           "problem": "no solution vector was recorded"})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + ["objective recomputation (no solution "
                                       "vector was recorded)"],
                intent=intent,
                conclusion=("the recomputation needs the recorded solution "
                            "vector and this execution carries none"))
        missing = [str(n) for n in coefficients
                   if not _variable_value(values, str(n))[0]]
        if missing:
            checks.append({"check": "recompute_objective", "source": FRAMEWORK,
                           "ran": False, "missing_variables": sorted(missing)})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + ["objective recomputation (variables "
                                       "absent from the solution vector: "
                                       + ", ".join(sorted(missing)) + ")"],
                intent=intent,
                conclusion=("the recomputation needs variables the solution "
                            "vector does not contain, so it could not run"))
        total = _finite(spec.get("constant")) or 0.0
        unreadable: List[str] = []
        for name, coefficient in coefficients.items():
            _, raw = _variable_value(values, str(name))
            value = _finite(raw)
            coef = _finite(coefficient)
            if value is None or coef is None:
                unreadable.append(str(name))
                continue
            total += coef * value
        if unreadable:
            checks.append({"check": "recompute_objective", "source": FRAMEWORK,
                           "ran": False, "unreadable_variables": sorted(unreadable)})
            return _task_check_report(
                TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                checks=checks, diffs=diffs, basis=declared,
                unchecked=unchecked + ["objective recomputation (a "
                                       "coefficient or value was not numeric: "
                                       + ", ".join(sorted(unreadable)) + ")"],
                intent=intent,
                conclusion=("a declared coefficient or solution value was not "
                            "numeric, so the objective could not be recomputed"))
        reported = _finite(fact["quality"].get("objective"))
        tol = _finite(spec.get("tolerance"))
        tol = (tol if tol is not None
               else 1e-6 * max(1.0, abs(reported if reported is not None
                                        else total)))
        diff = None if reported is None else abs(reported - total)
        ok = reported is not None and diff <= tol
        checks.append({"check": "recompute_objective", "source": FRAMEWORK,
                       "recomputed": round(total, 9), "reported": reported,
                       "tolerance": tol, "abs_diff": diff, "ok": ok})
        if not ok:
            diffs.append({"basis": "recompute_objective",
                          "observed": reported, "expected": round(total, 9),
                          "tolerance": tol, "abs_diff": diff,
                          "reason": ("no objective value was reported"
                                     if reported is None else
                                     f"the reported objective differs from the "
                                     f"value recomputed from the solution by "
                                     f"{diff:.6g}, outside the tolerance "
                                     f"{tol:.6g}: the answer does not satisfy "
                                     f"the declared objective")})
            return _task_check_report(
                TASK_CHECK_FAILED, execution_id=execution_id, checks=checks,
                diffs=diffs, basis=declared, unchecked=unchecked,
                intent=intent,
                conclusion=("the objective recomputed from the recorded "
                            "solution does not match the reported one: the "
                            "answer does not satisfy the declared objective"))

    # -- value probes (the same evaluator admission uses) ----------------------
    probe = check.get("semantic_probe")
    if probe is not None:
        for item in (probe if isinstance(probe, list) else [probe]):
            ran += 1
            result = _evaluate_probe(fact, dict(item or {}))
            result["check"] = "semantic_probe"
            checks.append(result)
            if result.get("ok") is None:
                return _task_check_report(
                    TASK_CHECK_INSUFFICIENT, execution_id=execution_id,
                    checks=checks, diffs=diffs, basis=declared,
                    unchecked=unchecked + [f"probe {result.get('path')!r} "
                                           "(declares no comparison)"],
                    intent=intent,
                    conclusion=(f"the probe {result.get('path')!r} declares "
                                "no comparison, so it could not decide"))
            if not result["ok"]:
                diffs.append({"basis": "semantic_probe",
                              "path": result.get("path"),
                              "observed": result.get("observed"),
                              "expected": (result.get("expected")
                                           if "expected" in result
                                           else {"min": result.get("min"),
                                                 "max": result.get("max")}),
                              "reason": (f"the value at {result.get('path')!r} "
                                         "does not satisfy the declared "
                                         "comparison")})
                return _task_check_report(
                    TASK_CHECK_FAILED, execution_id=execution_id,
                    checks=checks, diffs=diffs, basis=declared,
                    unchecked=unchecked, intent=intent,
                    conclusion=(f"the declared probe "
                                f"{result.get('path')!r} failed on the "
                                "recorded values"))

    if ran == 0:
        # Unreachable while `declared` is non-empty, but kept explicit: a
        # silent `passed` with nothing evaluated is the exact failure mode
        # this layer exists to prevent.
        return _task_check_report(
            TASK_CHECK_INSUFFICIENT, execution_id=execution_id, checks=checks,
            diffs=diffs, basis=declared, unchecked=unchecked, intent=intent,
            conclusion=("no declared check actually ran, so the answer's "
                        "validity is UNKNOWN"))
    return _task_check_report(
        TASK_CHECK_PASSED, execution_id=execution_id, checks=checks,
        diffs=[], basis=declared, unchecked=unchecked, intent=intent,
        conclusion=("every declared check passed on the recorded values. This "
                    "covers the declared bases only: it is not a proof that "
                    "the model represents the task, and the unchecked items "
                    "above are still unknown"))


__all__ = ["verify_candidate", "verify_relation", "verify_task_result",
           "VERIFIED", "INSUFFICIENT",
           "REFUTED", "PURPOSE_RULE", "PURPOSE_REPAIR", "PURPOSE_COST_SAVING",
           "PURPOSE_RELATION", "TASK_CHECK_PASSED", "TASK_CHECK_FAILED",
           "TASK_CHECK_INSUFFICIENT", "TASK_INTENTS", "INTENT_RELAXATION",
           "INTENT_INTERMEDIATE"]