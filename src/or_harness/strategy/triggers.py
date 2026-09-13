"""Induction trigger criteria C1-C6.

Checked cheaply and automatically after every ``record``; any hit produces an
induction_hint with a concrete evidence structure (never a bare counter). The
criteria are OR-ed — there is no "all satisfied" state machine. Hints never
induce by themselves: ``induce`` is the harness's explicit call, and the
harness may also induct from its own business knowledge (criteria do not
monopolize induction).

Divergence from priors/entries requires n >= 2 (single observations never
count as divergence — that restraint is by design).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

from or_harness.core.schema import CostVector, ExecutionRecord, Strategy, group_key
from or_harness.strategy.stats import ConditionalStats, GroupStats, quality_score

MIN_DIVERGENCE_N = 2
SIGNIFICANT_QUALITY_DELTA = 0.10
SIGNIFICANT_COST_RATIO = 0.25  # one side >= 25% cheaper counts as a cost gap
TREND_MIN_N = 3
STABLE_SUCCESS_MIN_N = 4

#: Quality level above which a strategy's performance counts as "high" and
#: below which it counts as "low" (in the [0, 1] quality-score space).
QUALITY_HIGH_THRESHOLD = 0.75
QUALITY_LOW_THRESHOLD = 0.35


@dataclass
class InductionHint:
    criterion: str  # "C1".."C6"
    strategy_ids: List[str]
    group_key: str
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "strategy_ids": list(self.strategy_ids),
            "group_key": self.group_key,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def check_triggers(record: ExecutionRecord, stats: ConditionalStats,
                   catalog: Dict[str, Strategy],
                   entries_expected: Optional[Dict[str, Dict[str, float]]] = None,
                   prior_failures: Optional[List[ExecutionRecord]] = None
                   ) -> List[InductionHint]:
    """Evaluate C1-C6 for the group of ``record`` after it was appended.

    ``entries_expected``: optional {strategy_id: {"quality": q}} of matching
    strategic entries, so C1/C2 treat "memory already encodes this" as
    non-divergent.

    ``prior_failures``: same-task failed executions (from the Experience
    Bank and/or the pending staging area) for cross-execution recovery
    detection in C4.

    Note: triggers no longer reference catalog priors (which have been
    removed). All criteria are now purely statistical — they detect
    patterns in observed data, not divergence from fabricated baselines.
    """
    group = record.group_l1
    cells = stats.group(group)
    hints: List[InductionHint] = []
    expected_map = dict(entries_expected or {})

    hint = _c1_strategy_contrast(cells, expected_map, group)
    if hint:
        hints.append(hint)
    hint = _c2_extreme_performance(record, cells, group, expected_map)
    if hint:
        hints.append(hint)
    hint = _c3_drift(record, cells, group)
    if hint:
        hints.append(hint)
    hint = _c4_failure_recovery(record, group, prior_failures or [])
    if hint:
        hints.append(hint)
    hints.extend(_c5_cross_family(record, stats, expected_map))
    hint = _c6_stable_success(cells, group)
    if hint:
        hints.append(hint)
    return hints


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def _c1_strategy_contrast(cells: Dict[str, GroupStats],
                          expected_map: Dict[str, float],
                          group: str
                          ) -> Optional[InductionHint]:
    """C1: >= 2 strategies in one group differ significantly (quality or cost).

    A difference that existing strategic entries already encode (via
    ``expected_map``) does NOT trigger — the memory already captured it.
    Without priors, the trigger fires on the first significant contrast
    that is not already in the strategic layer.
    """
    eligible = [c for c in cells.values() if c.n >= MIN_DIVERGENCE_N]
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            dq = a.mean_quality - b.mean_quality
            q_gap = abs(dq)
            # Check if existing entries already encode this contrast.
            entry_a = expected_map.get(a.strategy_id, {}).get("quality")
            entry_b = expected_map.get(b.strategy_id, {}).get("quality")
            if entry_a is not None and entry_b is not None:
                entry_gap = entry_a - entry_b
                if abs(entry_gap - dq) < SIGNIFICANT_QUALITY_DELTA / 2:
                    continue  # already encoded
            quality_contrast = q_gap >= SIGNIFICANT_QUALITY_DELTA
            cost_contrast = False
            cost_evidence: Dict[str, Any] = {}
            for dim in ("llm_tokens", "solver_runtime_s"):
                if a.n_measured.get(dim, 0) == 0 or b.n_measured.get(dim, 0) == 0:
                    continue  # unknown on either side never drives a contrast
                ca = getattr(a.mean_cost, dim)
                cb = getattr(b.mean_cost, dim)
                lo, hi = min(ca, cb), max(ca, cb)
                observed_gap = hi > 0 and (hi - lo) / hi >= SIGNIFICANT_COST_RATIO
                if not observed_gap:
                    continue
                # Decision-changing? Either quality is tied (cost is the only
                # basis of choice) or the cheaper side also wins on quality.
                cheaper_is_a = ca < cb
                cheaper_mean_q = a.mean_quality if cheaper_is_a else b.mean_quality
                pricier_mean_q = b.mean_quality if cheaper_is_a else a.mean_quality
                decision_changing = (q_gap < SIGNIFICANT_QUALITY_DELTA
                                     or cheaper_mean_q > pricier_mean_q)
                if not decision_changing:
                    continue
                cost_contrast = True
                cost_evidence = {
                    "dimension": dim,
                    "observed": {a.strategy_id: round(ca, 4),
                                 b.strategy_id: round(cb, 4)},
                }
                break
            if not (quality_contrast or cost_contrast):
                continue
            kind = "quality" if quality_contrast else "cost"
            return InductionHint(
                criterion="C1",
                strategy_ids=[a.strategy_id, b.strategy_id],
                group_key=group,
                reason=(f"{kind} contrast between strategies: "
                        f"{a.strategy_id} meanQ={a.mean_quality:.2f} vs "
                        f"{b.strategy_id} meanQ={b.mean_quality:.2f}"),
                evidence={
                    "kind": kind,
                    "observed_quality": {a.strategy_id: round(a.mean_quality, 4),
                                         b.strategy_id: round(b.mean_quality, 4)},
                    "cost": cost_evidence,
                    "n": {a.strategy_id: a.n, b.strategy_id: b.n},
                    "execution_ids": {a.strategy_id: a.execution_ids,
                                      b.strategy_id: b.execution_ids},
                })
    return None


def _c2_extreme_performance(record: ExecutionRecord, cells: Dict[str, GroupStats],
                         group: str,
                         expected_map: Dict[str, Dict[str, float]]
                         ) -> Optional[InductionHint]:
    """C2: a strategy's observed performance is extreme (very high or very low)
    with n >= 2, and existing entries don't already capture it.

    Without fabricated priors, the trigger fires on observed extremes — the
    first evidence that a strategy is notably good or bad in this structural
    group, worth consolidating into a strategic entry.
    """
    cell = cells.get(record.strategy_id)
    if cell is None or cell.n < MIN_DIVERGENCE_N:
        return None
    # Already encoded by an existing entry?
    entry_q = expected_map.get(record.strategy_id, {}).get("quality")
    if entry_q is not None and abs(entry_q - cell.mean_quality) < SIGNIFICANT_QUALITY_DELTA:
        return None
    mean_q = cell.mean_quality
    if mean_q >= QUALITY_HIGH_THRESHOLD:
        direction = "high"
    elif mean_q <= QUALITY_LOW_THRESHOLD:
        direction = "low"
    else:
        return None
    return InductionHint(
        criterion="C2", strategy_ids=[record.strategy_id], group_key=group,
        reason=(f"{record.strategy_id} performs {direction}: "
                f"meanQ={mean_q:.2f} over n={cell.n}"),
        evidence={"observed_mean_quality": round(mean_q, 4),
                  "direction": direction, "n": cell.n,
                  "execution_ids": list(cell.execution_ids)})


def _c3_drift(record: ExecutionRecord, cells: Dict[str, GroupStats],
              group: str) -> Optional[InductionHint]:
    """C3: same strategy, same group, n >= 3 with a quality trend."""
    cell = cells.get(record.strategy_id)
    if cell is None or cell.n < TREND_MIN_N:
        return None
    trend = cell.quality_trend()
    if abs(trend) < SIGNIFICANT_QUALITY_DELTA / 2:
        return None
    return InductionHint(
        criterion="C3", strategy_ids=[record.strategy_id], group_key=group,
        reason=(f"quality trend within group: {trend:+.3f} over n={cell.n}"),
        evidence={"trend": round(trend, 4), "n": cell.n,
                  "quality_series": [round(q, 4) for q in cell.quality_values],
                  "execution_ids": list(cell.execution_ids)})


def _c4_failure_recovery(record: ExecutionRecord, group: str,
                         prior_failures: Optional[List[ExecutionRecord]] = None
                         ) -> Optional[InductionHint]:
    """C4: a fallback was actually triggered. Failure evidence is the most
    valuable induction raw material.

    Two detection paths:
    1. within-execution: ``failures[].recovery_action`` set on this record.
    2. cross-execution: this record succeeded while a same-task earlier
       execution failed under a DIFFERENT solver — the recovery chain
       (failed -> switched solver -> succeeded) emerges from two independent
       facts, so the harness never needs to narrate it into a record.
       Retrying the same solver is not a recovery chain.
    """
    triggered = [f for f in record.failures if f.recovery_action]
    if triggered:
        return InductionHint(
            criterion="C4", strategy_ids=[record.strategy_id], group_key=group,
            reason="fallback recovery was exercised in this execution",
            evidence={"execution_id": record.execution_id,
                      "failures": [f.to_dict() for f in triggered],
                      "final_status": record.quality.get("status")})

    if prior_failures and record.quality.get("feasible"):
        this_solver = str((record.solver or {}).get("name", ""))
        for failed in prior_failures:
            failed_solver = str((failed.solver or {}).get("name", ""))
            if failed_solver and failed_solver != this_solver:
                return InductionHint(
                    criterion="C4", strategy_ids=[record.strategy_id],
                    group_key=group,
                    reason=(f"cross-execution recovery: {failed_solver} failed, "
                            f"switched to {this_solver} and succeeded"),
                    evidence={
                        "kind": "cross_execution_recovery",
                        "failed": {"execution_id": failed.execution_id,
                                   "solver": failed_solver,
                                   "error_class": classify_failure(failed),
                                   "error": _first_error(failed)},
                        "recovered_by": {"execution_id": record.execution_id,
                                         "solver": this_solver},
                    })
    return None


def classify_failure(record: ExecutionRecord) -> str:
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


def _first_error(record: ExecutionRecord) -> str:
    if record.failures:
        return record.failures[0].error
    return str(record.quality.get("status", ""))


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


def _c5_cross_family(record: ExecutionRecord, stats: ConditionalStats,
                     expected_map: Dict[str, Dict[str, float]]
                     ) -> List[InductionHint]:
    """C5: the same strategy shows the same-direction advantage in >= 2
    families — the 'learn once, apply elsewhere' detector.

    Informational: it reports reproduction across independently observed
    families from the data alone. (There is no widening operation to suggest:
    a claim's applicability is read off its own evidence.)"""
    sid = record.strategy_id
    cells = stats.cross_family(sid)
    per_family = [c for c in cells if c.n >= MIN_DIVERGENCE_N]
    if len(per_family) < 2:
        return []
    # Same-direction: all high or all low.
    all_high = all(c.mean_quality >= QUALITY_HIGH_THRESHOLD for c in per_family)
    all_low = all(c.mean_quality <= QUALITY_LOW_THRESHOLD for c in per_family)
    if not (all_high or all_low):
        return []
    direction = "high" if all_high else "low"
    magnitude_spread = max(c.mean_quality for c in per_family) - \
        min(c.mean_quality for c in per_family)
    if magnitude_spread > SIGNIFICANT_QUALITY_DELTA:
        return []

    def _family(cell: GroupStats) -> str:
        return cell.group_key.split("family=")[-1]

    return [InductionHint(
        criterion="C5", strategy_ids=[sid],
        group_key=record.group_l1,
        reason=(f"{direction} performance reproduces independently in "
                f"{len(per_family)} families"),
        evidence={"families": [_family(c) for c in per_family],
                  "direction": direction,
                  "mean_qualities": {_family(c): round(c.mean_quality, 4)
                                     for c in per_family},
                  "execution_ids": {_family(c): c.execution_ids
                                    for c in per_family}})]


def _c6_stable_success(cells: Dict[str, GroupStats],
                       group: str) -> Optional[InductionHint]:
    """C6: same strategy, same group, n >= 4 with zero failures and zero
    retries — consolidation channel for pure success patterns. Retries must
    be MEASURED on every supporting record: unknown retries never count as
    proof of stability."""
    for cell in cells.values():
        if (cell.n >= STABLE_SUCCESS_MIN_N and cell.n_failures == 0
                and cell.n_measured.get("retries", 0) == cell.n
                and cell.total_retries == 0 and cell.mean_quality > 0.0):
            return InductionHint(
                criterion="C6", strategy_ids=[cell.strategy_id], group_key=group,
                reason=(f"stable success: n={cell.n}, zero failures, zero retries, "
                        f"meanQ={cell.mean_quality:.2f}"),
                evidence={"n": cell.n, "mean_quality": round(cell.mean_quality, 4),
                          "mean_cost": {d: round(v, 4) for d, v in
                                        cell.mean_cost.to_dict().items()},
                          "execution_ids": list(cell.execution_ids)})
    return None
