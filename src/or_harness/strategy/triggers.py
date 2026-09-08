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


#: Priors encode "similar" costs within this relative gap; beyond it a cost
#: difference counts as prior-encoded knowledge.
PRIOR_COST_SIMILARITY_RATIO = 0.20


@dataclass
class InductionHint:
    criterion: str  # "C1".."C6"
    strategy_ids: List[str]
    group_key: str
    scope_suggestion: str = "L1"
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "strategy_ids": list(self.strategy_ids),
            "group_key": self.group_key,
            "scope_suggestion": self.scope_suggestion,
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
    non-divergent (same role as priors).

    ``prior_failures``: same-task failed executions (from the Experience
    Bank and/or the pending staging area) for cross-execution recovery
    detection in C4.
    """
    group = record.group_l1
    cells = stats.group(group)
    hints: List[InductionHint] = []
    priors = {sid: s.expected_quality for sid, s in catalog.items()}
    expected_map = dict(priors)
    for sid, exp in (entries_expected or {}).items():
        expected_map[sid] = exp.get("quality", expected_map.get(sid, 0.5))

    hint = _c1_strategy_contrast(cells, expected_map, group,
                                 catalog_costs=catalog)
    if hint:
        hints.append(hint)
    hint = _c2_prior_divergence(record, cells, catalog, group)
    if hint:
        hints.append(hint)
    hint = _c3_drift(record, cells, group)
    if hint:
        hints.append(hint)
    hint = _c4_failure_recovery(record, group, prior_failures or [])
    if hint:
        hints.append(hint)
    hints.extend(_c5_cross_family(record, stats, catalog, expected_map))
    hint = _c6_stable_success(cells, group)
    if hint:
        hints.append(hint)
    return hints


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def _c1_strategy_contrast(cells: Dict[str, GroupStats],
                          expected_map: Dict[str, float],
                          group: str,
                          catalog_costs: Optional[Dict[str, Strategy]] = None
                          ) -> Optional[InductionHint]:
    """C1: >= 2 strategies in one group differ significantly (quality or cost)
    AND the difference contradicts priors/entries. A difference the prior
    already encodes does NOT trigger."""
    eligible = [c for c in cells.values() if c.n >= MIN_DIVERGENCE_N]
    prior_costs = {sid: s.expected_cost for sid, s in catalog_costs.items()} \
        if catalog_costs else {}
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            dq = a.mean_quality - b.mean_quality
            q_gap = abs(dq)
            prior_a = expected_map.get(a.strategy_id, 0.5)
            prior_b = expected_map.get(b.strategy_id, 0.5)
            prior_gap = prior_a - prior_b
            quality_contradicts = (
                q_gap >= SIGNIFICANT_QUALITY_DELTA
                and (abs(prior_gap) < SIGNIFICANT_QUALITY_DELTA / 2
                     or (prior_gap > 0) != (dq > 0)))
            cost_contradicts = False
            cost_evidence: Dict[str, Any] = {}
            for dim in ("llm_tokens", "solver_runtime_s"):
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
                # Prior-consistent? The prior counts as encoding this knowledge
                # only when it shows a same-direction gap beyond the similarity
                # band. Same-sign-but-shallower priors are treated as
                # "costs similar" — a much larger observed gap still teaches.
                prior_ca = getattr(prior_costs.get(a.strategy_id), dim, 0.0)
                prior_cb = getattr(prior_costs.get(b.strategy_id), dim, 0.0)
                plo, phi = min(prior_ca, prior_cb), max(prior_ca, prior_cb)
                prior_gap_ratio = ((phi - plo) / phi) if phi > 0 else 0.0
                prior_same_side = (
                    prior_gap_ratio >= PRIOR_COST_SIMILARITY_RATIO
                    and (prior_ca > prior_cb) == (ca > cb)
                    and prior_gap_ratio >= 0.8 * ((hi - lo) / hi))
                if prior_same_side:
                    continue
                cost_contradicts = True
                cost_evidence = {
                    "dimension": dim,
                    "observed": {a.strategy_id: round(ca, 4),
                                 b.strategy_id: round(cb, 4)},
                    "prior": {a.strategy_id: prior_ca,
                              b.strategy_id: prior_cb},
                }
                break
            if not (quality_contradicts or cost_contradicts):
                continue
            kind = "quality" if quality_contradicts else "cost"
            return InductionHint(
                criterion="C1",
                strategy_ids=[a.strategy_id, b.strategy_id],
                group_key=group,
                reason=(f"{kind} contrast contradicts prior expectations: "
                        f"{a.strategy_id} meanQ={a.mean_quality:.2f} vs "
                        f"{b.strategy_id} meanQ={b.mean_quality:.2f}"),
                evidence={
                    "kind": kind,
                    "observed_quality": {a.strategy_id: round(a.mean_quality, 4),
                                         b.strategy_id: round(b.mean_quality, 4)},
                    "prior_quality": {a.strategy_id: prior_a, b.strategy_id: prior_b},
                    "cost": cost_evidence,
                    "n": {a.strategy_id: a.n, b.strategy_id: b.n},
                    "execution_ids": {a.strategy_id: a.execution_ids,
                                      b.strategy_id: b.execution_ids},
                })
    return None


def _c2_prior_divergence(record: ExecutionRecord, cells: Dict[str, GroupStats],
                         catalog: Dict[str, Strategy],
                         group: str) -> Optional[InductionHint]:
    """C2: one strategy's observed performance systematically departs from its
    built-in prior (n >= 2)."""
    cell = cells.get(record.strategy_id)
    if cell is None or cell.n < MIN_DIVERGENCE_N:
        return None
    strategy = catalog.get(record.strategy_id)
    if strategy is None:
        return None
    prior_q = strategy.expected_quality
    delta = cell.mean_quality - prior_q
    if abs(delta) < SIGNIFICANT_QUALITY_DELTA:
        return None
    direction = "better" if delta > 0 else "worse"
    return InductionHint(
        criterion="C2", strategy_ids=[record.strategy_id], group_key=group,
        reason=(f"{record.strategy_id} performs {direction} than its prior: "
                f"E[gap-quality]={cell.mean_quality:.2f} vs prior {prior_q:.2f}"),
        evidence={"observed_mean_quality": round(cell.mean_quality, 4),
                  "prior_quality": prior_q, "n": cell.n,
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
                     catalog: Dict[str, Strategy],
                     expected_map: Dict[str, float]) -> List[InductionHint]:
    """C5: the same strategy shows the same-direction, same-magnitude
    advantage in >= 2 families with similar structure, contradicting priors.
    The 'learn once, apply elsewhere' detector; suggests widening to L2."""
    sid = record.strategy_id
    cells = stats.cross_family(record.profile_snapshot, sid, level="L2")
    per_family = [c for c in cells if c.n >= MIN_DIVERGENCE_N]
    if len(per_family) < 2:
        return []
    prior = expected_map.get(sid, 0.5)
    deltas = [c.mean_quality - prior for c in per_family]
    same_direction = all(d >= SIGNIFICANT_QUALITY_DELTA for d in deltas) or \
        all(d <= -SIGNIFICANT_QUALITY_DELTA for d in deltas)
    if not same_direction:
        return []
    magnitude_spread = max(deltas) - min(deltas)
    if magnitude_spread > SIGNIFICANT_QUALITY_DELTA:
        return []
    return [InductionHint(
        criterion="C5", strategy_ids=[sid],
        group_key=group_key(record.profile_snapshot, "L2"),
        scope_suggestion="L2",
        reason=(f"advantage reproduces independently in {len(per_family)} "
                "families at similar structure; consider widening to L2"),
        evidence={"families": [c.group_key.split("#family=")[-1] for c in per_family],
                  "deltas_vs_prior": [round(d, 4) for d in deltas],
                  "prior_quality": prior,
                  "execution_ids": {c.group_key.split("#family=")[-1]: c.execution_ids
                                    for c in per_family}})]


def _c6_stable_success(cells: Dict[str, GroupStats],
                       group: str) -> Optional[InductionHint]:
    """C6: same strategy, same group, n >= 4 with zero failures and zero
    retries — consolidation channel for pure success patterns."""
    for cell in cells.values():
        if (cell.n >= STABLE_SUCCESS_MIN_N and cell.n_failures == 0
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
