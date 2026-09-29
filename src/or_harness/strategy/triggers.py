"""Induction-worthy evidence patterns.

Checked cheaply and automatically after every ``record``; any hit produces an
induction_hint with a concrete evidence structure (never a bare counter). The
patterns are OR-ed — there is no "all satisfied" state machine. Hints never
induce by themselves: ``induce`` is the harness's explicit call, and the
harness may also induct from its own business knowledge (the detectors do not
monopolize induction).

Four patterns are worth generalizing, and they are named for what they are —
no historical criterion numbers are used anywhere:

- **strategy_contrast** — structurally comparable evidence shows different
  strategies differing in quality or cost. The lesson is the
  "structural condition -> strategy effect" relation, not one win.
- **intervention_recovery** — a real result changed after an intervention
  (a fallback, a repair, a modeling change, a solver switch). The change is
  evidence, never proof of causation by itself.
- **structural_reproduction** — the same strategy relation recurs in a
  structurally comparable but INDEPENDENT task/family. One task's
  observation is not transferable knowledge.
- **advantage_reversal** — a strategy's advantage weakens, disappears or
  flips as the structural condition changes. The lesson is an applicability
  boundary or a counterexample, not a success count.

Every detector requires n >= 2 supporting executions: a single observation
never counts as a pattern — that restraint is by design.

Hints are EVIDENCE, not orders, and they are persisted onto the record that
produced them (``execution_features.induction_hints``) so the offline
candidate builder can reuse the detector's own cross-execution evidence
references instead of re-deriving them from bare counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    GROUPING_FEATURES,
    ExecutionRecord,
    bin_label,
    group_key,
    normalize_method,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats

#: Evidence below this count never forms a pattern: single observations are
#: not patterns.
MIN_DIVERGENCE_N = 2
SIGNIFICANT_QUALITY_DELTA = 0.10
SIGNIFICANT_COST_RATIO = 0.25  # one side >= 25% cheaper counts as a cost gap

#: Quality level above which a strategy's performance counts as "high" and
#: below which it counts as "low" (in the [0, 1] quality-score space).
QUALITY_HIGH_THRESHOLD = 0.75
QUALITY_LOW_THRESHOLD = 0.35


@dataclass
class InductionHint:
    pattern: str  # "strategy_contrast" | "intervention_recovery" |
    #               "structural_reproduction" | "advantage_reversal"
    strategy_ids: List[str]
    group_key: str
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "strategy_ids": list(self.strategy_ids),
            "group_key": self.group_key,
            "reason": self.reason,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InductionHint":
        """Rebuild a hint from its persisted form (``record``'s
        ``execution_features.induction_hints``).

        Used by the offline candidate builder, which consumes the hint's own
        cross-execution evidence references instead of re-deriving them. An
        unknown pattern is rejected rather than silently carried as a label:
        a stored hint no detector produced is not evidence.
        """
        pattern = str(data.get("pattern", ""))
        if pattern not in PATTERNS:
            raise ValueError(f"unknown induction pattern {pattern!r}")
        return cls(
            pattern=pattern,
            strategy_ids=[str(s) for s in (data.get("strategy_ids") or [])],
            group_key=str(data.get("group_key", "")),
            reason=str(data.get("reason", "")),
            evidence=dict(data.get("evidence") or {}),
        )


#: The four detector names, in one place: the persisted-hint reader refuses
#: anything else, so a label can never be invented on the storage boundary.
PATTERNS = ("strategy_contrast", "intervention_recovery",
            "structural_reproduction", "advantage_reversal")


def evidence_execution_ids(evidence: Dict[str, Any]) -> List[str]:
    """Every execution id a hint's ``evidence`` block refers to, flattened.

    The shapes differ by pattern (a flat ``execution_id``, a
    {strategy: [ids]} map, a {family: [ids]} map, a failed/recovered_by
    pair). The candidate builder needs the SET, because a contrast's two
    sides must travel into one candidate rather than being split into
    unrelated statistical bins.
    """
    found: List[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            for item in value:
                _collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                _collect(item)

    if not isinstance(evidence, dict):
        return found
    _collect(evidence.get("execution_ids"))
    for key in ("execution_id", "failed", "recovered_by"):
        value = evidence.get(key)
        if isinstance(value, dict):
            _collect(value.get("execution_id"))
        elif isinstance(value, str):
            found.append(value)
    seen: List[str] = []
    for eid in found:
        if eid and eid not in seen:
            seen.append(eid)
    return seen


def check_triggers(record: ExecutionRecord, stats: ConditionalStats,
                   entries_expected: Optional[Dict[str, Dict[str, float]]] = None,
                   prior_failures: Optional[List[ExecutionRecord]] = None
                   ) -> List[InductionHint]:
    """Evaluate the four induction-worthy patterns for the structural group
    of ``record`` after it was appended.

    ``entries_expected``: optional {strategy_id: {"quality": q}} of matching
    strategic entries, so strategy_contrast treats "memory already encodes
    this" as non-divergent.

    ``prior_failures``: same-task failed executions (from the Experience
    Bank and/or the pending staging area) for cross-execution recovery
    detection in intervention_recovery.

    Scope: every detector reads the cell the record's profile belongs to
    (:meth:`ConditionalStats.for_profile`), never the whole family — evidence
    from a structurally different region must not drive, dilute, or veto a
    contrast. ``structural_reproduction`` is the only cross-family pattern
    and it is scoped to the same cell in each family;
    ``advantage_reversal`` is the only cross-cell pattern and it stays inside
    one family.

    All detectors are purely statistical — they detect patterns in observed
    data, not divergence from fabricated baselines.
    """
    cells = stats.for_profile(record.profile_snapshot)
    group = group_key(record.profile_snapshot)
    hints: List[InductionHint] = []
    expected_map = dict(entries_expected or {})

    hint = _strategy_contrast(cells, expected_map, group)
    if hint:
        hints.append(hint)
    hint = _intervention_recovery(record, group, prior_failures or [])
    if hint:
        hints.append(hint)
    hints.extend(_structural_reproduction(record, stats))
    hint = _advantage_reversal(record, stats, group)
    if hint:
        hints.append(hint)
    return hints


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------


def _strategy_contrast(cells: Dict[str, GroupStats],
                       expected_map: Dict[str, float],
                       group: str
                       ) -> Optional[InductionHint]:
    """strategy_contrast: >= 2 strategies in one structural cell differ
    significantly in quality OR in cost.

    The lesson is the relation between a structural condition and a strategy's
    effect, not a single win: two strategies observed in the SAME cell with a
    material difference is what makes the comparison meaningful.

    A difference that existing strategic entries already encode (via
    ``expected_map``) does NOT trigger — the memory already captured it.
    """
    eligible = [c for c in cells.values() if c.n >= MIN_DIVERGENCE_N]
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            dq = a.mean_quality - b.mean_quality
            q_gap = abs(dq)
            quality_contrast = q_gap >= SIGNIFICANT_QUALITY_DELTA
            # Existing entries explaining the QUALITY gap says nothing about
            # the COST gap: the two comparisons are independent, so a
            # quality-only dedup must not skip the price comparison. (It used
            # to `continue` here, hiding a 5x-token difference whenever both
            # sides' quality was already encoded.)
            quality_encoded = False
            entry_a = expected_map.get(a.strategy_id, {}).get("quality")
            entry_b = expected_map.get(b.strategy_id, {}).get("quality")
            if entry_a is not None and entry_b is not None:
                entry_gap = entry_a - entry_b
                quality_encoded = abs(entry_gap - dq) < SIGNIFICANT_QUALITY_DELTA / 2
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
            if not quality_contrast and not cost_contrast:
                continue
            if quality_contrast and quality_encoded and not cost_contrast:
                continue  # the only contrast found is already in memory
            kind = "quality" if quality_contrast and not quality_encoded else "cost"
            if kind == "cost" and not cost_contrast:
                continue
            if kind == "quality" and quality_encoded:
                continue
            return InductionHint(
                pattern="strategy_contrast",
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


def _intervention_recovery(record: ExecutionRecord, group: str,
                           prior_failures: Optional[List[ExecutionRecord]] = None
                           ) -> Optional[InductionHint]:
    """intervention_recovery: a real result changed after an intervention.

    Failure evidence is the most valuable induction raw material. The
    intervention may be a fallback, a repair, a MODELING CHANGE or a solver
    switch; what the detector reads is the CHANGE in real outcome plus a
    concrete link to the change, never a narrative.

    Three detection paths, in order of directness:
    1. within-execution: ``failures[].recovery_action`` set on this record
       (the harness itself names the intervention it applied).
    2. cross-execution, SAME solver: this record succeeded while a same-task
       earlier attempt failed AND this record carries evidence of a real
       change — a method receipt whose performed steps differ from the
       failed attempt's, or a recorded change delta. A plain retry of the
       same solver with no evidence of a change does NOT trigger: a success
       on the second try is not a demonstrated recovery.
    3. cross-execution, solver SWITCHED: the recovery chain
       (failed -> switched solver -> succeeded) emerges from two independent
       facts, so the harness never needs to narrate it into a record.

    The hint is evidence, never proof of causation: it names WHAT changed and
    links the two executions, and the verification layer decides whether the
    claim holds.
    """
    triggered = [f for f in record.failures if f.recovery_action]
    if triggered:
        return InductionHint(
            pattern="intervention_recovery",
            strategy_ids=[record.strategy_id], group_key=group,
            reason="fallback recovery was exercised in this execution",
            evidence={"execution_id": record.execution_id,
                      "failures": [f.to_dict() for f in triggered],
                      "final_status": record.quality.get("status")})

    if not (prior_failures and record.quality.get("feasible")):
        return None
    this_solver = str((record.solver or {}).get("name", ""))
    for failed in sorted(prior_failures, key=lambda r: r.created_at,
                         reverse=True):
        if not failed.quality.get("feasible", False):
            failed_solver = str((failed.solver or {}).get("name", ""))
            if failed_solver and failed_solver != this_solver:
                return InductionHint(
                    pattern="intervention_recovery",
                    strategy_ids=[record.strategy_id],
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
    # Same-solver modeling fix: a real CHANGE must be visible, otherwise a
    # plain retry would masquerade as a recovery.
    change = _recorded_change(record, prior_failures)
    if change is None:
        return None
    failed_record = change.pop("failed_record", None)
    return InductionHint(
        pattern="intervention_recovery",
        strategy_ids=[record.strategy_id],
        group_key=group,
        reason=("cross-execution recovery under the SAME solver "
                f"({this_solver}): a recorded {change['kind']} changed the "
                "result from failed to feasible"),
        evidence={
            "kind": "same_solver_intervention",
            "change": change,
            "failed": {"execution_id": change["from_execution_id"],
                       "solver": this_solver,
                       "error": _first_error(failed_record)
                       if failed_record is not None else "",
                       "error_class": classify_failure(failed_record)
                       if failed_record is not None else None},
            "recovered_by": {"execution_id": record.execution_id,
                             "solver": this_solver},
        })


def _recorded_change(record: ExecutionRecord,
                     prior_failures: List[ExecutionRecord]
                     ) -> Optional[Dict[str, Any]]:
    """Evidence that something really CHANGED between a failed attempt and
    this successful one — the fact that distinguishes a modeling fix from a
    plain retry.

    Two kinds of provable change are accepted:
    * ``method`` — this record's performed method (its script receipt, or a
      harness declaration) differs from the failed attempt's; the difference
      is described step-by-step;
    * ``declared`` — the record carries an explicit change delta (a
      recording-time declaration such as ``intervention``), which is the
      harness naming the change it made.

    Returns None when nothing observable changed. The DETECTOR then reports
    nothing: a retry that happened to succeed is not evidence that a fix
    caused it.
    """
    latest = max(prior_failures, key=lambda r: r.created_at)
    declared = record.execution_features.get("intervention")
    if isinstance(declared, dict) and declared.get("change"):
        return {"kind": "declared", "change": dict(declared),
                "from_execution_id": latest.execution_id,
                "failed_record": latest}
    this_method = normalize_method(record.method_actual) or \
        normalize_method(record.method_planned)
    if this_method is None:
        return None
    other = normalize_method(latest.method_actual) or \
        normalize_method(latest.method_planned)
    if other is None:
        # The failed attempt recorded no method: a difference cannot be
        # established from one side alone.
        return None
    if _method_signature(this_method) == _method_signature(other):
        return None
    return {
        "kind": "method",
        "from_execution_id": latest.execution_id,
        # Internal handle for the caller (stripped before the hint is
        # serialized); the RECORD itself is never part of the evidence.
        "failed_record": latest,
        "from_method": {k: other.get(k) for k in ("name", "steps")},
        "to_method": {k: this_method.get(k) for k in ("name", "steps")},
    }


def _method_signature(method: Dict[str, Any]) -> List[str]:
    steps = [" ".join(str(s).lower().split())
             for s in (method.get("steps") or [])]
    return [" ".join(str(method.get("name", "")).lower().split())] + steps


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


def _structural_reproduction(record: ExecutionRecord,
                             stats: ConditionalStats
                             ) -> List[InductionHint]:
    """structural_reproduction: the same strategy relation recurs in a
    structurally comparable but INDEPENDENT task/family — the 'learn once,
    apply elsewhere' detector. Informational: it reports reproduction across
    independently observed families from the data alone.

    One task's observation is never transferable knowledge on its own, and
    neither is one family's: reproduction needs >= 2 families, each with its
    own >= 2 supporting executions.

    Structural comparability is required BEFORE the reproduction claim is
    made: each family's evidence is scoped to the record's own cell (same
    rc/tc/rx interval), so a family whose behaviour comes from an unrelated
    structure can neither be mixed into the statistic nor veto a genuine
    reproduction between two comparable families.

    Unknown structure never counts. If the reference dimension is unmeasured
    there is nothing to compare, and this returns nothing: "both sides are
    unknown" is a shared absence of evidence, not evidence of structural
    similarity."""
    sid = record.strategy_id
    profile = record.profile_snapshot
    for f in GROUPING_FEATURES:
        if getattr(profile, f) is None:
            return []
    cells = stats.cross_family(sid, like=profile)
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

    structure = {f: bin_label(getattr(profile, f)) for f in GROUPING_FEATURES}
    return [InductionHint(
        pattern="structural_reproduction", strategy_ids=[sid],
        group_key=group_key(profile),
        reason=(f"{direction} performance reproduces independently in "
                f"{len(per_family)} families at the same structure "
                f"({', '.join(f'{k}{v}' for k, v in structure.items())})"),
        evidence={"families": [_family(c) for c in per_family],
                  "direction": direction,
                  "structure": structure,
                  "mean_qualities": {_family(c): round(c.mean_quality, 4)
                                     for c in per_family},
                  "execution_ids": {_family(c): c.execution_ids
                                    for c in per_family}})]


def _advantage_reversal(record: ExecutionRecord, stats: ConditionalStats,
                        group: str) -> Optional[InductionHint]:
    """advantage_reversal: the same strategy's observed quality weakens,
    disappears or FLIPS as the structural condition changes.

    The lesson is an applicability BOUNDARY or a counterexample — never a
    success count. The detector compares the strategy's own cells inside one
    family: one cell where it performs high (>= QUALITY_HIGH_THRESHOLD) and
    another where it performs low (<= QUALITY_LOW_THRESHOLD), each with
    >= 2 supporting executions.

    Scope: cells of the SAME family only. A cross-family difference is a
    different question (that is ``structural_reproduction``), and pooling
    families here would let a family's own structure masquerade as a
    boundary. Unknown structure never contributes: a cell whose dimensions
    are unmeasured is not a structural condition."""
    sid = record.strategy_id
    family = record.profile_snapshot.family
    cells = stats.cells_in_family(sid, family)
    eligible = [c for c in cells.values() if c.n >= MIN_DIVERGENCE_N]
    high = [c for c in eligible
            if c.mean_quality >= QUALITY_HIGH_THRESHOLD]
    low = [c for c in eligible
           if c.mean_quality <= QUALITY_LOW_THRESHOLD]
    if not high or not low:
        return None
    best = max(high, key=lambda c: c.mean_quality)
    worst = min(low, key=lambda c: c.mean_quality)
    return InductionHint(
        pattern="advantage_reversal",
        strategy_ids=[sid],
        group_key=group,
        reason=(f"{sid} advantage reverses across structural cells of "
                f"{family}: meanQ={best.mean_quality:.2f} at "
                f"{best.group_key.split('|', 1)[-1]} vs "
                f"{worst.mean_quality:.2f} at "
                f"{worst.group_key.split('|', 1)[-1]}"),
        evidence={
            "kind": "advantage_reversal",
            "family": family,
            "advantageous_cell": {
                "group_key": best.group_key,
                "mean_quality": round(best.mean_quality, 4),
                "n": best.n,
                "execution_ids": list(best.execution_ids),
            },
            "adverse_cell": {
                "group_key": worst.group_key,
                "mean_quality": round(worst.mean_quality, 4),
                "n": worst.n,
                "execution_ids": list(worst.execution_ids),
            },
        })
