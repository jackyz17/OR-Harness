"""Conditional statistics: on-the-fly aggregation over the Experience Bank.

Statistics are query results — "what happened before" — never knowledge. They
are computed on demand, never persisted, and always rebuildable. An entry that
merely restates statistics is redundant and must not be created: a Strategic
Entry is a *commitment* (prediction intervals + calibration + cross-group
predicates), statistics are a *recount*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    GROUPING_FEATURES,
    bin_label,
    group_key,
)
from or_harness.strategy.experience_bank import ExperienceBank


def cell_token(profile) -> str:
    """The structural part of a group key, without the family.

    Two executions in different families are still STRUCTURALLY comparable
    when their measured coupling values fall in the same cells — that is
    exactly what C5's cross-family reproduction is about, so it cannot
    compare whole keys (which include the family)."""
    return "|".join(bin_label(getattr(profile, f)) for f in GROUPING_FEATURES)


def quality_score(record: ExecutionRecord) -> float:
    """Scalar quality in [0, 1] for comparison.

    Infeasible = 0. Otherwise 1 - clamped gap. Gap semantics come from the
    execution verification layer; statistics never reinterpret them.
    """
    q = record.quality or {}
    if not q.get("feasible", False):
        return 0.0
    gap = q.get("gap")
    if gap is None:
        return 1.0 if q.get("status") == "optimal" else 0.5
    return max(0.0, min(1.0, 1.0 - float(gap)))


@dataclass
class GroupStats:
    """Aggregates for one (group, strategy) cell."""

    group_key: str
    strategy_id: str
    n: int = 0
    n_feasible: int = 0
    n_failures: int = 0
    total_retries: float = 0.0
    mean_quality: float = 0.0
    std_quality: float = 0.0
    quality_values: List[float] = field(default_factory=list)
    mean_cost: CostVector = field(default_factory=CostVector)
    #: Per-dimension count of records where the dimension was actually
    #: measured. A dimension with n_measured == 0 is UNKNOWN for this cell
    #: (mean_cost reports a placeholder 0, never evidence of cheap).
    n_measured: Dict[str, int] = field(default_factory=dict)
    #: Per-scale-feature [min, max] over the supporting records, for
    #: pre-execution scale comparability checks (no thresholds — callers
    #: decide whether a target profile lies outside the sample coverage).
    scale_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fallback_triggered: int = 0
    execution_ids: List[str] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        return self.n_failures / self.n if self.n else 0.0

    @property
    def mean_retries(self) -> float:
        return self.total_retries / self.n if self.n else 0.0

    def measured_cost(self, dim: str) -> Optional[float]:
        """Mean of ``dim`` over records that measured it, or None when no
        record measured it (unknown — never a default zero)."""
        if self.n_measured.get(dim, 0) == 0:
            return None
        return getattr(self.mean_cost, dim)

    def quality_trend(self) -> float:
        """Simple slope proxy: mean(second half) - mean(first half)."""
        if len(self.quality_values) < 3:
            return 0.0
        mid = len(self.quality_values) // 2
        first, second = self.quality_values[:mid], self.quality_values[mid:]
        return mean(second) - mean(first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_key": self.group_key,
            "strategy_id": self.strategy_id,
            "n": self.n,
            "n_feasible": self.n_feasible,
            "fail_rate": round(self.fail_rate, 4),
            "mean_quality": round(self.mean_quality, 4),
            "std_quality": round(self.std_quality, 4),
            "mean_cost": {d: round(v, 4) for d, v in self.mean_cost.to_dict().items()},
            "n_measured": dict(self.n_measured),
            "scale_ranges": {d: [round(lo, 4), round(hi, 4)]
                             for d, (lo, hi) in self.scale_ranges.items()},
            "mean_retries": round(self.mean_retries, 4),
            "fallback_triggered": self.fallback_triggered,
            "execution_ids": list(self.execution_ids),
        }


class ConditionalStats:
    """Computes (group x strategy) aggregates directly from the fact layer.

    A group is one evidence set: a (family, structural cell, strategy) triple.
    Membership is decided by the FAMILY column plus the structural key derived
    from each record's own profile snapshot — never by the stored ``group_l1``
    index, which has been written in more than one format over time."""

    def __init__(self, bank: ExperienceBank):
        self.bank = bank

    def _all_attempt_records(self) -> List[ExecutionRecord]:
        return [r for r in self.bank.all() if self._is_attempt_evidence(r)]

    def cell(self, group_l1: str, strategy_id: str) -> GroupStats:
        """One (structural group key, strategy) cell.

        The key is compared against the record's DERIVED key (from its own
        profile snapshot), so a stale index column can never smuggle a
        foreign fact into the cell. ``group_l1`` is itself the structured key
        (``family=..|rc[..]|..``); its ``family=`` prefix selects the family
        and the rest is recomputed per record."""
        family = group_l1.split("|", 1)[0].replace("family=", "", 1)
        records = [r for r in self._all_attempt_records()
                   if r.strategy_id == strategy_id
                   and r.profile_snapshot.family == family
                   and group_key(r.profile_snapshot) == group_l1]
        return self._aggregate(group_l1, strategy_id, records)

    def group(self, group_l1: str) -> Dict[str, GroupStats]:
        """Every strategy cell inside ONE structural group key.

        Raw view: the key is taken literally (legacy keys included), so this
        is a diagnostic/administrative entry point. Trigger and prediction
        paths use :meth:`for_profile` instead, which scopes to the cell the
        target profile actually belongs to."""
        family = group_l1.split("|", 1)[0].replace("family=", "", 1)
        cells: Dict[str, List[ExecutionRecord]] = {}
        for rec in self._all_attempt_records():
            if rec.profile_snapshot.family != family:
                continue
            if group_key(rec.profile_snapshot) != group_l1:
                continue
            cells.setdefault(rec.strategy_id, []).append(rec)
        return {sid: self._aggregate(group_l1, sid, recs) for sid, recs in cells.items()}

    #: The single membership rule for aggregated evidence: an EXECUTED,
    #: ATTEMPT-scope fact, in the target's own structural cell.
    @staticmethod
    def _is_attempt_evidence(rec: ExecutionRecord) -> bool:
        return rec.source == "executed" and rec.measurement_scope == "attempt"

    def _family_records(self, profile) -> List[ExecutionRecord]:
        """Executed attempt-scope facts of the profile's family.

        Selected by the FAMILY column and then filtered by the structural key
        derived from each record's own profile snapshot — never by the
        ``group_l1`` index column. The index was written in several formats
        over time (``family=routing``, ``family=routing|rc[..]|..``), so
        querying it directly let historical facts silently drop out of the
        statistics."""
        return [r for r in self.bank.query(family=profile.family)
                if self._is_attempt_evidence(r)]

    def for_profile(self, profile) -> Dict[str, GroupStats]:
        """Cells of the structural group ``profile`` belongs to.

        Only executed attempt-scope facts in the SAME structural cell
        contribute: a family can hold regions with opposite behaviour
        (low-coupling Q=1.0 vs high-coupling Q=0.1), and pooling them into
        one cell would erase the relation between structure and performance."""
        key = group_key(profile)
        cells: Dict[str, List[ExecutionRecord]] = {}
        for rec in self._family_records(profile):
            if group_key(rec.profile_snapshot) != key:
                continue
            cells.setdefault(rec.strategy_id, []).append(rec)
        return {sid: self._aggregate(key, sid, recs)
                for sid, recs in cells.items()}

    def evidence(self, profile, strategy_id: str) -> List[ExecutionRecord]:
        """The records of one (structural group, strategy) evidence set.

        SAME membership rule :meth:`for_profile` aggregates over, exposed so
        induction reads a claim's predicates off the very records it
        aggregates — sample count, independent-task count, ranges,
        statistics and prediction can therefore never disagree about which
        facts supported a claim (mixing them is how a task-scope total once
        masqueraded as an independent attempt observation)."""
        key = group_key(profile)
        return [r for r in self._family_records(profile)
                if r.strategy_id == strategy_id
                and group_key(r.profile_snapshot) == key]

    def aggregate(self, group_l1: str, strategy_id: str,
                  records: Sequence[ExecutionRecord]) -> GroupStats:
        """Public aggregator (used by induction for a cell it already read)."""
        return self._aggregate(group_l1, strategy_id, records)

    def cross_family(self, strategy_id: str, *,
                     like=None) -> List[GroupStats]:
        """One strategy's evidence partitioned by family, structurally
        scoped to the reference ``like`` when one is given.

        The "learn once, apply elsewhere" view. With ``like=profile`` only
        executions in the SAME structural cell as the reference contribute,
        so families whose behaviour comes from an unrelated structure can
        neither be mixed into the statistic nor veto a genuine reproduction
        between two comparable families.

        With ``like=None`` there is no common structure to compare against:
        every executed attempt-scope fact of the strategy contributes, and
        the caller must treat the result as "each family's own behaviour"
        rather than as evidence of a shared structure."""
        by_family: Dict[str, List[ExecutionRecord]] = {}
        reference_cell = cell_token(like) if like is not None else None
        for rec in self.bank.query(strategy_id=strategy_id):
            if not self._is_attempt_evidence(rec):
                continue
            if (reference_cell is not None
                    and cell_token(rec.profile_snapshot) != reference_cell):
                continue
            by_family.setdefault(rec.profile_snapshot.family, []).append(rec)
        return [self._aggregate(f"family={fam}", strategy_id, recs)
                for fam, recs in sorted(by_family.items())]

    def rebuild_check(self) -> bool:
        """Consistency invariant: aggregating a full scan equals per-group
        aggregation (statistics are derivable from facts at any time).

        Groups are keyed by the DERIVED key of each record's own profile, so
        the invariant holds even when the stored index column is stale."""
        everything = [r for r in self.bank.all() if r.source == "executed"]
        by_key: Dict[str, List[ExecutionRecord]] = {}
        for rec in everything:
            by_key.setdefault(group_key(rec.profile_snapshot), []).append(rec)
        for g, recs in by_key.items():
            via_group = self.group(g)
            via_scan: Dict[str, List[ExecutionRecord]] = {}
            for rec in recs:
                via_scan.setdefault(rec.strategy_id, []).append(rec)
            for sid, cell in via_group.items():
                other = self._aggregate(g, sid, via_scan.get(sid, []))
                if cell.to_dict() != other.to_dict():
                    return False
        return True

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _aggregate(group: str, strategy_id: str,
                   records: Sequence[ExecutionRecord]) -> GroupStats:
        stats = GroupStats(group_key=group, strategy_id=strategy_id)
        if not records:
            return stats
        # Per-dimension sums over MEASURED values only, and only over
        # attempt-scope records: an unmeasured dimension never contributes
        # its placeholder zero, and a task-scope record never contaminates
        # attempt-scope statistics.
        sums: Dict[str, float] = {d: 0.0 for d in COST_DIMENSIONS}
        counts: Dict[str, int] = {d: 0 for d in COST_DIMENSIONS}
        scale_mins: Dict[str, float] = {}
        scale_maxs: Dict[str, float] = {}
        for rec in records:
            if rec.measurement_scope != "attempt":
                continue
            stats.n += 1
            q = quality_score(rec)
            stats.quality_values.append(q)
            if rec.quality.get("feasible", False):
                stats.n_feasible += 1
            if rec.failures:
                stats.n_failures += 1
                if any(f.recovery_action for f in rec.failures):
                    stats.fallback_triggered += 1
            stats.execution_ids.append(rec.execution_id)
            measured = rec.cost.measured_dims()
            if "retries" in measured:
                stats.total_retries += rec.cost.retries
            for d in COST_DIMENSIONS:
                if d in measured:
                    sums[d] += getattr(rec.cost, d)
                    counts[d] += 1
            for feat, value in rec.profile_snapshot.scale_features.items():
                scale_mins[feat] = min(scale_mins.get(feat, float("inf")), value)
                scale_maxs[feat] = max(scale_maxs.get(feat, float("-inf")), value)
        if stats.n == 0:
            return stats
        stats.mean_quality = mean(stats.quality_values)
        stats.std_quality = pstdev(stats.quality_values) if stats.n > 1 else 0.0
        stats.n_measured = counts
        stats.scale_ranges = {feat: (scale_mins[feat], scale_maxs[feat])
                              for feat in scale_mins}
        measured_dims = {d for d in COST_DIMENSIONS if counts[d] > 0}
        stats.mean_cost = CostVector(
            **{d: (sums[d] / counts[d] if counts[d] else 0.0)
               for d in COST_DIMENSIONS},
            measured=measured_dims)
        return stats
