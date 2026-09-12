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

from or_harness.core.schema import COST_DIMENSIONS, CostVector, ExecutionRecord, group_key
from or_harness.strategy.experience_bank import ExperienceBank


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
    """Computes (group x strategy) aggregates directly from the fact layer."""

    def __init__(self, bank: ExperienceBank):
        self.bank = bank

    def cell(self, group_l1: str, strategy_id: str) -> GroupStats:
        records = [r for r in self.bank.query(group_l1=group_l1,
                                              strategy_id=strategy_id)
                   if r.source == "executed"]
        return self._aggregate(group_l1, strategy_id, records)

    def group(self, group_l1: str) -> Dict[str, GroupStats]:
        """All strategy cells within one structural group."""
        cells: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.query(group_l1=group_l1):
            if rec.source != "executed":
                continue
            cells.setdefault(rec.strategy_id, []).append(rec)
        return {sid: self._aggregate(group_l1, sid, recs) for sid, recs in cells.items()}

    def for_profile(self, profile, level: str = "L1") -> Dict[str, GroupStats]:
        return self.group(group_key(profile, level))

    def cross_family(self, profile, strategy_id: str,
                     level: str = "L2") -> List[GroupStats]:
        """Same strategy across families at a wider scope.

        The "learn once, apply elsewhere" detector: partitions the matched
        facts by family so callers can check whether an advantage reproduces
        independently in >= 2 families.
        """
        target = group_key(profile, level)
        by_family: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.query(strategy_id=strategy_id):
            if rec.source != "executed":
                continue
            if group_key(rec.profile_snapshot, level) == target:
                by_family.setdefault(rec.profile_snapshot.family, []).append(rec)
        return [self._aggregate(f"{target}#family={fam}", strategy_id, recs)
                for fam, recs in sorted(by_family.items())]

    def cells_matching(self, strategy_id: str,
                       predicates: Dict[str, Any]) -> List[GroupStats]:
        """Partition a strategy's executions matching ``predicates`` by family."""
        from or_harness.core.schema import profile_matches
        by_family: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.query(strategy_id=strategy_id):
            if rec.source != "executed":
                continue
            if profile_matches(rec.profile_snapshot, predicates):
                by_family.setdefault(rec.profile_snapshot.family, []).append(rec)
        return [self._aggregate(f"match#family={fam}", strategy_id, recs)
                for fam, recs in sorted(by_family.items())]

    def cross_family_from_predicates(self, entry, level: str) -> List[GroupStats]:
        """Partition an entry's evidence by family at ``level``.

        Cells are keyed by the entry's predicate bins rather than the raw
        profile bins, so provenance records in adjacent fine bins still count
        toward the entry's own pattern."""
        from or_harness.core.schema import profile_matches
        by_family: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.query(strategy_id=entry.strategy_id):
            if rec.source != "executed":
                continue
            if profile_matches(rec.profile_snapshot, entry.predicates):
                by_family.setdefault(rec.profile_snapshot.family, []).append(rec)
        return [self._aggregate(f"{level}#family={fam}", entry.strategy_id, recs)
                for fam, recs in sorted(by_family.items())]

    def rebuild_check(self) -> bool:
        """Consistency invariant: aggregating a full scan equals per-group
        aggregation (statistics are derivable from facts at any time)."""
        everything = [r for r in self.bank.all() if r.source == "executed"]
        groups = {rec.group_l1 for rec in everything}
        for g in groups:
            via_group = self.group(g)
            via_scan: Dict[str, List[ExecutionRecord]] = {}
            for rec in everything:
                if rec.group_l1 == g:
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
