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
from typing import Any, Dict, List, Optional, Sequence

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
    fallback_triggered: int = 0
    execution_ids: List[str] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        return self.n_failures / self.n if self.n else 0.0

    @property
    def mean_retries(self) -> float:
        return self.total_retries / self.n if self.n else 0.0

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
        costs: Dict[str, float] = {d: 0.0 for d in COST_DIMENSIONS}
        for rec in records:
            stats.n += 1
            q = quality_score(rec)
            stats.quality_values.append(q)
            if rec.quality.get("feasible", False):
                stats.n_feasible += 1
            if rec.failures:
                stats.n_failures += 1
                if any(f.recovery_action for f in rec.failures):
                    stats.fallback_triggered += 1
            stats.total_retries += rec.cost.retries
            stats.execution_ids.append(rec.execution_id)
            for d in COST_DIMENSIONS:
                costs[d] += getattr(rec.cost, d)
        stats.mean_quality = mean(stats.quality_values)
        stats.std_quality = pstdev(stats.quality_values) if stats.n > 1 else 0.0
        stats.mean_cost = CostVector(**{d: costs[d] / stats.n for d in COST_DIMENSIONS})
        return stats
