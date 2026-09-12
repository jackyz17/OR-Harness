"""Memory disposal: garbage collection over the derived layer.

Disposal targets the derived layer only — facts (ExecutionRecords) stay
neutral forever; compacted ledger lines are bookkeeping summaries, not
StrategicEntries (no commitments, no lifecycle).

Compaction of redundant evidence does NOT invalidate admitted Strategic
Knowledge: knowledge is validated at induction time and carries only
lightweight origin metadata afterwards, so GC imposes no referential
integrity between the two Banks. Evidence retention follows retention
classes instead:
  - recent raw rows (outside the window only are eligible),
  - representative raw rows (``retention_reason`` set — never compacted),
  - compacted ledger lines (counts, mean quality, per-dimension cost
    mean/min/max, failure-class and recovery-action counts — the
    strategically informative summaries).

``orx gc [--dry-run] [--mode compact|purge]`` — always the harness's explicit
call; dry-run lists what would be disposed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector, ExecutionRecord
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import StrategicBank

COMPACT_MIN_PER_CELL = 10          # raw rows per (group, strategy) before compaction
RECENT_TASK_WINDOW = 50            # recent tasks whose raw rows are kept
EXPLORATORY_GROUP_MIN_N = 5        # groups below this stay raw (still exploring)


@dataclass
class GcAction:
    kind: str          # "compact" | "retire" | "purge"
    target: str        # group key / entry id / pattern hash
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target,
                "reason": self.reason, "detail": self.detail}


class GarbageCollector:
    def __init__(self, bank: ExperienceBank, sbank: StrategicBank,
                 stats: ConditionalStats):
        self.bank = bank
        self.sbank = sbank
        self.stats = stats

    # -- planning -----------------------------------------------------------------

    def plan(self, mode: str = "compact") -> List[GcAction]:
        if mode not in ("compact", "purge"):
            raise ValueError("mode must be 'compact' or 'purge'")
        actions: List[GcAction] = []
        if mode == "compact":
            actions.extend(self._plan_compaction())
        else:
            actions.extend(self._plan_purge())
        return actions

    def _plan_compaction(self) -> List[GcAction]:
        actions: List[GcAction] = []
        recent_cutoff = self._recent_cutoff()
        validated = self.sbank.list(status="validated") + \
            self.sbank.list(status="candidate")
        by_group: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.all():
            if rec.source == "executed":
                by_group.setdefault(rec.group_l1, []).append(rec)
        for group, records in sorted(by_group.items()):
            if len(records) < EXPLORATORY_GROUP_MIN_N:
                continue  # exploratory group: keep raw
            covering = [e for e in validated if e.scope_level == "L1"
                        and any(e.matches(r.profile_snapshot) for r in records)]
            if not covering:
                continue  # only compact groups covered by entries
            by_strategy: Dict[str, List[ExecutionRecord]] = {}
            for rec in records:
                by_strategy.setdefault(rec.strategy_id, []).append(rec)
            for sid, recs in sorted(by_strategy.items()):
                if len(recs) <= COMPACT_MIN_PER_CELL:
                    continue
                keep_raw = [r for r in recs if r.created_at >= recent_cutoff]
                keep_retained = [r for r in recs
                                 if r.retention_reason is not None]
                eligible = [r for r in recs
                            if r.created_at < recent_cutoff
                            and r.retention_reason is None]
                if len(eligible) <= COMPACT_MIN_PER_CELL:
                    continue
                actions.append(GcAction(
                    kind="compact",
                    target=f"{group}#{sid}",
                    reason=(f"group covered by entry {covering[0].entry_id}; "
                            f"{len(eligible)} raw rows beyond K={COMPACT_MIN_PER_CELL} "
                            f"and outside the recent window N={RECENT_TASK_WINDOW}"),
                    detail={"execution_ids": [r.execution_id for r in eligible],
                            "kept_recent": [r.execution_id for r in keep_raw],
                            "kept_retained": [r.execution_id for r in keep_retained]}))
        return actions

    def _plan_purge(self) -> List[GcAction]:
        actions: List[GcAction] = []
        for entry in self.sbank.list():
            if entry.status == "suspect":
                actions.append(GcAction(
                    kind="retire", target=entry.entry_id,
                    reason="suspect entry; retirement is irreversible and stays "
                           "the harness's explicit decision",
                    detail={"hit_rate": round(entry.prediction_track.hit_rate, 4),
                            "consecutive_misses":
                                entry.prediction_track.consecutive_misses}))
            elif entry.status == "dormant":
                actions.append(GcAction(
                    kind="retire", target=entry.entry_id,
                    reason="long-dormant entry; candidate for cold archive"))
        return actions

    # -- execution -----------------------------------------------------------------

    def run(self, mode: str = "compact", dry_run: bool = False) -> Dict[str, Any]:
        actions = self.plan(mode)
        if dry_run:
            return {"dry_run": True, "actions": [a.to_dict() for a in actions]}
        compacted = 0
        retired: List[str] = []
        if mode == "compact":
            compacted = self._apply_compaction(actions)
        return {"dry_run": False, "compacted_cells": compacted,
                "planned_retirements": [a.target for a in actions
                                        if a.kind == "retire"],
                "note": "retirements are not applied by gc; retire entries via "
                        "the retire command after reviewing this plan"}

    def _apply_compaction(self, actions: List[GcAction]) -> int:
        by_group: Dict[str, List[ExecutionRecord]] = {}
        for rec in self.bank.all():
            if rec.source == "executed":
                by_group.setdefault(rec.group_l1, []).append(rec)
        rewritten = list(self.bank.all())
        applied = 0
        for action in actions:
            group, sid = action.target.rsplit("#", 1)
            ids = set(action.detail["execution_ids"])
            cell_records = [r for r in by_group.get(group, [])
                            if r.execution_id in ids]
            if len(cell_records) <= COMPACT_MIN_PER_CELL:
                continue
            ledger = self._compact_line(group, sid, cell_records)
            rewritten = [r for r in rewritten if r.execution_id not in ids]
            rewritten.append(ledger)
            applied += 1
        if applied:
            self.bank.replace_all(rewritten)
        return applied

    @staticmethod
    def _compact_line(group: str, sid: str,
                      records: List[ExecutionRecord]) -> ExecutionRecord:
        """Raw rows -> one ledger line.

        The ledger preserves what future cost learning and induction need
        from redundant mass: counts, mean quality, per-dimension cost
        mean/min/max, failure-class counts, recovery-action counts, and
        verification-level counts. Trajectory detail and artifacts do not
        survive here — representative raw rows (``retention_reason``) carry
        those. The derived layer never consumes ledger lines, so no fake
        quality/cost values are fabricated: ``gap`` is None and the summary
        lives in ``quality["compacted"]``.
        """
        n = len(records)
        profile = records[0].profile_snapshot
        cost_totals = {d: 0.0 for d in COST_DIMENSIONS}
        cost_min = {d: float("inf") for d in COST_DIMENSIONS}
        cost_max = {d: float("-inf") for d in COST_DIMENSIONS}
        failures = 0
        failure_classes: Dict[str, int] = {}
        recovery_actions: Dict[str, int] = {}
        verification_levels: Dict[str, int] = {}
        quality_values = []
        for r in records:
            for d in COST_DIMENSIONS:
                v = float(getattr(r.cost, d))
                cost_totals[d] += v
                cost_min[d] = min(cost_min[d], v)
                cost_max[d] = max(cost_max[d], v)
            for f in r.failures:
                failures += 1
                if f.error_class:
                    failure_classes[f.error_class] = \
                        failure_classes.get(f.error_class, 0) + 1
                if f.recovery_action:
                    recovery_actions[f.recovery_action] = \
                        recovery_actions.get(f.recovery_action, 0) + 1
            verification_levels[r.verification_level] = \
                verification_levels.get(r.verification_level, 0) + 1
            quality_values.append(quality_score(r))
        mean_cost = {d: round(v / n, 6) for d, v in cost_totals.items()}
        mean_q = sum(quality_values) / n
        return ExecutionRecord(
            execution_id=f"cx_{group[-20:]}_{sid}".replace("|", "_")[:60],
            task_id=f"compacted:{group}",
            strategy_id=sid,
            profile_snapshot=profile,
            trajectory=[],
            quality={"feasible": True, "objective": None, "gap": None,
                     "status": "compacted",
                     "compacted": {
                         "n": n,
                         "mean_quality": round(mean_q, 6),
                         "failures": failures,
                         "cost_min": {d: round(v, 6)
                                      for d, v in cost_min.items()},
                         "cost_max": {d: round(v, 6)
                                      for d, v in cost_max.items()},
                         "failure_classes": failure_classes,
                         "recovery_actions": recovery_actions,
                         "verification_levels": verification_levels,
                     }},
            cost=CostVector(**mean_cost),
            failures=[],
            solver={"name": "compacted"},
            source="compacted",
            created_at=max(r.created_at for r in records),
        )

    # -- helpers -------------------------------------------------------------------

    def _recent_cutoff(self) -> float:
        rows = self.bank.store.conn.execute(
            "SELECT DISTINCT created_at FROM executions "
            "ORDER BY created_at DESC LIMIT ?",
            (RECENT_TASK_WINDOW,)).fetchall()
        if len(rows) < RECENT_TASK_WINDOW:
            return float("inf")  # nothing old enough to compact
        return float(rows[-1]["created_at"])
