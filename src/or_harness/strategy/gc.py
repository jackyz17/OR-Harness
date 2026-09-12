"""Memory disposal: garbage collection over the derived layer.

Disposal targets the derived layer only — facts (ExecutionRecords) stay
neutral forever. Current scope:

- ``purge``: lists retirement candidates (suspect / dormant entries) — never
  retires them itself; retirement stays the harness's explicit call.
- ``compact``: DEFERRED. Lossy evidence compaction is paused until the
  summary consumption contract exists: statistics and induction currently
  ignore ``source="compacted"`` rows, so replacing raw facts with summaries
  would bias conditional statistics (e.g. 90 successes + 10 failures would
  read as 100% failure rate after compaction). The future compaction policy
  will honor the explicit ``retention_reason`` marker on representative
  episodes. No referential integrity exists between the two Banks —
  knowledge admission never depends on evidence survival.

``orx gc [--dry-run] [--mode compact|purge]`` — always the harness's explicit
call; dry-run lists what would be disposed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank

#: Why lossy compaction is paused. Surfaced verbatim by ``gc compact`` so
#: callers see the honest deferral instead of a silent no-op.
COMPACTION_DEFERRED = (
    "lossy evidence compaction is deferred: statistics and induction ignore "
    "source='compacted' rows, so summarizing raw facts would bias conditional "
    "statistics; it will be re-enabled once the summary consumption contract "
    "exists (Cost/Induction rounds)"
)


@dataclass
class GcAction:
    kind: str          # "retire" (compaction is deferred)
    target: str        # entry id / pattern hash
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
        if mode == "compact":
            return []  # deferred — raw facts are never touched
        return self._plan_purge()

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
        if mode == "compact":
            # Honest deferral: raw facts stay untouched, so conditional
            # statistics and induction inputs are never biased.
            return {"dry_run": dry_run, "deferred": COMPACTION_DEFERRED,
                    "compacted_cells": 0, "actions": []}
        actions = self.plan(mode)
        if dry_run:
            return {"dry_run": True, "actions": [a.to_dict() for a in actions]}
        return {"dry_run": False, "compacted_cells": 0,
                "planned_retirements": [a.target for a in actions
                                        if a.kind == "retire"],
                "note": "retirements are not applied by gc; retire entries via "
                        "the retire command after reviewing this plan"}
