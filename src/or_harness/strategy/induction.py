"""Induction: consolidate episodic facts into calibrated commitments.

The core question of induction is "which rung of the scope ladder does the
evidence support?" — L1 (family + fine bins), L2 (fine bins, any family), L3
(coarse bins, any family). Widening is falsifiable: a wider entry makes
riskier predictions, and a cross-family miss tightens the pattern back down.

Entries are never born validated. Predictions are verified by future
executions (forward validation), not by self-test on training data. The only
LLM injection point is phrasing: the harness may supply applicability/risk
text at induce time, but citations are checked (every execution_id must be
real and numeric claims must agree with the cited records) and unverified
conditions never enter scoring.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    ApplicabilityCondition,
    CostVector,
    ProblemProfile,
    StrategicEntry,
    group_key,
    min_interval_width,
    pattern_for,
    predicates_cover,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats, quality_score
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.strategy.triggers import InductionHint

WIDEN_LADDER = {"L1": "L2", "L2": "L3"}
NARROW_LADDER = {"L3": "L2", "L2": "L1"}

#: v1 cost interval: multiplicative band around the point estimate. Actual
#: cost within [0.5x, 2.0x] of the prediction counts as a hit.
COST_INTERVAL_BAND = (0.5, 2.0)


class InductionEngine:
    def __init__(self, stats: ConditionalStats, sbank: StrategicBank):
        self.stats = stats
        self.sbank = sbank

    # -- consolidation -----------------------------------------------------------

    def induce(self, profile: ProblemProfile, strategy_id: str, *,
               scope: str = "L1",
               llm_conditions: Optional[List[Dict[str, Any]]] = None,
               dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        """Create or refresh the entry for (strategy, structural group).

        Guard rails:
        - cold-archive veto (anti-resurrection) unless ``force``;
        - honest intervals: width floored by sample size;
        - no restatement-only entries: if an equally-wide or wider entry
          already covers the pattern with the same prediction, this is a no-op.
        """
        key = group_key(profile, scope)
        cell = self._cell_for(profile, strategy_id, scope)
        if cell.n < 2:
            return {"created": None, "skipped": "fewer than 2 supporting executions",
                    "cell": cell.to_dict()}
        predicates = pattern_for(profile, scope)
        veto = self.sbank.archive_vetoes(strategy_id, predicates)
        if veto is not None and not force:
            return {"created": None,
                    "vetoed": {"pattern_hash": veto.pattern_hash,
                               "reason": veto.reason},
                    "skipped": "cold-archive veto (use --force to override)"}

        existing = self._find_covering(strategy_id, predicates)
        quality_hat = cell.mean_quality
        lo, hi = self._honest_interval(cell)
        cost_hat = cell.mean_cost
        fail_prob = cell.fail_rate
        # Cost intervals only for dimensions that were ever measured — an
        # unmeasured dimension carries no interval (unknown), never a
        # fabricated band around a placeholder zero. The interval key set
        # doubles as the entry's measured-dimension mask.
        measured_cost_interval = {d: band for d, band in
                                  self._cost_interval().items()
                                  if cell.n_measured.get(d, 0) > 0}

        conditions: List[ApplicabilityCondition] = []
        rejected_conditions: List[Dict[str, Any]] = []
        for raw in (llm_conditions or []):
            ok, why, cond = self._check_condition(raw)
            if ok:
                conditions.append(cond)
            else:
                rejected_conditions.append({"text": str(raw.get("text", ""))[:120],
                                            "reason": why})

        if existing is not None:
            changed = (abs(existing.expected_quality_hat - quality_hat) > 0.02
                       or existing.support_n != cell.n
                       or abs(existing.failure_prob - fail_prob) > 0.02
                       or self._cost_estimates_changed(existing, cost_hat,
                                                       measured_cost_interval,
                                                       cell.n_measured))
            if not changed and not conditions:
                return {"created": None,
                        "skipped": f"entry {existing.entry_id} already encodes this "
                                   "evidence (restatement-only entries are forbidden)",
                        "entry_id": existing.entry_id}
            if dry_run:
                return {"created": None, "would_update": existing.entry_id,
                        "cell": cell.to_dict()}
            existing.expected_quality_hat = quality_hat
            existing.quality_interval = (lo, hi)
            existing.expected_cost_hat = cost_hat
            existing.cost_interval = measured_cost_interval
            existing.cost_support_n = dict(cell.n_measured)
            existing.failure_prob = fail_prob
            existing.support_n = cell.n
            existing.provenance = cell.execution_ids[:50]
            if conditions:
                existing.applicability.extend(conditions)
            self.sbank.update(existing)
            return {"updated": existing.entry_id, "cell": cell.to_dict(),
                    "conditions_added": len(conditions),
                    "conditions_rejected": rejected_conditions}

        if dry_run:
            return {"would_create": {"strategy_id": strategy_id,
                                     "scope": scope, "predicates": predicates},
                    "cell": cell.to_dict()}
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=strategy_id,
            pattern={"scope_level": scope, "predicates": predicates},
            expected_quality_hat=quality_hat,
            quality_interval=(lo, hi),
            expected_cost_hat=cost_hat,
            cost_interval=measured_cost_interval,
            cost_support_n=dict(cell.n_measured),
            failure_prob=fail_prob,
            applicability=conditions,
            fallback_strategy_id=None,
            provenance=cell.execution_ids[:50],
            support_n=cell.n,
        )
        self.sbank.add(entry)
        return {"created": entry.entry_id, "entry": entry.to_dict(),
                "cell": cell.to_dict(),
                "conditions_rejected": rejected_conditions}

    def induce_from_hint(self, hint: InductionHint, profile: ProblemProfile, *,
                         dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        scope = hint.scope_suggestion if hint.criterion == "C5" else "L1"
        results = [self.induce(profile, sid, scope=scope, dry_run=dry_run,
                               force=force)
                   for sid in hint.strategy_ids]
        return {"hint": hint.to_dict(), "results": results}

    # -- scope ladder ----------------------------------------------------------------

    def widen(self, entry_id: str) -> Dict[str, Any]:
        """Move an entry up the ladder when cross-group evidence supports it."""
        entry = self.sbank.get(entry_id)
        if entry is None:
            return {"error": f"unknown entry {entry_id}"}
        target = WIDEN_LADDER.get(entry.scope_level)
        if target is None:
            return {"error": "already at widest scope (L3)"}
        merged = self._merged_predicates(entry, target)
        if merged is None:
            return {"error": "cross-group evidence does not reproduce the "
                             "advantage; widening refused"}
        entry.pattern = {"scope_level": target, "predicates": merged}
        self.sbank.update(entry)
        return {"widened": entry.entry_id, "new_scope": target,
                "predicates": merged}

    def tighten(self, entry_id: str) -> Dict[str, Any]:
        """Move an entry down the ladder after a cross-family miss.

        A scope miss is not a suspect-demote: the prediction content may be
        right where it belongs; only the generalization range was wrong."""
        entry = self.sbank.get(entry_id)
        if entry is None:
            return {"error": f"unknown entry {entry_id}"}
        target = NARROW_LADDER.get(entry.scope_level)
        if target is None:
            return {"error": "already at narrowest scope (L1)"}
        predicates = dict(entry.predicates)
        if target == "L1":
            family = self._provenance_family(entry)
            if family is None:
                return {"error": "cannot tighten to L1 without provenance family"}
            predicates["family"] = family
        else:  # L3 -> L2: keep predicates (fine bins unknown); intersect to L2 bins
            predicates = {k: v for k, v in predicates.items() if k != "family"}
        entry.pattern = {"scope_level": target, "predicates": predicates}
        self.sbank.update(entry)
        return {"tightened": entry.entry_id, "new_scope": target,
                "predicates": predicates}

    # -- rebuild ------------------------------------------------------------------

    def rebuild(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Re-induce the entire Strategic Knowledge Bank from the evidence
        currently retained in the Evidence Bank (raw ``source="executed"``
        rows).

        This is re-induction, NOT exact reconstruction: the resulting bank
        may legitimately differ from the previous one (induction logic,
        evidence set, and validation criteria all evolve). Tombstones in the
        cold archive are preserved (they are disposal decisions, not
        derivations)."""
        bank = self.stats.bank
        groups: Dict[Tuple[str, str], List[str]] = {}
        for rec in bank.all():
            if rec.source != "executed":
                continue
            groups.setdefault((rec.group_l1, rec.strategy_id), []).append(rec.execution_id)
        plan = []
        for (group, sid), ids in sorted(groups.items()):
            if len(ids) < 2:
                continue
            sample = bank.get(ids[0])
            plan.append({"strategy_id": sid, "group": group, "n": len(ids),
                         "profile": sample.profile_snapshot})
        if dry_run:
            return {"would_rebuild": len(plan),
                    "cells": [{"strategy_id": p["strategy_id"], "group": p["group"],
                               "n": p["n"]} for p in plan]}
        # Wipe hot entries (archive kept), re-induct each cell at L1.
        with self.sbank.store.transaction() as conn:
            conn.execute("DELETE FROM strategic_entries")
        created = []
        for p in plan:
            result = self.induce(p["profile"], p["strategy_id"], scope="L1")
            if result.get("created"):
                created.append(result["created"])
        return {"rebuilt": len(created), "entry_ids": created}

    def _cost_estimates_changed(self, existing: StrategicEntry,
                                new_hat: CostVector,
                                new_interval: Dict[str, Tuple[float, float]],
                                new_support: Optional[Dict[str, int]] = None
                                ) -> bool:
        """True when the entry's cost estimate needs refreshing: the measured
        dimension set changed, any measured dimension's point estimate moved
        materially (relative to its previous magnitude), or any dimension's
        effective sample size changed (identical means with more measured
        samples still raise the entry's — and its predictions' — support).
        This is the induction update-connection only — no induction
        refactoring."""
        if set(existing.cost_interval.keys()) != set(new_interval.keys()):
            return True
        if new_support is not None and dict(existing.cost_support_n) != dict(new_support):
            return True
        for dim in new_interval:
            old = getattr(existing.expected_cost_hat, dim)
            new = getattr(new_hat, dim)
            if abs(old - new) > 0.02 * max(abs(old), 1.0):
                return True
        return False

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _cost_interval() -> Dict[str, Tuple[float, float]]:
        """v1: fixed multiplicative band per dimension."""
        from or_harness.core.schema import COST_DIMENSIONS
        return {d: COST_INTERVAL_BAND for d in COST_DIMENSIONS}

    def _cell_for(self, profile: ProblemProfile, strategy_id: str,
                  scope: str) -> GroupStats:
        if scope == "L1":
            return self.stats.cell(group_key(profile, "L1"), strategy_id)
        target = group_key(profile, scope)
        bank = self.stats.bank
        records = [r for r in bank.query(strategy_id=strategy_id)
                   if r.source == "executed"
                   and group_key(r.profile_snapshot, scope) == target]
        return self.stats._aggregate(target, strategy_id, records)

    @staticmethod
    def _honest_interval(cell: GroupStats) -> Tuple[float, float]:
        """Interval honest to sample size: never narrower than the floor for
        n, centered on the observed mean, spread by the observed std."""
        spread = max(cell.std_quality, min_interval_width(cell.n) / 2.0)
        lo = max(0.0, cell.mean_quality - spread)
        hi = min(1.0, cell.mean_quality + spread)
        if hi - lo < min_interval_width(cell.n):
            hi = min(1.0, lo + min_interval_width(cell.n))
            lo = max(0.0, hi - min_interval_width(cell.n))
        return (round(lo, 4), round(hi, 4))

    def _find_covering(self, strategy_id: str,
                       predicates: Dict[str, Any]) -> Optional[StrategicEntry]:
        for entry in self.sbank.list(strategy_id=strategy_id, include_dormant=False):
            if predicates_cover(entry.predicates, predicates):
                return entry
        return None

    def _check_condition(self, raw: Dict[str, Any]
                         ) -> Tuple[bool, str, Optional[ApplicabilityCondition]]:
        """Citation binding: every supporting id must be a real record, and
        numeric claims in the text must agree with the cited records."""
        text = str(raw.get("text", "")).strip()
        if not text:
            return False, "empty condition text", None
        ids = [str(i) for i in (raw.get("supporting_execution_ids") or [])]
        if not ids:
            return False, "conditions must cite supporting_execution_ids", None
        records = []
        for ex_id in ids:
            rec = self.stats.bank.get(ex_id)
            if rec is None:
                return False, f"cited execution {ex_id} does not exist", None
            records.append(rec)
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if numbers:
            gaps = [quality_score(r) * 100.0 for r in records]
            claimed = [float(n) for n in numbers]
            lo, hi = min(gaps) - 15.0, max(gaps) + 15.0
            if not all(lo <= c <= hi for c in claimed):
                return (False,
                        f"numeric claims {claimed} disagree with cited records "
                        f"(quality range {min(gaps):.0f}-{max(gaps):.0f}%)",
                        None)
        return True, "", ApplicabilityCondition(
            text=text, verified=False, supporting_execution_ids=ids)

    def _merged_predicates(self, entry: StrategicEntry,
                           target: str) -> Optional[Dict[str, Any]]:
        """Merge predicate intervals across families when the strategy's
        advantage reproduces in the same direction (predicate-interval union
        replaces the legacy 500-line LLM alignment pipeline)."""
        wide_predicates = {k: v for k, v in entry.predicates.items()
                           if k != "family"}
        cells = self.stats.cells_matching(entry.strategy_id, wide_predicates)
        if len(cells) < 2:
            return None
        direction = entry.expected_quality_hat - 0.5
        deltas = [c.mean_quality - 0.5 for c in cells]
        if not all((d > 0) == (direction > 0) for d in deltas):
            return None
        return wide_predicates

    def _provenance_family(self, entry: StrategicEntry) -> Optional[str]:
        for ex_id in entry.provenance:
            rec = self.stats.bank.get(ex_id)
            if rec is not None:
                return rec.profile_snapshot.family
        return None


# ---------------------------------------------------------------------------
# Pattern reflow (interface reservation only — deliberately unimplemented)
# ---------------------------------------------------------------------------


class PatternReflowEngine:
    """Interface reservation for pattern reflow. NOT implemented in v1.

    The future contract (documented, not built):

        new strategic pattern
              ↓
        find related experiences (facts whose profiles match the new
        pattern's predicates or mechanisms, including already-compacted
        groups)
              ↓
        reinterpret / relink them under the new pattern
              ↓
        update the strategic structure (entries, provenance links)

    Invariants any future implementation must preserve:
    - the Experience Bank stays append-only and neutral — reflow reorganizes
      the DERIVED layer only;
    - reflow is the harness's explicit call, like induce and gc;
    - reinterpreted links carry provenance back to the original facts.
    """

    def __init__(self, stats: ConditionalStats, sbank: StrategicBank):
        self.stats = stats
        self.sbank = sbank

    def propose_reflow(self, new_entry_ids: List[str]) -> List[Dict[str, Any]]:
        """Currently a no-op: returns an empty proposal list. The harness
        surface (``orx induce``) does not expose reflow yet."""
        return []
