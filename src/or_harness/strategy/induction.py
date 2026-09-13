"""Induction: consolidate episodic facts into calibrated commitments.

The core question of induction is "what does the evidence entitle me to
claim?" — a claim's applicability is read off the very executions that
support it: the family they came from, and the span of structural features
they covered. Nothing is quantized into fixed bins and nothing has to be
manually widened: as evidence accumulates (a task at another scale, a
neighbouring coupling magnitude), the claim's intervals follow it, and
in-scope failures are what pull it back down.

Entries are never born validated. Predictions are verified by future
executions (forward validation), not by self-test on training data. Those
verifications happen offline: recording freezes each check onto the fact, and
``revise`` replays them at induction time — promotion, demotion, and dormancy
wakeup all live here, never in the record chain. Quality checks are isolated
by strategy and scope, and cost deviations stay on the facts as evidence.

Admission is gated, revision is not: creating a claim takes two independent
tasks (three runs of one instance generalize about that instance), while an
entry that already exists is refreshed by any new matching evidence.

The only LLM injection point is phrasing: the harness may attach free-text
applicability notes, which are kept for the reader and never scored.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    CostVector,
    GROUPING_FEATURES,
    PredictionTrack,
    ProblemProfile,
    StrategicEntry,
    empty_verification,
    evidence_predicates,
    group_key,
    min_interval_width,
    predicates_cover,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats
from or_harness.strategy.strategic_bank import StrategicBank, apply_transitions
from or_harness.strategy.verification import VERIFIED, verify_candidate

#: v1 cost interval: multiplicative band around the point estimate. Actual
#: cost within [0.5x, 2.0x] of the prediction counts as a hit.
COST_INTERVAL_BAND = (0.5, 2.0)

#: Reported whenever a candidate is formed without an admission verdict: the
#: entry exists (and collects forward checks) but is not published knowledge.
UNVERIFIED_NOTE = ("recorded as an unverified candidate: not published as "
                   "strategic knowledge — recall falls back to conditional "
                   "statistics until an admission check passes")


class InductionEngine:
    def __init__(self, stats: ConditionalStats, sbank: StrategicBank):
        self.stats = stats
        self.sbank = sbank

    # -- consolidation -----------------------------------------------------------

    def induce(self, profile: ProblemProfile, strategy_id: str, *,
               notes: Optional[List[str]] = None,
               dry_run: bool = False, force: bool = False,
               verify: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create or refresh the entry for (strategy, evidence set).

        Guard rails:
        - cold-archive veto (anti-resurrection): a retired pattern is refused
          unless ``force``, and ``force`` LIFTS the veto (removes the card)
          so the harness's judgment is made once, not repeated every round;
          a veto reports before the admission gate — it is the real blocker;
        - honest intervals: width floored by sample size;
        - no restatement-only entries: if an equally-wide or wider entry
          already covers the pattern with the same prediction, this is a no-op.

        Creating a claim also requires INDEPENDENT evidence: at least two
        distinct tasks. Repeating one task is repetition, not reproduction —
        it can still refresh a claim that already exists, but it cannot
        create one.

        The claim's predicates are read off the supporting records
        (:func:`evidence_predicates`), so refreshing with new evidence is what
        widens (or narrows) its applicability.

        ``notes`` are harness-written applicability notes (free text): kept on
        the entry for the reader, never scored.

        ``verify`` optionally carries the harness's offline admission check
        ``{"purpose", "claim", "check", "executions", "supporting"}``; the
        framework computes the verdict from real executions
        (:mod:`or_harness.strategy.verification`). Without it the entry is
        created ``unverified`` — a candidate that is NOT published as
        strategic knowledge.
        """
        records = self.stats.evidence(profile, strategy_id)
        cell = self.stats.aggregate(group_key(profile), strategy_id, records)
        if cell.n < 2:
            return {"created": None, "skipped": "fewer than 2 supporting executions",
                    "cell": cell.to_dict()}
        predicates = evidence_predicates(
            records, family=profile.family)
        existing = self._find_existing(strategy_id, predicates,
                                       include_dormant=True)
        veto = self.sbank.archive_vetoes(strategy_id, predicates)
        if veto is not None:
            if not force:
                return {"created": None,
                        "vetoed": {"pattern_hash": veto.pattern_hash,
                                   "reason": veto.reason},
                        "skipped": "cold-archive veto (use --force to override)"}
            # The harness judged the environment drifted: lift the veto.
            # A rehearsal must not write — the lift happens only for real.
            if not dry_run:
                self.sbank.revive(veto.pattern_hash, force=True)
            else:
                veto = None
        tasks = sorted({r.task_id for r in records})
        if existing is None and len(tasks) < 2:
            # A veto (above) is the real blocker and reports first: telling the
            # harness to go collect a second task would send it down a path
            # that cannot succeed while the card stands.
            return {"created": None,
                    "verification": {"tasks": tasks, "required_tasks": 2},
                    "skipped": (f"needs independent evidence: all {cell.n} "
                                f"observations come from {len(tasks)} task "
                                f"{tasks} — a claim requires >=2 tasks"),
                    "cell": cell.to_dict()}

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
        note_texts = [str(n).strip() for n in (notes or []) if str(n).strip()]
        verification, verification_note = self._run_verification(verify, dry_run)

        if existing is not None:
            changed = (abs(existing.expected_quality_hat - quality_hat) > 0.02
                       or existing.support_n != cell.n
                       or existing.predicates != predicates
                       or abs(existing.failure_prob - fail_prob) > 0.02
                       or verification is not None
                       or self._cost_estimates_changed(existing, cost_hat,
                                                       measured_cost_interval,
                                                       cell.n_measured))
            if not changed and not note_texts:
                return {"created": None,
                        "skipped": f"entry {existing.entry_id} already encodes this "
                                   "evidence (restatement-only entries are forbidden)",
                        "entry_id": existing.entry_id}
            if dry_run:
                out = {"created": None, "would_update": existing.entry_id,
                       "cell": cell.to_dict()}
                if verification is not None:
                    out["verification"] = verification
                return out
            existing.pattern = {"predicates": predicates}
            existing.expected_quality_hat = quality_hat
            existing.quality_interval = (lo, hi)
            existing.expected_cost_hat = cost_hat
            existing.cost_interval = measured_cost_interval
            existing.cost_support_n = dict(cell.n_measured)
            existing.failure_prob = fail_prob
            existing.support_n = cell.n
            existing.provenance = cell.execution_ids[:50]
            if verification is not None:
                existing.verification = verification
            if note_texts:
                existing.applicability.extend(note_texts)
            self.sbank.update(existing)
            out = {"updated": existing.entry_id, "cell": cell.to_dict(),
                   "predicates": predicates,
                   "notes_added": len(note_texts)}
            if verification is not None:
                out["verification"] = verification
            if verification_note is not None:
                out["skipped"] = verification_note
            return out

        if dry_run:
            out: Dict[str, Any] = {"would_create": {"strategy_id": strategy_id,
                                                    "predicates": predicates},
                                   "cell": cell.to_dict()}
            if verification is not None:
                out["verification"] = verification
            return out
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=strategy_id,
            pattern={"predicates": predicates},
            expected_quality_hat=quality_hat,
            quality_interval=(lo, hi),
            expected_cost_hat=cost_hat,
            cost_interval=measured_cost_interval,
            cost_support_n=dict(cell.n_measured),
            failure_prob=fail_prob,
            applicability=note_texts,
            fallback_strategy_id=None,
            provenance=cell.execution_ids[:50],
            support_n=cell.n,
            verification=(verification if verification is not None
                          else empty_verification()),
        )
        self.sbank.add(entry)
        out = {"created": entry.entry_id, "entry": entry.to_dict(),
               "predicates": predicates, "cell": cell.to_dict()}
        if verification is not None:
            out["verification"] = verification
        if verification_note is not None:
            out["skipped"] = verification_note
        elif verification is None:
            # No verdict supplied: the entry is recorded as a candidate but
            # is NOT published, and the caller is told exactly that instead
            # of having to infer it from an empty field.
            out["skipped"] = UNVERIFIED_NOTE
        return out

    def _run_verification(self, verify: Optional[Dict[str, Any]],
                          dry_run: bool) -> Tuple[Optional[Dict[str, Any]],
                                                 Optional[str]]:
        """Compute the admission verdict from real executions.

        Nothing is executed here and nothing is written: the harness hands in
        the executions it already produced (or intends to), and the framework
        applies the declared check. Returning ``None`` preserves whatever the
        entry already carries — re-inducing from the same evidence must not
        silently demote a verified claim back to unverified."""
        if not verify:
            return None, None
        report = verify_candidate(
            verify.get("purpose"), str(verify.get("claim", "")),
            check=verify.get("check"),
            executions=verify.get("executions") or (),
            supporting=verify.get("supporting") or ())
        state = report.get("state")
        note = None
        if state != VERIFIED:
            note = (f"candidate is {state}: not published as strategic "
                    "knowledge — " + str(report.get("conclusion", "")))
        return report, note

    # -- rebuild ------------------------------------------------------------------

    def rebuild(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Re-induce the entire Strategic Knowledge Bank from the evidence
        currently retained in the Evidence Bank (raw ``source="executed"``
        attempt-scope rows).

        This is re-induction, NOT exact reconstruction: the resulting bank
        may legitimately differ from the previous one (induction logic,
        evidence set, and validation criteria all evolve). Cold-archive cards
        are preserved (they are disposal decisions, not derivations).

        ``dry_run`` plans only — nothing is wiped, created or revived."""
        bank = self.stats.bank
        groups: Dict[Tuple[str, str], List[str]] = {}
        for rec in bank.all():
            if rec.source != "executed" or rec.measurement_scope != "attempt":
                continue
            # Group by the DERIVED key: a stale index column (pre-retirement
            # format) must not decide which cells are rebuilt.
            key = group_key(rec.profile_snapshot)
            groups.setdefault((key, rec.strategy_id), []).append(rec.execution_id)
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
        # Wipe hot entries (archive kept), re-induct each cell.
        with self.sbank.store.transaction() as conn:
            conn.execute("DELETE FROM strategic_entries")
        created = []
        for p in plan:
            result = self.induce(p["profile"], p["strategy_id"])
            if result.get("created"):
                created.append(result["created"])
        return {"rebuilt": len(created), "entry_ids": created}

    # -- offline revalidation --------------------------------------------------

    def revise(self, strategy_id: Optional[str] = None, *,
               dry_run: bool = False) -> List[Dict[str, Any]]:
        """Re-derive lifecycle state from the frozen evidence (offline).

        Online recording never changes knowledge: it writes one frozen check
        per matching entry onto the fact
        (``execution_features.quality_feedback``), taken against the interval
        that was in force at that moment. This pass replays those checks:

        - the forward track (n_predictions / hits / consecutive misses /
          calibration error) is REBUILT from the frozen checks — never
          re-scored against the entry's current interval;
        - a check that missed is a CONTENT miss: the claim covered that task
          when the execution ran, so the failure is evidence against the
          claim. Three consecutive content misses demote to ``suspect``;
        - promotion (n >= 5 checks, hit rate >= 0.7) and demotion are applied
          HERE, through the same rules the per-event API uses
          (:func:`apply_transitions`);
        - a dormant entry wakes when matching evidence is newer than its
          recency mark (``last_consulted_at`` or ``created_at``) — the same
          mark dormancy aging uses.

        ``dry_run`` returns the same report without writing anything.
        """
        checks_by_entry = self._frozen_checks()
        report: List[Dict[str, Any]] = []
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=True):
            checks = checks_by_entry.get(entry.entry_id)
            if not checks:
                continue  # no forward evidence for this entry yet
            rebuilt = self._replay(checks)
            probe = StrategicEntry.from_dict(entry.to_dict())
            track = probe.prediction_track
            track.n_predictions = rebuilt.n_predictions
            track.n_hits = rebuilt.n_hits
            track.consecutive_misses = rebuilt.consecutive_misses
            track.calibration_error = rebuilt.calibration_error
            newest = max(c["created_at"] for c in checks)
            recency = probe.last_consulted_at or probe.created_at
            transitions = (apply_transitions(probe)
                           if probe.status != "dormant" or newest > recency
                           else [])
            item: Dict[str, Any] = {
                "entry_id": entry.entry_id,
                "strategy_id": entry.strategy_id,
                "forward": {"n_predictions": track.n_predictions,
                            "n_hits": track.n_hits,
                            "hit_rate": round(track.hit_rate, 4),
                            "consecutive_misses": track.consecutive_misses,
                            "calibration_error": round(track.calibration_error, 4)},
                "misses": [c["execution_id"] for c in checks if not c["hit"]],
                "transitions": list(transitions),
            }
            if dry_run:
                report.append(item)
                continue
            live = entry.prediction_track
            live.n_predictions = track.n_predictions
            live.n_hits = track.n_hits
            live.consecutive_misses = track.consecutive_misses
            live.calibration_error = track.calibration_error
            entry.status = probe.status
            self.sbank.update(entry)
            report.append(item)
        return report

    def _frozen_checks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Frozen forward checks in the Evidence Bank, grouped by entry.

        Each check carries the fact it came from (execution id, creation
        time) so the replay can order the checks chronologically."""
        by_entry: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self.stats.bank.all():
            if rec.source != "executed":
                continue
            for raw in (rec.execution_features.get("quality_feedback") or []):
                entry_id = str(raw.get("entry_id", ""))
                if not entry_id:
                    continue
                by_entry.setdefault(entry_id, []).append({
                    "execution_id": rec.execution_id,
                    "created_at": rec.created_at,
                    "hit": bool(raw.get("hit", False)),
                    "observed": float(raw.get("observed", 0.0)),
                    "predicted": float(raw.get("predicted", 0.0)),
                })
        for checks in by_entry.values():
            checks.sort(key=lambda c: (c["created_at"], c["execution_id"]))
        return by_entry

    @staticmethod
    def _replay(checks: Sequence[Dict[str, Any]]) -> PredictionTrack:
        """Rebuild a forward track from frozen checks, chronologically."""
        track = PredictionTrack()
        for check in checks:
            track.record(check["hit"],
                         abs(check["observed"] - check["predicted"]))
        return track

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

    @staticmethod
    def _honest_interval(cell: GroupStats) -> Tuple[float, float]:
        """Interval honest to sample size: never narrower than the floor for
        n, centered on the observed mean, spread by the observed std.

        Full precision — this value is persisted (``StrategicEntry.to_dict``
        feeds the payload) and used for matching decisions, so rounding here
        would only move the defect to disk. Human-facing summaries round."""
        spread = max(cell.std_quality, min_interval_width(cell.n) / 2.0)
        lo = max(0.0, cell.mean_quality - spread)
        hi = min(1.0, cell.mean_quality + spread)
        if hi - lo < min_interval_width(cell.n):
            hi = min(1.0, lo + min_interval_width(cell.n))
            lo = max(0.0, hi - min_interval_width(cell.n))
        return (lo, hi)

    def _find_existing(self, strategy_id: str,
                       predicates: Dict[str, Any],
                       *, include_dormant: bool = True) -> Optional[StrategicEntry]:
        """The entry this evidence belongs to, if any.

        - an entry whose predicates already cover the new ones (the
          restatement case, including a family-free pattern a harness wrote);
        - otherwise the entry of the same (family, structural cell) evidence
          set: one evidence set owns exactly one claim, so growing evidence
          REFRESHES that claim instead of spawning a second one.

        Dormant entries are INCLUDED by default. Excluding them meant a
        dormant claim was invisible to the dedup pass, so induction created a
        fresh entry and then ``revise`` woke the old one — two entries for one
        knowledge object. Waking it and refreshing it are offline decisions
        made here, on the same entry id."""
        covering = self._find_covering(strategy_id, predicates,
                                       include_dormant=include_dormant)
        if covering is not None:
            return covering
        # Otherwise: the entry of the SAME CELL. Matching on the family alone
        # was the defect that merged structurally opposite regions (the fallback
        # below used to accept any entry naming the same family, so the second
        # cell "refreshed" the first cell's claim). The cell is fully described
        # by (family, the grouping-dimension predicates), so that is what is
        # compared — including the unknown bucket.
        if predicates.get("family") is None:
            return None
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=include_dormant):
            if self._same_cell(entry.predicates, predicates):
                return entry
        return None

    @staticmethod
    def _same_cell(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """True when two predicate sets describe the same structural cell."""
        if a.get("family") != b.get("family"):
            return False
        for f in GROUPING_FEATURES:
            if a.get(f) != b.get(f):
                return False
        return True

    def _find_covering(self, strategy_id: str,
                       predicates: Dict[str, Any],
                       *, include_dormant: bool = True) -> Optional[StrategicEntry]:
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=include_dormant):
            if predicates_cover(entry.predicates, predicates):
                return entry
        return None

