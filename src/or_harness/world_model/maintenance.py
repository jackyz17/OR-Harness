"""M4 maintenance layer: traceable induction candidate bundles + offline
consequence and value assessment.

M4 connects experience accumulation to offline strategic induction:
1. Candidate bundling (:func:`build_induction_candidates`) produces frozen,
   traceable :class:`InductionCandidateBundle` objects from REAL executed
   evidence, from TWO sources: the online detectors' own persisted hints
   (strategy_contrast, intervention_recovery, structural_reproduction,
   advantage_reversal - each carrying its cross-execution evidence) and the
   structural-cell statistics (with the sample-count evidence gate). Sample
   count is an EVIDENCE THRESHOLD, never by itself a reason to abstract a
   method.
2. Every bundle also freezes per-execution METHOD material, so the outer
   agent can read "condition -> how it was done -> what followed ->
   boundary" and submit a relation claim; a bundle whose evidence reports no
   method is marked ``insufficient`` and nothing is invented from a name and
   a mean.
3. Value assessment, explicit choice and consequence binding are NOT
   duplicated here: they already live on the capability channel
   (``predict_capability_evolution`` / ``accept_capability_operation`` /
   ``reject_capability_operation`` / ``bind_capability_maintenance``), which
   consumes a bundle directly.

This is a MAINTENANCE-SCOPE facility — it does NOT modify the online M3
search tree, does not run background loops, and does not create a third
knowledge bank.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import ProblemProfile, group_key
from or_harness.core.storage import StorageError
from or_harness.strategy.triggers import (
    PATTERNS,
    InductionHint,
    evidence_execution_ids,
)
from or_harness.world_model.prediction import ActionSpec
from or_harness.world_model.state import MAINTENANCE_TASK_ID


@dataclass
class InductionCandidateBundle:
    """A frozen, traceable bundle of real execution evidence supporting
    an induction or revision candidate.

    Carries the candidate kind (new claim vs revision of an existing
    entry), the target strategy/family/cell, the exact execution IDs,
    the frozen statistical and quality summaries, and the trigger reasons
    that flagged it. Never dynamically re-queries the bank — once
    constructed, its content is fixed."""

    bundle_id: str
    kind: str  # "new_claim" | "revision" | "pattern"
    strategy_id: str
    family: str
    cell_token: str
    group_key: str
    execution_ids: List[str]
    tasks: List[str]
    n_supporting: int
    trigger_reasons: List[str]
    # Frozen summary of the supporting evidence.
    mean_quality: Optional[float] = None
    mean_cost: Dict[str, float] = field(default_factory=dict)
    cost_measured: List[str] = field(default_factory=list)
    failure_rate: float = 0.0
    # Revision-specific: the existing entry being revised and its state
    # at the time the bundle was formed.
    target_entry_id: Optional[str] = None
    entry_before: Optional[Dict[str, Any]] = None
    #: For a PATTERN bundle: which detector produced it, and the detector's
    #: OWN evidence blocks (both sides of a contrast, the failed/recovered
    #: pair). The evidence references travel WITH the candidate so the two
    #: sides of a contrast are never split into unrelated statistical packs.
    pattern: Optional[str] = None
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-execution METHOD evidence, frozen so an offline material read (and
    #: therefore an honest induction) has the how-to content, not just a name
    #: and a mean. Each entry: {"execution_id", "planned", "actual",
    #: "trajectory"}. Empty when the evidence reports no method — material
    #: insufficiency is then VISIBLE rather than papered over.
    methods: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        """A fresh random candidate id (used only where no address exists)."""
        return f"cb_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def content_id(kind: str, group_key_: str, strategy_ids: Sequence[str],
                   execution_ids: Sequence[str],
                   pattern: Optional[str] = None) -> str:
        """Deterministic id from what the candidate IS.

        Bundles are rebuilt from current evidence on every scan (they are
        not persisted), so a random id would make the id `orx
        induction-candidates` printed useless one call later — the id the
        user passes to `--bundle` must still name the same candidate. The
        address covers the candidate's identity (kind, pattern, group,
        strategies, evidence set) and nothing volatile (no timestamps, no
        statistics), so regenerating candidates over unchanged evidence
        yields the same ids.
        """
        address = {
            "kind": kind,
            "pattern": pattern or "",
            "group": group_key_,
            "strategies": sorted(strategy_ids),
            "executions": sorted(execution_ids),
        }
        payload = json.dumps(address, sort_keys=True, separators=(",", ":"))
        return "cb_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "kind": self.kind,
            "strategy_id": self.strategy_id,
            "family": self.family,
            "cell_token": self.cell_token,
            "group_key": self.group_key,
            "execution_ids": list(self.execution_ids),
            "tasks": list(self.tasks),
            "n_supporting": int(self.n_supporting),
            "trigger_reasons": list(self.trigger_reasons),
            "mean_quality": self.mean_quality,
            "mean_cost": copy.deepcopy(self.mean_cost),
            "cost_measured": list(self.cost_measured),
            "failure_rate": float(self.failure_rate),
            "target_entry_id": self.target_entry_id,
            "entry_before": copy.deepcopy(self.entry_before),
            "pattern": self.pattern,
            "evidence_refs": copy.deepcopy(self.evidence_refs),
            "methods": copy.deepcopy(self.methods),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InductionCandidateBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            kind=str(data.get("kind", "new_claim")),
            strategy_id=str(data.get("strategy_id", "")),
            family=str(data.get("family", "")),
            cell_token=str(data.get("cell_token", "")),
            group_key=str(data.get("group_key", "")),
            execution_ids=list(data.get("execution_ids") or []),
            tasks=list(data.get("tasks") or []),
            n_supporting=int(data.get("n_supporting", 0)),
            trigger_reasons=list(data.get("trigger_reasons") or []),
            mean_quality=data.get("mean_quality"),
            mean_cost=dict(data.get("mean_cost") or {}),
            cost_measured=list(data.get("cost_measured") or []),
            failure_rate=float(data.get("failure_rate", 0.0)),
            target_entry_id=data.get("target_entry_id"),
            entry_before=dict(data["entry_before"]) if data.get("entry_before") else None,
            pattern=data.get("pattern"),
            evidence_refs=list(data.get("evidence_refs") or []),
            methods=list(data.get("methods") or []),
            created_at=float(data.get("created_at", time.time())),
        )

    def material_state(self) -> Dict[str, Any]:
        """Whether this bundle carries enough METHOD material to induce from.

        A candidate whose evidence reports no method content (only a
        strategy name and a mean) is ``insufficient``: the framework refuses
        to turn a name plus a number into a technique. The state is derived
        from the frozen evidence, never from a threshold the caller can
        lower — an empty method list IS the insufficiency.
        """
        described = [m for m in self.methods
                     if (m.get("planned") or m.get("actual"))]
        if not self.methods:
            return {"state": "insufficient",
                    "reason": ("no supporting execution reports a method: a "
                               "strategy name and a mean are not a "
                               "technique, and none will be invented from "
                               "them")}
        if not described:
            return {"state": "insufficient",
                    "reason": ("the supporting executions report no method "
                               "content: there is nothing to abstract")}
        return {"state": "sufficient",
                "n_with_method": len(described),
                "n_supporting": len(self.methods)}


def _method_material(records: Sequence[Any]) -> List[Dict[str, Any]]:
    """Freeze the METHOD evidence of a set of executions.

    For each record: its planned method, the method it reports as actually
    performed, and the trajectory steps that actually happened. A record
    that reports no method contributes a placeholder with both sides None —
    so ``len(methods) == len(executions)`` always holds and an insufficiency
    is visible as missing content rather than as a silently shorter list.
    """
    material: List[Dict[str, Any]] = []
    for rec in records:
        material.append({
            "execution_id": rec.execution_id,
            "task_id": rec.task_id,
            "planned": copy.deepcopy(rec.method_planned),
            "actual": copy.deepcopy(rec.method_actual),
            "trajectory": [t.to_dict() for t in (rec.trajectory or [])],
        })
    return material


def build_induction_candidates(harness
                               ) -> List[InductionCandidateBundle]:
    """Scan the Experience Bank and Strategic Bank for induction candidates.

    Two sources, both frozen into the same bundle shape:

    * DETECTOR candidates (``kind="pattern"``): every persisted induction
      hint on a real executed record becomes a bundle that carries the
      detector's OWN evidence references. A contrast's two sides stay in
      ONE bundle — they are one observation, not two statistical bins.
    * CELL candidates (``kind="new_claim"`` / ``"revision"``): a structural
      cell with >= 2 executions and >= 2 independent tasks, when no
      published entry covers it (new claim) or an existing entry has
      accumulated misses / divergence / new evidence (revision).

    Every bundle freezes the METHOD material of its supporting executions so
    an offline material read (and therefore an honest induction) has the
    how-to content, not just a name and a mean.

    Returns a list of frozen :class:`InductionCandidateBundle` objects.
    Empty when no candidate has sufficient supporting evidence."""
    bundles: List[InductionCandidateBundle] = []
    # Collect all executed attempt-scope records.
    records = [r for r in harness.bank.all()
               if r.source == "executed" and r.measurement_scope == "attempt"]
    if not records:
        return bundles
    by_id = {r.execution_id: r for r in records}

    bundles.extend(_detector_candidates(records, by_id))
    bundles.extend(_cell_candidates(harness, records))
    return bundles


def _detector_candidates(records: Sequence[Any],
                         by_id: Dict[str, Any]
                         ) -> List[InductionCandidateBundle]:
    """Turn every persisted detector hint into a frozen candidate bundle.

    The hint was produced online, at record time, from the evidence that was
    available then. Re-reading it here (instead of re-running the detectors
    offline) keeps ONE implementation of each pattern and keeps the hint's
    own cross-execution references intact. A hint whose referenced
    executions are no longer in the evidence set (excluded, or a different
    scope) is skipped — a claim must rest on evidence that still counts.
    """
    bundles: List[InductionCandidateBundle] = []
    seen: set = set()
    for rec in sorted(records, key=lambda r: r.created_at):
        stored = (rec.execution_features or {}).get("induction_hints")
        if not isinstance(stored, list):
            continue
        for raw in stored:
            try:
                hint = InductionHint.from_dict(raw)
            except (ValueError, AttributeError, TypeError):
                continue  # a stored hint no detector produced is not evidence
            evidence_ids = [eid for eid in evidence_execution_ids(hint.evidence)
                            if eid in by_id]
            if not evidence_ids:
                continue
            # One bundle per (pattern, group, evidence set): re-recording an
            # execution must not multiply the same candidate.
            signature = (hint.pattern, hint.group_key,
                         tuple(sorted(evidence_ids)))
            if signature in seen:
                continue
            seen.add(signature)
            supporting = [by_id[eid] for eid in evidence_ids]
            tasks = sorted({r.task_id for r in supporting})
            profile = supporting[0].profile_snapshot
            # Every strategy the hint names, so a contrast's two sides are
            # both visible on the bundle.
            strategies = list(hint.strategy_ids) or sorted(
                {r.strategy_id for r in supporting})
            bundles.append(InductionCandidateBundle(
                bundle_id=InductionCandidateBundle.content_id(
                    "pattern", hint.group_key, strategies, evidence_ids,
                    pattern=hint.pattern),
                kind="pattern",
                strategy_id=strategies[0] if strategies else "",
                family=(profile.family if profile is not None else ""),
                cell_token=group_key(profile).split("|", 1)[-1]
                if profile is not None else "",
                group_key=hint.group_key,
                execution_ids=sorted(evidence_ids),
                tasks=tasks,
                n_supporting=len(supporting),
                trigger_reasons=[
                    f"detector {hint.pattern}: {hint.reason}"],
                pattern=hint.pattern,
                evidence_refs=[copy.deepcopy(hint.evidence)],
                methods=_method_material(supporting),
            ))
    return bundles


def _cell_candidates(harness,
                     records: Sequence[Any]
                     ) -> List[InductionCandidateBundle]:
    """Structural-cell candidates: the sample-count evidence gate.

    This is the SAMPLE gate, not a method abstraction: it reports that a cell
    has enough independent evidence to be worth an induction, or that an
    existing entry no longer matches the evidence. The how-to content (if
    any) travels in ``methods``; whether it is sufficient to abstract a
    technique is decided by ``material_state()``.
    """
    bundles: List[InductionCandidateBundle] = []
    # Group by (family, strategy_id, cell_token).
    by_cell: Dict[tuple, List[Any]] = {}
    for r in records:
        prof = r.profile_snapshot
        if prof is None:
            continue
        gkey = group_key(prof)
        # gkey format: "family=<name>|rc[..]|tc[..]|rx[..]"
        parts = gkey.split("|", 1)
        family = parts[0].removeprefix("family=") if "=" in parts[0] else parts[0]
        cell_token = parts[1] if len(parts) > 1 else ""
        key = (family, r.strategy_id, cell_token, gkey)
        by_cell.setdefault(key, []).append(r)

    # Check triggers against current stats.
    # Group existing entries by strategy.
    for (family, sid, cell_token, gkey), recs in sorted(by_cell.items()):
        # Deduplicate executions by execution_id.
        seen_ids = set()
        unique_recs = []
        for r in recs:
            if r.execution_id not in seen_ids:
                seen_ids.add(r.execution_id)
                unique_recs.append(r)
        if len(unique_recs) < 2:
            continue
        tasks = sorted({r.task_id for r in unique_recs})
        execution_ids = sorted(r.execution_id for r in unique_recs)
        cell = harness.stats.aggregate(gkey, sid, unique_recs)

        # Check if an entry already covers this.
        from or_harness.strategy.induction import evidence_predicates
        predicates = evidence_predicates(unique_recs, family=family)
        existing = harness.induction._find_existing(sid, predicates,
                                                    include_dormant=True)
        # Collect trigger reasons.
        trigger_reasons: List[str] = []
        if len(tasks) >= 2 and existing is None:
            trigger_reasons.append(
                f"sufficient independent evidence ({len(tasks)} tasks, "
                f"n={len(unique_recs)}) for new claim")
        elif existing is not None:
            # Check for substantive differences or misses.
            misses = (existing.prediction_track.consecutive_misses
                      if existing.prediction_track else 0)
            if misses >= 2:
                trigger_reasons.append(
                    f"existing entry {existing.entry_id} has {misses} "
                    "consecutive prediction misses")
            if abs(existing.expected_quality_hat - cell.mean_quality) > 0.05:
                trigger_reasons.append(
                    f"observed quality ({cell.mean_quality:.2f}) diverged from "
                    f"claim ({existing.expected_quality_hat:.2f})")
            if cell.n > existing.support_n:
                trigger_reasons.append(
                    f"new evidence available (n={cell.n} vs entry support_n="
                    f"{existing.support_n})")
        if not trigger_reasons:
            continue

        measured_dims = sorted(
            d for d, n in cell.n_measured.items() if n > 0)
        mean_cost = {d: float(getattr(cell.mean_cost, d, 0.0))
                     for d in measured_dims}

        bundle = InductionCandidateBundle(
            bundle_id=InductionCandidateBundle.content_id(
                "revision" if existing is not None else "new_claim",
                gkey, [sid], execution_ids),
            kind="revision" if existing is not None else "new_claim",
            strategy_id=sid,
            family=family,
            cell_token=cell_token,
            group_key=gkey,
            execution_ids=execution_ids,
            tasks=tasks,
            n_supporting=len(unique_recs),
            trigger_reasons=trigger_reasons,
            mean_quality=round(cell.mean_quality, 4),
            mean_cost={d: round(v, 4) for d, v in mean_cost.items()},
            cost_measured=measured_dims,
            failure_rate=round(cell.fail_rate, 4),
            target_entry_id=existing.entry_id if existing else None,
            entry_before=existing.to_dict() if existing else None,
            methods=_method_material(unique_recs),
        )
        bundles.append(bundle)

    return bundles
