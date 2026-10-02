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
    kind: str  # "new_claim" | "revision" | "pattern" | "pattern_unavailable"
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
    #: Why this candidate cannot be used at all (e.g. the comparison lost a
    #: whole side to an exclusion). None when the candidate is usable.
    #: A non-None value means `material_state` reports ``unavailable``: the
    #: hint is shown with its reason, never upgraded into a claim.
    material_problem: Optional[str] = None
    #: WHAT an induction from this candidate is FOR. ``method_induction`` —
    #: the evidence holds a comparison/recovery worth abstracting into a
    #: technique (the detectors' own candidates). ``statistical_refresh`` —
    #: a structural cell has enough independent evidence to create or
    #: refresh a STATISTICAL claim (its quality/cost/failure estimate). The
    #: two answer different questions and must not be conflated: a cell
    #: clearing the count gate is NOT by itself a reason to abstract a
    #: method.
    purpose: str = "statistical_refresh"
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
            "purpose": self.purpose,
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
            "material_problem": self.material_problem,
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InductionCandidateBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            kind=str(data.get("kind", "new_claim")),
            purpose=str(data.get("purpose", "statistical_refresh")),
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
            material_problem=data.get("material_problem"),
            created_at=float(data.get("created_at", time.time())),
        )

    def material_state(self) -> Dict[str, Any]:
        """Whether this bundle carries enough METHOD material to induce from.

        Four states; the distinction matters because "a plan is not a
        performed method" is only fatal to ONE kind of claim:

        * ``unavailable`` — the candidate cannot be used at all (its
          comparison lost a whole side to an exclusion). Reported with the
          reason, never presented as a claim.
        * ``insufficient`` — nothing to abstract: no method content at all,
          or a PATTERN candidate (a how-to abstraction) whose evidence
          reports only PLANNED methods. A plan does not show how the work was
          actually done, so it cannot ground a technique claim on its own —
          this judgment is unchanged for how-to claims.
        * ``sufficient_limited`` — a CELL/statistical candidate whose
          evidence reports PLANNED methods with at least one task-level
          check PASSED. This grounds a CONDITIONAL FACT ("under condition C,
          method M produced a checked-correct answer"), NOT a performed
          technique: "I observed the work run" is not part of what such a
          claim may assert. The limitation is carried in the state and every
          evidence entry, so a reader sees the basis is ``planned_only``.
        * ``sufficient`` — at least one PERFORMED method (``actual``) is on
          record, or the agent stated the method in a claim of its own (which
          is checked at submission, not here).

        The state is derived from the frozen evidence, never from a threshold
        the caller can lower. ``purpose`` splits how-to from conditional-fact
        claims; it never lets a claim assert a performed method it did not
        observe.
        """
        if self.material_problem:
            return {"state": "unavailable",
                    "reason": self.material_problem}
        if not self.methods:
            return {"state": "insufficient",
                    "reason": ("no supporting execution reports a method: a "
                               "strategy name and a mean are not a "
                               "technique, and none will be invented from "
                               "them")}
        performed = [m for m in self.methods if m.get("actual")]
        planned_only = [m for m in self.methods
                        if m.get("planned") and not m.get("actual")]
        if performed:
            return {"state": "sufficient",
                    "n_with_method": len(performed),
                    "n_planned_only": len(planned_only),
                    "n_supporting": len(self.methods)}
        # No PERFORMED method. Whether the PLAN alone can ground a claim
        # depends on WHAT the claim is for.
        if not planned_only:
            # No method content anywhere (every entry is a bare name/mean).
            return {"state": "insufficient",
                    "reason": ("no supporting execution reports a method: a "
                               "strategy name and a mean are not a "
                               "technique, and none will be invented from "
                               "them")}
        if self.purpose == "method_induction":
            # A how-to abstraction: a plan is intent, not an observation.
            return {"state": "insufficient",
                    "reason": ("the evidence reports only PLANNED methods: a "
                               "plan is intent, not a performed method, so "
                               "there is nothing observed to abstract a "
                               "technique from (record what actually ran — "
                               "`execute --method` is the plan; the solve "
                               "script's 'method_performed' receipt is the "
                               "observation)"),
                    "n_planned_only": len(planned_only),
                    "n_supporting": len(self.methods)}
        # A conditional-FACT candidate (statistical refresh). A plan plus a
        # PASSED task check grounds "under condition C, method M produced a
        # checked-correct answer" — a limited claim that does NOT assert the
        # work was observed running. Require at least one checked pass so the
        # claim never rests on an unjudged answer.
        checked_pass = [m for m in self.methods
                        if (m.get("task_check") or {}).get("state") == "passed"]
        if checked_pass:
            return {"state": "sufficient_limited",
                    "basis": "planned_only",
                    "n_planned_only": len(planned_only),
                    "n_checked_pass": len(checked_pass),
                    "n_supporting": len(self.methods),
                    "reason": ("the evidence reports PLANNED methods with a "
                               "passed task check: this grounds a "
                               "CONDITIONAL FACT (condition -> method -> "
                               "checked outcome), NOT a performed technique. "
                               "A claim from it must not assert the method "
                               "was observed running; a stronger claim needs "
                               "a 'method_performed' receipt")}
        return {"state": "insufficient",
                "reason": ("the evidence reports only PLANNED methods and no "
                           "task-level check has PASSED: a plan plus an "
                           "unverified answer grounds nothing"),
                "n_planned_only": len(planned_only),
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
            # The TASK-level verdict, frozen with the method material: a
            # PLANNED-only record can ground a conditional-fact claim ONLY
            # when its answer was checked and PASSED, so the state reader
            # needs this here (never re-read live, so the frozen bundle and
            # its state agree).
            "task_check": copy.deepcopy(
                (rec.execution_features or {}).get("task_check")),
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
    own cross-execution references intact.

    A hint is RE-VALIDATED against the CURRENT evidence set before it
    becomes a candidate, because later events change what it may claim:

    * an execution that has been excluded (or is no longer attempt-scope
      evidence) drops out of the citation. When that removes a REQUIRED
      side — a contrast's other strategy, a recovery's failed attempt — the
      hint can no longer be reproduced from live evidence, so the candidate
      is SKIPPED with a reason rather than presented carrying a stale,
      half-excluded comparison. A hint that merely loses one of several
      supporting records on one side stays, with its evidence pruned.
    * the METHOD material is read from the LIVE records, not from the hint,
      so a later correction (a method added, a cost amended) is reflected.

    Dropped evidence is reported on the bundle's trigger reasons, so a
    reader can see that the claim rests on a reduced set.
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
            all_ids = evidence_execution_ids(hint.evidence)
            live_ids = [eid for eid in all_ids if eid in by_id]
            missing = [eid for eid in all_ids if eid not in by_id]
            gap = _missing_side_problem(hint, all_ids, live_ids)
            if gap is None:
                # The hint fired online, before later facts existed: a task
                # check may since have REFUTED the side it calls a success,
                # at which point the recovery it describes is no longer
                # true. Re-read the premise from live records.
                gap = _stale_premise_problem(hint, by_id)
            if gap is not None:
                # A whole side of the comparison is gone: the hint is not
                # reproducible from live evidence. Reported, not silently
                # upgraded into a claim.
                bundles.append(_skipped_pattern_bundle(hint, gap, missing))
                continue
            if not live_ids:
                continue
            # One bundle per (pattern, group, evidence set): re-recording an
            # execution must not multiply the same candidate.
            signature = (hint.pattern, hint.group_key,
                         tuple(sorted(live_ids)))
            if signature in seen:
                continue
            seen.add(signature)
            supporting = [by_id[eid] for eid in live_ids]
            tasks = sorted({r.task_id for r in supporting})
            profile = supporting[0].profile_snapshot
            # Every strategy the hint names, so a contrast's two sides are
            # both visible on the bundle.
            strategies = list(hint.strategy_ids) or sorted(
                {r.strategy_id for r in supporting})
            reasons = [f"detector {hint.pattern}: {hint.reason}"]
            if missing:
                reasons.append(
                    "evidence pruned: execution(s) "
                    f"{sorted(missing)} are no longer in the evidence set "
                    "(excluded or out of scope)")
            bundles.append(InductionCandidateBundle(
                bundle_id=InductionCandidateBundle.content_id(
                    "pattern", hint.group_key, strategies, live_ids,
                    pattern=hint.pattern),
                kind="pattern",
                purpose="method_induction",
                strategy_id=strategies[0] if strategies else "",
                family=(profile.family if profile is not None else ""),
                cell_token=group_key(profile).split("|", 1)[-1]
                if profile is not None else "",
                group_key=hint.group_key,
                execution_ids=sorted(live_ids),
                tasks=tasks,
                n_supporting=len(supporting),
                trigger_reasons=reasons,
                pattern=hint.pattern,
                evidence_refs=_pruned_evidence_refs(hint.evidence, live_ids),
                # Read from the LIVE records: a later correction to a method
                # or a cost is reflected, never a frozen copy of what the
                # record said when the hint first fired.
                methods=_method_material(supporting),
                material_problem=None,
            ))
    return bundles


def _skipped_pattern_bundle(hint: InductionHint, problem: str,
                            missing: Sequence[str]
                            ) -> InductionCandidateBundle:
    """A candidate that CANNOT be formed, reported rather than dropped.

    Silence would look like "no pattern here", when the truth is "the
    pattern's evidence no longer holds together". The bundle carries the
    reason so `orx induction-material` shows the reader why this hint is not
    a claim.
    """
    return InductionCandidateBundle(
        bundle_id=InductionCandidateBundle.content_id(
            "pattern_unavailable", hint.group_key, hint.strategy_ids,
            list(missing), pattern=hint.pattern),
        kind="pattern_unavailable",
        purpose="method_induction",
        strategy_id=hint.strategy_ids[0] if hint.strategy_ids else "",
        family="",
        cell_token=hint.group_key.split("|", 1)[-1],
        group_key=hint.group_key,
        execution_ids=[],
        tasks=[],
        n_supporting=0,
        trigger_reasons=[f"detector {hint.pattern}: {hint.reason}",
                         problem],
        pattern=hint.pattern,
        evidence_refs=[],
        methods=[],
        material_problem=problem,
    )


def _stale_premise_problem(hint: InductionHint,
                           by_id: Dict[str, Any]) -> Optional[str]:
    """Whether a live task check has refuted the hint's own premise.

    The detectors read the SOLVER's quality, which speaks about the model,
    not the task. A later ``check-task`` can reveal that the record a
    recovery hint calls "recovered" actually FAILED the task — the answer
    was wrong however optimal the solve. Reusing that hint offline would
    present a refuted success as the recovered side of a repair, so the
    premise is re-read here and the candidate is withdrawn when it no
    longer holds.

    Only the recovery pattern states such a premise (the "after" side is a
    usable success, the "before" side a real failure); the comparison
    patterns say nothing about answer validity.
    """
    if hint.pattern != "intervention_recovery":
        return None
    kind = str((hint.evidence or {}).get("kind") or "")
    if kind == "same_solver_intervention":
        recovered = (hint.evidence.get("recovered_by") or {}) \
            .get("execution_id")
        failed = (hint.evidence.get("failed") or {}).get("execution_id")
    elif kind == "cross_execution_recovery":
        recovered = (hint.evidence.get("recovered_by") or {}) \
            .get("execution_id")
        failed = (hint.evidence.get("failed") or {}).get("execution_id")
    else:
        return None
    from or_harness.core.schema import task_check_state
    after = by_id.get(recovered) if recovered else None
    before = by_id.get(failed) if failed else None
    if after is not None and task_check_state(after) == "failed":
        return ("the record this hint calls the recovered success has since "
                "FAILED its task check: the answer was wrong however optimal "
                "the solve, so the recovery it describes is no longer true")
    if before is not None and task_check_state(before) == "passed":
        return ("the record this hint calls the failed attempt has since "
                "PASSED its task check: the 'before' side is not a failure, "
                "so there is no demonstrated recovery")
    return None


def _missing_side_problem(hint: InductionHint, all_ids: Sequence[str],
                          live_ids: Sequence[str]) -> Optional[str]:
    """Whether the pruning removed a side the pattern needs, or None.

    A comparison is only reproducible when BOTH of its sides still hold
    evidence. The check reads the hint's own evidence blocks (per strategy /
    per family / per cell for a reversal, or the failed/recovered pair for a
    recovery), so losing one whole side is caught however the ids were laid
    out — not only when ALL ids vanish.
    """
    if not all_ids:
        return None
    live = set(live_ids)
    groups = _evidence_side_groups(hint.evidence)
    if not groups:
        return None
    lost = [side for side, ids in groups.items()
            if ids and not (set(ids) & live)]
    if len(groups) >= 2 and lost:
        return ("the comparison can no longer be reproduced from live "
                f"evidence: side(s) {sorted(lost)} hold no record that still "
                "counts (excluded or out of scope)")
    return None


def _evidence_side_groups(evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    """The distinct evidence SIDES a hint's payload compares, as
    ``{side label: [execution ids]}``.

    * contrast / reproduction: ``execution_ids`` is ``{strategy: [ids]}``;
    * advantage reversal: the two nested cells;
    * recovery: the ``failed`` and ``recovered_by`` blocks.
    """
    groups: Dict[str, List[str]] = {}
    ids_map = evidence.get("execution_ids")
    if isinstance(ids_map, dict):
        for label, value in ids_map.items():
            ids = [v for v in (value or []) if isinstance(v, str)]
            if ids:
                groups[str(label)] = ids
    for key in ("advantageous_cell", "adverse_cell"):
        cell = evidence.get(key)
        if isinstance(cell, dict):
            ids = [v for v in (cell.get("execution_ids") or [])
                   if isinstance(v, str)]
            if ids:
                groups[str(key)] = ids
    for key in ("failed", "recovered_by"):
        block = evidence.get(key)
        if isinstance(block, dict) and isinstance(
                block.get("execution_id"), str):
            groups[str(key)] = [block["execution_id"]]
    return groups


def _pruned_evidence_refs(evidence: Dict[str, Any],
                          live_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """A copy of the hint's evidence with dead ids removed.

    The reference is kept for the reader (what was compared), but an id that
    no longer counts is struck from it: a candidate must not carry a pointer
    to an execution the evidence set has withdrawn.
    """
    live = set(live_ids)

    def _prune(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _prune(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_prune(v) for v in value]
        if isinstance(value, str) and value.startswith("ex_"):
            return value if value in live else None
        return value

    pruned = _prune(copy.deepcopy(evidence))
    return [pruned]


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
            purpose="statistical_refresh",
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
