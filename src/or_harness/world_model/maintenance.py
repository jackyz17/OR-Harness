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
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import (
    GROUPING_FEATURES,
    ProblemProfile,
    group_key,
    task_check_state,
)
from or_harness.core.storage import StorageError
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
    kind: str  # "new_claim" | "revision" | "cell_observation"
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
    #: Per-execution METHOD evidence, frozen so an offline material read (and
    #: therefore an honest induction) has the how-to content, not just a name
    #: and a mean. Each entry: {"execution_id", "planned", "actual",
    #: "trajectory", "task_check"}. An absent method is VISIBLE rather than
    #: papered over.
    methods: List[Dict[str, Any]] = field(default_factory=list)
    #: Why this candidate cannot be used at all (e.g. missing material). None
    #: when the candidate is usable. Non-None makes ``material_report``
    #: report ``unavailable``: shown with its reason, never upgraded.
    material_problem: Optional[str] = None
    #: A NOTE about what a CLAIM from this candidate may later assert (a thin
    #: cell, a single task's repeated runs). It never hides the material — it
    #: separates "this is reviewable" from "this is admissible as a
    #: transferable claim". None when the candidate is already admissible.
    admission_note: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        """A fresh random candidate id (used only where no address exists)."""
        return f"cb_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def content_id(kind: str, group_key_: str, strategy_ids: Sequence[str],
                   execution_ids: Sequence[str]) -> str:
        """Deterministic id from what the candidate IS.

        Bundles are rebuilt from current evidence on every scan (they are
        not persisted), so a random id would make the id `orx
        induction-candidates` printed useless one call later — the id the
        user passes to `--bundle` must still name the same candidate. The
        address covers the candidate's identity (kind, group, strategies,
        evidence set) and nothing volatile (no timestamps, no statistics),
        so regenerating candidates over unchanged evidence yields the same
        ids.
        """
        address = {
            "kind": kind,
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
            "methods": copy.deepcopy(self.methods),
            "material_problem": self.material_problem,
            "admission_note": self.admission_note,
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
            methods=list(data.get("methods") or []),
            material_problem=data.get("material_problem"),
            admission_note=data.get("admission_note"),
            created_at=float(data.get("created_at", time.time())),
        )

    def material_report(self) -> Dict[str, Any]:
        """WHAT THIS EVIDENCE CARRIES — a report, never a verdict.

        The framework does NOT decide whether "there is enough to abstract":
        that is the outer agent's judgment, made from the material itself.
        What the framework reports, from the frozen evidence, is the FACTS a
        reader needs:

        * ``unavailable`` — the candidate cannot be used at all (with the
          reason). Only ever set by an explicit ``material_problem``.
        * otherwise ``basis`` names the STRONGEST method content present:
          ``performed`` (at least one ``method_actual``), ``planned_only``
          (only plans, no performed method), or ``none`` (a name and a mean).
          ``n_supporting`` / ``n_planned_only`` / ``n_with_method`` /
          ``n_checked_pass`` report the counts, and ``missing`` lists the
          content that is absent (so the agent knows what to go read).

        No state says a claim is or is not admissible; the reviewer decides
        that with the material in hand.
        """
        if self.material_problem:
            return {"state": "unavailable", "reason": self.material_problem,
                    "missing": []}
        performed = [m for m in self.methods if m.get("actual")]
        planned_only = [m for m in self.methods
                        if m.get("planned") and not m.get("actual")]
        checked_pass = [m for m in self.methods
                        if (m.get("task_check") or {}).get("state") == "passed"]
        missing: List[str] = []
        if not performed:
            missing.append("method_performed (no execution reported how the "
                           "work actually ran; only a plan or nothing)")
        if not checked_pass:
            missing.append("task_check (no task-level check PASSED on the "
                           "evidence)")
        if performed:
            basis = "performed"
        elif planned_only:
            basis = "planned_only"
        else:
            basis = "none"
        return {
            "state": "reviewable",
            "basis": basis,
            "n_supporting": len(self.methods),
            "n_with_method": len(performed),
            "n_planned_only": len(planned_only),
            "n_checked_pass": len(checked_pass),
            "missing": missing,
            "note": ("a report of what the evidence carries — NOT an "
                     "admission judgment. Read the material and decide "
                     "yourself what (if anything) to abstract."),
        }


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
    """Scan the Experience Bank for structural-cell LEADS.

    Only CELL candidates are built now: the framework no longer turns
    online detector hits into induction candidates (the detectors were
    removed — the outer agent abstracts the method from the material
    itself; see ``orx review-material``, the default entry point).

    A CELL candidate is a structural cell, gated on NOTHING but visibility.
    A cell with >= 2 executions and >= 2 independent tasks is a
    ``new_claim`` (or ``revision``); a thin cell (fewer than 2 executions)
    or a single task's repeated runs is a ``cell_observation`` carrying an
    ``admission_note`` that says a transferable claim is not yet
    admissible. The material is ALWAYS visible — the sample count limits
    what a claim may later assert, never what may be read.

    Every bundle freezes the METHOD material of its supporting executions so
    an offline material read (and therefore an honest induction) has the
    how-to content, not just a name and a mean.

    Returns a list of frozen :class:`InductionCandidateBundle` objects."""
    records = [r for r in harness.bank.all()
               if r.source == "executed" and r.measurement_scope == "attempt"]
    if not records:
        return []
    return _cell_candidates(harness, records)


def _cell_candidates(harness,
                     records: Sequence[Any]
                     ) -> List[InductionCandidateBundle]:
    """Structural-cell candidates: a LEAD to look at, never an admission bar.

    The sample count used to be a hard gate (``< 2 executions -> continue``)
    and independent tasks used to be a hard precondition, so a cell that fired
    neither simply VANISHED from ``induction-candidates`` — even though its
    material was perfectly reviewable. Sample count is now a NOTE on the
    bundle: a thin cell is reported as ``cell_observation`` with an
    ``admission_note`` saying it is not yet publishable as a transferable
    claim, while its material stays visible. "At least two tasks" remains a
    PUBLICATION bar for transfer, never a bar on what may be read (the
    independent material entry point is ``orx review-material``).

    The how-to content (if any) travels in ``methods``; what the evidence
    carries is reported by ``material_report()`` — a report the agent reads,
    not an admission verdict.
    """
    bundles: List[InductionCandidateBundle] = []
    # Group by (anchor, strategy_id, cell_token). The anchor is the derived
    # PROBLEM CLASS (the first segment of the group key, ``class=<name>``) —
    # NOT the free-text family label, which is no longer a grouping anchor.
    by_cell: Dict[tuple, List[Any]] = {}
    for r in records:
        prof = r.profile_snapshot
        if prof is None:
            continue
        gkey = group_key(prof)
        # gkey format: "class=<name>|rc[..]|tc[..]|rx[..]"
        parts = gkey.split("|", 1)
        anchor = parts[0].removeprefix("class=") if "=" in parts[0] \
            else parts[0]
        cell_token = parts[1] if len(parts) > 1 else ""
        key = (anchor, r.strategy_id, cell_token, gkey)
        by_cell.setdefault(key, []).append(r)

    # Check triggers against current stats.
    # Group existing entries by strategy.
    for (anchor, sid, cell_token, gkey), recs in sorted(by_cell.items()):
        # Deduplicate executions by execution_id.
        seen_ids = set()
        unique_recs = []
        for r in recs:
            if r.execution_id not in seen_ids:
                seen_ids.add(r.execution_id)
                unique_recs.append(r)
        tasks = sorted({r.task_id for r in unique_recs})
        execution_ids = sorted(r.execution_id for r in unique_recs)
        cell = harness.stats.aggregate(gkey, sid, unique_recs)
        thin = len(unique_recs) < 2
        single_task = len(tasks) < 2

        # Check if an entry already covers this.
        from or_harness.strategy.induction import evidence_predicates
        # The predicate's ``family`` is the record's OWN family label — a
        # claim still speaks for the family it was induced from, even though
        # the grouping ANCHOR is the problem class.
        family = unique_recs[0].profile_snapshot.family
        predicates = evidence_predicates(unique_recs, family=family)
        existing = harness.induction._find_existing(sid, predicates,
                                                    include_dormant=True)
        # Collect trigger reasons — a LEAD to look at, not an admission bar.
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
        # A thin cell is a LEAD, not a silent drop. Its material is
        # reviewable; the sample count only limits what a CLAIM may later
        # assert. Report it under its own kind so the reader can tell a
        # not-yet-publishable observation from a publishable candidate.
        kind = ("revision" if existing is not None
                else ("new_claim" if not thin and not single_task
                      else "cell_observation"))
        admission_note = None
        if thin:
            trigger_reasons.append(
                f"thin cell: {len(unique_recs)} execution(s) — reviewable "
                "material, but NOT yet publishable as a transferable claim "
                "(needs >=2 executions)")
            admission_note = ("thin cell: fewer than 2 executions; material "
                              "is readable, a transferable claim is not "
                              "admissible yet")
        elif single_task:
            trigger_reasons.append(
                f"all {len(unique_recs)} executions come from one task "
                f"({tasks}) — a retry is repetition, not independent "
                "cross-task support")
            admission_note = ("single task: repeated runs are repetition, not "
                              "reproduction; a transferable claim needs >=2 "
                              "distinct tasks")
        if not trigger_reasons:
            continue

        measured_dims = sorted(
            d for d, n in cell.n_measured.items() if n > 0)
        mean_cost = {d: float(getattr(cell.mean_cost, d, 0.0))
                     for d in measured_dims}

        bundle = InductionCandidateBundle(
            bundle_id=InductionCandidateBundle.content_id(
                "revision" if existing is not None else kind,
                gkey, [sid], execution_ids),
            kind=kind,
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
            admission_note=admission_note,
        )
        bundles.append(bundle)

    return bundles


# ---------------------------------------------------------------------------
# Review material: read a BATCH of completed tasks WITHOUT a candidate gate
# ---------------------------------------------------------------------------
#
# The candidate path above starts from a DETECTOR or a sample-count gate, so a
# batch that fires neither (distinct tasks, a failed-only cell, a cell with a
# single run) never becomes visible for offline review. This entry point is
# the material SIDE of semantic induction: it organizes what a batch of
# COMPLETED tasks actually recorded so the outer agent can read it, compare
# and decide — including new, failed, cross-cell and cross-method-name
# material. It performs no statistical gating and creates no candidate: the
# only limits are a scope filter and a character BUDGET, and both are
# REPORTED so nothing is silently invisible.

#: Default character budget for one review-material batch. Overridden by
#: ``OR_HARNESS_REVIEW_MATERIAL_CHARS``. A batch over the budget is REPORTED
#: as truncated (with the omitted execution ids) and the caller narrows the
#: scope (`--strategy` / `--task` / `--limit`) or raises the budget — new
#: material is never made permanently invisible by missing a candidate.
#:
#: Larger than the world-model paired block (8000): induction compares
#: METHODS across tasks, so a useful batch is several full records, and one
#: record (profile + planned method + outcome) runs ~2 KB.
DEFAULT_REVIEW_MATERIAL_CHARS = 24000


def _review_material_budget() -> int:
    raw = os.environ.get("OR_HARNESS_REVIEW_MATERIAL_CHARS")
    if raw is None or not str(raw).strip():
        return DEFAULT_REVIEW_MATERIAL_CHARS
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_REVIEW_MATERIAL_CHARS
    return value if value >= 0 else DEFAULT_REVIEW_MATERIAL_CHARS


#: Characters of task text echoed with each material entry. Enough to name
#: the problem's semantics and key constraints without the whole prompt.
_REVIEW_TEXT_CHARS = 280
#: Characters kept per method step / failure error. A step is echoed to show
#: HOW the work was organised, not to reproduce the whole script.
_REVIEW_METHOD_STEP_CHARS = 200
#: Method steps echoed per side (planned / actual).
_REVIEW_MAX_STEPS = 8
#: Characters kept per failure's error text.
_REVIEW_ERROR_CHARS = 200
#: Trajectory steps echoed (the recent tail is kept when there are more).
_REVIEW_MAX_TRAJECTORY = 6


def _truncate(value: Any, limit: int) -> Any:
    """A string clipped to ``limit`` chars, with an explicit marker.

    Truncation is REPORTED (``…[+N chars]``) rather than silent: a clipped
    step must not read as if the step ended there.
    """
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit} chars]"


def _compact_cir(cir: Any) -> Dict[str, Any]:
    """A COUNT-based summary of a CIR, never the whole joint representation.

    The reviewer needs the problem's SHAPE (how many decisions, constraints,
    couplings, entities there are) to judge scale, not the full text of every
    element. Each count is a real field of the CIR; a missing block reports
    ``unknown`` rather than zero.
    """
    if not isinstance(cir, dict) or not cir:
        return {"unknown": "no CIR snapshot"}
    def _n(key: str) -> Any:
        value = cir.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        return value if value is not None else 0
    summary: Dict[str, Any] = {
        "n_decisions": _n("decisions"),
        "n_constraints": _n("constraints"),
        "n_coupling_groups": _n("coupling_groups"),
        "n_entities": _n("entities"),
        "n_relations": _n("relations"),
    }
    issues = cir.get("issues")
    if issues:
        summary["issues"] = [str(i)[:_REVIEW_METHOD_STEP_CHARS]
                             for i in (issues if isinstance(issues, list)
                                       else [issues])][:5]
    return summary


def _compact_method(method: Any) -> Optional[Dict[str, Any]]:
    """A method description with its steps clipped, or None when absent."""
    if not isinstance(method, dict):
        return None
    name = method.get("name")
    steps = method.get("steps") or []
    if not name and not steps:
        return None
    return {
        "name": name,
        "steps": [_truncate(s, _REVIEW_METHOD_STEP_CHARS)
                  for s in steps[:_REVIEW_MAX_STEPS]],
        "n_steps": len(steps),
        "truncated_steps": max(0, len(steps) - _REVIEW_MAX_STEPS),
    }


def _profile_summary(profile: Any) -> Dict[str, Any]:
    """The structural identity + grouping features, not the whole profile."""
    if profile is None:
        return {"unknown": "no profile snapshot"}
    return {
        "problem_class": getattr(profile, "problem_class", None),
        "family": getattr(profile, "family", None),
        **{f: getattr(profile, f, None) for f in GROUPING_FEATURES},
    }


def _review_entry(harness, record: Any,
                  *, previous: Any = None) -> Dict[str, Any]:
    """One completed task's reviewable material, compact and missing-marked.

    Every field is echoed from the RECORD itself (never re-derived from live
    state) so the batch is a faithful picture of what was observed. A field
    the record does not carry is reported as an explicit ``unknown`` marker
    rather than dropped, so a missing task check never hides a real method.

    The entry is COMPACT on purpose: the full CIR, solution vector, profile
    and host usage do NOT travel here (they are re-readable from the record
    by id when a claim needs them). What travels is what a reviewer compares:
    the problem's shape, the method actually organised, the KEY CHANGES
    against the previous same-task attempt (code hash, planned method), the
    outcome with its task check, the failures, and the measured cost.

    ``previous`` (the previous same-task attempt, when one exists) is used
    ONLY to compute the ``changes`` block — the framework reports the code
    hash and method differences as FACTS and never explains their meaning.
    """
    features = record.execution_features or {}
    task_check = features.get("task_check")
    planned = record.method_planned
    actual = record.method_actual
    if isinstance(actual, dict) and (actual.get("name") or actual.get("steps")):
        method_basis = "performed"
    elif isinstance(planned, dict) and (planned.get("name")
                                        or planned.get("steps")):
        method_basis = "planned_only"
    else:
        method_basis = "none"
    text = None
    if record.task_text_digest:
        stored = harness.store.get_task_text(record.task_id,
                                             record.task_text_digest)
        if stored is not None:
            text = _truncate(stored, _REVIEW_TEXT_CHARS)
    profile = record.profile_snapshot
    code_hash = (record.solver or {}).get("code_hash")
    entry: Dict[str, Any] = {
        "execution_id": record.execution_id,
        "task_id": record.task_id,
        "strategy_id": record.strategy_id,
        "family": (profile.family if profile is not None else None),
        "group_key": group_key(profile) if profile is not None else None,
        "created_at": record.created_at,
        # Task semantics: the text excerpt when the version is retained,
        # else an explicit marker (never a fabricated summary).
        "task_text": text if text is not None
                     else {"unknown": "task text not retained for this "
                                      "version"},
        # Problem SHAPE (counts), measured BEFORE modeling. Never the whole
        # CIR — a reviewer reads scale and coupling from the counts.
        "problem": {
            "profile": _profile_summary(profile),
            "cir": _compact_cir(record.cir_snapshot),
        },
        # Method evidence: the PLAN the agent declared and the method the run
        # reports it ACTUALLY performed (a plan is never promoted to a fact).
        "method": {
            "planned": _compact_method(planned),
            "actual": _compact_method(actual),
            "basis": method_basis,
        },
        "outcome": {
            "status": (record.quality or {}).get("status"),
            "feasible": (record.quality or {}).get("feasible"),
            "objective": (record.quality or {}).get("objective"),
            "gap": (record.quality or {}).get("gap"),
            # The executed code's own hash: the ONLY thing that can back a
            # "the code was unchanged" claim. Absent (never filled in) when
            # the record recorded none.
            "code_hash": code_hash,
        },
        # TASK verdict, a SEPARATE fact from the solver's own status: absent
        # is reported as never-checked, NOT as a pass. The reference
        # provenance (who supplied the reference value) travels with it.
        "task_check": {
            "state": (task_check_state(record) or "never_checked"),
            "basis": ((task_check or {}).get("scope") or {}).get("basis"),
            "reference_source": (task_check or {}).get("reference_source"),
        },
        # Flat mirror of the state, kept for callers that read it directly.
        "task_check_state": (task_check_state(record) or "never_checked"),
        "failures": [
            {"error": _truncate(f.error, _REVIEW_ERROR_CHARS),
             "error_class": f.error_class,
             "recovery_action": f.recovery_action,
             "action": f.action}
            for f in (record.failures or [])],
        "trajectory": [t.to_dict()
                       for t in (record.trajectory or [])[-_REVIEW_MAX_TRAJECTORY:]],
        "cost": record.cost.to_dict(),
        "cost_measured": (sorted(record.cost.measured)
                          if record.cost.measured is not None else None),
        "measurement_scope": record.measurement_scope,
    }
    if previous is not None:
        prev_hash = (previous.solver or {}).get("code_hash")
        prev_planned = previous.method_planned or {}
        planned_name = (planned or {}).get("name")
        prev_name = prev_planned.get("name")
        entry["changes"] = {
            "previous_execution_id": previous.execution_id,
            "code_hash": code_hash,
            "previous_code_hash": prev_hash,
            "code_hash_changed": (bool(code_hash) and bool(prev_hash)
                                  and code_hash != prev_hash),
            "method_planned_changed": ((planned_name or prev_name) is not None
                                       and planned_name != prev_name),
            "note": ("framework-reported DIFFERENCES against the previous "
                     "same-task attempt; they are facts, and the framework "
                     "does not interpret what caused them"),
        }
    return entry


def _existing_knowledge_for(harness, records: Sequence[Any]
                            ) -> List[Dict[str, Any]]:
    """Existing entries that could RELATE to this batch, with the WHY.

    Lets a reviewer see what knowledge ALREADY exists before deciding to add,
    revise or leave alone — including the claim text and verification state,
    so an entry that a new counterexample would change is visible.

    Matching is by CONTENT and provenance, not method-name equality alone:
    ``strategy_id`` equality, a claim that CITES one of this batch's
    executions, or overlapping applicability predicates. A claim-only entry
    with a free-form subject is therefore still offered for revision when its
    evidence or conditions overlap — the reviewer is never blind to a
    revisable entry just because its name differs from the strategy id.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    strategy_ids = {r.strategy_id for r in records}
    execution_ids = {r.execution_id for r in records}
    batch_families = {r.profile_snapshot.family for r in records
                      if r.profile_snapshot is not None}
    for entry in harness.sbank.list(include_dormant=True):
        if entry.entry_id in seen:
            continue
        matches: List[str] = []
        if entry.strategy_id in strategy_ids:
            matches.append("strategy_id")
        cited = {e.get("execution_id")
                 for e in ((entry.claim or {}).get("evidence") or [])}
        if cited & execution_ids:
            matches.append("cites this batch's evidence")
        entry_family = entry.predicates.get("family")
        if entry_family is not None and entry_family in batch_families:
            matches.append("same family")
        if not matches:
            continue
        seen.add(entry.entry_id)
        out.append({
            "entry_id": entry.entry_id,
            "strategy_id": entry.strategy_id,
            "claim_text": ((entry.claim or {}).get("text") or None),
            "kind": ((entry.claim or {}).get("kind") or None),
            "predicates": entry.predicates,
            "support_n": entry.support_n,
            "verification_state": entry.verification_state,
            "status": entry.status,
            "related_by": matches,
        })
    return out


def build_review_material(harness, *,
                          strategy_id: Optional[str] = None,
                          task_id: Optional[str] = None,
                          limit: Optional[int] = None,
                          cursor: Optional[str] = None
                          ) -> Dict[str, Any]:
    """Organize a BATCH of completed tasks for offline review.

    This is the independent MATERIAL entry point: it reads the Experience
    Bank directly and needs NO detector candidate and NO sample-count gate,
    so new, failed, cross-cell and cross-method-name material all reach the
    reviewer. Success and failure are BOTH included; a missing field is
    reported as ``unknown`` rather than dropping the rest of the fact.

    Scope filters (all optional) select which completed tasks are read:
    ``strategy_id`` / ``task_id`` narrow to one strategy or task. ``limit``
    caps how many entries are returned (the NEWEST are kept), and the
    character budget (``OR_HARNESS_REVIEW_MATERIAL_CHARS``) caps how much
    material travels — eviction is REPORTED, never silent.

    **Progress without repetition.** Each entry carries a ``cursor`` (its
    ``created_at`` and execution id). When the budget omits material, the
    response reports ``next_cursor`` = the cursor of the OLDEST entry it did
    return; passing that back as ``cursor`` continues with the next older
    batch. So a long history is walked in distinct batches rather than the
    same newest records being re-shown on every call — application code
    never has to guess how to page.

    Every attempt is kept in the flat ``material`` list (each failure with its
    own cost and source), and ``task_chains`` groups their execution ids per
    task in chronological order, so a same-task retry is never mistaken for
    independent cross-task support and no attempt is hidden.

    Read-only: nothing is written and no claim is formed here.
    """
    records = [r for r in harness.bank.all()
               if r.source == "executed" and r.measurement_scope == "attempt"]
    if strategy_id is not None:
        records = [r for r in records if r.strategy_id == str(strategy_id)]
    if task_id is not None:
        records = [r for r in records if r.task_id == str(task_id)]
    records.sort(key=lambda r: (r.created_at, r.execution_id))

    # Same-task attempts: the retry count is a fact the reviewer must see so
    # repeated runs of ONE instance are not read as cross-task support. The
    # previous same-task attempt (chronologically) is the baseline against
    # which the KEY CHANGES are reported.
    attempts_by_task: Dict[str, int] = {}
    kindex_by_task: Dict[str, int] = {}
    previous_of: Dict[str, Any] = {}
    task_chains: Dict[str, List[str]] = {}
    last: Dict[str, Any] = {}
    for r in records:
        attempts_by_task[r.task_id] = attempts_by_task.get(r.task_id, 0) + 1
        if r.task_id in last:
            previous_of[r.execution_id] = last[r.task_id]
        last[r.task_id] = r
        task_chains.setdefault(r.task_id, []).append(r.execution_id)

    total = len(records)
    truncated_by_limit = False
    if limit is not None and limit >= 0 and len(records) > limit:
        records = records[len(records) - limit:]  # keep the NEWEST
        truncated_by_limit = True

    # Cursor: continue with material OLDER than the supplied cursor.
    if cursor:
        boundary = _parse_review_cursor(cursor)
        if boundary is not None:
            records = [r for r in records
                       if (r.created_at, r.execution_id) < boundary]

    budget = _review_material_budget()
    # Budget eviction keeps the NEWEST material, matching ``--limit``: a
    # reviewer should see the most recent work first, and an older record is
    # what is omitted when the budget runs out. Pack from the newest
    # backwards, then restore chronological order for reading.
    entries_newest_first: List[Dict[str, Any]] = []
    used = 0
    omitted: List[str] = []
    for record in reversed(records):
        entry = _review_entry(harness, record,
                              previous=previous_of.get(record.execution_id))
        entry["attempts_of_task"] = attempts_by_task.get(record.task_id, 1)
        entry["attempt_index"] = None
        entry["independent_task"] = entry["attempts_of_task"] == 1
        entry["cursor"] = _review_cursor(record)
        size = len(json.dumps(entry, ensure_ascii=False, default=str))
        if entries_newest_first and used + size > budget:
            omitted.append(record.execution_id)
            continue
        entries_newest_first.append(entry)
        used += size
    material: List[Dict[str, Any]] = list(reversed(entries_newest_first))
    omitted.reverse()

    # Attempt indices, computed on the geometry of the FULL chronological
    # chain (so an evicted neighbour does not renumber what is shown).
    for entry in material:
        chain = task_chains.get(entry["task_id"]) or []
        if entry["execution_id"] in chain:
            entry["attempt_index"] = chain.index(entry["execution_id"]) + 1

    tasks = sorted({m["task_id"] for m in material})
    passed = sum(1 for m in material if m["task_check"]["state"] == "passed")
    failed = sum(1 for m in material if m["task_check"]["state"] == "failed")
    untested = len(material) - passed - failed
    reviewed = [harness.bank.get(m["execution_id"]) for m in material]
    # The cursor that reads the NEXT (older) batch, when material was
    # omitted: the oldest entry returned. None when everything fit.
    next_cursor = material[0]["cursor"] if (omitted and material) else None
    # CROSS-TASK AFFORDANCE. A batch of single-task claims is a symptom, not
    # a goal: a transferable claim needs the SAME mechanism observed on >=2
    # INDEPENDENT tasks, and that evidence has to come from more than one
    # batch. The framework reports WHICH task ids exist (here and, by
    # implication, beyond this batch) so the agent can deliberately look for
    # a mechanism that recurs across them instead of abstracting one task.
    n_tasks = len(tasks)
    return {
        "count": len(material),
        "total_completed": total,
        "tasks": tasks,
        "n_distinct_tasks": n_tasks,
        "n_attempts": len(material),
        "cross_task_hint": {
            "n_distinct_tasks_in_batch": n_tasks,
            "distinct_tasks_across_history": len(task_chains),
            "note": ("a TRANSFERABLE claim needs the same mechanism observed "
                     "on >=2 INDEPENDENT tasks (distinct task_id). This batch "
                     "spans "
                     + (f"{n_tasks} task(s); look for a mechanism that recurs "
                        "across them and across other tasks in the history — "
                        "not a restatement of one task."
                        if n_tasks >= 2 else
                        "1 task only; if you are about to publish from it, a "
                        "`conditional_fact` is a fact about THAT task, not "
                        "transferable knowledge — widen the task set (read "
                        "other batches / record more tasks) before claiming a "
                        "rule.")),
        },
        "check_states": {"passed": passed, "failed": failed,
                         "never_checked_or_insufficient": untested},
        "material": material,
        # Per-task chronological chains: every attempt id in order, so a
        # reviewer sees the whole history of one instance (not only the
        # final success).
        "task_chains": {t: task_chains[t] for t in tasks},
        "existing_knowledge": _existing_knowledge_for(
            harness, [r for r in reviewed if r is not None]),
        "budget": {
            "chars_used": used,
            "chars_limit": budget,
            "truncated_by_budget": bool(omitted),
            "omitted_execution_ids": omitted,
            "truncated_by_limit": truncated_by_limit,
            "next_cursor": next_cursor,
        },
        "note": ("read a BATCH of completed tasks directly — no detector "
                 "candidate and no sample-count gate is required. Success, "
                 "failure, cross-cell and cross-method-name material are all "
                 "here; a missing field is marked 'unknown', never dropped. "
                 "When material is omitted, pass budget.next_cursor back as "
                 "`cursor` to read the next (older) batch. Form a claim "
                 "(condition -> how -> consequence -> boundary) and submit it "
                 "with `orx induce --relation`."),
    }


def _review_cursor(record: Any) -> str:
    """A stable cursor for one entry: its created_at and execution id.

    ``created_at`` is serialized at FULL precision (``repr`` round-trips a
    float exactly). A rounded form (``:.6f``) could round UP, so the parsed
    boundary would no longer exclude the very entry it names — that entry
    would reappear in the next batch.
    """
    return f"{repr(float(record.created_at))}|{record.execution_id}"


def _parse_review_cursor(cursor: Any) -> Optional[tuple]:
    """Parse a cursor back into ``(created_at, execution_id)`` or None."""
    try:
        created, execution_id = str(cursor).split("|", 1)
        return (float(created), execution_id)
    except (ValueError, AttributeError):
        return None
