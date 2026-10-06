"""M4 maintenance layer: the offline INDUCTION MATERIAL entry point.

The framework ORGANIZES facts and REPORTS what they carry; the outer agent
reads that material and forms the strategy in its own words. This module
holds exactly ONE material entry point:

:func:`build_induction_material` reads the Experience Bank DIRECTLY and
organizes a batch of completed tasks — this task and related history, the
methods actually performed, the outcomes, the failures, the costs, the
before/after changes, the same-task attempt chains, paging (so a long
history is walked in distinct pages rather than the newest page standing in
for the whole bank) and the sources needed to re-read raw material, plus the
existing strategies a reader might extend, merge or revise. It performs NO
statistical gate and emits NO candidate verdict (no ``new_claim``, no
``cell_observation``, no "this cell cannot be induced"): independent task
counts, method names, cells and missing fields are FACTS the agent weighs,
never bars to reading or to citing evidence.

Nothing here writes knowledge, calls a model or runs in the background: the
knowledge WRITE lives on the ``induce`` path
(:meth:`or_harness.strategy.induction.InductionEngine.submit_relation`) and
the maintenance operations on the capability channel
(``predict_capability_evolution`` / ``accept_capability_operation`` /
``reject_capability_operation`` / ``bind_capability_maintenance`` /
``evaluate_capability_effect``). A capability prediction's frozen scope is
built from the evidence directly
(:func:`or_harness.world_model.capability_evolution
.learning_material_for_evidence`), not from a candidate generator.

The four observation angles the agent is prompted with — method contrast,
recovery after an intervention, structural reproduction, advantage reversal
— are THINKING AIDS in the guidance text, not detectors this module runs;
the old detectors are gone.

This is a MAINTENANCE-SCOPE facility — it does NOT modify the online M3
search tree, does not run background loops, and does not create a third
knowledge bank.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import (
    GROUPING_FEATURES,
    group_key,
    task_check_state,
)


# ---------------------------------------------------------------------------
# Induction material: read a BATCH of completed tasks WITHOUT any gate
# ---------------------------------------------------------------------------
#
# The framework's job here is to ORGANIZE FACTS and NAME THEIR SOURCES; the
# agent's job is to find the mechanism and decide the claim. So this entry
# point performs NO statistical gating and emits NO candidate verdict: it
# organizes what a batch of COMPLETED tasks actually recorded — new, failed,
# cross-cell and cross-method-name material alike — so the agent can read it,
# compare and decide. The only limits are a scope filter and a character
# BUDGET, and both are REPORTED so nothing is silently invisible. Paging walks
# a long history in distinct pages; the newest page standing in for the whole
# bank is exactly the failure this avoids.

#: Default character budget for one induction-material batch. Overridden by
#: ``OR_HARNESS_INDUCTION_MATERIAL_CHARS``. A batch over the budget is
#: REPORTED as truncated (with the omitted execution ids and a ``next_cursor``
#: to read the older page) — new material is never made permanently invisible
#: by missing a gate, and a bigger default is NOT the fix for that.
#:
#: Larger than the world-model paired block (8000): induction compares
#: METHODS across tasks, so a useful batch is several full records, and one
#: record (profile + planned method + outcome) runs ~2 KB.
DEFAULT_INDUCTION_MATERIAL_CHARS = 24000


def _induction_material_budget() -> int:
    raw = os.environ.get("OR_HARNESS_INDUCTION_MATERIAL_CHARS")
    if raw is None or not str(raw).strip():
        return DEFAULT_INDUCTION_MATERIAL_CHARS
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_INDUCTION_MATERIAL_CHARS
    return value if value >= 0 else DEFAULT_INDUCTION_MATERIAL_CHARS


#: Characters of task text echoed with each material entry. Enough to name
#: the problem's semantics and key constraints without the whole prompt.
_MATERIAL_TEXT_CHARS = 280
#: Characters kept per method step / failure error. A step is echoed to show
#: HOW the work was organised, not to reproduce the whole script.
_MATERIAL_METHOD_STEP_CHARS = 200
#: Method steps echoed per side (planned / actual).
_MATERIAL_MAX_STEPS = 8
#: Characters kept per failure's error text.
_MATERIAL_ERROR_CHARS = 200
#: Trajectory steps echoed (the recent tail is kept when there are more).
_MATERIAL_MAX_TRAJECTORY = 6


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
        summary["issues"] = [str(i)[:_MATERIAL_METHOD_STEP_CHARS]
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
        "steps": [_truncate(s, _MATERIAL_METHOD_STEP_CHARS)
                  for s in steps[:_MATERIAL_MAX_STEPS]],
        "n_steps": len(steps),
        "truncated_steps": max(0, len(steps) - _MATERIAL_MAX_STEPS),
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


def _material_entry(harness, record: Any,
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
            text = _truncate(stored, _MATERIAL_TEXT_CHARS)
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
            {"error": _truncate(f.error, _MATERIAL_ERROR_CHARS),
             "error_class": f.error_class,
             "recovery_action": f.recovery_action,
             "attempt": f.attempt}
            for f in (record.failures or [])],
        "trajectory": [t.to_dict()
                       for t in (record.trajectory or [])[-_MATERIAL_MAX_TRAJECTORY:]],
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


def build_induction_material(harness, *,
                             strategy_id: Optional[str] = None,
                             task_id: Optional[str] = None,
                             limit: Optional[int] = None,
                             cursor: Optional[str] = None
                             ) -> Dict[str, Any]:
    """Organize a BATCH of completed tasks as induction material.

    This is the ONE material entry point: it reads the Experience Bank
    directly and needs NO candidate and NO sample-count gate, so new, failed,
    cross-cell and cross-method-name material all reach the reviewer. Success
    and failure are BOTH included; a missing field is reported as ``unknown``
    rather than dropping the rest of the fact.

    Scope filters (all optional) select which completed tasks are read:
    ``strategy_id`` / ``task_id`` narrow to one strategy or task. ``limit``
    caps how many entries are returned (the NEWEST are kept), and the
    character budget (``OR_HARNESS_INDUCTION_MATERIAL_CHARS``) caps how much
    material travels — eviction is REPORTED, never silent.

    **Progress without repetition.** Each entry carries a ``cursor`` (its
    ``created_at`` and execution id). When the budget omits material, the
    response reports ``next_cursor`` = the cursor of the OLDEST entry it did
    return; passing that back as ``cursor`` continues with the next older
    batch. So a long history is walked in distinct batches rather than the
    same newest records being re-shown on every call — application code
    never has to guess how to page, and one page is never taken for the whole
    bank.

    Every attempt is kept in the flat ``material`` list (each failure with its
    own cost and source), and ``task_chains`` groups their execution ids per
    task in chronological order, so a same-task retry is never mistaken for
    independent cross-task support and no attempt is hidden.

    Read-only: nothing is written and no strategy is formed here.
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

    # Cursor FIRST: continue with material OLDER than the supplied cursor.
    # Applying the limit before the cursor would re-pick the same newest
    # records and then filter them all out, so a paged read would silently
    # stop after one page.
    if cursor:
        boundary = _parse_material_cursor(cursor)
        if boundary is not None:
            records = [r for r in records
                       if (r.created_at, r.execution_id) < boundary]

    truncated_by_limit = False
    if limit is not None and limit >= 0 and len(records) > limit:
        records = records[len(records) - limit:]  # keep the NEWEST
        truncated_by_limit = True

    budget = _induction_material_budget()
    # Budget eviction keeps the NEWEST material, matching ``--limit``: a
    # reviewer should see the most recent work first, and an older record is
    # what is omitted when the budget runs out. Pack from the newest
    # backwards, then restore chronological order for reading.
    entries_newest_first: List[Dict[str, Any]] = []
    used = 0
    omitted: List[str] = []
    for record in reversed(records):
        entry = _material_entry(harness, record,
                                previous=previous_of.get(record.execution_id))
        entry["attempts_of_task"] = attempts_by_task.get(record.task_id, 1)
        entry["attempt_index"] = None
        entry["independent_task"] = entry["attempts_of_task"] == 1
        entry["cursor"] = _material_cursor(record)
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
        "note": ("read a BATCH of completed tasks directly — no candidate "
                 "and no sample-count gate is required. Success, failure, "
                 "cross-cell and cross-method-name material are all here; a "
                 "missing field is marked 'unknown', never dropped. When "
                 "material is omitted, pass budget.next_cursor back as "
                 "`cursor` to read the next (older) batch. Form the new "
                 "strategy (condition -> how -> consequence -> boundary) and "
                 "submit it with `orx induce --relation`."),
    }


def _material_cursor(record: Any) -> str:
    """A stable cursor for one entry: its created_at and execution id.

    ``created_at`` is serialized at FULL precision (``repr`` round-trips a
    float exactly). A rounded form (``:.6f``) could round UP, so the parsed
    boundary would no longer exclude the very entry it names — that entry
    would reappear in the next batch.
    """
    return f"{repr(float(record.created_at))}|{record.execution_id}"


def _parse_material_cursor(cursor: Any) -> Optional[tuple]:
    """Parse a cursor back into ``(created_at, execution_id)`` or None."""
    try:
        created, execution_id = str(cursor).split("|", 1)
        return (float(created), execution_id)
    except (ValueError, AttributeError):
        return None
