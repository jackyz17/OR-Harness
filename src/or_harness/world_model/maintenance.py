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

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
DEFAULT_INDUCTION_MATERIAL_CHARS = 32000


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
#: Characters of task text echoed with each material entry — NO LONGER a
#: truncation limit. r14 returns the task text WHOLE (by version) so a
#: reviewer sees the cost coefficients and bounds a summary had dropped. The
#: row is kept only for backward compatibility with callers that read it.
_MATERIAL_TEXT_CHARS = 280
#: Characters kept per method step / failure error. NO LONGER limits: r14
#: returns steps and errors whole. Kept for backward compatibility.
_MATERIAL_METHOD_STEP_CHARS = 200
#: Method steps echoed per side (planned / actual). NO LONGER a cap.
_MATERIAL_MAX_STEPS = 8
#: Characters kept per failure's error text. NO LONGER a cap.
_MATERIAL_ERROR_CHARS = 200
#: Trajectory steps echoed. NO LONGER a cap.
_MATERIAL_MAX_TRAJECTORY = 6


def _truncate(value: Any, limit: int) -> Any:
    """A string clipped to ``limit`` chars, with an explicit marker.

    Retained for the few places that still need a bounded echo; the material
    entry no longer truncates task text, method steps, errors or the
    trajectory — a reviewer reads the selected records WHOLE.
    """
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit} chars]"


#: CIR keys whose CONTENT is kept (not just counted) when the snapshot
#: carries them. This is the EXISTING retained structure — the constraints'
#: expressions and kinds, the decisions'/entities' names, the relations and
#: coupling groups — never a new extraction scheme and never the full CIR.
#: The material still reports the counts beside them. A CIR that carries no
#: expression at all degrades to the counts (never fabricated).
_CIR_KEY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "constraints": ("id", "kind", "expr", "label", "name"),
    "decisions": ("id", "name", "kind", "label"),
    "entities": ("id", "name", "kind", "label"),
    "relations": ("id", "kind", "source", "target", "detail", "label"),
    "coupling_groups": ("id", "kind", "members", "resource", "detail"),
}


def _cir_key_structure(cir: Any) -> Dict[str, Any]:
    """The problem's SHAPE (counts) alongside its KEY RETAINED structure.

    r14: the counts alone dropped the constraints' expressions and bounds a
    reviewer needs. This keeps the CIR's EXISTING retained fields (each
    element's own ``id``/``kind``/``expr``/... from the snapshot) for the
    elements that carry them — the SAME structure the record already
    stores, not a new extraction and not the whole CIR. Every element is
    returned WHOLE (no per-element truncation); the caller's budget bounds
    the block as a unit. A CIR with no such fields degrades to the counts.
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
    # Retained structure: each element's own identifying/expressing fields.
    elements: Dict[str, Any] = {}
    for key, fields in _CIR_KEY_FIELDS.items():
        raw = cir.get(key)
        if not isinstance(raw, list):
            continue
        kept: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                row = {f: item[f] for f in fields if item.get(f) is not None}
                # Keep any other simple, non-derived field the element
                # carries so a constraint's coefficients/bounds written
                # under another key are not lost. Nested structures are
                # kept whole too (they are the existing content).
                for k, v in item.items():
                    if k in row:
                        continue
                    if isinstance(v, (str, int, float, bool, list, dict)) \
                            and v is not None:
                        row[k] = v
                if row:
                    kept.append(row)
            elif item is not None:
                kept.append(item)
        if kept:
            elements[key] = kept
    if elements:
        summary["elements"] = elements
    issues = cir.get("issues")
    if issues:
        summary["issues"] = list(issues if isinstance(issues, list)
                                 else [issues])
    return summary


def _compact_method(method: Any) -> Optional[Dict[str, Any]]:
    """A method description returned WHOLE, or None when absent.

    r14: ``steps``, ``why`` and ``fallback`` are no longer clipped and no
    longer capped at 8 steps — a reviewer comparing methods needs the
    ACTUAL steps and the reason a step works, and a bound or premise often
    sits in a later step. ``why``/``fallback``/``source`` travel in full.
    The block is counted as a unit against the caller's budget, so an
    oversized method is bounded as a WHOLE record, never field-by-field.
    """
    if not isinstance(method, dict):
        return None
    name = method.get("name")
    steps = method.get("steps") or []
    if not name and not steps:
        return None
    out: Dict[str, Any] = {
        "name": name,
        "steps": [" ".join(str(s).split()) for s in steps],
        "n_steps": len(steps),
        "truncated_steps": 0,
    }
    why = method.get("why")
    if why:
        out["why"] = " ".join(str(why).split())
    fallback = method.get("fallback")
    if fallback:
        out["fallback"] = " ".join(str(fallback).split())
    if method.get("source"):
        out["source"] = str(method["source"])
    return out


def _compact_checked(checked: Any) -> Any:
    """The task check's per-field verdicts, returned WHOLE, or None.

    r14: no 7-key whitelist and no 8-item cap. The check block carries what
    was actually compared (a reference value, a status, declared domains, an
    objective recomputation, explicit probes); every field and every item
    travels so a source warning never hides the real reason. A non-dict item
    is echoed whole rather than stringified to 200 chars.
    """
    if not isinstance(checked, list) or not checked:
        return None
    out: List[Any] = []
    for item in checked:
        if isinstance(item, dict):
            out.append({k: v for k, v in item.items()})
        else:
            out.append(item)
    return out


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
    """One completed task's reviewable material — WHOLE, missing-marked.

    r14: the entry echoes the SELECTED record's full content, not a summary.
    The task text is returned WHOLE and referenced by version (``task_text``
    for backward compatibility plus ``task_text_ref``; ``build_induction_
    material`` also collects the texts into a response-level ``task_texts``
    map so the same version is echoed ONCE per response, never per attempt).
    The problem's retained CIR structure, the method steps/why/fallback, the
    task check's checked fields, the failures and the trajectory all travel
    WHOLE — a reviewer reads what was actually recorded rather than a
    fixed-prefix excerpt.

    Every field is echoed from the RECORD itself (never re-derived from live
    state); a field the record does not carry is reported as an explicit
    ``unknown`` marker rather than dropped.

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
    # The task text of THIS version, WHOLE. A missing version is reported,
    # never replaced by another version's text.
    text = None
    text_ref: Any = {"unknown": "task text not retained for this version"}
    if record.task_text_digest:
        stored = harness.store.get_task_text(record.task_id,
                                             record.task_text_digest)
        if stored is not None:
            text = stored
            text_ref = {"task_id": record.task_id,
                        "task_text_digest": record.task_text_digest}
    profile = record.profile_snapshot
    code_hash = (record.solver or {}).get("code_hash")
    entry: Dict[str, Any] = {
        "execution_id": record.execution_id,
        "task_id": record.task_id,
        "strategy_id": record.strategy_id,
        "family": (profile.family if profile is not None else None),
        "group_key": group_key(profile) if profile is not None else None,
        "created_at": record.created_at,
        # Task semantics: the FULL text of this version (never truncated),
        # plus a reference so a caller can de-duplicate by version.
        "task_text": text if text is not None
                     else {"unknown": "task text not retained for this "
                                      "version"},
        "task_text_ref": text_ref,
        # Problem structure: the counts AND the CIR's retained key
        # structure (constraint expressions/kinds, decision/entity names,
        # relations, coupling groups). Never the full CIR.
        "problem": {
            "profile": _profile_summary(profile),
            "cir": _cir_key_structure(record.cir_snapshot),
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
        # provenance (who supplied the reference value) travels with it, and
        # the verdict's OWN details (the checked fields and the conclusion)
        # are carried WHOLE so a source warning never hides the real reason.
        "task_check": {
            "state": (task_check_state(record) or "never_checked"),
            "basis": ((task_check or {}).get("scope") or {}).get("basis"),
            "reference_source": (task_check or {}).get("reference_source"),
            "conclusion": (str((task_check or {}).get("conclusion") or "")
                           or None),
            "checked": _compact_checked((task_check or {}).get("checked")),
        },
        # Flat mirror of the state, kept for callers that read it directly.
        "task_check_state": (task_check_state(record) or "never_checked"),
        # Failures WHOLE (the error text is not clipped): a reviewer reads
        # the real failure, and the class/recovery/attempt travel beside it.
        "failures": [
            {"error": f.error,
             "error_class": f.error_class,
             "recovery_action": f.recovery_action,
             "attempt": f.attempt}
            for f in (record.failures or [])],
        # Trajectory WHOLE: the gap between planning and result is often the
        # most informative part, so no "last 6 steps" cap.
        "trajectory": [t.to_dict() for t in (record.trajectory or [])],
        # The answer's own variable values, when the run reported them.
        "solution_variables": (copy.deepcopy(
            features.get("solution_variables"))
            if isinstance(features.get("solution_variables"), dict)
            else None),
        # Citations of past executions this attempt was adapted from.
        "reuse_trace": (copy.deepcopy(features.get("reuse_trace"))
                        if isinstance(features.get("reuse_trace"), dict)
                        else None),
        "cost": record.cost.to_dict(),
        "cost_measured": (sorted(record.cost.measured)
                          if record.cost.measured is not None else None),
        "measurement_scope": record.measurement_scope,
        # Flat mirrors of the most-read fields, kept for callers that read
        # them directly (consistent with ``task_check_state``): the execution
        # status and the quality it observed.
        "status": (record.quality or {}).get("status"),
        "observed_quality": dict(record.quality or {}),
        # A read entry point for the FULL record (the material is already
        # whole, but the raw fact row is still the deepest read path).
        "inspect_hint": (
            f"read the FULL record by id `{record.execution_id}` "
            f"(its code hash {code_hash})"),
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


def _material_entry_by_id(harness, execution_id: str
                          ) -> Optional[Dict[str, Any]]:
    """Expand ONE already-selected execution id to its full material entry.

    The SHARED expansion both the current batch and ``related_history`` use,
    so a retrieved execution and a batch execution carry the SAME field set
    (a hit is not a lesser summary). Returns ``None`` when the fact is gone
    or is not part of the evidence set. Read-only.
    """
    record = harness.bank.get(execution_id)
    if record is None:
        record = harness.bank.get_pending(execution_id)
    if record is None:
        return None
    if str(record.source) not in ("executed", "staged"):
        return None
    return _material_entry(harness, record)


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
                             cursor: Optional[str] = None,
                             related_top_k: int = 5
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

    **Related history by SEMANTIC discovery (``related_top_k``).** Narrowing
    the batch to one task (``--task``) is how a per-task review is sped up,
    but it must not blind the reviewer to comparable work on OTHER tasks.
    When ``related_top_k > 0`` the module runs ONE retrieval whose query text
    is built from THIS task's own recorded METHOD (the performed method when
    one exists, else the plan with a basis marker) plus its structural
    summary — NO model call is made to write the query. Execution hits come
    back UNFILTERED (a failed or cross-cell record is exactly the material a
    boundary check needs), knowledge hits include UNPUBLISHED entries, and
    the result reports ``no_hits`` vs a retrieval FAILURE separately. A
    similarity hit is a DISCOVERY signal, never a support strength, and no
    hit is NOT evidence that no counterexample exists. ``related_top_k=0``
    disables the channel entirely (identical to the pre-r11 behaviour).

    Every attempt is kept in the flat ``material`` list (each failure with its
    own cost and source), and ``task_chains`` groups their execution ids per
    task in chronological order, so a same-task retry is never mistaken for
    independent cross-task support and no attempt is hidden.

    Read-only: nothing is written and no strategy is formed here.
    """
    records = [r for r in harness.bank.all()
               if r.source == "executed" and r.measurement_scope == "attempt"]
    # Distinct tasks across the WHOLE history, captured BEFORE the scope
    # filters: ``distinct_tasks_across_history`` must report how many tasks
    # the memory actually spans even when the caller narrowed the batch with
    # ``--task``. Computing it after the filter made it equal the batch's own
    # task count (always 1 under ``--task``) — the field name promised more
    # than it delivered.
    all_history_tasks = {str(r.task_id) for r in records if r.task_id}
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
    limit_dropped: List[Any] = []
    if limit is not None and limit >= 0 and len(records) > limit:
        # Keep the NEWEST ``limit`` records; the older ones are DROPPED but
        # must stay reachable through the cursor (see below). Dropping them
        # without a cursor was how a ``--limit`` read silently lost history.
        limit_dropped = records[:len(records) - limit]
        records = records[len(records) - limit:]
        truncated_by_limit = True

    budget = _induction_material_budget()
    # Budget eviction keeps a CONTIGUOUS newest run: pack from the newest
    # backwards and STOP at the first entry that does not fit. The old
    # behaviour skipped a too-large entry and kept packing OLDER ones — those
    # skipped records then fell on the wrong side of ``next_cursor`` and were
    # never reachable again. Stopping keeps every omitted record strictly
    # older than the oldest returned one, so one cursor walks the whole
    # history exactly once. The FIRST entry is always kept, however large, so
    # a single oversized record cannot stall progress.
    records_newest_first = list(reversed(records))
    entries_newest_first: List[Dict[str, Any]] = []
    # The response-level task-text version map: a version is echoed ONCE,
    # however many attempts reference it, and each entry points at it by
    # ``task_text_ref`` (its ``task_text`` still carries the full text for
    # callers that read the entry standalone).
    task_texts: Dict[str, Any] = {}
    charged_text_keys: set = set()
    used = 0
    stop_index: Optional[int] = None
    for index, record in enumerate(records_newest_first):
        entry = _material_entry(harness, record,
                                previous=previous_of.get(record.execution_id))
        entry["attempts_of_task"] = attempts_by_task.get(record.task_id, 1)
        entry["attempt_index"] = None
        entry["independent_task"] = entry["attempts_of_task"] == 1
        entry["cursor"] = _material_cursor(record)
        # The budget counts the WHOLE entry (task text included). The task
        # text lives ONCE in the ``task_texts`` map, so a repeating entry is
        # not billed for it again and a large shared text does not evict
        # unrelated records; everything else is counted whole (an oversized
        # method/failure/trajectory is bounded as a unit, never truncated).
        size = _entry_budget_size(entry, charged_text_keys)
        if entries_newest_first and used + size > budget:
            stop_index = index
            break
        entries_newest_first.append(entry)
        # Register this version in the response map (deduplicated) now that
        # the entry is included, and charge it once.
        ref = entry.get("task_text_ref")
        if isinstance(ref, dict) and ref.get("task_text_digest") is not None \
                and isinstance(entry.get("task_text"), str):
            task_texts.setdefault(
                f"{ref['task_id']}|{ref['task_text_digest']}",
                entry["task_text"])
            charged_text_keys.add(
                f"{ref['task_id']}|{ref['task_text_digest']}")
        used += size
    budget_dropped = (records_newest_first[stop_index:]
                      if stop_index is not None else [])
    material: List[Dict[str, Any]] = list(reversed(entries_newest_first))
    omitted = ([r.execution_id for r in budget_dropped]
               + [r.execution_id for r in limit_dropped])

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
    # The cursor that reads the NEXT (older) batch whenever ANYTHING was
    # dropped (by the budget OR by ``--limit``): the oldest entry RETURNED.
    # Because eviction is contiguous, every omitted record is strictly older
    # than it, so one cursor reaches them all — no gap, no repeat.
    next_cursor = None
    if omitted and material:
        oldest = records_newest_first[len(entries_newest_first) - 1]
        next_cursor = _material_cursor(oldest)
    # CROSS-TASK AFFORDANCE. The framework reports WHICH task ids exist here
    # and across the memory as FACTS — it does not turn the count into a
    # rule. A transferable claim is the agent's judgement, weighing whether a
    # mechanism recurs; two tasks are not a proof and one task can reveal a
    # conditional method with a derivation behind it.
    n_tasks = len(tasks)
    # RELATED HISTORY. Narrowing the batch to one task speeds the read up;
    # the retrieval below puts cross-task material back so the narrowing
    # never blinds the reviewer to a comparable method, a failure or a
    # boundary case on ANOTHER task.
    related_history = _related_history(
        harness, reviewed, related_top_k=related_top_k)
    memory_state = _memory_state(harness, related_history,
                                 n_records=len(material))
    return {
        "count": len(material),
        "total_completed": total,
        "tasks": tasks,
        "n_distinct_tasks": n_tasks,
        "n_attempts": len(material),
        # The STATE of the two banks, kept apart and reported honestly: an
        # empty strategic bank, an empty evidence bank, no matches and a
        # retrieval failure are DIFFERENT facts. Crucially, an empty bank is
        # NOT a gate: the agent may still solve and then induce from THIS
        # task's own evidence.
        "memory_state": memory_state,
        "cross_task_hint": {
            "n_distinct_tasks_in_batch": n_tasks,
            "distinct_tasks_across_history": len(all_history_tasks),
            "note": ("this batch spans "
                     + (f"{n_tasks} task(s) and the memory spans "
                        f"{len(all_history_tasks)} task(s) in total. Look for "
                        "a mechanism that recurs across tasks — but the "
                        "count is a FACT, not a threshold: a single task can "
                        "reveal a conditional method worth recording, and two "
                        "tasks are not by themselves a proof."
                        if n_tasks >= 2 else
                        "1 task only; the memory spans "
                        f"{len(all_history_tasks)} task(s) in total. A "
                        "`conditional_fact` is a fact about THAT task's "
                        "structure and method — state its scope; you may also "
                        "form a conditional method here if the derivation "
                        "supports it, or read other batches to find a "
                        "mechanism that recurs.")),
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
        "related_history": related_history,
        "budget": {
            "chars_used": used,
            "chars_limit": budget,
            "truncated_by_budget": bool(budget_dropped),
            "chars_dropped": len(budget_dropped),
            "truncated_by_limit": truncated_by_limit,
            "limit_dropped": len(limit_dropped),
            "omitted_execution_ids": omitted,
            "next_cursor": next_cursor,
            "cursor_note": (
                "every omitted record is strictly OLDER than next_cursor, so "
                "passing it back reads the next contiguous batch with no gap "
                "and no repeat; None means the whole history was returned"),
        },
        # The task-text VERSION map: each retained version echoed ONCE for
        # the whole response, keyed ``task_id|task_text_digest``; an entry
        # references its version via ``task_text_ref``. The same version is
        # never repeated per attempt.
        "task_texts": task_texts,
        "task_texts_note": (
            "each value is the WHOLE text of one task version, echoed once "
            "for this response; an entry's ``task_text_ref`` names the "
            "version it belongs to. A version not retained is absent (the "
            "entry says so) and is never replaced by another version"),
        # The SAVED joint H+ block (from the same strategy-outcome
        # predictions of this batch's executions), for the reviewer to
        # weigh alongside the evidence. EXPLANATORY ONLY: it never gates
        # induction, never ranks a candidate and never proves an effect.
        "joint_hplus": _joint_hplus_for(harness, reviewed),
        "guidance": {
            "structural_contrast": (
                "compare the constraint relations against the method's steps; "
                "do NOT call a method transferable because two tasks share a "
                "solver, a method name, or a 'both succeeded' outcome"),
            "key_step_explanation": (
                "say WHY the key step works and what premise it depends on; "
                "report speculation and incomplete derivations as such"),
            "boundary_check": (
                "which condition, if changed, makes the method fail? Separate "
                "'still correct but slower' from 'the method no longer "
                "holds'"),
        },
        "note": ("read a BATCH of completed tasks directly — no candidate "
                 "and no sample-count gate is required. Success, failure, "
                 "cross-cell and cross-method-name material are all here, "
                 "each record WHOLE; a missing field is marked 'unknown', "
                 "never dropped. Shared task-text versions live under "
                 "`task_texts` (referenced by `task_text_ref`), and the "
                 "batch's saved joint H+ stances under `joint_hplus` "
                 "(explanatory, not a gate). When material is omitted, pass "
                 "budget.next_cursor back as `cursor` to read the next "
                 "(older) batch. Form the new strategy (condition -> how -> "
                 "consequence -> boundary) and submit it with `orx induce "
                 "--relation`."),
    }


def _joint_hplus_for(harness, records: Sequence[Any]) -> Dict[str, Any]:
    """The SAVED joint H+ (capability-gain) stances of this batch.

    A strategy-outcome prediction carries BOTH the cost/benefit/risk AND the
    candidate's H+ stance in ONE answer; that stance is archived as a
    ``CapabilityTrace`` (see ``trace_archive``). This block surfaces it for
    the reviewer, keyed by the executions they were made for — the SAME
    binding the evidence uses (``trace.prediction_id`` ->
    ``candidate.{task_id,episode_id}``; ``trace.bound_execution_ids``).

    It does NOT call a model, does NOT touch the capability-evidence store
    and does NOT gate anything: an EMPTY ``capability_predictions`` bank does
    not mean "no H+ was predicted" — the joint block lives here. The FOUR
    states are kept apart (``expected`` / ``none`` / ``insufficient_basis`` /
    unstated), and an UNEXECUTED candidate keeps ``bound_execution_ids=[]``
    with ``executed=False`` (no observation is fabricated).
    """
    batch_tasks = {str(r.task_id) for r in records if r is not None}
    batch_execs = {str(r.execution_id) for r in records if r is not None}
    if not batch_tasks and not batch_execs:
        return {"items": [], "n": 0,
                "note": "no executions in this batch to associate H+ with"}
    try:
        from or_harness.world_model.trace_archive import (
            iter_capability_traces,
        )
        traces = iter_capability_traces(harness)
    except Exception as exc:  # noqa: BLE001 - a degraded read is reported
        return {"items": [], "n": 0,
                "failure": {"reason": f"{type(exc).__name__}: {exc}",
                            "note": ("the joint H+ archive could not be "
                                     "read; that is NOT 'no gain was "
                                     "predicted'")}}
    items: List[Dict[str, Any]] = []
    for trace in traces:
        bound = {str(e) for e in (trace.bound_execution_ids or [])}
        related = (str(trace.task_id) in batch_tasks) or bool(bound & batch_execs)
        if not related:
            continue
        assessment = str(getattr(trace, "assessment", "") or "")
        items.append({
            "prediction_id": trace.prediction_id,
            "strategy_id": trace.strategy_id,
            "task_id": trace.task_id,
            "episode_id": trace.episode_id,
            "assessment": (assessment or "unstated"),
            "claim": trace.claim,
            "applies_to": list(trace.applies_to),
            "expected_changes": copy.deepcopy(trace.expected_changes),
            "verification_conditions": copy.deepcopy(
                trace.verification_conditions),
            "uncertainty": list(trace.uncertainty),
            "bound_execution_ids": list(trace.bound_execution_ids),
            "executed": bool(trace.bound_execution_ids),
            "state": trace.state,
            "effect_state": trace.effect_state,
            "effect_verified": bool(trace.effect_verified),
        })
    return {
        "items": items,
        "n": len(items),
        "n_expected": sum(1 for i in items
                          if i["assessment"] == "expected"),
        "n_none": sum(1 for i in items if i["assessment"] == "none"),
        "n_insufficient_basis": sum(1 for i in items
                                    if i["assessment"] == "insufficient_basis"),
        "n_unstated": sum(1 for i in items if i["assessment"] == "unstated"),
        "note": ("the SAVED joint H+ stances of this batch's predictions, "
                 "for the reviewer to weigh. EXPLANATORY ONLY: it does not "
                 "gate induction, does not rank a candidate and is not a "
                 "post-hoc effect proof. `expected` does not require "
                 "publication, `none` does not forbid it, and an unexecuted "
                 "candidate has no observation (`executed=false`)."),
    }


def _query_text_for(records: Sequence[Any]) -> Dict[str, Any]:
    """Build the retrieval query from THIS batch's own recorded content.

    The query is assembled from what the records already carry — the METHOD
    that ran (``method_actual`` preferred, else ``method_planned`` with a
    basis marker) and the structural summary (family + coupling cell). It is
    deliberately NOT built from the outcome, the task number or the solver
    name, so the search is not biased toward successes or toward one tool;
    the solver and failure text can still be put in the query by the agent,
    which may rewrite it freely.

    NO model call is made: the text is a deterministic join of recorded
    fields. A record that reports no method yields a query with an explicit
    ``method: null`` marker — the plan is never presented as the performed
    fact.
    """
    names: List[str] = []
    steps: List[str] = []
    basis = "none"
    family = None
    for record in records:
        actual = getattr(record, "method_actual", None)
        planned = getattr(record, "method_planned", None)
        chosen = None
        if isinstance(actual, dict) and (actual.get("name")
                                         or actual.get("steps")):
            chosen, side = actual, "performed"
        elif isinstance(planned, dict) and (planned.get("name")
                                            or planned.get("steps")):
            chosen, side = planned, "planned_only"
        if chosen is not None:
            if side == "performed":
                basis = "performed" if basis != "performed" else basis
            elif basis == "none":
                basis = "planned_only"
            if chosen.get("name"):
                names.append(str(chosen["name"]))
            steps.extend(str(s) for s in (chosen.get("steps") or [])
                         if str(s).strip())
        profile = getattr(record, "profile_snapshot", None)
        if family is None and profile is not None:
            family = getattr(profile, "family", None)
    parts: List[str] = []
    if names:
        parts.append("methods " + ", ".join(dict.fromkeys(names)))
    if steps:
        parts.append("steps " + "; ".join(steps[:_MATERIAL_MAX_STEPS * 2]))
    if family:
        parts.append(f"family {family}")
    text = " ".join(parts).strip()
    return {
        "text": text,
        "method": ({"name": names[0] if names else "",
                    "steps": steps[:8], "basis": basis}
                   if names or steps else None),
        "basis": basis,
        "note": ("query built from THIS batch's recorded method and family; "
                 "no model call, and the outcome/task number/solver name are "
                 "NOT used so the search is not biased toward successes")
        if text else "no method or structure recorded on this batch: the "
                     "related-history query is empty",
    }


def _memory_state(harness, related_history: Dict[str, Any], *,
                  n_records: int) -> Dict[str, Any]:
    """Report the STATE of the two banks WITHOUT turning it into a gate.

    Cold-start honesty: an empty strategic bank, an empty evidence bank, no
    matching material and a retrieval failure are DIFFERENT facts, and each
    one means something SPECIFIC that is NOT "you cannot induce":

    - an empty strategic bank means there is no PRIOR knowledge to reuse — it
      does NOT mean there is nothing to learn from (the evidence bank may be
      full), and it does NOT force the agent to publish a first entry;
    - an empty evidence bank means nothing has been solved yet: solve the
      task and its own real evidence becomes the first material;
    - ``recall_no_match`` (both banks have content, nothing related) means the
      retrieval ran and found nothing comparable — NOT that no counterexample
      exists;
    - a retrieval FAILURE is reported separately and is never read as "no
      hits".

    The framework REPORTS; the agent decides whether to induce.
    """
    try:
        sbank_count = harness.sbank.count()
    except Exception:  # noqa: BLE001 - a read failure is reported, not fatal
        sbank_count = None
    try:
        bank_count = harness.bank.count()
    except Exception:  # noqa: BLE001
        bank_count = None
    strategic_empty = sbank_count == 0
    evidence_empty = (bank_count == 0) and n_records == 0
    both_empty = bool(strategic_empty and evidence_empty)
    retrieval_failed = related_history.get("failure") is not None
    retrieval_disabled = not related_history.get("enabled")
    # "no matches" only when the retrieval really RAN and came back empty.
    no_match = bool(related_history.get("no_hits")) and not retrieval_failed \
        and not retrieval_disabled and not both_empty

    notes: List[str] = []
    if both_empty:
        state = "both_banks_empty"
        notes.append(
            "BOTH banks are empty: propose a method from the task and the "
            "available tools, execute it, and THIS task's own real evidence "
            "is the first induction material. No prior knowledge is needed "
            "and none is fabricated; an empty bank is NOT a reason to refuse "
            "to induce, and NOT a reason to publish a first entry if nothing "
            "worth keeping was found.")
    elif strategic_empty:
        state = "strategic_bank_empty"
        notes.append(
            "the STRATEGIC bank is empty (no prior knowledge entries), but "
            "the evidence bank has real executions: induce from their "
            "methods, structures, failures and repairs. No prior knowledge "
            "is a precondition for nothing here.")
    elif evidence_empty:
        state = "evidence_bank_empty"
        notes.append(
            "the EVIDENCE bank has no executed attempts yet: solve the task "
            "first; its own evidence becomes the material.")
    elif no_match:
        state = "recall_no_match"
        notes.append(
            "both banks have content, but nothing related was retrieved: "
            "that is NOT evidence that no counterexample exists. Widen the "
            "read (`--cursor`) or add candidates before concluding.")
    else:
        state = "has_material"
    if retrieval_failed:
        notes.append(
            "related-history retrieval could NOT run (see "
            "related_history.failure): a degraded channel is NOT 'nothing "
            "similar exists'.")
    if retrieval_disabled:
        notes.append(
            "related-history discovery was DISABLED for this read "
            "(`related_top_k=0`), so its emptiness is by configuration, not "
            "by absence.")

    return {
        "state": state,
        "strategic_bank_empty": strategic_empty,
        "evidence_bank_empty": evidence_empty,
        "both_empty": both_empty,
        "strategic_entry_count": sbank_count,
        "evidence_count": bank_count,
        "recall_no_match": no_match,
        "retrieval_failed": retrieval_failed,
        "retrieval_disabled": retrieval_disabled,
        "note": ("an empty bank does NOT gate induction: the framework "
                 "reports the state and the agent decides whether to induce "
                 "from the available evidence"),
        "details": notes,
    }


def _related_history(harness, records: Sequence[Any], *,
                     related_top_k: int) -> Dict[str, Any]:
    """A SMALL, unfiltered retrieval around this batch's own method.

    Reuses the existing embedding index and ``recall_vectors`` machinery —
    no second index, no per-task rebuild. The execution channel is returned
    WITHOUT the online admission filter (an ``error`` record or a
    cross-cell/pending entry is exactly the material a boundary check
    needs), and the knowledge channel includes UNPUBLISHED entries so a
    revisable draft is still visible. A retrieval that could not run is
    reported as ``failure``; a retrieval that ran with no hits is ``[]`` —
    the two are never conflated, and neither is read as "no counterexample
    exists".
    """
    valid = [r for r in records if r is not None]
    if related_top_k <= 0 or not valid:
        return {
            "enabled": False,
            "top_k": max(0, related_top_k),
            "note": ("related-history discovery is OFF"
                     if related_top_k <= 0 else
                     "no records in this batch to build a query from"),
        }
    query = _query_text_for(valid)
    base = {
        "enabled": True,
        "top_k": int(related_top_k),
        "query_basis": query,
        "note": ("a similarity hit is a DISCOVERY signal, not support "
                 "strength; no hit is NOT evidence that no counterexample "
                 "exists. Execution hits are unfiltered (failures and "
                 "cross-cell cases included); knowledge hits include "
                 "unpublished entries. This batch's OWN executions are "
                 "excluded from the execution layer — otherwise the query "
                 "(built from their methods) would retrieve the batch "
                 "itself."),
    }
    if not query["text"]:
        base["executions"] = []
        base["knowledge"] = []
        base["no_hits"] = True
        base["failure"] = None
        return base
    # The batch's own executions are the query's source, so they must not be
    # returned as "related history" (an identical document scores highest
    # and would fill the whole top_k, hiding real cross-task cases).
    batch_ids = {str(r.execution_id) for r in valid}
    try:
        from or_harness.strategy.vector_recall import recall_vectors
        vectors = recall_vectors(
            harness, query["text"],
            top_k=int(related_top_k), include_unverified=True,
            task_profile=getattr(valid[-1], "profile_snapshot", None),
            exclude_execution_ids=batch_ids)
    except Exception as exc:  # noqa: BLE001 - a degraded channel is reported
        base["executions"] = []
        base["knowledge"] = []
        base["no_hits"] = None
        base["failure"] = {
            "reason": f"{type(exc).__name__}: {exc}",
            "note": ("related-history retrieval could NOT run; that is NOT "
                     "'nothing similar exists'. Read other batches (--cursor) "
                     "or rebuild the index (`orx rebuild-index`)"),
        }
        return base
    # The retrieval selects IDS (a discovery signal). The FULL content of
    # each selected execution is produced by the SAME ``_material_entry``
    # the batch uses, so a hit is not a lesser summary: the two channels
    # carry the same field set. The retrieval's own score/excerpt metadata
    # is kept beside the expansion as DISCOVERY metadata, never in place of
    # the record.
    retrieval_hits = list(vectors.get("execution_evidence") or [])
    expanded: List[Dict[str, Any]] = []
    for hit in retrieval_hits:
        execution_id = hit.get("execution_id")
        entry = (_material_entry_by_id(harness, str(execution_id))
                 if execution_id else None)
        if entry is None:
            # The fact vanished between indexing and now: report the stale
            # hit rather than dropping it silently.
            expanded.append({"execution_id": execution_id,
                             "stale": "the execution is no longer in the "
                                      "evidence bank"})
            continue
        # Discovery metadata (selection signal), never the record itself.
        entry["discovery"] = {
            "similarity": hit.get("similarity"),
            "structural_match": hit.get("structural_match"),
            "profile_cell": hit.get("profile_cell"),
            "note": ("similarity is a DISCOVERY signal, not support "
                     "strength; the record's OWN content above is the "
                     "material"),
        }
        expanded.append(entry)
    base["executions"] = expanded
    base["knowledge"] = list(vectors.get("strategic_knowledge") or [])
    base["degraded_layers"] = vectors.get("degraded_layers")
    base["no_hits"] = not (base["executions"] or base["knowledge"])
    base["failure"] = None
    return base


def _entry_budget_size(entry: Dict[str, Any],
                       charged_text_keys: set) -> int:
    """The BUDGET size of one entry: its JSON with the shared task text
    replaced by its version reference.

    The task text lives ONCE in the response-level ``task_texts`` map; a
    repeating entry must not be billed for it again, and — crucially — a
    large shared text must not evict other, unrelated records. The text is
    charged once, when its version is first seen (``charged_text_keys``
    tracks which versions have already been charged). Everything else is
    counted WHOLE, so an oversized method/failure/trajectory is bounded as a
    unit (the record is skipped and reported) rather than truncated.
    """
    text = entry.get("task_text")
    ref = entry.get("task_text_ref")
    if text is None or not isinstance(ref, dict) \
            or ref.get("task_text_digest") is None:
        return len(json.dumps(entry, ensure_ascii=False, default=str))
    # Replace the full text with its reference for the size measurement.
    probe = dict(entry)
    probe["task_text"] = ref
    size = len(json.dumps(probe, ensure_ascii=False, default=str))
    key = f"{ref.get('task_id')}|{ref.get('task_text_digest')}"
    if key not in charged_text_keys:
        # First occurrence: the version map pays for the text once.
        size += len(json.dumps({key: text}, ensure_ascii=False, default=str))
    return size


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
