"""M6 knowledge-prediction feedback: evaluation, attribution, reliability.

This module closes the loop the earlier milestones left open: a NORMAL
solving action predicts how the harness's strategic knowledge will change,
and that prediction is later evaluated against what actually happened —
first immediately (the execution's own evidence), then after the offline
consolidation that turns evidence into knowledge.

Three mechanisms, all framework-side and store-backed:

1. **Stage-partitioned evaluation.** A prediction's feedback is keyed by
   ``(change_index, stage)``. The immediate X/B comparison and the
   knowledge evaluation therefore never block each other: an already
   compared prediction can still receive its knowledge verdict later, and a
   ``pending`` item can still move to a final state. Re-evaluating the SAME
   stage is a no-op (idempotent); evaluating a DIFFERENT stage is allowed.

2. **Deduplicated attribution.** Several real actions can predict the same
   knowledge proposition. When it resolves, the credit goes to the earliest
   prediction inside the opportunity window; the others are recorded as
   ``co_attributed`` rather than each collecting the full benefit. Adding
   EVIDENCE is genuinely cumulative and stays per-action.

3. **Class reliability.** Resolved knowledge predictions are aggregated per
   ``(change, horizon)`` class. That measured reliability — not the model's
   self-reported uncertainty — is what a later prediction's value may draw
   on. Below a minimum sample count the class stays UNKNOWN, and an unknown
   class grants no value (never a guessed one).

Statuses (five, not two):

    pending       — the opportunity has not arrived, or a precondition is
                    still unmet. NEVER treated as a failure.
    fulfilled     — the opportunity arrived and the predicted change
                    happened as predicted.
    contradicted  — the opportunity arrived and the change went the
                    opposite way.
    missed        — the opportunity arrived and everything the prediction
                    depended on was satisfiable, but the predicted change
                    did not happen.
    inconclusive  — the opportunity is gone for good (e.g. the target entry
                    was retired); neither right nor wrong.

Honesty boundary: ``candidate_forms`` being ``fulfilled`` means an entry
FORMED. It does NOT mean publishable strategic knowledge exists — admission
verification and ``is_publishable`` still gate that, unchanged. Likewise a
cell's mean quality moving does not PROVE a rule is supported or refuted;
it is statistical movement, and the verdict text says so.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from or_harness.world_model.knowledge import MEASUREMENT_PREFIX

#: Knowledge-evaluation states (see module docstring).
KNOWLEDGE_STATUSES = ("pending", "fulfilled", "contradicted", "missed",
                      "inconclusive")

#: Terminal states: an evaluation that has actually decided something.
#: ``pending`` is deliberately NOT one of them — it records that the
#: opportunity has not arrived, so it must remain promotable.
TERMINAL_STATUSES = ("fulfilled", "contradicted", "missed", "inconclusive")

#: Evaluation stages.
STAGE_EXECUTION = "after_execution"
STAGE_CONSOLIDATION = "after_consolidation"
STAGES = (STAGE_EXECUTION, STAGE_CONSOLIDATION)

#: Minimum resolved samples before a class's reliability may be claimed.
MIN_CLASS_SAMPLES = 5

#: Key under which the stage-partitioned knowledge verdicts live inside a
#: prediction's ``feedback`` block. Chosen so a pre-M6 flat feedback block
#: (which has no such key) reads back as "no knowledge stage evaluated yet".
KNOWLEDGE_FEEDBACK_KEY = "knowledge"


# ---------------------------------------------------------------------------
# feedback partitioning (idempotent per stage, never per prediction)
# ---------------------------------------------------------------------------


def knowledge_feedback(prediction) -> Dict[str, Any]:
    """The stage-partitioned knowledge verdicts of one prediction.

    A legacy flat ``feedback`` block (pre-M6: the immediate X/B comparison
    only) has no knowledge partition, which correctly reads back as "no
    knowledge stage has been evaluated yet" — the old record is not
    retro-actively credited with a verdict it never had."""
    feedback = getattr(prediction, "feedback", None)
    if not isinstance(feedback, dict):
        return {}
    partition = feedback.get(KNOWLEDGE_FEEDBACK_KEY)
    return partition if isinstance(partition, dict) else {}


def stage_verdict(prediction, change_index: int,
                  stage: str) -> Optional[Dict[str, Any]]:
    """The verdict recorded for one (change, stage), if any."""
    return (knowledge_feedback(prediction)
            .get(str(change_index), {})
            .get(stage))


def record_stage_verdict(prediction, change_index: int, stage: str,
                         status: str, detail: Dict[str, Any]) -> bool:
    """Write one stage verdict; return True when it changed something.

    Idempotency is per ``(change_index, stage)`` with one deliberate
    exception: a ``pending`` verdict may be PROMOTED to a terminal one. A
    pending verdict says "the opportunity has not arrived", which is a
    statement about the current moment, not a settled judgement — refusing
    to promote it would freeze the item at pending forever and silently
    drop it from :func:`unresolved_stages`, so the prediction would never
    be evaluated at all.

    A TERMINAL verdict is final: a settled judgement is never rewritten by
    a later evaluation pass, so evidence cannot be retro-edited. Another
    stage of the same change stays independently writable throughout.
    """
    if status not in KNOWLEDGE_STATUSES:
        raise ValueError(f"status must be one of {KNOWLEDGE_STATUSES}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if getattr(prediction, "feedback", None) is None:
        prediction.feedback = {}
    if not isinstance(prediction.feedback, dict):
        raise ValueError("prediction.feedback must be a JSON object")
    partition = prediction.feedback.setdefault(KNOWLEDGE_FEEDBACK_KEY, {})
    slot = partition.setdefault(str(change_index), {})
    previous = slot.get(stage)
    if isinstance(previous, dict):
        previous_status = previous.get("status")
        if previous_status in TERMINAL_STATUSES:
            return False
        if status == "pending":
            # Re-confirming a pending verdict changes nothing observable.
            return False
    slot[stage] = {
        "status": status,
        "stage": stage,
        "evaluated_at": time.time(),
        **({"promoted_from": previous.get("status")}
           if isinstance(previous, dict) else {}),
        **copy.deepcopy(detail),
    }
    return True


def unresolved_stages(prediction) -> List[Tuple[int, str]]:
    """Every ``(change_index, stage)`` of this prediction still awaiting a
    TERMINAL verdict — the work list for the next evaluation pass.

    A ``pending`` verdict is still unresolved: the opportunity may arrive
    later, so the item must stay eligible rather than be dropped.
    """
    partition = knowledge_feedback(prediction)
    out: List[Tuple[int, str]] = []
    for item_index, item in enumerate(
            (getattr(prediction, "predicted", None) or {})
            .get("knowledge_changes") or []):
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        stage = item.get("horizon")
        if stage not in STAGES:
            continue
        recorded = (partition.get(str(item_index)) or {}).get(stage) or {}
        if recorded.get("status") not in TERMINAL_STATUSES:
            out.append((item_index, stage))
    return out


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def _target_value(target: Any, key: str, default: Any = None) -> Any:
    """Read one field off a target that may be a ``KnowledgeTarget`` or the
    plain dict it serializes to. The evaluation gets whichever the caller
    has at hand (the API keeps the live objects), so normalizing here keeps
    one code path instead of two parallel ones."""
    if target is None:
        return default
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _claim_interval(item: Dict[str, Any],
                    target: Any) -> Optional[Tuple[float, float]]:
    """The interval a support/refute verdict is judged against.

    Prefers the claim's OWN stated interval (reusing what the entry already
    promises) over any tolerance invented here."""
    raw = _target_value(target, "quality_interval")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    return None


def _observation_satisfied(observation: Any,
                           record: Any) -> Optional[bool]:
    """Whether the execution's REAL observations satisfy the expected
    observation, or None when the expectation cannot be checked.

    The check is deliberately literal: every ``key: value`` the prediction
    named must be present on the execution with the same value. A key the
    framework cannot read produces ``None`` (uncheckable) rather than a
    silent pass — believing an unchecked expectation would let a prediction
    be confirmed by an execution that says nothing about it.

    ``check_condition`` is free text a model wrote; the framework cannot
    evaluate prose, and this function does NOT pretend to. What it CAN do
    is refuse to call an unverified expectation fulfilled.
    """
    if not isinstance(observation, dict) or not observation:
        return None
    observed: Dict[str, Any] = {}
    quality = getattr(record, "quality", None)
    if isinstance(quality, dict):
        observed.update(quality)
    features = getattr(record, "execution_features", None)
    if isinstance(features, dict):
        observed.update({k: v for k, v in features.items()
                         if k not in observed})
    checkable = 0
    for key, expected in observation.items():
        if key not in observed:
            continue
        checkable += 1
        actual = observed[key]
        if isinstance(expected, bool) or isinstance(actual, bool):
            if bool(actual) != bool(expected):
                return False
        elif actual != expected:
            return False
    if checkable == 0:
        return None
    return True


def _preconditions_unmet(item: Dict[str, Any],
                         record: Any,
                         observation_ok: Optional[bool]
                         ) -> List[str]:
    """The standing conditions this execution demonstrably did NOT satisfy.

    A precondition names work that must happen for the predicted change to
    be observable. Only conditions the framework can actually adjudicate
    are reported here; the rest are treated as unmet for the purpose of
    granting a fulfilment (an unverifiable precondition must not silently
    count as met).

    - ``contrast_execution``: requires a genuine contrast — a companion
      execution of the same task under a different strategy. One execution
      alone cannot be a contrast.
    - ``independent_replication``: requires evidence from a task other than
      the one this prediction was made under.
    - ``verification_check``: requires an admission verdict, which by
      construction does not exist yet at execution time.
    - ``additional_measurement:<dim>``: requires that dimension to be
      measured on THIS execution.
    """
    unmet: List[str] = []
    for precondition in item.get("preconditions") or []:
        precondition = str(precondition)
        if precondition == "contrast_execution":
            if not _is_contrast_evidence(record):
                unmet.append(precondition)
        elif precondition == "independent_replication":
            if str(getattr(record, "task_id", "")) \
                    == str(item.get("_predicted_for_task") or ""):
                unmet.append(precondition)
        elif precondition == "verification_check":
            unmet.append(precondition)
        elif precondition.startswith(MEASUREMENT_PREFIX):
            dim = precondition[len(MEASUREMENT_PREFIX):]
            cost = getattr(record, "cost", None)
            measured = (cost.measured_dims() if cost is not None
                        and hasattr(cost, "measured_dims") else set())
            if dim not in measured:
                unmet.append(precondition)
    return unmet


def _is_contrast_evidence(record: Any) -> bool:
    """Whether the record's own labels mark it as contrast evidence.

    The framework records contrast intent explicitly (``retention_reason``,
    or a ``contrast`` marker in the execution features). Absent that, the
    execution is ordinary evidence and cannot satisfy a contrast condition.
    """
    reason = str(getattr(record, "retention_reason", "") or "").lower()
    if "contrast" in reason:
        return True
    features = getattr(record, "execution_features", None) or {}
    return bool(features.get("contrast"))


def _observed_quality(record) -> Tuple[Optional[float], Optional[str]]:
    """The quality actually observed, with the basis that justifies it.

    Returns ``(None, reason)`` when no MEASURED quality exists. The strategy
    layer's ``quality_score`` is NOT used as an observation here: it falls
    back to a 0.5 heuristic for a feasible result with no gap or optimality
    basis, and treating that placeholder as measured quality would let
    "a record was produced" masquerade as "the expected knowledge was
    supported". The knowledge channel judges claims, so it may only use
    measurements.
    """
    quality = getattr(record, "quality", None)
    if not isinstance(quality, dict) or not quality.get("feasible"):
        return None, "the execution produced no feasible result"
    gap = quality.get("gap")
    if gap is not None:
        return max(0.0, min(1.0, 1.0 - float(gap))), "measured gap"
    if quality.get("status") == "optimal":
        return 1.0, "optimal status"
    return None, ("the result is feasible but carries no gap or optimality "
                  "basis: its quality is not measured, so no claim can be "
                  "judged against it")


def evaluate_execution(item: Dict[str, Any],
                       target: Any,
                       record) -> Tuple[str, Dict[str, Any]]:
    """Verdict for an ``after_execution`` knowledge change.

    The predicted PROPOSITION is what gets judged:

    1. a standing precondition that this execution did not satisfy leaves
       the item ``pending`` — the opportunity has not arrived, which is not
       a failure;
    2. a predicted ``expected_observation`` that the real result contradicts
       yields ``contradicted``;
    3. ``adds_evidence`` asks exactly what it says — did the execution land
       in the predicted cell (and meet the conditions);
    4. support/refute/revise/narrow verdicts need a MEASURED quality and
       the claim's own interval; without a measurement the item stays
       ``pending`` rather than being judged against a placeholder.
    """
    change = item.get("change")
    detail: Dict[str, Any] = {"change": change,
                              "execution_id": getattr(record, "execution_id",
                                                      None)}
    if target is None:
        return "inconclusive", {**detail,
                                "reason": "target no longer resolvable"}
    # (1) Standing conditions come FIRST: a change whose preconditions were
    #     never met has not had its opportunity, whatever else is true.
    observation_ok = _observation_satisfied(
        item.get("expected_observation"), record)
    if observation_ok is False:
        detail["expected_observation"] = item.get("expected_observation")
        detail["reason"] = ("the observed result contradicts the expected "
                            "observation the prediction named")
        return "contradicted", detail
    unmet = _preconditions_unmet(item, record, observation_ok)
    if unmet:
        detail["unmet_preconditions"] = unmet
        detail["reason"] = ("the execution did not satisfy the standing "
                            "condition(s) " + ", ".join(unmet)
                            + ": the opportunity has not arrived")
        return "pending", detail
    # (2) The evidence must belong where the prediction said it would.
    predicted_cell = _target_value(target, "cell_token")
    actual_cell = _record_cell(record)
    if predicted_cell and actual_cell and predicted_cell != actual_cell:
        detail["predicted_cell"] = predicted_cell
        detail["actual_cell"] = actual_cell
        return "missed", {**detail,
                          "reason": "the execution landed in a different "
                                    "structural cell than predicted"}
    # (3) A hypothesis carrying an expectation we could not check is NOT
    #     thereby fulfilled: an unchecked expectation is unknown.
    if item.get("expected_observation") and observation_ok is None:
        detail["reason"] = ("the expected observation could not be checked "
                            "against the execution's own fields; the "
                            "expectation stays unevaluated")
        return "pending", detail
    if change == "adds_evidence":
        detail["reason"] = ("the execution is a new observation in the "
                            "predicted cell")
        return "fulfilled", detail
    quality, basis = _observed_quality(record)
    if quality is None:
        return "pending", {**detail,
                           "reason": f"no measured quality to judge the "
                                     f"claim against: {basis}"}
    interval = _claim_interval(item, target)
    if interval is None:
        return "pending", {**detail,
                           "reason": "the claim states no interval: the "
                                     "movement cannot be judged"}
    lo, hi = interval
    detail["observed_quality"] = round(float(quality), 6)
    detail["quality_basis"] = basis
    detail["claim_interval"] = [lo, hi]
    inside = lo <= float(quality) <= hi
    if change == "supports":
        if inside:
            detail["reason"] = ("measured quality is inside the claim's own "
                                "interval; this is statistical movement in "
                                "the predicted direction, not proof")
            return "fulfilled", detail
        detail["reason"] = ("measured quality fell outside the claim's own "
                            "interval, the opposite of a supporting "
                            "observation")
        return "contradicted", detail
    if change in ("refutes", "revises", "narrows"):
        if not inside:
            detail["reason"] = ("measured quality fell outside the claim's "
                                "own interval, in the predicted direction")
            return "fulfilled", detail
        detail["reason"] = ("measured quality stayed inside the claim's own "
                            "interval; the claim was not moved")
        return "missed", detail
    return "inconclusive", {**detail,
                            "reason": f"change {change!r} is not judged on "
                                      "execution evidence"}


def _record_cell(record) -> Optional[str]:
    """The structural cell an execution actually landed in."""
    profile = getattr(record, "profile_snapshot", None)
    if profile is None:
        return None
    try:
        from or_harness.core.schema import group_key
        return group_key(profile)
    except Exception:
        return None


def evaluate_consolidation(item: Dict[str, Any],
                           target: Any,
                           entry_changes: Sequence[Dict[str, Any]],
                           created: Sequence[str],
                           created_strategies: Sequence[str],
                           ) -> Tuple[str, Dict[str, Any]]:
    """Verdict for an ``after_consolidation`` knowledge change.

    Judged against the induction's OWN recorded transition (``entry_changes``
    / ``entries_created``), so the verdict speaks about the specific
    proposition predicted, not about the induction having run.

    Honesty: ``candidate_forms`` fulfilled means an entry FORMED, nothing
    more — admission verification still decides whether it is publishable.
    """
    change = item.get("change")
    detail: Dict[str, Any] = {"change": change}
    if target is None:
        return "inconclusive", {**detail,
                                "reason": "target no longer resolvable"}
    entry_id = _target_value(target, "entry_id")
    strategy_id = _target_value(target, "strategy_id")
    if change == "candidate_forms":
        if strategy_id in created_strategies:
            detail["reason"] = ("an entry formed for this strategy; whether "
                                "it is publishable is decided separately by "
                                "admission verification")
            return "fulfilled", detail
        return "missed", {**detail,
                          "reason": "the induction ran on this scope but no "
                                    "entry formed for the predicted strategy"}
    if entry_id is None:
        return "inconclusive", {**detail,
                                "reason": "no entry named by the target"}
    changed = next((c for c in entry_changes
                    if c.get("entry_id") == entry_id), None)
    if changed is None:
        if entry_id in set(created):
            detail["reason"] = "the entry was (re)created"
            return "fulfilled", detail
        return "missed", {**detail,
                          "reason": "the induction ran but this entry's "
                                    "claim did not change"}
    fields = (changed.get("changed") or {})
    detail["changed_fields"] = sorted(fields)
    if change == "revises":
        if "expected_quality_hat" in fields or "predicates" in fields:
            detail["reason"] = "the claim's estimate/predicates were revised"
            return "fulfilled", detail
        return "missed", {**detail,
                          "reason": "the entry changed, but not the "
                                    "estimate the prediction named"}
    if change == "narrows":
        interval = fields.get("quality_interval")
        if not isinstance(interval, dict):
            return "missed", {**detail,
                              "reason": "the claim's interval did not move"}
        before = (interval.get("before") or [0.0, 1.0])
        after = (interval.get("after") or [0.0, 1.0])
        narrowed = (float(after[1]) - float(after[0])) \
            < (float(before[1]) - float(before[0])) - 1e-9
        detail["width_before"] = round(float(before[1]) - float(before[0]), 6)
        detail["width_after"] = round(float(after[1]) - float(after[0]), 6)
        if narrowed:
            detail["reason"] = "the claim's interval narrowed"
            return "fulfilled", detail
        return "missed", {**detail,
                          "reason": "the claim's interval did not narrow"}
    if change in ("refutes", "supports"):
        status = fields.get("status") or {}
        after_status = status.get("after") if isinstance(status, dict) else None
        if change == "refutes" and after_status in ("suspect", "refuted",
                                                    "dormant"):
            detail["reason"] = f"the entry moved to {after_status!r}"
            return "fulfilled", detail
        if change == "supports" and after_status in ("validated", "verified"):
            detail["reason"] = f"the entry moved to {after_status!r}"
            return "fulfilled", detail
        return "missed", {**detail,
                          "reason": "the entry's standing did not move as "
                                    "predicted"}
    if change == "adds_evidence":
        detail["reason"] = ("evidence accumulation is judged at the "
                            "execution stage, not at consolidation")
        return "inconclusive", detail
    return "inconclusive", {**detail, "reason": "not judged at this stage"}


# ---------------------------------------------------------------------------
# deduplicated attribution
# ---------------------------------------------------------------------------


def proposition_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """The identity of one knowledge PROPOSITION: target x change x horizon.

    Two actions predicting the same proposition are predicting the same
    thing about the world, so the outcome may only be credited once. The
    two horizons are distinct propositions (the same claim at two different
    times), so they never collide."""
    target = item.get("target") or {}
    identity = str(target.get("entry_id")
                   or target.get("strategy_id")
                   or target.get("cell_token") or "")
    return (identity, str(item.get("change") or ""),
            str(item.get("horizon") or ""))


def dedupe_attribution(entries: Sequence[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    """Mark repeated propositions so credit is not counted twice.

    ``entries`` are ``{"prediction_id", "action_id", "created_at", "item"}``
    records (all candidates whose prediction covered the proposition). The
    EARLIEST prediction in the window is the attributed one; the rest are
    marked ``co_attributed`` and keep their verdict (so the evaluation is
    still auditable) while being excluded from the benefit count.

    ``adds_evidence`` is exempt on purpose: evidence genuinely accumulates,
    one observation per action, and pooling them is not double counting.
    """
    out: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for entry in entries:
        item = entry.get("item") or {}
        if item.get("change") == "adds_evidence":
            out.append({**entry, "attribution": "per_action"})
            continue
        groups.setdefault(proposition_key(item), []).append(entry)
    for key, group in groups.items():
        ordered = sorted(group, key=lambda e: (e.get("created_at") or 0.0,
                                               str(e.get("prediction_id"))))
        for index, entry in enumerate(ordered):
            out.append({**entry,
                        "attribution": ("attributed" if index == 0
                                        else "co_attributed"),
                        "proposition": list(key)})
    return out


# ---------------------------------------------------------------------------
# class reliability (the feedback that changes future decisions)
# ---------------------------------------------------------------------------


def _dedupe_resolved(records: Sequence[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """Collapse repeated PROPOSITIONS so one outcome is credited once.

    Two actions predicting the same ``(target, change, horizon)`` are
    predicting the same thing about the world, so when it resolves the event
    must not be counted once per action — that would inflate the class's
    measured reliability, which is exactly the self-correction signal the
    decision layer trusts.

    The EARLIEST prediction in each proposition group is kept (the others
    are the ones that merely followed it), and its verdict is what counts.
    ``adds_evidence`` is exempt: observations genuinely accumulate per
    action, so pooling them is not double counting. Records carrying no
    proposition identity are kept as-is rather than silently merged.
    """
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for record in records:
        proposition = record.get("proposition")
        if not proposition or record.get("change") == "adds_evidence":
            passthrough.append(record)
            continue
        key = tuple(str(p) for p in proposition)
        groups.setdefault(key, []).append(record)
    kept = list(passthrough)
    for group in groups.values():
        ordered = sorted(group,
                         key=lambda r: (r.get("created_at") or 0.0,
                                        str(r.get("prediction_id") or "")))
        kept.append(ordered[0])
    return kept


def class_reliability(records: Iterable[Dict[str, Any]],
                      *, min_samples: int = MIN_CLASS_SAMPLES,
                      dedupe: bool = True
                      ) -> Dict[str, Any]:
    """Measured reliability per ``(change, horizon)`` class.

    ``records`` are resolved knowledge verdicts: ``{"change", "horizon",
    "status"}. ``fulfilled`` counts as a hit, ``contradicted`` and
    ``missed`` as misses, ``inconclusive`` is excluded (it carries no
    information either way), and ``pending`` never reaches here.

    ``dedupe`` (default on) collapses repeated proposions so that ONE
    knowledge outcome is credited once: several actions predicting the same
    proposition would otherwise each score a hit for a single event, and
    the class would look more reliable than it is. Evidence accumulation
    (``adds_evidence``) is exempt — that genuinely accumulates per action.

    A class below ``min_samples`` resolves to ``None`` — unknown stays
    unknown, because a reliability guessed from two samples would be worse
    than admitting we do not know it.
    """
    resolved = [r for r in records
                if r.get("status") not in ("pending", "inconclusive", None)]
    if dedupe:
        resolved = _dedupe_resolved(resolved)
    buckets: Dict[Tuple[str, str], Dict[str, int]] = {}
    for record in resolved:
        status = record.get("status")
        key = (str(record.get("change") or ""),
               str(record.get("horizon") or ""))
        bucket = buckets.setdefault(key, {"hits": 0, "misses": 0})
        if status == "fulfilled":
            bucket["hits"] += 1
        elif status in ("contradicted", "missed"):
            bucket["misses"] += 1
    out: Dict[str, Any] = {}
    for (change, horizon), bucket in buckets.items():
        total = bucket["hits"] + bucket["misses"]
        key = f"{change}|{horizon}"
        if total < min_samples:
            out[key] = {
                "n": total,
                "reliability": None,
                "basis": "insufficient_history",
                "note": (f"{total} resolved sample(s) < {min_samples}: no "
                         "reliability is claimed"),
            }
        else:
            out[key] = {
                "n": total,
                "hits": bucket["hits"],
                "misses": bucket["misses"],
                "reliability": round(bucket["hits"] / total, 6),
                "basis": "measured",
                "note": ("measured fulfilment rate of this class of "
                         "prediction; a contract-mean, not a model's "
                         "self-report"),
            }
    return out


def reliability_lookup(table: Dict[str, Any]):
    """A ``(change, horizon) -> Optional[float]`` callable over a table.

    Returns an instruction the value assembly can consult. ``None`` means
    the class is unknown or too thin, and the caller must then grant NO
    value — the model's own uncertainty is not a substitute for measured
    reliability."""
    def _lookup(change: Any, horizon: Any) -> Optional[float]:
        entry = table.get(f"{change}|{horizon}")
        if not isinstance(entry, dict):
            return None
        value = entry.get("reliability")
        return float(value) if value is not None else None
    return _lookup


def collect_verdicts(predictions: Iterable[Any]) -> List[Dict[str, Any]]:
    """Flatten every resolved knowledge verdict, for reliability accounting.

    Reads the predictions' stored partitions only — no model call, no
    recomputation, and no second source of truth about what happened."""
    out: List[Dict[str, Any]] = []
    for prediction in predictions:
        items = ((getattr(prediction, "predicted", None) or {})
                 .get("knowledge_changes") or [])
        partition = knowledge_feedback(prediction)
        for slot_index, slot in partition.items():
            try:
                item = items[int(slot_index)]
            except (ValueError, IndexError, TypeError):
                continue
            for stage in STAGES:
                verdict = slot.get(stage)
                if not isinstance(verdict, dict):
                    continue
                if verdict.get("status") == "pending":
                    continue
                out.append({
                    "prediction_id": getattr(prediction, "prediction_id",
                                             None),
                    "action_id": getattr(prediction, "bound_action_id", None),
                    "created_at": getattr(prediction, "created_at", None),
                    "change": item.get("change"),
                    "horizon": stage,
                    "status": verdict.get("status"),
                    # The proposition identity drives the one-credit-per-
                    # outcome rule in class_reliability.
                    "proposition": proposition_key(item),
                })
    return out
