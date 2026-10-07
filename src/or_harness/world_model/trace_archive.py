"""The bridge from an ONLINE capability-gain prediction to its effect check.

A strategy-outcome prediction may claim a POTENTIAL capability gain (H+)
in the same call that predicts benefit/cost/risk. That claim is
EXPLANATORY: it never ranks candidates and never writes capability
evidence. But it is also a real, falsifiable statement about what the
candidate might teach, and until this module existed it died the moment it
was printed: nothing could bind the later real execution to it, nothing
could ask whether the gain showed up, and the offline evaluator rejected
its ``sp_`` id as "unknown capability prediction".

This module is the MINIMUM that connects the two:

1. :func:`archive_capability_gain` writes a TRACE row (the ``meta`` table)
   when a strategy prediction carrying a claimed gain is saved. The trace
   is metadata only — it never copies the prediction, never re-predicts,
   and never enters the utility.
2. :func:`effect_prediction_of` reads a trace and MATERIALIZES a
   ``CapabilityEvolutionPrediction`` on demand, from the trace's frozen
   claim plus the REAL facts (the bound action's execution and the
   knowledge the adopted operation actually produced). The materialized
   object is never stored: it is a view the existing stage-1 / stage-2
   machinery consumes unchanged.
3. The materialized prediction deliberately spends NOTHING:
   ``service_available`` is False, so it can never be mistaken for a paid
   offline prediction, and the trace records what it was derived from.

The trace is not a second bank: it holds no knowledge and no statistics —
one row per claimed online gain, keyed by the prediction's own id (which
causes no clash with the ``hp_`` table, since ``sp_`` and ``hp_`` ids are
disjoint).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Version of the trace row schema.
CAPABILITY_TRACE_VERSION = "wm-ctrace/1"

#: Where the trace rows live inside the shared key/value table.
TRACE_PREFIX = "capability_trace"


@dataclass
class CapabilityTrace:
    """One online capability-gain claim, recorded so it can be followed up.

    It carries the ORIGINAL claim (never a re-prediction), the frozen
    target the claim was made under, and the identifiers a later reader
    needs to follow it. It also carries live state that the readers update:

    * ``state`` — ``pending`` until a real execution lands, then
      ``bound`` / ``verified`` / whichever stage-2 verdict the existing
      evaluator reports;
    * ``effect_state`` / ``effect_verified`` — the stage-2 verdict.

    The claim fields are NEVER rewritten: a later reader sees what was
    predicted before the result, not a post-hoc restatement.
    """

    trace_id: str = ""
    prediction_id: str = ""
    task_id: str = ""
    episode_id: Optional[str] = None
    strategy_id: Optional[str] = None
    family: Optional[str] = None
    cell_token: Optional[str] = None
    task_targeting: Dict[str, Any] = field(default_factory=dict)
    horizon: str = ""
    horizon_tasks: Optional[int] = None
    expected_changes: List[Dict[str, Any]] = field(default_factory=list)
    verification_conditions: List[Dict[str, Any]] = field(
        default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    basis: List[str] = field(default_factory=list)
    claim: str = ""
    applies_to: List[str] = field(default_factory=list)
    #: The model's stated H+ stance ("expected"/"none"/
    #: "insufficient_basis"; empty = not stated, e.g. an old payload).
    assessment: str = ""
    created_at: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)
    #: The strategy prediction's own status at archive time.
    prediction_status: str = "valid"
    #: Live lifecycle: ``pending`` / ``bound`` / ``verified`` / ...
    state: str = "pending"
    bound_action_id: Optional[str] = None
    bound_execution_ids: List[str] = field(default_factory=list)
    effect_state: Optional[str] = None
    effect_verified: bool = False
    updated_at: Optional[float] = None
    trace_version: str = CAPABILITY_TRACE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_version": self.trace_version,
            "trace_id": self.trace_id,
            "prediction_id": self.prediction_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "strategy_id": self.strategy_id,
            "family": self.family,
            "cell_token": self.cell_token,
            "task_targeting": copy.deepcopy(self.task_targeting),
            "horizon": self.horizon,
            "horizon_tasks": self.horizon_tasks,
            "expected_changes": copy.deepcopy(self.expected_changes),
            "verification_conditions": copy.deepcopy(
                self.verification_conditions),
            "uncertainty": list(self.uncertainty),
            "basis": list(self.basis),
            "claim": self.claim,
            "applies_to": list(self.applies_to),
            "assessment": self.assessment,
            "created_at": self.created_at,
            "notes": list(self.notes),
            "prediction_status": self.prediction_status,
            "state": self.state,
            "bound_action_id": self.bound_action_id,
            "bound_execution_ids": list(self.bound_execution_ids),
            "effect_state": self.effect_state,
            "effect_verified": bool(self.effect_verified),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityTrace":
        data = data or {}
        return cls(
            trace_id=str(data.get("trace_id", "")),
            prediction_id=str(data.get("prediction_id", "")),
            task_id=str(data.get("task_id", "")),
            episode_id=(str(data["episode_id"])
                        if data.get("episode_id") else None),
            strategy_id=(str(data["strategy_id"])
                         if data.get("strategy_id") else None),
            family=(str(data["family"]) if data.get("family") else None),
            cell_token=(str(data["cell_token"])
                        if data.get("cell_token") else None),
            task_targeting=copy.deepcopy(dict(data.get("task_targeting")
                                              or {})),
            horizon=str(data.get("horizon", "")),
            horizon_tasks=(int(data["horizon_tasks"])
                           if data.get("horizon_tasks") is not None
                           else None),
            expected_changes=copy.deepcopy(
                list(data.get("expected_changes") or [])),
            verification_conditions=copy.deepcopy(
                list(data.get("verification_conditions") or [])),
            uncertainty=[str(u) for u in (data.get("uncertainty") or [])],
            basis=[str(b) for b in (data.get("basis") or [])],
            claim=str(data.get("claim", "")),
            applies_to=[str(a) for a in (data.get("applies_to") or [])],
            assessment=str(data.get("assessment", "")),
            created_at=float(data.get("created_at", time.time())),
            notes=[str(n) for n in (data.get("notes") or [])],
            prediction_status=str(data.get("prediction_status", "valid")),
            state=str(data.get("state", "pending")),
            bound_action_id=data.get("bound_action_id"),
            bound_execution_ids=[str(e) for e in
                                 (data.get("bound_execution_ids") or [])],
            effect_state=data.get("effect_state"),
            effect_verified=bool(data.get("effect_verified", False)),
            updated_at=(float(data["updated_at"])
                        if data.get("updated_at") is not None else None),
            trace_version=str(data.get("trace_version",
                                       CAPABILITY_TRACE_VERSION)),
        )


def _trace_key(prediction_id: str) -> str:
    return f"{TRACE_PREFIX}|{prediction_id}"


def knowledge_result_of(*, outcome: Optional[Dict[str, Any]] = None,
                        params: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """The knowledge-change block of an operation action, in EITHER shape.

    Two writers produce the SAME meaning under different nesting:

    * a DIRECT induction (``api.induce`` / ``_end_induce_action``) writes
      ``outcome.knowledge_delta`` and ``outcome.execution_ids`` at the TOP
      level of the action's outcome;
    * a WRAPPED operation (``bind_capability_maintenance``) writes them
      inside ``outcome.operation_result``.

    Reading only the wrapped shape made a direct induction that really
    created one entry read as ZERO — the r10 defect. This locator accepts
    both, preferring the wrapped block when present (a wrapped operation's
    own report), then the top-level delta. It never fabricates: when neither
    carries a delta the result is empty.
    """
    outcome = outcome or {}
    params = params or {}
    wrapped = (outcome.get("operation_result")
               or params.get("operation_result") or {})
    if isinstance(wrapped, dict) and wrapped.get("knowledge_delta"):
        return dict(wrapped)
    # The direct shape (or the wrapped shape with no delta of its own).
    result: Dict[str, Any] = {}
    delta = outcome.get("knowledge_delta") or params.get("knowledge_delta")
    if isinstance(delta, dict):
        result["knowledge_delta"] = delta
    for key in ("business_result", "execution_ids", "verification",
                "knowledge_after"):
        value = outcome.get(key)
        if value is None:
            value = params.get(key)
        if value is not None:
            result[key] = value
    if not result and isinstance(wrapped, dict):
        return dict(wrapped)
    return result


def _put_trace(harness, trace: CapabilityTrace) -> None:
    with harness.store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (_trace_key(trace.prediction_id),
             harness.store.dumps(trace.to_dict())))


def archive_capability_gain(harness, prediction) -> Optional[CapabilityTrace]:
    """Record an online capability-gain stance so it can be followed up.

    Called when a strategy-outcome prediction carrying an H+ block is
    saved. Returns the stored trace, or ``None`` when the block neither
    states a stance nor claims a gain (an absent block leaves no trace —
    the prediction's ``unsupported_fields`` already records that the model
    did not state one).

    A STATED stance is archived even when it is ``assessment="none"``: "no
    gain expected" is a real, falsifiable position, and dropping it would
    make the model's silence and its explicit "none" indistinguishable —
    exactly the confusion the stance requirement exists to remove.

    The trace COPIES the claim verbatim. It never re-predicts, never
    scores, and never touches the capability evidence.
    """
    gain = getattr(prediction, "capability_gain", None)
    if gain is None or not (getattr(gain, "assessed", False)
                            or gain.claimed):
        return None
    candidate = prediction.candidate
    trace = CapabilityTrace(
        trace_id=f"ct_{hash(prediction.prediction_id) & 0xffffffff:08x}",
        prediction_id=prediction.prediction_id,
        task_id=candidate.task_id,
        episode_id=candidate.episode_id,
        strategy_id=candidate.strategy_id,
        task_targeting={
            "task_ids": [candidate.task_id] if candidate.task_id else [],
            "description": ("the online gain stance was stated for this "
                            "candidate's task and execution"),
        },
        expected_changes=[c.to_dict() for c in gain.expected_changes],
        verification_conditions=[v.to_dict()
                                 for v in gain.verification_conditions],
        uncertainty=list(gain.uncertainty),
        basis=list(gain.basis),
        claim=gain.claim,
        applies_to=list(gain.applies_to),
        assessment=str(getattr(gain, "assessment", "") or ""),
        prediction_status=str(getattr(prediction, "status", "valid")),
        state="pending",
    )
    trace.notes.append(
        "an ONLINE capability-gain stance, recorded so the real execution "
        "and its later effect can be followed up. It is explanatory only: "
        "it never ranked the candidate, it is not the prediction's utility, "
        "and it is not capability evidence")
    if getattr(gain, "assessment", "") == "none":
        trace.notes.append(
            "the model stated assessment='none': it expects NO new gain "
            "from this candidate. The stance is kept so an explicit 'no "
            "gain' is never silently conflated with an unstated one")
    elif getattr(gain, "assessment", "") == "insufficient_basis":
        trace.notes.append(
            "the model stated assessment='insufficient_basis': it could "
            "not judge the gain from the evidence given")
    if not gain.expected_changes:
        trace.notes.append(
            "the claim named no expected change: there is nothing "
            "quantifiable to verify, so the trace can only report that the "
            "gain was claimed and never confirmed")
    # A prediction is saved MORE THAN ONCE (the pre-execution claim, then
    # the post-execution result). The CLAIM fields above are re-derived
    # fresh, but the LIVE state is not: re-archiving must never reset a
    # trace that has already been bound or evaluated back to ``pending``.
    existing = get_capability_trace(harness, prediction.prediction_id)
    if existing is not None:
        trace.state = existing.state
        trace.bound_action_id = existing.bound_action_id
        trace.bound_execution_ids = list(existing.bound_execution_ids)
        trace.effect_state = existing.effect_state
        trace.effect_verified = existing.effect_verified
        trace.updated_at = existing.updated_at
    _put_trace(harness, trace)
    return trace


def get_capability_trace(harness, prediction_id: str
                         ) -> Optional[CapabilityTrace]:
    """The stored trace of one prediction's online gain, or None."""
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?",
        (_trace_key(prediction_id),)).fetchone()
    if row is None:
        return None
    return CapabilityTrace.from_dict(harness.store.loads(row["value"]))


def iter_capability_traces(harness) -> List[CapabilityTrace]:
    """Every stored online-gain trace, oldest first."""
    rows = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key LIKE ? ORDER BY key ASC",
        (f"{TRACE_PREFIX}|%",)).fetchall()
    traces: List[CapabilityTrace] = []
    for row in rows:
        try:
            traces.append(CapabilityTrace.from_dict(
                harness.store.loads(row["value"])))
        except Exception:  # noqa: BLE001 - one bad row never breaks the list
            continue
    return traces


def update_trace_state(harness, prediction_id: str, *,
                       state: Optional[str] = None,
                       bound_action_id: Optional[str] = None,
                       bound_execution_ids: Optional[List[str]] = None,
                       effect_state: Optional[str] = None,
                       effect_verified: Optional[bool] = None
                       ) -> Optional[CapabilityTrace]:
    """Advance one trace's live state (bookkeeping only, no model call).

    Only the fields the caller passes are changed. The CLAIM is never
    touched: the original target, changes and conditions stay frozen, so a
    later reader sees what was predicted, not a post-hoc restatement.
    """
    trace = get_capability_trace(harness, prediction_id)
    if trace is None:
        return None
    if state is not None:
        trace.state = str(state)
    if bound_action_id is not None:
        trace.bound_action_id = str(bound_action_id)
    if bound_execution_ids is not None:
        trace.bound_execution_ids = [str(e) for e in bound_execution_ids]
    if effect_state is not None:
        trace.effect_state = str(effect_state)
    if effect_verified is not None:
        trace.effect_verified = bool(effect_verified)
    trace.updated_at = time.time()
    _put_trace(harness, trace)
    return trace


def _operation_entries(harness, adoption_action_id: Optional[str]
                       ) -> Dict[str, Any]:
    """The knowledge the adopted operation really produced.

    Read from the ADOPTION action's recorded outcome — the operation's own
    report of what it created/updated/retired — never inferred.
    """
    empty: Dict[str, Any] = {"created_entry_ids": [],
                             "updated_entry_ids": [],
                             "retired_entry_ids": [],
                             "actual_execution_ids": []}
    if not adoption_action_id:
        return empty
    action = harness.actions.get(adoption_action_id)
    if action is None:
        return empty
    params = action.params or {}
    outcome = action.outcome or {}
    result = knowledge_result_of(outcome=outcome, params=params)
    delta = result.get("knowledge_delta") or {}

    def _ids(records: Any) -> List[str]:
        out: List[str] = []
        for item in records or []:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                entry_id = item.get("entry_id") or item.get("id")
                if entry_id:
                    out.append(str(entry_id))
                elif isinstance(item.get("after"), dict) \
                        and item["after"].get("entry_id"):
                    out.append(str(item["after"]["entry_id"]))
        return out

    return {
        "created_entry_ids": _ids(delta.get("entries_created")),
        "updated_entry_ids": _ids(delta.get("entry_changes")),
        "retired_entry_ids": _ids(delta.get("entries_retired")),
        "actual_execution_ids": [str(e) for e in
                                 (result.get("execution_ids") or [])],
        "adoption_action_id": adoption_action_id,
    }


def operation_action_for(harness, prediction_id: str) -> Optional[Any]:
    """The operation action whose knowledge the ONLINE gain claim produced.

    Correlates by the REAL execution ids: an online H+ claim is made about
    an attempt, and the induction that follows cites that attempt's
    execution. The operation whose outcome names the claim's
    ``bound_execution_ids`` IS the operation; when none matches, the bound
    action itself is returned (the attempt really happened, even if it fed
    no induction) — never a guess.
    """
    trace = get_capability_trace(harness, prediction_id)
    bound = bound_action_of(harness, prediction_id)
    if bound is None:
        return None
    claim_execs = {str(e) for e in (trace.bound_execution_ids or [])} \
        if trace is not None else set()
    if bound.linked_execution_id:
        claim_execs.add(str(bound.linked_execution_id))
    if claim_execs:
        best = None
        for action in harness.actions.query():
            if action.action_type != "induce":
                continue
            if action.action_id == bound.action_id:
                continue
            outcome = action.outcome or {}
            params = action.params or {}
            cited = {str(e) for e in
                     (knowledge_result_of(outcome=outcome, params=params)
                      .get("execution_ids") or [])}
            if claim_execs & cited:
                # Prefer the latest matching operation (the one that
                # actually consolidated this claim's evidence).
                if best is None or (action.ended_at or 0) >= (best.ended_at
                                                              or 0):
                    best = action
        if best is not None:
            return best
    return bound


def bound_action_of(harness, prediction_id: str):
    """The real action the traced online prediction was bound to, or None.

    The forward link lives on the prediction's own ``model_info`` (written
    by ``bind_strategy_outcome`` / ``execute``), so it survives the
    prediction being saved and re-read.
    """
    prediction = harness.strategy_predictions.get(prediction_id)
    if prediction is None:
        return None
    action_id = (prediction.trace.model_info or {}).get("bound_action_id")
    if not action_id:
        return None
    return harness.actions.get(action_id)


def refresh_trace(harness, prediction_id: str) -> Optional[CapabilityTrace]:
    """Re-derive a trace's live state from the real facts.

    Idempotent, and read-only apart from the trace row it updates. Called
    after a bind/evaluation so the trace reflects the real execution
    without re-running anything.
    """
    from or_harness.world_model.maintenance_decision import (
        get_effect_evaluation,
    )
    trace = get_capability_trace(harness, prediction_id)
    if trace is None:
        return None
    action = bound_action_of(harness, prediction_id)
    bound_action_id = action.action_id if action is not None else None
    execution_ids = ([str(action.linked_execution_id)]
                     if action is not None and action.linked_execution_id
                     else [])
    evaluation = get_effect_evaluation(harness, prediction_id)
    if evaluation is not None:
        state = ("verified" if evaluation.effect_verified
                 else str(evaluation.state))
    elif bound_action_id is not None:
        state = "bound"
    else:
        state = "pending"
    return update_trace_state(
        harness, prediction_id, state=state,
        bound_action_id=bound_action_id,
        bound_execution_ids=execution_ids,
        effect_state=(evaluation.state if evaluation is not None else None),
        effect_verified=(evaluation.effect_verified
                         if evaluation is not None else None))


def effect_prediction_of(harness, prediction_id: str, *,
                         adoption_action_id: Optional[str] = None):
    """Materialize the offline prediction stage 2 needs, from a trace.

    Returns ``None`` when there is no online-gain trace for the id — the
    caller then treats the id as an unknown capability prediction exactly
    as before.

    The materialized ``CapabilityEvolutionPrediction`` is built from:

    * the TRACE's frozen claim (its expected changes, conditions and
      target), verbatim;
    * the REAL facts the harness recorded — the bound action's execution
      and what the adopted operation actually produced.

    It is NOT stored and NOT billed (``service_available`` stays False): it
    exists only so the existing stage-1/-2 machinery can judge the online
    claim against real evidence. The original prediction is never
    rewritten, and no new prediction is made after seeing the result.
    """
    from or_harness.world_model.contracts import (
        CapabilityEvolutionPrediction,
        ExperienceScope,
        ExpectedChange,
        HarnessCapabilityEvidence,
        LearningOperation,
        TaskTargeting,
        VerificationCondition,
    )

    trace = get_capability_trace(harness, prediction_id)
    if trace is None:
        return None
    # The operation that produced this claim's knowledge: NOT assumed to be
    # the bound execution action (which is an attempt, not an induction).
    # Correlated by the real execution ids the induction cited.
    correlated = operation_action_for(harness, prediction_id)
    adoption = (adoption_action_id
                or (correlated.action_id if correlated is not None else None)
                or trace.bound_action_id)
    produced = _operation_entries(harness, adoption)
    action = bound_action_of(harness, prediction_id)
    execution_ids = list(trace.bound_execution_ids)
    if action is not None and action.linked_execution_id:
        execution_ids = [str(action.linked_execution_id)] + execution_ids
    changes = []
    for raw in trace.expected_changes:
        if not isinstance(raw, dict):
            continue
        try:
            changes.append(ExpectedChange.from_dict(raw))
        except (ValueError, TypeError):
            continue
    conditions = []
    for raw in trace.verification_conditions:
        if not isinstance(raw, dict):
            continue
        try:
            conditions.append(VerificationCondition.from_dict(raw))
        except (ValueError, TypeError):
            continue
    prediction = CapabilityEvolutionPrediction(
        current_evidence=HarnessCapabilityEvidence(),
        candidate_operation=LearningOperation(
            operation_type="induce",
            strategy_id=trace.strategy_id,
            description=("the learning operation that followed the online "
                         "capability claim made with this strategy-outcome "
                         "prediction"),
            scope=ExperienceScope(
                execution_ids=produced["actual_execution_ids"],
                task_ids=[trace.task_id] if trace.task_id else [],
                family=trace.family,
                cell_token=trace.cell_token,
                note="the evidence the online claim was made about")),
        prediction_id=trace.prediction_id,
        status="valid",
        experience_scope=ExperienceScope(
            execution_ids=execution_ids,
            task_ids=[trace.task_id] if trace.task_id else [],
            note="the real execution the online claim was bound to"),
        task_targeting=TaskTargeting(
            description=str(trace.task_targeting.get("description") or ""),
            family=(trace.family or trace.task_targeting.get("family")),
            cell_token=(trace.cell_token
                        or trace.task_targeting.get("cell_token")),
            task_ids=list(trace.task_targeting.get("task_ids") or [])),
        horizon=trace.horizon,
        horizon_tasks=trace.horizon_tasks,
        expected_changes=changes,
        verification_conditions=conditions,
        service_available=False,
        provider_configured=False,
        service_implemented=True,
    )
    prediction.notes.append(
        "materialized ON DEMAND from the online capability-gain trace "
        f"{trace.trace_id!r}: the claim is the ORIGINAL one, never a "
        "re-prediction after seeing the result. It is not a stored "
        "capability prediction and not a billed model call")
    if not changes:
        prediction.notes.append(
            "the online claim named no expected change: there is nothing to "
            "verify, and the effect evaluation will report exactly that")
    info = prediction.trace.model_info
    info["materialized_from_online_trace"] = trace.trace_id
    info["online_prediction_id"] = trace.prediction_id
    return prediction
