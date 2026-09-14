"""M2 prediction contracts: candidate actions and structured outcomes.

Two dataclasses, deliberately separate from the M1 records:

- :class:`ActionSpec` describes a CANDIDATE action — something that has
  NOT happened yet. It reuses the M1 action-type vocabulary and parameter
  semantics, but a candidate has no result, no cost, and no id collision
  with real actions. Predicting over a spec never writes a fact.
- :class:`OutcomePrediction` is the frozen, structured result of asking a
  world model "what happens if this action runs from this state?". It
  carries the predicted successor observations (status / feasibility /
  quality / failure risk / cost / state changes), the model's own account
  of its evidence, the ACTUAL cost of the model call, and — after the real
  action runs — a feedback block that is APPENDED, never overwriting the
  original prediction.

Validation is framework-side: the model's output is data. Nothing in it is
executed, and an invalid payload becomes an explicit ``invalid_output``
status with whatever call cost is known — never a silently repaired
"prediction".
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.world_model.actions import ACTION_TYPES

#: Execution statuses a prediction may name (mirrors the executor's
#: ALLOWED_STATUSES; "unknown" = the model declined to predict).
PREDICTABLE_STATUSES = ("optimal", "feasible", "infeasible", "unbounded",
                        "timeout", "error", "unknown")

#: Lifecycle of a prediction record.
PREDICTION_STATUSES = ("valid", "invalid_output", "provider_error",
                       "not_configured", "unsupported_action")

#: Prompt template version — bumped when the request schema changes, so
#: stored predictions stay interpretable.
PROMPT_TEMPLATE_VERSION = "wm2/1"

#: Action types the M2 single-step prediction path supports. Others return
#: ``unsupported_action`` rather than a fabricated result.
SUPPORTED_ACTION_TYPES = ("execute_strategy",)


@dataclass
class ActionSpec:
    """A CANDIDATE action to be evaluated (not yet executed).

    Carries the action type, the strategy/solver and key configuration, the
    task/episode it belongs to, and the measurement scope the prediction
    should be compared under. Unknown implementation details (e.g. the code
    version that does not exist yet) stay unknown — the spec never
    fabricates them."""

    action_type: str
    task_id: str
    episode_id: Optional[str] = None
    strategy_id: Optional[str] = None
    solver: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    measurement_scope: str = "attempt"
    budget_hint: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {ACTION_TYPES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "strategy_id": self.strategy_id,
            "solver": self.solver,
            "params": copy.deepcopy(self.params),
            "measurement_scope": self.measurement_scope,
            "budget_hint": (dict(self.budget_hint)
                            if self.budget_hint is not None else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionSpec":
        if not isinstance(data, dict):
            raise ValueError("ActionSpec must be a JSON object")
        if not data.get("action_type"):
            raise ValueError("ActionSpec.action_type is required")
        return cls(
            action_type=str(data["action_type"]),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            strategy_id=data.get("strategy_id"),
            solver=data.get("solver"),
            params=copy.deepcopy(dict(data.get("params") or {})),
            measurement_scope=str(data.get("measurement_scope", "attempt")),
            budget_hint=(dict(data["budget_hint"])
                         if data.get("budget_hint") else None),
        )


@dataclass
class OutcomePrediction:
    """The frozen, structured prediction of one candidate action's
    consequences (see module docstring)."""

    prediction_id: str
    input_snapshot_id: str
    action_spec: ActionSpec
    status: str = "valid"
    #: What the model predicted. Every field is optional-by-design: a field
    #: the model did not (or could not) predict stays absent and is listed
    #: in ``unsupported_fields`` — an all-empty object with status=valid is
    #: rejected by validation.
    predicted: Dict[str, Any] = field(default_factory=dict)
    #: Fields explicitly not predicted / not applicable, with reasons.
    unsupported_fields: Dict[str, str] = field(default_factory=dict)
    #: Model self-reported confidence (NOT a calibrated probability).
    confidence: Optional[float] = None
    #: Evidence the model claims to rely on (entry ids, statistics refs).
    evidence_basis: List[str] = field(default_factory=list)
    #: Model identity, prompt template version, non-sensitive config
    #: summary, generation time.
    model_info: Dict[str, Any] = field(default_factory=dict)
    #: The ACTUAL cost of the model call itself (usage from the provider
    #: when available; unknown dimensions stay unmeasured, never zero).
    call_cost: Optional[CostVector] = None
    #: Which input-view keys were actually provided to the model.
    input_view_keys: List[str] = field(default_factory=list)
    #: Provider error detail when status is provider_error / invalid_output.
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    #: Binding + feedback (APPENDED after the real action; never part of
    #: the frozen prediction itself).
    bound_action_id: Optional[str] = None
    binding_mismatch: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None

    @staticmethod
    def new_id() -> str:
        return f"wp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "input_snapshot_id": self.input_snapshot_id,
            "action_spec": self.action_spec.to_dict(),
            "status": self.status,
            "predicted": copy.deepcopy(self.predicted),
            "unsupported_fields": dict(self.unsupported_fields),
            "confidence": self.confidence,
            "evidence_basis": list(self.evidence_basis),
            "model_info": copy.deepcopy(self.model_info),
            "call_cost": (self.call_cost.to_dict()
                          if self.call_cost is not None else None),
            "call_cost_measured": (sorted(self.call_cost.measured)
                                    if self.call_cost is not None
                                    and self.call_cost.measured is not None
                                    else None),
            "input_view_keys": list(self.input_view_keys),
            "error": self.error,
            "created_at": self.created_at,
            "bound_action_id": self.bound_action_id,
            "binding_mismatch": self.binding_mismatch,
            "feedback": copy.deepcopy(self.feedback),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomePrediction":
        if not isinstance(data, dict) or not data.get("prediction_id"):
            raise ValueError("OutcomePrediction.prediction_id is required")
        status = str(data.get("status", "valid"))
        if status not in PREDICTION_STATUSES:
            raise ValueError(
                f"status must be one of {PREDICTION_STATUSES}")
        raw_cost = data.get("call_cost")
        call_cost = (CostVector.from_dict(raw_cost)
                     if isinstance(raw_cost, dict) else None)
        raw_measured = data.get("call_cost_measured")
        if call_cost is not None and isinstance(raw_measured, list):
            call_cost.measured = {str(d) for d in raw_measured
                                  if d in COST_DIMENSIONS}
        confidence = data.get("confidence")
        return cls(
            prediction_id=str(data["prediction_id"]),
            input_snapshot_id=str(data.get("input_snapshot_id", "")),
            action_spec=ActionSpec.from_dict(
                data.get("action_spec") or {}),
            status=status,
            predicted=copy.deepcopy(dict(data.get("predicted") or {})),
            unsupported_fields=dict(data.get("unsupported_fields") or {}),
            confidence=(float(confidence)
                        if confidence is not None else None),
            evidence_basis=[str(e) for e in (data.get("evidence_basis") or [])],
            model_info=copy.deepcopy(dict(data.get("model_info") or {})),
            call_cost=call_cost,
            input_view_keys=[str(k) for k in
                             (data.get("input_view_keys") or [])],
            error=data.get("error"),
            created_at=float(data.get("created_at", time.time())),
            bound_action_id=data.get("bound_action_id"),
            binding_mismatch=data.get("binding_mismatch"),
            feedback=copy.deepcopy(dict(data.get("feedback"))
                                   if data.get("feedback") else None),
        )


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def validate_prediction_payload(payload: Dict[str, Any],
                                action_spec: ActionSpec) -> List[str]:
    """Framework-side validation of a model's raw payload.

    Returns a list of problems (empty = valid). Checks: field structure,
    status values, numeric finiteness, non-negative costs, probability
    ranges, and action semantics. The payload is DATA — nothing in it is
    executed."""
    problems: List[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    # A "prediction" with no predicted content is not a prediction.
    content_keys = ("outcome_status", "feasible", "quality", "failure_prob",
                    "cost", "state_changes", "expected_error_kinds")
    if not any(k in payload for k in content_keys):
        problems.append("no predicted content: at least one of "
                         + ", ".join(content_keys) + " is required")
    status = payload.get("outcome_status")
    if status is not None and status not in PREDICTABLE_STATUSES:
        problems.append(f"outcome_status {status!r} not in "
                       f"{PREDICTABLE_STATUSES}")
    feasible = payload.get("feasible")
    if feasible is not None and not isinstance(feasible, bool):
        problems.append("feasible must be a boolean")
    quality = payload.get("quality")
    if quality is not None:
        if not _finite(quality):
            problems.append("quality is not a finite number")
        elif not (0.0 <= float(quality) <= 1.0):
            problems.append("quality must be in [0, 1]")
    failure = payload.get("failure_prob")
    if failure is not None:
        if not _finite(failure):
            problems.append("failure_prob is not a finite number")
        elif not (0.0 <= float(failure) <= 1.0):
            problems.append("failure_prob must be in [0, 1]")
    cost = payload.get("cost")
    if cost is not None:
        if not isinstance(cost, dict):
            problems.append("cost must be a JSON object of dimensions")
        else:
            for d, v in cost.items():
                if d not in COST_DIMENSIONS:
                    problems.append(f"unknown cost dimension {d!r}")
                elif not _finite(v) or float(v) < 0:
                    problems.append(f"cost.{d} must be finite and >= 0")
    changes = payload.get("state_changes")
    if changes is not None and not isinstance(changes, dict):
        problems.append("state_changes must be a JSON object")
    kinds = payload.get("expected_error_kinds")
    if kinds is not None and not isinstance(kinds, list):
        problems.append("expected_error_kinds must be a list")
    # Action semantics: an execute_strategy prediction without a strategy_id
    # in the spec is meaningless (the model cannot know it either).
    if (action_spec.action_type == "execute_strategy"
            and not action_spec.strategy_id):
        problems.append("execute_strategy spec requires strategy_id")
    return problems
