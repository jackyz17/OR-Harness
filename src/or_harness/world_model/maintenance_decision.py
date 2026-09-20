"""M5 offline-improvement decision + two-stage capability feedback.

Three units, kept strictly apart:

1. **The comparison** (:func:`compare_capability_predictions`,
   :func:`recommend_maintenance`): given several FROZEN capability
   predictions over one shared input, produce a bounded, explainable
   recommendation — accept one, or ``defer``. It never touches the
   Strategic Bank and never ranks predictions that are not comparable.
2. **Stage 1 — the maintenance FACT** (:func:`bind_maintenance_fact`):
   after the operation really ran, record what happened: the operation
   identity, the knowledge delta, the verification outcome, the real cost
   and the evidence scope. This answers "did it happen, and what
   changed?" — NEVER "did the harness get stronger".
3. **Stage 2 — the EFFECT** (:func:`evaluate_capability_effect`): after
   qualified LATER tasks (or a pre-arranged paired evaluation) have real
   results, compare the predicted change against the observed change. Only
   here can ``effect_verified`` become True, and only for the sources the
   real evidence supports.

Design boundaries:

- **One comparison rule.** Under a declared quality-non-degradation
  constraint, compare the predicted resource saving over the declared
  horizon against the predicted maintenance cost. No universal weighted H
  score, no five-dimensional sum. Predictions whose metric/unit/baseline/
  quality constraints are not comparable are NOT auto-ranked: they are
  listed for the agent to choose between, and said to be incomparable.
- **``defer`` is a first-class outcome**, not a failure: no comparable
  benefit, insufficient evidence or an exhausted budget all yield an
  explainable defer with its reason.
- **A single-task benefit is never extrapolated** into a total: with no
  declared task count the comparison says so and stays per-task.
- **Facts are not effects.** A verified knowledge entry proves the
  knowledge changed; it says nothing about future performance.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.world_model.contracts import (
    CapabilityEvolutionPrediction,
    HarnessCapabilityEvidence,
)

#: The comparison's own version.
MAINTENANCE_COMPARISON_VERSION = "wm-maint/1"

#: The effect-evaluation version.
CAPABILITY_EFFECT_VERSION = "wm-effect/1"

#: The recommendation vocabulary. ``defer`` covers every honest "not now".
MAINTENANCE_RECOMMENDATIONS = ("accept", "defer", "insufficient_evidence")

#: The learning-operation types this build can really CARRY OUT through an
#: existing path. ``induce`` also performs a revision when the target cell
#: already has an entry, so ``revise`` is covered by the same path.
#: ``reverify`` has no execution entry yet: it may be predicted, but it must
#: never be RECOMMENDED or silently executed as something else.
EXECUTABLE_LEARNING_OPERATIONS = ("induce", "revise", "retire")

#: The metric families this build can observe on LATER real tasks. Only a
#: metric with a real observation channel may ever become ``effect_verified``.
OBSERVABLE_CAPABILITY_METRICS = (
    "normalized_solution_quality",
    "effective_completion_rate",
    "resource_cost",
)

#: Aliases accepted for the observable metrics (the model's own naming is
#: not the framework's vocabulary; an alias maps onto ONE real channel).
_METRIC_ALIASES: Dict[str, str] = {
    "normalized_solution_quality": "normalized_solution_quality",
    "solution_quality": "normalized_solution_quality",
    "normalized_quality": "normalized_solution_quality",
    "normalized_objective_gap": "normalized_solution_quality",
    "quality": "normalized_solution_quality",
    "effective_completion_rate": "effective_completion_rate",
    "completion_rate": "effective_completion_rate",
    "effective_completion": "effective_completion_rate",
    "resource_cost": "resource_cost",
    "cost": "resource_cost",
    "solver_runtime_s": "resource_cost",
    "normalized_cost": "resource_cost",
}

#: Effect-evaluation states. ``pending`` is NOT a terminal state: a horizon
#: that has not been reached must be re-evaluable, never frozen.
EFFECT_STATES = ("pending", "observed_improvement", "observed_degradation",
                 "no_change", "inconclusive", "insufficient_evidence",
                 "not_evaluable")

#: The states that are FINAL conclusions: qualified evidence really
#: SUPPORTED or OPPOSED the claim (or the operation provably changed
#: nothing). Every other state — a pending horizon, a merely descriptive
#: before/after movement, insufficient evidence, or a prediction that could
#: not be evaluated yet — stays RE-EVALUABLE: the first look is never
#: frozen, so a later paired reference or a newly closed episode can still
#: turn it into a verdict. The stored record is still replaced, never
#: appended, so re-evaluating never doubles the sample.
EFFECT_FINAL_STATES = ("observed_improvement", "observed_degradation",
                       "no_change")


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _observable_metric(metric: Any) -> Optional[str]:
    """The real observation channel a metric name maps onto, or None."""
    key = str(metric or "").strip().lower()
    return _METRIC_ALIASES.get(key)


def _new_id(prefix: str) -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 1. comparison and recommendation
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceRecommendation:
    """The auditable result of comparing candidate learning operations.

    Carries the recommendation, the reasoning, the per-candidate comparison
    entries, the ones that were NOT comparable (with their reasons), and the
    real spend of the comparison itself.
    """

    recommendation_id: str = field(default_factory=lambda: _new_id("mr"))
    created_at: float = field(default_factory=time.time)
    recommendation: str = "defer"
    basis: str = ""
    #: The candidate the recommendation names, when one is named.
    selected_prediction_id: Optional[str] = None
    selected_operation_type: Optional[str] = None
    #: Per-candidate comparison entries (comparable ones only).
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    #: Candidates that could NOT be auto-ranked, with their reasons.
    incomparable: List[Dict[str, Any]] = field(default_factory=list)
    #: The comparison rule actually applied, stated explicitly.
    rule: Dict[str, Any] = field(default_factory=dict)
    shared_input: Dict[str, Any] = field(default_factory=dict)
    #: The real spend of the comparison calls (never the predicted cost).
    comparison_cost: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_version": MAINTENANCE_COMPARISON_VERSION,
            "recommendation_id": self.recommendation_id,
            "created_at": self.created_at,
            "recommendation": self.recommendation,
            "basis": self.basis,
            "selected_prediction_id": self.selected_prediction_id,
            "selected_operation_type": self.selected_operation_type,
            "comparisons": copy.deepcopy(self.comparisons),
            "incomparable": copy.deepcopy(self.incomparable),
            "rule": copy.deepcopy(self.rule),
            "shared_input": copy.deepcopy(self.shared_input),
            "comparison_cost": copy.deepcopy(self.comparison_cost),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]
                  ) -> "MaintenanceRecommendation":
        data = data or {}
        return cls(
            recommendation_id=str(data.get("recommendation_id")
                                  or _new_id("mr")),
            created_at=float(data.get("created_at", time.time())),
            recommendation=str(data.get("recommendation", "defer")),
            basis=str(data.get("basis", "")),
            selected_prediction_id=data.get("selected_prediction_id"),
            selected_operation_type=data.get("selected_operation_type"),
            comparisons=copy.deepcopy(list(data.get("comparisons") or [])),
            incomparable=copy.deepcopy(list(data.get("incomparable") or [])),
            rule=copy.deepcopy(dict(data.get("rule") or {})),
            shared_input=copy.deepcopy(dict(data.get("shared_input") or {})),
            comparison_cost=copy.deepcopy(data.get("comparison_cost")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def _quality_constraint(prediction: CapabilityEvolutionPrediction
                        ) -> Optional[Dict[str, Any]]:
    """The prediction's own quality-non-degradation claim, when it makes one.

    Returns the QUALITY change the prediction expects (the one whose metric
    maps onto the observed quality channel), so the comparison can require
    that it does not predict a degradation. A prediction that says nothing
    about quality has NO constraint — which is not the same as a promise
    that quality holds.

    ``unchanged`` is NOT a degradation: a candidate that expects quality to
    hold is exactly the quality-safe case the constraint is looking for.
    Only a KNOWN direction that is the OPPOSITE of the beneficial one counts
    as a predicted degradation.
    """
    for change in prediction.expected_changes:
        if _observable_metric(change.metric) != \
                "normalized_solution_quality":
            continue
        return {
            "metric": change.metric,
            "direction": change.direction,
            "beneficial_direction": change.beneficial_direction,
            "is_improvement": change.is_improvement,
            "value": change.value,
            "predicts_degradation": _predicts_degradation(change),
        }
    return None


def _predicts_degradation(change: Any) -> bool:
    """Whether a change predicts a DEGRADATION on its own metric.

    Only a stated direction that contradicts the beneficial one counts:
    ``unchanged`` and ``unknown`` are not degradations (they are the
    absence of a claim, not a claim of loss).
    """
    direction = str(getattr(change, "direction", "unknown"))
    beneficial = str(getattr(change, "beneficial_direction", "either"))
    if beneficial == "either" or direction in ("unknown", "unchanged"):
        return False
    return direction != beneficial


def _cost_saving(prediction: CapabilityEvolutionPrediction,
                 horizon_tasks: Optional[int]) -> Optional[Dict[str, Any]]:
    """The predicted resource saving over the declared horizon, if any.

    Only a ``resource_cost`` change with a beneficial direction and a
    usable magnitude produces a saving. A saving is never manufactured from
    a cost block (the cost block is the LEARNING cost, not a saving), and a
    per-task saving is never multiplied into a total unless the caller
    DECLARED a task count.
    """
    for change in prediction.expected_changes:
        if _observable_metric(change.metric) != "resource_cost":
            continue
        if change.is_improvement is not True:
            continue
        if change.value is None:
            return {
                "kind": "unquantified",
                "metric": change.metric,
                "unit": change.unit,
                "note": ("the direction is beneficial but no magnitude was "
                         "predicted: the saving is not quantified"),
            }
        magnitude = abs(float(change.value))
        per_task = magnitude
        if change.value_kind == "relative":
            # A relative saving needs an ABSOLUTE value in the SAME metric
            # to become a per-task number. The prediction-level baseline
            # usually describes the QUALITY metric, so it must not be used
            # to convert a COST ratio — only the change's OWN baseline
            # counts.
            base_value = None
            if change.baseline is not None \
                    and change.baseline.value is not None:
                base_value = float(change.baseline.value)
            if base_value is None:
                return {
                    "kind": "relative",
                    "metric": change.metric,
                    "unit": change.unit,
                    "per_task_ratio": round(per_task, 6),
                    "note": ("a relative saving with no absolute baseline "
                             "value for ITS OWN metric: reported as a "
                             "ratio, never converted into a unit count"),
                }
            per_task = magnitude * base_value
        entry: Dict[str, Any] = {
            "kind": "quantified",
            "metric": change.metric,
            "unit": change.unit,
            "per_task": round(per_task, 6),
            "value_kind": change.value_kind,
        }
        if horizon_tasks is not None:
            entry["horizon_tasks"] = int(horizon_tasks)
            entry["total"] = round(per_task * int(horizon_tasks), 6)
        else:
            entry["total"] = None
            entry["note"] = ("no task count was declared: the saving is "
                             "reported PER TASK and never extrapolated into "
                             "a total")
        return entry
    return None


def _learning_cost_total(prediction: CapabilityEvolutionPrediction
                         ) -> Optional[Dict[str, Any]]:
    """The predicted LEARNING cost, as per-dimension numbers.

    Only dimensions the model actually predicted participate — an omitted
    dimension is unknown, never zero. No cross-dimension sum is invented:
    tokens and seconds do not add.
    """
    cost = prediction.learning_cost
    if cost is None or cost.expected is None:
        return None
    dims = {dim: round(float(getattr(cost.expected, dim)), 6)
            for dim in sorted(cost.expected.measured_dims())}
    if not dims:
        return None
    return {
        "per_dim": dims,
        "measured": sorted(dims),
        "note": ("the predicted cost of the offline learning operation, per "
                 "dimension: dimensions nobody predicted are unknown, and "
                 "different units are never summed into one number"),
    }


#: The unit a cost dimension is expressed in. A saving measured in seconds
#: and a learning cost measured in tokens are DIFFERENT currencies: netting
#: them off would be adding seconds to tokens.
COST_DIMENSION_UNITS: Dict[str, str] = {
    "solver_runtime_s": "s",
    "latency_s": "s",
    "llm_tokens": "tokens",
    "tool_calls": "calls",
    "retries": "retries",
}

#: Common spellings of a unit, folded onto one token so "s" and "seconds"
#: are recognized as the same currency.
_UNIT_ALIASES: Dict[str, str] = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "token": "tokens", "tokens": "tokens",
    "call": "calls", "calls": "calls",
    "retry": "retries", "retries": "retries",
}


def _normalize_unit(unit: Any) -> Optional[str]:
    """Fold a unit spelling onto its canonical token, or None when empty."""
    name = str(unit or "").strip().lower()
    if not name:
        return None
    return _UNIT_ALIASES.get(name, name)


def _maintenance_cost_in_unit(prediction: CapabilityEvolutionPrediction,
                              unit: Optional[str]
                              ) -> Optional[Dict[str, Any]]:
    """The predicted learning cost expressed in ONE unit, when it has one.

    The saving and the maintenance cost can only be netted when they are the
    SAME currency. A candidate whose learning cost is in tokens and whose
    saving is in seconds has NO comparable net: reporting a total would add
    two different quantities. Returns None in that case — "not comparable"
    is the honest answer, not zero.
    """
    total = _learning_cost_total(prediction)
    if total is None:
        return None
    if unit is None:
        return None
    matches = {dim: value for dim, value in total["per_dim"].items()
               if COST_DIMENSION_UNITS.get(dim) == unit}
    if not matches:
        return None
    return {
        "per_dim": matches,
        "unit": unit,
        "total_in_unit": round(sum(matches.values()), 6),
        "all_predicted_dims": total["per_dim"],
        "note": ("the learning cost restricted to the dimension(s) measured "
                 "in the SAME unit as the saving: dimensions in another "
                 "currency are reported but never added in"),
    }


def compare_capability_predictions(
        predictions: Sequence[CapabilityEvolutionPrediction], *,
        horizon_tasks: Optional[int] = None,
        require_quality_nondegradation: bool = True,
        ) -> MaintenanceRecommendation:
    """Compare frozen capability predictions and recommend one, or defer.

    The ONE comparison rule implemented here, stated in the result:

    > Among candidates that predict a QUANTIFIED resource saving over the
    > declared horizon and do NOT predict a quality degradation, pick the
    > largest per-task saving. Everything else is reported as incomparable
    > and left to the agent.

    Why this bounded rule: it needs one observable metric family, it cannot
    be gamed by an unstated direction (an omitted ``beneficial_direction``
    yields no benefit), and it never produces a universal score. A
    candidate whose benefit is real but not comparable (a different metric,
    a different unit, an unquantified direction) is listed rather than
    silently ranked, and ``defer`` is a legitimate result.

    Nothing here modifies the Strategic Bank, calls a model, or executes an
    operation.
    """
    result = MaintenanceRecommendation()
    result.rule = {
        "name": "largest_net_saving_under_quality_constraint",
        "description": (
            "among candidates that predict a quantified resource saving over "
            "the declared horizon and do NOT predict a quality degradation, "
            "compare each candidate's saving against ITS OWN predicted "
            "learning cost IN THE SAME UNIT, and recommend the largest net "
            "saving; a candidate whose saving and learning cost are in "
            "different units, or whose net saving is not positive, is "
            "reported rather than ranked"),
        "quality_constraint_applied": bool(require_quality_nondegradation),
        "horizon_tasks": horizon_tasks,
        "no_universal_score": True,
        "note": ("one bounded rule, not a weighted H score: metrics with no "
                 "shared yardstick are reported, never pooled, and a saving "
                 "is never compared against a cost in another currency"),
    }
    comparable: List[Dict[str, Any]] = []
    incomparable: List[Dict[str, Any]] = []
    for prediction in predictions:
        entry: Dict[str, Any] = {
            "prediction_id": prediction.prediction_id,
            "operation_type": prediction.candidate_operation.operation_type,
            "strategy_id": prediction.candidate_operation.strategy_id,
            "status": prediction.status,
        }
        if prediction.status != "valid":
            entry["reason"] = (
                f"status {prediction.status!r}: a prediction that did not "
                "produce observable consequences cannot be compared")
            incomparable.append(entry)
            continue
        if entry["operation_type"] not in EXECUTABLE_LEARNING_OPERATIONS:
            entry["reason"] = (
                f"operation_type {entry['operation_type']!r} has no "
                "execution path in this build "
                f"({sorted(EXECUTABLE_LEARNING_OPERATIONS)}): recommending "
                "an operation that cannot be carried out would invite "
                "running a DIFFERENT one under its name")
            incomparable.append(entry)
            continue
        quality = _quality_constraint(prediction)
        entry["quality_constraint"] = quality
        if require_quality_nondegradation and quality is not None \
                and quality.get("predicts_degradation"):
            entry["reason"] = (
                f"the prediction expects a QUALITY degradation "
                f"({quality.get('metric')} {quality.get('direction')}): a "
                "candidate that trades quality for cost is not auto-ranked")
            incomparable.append(entry)
            continue
        saving = _cost_saving(prediction, horizon_tasks)
        entry["saving"] = saving
        if saving is None:
            entry["reason"] = (
                "no resource-cost saving is predicted: a capability "
                "prediction that only improves quality (or predicts "
                "nothing observable) has no shared yardstick with the "
                "others here, so it is reported for the agent to choose "
                "rather than ranked")
            incomparable.append(entry)
            continue
        if saving.get("kind") != "quantified":
            entry["reason"] = (
                f"the saving is not a comparable magnitude "
                f"({saving.get('kind')}): a ratio without an absolute "
                "baseline, or a direction without a magnitude, cannot be "
                "ranked against a quantified saving")
            incomparable.append(entry)
            continue
        entry["learning_cost"] = _learning_cost_total(prediction)
        saving_unit = _normalize_unit(saving.get("unit"))
        net = _maintenance_cost_in_unit(prediction, saving_unit)
        # Decomposition, per dimension, so the agent can see the trade: a
        # saving in seconds and a cost in tokens are DIFFERENT currencies
        # and are never netted off into one number.
        entry["decomposition"] = {
            "saving_per_task": saving["per_task"],
            "saving_unit": saving.get("unit"),
            "learning_cost_per_dim": (
                (entry["learning_cost"] or {}).get("per_dim")),
            "note": ("the saving and the learning cost are in different "
                     "currencies and are shown side by side, never summed"),
        }
        if net is None:
            entry["reason"] = (
                "the predicted learning cost is not expressed in the "
                f"saving's own unit ({saving.get('unit')!r}): the saving "
                "and the maintenance cost are different currencies, so "
                "there is no net figure to rank — a bigger number in "
                "another unit is not a bigger saving")
            incomparable.append(entry)
            continue
        entry["net_saving"] = {
            "per_task": saving["per_task"],
            "unit": saving_unit,
            "maintenance_cost": net["total_in_unit"],
            "net_per_task": round(saving["per_task"]
                                  - net["total_in_unit"], 6),
            "net_over_horizon": (
                round((saving["per_task"] * int(horizon_tasks))
                      - net["total_in_unit"], 6)
                if horizon_tasks is not None else None),
            "maintenance_cost_dims": net["per_dim"],
            "other_cost_dims": {
                dim: value for dim, value
                in net["all_predicted_dims"].items()
                if dim not in net["per_dim"]},
            "note": ("the per-task saving minus the WHOLE predicted "
                     "maintenance cost, in the saving's own unit; learning "
                     "costs in another currency are reported but not "
                     "subtracted"),
        }
        if entry["net_saving"]["net_per_task"] <= 0:
            entry["reason"] = (
                f"the predicted maintenance cost "
                f"({net['total_in_unit']} {saving_unit}) is at least the "
                f"predicted saving ({saving['per_task']} {saving_unit}): "
                "an operation that costs more than it saves is not a "
                "recommendation, whatever the gross saving looks like")
            incomparable.append(entry)
            continue
        comparable.append(entry)

    result.comparisons = comparable
    result.incomparable = incomparable
    if not comparable:
        result.recommendation = (
            "insufficient_evidence" if not predictions else "defer")
        result.basis = (
            "no candidate predicts a quantified, quality-safe resource "
            "saving that exceeds its own predicted maintenance cost in the "
            "same unit: deferring is the honest result — no operation is "
            "recommended on incomparable, unquantified or loss-making "
            "predictions")
        result.notes.append(
            "defer is a legitimate decision outcome, not a failure: it is "
            "not a Harness action and it changes no knowledge")
        return result

    # Candidates are only comparable to EACH OTHER when their savings are in
    # the same unit. Ranking "10 seconds" against "100 tokens" would pick
    # whichever number happens to be larger, which is a unit conversion the
    # framework has no basis for. When the comparable set spans more than
    # one unit, the honest answer is to report both groups and defer.
    by_unit: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for entry in comparable:
        by_unit.setdefault(entry["net_saving"]["unit"], []).append(entry)
    if len(by_unit) > 1:
        result.comparisons = []
        for unit, entries in sorted(by_unit.items(),
                                    key=lambda kv: str(kv[0])):
            for entry in entries:
                entry["reason"] = (
                    f"the saving is in {unit!r} while other candidates "
                    "predict savings in a different unit: the numbers are "
                    "not mutually comparable, so no cross-unit ranking is "
                    "performed — the agent chooses between the groups")
                incomparable.append(entry)
        result.recommendation = "defer"
        result.basis = (
            "the comparable candidates predict savings in different units "
            f"({', '.join(sorted(str(u) for u in by_unit))}): the "
            "framework has no basis for converting one into another, so no "
            "candidate is ranked and the choice is left to the agent")
        result.notes.append(
            "a cross-unit comparison would pick whichever number is "
            "larger, which is exactly the kind of unit-blind ranking this "
            "rule refuses to do")
        return result

    best = max(comparable, key=lambda e: (e["net_saving"]["net_per_task"],
                                          e["prediction_id"]))
    result.recommendation = "accept"
    result.selected_prediction_id = best["prediction_id"]
    result.selected_operation_type = best["operation_type"]
    result.basis = (
        f"the largest net per-task saving "
        f"({best['net_saving']['net_per_task']} "
        f"{best['net_saving']['unit'] or ''} after "
        f"{best['net_saving']['maintenance_cost']} of predicted maintenance "
        f"cost, over "
        f"{horizon_tasks if horizon_tasks is not None else 'an undeclared '
        'number of'} tasks) with no predicted quality degradation")
    result.notes.append(
        "a recommendation is NOT an execution: the outer agent must "
        "explicitly accept it, and only acceptance runs the real operation")
    if len(comparable) > 1:
        result.notes.append(
            f"{len(comparable)} candidates were comparable; the others are "
            "reported in ``comparisons`` for a human/agent override")
    return result


# ---------------------------------------------------------------------------
# 2. stage 1 — the maintenance FACT
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceFactBinding:
    """Stage 1: what the offline operation ACTUALLY did.

    Answers only "did it happen, and what changed". A verified entry
    appearing here is a KNOWLEDGE change — it is not evidence that future
    task performance improved, and this record can never set
    ``effect_verified``.
    """

    binding_id: str = field(default_factory=lambda: _new_id("mf"))
    prediction_id: str = ""
    created_at: float = field(default_factory=time.time)
    operation_type: str = ""
    #: The adoption action that triggered the real operation.
    adoption_action_id: Optional[str] = None
    #: The real induction/operation action and its business result.
    operation_action_id: Optional[str] = None
    business_result: Optional[str] = None
    #: Knowledge before/after, as the operation recorded them.
    knowledge_delta: Dict[str, Any] = field(default_factory=dict)
    #: The evidence scope the operation REALLY used (verified against the
    #: predicted scope — a widened scope is reported, never accepted).
    actual_execution_ids: List[str] = field(default_factory=list)
    scope_consistent: bool = False
    scope_problems: List[str] = field(default_factory=list)
    #: Real cost of the operation, per dimension, from the action log.
    real_cost: Optional[Dict[str, Any]] = None
    #: The verification outcome of the knowledge the operation produced.
    verification: Dict[str, Any] = field(default_factory=dict)
    #: Whether the operation changed anything at all.
    changed: bool = False
    #: Stage marker: this is a FACT binding, never an effect verdict.
    stage: str = "maintenance_fact"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_version": MAINTENANCE_COMPARISON_VERSION,
            "binding_id": self.binding_id,
            "prediction_id": self.prediction_id,
            "created_at": self.created_at,
            "operation_type": self.operation_type,
            "adoption_action_id": self.adoption_action_id,
            "operation_action_id": self.operation_action_id,
            "business_result": self.business_result,
            "knowledge_delta": copy.deepcopy(self.knowledge_delta),
            "actual_execution_ids": list(self.actual_execution_ids),
            "scope_consistent": bool(self.scope_consistent),
            "scope_problems": list(self.scope_problems),
            "real_cost": copy.deepcopy(self.real_cost),
            "verification": copy.deepcopy(self.verification),
            "changed": bool(self.changed),
            "stage": self.stage,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaintenanceFactBinding":
        data = data or {}
        return cls(
            binding_id=str(data.get("binding_id") or _new_id("mf")),
            prediction_id=str(data.get("prediction_id", "")),
            created_at=float(data.get("created_at", time.time())),
            operation_type=str(data.get("operation_type", "")),
            adoption_action_id=data.get("adoption_action_id"),
            operation_action_id=data.get("operation_action_id"),
            business_result=data.get("business_result"),
            knowledge_delta=copy.deepcopy(
                dict(data.get("knowledge_delta") or {})),
            actual_execution_ids=[str(e) for e in
                                  (data.get("actual_execution_ids") or [])],
            scope_consistent=bool(data.get("scope_consistent", False)),
            scope_problems=[str(p) for p in
                            (data.get("scope_problems") or [])],
            real_cost=copy.deepcopy(data.get("real_cost")),
            verification=copy.deepcopy(dict(data.get("verification") or {})),
            changed=bool(data.get("changed", False)),
            stage=str(data.get("stage", "maintenance_fact")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def bind_maintenance_fact(harness, prediction_id: str, *,
                          adoption_action_id: Optional[str] = None
                          ) -> MaintenanceFactBinding:
    """Stage 1: bind the REAL maintenance fact to one prediction.

    Finds the operation that actually ran (through the adoption action, or
    through any ``induce`` action that references the prediction), records
    the knowledge delta, the REAL cost, the verification outcome and the
    evidence scope, and checks the scope against the predicted one. A
    widened scope is REPORTED, never silently accepted.

    It cannot set ``effect_verified``: creating ten verified entries proves
    the knowledge changed, not that future performance improved.
    """
    prediction = harness.capability_predictions.get(prediction_id)
    if prediction is None:
        raise ValueError(f"unknown capability prediction {prediction_id!r}")
    binding = MaintenanceFactBinding(
        prediction_id=prediction_id,
        operation_type=prediction.candidate_operation.operation_type,
        adoption_action_id=adoption_action_id,
    )
    adoption = None
    if adoption_action_id:
        adoption = harness.actions.get(adoption_action_id)
    if adoption is None:
        # Locate the adoption action that names this prediction. The
        # prediction id is recorded on the adoption's params so the binding
        # never has to guess which operation followed.
        for action in harness.actions.query():
            params = action.params or {}
            outcome = action.outcome or {}
            if params.get("capability_prediction_id") == prediction_id \
                    or outcome.get("capability_prediction_id") == \
                    prediction_id:
                adoption = action
                break
    if adoption is None:
        binding.notes.append(
            "no adoption action references this prediction: the operation "
            "was never explicitly accepted, so no maintenance fact exists "
            "to bind (deferring or rejecting leaves the knowledge alone)")
        return binding
    binding.adoption_action_id = adoption.action_id
    params = adoption.params or {}
    outcome = adoption.outcome or {}
    binding.operation_action_id = (
        outcome.get("operation_action_id")
        or params.get("operation_action_id"))
    operation_result = (outcome.get("operation_result")
                        or params.get("operation_result") or {})
    binding.business_result = operation_result.get("business_result")
    delta = operation_result.get("knowledge_delta") or {}
    binding.knowledge_delta = copy.deepcopy(delta)
    binding.actual_execution_ids = [str(e) for e in
                                    (operation_result.get("execution_ids")
                                     or [])]
    binding.changed = bool(delta.get("entries_created")
                           or delta.get("entries_updated")
                           or delta.get("entry_changes"))

    # Scope consistency: the operation must have used exactly the scope the
    # prediction was made about. A widened scope means the prediction was
    # not about the thing that ran.
    predicted_scope = prediction.experience_scope
    predicted_ids = set(predicted_scope.execution_ids) \
        if predicted_scope is not None else set()
    actual_ids = set(binding.actual_execution_ids)
    if predicted_ids and actual_ids:
        widened = actual_ids - predicted_ids
        if widened:
            binding.scope_problems.append(
                f"the operation used {len(widened)} execution(s) outside the "
                "predicted scope: the prediction was about a NARROWER "
                "evidence set than the one consolidated")
        if predicted_ids - actual_ids:
            binding.scope_problems.append(
                f"{len(predicted_ids - actual_ids)} predicted execution(s) "
                "were not part of the operation's scope")
        binding.scope_consistent = not binding.scope_problems
    else:
        binding.scope_problems.append(
            "the scope could not be compared (one side records no "
            "execution ids): scope consistency is UNKNOWN, never assumed")

    # Real cost: from the action log, never from the prediction.
    if binding.operation_action_id:
        action = harness.actions.get(binding.operation_action_id)
        if action is not None:
            if action.cost is None:
                binding.real_cost = {
                    "per_dim": None,
                    "measured": [],
                    "note": ("the operation action records NO cost: unknown, "
                             "never zero"),
                }
            else:
                dims = {d: round(float(getattr(action.cost, d)), 6)
                        for d in sorted(action.cost.measured_dims())}
                binding.real_cost = {
                    "per_dim": dims,
                    "measured": sorted(dims),
                    "note": ("the REAL measured spend of the offline "
                             "operation, from the action log"),
                }
    verification = operation_result.get("verification") or {}
    binding.verification = {
        "verdict": verification.get("verdict"),
        "n_verified": verification.get("n_verified"),
        "note": ("the knowledge the operation produced and its admission "
                 "verdict: a VERIFIED ENTRY is a knowledge change, not a "
                 "capability improvement"),
    }
    binding.notes.append(
        "stage 1 of 2: this binding answers whether the operation happened "
        "and what it changed. It never sets effect_verified — that requires "
        "real later-task performance, which has not been observed yet")
    if not binding.changed:
        binding.notes.append(
            "the operation produced NO knowledge change (no entry created "
            "or updated): an operation with no change is a recorded outcome, "
            "not a capability gain")
    return binding


# ---------------------------------------------------------------------------
# 3. stage 2 — the capability EFFECT
# ---------------------------------------------------------------------------


@dataclass
class CapabilityEffectEvaluation:
    """Stage 2: the predicted capability change vs the OBSERVED change.

    Built only from real later-task results (or a pre-arranged paired
    evaluation). Every field carries its own eligibility; an inconclusive
    or insufficient-evidence result is a first-class outcome, and a
    ``pending`` horizon can always be re-evaluated later (it is never
    frozen).
    """

    evaluation_id: str = field(default_factory=lambda: _new_id("ce"))
    prediction_id: str = ""
    binding_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    state: str = "pending"
    #: Per-change comparison entries.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    #: Which capability sources the REAL evidence supports, and at what
    #: strength. Only these may be reported as ``direct_evidence``.
    source_evidence: Dict[str, Any] = field(default_factory=dict)
    #: The real evidence the evaluation read.
    evidence: Dict[str, Any] = field(default_factory=dict)
    exclusion_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def effect_verified(self) -> bool:
        """Whether the real evidence SUPPORTS the predicted improvement.

        True only for an observed improvement under a comparable setup.
        An unverified horizon, a missing reference or an inconclusive
        result is never True.
        """
        return self.state == "observed_improvement"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect_version": CAPABILITY_EFFECT_VERSION,
            "evaluation_id": self.evaluation_id,
            "prediction_id": self.prediction_id,
            "binding_id": self.binding_id,
            "created_at": self.created_at,
            "state": self.state,
            "effect_verified": self.effect_verified,
            "changes": copy.deepcopy(self.changes),
            "source_evidence": copy.deepcopy(self.source_evidence),
            "evidence": copy.deepcopy(self.evidence),
            "exclusion_reasons": list(self.exclusion_reasons),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]
                  ) -> "CapabilityEffectEvaluation":
        data = data or {}
        return cls(
            evaluation_id=str(data.get("evaluation_id") or _new_id("ce")),
            prediction_id=str(data.get("prediction_id", "")),
            binding_id=data.get("binding_id"),
            created_at=float(data.get("created_at", time.time())),
            state=str(data.get("state", "pending")),
            changes=copy.deepcopy(list(data.get("changes") or [])),
            source_evidence=copy.deepcopy(
                dict(data.get("source_evidence") or {})),
            evidence=copy.deepcopy(dict(data.get("evidence") or {})),
            exclusion_reasons=[str(r) for r in
                               (data.get("exclusion_reasons") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def evaluate_capability_effect(
        harness, prediction_id: str, *,
        binding: Optional[MaintenanceFactBinding] = None,
        task_ids: Optional[Sequence[str]] = None,
        require_paired_reference: bool = True,
        ) -> CapabilityEffectEvaluation:
    """Stage 2: evaluate one prediction against REAL later-task results.

    Reads only evidence with a real execution behind it:

    - the LATER tasks named by ``task_ids`` (their CLOSED episodes'
      evaluations), or
    - every closed episode of a task that ran AFTER the operation.

    A pre-arranged paired evaluation is accepted when the caller records
    one through :func:`record_paired_evaluation`; without a reference the
    evaluation can describe a before/after change but may NOT claim the
    operation caused it (``inconclusive`` says so).

    The horizon decides ``pending``: an unreached horizon stays re-
    evaluable forever. Repeating the call is idempotent (the stored
    evaluation is returned), so a second look never doubles the sample.
    """
    from or_harness.world_model.episode_closeout import (
        _iter_closed_evaluations,
    )
    prediction = harness.capability_predictions.get(prediction_id)
    if prediction is None:
        raise ValueError(f"unknown capability prediction {prediction_id!r}")
    if binding is None:
        binding = get_maintenance_binding(harness, prediction_id)
    evaluation = CapabilityEffectEvaluation(
        prediction_id=prediction_id,
        binding_id=(binding.binding_id if binding is not None else None),
    )
    if binding is None:
        evaluation.state = "not_evaluable"
        evaluation.exclusion_reasons.append(
            "no maintenance fact is bound: the operation's real effect "
            "cannot be judged before knowing whether it happened")
        return evaluation
    if not binding.changed:
        evaluation.state = "no_change"
        evaluation.exclusion_reasons.append(
            "the operation produced no knowledge change, so there is no "
            "capability change to evaluate")
        evaluation.notes.append(
            "a no-change operation is recorded honestly: it is not a "
            "failure and not an improvement")
        return evaluation

    # The metric this build can actually observe. A prediction whose
    # changes are all unobservable is NOT evaluated against a substitute.
    observable_changes = [c for c in prediction.expected_changes
                          if _observable_metric(c.metric) is not None]
    if not observable_changes:
        evaluation.state = "not_evaluable"
        evaluation.exclusion_reasons.append(
            "no expected change names a metric with a real observation "
            "channel in this build "
            f"({OBSERVABLE_CAPABILITY_METRICS}): the predicted metrics are "
            "reported, never silently replaced by an observable one")
        return evaluation

    # Collect the LATER real evaluations. Independence is enforced by the
    # full (task_id, episode_id) identity, and an episode that was already
    # part of the prediction's own experience scope is EXCLUDED from the
    # validation sample: reusing the induction tasks checks consistency,
    # never transfer.
    #
    # A task only counts as LATER when it really finished AFTER the
    # operation ran, and it only counts as MATCHING when it falls inside
    # the prediction's FROZEN targeting. A task that closed before the
    # maintenance cannot be evidence about it, and a task of another family
    # or structural cell cannot validate a claim about this population —
    # admitting either would let the verdict rest on a sample the claim was
    # never about.
    induction_tasks = set(prediction.experience_scope.task_ids) \
        if prediction.experience_scope is not None else set()
    maintenance_at = _maintenance_started_at(harness, prediction_id, binding)
    declared_horizon = prediction.horizon_tasks
    later: List[Any] = []
    excluded_not_later: List[Dict[str, Any]] = []
    excluded_off_target: List[Dict[str, Any]] = []
    unverifiable: List[Dict[str, Any]] = []
    for item in _iter_closed_evaluations(harness):
        if item.state != "evaluated":
            continue
        if task_ids is not None and item.task_id not in set(task_ids):
            continue
        if item.task_id in induction_tasks:
            continue
        closed_at = _episode_closed_at(harness, item.task_id,
                                       item.episode_id)
        if maintenance_at is None or closed_at is None:
            unverifiable.append({
                "task_id": item.task_id,
                "episode_id": item.episode_id,
                "reason": ("the maintenance time or the episode's close time "
                           "is unknown, so whether this task really ran "
                           "AFTER the operation cannot be established"),
            })
            continue
        if closed_at < maintenance_at:
            excluded_not_later.append({
                "task_id": item.task_id,
                "episode_id": item.episode_id,
                "closed_at": closed_at,
                "maintenance_at": maintenance_at,
                "reason": ("the episode closed BEFORE the offline operation "
                           "ran: it cannot be evidence about the operation"),
            })
            continue
        profile = None
        try:
            records = harness.bank.query(task_id=item.task_id)
            if records:
                profile = getattr(records[-1], "profile_snapshot", None)
        except Exception:
            profile = None
        target_check = _targeting_matches(prediction, item.task_id, profile)
        if not target_check.get("matches"):
            excluded_off_target.append({
                "task_id": item.task_id,
                "episode_id": item.episode_id,
                "reason": ("the task is outside the prediction's declared "
                           "target: " + "; ".join(
                               target_check.get("problems") or [])),
            })
            continue
        later.append(item)

    # The horizon bounds the sample: a claim made over "the next N matching
    # tasks" is validated by at most N of them, and reaching the horizon is
    # what makes the evidence COMPLETE. Fewer than the declared count leaves
    # the evaluation pending rather than declared supported.
    horizon_met = (declared_horizon is None
                   or len(later) >= int(declared_horizon))
    if declared_horizon is not None and len(later) > int(declared_horizon):
        later = later[:int(declared_horizon)]
    evaluation.evidence = {
        "n_later_evaluations": len(later),
        "task_ids": sorted({i.task_id for i in later}),
        "distinct_episodes": len({(i.task_id, i.episode_id or "")
                                  for i in later}),
        "induction_tasks_excluded": sorted(induction_tasks),
        "excluded_not_later": excluded_not_later,
        "excluded_off_target": excluded_off_target,
        "unverifiable_timing": unverifiable,
        "maintenance_started_at": maintenance_at,
        "horizon_tasks": declared_horizon,
        "horizon_met": horizon_met,
        "note": ("only CLOSED episodes of tasks that finished AFTER the "
                 "operation, fall inside the prediction's frozen targeting "
                 "and are NOT part of its own experience scope participate: "
                 "reusing the induction tasks would check consistency, not "
                 "transfer"),
    }
    if not later:
        evaluation.state = "pending"
        evaluation.exclusion_reasons.append(
            "no qualified LATER matching task has produced a closed episode "
            "yet: the horizon is unmet and the evaluation stays pending — a "
            "pending horizon is re-evaluable, never frozen")
        return evaluation
    if not horizon_met:
        evaluation.state = "pending"
        evaluation.exclusion_reasons.append(
            f"only {len(later)} of the declared {declared_horizon} matching "
            "later task(s) have closed: the declared horizon is not "
            "reached, so the evidence is INCOMPLETE and the evaluation "
            "stays pending (it is re-evaluable, never frozen)")
        evaluation.notes.append(
            "the horizon was declared BEFORE the operation ran: shortening "
            "it once results are visible would be reading the result "
            "backwards")
        return evaluation

    # Per-change comparison, against the prediction's OWN declared
    # metric/unit/baseline. A paired record, when the caller arranged one,
    # is READ AND USED: its treated-vs-reference difference IS the
    # attributable observed change for the channel it covers. Merely
    # knowing that a record exists proves nothing — a pair that shows a
    # COLLAPSE must be able to refute the claim it was recorded for.
    paired = get_paired_evaluation(harness, prediction_id)
    paired_used_for: List[str] = []
    paired_problems: List[Dict[str, Any]] = []
    outcomes: List[str] = []
    for change in observable_changes:
        channel = _observable_metric(change.metric)
        observed = _observed_change(
            later, channel, baseline_value=_baseline_value_for(
                change, prediction))
        entry: Dict[str, Any] = {
            "metric": change.metric,
            "channel": channel,
            "unit": change.unit,
            "predicted_direction": change.direction,
            "predicted_value": change.value,
            "beneficial_direction": change.beneficial_direction,
            "predicted_is_improvement": change.is_improvement,
        }
        if paired is not None:
            check = _paired_covers(paired, change, channel)
            if check.get("covers"):
                # The comparison must rest on episodes that really closed:
                # a pair naming tasks with no closed episode is an assertion
                # with a citation, not a measurement.
                evidence_check = _paired_task_evidence(harness, paired)
                if not evidence_check["ok"]:
                    paired_problems.append({
                        "metric": change.metric,
                        "reason": evidence_check["reason"],
                    })
                else:
                    entry["paired_reference"] = {
                        "reference_value": paired["reference_value"],
                        "treated_value": paired["treated_value"],
                        "source": paired.get("source"),
                        "reference_task_ids":
                            paired.get("reference_task_ids"),
                        "note": ("the change is read from the caller's "
                                 "PAIRED comparison (treated minus "
                                 "reference), not from an uncontrolled "
                                 "before/after movement"),
                    }
                    observed = {
                        "observed_n": len(paired.get("reference_task_ids")
                                          or []) or None,
                        "observed_mean": paired["treated_value"],
                        "observed_change": float(paired["change"]),
                        "change_basis": (
                            "the paired comparison's treated value minus "
                            "its reference value: a like-for-like "
                            "difference the caller really arranged"),
                        "paired": True,
                    }
                    paired_used_for.append(change.metric)
                    check = {"covers": True}
            if not check.get("covers"):
                paired_problems.append({
                    "metric": change.metric,
                    "reason": check.get("reason"),
                })
        if observed is None or observed.get("observed_change") is None:
            entry["eligibility"] = "unobserved"
            entry["reason"] = (
                (observed or {}).get("change_basis")
                or f"no real observation on the {channel} channel: the "
                   "change is reported, never scored against a substitute")
            evaluation.changes.append(entry)
            continue
        entry.update(observed)
        entry["eligibility"] = "evaluable"
        observed_change = observed["observed_change"]
        entry["observed_is_improvement"] = (
            observed_change > 0 if change.beneficial_direction == "increase"
            else observed_change < 0
            if change.beneficial_direction == "decrease" else None)
        if change.is_improvement is None:
            entry["agreement"] = "not_compared"
            entry["reason"] = (
                "the prediction did not state which direction is an "
                "improvement, so the observed change cannot be called a "
                "confirmation or a refutation")
        else:
            entry["agreement"] = ("confirmed"
                                  if change.is_improvement
                                  == bool(entry["observed_is_improvement"])
                                  else "refuted")
        evaluation.changes.append(entry)
        if entry["agreement"] == "confirmed":
            outcomes.append("improvement"
                            if entry["observed_is_improvement"] else
                            "degradation")
        elif entry["agreement"] == "refuted":
            outcomes.append("refuted")

    evaluable = [c for c in evaluation.changes
                 if c.get("eligibility") == "evaluable"
                 and c.get("agreement") != "not_compared"]
    evaluation.evidence["paired_reference_used_for"] = paired_used_for
    if paired_problems:
        evaluation.evidence["paired_reference_unusable"] = paired_problems
    if not evaluable:
        evaluation.state = "insufficient_evidence"
        evaluation.exclusion_reasons.append(
            "no expected change could be compared against a real "
            "observation with a declared beneficial direction")
    elif require_paired_reference and not paired_used_for:
        # Attribution needs a comparison the operation's change was
        # actually READ FROM. A stored pair that covers none of the
        # predicted channels (a different metric, a different unit, no
        # execution evidence) is NOT a control for this claim, and a
        # before/after movement with no control describes what happened,
        # not what caused it.
        evaluation.state = "inconclusive"
        evaluation.exclusion_reasons.append(
            "no paired reference or comparable control APPLIES to any "
            "predicted change: the observed change is DESCRIPTIVE (a "
            "before/after movement) and cannot be attributed to this "
            "operation")
        if paired is not None:
            evaluation.exclusion_reasons.append(
                "a paired record exists but covers none of the predicted "
                "metrics/units — its existence alone is not attribution")
        evaluation.notes.append(
            "record a paired evaluation through record_paired_evaluation "
            "that matches the predicted metric AND unit to make the change "
            "attributable, or read this as a descriptive observation only")
    elif all(c["agreement"] == "confirmed" for c in evaluable) \
            and "improvement" in outcomes:
        evaluation.state = "observed_improvement"
        evaluation.notes.append(
            "the real evidence supports the predicted direction of "
            "improvement under a comparable setup")
    elif "degradation" in outcomes:
        evaluation.state = "observed_degradation"
        evaluation.notes.append(
            "the real evidence shows a DEGRADATION: a negative result is "
            "kept, never dropped from the record")
    elif "refuted" in outcomes:
        evaluation.state = "no_change"
        evaluation.notes.append(
            "the observed change contradicts the prediction: recorded as a "
            "refuted prediction, which is not the same as a degradation")
    else:
        evaluation.state = "inconclusive"

    # Source evidence: which capability sources the real evidence supports.
    # Only the sources with a real observation behind them become direct;
    # everything else stays indirect or absent.
    evaluation.source_evidence = _source_evidence_from_evaluation(
        prediction, evaluation)
    return evaluation


def _baseline_value_for(change: Any, prediction: Any
                        ) -> Optional[float]:
    """The absolute reference an observed change is measured against.

    The change's OWN baseline wins; the prediction-level baseline is used
    only when the change declares none AND the metric matches the one the
    prediction-level baseline describes (it usually describes quality, so
    it must not be silently applied to a cost change).
    """
    if change.baseline is not None and change.baseline.value is not None:
        return float(change.baseline.value)
    baseline = getattr(prediction, "baseline", None)
    if baseline is not None and baseline.value is not None \
            and _observable_metric(change.metric) == \
            "normalized_solution_quality":
        return float(baseline.value)
    return None


def _observed_change(evaluations: Sequence[Any], channel: str, *,
                     baseline_value: Optional[float] = None
                     ) -> Optional[Dict[str, Any]]:
    """The mean real observation on one channel, when it exists.

    Reads the M4 evaluation records — real, closed-episode measurements —
    never a model's self-report. The channel decides what is read: the
    quality channel uses the observed normalized quality, the cost channel
    the measured solver runtime.

    ``observed_change`` is the observed mean MINUS the declared baseline:
    the observed LEVEL is not itself a change, and reporting it as one
    would credit the operation with the whole distance from zero.
    """
    values: List[float] = []
    if channel == "normalized_solution_quality":
        for item in evaluations:
            benefit = item.benefit or {}
            if benefit.get("eligibility") == "evaluable" \
                    and benefit.get("observed") is not None:
                values.append(float(benefit["observed"]))
    elif channel == "resource_cost":
        for item in evaluations:
            cost = (item.cost or {}).get("per_dim") or {}
            entry = cost.get("solver_runtime_s")
            if isinstance(entry, dict) and entry.get("actual") is not None:
                values.append(float(entry["actual"]))
    elif channel == "effective_completion_rate":
        for item in evaluations:
            risk = item.risk or {}
            # A completed scope with no observed failure is the closest
            # real completion signal this build has: it is derived from
            # the execution statuses, never from a model's opinion.
            for scored in risk.get("scored") or []:
                if scored.get("event") in ("no_feasible_solution",
                                           "model_invalid") \
                        and scored.get("label") is not None:
                    values.append(0.0 if scored["label"] == "occurred"
                                  else 1.0)
    if not values:
        return None
    mean = sum(values) / len(values)
    entry: Dict[str, Any] = {
        "observed_n": len(values),
        "observed_mean": round(mean, 6),
        "note": ("the mean real observation on this channel over the later "
                 "closed episodes; no model self-report participates"),
    }
    if baseline_value is None:
        entry["observed_change"] = None
        entry["change_basis"] = (
            "no baseline was declared for this metric, so the observed "
            "LEVEL is reported but no CHANGE can be computed against it")
    else:
        entry["baseline_value"] = round(float(baseline_value), 6)
        entry["observed_change"] = round(
            mean - float(baseline_value), 6)
        entry["change_basis"] = (
            "the observed mean minus the baseline the prediction declared, "
            "frozen before the operation ran")
    return entry


def _paired_covers(paired: Dict[str, Any], change: Any,
                   channel: Optional[str]) -> Dict[str, Any]:
    """Whether a paired record is a CONTROL for one predicted change.

    A paired comparison only speaks to the change it was taken on. It
    covers a change when:

    - its metric maps onto the SAME observation channel (a quality pair
      says nothing about cost), and
    - its unit matches the change's unit, folded onto one spelling (seconds
      are not tokens), and
    - it carries a finite reference/treated pair with a usable difference
      and names the tasks it compared, so the comparison has real
      execution evidence behind it.

    Anything else is reported as UNUSABLE for this change: the record's
    mere existence is not attribution.
    """
    if not isinstance(paired, dict):
        return {"covers": False, "reason": "the paired record is malformed"}
    paired_metric = _observable_metric(paired.get("metric"))
    if paired_metric is None:
        return {"covers": False,
                "reason": (f"the paired record's metric "
                           f"{paired.get('metric')!r} has no observation "
                           "channel in this build")}
    if channel is not None and paired_metric != channel:
        return {"covers": False,
                "reason": (f"the paired record measures {paired_metric!r} "
                           f"while the change is about {channel!r}: a pair "
                           "taken on another channel is not a control")}
    paired_unit = _normalize_unit(paired.get("unit"))
    change_unit = _normalize_unit(getattr(change, "unit", None))
    if paired_unit is not None and change_unit is not None \
            and paired_unit != change_unit:
        return {"covers": False,
                "reason": (f"the paired record is in {paired_unit!r} while "
                           f"the change is in {change_unit!r}: different "
                           "units are different quantities")}
    reference = paired.get("reference_value")
    treated = paired.get("treated_value")
    if not _finite(reference) or not _finite(treated):
        return {"covers": False,
                "reason": "the paired record has a non-finite value"}
    change_value = paired.get("change")
    if not _finite(change_value):
        return {"covers": False,
                "reason": "the paired record has no usable difference"}
    if abs((float(treated) - float(reference)) - float(change_value)) > 1e-6:
        return {"covers": False,
                "reason": ("the paired record's difference does not equal "
                           "treated minus reference: the record is "
                           "internally inconsistent")}
    if not paired.get("reference_task_ids"):
        return {"covers": False,
                "reason": ("the paired record names no compared task, so "
                           "there is no execution evidence behind the "
                           "comparison")}
    return {"covers": True}


def _maintenance_started_at(harness, prediction_id: str,
                            binding: Optional[MaintenanceFactBinding]
                            ) -> Optional[float]:
    """When the offline operation really ran (the cut-off for LATER tasks).

    A "later" task must have finished AFTER the maintenance it is supposed
    to validate; a task that closed before the operation ran cannot be
    evidence about it. The adoption action's timestamp is the anchor, with
    the binding's creation time as a fallback. None means the time is
    UNKNOWN, and an unknown cut-off is not treated as "everything is
    later": the sample is reported as unverifiable instead.
    """
    action_id = (binding.adoption_action_id if binding is not None else None)
    if action_id:
        for act in harness.actions.query():
            if act.action_id == action_id:
                if act.started_at is not None:
                    return float(act.started_at)
                if act.ended_at is not None:
                    return float(act.ended_at)
    if binding is not None and binding.created_at is not None:
        return float(binding.created_at)
    return None


def _episode_closed_at(harness, task_id: str,
                       episode_id: Optional[str]) -> Optional[float]:
    """When one episode's close-out was recorded, or None when unknown."""
    from or_harness.world_model.episode_closeout import (
        episode_closeout_record,
    )
    record = episode_closeout_record(harness, task_id, episode_id)
    return float(record.created_at) if record is not None else None


def _cell_tokens_match(declared: str, actual: str) -> bool:
    """Whether two structural-cell tokens describe the SAME cell.

    Cell tokens are interval labels (``rc[0.25,0.50]|tc[...]``). Comparing
    them as raw strings would treat ``[0.25,0.5]`` and ``[0.25,0.50]`` as
    different cells purely because of formatting, so the intervals are
    compared NUMERICALLY per dimension. A token that cannot be parsed falls
    back to an exact string comparison rather than being assumed to match.
    """
    if declared == actual:
        return True
    left = _parse_cell_token(declared)
    right = _parse_cell_token(actual)
    if left is None or right is None:
        return False
    if set(left) != set(right):
        return False
    for dim, (lo_a, hi_a) in left.items():
        lo_b, hi_b = right[dim]
        if abs(lo_a - lo_b) > 1e-9 or abs(hi_a - hi_b) > 1e-9:
            return False
    return True


def _parse_cell_token(token: str) -> Optional[Dict[str, Tuple[float, float]]]:
    """Parse ``rc[0.25,0.50]|tc[0.00,0.25]`` into per-dimension intervals."""
    out: Dict[str, Tuple[float, float]] = {}
    for part in str(token or "").split("|"):
        part = part.strip()
        if "[" not in part or not part.endswith("]"):
            return None
        dim, _, bounds = part.partition("[")
        bounds = bounds[:-1]
        pieces = bounds.split(",")
        if len(pieces) != 2:
            return None
        try:
            out[dim.strip()] = (float(pieces[0]), float(pieces[1]))
        except ValueError:
            return None
    return out or None


def _paired_task_evidence(harness, paired: Dict[str, Any]) -> Dict[str, Any]:
    """Whether the tasks a paired record names really produced evidence.

    A pair is a comparison between REAL measurements, so the tasks it names
    must have closed episodes behind them (or real executions at the very
    least). A citation to a task nobody ran is an assertion wearing a
    reference, and it must not become an effect verdict.
    """
    from or_harness.world_model.episode_closeout import (
        episode_closeout_record,
    )
    task_ids = [str(t) for t in (paired.get("reference_task_ids") or [])]
    if not task_ids:
        return {"ok": False,
                "reason": ("the paired record names no compared task, so "
                           "there is no execution evidence behind the "
                           "comparison")}
    missing: List[str] = []
    for task_id in task_ids:
        closed = False
        try:
            rows = harness.store.conn.execute(
                "SELECT key FROM meta WHERE key LIKE ?",
                (f"episode_closeout|{task_id}|%",)).fetchall()
            closed = bool(rows)
        except Exception:
            closed = False
        if not closed:
            try:
                closed = bool(harness.bank.query(task_id=task_id))
            except Exception:
                closed = False
        if not closed:
            missing.append(task_id)
    if missing:
        return {"ok": False,
                "reason": (f"the paired record cites task(s) {missing} with "
                           "no closed episode and no recorded execution: a "
                           "comparison between measurements nobody made is "
                           "not evidence")}
    return {"ok": True}


def _targeting_matches(prediction: CapabilityEvolutionPrediction,
                       task_id: str, profile: Any) -> Dict[str, Any]:
    """Whether one later task falls inside the prediction's declared target.

    The frozen targeting is the filter: a task of another family, or one
    outside the declared structural cell, is NOT a "matching task" and must
    not silently validate a claim made about a different population. A
    targeting that declares nothing about an attribute does not restrict
    it; an attribute the profile cannot supply is reported as unverifiable
    rather than assumed to match.
    """
    targeting = prediction.task_targeting
    if targeting is None:
        return {"matches": True, "basis": "no targeting was declared"}
    problems: List[str] = []
    if targeting.family:
        actual = getattr(profile, "family", None)
        if actual is None:
            problems.append(
                f"family {targeting.family!r} is declared but the task's "
                "family is unknown")
        elif str(actual) != str(targeting.family):
            problems.append(
                f"family {actual!r} is outside the declared target "
                f"{targeting.family!r}")
    if targeting.cell_token:
        from or_harness.core.schema import group_key
        actual_cell = None
        if profile is not None:
            parts = group_key(profile).split("|", 1)
            actual_cell = parts[1] if len(parts) > 1 else ""
        if not actual_cell:
            problems.append(
                f"cell {targeting.cell_token!r} is declared but the task's "
                "structural cell is unknown")
        elif not _cell_tokens_match(str(targeting.cell_token),
                                    str(actual_cell)):
            problems.append(
                f"cell {actual_cell!r} is outside the declared target "
                f"{targeting.cell_token!r}")
    if targeting.task_ids and task_id not in set(targeting.task_ids):
        problems.append(
            f"task {task_id!r} is not among the declared target tasks")
    return {"matches": not problems, "problems": problems}


def _source_evidence_from_evaluation(
        prediction: CapabilityEvolutionPrediction,
        evaluation: CapabilityEffectEvaluation) -> Dict[str, Any]:
    """Which capability sources the REAL evidence supports.

    A source becomes ``direct_evidence`` only when a real later-task
    observation speaks to it. Everything else keeps its previous strength,
    and sources nothing observed stay absent rather than being upgraded on
    the strength of the operation having run.
    """
    observed_channels = {c.get("channel") for c in evaluation.changes
                         if c.get("eligibility") == "evaluable"}
    out: Dict[str, Any] = {}
    for name in ("m", "w_or", "pi", "r", "t"):
        previous = prediction.current_evidence.sources.get(name)
        prior_status = (previous.status if previous is not None
                        else "no_evidence")
        out[name] = {
            "previous_status": prior_status,
            "status": prior_status,
            "note": ("no real later-task observation speaks to this source: "
                     "its evidence strength is unchanged by the operation "
                     "having run"),
        }
    # M (experience/knowledge material) is supported by a real quality or
    # completion observation on later tasks.
    if observed_channels & {"normalized_solution_quality",
                            "effective_completion_rate"}:
        out["m"] = {
            "previous_status": out["m"]["previous_status"],
            "status": "direct_evidence",
            "note": ("a real later-task quality/completion observation "
                     "exists: it speaks to the experience/knowledge source"),
        }
    # T (tool/solver use) is supported by a real cost observation.
    if "resource_cost" in observed_channels:
        out["t"] = {
            "previous_status": out["t"]["previous_status"],
            "status": "direct_evidence",
            "note": ("a real later-task cost observation exists: it speaks "
                     "to how the tools were used"),
        }
    # W_OR is NEVER upgraded here: OR-consequence-prediction accuracy
    # comes from independent strategy-outcome evaluation evidence, and a
    # capability prediction asserting its own improvement proves nothing.
    out["w_or"]["note"] = (
        "W_OR is not advanced by a capability-effect observation: its "
        "evidence must come from independent OR strategy-outcome "
        "prediction-error records, and the capability predictor's own "
        "report about itself proves nothing")
    return out


# ---------------------------------------------------------------------------
# 4. persistence for both stages
# ---------------------------------------------------------------------------


def _put_record(harness, key: str, payload: Dict[str, Any]) -> None:
    with harness.store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, harness.store.dumps(payload)))


def _get_record(harness, key: str) -> Optional[Dict[str, Any]]:
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    return harness.store.loads(row["value"])


def record_maintenance_binding(harness, binding: MaintenanceFactBinding
                               ) -> None:
    """Persist one stage-1 binding (idempotent by prediction id).

    Keyed by the PREDICTION, not by a fresh id: binding the same prediction
    twice returns the same stored fact and counts nothing twice.
    """
    _put_record(harness, f"capability_binding|{binding.prediction_id}",
                binding.to_dict())


def get_maintenance_binding(harness, prediction_id: str
                            ) -> Optional[MaintenanceFactBinding]:
    """The stored stage-1 binding of one prediction, or None."""
    data = _get_record(harness,
                       f"capability_binding|{prediction_id}")
    if data is None:
        return None
    return MaintenanceFactBinding.from_dict(data)


def record_effect_evaluation(harness,
                             evaluation: CapabilityEffectEvaluation
                             ) -> None:
    """Persist one stage-2 evaluation (idempotent by prediction id).

    Re-evaluating the same prediction REPLACES its evaluation rather than
    appending a second one: a repeated look must never double the sample.
    """
    _put_record(harness, f"capability_effect|{evaluation.prediction_id}",
                evaluation.to_dict())


def get_effect_evaluation(harness, prediction_id: str
                          ) -> Optional[CapabilityEffectEvaluation]:
    """The stored stage-2 evaluation of one prediction, or None."""
    data = _get_record(harness, f"capability_effect|{prediction_id}")
    if data is None:
        return None
    return CapabilityEffectEvaluation.from_dict(data)


def record_paired_evaluation(harness, prediction_id: str, *,
                             metric: str, reference_value: float,
                             treated_value: float, unit: str = "",
                             source: str = "external_paired_evaluation",
                             reference_task_ids: Optional[Sequence[str]] =
                             None,
                             note: str = "",
                             ) -> Dict[str, Any]:
    """Record a pre-arranged PAIRED evaluation (the causal reference).

    The framework does not clone the harness or run counterfactual
    operations; it accepts a paired comparison the caller really arranged
    (a control group, an A/B window, an external benchmark). The record
    names its own source and the tasks it covers, so a later reader can
    tell a real comparison from an assertion.
    """
    if not _finite(reference_value) or not _finite(treated_value):
        raise ValueError("paired values must be finite numbers")
    if _observable_metric(metric) is None:
        raise ValueError(
            f"metric {metric!r} has no observation channel in this build; "
            f"expected one of {OBSERVABLE_CAPABILITY_METRICS}")
    record = {
        "prediction_id": prediction_id,
        "metric": metric,
        "unit": unit,
        "reference_value": float(reference_value),
        "treated_value": float(treated_value),
        "change": round(float(treated_value) - float(reference_value), 6),
        "source": str(source),
        "reference_task_ids": [str(t) for t in
                               (reference_task_ids or [])],
        "note": str(note),
        "created_at": time.time(),
        "discipline": (
            "an EXTERNAL paired comparison the caller really arranged; the "
            "framework does not fabricate a counterfactual"),
    }
    _put_record(harness, f"capability_paired_reference|{prediction_id}",
                record)
    return record


def get_paired_evaluation(harness, prediction_id: str
                          ) -> Optional[Dict[str, Any]]:
    """The stored paired reference of one prediction, or None."""
    return _get_record(harness,
                       f"capability_paired_reference|{prediction_id}")


def capability_evidence_from_effects(harness, *,
                                     base_evidence:
                                     Optional[HarnessCapabilityEvidence] =
                                     None,
                                     task: Optional[Dict[str, Any]] = None,
                                     snapshot: Any = None
                                     ) -> HarnessCapabilityEvidence:
    """Build capability evidence that includes VERIFIED effect results.

    Only an evaluation whose state is ``observed_improvement`` contributes
    ``direct_evidence`` for the sources it really observed. Everything else
    keeps its previous strength — a pending horizon, an inconclusive result
    and a refuted prediction all leave the evidence where it was. This is
    how a LATER capability prediction conditions on what was really
    verified, instead of on what was merely asserted.
    """
    if base_evidence is not None:
        evidence = copy.deepcopy(base_evidence)
    else:
        evidence = harness.capability_evidence(task, snapshot=snapshot)
    verified: List[str] = []
    for prediction in harness.capability_predictions.query():
        evaluation = get_effect_evaluation(harness,
                                           prediction.prediction_id)
        if evaluation is None or not evaluation.effect_verified:
            continue
        verified.append(prediction.prediction_id)
        for name, item in (evaluation.source_evidence or {}).items():
            if item.get("status") != "direct_evidence":
                continue
            source = evidence.source(name)
            source.status = "direct_evidence"
            source.evidence.append(
                __import__("or_harness.world_model.contracts",
                           fromlist=["EvidenceRef"]).EvidenceRef(
                    ref_type="capability_effect",
                    ref_id=evaluation.evaluation_id,
                    note=("a real later-task observation after an offline "
                          "operation")))
            source.notes.append(
                f"advanced by verified effect evaluation "
                f"{evaluation.evaluation_id}")
    evidence.notes.append(
        f"built from {len(verified)} verified effect evaluation(s); every "
        "other source keeps its previous strength, and W_OR is never "
        "advanced by a capability prediction's own report")
    # Stated UNCONDITIONALLY, so a reader never has to infer the rule from
    # an absence: no capability-effect evaluation can advance W_OR, whether
    # or not anything was verified.
    w_or = evidence.source("w_or")
    w_or.notes.append(
        "W_OR is not advanced by a capability-effect observation: its "
        "evidence must come from independent OR strategy-outcome "
        "prediction-error records, and the capability predictor's own "
        "report about itself proves nothing")
    return evidence


def capability_effect_summary(harness) -> Dict[str, Any]:
    """Every capability prediction's two-stage feedback state.

    Read-only: no model call, no re-evaluation, no re-billing.
    """
    out: List[Dict[str, Any]] = []
    for prediction in harness.capability_predictions.query():
        binding = get_maintenance_binding(harness, prediction.prediction_id)
        evaluation = get_effect_evaluation(harness,
                                           prediction.prediction_id)
        paired = get_paired_evaluation(harness, prediction.prediction_id)
        out.append({
            "prediction_id": prediction.prediction_id,
            "status": prediction.status,
            "operation_type": prediction.candidate_operation.operation_type,
            "horizon": prediction.horizon,
            "fact_bound": binding is not None,
            "changed": (binding.changed if binding is not None else None),
            "effect_state": (evaluation.state if evaluation is not None
                             else "pending"),
            "effect_verified": bool(evaluation is not None
                                    and evaluation.effect_verified),
            "paired_reference": bool(paired),
        })
    return {
        "effect_version": CAPABILITY_EFFECT_VERSION,
        "n_predictions": len(out),
        "n_fact_bound": sum(1 for e in out if e["fact_bound"]),
        "n_effect_verified": sum(1 for e in out if e["effect_verified"]),
        "predictions": out,
        "note": ("facts and effects are separate stages: a bound fact says "
                 "the operation happened, an effect verdict says real later "
                 "performance moved — only the latter supports a claim that "
                 "the harness got stronger"),
    }
