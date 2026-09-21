"""Training-free harness-capability evolution prediction (world-model M5).

This module implements the CAPABILITY EVOLUTION prediction SERVICE — the
slow-time-scale counterpart of the strategy-outcome service (wm-so/1). It
takes FROZEN evidence about the harness's capability state H, a candidate
OFFLINE learning operation, the experience scope it would consume, the task
types the claim is about, a frozen baseline and a horizon, asks the
configured provider under the ``wm-ce/1`` protocol what would change, and
persists the answer so it can later be bound to the real maintenance fact
and, after qualified later tasks, evaluated against real performance.

Design boundaries (the reason this is a separate module):

- **One protocol, one prompt.** The framework fixes the operation, the
  experience scope, the target task types, the baseline and the horizon;
  the model fills ONLY the prediction content. A payload that tries to
  restate the operation, the baseline or ``effect_verified`` has those
  fields IGNORED and the attempt RECORDED — the identity is not negotiable.
- **The provider receives CONTENT, not ids.** The real capability evidence
  (with its per-source detail), the real execution/knowledge material of
  the scope and the target structure travel in the request, so the model
  conditions on what the harness actually gathered.
- **H is judged through observable consequences.** Every prediction must
  carry at least one :class:`ExpectedChange` with a metric, a unit, a
  beneficial direction and a baseline. "All unknown" is a recorded
  NON-prediction, never a content-ful capability forecast.
- **Honest failure states.** Not configured, malformed JSON, NaN/Infinity,
  out-of-range probabilities, unknown cost dimensions, wrong nested types
  and provider errors each produce a distinguishable result. A failed call
  keeps whatever usage it consumed.
- **Predicting, binding and verifying are three different things.** This
  module only ever produces the FIRST. Nothing here can set
  ``effect_verified``; that is advanced by the framework from real facts.
- **Legacy untouched.** Nothing here renames an old induction assessment or
  wraps a legacy knowledge-gain score as an H improvement.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.schema import (
    is_finite_number as _finite,
    is_probability as _prob,
)
from or_harness.world_model.contracts import (
    BaselineStatement,
    CapabilityEvolutionPrediction,
    EvidenceRef,
    ExpectedChange,
    ExpectedCost,
    ExperienceScope,
    HarnessCapabilityEvidence,
    LearningOperation,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    TaskTargeting,
    UncertaintyStatement,
    VerificationCondition,
    validate_capability_evolution,
)

#: Version of this prediction protocol (request schema + output schema).
CAPABILITY_EVOLUTION_PROTOCOL_VERSION = "wm-ce/1"

#: The request key that selects this protocol on the provider call.
PROTOCOL_REQUEST_KEY = "prediction_protocol"

#: The system prompt for the capability-evolution protocol. The framework
#: supplies the operation, the scope, the target, the baseline and the
#: horizon; the model fills only the predicted consequences.
CAPABILITY_EVOLUTION_SYSTEM_PROMPT = (
    "You are a world model for an operations-research harness, answering "
    "under the wm-ce/1 HARNESS-CAPABILITY-EVOLUTION prediction protocol. "
    "You are given FROZEN evidence about the harness's current capability "
    "state (five interacting sources: M experience/knowledge material, "
    "W_OR OR-consequence-prediction ability, Pi strategy generation and "
    "selection, R knowledge retrieval and transfer, T tool/solver "
    "selection), the REAL experience and knowledge content a candidate "
    "OFFLINE learning operation would consume, the exact experience scope, "
    "the task types the claim is about, a frozen baseline and a horizon.\n"
    "OUTPUT RULES (read first, follow strictly):\n"
    "1. Respond with EXACTLY ONE raw JSON object and NOTHING else. No "
    "markdown, no code fences, no explanation outside the JSON.\n"
    "2. You predict how FUTURE task performance would change IF the "
    "operation were executed. It has NOT been executed. Do NOT claim any "
    "effect was verified, and do NOT output a status field.\n"
    "3. Every field is OPTIONAL. If you lack evidence for a field, OMIT it "
    "entirely — never write null, \"unknown\", -1 or a guess as a "
    "placeholder. An omitted field is an honest unknown. Do NOT fill a "
    "field with 0 to mean \"no change\" unless you really predict no "
    "change.\n"
    "4. All numbers must be valid JSON numbers (not strings, never NaN or "
    "Infinity).\n"
    "5. You may NOT restate or modify the operation, the experience scope, "
    "the task targeting, the baseline, the horizon or the verification "
    "status: they are FIXED in the request and anything you send for them "
    "is ignored and recorded.\n"
    "6. More knowledge is NOT more capability. Adding entries, accumulating "
    "evidence, or your own assertion of improvement proves nothing: only "
    "an observable consequence on future tasks counts.\n"
    "Fields (all optional):\n"
    "- expected_changes: list of objects. {\"metric\": short name of WHAT "
    "changes (required), \"unit\": the unit of the metric, \"direction\": "
    "one of increase|decrease|unchanged|unknown, \"value\": number — the "
    "SIGNED change `after - baseline` in the metric's own unit (a shorter "
    "runtime is a NEGATIVE value), \"value_kind\": absolute|relative (a "
    "relative value is a RATIO: express 20% as 0.2), "
    "\"beneficial_direction\": increase|decrease|either — which direction "
    "is the IMPROVEMENT for this metric (required for us to interpret a "
    "signed value; a lower completion rate is not an improvement), "
    "\"interval\": [lo, hi], \"baseline\": {\"kind\": one of "
    "no_knowledge|conditional_stats|current_entry|current_solution|"
    "declared|unknown, \"value\": number, \"note\": text}, \"notes\": "
    "[text]}. Predict only consequences you have evidence for; if you "
    "cannot say which way a metric moves, omit that change entirely "
    "rather than emitting direction=unknown.\n"
    "- learning_cost: object of the resource dimensions the OFFLINE "
    "operation itself would consume, each a non-negative number: "
    "{llm_tokens, tool_calls, solver_runtime_s, retries, latency_s}. This "
    "is the cost of LEARNING, not the cost of future solving and not the "
    "cost of this call. Include ONLY dimensions you predict.\n"
    "- degradation_risk: object {\"events\": [{\"event\": a short name of "
    "the RISK (e.g. overgeneralized_entry, revised_performance_regression, "
    "applicability_overreach), \"probability\": number in [0,1] or omitted "
    "when you have no basis, \"severity\": number or omitted, "
    "\"severity_unit\": text, \"basis\": text, \"notes\": [text]}]}. A "
    "risk whose probability you cannot estimate must OMIT the probability "
    "— never default it to 0.\n"
    "- uncertainty: object {\"knowledge_gap\": number in [0,1] — how "
    "little evidence this prediction rests on, \"execution_randomness\": "
    "number in [0,1] — variance of the future tasks themselves, "
    "\"basis\": [text], \"missing\": {field: reason}, \"notes\": [text]}. "
    "Separate the two: a gap in the EVIDENCE is not the same as the "
    "randomness of the future. These are your own UNCALIBRATED estimates "
    "and are recorded as such.\n"
    "- verification_conditions: list of objects {\"condition\": text — "
    "what real later evidence would confirm or refute this prediction, "
    "\"check_basis\": text — the metric and comparison to use, "
    "\"evaluable\": boolean}. Describe what WOULD confirm it; you may not "
    "state that anything already has.\n"
    "- evidence_basis: list of strings naming the evidence keys you relied "
    "on (e.g. \"capability_evidence.sources.m\", "
    "\"learning_material.executions[0]\").\n"
    "- unsupported_fields: object {field: reason} for fields you cannot or "
    "will not predict.\n"
    "Do NOT fabricate evidence. If the material contains no relevant "
    "experience for the operation, say so in unsupported_fields and leave "
    "the numeric fields omitted. Output the JSON object only."
)

#: Payload keys the model must never be able to set (they are fixed by the
#: framework). Sending one is recorded and ignored — never applied.
FORBIDDEN_PAYLOAD_KEYS = (
    "operation", "candidate_operation", "learning_operation",
    "experience_scope", "scope", "task_targeting", "targeting",
    "baseline", "horizon", "horizon_tasks", "status", "effect_verified",
    "fact_bound", "prediction_made", "service_available",
    "service_implemented", "current_evidence", "capability_evidence",
)

#: The maximum number of expected changes accepted from one payload.
MAX_EXPECTED_CHANGES = 12

#: The maximum number of degradation-risk events accepted.
MAX_DEGRADATION_EVENTS = 20

#: The maximum number of evidence-basis entries accepted.
MAX_EVIDENCE_BASIS = 40

#: The maximum number of verification conditions accepted.
MAX_VERIFICATION_CONDITIONS = 12


def build_capability_evolution_request(
        evidence: HarnessCapabilityEvidence,
        operation: LearningOperation, *,
        experience_scope: Optional[ExperienceScope] = None,
        task_targeting: Optional[TaskTargeting] = None,
        baseline: Optional[BaselineStatement] = None,
        horizon: str = "",
        horizon_tasks: Optional[int] = None,
        learning_material: Optional[Dict[str, Any]] = None,
        maintenance_budget: Optional[Dict[str, float]] = None,
        baselines_by_metric: Optional[Dict[str, BaselineStatement]] = None,
        ) -> Dict[str, Any]:
    """Assemble the provider request for ONE candidate learning operation.

    The framework fixes the identity — the operation, the experience scope,
    the task targeting, the baseline, the horizon — and supplies the real
    CONTENT (capability evidence with its per-source detail, the execution
    and knowledge material of the scope). The model fills only the
    predicted consequences, and the request says so explicitly.

    ``baselines_by_metric`` carries the FRAMEWORK-FROZEN reference for each
    metric. The model sees them so it can reason against a real yardstick,
    but it may only CITE one: the frozen value is what the parsed prediction
    keeps, so a model cannot move the reference after seeing the evidence.
    """
    request: Dict[str, Any] = {
        PROTOCOL_REQUEST_KEY: CAPABILITY_EVOLUTION_PROTOCOL_VERSION,
        "request_kind": "capability_evolution_prediction",
        "capability_evidence": evidence.to_dict(),
        "candidate_operation": operation.to_dict(),
        "experience_scope": (experience_scope.to_dict()
                             if experience_scope is not None else None),
        "task_targeting": (task_targeting.to_dict()
                           if task_targeting is not None else None),
        "baseline": (baseline.to_dict() if baseline is not None else None),
        "baselines_by_metric": {
            k: v.to_dict() for k, v in
            (baselines_by_metric or {}).items()},
        "horizon": horizon,
        "horizon_tasks": horizon_tasks,
        "output_contract": {
            "allowed_fields": ["expected_changes", "learning_cost",
                               "degradation_risk", "uncertainty",
                               "verification_conditions", "evidence_basis",
                               "unsupported_fields"],
            "fixed_by_framework": ["candidate_operation", "experience_scope",
                                   "task_targeting", "baseline",
                                   "baselines_by_metric", "horizon",
                                   "verification status"],
            "forbidden_fields": list(FORBIDDEN_PAYLOAD_KEYS),
            "note": ("the model fills prediction content only; it may not "
                     "rewrite the operation, the scope, the target, the "
                     "baseline, the horizon, the evidence sources, or "
                     "claim that an effect is verified. A per-metric "
                     "baseline may be CITED, never set"),
        },
    }
    if learning_material:
        request["learning_material"] = copy.deepcopy(learning_material)
    if maintenance_budget:
        request["maintenance_budget"] = copy.deepcopy(maintenance_budget)
    return request


def learning_material_for_bundle(harness, bundle: Any
                                 ) -> Dict[str, Any]:
    """The REAL content a candidate bundle would consolidate.

    The provider must condition on actual evidence, not on ids: this reads
    the scope's execution records (quality, cost, status, failure classes)
    and, for a revision, the target entry's content as it stood BEFORE the
    operation. Nothing here is a summary the model could not verify
    against the request.
    """
    material: Dict[str, Any] = {
        "executions": [],
        "tasks": [],
        "note": ("the real evidence content the candidate operation would "
                 "consolidate; ids alone are not material"),
    }
    if bundle is None:
        material["note"] += " (no candidate bundle was supplied)"
        return material
    data = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
    material["candidate_kind"] = data.get("kind")
    material["strategy_id"] = data.get("strategy_id")
    material["family"] = data.get("family")
    material["cell_token"] = data.get("cell_token")
    material["n_supporting"] = data.get("n_supporting")
    material["distinct_tasks"] = len(set(data.get("tasks") or []))
    material["trigger_reasons"] = list(data.get("trigger_reasons") or [])
    material["mean_quality"] = data.get("mean_quality")
    material["mean_cost"] = copy.deepcopy(data.get("mean_cost") or {})
    material["cost_measured"] = list(data.get("cost_measured") or [])
    material["failure_rate"] = data.get("failure_rate")
    material["tasks"] = list(data.get("tasks") or [])
    for execution_id in (data.get("execution_ids") or []):
        record = (harness.bank.get_pending(execution_id)
                  or harness.bank.get(execution_id))
        if record is None:
            material["executions"].append({
                "execution_id": execution_id,
                "available": False,
                "reason": "not found in the Experience Bank",
            })
            continue
        quality = record.quality or {}
        material["executions"].append({
            "execution_id": execution_id,
            "task_id": record.task_id,
            "strategy_id": record.strategy_id,
            "available": True,
            "status": quality.get("status"),
            "objective_value": quality.get("objective_value"),
            "objective_bound": quality.get("objective_bound"),
            "gap": quality.get("gap"),
            "feasible": quality.get("feasible"),
            "cost": record.cost.to_dict(),
            "cost_measured": sorted(record.cost.measured_dims()),
            "failure_classes": [str(getattr(f, "error_class", None) or
                                    "model") for f in (record.failures or [])],
            "measurement_scope": record.measurement_scope,
        })
    entry_before = data.get("entry_before")
    if entry_before:
        material["entry_before"] = {
            "entry_id": entry_before.get("entry_id"),
            "status": entry_before.get("status"),
            "expected_quality_hat": entry_before.get("expected_quality_hat"),
            "support_n": entry_before.get("support_n"),
            "applicability": copy.deepcopy(
                entry_before.get("applicability") or {}),
            "actions": list(entry_before.get("actions") or []),
            "note": ("the existing entry's content as it stood BEFORE the "
                     "operation: a revision is judged against this state, "
                     "not against the entry's later value"),
        }
    return material


def _frozen_baseline_for(
        baselines_by_metric: Optional[Dict[str, BaselineStatement]],
        metric: str, unit: str) -> Optional[BaselineStatement]:
    """The frozen reference for one metric and unit, or None.

    Several cost dimensions share the ``resource_cost`` metric but their
    units do not convert, so a cost reference is looked up by the change's
    UNIT. A cost change that names no unit therefore gets NO reference: it
    is genuinely ambiguous which dimension it is about, and silently
    handing it one dimension's yardstick is exactly the unit-mixing this
    lookup exists to prevent. The metric's own (non-cost) key is used as
    the fallback.
    """
    mapping = baselines_by_metric or {}
    wanted = str(unit or "").strip().lower()
    if wanted:
        for key, statement in mapping.items():
            if not str(key).startswith("cost:"):
                continue
            if str(statement.unit or "").strip().lower() == wanted:
                return statement
        return None
    return mapping.get(metric)


def parse_capability_evolution_payload(
        payload: Any, *,
        evidence: HarnessCapabilityEvidence,
        operation: LearningOperation,
        experience_scope: Optional[ExperienceScope] = None,
        task_targeting: Optional[TaskTargeting] = None,
        baseline: Optional[BaselineStatement] = None,
        baselines_by_metric: Optional[Dict[str, BaselineStatement]] = None,
        horizon: str = "",
        horizon_tasks: Optional[int] = None,
        provider_result: Optional[Dict[str, Any]] = None,
        prediction_id: Optional[str] = None,
        ) -> CapabilityEvolutionPrediction:
    """Parse a model payload into a validated capability prediction.

    The operation, the scope, the targeting, the baseline and the horizon
    come from the FRAMEWORK arguments, never from the payload: a payload
    that tries to restate them has those keys ignored and the attempt
    recorded on the trace. Every malformed value becomes an explicit
    problem. A payload with no usable expected change is a recorded
    NON-prediction (``contract_only``), never a content-ful forecast.

    A per-change ``baseline`` is the framework's too: the model may CITE a
    frozen reference (the parsed change keeps the frozen value), and a
    reference it invents is recorded as an override and replaced. A model
    that could choose its own yardstick after seeing the evidence would be
    grading its own work.
    """
    problems: List[str] = []
    notes: List[str] = []
    unsupported: Dict[str, str] = {}
    overrides: Dict[str, Any] = {}

    if not isinstance(payload, dict):
        payload = {}
        problems.append("payload is not a JSON object")

    # -- identity: the framework's values win, attempts are recorded -------
    for key in FORBIDDEN_PAYLOAD_KEYS:
        if key in payload:
            overrides[key] = payload[key]
    if overrides:
        notes.append(
            "the model tried to restate framework-fixed fields "
            f"({', '.join(sorted(overrides))}): the attempt is recorded and "
            "IGNORED — the operation, scope, target, baseline and horizon "
            "are fixed by the framework, and verification status is never "
            "the model's to set")

    # -- expected changes --------------------------------------------------
    changes: List[ExpectedChange] = []
    raw_changes = payload.get("expected_changes")
    if raw_changes is None:
        unsupported["expected_changes"] = "not predicted by the model"
    elif not isinstance(raw_changes, list):
        problems.append("expected_changes must be a list")
    else:
        if len(raw_changes) > MAX_EXPECTED_CHANGES:
            problems.append(
                f"expected_changes is bounded at {MAX_EXPECTED_CHANGES}; "
                f"{len(raw_changes)} were supplied")
            raw_changes = raw_changes[:MAX_EXPECTED_CHANGES]
        for index, raw in enumerate(raw_changes):
            if not isinstance(raw, dict):
                problems.append(f"expected_changes[{index}] must be a JSON "
                                "object")
                continue
            metric = str(raw.get("metric") or "")
            if not metric:
                problems.append(
                    f"expected_changes[{index}].metric is required (what "
                    "changes)")
                continue
            direction = str(raw.get("direction") or "unknown")
            value = raw.get("value")
            if value is not None and not _finite(value):
                problems.append(
                    f"expected_changes[{index}].value is not a finite number")
                value = None
            if value is not None and direction == "unknown":
                problems.append(
                    f"expected_changes[{index}]: a value with "
                    "direction='unknown' is contradictory — state the "
                    "direction or omit the value")
                value = None
            # Beneficial direction: without it a signed change cannot be
            # read as an improvement. Omitting it does NOT default to
            # "increase" — that would hand the operation a free benefit.
            raw_beneficial = raw.get("beneficial_direction")
            if raw_beneficial is None:
                beneficial = "either"
                if value is not None or direction != "unknown":
                    unsupported[f"expected_changes[{index}]"
                                ".beneficial_direction"] = (
                        "not stated: which direction is the improvement is "
                        "unknown, so this change is reported but cannot "
                        "count as a benefit or a degradation")
            elif str(raw_beneficial) in ("increase", "decrease", "either"):
                beneficial = str(raw_beneficial)
            else:
                problems.append(
                    f"expected_changes[{index}].beneficial_direction must be "
                    "increase/decrease/either")
                beneficial = "either"
            value_kind = str(raw.get("value_kind") or "absolute")
            if value_kind not in ("absolute", "relative"):
                problems.append(
                    f"expected_changes[{index}].value_kind must be "
                    "absolute/relative")
                value_kind = "absolute"
            if value_kind == "relative" and value is not None \
                    and not -1.0 <= float(value) <= 1.0:
                problems.append(
                    f"expected_changes[{index}].value_kind='relative' "
                    "requires a RATIO in [-1, 1] (express 20% as 0.2)")
                value = None
            interval = None
            raw_interval = raw.get("interval")
            if isinstance(raw_interval, (list, tuple)) \
                    and len(raw_interval) == 2:
                lo, hi = raw_interval
                if _finite(lo) and _finite(hi) and float(lo) <= float(hi):
                    interval = (float(lo), float(hi))
                else:
                    problems.append(
                        f"expected_changes[{index}].interval must be finite "
                        "with lo<=hi")
            change_baseline = None
            raw_baseline = raw.get("baseline")
            if isinstance(raw_baseline, dict):
                try:
                    change_baseline = BaselineStatement.from_dict(
                        raw_baseline)
                except ValueError as exc:
                    problems.append(
                        f"expected_changes[{index}].baseline: {exc}")
                    change_baseline = None
            elif value is not None and baseline is None \
                    and not (baselines_by_metric or {}).get(metric):
                problems.append(
                    f"expected_changes[{index}].value requires a baseline "
                    "(its own or the prediction's): a change with nothing "
                    "to measure it against is not falsifiable")
            change_unit = str(raw.get("unit") or "")
            # The reference belongs to the FRAMEWORK. When one is frozen for
            # this metric — and, for a cost change, for this UNIT — the
            # frozen statement WINS: a model may cite it, never restate it.
            # An invented (or mismatched) reference is recorded as an
            # override attempt and replaced, so the prediction never keeps a
            # self-chosen yardstick. Matching on the unit matters because
            # several cost dimensions share the ``resource_cost`` metric
            # while their units do not convert.
            frozen = _frozen_baseline_for(baselines_by_metric, metric,
                                          change_unit)
            if frozen is None and baseline is not None \
                    and baseline.metric in (None, metric) \
                    and (not change_unit or not baseline.unit
                         or str(baseline.unit).strip().lower()
                         == change_unit.strip().lower()):
                frozen = baseline
            if frozen is not None:
                if change_baseline is not None and (
                        (change_baseline.value is not None
                         and frozen.value is not None
                         and abs(float(change_baseline.value)
                                 - float(frozen.value)) > 1e-9)
                        or (change_baseline.kind != frozen.kind
                            and change_baseline.value is not None)):
                    overrides[f"expected_changes[{index}].baseline"] = (
                        change_baseline.to_dict())
                    notes.append(
                        f"expected_changes[{index}] declared its own "
                        f"baseline ({change_baseline.value}) for metric "
                        f"{metric!r}"
                        + (f" in unit {change_unit!r}" if change_unit else "")
                        + f", but the framework froze {frozen.value}: the "
                        "model's value is recorded and IGNORED — a baseline "
                        "is fixed before the operation runs, so it is never "
                        "the model's to move")
                change_baseline = frozen
            try:
                changes.append(ExpectedChange(
                    metric=metric,
                    direction=direction,
                    value=(float(value) if value is not None else None),
                    interval=interval,
                    unit=change_unit,
                    baseline=change_baseline,
                    beneficial_direction=beneficial,
                    value_kind=value_kind,
                    notes=[str(n) for n in (raw.get("notes") or [])]))
            except ValueError as exc:
                problems.append(f"expected_changes[{index}]: {exc}")

    # -- learning cost -----------------------------------------------------
    learning_cost: Optional[ExpectedCost] = None
    raw_cost = payload.get("learning_cost")
    if raw_cost is None:
        unsupported["learning_cost"] = "not predicted by the model"
    elif not isinstance(raw_cost, dict):
        problems.append("learning_cost must be a JSON object of dimensions")
    else:
        dims: Dict[str, float] = {}
        for dim, value in raw_cost.items():
            if dim not in COST_DIMENSIONS:
                problems.append(
                    f"unknown learning_cost dimension {dim!r}: a resource "
                    f"dimension must be one of {COST_DIMENSIONS}")
            elif not _finite(value) or float(value) < 0:
                problems.append(
                    f"learning_cost.{dim} must be finite and >= 0")
            else:
                dims[dim] = float(value)
        if dims:
            vector = CostVector(**dims, measured=set(dims))
            learning_cost = ExpectedCost(
                expected=vector, measured=sorted(dims),
                notes=["predicted cost of the OFFLINE learning operation; "
                       "kept separate from this prediction call's own real "
                       "spend and from any future solving cost"])
        elif not problems:
            unsupported["learning_cost"] = ("no recognized resource "
                                            "dimension was predicted")

    # -- degradation risk ---------------------------------------------------
    degradation_risk: Optional[RiskStatement] = None
    raw_risk = payload.get("degradation_risk")
    if raw_risk is None:
        unsupported["degradation_risk"] = "not predicted by the model"
    elif not isinstance(raw_risk, dict):
        problems.append(
            "degradation_risk must be a JSON object with an events list")
    else:
        events: List[RiskEvent] = []
        raw_events = raw_risk.get("events")
        if raw_events is None:
            unsupported["degradation_risk"] = "no risk event was predicted"
        elif not isinstance(raw_events, list):
            problems.append("degradation_risk.events must be a list")
        else:
            if len(raw_events) > MAX_DEGRADATION_EVENTS:
                problems.append(
                    f"degradation_risk.events is bounded at "
                    f"{MAX_DEGRADATION_EVENTS}; {len(raw_events)} were "
                    "supplied")
                raw_events = raw_events[:MAX_DEGRADATION_EVENTS]
            for index, raw in enumerate(raw_events):
                if not isinstance(raw, dict) or not raw.get("event"):
                    problems.append(
                        f"degradation_risk.events[{index}].event is required")
                    continue
                probability = raw.get("probability")
                if probability is not None and not _prob(probability):
                    problems.append(
                        f"degradation_risk.events[{index}].probability must "
                        "be in [0, 1]")
                    probability = None
                severity = raw.get("severity")
                if severity is not None and not _finite(severity):
                    problems.append(
                        f"degradation_risk.events[{index}].severity must be "
                        "finite")
                    severity = None
                events.append(RiskEvent(
                    event=str(raw["event"]),
                    probability=(float(probability)
                                 if probability is not None else None),
                    severity=(float(severity)
                              if severity is not None else None),
                    severity_unit=str(raw.get("severity_unit") or ""),
                    basis=(str(raw["basis"]) if raw.get("basis") else None),
                    notes=[str(n) for n in (raw.get("notes") or [])]))
            if events:
                degradation_risk = RiskStatement(
                    events=events,
                    notes=[str(n) for n in (raw_risk.get("notes") or [])])

    # -- uncertainty --------------------------------------------------------
    uncertainty: Optional[UncertaintyStatement] = None
    raw_unc = payload.get("uncertainty")
    if raw_unc is None:
        unsupported["uncertainty"] = "not predicted by the model"
    elif not isinstance(raw_unc, dict):
        problems.append("uncertainty must be a JSON object")
    else:
        components: Dict[str, Optional[float]] = {}
        for name in ("execution_randomness", "knowledge_gap"):
            value = raw_unc.get(name)
            if value is None:
                continue
            if not _prob(value):
                problems.append(f"uncertainty.{name} must be in [0, 1]")
                continue
            components[name] = float(value)
        if components:
            # The model's own numbers are its UNCALIBRATED statement: they
            # are recorded in the notes (the contract validator refuses
            # numeric components presented as calibrated), with the source
            # labelled model_self_report.
            uncertainty = UncertaintyStatement(
                source="model_self_report",
                basis=[str(b) for b in (raw_unc.get("basis") or [])],
                missing={str(k): str(v) for k, v in
                         (raw_unc.get("missing") or {}).items()},
                notes=[str(n) for n in (raw_unc.get("notes") or [])]
                + [f"knowledge_gap (self-reported, uncalibrated): "
                   f"{components.get('knowledge_gap')}"
                   if "knowledge_gap" in components else ""]
                + [f"execution_randomness (self-reported, uncalibrated): "
                   f"{components.get('execution_randomness')}"
                   if "execution_randomness" in components else ""]
                + ["the model's self-reported uncertainty is recorded as an "
                   "UNCALIBRATED estimate, never as a measured probability"])
            uncertainty.notes = [n for n in uncertainty.notes if n]
        elif not problems:
            unsupported["uncertainty"] = ("no uncertainty component was "
                                          "predicted")

    # -- verification conditions -------------------------------------------
    conditions: List[VerificationCondition] = []
    raw_conditions = payload.get("verification_conditions")
    if raw_conditions is None:
        unsupported["verification_conditions"] = (
            "not proposed by the model")
    elif not isinstance(raw_conditions, list):
        problems.append("verification_conditions must be a list")
    else:
        for index, raw in enumerate(raw_conditions[:MAX_VERIFICATION_CONDITIONS]):
            if not isinstance(raw, dict) or not raw.get("condition"):
                problems.append(
                    f"verification_conditions[{index}].condition is required")
                continue
            # The model may DESCRIBE what would confirm the prediction; it
            # may never assert that anything already has. Those keys are
            # forced to the only honest values here.
            if raw.get("effect_verified") or raw.get("fact_bound"):
                notes.append(
                    f"verification_conditions[{index}] claimed a "
                    "verification status: forced back to "
                    "prediction_made-only — the framework advances "
                    "fact_bound and effect_verified from real evidence")
            conditions.append(VerificationCondition(
                condition=str(raw["condition"]),
                evaluable=bool(raw.get("evaluable", False)),
                check_basis=(str(raw["check_basis"])
                             if raw.get("check_basis") else None),
                prediction_made=True,
                fact_bound=False,
                effect_verified=False,
                notes=[str(n) for n in (raw.get("notes") or [])]))

    # -- evidence basis ------------------------------------------------------
    evidence_basis: List[EvidenceRef] = []
    raw_basis = payload.get("evidence_basis")
    if isinstance(raw_basis, list):
        if len(raw_basis) > MAX_EVIDENCE_BASIS:
            raw_basis = raw_basis[:MAX_EVIDENCE_BASIS]
        for item in raw_basis:
            evidence_basis.append(EvidenceRef(ref_type="capability_evidence",
                                              ref_id=str(item)))
    raw_unsupported = payload.get("unsupported_fields")
    if isinstance(raw_unsupported, dict):
        unsupported.update({str(k): str(v) for k, v in
                            raw_unsupported.items()})

    has_content = bool(changes) or learning_cost is not None \
        or degradation_risk is not None or uncertainty is not None
    # A capability forecast needs at least one OBSERVABLE consequence: a
    # cost/risk/uncertainty block with no expected change is not a claim
    # about future performance.
    has_change = bool(changes)

    trace = PredictionTrace(
        prediction_kind="capability_evolution",
        input_snapshot_id=str(getattr(evidence, "version", "") or ""),
        input_version=str(getattr(evidence, "version", "") or ""),
        prediction_version=CAPABILITY_EVOLUTION_PROTOCOL_VERSION,
        evidence_basis=evidence_basis,
        unsupported_fields=unsupported,
        comparable=False,
        not_comparable_reasons=[
            "the learning operation has not been executed and no later "
            "qualified task has run: nothing real exists to score against"],
    )
    if overrides:
        trace.model_info["attempted_field_overrides"] = copy.deepcopy(
            overrides)
    if provider_result is not None:
        usage = provider_result.get("usage") or {}
        tokens = usage.get("completion_tokens")
        latency = provider_result.get("latency_s")
        vector = CostVector(measured=set())
        if isinstance(tokens, (int, float)) and tokens >= 0:
            vector.llm_tokens = float(tokens)
            vector.mark_measured("llm_tokens")
        if latency is not None:
            vector.latency_s = float(latency)
            vector.mark_measured("latency_s")
        if vector.measured_dims():
            trace.call_cost = vector
        if latency is not None:
            trace.model_info["call_latency_s"] = round(float(latency), 4)

    if problems:
        status = "invalid"
        notes.extend(f"validation: {p}" for p in problems)
    elif not has_change:
        # A well-formed object with no usable expected change: the honest
        # "nothing was predicted" state. It is NEVER a content-ful
        # capability prediction, whatever else it carries.
        status = "contract_only"
        notes.append(
            "the model produced no usable expected_change: a capability "
            "forecast needs at least one observable consequence on future "
            "task performance, so this is a recorded non-prediction"
            + (" (other blocks were supplied but are not a performance "
               "claim)" if has_content else ""))
    else:
        status = "valid"

    prediction = CapabilityEvolutionPrediction(
        current_evidence=evidence,
        candidate_operation=operation,
        status=status,
        experience_scope=experience_scope,
        task_targeting=task_targeting,
        baseline=baseline,
        horizon=horizon,
        horizon_tasks=horizon_tasks,
        expected_changes=changes,
        learning_cost=learning_cost,
        degradation_risk=degradation_risk,
        uncertainty=uncertainty,
        verification_conditions=conditions,
        baselines_by_metric=dict(baselines_by_metric or {}),
        trace=trace,
        service_available=True,
        provider_configured=True,
        service_implemented=True,
        notes=notes,
    )
    if prediction_id:
        prediction.prediction_id = prediction_id
    if status == "valid":
        contract_problems = validate_capability_evolution(prediction)
        if contract_problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}"
                                    for p in contract_problems)
    return prediction


class CapabilityEvolutionService:
    """Predict / persist / read capability-evolution predictions (wm-ce/1)."""

    def __init__(self, store, provider):
        self.store = store
        self.provider = provider

    def predict(self, evidence: HarnessCapabilityEvidence,
                operation: LearningOperation, *,
                experience_scope: Optional[ExperienceScope] = None,
                task_targeting: Optional[TaskTargeting] = None,
                baseline: Optional[BaselineStatement] = None,
                baselines_by_metric: Optional[Dict[str, BaselineStatement]] =
                None,
                horizon: str = "",
                horizon_tasks: Optional[int] = None,
                learning_material: Optional[Dict[str, Any]] = None,
                maintenance_budget: Optional[Dict[str, float]] = None,
                timeout_s: Optional[float] = None,
                task_id: str = "",
                episode_id: Optional[str] = None,
                ) -> CapabilityEvolutionPrediction:
        """One provider call, parsed and persisted (failures included).

        The result is ALWAYS a persisted prediction object: a failed call is
        a recorded failure with whatever usage it consumed, never an
        exception the caller must catch to learn the model is down.
        """
        request = build_capability_evolution_request(
            evidence, operation, experience_scope=experience_scope,
            task_targeting=task_targeting, baseline=baseline,
            baselines_by_metric=baselines_by_metric,
            horizon=horizon, horizon_tasks=horizon_tasks,
            learning_material=learning_material,
            maintenance_budget=maintenance_budget)
        try:
            try:
                result = self.provider.predict(request, timeout_s=timeout_s)
            except TypeError:
                result = self.provider.predict(request)
        except Exception as exc:
            prediction = self._failed(
                evidence, operation, "provider_error",
                f"{type(exc).__name__}: {exc}",
                experience_scope=experience_scope,
                task_targeting=task_targeting, baseline=baseline,
                horizon=horizon, horizon_tasks=horizon_tasks)
            self._save(prediction, task_id=task_id, episode_id=episode_id)
            return prediction
        if result.get("not_configured"):
            prediction = self._failed(
                evidence, operation, "contract_only",
                str(result.get("error") or "provider not configured"),
                provider_configured=False, service_available=False,
                experience_scope=experience_scope,
                task_targeting=task_targeting, baseline=baseline,
                horizon=horizon, horizon_tasks=horizon_tasks)
            self._save(prediction, task_id=task_id, episode_id=episode_id)
            return prediction
        payload = result.get("payload")
        if payload is None:
            prediction = self._failed(
                evidence, operation, "invalid",
                str(result.get("error") or "no payload"),
                provider_result=result,
                experience_scope=experience_scope,
                task_targeting=task_targeting, baseline=baseline,
                horizon=horizon, horizon_tasks=horizon_tasks)
            self._save(prediction, task_id=task_id, episode_id=episode_id)
            return prediction
        try:
            prediction = parse_capability_evolution_payload(
                payload, evidence=evidence, operation=operation,
                experience_scope=experience_scope,
                task_targeting=task_targeting, baseline=baseline,
                baselines_by_metric=baselines_by_metric,
                horizon=horizon, horizon_tasks=horizon_tasks,
                provider_result=result)
        except Exception as exc:
            # A parse crash is THIS prediction's invalid result, never an
            # exception that escapes and kills the other candidates. The
            # call really happened, so its usage is kept.
            prediction = self._failed(
                evidence, operation, "invalid",
                f"payload could not be parsed: {type(exc).__name__}: {exc}",
                provider_result=result,
                experience_scope=experience_scope,
                task_targeting=task_targeting, baseline=baseline,
                horizon=horizon, horizon_tasks=horizon_tasks)
            self._save(prediction, task_id=task_id, episode_id=episode_id)
            return prediction
        if prediction.status == "invalid" \
                and not prediction.trace.model_info.get("parse_error"):
            prediction.trace.model_info["parse_error"] = (
                "the model's payload failed validation; see notes")
        self._save(prediction, task_id=task_id, episode_id=episode_id)
        return prediction

    def _failed(self, evidence: HarnessCapabilityEvidence,
                operation: LearningOperation, status: str, error: str,
                *, provider_configured: bool = True,
                service_available: bool = True,
                provider_result: Optional[Dict[str, Any]] = None,
                experience_scope: Optional[ExperienceScope] = None,
                task_targeting: Optional[TaskTargeting] = None,
                baseline: Optional[BaselineStatement] = None,
                horizon: str = "",
                horizon_tasks: Optional[int] = None,
                ) -> CapabilityEvolutionPrediction:
        trace = PredictionTrace(
            prediction_kind="capability_evolution",
            input_snapshot_id=str(getattr(evidence, "version", "") or ""),
            input_version=str(getattr(evidence, "version", "") or ""),
            prediction_version=CAPABILITY_EVOLUTION_PROTOCOL_VERSION,
            comparable=False,
            not_comparable_reasons=["no prediction content was produced"],
        )
        trace.model_info["error"] = error
        if provider_result is not None:
            usage = provider_result.get("usage") or {}
            tokens = usage.get("completion_tokens")
            latency = provider_result.get("latency_s")
            vector = CostVector(measured=set())
            if isinstance(tokens, (int, float)) and tokens >= 0:
                vector.llm_tokens = float(tokens)
                vector.mark_measured("llm_tokens")
            if latency is not None:
                vector.latency_s = float(latency)
                vector.mark_measured("latency_s")
            if vector.measured_dims():
                trace.call_cost = vector
        return CapabilityEvolutionPrediction(
            current_evidence=evidence,
            candidate_operation=operation,
            status=status,
            experience_scope=experience_scope,
            task_targeting=task_targeting,
            baseline=baseline,
            horizon=horizon,
            horizon_tasks=horizon_tasks,
            trace=trace,
            service_available=service_available,
            provider_configured=provider_configured,
            service_implemented=True,
            notes=[f"prediction failed: {error}"] if error else [],
        )

    # -- persistence ---------------------------------------------------------

    def _save(self, prediction: CapabilityEvolutionPrediction, *,
              task_id: str = "", episode_id: Optional[str] = None) -> None:
        """Persist one capability prediction in its OWN table.

        Deliberately NOT ``contract_predictions``: that table is read as
        strategy-outcome payloads by the ledger and the query entry, and
        mixing the two generations there would make a capability record
        mis-deserialize as an OR prediction (or silently vanish from the
        budget). A separate table keeps both readers honest.
        """
        self.store.put_capability_prediction(
            prediction.prediction_id, str(task_id or ""),
            episode_id, self.store.dumps(prediction.to_dict()),
            created_at=prediction.trace.created_at)

    def get(self, prediction_id: str
            ) -> Optional[CapabilityEvolutionPrediction]:
        raw = self.store.get_capability_prediction(prediction_id)
        if raw is None:
            return None
        return CapabilityEvolutionPrediction.from_dict(
            self.store.loads(raw))

    def query(self, *, task_id: Optional[str] = None,
              episode_id: Optional[str] = None
              ) -> List[CapabilityEvolutionPrediction]:
        return [CapabilityEvolutionPrediction.from_dict(
            self.store.loads(raw))
            for raw in self.store.capability_predictions_for(
                task_id=(str(task_id) if task_id is not None else None),
                episode_id=episode_id)]
