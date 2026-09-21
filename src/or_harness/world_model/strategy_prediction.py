"""Training-free OR strategy-consequence prediction (world-model M3).

This module implements the STRATEGY OUTCOME prediction SERVICE: it takes one
FROZEN :class:`~or_harness.world_model.context.PredictionContext` and one
candidate, asks the configured provider for the candidate's predicted
benefit / cost / risk / uncertainty under the NEW protocol, parses the
answer into a :class:`~or_harness.world_model.contracts
.StrategyOutcomePrediction`, and persists it so it can later be bound to the
real execution.

Design boundaries (the reason this is a separate module):

- **One protocol, one prompt.** The request is assembled under the
  ``wm-so/1`` protocol: the framework fixes the candidate and the input
  identity; the model fills ONLY the allowed prediction content. It cannot
  rewrite the task, the candidate, the scope or the evidence sources.
- **The provider receives CONTENT, not ids.** The joint problem
  representation, the retrieval evidence and the capability evidence travel
  in the request (the context's ``provider_view``), so the model conditions
  on what the harness actually gathered.
- **Honest failure states.** Not configured, protocol unsupported, empty
  payload, invalid JSON, NaN/Infinity, out-of-range probabilities, wrong
  types, and "measured" claims with no evidence each produce a
  distinguishable result. A failed call keeps whatever usage it consumed —
  unknown is never free, and never a default success.
- **No retries, no defaults.** One call per prediction. A failed prediction
  is a recorded failure, not a retry loop.
- **Legacy untouched.** The old ``OutcomePrediction`` path
  (``predict_outcome``) keeps its own protocol and prompt; nothing here
  wraps an old payload and calls it the new protocol.
"""

from __future__ import annotations

import copy
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.schema import (
    is_finite_number as _finite,
    is_probability as _prob,
)
from or_harness.world_model.contracts import (
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    EvidenceRef,
    ExpectedCost,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    StrategyOutcomePrediction,
    UncertaintyStatement,
    validate_strategy_outcome,
)

#: Version of this prediction protocol (request schema + output schema).
STRATEGY_OUTCOME_PROTOCOL_VERSION = "wm-so/1"

#: Which request key selects the new protocol on the provider call.
PROTOCOL_REQUEST_KEY = "prediction_protocol"

#: The system prompt for the strategy-outcome protocol. The model fills a
#: FIXED schema: the framework supplies the candidate and the input identity,
#: the model may only fill the prediction content fields.
STRATEGY_OUTCOME_SYSTEM_PROMPT = (
    "You are a world model for an operations-research harness, answering "
    "under the wm-so/1 strategy-outcome prediction protocol. You are "
    "given ONE frozen problem context (joint problem representation, "
    "solving state, retrieval evidence, harness capability evidence, "
    "execution constraints) and ONE candidate strategy. Predict the "
    "consequences of EXECUTING that candidate under those conditions.\n"
    "OUTPUT RULES (read first, follow strictly):\n"
    "1. Respond with EXACTLY ONE raw JSON object and NOTHING else. No "
    "markdown, no code fences, no explanation outside the JSON.\n"
    "2. You predict WHAT WOULD HAPPEN IF the candidate runs. You have NOT "
    "run it. Do NOT solve the problem: no objective value, no solution, no "
    "decision values anywhere in your output.\n"
    "3. Every field is OPTIONAL. If you lack evidence for a field, OMIT it "
    "entirely — never write null, \"unknown\", -1 or a guess as a "
    "placeholder. An omitted field is an honest unknown.\n"
    "4. All numbers must be valid JSON numbers (not strings, never NaN or "
    "Infinity).\n"
    "5. You may NOT change the task, the candidate, the scope or the "
    "evidence: they are fixed in the request. Your output is prediction "
    "content only.\n"
    "Fields (all optional):\n"
    "- benefit: object. {\"kind\": one of effective_completion|"
    "solution_quality|valid_progress|correct_infeasibility_diagnosis, "
    "\"metric\": a short name of WHAT is measured (required when benefit "
    "is present), \"unit\": the unit of the metric, \"value\": number — "
    "for kind=solution_quality this MUST be a normalized value in [0,1] "
    "(e.g. 1-gap); a raw objective value is NOT a solution_quality, "
    "\"interval\": [lo, hi], \"baseline\": {\"kind\": one of "
    "no_knowledge|conditional_stats|current_entry|current_solution|"
    "declared|unknown, \"value\": number, \"note\": text}, \"feasible\": "
    "boolean, \"notes\": [text]}. A value with no baseline is invalid.\n"
    "- cost: object of the resource dimensions you have evidence for, each "
    "a non-negative number: {llm_tokens, tool_calls, solver_runtime_s, "
    "retries, latency_s}. Include ONLY dimensions you actually predict; an "
    "omitted dimension is unknown, never zero.\n"
    "- risk: object {\"events\": [{\"event\": a short name of the risk "
    "EVENT (e.g. model_invalid, constraint_violation, budget_exhausted, "
    "no_feasible_solution), \"probability\": number in [0,1] or omitted "
    "when you have no basis, \"severity\": number or omitted, "
    "\"severity_unit\": text, \"basis\": text, \"notes\": [text]}]}. Risk "
    "events are LOSSES separate from cost; rework already counted in cost "
    "is not repeated here.\n"
    "- uncertainty: object {\"execution_randomness\": number in [0,1] — "
    "variance inherent to running the candidate (solver heuristics, "
    "timing), \"knowledge_gap\": number in [0,1] — how little evidence the "
    "prediction rests on, \"basis\": [text], \"missing\": {field: reason}, "
    "\"notes\": [text]}. These are your own UNCALIBRATED estimates; the "
    "framework records them as such and never as measured probabilities.\n"
    "- evidence_basis: list of strings naming the evidence keys you relied "
    "on (e.g. \"retrieval_evidence.hits[0]\", \"capability.sources.m\").\n"
    "- unsupported_fields: object {field: reason} for fields you cannot or "
    "will not predict.\n"
    "Do NOT fabricate evidence. If the context contains no relevant "
    "experience or knowledge for the candidate, say so in "
    "unsupported_fields and leave the numeric fields omitted. Output the "
    "JSON object only."
)

#: The maximum number of risk events accepted from one payload (bounded on
#: purpose: a prediction is not a risk register).
MAX_RISK_EVENTS = 20

#: The maximum number of evidence-basis entries accepted.
MAX_EVIDENCE_BASIS = 40


def build_strategy_outcome_request(
        context: Any, candidate: CandidateRef,
        *, benefit_baseline_hint: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
    """Assemble the provider request for one candidate under one context.

    The framework fixes the identity (context id/version, task digest,
    candidate) and supplies the CONTENT (joint representation, retrieval
    evidence, capability evidence, constraints). The model fills only the
    prediction content — the request says so explicitly.
    """
    request: Dict[str, Any] = {
        PROTOCOL_REQUEST_KEY: STRATEGY_OUTCOME_PROTOCOL_VERSION,
        "request_kind": "strategy_outcome_prediction",
        "prediction_context": context.provider_view(),
        "candidate": candidate.to_dict(),
        "output_contract": {
            "allowed_fields": ["benefit", "cost", "risk", "uncertainty",
                               "evidence_basis", "unsupported_fields"],
            "fixed_by_framework": ["task", "candidate", "scope",
                                   "evidence sources"],
            "note": ("the model fills prediction content only; it may not "
                     "rewrite the task, the candidate, the scope or the "
                     "evidence"),
        },
    }
    if benefit_baseline_hint:
        request["benefit_baseline_hint"] = copy.deepcopy(
            benefit_baseline_hint)
    return request


def parse_strategy_outcome_payload(
        payload: Any, candidate: CandidateRef,
        *, context: Any = None, provider_result: Optional[Dict[str, Any]] = None
        ) -> StrategyOutcomePrediction:
    """Parse a model payload into a validated StrategyOutcomePrediction.

    Every malformed value becomes an explicit problem (the prediction is
    returned with ``status="invalid"`` and the problems listed) — never a
    silently repaired "prediction". A payload with NO predicted content at
    all is ``contract_only``-shaped: the model produced nothing, and that is
    recorded rather than dressed up.
    """
    problems: List[str] = []
    notes: List[str] = []
    unsupported: Dict[str, str] = {}

    if not isinstance(payload, dict):
        payload = {}
        problems.append("payload is not a JSON object")

    # -- benefit -----------------------------------------------------------
    benefit: Optional[BenefitEstimate] = None
    raw_benefit = payload.get("benefit")
    if raw_benefit is None:
        unsupported["benefit"] = "not predicted by the model"
    elif not isinstance(raw_benefit, dict):
        problems.append("benefit must be a JSON object")
    else:
        kind = str(raw_benefit.get("kind") or "solution_quality")
        metric = str(raw_benefit.get("metric") or "")
        if not metric:
            problems.append("benefit.metric is required (what is measured)")
        value = raw_benefit.get("value")
        if value is not None and not _finite(value):
            problems.append("benefit.value is not a finite number")
            value = None
        if kind == "solution_quality" and value is not None \
                and not (0.0 <= float(value) <= 1.0):
            problems.append(
                "benefit.value for kind='solution_quality' must be a "
                "NORMALIZED value in [0,1] (e.g. 1-gap); a raw objective "
                "value is not silently clamped — restate the metric or use "
                "a different kind")
        interval = None
        raw_interval = raw_benefit.get("interval")
        if isinstance(raw_interval, (list, tuple)) and len(raw_interval) == 2:
            lo, hi = raw_interval
            if _finite(lo) and _finite(hi) and float(lo) <= float(hi):
                interval = (float(lo), float(hi))
            else:
                problems.append("benefit.interval must be finite with "
                                "lo<=hi")
        baseline = None
        raw_baseline = raw_benefit.get("baseline")
        if isinstance(raw_baseline, dict):
            # A malformed baseline (an unknown kind, a non-numeric value)
            # is a PROBLEM about this payload, never an exception that
            # escapes into the caller's planning loop and kills the other
            # candidates' predictions.
            try:
                baseline = BaselineStatement.from_dict(raw_baseline)
            except ValueError as exc:
                problems.append(f"benefit.baseline: {exc}")
                baseline = None
        elif value is not None:
            problems.append("benefit.value requires a baseline: a gain with "
                            "no baseline is not a prediction")
        feasible = raw_benefit.get("feasible")
        if feasible is not None and not isinstance(feasible, bool):
            problems.append("benefit.feasible must be a boolean")
            feasible = None
        try:
            benefit = BenefitEstimate(
                kind=kind, metric=metric or "(unnamed metric)",
                unit=str(raw_benefit.get("unit") or ""),
                value=(float(value) if value is not None else None),
                interval=interval, baseline=baseline, feasible=feasible,
                notes=[str(n) for n in (raw_benefit.get("notes") or [])])
        except ValueError as exc:
            problems.append(f"benefit: {exc}")
            benefit = None

    # -- cost ---------------------------------------------------------------
    cost: Optional[ExpectedCost] = None
    raw_cost = payload.get("cost")
    if raw_cost is None:
        unsupported["cost"] = "not predicted by the model"
    elif not isinstance(raw_cost, dict):
        problems.append("cost must be a JSON object of dimensions")
    else:
        dims: Dict[str, float] = {}
        for dim, value in raw_cost.items():
            if dim not in COST_DIMENSIONS:
                problems.append(f"unknown cost dimension {dim!r}")
            elif not _finite(value) or float(value) < 0:
                problems.append(f"cost.{dim} must be finite and >= 0")
            else:
                dims[dim] = float(value)
        if dims:
            try:
                vector = CostVector(**dims, measured=set(dims))
            except (TypeError, ValueError) as exc:
                problems.append(f"cost: {exc}")
                cost = None
            else:
                cost = ExpectedCost(
                    expected=vector, measured=sorted(dims),
                    notes=["predicted by the model under the strategy-outcome "
                           "protocol; the measured mask marks PREDICTED "
                           "dimensions, not observed ones"])
        elif not problems:
            unsupported["cost"] = ("no recognized cost dimension was "
                                   "predicted")

    # -- risk ---------------------------------------------------------------
    risk: Optional[RiskStatement] = None
    raw_risk = payload.get("risk")
    if raw_risk is None:
        unsupported["risk"] = "not predicted by the model"
    elif not isinstance(raw_risk, dict):
        problems.append("risk must be a JSON object with an events list")
    else:
        events: List[RiskEvent] = []
        raw_events = raw_risk.get("events")
        if raw_events is None:
            unsupported["risk"] = "no risk event was predicted"
        elif not isinstance(raw_events, list):
            problems.append("risk.events must be a list")
        else:
            if len(raw_events) > MAX_RISK_EVENTS:
                problems.append(
                    f"risk.events is bounded at {MAX_RISK_EVENTS}; "
                    f"{len(raw_events)} were supplied")
                raw_events = raw_events[:MAX_RISK_EVENTS]
            for index, raw in enumerate(raw_events):
                if not isinstance(raw, dict) or not raw.get("event"):
                    problems.append(f"risk.events[{index}].event is "
                                    "required")
                    continue
                probability = raw.get("probability")
                if probability is not None and not _prob(probability):
                    problems.append(f"risk.events[{index}].probability must "
                                    "be in [0, 1]")
                    probability = None
                severity = raw.get("severity")
                if severity is not None and not _finite(severity):
                    problems.append(f"risk.events[{index}].severity must be "
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
                try:
                    risk = RiskStatement(
                        events=events,
                        notes=[str(n) for n in
                               (raw_risk.get("notes") or [])])
                except ValueError as exc:
                    problems.append(f"risk: {exc}")
                    risk = None

    # -- uncertainty ----------------------------------------------------------
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
            # The model's own estimates are recorded as an honest
            # self-report: the NUMBERS are kept (they are the model's
            # uncertainty statement) but the source is labelled
            # model_self_report and the notes say explicitly that they are
            # UNCALIBRATED — never a measured probability. The contract
            # validator rejects numeric components under model_self_report
            # being passed off as calibrated, so the components are stored
            # in the notes and missing blocks instead of the numeric
            # fields, which a reader could mistake for calibrated values.
            try:
                uncertainty = UncertaintyStatement(
                    source="model_self_report",
                    basis=[str(b) for b in (raw_unc.get("basis") or [])],
                    missing={str(k): str(v) for k, v in
                             (raw_unc.get("missing") or {}).items()},
                    notes=[str(n) for n in (raw_unc.get("notes") or [])]
                    + [f"execution_randomness (self-reported, uncalibrated): "
                       f"{components.get('execution_randomness')}"
                       if "execution_randomness" in components else ""]
                    + [f"knowledge_gap (self-reported, uncalibrated): "
                       f"{components.get('knowledge_gap')}"
                       if "knowledge_gap" in components else ""]
                    + ["the model's self-reported uncertainty is recorded "
                       "as an UNCALIBRATED estimate in these notes, never "
                       "as a measured probability; it is not used as one "
                       "anywhere"])
            except ValueError as exc:
                problems.append(f"uncertainty: {exc}")
                uncertainty = None
            else:
                uncertainty.notes = [n for n in uncertainty.notes if n]
        elif not problems:
            unsupported["uncertainty"] = ("no uncertainty component was "
                                          "predicted")

    # -- evidence basis / unsupported ---------------------------------------
    evidence_basis: List[EvidenceRef] = []
    raw_basis = payload.get("evidence_basis")
    if isinstance(raw_basis, list):
        if len(raw_basis) > MAX_EVIDENCE_BASIS:
            raw_basis = raw_basis[:MAX_EVIDENCE_BASIS]
        for item in raw_basis:
            evidence_basis.append(EvidenceRef(
                ref_type="context_evidence", ref_id=str(item)))
    raw_unsupported = payload.get("unsupported_fields")
    if isinstance(raw_unsupported, dict):
        unsupported.update({str(k): str(v) for k, v in
                            raw_unsupported.items()})

    has_content = any(value is not None for value in
                      (benefit, cost, risk, uncertainty))
    trace = PredictionTrace(
        prediction_kind="strategy_outcome",
        input_snapshot_id=str(getattr(context, "snapshot_id", "") or ""),
        input_version=str(getattr(context, "effective_input_digest", "")
                          or getattr(context, "task_digest", "") or ""),
        prediction_version=STRATEGY_OUTCOME_PROTOCOL_VERSION,
        evidence_basis=evidence_basis,
        unsupported_fields=unsupported,
        comparable=False,
        not_comparable_reasons=["the candidate has not been executed: no "
                                "real window exists to score against"],
    )
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
        latency_value = provider_result.get("latency_s")
        if latency_value is not None:
            trace.model_info["call_latency_s"] = round(
                float(latency_value), 4)

    if not has_content and not problems:
        # The model produced a well-formed object with NO prediction in it.
        # That is the honest "nothing was predicted" state — never a valid
        # forecast, and never an error either.
        notes.append(
            "the model returned no predicted content: every field was "
            "omitted, so this is a recorded non-prediction, not a forecast")
        status = "contract_only"
    elif problems:
        status = "invalid"
        notes.extend(f"validation: {p}" for p in problems)
    else:
        status = "valid"

    prediction = StrategyOutcomePrediction(
        candidate=candidate,
        status=status,
        benefit=benefit,
        cost=cost,
        risk=risk,
        uncertainty=uncertainty,
        trace=trace,
        service_available=True,
        provider_configured=True,
        service_implemented=True,
        notes=notes,
    )
    if status == "valid":
        # Re-run the contract validator on the assembled object: the parse
        # rules and the contract rules must agree.
        contract_problems = validate_strategy_outcome(prediction)
        if contract_problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}"
                                    for p in contract_problems)
    return prediction


class StrategyOutcomeService:
    """Predict / persist / bind strategy-outcome predictions (wm-so/1)."""

    def __init__(self, store, provider):
        self.store = store
        self.provider = provider

    # -- predict -----------------------------------------------------------

    def predict(self, context: Any, candidate: CandidateRef,
                *, timeout_s: Optional[float] = None,
                benefit_baseline_hint: Optional[Dict[str, Any]] = None,
                ) -> StrategyOutcomePrediction:
        """One provider call, parsed and persisted (failures included).

        The result is ALWAYS a persisted prediction object: a failed call is
        a recorded failure with whatever usage it consumed, never an
        exception the caller must catch to learn the model is down.
        """
        request = build_strategy_outcome_request(
            context, candidate, benefit_baseline_hint=benefit_baseline_hint)
        try:
            try:
                result = self.provider.predict(request, timeout_s=timeout_s)
            except TypeError:
                result = self.provider.predict(request)
        except Exception as exc:  # provider adapter failure
            prediction = self._failed(
                context, candidate, "provider_error",
                f"{type(exc).__name__}: {exc}")
            self._save(prediction)
            return prediction
        if result.get("not_configured"):
            prediction = self._failed(
                context, candidate, "contract_only",
                str(result.get("error") or "provider not configured"),
                provider_configured=False, service_available=False)
            self._save(prediction)
            return prediction
        payload = result.get("payload")
        if payload is None:
            prediction = self._failed(
                context, candidate, "invalid",
                str(result.get("error") or "no payload"),
                provider_result=result)
            self._save(prediction)
            return prediction
        try:
            prediction = parse_strategy_outcome_payload(
                payload, candidate, context=context,
                provider_result=result)
        except Exception as exc:
            # A parse/validation crash is THIS candidate's invalid result,
            # never an exception that escapes into the caller's planning
            # loop and kills the remaining candidates' predictions. The
            # call really happened, so whatever usage it consumed is kept.
            prediction = self._failed(
                context, candidate, "invalid",
                f"payload could not be parsed: {type(exc).__name__}: {exc}",
                provider_result=result)
            self._save(prediction)
            return prediction
        if prediction.status == "invalid" \
                and not prediction.trace.model_info.get("parse_error"):
            prediction.trace.model_info["parse_error"] = (
                "the model's payload failed validation; see notes")
        self._save(prediction)
        return prediction

    def _failed(self, context: Any, candidate: CandidateRef,
                status: str, error: str,
                *, provider_configured: bool = True,
                service_available: bool = True,
                provider_result: Optional[Dict[str, Any]] = None
                ) -> StrategyOutcomePrediction:
        trace = PredictionTrace(
            prediction_kind="strategy_outcome",
            input_snapshot_id=str(getattr(context, "snapshot_id", "") or ""),
            input_version=str(
                getattr(context, "effective_input_digest", "")
                or getattr(context, "task_digest", "") or ""),
            prediction_version=STRATEGY_OUTCOME_PROTOCOL_VERSION,
            comparable=False,
            not_comparable_reasons=["the candidate has not been executed"],
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
        return StrategyOutcomePrediction(
            candidate=candidate,
            status=status,
            trace=trace,
            service_available=service_available,
            provider_configured=provider_configured,
            service_implemented=True,
            notes=[f"prediction failed: {error}"] if error else [],
        )

    # -- persistence ---------------------------------------------------------

    def _save(self, prediction: StrategyOutcomePrediction) -> None:
        self.store.put_contract_prediction(
            prediction.prediction_id,
            prediction.candidate.task_id,
            prediction.candidate.episode_id,
            self.store.dumps(prediction.to_dict()),
            created_at=prediction.trace.created_at)

    def get(self, prediction_id: str) -> Optional[StrategyOutcomePrediction]:
        raw = self.store.get_contract_prediction(prediction_id)
        if raw is None:
            return None
        from or_harness.world_model.contracts import (
            StrategyOutcomePrediction as SOP,
        )
        return SOP.from_dict(self.store.loads(raw))

    def query(self, *, task_id: Optional[str] = None,
              episode_id: Optional[str] = None
              ) -> List[StrategyOutcomePrediction]:
        from or_harness.world_model.contracts import (
            StrategyOutcomePrediction as SOP,
        )
        return [SOP.from_dict(self.store.loads(raw))
                for raw in self.store.contract_predictions_for(
                    task_id=task_id, episode_id=episode_id)]
