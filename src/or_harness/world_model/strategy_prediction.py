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
    CapabilityGain,
    EvidenceRef,
    ExpectedCost,
    INTERVAL_KINDS,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    StrategyOutcomePrediction,
    UncertaintyStatement,
    validate_strategy_outcome,
)

#: Version of this prediction protocol (request schema + output schema).
STRATEGY_OUTCOME_PROTOCOL_VERSION = "wm-so/1"

#: Why a finished call produced no usable prediction. The set is deliberately
#: small and DECIDABLE from what the adapter observed — never a guess about
#: the endpoint:
#:
#: - ``network_error``: the HTTP call itself failed (connect/read/TLS/HTTP
#:   status). The transport never delivered a response body.
#: - ``timeout``: the failure is specifically a timeout.
#: - ``truncated``: a response arrived but the provider said it stopped for
#:   length, OR the content is not parsable — a cut-off answer, not a
#:   different shape. ``finish_reason == "length"`` is the authoritative
#:   signal; an ABSENT ``finish_reason`` is UNKNOWN and is NOT read as a
#:   normal ending.
#: - ``empty_response``: usable content was empty (``""`` or ``null``).
#: - ``wrong_top_level``: valid JSON whose top level is not an object — the
#:   field case of ``[]``.
#: - ``unparsable``: non-JSON content, or JSON that is not an object.
#: - ``unusable_payload``: a JSON object that failed contract validation.
FAILURE_KINDS = (
    "network_error", "timeout", "truncated", "empty_response",
    "wrong_top_level", "unparsable", "unusable_payload",
)

#: ``finish_reason`` values meaning "the model was cut off", not "it chose
#: to stop". Anything else — including a MISSING field — is not an ending
#: claim: the field's absence is recorded as unknown rather than assumed OK.
_STOP_BY_LENGTH = ("length", "max_tokens", "max_output_tokens",
                   "token_limit")


def diagnose_provider_failure(
        result: Optional[Dict[str, Any]],
        exc: Optional[BaseException],
        *, top_level: Any = None,
        validation_error: Optional[str] = None) -> Dict[str, Any]:
    """Classify WHY a call produced no usable prediction.

    Reads only what the adapter actually observed. A field the endpoint did
    not send stays absent from the diagnosis (``finish_reason`` missing is
    reported as ``None``, never as ``"stop"``): an unobserved value must not
    be turned into an observed one.
    """
    diagnostics = dict((result or {}).get("diagnostics") or {})
    message = ""
    if exc is not None:
        message = f"{type(exc).__name__}: {exc}"
    elif result is not None:
        message = str(result.get("error") or "")
    lowered = message.lower()

    finish_reason = diagnostics.get("finish_reason")
    content_kind = diagnostics.get("content_kind")

    if exc is not None and "timed out" not in lowered \
            and "timeout" not in lowered:
        kind = "network_error"
    elif "timed out" in lowered or "timeout" in lowered:
        kind = "timeout"
    elif validation_error is not None:
        kind = "unusable_payload"
    elif top_level is not None and not isinstance(top_level, dict):
        kind = "wrong_top_level"
    elif finish_reason in _STOP_BY_LENGTH:
        kind = "truncated"
    elif content_kind in ("empty", "null") or not message and result is not None \
            and (result.get("payload") is None):
        kind = ("empty_response" if content_kind in ("empty", "null")
                else "unparsable")
    else:
        kind = "unparsable"

    diagnosis: Dict[str, Any] = {"kind": kind}
    if message:
        diagnosis["detail"] = message[:500]
    if validation_error:
        diagnosis["validation_error"] = str(validation_error)[:300]
    if top_level is not None and not isinstance(top_level, dict):
        diagnosis["top_level"] = type(top_level).__name__
    # Only fields the adapter really reported are echoed back.
    for key in ("finish_reason", "content_kind", "raw_content_chars",
                "effective", "reasoning_tokens"):
        if key in diagnostics and diagnostics[key] is not None:
            diagnosis[key] = diagnostics[key]
    if finish_reason is None:
        diagnosis["finish_reason"] = None
        diagnosis["note"] = (
            "the endpoint reported no finish_reason, so how the answer "
            "ended is UNKNOWN — not evidence that it ended normally")
    # A count, never the reasoning TEXT: the diagnosis must stay bounded.
    if diagnostics.get("reasoning_tokens") is not None \
            and diagnostics.get("completion_tokens") is not None:
        diagnosis["reasoning_tokens_included_in_completion"] = (
            "reported separately by the endpoint; not added again")
    return diagnosis


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
    "6. The candidate carries a \"method\" object describing what the method "
    "IS: its \"name\" and its \"steps\" (plus optional \"why\"/\"fallback\"). "
    "Base your prediction on those steps. When \"method\" is empty the "
    "caller did not describe it — report the fields you cannot judge under "
    "unsupported_fields rather than guessing from the id or the solver "
    "name, and never treat the solver/config as the method.\n"
    "Fields (all optional):\n"
    "- benefit: object. {\"kind\": one of effective_completion|"
    "solution_quality|valid_progress|correct_infeasibility_diagnosis, "
    "\"metric\": a short name of WHAT is measured (required when benefit "
    "is present), \"unit\": the unit of the metric, \"value\": number — "
    "for kind=solution_quality this MUST be a normalized value in [0,1] "
    "(e.g. 1-gap); a raw objective value is NOT a solution_quality, "
    "\"interval\": [lo, hi], \"interval_kind\": one of outcome|mean (WHAT "
    "the range is about — \"outcome\" = this ONE run's value, \"mean\" = "
    "the average over repeated runs), \"interval_coverage\": the nominal "
    "coverage level you claim (e.g. 0.9), \"baseline\": {\"kind\": one of "
    "no_knowledge|conditional_stats|current_entry|current_solution|"
    "declared|unknown, \"value\": number, \"note\": text}, \"feasible\": "
    "boolean, \"notes\": [text]}. A value with no baseline is invalid.\n"
    "  BENEFIT CONVENTION: the request carries a `benefit_convention` "
    "block naming the ONE yardstick this decision uses. Predict under it — "
    "`kind=solution_quality` with `metric=normalized_objective_gap` measures "
    "how well the SOLVER solved the model; use `kind=effective_completion` "
    "with `metric=task_result_check_passed` ONLY when the question is "
    "whether the ANSWER satisfies the TASK. These measure different things "
    "and are never interchanged. If you declare a different kind or metric, "
    "your prediction is kept and reported, but it is NOT ranked on the "
    "comparison's single yardstick: an upside in a currency the comparison "
    "cannot read is never scored as zero.\n"
    "  INTERVAL SEMANTICS: an interval must say WHAT it is about. "
    "`interval_kind=\"outcome\"` covers ONE execution's observed value; "
    "`interval_kind=\"mean\"` covers the average over repeated runs. A "
    "success PROBABILITY (kind=effective_completion) is a mean over runs, "
    "NOT a single 0/1 label — do not wrap it in a narrow `outcome` interval "
    "and do not present a narrow probability range as if it bounded one "
    "run. Reporting an interval without a kind leaves the semantics "
    "unstated, and the framework will say so.\n"
    "- cost: object of the resource dimensions you have evidence for, each "
    "a non-negative number: {llm_tokens, tool_calls, solver_runtime_s, "
    "retries, latency_s}. Include ONLY dimensions you actually predict; an "
    "omitted dimension is unknown, never zero.\n"
    "- risk: object {\"events\": [{\"event\": a short name of the risk "
    "EVENT, \"probability\": number in [0,1] or omitted when you have no "
    "basis, \"severity\": number or omitted, \"severity_unit\": text, "
    "\"basis\": text, \"notes\": [text]}]}. Risk events are LOSSES separate "
    "from cost; rework already counted in cost is not repeated here.\n"
    "  OBSERVABLE EVENT NAMES (use EXACTLY one of these when it applies — "
    "the framework has a channel that can confirm or refute it, and an "
    "event outside this list can never be scored):\n"
    "    * timeout — the solve would exceed its time limit;\n"
    "    * solver_reported_infeasible — the solver would report the model "
    "infeasible (a VERDICT about the model as given, not by itself a "
    "strategy failure);\n"
    "    * environment_failure — the script could not run here (sandbox "
    "policy, missing module);\n"
    "    * implementation_failure — the harness's own script/stack would "
    "fail;\n"
    "    * task_check_failed — the answer would FAIL the declared "
    "task-result check (this is the check's outcome, NOT a verdict that "
    "the model was wrong);\n"
    "    * budget_exhausted — the episode's declared budget would be "
    "exceeded.\n"
    "  Do NOT invent names like \"model_invalid\" or "
    "\"no_feasible_solution\": they were retired because no channel can "
    "establish them (a solver error status or a failed check does not prove "
    "the MODEL was wrong). A business risk with no observation channel may "
    "still be described in \"basis\"/\"notes\", but it must NOT be given an "
    "invented event name that looks scoreable.\n"
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
    "- capability_gain: object — the POTENTIAL capability gain (H+) of this "
    "candidate, predicted in THIS SAME answer. It is EXPLANATORY: it never "
    "ranks the candidates by itself and it is NEVER a claim that the "
    "harness got stronger. You MUST include this block for EVERY "
    "candidate and state an explicit stance in \"assessment\", exactly "
    "one of: \"expected\" (a gain is expected — then carry a \"claim\" "
    "and/or \"expected_changes\"), \"none\" (no new gain expected — the "
    "evidence fields may be empty), or \"insufficient_basis\" (you cannot "
    "judge from the evidence given — the evidence fields may be empty). "
    "An absent block is NOT \"no gain\": it is an unstated stance, and the "
    "framework records it as such. Shape: {\"assessment\": "
    "expected|none|insufficient_basis, \"claim\": short text of WHAT "
    "capability could improve, "
    "\"applies_to\": [text: which problems/structures the gain would apply "
    "to], \"expected_changes\": [{\"metric\": text, \"direction\": "
    "increase|decrease|unchanged|unknown, \"value\": number?, \"unit\": "
    "text?, \"value_kind\": absolute|relative?, \"beneficial_direction\": "
    "increase|decrease|either, \"baseline\": {...}?}] each in the metric's "
    "OWN unit — do NOT compress them into a 0-1 score, "
    "\"evidence_required\": [text: the REAL evidence / reuse / verification "
    "that would confirm the gain], \"verification_conditions\": "
    "[{\"condition\": text}], \"degradation_risk\": {\"events\": [{...}]} "
    "(what could make it WORSE), \"uncertainty\": [text: what you are "
    "unsure about and what might yield NO gain], \"basis\": [text]}. A new "
    "method, one more experience entry, or having tried and failed once is "
    "NOT by itself a capability gain: say what the improvement IS, on which "
    "metric, and how it would be confirmed — or state "
    "assessment=\"none\"/\"insufficient_basis\" honestly.\n"
    "CALIBRATION EVIDENCE: the context may carry a "
    "`strategy_outcome_calibration` block describing how THIS model's past "
    "predictions turned out. Use it to adjust your numbers, and read it "
    "carefully:\n"
    "  * it is grouped by (model, metric, unit, scope). Only groups whose "
    "model/metric/unit/scope MATCH the metric and scope you are predicting "
    "under are your evidence. A group with a different metric, unit or "
    "scope is NOT comparable to what you are about to predict — do not cite "
    "it as support for this number.\n"
    "  * `mean_benefit_signed_error` is directed: POSITIVE means the real "
    "outcome was historically BETTER than predicted (this model "
    "under-predicts benefit); NEGATIVE means it over-predicts.\n"
    "  * `mean_cost_log_ratio` is directed per dimension: POSITIVE means the "
    "real cost was historically HIGHER than predicted (under-predicts "
    "cost); NEGATIVE means over-predicts.\n"
    "  * `basis: insufficient_evidence` means too few independent episodes "
    "to say anything: do NOT treat such a group as calibration.\n"
    "  * these are measured PAST errors, not a promise about this "
    "prediction, and they carry NO per-strategy breakdown (they are a "
    "global diagnostic).\n"
    "PAIRED FEEDBACK: the context may carry a `prediction_execution_pairs` "
    "block: the compact, per-episode facts of how THIS model's past "
    "predictions turned out. Each pair names the conditions, the planned "
    "method, the ORIGINAL predicted value, the REAL observation, the "
    "per-field difference (`signed_error` positive = the real outcome was "
    "BETTER than predicted; `cost_log_ratio` positive = real cost was "
    "HIGHER), the task-check outcome and any not-comparable reasons. Read "
    "them as EVIDENCE about your own past accuracy under similar "
    "conditions — successes AND failures are included, and a cross-cell or "
    "cross-strategy-name case is still evidence about your bias. When a "
    "pair's field is unknown or not comparable, that does NOT invalidate "
    "its other fields. Do NOT read a pair as a menu: it is a record of what "
    "was predicted and what happened, never a recommendation.\n"
    "PREDICTION REMINDERS: a `prediction_reminders` block may carry "
    "DETERMINISTIC 'watch this next time' notes derived from the measured "
    "statistics above (never written by a model). Each states the "
    "applicability it came from and the observed bias; apply it only where "
    "its applicability matches the metric/scope you are predicting under, "
    "and treat the observed bias as a measured FACT, the instruction as a "
    "reminder.\n"
    "CAPABILITY-GAIN FOLLOW-UP: a `hplus_feedback` block may describe how "
    "earlier H+ claims turned out. `pending` means the real follow-up has "
    "not happened yet — NOT a failure, and NOT an improvement. Only "
    "`effect_verified` confirms a claim was borne out. If your own past "
    "'expected' stances repeatedly failed to verify, be more conservative "
    "about claiming a gain now.\n"
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

#: The SHARED benefit convention every candidate of one decision predicts
#: under. It travels in the request so the model is told the ONE yardstick
#: the comparison and the close-out will use — not left to pick a kind and a
#: metric per candidate and hope they line up. The comparison
#: (``planner.COMPARABLE_BENEFIT_METRIC``) and the close-out
#: (``episode_closeout.OBSERVABLE_BENEFIT_METRIC``) read the SAME
#: definition, so a benefit that can be ranked here can also be evaluated
#: there.
BENEFIT_CONVENTION: Dict[str, Any] = {
    "default": {
        "kind": "solution_quality",
        "metric": "normalized_objective_gap",
        "unit": "1-gap",
        "scope": "the prediction's own candidate scope",
        "observed_from": ("the solver's own normalized gap (an `optimal` "
                          "status is a gap of 0; otherwise 1-gap)"),
        "note": ("how well the solver solved the model it was given. Use "
                 "this unless the question is whether the ANSWER satisfies "
                 "the TASK"),
    },
    "completion": {
        "kind": "effective_completion",
        "metric": "task_result_check_passed",
        "unit": "boolean",
        "scope": "the prediction's own candidate scope",
        "observed_from": ("the execution's own task-result check "
                          "(`check-task`): 1.0 passed, 0.0 confirmed failed, "
                          "UNKNOWN when unchecked or insufficient"),
        "note": ("whether the ANSWER satisfies the TASK — a different "
                 "measurement that lands in [0,1] as well and is NEVER "
                 "interchanged with the solver's gap"),
    },
    "comparison_yardstick": {
        "kind": "solution_quality",
        "metric": "normalized_objective_gap",
        "note": ("only this pair is on the planner's single comparable "
                 "yardstick. A benefit in another kind or metric is "
                 "reported and observed, but it does NOT enter the full "
                 "utility ranking (an unread upside is never scored as "
                 "zero)"),
    },
    "requirements": {
        "declare_before_the_run": ("choose the kind/metric from the TASK "
                                   "before executing: the close-out "
                                   "observes the metric you declared, and "
                                   "it never re-labels another measurement "
                                   "into it"),
        "value_and_baseline": ("a benefit VALUE requires a baseline; a "
                               "`solution_quality` value is NORMALIZED in "
                               "[0,1] (a raw objective value is refused, "
                               "never clamped)"),
        "same_convention_per_decision": ("every candidate of one decision "
                                         "is predicted under the SAME "
                                         "convention, so the comparison "
                                         "reads one currency"),
        "interval_semantics": ("an interval states WHAT it is about: "
                               "`interval_kind='outcome'` bounds ONE run's "
                               "value, `interval_kind='mean'` bounds the "
                               "average over runs. A completion PROBABILITY "
                               "is a mean over runs, never a single 0/1 "
                               "label's interval; `interval_coverage` "
                               "records the nominal level claimed. Omitting "
                               "the kind leaves the range's meaning "
                               "unstated and the close-out says so"),
    },
}


def build_strategy_outcome_request(
        context: Any, candidate: CandidateRef,
        *, benefit_baseline_hint: Optional[Dict[str, Any]] = None,
        benefit_convention: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
    """Assemble the provider request for one candidate under one context.

    The framework fixes the identity (context id/version, task digest,
    candidate) and supplies the CONTENT (joint representation, retrieval
    evidence, capability evidence, constraints). The model fills only the
    prediction content — the request says so explicitly.

    ``benefit_convention`` is the ONE convention THIS DECISION compares
    under. When the caller declared one (e.g. completion rather than
    quality) it is sent as the REQUIRED convention rather than offered as
    the default, so the candidates are predicted under the currency the
    comparison will use. Without a declaration the general block travels,
    and the candidates' own agreement decides.

    The context's calibration block is FILTERED before it is sent: only the
    groups whose SCOPE matches this candidate's scope are kept, because a
    window-scope error statistic is not evidence about a single attempt (and
    the reverse). The filtering is reported on the request, so a reader can
    see that a group was withheld rather than absent.
    """
    declared = dict(benefit_convention or {})
    declared_kind = declared.get("kind")
    declared_metric = declared.get("metric")
    fixed_convention: Optional[Dict[str, Any]] = None
    if declared_kind and declared_metric:
        fixed_convention = {
            "kind": str(declared_kind),
            "metric": str(declared_metric),
            "unit": declared.get("unit"),
            "scope": "the prediction's own candidate scope",
            "observed_from": declared.get("observed_from"),
            "source": declared.get("source", "declared"),
            "required": True,
            "note": ("THIS DECISION compares benefit under this convention. "
                     "Predict this candidate's benefit with EXACTLY this "
                     "kind and metric: a different kind or metric cannot be "
                     "ranked against the other candidates, and it is NOT "
                     "silently converted"),
        }
    request: Dict[str, Any] = {
        PROTOCOL_REQUEST_KEY: STRATEGY_OUTCOME_PROTOCOL_VERSION,
        "request_kind": "strategy_outcome_prediction",
        "prediction_context": context.provider_view(),
        "candidate": candidate.to_dict(),
        # The SHARED benefit convention: the framework fixes the yardstick
        # so every candidate of one decision is predicted on the SAME
        # currency, and the comparison and the close-out read the same
        # definition. A model that deviates is reported (its prediction is
        # kept, but the benefit does not enter the full utility ranking).
        "benefit_convention": (fixed_convention
                               if fixed_convention is not None
                               else copy.deepcopy(BENEFIT_CONVENTION)),
        "output_contract": {
            "allowed_fields": ["benefit", "cost", "risk", "uncertainty",
                               "capability_gain", "evidence_basis",
                               "unsupported_fields"],
            "fixed_by_framework": ["task", "candidate", "scope",
                                   "evidence sources", "benefit convention"],
            "note": ("the model fills prediction content only; it may not "
                     "rewrite the task, the candidate, the scope or the "
                     "evidence. Use the `benefit_convention` kind/metric "
                     "unless the question is task completion"),
        },
    }
    # Scope filtering of the calibration block (the model identity was
    # already filtered when the context was frozen, since that is decided by
    # the attached provider, not by this candidate).
    view = request["prediction_context"]
    calibration = view.get("strategy_outcome_calibration")
    if isinstance(calibration, dict) and calibration.get("groups"):
        filtered, withheld = _filter_calibration_by_scope(
            calibration, candidate.scope)
        view["strategy_outcome_calibration"] = filtered
        if withheld:
            view["strategy_outcome_calibration"]["withheld_groups"] = withheld
            view["strategy_outcome_calibration"]["filter_note"] = (
                f"{len(withheld)} calibration group(s) under a different "
                f"scope were withheld: their errors describe a different "
                "measurement unit and are not evidence about this "
                f"candidate's scope ({candidate.scope!r})")
    if benefit_baseline_hint:
        request["benefit_baseline_hint"] = copy.deepcopy(
            benefit_baseline_hint)
    return request


def _filter_calibration_by_scope(calibration: Dict[str, Any],
                                 scope: str) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only the calibration groups whose declared scope matches.

    Group keys are ``strategy_outcome|<model>|<metric>|<unit>|<scope>``
    (a trailing ``|<observation_rule_version>`` part may be present); a
    group under a different scope measures a different unit (one attempt vs
    a whole strategy window), so its error statistics are not evidence
    about this candidate. Returns ``(filtered_calibration, withheld_keys)``.
    """
    filtered = copy.deepcopy(calibration)
    kept: Dict[str, Any] = {}
    withheld: List[str] = []
    for key, group in (calibration.get("groups") or {}).items():
        parts = str(key).split("|")
        group_scope = parts[4] if len(parts) > 4 else None
        if group_scope is None or group_scope == str(scope):
            kept[key] = copy.deepcopy(group)
        else:
            withheld.append(str(key))
    filtered["groups"] = kept
    return filtered, withheld


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
        # WHAT the interval is about and its NOMINAL coverage. Both are
        # OPTIONAL and preserved as stated: an absent kind is the legacy
        # "unstated" case, never auto-filled; an unknown kind string is kept
        # verbatim so validation reports it instead of erasing it.
        raw_interval_kind = raw_benefit.get("interval_kind")
        interval_kind = (str(raw_interval_kind).strip().lower()
                         if raw_interval_kind else None)
        if interval is not None and interval_kind is None \
                and kind == "effective_completion":
            # A COMPLETION prediction is a probability of a 0/1 label; a
            # narrow range around it is NOT a single-label interval. Say so
            # in a note rather than refuse the payload — the number is still
            # reported, and the observation channel stays the task check.
            problems.append(
                "benefit.interval is stated for kind='effective_completion' "
                "without interval_kind: a success-PROBABILITY interval must "
                "not be read as a single-execution 0/1 interval. State "
                "interval_kind='mean' (a probability over runs) or omit the "
                "interval")
        raw_coverage = raw_benefit.get("interval_coverage")
        interval_coverage = None
        if raw_coverage is not None:
            if _finite(raw_coverage) and 0.0 < float(raw_coverage) < 1.0:
                interval_coverage = float(raw_coverage)
            else:
                problems.append("benefit.interval_coverage must be a "
                                "nominal level in (0, 1)")
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
                interval=interval, interval_kind=interval_kind,
                interval_coverage=interval_coverage,
                baseline=baseline, feasible=feasible,
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

    # -- capability gain (H+) -------------------------------------------------
    # EXPLANATORY only: it is parsed and recorded, NEVER scored and NEVER
    # written into the harness capability evidence. Three rules govern a
    # bad block:
    #
    # * a MISSING block is a contract violation — the model was REQUIRED to
    #   state a stance — but it must not invalidate the rest of the
    #   prediction. It must SAY SO: the framework records why the block is
    #   absent ("the model did not state an H+ stance"), exactly as it does
    #   for benefit/cost/risk, so a null is never a silent omission and
    #   never read as "no gain";
    # * a block with NO assessment is kept (an old payload, or a model that
    #   answered the old prompt): it reads as UNASSESSED, never as "none";
    # * a MALFORMED block (a bad metric, an unparsable entry, an illegal
    #   assessment) is a local error and takes only H+ down: the block is
    #   dropped with the reason recorded and the valid G/C/R prediction is
    #   KEPT. An explanatory extra must never destroy the forecast it
    #   explains.
    capability_gain: Optional[CapabilityGain] = None
    raw_gain = payload.get("capability_gain")
    if raw_gain is None:
        unsupported["capability_gain"] = (
            "not predicted by the model (the model did not state an H+ "
            "stance for this candidate, which the contract requires; an "
            "absent block means 'not assessed', never 'no gain')")
    elif not isinstance(raw_gain, dict):
        unsupported["capability_gain"] = (
            "malformed: capability_gain must be a JSON object; the block "
            "was dropped and the benefit/cost/risk prediction is kept")
        notes.append(
            "capability_gain was malformed and dropped: only the gain "
            "block is affected — the benefit/cost/risk prediction stands")
    else:
        try:
            capability_gain = CapabilityGain.from_dict(raw_gain)
        except (ValueError, TypeError) as exc:
            capability_gain = None
            unsupported["capability_gain"] = f"malformed: {exc}"
            notes.append(
                "capability_gain was malformed and dropped (" + str(exc) +
                "): only the gain block is affected — the benefit/cost/"
                "risk prediction stands")
        else:
            if not capability_gain.assessed:
                notes.append(
                    "capability_gain carried no assessment: recorded as "
                    "UNASSESSED (the model did not state a stance), never "
                    "as 'no gain'")
            if not capability_gain.claimed:
                notes.append(
                    "capability_gain was present but empty: recorded as "
                    "'no gain claimed', never a default positive")
            # The model may PREDICT; it may not declare its prediction
            # BOUND or VERIFIED. Those two flags are framework-maintained
            # facts about the real evidence, so whatever the payload says
            # about them is reset here.
            for condition in capability_gain.verification_conditions:
                condition.prediction_made = True
                condition.fact_bound = False
                condition.effect_verified = False

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
        # ONE token-accounting rule: the FULL口径 total (prompt +
        # completion), reasoning/cached kept as sub-facts. The breakdown
        # travels with the call so the口径 is traceable.
        from or_harness.world_model.usage import usage_cost_vector
        vector, breakdown = usage_cost_vector(
            provider_result.get("usage"), provider_result.get("latency_s"))
        if vector is not None:
            trace.call_cost = vector
        if breakdown is not None:
            trace.model_info["call_usage"] = breakdown
        if provider_result.get("latency_s") is not None:
            trace.model_info["call_latency_s"] = round(
                float(provider_result["latency_s"]), 4)

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
        capability_gain=capability_gain,
        trace=trace,
        service_available=True,
        provider_configured=True,
        service_implemented=True,
        notes=notes,
    )
    if status == "valid":
        # Re-run the contract validator on the assembled object: the parse
        # rules and the contract rules must agree. H+ problems are filtered
        # out here: a malformed explanatory gain block was already dropped
        # with its reason recorded, and it must not downgrade the valid
        # forecast beside it.
        contract_problems = [p for p in validate_strategy_outcome(prediction)
                             if not _is_capability_gain_problem(p)]
        if contract_problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}"
                                    for p in contract_problems)
    return prediction


def _is_capability_gain_problem(problem: str) -> bool:
    """Whether a contract problem belongs to the H+ block alone.

    The contract validator reports every problem as a flat string; the
    cause is the leading token. An H+ problem must be a LOCAL one: it is
    recorded as the block's drop reason and never invalidates the
    benefit/cost/risk forecast.
    """
    return str(problem).startswith("capability_gain")


def model_identity_label(identity: Optional[Dict[str, Any]]) -> str:
    """The single string that names a model identity, or ``(unknown)``.

    ONE rule, used both when a prediction records its identity and when the
    calibration groups by it, so a prediction and its group can never
    disagree about which model produced them. A model NAME is not a model
    VERSION: when only a name is known the label says so, and when nothing
    is known the answer is ``(unknown)`` — never the provider name standing
    in for a version.
    """
    identity = identity or {}
    model = identity.get("provider_model")
    version = identity.get("provider_version")
    if model and version:
        return f"{model}@{version}"
    if model:
        return str(model)
    if version:
        return f"(model unknown)@{version}"
    return "(unknown)"


class StrategyOutcomeService:
    """Predict / persist / bind strategy-outcome predictions (wm-so/1)."""

    def __init__(self, store, provider):
        self.store = store
        self.provider = provider
        # The provider's OWN identity, read once. Different models are
        # different predictors, so the calibration groups by this identity
        # and never pools their errors. ``provider_model`` is the name the
        # provider reports; ``provider_version`` is recorded ONLY when the
        # provider actually exposes one — a name is not a version, and
        # inventing a version from a name would silently merge two builds.
        self.model_identity = self._describe_identity()

    @property
    def model_identity_label(self) -> str:
        """This service's identity as the calibration group label."""
        return model_identity_label(self.model_identity)

    def _describe_identity(self) -> Dict[str, Any]:
        try:
            described = self.provider.describe() or {}
        except Exception:  # a provider that cannot describe itself
            return {}
        identity: Dict[str, Any] = {}
        model = described.get("model")
        if model:
            identity["provider_model"] = str(model)
        version = described.get("model_version") or described.get("version")
        if version:
            identity["provider_version"] = str(version)
        provider_name = described.get("provider")
        if provider_name:
            identity["provider_name"] = str(provider_name)
        return identity

    # -- predict -----------------------------------------------------------

    def _stamp_identity(self, prediction: StrategyOutcomePrediction
                        ) -> None:
        """Record which model made this prediction, on the prediction itself.

        The identity travels with the FROZEN prediction (not looked up at
        close-out), so a later evaluation can group by it without trusting
        today's provider configuration: the model that is attached now may
        not be the model that made an old prediction.
        """
        for key, value in self.model_identity.items():
            prediction.trace.model_info.setdefault(key, value)

    def predict(self, context: Any, candidate: CandidateRef,
                *, timeout_s: Optional[float] = None,
                benefit_baseline_hint: Optional[Dict[str, Any]] = None,
                benefit_convention: Optional[Dict[str, Any]] = None,
                ) -> StrategyOutcomePrediction:
        """One provider call, parsed and persisted (failures included).

        ``benefit_convention`` (when given) is the ONE convention the
        decision compares under; it travels in the request as a REQUIRED
        instruction so the candidate is predicted in that currency rather
        than merely compared in it afterwards.

        The result is ALWAYS a persisted prediction object: a failed call is
        a recorded failure with whatever usage it consumed, never an
        exception the caller must catch to learn the model is down.
        """
        request = build_strategy_outcome_request(
            context, candidate, benefit_baseline_hint=benefit_baseline_hint,
            benefit_convention=benefit_convention)
        try:
            try:
                result = self.provider.predict(request, timeout_s=timeout_s)
            except TypeError:
                result = self.provider.predict(request)
        except Exception as exc:  # provider adapter failure
            # The contract vocabulary has ONE honest word for "a prediction
            # was attempted and did not produce one": ``invalid``. The
            # REASON (a network failure, a timeout, a truncated answer) is
            # structured diagnostics, not a status — otherwise every reader
            # would have to learn a second status vocabulary that the
            # validator does not recognize.
            prediction = self._failed(
                context, candidate, "invalid",
                f"{type(exc).__name__}: {exc}",
                failure=diagnose_provider_failure(None, exc))
            self._save(prediction)
            return prediction
        if result.get("not_configured"):
            prediction = self._failed(
                context, candidate, "contract_only",
                str(result.get("error") or "provider not configured"),
                provider_configured=False, service_available=False,
                failure=diagnose_provider_failure(result, None))
            self._save(prediction)
            return prediction
        payload = result.get("payload")
        if payload is None:
            prediction = self._failed(
                context, candidate, "invalid",
                str(result.get("error") or "no payload"),
                provider_result=result,
                failure=diagnose_provider_failure(result, None))
            self._save(prediction)
            return prediction
        if not isinstance(payload, dict):
            # A JSON ARRAY (the field case: the endpoint returned ``[]``) is
            # valid JSON but not a prediction object. It is reported as its
            # own failure kind — "not a JSON object" alone would hide
            # whether the answer was truncated, fenced, or simply the wrong
            # top-level type.
            prediction = self._failed(
                context, candidate, "invalid",
                f"model returned a JSON {type(payload).__name__}, not a "
                "prediction object",
                provider_result=result,
                failure=diagnose_provider_failure(result, None,
                                                  top_level=payload))
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
                provider_result=result,
                failure=diagnose_provider_failure(result, None,
                                                  top_level=payload))
            self._save(prediction)
            return prediction
        if prediction.status == "invalid" \
                and not prediction.trace.model_info.get("parse_error"):
            prediction.trace.model_info["parse_error"] = (
                "the model's payload failed validation; see notes")
        self._stamp_identity(prediction)
        self._save(prediction)
        return prediction

    def _failed(self, context: Any, candidate: CandidateRef,
                status: str, error: str,
                *, provider_configured: bool = True,
                service_available: bool = True,
                provider_result: Optional[Dict[str, Any]] = None,
                failure: Optional[Dict[str, Any]] = None
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
        # The structured WHY lives on the trace beside the identity: a
        # reader must be able to tell a network failure from a truncated
        # answer without parsing an English sentence, and the diagnosis is
        # what survives into the stored record.
        if failure:
            trace.model_info["failure"] = failure
        if provider_result is not None:
            diagnostics = provider_result.get("diagnostics") or {}
            # The SAME one token-accounting rule as the success path: a
            # failed call still consumed real usage, recorded at the full
            #口径.
            from or_harness.world_model.usage import usage_cost_vector
            vector, breakdown = usage_cost_vector(
                provider_result.get("usage"),
                provider_result.get("latency_s"))
            if vector is not None:
                trace.call_cost = vector
            if breakdown is not None:
                trace.model_info["call_usage"] = breakdown
            # Effective parameters really used for this call, so a stored
            # failure explains itself without re-reading today's config.
            if diagnostics.get("effective"):
                trace.model_info["effective_parameters"] = copy.deepcopy(
                    diagnostics["effective"])
        # A FAILED call was still made by THIS model: its identity is
        # recorded too, so an evaluation of a failed prediction can tell
        # which model produced it.
        for key, value in self.model_identity.items():
            trace.model_info.setdefault(key, value)
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
        # An online capability-gain STANCE is archived as a trace so the real
        # execution and its later effect can be followed up, and so the
        # offline stage-2 evaluator can read it by id. The trace is metadata
        # only: it never changes the prediction, never ranks the candidate
        # and never touches the capability evidence. A trace is written for
        # every ASSESSED block (a stated "none" is a real, falsifiable
        # stance that must not vanish) and for every CLAIMED block (an
        # unassessed old payload that claims a gain still has something to
        # follow up). Best-effort — a trace failure must never fail the
        # prediction that was already saved.
        gain = getattr(prediction, "capability_gain", None)
        if gain is not None and (gain.assessed or gain.claimed):
            try:
                from or_harness.world_model.trace_archive import (
                    archive_capability_gain,
                )
                archive_capability_gain(self._harness(), prediction)
            except Exception:  # noqa: BLE001
                pass

    def _harness(self):
        """The harness whose store this service writes to.

        The service is constructed with an explicit store and an optional
        ``harness`` handle; the trace archive needs the latter (it reads the
        action log to follow a claim). A service built without one simply
        skips the trace.
        """
        return getattr(self, "harness", None)

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
