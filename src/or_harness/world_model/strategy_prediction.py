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

from or_harness.core.schema import (
    COST_DIMENSIONS, PREDICTED_COST_DIMENSIONS, CostVector, ProblemProfile,
    group_key,
)
from or_harness.core.schema import (
    is_finite_number as _finite,
    is_probability as _prob,
)
from or_harness.world_model.contracts import (
    FAITHFUL_G_CONVENTION,
    FAITHFUL_G_METRIC,
    FAITHFUL_G_OBSERVATION_VERSION,
    COST_PREDICTION_CONTRACT_VERSION,
    RETRIES_FUTURE_VERSION,
    is_g_probability,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityGain,
    EvidenceRef,
    ExpectedCost,
    INTERVAL_KINDS,
    PredictionTrace,
    normalize_text_list,
    R_AGGREGATION_NOTE,
    R_EVENT_DEFINITIONS,
    R_EVENTS,
    R_SCORING_VERSION,
    is_risk_event_probability,
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

#: The three R events, rendered from the ONE shared definition table so the
#: prompt the model reads and the label the framework records cannot drift.
_R_EVENT_PROMPT_BULLETS = "".join(
    "    * " + _name + " — " + R_EVENT_DEFINITIONS[_name] + ";\n"
    for _name in ("timeout", "environment_failure",
                  "implementation_failure"))

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
    "3. Prediction fields are OPTIONAL except capability_gain.decision, "
    "which must follow the binary H+ rules below. If you lack evidence for "
    "an optional field, OMIT it entirely — never write null, \"unknown\", -1 or a guess as a "
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
    "Fields (with the binary H+ requirements stated below):\n"
    "- benefit: object. {\"kind\": one of effective_completion|"
    "solution_quality|valid_progress|correct_infeasibility_diagnosis, "
    "\"metric\": a short name of WHAT is measured (required when benefit "
    "is present), \"unit\": the unit of the metric, \"value\": number — "
    "for kind=solution_quality the value IS the QUALITY Q = max(0, 1 - gap), "
    "NOT the gap: optimal (gap=0) means Q=1.0 and a wide gap means Q near 0. "
    "Q is NOT the raw objective. Do NOT write 0 for an expected-optimal run. "
    "\"interval\": [lo, hi], \"interval_kind\": one of outcome|mean (WHAT "
    "the range is about — \"outcome\" = this ONE run's value, \"mean\" = "
    "the average over repeated runs), \"interval_coverage\": the nominal "
    "coverage level you claim (e.g. 0.9), \"baseline\": {\"kind\": one of "
    "no_knowledge|conditional_stats|current_entry|current_solution|"
    "declared|unknown, \"value\": number, \"note\": text}, \"feasible\": "
    "boolean, \"notes\": [text]}. A value with no baseline is invalid.\n"
    "  BENEFIT CONVENTION: the request carries a `benefit_convention` "
    "block naming the ONE yardstick this decision uses. Predict under it. "
    "The PRIMARY benefit is `kind=effective_completion` with "
    "`metric=task_check_passed_given_faithful_execution`: the probability "
    "that the current Agent and Harness, faithfully following the candidate "
    "steps under the agreed conditions, effectively complete and sufficiently "
    "verify the original task. Task understanding, modelling, solving and "
    "verification ability all affect G. Faithful means the code follows the candidate steps; "
    "constraints omitted by the METHOD remain omitted in this assumption. "
    "Focus on the original requirements, candidate steps and expected "
    "result. Before estimating G, identify the candidate's decisive premises: "
    "as relevant, variable domains, training or delivery timing, resource "
    "availability, billing units and constraint scope. Check whether each "
    "is supported by the task or reliable supplied knowledge, conflicts "
    "with the task, or remains uncertain; trace its effect through the "
    "candidate steps to the expected result. Historical cases support "
    "mechanisms and boundaries, not a high G merely because similar tasks "
    "passed. If the supplied material includes multiple candidates, identify "
    "their outcome-relevant differences; with one, assess it alone. Do not "
    "request extra candidates or calls. When evidence cannot distinguish "
    "candidates, allow ties and uncertainty without manufacturing probability "
    "differences or a general upward/downward G adjustment. A modelling "
    "premise that misrepresents the task affects G; it does not itself raise "
    "technical implementation risk R. Whether a script can be implemented successfully belongs to "
    "implementation risk, not this conditional G. Pure code/API failures and "
    "harness wall-clock interruption belong to R; do not also lower G for the "
    "same purely technical hazard. An invalid mathematical model, an unsafe "
    "reduction or missing solution-quality guarantee affects G. A declared "
    "time-limited approximation may affect G through the quality delivered on "
    "normal completion; mathematical correctness alone does not imply G=1. "
    "Under wm-g-faithful/2, a faithful evaluable run is G=1 only with sufficient "
    "credible original-task verification. Failure to complete or verify, "
    "partial coverage, circular evidence and unresolved trusted-reference "
    "conflicts are G=0 even when the cause is unknown; this does not prove "
    "the mathematical method wrong. Unknown means no evaluation opportunity, "
    "unestablished fidelity, missing execution/check facts or attribution, "
    "external disruption, or a defective evaluation standard. Verification "
    "insufficiency is an observed capability limit, not missing opportunity. "
    "`kind=solution_quality` with "
    "`metric=normalized_objective_gap` measures how well the SOLVER solved "
    "the MODEL it was given — that is a SUPPORTING fact, NOT the same as "
    "completing the task: a solved model whose answer does not satisfy the "
    "task is NOT an effective completion. Keep feasibility, "
    "solution quality and optimality as SEPARATE supporting facts; never "
    "treat 'not proven optimal' as 'no benefit'. These measures are never "
    "interchanged. Use EXACTLY the framework's required kind, metric, unit "
    "and scope. A mismatch or non-finite/out-of-range G is unavailable. "
    "In benefit.notes write ONE evidence-grounded sentence linking the "
    "decisive premise, its evidence, candidate steps and expected consequence. "
    "If evidence is insufficient, name the missing condition. Preserve "
    "this sentence even when no numeric G is justified. Example: if the task "
    "excludes coaches from production throughout two weeks of training, "
    "deducting capacity only in the start week may overstate second-week "
    "capacity despite a normal optimal solve; deducting both weeks matches "
    "that premise. This illustrates a task-dependent check, not a fixed score. "
    "The main agent decides how to revise the method.\n"
    "  INTERVAL SEMANTICS: an interval must say WHAT it is about. "
    "`interval_kind=\"outcome\"` covers ONE execution's observed value; "
    "`interval_kind=\"mean\"` covers the average over repeated runs. A "
    "success PROBABILITY (kind=effective_completion) is a mean over runs, "
    "NOT a single 0/1 label — do not wrap it in a narrow `outcome` interval "
    "and do not present a narrow probability range as if it bounded one "
    "run. Reporting an interval without a kind leaves the semantics "
    "unstated, and the framework will say so.\n"
    "- cost: object of the FOUR active remaining-task resource dimensions "
    "you have evidence for, each a non-negative number: "
    "{remaining_latency_s, output_tokens, tool_calls, retries}. Include ONLY "
    "dimensions you actually predict; an omitted dimension is unknown, never "
    "zero. output_tokens may also be null for unknown; put its reason in "
    "unsupported_fields, outside cost.\n"
    "  COST SCOPE (all candidates of one decision share ONE anchor — the "
    "decision point — and ONE end — the end of the task): every active "
    "dimension covers the SAME remaining task window. C predicts the "
    "resources still needed to FINISH THE TASK, not one sandbox attempt.\n"
    "    * `remaining_latency_s` is the PRIMARY latency: the wall-clock from "
    "THIS decision's anchor to the END of the task, covering the later "
    "modelling, coding, tool calls, solving, checking, repair and retries;\n"
    "    * `output_tokens` is the number of tokens the EXECUTION MODEL is "
    "expected to GENERATE from THIS decision point until the task ends, "
    "including later modelling, writing code, self-checks and expected "
    "rework/retries. The task text and its length are already known input "
    "conditions, not generated tokens. Exclude generation before this "
    "decision. Raw usage/budget `llm_tokens` still means input plus output "
    "and must not substitute for output_tokens;\n"
    "    * `tool_calls` is the expected number of tool calls in the SAME "
    "remaining task window, not a whole-task cumulative total and not one "
    "script run;\n"
    "    * `retries` is the expected count of additional corrective/retry rounds STARTED "
    "from this decision anchor until task termination, including later strategy changes. "
    "Initial implementation is not a retry. Count actual substantive repair/reimplementation "
    "after a defect, failed execute or failed acceptance, including environment/framework "
    "rework. Do not count ordinary API probes, planned algorithm substeps, successful "
    "checks, duplicate logs, individual edits within one round, or errors without started "
    "correction. Failure, local repair and formal resubmission in one continuous round "
    "count once. Use candidate complexity, API and variable-domain compatibility, reusable "
    "code, known conditions, related failure/repair facts, this Agent's implementation "
    "experience and foreseeable diagnostics/rework. Use a nonnegative expected count, "
    "including fractions or zero, or omit when unknown; no fixed 0.1 default, floor or "
    "solver-specific preset. Formal execution retry limits remain separate.\n"
    "  DO NOT predict `latency_s` or `solver_runtime_s`: they are "
    "EXECUTION OBSERVATIONS (one attempt's sandbox wall-clock and the "
    "solver's inner runtime), not remaining-task costs. They are never "
    "renamed into `remaining_latency_s`, never added to it, and a new "
    "prediction containing either is rejected as a contract error. Historical "
    "observations of them remain visible only as execution diagnostics. "
    "Do not mix scopes. When a number does NOT measure the whole remaining "
    "task, state which scope it measures in `unsupported_fields` — NEVER add "
    "a justification inside `cost`. The `cost` object must contain ONLY the "
    "four numeric dimension keys listed above: a key such as `basis` or "
    "`notes` is NOT a dimension and invalidates the WHOLE prediction.\n"
    "- risk: object {\"events\": [{\"event\": a short name of the risk "
    "EVENT, \"probability\": number in [0,1] or omitted when you have no "
    "basis, \"severity\": number or omitted, \"severity_unit\": text, "
    "\"basis\": text, \"notes\": [text]}]}. Risk events are LOSSES separate "
    "from resource counts; R labels the defect event while C separately counts started corrective rounds.\n"
    "  OBSERVABLE EVENT NAMES (use EXACTLY one of these when it applies — "
    "the framework has a channel that can confirm or refute it, and an "
    "event outside this list can never be scored):\n"
    + _R_EVENT_PROMPT_BULLETS +
    "    * solver_reported_infeasible — the solver would report the model "
    "infeasible (a VERDICT about the model as given, not by itself a "
    "strategy failure);\n"
    "    * task_check_failed — the answer would FAIL the declared "
    "task-result check (this is the check's outcome, NOT a verdict that "
    "the model was wrong);\n"
    "    * budget_exhausted — the episode's declared budget would be "
    "exceeded. Its RANGE is the WHOLE remaining episode: from this "
    "decision's anchor, adopting the candidate and continuing to solve, "
    "check, repair and retry until the solving episode ends — whether the "
    "EPISODE budget is exceeded or a hard budget limit stops it. This is "
    "NOT about one attempt, NOT the candidate's own execution scope, and "
    "NOT a claim that the candidate alone caused the whole episode's spend. "
    "Any attempt/strategy-window candidate may predict it. Judge it only "
    "from the EPISODE-scoped condition in `execution_constraints."
    "budget_condition` (a per-attempt latency overrun is NOT this event). "
    "When no valid episode-budget condition exists, or the remaining budget "
    "is not confirmed, you may DECLINE to predict it (omit probability) — "
    "never default it to 0, and treat a known consumption with an unmeasured "
    "declared dimension as a LOWER BOUND, not a confirmed remaining "
    "budget.\n"
    "  SCORING R CONVENTION: R is the MAXIMUM single-event probability common to every ranked candidate; if "
    "no event is common, omit R for all. NOT a joint failure probability: never sum or combine events. "
    "Consider ALL THREE explicitly: finite JSON [0,1], including 0, or omit with a reason. Missing, duplicate"
    " or invalid probabilities stay unknown, never coerced or averaged.\n"
    "  Implementation covers actual selection through implementation/execution/necessary technical "
    "verification completion or the next formal decision, whichever comes first. Repaired local/internal "
    "defects count. Exclude preselection work, later candidates, routine probes, expected exceptions, duplicate logs "
    "and unverified suspicions. One evidenced substantive defect supplies 1; 0 needs actual implementation "
    "and complete coverage, never final execute success alone. Timeout/environment retain bound formal "
    "scopes; other events stay diagnostics. G is acceptance given faithful execution; C counts rework "
    "resources.\n"
    "  Prefer interface/method-similar cases in implementation_r_evidence; numeric feedback requires matching"
    " versions. global_event_base_rates uses known labels only; global_prediction_performance is background, "
    "not a candidate probability. A tested script proves technical execution, not business correctness; "
    "absent historical failures do not prove zero risk.\n"
    "  CANDIDATE COMPLEXITY AND TIMEOUT: assess the actual algorithm, variable domains, "
    "discrete choices and constraint coupling, not just the number of variables or solver "
    "name. For enumeration, correctly count or bound the unpruned search space from domain "
    "cardinalities before crediting pruning, preprocessing, decomposition or structural "
    "reductions actually supported by the candidate. Few variables can have a huge product "
    "of domain sizes. State which bounds are conditional; include per-node work and the "
    "implementation (including branching/search or branch-and-bound). A combinatorial upper "
    "bound alone does not establish inevitable timeout. Relate the supported reduction, "
    "algorithm/solver, environment and comparable executed history to the formal harness "
    "wall-clock limit in execution_constraints.executor.timeout_seconds; distinguish an "
    "inner solver time limit that may still deliver a usable result. "
    "Give candidate-specific probabilities only with evidence for either low or high risk; "
    "do not copy fixed default rates or a solver-reputation penalty. Near-zero risk is "
    "allowed for a demonstrably small, bounded computation; absent timeout history alone "
    "does not justify zero. If evidence is insufficient, omit probability and explain why "
    "under the existing unknown-field contract; do not invent precision or impose a risk floor. "
    "R.timeout remains interruption of the bound formal execution before a usable result. "
    "Agent analysis, probing, code writing and repeated tool calls are not timeout labels; "
    "their elapsed time belongs to C.remaining_latency_s, not a new task-timeout event.\n"
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
    "- capability_gain: object. Set decision to JSON number 0 or 1 in THIS SAME prediction: the frozen "
    "binary H+ induction switch, separate from G/C/R utility. Predict whether learning from this "
    "candidate could improve a future Agent's decisions, not a guarantee of publishable knowledge.\n"
    "H+ DECISION PROCEDURE (in order, for THIS candidate):\n"
    "1. GENERALIZE FIRST. Imagine a future task with similar mathematical structure or operational "
    "requirements but different instance details. Start from the current Harness's visible published "
    "strategic knowledge and identify which concrete decisions could still be improved. Compare an Agent "
    "equipped with this existing knowledge against the same Agent additionally receiving a reusable rule "
    "distilled from this execution. Treat existing rules as scoped guidance, not proof that the Agent has "
    "mastered the entire method or domain. What concrete decision could it make better, under what "
    "conditions, and why would the difference matter? Identify a concrete problem-understanding, "
    "modelling, solving, implementation or verification decision affecting feasibility, optimality or "
    "solution quality, search space, computational work or reliability. When relevant executions exist, "
    "align their mathematical or operational roles with the current task. Abstract case-specific entities "
    "by role and capacities or coefficients as parameters. Different domains or family labels do not rule "
    "out a shared mechanism; similar solver/algorithm names or terms do not establish one.\n"
    "2. STATE THE LEARNING HYPOTHESIS. State a possible When/Do/Expect rule: which future decision this "
    "mechanism could improve, why it might work, and how the Agent would act differently. Future premises "
    "must be recognizable before solving, without a hidden answer. Consider semantic disambiguation "
    "grounded in original task inputs, a classical method's structural applicability conditions, "
    "reductions, decomposition, valid bounds, a failure-and-repair implementation technique, purposeful "
    "combinations or reasonably foreseeable checking, diagnosis and verification. For example, deriving a "
    "tight activation bound from resource rows can prevent an unnecessarily weak formulation on other "
    "tasks; distinguishing per-unit from one-time charges can prevent a wrong objective; enumerating few "
    "opening subsets with residual transportation LPs can provide an independent check. These are "
    "examples of reuse value, not automatic positives. A consequential, testable interpretation or repair "
    "is not mere translation. This is prospective generalization, not completed induction. Independently "
    "assess each candidate: a shared understanding or verification step can support the same hypothesis "
    "for multiple candidates when each can provide evidence. Do not borrow another candidate's H+, merge "
    "unselected candidates' learning value or invent unrelated execution branches; never use "
    "post-execution facts to revise this forecast.\n"
    "3. CHECK THE ACTUAL KNOWLEDGE GAP. The comparison is against the current Harness's explicit strategy "
    "library, NOT the model's pretrained knowledge or novelty to OR research. A familiar method can "
    "improve future decisions if its conditions, use or checks are made explicit; an unfamiliar method "
    "need not do so. ONLY a currently visible PUBLISHED strategy rule can establish existing-knowledge "
    "coverage. Compare When (conditions, mathematical structure and prerequisites), Do (core mathematical "
    "or operational mechanism), and Expect (effect, guarantee or verification purpose). All its "
    "substantive content must be covered. Ask whether the proposed rule offers a concrete decision "
    "improvement, a broader applicability scope, a correction, or stronger supporting evidence beyond the "
    "existing published knowledge. Judge only the knowledge visible in this request; missing retrieval "
    "does not prove global absence. Execution Evidence records what happened; Strategic Knowledge records "
    "the transferable mechanism. Successful use is not proof of saved knowledge. Execution Evidence is "
    "induction material, not completed induction. If earlier executions support an unsaved mechanism, "
    "treat them as support for review, not a reason to reject it. Partial coverage can leave a learning "
    "opportunity, including scope extension, correction or counterexample, or meaningful independent "
    "validation of an existing weakly supported mechanism. A previous H+ judgement (positive or negative) "
    "is not evidence of rule coverage; use supported review feedback below.\n"
    "4. CHECK EVIDENCE AND DECIDE. Can this candidate reasonably support, refine or challenge the rule "
    "through execution or verification? Do not require advance proof of the mechanism or transferability, "
    "multiple independent executions, or prior failure; publication standards remain the offline review's "
    "responsibility. Return 1 when the rule could improve a concrete future decision and this run offers "
    "a reasonable evidence opportunity not already satisfied by the library. Return 0 if no concrete "
    "reasonable learning hypothesis exists, this run cannot reasonably investigate it, or a visible "
    "published rule fully covers the mechanism and this run offers no meaningful new evidence. Use 0 for "
    "mere task translation, ordinary solver use or routine checking when no specific transferable "
    "mechanism or finding is identified. A known technique is not by itself a reason for 0; an empty "
    "library or missing review history is NEVER sufficient for 1 and is not itself a reason for 0.\n"
    "OUTPUT: For 1, claim states one concise potential reusable mechanism and the future decision it "
    "could change. Basis explains the concrete improvement over not receiving the rule, with the same "
    "existing strategic knowledge on both sides, task-structural grounds, coverage and, when relevant "
    "history contributes, the shared mathematical or operational relationship. Keep verification_conditions:[{condition: text, check_basis: text}] to the minimum "
    "feasible check that could support, refine or challenge that mechanism in this run. Reuse self-check "
    "evidence: no default full re-solve, large experiment or extra execution branch. Affordable "
    "enumeration, original-versus-reduced-method comparison or targeted reproduction is allowed. Use "
    "applies_to for future premises and expected_changes only for meaningful metrics in their original "
    "units. For 0 give the candidate-specific reason in basis and OMIT claim: name the decision "
    "considered and why no reusable mechanism or feasible evidence was found, or cite the visible rule "
    "ID/version and compare When/Do/Expect and explain the absence of new evidence. A 1 ALWAYS causes one "
    "review after a bound real attempt, including failures or unfinished checks; 0 skips induction. "
    "Record checks and learning in the factual receipt without changing the frozen switch. A refuted "
    "hypothesis may yield a useful boundary rule. Never substitute legacy "
    "expected/none/insufficient_basis for decision. Missing/illegal decision schedules 0 with a separate "
    "diagnostic.\n"
    "CALIBRATION EVIDENCE: the context may carry a "
    "`strategy_outcome_calibration` block describing how THIS model's past "
    "predictions turned out. Read its definitions and support carefully; "
    "global benefit residuals are diagnostics and supply no G adjustment:\n"
    "  * it is grouped by (model, metric, unit, scope). Only groups whose "
    "model/metric/unit/scope MATCH the metric and scope you are predicting "
    "under are your evidence. A group with a different metric, unit or "
    "scope is NOT comparable to what you are about to predict — do not cite "
    "it as support for this number.\n"
    "  * `mean_benefit_signed_error` is directed: POSITIVE means the real "
    "outcome was historically BETTER than predicted (this model "
    "under-predicts benefit); NEGATIVE means it over-predicts. Neither sign "
    "is an instruction to change this candidate's G.\n"
    "  * global cost residuals are audit diagnostics, never multipliers "
    "or instructions to raise/lower C. Use the `cost_evidence` block: its "
    "`cases` are LOCAL similar historical executions whose reported cost "
    "covers the SAME remaining task (never one sandbox attempt), each with "
    "whatever of the four dimensions was actually observed and a "
    "`missing_dimensions` list; its `global_cost` block gives the pre-decision "
    "n/p50/p90 per dimension plus prediction-vs-actual error direction and "
    "rank correlation. The local cases are the numerical reference; the "
    "global distribution is BACKGROUND. Do not raise every cost by the global "
    "error ratio, and do not omit a comparable local number. C no longer uses "
    "any solving phase: judge the remaining work from the candidate plan, the "
    "current artifacts and these comparable costs.\n"
    "  * `basis: insufficient_evidence` means too few independent episodes "
    "to say anything: do NOT treat such a group as calibration.\n"
    "  * these are measured PAST errors, not a promise about this "
    "prediction, and they carry NO per-strategy breakdown (they are a "
    "global diagnostic).\n"
    "  * `prediction_context.g_candidate_evidence` carries the LOCAL "
    "historical cases recalled as most similar to THIS candidate's method and "
    "conditions by the embedding retrieval (not by shared keywords). Its "
    "`relevance` code says WHY a case was kept: `same_task` (the same task), "
    "`similar_method` (the candidate-method similarity reached its own "
    "threshold), `similar_conditions` (same structural cell) or "
    "`task_similar` (only general text similarity — the method was NOT shown "
    "similar). Read `method` and the observed G label together; a generic "
    "word overlap is never a method match. `g.observed` is the recorded label "
    "and null means UNKNOWN — an execution error is not a G modelling "
    "failure. `recorded_cause` appears only when the close-out attribution "
    "recorded it; never invent a cause from a failed check or execution "
    "status. `global_history` (n observed/tasks/episodes, MAE, Brier, AUC, "
    "bias direction, reliability bins) is a diagnostic of this model's past "
    "global error, NOT a candidate-specific prior and NOT a G correction.\n"
    "PAIRED FEEDBACK: the context may carry a `prediction_execution_pairs` "
    "block: the compact, per-episode facts of how THIS model's past "
    "predictions turned out. Each pair names the conditions, the planned "
    "method, the ORIGINAL predicted value, the REAL observation, the "
    "per-field difference (`signed_error` positive = the real outcome was "
    "BETTER than predicted; `cost_log_ratio` positive = real cost was "
    "HIGHER), the task-check outcome and any not-comparable reasons. Read "
    "them as EVIDENCE about your own past accuracy under similar "
    "conditions — successes AND failures are included, and a cross-cell or "
    "cross-strategy-name case is a case to inspect. Global G residuals are "
    "diagnostic only: never turn their direction or a single large G error "
    "into an instruction to raise or lower G. Use attributable conditional "
    "facts and same-task comparisons within their stated support. When a "
    "past prediction differs from execution, use the cited evidence to "
    "locate the affected premise, step or condition. Distinguish confirmed "
    "causes from hypotheses and unknowns; optimal status, a self-authored "
    "check passing or a reference mismatch alone does not establish semantic "
    "correctness or a failure cause. When a "
    "pair's field is unknown or not comparable, that does NOT invalidate "
    "its other fields. Do NOT read a pair as a menu: it is a record of what "
    "was predicted and what happened, never a recommendation. The pairs "
    "shown were SELECTED as the most relevant to THIS candidate and problem "
    "(each carries `selected_reason` and a `relevance` rank; a low rank is "
    "relevance to you, NOT a claim that two methods are the same), and are "
    "capped for display — `n_pairs_available` / `n_pairs_included` / "
    "`n_pairs_omitted` tell you how many existed. A pair's predicted risk "
    "probabilities live ONCE in `risk_predicted`; `risk_actual` gives the "
    "OBSERVED label per event and points back with `probability_ref` (a "
    "`predicted: false` event had NO predicted probability — never read "
    "that as zero).\n"
    "RETRIEVAL EVIDENCE: `retrieval_evidence.hits` carries each memory "
    "item's content ONCE (a verified claim's boundary, method and "
    "verification scope; an execution's observed outcome and cost). The "
    "`structural` and `semantic` lists are INDEXES: they keep their own "
    "signal (applicability for structural, `similarity` for semantic) and "
    "point at the content (`knowledge_ref` / `claim_ref` / `full_text_ref`) "
    "rather than repeating it. A similarity is a DISCOVERY signal only, "
    "never a quality or risk estimate.\n"
    "PREDICTION REMINDERS: C/R reminders retain their own applicability. "
    "They do not supply a G correction. Conditional G statistics report "
    "passes, failures, unknowns and independent episodes; sparse support "
    "does not establish a general probability. Same-task method outcomes "
    "are observed cases, never unexecuted counterfactuals; two passes on "
    "the same binary checker are a tie on acceptance.\n"
    "CAPABILITY-GAIN FOLLOW-UP: binary hplus_feedback contains relevant settled training receipts. "
    "Compare the frozen hypothesis, actual facts, supported conclusion and evidence scope. "
    "An agent-supported substantive increment is positive. A well-supported covered/no-increment "
    "conclusion is negative evidence: apply it only to a specifically comparable mechanism "
    "and its premises. "
    "Treat insufficient evidence, unfinished induction or unobserved work, including an "
    "execution failure that prevented verification, as unknown about the hypothesis, not a "
    "refutation. Do not globally tighten H+ from such receipts; empty feedback does not imply "
    "no learning opportunity. Publication and retirement alone do not determine learning labels. "
    "Legacy feedback retains its original delayed-effect meaning: pending is unobserved, "
    "and effect_verified refers to the legacy later-performance check. Do not mix these "
    "labels with the new induction-increment labels.\n"
    "Do NOT fabricate evidence. If the context contains no relevant "
    "experience or knowledge for the candidate, say so in "
    "unsupported_fields and leave unsupported numeric estimates omitted. "
    "This omission rule does NOT waive capability_gain.decision: still return the JSON number "
    "0 or 1 under the H+ rules above. An empty history or hplus_feedback does not establish "
    "rule coverage or rule out a concrete, testable hypothesis from this task and candidate. Output the "
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
    **FAITHFUL_G_CONVENTION,
    "required": True,
    "scope": "the prediction's own candidate scope",
    "note": ("probability the current Agent effectively completes and sufficiently verifies "
             "the original task GIVEN faithful execution of the candidate steps; "
             "Agent verification insufficiency is a negative observation, while purely "
             "technical failure without an evaluable result belongs to R"),
}

#: The COST span every candidate predicts under by default: from the
#: decision's anchor to the END of the task (see ``COST_SCOPE_CONVENTION``).
COST_SCOPE_DEFAULT = "remaining_to_task_end"

#: The SHARED cost scope every candidate of one decision predicts under.
#: ONE anchor (the decision point) and ONE end (the end of the task), so a
#: candidate cannot silently measure a different span from its peers. C has
#: exactly FOUR active prediction targets, all covering that remaining span:
#: remaining latency, generated output tokens, tool calls and future retries.
#: ``latency_s`` (one sandbox attempt) and ``solver_runtime_s`` (the solver's
#: inner runtime) are EXECUTION OBSERVATIONS kept for diagnostics — they are
#: never predicted, renamed, or added into the remaining span.
COST_SCOPE_CONVENTION: Dict[str, Any] = {
    "anchor": "decision_point",
    "end": "end_of_task",
    "prediction_dimensions": list(PREDICTED_COST_DIMENSIONS),
    "primary_dimension": "remaining_latency_s",
    "token_dimension": "output_tokens",
    "remaining_latency_definition": (
        "wall-clock from this decision to task termination, covering the "
        "later modelling, coding, tool calls, solving, checking, repair and "
        "retries"),
    "token_definition": ("tokens generated by the execution model from this "
                         "decision to task termination; observed from host "
                         "completion_tokens, excluding input tokens"),
    "tool_calls_definition": ("tool calls within the SAME remaining task "
                              "window, not a whole-task or single-attempt "
                              "total"),
    "retries_convention": {
        "observation_rule_version": RETRIES_FUTURE_VERSION,
        "baseline": "this prediction decision anchor; normal initial implementation is excluded",
        "definition": "expected distinct corrective/retry rounds started from decision anchor through task termination, including later strategy rework",
        "unit": "expected count; non-negative fractions and zero are valid",
    },
    "covers": ("the later modelling, coding, tool calls, solving, checking, "
               "repair and retries until the task ends"),
    "shared_by": "every candidate of one decision (same anchor, same end)",
    "observation_only_dimensions": {
        "latency_s": ("ONE attempt's sandbox wall-clock; an execution "
                      "observation, NOT a prediction target and never "
                      "renamed into remaining_latency_s"),
        "solver_runtime_s": ("the solver's own inner solve time; an "
                             "execution observation/diagnostic, NOT a "
                             "separate C prediction target"),
        "llm_tokens": ("raw input-plus-output accounting total, not the "
                       "generated output_tokens C predicts"),
    },
    "note": ("all four active dimensions cover the SAME remaining scope; "
             "overlapping prediction ranges are never summed. A missing "
             "dimension is unknown, never zero. Historical latency_s/"
             "solver_runtime_s observations stay readable for execution "
             "diagnostics but do not enter C references, normalization, U or "
             "the new C calibration."),
}

def build_strategy_outcome_request(
        context: Any, candidate: CandidateRef,
        *, benefit_baseline_hint: Optional[Dict[str, Any]] = None,
        benefit_convention: Optional[Dict[str, Any]] = None,
        provider: Any = None
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
    comparison will use. Without a declaration the framework fixes the new faithful-execution
    convention before calling the provider.

    The context's calibration block is FILTERED before it is sent: only the
    groups whose SCOPE matches this candidate's scope are kept, because a
    window-scope error statistic is not evidence about a single attempt (and
    the reverse). The filtering is reported on the request, so a reader can
    see that a group was withheld rather than absent.
    """
    from or_harness.world_model.planner import (
        normalize_benefit_convention, KNOWN_BENEFIT_CONVENTIONS,
    )
    declared = dict(benefit_convention or BENEFIT_CONVENTION)
    resolved = normalize_benefit_convention(declared.get("kind"), declared.get("metric"))
    if resolved is None:
        raise ValueError("unsupported benefit convention: declare a registered kind/metric")
    known = KNOWN_BENEFIT_CONVENTIONS[resolved]
    if declared.get("unit") not in (None, "", known["unit"]):
        raise ValueError("benefit convention unit disagrees with the registered metric")
    declared.update(kind=resolved[0], metric=resolved[1], unit=known["unit"])
    declared.setdefault("observed_from", known["observed_from"])
    if resolved[1] == FAITHFUL_G_METRIC:
        if declared.get("observation_rule_version") not in (None, FAITHFUL_G_OBSERVATION_VERSION):
            raise ValueError("unsupported faithful G observation rule")
        declared["observation_rule_version"] = FAITHFUL_G_OBSERVATION_VERSION
    declared_kind = declared.get("kind")
    declared_metric = declared.get("metric")
    fixed_convention: Optional[Dict[str, Any]] = None
    if declared_kind and declared_metric:
        fixed_convention = {
            "kind": str(declared_kind),
            "metric": str(declared_metric),
            "unit": declared.get("unit"),
            "scope": candidate.scope,
            "observation_rule_version": declared.get("observation_rule_version"),
            "observed_from": declared.get("observed_from"),
            "source": declared.get("source", "declared"),
            "required": True,
            "note": ("THIS DECISION compares benefit under this convention. "
                     "Predict this candidate's benefit with EXACTLY this "
                     "kind and metric: a different kind or metric cannot be "
                     "ranked against the other candidates, and it is NOT "
                     "silently converted"),
        }
    from or_harness.world_model.context import PredictionContext
    from or_harness.world_model.g_signal import select_conditional_g, requested_solving_conditions
    view = (context.provider_view(g_candidate=candidate, g_convention=fixed_convention)
            if isinstance(context, PredictionContext) else context.provider_view())
    if not isinstance(context, PredictionContext):
        view["strategy_outcome_calibration"] = select_conditional_g(
            view.get("strategy_outcome_calibration"),
            profile=(view.get("joint_problem") or {}).get("profile") or {},
            conditions=requested_solving_conditions(candidate, view.get("execution_constraints")),
            convention=fixed_convention, scope=candidate.scope, task_id=candidate.task_id)
    from or_harness.world_model.contracts import R_CONVENTION
    request: Dict[str, Any] = {
        "hplus_convention": {"version": "wm-hplus-binary/1", "field": "capability_gain.decision",
                             "values": [0, 1], "missing_or_invalid_schedule": 0,
                             "role": "frozen induction switch; never part of solving utility"},
        "risk_convention": {
            **copy.deepcopy(R_CONVENTION),
            "event_definitions": dict(R_EVENT_DEFINITIONS),
            "interpretation": R_AGGREGATION_NOTE,
        },
        PROTOCOL_REQUEST_KEY: STRATEGY_OUTCOME_PROTOCOL_VERSION,
        "request_kind": "strategy_outcome_prediction",
        "prediction_context": view,
        "candidate": candidate.to_dict(),
        # The SHARED benefit convention: the framework fixes the yardstick
        # so every candidate of one decision is predicted on the SAME
        # currency, and the comparison and the close-out read the same
        # definition. A model that deviates is reported (its prediction is
        # retained in diagnostics, but rejected G numbers are unavailable).
        "benefit_convention": (fixed_convention
                               if fixed_convention is not None
                               else copy.deepcopy(BENEFIT_CONVENTION)),
        # The SHARED cost scope every candidate of this decision predicts
        # under: ONE anchor (the decision point) and ONE end (the end of the
        # task). Travels in the request so the model cannot pick a different
        # anchor per candidate, and so a reader sees which span a predicted
        # cost measures. The framework measures the real remaining span at
        # close-out; this block only FIXES the definition.
        "cost_scope": copy.deepcopy(COST_SCOPE_CONVENTION),
        "output_contract": {
            "allowed_fields": ["benefit", "cost", "risk", "uncertainty",
                               "capability_gain", "evidence_basis",
                               "unsupported_fields"],
            "fixed_by_framework": ["task", "candidate", "scope", "anchor",
                                   "evidence sources", "benefit convention"],
            "note": ("the model fills prediction content only; it may not "
                     "rewrite the task, the candidate, the scope, the cost "
                     "anchor or the evidence. Use the `benefit_convention` "
                     "kind/metric and the `cost_scope` dimensions"),
        },
    }
    # Scope filtering of the calibration block (the model identity was
    # already filtered when the context was frozen, since that is decided by
    # the attached provider, not by this candidate).
    view = request["prediction_context"]
    # Defend the final boundary too: callers may supply an old/custom view.
    from or_harness.world_model.conditions import (
        filter_prediction_reminders, calibration_projection,
    )
    view["prediction_reminders"] = filter_prediction_reminders(view.get("prediction_reminders"))
    view["strategy_outcome_calibration"] = calibration_projection(view.get("strategy_outcome_calibration"))
    view["strategy_outcome_calibration"]["conditional_g"] = [
        row for row in view["strategy_outcome_calibration"].get("conditional_g", [])
        if row.get("metric") == declared_metric and row.get("scope") == candidate.scope
        and row.get("observation_rule_version") == declared.get("observation_rule_version")]
    pairs = view.get("prediction_execution_pairs") or {}
    if "same_task_comparisons" in pairs:
        pairs["same_task_comparisons"] = [c for c in pairs["same_task_comparisons"]
            if c.get("metric") == declared_metric and c.get("scope") == candidate.scope
            and c.get("observation_rule_version") == declared.get("observation_rule_version")]
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
    return _finalize_request_size(request, STRATEGY_OUTCOME_SYSTEM_PROMPT,
                                  provider=provider)


def _cap_displayed_pairs(request: Dict[str, Any]) -> None:
    """Bound the DISPLAYED prediction-execution pairs at the request level.

    The selection has already chosen the relevant, bounded set; this is the
    final call-level safeguard: if a caller supplied a block with MORE pairs
    than the configured display ceiling (a legacy context, or a direct
    caller), the sent list is truncated to the ceiling and the omission is
    REPORTED on the block. It is a DISPLAY ceiling, never a calibration
    window — the counts stay, so a reader never mistakes the shown pairs for
    the calibration population.
    """
    from or_harness.world_model.episode_closeout import (
        _paired_feedback_max_pairs,
    )
    context = request.get("prediction_context")
    if not isinstance(context, dict):
        return
    block = context.get("prediction_execution_pairs")
    if not isinstance(block, dict):
        return
    pairs = block.get("pairs")
    if not isinstance(pairs, list):
        return
    cap = _paired_feedback_max_pairs()
    if len(pairs) <= cap:
        return
    block["pairs"] = pairs[:cap]
    from or_harness.world_model.g_signal import case_comparisons_for_rows
    block.update(case_comparisons_for_rows(
        block.get("same_task_comparisons") or [], block["pairs"], cap))
    block["n_pairs_included"] = len(block["pairs"])
    block["display_capped"] = {
        "ceiling": cap,
        "dropped_for_display": len(pairs) - cap,
        "note": ("the displayed pairs were capped at the request level "
                 "(a DISPLAY ceiling, not a calibration sample size); the "
                 "counts above still describe the selection"),
    }


def _finalize_request_size(request: Dict[str, Any],
                           system_prompt: str,
                           *, provider: Any = None) -> Dict[str, Any]:
    """Measure the WHOLE request and enforce the deployment's input bound.

    The context's own ``_bound_request`` measures only the ``prediction_context``
    fragment, so a request could stay "within bound" while the CANDIDATE, the
    system prompt and the output contract pushed the real total far over.
    This final check measures system prompt + full JSON body (with the
    provider's real tokenizer when it exposes one, otherwise an explicitly
    labelled conservative estimate that counts CJK/JSON-dense text more
    heavily than ``chars / 4``).

    When over budget it reduces the HISTORY FEEDBACK first (per-row, then
    per-case), then the G/C/R evidence and the other non-core blocks. The
    current task, the candidate, the execution constraints and the prediction
    contract are never truncated: if the protected core alone still exceeds
    the budget, ``sendable`` is false and the service refuses to send the
    request instead of issuing an uncontrolled one.
    """
    from or_harness.world_model.request_budget import (
        enforce_full_request_budget,
    )
    _cap_displayed_pairs(request)
    request["request_size"] = enforce_full_request_budget(
        request, system_prompt, provider=provider)
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
        *, context: Any = None, provider_result: Optional[Dict[str, Any]] = None,
        benefit_convention: Optional[Dict[str, Any]] = None
        ) -> StrategyOutcomePrediction:
    """Parse a model payload into a validated StrategyOutcomePrediction.

    Malformed content is reported, never silently repaired. A G convention
    mismatch or invalid value makes that field unavailable; valid C/R and
    the benefit sentence survive. Other malformed prediction blocks keep
    their existing invalid status behaviour. A payload with NO predicted content at
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
    benefit_problem_start = len(problems)
    benefit: Optional[BenefitEstimate] = None
    benefit_notes: List[str] = []
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
        if metric == FAITHFUL_G_METRIC and value is not None and not is_g_probability(value):
            problems.append("benefit.value must be a finite JSON probability in [0,1]")
            value = None
        if value is not None and not _finite(value):
            problems.append("benefit.value is not a finite number")
            value = None
        if kind == "effective_completion" and value is not None and not is_g_probability(value):
            problems.append("benefit.value must be a finite probability in [0,1]")
        if benefit_convention:
            for key in ("kind", "metric", "unit"):
                if key == "unit" and not raw_benefit.get(key) and metric != FAITHFUL_G_METRIC:
                    continue  # old payloads could omit this optional field
                if raw_benefit.get(key) != benefit_convention.get(key):
                    problems.append(f"benefit.{key} differs from the required convention")
            if raw_benefit.get("scope", candidate.scope) != candidate.scope:
                problems.append("benefit.scope differs from the required execution scope")
            rule = raw_benefit.get("observation_rule_version")
            if rule is not None and rule != benefit_convention.get("observation_rule_version"):
                problems.append("benefit observation rule differs from the required convention")
        if kind == "solution_quality" and value is not None \
                and not (0.0 <= float(value) <= 1.0):
            problems.append(
                "benefit.value for kind='solution_quality' must be a "
                "NORMALIZED value in [0,1] (e.g. 1-gap); a raw objective "
                "value is not silently clamped — restate the metric or use "
                "a different kind")
        # GAP-vs-QUALITY CONFLICT: a solution_quality of 0.0 whose own notes
        # say the run will be OPTIMAL describes a gap of 0 (i.e. Q=1), not a
        # quality of 0 — the model wrote the GAP where the QUALITY belongs.
        # The framework does NOT quietly convert it (that would let the
        # model's number stand as the framework's): it reports the conflict
        # so the call can be re-predicted, and the original stays unchanged.
        if kind == "solution_quality" and value is not None \
                and float(value) == 0.0:
            notes_text = " ".join(str(n) for n in
                                  (raw_benefit.get("notes") or [])).lower()
            optimal_claims = ("optimal", "gap=0", "gap = 0", "gap of 0",
                              "gap 0", "zero gap", "gap=0.0")
            conflicts = any(token in notes_text for token in optimal_claims)
            if raw_benefit.get("feasible") is True and conflicts:
                problems.append(
                    "benefit.value is 0.0 for kind='solution_quality' while "
                    "the notes/feasible flag describe an OPTIMAL outcome: "
                    "the value must be the QUALITY Q = 1 - gap (optimal "
                    "=> Q=1.0), not the gap. Restate value as Q, or omit it "
                    "— the framework does not convert the number for you")
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
            benefit_notes = normalize_text_list(
                raw_benefit.get("notes"), field="benefit.notes")
        except ValueError as exc:
            # A malformed annotation field is a PROBLEM about the payload,
            # not a reason to discard the forecast's numeric content.
            problems.append(str(exc))
            benefit_notes = []
        try:
            benefit = BenefitEstimate(
                kind=kind, metric=metric or "(unnamed metric)",
                unit=str(raw_benefit.get("unit") or ""),
                value=(float(value) if value is not None else None),
                interval=interval, interval_kind=interval_kind,
                interval_coverage=interval_coverage,
                baseline=baseline, feasible=feasible,
                observation_rule_version=(FAITHFUL_G_OBSERVATION_VERSION
                                          if metric == FAITHFUL_G_METRIC else None),
                notes=list(benefit_notes))
        except ValueError as exc:
            problems.append(f"benefit: {exc}")
            benefit = None

    # A bad G does not erase good cost/risk or the model's causal sentence.
    # Standalone legacy parsing retains its original validation behaviour.
    if len(problems) > benefit_problem_start and (benefit_convention or
            isinstance(raw_benefit, dict) and raw_benefit.get("metric") == FAITHFUL_G_METRIC):
        unsupported["benefit"] = "; ".join(problems[benefit_problem_start:])
        notes.append("benefit unavailable: " + unsupported["benefit"])
        del problems[benefit_problem_start:]
        if benefit is None and isinstance(raw_benefit, dict):
            expected = benefit_convention or FAITHFUL_G_CONVENTION
            benefit = BenefitEstimate(
                kind=expected["kind"], metric=expected["metric"], unit=expected["unit"],
                observation_rule_version=expected.get("observation_rule_version"),
                notes=list(benefit_notes))
        if benefit is not None:
            benefit.value = None
            benefit.interval = None
            if benefit_convention:
                # Only an UNAVAILABLE slot adopts the request's metadata;
                # no rejected model number is converted to the new currency.
                benefit.kind = benefit_convention["kind"]
                benefit.metric = benefit_convention["metric"]
                benefit.unit = benefit_convention["unit"]
                benefit.observation_rule_version = benefit_convention.get("observation_rule_version")

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
            if dim not in PREDICTED_COST_DIMENSIONS:
                if dim in COST_DIMENSIONS:
                    problems.append(
                        f"cost.{dim} is a historical observation "
                        "dimension, not an active C prediction target; "
                        "predict only "
                        f"{list(PREDICTED_COST_DIMENSIONS)} "
                        "(C covers the remaining task, never one sandbox "
                        "attempt or the solver's inner runtime)")
                else:
                    problems.append(f"unknown cost dimension {dim!r}")
            elif dim == "output_tokens" and value is None:
                unsupported["cost.output_tokens"] = "reported as unknown by the model"
            elif not _finite(value) or float(value) < 0:
                problems.append(f"cost.{dim} must be finite and >= 0")
            else:
                dims[dim] = float(value)
        if "output_tokens" not in dims:
            unsupported.setdefault("cost.output_tokens",
                "not predicted; raw input-plus-output tokens are not "
                "generated output tokens")
        if dims:
            try:
                vector = CostVector(**dims, measured=set(dims))
            except (TypeError, ValueError) as exc:
                problems.append(f"cost: {exc}")
                cost = None
            else:
                cost = ExpectedCost(
                    expected=vector, measured=sorted(dims),
                    scope=COST_SCOPE_DEFAULT,
                    prediction_contract=COST_PREDICTION_CONTRACT_VERSION,
                    notes=["predicted by the model under the active "
                           "four-dimensional remaining-task C contract; the "
                           "measured mask marks PREDICTED dimensions, not "
                           "observed ones. scope=" + COST_SCOPE_DEFAULT +
                           " (decision anchor -> end of task): the close-out "
                           "observes the SAME span, not one attempt"])
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
        invalid_r_events: set = set()
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
                event_name = str(raw["event"])
                is_r_event = event_name in R_EVENTS
                valid_probability = (is_risk_event_probability(probability)
                                     if is_r_event else _prob(probability))
                if probability is not None and not valid_probability:
                    if is_r_event:
                        # An R event's illegal probability makes THAT event
                        # unusable (unknown) for this prediction. It is never
                        # clamped or converted, and the rest of the
                        # prediction stands.
                        invalid_r_events.add(event_name)
                    else:
                        problems.append(f"risk.events[{index}].probability must be in [0, 1]")
                    probability = None
                severity = raw.get("severity")
                if severity is not None and not _finite(severity):
                    problems.append(f"risk.events[{index}].severity must be "
                                    "finite")
                    severity = None
                try:
                    event_notes = normalize_text_list(
                        raw.get("notes"),
                        field=f"risk.events[{index}].notes")
                except ValueError as exc:
                    problems.append(str(exc))
                    event_notes = []
                events.append(RiskEvent(
                    event=event_name,
                    probability=(float(probability)
                                 if probability is not None else None),
                    severity=(float(severity)
                              if severity is not None else None),
                    severity_unit=str(raw.get("severity_unit") or ""),
                    basis=(str(raw["basis"]) if raw.get("basis") else None),
                    notes=event_notes))
            # ONE event per R name. A duplicate makes EVERY copy unusable
            # (unknown), never summed or averaged; the rest of the payload
            # is kept.
            for name in R_EVENTS:
                copies = [e for e in events if e.event == name]
                if len(copies) > 1:
                    unsupported["risk." + name] = (
                        "duplicate " + name + " events are ambiguous: their "
                        "probabilities are unusable (unknown), never summed "
                        "or averaged")
                    for event in copies:
                        event.probability = None
                elif name in invalid_r_events:
                    unsupported["risk." + name] = (
                        "probability must be a finite JSON number in [0,1]; "
                        "this event is unusable (unknown), never converted")
            if events:
                try:
                    risk_notes = normalize_text_list(
                        raw_risk.get("notes"), field="risk.notes")
                except ValueError as exc:
                    problems.append(str(exc))
                    risk_notes = []
                try:
                    risk = RiskStatement(events=events, notes=risk_notes)
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
                unc_basis = normalize_text_list(
                    raw_unc.get("basis"), field="uncertainty.basis")
                unc_notes = normalize_text_list(
                    raw_unc.get("notes"), field="uncertainty.notes")
            except ValueError as exc:
                problems.append(str(exc))
                unc_basis, unc_notes = [], []
            try:
                uncertainty = UncertaintyStatement(
                    source="model_self_report",
                    basis=unc_basis,
                    missing={str(k): str(v) for k, v in
                             (raw_unc.get("missing") or {}).items()},
                    notes=unc_notes
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
    from or_harness.world_model.contracts import R_CONVENTION
    trace.model_info["risk_convention"] = copy.deepcopy(R_CONVENTION)
    from or_harness.world_model.hplus import signal_from_payload
    trace.model_info["hplus"] = signal_from_payload(raw_gain)
    if benefit_convention:
        trace.model_info["benefit_convention"] = copy.deepcopy(benefit_convention)
        if unsupported.get("benefit") and isinstance(raw_benefit, dict):
            trace.model_info["rejected_benefit_convention"] = {
                key: raw_benefit.get(key) for key in ("kind", "metric", "unit", "scope", "observation_rule_version")}
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
        from or_harness.world_model.conditions import prepare_context, lease_context
        prepare_context(self.store, context)
        with lease_context(self.store, context):
            request = build_strategy_outcome_request(
                context, candidate, benefit_baseline_hint=benefit_baseline_hint,
                benefit_convention=benefit_convention,
                provider=self.provider)
            pending = self._failed(context, candidate, "invalid",
                "provider call started; result and usage are unknown until a response is saved",
                failure={"kind": "in_flight", "state": "pending"})
            pending.trace.model_info["call_state"] = "in_flight"
            self._save_with_conditions(pending, context, request)

            def save_result(prediction):
                prediction.prediction_id = pending.prediction_id
                prediction.trace.model_info["call_state"] = "returned"
                self._save_with_conditions(prediction, context, request)

            size = request.get("request_size") or {}
            if size.get("sendable") is False:
                # The FULL request is over the deployment's input bound even
                # after the history feedback and other non-core blocks were
                # reduced, and the protected core alone is over. Sending it
                # would be an uncontrolled request: refuse, record WHY and
                # keep the diagnostic sizes instead.
                over = size.get("over_budget") or {}
                prediction = self._failed(
                    context, candidate, "invalid",
                    "input request exceeds the configured bound: "
                    f"{size.get('tokens')} estimated token(s) vs "
                    f"{size.get('bound_tokens')} input tokens; the request "
                    "was NOT sent",
                    failure={
                        "kind": "request_over_budget",
                        "tokens": size.get("tokens"),
                        "bound_tokens": size.get("bound_tokens"),
                        "token_basis": size.get("token_basis"),
                        "core_chars": (size.get("core") or {}).get(
                            "core_chars"),
                        "over_budget": copy.deepcopy(over),
                        "trimming": copy.deepcopy(
                            size.get("trimming") or []),
                    })
                prediction.trace.model_info["request_size"] = copy.deepcopy(
                    size)
                prediction.trace.model_info["request_trimming"] =                     copy.deepcopy(size.get("trimming") or [])
                save_result(prediction)
                return prediction

            # The provider receives the request AS SENT: the local
            # request_size/trimming diagnostics are excluded. The budget was
            # measured on exactly this wire object, so a request reported
            # within the bound cannot exceed it because of diagnostics.
            from or_harness.world_model.request_budget import (
                strip_request_diagnostics,
            )
            wire_request = strip_request_diagnostics(request)
            try:
                try:
                    result = self.provider.predict(wire_request,
                                                   timeout_s=timeout_s)
                except TypeError:
                    result = self.provider.predict(wire_request)
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
                save_result(prediction)
                return prediction
            if result.get("not_configured"):
                prediction = self._failed(
                    context, candidate, "contract_only",
                    str(result.get("error") or "provider not configured"),
                    provider_configured=False, service_available=False,
                    failure=diagnose_provider_failure(result, None))
                save_result(prediction)
                return prediction
            payload = result.get("payload")
            if payload is None:
                prediction = self._failed(
                    context, candidate, "invalid",
                    str(result.get("error") or "no payload"),
                    provider_result=result,
                    failure=diagnose_provider_failure(result, None))
                save_result(prediction)
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
                save_result(prediction)
                return prediction
            try:
                prediction = parse_strategy_outcome_payload(
                    payload, candidate, context=context,
                    provider_result=result,
                    benefit_convention=request["benefit_convention"])
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
                save_result(prediction)
                return prediction
            if prediction.status == "invalid" \
                    and not prediction.trace.model_info.get("parse_error"):
                prediction.trace.model_info["parse_error"] = (
                    "the model's payload failed validation; see notes")
            self._stamp_identity(prediction)
            save_result(prediction)
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

    def _save_with_conditions(self, prediction, context, request):
        from or_harness.world_model.conditions import digest, compact_budget, put_value
        from or_harness.world_model.g_signal import requested_solving_conditions
        info = prediction.trace.model_info
        header_raw = self.store.get_prediction_context(context.context_id)
        header = self.store.loads(header_raw) if header_raw else {}
        refs = {row.get("identity"): row.get("value_ref") for row in header.get("evidence_versions", [])}
        sent = request.get("prediction_context") or {}
        sent_history = sent.get("history_retrieval") or {}
        # The frozen context's history_retrieval.config is authoritative.
        # Its provider view is flat: copy the configuration ACTUALLY SENT,
        # rather than looking for an absent config child or recomputing it
        # from today's backend/environment. Case bodies are not copied here.
        sent_history_config = {
            key: copy.deepcopy(sent_history[key]) for key in (
                "backend_model_id", "max_local_cases", "pool_size",
                "similarity_thresholds", "method_similarity_threshold",
                "threshold_verified", "threshold_sources")
            if key in sent_history}
        info["input_selection"] = {
            "evidence": [{"identity": hit.get("identity"), "version_ref": refs.get(hit.get("identity")),
                          "version": hit.get("version")}
                         for hit in (sent.get("retrieval_evidence") or {}).get("hits", [])],
            "pair_ids": [str(pair["evaluation_id"]) for pair in (sent.get("prediction_execution_pairs") or {}).get("pairs", [])
                         if isinstance(pair, dict) and pair.get("evaluation_id")],
            "feedback_versions": header.get("feedback_versions") or {},
            "display_cap": copy.deepcopy((sent.get("prediction_execution_pairs") or {}).get("display_capped")),
            # What the unified history retrieval actually DELIVERED to this
            # request (not merely what the stored context contains): the
            # status/config/accounting plus the local-case counts per signal.
            "history_selection": {
                "retrieval": {
                    **{key: copy.deepcopy(sent_history.get(key))
                       for key in ("version", "status", "accounting")},
                    "config": sent_history_config,
                    "config_source": "prediction_context.history_retrieval (actual sent projection)",
                },
                "g": copy.deepcopy((sent.get("g_candidate_evidence") or {}).get("selection")),
                "c": copy.deepcopy((sent.get("cost_evidence") or {}).get("reference_coverage")),
                "r": copy.deepcopy((sent.get("implementation_r_evidence") or {}).get("selection")),
            },
            "source": "actually displayed provider input; adoption/effect are separate facts",
        }
        profile = context.joint.profile or {}
        from or_harness.world_model.cost_signal import cost_conditions
        info["cost_conditions"] = cost_conditions(context, prediction.candidate)
        info['cost_observation_rule_versions']={'retries':request['cost_scope']['retries_convention']['observation_rule_version']}
        info["risk_convention"] = copy.deepcopy(request["risk_convention"])
        info.update({
            "family": profile.get("family"),
            "cell_token": (group_key(ProblemProfile.from_dict(profile))
                           if profile.get("problem_id") and profile.get("family") else None),
            "g_execution_conditions": requested_solving_conditions(
                prediction.candidate, context.execution_constraints),
            "benefit_convention": copy.deepcopy(request.get("benefit_convention")),
            "prediction_context_id": context.context_id,
            "prediction_context_version": context.version,
            "prediction_context_task_digest": context.task_digest,
            "effective_input_digest": context.effective_input_digest,
            "problem_identity_digest": context.snapshot.get("problem_identity_digest"),
            "hypothetical": context.snapshot.get("hypothetical"),
            "frozen_budget_state": compact_budget(context.solving_context.get("budget_state")),
            "conditions_id": context.snapshot_id,
            "request_digest": digest(request),
            "request_size": copy.deepcopy(request.get("request_size")),
            "request_omissions": copy.deepcopy((request.get("prediction_context") or {}).get("omissions") or []),
        })
        # The context is already durable before the call. The first save,
        # including every failure path, contains all binding conditions.
        with self.store.transaction() as conn:
            from or_harness.world_model.hplus import freeze_signal
            if self._harness() is not None:
                freeze_signal(self._harness(), prediction, context, request)
            info["prompt_ref"] = put_value(conn, self.store, "prompt", {"system": STRATEGY_OUTCOME_SYSTEM_PROMPT})
            info["prompt_version"] = "wm-so/1:" + digest(STRATEGY_OUTCOME_SYSTEM_PROMPT)
            self._save(prediction)

    def _save(self, prediction: StrategyOutcomePrediction) -> None:
        from or_harness.core.evidence_encoding import share_values
        with self.store.transaction():
            self.store.put_contract_prediction(
                prediction.prediction_id,
                prediction.candidate.task_id,
                prediction.candidate.episode_id,
                self.store.dumps(share_values(self.store,prediction.to_dict())),
                created_at=prediction.trace.created_at)
    def _harness(self):
        """The harness that freezes the same-call binary H+ signal."""
        return getattr(self, "harness", None)

    def get(self, prediction_id: str) -> Optional[StrategyOutcomePrediction]:
        raw = self.store.get_contract_prediction(prediction_id)
        if raw is None:
            return None
        from or_harness.world_model.contracts import (
            StrategyOutcomePrediction as SOP,
        )
        from or_harness.core.evidence_encoding import resolve_values
        return SOP.from_dict(resolve_values(self.store,self.store.loads(raw)))

    def query(self, *, task_id: Optional[str] = None,
              episode_id: Optional[str] = None,exact_episode: bool = False
              ) -> List[StrategyOutcomePrediction]:
        from or_harness.world_model.contracts import (
            StrategyOutcomePrediction as SOP,
        )
        from or_harness.core.evidence_encoding import resolve_values
        return [SOP.from_dict(resolve_values(self.store,self.store.loads(raw)))
                for raw in self.store.contract_predictions_for(
                    task_id=task_id, episode_id=episode_id,exact_episode=exact_episode)]
