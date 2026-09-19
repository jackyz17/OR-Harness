"""Episode close-out: real outcomes, post-hoc evaluation, experience
calibration (world-model M4).

This module closes the loop the M3 service opened. A strategy-outcome
prediction (wm-so/1) was made BEFORE execution; M4 answers, AFTER the
episode really ended, three questions in order:

1. **What actually happened?** :func:`summarize_real_outcome` builds a
   traceable summary of one bound prediction's real execution window —
   the actual selection, the execution trajectory references, the
   validity/verification basis, the benefit-relevant observations, the
   real cost (scoped to what the prediction covered) and the observed
   risk events. Every field carries its own eligibility: evaluable,
   missing, unverified, scope-mismatched or identity-mismatched.
2. **How good was the prediction?** :func:`evaluate_strategy_prediction`
   compares the FROZEN prediction against that summary, field by field —
   benefit error only under the same metric/unit/baseline/scope, cost
   error per dimension only where both sides measured the same scope,
   Brier scores only for events with a known predicted probability AND a
   reliable binary label, interval coverage only where both were saved.
   The original prediction is never rewritten; the evaluation is a
   separate, append-only record.
3. **What has the model been worth historically?**
   :func:`build_calibration_summary` aggregates the evaluations of CLOSED
   episodes into a versioned summary — error statistics, interval
   coverage, event scores — grouped by protocol/metric/scope, with sample
   counts, distinct-episode counts and exclusion reasons. A group below
   the minimum sample threshold reports ``insufficient_evidence``, never
   a guessed reliability.

Design boundaries (the reasons this module is shaped the way it is):

- **Close-out is the single entry.** The summary evaluation and the
  calibration publication happen when the EPISODE ends, not per action:
  in-task feedback must not become in-task calibration. A window may
  terminate early (its facts are saved), but its aggregate evaluation is
  published only at close-out.
- **Unknown is never a truth.** A missing baseline, an unverified
  solution, an unobserved risk event, an identity field the action could
  not record — each is reported as such and EXCLUDED from the statistics
  it would pollute. No counterfactual label is fabricated for an
  unexecuted candidate.
- **The prediction stays frozen.** The evaluation references the
  prediction id; nothing on the prediction's predicted content is edited
  after the fact. A "better" prediction cannot be regenerated closer to
  the result.
- **Idempotent and recoverable.** Closing the same episode twice,
  re-reading after a restart, or re-binding produces the same facts once:
  the close-out record is keyed by (task, episode) and the evaluation
  records by prediction id.
- **No model calls, no solvers, no induction.** Close-out reads what was
  recorded. It never runs a strategy, never invokes the provider, never
  triggers knowledge induction — those are separate explicit actions.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import COST_DIMENSIONS, CostVector

#: Version of the close-out record schema.
EPISODE_CLOSEOUT_VERSION = "wm-closeout/1"

#: Version of the calibration summary schema.
CALIBRATION_SUMMARY_VERSION = "wm-calib/1"

#: Terminal states an episode may be closed under. Only ``completed``
#: claims success; the others are honest endings, never dressed up.
EPISODE_TERMINAL_STATES = ("completed", "failed", "aborted",
                           "budget_exhausted")

#: Minimum resolved samples before a calibration group reports a
#: reliability figure (below it: ``insufficient_evidence``). Configurable
#: at build time; the effective value is recorded on the summary.
DEFAULT_MIN_CALIBRATION_SAMPLES = 5

#: Field eligibility values. ``evaluable`` is the only one that enters
#: statistics; every other value says WHY it does not.
FIELD_ELIGIBILITY = ("evaluable", "missing", "unverified", "scope_mismatch",
                     "identity_mismatch", "not_predicted", "unobserved",
                     "unreliable_label")

#: The ONE benefit metric this build can actually OBSERVE from an
#: execution: the solver's normalized gap (``1 - mip_gap``; an ``optimal``
#: status is a gap of 0). A prediction declaring any other metric is
#: reported ``scope_mismatch`` — the observed number is never re-labelled
#: as a business ratio it did not measure.
OBSERVABLE_BENEFIT_METRIC = "normalized_objective_gap"

#: Declared-metric spellings that map onto the observable metric.
_BENEFIT_METRIC_ALIASES = {
    "normalized_objective_gap": "normalized_objective_gap",
    "solution_quality": "normalized_objective_gap",
    "normalized_quality": "normalized_objective_gap",
    "normalized_gap": "normalized_objective_gap",
}

#: Risk-event names this build can actually OBSERVE, and from what: the
#: in-scope executions' statuses/failure classes and the episode's budget
#: view. An event outside this vocabulary has NO observation channel —
#: its label stays unknown and it is never scored, because "no failure
#: log" is not evidence that a business risk did not happen.
OBSERVABLE_RISK_EVENTS = ("model_invalid", "no_feasible_solution",
                          "timeout", "environment_failure",
                          "budget_exhausted")


def _observable_benefit_metric(metric: Any) -> Optional[str]:
    """The canonical observable metric a declared metric maps to, or None."""
    name = str(metric or "").strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    return _BENEFIT_METRIC_ALIASES.get(name)


def _normalize_event_name(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 1. the real outcome summary
# ---------------------------------------------------------------------------


@dataclass
class RealOutcomeSummary:
    """What actually happened in one bound prediction's real scope.

    Built from the action log and the execution records — never from the
    prediction. Each block carries its own eligibility so a later reader
    can tell "measured and comparable" from "absent", "unverified" or
    "belongs to another identity".
    """

    prediction_id: str
    task_id: str
    episode_id: Optional[str]
    strategy_id: Optional[str]
    action_id: Optional[str] = None
    execution_ids: List[str] = field(default_factory=list)
    #: The terminal state of the scope the prediction covered.
    scope_status: str = ""
    #: Benefit-relevant observations, with the metric/unit the PREDICTION
    # declared (the comparison must use the prediction's own yardstick).
    benefit: Dict[str, Any] = field(default_factory=dict)
    #: Real cost, scoped to the prediction's declared scope; auxiliary
    # overhead reported separately (it is real spend, but not what the
    # prediction spoke about).
    cost: Dict[str, Any] = field(default_factory=dict)
    auxiliary_cost: Dict[str, Any] = field(default_factory=dict)
    #: Observed risk events: occurred / not_occurred / unknown + basis.
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    #: Verification evidence for the solution the window produced.
    verification: Dict[str, Any] = field(default_factory=dict)
    #: Per-field eligibility: field -> (eligibility, reason).
    eligibility: Dict[str, Dict[str, str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closeout_version": EPISODE_CLOSEOUT_VERSION,
            "prediction_id": self.prediction_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "strategy_id": self.strategy_id,
            "action_id": self.action_id,
            "execution_ids": list(self.execution_ids),
            "scope_status": self.scope_status,
            "benefit": copy.deepcopy(self.benefit),
            "cost": copy.deepcopy(self.cost),
            "auxiliary_cost": copy.deepcopy(self.auxiliary_cost),
            "risk_events": copy.deepcopy(self.risk_events),
            "verification": copy.deepcopy(self.verification),
            "eligibility": copy.deepcopy(self.eligibility),
            "notes": list(self.notes),
        }


def _aggregate_costs(vectors: Sequence[Optional[CostVector]]
                     ) -> Dict[str, Any]:
    """Per-dimension totals over measured costs, with completeness.

    Same arithmetic as the execution-window aggregation: a dimension's
    total is the sum over the items that MEASURED it; ``complete`` is True
    only when every item measured it; a dimension nobody measured stays
    ``None`` (unknown, never zero). ``latency_s`` is never summed — it is
    reported per item elsewhere and never folded into an end-to-end figure.
    """
    dims: Dict[str, Any] = {}
    for dim in COST_DIMENSIONS:
        values: List[float] = []
        for cost in vectors:
            if cost is None or dim not in cost.measured_dims():
                continue
            values.append(float(getattr(cost, dim)))
        dims[dim] = {
            "total": round(sum(values), 6) if values else None,
            "n_measured": len(values),
            "n_items": len(vectors),
            "complete": bool(vectors) and len(values) == len(vectors),
        }
    return dims


def summarize_real_outcome(harness, prediction) -> RealOutcomeSummary:
    """Build the real-outcome summary of ONE bound prediction.

    ``prediction`` is a :class:`~or_harness.world_model.contracts
    .StrategyOutcomePrediction` that has been bound to a real action
    (``trace.model_info.bound_action_id``). The summary is derived from
    the action log and the execution records; the prediction itself is
    read only for its DECLARED scope/metric (the yardstick the comparison
    must use), never for content.
    """
    info = prediction.trace.model_info
    action_id = info.get("bound_action_id")
    candidate = prediction.candidate
    summary = RealOutcomeSummary(
        prediction_id=prediction.prediction_id,
        task_id=candidate.task_id,
        episode_id=candidate.episode_id,
        strategy_id=candidate.strategy_id,
        action_id=action_id,
    )
    mismatch = info.get("binding_mismatch") or {}
    unknown = info.get("binding_unknown") or {}
    if mismatch:
        for key in mismatch:
            summary.eligibility[f"identity.{key}"] = {
                "eligibility": "identity_mismatch",
                "reason": "the executed action differs from the predicted "
                          "candidate on this field",
            }
    if unknown:
        for key in unknown:
            summary.eligibility[f"identity.{key}"] = {
                "eligibility": "identity_mismatch",
                "reason": "the executed action could not confirm this "
                          "identity field (unknown, never a match)",
            }
    if action_id is None:
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the prediction was never bound to a real action: "
                      "an unexecuted candidate has no real outcome",
        }
        summary.notes.append(
            "unexecuted candidate: no counterfactual truth is fabricated")
        return summary

    action = harness.actions.get(action_id)
    if action is None:
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the bound action no longer exists in the log",
        }
        return summary

    # The real scope. An ATTEMPT-scope prediction is compared against the
    # bound action's own execution. A STRATEGY-WINDOW-scope prediction is
    # compared against the WHOLE window of its selection round: every
    # in-scope execution of that (task, episode, strategy, round) — a
    # failed first attempt and the repaired retry both count, so an
    # 11s-then-17s window reports 28s of real solve time, not just the
    # last attempt's 17s.
    records: List[Any] = []
    if candidate.scope == "strategy_window":
        # The DECLARED window identity (including its selection round, when
        # the window_id carries one) decides which window aggregates into
        # the real outcome: a round-1 prediction is scored against round
        # 1's executions only, never the whole-episode aggregation.
        declared_round = None
        if candidate.window_id:
            from or_harness.world_model.execution_window import (
                parse_window_id,
            )
            parsed = parse_window_id(candidate.window_id)
            if parsed is not None:
                declared_round = parsed.round_index
        window = harness.strategy_execution_window(
            action.task_id, action.episode_id,
            strategy_id=candidate.strategy_id,
            round_index=declared_round)
        window_actions = {a.action_id for a in window.attempts}
        if action.action_id not in window_actions:
            # The bound action is not in the derived window (it may be an
            # agent-reported action the derivation cannot see): fall back
            # to the bound action's own execution so the summary is never
            # empty, and say so.
            summary.notes.append(
                "the bound action is not among the derived window "
                "attempts; the summary covers the bound action's own "
                "execution only")
        for attempt in window.attempts:
            if attempt.execution_id is None:
                continue
            record = (harness.bank.get_pending(attempt.execution_id)
                      or harness.bank.get(attempt.execution_id))
            if record is not None:
                records.append(record)
            else:
                summary.eligibility["scope.execution"] = {
                    "eligibility": "missing",
                    "reason": (f"window attempt {attempt.action_id} has no "
                               "execution in the bank (staged or "
                               "recorded)"),
                }
        if not records and action.linked_execution_id:
            record = (harness.bank.get_pending(action.linked_execution_id)
                      or harness.bank.get(action.linked_execution_id))
            if record is not None:
                records.append(record)
        summary.notes.append(
            f"strategy-window scope (round "
            f"{declared_round if declared_round is not None else 'all'}): "
            f"{len(records)} in-scope execution(s) aggregate into the real "
            "outcome (failed retries included)")
    if action.linked_execution_id and not records:
        record = (harness.bank.get_pending(action.linked_execution_id)
                  or harness.bank.get(action.linked_execution_id))
        if record is not None:
            records.append(record)
        else:
            summary.eligibility["scope.execution"] = {
                "eligibility": "missing",
                "reason": "the bound action's linked execution is not in "
                          "the bank (staged or recorded)",
            }
    elif not records and not action.linked_execution_id:
        summary.eligibility["scope.execution"] = {
            "eligibility": "missing",
            "reason": "the bound action has no linked execution",
        }
    summary.execution_ids = [r.execution_id for r in records]

    # Scope status: the terminal state of what the prediction covered.
    if action.status == "running":
        summary.scope_status = "running"
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the bound action has not ended: a running scope has "
                      "no final numbers",
        }
    else:
        summary.scope_status = str(action.status)
        if records:
            exec_status = records[-1].quality.get("status")
            summary.scope_status = f"{action.status}/{exec_status}"

    # -- benefit observations -------------------------------------------
    # The yardstick is the PREDICTION's own declared metric/unit/baseline:
    # the comparison must not invent a more convenient one afterwards.
    # Only ONE metric has an observation adapter in this build — the
    # solver's normalized gap. A prediction declaring anything else (a
    # business cost-saving ratio, a completion rate) is reported
    # scope_mismatch: the solver's 1-gap number is never re-labelled as
    # that metric, because it did not measure it.
    benefit = prediction.benefit
    if benefit is not None and benefit.kind == "solution_quality":
        observable = _observable_benefit_metric(benefit.metric)
        if observable is None:
            summary.eligibility["benefit"] = {
                "eligibility": "scope_mismatch",
                "reason": (f"no observation adapter exists for the declared "
                           f"metric {benefit.metric!r}: this build observes "
                           f"only {OBSERVABLE_BENEFIT_METRIC!r} "
                           "(the solver's normalized gap). The observed "
                           "gap is never re-labelled as a metric it did "
                           "not measure"),
            }
        else:
            observed: List[float] = []
            for record in records:
                quality = record.quality or {}
                gap = quality.get("gap")
                status = quality.get("status")
                if status == "optimal":
                    observed.append(1.0)
                elif gap is not None and _finite(gap):
                    observed.append(max(0.0, 1.0 - float(gap)))
                elif quality.get("feasible"):
                    # A feasible solution with no gap/bound: the 0.5
                    # heuristic is NOT an observed quality truth.
                    summary.eligibility["benefit"] = {
                        "eligibility": "unverified",
                        "reason": "a feasible execution produced no "
                                  "gap/bound: the normalized quality is "
                                  "the 0.5 heuristic, not an observation",
                    }
            if observed:
                # Window rule (declared BEFORE evaluation): the LAST
                # in-scope attempt's qualified solution is the window's
                # benefit observation — the rule the prediction's scope
                # statement implies, applied uniformly, never chosen per
                # result.
                summary.benefit = {
                    "kind": "solution_quality",
                    "metric": benefit.metric,
                    "unit": benefit.unit,
                    "observed": round(observed[-1], 6),
                    "rule": "last qualified in-scope attempt",
                    "n_observations": len(observed),
                    "all_observations": [round(v, 6) for v in observed],
                }
                if "benefit" not in summary.eligibility:
                    summary.eligibility["benefit"] = {
                        "eligibility": "evaluable",
                        "reason": "normalized quality observed on the "
                                  "in-scope execution(s)",
                    }
            elif "benefit" not in summary.eligibility:
                summary.eligibility["benefit"] = {
                    "eligibility": "unobserved",
                    "reason": "no in-scope execution produced a qualified "
                              "solution quality",
                }
    elif benefit is not None:
        summary.eligibility["benefit"] = {
            "eligibility": "scope_mismatch",
            "reason": (f"no observation adapter exists for benefit kind "
                       f"{benefit.kind!r}: the benefit is reported, not "
                       "evaluated — this build evaluates normalized "
                       "solution quality only"),
        }
    else:
        summary.eligibility["benefit"] = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no benefit to observe against",
        }

    # -- cost observations ------------------------------------------------
    in_scope_costs = [r.cost for r in records]
    summary.cost = _aggregate_costs(in_scope_costs)
    # Auxiliary overhead: the OTHER actions of the same episode (model /
    # verify / select_strategy / other executions) — real spend, reported,
    # never folded into the predicted scope's comparison.
    auxiliary: List[CostVector] = []
    for other in harness.actions.query(task_id=candidate.task_id,
                                        episode_id=candidate.episode_id):
        if other.action_id == action_id or other.rollup == "reference":
            continue
        if other.action_type == "execute_strategy" \
                and other.action_id != action_id:
            continue
        if other.cost is not None:
            auxiliary.append(other.cost)
    summary.auxiliary_cost = _aggregate_costs(auxiliary)
    summary.notes.append(
        "cost comparison uses the in-scope totals only; auxiliary overhead "
        "(modelling, verification, other attempts) is real spend reported "
        "separately and never charged to this prediction's scope")

    # -- risk event observations -------------------------------------------
    # An event is observed ONLY through a channel this build really has:
    # the in-scope executions' statuses and failure classes, and the
    # episode's declared-budget view. An event outside that vocabulary
    # (a business risk with no observation channel) keeps label=None —
    # "no failure log" is not evidence it did not happen.
    observed_events: Dict[str, str] = {}
    for record in records:
        status = (record.quality or {}).get("status")
        if status == "error":
            observed_events["model_invalid"] = "occurred"
        elif status == "timeout":
            observed_events["timeout"] = "occurred"
        elif status == "infeasible":
            observed_events["no_feasible_solution"] = "occurred"
        for failure in record.failures or []:
            error_class = str(getattr(failure, "error_class", None)
                              or "model")
            observed_events[f"{error_class}_failure"] = "occurred"
    # The budget channel, THREE states: a declared budget CONFIRMED
    # exceeded by real consumption is an OBSERVED budget_exhausted event;
    # a CONFIRMED within-budget verdict (status "ok") over a completed
    # scope is a not_occurred observation; anything else — an UNCONFIRMED
    # ledger (some cost dimension unmeasured), no declared budget, or an
    # incomplete scope — keeps the label unknown: "not yet known to
    # exceed" is NOT "did not exhaust".
    budget_event_label: Optional[str] = None
    budget_event_basis = ""
    declared_budget = (harness._load_budget(
        candidate.task_id, candidate.episode_id) or {})
    if declared_budget:
        budget_view = harness.budget.view(
            candidate.task_id, candidate.episode_id,
            budget=declared_budget)
        budget_status = budget_view.get("status")
        if budget_status == "exceeded":
            budget_event_label = "occurred"
            budget_event_basis = ("declared budget exceeded by measured "
                                  "real consumption")
        elif budget_status == "ok":
            budget_event_label = "not_occurred"
            budget_event_basis = ("declared budget confirmed within "
                                  "limits (every dimension measured)")
        else:
            # "unconfirmed" / "no_budget_declared": the ledger cannot
            # establish either direction — the label stays unknown.
            budget_event_label = None
            budget_event_basis = (
                f"budget status {budget_status!r}: the consumption cannot "
                "be confirmed in either direction, so the label stays "
                "unknown and the event is excluded from scoring")
    window_complete = bool(records) and action.status != "running"
    risk_events: List[Dict[str, Any]] = []
    predicted_events = (prediction.risk.events
                        if prediction.risk is not None else [])
    for event in predicted_events:
        canonical = _normalize_event_name(event.event)
        label: Optional[str]
        if canonical in observed_events:
            label = "occurred"
        elif canonical == "budget_exhausted":
            # The budget channel decides this event alone: the generic
            # "completed scope without the event" branch never applies to
            # it (an unconfirmed ledger is not evidence of no exhaustion).
            label = budget_event_label
        elif canonical not in OBSERVABLE_RISK_EVENTS:
            # No observation channel exists for this event name: the
            # label stays unknown whatever the executions did — an
            # unobserved business risk is never scored as "did not
            # happen" on the strength of an absent log.
            label = None
        elif window_complete:
            # The event IS in the observable vocabulary and the scope
            # completed without it appearing: the absence is an
            # observation of THIS scope (still a single trajectory).
            label = "not_occurred"
        else:
            label = None
        if canonical == "budget_exhausted":
            basis = ("observed on the in-scope execution(s)"
                     if label == "occurred" else budget_event_basis)
        else:
            basis = ("observed on the in-scope execution(s)"
                     if label == "occurred" else
                     "completed scope with no such observed event "
                     "(single trajectory)"
                     if label == "not_occurred" else
                     (f"no observation channel exists for event "
                      f"{event.event!r}: the label stays unknown "
                      "and the event is excluded from scoring"
                      if canonical not in OBSERVABLE_RISK_EVENTS
                      else
                      "unknown: the scope was not fully observed, "
                      "so absence is not evidence"))
        risk_events.append({
            "event": event.event,
            "predicted_probability": event.probability,
            "label": label,
            "label_basis": basis,
        })
    summary.risk_events = risk_events

    # -- verification -----------------------------------------------------
    verification_level = (action.params or {}).get("verification_level")
    verified_actions = [
        a for a in harness.actions.query(task_id=candidate.task_id,
                                         episode_id=candidate.episode_id)
        if a.action_type == "verify"]
    summary.verification = {
        "execution_verification_level": verification_level,
        "n_verify_actions": len(verified_actions),
        "solver_reported_status": (records[0].quality.get("status")
                                   if records else None),
        "note": ("solver 'optimal' does not by itself prove the business "
                 "requirements are satisfied; verification evidence is "
                 "whatever verify actions and the executor's checks "
                 "recorded"),
    }
    return summary


# ---------------------------------------------------------------------------
# 2. the post-hoc evaluation
# ---------------------------------------------------------------------------


@dataclass
class StrategyPredictionEvaluation:
    """The post-hoc comparison of ONE frozen prediction against its real
    outcome.

    Appended at close-out; the prediction itself is never modified. Every
    compared field names its own basis, and every excluded field names its
    exclusion reason — an evaluation with nothing comparable says so
    instead of scoring nothing as success.
    """

    evaluation_id: str = field(default_factory=lambda: _new_id("ev"))
    prediction_id: str = ""
    task_id: str = ""
    episode_id: Optional[str] = None
    #: The prediction's declared measurement scope (attempt /
    #: strategy_window): part of the calibration grouping key, so samples
    #: under different scopes never pool.
    scope: str = "attempt"
    created_at: float = field(default_factory=time.time)
    #: benefit / cost / risk / interval blocks, each with eligibility.
    benefit: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    interval: Dict[str, Any] = field(default_factory=dict)
    #: Overall state: evaluated (>=1 field compared), excluded (nothing
    #: comparable, with reasons), or pending (the scope is not finished).
    state: str = "evaluated"
    exclusion_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_version": EPISODE_CLOSEOUT_VERSION,
            "evaluation_id": self.evaluation_id,
            "prediction_id": self.prediction_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "scope": self.scope,
            "created_at": self.created_at,
            "state": self.state,
            "benefit": copy.deepcopy(self.benefit),
            "cost": copy.deepcopy(self.cost),
            "risk": copy.deepcopy(self.risk),
            "interval": copy.deepcopy(self.interval),
            "exclusion_reasons": list(self.exclusion_reasons),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyPredictionEvaluation":
        data = data or {}
        return cls(
            evaluation_id=str(data.get("evaluation_id")
                              or _new_id("ev")),
            prediction_id=str(data.get("prediction_id", "")),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            scope=str(data.get("scope", "attempt")),
            created_at=float(data.get("created_at", time.time())),
            state=str(data.get("state", "evaluated")),
            benefit=copy.deepcopy(dict(data.get("benefit") or {})),
            cost=copy.deepcopy(dict(data.get("cost") or {})),
            risk=copy.deepcopy(dict(data.get("risk") or {})),
            interval=copy.deepcopy(dict(data.get("interval") or {})),
            exclusion_reasons=[str(r) for r in
                               (data.get("exclusion_reasons") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def evaluate_strategy_prediction(prediction, summary: RealOutcomeSummary
                                 ) -> StrategyPredictionEvaluation:
    """Compare ONE frozen prediction against its real-outcome summary.

    Field-by-field eligibility, never a blanket verdict:

    - **benefit**: only when the prediction declared a value AND the
      summary observed the SAME metric (normalized solution quality) under
      the same scope. The error is the absolute difference; the baseline
      the prediction declared is carried along (a gain vs its baseline is
      the prediction's own claim, restated not recomputed).
    - **cost**: per dimension, only where the prediction predicted the
      dimension AND the real scope measured it completely. The error is
      the absolute log-ratio (the shared ``cost_error_per_dim``
      arithmetic); a zero/missing denominator produces no relative error.
    - **risk**: per event, a Brier score only when the prediction gave a
      probability AND the label is known. A single trajectory does not
      prove a probability; the score is recorded as one sample, never as
      an accuracy verdict.
    - **interval**: coverage only when the prediction saved an interval
      AND the observed value exists.

    The original prediction is read, never written.
    """
    evaluation = StrategyPredictionEvaluation(
        prediction_id=prediction.prediction_id,
        task_id=prediction.candidate.task_id,
        episode_id=prediction.candidate.episode_id,
        scope=prediction.candidate.scope,
    )
    identity_problems = {k: v for k, v in summary.eligibility.items()
                         if k.startswith("identity.")}
    n_compared = 0

    # -- benefit -----------------------------------------------------------
    benefit = prediction.benefit
    if benefit is None or benefit.value is None:
        evaluation.benefit = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no benefit value",
        }
    elif "observed" not in summary.benefit:
        entry = summary.eligibility.get("benefit", {})
        evaluation.benefit = {
            "eligibility": entry.get("eligibility", "unobserved"),
            "reason": entry.get("reason", "no benefit observation"),
        }
    elif identity_problems:
        evaluation.benefit = {
            "eligibility": "identity_mismatch",
            "reason": "the binding identity is not established: the real "
                      "outcome may not be this prediction's truth",
        }
    else:
        observed = summary.benefit["observed"]
        evaluation.benefit = {
            "eligibility": "evaluable",
            "metric": benefit.metric,
            "unit": benefit.unit,
            "predicted": round(float(benefit.value), 6),
            "observed": observed,
            "abs_error": round(abs(float(benefit.value) - observed), 6),
            "baseline": (benefit.baseline.to_dict()
                         if benefit.baseline is not None else None),
            "note": ("the metric/unit/baseline are the prediction's OWN "
                     "declared yardstick, restated — never re-chosen after "
                     "the result was seen"),
        }
        n_compared += 1

    # -- cost -----------------------------------------------------------------
    predicted_cost = prediction.cost
    if predicted_cost is None or predicted_cost.expected is None:
        evaluation.cost = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no cost",
        }
    elif identity_problems:
        evaluation.cost = {
            "eligibility": "identity_mismatch",
            "reason": "the binding identity is not established",
        }
    else:
        per_dim: Dict[str, Any] = {}
        excluded: Dict[str, str] = {}
        predicted_dims = predicted_cost.expected.measured_dims()
        for dim in sorted(predicted_dims):
            real = summary.cost.get(dim) or {}
            if real.get("total") is None:
                excluded[dim] = ("not measured on the real scope"
                                 if real.get("n_items") else
                                 "no real execution to measure it")
                continue
            if not real.get("complete"):
                excluded[dim] = ("partially measured on the real scope "
                                 f"({real.get('n_measured')}/"
                                 f"{real.get('n_items')}): an incomplete "
                                 "total is not a truth to score against")
                continue
            p = float(getattr(predicted_cost.expected, dim))
            a = float(real["total"])
            entry: Dict[str, Any] = {
                "predicted": round(p, 6),
                "actual": round(a, 6),
                "abs_error": round(abs(p - a), 6),
            }
            if p > 0 and a > 0:
                entry["log_error"] = round(abs(__import__("math").log(
                    a / p)), 4)
            else:
                entry["log_error"] = None
                entry["note"] = ("zero predicted or actual: no relative "
                                 "error is manufactured")
            per_dim[dim] = entry
            n_compared += 1
        evaluation.cost = {
            "eligibility": "evaluable" if per_dim else "unobserved",
            "per_dim": per_dim,
            "excluded": excluded,
            "note": ("only dimensions the prediction predicted AND the "
                     "real scope measured completely participate; "
                     "auxiliary overhead is never charged to this scope"),
        }

    # -- risk -------------------------------------------------------------------
    risk = prediction.risk
    if risk is None or not risk.events:
        evaluation.risk = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no risk events",
        }
    elif identity_problems:
        evaluation.risk = {
            "eligibility": "identity_mismatch",
            "reason": "the binding identity is not established: the real "
                      "outcome may not be this prediction's truth",
        }
    else:
        scored: List[Dict[str, Any]] = []
        unscored: List[Dict[str, Any]] = []
        for observed_event in summary.risk_events:
            probability = observed_event["predicted_probability"]
            label = observed_event["label"]
            if probability is None:
                unscored.append({
                    "event": observed_event["event"],
                    "reason": "the prediction gave no probability",
                })
                continue
            if label is None:
                unscored.append({
                    "event": observed_event["event"],
                    "reason": observed_event["label_basis"],
                })
                continue
            y = 1.0 if label == "occurred" else 0.0
            scored.append({
                "event": observed_event["event"],
                "predicted_probability": round(float(probability), 6),
                "label": label,
                "label_basis": observed_event["label_basis"],
                "brier": round((float(probability) - y) ** 2, 6),
            })
            n_compared += 1
        evaluation.risk = {
            "eligibility": ("evaluable" if scored else "unobserved"),
            "scored": scored,
            "unscored": unscored,
            "note": ("a Brier score is one sample of a probability's "
                     "quality, never a verdict from a single trajectory; "
                     "events are never averaged across different names"),
        }

    # -- interval ------------------------------------------------------------
    if benefit is not None and benefit.interval is not None:
        if "observed" in summary.benefit and not identity_problems:
            lo, hi = benefit.interval
            observed = summary.benefit["observed"]
            evaluation.interval = {
                "eligibility": "evaluable",
                "predicted_interval": [lo, hi],
                "observed": observed,
                "covered": bool(lo <= observed <= hi),
            }
            n_compared += 1
        else:
            evaluation.interval = {
                "eligibility": "unobserved",
                "reason": "no comparable observed value for the interval",
            }
    else:
        evaluation.interval = {
            "eligibility": "not_predicted",
            "reason": "the prediction saved no interval",
        }

    if summary.scope_status == "running":
        evaluation.state = "pending"
        evaluation.exclusion_reasons.append(
            "the bound scope has not ended: a running window has no final "
            "numbers to evaluate against")
    elif n_compared == 0:
        evaluation.state = "excluded"
        reasons = [f"{block.get('eligibility')}: {block.get('reason')}"
                   for block in (evaluation.benefit, evaluation.cost,
                                 evaluation.risk)
                   if isinstance(block, dict)
                   and block.get("eligibility") != "evaluable"]
        evaluation.exclusion_reasons.extend(
            reasons or ["no field was comparable"])
        evaluation.notes.append(
            "an excluded evaluation is neither a hit nor a miss: it never "
            "enters a denominator")
    else:
        evaluation.state = "evaluated"
    return evaluation


# ---------------------------------------------------------------------------
# 3. episode close-out
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCloseout:
    """The record that one episode ended, and what its predictions were
    worth.

    Created ONCE per (task, episode) by :func:`close_episode`; a repeated
    close is idempotent (the stored record is returned, nothing is
    re-counted). The close-out itself runs no solver, calls no model and
    triggers no induction.
    """

    closeout_id: str = field(default_factory=lambda: _new_id("cl"))
    task_id: str = ""
    episode_id: Optional[str] = None
    terminal_state: str = "completed"
    created_at: float = field(default_factory=time.time)
    #: The finish_task action that closed the episode, when one exists.
    finish_action_id: Optional[str] = None
    #: Evaluation records produced by this close-out (ids).
    evaluation_ids: List[str] = field(default_factory=list)
    #: Whether the calibration summary included this episode's samples.
    calibration_published: bool = False
    #: Unfinished actions at close time (reported, never fabricated away).
    unfinished_actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closeout_version": EPISODE_CLOSEOUT_VERSION,
            "closeout_id": self.closeout_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "terminal_state": self.terminal_state,
            "created_at": self.created_at,
            "finish_action_id": self.finish_action_id,
            "evaluation_ids": list(self.evaluation_ids),
            "calibration_published": bool(self.calibration_published),
            "unfinished_actions": list(self.unfinished_actions),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeCloseout":
        data = data or {}
        return cls(
            closeout_id=str(data.get("closeout_id") or _new_id("cl")),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            terminal_state=str(data.get("terminal_state", "completed")),
            created_at=float(data.get("created_at", time.time())),
            finish_action_id=data.get("finish_action_id"),
            evaluation_ids=[str(e) for e in
                            (data.get("evaluation_ids") or [])],
            calibration_published=bool(
                data.get("calibration_published", False)),
            unfinished_actions=[str(a) for a in
                                (data.get("unfinished_actions") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def close_episode(harness, task_id: str, episode_id: Optional[str], *,
                  terminal_state: str = "completed",
                  finish_action_id: Optional[str] = None,
                  min_calibration_samples: int =
                  DEFAULT_MIN_CALIBRATION_SAMPLES,
                  ) -> Dict[str, Any]:
    """Close ONE episode and publish its experience calibration.

    The single M4 entry: it (1) refuses when actions are still running
    (pending, never a fabricated ending), (2) summarizes and evaluates
    every BOUND strategy-outcome prediction of the episode against its
    real outcome, (3) records the close-out once (idempotent), and (4)
    folds the eligible evaluations into the versioned calibration summary
    that later episodes' prediction contexts read.

    It does NOT run a solver, call the prediction model, or trigger
    induction. A failed/aborted/budget-exhausted episode closes honestly
    under its own terminal state — never dressed up as completed.
    """
    if terminal_state not in EPISODE_TERMINAL_STATES:
        raise ValueError(
            f"terminal_state must be one of {EPISODE_TERMINAL_STATES}")
    store = harness.store
    key = f"episode_closeout|{task_id}|{episode_id or ''}"
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is not None:
        stored = EpisodeCloseout.from_dict(store.loads(row["value"]))
        return {
            "closeout": stored.to_dict(),
            "already_closed": True,
            "note": ("this episode was already closed: the stored record "
                     "stands, nothing was re-counted or re-billed"),
        }

    # Unfinished actions BLOCK the close-out: an episode with a running
    # action has no final state to evaluate against, and closing anyway
    # would freeze a "completed" record whose pending scopes can never be
    # back-filled (a re-close just returns the stored record). The
    # close-out stays PENDING — end the running actions first (or let them
    # finish), then close. This is a refusal to fabricate an ending, not
    # an error.
    unfinished = [a.action_id for a in harness.actions.query(
        task_id=task_id, episode_id=episode_id, status="running")]
    if unfinished:
        return {
            "closeout": None,
            "already_closed": False,
            "state": "pending",
            "unfinished_actions": unfinished,
            "note": ("the episode still has running action(s): the "
                     "close-out is REFUSED until they end. End them "
                     "(or let them finish), then close — a running scope "
                     "has no final numbers, and closing now would freeze "
                     "a record their results could never enter"),
        }

    closeout = EpisodeCloseout(
        task_id=task_id,
        episode_id=episode_id,
        terminal_state=terminal_state,
        finish_action_id=finish_action_id,
        unfinished_actions=[],
    )
    if terminal_state != "completed":
        closeout.notes.append(
            f"the episode ended in {terminal_state!r}: an honest terminal "
            "state, never reported as success")

    # Evaluate every BOUND strategy-outcome prediction of this episode.
    evaluations: List[StrategyPredictionEvaluation] = []
    for prediction in harness.strategy_predictions.query(
            task_id=task_id, episode_id=episode_id):
        if not prediction.trace.model_info.get("bound_action_id"):
            continue
        summary = summarize_real_outcome(harness, prediction)
        evaluation = evaluate_strategy_prediction(prediction, summary)
        evaluations.append(evaluation)

    # Persist the evaluations FIRST (append-only, keyed by evaluation id;
    # a re-close is blocked by the closeout key below, so no duplicate
    # contributions can appear).
    for evaluation in evaluations:
        _put_evaluation(store, evaluation)
        closeout.evaluation_ids.append(evaluation.evaluation_id)

    # THEN record the close-out, and only afterwards build the summary:
    # the summary aggregates CLOSED episodes' evaluations, so this
    # episode's samples must be persisted AND its close-out recorded
    # before the summary is generated — otherwise the first close-out's
    # own return misses this round's samples.
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, store.dumps(closeout.to_dict())))
    summary_block = build_calibration_summary(
        harness, min_samples=min_calibration_samples)
    closeout.calibration_published = True
    # Re-write the record with the publication flag (the summary itself
    # is derived data; only the flag is stored).
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, store.dumps(closeout.to_dict())))
    return {
        "closeout": closeout.to_dict(),
        "already_closed": False,
        "evaluations": [e.to_dict() for e in evaluations],
        "calibration_summary": summary_block,
    }


def _put_evaluation(store, evaluation: StrategyPredictionEvaluation
                    ) -> None:
    """Persist one evaluation record (idempotent by evaluation id)."""
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (f"strategy_evaluation|{evaluation.evaluation_id}",
             store.dumps(evaluation.to_dict())))


def get_evaluation(store, evaluation_id: str
                   ) -> Optional[StrategyPredictionEvaluation]:
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key=?",
        (f"strategy_evaluation|{evaluation_id}",)).fetchone()
    if row is None:
        return None
    return StrategyPredictionEvaluation.from_dict(store.loads(row["value"]))


def episode_closeout_record(harness, task_id: str,
                            episode_id: Optional[str]
                            ) -> Optional[EpisodeCloseout]:
    """The stored close-out of one episode, or None when it is open."""
    key = f"episode_closeout|{task_id}|{episode_id or ''}"
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    return EpisodeCloseout.from_dict(harness.store.loads(row["value"]))


# ---------------------------------------------------------------------------
# 4. the experience calibration summary
# ---------------------------------------------------------------------------


def _iter_closed_evaluations(harness
                             ) -> List[StrategyPredictionEvaluation]:
    """Every evaluation of every CLOSED episode (the only admissible
    samples: an open episode's feedback must not calibrate anything,
    least of all itself)."""
    out: List[StrategyPredictionEvaluation] = []
    rows = harness.store.conn.execute(
        "SELECT key, value FROM meta WHERE key LIKE "
        "'strategy_evaluation|%'").fetchall()
    for row in rows:
        try:
            evaluation = StrategyPredictionEvaluation.from_dict(
                harness.store.loads(row["value"]))
        except Exception:
            continue
        if episode_closeout_record(harness, evaluation.task_id,
                                   evaluation.episode_id) is None:
            continue
        out.append(evaluation)
    return out


def build_calibration_summary(harness, *,
                              min_samples: int =
                              DEFAULT_MIN_CALIBRATION_SAMPLES
                              ) -> Dict[str, Any]:
    """Aggregate closed-episode evaluations into a versioned summary.

    Grouped by (metric, unit, scope) so different definitions never mix —
    the same metric name under a different unit or a different scope is a
    DIFFERENT group, never pooled. Risk events are scored PER EVENT NAME:
    different events are different random variables and their Brier
    scores are never averaged together. The sample threshold counts
    DISTINCT EPISODES (independent truths), not predictions: one truth
    bound to five re-planning predictions is ONE episode's evidence, and
    counting the predictions would let a single execution cross the
    threshold five times over.

    Each group reports: sample count, distinct-episode count (repeated
    predictions marked correlated, never counted as independent), mean
    absolute benefit error, mean per-dimension cost log-error, interval
    coverage rate, and per-event Brier means. A group below
    ``min_samples`` DISTINCT EPISODES reports ``insufficient_evidence`` —
    no reliability figure is invented.

    This is measured EXPERIENCE reliability, not a fitted calibrator and
    not a model weight: writing the summary does not promise future
    predictions improve. It is kept SEPARATE from the legacy knowledge
    prediction reliability (``prediction_class_reliability``): a knowledge
    hit rate never proves OR strategy-outcome accuracy.
    """
    evaluations = _iter_closed_evaluations(harness)
    groups: Dict[str, Dict[str, Any]] = {}
    exclusions: Dict[str, int] = {}
    for evaluation in evaluations:
        if evaluation.state != "evaluated":
            state = evaluation.state or "other"
            exclusions[state] = exclusions.get(state, 0) + 1
            continue
        benefit = evaluation.benefit or {}
        metric = str(benefit.get("metric") or "(none)")
        unit = str(benefit.get("unit") or "(none)")
        scope = str(evaluation.scope or "(none)")
        group_key = f"strategy_outcome|{metric}|{unit}|{scope}"
        group = groups.setdefault(
            group_key, {
                "n": 0, "episodes": set(), "benefit_abs_errors": [],
                "cost_log_errors": {}, "interval_covered": [],
                "brier_by_event": {}, "correlated_predictions": 0,
            })
        group["n"] += 1
        group["episodes"].add(evaluation.episode_id or "")
        if benefit.get("eligibility") == "evaluable" \
                and benefit.get("abs_error") is not None:
            group["benefit_abs_errors"].append(
                float(benefit["abs_error"]))
        cost = evaluation.cost or {}
        for dim, entry in (cost.get("per_dim") or {}).items():
            if entry.get("log_error") is not None:
                group["cost_log_errors"].setdefault(dim, []).append(
                    float(entry["log_error"]))
        interval = evaluation.interval or {}
        if interval.get("eligibility") == "evaluable":
            group["interval_covered"].append(
                1.0 if interval.get("covered") else 0.0)
        risk = evaluation.risk or {}
        for entry in risk.get("scored") or []:
            # PER-EVENT accounting: different event names are different
            # variables; their scores are reported separately and never
            # averaged into one number.
            event_name = str(entry.get("event") or "(unnamed)")
            group["brier_by_event"].setdefault(event_name, []).append(
                float(entry["brier"]))

    def _mean(values: Sequence[float]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    out_groups: Dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        n = group["n"]
        distinct = len(group["episodes"])
        entry: Dict[str, Any] = {
            "n_samples": n,
            "n_distinct_episodes": distinct,
            "correlated_predictions": n - distinct,
            "mean_benefit_abs_error": _mean(group["benefit_abs_errors"]),
            "mean_cost_log_error": {
                dim: _mean(values)
                for dim, values in sorted(
                    group["cost_log_errors"].items())},
            "interval_coverage": _mean(group["interval_covered"]),
            "n_interval_samples": len(group["interval_covered"]),
            "mean_brier_by_event": {
                event: _mean(values)
                for event, values in sorted(
                    group["brier_by_event"].items())},
            "n_brier_samples_by_event": {
                event: len(values)
                for event, values in sorted(
                    group["brier_by_event"].items())},
        }
        # The threshold counts DISTINCT EPISODES (independent truths).
        # Predictions are correlated samples of the same truth: five
        # re-planning predictions over one execution are one episode's
        # evidence, and counting them as five would let a single outcome
        # cross the threshold.
        if distinct < min_samples:
            entry["reliability"] = None
            entry["basis"] = "insufficient_evidence"
            entry["note"] = (f"{distinct} distinct episode(s) < "
                             f"{min_samples}: no reliability figure is "
                             "claimed (correlated predictions of the same "
                             "truth do not count as independent samples)")
        else:
            entry["reliability"] = "measured_experience"
            entry["basis"] = "measured"
            entry["note"] = ("measured error/coverage statistics of past "
                             "closed-episode evaluations; a record of what "
                             "happened, never a promise that future "
                             "predictions improve")
        out_groups[name] = entry

    return {
        "calibration_version": CALIBRATION_SUMMARY_VERSION,
        "protocol": "wm-so/1",
        "min_samples": int(min_samples),
        "min_samples_basis": "distinct episodes",
        "n_evaluations_total": len(evaluations),
        "n_evaluated": sum(1 for e in evaluations
                           if e.state == "evaluated"),
        "exclusions": exclusions,
        "groups": out_groups,
        "note": ("experience calibration of the strategy-outcome "
                 "prediction service, from CLOSED episodes only; grouped "
                 "by metric/unit/scope, risk events scored per event name, "
                 "and the sample threshold counts DISTINCT EPISODES — kept "
                 "separate from the legacy knowledge-prediction "
                 "reliability (a knowledge hit rate never proves OR "
                 "prediction accuracy), and never a model weight or a "
                 "fitted calibrator"),
    }


def calibration_summary_for_context(harness) -> Dict[str, Any]:
    """The published calibration summary a NEW prediction context reads.

    Returns the summary's actual CONTENT (groups, counts, sources), not an
    id. An active episode does not see its own not-yet-closed feedback:
    only CLOSED episodes contribute (enforced by
    :func:`_iter_closed_evaluations`), so a context built mid-episode is
    calibrated by history alone.
    """
    return build_calibration_summary(harness)
