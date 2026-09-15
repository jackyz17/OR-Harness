"""Bounded planning over the world model (M3).

The planner answers ONE question for the outer agent: "from the current
frozen information state, which single action should I take next, and
why?" It compares a SMALL set of candidate actions (and, at horizon=2,
short two-step sequences) by their PREDICTED consequences — never by
re-wrapping the selector's Q/C ordering.

The decision loop, per ``plan_next`` call:

    freeze ONE root snapshot
      -> for each root candidate (<= limits.max_root_candidates):
           predict the first step's consequences (same root state,
           same evaluation yardstick)
           -> horizon=2: build a HYPOTHETICAL successor state from the
              first prediction's state_changes, predict the second step
              FROM that successor (not a second root prediction)
           -> evaluate the path: U = alpha*Q_terminal - beta*C_path
              - gamma*R_terminal
      -> recommend the FIRST step of the best path only

Evidence boundaries enforced here:

- all root candidates of one decision share the SAME frozen root
  snapshot; the planner never re-reads the banks mid-decision;
- hypothetical successor states are BeliefSnapshots with
  ``hypothetical=True``: they never enter the real state chain, the
  episode's progress inheritance, the budget, or any knowledge bank.
  Predicted changes are labelled ``epistemic="inferred"``; unknown stays
  unknown. Persistent H is untouched — a prediction never writes
  knowledge;
- a prediction made FROM a hypothetical state is a conditional outlook:
  it cannot be bound as a real one-step feedback sample (enforced in
  PredictionService.bind_outcome);
- predicted future-action costs stay hypothetical; the model calls made
  to obtain them are REAL spend, charged to the decision (parent) action
  as own cost and reported once in the PlanResult — never amortized onto
  candidate paths (sunk planning cost is identical for all candidates
  and must not change their ranking);
- quality is judged at the TERMINAL state of the path (or as improvement
  over the current solution), so longer paths are never preferred merely
  for accumulating more quality terms; risk is the terminal failure
  probability — step failure probabilities are neither summed nor
  multiplied (no independence assumption);
- a prediction that a solution "will exist" never materializes a real
  solution file: a second step that depends on an incumbent the first
  prediction did not establish is truncated and marked
  ``conditional_unsupported``;
- limits are hard: candidate count, horizon, model-call budget, and a
  wall-clock budget are checked before every new model call; exhaustion
  yields an explicit truncation reason, never a silent partial answer.

The defaults in :class:`PlanLimits` are ENGINEERING defaults written
into every plan record — not calibrated research conclusions.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.world_model.prediction import ActionSpec, OutcomePrediction
from or_harness.world_model.state import BeliefSnapshot

#: Action types the M3 planner can actually reason about. Other types are
#: reported as unsupported rather than receiving a fabricated result model.
PLANNABLE_ACTION_TYPES = ("execute_strategy",)


@dataclass
class PlanLimits:
    """Hard bounds for one planning call (engineering defaults, recorded
    verbatim into every plan for later sensitivity analysis and ablation).

    ``max_model_calls`` caps the number of world-model invocations of the
    whole decision (root candidates x horizon). ``time_budget_s`` caps the
    wall-clock of the evaluation phase. alpha/beta/gamma and cost_weights
    are the evaluation yardstick for THIS comparison — configurable, never
    re-tuned here, and always echoed into the plan record."""

    max_root_candidates: int = 3
    horizon: int = 1
    max_model_calls: int = 6
    time_budget_s: float = 120.0
    #: alpha/beta/gamma/cost_weights of None = "not overridden" — the
    #: caller (ORHarness.plan_next) fills them from its own configured
    #: evaluation yardstick. An explicit value (including an explicit
    #: empty cost_weights dict) always wins.
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    cost_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_root_candidates": int(self.max_root_candidates),
            "horizon": int(self.horizon),
            "max_model_calls": int(self.max_model_calls),
            "time_budget_s": float(self.time_budget_s),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "cost_weights": (dict(self.cost_weights)
                             if self.cost_weights is not None else None),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlanLimits":
        if not data:
            return cls()
        return cls(
            max_root_candidates=int(data.get("max_root_candidates", 3)),
            horizon=int(data.get("horizon", 1)),
            max_model_calls=int(data.get("max_model_calls", 6)),
            time_budget_s=float(data.get("time_budget_s", 120.0)),
            alpha=(None if data.get("alpha") is None
                   else float(data["alpha"])),
            beta=(None if data.get("beta") is None
                  else float(data["beta"])),
            gamma=(None if data.get("gamma") is None
                   else float(data["gamma"])),
            cost_weights=(None if data.get("cost_weights") is None
                          else dict(data["cost_weights"])),
        )


@dataclass
class PathStep:
    """One step of a candidate path: the spec and its prediction."""

    action_spec: ActionSpec
    prediction_id: Optional[str] = None
    #: "valid" | prediction failure status | "not_planned" (budget cut)
    status: str = "not_planned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_spec": self.action_spec.to_dict(),
            "prediction_id": self.prediction_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathStep":
        return cls(
            action_spec=ActionSpec.from_dict(data["action_spec"]),
            prediction_id=data.get("prediction_id"),
            status=str(data.get("status", "not_planned")),
        )


@dataclass
class CandidatePath:
    """One evaluated candidate: the root action, its (optional) conditional
    continuation, and the full utility decomposition."""

    steps: List[PathStep]
    #: Utility decomposition for THIS comparison's yardstick.
    utility: Optional[float] = None
    q_terminal: Optional[float] = None
    c_path: Optional[float] = None
    r_terminal: Optional[float] = None
    #: Which fields could not be compared and why (unknown never
    #: auto-scores better than known).
    incomparable: Dict[str, str] = field(default_factory=dict)
    #: Truncation/fallback notes for this path (e.g. second step
    #: conditional_unsupported, prediction invalid_output).
    notes: List[str] = field(default_factory=list)
    #: Snapshot id of the hypothetical successor this path's second step
    #: was conditioned on (horizon=2 only).
    hypothetical_snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "utility": self.utility,
            "q_terminal": self.q_terminal,
            "c_path": self.c_path,
            "r_terminal": self.r_terminal,
            "incomparable": dict(self.incomparable),
            "notes": list(self.notes),
            "hypothetical_snapshot_id": self.hypothetical_snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatePath":
        return cls(
            steps=[PathStep.from_dict(s) for s in data.get("steps") or []],
            utility=data.get("utility"),
            q_terminal=data.get("q_terminal"),
            c_path=data.get("c_path"),
            r_terminal=data.get("r_terminal"),
            incomparable=dict(data.get("incomparable") or {}),
            notes=[str(n) for n in data.get("notes") or []],
            hypothetical_snapshot_id=data.get("hypothetical_snapshot_id"),
        )


@dataclass
class PlanResult:
    """The auditable record of one planning decision.

    Small on purpose: references (snapshot/prediction/action ids) rather
    than copies of predictions — the predictions themselves live in the
    prediction log. The planning spend recorded here is REAL (the model
    calls that happened); per-path predicted costs stay hypothetical."""

    plan_id: str
    root_snapshot_id: str
    decision_action_id: Optional[str]
    task_id: str
    episode_id: Optional[str]
    limits: PlanLimits
    paths: List[CandidatePath] = field(default_factory=list)
    #: The recommended FIRST step (an ActionSpec), or None when no
    #: recommendation could be made (see ``status``/``truncation_reason``).
    suggested: Optional[ActionSpec] = None
    suggestion_basis: str = ""
    status: str = "ok"  # ok | truncated | fallback | no_candidates |
                        # no_valid_predictions | disabled
    truncation_reason: Optional[str] = None
    #: REAL planning spend: model calls made and their aggregated measured
    #: cost (also charged to the decision action as own cost). Sunk cost —
    #: never part of any path's utility.
    model_calls_made: int = 0
    planning_cost: Optional[Dict[str, Any]] = None
    #: Budget honesty: with a declared budget whose consumption has unknown
    #: dimensions, the plan never claims to be within budget.
    budget_confirmation: str = "unknown"  # ok | unconfirmed | exceeded | unknown
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return f"pl_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "root_snapshot_id": self.root_snapshot_id,
            "decision_action_id": self.decision_action_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "limits": self.limits.to_dict(),
            "paths": [p.to_dict() for p in self.paths],
            "suggested": (self.suggested.to_dict()
                          if self.suggested is not None else None),
            "suggestion_basis": self.suggestion_basis,
            "status": self.status,
            "truncation_reason": self.truncation_reason,
            "model_calls_made": int(self.model_calls_made),
            "planning_cost": copy.deepcopy(self.planning_cost),
            "budget_confirmation": self.budget_confirmation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanResult":
        return cls(
            plan_id=str(data["plan_id"]),
            root_snapshot_id=str(data.get("root_snapshot_id", "")),
            decision_action_id=data.get("decision_action_id"),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            limits=PlanLimits.from_dict(data.get("limits")),
            paths=[CandidatePath.from_dict(p)
                   for p in data.get("paths") or []],
            suggested=(ActionSpec.from_dict(data["suggested"])
                       if data.get("suggested") else None),
            suggestion_basis=str(data.get("suggestion_basis", "")),
            status=str(data.get("status", "ok")),
            truncation_reason=data.get("truncation_reason"),
            model_calls_made=int(data.get("model_calls_made", 0)),
            planning_cost=copy.deepcopy(data.get("planning_cost")),
            budget_confirmation=str(data.get("budget_confirmation",
                                             "unknown")),
            created_at=float(data.get("created_at", time.time())),
        )


def predicted_cost_vector(prediction: OutcomePrediction) -> CostVector:
    """The predicted incremental cost of ONE step.

    The measured mask comes from the persisted ``cost_measured`` key
    (written by PredictionService). Dimensions absent from the mask are
    UNKNOWN, never silently treated as a predicted zero — a candidate
    that predicted only tokens must not be scored as "zero solver
    runtime". Legacy rows without the mask fall back to "non-null value
    implies predicted" (value-level, same discipline as CostVector)."""
    raw = (prediction.predicted or {}).get("cost")
    if not isinstance(raw, dict):
        return CostVector(measured=set())
    vector = CostVector(
        **{d: float(v) for d, v in raw.items()
           if d in COST_DIMENSIONS and v is not None})
    mask = (prediction.predicted or {}).get("cost_measured")
    if isinstance(mask, list):
        vector.measured = {str(d) for d in mask if d in COST_DIMENSIONS}
    else:
        vector.measured = {d for d, v in raw.items()
                           if d in COST_DIMENSIONS and v is not None}
    return vector


def terminal_quality(prediction: OutcomePrediction) -> Optional[float]:
    """The predicted terminal quality of a path's LAST valid step.

    Quality yardstick: the predicted quality of the solution the path ends
    with (predicted ``quality`` in [0,1], 1 = optimal). A step that
    predicts infeasibility contributes 0 quality. Missing prediction =>
    None (unknown — reported, never defaulted)."""
    if prediction.status != "valid":
        return None
    predicted = prediction.predicted or {}
    if predicted.get("feasible") is False:
        return 0.0
    quality = predicted.get("quality")
    if quality is None:
        return None
    return float(quality)


def terminal_risk(prediction: OutcomePrediction) -> Optional[float]:
    """Terminal failure risk: the LAST step's predicted failure
    probability. Step risks are deliberately NOT composed (no
    independence assumption is made)."""
    if prediction.status != "valid":
        return None
    risk = (prediction.predicted or {}).get("failure_prob")
    return float(risk) if risk is not None else None


def evaluate_path(steps_predictions: List[OutcomePrediction],
                  limits: PlanLimits,
                  norms: Dict[str, float],
                  cost_basis: List[str]) -> CandidatePath:
    """Utility of one path: U = alpha*Q_terminal - beta*C_path - gamma*R.

    ``norms``/``cost_basis`` are the COMMON yardstick of this comparison
    (computed once over all candidates' predicted costs, restricted to
    dimensions every predicted cost measured — missing data never scores
    as cheap). The path cost sums only the steps' predicted INCREMENTAL
    costs; the planning calls' own spend is sunk and excluded."""
    steps = [PathStep(action_spec=p.action_spec,
                      prediction_id=p.prediction_id,
                      status=p.status)
             for p in steps_predictions]
    path = CandidatePath(steps=steps)
    valid = [p for p in steps_predictions if p.status == "valid"]
    if not valid:
        path.incomparable["all"] = ("no valid prediction on this path "
                                    f"(statuses: {[p.status for p in steps_predictions]})")
        return path
    terminal = valid[-1]
    q = terminal_quality(terminal)
    r = terminal_risk(terminal)
    if q is None:
        path.incomparable["quality"] = ("terminal quality not predicted "
                                        "(unknown never auto-scores)")
    # Unknown risk is a DEFICIT, not a zero: a candidate whose failure
    # risk is unknown must not beat one with a known non-zero risk. It is
    # charged the full risk weight and reported.
    risk_effective = r if r is not None else 1.0
    if r is None:
        path.incomparable["risk"] = (
            "terminal failure risk not predicted; charged the full gamma "
            "weight (unknown risk is a deficit, never free)")
    # Path cost: predicted increments of every VALID step over the
    # decision's common basis (the UNION of dimensions any candidate
    # measured). A step that did not measure a basis dimension is charged
    # that dimension's PEAK normalized share — the conservative penalty
    # keeps unknown cost from ever being free, and one silent candidate
    # can never erase the measured cost differences between the others.
    # Explicit predicted zeros stay zeros (they were measured); only a
    # MISSING dimension is charged the penalty.
    c_path = 0.0
    cost_known = False
    missing_dims: List[str] = []
    weights = limits.cost_weights or {}
    for prediction in valid:
        vector = predicted_cost_vector(prediction)
        measured = vector.measured_dims()
        if set(cost_basis) & measured:
            cost_known = True
        for dim in cost_basis:
            if dim in measured:
                c_path += (weights.get(dim, 0.0)
                           * getattr(vector, dim)
                           / max(norms.get(dim, 1.0), 1e-9))
            else:
                # Unknown on this basis dimension: peak normalized share
                # (weight * peak/peak = weight). A missing step never
                # gives the path a free advantage on that dimension.
                c_path += weights.get(dim, 0.0)
                if dim not in missing_dims:
                    missing_dims.append(dim)
    if missing_dims:
        path.incomparable["cost"] = (
            f"cost dimensions {sorted(missing_dims)} not predicted by "
            "every step; charged each at the peak normalized share of "
            "the candidates that measured them (unknown cost is never "
            "free)")
    if not cost_known and cost_basis:
        path.incomparable.setdefault(
            "cost_basis",
            "no step measured any basis dimension; the entire cost term "
            "is the conservative penalty, not a measurement — no cost "
            "optimization claim is made on this basis")
    elif not cost_basis:
        path.incomparable["cost"] = ("no candidate carried any cost "
                                     "prediction; the cost basis is "
                                     "incomparable, never fabricated")
    path.q_terminal = q
    path.r_terminal = r
    path.c_path = round(c_path, 6)
    alpha = limits.alpha if limits.alpha is not None else 1.0
    beta = limits.beta if limits.beta is not None else 1.0
    gamma = limits.gamma if limits.gamma is not None else 1.0
    path.utility = round(
        alpha * (q if q is not None else 0.0)
        - beta * c_path
        - gamma * risk_effective, 6)
    if path.incomparable:
        path.notes.append(
            "incomparable fields are charged conservatively (unknown "
            "risk => full gamma weight; unknown cost dimension => the "
            "peak normalized share of the candidates that measured it) "
            "and reported explicitly — unknown never auto-wins")
    return path


def comparison_norms(predictions: List[OutcomePrediction],
                     cost_weights: Optional[Dict[str, float]] = None
                     ) -> tuple[List[str], Dict[str, float]]:
    """The common cost yardstick for one comparison: every dimension
    predicted by AT LEAST ONE valid prediction (union), and per-dimension
    normalization divisors from the candidates that DID measure it
    (selector-style pure computation — no bank reads).

    A candidate that did NOT measure a basis dimension is NOT let off:
    ``evaluate_path`` charges it that dimension's maximum normalized
    share (the conservative penalty — unknown cost never wins by
    default, and one silent candidate never erases the cost differences
    between the others). A dimension NO candidate measured is simply
    absent from the basis (reported as incomparable, never fabricated).

    Also computes ``_max_c_path``: the largest weighted per-step cost
    among candidates that DID predict on the basis — kept for backward
    compatibility with stored plans."""
    valid = [p for p in predictions if p.status == "valid"]
    if not valid:
        return [], {}
    measured_sets = [predicted_cost_vector(p).measured_dims() for p in valid]
    basis: List[str] = sorted(set().union(*measured_sets))
    norms: Dict[str, float] = {}
    # Per-dimension peak among the candidates that MEASURED it (a
    # candidate that did not measure a dimension contributes nothing to
    # its yardstick — its share is charged later at this peak).
    for dim in basis:
        measured_values = [getattr(predicted_cost_vector(p), dim)
                           for p in valid
                           if dim in predicted_cost_vector(p).measured_dims()]
        peak = max(measured_values) if measured_values else 0.0
        norms[dim] = float(peak) if peak > 0 else 1.0
    # Per-step maximum weighted cost over the basis dimensions the
    # candidate measured (unmeasured dims charged at full peak share).
    weights = cost_weights or {}
    max_step_cost = 0.0
    for prediction in valid:
        vector = predicted_cost_vector(prediction)
        measured = vector.measured_dims()
        step_cost = 0.0
        for dim in basis:
            if dim in measured:
                step_cost += (weights.get(dim, 0.0)
                              * getattr(vector, dim)
                              / max(norms.get(dim, 1.0), 1e-9))
            else:
                # Unknown on this dimension: charged at the dimension's
                # peak normalized share (i.e. weight * peak/peak = weight).
                step_cost += weights.get(dim, 0.0)
        max_step_cost = max(max_step_cost, step_cost)
    norms["_max_c_path"] = max_step_cost
    return basis, norms


def build_hypothetical_successor(root: BeliefSnapshot,
                                 prediction: OutcomePrediction,
                                 task: Dict[str, Any]) -> "BeliefSnapshot":
    """Construct the hypothetical post-state of a predicted first step.

    Reuses the root snapshot's frozen H/P (knowledge, tools, problem) —
    a prediction never rewrites knowledge — and updates only the X/B
    fields the prediction actually speaks about:

    - ``state_changes.current_solution`` (or the whole ``state_changes``
      object) becomes X.current_solution with ``epistemic="inferred"``
      and the prediction as evidence ref;
    - the predicted cost increment is noted in B as an INFERRED
      hypothetical consumption (never entering the real ledger);
    - everything not predicted stays exactly as frozen at the root
      (unknown stays unknown).

    The result is marked ``hypothetical=True`` and is excluded from every
    real-state query by the existing BeliefSnapshot discipline. It is a
    LIGHTWEIGHT VIEW: it is not persisted as a real snapshot row by this
    function — the caller decides whether to persist it for auditability.
    """
    changes = (prediction.predicted or {}).get("state_changes") or {}
    progress = copy.deepcopy(root.task_progress)
    if changes:
        solution = changes.get("current_solution", changes)
        progress["current_solution"] = {
            "value": copy.deepcopy(solution),
            "provenance": "observed",
            "epistemic": "inferred",
            "evidence_ref": prediction.prediction_id,
            "hypothetical": True,
        }
    budget = copy.deepcopy(root.budget_state)
    vector = predicted_cost_vector(prediction)
    if vector.measured_dims():
        budget["hypothetical_step_cost"] = {
            "cost": vector.to_dict(),
            "measured": sorted(vector.measured_dims()),
            "epistemic": "inferred",
            "evidence_ref": prediction.prediction_id,
            "note": "predicted increment of the imagined step; the REAL "
                    "ledger is untouched",
        }
    return BeliefSnapshot.build(
        task, root.episode_id,
        harness_state=copy.deepcopy(root.harness_state),
        problem_state=copy.deepcopy(root.problem_state),
        task_progress=progress,
        budget_state=budget,
        coverage=copy.deepcopy(root.coverage),
        hypothetical=True,
    )


def second_step_dependency_ok(prediction: OutcomePrediction) -> bool:
    """Whether the first prediction established what a follow-up execution
    needs: a predicted current_solution (incumbent) or an explicit
    feasible outcome. A bare target value is NOT a usable incumbent; when
    the dependency is unmet the second step is conditional-only."""
    if prediction.status != "valid":
        return False
    predicted = prediction.predicted or {}
    changes = predicted.get("state_changes") or {}
    if changes.get("current_solution"):
        return True
    return predicted.get("feasible") is True
