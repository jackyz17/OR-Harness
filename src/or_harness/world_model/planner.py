"""One-step strategy scoring and readable historical plan records.

Current planning compares caller-proposed strategies once under a shared frozen
context with comparable G and common available C/R. Resource scales come from
configuration or applicable prior facts; H+ schedules later review independently.
Planning-call spend is real and charged separately from candidate utility.

PathStep, CandidatePath and the legacy PlanResult fields remain serialization
records for inspecting saved two-step experiments. The retired hypothetical
successor and knowledge-bonus calculations are not a live planning mode.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from or_harness.core.schema import (
    COST_DIMENSIONS, DEFAULT_PREDICTED_COST_WEIGHTS,
    LEGACY_PREDICTION_COST_DIMENSIONS, PREDICTED_COST_DIMENSIONS,
)
from or_harness.core.schema import chargeable_dimensions
from or_harness.world_model.contracts import (
    FAITHFUL_G_CONVENTION, FAITHFUL_G_METRIC, FAITHFUL_G_OBSERVATION_VERSION, is_g_probability,
)
from or_harness.world_model.prediction import ActionSpec

@dataclass
class PlanLimits:
    """Hard bounds for one planning call (engineering defaults, recorded
    verbatim into every plan for later sensitivity analysis and ablation).

    ``max_model_calls`` caps the number of world-model invocations of the
    whole decision (one prediction per retained candidate). ``time_budget_s`` caps the
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
    #: Candidate-independent resource peaks; absent dimensions need applicable
    #: pre-decision measured history. No candidate maximum is a scale.
    cost_scales: Optional[Dict[str, Any]] = None
    #: Legacy experiment setting retained for saved-plan compatibility.
    #: Current strategy U uses only G/C/R regardless of this field.
    delta: Optional[float] = None
    #: The benefit CONVENTION this ONE decision compares under. ``None`` on
    #: both = the candidates' own agreement is used when they share one, else
    #: the build's default. Declaring it lets a decision compare completion
    #: (``effective_completion`` / ``task_result_check_passed``) instead of
    #: quality — the same convention every candidate is then predicted and
    #: evaluated under.
    benefit_kind: Optional[str] = None
    benefit_metric: Optional[str] = None
    benefit_unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_root_candidates": int(self.max_root_candidates),
            "horizon": int(self.horizon),
            "max_model_calls": int(self.max_model_calls),
            "time_budget_s": float(self.time_budget_s),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "benefit_kind": self.benefit_kind,
            "benefit_metric": self.benefit_metric,
            "benefit_unit": self.benefit_unit,
            "cost_weights": (dict(self.cost_weights)
                             if self.cost_weights is not None else None),
            "cost_scales": copy.deepcopy(self.cost_scales),
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
            delta=(None if data.get("delta") is None
                   else float(data["delta"])),
            benefit_kind=(str(data["benefit_kind"])
                          if data.get("benefit_kind") else None),
            benefit_metric=(str(data["benefit_metric"])
                            if data.get("benefit_metric") else None),
            benefit_unit=(str(data["benefit_unit"])
                          if data.get("benefit_unit") else None),
            cost_weights=(None if data.get("cost_weights") is None
                          else dict(data["cost_weights"])),
            cost_scales=copy.deepcopy(data.get('cost_scales')),
        )


@dataclass
class PathStep:
    """A serialized step from a historical candidate path."""

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
    """Historical candidate/path decomposition, retained for inspection."""

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
    #: Knowledge term of this path: the value actually added to the utility
    #: and an honest label of what it is. ``delta_knowledge`` is 0.0 when no
    #: justified value existed (unknown upside is NEVER rewarded) — the
    #: ``knowledge_detail`` says whether that zero means "computed as zero"
    #: or "no justified value was claimable".
    delta_knowledge: float = 0.0
    knowledge_detail: Dict[str, Any] = field(default_factory=dict)

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
            "delta_knowledge": self.delta_knowledge,
            "knowledge_detail": copy.deepcopy(self.knowledge_detail),
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
            delta_knowledge=float(data.get("delta_knowledge") or 0.0),
            knowledge_detail=dict(data.get("knowledge_detail") or {}),
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
    #: The ONE benefit convention this decision compared under (kind,
    #: metric, unit, source). Recorded so a reader can see WHICH currency
    #: the ranking used, and whether it was declared, agreed or defaulted.
    benefit_convention: Optional[Dict[str, Any]] = None
    selected_prediction_id: Optional[str] = None
    selected_score: Optional[Dict[str, Any]] = None
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
            "benefit_convention": copy.deepcopy(self.benefit_convention),
            "selected_prediction_id":self.selected_prediction_id,
            "selected_score":copy.deepcopy(self.selected_score),
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
            benefit_convention=copy.deepcopy(data.get("benefit_convention")),
            selected_prediction_id=data.get('selected_prediction_id'),
            selected_score=copy.deepcopy(data.get('selected_score')),
            created_at=float(data.get("created_at", time.time())),
        )


# ---------------------------------------------------------------------------
# strategy-outcome protocol (M3): comparing StrategyOutcomePrediction objects
# ---------------------------------------------------------------------------


@dataclass
class StrategyOutcomeScore:
    """The conservative comparison result for ONE strategy-outcome candidate.

    The yardstick is deliberately simple and explainable:

    - ``benefit_value``: the candidate's own benefit value, when the metric
      is comparable (this build compares NORMALIZED solution-quality
      benefits; a candidate whose benefit is absent or a different currency
      reports ``None`` and never auto-wins);
    - ``cost_normalized``: cost on shared, same-scope declared dimensions;
      no common dimensions means unknown C, omitted for the comparison;
    - ``risk_effective``: the MAXIMUM single-event probability over the R
      events (implementation_failure / timeout / environment_failure) that
      ALL ranked candidates supplied validly; when no event is common, or the
      candidates do not share one R scoring convention, R is omitted for
      everyone. It is a maximum single-item RISK SCORE, never a joint
      failure probability; ``risk_probability`` preserves the legacy single
      ``implementation_failure`` probability and ``risk_reported`` every
      original prediction;
    - ``utility``: ``alpha*G - beta*C - gamma*R`` with the SAME weights the
      harness scores everything else with. The knowledge term is OFF for
      this protocol (delta=0): an old knowledge-gain score is not an H
      improvement and must not leak into the new comparison.
    """

    prediction_id: Optional[str] = None
    benefit_value: Optional[float] = None
    cost_normalized: Optional[float] = None
    risk_effective: Optional[float] = None
    utility: Optional[float] = None
    #: EVERY predicted risk event (name + probability), including any whose
    #: loss is already inside the benefit under this convention. This is the
    #: DIAGNOSTIC view: ``risk_effective`` is what the utility charged, this
    #: is what was predicted. The two differ when a loss is not double-counted.
    risk_reported: List[Dict[str, Any]] = field(default_factory=list)
    #: The LEGACY single ``implementation_failure`` probability, retained for
    #: continuity/diagnostics. The multi-event R actually charged is
    #: ``risk_effective`` (a maximum over the common valid events).
    risk_probability: Optional[float] = None
    cost_reported: Dict[str, Any] = field(default_factory=dict)
    scoring_basis: Dict[str, Any] = field(default_factory=dict)
    incomparable: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "benefit_value": self.benefit_value,
            "cost_normalized": self.cost_normalized,
            "risk_effective": self.risk_effective,
            "risk_reported": [dict(e) for e in self.risk_reported],
            "risk_probability": self.risk_probability,
            "cost_reported": copy.deepcopy(self.cost_reported),
            "scoring_basis": copy.deepcopy(self.scoring_basis),
            "utility": self.utility,
            "incomparable": dict(self.incomparable),
            "notes": list(self.notes),
        }


#: The benefit KINDS this build can compare. ``effective_completion`` (the
#: primary, task-completion convention) and ``solution_quality`` (the
#: solver's own gap) are DIFFERENT currencies and are never mixed within
#: one decision; each decision compares under ONE of them.
COMPARABLE_BENEFIT_KINDS = ("solution_quality", "effective_completion")

#: The benefit METRICS on that same default yardstick. The kind alone is not
#: enough: two ``solution_quality`` predictions that measure different things
#: (a solver-side normalized gap vs some other normalized score) are not one
#: currency, so the metric must match too.
COMPARABLE_BENEFIT_METRIC = "normalized_objective_gap"

#: Declared-metric spellings that resolve to the default comparable metric.
COMPARABLE_BENEFIT_METRIC_ALIASES = {
    "normalized_objective_gap": "normalized_objective_gap",
    "solution_quality": "normalized_objective_gap",
    "normalized_quality": "normalized_objective_gap",
    "normalized_gap": "normalized_objective_gap",
}

#: The ONE completion convention this comparison ALSO understands. It is not
#: a synonym for quality: it measures whether the ANSWER satisfies the TASK,
#: and candidates predicted under it are compared against EACH OTHER (never
#: against a quality candidate). A decision that wants to compare completion
#: probabilities declares this convention explicitly.
COMPLETION_BENEFIT_KIND = "effective_completion"
COMPLETION_BENEFIT_METRIC = "task_result_check_passed"
COMPLETION_BENEFIT_UNIT = "boolean"

#: The conventions a decision may DECLARE, keyed by ``(kind, metric)`` after
#: normalisation. Anything outside this table cannot be compared: an unknown
#: currency has no shared scale, and inventing one would rank numbers that
#: do not mean the same thing.
KNOWN_BENEFIT_CONVENTIONS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("effective_completion", FAITHFUL_G_METRIC): dict(FAITHFUL_G_CONVENTION),
    ("solution_quality", COMPARABLE_BENEFIT_METRIC): {
        "unit": "1-gap",
        "observed_from": ("the solver's own normalized gap (an `optimal` "
                          "status is a gap of 0; otherwise 1-gap)"),
    },
    (COMPLETION_BENEFIT_KIND, COMPLETION_BENEFIT_METRIC): {
        "unit": COMPLETION_BENEFIT_UNIT,
        "observed_from": ("the execution's own task-result check: 1.0 "
                          "passed, 0.0 confirmed failed, UNKNOWN when "
                          "unchecked or insufficient"),
    },
}

#: The PRIMARY benefit is original-task acceptance given faithful execution.
#: Explicit legacy completion/gap forecasts keep their registered conventions. The
#: solver's own gap measures how well the MODEL was solved, which is NOT
#: the same as the agent completing the task it was given — defaulting to
#: it let an optimal solve of a mis-modelled problem read as full success.
#: ``solution_quality`` stays a fully supported, explicitly declarable
#: convention; it is simply no longer the default.
PRIMARY_BENEFIT_CONVENTION = ("effective_completion", FAITHFUL_G_METRIC)

#: The default convention when a decision declares none and the candidates
#: do not agree on one. It is the PRIMARY (task completion) convention: the
#: main benefit of choosing a method is that the ORIGINAL TASK gets
#: completed, so absent any declaration the comparison optimises that.
DEFAULT_BENEFIT_CONVENTION = PRIMARY_BENEFIT_CONVENTION


#: Risk events whose loss is ALREADY represented by the benefit when the
#: decision compares under task completion. ``task_check_failed`` is the
#: complement of ``effective_completion``: a candidate predicted to complete
#: the task with probability p has already had (1-p) of a task-check failure
#: priced into its benefit. Charging the SAME loss again through the risk
#: term would double-count one loss. The probability is still REPORTED (for
#: diagnosis and calibration) — it is only removed from the UTILITY term.
#: This list applies ONLY when the convention is completion; under
#: ``solution_quality`` a task-check failure is an independent loss.
BENEFIT_IMPLIED_RISK_EVENTS = {
    COMPLETION_BENEFIT_KIND: ("task_check_failed",),
}


def comparable_benefit_metric(metric: Any) -> Optional[str]:
    """The canonical DEFAULT metric a declared metric maps to, or None.

    ``None`` means this build has no common currency for that metric: the
    benefit is reported, and it does NOT enter the utility ranking — an
    incomparable upside is not silently scored as zero (which would let it
    be ranked as if it were a real, worthless benefit).
    """
    name = str(metric or "").strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    return COMPARABLE_BENEFIT_METRIC_ALIASES.get(name)


def normalize_benefit_convention(kind: Any, metric: Any
                                 ) -> Optional[Tuple[str, str]]:
    """The canonical ``(kind, metric)`` a declaration names, or None.

    ``None`` = this build has no comparable convention for that pair, so the
    decision cannot rank benefits by it (the caller reports that rather than
    fabricating a scale).
    """
    k = str(kind or "").strip().lower().replace(" ", "_").replace("-", "_")
    m = str(metric or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not k or not m:
        return None
    if k == "solution_quality":
        canonical = comparable_benefit_metric(m)
        if canonical:
            return ("solution_quality", canonical)
        return None
    if k == COMPLETION_BENEFIT_KIND:
        if m == FAITHFUL_G_METRIC:
            return (k, m)
        if m in (COMPLETION_BENEFIT_METRIC, "task_check_passed",
                 "completion_rate"):
            return (COMPLETION_BENEFIT_KIND, COMPLETION_BENEFIT_METRIC)
        return None
    return None


def benefit_implied_risk_events(kind: Any) -> Tuple[str, ...]:
    """Risk event names already priced into the benefit under ``kind``.

    Returns the events whose loss the benefit's own distribution already
    carries, so the utility term must not charge them a second time. Empty
    when no such overlap exists (e.g. under ``solution_quality``, where a
    task-check failure is an INDEPENDENT loss beside the solver quality).
    """
    k = str(kind or "").strip().lower().replace(" ", "_").replace("-", "_")
    return BENEFIT_IMPLIED_RISK_EVENTS.get(k, ())


def resolve_benefit_convention(predictions: List[Any], *, kind: Any = None,
                               metric: Any = None
                               ) -> Dict[str, Any]:
    """The ONE convention a decision compares under, and where it came from.

    Precedence, and it is reported rather than implied:

    1. **declared** — the caller named ``kind``/``metric`` for this decision.
       Every valid candidate must then be predicted under it; a mismatch is
       reported per candidate and never coerced.
    2. **agreed** — no declaration, and every valid prediction that CARRIES a
       benefit declares the SAME ``(kind, metric)``. This is the case that
       makes same-convention comparison work without ceremony: two candidates
       both predicting ``effective_completion`` are compared on completion,
       not rejected for not being quality.
    3. **default** — otherwise the build's default yardstick
       (``solution_quality`` / ``normalized_objective_gap``) is used, and the
       disagreement is reported so a reader sees that some candidates were
       not on it.

    Returns ``{"kind", "metric", "unit", "source", "reason",
    "observed_from"}``. An unrecognised DECLARATION is reported as
    ``source="unknown_declaration"`` with ``comparable=False``: no ranking is
    produced, because ranking by an invented scale would be worse than
    saying there is none.
    """
    if kind is not None or metric is not None:
        resolved = normalize_benefit_convention(kind, metric)
        if resolved is None:
            return {
                "kind": (str(kind) if kind else None),
                "metric": (str(metric) if metric else None),
                "unit": None,
                "source": "unknown_declaration",
                "comparable": False,
                "reason": (f"this build has no comparable benefit convention "
                           f"for kind={kind!r} metric={metric!r}; the "
                           f"declared currencies are "
                           f"{sorted(KNOWN_BENEFIT_CONVENTIONS)}. Ranking by "
                           "an invented scale would be worse than reporting "
                           "that nothing is comparable"),
            }
        k, m = resolved
        return {
            "kind": k, "metric": m,
            "unit": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["unit"],
            "source": "declared",
            "comparable": True,
            "reason": ("declared by the caller for this decision, so every "
                       "candidate is compared on the SAME currency"),
            "observed_from": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["observed_from"],
        }
    declared: Dict[Tuple[str, str], int] = {}
    for prediction in predictions:
        if getattr(prediction, "status", None) != "valid":
            continue
        benefit = getattr(prediction, "benefit", None)
        if benefit is None or (benefit.value is None and benefit.metric != FAITHFUL_G_METRIC):
            continue
        resolved = normalize_benefit_convention(benefit.kind, benefit.metric)
        if resolved is None:
            continue
        declared[resolved] = declared.get(resolved, 0) + 1
    if len(declared) == 1:
        k, m = next(iter(declared))
        return {
            "kind": k, "metric": m,
            "unit": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["unit"],
            "source": "agreed",
            "comparable": True,
            "reason": (f"every candidate carrying a benefit declared the "
                       f"same convention ({k}/{m}), so it was used for the "
                       "comparison without being declared"),
            "observed_from": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["observed_from"],
        }
    k, m = DEFAULT_BENEFIT_CONVENTION
    detail = (f"the candidates did not agree on one convention "
              f"({ {f'{a}/{b}': n for (a, b), n in declared.items()} })"
              if len(declared) > 1 else "no candidate declared a benefit")
    return {
        "kind": k, "metric": m,
        "unit": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["unit"],
        "source": "default",
        "comparable": True,
        "reason": (f"{detail}; the build's PRIMARY yardstick ({k}/{m} — "
                   "whether the ORIGINAL TASK was effectively completed) "
                   "was used as the default. Declare a convention (e.g. "
                   "solution_quality with normalized_objective_gap) to "
                   "compare a different currency"),
        "observed_from": KNOWN_BENEFIT_CONVENTIONS[(k, m)]["observed_from"],
    }


def _legacy_cost_weight_diagnostic(weights: Dict[str, float]
                                   ) -> Optional[Dict[str, Any]]:
    """Explain ignored legacy configuration without changing the yardstick."""
    ignored = sorted(set(weights) & set(LEGACY_PREDICTION_COST_DIMENSIONS))
    if not ignored:
        return None
    positive = sorted(d for d in PREDICTED_COST_DIMENSIONS
                      if isinstance((value := weights.get(d)), (int, float))
                      and not isinstance(value, bool)
                      and math.isfinite(value) and value > 0)
    missing = sorted(set(PREDICTED_COST_DIMENSIONS) - set(positive))
    return {
        "ignored_legacy_dimensions": ignored,
        "positive_active_dimensions": positive,
        "active_dimensions_without_positive_weight": missing,
        "message": (
            "Warning: legacy cost-weight dimensions are ignored by active C: "
            + ", ".join(ignored) + ". Positive active configured dimensions "
            + f"({len(positive)}/{len(PREDICTED_COST_DIMENSIONS)}): "
            + (", ".join(positive) or "none")
            + ". Active dimensions without a positive configured weight: "
            + (", ".join(missing) or "none")
            + ". Explicit weights still replace defaults; no weights were "
              "added. Charged dimensions depend on common valid predictions "
              "and independent scales; see scoring_basis.cost_dimensions."),
    }


def score_strategy_outcome_predictions(
        predictions: List[Any], limits: PlanLimits, *, cost_scales=None
        ) -> List[StrategyOutcomeScore]:
    """Compare rankable candidates using shared C dimensions and shared R.

    Scales are independent of candidate values. Missing fields or scales
    remove a term/dimension for the entire comparison, never add a penalty
    to one candidate. Original declarations remain available for inspection.
    """
    from or_harness.world_model.r_signal import implementation_probability
    convention = resolve_benefit_convention(predictions,
        kind=getattr(limits, "benefit_kind", None), metric=getattr(limits, "benefit_metric", None))

    def benefit_reason(prediction):
        benefit = prediction.benefit
        if prediction.status != "valid":
            return "prediction is not valid"
        if not convention["comparable"]:
            return convention["reason"]
        if benefit is None or benefit.value is None:
            return "no benefit value was predicted"
        declared = normalize_benefit_convention(benefit.kind, benefit.metric)
        if declared != (convention["kind"], convention["metric"]):
            return "benefit kind/metric differs from this decision's convention; different currencies never rank together"
        if benefit.metric == FAITHFUL_G_METRIC and (
                not is_g_probability(benefit.value) or benefit.unit != "probability" or
                benefit.observation_rule_version != FAITHFUL_G_OBSERVATION_VERSION):
            return "invalid faithful G value/unit/observation rule"
        if not isinstance(benefit.value, (int, float)) or isinstance(benefit.value, bool) or not math.isfinite(benefit.value):
            return "benefit value is not finite numeric data"
        return None

    def cost_dims(prediction):
        cost = prediction.cost
        if cost is None or cost.expected is None:
            return set()
        # U prices ONLY the four remaining-task C targets. A legacy
        # observation dimension (latency_s / solver_runtime_s) stays in the
        # stored prediction for audit, but is never a new U term.
        return {dim for dim in set(cost.measured) & set(cost.expected.measured_dims())
                & set(PREDICTED_COST_DIMENSIONS)
                if isinstance((v := getattr(cost.expected, dim)), (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0}

    def measurement_key(prediction, dim):
        if dim not in PREDICTED_COST_DIMENSIONS:
            return None
        # Every active dimension is defined over the SAME remaining window.
        return ("remaining_to_task_end"
                if prediction.cost.scope == "remaining_to_task_end" else None)

    participants = [p for p in predictions if benefit_reason(p) is None]
    declared_dims = {p.prediction_id: cost_dims(p) for p in predictions}
    common = set.intersection(*(declared_dims[p.prediction_id] for p in participants)) if participants else set()
    all_dims = set.union(*(declared_dims[p.prediction_id] for p in participants)) if participants else set()
    omitted_dims = {}
    for dim in sorted(all_dims):
        missing = [p.prediction_id for p in participants if dim not in declared_dims[p.prediction_id]]
        keys = {measurement_key(p, dim) for p in participants if dim in declared_dims[p.prediction_id]}
        if missing:
            omitted_dims[dim] = {"reason": "missing/invalid expected_measured prediction", "prediction_ids": missing}
        elif None in keys or len(keys) != 1:
            common.discard(dim)
            omitted_dims[dim] = {"reason": "different or unknown measurement scopes"}
        elif dim=='retries':
            from or_harness.world_model.cost_signal import retries_convention
            if len({retries_convention(p) for p in participants})!=1:
                common.discard(dim)
                omitted_dims[dim]={'reason':'different retry observation conventions'}
    # Archived observation dimensions are kept in the prediction record but
    # are NEVER active C terms: no U weight, no scale, no calibration entry.
    for prediction in predictions:
        legacy = set(getattr(prediction.cost, "measured", []) or [])
        for dim in sorted(legacy - set(PREDICTED_COST_DIMENSIONS)):
            omitted_dims.setdefault(dim, {
                "reason": ("historical observation dimension; not an active "
                           "four-dimensional C prediction target")})
    # Keep old declarations inspectable in their own units. Strategy U prices
    # generated tokens, never the raw input-plus-output total.
    if "llm_tokens" in all_dims:
        common.discard("llm_tokens")
        omitted_dims["llm_tokens"] = {"reason": "raw input-plus-output total; token C uses output_tokens"}
    scales=cost_scales if cost_scales is not None else limits.cost_scales or {}
    if not isinstance(scales,dict):scales={}
    norms={};scale_sources={}
    # Only an unspecified map receives defaults. Explicit partial/empty maps
    # and zero weights must not acquire an implicit output-token charge.
    weights = dict(DEFAULT_PREDICTED_COST_WEIGHTS
                   if limits.cost_weights is None else limits.cost_weights)
    weight_diagnostic = _legacy_cost_weight_diagnostic(weights)
    for dim in sorted(common):
        raw=scales.get(dim)
        info=copy.deepcopy(raw) if isinstance(raw,dict) else {'peak':raw,'source':'explicit_plan_limits'}
        peak=info.get('peak')
        key=measurement_key(participants[0],dim)
        if (not isinstance(peak,(int,float)) or isinstance(peak,bool) or
                not math.isfinite(peak) or peak<=0 or info.get('scope',key)!=key):
            common.discard(dim)
            omitted_dims[dim]={'reason':'no reliable positive candidate-independent scale for this measurement scope'}
            continue
        weight=weights.get(dim,0.0)
        if not isinstance(weight,(int,float)) or isinstance(weight,bool) or not math.isfinite(weight) or weight<=0:
            common.discard(dim);omitted_dims[dim]={'reason':'no positive finite cost weight'};continue
        norms[dim]=float(peak);scale_sources[dim]={**info,'peak':float(peak),'scope':key}
    # Intersect and establish usable scales FIRST; then deduplicate the
    # dimensions actually charged, with identical normalized weights.
    basis = sorted(chargeable_dimensions(common))
    for dim in sorted(common - set(basis)):
        omitted_dims[dim] = {"reason": "component deduplicated on common dimensions"}
    norms={d:norms[d] for d in basis};scale_sources={d:scale_sources[d] for d in basis}
    probabilities = {p.prediction_id: implementation_probability(p) for p in predictions}
    # -- shared R convention and common event set ---------------------------
    # R is the MAXIMUM single-event probability over the events every ranked
    # candidate supplied validly. The set E is COMMON: an event a participant
    # left unknown (or gave an unusable probability for) is dropped for the
    # WHOLE comparison — an unknown is never imputed as 0, and one candidate's
    # extra event never becomes R for candidates that did not supply it.
    from or_harness.world_model.contracts import (
        R_AGGREGATION_NOTE, R_EVENTS_BY_SCORING_VERSION,
    )
    from or_harness.world_model.r_signal import (
        r_event_probabilities, r_scoring_version,
    )
    participant_ids = [p.prediction_id for p in participants]
    declared_rules = {r_scoring_version(p) for p in participants}
    scoring_rule = (next(iter(declared_rules))
                    if len(declared_rules) == 1 else None)
    candidate_events = tuple(R_EVENTS_BY_SCORING_VERSION.get(
        scoring_rule, ())) if scoring_rule else ()
    risk_usable: Dict[str, Dict[str, float]] = {}
    risk_unusable: Dict[str, Dict[str, str]] = {}
    risk_missing: Dict[str, List[str]] = {}
    for prediction in participants:
        usable, unusable, missing = r_event_probabilities(
            prediction, events=candidate_events)
        risk_usable[prediction.prediction_id] = usable
        risk_unusable[prediction.prediction_id] = unusable
        risk_missing[prediction.prediction_id] = missing
    common_events = [name for name in candidate_events
                     if all(name in risk_usable[pid] for pid in participant_ids)]
    omitted_risk_events: Dict[str, Dict[str, Any]] = {}
    for name in candidate_events:
        if name in common_events:
            continue
        reasons = {}
        for pid in participant_ids:
            if name in risk_usable[pid]:
                continue
            if risk_unusable[pid].get(name):
                reasons[pid] = risk_unusable[pid][name]
            elif name in risk_missing[pid]:
                reasons[pid] = ("no probability was supplied (unknown, never "
                                "imputed as 0)")
            else:
                reasons[pid] = "missing"
        omitted_risk_events[name] = {
            "prediction_ids": sorted(reasons),
            "reasons": reasons,
            "reason": "not common to every ranked candidate",
        }
    use_r = bool(participants) and bool(common_events)
    risk_effective: Dict[str, float] = {}
    risk_max_source: Dict[str, Dict[str, Any]] = {}
    if use_r:
        for pid in participant_ids:
            usable = risk_usable[pid]
            best_value = max(usable[name] for name in common_events)
            # Deterministic tie-break: the first declared event wins.
            best_event = next(name for name in common_events
                              if usable[name] == best_value)
            risk_effective[pid] = best_value
            risk_max_source[pid] = {"event": best_event,
                                    "probability": best_value}
    if not participants:
        omitted_risk_reason = "no rankable candidates"
    elif scoring_rule is None:
        omitted_risk_reason = (
            "ranked candidates declare different or unknown R scoring "
            "conventions; a shared R convention is required, so R is omitted "
            "for everyone rather than mixing old and new meanings")
    elif not common_events:
        omitted_risk_reason = (
            "no R event has a valid probability on every ranked candidate; R "
            "is omitted for everyone (an unknown is never imputed as 0)")
    else:
        omitted_risk_reason = None
    if scoring_rule is None:
        missing_r = list(participant_ids)
    else:
        missing_r = sorted({pid for pid in participant_ids
                            if any(name not in risk_usable[pid]
                                   for name in candidate_events)})
    total_weight=sum(weights[d] for d in basis)
    normalized_weights={d:weights[d]/total_weight for d in basis}
    alpha = limits.alpha if limits.alpha is not None else 1.0
    beta = limits.beta if limits.beta is not None else 1.0
    gamma = limits.gamma if limits.gamma is not None else 1.0
    scores = []
    for prediction in predictions:
        score = StrategyOutcomeScore(prediction_id=prediction.prediction_id)
        score.risk_probability = probabilities[prediction.prediction_id]
        score.risk_reported = [{"event": e.event, "probability": e.probability}
                              for e in getattr(prediction.risk, "events", [])]
        cost = prediction.cost
        score.cost_reported = {"scope": getattr(cost, "scope", None),
            "expected_measured": list(getattr(cost, "measured", [])),
            "values": {d: (getattr(cost.expected, d) if d in declared_dims[prediction.prediction_id] else None)
                       for d in getattr(cost, "measured", []) if d in PREDICTED_COST_DIMENSIONS} if cost and cost.expected else {}}
        reason = benefit_reason(prediction)
        score.scoring_basis = {
            "participant_prediction_ids": participant_ids,
            "terms": (["G"] + (["C"] if basis else []) + (["R"] if use_r else [])) if reason is None else [],
            "cost_dimensions": basis, "cost_scales": dict(norms),
            "cost_weights": normalized_weights,
            "cost_normalized_values": {},
            "cost_scale_sources":copy.deepcopy(scale_sources),
            "cost_normalization":"clip(log1p(predicted) / log1p(peak), 0, 1)",
            "utility_weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "cost_measurement_scopes": {d: measurement_key(participants[0], d) for d in basis},
            "omitted_cost_dimensions": copy.deepcopy(omitted_dims),
            "risk_scoring_rule": scoring_rule,
            "risk_events": list(common_events),
            "risk_aggregation": "max_common_event",
            "risk_interpretation": R_AGGREGATION_NOTE,
            "risk_event_probabilities": {
                pid: dict(risk_usable[pid]) for pid in participant_ids},
            "risk_event_problems": {
                pid: dict(risk_unusable[pid]) for pid in participant_ids
                if risk_unusable.get(pid)},
            "risk_max_source": copy.deepcopy(
                risk_max_source.get(prediction.prediction_id)),
            "risk_event": (common_events[0] if len(common_events) == 1 else None),
            "omitted_risk_events": copy.deepcopy(omitted_risk_events),
            "omitted_risk_reason": omitted_risk_reason,
            "missing_risk_prediction_ids": missing_r,
            "partial": not basis or not use_r or any(v["reason"] != "component deduplicated on common dimensions"
                                                    for v in omitted_dims.values()),
        }
        if weight_diagnostic is not None:
            score.scoring_basis["cost_weight_diagnostic"] = copy.deepcopy(weight_diagnostic)
            score.notes.append(weight_diagnostic["message"])
        if reason is not None:
            score.incomparable["prediction" if prediction.status != "valid" else "benefit"] = reason
            scores.append(score)
            continue
        score.benefit_value = float(prediction.benefit.value)
        normalized_values = {d: min(1.0, max(0.0,
            math.log1p(float(getattr(prediction.cost.expected, d)))
            / math.log1p(norms[d]))) for d in basis}
        # Preserve full precision here so the existing score audit can
        # reproduce C and U, each rounded only at its final output boundary.
        score.scoring_basis["cost_normalized_values"] = normalized_values
        cost_value = sum(normalized_weights[d] * normalized_values[d] for d in basis)
        score.cost_normalized = round(cost_value, 6) if basis else None
        if not basis:
            score.incomparable["cost"] = "no common same-scope valid cost dimensions; C omitted for all ranked candidates"
        score.risk_effective = risk_effective.get(prediction.prediction_id)
        if not use_r:
            score.incomparable["risk"] = (omitted_risk_reason
                or "R omitted for all ranked candidates")
        score.utility = round(alpha * score.benefit_value - beta * cost_value -
                              gamma * (score.risk_effective
                                       if score.risk_effective is not None
                                       else 0), 6)
        if not math.isfinite(score.utility):
            score.utility=None;score.incomparable['utility']='configured utility weights do not produce a finite U'
        if score.scoring_basis["partial"]:
            score.notes.append("partial utility over the recorded common terms; omitted data is unknown and incurs no candidate-specific penalty")
        scores.append(score)
    return scores
