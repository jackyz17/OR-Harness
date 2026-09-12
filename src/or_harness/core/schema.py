"""Core data schemas for or_harness.

Five schemas define the whole system:

- :class:`ProblemProfile`   — what the problem looks like (structure, coupling).
- :class:`Strategy`         — a named solving strategy with applicability and priors.
- :class:`CostVector`       — five-dimensional execution cost, never collapsed at rest.
- :class:`ExecutionRecord`  — Execution Evidence storage unit: an append-only fact.
- :class:`StrategicEntry`   — Strategic Knowledge storage unit: a calibrated commitment.

Terminology discipline (do not blur):
  Execution Evidence Bank = episodic facts ("what actually happened"):
                    actual strategy / actual quality / actual cost / observed
                    failures / implementation artifacts. No generalization
                    claims. Append-only; only cost dimensions may be
                    backfilled.
  Strategic Knowledge Bank = generalized commitments ("what to do next
                    time"): expected quality / expected cost / expected
                    failure risk, with prediction intervals and calibration
                    tracking. Induction-time validated: evidence must support
                    a candidate BEFORE admission; after admission an entry
                    carries only lightweight origin metadata (support_n,
                    optional representative execution ids) and its validity
                    does NOT depend on the survival of the original evidence
                    rows — compacting old evidence never invalidates an
                    entry. Never stores raw execution detail.
  Conditional statistics = on-the-fly aggregation over the Evidence Bank,
                    never persisted, always rebuildable. It is arithmetic, not
                    knowledge. An entry that merely restates statistics is
                    redundant and must not be created.
  Structural group = family + feature bins; the similarity key for statistics.

Actual vs expected naming: facts store ACTUAL observations (``quality``,
``cost`` — alias properties ``actual_quality`` / ``actual_cost`` make this
explicit); knowledge stores EXPECTED quantities (``expected_quality_hat``,
``expected_cost_hat``, ``failure_prob`` — aliases ``expected_quality`` /
``expected_cost`` / ``expected_failure_risk``). Serialization keys never
change; the aliases are API-level clarifications only.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Cost vector
# ---------------------------------------------------------------------------

COST_DIMENSIONS: Tuple[str, ...] = (
    "llm_tokens",
    "tool_calls",
    "solver_runtime_s",
    "retries",
    "latency_s",
)

#: Default scalarization weights. Used only inside the selector; never at rest.
DEFAULT_COST_WEIGHTS: Dict[str, float] = {
    "llm_tokens": 1.0,
    "tool_calls": 0.2,
    "solver_runtime_s": 0.5,
    "retries": 2.0,
    "latency_s": 0.1,
}


@dataclass
class CostVector:
    """Five-dimensional execution cost.

    Stored raw, never folded into a scalar at rest. ``retries`` is a cost:
    "model wrong -> repair -> rerun" must be more expensive than getting it
    right the first time, even when solver runtime is similar.
    ``llm_tokens`` is unknown at execution time (the harness owns the LLM)
    and is backfilled later via ``orx record --override llm_tokens=...``.
    """

    llm_tokens: float = 0.0
    tool_calls: float = 0.0
    solver_runtime_s: float = 0.0
    retries: float = 0.0
    latency_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in COST_DIMENSIONS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostVector":
        if not isinstance(data, dict):
            raise ValueError("CostVector must be a JSON object")
        return cls(**{d: float(data.get(d, 0.0)) for d in COST_DIMENSIONS})

    def scalarize(self, weights: Optional[Dict[str, float]] = None,
                  norms: Optional[Dict[str, float]] = None) -> float:
        """Weighted sum of (optionally normalized) dimensions.

        Normalization divisors default to 1.0 (raw units). Callers that want
        scale-free comparison should pass per-dimension norms (e.g. running
        maxima). Weights are configurable, never hard-coded at call sites.
        """
        w = dict(DEFAULT_COST_WEIGHTS if weights is None else weights)
        n = norms or {}
        total = 0.0
        for d in COST_DIMENSIONS:
            divisor = float(n.get(d, 1.0)) or 1.0
            total += float(w.get(d, 0.0)) * (float(getattr(self, d)) / divisor)
        return total

    def plus(self, other: "CostVector") -> "CostVector":
        return CostVector(**{d: getattr(self, d) + getattr(other, d) for d in COST_DIMENSIONS})


# ---------------------------------------------------------------------------
# Problem profile
# ---------------------------------------------------------------------------

#: Coupling dimensions that participate in structural grouping. All are floats
#: in [0, 1]; ``None`` means "unknown" and bins to its own bucket.
COUPLING_FEATURES: Tuple[str, ...] = (
    "semantic_coupling",
    "resource_coupling",
    "temporal_coupling",
    "route_complexity",
)

SCALE_FEATURES: Tuple[str, ...] = (
    "n_vars",
    "n_constraints",
    "n_int_vars",
    "density",
)

#: Feature-bin edges for the scope ladder. Fine bins drive L1/L2; coarse bins
#: drive L3. Configurable by callers that construct profiles directly, but the
#: defaults below are the canonical grouping contract.
FINE_BIN_EDGES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
COARSE_BIN_EDGES: Tuple[float, ...] = (0.0, 0.5, 1.0)

PROFILE_SOURCES = ("harness_supplied", "derived")


@dataclass
class ProblemProfile:
    """Structural portrait of one task.

    Coupling dimensions may be supplied directly by the harness (which may
    already understand the problem upstream) or derived deterministically by
    the profiler from a structured spec or model representation.
    """

    problem_id: str
    family: str
    scale_features: Dict[str, float] = field(default_factory=dict)
    semantic_coupling: Optional[float] = None
    resource_coupling: Optional[float] = None
    temporal_coupling: Optional[float] = None
    route_complexity: Optional[float] = None
    risk_features: Dict[str, Any] = field(default_factory=dict)
    source: str = "derived"
    annotations: Dict[str, Any] = field(default_factory=dict)

    def coupling(self) -> Dict[str, Optional[float]]:
        return {f: getattr(self, f) for f in COUPLING_FEATURES}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "family": self.family,
            "scale_features": {k: float(v) for k, v in self.scale_features.items()},
            **{f: getattr(self, f) for f in COUPLING_FEATURES},
            "risk_features": self.risk_features,
            "source": self.source,
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemProfile":
        if not isinstance(data, dict):
            raise ValueError("ProblemProfile must be a JSON object")
        for key in ("problem_id", "family"):
            if not data.get(key):
                raise ValueError(f"ProblemProfile.{key} is required")
        source = data.get("source", "derived")
        if source not in PROFILE_SOURCES:
            raise ValueError(f"ProblemProfile.source must be one of {PROFILE_SOURCES}")

        def _opt_float(name: str) -> Optional[float]:
            v = data.get(name)
            return None if v is None else float(v)

        return cls(
            problem_id=str(data["problem_id"]),
            family=str(data["family"]),
            scale_features={k: float(v) for k, v in (data.get("scale_features") or {}).items()},
            semantic_coupling=_opt_float("semantic_coupling"),
            resource_coupling=_opt_float("resource_coupling"),
            temporal_coupling=_opt_float("temporal_coupling"),
            route_complexity=_opt_float("route_complexity"),
            risk_features=dict(data.get("risk_features") or {}),
            source=source,
            annotations=dict(data.get("annotations") or {}),
        )


# ---------------------------------------------------------------------------
# Structural grouping (the scope ladder)
# ---------------------------------------------------------------------------

#: Scope levels. L1 = family + fine bins (narrowest). L2 = fine bins, any
#: family. L3 = coarse bins, any family (widest). Widening is falsifiable:
#: a wide entry makes riskier predictions; a cross-family miss tightens the
#: pattern back down the ladder.
SCOPE_LEVELS = ("L1", "L2", "L3")


def _bin_label(value: Optional[float], edges: Tuple[float, ...]) -> str:
    if value is None:
        return "unknown"
    v = min(max(float(value), edges[0]), edges[-1])
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v <= hi and (v < hi or hi == edges[-1]):
            return f"[{lo:.2f},{hi:.2f}]"
    return f"[{edges[-2]:.2f},{edges[-1]:.2f}]"


def group_key(profile: ProblemProfile, level: str = "L1") -> str:
    """Similarity key for conditional statistics at a given scope level."""
    if level not in SCOPE_LEVELS:
        raise ValueError(f"level must be one of {SCOPE_LEVELS}")
    edges = FINE_BIN_EDGES if level in ("L1", "L2") else COARSE_BIN_EDGES
    parts: List[str] = []
    if level == "L1":
        parts.append(f"family={profile.family}")
    else:
        parts.append("family=*")
    short = {"semantic_coupling": "sc", "resource_coupling": "rc",
             "temporal_coupling": "tc", "route_complexity": "rx"}
    for f in COUPLING_FEATURES:
        parts.append(f"{short[f]}{_bin_label(getattr(profile, f), edges)}")
    return "|".join(parts)


def pattern_for(profile: ProblemProfile, level: str = "L1") -> Dict[str, Any]:
    """Feature predicates describing the structural group of ``profile``.

    Predicate format: ``{"family": "routing"?, "<feature>": [lo, hi]}`` with
    inclusive bounds. L1 includes family and fine bins; L2 drops family;
    L3 uses coarse bins.
    """
    if level not in SCOPE_LEVELS:
        raise ValueError(f"level must be one of {SCOPE_LEVELS}")
    edges = FINE_BIN_EDGES if level in ("L1", "L2") else COARSE_BIN_EDGES
    pred: Dict[str, Any] = {}
    if level == "L1":
        pred["family"] = profile.family
    for f in COUPLING_FEATURES:
        v = getattr(profile, f)
        if v is None:
            continue
        label = _bin_label(v, edges)
        lo, hi = (float(x) for x in label.strip("[]").split(","))
        pred[f] = [lo, hi]
    return pred


def pattern_hash(predicates: Dict[str, Any], strategy_id: str = "") -> str:
    """Stable hash of a predicate set (used for cold-archive dedup)."""
    blob = json.dumps({"strategy": strategy_id, "predicates": predicates},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def profile_matches(profile: ProblemProfile, predicates: Dict[str, Any]) -> bool:
    """True when the profile satisfies every predicate."""
    family_pred = predicates.get("family")
    if family_pred is not None and profile.family != family_pred:
        return False
    for f in COUPLING_FEATURES:
        if f not in predicates:
            continue
        lo, hi = (float(x) for x in predicates[f])
        v = getattr(profile, f)
        if v is None or not (lo <= float(v) <= hi):
            return False
    return True


def predicates_cover(outer: Dict[str, Any], inner: Dict[str, Any]) -> bool:
    """True when every profile matching ``inner`` also matches ``outer``."""
    if "family" in outer and outer.get("family") != inner.get("family"):
        return False
    for f in COUPLING_FEATURES:
        if f not in outer:
            continue
        if f not in inner:
            return False
        o_lo, o_hi = (float(x) for x in outer[f])
        i_lo, i_hi = (float(x) for x in inner[f])
        if not (o_lo <= i_lo and i_hi <= o_hi):
            return False
    return True


def scope_of(predicates: Dict[str, Any]) -> str:
    """Infer the ladder level of a predicate set."""
    edges = COARSE_BIN_EDGES
    is_coarse = any(
        f in predicates and any(float(x) in edges for x in predicates[f])
        and tuple(float(x) for x in predicates[f]) in zip(edges, edges[1:])
        for f in COUPLING_FEATURES
    )
    if "family" in predicates:
        return "L1"
    return "L3" if is_coarse else "L2"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A named solving strategy.

    The catalog is a cold-start *vocabulary*: it carries structural knowledge
    (applicability conditions, modeling actions, fallback chain, solver family)
    but NO prior quality/cost/risk scores. Without accumulated experience the
    selector honestly reports "no evidence" rather than fabricating priors.
    """

    strategy_id: str
    name: str
    description: str = ""
    applicability: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    fallback: Optional[str] = None
    solver_family: Optional[str] = None
    #: Strategy type (free-form: modeling/decomposition/solver_selection/
    #: execution/recovery). Vocabulary-level tag; induction may inherit it
    #: into ``StrategicEntry.strategy_type``.
    strategy_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability,
            "actions": list(self.actions),
            "fallback": self.fallback,
            "solver_family": self.solver_family,
            "strategy_type": self.strategy_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        if not isinstance(data, dict):
            raise ValueError("Strategy must be a JSON object")
        for key in ("strategy_id", "name"):
            if not data.get(key):
                raise ValueError(f"Strategy.{key} is required")
        return cls(
            strategy_id=str(data["strategy_id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            applicability=dict(data.get("applicability") or {}),
            actions=[str(a) for a in (data.get("actions") or [])],
            fallback=data.get("fallback"),
            solver_family=data.get("solver_family"),
            strategy_type=data.get("strategy_type"),
        )


# ---------------------------------------------------------------------------
# Execution record (Execution Evidence storage unit)
# ---------------------------------------------------------------------------

VERIFICATION_LEVELS = ("basic", "strong")
EXECUTION_SOURCES = ("executed", "compacted")


@dataclass
class TrajectoryStep:
    action: str
    outcome: str
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "outcome": self.outcome,
                "duration_s": float(self.duration_s)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(action=str(data.get("action", "")),
                   outcome=str(data.get("outcome", "")),
                   duration_s=float(data.get("duration_s", 0.0)))


@dataclass
class FailureRecord:
    attempt: int
    error: str
    recovery_action: Optional[str] = None
    #: "environment" (solver/stack cannot run here: sandbox policy, missing
    #: module) vs "model" (the harness's own code failed). Computed once at
    #: record time and persisted — a first-class fact, not a re-derived view.
    error_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"attempt": int(self.attempt), "error": self.error,
                "recovery_action": self.recovery_action,
                "error_class": self.error_class}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureRecord":
        return cls(attempt=int(data.get("attempt", 0)),
                   error=str(data.get("error", "")),
                   recovery_action=data.get("recovery_action"),
                   error_class=data.get("error_class"))


@dataclass
class ExecutionRecord:
    """One execution episode — the Execution Evidence unit. Append-only fact:
    records what actually happened, never what will happen. Facts are
    permanently neutral — disposal applies only to the derived layer
    (Strategic Knowledge).

    Field discipline (do not blur):
      - ``strategy_id`` is the strategy ACTUALLY used in this episode. It
        makes no generalization claim.
      - ``quality`` / ``cost`` are ACTUAL observations (alias properties
        ``actual_quality`` / ``actual_cost`` make this explicit).
      - ``solver`` and ``execution_features`` hold implementation artifacts
        (model/solver outputs). Artifacts are evidence, not a separate
        memory class.
      - ``cir_snapshot`` preserves the coupling-aware representation that
        was actually solved (optional; pre-CIR records omit it).
      - ``retention_reason`` marks representative evidence (free-form
        string, e.g. "failure_recovery", "boundary_outcome", "high_cost",
        "recovery_chain", or a harness-supplied reason). Marked rows are
        never compacted by GC. Retention serves future induction, cost
        learning, and explanation — it is NOT referential integrity:
        knowledge admission never depends on which evidence rows survive.
    """

    execution_id: str
    task_id: str
    strategy_id: str
    profile_snapshot: ProblemProfile
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    cost: CostVector = field(default_factory=CostVector)
    failures: List[FailureRecord] = field(default_factory=list)
    solver: Dict[str, Any] = field(default_factory=dict)
    #: Post-strategy diagnostics — separate from the frozen pre-strategy
    #: signature (profile_snapshot). Carries solver diagnostics, model
    #: diagnostics, and other execution-time observations. Never modifies
    #: task identity; available to offline induction but not online retrieval.
    execution_features: Dict[str, Any] = field(default_factory=dict)
    verification_level: str = "basic"
    created_at: float = field(default_factory=time.time)
    source: str = "executed"
    #: Coupling-aware representation snapshot (the CIR that was actually
    #: solved). Optional: records from tasks without a supplied CIR omit it.
    #: Preserved so future induction can re-bin evidence by structural
    #: context beyond the four scalar coupling features.
    cir_snapshot: Optional[Dict[str, Any]] = None
    #: Representative-evidence retention marker (free-form string). Non-empty
    #: values exempt this episode from GC compaction. Set automatically by
    #: ``api.record`` for strategically informative episodes and overridable
    #: by the harness.
    retention_reason: Optional[str] = None

    @staticmethod
    def new_id() -> str:
        return f"ex_{uuid.uuid4().hex[:12]}"

    @property
    def group_l1(self) -> str:
        return group_key(self.profile_snapshot, "L1")

    @property
    def actual_quality(self) -> Dict[str, Any]:
        """Alias for ``quality``: the quality ACTUALLY observed in this
        episode. Serialization key stays ``quality``."""
        return self.quality

    @property
    def actual_cost(self) -> CostVector:
        """Alias for ``cost``: the cost ACTUALLY paid in this episode.
        Serialization key stays ``cost``."""
        return self.cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "profile_snapshot": self.profile_snapshot.to_dict(),
            "trajectory": [t.to_dict() for t in self.trajectory],
            "quality": self.quality,
            "cost": self.cost.to_dict(),
            "failures": [f.to_dict() for f in self.failures],
            "solver": self.solver,
            "execution_features": dict(self.execution_features),
            "verification_level": self.verification_level,
            "created_at": self.created_at,
            "source": self.source,
            "cir_snapshot": (dict(self.cir_snapshot)
                             if self.cir_snapshot is not None else None),
            "retention_reason": self.retention_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        if not isinstance(data, dict):
            raise ValueError("ExecutionRecord must be a JSON object")
        for key in ("execution_id", "task_id", "strategy_id", "profile_snapshot"):
            if key not in data:
                raise ValueError(f"ExecutionRecord.{key} is required")
        verification = data.get("verification_level", "basic")
        if verification not in VERIFICATION_LEVELS:
            raise ValueError(f"verification_level must be one of {VERIFICATION_LEVELS}")
        return cls(
            execution_id=str(data["execution_id"]),
            task_id=str(data["task_id"]),
            strategy_id=str(data["strategy_id"]),
            profile_snapshot=ProblemProfile.from_dict(data["profile_snapshot"]),
            trajectory=[TrajectoryStep.from_dict(t) for t in (data.get("trajectory") or [])],
            quality=dict(data.get("quality") or {}),
            cost=CostVector.from_dict(data.get("cost") or {}),
            failures=[FailureRecord.from_dict(f) for f in (data.get("failures") or [])],
            solver=dict(data.get("solver") or {}),
            execution_features=dict(data.get("execution_features") or {}),
            verification_level=verification,
            created_at=float(data.get("created_at", time.time())),
            source=str(data.get("source", "executed")),
            cir_snapshot=(dict(data["cir_snapshot"])
                          if data.get("cir_snapshot") else None),
            retention_reason=data.get("retention_reason"),
        )


# ---------------------------------------------------------------------------
# Strategic entry (Strategic Knowledge storage unit)
# ---------------------------------------------------------------------------

#: Hot-store lifecycle states. ``retired`` is terminal: the entry moves to the
#: cold archive and leaves the hot store entirely.
ENTRY_STATUSES = ("candidate", "validated", "suspect", "dormant")

#: Honest interval floor by supporting sample size: with n=2 you may not claim
#: [0.95, 1.00]-style narrow intervals (pretending to be more certain than the
#: data). Maps max(n) -> minimum allowed interval width.
MIN_INTERVAL_WIDTH = {2: 0.50, 3: 0.35, 4: 0.25, 5: 0.15}


def min_interval_width(n: int) -> float:
    """Minimum allowed prediction-interval width for n supporting samples."""
    for threshold in sorted(MIN_INTERVAL_WIDTH):
        if n <= threshold:
            return MIN_INTERVAL_WIDTH[threshold]
    return 0.0


@dataclass
class ApplicabilityCondition:
    """Natural-language applicability text, optionally LLM-phrased by the
    outer harness. While ``verified`` is false it never enters scoring.
    Citations are unfalsifiable-by-construction-checked: every
    ``supporting_execution_ids`` must reference real records and numeric
    claims must agree with them, or the condition is rejected."""

    text: str
    verified: bool = False
    supporting_execution_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "verified": self.verified,
                "supporting_execution_ids": list(self.supporting_execution_ids)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicabilityCondition":
        if not isinstance(data, dict) or not data.get("text"):
            raise ValueError("ApplicabilityCondition.text is required")
        return cls(text=str(data["text"]), verified=bool(data.get("verified", False)),
                   supporting_execution_ids=[str(i) for i in
                                             (data.get("supporting_execution_ids") or [])])


@dataclass
class PredictionTrack:
    """Forward-validation bookkeeping: how this entry's predictions fared
    against *future* executions. Promotion: n_predictions >= 5 and
    hit_rate >= 0.7 (candidate -> validated). Demotion: 3 consecutive misses
    (-> suspect, score x0.5).

    Cost predictions carry their own, SEPARATE track: a cost miss means the
    entry's cost estimate is unreliable — it warns, it never demotes. Quality
    and cost errors have different reversibility and different remedies
    (retire vs. re-weigh), so they never share a verdict."""

    n_predictions: int = 0
    n_hits: int = 0
    consecutive_misses: int = 0
    calibration_error: float = 0.0
    # Cost-side track (parallel, warning-only).
    n_cost_predictions: int = 0
    n_cost_hits: int = 0
    cost_calibration: Dict[str, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_predictions if self.n_predictions else 0.0

    @property
    def cost_hit_rate(self) -> float:
        return self.n_cost_hits / self.n_cost_predictions \
            if self.n_cost_predictions else 0.0

    def record(self, hit: bool, calibration_err: float = 0.0) -> None:
        self.n_predictions += 1
        if hit:
            self.n_hits += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        n = self.n_predictions
        self.calibration_error = ((self.calibration_error * (n - 1)) + calibration_err) / n

    def record_cost(self, hit: bool, per_dimension_log_error: Dict[str, float]) -> None:
        """Record a cost-prediction check. Warning-only: never touches
        consecutive_misses or the quality-side counters."""
        self.n_cost_predictions += 1
        if hit:
            self.n_cost_hits += 1
        n = self.n_cost_predictions
        for dim, err in per_dimension_log_error.items():
            prev = self.cost_calibration.get(dim, 0.0)
            self.cost_calibration[dim] = (prev * (n - 1) + err) / n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "n_hits": self.n_hits,
            "hit_rate": round(self.hit_rate, 4),
            "consecutive_misses": self.consecutive_misses,
            "calibration_error": round(self.calibration_error, 4),
            "n_cost_predictions": self.n_cost_predictions,
            "n_cost_hits": self.n_cost_hits,
            "cost_hit_rate": round(self.cost_hit_rate, 4),
            "cost_calibration": {k: round(v, 4) for k, v in
                                 self.cost_calibration.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTrack":
        data = data or {}
        return cls(
            n_predictions=int(data.get("n_predictions", 0)),
            n_hits=int(data.get("n_hits", 0)),
            consecutive_misses=int(data.get("consecutive_misses", 0)),
            calibration_error=float(data.get("calibration_error", 0.0)),
            n_cost_predictions=int(data.get("n_cost_predictions", 0)),
            n_cost_hits=int(data.get("n_cost_hits", 0)),
            cost_calibration={k: float(v) for k, v in
                              (data.get("cost_calibration") or {}).items()},
        )


@dataclass
class StrategicEntry:
    """A generalized, calibrated commitment: 'for problems matching this
    pattern, this strategy will perform within these intervals.'

    The Strategic Knowledge unit: derived (not primary) knowledge — a
    revisable belief validated at INDUCTION time against supporting evidence.
    After admission the entry keeps only lightweight origin metadata
    (``provenance`` = optional representative execution ids, ``support_n``);
    its continued validity does NOT depend on the survival of those evidence
    rows — compacting or deleting old evidence never invalidates an admitted
    entry, and exact reconstruction of past entries is never required
    (``induce --rebuild`` re-induces from whatever evidence is currently
    retained). Not a restatement of statistics — a claim about the future,
    with an interval, calibration tracking, and cross-group feature
    predicates.

    Extension points for future induction (all optional, backward
    compatible): ``strategy_type`` (what kind of strategy — modeling,
    decomposition, solver selection, execution, recovery — one knowledge
    layer holds all types, never one bank per type), ``principle`` (a
    reusable strategic principle), ``actions`` (recommended
    actions/adaptations). ``strategy_id`` stays required in v1; future
    induction may relax it for cross-strategy principles.
    """

    entry_id: str
    strategy_id: str
    pattern: Dict[str, Any]  # {"scope_level": "L1"|"L2"|"L3", "predicates": {...}}
    expected_quality_hat: float = 0.5
    quality_interval: Tuple[float, float] = (0.0, 1.0)
    expected_cost_hat: CostVector = field(default_factory=CostVector)
    #: Per-dimension multiplicative cost interval: actual cost is expected
    #: within [lo * hat, hi * hat] per dimension. v1 uses a fixed 2x band
    #: ([0.5, 2.0]); the interval is checked (not just the point estimate)
    #: by the record chain's cost validation.
    cost_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Expected failure risk (alias: ``expected_failure_risk``).
    failure_prob: float = 0.5
    applicability: List[ApplicabilityCondition] = field(default_factory=list)
    risk_conditions: List[str] = field(default_factory=list)
    fallback_strategy_id: Optional[str] = None
    status: str = "candidate"
    prediction_track: PredictionTrack = field(default_factory=PredictionTrack)
    provenance: List[str] = field(default_factory=list)  # execution_ids
    last_consulted_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    support_n: int = 0
    #: Strategy type — free-form string (e.g. "modeling", "decomposition",
    #: "solver_selection", "execution", "recovery"). No rigid taxonomy;
    #: multiple types coexist inside one Strategic Knowledge Bank.
    strategy_type: Optional[str] = None
    #: Reusable strategic principle (natural language). Future induction
    #: output slot; v1 leaves it None.
    principle: Optional[str] = None
    #: Recommended actions/adaptations, inherited from the catalog
    #: vocabulary at induce time (harness may override).
    actions: List[str] = field(default_factory=list)

    @staticmethod
    def new_id() -> str:
        return f"se_{uuid.uuid4().hex[:12]}"

    @property
    def scope_level(self) -> str:
        return str(self.pattern.get("scope_level", "L1"))

    @property
    def expected_quality(self) -> float:
        """Alias for ``expected_quality_hat`` (serialization key unchanged)."""
        return self.expected_quality_hat

    @property
    def expected_cost(self) -> CostVector:
        """Alias for ``expected_cost_hat`` (serialization key unchanged)."""
        return self.expected_cost_hat

    @property
    def expected_failure_risk(self) -> float:
        """Alias for ``failure_prob``: the expected failure probability."""
        return self.failure_prob

    @property
    def predicates(self) -> Dict[str, Any]:
        return dict(self.pattern.get("predicates") or {})

    def matches(self, profile: ProblemProfile) -> bool:
        return profile_matches(profile, self.predicates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "strategy_id": self.strategy_id,
            "pattern": {"scope_level": self.scope_level,
                        "predicates": self.predicates},
            "expected": {
                "quality_hat": self.expected_quality_hat,
                "quality_interval": [self.quality_interval[0], self.quality_interval[1]],
                "cost_hat": self.expected_cost_hat.to_dict(),
                "cost_interval": {d: [lo, hi] for d, (lo, hi)
                                  in self.cost_interval.items()},
                "failure_prob": self.failure_prob,
            },
            "applicability": [a.to_dict() for a in self.applicability],
            "risk_conditions": list(self.risk_conditions),
            "fallback_strategy_id": self.fallback_strategy_id,
            "status": self.status,
            "prediction_track": self.prediction_track.to_dict(),
            "provenance": list(self.provenance),
            "support_n": self.support_n,
            "strategy_type": self.strategy_type,
            "principle": self.principle,
            "actions": list(self.actions),
            "last_consulted_at": self.last_consulted_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategicEntry":
        if not isinstance(data, dict):
            raise ValueError("StrategicEntry must be a JSON object")
        for key in ("entry_id", "strategy_id", "pattern"):
            if key not in data:
                raise ValueError(f"StrategicEntry.{key} is required")
        pattern = dict(data["pattern"])
        if pattern.get("scope_level") not in SCOPE_LEVELS:
            raise ValueError(f"pattern.scope_level must be one of {SCOPE_LEVELS}")
        status = data.get("status", "candidate")
        if status not in ENTRY_STATUSES:
            raise ValueError(f"status must be one of {ENTRY_STATUSES}")
        expected = data.get("expected") or {}
        interval = expected.get("quality_interval") or [0.0, 1.0]
        cost_interval = {d: (float(lo), float(hi))
                         for d, (lo, hi) in
                         (expected.get("cost_interval") or {}).items()}
        return cls(
            entry_id=str(data["entry_id"]),
            strategy_id=str(data["strategy_id"]),
            pattern={"scope_level": pattern["scope_level"],
                     "predicates": dict(pattern.get("predicates") or {})},
            expected_quality_hat=float(expected.get("quality_hat", 0.5)),
            quality_interval=(float(interval[0]), float(interval[1])),
            expected_cost_hat=CostVector.from_dict(expected.get("cost_hat") or {}),
            cost_interval=cost_interval,
            failure_prob=float(expected.get("failure_prob", 0.5)),
            applicability=[ApplicabilityCondition.from_dict(a)
                           for a in (data.get("applicability") or [])],
            risk_conditions=[str(r) for r in (data.get("risk_conditions") or [])],
            fallback_strategy_id=data.get("fallback_strategy_id"),
            status=status,
            prediction_track=PredictionTrack.from_dict(data.get("prediction_track")),
            provenance=[str(p) for p in (data.get("provenance") or [])],
            support_n=int(data.get("support_n", 0)),
            strategy_type=data.get("strategy_type"),
            principle=data.get("principle"),
            actions=[str(a) for a in (data.get("actions") or [])],
            last_consulted_at=data.get("last_consulted_at"),
            created_at=float(data.get("created_at", time.time())),
        )


# ---------------------------------------------------------------------------
# Cold archive card
# ---------------------------------------------------------------------------


@dataclass
class ColdArchiveCard:
    """Compressed tombstone (~200 B) for a retired entry. Kept forever by
    default; consulted before induction so the same evidence cannot resurrect
    the same failed generalization (anti-resurrection)."""

    pattern_hash: str
    strategy_id: str
    predicates: Dict[str, Any]
    outcome: str
    reason: str
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    archived_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_hash": self.pattern_hash,
            "strategy_id": self.strategy_id,
            "predicates": self.predicates,
            "outcome": self.outcome,
            "reason": self.reason,
            "evidence_summary": self.evidence_summary,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColdArchiveCard":
        for key in ("pattern_hash", "strategy_id", "predicates"):
            if key not in data:
                raise ValueError(f"ColdArchiveCard.{key} is required")
        return cls(
            pattern_hash=str(data["pattern_hash"]),
            strategy_id=str(data["strategy_id"]),
            predicates=dict(data["predicates"]),
            outcome=str(data.get("outcome", "")),
            reason=str(data.get("reason", "")),
            evidence_summary=dict(data.get("evidence_summary") or {}),
            archived_at=float(data.get("archived_at", time.time())),
        )
