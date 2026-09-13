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
                    tracking. Mutation happens at INDUCTION time only: online
                    execution records new evidence (frozen checks on the
                    facts) and touches no entry; the next induction creates,
                    refreshes, and revises entries from that evidence, and a
                    claim's applicability is read off the evidence that
                    supports it. Admission never depends on the survival of
                    the original evidence rows. Never stores raw execution
                    detail.
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
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

    Stored raw, never folded into a scalar at rest. ``retries`` counts only
    *extra* attempts beyond the first (a first failure is not a retry).
    ``llm_tokens`` is unknown at execution time (the harness owns the LLM)
    and is backfilled later via ``orx record --override llm_tokens=...``.

    Unknown vs measured-zero is carried by ``measured`` (the set of
    dimensions actually measured). A placeholder zero for an unmeasured
    dimension is NEVER evidence of cheapness: consumers must consult
    :meth:`measured_dims` and exclude unmeasured dimensions from means,
    errors, and comparisons. ``measured=None`` marks legacy data (payloads
    written before this field existed); the value-level inference in
    :meth:`measured_dims` keeps non-zero legacy values usable (including
    already-backfilled non-zero ``llm_tokens``) while unconfirmable legacy
    zeros stay unknown.
    """

    llm_tokens: float = 0.0
    tool_calls: float = 0.0
    solver_runtime_s: float = 0.0
    retries: float = 0.0
    latency_s: float = 0.0
    #: Measured-dimension mask. ``None`` = legacy/unmarked data.
    measured: Optional[Set[str]] = None

    def measured_dims(self) -> Set[str]:
        """Dimensions confirmed measured.

        Legacy inference (``measured is None``): any non-zero value can only
        come from a measurement — including non-zero ``llm_tokens`` that
        older harnesses already backfilled — while zero values cannot be
        confirmed and stay unknown. This is value-level, not dimension-level:
        legacy non-zero tokens are NOT excluded.
        """
        if self.measured is not None:
            return {d for d in self.measured if d in COST_DIMENSIONS}
        return {d for d in COST_DIMENSIONS if getattr(self, d) != 0.0}

    def mark_measured(self, *dims: str) -> None:
        """Explicitly mark dimensions measured (e.g. after backfill)."""
        m = self.measured_dims()
        m.update(dims)
        self.measured = m

    def to_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in COST_DIMENSIONS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostVector":
        if not isinstance(data, dict):
            raise ValueError("CostVector must be a JSON object")
        vector = cls(**{d: float(data.get(d, 0.0)) for d in COST_DIMENSIONS})
        raw_measured = data.get("measured")
        if isinstance(raw_measured, (list, tuple, set)):
            vector.measured = {str(d) for d in raw_measured if d in COST_DIMENSIONS}
        return vector

    def scalarize(self, weights: Optional[Dict[str, float]] = None,
                  norms: Optional[Dict[str, float]] = None) -> float:
        """Weighted sum of (optionally normalized) dimensions.

        Normalization divisors default to 1.0 (raw units). Callers that want
        scale-free comparison should pass per-dimension norms (e.g. running
        maxima). Weights are configurable, never hard-coded at call sites.
        Unmeasured dimensions carry placeholder zeros here — callers doing
        selection-time comparison must pass weights restricted to the
        comparable (commonly measured) dimensions, so missing data never
        scores as cheap.
        """
        w = dict(DEFAULT_COST_WEIGHTS if weights is None else weights)
        n = norms or {}
        total = 0.0
        for d in COST_DIMENSIONS:
            divisor = float(n.get(d, 1.0)) or 1.0
            total += float(w.get(d, 0.0)) * (float(getattr(self, d)) / divisor)
        return total


# ---------------------------------------------------------------------------
# Prediction snapshot (pre-execution expectation, for feedback alignment)
# ---------------------------------------------------------------------------

#: Provenance labels for a prediction snapshot. ``entry`` = a matching
#: StrategicEntry's expectation; ``stats`` = conditional statistics over the
#: Execution Evidence Bank (a recount, weaker evidence); ``unknown`` = no
#: usable evidence (no data, or data out of scope/scale).
SNAPSHOT_SOURCES = ("entry", "stats", "unknown")


@dataclass
class PredictionSnapshot:
    """The prediction ACTUALLY used before execution, frozen at that time.

    Cost-side feedback alignment: an execution is compared against the
    expectation that informed its choice — never against whatever estimate
    happens to be current after the fact. Fields:

    - ``strategy_id``: the strategy this snapshot predicted for.
    - ``expected_cost``: expected CostVector (with per-dimension measured
      mask), or ``None`` when there is no usable evidence — unknown is never
      represented as a placeholder zero.
    - ``source``: ``entry`` / ``stats`` / ``unknown``.
    - ``measurement_scope``: what the expectation covers (``attempt``).
      Predictions must reuse the same scope as the historical records they
      were estimated from and as the execution they will be checked against.
    - ``support_n``: total supporting executions; ``support_per_dim``: the
      per-dimension effective sample counts — total support never masquerades
      as per-dimension support.
    - ``evidence_refs``: lightweight provenance (entry id / execution ids),
      no strict reference-integrity dependency.
    - ``note``: lightweight explanation (e.g. scale mismatch, insufficient
      evidence). Post-modeling information never enters this snapshot.
    """

    strategy_id: str
    expected_cost: Optional[CostVector] = None
    source: str = "unknown"
    measurement_scope: str = "attempt"
    support_n: int = 0
    support_per_dim: Dict[str, int] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "expected_cost": (self.expected_cost.to_dict()
                              if self.expected_cost is not None else None),
            "cost_measured": (sorted(self.expected_cost.measured)
                              if self.expected_cost is not None
                              and self.expected_cost.measured is not None
                              else None),
            "source": self.source,
            "measurement_scope": self.measurement_scope,
            "support_n": int(self.support_n),
            "support_per_dim": {d: int(n) for d, n in
                                self.support_per_dim.items()},
            "evidence_refs": list(self.evidence_refs),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionSnapshot":
        if not isinstance(data, dict) or not data.get("strategy_id"):
            raise ValueError("PredictionSnapshot.strategy_id is required")
        source = str(data.get("source", "unknown"))
        if source not in SNAPSHOT_SOURCES:
            raise ValueError(f"PredictionSnapshot.source must be one of {SNAPSHOT_SOURCES}")
        raw_cost = data.get("expected_cost")
        expected_cost = (CostVector.from_dict(raw_cost)
                         if isinstance(raw_cost, dict) else None)
        raw_measured = data.get("cost_measured")
        if expected_cost is not None and isinstance(raw_measured, list):
            expected_cost.measured = {str(d) for d in raw_measured
                                      if d in COST_DIMENSIONS}
        return cls(
            strategy_id=str(data["strategy_id"]),
            expected_cost=expected_cost,
            source=source,
            measurement_scope=str(data.get("measurement_scope", "attempt")),
            support_n=int(data.get("support_n", 0)),
            support_per_dim={str(d): int(n) for d, n in
                             (data.get("support_per_dim") or {}).items()},
            evidence_refs=[str(r) for r in (data.get("evidence_refs") or [])],
            note=str(data.get("note", "")),
        )


def compute_cost_feedback(snapshot: Optional["PredictionSnapshot"],
                          actual_strategy_id: str,
                          actual_scope: str,
                          actual_cost: "CostVector") -> Optional[Dict[str, Any]]:
    """Compute cost feedback from the frozen snapshot and the actual cost.

    Pure function (no storage access) shared by the record chain and by
    backfill-time re-computation, so a later ``update_cost`` keeps any
    persisted feedback consistent with the amended actual value.

    Feedback is produced only when:
    - a snapshot with a concrete expected cost exists;
    - the snapshot's strategy equals the actual strategy (A never audits B);
    - the snapshot's scope equals the record's scope (an attempt never
      audits a task-scope prediction);
    - a dimension is measured on BOTH sides (prediction side included — an
      unmeasured predicted zero never fabricates an error against a measured
      actual).
    Otherwise returns None — no pseudo-errors.
    """
    if snapshot is None or snapshot.expected_cost is None:
        return None
    if snapshot.strategy_id != actual_strategy_id:
        return None
    if snapshot.measurement_scope != actual_scope:
        return None
    per_dim: Dict[str, Dict[str, float]] = {}
    for dim in snapshot.expected_cost.measured_dims() & actual_cost.measured_dims():
        predicted = getattr(snapshot.expected_cost, dim)
        actual = getattr(actual_cost, dim)
        if predicted <= 0 and actual <= 0:
            continue  # both placeholders: nothing to learn
        per_dim[dim] = {
            "predicted": round(predicted, 6),
            "actual": round(actual, 6),
            "log_error": round(abs(math.log(max(actual, 1e-9)
                                            / max(predicted, 1e-9))), 4),
        }
    if not per_dim:
        return None
    return {
        "source": snapshot.source,
        "measurement_scope": snapshot.measurement_scope,
        "support_n": snapshot.support_n,
        "support_per_dim": dict(snapshot.support_per_dim),
        "per_dimension": per_dim,
    }


# ---------------------------------------------------------------------------
# Problem profile
# ---------------------------------------------------------------------------

#: Coupling dimensions of a ProblemProfile, all floats in [0, 1]; ``None``
#: means "unknown" and bins to its own bucket.
COUPLING_FEATURES: Tuple[str, ...] = (
    "semantic_coupling",
    "resource_coupling",
    "temporal_coupling",
    "route_complexity",
)

#: The MEASURABLE subset — the only dimensions that may condition statistics.
#: ``resource_coupling`` / ``temporal_coupling`` / ``route_complexity`` are
#: derived from structure (CIR > model > spec), so every task's value is
#: reproducible from its own artifacts. ``semantic_coupling`` is never
#: derived — it is whatever number the harness typed — so it stays a profile
#: attribute (understanding, self-judgment) and is deliberately kept OUT of
#: grouping keys and predicates: an unverifiable input must not split the
#: evidence, and an entry carrying it could never match a task where it was
#: not supplied.
GROUPING_FEATURES: Tuple[str, ...] = (
    "resource_coupling",
    "temporal_coupling",
    "route_complexity",
)

#: Structural grouping edges — the SAME four intervals the framework used
#: before the scope ladder was retired. Restoring them restores "structurally
#: similar evidence is what gets aggregated", WITHOUT restoring the ladder
#: lifecycle (no levels, no scope_of, no widen/tighten, no automatic
#: re-scoping): widening a claim's structure is the harness's judgment, not
#: a framework command.
#:
#: The trade-off is accepted deliberately: when a family's executions scatter
#: across cells, a cell may hold too few samples to form a claim — the
#: statistics remain visible, no knowledge is created, and no automatic
#: cell-merging is attempted. (Merging is what produced claims that averaged
#: opposite behaviours into a single "0.55".)
BIN_EDGES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Bucket label for a grouping dimension with no measured value. Unknown is
#: its OWN cell; it is never pooled with measured evidence, and (unlike the
#: pre-ladder-retirement behaviour, which simply omitted the dimension and so
#: matched everything) it does not silently widen a claim's applicability.
UNKNOWN_BUCKET = "unknown"

SCALE_FEATURES: Tuple[str, ...] = (
    "n_vars",
    "n_constraints",
    "n_int_vars",
    "density",
)

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
# Structural grouping (evidence sets)
# ---------------------------------------------------------------------------

def bin_label(value: Optional[float],
              edges: Tuple[float, ...] = BIN_EDGES) -> str:
    """The interval label a grouping-dimension value falls into.

    Edges are inclusive on both sides (``[0.25, 0.50]`` contains 0.25), with
    the top interval closed so a value of exactly 1.0 lands in the last cell.
    ``None`` (unmeasured) labels as :data:`UNKNOWN_BUCKET` — its own cell.
    Exact edge values are binary-exact (0.25, 0.5, 0.75, 1.0), so this
    quantization introduces no rounding drift.
    """
    if value is None:
        return f"[{UNKNOWN_BUCKET}]"
    v = min(max(float(value), edges[0]), edges[-1])
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v <= hi and (v < hi or hi == edges[-1]):
            return f"[{lo:.2f},{hi:.2f}]"
    return f"[{edges[-2]:.2f},{edges[-1]:.2f}]"


def group_key(profile: ProblemProfile) -> str:
    """Similarity key of one evidence set: family + structural cells.

    One (family, structural cell, strategy) triple is one evidence set — the
    observations that may be aggregated together. Structure is the legacy
    four-interval quantization of the measurable coupling dims, so evidence
    from structurally incomparable regions of a family is never pooled (the
    defect that averaged a Q=1.0 region and a Q=0.1 region into one claim).
    """
    parts: List[str] = [f"family={profile.family}"]
    short = {"resource_coupling": "rc", "temporal_coupling": "tc",
             "route_complexity": "rx"}
    for f in GROUPING_FEATURES:
        parts.append(f"{short[f]}{bin_label(getattr(profile, f))}")
    return "|".join(parts)


def bin_interval(label: str) -> Optional[Tuple[float, float]]:
    """Numeric bounds of a bin label, or None for the unknown bucket."""
    if label == f"[{UNKNOWN_BUCKET}]":
        return None
    lo, hi = label.strip("[]").split(",")
    return (float(lo), float(hi))


def evidence_predicates(records: Sequence["ExecutionRecord"],
                        family: Optional[str] = None) -> Dict[str, Any]:
    """The applicability of a claim, read off the evidence supporting it.

    One predicate per grouping dimension: the structural CELL the supporting
    executions occupy — ``[lo, hi]`` for a measured cell, or the string
    :data:`UNKNOWN_BUCKET` when none of them measured the dimension. Read off
    the cell (not a cross-sample min/max span) so that applicability says
    exactly what was demonstrated: the observed range could otherwise stretch
    across structurally incomparable regions merely because samples sat at
    both ends, and precision loss during rounding once made a claim fail to
    match its own supporting executions.

    ``family`` is the scope key of the evidence set. Callers that already
    know the family (induction) pass it explicitly; otherwise it is read off
    the first record.
    """
    if not records:
        return {}
    predicates: Dict[str, Any] = {
        "family": family or records[0].profile_snapshot.family}
    for f in GROUPING_FEATURES:
        labels = {bin_label(getattr(r.profile_snapshot, f)) for r in records}
        measured = sorted(l for l in labels if l != f"[{UNKNOWN_BUCKET}]")
        if len(measured) == 1:
            lo, hi = bin_interval(measured[0])
            predicates[f] = [lo, hi]
        elif len(measured) > 1:
            # A mixed measured set can only arise when the caller pooled
            # across cells; widening to the envelope keeps the predicate
            # honest about what was covered.
            bounds = [bin_interval(l) for l in measured]
            predicates[f] = [min(b[0] for b in bounds),
                             max(b[1] for b in bounds)]
        else:
            # Every supporting execution left this dimension unmeasured:
            # the claim says nothing about it — and must NOT match tasks
            # that DO have a measured value there.
            predicates[f] = f"[{UNKNOWN_BUCKET}]"
    return predicates


def pattern_hash(predicates: Dict[str, Any], strategy_id: str = "") -> str:
    """Stable hash of a predicate set (used for cold-archive dedup)."""
    blob = json.dumps({"strategy": strategy_id, "predicates": predicates},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def profile_matches(profile: ProblemProfile, predicates: Dict[str, Any]) -> bool:
    """True when the profile satisfies every predicate.

    An ``unknown`` predicate matches ONLY an unmeasured profile value on that
    dimension: "we never measured this" is not evidence that the structure is
    similar, so an unknown-cell claim must not be applied to a task whose
    structure is known (nor the reverse). A missing predicate still means
    "no constraint on this dimension" (harness-authored predicates)."""
    family_pred = predicates.get("family")
    if family_pred is not None and profile.family != family_pred:
        return False
    for f in GROUPING_FEATURES:
        if f not in predicates:
            continue
        pred = predicates[f]
        v = getattr(profile, f)
        if isinstance(pred, str):
            if v is not None:
                return False
            continue
        lo, hi = (float(x) for x in pred)
        if v is None or not (lo <= float(v) <= hi):
            return False
    return True


def predicates_cover(outer: Dict[str, Any], inner: Dict[str, Any]) -> bool:
    """True when every profile matching ``inner`` also matches ``outer``."""
    if "family" in outer and outer.get("family") != inner.get("family"):
        return False
    for f in GROUPING_FEATURES:
        if f not in outer:
            continue
        if f not in inner:
            return False
        o, i = outer[f], inner[f]
        o_unknown, i_unknown = isinstance(o, str), isinstance(i, str)
        if o_unknown or i_unknown:
            # An unknown predicate covers exactly one thing: the unknown
            # cell. It never covers measured evidence, and measured evidence
            # never covers it.
            if outer[f] != inner[f]:
                return False
            continue
        o_lo, o_hi = (float(x) for x in o)
        i_lo, i_hi = (float(x) for x in i)
        if not (o_lo <= i_lo and i_hi <= o_hi):
            return False
    return True


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
        string, harness-supplied, e.g. "contrast"). Explicit only — no
        automatic marking. Reserved for future compaction policies; lossy
        GC compaction is currently deferred.
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
    #: Representative-evidence retention marker (free-form string, explicit
    #: only — set via ``api.record(retain_reason=...)`` /
    #: ``orx record --retain-reason``). Reserved for future compaction
    #: policies; lossy GC compaction is currently deferred.
    retention_reason: Optional[str] = None
    #: What ``cost`` covers: "attempt" (this single execution attempt) or a
    #: wider harness-declared scope (e.g. "task"). One record = one attempt
    #: by default; conditional statistics, predictions, task summaries and
    #: feedback all consume attempt-scope records only, so a task-scope
    #: record can never be re-labelled as an attempt prediction. Legacy
    #: payloads without the key load as "attempt" (pre-scope records were
    #: all single attempts).
    measurement_scope: str = "attempt"
    #: Provenance of ``cost.solver_runtime_s``: "reported" (the solve
    #: script's result.json runtime_seconds) or "wall_proxy" (wall-clock of
    #: the whole script — an explicit proxy, never to be read as precise
    #: solver runtime). None for legacy payloads.
    solver_runtime_provenance: Optional[str] = None
    #: The pre-execution prediction actually used for this attempt (strategy,
    #: expected cost with per-dimension mask, source, scope), frozen at
    #: record time. Feedback is computed against this snapshot — never
    #: against a post-hoc re-read of current estimates. None when no
    #: prediction was supplied.
    prediction_snapshot: Optional[PredictionSnapshot] = None

    @staticmethod
    def new_id() -> str:
        return f"ex_{uuid.uuid4().hex[:12]}"

    @property
    def group_l1(self) -> str:
        """Evidence-set key of this record (its family).

        Name kept from the retired scope ladder because it is an on-disk
        column (``executions.group_l1``) with no migration mechanism — the
        value stored there is exactly :func:`group_key`.
        """
        return group_key(self.profile_snapshot)

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
            "cost_measured": (sorted(self.cost.measured)
                              if self.cost.measured is not None else None),
            "measurement_scope": self.measurement_scope,
            "solver_runtime_provenance": self.solver_runtime_provenance,
            "prediction_snapshot": (self.prediction_snapshot.to_dict()
                                    if self.prediction_snapshot is not None
                                    else None),
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
        cost = CostVector.from_dict(data.get("cost") or {})
        raw_measured = data.get("cost_measured")
        if isinstance(raw_measured, list):
            cost.measured = {str(d) for d in raw_measured if d in COST_DIMENSIONS}
        raw_snapshot = data.get("prediction_snapshot")
        snapshot = (PredictionSnapshot.from_dict(raw_snapshot)
                    if isinstance(raw_snapshot, dict) else None)
        return cls(
            execution_id=str(data["execution_id"]),
            task_id=str(data["task_id"]),
            strategy_id=str(data["strategy_id"]),
            profile_snapshot=ProblemProfile.from_dict(data["profile_snapshot"]),
            trajectory=[TrajectoryStep.from_dict(t) for t in (data.get("trajectory") or [])],
            quality=dict(data.get("quality") or {}),
            cost=cost,
            failures=[FailureRecord.from_dict(f) for f in (data.get("failures") or [])],
            solver=dict(data.get("solver") or {}),
            execution_features=dict(data.get("execution_features") or {}),
            verification_level=verification,
            created_at=float(data.get("created_at", time.time())),
            source=str(data.get("source", "executed")),
            cir_snapshot=(dict(data["cir_snapshot"])
                          if data.get("cir_snapshot") else None),
            retention_reason=data.get("retention_reason"),
            measurement_scope=str(data.get("measurement_scope", "attempt")),
            solver_runtime_provenance=data.get("solver_runtime_provenance"),
            prediction_snapshot=snapshot,
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
class PredictionTrack:
    """Forward-validation bookkeeping: how this entry's predictions fared
    against *future* executions. Promotion: n_predictions >= 5 and
    hit_rate >= 0.7 (candidate -> validated). Demotion: 3 consecutive misses
    (-> suspect, score x0.5). Both are applied by the offline induction pass
    from the frozen checks recorded on facts — never online.

    Quality only. Cost deviations are recorded PER EXECUTION as evidence
    (``execution_features.cost_feedback``, with per-dimension log errors);
    they never accumulate here, so there is no second lifecycle to keep in
    sync. Anyone wanting a cost-calibration view aggregates that evidence on
    demand, the way solver advisories do."""

    n_predictions: int = 0
    n_hits: int = 0
    consecutive_misses: int = 0
    calibration_error: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_predictions if self.n_predictions else 0.0

    def record(self, hit: bool, calibration_err: float = 0.0) -> None:
        self.n_predictions += 1
        if hit:
            self.n_hits += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        n = self.n_predictions
        self.calibration_error = ((self.calibration_error * (n - 1)) + calibration_err) / n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "n_hits": self.n_hits,
            "hit_rate": round(self.hit_rate, 4),
            "consecutive_misses": self.consecutive_misses,
            "calibration_error": round(self.calibration_error, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTrack":
        data = data or {}
        return cls(
            n_predictions=int(data.get("n_predictions", 0)),
            n_hits=int(data.get("n_hits", 0)),
            consecutive_misses=int(data.get("consecutive_misses", 0)),
            calibration_error=float(data.get("calibration_error", 0.0)),
        )


@dataclass
class StrategicEntry:
    """A generalized, calibrated commitment: 'for problems matching this
    pattern, this strategy will perform within these intervals.'

    The Strategic Knowledge unit: derived (not primary) knowledge — a
    revisable belief. Mutation happens at INDUCTION time only: online
    execution records evidence (frozen checks on the facts) and touches no
    entry; ``orx induce`` creates, refreshes, and revises entries, replaying
    that evidence. The entry keeps lightweight origin metadata
    (``provenance`` = optional representative execution ids, ``support_n``);
    its continued validity does NOT depend on the survival of those evidence
    rows, and exact reconstruction of past entries is never required
    (``induce --rebuild`` re-induces from whatever evidence is currently
    retained). Not a restatement of statistics — a claim about the future,
    with an interval, calibration tracking, and cross-group feature
    predicates.

    ``strategy_type`` / ``actions`` / ``fallback_strategy_id`` inherit the
    catalog vocabulary at induce time so the entry is self-contained (the
    catalog may evolve after the entry was written). ``applicability`` holds
    harness-written notes: free text, kept for the reader, never scored.
    """

    entry_id: str
    strategy_id: str
    pattern: Dict[str, Any]  # {"predicates": {"family": ..., "<dim>": [lo, hi]}}
    expected_quality_hat: float = 0.5
    quality_interval: Tuple[float, float] = (0.0, 1.0)
    expected_cost_hat: CostVector = field(default_factory=CostVector)
    #: Per-dimension multiplicative cost interval: actual cost is expected
    #: within [lo * hat, hi * hat] per dimension. v1 uses a fixed 2x band
    #: ([0.5, 2.0]); the interval is checked (not just the point estimate)
    #: by the record chain's cost validation. Only dimensions that were
    #: actually measured carry an interval — the key set doubles as the
    #: entry's measured-dimension mask for ``expected_cost_hat``.
    cost_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Per-dimension effective sample size behind ``expected_cost_hat``
    #: (records that actually measured the dimension). Total ``support_n``
    #: never masquerades as per-dimension cost support. Empty dict for
    #: pre-upgrade entries (per-dimension support unknown there).
    cost_support_n: Dict[str, int] = field(default_factory=dict)
    #: Expected failure risk (alias: ``expected_failure_risk``).
    failure_prob: float = 0.5
    #: Harness-written applicability notes (free text). Displayed by
    #: ``inspect``; never scored, never validated — the framework cannot
    #: check a sentence, and pretending to would be theatre.
    applicability: List[str] = field(default_factory=list)
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
    #: Recommended actions/adaptations, inherited from the catalog
    #: vocabulary at induce time (harness may override).
    actions: List[str] = field(default_factory=list)
    #: Admission verification (offline only). ``state`` gates publishing: a
    #: candidate with ``state != "verified"`` is not published as strategic
    #: knowledge (see ``Selector`` / ``ORHarness.predict_cost``). The rest of
    #: the block is the audit trail: what claim was checked, which executions
    #: were used, what check was applied, and why that conclusion followed.
    #: Never a bare boolean — a claim is only ever backed by a readable check.
    verification: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_published(self) -> bool:
        """True when this entry's claim passed admission verification."""
        return (self.verification or {}).get("state") == "verified"

    @property
    def verification_state(self) -> str:
        """Admission state, defaulting to ``unverified`` for legacy entries
        written before the block existed (absence of evidence is never
        treated as verification)."""
        return str((self.verification or {}).get("state", "unverified"))

    @staticmethod
    def new_id() -> str:
        return f"se_{uuid.uuid4().hex[:12]}"

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
            "pattern": {"predicates": self.predicates},
            "expected": {
                "quality_hat": self.expected_quality_hat,
                "quality_interval": [self.quality_interval[0], self.quality_interval[1]],
                "cost_hat": self.expected_cost_hat.to_dict(),
                "cost_interval": {d: [lo, hi] for d, (lo, hi)
                                  in self.cost_interval.items()},
                # Measured-dimension mask for the cost estimate. Written only
                # when known (None = unknown provenance, e.g. entries induced
                # before this field existed) — legacy payloads are NOT
                # retroactively granted a mask.
                "cost_measured": (sorted(self.expected_cost_hat.measured)
                                  if self.expected_cost_hat.measured is not None
                                  else None),
                "failure_prob": self.failure_prob,
            },
            "applicability": list(self.applicability),
            "risk_conditions": list(self.risk_conditions),
            "fallback_strategy_id": self.fallback_strategy_id,
            "status": self.status,
            "prediction_track": self.prediction_track.to_dict(),
            "provenance": list(self.provenance),
            "support_n": self.support_n,
            "cost_support_n": {d: int(n) for d, n in
                                self.cost_support_n.items()},
            "strategy_type": self.strategy_type,
            "actions": list(self.actions),
            "verification": (dict(self.verification)
                             if self.verification else None),
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
        # Canonical form: predicates are the only content of a pattern.
        predicates = dict(pattern.get("predicates") or {})
        status = data.get("status", "candidate")
        if status not in ENTRY_STATUSES:
            raise ValueError(f"status must be one of {ENTRY_STATUSES}")
        expected = data.get("expected") or {}
        interval = expected.get("quality_interval") or [0.0, 1.0]
        cost_interval = {d: (float(lo), float(hi))
                         for d, (lo, hi) in
                         (expected.get("cost_interval") or {}).items()}
        cost_hat = CostVector.from_dict(expected.get("cost_hat") or {})
        # Measured-dimension mask comes ONLY from the explicit serialized
        # key. It is deliberately NOT inferred from the cost_interval key
        # set: legacy entries were given a fixed interval for all five
        # dimensions (including never-backfilled llm_tokens), so the keys
        # cannot prove measurement. Without the key the vector stays
        # maskless and falls back to value-level legacy inference — an
        # unconfirmed legacy zero stays unknown.
        raw_measured = expected.get("cost_measured")
        if isinstance(raw_measured, list):
            cost_hat.measured = {str(d) for d in raw_measured
                                 if d in COST_DIMENSIONS}
        return cls(
            entry_id=str(data["entry_id"]),
            strategy_id=str(data["strategy_id"]),
            pattern={"predicates": predicates},
            expected_quality_hat=float(expected.get("quality_hat", 0.5)),
            quality_interval=(float(interval[0]), float(interval[1])),
            expected_cost_hat=cost_hat,
            cost_interval=cost_interval,
            failure_prob=float(expected.get("failure_prob", 0.5)),
            applicability=_applicability_notes(data.get("applicability")),
            risk_conditions=[str(r) for r in (data.get("risk_conditions") or [])],
            fallback_strategy_id=data.get("fallback_strategy_id"),
            status=status,
            prediction_track=PredictionTrack.from_dict(data.get("prediction_track")),
            provenance=[str(p) for p in (data.get("provenance") or [])],
            support_n=int(data.get("support_n", 0)),
            cost_support_n={str(d): int(n) for d, n in
                            (data.get("cost_support_n") or {}).items()},
            strategy_type=data.get("strategy_type"),
            actions=[str(a) for a in (data.get("actions") or [])],
            verification=(_verification_block(data["verification"])
                          if data.get("verification") is not None else {}),
            last_consulted_at=data.get("last_consulted_at"),
            created_at=float(data.get("created_at", time.time())),
        )


#: Admission-verification outcomes. ``unverified`` is the honest default for
#: an entry that has not been through an offline check (including every entry
#: written before this field existed); ``insufficient_evidence`` covers "the
#: check ran but could not decide" — a failed execution is NOT a refutation.
VERIFICATION_STATES = ("unverified", "verified", "insufficient_evidence",
                       "refuted")


def _verification_block(raw: Any) -> Dict[str, Any]:
    """Normalize the verification block (compact, one dict).

    Shape: ``{"state", "purpose", "claim", "checks", "evidence",
    "conclusion", "verified_at"}``. Only ``state`` is load-bearing for
    admission; the rest is the audit trail the harness reads back."""
    data = dict(raw) if isinstance(raw, dict) else {}
    state = str(data.get("state", "unverified"))
    if state not in VERIFICATION_STATES:
        state = "unverified"
    return {
        "state": state,
        "purpose": data.get("purpose"),
        "claim": data.get("claim"),
        "checks": list(data.get("checks") or []),
        "evidence": list(data.get("evidence") or []),
        "conclusion": data.get("conclusion"),
        "verified_at": data.get("verified_at"),
    }


def empty_verification() -> Dict[str, Any]:
    """The honest default block for an entry nobody has verified yet."""
    return _verification_block(None)


def _applicability_notes(raw: Any) -> List[str]:
    """Harness notes: plain strings. Legacy payloads stored condition
    objects ``{text, verified, supporting_execution_ids}`` — the text is
    kept, the (never-populated) verification flags are dropped."""
    notes: List[str] = []
    for item in (raw or []):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            notes.append(text)
    return notes


# ---------------------------------------------------------------------------
# Cold archive card
# ---------------------------------------------------------------------------


@dataclass
class ColdArchiveCard:
    """Compressed card (~200 B) for a retired entry — the cold archive
    record. Kept by default; consulted before induction so the same evidence
    cannot resurrect the same failed generalization (anti-resurrection).
    ``orx induce --force`` removes the card for this pattern (the harness
    judging the environment drifted), after which induction proceeds
    normally."""

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
