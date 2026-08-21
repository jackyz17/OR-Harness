"""Modeling Bank schemas: structural signature, modeling experience, episode.

Per redesign-plan.md Phase 0 (modules 0.2 / 0.3 / 0.5):
- Structural signature uses core-4 dims (O/D/C/I) + open feature slots (Decision D9).
- The modeling schema is ONLY for modeling bank; other layers keep flat ExperienceRecord (D10).
- All records are peers — no depth hierarchy. Induced records (status=validated) coexist
  with directly-solved records (status=null).
- Each record has a required modeling_aspect (constraint/objective/variable/classification/structure).
- method is positive/negative unified: wrong_form + action_template in one record (D12).
- role_schema/role_mappings belong to induced records as provenance detail of the induction.

This module is deliberately independent from schemas.py so the three flat layers
(implementation / repair / solving) are untouched. All records remain append-only facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .schemas import utc_now


# ---------------------------------------------------------------------------
# Step 1: Structural Signature (module 0.2, Plan B: core-4 + open feature slots)
# ---------------------------------------------------------------------------

# Controlled vocabularies for the four core dimensions. Small and stable by design.
# Multi-valued dims (D, C) hold a list; single-valued dims (O, I) hold one value.
OBJECTIVE_STRUCTURES = (
    "linear",
    "convex",
    "minmax",
    "multi_objective_weighted",
    "feasibility_only",
)

DECISION_STRUCTURES = (
    "binary_assignment",
    "integer_batch",
    "continuous_flow",
    "multi_index_2d",
    "multi_index_3d",
)

CONSTRAINT_STRUCTURES = (
    "capacity",
    "flow_conservation",
    "assignment_exactly_once",
    "covering",
    "precedence",
    "big_m_linking",
)

INTERACTION_COUPLINGS = (
    "independent",
    "shared_resource_coupled",
    "fixed_charge_coupling",
    "nonlinear_interaction",
)

# Recommended (non-binding) keys for the open feature slot. Free growth is allowed;
# high-frequency keys get periodically adopted into this recommendation list (D9).
RECOMMENDED_FEATURE_KEYS = (
    "temporal",
    "network",
    "resource",
    "uncertainty",
)


class SignatureValidationError(ValueError):
    """Raised when a structural signature uses an out-of-vocabulary core value."""


@dataclass
class StructuralSignature:
    """Structural fingerprint of an optimization model.

    Core four dims (objective/decision/constraint/interaction) always carry
    information for any math-programming problem; problem-specific structure
    (temporal, network, resource, uncertainty, ...) goes to open feature slots
    and is only matched on key intersection during alignment (missing != penalty).
    """

    objective: str = "linear"                       # O: single value from OBJECTIVE_STRUCTURES
    decision: List[str] = field(default_factory=list)    # D: multi-value from DECISION_STRUCTURES
    constraint: List[str] = field(default_factory=list)  # C: multi-value from CONSTRAINT_STRUCTURES
    interaction: str = "independent"                # I: single value from INTERACTION_COUPLINGS
    features: Dict[str, str] = field(default_factory=dict)  # open slots, e.g. temporal/network/resource

    def validate(self) -> "StructuralSignature":
        """Validate core-dim values against controlled vocabularies.

        Features are intentionally NOT validated (open slot). Returns self for chaining.
        """
        if self.objective not in OBJECTIVE_STRUCTURES:
            raise SignatureValidationError(
                "objective {!r} not in OBJECTIVE_STRUCTURES".format(self.objective)
            )
        for value in self.decision:
            if value not in DECISION_STRUCTURES:
                raise SignatureValidationError(
                    "decision {!r} not in DECISION_STRUCTURES".format(value)
                )
        for value in self.constraint:
            if value not in CONSTRAINT_STRUCTURES:
                raise SignatureValidationError(
                    "constraint {!r} not in CONSTRAINT_STRUCTURES".format(value)
                )
        if self.interaction not in INTERACTION_COUPLINGS:
            raise SignatureValidationError(
                "interaction {!r} not in INTERACTION_COUPLINGS".format(self.interaction)
            )
        return self

    def math_type_summary(self) -> str:
        """One-line human summary generated from the signature (D15: math_type is
        not stored separately, it is derived to avoid dual maintenance)."""
        decision = "+".join(self.decision) if self.decision else "unknown_decision"
        return "{d} problem with {i} coupling, {o} objective".format(
            d=decision, i=self.interaction, o=self.objective
        )

    def core_key(self) -> str:
        """Canonical core-dim key for the fixed inverted index (module 0.2)."""
        return "|".join(
            [
                self.objective,
                ",".join(sorted(self.decision)),
                ",".join(sorted(self.constraint)),
                self.interaction,
            ]
        )

    def shared_feature_keys(self, other: "StructuralSignature") -> List[str]:
        """Feature-key intersection used by alignment (missing keys are not penalized)."""
        return sorted(set(self.features) & set(other.features))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuralSignature":
        payload = dict(data or {})
        payload.setdefault("objective", "linear")
        payload.setdefault("decision", [])
        payload.setdefault("constraint", [])
        payload.setdefault("interaction", "independent")
        payload.setdefault("features", {})
        return cls(**payload).validate()


# ---------------------------------------------------------------------------
# Step 2: Modeling Bank schema — all records are peers (no depth hierarchy)
# ---------------------------------------------------------------------------

# Modeling aspect: which part of the model this experience targets. Single required
# value; agent self-classifies during comparative synthesis.
MODELING_ASPECTS = (
    "constraint",       # how to write constraints (Big-M linking, flow conservation, etc.)
    "objective",        # how to write the objective function (cost terms, holding cost timing, etc.)
    "variable",         # variable type/index selection (binary vs integer, multi-index, etc.)
    "classification",  # problem classification (is this really an optimization? direct calculation?)
    "structure",        # model structure / decomposition strategy (time decomposition, etc.)
)

# Validation status: available to ALL records (not just induced ones).
#   null        = directly solved, no unseen-transfer validation
#   "validated" = passed unseen transfer (induced records only)
#   "refuted"   = failed unseen transfer (never appended to the bank)
VALIDATION_STATUSES = (None, "validated", "refuted")


@dataclass
class MathScope:
    """Block 2: mathematical applicability scope of a modeling experience."""

    structural_signature: StructuralSignature = field(default_factory=StructuralSignature)
    exclusions: List[str] = field(default_factory=list)  # realization-level exclusion conditions

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathScope":
        payload = dict(data or {})
        payload["structural_signature"] = StructuralSignature.from_dict(
            payload.get("structural_signature", {})
        )
        payload.setdefault("exclusions", [])
        return cls(**payload)


@dataclass
class MethodBody:
    """Block 3 (method): positive/negative unified method body (D12).

    action_template and wrong_form live in the SAME record so the correct and
    incorrect forms never become two detached experiences.
    """

    action_template: str = ""          # parameterizable correct-action template
    wrong_form: Optional[str] = None   # typical wrong form (unified with action, D12)
    rationale: str = ""                # why the action works
    derivation_ref: Optional[str] = None  # textbook / paper / formula reference

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodBody":
        return cls(**dict(data or {}))


@dataclass
class RoleMappingEntry:
    """One realization's role binding inside a pattern's cross-problem mapping."""

    realization_id: str = ""
    problem_family: str = ""
    mapping: Dict[str, str] = field(default_factory=dict)  # role -> concrete entity

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleMappingEntry":
        payload = dict(data or {})
        payload.setdefault("mapping", {})
        return cls(**payload)


@dataclass
class CounterexampleRecord:
    """A solver-refuted case that shrinks a pattern's applicability (module 3.5)."""

    counterexample_id: str = field(default_factory=lambda: "cx_" + uuid4().hex[:12])
    summary: str = ""
    solver_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterexampleRecord":
        return cls(**dict(data or {}))


@dataclass
class TransferTest:
    """with/without-principle comparison on an unseen task (module 3.6)."""

    task: str = ""
    with_principle_objective: Optional[float] = None
    without_principle_objective: Optional[float] = None
    improvement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransferTest":
        return cls(**dict(data or {}))


@dataclass
class PatternValidation:
    """Validation evidence for a pattern: source consistency + unseen transfer."""

    source_consistency: str = ""                       # e.g. "3/3 source realizations satisfy"
    transfer_tests: List[TransferTest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternValidation":
        payload = dict(data or {})
        payload["transfer_tests"] = [
            TransferTest.from_dict(item) for item in payload.get("transfer_tests", [])
        ]
        return cls(**payload)


@dataclass
class PatternScoring:
    """Score decomposition: alpha*C + beta*T + gamma*V + delta*N - lambda*K - mu*X."""

    coverage: float = 0.0
    transferability: float = 0.0
    validation: float = 0.0
    novelty: float = 0.0
    complexity: float = 0.0
    counterexample_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternScoring":
        return cls(**dict(data or {}))


@dataclass
class ModelingEvidence:
    """Block 5: evidence chain. source_episodes points to Episode records; the
    attempt-level trajectory detail is owned by the Episode itself (D8)."""

    source_episodes: List[str] = field(default_factory=list)
    solver_feedback_summary: str = ""
    validation_level: str = "unverified"
    causal_confidence: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelingEvidence":
        payload = dict(data or {})
        payload.setdefault("source_episodes", [])
        return cls(**payload)


@dataclass
class ModelingExperience:
    """Modeling-bank record: a modeling method / math technique. All records are peers.

    Whether a record came from a single solve (derived_from_experience_ids is empty) or
    from offline induction (derived_from is non-empty), it is the SAME type of knowledge:
    a reusable modeling method. Provenance is metadata, not a hierarchy level.

    Induction-produced records additionally populate role_schema, role_mappings,
    applicability_conditions, counterexamples, validation, and scoring as provenance
    detail of the induction process.
    """

    # Block 1: identity & classification
    title: str = ""
    polarity: str = "positive"
    retrieval_text: str = ""
    layer: str = "modeling"
    experience_id: str = field(default_factory=lambda: "exp_" + uuid4().hex)

    # Block 2/3: math scope + method body
    math_scope: MathScope = field(default_factory=MathScope)
    method: MethodBody = field(default_factory=MethodBody)

    # Modeling aspect (required): which part of the model this experience targets.
    modeling_aspect: str = "constraint"

    # Block 5: evidence chain
    evidence: ModelingEvidence = field(default_factory=ModelingEvidence)

    # Block 6: lineage relations
    derived_from_experience_ids: List[str] = field(default_factory=list)
    contradicts_experience_ids: List[str] = field(default_factory=list)

    # Block 7: metadata
    created_at: str = field(default_factory=utc_now)
    content_hash: str = ""

    # Validation status (available to ALL records, not just induced ones).
    # null = directly solved; "validated" = passed unseen transfer; "refuted" = never appended.
    status: Optional[str] = None

    # Induction provenance detail (populated only for induced records; empty otherwise)
    role_schema: Dict[str, str] = field(default_factory=dict)
    role_mappings: List[RoleMappingEntry] = field(default_factory=list)
    applicability_conditions: List[str] = field(default_factory=list)
    counterexamples: List[CounterexampleRecord] = field(default_factory=list)
    validation: PatternValidation = field(default_factory=PatternValidation)
    scoring: PatternScoring = field(default_factory=PatternScoring)

    def validate(self) -> "ModelingExperience":
        self.math_scope.structural_signature.validate()
        if self.modeling_aspect not in MODELING_ASPECTS:
            raise ValueError(
                "modeling_aspect {!r} not in MODELING_ASPECTS".format(self.modeling_aspect)
            )
        if self.status is not None and self.status not in ("validated", "refuted"):
            raise ValueError("status {!r} must be null, 'validated', or 'refuted'".format(self.status))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_content_hash(self) -> str:
        """Hash over semantic content, excluding identity/time/hash fields."""
        data = self.to_dict()
        for ignored in ("experience_id", "created_at", "content_hash"):
            data.pop(ignored, None)
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.content_hash = hashlib.sha256(encoded).hexdigest()
        return self.content_hash

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelingExperience":
        payload = dict(data or {})
        payload["math_scope"] = MathScope.from_dict(payload.get("math_scope", {}))
        payload["method"] = MethodBody.from_dict(payload.get("method", {}))
        payload["evidence"] = ModelingEvidence.from_dict(payload.get("evidence", {}))
        payload["role_mappings"] = [
            RoleMappingEntry.from_dict(item) for item in payload.get("role_mappings", [])
        ]
        payload["counterexamples"] = [
            CounterexampleRecord.from_dict(item) for item in payload.get("counterexamples", [])
        ]
        payload["validation"] = PatternValidation.from_dict(payload.get("validation", {}))
        payload["scoring"] = PatternScoring.from_dict(payload.get("scoring", {}))
        payload.setdefault("derived_from_experience_ids", [])
        payload.setdefault("contradicts_experience_ids", [])
        payload.setdefault("role_schema", {})
        payload.setdefault("applicability_conditions", [])
        payload.setdefault("modeling_aspect", "constraint")
        payload.setdefault("status", None)
        # Migration: remove obsolete fields from old-format records
        payload.pop("abstraction_depth", None)
        payload.pop("realization_of_pattern_id", None)
        payload.pop("general_principle", None)
        return cls(**payload)


# ---------------------------------------------------------------------------
# Step 3: Episode record (module 0.5) — problem-level scene snapshot
# ---------------------------------------------------------------------------

@dataclass
class BranchSummary:
    """Compact per-branch outcome inside an Episode."""

    solver: str = ""
    status: str = "unknown"
    attempts: int = 0
    objective_value: Optional[float] = None
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchSummary":
        return cls(**dict(data or {}))


@dataclass
class EpisodeRecord:
    """Problem-level scene snapshot: the narrative record of one solve task.

    Episode answers "what happened on this problem" and is the provenance target
    of realizations (evidence.source_episodes) plus the raw material for offline
    induction. It is NOT used for online retrieval (D8).
    """

    problem: str = ""
    normalized_spec: Dict[str, Any] = field(default_factory=dict)
    structural_signature: StructuralSignature = field(default_factory=StructuralSignature)
    branches: List[BranchSummary] = field(default_factory=list)
    final_objective: Optional[float] = None
    gold_answer: Optional[float] = None
    produced_realization_ids: List[str] = field(default_factory=list)
    episode_id: str = field(default_factory=lambda: "ep_" + uuid4().hex[:12])
    problem_id: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeRecord":
        payload = dict(data or {})
        payload["structural_signature"] = StructuralSignature.from_dict(
            payload.get("structural_signature", {})
        )
        payload["branches"] = [BranchSummary.from_dict(item) for item in payload.get("branches", [])]
        payload.setdefault("produced_realization_ids", [])
        payload.setdefault("normalized_spec", {})
        return cls(**payload)


__all__ = [
    "OBJECTIVE_STRUCTURES",
    "DECISION_STRUCTURES",
    "CONSTRAINT_STRUCTURES",
    "INTERACTION_COUPLINGS",
    "RECOMMENDED_FEATURE_KEYS",
    "SignatureValidationError",
    "StructuralSignature",
    "MODELING_ASPECTS",
    "VALIDATION_STATUSES",
    "MathScope",
    "MethodBody",
    "RoleMappingEntry",
    "CounterexampleRecord",
    "TransferTest",
    "PatternValidation",
    "PatternScoring",
    "ModelingEvidence",
    "ModelingExperience",
    "BranchSummary",
    "EpisodeRecord",
]
