"""Strict, dependency-free domain schemas for the OR experience bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ExperienceLayer(StrEnum):
    MODELING = "modeling"
    IMPLEMENTATION = "implementation"
    REPAIR = "repair"
    SOLVING = "solving"


class ExperiencePolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ExperienceGenerality(StrEnum):
    SOLVER_AGNOSTIC = "solver_agnostic"
    SOLVER_FAMILY = "solver_family"
    SOLVER_SPECIFIC = "solver_specific"


class TerminationReason(StrEnum):
    SOLVED = "solved"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    MAX_ATTEMPTS = "max_attempts"
    REPEATED_ERROR = "repeated_error"
    UNCHANGED_CODE = "unchanged_code"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL_ISSUE = "numerical_issue"
    UNSUPPORTED_SOLVER = "unsupported_solver"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    LICENSE_ERROR = "license_error"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN = "unknown"


class ValidationLevel(StrEnum):
    UNVERIFIED = "unverified"
    RUNTIME_ONLY = "runtime_only"
    SOLVER_FEASIBLE = "solver_feasible"
    SEMANTIC_CHECKED = "semantic_checked"
    CROSS_SOLVER_CONSISTENT = "cross_solver_consistent"


@dataclass
class ProblemContext:
    problem_family: str = "general_milp"
    objective_type: str = "unknown"
    stage: str = "formulation"
    keywords: List[str] = field(default_factory=list)


@dataclass
class ExperienceScope:
    generality: str = ExperienceGenerality.SOLVER_AGNOSTIC.value
    solver_family: Optional[str] = None
    solver: Optional[str] = None
    language: Optional[str] = "python"
    api: Optional[str] = None


@dataclass
class ExperienceTrigger:
    situation: str = ""
    normalized_error: Optional[str] = None
    solver_status: Optional[str] = None
    performance_symptom: Optional[str] = None


@dataclass
class ExperiencePolicy:
    diagnosis: str = ""
    action: str = ""
    rationale: str = ""
    example: Optional[str] = None
    limitations: Optional[str] = None


@dataclass
class ExperienceEvidence:
    problem_id: str = ""
    branch_ids: List[str] = field(default_factory=list)
    attempt_ids: List[str] = field(default_factory=list)
    solver_feedback_summary: str = ""
    validation_level: str = ValidationLevel.UNVERIFIED.value
    causal_confidence: str = "low"


@dataclass
class ExperienceRecord:
    layer: str
    polarity: str
    title: str
    retrieval_text: str
    problem_context: ProblemContext
    scope: ExperienceScope
    trigger: ExperienceTrigger
    policy: ExperiencePolicy
    evidence: ExperienceEvidence
    schema_version: str = "1.0"
    experience_id: str = field(default_factory=lambda: "exp_" + uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    related_experience_ids: List[str] = field(default_factory=list)
    derived_from_experience_ids: List[str] = field(default_factory=list)
    contradicts_experience_ids: List[str] = field(default_factory=list)
    possible_duplicate_of: Optional[str] = None
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        payload = dict(data)
        payload["problem_context"] = ProblemContext(**payload["problem_context"])
        payload["scope"] = ExperienceScope(**payload["scope"])
        payload["trigger"] = ExperienceTrigger(**payload["trigger"])
        payload["policy"] = ExperiencePolicy(**payload["policy"])
        payload["evidence"] = ExperienceEvidence(**payload["evidence"])
        return cls(**payload)


@dataclass
class RetrievalHit:
    experience_id: str
    layer: str
    title: str
    score: float
    polarity: str
    scope: Dict[str, Any]
    retrieval_text: str
    record: Dict[str, Any]


@dataclass
class ValidationReport:
    valid: bool
    validation_level: str = ValidationLevel.UNVERIFIED.value
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    objective_comparable: bool = True


@dataclass
class SolverExecutionResult:
    status: str
    solver: str
    exit_code: Optional[int] = None
    objective_sense: str = "unknown"
    objective_value: Optional[float] = None
    objective_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_seconds: Optional[float] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    normalized_error: Optional[str] = None
    result_path: Optional[str] = None


@dataclass
class AttemptRecord:
    attempt_id: str
    problem_id: str
    branch_id: str
    solver: str
    attempt_number: int
    started_at: str
    finished_at: str
    retrieved_experience_ids: Dict[str, List[str]]
    problem_summary: str
    formulation_summary: str
    code_path: str
    code_hash: str
    stdout_summary: str = ""
    stderr_summary: str = ""
    normalized_error: Optional[str] = None
    solver_status: str = "unknown"
    objective_value: Optional[float] = None
    objective_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_seconds: Optional[float] = None
    validator_report: Dict[str, Any] = field(default_factory=dict)
    validation_level: str = ValidationLevel.UNVERIFIED.value
    repair_action_summary: str = ""
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BranchState:
    problem_id: str
    branch_id: str
    solver: str
    workspace: str
    current_formulation: str = ""
    current_code: str = ""
    latest_feedback: str = ""
    resolved_issues: List[str] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    ineffective_repairs: List[str] = field(default_factory=list)
    attempts: List[AttemptRecord] = field(default_factory=list)


@dataclass
class BranchResult:
    branch_id: str
    solver: str
    workspace: str
    attempts: List[AttemptRecord]
    execution: SolverExecutionResult
    validation: ValidationReport
    termination_reason: str

    @property
    def normalized_error(self) -> Optional[str]:
        """Shortcut to the latest execution's normalized_error — saves drilling
        into ``self.execution.normalized_error`` for quick debugging."""
        return self.execution.normalized_error if self.execution else None

    @property
    def solver_status(self) -> str:
        """Shortcut to the latest execution's status."""
        return self.execution.status if self.execution else "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveResult:
    problem_id: str
    selected_branch_id: Optional[str]
    selection_reason: str
    branches: List[BranchResult]
    retrieved_experience_ids: Dict[str, List[str]]
    appended_experience_ids: List[str]
    duplicate_experience_ids: List[str]
    validation_level: str
    warnings: List[str]
    timeline: List[Dict[str, Any]]
    objective_comparable: bool = True
    branch_discrepancies: List[str] = field(default_factory=list)

    @property
    def branch_errors(self) -> Dict[str, str]:
        """Map of branch_id -> normalized_error for all branches that have an error.
        Quick diagnostic: ``result.branch_errors`` shows which branches failed and why,
        without drilling into ``result.branches[i].execution.normalized_error``."""
        errors: Dict[str, str] = {}
        for b in self.branches:
            if b.normalized_error:
                errors[b.branch_id] = b.normalized_error
        return errors

    @property
    def branch_statuses(self) -> Dict[str, str]:
        """Map of branch_id -> solver_status for all branches."""
        return {b.branch_id: b.solver_status for b in self.branches}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceAppendResult:
    status: str
    experience_id: str
    content_hash: str
    layer: str
    duplicate_of: Optional[str] = None


@dataclass
class AvailabilityResult:
    available: bool
    reason: str = ""
    termination_reason: Optional[str] = None

