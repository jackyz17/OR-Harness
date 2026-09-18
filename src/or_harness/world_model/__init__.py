"""World-model substrate and unified prediction contracts.

This package is NOT a world model. It is the data foundation and the
CONTRACT layer one needs before a world model can be built: frozen
pre-action belief snapshots, a unified record of the six action classes
(including the ones the OUTER agent performs), budget accounting, the links
between them, and the versioned prediction contracts that a model service
will later fill in.

Design rules inherited from the research consensus:

- Facts vs inference vs unknown: every key field carries an origin label
  and, where applicable, an evidence reference. Unknown is never written
  as a measured zero.
- Recommendation != selection != execution: a recall result is never
  auto-interpreted as a chosen plan; a chosen plan never fabricates
  execution quality.
- Real vs hypothetical: hypothetical branches are labelled
  ``source="hypothetical"`` and can never enter execution statistics or
  the knowledge-verification gate.
- The verification gate is absolute: nothing recorded here promotes a
  candidate into verified knowledge — that remains ``induce --verify``.
- No hidden calls, no background loops: everything is an explicit,
  stateless invocation against an explicit home, like the rest of the
  harness.

Prediction contracts (see :mod:`or_harness.world_model.contracts`):

- :class:`StrategyOutcomePrediction` — a candidate strategy's benefit,
  resource cost, risk and uncertainty;
- :class:`CapabilityEvolutionPrediction` — how a candidate learning
  operation would change future task performance, judged through
  observable consequences of ``H = F(M, W_OR, Pi, R, T)``.

Both are versioned, serializable and validatable, and both can be built
BEFORE a prediction service exists: the resulting object says
``contract_only`` rather than pretending a forecast was made.
"""

from or_harness.world_model.actions import (
    ACTION_TYPES,
    LEGACY_ACTION_TYPES,
    ActionLog,
    ActionRecord,
)
from or_harness.world_model.budget import BUDGET_STATUSES, BudgetLedger
from or_harness.world_model.contracts import (
    BASELINE_KINDS,
    BENEFIT_KINDS,
    CAPABILITY_EVIDENCE_STATUSES,
    CAPABILITY_SOURCES,
    CONTRACT_STATUSES,
    CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    LEGACY_FIELD_MIGRATION,
    LEGACY_MODE_TO_KIND,
    LEGACY_UNMAPPABLE,
    PREDICTION_KINDS,
    PREDICTION_SCOPES,
    SUPPORTED_CONTRACT_VERSIONS,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityEvolutionPrediction,
    CapabilitySourceEvidence,
    EvidenceRef,
    ExpectedChange,
    ExpectedCost,
    ExperienceScope,
    HarnessCapabilityEvidence,
    LearningOperation,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    StrategyOutcomePrediction,
    TaskTargeting,
    UncertaintyStatement,
    UnsupportedContractVersion,
    VerificationCondition,
    capability_evidence_from_legacy_harness_state,
    detect_payload_version,
    legacy_prediction_view,
    load_contract_payload,
    prediction_kinds_for_mode,
    validate_capability_evolution,
    validate_capability_evidence,
    validate_contract,
    validate_strategy_outcome,
)
from or_harness.world_model.execution_window import (
    DEFAULT_AUXILIARY_ACTION_TYPES,
    DEFAULT_IN_SCOPE_ACTION_TYPES,
    WINDOW_ACTION_TYPES,
    StrategyExecutionWindow,
    build_execution_window,
    window_id_for,
    window_ref,
)
from or_harness.world_model.state import (
    BeliefSnapshot,
    KnowledgeRef,
    MAINTENANCE_TASK_ID,
    task_text,
    task_text_digest,
    verified_knowledge_view,
)

__all__ = [
    # state / actions / budget substrate
    "ACTION_TYPES",
    "LEGACY_ACTION_TYPES",
    "ActionLog",
    "ActionRecord",
    "BUDGET_STATUSES",
    "BudgetLedger",
    "BeliefSnapshot",
    "KnowledgeRef",
    "MAINTENANCE_TASK_ID",
    "task_text",
    "task_text_digest",
    "verified_knowledge_view",
    # execution window (attempt vs strategy window)
    "DEFAULT_AUXILIARY_ACTION_TYPES",
    "DEFAULT_IN_SCOPE_ACTION_TYPES",
    "WINDOW_ACTION_TYPES",
    "StrategyExecutionWindow",
    "build_execution_window",
    "window_id_for",
    "window_ref",
    # unified contracts
    "CONTRACT_VERSION",
    "LEGACY_CONTRACT_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS",
    "PREDICTION_KINDS",
    "CONTRACT_STATUSES",
    "PREDICTION_SCOPES",
    "BENEFIT_KINDS",
    "BASELINE_KINDS",
    "CAPABILITY_SOURCES",
    "CAPABILITY_EVIDENCE_STATUSES",
    "LEGACY_FIELD_MIGRATION",
    "LEGACY_MODE_TO_KIND",
    "LEGACY_UNMAPPABLE",
    "StrategyOutcomePrediction",
    "CapabilityEvolutionPrediction",
    "CandidateRef",
    "BenefitEstimate",
    "ExpectedCost",
    "RiskEvent",
    "RiskStatement",
    "UncertaintyStatement",
    "PredictionTrace",
    "EvidenceRef",
    "HarnessCapabilityEvidence",
    "CapabilitySourceEvidence",
    "LearningOperation",
    "ExperienceScope",
    "TaskTargeting",
    "BaselineStatement",
    "ExpectedChange",
    "VerificationCondition",
    "UnsupportedContractVersion",
    "validate_contract",
    "validate_strategy_outcome",
    "validate_capability_evolution",
    "validate_capability_evidence",
    "detect_payload_version",
    "load_contract_payload",
    "legacy_prediction_view",
    "capability_evidence_from_legacy_harness_state",
    "prediction_kinds_for_mode",
]

