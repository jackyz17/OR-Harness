"""Offline structural induction package (Phase 3, PART 3).

Implements the Realization -> Pattern -> Repository induction loop:
discover structurally-isomorphic-but-heterogeneous realization clusters, align
their roles, induce candidate principles (hypotheses), refute/refine them via
solver-backed counterexample search, and validate them on unseen transfer tasks
before appending validated patterns (abstraction_depth=2) back to the Modeling Bank.

Design red lines (redesign-plan.md §0):
- Induction != Summary: Structure -> Relation -> Hypothesis -> Verification.
- Heterogeneous complementarity: A!=B!=C but Structure(A)~=Structure(B)~=Structure(C),
  NOT Auto-Dreamer-style redundancy compression.
- Hypotheses must survive counterexample search + solver transfer validation.
- Append-only: induced records are NEW peer records (status=validated); source records stay untouched.
- D18 harness principle: the framework emits prompts + validation; the agent owns the LLM.
"""

from .alignment import (
    AlignmentMap,
    LLMBackedAligner,
    StructuralAligner,
)
from .candidates import (
    CandidateCluster,
    ClusterMember,
    SignatureClusterer,
)
from .encoding import (
    EncodingResult,
    LLMBackedEncoder,
    StructuralEncoder,
)
from .counterexample import (
    CounterexampleSearcher,
    LLMBackedCounterexampleSearcher,
    RefutationAttempt,
    RefutationResult,
)
from .inducer import (
    LLMBackedInducer,
    PatternInducer,
    PrincipleHypothesis,
)
from .pipeline import (
    InductionPipeline,
    InductionRunReport,
    run_induction_sync,
)
from .trigger import (
    InductionTrigger,
    TriggerDecision,
)
from .validation import (
    LLMBackedValidator,
    PatternValidator,
    ScoringWeights,
    ValidationOutcome,
)

__all__ = [
    "AlignmentMap",
    "CandidateCluster",
    "ClusterMember",
    "CounterexampleSearcher",
    "EncodingResult",
    "InductionPipeline",
    "InductionRunReport",
    "InductionTrigger",
    "LLMBackedAligner",
    "LLMBackedCounterexampleSearcher",
    "LLMBackedEncoder",
    "LLMBackedInducer",
    "LLMBackedValidator",
    "PatternInducer",
    "PatternValidator",
    "PrincipleHypothesis",
    "RefutationAttempt",
    "RefutationResult",
    "ScoringWeights",
    "SignatureClusterer",
    "StructuralAligner",
    "StructuralEncoder",
    "TriggerDecision",
    "ValidationOutcome",
    "run_induction_sync",
]
