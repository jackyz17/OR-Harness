"""Validation & selection (module 3.6).

This is the VERIFICATION step that separates Induction from Summary. A candidate principle
only becomes a validated Pattern if BOTH hold:
  - Source consistency:  P(M_i) ~= true for every source realization it was induced from.
  - Unseen transfer:     Transfer(P, unseen OR tasks) > 0 — solving an UNSEEN problem WITH
                         the principle beats solving it WITHOUT (objective/gap/feasibility).

If we skipped unseen transfer, a "principle" could be a mere restatement of its sources
(summary). The with/without comparison on a task NOT in the source set is the operational
proof that something generalizable was actually induced (P not in M_i, yet Transfer(P)>0).

Scoring (discussion draft):  Score = alpha*C + beta*T + gamma*V + delta*N - lambda*K - mu*X
  C Coverage        fraction of source realizations the principle is consistent with
  T Transferability measured improvement on unseen tasks
  V Validation      solver-validation strength (from counterexample survival)
  N Novelty         1 if the principle asserts structure absent from each single source
  K Complexity      Minimum-Explanation penalty (from the hypothesis)
  X Counterexample  penalty per confirmed solver-refuted counterexample

Outcome is a status machine transition: hypothesis -> validated | refuted. Refuted
hypotheses are KEPT (append-only), never deleted.

The solve calls for transfer tests are injectable (the harness reuses the orchestrator);
this module owns the comparison + scoring logic, never the LLM or solver itself.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.modeling_schemas import PatternScoring, PatternValidation, TransferTest
from .counterexample import RefutationResult
from .inducer import PrincipleHypothesis


# A transfer solver: given (task_text, principle_or_None) -> objective (float) or None.
# Lower-is-better vs higher-is-better is normalized by the caller via `sense`.
TransferSolver = Callable[[str, Optional[str]], Awaitable[Optional[float]]]


@dataclass
class ScoringWeights:
    alpha: float = 1.0    # coverage
    beta: float = 1.0     # transferability
    gamma: float = 1.0    # validation
    delta: float = 1.0    # novelty
    lam: float = 1.0      # complexity penalty
    mu: float = 1.0       # counterexample penalty


@dataclass
class ValidationOutcome:
    hypothesis_id: str
    scoring: PatternScoring = field(default_factory=PatternScoring)
    validation: PatternValidation = field(default_factory=PatternValidation)
    status: str = "hypothesis"          # validated | refuted
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "scoring": self.scoring.to_dict(),
            "validation": self.validation.to_dict(),
            "rationale": self.rationale,
        }


class PatternValidator:
    """Owns source-consistency, unseen-transfer comparison, and the scoring function."""

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        validation_threshold: float = 0.5,
        min_transfer_improvement: float = 0.0,
    ):
        self.weights = weights or ScoringWeights()
        self.validation_threshold = validation_threshold
        self.min_transfer_improvement = min_transfer_improvement

    # -- source consistency -------------------------------------------------

    def source_consistency(
        self, hypothesis: PrincipleHypothesis, consistent_source_ids: List[str]
    ) -> float:
        """Coverage C = fraction of the hypothesis's sources it is consistent with."""
        sources = hypothesis.source_realization_ids
        if not sources:
            return 0.0
        hits = len([s for s in sources if s in set(consistent_source_ids)])
        return hits / float(len(sources))

    # -- unseen transfer ------------------------------------------------------

    async def unseen_transfer(
        self,
        hypothesis: PrincipleHypothesis,
        unseen_tasks: List[str],
        solver: TransferSolver,
        sense: str = "minimize",
    ) -> List[TransferTest]:
        """with/without-principle comparison on tasks NOT in the source set."""
        tests: List[TransferTest] = []
        for task in unseen_tasks:
            with_p = await solver(task, hypothesis.statement)
            without_p = await solver(task, None)
            improvement = self._improvement(with_p, without_p, sense)
            tests.append(
                TransferTest(
                    task=task,
                    with_principle_objective=with_p,
                    without_principle_objective=without_p,
                    improvement=improvement,
                )
            )
        return tests

    @staticmethod
    def _improvement(with_p: Optional[float], without_p: Optional[float], sense: str) -> Optional[str]:
        if with_p is None or without_p is None:
            return None
        delta = (without_p - with_p) if sense == "minimize" else (with_p - without_p)
        if delta > 1e-9:
            return "improved"
        if delta < -1e-9:
            return "degraded"
        return "neutral"

    def transferability(self, tests: List[TransferTest]) -> float:
        """Transferability T = fraction of unseen tasks the principle improved."""
        if not tests:
            return 0.0
        improved = len([t for t in tests if t.improvement == "improved"])
        return improved / float(len(tests))

    # -- novelty / complexity / counterexample --------------------------------

    def novelty(self, hypothesis: PrincipleHypothesis, source_texts: List[str]) -> float:
        """Novel proxy: 1.0 if the principle's statement is NOT contained in any single source.

        A summary restates a source; an induction asserts structure present in NONE of them.
        """
        statement = hypothesis.statement.strip().lower()
        if not statement:
            return 0.0
        for text in source_texts:
            if statement and statement in (text or "").lower():
                return 0.0
        return 1.0

    def counterexample_penalty(self, refutation: Optional[RefutationResult]) -> float:
        if refutation is None:
            return 0.0
        return float(len(refutation.counterexamples))

    def validation_strength(self, refutation: Optional[RefutationResult]) -> float:
        """V: 1.0 if the hypothesis survived counterexample search with executed attempts."""
        if refutation is None or not refutation.attempts:
            return 0.5  # unsearched: neutral, not penalized as refuted
        if refutation.refuted:
            return 0.0
        executed = [a for a in refutation.attempts if a.executed]
        return 1.0 if executed else 0.5

    # -- scoring ----------------------------------------------------------------

    def score(
        self,
        hypothesis: PrincipleHypothesis,
        consistent_source_ids: List[str],
        transfer_tests: List[TransferTest],
        refutation: Optional[RefutationResult],
        source_texts: List[str],
    ) -> PatternScoring:
        w = self.weights
        coverage = self.source_consistency(hypothesis, consistent_source_ids)
        transfer = self.transferability(transfer_tests)
        validation = self.validation_strength(refutation)
        novelty = self.novelty(hypothesis, source_texts)
        complexity = hypothesis.complexity
        cx = self.counterexample_penalty(refutation)
        total = (
            w.alpha * coverage
            + w.beta * transfer
            + w.gamma * validation
            + w.delta * novelty
            - w.lam * complexity
            - w.mu * cx
        )
        return PatternScoring(
            coverage=coverage,
            transferability=transfer,
            validation=validation,
            novelty=novelty,
            complexity=complexity,
            counterexample_penalty=cx,
            total=total,
        )

    # -- verdict ------------------------------------------------------------------

    def decide(
        self,
        hypothesis: PrincipleHypothesis,
        scoring: PatternScoring,
        validation: PatternValidation,
        refutation: Optional[RefutationResult],
    ) -> ValidationOutcome:
        """hypothesis -> validated | refuted. Refuted is kept, not deleted (append-only)."""
        transfer_improved = any(t.improvement == "improved" for t in validation.transfer_tests)
        passes = (
            not (refutation and refutation.refuted and scoring.validation == 0.0)
            and scoring.total >= self.validation_threshold
            and transfer_improved
        )
        status = "validated" if passes else "refuted"
        rationale = (
            "total={:.3f} (threshold {:.3f}), coverage={:.2f}, transfer={:.2f}, "
            "validation={:.2f}, novelty={:.2f}, transfer_improved={}".format(
                scoring.total, self.validation_threshold, scoring.coverage,
                scoring.transferability, scoring.validation, scoring.novelty, transfer_improved,
            )
        )
        return ValidationOutcome(
            hypothesis_id=hypothesis.hypothesis_id,
            scoring=scoring,
            validation=validation,
            status=status,
            rationale=rationale,
        )


class LLMBackedValidator:
    """OPTIONAL end-to-end driver for standalone runs/tests (NOT used in harness mode)."""

    def __init__(self, validator: Optional[PatternValidator] = None):
        self.validator = validator or PatternValidator()

    async def validate(
        self,
        hypothesis: PrincipleHypothesis,
        consistent_source_ids: List[str],
        unseen_tasks: List[str],
        solver: TransferSolver,
        refutation: Optional[RefutationResult],
        source_texts: List[str],
        sense: str = "minimize",
    ) -> ValidationOutcome:
        transfer_tests = await self.validator.unseen_transfer(hypothesis, unseen_tasks, solver, sense)
        validation = PatternValidation(
            source_consistency="{}/{} sources consistent".format(
                len(consistent_source_ids), len(hypothesis.source_realization_ids)
            ),
            transfer_tests=transfer_tests,
        )
        scoring = self.validator.score(
            hypothesis, consistent_source_ids, transfer_tests, refutation, source_texts
        )
        return self.validator.decide(hypothesis, scoring, validation, refutation)


__all__ = [
    "PatternValidator",
    "LLMBackedValidator",
    "ScoringWeights",
    "ValidationOutcome",
]
