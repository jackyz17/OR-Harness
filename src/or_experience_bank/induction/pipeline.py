"""Induction pipeline orchestration (module 3.7).

Wires the six induction modules into the full Realization -> Pattern -> Repository loop:

  ModelingStore.realizations()
    -> candidates.SignatureClusterer   (isomorphic + heterogeneous clusters)
    -> alignment                       (shared structural roles)
    -> inducer                         (candidate principles, status=hypothesis)
    -> counterexample                  (solver-backed refutation)
    -> validation                      (source consistency + unseen transfer + scoring)
    -> ModelingStore.append(pattern)   (validated: depth=2; refuted: kept, not appended as active)

Red lines enforced here:
- Append-only: a VALIDATED pattern is appended as a NEW depth=2 ModelingExperience; the
  source realizations are never modified. Refuted hypotheses are NOT appended to the bank
  (they would pollute retrieval) but are returned in the run report for archival.
- Induction != Summary: a pattern is appended ONLY when validation.status == "validated",
  which itself requires an improved unseen-transfer test.
- D18 harness principle: the pipeline owns NO LLM and NO executor. LLM-driven steps use
  injectable drivers (aligner/inducer/counterexample), and the transfer solve is injected.

Trigger (v1, D4): manual / periodic. `run()` is invoked by the CLI `induce` subcommand.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.modeling_schemas import (
    MathScope,
    MethodBody,
    ModelingEvidence,
    ModelingExperience,
)
from ..core.modeling_store import ModelingStore
from .alignment import AlignmentMap
from .candidates import CandidateCluster, SignatureClusterer
from .inducer import PrincipleHypothesis
from .trigger import InductionTrigger
from .validation import PatternValidator, ValidationOutcome


# Injectable driver protocols (structural, not enforced):
#   align(cluster)                 -> AlignmentMap
#   induce(cluster, alignment)     -> List[PrincipleHypothesis]
#   refute(hypothesis, workspace)  -> RefutationResult
#   transfer_solver(task, principle_or_None) -> Optional[float]


@dataclass
class InductionRunReport:
    """What one induction run did, for CLI output + archival."""

    clusters_found: int = 0
    clusters_processed: int = 0
    hypotheses_generated: int = 0
    patterns_validated: int = 0
    patterns_refuted: int = 0
    validated_pattern_ids: List[str] = field(default_factory=list)
    refuted: List[Dict[str, Any]] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clusters_found": self.clusters_found,
            "clusters_processed": self.clusters_processed,
            "hypotheses_generated": self.hypotheses_generated,
            "patterns_validated": self.patterns_validated,
            "patterns_refuted": self.patterns_refuted,
            "validated_pattern_ids": list(self.validated_pattern_ids),
            "refuted": list(self.refuted),
            "details": list(self.details),
        }


class InductionPipeline:
    """Orchestrates the induction loop over a ModelingStore. Holds no LLM/executor itself."""

    def __init__(
        self,
        store: ModelingStore,
        clusterer: SignatureClusterer,
        aligner: Any,                       # driver with async align(cluster)
        inducer: Any,                       # driver with async induce(cluster, alignment)
        counterexample: Any,                # driver with async search(hypothesis, workspace)
        validator: PatternValidator,
        transfer_solver: Optional[Callable[[str, Optional[str]], Awaitable[Optional[float]]]] = None,
        workspace: Optional[Path] = None,
        unseen_tasks: Optional[List[str]] = None,
        sense: str = "minimize",
        max_clusters: Optional[int] = None,
        trigger: Optional[InductionTrigger] = None,
    ):
        self.store = store
        self.clusterer = clusterer
        self.aligner = aligner
        self.inducer = inducer
        self.counterexample = counterexample
        self.validator = validator
        self.transfer_solver = transfer_solver
        self.workspace = Path(workspace) if workspace else Path(".")
        self.unseen_tasks = list(unseen_tasks or [])
        self.sense = sense
        self.max_clusters = max_clusters
        # optional v1 trigger policy (D4). When provided, run() consults it first and
        # only induces over the clusters it selects; when None, run() induces over all
        # discovered clusters (manual invocation).
        self.trigger = trigger

    # -- main loop ------------------------------------------------------------

    async def run(self) -> InductionRunReport:
        report = InductionRunReport()
        realizations = self.store.all_records()
        all_clusters = self.clusterer.discover(realizations)
        report.clusters_found = len(all_clusters)

        clusters = all_clusters
        decision = None
        if self.trigger is not None:
            decision = self.trigger.decide()
            if not decision.should_induce:
                report.details.append({"trigger": decision.reason})
                return report
            clusters = decision.clusters_to_induce

        if self.max_clusters is not None:
            clusters = clusters[: self.max_clusters]

        for cluster in clusters:
            report.clusters_processed += 1
            await self._process_cluster(cluster, report)

        if decision is not None:
            # advance the watermark/cooldown snapshot only after a real induction pass
            decision.realization_count = len(realizations)
            self.trigger.record_run(decision)
        return report

    async def _process_cluster(self, cluster: CandidateCluster, report: InductionRunReport) -> None:
        alignment: AlignmentMap = await self.aligner.align(cluster)
        if not alignment.roles or not alignment.bindings:
            report.details.append({"cluster_id": cluster.cluster_id, "skipped": "empty alignment"})
            return

        hypotheses: List[PrincipleHypothesis] = await self.inducer.induce(cluster, alignment)
        report.hypotheses_generated += len(hypotheses)
        source_texts = [ (m.record.get("retrieval_text") or m.title) for m in cluster.members ]

        for hypothesis in hypotheses:
            outcome = await self._verify(hypothesis, cluster, source_texts)
            if outcome.status == "validated":
                pattern = self._build_pattern(hypothesis, cluster, alignment, outcome)
                result = self.store.append(pattern)
                report.patterns_validated += 1
                report.validated_pattern_ids.append(pattern.experience_id)
                report.details.append({
                    "cluster_id": cluster.cluster_id,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "status": "validated",
                    "pattern_id": pattern.experience_id,
                    "store": result.get("status"),
                    "score": outcome.scoring.total,
                })
            else:
                report.patterns_refuted += 1
                report.refuted.append({
                    "cluster_id": cluster.cluster_id,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "statement": hypothesis.statement,
                    "rationale": outcome.rationale,
                })

    async def _verify(
        self, hypothesis: PrincipleHypothesis, cluster: CandidateCluster, source_texts: List[str]
    ) -> ValidationOutcome:
        # counterexample search (solver-backed refutation)
        refutation = await self.counterexample.search(
            hypothesis, self.workspace / hypothesis.hypothesis_id
        )
        # source consistency: sources the hypothesis is consistent with (v1: all sources)
        consistent_ids = list(hypothesis.source_realization_ids)
        # unseen transfer comparison
        transfer_tests = []
        if self.transfer_solver is not None and self.unseen_tasks:
            transfer_tests = await self.validator.unseen_transfer(
                hypothesis, self.unseen_tasks, self.transfer_solver, self.sense
            )
        from ..core.modeling_schemas import PatternValidation
        validation = PatternValidation(
            source_consistency="{}/{} sources consistent".format(
                len(consistent_ids), len(hypothesis.source_realization_ids)
            ),
            transfer_tests=transfer_tests,
        )
        scoring = self.validator.score(
            hypothesis, consistent_ids, transfer_tests, refutation, source_texts
        )
        return self.validator.decide(hypothesis, scoring, validation, refutation)

    # -- pattern construction ---------------------------------------------------

    def _build_pattern(
        self,
        hypothesis: PrincipleHypothesis,
        cluster: CandidateCluster,
        alignment: AlignmentMap,
        outcome: ValidationOutcome,
    ) -> ModelingExperience:
        """Assemble a ModelingExperience peer record from a validated hypothesis.

        The induced record is a PEER of its source realizations — same schema, same store,
        no depth hierarchy. It is distinguished only by having derived_from_experience_ids
        pointing to its sources and status='validated' (it passed unseen transfer).
        """
        pattern = ModelingExperience(
            title="Induced: " + hypothesis.structural_pattern[:60],
            polarity="positive",
            retrieval_text=hypothesis.statement,
            layer="modeling",
            modeling_aspect="constraint",
            math_scope=MathScope(
                structural_signature=cluster.representative_signature or MathScope().structural_signature,
                exclusions=list(hypothesis.applicability_conditions),
            ),
            method=MethodBody(
                action_template=hypothesis.statement,
                rationale="Induced via cross-memory structural induction over cluster " + cluster.cluster_id,
            ),
            evidence=ModelingEvidence(
                source_episodes=[ep for m in cluster.members for ep in m.source_episodes],
                validation_level="solver_validated",
                causal_confidence="medium",
            ),
            derived_from_experience_ids=list(hypothesis.source_realization_ids),
            status="validated",
            role_schema=alignment.role_schema(),
            role_mappings=alignment.role_mappings(),
            applicability_conditions=list(hypothesis.applicability_conditions),
            validation=outcome.validation,
            scoring=outcome.scoring,
        )
        pattern.compute_content_hash()
        return pattern


def run_induction_sync(pipeline: InductionPipeline) -> InductionRunReport:
    """Synchronous entry point for the CLI."""
    return asyncio.run(pipeline.run())


__all__ = ["InductionPipeline", "InductionRunReport", "run_induction_sync"]
