"""M4 maintenance layer: traceable induction candidate bundles + offline
consequence and value assessment.

M4 connects experience accumulation to offline strategic induction via
world-model prediction:
1. Candidate bundling (:func:`build_induction_candidates`): scans C1–C6
   hints and cell statistics over real executed evidence, producing
   frozen, traceable :class:`InductionCandidateBundle` objects.
2. Value assessment (:meth:`ORHarness.assess_induction`): asks the world
   model to predict the consequence of inducting a bundle (candidate
   formation, verification cost, expected reuse benefit, generalization
   risk), returning an auditable :class:`InductionAssessment`.
3. Explicit choice + adoption (:meth:`ORHarness.accept_induction` /
   :meth:`reject_induction`): the outer agent explicitly accepts or
   rejects the recommendation. Acceptance invokes the existing
   ``induce`` machinery strictly on the chosen bundle's scope.
4. Consequence binding (:meth:`ORHarness.bind_induction_outcome`):
   compares the assessment's predictions against the actual induction
   outcome (entries created/updated, verification verdict, business
   result) and records the verdict on the assessment's feedback partition.

This is a MAINTENANCE-SCOPE facility — it does NOT modify the online M3
search tree, does not run background loops, and does not create a third
knowledge bank.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import ProblemProfile, group_key
from or_harness.core.storage import StorageError
from or_harness.strategy.triggers import InductionHint
from or_harness.world_model.prediction import ActionSpec
from or_harness.world_model.state import MAINTENANCE_TASK_ID


@dataclass
class InductionCandidateBundle:
    """A frozen, traceable bundle of real execution evidence supporting
    an induction or revision candidate.

    Carries the candidate kind (new claim vs revision of an existing
    entry), the target strategy/family/cell, the exact execution IDs,
    the frozen statistical and quality summaries, and the trigger reasons
    that flagged it. Never dynamically re-queries the bank — once
    constructed, its content is fixed."""

    bundle_id: str
    kind: str  # "new_claim" | "revision"
    strategy_id: str
    family: str
    cell_token: str
    group_key: str
    execution_ids: List[str]
    tasks: List[str]
    n_supporting: int
    trigger_reasons: List[str]
    # Frozen summary of the supporting evidence.
    mean_quality: Optional[float] = None
    mean_cost: Dict[str, float] = field(default_factory=dict)
    cost_measured: List[str] = field(default_factory=list)
    failure_rate: float = 0.0
    # Revision-specific: the existing entry being revised and its state
    # at the time the bundle was formed.
    target_entry_id: Optional[str] = None
    entry_before: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return f"cb_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "kind": self.kind,
            "strategy_id": self.strategy_id,
            "family": self.family,
            "cell_token": self.cell_token,
            "group_key": self.group_key,
            "execution_ids": list(self.execution_ids),
            "tasks": list(self.tasks),
            "n_supporting": int(self.n_supporting),
            "trigger_reasons": list(self.trigger_reasons),
            "mean_quality": self.mean_quality,
            "mean_cost": copy.deepcopy(self.mean_cost),
            "cost_measured": list(self.cost_measured),
            "failure_rate": float(self.failure_rate),
            "target_entry_id": self.target_entry_id,
            "entry_before": copy.deepcopy(self.entry_before),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InductionCandidateBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            kind=str(data.get("kind", "new_claim")),
            strategy_id=str(data.get("strategy_id", "")),
            family=str(data.get("family", "")),
            cell_token=str(data.get("cell_token", "")),
            group_key=str(data.get("group_key", "")),
            execution_ids=list(data.get("execution_ids") or []),
            tasks=list(data.get("tasks") or []),
            n_supporting=int(data.get("n_supporting", 0)),
            trigger_reasons=list(data.get("trigger_reasons") or []),
            mean_quality=data.get("mean_quality"),
            mean_cost=dict(data.get("mean_cost") or {}),
            cost_measured=list(data.get("cost_measured") or []),
            failure_rate=float(data.get("failure_rate", 0.0)),
            target_entry_id=data.get("target_entry_id"),
            entry_before=dict(data["entry_before"]) if data.get("entry_before") else None,
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class InductionAssessment:
    """The auditable record of an offline induction value assessment.

    Captures the bundle evaluated, the recommendation (induce_new /
    revise / defer / insufficient_evidence), the value decomposition,
    the predicted induction consequence, uncertainty, and budget."""

    assessment_id: str
    bundle_id: str
    decision_action_id: Optional[str]
    recommendation: str  # "induce_new" | "revise" | "defer" | "insufficient_evidence"
    target_strategy_id: str
    target_family: str
    # Value decomposition: net_value = reuse_benefit - induction_cost - generalization_risk
    expected_reuse_benefit: Optional[float] = None
    predicted_induction_cost: Optional[Dict[str, float]] = None
    predicted_generalization_risk: Optional[float] = None
    net_value: Optional[float] = None
    recommendation_basis: str = ""
    # Consequence prediction.
    candidate_formation_prob: float = 1.0
    predicted_quality_claim: Optional[float] = None
    predicted_cost_claim: Optional[Dict[str, float]] = None
    # Uncertainty & evidence gaps.
    confidence: Optional[float] = None
    evidence_gaps: List[str] = field(default_factory=list)
    unsupported_fields: Dict[str, str] = field(default_factory=dict)
    # Workload forecast context (when supplied by caller).
    workload_forecast: Optional[Dict[str, Any]] = None
    #: The evidence scope this assessment speaks about: the exact execution
    #: ids of its bundle. Carried so (a) acceptance can restrict the
    #: induction to that scope instead of re-deriving it, and (b) the
    #: delayed binding can verify which evidence it is judging against.
    execution_ids: List[str] = field(default_factory=list)
    # Audit trail.
    prediction_id: Optional[str] = None
    status: str = "ok"  # ok | truncated | fallback | not_configured
    assessment_cost: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return f"ia_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "bundle_id": self.bundle_id,
            "decision_action_id": self.decision_action_id,
            "recommendation": self.recommendation,
            "target_strategy_id": self.target_strategy_id,
            "target_family": self.target_family,
            "expected_reuse_benefit": self.expected_reuse_benefit,
            "predicted_induction_cost": copy.deepcopy(self.predicted_induction_cost),
            "predicted_generalization_risk": self.predicted_generalization_risk,
            "net_value": self.net_value,
            "recommendation_basis": self.recommendation_basis,
            "candidate_formation_prob": float(self.candidate_formation_prob),
            "predicted_quality_claim": self.predicted_quality_claim,
            "predicted_cost_claim": copy.deepcopy(self.predicted_cost_claim),
            "confidence": self.confidence,
            "evidence_gaps": list(self.evidence_gaps),
            "unsupported_fields": copy.deepcopy(self.unsupported_fields),
            "workload_forecast": copy.deepcopy(self.workload_forecast),
            "execution_ids": list(self.execution_ids),
            "prediction_id": self.prediction_id,
            "status": self.status,
            "assessment_cost": copy.deepcopy(self.assessment_cost),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InductionAssessment":
        return cls(
            assessment_id=str(data["assessment_id"]),
            bundle_id=str(data.get("bundle_id", "")),
            decision_action_id=data.get("decision_action_id"),
            recommendation=str(data.get("recommendation", "defer")),
            target_strategy_id=str(data.get("target_strategy_id", "")),
            target_family=str(data.get("target_family", "")),
            expected_reuse_benefit=data.get("expected_reuse_benefit"),
            predicted_induction_cost=dict(data["predicted_induction_cost"])
                if data.get("predicted_induction_cost") else None,
            predicted_generalization_risk=data.get("predicted_generalization_risk"),
            net_value=data.get("net_value"),
            recommendation_basis=str(data.get("recommendation_basis", "")),
            candidate_formation_prob=float(data.get("candidate_formation_prob", 1.0)),
            predicted_quality_claim=data.get("predicted_quality_claim"),
            predicted_cost_claim=dict(data["predicted_cost_claim"])
                if data.get("predicted_cost_claim") else None,
            confidence=data.get("confidence"),
            evidence_gaps=list(data.get("evidence_gaps") or []),
            unsupported_fields=dict(data.get("unsupported_fields") or {}),
            workload_forecast=dict(data["workload_forecast"])
                if data.get("workload_forecast") else None,
            execution_ids=list(data.get("execution_ids") or []),
            prediction_id=data.get("prediction_id"),
            status=str(data.get("status", "ok")),
            assessment_cost=dict(data["assessment_cost"])
                if data.get("assessment_cost") else None,
            created_at=float(data.get("created_at", time.time())),
        )


def build_induction_candidates(harness) -> List[InductionCandidateBundle]:
    """Scan the Experience Bank and Strategic Bank for induction candidates.

    Filters real executed evidence (attempt scope only, deduplicated by
    execution_id). Identifies:
    - NEW claims: cells with >=2 executions where no published entry covers
      the predicates.
    - REVISIONS: cells where existing entries have accumulated forward-check
      misses, performance drift (C3), or substantively changed evidence.

    Returns a list of frozen :class:`InductionCandidateBundle` objects.
    Empty when no candidate has sufficient supporting evidence."""
    bundles: List[InductionCandidateBundle] = []
    # Collect all executed attempt-scope records.
    records = [r for r in harness.bank.all()
               if r.source == "executed" and r.measurement_scope == "attempt"]
    if not records:
        return bundles

    # Group by (family, strategy_id, cell_token).
    by_cell: Dict[tuple, List[Any]] = {}
    for r in records:
        prof = r.profile_snapshot
        if prof is None:
            continue
        gkey = group_key(prof)
        # gkey format: "family=<name>|rc[..]|tc[..]|rx[..]"
        parts = gkey.split("|", 1)
        family = parts[0].removeprefix("family=") if "=" in parts[0] else parts[0]
        cell_token = parts[1] if len(parts) > 1 else ""
        key = (family, r.strategy_id, cell_token, gkey)
        by_cell.setdefault(key, []).append(r)

    # Check triggers against current stats.
    # Group existing entries by strategy.
    for (family, sid, cell_token, gkey), recs in sorted(by_cell.items()):
        # Deduplicate executions by execution_id.
        seen_ids = set()
        unique_recs = []
        for r in recs:
            if r.execution_id not in seen_ids:
                seen_ids.add(r.execution_id)
                unique_recs.append(r)
        if len(unique_recs) < 2:
            continue
        tasks = sorted({r.task_id for r in unique_recs})
        execution_ids = sorted(r.execution_id for r in unique_recs)
        cell = harness.stats.aggregate(gkey, sid, unique_recs)

        # Check if an entry already covers this.
        from or_harness.strategy.induction import evidence_predicates
        predicates = evidence_predicates(unique_recs, family=family)
        existing = harness.induction._find_existing(sid, predicates,
                                                    include_dormant=True)
        # Collect trigger reasons.
        trigger_reasons: List[str] = []
        if len(tasks) >= 2 and existing is None:
            trigger_reasons.append(
                f"sufficient independent evidence ({len(tasks)} tasks, "
                f"n={len(unique_recs)}) for new claim")
        elif existing is not None:
            # Check for substantive differences or misses.
            misses = (existing.prediction_track.consecutive_misses
                      if existing.prediction_track else 0)
            if misses >= 2:
                trigger_reasons.append(
                    f"existing entry {existing.entry_id} has {misses} "
                    "consecutive prediction misses")
            if abs(existing.expected_quality_hat - cell.mean_quality) > 0.05:
                trigger_reasons.append(
                    f"observed quality ({cell.mean_quality:.2f}) diverged from "
                    f"claim ({existing.expected_quality_hat:.2f})")
            if cell.n > existing.support_n:
                trigger_reasons.append(
                    f"new evidence available (n={cell.n} vs entry support_n="
                    f"{existing.support_n})")
        if not trigger_reasons:
            continue

        measured_dims = sorted(
            d for d, n in cell.n_measured.items() if n > 0)
        mean_cost = {d: float(getattr(cell.mean_cost, d, 0.0))
                     for d in measured_dims}

        bundle = InductionCandidateBundle(
            bundle_id=InductionCandidateBundle.new_id(),
            kind="revision" if existing is not None else "new_claim",
            strategy_id=sid,
            family=family,
            cell_token=cell_token,
            group_key=gkey,
            execution_ids=execution_ids,
            tasks=tasks,
            n_supporting=len(unique_recs),
            trigger_reasons=trigger_reasons,
            mean_quality=round(cell.mean_quality, 4),
            mean_cost={d: round(v, 4) for d, v in mean_cost.items()},
            cost_measured=measured_dims,
            failure_rate=round(cell.fail_rate, 4),
            target_entry_id=existing.entry_id if existing else None,
            entry_before=existing.to_dict() if existing else None,
        )
        bundles.append(bundle)

    return bundles
