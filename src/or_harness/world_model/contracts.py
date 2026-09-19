"""Unified world-model contracts: two prediction modules, one traceability shape.

This module is the SINGLE AUTHORITY for the versioned prediction contracts.
It adds no prediction service, no model call, and no new table: it defines
what a prediction IS, how it is validated, how it round-trips, and how
legacy payloads are read without being silently given new meaning.

Two prediction modules over one shared frame
--------------------------------------------

- :class:`StrategyOutcomePrediction` — *OR strategy consequence prediction*:
  from the current problem (P), solving context (X) and harness capability
  condition, what benefit G, resource cost c, risk/loss L and uncertainty
  would a CANDIDATE strategy produce? Its benefit carries an explicit
  metric, unit and baseline, because "benefit" is not one arbitrary 0-1
  score: a validated task completion, a qualified solution quality, valid
  progress and a correct infeasibility diagnosis are different currencies.
- :class:`CapabilityEvolutionPrediction` — *harness capability evolution
  prediction*: from the current capability EVIDENCE, real experience and a
  candidate learning operation u, how would future task performance change,
  at what learning cost, with what degradation risk and what verification
  conditions? It predicts OBSERVABLE consequences of H, never a latent
  vector and never a composite "H score".

Both share :class:`PredictionTrace` (protocol version, prediction type,
input snapshot/version reference, generation time, evidence basis,
unsupported fields with reasons, and the prediction call's own measured
cost when one exists). Derivable information is derived — a caller never
re-types the prediction type or the contract version.

Three separations this module enforces
--------------------------------------

1. **Prediction != fact != verified effect.** ``status`` distinguishes
   "contract built, no prediction service attached" (``contract_only``)
   from a real prediction (``valid``) and from an explicit refusal
   (``unsupported`` / ``invalid``). Adding knowledge entries, accumulating
   evidence, or a model asserting an improvement never confirms a
   capability gain; :class:`VerificationCondition` records what WOULD
   confirm it.
2. **H evidence != H.** :class:`HarnessCapabilityEvidence` holds the five
   interacting sources (M / W_OR / Pi / R / T) with an explicit evidence
   STATUS per source and no numeric score. The old ``harness_state``
   knowledge refs, experience counts and tool config are *evidence about*
   H, not measured H — renaming them would be a measurement claim nobody
   made.
3. **Attempt != strategy execution window.** A prediction's ``scope`` is
   ``"attempt"`` or ``"strategy_window"``, and only a window that really
   corresponds to recorded actions may be marked comparable
   (see :mod:`or_harness.world_model.execution_window`).

``B`` is deliberately NOT a predicted world-state object here: budget
declarations, execution limits and the real resource ledger stay in
:class:`~or_harness.world_model.budget.BudgetLedger`. Cost appears in these
contracts as the resource consequence of a candidate (predicted) or as the
measured spend of the prediction call itself — two separate carriers, never
summed into one number.

History is not a capability term: there is no ``E_hist`` component. H's five
sources are M (execution evidence / strategic knowledge and how usable it
actually is), W_OR (consequence prediction and reliability judgement for OR
solving), Pi (strategy generation, composition, comparison, selection,
switching, stopping), R (knowledge retrieval, applicability matching,
context adaptation — not the final strategy decision), and T (tool/solver
selection, composition, invocation, result handling).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import COST_DIMENSIONS, CostVector

#: The current contract version. Bumped when the shape changes; a stored
#: payload is always interpretable through the version it names.
CONTRACT_VERSION = "wm-contract/1"

#: The version label for payloads written BEFORE versioning existed. They
#: are read through the legacy view, never silently upgraded.
LEGACY_CONTRACT_VERSION = "legacy/unversioned"

#: Versions this build can read. An unknown version is reported as
#: unsupported — never guessed at.
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION,)

#: Prediction module kinds.
PREDICTION_KINDS = ("strategy_outcome", "capability_evolution")

#: Which prediction kinds this build has a real SERVICE for. A kind can have
#: a fully implemented CONTRACT and no service — that is the whole point of
#: the ``contract_only`` status, and the two must never be conflated:
#: configuring a provider makes a provider AVAILABLE, it does not make a
#: capability-evolution prediction service exist.
SERVICE_IMPLEMENTED_KINDS: Tuple[str, ...] = ("strategy_outcome",
                                              "capability_evolution")

#: Which direction of an :class:`ExpectedChange` counts as an IMPROVEMENT
#: for its metric. Fixed when the change is declared, never re-read after
#: the result was seen: a lower runtime is a ``decrease`` and an
#: improvement, a higher completion rate is an ``increase`` and an
#: improvement. ``either`` exists for a metric whose sign carries no
#: quality meaning.
BENEFICIAL_DIRECTIONS = ("increase", "decrease", "either")

#: How an :class:`ExpectedChange` value is expressed. ``absolute`` is the
#: metric's own unit, ``relative`` is a ratio/percentage of the baseline.
#: Recorded because adding a percentage to a unit count is meaningless.
CHANGE_VALUE_KINDS = ("absolute", "relative")

#: How a completed legacy ``OutcomePrediction`` status maps onto a contract
#: status. Only ``valid`` (a real prediction, produced and validated) yields
#: ``valid``: a contract may never claim to be a prediction merely because a
#: provider object is configured.
LEGACY_STATUS_TO_CONTRACT_STATUS: Dict[str, str] = {
    "valid": "valid",
    "not_configured": "contract_only",
    "unsupported_action": "unsupported",
    "invalid_output": "invalid",
    "provider_error": "invalid",
}

#: Lifecycle of a contract object.
#:
#: - ``draft``: being built, not yet validated.
#: - ``contract_only``: the contract is implemented but NO prediction
#:   service is attached (or none is deployed) — the honest "schema done,
#:   service not wired" state. It never carries fabricated numbers.
#: - ``valid``: a real prediction, produced and validated.
#: - ``unsupported``: the request is outside what this build's prediction
#:   path supports; nothing was predicted.
#: - ``invalid``: a prediction was attempted and failed validation.
CONTRACT_STATUSES = ("draft", "contract_only", "valid", "unsupported",
                     "invalid")

#: What a benefit number MEANS. Benefit is not one universal 0-1 score.
BENEFIT_KINDS = ("effective_completion", "solution_quality",
                 "valid_progress", "correct_infeasibility_diagnosis")

#: How a baseline was established.
BASELINE_KINDS = ("no_knowledge", "conditional_stats", "current_entry",
                  "current_solution", "declared", "unknown")

#: Where an uncertainty estimate came from. A model's self-reported
#: confidence is recorded as such and is NEVER a calibrated probability.
UNCERTAINTY_SOURCES = ("not_estimated", "framework_heuristic",
                       "measured", "model_self_report")

#: Evidence status of one capability source. Deliberately ordinal words,
#: never a number: an unreviewed count of knowledge refs is not a score.
CAPABILITY_EVIDENCE_STATUSES = ("no_evidence", "indirect_evidence",
                                "direct_evidence")

#: The five interacting capability sources of H = F(M, W_OR, Pi, R, T).
CAPABILITY_SOURCES = ("m", "w_or", "pi", "r", "t")

#: What a learning operation does (the offline candidate vocabulary).
LEARNING_OPERATION_TYPES = ("induce", "revise", "reverify", "retire")

#: Measurement scopes a prediction may declare.
PREDICTION_SCOPES = ("attempt", "strategy_window")

#: Legacy ``measurement_scope`` values with NO contract equivalent. Reading
#: one must report it as UNSUPPORTED: silently shrinking a whole-task
#: measurement to a single attempt would compare two different units while
#: claiming they are the same thing.
LEGACY_UNMAPPABLE_SCOPES: Dict[str, str] = {
    "task": (
        "a legacy 'task' scope covers the WHOLE task (every attempt plus its "
        "auxiliary work), which is neither one attempt nor one strategy "
        "execution window; this contract cannot express it"),
}

#: The two legacy contract generations this module knows how to identify.
PAYLOAD_VERSION_UNKNOWN_PREFIX = "unknown/"

#: Explicit mapping from the legacy experiment/prediction modes onto the
#: unified contract kinds. The legacy mode names are NOT renamed — they are
#: still the ablation switches of the experiment runner — but they are no
#: longer the primary vocabulary a caller must learn.
LEGACY_MODE_TO_KIND: Dict[str, Tuple[str, ...]] = {
    "x-b-only": ("strategy_outcome",),
    "h-x-b": ("strategy_outcome", "capability_evolution"),
    "h-x-b-value": ("strategy_outcome", "capability_evolution"),
}

#: Legacy field -> new-contract field, for the migration table in the docs
#: and for :func:`legacy_prediction_view`. Only MECHANICAL mappings appear
#: here; anything requiring a measurement nobody made is in
#: :data:`LEGACY_UNMAPPABLE`.
LEGACY_FIELD_MIGRATION: Dict[str, str] = {
    "action_spec": "candidate (CandidateRef)",
    "predicted.outcome_status": "candidate.expected_status",
    "predicted.feasible": "benefit.feasible",
    "predicted.quality": "benefit.value (kind=solution_quality)",
    "predicted.failure_prob": "risk.events[event=task_failure].probability",
    "predicted.cost": "cost.expected (CostVector)",
    "predicted.cost_measured": "cost.expected_measured",
    "predicted.state_changes": "benefit.progress_shape (X only, no values)",
    "predicted.knowledge_changes": "capability_evolution.expected_changes",
    "predicted.candidate_formation_prob": "capability_evolution.expected_changes",
    "predicted.expected_reuse_benefit": "capability_evolution.expected_changes",
    "predicted.generalization_risk": "capability_evolution.degradation_risk",
    "confidence": "uncertainty (source=model_self_report, uncalibrated)",
    "evidence_basis": "trace.evidence_basis",
    "unsupported_fields": "trace.unsupported_fields",
    "call_cost": "trace.call_cost (the model call's OWN spend)",
    "input_snapshot_id": "trace.input_snapshot_id",
    "created_at": "trace.created_at",
    "model_info": "trace.model_info",
}

#: Legacy content that has NO honest new-contract equivalent. Reading it
#: must not invent a measurement.
LEGACY_UNMAPPABLE: Dict[str, str] = {
    "harness_state.knowledge": (
        "knowledge REFERENCES and their counts are evidence ABOUT H, not a "
        "measured capability level or increment"),
    "harness_state.experience.task_execution_count": (
        "an experience count is evidence volume, not capability"),
    "harness_state.tool_config": (
        "tool AVAILABILITY is a precondition for T, not a measured T level"),
    "budget_state": (
        "budget declarations/consumption stay in BudgetLedger; B is not a "
        "predicted world-state object"),
    "state_changes (field-level detail)": (
        "the legacy fine-grained successor-state detail is read as an X "
        "progress SHAPE only; per-field values are not carried forward as "
        "predictions"),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _prob(value: Any) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def new_prediction_id(prefix: str = "cp") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# shared traceability
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRef:
    """One piece of evidence a prediction rests on (a reference, not a copy).

    ``ref_type`` is deliberately open-ended (execution / entry / snapshot /
    prediction / action / task / external) so a new evidence producer does
    not require a schema change; ``ref_id`` is what the reader resolves.
    """

    ref_type: str
    ref_id: str
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ref_type": self.ref_type, "ref_id": self.ref_id,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        if not isinstance(data, dict):
            raise ValueError("EvidenceRef must be a JSON object")
        ref_type = str(data.get("ref_type", "")).strip()
        ref_id = str(data.get("ref_id", "")).strip()
        if not ref_type or not ref_id:
            raise ValueError("EvidenceRef requires ref_type and ref_id")
        return cls(ref_type=ref_type, ref_id=ref_id,
                   note=(str(data["note"]) if data.get("note") else None))


@dataclass
class PredictionTrace:
    """The traceability block shared by both prediction modules.

    Everything a later reader needs to interpret a prediction WITHOUT
    re-deriving it: which version of the contract produced it, what it was
    conditioned on, when, on what evidence, what it refused to predict, and
    what the prediction call itself cost.
    """

    prediction_kind: str
    #: The frozen input snapshot this prediction was conditioned on. Empty
    #: string when the contract was built without one (a schema-level
    #: object, not a prediction).
    input_snapshot_id: str = ""
    #: Version/digest of the input, when the caller has one (e.g. the task
    #: digest recorded on the snapshot).
    input_version: Optional[str] = None
    #: Version of the prediction SERVICE/adapter, when one exists.
    prediction_version: Optional[str] = None
    #: Prompt/template version, when a model produced the payload.
    prompt_template_version: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    evidence_basis: List[EvidenceRef] = field(default_factory=list)
    #: Field -> why it was not predicted. An absent field is unknown, and
    #: saying WHY is part of the contract, not an afterthought.
    unsupported_fields: Dict[str, str] = field(default_factory=dict)
    #: The prediction CALL's own measured spend (provider usage), never the
    #: predicted cost of the target action.
    call_cost: Optional[CostVector] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    #: Whether this prediction corresponds to a real, completed scope and
    #: may therefore be scored. Defaults to False: an unproven scope is not
    #: comparable, and the reasons say why.
    comparable: bool = False
    not_comparable_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_kind": self.prediction_kind,
            "input_snapshot_id": self.input_snapshot_id,
            "input_version": self.input_version,
            "prediction_version": self.prediction_version,
            "prompt_template_version": self.prompt_template_version,
            "created_at": self.created_at,
            "evidence_basis": [e.to_dict() for e in self.evidence_basis],
            "unsupported_fields": dict(self.unsupported_fields),
            "call_cost": (self.call_cost.to_dict()
                          if self.call_cost is not None else None),
            "call_cost_measured": (
                sorted(self.call_cost.measured_dims())
                if self.call_cost is not None else None),
            "model_info": copy.deepcopy(self.model_info),
            "comparable": bool(self.comparable),
            "not_comparable_reasons": list(self.not_comparable_reasons),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTrace":
        raw_cost = data.get("call_cost")
        call_cost = (CostVector.from_dict(raw_cost)
                     if isinstance(raw_cost, dict) else None)
        if call_cost is not None:
            raw_measured = data.get("call_cost_measured")
            if isinstance(raw_measured, list):
                call_cost.measured = {str(d) for d in raw_measured
                                      if d in COST_DIMENSIONS}
        kind = str(data.get("prediction_kind", ""))
        if kind not in PREDICTION_KINDS:
            raise ValueError(
                f"trace.prediction_kind must be one of {PREDICTION_KINDS}")
        return cls(
            prediction_kind=kind,
            input_snapshot_id=str(data.get("input_snapshot_id") or ""),
            input_version=(str(data["input_version"])
                           if data.get("input_version") else None),
            prediction_version=(str(data["prediction_version"])
                                if data.get("prediction_version") else None),
            prompt_template_version=(
                str(data["prompt_template_version"])
                if data.get("prompt_template_version") else None),
            created_at=float(data.get("created_at", time.time())),
            evidence_basis=[EvidenceRef.from_dict(e)
                            for e in (data.get("evidence_basis") or [])],
            unsupported_fields={str(k): str(v) for k, v in
                                (data.get("unsupported_fields") or {}).items()},
            call_cost=call_cost,
            model_info=copy.deepcopy(dict(data.get("model_info") or {})),
            comparable=bool(data.get("comparable", False)),
            not_comparable_reasons=[str(r) for r in
                                    (data.get("not_comparable_reasons")
                                     or [])],
        )


# ---------------------------------------------------------------------------
# capability evidence (about H — not a measured H)
# ---------------------------------------------------------------------------


@dataclass
class PredictionServiceStatus:
    """The three DIFFERENT facts a caller keeps conflating.

    "A provider is configured", "this build implements a service for this
    prediction kind" and "a prediction actually completed" are three
    independent facts, and a contract may only be ``valid`` when all three
    hold. Reporting the first as the third is how an empty contract came to
    be labelled a forecast.
    """

    kind: str
    #: A provider object is attached to the instance (not the no-op
    #: :class:`~or_harness.world_model.provider.NotConfiguredProvider`).
    provider_configured: bool = False
    #: This build has a real service implementation for this kind. A kind
    #: can have a complete CONTRACT and no service at all.
    service_implemented: bool = False
    #: A prediction really completed and passed validation.
    prediction_completed: bool = False
    #: Version label of the attached provider (never credentials).
    provider_name: str = "not-attached"

    def __post_init__(self) -> None:
        if self.kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind {self.kind!r}")

    @property
    def service_available(self) -> bool:
        """A usable prediction SERVICE exists for this kind.

        Requires BOTH a configured provider and an implementation for the
        kind: a configured provider alone does not make an unimplemented
        service available.
        """
        return bool(self.provider_configured and self.service_implemented)

    @property
    def status(self) -> str:
        """The contract status this service state justifies.

        Only a completed prediction yields ``valid``. Everything else is
        ``contract_only``: the schema is implemented, no forecast was made.
        """
        if self.prediction_completed and self.service_available:
            return "valid"
        return "contract_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "provider_configured": bool(self.provider_configured),
            "service_implemented": bool(self.service_implemented),
            "service_available": self.service_available,
            "prediction_completed": bool(self.prediction_completed),
            "provider_name": self.provider_name,
            "status_justified": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionServiceStatus":
        data = data or {}
        return cls(
            kind=str(data.get("kind", "")),
            provider_configured=bool(data.get("provider_configured", False)),
            service_implemented=bool(data.get("service_implemented", False)),
            prediction_completed=bool(data.get("prediction_completed",
                                              False)),
            provider_name=str(data.get("provider_name", "not-attached")),
        )


def contract_status_from_legacy_status(legacy_status: str) -> str:
    """Map a completed legacy prediction's status onto a contract status.

    Only ``valid`` becomes ``valid``: that is the one legacy status that
    means "a prediction was produced and validated". Everything else is a
    refusal of one kind or another and must not be dressed up as a
    forecast.
    """
    return LEGACY_STATUS_TO_CONTRACT_STATUS.get(
        str(legacy_status), "invalid")


@dataclass
class CapabilitySourceEvidence:
    """Evidence about ONE capability source, with an explicit status.

    No number, on purpose. ``status`` is the honest ordinal claim:
    ``no_evidence`` (nothing observed), ``indirect_evidence`` (a proxy
    exists — a knowledge reference, a tool listing, an execution count),
    ``direct_evidence`` (a measured outcome that speaks to this source).
    ``detail`` carries the raw observations the caller already has; it is
    never converted into a score here.
    """

    source: str
    status: str = "no_evidence"
    evidence: List[EvidenceRef] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source not in CAPABILITY_SOURCES:
            raise ValueError(
                f"capability source must be one of {CAPABILITY_SOURCES}")
        if self.status not in CAPABILITY_EVIDENCE_STATUSES:
            raise ValueError("capability evidence status must be one of "
                             f"{CAPABILITY_EVIDENCE_STATUSES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "detail": copy.deepcopy(self.detail),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilitySourceEvidence":
        if not isinstance(data, dict):
            raise ValueError("CapabilitySourceEvidence must be a JSON object")
        return cls(
            source=str(data.get("source", "")),
            status=str(data.get("status", "no_evidence")),
            evidence=[EvidenceRef.from_dict(e)
                      for e in (data.get("evidence") or [])],
            detail=copy.deepcopy(dict(data.get("detail") or {})),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


@dataclass
class HarnessCapabilityEvidence:
    """Observable evidence about the harness's latent capability state H.

    ``H = F(M, W_OR, Pi, R, T)`` — five INTERACTING sources, deliberately
    not a fixed five-dimensional score and not a sum. This object carries
    what was OBSERVED about each source plus the honest status of that
    evidence. It never produces a composite number: ``composite_score``
    does not exist as a field, and :func:`validate_capability_evidence`
    rejects any payload that tries to smuggle one in.

    The legacy ``harness_state`` (knowledge references, experience counts,
    tool configuration) maps onto these sources as ``indirect_evidence``
    only — see :func:`capability_evidence_from_legacy_harness_state`.
    """

    as_of: float = field(default_factory=time.time)
    #: Version/digest of the capability snapshot this evidence describes.
    version: Optional[str] = None
    sources: Dict[str, CapabilitySourceEvidence] = field(default_factory=dict)
    #: Anything observed that does not belong to a single source.
    notes: List[str] = field(default_factory=list)
    #: Explicitly recorded so a reader never has to guess whether a score
    #: exists: it does not, by design.
    score_scheme: str = "no_composite_score"

    def __post_init__(self) -> None:
        for key in list(self.sources):
            if key not in CAPABILITY_SOURCES:
                raise ValueError(
                    f"capability source {key!r} not in "
                    f"{CAPABILITY_SOURCES}")

    def source(self, name: str) -> CapabilitySourceEvidence:
        """The evidence record for one source (created empty on demand)."""
        if name not in CAPABILITY_SOURCES:
            raise ValueError(f"unknown capability source {name!r}")
        if name not in self.sources:
            self.sources[name] = CapabilitySourceEvidence(source=name)
        return self.sources[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "version": self.version,
            "score_scheme": self.score_scheme,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessCapabilityEvidence":
        if not isinstance(data, dict):
            raise ValueError("HarnessCapabilityEvidence must be a JSON object")
        raw = data.get("sources") or {}
        if not isinstance(raw, dict):
            raise ValueError("capability evidence sources must be an object")
        return cls(
            as_of=float(data.get("as_of", time.time())),
            version=(str(data["version"]) if data.get("version") else None),
            sources={str(k): CapabilitySourceEvidence.from_dict(v)
                     for k, v in raw.items()},
            notes=[str(n) for n in (data.get("notes") or [])],
            score_scheme=str(data.get("score_scheme", "no_composite_score")),
        )


def validate_capability_evidence(evidence: HarnessCapabilityEvidence
                                 ) -> List[str]:
    """Structural validation, including the no-composite-score rule.

    A payload that carries ``composite_score`` / ``h_score`` / ``capability``
    as a number is rejected: this build has no measured H, and inventing one
    from counts would be exactly the mistake the contract exists to prevent.
    """
    problems: List[str] = []
    if evidence.score_scheme != "no_composite_score":
        problems.append(
            f"score_scheme {evidence.score_scheme!r} is not supported: this "
            "build defines no composite capability score")
    for name in CAPABILITY_SOURCES:
        item = evidence.sources.get(name)
        if item is None:
            continue
        if item.status == "direct_evidence" and not item.evidence:
            problems.append(
                f"source {name!r} claims direct_evidence but carries no "
                "evidence reference")
    return problems


def capability_evidence_from_legacy_harness_state(
        harness_state: Dict[str, Any], *,
        coverage: Optional[Dict[str, Any]] = None,
        as_of: Optional[float] = None) -> HarnessCapabilityEvidence:
    """Read a legacy ``harness_state`` (+coverage) as capability EVIDENCE.

    The mapping is deliberately conservative and label-preserving:

    - knowledge references (verified / legacy / unverified counts, and the
      coverage cell statistics) -> ``M`` as ``indirect_evidence``: how much
      knowledge is REFERENCED, never how capable the harness is;
    - available solver families / executor limits -> ``T`` as
      ``indirect_evidence``: availability is a precondition, not a level;
    - recent execution ids -> ``M`` evidence volume, still indirect.

    ``W_OR``, ``Pi`` and ``R`` are left at ``no_evidence``: nothing in the
    legacy state observed consequence-prediction reliability, strategy
    selection quality, or retrieval adaptation. Filling them with a count
    would be the fabrication this function exists to avoid.
    """
    state = dict(harness_state or {})
    knowledge = dict(state.get("knowledge") or {})
    experience = dict(state.get("experience") or {})
    tools = dict(state.get("tool_config") or {})
    coverage = dict(coverage or {})
    out = HarnessCapabilityEvidence(as_of=as_of if as_of is not None
                                    else time.time())
    out.notes.append(
        "derived from a legacy harness_state: every item is evidence ABOUT "
        "H, not a measured capability level")

    m = out.source("m")
    layers = {k: len(v or []) for k, v in knowledge.items()}
    cells = coverage.get("cell_statistics") or {}
    if layers or experience or cells:
        m.status = "indirect_evidence"
        m.detail = {
            "knowledge_ref_counts": layers,
            "experience": {
                "task_execution_count": experience.get("task_execution_count"),
                "total_executions": experience.get("total_executions"),
            },
            "cells_with_evidence": len(cells),
        }
        m.notes.append(
            "knowledge REFERENCES and experience VOLUME: usable knowledge "
            "is not the same as referenced knowledge")
        for layer, count in sorted(layers.items()):
            if count:
                m.evidence.append(EvidenceRef(
                    ref_type="knowledge_layer", ref_id=layer,
                    note=f"{count} reference(s)"))

    t = out.source("t")
    families = tools.get("available_solver_families")
    if families:
        t.status = "indirect_evidence"
        t.detail = {"available_solver_families": copy.deepcopy(families),
                    "executor_timeout_seconds":
                        tools.get("executor_timeout_seconds")}
        t.notes.append(
            "tool AVAILABILITY and configuration: a precondition for using "
            "tools, not a measured ability to choose or combine them")

    for name in ("w_or", "pi", "r"):
        out.source(name).notes.append(
            "no observation of this source exists in the legacy state: left "
            "at no_evidence rather than inferred from unrelated counts")
    return out


# ---------------------------------------------------------------------------
# candidate and scope
# ---------------------------------------------------------------------------


@dataclass
class ExperienceScope:
    """The real experience a prediction's evidence rests on.

    Counts INDEPENDENT TASKS alongside executions, because repeating one
    task is repetition, not reproduction — the same rule the induction gate
    uses.
    """

    execution_ids: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    family: Optional[str] = None
    cell_token: Optional[str] = None
    note: Optional[str] = None

    @property
    def distinct_tasks(self) -> int:
        return len({t for t in self.task_ids if t})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_ids": list(self.execution_ids),
            "task_ids": list(self.task_ids),
            "distinct_tasks": self.distinct_tasks,
            "family": self.family,
            "cell_token": self.cell_token,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceScope":
        data = data or {}
        return cls(
            execution_ids=[str(e) for e in (data.get("execution_ids") or [])],
            task_ids=[str(t) for t in (data.get("task_ids") or [])],
            family=(str(data["family"]) if data.get("family") else None),
            cell_token=(str(data["cell_token"])
                        if data.get("cell_token") else None),
            note=(str(data["note"]) if data.get("note") else None),
        )


@dataclass
class TaskTargeting:
    """Which tasks a capability-evolution prediction speaks about."""

    description: str = ""
    family: Optional[str] = None
    cell_token: Optional[str] = None
    predicates: Dict[str, Any] = field(default_factory=dict)
    task_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "family": self.family,
            "cell_token": self.cell_token,
            "predicates": copy.deepcopy(self.predicates),
            "task_ids": list(self.task_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskTargeting":
        data = data or {}
        return cls(
            description=str(data.get("description", "")),
            family=(str(data["family"]) if data.get("family") else None),
            cell_token=(str(data["cell_token"])
                        if data.get("cell_token") else None),
            predicates=copy.deepcopy(dict(data.get("predicates") or {})),
            task_ids=[str(t) for t in (data.get("task_ids") or [])],
        )


@dataclass
class BaselineStatement:
    """What the predicted change is measured AGAINST.

    A predicted improvement with no baseline is not a prediction — it is an
    advertisement. ``kind`` names how the baseline was established and
    ``ref`` points at it when one exists.
    """

    kind: str = "unknown"
    value: Optional[float] = None
    ref: Optional[EvidenceRef] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in BASELINE_KINDS:
            raise ValueError(f"baseline kind must be one of {BASELINE_KINDS}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "ref": (self.ref.to_dict() if self.ref is not None else None),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineStatement":
        data = data or {}
        raw_ref = data.get("ref")
        return cls(
            kind=str(data.get("kind", "unknown")),
            value=(float(data["value"])
                   if data.get("value") is not None else None),
            ref=(EvidenceRef.from_dict(raw_ref)
                 if isinstance(raw_ref, dict) else None),
            note=(str(data["note"]) if data.get("note") else None),
        )


@dataclass
class CandidateRef:
    """The candidate a strategy-outcome prediction is about.

    Carries what a caller can actually know BEFORE execution: the strategy
    method/identifier, the configuration that matters, preconditions, the
    expected execution scope, and the stop conditions. Step descriptions
    may be coarse — this is not a code generator, a mathematical model, or
    an action sequence.

    ``scope`` separates ONE execution attempt from a STRATEGY EXECUTION
    WINDOW (modeling + solve + repair + verify). The two are never silently
    interchanged; see :mod:`or_harness.world_model.execution_window`.
    """

    action_type: str
    strategy_id: Optional[str] = None
    solver: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_scope: List[str] = field(default_factory=list)
    stop_conditions: List[str] = field(default_factory=list)
    task_id: str = ""
    episode_id: Optional[str] = None
    scope: str = "attempt"
    #: Id of the recorded execution window this candidate maps to, when one
    #: exists. Without it a window-scope prediction cannot be comparable.
    window_id: Optional[str] = None
    #: How ``scope`` was established: ``declared`` (the caller said so),
    #: ``legacy_attempt`` (the legacy spec's ``measurement_scope`` was
    #: ``attempt``) or ``default``. Recorded because "one attempt" is a
    #: narrower claim than "the whole task", and a reader must be able to
    #: tell which one was actually made.
    scope_basis: str = "default"

    def __post_init__(self) -> None:
        if self.scope not in PREDICTION_SCOPES:
            raise ValueError(f"candidate scope must be one of "
                             f"{PREDICTION_SCOPES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "strategy_id": self.strategy_id,
            "solver": self.solver,
            "config": copy.deepcopy(self.config),
            "preconditions": list(self.preconditions),
            "expected_scope": list(self.expected_scope),
            "stop_conditions": list(self.stop_conditions),
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "scope": self.scope,
            "scope_basis": self.scope_basis,
            "window_id": self.window_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateRef":
        if not isinstance(data, dict) or not data.get("action_type"):
            raise ValueError("CandidateRef.action_type is required")
        return cls(
            action_type=str(data["action_type"]),
            strategy_id=(str(data["strategy_id"])
                         if data.get("strategy_id") else None),
            solver=(str(data["solver"]) if data.get("solver") else None),
            config=copy.deepcopy(dict(data.get("config") or {})),
            preconditions=[str(p) for p in (data.get("preconditions") or [])],
            expected_scope=[str(s) for s in (data.get("expected_scope")
                                             or [])],
            stop_conditions=[str(s) for s in (data.get("stop_conditions")
                                              or [])],
            task_id=str(data.get("task_id", "")),
            episode_id=(str(data["episode_id"])
                        if data.get("episode_id") else None),
            scope=str(data.get("scope", "attempt")),
            scope_basis=str(data.get("scope_basis", "default")),
            window_id=(str(data["window_id"]) if data.get("window_id")
                       else None),
        )

    @classmethod
    def from_action_spec(cls, spec, *, scope: Optional[str] = None
                         ) -> "CandidateRef":
        """Build from the legacy :class:`ActionSpec` (mechanical mapping).

        Two things this mapping refuses to do silently:

        - **Lose execution-affecting configuration.** The legacy spec's
          ``params`` (a time limit, a MIP gap target, a seed, a solver
          family, ...) and its ``budget_hint`` are carried through
          VERBATIM. Dropping them would produce a contract that describes a
          *different* candidate than the one that was actually proposed — a
          ``time_limit=60, mip_gap=0.01, seed=42`` run is not the same
          candidate as an unbounded one.
        - **Shrink a measurement scope.** The legacy ``measurement_scope``
          only maps mechanically when it is ``attempt`` (or
          ``strategy_window``). A legacy ``"task"`` scope covers the whole
          task — every attempt plus auxiliary work — which is neither of
          this contract's scopes; passing it raises rather than quietly
          relabelling a task-wide measurement as one solve attempt.

        Pass ``scope=`` explicitly to declare the scope yourself (that is
        the caller taking responsibility for the narrowing).
        """
        params = copy.deepcopy(dict(getattr(spec, "params", None) or {}))
        budget_hint = getattr(spec, "budget_hint", None)
        if budget_hint:
            # The legacy budget hint is an execution limit, not decoration:
            # keep it under its own key so it stays distinguishable from
            # the strategy's own parameters.
            params.setdefault("budget_hint", copy.deepcopy(dict(budget_hint)))
        declared = getattr(spec, "measurement_scope", "attempt")
        declared = str(declared or "attempt")
        if scope is not None:
            resolved, basis = str(scope), "declared"
        elif declared in PREDICTION_SCOPES:
            resolved = declared
            basis = "legacy_attempt" if declared == "attempt" else "legacy"
        else:
            raise ValueError(
                f"legacy measurement_scope {declared!r} has no contract "
                f"equivalent: {LEGACY_UNMAPPABLE_SCOPES.get(declared, 'it is ' + 'not one of ' + str(PREDICTION_SCOPES))} "
                f"Pass scope=... explicitly to declare "
                f"{'/'.join(PREDICTION_SCOPES)} instead of silently "
                "shrinking it to one attempt")
        return cls(
            action_type=str(getattr(spec, "action_type", "")),
            strategy_id=getattr(spec, "strategy_id", None),
            solver=getattr(spec, "solver", None),
            config=params,
            task_id=str(getattr(spec, "task_id", "")),
            episode_id=getattr(spec, "episode_id", None),
            scope=resolved,
            scope_basis=basis,
        )


# ---------------------------------------------------------------------------
# benefit / cost / risk / uncertainty
# ---------------------------------------------------------------------------


@dataclass
class BenefitEstimate:
    """The predicted GAIN, with its metric, unit and baseline.

    ``kind`` says what currency the number is in. ``value`` of ``None``
    means unknown — never a placeholder zero, which would read as "no
    gain" when the truth is "not predicted".
    """

    kind: str
    metric: str
    unit: str = ""
    value: Optional[float] = None
    interval: Optional[Tuple[float, float]] = None
    baseline: Optional[BaselineStatement] = None
    feasible: Optional[bool] = None
    progress_shape: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in BENEFIT_KINDS:
            raise ValueError(f"benefit kind must be one of {BENEFIT_KINDS}")
        if not self.metric:
            raise ValueError("benefit metric is required (what is measured)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "unit": self.unit,
            "value": self.value,
            "interval": (list(self.interval)
                         if self.interval is not None else None),
            "baseline": (self.baseline.to_dict()
                         if self.baseline is not None else None),
            "feasible": self.feasible,
            "progress_shape": copy.deepcopy(self.progress_shape),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenefitEstimate":
        if not isinstance(data, dict):
            raise ValueError("BenefitEstimate must be a JSON object")
        raw_interval = data.get("interval")
        raw_baseline = data.get("baseline")
        return cls(
            kind=str(data.get("kind", "solution_quality")),
            metric=str(data.get("metric", "")),
            unit=str(data.get("unit", "")),
            value=(float(data["value"])
                   if data.get("value") is not None else None),
            interval=((float(raw_interval[0]), float(raw_interval[1]))
                      if isinstance(raw_interval, (list, tuple))
                      and len(raw_interval) == 2 else None),
            baseline=(BaselineStatement.from_dict(raw_baseline)
                      if isinstance(raw_baseline, dict) else None),
            feasible=data.get("feasible"),
            progress_shape=copy.deepcopy(data.get("progress_shape")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


@dataclass
class ExpectedCost:
    """The predicted resource cost of the candidate (CostVector + mask).

    Kept SEPARATE from :attr:`PredictionTrace.call_cost` — the prediction
    call's own real spend. Conflating the two would either hide the
    prediction's cost or bill the candidate for a call it never made.
    """

    expected: Optional[CostVector] = None
    measured: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected": (self.expected.to_dict()
                         if self.expected is not None else None),
            "expected_measured": list(self.measured),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedCost":
        data = data or {}
        raw = data.get("expected")
        vector = (CostVector.from_dict(raw) if isinstance(raw, dict)
                  else None)
        measured = [str(d) for d in (data.get("expected_measured") or [])
                    if d in COST_DIMENSIONS]
        if vector is not None:
            vector.measured = set(measured)
        return cls(expected=vector, measured=measured,
                   notes=[str(n) for n in (data.get("notes") or [])])


@dataclass
class RiskEvent:
    """One risk EVENT with its loss meaning, kept separate from cost.

    ``probability`` and ``severity`` are ``None`` when no basis exists —
    neither is invented. Rework cost is NOT modelled here: it belongs to the
    cost vector, so a decision layer cannot double-count it.
    """

    event: str
    probability: Optional[float] = None
    severity: Optional[float] = None
    severity_unit: str = ""
    basis: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "probability": self.probability,
            "severity": self.severity,
            "severity_unit": self.severity_unit,
            "basis": self.basis,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskEvent":
        if not isinstance(data, dict) or not data.get("event"):
            raise ValueError("RiskEvent.event is required")
        return cls(
            event=str(data["event"]),
            probability=(float(data["probability"])
                         if data.get("probability") is not None else None),
            severity=(float(data["severity"])
                      if data.get("severity") is not None else None),
            severity_unit=str(data.get("severity_unit", "")),
            basis=(str(data["basis"]) if data.get("basis") else None),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


@dataclass
class RiskStatement:
    """The predicted risk/loss of the candidate, as named events."""

    events: List[RiskEvent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events],
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskStatement":
        data = data or {}
        return cls(events=[RiskEvent.from_dict(e)
                           for e in (data.get("events") or [])],
                   notes=[str(n) for n in (data.get("notes") or [])])


@dataclass
class UncertaintyStatement:
    """Uncertainty, with execution randomness separated from evidence gaps.

    Two optional components, because they are different things and a caller
    need not produce both:

    - ``execution_randomness``: variance inherent to running the candidate
      (solver heuristics, timing).
    - ``knowledge_gap``: how little evidence the prediction rests on.

    ``source`` records where the numbers came from. A model's self-reported
    confidence is ``model_self_report`` and is explicitly NOT a calibrated
    probability; ``missing`` says which component could not be estimated and
    why, so an absent number is never read as "no uncertainty".
    """

    source: str = "not_estimated"
    execution_randomness: Optional[float] = None
    knowledge_gap: Optional[float] = None
    basis: List[str] = field(default_factory=list)
    missing: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source not in UNCERTAINTY_SOURCES:
            raise ValueError(
                f"uncertainty source must be one of {UNCERTAINTY_SOURCES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "execution_randomness": self.execution_randomness,
            "knowledge_gap": self.knowledge_gap,
            "basis": list(self.basis),
            "missing": dict(self.missing),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyStatement":
        data = data or {}
        return cls(
            source=str(data.get("source", "not_estimated")),
            execution_randomness=(
                float(data["execution_randomness"])
                if data.get("execution_randomness") is not None else None),
            knowledge_gap=(float(data["knowledge_gap"])
                           if data.get("knowledge_gap") is not None
                           else None),
            basis=[str(b) for b in (data.get("basis") or [])],
            missing={str(k): str(v) for k, v in
                     (data.get("missing") or {}).items()},
            notes=[str(n) for n in (data.get("notes") or [])],
        )


# ---------------------------------------------------------------------------
# capability evolution: learning operation, change, verification
# ---------------------------------------------------------------------------


@dataclass
class LearningOperation:
    """The candidate learning operation ``u`` an offline prediction is about.

    The vocabulary is the EXISTING offline capability — the ``induce``
    action's business outcomes (create / revise / reverify / retire). It
    deliberately does not include automatically rewriting the planner,
    upgrading a solver, or fine-tuning a model: those are not operations
    this build can carry out, and predicting their consequences would be
    predicting a service that does not exist.
    """

    operation_type: str
    strategy_id: Optional[str] = None
    description: str = ""
    scope: Optional[ExperienceScope] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation_type not in LEARNING_OPERATION_TYPES:
            raise ValueError(
                f"operation_type must be one of {LEARNING_OPERATION_TYPES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "strategy_id": self.strategy_id,
            "description": self.description,
            "scope": (self.scope.to_dict() if self.scope is not None
                      else None),
            "config": copy.deepcopy(self.config),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningOperation":
        if not isinstance(data, dict) or not data.get("operation_type"):
            raise ValueError("LearningOperation.operation_type is required")
        raw_scope = data.get("scope")
        return cls(
            operation_type=str(data["operation_type"]),
            strategy_id=(str(data["strategy_id"])
                         if data.get("strategy_id") else None),
            description=str(data.get("description", "")),
            scope=(ExperienceScope.from_dict(raw_scope)
                   if isinstance(raw_scope, dict) else None),
            config=copy.deepcopy(dict(data.get("config") or {})),
        )


@dataclass
class ExpectedChange:
    """One predicted future-performance change, with its baseline.

    ``direction`` is ``increase`` / ``decrease`` / ``unchanged`` /
    ``unknown``. ``value`` is optional and, when present, is in the metric's
    own unit — the contract never forces every change into a 0-1 score.
    """

    metric: str
    direction: str = "unknown"
    value: Optional[float] = None
    interval: Optional[Tuple[float, float]] = None
    unit: str = ""
    baseline: Optional[BaselineStatement] = None
    #: Which direction is the IMPROVEMENT for this metric (see
    #: :data:`BENEFICIAL_DIRECTIONS`). Without it a negative value is
    #: ambiguous: a shorter runtime is good, a lower completion rate is
    #: not.
    beneficial_direction: str = "increase"
    #: ``absolute`` or ``relative`` (see :data:`CHANGE_VALUE_KINDS`).
    value_kind: str = "absolute"
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("increase", "decrease", "unchanged",
                                  "unknown"):
            raise ValueError("direction must be increase/decrease/unchanged/"
                             "unknown")
        if self.beneficial_direction not in BENEFICIAL_DIRECTIONS:
            raise ValueError(f"beneficial_direction must be one of "
                             f"{BENEFICIAL_DIRECTIONS}")
        if self.value_kind not in CHANGE_VALUE_KINDS:
            raise ValueError(f"value_kind must be one of "
                             f"{CHANGE_VALUE_KINDS}")

    @property
    def is_improvement(self) -> Optional[bool]:
        """Whether the predicted change is an improvement, when known.

        ``None`` when the direction is unknown or the metric's sign
        carries no quality meaning: an unstated direction is not
        silently read as good news.
        """
        if self.direction == "unknown" or self.beneficial_direction == \
                "either":
            return None
        return self.direction == self.beneficial_direction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "value": self.value,
            "interval": (list(self.interval)
                         if self.interval is not None else None),
            "unit": self.unit,
            "baseline": (self.baseline.to_dict()
                         if self.baseline is not None else None),
            "beneficial_direction": self.beneficial_direction,
            "value_kind": self.value_kind,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedChange":
        if not isinstance(data, dict) or not data.get("metric"):
            raise ValueError("ExpectedChange.metric is required")
        raw_interval = data.get("interval")
        raw_baseline = data.get("baseline")
        return cls(
            metric=str(data["metric"]),
            direction=str(data.get("direction", "unknown")),
            value=(float(data["value"])
                   if data.get("value") is not None else None),
            interval=((float(raw_interval[0]), float(raw_interval[1]))
                      if isinstance(raw_interval, (list, tuple))
                      and len(raw_interval) == 2 else None),
            unit=str(data.get("unit", "")),
            baseline=(BaselineStatement.from_dict(raw_baseline)
                      if isinstance(raw_baseline, dict) else None),
            beneficial_direction=str(data.get("beneficial_direction",
                                              "increase")),
            value_kind=str(data.get("value_kind", "absolute")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


@dataclass
class VerificationCondition:
    """What would actually CONFIRM the predicted capability change.

    Three separate things are recorded, because they are routinely confused:

    - ``prediction_made``: the prediction exists (this object);
    - ``fact_bound``: the real evidence landed and can be bound to it;
    - ``effect_verified``: the predicted improvement was observed.

    Only the third supports a claim that the harness got stronger. Adding
    knowledge entries, accumulating evidence, or a model asserting an
    improvement is none of the three.
    """

    condition: str
    evaluable: bool = False
    check_basis: Optional[str] = None
    prediction_made: bool = True
    fact_bound: bool = False
    effect_verified: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "evaluable": bool(self.evaluable),
            "check_basis": self.check_basis,
            "prediction_made": bool(self.prediction_made),
            "fact_bound": bool(self.fact_bound),
            "effect_verified": bool(self.effect_verified),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationCondition":
        if not isinstance(data, dict) or not data.get("condition"):
            raise ValueError("VerificationCondition.condition is required")
        return cls(
            condition=str(data["condition"]),
            evaluable=bool(data.get("evaluable", False)),
            check_basis=(str(data["check_basis"])
                         if data.get("check_basis") else None),
            prediction_made=bool(data.get("prediction_made", True)),
            fact_bound=bool(data.get("fact_bound", False)),
            effect_verified=bool(data.get("effect_verified", False)),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


# ---------------------------------------------------------------------------
# the two prediction contracts
# ---------------------------------------------------------------------------


@dataclass
class StrategyOutcomePrediction:
    """Contract 1: what a candidate strategy's execution would produce.

    Answers, for one candidate under the current P/X and harness condition:
    benefit G (with metric, unit and baseline), resource cost c (CostVector,
    measured mask preserved), risk/loss L (named events, separate from cost)
    and uncertainty (execution randomness vs evidence gap). ``status`` of
    ``contract_only`` means the contract exists but no prediction service
    produced values — the honest state of this build.
    """

    candidate: CandidateRef
    prediction_id: str = field(default_factory=lambda: new_prediction_id("sp"))
    prediction_type: str = "strategy_outcome"
    contract_version: str = CONTRACT_VERSION
    status: str = "draft"
    benefit: Optional[BenefitEstimate] = None
    cost: Optional[ExpectedCost] = None
    risk: Optional[RiskStatement] = None
    uncertainty: Optional[UncertaintyStatement] = None
    trace: Optional[PredictionTrace] = None
    #: Whether a prediction SERVICE is deployed for this kind in this build.
    #: False is a first-class state: the contract is implemented, the
    #: service is not attached. It requires BOTH a configured provider and
    #: an implementation for the kind.
    service_available: bool = False
    #: A provider object is attached (a weaker fact than
    #: ``service_available``: a configured provider does not by itself make
    #: any prediction happen).
    provider_configured: bool = False
    #: This build implements a service for this kind at all.
    service_implemented: bool = False
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.prediction_type != "strategy_outcome":
            raise ValueError("prediction_type must be 'strategy_outcome'")
        if self.trace is None:
            self.trace = PredictionTrace(
                prediction_kind="strategy_outcome",
                input_snapshot_id="")
        elif self.trace.prediction_kind != "strategy_outcome":
            raise ValueError("trace.prediction_kind must be "
                             "'strategy_outcome'")

    @property
    def scope(self) -> str:
        return self.candidate.scope

    @property
    def prediction_made(self) -> bool:
        """Whether a real prediction was produced (not just a schema).

        Deliberately narrower than ``status == 'valid'``: this is the
        single question "did a model actually produce and pass a
        prediction here?".
        """
        return self.status == "valid"

    @property
    def has_predicted_content(self) -> bool:
        """Whether any predicted quantity is present at all.

        An all-empty contract is a schema object, never a forecast — the
        check that stops "no prediction was made" from being labelled
        ``valid``.
        """
        return any(value is not None for value in
                   (self.benefit, self.cost, self.risk, self.uncertainty))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type,
            "status": self.status,
            "prediction_made": self.prediction_made,
            "has_predicted_content": self.has_predicted_content,
            "service_available": bool(self.service_available),
            "provider_configured": bool(self.provider_configured),
            "service_implemented": bool(self.service_implemented),
            "scope": self.candidate.scope,
            "candidate": self.candidate.to_dict(),
            "benefit": (self.benefit.to_dict()
                        if self.benefit is not None else None),
            "cost": self.cost.to_dict() if self.cost is not None else None,
            "risk": self.risk.to_dict() if self.risk is not None else None,
            "uncertainty": (self.uncertainty.to_dict()
                            if self.uncertainty is not None else None),
            "trace": self.trace.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyOutcomePrediction":
        if not isinstance(data, dict):
            raise ValueError("StrategyOutcomePrediction must be a JSON object")
        version = str(data.get("contract_version", ""))
        if version not in SUPPORTED_CONTRACT_VERSIONS:
            raise UnsupportedContractVersion(version)
        if not data.get("prediction_id"):
            raise ValueError("prediction_id is required")
        status = str(data.get("status", "draft"))
        if status not in CONTRACT_STATUSES:
            raise ValueError(f"status must be one of {CONTRACT_STATUSES}")
        raw_benefit = data.get("benefit")
        raw_cost = data.get("cost")
        raw_risk = data.get("risk")
        raw_unc = data.get("uncertainty")
        raw_trace = data.get("trace")
        return cls(
            candidate=CandidateRef.from_dict(data.get("candidate") or {}),
            prediction_id=str(data["prediction_id"]),
            prediction_type=str(data.get("prediction_type",
                                         "strategy_outcome")),
            contract_version=version,
            status=status,
            benefit=(BenefitEstimate.from_dict(raw_benefit)
                     if isinstance(raw_benefit, dict) else None),
            cost=(ExpectedCost.from_dict(raw_cost)
                  if isinstance(raw_cost, dict) else None),
            risk=(RiskStatement.from_dict(raw_risk)
                  if isinstance(raw_risk, dict) else None),
            uncertainty=(UncertaintyStatement.from_dict(raw_unc)
                         if isinstance(raw_unc, dict) else None),
            trace=PredictionTrace.from_dict(raw_trace or {}),
            service_available=bool(data.get("service_available", False)),
            provider_configured=bool(data.get("provider_configured", False)),
            service_implemented=bool(data.get("service_implemented", False)),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


@dataclass
class CapabilityEvolutionPrediction:
    """Contract 2: how a candidate learning operation would change future
    task performance.

    H is inspected through these OBSERVABLE consequences — current
    capability evidence, the candidate operation u, the experience scope it
    uses, the task types it is evaluated on, the baseline and horizon, the
    predicted performance change, the learning cost, the degradation risk,
    and the conditions under which the prediction would be confirmed. No
    latent vector, no composite score.

    ``status`` of ``contract_only`` means exactly what it says: the
    contract is implemented, the prediction service is not attached. It is
    NOT a claim that a capability prediction was made.
    """

    current_evidence: HarnessCapabilityEvidence
    candidate_operation: LearningOperation
    prediction_id: str = field(default_factory=lambda: new_prediction_id("hp"))
    prediction_type: str = "capability_evolution"
    contract_version: str = CONTRACT_VERSION
    status: str = "draft"
    experience_scope: Optional[ExperienceScope] = None
    task_targeting: Optional[TaskTargeting] = None
    baseline: Optional[BaselineStatement] = None
    #: The prediction horizon: what time/task window the change is claimed
    #: over (e.g. "next 10 matching tasks"). Free text with a task count
    #: when known, because a change with no horizon is unfalsifiable.
    horizon: str = ""
    horizon_tasks: Optional[int] = None
    expected_changes: List[ExpectedChange] = field(default_factory=list)
    learning_cost: Optional[ExpectedCost] = None
    degradation_risk: Optional[RiskStatement] = None
    uncertainty: Optional[UncertaintyStatement] = None
    verification_conditions: List[VerificationCondition] = field(
        default_factory=list)
    trace: Optional[PredictionTrace] = None
    service_available: bool = False
    #: A provider object is attached (weaker than ``service_available``).
    provider_configured: bool = False
    #: This build implements a capability-evolution service. It does NOT:
    #: the contract exists, the service does not — so a configured provider
    #: must never make this kind look available.
    service_implemented: bool = False
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.prediction_type != "capability_evolution":
            raise ValueError(
                "prediction_type must be 'capability_evolution'")
        if self.trace is None:
            self.trace = PredictionTrace(
                prediction_kind="capability_evolution",
                input_snapshot_id="")
        elif self.trace.prediction_kind != "capability_evolution":
            raise ValueError("trace.prediction_kind must be "
                             "'capability_evolution'")

    @property
    def prediction_made(self) -> bool:
        """Whether a real capability prediction was produced.

        A capability forecast needs its observable consequences, not just a
        status: this requires ``valid`` AND at least one expected change.
        """
        return self.status == "valid" and bool(self.expected_changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type,
            "status": self.status,
            "prediction_made": self.prediction_made,
            "service_available": bool(self.service_available),
            "provider_configured": bool(self.provider_configured),
            "service_implemented": bool(self.service_implemented),
            "current_evidence": self.current_evidence.to_dict(),
            "candidate_operation": self.candidate_operation.to_dict(),
            "experience_scope": (self.experience_scope.to_dict()
                                 if self.experience_scope is not None
                                 else None),
            "task_targeting": (self.task_targeting.to_dict()
                               if self.task_targeting is not None else None),
            "baseline": (self.baseline.to_dict()
                         if self.baseline is not None else None),
            "horizon": self.horizon,
            "horizon_tasks": self.horizon_tasks,
            "expected_changes": [c.to_dict() for c in self.expected_changes],
            "learning_cost": (self.learning_cost.to_dict()
                              if self.learning_cost is not None else None),
            "degradation_risk": (self.degradation_risk.to_dict()
                                 if self.degradation_risk is not None
                                 else None),
            "uncertainty": (self.uncertainty.to_dict()
                            if self.uncertainty is not None else None),
            "verification_conditions": [v.to_dict() for v
                                        in self.verification_conditions],
            "trace": self.trace.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]
                  ) -> "CapabilityEvolutionPrediction":
        if not isinstance(data, dict):
            raise ValueError(
                "CapabilityEvolutionPrediction must be a JSON object")
        version = str(data.get("contract_version", ""))
        if version not in SUPPORTED_CONTRACT_VERSIONS:
            raise UnsupportedContractVersion(version)
        if not data.get("prediction_id"):
            raise ValueError("prediction_id is required")
        status = str(data.get("status", "draft"))
        if status not in CONTRACT_STATUSES:
            raise ValueError(f"status must be one of {CONTRACT_STATUSES}")
        raw_scope = data.get("experience_scope")
        raw_targeting = data.get("task_targeting")
        raw_baseline = data.get("baseline")
        raw_cost = data.get("learning_cost")
        raw_risk = data.get("degradation_risk")
        raw_unc = data.get("uncertainty")
        return cls(
            current_evidence=HarnessCapabilityEvidence.from_dict(
                data.get("current_evidence") or {}),
            candidate_operation=LearningOperation.from_dict(
                data.get("candidate_operation") or {}),
            prediction_id=str(data["prediction_id"]),
            prediction_type=str(data.get("prediction_type",
                                         "capability_evolution")),
            contract_version=version,
            status=status,
            experience_scope=(ExperienceScope.from_dict(raw_scope)
                              if isinstance(raw_scope, dict) else None),
            task_targeting=(TaskTargeting.from_dict(raw_targeting)
                            if isinstance(raw_targeting, dict) else None),
            baseline=(BaselineStatement.from_dict(raw_baseline)
                      if isinstance(raw_baseline, dict) else None),
            horizon=str(data.get("horizon", "")),
            horizon_tasks=(int(data["horizon_tasks"])
                           if data.get("horizon_tasks") is not None
                           else None),
            expected_changes=[ExpectedChange.from_dict(c) for c
                              in (data.get("expected_changes") or [])],
            learning_cost=(ExpectedCost.from_dict(raw_cost)
                           if isinstance(raw_cost, dict) else None),
            degradation_risk=(RiskStatement.from_dict(raw_risk)
                              if isinstance(raw_risk, dict) else None),
            uncertainty=(UncertaintyStatement.from_dict(raw_unc)
                         if isinstance(raw_unc, dict) else None),
            verification_conditions=[VerificationCondition.from_dict(v)
                                     for v in
                                     (data.get("verification_conditions")
                                      or [])],
            trace=PredictionTrace.from_dict(data.get("trace") or {}),
            service_available=bool(data.get("service_available", False)),
            provider_configured=bool(data.get("provider_configured", False)),
            service_implemented=bool(data.get("service_implemented", False)),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


class UnsupportedContractVersion(ValueError):
    """Raised when a payload names a contract version this build cannot read.

    Explicit failure, never a guessed parse: reading an unknown version as
    if it were the current one would silently invent semantics.
    """

    def __init__(self, version: str):
        self.version = version
        super().__init__(
            f"unsupported contract version {version!r}: this build reads "
            f"{SUPPORTED_CONTRACT_VERSIONS} plus legacy unversioned payloads "
            f"({LEGACY_CONTRACT_VERSION})")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_strategy_outcome(prediction: StrategyOutcomePrediction
                              ) -> List[str]:
    """Validate contract 1. Empty list = valid.

    ``contract_only`` is validated too: a contract with no service attached
    must still be internally consistent, and it must NOT carry numbers that
    only a service could have produced.
    """
    problems: List[str] = []
    if prediction.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        problems.append(f"unsupported contract_version "
                        f"{prediction.contract_version!r}")
    if prediction.status not in CONTRACT_STATUSES:
        problems.append(f"status {prediction.status!r} not in "
                        f"{CONTRACT_STATUSES}")
    if prediction.status == "valid" and not prediction.service_available:
        problems.append(
            "status='valid' requires a prediction service: a contract with "
            "no service attached is 'contract_only'")
    if prediction.status == "valid" and not prediction.has_predicted_content:
        problems.append(
            "status='valid' requires predicted content: a contract with no "
            "benefit/cost/risk/uncertainty is a SCHEMA object, not a "
            "prediction — a configured provider does not by itself make a "
            "forecast happen")
    if prediction.status == "contract_only" \
            and prediction.service_available \
            and prediction.has_predicted_content:
        problems.append(
            "status='contract_only' with predicted content and an available "
            "service is contradictory: carrying a real prediction while "
            "claiming none was made is exactly the confusion this status "
            "exists to prevent")
    if prediction.status == "contract_only" \
            and not prediction.provider_configured \
            and not prediction.service_implemented:
        # Not a problem: the honest default. Named here so the reader sees
        # the state is deliberate.
        pass
    if not prediction.candidate.strategy_id \
            and prediction.candidate.action_type == "execute_strategy":
        problems.append("an execute_strategy candidate requires a "
                        "strategy_id (the prediction is about a strategy)")
    if prediction.candidate.scope == "strategy_window" \
            and not prediction.candidate.window_id:
        problems.append(
            "scope='strategy_window' requires a recorded window_id: a "
            "window-scope prediction with no real window is not comparable")
    if prediction.candidate.scope == "strategy_window" \
            and prediction.status == "valid" \
            and prediction.trace.comparable \
            and prediction.trace.not_comparable_reasons:
        problems.append(
            "a comparable window-scope prediction must carry no "
            "not_comparable_reasons: the two flags contradict each other")
    benefit = prediction.benefit
    if benefit is not None:
        if benefit.value is not None and not _finite(benefit.value):
            problems.append("benefit.value is not a finite number")
        if benefit.kind == "solution_quality" and benefit.value is not None \
                and not (0.0 <= float(benefit.value) <= 1.0):
            problems.append(
                "benefit.value for kind='solution_quality' must be in "
                "[0, 1]: this build's solution-quality currency is a "
                "NORMALIZED metric (e.g. 1-gap). A raw objective value is "
                "not silently clamped or rescaled — declare the metric you "
                "actually mean (metric/unit) or use a different kind")
        if benefit.interval is not None:
            lo, hi = benefit.interval
            if not _finite(lo) or not _finite(hi) or lo > hi:
                problems.append("benefit.interval must be finite with lo<=hi")
        if benefit.value is not None and benefit.baseline is None:
            problems.append(
                "benefit.value requires a baseline: a gain with no baseline "
                "is not a prediction")
    cost = prediction.cost
    if cost is not None and cost.expected is not None:
        for dim in COST_DIMENSIONS:
            value = getattr(cost.expected, dim)
            if not _finite(value) or float(value) < 0:
                problems.append(f"cost.expected.{dim} must be finite and "
                                ">= 0")
        for dim in cost.measured:
            if dim not in COST_DIMENSIONS:
                problems.append(f"cost.expected_measured has unknown "
                                f"dimension {dim!r}")
    risk = prediction.risk
    if risk is not None:
        for index, event in enumerate(risk.events):
            if not event.event:
                problems.append(f"risk.events[{index}].event is required")
            if event.probability is not None \
                    and not _prob(event.probability):
                problems.append(
                    f"risk.events[{index}].probability must be in [0, 1]")
            if event.severity is not None and not _finite(event.severity):
                problems.append(
                    f"risk.events[{index}].severity must be finite")
    uncertainty = prediction.uncertainty
    if uncertainty is not None:
        for name in ("execution_randomness", "knowledge_gap"):
            value = getattr(uncertainty, name)
            if value is not None and not _prob(value):
                problems.append(f"uncertainty.{name} must be in [0, 1]")
        if uncertainty.source == "model_self_report" \
                and (uncertainty.execution_randomness is not None
                     or uncertainty.knowledge_gap is not None):
            problems.append(
                "uncertainty.source='model_self_report' carrying numeric "
                "components is not a calibrated probability: store the "
                "model's self-report in notes/model_info, or restate the "
                "components under an honest source "
                "(framework_heuristic/measured). A self-report may not be "
                "relabeled to pass validation")
    if prediction.trace is None:
        problems.append("trace is required (version, input, evidence, "
                        "unsupported fields)")
    else:
        if prediction.trace.prediction_kind != "strategy_outcome":
            problems.append("trace.prediction_kind must be "
                            "'strategy_outcome'")
        if prediction.status == "valid" \
                and not prediction.trace.input_snapshot_id:
            problems.append(
                "a valid prediction requires the frozen input snapshot id "
                "it was conditioned on")
    return problems


def validate_capability_evolution(prediction: CapabilityEvolutionPrediction
                                  ) -> List[str]:
    """Validate contract 2. Empty list = valid."""
    problems: List[str] = []
    if prediction.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        problems.append(f"unsupported contract_version "
                        f"{prediction.contract_version!r}")
    if prediction.status not in CONTRACT_STATUSES:
        problems.append(f"status {prediction.status!r} not in "
                        f"{CONTRACT_STATUSES}")
    if prediction.status == "valid" and not prediction.service_available:
        problems.append(
            "status='valid' requires a prediction service: a contract with "
            "no service attached is 'contract_only'")
    if prediction.status == "valid" and not prediction.prediction_made:
        problems.append(
            "status='valid' requires observable consequences: a capability "
            "forecast with no expected_change is a SCHEMA object, not a "
            "prediction")
    if prediction.status == "contract_only" \
            and prediction.service_available \
            and prediction.expected_changes:
        problems.append(
            "status='contract_only' with expected changes and an available "
            "service is contradictory: that is a 'valid' prediction")
    problems.extend(
        f"current_evidence: {p}" for p
        in validate_capability_evidence(prediction.current_evidence))
    if not prediction.horizon:
        problems.append(
            "horizon is required: a capability change with no horizon "
            "cannot be falsified")
    if prediction.status == "valid" and not prediction.expected_changes:
        problems.append(
            "a valid capability prediction requires at least one "
            "expected_change (H is judged through observable consequences)")
    if prediction.status == "valid" and not prediction.task_targeting:
        problems.append(
            "a valid capability prediction requires task_targeting: the "
            "task types it is claimed over")
    if prediction.status == "valid" and prediction.baseline is None:
        problems.append(
            "a valid capability prediction requires a baseline to measure "
            "the change against")
    if prediction.status == "valid" \
            and not prediction.verification_conditions:
        problems.append(
            "a valid capability prediction requires verification_conditions: "
            "predicting, binding and verifying are three different things")
    for index, change in enumerate(prediction.expected_changes):
        if not change.metric:
            problems.append(f"expected_changes[{index}].metric is required")
        if change.value is not None and not _finite(change.value):
            problems.append(
                f"expected_changes[{index}].value must be finite")
        if change.direction == "unknown" and change.value is not None:
            problems.append(
                f"expected_changes[{index}]: a value with direction="
                "'unknown' is contradictory")
        if change.value is not None and change.baseline is None \
                and prediction.baseline is None:
            problems.append(
                f"expected_changes[{index}]: a value requires a baseline "
                "(its own or the prediction's): a change with nothing to "
                "measure it against is not falsifiable")
        if change.value_kind == "relative" and change.value is not None \
                and not -1.0 <= float(change.value) <= 1.0:
            problems.append(
                f"expected_changes[{index}].value_kind='relative' requires "
                "a RATIO in [-1, 1] (express 20% as 0.2); a relative "
                "change outside that range is not a ratio, and a "
                "percentage and a ratio must never be added together")
    cost = prediction.learning_cost
    if cost is not None and cost.expected is not None:
        for dim in COST_DIMENSIONS:
            value = getattr(cost.expected, dim)
            if not _finite(value) or float(value) < 0:
                problems.append(f"learning_cost.expected.{dim} must be "
                                "finite and >= 0")
    risk = prediction.degradation_risk
    if risk is not None:
        for index, event in enumerate(risk.events):
            if event.probability is not None \
                    and not _prob(event.probability):
                problems.append(f"degradation_risk.events[{index}]."
                                "probability must be in [0, 1]")
    uncertainty = prediction.uncertainty
    if uncertainty is not None:
        for name in ("execution_randomness", "knowledge_gap"):
            value = getattr(uncertainty, name)
            if value is not None and not _prob(value):
                problems.append(f"uncertainty.{name} must be in [0, 1]")
    if prediction.trace is None:
        problems.append("trace is required")
    elif prediction.trace.prediction_kind != "capability_evolution":
        problems.append("trace.prediction_kind must be "
                        "'capability_evolution'")
    return problems


def validate_contract(prediction) -> List[str]:
    """Dispatch validation by prediction type."""
    if isinstance(prediction, StrategyOutcomePrediction):
        return validate_strategy_outcome(prediction)
    if isinstance(prediction, CapabilityEvolutionPrediction):
        return validate_capability_evolution(prediction)
    return [f"unknown contract type {type(prediction).__name__}"]


# ---------------------------------------------------------------------------
# version identification + legacy compatibility
# ---------------------------------------------------------------------------


def detect_payload_version(payload: Any) -> str:
    """Identify a stored payload's contract version WITHOUT guessing.

    Returns one of:

    - ``"wm-contract/1"`` — the current contract;
    - ``"legacy/unversioned"`` — a payload that has no version field at all
      (the old ``OutcomePrediction`` shape, belief snapshots, budget state);
    - ``"unknown/<label>"`` — a payload that NAMES a version this build does
      not read. Callers must report this as unsupported rather than parsing
      it.
    """
    if not isinstance(payload, dict):
        return f"{PAYLOAD_VERSION_UNKNOWN_PREFIX}not-an-object"
    if "contract_version" in payload:
        version = str(payload.get("contract_version") or "")
        if version in SUPPORTED_CONTRACT_VERSIONS:
            return version
        return f"{PAYLOAD_VERSION_UNKNOWN_PREFIX}{version or 'empty'}"
    return LEGACY_CONTRACT_VERSION


def load_contract_payload(payload: Any):
    """Load a current-contract payload, raising on an unreadable version.

    A legacy payload is NOT accepted here: reading it as the current
    contract would invent semantics it never had. Use
    :func:`legacy_prediction_view` for that.
    """
    version = detect_payload_version(payload)
    if version == LEGACY_CONTRACT_VERSION:
        raise UnsupportedContractVersion(LEGACY_CONTRACT_VERSION)
    if version.startswith(PAYLOAD_VERSION_UNKNOWN_PREFIX):
        raise UnsupportedContractVersion(
            version[len(PAYLOAD_VERSION_UNKNOWN_PREFIX):])
    kind = str((payload or {}).get("prediction_type", ""))
    if kind == "strategy_outcome":
        return StrategyOutcomePrediction.from_dict(payload)
    if kind == "capability_evolution":
        return CapabilityEvolutionPrediction.from_dict(payload)
    raise ValueError(f"prediction_type {kind!r} not in {PREDICTION_KINDS}")


def legacy_prediction_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Read an old unversioned prediction payload through a NEW view.

    The old semantics are preserved exactly; the view only ADDS structure:

    - ``legacy_status`` / ``legacy_fields`` record that this is a legacy
      payload, so nothing downstream can mistake it for a current-contract
      prediction;
    - ``capability_evidence`` is derived as INDIRECT evidence (counts and
      references), never as a measured capability increment;
    - risk severity is left absent unless the legacy payload carried one,
      and rework cost is never re-modelled (it stays in the cost vector);
    - fields with no honest equivalent are listed in
      :data:`LEGACY_UNMAPPABLE` under ``unmappable``.

    An empty/invalid payload still returns a view with explicit gaps rather
    than raising: the reader's job is to show what is known.
    """
    payload = dict(payload or {})
    predicted = dict(payload.get("predicted") or {})
    spec = dict(payload.get("action_spec") or {})
    view: Dict[str, Any] = {
        "legacy_status": LEGACY_CONTRACT_VERSION,
        "is_current_contract": False,
        "prediction_id": payload.get("prediction_id"),
        "scope": spec.get("measurement_scope", "attempt"),
        "candidate": {
            "action_type": spec.get("action_type"),
            "strategy_id": spec.get("strategy_id"),
            "solver": spec.get("solver"),
            "task_id": spec.get("task_id", ""),
            "episode_id": spec.get("episode_id"),
        },
        "mapped_fields": {},
        "unmappable": copy.deepcopy(LEGACY_UNMAPPABLE),
        "derived": [],
        "gaps": [],
    }
    mapped = view["mapped_fields"]
    if "outcome_status" in predicted:
        mapped["candidate.expected_status"] = predicted["outcome_status"]
    if "feasible" in predicted:
        mapped["benefit.feasible"] = predicted["feasible"]
    if predicted.get("quality") is not None:
        mapped["benefit.value"] = predicted["quality"]
        mapped["benefit.kind"] = "solution_quality"
        view["gaps"].append(
            "benefit.metric/unit/baseline are NOT derivable from a legacy "
            "payload: the old quality number carried no stated metric")
    if predicted.get("failure_prob") is not None:
        mapped["risk.events"] = [{
            "event": "task_failure",
            "probability": predicted["failure_prob"],
            "severity": None,
            "basis": "legacy failure_prob",
        }]
        view["gaps"].append(
            "risk severity was never recorded by the legacy contract: left "
            "absent rather than invented")
    if isinstance(predicted.get("cost"), dict):
        mapped["cost.expected"] = copy.deepcopy(predicted["cost"])
        mapped["cost.expected_measured"] = list(
            predicted.get("cost_measured") or [])
    if predicted.get("state_changes") is not None:
        mapped["benefit.progress_shape"] = copy.deepcopy(
            predicted["state_changes"])
        view["gaps"].append(
            "the legacy state_changes detail is read as an X progress SHAPE "
            "only; per-field successor values are not carried forward")
    if payload.get("confidence") is not None:
        mapped["uncertainty"] = {
            "source": "model_self_report",
            "value": payload["confidence"],
            "calibrated": False,
        }
        view["gaps"].append(
            "the model's self-reported confidence is NOT a calibrated "
            "probability and is not carried as one")
    if payload.get("evidence_basis") is not None:
        mapped["trace.evidence_basis"] = list(payload["evidence_basis"])
    if payload.get("unsupported_fields") is not None:
        mapped["trace.unsupported_fields"] = dict(
            payload["unsupported_fields"])
    if isinstance(payload.get("call_cost"), dict):
        mapped["trace.call_cost"] = copy.deepcopy(payload["call_cost"])
    if payload.get("input_snapshot_id"):
        mapped["trace.input_snapshot_id"] = payload["input_snapshot_id"]
    if predicted.get("knowledge_changes") is not None:
        mapped["capability_evolution.expected_changes_raw"] = copy.deepcopy(
            predicted["knowledge_changes"])
        view["derived"].append(
            "legacy knowledge_changes are knowledge-CHANGE hypotheses, not a "
            "measured capability increment: they are carried as raw "
            "hypotheses for reference only")
    # Capability evidence: derived as INDIRECT evidence from whatever the
    # legacy payload (or its snapshot) recorded. Counts are never a level.
    view["capability_evidence"] = {
        "score_scheme": "no_composite_score",
        "note": ("derived from legacy references/counts as indirect "
                 "evidence; no capability level or increment is implied"),
        "sources": {
            "m": {"status": ("indirect_evidence"
                             if payload.get("evidence_basis")
                             else "no_evidence")},
            "w_or": {"status": "no_evidence"},
            "pi": {"status": "no_evidence"},
            "r": {"status": "no_evidence"},
            "t": {"status": "no_evidence"},
        },
    }
    return view


def prediction_kinds_for_mode(mode: str) -> Tuple[str, ...]:
    """Which contract kinds a legacy prediction mode covers.

    The legacy mode names remain the experiment ablation switches; this maps
    them onto the unified vocabulary so a reader does not need both.
    """
    if mode not in LEGACY_MODE_TO_KIND:
        raise ValueError(f"unknown prediction mode {mode!r}; expected one of "
                         f"{sorted(LEGACY_MODE_TO_KIND)}")
    return LEGACY_MODE_TO_KIND[mode]
