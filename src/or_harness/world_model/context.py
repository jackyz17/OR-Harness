"""The prediction input context: what a prediction is actually conditioned on.

Phase 2 of the world-model reconstruction. Phase 1 defined HOW state,
candidates and predictions are expressed; this module defines WHAT
information a prediction actually uses, and how that information reaches the
model consistently, completely and traceably.

The context is the frozen, versioned bundle of everything a prediction may
legitimately condition on:

- **the joint problem representation** — the task's own text and payload, its
  CIR (relational structure, kept as relations rather than compressed into
  three coupling numbers), its math attributes with an explicit ORIGIN each,
  the structural profile and its derivation report. Available BEFORE any
  mathematical model exists: a task with no ``model`` and no solve.py is a
  normal input, and every attribute that cannot be established stays
  ``unknown`` instead of being guessed from the scenario's name;
- **X** — the current solving context (task progress, budget state) frozen
  from the same snapshot the prediction is conditioned on;
- **retrieval evidence** — the two EXISTING channels (structural
  recommendations, text-similarity recall), deduplicated by evidence
  identity, each hit carrying its content, task conditions, observed
  result/cost, verification state, applicability and reuse verdict. The two
  channels stay separate: no composite retrieval score is invented;
- **Harness capability evidence** — the phase-1
  :class:`~or_harness.world_model.contracts.HarnessCapabilityEvidence`
  (``H = F(M, W_OR, Pi, R, T)``) plus a capability VERSION block that
  distinguishes harness config, model/prompt, tools and memory content. No
  composite H score, no fabricated source;
- **external execution constraints** — declared budget, consumption view,
  available solver families, executor limits.

Design boundaries enforced here (they are the reason this module exists):

- Building a context performs NO solver execution, NO prediction-model call
  and NO induction. It may call the embedding backend (the existing
  retrieval path) and it persists the frozen context;
- the context is FROZEN: later edits to the task object, knowledge entries,
  banks or tool configuration never change it. A mutable id alone is not
  enough to reproduce a prediction input — the content and its version are;
- reuse is VERSION-VERIFIED: a recall result or snapshot produced for a
  different task version is refused by
  :func:`context_identity_problems`, never silently reused;
- "retrieved" is not "verified": unverified candidates and hypothetical
  predictions are carried with their own evidence class and never become
  verified knowledge or real evidence by being retrieved;
- missing/degraded parts are recorded per part, so "the semantic channel did
  not run" and "the semantic channel ran and found nothing" stay different
  facts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import ProblemProfile
from or_harness.world_model.contracts import (
    HarnessCapabilityEvidence,
    capability_evidence_from_legacy_harness_state,
)

#: Version of the context SCHEMA. A reader that does not know this version
#: must refuse the payload rather than guess at it.
PREDICTION_CONTEXT_VERSION = "wm-context/1"

#: Version of the joint problem representation's shape.
JOINT_REPRESENTATION_VERSION = "joint/1"

#: Where an attribute's value came from. ``unknown`` is a first-class origin:
#: an attribute nobody established is reported as unknown, never defaulted.
MATH_ATTRIBUTE_ORIGINS = ("declared", "derived", "spec", "cir", "model",
                          "unknown")

#: The math attributes the joint representation carries.
MATH_ATTRIBUTE_NAMES = ("integrality", "linearity", "objective_kind",
                        "constraint_kinds")

#: The two retrieval channels. Never blended into one number.
RETRIEVAL_CHANNELS = ("structural", "semantic")

#: How a retrieved item may be used. Kept separate from its similarity: a
#: discovery signal is not a reuse licence.
EVIDENCE_CLASSES = ("execution_fact", "verified_knowledge",
                    "unverified_knowledge", "legacy_knowledge",
                    "structural_recommendation")

#: Bounds. A context carries BOUNDED evidence, never the whole bank.
MAX_HITS_PER_CHANNEL = 20
MAX_CIR_RELATIONS = 50
MAX_TEXT_CHARS = 4000
MAX_EXCERPT_CHARS = 400
MAX_ENTITY_NAMES = 30


class UnsupportedContextVersion(Exception):
    """The payload's context version is not one this build understands."""


def _stable_digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def resolve_effective_cir(task: Dict[str, Any],
                          cir: Optional[Any] = None) -> Any:
    """The ONE CIR this request is about.

    An explicit ``cir`` wins over the task's own ``coupling`` field. This is
    deliberately a single resolution point: the joint representation, the
    profile (hence the snapshot's structural cell) and the retrieval channels
    must all describe the SAME problem structure, or one request would carry
    two structural judgments and could retrieve knowledge from the wrong
    cell.
    """
    if cir is not None:
        return cir
    if isinstance(task, dict) and isinstance(task.get("coupling"), dict):
        return task["coupling"]
    return None


def task_with_effective_cir(task: Dict[str, Any],
                            cir: Optional[Any] = None) -> Dict[str, Any]:
    """A value copy of ``task`` whose ``coupling`` IS the effective CIR.

    Passing the task this way is how an explicit ``cir`` reaches every
    existing consumer (``profile``, ``snapshot``, ``recall``) without adding
    a parallel parameter to each — they already read ``task["coupling"]``, so
    they end up reading the same structure the joint representation used.
    With no effective CIR the task is returned with its ``coupling`` removed,
    so a caller-supplied CIR REPLACES the task's own rather than being
    silently overridden by it.
    """
    resolved = resolve_effective_cir(task, cir)
    copy_task = dict(task or {})
    if resolved is None:
        copy_task.pop("coupling", None)
    elif hasattr(resolved, "to_dict"):
        copy_task["coupling"] = resolved.to_dict()
    else:
        copy_task["coupling"] = copy.deepcopy(dict(resolved))
    return copy_task


def effective_input_version(task: Dict[str, Any],
                            cir: Optional[Any] = None) -> str:
    """The version identity of the EFFECTIVE problem input of a request.

    ``task_text_digest(task)`` digests the task JSON as the CALLER supplied
    it, so the same problem expressed once through an explicit ``cir`` and
    once through the task's own ``coupling`` field produces two DIFFERENT
    digests — and a context built the first way was then refused when reused
    the second way, even though both requests describe the same effective
    problem. This digest is computed over the RESOLVED input: the task with
    the effective CIR already merged in (``task_with_effective_cir``), so
    the identity follows the problem the prediction actually conditions on.

    Two different CIRs that happen to produce the same rc/tc/rx still
    produce different digests here, because the RELATIONS are part of the
    digested content — three coupling numbers agreeing is not relations
    agreeing.
    """
    return _stable_digest(task_with_effective_cir(task, cir))


#: The dimensions that define a structural CELL. Two descriptions that
#: disagree on any of these are describing different structures, so a
#: snapshot (or a reused retrieval result) may not be combined with the
#: other one.
STRUCTURE_DIMENSIONS = ("resource_coupling", "temporal_coupling",
                        "route_complexity")


def _profile_dims(source: Any) -> Dict[str, Optional[float]]:
    """The structural dims of a profile-like object or dict."""
    if source is None:
        return {}
    out: Dict[str, Optional[float]] = {}
    for dim in STRUCTURE_DIMENSIONS:
        value = (source.get(dim) if isinstance(source, dict)
                 else getattr(source, dim, None))
        out[dim] = None if value is None else float(value)
    return out


def structure_problems(*, expected: Any, actual: Any, label: str,
                       tolerance: float = 1e-9) -> List[str]:
    """Reasons a frozen artifact's structure disagrees with the effective one.

    Used AFTER the effective input (including any explicit CIR) has been
    resolved: a snapshot or a reused recall result built from a DIFFERENT
    structure may not be combined with this request, because the joint
    representation would then describe one problem while the state and the
    retrieval described another — and knowledge could be pulled from the
    wrong structural cell.

    A dimension that is unknown on either side is NOT a mismatch: "cannot be
    compared" is different from "disagrees", and refusing on an unmeasured
    value would reject legitimate inputs. ``label`` names the artifact in the
    message so the caller can say which one conflicted.
    """
    expected_dims = _profile_dims(expected)
    actual_dims = _profile_dims(actual)
    if not expected_dims or not actual_dims:
        return []
    problems: List[str] = []
    for dim in STRUCTURE_DIMENSIONS:
        want = expected_dims.get(dim)
        got = actual_dims.get(dim)
        if want is None or got is None:
            continue
        if abs(want - got) > tolerance:
            problems.append(
                f"{label} describes a different structure: {dim}={got} but "
                f"the effective problem input has {dim}={want}")
    return problems


def frozen_knowledge_view(snapshot: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The knowledge view a snapshot FROZE, by value.

    A belief snapshot's ``coverage.knowledge_layers`` is the
    ``verified_knowledge_view`` as it stood when the snapshot was taken
    (``KnowledgeRef`` value copies, not ids). Historical reconstruction must
    read THAT rather than today's bank: an entry that was revised after the
    snapshot would otherwise be reported at its NEW expected quality inside a
    context describing the earlier state.
    """
    layers = (getattr(snapshot, "coverage", None) or {}).get(
        "knowledge_layers") or {}
    return {
        "verified": copy.deepcopy(list(layers.get("verified") or [])),
        "legacy_unknown": copy.deepcopy(
            list(layers.get("legacy_unknown") or [])),
        "unverified": copy.deepcopy(list(layers.get("unverified") or [])),
    }


def frozen_knowledge_available(snapshot: Any) -> bool:
    """Whether a snapshot actually carries a frozen knowledge view.

    An absent/empty ``knowledge_layers`` means the view was never recorded,
    which is NOT the same as "there was no knowledge" — the caller must
    report it as missing rather than reading today's bank to fill the gap.
    """
    coverage = getattr(snapshot, "coverage", None) or {}
    return isinstance(coverage.get("knowledge_layers"), dict)


def _bounded(items: Sequence[Any], limit: int,
             notes: List[str], label: str) -> List[Any]:
    """First ``limit`` items, recording the truncation (never silent)."""
    if len(items) > limit:
        notes.append(
            f"{label}: {len(items)} item(s) available, {limit} carried — "
            "the context is bounded on purpose and never inlines the whole "
            "bank")
        return list(items[:limit])
    return list(items)


# ---------------------------------------------------------------------------
# joint problem representation (available BEFORE any model exists)
# ---------------------------------------------------------------------------


@dataclass
class MathAttributes:
    """Mathematical properties of P, each with the origin of its value.

    Established from an explicit declaration, a structured spec, or the
    declared model — never from the scenario's name. ``unknown`` is honest
    and first-class: before a model exists most of these are genuinely
    unknown, and inventing "it is a MILP" from the word "routing" would be
    exactly the fabrication this representation exists to prevent.
    """

    integrality: Optional[str] = None
    linearity: Optional[str] = None
    objective_kind: Optional[str] = None
    constraint_kinds: List[str] = field(default_factory=list)
    #: attribute -> origin (see :data:`MATH_ATTRIBUTE_ORIGINS`).
    origins: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def origin(self, name: str) -> str:
        return self.origins.get(name, "unknown")

    @property
    def unknowns(self) -> List[str]:
        """Attributes nothing established (an empty list is not a value)."""
        out: List[str] = []
        for name in MATH_ATTRIBUTE_NAMES:
            if name == "constraint_kinds":
                if not self.constraint_kinds:
                    out.append(name)
                continue
            if getattr(self, name) is None:
                out.append(name)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integrality": self.integrality,
            "linearity": self.linearity,
            "objective_kind": self.objective_kind,
            "constraint_kinds": list(self.constraint_kinds),
            "origins": dict(self.origins),
            "unknowns": self.unknowns,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathAttributes":
        data = data or {}
        return cls(
            integrality=data.get("integrality"),
            linearity=data.get("linearity"),
            objective_kind=data.get("objective_kind"),
            constraint_kinds=[str(k) for k in
                              (data.get("constraint_kinds") or [])],
            origins={str(k): str(v) for k, v in
                     (data.get("origins") or {}).items()},
            notes=[str(n) for n in (data.get("notes") or [])],
        )


#: Sense words the model representation or an explicit spec may carry.
_SENSE_WORDS = ("minimize", "maximize", "min", "max")


def _normalize_sense(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower()
    for word in _SENSE_WORDS:
        if token.startswith(word):
            return "max" if word.startswith("max") else "min"
    return None


def math_attributes(task: Dict[str, Any], *,
                    cir: Any = None,
                    declared: Optional[Dict[str, Any]] = None
                    ) -> MathAttributes:
    """Derive the math attributes of P from what is really available.

    Priority per attribute: an explicit ``declared`` mapping (the outer agent
    taking responsibility) > the declared model representation > the
    structured spec > the CIR. Nothing is derived from ``family``: the
    problem's NAME is not evidence about its mathematics.

    ``linearity`` is a SYNTACTIC candidate read off the declared model's
    expressions (a product of two declared decision variables). It is
    labelled as such in ``notes`` and is not a proof of non-convexity.
    """
    from or_harness.profiling.model_syntax import parse_model

    out = MathAttributes()
    declared = dict(declared or {})
    spec = dict(task.get("spec") or {})
    model_text = task.get("model")
    parsed = None
    if isinstance(model_text, str) and model_text.strip():
        parsed = parse_model(model_text)

    # integrality
    if declared.get("integrality"):
        out.integrality = str(declared["integrality"])
        out.origins["integrality"] = "declared"
    elif parsed is not None and parsed.variables:
        types = {str(v).lower() for v in parsed.variables.values()}
        discrete = types & {"binary", "integer"}
        if discrete and types - {"binary", "integer"}:
            out.integrality = "mixed"
        elif discrete:
            out.integrality = "integer"
        else:
            out.integrality = "continuous"
        out.origins["integrality"] = "model"
    elif "n_int_vars" in spec and "n_vars" in spec:
        try:
            n_int = float(spec["n_int_vars"])
            n_vars = float(spec["n_vars"])
        except (TypeError, ValueError):
            n_int = n_vars = None
        if n_vars:
            if n_int <= 0:
                out.integrality = "continuous"
            elif n_int >= n_vars:
                out.integrality = "integer"
            else:
                out.integrality = "mixed"
            out.origins["integrality"] = "spec"

    # objective sense
    if declared.get("objective_kind"):
        out.objective_kind = str(declared["objective_kind"])
        out.origins["objective_kind"] = "declared"
    else:
        sense = (_normalize_sense(spec.get("objective_sense"))
                 or _normalize_sense(spec.get("sense")))
        if sense:
            out.objective_kind = sense
            out.origins["objective_kind"] = "spec"
        elif parsed is not None and parsed.objective:
            sense = _normalize_sense(parsed.objective)
            if sense:
                out.objective_kind = sense
                out.origins["objective_kind"] = "model"

    # linearity (syntactic candidate from the declared model)
    if declared.get("linearity"):
        out.linearity = str(declared["linearity"])
        out.origins["linearity"] = "declared"
    elif parsed is not None and parsed.variables:
        if _has_variable_product(parsed):
            out.linearity = "nonlinear_candidate"
        else:
            out.linearity = "linear_candidate"
        out.origins["linearity"] = "model"
        out.notes.append(
            "linearity is a SYNTACTIC candidate read off the declared "
            "model's expressions (a product of two declared decision "
            "variables); it is not a proof of convexity or non-convexity")

    # constraint kinds
    kinds: List[str] = []
    origin = "unknown"
    if declared.get("constraint_kinds"):
        kinds = [str(k) for k in declared["constraint_kinds"]]
        origin = "declared"
    else:
        cir_kinds: List[str] = []
        constraints = getattr(cir, "constraints", None)
        if constraints is None and isinstance(cir, dict):
            constraints = cir.get("constraints")
        for constraint in constraints or []:
            kind = (constraint.get("kind") if isinstance(constraint, dict)
                    else getattr(constraint, "kind", None))
            if kind and str(kind) != "other" and str(kind) not in cir_kinds:
                cir_kinds.append(str(kind))
        if cir_kinds:
            kinds, origin = sorted(cir_kinds), "cir"
        elif spec.get("constraint_kinds"):
            kinds = [str(k) for k in spec["constraint_kinds"]]
            origin = "spec"
    if kinds:
        out.constraint_kinds = kinds
        out.origins["constraint_kinds"] = origin
    return out


def _has_variable_product(parsed: Any) -> bool:
    """Whether two DECLARED variables are multiplied in objective/constraints.

    Syntactic only, and deliberately narrow: ``x[i] * y`` counts, ``2 * x``
    does not (a scalar times a variable is still linear).
    """
    import re
    names = set(parsed.variables)
    if not names:
        return False
    expressions = [parsed.objective] + [expr for _label, expr in
                                        parsed.constraints]
    for expr in expressions:
        for match in re.finditer(r"([A-Za-z_]\w*)\s*\[[^\]]*\]\s*\*|"
                                 r"\*\s*([A-Za-z_]\w*)\s*\[", expr):
            for group in match.groups():
                if group and group in names:
                    return True
        for match in re.finditer(r"([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)",
                                 expr):
            left, right = match.group(1), match.group(2)
            if left in names and right in names:
                return True
    return False


@dataclass
class JointProblemRepresentation:
    """P as one representation: semantics + structure + mathematics.

    Everything a prediction needs to know WHAT the problem is, kept in its
    own form — the CIR's relations stay relations (not three coupling
    numbers), the task's text stays text (not a hash), and the math
    attributes carry their origin. Absent parts are reported as absent, so
    "no model yet" is a stated input state rather than a silent gap.
    """

    task_id: str
    task_digest: str
    text: str = ""
    text_digest: Optional[str] = None
    task_payload: Dict[str, Any] = field(default_factory=dict)
    has_model: bool = False
    model_digest: Optional[str] = None
    cir: Dict[str, Any] = field(default_factory=dict)
    math: MathAttributes = field(default_factory=MathAttributes)
    profile: Dict[str, Any] = field(default_factory=dict)
    derivation: Dict[str, Any] = field(default_factory=dict)
    #: representation part -> where its content came from.
    sources: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    version: str = JOINT_REPRESENTATION_VERSION

    @property
    def cir_present(self) -> bool:
        return bool(self.cir.get("present"))

    @property
    def unknowns(self) -> List[str]:
        return self.math.unknowns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "task_digest": self.task_digest,
            "text": self.text,
            "text_digest": self.text_digest,
            "task_payload": copy.deepcopy(self.task_payload),
            "has_model": bool(self.has_model),
            "model_digest": self.model_digest,
            "cir": copy.deepcopy(self.cir),
            "math": self.math.to_dict(),
            "profile": copy.deepcopy(self.profile),
            "derivation": copy.deepcopy(self.derivation),
            "sources": dict(self.sources),
            "missing": list(self.missing),
            "unknowns": self.unknowns,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JointProblemRepresentation":
        data = data or {}
        return cls(
            task_id=str(data.get("task_id", "")),
            task_digest=str(data.get("task_digest", "")),
            text=str(data.get("text", "")),
            text_digest=(str(data["text_digest"])
                         if data.get("text_digest") else None),
            task_payload=copy.deepcopy(dict(data.get("task_payload") or {})),
            has_model=bool(data.get("has_model", False)),
            model_digest=(str(data["model_digest"])
                          if data.get("model_digest") else None),
            cir=copy.deepcopy(dict(data.get("cir") or {})),
            math=MathAttributes.from_dict(data.get("math") or {}),
            profile=copy.deepcopy(dict(data.get("profile") or {})),
            derivation=copy.deepcopy(dict(data.get("derivation") or {})),
            sources={str(k): str(v) for k, v in
                     (data.get("sources") or {}).items()},
            missing=[str(m) for m in (data.get("missing") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
            version=str(data.get("version", JOINT_REPRESENTATION_VERSION)),
        )


def build_joint_representation(
        task: Dict[str, Any], *, profile: Optional[ProblemProfile] = None,
        derivation: Optional[Dict[str, Any]] = None,
        cir: Any = None,
        math_declared: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        cir_source: Optional[str] = None,
        notes: Optional[List[str]] = None) -> JointProblemRepresentation:
    """Assemble P's joint representation from what already exists.

    No new NLP classifier, no extra LLM analysis, no ontology: the task
    payload, the CIR, the profile and its derivation report are all things
    the harness already has. What this adds is ONE place where they are
    presented together with their sources, their unknowns and their
    missing parts.
    """
    from or_harness.world_model.state import (
        _task_payload,
        _stable_digest,
        task_text,
        task_text_digest,
    )

    out_notes = list(notes or [])
    task = task if isinstance(task, dict) else {}
    payload = _task_payload(task)
    if text is None:
        text = task_text(task)
    digest = task_text_digest(task)

    # CIR: relations kept as relations. A dict or a CouplingAwareIR both work.
    cir_dict: Dict[str, Any] = {"present": False}
    resolved_cir = cir
    if resolved_cir is None and isinstance(task.get("coupling"), dict):
        resolved_cir = task["coupling"]
    if cir_source is None:
        cir_source = ("caller_supplied" if cir is not None
                      else "task_coupling")
    if resolved_cir is not None:
        raw = (resolved_cir.to_dict()
               if hasattr(resolved_cir, "to_dict") else dict(resolved_cir))
        relations = list(raw.get("relations") or [])
        cir_dict = {
            "present": True,
            "entities": _bounded(raw.get("entities") or [], MAX_ENTITY_NAMES,
                                 out_notes, "CIR entities"),
            "decisions": _bounded(raw.get("decisions") or [], MAX_ENTITY_NAMES,
                                  out_notes, "CIR decisions"),
            "constraints": _bounded(raw.get("constraints") or [],
                                    MAX_ENTITY_NAMES, out_notes,
                                    "CIR constraints"),
            "relations": _bounded(relations, MAX_CIR_RELATIONS, out_notes,
                                  "CIR relations"),
            "coupling_groups": _bounded(raw.get("coupling_groups") or [],
                                        MAX_ENTITY_NAMES, out_notes,
                                        "CIR coupling groups"),
            "issues": list(raw.get("issues") or []),
            "n_relations": len(relations),
        }
    else:
        out_notes.append(
            "no CIR was supplied for this task: the relational structure is "
            "absent, and the structural dimensions come from the profile "
            "alone — an absent CIR is not evidence of weak coupling")

    model_text = task.get("model")
    has_model = isinstance(model_text, str) and bool(model_text.strip())
    model_digest = _stable_digest(model_text) if has_model else None

    math = math_attributes(task, cir=resolved_cir, declared=math_declared)

    sources: Dict[str, str] = {
        "text": "task_payload" if text.strip() else "none",
        "cir": (cir_source if cir_dict["present"] else "none"),
        "model": "task_model" if has_model else "none",
        "profile": "profiler" if profile is not None else "none",
        "derivation": "profiler" if derivation else "none",
    }
    for name in MATH_ATTRIBUTE_NAMES:
        sources[f"math.{name}"] = math.origin(name)

    missing: List[str] = []
    if not text.strip():
        missing.append(
            "task text: the task JSON carries no textual field, so the "
            "semantic channel has nothing to embed (this is NOT 'no similar "
            "memory exists')")
    if not cir_dict["present"]:
        missing.append("CIR: no relational structure supplied")
    if not has_model:
        missing.append(
            "mathematical model: not written yet — the math attributes come "
            "from the spec/CIR or stay unknown; this is a normal state and "
            "does not block a prediction")
    for name in math.unknowns:
        missing.append(f"math.{name}: unknown (not established by any source)")
    if profile is None:
        missing.append("profile: not built")

    return JointProblemRepresentation(
        task_id=str(task.get("task_id", "")),
        task_digest=digest,
        text=_excerpt(text, MAX_TEXT_CHARS),
        text_digest=digest,
        task_payload=payload,
        has_model=has_model,
        model_digest=model_digest,
        cir=cir_dict,
        math=math,
        profile=(profile.to_dict() if profile is not None else {}),
        derivation=copy.deepcopy(derivation or {}),
        sources=sources,
        missing=missing,
        notes=out_notes,
    )


# ---------------------------------------------------------------------------
# retrieval evidence
# ---------------------------------------------------------------------------


def evidence_identity(layer: str, evidence_id: str) -> str:
    """The identity of one piece of evidence: layer + id (version separate).

    Deliberately NOT a content hash: two channels reporting the same memory
    must collapse onto one identity, and a content change must surface as a
    VERSION difference on that same identity rather than as a new item.
    """
    return f"{layer}:{evidence_id}"


def classify_evidence(layer: str, row: Dict[str, Any]) -> str:
    """Which kind of thing a retrieved item is.

    The distinction that matters: an execution fact was OBSERVED, a verified
    knowledge entry was ADMITTED, an unverified candidate was neither, and a
    structural recommendation may be backed by nothing at all. Retrieval
    never upgrades one class into another.
    """
    if layer == "execution_evidence":
        return "execution_fact"
    if layer == "strategic_knowledge":
        if row.get("reusable") and str(row.get("verification_state")
                                       or "") == "verified":
            return "verified_knowledge"
        if not row.get("reusable"):
            return "unverified_knowledge"
        return "legacy_knowledge"
    return "structural_recommendation"


def _version_of(layer: str, row: Dict[str, Any]) -> str:
    """The content version of one evidence item (or ``unknown``)."""
    if layer == "execution_evidence":
        return str(row.get("task_text_digest") or "unknown")
    if layer == "strategic_knowledge":
        claim = row.get("claim_text")
        return _stable_digest(claim) if claim else "unknown"
    return "unknown"


def _hit(layer: str, evidence_id: str, channel: str,
         row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer": layer,
        "evidence_id": evidence_id,
        "identity": evidence_identity(layer, evidence_id),
        "version": _version_of(layer, row),
        "channels": [channel],
        "evidence_class": classify_evidence(layer, row),
        "content": copy.deepcopy(row),
    }


def _structural_hits(recommendations: Sequence[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """The structural channel as evidence items.

    A recommendation backed by a published entry is keyed on that ENTRY's id,
    so the same entry surfacing through both channels deduplicates onto one
    identity. A recommendation backed only by conditional statistics (or by
    nothing) is keyed on the strategy it is about.
    """
    hits: List[Dict[str, Any]] = []
    for rec in recommendations or []:
        refs = [str(r) for r in (rec.get("evidence_refs") or [])
                if str(r).startswith("se_")]
        if refs:
            for ref in refs:
                hits.append(_hit("strategic_knowledge", ref, "structural",
                                 {"entry_id": ref,
                                  "strategy_id": rec.get("strategy_id"),
                                  "evidence": rec.get("evidence"),
                                  "expected": copy.deepcopy(
                                      rec.get("expected") or {}),
                                  "confidence": rec.get("confidence"),
                                  "basis": rec.get("basis"),
                                  "reusable": True}))
        else:
            hits.append(_hit(
                "recommendation", str(rec.get("strategy_id") or "unknown"),
                "structural",
                {"strategy_id": rec.get("strategy_id"),
                 "evidence": rec.get("evidence"),
                 "expected": copy.deepcopy(rec.get("expected") or {}),
                 "confidence": rec.get("confidence"),
                 "risk_warnings": list(rec.get("risk_warnings") or []),
                 "basis": rec.get("basis"),
                 "evidence_refs": list(rec.get("evidence_refs") or [])}))
    return hits


def _semantic_hits(vector_recall: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for row in vector_recall.get("execution_evidence") or []:
        hits.append(_hit("execution_evidence", str(row.get("execution_id")),
                         "semantic", row))
    for row in vector_recall.get("strategic_knowledge") or []:
        hits.append(_hit("strategic_knowledge", str(row.get("entry_id")),
                         "semantic", row))
    return hits


def dedupe_evidence(hits: Sequence[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
    """Collapse repeated hits by identity, keeping every channel.

    One memory hit by two channels is ONE piece of evidence with two
    discovery paths — counting it twice would inflate its apparent support.
    Versions are collected rather than overwritten: two channels reporting
    different versions of one id is a conflict, reported as such, not
    silently resolved.
    """
    order: List[str] = []
    merged: Dict[str, Dict[str, Any]] = {}
    duplicates: List[Dict[str, Any]] = []
    for hit in hits:
        identity = hit["identity"]
        if identity not in merged:
            entry = copy.deepcopy(hit)
            entry["versions"] = [hit["version"]]
            entry["duplicate_count"] = 0
            merged[identity] = entry
            order.append(identity)
            continue
        entry = merged[identity]
        entry["duplicate_count"] += 1
        if hit["version"] not in entry["versions"]:
            entry["versions"].append(hit["version"])
        for channel in hit["channels"]:
            if channel not in entry["channels"]:
                entry["channels"].append(channel)
        duplicates.append({"identity": identity,
                           "channel": hit["channels"][0],
                           "note": "same evidence already carried by "
                                   f"{entry['channels']}"})
    items = [merged[i] for i in order]
    conflicts = [i for i in items if len(i["versions"]) > 1]
    for item in items:
        if len(item["versions"]) > 1:
            item["version_conflict"] = (
                "the same evidence id was reported under different versions "
                f"({item['versions']}): the channels disagree about which "
                "content they surfaced")
    execution_tasks = {str(i["content"].get("task_id"))
                       for i in items if i["layer"] == "execution_evidence"}
    summary = {
        "hits_seen": len(hits),
        "hits_kept": len(items),
        "duplicates_collapsed": len(duplicates),
        "duplicates": duplicates[:20],
        "version_conflicts": [i["identity"] for i in conflicts],
        "distinct_tasks_in_execution_hits": len(execution_tasks),
        "note": ("one memory hit by several channels is ONE piece of "
                 "evidence: support is never inflated by counting channels, "
                 "and repeat runs of one task_id are not independent tasks"),
    }
    return items, summary


@dataclass
class RetrievalView:
    """The two retrieval channels, their hits, and their honest statuses."""

    channels_run: List[str] = field(default_factory=list)
    hits: List[Dict[str, Any]] = field(default_factory=list)
    structural: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    deduplication: Dict[str, Any] = field(default_factory=dict)
    degraded: List[Dict[str, str]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: The task version this retrieval was produced for. Reuse is refused
    #: when it disagrees with the current task version.
    task_digest: Optional[str] = None
    recall_digest: Optional[str] = None

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    def evidence_classes(self) -> Dict[str, int]:
        """How many pieces of evidence of each class this view carries."""
        counts: Dict[str, int] = {}
        for hit in self.hits:
            key = str(hit.get("evidence_class"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels_run": list(self.channels_run),
            "n_hits": self.n_hits,
            "hits": copy.deepcopy(self.hits),
            "structural": copy.deepcopy(self.structural),
            "semantic": copy.deepcopy(self.semantic),
            "deduplication": copy.deepcopy(self.deduplication),
            "degraded": [dict(d) for d in self.degraded],
            "missing": list(self.missing),
            "notes": list(self.notes),
            "task_digest": self.task_digest,
            "recall_digest": self.recall_digest,
            "evidence_classes": self.evidence_classes(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalView":
        data = data or {}
        return cls(
            channels_run=[str(c) for c in (data.get("channels_run") or [])],
            hits=copy.deepcopy(list(data.get("hits") or [])),
            structural=copy.deepcopy(dict(data.get("structural") or {})),
            semantic=copy.deepcopy(dict(data.get("semantic") or {})),
            deduplication=copy.deepcopy(dict(data.get("deduplication") or {})),
            degraded=[dict(d) for d in (data.get("degraded") or [])],
            missing=[str(m) for m in (data.get("missing") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
            task_digest=(str(data["task_digest"])
                         if data.get("task_digest") else None),
            recall_digest=(str(data["recall_digest"])
                           if data.get("recall_digest") else None),
        )


def build_retrieval_view(recall_result: Dict[str, Any], *,
                         task_digest: Optional[str] = None,
                         top_k: int = 5,
                         include_unverified: bool = False) -> RetrievalView:
    """Turn ONE existing recall result into the context's retrieval view.

    Nothing new is retrieved here: the two existing channels are read as they
    are, their hits are bounded, deduplicated by identity, and each is
    labelled with its evidence class and reuse verdict. The channels are
    reported separately — no blended score is computed, because a discovery
    signal and an applicability verdict are different claims.
    """
    result = dict(recall_result or {})
    view = RetrievalView(task_digest=task_digest)
    view.recall_digest = _stable_digest(result)
    notes: List[str] = []

    recommendations = list(result.get("recommendations") or [])
    view.structural = {
        "status": "ok",
        "recommendations": _bounded(recommendations, top_k, notes,
                                    "structural recommendations"),
        "available_solver_families": list(
            result.get("available_solver_families") or []),
        "solver_advisories": list(result.get("solver_advisories") or []),
        "coupling_warnings": list(result.get("coupling_warnings") or []),
        "note": ("the structural channel decides APPLICABILITY from the "
                 "profile; its scores are evidence-based estimates, not "
                 "measurements"),
    }
    view.channels_run.append("structural")

    vector = result.get("vector_recall")
    if vector:
        vector = dict(vector)
        view.semantic = {
            "status": "ok",
            "backend": copy.deepcopy(vector.get("backend") or {}),
            "execution_evidence": _bounded(
                list(vector.get("execution_evidence") or []), top_k, notes,
                "semantic execution hits"),
            "strategic_knowledge": _bounded(
                list(vector.get("strategic_knowledge") or []), top_k, notes,
                "semantic knowledge hits"),
            "stale_indexed": copy.deepcopy(vector.get("stale_indexed") or {}),
            "unindexed": copy.deepcopy(vector.get("unindexed") or {}),
            "note": ("similarity is a DISCOVERY signal only; it is never a "
                     "quality, cost or risk estimate, and a cross-cell hit "
                     "never enters the target cell's statistics"),
        }
        for layer, payload in (vector.get("degraded_layers") or {}).items():
            view.degraded.append({
                "part": f"retrieval.semantic.{layer}",
                "reason": str(payload.get("reason") or "layer unusable")})
        if not include_unverified:
            view.notes.append(
                "unverified candidates are NOT surfaced by default; a "
                "candidate is not knowledge until admission verification "
                "passes")
        view.channels_run.append("semantic")
    elif result.get("degraded"):
        reason = str((result.get("degraded") or {}).get("reason")
                     or "unknown reason")
        view.semantic = {"status": "degraded", "reason": reason,
                         "execution_evidence": [],
                         "strategic_knowledge": []}
        view.degraded.append({"part": "retrieval.semantic", "reason": reason})
        view.missing.append(
            "semantic retrieval: did NOT run — the structural channel alone "
            "is present, and its result must not be read as 'no similar "
            "memory exists'")
    else:
        view.semantic = {"status": "absent"}
        view.missing.append(
            "semantic retrieval: no result supplied at all (neither a run "
            "nor a stated reason)")

    hits = _structural_hits(view.structural["recommendations"])
    hits += _semantic_hits({"execution_evidence":
                            view.semantic.get("execution_evidence") or [],
                            "strategic_knowledge":
                            view.semantic.get("strategic_knowledge") or []})
    items, summary = dedupe_evidence(hits)
    view.hits = items
    view.deduplication = summary
    view.notes.extend(notes)
    if not items:
        view.notes.append(
            "no evidence was retrieved on any channel that ran; with a "
            "degraded channel this is NOT the same as 'nothing comparable "
            "exists'")
    return view


# ---------------------------------------------------------------------------
# capability version (config/model/tools/memory identity — NOT a level)
# ---------------------------------------------------------------------------


#: Keys that record WHEN something was read rather than WHAT it says. They
#: are excluded from the memory-content digest: a fresh read timestamp is not
#: a change of content, and letting one move the digest would make every
#: rebuild look like a knowledge revision.
_VOLATILE_CONTENT_KEYS = ("snapshot_at", "created_at", "last_consulted_at",
                          "last_consulted", "consulted_at", "as_of")


def _strip_volatile(value: Any) -> Any:
    """A copy of ``value`` with read timestamps removed (recursively)."""
    if isinstance(value, dict):
        return {str(k): _strip_volatile(v) for k, v in value.items()
                if k not in _VOLATILE_CONTENT_KEYS}
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(v) for v in value]
    return value


def memory_content_digest(*, knowledge: Optional[Dict[str, Any]] = None,
                          retrieval: Optional[Any] = None,
                          executions: Optional[Sequence[Any]] = None,
                          entries: Optional[Sequence[Any]] = None
                          ) -> Dict[str, Any]:
    """Digest of the memory CONTENT a context actually used.

    Counts alone are not a version: editing an entry's expected quality, its
    interval, its predicates, its applicability notes or its strategy actions
    changes the capability condition, and a digest built from ``len(...)``
    would report the same value before and after that revision. So this
    digests the decision-relevant FIELDS of the memory that was really
    consulted — the layered knowledge view, the full entries behind it (whose
    ``applicability`` / ``actions`` / ``risk_conditions`` the ``KnowledgeRef``
    subset deliberately omits), and the evidence the retrieval carried —
    while excluding read timestamps (which move on every read and mean
    nothing about content).

    Bounded by construction: it covers the entries of the scoped knowledge
    view and the hits the context carries, never the whole bank, so no global
    snapshot mechanism is introduced.
    """
    knowledge = dict(knowledge or {})
    digest_entries: List[Dict[str, Any]] = []
    for layer in ("verified", "legacy_unknown", "unverified"):
        for entry in knowledge.get(layer) or []:
            if not isinstance(entry, dict):
                continue
            digest_entries.append({"layer": layer,
                                   **_strip_volatile(entry)})
    # The full entries add the fields the KnowledgeRef subset omits — the
    # harness-written applicability notes, the strategy actions and the risk
    # conditions are all part of what the memory SAYS.
    for entry in entries or []:
        if isinstance(entry, dict):
            digest_entries.append(_strip_volatile(entry))
            continue
        digest_entries.append({
            "entry_id": getattr(entry, "entry_id", None),
            "strategy_id": getattr(entry, "strategy_id", None),
            "expected_quality_hat": getattr(entry, "expected_quality_hat",
                                            None),
            "quality_interval": list(getattr(entry, "quality_interval",
                                             ()) or ()),
            "predicates": copy.deepcopy(
                getattr(entry, "pattern", None) or {}),
            "applicability": list(getattr(entry, "applicability", None) or []),
            "risk_conditions": list(getattr(entry, "risk_conditions",
                                            None) or []),
            "actions": list(getattr(entry, "actions", None) or []),
            "status": getattr(entry, "status", None),
            "support_n": getattr(entry, "support_n", None),
            "verification_state": getattr(entry, "verification_state", None),
            "failure_prob": getattr(entry, "failure_prob", None),
        })
    digest_entries.sort(key=lambda e: (str(e.get("layer")),
                                       str(e.get("entry_id"))))

    hits: List[Dict[str, Any]] = []
    if retrieval is not None:
        for hit in getattr(retrieval, "hits", None) or []:
            hits.append({
                "identity": hit.get("identity"),
                "version": hit.get("version"),
                "evidence_class": hit.get("evidence_class"),
                "content": _strip_volatile(hit.get("content") or {}),
            })
        hits.sort(key=lambda h: str(h.get("identity")))

    execution_ids: List[Dict[str, Any]] = []
    for record in executions or []:
        if isinstance(record, dict):
            execution_ids.append(_strip_volatile(record))
        else:
            execution_ids.append({
                "execution_id": getattr(record, "execution_id", None),
                "task_id": getattr(record, "task_id", None),
                "strategy_id": getattr(record, "strategy_id", None),
                "task_text_digest": getattr(record, "task_text_digest", None),
            })
    execution_ids.sort(key=lambda e: str(e.get("execution_id")))

    content = {"knowledge": digest_entries, "retrieval_hits": hits,
               "executions": execution_ids}
    layer_counts: Dict[str, int] = {}
    for entry in digest_entries:
        key = str(entry.get("layer") or "entry")
        layer_counts[key] = layer_counts.get(key, 0) + 1
    # Only the digest and its COMPOSITION are returned, never the digested
    # content: the version identity must not become a second, unbounded copy
    # of the knowledge view (and an unverified candidate must not leak into
    # the context merely by being counted in a version).
    return {
        "digest": _stable_digest(content),
        "knowledge_entries": len(digest_entries),
        "knowledge_by_layer": layer_counts,
        "retrieval_hits": len(hits),
        "executions": len(execution_ids),
        "excluded_keys": list(_VOLATILE_CONTENT_KEYS),
        "note": ("the digest covers the decision-relevant CONTENT of the "
                 "memory actually consulted (entry fields — including "
                 "applicability and actions — and retrieved hits), with read "
                 "timestamps excluded: a knowledge revision moves it, a "
                 "re-read does not. Only the digest and its composition are "
                 "carried, never the digested content"),
    }


def capability_version(*, harness_config: Optional[Dict[str, Any]] = None,
                       provider: Optional[Dict[str, Any]] = None,
                       prompt_template_version: Optional[str] = None,
                       tools: Optional[Sequence[str]] = None,
                       knowledge_content: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """The VERSION identity of the capability condition.

    Four things that can move independently, kept apart: harness
    configuration, model/prompt, tools, and the memory content the evidence
    was read from. This is an identity, not a capability level — a content
    digest says WHICH memories were read and what they said, never how
    capable the harness is, and a fresh read timestamp is not a capability
    change.

    ``knowledge_content`` should be the result of
    :func:`memory_content_digest` (real content). A bare count mapping is
    accepted for backwards compatibility but is NOT a version — it cannot
    distinguish a knowledge revision from an unchanged entry.
    """
    knowledge_content = dict(knowledge_content or {})
    content_digest = knowledge_content.get("digest")
    if content_digest is None:
        content_digest = _stable_digest(knowledge_content)
        knowledge_content = {
            **knowledge_content,
            "digest": content_digest,
            "note": ("derived from counts only: this is NOT a content "
                     "version — it cannot distinguish a knowledge revision "
                     "from an unchanged entry"),
        }
    return {
        "harness_config_digest": _stable_digest(harness_config or {}),
        "provider": copy.deepcopy(provider or {}),
        "prompt_template_version": prompt_template_version,
        "tools": sorted(str(t) for t in (tools or [])),
        "knowledge_content": knowledge_content,
        "knowledge_content_digest": content_digest,
        "note": ("this is a version IDENTITY (what configuration, model, "
                 "tools and memory content the evidence was read under), not "
                 "a capability level; a content digest is not an ability "
                 "measurement and a read timestamp is not a change"),
    }


# ---------------------------------------------------------------------------
# the context
# ---------------------------------------------------------------------------


@dataclass
class PredictionContext:
    """The frozen, versioned input bundle of a prediction.

    Build it ONCE per decision and reuse it across candidates: the problem,
    the solving context, the capability evidence and the memory view are
    identical for every candidate of one comparison, and only the candidate's
    own configuration and scope change. Building a second context per
    candidate would let a bank that moved mid-decision contaminate the
    comparison.
    """

    context_id: str
    task_id: str
    task_digest: str
    episode_id: Optional[str] = None
    #: The version identity of the EFFECTIVE problem input (the task with
    #: the resolved CIR merged in — see ``effective_input_version``). The
    #: plain ``task_digest`` stays as the caller-supplied task's version,
    #: kept for provenance; identity checks that decide whether an
    #: artifact may be REUSED compare against THIS digest, because it is
    #: the one that follows the problem the prediction conditions on.
    effective_input_digest: str = ""
    snapshot_id: str = ""
    joint: JointProblemRepresentation = field(
        default_factory=lambda: JointProblemRepresentation("", ""))
    #: X — the solving context, frozen from the same snapshot.
    solving_context: Dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalView = field(default_factory=RetrievalView)
    #: Evidence about H (phase-1 contract), never a measured capability.
    capability: Dict[str, Any] = field(default_factory=dict)
    capability_version: Dict[str, Any] = field(default_factory=dict)
    #: External execution constraints (declared budget, consumption, tools).
    execution_constraints: Dict[str, Any] = field(default_factory=dict)
    #: The framework's structural knowledge-target proposal set, frozen at
    #: build time. A reused context must send THESE targets: re-deriving them
    #: would read banks that may have moved since, so the prediction would
    #: mix a frozen input with a current proposal.
    knowledge_targets: List[Dict[str, Any]] = field(default_factory=list)
    #: Measured reliability of past predictions, frozen at build time (it
    #: comes from the prediction LOG, so a later call would change it).
    reliability: Dict[str, Any] = field(default_factory=dict)
    #: The published EXPERIENCE calibration of the strategy-outcome
    #: service (M4), frozen at build time. Closed episodes only: an active
    #: episode never reads its own not-yet-closed feedback. Kept SEPARATE
    #: from ``reliability`` (the legacy knowledge-prediction table): a
    #: knowledge hit rate never proves OR prediction accuracy.
    strategy_calibration: Dict[str, Any] = field(default_factory=dict)
    #: The frozen X/B + coverage + harness-condition blocks of the snapshot
    #: this context was built from, so a reused context can rebuild a
    #: snapshot that is BYTE-EQUIVALENT to the frozen one instead of taking a
    #: fresh one (which would carry later progress).
    snapshot: Dict[str, Any] = field(default_factory=dict)
    #: Per strategy, the DISTINCT tasks whose executions supported that
    #: cell's statistics at build time. ``learning_needs`` counts independent
    #: tasks to decide whether a claim could form, so a reused context must
    #: read that count from here rather than recounting today's bank.
    cell_evidence: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    degraded: List[Dict[str, str]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    frozen: bool = False
    version: str = PREDICTION_CONTEXT_VERSION

    @staticmethod
    def new_id() -> str:
        return f"ctx_{uuid.uuid4().hex[:12]}"

    @property
    def n_evidence(self) -> int:
        return self.retrieval.n_hits

    def freeze(self) -> None:
        """Deep-copy every nested structure (idempotent)."""
        self.joint = copy.deepcopy(self.joint)
        self.solving_context = copy.deepcopy(self.solving_context)
        self.retrieval = copy.deepcopy(self.retrieval)
        self.capability = copy.deepcopy(self.capability)
        self.capability_version = copy.deepcopy(self.capability_version)
        self.execution_constraints = copy.deepcopy(self.execution_constraints)
        self.knowledge_targets = copy.deepcopy(self.knowledge_targets)
        self.reliability = copy.deepcopy(self.reliability)
        self.strategy_calibration = copy.deepcopy(self.strategy_calibration)
        self.snapshot = copy.deepcopy(self.snapshot)
        self.cell_evidence = copy.deepcopy(self.cell_evidence)
        self.sources = dict(self.sources)
        self.degraded = [dict(d) for d in self.degraded]
        self.missing = list(self.missing)
        self.notes = list(self.notes)
        self.frozen = True

    def evidence_classes(self) -> Dict[str, int]:
        """How many pieces of evidence of each class the context carries."""
        return self.retrieval.evidence_classes()

    def provider_view(self) -> Dict[str, Any]:
        """The fragment of this context a provider is actually given.

        Kept as its own method so the test that asserts "the new joint
        evidence really reached the provider" has ONE place to look, and so
        a stored context can be inspected without re-deriving anything.
        """
        return {
            "context_id": self.context_id,
            "context_version": self.version,
            "created_at": self.created_at,
            "task_id": self.task_id,
            "task_digest": self.task_digest,
            "effective_input_digest": self.effective_input_digest,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "joint_problem": self.joint.to_dict(),
            "solving_context": copy.deepcopy(self.solving_context),
            "retrieval_evidence": self.retrieval.to_dict(),
            "harness_capability": copy.deepcopy(self.capability),
            "capability_version": copy.deepcopy(self.capability_version),
            "execution_constraints": copy.deepcopy(self.execution_constraints),
            "strategy_outcome_calibration": copy.deepcopy(
                self.strategy_calibration),
            "sources": dict(self.sources),
            "degraded": [dict(d) for d in self.degraded],
            "missing": list(self.missing),
            "notes": list(self.notes),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_version": self.version,
            "context_id": self.context_id,
            "task_id": self.task_id,
            "task_digest": self.task_digest,
            "effective_input_digest": self.effective_input_digest,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "frozen": self.frozen,
            "joint": self.joint.to_dict(),
            "solving_context": copy.deepcopy(self.solving_context),
            "retrieval": self.retrieval.to_dict(),
            "capability": copy.deepcopy(self.capability),
            "capability_version": copy.deepcopy(self.capability_version),
            "execution_constraints": copy.deepcopy(self.execution_constraints),
            "knowledge_targets": copy.deepcopy(self.knowledge_targets),
            "reliability": copy.deepcopy(self.reliability),
            "strategy_calibration": copy.deepcopy(self.strategy_calibration),
            "snapshot": copy.deepcopy(self.snapshot),
            "cell_evidence": copy.deepcopy(self.cell_evidence),
            "sources": dict(self.sources),
            "degraded": [dict(d) for d in self.degraded],
            "missing": list(self.missing),
            "notes": list(self.notes),
            "n_evidence": self.n_evidence,
            "evidence_classes": self.evidence_classes(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionContext":
        if not isinstance(data, dict):
            raise ValueError("PredictionContext must be a JSON object")
        version = str(data.get("context_version", ""))
        if version != PREDICTION_CONTEXT_VERSION:
            raise UnsupportedContextVersion(version or "(absent)")
        if not data.get("context_id"):
            raise ValueError("PredictionContext.context_id is required")
        ctx = cls(
            context_id=str(data["context_id"]),
            task_id=str(data.get("task_id", "")),
            task_digest=str(data.get("task_digest", "")),
            effective_input_digest=str(data.get("effective_input_digest", "")
                                       or data.get("task_digest", "")),
            episode_id=data.get("episode_id"),
            snapshot_id=str(data.get("snapshot_id", "")),
            joint=JointProblemRepresentation.from_dict(data.get("joint")),
            solving_context=copy.deepcopy(
                dict(data.get("solving_context") or {})),
            retrieval=RetrievalView.from_dict(data.get("retrieval")),
            capability=copy.deepcopy(dict(data.get("capability") or {})),
            capability_version=copy.deepcopy(
                dict(data.get("capability_version") or {})),
            execution_constraints=copy.deepcopy(
                dict(data.get("execution_constraints") or {})),
            knowledge_targets=copy.deepcopy(
                list(data.get("knowledge_targets") or [])),
            reliability=copy.deepcopy(dict(data.get("reliability") or {})),
            strategy_calibration=copy.deepcopy(
                dict(data.get("strategy_calibration") or {})),
            snapshot=copy.deepcopy(dict(data.get("snapshot") or {})),
            cell_evidence=copy.deepcopy(dict(data.get("cell_evidence") or {})),
            sources={str(k): str(v) for k, v in
                     (data.get("sources") or {}).items()},
            degraded=[dict(d) for d in (data.get("degraded") or [])],
            missing=[str(m) for m in (data.get("missing") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
            created_at=float(data.get("created_at", time.time())),
            frozen=bool(data.get("frozen", False)),
            version=version,
        )
        ctx.freeze()
        return ctx


def context_identity_problems(context: Any, *,
                              task_id: Optional[str] = None,
                              task_digest: Optional[str] = None,
                              episode_id: Optional[str] = None,
                              allow_unknown_digest: bool = True,
                              effective_input_digest: Optional[str] = None
                              ) -> List[str]:
    """Reasons a context does NOT belong to the identity it is used for.

    Mirrors :func:`~or_harness.world_model.execution_window
    .window_identity_problems`: an expected value of ``None`` means "no
    expectation recorded" and is not a mismatch, while a context naming a
    DIFFERENT task, task version or episode always is.

    ``task_digest`` is the version check: a context built for an older
    version of the task describes a different problem, and reusing it would
    silently condition a prediction on inputs that no longer apply.
    ``allow_unknown_digest`` keeps a legacy context (built before digests
    existed) usable — its version cannot be established either way, which is
    different from a KNOWN disagreement.

    ``effective_input_digest`` (when supplied) is checked against the
    context's OWN effective-input digest — the version of the problem with
    the resolved CIR merged in. This is the check that lets a context built
    with an explicit ``cir`` be reused against the same task (the plain
    task digests differ, because the caller-supplied task JSON differs);
    the caller passes ``effective_input_version(task, cir)``. A context
    whose effective-input digest disagrees describes a different problem
    input, whatever the plain task digests say.
    """
    problems: List[str] = []
    label = getattr(context, "context_id", None) or "context"
    ctx_task = getattr(context, "task_id", None)
    ctx_digest = getattr(context, "task_digest", None)
    ctx_episode = getattr(context, "episode_id", None)
    if task_id is not None and ctx_task != task_id:
        problems.append(
            f"context {label!r} belongs to task {ctx_task!r}, not "
            f"{task_id!r}")
    if task_digest is not None:
        if ctx_digest and ctx_digest != task_digest:
            problems.append(
                f"context {label!r} was built for task version {ctx_digest!r} "
                f"but the task is now {task_digest!r}: the problem changed, "
                "so the frozen input no longer applies (build a new context "
                "rather than silently reusing the old one)")
        elif not ctx_digest and not allow_unknown_digest:
            problems.append(
                f"context {label!r} records no task version, so it cannot be "
                "confirmed as describing the current task")
    if effective_input_digest is not None:
        ctx_effective = getattr(context, "effective_input_digest", None)
        if ctx_effective and ctx_effective != effective_input_digest:
            problems.append(
                f"context {label!r} was built for effective problem input "
                f"{ctx_effective!r} but this request's effective input is "
                f"{effective_input_digest!r}: the problem the prediction "
                "would condition on differs (a different CIR is a different "
                "problem input, even when the coupling numbers agree) — "
                "build a new context instead")
    if episode_id is not None and ctx_episode != episode_id:
        problems.append(
            f"context {label!r} belongs to episode {ctx_episode!r}, not "
            f"{episode_id!r}")
    return problems


def retrieval_reuse_problems(retrieval: Any, *,
                             task_digest: Optional[str] = None
                             ) -> List[str]:
    """Reasons a supplied recall result may NOT be reused as frozen evidence.

    The rule the phase requires: an externally supplied result must be
    confirmable as belonging to the CURRENT task version. A result with no
    recorded version, or one recorded for another version, is refused rather
    than dressed up as aligned evidence.
    """
    problems: List[str] = []
    recorded = getattr(retrieval, "task_digest", None)
    if task_digest is None:
        return problems
    if not recorded:
        problems.append(
            "the supplied recall result records no task version, so it "
            "cannot be confirmed as describing the current task; build the "
            "context instead of passing a result of unknown provenance")
    elif recorded != task_digest:
        problems.append(
            f"the supplied recall result was produced for task version "
            f"{recorded!r} but the task is now {task_digest!r}: it is not "
            "evidence about this problem")
    return problems


def build_context(
        *, task: Dict[str, Any], profile: Optional[ProblemProfile],
        snapshot: Any, recall_result: Optional[Dict[str, Any]],
        derivation: Optional[Dict[str, Any]] = None,
        cir: Any = None,
        math_declared: Optional[Dict[str, Any]] = None,
        capability: Optional[HarnessCapabilityEvidence] = None,
        capability_version_block: Optional[Dict[str, Any]] = None,
        execution_constraints: Optional[Dict[str, Any]] = None,
        structural_recommendations: Optional[Sequence[Dict[str, Any]]] = None,
        knowledge_targets: Optional[Sequence[Any]] = None,
        reliability: Optional[Dict[str, Any]] = None,
        strategy_calibration: Optional[Dict[str, Any]] = None,
        cell_evidence: Optional[Dict[str, Any]] = None,
        cir_source: Optional[str] = None,
        retrieval_bounding: Optional[Dict[str, Any]] = None,
        top_k: int = 5, include_unverified: bool = False,
        context_id: Optional[str] = None,
        created_at: Optional[float] = None) -> PredictionContext:
    """Assemble a :class:`PredictionContext` from already-gathered pieces.

    A pure assembly step: every input is something the caller already has
    (a frozen snapshot, one recall result, the profile and its derivation
    report, the phase-1 capability evidence, the structural knowledge
    targets and the measured reliability). Nothing is executed, called or
    induced here — that separation is what makes "the context build performs
    no model call" checkable.

    The knowledge targets and the reliability table are frozen INTO the
    context on purpose: both come from live banks/logs, and a reused context
    must send the values as they stood at build time rather than re-deriving
    them from banks that may have moved since.
    """
    from or_harness.world_model.state import task_text_digest

    digest = task_text_digest(task)
    joint = build_joint_representation(
        task, profile=profile, derivation=derivation, cir=cir,
        math_declared=math_declared, cir_source=cir_source)
    snapshot_id = str(getattr(snapshot, "snapshot_id", "") or "")
    problem_state = dict(getattr(snapshot, "problem_state", None) or {})
    solving_context = {
        "task_progress": copy.deepcopy(
            getattr(snapshot, "task_progress", None) or {}),
        "budget_state": copy.deepcopy(
            getattr(snapshot, "budget_state", None) or {}),
        "note": ("X is the episode's accumulated solving context frozen "
                 "BEFORE this prediction; a hypothetical snapshot's progress "
                 "never appears here"),
    }
    if recall_result is None:
        recall_result = {"recommendations": list(
            structural_recommendations or [])}
    retrieval = build_retrieval_view(recall_result, task_digest=digest,
                                     top_k=top_k,
                                     include_unverified=include_unverified)
    if retrieval_bounding:
        # The retrieval was bounded to a frozen moment. That is a fact about
        # the evidence set, so it is recorded ON the view rather than only
        # beside it — otherwise a reader would see the bounded hits without
        # knowing anything was excluded.
        retrieval.notes.append(str(retrieval_bounding.get("note") or ""))
        retrieval.structural["retrieval_bounding"] = copy.deepcopy(
            retrieval_bounding)
        if retrieval_bounding.get("rebuilt_from") == "none_saved":
            retrieval.missing.append(
                "retrieval: no historical retrieval was saved with the "
                "supplied snapshot, so none was rebuilt (the evidence set is "
                "empty because it was not recorded, not because nothing "
                "existed)")
        if retrieval_bounding.get("dropped"):
            retrieval.degraded.append({
                "part": "retrieval.bounding",
                "reason": (f"{retrieval_bounding['dropped']} memory item(s) "
                           "were created after the frozen moment and were "
                           "excluded from this context")})
    if not recall_result.get("vector_recall") \
            and not recall_result.get("degraded"):
        retrieval.missing.append(
            "retrieval: no recall result was gathered for this context; the "
            "evidence set is EMPTY BY OMISSION, not because nothing exists")

    evidence = capability or HarnessCapabilityEvidence()
    version_block = capability_version_block or capability_version()
    frozen_targets = [t.to_dict() if hasattr(t, "to_dict") else dict(t)
                      for t in (knowledge_targets or [])]
    ctx = PredictionContext(
        context_id=context_id or PredictionContext.new_id(),
        task_id=str(task.get("task_id", "")),
        task_digest=digest,
        effective_input_digest=effective_input_version(task, cir),
        episode_id=getattr(snapshot, "episode_id", None),
        snapshot_id=snapshot_id,
        joint=joint,
        solving_context=solving_context,
        retrieval=retrieval,
        capability=evidence.to_dict(),
        capability_version=version_block,
        execution_constraints=copy.deepcopy(execution_constraints or {}),
        knowledge_targets=frozen_targets,
        reliability=copy.deepcopy(reliability or {}),
        strategy_calibration=copy.deepcopy(strategy_calibration or {}),
        snapshot=snapshot_conditions(snapshot),
        cell_evidence=copy.deepcopy(cell_evidence or {}),
        sources={
            "snapshot": snapshot_id or "none",
            "task_version": digest,
            "problem_state_version": str(problem_state.get("task_digest")
                                         or "unknown"),
            "profile": joint.sources.get("profile", "none"),
            "retrieval": ",".join(retrieval.channels_run) or "none",
        },
        degraded=list(retrieval.degraded),
        missing=list(joint.missing) + list(retrieval.missing),
        notes=list(joint.notes) + list(retrieval.notes),
        created_at=created_at if created_at is not None else time.time(),
    )
    if not snapshot_id:
        ctx.missing.append(
            "snapshot: the context was built without a frozen snapshot, so "
            "X and B are absent rather than empty")
    if not frozen_targets:
        ctx.notes.append(
            "no structural knowledge target was proposed for this context: "
            "the frozen proposal set is empty (a first-class answer, not an "
            "error)")
    ctx.freeze()
    return ctx


def snapshot_conditions(snapshot: Any) -> Dict[str, Any]:
    """The frozen condition blocks of a snapshot, by value.

    Only the pieces a prediction is conditioned on — the labelled task
    progress (X), the budget state (B), the layered coverage view and the
    harness condition (H) — plus the identity needed to rebuild an
    equivalent snapshot. Read timestamps and the task payload are excluded:
    they are not prediction conditions, and the joint representation already
    carries the problem's own content.
    """
    return {
        "task_id": str(getattr(snapshot, "task_id", "") or ""),
        "episode_id": getattr(snapshot, "episode_id", None),
        "snapshot_id": str(getattr(snapshot, "snapshot_id", "") or ""),
        "task_progress": copy.deepcopy(
            getattr(snapshot, "task_progress", None) or {}),
        "budget_state": copy.deepcopy(
            getattr(snapshot, "budget_state", None) or {}),
        "coverage": copy.deepcopy(getattr(snapshot, "coverage", None) or {}),
        "harness_state": copy.deepcopy(
            getattr(snapshot, "harness_state", None) or {}),
    }


def snapshot_from_context(context: "PredictionContext") -> Any:
    """Rebuild the FROZEN snapshot a reused context was conditioned on.

    This is the fix for "reusing a context still took a fresh snapshot": the
    rebuilt object carries the frozen snapshot's own id and condition blocks,
    so the prediction request's ``state`` is the frozen X/B rather than the
    episode's current progress, and the recorded ``input_snapshot_id``
    matches the context's snapshot.

    The object is NOT persisted: it is the frozen input being replayed, not a
    new state observation, and writing it would add a snapshot that never
    really happened. A context with no frozen snapshot returns a snapshot
    with an empty id — the honest "this context has no X/B" state.
    """
    from or_harness.world_model.state import BeliefSnapshot

    frozen = context.snapshot or {}
    return BeliefSnapshot.build(
        {"task_id": context.task_id},
        context.episode_id,
        harness_state=copy.deepcopy(frozen.get("harness_state") or {}),
        problem_state={"task_digest": context.task_digest},
        task_progress=copy.deepcopy(frozen.get("task_progress") or {}),
        budget_state=copy.deepcopy(frozen.get("budget_state") or {}),
        coverage=copy.deepcopy(frozen.get("coverage") or {}),
        snapshot_id=(str(frozen.get("snapshot_id") or "")
                     or context.snapshot_id or None),
        created_at=context.created_at,
    )


def capability_evidence_with_sources(
        *, knowledge: Dict[str, Any], coverage: Optional[Dict[str, Any]],
        experience: Optional[Dict[str, Any]],
        tools: Optional[Dict[str, Any]],
        retrieval_view: Optional[RetrievalView] = None,
        reliability: Optional[Dict[str, Any]] = None,
        recorded_choices: Optional[Dict[str, Any]] = None
        ) -> HarnessCapabilityEvidence:
    """Phase-1 capability evidence, plus what THIS phase can really observe.

    The phase-1 mapping is reused unchanged (it is the conservative baseline:
    knowledge references and tool availability are ``indirect_evidence``, and
    ``W_OR`` / ``Pi`` / ``R`` stay ``no_evidence``). This function adds only
    observations that genuinely exist:

    - ``R`` — the retrieval channels that actually ran, with their
      applicability labels. Indirect: running a channel and labelling
      applicability is not evidence that a transfer succeeded;
    - ``W_OR`` — the measured reliability of past predictions, when the
      prediction log has resolved samples. Indirect: a track record is not a
      claim of accuracy;
    - ``Pi`` — recorded strategy choices and their outcomes. Indirect: a
      recorded choice is not evidence that the choice was optimal.

    Nothing is invented to fill the five sources, and no composite score is
    produced.
    """
    evidence = capability_evidence_from_legacy_harness_state(
        {"knowledge": knowledge,
         "experience": experience or {},
         "tool_config": tools or {}},
        coverage=coverage)
    evidence.notes.append(
        "capability evidence assembled for a prediction context: every item "
        "is evidence ABOUT H with an explicit status, never a level")

    if retrieval_view is not None and retrieval_view.channels_run:
        r = evidence.source("r")
        r.status = "indirect_evidence"
        labels: Dict[str, int] = {}
        for hit in retrieval_view.hits:
            content = hit.get("content") or {}
            label = str(content.get("structural_match") or "unlabelled")
            labels[label] = labels.get(label, 0) + 1
        r.detail = {
            "channels_run": list(retrieval_view.channels_run),
            "hits": retrieval_view.n_hits,
            "evidence_classes": retrieval_view.evidence_classes(),
            "applicability_labels": labels,
        }
        r.notes.append(
            "retrieval channels RAN and their hits carry applicability "
            "labels; a hit count or a similarity is not evidence that a "
            "transfer succeeded")

    if reliability:
        resolved = {k: v for k, v in reliability.items()
                    if isinstance(v, dict) and v.get("value") is not None}
        if resolved:
            w = evidence.source("w_or")
            w.status = "indirect_evidence"
            w.detail = {"reliability_classes": copy.deepcopy(reliability)}
            w.notes.append(
                "measured reliability of PAST predictions: a track record is "
                "not a claim of present accuracy, and a class with too few "
                "resolved samples reports no value at all")

    if recorded_choices and recorded_choices.get("n_recorded"):
        pi = evidence.source("pi")
        pi.status = "indirect_evidence"
        pi.detail = copy.deepcopy(recorded_choices)
        pi.notes.append(
            "recorded strategy choices and their outcomes: a recorded choice "
            "is an OBSERVATION of what was decided, never evidence that the "
            "decision was optimal")

    problems = []
    for name in ("m", "w_or", "pi", "r", "t"):
        item = evidence.sources.get(name)
        if item is None:
            continue
        if item.status == "direct_evidence" and not item.evidence:
            problems.append(name)
    if problems:  # pragma: no cover - defensive: nothing here sets direct
        for name in problems:
            evidence.sources[name].status = "indirect_evidence"
            evidence.sources[name].notes.append(
                "downgraded from direct_evidence: no evidence reference "
                "supported the stronger claim")
    return evidence


# ---------------------------------------------------------------------------
# frozen per-strategy evidence (what a reused context must not re-count)
# ---------------------------------------------------------------------------


def cell_evidence_from_stats(stats: Any, profile: Optional[ProblemProfile],
                             strategies: Sequence[str]) -> Dict[str, Any]:
    """The supporting evidence per strategy, frozen at build time.

    ``learning_needs`` counts DISTINCT TASKS behind a cell to decide whether
    a claim could form, and it reads the cell's real records to do so. A
    reused context must use the count as it stood when the context was
    built — recounting today's bank would let evidence that arrived after the
    snapshot change a frozen prediction condition.

    Bounded on purpose: only the distinct-task count and the task ids are
    kept per strategy, not the records themselves. The count is the quantity
    the heuristic consumes; the ids make it checkable.
    """
    out: Dict[str, Any] = {}
    for strategy_id in strategies or []:
        if not strategy_id or profile is None:
            continue
        try:
            records = stats.evidence(profile, strategy_id)
        except Exception:
            out[strategy_id] = {"distinct_tasks": None,
                                "task_ids": [],
                                "note": "cell evidence unavailable at build "
                                        "time (unknown, never zero)"}
            continue
        task_ids = sorted({str(getattr(r, "task_id", "")) for r in records})
        out[str(strategy_id)] = {
            "distinct_tasks": len(task_ids),
            "task_ids": task_ids,
            "executions": len(records),
        }
    return out


def knowledge_targets_from_context(context: "PredictionContext",
                                   action_spec: Any) -> List[Any]:
    """The FROZEN knowledge targets for one candidate strategy.

    Reused contexts must send the proposal set as it stood at build time.
    When the strategy's targets were not frozen (the strategy was not in the
    context's candidate set), an empty list is returned with the reason
    recorded in ``context.missing`` — an honest "not proposed", never a
    silent re-derivation from today's banks.
    """
    from or_harness.world_model.knowledge import KnowledgeTarget

    strategy_id = str(getattr(action_spec, "strategy_id", None) or "")
    targets = [KnowledgeTarget.from_dict(t) for t in
               (context.knowledge_targets or [])
               if str((t or {}).get("strategy_id") or "") == strategy_id]
    if not targets:
        note = (f"frozen knowledge targets: none were proposed for strategy "
                f"{strategy_id or '(unnamed)'} when this context was built, "
                "so none are sent (the proposal is not re-derived from the "
                "current bank)")
        if note not in context.missing:
            context.missing.append(note)
    return targets
