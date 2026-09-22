"""Coupling-Aware Intermediate Representation (CIR).

A lightweight, domain-general structured representation of *how* the
components of an optimization problem interact.  It is intentionally NOT a
scalar profile — the primary representation is an explicit set of entities,
decisions, constraints, and relations.  Scalar coupling scores may later be
derived from this structure, but they are summaries, never substitutes.

Design constraints honoured here:

* **CIR exists before the canonical model.**  The outer harness agent
  produces a CIR from the natural-language task description *before* writing
  the GAMS-style ``model`` field.  The model may later cross-check the CIR,
  but is never required to create one.

* **Co-occurrence is structural evidence only.**  Constraint-variable
  co-occurrence can produce generic ``depends_on`` edges, but a semantic
  relation (``shares_resource``, ``competes_for``, …) is only emitted when
  additional entity/constraint semantics support the upgrade.

* **Domain-general.**  Every ``kind`` / ``type`` field is a free-form string.
  The core schema never hard-codes supply-chain-specific vocabulary.

* **Zero runtime LLM.**  The harness agent does semantic extraction; this
  module only validates, infers deterministically, and renders guidance.

The module is pure-stdlib, has no dependency on the rest of ``or_harness``
(except an optional import of :class:`~or_harness.profiling.model_syntax
.ParsedModel` for structural inference), and is fully JSON-round-trippable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Shape contract (fail-closed)
# ---------------------------------------------------------------------------

#: The ONLY keys a CIR object may carry.  ``issues`` is included because
#: :meth:`CouplingAwareIR.to_dict` emits it, so a round trip must parse.
_CIR_KEYS: Tuple[str, ...] = ("entities", "decisions", "constraints",
                              "relations", "coupling_groups", "issues")
#: The list-valued keys (``issues`` is excluded: it is a validation report,
#: not a structural input, and a caller may legitimately drop it).
_CIR_LIST_KEYS: Tuple[str, ...] = _CIR_KEYS[:-1]
#: Scalar coupling dimensions.  These are NOT CIR keys: they belong in
#: ``annotations.coupling``.  Named here only so a misplaced scalar gets a
#: hint that says where it should have gone.
_SCALAR_KEYS: Tuple[str, ...] = ("semantic_coupling", "resource_coupling",
                                 "temporal_coupling", "route_complexity")

#: Environment escape hatch.  ``OR_CIR_STRICT=0`` downgrades the POLICY
#: checks (unknown keys, empty CIR) so a legacy task that carries extra keys
#: keeps loading.  Structural checks are never downgraded — a non-object or a
#: wrongly typed list cannot be deserialized at all, so pretending to accept
#: it would only move the failure somewhere less legible.
CIR_STRICT_ENV = "OR_CIR_STRICT"


def cir_strict() -> bool:
    """Whether the CIR policy checks are enforced (see :data:`CIR_STRICT_ENV`)."""
    raw = os.environ.get(CIR_STRICT_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


class CIRFormatError(ValueError):
    """A CIR payload does not have the documented shape.

    Carries the offending key(s) plus a repair hint, so an agent that wrote
    the CIR can fix it in one turn instead of guessing.  ``str(exc)`` is the
    detail AND the hint, so even a caller that only logs the message gets an
    actionable sentence.
    """

    def __init__(self, kind: str, detail: str, hint: str, **ctx: Any) -> None:
        super().__init__(f"{detail} {hint}")
        self.kind = kind
        self.detail = detail
        self.hint = hint
        self.ctx = ctx

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "hint": self.hint, **self.ctx}


def _nested_hint(unknown: List[str], data: Dict[str, Any]) -> Optional[str]:
    """A hint for ``{"cir": {...}}`` — the nesting mistake we actually hit."""
    nested = [k for k in unknown if isinstance(data[k], dict)
              and any(kk in data[k] for kk in _CIR_KEYS)]
    if not nested:
        return None
    return (f"The CIR looks NESTED under {nested!r}. Put its keys DIRECTLY "
            f"under 'coupling': {{\"entities\": [...], ...}} — NOT "
            f"{{\"cir\": {{...}}}}.")


def _scalar_hint(unknown: List[str]) -> Optional[str]:
    """A hint for scalar coupling values written where the CIR belongs."""
    scalar = [k for k in unknown if k in _SCALAR_KEYS]
    if not scalar:
        return None
    return (f"{scalar} are SCALAR coupling values, not CIR keys — they belong "
            f"in annotations.coupling, not task.coupling.")


def cir_shape_problems(data: Any, *, allow_empty: bool = False,
                       strict: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Every shape problem with a CIR payload, as structured dicts.

    Never raises. Used by :func:`_validate_shape` to decide what to reject,
    and by a caller that wants to LINT a payload instead (``OR_CIR_STRICT=0``
    downgrades the policy checks, and this is how the downgraded problems are
    still reported rather than silently swallowed).

    ``severity`` separates the two kinds:

    * ``structural`` — the payload cannot be deserialized at all (not an
      object, a list key holding a non-list). These are ALWAYS errors: there
      is no lenient reading of ``"entities": {...}``.
    * ``policy`` — the payload parses but violates the documented contract
      (unknown keys, an empty CIR). These are what ``OR_CIR_STRICT=0``
      downgrades.
    """
    if strict is None:
        strict = cir_strict()
    problems: List[Dict[str, Any]] = []

    def add(kind: str, detail: str, hint: str, severity: str,
            **ctx: Any) -> None:
        problems.append({"severity": severity, "kind": kind, "detail": detail,
                         "hint": hint, **ctx})

    if not isinstance(data, dict):
        add("not_object",
            f"CIR must be a JSON object, got {type(data).__name__}.",
            'Wrap it as {"entities": [...], "decisions": [...], '
            '"relations": [...]}.', "structural")
        return problems

    unknown = [k for k in data if k not in _CIR_KEYS]
    if unknown:
        add("unknown_keys", f"Unknown CIR key(s): {unknown}.",
            _nested_hint(unknown, data) or _scalar_hint(unknown)
            or f"Allowed keys: {list(_CIR_KEYS)}.", "policy",
            unknown_keys=unknown, allowed=list(_CIR_KEYS))

    for key in _CIR_LIST_KEYS:
        value = data.get(key)
        if value is not None and not isinstance(value, list):
            add("bad_type",
                f"'{key}' must be a list, got {type(value).__name__}.",
                f'Use "{key}": [ ... ].', "structural", key=key)

    if not allow_empty and not any(data.get(k) for k in _CIR_LIST_KEYS):
        add("empty_cir",
            "CIR present but carries no entities/decisions/constraints/"
            "relations.",
            "Fill in at least entities + decisions + relations, or omit the "
            "'coupling' field entirely if this task has no CIR.", "policy")

    if strict:
        return problems
    # Downgraded: the POLICY problems become lints; the STRUCTURAL ones stay
    # errors, because a payload that cannot be deserialized has no lenient
    # reading — downgrading them would only move the failure into a
    # nonsense TypeError deeper in the parser.
    for p in problems:
        if p["severity"] == "policy":
            p["severity"] = "lint"
    return problems


def _validate_shape(data: Any, *, allow_empty: bool = True,
                    strict: Optional[bool] = None) -> None:
    """Reject a malformed CIR payload, or return quietly.

    Structural problems (not an object, a list key holding a non-list) always
    raise: ``from_dict`` cannot deserialize them, so accepting them would
    either crash with a nonsense ``TypeError`` or silently parse to an EMPTY
    CIR — which is the failure mode this gate exists to kill.

    Policy problems (unknown keys, an empty CIR) raise only when ``strict``
    (default: :func:`cir_strict`); ``OR_CIR_STRICT=0`` downgrades them, and
    :func:`cir_shape_problems` is how the downgraded ones stay visible.
    """
    for problem in cir_shape_problems(data, allow_empty=allow_empty,
                                      strict=strict):
        if problem["severity"] == "lint":
            continue
        ctx = {k: v for k, v in problem.items()
               if k not in ("severity", "kind", "detail", "hint")}
        raise CIRFormatError(problem["kind"], problem["detail"],
                             problem["hint"], **ctx)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A business object in the problem (open-ended ``kind``).

    ``kind`` examples (non-exhaustive, never enumerated here): resource,
    product, site, period, route, machine, …  The schema accepts any string.
    """

    name: str
    kind: str = "other"
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(name=str(data["name"]), kind=str(data.get("kind", "other")),
                   attrs=dict(data.get("attrs") or {}))


@dataclass
class Decision:
    """A decision variable or decision group in the problem."""

    name: str
    kind: str = "other"
    indexes: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "indexes": list(self.indexes), "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(name=str(data["name"]), kind=str(data.get("kind", "other")),
                   indexes=list(data.get("indexes") or []),
                   attrs=dict(data.get("attrs") or {}))


@dataclass
class ConstraintRef:
    """A reference to a constraint or business rule."""

    id: str
    kind: str = "other"
    expr: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "expr": self.expr,
                "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintRef":
        return cls(id=str(data["id"]), kind=str(data.get("kind", "other")),
                   expr=str(data.get("expr", "")),
                   attrs=dict(data.get("attrs") or {}))


#: Evidence levels for a relation.
#:
#: ``structural``  — derived from constraint-variable co-occurrence only.
#: ``semantic``    — supported by entity/constraint semantics or agent declaration.
#: ``declared``    — directly asserted by the agent (subject to validation).
EVIDENCE_LEVELS: Tuple[str, ...] = ("structural", "semantic", "declared")


@dataclass
class Relation:
    """A directed relationship between two nodes (entities, decisions, or
    constraints).  ``type`` is a free-form string; common examples include
    ``uses_resource``, ``shares_resource``, ``competes_for``, ``precedes``,
    ``flows_to``, ``depends_on``, ``constrained_by`` …"""

    source: str
    target: str
    type: str = "depends_on"
    evidence: str = "structural"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "type": self.type, "evidence": self.evidence,
                "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(source=str(data["source"]), target=str(data["target"]),
                   type=str(data.get("type", "depends_on")),
                   evidence=str(data.get("evidence", "structural")),
                   detail=str(data.get("detail", "")))


@dataclass
class CouplingGroup:
    """A derived coupling structure — a recurring pattern that carries
    modeling implications.  ``type`` is a free-form string; common examples
    include ``shared_bottleneck``, ``route_convergence``,
    ``temporal_propagation_chain``, ``global_constraint``,
    ``cross_stage_coupling`` …"""

    type: str
    members: List[str] = field(default_factory=list)
    resource: Optional[str] = None
    implication: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "members": list(self.members),
                "resource": self.resource, "implication": self.implication}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CouplingGroup":
        return cls(type=str(data["type"]),
                   members=list(data.get("members") or []),
                   resource=data.get("resource"),
                   implication=str(data.get("implication", "")))


@dataclass
class CouplingIssue:
    """A validation issue, mirroring :class:`~or_harness.profiling.model_syntax
    .ModelIssue` in shape."""

    layer: str   # "L1" | "L2"
    code: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"layer": self.layer, "code": self.code, "detail": self.detail}


@dataclass
class CouplingAwareIR:
    """The top-level CIR object.

    All child lists are JSON-round-trippable.  The object is intentionally
    plain — no graph database, no embeddings, no GNN.
    """

    entities: List[Entity] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    constraints: List[ConstraintRef] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    coupling_groups: List[CouplingGroup] = field(default_factory=list)
    issues: List[CouplingIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no validation issues were found."""
        return not self.issues

    # -- node name registry ---------------------------------------------------

    def _node_names(self) -> Set[str]:
        names: Set[str] = set()
        for e in self.entities:
            names.add(e.name)
        for d in self.decisions:
            names.add(d.name)
        for c in self.constraints:
            names.add(c.id)
        return names

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "decisions": [d.to_dict() for d in self.decisions],
            "constraints": [c.to_dict() for c in self.constraints],
            "relations": [r.to_dict() for r in self.relations],
            "coupling_groups": [g.to_dict() for g in self.coupling_groups],
            "issues": [i.to_dict() for i in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *,
                  strict: Optional[bool] = None,
                  allow_empty: bool = True) -> "CouplingAwareIR":
        """Build a CIR from a JSON object.

        ``strict`` (default: :func:`cir_strict`) enforces the shape contract
        first: without it a misspelled or nested payload parses into an EMPTY
        CIR with no error, and every consumer downstream then treats a
        malformed problem as an uncoupled one.  ``allow_empty=False`` rejects
        a CIR that carries no structural content at all — appropriate for an
        agent-supplied CIR, never for the internal round trip (an empty CIR
        is a legitimate value that :meth:`to_dict` can produce).
        """
        _validate_shape(data, allow_empty=allow_empty, strict=strict)
        return cls(
            entities=[Entity.from_dict(d) for d in (data.get("entities") or [])],
            decisions=[Decision.from_dict(d) for d in (data.get("decisions") or [])],
            constraints=[ConstraintRef.from_dict(d)
                         for d in (data.get("constraints") or [])],
            relations=[Relation.from_dict(d)
                       for d in (data.get("relations") or [])],
            coupling_groups=[CouplingGroup.from_dict(d)
                             for d in (data.get("coupling_groups") or [])],
            issues=[CouplingIssue(layer=i.get("layer", "L1"),
                                  code=i.get("code", ""),
                                  detail=i.get("detail", ""))
                    for i in (data.get("issues") or [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))


def cir_from_task(task: Any, *, allow_empty: bool = False,
                  strict: Optional[bool] = None) -> Optional["CouplingAwareIR"]:
    """The CIR a task's own ``coupling`` field declares, or ``None``.

    THE single resolution point for the task-side CIR, so ``profile``,
    ``snapshot``, ``execute``, ``recall`` and the prediction context cannot
    disagree about whether a task has a CIR — or about whether the one it has
    is well-formed.

    Three outcomes, and the middle one is the point of this function:

    * no ``coupling`` key (or ``None``) -> ``None``: the task has no CIR,
      which is a normal state;
    * a well-formed object -> the parsed :class:`CouplingAwareIR`;
    * anything else -> :class:`CIRFormatError`.  A ``coupling`` field that is
      a string, a list, or an object of the wrong shape is NOT "no CIR": the
      caller MEANT to declare one, so silently reading it as absent would
      make a malformed problem indistinguishable from an uncoupled one.

    ``allow_empty`` defaults to False because this is the INPUT boundary: a
    task that carries ``"coupling": {}`` declared a CIR and supplied nothing,
    which is far more often a broken payload than an intentional one.  The
    internal deserializer (:meth:`CouplingAwareIR.from_dict`) defaults the
    other way, because an empty CIR is a value ``to_dict`` can legitimately
    produce and a round trip must survive it.
    """
    if not isinstance(task, dict) or task.get("coupling") is None:
        return None
    raw = task["coupling"]
    if isinstance(raw, CouplingAwareIR):
        return raw
    return CouplingAwareIR.from_dict(raw, strict=strict,
                                     allow_empty=allow_empty)


def coerce_cir(cir: Any, *, allow_empty: bool = False,
               strict: Optional[bool] = None) -> Optional["CouplingAwareIR"]:
    """A :class:`CouplingAwareIR` for *cir*, accepting a dict or an instance.

    ``profile_task`` takes the effective CIR from several callers; some hand
    it a parsed object and some a raw JSON object (a CLI ``--cir`` literal).
    Normalizing here means the shape gate sees BOTH forms, instead of the
    dict form slipping past it into ``coupling_from_cir``.
    """
    if cir is None or isinstance(cir, CouplingAwareIR):
        return cir
    return CouplingAwareIR.from_dict(cir, strict=strict,
                                     allow_empty=allow_empty)


# ---------------------------------------------------------------------------
# L1 / L2 validation (deterministic, no LLM)
# ---------------------------------------------------------------------------


def validate_cir(cir: CouplingAwareIR) -> CouplingAwareIR:
    """Validate *cir* in place and return it.

    **L1 — format**: required fields present, types correct, no duplicate
    node names.

    **L2 — referential integrity**: every ``relation.source`` and
    ``relation.target`` must resolve to a known entity, decision, or
    constraint.  Every ``coupling_group.member`` must likewise resolve.
    """
    cir.issues.clear()
    seen_names: Set[str] = set()

    # -- L1: entities ----------------------------------------------------------
    for e in cir.entities:
        if not e.name:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "entity with empty name"))
        elif e.name in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"entity name {e.name!r} duplicated"))
        else:
            seen_names.add(e.name)

    # -- L1: decisions ---------------------------------------------------------
    for d in cir.decisions:
        if not d.name:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "decision with empty name"))
        elif d.name in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"decision name {d.name!r} duplicated"))
        else:
            seen_names.add(d.name)

    # -- L1: constraints -------------------------------------------------------
    for c in cir.constraints:
        if not c.id:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "constraint with empty id"))
        elif c.id in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"constraint id {c.id!r} duplicated"))
        else:
            seen_names.add(c.id)

    # -- L1: relations ---------------------------------------------------------
    for r in cir.relations:
        if not r.source or not r.target:
            cir.issues.append(CouplingIssue("L1", "empty_endpoint",
                                            f"relation {r.type!r} has empty endpoint"))
        if r.evidence not in EVIDENCE_LEVELS:
            cir.issues.append(CouplingIssue("L1", "bad_evidence",
                                            f"relation {r.source}->{r.target} "
                                            f"has evidence={r.evidence!r}; "
                                            f"expected one of {EVIDENCE_LEVELS}"))

    # -- L2: referential integrity --------------------------------------------
    node_names = seen_names  # all declared node names

    for i, r in enumerate(cir.relations):
        if r.source and r.source not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_source",
                                            f"relation#{i} source {r.source!r} "
                                            "does not match any entity/decision/"
                                            "constraint"))
        if r.target and r.target not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_target",
                                            f"relation#{i} target {r.target!r} "
                                            "does not match any entity/decision/"
                                            "constraint"))

    for i, g in enumerate(cir.coupling_groups):
        for m in g.members:
            if m not in node_names:
                cir.issues.append(CouplingIssue("L2", "dangling_member",
                                                f"coupling_group#{i} member "
                                                f"{m!r} does not match any "
                                                "entity/decision/constraint"))
        if g.resource is not None and g.resource not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_resource",
                                            f"coupling_group#{i} resource "
                                            f"{g.resource!r} does not match "
                                            "any entity/decision/constraint"))

    return cir


# ---------------------------------------------------------------------------
# Deterministic structural inference
# ---------------------------------------------------------------------------

#: Constraint-expression keywords that hint at capacity / resource semantics.
#: Deliberately narrow: broad words like "demand", "max", "supply" are NOT
#: capacity evidence — "demand" is a consumption requirement, "max" matches
#: too many non-capacity patterns.
_CAPACITY_KEYWORDS = {"capacity", "limit", "cap", "budget", "resource",
                      "available"}
#: Constraint ``kind`` values that hint at capacity / resource semantics.
_CAPACITY_KINDS = {"capacity", "resource", "budget"}
#: Entity-kind values that hint at a resource.
_RESOURCE_KIND_HINTS = {"resource", "capacity", "machine", "crew", "vehicle",
                        "worker", "budget", "stock", "inventory"}


def _has_capacity_semantics(expr: str) -> bool:
    """Heuristic: does *expr* look like a capacity/limit constraint?"""
    lower = expr.lower()
    return any(kw in lower for kw in _CAPACITY_KEYWORDS)


def _constraint_capacityish(constraint: ConstraintRef) -> bool:
    """Capacity semantics from the constraint's declared kind OR expression."""
    return (constraint.kind.lower() in _CAPACITY_KINDS
            or _has_capacity_semantics(constraint.expr))


def _constraint_mentions_resource(expr: str, resource_name: str) -> bool:
    """Does *expr* reference *resource_name*?

    Comparison ignores whitespace/underscore/hyphen/slash separators so that
    "A-07 LDA" matches "cap_A07_LDA".  A resource that never appears in the
    constraint expression is NOT evidence that the constraint couples to it.
    """
    cleaned_resource = re.sub(r"[\s_\-/]+", "", resource_name.lower())
    if not cleaned_resource:
        return False
    cleaned_expr = re.sub(r"[\s_\-/]+", "", expr.lower())
    return cleaned_resource in cleaned_expr


def infer_structural_relations(
        cir: CouplingAwareIR,
        parsed_model: Optional[Any] = None) -> CouplingAwareIR:
    """Populate *cir.relations* with **structural-only** edges.

    When *parsed_model* (a :class:`ParsedModel`) is provided, constraint-
    variable co-occurrence produces generic ``depends_on`` edges between
    decisions that share a constraint.  These edges carry
    ``evidence="structural"`` — they are evidence, not semantic claims.

    A structural ``depends_on`` edge may be upgraded to ``uses_resource``
    only when ALL of the following hold (tight on purpose — the deterministic
    layer must not invent semantics):

    1. the target entity is resource-like (``kind`` in the resource hints);
    2. some CIR constraint has capacity semantics (capacity-ish ``kind`` or
       a narrow capacity keyword in its expression);
    3. that constraint's expression mentions the source decision;
    4. that constraint's expression also mentions the target resource
       (so the constraint is evidence the *target* is coupled, not just
       any resource-flavored constraint the source happens to appear in).

    All other semantic relations must come from the agent.  The method is
    idempotent: calling it twice does not duplicate edges.
    """
    existing: Set[Tuple[str, str, str, str]] = {
        (r.source, r.target, r.type, r.evidence) for r in cir.relations
    }

    # -- upgrade pass: agent-declared structural edges with semantic support --
    # Build lookup tables.
    entity_by_name: Dict[str, Entity] = {e.name: e for e in cir.entities}
    constraint_by_id: Dict[str, ConstraintRef] = {c.id: c for c in cir.constraints}

    for rel in cir.relations:
        if rel.evidence != "structural" or rel.type != "depends_on":
            continue
        # Can we upgrade?  Check target entity kind + source constraint.
        target_entity = entity_by_name.get(rel.target)
        if target_entity is None:
            continue
        if target_entity.kind.lower() not in _RESOURCE_KIND_HINTS:
            continue
        # Look for a constraint with capacity semantics that mentions BOTH
        # the source decision AND the target resource.
        for c in cir.constraints:
            if not _constraint_capacityish(c):
                continue
            if rel.source not in c.expr:
                continue
            if not _constraint_mentions_resource(c.expr, target_entity.name):
                continue
            rel.type = "uses_resource"
            rel.evidence = "semantic"
            rel.detail = (f"upgraded from structural co-occurrence: "
                          f"constraint {c.id} has capacity semantics and "
                          f"explicitly references resource {rel.target}")
            break

    # -- co-occurrence inference from parsed model ----------------------------
    if parsed_model is not None:
        _infer_from_parsed_model(cir, parsed_model, existing, entity_by_name,
                                 constraint_by_id)

    return cir


def _infer_from_parsed_model(
        cir: CouplingAwareIR,
        model: Any,
        existing: Set[Tuple[str, str, str, str]],
        entity_by_name: Dict[str, Entity],
        constraint_by_id: Dict[str, ConstraintRef]) -> None:
    """Add structural ``depends_on`` edges from constraint-variable
    co-occurrence in *model* (a :class:`ParsedModel`)."""

    var_sets = model.constraint_variable_sets()
    var_names = list(model.variables)

    for ci, (label, expr) in enumerate(model.constraints):
        # Find which variables co-occur in this constraint.
        vars_here = var_sets[ci] if ci < len(var_sets) else set()
        if len(vars_here) < 2:
            continue
        constraint_id = label or f"C{ci + 1}"
        # If the constraint is already in the CIR, use its id; otherwise
        # create a lightweight ConstraintRef.
        if constraint_id not in constraint_by_id:
            cir.constraints.append(ConstraintRef(
                id=constraint_id, kind="other", expr=expr))
            constraint_by_id[constraint_id] = cir.constraints[-1]
        # Add depends_on edges between pairs of co-occurring decisions.
        vars_list = sorted(vars_here)
        for i_a in range(len(vars_list)):
            for i_b in range(i_a + 1, len(vars_list)):
                va, vb = vars_list[i_a], vars_list[i_b]
                key = (va, vb, "depends_on", "structural")
                if key in existing:
                    continue
                existing.add(key)
                cir.relations.append(Relation(
                    source=va, target=vb, type="depends_on",
                    evidence="structural",
                    detail=f"co-occur in constraint {constraint_id}"))
    # Re-validate after inference.
    validate_cir(cir)


# ---------------------------------------------------------------------------
# Coupling-group derivation + modeling guidance
# ---------------------------------------------------------------------------

#: Mapping from coupling-group type to a modeling-implication template.
#: The type strings are examples, not an exhaustive enum — unknown types
#: pass through with a generic message.
_IMPLICATION_TEMPLATES: Dict[str, str] = {
    "shared_bottleneck": (
        "Multiple decisions ({members}) consume the same resource "
        "({resource}); ensure one aggregate capacity constraint covers "
        "all relevant decisions."),
    "route_convergence": (
        "Multiple routes ({members}) converge on a shared downstream "
        "stage ({resource}); do not treat route capacities independently "
        "after convergence."),
    "temporal_propagation_chain": (
        "Decisions ({members}) propagate across time via state "
        "({resource}); ensure state-transition / inventory linkage is "
        "modeled across periods."),
    "global_constraint": (
        "Many local decisions ({members}) are bounded by one global "
        "constraint ({resource}); preserve the global constraint during "
        "decomposition."),
    "cross_stage_coupling": (
        "Decisions ({members}) in different stages are linked via "
        "{resource}; model the inter-stage dependency explicitly."),
}


def derive_coupling_groups(cir: CouplingAwareIR) -> CouplingAwareIR:
    """Derive :class:`CouplingGroup` objects from the relation structure.

    Deliberately minimal: the deterministic layer detects exactly ONE pattern:

    * **shared_bottleneck** — two or more ``uses_resource`` relations point
      to the same resource entity.

    Other group types (``route_convergence``,
    ``temporal_propagation_chain``, ``global_constraint``,
    ``cross_stage_coupling``) are the agent's responsibility: the agent
    declares them (they are preserved and rendered as-is); the framework
    does not invent them.  The division of labor is:

    > **Agent extracts semantic structure; the framework validates it and
    > derives only this one structural pattern.**
    """
    existing_types = {g.type for g in cir.coupling_groups}

    # -- shared_bottleneck: group uses_resource edges by target ---------------
    resource_users: Dict[str, List[str]] = {}
    for r in cir.relations:
        if r.type == "uses_resource":
            resource_users.setdefault(r.target, []).append(r.source)

    for resource, users in resource_users.items():
        if len(users) < 2:
            continue
        gtype = "shared_bottleneck"
        members = sorted(users)
        # Avoid duplicating an agent-declared group.
        key = (gtype, resource)
        if any(g.type == gtype and g.resource == resource for g in cir.coupling_groups):
            continue
        cir.coupling_groups.append(CouplingGroup(
            type=gtype, members=members, resource=resource,
            implication=_render_implication(gtype, members, resource)))

    return cir


def _render_implication(gtype: str, members: List[str],
                        resource: Optional[str]) -> str:
    template = _IMPLICATION_TEMPLATES.get(gtype)
    if template is None:
        return (f"Coupling structure of type {gtype!r} detected among "
                f"{members}; review modeling implications.")
    return template.format(
        members=", ".join(members),
        resource=resource or "unknown")


def render_modeling_guidance(cir: CouplingAwareIR) -> List[Dict[str, Any]]:
    """Render coupling groups as structured modeling guidance for the agent.

    Each entry has ``type``, ``members``, ``resource``, and ``implication``
    — all inspectable, none hidden in embeddings or free-form prose.
    """
    guidance: List[Dict[str, Any]] = []
    for g in cir.coupling_groups:
        if not g.implication:
            g.implication = _render_implication(g.type, g.members, g.resource)
        guidance.append({
            "type": g.type,
            "members": list(g.members),
            "resource": g.resource,
            "implication": g.implication,
        })
    return guidance


# ---------------------------------------------------------------------------
# Scalar signature derivation (derived summaries, NOT the primary form)
# ---------------------------------------------------------------------------

#: Relation types that count as resource coupling for the scalar signature.
_RESOURCE_RELATION_TYPES = ("uses_resource", "shares_resource",
                            "competes_for")
#: Time-like index tokens (single-letter tokens match by equality only).
_TEMPORAL_INDEX_TOKENS = ("t", "time", "period", "stage", "day", "week",
                          "month", "hour", "slot", "shift")
#: Network-like index tokens (single-letter tokens match by equality only).
_NETWORK_INDEX_TOKENS = ("arc", "edge", "road", "link", "route", "leg", "node")


def _index_matches(index: str, tokens: Tuple[str, ...]) -> bool:
    low = index.lower()
    for tok in tokens:
        if len(tok) == 1:
            if low == tok:
                return True
        elif tok in low:
            return True
    return False


def coupling_from_cir(cir: CouplingAwareIR) -> Dict[str, Optional[float]]:
    """Derive the scalar coupling dimensions from the CIR structure.

    These are **derived summaries** for the ProblemSignature — the CIR
    itself remains the primary representation.  Operational definitions
    mirror the model-based ones:

    - ``resource_coupling``: fraction of decisions that are the source of at
      least one resource relation (``uses_resource`` / ``shares_resource`` /
      ``competes_for``).
    - ``temporal_coupling``: fraction of decisions with a time-like index
      (t/time/period/stage/day/...).
    - ``route_complexity``: fraction of decisions with a network-like index
      (arc/edge/road/link/route/...).
    - ``semantic_coupling``: never derived here — the CIR's relations ARE
      the semantic understanding; the scalar stays the harness's call so the
      grouping contract remains stable.

    Returns None for every dimension when there are no decisions.
    """
    n = len(cir.decisions)
    if n == 0:
        return {"resource_coupling": None, "temporal_coupling": None,
                "route_complexity": None, "semantic_coupling": None}

    decision_names = {d.name for d in cir.decisions}
    resource_linked = {r.source for r in cir.relations
                       if r.type in _RESOURCE_RELATION_TYPES
                       and r.source in decision_names}
    temporal_count = sum(1 for d in cir.decisions
                         if any(_index_matches(i, _TEMPORAL_INDEX_TOKENS)
                                for i in d.indexes))
    network_count = sum(1 for d in cir.decisions
                        if any(_index_matches(i, _NETWORK_INDEX_TOKENS)
                               for i in d.indexes))
    return {
        "resource_coupling": round(len(resource_linked) / n, 4),
        "temporal_coupling": round(temporal_count / n, 4),
        "route_complexity": round(network_count / n, 4),
        "semantic_coupling": None,
    }
