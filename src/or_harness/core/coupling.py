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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    def from_dict(cls, data: Dict[str, Any]) -> "CouplingAwareIR":
        if not isinstance(data, dict):
            raise ValueError("CIR must be a JSON object")
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
