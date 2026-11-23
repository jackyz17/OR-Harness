"""Semantic applicability vs deterministic machine conditions.

Design (stage 2):

- a rule's ``when`` paragraph is the ONE source of its full semantic
  applicability, authored by the induction agent. The framework does NOT
  decide whether an abstract semantic condition holds;
- the framework checks only the deterministic STRUCTURAL conditions it can
  evaluate (the ``family`` scope label and the coupling-cell intervals in
  ``predicates``). A pass is a NECESSARY machine check, never proof that the
  whole rule applies;
- an empty predicate set means "no machine-checkable structural condition",
  NOT "applies to everything";
- a different ``family`` label is NOT by itself a contradiction: a shared
  mathematical mechanism may transfer across scenario labels. Only a
  MEASURED structural value outside the rule's claimed cell is a confirmed
  conflict;
- a semantic hit is a DISCOVERY, never an adoption. The agent decides
  adoption in its normal solving flow (``used_entries`` / effect
  attribution), after reading ``when``.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    GROUPING_FEATURES,
    UNKNOWN_BUCKET,
    unsupported_predicate_keys,
)

#: Predicate keys this framework evaluates deterministically against a
#: profile. ``family`` is a SCOPE LABEL (weak, never a veto); the coupling
#: cells are structural values.
MACHINE_PREDICATE_KEYS = ("family",) + GROUPING_FEATURES

#: The states a machine check can return.
MACHINE_STATES = ("compatible", "partial", "unknown", "not_specified",
                  "conflict")

#: A recall hit is only ever a candidate for adoption. The actual adoption
#: decision is recorded by the agent's own used-entry facts.
ADOPTION_STATE_NOT_ADOPTED = "not_adopted"

_ADOPTION_NOTE = (
    "a retrieval hit or a machine-compatible structure is NOT adoption: "
    "the agent decides whether to use the rule after reading its `when`, "
    "and records the adoption through the existing used-entry / effect "
    "attribution facts")


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def rule_body(entry: Any) -> Dict[str, Any]:
    """The rule's ``when`` / ``do`` / ``expect``, read as recorded.

    New v2 rules carry exactly these three paragraphs. A legacy entry (no
    versioned rule body) is rendered from its OWN stored claim fields — the
    claim text, its conditions note and its method steps — never paraphrased
    and never filled in from a directory. Absent content stays absent.
    """
    if entry is None:
        return {"when": None, "do": None, "expect": None,
                "source": "absent", "predicates": {}}
    rule = getattr(entry, "rule", None) or {}
    body = rule.get("body") or {}
    if body:
        return {
            "when": _as_text(body.get("when")),
            "do": _as_text(body.get("do")),
            "expect": _as_text(body.get("expect")),
            "source": "rule",
            "predicates": copy.deepcopy(
                body.get("predicates") or getattr(entry, "predicates", {})),
        }
    claim = getattr(entry, "claim", None) or {}
    conditions = claim.get("conditions") or {}
    method = claim.get("method") or {}
    steps = [str(step) for step in (method.get("steps") or [])
             if str(step).strip()]
    actions = [str(action) for action in (getattr(entry, "actions", None)
                                          or []) if str(action).strip()]
    return {
        "when": _as_text(conditions.get("note")),
        "do": _as_text("\n".join(steps) if steps else "\n".join(actions)),
        "expect": _as_text(claim.get("text")),
        "source": "legacy_claim" if claim else "absent",
        "predicates": copy.deepcopy(getattr(entry, "predicates", {}) or {}),
    }


def _family_label(profile: Any, claimed: Any) -> Dict[str, Any]:
    actual = getattr(profile, "family", None) if profile is not None else None
    if claimed is None:
        return {"claim": None, "task": actual, "relation": "not_specified",
                "treated_as": "scope_label_not_machine_condition",
                "note": "the rule states no family label"}
    if actual is None:
        return {"claim": str(claimed), "task": None, "relation": "unknown",
                "treated_as": "scope_label_not_machine_condition",
                "note": ("the task family is unmeasured; a label difference "
                         "cannot be confirmed")}
    if str(actual) == str(claimed):
        return {"claim": str(claimed), "task": str(actual),
                "relation": "same",
                "treated_as": "scope_label_not_machine_condition"}
    return {
        "claim": str(claimed), "task": str(actual),
        "relation": "different",
        "treated_as": "scope_label_not_machine_condition",
        "note": ("the family LABELS differ. A label is not a structural "
                 "contradiction: a shared mathematical mechanism can still "
                 "transfer between scenarios, so the agent must assess "
                 "`when` rather than treat this as inapplicable"),
    }


def machine_applicability(profile: Any,
                          predicates: Optional[Dict[str, Any]]
                          ) -> Dict[str, Any]:
    """The deterministic machine check of one predicate set.

    Returns a copyable block: ``state``, ``machine_compatible``, the exact
    checks with their observed values, unknown fields, confirmed conflicts,
    the family-label relation, the semantic (agent-only) predicate keys and
    a readable ``reason``. Nothing here is a full applicability verdict.
    """
    predicates = dict(predicates or {})
    checks: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    unknown: List[str] = []
    machine_keys = [key for key in predicates
                    if key in MACHINE_PREDICATE_KEYS]
    semantic_keys = unsupported_predicate_keys(predicates)
    family = _family_label(profile, predicates.get("family"))
    if profile is None:
        return {
            "state": "unknown", "machine_compatible": False,
            "checks": [], "unknown_fields": ["profile: not supplied"],
            "conflicts": [], "semantic_predicate_keys": semantic_keys,
            "family_label": family,
            "reason": "no profile is available to evaluate machine conditions",
            "requires_agent_assessment": True,
            "note": ("machine conditions could not be checked; the full "
                     "applicability is the rule's `when`"),
        }
    if not predicates:
        return {
            "state": "not_specified", "machine_compatible": False,
            "checks": [], "unknown_fields": [], "conflicts": [],
            "semantic_predicate_keys": [],
            "family_label": family,
            "reason": ("no machine-checkable structural condition was "
                       "recorded: this is NOT unconditional applicability — "
                       "the full applicability is the rule's `when`, which "
                       "requires the agent's assessment"),
            "requires_agent_assessment": True,
            "note": ("empty machine conditions are not a match; they are "
                     "the absence of a machine-checkable condition"),
        }
    for dim in GROUPING_FEATURES:
        if dim not in predicates:
            continue
        claim = predicates[dim]
        value = getattr(profile, dim)
        if isinstance(claim, str):
            if claim != UNKNOWN_BUCKET:
                unknown.append(f"{dim} (unreadable claim value {claim!r})")
                checks.append({"field": dim, "claim": claim,
                               "observed": value, "result": "unknown"})
                continue
            if value is not None:
                conflicts.append({
                    "field": dim, "claim": UNKNOWN_BUCKET,
                    "observed": value,
                    "reason": (f"{dim}={value} is measured but the rule "
                               "claims only the '{UNKNOWN_BUCKET}' cell")})
                checks.append({"field": dim, "claim": claim,
                               "observed": value, "result": "conflict"})
            else:
                checks.append({"field": dim, "claim": claim,
                               "observed": value, "result": "satisfied"})
            continue
        try:
            low, high = (float(item) for item in claim)
        except (TypeError, ValueError):
            unknown.append(f"{dim} (claim is not a readable interval)")
            checks.append({"field": dim, "claim": copy.deepcopy(claim),
                           "observed": value, "result": "unknown"})
            continue
        if value is None:
            unknown.append(
                f"{dim} (rule covers [{low:.2f}, {high:.2f}], task value "
                "unmeasured)")
            checks.append({"field": dim,
                           "claim": copy.deepcopy(claim), "observed": None,
                           "result": "unknown"})
        elif not (low <= float(value) <= high):
            conflicts.append({
                "field": dim, "claim": [low, high], "observed": value,
                "reason": (f"{dim}={value} is measured OUTSIDE the rule's "
                           f"[{low:.2f}, {high:.2f}] cell")})
            checks.append({"field": dim,
                           "claim": copy.deepcopy(claim), "observed": value,
                           "result": "conflict"})
        else:
            checks.append({"field": dim,
                           "claim": copy.deepcopy(claim), "observed": value,
                           "result": "satisfied"})
    satisfied = sum(1 for check in checks if check["result"] == "satisfied")
    if conflicts:
        state = "conflict"
        reason = "; ".join(item["reason"] for item in conflicts)
    elif semantic_keys:
        # Legacy/agent-authored predicate keys the framework cannot
        # evaluate: the machine check is incomplete, so it never defaults to
        # compatible.
        state = "unknown"
        reason = ("machine conditions cannot be fully evaluated: predicate "
                  "key(s) only an agent can judge: "
                  + ", ".join(semantic_keys))
        unknown.extend(semantic_keys)
    elif unknown:
        state = "partial" if satisfied else "unknown"
        reason = "machine check incomplete: " + "; ".join(unknown)
    elif family.get("relation") == "different":
        state = "compatible"
        reason = family.get("note")
    else:
        state = "compatible"
        reason = None
    return {
        "state": state,
        "machine_compatible": state == "compatible",
        "checked_conditions": checks,
        "unknown_fields": unknown,
        "conflicts": conflicts,
        "semantic_predicate_keys": semantic_keys,
        "family_label": family,
        "reason": reason,
        "requires_agent_assessment": True,
        "note": ("machine conditions are deterministic checks only. A "
                 "compatible result is NECESSARY, never sufficient: the "
                 "full applicability is the rule's `when` and requires the "
                 "agent's assessment"),
    }


def semantic_applicability(entry: Any) -> Dict[str, Any]:
    """The rule's semantic applicability surface: its ``when`` paragraph.

    The framework supplies the TEXT and states that the decision is the
    agent's. It never judges the sentence.
    """
    body = rule_body(entry)
    return {
        "state": "requires_agent_assessment",
        "source": body.get("source"),
        "when": body.get("when"),
        "note": ("the framework does not decide whether this semantic "
                 "condition holds; read `when` and decide whether to adopt "
                 "the rule"),
    }


def applicability_annotations(profile: Any, entry: Any) -> Dict[str, Any]:
    """The complete provider-facing applicability block for one entry.

    Layers are strict and separate:

    - ``retrieval``: how the entry was surfaced (added by the caller);
    - ``machine_applicability``: deterministic structural checks only;
    - ``semantic_applicability``: the rule's ``when``, for the agent;
    - ``adoption_state``: a recall/structural hit is NOT adopted.
    """
    return {
        "rule_body": rule_body(entry),
        "machine_applicability": machine_applicability(
            profile, getattr(entry, "predicates", None)),
        "semantic_applicability": semantic_applicability(entry),
        "requires_agent_assessment": True,
        "adoption_state": ADOPTION_STATE_NOT_ADOPTED,
        "adoption_note": _ADOPTION_NOTE,
    }


def machine_matched(profile: Any, predicates: Optional[Dict[str, Any]]
                    ) -> bool:
    """True only when the machine conditions are CONFIRMED compatible."""
    return bool(machine_applicability(profile, predicates)
                ["machine_compatible"])


def possibly_applicable(profile: Any, predicates: Optional[Dict[str, Any]]
                        ) -> bool:
    """A retrieval-side gate: an entry is kept unless a CONFLICT is confirmed.

    ``not_specified`` and ``unknown`` are kept for discovery with their
    explicit state; only a measured structural contradiction excludes the
    entry from the structural channel. Semantic relevance therefore still
    reaches the agent.
    """
    return machine_applicability(profile, predicates)["state"] != "conflict"
