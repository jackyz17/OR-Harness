"""Belief snapshots: the frozen pre-action state contract (M1).

A BeliefSnapshot is the harness's information state at one moment, frozen
BEFORE an action runs, so that later bank writes can never silently change
what the snapshot meant. It is the ``b_t`` of the partial-observability
model: facts, inferences, and unknowns about H (harness), P (problem), X
(task progress), and B (budget) — each key field carrying BOTH an
information-source label (``provenance``: observed vs agent_reported) and an
epistemic label (``epistemic``: fact / inferred / unknown). The two are
orthogonal: an agent's report can itself be an observation or an inference.

Freezing is enforced by defensive copying at every boundary: the snapshot
deep-copies nested structures at construction, ``to_dict`` returns fresh
objects, and ``from_dict`` rebuilds. Mutating the caller's originals, a
returned dict, or a loaded snapshot's nested fields never changes what is
persisted.

This is NOT a world model and NOT a knowledge bank: it is an index/log
substrate. The verified-knowledge view (``verified_knowledge_view``) layers
admission state over ``StrategicBank.matching`` — which matches predicates
only and never filters by verification state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from or_harness.core.schema import ProblemProfile

#: Information source: who produced this piece of the state.
PROVENANCE_LABELS = ("observed", "agent_reported")
#: Epistemic status: what kind of claim the piece is.
EPISTEMIC_LABELS = ("fact", "inferred", "unknown")

#: The maintenance scope marker for offline actions that belong to no task.
MAINTENANCE_TASK_ID = "__maintenance__"


def _stable_digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _labelled(value: Any, provenance: str = "observed",
              epistemic: str = "fact",
              evidence_ref: Optional[str] = None) -> Dict[str, Any]:
    """One labelled field: value + source + epistemic status + evidence."""
    if provenance not in PROVENANCE_LABELS:
        raise ValueError(f"provenance must be one of {PROVENANCE_LABELS}")
    if epistemic not in EPISTEMIC_LABELS:
        raise ValueError(f"epistemic must be one of {EPISTEMIC_LABELS}")
    return {
        "value": copy.deepcopy(value),
        "provenance": provenance,
        "epistemic": epistemic,
        "evidence_ref": evidence_ref,
    }


#: Task keys that carry problem semantics (kept in the snapshot's P when no
#: explicit task_ref exists, so a future model can recover WHAT was asked,
#: not just a hash). Bulky/derived artifacts are excluded. ``text`` is the
#: plain "the task is this sentence" field a caller naturally reaches for;
#: it is accepted alongside the structured keys so a task written that way
#: is never silently text-less.
_TASK_PAYLOAD_KEYS = ("task_id", "family", "description", "text",
                      "objective", "constraints", "spec", "requirements",
                      "business_rules", "data_ref")


def _task_payload(task: Dict[str, Any]) -> Dict[str, Any]:
    """The problem-relevant subset of a task payload (deep-copied)."""
    return {k: copy.deepcopy(task[k]) for k in _TASK_PAYLOAD_KEYS
            if task.get(k) is not None}


#: Keys whose CONTENT is the problem's own text, in reading order. A subset
#: of :data:`_TASK_PAYLOAD_KEYS` so the text of a task can be rebuilt from a
#: stored belief snapshot's ``problem_state.task_payload`` with IDENTICAL
#: output — that is how ``record`` recovers the text of an execution that
#: never went through ``execute``. ``family`` is excluded on purpose: it is a
#: grouping LABEL, not prose, and counting it would make almost every task
#: "have text" — hiding the honest "no task text" state behind a one-word
#: document that matches nothing meaningfully.
TEXT_TASK_KEYS = ("text", "description", "objective", "requirements",
                  "business_rules", "constraints", "spec")


def task_text(task: Dict[str, Any]) -> str:
    """The task's own text — the retrieval document's dominant input.

    Reading order over the textual task fields; non-string values are
    serialized compactly so a structured ``spec`` still contributes. Empty
    string when the task JSON carries no textual field at all: an empty
    retrieval document is reported as "no task text" rather than matched
    against every memory as a zero vector.
    """
    return task_text_from_payload(_task_payload(task))


def task_text_from_payload(payload: Optional[Dict[str, Any]]) -> str:
    """The text of a stored task payload (see :func:`task_text`)."""
    data = payload or {}
    parts: List[str] = []
    for key in TEXT_TASK_KEYS:
        value = data.get(key)
        if value is None:
            continue
        text = (value.strip() if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, sort_keys=True))
        if text:
            parts.append(text)
    return "\n".join(parts)


def task_text_digest(task: Dict[str, Any]) -> str:
    """Version key of a task text: the digest of the WHOLE task payload.

    Reuses :func:`_stable_digest` — the same convention
    ``BeliefSnapshot.problem_state.task_digest`` already uses — so one task
    version has ONE identity across the snapshot layer, the execution fact,
    and the retrieval index. The digest moves whenever any part of the task
    changes (including annotations or the model), which is exactly what makes
    "the text this execution was produced under" checkable.
    """
    return _stable_digest(task)


@dataclass
class KnowledgeRef:
    """A value copy of one strategic entry's decision-relevant fields.

    Future prediction needs the knowledge AS IT WAS: applicability
    (predicates), the expected quality/cost with intervals, and the
    admission state — not just an id and a status token. Copied by value at
    snapshot time; later revisions of the entry never rewrite this."""

    entry_id: str
    strategy_id: str
    predicates: Dict[str, Any]
    expected_quality_hat: float
    quality_interval: Tuple[float, float]
    expected_cost_hat: Dict[str, float]
    cost_interval: Dict[str, Tuple[float, float]]
    failure_prob: float
    support_n: int
    status: str
    verification_state: str
    snapshot_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "strategy_id": self.strategy_id,
            "predicates": copy.deepcopy(self.predicates),
            "expected_quality_hat": self.expected_quality_hat,
            "quality_interval": [self.quality_interval[0],
                                 self.quality_interval[1]],
            "expected_cost_hat": dict(self.expected_cost_hat),
            "cost_interval": {d: [lo, hi] for d, (lo, hi)
                              in self.cost_interval.items()},
            "failure_prob": self.failure_prob,
            "support_n": self.support_n,
            "status": self.status,
            "verification_state": self.verification_state,
            "snapshot_at": self.snapshot_at,
        }

    @classmethod
    def from_entry(cls, entry) -> "KnowledgeRef":
        return cls(
            entry_id=entry.entry_id,
            strategy_id=entry.strategy_id,
            predicates=dict(entry.predicates),
            expected_quality_hat=entry.expected_quality_hat,
            quality_interval=(entry.quality_interval[0],
                             entry.quality_interval[1]),
            expected_cost_hat=entry.expected_cost_hat.to_dict(),
            cost_interval={d: (lo, hi) for d, (lo, hi)
                           in entry.cost_interval.items()},
            failure_prob=entry.failure_prob,
            support_n=entry.support_n,
            status=entry.status,
            verification_state=entry.verification_state,
            snapshot_at=time.time(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRef":
        interval = data.get("quality_interval") or [0.0, 1.0]
        return cls(
            entry_id=str(data["entry_id"]),
            strategy_id=str(data.get("strategy_id", "")),
            predicates=dict(data.get("predicates") or {}),
            expected_quality_hat=float(data.get("expected_quality_hat", 0.5)),
            quality_interval=(float(interval[0]), float(interval[1])),
            expected_cost_hat=dict(data.get("expected_cost_hat") or {}),
            cost_interval={d: (float(lo), float(hi)) for d, (lo, hi)
                           in (data.get("cost_interval") or {}).items()},
            failure_prob=float(data.get("failure_prob", 0.5)),
            support_n=int(data.get("support_n", 0)),
            status=str(data.get("status", "candidate")),
            verification_state=str(data.get("verification_state",
                                            "unverified")),
            snapshot_at=float(data.get("snapshot_at", time.time())),
        )


@dataclass
class BeliefSnapshot:
    """The frozen information state (see module docstring).

    ``problem_state`` deliberately carries MORE than the ProblemProfile: a
    profile is a structural summary, and two problems can share a profile
    while differing in requirements, objectives, or constraints. The task
    digest (stable hash of the task JSON), an optional task reference, the
    frozen CIR, and a model digest disambiguate them.
    """

    snapshot_id: str
    task_id: str
    episode_id: Optional[str]
    created_at: float
    #: H — harness state: value-copied knowledge refs, experience counts,
    #: catalog version, tool configuration summary.
    harness_state: Dict[str, Any] = field(default_factory=dict)
    #: P — problem state: profile + task digest + optional task_ref /
    #: cir_snapshot / model_digest.
    problem_state: Dict[str, Any] = field(default_factory=dict)
    #: X — task progress: labelled fields (model_artifact,
    #: selected_plan, current_solution, errors, verification_evidence,
    #: finished).
    task_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: B — budget state: declaration + consumption view (see budget.py).
    budget_state: Dict[str, Any] = field(default_factory=dict)
    #: Conditional capability evidence view, layered by admission state.
    coverage: Dict[str, Any] = field(default_factory=dict)
    frozen: bool = False
    #: A hypothetical rollout's successor state (M2 preview). Hypothetical
    #: snapshots are excluded from real-state queries and from episode
    #: progress inheritance — an imagined outcome never contaminates the
    #: real state chain, the budget, or knowledge views.
    hypothetical: bool = False

    @staticmethod
    def new_id() -> str:
        return f"bs_{uuid.uuid4().hex[:12]}"

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, task: Dict[str, Any], episode_id: Optional[str],
              *, harness_state: Optional[Dict[str, Any]] = None,
              problem_state: Optional[Dict[str, Any]] = None,
              task_progress: Optional[Dict[str, Any]] = None,
              budget_state: Optional[Dict[str, Any]] = None,
              coverage: Optional[Dict[str, Any]] = None,
              snapshot_id: Optional[str] = None,
              created_at: Optional[float] = None,
              hypothetical: bool = False) -> "BeliefSnapshot":
        """Assemble a snapshot from caller-supplied pieces.

        The caller (ORHarness.snapshot) fills harness/problem/budget/coverage
        from the live banks; this constructor only deep-copies them. The
        problem-side task digest is derived here so it is always consistent
        with the task payload actually being solved.

        P content: when the task carries no explicit ``task_ref``, the
        problem-relevant payload (description / objective / constraints /
        spec — everything except bulky artifacts) is preserved so a future
        model can recover the problem's semantics, not just its hash."""
        task = copy.deepcopy(task)
        problem = dict(problem_state or {})
        problem.setdefault("task_digest", _stable_digest(task))
        problem.setdefault("task_ref", task.get("task_ref"))
        if task.get("coupling") and "cir_snapshot" not in problem:
            problem["cir_snapshot"] = copy.deepcopy(task["coupling"])
        if task.get("model") is not None and "model_digest" not in problem:
            problem["model_digest"] = _stable_digest(task["model"])
        if problem.get("task_ref") is None:
            problem["task_payload"] = _task_payload(task)
        snap = cls(
            snapshot_id=snapshot_id or cls.new_id(),
            task_id=str(task.get("task_id", "")),
            episode_id=episode_id,
            created_at=created_at if created_at is not None else time.time(),
            harness_state=copy.deepcopy(harness_state or {}),
            problem_state=problem,
            task_progress=copy.deepcopy(task_progress or {}),
            budget_state=copy.deepcopy(budget_state or {}),
            coverage=copy.deepcopy(coverage or {}),
            hypothetical=hypothetical,
        )
        snap.freeze()
        return snap

    def freeze(self) -> None:
        """Freeze: deep-copy every nested structure defensively.

        Idempotent. After freezing, mutation attempts on this object's dict
        fields are not intercepted (Python dataclass), but every serialized
        boundary (to_dict/from_dict/persistence) rebuilds fresh objects, so
        persisted content cannot be changed by mutating in-memory values —
        that is the guarantee the tests assert.
        """
        self.harness_state = copy.deepcopy(self.harness_state)
        self.problem_state = copy.deepcopy(self.problem_state)
        self.task_progress = copy.deepcopy(self.task_progress)
        self.budget_state = copy.deepcopy(self.budget_state)
        self.coverage = copy.deepcopy(self.coverage)
        self.frozen = True

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "frozen": self.frozen,
            "hypothetical": self.hypothetical,
            "harness_state": copy.deepcopy(self.harness_state),
            "problem_state": copy.deepcopy(self.problem_state),
            "task_progress": copy.deepcopy(self.task_progress),
            "budget_state": copy.deepcopy(self.budget_state),
            "coverage": copy.deepcopy(self.coverage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefSnapshot":
        if not isinstance(data, dict) or not data.get("snapshot_id"):
            raise ValueError("BeliefSnapshot.snapshot_id is required")
        snap = cls(
            snapshot_id=str(data["snapshot_id"]),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            created_at=float(data.get("created_at", time.time())),
            harness_state=copy.deepcopy(dict(data.get("harness_state") or {})),
            problem_state=copy.deepcopy(dict(data.get("problem_state") or {})),
            task_progress=copy.deepcopy(dict(data.get("task_progress") or {})),
            budget_state=copy.deepcopy(dict(data.get("budget_state") or {})),
            coverage=copy.deepcopy(dict(data.get("coverage") or {})),
            frozen=bool(data.get("frozen", False)),
            hypothetical=bool(data.get("hypothetical", False)),
        )
        snap.freeze()
        return snap

    # -- labelled progress fields -------------------------------------------

    def set_progress(self, key: str, value: Any, *,
                     provenance: str = "observed",
                     epistemic: str = "fact",
                     evidence_ref: Optional[str] = None) -> None:
        """Set one labelled task-progress field (pre-freeze only)."""
        if self.frozen:
            raise RuntimeError("snapshot is frozen: task_progress is immutable")
        self.task_progress[key] = _labelled(value, provenance, epistemic,
                                            evidence_ref)


def verified_knowledge_view(profile: ProblemProfile, sbank,
                             ) -> Dict[str, List[Dict[str, Any]]]:
    """Layer ``StrategicBank.matching`` by admission state.

    ``matching()`` filters by predicates only — it does NOT filter by
    verification state (that happens in ``Selector.recall`` via
    ``is_publishable``). Any new consumer (H snapshots, coverage) must go
    through THIS view so an unverified candidate is never counted as
    verified knowledge:

    - ``verified``: verification.state == "verified" (and not stale after a
      substantive revision — see ``is_publishable``).
    - ``legacy_unknown``: no verification block at all (entries written
      before admission verification existed). Kept for historical recall
      compatibility, but never counted as verified-knowledge growth.
    - ``unverified``: unverified / insufficient_evidence / refuted
      candidates — held by the framework, not knowledge.
    """
    from or_harness.strategy.selector import is_publishable
    layers: Dict[str, List[Dict[str, Any]]] = {
        "verified": [], "legacy_unknown": [], "unverified": []}
    for entry in sbank.matching(profile, include_dormant=True):
        block = entry.verification or {}
        if not block:
            layer = "legacy_unknown"
        elif block.get("state") == "verified" and is_publishable(entry):
            layer = "verified"
        else:
            layer = "unverified"
        layers[layer].append(KnowledgeRef.from_entry(entry).to_dict())
    return layers
