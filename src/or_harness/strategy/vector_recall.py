"""Vector recall: embedding-based DISCOVERY of relevant memories.

The design split this module implements (and must never blur):

- Discovery (this module): text similarity decides WHICH memories are
  surfaced. Cross-cell recall is the entire point — a task whose text is
  nearly identical to a past one but whose structural coupling lands in a
  different bucket must still SEE that past execution. Nothing here filters
  by ``group_key``.
- Reuse (the existing layers): ``group_key`` / ``profile_matches`` /
  ``evidence_predicates`` decide whether a surfaced memory may be APPLIED.
  That judgment is ANNOTATED here (``structural_match``, ``reusable``) and
  never auto-enforced into the statistics: a cross-cell hit never joins the
  target's evidence set. ``for_profile`` / ``evidence_predicates`` / the
  score formula are untouched, so a cross-bucket recall can never move a
  single number in the aggregation.

Two independent annotations, deliberately not derived from one boolean:

- ``same_cell`` / ``different_cell`` (execution path) compares the recall's
  ``group_key`` with the record's own ``group_key``.
- ``applies`` / ``conflicts`` / ``unknown`` (entry path) evaluates the
  entry's predicate set against the profile, distinguishing "the claim does
  not cover this structure" from "the information needed to decide is
  missing". Missing information is never treated as applicability.

Staleness: an index item stores only ``{id, doc_digest, vector}``. Every hit
re-reads the CURRENT record by id, so validity (status, quality, verification
state, retirement) is always current, and the re-derived document digest is
compared with the indexed one — a changed document makes the stored vector
describe something that no longer exists, and the item is reported stale
rather than presented as a current similarity.

Read-only: recall embeds the QUERY in memory only. It writes no text, no
index, and no record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from or_harness.core.schema import (
    GROUPING_FEATURES,
    UNKNOWN_BUCKET,
    ProblemProfile,
    group_key,
    task_check_state,
)
from or_harness.strategy.embedding_index import (
    LAYER_EXECUTION,
    LAYER_STRATEGIC,
    LAYERS,
    cosine,
    describe_backend,
    document_digest,
    document_entry,
    document_execution,
)
from or_harness.strategy.selector import is_publishable

#: Characters of the task text echoed back with an execution hit. The point
#: of returning the excerpt at all: a surfaced memory without its text is a
#: number without a subject.
EXCERPT_CHARS = 200


class VectorRecallUnavailable(Exception):
    """The vector channel cannot run — the caller degrades to profile recall.

    One exception type for every honest refusal (no backend configured, no
    task text, missing/incompatible index, backend call failure). The message
    is quoted verbatim into ``degraded.reason`` so the harness sees WHY the
    text channel was skipped instead of silently getting profile-only results.
    """


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """First ``limit`` characters of a text, whitespace-collapsed."""
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def classify_applicability(profile: Optional[ProblemProfile],
                           predicates: Dict[str, Any]
                           ) -> Tuple[str, bool, Optional[str]]:
    """``(structural_match, reusable, reason)`` for one entry's predicates.

    - ``applies``: every predicate is satisfied — the claim covers this
      structure and may be reused.
    - ``conflicts``: at least one predicate is contradicted BY A KNOWN value
      (family mismatch, value outside the interval, or a measured value
      against an unknown-bucket claim). Not reusable, with the exact
      conflict stated.
    - ``unknown``: nothing is contradicted, but a predicate needs a value the
      profile does not supply — undecidable. NOT reusable ("we could not
      check" is not "it applies"), with the missing dimension named.

    When every predicate is decidable, ``applies`` coincides exactly with
    :func:`or_harness.core.schema.profile_matches` — this function refines
    that boolean by separating contradiction from missing information.
    """
    if profile is None:
        return "unknown", False, "no profile available to evaluate predicates"
    if not predicates:
        return "applies", True, None
    family_pred = predicates.get("family")
    if family_pred is not None and profile.family != family_pred:
        return ("conflicts", False,
                f"family mismatch: claim covers family {family_pred}, task is "
                f"family {profile.family}")
    undecided: List[str] = []
    for dim in GROUPING_FEATURES:
        if dim not in predicates:
            continue
        pred = predicates[dim]
        value = getattr(profile, dim)
        if isinstance(pred, str):
            # A claim that covers only the unknown bucket says nothing about
            # a measured task.
            if value is not None:
                return ("conflicts", False,
                        f"{dim}={value} is measured but the claim only covers "
                        f"the '{UNKNOWN_BUCKET}' cell")
            continue
        try:
            lo, hi = (float(x) for x in pred)
        except (TypeError, ValueError):
            continue
        if value is None:
            undecided.append(f"{dim} (claim covers [{lo:.2f}, {hi:.2f}], task "
                             f"value unmeasured)")
            continue
        if not (lo <= float(value) <= hi):
            return ("conflicts", False,
                    f"{dim}={value} outside the claim's [{lo:.2f}, {hi:.2f}]")
    if undecided:
        return ("unknown", False,
                "cannot decide applicability: " + "; ".join(undecided))
    return "applies", True, None


def _execution_entry(harness, item: Dict[str, Any], score: float,
                     task_profile: Optional[ProblemProfile],
                     query_cell: Optional[str]) -> Optional[Dict[str, Any]]:
    """One execution hit, re-read from the CURRENT record (or None to skip)."""
    record = harness.bank.get(str(item.get("id")))
    if record is None:
        return None  # the fact is gone; a vector without a record is nothing
    stored_text = (harness.store.get_task_text(record.task_id,
                                               record.task_text_digest)
                   if record.task_text_digest else None)
    if stored_text is None:
        return None  # the text this vector describes is no longer available
    if document_digest(document_execution(record, stored_text)) != \
            item.get("doc_digest"):
        # The document changed under the index: the stored vector describes a
        # text that no longer exists. Excluded from hits (reported as stale).
        return None
    record_cell = group_key(record.profile_snapshot)
    if query_cell is None:
        structural = "unknown"
    else:
        structural = ("same_cell" if record_cell == query_cell
                      else "different_cell")
    return {
        "execution_id": record.execution_id,
        "task_id": record.task_id,
        "strategy_id": record.strategy_id,
        "similarity": round(score, 6),
        "task_text_excerpt": excerpt(stored_text),
        "task_text_digest": record.task_text_digest,
        "observed_quality": dict(record.quality),
        "observed_cost": record.cost.to_dict(),
        "cost_measured": (sorted(record.cost.measured)
                          if record.cost.measured is not None else None),
        "failures": len(record.failures),
        "status": record.quality.get("status"),
        "measurement_scope": record.measurement_scope,
        # The TASK-level verdict travels with the hit. A hit whose answer was
        # confirmed not to satisfy the task is still worth reading (its cost
        # and failure are real) but its `observed_quality` must not be read
        # as a quality the strategy achieved. `None` = never checked, which
        # is NOT a pass.
        "task_check": task_check_state(record),
        "task_check_limitations": (
            ["the answer was confirmed NOT to satisfy the task: the observed "
             "quality is a solver-side figure, not a task result"]
            if task_check_state(record) == "failed"
            else (["the answer's validity is UNKNOWN: no task-level check has "
                   "been run"] if task_check_state(record) is None else [])),
        "profile_cell": record_cell,
        "structural_match": structural,
    }


def _knowledge_entry(harness, item: Dict[str, Any], score: float,
                     task_profile: Optional[ProblemProfile],
                     include_unverified: bool) -> Optional[Dict[str, Any]]:
    """One strategic-knowledge hit, re-read from the CURRENT entry."""
    entry = harness.sbank.get(str(item.get("id")))
    if entry is None:
        return None  # retired: the entry left the hot store entirely
    if entry.status == "dormant":
        return None  # dormant entries are not consulted
    if not include_unverified and not is_publishable(entry):
        return None  # a candidate is not knowledge yet
    claim_text = document_entry(entry)
    if document_digest(claim_text) != item.get("doc_digest"):
        return None  # the claim text changed; the vector is stale
    structural, reusable, reason = classify_applicability(task_profile,
                                                          entry.predicates)
    out = {
        "entry_id": entry.entry_id,
        "strategy_id": entry.strategy_id,
        "similarity": round(score, 6),
        "claim_text": claim_text,
        "expected_quality_hat": entry.expected_quality_hat,
        "expected_cost_hat": entry.expected_cost_hat.to_dict(),
        "cost_measured": (sorted(entry.expected_cost_hat.measured)
                          if entry.expected_cost_hat.measured is not None
                          else None),
        "failure_prob": entry.failure_prob,
        "status": entry.status,
        "verification_state": entry.verification_state,
        "support_n": entry.support_n,
        "applicability": list(entry.applicability),
        "risk_conditions": list(entry.risk_conditions),
        "predicates": entry.predicates,
        "structural_match": structural,
        "reusable": reusable,
    }
    if reason:
        out["reason"] = reason
    return out


def _scan(index, layer: str, query_vector: List[float]) -> List[Tuple[Any, float]]:
    """Cosine scores of every item, best first (ties broken by id)."""
    scored = [(item, cosine(query_vector, item.get("vector") or []))
              for item in index.items(layer)]
    scored.sort(key=lambda pair: (-pair[1], str(pair[0].get("id") or "")))
    return scored


def recall_vectors(harness, task_text: str, *, top_k: int = 5,
                   include_unverified: bool = False,
                   task_profile: Optional[ProblemProfile] = None
                   ) -> Dict[str, Any]:
    """Embedding-first discovery over both memory layers.

    Raises :class:`VectorRecallUnavailable` (never a partial result) when the
    channel cannot run at all; the caller turns that into ``degraded``.
    Eligibility (publishability, lifecycle, staleness) is applied BEFORE the
    ``top_k`` cut, so an ineligible item never occupies a slot that a usable
    memory could have taken.
    """
    backend = getattr(harness, "embedding_index", None)
    if backend is None:
        raise VectorRecallUnavailable("no embedding backend configured")
    if not (task_text or "").strip():
        raise VectorRecallUnavailable("no task text in task JSON")

    statuses = {layer: backend.status(layer) for layer in LAYERS}
    usable = {layer: s for layer, s in statuses.items() if s["usable"]}
    if not usable:
        # Report the FIRST reason in a stable order (execution layer first):
        # a missing index and a model change are the two realistic causes.
        reason = statuses[LAYER_EXECUTION]["reason"] or "index unusable"
        if not statuses[LAYER_EXECUTION]["exists"] and \
                not statuses[LAYER_STRATEGIC]["exists"]:
            reason = ("index missing; run `orx rebuild-index`")
        raise VectorRecallUnavailable(reason)

    try:
        query_vector = backend.backend.embed_query(task_text)
    except Exception as exc:  # noqa: BLE001 - any backend failure degrades
        raise VectorRecallUnavailable(
            f"embedding backend error: {type(exc).__name__}: {exc}") from exc

    query_cell = group_key(task_profile) if task_profile is not None else None

    result: Dict[str, Any] = {
        "backend": describe_backend(backend.backend),
        "execution_evidence": [],
        "strategic_knowledge": [],
    }
    stale: Dict[str, int] = {LAYER_EXECUTION: 0, LAYER_STRATEGIC: 0}

    if LAYER_EXECUTION in usable:
        for item, score in _scan(backend, LAYER_EXECUTION, query_vector):
            if len(result["execution_evidence"]) >= top_k:
                break
            row = _execution_entry(harness, item, score, task_profile,
                                   query_cell)
            if row is None:
                stale[LAYER_EXECUTION] += 1
                continue
            result["execution_evidence"].append(row)

    if LAYER_STRATEGIC in usable:
        for item, score in _scan(backend, LAYER_STRATEGIC, query_vector):
            if len(result["strategic_knowledge"]) >= top_k:
                break
            row = _knowledge_entry(harness, item, score, task_profile,
                                   include_unverified)
            if row is None:
                stale[LAYER_STRATEGIC] += 1
                continue
            result["strategic_knowledge"].append(row)

    result["stale_indexed"] = stale
    result["stale_note"] = (
        "stale counts index items skipped while filling top_k — items below "
        "the cut are not examined, so a zero here means 'none encountered', "
        "not 'none exist'")
    result["unindexed"] = _unindexed_summary(harness, backend)
    degraded_layers = {layer: {"reason": statuses[layer]["reason"]}
                       for layer in LAYERS if not statuses[layer]["usable"]}
    if degraded_layers:
        result["degraded_layers"] = degraded_layers
    return result


def _indexed_ids(backend, layer: str) -> set:
    return {str(item.get("id")) for item in backend.items(layer)}


def _unindexed_summary(harness, backend) -> Dict[str, Any]:
    """How many memories the text channel cannot see, and where to find them.

    An unindexed memory must never simply disappear: it is a legacy record
    (text never captured) or a write whose index sync was deferred. Both
    reasons are counted here and the query entry point is stated, so the
    harness can reach them through the ordinary profile/inspect paths.
    """
    synchronizer = getattr(harness, "index_sync", None)
    if synchronizer is not None:
        counts = synchronizer.unindexed_counts()
    else:  # pragma: no cover - defensive, ORHarness always wires one
        exec_indexed = _indexed_ids(backend, LAYER_EXECUTION)
        knowledge_indexed = _indexed_ids(backend, LAYER_STRATEGIC)
        executions = [r for r in harness.bank.all() if r.source == "executed"]
        entries = harness.sbank.list(include_dormant=True)
        counts = {
            LAYER_EXECUTION: sum(1 for r in executions
                                 if r.execution_id not in exec_indexed),
            LAYER_STRATEGIC: sum(1 for e in entries
                                 if e.entry_id not in knowledge_indexed),
        }
    return {
        "execution_evidence": counts[LAYER_EXECUTION],
        "strategic_knowledge": counts[LAYER_STRATEGIC],
        "note": ("unindexed memories have no vector (legacy records whose "
                 "text was never captured, or a deferred index sync); they "
                 "remain available through `orx inspect --bank "
                 "experience|strategic`, `orx inspect --bank texts`, and the "
                 "profile-based retrieval path"),
    }