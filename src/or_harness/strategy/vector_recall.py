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

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    GROUPING_FEATURES,
    UNKNOWN_BUCKET,
    ProblemProfile,
    group_key,
    task_check_state,
    unsupported_predicate_keys,
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
    # A key the framework cannot EVALUATE is never assumed satisfied. It may
    # be a semantic condition only an agent can judge, so it makes the claim
    # UNDECIDABLE (``unknown``) rather than applicable — reported with the
    # exact keys so a reader knows what the code could not check.
    semantic = unsupported_predicate_keys(predicates)
    if semantic:
        undecided.append(
            "unsupported predicate key(s) this framework cannot evaluate: "
            + ", ".join(semantic))
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


def _first_error_line(error: str) -> str:
    """The first non-blank line of a traceback — enough to name the failure."""
    for line in str(error or "").splitlines():
        line = line.strip()
        if line and not line.lower().startswith("traceback"):
            return line
    return ""


def method_summary(record) -> Dict[str, Any]:
    """The reusable METHOD content of one fact, for a recall hit.

    A hit that reports only an id, a score and an outcome is unusable: the
    question a new task asks of past evidence is HOW the work was done. The
    method the record already carries is echoed here — the PLAN the candidate
    proposed (``planned``) and the method the run reports as actually
    performed (``actual``) — never a paraphrase and never the plan promoted
    to a performed method. ``basis`` states which side grounds the summary:

    * ``performed`` — a ``method_actual`` receipt exists: the strongest basis;
    * ``planned_only`` — only the intended plan is on record: real, but a
      plan is intent, not an observation;
    * ``none`` — the record reports no method at all (UNKNOWN, never filled).

    ``steps`` are returned whole (they are already bounded by what the agent
    wrote); a reader that needs the raw record uses ``inspect_hint``.
    """
    planned = record.method_planned
    actual = record.method_actual

    def _side(method: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(method, dict):
            return None
        name = str(method.get("name") or "").strip()
        steps = [str(s) for s in (method.get("steps") or []) if str(s).strip()]
        if not name and not steps:
            return None
        out: Dict[str, Any] = {"name": name, "steps": steps}
        for key in ("why", "fallback", "source"):
            if method.get(key):
                out[key] = method[key]
        return out

    p_side = _side(planned)
    a_side = _side(actual)
    if a_side is not None:
        basis = "performed"
    elif p_side is not None:
        basis = "planned_only"
    else:
        basis = "none"
    return {"planned": p_side, "actual": a_side, "basis": basis}


def _failure_summary(record) -> List[Dict[str, Any]]:
    """One line per failure: its class and the first traceback line.

    The full traceback stays on the record; a hit carries just enough to
    tell a real failure from a clean run. An empty list means no failure was
    recorded — it does NOT mean the answer was correct (that is
    ``task_check``).
    """
    out: List[Dict[str, Any]] = []
    for failure in (record.failures or []):
        data = failure.to_dict()
        out.append({
            "error_class": data.get("error_class"),
            "error": _first_error_line(str(data.get("error") or "")),
            "recovery_action": data.get("recovery_action"),
        })
    return out


def _suggested_candidates(hits: Sequence[Dict[str, Any]],
                          structural_empty: bool) -> List[Dict[str, Any]]:
    """Candidate methods named by the EXECUTION EVIDENCE of this recall.

    When the structural (strategic) channel holds nothing for this cell, the
    text channel's execution hits are the only concrete "what was tried on a
    comparable problem" material. Rather than leaving the agent to re-invent
    a method, the methods actually recorded on those hits are surfaced as
    SUGGESTIONS — each stamped with the evidence it came from and how strong
    that evidence is (``basis``), and explicitly NOT ranked. This is a
    presentation of past facts, never a menu the framework vouches for and
    never a substitute for predicting, executing and checking the choice.

    Only distinct method names are suggested, in the recall's own similarity
    order, and a hit whose record reports no method contributes nothing
    (absence stays absent). ``structural_empty`` is accepted for symmetry and
    reporting; suggestions are always derived from the hits themselves so a
    caller can see WHERE each came from.
    """
    suggestions: List[Dict[str, Any]] = []
    seen: set = set()
    for hit in hits:
        summary = hit.get("method") or {}
        if summary.get("basis") == "none":
            continue
        side = summary.get("actual") or summary.get("planned") or {}
        name = str(side.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        suggestions.append({
            "method_name": name,
            "steps": list(side.get("steps") or []),
            "from_execution_id": hit.get("execution_id"),
            "from_task_id": hit.get("task_id"),
            "strategy_id": hit.get("strategy_id"),
            "similarity": hit.get("similarity"),
            "basis": summary.get("basis"),
            "task_check": hit.get("task_check"),
            "structural_match": hit.get("structural_match"),
            "note": ("a method actually recorded on a past execution; "
                     "presented for consideration, not ranked or vouched for "
                     "— predict, execute and check it like any candidate"),
        })
    return suggestions


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
        # HOW the work was done — the half that made a hit reusable. Echoed
        # from the record, never rewritten; `basis` says how strong it is
        # (performed > planned_only > none). See `method_summary`.
        "method": method_summary(record),
        # The answer's variable values, when the run reported them, so a
        # reader can see WHAT was produced — not to be copied blindly, and
        # absent when the script did not report them (never fabricated).
        "solution_variables": (copy.deepcopy(
            record.execution_features.get("solution_variables"))
            if isinstance(record.execution_features.get("solution_variables"),
                          dict) else None),
        # One line per failure (class + first traceback line + any recorded
        # recovery). Empty is "no failure recorded", not "the answer was
        # right" — that is `task_check`.
        "failure_summary": _failure_summary(record),
        "method_basis": method_summary(record)["basis"],
        # The full record (CIR snapshot, complete tracebacks, artifacts) is
        # NOT inlined — this names the call that fetches it on demand.
        "inspect_hint": (f"orx inspect --bank experience --id "
                         f"{record.execution_id}"),
        # Citations of past executions this attempt was adapted from (the
        # agent's own `--adapted-from`), when recorded. A citation is a record
        # that the case was READ, never that reuse succeeded.
        "reuse_trace": (copy.deepcopy(
            record.execution_features.get("reuse_trace"))
            if isinstance(record.execution_features.get("reuse_trace"), dict)
            else None),
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
    # When the text channel finds NO published knowledge, the execution hits
    # are the only concrete "what was tried on a comparable problem" material
    # — so name their methods as suggestions. With strategic knowledge
    # present, the entry path already carries the reusable claim and the
    # evidence stays a discovery signal; suggestions are therefore scoped to
    # the knowledge-empty case, never a second menu bolted on top.
    if not result["strategic_knowledge"]:
        result["suggested_candidates"] = _suggested_candidates(
            result["execution_evidence"], structural_empty=True)
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