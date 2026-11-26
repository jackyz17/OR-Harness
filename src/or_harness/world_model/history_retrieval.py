"""Unified historical-execution retrieval for the world model (stage E/1).

The G/C/R prediction signals used to select their evidence from three
DIFFERENT condition-grouping systems: G scanned paired feedback for keyword
overlap, C grouped cost observations by phase/family/solver/structure, R
grouped technical attempts by solver interface + structure. Those systems
all suffered from the same failure: a sparse history fragments into many
singleton groups, so a decision with real comparable executions in memory
receives no numerical reference at all.

This module replaces the discovery/selection half of that design with ONE
retrieval base:

1. build candidate-aware QUERY VIEWS (the problem text, each candidate's
   declared method, the structural/solving conditions);
2. run the EXISTING embedding index (:mod:`or_harness.strategy
   .vector_recall`) once per view and merge the hits into one pool
   (union, max similarity, deterministic tie-break);
3. apply a per-backend cosine threshold and a task-diversification rule so
   the pool is not filled by near-identical attempts at one task;
4. attach the facts the signals need — the recorded execution (status,
   method, failures, R technical-event labels) and, when the execution is
   part of a CLOSED episode, the paired evaluation (G label, remaining-task
   cost values, R event labels, recorded attribution);
5. G/C/R each perform a LIGHT rerank over that shared pool with their own
   relevance rule, capped at 10 cases.

Alongside the local cases this module derives GLOBAL history from the same
pre-decision material: G prediction/observation error, C four-dimensional
distributions and error direction, R event base rates and Brier score. The
global block is BACKGROUND: it never becomes a candidate-specific prior and
it is never a correction multiplier.

Design boundaries (the reason this is a separate module):

- No second vector store and no re-embedding of history: the embedding
  index built by the existing sync path is the only discovery channel.
- No new persistence: the block is frozen INTO the prediction context by
  :func:`build_history_retrieval`; nothing is written by a read.
- Prediction-time discipline: pairs come from the published closed-episode
  window and executions after the frozen moment (when a cutoff is given) are
  excluded. A degraded embedding backend is REPORTED, never silently
  replaced by the old keyword/condition matching.
- Thresholds are configuration, and the ACTUAL value, embedding backend and
  score definition travel with the block so a reader can audit why a case
  was or was not selected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from or_harness.core.schema import PREDICTED_COST_DIMENSIONS
from or_harness.world_model.contracts import (
    FAITHFUL_G_METRIC,
    FAITHFUL_G_OBSERVATION_VERSION,
    RETRIES_FUTURE_VERSION,
)

#: Version of the frozen retrieval block.
HISTORY_RETRIEVAL_VERSION = "wm-history/1"

#: Each signal keeps at most this many local cases. It is a CEILING, never a
#: target: an unrelated execution is dropped instead of padding the list.
DEFAULT_MAX_LOCAL_CASES = 10
#: The embedding candidate pool the signals rerank over. The recall is
#: deliberately wider than the final 10 so the per-signal rerank has room.
DEFAULT_POOL_SIZE = 30
#: How many attempts of the SAME task may enter the pool. A failure/repair
#: chain is real evidence; a task repeated ten times is one task, not ten.
DEFAULT_MAX_TASK_ATTEMPTS = 2

#: Backend-specific cosine floors — PER QUERY VIEW. These are NOT
#: interchangeable: different embedding backends, and different query views
#: inside one backend, have different similarity scales. A task-text query is
#: only a subset of an execution document, so its score is systematically
#: lower than a document-to-document comparison; a method query is lower
#: still. Thresholding every view at one number would therefore either miss
#: same-task executions or admit unrelated ones through the view with the
#: loosest scale.
#:
#: ``problem`` / ``candidate_method`` / ``conditions`` are the per-view
#: DISCOVERY floors. ``method_relevance`` is the separate floor a case must
#: reach on the CANDIDATE-METHOD view to be labelled ``similar_method``
#: (G/R) instead of merely task-similar.
#:
#: The local-hashing values were measured on the review-bank history
#: available in this repository (12 executions / 8 tasks):
#:   problem view (task-text query vs execution doc):
#:     same task 0.843-0.944, different family 0.403-0.804
#:     -> 0.82 keeps every same-task execution and rejects the
#:        different-family maximum;
#:   candidate-method view (method query vs execution doc):
#:     self 0.580-0.706, same-task other 0.647-0.680, other task
#:     0.240-0.704 -> NO clean separation; 0.62 is a weak-relevance floor
#:     and a `similar_method` label stays low-confidence;
#:   conditions view (structure/solver query vs execution doc):
#:     same-task 0.151-0.334, different family 0.062-0.224 -> 0.30 keeps
#:     the strongest structural hits only.
#: The sample is small: these are cautious starting values for a LEXICAL
#: backend, not universal constants. Any other backend must re-derive them
#: (docs/world_model/history_threshold_calibration.md).
_BACKEND_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "local-hashing-embedding-v1": {
        "problem": 0.62, "candidate_method": 0.60, "conditions": 0.30,
        "method_relevance": 0.62,
        "basis": (
            "measured on the repository review-bank history with the "
            "deterministic local-hashing backend (12 executions / 8 tasks). "
            "The problem view compares TASK TEXT with the item's task-text "
            "vector: same-task pairs score 1.000, same-family 0.507-1.000, "
            "different-family 0.401-0.890, so 0.62 keeps the relevant range. "
            "A LEGACY index without task-text vectors falls back to the full "
            "document, whose score is diluted by the method/outcome suffix "
            "(same task 0.843-0.944; a short task text can fall to ~0.5-0.7), "
            "so a rebuild is recommended. A text-overlap admission is NOT a "
            "method match: the per-signal relevance stays `task_similar` "
            "unless the candidate-method view reaches its own floor (0.62, "
            "weak: the lexical backend has no clean separation). Small "
            "sample: cautious starting values, re-derive per backend."),
        "verified": True,
    },
    "openai-compatible": {
        "problem": 0.45, "candidate_method": 0.45, "conditions": 0.40,
        "method_relevance": 0.45,
        "basis": (
            "NO verified calibration exists in this repository for a remote "
            "embedding model: these are PROVISIONAL working floors, not a "
            "measured value, and the block reports threshold_verified=false. "
            "Scores from different backends are not comparable; set "
            "OR_HARNESS_HISTORY_THRESHOLD / _METHOD_THRESHOLD / "
            "_CONDITIONS_THRESHOLD from the same labeled-pair experiment "
            "before relying on this path."),
        "verified": False,
    },
}

#: Event names R scores. Kept here so the retrieval never invents a category;
#: the authority remains :mod:`or_harness.world_model.contracts`.
_R_EVENTS = ("implementation_failure", "timeout", "environment_failure")

#: Minimum paired samples before a global statistic is labelled measured.
_GLOBAL_MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _backend_thresholds(backend: Any) -> Dict[str, Any]:
    """The per-view cosine floors used for THIS backend, with provenance."""
    from or_harness.strategy.embedding_index import describe_backend
    identity = describe_backend(backend)
    model_id = identity.get("model_id")
    source = identity.get("source") or "unknown"
    diagnostics: List[str] = []
    local_key = "local-hashing-embedding-v1"
    if source == "local-hashing" and model_id == local_key:
        key = local_key
        threshold_source = "profile:" + key
    elif source == "remote":
        key = (model_id if model_id in _BACKEND_THRESHOLDS
               and model_id != local_key else "openai-compatible")
        threshold_source = "profile:" + key
    else:
        # Keep the existing generic provisional numbers, but do NOT identify
        # an unknown backend as hashing (or pretend it has been calibrated).
        key = "openai-compatible"
        threshold_source = "fallback:" + key
        diagnostics.append(
            "backend identity has no calibrated threshold profile; using "
            "the unverified generic provisional fallback, not local-hashing "
            "calibration. No backend identity is inferred from vector width.")
    chosen = copy.deepcopy(_BACKEND_THRESHOLDS[key])
    chosen["basis"] = (f"Backend model_id={model_id!r}, source={source}. "
                       + chosen["basis"])
    if diagnostics:
        chosen["basis"] += " " + " ".join(diagnostics)
    overrides = {
        "problem": "OR_HARNESS_HISTORY_THRESHOLD",
        "candidate_method": "OR_HARNESS_HISTORY_METHOD_THRESHOLD",
        "conditions": "OR_HARNESS_HISTORY_CONDITIONS_THRESHOLD",
        "method_relevance": "OR_HARNESS_HISTORY_METHOD_RELEVANCE_THRESHOLD",
    }
    sources = {name: threshold_source for name in overrides}
    override_requested = False
    for name, env_name in overrides.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        override_requested = True
        value = _env_float(env_name)
        if value is None:
            diagnostics.append(
                f"invalid threshold override {env_name} for {name} ignored: "
                "expected a finite number; the recorded profile value remains in use")
            continue
        chosen[name] = value
        sources[name] = f"env:{env_name}"
    if override_requested:
        chosen["verified"] = False
        chosen["basis"] += (
            " Manual threshold override requests are not independent "
            "calibration evidence; the effective configuration is unverified.")
    chosen["threshold_sources"] = sources
    chosen["model_id"] = model_id
    chosen["source"] = source
    chosen["threshold_profile"] = key
    chosen["threshold_diagnostics"] = diagnostics
    chosen["score_definition"] = (
        "cosine similarity between ONE query view's embedding and the "
        "existing execution-index channel: problem uses the task-text vector "
        "when available, otherwise the legacy full document; method and "
        "conditions use the full document (task text + family/outcome + "
        "planned/performed method). Each view has its own discovery floor; a row is "
        "admitted when ANY view reaches its floor, and `similarity.overall` "
        "is the largest score/floor ratio (a relative relevance for "
        "ranking, not a cross-view comparable cosine)")
    return chosen


def history_config(backend: Any = None) -> Dict[str, Any]:
    """The ACTUAL retrieval configuration recorded on every block."""
    thresholds = _backend_thresholds(backend)
    return {
        "version": HISTORY_RETRIEVAL_VERSION,
        "max_local_cases": _env_int("OR_HARNESS_HISTORY_MAX_CASES",
                                    DEFAULT_MAX_LOCAL_CASES),
        "pool_size": _env_int("OR_HARNESS_HISTORY_POOL",
                              DEFAULT_POOL_SIZE),
        "max_task_attempts": _env_int("OR_HARNESS_HISTORY_TASK_ATTEMPTS",
                                      DEFAULT_MAX_TASK_ATTEMPTS),
        "similarity_thresholds": {
            name: thresholds[name]
            for name in ("problem", "candidate_method", "conditions")},
        "method_similarity_threshold": thresholds["method_relevance"],
        "admission_rule": ("a recalled execution enters the pool when at "
                           "least one query view reaches ITS floor"),
        "threshold_sources": thresholds.get("threshold_sources") or {},
        "threshold_verified": thresholds.get("verified"),
        "threshold_basis": thresholds.get("basis")
        or "configured; no local calibration available",
        "score_definition": thresholds["score_definition"],
        "backend_model_id": thresholds.get("model_id"),
        "backend_source": thresholds["source"],
        "threshold_profile": thresholds["threshold_profile"],
        "threshold_diagnostics": list(thresholds["threshold_diagnostics"]),
    }


# ---------------------------------------------------------------------------
# query views
# ---------------------------------------------------------------------------

def _method_view(method: Any) -> Dict[str, Any]:
    method = method if isinstance(method, dict) else {}
    name = str(method.get("name") or "").strip()
    steps = [str(step).strip() for step in (method.get("steps") or [])
             if str(step).strip()]
    return {"name": name, "steps": steps}


def _method_query_text(method: Any) -> str:
    view = _method_view(method)
    parts = [view["name"]] if view["name"] else []
    parts.extend(view["steps"])
    return " method ".join(parts).strip()


def _conditions_query_text(profile: Dict[str, Any],
                           candidate: Any) -> str:
    """Structural/solving facts only — never the problem's prose.

    Keeping the prose out is what stops a generic shared word from becoming
    a "similar conditions" claim; the conditions view is about the shape of
    the problem and the declared solving setup.
    """
    profile = profile or {}
    parts: List[str] = []
    family = profile.get("family")
    if family:
        parts.append(f"family {family}")
    problem_class = profile.get("problem_class")
    if problem_class:
        parts.append(f"problem_class {problem_class}")
    for key in ("resource_coupling", "temporal_coupling", "route_complexity"):
        value = profile.get(key)
        if value not in (None, "", "unknown"):
            parts.append(f"{key} {value}")
    solver = getattr(candidate, "solver", None)
    if solver:
        parts.append(f"solver {solver}")
    config = getattr(candidate, "config", None) or {}
    for key in sorted(config):
        value = config[key]
        if isinstance(value, (str, int, float, bool)) and value not in (None, ""):
            parts.append(f"{key} {value}")
    return " ".join(parts).strip()


def _method_digest(method: Any) -> str:
    payload = json.dumps(_method_view(method), sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_query_views(task: Dict[str, Any], profile: Any,
                      candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    """The multi-perspective retrieval requests for ONE decision.

    Returns one problem view, one view per candidate method and one
    conditions view. A view whose text is empty is omitted (an empty query
    would match nothing honestly, but it would still cost an embedding
    call).
    """
    from or_harness.world_model.state import task_text
    profile_dict = (profile.to_dict() if hasattr(profile, "to_dict")
                    else dict(profile or {}))
    views: List[Dict[str, Any]] = []
    problem = " ".join(task_text(task).split())
    if problem:
        views.append({"name": "problem", "kind": "problem",
                      "text": problem,
                      "family": profile_dict.get("family")})
    for index, candidate in enumerate(candidates or []):
        method_text = _method_query_text(getattr(candidate, "method", None))
        if not method_text:
            continue
        views.append({
            "name": f"candidate:{index}", "kind": "candidate_method",
            "text": method_text,
            "strategy_id": getattr(candidate, "strategy_id", None),
            "method_digest": _method_digest(getattr(candidate, "method", None)),
        })
    conditions = _conditions_query_text(profile_dict,
                                        candidates[0] if candidates else None)
    if conditions:
        views.append({"name": "conditions", "kind": "conditions",
                      "text": conditions})
    return views


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------

def _number(value: Any, *, allow_zero: bool = True) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if value < 0 or (value == 0 and not allow_zero):
        return None
    return value


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (index - low), 6)


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation; None when it is not defined."""
    if len(left) != len(right) or len(left) < 3:
        return None
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    n = len(left_ranks)
    mean_left = sum(left_ranks) / n
    mean_right = sum(right_ranks) / n
    numerator = sum((left_ranks[i] - mean_left) * (right_ranks[i] - mean_right)
                    for i in range(n))
    denom_left = math.sqrt(sum((value - mean_left) ** 2
                               for value in left_ranks))
    denom_right = math.sqrt(sum((value - mean_right) ** 2
                                for value in right_ranks))
    if denom_left == 0 or denom_right == 0:
        return None
    return round(numerator / (denom_left * denom_right), 6)


def _auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Rank-based AUC; None unless BOTH classes are present."""
    pairs = [(label, score) for label, score in zip(labels, scores)
             if label in (0, 1) and score is not None]
    positives = [score for label, score in pairs if label == 1]
    negatives = [score for label, score in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 6)


def _brier(labels: Sequence[int], probabilities: Sequence[float]
           ) -> Optional[float]:
    pairs = [(label, probability) for label, probability in
             zip(labels, probabilities) if label in (0, 1)
             and probability is not None]
    if not pairs:
        return None
    return round(sum((probability - label) ** 2
                     for label, probability in pairs) / len(pairs), 6)


# ---------------------------------------------------------------------------
# embedding retrieval
# ---------------------------------------------------------------------------

def _retrieve_view(harness, text: str, *, top_k: int,
                   task_profile: Any) -> Tuple[List[Dict[str, Any]],
                                               Optional[str]]:
    from or_harness.strategy.vector_recall import (
        VectorRecallUnavailable, recall_vectors,
    )
    try:
        result = recall_vectors(harness, text, top_k=top_k,
                                task_profile=task_profile)
    except VectorRecallUnavailable as exc:
        return [], str(exc)
    except Exception as exc:  # noqa: BLE001 - any backend fault degrades
        return [], f"embedding backend error: {type(exc).__name__}: {exc}"
    return list(result.get("execution_evidence") or []), None


def _task_attempt_priority(row: Dict[str, Any], first_status: Any) -> int:
    """Prefer a recorded failure or a method change for a second attempt."""
    status = str(row.get("status") or "")
    failed = status in ("error", "timeout", "infeasible", "unbounded")
    changed = (row.get("method") or {}).get("basis") != first_status
    return (2 if failed else 0) + (1 if changed else 0)


def _diversify(rows: Sequence[Dict[str, Any]], *,
               pool_size: int, max_task_attempts: int
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One row per task first, then valuable same-task repairs."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for row in rows:
        key = str(row.get("task_id") or "unknown")
        if key not in by_task:
            by_task[key] = []
            order.append(key)
        by_task[key].append(row)
    selected: List[Dict[str, Any]] = []
    for key in order:
        if len(selected) >= pool_size:
            break
        selected.append(by_task[key][0])
    extra = 0
    if len(selected) < pool_size and max_task_attempts > 1:
        candidates: List[Tuple[int, float, str, Dict[str, Any]]] = []
        for key in order:
            first = by_task[key][0]
            first_basis = (first.get("method") or {}).get("basis")
            for row in by_task[key][1:]:
                candidates.append((_task_attempt_priority(row, first_basis),
                                   float((row.get("similarity") or {}).get(
                                       "overall") or 0.0), key, row))
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        used = {key: 1 for key in order}
        for _priority, _score, key, row in candidates:
            if len(selected) >= pool_size:
                break
            if used.get(key, 0) >= max_task_attempts:
                continue
            selected.append(row)
            used[key] = used.get(key, 0) + 1
            extra += 1
    report = {
        "pool_size": len(selected),
        "n_distinct_tasks": len({str(row.get("task_id") or "unknown")
                                 for row in selected}),
        "same_task_extra_attempts": extra,
        "max_task_attempts": max_task_attempts,
        "note": ("one execution per task first, then a valuable same-task "
                 "failure/repair attempt while slots remain: repeated runs "
                 "of one task never masquerade as independent tasks"),
    }
    return selected, report


def _pair_bound_execution_id(pair: Dict[str, Any]) -> Optional[str]:
    """The ONE execution a pair's G/R observation is attributed to.

    A remaining-task evaluation can cover several executions (a window), but
    the G label and the R labels belong to the BOUND attempt — the execution
    the prediction was bound to. The close-out records it under
    ``execution_conditions.execution_id`` (and in the implementation
    observation). Guessing from the window's first id would attribute one
    method's outcome to a different method.
    """
    conditions = pair.get("execution_conditions") or {}
    bound = conditions.get("execution_id") if isinstance(conditions, dict) \
        else None
    if not bound:
        observation = pair.get("implementation_observation") or {}
        if isinstance(observation, dict):
            bound = observation.get("execution_id")
    return str(bound) if bound else None


def _pair_cost_carrier_id(pair: Dict[str, Any]) -> Optional[str]:
    """The ONE execution that carries a remaining-task COST reference.

    The cost window observation covers the whole remaining task, so it is a
    single reference, not one per execution in the window. It is attached to
    the bound execution when that execution is part of the window, else to
    the window's first recorded execution (order is the close-out's window
    order, never re-sorted, so it is deterministic).
    """
    execution_ids = [str(execution_id)
                     for execution_id in (pair.get("execution_ids") or [])]
    bound = _pair_bound_execution_id(pair)
    if bound and bound in execution_ids:
        return bound
    return execution_ids[0] if execution_ids else None


def _attach_evaluation_facts(pool: List[Dict[str, Any]],
                             pairs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Join recalled executions to their CLOSED-evaluation pair facts.

    The G/R observation is attached ONLY to the bound execution. The C
    window reference is attached to ONE canonical carrier execution, so a
    multi-execution window is never counted as several cost samples. A row
    that is not the bound execution may still carry the cost fact (its
    ``evaluation`` block names the bound execution), and the selectors use
    that explicit attribution rather than array position.
    """
    by_execution: Dict[str, Dict[str, Any]] = {}

    def _offer(execution_id: Optional[str], pair: Dict[str, Any]) -> None:
        if not execution_id:
            return
        current = by_execution.get(str(execution_id))
        if current is None or _pair_rank(pair) > _pair_rank(current):
            by_execution[str(execution_id)] = pair

    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        _offer(_pair_bound_execution_id(pair), pair)
        carrier = _pair_cost_carrier_id(pair)
        if carrier and carrier != _pair_bound_execution_id(pair):
            _offer(carrier, pair)
    attached = 0
    bound_rows = 0
    cost_rows = 0
    for row in pool:
        pair = by_execution.get(str(row.get("execution_id")))
        if pair is None:
            continue
        fact = _evaluation_fact(pair)
        row["evaluation"] = fact
        attached += 1
        if fact.get("bound_execution_id") == str(row.get("execution_id")):
            bound_rows += 1
        if fact.get("cost_carrier_execution_id") == str(row.get("execution_id")):
            cost_rows += 1
    return {"attached_rows": attached, "bound_rows": bound_rows,
            "cost_rows": cost_rows}


def _pair_rank(pair: Dict[str, Any]) -> Tuple[int, int, int, str]:
    from or_harness.world_model.g_signal import _pair_g_observation
    label, _ = _pair_g_observation(pair)
    observed = 1 if label is not None else 0
    cost_values = (pair.get("cost_observed_fact") or {}).get("values") or {}
    risk = len(pair.get("risk_actual") or [])
    return (observed, len(cost_values), risk,
            str(pair.get("evaluation_id") or ""))


def _evaluation_fact(pair: Dict[str, Any]) -> Dict[str, Any]:
    """The compact, provider-safe facts of one closed evaluation pair."""
    fact: Dict[str, Any] = {
        "evaluation_id": pair.get("evaluation_id"),
        # Explicit attribution: the G/R label belongs to the bound attempt;
        # the cost window covers `covers_execution_ids` and is carried by
        # `cost_carrier_execution_id` exactly once.
        "bound_execution_id": _pair_bound_execution_id(pair),
        "cost_carrier_execution_id": _pair_cost_carrier_id(pair),
        "covers_execution_ids": [str(execution_id) for execution_id
                                 in (pair.get("execution_ids") or [])],
    }
    from or_harness.world_model.g_signal import pair_g_fact, _pair_g_observation
    benefit = pair_g_fact(pair)
    g_block: Dict[str, Any] = {}
    for key in ("metric", "observation_rule_version", "predicted",
                "signed_error", "label_eligibility", "reason", "attribution_ref"):
        if key in benefit:
            g_block[key] = copy.deepcopy(benefit[key])
    label, _ = _pair_g_observation(pair)
    if label is not None:
        g_block['observed'] = label
    if g_block:
        fact["g"] = g_block
    attribution = benefit.get("attribution")
    if isinstance(attribution, dict) and attribution:
        fact["attribution"] = {
            key: copy.deepcopy(attribution[key]) for key in (
                "method_fidelity", "failure_source", "evidence_status", "acceptance_scope", "source", "assessment")
            if attribution.get(key) not in (None, "", [], {})}
    if pair.get("task_check") is not None:
        fact["task_check"] = copy.deepcopy(pair["task_check"])
    cost = pair.get("cost_observed_fact") or {}
    if cost:
        conditions = cost.get("conditions") or {}
        actuals = copy.deepcopy(
            conditions.get("actual_solving_conditions") or [])
        if not actuals:
            # The provider projection keeps the pair's compact execution
            # conditions; they describe the SAME attempt's solver/config.
            comparison = (pair.get("execution_conditions") or {}).get(
                "comparison_key")
            if isinstance(comparison, dict):
                actuals = [{"solver": comparison.get("solver"),
                            "config": comparison.get("config") or {}}]
        fact["cost"] = {
            "values": _cost_values_from_pair(pair),
            "retry_observation_rule_version": conditions.get("retry_observation_rule_version") or
                (cost.get("dimension_observation_rule_versions") or {}).get("retries"),
            "scope": conditions.get("scope"),
            "window_coverage": conditions.get("window_coverage"),
            "family": conditions.get("family"),
            "structure": copy.deepcopy(conditions.get("structure") or {}),
            "solver_conditions": copy.deepcopy(
                conditions.get("solver_conditions") or {}),
            "actual_solving_conditions": actuals,
            "reusable_artifacts": bool(conditions.get("reusable_artifacts")),
            "lower_bounds": copy.deepcopy(cost.get("lower_bounds") or {}),
            "source_refs": list(cost.get("source_refs") or [])[:4],
        }
    measured = pair.get("cost_measured") or {}
    if measured:
        fact["cost_predicted_actual"] = {
            dim: {"predicted": _number(entry.get("predicted")),
                  "actual": _number(entry.get("actual"))}
            for dim, entry in measured.items()
            if dim in PREDICTED_COST_DIMENSIONS}
    risk_actual = pair.get("risk_actual")
    if risk_actual:
        predicted = {str(entry.get("event")): entry.get("probability")
                     for entry in (pair.get("risk_predicted") or [])
                     if isinstance(entry, dict)}
        fact["risk_actual"] = [
            {"event": entry.get("event"),
             "label": entry.get("label"),
             "predicted": predicted.get(str(entry.get("event"))),
             "brier": entry.get("brier"),
             "label_basis": entry.get("label_basis"),
             "observation_rule_version": entry.get("observation_rule_version") or pair.get("risk_observation_rule_version") or
                ((pair.get("implementation_observation") or {}).get("observation_rule_version")
                 if entry.get("event") == "implementation_failure" else None)}
            for entry in risk_actual if isinstance(entry, dict)][:6]
    if pair.get("failure_classes"):
        fact["failure_classes"] = list(pair["failure_classes"])[:4]
    return fact


# ---------------------------------------------------------------------------
# global (pre-decision) statistics
# ---------------------------------------------------------------------------

def _pairs_of(paired_feedback: Optional[Dict[str, Any]]
              ) -> List[Dict[str, Any]]:
    return [pair for pair in ((paired_feedback or {}).get("pairs") or [])
            if isinstance(pair, dict)]


def _compact_global(value: Any, *, keep_empty: bool = False) -> Any:
    """Drop null/empty diagnostics from a GLOBAL scalar block.

    Global statistics are BACKGROUND: a null field (no AUC with one class,
    no quantile with no samples) carries no information the counts do not
    already give. The counts and the real distributions are never dropped.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            compact = _compact_global(item, keep_empty=keep_empty)
            if compact not in (None, "", [], {}) or keep_empty:
                out[key] = compact
        return out
    if isinstance(value, list):
        return [_compact_global(item) for item in value]
    return value


def global_g_statistics(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How this model's G predictions turned out over the closed window.

    ONE label convention (faithful-execution acceptance) and ONE aligned
    sample: every prediction-calibration number (means, MAE, signed bias,
    Brier, AUC, reliability bins) is computed from the SAME
    ``(prediction, observation)`` pairs. An observation with no prediction is
    counted separately and NEVER paired by array index with another task's
    probability. Diagnostics only: never a candidate prior or correction.
    """
    labels: List[int] = []
    scored: List[Tuple[float, int]] = []
    observed_without_prediction: List[int] = []
    tasks, episodes = set(), set()
    unknown = 0
    unknown_reasons: Dict[str, int] = {}
    n_predicted = 0
    n_current = 0
    from or_harness.world_model.g_signal import pair_g_fact, _pair_g_observation
    for pair in pairs:
        benefit = pair_g_fact(pair)
        metric = benefit.get("metric")
        rule = benefit.get("observation_rule_version")
        if metric != FAITHFUL_G_METRIC or rule != FAITHFUL_G_OBSERVATION_VERSION:
            continue
        n_current += 1
        observed, _ = _pair_g_observation(pair)
        predicted_raw = benefit.get("predicted")
        predicted = (float(predicted_raw)
                     if isinstance(predicted_raw, (int, float))
                     and not isinstance(predicted_raw, bool) else None)
        if predicted is not None:
            n_predicted += 1
        if observed in (0, 1):
            labels.append(int(observed))
            tasks.add(str(pair.get("task_id") or "unknown"))
            episodes.add(str(pair.get("episode_id") or "unknown"))
            if predicted is not None:
                scored.append((predicted, int(observed)))
            else:
                observed_without_prediction.append(int(observed))
            continue
        unknown += 1
        reason = str(benefit.get("reason")
                     or benefit.get("label_eligibility")
                     or "no attributable label in the closed pair")
        unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
    n_scored = len(scored)
    scored_labels = [label for _probability, label in scored]
    probabilities = [probability for probability, _label in scored]
    errors = [abs(probability - label) for probability, label in scored]
    signed = [probability - label for probability, label in scored]
    observed_mean = _mean([float(label) for label in scored_labels])
    predicted_mean = _mean(probabilities)
    bins: List[Dict[str, Any]] = []
    if n_scored >= 8:
        for index in range(5):
            bucket = [(probability, label) for probability, label in scored
                      if min(4, int(probability * 5)) == index]
            if bucket:
                bins.append({
                    "lo": index / 5, "hi": (index + 1) / 5,
                    "n": len(bucket),
                    "mean_prediction": round(
                        sum(probability for probability, _label in bucket)
                        / len(bucket), 6),
                    "observed_rate": round(
                        sum(label for _probability, label in bucket)
                        / len(bucket), 6)})
    return _compact_global({
        "version": "wm-g-global/1",
        "metric": FAITHFUL_G_METRIC,
        "observation_rule_version": FAITHFUL_G_OBSERVATION_VERSION,
        "scope": ("closed calibration window, decisions made before this "
                  "prediction; same model identity only"),
        "sample_rule": ("all prediction-calibration metrics use the SAME "
                        "scored (prediction, observation) pairs; an observed "
                        "label without a prediction is counted separately and "
                        "never paired by array position"),
        "n_pairs": n_current,
        "n_other_conventions": len(pairs) - n_current,
        "n_observed": len(labels),
        "n_predicted": n_predicted,
        "n_scored": n_scored,
        "n_observed_without_prediction": len(observed_without_prediction),
        "observed_without_prediction_mean": _mean(
            [float(label) for label in observed_without_prediction]),
        "n_distinct_tasks": len(tasks),
        "n_distinct_episodes": len(episodes),
        "observed_mean": observed_mean,
        "predicted_mean": predicted_mean,
        "mae": _mean(errors),
        "signed_error_mean": _mean(signed),
        # signed = predicted - observed: a POSITIVE mean means the model
        # systematically predicted HIGHER than the realized outcome.
        "bias_direction": (None if not signed else
                           "over_predicted" if sum(signed) > 0 else
                           "under_predicted" if sum(signed) < 0 else "none"),
        "brier": (round(sum((probability - label) ** 2
                            for probability, label in scored)
                        / n_scored, 6) if n_scored else None),
        "auc": _auc(scored_labels, probabilities),
        "reliability_bins": bins,
        "unknown": {"n": unknown, "reasons": unknown_reasons},
        "support": ("measured" if len(tasks) >= _GLOBAL_MIN_SAMPLES
                    else "insufficient_independent_tasks"),
        "note": ("BACKGROUND diagnostic of this model's past G error; it is "
                 "not a candidate-specific success probability and it does "
                 "not license a G correction"),
    })


def _cost_values_from_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    """The four active ACTUAL cost values of one pair, missing dims absent.

    A published pair deduplicates the measured values from up to three
    places (``cost_observed_fact.values`` for unpredicted dimensions,
    ``values_not_scored`` for the rest, and ``cost_measured[dim].actual``
    when that dimension WAS predicted). They describe the SAME observed
    window, so they are merged by dimension — never summed, never
    zero-filled. ``retries`` is kept only under the current observation
    rule; a legacy retry count is not re-read as the new target.
    """
    fact = pair.get("cost_observed_fact") or {}
    conditions = fact.get("conditions") or {}
    merged: Dict[str, Optional[float]] = {}
    for source in (fact.get("values") or {},
                   fact.get("values_not_scored") or {}):
        for dim, value in source.items():
            if dim in PREDICTED_COST_DIMENSIONS:
                merged[dim] = _number(value)
    for dim, entry in (pair.get("cost_measured") or {}).items():
        if dim in PREDICTED_COST_DIMENSIONS and isinstance(entry, dict):
            actual = _number(entry.get("actual"))
            if actual is not None:
                merged[dim] = actual
    rule = (conditions.get("retry_observation_rule_version")
            or ((fact.get("dimension_observation_rule_versions") or {})
                .get("retries")))
    if rule != RETRIES_FUTURE_VERSION:
        merged.pop("retries", None)
    return {dim: value for dim, value in merged.items() if value is not None}


def global_c_statistics(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The four-dimensional remaining-task cost distribution and error.

    Both layers come from the SAME valid pairs: actual values from the
    completed observation window, predicted-vs-actual only where BOTH sides
    exist. A dimension with no complete observation stays absent; it is
    never zero-filled.
    """
    dimension_values: Dict[str, List[float]] = {
        dim: [] for dim in PREDICTED_COST_DIMENSIONS}
    dimension_refs: Dict[str, set] = {
        dim: set() for dim in PREDICTED_COST_DIMENSIONS}
    predicted_pairs: Dict[str, List[Tuple[float, float]]] = {
        dim: [] for dim in PREDICTED_COST_DIMENSIONS}
    incomplete: Dict[str, int] = {dim: 0 for dim in PREDICTED_COST_DIMENSIONS}
    incomplete_reasons: Dict[str, Dict[str, int]] = {
        dim: {} for dim in PREDICTED_COST_DIMENSIONS}
    tasks, episodes = set(), set()
    n_pairs = 0
    for pair in pairs:
        values = _cost_values_from_pair(pair)
        if not values and not pair.get("cost_measured"):
            continue
        n_pairs += 1
        tasks.add(str(pair.get("task_id") or "unknown"))
        episodes.add(str(pair.get("episode_id") or "unknown"))
        for dim in PREDICTED_COST_DIMENSIONS:
            if dim in values:
                dimension_values[dim].append(values[dim])
                fact_refs = ((pair.get("cost_observed_fact") or {})
                             .get("source_refs") or [])
                dimension_refs[dim].update(str(ref) for ref in fact_refs)
                if pair.get("evaluation_id"):
                    dimension_refs[dim].add(str(pair["evaluation_id"]))
            else:
                fact = pair.get("cost_observed_fact") or {}
                lower = ((fact.get("lower_bounds") or {}).get(dim) or {})
                if lower:
                    reason = "incomplete_or_excluded_observation"
                elif (fact.get("conditions") or {}).get(
                        "window_coverage") == "partial":
                    reason = "partial_window_coverage"
                elif dim == "retries":
                    reason = "legacy_retry_observation_rule"
                else:
                    reason = "dimension_not_completely_observed"
                incomplete[dim] += 1
                incomplete_reasons[dim][reason] = \
                    incomplete_reasons[dim].get(reason, 0) + 1
            measured = (pair.get("cost_measured") or {}).get(dim) or {}
            predicted = _number(measured.get("predicted"))
            actual = _number(measured.get("actual"))
            if dim == "retries" and dim not in values:
                continue
            if predicted is not None and actual is not None:
                predicted_pairs[dim].append((predicted, actual))
    dimensions: Dict[str, Any] = {}
    for dim in PREDICTED_COST_DIMENSIONS:
        actuals = dimension_values[dim]
        errors = [abs(predicted - actual)
                  for predicted, actual in predicted_pairs[dim]]
        ratios = [math.log(predicted / actual)
                  for predicted, actual in predicted_pairs[dim]
                  if actual > 0 and predicted > 0]
        relative = [abs(predicted - actual) / actual
                    for predicted, actual in predicted_pairs[dim]
                    if actual > 0]
        signed_ratio = _mean(ratios)
        entry = {
            "n": len(actuals),
            "p50": _quantile(actuals, 0.5),
            "p90": _quantile(actuals, 0.9),
            "n_predicted_actual": len(predicted_pairs[dim]),
            "median_abs_error": _quantile(errors, 0.5),
            "median_abs_relative_error": _quantile(relative, 0.5),
            "mean_log_ratio": (round(signed_ratio, 6)
                               if signed_ratio is not None else None),
            "error_direction": (None if signed_ratio is None else
                                "actual_higher" if signed_ratio < 0 else
                                "predicted_higher" if signed_ratio > 0
                                else "none"),
            "spearman_predicted_actual": _correlation(
                [predicted for predicted, _actual in predicted_pairs[dim]],
                [actual for _predicted, actual in predicted_pairs[dim]]),
            "n_incomplete": incomplete[dim],
            "incomplete_reasons": dict(incomplete_reasons[dim]),
            "source_refs": sorted(dimension_refs[dim])[:6],
        }
        compact = _compact_global(entry)
        if dim == "retries":
            from or_harness.world_model.cost_signal import retry_error_statistics
            compact["observation_rule_version"] = RETRIES_FUTURE_VERSION
            compact["corrective_round_error"] = retry_error_statistics(predicted_pairs[dim])
        if compact.get("n") or compact.get("n_incomplete"):
            dimensions[dim] = compact
    return _compact_global({
        "version": "wm-c-global/1",
        "scope": ("remaining_to_task_end observations from the closed "
                  "calibration window; pre-decision only"),
        "n_pairs_with_cost_fact": n_pairs,
        "n_distinct_tasks": len(tasks),
        "n_distinct_episodes": len(episodes),
        "dimensions": dimensions,
        "observation_basis": (
            "actual = the complete observed remaining-to-task-end window; a "
            "dimension without a complete ledger observation is counted as "
            "incomplete and stays missing (never zero). Legacy retry-rule "
            "observations are not counted as retries."),
        "support": ("measured" if len(tasks) >= _GLOBAL_MIN_SAMPLES
                    else "insufficient_independent_tasks"),
        "note": ("BACKGROUND distribution, never a multiplier. Compare a "
                 "candidate's own plan with the p50/p90 of comparable work; "
                 "do not raise every cost by a global error ratio."),
    })


def global_r_statistics(strategy_calibration: Optional[Dict[str, Any]]
                        ) -> Dict[str, Any]:
    """Event base rates and R prediction performance over the closed window.

    Reuses the published ``technical_r`` statistics: no bank rescan, no new
    label rule. Each event is its own random variable; unknown labels are
    excluded from the denominator, never read as non-occurrence.
    """
    from or_harness.world_model.r_signal import global_event_base_rates
    technical = (strategy_calibration or {}).get("technical_r") or {}
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION
    if technical.get("observation_rule_version") != R_OBSERVATION_VERSION:
        return {"observation_rule_version": R_OBSERVATION_VERSION,
                "support": "insufficient_evidence", "events": {},
                "note": "historical formal R statistics are not current implementation-window calibration; unavailable labels are not zero risk"}
    base = global_event_base_rates(technical)
    source_events = technical.get("events") or {}
    events: Dict[str, Any] = {}
    for name in _R_EVENTS:
        entry = (base.get("events") or {}).get(name)
        published = source_events.get(name) or {}
        if not entry and not published:
            continue
        item = dict(entry or {})
        for key in ("n_asked", "n_reported_probabilities", "n_scorable",
                    "mean_brier", "n_scored_distinct_episodes",
                    "brier_support", "n_positive_scored",
                    "n_negative_scored", "mean_probability_observed",
                    "mean_probability_not_observed", "discrimination"):
            if key in published:
                item[key] = copy.deepcopy(published[key])
        events[name] = _compact_global(item)
    n_attempts = int((base.get("n_attempts")
                      or (technical.get("coverage") or {}).get(
                          "n_actual_attempts") or 0))
    return _compact_global({
        "version": "wm-r-global/1",
        "observation_rule_version": base.get("observation_rule_version"),
        "scope": base.get("scope") or (
            "all executed attempts in the closed calibration window; broad "
            "background, not a candidate-specific condition"),
        "n_attempts": n_attempts,
        "events": events,
        "rate_basis": base.get("rate_basis"),
        "support": (base.get("support")
                    or ("measured_global_base_rate" if events
                        else "insufficient_evidence")),
        "note": ("The three events are independent; the maximum is a risk "
                 "score, not a joint probability. A base rate is background "
                 "and never replaces the candidate-specific technical "
                 "evidence. No observed failure is not zero risk."),
    })


# ---------------------------------------------------------------------------
# block assembly
# ---------------------------------------------------------------------------

def _candidate_views(candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates or []):
        text = _method_query_text(getattr(candidate, "method", None))
        if not text:
            continue
        out.append({
            "view": f"candidate:{index}",
            "strategy_id": getattr(candidate, "strategy_id", None),
            "method_digest": _method_digest(getattr(candidate, "method", None)),
        })
    return out


def _view_floor(view_name: str, floors: Dict[str, Any]) -> Optional[float]:
    """The discovery floor belonging to one query view."""
    if view_name == "problem":
        return floors.get("problem")
    if view_name == "conditions":
        return floors.get("conditions")
    if view_name.startswith("candidate:"):
        return floors.get("candidate_method")
    return None


def _recalled_from_index(harness, execution_id: str) -> Dict[str, Any]:
    """Minimal recall-shaped row for a task-vector-only admission.

    The reference store is never re-read: the task-vector channel names an
    execution id and the pool builder reads the CURRENT record (and drops a
    vanished or stale one). The task excerpt is filled from the record's
    stored text when available.
    """
    record = harness.bank.get(execution_id)
    text = None
    if record is not None and record.task_text_digest:
        text = harness.store.get_task_text(record.task_id,
                                           record.task_text_digest)
    return {"execution_id": execution_id,
            "task_id": getattr(record, "task_id", None),
            "strategy_id": getattr(record, "strategy_id", None),
            "task_text_excerpt": (
                " ".join((text or "").split())[:200] if text else None),
            "profile_cell": None,
            "method": {}, "status": None, "feasibility": None}


def _pool_row(harness, recalled: Dict[str, Any], *,
              view_scores: Dict[str, float], floors: Dict[str, Any],
              cutoff: Optional[float], query_cell: Optional[str],
              expected_doc_digest: Optional[str] = None
              ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve one recall hit into the shared pool row, or a DROP reason.

    The task-vector channel can name an execution directly from the index
    without the full-document recall path, so this function re-applies the
    SAME validity constraints as :mod:`or_harness.strategy.vector_recall`:
    the record must still exist, its source must still be ``executed``, it
    must not lie after the frozen moment, and its CURRENT retrieval document
    must match the indexed document digest (a changed record makes the stored
    vector describe something that no longer exists). A dropped row is
    counted by reason — never silently included.
    """
    execution_id = str(recalled.get("execution_id") or "")
    record = harness.bank.get(execution_id) if execution_id else None
    if record is None:
        return None, "missing_record"
    if str(getattr(record, "source", "executed")) != "executed":
        return None, "invalid_source"
    if cutoff is not None and float(record.created_at or 0.0) > cutoff:
        return None, "after_cutoff"
    if expected_doc_digest is not None:
        from or_harness.strategy.embedding_index import (
            document_digest, document_execution,
        )
        text = (harness.store.get_task_text(record.task_id,
                                            record.task_text_digest)
                if record.task_text_digest else None)
        if text is None:
            return None, "unreadable_task_text"
        current = document_digest(document_execution(record, text))
        if current != expected_doc_digest:
            return None, "stale_index"
    from or_harness.world_model.r_signal import technical_attempt_fact
    technical = technical_attempt_fact(record)
    from or_harness.world_model.implementation_facts import execution_implementation_evidence, compact_evidence
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION
    implementation = execution_implementation_evidence(harness, record, cutoff=cutoff)
    technical["r_events"]["implementation_failure"] = {
        "label": implementation.get("label"),
        "basis": implementation.get("reason") or "historical formal receipt only; current implementation coverage unknown",
        "observation_rule_version": implementation.get("observation_rule_version")}
    record_cell = recalled.get("profile_cell")
    structural = ("same_cell" if query_cell and record_cell == query_cell
                  else "different_cell" if record_cell else "unknown")
    method = recalled.get("method") or {}
    relative = {name: (round(value / floor, 6) if floor and floor > 0
                       else None)
                for name, value in view_scores.items()
                for floor in [_view_floor(name, floors)]}
    row = {
        "execution_id": execution_id,
        "task_id": recalled.get("task_id"),
        "implementation_evidence": compact_evidence(implementation),
        "formal_technical_outcome": {"label": technical.get("label"),
            "observation_rule_version": technical.get("observation_rule_version")},
        "strategy_id": recalled.get("strategy_id"),
        "task_excerpt": recalled.get("task_text_excerpt"),
        "similarity": {
            # RELATIVE relevance: the largest raw score divided by that
            # view's own discovery floor. It is used for ordering inside the
            # pool; it is NOT a cross-view comparable cosine.
            "overall": (round(max(value for value in relative.values()
                                  if value is not None), 6)
                        if any(value is not None
                               for value in relative.values()) else None),
            "by_view": {key: round(value, 6)
                        for key, value in sorted(view_scores.items())},
            "relative_by_view": {key: value
                                 for key, value in sorted(relative.items())},
        },
        "structural_match": structural,
        "profile_cell": record_cell,
        "method": {
            "basis": method.get("basis"),
            "planned": _method_view(record.method_planned),
            "actual": _method_view(record.method_actual),
        },
        "status": (record.quality or {}).get("status"),
        "feasible": (record.quality or {}).get("feasible"),
        "measurement_scope": record.measurement_scope,
        "failure_classes": [getattr(failure, "error_class", None)
                            for failure in (record.failures or [])][:4],
        "r_events": {name: copy.deepcopy(
            ((technical.get("r_events") or {}).get(name) or {}))
            for name in _R_EVENTS},
        "solver_interface": (record.solver or {}).get("name"),
        "has_reusable_script": bool(technical.get("reusable_script")),
        "created_at": float(record.created_at or 0.0),
    }
    inspected = recalled.get("inspect_hint")
    if inspected:
        row["inspect_hint"] = inspected
    return row, None


def _passes_admission(view_scores: Dict[str, float],
                      floors: Dict[str, Any]) -> bool:
    """A row is admitted when ANY query view reaches ITS OWN floor."""
    for name, value in view_scores.items():
        floor = _view_floor(name, floors)
        if floor is not None and value >= floor:
            return True
    return False


def build_history_retrieval(
        harness, *,
        task: Dict[str, Any], profile: Any,
        candidates: Sequence[Any] = (),
        episode_id: Optional[str] = None,
        cutoff: Optional[float] = None,
        paired_feedback: Optional[Dict[str, Any]] = None,
        strategy_calibration: Optional[Dict[str, Any]] = None,
        pool_size: Optional[int] = None) -> Dict[str, Any]:
    """Gather ONE frozen local+global history block for this decision.

    Read-only: embeds the views, reads the existing index/records/pairs and
    returns a JSON-safe block. It writes no index and no record.
    """
    from or_harness.strategy.embedding_index import (
        LAYER_EXECUTION, describe_backend,
    )
    backend = getattr(harness, "embedding_index", None)
    config = history_config(backend)
    if pool_size is not None:
        config["pool_size"] = int(pool_size)
    pairs = _pairs_of(paired_feedback)
    block: Dict[str, Any] = {
        "version": HISTORY_RETRIEVAL_VERSION,
        "generated_at": time.time(),
        "cutoff": cutoff,
        "episode_id": episode_id,
        "status": "ok",
        "config": config,
        "queries": [],
        "pool": [],
        "global": {
            "g": global_g_statistics(pairs),
            "c": global_c_statistics(pairs),
            "r": global_r_statistics(strategy_calibration),
        },
        "accounting": {},
        "degraded": [],
        "missing": [],
        "notes": [
            "local cases = embedding-recalled executions bounded by the "
            "similarity floor; they are DISCOVERY material, not a quality or "
            "risk estimate",
            "global statistics use only closed episodes that existed before "
            "this prediction; they are background and never a correction "
            "multiplier",
        ],
    }
    if backend is None:
        block["status"] = "degraded"
        block["degraded"].append({
            "part": "local_history",
            "reason": ("no embedding backend configured; local similar-case "
                       "retrieval did NOT run. Global statistics are still "
                       "provided. No keyword or condition-group fallback was "
                       "substituted.")})
        block["accounting"] = {"index_items": 0, "scanned": 0,
                               "above_threshold": 0, "pool_size": 0,
                               "distinct_tasks": 0}
        return block
    status = backend.status(LAYER_EXECUTION)
    if not status.get("usable"):
        block["status"] = "degraded"
        block["degraded"].append({
            "part": "local_history",
            "reason": (f"execution index unusable: {status.get('reason')}; "
                       "local similar-case retrieval did NOT run and was not "
                       "replaced by keyword matching")})
        block["accounting"] = {"index_items": 0, "scanned": 0,
                               "above_threshold": 0, "pool_size": 0,
                               "distinct_tasks": 0}
        return block
    block["backend"] = describe_backend(backend)
    views = build_query_views(task, profile, candidates)
    block["queries"] = [{"name": view["name"], "kind": view["kind"],
                         "chars": len(view["text"]),
                         "strategy_id": view.get("strategy_id"),
                         "method_digest": view.get("method_digest")}
                        for view in views]
    block["candidate_views"] = _candidate_views(candidates)
    merged: Dict[str, Dict[str, Any]] = {}
    degraded_reason: Optional[str] = None
    # The PROBLEM view is compared against the execution item's TASK-TEXT
    # vector when the index carries one: comparing a task-text query with the
    # whole document would let the method's length move the same-task score.
    # A legacy index without auxiliary vectors falls back to the full-document
    # channel, and that fallback is RECORDED.
    task_scores: Dict[str, float] = {}
    problem_channel = "legacy_full_document"
    from or_harness.strategy.embedding_index import (
        LAYER_EXECUTION as _EXEC_LAYER, cosine as _cosine,
    )
    problem_view = next((view for view in views
                         if view.get("kind") == "problem"), None)
    indexed_items = backend.items(_EXEC_LAYER)
    index_digests = {str(item.get("id")): item.get("doc_digest")
                     for item in indexed_items if item.get("id")}
    task_vector_items = [item for item in indexed_items
                         if item.get("task_vector")]
    if problem_view is not None and task_vector_items:
        try:
            query_vector = backend.backend.embed_query(problem_view["text"])
        except Exception as exc:  # noqa: BLE001 - backend fault degrades
            degraded_reason = (f"embedding backend error: "
                               f"{type(exc).__name__}: {exc}")
        else:
            problem_channel = "task_text_vector"
            for item in task_vector_items:
                task_scores[str(item.get("id"))] = _cosine(
                    query_vector, item["task_vector"])
    for view in views:
        if view.get("kind") == "problem" and problem_channel == \
                "task_text_vector":
            continue
        rows, reason = _retrieve_view(harness, view["text"],
                                      top_k=config["pool_size"],
                                      task_profile=profile)
        if reason is not None:
            degraded_reason = reason
            break
        for row in rows:
            execution_id = str(row.get("execution_id") or "")
            if not execution_id:
                continue
            entry = merged.setdefault(execution_id, {"recalled": row,
                                                     "scores": {}})
            entry["scores"][view["name"]] = float(
                row.get("similarity") or 0.0)
    if degraded_reason is None:
        for execution_id, score in task_scores.items():
            entry = merged.setdefault(execution_id, {"recalled": {},
                                                     "scores": {}})
            entry["scores"]["problem"] = float(score)
    if degraded_reason is not None:
        block["status"] = "degraded"
        block["degraded"].append({
            "part": "local_history",
            "reason": (f"embedding retrieval failed: {degraded_reason}; "
                       "local similar-case retrieval is UNAVAILABLE (not "
                       "'nothing similar exists') and no keyword fallback "
                       "was substituted")})
        merged = {}
    query_cell = None
    try:
        from or_harness.core.schema import ProblemProfile, group_key
        if getattr(profile, "problem_id", None):
            query_cell = group_key(ProblemProfile.from_dict(
                profile.to_dict() if hasattr(profile, "to_dict") else profile))
    except Exception:  # noqa: BLE001 - structural label is optional
        query_cell = None
    floors = config["similarity_thresholds"]
    rows: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}

    def _drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for execution_id, entry in merged.items():
        view_scores = entry["scores"]
        if not _passes_admission(view_scores, floors):
            continue
        recalled = entry["recalled"] or _recalled_from_index(
            harness, execution_id)
        row, reason = _pool_row(
            harness, recalled, view_scores=view_scores, floors=floors,
            cutoff=cutoff, query_cell=query_cell,
            expected_doc_digest=index_digests.get(execution_id))
        if row is None:
            _drop(reason or "invalid")
            continue
        rows.append(row)
    rows.sort(key=lambda row: (
        -float((row.get("similarity") or {}).get("overall") or 0.0),
        str(row.get("execution_id") or "")))
    above_threshold = len(rows)
    pool, diversify = _diversify(
        rows, pool_size=config["pool_size"],
        max_task_attempts=config["max_task_attempts"])
    attached = _attach_evaluation_facts(pool, pairs)
    # Similarity is candidate-relative: the same recalled execution may be a
    # method view match for one candidate and only a problem-text match for
    # another. The per-view scores are frozen in the row so a per-candidate
    # selector can rerank WITHOUT running the embedding backend again.
    block["pool"] = pool
    block["accounting"] = {
        "index_items": int(status.get("count") or 0),
        "scanned": int(status.get("count") or 0),
        "above_threshold": above_threshold,
        "below_threshold": len(merged) - above_threshold,
        "pool_size": len(pool),
        "distinct_tasks": diversify["n_distinct_tasks"],
        "same_task_extra_attempts": diversify["same_task_extra_attempts"],
        "n_with_closed_evaluation": attached["attached_rows"],
        "n_bound_evaluation_rows": attached["bound_rows"],
        "n_cost_reference_rows": attached["cost_rows"],
        "excluded_after_cutoff": dropped.get("after_cutoff", 0),
        "excluded_invalid_source": dropped.get("invalid_source", 0),
        "excluded_stale_index": dropped.get("stale_index", 0),
        "excluded_missing_record": dropped.get("missing_record", 0),
        "excluded_unreadable_task_text": dropped.get(
            "unreadable_task_text", 0),
        "views_run": len(views),
        "problem_view_channel": problem_channel,
        "task_vector_items": len(task_vector_items),
        "note": diversify["note"],
    }
    if not pool and block["status"] == "ok":
        block["missing"].append(
            "local_history: no recalled execution reached the similarity "
            "floor; the signals receive the global statistics only (this is "
            "not 'nothing exists' — executions may exist below the floor)")
    return block


# ---------------------------------------------------------------------------
# G / C / R local case selection (candidate-aware rerank)
# ---------------------------------------------------------------------------

def _case_steps(method: Any, *, limit: int = 3,
                chars: int = 160) -> List[str]:
    steps = _method_view(method)["steps"]
    out = []
    for step in steps[:limit]:
        text = " ".join(step.split())
        out.append(text[:chars])
    return out


def _method_scores(row: Dict[str, Any], history: Dict[str, Any],
                   candidate: Any) -> Dict[str, Optional[float]]:
    """The precomputed similarities relevant to ONE candidate.

    Candidate-method views are matched by the candidate METHOD digest, so a
    shared view is never credited to a different method. The digest is the
    identity: two candidates with the same words but a different method
    structure do not share a method-similarity score.
    """
    by_view = (row.get("similarity") or {}).get("by_view") or {}
    digest = _method_digest(getattr(candidate, "method", None))
    method_scores: List[float] = []
    matched: List[str] = []
    for entry in history.get("candidate_views") or []:
        view = entry.get("view")
        if view not in by_view:
            continue
        if entry.get("method_digest") == digest:
            method_scores.append(float(by_view[view]))
            matched.append(view)
    return {
        "overall": float((row.get("similarity") or {}).get("overall") or 0.0),
        "method": max(method_scores) if method_scores else None,
        "conditions": (float(by_view["conditions"])
                       if "conditions" in by_view else None),
        "problem": (float(by_view["problem"])
                    if "problem" in by_view else None),
        "matched_views": matched,
    }


def _row_task_excerpt(row: Dict[str, Any], *, chars: int = 110) -> str:
    return " ".join(str(row.get("task_excerpt") or "").split())[:chars]


def _scope_name(context: Any) -> str:
    return "remaining_to_task_end"


def _current_cost_conditions(context: Any, candidate: Any) -> Dict[str, Any]:
    """The C conditions a candidate is compared under — NO phase.

    The optional model artifact is a normal context fact, kept as
    ``reusable_artifacts``; it is never converted into a phase label.
    """
    from or_harness.world_model.g_signal import requested_solving_conditions
    profile = getattr(getattr(context, "joint", None), "profile", None) or {}
    solving = getattr(context, "solving_context", None) or {}
    progress = solving.get("task_progress") or {}
    artifact = progress.get("model_artifact") or {}
    if artifact.get("hypothetical") or artifact.get("epistemic") == "inferred":
        artifact = {}
    value = artifact.get("value")
    available = bool(value) and not (
        isinstance(value, dict) and value.get("status") in (
            "missing", "unavailable", "failed", "not_created", "not_started"))
    runner = solving.get("execution_context") or {}
    solver = (getattr(candidate, "solver", None)
              or runner.get("solver_interface"))
    conditions = requested_solving_conditions(candidate,
                                              getattr(context, "execution_constraints", None))
    if not conditions.get("solver"):
        conditions["solver"] = solver
    return {
        "scope": _scope_name(context),
        "family": profile.get("family"),
        "structure": {key: value for key, value in profile.items()
                      if key in ("problem_class", "scale_features",
                                 "semantic_coupling", "resource_coupling",
                                 "temporal_coupling", "route_complexity")
                      and value not in (None, "", {}, "unknown")},
        "solver_conditions": conditions,
        "reusable_artifacts": ([] if not available else [{
            "value": copy.deepcopy(value),
            "evidence_ref": artifact.get("evidence_ref")}]),
    }


def _known(value: Any) -> bool:
    return value not in (None, "", "unknown", [], {})


def _cost_grade(case: Dict[str, Any], current: Dict[str, Any],
                *, strict: bool) -> Tuple[str, List[str], List[str]]:
    """``(grade, unknowns, differences)`` for one cost case — NO phase.

    The ONLY hard exclusion is the measurement window: a case whose coverage
    is incomplete or whose scope is not the SAME remaining task is
    ``unavailable`` — its number cannot be compared to a remaining-task
    forecast at all.

    Every OTHER difference is a comparability deduction, not an exclusion:
    a different solver (HiGHS vs CBC), a different solver config or a
    different known structural field still provides real modelling/token/
    tool-call work. Such cases stay as ``broader_reference`` with an explicit
    ``differences`` list and a lower ``relevance_weight``; unknown conditions
    are named as unknowns. This keeps build4's "history exists but no
    reference was offered" failure from returning.
    """
    if case.get("window_coverage") != "complete":
        return ("unavailable",
                ["historical window coverage is not complete"], [])
    if case.get("scope") != current.get("scope"):
        return "unavailable", ["measurement scope differs"], []
    unknowns: List[str] = []
    blocking_unknowns: List[str] = []
    differences: List[str] = []
    case_family, current_family = case.get("family"), current.get("family")
    if not (_known(case_family) and _known(current_family)):
        unknowns.append("family unknown")
    elif case_family != current_family:
        differences.append(f"family {case_family} vs {current_family}")
    case_structure = case.get("structure") or {}
    current_structure = current.get("structure") or {}
    for key in sorted(set(case_structure) & set(current_structure)):
        if case_structure[key] != current_structure[key]:
            differences.append(f"structure.{key}")
    for key in sorted(set(case_structure) ^ set(current_structure)):
        unknowns.append(f"structure.{key} unrecorded on one side")
    requested = current.get("solver_conditions") or {}
    actuals = [actual for actual in
               (case.get("actual_solving_conditions") or [])
               if isinstance(actual, dict)]
    requested_solver = requested.get("solver")
    if requested_solver:
        known_solvers = [actual.get("solver") for actual in actuals
                         if actual.get("solver")]
        if not known_solvers:
            unknowns.append("historical solver interface unrecorded")
            blocking_unknowns.append("solver interface unrecorded")
        elif requested_solver not in known_solvers:
            differences.append(
                f"solver {requested_solver} vs "
                + "/".join(sorted(set(known_solvers))))
    requested_config = requested.get("config") or {}
    if requested_config:
        if not actuals:
            unknowns.append("historical solving config unrecorded")
        for key, value in sorted(requested_config.items()):
            known = [actual.get("config") or {} for actual in actuals]
            if not any(key in config for config in known):
                unknowns.append(f"solving config {key} unrecorded")
                blocking_unknowns.append(f"solving config {key} unrecorded")
                continue
            if any(config.get(key) != value
                   for config in known if key in config):
                differences.append(f"solving config {key}")
    # A KNOWN difference always weakens the reference. An unknown auxiliary
    # STRUCTURE field is informational (it cannot prove a difference); an
    # unrecorded DECLARED solver/config is blocking because the comparability
    # of the solving setup cannot be confirmed.
    grade = ("strict" if strict and not differences
             and not blocking_unknowns else "broader_reference")
    return grade, unknowns, differences


def _g_case(row: Dict[str, Any], *, similarity: Dict[str, Any],
            relevance: str) -> Dict[str, Any]:
    evaluation = row.get("evaluation") or {}
    g_block = evaluation.get("g") or {}
    observed = g_block.get("observed")
    case: Dict[str, Any] = {
        "execution_id": row.get("execution_id"),
        "task_id": row.get("task_id"),
        "task_excerpt": _row_task_excerpt(row),
        "similarity": {"overall": similarity.get("overall"),
                       "method": similarity.get("method")},
        "relevance": relevance,
        "method": {
            "name": ((row.get("method") or {}).get("actual") or {}).get("name")
            or ((row.get("method") or {}).get("planned") or {}).get("name"),
            "steps": (_case_steps((row.get("method") or {}).get("actual"))
                      or _case_steps((row.get("method") or {}).get("planned"))),
            "basis": (row.get("method") or {}).get("basis"),
        },
        "status": row.get("status"),
    }
    if observed in (0, 1):
        case["g"] = {
            "observed": int(observed),
            "predicted": g_block.get("predicted"),
            "signed_error": g_block.get("signed_error"),
            "metric": g_block.get("metric"),
            "rule": g_block.get("observation_rule_version"),
        }
    else:
        case["g"] = {
            "observed": None,
            "metric": g_block.get("metric"),
            "rule": g_block.get("observation_rule_version"),
            "reason": (g_block.get("reason")
                       or "no attributable G label in this closed pair"),
        }
    recorded = evaluation.get("attribution") or g_block.get("attribution")
    if isinstance(recorded, dict) and recorded:
        case["recorded_cause"] = recorded
    if g_block.get('attribution_ref'):
        case['g']['attribution_ref'] = g_block['attribution_ref']
    if evaluation.get("bound_execution_id"):
        case["bound_execution_id"] = evaluation["bound_execution_id"]
    if evaluation.get("task_check") is not None:
        case["task_check"] = (evaluation.get("task_check") or {}).get("state")
    else:
        case["task_check"] = None
    if row.get("failure_classes"):
        case["technical_failure_classes"] = list(row["failure_classes"])[:3]
    if evaluation.get("evaluation_id"):
        case["evaluation_id"] = evaluation["evaluation_id"]
    case["source_refs"] = [str(row.get("execution_id"))] + (
        [str(evaluation["evaluation_id"])]
        if evaluation.get("evaluation_id") else [])
    return case


def select_g_history_evidence(
        history: Optional[Dict[str, Any]], *,
        candidate: Any, convention: Optional[Dict[str, Any]] = None,
        scope: Optional[str] = None, task_id: str = "",
        limit: Optional[int] = None) -> Dict[str, Any]:
    """G local cases: method/condition relevance over the shared pool.

    A case is NEVER selected for a generic word overlap: a shared word does
    not raise its relevance above ``task_similar``. ``similar_method``
    requires the CANDIDATE-METHOD view to reach its own threshold, and
    ``same_task`` is a deterministic fact. Cases without an observed label
    are carried with ``g.observed=null`` and a reason — an execution error is
    not a G modelling failure.
    """
    history = history or {}
    config = history.get("config") or history_config()
    cap = int(limit or config.get("max_local_cases") or DEFAULT_MAX_LOCAL_CASES)
    convention = convention or {}
    wanted_metric = convention.get("metric")
    wanted_rule = convention.get("observation_rule_version")
    pool = history.get("pool") or []
    method_floor = float(config.get("method_similarity_threshold") or 0.0)
    # `overall` is already relative to each view's own floor, so the
    # admission floor on this value is 1.0.
    relevance_floor = 1.0
    ranked: List[Tuple[Tuple[int, int, int, float, str], Dict[str, Any]]] = []
    for row in pool:
        evaluation = row.get("evaluation") or {}
        if "bound_execution_id" in evaluation:
            bound = evaluation.get("bound_execution_id")
            if not bound or str(bound) != str(row.get("execution_id")):
                # The pair's G label belongs to the bound attempt. Attaching
                # it to a DIFFERENT execution in the same window would credit
                # one method with another method's outcome. A pair that
                # records no bound execution cannot attribute its label at all.
                evaluation = {}
        g_block = evaluation.get("g") or {}
        metric = g_block.get("metric")
        rule = g_block.get("observation_rule_version")
        if wanted_metric and metric and metric != wanted_metric:
            continue
        if wanted_rule and rule and rule != wanted_rule:
            continue
        similarity = _method_scores(row, history, candidate)
        same_task = bool(task_id) and str(row.get("task_id")) == str(task_id)
        similar_method = (similarity.get("method") is not None
                          and similarity["method"] >= method_floor)
        same_conditions = (row.get("structural_match") == "same_cell")
        if same_task:
            relevance = "same_task"
        elif similar_method:
            relevance = "similar_method"
        elif same_conditions:
            relevance = "similar_conditions"
        elif (similarity.get("overall") or 0.0) >= relevance_floor:
            relevance = "task_similar"
        else:
            continue
        observed = g_block.get("observed")
        has_label = 1 if observed in (0, 1) else 0
        # Label first, then a real method match, then same task, then score.
        key = (has_label, 1 if similar_method else 0,
               1 if same_task else 0,
               float(similarity.get("overall") or 0.0),
               str(row.get("execution_id")))
        ranked.append((key, _g_case(row, similarity=similarity,
                                    relevance=relevance)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    cases = [case for _key, case in ranked[:cap]]
    observed = sum(1 for case in cases
                   if (case.get("g") or {}).get("observed") in (0, 1))
    block = {
        "version": "wm-g-history/1",
        "selection": {
            "n_pool": len(pool),
            "n_condition_relevant": len(ranked),
            "n_included": len(cases),
            "n_omitted": max(0, len(ranked) - len(cases)),
            "n_with_observed_label": observed,
            "limit": cap,
            "candidate_method_digest": _method_digest(
                getattr(candidate, "method", None)),
            "basis": ("embedding-similarity pool, reranked by G relevance: "
                      "an observed label first, then a real candidate-method "
                      "match, then same-task; a generic word overlap alone "
                      "does NOT qualify as a method match"),
            "similarity_threshold": relevance_floor,
            "method_similarity_threshold": method_floor,
        },
        "cases": cases,
        "global_history": history.get("global", {}).get("g") or {},
        "degraded": copy.deepcopy(history.get("degraded") or []),
        "missing": list(history.get("missing") or []),
        "note": ("local cases are executed/closed-episode facts selected by "
                 "text similarity and reranked for G, not a candidate "
                 "correction coefficient. `g.observed` is the recorded "
                 "label; null means UNKNOWN, never failure. `recorded_cause` "
                 "appears only when the close-out recorded an attribution. "
                 "global_history is a diagnostic background."),
    }
    if not cases:
        block["missing"].append(
            "g_candidate_evidence: no local G case reached the relevance "
            "rule; the global G diagnostic is background only")
    return block


def _provider_global_cost(global_cost: Dict[str, Any]) -> Dict[str, Any]:
    """The provider copy of the global cost block (no long source lists)."""
    block = copy.deepcopy(global_cost or {})
    dimensions = block.get("dimensions") or {}
    if (dimensions.get("retries") or {}).get("observation_rule_version") != RETRIES_FUTURE_VERSION:
        dimensions.pop("retries", None)
    for entry in (block.get("dimensions") or {}).values():
        if isinstance(entry, dict):
            entry.pop("source_refs", None)
    return block


def _c_case(row: Dict[str, Any], *, similarity: Dict[str, Any],
            grade: str, reasons: Sequence[str],
            differences: Sequence[str],
            current: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = row.get("evaluation") or {}
    cost = evaluation.get("cost") or {}
    values = {dim: cost.get("values", {}).get(dim)
              for dim in PREDICTED_COST_DIMENSIONS}
    if cost.get("retry_observation_rule_version") != RETRIES_FUTURE_VERSION:
        values["retries"] = None
    missing = [dim for dim, value in values.items() if value is None]
    predicted = {}
    for dim, entry in (evaluation.get("cost_predicted_actual") or {}).items():
        if dim == "retries" and cost.get("retry_observation_rule_version") != RETRIES_FUTURE_VERSION:
            continue
        if entry.get("predicted") is not None:
            predicted[dim] = entry["predicted"]
    case: Dict[str, Any] = {
        "execution_id": row.get("execution_id"),
        "task_id": row.get("task_id"),
        "task_excerpt": _row_task_excerpt(row),
        "grade": grade,
        "similarity": {"overall": similarity.get("overall"),
                       "work": similarity.get("method")},
        "cost": {dim: value for dim, value in values.items()
                 if value is not None},
        "missing_dimensions": missing,
        "scope": cost.get("scope"),
        "window_coverage": cost.get("window_coverage"),
        # A remaining-task cost window is ONE reference. `covers_execution_ids`
        # says which executions it spans; it is not repeated as one case per
        # execution.
        "covers_execution_ids": list(
            evaluation.get("covers_execution_ids") or []),
        "work": {
            "name": ((row.get("method") or {}).get("actual") or {}).get("name")
            or ((row.get("method") or {}).get("planned") or {}).get("name"),
            "steps": (_case_steps((row.get("method") or {}).get("actual"))
                      or _case_steps((row.get("method") or {}).get("planned"))),
            "reusable_artifacts": bool(cost.get("reusable_artifacts")),
        },
    }
    if evaluation.get("evaluation_id"):
        case["evaluation_id"] = evaluation["evaluation_id"]
    if predicted:
        case["predicted"] = predicted
    if grade == "strict":
        case["relevance_weight"] = 1.0
    else:
        if reasons:
            case["unknown_conditions"] = list(dict.fromkeys(reasons))[:6]
        if differences:
            case["differences"] = list(dict.fromkeys(differences))[:6]
        # A comparability deduction, not an exclusion: the more known
        # differences, the weaker the reference.
        case["relevance_weight"] = round(
            max(0.3, 0.9 - 0.1 * len(set(differences))), 2)
    case["source_refs"] = [str(row.get("execution_id"))] + (
        [str(evaluation["evaluation_id"])]
        if evaluation.get("evaluation_id") else [])
    return case


def select_c_history_evidence(
        history: Optional[Dict[str, Any]], *,
        context: Any, candidate: Any,
        limit: Optional[int] = None) -> Dict[str, Any]:
    """C local cases: comparable remaining-task costs over the shared pool.

    The ONE hard constraint is the measurement scope: an attempt-scope or
    inner-solver runtime is never passed off as a remaining-task cost. There
    is NO phase classification: the candidate's current work/artifacts are
    ordinary context, and similarity + family/structure/solver consistency
    decide strict vs weak reference.
    """
    history = history or {}
    config = history.get("config") or history_config()
    cap = int(limit or config.get("max_local_cases") or DEFAULT_MAX_LOCAL_CASES)
    current = _current_cost_conditions(context, candidate)
    pool = history.get("pool") or []
    ranked: List[Tuple[Tuple[int, int, int, float, str],
                       Dict[str, Any]]] = []
    counts = {"strict": 0, "broader_reference": 0, "unavailable": 0,
              "without_observed_values": 0}
    seen_evaluations: set = set()
    for row in pool:
        evaluation = row.get("evaluation") or {}
        if "cost_carrier_execution_id" in evaluation:
            carrier = evaluation.get("cost_carrier_execution_id")
            if not carrier or str(carrier) != str(row.get("execution_id")):
                # This execution is part of a window whose cost reference is
                # already carried by ONE canonical execution; it must not be
                # a second copy of the same sample.
                continue
        evaluation_id = evaluation.get("evaluation_id")
        if evaluation_id is not None:
            if str(evaluation_id) in seen_evaluations:
                continue
            seen_evaluations.add(str(evaluation_id))
        cost = evaluation.get("cost") or {}
        values = {dim: value for dim, value in (cost.get("values") or {}).items()
                  if dim in PREDICTED_COST_DIMENSIONS
                  and _number(value) is not None}
        if cost.get("retry_observation_rule_version") != RETRIES_FUTURE_VERSION:
            values.pop("retries", None)
        if not values:
            counts["without_observed_values"] += 1
            continue
        similarity = _method_scores(row, history, candidate)
        strict = (cost.get("scope") == current.get("scope")
                  and _known(cost.get("family"))
                  and cost.get("family") == current.get("family"))
        grade, reasons, differences = _cost_grade(
            cost, current, strict=strict)
        counts[grade] += 1
        if grade == "unavailable":
            continue
        key = (1 if grade == "strict" else 0,
               -len(set(differences)),
               len(values),
               float(similarity.get("overall") or 0.0),
               str(row.get("execution_id")))
        ranked.append((key, _c_case(row, similarity=similarity, grade=grade,
                                    reasons=reasons,
                                    differences=differences,
                                    current=current)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    cases = [case for _key, case in ranked[:cap]]
    # Keep the grades visible after the cap: a weak reference must never
    # look like the strict population.
    coverage = {
        "strict": {
            "n_cases_available": counts["strict"],
            "n_cases_included": sum(1 for case in cases
                                    if case["grade"] == "strict"),
        },
        "broader_reference": {
            "n_cases_available": counts["broader_reference"],
            "n_cases_included": sum(1 for case in cases
                                    if case["grade"] == "broader_reference"),
        },
        "unavailable": counts["unavailable"],
        "n_without_observed_values": counts["without_observed_values"],
        "n_cases_included": len(cases),
        "n_cases_omitted": max(0, len(ranked) - len(cases)),
        "unavailable_reason": (
            "no historical execution carries a complete remaining-to-task "
            "observation comparable to this decision"
            if not ranked else None),
    }
    block = {
        "version": "wm-c-history/1",
        "observation_rule_version": "wm-c-window/1",
        "current_conditions": {
            "scope": current.get("scope"),
            "family": current.get("family"),
            "structure": current.get("structure"),
            "solver": (current.get("solver_conditions") or {}).get("solver"),
            "reusable_artifacts": current.get("reusable_artifacts"),
            "phase_used": False,
        },
        "candidate_work": {
            "steps_ref": "candidate.method.steps",
            "measurement_scope": current.get("scope"),
        },
        "cases": cases,
        "reference_coverage": coverage,
        "global_cost": _provider_global_cost(
            (history.get("global") or {}).get("c") or {}),
        "evidence_status": ("strict_and_broader" if coverage["strict"]["n_cases_included"]
                            and coverage["broader_reference"]["n_cases_included"]
                            else "strict" if coverage["strict"]["n_cases_included"]
                            else "broader_reference_only"
                            if coverage["broader_reference"]["n_cases_included"]
                            else "insufficient_evidence"),
        "degraded": copy.deepcopy(history.get("degraded") or []),
        "missing": list(history.get("missing") or []),
        "applicability": ("the ONLY hard exclusion is an incomparable "
                          "measurement window. strict = same known family and "
                          "no known structure/solver/config difference. A "
                          "known difference (solver, config, structure, "
                          "family) or an unknown condition keeps the case as "
                          "broader_reference with `differences`/"
                          "`unknown_conditions` and a lower "
                          "`relevance_weight` — it is not discarded. No phase "
                          "is used. Missing dimensions are unknown, never "
                          "zero."),
        "note": ("Totals already include the work in each reference case. "
                 "Compare added/saved candidate steps; count each step once. "
                 "A broader_reference is NOT a strict calibrated prior. "
                 "global_cost is background; never raise every cost by a "
                 "global ratio."),
    }
    if not cases:
        block["missing"].append(
            "cost_evidence: no historical execution carries a complete "
            "remaining-task cost observation for this decision")
    return block


def refresh_c_history_coverage(block: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the strict/weak counts true after a provider-budget trim."""
    block = block if isinstance(block, dict) else {}
    coverage = block.setdefault("reference_coverage", {})
    cases = block.get("cases") or []
    for grade in ("strict", "broader_reference"):
        entry = coverage.setdefault(grade, {})
        entry["n_cases_included"] = sum(1 for case in cases
                                        if case.get("grade") == grade)
        entry["n_cases_omitted"] = max(
            0, int(entry.get("n_cases_available") or 0)
            - entry["n_cases_included"])
    coverage["n_cases_included"] = len(cases)
    block["evidence_status"] = (
        "strict_and_broader"
        if coverage["strict"]["n_cases_included"]
        and coverage["broader_reference"]["n_cases_included"]
        else "strict" if coverage["strict"]["n_cases_included"]
        else "broader_reference_only"
        if coverage["broader_reference"]["n_cases_included"]
        else "insufficient_evidence")
    return block


def refresh_g_history_coverage(block: Dict[str, Any]) -> Dict[str, Any]:
    block = block if isinstance(block, dict) else {}
    selection = block.setdefault("selection", {})
    cases = block.get("cases") or []
    selection["n_cases_included"] = len(cases)
    selection["n_cases_omitted"] = max(
        0, int(selection.get("n_condition_relevant") or 0) - len(cases))
    selection["n_with_observed_label"] = sum(
        1 for case in cases
        if (case.get("g") or {}).get("observed") in (0, 1))
    return block


def refresh_r_history_coverage(block: Dict[str, Any]) -> Dict[str, Any]:
    block = block if isinstance(block, dict) else {}
    selection = block.setdefault("selection", {})
    cases = block.get("cases") or []
    selection["n_cases_included"] = len(cases)
    selection["n_cases_omitted"] = max(
        0, int(selection.get("n_condition_relevant") or 0) - len(cases))
    block["support"] = ("local_cases_with_global_base_rates" if cases
                        and (block.get("global_event_base_rates") or {})
                        else "local_cases" if cases
                        else "global_base_rates_only"
                        if (block.get("global_event_base_rates") or {})
                        else "insufficient_evidence")
    return block


def _r_case(row: Dict[str, Any], *, similarity: Dict[str, Any],
            relevance: str) -> Dict[str, Any]:
    evaluation = row.get("evaluation") or {}
    observed = {str(entry.get("event")): entry
                for entry in (evaluation.get("risk_actual") or [])
                if isinstance(entry, dict)}
    events: Dict[str, Any] = {}
    for name in _R_EVENTS:
        local = row.get("r_events", {}).get(name) or {}
        paired = observed.get(name) or {}
        if name == "implementation_failure":
            from or_harness.world_model.contracts import R_OBSERVATION_VERSION
            if paired.get("observation_rule_version") != R_OBSERVATION_VERSION:
                paired = {}
            if local.get("observation_rule_version") != R_OBSERVATION_VERSION:
                local = {"label": None, "basis": "historical formal diagnostic; implementation-window coverage unknown"}
        label = paired.get("label", local.get("label"))
        item: Dict[str, Any] = {
            "label": label,
            "basis": local.get("basis") or paired.get("label_basis"),
        }
        predicted = paired.get("predicted")
        if predicted is not None:
            item["predicted"] = predicted
            if paired.get("brier") is not None:
                item["brier"] = paired["brier"]
        events[name] = item
    case: Dict[str, Any] = {
        "execution_id": row.get("execution_id"),
        "task_id": row.get("task_id"),
        "task_excerpt": _row_task_excerpt(row),
        "similarity": {
            "overall": similarity.get("overall"),
            "technical": (similarity.get("method")
                          if similarity.get("method") is not None
                          else similarity.get("conditions")),
        },
        "relevance": relevance,
        "solver_interface": row.get("solver_interface"),
        "status": row.get("status"),
        "events": events,
    }
    if row.get("failure_classes"):
        case["failure_classes"] = list(row["failure_classes"])[:3]
    if row.get("implementation_evidence"):
        case["implementation_evidence"] = copy.deepcopy(row["implementation_evidence"])
    if row.get("formal_technical_outcome"):
        case["formal_technical_outcome"] = copy.deepcopy(row["formal_technical_outcome"])
    if row.get("has_reusable_script"):
        case["reusable_script"] = True
    if evaluation.get("evaluation_id"):
        case["evaluation_id"] = evaluation["evaluation_id"]
        if evaluation.get("bound_execution_id"):
            case["bound_execution_id"] = evaluation["bound_execution_id"]
        case["evidence_status"] = "closed_pair"
    else:
        # The execution is visible at the decision moment but its episode is
        # not closed: its technical label is an EXECUTION fact, not a
        # calibration sample. The global base rates use closed attempts only.
        case["evidence_status"] = "execution_only"
    case["source_refs"] = [str(row.get("execution_id"))] + (
        [str(evaluation["evaluation_id"])]
        if evaluation.get("evaluation_id") else [])
    return case


def select_r_history_evidence(
        history: Optional[Dict[str, Any]], *,
        context: Any, candidate: Any,
        limit: Optional[int] = None) -> Dict[str, Any]:
    """R local cases: technical similarity + global event rates.

    The three R events keep their existing definitions/contract. A local
    case carries the observed label from the SAME bound attempt when a
    closed pair exists, else the execution record's technical event label.
    A sparse condition set never becomes a per-condition probability, and
    the global base rate is a separate background block.
    """
    history = history or {}
    config = history.get("config") or history_config()
    cap = int(limit or config.get("max_local_cases") or DEFAULT_MAX_LOCAL_CASES)
    pool = history.get("pool") or []
    interface = getattr(candidate, "solver", None) or (
        ((getattr(context, "solving_context", None) or {}).get(
            "execution_context") or {}).get("solver_interface"))
    method_floor = float(config.get("method_similarity_threshold") or 0.0)
    ranked: List[Tuple[Tuple[int, int, float, str], Dict[str, Any]]] = []
    for row in pool:
        evaluation = row.get("evaluation") or {}
        if "bound_execution_id" in evaluation:
            bound = evaluation.get("bound_execution_id")
            if not bound or str(bound) != str(row.get("execution_id")):
                # The technical labels are per ACTUAL execution. A closed
                # pair's risk label describes its bound attempt only; another
                # window execution keeps its own record-derived events.
                row = {**row, "evaluation": {}}
        similarity = _method_scores(row, history, candidate)
        same_interface = bool(interface) and row.get("solver_interface") == interface
        technical = (similarity.get("method") is not None
                     and similarity["method"] >= method_floor)
        if same_interface:
            relevance = "same_solver_interface"
        elif technical:
            relevance = "similar_technical_work"
        elif (similarity.get("overall") or 0.0) >= 1.0:
            relevance = "similar_execution"
        else:
            continue
        events = (row.get("r_events") or {})
        known = sum(1 for entry in events.values()
                    if (entry or {}).get("label") in (0, 1))
        key = (known, 1 if same_interface else 0,
               float(similarity.get("overall") or 0.0),
               str(row.get("execution_id")))
        ranked.append((key, _r_case(row, similarity=similarity,
                                    relevance=relevance)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    cases = [case for _key, case in ranked[:cap]]
    global_rates = (history.get("global", {}).get("r") or {})
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION
    if global_rates.get("observation_rule_version") != R_OBSERVATION_VERSION:
        global_rates = {}
    block = {
        "version": "wm-r-history/1",
        "solver_interface": interface,
        "event_definitions": {
            "implementation_failure": (
                "evidenced substantive defect in the selected candidate implementation window; "
                "zero requires complete implementation coverage; historical formal receipts are qualitative only"),
            "timeout": (
                "the harness wall-clock limit INTERRUPTED the attempt "
                "before a usable result"),
            "environment_failure": (
                "an environment/sandbox/dependency blockage stopped the "
                "attempt"),
        },
        "selection": {
            "n_pool": len(pool),
            "n_condition_relevant": len(ranked),
            "n_included": len(cases),
            "n_omitted": max(0, len(ranked) - len(cases)),
            "limit": cap,
            "basis": ("shared embedding pool reranked for technical "
                      "relevance: same solver interface, then candidate "
                      "method similarity, then general similarity"),
        },
        "cases": cases,
        "global_event_base_rates": global_rates.get("events") or {},
        "global_prediction_performance": {
            name: {key: entry.get(key) for key in (
                "n_reported_probabilities", "n_scorable",
                "n_scored_distinct_episodes", "mean_brier", "brier_support")}
            for name, entry in (global_rates.get("events") or {}).items()},
        "global_scope": global_rates.get("scope"),
        "global_rate_basis": global_rates.get("rate_basis"),
        "missing": [] if interface else ["solver interface unknown"],
        "degraded": copy.deepcopy(history.get("degraded") or []),
        "note": ("prefer a local technical case for THIS candidate; the "
                 "separate global_event_base_rates block is broad background, "
                 "never the candidate's conditional probability. Unknown is "
                 "not 'did not happen'; the maximum event probability is NOT "
                 "a joint failure probability."),
    }
    block["missing"].extend(history.get("missing") or [])
    return refresh_r_history_coverage(block)


# ---------------------------------------------------------------------------
# decision cost scales (U normalisation) — no phase, no condition groups
# ---------------------------------------------------------------------------

def decision_cost_scales(context: Any, candidates: Sequence[Any],
                         configured: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Shared positive cost peaks for U normalization.

    Configured peaks win. Otherwise the peak is the p90 of the pre-decision
    measured remaining-task history (global C block frozen on the context) —
    the same population for every candidate, because a shared scale must not
    depend on which candidate it is scoring. Nothing from the CURRENT
    forecasts enters the peak.
    """
    configured = configured or {}
    if not isinstance(configured, dict):
        raise ValueError("cost_scales must be a dimension-to-peak object")
    history = getattr(context, "history_retrieval", None) or {}
    global_c = ((history.get("global") or {}).get("c")) or {}
    dimensions = global_c.get("dimensions") or {}
    created_at = getattr(context, "created_at", None)
    result: Dict[str, Any] = {}
    for dimension in PREDICTED_COST_DIMENSIONS:
        if dimension in configured:
            value = configured[dimension]
            info = (copy.deepcopy(value) if isinstance(value, dict)
                    else {"peak": value})
            result[dimension] = {
                **info, "source": "explicit_plan_limits",
                "reference": info.get("reference")
                or "limits.cost_scales." + dimension,
                "configuration_ref": "limits.cost_scales." + dimension,
                "as_of": created_at}
            continue
        entry = dimensions.get(dimension) or {}
        if dimension == "retries" and entry.get("observation_rule_version") != RETRIES_FUTURE_VERSION:
            continue
        peak = _number(entry.get("p90"))
        if peak is not None and peak > 0:
            result[dimension] = {
                "peak": peak,
                "scope": "remaining_to_task_end",
                "source": "pre_decision_measured_history",
                "observation_rule_version": "wm-c-window/1",
                "as_of": created_at,
                "n": entry.get("n"),
                "source_refs": list(entry.get("source_refs") or [])[:6],
                "statistic": "p90 of the global pre-decision remaining-task "
                             "distribution",
            }
    return result
