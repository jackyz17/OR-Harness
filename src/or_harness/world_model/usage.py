"""One token accounting rule for every model call.

The framework makes model calls in several places (strategy-outcome
prediction, capability-evolution prediction, the legacy prediction service).
Each used to read ``usage.completion_tokens`` on its own, which had two
effects:

* **prompt tokens vanished.** A call that billed 9000 prompt + 100
  completion tokens was recorded as ``llm_tokens=100`` — five orders of
  magnitude off — because only the completion side was read.
* **three near-identical copies** of the extraction drifted apart.

This module is the ONE place a provider's ``usage`` is turned into a
:class:`~or_harness.core.schema.CostVector`. Its rules:

* ``llm_tokens`` is the **full口径** — the TOTAL the provider billed:
  ``prompt_tokens + completion_tokens``. When only one side is reported it
  falls back to that side (never a guess), and when neither is reported the
  dimension stays UNMEASURED (never zero-as-free).
* cache / reasoning tokens are recorded as **sub-facts**, never added to the
  total: ``reasoning_tokens`` is already counted INSIDE ``completion_tokens``
  (and ``cached_tokens`` inside ``prompt_tokens``) on every provider this
  build targets, so adding them would double-count.
* the breakdown travels with the call so a reader can tell WHICH口径
  produced the number (see :data:`TOKEN_BASIS_*`).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from or_harness.core.schema import CostVector

#: The token number is the provider's FULL total (prompt + completion).
TOKEN_BASIS_TOTAL = "provider_total"
#: Only the completion side was reported: the total is that side alone.
TOKEN_BASIS_COMPLETION_ONLY = "completion_only"
#: Only the prompt side was reported.
TOKEN_BASIS_PROMPT_ONLY = "prompt_only"

#: Every basis this module can produce.
TOKEN_BASES = (TOKEN_BASIS_TOTAL, TOKEN_BASIS_COMPLETION_ONLY,
               TOKEN_BASIS_PROMPT_ONLY)


def _as_count(value: Any) -> Optional[float]:
    """A non-negative finite token count, or None when not a real number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number >= 0:
            return number
    return None


def token_breakdown(usage: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a provider ``usage`` object into the ONE accounting shape.

    Returns::

        {
          "llm_tokens": float | None,      # the FULL口径 total, or None
          "basis": "provider_total" | "completion_only" | "prompt_only",
          "prompt_tokens": float | None,   # reported sub-fact
          "completion_tokens": float | None,
          "reasoning_tokens": float | None,  # INSIDE completion, not added
          "cached_tokens": float | None,     # INSIDE prompt, not added
          "note": str,                      # how the total was formed
        }

    Nothing is fabricated: a side the provider did not report is ``None``,
    and when neither side is reported ``llm_tokens`` is ``None`` (unknown),
    never 0.0.
    """
    usage = usage or {}
    prompt = _as_count(usage.get("prompt_tokens"))
    completion = _as_count(usage.get("completion_tokens"))
    details = usage.get("completion_tokens_details") or {}
    reasoning = _as_count(
        details.get("reasoning_tokens")
        if isinstance(details, Mapping) else None)
    if reasoning is None:
        reasoning = _as_count(usage.get("reasoning_tokens"))
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = _as_count(
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, Mapping) else None)
    if cached is None:
        cached = _as_count(usage.get("cached_tokens"))
    # The provider may report a total directly; prefer it when present, but
    # only after the two sides are known (a total alone still means the
    # total口径).
    reported_total = _as_count(usage.get("total_tokens"))

    if prompt is not None and completion is not None:
        total, basis = prompt + completion, TOKEN_BASIS_TOTAL
        note = ("llm_tokens is the FULL口径 total (prompt + completion); "
                "reasoning/cached tokens are sub-facts already inside those "
                "sides and are never added again")
    elif completion is not None:
        total, basis = completion, TOKEN_BASIS_COMPLETION_ONLY
        note = ("only completion tokens were reported: llm_tokens is that "
                "side alone, a LOWER bound on the true total")
    elif prompt is not None:
        total, basis = prompt, TOKEN_BASIS_PROMPT_ONLY
        note = ("only prompt tokens were reported: llm_tokens is that side "
                "alone, a LOWER bound on the true total")
    elif reported_total is not None:
        total, basis = reported_total, TOKEN_BASIS_TOTAL
        note = ("the provider reported only a total_tokens figure; it is "
                "used verbatim")
    else:
        total, basis = None, None
        note = ("the provider reported no token usage: llm_tokens stays "
                "UNKNOWN, never zero-as-free")
    return {
        "llm_tokens": total,
        "basis": basis,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "note": note,
    }


def usage_cost_vector(usage: Optional[Mapping[str, Any]],
                      latency_s: Any = None
                      ) -> Tuple[Optional[CostVector], Optional[Dict[str, Any]]]:
    """The call's OWN measured cost from provider usage.

    Returns ``(vector, breakdown)``. ``vector`` is ``None`` when neither a
    token count nor a latency was reported (there is no cost to record —
    never a zero-filled vector that reads as "free"). ``breakdown`` is the
    :func:`token_breakdown` result, or ``None`` when no usage was reported;
    it is what a caller stores alongside the cost so the口径 is traceable.

    The vector marks ONLY the dimensions actually reported: an unreported
    dimension stays unmeasured.
    """
    breakdown = token_breakdown(usage) if usage else None
    if not usage and latency_s is None:
        return None, None
    vector = CostVector(measured=set())
    if breakdown is not None and breakdown["llm_tokens"] is not None:
        vector.llm_tokens = float(breakdown["llm_tokens"])
        vector.mark_measured("llm_tokens")
    if latency_s is not None:
        vector.latency_s = float(latency_s)
        vector.mark_measured("latency_s")
    if not vector.measured_dims():
        return None, breakdown
    return vector, breakdown


# ---------------------------------------------------------------------------
# host usage ingestion: the outer agent framework reports the attempt's usage
# ---------------------------------------------------------------------------

#: The attempt-level usage report a HOST framework provides through a hook,
#: log or storage. ``or_harness`` cannot see the outer agent's own LLM calls
#: (it runs inside the host), so the host must report them. This is the SHAPE
#: the framework accepts — it is deliberately permissive: every key is
#: optional, and a hook that reports only what it knows is still valid.
#:
#: ``or_harness`` does NOT count tokens itself and never re-implements a
#: tokenizer: it consumes what the host reports. ``source`` names the host
#: so a reader can tell an OpenClaw report from a Hermes one, and
#: ``calls`` optionally breaks the total down per model call.
HOST_USAGE_KEYS: Tuple[str, ...] = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "reasoning_tokens", "cached_tokens", "model", "source", "calls",
)

#: Hosts this build knows how to NORMALIZE a report from. The adapters below
#: are LIGHT: they read the fields each host documents. A host whose hook or
#: field names differ can report the canonical shape directly (see
#: :func:`normalize_host_usage`), so an unverified host is never a blocker.
HOST_ADAPTERS: Tuple[str, ...] = ("openclaw", "hermes", "generic")


def _sum_tokens(values: Any) -> Optional[float]:
    total = 0.0
    seen = False
    for value in (values or []):
        count = _as_count(value)
        if count is not None:
            total += count
            seen = True
    return total if seen else None


def normalize_host_usage(report: Mapping[str, Any],
                         source: Optional[str] = None
                         ) -> Dict[str, Any]:
    """Normalize a HOST usage report into the canonical attempt-level shape.

    The canonical shape (what :func:`host_usage_cost_vector` consumes)::

        {
          "prompt_tokens": float | None,
          "completion_tokens": float | None,
          "reasoning_tokens": float | None,
          "cached_tokens": float | None,
          "model": str | None,
          "source": str,          # which host/hook reported it
          "calls": [ {...per-call...} ] | None,
        }

    The adapters are tolerant on purpose — an installed host version may
    omit or rename a field, and a report that carries the numbers under a
    known alias must still be read. Unknown shapes pass through unchanged, so
    a caller can always supply the canonical form directly.
    """
    report = dict(report or {})
    source = (source or report.get("source") or "generic").lower()
    calls = report.get("calls") or report.get("events")
    prompt = _as_count(report.get("prompt_tokens"))
    completion = _as_count(report.get("completion_tokens"))
    reasoning = _as_count(report.get("reasoning_tokens"))
    cached = _as_count(report.get("cached_tokens"))
    total = _as_count(report.get("total_tokens"))
    # The model name travels under a few names across hosts.
    model = report.get("model") or report.get("model_id") \
        or report.get("model_name")
    # -- per-call breakdown ------------------------------------------------
    if isinstance(calls, (list, tuple)):
        normalized_calls = []
        for entry in calls:
            if not isinstance(entry, Mapping):
                continue
            usage = entry.get("usage") if isinstance(entry.get("usage"),
                                                     Mapping) else entry
            normalized_calls.append({
                "prompt_tokens": _as_count(usage.get("prompt_tokens")),
                "completion_tokens": _as_count(usage.get("completion_tokens")),
                "reasoning_tokens": _as_count(usage.get("reasoning_tokens")
                                              or (usage.get(
                                                  "completion_tokens_details")
                                                  or {}).get(
                                                      "reasoning_tokens")),
                "cached_tokens": _as_count(usage.get("cached_tokens")),
                "model": usage.get("model") or entry.get("model") or model,
            })
        if normalized_calls:
            # A per-call breakdown is authoritative when the TOTALS are
            # absent: sum the parts (no double count — a call's prompt and
            # completion are its own).
            if prompt is None:
                prompt = _sum_tokens(c["prompt_tokens"]
                                     for c in normalized_calls)
            if completion is None:
                completion = _sum_tokens(c["completion_tokens"]
                                         for c in normalized_calls)
            if reasoning is None:
                reasoning = _sum_tokens(c["reasoning_tokens"]
                                        for c in normalized_calls)
    else:
        normalized_calls = None
    # ``total_tokens`` alone is kept so :func:`token_breakdown` can use it
    # verbatim when neither side is known.
    if prompt is None and completion is None and total is not None:
        prompt = None
        completion = None
        # Represent a total-only report by overloading prompt=None and
        # letting token_breakdown see total_tokens.
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "model": str(model) if model else None,
        "source": source,
        "calls": normalized_calls,
    }


def host_usage_cost_vector(report: Optional[Mapping[str, Any]],
                           latency_s: Any = None, *,
                           source: Optional[str] = None
                           ) -> Tuple[Optional[CostVector],
                                      Optional[Dict[str, Any]]]:
    """A HOST attempt-level usage report -> (cost vector, breakdown).

    The vector's ``llm_tokens`` is the same full口径 total; its provenance is
    recorded in the breakdown as ``host:<source>`` so a reader can tell a
    host-reported attempt cost from a provider-reported CALL cost.

    A report that carries no token number yields ``(None, breakdown)`` — an
    unknown cost, never zero.
    """
    if report is None:
        return None, None
    normalized = normalize_host_usage(report, source=source)
    vector, breakdown = usage_cost_vector(normalized, latency_s)
    if breakdown is not None:
        breakdown["host"] = normalized.get("source")
        breakdown["model"] = normalized.get("model")
        breakdown["scope"] = "attempt"
        breakdown["calls"] = normalized.get("calls")
    return vector, breakdown


# ---------------------------------------------------------------------------
# token口径 MIXING: a legacy completion-only figure is a different quantity
# ---------------------------------------------------------------------------

#: Default口径 for a record whose token figure predates the full-口径 rule.
LEGACY_TOKEN_BASIS = TOKEN_BASIS_COMPLETION_ONLY


def token_basis_of(record: Any) -> Optional[str]:
    """The token口径 a RECORD's ``llm_tokens`` was recorded under.

    ``None`` when the record carries no ``llm_tokens`` measurement at all.
    A record written by this build carries
    ``execution_features.cost_provenance.llm_tokens.basis``; a record from
    before this round carries none, and its figure is the LEGACY
    completion-only number — a LOWER bound, never comparable with a
    full-口径 total. Callers that pool token figures across records must
    group by this basis (or exclude the legacy ones), because summing a
    completion-only figure with a full total is a silent unit error.
    """
    cost = getattr(record, "cost", None)
    if cost is None or "llm_tokens" not in cost.measured_dims():
        return None
    provenance = (getattr(record, "execution_features", {})
                  or {}).get("cost_provenance") or {}
    entry = provenance.get("llm_tokens")
    if isinstance(entry, dict) and entry.get("basis"):
        return str(entry["basis"])
    return LEGACY_TOKEN_BASIS


def token_bases_mixed(records: Sequence[Any]) -> bool:
    """Whether a set of records mixes token口径 that must not be pooled."""
    bases = {b for b in (token_basis_of(r) for r in records) if b}
    return len(bases) > 1


