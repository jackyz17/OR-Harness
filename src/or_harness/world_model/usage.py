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

import json
from pathlib import Path
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
#:
#: IDEMPOTENCY. A host hook may fire more than once for one attempt (a
#: retry of the hook, a re-read of the same log line, a replay). Every call
#: entry may therefore carry an ``id`` (or ``call_id``) — the provider's own
#: identifier for that model call — and a report may carry a
#: ``report_id``/``usage_id``. When they are present,
#: :func:`dedupe_host_reports` collapses repeated events so the same tokens
#: are never counted twice; when they are absent the report is applied once
#: (``replace`` is idempotent), so a late duplicate can never double-count.
HOST_USAGE_KEYS: Tuple[str, ...] = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "reasoning_tokens", "cached_tokens", "model", "source", "calls",
    "report_id", "usage_id", "late", "final",
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
    # verbatim when neither side is known (a provider that reports only a
    # total is still reporting the FULL口径).
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


def dedupe_host_reports(reports: Sequence[Mapping[str, Any]]
                        ) -> Tuple[list, Dict[str, Any]]:
    """Collapse repeated host usage events so nothing is counted twice.

    A host hook can fire more than once for the same model call (a retried
    hook, a log re-read, a replay of the same storage row). ``replace`` mode
    already makes re-applying the SAME numbers idempotent, but a report that
    arrives TWICE with per-call entries would otherwise sum both copies.

    The rule is small and uses only what the host already reports:

    * a report carrying ``report_id``/``usage_id`` seen before is dropped
      whole (it is the same event);
    * otherwise, call entries are keyed by their own ``id``/``call_id``;
      a call id seen before is dropped. A LATE report (``late=True`` or
      ``final=True``) for a known call REPLACES the earlier figure rather
      than adding to it: a correction supersedes, it does not accumulate;
    * entries with no id are kept as-is — deduplication is never guessed.

    Returns ``(kept_reports, summary)`` where ``summary`` reports what was
    dropped/replaced so the caller can record it honestly.
    """
    kept: list = []
    seen_reports: set = set()
    seen_calls: Dict[str, Tuple[int, int]] = {}
    dropped_reports = 0
    dropped_calls = 0
    replaced_calls = 0
    for report in (reports or []):
        if not isinstance(report, Mapping):
            continue
        report = dict(report)
        report_id = report.get("report_id") or report.get("usage_id")
        if report_id is not None:
            key = str(report_id)
            if key in seen_reports:
                dropped_reports += 1
                continue
            seen_reports.add(key)
        calls = report.get("calls") or report.get("events")
        is_late = bool(report.get("late") or report.get("final"))
        if isinstance(calls, (list, tuple)):
            report_index = len(kept)
            kept_calls = []
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                call = dict(call)
                call_id = call.get("id") or call.get("call_id")
                if call_id is None:
                    kept_calls.append(call)
                    continue
                key = str(call_id)
                position = seen_calls.get(key)
                if position is not None:
                    if is_late:
                        # A correction: replace the earlier figure in place
                        # rather than adding to it.
                        kept[position[0]]["calls"][position[1]] = call
                        replaced_calls += 1
                    else:
                        dropped_calls += 1
                    continue
                seen_calls[key] = (report_index, len(kept_calls))
                kept_calls.append(call)
            report["calls"] = kept_calls
        kept.append(report)
    summary = {
        "dropped_reports": dropped_reports,
        "dropped_calls": dropped_calls,
        "replaced_calls": replaced_calls,
        "note": ("repeated host usage events are collapsed so the same "
                 "tokens are counted once; a late report for a known call "
                 "replaces its figure instead of adding to it"),
    }
    if not (dropped_reports or dropped_calls or replaced_calls):
        summary = {}
    return kept, summary


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
    # A host hook may report the SAME call more than once (a retried hook, a
    # re-read log line). Collapse repeats by their own ids before summing, so
    # the same tokens are never counted twice.
    reports, dedupe = dedupe_host_reports([report])
    normalized = normalize_host_usage(
        reports[0] if reports else report, source=source)
    vector, breakdown = usage_cost_vector(normalized, latency_s)
    if breakdown is not None:
        breakdown["host"] = normalized.get("source")
        breakdown["model"] = normalized.get("model")
        breakdown["scope"] = "attempt"
        breakdown["calls"] = normalized.get("calls")
        if dedupe:
            breakdown["dedup"] = dedupe
    return vector, breakdown


# ---------------------------------------------------------------------------
# host usage adapters: locate + normalize a host's attempt-level report
# ---------------------------------------------------------------------------

#: The schema tag of the attempt-level HostUsageReport a host produces
#: (see :class:`HostUsageAdapter`). A report carrying this tag is read
#: through :func:`host_report_dimensions`, which honours its ``measured``
#: whitelist and per-dimension ``provenance`` map.
HOST_USAGE_SCHEMA = "or-host-usage/1"

#: The dimensions a HostUsageReport can carry, with the provenance each is
#: recorded under when the report supplies no explicit map.
_HOST_REPORT_DEFAULT_SOURCES = {
    "llm_tokens": "provider_usage",
    "tool_calls": "agent_observed",
}


def is_host_usage_report(report: Any) -> bool:
    """Whether a payload is a versioned HostUsageReport.

    Only the SCHEMA TAG decides: a report that says ``or-host-usage/1`` is
    read under that contract (its ``measured`` whitelist governs what may
    enter ``cost_measured``); anything else is the older permissive shape
    and keeps its existing behaviour.
    """
    return isinstance(report, Mapping) \
        and str((report or {}).get("schema") or "") == HOST_USAGE_SCHEMA


def host_report_dimensions(report: Mapping[str, Any]
                           ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The MEASURED dimensions of a versioned HostUsageReport.

    Returns ``(dimensions, info)``. ``dimensions`` holds only what the
    report's ``measured`` whitelist allows — a number outside the whitelist
    is a sub-fact, never a cost measurement — and ``llm_tokens`` is the
    full口径 total (prompt + completion; reasoning/cached are sub-facts
    inside those sides and are never added again). ``info`` carries the
    per-dimension provenance map, the token basis and the sub-facts, so the
    caller can store the口径 beside the number.

    ``tool_calls`` is validated against the report's own
    ``tool_calls_lower_bound`` when the report carries one: a total below a
    bound the host itself proved is a contradiction, and it is refused
    (``ValueError``) rather than stored.
    """
    report = report or {}
    tokens = report.get("tokens") if isinstance(report.get("tokens"),
                                                Mapping) else report
    prompt = _as_count(tokens.get("prompt_tokens"))
    completion = _as_count(tokens.get("completion_tokens"))
    reasoning = _as_count(tokens.get("reasoning_tokens"))
    cached = _as_count(tokens.get("cached_tokens"))
    calls = _as_count(tokens.get("calls"))
    total: Optional[float]
    if prompt is not None and completion is not None:
        total, basis = prompt + completion, TOKEN_BASIS_TOTAL
    elif completion is not None:
        total, basis = completion, TOKEN_BASIS_COMPLETION_ONLY
    elif prompt is not None:
        total, basis = prompt, TOKEN_BASIS_PROMPT_ONLY
    else:
        total, basis = None, None
    tool_calls = _as_count(report.get("tool_calls"))
    lower_bound = _as_count(report.get("tool_calls_lower_bound"))
    if tool_calls is not None and lower_bound is not None \
            and tool_calls < lower_bound:
        raise ValueError(
            f"tool_calls={tool_calls} is below the report's own "
            f"tool_calls_lower_bound={lower_bound}: the host proved at "
            "least that many invocations happened. tool_calls counts ALL "
            "tool invocations in the attempt's scope (shell commands, "
            "file reads/writes, sandbox runs, solver calls)")
    measured = {str(m) for m in (report.get("measured") or [])}
    # The whitelist names the report's OWN fields (prompt_tokens,
    # completion_tokens, tool_calls); the cost dimensions they feed are
    # llm_tokens and tool_calls. Either side of the token report being
    # measured means the FULL口径 total is measured (it is their sum).
    if measured & {"prompt_tokens", "completion_tokens", "total_tokens"}:
        measured.add("llm_tokens")
    provenance = report.get("provenance") \
        if isinstance(report.get("provenance"), Mapping) else {}
    dimensions: Dict[str, float] = {}
    if total is not None and "llm_tokens" in measured:
        dimensions["llm_tokens"] = float(total)
    if tool_calls is not None and "tool_calls" in measured:
        dimensions["tool_calls"] = float(tool_calls)
    info: Dict[str, Any] = {
        "schema": HOST_USAGE_SCHEMA,
        "host": str(report.get("host") or "generic"),
        "model": (str(report["model"]) if report.get("model") else None),
        "scope": "attempt",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "model_calls": calls,
        "tool_calls_lower_bound": lower_bound,
        "basis": basis,
        "provenance": {
            dim: str(provenance.get(dim)
                     or _HOST_REPORT_DEFAULT_SOURCES.get(dim, "agent_estimate"))
            for dim in dimensions
        },
        "note": ("llm_tokens is the FULL口径 total (prompt + completion); "
                 "reasoning/cached tokens are sub-facts already inside those "
                 "sides and are never added again. Only dimensions listed in "
                 "the report's 'measured' whitelist enter cost_measured; "
                 "everything else stays unknown, never zero"),
    }
    return dimensions, info


class HostUsageAdapter:
    """Locate and normalize ONE host's attempt-level usage report.

    A host (OpenClaw, Hermes, ...) runs the outer agent loop: it owns the
    LLM connection and the tool invocations, so the REAL ``llm_tokens`` and
    ``tool_calls`` of an attempt are facts only it can produce. This
    abstract interface is the pluggable bridge — the core never depends on
    a concrete host at import time, and an unavailable host degrades
    gracefully to "unknown" instead of failing.

    Implementations do TWO things and nothing else:

    * ``locate(scope)`` — find the report file for one attempt (or None);
    * ``load(scope)`` — read it (or None when missing/unreadable);
    * ``normalize(report)`` — turn the host's raw shape into
      ``(dimensions, info)`` in the canonical form
      (:func:`host_report_dimensions`).

    They must NOT re-implement accounting: the口径 rules (full total,
    sub-facts, measured whitelist, lower-bound check) live in ONE place.
    """

    #: The host name this adapter answers to (lower-case).
    name = ""

    #: Default file names this host writes its attempt report under, tried
    #: in order inside the scope's directory. Subclasses may override.
    report_filenames: Tuple[str, ...] = ()

    def locate(self, scope: Mapping[str, Any]) -> Optional[str]:
        """The path of the report for one attempt, or None.

        ``scope`` carries at least ``home`` (the harness memory directory)
        and usually ``execution_id`` / ``task_id`` / ``episode_id``. The
        default implementation looks for the adapter's
        ``report_filenames`` under ``<home>/host_usage/`` and accepts an
        explicit ``usage_file`` override in the scope.
        """
        scope = scope or {}
        explicit = scope.get("usage_file")
        if explicit:
            path = Path(str(explicit))
            return str(path) if path.is_file() else None
        home = scope.get("home")
        if not home:
            return None
        base = Path(str(home)) / "host_usage"
        for name in self.report_filenames:
            for key in ("execution_id", "attempt_id"):
                identifier = scope.get(key)
                if identifier:
                    candidate = base / f"{name}.{identifier}.json"
                    if candidate.is_file():
                        return str(candidate)
            candidate = base / f"{name}.json"
            if candidate.is_file():
                return str(candidate)
        return None

    def load(self, scope: Mapping[str, Any]
             ) -> Optional[Dict[str, Any]]:
        """The parsed report for one attempt, or None when absent/invalid.

        Never raises for a missing or malformed file: an unreadable report
        is the honest "unknown", not an error the caller must handle.
        """
        path = self.locate(scope)
        if path is None:
            return None
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def normalize(self, report: Mapping[str, Any]
                  ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """The host's raw report -> (measured dimensions, info).

        The default reads the versioned HostUsageReport shape
        (:func:`host_report_dimensions`); a host whose raw shape differs
        overrides this to MAP its fields onto that shape first.
        """
        return host_report_dimensions(report)

    def collect(self, scope: Mapping[str, Any]
                ) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        """Locate + load + normalize in one step, or None.

        The convenience entry point the CLI uses: one call returns the
        measured dimensions and their info, or None when the host produced
        no readable report (the dimensions then stay UNKNOWN — never zero).
        """
        report = self.load(scope)
        if report is None:
            return None
        try:
            return self.normalize(report)
        except ValueError:
            # A contradictory report (e.g. tool_calls below its own lower
            # bound) is refused by normalize; the caller decides whether to
            # surface it. Collect treats it as "no usable report".
            return None


class OpenClawUsageAdapter(HostUsageAdapter):
    """The OpenClaw host: locate + normalize its attempt usage report.

    OpenClaw writes one ``or-host-usage/1`` JSON per attempt. This adapter
    only FINDS and READS that file — it contains no OpenClaw code, imports
    nothing from the host, and degrades to "unknown" when the file is not
    there. The report itself is produced by the host (see the docs for the
    3-line contract a host implements).
    """

    name = "openclaw"
    report_filenames = ("openclaw-usage", "host-usage")


class GenericHostUsageAdapter(HostUsageAdapter):
    """The fallback: a report in the canonical shape, any host name."""

    name = "generic"
    report_filenames = ("host-usage",)


#: The registry the CLI resolves ``--usage-host <host>`` against. A host
#: not listed here still works through ``--usage-file`` (the file IS the
#: report); the registry only automates LOCATING it.
HOST_USAGE_ADAPTERS: Dict[str, HostUsageAdapter] = {
    adapter.name: adapter
    for adapter in (OpenClawUsageAdapter(), GenericHostUsageAdapter())
}


def host_usage_adapter(host: Optional[str]) -> Optional[HostUsageAdapter]:
    """The adapter for a host name, or None when the host is unknown.

    Unknown hosts are NOT an error at resolution time: the caller reports
    "no adapter for host X" and the dimensions stay unknown, exactly as a
    missing report would.
    """
    if not host:
        return None
    return HOST_USAGE_ADAPTERS.get(str(host).strip().lower())


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


