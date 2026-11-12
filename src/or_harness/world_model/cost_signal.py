"""Measured C windows and small, applicable cost experience (no calibrator).

The host's model is the execution host, never the world-model provider.
Missing boundaries/counts stay unknown. This module consumes existing
receipts/reports and the closed-episode publication; it performs no I/O.
"""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict

from or_harness.core.schema import COST_DIMENSIONS

COST_OBSERVATION_VERSION = "wm-c-window/1"
TRUSTED = {"agent_observed", "provider_usage"}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def host_key(info):
    return {"host": info.get("host"), "model": info.get("model")}


def normalize_execution_context(value):
    """Known runner inputs only; no inferred model, phase or credentials."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("execution_context must be an object")
    out = {}
    for key in ("host", "model", "agent_id", "solver_interface", "phase", "source", "source_ref"):
        if value.get(key) is not None:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError("execution_context." + key + " must be nonempty text")
            out[key] = value[key].strip()
    if out.get("phase") not in (None, "initial_modelling", "existing_model", "unknown"):
        raise ValueError("execution_context.phase must be initial_modelling, existing_model or unknown")
    if out:
        out.setdefault("source", "runner_execution_context")
    return out


def cost_conditions(context, candidate=None):
    """Freeze only information visible before the decision."""
    from or_harness.world_model.g_signal import requested_solving_conditions
    solving = context.solving_context or {}
    progress = solving.get("task_progress") or {}
    artifact = progress.get("model_artifact") or {}
    if artifact.get("hypothetical") or artifact.get("epistemic") == "inferred":
        artifact = {}
    value = artifact.get("value")
    if isinstance(value, dict):
        value = {k: copy.deepcopy(v) for k, v in value.items() if k != "host_usage"}
    absent = isinstance(value, dict) and value.get("status") in (
        "missing", "unavailable", "failed", "not_created", "not_started")
    available = (bool(value) if isinstance(value, str) else
        isinstance(value, dict) and (value.get("status") == "available" or
            any(value.get(k) for k in ("artifact", "path", "model_ref", "model_digest")))) and not absent
    artifact_view = ({k: copy.deepcopy(v) for k, v in value.items()
                      if k in ("artifact", "path", "model_ref", "model_digest", "digest", "status")}
                     if isinstance(value, dict) else value)
    runner = solving.get("execution_context") or {}
    phase = "existing_model" if available else ("initial_modelling" if absent else runner.get("phase", "unknown"))
    return {
        "scope": "remaining_to_task_end",
        "phase": phase,
        "phase_basis": ("frozen task_progress.model_artifact" if available or absent else
                        "frozen execution_context.phase" if runner.get("phase") else "unknown"),
        "reusable_artifacts": ([{"value": artifact_view,
            "evidence_ref": artifact.get("evidence_ref")} ] if available else []),
        "execution_host": copy.deepcopy(solving.get("execution_host") or {}),
        "solver_conditions": requested_solving_conditions(candidate, context.execution_constraints),
        "family": (context.joint.profile or {}).get("family"),
        "structure": {k: v for k, v in (context.joint.profile or {}).items()
                      if k in ("problem_class", "scale_features", "semantic_coupling",
                               "resource_coupling", "temporal_coupling", "route_complexity")
                      and v is not None and v != {}},
    }


def latest_host(actions, records, *, cutoff, episode_id):
    """Latest real report in this episode, bounded by the frozen snapshot."""
    reports = []
    by_execution = {r.execution_id: r for r in records}
    for action in actions:
        if str(action.episode_id or "") != str(episode_id or "") or action.source == "hypothetical":
            continue
        ended = number(action.ended_at)
        if ended is None or ended > cutoff:
            continue
        info = (action.outcome or {}).get("host_usage") or {}
        record = by_execution.get(action.linked_execution_id)
        if record is not None:
            info = (record.execution_features or {}).get("host_usage") or info
        if info.get("host") and info.get("model"):
            reports.append((ended, {**host_key(info), "source_ref": action.action_id}))
    return max(reports, key=lambda x: x[0])[1] if reports else {}


def _call_values(info, anchor, end):
    """Per-dimension ledger totals/bounds; absence of model calls is unknown.

    Legacy complete ledgers establish only the call kinds they contain.
    An explicit call_details_complete list can declare zero/partial kinds.
    Internal bounds/conflicts are consumed before publishing work facts.
    """
    calls = info.get("call_details")
    if not isinstance(calls, list):
        return {}, {}
    unique, conflicts = {}, {}
    def dimensions(call):
        return ({"tool_calls"} if call.get("kind") == "tool" else
                {"llm_tokens"} if call.get("kind") in ("model", "llm") else
                {"tool_calls", "llm_tokens"})
    def invalid(call, reason):
        for dim in dimensions(call):
            conflicts[dim] = reason
    for call in calls:
        if not isinstance(call, dict):
            invalid({}, "malformed call detail")
            continue
        identity = call.get("id") or call.get("call_id")
        start, stop = number(call.get("started_at")), number(call.get("ended_at"))
        if not isinstance(identity, str) or not identity or start is None or stop is None or stop < start:
            invalid(call, "call detail lacks a usable ID/time interval")
            continue
        if call.get("kind") not in ("tool", "model", "llm"):
            invalid(call, "unknown call kind")
            continue
        if start < anchor or stop > end:
            # A call crossing the anchor cannot be split honestly.
            if start < anchor < stop or start < end < stop:
                invalid(call, "call crosses a measurement boundary")
            continue
        if identity in unique and unique[identity] != call:
            invalid(call, "conflicting details for one call ID")
            invalid(unique[identity], "conflicting details for one call ID")
            continue
        unique[identity] = call
    tools = [c for c in unique.values() if c.get("kind") == "tool"]
    models = [c for c in unique.values() if c.get("kind") in ("model", "llm")]
    complete = info.get("call_details_complete")
    if isinstance(complete, list) and not all(isinstance(d, str) and
            d in ("tool_calls", "llm_tokens") for d in complete):
        invalid({}, "invalid per-dimension call-detail coverage")
        complete = []
    complete = (set(complete) if isinstance(complete, list) else
                ({"tool_calls"} if tools else set()) | ({"llm_tokens"} if models else set()))
    bounds = {"tool_calls": float(len(tools)), "llm_tokens": sum(
        (number(c.get("prompt_tokens")) or 0) + (number(c.get("completion_tokens")) or 0) for c in models)}
    values = {}
    if "tool_calls" in complete:
        values["tool_calls"] = float(len(tools))
    if "llm_tokens" in complete and all(number(c.get("prompt_tokens")) is not None and
                                       number(c.get("completion_tokens")) is not None for c in models):
        values["llm_tokens"] = sum(float(c["prompt_tokens"]) + float(c["completion_tokens"])
                                   for c in models)
    facts = {"call_ids": sorted(unique), "_bounds": bounds, "_conflicts": conflicts}
    if "llm_tokens" in complete and "llm_tokens" not in conflicts:
        facts["host_rounds"] = len(models)
    explicit_solves = [c for c in tools if c.get("phase") == "solve"]
    if explicit_solves:
        facts["observed_solve_calls"] = len(explicit_solves)
    return values, facts


def _execution_tool_floor(actions, records, anchor, end):
    """Sum disjoint, proven execution floors once, only inside this window."""
    # verify/reference actions can link the same execution later. They do
    # not establish when the executor's proven invocation happened.
    actions_by_id = {a.linked_execution_id: a for a in actions if a.linked_execution_id
                     and a.action_type == "execute_strategy" and a.source != "hypothetical"}
    floors = {}
    for record in records:
        action = actions_by_id.get(record.execution_id)
        if action is None or action.started_at is None or action.ended_at is None:
            continue
        if action.started_at < anchor - 1e-6 or action.ended_at > end + 1e-6:
            continue
        lower = number((record.execution_features or {}).get("tool_calls_lower_bound"))
        if lower is not None:
            floors[record.execution_id] = lower
    return sum(floors.values())


def scoped_host_observations(actions, records, *, anchor, anchor_id, end, cutoff,
                             task_id=None, episode_id=None):
    """Read counters/full spans or a complete timed ledger for this window.

    Reports may be referenced from several records/actions. Full-window
    totals are alternatives, never summands. Conflicting totals or hosts
    are unknown. Attempt totals are not evidence of the remaining window.
    """
    candidates = []
    for record in records:
        info = (record.execution_features or {}).get("host_usage")
        if info:
            candidates.append((info, record.execution_id))
    for action in actions:
        if action.source == "hypothetical" or (cutoff is not None and
                number(action.ended_at) is not None and action.ended_at > cutoff):
            continue
        info = (action.outcome or {}).get("host_usage")
        if info:
            candidates.append((info, action.action_id))
    observed, sources, facts, hosts = defaultdict(list), defaultdict(list), {}, []
    rejected, conflicted = [], set()
    if anchor is None:
        return {}, {}, {}, {}, ["decision anchor unknown"]
    for info, ref in candidates:
        identity = info.get("scope_identity") or {}
        if (identity.get("task_id") not in (None, task_id) or
                str(identity.get("episode_id") or "") not in ("", str(episode_id or ""))):
            rejected.append(ref + ": report belongs to a different task/episode")
            continue
        window = info.get("window") or {}
        start, stop = number(window.get("started_at")), number(window.get("ended_at"))
        same_anchor = (start is not None and abs(start - anchor) <= 1e-6)
        if window.get("anchor_action_id"):
            same_anchor = (window["anchor_action_id"] == anchor_id and
                           (start is None or abs(start - anchor) <= 1e-6))
        target_end = end
        if end is None and stop is not None and window.get("end_event") in (
                "finish_task", "final_failure", "confirmed_abort"):
            target_end = stop
        # A complete full span can establish termination even when no
        # finish_task action was recorded. A mere last-action end cannot.
        full = (bool(window.get("complete")) and target_end is not None and
                target_end >= anchor and (cutoff is None or target_end <= cutoff))
        numeric_window = (full and stop is not None and abs(stop - target_end) <= 1e-6
                          and same_anchor and info.get("scope") == "remaining_to_task_end")
        ledger_covers = (full and start is not None and start <= anchor and
                         stop is not None and stop >= target_end)
        values, call_facts = ({}, {})
        if ledger_covers:
            values, call_facts = _call_values(info, anchor, target_end)
        detail_dims = set(values)
        if not numeric_window and not values and not call_facts.get("_conflicts"):
            rejected.append(ref + ": no complete matching decision-to-termination window")
            continue
        if numeric_window:
            # Numeric reports/counter deltas have the declared full window.
            for dim, value in (info.get("dimensions") or {}).items():
                if dim == "llm_tokens" and info.get("basis") != "provider_total":
                    bounds = call_facts.setdefault("_bounds", {})
                    bounds[dim] = max(bounds.get(dim, 0), value)
                    continue
                if dim in values and not math.isclose(values[dim], value, rel_tol=1e-9, abs_tol=1e-6):
                    call_facts.setdefault("_conflicts", {})[dim] = (
                        f"aggregate disagrees with complete call details ({value:g} vs {values[dim]:g})")
                values[dim] = value
            if "remaining_latency_s" in (info.get("measured") or []):
                values.setdefault("remaining_latency_s", target_end - anchor)
        bounds = call_facts.pop("_bounds", {})
        conflicts = {**(info.get("dimension_conflicts") or {}), **call_facts.pop("_conflicts", {})}
        if numeric_window and "host_rounds" in call_facts and number(info.get("model_calls")) is not None \
                and call_facts["host_rounds"] != info["model_calls"]:
            conflicts["host_rounds"] = "model-call total disagrees with complete call details"
        floor = _execution_tool_floor(actions, records, anchor, target_end) if full else 0
        bounds["tool_calls"] = max(bounds.get("tool_calls", 0), floor,
            (number(info.get("tool_calls_lower_bound")) or 0) if numeric_window else 0)
        usable = {}
        for dim, value in values.items():
            source = (info.get("provenance") or {}).get(dim)
            # Call-only reports still declare the dimensions they measured.
            if dim not in (info.get("measured") or info.get("dimensions") or {}):
                continue
            if source not in TRUSTED or number(value) is None:
                continue
            if dim == "llm_tokens" and dim not in detail_dims and info.get("basis") != "provider_total":
                continue
            if value < bounds.get(dim, 0) - 1e-6:
                conflicts[dim] = f"reported total {value:g} is below a proven in-window lower bound {bounds[dim]:g}"
            if dim in conflicts:
                continue
            usable[dim] = float(value)
        for dim, reason in conflicts.items():
            if (info.get("provenance") or {}).get(dim) in TRUSTED:
                conflicted.add(dim)
                rejected.append(ref + ": " + dim + ": " + reason)
        if "llm_tokens" in conflicts or "host_rounds" in conflicts:
            call_facts.pop("host_rounds", None)
        if usable:
            hosts.append(host_key(info))
            for dim, value in usable.items():
                observed[dim].append(value)
                sources[dim].append(ref)
            if "host_rounds" not in call_facts and info.get("model_calls") is not None \
                    and numeric_window and "host_rounds" not in conflicts and "llm_tokens" not in conflicts:
                call_facts["host_rounds"] = info["model_calls"]
            facts.update(call_facts)
            # Timestamps are part of a trusted full-span host observation.
            observed["remaining_latency_s"].append(target_end - anchor)
            sources["remaining_latency_s"].append(ref)
    host_set = {json.dumps(h, sort_keys=True) for h in hosts}
    if len(host_set) > 1:
        return {}, {}, {}, {}, rejected + ["conflicting execution hosts for the same window"]
    actuals = {}
    for dim, values in observed.items():
        if len(set(values)) == 1 and dim not in conflicted:
            actuals[dim] = values[0]
        else:
            rejected.append(dim + ": conflicting reports for the same window")
    return actuals, dict(sources), hosts[0] if hosts else {}, facts, rejected


def observe_cost_window(summary, prediction, actions, records, *, anchor, anchor_id,
                        end, end_basis, cutoff):
    """Add full-window host facts without charging them to attempt budgets."""
    scope = summary.cost.get("scope")
    summary.cost["observation_rule_version"] = COST_OBSERVATION_VERSION
    from or_harness.world_model.g_signal import actual_solving_conditions
    summary.cost["actual_solving_conditions"] = [actual_solving_conditions(r) for r in records]
    if any(getattr(r, "solver_runtime_provenance", None) == "wall_proxy" for r in records):
        real = summary.cost.get("solver_runtime_s") or {}
        real["complete"] = False
        real["comparable"] = {"value": None, "is_sum": False,
            "basis": "script wall-clock proxy is not measured inner solver runtime"}
    summary.cost["window"] = {"anchor": anchor, "anchor_action_id": anchor_id,
                              "end": end, "end_basis": end_basis}
    if scope != "remaining_to_task_end":
        return
    actual, sources, host, facts, rejected = scoped_host_observations(
        actions, records, anchor=anchor, anchor_id=anchor_id,
        end=(end if end_basis == "finish_task" else None), cutoff=cutoff,
        task_id=prediction.candidate.task_id, episode_id=prediction.candidate.episode_id)
    if not host:
        recorded_hosts = {json.dumps(host_key(info), sort_keys=True)
            for r in records if (info := (r.execution_features or {}).get("host_usage") or {})
            and info.get("host") and info.get("model")}
        if len(recorded_hosts) == 1:
            host = json.loads(next(iter(recorded_hosts)))
    for dim in ("llm_tokens", "tool_calls"):
        legacy = summary.cost.get(dim) or {}
        if dim not in actual and (legacy.get("comparable") or {}).get("value") is not None:
            legacy["observation_scope"] = "execution_attempts"
            legacy["complete"] = False
            legacy["boundary_note"] = "attempt totals are a lower bound, not the full remaining window"
    for dim, value in actual.items():
        summary.cost[dim] = {
            "total": value if dim != "remaining_latency_s" else None,
            "comparable": {"value": value, "basis": "complete host decision-to-termination window", "is_sum": False},
            "n_measured": 1, "n_items": 1, "complete": True,
            "observation_scope": scope, "source_refs": sorted(set(sources.get(dim) or [])),
            "observed_from": {"source": "host_usage", "end_basis": "complete_host_span"},
        }
    facts["execution_attempts"] = len(records)
    facts["solver_reported_attempts"] = sum(
        getattr(r, "solver_runtime_provenance", None) == "reported" for r in records)
    facts["failed_attempts"] = sum(bool(r.failures) for r in records)
    methods = [{"execution_id": r.execution_id, "method": copy.deepcopy(r.method_actual)}
               for r in records if r.method_actual]
    if len({json.dumps(m["method"].get("steps"), sort_keys=True) for m in methods}) > 1:
        facts["method_change_receipts"] = methods[:4]
        facts["n_method_receipts"] = len(methods)
    repairs = [a.action_id for a in actions if a.action_type == "model" and
               (a.params or {}).get("phase") == "repair" and
               a.started_at is not None and anchor is not None and a.started_at >= anchor and
               a.ended_at is not None and (end is None or a.ended_at <= end)]
    if repairs:
        facts["repair_actions"] = repairs
    summary.cost["execution_host"] = host
    summary.cost["work_facts"] = facts
    summary.cost["boundary_exclusions"] = rejected


def cost_observed_fact(summary, prediction):
    values = {}
    for dim in COST_DIMENSIONS:
        real = summary.cost.get(dim) or {}
        value = (real.get("comparable") or {}).get("value")
        if real.get("complete") and not real.get("excluded") and number(value) is not None:
            values[dim] = value
    info = prediction.trace.model_info or {}
    conditions = copy.deepcopy(info.get("cost_conditions") or {})
    conditions["execution_host"] = copy.deepcopy(summary.cost.get("execution_host") or {})
    conditions["scope"] = summary.cost.get("scope")
    conditions["window_coverage"] = ("complete" if
        (summary.cost.get("remaining_latency_s") or {}).get("complete") else "partial")
    conditions["actual_solving_conditions"] = [json.loads(raw) for raw in
        sorted({json.dumps(c["comparison_key"], sort_keys=True)
                for c in summary.cost.get("actual_solving_conditions") or []})]
    lower_bounds = {dim: {"value": (real.get("comparable") or {}).get("value"),
                         "scope": real.get("observation_scope") or conditions["scope"],
                         "source_refs": list(summary.cost_records)}
                    for dim in COST_DIMENSIONS if (real := summary.cost.get(dim) or {})
                    and not real.get("complete") and not real.get("excluded")
                    and (real.get("comparable") or {}).get("value") is not None}
    return {"values": values, "conditions": conditions,
            "boundary_exclusions": list(summary.cost.get("boundary_exclusions") or []),
            "lower_bounds": lower_bounds,
            "dimension_scopes": {d: ("attempt" if d == "latency_s" else
                "recorded_solver_receipts" if d == "solver_runtime_s" else conditions["scope"])
                for d in values},
            "observation_rule_version": COST_OBSERVATION_VERSION,
            "window": copy.deepcopy(summary.cost.get("window") or {}),
            "execution_ids": list(summary.cost_records),
            "work_facts": copy.deepcopy(summary.cost.get("work_facts") or {}),
            "source_refs": sorted({ref for dim in values
                for ref in (summary.cost[dim].get("source_refs") or summary.cost_records)}),
            "method_actual": copy.deepcopy(summary.method_actual)}


def _quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return round(values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo), 6)


def conditional_cost_statistics(samples, min_samples=5):
    """Truth population is independent of whether the model supplied C."""
    groups, cases, seen = {}, [], set()
    coverage = {d: {"n_actual": 0, "n_predicted": 0, "n_error": 0,
                    "n_log_ratio": 0} for d in COST_DIMENSIONS}
    for evaluation, prediction in samples:
        cost = evaluation.cost or {}
        predicted = getattr(prediction, "cost", None)
        predicted_dims = predicted.expected.measured_dims() if predicted and predicted.expected else set()
        for dim in predicted_dims:
            coverage[dim]["n_predicted"] += 1
        for dim, error in (cost.get("per_dim") or {}).items():
            coverage[dim]["n_error"] += 1
            coverage[dim]["n_log_ratio"] += error.get("log_ratio") is not None
        fact = cost.get("observed_fact") or {}
        values = fact.get("values") or {}
        conditions = fact.get("conditions") or {}
        window = fact.get("window") or {}
        identity = json.dumps([evaluation.task_id, evaluation.episode_id,
            conditions.get("scope"), fact.get("execution_ids"),
            window.get("anchor_action_id") or window.get("anchor"), window.get("end")], sort_keys=True)
        if not values or identity in seen:
            continue
        seen.add(identity)
        # Reusable artifact CONTENT is explanatory, not a grouping key.
        grouping = {k: conditions.get(k) for k in (
            "execution_host", "scope", "window_coverage", "phase", "family", "structure", "solver_conditions", "actual_solving_conditions")}
        requested = conditions.get("solver_conditions") or {}
        actual_configs = [c.get("config") or {} for c in conditions.get("actual_solving_conditions") or []]
        grouping["solver_conditions"] = {
            "constraints": requested.get("constraints"),
            "unconfirmed_config": {k: v for k, v in (requested.get("config") or {}).items()
                if not actual_configs or any(k not in a for a in actual_configs)}}
        key = json.dumps(grouping, sort_keys=True)
        group = groups.setdefault(key, {"conditions": grouping, "observations": defaultdict(list),
                                       "dimension_episodes": defaultdict(set),
                                       "episodes": set(), "source_refs": set()})
        group["episodes"].add((evaluation.task_id, evaluation.episode_id))
        group["source_refs"].update(fact.get("source_refs") or [])
        for dim, value in values.items():
            group["observations"][dim].append(value)
            group["dimension_episodes"][dim].add((evaluation.task_id, evaluation.episode_id))
            coverage[dim]["n_actual"] += 1
        cases.append({"evaluation_id": evaluation.evaluation_id,
            "task_id": evaluation.task_id, "episode_id": evaluation.episode_id,
            **copy.deepcopy(fact)})
    output = []
    for key, group in sorted(groups.items()):
        n = len(group["episodes"])
        output.append({"conditions": group["conditions"],
            "observation_rule_version": COST_OBSERVATION_VERSION,
            "n_distinct_episodes": n, "support": "measured" if n >= min_samples else "sparse",
            "support_note": "range of these recorded windows only; not a universal cost prior",
            "dimensions": {dim: {"p50": _quantile(v, .5), "p75": _quantile(v, .75),
                                  "min": min(v), "max": max(v), "n": len(v),
                                  "n_distinct_episodes": len(group["dimension_episodes"][dim])}
                           for dim, v in sorted(group["observations"].items())},
            "source_refs": sorted(group["source_refs"])})
    return {"version": COST_OBSERVATION_VERSION, "coverage": coverage,
            "groups": output, "cases": cases,
            "counting_unit": "deduplicated actual execution window; errors count predictions"}


def select_cost_evidence(summary, context, candidate):
    """Select applicable history BEFORE generic request-size trimming."""
    current = cost_conditions(context, candidate)
    audit = (summary or {}).get("conditional_cost") or {}
    host = current["execution_host"]
    def applicable(conditions):
        other_host = conditions.get("execution_host") or {}
        if not host.get("host") or not host.get("model") or host_key(other_host) != host_key(host):
            return False
        if current.get("phase") == "unknown":
            return False
        if conditions.get("window_coverage") != "complete":
            return False
        if any(conditions.get(key) != current.get(key) for key in ("scope", "phase", "family")):
            return False
        if ((conditions.get("solver_conditions") or {}).get("constraints") !=
                current["solver_conditions"].get("constraints")):
            return False
        requested = current["solver_conditions"]
        for actual in conditions.get("actual_solving_conditions") or []:
            if requested.get("solver") and actual.get("solver") != requested["solver"]:
                return False
            if any(k in (actual.get("config") or {}) and actual["config"][k] != v
                   for k, v in (requested.get("config") or {}).items()):
                return False
        a, b = conditions.get("structure") or {}, current.get("structure") or {}
        return all(a[k] == b[k] for k in a.keys() & b.keys())
    groups = [copy.deepcopy(g) for g in audit.get("groups") or [] if applicable(g.get("conditions") or {})]
    groups.sort(key=lambda g: (len((g["conditions"].get("structure") or {}).keys() & current["structure"].keys()),
                               g.get("n_distinct_episodes") or 0), reverse=True)
    for group in groups:
        group["source_refs"] = group["source_refs"][:6]
        other = group["conditions"].get("structure") or {}
        group["unknown_structure_dimensions"] = sorted(other.keys() ^ current["structure"].keys())
        group["support_note"] += "; missing structure/undeclared solver settings remain unknown"
    method = getattr(candidate, "method", None) or {}
    words = set(" ".join(map(str, method.get("steps") or [])).lower().split())
    cases = [copy.deepcopy(c) for c in audit.get("cases") or [] if applicable(c.get("conditions") or {})]
    def rank(case):
        steps = ((case.get("method_actual") or {}).get("steps") or [])
        return (case.get("task_id") == context.task_id,
                len(words & set(" ".join(map(str, steps)).lower().split())))
    cases.sort(key=rank, reverse=True)
    from or_harness.world_model.episode_closeout import _compact_method_steps
    cases = cases[:3]
    for case in cases:
        case["source_refs"] = (case.get("source_refs") or [])[:6]
        methods = [case.get("method_actual") or {}] + [m.get("method") or {} for m in
            (case.get("work_facts") or {}).get("method_change_receipts") or []]
        for observed in methods:
            if isinstance(observed.get("steps"), list):
                observed["steps"] = _compact_method_steps(observed["steps"])
    return {"observation_rule_version": COST_OBSERVATION_VERSION,
        "current_conditions": current,
        "candidate_work": {"steps_ref": "candidate.method.steps",
                           "reusable_artifacts_ref": "cost_evidence.current_conditions.reusable_artifacts"},
        "reference_ranges": groups[:4], "cases": cases,
        "missing": ([] if host.get("model") else ["execution-host model identity unavailable"]) +
                   (["decision phase unavailable"] if current.get("phase") == "unknown" else []),
        "applicability": "same execution host/model, remaining window, phase, family and solving limits; absent structure is unknown",
        "note": "Totals already include the work in each reference case. Compare added/saved candidate steps; count each step once. Sparse ranges describe only the cited executions."}
