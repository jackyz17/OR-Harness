"""One-attempt technical R: sourced labels and small applicable facts.

Business acceptance, ordinary deadlines and external blockage are separate
diagnostics. Old predictions retain their original event-label convention.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter

from or_harness.world_model.contracts import (
    R_CONVENTION, R_OBSERVATION_VERSION, is_implementation_probability,
)

_EXCEPTION = re.compile(r"^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Exception)):", re.MULTILINE)
_EXTERNAL = {"ModuleNotFoundError", "ImportError", "PermissionError", "ConnectionError",
             "ConnectionRefusedError", "ConnectionResetError", "TimeoutError", "OSError"}


def implementation_probability(prediction):
    events = [e for e in getattr(getattr(prediction, "risk", None), "events", [])
              if e.event == "implementation_failure"]
    if len(events) != 1 or not is_implementation_probability(events[0].probability):
        return None
    return float(events[0].probability)


def uses_technical_r(prediction):
    return (prediction.trace.model_info or {}).get("risk_convention", {}).get(
        "observation_rule_version") == R_OBSERVATION_VERSION


def _exception_type(text):
    matches = _EXCEPTION.findall(str(text or ""))
    return matches[-1] if matches else None


def execution_receipt(outcome, checks):
    """Evidence the executor can establish before any business checker."""
    fact = {**R_CONVENTION, "label": None, "source": "SafePythonExecutor",
            "executed": bool(outcome.executed), "exit_code": outcome.exit_code,
            "result_received": bool(outcome.result_received),
            "protocol_problems": list(checks.get("problems") or []),
            "evidence_refs": ["execution_features.technical_outcome"]}
    text = outcome.stderr or outcome.normalized_error or outcome.message
    exception = _exception_type(text)
    if exception:
        fact["exception_type"] = exception
    if outcome.protocol_error and outcome.protocol_error.startswith("code construction:"):
        fact["label"] = 1
        reason = outcome.protocol_error
    elif not outcome.executed:
        reason = "execution blocked before the script ran"
    elif outcome.status == "timeout":
        reason = "ordinary time limit; no verified code-caused timeout"
    elif outcome.result_received and outcome.exit_code == 0 and outcome.status not in ("error", "timeout") \
            and not outcome.protocol_error and not checks.get("problems"):
        # A caught/logged exception is not a failed attempt when the same
        # process subsequently delivers a valid result normally.
        fact["label"] = 0
        reason = "process completed and delivered a readable protocol-valid result"
    elif exception in _EXTERNAL or (exception not in ("ValueError", "TypeError", "SyntaxError", "NameError", "KeyError", "AttributeError")
            and any(m in str(text).lower() for m in ("license", "not installed", "connection refused", "no module named"))):
        reason = "external environment/interface blockage: " + str(exception or "receipt")
    elif exception:
        fact["label"] = 1
        reason = "captured code/API exception: " + exception
    elif outcome.protocol_error or checks.get("problems"):
        fact["label"] = 1
        reason = "result delivery protocol error: " + (outcome.protocol_error or "; ".join(checks["problems"]))
    elif outcome.exit_code is not None and outcome.exit_code != 0:
        from or_harness.strategy.triggers import _ENVIRONMENT_MARKERS
        if outcome.exit_code in (-9, -15, -24) or any(m in str(text).lower() for m in _ENVIRONMENT_MARKERS):
            reason = "external/resource termination; technical cause unconfirmed"
        else:
            fact["label"] = 1
            reason = "abnormal solver/script process exit"
    else:
        reason = "no directly verifiable technical outcome"
    fact["reason"] = reason
    return fact


def technical_attempt_fact(record):
    """Derive facts without rewriting old receipts or old probability errors."""
    features = record.execution_features or {}
    saved = features.get("technical_outcome") or {}
    if saved.get("observation_rule_version") == R_OBSERVATION_VERSION and saved.get("source") == "SafePythonExecutor":
        fact = copy.deepcopy(saved)
    else:
        fact = {**R_CONVENTION, "label": None, "source": "legacy execution evidence",
                "reason": "insufficient execution/protocol evidence", "evidence_refs": []}
        status = (record.quality or {}).get("status")
        artifacts = features.get("artifacts") or {}
        captured_error = features.get("executed") is True and bool(artifacts.get("stderr_clip"))
        errors = "\n".join(str(f.error) for f in record.failures or [])
        exception = _exception_type(errors)
        if features.get("executed") is False or status == "timeout":
            fact["reason"] = "blocked or timed-out attempt; technical result unknown"
        elif exception and exception not in _EXTERNAL and captured_error:
            fact.update(label=1, reason="recorded exception type: " + exception,
                        evidence_refs=["failures.error"])
        elif any(str(p).startswith(("feasible status without objective_value", "objective_value is not finite", "illegal status"))
                 for p in (record.quality or {}).get("problems") or []):
            fact.update(label=1, reason="recorded result protocol problems", evidence_refs=["quality.problems"])
        elif status in ("optimal", "feasible", "infeasible", "unbounded", "unknown") and \
                (features.get("artifacts") or {}).get("workspace_result") and not record.failures:
            fact.update(label=0, reason="captured readable result and successful protocol checks",
                        evidence_refs=["artifacts.workspace_result", "quality"])
    # A timeout requires an explicit sourced investigation, with BOTH code
    # and captured trace/result references. Step similarity is not evidence.
    audit = features.get("technical_failure_evidence") or {}
    refs = audit.get("evidence_refs") or []
    artifacts = features.get("artifacts") or {}
    resolved = {"artifacts." + k for k, v in artifacts.items() if isinstance(v, dict) and v.get("sha256")}
    if (record.quality or {}).get("status") == "timeout" and audit.get("kind") == "code_caused_timeout" \
            and isinstance(audit.get("source"), str) and audit["source"].strip() \
            and isinstance(audit.get("assessment"), str) and audit["assessment"].strip() and isinstance(refs, list) and refs \
            and all(isinstance(ref, str) and ref in resolved for ref in refs) \
            and "artifacts.submitted_script" in refs \
            and set(refs).intersection(("artifacts.stderr_clip", "artifacts.workspace_result")):
        fact.update(label=1, source=audit["source"], reason=audit["assessment"], evidence_refs=list(refs))
    host = features.get("host_usage") or {}
    runner = features.get("execution_host") or {}
    if not host.get("host") or not host.get("model"):
        host = runner
    elif all(host.get(k) == runner.get(k) for k in ("host", "model")) and runner.get("agent_id"):
        host = {**host, "agent_id": runner["agent_id"]}
    fact.update(execution_id=record.execution_id, task_id=record.task_id,
        execution_agent={k: host[k] for k in ("host", "model", "agent_id") if host.get(k)},
        solver_interface=(record.solver or {}).get("name"),
        interface_source="ExecutionRecord.solver.name",
        structure={"family": record.profile_snapshot.family,
                   "problem_class": record.profile_snapshot.problem_class})
    script = artifacts.get("submitted_script")
    if fact["label"] == 0 and isinstance(script, dict) and script.get("sha256"):
        fact["reusable_script"] = copy.deepcopy(script)
    check = features.get("task_check") or {}
    if check.get("g_attribution"):
        fact["semantic_attribution"] = copy.deepcopy(check["g_attribution"])
    return fact


def bound_r_observation(harness, prediction, action):
    execution_id = getattr(action, "linked_execution_id", None)
    record = (harness.bank.get_pending(execution_id) or harness.bank.get(execution_id)) if execution_id else None
    if record is None:
        return {**R_CONVENTION, "label": None, "execution_id": execution_id,
                "reason": "no attributable bound execution", "evidence_refs": []}
    return technical_attempt_fact(record)


def technical_r_statistics(facts_by_episode, samples, *, min_samples=5):
    """All actual attempts; labels dedup by execution, scores by prediction."""
    groups, facts = {}, {}
    for (task_id, episode), records in facts_by_episode.items():
        for record in records:
            if record.source != "executed":
                continue
            fact = technical_attempt_fact(record)
            identity = (str(task_id), str(episode), record.execution_id)
            facts[identity] = fact
    predicted, reported, scored = set(), [], []
    for evaluation, prediction in samples:
        if not uses_technical_r(prediction):
            continue
        p = implementation_probability(prediction)
        if p is None:
            continue
        predicted.add(prediction.prediction_id)
        observation = (evaluation.risk or {}).get("implementation_observation") or {}
        identity = (str(evaluation.task_id), str(evaluation.episode_id or ""), observation.get("execution_id"))
        if identity in facts:
            reported.append({"prediction_id": prediction.prediction_id, "identity": identity})
        if identity in facts and facts[identity]["label"] in (0, 1) and observation.get("comparable"):
            scored.append({"prediction_id": prediction.prediction_id, "identity": identity,
                           "probability": p, "brier": (p - facts[identity]["label"]) ** 2})
    for identity, fact in facts.items():
        condition = {k: fact[k] for k in ("execution_agent", "solver_interface", "structure")}
        key = json.dumps(condition, sort_keys=True)
        group = groups.setdefault(key, {"conditions": condition, "facts": [], "identities": set()})
        group["facts"].append(fact)
        group["identities"].add(identity)
    output = []
    for key, group in sorted(groups.items()):
        units = group["facts"]
        counts = Counter(f["label"] for f in units)
        scores = [s for s in scored if s["identity"] in group["identities"]]
        n_known = counts[0] + counts[1]
        labelled_episodes = len({i[:2] for i in group["identities"] if facts[i]["label"] in (0, 1)})
        output.append({"observation_rule_version": R_OBSERVATION_VERSION,
            "conditions": group["conditions"], "n_attempts": len(units),
            "technical_failures": counts[1], "technical_successes": counts[0], "unknown": counts[None],
            "n_known_labels": n_known, "n_scorable": len(scores),
            "n_reported_probabilities": len({s["prediction_id"] for s in reported if s["identity"] in group["identities"]}),
            "n_labelled_distinct_episodes": labelled_episodes,
            "n_distinct_episodes": len({i[:2] for i in group["identities"]}),
            "failure_rate": counts[1] / n_known if n_known else None,
            "mean_brier": sum(s["brier"] for s in scores) / len(scores) if scores else None,
            "n_scored_distinct_episodes": len({s["identity"][:2] for s in scores}),
            "brier_support": "measured" if len({s["identity"][:2] for s in scores}) >= min_samples else "insufficient_evidence",
            "support": "measured" if labelled_episodes >= min_samples else "insufficient_evidence",
            "source_refs": [f["execution_id"] for f in units],
            "reusable_scripts": [{"execution_id": f["execution_id"], "artifact": f["reusable_script"]}
                                 for f in units if f.get("reusable_script")][:2]})
    return {"observation_rule_version": R_OBSERVATION_VERSION, "groups": output,
        "coverage": {"n_actual_attempts": len(facts), "n_known_labels": sum(f["label"] in (0, 1) for f in facts.values()),
                     "n_reported_probabilities": len(predicted), "n_scorable": len(scored)},
        "note": "attempts/labels deduplicate execution IDs; Brier uses original probabilities and same-rule known labels"}


def select_r_evidence(summary, context, candidate):
    current = context.solving_context.get("execution_host") or {}
    runner = context.solving_context.get("execution_context") or {}
    interface = getattr(candidate, "solver", None) or runner.get("solver_interface")
    missing = []
    if not current.get("host") or not current.get("model"):
        missing.append("execution agent host/model unknown")
    if not interface:
        missing.append("solver interface unknown")
    groups = []
    for g in ((summary or {}).get("technical_r") or {}).get("groups") or []:
        other = (g.get("conditions") or {}).get("execution_agent") or {}
        if missing or any(current.get(k) != other.get(k) for k in ("host", "model", "agent_id")) \
                or g.get("observation_rule_version") != R_OBSERVATION_VERSION \
                or (g.get("conditions") or {}).get("solver_interface") != interface:
            continue
        structure = (g.get("conditions") or {}).get("structure") or {}
        wanted_class = (context.joint.profile or {}).get("problem_class")
        if g.get("support") == "measured" and wanted_class and structure.get("problem_class") \
                and wanted_class != structure["problem_class"]:
            continue
        item = copy.deepcopy(g)
        for diagnostic in ("mean_brier", "n_scorable", "n_reported_probabilities", "n_scored_distinct_episodes", "brier_support"):
            item.pop(diagnostic, None)  # prediction diagnostics stay in audit, not the empirical prior
        item["source_refs"] = item["source_refs"][:6]
        item["support_note"] = "closed attempts for this agent/interface; sparse or missing structure supports these cases only"
        groups.append(item)
    groups.sort(key=lambda g: (g["conditions"].get("structure") or {}).get("family") !=
                             (context.joint.profile or {}).get("family"))
    return {**R_CONVENTION, "execution_agent": copy.deepcopy(current), "solver_interface": interface,
            "groups": groups[:3], "missing": missing,
            "support": "applicable_facts" if groups else "insufficient_evidence",
            "note": "use applicable empirical facts; shared base rates or similar candidate probabilities are allowed when supported"}
