"""One-attempt technical R: sourced labels and small applicable facts.

R is the MAXIMUM single-event probability over the R events every ranked
candidate supplied (implementation_failure / timeout / environment_failure).
It is a maximum single-item risk score, NOT a joint failure probability.
Business acceptance, ordinary deadlines, reported infeasibility and external
blockage that is not a technical hazard stay separate diagnostics. Old
predictions retain their original single-event (implementation_failure)
convention and are never retro-fitted.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter

from or_harness.world_model.contracts import (
    R_CONVENTION, R_EVENT_DEFINITIONS, R_EVENTS,
    R_EVENTS_BY_SCORING_VERSION, R_LEGACY_SCORING_VERSION,
    R_OBSERVATION_VERSION, R_SCORING_VERSION,
    is_risk_event_probability,
)

_EXCEPTION = re.compile(r"^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Exception)):", re.MULTILINE)
_EXTERNAL = {"ModuleNotFoundError", "ImportError", "PermissionError", "ConnectionError",
             "ConnectionRefusedError", "ConnectionResetError", "TimeoutError", "OSError"}


def r_scoring_version(prediction):
    """The R scoring convention a prediction was frozen under.

    ``None`` means the stored convention DECLARES an unknown version: it is
    not guessed at (no silent fallback to a different convention).
    """
    convention = (getattr(prediction, "trace", None)
                  and prediction.trace.model_info
                  or {}).get("risk_convention") or {}
    version = convention.get("scoring_rule_version")
    if version in R_EVENTS_BY_SCORING_VERSION:
        return version
    if version is not None:
        return None
    # A prediction archived before the multi-event convention stored only
    # ``observation_rule_version``. It keeps the ORIGINAL single-event rule.
    return R_LEGACY_SCORING_VERSION


def r_scoring_events(prediction):
    """The R events under this prediction's declared convention, or None."""
    return R_EVENTS_BY_SCORING_VERSION.get(r_scoring_version(prediction))


def r_event_probabilities(prediction, *, events=None):
    """Strict per-event probabilities from ONE prediction.

    Returns ``(valid, unusable, missing)``:

    - ``valid``: event -> float for a single event with a finite JSON number
      in [0,1] (an explicit 0 is a real forecast);
    - ``unusable``: event -> reason when the payload carries a DUPLICATE or
      an illegal probability. The event is unknown, never coerced/clamped;
    - ``missing``: events with no probability at all (also unknown, never 0).
    """
    events = tuple(events if events is not None
                   else (r_scoring_events(prediction) or ()))
    raw = list(getattr(getattr(prediction, "risk", None), "events", []) or [])
    found = {name: [] for name in events}
    for event in raw:
        name = str(getattr(event, "event", ""))
        if name in found:
            found[name].append(event)
    valid, unusable, missing = {}, {}, []
    for name in events:
        matches = found[name]
        if len(matches) > 1:
            unusable[name] = ("duplicate " + name + " events are ambiguous: "
                              "the probability is unusable, never summed or "
                              "averaged")
        elif not matches:
            missing.append(name)
        elif not is_risk_event_probability(matches[0].probability):
            unusable[name] = ("probability must be a finite JSON number in "
                              "[0,1]; this event is unusable, never converted")
        else:
            valid[name] = float(matches[0].probability)
    return valid, unusable, missing


def implementation_probability(prediction):
    """The legacy single implementation_failure probability, if valid.

    Kept for backward-compatible diagnostics (``risk_probability`` and old
    evaluations); the multi-event R term is computed by
    :func:`r_event_probabilities`.
    """
    valid, _, _ = r_event_probabilities(
        prediction, events=("implementation_failure",))
    return valid.get("implementation_failure")


def uses_technical_r(prediction):
    return (prediction.trace.model_info or {}).get("risk_convention", {}).get(
        "observation_rule_version") == R_OBSERVATION_VERSION


def _failure_classes(record):
    return [getattr(f, "error_class", None) for f in (record.failures or [])]


def timeout_event_label(record):
    """Did THIS attempt get INTERRUPTED by the wall-clock limit?

    Returns ``(label, basis)`` where label is 1 (interrupted), 0 (not
    interrupted) or None (the record cannot distinguish the two).

    The distinction the prompt states is the one this reads:
    - the executor's wall-clock limit kills the process before a result ->
      OCCURRED;
    - the solver stops at its OWN time limit and returns a readable result
      (``result_received`` / a captured workspace result) -> NOT occurred;
    - a legacy record that says neither stays UNKNOWN, never a guess.
    """
    features = getattr(record, "execution_features", None) or {}
    status = (getattr(record, "quality", None) or {}).get("status")
    if status != "timeout":
        return 0, ("the execution finished without hitting the harness "
                   "wall-clock limit")
    receipt = features.get("technical_outcome") or {}
    if receipt.get("source") == "SafePythonExecutor" \
            and "result_received" in receipt:
        if receipt.get("result_received"):
            return 0, ("the solver stopped at its own time limit and "
                       "delivered a readable result; the harness did not "
                       "interrupt the attempt")
        return 1, ("the harness wall-clock limit interrupted the attempt "
                   "before a usable result")
    artifacts = features.get("artifacts") or {}
    if isinstance(artifacts.get("workspace_result"), dict):
        return 0, ("a result was captured, so the solver returned at its own "
                   "time limit instead of being interrupted by the harness")
    return None, ("the execution status is timeout but the record does not "
                  "say whether the harness interrupted it or the solver "
                  "returned a time-limit result: the two are not inferred "
                  "from prose")


def environment_event_label(record):
    """Did an ENVIRONMENT blockage stop THIS attempt? ``(label, basis)``."""
    features = getattr(record, "execution_features", None) or {}
    status = (getattr(record, "quality", None) or {}).get("status")
    classes = [c for c in _failure_classes(record) if c]
    if "environment" in classes:
        return 1, ("the executor recorded an ENVIRONMENT failure (sandbox "
                   "policy / missing dependency / unavailable backend)")
    if status == "error" and not classes:
        return None, ("the failure carries no recorded error_class: the "
                      "cause is UNKNOWN and is not inferred from the error "
                      "text")
    if classes and all(c == "unknown" for c in classes):
        return None, ("the failure was classified UNKNOWN: the absence of an "
                      "environment cause is not established")
    if features.get("executed") is False and not classes:
        return None, ("the script never ran and no failure class was "
                      "recorded: the environment cause is unknown")
    return 0, "the execution did not record an environment failure class"


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
    fact.update(execution_id=record.execution_id, task_id=record.task_id,
        solver_interface=(record.solver or {}).get("name"),
        interface_source="ExecutionRecord.solver.name",
        structure={"family": record.profile_snapshot.family,
                   "problem_class": record.profile_snapshot.problem_class})
    # ALL THREE R events carry a label on the SAME bound attempt, so the
    # close-out never mixes in a later retry. implementation_failure keeps
    # the technical receipt above; timeout/environment read their own rules.
    timeout_label, timeout_basis = timeout_event_label(record)
    environment_label, environment_basis = environment_event_label(record)
    fact["r_events"] = {
        "implementation_failure": {
            "label": fact["label"], "basis": fact["reason"]},
        "timeout": {"label": timeout_label, "basis": timeout_basis},
        "environment_failure": {
            "label": environment_label, "basis": environment_basis},
    }
    script = artifacts.get("submitted_script")
    if fact["label"] == 0 and isinstance(script, dict) and script.get("sha256"):
        fact["reusable_script"] = copy.deepcopy(script)
    return fact


def bound_r_observation(harness, prediction, action):
    execution_id = getattr(action, "linked_execution_id", None)
    record = (harness.bank.get_pending(execution_id) or harness.bank.get(execution_id)) if execution_id else None
    if record is None:
        return {**R_CONVENTION, "label": None, "execution_id": execution_id,
                "reason": "no attributable bound execution", "evidence_refs": [],
                "r_events": {name: {"label": None,
                                    "basis": "no attributable bound execution"}
                             for name in R_EVENTS}}
    return technical_attempt_fact(record)


def _event_label_counts(facts):
    """Per-event label counts over the facts of one group/window."""
    counts = {}
    for name in R_EVENTS:
        labels = [((fact.get("r_events") or {}).get(name) or {}).get("label")
                  for fact in facts]
        occurred = sum(1 for label in labels if label == 1)
        not_occurred = sum(1 for label in labels if label == 0)
        unknown = sum(1 for label in labels if label is None)
        known = occurred + not_occurred
        counts[name] = {
            "occurred": occurred,
            "not_occurred": not_occurred,
            "unknown": unknown,
            "n_known_labels": known,
            "occurrence_rate": (occurred / known) if known else None,
        }
    return counts


def technical_r_statistics(facts_by_episode, samples, *, min_samples=5):
    """All actual attempts, labeled per event on the ONE bound attempt.

    Two scoring layers are reported and NEVER mixed:

    - the LEGACY single-event layer (``implementation_failure`` against the
      ``wm-r-technical/1`` attempt label) keeps its original coverage keys,
      so archived predictions and their Brier samples read exactly as before;
    - the MULTI-EVENT layer scores EACH R event against the SAME bound
      attempt (``evaluation.risk.r_events_observation``), reports its own
      per-event coverage, and computes NO Brier score for the maximum: the
      max R that U charges is not an event, so there is no joint outcome to
      score (a joint/max "probability" would be a fabricated variable).

    Later retries never enter either layer: the identity is the execution the
    prediction was bound to, not the episode's later attempts.
    """
    groups, facts = {}, {}
    for (task_id, episode), records in facts_by_episode.items():
        for record in records:
            if record.source != "executed":
                continue
            fact = technical_attempt_fact(record)
            identity = (str(task_id), str(episode), record.execution_id)
            facts[identity] = fact
    predicted, reported, scored = set(), [], []
    event_reported = {name: {} for name in R_EVENTS}
    event_scored = {name: [] for name in R_EVENTS}
    # Coverage denominators: every prediction on the multi-event convention
    # was ASKED about all three events; a legacy prediction was asked about
    # implementation_failure only.
    event_asked = {name: 0 for name in R_EVENTS}
    for evaluation, prediction in samples:
        if not uses_technical_r(prediction):
            continue
        declared_events = R_EVENTS_BY_SCORING_VERSION.get(
            r_scoring_version(prediction)) or ()
        for name in declared_events:
            event_asked[name] += 1
        p = implementation_probability(prediction)
        if p is not None:
            predicted.add(prediction.prediction_id)
            observation = (evaluation.risk or {}).get(
                "implementation_observation") or {}
            identity = (str(evaluation.task_id),
                        str(evaluation.episode_id or ""),
                        observation.get("execution_id"))
            event_reported["implementation_failure"][
                prediction.prediction_id] = identity
            if identity in facts:
                reported.append({"prediction_id": prediction.prediction_id,
                                 "identity": identity})
            if identity in facts and facts[identity]["label"] in (0, 1) \
                    and observation.get("comparable"):
                scored.append({"prediction_id": prediction.prediction_id,
                               "identity": identity, "probability": p,
                               "brier": (p - facts[identity]["label"]) ** 2})
        # The multi-event layer reads the SAME bound attempt: the close-out
        # records one label per event for that execution, never a later
        # retry's label.
        observation = (evaluation.risk or {}).get(
            "r_events_observation") or {}
        identity = (str(evaluation.task_id),
                    str(evaluation.episode_id or ""),
                    observation.get("execution_id"))
        for name in ("timeout", "environment_failure"):
            entry = (observation.get("events") or {}).get(name)
            if not isinstance(entry, dict):
                continue
            probability = entry.get("predicted_probability")
            if probability is None:
                continue
            event_reported[name][prediction.prediction_id] = identity
            label = entry.get("label")
            if label in (0, 1) and entry.get("comparable"):
                event_scored[name].append({
                    "prediction_id": prediction.prediction_id,
                    "identity": identity,
                    "probability": float(probability),
                    "brier": (float(probability) - label) ** 2})
    event_scored["implementation_failure"] = scored
    for identity, fact in facts.items():
        condition = {k: fact[k] for k in ("solver_interface", "structure")}
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
        event_counts = _event_label_counts(units)
        group_events = {}
        for name in R_EVENTS:
            name_scores = [s for s in event_scored[name]
                           if s["identity"] in group["identities"]]
            episodes = {s["identity"][:2] for s in name_scores}
            reported_ids = [pid for pid, ident in
                            event_reported[name].items()
                            if ident in group["identities"]]
            group_events[name] = {
                **event_counts[name],
                "n_reported_probabilities": len(reported_ids),
                "n_scorable": len(name_scores),
                "mean_brier": (sum(s["brier"] for s in name_scores)
                               / len(name_scores)) if name_scores else None,
                "n_scored_distinct_episodes": len(episodes),
                "brier_support": ("measured" if len(episodes) >= min_samples
                                  else "insufficient_evidence"),
            }
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
            # The three R events on the SAME attempts. Included in the
            # evidence block so a later prediction sees the timeout and
            # environment facts, not only the implementation labels.
            "events": group_events,
            "source_refs": [f["execution_id"] for f in units],
            "reusable_scripts": [{"execution_id": f["execution_id"], "artifact": f["reusable_script"]}
                                 for f in units if f.get("reusable_script")][:2]})
    window_counts = _event_label_counts(list(facts.values()))
    window_events = {}
    for name in R_EVENTS:
        name_scores = event_scored[name]
        episodes = {s["identity"][:2] for s in name_scores}
        window_events[name] = {
            "definition": R_EVENT_DEFINITIONS[name],
            **window_counts[name],
            "n_asked": event_asked[name],
            "n_reported_probabilities": len(event_reported[name]),
            "n_scorable": len(name_scores),
            "n_scored_distinct_episodes": len(episodes),
            "mean_brier": (sum(s["brier"] for s in name_scores)
                           / len(name_scores)) if name_scores else None,
            "brier_support": ("measured" if len(episodes) >= min_samples
                              else "insufficient_evidence"),
        }
    coverage_by_event = {
        name: {
            "n_asked": event_asked[name],
            "n_actual_attempts": len(facts),
            "n_known_labels": window_counts[name]["n_known_labels"],
            "n_reported_probabilities": len(event_reported[name]),
            "n_scorable": len(event_scored[name]),
        }
        for name in R_EVENTS}
    return {"observation_rule_version": R_OBSERVATION_VERSION, "groups": output,
        "coverage": {"n_actual_attempts": len(facts), "n_known_labels": sum(f["label"] in (0, 1) for f in facts.values()),
                     "n_reported_probabilities": len(predicted), "n_scorable": len(scored)},
        "coverage_by_event": coverage_by_event,
        "events": window_events,
        "aggregation": {
            "scoring_rule_version": R_SCORING_VERSION,
            "aggregation": "max_common_event",
            "note": ("R is the maximum over the common valid event "
                     "probabilities; per-event Brier only — the max is not a "
                     "joint event and no joint/max Brier exists")},
        "note": ("attempts/labels deduplicate execution IDs; Brier uses "
                 "original probabilities and same-rule known labels; each of "
                 "the three R events is scored against the ONE bound attempt, "
                 "never against a later retry")}


def select_r_evidence(summary, context, candidate):
    runner = context.solving_context.get("execution_context") or {}
    interface = getattr(candidate, "solver", None) or runner.get("solver_interface")
    missing = []
    if not interface:
        missing.append("solver interface unknown")
    groups = []
    for g in ((summary or {}).get("technical_r") or {}).get("groups") or []:
        if missing or g.get("observation_rule_version") != R_OBSERVATION_VERSION \
                or (g.get("conditions") or {}).get("solver_interface") != interface:
            continue
        structure = (g.get("conditions") or {}).get("structure") or {}
        wanted_class = (context.joint.profile or {}).get("problem_class")
        if g.get("support") == "measured" and wanted_class and structure.get("problem_class") \
                and wanted_class != structure["problem_class"]:
            continue
        item = copy.deepcopy(g)
        item["conditions"].pop("execution_agent", None)
        for diagnostic in ("mean_brier", "n_scorable", "n_reported_probabilities", "n_scored_distinct_episodes", "brier_support"):
            item.pop(diagnostic, None)  # prediction diagnostics stay in audit, not the empirical prior
        # The SAME trim applies to the per-event block: the model gets the
        # observed counts for each R event (including timeout and
        # environment_failure), never its own past Brier diagnostics.
        for event_entry in (item.get("events") or {}).values():
            if isinstance(event_entry, dict):
                event_entry.pop("definition", None)
                for diagnostic in ("mean_brier", "n_scorable",
                                   "n_reported_probabilities",
                                   "n_scored_distinct_episodes",
                                   "brier_support"):
                    event_entry.pop(diagnostic, None)
        item["source_refs"] = item["source_refs"][:6]
        item["support_note"] = "closed attempts for this solver interface; sparse or missing structure supports these cases only"
        groups.append(item)
    groups.sort(key=lambda g: (g["conditions"].get("structure") or {}).get("family") !=
                             (context.joint.profile or {}).get("family"))
    return {**R_CONVENTION, "solver_interface": interface,
            "event_definitions": dict(R_EVENT_DEFINITIONS),
            "groups": groups[:3], "missing": missing,
            "support": "applicable_facts" if groups else "insufficient_evidence",
            "note": ("use the applicable empirical facts for EACH R event "
                     "(implementation_failure, timeout, environment_failure); "
                     "shared base rates or similar candidate probabilities are "
                     "allowed when supported")}
