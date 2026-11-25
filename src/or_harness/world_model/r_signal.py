"""Versioned implementation-window R and separate formal diagnostics.

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
    R_FORMAL_OBSERVATION_VERSION, R_FORMAL_SCORING_VERSION, R_FORMAL_CONVENTION,
    is_risk_event_probability,
)

_EXCEPTION = re.compile(r"^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Exception)):", re.MULTILINE)
_EXTERNAL = {"ModuleNotFoundError", "ImportError", "PermissionError", "SandboxDependencyError", "ConnectionError",
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
        "observation_rule_version") in (R_OBSERVATION_VERSION, R_FORMAL_OBSERVATION_VERSION)


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
        if features.get("executed") is False:
            return 0, ("the formal attempt was not interrupted by the harness timeout; "
                       "the script never ran, so this is no evidence that the algorithm avoided timeout")
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
    receipt = features.get("technical_outcome") or {}
    if receipt.get("source") == "SafePythonExecutor":
        if receipt.get("failure_cause") == "environment_unavailable":
            return 1, "the executor recorded an unavailable dependency, license, process or isolation environment"
        if receipt.get("failure_cause") == "input_unavailable":
            return None, "the declared input is missing, changed or invalid; responsibility is unconfirmed"
        if receipt.get("failure_cause") == "contract_violation":
            return 0, "the recorded cause is a script-contract violation, not an unavailable environment"
    if "environment" in classes:
        return 1, ("the executor recorded an ENVIRONMENT failure (missing "
                   "dependency / unavailable backend / isolation unavailable)")
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
    fact = {**R_FORMAL_CONVENTION, "label": None, "source": "SafePythonExecutor",
            "executed": bool(outcome.executed), "exit_code": outcome.exit_code,
            "process_started": bool(outcome.process_started),
            "failure_stage": outcome.failure_stage,
            "failure_cause": outcome.failure_cause,
            "submission_rejected": outcome.failure_stage == "submission",
            "runtime_eligible": bool(outcome.executed),
            "result_received": bool(outcome.result_received),
            "protocol_problems": list(checks.get("problems") or []),
            "evidence_refs": ["execution_features.technical_outcome"]}
    text = outcome.stderr or outcome.normalized_error or outcome.message
    exception = _exception_type(text)
    if exception:
        fact["exception_type"] = exception
    if outcome.failure_cause == "contract_violation":
        fact["label"] = 1
        reason = "submitted code violated the execution contract: " + (outcome.protocol_error or outcome.normalized_error)
    elif outcome.protocol_error and outcome.protocol_error.startswith("code construction:"):
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
        fact["failure_cause"] = "environment_unavailable"
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
    if saved.get("observation_rule_version") == R_FORMAL_OBSERVATION_VERSION and saved.get("source") == "SafePythonExecutor":
        fact = copy.deepcopy(saved)
    else:
        fact = {**R_FORMAL_CONVENTION, "label": None, "source": "legacy execution evidence",
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
    fact["runtime_eligible"] = features.get("executed")
    fact["runtime_eligibility_note"] = (
        "script startup is established; this alone does not prove a solver ran"
        if features.get("executed") is True else
        "no script runtime is established; the formal timeout label is not algorithm runtime evidence")
    fact["r_events"] = {
        "implementation_failure": {
            "label": fact["label"], "basis": fact["reason"]},
        "timeout": {"label": timeout_label, "basis": timeout_basis,
                    "runtime_eligible": features.get("executed")},
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
        fact = {**R_FORMAL_CONVENTION, "label": None, "execution_id": execution_id,
                "reason": "no attributable bound execution", "evidence_refs": [],
                "r_events": {name: {"label": None,
                                    "basis": "no attributable bound execution"}
                             for name in R_EVENTS}}
    else:
        fact = technical_attempt_fact(record)
    convention = (prediction.trace.model_info or {}).get("risk_convention") or {}
    if convention.get("observation_rule_version") == R_OBSERVATION_VERSION:
        from or_harness.world_model.implementation_facts import implementation_window, episode_actions
        actions = episode_actions(harness, prediction.candidate.task_id, prediction.candidate.episode_id)
        window = implementation_window(prediction, actions, [record] if record else [])
        fact["formal_technical_outcome"] = {k: copy.deepcopy(fact.get(k)) for k in
                                             ("label", "reason", "observation_rule_version", "scoring_rule_version", "unit", "scope")}
        fact.update(copy.deepcopy(R_CONVENTION))
        fact.update(window)
        fact["r_events"]["implementation_failure"] = {"label": window["label"],
            "basis": window["reason"], "scope": window["scope"]}
        for name in ("timeout", "environment_failure"):
            fact["r_events"][name]["scope"] = "bound_formal_execution"
    return fact


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


def _event_performance(name, scores, facts):
    """Mean predicted probability by observed outcome + discrimination.

    Computed from the SAME scored samples as the per-event Brier: a
    probability is compared with the label of the ONE bound attempt. A
    discrimination value needs positive AND negative outcomes, so it is
    absent (never zero) when either side is empty.
    """
    positive, negative = [], []
    for sample in scores:
        fact = facts.get(sample["identity"]) or {}
        label = (((fact.get("r_events") or {}).get(name)) or {}).get("label")
        if label == 1:
            positive.append(float(sample["probability"]))
        elif label == 0:
            negative.append(float(sample["probability"]))
    mean_positive = (round(sum(positive) / len(positive), 6)
                     if positive else None)
    mean_negative = (round(sum(negative) / len(negative), 6)
                     if negative else None)
    discrimination = (round(mean_positive - mean_negative, 6)
                      if mean_positive is not None
                      and mean_negative is not None else None)
    return {
        "n_positive_scored": len(positive),
        "n_negative_scored": len(negative),
        "mean_probability_observed": mean_positive,
        "mean_probability_not_observed": mean_negative,
        "discrimination": discrimination,
    }


def _formal_technical_r_statistics(facts_by_episode, samples, *, min_samples=5):
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
                **_event_performance(name, name_scores, facts),
            }
        output.append({"observation_rule_version": R_FORMAL_OBSERVATION_VERSION,
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
            **_event_performance(name, name_scores, facts),
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
    return {"observation_rule_version": R_FORMAL_OBSERVATION_VERSION, "groups": output,
        "runtime_coverage": {"script_started": sum(f.get("runtime_eligible") is True for f in facts.values()),
            "script_not_started": sum(f.get("runtime_eligible") is False for f in facts.values()),
            "unknown": sum(f.get("runtime_eligible") is None for f in facts.values()),
            "note": "formal timeout labels and Brier keep their definition; not-started attempts are not algorithm runtime evidence"},
        "coverage": {"n_actual_attempts": len(facts), "n_known_labels": sum(f["label"] in (0, 1) for f in facts.values()),
                     "n_reported_probabilities": len(predicted), "n_scorable": len(scored)},
        "coverage_by_event": coverage_by_event,
        "events": window_events,
        "aggregation": {
            "scoring_rule_version": R_FORMAL_SCORING_VERSION,
            "aggregation": "max_common_event",
            "note": ("R is the maximum over the common valid event "
                     "probabilities; per-event Brier only — the max is not a "
                     "joint event and no joint/max Brier exists")},
        "note": ("attempts/labels deduplicate execution IDs; Brier uses "
                 "original probabilities and same-rule known labels; each of "
                 "the three R events is scored against the ONE bound attempt, "
                 "never against a later retry")}


def technical_r_statistics(facts_by_episode, samples, *, min_samples=5, harness=None):
    """Current window labels and historical formal diagnostics never pool."""
    current, legacy = [], []
    for evaluation, prediction in samples:
        convention = (prediction.trace.model_info or {}).get("risk_convention") or {}
        (current if convention.get("observation_rule_version") == R_OBSERVATION_VERSION
         else legacy).append((evaluation, prediction))
    formal = _formal_technical_r_statistics(facts_by_episode, legacy, min_samples=min_samples)
    facts, coverage_gaps = {}, []
    if harness is not None:
        from or_harness.world_model.implementation_facts import choices, implementation_window, episode_actions
        for (task_id, episode_id), records in facts_by_episode.items():
            actions = episode_actions(harness, task_id, episode_id)
            from or_harness.world_model.episode_closeout import _closeout_created_at
            closed_at = _closeout_created_at(harness, task_id, episode_id)
            for choice in choices(actions):
                if closed_at is not None and choice.started_at > closed_at:
                    continue
                prediction = harness.strategy_predictions.get(choice.outcome["prediction_id"])
                if not prediction or (prediction.trace.model_info.get("risk_convention") or {}).get("observation_rule_version") != R_OBSERVATION_VERSION:
                    continue
                action_id = prediction.trace.model_info.get("bound_action_id")
                action = harness.actions.get(action_id) if action_id else None
                if action:
                    fact = bound_r_observation(harness, prediction, action)
                else:
                    fact = implementation_window(prediction, actions, records)
                    fact["r_events"] = {name: {"label": fact["label"] if name == "implementation_failure" else None}
                                        for name in R_EVENTS}
                    coverage_gaps.append({"prediction_id": prediction.prediction_id,
                        "selection_action_id": choice.action_id, "label": fact["label"],
                        "reason": "selected implementation has no formal execution; facts retained, no formal Evaluation fabricated"})
                facts[(str(task_id), str(episode_id or ""), choice.action_id)] = fact
    for evaluation, prediction in current:
        fact = (evaluation.risk or {}).get("implementation_observation") or {}
        key = (str(evaluation.task_id), str(evaluation.episode_id or ""), fact.get("selection_action_id"))
        if key not in facts and fact.get("selection_action_id"):
            facts[key] = fact
    scores = {name: [] for name in R_EVENTS}
    asked = {name: 0 for name in R_EVENTS}
    reported = {name: 0 for name in R_EVENTS}
    for evaluation, prediction in current:
        valid, _, _ = r_event_probabilities(prediction)
        for name in R_EVENTS:
            asked[name] += 1
            reported[name] += name in valid
        observation = (evaluation.risk or {}).get("implementation_observation") or {}
        identity = (str(evaluation.task_id), str(evaluation.episode_id or ""), observation.get("selection_action_id"))
        if identity not in facts or r_scoring_version(prediction) != R_SCORING_VERSION:
            continue
        for name, probability in valid.items():
            detail = ((evaluation.risk or {}).get("r_events_observation") or {}).get("events", {}).get(name) or {}
            label = detail.get("label")
            if label in (0, 1) and detail.get("comparable") and (
                    name != "implementation_failure" or detail.get("observation_rule_version") == R_OBSERVATION_VERSION):
                scores[name].append({"identity": identity, "probability": probability,
                                     "brier": (probability - label) ** 2})
    counts = _event_label_counts(list(facts.values()))
    n_formal = sum(bool(f.get("execution_id")) for f in facts.values())
    events, coverage = {}, {}
    for name in R_EVENTS:
        rows = scores[name]
        episodes = {s["identity"][:2] for s in rows}
        events[name] = {**counts[name], "n_asked": asked[name],
            "n_reported_probabilities": reported[name], "n_scorable": len(rows),
            "n_scored_distinct_episodes": len(episodes),
            "mean_brier": sum(s["brier"] for s in rows) / len(rows) if rows else None,
            "brier_support": "measured" if len(episodes) >= min_samples else "insufficient_evidence",
            **_event_performance(name, rows, facts)}
        coverage[name] = {"n_asked": asked[name], "n_actual_attempts": n_formal, "n_reported_probabilities": reported[name],
                          "n_scorable": len(rows), "n_known_labels": counts[name]["n_known_labels"]}
    groups = {}
    for identity, fact in facts.items():
        conditions = {k: fact.get(k) for k in ("solver_interface", "structure")}
        key = json.dumps(conditions, sort_keys=True)
        groups.setdefault(key, {"conditions": conditions, "identities": []})["identities"].append(identity)
    group_rows = []
    for group in groups.values():
        identities = set(group.pop("identities"))
        rows = [s for s in scores["implementation_failure"] if s["identity"] in identities]
        group_rows.append({**group, "observation_rule_version": R_OBSERVATION_VERSION,
            "implementation_clear_windows": sum(facts[i]["label"] == 0 for i in identities),
            "implementation_defect_windows": sum(facts[i]["label"] == 1 for i in identities),
            "unknown_windows": sum(facts[i]["label"] is None for i in identities),
            "n_distinct_episodes": len({i[:2] for i in identities}),
            "n_scored_distinct_episodes": len({s["identity"][:2] for s in rows}),
            "mean_brier": sum(s["brier"] for s in rows) / len(rows) if rows else None})
    return {"observation_rule_version": R_OBSERVATION_VERSION,
        "scoring_rule_version": R_SCORING_VERSION, "events": events, "groups": group_rows,
        "coverage": {"n_selected_windows": len(facts), "n_actual_attempts": n_formal,
                     "n_known_labels": counts["implementation_failure"]["n_known_labels"],
                     "n_reported_probabilities": reported["implementation_failure"],
                     "n_scorable": len(scores["implementation_failure"])},
        "coverage_by_event": coverage, "selected_without_execution": coverage_gaps,
        "historical_formal": formal,
        "runtime_coverage": copy.deepcopy(formal.get("runtime_coverage") or {}),
        "aggregation": {"scoring_rule_version": R_SCORING_VERSION, "aggregation": "max_common_event",
                        "note": "maximum single-event score, not a joint event"},
        "note": "implementation uses selected windows; timeout/environment use formal attempts; historical formal statistics are separate"}


def _empty_event_base_rate():
    return {"occurred": 0, "not_occurred": 0, "unknown": 0,
            "n_known_labels": 0, "occurrence_rate": None}


def global_event_base_rates(technical_r):
    """The window's broad technical-event background, per event.

    Reuses ``technical_r_statistics`` - it does NOT rescan the bank. The
    denominator is KNOWN labels only (occurred + not_occurred): an unknown
    label is not a non-occurrence, and an unexecuted candidate or a
    prediction-stage request timeout never becomes an attempt in this
    population. Each R event stays its own random variable; this block is
    background, never a candidate-specific conditional probability and never
    a joint/max probability.
    """
    technical_r = technical_r if isinstance(technical_r, dict) else {}
    if technical_r.get("observation_rule_version") != R_OBSERVATION_VERSION:
        return {}
    source_events = technical_r.get("events")
    if not isinstance(source_events, dict) or not source_events:
        # Older published summaries may predate the window-level ``events``
        # block. The per-condition groups still carry the SAME attempt labels
        # and partition the facts, so their counts can be summed without
        # double counting; a truly legacy record without either stays
        # unavailable rather than being guessed.
        aggregated = {name: _empty_event_base_rate() for name in R_EVENTS}
        found = False
        for group in technical_r.get("groups") or []:
            events = (group or {}).get("events") or {}
            for name in R_EVENTS:
                entry = events.get(name)
                if not isinstance(entry, dict):
                    continue
                found = True
                for key in ("occurred", "not_occurred", "unknown"):
                    aggregated[name][key] += int(entry.get(key) or 0)
        if not found:
            return {}
        source_events = aggregated
    events = {}
    any_known = False
    for name in R_EVENTS:
        entry = source_events.get(name)
        if not isinstance(entry, dict):
            continue
        occurred = int(entry.get("occurred") or 0)
        not_occurred = int(entry.get("not_occurred") or 0)
        unknown = int(entry.get("unknown") or 0)
        known = occurred + not_occurred
        any_known = any_known or known > 0
        events[name] = {
            "occurred": occurred,
            "not_occurred": not_occurred,
            "unknown": unknown,
            "n_known_labels": known,
            "occurrence_rate": (round(occurred / known, 6) if known else None),
        }
    if not events:
        return {}
    return {
        "version": "wm-r-global-base-rates/1",
        "observation_rule_version": technical_r.get(
            "observation_rule_version") or R_OBSERVATION_VERSION,
        "scope": ("selected implementation windows and their bound formal attempts in the closed calibration window; "
                  "broad background, NOT a candidate-specific condition"),
        "n_attempts": int((technical_r.get("coverage") or {}).get(
            "n_actual_attempts") or 0),
        "n_selected_windows": int((technical_r.get("coverage") or {}).get("n_selected_windows") or 0),
        "events": events,
        "rate_basis": ("occurred / (occurred + not_occurred); unknown labels "
                       "are excluded, never counted as non-occurrence"),
        "source": ("technical_r_statistics labels on actual selected implementation windows; "
                   "timeout/environment still require a formal execution; unselected candidates are absent"),
        "support": ("measured_global_base_rate" if any_known
                    else "insufficient_evidence"),
        "note": ("local technical cases describe the candidate's own "
                 "solver/implementation; this global block is wider "
                 "background. The three events are independent and the "
                 "maximum is not a joint probability."),
    }
