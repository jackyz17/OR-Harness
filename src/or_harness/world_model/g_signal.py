"""Attributable G facts. No fitted correction and no model calls.

An audit is a sourced assessment by the caller, not a proof manufactured
from identical prose or from the checker's verdict. Artifact references
are resolved against the execution before the assessment is recorded.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

from or_harness.core.schema import normalize_method
from or_harness.world_model.contracts import (
    FAITHFUL_G_METRIC, FAITHFUL_G_OBSERVATION_VERSION, is_g_probability,
)


def method_key(method):
    method = normalize_method(method) or {}
    return {k: v for k, v in method.items() if k != "source"}


def checked_attribution(record, declaration):
    """Validate optional check-task evidence without changing its verdict."""
    if not isinstance(declaration, dict):
        return {"issues": ["g_attribution must be an object"]}
    out = copy.deepcopy(declaration)
    issues = []
    for key, choices in {
        "method_fidelity": ("faithful", "deviated", "unknown"),
        "failure_source": ("candidate_method", "implementation", "environment", "unknown"),
        "acceptance_scope": ("original_task", "partial", "unknown"),
    }.items():
        if out.get(key) not in choices:
            issues.append(f"{key} must be one of {choices}")
    for key in ("source", "assessment", "acceptance_id"):
        if not isinstance(out.get(key), str) or not out[key].strip():
            issues.append(f"{key} must name the assessment/source/acceptance standard")
    features = record.execution_features or {}
    available = {
        "artifacts." + key: ref for key, ref in (features.get("artifacts") or {}).items()
        if isinstance(ref, dict) and ref.get("sha256")
    }
    if (record.solver or {}).get("code_hash"):
        available["solver.code_hash"] = {"sha256": record.solver["code_hash"]}
    if features.get("solution_variables"):
        available["execution_features.solution_variables"] = {
            "sha256": hashlib.sha256(json.dumps(features["solution_variables"],
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    refs = out.get("evidence_refs")
    if not isinstance(refs, list) or not refs or any(
            not isinstance(ref, str) or ref not in available for ref in refs):
        issues.append("evidence_refs must resolve to this execution's code/result evidence")
        refs = []
    if not set(refs).intersection(("artifacts.submitted_script", "solver.code_hash")):
        issues.append("method fidelity needs a code reference")
    if not set(refs).intersection(("artifacts.workspace_result", "execution_features.solution_variables")):
        issues.append("acceptance needs a result reference")
    out["resolved_refs"] = {ref: copy.deepcopy(available[ref]) for ref in refs}
    out["method_planned"] = method_key(record.method_planned)
    out["method_actual"] = method_key(record.method_actual)
    out["execution_id"] = record.execution_id
    out["issues"] = issues
    return out


def observe_faithful_g(summary, records, prediction):
    """Observe the terminal in-scope execution, never an earlier known one."""
    benefit = prediction.benefit
    candidate = prediction.candidate
    fact = {
        "kind": "effective_completion", "metric": FAITHFUL_G_METRIC,
        "unit": "probability", "scope": candidate.scope,
        "observation_rule_version": benefit.observation_rule_version,
        "rule": "terminal in-scope attempt with sourced fidelity and acceptance audit",
    }
    reasons = []
    if summary.eligibility.get("scope.execution"):
        reasons.append("the declared execution scope has missing records")
    if benefit.observation_rule_version != FAITHFUL_G_OBSERVATION_VERSION:
        reasons.append("unsupported or missing faithful G observation rule")
    if not records:
        reasons.append("no in-scope execution")
    else:
        record = records[-1]
        features = record.execution_features or {}
        report = features.get("task_check") or {}
        audit = report.get("g_attribution") or {}
        fact.update(execution_id=record.execution_id,
                    task_check={"state": report.get("state"),
                                "scope": copy.deepcopy(report.get("scope")),
                                "action_id": report.get("action_id")},
                    attribution=copy.deepcopy(audit))
        if features.get("executed") is False or (record.quality or {}).get("status") in ("error", "timeout"):
            reasons.append("crash/timeout or no executed evaluable result")
        if not audit or audit.get("issues"):
            reasons.append("missing or invalid sourced G attribution")
        if audit and checked_attribution(record, audit).get("issues"):
            reasons.append("attribution references no longer resolve to the execution evidence")
        if not candidate.method or method_key(record.method_planned) != method_key(candidate.method):
            reasons.append("the executed plan is not established as this candidate method")
        if not record.method_actual:
            reasons.append("method_actual was not recorded")
        if audit.get("method_fidelity") != "faithful":
            reasons.append("execution deviated from the candidate or fidelity is unknown")
        if audit and (audit.get("execution_id") != record.execution_id
                or audit.get("method_planned") != method_key(record.method_planned)
                or audit.get("method_actual") != method_key(record.method_actual)):
            reasons.append("attribution does not identify this execution and its methods")
        if audit.get("acceptance_scope") != "original_task" or not audit.get("acceptance_id"):
            reasons.append("checker coverage of original-task acceptance is insufficient")
        if report.get("intent") or not (report.get("scope") or {}).get("basis"):
            reasons.append("intermediate result or no declared acceptance basis")
        state = report.get("state")
        if state not in ("passed", "failed"):
            reasons.append("no decisive task-check verdict")
        if state == "failed" and audit.get("failure_source") != "candidate_method":
            reasons.append("failure is not attributed to the candidate method")
        if state == "passed" and audit.get("failure_source") not in ("unknown", None):
            reasons.append("passing result contradicts the failure-source assessment")
        if not reasons:
            fact["observed"] = 1.0 if state == "passed" else 0.0
            fact["acceptance_id"] = audit["acceptance_id"]
            fact["evidence_refs"] = copy.deepcopy(audit.get("resolved_refs"))
            # A business objective is a case fact, never a second binary win.
            if any(c.get("ok") and (c.get("check") == "recompute_objective" or
                    c.get("check") == "reference_objective" and
                    (report.get("reference_source") or {}).get("independent"))
                   for c in report.get("checks") or []):
                fact["verified_objective"] = (record.quality or {}).get("objective")
    summary.benefit = fact
    if reasons:
        summary.eligibility["benefit"] = {
            "eligibility": "unobserved", "reason": "; ".join(reasons)}


def conditional_g_statistics(samples, min_samples):
    """Count independent executed facts; score only numeric/label pairs.

    samples are (evaluation, prediction) pairs from the existing closed
    episode window. Repeated predictions use the earliest prediction for
    each execution/range; other predictions remain in diagnostic groups.
    """
    groups = {}
    selected = {}
    for evaluation, prediction in sorted(samples, key=lambda pair: (
            pair[1].trace.created_at, pair[1].prediction_id)):
        benefit = prediction.benefit
        if benefit is None or benefit.metric != FAITHFUL_G_METRIC:
            continue
        fact = evaluation.benefit or {}
        info = prediction.trace.model_info or {}
        if not info.get("bound_action_id"):
            continue
        ids = tuple(sorted({str(x) for x in (fact.get("execution_ids") or [])}))
        if not ids:
            ids = (str(info["bound_action_id"]),)
        identity = (benefit.metric, benefit.observation_rule_version, evaluation.scope,
                    evaluation.model_identity, evaluation.task_id, evaluation.episode_id, ids)
        previous = selected.get(identity)
        if previous and ((previous[0].benefit or {}).get("label_eligibility") == "attributable"
                         or fact.get("label_eligibility") != "attributable"):
            continue
        selected[identity] = (evaluation, prediction)
    # Choose one fact before condition grouping: a later context's remaining
    # budget may differ, but that cannot turn one execution into two units.
    for evaluation, prediction in selected.values():
        benefit = prediction.benefit
        fact = evaluation.benefit or {}
        info = prediction.trace.model_info or {}
        family, cell = info.get("family") or "unknown", info.get("cell_token") or "unknown"
        rule = benefit.observation_rule_version or "unknown"
        execution_conditions = info.get("g_execution_conditions")
        key = json.dumps([benefit.metric, rule, evaluation.scope, family, cell,
                          evaluation.model_identity, execution_conditions], ensure_ascii=False, sort_keys=True)
        group = groups.setdefault(key, {"metric": benefit.metric,
            "observation_rule_version": rule, "scope": evaluation.scope,
            "conditions": {"family": family, "cell": cell},
            "execution_conditions": copy.deepcopy(execution_conditions),
            "model_identity": evaluation.model_identity, "units": {}})
        ids = sorted({str(x) for x in (fact.get("execution_ids") or [])})
        if not ids:
            # Do not merge unrelated unresolved bindings into one fact.
            ids = [str(info.get("bound_action_id") or prediction.prediction_id)]
        identity = (evaluation.task_id, evaluation.episode_id, tuple(ids))
        label = fact.get("observed") if fact.get("label_eligibility") == "attributable" else None
        if identity in group["units"]:
            previous = group["units"][identity]
            if previous["label"] is not None or label is None:
                continue
        p = fact.get("predicted")
        group["units"][identity] = {"label": label,
            "predicted": p if is_g_probability(p) and label in (0, 1) else None,
            "reason": fact.get("reason"), "evaluation_id": evaluation.evaluation_id}
    out = []
    for group in groups.values():
        units = group.pop("units")
        labelled = [u for u in units.values() if u["label"] in (0, 1)]
        scored = [u for u in labelled if u["predicted"] is not None]
        episodes = len({key[:2] for key in units})
        labelled_episodes = len({key[:2] for key, u in units.items() if u["label"] in (0, 1)})
        bins = []
        for index in range(5):
            bucket = [u for u in scored if min(4, int(u["predicted"] * 5)) == index]
            if bucket:
                bins.append({"lo": index / 5, "hi": (index + 1) / 5, "n": len(bucket),
                    "mean_prediction": round(sum(u["predicted"] for u in bucket) / len(bucket), 6),
                    "pass_rate": round(sum(u["label"] for u in bucket) / len(bucket), 6)})
        group.update(passed=sum(u["label"] == 1 for u in labelled),
            failed=sum(u["label"] == 0 for u in labelled), unknown=len(units)-len(labelled),
            n_execution_units=len(units), n_distinct_episodes=episodes,
            n_labelled_episodes=labelled_episodes, n_scored=len(scored),
            attributable_label_coverage=round(len(labelled)/len(units), 6),
            brier=round(sum((u["predicted"]-u["label"])**2 for u in scored)/len(scored), 6) if scored else None,
            probability_bins=bins,
            unknown_reasons=dict(Counter(u["reason"] or "no attributable label"
                for u in units.values() if u["label"] is None)),
            support="observed conditions only" if labelled_episodes >= min_samples else "sparse: these executed cases only",
            evidence_evaluation_ids=[u["evaluation_id"] for u in units.values()],
            deduplication="one terminal execution/range; earliest attributable prediction per unit")
        out.append(group)
    return out


def same_task_comparisons(rows):
    """Pair distinct ACTUAL methods only on the same problem and checker."""
    eligible = []
    seen = set()
    for row in rows:
        fact = row.get("g_observed_fact") or row.get("benefit") or {}
        if fact.get("label_eligibility") != "attributable":
            continue
        actual = row.get("method_actual") or {}
        method = actual.get("steps")
        if method == "same as method_planned.steps":
            method = (row.get("method_planned") or {}).get("steps")
        if not method:
            continue
        identity = tuple(sorted(row.get("execution_ids") or []))
        if not identity or identity in seen:
            continue
        seen.add(identity)
        eligible.append((row, method, fact))
    out = []
    for i, (left, lm, a) in enumerate(eligible):
        for right, rm, b in eligible[i+1:]:
            problem = left.get("problem_identity_digest")
            if (not problem or problem != right.get("problem_identity_digest")
                    or left.get("task_id") != right.get("task_id") or lm == rm
                    or left.get("execution_conditions") != right.get("execution_conditions")
                    or left.get("scope") != right.get("scope")
                    or not a.get("acceptance_id") or a.get("acceptance_id") != b.get("acceptance_id")
                    or a.get("metric") != b.get("metric")
                    or a.get("observation_rule_version") != b.get("observation_rule_version")):
                continue
            out.append({"task_id": left["task_id"], "problem_identity_digest": problem,
                "acceptance_id": a["acceptance_id"], "scope": left.get("scope"),
                "metric": a["metric"], "observation_rule_version": a["observation_rule_version"],
                "left": {"evaluation_id": left["evaluation_id"], "method_actual": {
                            **left.get("method_actual", {}), "steps": copy.deepcopy(lm)},
                         "observed": a["observed"], "evidence_refs": a.get("evidence_refs")},
                "right": {"evaluation_id": right["evaluation_id"], "method_actual": {
                            **right.get("method_actual", {}), "steps": copy.deepcopy(rm)},
                          "observed": b["observed"], "evidence_refs": b.get("evidence_refs")},
                "acceptance_comparison": "tie" if a["observed"] == b["observed"] else
                    ("left_passed_right_failed" if a["observed"] > b["observed"] else "right_passed_left_failed")})
    return out


def case_comparisons_for_rows(comparisons, rows, limit):
    """Keep the observed partner of a displayed case, even if its row was
    omitted by the existing per-strategy display ceiling. Both sides are
    self-contained facts with method steps and evidence references.
    """
    displayed = {row.get("evaluation_id") for row in rows}
    relevant = [copy.deepcopy(c) for c in comparisons if
        c.get("left", {}).get("evaluation_id") in displayed or
        c.get("right", {}).get("evaluation_id") in displayed]
    return {"same_task_comparisons": relevant[:limit],
            "n_same_task_comparisons_available": len(relevant),
            "n_same_task_comparisons_omitted": max(0, len(relevant)-limit)}
