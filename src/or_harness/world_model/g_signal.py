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


def g_observation_convention(prediction):
    """A framework-fixed G request survives a missing model benefit block.

    A recorded explicit legacy convention takes precedence over the payload.
    Older new-G predictions without request metadata keep their stated rule.
    """
    info = getattr(getattr(prediction, "trace", None), "model_info", {}) or {}
    declared = info.get("benefit_convention")
    if declared is not None:
        return declared if declared.get("metric") == FAITHFUL_G_METRIC else None
    benefit = getattr(prediction, "benefit", None)
    if benefit is not None and benefit.metric == FAITHFUL_G_METRIC:
        return benefit.to_dict()
    return None


def canonical_evidence(value):
    """Numeric JSON spellings such as 60 and 60.0 have one rule identity."""
    if isinstance(value, dict):
        return {k: canonical_evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_evidence(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def actual_solving_conditions(record, planned_config=None):
    """Separate observed solver settings from instructions and predictions."""
    block = (record.execution_features or {}).get("execution_config") or {}
    values, sources = block.get("values") or {}, block.get("sources") or {}
    observed = {key: copy.deepcopy(value) for key, value in values.items()
                if sources.get(key) in ("executor", "script_reported") and key != "solver"}
    solver = (record.solver or {}).get("name")
    unknown = [key for key in (planned_config or {}) if key != "solver" and observed.get(key) is None]
    if not solver:
        unknown.append("solver")
    if observed.get("script_timeout_s") is None:
        unknown.append("script_timeout_s")
    if not any(observed.get(key) is not None for key in ("time_limit", "solver_timeout_s")):
        unknown.append("solver time limit")
    if block.get("conflicts"):
        unknown.extend("conflict:" + key for key in block["conflicts"])
    key = {"solver": solver, "config": observed}
    return {**key, "comparison_key": canonical_evidence(key), "comparable": not unknown,
        "unknown": sorted(set(unknown)), "sources": {
            "solver": "ExecutionRecord.solver.name",
            **{name: sources[name] for name in observed}},
        "execution_id": record.execution_id,
        "source": "ExecutionRecord and execution_features.execution_config"}


def requested_solving_conditions(candidate, constraints):
    """Stable declarations for applicability, never an actual-execution receipt."""
    return {"solver": getattr(candidate, "solver", None),
        "config": copy.deepcopy(getattr(candidate, "config", None) or {}),
        "constraints": {key: copy.deepcopy((constraints or {}).get(key))
                        for key in ("executor", "declared_budget")}}


def _acceptance_support(report):
    """Only checks that ran and support THIS verdict can supply a G label."""
    decisive = False if report.get("state") == "failed" else True
    trusted = []
    for check in report.get("checks") or []:
        if check.get("source") != "framework" or check.get("ran") is False:
            continue
        if check.get("ok") is not decisive:
            continue
        kind = check.get("check")
        if kind in ("reference_objective", "reference_status"):
            if not (report.get("reference_source") or {}).get("independent"):
                continue
        elif kind not in ("semantic_probe", "integer_domains", "recompute_objective"):
            continue
        trusted.append(kind)
    return trusted


def _structure(cell):
    return {part.split("[", 1)[0].split("=", 1)[0]: part
            for part in str(cell or "").split("|") if part and "unknown" not in part.lower()}


def select_conditional_g(calibration, *, profile, conditions, convention, scope=None,
                         task_id="", limit=6):
    """Bound applicable facts BEFORE general context-size trimming.

    Unknown structure/settings never count as a match. Retained partial
    matches describe cases, not a calibrated prior for the current candidate.
    The complete groups remain on the stored/audit calibration summary.
    """
    from or_harness.core.schema import ProblemProfile, group_key
    out = copy.deepcopy(calibration or {})
    if "conditional_g" not in out:
        return out  # preserve legacy/empty provider views without new evidence
    rows = out.get("conditional_g") or []
    family = profile.get("family")
    current = _structure(group_key(ProblemProfile.from_dict(profile))) if profile.get("problem_id") and family else {}
    wanted_config = conditions.get("config") or {}
    wanted_solver = conditions.get("solver")
    executor = (conditions.get("constraints") or {}).get("executor") or {}
    if executor.get("timeout_seconds") is not None:
        wanted_config = {**wanted_config, "script_timeout_s": executor["timeout_seconds"]}
    ranked = []
    for index, row in enumerate(rows):
        if (not family or (row.get("conditions") or {}).get("family") != family
                or row.get("metric") != convention.get("metric")
                or row.get("observation_rule_version") != convention.get("observation_rule_version")
                or scope is not None and row.get("scope") != scope):
            continue
        structure = _structure((row.get("conditions") or {}).get("cell"))
        if any(structure[k] != current[k] for k in structure.keys() & current.keys()):
            continue
        actual = row.get("execution_conditions") or {}
        actual = actual.get("comparison_key") or actual
        known_config = actual.get("config") or {}
        if wanted_solver and actual.get("solver") and wanted_solver != actual["solver"]:
            continue
        if any(k in known_config and known_config[k] != value for k, value in wanted_config.items()):
            continue
        budget = (conditions.get("constraints") or {}).get("declared_budget")
        if budget != row.get("declared_budget"):
            continue
        unknown = sorted({"class", "rc", "tc", "rx"} - (current.keys() & structure.keys()))
        unknown += [k for k in wanted_config if k not in known_config]
        unknown += [k for k in known_config if k not in wanted_config]
        if not wanted_solver or not actual.get("solver"):
            unknown.append("solver")
        if not current or not structure:
            unknown.append("structure")
        item = copy.deepcopy(row)
        item["applicability"] = {"kind": "partial_case_support" if unknown else "matching_conditions",
            "unknown_fields": sorted(set(unknown)),
            "note": "unknown dimensions are not matches or a current-candidate prior; sparse support remains limited to executed cases"}
        shared = len(current.keys() & structure.keys())
        same_task = task_id in (row.get("task_ids") or [])
        ranked.append((not same_task, -shared, len(unknown), index, item))
    ranked.sort(key=lambda entry: entry[:4])
    out["conditional_g"] = [entry[-1] for entry in ranked[:limit]]
    out["conditional_g_selection"] = {"n_available": len(rows), "n_applicable": len(ranked),
        "n_included": len(out["conditional_g"]), "n_omitted": len(rows)-len(out["conditional_g"]),
        "limit": limit, "basis": "same family, compatible known structure and solving limits; unknown stays unknown"}
    return out


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
    convention = g_observation_convention(prediction) or {}
    candidate = prediction.candidate
    fact = {
        "kind": "effective_completion", "metric": FAITHFUL_G_METRIC,
        "unit": "probability", "scope": candidate.scope,
        "observation_rule_version": convention.get("observation_rule_version"),
        "rule": "terminal in-scope attempt with sourced fidelity and acceptance audit",
    }
    reasons = []
    if summary.eligibility.get("scope.execution"):
        reasons.append("the declared execution scope has missing records")
    if convention.get("observation_rule_version") != FAITHFUL_G_OBSERVATION_VERSION:
        reasons.append("unsupported or missing faithful G observation rule")
    if not records:
        reasons.append("no in-scope execution")
    else:
        record = records[-1]
        features = record.execution_features or {}
        report = features.get("task_check") or {}
        audit = report.get("g_attribution") or {}
        conditions = actual_solving_conditions(record, candidate.config)
        if len(records) > 1:
            in_scope = [actual_solving_conditions(r, candidate.config) for r in records]
            distinct = {json.dumps(c["comparison_key"], sort_keys=True) for c in in_scope}
            if len(distinct) > 1 or any(not c["comparable"] for c in in_scope):
                conditions["comparable"] = False
                conditions["unknown"].append("in-scope conditions differ or are unconfirmed")
                conditions["comparison_key"] = {**conditions["comparison_key"],
                    "in_scope_conditions": [json.loads(key) for key in sorted(distinct)]}
            conditions["in_scope_execution_ids"] = [r.execution_id for r in records]
        fact.update(execution_id=record.execution_id,
                    execution_conditions=conditions,
                    acceptance_rule=copy.deepcopy(report.get("acceptance_rule")),
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
        support = _acceptance_support(report)
        fact["acceptance_support"] = support
        if not support:
            reasons.append("no trusted executed acceptance check supports the verdict; self-derived or unstated reference values are insufficient")
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
        convention = g_observation_convention(prediction)
        if convention is None:
            continue
        fact = evaluation.benefit or {}
        info = prediction.trace.model_info or {}
        if not info.get("bound_action_id"):
            continue
        ids = tuple(sorted({str(x) for x in (fact.get("execution_ids") or [])}))
        if not ids:
            ids = (str(info["bound_action_id"]),)
        identity = (convention["metric"], convention.get("observation_rule_version"), evaluation.scope,
                    evaluation.model_identity, evaluation.task_id, evaluation.episode_id, ids)
        previous = selected.get(identity)
        if previous and ((previous[0].benefit or {}).get("label_eligibility") == "attributable"
                         or fact.get("label_eligibility") != "attributable"):
            continue
        selected[identity] = (evaluation, prediction)
    # Choose one fact before condition grouping: a later context's remaining
    # budget may differ, but that cannot turn one execution into two units.
    for evaluation, prediction in selected.values():
        convention = g_observation_convention(prediction)
        fact = evaluation.benefit or {}
        info = prediction.trace.model_info or {}
        family, cell = info.get("family") or "unknown", info.get("cell_token") or "unknown"
        rule = convention.get("observation_rule_version") or "unknown"
        actual = fact.get("execution_conditions") or {}
        execution_conditions = actual.get("comparison_key") or {"solver": None, "config": {}}
        budget = ((info.get("g_execution_conditions") or {}).get("constraints") or {}).get("declared_budget")
        key = json.dumps([convention["metric"], rule, evaluation.scope, family, cell,
                          evaluation.model_identity, execution_conditions, budget], ensure_ascii=False, sort_keys=True)
        group = groups.setdefault(key, {"metric": convention["metric"],
            "observation_rule_version": rule, "scope": evaluation.scope,
            "conditions": {"family": family, "cell": cell},
            "execution_conditions": copy.deepcopy(execution_conditions),
            "declared_budget": copy.deepcopy(budget), "task_ids": [],
            "model_identity": evaluation.model_identity, "units": {}})
        if evaluation.task_id not in group["task_ids"]:
            group["task_ids"].append(evaluation.task_id)
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
            lc, rc = left.get("execution_conditions") or {}, right.get("execution_conditions") or {}
            ar, br = a.get("acceptance_rule") or {}, b.get("acceptance_rule") or {}
            if (not problem or problem != right.get("problem_identity_digest")
                    or left.get("task_id") != right.get("task_id") or lm == rm
                    or not lc.get("comparable") or not rc.get("comparable")
                    or lc.get("comparison_key") != rc.get("comparison_key")
                    or not ar.get("fingerprint") or ar.get("fingerprint") != br.get("fingerprint")
                    or left.get("scope") != right.get("scope")
                    or not a.get("acceptance_id") or a.get("acceptance_id") != b.get("acceptance_id")
                    or a.get("metric") != b.get("metric")
                    or a.get("observation_rule_version") != b.get("observation_rule_version")):
                continue
            out.append({"task_id": left["task_id"], "problem_identity_digest": problem,
                "acceptance_id": a["acceptance_id"], "scope": left.get("scope"),
                "metric": a["metric"], "observation_rule_version": a["observation_rule_version"],
                "acceptance_rule_fingerprint": ar["fingerprint"],
                "execution_conditions": copy.deepcopy(lc["comparison_key"]),
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
