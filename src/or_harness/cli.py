"""orx: thin CLI shell over ORHarness.

stdout is ALWAYS a single compact JSON object:
    {"result": {...}, "summary": "2-4 sentence agent-readable text"}
Exit codes: 0 success; 2 usage/precondition error; 1 crash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from or_harness.api import ORHarness, PREDICTION_MODES
from or_harness.core.coupling import CIRFormatError
from or_harness.core.schema import ExecutionRecord
from or_harness.core.storage import StorageError


def _emit(result: Dict[str, Any], summary: str) -> int:
    print(json.dumps({"result": result, "summary": summary},
                     ensure_ascii=False, separators=(",", ":")))
    return 0


def _fail(message: str, exit_code: int = 2) -> int:
    print(json.dumps({"result": {"error": message}, "summary": message},
                     ensure_ascii=False, separators=(",", ":")))
    return exit_code


def _fail_structured(error: Dict[str, Any], summary: str,
                     exit_code: int = 2) -> int:
    """A precondition failure whose payload is a MACHINE-READABLE object.

    ``_fail`` carries one prose string, which is enough for "unknown
    strategy_id" but not for a malformed CIR: the agent needs the offending
    key(s) and the repair hint as separate fields to act on without parsing
    English. Same exit code as every other precondition failure.
    """
    print(json.dumps({"result": {"error": error}, "summary": summary},
                     ensure_ascii=False, separators=(",", ":")))
    return exit_code


def _online_gains_line(summary: Any) -> str:
    """One line describing the ONLINE capability-gain follow-up state."""
    if not isinstance(summary, dict) or not summary.get("n_online_gains"):
        return ""
    return (f" Online gains: {summary['n_online_gains']} claimed, "
            f"{summary['n_pending']} pending, {summary['n_bound']} bound to "
            f"a real execution, {summary['n_effect_verified']} with a "
            "VERIFIED effect.")


def _cir_error_payload(exc: Any) -> Dict[str, Any]:
    """The structured error for a rejected CIR payload."""
    detail = getattr(exc, "detail", None)
    hint = getattr(exc, "hint", None)
    if detail is None:                       # a plain ValueError/TypeError
        detail, hint = str(exc), ("Check the CIR shape: an object with "
                                  "list-valued entities/decisions/constraints/"
                                  "relations.")
    payload: Dict[str, Any] = {"kind": "cir_format",
                               "cause": getattr(exc, "kind", "invalid"),
                               "detail": detail, "hint": hint}
    payload.update(getattr(exc, "ctx", {}) or {})
    return payload


def _load_json_arg(value: str) -> Any:
    """Accept a JSON literal or a path to a JSON file. JSON literals win when
    the string parses as JSON; otherwise an existing path is read."""
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    path = Path(value)
    if path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _parse_dimension_pairs(text: str) -> Dict[str, float]:
    """Parse 'llm_tokens=1500,tool_calls=8' into a float-valued dict.

    One parser for the three places a command takes dimension=value input
    (cost weights, cost overrides, budget declarations): they must agree on
    what a malformed pair means, and a second copy would be a second answer."""
    try:
        return {k: float(v) for k, v in
                (pair.split("=") for pair in text.split(","))}
    except ValueError as exc:
        raise ValueError(
            f"expected 'name=value,name=value', got {text!r}: {exc}") from exc


def _known_usage_hosts() -> List[str]:
    """The host names ``--usage-host`` can resolve, for error messages."""
    from or_harness.world_model.usage import HOST_USAGE_ADAPTERS
    return sorted(HOST_USAGE_ADAPTERS)


def _wm_env_float(name: str) -> Optional[float]:
    """Read a float from the environment, or None when unset/invalid.

    An unparsable value is reported rather than silently ignored: a typo in
    ``OR_WM_MAX_TOKENS=8k192`` must not look like "not configured".
    """
    import os
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number")


def _wm_max_tokens(args) -> int:
    """The output budget actually used: CLI flag > env var > class default.

    The precedence is explicit and reported, because the value changes what
    the endpoint is ASKED for and a silent fallback would make a truncated
    answer look like a model failure. The default is deliberately left at
    the adapter's own default (2048): a larger budget is a deployment
    decision, not something this build should assume for every model.
    """
    explicit = getattr(args, "wm_max_tokens", None)
    if explicit is not None:
        return int(explicit)
    from or_harness.world_model.provider import HttpChatProvider as _P
    env = _wm_env_float("OR_WM_MAX_TOKENS")
    if env is not None:
        return int(env)
    import inspect
    return int(inspect.signature(_P.__init__).parameters[
        "max_output_tokens"].default)


def _wm_timeout(args) -> float:
    """The world-model socket timeout actually used: flag > env > default.

    The ``--wm-timeout`` default is ``None`` rather than a number so that
    "not given" is DISTINGUISHABLE from "given as 300": with a numeric
    default, ``$OR_WM_TIMEOUT`` could never win, and an operator who sets
    the environment variable would be silently ignored. The fallback is
    the adapter's own default (read from the signature, not duplicated),
    so this layer and the provider cannot disagree about it.
    """
    explicit = getattr(args, "wm_timeout", None)
    if explicit is not None:
        return float(explicit)
    env = _wm_env_float("OR_WM_TIMEOUT")
    if env is not None:
        return float(env)
    from or_harness.world_model.provider import HttpChatProvider as _P
    import inspect
    return float(inspect.signature(_P.__init__).parameters[
        "timeout_s"].default)


def _harness(args) -> ORHarness:
    weights = None
    if getattr(args, "cost_weights", None):
        weights = _parse_dimension_pairs(args.cost_weights)
    provider = None
    wm = getattr(args, "world_model", None)
    if wm:
        from or_harness.world_model.provider import HttpChatProvider
        parts = wm.split("::", 1)
        if len(parts) != 2:
            raise ValueError(
                "--world-model must be BASE_URL::MODEL (api key comes from "
                "the OR_WM_API_KEY environment variable)")
        import os
        api_key = os.environ.get("OR_WM_API_KEY", "")
        provider = HttpChatProvider(
            parts[0], parts[1], api_key,
            timeout_s=_wm_timeout(args),
            max_output_tokens=_wm_max_tokens(args))
    return ORHarness(home=args.home, alpha=args.alpha, beta=args.beta,
                     gamma=args.gamma, cost_weights=weights,
                     delta=getattr(args, "delta", 0.0),
                     prediction_mode=getattr(args, "prediction_mode",
                                            "h-x-b-value"),
                     world_model=provider)


#: Candidate identity fields that must ALL match before a plan entry is read
#: as "the entry the suggestion is about". ``strategy_id`` alone is not an
#: identity: the same strategy run with a different solver or config is a
#: DIFFERENT candidate with its own prediction, and handing over the wrong
#: id loses the sample to a binding mismatch.
_CANDIDATE_IDENTITY_FIELDS = ("action_type", "strategy_id", "solver")


def _candidate_identity(spec: Dict[str, Any]) -> tuple:
    """The tuple that makes one planned candidate distinguishable.

    ``params`` is normalised to sorted items so two dicts with the same
    content but a different insertion order compare equal.
    """
    params = spec.get("params") or {}
    return tuple(
        [spec.get(field) for field in _CANDIDATE_IDENTITY_FIELDS]
        + [json.dumps(params, sort_keys=True, default=str)])


def _matching_candidate_prediction_id(compared: Sequence[Dict[str, Any]],
                                      suggested: Dict[str, Any]
                                      ) -> Optional[str]:
    """The prediction id of the candidate the suggestion NAMES.

    Matched on the full candidate identity. Returns ``None`` when no entry
    matches — a caller must never be handed a plausible-looking id for a
    different candidate, because the subsequent bind would reject it and the
    real prediction would go unevaluated.
    """
    wanted = _candidate_identity(suggested)
    for entry in compared:
        if _candidate_identity(entry.get("action_spec") or {}) == wanted:
            return entry.get("prediction_id")
    return None


def _summarize_recall(result: Dict[str, Any]) -> str:
    recs = result["recommendations"]
    parts: List[str] = []
    if not recs:
        basis = result.get("recommendations_basis") or {}
        parts.append("No memory for this problem. "
                     + str(basis.get("reason") or ""))
    else:
        top = recs[0]
        parts.append(f"Top recalled strategy: {top['strategy_id']}, "
                     f"score {top['score']}, evidence={top['evidence']}, "
                     f"E[Q]={top['expected']['quality']}, "
                     f"P(fail)={top['expected']['failure_prob']}.")
        if top["risk_warnings"]:
            parts.append("Warnings: " + "; ".join(top["risk_warnings"]))
    vector = result.get("vector_recall")
    if vector:
        n_exec = len(vector.get("execution_evidence") or [])
        n_kn = len(vector.get("strategic_knowledge") or [])
        backend = vector.get("backend") or {}
        parts.append(f"Text similarity ({backend.get('model_id')}): "
                     f"{n_exec} execution(s), {n_kn} knowledge entr(y/ies) "
                     "surfaced. Similarity is a DISCOVERY signal only — "
                     "cross-cell hits are labels, never reusable "
                     "statistics.")
        unindexed = vector.get("unindexed") or {}
        if unindexed.get("execution_evidence") or \
                unindexed.get("strategic_knowledge"):
            parts.append(f"{unindexed.get('execution_evidence', 0)} rejected "
                         "candidate(s) are excluded from the text channel "
                         "(no vector yet); they remain visible via "
                         "profile retrieval and `orx inspect`. See "
                         "vector_recall.unindexed.note.")
    elif result.get("degraded"):
        parts.append(f"Text search skipped ({result['degraded']['reason']}); "
                     "recall fell back to profile matching.")
    advisories = result.get("solver_advisories") or []
    for adv in advisories:
        parts.append(f"Solver advisory: {adv['solver']} has "
                     f"{adv['environment_failures']} environment-class "
                     f"failure(s) in this memory ({', '.join(adv['error_classes'])}); "
                     f"consider a different solver.")
    for w in result.get("coupling_warnings") or []:
        parts.append("WARNING: " + w["message"])
    parts.append(f"{len(recs)} candidates returned. You remain the orchestrator: "
                 "you may refuse, exclude, or override any of them.")
    return " ".join(parts)


def cmd_profile(args) -> int:
    """The single analysis entry: CIR validation + modeling guidance +
    problem profile + derivation report, in one call.

    A task WITHOUT a ``model`` field is a normal state: strategy selection
    relies on the task text, the CIR, and the profile — the model is an
    intermediate representation written AFTER the strategy is chosen, so it
    is VERIFIED here (L1/L2) and its coupling is reported as a diagnostic,
    never used to move the structural key."""
    h = _harness(args)
    try:
        from or_harness.core.coupling import (
            CIRFormatError,
            cir_from_task,
            cir_shape_problems,
            coerce_cir,
            derive_coupling_groups,
            infer_structural_relations,
            render_modeling_guidance,
            validate_cir,
        )
        from or_harness.profiling.model_syntax import verify_model
        task = _load_json_arg(args.task)
        allow_empty = bool(getattr(args, "allow_empty_cir", False))
        # -- CIR side (validation, structural inference, groups, guidance) --
        # The shape gate runs FIRST: a malformed CIR is rejected here with a
        # named key and a repair hint, instead of parsing to an empty
        # structure that silently runs the whole episode in [unknown].
        coupling: Dict[str, Any] = {"cir": None, "modeling_guidance": []}
        cir_obj = None
        try:
            if args.cir:
                cir_obj = coerce_cir(_load_json_arg(args.cir),
                                     allow_empty=allow_empty)
            else:
                cir_obj = cir_from_task(task, allow_empty=allow_empty)
        except (CIRFormatError, TypeError) as exc:
            payload = _cir_error_payload(exc)
            return _fail_structured(
                payload,
                f"CIR rejected ({payload['cause']}): {payload['detail']} "
                f"{payload['hint']}")
        if cir_obj is not None:
            validate_cir(cir_obj)
            parsed = None
            model_text = task.get("model")
            if isinstance(model_text, str) and model_text.strip():
                try:
                    parsed = verify_model(model_text).parsed
                except Exception:
                    parsed = None
            infer_structural_relations(cir_obj, parsed)
            derive_coupling_groups(cir_obj)
            coupling = {
                "cir": cir_obj.to_dict(),
                "modeling_guidance": render_modeling_guidance(cir_obj),
            }
        else:
            coupling["message"] = (
                "No 'coupling' field found in the task. A CIR is optional "
                "but recommended: it is the pre-model understanding that "
                "improves both the profile derivation and the model you "
                "write after choosing a strategy.")
        # With OR_CIR_STRICT=0 the policy checks are downgraded, not dropped:
        # every problem the gate found is still reported, as a lint. Without
        # this, "lenient" would mean "silent" — the exact failure the shape
        # gate exists to prevent.
        raw_cir = (_load_json_arg(args.cir) if args.cir
                   else task.get("coupling"))
        if raw_cir is not None:
            lints = [p for p in cir_shape_problems(
                raw_cir, allow_empty=allow_empty)
                if p["severity"] == "lint"]
            if lints:
                coupling["shape_lints"] = lints
        # -- Profile side --
        profile = h.profile(task, cir=cir_obj)
        report = h.derivation_report(task, cir=cir_obj)
        # An honest account of what the CIR actually contributed. Reported
        # even when the profile came out fully determined, because the
        # failure this guards against is a CIR that PARSED to nothing while
        # looking present — see the shape gate above.
        coupling["health"] = _cir_health(coupling.get("cir"))
        result = {"profile": profile.to_dict(), "derivation": report,
                  "coupling": coupling}
        return _emit(result, _summarize_profile(profile, report, coupling))
    finally:
        h.close()


def _cir_health(cir: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a parsed CIR really carries: counts, issues, and honesty flags.

    ``parsed`` is the field that matters. A CIR object that is PRESENT but
    contributes no decisions cannot derive any scalar dimension
    (``coupling_from_cir`` returns None for every one of them), so a caller
    that only checks ``cir is not None`` would believe it had a structural
    signal it does not have.
    """
    if cir is None:
        return {"present": False, "parsed": False, "entities": 0,
                "decisions": 0, "constraints": 0, "relations": 0,
                "issues": 0, "contributes_scalars": False,
                "note": ("no CIR supplied: the structural dimensions come "
                         "from the spec or stay unknown")}
    counts = {k: len(cir.get(k) or [])
              for k in ("entities", "decisions", "constraints", "relations")}
    parsed = any(counts.values())
    issues = len(cir.get("issues") or [])
    contributes = counts["decisions"] > 0
    if not parsed:
        note = ("the CIR is present but EMPTY: it derives no scalar "
                "dimensions, so the profile's coupling comes from the spec "
                "or stays unknown — this is not evidence of weak coupling")
    elif not contributes:
        note = ("the CIR has entities but no decisions: no scalar dimension "
                "can be derived from it (only decisions carry indexes and "
                "resource relations)")
    elif issues:
        note = (f"{issues} validation issue(s): the CIR was still used, but "
                "the structure it describes may be incomplete")
    else:
        note = "CIR parsed and contributes structural signal"
    return {"present": True, "parsed": parsed, **counts, "issues": issues,
            "contributes_scalars": contributes, "note": note}


def _summarize_profile(profile, report, coupling=None) -> str:
    coupling = coupling or {}
    cir = coupling.get("cir")
    parts = []
    if cir is not None:
        parts.append(
            f"CIR validated: {len(cir.get('entities', []))} entities, "
            f"{len(cir.get('decisions', []))} decisions, "
            f"{len(cir.get('constraints', []))} constraints, "
            f"{len(cir.get('relations', []))} relations.")
        guidance = coupling.get("modeling_guidance") or []
        if guidance:
            parts.append(f"Modeling guidance ({len(guidance)}): "
                         + "; ".join(f"[{g['type']}] {g['implication']}"
                                    for g in guidance[:4])
                         + (" ..." if len(guidance) > 4 else ""))
        health = coupling.get("health") or {}
        if health and not health.get("contributes_scalars"):
            parts.append("CIR contribution: " + str(health.get("note")) + ".")
    elif coupling.get("message"):
        parts.append(coupling["message"])
    parts.append(f"Profile for {profile.problem_id} "
                 f"(family={profile.family}):")
    for dim in ("resource_coupling", "temporal_coupling",
                "route_complexity", "semantic_coupling"):
        entry = report.get(dim) or {}
        value = entry.get("value")
        origin = entry.get("origin", "?")
        parts.append(f"{dim}={value} ({origin})" if value is not None
                     else f"{dim}=null ({origin})")
    verification = report.get("model_verification")
    if verification is not None:
        if verification.get("passed"):
            parts.append("Model representation verified (L1+L2).")
        else:
            issues = verification.get("issues") or []
            parts.append(f"Model representation has {len(issues)} issue(s): "
                         + "; ".join(f"[{i['layer']}] {i['code']}: {i['detail']}"
                                    for i in issues[:3])
                         + (" ..." if len(issues) > 3 else ""))
    model_coupling = report.get("model_coupling") or {}
    if model_coupling:
        parts.append("Model-structure coupling (diagnostic, not the key): "
                     + ", ".join(f"{k}={v}" for k, v in
                                 sorted(model_coupling.items())) + ".")
    warnings = report.get("coupling_warnings") or []
    for w in warnings:
        parts.append("WARNING: " + w["message"])
    return " ".join(parts)


def cmd_recall(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        result = h.recall(task, top=args.top,
                          exclude=args.exclude or [],
                          candidates=args.candidate or None,
                          memory_mode=args.memory_mode,
                          include_unverified=args.include_unverified)
        return _emit(result, _summarize_recall(result))
    finally:
        h.close()


def cmd_predict(args) -> int:
    """Pre-execution COST expectation snapshot for one (task, strategy)."""
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        snapshot = h.predict_cost(task, args.strategy)
        result = {"prediction": snapshot.to_dict()}
        if snapshot.expected_cost is None:
            summary = (f"No usable cost evidence for {args.strategy} "
                       f"(source=unknown). {snapshot.note or 'Expected cost is '
                       'UNKNOWN — not zero.'} Pass this snapshot back at "
                       "record time so feedback never fabricates an error "
                       "against a placeholder zero.")
        else:
            cost = {k: round(v, 4) for k, v
                    in snapshot.expected_cost.to_dict().items()}
            summary = (f"Expected cost for {args.strategy} (scope="
                       f"{snapshot.measurement_scope}, source={snapshot.source}, "
                       f"n={snapshot.support_n}, per-dim "
                       f"{snapshot.support_per_dim or 'n/a'}): {cost}. "
                       "Pass this snapshot back at record time so feedback "
                       "compares against the prediction actually used.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_execute(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        prediction_id = getattr(args, "prediction", None)
        method = _load_json_arg(args.method) if getattr(args, "method", None) \
            else None
        try:
            record = h.execute(task, args.strategy, args.code, args.workspace,
                               solver=args.solver,
                               episode_id=getattr(args, "episode", None),
                               prediction_id=prediction_id,
                               method=method)
        except ValueError as exc:
            # A prediction-driven precondition failure (unknown prediction,
            # a changed problem, a contradictory explicit argument, a
            # candidate already claimed, a solve script outside the
            # workspace): refused BEFORE anything ran, so the prediction is
            # not spent and no cost was incurred. Reported as a usage error
            # with the reason, never a crash.
            if prediction_id:
                return _fail(f"{exc}")
            raise
        except Exception as exc:
            # The executor raised AFTER the attempt started (a filesystem
            # error, an interruption). The attempt really ran and really
            # failed: its failure fact is staged with its cost and failure
            # class, and (for a prediction-driven run) the prediction is
            # freed so it can be tested against a real attempt. Report an
            # actionable line instead of a bare traceback, and point at the
            # staged failure so it is not mistaken for "nothing happened".
            staged = h.bank.pending(task_id=str(task.get("task_id", "")))
            staged_ids = [r.execution_id for r in staged]
            note = (f"the attempt started and FAILED before producing a "
                    f"record ({type(exc).__name__}: {exc}). Its failure fact "
                    "is staged (with its measured cost and failure class) "
                    "and will be counted at close-out"
                    + (f"; staged executions: {staged_ids}" if staged_ids
                       else "")
                    + ". Retry with a NEW attempt"
                    + (" (the prediction was freed for it)" if prediction_id
                       else "") + "; record the failure with `orx record "
                    "--from-staged <execution_id>` to keep it as a fact.")
            return _fail(note)
        out = {"execution": record.to_dict(),
               "execution_id": record.execution_id,
               "action_id": record.action_id}
        q = record.quality
        summary = (f"Execution {record.execution_id} finished with status "
                   f"{q['status']} (feasible={q['feasible']}, "
                   f"gap={q['gap']}). Cost so far: {record.cost.to_dict()}. "
                   "Nothing is recorded yet — call `orx record` to persist, "
                   "or discard.")
        binding = record.execution_features.get("prediction_binding")
        if prediction_id is not None:
            if binding and binding.get("bound"):
                info = binding.get("trace") or {}
                mismatch = info.get("binding_mismatch")
                unknown = info.get("binding_unknown")
                real_config = binding.get("config_observed") or {}
                if mismatch:
                    out["prediction_binding"] = binding
                    summary += (f" Prediction {prediction_id} was associated "
                                f"before the run and scored WITH MISMATCH "
                                f"({mismatch}): "
                                + (_attribution_summary(info)
                                   or "the comparison covers only matching "
                                      "parts; mismatched fields are "
                                      "recorded, never scored."))
                else:
                    out["prediction_binding"] = binding
                    summary += (f" Prediction {prediction_id} associated "
                                "before the run"
                                + (f" (predicted config confirmed: "
                                   f"{sorted(real_config)})"
                                   if real_config else "")
                                + (f" (unconfirmed fields: {unknown}; a "
                                   "missing receipt is a caveat, not a "
                                   "discard)"
                                   if unknown else "")
                                + "; the close-out will evaluate it.")
            else:
                out["prediction_binding"] = binding or {
                    "bound": False,
                    "reason": "the prediction was not bound"}
                summary += (f" WARNING: prediction {prediction_id} could NOT "
                            "be bound after the run ("
                            f"{(binding or {}).get('reason') or 'unknown'}); "
                            "bind it explicitly with `orx bind-strategy` "
                            "once the action is on record.")
        return _emit(out, summary)
    finally:
        h.close()


def cmd_record(args) -> int:
    h = _harness(args)
    try:
        if args.discard_staged:
            staged = h.bank.get_pending(args.discard_staged)
            if staged is None:
                return _fail(f"no staged execution {args.discard_staged!r}")
            h.bank.clear_pending(args.discard_staged)
            return _emit({"discarded": args.discard_staged},
                         f"Staged execution {args.discard_staged} discarded. "
                         "It never entered the Experience Bank.")
        if args.from_staged:
            staged = h.bank.get_pending(args.from_staged)
            if staged is None:
                return _fail(f"no staged execution {args.from_staged!r}")
            record = staged  # original payload, verbatim — no re-typing, no drift
        elif args.execution:
            data = _load_json_arg(args.execution)
            # Accept the bare record, the `orx execute` envelope
            # ({"result": {"execution": {...}}}) and the legacy one-level
            # form ({"execution": {...}}). `predict` already accepted all
            # three; requiring the caller to hand-strip the envelope made
            # the documented `execute > file` then `record --execution file`
            # flow fail with a confusing "execution_id is required".
            if isinstance(data.get("execution"), dict):
                data = data["execution"]
            elif isinstance(data.get("result", {}).get("execution"), dict):
                data = data["result"]["execution"]
            record = ExecutionRecord.from_dict(data)
        elif args.record_file:
            record = ExecutionRecord.from_dict(_load_json_arg(args.record_file))
        else:
            return _fail("record requires --execution <json|path>, "
                         "--from-staged <id>, or --record-file")
        override = None
        if args.override:
            override = _parse_dimension_pairs(args.override)
        # HOST USAGE: an attempt-level usage report the outer framework
        # captured (its hook / log / storage). The framework never guesses
        # tokens; a supplied report is the REAL spend, recorded with the
        # full口径 (prompt + completion) and marked as a host observation.
        # Three delivery channels, in precedence order:
        #   1. --usage-file <path|@path|JSON> — the report itself;
        #   2. --usage-host <host> — a named adapter locates the report
        #      (e.g. openclaw under <home>/host_usage/);
        #   3. $OR_HOST_USAGE_FILE — a default path, same as (1).
        host_usage = None
        usage_note = None
        if getattr(args, "usage_file", None):
            raw = _load_json_arg(args.usage_file)
            from or_harness.world_model.usage import (
                is_host_usage_report,
                normalize_host_usage,
            )
            host_usage = (raw if is_host_usage_report(raw)
                          else normalize_host_usage(
                              raw, source=getattr(args, "usage_source",
                                                  None)))
        elif getattr(args, "usage_host", None) \
                or os.environ.get("OR_HOST_USAGE_FILE"):
            from or_harness.world_model.usage import (
                is_host_usage_report,
                normalize_host_usage,
            )
            scope = {
                "home": args.home,
                "execution_id": (getattr(args, "from_staged", None)
                                 or (record.execution_id
                                     if record is not None else None)),
                "task_id": (record.task_id
                            if record is not None else None),
                "usage_file": os.environ.get("OR_HOST_USAGE_FILE"),
            }
            report = None
            host_name = getattr(args, "usage_host", None)
            if host_name:
                from or_harness.world_model.usage import host_usage_adapter
                adapter = host_usage_adapter(host_name)
                if adapter is None:
                    return _fail(
                        f"no usage adapter for host {host_name!r}: known "
                        "hosts are "
                        f"{sorted(set(_known_usage_hosts()))}. Pass the "
                        "report directly with --usage-file, or leave the "
                        "dimensions unknown (never zero)")
                report = adapter.load(scope)
                if report is None:
                    usage_note = (
                        f"the {host_name} adapter found no usage report for "
                        "this attempt: llm_tokens/tool_calls stay UNKNOWN "
                        "(never zero). Have the host produce the report, or "
                        "pass it with --usage-file")
            elif scope["usage_file"]:
                try:
                    report = _load_json_arg(scope["usage_file"])
                except (OSError, ValueError):
                    report = None
                    usage_note = (
                        f"$OR_HOST_USAGE_FILE={scope['usage_file']!r} could "
                        "not be read: the dimensions stay UNKNOWN (never "
                        "zero)")
            if report is not None:
                host_usage = (report if is_host_usage_report(report)
                              else normalize_host_usage(
                                  report, source=host_name))
        prediction = None
        if args.prediction:
            from or_harness.core.schema import PredictionSnapshot
            raw = _load_json_arg(args.prediction)
            # Accept either the bare snapshot JSON or the full `orx predict`
            # output envelope ({"result": {"prediction": {...}}, ...}).
            if isinstance(raw.get("prediction"), dict):
                raw = raw["prediction"]
            elif isinstance(raw.get("result", {}).get("prediction"), dict):
                raw = raw["result"]["prediction"]
            prediction = PredictionSnapshot.from_dict(raw)
        result = h.record(record, override=override,
                          override_mode=args.override_mode,
                          override_source=getattr(args, "override_source",
                                                 "agent_estimate"),
                          override_force=bool(getattr(args,
                                                      "override_force",
                                                      False)),
                          host_usage=host_usage,
                          prediction=prediction,
                          method=(_load_json_arg(args.method)
                                  if getattr(args, "method", None) else None),
                          method_actual=(_load_json_arg(args.method_actual)
                                         if getattr(args, "method_actual", None)
                                         else None))
        hints = result["induction_hints"]
        checks = result["prediction_checks"]
        summary = [f"Recorded {result['execution_id']}."]
        if checks:
            hits = sum(1 for c in checks if c["hit"])
            summary.append(f"Prediction checks: {hits}/{len(checks)} hits "
                           f"across matching entries.")
        if hints:
            summary.append("Induction hints: " + "; ".join(
                f"{h['pattern']}({','.join(h['strategy_ids'])})"
                for h in hints)
                + ". Hints are evidence, not orders — induce only when you judge "
                  "the pattern worth generalizing.")
        else:
            summary.append("No induction hints.")
        unrecorded = result.get("unrecorded_staged_executions") or []
        if unrecorded:
            summary.append(
                f"NOTE: {len(unrecorded)} staged execution(s) for this task are "
                f"still unrecorded ({', '.join(unrecorded)}). If one is a failed "
                "attempt you abandoned, record it with `orx record --from-staged "
                "<id>` — failures are the most valuable induction raw material.")
        if usage_note:
            summary.append(f"NOTE: {usage_note}.")
        return _emit(result, " ".join(summary))
    finally:
        h.close()


def cmd_check_task(args) -> int:
    """Check whether an execution's ANSWER satisfies the original task.

    The step between `execute` and `record`: the executor's own verdict says
    the MODEL was solved (a legal status, a finite objective, a gap), never
    that the answer is a valid answer to the TASK. A relaxed LP answered with
    fractional values is `optimal` with `gap=0` and still wrong, and without
    this call it would enter recall, the conditional statistics, the
    world-model feedback and offline induction as a success sample.
    """
    h = _harness(args)
    try:
        check = _load_json_arg(args.check) if args.check else {}
        if not isinstance(check, dict):
            return _fail("--check must be a JSON object")
        try:
            result = h.check_task_result(args.execution_id, check,
                                         episode_id=args.episode)
        except ValueError as exc:
            return _fail(str(exc))
        report = result["report"]
        state = report["state"]
        if state == "passed":
            summary = (f"Task check for {args.execution_id}: PASSED on "
                       f"{', '.join(report['scope']['basis'])}. This covers "
                       "the declared bases only — it is not a proof that the "
                       "model represents the task. Unchecked: "
                       + "; ".join(report["scope"]["unchecked"]) + ".")
        elif state == "failed":
            diffs = "; ".join(d["reason"] for d in report["diffs"][:3])
            summary = (f"Task check for {args.execution_id}: FAILED. {diffs}. "
                       "The attempt stays recorded with its real cost and "
                       "can be cited as contrast evidence; it can no longer "
                       "count as a success sample. Diagnose the cause "
                       "yourself and re-solve in the same episode — never "
                       "change the task to match a reference value.")
        else:
            summary = (f"Task check for {args.execution_id}: INSUFFICIENT — "
                       f"the answer's validity is UNKNOWN, not confirmed. "
                       f"{report['conclusion']}")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_amend_cost(args) -> int:
    """Backfill cost dimensions of an already-recorded execution.

    The documented repair path for an incomplete cost claim: ``update_cost``
    has always supported amending a stored fact in place, but it had no CLI
    surface, so the advice printed by ``record``/``induce`` pointed at
    nothing an agent could actually run."""
    h = _harness(args)
    try:
        dimensions = {}
        result_breakdown = None
        per_dim_sources: Dict[str, str] = {}
        if getattr(args, "usage_file", None) \
                or getattr(args, "usage_host", None) \
                or os.environ.get("OR_HOST_USAGE_FILE"):
            # A HOST usage report is the late-backfill channel: the outer
            # framework's own numbers, applied instead of hand-typed ones.
            # A versioned HostUsageReport (or-host-usage/1) can carry BOTH
            # llm_tokens (provider_usage) and tool_calls (agent_observed)
            # in one report, each recorded under its own provenance.
            report = None
            if getattr(args, "usage_file", None):
                try:
                    report = _load_json_arg(args.usage_file)
                except ValueError as exc:
                    return _fail(str(exc))
            else:
                host_name = getattr(args, "usage_host", None)
                if host_name:
                    from or_harness.world_model.usage import (
                        host_usage_adapter,
                    )
                    adapter = host_usage_adapter(host_name)
                    if adapter is None:
                        return _fail(
                            f"no usage adapter for host {host_name!r}: "
                            f"known hosts are "
                            f"{sorted(set(_known_usage_hosts()))}. Pass the "
                            "report directly with --usage-file, or leave "
                            "the dimensions unknown (never zero)")
                    report = adapter.load({"home": args.home,
                                           "execution_id": args.execution_id})
                    if report is None:
                        return _fail(
                            f"the {host_name} adapter found no usage report "
                            f"for {args.execution_id}: there is nothing to "
                            "backfill (a missing usage is UNKNOWN, never "
                            "zero). Have the host produce the report, or "
                            "pass it with --usage-file")
                else:
                    try:
                        report = _load_json_arg(
                            os.environ["OR_HOST_USAGE_FILE"])
                    except (OSError, ValueError):
                        return _fail(
                            "$OR_HOST_USAGE_FILE could not be read: the "
                            "dimensions stay UNKNOWN (never zero)")
            from or_harness.world_model.usage import (
                host_report_dimensions,
                host_usage_cost_vector,
                is_host_usage_report,
            )
            if is_host_usage_report(report):
                try:
                    dimensions, result_breakdown = \
                        host_report_dimensions(report)
                except ValueError as exc:
                    return _fail(str(exc))
                per_dim_sources = result_breakdown.get("provenance") or {}
                if not dimensions:
                    return _fail(
                        "the usage report's 'measured' whitelist carried no "
                        "cost dimension: there is nothing to backfill (a "
                        "missing usage is UNKNOWN, never zero)")
            else:
                vector, breakdown = host_usage_cost_vector(
                    report, source=getattr(args, "usage_source", None))
                if vector is None or "llm_tokens" not in \
                        vector.measured_dims():
                    return _fail(
                        "the usage report carried no token count: there is "
                        "nothing to backfill (a missing usage is UNKNOWN, "
                        "never zero). Provide the report from the host's "
                        "hook/log or hand-type --override llm_tokens=<number>")
                dimensions["llm_tokens"] = float(vector.llm_tokens)
                # A host report is a real observation: record it as such,
                # not as whatever --source defaulted to.
                per_dim_sources["llm_tokens"] = "provider_usage"
                result_breakdown = breakdown
        else:
            try:
                dimensions = _parse_dimension_pairs(args.override)
            except ValueError as exc:
                return _fail(str(exc))
        if not dimensions:
            return _fail("--override requires at least one dimension=value pair")
        from or_harness.core.schema import COST_DIMENSIONS
        unknown = sorted(set(dimensions) - set(COST_DIMENSIONS))
        if unknown:
            return _fail(f"unknown cost dimensions {unknown}; "
                         f"expected any of {list(COST_DIMENSIONS)}")
        try:
            # A versioned report may carry several dimensions with
            # DIFFERENT provenance (llm_tokens=provider_usage,
            # tool_calls=agent_observed): amend per dimension so each number
            # is recorded under where IT came from.
            if per_dim_sources and len(dimensions) > 1:
                record = None
                for dim, value in dimensions.items():
                    record = h.bank.update_cost(
                        args.execution_id,
                        mode=args.mode,
                        source=per_dim_sources.get(dim, "provider_usage"),
                        force=bool(getattr(args, "amend_force", False)),
                        basis=(result_breakdown or {}).get("basis")
                        if dim == "llm_tokens" else None,
                        **{dim: float(value)})
            else:
                record = h.bank.update_cost(
                    args.execution_id,
                    mode=args.mode,
                    source=(per_dim_sources.get(
                        next(iter(dimensions)),
                        getattr(args, "source", "agent_estimate"))
                        if per_dim_sources
                        else getattr(args, "source", "agent_estimate")),
                    force=bool(getattr(args, "amend_force", False)),
                    basis=(result_breakdown or {}).get("basis"),
                    **dimensions)
        except StorageError as exc:
            return _fail(str(exc))
        measured = sorted(record.cost.measured_dims())
        missing = [d for d in COST_DIMENSIONS if d not in measured]
        result = {
            "execution_id": record.execution_id,
            "mode": args.mode,
            "cost": {d: round(v, 6) for d, v in record.cost.to_dict().items()},
            "cost_measured": measured,
            "still_missing": missing,
        }
        if result_breakdown is not None:
            result["usage"] = result_breakdown
        summary = (f"Amended {record.execution_id} ({args.mode}): "
                   + ", ".join(f"{d}={dimensions[d]:g}" for d in dimensions)
                   + ".")
        if missing:
            summary += (f" Still unmeasured: {', '.join(missing)} — these "
                        "dimensions support no cost claim until backfilled.")
        else:
            summary += " Every cost dimension is now measured."
        return _emit(result, summary)
    finally:
        h.close()


def cmd_induce(args) -> int:
    h = _harness(args)
    try:
        notes = list(args.note or []) or None
        verify = None
        if getattr(args, "verify", None):
            try:
                verify = json.loads(args.verify)
            except json.JSONDecodeError as exc:
                return _fail(f"--verify must be JSON: {exc}", 2)
            if not isinstance(verify, dict):
                return _fail("--verify must be a JSON object", 2)
        relations = None
        if getattr(args, "relation", None):
            relations = []
            for raw in args.relation:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    return _fail(f"--relation must be JSON: {exc}", 2)
                if not isinstance(parsed, dict):
                    return _fail("--relation must be a JSON object", 2)
                relations.append(parsed)
        result = h.induce(strategy_id=args.strategy, all_=args.all,
                          dry_run=args.dry_run,
                          force=args.force, notes=notes, verify=verify,
                          family=getattr(args, "family", None),
                          cell=getattr(args, "cell", None),
                          relations=relations)
        return _emit(result, _summarize_induce(result, args))
    finally:
        h.close()


def _summarize_induce(result: Dict[str, Any], args) -> str:
    if getattr(args, "relation", None):
        return _summarize_relations(result)
    created = [r for r in result.get("results", []) if r.get("created")]
    revised = [r for r in result.get("results", []) if r.get("updated")]
    skipped = [r for r in result.get("results", []) if r.get("skipped")]
    parts = []
    if created:
        parts.append(f"Created {len(created)} entries: "
                     + ", ".join(r["created"] for r in created))
    if revised:
        parts.append(f"Refreshed {len(revised)} entries: "
                     + ", ".join(r["updated"] for r in revised))
    if skipped:
        parts.append(f"Skipped {len(skipped)}: {skipped[0]['skipped']}")
    scope = []
    if getattr(args, "family", None):
        scope.append(f"family={args.family}")
    if getattr(args, "cell", None):
        scope.append(f"cell={args.cell}")
    if scope:
        parts.append("Scoped to " + ", ".join(scope) + ".")
    transitions = [rev for rev in (result.get("revisions") or [])
                   if rev.get("transitions")]
    for rev in transitions:
        parts.append(f"Revision on {rev['entry_id']} ({rev['strategy_id']}): "
                     f"{', '.join(rev['transitions'])} — from "
                     f"{rev['forward']['n_predictions']} frozen check(s), "
                     f"{rev['forward']['consecutive_misses']} consecutive "
                     "miss(es)")
    if not parts:
        parts.append("Nothing to induce.")
    action = result.get("action") or {}
    if action.get("action_id"):
        parts.append(f"Induction action {action['action_id']} recorded in "
                     f"the maintenance scope (business result: "
                     f"{action.get('business_result', 'unknown')}).")
    return " ".join(parts)


def _summarize_relations(result: Dict[str, Any]) -> str:
    """Summary for a relation submission (``induce --relation``)."""
    parts: List[str] = []
    saved = result.get("saved", 0)
    published = result.get("published", 0)
    if saved:
        parts.append(f"Saved {saved} relation claim(s)")
    if published:
        parts.append(f"{published} published as knowledge (own verification + "
                     ">=2 independent tasks)")
    for item in result.get("relations") or []:
        entry_id = item.get("created_entry") or item.get("saved")
        if not entry_id:
            if item.get("skipped"):
                parts.append(f"Refused: {item['skipped']}")
            continue
        publication = item.get("publication") or {}
        claim = item.get("claim") or {}
        text = str(claim.get("text") or "")[:80]
        state = publication.get("state", "unverified")
        if publication.get("published"):
            parts.append(f"Entry {entry_id}: {state} claim published — "
                         f"{text}")
        else:
            reasons = "; ".join(publication.get("reasons") or []) or \
                "not yet publishable"
            parts.append(f"Entry {entry_id}: claim saved as {state} but NOT "
                         f"published ({reasons})")
    if not parts:
        parts.append("No claim was submitted.")
    return " ".join(parts)


def cmd_induction_candidates(args) -> int:
    """Scan the bank for induction candidate bundles (no model call).

    This is the EVIDENCE PACKAGE generator: it freezes, per candidate, the
    experience scope, the task targeting and the baseline that
    ``predict-capability --bundle`` consumes. It makes no model call and
    changes no knowledge."""
    h = _harness(args)
    try:
        bundles = h.induction_candidates()
        if not bundles:
            return _emit(
                {"count": 0, "candidates": []},
                "No induction candidates with sufficient evidence. A claim "
                "needs >=2 supporting executions from >=2 distinct tasks — "
                "keep solving and recording.")
        return _emit(
            {"count": len(bundles), "candidates": bundles},
            f"Found {len(bundles)} induction candidate(s). Each bundle "
            "carries its own frozen evidence scope; pass one to "
            "`orx predict-capability --bundle` to price the operation "
            "before committing to it.")
    finally:
        h.close()


def cmd_induction_material(args) -> int:
    """Organize the READABLE material for an offline induction step.

    This is the framework's half of semantic induction: gather the methods
    that were actually used, what changed, the two sides of a comparison and
    what followed, so YOU (the outer agent) can read it and form a claim in
    your own words. It calls no model, writes nothing, and does not invent a
    technique: a candidate whose evidence reports no method is reported as
    ``insufficient``, and the honest response is to record how the work was
    really done."""
    h = _harness(args)
    try:
        result = h.induction_material(
            bundle_id=args.bundle, pattern=args.pattern,
            strategy_id=args.strategy)
        if not result["count"]:
            return _emit(result,
                         "No induction material: no candidate currently has "
                         "supporting evidence. (A claim needs >=2 executions "
                         "from >=2 distinct tasks.)")
        parts = []
        for item in result["material"]:
            state = item["material_state"]
            label = item.get("pattern") or item["kind"]
            line = (f"{item['bundle_id']} [{label}] "
                    f"{item['strategy_id'] or '(unnamed)'} "
                    f"n={item['n_supporting']} tasks={len(item['tasks'])} "
                    f"material={state['state']}")
            if state["state"] == "insufficient":
                line += f" ({state['reason']})"
            parts.append(line)
        return _emit(result, " | ".join(parts)
                     + ". Read the material, compare the evidence, and "
                       "submit a claim with `orx induce --relation` — the "
                       "framework checks what you submit, it does not "
                       "write the technique for you.")
    finally:
        h.close()


def cmd_inspect(args) -> int:
    h = _harness(args)
    try:
        result = h.inspect(bank=args.bank, task_id=args.task,
                           strategy_id=args.strategy, status=args.status,
                           episode_id=getattr(args, "episode", None),
                           evaluation_id=getattr(args, "evaluation", None),
                           prediction_id=getattr(args, "prediction", None))
        if args.bank == "retention":
            online = result["online"]
            archive = result["archive"]
            return _emit(
                result,
                f"Online: {online['n_window_episodes']} window episode(s), "
                f"{online['n_closeouts_total']} closed total, "
                f"{online['n_contract_predictions']} prediction(s), "
                f"{online['n_prediction_contexts']} context(s). "
                f"Archive: {len(archive['files'])} file(s), "
                f"{archive['total_bytes']} bytes. Three separate scopes — "
                "see `policy`.")
        if args.bank == "predictions" and getattr(args, "prediction", None):
            # A KEY lookup is reported as one, so the caller sees the
            # generation it actually resolved instead of a log summary.
            found = result.get("prediction") or {}
            return _emit(
                result,
                f"Prediction {args.prediction}: generation "
                f"{found.get('kind')}. A single prediction is read directly "
                "by key — no listing of the prediction logs.")
        if args.bank == "capability":
            if getattr(args, "prediction", None):
                single = result["prediction"]
                return _emit(
                    result,
                    f"Prediction {args.prediction}: status "
                    f"{single['status']}, operation "
                    f"{single['candidate_operation']['operation_type']}.")
            return _emit(
                result,
                f"{result['n_predictions']} capability prediction(s): "
                f"{result['n_fact_bound']} with a bound maintenance fact, "
                f"{result['n_effect_verified']} with a VERIFIED effect. A "
                "bound fact says the operation happened; only a verified "
                "effect says real later performance moved."
                + _online_gains_line(result.get("online_gains")))
        if args.bank == "evaluations":
            if getattr(args, "evaluation", None):
                return _emit(result, f"Evaluation {args.evaluation}: state "
                                     f"{result['evaluation'].get('state')}.")
            evaluations = result["evaluations"]
            n_evaluated = sum(1 for e in evaluations
                              if e.get("state") == "evaluated")
            return _emit(
                result,
                f"{len(evaluations)} evaluation(s) stored ({n_evaluated} "
                "evaluated). Excluded evaluations are neither hits nor "
                "misses; pending ones wait for their scope to end.")
        count = result["count"]
        noun = {"experience": "records", "strategic": "entries",
                "archive": "cards", "actions": "actions",
                "snapshots": "snapshots",
                "predictions": "predictions", "texts": "task texts"}[args.bank]
        if count == 1:
            noun = noun[:-1]           # "1 card", not "1 cards"
        detail = ""
        if args.bank == "strategic":
            detail = (" Entry track records show n_predictions, hit_rate, "
                      "calibration_error, and consecutive_misses. Suspect / "
                      "dormant statuses are the retirement CANDIDATES "
                      "(`--status suspect|dormant`); retiring them is "
                      "explicit (`orx retire`).")
        elif args.bank == "archive":
            detail = (" A card blocks re-inducing that same pattern; lift it "
                      "with `induce --force` when the environment has "
                      "genuinely drifted.")
        elif args.bank == "actions":
            detail = (" Actions carry lifecycle status (running = begun, "
                      "not ended) and, for induce, a business result "
                      "separate from the lifecycle status.")
        elif args.bank == "predictions":
            if getattr(args, "prediction", None):
                # A KEY lookup: report the one record and its generation.
                found = result.get("prediction") or {}
                detail = (f" generation={found.get('kind')}. "
                          "A single prediction is read directly, not by "
                          "listing the logs.")
            else:
                detail = (f" Legacy M2: "
                          f"{len(result['legacy_predictions'])}, "
                          f"strategy-outcome (wm-so/1): "
                          f"{len(result['strategy_predictions'])}, "
                          f"capability-evolution (wm-ce/1): "
                          f"{len(result['capability_predictions'])}. Separate "
                          "logs, separate questions — only the wm-so/1 channel "
                          "feeds `orx calibration`.")
        elif args.bank == "texts":
            detail = (" These are the retrieval SOURCE documents, not a "
                      "knowledge bank: they feed the embedding index and "
                      "are the look-up entry point for records that carry "
                      "no vector.")
        return _emit(result, f"{count} {noun} in {args.bank} bank"
                             + (f" (status={args.status})" if args.status else "")
                             + "." + detail)
    finally:
        h.close()


def cmd_snapshot(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        snap = h.snapshot(task, episode_id=args.episode)
        result = {"snapshot": snap.to_dict(),
                  "snapshot_id": snap.snapshot_id}
        layers = snap.coverage.get("knowledge_layers", {})
        return _emit(result,
                     f"Snapshot {snap.snapshot_id} frozen for task "
                     f"{snap.task_id} (episode={snap.episode_id}). Knowledge "
                     f"layers: {len(layers.get('verified', []))} verified, "
                     f"{len(layers.get('legacy_unknown', []))} legacy, "
                     f"{len(layers.get('unverified', []))} unverified. "
                     "Later bank writes cannot change this snapshot.")
    finally:
        h.close()


def cmd_action(args) -> int:
    h = _harness(args)
    try:
        if args.amend_cost:
            if not args.cost:
                return _fail("--amend-cost requires --cost")
            cost = _load_json_arg(args.cost)
            if not isinstance(cost, dict):
                return _fail("--cost must be a JSON object")
            result = h.amend_action_cost(args.amend_cost, **cost)
            return _emit(result,
                         f"Amended cost of action {args.amend_cost} "
                         "(replace semantics, idempotent).")
        if not args.report:
            return _fail("action requires --report TYPE or --amend-cost "
                         "ACTION_ID")
        if not args.task:
            return _fail("--report requires --task")
        task = _load_json_arg(args.task)
        params = _load_json_arg(args.params) if args.params else None
        outcome = _load_json_arg(args.outcome) if args.outcome else None
        cost = None
        if args.cost:
            cost = _load_json_arg(args.cost)
            if not isinstance(cost, dict):
                return _fail("--cost must be a JSON object")
        result = h.report_action(
            args.report, task, episode_id=args.episode, params=params,
            outcome=outcome, status=args.status, cost=cost)
        return _emit(result,
                     f"Recorded {args.report} action {result['action_id']} "
                     f"(source=agent_reported, status={result['status']}). "
                     "The library did not execute this action — the report "
                     "is your statement, labelled as such.")
    finally:
        h.close()


def cmd_budget(args) -> int:
    h = _harness(args)
    try:
        if args.declare:
            budget = _parse_dimension_pairs(args.declare)
            h.declare_budget(args.task, budget, episode_id=args.episode)
        result = h.budget_view(args.task, episode_id=args.episode)
        cons = result["consumption"]
        summary = (f"Budget status for task {args.task}: "
                   f"{result['status']}. {result['status_note']} "
                   f"Consumption: {cons['n_recorded']} recorded + "
                   f"{cons['n_staged']} staged execution(s), "
                   f"{len(cons['action_costs'])} own-cost action(s).")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_predict_strategy(args) -> int:
    """Predict ONE candidate's consequences under the wm-so/1 protocol."""
    h = _harness(args)
    try:
        from or_harness.world_model.prediction import ActionSpec
        task = _load_json_arg(args.task)
        raw = _load_json_arg(args.candidate)
        if isinstance(raw, dict) and "measurement_scope" in raw:
            candidate: Any = ActionSpec.from_dict(raw)
        else:
            candidate = raw
        context = None
        if args.context:
            context = h.get_prediction_context(args.context)
            if context is None:
                return _fail(f"unknown context_id {args.context!r}")
        cir = _load_json_arg(args.cir) if args.cir else None
        prediction = h.predict_strategy_outcome(
            task, candidate, args.episode, context=context, cir=cir)
        result = {"prediction": prediction.to_dict(),
                  "prediction_id": prediction.prediction_id}
        info = prediction.trace.model_info
        failure = info.get("failure")
        if failure:
            # A failed prediction must SAY WHY in its own output: the
            # difference between "the model was truncated" and "the model
            # answered with the wrong shape" decides whether to raise the
            # output budget or fix the request.
            result["failure"] = failure
        if info.get("effective_parameters"):
            result["effective_parameters"] = info["effective_parameters"]
        if prediction.status == "contract_only" \
                and not prediction.provider_configured:
            return _fail(f"prediction not enabled: "
                         f"{(prediction.notes or ['no provider'])[0]}", 2)
        parts = [f"Strategy-outcome prediction {prediction.prediction_id} "
                 f"({prediction.status}) for "
                 f"{prediction.candidate.strategy_id or '(unnamed)'}: "]
        if failure:
            parts.append(f"FAILED [{failure.get('kind')}] "
                         f"{failure.get('detail') or ''} "
                         f"(finish_reason="
                         f"{failure.get('finish_reason')!r}); ")
        if prediction.benefit is not None:
            b = prediction.benefit
            parts.append(f"G={b.value} ({b.kind}/{b.metric}"
                         + (f", baseline {b.baseline.kind}" if b.baseline
                            else "") + "); ")
        if prediction.cost is not None and prediction.cost.expected:
            dims = prediction.cost.expected.measured_dims()
            parts.append("c={" + ", ".join(
                f"{d}: {getattr(prediction.cost.expected, d)}"
                for d in sorted(dims)) + "}; ")
        if prediction.risk is not None and prediction.risk.events:
            parts.append(f"L={len(prediction.risk.events)} event(s); ")
        if prediction.uncertainty is not None:
            parts.append("uncertainty "
                         f"(source={prediction.uncertainty.source}); ")
        if prediction.trace.unsupported_fields:
            parts.append(f"not predicted: "
                         f"{', '.join(sorted(prediction.trace.unsupported_fields))}; ")
        parts.append("This is a hypothesis — execute the candidate "
                     "yourself, then `orx bind-strategy --prediction "
                     f"{prediction.prediction_id} --action <id>`.")
        return _emit(result, "".join(parts))
    finally:
        h.close()


def _attribution_summary(info: Dict[str, Any]) -> str:
    """A short, honest sentence about which comparisons a binding's
    unconfirmed/mismatched fields actually block.

    Reuses the SAME attribution table the close-out uses, so the CLI never
    tells the agent something the evaluation layer will contradict.
    """
    from or_harness.world_model.attribution import (
        binding_attribution,
        blocked_dimensions,
    )
    attribution = binding_attribution(
        info.get("binding_mismatch") or {},
        info.get("binding_unknown") or {},
        info.get("method_observed"))
    blocked = blocked_dimensions(attribution)
    if not blocked:
        return ""
    parts = []
    for dim, entries in sorted(blocked.items()):
        fields = sorted({str(e.get("field")) for e in entries})
        parts.append(f"{dim} (blocked by {', '.join(fields)})")
    return ("this blocks only " + "; ".join(parts) +
            " — the rest of the sample is still scored, and the real spend "
            "is kept wherever the comparison does not need that field.")


def cmd_bind_strategy(args) -> int:
    """Bind a strategy-outcome prediction to the real action that ran."""
    h = _harness(args)
    try:
        prediction = h.bind_strategy_outcome(args.prediction, args.action)
        info = prediction.trace.model_info
        if info.get("binding_mismatch"):
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} WITH MISMATCH: "
                       f"{info['binding_mismatch']}. The prediction is NOT "
                       "comparable — the executed action differs from the "
                       "predicted candidate.")
        elif info.get("binding_unknown"):
            unknown = info["binding_unknown"]
            blocked = _attribution_summary(info)
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} with UNCONFIRMED identity fields: "
                       f"{sorted(unknown)}. A missing receipt is a caveat, "
                       "not a discard: "
                       + (blocked if blocked else
                          "nothing is blocked — the whole sample is scored, "
                          "with the unconfirmed fields reported."))
            summary += " (`orx close-episode` reports each part.)"
        elif prediction.trace.comparable:
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} and COMPARABLE: the real execution "
                       "may be scored against it at episode close-out "
                       "(`orx close-episode`).")
        else:
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action}, not yet comparable: "
                       f"{prediction.trace.not_comparable_reasons}.")
        return _emit({"prediction": prediction.to_dict()}, summary)
    finally:
        h.close()


def cmd_close_episode(args) -> int:
    """Close one episode: evaluate bound predictions, publish calibration.

    Reads what was recorded — no solver run, no model call, no induction.
    Idempotent: re-closing returns the stored record and counts nothing
    twice."""
    h = _harness(args)
    try:
        result = h.close_episode(
            args.task, args.episode, terminal_state=args.terminal,
            finish_action_id=args.finish_action,
            min_calibration_samples=args.min_samples
            if args.min_samples is not None else None)
        closeout = result["closeout"]
        if closeout is None:
            # The close-out was REFUSED (there is still a running action).
            # `closeout` is None by contract in that case, so the pending
            # state must be handled BEFORE anything reads its fields —
            # reaching for `closeout['terminal_state']` crashed with a
            # TypeError and reported "unexpected TypeError" instead of the
            # actionable reason.
            unfinished = result.get("unfinished_actions") or []
            return _fail(
                f"Episode {args.task}/{args.episode} is still OPEN: "
                f"{len(unfinished)} action(s) are running "
                f"({', '.join(unfinished)}). {result.get('note', '')}",
                2)
        if result.get("already_closed"):
            return _emit(result,
                         f"Episode {args.task}/{args.episode} was already "
                         f"closed (state {closeout['terminal_state']}); "
                         "the stored record stands — nothing re-counted.")
        n_evaluated = sum(1 for e in result.get("evaluations", [])
                          if e.get("state") == "evaluated")
        n_excluded = sum(1 for e in result.get("evaluations", [])
                         if e.get("state") == "excluded")
        parts = [f"Episode {args.task}/{args.episode} closed "
                 f"({closeout['terminal_state']}).",
                 f"{n_evaluated} prediction(s) evaluated, {n_excluded} "
                 "excluded (excluded is neither hit nor miss)."]
        if closeout.get("unfinished_actions"):
            parts.append(f"{len(closeout['unfinished_actions'])} unfinished "
                         "action(s) reported; their predictions stay "
                         "pending.")
        calibration = result.get("calibration_summary") or {}
        groups = calibration.get("groups") or {}
        if groups:
            thin = [name for name, g in groups.items()
                    if g.get("basis") == "insufficient_evidence"]
            parts.append(f"Calibration published over {len(groups)} "
                         f"group(s)"
                         + (f"; {len(thin)} below the sample minimum "
                            "(insufficient_evidence, no figure claimed)"
                            if thin else "")
                         + ".")
        else:
            parts.append("No calibration samples yet "
                         "(insufficient_evidence until closed episodes "
                         "accumulate).")
        # The close-out ends the episode; it does NOT certify the answer.
        # Reporting the task-check coverage here is what keeps those two
        # facts apart in the caller's head.
        checks = result.get("task_checks") or {}
        if checks.get("n_executions"):
            verdicts = checks.get("verdicts") or {}
            rendered = ", ".join(f"{k}={v}"
                                 for k, v in sorted(verdicts.items()))
            parts.append(
                f"Task-result checks: {rendered or 'none'} "
                f"({checks.get('unchecked', 0)} of "
                f"{checks['n_executions']} unchecked)."
                + (" " + checks["note"] if checks.get("note") else ""))
        return _emit(result, " ".join(parts))
    finally:
        h.close()


def cmd_calibration(args) -> int:
    """Read the published experience-calibration summary (read-only)."""
    h = _harness(args)
    try:
        result = h.calibration_summary(
            min_samples=args.min_samples
            if args.min_samples is not None else None,
            rebuild=bool(getattr(args, "rebuild", False)))
        groups = result.get("groups") or {}
        if not groups:
            summary = ("No calibration samples in the current window: "
                       "reliability is unknown (insufficient_evidence), "
                       "never guessed.")
            if result.get("missing"):
                summary += " " + str(result["missing"])
        else:
            parts = []
            for name, group in sorted(groups.items()):
                if group.get("basis") == "insufficient_evidence":
                    parts.append(f"{name}: {group['n_samples']} sample(s), "
                                 "insufficient evidence")
                else:
                    parts.append(
                        f"{name}: n={group['n_samples']} "
                        f"({group['n_distinct_episodes']} episode(s)), "
                        f"benefit MAE="
                        f"{group.get('mean_benefit_abs_error')}, "
                        f"signed="
                        f"{group.get('mean_benefit_signed_error')}, "
                        f"interval coverage="
                        f"{group.get('interval_coverage')}")
            summary = ("Strategy-outcome experience calibration: "
                       + "; ".join(parts)
                       + ". A measured record of past closed episodes — "
                         "not a promise that future predictions improve.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_archive_calibration(args) -> int:
    """Move OUT-OF-WINDOW episode detail to the archive (retention)."""
    h = _harness(args)
    try:
        result = h.archive_calibration(dry_run=bool(args.dry_run))
        removed = result.get("removed") or {}
        n_records = result.get("n_records", 0)
        if result.get("dry_run"):
            summary = (f"Dry run: {n_records} record(s) would be archived "
                       f"({len(result.get('held_for_late_check') or [])} "
                       "episode(s) held online for a possible late check).")
        else:
            summary = (
                f"Archived {n_records} record(s) "
                f"({removed.get('evaluations', 0)} evaluation(s), "
                f"{removed.get('contract_predictions', 0)} prediction(s), "
                f"{removed.get('prediction_contexts', 0)} context(s)); "
                f"{len(result.get('held_for_late_check') or [])} episode(s) "
                "held online for a possible late check. Registry rows stay "
                "online, so a repeated close remains idempotent.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_enforce_window(args) -> int:
    """Bound the Execution Evidence Bank to a recent window of episodes."""
    h = _harness(args)
    try:
        result = h.enforce_window(window_episodes=args.window_episodes,
                                  open_grace_days=args.open_grace_days,
                                  dry_run=bool(args.dry_run))
        n_evicted = result.get("n_evicted", 0)
        n_unclosed = result.get("n_evicted_unclosed", 0)
        if result.get("dry_run"):
            summary = (f"Dry run: {n_evicted} closed episode(s) and "
                       f"{n_unclosed} aged-out unclosed episode(s) would be "
                       "evicted; nothing was written.")
        else:
            summary = (
                f"Evicted {n_evicted} closed episode(s) "
                f"({result.get('removed_executions', 0)} execution(s)) and "
                f"{n_unclosed} aged-out unclosed episode(s). "
                f"{result.get('n_protected', 0)} episode(s) were protected "
                "(calibration window / late-check grace / young unclosed). "
                "An eviction is a source reference expiring, never a "
                "refutation or a withdrawal.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_migrate_relations(args) -> int:
    """One-way migration of legacy relations into standalone claim entries."""
    h = _harness(args)
    try:
        result = h.migrate_relations(dry_run=bool(args.dry_run))
        if result.get("dry_run"):
            summary = (f"Dry run: {result.get('relations_found', 0)} legacy "
                       f"relation(s) across "
                       f"{result.get('entries_with_relations', 0)} entry(ies) "
                       "would be migrated; nothing was written.")
        elif result.get("relations_found"):
            summary = (
                f"Migrated {result.get('created_entries', 0)} claim "
                f"entr(ies) from {result.get('relations_found', 0)} legacy "
                f"relation(s). Each relation became its OWN entry with its "
                "verification copied verbatim (a migration never widens what "
                "was verified).")
        else:
            summary = "Nothing to migrate: no legacy relation lists found."
        return _emit(result, summary)
    finally:
        h.close()


def cmd_predict_capability(args) -> int:
    """Predict what an offline learning operation would change (M5, wm-ce/1).

    This is the SLOW-time-scale prediction: it answers what future task
    performance would change, at what learning cost and risk, if the
    candidate operation were executed. It does NOT execute it, does not
    touch the Strategic Bank, and does not claim any effect is verified.
    """
    h = _harness(args)
    try:
        operation = _load_json_arg(args.operation)
        if not isinstance(operation, dict) or not operation.get("operation_type"):
            return _fail("--operation needs a JSON object with at least "
                         "'operation_type' (induce|revise|reverify|retire)")
        task = _load_json_arg(args.task) if args.task else None
        bundle = _load_json_arg(args.bundle) if args.bundle else None
        if bundle is not None and not isinstance(bundle, dict):
            return _fail("--bundle must be a JSON object (a candidate "
                         "bundle from `orx assess-induction "
                         "--candidates-only`)")
        budget = _load_json_arg(args.budget) if args.budget else None
        prediction = h.predict_capability_evolution(
            operation, task=task, bundle=bundle,
            horizon=args.horizon or "", horizon_tasks=args.horizon_tasks,
            maintenance_budget=budget, timeout_s=args.timeout,
            task_id=args.task_id or "", episode_id=args.episode)
        out = prediction.to_dict()
        if prediction.status != "valid":
            return _emit(out,
                         f"Capability-evolution prediction {prediction.status}"
                         f": no usable expected change was produced, so this "
                         f"is a recorded NON-prediction, not a capability "
                         f"forecast. {' '.join(prediction.notes)}")
        changes = "; ".join(
            f"{c.metric} {c.direction}"
            + (f" {c.value}{c.unit}" if c.value is not None else "")
            + ("" if c.is_improvement is None
               else (" (improvement)" if c.is_improvement
                     else " (DEGRADATION)"))
            for c in prediction.expected_changes)
        return _emit(out,
                     f"Capability-evolution prediction {prediction.status}: "
                     f"{len(prediction.expected_changes)} expected change(s) "
                     f"[{changes}] over {prediction.horizon!r}. The operation "
                     "has NOT run and no effect is verified — binding the "
                     "fact and judging the effect are separate later steps.")
    finally:
        h.close()


def cmd_compare_capability(args) -> int:
    """Compare capability predictions and recommend one, or defer (M5).

    Read-only with respect to knowledge: no operation runs, no model is
    called, the Strategic Bank is untouched.
    """
    h = _harness(args)
    try:
        ids = [p for p in (args.predictions or "").split(",") if p]
        if not ids:
            return _fail("--predictions needs one or more comma-separated "
                         "prediction ids")
        result = h.compare_capability_evolution(
            ids, horizon_tasks=args.horizon_tasks,
            require_quality_nondegradation=not args.allow_quality_loss)
        if result["recommendation"] == "accept":
            summary = (f"Recommendation: ACCEPT "
                       f"{result['selected_prediction_id']} "
                       f"({result['selected_operation_type']}) — "
                       f"{result['basis']}. Nothing has run: the operation "
                       "requires an explicit accept.")
        else:
            summary = (f"Recommendation: "
                       f"{result['recommendation'].upper()} — "
                       f"{result['basis']}")
            if result.get("incomparable"):
                summary += (f" {len(result['incomparable'])} candidate(s) "
                            "could not be ranked; they are listed for you "
                            "to choose between.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_accept_capability(args) -> int:
    """EXPLICITLY accept a recommendation and run the real operation (M5).

    The only M5 command that changes knowledge. The induction runs on the
    prediction's OWN frozen experience scope — never widened by re-reading
    the current bank.
    """
    h = _harness(args)
    try:
        recommendation = _load_json_arg(args.recommendation)
        if not isinstance(recommendation, dict):
            return _fail("--recommendation must be a JSON object from "
                         "`orx compare-capability`")
        verify = _load_json_arg(args.verify) if args.verify else None
        notes = [args.note] if args.note else None
        result = h.accept_capability_operation(
            recommendation, prediction_id=args.prediction,
            verify=verify, notes=notes, force=args.force)
        delta = ((result["operation_result"] or {}).get("knowledge_delta")
                 or {})
        created = delta.get("entries_created") or []
        binding = result.get("maintenance_binding") or {}
        if binding.get("bound"):
            bind_note = ("The real maintenance fact was bound "
                         "automatically"
                         + (" (already bound: nothing was re-counted)"
                            if binding.get("already_bound") else "")
                         + ".")
        else:
            bind_note = ("WARNING: the maintenance fact could NOT be bound "
                         f"automatically ({binding.get('reason') or 'unknown'}"
                         "); bind it with `orx bind-capability`.")
        return _emit(result,
                     f"Accepted {result['capability_prediction_id']}: the "
                     f"operation ran on {len(result['execution_ids'])} "
                     f"scoped execution(s), {len(created)} entr(y/ies) "
                     f"created. {bind_note} Do NOT call `induce`/`retire` "
                     "again for this decision — the operation has already "
                     "run. The capability EFFECT stays unverified until "
                     "qualified later tasks produce real results "
                     "(`orx evaluate-capability`).")
    finally:
        h.close()


def cmd_reject_capability(args) -> int:
    """Explicitly decline or defer a capability recommendation (M5).

    Records the decision with its reason and changes NO knowledge.
    """
    h = _harness(args)
    try:
        recommendation = _load_json_arg(args.recommendation)
        if not isinstance(recommendation, dict):
            return _fail("--recommendation must be a JSON object from "
                         "`orx compare-capability`")
        result = h.reject_capability_operation(
            recommendation, reason=args.reason,
            prediction_id=args.prediction)
        return _emit(result,
                     "Declined: no operation ran and no knowledge changed. "
                     "A declined recommendation is not a wrong prediction — "
                     "it was never given a chance to come true.")
    finally:
        h.close()


def cmd_bind_capability(args) -> int:
    """Stage 1: bind the REAL maintenance fact to a prediction (M5).

    Answers only "did the operation happen, and what changed?" — never
    "did the harness get stronger". Idempotent by prediction.
    """
    h = _harness(args)
    try:
        result = h.bind_capability_maintenance(
            args.prediction, adoption_action_id=args.adoption_action)
        binding = result["binding"]
        if result.get("already_bound"):
            return _emit(result,
                         f"Prediction {args.prediction} was already bound: "
                         "the stored fact stands (nothing re-counted).")
        if result.get("state") == "not_adopted":
            return _emit(result,
                         f"Nothing to bind: no operation was accepted for "
                         f"prediction {args.prediction}. A deferred or "
                         "declined recommendation leaves the knowledge "
                         "alone.")
        delta = binding.get("knowledge_delta") or {}
        return _emit(result,
                     f"Maintenance fact bound: operation "
                     f"{binding['operation_type']} "
                     f"{'changed knowledge' if binding['changed'] else 'produced NO knowledge change'}"
                     f" (created={len(delta.get('entries_created') or [])}, "
                     f"scope_consistent={binding['scope_consistent']}). "
                     "This is stage 1 of 2: the capability EFFECT is still "
                     "unverified.")
    finally:
        h.close()


def cmd_evaluate_capability(args) -> int:
    """Stage 2: judge a prediction against REAL later-task results (M5).

    Reads the M4 evaluations of CLOSED episodes of tasks outside the
    prediction's own experience scope, or a pre-arranged paired
    comparison. An unreached horizon stays pending and re-evaluable.
    """
    h = _harness(args)
    try:
        if args.paired:
            paired = _load_json_arg(args.paired)
            if not isinstance(paired, dict):
                return _fail("--paired must be a JSON object with 'metric', "
                             "'reference_value' and 'treated_value'")
            result = h.record_capability_paired_evaluation(
                args.prediction,
                metric=paired.get("metric"),
                reference_value=paired.get("reference_value"),
                treated_value=paired.get("treated_value"),
                unit=paired.get("unit", ""),
                source=paired.get("source", "external_paired_evaluation"),
                reference_task_ids=paired.get("reference_task_ids"),
                note=paired.get("note", ""))
            return _emit(result,
                         "Paired reference recorded: the change can now be "
                         "attributed to the operation rather than merely "
                         "described.")
        task_ids = [t for t in (args.tasks or "").split(",") if t] or None
        result = h.evaluate_capability_effect(
            args.prediction, task_ids=task_ids,
            require_paired_reference=not args.allow_descriptive)
        evaluation = result["evaluation"]
        state = evaluation["state"]
        if result.get("already_evaluated"):
            return _emit(result,
                         f"Prediction {args.prediction} was already "
                         f"evaluated ({state}): the stored verdict stands "
                         "and the sample was not counted again.")
        explanations = {
            "pending": ("the horizon is unmet: no qualified LATER task has "
                        "produced a closed episode yet. Pending is "
                        "re-evaluable, never frozen."),
            "observed_improvement": ("the real evidence supports the "
                                     "predicted improvement."),
            "observed_degradation": ("the real evidence shows a "
                                     "DEGRADATION: recorded, never dropped."),
            "no_change": ("the observed change contradicts the prediction: "
                          "a refuted prediction, which is not a "
                          "degradation."),
            "inconclusive": ("the change was observed but cannot be "
                             "attributed: no comparable reference exists."),
            "insufficient_evidence": ("nothing could be compared against a "
                                      "real observation."),
            "not_evaluable": ("no expected change names a metric this build "
                              "can observe."),
        }
        return _emit(result,
                     f"Capability effect {state}: "
                     f"{explanations.get(state, '')} "
                     f"effect_verified={evaluation['effect_verified']}.")
    finally:
        h.close()


def cmd_plan_next(args) -> int:
    h = _harness(args)
    try:
        from or_harness.world_model.prediction import ActionSpec
        task = _load_json_arg(args.task)
        if getattr(args, "horizon", None) not in (None, 1):
            # Refused EXPLICITLY rather than accepted-and-ignored: there is
            # one horizon, and an agent that asked for a two-step rollout
            # must be told what replaced it.
            return _fail(
                f"--horizon {args.horizon} is not supported: planning "
                "compares macro strategy candidates once (horizon=1), then "
                "you re-plan from the REAL observation of the chosen step. "
                "Execute the step and plan again with what actually "
                "happened — no imagined multi-step rollout is built")
        candidates = None
        if args.candidates:
            raw = _load_json_arg(args.candidates)
            if not isinstance(raw, list):
                return _fail("--candidates must be a JSON list of "
                             "ActionSpec objects")
            candidates = [ActionSpec.from_dict(c) for c in raw]
        limits = {"horizon": args.horizon,
                  "max_model_calls": args.max_calls,
                  "time_budget_s": getattr(args, "time_budget", None)}
        limits = {k: v for k, v in limits.items() if v is not None}
        if getattr(args, "delta", None) is not None:
            limits["delta"] = args.delta
        if getattr(args, "benefit_kind", None):
            limits["benefit_kind"] = args.benefit_kind
        if getattr(args, "benefit_metric", None):
            limits["benefit_metric"] = args.benefit_metric
        if getattr(args, "benefit_unit", None):
            limits["benefit_unit"] = args.benefit_unit
        plan = h.plan_next(task, episode_id=args.episode,
                           candidates=candidates, limits=limits)
        result = {"plan": plan, "decision_action_id": plan.get(
            "decision_action_id")}
        # The parameters that were REALLY in force, so a caller never has to
        # infer them from today's configuration.
        result["effective_parameters"] = {
            "max_model_calls": plan.get("max_model_calls"),
            "time_budget_s": plan.get("time_budget_s"),
        }
        if plan.get("protocol"):
            result["protocol"] = plan["protocol"]
        status = plan.get("status")
        if status in ("disabled", "no_candidates", "fallback"):
            return _emit(result, f"plan_next returned status={status}: "
                                 f"{plan.get('truncation_reason')}")
        # Every compared candidate carries the prediction it was scored
        # with. REUSE that id — never predict the chosen candidate again.
        compared = plan.get("candidates") or []
        parts = [f"Plan {plan['plan_id']}: {len(compared)} candidate(s) "
                 f"compared from context "
                 f"{plan.get('prediction_context_id')} "
                 f"(calls={plan.get('model_calls_made')}, "
                 f"planning_cost={plan.get('planning_cost')})."]
        if compared:
            by_strategy = {
                (c.get("action_spec") or {}).get("strategy_id"): c
                for c in compared
                if (c.get("action_spec") or {}).get("strategy_id")}
            ids = ", ".join(f"{sid}={c.get('prediction_id')}"
                            for sid, c in by_strategy.items())
            parts.append(f"Predictions: {ids}.")
        suggested = plan.get("suggested")
        if suggested:
            step = (f"Suggested first step: {suggested['action_type']}"
                    + (f" {suggested.get('strategy_id')}"
                       if suggested.get("strategy_id") else "")
                    + f" — {plan.get('suggestion_basis')}")
            # The prediction id comes from the SUGGESTED candidate itself,
            # matched on the FULL candidate identity — including solver and
            # config. Matching on strategy_id + action_type alone named the
            # FIRST candidate with that strategy, so "same strategy, HiGHS
            # vs Gurobi" bound the wrong solver's prediction and the later
            # identity check (correctly) rejected it as a mismatch: a real
            # sample lost to a display bug.
            suggested_id = _matching_candidate_prediction_id(
                compared, suggested)
            if suggested_id:
                step += (f". Accept by ID: `orx choose-next --decision "
                         f"{plan.get('decision_action_id')} --prediction "
                         f"{suggested_id}` — then run the step with the SAME "
                         "prediction (do NOT call predict-strategy again): "
                         f"`orx execute --task ... --prediction "
                         f"{suggested_id} --code solve.py --workspace ws` "
                         "(the strategy and solver come from the candidate).")
            else:
                step += (f". Accept with `orx choose-next --decision "
                         f"{plan.get('decision_action_id')} --chosen ...`.")
            parts.append(step)
        else:
            parts.append(plan.get("suggestion_basis")
                         or "No suggestion (see candidates).")
        return _emit(result, " ".join(parts))
    finally:
        h.close()


def cmd_choose_next(args) -> int:
    h = _harness(args)
    try:
        from or_harness.world_model.prediction import ActionSpec
        chosen = None
        if args.chosen:
            chosen = ActionSpec.from_dict(_load_json_arg(args.chosen))
        try:
            result = h.choose_next(args.decision, chosen=chosen,
                                   prediction_id=getattr(args, "prediction",
                                                         None),
                                   rejected=args.rejected,
                                   deviation_note=args.note)
        except ValueError as exc:
            # A precondition failure (a prediction that is not part of this
            # decision, or --chosen and --prediction disagreeing): refused
            # with the reason, before any selection is written.
            return _fail(str(exc))
        if result.get("rejected"):
            summary = (f"Decision {args.decision} recorded as REJECTED. "
                       "X.selected_plan is NOT written.")
        else:
            summary = (f"Choice recorded for decision {args.decision}: "
                       f"{(result.get('selected') or {}).get('action_type')} "
                       + (f"{(result.get('selected') or {}).get('strategy_id')}"
                          if (result.get('selected') or {}).get('strategy_id')
                          else "")
                       + ". X.selected_plan updated; execute the step with "
                         "`orx execute --prediction <the chosen prediction>`, "
                         "then record and close the episode.")
        if result.get("deviation"):
            summary += " (deviation from the suggestion recorded)"
        return _emit(result, summary)
    finally:
        h.close()


def cmd_retire(args) -> int:
    h = _harness(args)
    try:
        result = h.retire(args.entry, reason=args.reason)
        return _emit(result, f"Entry {args.entry} retired to the cold archive. "
                             "It left the hot store; its cold-archive card now "
                             "blocks re-inducing the same pattern (lift it with "
                             "`induce --force` if the environment has genuinely "
                             "drifted).")
    finally:
        h.close()


def cmd_exclude_execution(args) -> int:
    h = _harness(args)
    try:
        result = h.exclude_execution(
            args.execution, reason=args.reason,
            superseded_by=getattr(args, "superseded_by", None))
        parts = [f"Execution {args.execution} EXCLUDED from the evidence set "
                 f"(reason: {args.reason}). The fact is preserved for audit — "
                 "it is not deleted — but it no longer counts in statistics, "
                 "induction, triggers or vector recall."]
        if result.get("superseded_by"):
            parts.append(f"It is superseded by {result['superseded_by']}.")
        idx = result.get("index") or {}
        if idx:
            parts.append(f"Index: {idx.get('removed', 0)} vector removed.")
        parts.append("Derived layers pick this up at the next `orx induce`.")
        return _emit(result, " ".join(parts))
    finally:
        h.close()


def cmd_restore_execution(args) -> int:
    h = _harness(args)
    try:
        result = h.restore_execution(args.execution, reason=args.reason)
        return _emit(result, f"Execution {args.execution} RESTORED to the "
                             "evidence set (reason: "
                             f"{args.reason}). Run `orx rebuild-index "
                             "--layer execution` to re-index it, and "
                             "`orx induce` to refresh the derived layers.")
    finally:
        h.close()


def cmd_rebuild_index(args) -> int:
    """Explicit retrieval-index maintenance (first build / repair / model
    change). The only path allowed to embed in bulk — recall never writes."""
    h = _harness(args)
    try:
        result = h.rebuild_index(layer=args.layer, dry_run=args.dry_run)
        if h.embedding_index is None and not args.dry_run:
            return _fail("no embedding backend configured: set "
                         "OR_EMBEDDING_BASE_URL, OR_EMBEDDING_MODEL, and "
                         "OR_EMBEDDING_API_KEY before rebuilding the index")
        if args.dry_run:
            layers = result.get("layers") or {}
            detail = "; ".join(
                f"{name}: {info['would_index']} document(s)"
                + (f", {info['unindexable']} unindexable"
                   if info.get("unindexable") else "")
                for name, info in layers.items())
            return _emit(result, f"Index rebuild dry run — {detail}. No "
                                 "embedding call was made and no index file "
                                 "was written or modified.")
        layers = result.get("layers") or {}
        detail = "; ".join(
            f"{name}: {info.get('items')} item(s) under "
            f"{info.get('model_id')}"
            + (f", {info['unindexable']} unindexable"
               if info.get("unindexable") else "")
            for name, info in layers.items())
        return _emit(result, f"Retrieval index rebuilt — {detail}. The index "
                             "is derived data: it can be rebuilt at any time "
                             "and never changes a fact or an entry.")
    finally:
        h.close()


def cmd_doctor(args) -> int:
    h = _harness(args)
    try:
        result = h.doctor()
        avail = [s["name"] for s in result["solvers"] if s["available"]]
        missing = [s["name"] for s in result["solvers"] if not s["available"]]
        stale = result.get("index_health", {}).get("stale_group_index", 0)
        index_note = (f" {stale} row(s) carry a stale group_l1 index (read "
                      "correctly; the index is derived from family)."
                      if stale else "")
        retrieval = result.get("retrieval_index") or {}
        if not retrieval.get("configured"):
            retrieval_note = (" Text retrieval: no embedding backend "
                              "configured — recall uses profile matching "
                              "only (reported as degraded).")
        else:
            pieces = []
            for name, info in (retrieval.get("layers") or {}).items():
                if not info.get("exists"):
                    pieces.append(f"{name}: index missing "
                                  "(`orx rebuild-index`)")
                elif not info.get("usable"):
                    pieces.append(f"{name}: unusable — {info.get('reason')}")
                else:
                    pieces.append(
                        f"{name}: {info.get('count')} item(s) vs "
                        f"{info.get('documents')} current document(s), "
                        f"{info.get('stale')} stale, "
                        f"{info.get('missing')} missing, "
                        f"{info.get('orphaned')} orphaned")
            retrieval_note = (f" Text retrieval ({retrieval.get('backend')}): "
                              + "; ".join(pieces) + ".")
        return _emit(result,
                     f"Home: {result['home']}. Available solvers: "
                     f"{', '.join(avail) or 'none'}. Missing: "
                     f"{', '.join(missing) or 'none'}. Memory: "
                     f"{result['memory']}." + index_note + retrieval_note)
    finally:
        h.close()


def cmd_contract(args) -> int:
    """Build or read a unified world-model contract (no model call).

    Two modes:

    - ``--payload``: READ a stored prediction payload (current contract,
      legacy unversioned, or an explicitly unsupported version). Never
      writes, never calls a model.
    - ``--kind``: BUILD a contract object and return it. This is a
      schema-level construction: with no prediction service configured the
      object is returned as ``contract_only`` and says so — the contract is
      implemented, the service is not attached.
    """
    from or_harness.world_model.contracts import (
        BaselineStatement,
        BenefitEstimate,
        CandidateRef,
        ExpectedChange,
        ExpectedCost,
        ExperienceScope,
        LearningOperation,
        RiskStatement,
        TaskTargeting,
        VerificationCondition,
    )
    h = _harness(args)
    try:
        if args.payload:
            payload = _load_json_arg(args.payload)
            if not isinstance(payload, dict):
                return _fail("--payload must be a JSON object")
            result = h.read_prediction_payload(payload)
            version = result["contract_version"]
            if not result.get("supported"):
                return _fail(f"Unsupported contract version: {version}. "
                             f"{result.get('error', '')} The payload was NOT "
                             "parsed — an unknown version is never guessed at.")
            if result.get("legacy"):
                view = result["legacy_view"]
                gaps = view.get("gaps") or []
                return _emit(result,
                             f"Legacy unversioned payload read through the "
                             f"legacy view: {len(view['mapped_fields'])} "
                             f"field(s) mapped, {len(gaps)} gap(s) where the "
                             "old payload recorded nothing (left absent, not "
                             "invented). Capability evidence is INDIRECT "
                             "only; no capability increment is derived.")
            return _emit(result,
                         f"Contract payload {version} loaded "
                         f"({result['contract'].get('prediction_type')}).")
        kind = args.kind
        if kind == "strategy_outcome":
            task = _load_json_arg(args.task) if args.task else None
            if task is None:
                return _fail("--kind strategy_outcome requires --task")
            spec = _load_json_arg(args.spec)
            if not isinstance(spec, dict):
                return _fail("--spec must be a JSON object")
            benefit = None
            raw = getattr(args, "benefit", None)
            if raw:
                data = _load_json_arg(raw)
                if not isinstance(data, dict) or not data.get("metric"):
                    return _fail("--benefit needs a JSON object with at "
                                 "least 'metric' (and 'kind', 'value')")
                benefit = BenefitEstimate.from_dict(data)
            candidate = _candidate_from_spec(spec)
            prediction = h.build_strategy_outcome_contract(
                task, candidate,
                episode_id=getattr(args, "episode", None),
                benefit=benefit,
                cost=ExpectedCost.from_dict(_load_json_arg(args.cost))
                if getattr(args, "cost", None) else None,
                risk=RiskStatement.from_dict(_load_json_arg(args.risk))
                if getattr(args, "risk", None) else None)
            out = prediction.to_dict()
            return _emit(out,
                         f"Strategy outcome contract {prediction.status} "
                         f"(scope={prediction.scope}, "
                         f"comparable={prediction.trace.comparable}, "
                         f"provider_configured="
                         f"{prediction.provider_configured}, "
                         f"prediction_made={prediction.prediction_made}). "
                         + " ".join(prediction.notes))
        # capability_evolution
        task = _load_json_arg(args.task) if args.task else None
        op_raw = _load_json_arg(args.operation) if args.operation else None
        if op_raw is not None and not isinstance(op_raw, dict):
            return _fail("--operation must be a JSON object")
        operation = (LearningOperation.from_dict(op_raw) if op_raw
                     else LearningOperation(
                         operation_type="induce",
                         description=args.operation_desc or ""))
        if args.operation_desc:
            operation.description = args.operation_desc
        scope = None
        if args.scope:
            scope_data = _load_json_arg(args.scope)
            if not isinstance(scope_data, dict):
                return _fail("--scope must be a JSON object")
            scope = ExperienceScope.from_dict(scope_data)
        targeting = None
        if args.targeting:
            targeting_data = _load_json_arg(args.targeting)
            if not isinstance(targeting_data, dict):
                return _fail("--targeting must be a JSON object")
            targeting = TaskTargeting.from_dict(targeting_data)
        changes = []
        if args.expected_change:
            raw = _load_json_arg(args.expected_change)
            raw = raw if isinstance(raw, list) else [raw]
            changes = [ExpectedChange.from_dict(c) for c in raw]
        conditions = []
        for raw in (args.verification or []):
            data = _load_json_arg(raw)
            conditions.append(VerificationCondition.from_dict(data))
        prediction = h.build_capability_evolution_contract(
            task, operation, experience_scope=scope,
            task_targeting=targeting,
            baseline=(BaselineStatement.from_dict(
                _load_json_arg(args.baseline)) if args.baseline else None),
            horizon=args.horizon or "",
            horizon_tasks=args.horizon_tasks,
            expected_changes=changes,
            verification_conditions=conditions)
        out = prediction.to_dict()
        return _emit(out,
                     f"Capability evolution contract {prediction.status} "
                     f"(operation={operation.operation_type}, "
                     f"provider_configured={prediction.provider_configured}, "
                     f"service_implemented={prediction.service_implemented}, "
                     f"service_available={prediction.service_available}, "
                     f"prediction_made={prediction.prediction_made}). "
                     + " ".join(prediction.notes))
    finally:
        h.close()


def cmd_context(args) -> int:
    """Build or read the frozen prediction input context (world-model phase 2).

    Two modes:

    - ``--context-id``: READ a stored context back. Never re-runs retrieval,
      never calls a model, never executes anything.
    - ``--task``: BUILD a context — one snapshot, one recall over the two
      existing channels, the joint problem representation, the capability
      evidence and the external constraints, frozen and persisted.
    """
    h = _harness(args)
    try:
        if args.context_id:
            ctx = h.get_prediction_context(args.context_id)
            if ctx is None:
                return _fail(f"unknown context_id {args.context_id!r}")
            return _emit(ctx.to_dict(), _summarize_context(ctx, stored=True))
        if not args.task:
            return _fail("orx context requires --task (to build) or "
                         "--context-id (to read)")
        task = _load_json_arg(args.task)
        if not isinstance(task, dict):
            return _fail("--task must be a JSON object")
        cir = _load_json_arg(args.cir) if args.cir else None
        math = _load_json_arg(args.math) if args.math else None
        if math is not None and not isinstance(math, dict):
            return _fail("--math must be a JSON object")
        ctx = h.build_prediction_context(
            task, args.episode, top=args.top, cir=cir,
            math=math, include_unverified=args.include_unverified,
            persist=not args.no_persist)
        return _emit(ctx.to_dict(), _summarize_context(ctx))
    finally:
        h.close()


def _summarize_context(ctx, *, stored: bool = False) -> str:
    """Agent-readable summary of a prediction input context."""
    joint = ctx.joint
    parts = [
        (f"Stored context {ctx.context_id}" if stored
         else f"Built frozen context {ctx.context_id}") + ":",
        f"task version {ctx.task_digest},",
        f"snapshot {ctx.snapshot_id or '(none)'}.",
        f"Problem representation: text "
        f"{'present' if joint.text.strip() else 'ABSENT'},"
        f" CIR {'present' if joint.cir_present else 'absent'},"
        f" model {'present' if joint.has_model else 'not written yet'}.",
        "Math attributes: "
        + (", ".join(f"{k}={v}" for k, v in joint.math.to_dict().items()
                     if k in ("integrality", "linearity", "objective_kind")
                     and v) or "none established")
        + ("; unknown: " + ", ".join(joint.unknowns)
           if joint.unknowns else "; none unknown") + ".",
        f"Retrieval: channels {ctx.retrieval.channels_run or 'none'} ran,"
        f" {ctx.n_evidence} piece(s) of evidence carried"
        f" ({ctx.evidence_classes() or 'none'});"
        f" {ctx.retrieval.deduplication.get('duplicates_collapsed', 0)}"
        " duplicate hit(s) collapsed by identity.",
        f"Capability evidence sources: "
        + ", ".join(f"{k}={v.get('status')}"
                    for k, v in (ctx.capability.get("sources") or {}).items())
        + " (evidence ABOUT H, never a score).",
        f"{len(ctx.degraded)} degraded part(s), {len(ctx.missing)} missing "
        "part(s) reported.",
        "Building this context made NO model call, ran NO solver and "
        "induced nothing.",
    ]
    if ctx.degraded:
        parts.append("Degraded: " + "; ".join(
            f"{d['part']}: {d['reason']}" for d in ctx.degraded))
    if not ctx.retrieval.hits:
        parts.append("No evidence was carried: with a degraded channel this "
                     "is NOT the same as 'nothing comparable exists'.")
    return " ".join(parts)


def _candidate_from_spec(spec: Dict[str, Any]):
    """Build a ``CandidateRef`` from either candidate or legacy spec JSON.

    A payload naming ``measurement_scope`` is a LEGACY ``ActionSpec``: it is
    routed through ``CandidateRef.from_action_spec`` so its execution
    configuration and budget hint survive, and so an unmappable legacy scope
    (``task``) is refused instead of silently shrunk to one attempt.
    """
    from or_harness.world_model.contracts import CandidateRef
    from or_harness.world_model.prediction import ActionSpec
    if "measurement_scope" in spec or "budget_hint" in spec:
        return CandidateRef.from_action_spec(ActionSpec.from_dict(spec))
    return CandidateRef.from_dict(spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orx",
        description="OR-Harness: strategy-learning capability layer for OR agents. "
                    "stdout is always a single JSON {result, summary}.")
    parser.add_argument("--home", default=None,
                        help="memory directory (default: $OR_HARNESS_HOME or "
                             "./or_harness_home)")
    parser.add_argument("--alpha", type=float, default=1.0, help="quality weight")
    parser.add_argument("--beta", type=float, default=1.0, help="cost weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="risk weight")
    parser.add_argument("--cost-weights", default=None,
                        help="per-dimension cost weights, e.g. "
                             "'llm_tokens=1.0,retries=2.0'")
    parser.add_argument("--delta", type=float, default=0.0,
                        help="weight of the predicted knowledge term "
                             "(U = alpha*Q - beta*C - gamma*R + delta*K). "
                             "Default 0.0: an unknown knowledge value is "
                             "never rewarded, so a positive delta is a "
                             "deliberate experimental choice")
    parser.add_argument("--prediction-mode", default="h-x-b-value",
                        choices=list(PREDICTION_MODES),
                        help="what the world model predicts: 'x-b-only' "
                             "X/B only, 'h-x-b' also H (recorded, not "
                             "scored), 'h-x-b-value' also H with the "
                             "knowledge term live. Default h-x-b-value, "
                             "whose delta is still 0 unless set")
    parser.add_argument("--world-model", default=None, metavar="URL::MODEL",
                        help="world-model provider for outcome predictions: "
                             "OpenAI-compatible base URL and model name "
                             "(api key from $OR_WM_API_KEY). Omitted = "
                             "predictions return not_configured; no other "
                             "command is affected")
    parser.add_argument("--wm-timeout", type=float, default=None,
                        metavar="S",
                        help="world-model call timeout in seconds "
                             "(precedence: this flag > $OR_WM_TIMEOUT > "
                             "the adapter default 300). This is the "
                             "maximum time ONE blocking socket operation "
                             "may stall, not a hard deadline on the whole "
                             "request; the caller's own budget check bounds "
                             "the call in aggregate")
    parser.add_argument("--wm-max-tokens", type=int, default=None,
                        metavar="N",
                        help="output token budget for a world-model call "
                             "(precedence: this flag > $OR_WM_MAX_TOKENS > "
                             "the adapter default 2048). A larger budget "
                             "costs more and is a deployment decision: set "
                             "it explicitly rather than assuming every model "
                             "needs it")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile",
                       help="the single analysis entry: CIR validation + "
                            "modeling guidance + problem profile + "
                            "derivation report (a task without a 'model' "
                            "field is a normal state)")
    p.add_argument("--task", required=True, help="task JSON literal or file")
    p.add_argument("--cir", default=None,
                   help="optional CIR JSON literal/file (overrides the "
                        "task's 'coupling' field)")
    p.add_argument("--allow-empty-cir", action="store_true",
                   help="accept a CIR that carries no entities/decisions/"
                        "constraints/relations (default: reject it, so a "
                        "malformed or nested CIR is never mistaken for an "
                        "empty one). $OR_CIR_STRICT=0 downgrades all CIR "
                        "policy checks")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("recall",
                       help="recall REAL accumulated experience for a task "
                            "(no built-in candidate menu: an empty result "
                            "means memory holds nothing here)")
    p.add_argument("--task", required=True)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--candidate", action="append", default=None,
                   metavar="STRATEGY_ID",
                   help="a method YOU are considering (repeatable). "
                        "Restricts the result to these ids and reports the "
                        "ones with no memory under "
                        "`candidates_without_evidence`. The framework "
                        "never invents evidence for them and never blocks "
                        "their execution")
    p.add_argument("--memory-mode", default="cost-aware",
                   choices=["none", "cases", "strategic", "cost-aware"])
    p.add_argument("--include-unverified", action="store_true",
                   help="offline/inspection view: also surface UNPUBLISHED "
                        "candidates (unverified / insufficient_evidence) "
                        "in both channels. Default excludes them: a candidate "
                        "is held by the framework, not knowledge.")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser(
        "predict-cost",
        help="pre-execution COST expectation snapshot for one "
             "(task, strategy), supported by past evidence; pass it back to "
             "`orx record` via --prediction. This is NOT a world-model "
             "prediction: no provider is called and no benefit/risk is "
             "forecast")
    p.add_argument("--task", required=True)
    p.add_argument("--strategy", required=True)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser(
        "execute",
        help="sandbox-execute a solve script. Pass --prediction to run a "
             "FROZEN candidate: the strategy/solver/episode are taken from "
             "it and the association is established before the run")
    p.add_argument("--task", required=True)
    p.add_argument("--strategy", default=None,
                   help="the strategy that runs. Required for an unpredicted "
                        "attempt; when --prediction is given it is taken from "
                        "the candidate (pass it only to state the same value "
                        "— a conflicting value is refused BEFORE the run)")
    p.add_argument("--code", required=True, help="path to solve.py")
    p.add_argument("--workspace", required=True)
    p.add_argument("--solver", default=None,
                   help="the concrete solver. Required for an unpredicted "
                        "attempt; when --prediction is given it is taken from "
                        "the candidate (a conflicting value is refused before "
                        "the run)")
    p.add_argument("--episode", default=None,
                   help="episode id for the unified action record "
                        "(budget/progress scoping); taken from the candidate "
                        "when --prediction is given")
    p.add_argument("--prediction", default=None, metavar="PREDICTION_ID",
                   help="the strategy-outcome prediction (wm-so/1) this "
                        "attempt is testing, e.g. the id `plan-next` scored "
                        "the candidate with. The association is established "
                        "BEFORE the run: the frozen candidate supplies "
                        "strategy/solver/episode (do not re-type them), the "
                        "problem identity and the claim on the prediction are "
                        "checked, and a conflicting explicit argument is "
                        "refused before anything executes. After the run the "
                        "real result and the observed configuration are "
                        "added to the same association. Omit it for an "
                        "unpredicted attempt; `orx bind-strategy` can bind a "
                        "manual action later")
    p.add_argument("--method", default=None, metavar="JSON",
                   help="the method THIS attempt intends to use, in your own "
                        "words: '{\"name\": ..., \"steps\": [...]}'. When "
                        "--prediction is given the candidate's method is "
                        "used; pass --method only for an unpredicted "
                        "attempt. It is recorded as the PLAN "
                        "(method_planned) — the method that actually ran "
                        "comes solely from the script's result.json "
                        "'method_performed' receipt, never from this plan")
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser("record", help="append an ExecutionRecord to the Experience Bank")
    p.add_argument("--execution", default=None,
                   help="execution JSON literal/file (or the JSON printed by execute)")
    p.add_argument("--record-file", default=None)
    p.add_argument("--from-staged", default=None, metavar="EXECUTION_ID",
                   help="record a staged execution verbatim (the honest path "
                        "for backfilling a failed attempt — no re-typing)")
    p.add_argument("--discard-staged", default=None, metavar="EXECUTION_ID",
                   help="explicitly discard a staged execution")
    p.add_argument("--override", default=None,
                   help="cost backfill, e.g. 'llm_tokens=1840,tool_calls=9' "
                        "(or 'retries=1' to declare this attempt is itself "
                        "a retry)")
    p.add_argument("--override-mode", default="replace",
                   choices=["replace", "increment"],
                   help="backfill accounting: 'replace' (default, idempotent — "
                        "the value IS the measurement, re-applying never "
                        "double-counts) or 'increment' (an additional measured "
                        "amount within the record's scope)")
    p.add_argument("--override-source", dest="override_source",
                   default="agent_estimate",
                   choices=["provider_usage", "agent_observed",
                            "agent_estimate"],
                   help="where the backfilled number came from (recorded per "
                        "dimension): 'provider_usage' (the provider "
                        "reported it), 'agent_observed' (you READ it off a "
                        "real report) or 'agent_estimate' (default — a "
                        "hand-typed number with no stated source: shown, but "
                        "never used as a measured truth or a calibration "
                        "actual). Say provider_usage/agent_observed when the "
                        "number really came from a report")
    p.add_argument("--force", action="store_true", dest="override_force",
                   help="allow an override to overwrite a dimension the "
                        "FRAMEWORK measured (latency_s / solver_runtime_s); "
                        "refused by default so a real observation is never "
                        "silently rewritten")
    p.add_argument("--usage-file", dest="usage_file", default=None,
                   metavar="JSON|PATH",
                   help="the outer framework's ATTEMPT-level usage report "
                        "(its hook/log/storage output: prompt_tokens, "
                        "completion_tokens, reasoning_tokens, cached_tokens, "
                        "model, and optionally a per-call 'calls' list; a "
                        "versioned or-host-usage/1 report is read with its "
                        "own measured whitelist and provenance). "
                        "Recorded with the full token口径 and marked as a "
                        "host observation — the REAL spend, not an estimate")
    p.add_argument("--usage-host", dest="usage_host", default=None,
                   metavar="HOST",
                   help="locate the attempt's usage report through a NAMED "
                        "host adapter (e.g. 'openclaw') instead of passing "
                        "the file: the adapter looks under "
                        "<home>/host_usage/. When no report is found the "
                        "dimensions stay UNKNOWN (never zero) and the "
                        "summary says so")
    p.add_argument("--usage-source", dest="usage_source", default=None,
                   help="which host produced --usage-file (e.g. 'openclaw' "
                        "'hermes'); recorded so the figure is traceable")
    p.add_argument("--prediction", default=None,
                   help="pre-execution cost prediction snapshot (the JSON "
                        "printed by `orx predict`) actually used for this "
                        "attempt; feedback compares against it, never a "
                        "post-hoc estimate")
    p.add_argument("--method", default=None, metavar="JSON",
                   help="for a record assembled OUTSIDE `orx execute`: the "
                        "PLAN this attempt intended, '{\"name\": ..., "
                        "\"steps\": [...]}'. Recorded as method_planned; it "
                        "is never promoted to the performed method")
    p.add_argument("--method-actual", default=None, metavar="JSON",
                   help="for a record assembled OUTSIDE `orx execute`: the "
                        "method the harness DECLARES was actually performed, "
                        "'{\"name\": ..., \"steps\": [...]}'. Use it only "
                        "when you really observed the processing; the "
                        "integrity of the evidence depends on honesty here. "
                        "A value the record already carries is never "
                        "overwritten")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser(
        "check-task",
        help="check whether an execution's ANSWER satisfies the original "
             "task (not merely that the solver solved its own model)",
        epilog=("Run this between `execute` and `record` (or later, on an "
                "already-recorded execution — a late correction is a real "
                "event). The executor's verdict covers the solver's own "
                "model; this covers the TASK. Declare only the bases that "
                "apply: a check that cannot run reports `insufficient`, "
                "which is neither a pass nor a failure. `failed` never "
                "deletes the attempt — the cost is real and the failure is "
                "raw material — it stops the answer from counting as a "
                "success sample."))
    p.add_argument("execution_id", metavar="EXECUTION_ID")
    p.add_argument("--check", default=None, metavar="JSON",
                   help=("the check basis, e.g. '{\"reference_objective\": "
                         "10755, \"integer\": {\"variables\": [\"x1\", "
                         "\"x2\"]}}'. Supported keys: reference_objective "
                         "(+tolerance), reference_status, integer "
                         "({variables?, tolerance?}), recompute_objective "
                         "({coefficients, constant?, tolerance?}), "
                         "semantic_probe ([{path, equals|min|max|in}]), "
                         "intent (relaxation|intermediate). Omitted: the "
                         "verdict is `insufficient` — never a default pass"))
    p.add_argument("--episode", default=None,
                   help="episode to scope the check to (defaults to the "
                        "episode of the action that produced the execution)")
    p.set_defaults(func=cmd_check_task)

    p = sub.add_parser(
        "amend-cost",
        help="backfill cost dimensions of an ALREADY-RECORDED execution",
        epilog=("Amends the fact in place; nothing is re-run and nothing is "
                "appended. Use this to close a cost gap reported by "
                "`record`/`induce` (cost_completeness.missing / "
                "cost_claim_withheld): an unmeasured dimension supports no "
                "cost claim until every supporting record measures it."))
    p.add_argument("execution_id", metavar="EXECUTION_ID")
    p.add_argument("--override", required=True,
                   help="dimension=value pairs, e.g. 'llm_tokens=1840,"
                        "tool_calls=9'")
    p.add_argument("--mode", default="replace",
                   choices=["replace", "increment"],
                   help="'replace' (default, idempotent — the value IS the "
                        "measurement) or 'increment' (an additional measured "
                        "amount)")
    p.add_argument("--source", default="agent_estimate",
                   choices=["provider_usage", "agent_observed",
                            "agent_estimate"],
                   help="where the number came from: 'provider_usage' (the "
                        "provider reported it), 'agent_observed' (you READ "
                        "it off a real report) or 'agent_estimate' (default "
                        "— a hand-typed number with no stated source: shown "
                        "but never used as a measured truth). Recorded per "
                        "dimension so a real measurement is never "
                        "indistinguishable from an estimate")
    p.add_argument("--usage-file", dest="usage_file", default=None,
                   metavar="JSON|PATH",
                   help="a HOST attempt-level usage report to apply instead "
                        "of hand-typed overrides (the late-backfill channel: "
                        "the host's hook/log output, or '@path'). Its "
                        "llm_tokens is recorded with source=provider_usage; "
                        "a versioned or-host-usage/1 report can also carry "
                        "tool_calls (agent_observed) in the same report")
    p.add_argument("--usage-host", dest="usage_host", default=None,
                   metavar="HOST",
                   help="locate the report through a NAMED host adapter "
                        "(e.g. 'openclaw') instead of passing the file; "
                        "when no report is found the command fails with "
                        "the reason and the dimensions stay UNKNOWN")
    p.add_argument("--usage-source", dest="usage_source", default=None,
                   help="which host/hook reported --usage-file (e.g. "
                        "openclaw, hermes); used for the provenance record")
    p.add_argument("--force", action="store_true",
                   dest="amend_force",
                   help="allow overwriting a dimension the FRAMEWORK measured "
                        "(latency_s / solver_runtime_s / a provider-reported "
                        "dimension); refused by default so a real observation "
                        "is never silently rewritten")
    p.set_defaults(func=cmd_amend_cost)

    p = sub.add_parser(
        "induce", help="consolidate facts into strategic entries",
        epilog=("Creating an entry requires >=2 supporting executions from "
                ">=2 distinct task_ids: repeating one task is repetition, not "
                "reproduction. Refreshing an existing entry is never gated. "
                "Publishing is separate: an entry becomes strategic knowledge "
                "only once --verify renders 'verified'."))
    p.add_argument("--strategy", default=None)
    p.add_argument("--all", action="store_true")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="lift the cold-archive veto: the card blocking this "
                        "pattern is REMOVED, then induction proceeds "
                        "(reserve it for genuine environment drift — the "
                        "lift is one-time, no need to repeat it)")
    p.add_argument("--note", action="append", default=None, metavar="TEXT",
                   help="applicability note to attach to the entries this call "
                        "creates/refreshes (free text, kept for the reader, "
                        "never scored; repeatable)")
    p.add_argument("--verify", default=None, metavar="JSON",
                   help="admission check for the candidate this call forms: "
                        "{\"purpose\": \"rule|repair|cost_saving\", \"claim\": "
                        "TEXT, \"check\": {\"reference_status\" | "
                        "\"reference_objective\" | \"semantic_probe\" | "
                        "\"dimension\"+\"quality_floor\"}, \"executions\": "
                        "[...], \"supporting\": [...]}. The framework evaluates "
                        "those checks on the executions you supply; "
                        "feasibility alone is not a check, and without a "
                        "verdict the entry is NOT published. The check applies "
                        "ONLY to the targets this call selects — narrow with "
                        "--family/--cell so one claim never overwrites another "
                        "unit's verdict")
    p.add_argument("--family", default=None,
                   help="restrict induction to ONE family (implies --all "
                        "scope: naming a unit is an explicit selection)")
    p.add_argument("--cell", default=None, metavar="GROUP_KEY",
                   help="restrict induction to ONE structural cell (the full "
                        "group_key token, e.g. 'family=routing|rc[..]|..'); "
                        "implies its family and --all scope")
    p.add_argument("--relation", action="append", default=None, metavar="JSON",
                   help="submit a STRUCTURED relation claim (repeatable). "
                        "{\"claim\": TEXT, \"evidence\": [{\"execution_id\": "
                        "ID, \"role\": ROLE}, ...], \"subject\": NAME?, "
                        "\"conditions\": {\"predicates\": {...}, \"note\": "
                        "TEXT}?, \"check\": {\"assertions\": [...]}?, "
                        "\"kind\": NAME?}. The role names the part each "
                        "referenced execution plays in THIS claim "
                        "(dropped/preserved, before/after, strategy_a, ...); "
                        "tasks/family/strategy ids are DERIVED from the "
                        "recorded facts. An optional free-form 'subject' "
                        "(e.g. 'principle:cross_period_state') carries a "
                        "claim that belongs to no strategy id. The "
                        "relation is published on its OWN verification plus "
                        ">=2 independent tasks — it never publishes the host "
                        "entry's statistical claim")
    p.set_defaults(func=cmd_induce)

    p = sub.add_parser("inspect", help="query the memory layers")
    p.add_argument("--bank", default="experience",
                   choices=["experience", "strategic", "archive",
                            "actions", "snapshots", "predictions", "texts",
                            "evaluations", "retention", "capability"])
    p.add_argument("--task", default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--episode", default=None,
                   help="filter actions by episode id")
    p.add_argument("--evaluation", default=None, metavar="EVALUATION_ID",
                   help="with --bank evaluations: read ONE evaluation "
                        "instead of listing")
    p.add_argument("--prediction", default=None, metavar="PREDICTION_ID",
                   help="with --bank predictions: read ONE prediction "
                        "across the legacy / strategy-outcome / capability "
                        "generations; with --bank capability: read ONE "
                        "prediction's fact binding + effect evaluation")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("snapshot",
                       help="freeze and persist the current belief state "
                            "for a task (world-model M1)")
    p.add_argument("--task", required=True)
    p.add_argument("--episode", default=None)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("action",
                       help="report an action the outer agent performed "
                            "(model/select_strategy/verify/"
                            "finish_task) or amend an action's cost")
    p.add_argument("--report", default=None, metavar="TYPE",
                   help="action type to report (agent_reported)")
    p.add_argument("--task", default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--params", default=None, help="JSON params")
    p.add_argument("--outcome", default=None, help="JSON outcome")
    p.add_argument("--status", default="completed",
                   choices=["completed", "failed", "cancelled", "timeout"])
    p.add_argument("--cost", default=None,
                   help="JSON cost dimensions, e.g. '{\"llm_tokens\": 500}'")
    p.add_argument("--amend-cost", default=None, metavar="ACTION_ID",
                   help="amend an existing action's cost (replace, "
                        "idempotent); pair with --cost")
    p.set_defaults(func=cmd_action)

    p = sub.add_parser("budget",
                       help="budget view for a task/episode (all action "
                            "costs, staged executions included)")
    p.add_argument("--task", required=True)
    p.add_argument("--episode", default=None)
    p.add_argument("--declare", default=None,
                   help="declare a budget, e.g. 'llm_tokens=50000,"
                        "solver_runtime_s=600'")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser(
        "predict-strategy",
        help="predict ONE candidate strategy's benefit/cost/risk/"
             "uncertainty under the wm-so/1 protocol, from a frozen "
             "prediction context (world-model M3)")
    p.add_argument("--task", required=True,
                   help="task JSON (literal or file)")
    p.add_argument("--candidate", required=True,
                   help="candidate JSON (literal or file): a CandidateRef "
                        "{action_type, strategy_id, solver, config, scope} "
                        "or a legacy ActionSpec (measurement_scope/"
                        "budget_hint preserved verbatim)")
    p.add_argument("--episode", default=None)
    p.add_argument("--context", default=None, metavar="CTX_ID",
                   help="reuse a FROZEN prediction input context built by "
                        "`orx context` (identity verified, including the "
                        "effective input version); default builds a fresh "
                        "one for this call")
    p.add_argument("--cir", default=None,
                   help="CIR JSON (literal or file): the effective problem "
                        "input. Pass the SAME CIR you built the context "
                        "with when reusing one")
    p.set_defaults(func=cmd_predict_strategy)

    p = sub.add_parser(
        "bind-strategy",
        help="bind a strategy-outcome prediction to the real action that "
             "ran (identity checked: task/episode/strategy/solver/config; "
             "a mismatch is recorded, never scored; an UNKNOWN identity "
             "field is recorded separately and never counts as a match)")
    p.add_argument("--prediction", required=True)
    p.add_argument("--action", required=True)
    p.set_defaults(func=cmd_bind_strategy)

    p = sub.add_parser(
        "close-episode",
        help="close ONE episode (world-model M4): evaluate its bound "
             "strategy-outcome predictions against their real outcomes and "
             "publish the experience calibration. Reads what was recorded "
             "— no solver, no model call, no induction. Idempotent")
    p.add_argument("--task", required=True)
    p.add_argument("--episode", default=None)
    p.add_argument("--terminal", default="completed",
                   choices=["completed", "failed", "aborted",
                            "budget_exhausted"],
                   help="the honest terminal state (only 'completed' "
                        "claims success)")
    p.add_argument("--finish-action", default=None, metavar="ACTION_ID",
                   help="the finish_task action that ended the episode, "
                        "when one was recorded")
    p.add_argument("--min-samples", type=int, default=None,
                   help="minimum resolved samples before a calibration "
                        "group reports a figure (below it: "
                        "insufficient_evidence)")
    p.set_defaults(func=cmd_close_episode)

    p = sub.add_parser(
        "calibration",
        help="read the published strategy-outcome experience-calibration "
             "summary (window of closed episodes; read-only unless "
             "--rebuild)")
    p.add_argument("--min-samples", type=int, default=None,
                   help="override the minimum-sample threshold for this "
                        "read (the effective value is recorded on the "
                        "summary)")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild and republish the summary from the current "
                        "window instead of reading the published one (the "
                        "explicit migration/repair path)")
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser(
        "archive-calibration",
        help="move OUT-OF-WINDOW episode detail (evaluations, predictions, "
             "contexts) to the archive; registry rows stay online so a "
             "repeated close remains idempotent")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be archived without moving it")
    p.set_defaults(func=cmd_archive_calibration)

    p = sub.add_parser(
        "enforce-window",
        help="bound the Execution Evidence Bank to a recent window of "
             "COMPLETE episodes (oldest first; whole episodes only). "
             "Calibration-window episodes, episodes awaiting a late check, "
             "and young unclosed episodes are never evicted. An eviction is "
             "a source reference expiring, never a refutation")
    p.add_argument("--window-episodes", type=int, default=None,
                   help="how many complete episodes to retain "
                        "(default 800; $OR_EVIDENCE_WINDOW_EPISODES)")
    p.add_argument("--open-grace-days", type=float, default=None,
                   help="how long an UNCLOSED episode's executions may stay "
                        "before the whole episode is evicted (default 30; "
                        "$OR_EVIDENCE_OPEN_GRACE_DAYS)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be evicted without writing")
    p.set_defaults(func=cmd_enforce_window)

    p = sub.add_parser(
        "migrate-relations",
        help="ONE-WAY migration: turn legacy entry-level relation lists into "
             "standalone claim entries (one claim per entry). Idempotent; "
             "each relation's verification is copied verbatim (a migration "
             "never widens what was verified)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be migrated without writing")
    p.set_defaults(func=cmd_migrate_relations)

    p = sub.add_parser(
        "predict-capability",
        help="predict what an OFFLINE learning operation would change in "
             "future task performance, under the wm-ce/1 protocol "
             "(world-model M5). Does NOT execute the operation and claims "
             "no effect is verified")
    p.add_argument("--operation", required=True,
                   help="learning operation JSON (literal or file): "
                        "{operation_type: induce|revise|reverify|retire, "
                        "strategy_id, description, scope, config}")
    p.add_argument("--task", default=None,
                   help="task JSON (literal or file) scoping the capability "
                        "evidence; optional")
    p.add_argument("--bundle", default=None,
                   help="candidate bundle JSON from `orx assess-induction "
                        "--candidates-only` (literal or file). Supplies the "
                        "experience scope, the task targeting and the frozen "
                        "baseline, and its REAL evidence content is sent to "
                        "the provider")
    p.add_argument("--horizon", default=None,
                   help="what window the change is claimed over (free text); "
                        "defaults to 'the next matching tasks'")
    p.add_argument("--horizon-tasks", type=int, default=None,
                   help="the number of tasks the horizon covers, when "
                        "declared. Without it a per-task saving is NEVER "
                        "extrapolated into a total")
    p.add_argument("--budget", default=None,
                   help="maintenance budget JSON (literal or file): an "
                        "execution LIMIT passed to the provider as context, "
                        "never a prediction")
    p.add_argument("--task-id", default=None,
                   help="task id to file the prediction under (for later "
                        "queries)")
    p.add_argument("--episode", default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(func=cmd_predict_capability)

    p = sub.add_parser(
        "compare-capability",
        help="compare frozen capability predictions and recommend one, or "
             "defer (M5). Read-only: no operation runs, no knowledge "
             "changes")
    p.add_argument("--predictions", required=True,
                   help="comma-separated prediction ids to compare")
    p.add_argument("--horizon-tasks", type=int, default=None,
                   help="the task count the comparison totals over; without "
                        "it figures stay PER TASK")
    p.add_argument("--allow-quality-loss", action="store_true",
                   help="rank candidates that predict a QUALITY degradation "
                        "too (default: they are reported as incomparable "
                        "and never auto-ranked)")
    p.set_defaults(func=cmd_compare_capability)

    p = sub.add_parser(
        "accept-capability",
        help="EXPLICITLY accept a capability recommendation and run the real "
             "offline operation on the prediction's OWN scope (M5). The only "
             "M5 command that changes knowledge")
    p.add_argument("--recommendation", required=True,
                   help="recommendation JSON from `orx compare-capability` "
                        "(literal or file)")
    p.add_argument("--prediction", default=None,
                   help="override which prediction to execute (default: the "
                        "recommendation's selected one)")
    p.add_argument("--verify", default=None,
                   help="admission check JSON for the induction (literal or "
                        "file)")
    p.add_argument("--note", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_accept_capability)

    p = sub.add_parser(
        "reject-capability",
        help="explicitly decline or defer a capability recommendation (M5): "
             "no operation runs and NO knowledge changes")
    p.add_argument("--recommendation", required=True)
    p.add_argument("--prediction", default=None)
    p.add_argument("--reason", default=None)
    p.set_defaults(func=cmd_reject_capability)

    p = sub.add_parser(
        "bind-capability",
        help="stage 1 of the capability feedback: bind the REAL maintenance "
             "fact (did the operation happen, what knowledge changed, what "
             "did it really cost). Never sets effect_verified (M5)")
    p.add_argument("--prediction", required=True)
    p.add_argument("--adoption-action", default=None,
                   help="the adoption action id from `orx accept-capability`; "
                        "omit to locate it by the prediction id")
    p.set_defaults(func=cmd_bind_capability)

    p = sub.add_parser(
        "evaluate-capability",
        help="stage 2 of the capability feedback: judge a prediction "
             "against REAL later-task results, or record a pre-arranged "
             "paired evaluation (M5). An unmet horizon stays pending and "
             "re-evaluable")
    p.add_argument("--prediction", required=True)
    p.add_argument("--tasks", default=None,
                   help="comma-separated task ids to read as the validation "
                        "sample (default: every closed episode of a task "
                        "outside the prediction's own experience scope)")
    p.add_argument("--paired", default=None,
                   help="record a pre-arranged PAIRED comparison instead of "
                        "evaluating: JSON {metric, reference_value, "
                        "treated_value, unit, source, reference_task_ids, "
                        "note}. The framework does not fabricate a "
                        "counterfactual")
    p.add_argument("--allow-descriptive", action="store_true",
                   help="accept a before/after change with no comparable "
                        "reference as a descriptive result (default: such a "
                        "change is inconclusive and cannot be attributed to "
                        "the operation)")
    p.set_defaults(func=cmd_evaluate_capability)

    p = sub.add_parser("plan-next",
                       help="bounded next-step planning over predicted "
                            "action consequences (M3): freeze one root "
                            "snapshot, compare <=3 root candidates (and "
                            "optional horizon-2 continuations), suggest the "
                            "first step. Never executes, never writes "
                            "selected_plan")
    p.add_argument("--task", required=True)
    p.add_argument("--episode", default=None)
    p.add_argument("--candidates", default=None,
                   help="JSON list of ActionSpec objects (literal or file); "
                        "REQUIRED — the framework does not generate a "
                        "candidate menu. Propose the methods you want "
                        "compared; `recall` shows what memory already "
                        "holds for this problem")
    p.add_argument("--horizon", type=int, default=1,
                   help="fixed at 1: planning compares macro strategy "
                        "candidates once, then you re-plan from the REAL "
                        "observation. Any other value is refused")
    p.add_argument("--max-calls", type=int, default=6,
                   help="max world-model calls for the whole decision")
    p.add_argument("--time-budget", type=float, default=120.0,
                   dest="time_budget", metavar="S",
                   help="wall-clock budget for the WHOLE planning decision, "
                        "in seconds (default 120). Before each call the "
                        "remaining budget is passed down as that call's "
                        "timeout, and after the last call the elapsed time "
                        "is re-checked. Exhaustion returns a truncation "
                        "reason and KEEPS the calls already paid for")
    p.add_argument("--delta", type=float, default=argparse.SUPPRESS,
                   help="weight of the predicted knowledge term in the path "
                        "utility (U = alpha*Q - beta*C - gamma*R + delta*K). "
                        "Omitted = this harness's configured value, which is "
                        "0.0 unless set: an unknown knowledge value is NOT "
                        "rewarded, so a positive delta is a deliberate "
                        "experimental choice. Overrides the global --delta "
                        "for this one decision")
    p.add_argument("--prediction-mode", default=argparse.SUPPRESS,
                   choices=["x-b-only", "h-x-b", "h-x-b-value"],
                   help="what the world model predicts: 'x-b-only' predicts "
                        "X/B only (no knowledge targets, knowledge term "
                        "off), 'h-x-b' predicts H as well but keeps the "
                        "knowledge value out of the decision, "
                        "'h-x-b-value' lets it influence the choice")
    p.add_argument("--benefit-kind", default=None, metavar="KIND",
                   help="the benefit CONVENTION this decision compares under: "
                        "'solution_quality' (how well the solver solved the "
                        "model) or 'effective_completion' (whether the "
                        "ANSWER satisfies the TASK). Declaring it makes every "
                        "candidate be predicted AND compared in that "
                        "currency; without it the candidates' own agreement "
                        "is used, else solution_quality")
    p.add_argument("--benefit-metric", default=None, metavar="METRIC",
                   help="the metric of --benefit-kind (required with it): "
                        "'normalized_objective_gap' for solution_quality, "
                        "'task_result_check_passed' for effective_completion")
    p.add_argument("--benefit-unit", default=None, metavar="UNIT",
                   help="the unit of the declared metric (optional; the "
                        "convention's own unit is used when omitted)")
    p.set_defaults(func=cmd_plan_next)

    p = sub.add_parser("choose-next",
                       help="record your explicit choice after a plan: "
                            "accept the suggestion, pick another candidate "
                            "(deviation), or reject. Only this writes "
                            "X.selected_plan. Name the candidate by its "
                            "PREDICTION id instead of re-typing its JSON")
    p.add_argument("--decision", required=True, metavar="ACTION_ID",
                   help="the select_strategy decision action id from "
                        "plan-next")
    p.add_argument("--prediction", default=None, metavar="PREDICTION_ID",
                   help="the prediction id of the candidate you CHOSE, from "
                        "this decision's own `result.plan.candidates[]`. The "
                        "candidate is read from the decision's recorded "
                        "comparison, so it must belong to THIS decision; the "
                        "deviation test compares the candidate reference "
                        "(action type/strategy/solver/config), not the whole "
                        "JSON")
    p.add_argument("--chosen", default=None,
                   help="ActionSpec JSON of the action you will execute "
                        "(an alternative to --prediction; if both are given "
                        "they must name the same candidate)")
    p.add_argument("--rejected", action="store_true",
                   help="record that no suggestion/candidate was taken")
    p.add_argument("--note", default=None,
                   help="free-text note (e.g. deviation reason)")
    p.set_defaults(func=cmd_choose_next)

    p = sub.add_parser(
        "induction-candidates",
        help="scan the REAL bank for induction/revision candidate bundles "
             "(frozen evidence, no model call). The evidence package is "
             "what `predict-capability --bundle` consumes")
    p.set_defaults(func=cmd_induction_candidates)

    p = sub.add_parser(
        "induction-material",
        help="organize the READABLE material for an offline induction step: "
             "the methods actually used, what changed, both sides of a "
             "comparison, what followed. Read it, then submit a claim with "
             "`orx induce --relation`. No model call, no writes")
    p.add_argument("--bundle", default=None, metavar="BUNDLE_ID",
                   help="read the material of ONE candidate (default: every "
                        "candidate). Pass the id `orx induction-candidates` "
                        "printed")
    p.add_argument("--pattern", default=None,
                   choices=["strategy_contrast", "intervention_recovery",
                            "structural_reproduction", "advantage_reversal"],
                   help="restrict to candidates from one detector")
    p.add_argument("--strategy", default=None,
                   help="restrict to candidates about one strategy id")
    p.set_defaults(func=cmd_induction_material)

    p = sub.add_parser("retire", help="move an entry to the cold archive (explicit)")
    p.add_argument("--entry", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_retire)

    p = sub.add_parser(
        "exclude-execution",
        help="withdraw a wrong execution FACT from the evidence set "
             "(append-only: the row is preserved, but stops counting)")
    p.add_argument("--execution", required=True, metavar="EXECUTION_ID")
    p.add_argument("--reason", required=True,
                   help="why this fact is withdrawn (kept on the fact for "
                        "audit)")
    p.add_argument("--superseded-by", default=None, metavar="EXECUTION_ID",
                   help="the corrected re-run that replaces it (a link, never "
                        "an inference)")
    p.set_defaults(func=cmd_exclude_execution)

    p = sub.add_parser(
        "restore-execution",
        help="reverse an exclusion: the fact counts as evidence again")
    p.add_argument("--execution", required=True, metavar="EXECUTION_ID")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_restore_execution)

    p = sub.add_parser(
        "rebuild-index",
        help="rebuild the retrieval (embedding) index from the current facts "
             "and entries — explicit maintenance, not a routine path")
    p.add_argument("--layer", default="both",
                   choices=["both", "execution", "strategic"],
                   help="which index to rebuild (default: both)")
    p.add_argument("--dry-run", action="store_true",
                   help="count what would be indexed; touches nothing — no "
                        "embedding call, no index file written")
    p.set_defaults(func=cmd_rebuild_index)

    p = sub.add_parser("doctor", help="environment self-check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser(
        "contract",
        help="build or read a unified world-model contract "
             "(strategy outcome / capability evolution). No model call is "
             "made: building returns a contract_only object when no "
             "prediction service is attached, and reading never parses an "
             "unknown contract version")
    p.add_argument("--kind", default=None,
                   choices=["strategy_outcome", "capability_evolution"],
                   help="which contract to BUILD (omit when reading "
                        "--payload)")
    p.add_argument("--payload", default=None,
                   help="READ a stored prediction payload (literal or @file): "
                        "current contract, legacy unversioned, or an "
                        "explicitly unsupported version")
    p.add_argument("--task", default=None,
                   help="task JSON (literal or @file); required to build a "
                        "strategy_outcome contract, optional for capability_evolution")
    p.add_argument("--spec", default=None,
                   help="candidate JSON. A CandidateRef: action_type, "
                        "strategy_id, solver, config, preconditions, "
                        "expected_scope, stop_conditions, scope "
                        "(attempt|strategy_window). A LEGACY ActionSpec "
                        "(with measurement_scope / budget_hint) is also "
                        "accepted and mapped through from_action_spec, "
                        "which preserves its execution config and REFUSES "
                        "an unmappable scope such as 'task'")
    p.add_argument("--episode", default=None)
    p.add_argument("--benefit", default=None,
                   help="BenefitEstimate JSON: {kind, metric, unit, value, "
                        "baseline:{kind,value}} — 'metric' and a baseline are "
                        "required whenever 'value' is present")
    p.add_argument("--cost", default=None,
                   help="ExpectedCost JSON: {expected:{<dimension>: <value>}, "
                        "expected_measured:[...]}")
    p.add_argument("--risk", default=None,
                   help="RiskStatement JSON: {events:[{event, probability, "
                        "severity, basis}]} — risk events stay separate "
                        "from cost")
    p.add_argument("--operation", default=None,
                   help="LearningOperation JSON: {operation_type: induce|"
                        "revise|reverify|retire, strategy_id, description, "
                        "scope}")
    p.add_argument("--operation-desc", default=None,
                   help="free-text description for the learning operation")
    p.add_argument("--scope", default=None,
                   help="ExperienceScope JSON: {execution_ids, task_ids, "
                        "family, cell_token}")
    p.add_argument("--targeting", default=None,
                   help="TaskTargeting JSON: {description, family, "
                        "cell_token, predicates, task_ids}")
    p.add_argument("--baseline", default=None,
                   help="BaselineStatement JSON: {kind, value, note}")
    p.add_argument("--horizon", default=None,
                   help="what window the capability change is claimed over, "
                        "e.g. 'next 10 matching tasks'")
    p.add_argument("--horizon-tasks", type=int, default=None,
                   help="task count of the horizon, when known")
    p.add_argument("--expected-change", default=None,
                   help="ExpectedChange JSON (repeatable): {metric, "
                        "direction, value, baseline}")
    p.add_argument("--verification", action="append", default=None,
                   help="VerificationCondition JSON (repeatable): "
                        "{condition, evaluable, check_basis} — what would "
                        "actually CONFIRM the predicted change")
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser(
        "context",
        help="build or read the FROZEN prediction input context (world-model "
             "phase 2): the joint problem representation (text + CIR + math "
             "attributes), X/B from one snapshot, the retrieval evidence of "
             "both channels, the harness capability evidence and the "
             "external execution constraints. No model call and no solver "
             "run: the only external call is the configured embedding "
             "backend on the existing retrieval path")
    p.add_argument("--task", default=None,
                   help="task JSON (literal or @file) to BUILD a context for")
    p.add_argument("--episode", default=None)
    p.add_argument("--top", type=int, default=3,
                   help="bounded top-k for BOTH retrieval channels (default 3)")
    p.add_argument("--cir", default=None,
                   help="CIR JSON (literal or @file); defaults to the task's "
                        "own 'coupling' field")
    p.add_argument("--math", default=None,
                   help="explicit math attribute declarations JSON: "
                        "{integrality, linearity, objective_kind, "
                        "constraint_kinds} — the caller taking responsibility "
                        "for a value it knows")
    p.add_argument("--include-unverified", action="store_true",
                   help="surface unverified candidates too (inspection view); "
                        "they stay labelled, never published")
    p.add_argument("--context-id", default=None, metavar="CTX_ID",
                   help="READ a stored context instead of building one (never "
                        "re-runs retrieval or a model call)")
    p.add_argument("--no-persist", action="store_true",
                   help="build without writing the context to the store")
    p.set_defaults(func=cmd_context)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CIRFormatError as exc:
        # Every CIR entry point (profile, context, snapshot, execute, recall,
        # predict) funnels through the same shape gate, so the structured
        # rejection is handled ONCE here rather than in each command — a
        # per-command handler is exactly how one entry point ends up without
        # one.
        payload = _cir_error_payload(exc)
        return _fail_structured(
            payload,
            f"CIR rejected ({payload['cause']}): {payload['detail']} "
            f"{payload['hint']}")
    except (ValueError, StorageError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # pragma: no cover - crash guard
        return _fail(f"unexpected {type(exc).__name__}: {exc}", exit_code=1)


if __name__ == "__main__":
    sys.exit(main())
