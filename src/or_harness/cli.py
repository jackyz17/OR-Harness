"""orx: thin CLI shell over ORHarness.

stdout is ALWAYS a single compact JSON object:
    {"result": {...}, "summary": "2-4 sentence agent-readable text"}
Exit codes: 0 success; 2 usage/precondition error; 1 crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from or_harness.api import ORHarness, PREDICTION_MODES
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
        provider = HttpChatProvider(parts[0], parts[1], api_key,
                                    timeout_s=getattr(args, "wm_timeout", 30)
                                    or 30.0)
    return ORHarness(home=args.home, alpha=args.alpha, beta=args.beta,
                     gamma=args.gamma, cost_weights=weights,
                     delta=getattr(args, "delta", 0.0),
                     prediction_mode=getattr(args, "prediction_mode",
                                            "h-x-b-value"),
                     world_model=provider)


def _summarize_recall(result: Dict[str, Any]) -> str:
    recs = result["recommendations"]
    parts: List[str] = []
    if not recs:
        parts.append("No applicable strategies.")
    else:
        top = recs[0]
        parts.append(f"Top candidate: {top['strategy_id']} ({top['name']}), "
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
    intermediate representation written AFTER the strategy is chosen, and
    re-running profile then adds the CIR ↔ model cross-check."""
    h = _harness(args)
    try:
        from or_harness.core.coupling import (
            CouplingAwareIR,
            cross_check_cir_model,
            derive_coupling_groups,
            infer_structural_relations,
            render_modeling_guidance,
            validate_cir,
        )
        from or_harness.profiling.model_syntax import verify_model
        task = _load_json_arg(args.task)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else None
        # -- CIR side (validation, structural inference, groups, guidance) --
        coupling: Dict[str, Any] = {"cir": None, "modeling_guidance": [],
                                    "cir_warnings": []}
        cir_obj = None
        if args.cir:
            cir_obj = CouplingAwareIR.from_dict(_load_json_arg(args.cir))
        elif isinstance(task.get("coupling"), dict):
            cir_obj = CouplingAwareIR.from_dict(task["coupling"])
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
                "cir_warnings": cross_check_cir_model(cir_obj, parsed),
            }
        else:
            coupling["message"] = (
                "No 'coupling' field found in the task. A CIR is optional "
                "but recommended: it is the pre-model understanding that "
                "improves both the profile derivation and the model you "
                "write after choosing a strategy.")
        # -- Profile side --
        profile = h.profile(task, code, cir=cir_obj)
        report = h.derivation_report(task, code, cir=cir_obj)
        result = {"profile": profile.to_dict(), "derivation": report,
                  "coupling": coupling}
        return _emit(result, _summarize_profile(profile, report, coupling))
    finally:
        h.close()


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
    warnings = report.get("coupling_warnings") or []
    for w in warnings:
        parts.append("WARNING: " + w["message"])
    for w in coupling.get("cir_warnings") or []:
        parts.append(f"WARNING: {w['code']}: {w['detail']}")
    return " ".join(parts)


def cmd_recall(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else None
        result = h.recall(task, top=args.top,
                          exclude=args.exclude or [],
                          memory_mode=args.memory_mode, code=code,
                          include_unverified=args.include_unverified)
        return _emit(result, _summarize_recall(result))
    finally:
        h.close()


def cmd_predict(args) -> int:
    """Pre-execution cost expectation snapshot for one (task, strategy)."""
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else None
        snapshot = h.predict_cost(task, args.strategy, code=code)
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
        record = h.execute(task, args.strategy, args.code, args.workspace,
                           solver=args.solver,
                           verification_level=args.verification,
                           episode_id=getattr(args, "episode", None))
        out = {"execution": record.to_dict(),
               "execution_id": record.execution_id,
               "action_id": record.action_id}
        q = record.quality
        return _emit(out,
                     f"Execution {record.execution_id} finished with status "
                     f"{q['status']} (feasible={q['feasible']}, gap={q['gap']}). "
                     f"Cost so far: {record.cost.to_dict()}. Nothing is recorded "
                     "yet — call `orx record` to persist, or discard.")
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
            if "execution" in data:
                data = data["execution"]
            record = ExecutionRecord.from_dict(data)
        elif args.record_file:
            record = ExecutionRecord.from_dict(_load_json_arg(args.record_file))
        else:
            return _fail("record requires --execution <json|path>, "
                         "--from-staged <id>, or --record-file")
        override = None
        if args.override:
            override = _parse_dimension_pairs(args.override)
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
                          retain_reason=args.retain_reason,
                          prediction=prediction)
        hints = result["induction_hints"]
        checks = result["prediction_checks"]
        summary = [f"Recorded {result['execution_id']}."]
        if checks:
            hits = sum(1 for c in checks if c["hit"])
            summary.append(f"Prediction checks: {hits}/{len(checks)} hits "
                           f"across matching entries.")
        if hints:
            summary.append("Induction hints: " + "; ".join(
                f"{h['criterion']}({','.join(h['strategy_ids'])})" for h in hints)
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
        return _emit(result, " ".join(summary))
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
        result = h.induce(strategy_id=args.strategy, all_=args.all,
                          rebuild=args.rebuild, dry_run=args.dry_run,
                          force=args.force, notes=notes, verify=verify)
        return _emit(result, _summarize_induce(result, args))
    finally:
        h.close()


def _summarize_induce(result: Dict[str, Any], args) -> str:
    if args.rebuild:
        if result.get("dry_run") or "would_rebuild" in result:
            return (f"Rebuild plan: {result.get('would_rebuild', 0)} cells would "
                    "be re-induced from currently retained evidence. The cold "
                    "archive is preserved. Run without --dry-run to apply.")
        return (f"Re-induced {result.get('rebuilt', 0)} entries from retained "
                "evidence. Exact reconstruction is not a requirement — the "
                "re-induced bank may differ from the previous one.")
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


def cmd_assess_induction(args) -> int:
    """Evaluate induction candidates using the world model (M4)."""
    h = _harness(args)
    try:
        if args.candidates_only:
            bundles = h.induction_candidates()
            return _emit({"count": len(bundles), "candidates": bundles},
                         f"Found {len(bundles)} induction candidate(s).")
        if args.bundle:
            bundle = _load_json_arg(args.bundle)
        else:
            bundles = h.induction_candidates()
            if not bundles:
                return _emit({"status": "no_candidates", "candidates": []},
                             "No induction candidates with sufficient evidence.")
            bundle = bundles[0]
        workload = _load_json_arg(args.workload) if args.workload else None
        res = h.assess_induction(bundle, workload_forecast=workload)
        summary = (f"Induction assessment {res.get('assessment_id')}: "
                   f"recommendation={res.get('recommendation')} "
                   f"({res.get('recommendation_basis', '')})")
        return _emit(res, summary)
    finally:
        h.close()


def cmd_inspect(args) -> int:
    h = _harness(args)
    try:
        result = h.inspect(bank=args.bank, task_id=args.task,
                           strategy_id=args.strategy, status=args.status,
                           episode_id=getattr(args, "episode", None))
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
                      "calibration_error, and consecutive_misses.")
        elif args.bank == "archive":
            detail = (" A card blocks re-inducing that same pattern; lift it "
                      "with `induce --force` when the environment has "
                      "genuinely drifted.")
        elif args.bank == "actions":
            detail = (" Actions carry lifecycle status (running = begun, "
                      "not ended) and, for induce, a business result "
                      "separate from the lifecycle status.")
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


def cmd_predict_outcome(args) -> int:
    h = _harness(args)
    try:
        from or_harness.world_model.prediction import ActionSpec
        task = _load_json_arg(args.task)
        spec = ActionSpec.from_dict(_load_json_arg(args.action_spec))
        if args.no_context:
            context = False
        elif args.context:
            context = h.get_prediction_context(args.context)
            if context is None:
                return _fail(f"unknown context_id {args.context!r}")
        else:
            context = None
        prediction = h.predict_outcome(
            task, spec, args.episode,
            parent_action_id=args.parent_action, context=context)
        result = {"prediction": prediction.to_dict(),
                  "prediction_id": prediction.prediction_id}
        if prediction.status == "not_configured":
            return _fail(f"prediction not enabled: {prediction.error}", 2)
        if prediction.status != "valid":
            return _emit(result,
                         f"Prediction {prediction.prediction_id} failed "
                         f"with status {prediction.status}: "
                         f"{prediction.error or 'no detail'}. The call cost "
                         "(if any) is recorded on the prediction.")
        predicted = prediction.predicted
        parts = [f"Prediction {prediction.prediction_id} (shadow) for "
                 f"{spec.action_type}"
                 + (f" {spec.strategy_id}" if spec.strategy_id else "")
                 + ": "]
        if "outcome_status" in predicted:
            parts.append(f"expected status "
                         f"{predicted['outcome_status']}; ")
        if "quality" in predicted:
            parts.append(f"E[Q]={predicted['quality']}; ")
        if "failure_prob" in predicted:
            parts.append(f"P(fail)={predicted['failure_prob']}; ")
        cost = predicted.get("cost") or {}
        if cost:
            parts.append(f"cost={ {k: round(v, 2) for k, v in cost.items()} }; ")
        if prediction.unsupported_fields:
            parts.append(f"not predicted: "
                         f"{', '.join(prediction.unsupported_fields)}; ")
        parts.append("This is a hypothesis — it changes nothing. Execute "
                     "the action yourself, then `orx bind-outcome "
                     f"--prediction {prediction.prediction_id} --action "
                     "<id>`.")
        return _emit(result, "".join(parts))
    finally:
        h.close()


def cmd_bind_outcome(args) -> int:
    h = _harness(args)
    try:
        prediction = h.bind_outcome(args.prediction, args.action)
        if prediction.binding_mismatch:
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} WITH MISMATCH: "
                       f"{prediction.binding_mismatch}. The comparison "
                       "covers only matching parts; mismatched fields are "
                       "recorded, not scored.")
        else:
            prediction = h.compare_prediction(args.prediction)
            fb = prediction.feedback or {}
            if fb.get("compared"):
                fields = fb.get("compared_fields") or {}
                matched = sum(1 for k in ("outcome_status", "feasible")
                              if fields.get(k, {}).get("match"))
                total = sum(1 for k in ("outcome_status", "feasible")
                            if k in fields)
                parts = [f"Prediction {args.prediction} bound and compared"
                         f" against execution {fb.get('execution_id')}: "
                         f"category {matched}/{total} matched."]
                if "quality" in fields:
                    parts.append(f"quality error "
                                 f"{fields['quality']['abs_error']}; ")
                if "cost" in fields:
                    dims = ", ".join(
                        f"{d}(log_err {v['log_error']})"
                        for d, v in fields["cost"].items())
                    parts.append(f"cost: {dims}.")
                skipped = fb.get("not_compared") or {}
                if skipped:
                    parts.append(f"Not compared: "
                                 f"{', '.join(skipped)}.")
            else:
                parts = [f"Prediction {args.prediction} bound but NOT "
                         f"compared: {fb.get('reason')}."]
            summary = " ".join(parts)
        return _emit({"prediction": prediction.to_dict()}, summary)
    finally:
        h.close()


def cmd_bind_induction_outcome(args) -> int:
    h = _harness(args)
    try:
        result = h.bind_induction_outcome(args.assessment)
    except StorageError as exc:
        return _fail(str(exc))
    verdict = result.get("verdict") or {}
    if result.get("compared"):
        return _emit(result, f"Induction assessment {args.assessment} "
                             f"bound: {verdict.get('status')} "
                             f"(entries created: "
                             f"{len(verdict.get('entries_created') or [])}). "
                             "An entry forming is not the same as "
                             "publishable knowledge.")
    return _emit(result, f"Induction assessment {args.assessment} not "
                         f"compared: {result.get('reason') or verdict.get('reason')}")


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
        if prediction.status == "contract_only" \
                and not prediction.provider_configured:
            return _fail(f"prediction not enabled: "
                         f"{(prediction.notes or ['no provider'])[0]}", 2)
        parts = [f"Strategy-outcome prediction {prediction.prediction_id} "
                 f"({prediction.status}) for "
                 f"{prediction.candidate.strategy_id or '(unnamed)'}: "]
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
        elif prediction.trace.comparable:
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} and COMPARABLE: the real execution "
                       "may be scored against it. Window-level scoring "
                       "itself lands in M4.")
        else:
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action}, not yet comparable: "
                       f"{prediction.trace.not_comparable_reasons}.")
        return _emit({"prediction": prediction.to_dict()}, summary)
    finally:
        h.close()


def cmd_plan_next(args) -> int:
    h = _harness(args)
    try:
        from or_harness.world_model.prediction import ActionSpec
        task = _load_json_arg(args.task)
        candidates = None
        if args.candidates:
            raw = _load_json_arg(args.candidates)
            if not isinstance(raw, list):
                return _fail("--candidates must be a JSON list of "
                             "ActionSpec objects")
            candidates = [ActionSpec.from_dict(c) for c in raw]
        limits = {"horizon": args.horizon,
                  "max_model_calls": args.max_calls}
        if getattr(args, "delta", None) is not None:
            limits["delta"] = args.delta
        protocol = getattr(args, "protocol", "legacy") or "legacy"
        plan = h.plan_next(task, episode_id=args.episode,
                           candidates=candidates, limits=limits,
                           protocol=protocol)
        result = {"plan": plan, "decision_action_id": plan.get(
            "decision_action_id")}
        if plan.get("protocol"):
            result["protocol"] = plan["protocol"]
        status = plan.get("status")
        if status in ("disabled", "no_candidates", "fallback"):
            return _emit(result, f"plan_next returned status={status}: "
                                 f"{plan.get('truncation_reason')}")
        paths = plan.get("paths") or []
        parts = [f"Plan {plan['plan_id']}: {len(paths)} path(s) evaluated "
                 f"from snapshot {plan['root_snapshot_id']} "
                 f"(calls={plan.get('model_calls_made')}, "
                 f"planning_cost={plan.get('planning_cost')})."]
        suggested = plan.get("suggested")
        if suggested:
            parts.append(f"Suggested first step: {suggested['action_type']}"
                         + (f" {suggested.get('strategy_id')}"
                            if suggested.get("strategy_id") else "")
                         + f" — {plan.get('suggestion_basis')}. Accept with "
                           f"`orx choose-next --decision "
                           f"{plan.get('decision_action_id')} --chosen ...`, "
                           "or choose something else.")
        else:
            parts.append(plan.get("suggestion_basis")
                         or "No suggestion (see paths).")
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
        result = h.choose_next(args.decision, chosen=chosen,
                               rejected=args.rejected,
                               deviation_note=args.note)
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
                         "`orx execute`, then record and bind the "
                         "prediction.")
        if result.get("deviation"):
            summary += " (deviation from the suggestion recorded)"
        return _emit(result, summary)
    finally:
        h.close()


def cmd_gc(args) -> int:
    h = _harness(args)
    try:
        result = h.collect_garbage(mode=args.mode, dry_run=args.dry_run)
        if result.get("deferred"):
            return _emit(result, f"GC deferred: {result['deferred']}")
        if result.get("dry_run"):
            actions = result["actions"]
            if not actions:
                return _emit(result, "GC dry-run: nothing to dispose.")
            lines = [f"- {a['kind']} {a['target']}: {a['reason']}" for a in actions]
            return _emit(result, "GC dry-run plan:\n" + "\n".join(lines))
        return _emit(result,
                     f"GC applied: retirement candidates listed but NOT "
                     f"retired — retire entries explicitly.")
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
            task, args.episode, top=args.top, code=args.code, cir=cir,
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
    parser.add_argument("--wm-timeout", type=float, default=30.0,
                        help="world-model call timeout in seconds")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile",
                       help="the single analysis entry: CIR validation + "
                            "modeling guidance + problem profile + "
                            "derivation report (a task without a 'model' "
                            "field is a normal state)")
    p.add_argument("--task", required=True, help="task JSON literal or file")
    p.add_argument("--code", default=None, help="optional solve script for AST derivation")
    p.add_argument("--cir", default=None,
                   help="optional CIR JSON literal/file (overrides the "
                        "task's 'coupling' field)")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("recall", help="recall accumulated experience for a task")
    p.add_argument("--task", required=True)
    p.add_argument("--code", default=None)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--memory-mode", default="cost-aware",
                   choices=["none", "cases", "strategic", "cost-aware"])
    p.add_argument("--include-unverified", action="store_true",
                   help="offline/inspection view: also surface UNPUBLISHED "
                        "candidates (unverified / insufficient_evidence) "
                        "in both channels. Default excludes them: a candidate "
                        "is held by the framework, not knowledge.")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("predict",
                       help="pre-execution cost expectation snapshot for one "
                            "(task, strategy); pass it back to `orx record` "
                            "via --prediction")
    p.add_argument("--task", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--code", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("execute", help="sandbox-execute a solve script")
    p.add_argument("--task", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--code", required=True, help="path to solve.py")
    p.add_argument("--workspace", required=True)
    p.add_argument("--solver", required=True)
    p.add_argument("--verification", default="basic", choices=["basic", "strong"])
    p.add_argument("--episode", default=None,
                   help="episode id for the unified action record "
                        "(budget/progress scoping)")
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
    p.add_argument("--prediction", default=None,
                   help="pre-execution cost prediction snapshot (the JSON "
                        "printed by `orx predict`) actually used for this "
                        "attempt; feedback compares against it, never a "
                        "post-hoc estimate")
    p.add_argument("--retain-reason", default=None,
                   help="explicitly mark this episode as representative "
                        "evidence (reserved for future compaction policies), "
                        "e.g. 'contrast'; when omitted, any mark already on "
                        "the record is preserved")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser(
        "induce", help="consolidate facts into strategic entries",
        epilog=("Creating an entry requires >=2 supporting executions from "
                ">=2 distinct task_ids: repeating one task is repetition, not "
                "reproduction. Refreshing an existing entry is never gated. "
                "Publishing is separate: an entry becomes strategic knowledge "
                "only once --verify renders 'verified'."))
    p.add_argument("--strategy", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--rebuild", action="store_true")

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
                        "verdict the entry is NOT published")
    p.set_defaults(func=cmd_induce)

    p = sub.add_parser("inspect", help="query the memory layers")
    p.add_argument("--bank", default="experience",
                   choices=["experience", "strategic", "archive",
                            "actions", "snapshots", "predictions", "texts"])
    p.add_argument("--task", default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--episode", default=None,
                   help="filter actions by episode id")
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

    p = sub.add_parser("predict-outcome",
                       help="ask the configured world model for a "
                            "structured prediction of ONE candidate "
                            "action's consequences (shadow: never changes "
                            "recommendations)")
    p.add_argument("--task", required=True)
    p.add_argument("--action-spec", required=True,
                   help="candidate action JSON (literal or file): "
                        '{"action_type": "execute_strategy", "strategy_id": '
                        '"S01", "solver": "highs", ...}')
    p.add_argument("--episode", default=None)
    p.add_argument("--parent-action", default=None, metavar="ACTION_ID",
                   help="action this prediction call belongs to (its model "
                        "call cost is charged there as own cost)")
    p.add_argument("--context", default=None, metavar="CTX_ID",
                   help="reuse a FROZEN prediction input context built by "
                        "`orx context` (its identity is verified against this "
                        "task/version/episode); default builds a fresh one")
    p.add_argument("--no-context", action="store_true",
                   help="send no prediction context at all: the request keeps "
                        "its pre-phase-2 shape exactly (the compatibility "
                        "escape hatch; `x-b-only` byte-compatibility rests "
                        "on this)")
    p.set_defaults(func=cmd_predict_outcome)

    p = sub.add_parser("bind-outcome",
                       help="bind a prediction to the real action that ran, "
                            "then compare (type/strategy/solver checked)")
    p.add_argument("--prediction", required=True)
    p.add_argument("--action", required=True)
    p.set_defaults(func=cmd_bind_outcome)

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
             "a mismatch is recorded, never scored)")
    p.add_argument("--prediction", required=True)
    p.add_argument("--action", required=True)
    p.set_defaults(func=cmd_bind_strategy)

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
                        "omitted = catalog vocabulary filtered by "
                        "applicability and available solver families")
    p.add_argument("--horizon", type=int, default=1, choices=[1, 2])
    p.add_argument("--max-calls", type=int, default=6,
                   help="max world-model calls for the whole decision")
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
    p.add_argument("--protocol", default="legacy",
                   choices=["legacy", "strategy-outcome"],
                   help="prediction protocol: 'legacy' (default) keeps the "
                        "existing OutcomePrediction path and horizon 1-2; "
                        "'strategy-outcome' compares candidates under the "
                        "wm-so/1 strategy-outcome protocol (benefit/cost/"
                        "risk/uncertainty, horizon fixed at 1)")
    p.set_defaults(func=cmd_plan_next)

    p = sub.add_parser("bind-induction-outcome",
                       help="bind an induction assessment's predictions to "
                            "the REAL induction outcome and record the "
                            "verdict (the assessment's own slow feedback)")
    p.add_argument("--assessment", required=True, metavar="ASSESSMENT_ID")
    p.set_defaults(func=cmd_bind_induction_outcome)

    p = sub.add_parser("choose-next",
                       help="record your explicit choice after a plan: "
                            "accept the suggestion, pick another candidate "
                            "(deviation), or reject. Only this writes "
                            "X.selected_plan")
    p.add_argument("--decision", required=True, metavar="ACTION_ID",
                   help="the select_strategy decision action id from "
                        "plan-next")
    p.add_argument("--chosen", default=None,
                   help="ActionSpec JSON of the action you will execute")
    p.add_argument("--rejected", action="store_true",
                   help="record that no suggestion/candidate was taken")
    p.add_argument("--note", default=None,
                   help="free-text note (e.g. deviation reason)")
    p.set_defaults(func=cmd_choose_next)

    p = sub.add_parser("assess-induction",
                       help="evaluate induction/revision candidates using the "
                            "world model (M4)")
    p.add_argument("--bundle", default=None,
                   help="InductionCandidateBundle JSON (literal or @file); "
                        "omitted = auto-scan candidate bundles from bank")
    p.add_argument("--candidates-only", action="store_true",
                   help="scan and return candidate bundles without evaluating")
    p.add_argument("--workload", default=None,
                   help="workload forecast JSON (e.g. expected_matching_tasks)")
    p.set_defaults(func=cmd_assess_induction)

    p = sub.add_parser("gc", help="dispose of the derived layer (harness's call)")
    p.add_argument("--mode", default="compact", choices=["compact", "purge"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("retire", help="move an entry to the cold archive (explicit)")
    p.add_argument("--entry", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_retire)

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
    p.add_argument("--code", default=None,
                   help="solve.py used as a coupling source when profiling")
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
    except (ValueError, StorageError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # pragma: no cover - crash guard
        return _fail(f"unexpected {type(exc).__name__}: {exc}", exit_code=1)


if __name__ == "__main__":
    sys.exit(main())
