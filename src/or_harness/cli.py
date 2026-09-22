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
                          force=args.force, notes=notes, verify=verify,
                          family=getattr(args, "family", None),
                          cell=getattr(args, "cell", None))
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
        elif info.get("binding_unknown"):
            summary = (f"Prediction {args.prediction} bound to action "
                       f"{args.action} with UNKNOWN identity fields: "
                       f"{sorted(info['binding_unknown'])}. Unknown is not "
                       "a match: the fields that depend on them stay "
                       "unevaluable at close-out.")
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
        return _emit(result, " ".join(parts))
    finally:
        h.close()


def cmd_calibration(args) -> int:
    """Read the published experience-calibration summary (read-only)."""
    h = _harness(args)
    try:
        result = h.calibration_summary(
            min_samples=args.min_samples
            if args.min_samples is not None else None)
        groups = result.get("groups") or {}
        if not groups:
            summary = ("No calibration samples from closed episodes yet: "
                       "reliability is unknown (insufficient_evidence), "
                       "never guessed.")
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
                        f"interval coverage="
                        f"{group.get('interval_coverage')}")
            summary = ("Strategy-outcome experience calibration: "
                       + "; ".join(parts)
                       + ". A measured record of past closed episodes — "
                         "not a promise that future predictions improve.")
        return _emit(result, summary)
    finally:
        h.close()


def cmd_evaluations(args) -> int:
    """List stored post-hoc prediction evaluations (read-only)."""
    h = _harness(args)
    try:
        evaluations = h.strategy_prediction_evaluations(
            task_id=args.task, episode_id=args.episode)
        if args.evaluation:
            single = h.get_strategy_evaluation(args.evaluation)
            if single is None:
                return _fail(f"unknown evaluation_id {args.evaluation!r}")
            return _emit(single,
                         f"Evaluation {args.evaluation}: state "
                         f"{single.get('state')}.")
        n_evaluated = sum(1 for e in evaluations
                          if e.get("state") == "evaluated")
        return _emit(
            {"evaluations": evaluations, "count": len(evaluations)},
            f"{len(evaluations)} evaluation(s) stored ({n_evaluated} "
            "evaluated). Excluded evaluations are neither hits nor misses; "
            "pending ones wait for their scope to end.")
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
        return _emit(result,
                     f"Accepted {result['capability_prediction_id']}: the "
                     f"operation ran on {len(result['execution_ids'])} "
                     f"scoped execution(s), {len(created)} entr(y/ies) "
                     "created. Bind the real fact next "
                     "(`orx bind-capability --prediction ...`); the "
                     "capability EFFECT stays unverified until qualified "
                     "later tasks produce real results.")
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


def cmd_capability_feedback(args) -> int:
    """Every capability prediction's two-stage feedback state (M5)."""
    h = _harness(args)
    try:
        result = h.capability_feedback_summary()
        if args.prediction:
            single = h.get_capability_evolution_prediction(args.prediction)
            if single is None:
                return _fail(f"unknown prediction_id {args.prediction!r}")
            return _emit(
                {"prediction": single.to_dict(),
                 "binding": h.capability_maintenance_binding(
                     args.prediction),
                 "evaluation": h.capability_effect_evaluation(
                     args.prediction)},
                f"Prediction {args.prediction}: status {single.status}, "
                f"operation {single.candidate_operation.operation_type}.")
        return _emit(result,
                     f"{result['n_predictions']} capability prediction(s): "
                     f"{result['n_fact_bound']} with a bound maintenance "
                     f"fact, {result['n_effect_verified']} with a VERIFIED "
                     "effect. A bound fact says the operation happened; only "
                     "a verified effect says real later performance moved.")
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
    p.add_argument("--action", required=True, metavar="ACTION_ID|EXECUTION_ID",
                   help="the action id (ac_...) OR the execution id (ex_...) "
                        "the action produced — the latter is what `orx "
                        "execute` prints, so no separate lookup is needed")
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
             "summary (closed episodes only; read-only)")
    p.add_argument("--min-samples", type=int, default=None,
                   help="override the minimum-sample threshold for this "
                        "read (the effective value is recorded on the "
                        "summary)")
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser(
        "evaluations",
        help="list stored post-hoc evaluations of strategy-outcome "
             "predictions (read-only)")
    p.add_argument("--task", default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--evaluation", default=None, metavar="EVALUATION_ID",
                   help="read ONE evaluation instead of listing")
    p.set_defaults(func=cmd_evaluations)

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

    p = sub.add_parser(
        "capability-feedback",
        help="the two-stage feedback state of every capability prediction "
             "(M5): fact bound vs effect verified, read-only")
    p.add_argument("--prediction", default=None,
                   help="read ONE prediction's full state instead of "
                        "summarizing")
    p.set_defaults(func=cmd_capability_feedback)

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
