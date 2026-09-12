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

from or_harness.api import ORHarness
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


def _harness(args) -> ORHarness:
    weights = None
    if getattr(args, "cost_weights", None):
        weights = {k: float(v) for k, v in
                   (pair.split("=") for pair in args.cost_weights.split(","))}
    return ORHarness(home=args.home, alpha=args.alpha, beta=args.beta,
                     gamma=args.gamma, cost_weights=weights)


def _summarize_recall(result: Dict[str, Any]) -> str:
    recs = result["recommendations"]
    if not recs:
        return "No applicable strategies."
    top = recs[0]
    parts = [f"Top candidate: {top['strategy_id']} ({top['name']}), "
             f"score {top['score']}, evidence={top['evidence']}, "
             f"E[Q]={top['expected']['quality']}, "
             f"P(fail)={top['expected']['failure_prob']}."]
    if top["risk_warnings"]:
        parts.append("Warnings: " + "; ".join(top["risk_warnings"]))
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
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else None
        cir = None
        if args.cir:
            from or_harness.core.coupling import CouplingAwareIR
            cir_data = _load_json_arg(args.cir)
            cir = CouplingAwareIR.from_dict(cir_data)
        profile = h.profile(task, code, cir=cir)
        report = h.derivation_report(task, code, cir=cir)
        return _emit({"profile": profile.to_dict(), "derivation": report},
                     _summarize_profile(profile, report))
    finally:
        h.close()


def cmd_understand(args) -> int:
    """Pre-model coupling-aware understanding."""
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        result = h.understand(task)
        cir = result.get("cir")
        if cir is None:
            return _emit(result, result.get("message", "No CIR provided."))
        groups = cir.get("coupling_groups") or []
        guidance = result.get("modeling_guidance") or []
        warnings = result.get("cir_warnings") or []
        issues = cir.get("issues") or []
        parts = [f"CIR validated: {len(cir.get('entities', []))} entities, "
                 f"{len(cir.get('decisions', []))} decisions, "
                 f"{len(cir.get('constraints', []))} constraints, "
                 f"{len(cir.get('relations', []))} relations."]
        if issues:
            parts.append(f"Validation issues ({len(issues)}): "
                         + "; ".join(f"[{i['layer']}] {i['code']}" for i in issues[:3])
                         + (" ..." if len(issues) > 3 else ""))
        if guidance:
            parts.append(f"Modeling guidance ({len(guidance)}):")
            for g in guidance:
                parts.append(f"  - [{g['type']}] {g['implication']}")
        else:
            parts.append("No coupling groups detected — the CIR is structurally "
                         "valid but no shared bottleneck or global constraint "
                         "pattern was found.")
        for w in warnings:
            parts.append(f"WARNING: {w['code']}: {w['detail']}")
        return _emit(result, " ".join(parts))
    finally:
        h.close()


def _summarize_profile(profile, report) -> str:
    parts = [f"Profile for {profile.problem_id} (family={profile.family}):"]
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
    return " ".join(parts)


def cmd_recall(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else None
        result = h.recall(task, top=args.top,
                          exclude=args.exclude or [],
                          memory_mode=args.memory_mode, code=code)
        return _emit(result, _summarize_recall(result))
    finally:
        h.close()


def cmd_execute(args) -> int:
    h = _harness(args)
    try:
        task = _load_json_arg(args.task)
        record = h.execute(task, args.strategy, args.code, args.workspace,
                           solver=args.solver,
                           verification_level=args.verification)
        out = {"execution": record.to_dict(),
               "execution_id": record.execution_id}
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
            override = {k: float(v) for k, v in
                        (pair.split("=") for pair in args.override.split(","))}
        result = h.record(record, override=override,
                          retain_reason=args.retain_reason)
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
        conditions = None
        if args.llm_conditions:
            conditions = _load_json_arg(args.llm_conditions)
            if not isinstance(conditions, list):
                return _fail("--llm-conditions must be a JSON list of "
                             "{text, supporting_execution_ids}")
        result = h.induce(strategy_id=args.strategy, all_=args.all,
                          rebuild=args.rebuild, widen=args.widen,
                          tighten=args.tighten, dry_run=args.dry_run,
                          force=args.force, llm_conditions=conditions)
        return _emit(result, _summarize_induce(result, args))
    finally:
        h.close()


def _summarize_induce(result: Dict[str, Any], args) -> str:
    if args.rebuild:
        if result.get("dry_run") or "would_rebuild" in result:
            return (f"Rebuild plan: {result.get('would_rebuild', 0)} cells would "
                    "be re-induced from currently retained evidence. Cold "
                    "archive is preserved. Run without --dry-run to apply.")
        return (f"Re-induced {result.get('rebuilt', 0)} entries from retained "
                "evidence. Exact reconstruction is not a requirement — the "
                "re-induced bank may differ from the previous one.")
    if args.widen:
        return json.dumps(result) if "error" in result else \
            f"Entry {result['widened']} widened to {result['new_scope']}."
    if args.tighten:
        return json.dumps(result) if "error" in result else \
            f"Entry {result['tightened']} tightened to {result['new_scope']}."
    created = [r for r in result.get("results", []) if r.get("created")]
    skipped = [r for r in result.get("results", []) if r.get("skipped")]
    parts = []
    if created:
        parts.append(f"Created/updated {len(created)} entries: "
                     + ", ".join(r.get("created") or r.get("updated", "?")
                                 for r in created))
    if skipped:
        parts.append(f"Skipped {len(skipped)}: " + skipped[0]["skipped"])
    if not parts:
        parts.append("Nothing to induce.")
    return " ".join(parts)


def cmd_inspect(args) -> int:
    h = _harness(args)
    try:
        result = h.inspect(bank=args.bank, task_id=args.task,
                           strategy_id=args.strategy, status=args.status)
        count = result["count"]
        noun = {"experience": "records", "strategic": "entries",
                "archive": "cards"}[result["bank"]]
        return _emit(result, f"{count} {noun} in {result['bank']} bank"
                             + (f" (status={args.status})" if args.status else "")
                             + ". Entry track records show n_predictions, hit_rate, "
                               "calibration_error, and consecutive_misses.")
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
                             "It left the hot store; its tombstone vetoes "
                             "re-induction of the same pattern (revive only with "
                             "--force if the environment has genuinely drifted).")
    finally:
        h.close()


def cmd_doctor(args) -> int:
    h = _harness(args)
    try:
        result = h.doctor()
        avail = [s["name"] for s in result["solvers"] if s["available"]]
        missing = [s["name"] for s in result["solvers"] if not s["available"]]
        return _emit(result,
                     f"Home: {result['home']}. Available solvers: "
                     f"{', '.join(avail) or 'none'}. Missing: "
                     f"{', '.join(missing) or 'none'}. Memory: "
                     f"{result['memory']}.")
    finally:
        h.close()


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
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("understand",
                       help="pre-model coupling-aware understanding (CIR)")
    p.add_argument("--task", required=True, help="task JSON literal or file")
    p.set_defaults(func=cmd_understand)

    p = sub.add_parser("profile", help="build a ProblemProfile for a task")
    p.add_argument("--task", required=True, help="task JSON literal or file")
    p.add_argument("--code", default=None, help="optional solve script for AST derivation")
    p.add_argument("--cir", default=None,
                   help="optional CIR JSON literal/file for CIR↔model cross-check")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("recall", help="recall accumulated experience for a task")
    p.add_argument("--task", required=True)
    p.add_argument("--code", default=None)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--memory-mode", default="cost-aware",
                   choices=["none", "cases", "strategic", "cost-aware"])
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("execute", help="sandbox-execute a solve script")
    p.add_argument("--task", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--code", required=True, help="path to solve.py")
    p.add_argument("--workspace", required=True)
    p.add_argument("--solver", required=True)
    p.add_argument("--verification", default="basic", choices=["basic", "strong"])
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
                   help="cost backfill, e.g. 'llm_tokens=1840,tool_calls=9'")
    p.add_argument("--retain-reason", default=None,
                   help="explicitly mark this episode as representative "
                        "evidence (reserved for future compaction policies), "
                        "e.g. 'contrast'; when omitted, any mark already on "
                        "the record is preserved")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("induce", help="consolidate facts into strategic entries")
    p.add_argument("--strategy", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--widen", default=None, metavar="ENTRY_ID")
    p.add_argument("--tighten", default=None, metavar="ENTRY_ID")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="override cold-archive veto (environment drift)")
    p.add_argument("--llm-conditions", default=None,
                   help="JSON list of {text, supporting_execution_ids} phrased by you")
    p.set_defaults(func=cmd_induce)

    p = sub.add_parser("inspect", help="query the memory layers")
    p.add_argument("--bank", default="experience",
                   choices=["experience", "strategic", "archive"])
    p.add_argument("--task", default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--status", default=None)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("gc", help="dispose of the derived layer (harness's call)")
    p.add_argument("--mode", default="compact", choices=["compact", "purge"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("retire", help="move an entry to the cold archive (explicit)")
    p.add_argument("--entry", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_retire)

    p = sub.add_parser("doctor", help="environment self-check")
    p.set_defaults(func=cmd_doctor)
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
