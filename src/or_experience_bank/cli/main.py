"""orx — the OR Experience Bank command line for harness agents.

Design (ReAct-oriented, 2026-08):
  - The harness agent IS the orchestrator. It reads SKILL.md, thinks, and calls
    one `orx` command per step — each command is an independent process.
  - All cross-call state lives in the run directory (files + stamps), never in
    a server process. See cli/run_store.py for the layout.
  - stdout is ALWAYS a single compact JSON object (machine-readable observation);
    long content goes to files inside the run directory.

Exit codes: 0 = command succeeded (check "passed"/"consistent"/"status" fields
for semantic outcomes); 2 = chain/precondition error (read "error"); 1 = crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import ExperienceBankConfig
from .components import build_components
from .run_store import RunError


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=False))


def _fail(message: str, code: int = 2) -> None:
    _emit({"error": message})
    raise SystemExit(code)


def _load_config(args: argparse.Namespace) -> ExperienceBankConfig:
    overrides: Dict[str, Any] = {}
    if getattr(args, "bank_home", None):
        overrides["bank_home"] = args.bank_home
    return ExperienceBankConfig.load(
        config_path=getattr(args, "config", None), cli_overrides=overrides
    )


def _run_dir(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "run_dir", None) or Path.cwd())


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="orx",
        description="OR Experience Bank CLI for harness agents (stateless commands, file-based runs)",
    )
    root.add_argument("--config", help="YAML or JSON config file")
    root.add_argument("--bank-home", help="Override bank home directory")
    root.add_argument("--run-dir", help="Run directory (default: current working directory)")
    commands = root.add_subparsers(dest="command", required=True)

    # -- deployment ----------------------------------------------------------
    doctor = commands.add_parser("doctor", help="Environment self-check (run this first)")
    init = commands.add_parser("init", help="Initialize the bank directory structure")
    init.add_argument("--bank-home", help="Bank home to initialize")

    # -- online solve chain ----------------------------------------------------
    recall = commands.add_parser("recall", help="Start a run + fetch planning priors")
    recall.add_argument("--problem-file", required=True, help="Path to the problem text file")
    recall.add_argument("--top-k", type=int, default=5)

    validate = commands.add_parser("validate", help="L1+L2 gate on model.txt -> stamp")
    signature = commands.add_parser("signature", help="Vocabulary gate on signature.json -> stamp")

    hints = commands.add_parser("hints", help="Pull bank hints BEFORE writing solver code")
    hints.add_argument("--solver", required=True)

    solve = commands.add_parser("solve", help="Sandbox-execute branches/<solver>/solve.py")
    solve.add_argument("--solver", required=True,
                       help="One solver (single branch / repair retry) or comma-separated list "
                            "(parallel exploration: 'highs,pulp,scip' runs them concurrently)")

    cross = commands.add_parser("cross-validate", help="Compare >=2 valid branches")
    cross.add_argument("--tolerance", type=float, default=1e-4)

    gold = commands.add_parser("gold", help="Record the gold verdict (user-provided or consistency-only)")
    gold.add_argument("--answer", type=float, default=None, help="User-provided gold objective")
    gold.add_argument("--matched", dest="matched", action="store_true", default=None,
                      help="Explicitly declare match/mismatch (default: auto-compare)")

    append = commands.add_parser("append", help="Admit one experience file to the bank (gold gate)")
    append.add_argument("--file", required=True, help="Path to the experience JSON file")

    episode = commands.add_parser("episode", help="Terminal: record the episode + credit utility")
    episode.add_argument("--gold-answer", type=float, default=None,
                         help="Override gold answer (default: value recorded by `orx gold`)")

    new_round = commands.add_parser("new-round", help="Archive current artifacts for a reflection round")
    status = commands.add_parser("status", help="Where am I, what's next")

    # -- bank ------------------------------------------------------------------
    query = commands.add_parser("query", help="Search any bank layer")
    query.add_argument("--layer", required=True,
                       choices=["modeling", "implementation", "repair", "solving"])
    query.add_argument("--query", required=True)
    query.add_argument("--solver", default=None)
    query.add_argument("--top-k", type=int, default=5)

    show = commands.add_parser("show", help="Fetch one full record by id")
    show.add_argument("--id", required=True, dest="experience_id")

    deprecate = commands.add_parser("deprecate", help="Retire a record (lifecycle flip + cold archive)")
    deprecate.add_argument("--id", required=True, dest="experience_id")
    deprecate.add_argument("--reason", required=True)

    stats = commands.add_parser("stats", help="Bank statistics")

    # -- offline induction --------------------------------------------------------
    trigger = commands.add_parser("trigger", help="Check the induction trigger gates")
    clusters = commands.add_parser("clusters", help="List candidate clusters")

    align = commands.add_parser("align", help="Stamp role alignments (alignment.json)")
    align.add_argument("--cluster", required=True, dest="cluster_id")

    induce = commands.add_parser("induce", help="Stamp hypotheses (hypotheses.json)")
    induce.add_argument("--cluster", required=True, dest="cluster_id")

    refute = commands.add_parser("refute", help="Execute refutations; executor decides verdicts")
    refute.add_argument("--cluster", required=True, dest="cluster_id")

    validate_pattern = commands.add_parser("validate-pattern", help="Stamp transfer evidence (validation.json)")
    validate_pattern.add_argument("--cluster", required=True, dest="cluster_id")

    append_pattern = commands.add_parser("append-pattern", help="Score + append validated patterns (terminal)")
    append_pattern.add_argument("--cluster", required=True, dest="cluster_id")

    return root


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "init":
            from .bank_commands import cmd_init
            _emit(cmd_init(config_path=args.config, bank_home=args.bank_home))
            return 0

        comps = build_components(_load_config(args))

        if args.command == "doctor":
            from .bank_commands import cmd_doctor
            _emit(cmd_doctor(comps))
        elif args.command == "recall":
            from .solve_commands import cmd_recall
            _emit(cmd_recall(comps, _run_dir(args), Path(args.problem_file), top_k=args.top_k))
        elif args.command == "validate":
            from .solve_commands import cmd_validate
            _emit(cmd_validate(comps, _run_dir(args)))
        elif args.command == "signature":
            from .solve_commands import cmd_signature
            _emit(cmd_signature(comps, _run_dir(args)))
        elif args.command == "hints":
            from .solve_commands import cmd_hints
            _emit(cmd_hints(comps, _run_dir(args), args.solver))
        elif args.command == "solve":
            from .solve_commands import cmd_solve, cmd_solve_parallel
            solvers = [s.strip() for s in args.solver.split(",") if s.strip()]
            if len(solvers) > 1:
                # Heterogeneous parallel exploration: all branches concurrently.
                _emit(cmd_solve_parallel(comps, _run_dir(args), solvers))
            else:
                _emit(cmd_solve(comps, _run_dir(args), solvers[0]))
        elif args.command == "cross-validate":
            from .solve_commands import cmd_cross_validate
            _emit(cmd_cross_validate(comps, _run_dir(args), tolerance=args.tolerance))
        elif args.command == "gold":
            from .solve_commands import cmd_gold
            _emit(cmd_gold(comps, _run_dir(args), gold=args.answer, matched=args.matched))
        elif args.command == "append":
            from .solve_commands import cmd_append
            _emit(cmd_append(comps, _run_dir(args), Path(args.file)))
        elif args.command == "episode":
            from .solve_commands import cmd_episode
            _emit(cmd_episode(comps, _run_dir(args), gold_answer=args.gold_answer))
        elif args.command == "new-round":
            from .solve_commands import cmd_new_round
            _emit(cmd_new_round(comps, _run_dir(args)))
        elif args.command == "status":
            from .solve_commands import cmd_status
            _emit(cmd_status(comps, _run_dir(args)))
        elif args.command == "query":
            from .bank_commands import cmd_query
            _emit(cmd_query(comps, args.layer, args.query, solver=args.solver, top_k=args.top_k))
        elif args.command == "show":
            from .bank_commands import cmd_show
            _emit(cmd_show(comps, args.experience_id))
        elif args.command == "deprecate":
            from .bank_commands import cmd_deprecate
            _emit(cmd_deprecate(comps, args.experience_id, args.reason))
        elif args.command == "stats":
            from .bank_commands import cmd_stats
            _emit(cmd_stats(comps))
        elif args.command == "trigger":
            from .induction_commands import cmd_trigger
            _emit(cmd_trigger(comps))
        elif args.command == "clusters":
            from .induction_commands import cmd_clusters
            _emit(cmd_clusters(comps))
        elif args.command == "align":
            from .induction_commands import cmd_align
            _emit(cmd_align(comps, args.cluster_id))
        elif args.command == "induce":
            from .induction_commands import cmd_induce
            _emit(cmd_induce(comps, args.cluster_id))
        elif args.command == "refute":
            from .induction_commands import cmd_refute
            _emit(cmd_refute(comps, args.cluster_id))
        elif args.command == "validate-pattern":
            from .induction_commands import cmd_validate_pattern
            _emit(cmd_validate_pattern(comps, args.cluster_id))
        elif args.command == "append-pattern":
            from .induction_commands import cmd_append_pattern
            _emit(cmd_append_pattern(comps, args.cluster_id))
        else:  # pragma: no cover
            _fail("unknown command: {}".format(args.command))
        return 0
    except RunError as exc:
        _fail(str(exc))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't traceback
        _fail("{}: {}".format(type(exc).__name__, exc), code=1)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
