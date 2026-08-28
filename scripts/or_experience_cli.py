#!/usr/bin/env python3
"""Command-line entry point for the OR Experience Bank Skill."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Dict

SKILL_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = SKILL_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from or_experience_bank.config import ExperienceBankConfig
from or_experience_bank.core.lifecycle import LifecycleStore
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.core.utility_tracker import UtilityTracker
from or_experience_bank.retrieval.index import EmbeddingIndex, create_embedding_backend
from or_experience_bank.llm_client import CommandLLMClient, FakeLLMClient
from or_experience_bank.solving.orchestrator import NoSolverAvailable, ORExperienceOrchestrator
from or_experience_bank.retrieval.modeling_retriever import ModelingRetriever
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.core.schemas import ExperienceRecord, SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter
from or_experience_bank.solvers.registry import SolverRegistry
from or_experience_bank.core.store import AppendOnlyExperienceStore


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="or-experience-bank")
    root.add_argument("--config", help="YAML or JSON config file")
    commands = root.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve")
    problem_group = solve.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem")
    problem_group.add_argument("--problem-file")
    solve.add_argument("--solvers", default=None)
    solve.add_argument("--max-attempts", type=int)
    solve.add_argument("--auto-append", action=argparse.BooleanOptionalAction, default=None)
    solve.add_argument("--reference-objective", type=float)
    solve.add_argument("--no-utility", action="store_true",
                       help="Skip utility tracking / soft delete / prior recall for this run (debug)")
    solve.add_argument("--llm-command", help="Fixed provider-wrapper command accepting JSON stdin/stdout")
    solve.add_argument("--interactive-llm", action="store_true",
                       help="Harness mode: the framework prints prompts, YOU answer on stdin (you ARE the LLM, D18)")
    solve.add_argument("--mock-demo", action="store_true", help="Use explicit fake LLM and mock solvers")
    solve.add_argument("--json", action="store_true")

    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--layer", required=True, choices=["modeling", "implementation", "repair", "solving"])
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--solver")
    retrieve.add_argument("--solver-family")
    retrieve.add_argument("--generality")
    retrieve.add_argument("--problem-family")
    retrieve.add_argument("--stage")
    retrieve.add_argument("--polarity")
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--min-score", type=float)
    retrieve.add_argument("--json", action="store_true")

    append = commands.add_parser("append")
    append.add_argument("--input", required=True)
    append.add_argument("--json", action="store_true")

    induce = commands.add_parser("induce")
    induce.add_argument("--mock-demo", action="store_true", help="Run induction with fake LLM + stub solver")
    induce.add_argument("--auto", action="store_true", help="Apply the v1 trigger policy (candidate gate + watermark + cooldown) instead of inducing unconditionally")
    induce.add_argument("--min-new-realizations", type=int, default=3, help="Accumulation watermark for --auto")
    induce.add_argument("--max-clusters", type=int, default=None)
    induce.add_argument("--validation-threshold", type=float, default=0.5)
    induce.add_argument("--unseen-task", action="append", default=[], help="Unseen OR task text (repeatable)")
    induce.add_argument("--llm-command", help="Fixed provider-wrapper command accepting JSON stdin/stdout")
    induce.add_argument("--interactive-llm", action="store_true",
                        help="Harness mode: the framework prints prompts, YOU answer on stdin (you ARE the LLM, D18)")
    induce.add_argument("--json", action="store_true")

    for name in ("stats", "rebuild-index", "validate-bank"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true")
    return root


def components(config: ExperienceBankConfig):
    """Assemble the full wired stack (Phase 2.3 + 4.1 add-ons included by default).

    UtilityTracker / LifecycleStore / ModelingRetriever are injected here so the
    whole utility chain (retrieval counting -> gold-credited utility -> soft
    delete -> anti-resurrection) and planning-prior recall are active in every
    CLI-driven run, not just in tests that wire them manually.
    """
    config.ensure_directories()
    backend = create_embedding_backend(config.retrieval_backend, config.embedding_model)
    index = EmbeddingIndex(config.bank_home / "index", backend)
    utility_tracker = UtilityTracker(config.bank_home)
    lifecycle = LifecycleStore(config.bank_home)
    store = AppendOnlyExperienceStore(
        config.bank_home,
        lifecycle=lifecycle,
        embed=backend.embed_documents,
    )
    retriever = ExperienceRetriever(store, index, utility_tracker=utility_tracker, lifecycle=lifecycle)
    return store, retriever, utility_tracker, lifecycle


def episode_family_index(bank_home: Path) -> Dict[str, str]:
    """episode_id -> problem_family, resolved from recorded Episode provenance.

    Feeds SignatureClusterer so cross-family detection uses the orchestrator's
    normalized family (authoritative) instead of keyword guessing.
    """
    from or_experience_bank.core.episode import EpisodeStore

    index: Dict[str, str] = {}
    for record in EpisodeStore(bank_home).iter_records():
        spec = record.get("normalized_spec") or {}
        family = spec.get("problem_family")
        episode_id = record.get("episode_id")
        if family and episode_id:
            index[episode_id] = family
    return index


def emit(value, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


async def solve_command(args, config, store, retriever, utility_tracker, lifecycle):
    problem = args.problem if args.problem is not None else Path(args.problem_file).read_text(encoding="utf-8")
    names = [x.strip() for x in args.solvers.split(",")] if args.solvers else config.solvers
    if args.mock_demo:
        if not args.solvers:
            names = ["mock-a", "mock-b"]
        first = MockSolverAdapter(
            "mock-a",
            [
                SolverExecutionResult(status="error", solver="mock-a", exit_code=1, normalized_error="TypeError: invalid linear expression"),
                SolverExecutionResult(status="optimal", solver="mock-a", exit_code=0, objective_sense="minimize", objective_value=7.0, variables={"x": 1}),
            ],
            delay=0.02,
        )
        second = MockSolverAdapter(
            "mock-b",
            [SolverExecutionResult(status="optimal", solver="mock-b", exit_code=0, objective_sense="minimize", objective_value=7.0, variables={"x": 1})],
            delay=0.02,
        )
        registry = SolverRegistry([first, second])
        llm = FakeLLMClient(["# mock attempt one", "# mock branch two", "# repaired mock attempt"])
    else:
        registry = SolverRegistry()
        if args.interactive_llm:
            from or_experience_bank.llm_client import StdinLLMClient
            llm = StdinLLMClient()  # harness mode: YOU answer the prompts on stdin (D18)
        elif args.llm_command:
            llm = CommandLLMClient(shlex.split(args.llm_command))
        else:
            raise RuntimeError(
                "Real solve needs an LLM source: use --interactive-llm inside a harness "
                "environment (the framework prints prompts and you answer them — you ARE "
                "the LLM, D18), or --llm-command with an external wrapper for standalone runs."
            )
    modeling_retriever = None
    if utility_tracker is not None or lifecycle is not None:
        modeling_retriever = ModelingRetriever(
            ModelingStore(config.bank_home), lifecycle=lifecycle
        )
    orchestrator = ORExperienceOrchestrator(
        config, store, retriever, registry, llm,
        modeling_retriever=modeling_retriever,
        utility_tracker=utility_tracker,
    )
    result = await orchestrator.solve(
        problem, names, args.max_attempts, args.auto_append, args.reference_objective
    )
    return result.to_dict()

async def induce_command(args, config):
    """Offline structural induction: Realization -> Pattern -> Repository (Phase 3)."""
    from or_experience_bank.core.modeling_store import ModelingStore
    from or_experience_bank.induction import (
        CounterexampleSearcher,
        InductionPipeline,
        InductionTrigger,
        LLMBackedAligner,
        LLMBackedCounterexampleSearcher,
        LLMBackedInducer,
        PatternInducer,
        PatternValidator,
        SignatureClusterer,
        StructuralAligner,
    )

    store = ModelingStore(config.bank_home)
    workspace = config.bank_home / "induction_ws"
    workspace.mkdir(parents=True, exist_ok=True)
    clusterer = SignatureClusterer(
        episode_family_index=episode_family_index(config.bank_home)
    )
    trigger = (
        InductionTrigger(store, clusterer, min_new_realizations=args.min_new_realizations)
        if args.auto else None
    )

    if args.mock_demo:
        llm = FakeLLMClient(
            text_responses=["print('refutation')"],
            object_responses=[[], [], []],  # alignment/hypotheses/conditions: empty in bare demo
        )
        executor = None

        async def transfer_solver(task, principle):
            return 8.0 if principle is not None else 12.0
    else:
        from or_experience_bank.solving.execution import SafePythonExecutor
        if args.interactive_llm:
            from or_experience_bank.llm_client import StdinLLMClient
            llm = StdinLLMClient()  # harness mode: YOU answer the prompts on stdin (D18)
        elif args.llm_command:
            llm = CommandLLMClient(shlex.split(args.llm_command))
        else:
            raise RuntimeError(
                "Real induce needs an LLM source: use --interactive-llm inside a harness "
                "environment (the framework prints prompts and you answer them — you ARE "
                "the LLM, D18), or --llm-command with an external wrapper for standalone runs."
            )
        executor = SafePythonExecutor()

        async def transfer_solver(task, principle):
            raise RuntimeError("transfer solver must be supplied by the harness")

    pipeline = InductionPipeline(
        store=store,
        clusterer=clusterer,
        aligner=LLMBackedAligner(StructuralAligner(), llm),
        inducer=LLMBackedInducer(PatternInducer(), llm),
        counterexample=LLMBackedCounterexampleSearcher(
            CounterexampleSearcher(executor=executor), llm
        ),
        validator=PatternValidator(validation_threshold=args.validation_threshold),
        transfer_solver=transfer_solver,
        workspace=workspace,
        unseen_tasks=list(args.unseen_task),
        max_clusters=args.max_clusters,
        trigger=trigger,
    )
    report = await pipeline.run()
    return report.to_dict()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    try:
        config = ExperienceBankConfig.load(args.config)
        store, retriever, utility_tracker, lifecycle = components(config)
        if args.command == "solve" and getattr(args, "no_utility", False):
            # Bare stack for debugging: drop utility/lifecycle/prior-recall add-ons.
            store = AppendOnlyExperienceStore(config.bank_home)
            backend = create_embedding_backend(config.retrieval_backend, config.embedding_model)
            retriever = ExperienceRetriever(store, EmbeddingIndex(config.bank_home / "index", backend))
            utility_tracker = None
            lifecycle = None
        if args.command == "retrieve":
            filters = {
                key: getattr(args, key)
                for key in ("solver", "solver_family", "generality", "problem_family", "stage", "polarity")
                if getattr(args, key) is not None
            }
            hits = retriever.retrieve(args.layer, args.query, filters, args.top_k, args.min_score)
            output = [hit.__dict__ for hit in hits]
        elif args.command == "append":
            record = ExperienceRecord.from_dict(json.loads(Path(args.input).read_text(encoding="utf-8")))
            result = store.append(record)
            if result.status == "appended":
                retriever.rebuild(result.layer)
            output = result.__dict__
        elif args.command == "stats":
            output = store.stats()
            output["index"] = {
                layer: retriever.index.load(layer).get("model_id")
                for layer in ("modeling", "implementation", "repair", "solving")
            }
        elif args.command == "rebuild-index":
            output = {"rebuilt": retriever.rebuild()}
        elif args.command == "validate-bank":
            output = store.validate_bank()
            output["index"] = retriever.validate_indexes()
            output["valid"] = output["valid"] and output["index"]["valid"]
        elif args.command == "solve":
            output = asyncio.run(solve_command(args, config, store, retriever, utility_tracker, lifecycle))
        elif args.command == "induce":
            output = asyncio.run(induce_command(args, config))
        else:
            raise RuntimeError("Unsupported command")
        emit(output, args.json)
        return 0
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError, NoSolverAvailable) as exc:
        error = {"error": type(exc).__name__, "message": str(exc)}
        emit(error, getattr(args, "json", False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
