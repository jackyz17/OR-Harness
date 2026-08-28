"""Shared component assembly for orx commands (bank stores, retrievers, executor).

One process per command: components are constructed fresh, all state lives in
the bank directory (persistent) and the run directory (per-solve). Nothing is
kept in memory across commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import ExperienceBankConfig
from ..core.episode import EpisodeStore
from ..core.lifecycle import LifecycleStore
from ..core.modeling_store import ModelingStore
from ..core.store import AppendOnlyExperienceStore
from ..core.utility_tracker import UtilityTracker
from ..retrieval.index import EmbeddingIndex, create_embedding_backend
from ..retrieval.modeling_retriever import ModelingRetriever
from ..retrieval.retrieval import ExperienceRetriever
from ..solving.execution import SafePythonExecutor
from ..solving.validator import ResultValidator

# Solver -> family mapping used for bank metadata filtering. Mirrors the
# solver adapters' `solver_family` attribute without instantiating adapters.
SOLVER_FAMILY = {
    "gurobi": "milp",
    "scip": "milp",
    "highs": "milp",
    "copt": "milp",
    "ortools": "cp_sat",
    "pulp": "milp",
    "pyomo": "milp",
}


@dataclass
class Components:
    config: ExperienceBankConfig
    backend: object
    index: EmbeddingIndex
    utility_tracker: UtilityTracker
    lifecycle: LifecycleStore
    store: AppendOnlyExperienceStore
    retriever: ExperienceRetriever
    modeling_store: ModelingStore
    modeling_retriever: ModelingRetriever
    episode_store: EpisodeStore
    executor: SafePythonExecutor
    validator: ResultValidator


def build_components(config: Optional[ExperienceBankConfig] = None) -> Components:
    config = config or ExperienceBankConfig()
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
    modeling_store = ModelingStore(config.bank_home)
    modeling_retriever = ModelingRetriever(modeling_store, lifecycle=lifecycle, utility_tracker=utility_tracker)
    episode_store = EpisodeStore(config.bank_home)
    executor = SafePythonExecutor(
        timeout_seconds=config.python_timeout_seconds,
        solver_timeout_seconds=config.solver_timeout_seconds,
        max_stdout_chars=config.max_stdout_chars,
        max_stderr_chars=config.max_stderr_chars,
    )
    validator = ResultValidator()
    return Components(
        config=config,
        backend=backend,
        index=index,
        utility_tracker=utility_tracker,
        lifecycle=lifecycle,
        store=store,
        retriever=retriever,
        modeling_store=modeling_store,
        modeling_retriever=modeling_retriever,
        episode_store=episode_store,
        executor=executor,
        validator=validator,
    )


__all__ = ["Components", "build_components", "SOLVER_FAMILY"]
