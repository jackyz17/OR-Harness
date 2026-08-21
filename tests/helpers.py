from __future__ import annotations

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = SKILL_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from or_experience_bank.core.schemas import (
    ExperienceEvidence,
    ExperiencePolicy,
    ExperienceRecord,
    ExperienceScope,
    ExperienceTrigger,
    ProblemContext,
)


def build_system(home, adapters, text_responses):
    from or_experience_bank.config import ExperienceBankConfig
    from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
    from or_experience_bank.llm_client import FakeLLMClient
    from or_experience_bank.solving.orchestrator import ORExperienceOrchestrator
    from or_experience_bank.retrieval.retrieval import ExperienceRetriever
    from or_experience_bank.solvers.registry import SolverRegistry
    from or_experience_bank.core.store import AppendOnlyExperienceStore

    config = ExperienceBankConfig(bank_home=Path(home), max_attempts_per_branch=3)
    config.ensure_directories()
    store = AppendOnlyExperienceStore(Path(home))
    index = EmbeddingIndex(Path(home) / "index", LocalHashEmbeddingBackend(128))
    retriever = ExperienceRetriever(store, index)
    registry = SolverRegistry(adapters)

    # The structured-modeling stage (Phase 1, module 1.1) consumes the FIRST text
    # response (a <think>/<model> output) and the FIRST object response (a signature
    # JSON) before any branch code is generated. Prepend a passing model + signature
    # so existing branch-level text_responses stay aligned with branch attempts.
    modeling_output = (
        "<think>analysis</think>\n<model>"
        "SETS\n  i in Items = {a, b}\n"
        "PARAMETERS\n  cost[i]\n  cap\n"
        "VARIABLES\n  x[i] >= 0, continuous\n"
        "OBJECTIVE\n  minimize sum_i cost[i] * x[i]\n"
        "CONSTRAINTS\n  C1: sum_i x[i] <= cap"
        "</model>"
    )
    signature_object = {
        "objective": "linear",
        "decision": ["continuous_flow"],
        "constraint": ["capacity"],
        "interaction": "shared_resource_coupled",
        "features": {},
    }
    # Comparative synthesis (gold match) consumes one object = a candidate list; the
    # admission judge consumes one object per candidate = {"accept": True}. Provide a
    # default modeling candidate plus generous judge accepts so deferred extraction works.
    synthesis_candidates = [[{
        "layer": "modeling",
        "title": "Keep solver-independent semantics",
        "retrieval_text": "preserve formulation semantics across solvers",
        "polarity": "positive",
        "diagnosis": "branches consistent",
        "action": "define a solver-independent formulation contract",
        "rationale": "consistent objectives",
    }]]
    judge_accepts = [{"accept": True}] * 8
    # object consumption order: (1) signature, (2) modeling-stage L3 semantic judge
    # (empty list = clean model), then (3) synthesis candidates, (4) admission judge(s).
    llm = FakeLLMClient(
        text_responses=[modeling_output] + list(text_responses),
        object_responses=[signature_object, []] + synthesis_candidates + judge_accepts,
    )
    orchestrator = ORExperienceOrchestrator(config, store, retriever, registry, llm)
    return orchestrator, store, retriever, llm


def experience(
    title="Capacity constraints must cover all assigned demand",
    layer="modeling",
    solver=None,
    solver_family=None,
    generality="solver_agnostic",
    polarity="positive",
    error=None,
    problem_family="assignment",
):
    api = {"gurobi": "gurobipy", "scip": "pyscipopt", "ortools": "ortools.cp_model"}.get(solver)
    return ExperienceRecord(
        layer=layer,
        polarity=polarity,
        title=title,
        retrieval_text=title + ". Apply this rule when building or repairing an OR model.",
        problem_context=ProblemContext(
            problem_family=problem_family,
            objective_type="minimize",
            stage={"modeling": "formulation", "implementation": "implementation", "repair": "repair", "solving": "solving"}[layer],
            keywords=[problem_family, "capacity"],
        ),
        scope=ExperienceScope(
            generality=generality,
            solver_family=solver_family,
            solver=solver,
            language="python",
            api=api if generality == "solver_specific" else None,
        ),
        trigger=ExperienceTrigger(
            situation="when capacity or assignment constraints are generated",
            normalized_error=error,
            solver_status="error" if error else None,
        ),
        policy=ExperiencePolicy(
            diagnosis="The previous formulation or implementation omitted a required relationship.",
            action="Add the explicit capacity-to-assignment constraint before solving.",
            rationale="The constraint prevents assignments from exceeding available capacity.",
        ),
        evidence=ExperienceEvidence(
            problem_id="prob_test",
            branch_ids=["branch_test"],
            attempt_ids=["attempt_test"],
            solver_feedback_summary="validated test evidence",
            validation_level="solver_feasible",
            causal_confidence="high",
        ),
    )
