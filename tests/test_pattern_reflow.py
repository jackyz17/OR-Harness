"""End-to-end Phase 4.1: pattern reflow into online solve + utility attribution."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from or_experience_bank.config import ExperienceBankConfig
from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.llm_client import FakeLLMClient
from or_experience_bank.core.lifecycle import LifecycleStore
from or_experience_bank.core.modeling_schemas import (
    ModelingExperience,
)
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.core.store import AppendOnlyExperienceStore
from or_experience_bank.core.utility_tracker import UtilityTracker
from or_experience_bank.retrieval.modeling_retriever import ModelingRetriever
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.solvers.mock import MockSolverAdapter
from or_experience_bank.solvers.registry import SolverRegistry
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solving.orchestrator import ORExperienceOrchestrator

# think explicitly cites the injected principle P1 -> precise utility attribution.
MODEL_OUT = (
    "<think>shared scarce resource; the marginal-contribution principle applies [uses E1]</think>\n"
    "<model>SETS\n  i in Items = {a, b}\n"
    "PARAMETERS\n  cost[i]\n  cap\nVARIABLES\n  x[i] >= 0, continuous\n"
    "OBJECTIVE\n  minimize sum_i cost[i] * x[i]\nCONSTRAINTS\n  C1: sum_i x[i] <= cap</model>"
)
SIG = {"objective": "linear", "decision": ["continuous_flow"], "constraint": ["capacity"],
       "interaction": "shared_resource_coupled", "features": {}}
SYNTH = [{"layer": "modeling", "title": "keep semantics", "retrieval_text": "r",
          "polarity": "positive", "diagnosis": "d", "action": "define contract", "rationale": "ok"}]


def make_pattern():
    rec = ModelingExperience(
        title="resource allocation principle",
        retrieval_text="prioritize higher marginal contribution on shared scarce resource",
        status="validated",
        modeling_aspect="constraint",
    )
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict(SIG)
    rec.experience_id = "p_marginal"
    rec.compute_content_hash()
    return rec


class PatternReflowEndToEndTest(unittest.TestCase):
    def test_cited_pattern_gets_utility_on_gold_match(self):
        with tempfile.TemporaryDirectory() as temp:
            # seed the modeling bank with one validated pattern
            mstore = ModelingStore(Path(temp))
            mstore.append(make_pattern())

            adapter = MockSolverAdapter("gurobi", [
                SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0,
                                      objective_sense="minimize", objective_value=7,
                                      variables={"x": 1}),
            ], delay=0.01)
            config = ExperienceBankConfig(bank_home=Path(temp), max_attempts_per_branch=3)
            config.ensure_directories()
            store = AppendOnlyExperienceStore(Path(temp))
            index = EmbeddingIndex(Path(temp) / "index", LocalHashEmbeddingBackend(128))
            retriever = ExperienceRetriever(store, index)
            registry = SolverRegistry([adapter])

            lifecycle = LifecycleStore(Path(temp))
            utility = UtilityTracker(Path(temp))
            modeling_retriever = ModelingRetriever(
                mstore,
                EmbeddingIndex(Path(temp) / "index" / "modeling_bank", LocalHashEmbeddingBackend(128)),
                lifecycle,
            )
            modeling_retriever.rebuild()

            llm = FakeLLMClient(
                text_responses=[MODEL_OUT, "# code v1"],
                object_responses=[SIG, [], SYNTH] + [{"accept": True}] * 6,
            )
            orch = ORExperienceOrchestrator(
                config, store, retriever, registry, llm,
                modeling_retriever=modeling_retriever,
                utility_tracker=utility,
            )

            result = asyncio.run(orch.solve(
                "Assign scarce resource among tasks to minimize cost",
                ["gurobi"], defer_extraction=True, max_attempts=1,
            ))
            # priors were injected into the modeling prompt
            modeling_prompt = llm.prompts[0]
            self.assertIn("Past modeling experiences", modeling_prompt)
            self.assertIn("[E1]", modeling_prompt)
            # the think cited P1 -> framework parsed it into the pending payload
            self.assertIn("p_marginal", orch._pending["cited_principle_ids"])

            verdict = asyncio.run(orch.evaluate_with_gold(gold=7))
            self.assertTrue(verdict.matched)
            # precise attribution: the cited pattern got utility +1
            self.assertEqual(utility.utility_count("p_marginal"), 1)

    def test_without_wiring_behavior_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = MockSolverAdapter("gurobi", [
                SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0,
                                      objective_sense="minimize", objective_value=7,
                                      variables={"x": 1}),
            ], delay=0.01)
            config = ExperienceBankConfig(bank_home=Path(temp), max_attempts_per_branch=3)
            config.ensure_directories()
            store = AppendOnlyExperienceStore(Path(temp))
            index = EmbeddingIndex(Path(temp) / "index", LocalHashEmbeddingBackend(128))
            retriever = ExperienceRetriever(store, index)
            registry = SolverRegistry([adapter])
            llm = FakeLLMClient(
                text_responses=[MODEL_OUT, "# code v1"],
                object_responses=[SIG, [], SYNTH] + [{"accept": True}] * 6,
            )
            orch = ORExperienceOrchestrator(config, store, retriever, registry, llm)
            result = asyncio.run(orch.solve(
                "Assign tasks minimize cost", ["gurobi"], defer_extraction=True, max_attempts=1,
            ))
            # no priors injected without the retriever wiring
            self.assertNotIn("Past modeling experiences", llm.prompts[0])
            self.assertEqual(orch._pending.get("cited_principle_ids"), [])


if __name__ == "__main__":
    unittest.main()
