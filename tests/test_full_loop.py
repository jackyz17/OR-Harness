"""End-to-end: failure -> outer reflection -> re-solve -> gold match -> synthesis append.

Walks the full Phase 1+2 loop with mock solvers and a scripted FakeLLM. Verifies that
failures are buffered, a gold mismatch triggers reflection, a recovered solve matches
gold, and comparative synthesis candidates are admitted to the correct banks.
"""

import asyncio
import tempfile
import unittest

from or_experience_bank.config import ExperienceBankConfig
from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.llm_client import FakeLLMClient
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.solving.orchestrator import ORExperienceOrchestrator
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter
from or_experience_bank.core.store import AppendOnlyExperienceStore
from pathlib import Path


MODEL_OUT = (
    "<think>a</think>\n<model>SETS\n  i in Items = {a, b}\n"
    "PARAMETERS\n  cost[i]\n  cap\nVARIABLES\n  x[i] >= 0, continuous\n"
    "OBJECTIVE\n  minimize sum_i cost[i] * x[i]\nCONSTRAINTS\n  C1: sum_i x[i] <= cap</model>"
)
SIG = {"objective": "linear", "decision": ["continuous_flow"], "constraint": ["capacity"],
       "interaction": "shared_resource_coupled", "features": {}}
SYNTH = [{"layer": "modeling", "title": "keep semantics", "retrieval_text": "r",
          "polarity": "positive", "diagnosis": "d", "action": "define contract", "rationale": "ok"}]


class FullLoopTest(unittest.TestCase):
    def test_failure_reflection_success_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            # Branch: first solve attempt errors, second (post-reflection re-solve) optimal.
            adapter = MockSolverAdapter("gurobi", [
                SolverExecutionResult(status="error", solver="gurobi", exit_code=1,
                                      normalized_error="TypeError: bad", stderr="TypeError: bad"),
                SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0,
                                      objective_sense="minimize", objective_value=7, variables={"x": 1}),
            ], delay=0.01)

            config = ExperienceBankConfig(bank_home=Path(temp), max_attempts_per_branch=3)
            config.ensure_directories()
            store = AppendOnlyExperienceStore(Path(temp))
            index = EmbeddingIndex(Path(temp) / "index", LocalHashEmbeddingBackend(128))
            retriever = ExperienceRetriever(store, index)
            from or_experience_bank.solvers.registry import SolverRegistry
            registry = SolverRegistry([adapter])

            # text: [solve1 model] [solve1 code] ... [solve2 model] [solve2 code]
            # object: per solve -> signature, L3 judge([]); then synthesis, then judge accepts.
            llm = FakeLLMClient(
                text_responses=[MODEL_OUT, "# code v1", MODEL_OUT, "# code v2"],
                object_responses=[SIG, [], SIG, [], SYNTH] + [{"accept": True}] * 6,
            )
            orch = ORExperienceOrchestrator(config, store, retriever, registry, llm)

            # Round 1: solve -> branch errors -> gold mismatch
            r1 = asyncio.run(orch.solve("Assign tasks minimize cost", ["gurobi"], defer_extraction=True, max_attempts=1))
            self.assertTrue(orch.failures.count() >= 1)  # failure buffered
            v1 = asyncio.run(orch.evaluate_with_gold(gold=7))
            self.assertFalse(v1.matched)

            # Agent drives reflection, then re-solves (fresh solve call).
            prompt = orch.build_reflection_prompt(v1)
            self.assertIn("gold", prompt.lower())

            # Round 2 needs an adapter whose FIRST attempt succeeds (the mock indexes
            # outcomes by attempt number within a solve, not across solves).
            recovered = MockSolverAdapter("gurobi2", [
                SolverExecutionResult(status="optimal", solver="gurobi2", exit_code=0,
                                      objective_sense="minimize", objective_value=7, variables={"x": 1}),
            ], delay=0.01)
            orch.registry.register(recovered)

            r2 = asyncio.run(orch.solve("Assign tasks minimize cost", ["gurobi2"], defer_extraction=True, max_attempts=1))
            v2 = asyncio.run(orch.evaluate_with_gold(gold=7))
            self.assertTrue(v2.matched)
            appended = orch._pending.get("appended_experience_ids", [])
            self.assertTrue(appended)

            # modeling candidate went to the independent ModelingStore.
            self.assertTrue(orch.modeling_store.stats()["total"] >= 1)

            # Episodes recorded for both rounds + gold supplements.
            stats = orch.episode_store.stats()
            self.assertEqual(stats["base"], 2)
            self.assertEqual(stats["supplements"], 2)


if __name__ == "__main__":
    unittest.main()
