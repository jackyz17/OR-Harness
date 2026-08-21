"""Tests for Episode store and two-phase recording (Phase 2 step 6)."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from helpers import build_system
from or_experience_bank.core.episode import EpisodeStore, build_episode
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter


class EpisodeStoreTest(unittest.TestCase):
    def test_base_and_supplement_appended(self):
        with tempfile.TemporaryDirectory() as temp:
            store = EpisodeStore(Path(temp))
            episode = build_episode("prob text", "prob_1", {"problem_family": "assignment"}, [], 2, "success")
            store.record_episode(episode)
            store.record_gold_supplement("prob_1", gold=7.0, matched=True, produced_realization_ids=["exp_1"])
            stats = store.stats()
            self.assertEqual(stats["base"], 1)
            self.assertEqual(stats["supplements"], 1)
            supplements = store.supplements_for("prob_1")
            self.assertEqual(supplements[0]["gold_answer"], 7.0)
            self.assertTrue(supplements[0]["matched"])

    def test_base_episode_carries_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            store = EpisodeStore(Path(temp))
            spec = {
                "problem_family": "cvrp",
                "structural_signature": {
                    "objective": "linear", "decision": ["binary_assignment"],
                    "constraint": ["capacity"], "interaction": "shared_resource_coupled",
                    "features": {"network": "path_on_graph"},
                },
            }
            episode = build_episode("p", "prob_2", spec, [], 0, "success")
            store.record_episode(episode)
            base = store.base_episodes()[0]
            self.assertEqual(base["structural_signature"]["features"]["network"], "path_on_graph")


class EpisodeIntegrationTest(unittest.TestCase):
    def test_solve_records_episode_and_gold_supplement(self):
        adapters = [MockSolverAdapter("gurobi", [
            SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0,
                                  objective_sense="minimize", objective_value=7, variables={"x": 1}),
        ], delay=0.01)]
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, llm = build_system(temp, adapters, ["# c1"])
            asyncio.run(orchestrator.solve("Assign tasks minimize cost", ["gurobi"], defer_extraction=True))
            asyncio.run(orchestrator.evaluate_with_gold(gold=7))
            stats = orchestrator.episode_store.stats()
            self.assertEqual(stats["base"], 1)
            self.assertEqual(stats["supplements"], 1)
            base = orchestrator.episode_store.base_episodes()[0]
            self.assertIn("verified_model", base["normalized_spec"])


if __name__ == "__main__":
    unittest.main()
