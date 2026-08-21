from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import experience
from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.core.store import AppendOnlyExperienceStore


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        home = Path(self.temp.name)
        self.store = AppendOnlyExperienceStore(home)
        self.index = EmbeddingIndex(home / "index", LocalHashEmbeddingBackend(128))
        self.retriever = ExperienceRetriever(self.store, self.index)

    def tearDown(self):
        self.temp.cleanup()

    def test_layer_scope_topk_increment_and_rebuild(self):
        self.store.append(experience(title="Assignment capacity formulation", layer="modeling"))
        self.store.append(experience(
            title="Gurobi quicksum expression API", layer="implementation", solver="gurobi",
            solver_family="milp", generality="solver_specific",
        ))
        self.store.append(experience(
            title="General index mismatch repair", layer="repair", solver=None,
            solver_family=None, generality="solver_agnostic", error="IndexError",
        ))
        self.retriever.rebuild()
        modeling = self.retriever.retrieve("modeling", "assignment capacity", top_k=1)
        self.assertEqual(len(modeling), 1)
        self.assertEqual(modeling[0].layer, "modeling")
        hidden = self.retriever.retrieve("implementation", "quicksum expression", {"solver": "ortools", "solver_family": "cp_sat"})
        self.assertEqual(hidden, [])
        visible = self.retriever.retrieve("repair", "IndexError index mismatch", {"solver": "ortools", "solver_family": "cp_sat"})
        self.assertEqual(len(visible), 1)
        new = experience(title="Subtour elimination for routing", layer="modeling", problem_family="cvrp")
        self.store.append(new)
        self.retriever.rebuild("modeling")
        hit = self.retriever.retrieve("modeling", "routing subtour elimination", top_k=1)
        self.assertEqual(hit[0].experience_id, new.experience_id)
        self.retriever.rebuild("modeling")
        hit_again = self.retriever.retrieve("modeling", "routing subtour elimination", top_k=1)
        self.assertEqual(hit_again[0].experience_id, new.experience_id)
        self.assertEqual(self.index.load("modeling")["model_id"], "local-hashing-embedding-v1")

