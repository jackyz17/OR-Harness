"""Tests for derived error-transition-graph repair guidance (Option b, D16)."""

import tempfile
import unittest
from pathlib import Path

from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.core.schemas import (
    ExperienceEvidence,
    ExperiencePolicy,
    ExperienceRecord,
    ExperienceScope,
    ExperienceTrigger,
    ProblemContext,
)
from or_experience_bank.core.store import AppendOnlyExperienceStore


def _repair(solver, error, action, generality="solver_specific", solver_family=None, polarity="positive"):
    return ExperienceRecord(
        layer="repair", polarity=polarity, title=action, retrieval_text=action,
        problem_context=ProblemContext(problem_family="assignment", stage="repair"),
        scope=ExperienceScope(generality=generality, solver=solver, solver_family=solver_family, language="python"),
        trigger=ExperienceTrigger(situation=action, normalized_error=error),
        policy=ExperiencePolicy(diagnosis="d", action=action, rationale="r"),
        evidence=ExperienceEvidence(problem_id="p", branch_ids=["b"], validation_level="solver_feasible"),
    )


class RepairGuidanceTest(unittest.TestCase):
    def _retriever(self, temp):
        store = AppendOnlyExperienceStore(Path(temp))
        index = EmbeddingIndex(Path(temp) / "index", LocalHashEmbeddingBackend(128))
        return store, ExperienceRetriever(store, index)

    def test_guidance_rebuilt_on_demand(self):
        with tempfile.TemporaryDirectory() as temp:
            store, retriever = self._retriever(temp)
            store.append(_repair("gurobi", "TypeError: bad expr", "use linear expr"))
            store.append(_repair("scip", "TypeError: bad expr", "family fix", generality="solver_family", solver_family="milp"))
            guidance = retriever.repair_guidance("gurobi", "TypeError: bad expr")
            actions = [a["action"] for a in guidance["actions"]]
            self.assertIn("use linear expr", actions)      # solver_specific
            self.assertIn("family fix", actions)            # same milp family

    def test_guidance_empty_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            _, retriever = self._retriever(temp)
            guidance = retriever.repair_guidance("gurobi", "anything")
            self.assertEqual(guidance["actions"], [])
            self.assertEqual(guidance["repair_path"], [])

    def test_family_not_leaked_across(self):
        with tempfile.TemporaryDirectory() as temp:
            store, retriever = self._retriever(temp)
            store.append(_repair("scip", "Err", "milp only fix", generality="solver_family", solver_family="milp"))
            guidance = retriever.repair_guidance("ortools", "Err")  # ortools is cp_sat
            actions = [a["action"] for a in guidance["actions"]]
            self.assertNotIn("milp only fix", actions)


if __name__ == "__main__":
    unittest.main()
