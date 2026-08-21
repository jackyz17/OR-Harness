from __future__ import annotations

import unittest

from helpers import SRC_DIR
from or_experience_bank.retrieval.query_builder import StageAwareQueryBuilder


class QueryBuilderTests(unittest.TestCase):
    def test_stage_queries_include_required_context_and_sanitize(self):
        builder = StageAwareQueryBuilder()
        modeling = builder.modeling(
            {"description": "assign jobs", "problem_family": "assignment", "objective": "minimize", "entities": ["jobs"], "constraints": ["capacity"]},
            "construct assignment constraints",
        )
        self.assertIn("Objective: minimize", modeling)
        self.assertIn("capacity", modeling)
        implementation = builder.implementation("binary assignment", "gurobi", "milp", "gurobipy", "add variables")
        self.assertIn("Solver: gurobi", implementation)
        self.assertIn("API: gurobipy", implementation)
        repair = builder.repair(
            "gurobi", "milp", "TypeError: bad expr", "/Users/alice/private/model.py\nAPI_KEY=secret\n" + "x" * 10000,
            "error", "addConstr", "assignment model",
        )
        self.assertIn("TypeError", repair)
        self.assertNotIn("/Users/alice", repair)
        self.assertNotIn("secret", repair)
        self.assertLess(len(repair), 5000)
        solving = builder.solving("assignment", "large", 100, 200, "gurobi", "timeout", 60.0, 0.2, 10.0, performance_symptom="timeout")
        self.assertIn("MIP gap: 0.2", solving)
        self.assertIn("Runtime: 60.0", solving)

