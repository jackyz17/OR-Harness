"""Tests for the derived Repair-Bank error-transition graph (Decision D16)."""

import unittest

from or_experience_bank.retrieval.error_transition_graph import ErrorTransitionGraph
from or_experience_bank.core.schemas import ExperienceGenerality


def _build_graph() -> ErrorTransitionGraph:
    graph = ErrorTransitionGraph()
    # gurobi: errorA --fix_a--> errorB --fix_b--> success (a chain)
    graph.add_transition("gurobi", "Error A", "fix_a", "error", next_error="Error B")
    graph.add_transition("gurobi", "Error B", "fix_b", "optimal")
    # scip (same milp family): family-level repair for Error A
    graph.add_transition(
        "scip", "Error A", "family_fix", "feasible",
        generality=ExperienceGenerality.SOLVER_FAMILY.value,
    )
    # agnostic Python-level repair visible to all solvers
    graph.add_transition(
        "ortools", "Error A", "agnostic_fix", "feasible",
        generality=ExperienceGenerality.SOLVER_AGNOSTIC.value,
    )
    # ortools-specific fix must NOT leak to gurobi
    graph.add_transition("ortools", "Error A", "ortools_only_fix", "optimal")
    return graph


class ErrorTransitionGraphTest(unittest.TestCase):
    def test_specific_edges_rank_first(self):
        graph = _build_graph()
        results = graph.query("gurobi", "Error B")
        self.assertTrue(results)
        self.assertEqual(results[0]["action"], "fix_b")
        self.assertEqual(results[0]["source_solver"], "gurobi")

    def test_family_migration_within_milp(self):
        graph = _build_graph()
        actions = [r["action"] for r in graph.query("gurobi", "Error A")]
        # gurobi is milp; scip's family_fix should migrate to gurobi.
        self.assertIn("family_fix", actions)

    def test_family_not_crossed_to_other_family(self):
        graph = _build_graph()
        actions = [r["action"] for r in graph.query("ortools", "Error A")]
        # ortools is cp_sat; scip's milp family_fix must NOT migrate here.
        self.assertNotIn("family_fix", actions)

    def test_agnostic_visible_everywhere(self):
        graph = _build_graph()
        for solver in ("gurobi", "scip", "ortools"):
            actions = [r["action"] for r in graph.query(solver, "Error A")]
            self.assertIn("agnostic_fix", actions, solver)

    def test_solver_specific_isolated(self):
        graph = _build_graph()
        actions = [r["action"] for r in graph.query("gurobi", "Error A")]
        self.assertNotIn("ortools_only_fix", actions)

    def test_composite_key_isolation(self):
        graph = _build_graph()
        # gurobi has its own Error A edge (fix_a); querying gurobi must include it,
        # but it must never appear under a different solver's specific tier.
        gurobi_actions = [r["action"] for r in graph.query("gurobi", "Error A")]
        self.assertIn("fix_a", gurobi_actions)

    def test_known_pitfalls(self):
        graph = _build_graph()
        pitfalls = graph.known_pitfalls("gurobi", "Error A")
        self.assertEqual(pitfalls, ["Error B"])

    def test_shortest_repair_path_chains(self):
        graph = _build_graph()
        path = graph.shortest_repair_path("gurobi", "Error A")
        # fix_a leads to Error B, then fix_b succeeds.
        self.assertEqual(path, ["fix_a", "fix_b"])

    def test_success_rate_computed(self):
        graph = ErrorTransitionGraph()
        graph.add_transition("gurobi", "E", "act", "optimal")
        graph.add_transition("gurobi", "E", "act", "optimal")
        graph.add_transition("gurobi", "E", "act", "error")
        result = graph.query("gurobi", "E")[0]
        self.assertAlmostEqual(result["success_rate"], 2 / 3)
        self.assertEqual(result["frequency"], 3)

    def test_rebuild_from_records(self):
        records = [
            {
                "polarity": "positive",
                "scope": {"solver": "gurobi", "generality": "solver_specific"},
                "trigger": {"normalized_error": "Error X"},
                "policy": {"action": "do_x"},
            },
            {
                "polarity": "negative",
                "scope": {"solver": "gurobi", "generality": "solver_specific"},
                "trigger": {"normalized_error": "Error X"},
                "policy": {"action": "do_x"},
            },
        ]
        graph = ErrorTransitionGraph().rebuild(records)
        self.assertEqual(graph.stats()["nodes"], 1)
        result = graph.query("gurobi", "Error X")[0]
        self.assertAlmostEqual(result["success_rate"], 0.5)

    def test_add_record_skips_incomplete(self):
        graph = ErrorTransitionGraph()
        graph.add_record({"scope": {}, "trigger": {}, "policy": {}})  # no solver/error
        self.assertEqual(graph.stats()["nodes"], 0)


if __name__ == "__main__":
    unittest.main()
