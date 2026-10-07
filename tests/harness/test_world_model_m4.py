"""M4 maintenance tests: the offline induction-material entry point, and the
capability predictor's frozen evidence scope.

All tests use controlled scripted providers — nothing here claims real-LLM
behaviour.

The candidate BUNDLE generator (``induction-candidates`` /
``induction_candidates()``) was REMOVED: the framework no longer turns a
cell's counts into a candidate, and a claim can no longer cite a bundle. The
capability predictor still consumes a FROZEN evidence scope, but it is read
from the evidence directly (a mapping naming the executions) rather than
produced by a generator. What stays here — and is what M5 depends on — is a
frozen evidence scope.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.strategic_bank import StrategicEntry  # noqa: E402


def _profile(problem_id="t1", family="routing", **coupling):
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    from or_harness.core.schema import ProblemProfile
    return ProblemProfile(
        problem_id=problem_id,
        family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values,
    )


class TestM4InductionMaterial(HarnessTestCase):
    """M4-A: the material reads a batch of REAL evidence with no gate."""

    def test_material_needs_no_independent_evidence(self):
        """The material is visible from the FIRST execution: independent-task
        count is a fact it REPORTS, never a gate on what may be read."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        material = h.induction_material()
        self.assertEqual(material["count"], 1)
        self.assertEqual(material["n_distinct_tasks"], 1)
        self.assertEqual(material["material"][0]["strategy_id"], "S01")

    def test_material_reports_the_facts_a_reader_needs(self):
        """Across two distinct tasks the material reports the real tasks,
        the same-task attempt chains, the outcomes and the existing
        knowledge — with no ``new_claim`` / ``cell_observation`` verdict."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        prof = _profile()
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=prof))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=prof))
        material = h.induction_material()
        self.assertEqual(sorted(material["tasks"]), ["t1", "t2"])
        self.assertEqual(material["n_distinct_tasks"], 2)
        self.assertEqual(sorted(material["task_chains"]), ["t1", "t2"])
        self.assertIn("existing_knowledge", material)
        self.assertNotIn("candidates", material)
        blob = str(material)
        self.assertNotIn("cell_observation", blob)
        self.assertNotIn("bundle_id", blob)

    def test_existing_knowledge_is_offered_for_revision(self):
        """An existing entry matching the batch is surfaced, so the agent can
        extend, merge or revise it instead of creating a near-duplicate."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        entry = StrategicEntry(
            entry_id="",
            strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.5,
            verification={"state": "verified", "claim": "x"})
        h.sbank.add(entry)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        material = h.induction_material()
        ids = {k["entry_id"] for k in material["existing_knowledge"]}
        self.assertIn(entry.entry_id, ids)


class TestRemovedInductionCandidatesCli(HarnessTestCase):
    """The removed candidate command must not linger as a silently-different
    command: an agent that learned it gets a usage error, not a surprise."""

    def _run(self, argv):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home] + argv)
        finally:
            sys.stdout = old
        return code, buffer.getvalue()

    def test_old_candidate_command_is_gone(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(["induction-candidates"])
        self.assertEqual(ctx.exception.code, 2)

    def test_old_review_material_command_is_gone(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(["review-material"])
        self.assertEqual(ctx.exception.code, 2)

    def test_old_assessment_command_is_gone(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(["assess-induction", "--candidates-only"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
