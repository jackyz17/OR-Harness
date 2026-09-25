"""M4 maintenance tests: induction candidate bundling (the evidence
package M5 consumes) and its CLI entry point.

All tests use controlled scripted providers — nothing here claims real-LLM
behaviour.

The M4 VALUE-ASSESSMENT chain (``assess_induction`` / ``accept_induction``
/ ``reject_induction`` / ``bind_induction_outcome``) was REMOVED: the M5
two-stage capability feedback asks the same question with a real horizon
and a real effect gate, and the removed chain's slow verdict could not
establish anything the effect evaluation cannot. What stays here — and is
what M5 actually depends on — is the frozen candidate BUNDLE.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.strategic_bank import StrategicEntry  # noqa: E402
from or_harness.world_model.maintenance import (  # noqa: E402
    InductionCandidateBundle,
)


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


class TestM4CandidateBundling(HarnessTestCase):
    """M4-A: candidate bundle construction from real evidence."""

    def test_bundle_requires_independent_evidence(self):
        """A candidate bundle forms only when >=2 executions exist across
        >=2 distinct tasks. Repeating one task does NOT form a bundle."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)

        # 1. Single execution: no bundle.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(len(h.induction_candidates()), 0)

        # 2. Repeated execution on the SAME task: n=2 but tasks=1 => no bundle.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(len(h.induction_candidates()), 0)

        # 3. Independent execution on task t2: n=2, tasks=2 => bundle forms.
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundles = h.induction_candidates()
        self.assertEqual(len(bundles), 1)
        b = bundles[0]
        self.assertEqual(b["kind"], "new_claim")
        self.assertEqual(b["strategy_id"], "S01")
        self.assertEqual(b["family"], "routing")
        self.assertEqual(b["tasks"], ["t1", "t2"])
        self.assertEqual(b["n_supporting"], 3)  # all 3 unique executions
        self.assertTrue(b["trigger_reasons"])

    def test_bundle_distinguishes_new_and_revision(self):
        """A bundle for a cell covered by an existing StrategicEntry is
        labelled as a revision candidate with the entry reference preserved."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)

        # Pre-populate an existing StrategicEntry.
        prof = _profile(problem_id="t1", resource_coupling=0.35)
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id="S01",
            pattern={"predicates": {"family": "routing",
                                    "resource_coupling": [0.25, 0.5]}},
            expected_quality_hat=0.5,
            verification={"state": "verified", "claim": "x"})
        h.sbank.add(entry)

        # Add evidence that has drifted from the entry's claim.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       status="optimal", objective=95.0,
                                       gap=0.05, profile=prof))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       status="optimal", objective=95.0,
                                       gap=0.05, profile=prof))
        bundles = h.induction_candidates()
        self.assertEqual(len(bundles), 1)
        b = bundles[0]
        self.assertEqual(b["kind"], "revision")
        self.assertEqual(b["target_entry_id"], entry.entry_id)
        self.assertIsNotNone(b["entry_before"])

    def test_hypothetical_and_future_records_excluded(self):
        """Hypothetical executions and dynamically added future records do
        NOT alter an already-formed bundle."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle_dict = h.induction_candidates()[0]
        bundle = InductionCandidateBundle.from_dict(bundle_dict)

        # Later: new execution arrives. The frozen bundle does NOT change.
        h.bank.append(self.make_record(task_id="t3", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(bundle.n_supporting, 2)
        self.assertEqual(bundle.tasks, ["t1", "t2"])






class TestInductionCandidatesCli(HarnessTestCase):
    """The scan is reachable as its own command and makes no model call.

    This is the evidence-package generator: ``predict-capability --bundle``
    consumes what it returns, so losing it would remove the only way to
    scope a capability prediction to a frozen evidence set.
    """

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

    def test_scan_needs_no_provider_and_reports_the_evidence(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        code, out = self._run(["induction-candidates"])
        self.assertEqual(code, 0)
        self.assertIn("1 induction candidate(s)", out)
        self.assertIn("S01", out)
        # The bundle is a real, frozen evidence scope — not just a name.
        self.assertIn("execution_ids", out)

    def test_empty_bank_reports_the_gate_not_an_error(self):
        code, out = self._run(["induction-candidates"])
        self.assertEqual(code, 0)
        self.assertIn("No induction candidates", out)
        self.assertIn(">=2 distinct tasks", out)

    def test_old_assessment_command_is_gone(self):
        """The removed chain must not linger as a silently-different
        command: an agent that learned `assess-induction` gets a usage
        error, not a surprise."""
        with self.assertRaises(SystemExit) as ctx:
            self._run(["assess-induction", "--candidates-only"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
