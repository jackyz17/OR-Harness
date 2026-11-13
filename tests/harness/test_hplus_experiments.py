"""Real integer attempts in separate training/control and frozen test homes."""
from pathlib import Path

from helpers import HarnessTestCase
from or_harness.experiments.hplus import integer_fixture


class TestHplusBuildFreeze(HarnessTestCase):
    def test_equal_initial_libraries_zero_review_random_budget_and_frozen_test(self):
        report=integer_fixture(str(Path(self.home)/'experiment'),seed=7)
        arms=report['arms']
        self.assertEqual(len({a['initial_library_digest'] for a in arms.values()}),1)
        self.assertEqual(arms['hplus']['triggers'],arms['random']['triggers'])
        self.assertEqual(arms['hplus']['triggers'],1)
        self.assertEqual(arms['hplus']['receipts'],1)
        self.assertEqual(arms['nohplus']['zero_signal_reviews'],4)
        self.assertEqual(arms['nohplus']['confirmed_zero_missed_opportunities'],1)
        self.assertEqual(arms['nohplus']['independently_validated_increments'],1)
        for arm in arms.values():
            self.assertEqual(arm['triggers'],arm['receipts'])
            self.assertGreater(arm['induction_wall_s'],0)
            self.assertIsNone(arm['induction_tokens'])
            self.assertIsNone(arm['induction_tool_calls'])
            self.assertEqual(arm['frozen_tests'][0]['objective'],283)
            self.assertTrue(arm['frozen_tests'][0]['library_unchanged'])
            self.assertEqual(arm['frozen_passes'],1)
        for receipt in arms['nohplus']['receipt_details']:
            self.assertNotEqual(receipt['solve_action_id'],receipt['induction_action_id'])
            self.assertTrue(receipt['baseline_snapshot_ref'])
