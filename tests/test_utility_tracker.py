"""Tests for core/utility_tracker.py (utility stats + soft-delete scoring)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.utility_tracker import UtilityTracker


class CounterTest(unittest.TestCase):
    def test_record_retrieval_and_utility(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp))
            t.record_retrieval("exp_a")
            t.record_retrieval("exp_a")
            t.record_utility("exp_a")
            self.assertEqual(t.retrieval_count("exp_a"), 2)
            self.assertEqual(t.utility_count("exp_a"), 1)
            self.assertAlmostEqual(t.utility_ratio("exp_a"), 0.5)

    def test_ratio_none_when_never_retrieved(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp))
            self.assertIsNone(t.utility_ratio("exp_new"))

    def test_batch_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp))
            t.record_retrievals(["a", "a", "b"])
            t.record_utilities(["a"])
            self.assertEqual(t.retrieval_count("a"), 2)
            self.assertEqual(t.retrieval_count("b"), 1)
            self.assertEqual(t.utility_count("a"), 1)

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            UtilityTracker(Path(tmp)).record_retrieval("exp_p")
            self.assertEqual(UtilityTracker(Path(tmp)).retrieval_count("exp_p"), 1)


class SoftDeleteTest(unittest.TestCase):
    def test_new_experience_protected_by_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp), alpha=5, beta=0.1)
            # retrieved 3 times, 0 utility -> ratio 0, but below alpha=5 grace window
            t.record_retrievals(["exp_new"] * 3)
            self.assertFalse(t.is_low_utility("exp_new"))
            self.assertEqual(t.score_multiplier("exp_new"), 1.0)

    def test_low_utility_after_grace_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp), alpha=5, beta=0.1, penalty=0.3)
            # 10 retrievals, 0 utility -> ratio 0 < beta, freq >= alpha -> low utility
            t.record_retrievals(["exp_stale"] * 10)
            self.assertTrue(t.is_low_utility("exp_stale"))
            self.assertEqual(t.score_multiplier("exp_stale"), 0.3)
            self.assertAlmostEqual(t.apply_penalty("exp_stale", 0.9), 0.27)

    def test_useful_experience_not_penalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp), alpha=5, beta=0.1)
            t.record_retrievals(["exp_good"] * 10)
            t.record_utilities(["exp_good"] * 5)  # ratio 0.5 > beta
            self.assertFalse(t.is_low_utility("exp_good"))
            self.assertEqual(t.score_multiplier("exp_good"), 1.0)

    def test_constructor_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                UtilityTracker(Path(tmp), alpha=0)
            with self.assertRaises(ValueError):
                UtilityTracker(Path(tmp), beta=0)
            with self.assertRaises(ValueError):
                UtilityTracker(Path(tmp), penalty=1.5)

    def test_stats_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = UtilityTracker(Path(tmp), alpha=2, beta=0.5)
            t.record_retrievals(["a"] * 3)          # low utility (0/3)
            t.record_retrievals(["b"] * 3)
            t.record_utilities(["b"] * 3)           # useful (3/3)
            summary = t.stats()
            self.assertEqual(summary["tracked"], 2)
            self.assertEqual(summary["low_utility"], 1)


if __name__ == "__main__":
    unittest.main()
