"""Tests for the per-solve failure buffer (Phase 2 step 1)."""

import unittest

from or_experience_bank.experience.failure_buffer import FailureBuffer, FailureRecord


class FailureRecordTest(unittest.TestCase):
    def test_valid_stages(self):
        for stage in ("modeling", "branch_code", "reflection"):
            record = FailureRecord(stage=stage, summary="x")
            self.assertEqual(record.stage, stage)

    def test_invalid_stage_rejected(self):
        with self.assertRaises(ValueError):
            FailureRecord(stage="not_a_stage", summary="x")


class FailureBufferTest(unittest.TestCase):
    def test_add_and_group(self):
        buffer = FailureBuffer("prob_1")
        buffer.add("modeling", "missing OBJECTIVE block")
        buffer.add("branch_code", "code error", normalized_error="TypeError: x", solver="gurobi")
        buffer.add("branch_code", "another error", normalized_error="IndexError", solver="scip")
        self.assertEqual(buffer.count(), 3)
        self.assertEqual(len(buffer.by_stage("branch_code")), 2)
        self.assertEqual(len(buffer.by_stage("modeling")), 1)
        self.assertEqual(len(buffer.by_stage("reflection")), 0)

    def test_is_empty_and_clear(self):
        buffer = FailureBuffer("prob_1")
        self.assertTrue(buffer.is_empty())
        buffer.add("modeling", "x")
        self.assertFalse(buffer.is_empty())
        buffer.clear()
        self.assertTrue(buffer.is_empty())

    def test_to_dict_roundtrip_shape(self):
        buffer = FailureBuffer("prob_1")
        buffer.add("reflection", "model invalidated", context={"round": 2})
        data = buffer.to_dict()
        self.assertEqual(data["problem_id"], "prob_1")
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["records"][0]["stage"], "reflection")
        self.assertEqual(data["records"][0]["context"]["round"], 2)


if __name__ == "__main__":
    unittest.main()
