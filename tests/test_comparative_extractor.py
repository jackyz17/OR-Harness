"""Tests for comparative synthesis extraction (Phase 2 step 3)."""

import asyncio
import unittest

from or_experience_bank.experience.comparative_extractor import ComparativeSynthesisExtractor
from or_experience_bank.experience.failure_buffer import FailureBuffer
from or_experience_bank.llm_client import FakeLLMClient


def _buffer_with_failures() -> FailureBuffer:
    buffer = FailureBuffer("prob_1")
    buffer.add("modeling", "missing OBJECTIVE block")
    buffer.add("branch_code", "gurobi attempt 1 error", normalized_error="TypeError: bad expr", solver="gurobi")
    return buffer


class PromptRoutingTest(unittest.TestCase):
    def test_failures_trigger_comparative_prompt(self):
        extractor = ComparativeSynthesisExtractor(FakeLLMClient(object_responses=[[]]))
        buffer = _buffer_with_failures()
        self.assertTrue(extractor.has_failures(buffer))
        prompt = extractor.build_synthesis_prompt("prob", "success", buffer)
        self.assertIn("CONTRASTING", prompt)
        self.assertIn("TypeError: bad expr", prompt)

    def test_no_failures_use_success_only_prompt(self):
        extractor = ComparativeSynthesisExtractor(FakeLLMClient(object_responses=[[]]))
        buffer = FailureBuffer("prob_1")  # empty
        self.assertFalse(extractor.has_failures(buffer))
        prompt = extractor.build_success_only_prompt("prob", "success")
        self.assertIn("SUCCESSFUL", prompt)


class ParseCandidatesTest(unittest.TestCase):
    def test_bank_classification(self):
        raw = [
            {"layer": "modeling", "title": "t1", "action": "a1", "retrieval_text": "r"},
            {"layer": "repair", "title": "t2", "action": "a2"},
            {"layer": "not_a_layer", "title": "t3", "action": "a3"},  # filtered out
        ]
        candidates = ComparativeSynthesisExtractor().parse_candidates(raw)
        layers = {c["layer"] for c in candidates}
        self.assertEqual(layers, {"modeling", "repair"})

    def test_incomplete_candidates_dropped(self):
        raw = [{"layer": "modeling", "title": "", "action": "a"}, {"layer": "modeling", "title": "t", "action": ""}]
        self.assertEqual(ComparativeSynthesisExtractor().parse_candidates(raw), [])

    def test_non_list_returns_empty(self):
        self.assertEqual(ComparativeSynthesisExtractor().parse_candidates("junk"), [])


class SynthesizeTest(unittest.TestCase):
    def test_no_llm_returns_empty(self):
        result = asyncio.run(ComparativeSynthesisExtractor(None).synthesize("p", [], None))
        self.assertEqual(result, [])

    def test_synthesize_with_failures(self):
        raw = [{"layer": "repair", "title": "fix", "action": "do x", "retrieval_text": "r"}]
        llm = FakeLLMClient(object_responses=[raw])
        result = asyncio.run(
            ComparativeSynthesisExtractor(llm).synthesize("p", [], _buffer_with_failures(), "model")
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["layer"], "repair")

    def test_synthesize_without_failures_success_channel(self):
        raw = [{"layer": "modeling", "title": "m", "action": "do y"}]
        llm = FakeLLMClient(object_responses=[raw])
        result = asyncio.run(
            ComparativeSynthesisExtractor(llm).synthesize("p", [], FailureBuffer("prob"), "model")
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["layer"], "modeling")


if __name__ == "__main__":
    unittest.main()
