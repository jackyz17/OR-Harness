"""Tests for the structured-modeling stage and its orchestrator integration (module 1.1)."""

import asyncio
import unittest

from or_experience_bank.llm_client import FakeLLMClient
from or_experience_bank.modeling.modeling_stage import StructuredModelingStage


VALID_MODEL_BODY = """SETS
  i in Animals = {cow, sheep}

PARAMETERS
  sell_price[i]
  capacity

VARIABLES
  x[i] >= 0, continuous

OBJECTIVE
  maximize sum_i sell_price[i] * x[i]

CONSTRAINTS
  C1: sum_i x[i] <= capacity"""

GOOD_SIGNATURE = {
    "objective": "linear",
    "decision": ["continuous_flow"],
    "constraint": ["capacity"],
    "interaction": "shared_resource_coupled",
    "features": {},
}


def _model_output(model_body: str) -> str:
    return "<think>analysis</think>\n<model>" + model_body + "</model>"


class StructuredModelingStageTest(unittest.TestCase):
    def test_no_llm_fails(self):
        result = asyncio.run(StructuredModelingStage(None).run("problem"))
        self.assertFalse(result.success)
        self.assertEqual(result.issues[0]["type"], "no_llm")

    def test_valid_model_passes(self):
        # text: 1 modeling output; object: 1 signature JSON.
        llm = FakeLLMClient(
            text_responses=[_model_output(VALID_MODEL_BODY)],
            object_responses=[GOOD_SIGNATURE],
        )
        result = asyncio.run(StructuredModelingStage(llm).run("farm problem"))
        self.assertTrue(result.success, result.issues)
        self.assertIn("VARIABLES", result.model)
        self.assertIsNotNone(result.signature)
        self.assertEqual(result.rounds_used, 1)

    def test_invalid_model_retries_then_passes(self):
        bad_model = VALID_MODEL_BODY.replace("OBJECTIVE", "OBJ")  # missing OBJECTIVE block
        llm = FakeLLMClient(
            text_responses=[_model_output(bad_model), _model_output(VALID_MODEL_BODY)],
            object_responses=[GOOD_SIGNATURE, GOOD_SIGNATURE],
        )
        result = asyncio.run(StructuredModelingStage(llm).run("farm problem"))
        self.assertTrue(result.success, result.issues)
        self.assertEqual(result.rounds_used, 2)

    def test_persistent_failure_returns_issues(self):
        bad_model = VALID_MODEL_BODY.replace("OBJECTIVE", "OBJ")
        llm = FakeLLMClient(
            text_responses=[_model_output(bad_model)] * 3,
            object_responses=[GOOD_SIGNATURE] * 3,
        )
        result = asyncio.run(StructuredModelingStage(llm, ).run("farm problem"))
        self.assertFalse(result.success)
        self.assertTrue(result.issues)
        self.assertEqual(result.rounds_used, 3)

    def test_prompt_contains_feedback_on_retry(self):
        bad_model = VALID_MODEL_BODY.replace("OBJECTIVE", "OBJ")
        llm = FakeLLMClient(
            text_responses=[_model_output(bad_model), _model_output(VALID_MODEL_BODY)],
            object_responses=[GOOD_SIGNATURE, GOOD_SIGNATURE],
        )
        asyncio.run(StructuredModelingStage(llm).run("farm problem"))
        # second modeling prompt should carry the issues feedback
        modeling_prompts = [p for p in llm.prompts if "PROBLEM" in p]
        self.assertTrue(any("FAILED verification" in p for p in modeling_prompts[1:]))


if __name__ == "__main__":
    unittest.main()
