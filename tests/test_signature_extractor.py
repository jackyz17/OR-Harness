"""Tests for harness-mode signature extraction (module 0.6, D6/D18)."""

import asyncio
import json
import unittest

from or_experience_bank.core.modeling_schemas import OBJECTIVE_STRUCTURES
from or_experience_bank.modeling.signature_extractor import (
    LLMBackedExtractor,
    SignatureExtractor,
)

GOOD_SIGNATURE_JSON = {
    "objective": "linear",
    "decision": ["continuous_flow", "multi_index_2d"],
    "constraint": ["flow_conservation", "capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"temporal": "multi_period_balance"},
}


class PromptBuildTest(unittest.TestCase):
    def test_prompt_contains_vocabularies(self):
        prompt = SignatureExtractor().build_extraction_prompt("<model>VARIABLES x[i,t]</model>")
        for value in OBJECTIVE_STRUCTURES:
            self.assertIn(value, prompt)
        self.assertIn("decision", prompt)
        self.assertIn("features", prompt)

    def test_retry_errors_appended(self):
        prompt = SignatureExtractor().build_extraction_prompt("model", retry_errors=["objective 'banana' invalid"])
        self.assertIn("out-of-vocabulary", prompt)
        self.assertIn("banana", prompt)


class ParseValidateTest(unittest.TestCase):
    def test_valid_dict(self):
        result = SignatureExtractor().parse_and_validate(GOOD_SIGNATURE_JSON)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.signature.objective, "linear")
        self.assertEqual(result.signature.features["temporal"], "multi_period_balance")

    def test_valid_json_string(self):
        result = SignatureExtractor().parse_and_validate(json.dumps(GOOD_SIGNATURE_JSON))
        self.assertTrue(result.valid, result.errors)

    def test_json_embedded_in_prose(self):
        raw = "Here is: " + json.dumps(GOOD_SIGNATURE_JSON) + " done"
        result = SignatureExtractor().parse_and_validate(raw)
        self.assertTrue(result.valid, result.errors)

    def test_invalid_core_value_rejected(self):
        bad = dict(GOOD_SIGNATURE_JSON, objective="banana")
        result = SignatureExtractor().parse_and_validate(bad)
        self.assertFalse(result.valid)
        self.assertTrue(any("banana" in e for e in result.errors))
        self.assertTrue(result.retry_hint)

    def test_unparseable_rejected(self):
        self.assertFalse(SignatureExtractor().parse_and_validate("not json at all").valid)

    def test_open_features_not_validated(self):
        raw = dict(GOOD_SIGNATURE_JSON, features={"brand_new_dim": "whatever"})
        result = SignatureExtractor().parse_and_validate(raw)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.signature.features["brand_new_dim"], "whatever")


class LLMBackedExtractorTest(unittest.TestCase):
    def test_no_llm_returns_harness_hint(self):
        result = asyncio.run(LLMBackedExtractor(llm_client=None).extract("model"))
        self.assertFalse(result.valid)
        self.assertTrue(any("harness" in e for e in result.errors))

    def test_retry_loop_succeeds_after_fix(self):
        class _FlakyLLM:
            def __init__(self):
                self.calls = 0

            async def generate_object(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return dict(GOOD_SIGNATURE_JSON, objective="banana")
                return GOOD_SIGNATURE_JSON

        llm = _FlakyLLM()
        result = asyncio.run(LLMBackedExtractor(llm_client=llm).extract("model"))
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(llm.calls, 2)

    def test_retry_exhaustion(self):
        class _BadLLM:
            async def generate_object(self, prompt):
                return dict(GOOD_SIGNATURE_JSON, objective="banana")

        result = asyncio.run(LLMBackedExtractor(llm_client=_BadLLM()).extract("model"))
        self.assertFalse(result.valid)
        self.assertIn("exhausted", result.retry_hint)


if __name__ == "__main__":
    unittest.main()
