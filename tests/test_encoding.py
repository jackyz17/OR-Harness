"""Tests for induction/encoding.py (module 3.2: batch structural encoding)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.encoding import (
    LLMBackedEncoder,
    StructuralEncoder,
)
from or_experience_bank.llm_client import FakeLLMClient


def record_with_signature(exp_id="e1"):
    return {
        "experience_id": exp_id,
        "title": "capacity allocation",
        "math_scope": {
            "structural_signature": {
                "objective": "linear",
                "decision": ["binary_assignment"],
                "constraint": ["capacity"],
                "interaction": "shared_resource_coupled",
                "features": {"resource": "shared_scarce"},
            }
        },
    }


def record_without_signature(exp_id="e2"):
    return {
        "experience_id": exp_id,
        "title": "minimize routing cost",
        "retrieval_text": "assign vehicles to routes minimizing total travel cost",
        "method": {"action_template": "x[i,j] binary assignment with capacity"},
    }


class StructuralEncoderTest(unittest.TestCase):
    def test_reuses_valid_existing_signature(self):
        enc = StructuralEncoder()
        result = enc.encode(record_with_signature())
        self.assertEqual(result.status, "reused")
        self.assertTrue(result.ok)
        self.assertEqual(result.signature.objective, "linear")
        self.assertEqual(result.signature.features["resource"], "shared_scarce")

    def test_needs_llm_when_signature_missing(self):
        enc = StructuralEncoder()
        result = enc.encode(record_without_signature())
        self.assertEqual(result.status, "needs_llm")
        self.assertIsNotNone(result.prompt)
        self.assertIn("ALLOWED VOCABULARIES", result.prompt)

    def test_needs_llm_when_signature_invalid_vocab(self):
        rec = record_with_signature("e3")
        rec["math_scope"]["structural_signature"]["objective"] = "bogus"
        result = StructuralEncoder().encode(rec)
        self.assertEqual(result.status, "needs_llm")

    def test_submit_validates_agent_output(self):
        enc = StructuralEncoder()
        rec = record_without_signature()
        good = {
            "objective": "linear",
            "decision": ["binary_assignment"],
            "constraint": ["capacity", "assignment_exactly_once"],
            "interaction": "shared_resource_coupled",
            "features": {},
        }
        result = enc.submit(rec, good)
        self.assertEqual(result.status, "encoded")
        self.assertTrue(result.ok)

    def test_submit_rejects_out_of_vocab(self):
        enc = StructuralEncoder()
        rec = record_without_signature()
        bad = {"objective": "nope", "decision": [], "constraint": [], "interaction": "independent"}
        result = enc.submit(rec, bad)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(result.errors)

    def test_signatures_ready_maps_valid_only(self):
        enc = StructuralEncoder()
        ready = enc.signatures_ready([record_with_signature("a"), record_without_signature("b")])
        self.assertIn("a", ready)
        self.assertNotIn("b", ready)

    def test_encode_batch_mixed(self):
        enc = StructuralEncoder()
        results = enc.encode_batch([record_with_signature("a"), record_without_signature("b")])
        self.assertEqual([r.status for r in results], ["reused", "needs_llm"])


class LLMBackedEncoderTest(unittest.TestCase):
    def test_llm_loop_encodes_missing_signature(self):
        sig = {
            "objective": "linear",
            "decision": ["binary_assignment"],
            "constraint": ["capacity"],
            "interaction": "shared_resource_coupled",
            "features": {},
        }
        llm = FakeLLMClient(object_responses=[sig])
        enc = LLMBackedEncoder(llm_client=llm)
        result = asyncio.run(enc.encode(record_without_signature()))
        self.assertEqual(result.status, "encoded")
        self.assertTrue(result.ok)

    def test_llm_loop_skips_call_when_signature_present(self):
        llm = FakeLLMClient(object_responses=[])
        enc = LLMBackedEncoder(llm_client=llm)
        result = asyncio.run(enc.encode(record_with_signature()))
        self.assertEqual(result.status, "reused")
        self.assertEqual(llm.prompts, [])  # no LLM call made

    def test_llm_loop_requires_client(self):
        enc = LLMBackedEncoder(llm_client=None)
        result = asyncio.run(enc.encode(record_without_signature()))
        self.assertEqual(result.status, "invalid")
        self.assertTrue(any("no llm_client" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
