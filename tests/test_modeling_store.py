"""Tests for the independent Modeling Bank store (Option 3, D10)."""

import tempfile
import unittest
from pathlib import Path

from or_experience_bank.core.modeling_schemas import (
    MathScope,
    MethodBody,
    ModelingEvidence,
    ModelingExperience,
    StructuralSignature,
)
from or_experience_bank.core.modeling_store import ModelingStore


def _realization(title="balance eq") -> ModelingExperience:
    record = ModelingExperience(title=title, retrieval_text=title, polarity="positive")
    record.math_scope = MathScope(structural_signature=StructuralSignature(
        objective="linear", decision=["continuous_flow"], constraint=["flow_conservation"],
        interaction="shared_resource_coupled", features={},
    ))
    record.method = MethodBody(action_template="impose balance equality", rationale="couple periods")
    record.evidence = ModelingEvidence(source_episodes=["prob_1"])
    record.compute_content_hash()
    return record


class ModelingStoreTest(unittest.TestCase):
    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ModelingStore(Path(temp))
            result = store.append(_realization())
            self.assertEqual(result["status"], "appended")
            self.assertEqual(len(store.all_records()), 1)

    def test_exact_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ModelingStore(Path(temp))
            first = _realization()
            store.append(first)
            dup = _realization()  # same semantic content -> same hash
            dup.compute_content_hash()
            self.assertEqual(dup.content_hash, first.content_hash)
            result = store.append(dup)
            self.assertEqual(result["status"], "duplicate")

    def test_all_records_and_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ModelingStore(Path(temp))
            store.append(_realization("method one"))
            pattern = _realization("method two")
            pattern.status = "validated"
            pattern.compute_content_hash()
            store.append(pattern)
            self.assertEqual(len(store.all_records()), 2)
            self.assertEqual(len(store.validated_records()), 1)
            stats = store.stats()
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["validated"], 1)

    def test_invalid_pattern_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ModelingStore(Path(temp))
            bad = _realization("bad pattern")
            
            bad.status = "maybe"
            with self.assertRaises(ValueError):
                store.append(bad)


if __name__ == "__main__":
    unittest.main()
