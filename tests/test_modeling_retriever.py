"""Tests for retrieval/modeling_retriever.py (Phase 4.1: pattern reflow recall)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.lifecycle import LifecycleStore
from or_experience_bank.core.modeling_schemas import (
    ModelingExperience,
)
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.retrieval.modeling_retriever import ModelingRetriever, PlanningPriors


def make_realization(rid, title):
    rec = ModelingExperience(title=title, retrieval_text=title)
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict({
        "objective": "linear", "decision": ["binary_assignment"],
        "constraint": ["capacity"], "interaction": "shared_resource_coupled", "features": {},
    })
    rec.experience_id = rid
    rec.compute_content_hash()
    return rec


def make_record_for_deprecate(rid):
    rec = ModelingExperience(title=rid, retrieval_text=rid)
    rec.experience_id = rid
    rec.compute_content_hash()
    return rec.to_dict()


def make_pattern(pid, principle, status="validated"):
    rec = ModelingExperience(
        title="pattern " + pid,
        retrieval_text=principle,
        status=status,
        modeling_aspect="constraint",
    )
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict({
        "objective": "linear", "decision": ["binary_assignment"],
        "constraint": ["capacity"], "interaction": "shared_resource_coupled", "features": {},
    })
    rec.experience_id = pid
    rec.compute_content_hash()
    return rec


class ModelingRetrieverTest(unittest.TestCase):
    def _build(self, tmp, lifecycle=None):
        store = ModelingStore(Path(tmp))
        index = EmbeddingIndex(Path(tmp) / "index" / "modeling_bank", LocalHashEmbeddingBackend(128))
        return store, ModelingRetriever(store, index, lifecycle)

    def test_joint_recall_returns_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever = self._build(tmp)
            store.append(make_realization("e1", "allocate stock to warehouse capacity"))
            store.append(make_pattern("p1", "prioritize higher marginal contribution on shared scarce resource"))
            retriever.rebuild()
            priors = retriever.retrieve_priors("allocate scarce resource capacity")
            self.assertGreaterEqual(len(priors.records), 2)
            # backward-compat: patterns = validated, realizations = non-validated
            self.assertGreaterEqual(len(priors.patterns), 1)
            self.assertGreaterEqual(len(priors.realizations), 1)
            all_ids = set(priors.labels.values())
            self.assertIn("e1", all_ids)
            self.assertIn("p1", all_ids)

    def test_hypothesis_and_refuted_never_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever = self._build(tmp)
            store.append(make_pattern("p_hyp", "hypothesis principle", status=None))
            store.append(make_pattern("p_ref", "refuted principle", status="refuted"))
            store.append(make_pattern("p_val", "validated principle", status="validated"))
            retriever.rebuild()
            priors = retriever.retrieve_priors("principle", top_k=10)
            surfaced = {p["experience_id"] for p in priors.patterns}
            self.assertEqual(surfaced, {"p_val"})  # only validated

    def test_deprecated_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = LifecycleStore(Path(tmp))
            store, retriever = self._build(tmp, lifecycle)
            store.append(make_realization("e_good", "a good capacity rule"))
            store.append(make_realization("e_bad", "a harmful capacity rule"))
            lifecycle.mark_deprecated({"experience_id": "e_bad"}, reason="harmful")
            retriever.rebuild()
            priors = retriever.retrieve_priors("capacity rule", top_k=10)
            surfaced = {r["experience_id"] for r in priors.realizations}
            self.assertNotIn("e_bad", surfaced)
            self.assertIn("e_good", surfaced)

    def test_deprecated_excluded_from_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = LifecycleStore(Path(tmp))
            store, retriever = self._build(tmp, lifecycle)
            store.append(make_realization("e_old", "capacity allocation old method"))
            store.append(make_realization("e_new", "capacity allocation new method"))
            lifecycle.mark_deprecated(make_record_for_deprecate("e_old"), reason="low utility")
            retriever.rebuild()
            priors = retriever.retrieve_priors("capacity allocation", top_k=10)
            surfaced = [r["experience_id"] for r in priors.records]
            # deprecated is excluded
            self.assertNotIn("e_old", surfaced)
            self.assertIn("e_new", surfaced)

    def test_empty_bank_gives_empty_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever = self._build(tmp)
            retriever.rebuild()
            priors = retriever.retrieve_priors("anything")
            self.assertTrue(priors.is_empty())

    def test_index_lives_in_dedicated_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever = self._build(tmp)
            retriever.rebuild()
            # dedicated sub-directory: no collision with the flat layers' index files
            self.assertTrue((Path(tmp) / "index" / "modeling_bank" / "modeling.embedding.json").exists())
            self.assertFalse((Path(tmp) / "index" / "modeling.embedding.json").exists())


if __name__ == "__main__":
    unittest.main()
