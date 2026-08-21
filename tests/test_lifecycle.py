"""Tests for core/lifecycle.py (lifecycle states + compressed cold archive)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.lifecycle import (
    ACTIVE,
    DEPRECATED,
    LifecycleStore,
)
from or_experience_bank.retrieval.index import LocalHashEmbeddingBackend


def make_record(exp_id, title, retrieval_text=None, signature=None):
    return {
        "experience_id": exp_id,
        "content_hash": "hash_" + exp_id,
        "layer": "modeling",
        "polarity": "positive",
        "title": title,
        "retrieval_text": retrieval_text or title,
        "created_at": "2026-08-01",
        "math_scope": {"structural_signature": signature or {
            "objective": "linear", "decision": ["binary_assignment"],
            "constraint": ["capacity"], "interaction": "shared_resource_coupled", "features": {},
        }},
        "method": {"action_template": "very long method body " * 50, "rationale": "rationale"},
        "evidence": {"source_episodes": ["ep_" + exp_id]},
        "validation": {"source_consistency": "3/3", "transfer_tests": [{"task": "t"}]},
        "scoring": {"coverage": 1.0},
    }


class LifecycleStateTest(unittest.TestCase):
    def test_default_state_is_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            self.assertEqual(store.state_of("exp_x"), ACTIVE)
            self.assertTrue(store.is_active("exp_x"))

    def test_mark_deprecated_removes_from_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            store.mark_deprecated(make_record("exp_bad", "harmful method"), reason="utility 0.02")
            self.assertEqual(store.state_of("exp_bad"), DEPRECATED)
            self.assertNotIn("exp_bad", store.active_ids(["exp_bad"]))

    def test_state_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            LifecycleStore(Path(tmp)).mark_deprecated(make_record("exp_a", "old"), reason="low utility")
            store2 = LifecycleStore(Path(tmp))
            self.assertEqual(store2.state_of("exp_a"), DEPRECATED)


class ArchiveCompressionTest(unittest.TestCase):
    def test_archive_card_is_compressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            record = make_record("exp_big", "big record")
            card = store.mark_deprecated(record, reason="low utility")
            # bulky fields dropped
            self.assertNotIn("method", card)
            self.assertNotIn("retrieval_text", card)
            self.assertNotIn("validation", card)
            self.assertNotIn("scoring", card)
            # provenance fields kept
            self.assertEqual(card["experience_id"], "exp_big")
            self.assertEqual(card["content_hash"], "hash_exp_big")  # ORIGINAL hash
            self.assertEqual(card["source_episodes"], ["ep_exp_big"])
            self.assertEqual(card["deprecate_reason"], "low utility")
            self.assertTrue(card["summary"])

    def test_archive_card_keeps_vector_not_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalHashEmbeddingBackend(128)
            store = LifecycleStore(Path(tmp))
            card = store.mark_deprecated(
                make_record("exp_v", "a title", retrieval_text="assign jobs to machine capacity"),
                reason="harmful",
                embed=backend.embed_query,
            )
            self.assertIsNotNone(card["retrieval_vector"])
            self.assertEqual(len(card["retrieval_vector"]), 128)
            self.assertNotIn("retrieval_text", card)  # 方案甲: vector kept, text dropped

    def test_archive_appends_and_is_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            store.mark_deprecated(make_record("e1", "m1"), reason="r1")
            store.mark_deprecated(make_record("e2", "m2"), reason="r2")
            cards = store.iter_archive()
            self.assertEqual(len(cards), 2)
            self.assertEqual({c["experience_id"] for c in cards}, {"e1", "e2"})


class ArchiveDedupTest(unittest.TestCase):
    def test_exact_hash_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            store.mark_deprecated(make_record("exp_x", "m"), reason="r")
            hit = store.archive_match(content_hash="hash_exp_x")
            self.assertIsNotNone(hit)
            self.assertEqual(hit["match"], "exact")

    def test_approximate_match_blocks_reworded_resurrection(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalHashEmbeddingBackend(256)
            store = LifecycleStore(Path(tmp), similarity_threshold=0.8)
            original_text = "assign jobs to machine capacity to minimize makespan"
            store.mark_deprecated(
                make_record("exp_harm", "m", retrieval_text=original_text),
                reason="harmful",
                embed=backend.embed_query,
            )
            # a reworded but semantically near-identical candidate (same backend -> identical vector)
            hit = store.archive_match(
                content_hash="different_hash",
                retrieval_text=original_text,
                embed=backend.embed_query,
            )
            self.assertIsNotNone(hit)
            self.assertEqual(hit["match"], "approximate")
            self.assertGreaterEqual(hit["similarity"], 0.8)

    def test_no_match_for_genuinely_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalHashEmbeddingBackend(256)
            store = LifecycleStore(Path(tmp), similarity_threshold=0.8)
            store.mark_deprecated(
                make_record("exp_a", "m", retrieval_text="assign jobs to machine capacity"),
                reason="r",
                embed=backend.embed_query,
            )
            hit = store.archive_match(
                content_hash="new_hash",
                retrieval_text="completely unrelated topic about network flow conservation",
                embed=backend.embed_query,
            )
            self.assertIsNone(hit)

    def test_approximate_skipped_without_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LifecycleStore(Path(tmp))
            # deprecated WITHOUT embed -> no vector stored -> approximate layer unavailable
            store.mark_deprecated(make_record("exp_nv", "m", retrieval_text="some text"), reason="r")
            hit = store.archive_match(content_hash="other", retrieval_text="some text")
            self.assertIsNone(hit)


if __name__ == "__main__":
    unittest.main()
