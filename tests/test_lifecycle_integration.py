"""Integration tests: lifecycle/utility wired into retrieval + store (Phase 2.3)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import experience

from or_experience_bank.core.lifecycle import LifecycleStore
from or_experience_bank.core.store import AppendOnlyExperienceStore
from or_experience_bank.core.utility_tracker import UtilityTracker
from or_experience_bank.retrieval.index import EmbeddingIndex, LocalHashEmbeddingBackend
from or_experience_bank.retrieval.retrieval import ExperienceRetriever


def build(tmp, with_utility=False, with_lifecycle=False, alpha=2, penalty=0.3):
    home = Path(tmp)
    backend = LocalHashEmbeddingBackend(128)
    utility = UtilityTracker(home, alpha=alpha, penalty=penalty) if with_utility else None
    lifecycle = LifecycleStore(home) if with_lifecycle else None
    store = AppendOnlyExperienceStore(home, lifecycle=lifecycle, embed=backend.embed_query)
    index = EmbeddingIndex(home / "index", backend)
    retriever = ExperienceRetriever(store, index, utility_tracker=utility, lifecycle=lifecycle)
    return store, retriever, utility, lifecycle, backend


class RetrievalUtilityPenaltyTest(unittest.TestCase):
    def test_low_utility_record_sinks_in_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever, utility, _, _ = build(tmp, with_utility=True, alpha=2, penalty=0.1)
            # two DISTINCT records (different content -> different hash) sharing the query terms
            stale = experience(title="capacity constraint handling rule for stale record")
            stale.experience_id = "exp_stale"
            stale.retrieval_text = "capacity constraint handling rule stale variant"
            fresh = experience(title="capacity constraint handling rule for fresh record")
            fresh.experience_id = "exp_fresh"
            fresh.retrieval_text = "capacity constraint handling rule fresh variant"
            store.append(stale)
            store.append(fresh)
            # make the stale one low-utility: retrieved many times, never useful
            utility.record_retrievals(["exp_stale"] * 10)
            self.assertTrue(utility.is_low_utility("exp_stale"))

            retriever.rebuild("modeling")
            hits = retriever.retrieve("modeling", "capacity constraint handling rule", top_k=5)
            self.assertTrue(hits)
            # the fresh (unpenalized) record outranks the penalized stale one
            self.assertEqual(hits[0].experience_id, "exp_fresh")

    def test_retrieval_counts_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever, utility, _, _ = build(tmp, with_utility=True)
            rec = experience(title="a modeling rule")
            store.append(rec)
            retriever.rebuild("modeling")
            hits = retriever.retrieve("modeling", "modeling rule", top_k=5)
            self.assertTrue(hits)
            for h in hits:
                self.assertGreaterEqual(utility.retrieval_count(h.experience_id), 1)

    def test_no_tracker_means_no_penalty_and_no_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever, _, _, _ = build(tmp, with_utility=False)
            store.append(experience(title="a modeling rule"))
            retriever.rebuild("modeling")
            hits = retriever.retrieve("modeling", "modeling rule", top_k=5)
            self.assertTrue(hits)  # works fine without the tracker


class RetrievalDeprecatedExclusionTest(unittest.TestCase):
    def test_deprecated_excluded_from_rebuilt_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, retriever, _, lifecycle, _ = build(tmp, with_lifecycle=True)
            doomed = experience(title="harmful modeling rule")
            doomed.experience_id = "exp_doomed"
            store.append(doomed)
            lifecycle.mark_deprecated(doomed.to_dict(), reason="made things worse")
            retriever.rebuild("modeling")
            hits = retriever.retrieve("modeling", "harmful modeling rule", top_k=5)
            self.assertNotIn("exp_doomed", [h.experience_id for h in hits])


class StoreAntiResurrectionTest(unittest.TestCase):
    def test_reworded_resurrection_rejected_by_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _, _, lifecycle, backend = build(tmp, with_lifecycle=True)
            retrieval_text = "assign jobs to machine capacity minimize makespan"
            original = experience(title="capacity rule")
            original.retrieval_text = retrieval_text
            store.append(original)
            lifecycle.mark_deprecated(original.to_dict(), reason="harmful", embed=backend.embed_query)
            # reworded but semantically identical (same local-hash vector)
            twin = experience(title="different title")
            twin.retrieval_text = retrieval_text
            result = store.append(twin)
            self.assertEqual(result.status, "rejected_deprecated")

    def test_genuinely_new_experience_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _, _, lifecycle, backend = build(tmp, with_lifecycle=True)
            bad = experience(title="bad rule about capacity")
            store.append(bad)
            lifecycle.mark_deprecated(bad.to_dict(), reason="harmful", embed=backend.embed_query)
            good = experience(title="a completely different network flow conservation rule")
            good.retrieval_text = "network flow conservation with balance constraints on graph nodes"
            result = store.append(good)
            self.assertEqual(result.status, "appended")


if __name__ == "__main__":
    unittest.main()
