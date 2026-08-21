"""Tests: candidates wired to lifecycle (exclude deprecated) + utility (live priority)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.lifecycle import LifecycleStore
from or_experience_bank.core.utility_tracker import UtilityTracker
from or_experience_bank.induction.candidates import SignatureClusterer


def make_realization(exp_id, family):
    return {
        "experience_id": exp_id,
        "title": exp_id + " method",
        "retrieval_text": exp_id,
        "math_scope": {"structural_signature": {
            "objective": "linear", "decision": ["binary_assignment"],
            "constraint": ["capacity"], "interaction": "shared_resource_coupled",
            "features": {"resource": "shared_scarce"},
        }},
        "evidence": {"source_episodes": []},
        "_family": family,
    }


def resolver(record):
    return record.get("_family", "")


class CandidatesLifecycleTest(unittest.TestCase):
    def test_deprecated_excluded_from_clustering(self):
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = LifecycleStore(Path(tmp))
            reals = [
                make_realization("e1", "inventory"),
                make_realization("e2", "scheduling"),
                make_realization("e3", "workforce"),
            ]
            # e3 deprecated -> cluster drops to 2 members but still cross-family
            lifecycle.mark_deprecated(reals[2], reason="harmful")
            clusters = SignatureClusterer(
                family_resolver=resolver, lifecycle=lifecycle
            ).discover(reals)
            self.assertEqual(len(clusters), 1)
            member_ids = {m.realization_id for m in clusters[0].members}
            self.assertEqual(member_ids, {"e1", "e2"})
            self.assertNotIn("e3", member_ids)

    def test_cluster_dissolves_if_deprecated_drops_below_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = LifecycleStore(Path(tmp))
            reals = [
                make_realization("e1", "inventory"),
                make_realization("e2", "scheduling"),
            ]
            lifecycle.mark_deprecated(reals[1], reason="harmful")
            clusters = SignatureClusterer(
                family_resolver=resolver, lifecycle=lifecycle
            ).discover(reals)
            self.assertEqual(clusters, [])  # only e1 left -> below min_cluster_size

    def test_live_utility_drives_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            utility = UtilityTracker(Path(tmp))
            utility.record_retrievals(["e1"] * 7)
            utility.record_retrievals(["e2"] * 3)
            reals = [make_realization("e1", "inventory"), make_realization("e2", "scheduling")]
            clusters = SignatureClusterer(
                family_resolver=resolver, utility_tracker=utility
            ).discover(reals)
            member_hits = {m.realization_id: m.retrieval_hits for m in clusters[0].members}
            # live tracker values, not the (absent) static field
            self.assertEqual(member_hits["e1"], 7)
            self.assertEqual(member_hits["e2"], 3)

    def test_without_tracker_falls_back_to_static_field(self):
        reals = [make_realization("e1", "inventory"), make_realization("e2", "scheduling")]
        reals[0]["retrieval_count"] = 4
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        member_hits = {m.realization_id: m.retrieval_hits for m in clusters[0].members}
        self.assertEqual(member_hits["e1"], 4)
        self.assertEqual(member_hits["e2"], 0)


if __name__ == "__main__":
    unittest.main()
