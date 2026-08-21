"""Tests for induction/candidates.py (module 3.1: isomorphic cluster discovery)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.candidates import (
    CandidateCluster,
    ClusterMember,
    SignatureClusterer,
)


def make_realization(
    exp_id,
    title,
    objective="linear",
    decision=None,
    constraint=None,
    interaction="shared_resource_coupled",
    features=None,
    source_episodes=None,
    created_at="2026-08-19",
    retrieval_count=0,
    family=None,
):
    decision = decision if decision is not None else ["binary_assignment"]
    constraint = constraint if constraint is not None else ["capacity"]
    record = {
        "experience_id": exp_id,
        "title": title,
        "retrieval_text": title,
        "created_at": created_at,
        "retrieval_count": retrieval_count,
        "math_scope": {
            "structural_signature": {
                "objective": objective,
                "decision": decision,
                "constraint": constraint,
                "interaction": interaction,
                "features": features or {},
            }
        },
        "evidence": {"source_episodes": source_episodes or []},
    }
    if family is not None:
        record["_family"] = family
    return record


def resolver(record):
    return record.get("_family", "")


class SignatureClustererTest(unittest.TestCase):
    def test_discovers_heterogeneous_isomorphic_cluster(self):
        # Three different families, SAME core-4 signature -> one cluster.
        reals = [
            make_realization("e1", "warehouse capacity allocation", family="inventory"),
            make_realization("e2", "machine capacity scheduling", family="scheduling"),
            make_realization("e3", "workforce hour assignment", family="workforce"),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        self.assertEqual(len(clusters), 1)
        c = clusters[0]
        self.assertEqual(c.size, 3)
        self.assertEqual(sorted(c.problem_families), ["inventory", "scheduling", "workforce"])
        self.assertTrue(c.is_heterogeneous())

    def test_same_family_group_is_rejected(self):
        # Structurally isomorphic but all the SAME family -> redundancy, not ours.
        reals = [
            make_realization("e1", "inventory policy A", family="inventory"),
            make_realization("e2", "inventory policy B", family="inventory"),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        self.assertEqual(clusters, [])

    def test_different_core_signature_not_clustered(self):
        # Different core-4 -> not isomorphic -> no cluster even across families.
        reals = [
            make_realization("e1", "capacity alloc", family="inventory",
                             constraint=["capacity"]),
            make_realization("e2", "network flow", family="network",
                             constraint=["flow_conservation"]),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        self.assertEqual(clusters, [])

    def test_min_cluster_size_enforced(self):
        reals = [make_realization("e1", "solo", family="inventory")]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        self.assertEqual(clusters, [])

    def test_open_feature_slots_subsplit_core_bucket(self):
        # Same core-4 but disjoint feature VALUES -> separate sub-clusters.
        reals = [
            make_realization("e1", "inv multi-period", family="inventory",
                             features={"temporal": "multi_period_balance"}),
            make_realization("e2", "prod multi-period", family="production",
                             features={"temporal": "multi_period_balance"}),
            make_realization("e3", "net path", family="network",
                             features={"network": "path_on_graph"}),
            make_realization("e4", "routing path", family="routing",
                             features={"network": "path_on_graph"}),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        # Two feature-homogeneous sub-clusters, each cross-family.
        self.assertEqual(len(clusters), 2)
        feat_sets = sorted(tuple(c.shared_feature_keys) for c in clusters)
        self.assertEqual(feat_sets, [("network",), ("temporal",)])

    def test_missing_feature_key_not_penalized(self):
        # One member lacks the feature key entirely; core isomorphism still clusters them.
        reals = [
            make_realization("e1", "inv", family="inventory",
                             features={"resource": "shared_scarce"}),
            make_realization("e2", "alloc", family="allocation", features={}),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        self.assertEqual(len(clusters), 1)

    def test_provenance_handles_carried(self):
        reals = [
            make_realization("e1", "inv", family="inventory", source_episodes=["ep_a"]),
            make_realization("e2", "sched", family="scheduling", source_episodes=["ep_b"]),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        member_eps = {m.realization_id: m.source_episodes for m in clusters[0].members}
        self.assertEqual(member_eps["e1"], ["ep_a"])
        self.assertEqual(member_eps["e2"], ["ep_b"])

    def test_family_resolved_via_episode_index(self):
        ep_index = {"ep_a": "inventory", "ep_b": "scheduling"}
        reals = [
            make_realization("e1", "x", source_episodes=["ep_a"]),
            make_realization("e2", "y", source_episodes=["ep_b"]),
        ]
        clusters = SignatureClusterer(episode_family_index=ep_index).discover(reals)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0].problem_families), ["inventory", "scheduling"])

    def test_priority_score_prefers_larger_more_active(self):
        big = [
            make_realization("e1", "a", family="inventory", retrieval_count=5),
            make_realization("e2", "b", family="scheduling", retrieval_count=5),
            make_realization("e3", "c", family="workforce", retrieval_count=5),
        ]
        # A second, genuinely isomorphic but distinct cluster (different core-4) so the
        # two groups do NOT merge (both groups have empty features -> missing != penalty).
        small = [
            make_realization("e4", "d", family="network",
                             decision=["continuous_flow"], constraint=["flow_conservation"]),
            make_realization("e5", "e", family="routing",
                             decision=["continuous_flow"], constraint=["flow_conservation"]),
        ]
        clusters = SignatureClusterer(family_resolver=resolver).discover(big + small)
        self.assertEqual(len(clusters), 2)
        # Highest-priority cluster is the larger, more-active, more-heterogeneous one.
        self.assertEqual(clusters[0].size, 3)

    def test_record_without_signature_skipped(self):
        reals = [
            make_realization("e1", "inv", family="inventory"),
            {"experience_id": "bad", "math_scope": {}},  # default sig is valid; force bad below
        ]
        # default-constructed signature is valid (linear/independent) so it WILL cluster;
        # use an out-of-vocabulary value to force a parse failure.
        reals[1]["math_scope"] = {"structural_signature": {"objective": "not_a_vocab"}}
        reals[1]["_family"] = "scheduling"
        clusters = SignatureClusterer(family_resolver=resolver).discover(reals)
        # only e1 valid -> below min_cluster_size -> no cluster, no crash
        self.assertEqual(clusters, [])

    def test_constructor_guards(self):
        with self.assertRaises(ValueError):
            SignatureClusterer(min_cluster_size=1)
        with self.assertRaises(ValueError):
            SignatureClusterer(min_families=1)

    def test_to_dict_roundtrip_shape(self):
        reals = [
            make_realization("e1", "inv", family="inventory"),
            make_realization("e2", "sched", family="scheduling"),
        ]
        c = SignatureClusterer(family_resolver=resolver).discover(reals)[0]
        d = c.to_dict()
        for key in ("cluster_id", "core_key", "shared_feature_keys", "problem_families", "size", "score", "members"):
            self.assertIn(key, d)
        self.assertEqual(len(d["members"]), 2)


if __name__ == "__main__":
    unittest.main()
