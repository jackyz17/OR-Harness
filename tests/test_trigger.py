"""Tests for induction/trigger.py (v1 trigger policy, D4)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.modeling_schemas import ModelingExperience
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.induction.candidates import SignatureClusterer
from or_experience_bank.induction.trigger import InductionTrigger


_FAMILY = {"e_inv": "inventory", "e_prod": "production", "e_work": "workforce"}


def make_realization(rid, constraint=None):
    rec = ModelingExperience(title=rid + " method", retrieval_text=rid)
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict({
        "objective": "linear",
        "decision": ["binary_assignment"],
        "constraint": constraint or ["capacity"],
        "interaction": "shared_resource_coupled",
        "features": {"resource": "shared_scarce"},
    })
    rec.evidence.source_episodes = ["ep_" + rid]
    rec.experience_id = rid
    rec.compute_content_hash()
    return rec


def resolver(record):
    for ep in (record.get("evidence") or {}).get("source_episodes", []):
        rid = ep.replace("ep_", "")
        if rid in _FAMILY:
            return _FAMILY[rid]
    return record.get("experience_id", "")


def seed(store, rids):
    for rid in rids:
        store.append(make_realization(rid))


class InductionTriggerTest(unittest.TestCase):
    def test_gate0_blocks_when_no_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv"])  # only one family -> no heterogeneous cluster
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver))
            d = trig.decide()
            self.assertFalse(d.should_induce)
            self.assertIn("candidate gate", d.reason)

    def test_first_run_induces_when_cluster_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv", "e_prod"])
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver),
                                    min_new_realizations=3)
            d = trig.decide()
            # first run (last==0) bypasses the watermark and induces fresh clusters
            self.assertTrue(d.should_induce)
            self.assertEqual(len(d.clusters_to_induce), 1)

    def test_watermark_blocks_until_enough_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv", "e_prod"])
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver),
                                    min_new_realizations=3)
            first = trig.decide()
            trig.record_run(first)  # watermark = 2

            # add only ONE new realization -> below watermark of 3
            seed(store, ["e_work"])
            d = trig.decide()
            self.assertFalse(d.should_induce)
            self.assertIn("watermark", d.reason)

    def test_cooldown_skips_unchanged_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv", "e_prod"])
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver),
                                    min_new_realizations=1)
            first = trig.decide()
            trig.record_run(first)

            # no change at all -> same cluster signature -> cooldown blocks
            d = trig.decide()
            self.assertFalse(d.should_induce)
            self.assertIn("cooldown", d.reason)

    def test_new_cluster_member_retriggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv", "e_prod"])
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver),
                                    min_new_realizations=1)
            trig.record_run(trig.decide())

            # a NEW member joins the same core_key cluster -> membership changed -> re-induce
            seed(store, ["e_work"])
            d = trig.decide()
            self.assertTrue(d.should_induce)
            self.assertEqual(len(d.clusters_to_induce), 1)
            members = {m.realization_id for m in d.clusters_to_induce[0].members}
            self.assertEqual(members, {"e_inv", "e_prod", "e_work"})

    def test_log_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            seed(store, ["e_inv", "e_prod"])
            log = Path(tmp) / "bank" / "induction_trigger_log.jsonl"
            trig = InductionTrigger(store, SignatureClusterer(family_resolver=resolver),
                                    min_new_realizations=1, log_path=log)
            trig.record_run(trig.decide())
            seed(store, ["e_work"])
            trig.record_run(trig.decide())
            lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)  # appended, never rewritten

    def test_min_new_realizations_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))
            with self.assertRaises(ValueError):
                InductionTrigger(store, SignatureClusterer(), min_new_realizations=0)


if __name__ == "__main__":
    unittest.main()
