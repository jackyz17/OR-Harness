"""Tests for induction/alignment.py (module 3.3: cross-memory role alignment)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.alignment import (
    CANONICAL_ROLES,
    LLMBackedAligner,
    StructuralAligner,
)
from or_experience_bank.induction.candidates import SignatureClusterer
from or_experience_bank.llm_client import FakeLLMClient


def make_realization(exp_id, family, method_text):
    return {
        "experience_id": exp_id,
        "title": method_text,
        "retrieval_text": method_text,
        "math_scope": {
            "structural_signature": {
                "objective": "linear",
                "decision": ["binary_assignment"],
                "constraint": ["capacity"],
                "interaction": "shared_resource_coupled",
                "features": {"resource": "shared_scarce"},
            }
        },
        "method": {"action_template": method_text},
        "evidence": {"source_episodes": []},
        "_family": family,
    }


def resolver(record):
    return record.get("_family", "")


def make_cluster():
    reals = [
        make_realization("e1", "inventory", "allocate stock x[i] to warehouse capacity"),
        make_realization("e2", "scheduling", "assign jobs x[j] to machine capacity"),
        make_realization("e3", "workforce", "assign workers x[k] to labor-hour capacity"),
    ]
    return SignatureClusterer(family_resolver=resolver).discover(reals)[0]


def valid_alignment_json():
    return {
        "roles": ["resource_pool", "capacity_limit", "competing_decisions"],
        "bindings": [
            {"realization_id": "e1", "problem_family": "inventory",
             "mapping": {"resource_pool": "warehouse", "capacity_limit": "5000m3",
                         "competing_decisions": "stock x[i]"}},
            {"realization_id": "e2", "problem_family": "scheduling",
             "mapping": {"resource_pool": "machine", "capacity_limit": "8h/day",
                         "competing_decisions": "jobs x[j]"}},
            {"realization_id": "e3", "problem_family": "workforce",
             "mapping": {"resource_pool": "labor pool", "capacity_limit": "40h/week",
                         "competing_decisions": "workers x[k]"}},
        ],
        "confidence": 0.9,
        "notes": "all three share a shared-scarce-resource allocation structure",
    }


class StructuralAlignerTest(unittest.TestCase):
    def test_prompt_includes_roles_and_members(self):
        cluster = make_cluster()
        prompt = StructuralAligner().build_alignment_prompt(cluster)
        self.assertIn("CANONICAL ROLES", prompt)
        for rid in ("e1", "e2", "e3"):
            self.assertIn(rid, prompt)
        self.assertIn("resource_pool", prompt)

    def test_parse_valid_alignment(self):
        cluster = make_cluster()
        amap = StructuralAligner().parse_and_validate(valid_alignment_json(), cluster)
        self.assertEqual(amap.roles, ["resource_pool", "capacity_limit", "competing_decisions"])
        self.assertEqual(len(amap.bindings), 3)
        self.assertAlmostEqual(amap.confidence, 0.9)
        by_id = {b.realization_id: b.mapping for b in amap.bindings}
        self.assertEqual(by_id["e2"]["resource_pool"], "machine")

    def test_ungrounded_binding_dropped(self):
        cluster = make_cluster()
        payload = valid_alignment_json()
        payload["bindings"].append(
            {"realization_id": "ghost", "mapping": {"resource_pool": "nowhere"}}
        )
        amap = StructuralAligner().parse_and_validate(payload, cluster)
        self.assertEqual(len(amap.bindings), 3)  # ghost dropped (grounding red line)

    def test_unparseable_returns_empty_map(self):
        cluster = make_cluster()
        amap = StructuralAligner().parse_and_validate("not json at all", cluster)
        self.assertEqual(amap.bindings, [])
        self.assertIn("could not parse", amap.notes)

    def test_is_complete_requires_full_coverage(self):
        cluster = make_cluster()
        aligner = StructuralAligner()
        full = aligner.parse_and_validate(valid_alignment_json(), cluster)
        self.assertTrue(aligner.is_complete(full, cluster))
        partial = valid_alignment_json()
        partial["bindings"] = partial["bindings"][:2]  # drop a member
        self.assertFalse(aligner.is_complete(aligner.parse_and_validate(partial, cluster), cluster))

    def test_role_schema_and_mappings_pattern_ready(self):
        cluster = make_cluster()
        amap = StructuralAligner().parse_and_validate(valid_alignment_json(), cluster)
        schema = amap.role_schema()
        self.assertIn("resource_pool", schema)
        self.assertEqual(len(amap.role_mappings()), 3)

    def test_confidence_clamped(self):
        cluster = make_cluster()
        payload = valid_alignment_json()
        payload["confidence"] = 7.5
        amap = StructuralAligner().parse_and_validate(payload, cluster)
        self.assertEqual(amap.confidence, 1.0)


class LLMBackedAlignerTest(unittest.TestCase):
    def test_llm_loop_produces_alignment(self):
        cluster = make_cluster()
        llm = FakeLLMClient(object_responses=[valid_alignment_json()])
        amap = asyncio.run(LLMBackedAligner(llm_client=llm).align(cluster))
        self.assertEqual(len(amap.bindings), 3)
        self.assertEqual(len(llm.prompts), 1)

    def test_llm_loop_requires_client(self):
        cluster = make_cluster()
        amap = asyncio.run(LLMBackedAligner(llm_client=None).align(cluster))
        self.assertIn("no llm_client", amap.notes)


if __name__ == "__main__":
    unittest.main()
