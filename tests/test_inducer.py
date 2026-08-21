"""Tests for induction/inducer.py (module 3.4: hypothesis generation)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.alignment import StructuralAligner
from or_experience_bank.induction.candidates import SignatureClusterer
from or_experience_bank.induction.inducer import (
    LLMBackedInducer,
    PatternInducer,
)
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


def make_cluster_and_alignment():
    reals = [
        make_realization("e1", "inventory", "allocate stock x[i] to warehouse capacity"),
        make_realization("e2", "scheduling", "assign jobs x[j] to machine capacity"),
    ]
    cluster = SignatureClusterer(family_resolver=resolver).discover(reals)[0]
    alignment = StructuralAligner().parse_and_validate(
        {
            "roles": ["resource_pool", "competing_decisions", "objective_contribution"],
            "bindings": [
                {"realization_id": "e1", "problem_family": "inventory",
                 "mapping": {"resource_pool": "warehouse", "competing_decisions": "stock x[i]",
                             "objective_contribution": "holding cost"}},
                {"realization_id": "e2", "problem_family": "scheduling",
                 "mapping": {"resource_pool": "machine", "competing_decisions": "jobs x[j]",
                             "objective_contribution": "marginal profit"}},
            ],
            "confidence": 0.85,
            "notes": "shared scarce resource allocation",
        },
        cluster,
    )
    return cluster, alignment


def hypotheses_json():
    return [
        {
            "statement": "When multiple decisions compete for a shared scarce resource with "
                         "quantifiable marginal objective contribution, prioritize allocation to "
                         "higher marginal-contribution decisions subject to coupling constraints.",
            "structural_pattern": "shared scarce resource + marginal contribution",
            "roles_used": ["resource_pool", "competing_decisions", "objective_contribution"],
            "applicability_conditions": ["linear objective", "shared resource coupling"],
            "complexity": 0.3,
        },
        {
            "statement": "Identify the bottleneck resource first, then allocate around it.",
            "structural_pattern": "bottleneck-first",
            "roles_used": ["resource_pool"],
            "applicability_conditions": [],
            "complexity": 0.2,
        },
    ]


class PatternInducerTest(unittest.TestCase):
    def test_prompt_grounded_in_alignment_not_summary(self):
        cluster, alignment = make_cluster_and_alignment()
        prompt = PatternInducer().build_induction_prompt(cluster, alignment)
        self.assertIn("STRUCTURAL CORRESPONDENCE", prompt)
        self.assertIn("resource_pool", prompt)
        self.assertIn("Do NOT summarize", prompt)

    def test_parse_produces_hypothesis_status(self):
        cluster, alignment = make_cluster_and_alignment()
        hyps = PatternInducer().parse_and_validate(hypotheses_json(), cluster, alignment)
        self.assertEqual(len(hyps), 2)
        for h in hyps:
            self.assertEqual(h.status, "hypothesis")  # never born validated
            self.assertTrue(h.is_hypothesis())
            self.assertEqual(h.source_realization_ids, ["e1", "e2"])

    def test_hypothesis_carries_complexity(self):
        cluster, alignment = make_cluster_and_alignment()
        hyps = PatternInducer().parse_and_validate(hypotheses_json(), cluster, alignment)
        self.assertAlmostEqual(hyps[0].complexity, 0.3)

    def test_empty_statement_dropped(self):
        cluster, alignment = make_cluster_and_alignment()
        payload = hypotheses_json() + [{"statement": "  ", "complexity": 0.1}]
        hyps = PatternInducer().parse_and_validate(payload, cluster, alignment)
        self.assertEqual(len(hyps), 2)

    def test_single_object_tolerated(self):
        cluster, alignment = make_cluster_and_alignment()
        hyps = PatternInducer().parse_and_validate(hypotheses_json()[0], cluster, alignment)
        self.assertEqual(len(hyps), 1)

    def test_unparseable_returns_empty(self):
        cluster, alignment = make_cluster_and_alignment()
        self.assertEqual(PatternInducer().parse_and_validate("junk", cluster, alignment), [])

    def test_complexity_clamped(self):
        cluster, alignment = make_cluster_and_alignment()
        payload = [dict(hypotheses_json()[0], complexity=9.9)]
        hyps = PatternInducer().parse_and_validate(payload, cluster, alignment)
        self.assertEqual(hyps[0].complexity, 1.0)


class LLMBackedInducerTest(unittest.TestCase):
    def test_llm_loop_induces_hypotheses(self):
        cluster, alignment = make_cluster_and_alignment()
        llm = FakeLLMClient(object_responses=[hypotheses_json()])
        hyps = asyncio.run(LLMBackedInducer(llm_client=llm).induce(cluster, alignment))
        self.assertEqual(len(hyps), 2)
        self.assertTrue(all(h.is_hypothesis() for h in hyps))

    def test_llm_loop_requires_client(self):
        cluster, alignment = make_cluster_and_alignment()
        hyps = asyncio.run(LLMBackedInducer(llm_client=None).induce(cluster, alignment))
        self.assertEqual(hyps, [])


if __name__ == "__main__":
    unittest.main()
