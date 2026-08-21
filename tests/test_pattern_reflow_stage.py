"""Tests for Phase 4.1 prompt injection + citation parsing (modeling_stage).

After the unification: all modeling-bank records are peers, all labeled [En],
all citable with [uses En]. No more [Pn]/[En] split.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.modeling.modeling_stage import (
    StructuredModelingStage,
)
from or_experience_bank.retrieval.modeling_retriever import PlanningPriors


def make_priors():
    priors = PlanningPriors()
    priors.records = [
        {
            "experience_id": "e1",
            "title": "resource allocation principle",
            "modeling_aspect": "constraint",
            "method": {"action_template": "prioritize higher marginal contribution on a shared scarce resource"},
            "status": "validated",
        },
        {
            "experience_id": "e2",
            "title": "warehouse capacity allocation",
            "modeling_aspect": "constraint",
            "method": {"action_template": "allocate stock x[i] to capacity"},
            "status": None,
        },
    ]
    priors.labels = {"E1": "e1", "E2": "e2"}
    return priors


class PromptInjectionTest(unittest.TestCase):
    def test_prompt_contains_priors_when_supplied(self):
        stage = StructuredModelingStage()
        prompt = stage.build_modeling_prompt("a resource allocation problem", None, make_priors())
        self.assertIn("Past modeling experiences", prompt)
        self.assertIn("[E1]", prompt)
        self.assertIn("marginal contribution", prompt)
        self.assertIn("[E2]", prompt)
        self.assertIn("[uses En]", prompt)  # citation instruction

    def test_prompt_unchanged_without_priors(self):
        stage = StructuredModelingStage()
        prompt = stage.build_modeling_prompt("a problem", None, None)
        self.assertNotIn("Past modeling experiences", prompt)
        self.assertIn("PROBLEM:", prompt)

    def test_prompt_handles_empty_priors(self):
        stage = StructuredModelingStage()
        prompt = stage.build_modeling_prompt("a problem", None, PlanningPriors())
        self.assertNotIn("Past modeling experiences", prompt)


class CitationExtractionTest(unittest.TestCase):
    def test_extracts_cited_experience_ids(self):
        priors = make_priors()
        think = "This is a shared scarce resource problem [uses E1] so we prioritize marginal contribution."
        cited = StructuredModelingStage.extract_cited_principle_ids(think, priors)
        self.assertEqual(cited, ["e1"])

    def test_multiple_and_deduped(self):
        priors = make_priors()
        think = "[uses E1, E1] and again [uses E2]"
        cited = StructuredModelingStage.extract_cited_principle_ids(think, priors)
        self.assertEqual(cited, ["e1", "e2"])

    def test_unknown_tag_ignored(self):
        priors = make_priors()
        # LLM cannot invent a citation: E9 is not in the injected priors
        think = "[uses E9] something"
        self.assertEqual(StructuredModelingStage.extract_cited_principle_ids(think, priors), [])

    def test_no_priors_no_citations(self):
        think = "[uses E1]"
        self.assertEqual(StructuredModelingStage.extract_cited_principle_ids(think, None), [])


if __name__ == "__main__":
    unittest.main()
