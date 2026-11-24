"""build6.1 H+ prompt and wire contracts, using synthetic fixtures only.

Responses below are SCRIPTED, not model judgements or build6 replay results.
They check that the actual HTTP system prompt states each policy boundary,
frozen inputs/parameters reach the wire unchanged, and the existing binary
contract preserves the supplied decision and hypothesis fields.
"""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from or_harness.world_model.contracts import CapabilityGain
from or_harness.world_model.hplus import signal_from_payload
from or_harness.world_model.provider import HttpChatProvider
from or_harness.world_model.strategy_prediction import STRATEGY_OUTCOME_SYSTEM_PROMPT


def flat(text):
    return " ".join(text.split())


# Synthetic pre-execution descriptions and scripted responses. In particular,
# the facility fixture has no optimum, executed verification script or review.
CASES = (
    ("plain_translation", 0,
     "Write variables, the stated capacity constraint and profit objective.",
     "No specific reusable mechanism beyond literal task translation.",
     "mere task translation without a specific transferable mechanism"),
    ("classical_structural_applicability", 1,
     "Use min-cost flow for a network with integral capacities and conserved flow.",
     "Integral network structure may avoid integer branching in future matching tasks.",
     "a classical method's structural applicability conditions"),
    ("foreseeable_optimality_verification", 1,
     "Formulate and solve a fixed-charge facility-location MILP.",
     "For few facilities with a residual transportation LP, opening-subset enumeration "
     "may give an affordable independent optimality check on future tasks.",
     "reasonably foreseeable checking, diagnosis and verification"),
    ("fully_covered_without_new_evidence", 0,
     "Use the same endpoint rule under exactly its published premises.",
     "synthetic-rule@2 fully covers this mechanism; the same check offers no meaningful new evidence.",
     "fully covers the mechanism and this run offers no meaningful new evidence"),
    ("published_scope_revision", 1,
     "Test a lower endpoint for minimization where the published rule covers only maximization.",
     "The objective sign may require a scope correction to the current endpoint rule.",
     "scope extension, correction or counterexample"),
    ("empty_library_without_hypothesis", 0,
     "Call a generic MILP solver; no reusable mechanism or boundary is identified.",
     "An empty library alone supplies no concrete learning hypothesis.",
     "no concrete reasonable learning hypothesis"),
    ("opportunity_without_publication_guarantee", 1,
     "Use a typed solver-status adapter with a small API receipt check.",
     "A status-enum conversion may prevent a repeatable API failure; publication remains uncertain.",
     "not a guarantee of publishable knowledge"),
    ("independent_validation_of_existing_mechanism", 1,
     "Check a previously weakly supported reduction on a distinct structural instance.",
     "Independent evidence may strengthen or refute the current reduction's applicability.",
     "meaningful independent validation of an existing weakly supported mechanism"),
)


class TestHplusPredictionSemantics(unittest.TestCase):
    def test_scripted_boundaries_reach_the_actual_http_system_prompt(self):
        provider = HttpChatProvider(
            "https://synthetic.invalid/v1", "synthetic-build61-model", "synthetic-key",
            temperature=0.17, max_output_tokens=512)
        for name, decision, method, hypothesis, policy in CASES:
            with self.subTest(case=name):
                published = name in (
                    "fully_covered_without_new_evidence", "published_scope_revision",
                    "independent_validation_of_existing_mechanism")
                request = {
                    "prediction_protocol": "wm-so/1",
                    "request_kind": "strategy_outcome_prediction",
                    "candidate": {"strategy_id": "synthetic-" + name,
                                  "method": {"name": name, "steps": [method]}},
                    "prediction_context": {
                        "joint_problem": {"text": (
                            "Synthetic pre-execution instance: four potential facilities; "
                            "fixing openings leaves a transportation LP."
                            if name == "foreseeable_optimality_verification" else
                            "Synthetic pre-execution task for " + name)},
                        "retrieval_evidence": {"hits": ([{
                            "entry_id": "synthetic-rule", "version": 2,
                            "publication": {"state": "published"},
                            "rule_body": {"when": "stated synthetic premises",
                                          "do": "the currently recorded mechanism",
                                          "expect": "a scoped reusable decision"},
                        }] if published else [])},
                        "hplus_feedback": {"settled_examples": []},
                    },
                }
                gain = {"decision": decision, "basis": [hypothesis]}
                if decision:
                    gain.update(
                        claim=hypothesis,
                        verification_conditions=[{
                            "condition": "Collect one check bearing on this synthetic hypothesis.",
                            "check_basis": "Minimum feasible evidence from this candidate's run.",
                        }])
                response = MagicMock()
                response.__enter__.return_value = response
                response.read.return_value = json.dumps({
                    "choices": [{"message": {"content": json.dumps({"capability_gain": gain})},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }).encode()
                frozen = copy.deepcopy(request)
                with patch("urllib.request.urlopen", return_value=response) as send:
                    result = provider.predict(request, timeout_s=2)
                send.assert_called_once()
                wire = json.loads(send.call_args.args[0].data)
                system = wire["messages"][0]
                self.assertEqual(system["role"], "system")
                self.assertEqual(system["content"], STRATEGY_OUTCOME_SYSTEM_PROMPT)
                self.assertIn(policy, flat(system["content"]))
                self.assertEqual(json.loads(wire["messages"][1]["content"]), frozen)
                self.assertEqual(request, frozen)
                self.assertEqual(wire["model"], "synthetic-build61-model")
                self.assertEqual(wire["temperature"], 0.17)
                self.assertEqual(wire["max_tokens"], 512)
                parsed = result["payload"]["capability_gain"]
                self.assertEqual(parsed, gain)
                self.assertEqual(CapabilityGain.from_dict(parsed).decision, decision)
                signal = signal_from_payload(parsed)
                self.assertEqual(signal["decision_status"], "explicit")
                self.assertEqual(signal["decision"], decision)

    def test_feedback_does_not_turn_unknown_or_missing_evidence_into_global_negatives(self):
        followup = flat(STRATEGY_OUTCOME_SYSTEM_PROMPT.split(
            "CAPABILITY-GAIN FOLLOW-UP:", 1)[1].split("Do NOT fabricate evidence", 1)[0])
        for policy in (
            "only to a specifically comparable mechanism",
            "insufficient evidence, unfinished induction or unobserved work",
            "execution failure that prevented verification",
            "Do not globally tighten H+",
            "empty feedback does not imply no learning opportunity",
        ):
            self.assertIn(policy, followup)

    def test_candidate_scope_and_publication_proof_are_not_conflated(self):
        prompt = flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)
        for policy in (
            "Do not borrow another candidate's H+",
            "never use post-execution facts to revise this forecast",
            "Do not require advance proof of the mechanism or transferability",
            "publication standards remain the offline review's responsibility",
            "no default full re-solve, large experiment or extra execution branch",
        ):
            self.assertIn(policy, prompt)
        self.assertNotIn("STANDARD TASK TRANSLATION IS NOT AN INCREMENT", prompt)
        self.assertNotIn("ordinary, generally known step", prompt)


if __name__ == "__main__":
    unittest.main()
