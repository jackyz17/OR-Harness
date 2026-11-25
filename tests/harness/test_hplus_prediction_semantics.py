"""build6.2 H+ prompt and wire contracts, using synthetic fixtures only.

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

from g_fixtures import GCase
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

    def test_scripted_coverage_boundaries_preserve_rule_and_execution_evidence(self):
        # These are policy/transport fixtures, not measured model judgements.
        integer_rule = {
            'when': 'A small finite domain of bounded integer variables.',
            'do': 'Enumerate all integer tuples, check constraints and independently recompute the objective.',
            'expect': 'Find a feasible global optimum in that finite integer domain.',
        }
        cases = (
            ('translation', 0, 'Translate the stated limits into constraints.', False, False),
            ('fully-covered', 0, 'Repeat bounded integer enumeration without new evidence.', True, False),
            ('continuous-vertices', 1, 'Verify linear optimality at vertices of a bounded continuous polytope.', True, False),
            ('monotone-reduction', 1, 'Use objective monotonicity to eliminate an integer variable while preserving the optimum.', True, False),
            ('relaxation-integrality', 1, 'Check a structural certificate that LP relaxation retains an integer optimum.', True, False),
            ('execution-only', 1, 'Check a reusable modelling mechanism seen in a similar successful execution.', False, True),
            ('textbook-mechanism', 1, 'Check integral flow structure as a reusable reason to omit integer branching.', False, False),
        )
        provider = HttpChatProvider('https://synthetic.invalid/v1', 'synthetic-build62-model',
                                    'synthetic-key', temperature=0.17, max_output_tokens=512)
        for name, decision, mechanism, published, execution_only in cases:
            with self.subTest(case=name):
                hits = ([{'entry_id': 'synthetic-integer-rule', 'version': 1,
                          'publication': {'state': 'published'}, 'rule_body': integer_rule}]
                        if published else [])
                if execution_only:
                    hits.append({'execution_id': 'synthetic-past-execution',
                                 'evidence_class': 'execution_evidence',
                                 'method_actual': {'name': mechanism}, 'status': 'optimal'})
                request = {
                    'prediction_protocol': 'wm-so/1',
                    'candidate': {'strategy_id': 'synthetic-' + name,
                                  'method': {'name': name, 'steps': [mechanism]}},
                    'prediction_context': {'joint_problem': {'text': 'Synthetic pre-execution structure.'},
                                           'retrieval_evidence': {'hits': hits}},
                }
                gain = {'decision': decision, 'basis': ['Scripted policy boundary: ' + name]}
                if decision:
                    gain.update(claim=mechanism, verification_conditions=[{
                        'condition': 'Check the stated mechanism with minimal feasible evidence.',
                        'check_basis': 'Original task structure and this candidate implementation.'}])
                response = MagicMock()
                response.__enter__.return_value = response
                response.read.return_value = json.dumps({'choices': [{'message': {
                    'content': json.dumps({'capability_gain': gain})}, 'finish_reason': 'stop'}]}).encode()
                frozen = copy.deepcopy(request)
                with patch('urllib.request.urlopen', return_value=response) as send:
                    result = provider.predict(request, timeout_s=2)
                send.assert_called_once()
                wire = json.loads(send.call_args.args[0].data)
                self.assertEqual(wire['messages'][0]['content'], STRATEGY_OUTCOME_SYSTEM_PROMPT)
                self.assertEqual(json.loads(wire['messages'][1]['content']), frozen)
                self.assertEqual(request, frozen)
                self.assertEqual(wire['model'], 'synthetic-build62-model')
                self.assertEqual(wire['temperature'], 0.17)
                self.assertEqual(wire['max_tokens'], 512)
                self.assertEqual(result['payload']['capability_gain'], gain)
                self.assertEqual(signal_from_payload(gain)['decision'], decision)
        prompt = flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)
        for policy in (
            'When (conditions, mathematical structure and prerequisites)',
            'Do (core mathematical or operational mechanism)',
            'Expect (effect, guarantee or verification purpose)',
            'All its substantive content must be covered',
            'Successful use is not proof of saved knowledge',
            'Execution Evidence records what happened; Strategic Knowledge records the transferable mechanism',
            'Partial coverage can leave a learning opportunity',
            'textbook practice, a standard OR technique or a conventional solver/verification method is background, never by itself a reason for 0',
            'an empty library or missing review history is NEVER sufficient for 1',
            'ONLY a currently visible PUBLISHED strategy rule can establish existing-knowledge coverage',
            'Execution Evidence is induction material, not completed induction',
            'treat them as support for review, not a reason to reject it',
            "Do not justify 0 solely by 'existing execution evidence already covers the method', 'historical successful executions' or 'existing solver capability'",
        ):
            self.assertIn(policy, prompt)


class TestHistoricalEvidenceIsNotPublishedKnowledge(GCase):
    def test_empty_strategy_bank_keeps_multiple_supporting_executions_in_actual_request(self):
        # Scripted forecast + real local context construction. This proves
        # visibility and the binary contract, not a real model's judgement.
        method = {'name': 'equality coupling reduction', 'steps': [
            'Use one shared factory decision for equality-coupled demands, '
            'reducing independent choices while preserving feasible assignments.']}
        executions = []
        for index in range(3):
            self.task['task_id'] = 'synthetic-coupled-history-' + str(index)
            episode = 'history-' + str(index)
            p = self.predict_g(method, episode)
            record = self.run_g(p)
            self.check_g(record)
            self.close_g(episode)
            executions.append(record.execution_id)
        self.assertEqual(self.h.sbank.count(), 0)
        claim = ('Equality-coupled assignments may share a decision, preserving '
                 'the feasible set while reducing assignment search.')
        self.provider.payload['capability_gain'] = {
            'decision': 1, 'claim': claim,
            'basis': ['Synthetic execution cases support a concrete mechanism; no published rule exists.'],
            'verification_conditions': [{'condition': 'Check the equality contraction argument and its premises.'}],
        }
        self.task['task_id'] = 'synthetic-coupled-current'
        current = self.predict_g(method, 'next')
        request = self.provider.requests[-1]
        hits = request['prediction_context']['retrieval_evidence']['hits']
        recalled = {str(hit.get('evidence_id')) for hit in hits
                    if hit.get('evidence_class') == 'execution_fact'}
        self.assertTrue(set(executions) <= recalled)
        self.assertFalse(any(hit.get('evidence_class') == 'verified_knowledge' for hit in hits))
        for hit in hits:
            if hit.get('evidence_id') in executions:
                self.assertIn('shared factory decision', json.dumps(hit))
        self.assertEqual(current.capability_gain.decision, 1)
        self.assertEqual(current.capability_gain.claim, claim)
        self.assertEqual(current.trace.model_info['hplus']['decision'], 1)
        self.assertEqual(self.h.sbank.count(), 0)


if __name__ == "__main__":
    unittest.main()
