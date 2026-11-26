"""H+ generalization, prompt and wire contracts, using synthetic fixtures only.

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
     "Use 0 for mere task translation, ordinary solver use or routine checking when no specific "
     "transferable mechanism or finding is identified"),
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
    def test_prospective_cases_reach_the_provider_without_new_output_fields(self):
        # The decisions are scripted transport/contract fixtures, not model
        # judgements. Histories describe past runs, never this candidate's
        # future execution. No teaching problem or reference answer is sent.
        equality_rule = {
            'when': 'Equality-linked decisions have compatible domains and a linear objective.',
            'do': 'Use one variable per equality group, intersect member domains and aggregate objective coefficients.',
            'expect': 'Preserve feasible assignments and objective values with fewer independent choices.',
        }
        integer_rule = {
            'when': 'All decisions have bounded finite integer domains.',
            'do': 'Enumerate integer tuples and check each tuple.',
            'expect': 'Find an optimum over those finite integer domains.',
        }

        def execution(name, family, step, status='optimal'):
            return {'execution_id': 'synthetic-' + name, 'evidence_class': 'execution_evidence',
                    'profile': {'family': family}, 'method_actual': {'steps': [step]}, 'status': status}

        group_step = equality_rule['do']
        battery_task = ('Minimize a linear charge over bounded battery states. '
                        'States in each equality group must agree; intersect their bounds.')
        histories = [execution('team-groups', 'volunteer_assignment',
            'Equality-linked team assignments share a binary variable; intersect domains and sum their linear costs.')]
        delivery_history = [
            execution('writer-failed', 'allocation', 'Wrote objective instead of objective_value.', 'error'),
            execution('writer-repaired', 'allocation', 'Map actual status and write finite objective_value once.'),
            execution('writer-reused', 'scheduling', 'Reuse the status and objective mapping for result delivery.'),
        ]
        # name, task family/text, current operation, executions, published rule,
        # scripted decision, hypothesis/basis, minimum check.
        cases = (
            ('cross-domain-roles', 'battery_balancing', battery_task, group_step,
             histories, None, 1,
             'Equality-group contraction may preserve domains and linear cost across different task roles.',
             'Inspect the equality substitution, intersected bounds and aggregated objective.'),
            ('similar-name-different-mechanism', 'continuous_lp',
             'Minimize a linear objective over a nonempty bounded continuous polytope.',
             'Enumerate vertices to certify continuous linear optimality.', [], integer_rule, 1,
             'Continuous vertex optimality needs premises and a guarantee not supplied by finite integer enumeration.',
             'Check linearity, boundedness and the vertex-optimality argument.'),
            ('execution-evidence-without-rule', 'resource_allocation',
             'Deliver the selected solver result through the existing result protocol.',
             'Centralize actual status and finite objective mapping across solver interfaces.',
             delivery_history, None, 1,
             'The failed/repaired writer difference may support a reusable result-delivery technique; no rule is saved.',
             'Inspect the current status mapping, finite objective and protocol receipt.'),
            ('ordinary-translation', 'capacity_planning',
             'The task explicitly states total production must not exceed capacity.',
             'Translate the stated capacity into a linear constraint.', [], None, 0,
             'No specific reusable mechanism or boundary beyond the required task translation was identified.', None),
            ('published-coverage', 'battery_balancing', battery_task, group_step,
             histories, equality_rule, 0,
             'Visible published synthetic-rule@2 covers the equality premises, domain intersection, '
             'cost aggregation and preservation guarantee; this run adds no meaningful evidence.', None),
            ('semantic-domain-choice', 'purchasing',
             'Buy a divisible bulk ingredient by weight; a whole-package restriction applies only to sealed packages.',
             'Determine each variable domain from its purchasing granularity, then solve the continuous LP.',
             [], None, 1,
             'Distinguishing divisible quantities from package counts may prevent a false integer restriction.',
             'Check the variable domain against the stated purchasing units and divisibility.'),
            ('semantic-domain-choice-alternative', 'purchasing',
             'Buy a divisible bulk ingredient by weight; a whole-package restriction applies only to sealed packages.',
             'Use an integer package variable only for sealed packages; keep bulk weight continuous.',
             [], None, 1,
             'The same domain interpretation mechanism is testable with this alternative candidate.',
             'Check both variable domains against the original purchasing units.'),
            ('foreseeable-api-repair', 'allocation',
             'Return the actual solver objective and status for a linear allocation model.',
             'Use a supported status accessor and test the status-to-result mapping after a known interface failure.',
             delivery_history, None, 1,
             'The prior failure and repair may support a reusable status mapping mechanism.',
             'Inspect the selected accessor, status and result receipt.'),
            ('cold-start-structural-reduction', 'bounded_inventory',
             'For fixed Y, feasible integer X is a finite interval and the minimization objective increases in X.',
             'Select the lower feasible X endpoint for each Y instead of enumerating every X.', [], None, 1,
             'Monotonicity may remove X enumeration while preserving each fixed-Y optimum despite no history.',
             'Check the original interval bounds, monotonicity and selected feasible endpoints.'),
        )
        provider = HttpChatProvider('https://synthetic.invalid/v1', 'synthetic-prospective-model',
                                    'synthetic-key', temperature=0.17, max_output_tokens=512)
        for name, family, task, step, history, rule, decision, basis, condition in cases:
            with self.subTest(case=name):
                hits = copy.deepcopy(history)
                if rule is not None:
                    hits.append({'entry_id': 'synthetic-rule', 'version': 2,
                        'publication': {'state': 'published'}, 'rule_body': copy.deepcopy(rule)})
                request = {'prediction_protocol': 'wm-so/1',
                    'candidate': {'strategy_id': 'synthetic-' + name,
                                  'method': {'name': name, 'steps': [step]}},
                    'prediction_context': {
                        'joint_problem': {'text': task, 'profile': {'family': family}},
                        'retrieval_evidence': {'hits': hits},
                        'hplus_feedback': {'settled_examples': []}}}
                gain = {'decision': decision, 'basis': [basis]}
                if decision:
                    gain.update(claim=basis, verification_conditions=[{
                        'condition': condition, 'check_basis': 'This candidate and its original task; use minimum sufficient evidence.'}])
                response = MagicMock()
                response.__enter__.return_value = response
                response.read.return_value = json.dumps({'choices': [{'message': {
                    'content': json.dumps({'capability_gain': gain})}, 'finish_reason': 'stop'}]}).encode()
                frozen = copy.deepcopy(request)
                with patch('urllib.request.urlopen', return_value=response) as send:
                    result = provider.predict(request, timeout_s=2)
                send.assert_called_once()
                wire = json.loads(send.call_args.args[0].data)
                prompt = wire['messages'][0]['content']
                self.assertEqual(prompt, STRATEGY_OUTCOME_SYSTEM_PROMPT)
                self.assertEqual(json.loads(wire['messages'][1]['content']), frozen)
                self.assertEqual(request, frozen)
                self.assertEqual((wire['model'], wire['temperature'], wire['max_tokens']),
                                 ('synthetic-prospective-model', 0.17, 512))
                self.assertEqual(result['payload']['capability_gain'], gain)
                parsed = CapabilityGain.from_dict(result['payload']['capability_gain'])
                self.assertEqual(parsed.decision, decision)
                self.assertEqual(signal_from_payload(gain)['decision'], decision)
                self.assertEqual(set(gain), {'decision', 'basis', 'claim', 'verification_conditions'}
                                 if decision else {'decision', 'basis'})
                for policy in (
                    'Identify a concrete problem-understanding, modelling, solving, implementation or verification decision',
                    'feasibility, optimality or solution quality, search space, computational work or reliability',
                    'align their mathematical or operational roles with the current task',
                    'Different domains or family labels do not rule out a shared mechanism',
                    'similar solver/algorithm names or terms do not establish one',
                    'Abstract case-specific entities by role and capacities or coefficients as parameters',
                    'a failure-and-repair implementation technique',
                    'which future decision this mechanism could improve, why it might work',
                    'support, refine or challenge',
                    'Future premises must be recognizable before solving',
                    'prospective generalization, not completed induction',
                    'Do not require advance proof of the mechanism or transferability, multiple independent executions',
                    'claim states one concise potential reusable mechanism and the future decision it could change',
                    'when relevant history contributes, the shared mathematical or operational relationship',
                    'minimum feasible check that could support, refine or challenge that mechanism in this run',
                ):
                    self.assertIn(policy, flat(prompt))
                self.assertNotIn('SIRL', prompt)

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

    def test_learning_decision_order_and_candidate_independence(self):
        prompt = flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)
        steps = ('1. GENERALIZE FIRST.', '2. STATE THE LEARNING HYPOTHESIS.',
                 '3. CHECK THE ACTUAL KNOWLEDGE GAP.', '4. CHECK EVIDENCE AND DECIDE.')
        offsets = [prompt.index(step) for step in steps]
        self.assertEqual(offsets, sorted(offsets))
        for policy in (
            'semantic disambiguation grounded in original task inputs',
            'without a hidden answer',
            'a shared understanding or verification step can support the same hypothesis for multiple candidates',
            'A previous H+ judgement (positive or negative) is not evidence of rule coverage',
            'A consequential, testable interpretation or repair is not mere translation',
            'name the decision considered and why no reusable mechanism or feasible evidence was found',
        ):
            self.assertIn(policy, prompt)

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
