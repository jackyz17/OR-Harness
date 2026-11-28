"""Synthetic R prompt/wire boundaries, not measured timeout predictions.

Scripted zero/missing probabilities exercise the existing contract. They do
not calibrate a model or prove that a large search must time out.
"""
import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from or_harness.world_model.contracts import RiskStatement, R_EVENT_DEFINITIONS, R_EVENTS
from or_harness.world_model.provider import HttpChatProvider
from or_harness.world_model.strategy_prediction import STRATEGY_OUTCOME_SYSTEM_PROMPT


class TestRiskReasoningPrompt(unittest.TestCase):
    def test_complexity_scope_and_probability_policy_reach_the_actual_provider(self):
        cases = (
            ('simple-lp', ['Solve the small continuous LP with a supported dual certificate.'], [], 0.0),
            ('unpruned-enumeration', ['Enumerate every tuple without pruning.'], [165, 159, 117, 118], None),
            ('pruned-search', ['Use derived feasibility bounds to prune before evaluating residual subproblems.'], [165, 159, 117, 118], None),
            ('branch-and-bound', ['Solve the coupled binary MILP with branch-and-bound and a solver time limit.'], [2] * 30, None),
        )
        provider = HttpChatProvider('https://synthetic.invalid/v1', 'synthetic-build62-model',
                                    'synthetic-key', temperature=0.17, max_output_tokens=512)
        for name, steps, domains, probability in cases:
            with self.subTest(case=name):
                request = {
                    'prediction_protocol': 'wm-so/1',
                    'candidate': {'strategy_id': 'synthetic-' + name,
                                  'method': {'name': name, 'steps': steps},
                                  'config': {'time_limit': 10}},
                    'prediction_context': {
                        'joint_problem': {'text': 'Synthetic pre-execution instance.',
                                          'domain_cardinalities': domains},
                        'execution_constraints': {'executor': {'timeout_seconds': 30}},
                        'implementation_r_evidence': {'cases': []},
                    },
                }
                event = {'event': 'timeout', 'basis': 'Scripted fixture; no measured execution throughput.'}
                if probability is not None:
                    event['probability'] = probability
                payload = {'risk': {'events': [event]}}
                response = MagicMock()
                response.__enter__.return_value = response
                response.read.return_value = json.dumps({'choices': [{'message': {
                    'content': json.dumps(payload)}, 'finish_reason': 'stop'}]}).encode()
                frozen = copy.deepcopy(request)
                with patch('urllib.request.urlopen', return_value=response) as send:
                    result = provider.predict(request, timeout_s=2)
                send.assert_called_once()
                wire = json.loads(send.call_args.args[0].data)
                prompt = wire['messages'][0]['content']
                self.assertEqual(prompt, STRATEGY_OUTCOME_SYSTEM_PROMPT)
                self.assertEqual(json.loads(wire['messages'][1]['content']), frozen)
                self.assertEqual(request, frozen)
                self.assertEqual(wire['model'], 'synthetic-build62-model')
                self.assertEqual(wire['temperature'], 0.17)
                self.assertEqual(wire['max_tokens'], 512)
                self.assertEqual(result['payload'], payload)
                self.assertEqual(RiskStatement.from_dict(result['payload']['risk']).events[0].probability,
                                 probability)
                text = ' '.join(prompt.split())
                for policy in (
                    'unpruned search space from domain cardinalities before crediting pruning',
                    'preprocessing, decomposition or structural reductions actually supported by the candidate',
                    'constraint coupling', 'per-node work',
                    'A combinatorial upper bound alone does not establish inevitable timeout',
                    'execution_constraints.executor.timeout_seconds',
                    'inner solver time limit that may still deliver a usable result',
                    'evidence for either low or high risk',
                    'absent timeout history alone does not justify zero',
                    'omit probability and explain why',
                    'do not invent precision or impose a risk floor',
                    'Agent analysis, probing, code writing and repeated tool calls are not timeout labels',
                    'their elapsed time belongs to C.remaining_latency_s',
                    'Pure code/API failures and harness wall-clock interruption belong to R',
                    'do not also lower G for the same purely technical hazard',
                    'mathematical correctness alone does not imply G=1',
                    'With sufficient historical evidence, use global_event_base_rates (known labels only)',
                    "an empirical reference point, not the candidate's conditional probability",
                    'requires matching observation-rule versions',
                    'solver-interface experience, comparable implementation steps',
                    'recorded defects and successful implementations',
                    'genuinely reusable verified scripts',
                    'With weak candidate-specific evidence, avoid large unsupported departures',
                    'A measured reference rate is not an arbitrary fixed default',
                    'Do not invent differences between candidates',
                    'implementation_failure merely for solver name/reputation or mathematical difficulty',
                    'explain it in existing basis/notes',
                    'If neither source supports an estimate, omit probability',
                ):
                    self.assertIn(policy, text)
                for event_name in R_EVENTS:
                    self.assertIn(R_EVENT_DEFINITIONS[event_name], prompt)
                self.assertIn('MAXIMUM single-event probability', prompt)
                self.assertIn('NOT a joint failure probability', prompt)

    def test_synthetic_search_size_is_a_bound_not_a_runtime_label(self):
        # Fixture arithmetic only: no solver or performance claim is made.
        cardinalities = [165, 159, 117, 118]
        self.assertEqual(math.prod(cardinalities), 362_200_410)
        self.assertGreater(math.prod(cardinalities), 300_000_000)
        self.assertNotIn('362200410', STRATEGY_OUTCOME_SYSTEM_PROMPT)
        self.assertNotIn('362,200,410', STRATEGY_OUTCOME_SYSTEM_PROMPT)

    def test_agent_guidance_reuses_evidence_without_extra_cross_task_notes(self):
        root = Path(__file__).resolve().parents[2]
        skill = ' '.join((root / 'SKILL.md').read_text().split())
        checks = ' '.join((root / 'references/self_check.md').read_text().split())
        self.assertIn('do not persist extra solving notes for reuse across tasks', skill)
        self.assertIn('Per-task scratch files are allowed', skill)
        self.assertIn('batch necessary reads', skill)
        self.assertIn('only a missing detail', checks)
        self.assertIn('Neither H+=1 nor a missing G label requires another solve', checks)
        self.assertIn('sufficient analytic, dual or other certificate', checks)


if __name__ == '__main__':
    unittest.main()
