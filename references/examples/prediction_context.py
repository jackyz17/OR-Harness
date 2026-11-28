"""Share a frozen wm-so/1 input across candidate predictions.

Run from the repository root: PYTHONPATH=src python references/examples/prediction_context.py
This uses a deterministic local provider and a temporary memory directory.
"""
import copy
import json
import tempfile

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.contracts import ActionSpec, FAITHFUL_G_CONVENTION
from or_harness.world_model.provider import WorldModelProvider


class Provider(WorldModelProvider):
    name = 'frozen-context-example'

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        return {'payload': {
            'benefit': {**FAITHFUL_G_CONVENTION, 'value': .7,
                        'baseline': {'kind': 'unknown', 'note': 'Example has no prior cases.'}},
            'capability_gain': {'decision': 0, 'basis': ['This example tests input transport.']}},
            'usage': {}, 'latency_s': 0}


def main():
    task = {'task_id': 'context-example', 'family': 'allocation',
            'text': 'Allocate a divisible resource within the stated capacity.'}
    candidates = [ActionSpec('execute_strategy', task['task_id'], strategy_id=sid,
        solver='highs', params={'time_limit': 60},
        method={'name': name, 'steps': [step]}) for sid, name, step in (
            ('lp', 'linear allocation', 'Build a continuous linear allocation model.'),
            ('flow', 'network allocation', 'Express conserved resource flow on a network.'))]
    provider = Provider()
    with tempfile.TemporaryDirectory(prefix='or-context-example-') as home:
        harness = ORHarness(home=home, world_model=provider, embedding=LocalHashEmbeddingBackend())
        try:
            context = harness.build_prediction_context(task, 'example', candidates=candidates)
            original = copy.deepcopy(context.to_dict())
            for candidate in candidates:
                prediction = harness.predict_strategy_outcome(task, candidate, 'example', context=context)
                assert prediction.trace.input_snapshot_id == context.snapshot_id
                assert prediction.trace.model_info['prediction_context_id'] == context.context_id
            assert len(provider.requests) == len(candidates)
            assert context.to_dict() == original
            for request in provider.requests:
                assert request['prediction_protocol'] == 'wm-so/1'
                assert request['prediction_context']['context_id'] == context.context_id
                assert 'state' not in request
                assert 'candidate_knowledge_targets' not in request
                assert 'prediction_reliability' not in request
            restored = harness.get_prediction_context(context.context_id)
            assert restored.to_dict() == original
            print(json.dumps({'protocol': 'wm-so/1', 'predictions': len(provider.requests),
                'shared_context_id': context.context_id,
                'original_task': restored.joint.text,
                'input_unchanged': True}, ensure_ascii=False, indent=2))
        finally:
            harness.close()


if __name__ == '__main__':
    main()
