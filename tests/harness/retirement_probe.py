"""Deterministic disposable workflow probe; no experiment memory or model."""
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

# Resolve both source and fixtures from the checkout under comparison.
sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd() / "tests/harness"), str(Path.cwd())]

time.time = lambda: 1795737600.0
counter = iter(range(100000))
uuid.uuid4 = lambda: uuid.UUID(hex=hashlib.sha256(str(next(counter)).encode()).hexdigest()[:32])

from g_fixtures import GCase, COUPLED
from or_harness.world_model.contracts import FAITHFUL_G_CONVENTION
from or_harness.world_model.provider import WorldModelProvider


class Provider(WorldModelProvider):
    name = 'retirement-deterministic-fixture'

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        import copy
        self.requests.append(copy.deepcopy(request))
        return {'payload': {
            'benefit': {**FAITHFUL_G_CONVENTION,
                'value': .8 if request['candidate']['strategy_id'] == 'S04' else .6,
                'baseline': {'kind': 'unknown', 'note': 'Synthetic fixed-input probe.'}},
            'cost': {'remaining_latency_s': 1, 'output_tokens': 20, 'tool_calls': 2, 'retries': 0},
            'risk': {'events': [{'event': 'implementation_failure', 'probability': .2},
                               {'event': 'timeout', 'probability': .1},
                               {'event': 'environment_failure', 'probability': .05}]},
            'capability_gain': {'decision': 1,
                'claim': 'A shared factory constraint can prevent independent assignments.',
                'basis': ['Original same-factory requirement and candidate steps.'],
                'verification_conditions': [{'condition': 'Check that BE and CFC share a factory.'}]}},
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1}, 'latency_s': .001}


case = GCase()
case.setUp()
try:
    h = case.h
    provider = Provider()
    h.world_model = h.strategy_predictions.provider = provider
    calls = {name: 0 for name in ('knowledge_targets', 'prediction_reliability_table',
             'evaluate_knowledge_execution', 'evaluate_knowledge_consolidation')}
    for name in calls:
        if hasattr(h, name):
            original = getattr(h, name)
            def counted(*args, _name=name, _original=original, **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)
            setattr(h, name, counted)
    plan = h.plan_next(case.task, 'ep1', candidates=[
        {'strategy_id': sid, 'solver': 'highs', 'method': COUPLED}
        for sid in ('S04', 'S05')], limits={'cost_scales': {
            'remaining_latency_s': 10, 'output_tokens': 100, 'tool_calls': 10, 'retries': 2}})
    prediction = h.strategy_predictions.get(plan['selected_prediction_id'])
    record = case.run_g(prediction)
    checked = case.check_g(record)
    closed = case.close_g()
    job = closed['induction_schedule']['pending_ids'][0]
    material = h.induction_material(induction_id=job)
    reviewed = h.induce(induction_id=job, review={'outcome': 'insufficient_material',
        'conclusion': 'Fixture checks completion and transport, not a publishable increment.'})
    evaluation = next(e for e in closed['evaluations'] if e['prediction_id'] == prediction.prediction_id)
    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        if isinstance(value, str):
            return value.replace(str(case.home), '<fixture_home>')
        return value
    result = {'requests': provider.requests,
        'model_calls': len(provider.requests), 'selected_strategy': prediction.candidate.strategy_id,
        'scores': [row['score'] for row in plan['candidates']],
        'check_state': checked['state'],
        'g': {key: evaluation['benefit'].get(key) for key in (
            'observed', 'metric', 'observation_rule_version', 'label_eligibility')},
        'r': evaluation['risk'].get('scored'),
        'induction': {'scheduled': len(closed['induction_schedule']['pending_ids']),
            'outcome': reviewed['induction_receipt']['outcome'],
            'source_execution': material['induction_gate']['induction_id'] == job},
        'retired_calls': calls}
    Path(sys.argv[1]).write_text(json.dumps(normalized(result), ensure_ascii=False, indent=2, sort_keys=True))
finally:
    case.tearDown()
    case.doCleanups()


if len(sys.argv) > 2:
    import copy
    previous = json.loads(Path(sys.argv[2]).read_text())
    current = json.loads(Path(sys.argv[1]).read_text())
    def diagnostic_sizes(value):
        if isinstance(value, dict):
            return {k: diagnostic_sizes(v) for k, v in value.items()
                    if k not in ('chars', 'before_chars', 'after_chars', 'blocks')}
        if isinstance(value, list):
            return [diagnostic_sizes(v) for v in value]
        return value
    def effective(request):
        value = copy.deepcopy(request)
        context = value['prediction_context']
        for key in ('harness_config_digest', 'prompt_template_version'):
            context['capability_version'].pop(key, None)
        context['notes'] = [n for n in context['notes']
                            if not n.startswith('no structural knowledge target was proposed')]
        for key in ('budget_limits', 'request_size'):
            if key in context:
                context[key] = diagnostic_sizes(context[key])
        return value
    assert [effective(r) for r in previous['requests']] == [effective(r) for r in current['requests']]
    assert previous['requests'][0]['prediction_context']['capability_version']['prompt_template_version'] == 'wm2/2'
    assert current['requests'][0]['prediction_context']['capability_version']['prompt_template_version'] == 'wm-so/1'
    for key in previous.keys() - {'requests', 'retired_calls'}:
        assert previous[key] == current[key], key
    assert not any(current['retired_calls'].values())
    print(json.dumps({'effective_requests_equal': True, 'scores_equal': True,
        'model_calls': current['model_calls'], 'selected_strategy': current['selected_strategy'],
        'g': current['g'], 'induction': current['induction'],
        'retired_calls_before': previous['retired_calls'],
        'retired_calls_after': current['retired_calls']}, ensure_ascii=False))
