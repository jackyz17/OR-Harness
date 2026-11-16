"""World-model M3 tests: the strategy-outcome prediction service.

What this file asserts, one behaviour per test class:

1. the three pre-fixes hold: current prediction and planning really carry
   the retrieval evidence; an explicit-CIR context builds and is reused
   through the SAME CIR (with conflicts refused); the historical path
   refuses an external recall result and does not re-derive knowledge
   targets from today's bank;
2. one comparison gathers ONE context; candidates keep their own config
   and identity; later bank writes do not change the saved request;
3. the new protocol really reaches the provider (request shape, fixed
   identity, content not ids) and parses into the new contract; every
   failure state (not configured, empty payload, invalid JSON, NaN,
   out-of-range probability, wrong type) is honest;
4. an unexecuted strategy window can be validly predicted but not scored;
   incomplete cost is omitted on a shared basis; a business objective is not execution cost;
   a model self-report is never relabeled as measured;
5. predictions change the suggestion, only an explicit choice changes the
   selection; same strategy_id with different configs does not cross-bind;
   the fallback stays available and explains itself;
6. real call costs include failed calls and are counted once; budget and
   call limits really bind; one real execution binds; unexecuted
   candidates generate no facts.
"""

from decision_fixtures import bind_legacy, execute_scored, planned_prediction

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.contracts import (  # noqa: E402
    validate_strategy_outcome,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


def _cir():
    return {
        "entities": [{"name": "depot", "kind": "site"},
                     {"name": "route", "kind": "route"}],
        "decisions": [{"name": "load", "kind": "binary", "indexes": ["t"]}],
        "constraints": [{"id": "C1", "kind": "capacity",
                         "expr": "load[t] <= cap"}],
        "relations": [{"source": "load", "target": "depot",
                       "type": "uses_resource", "evidence": "declared"}],
        "coupling_groups": [], "issues": [],
    }


#: A well-formed wm-so/1 payload: benefit with baseline, cost on two dims,
#: two risk events, uncertainty as a self-report.
GOOD_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [
        {"event": "model_invalid", "probability": 0.2,
         "severity": 1.0, "severity_unit": "attempts",
         "basis": "no verified knowledge for this cell"},
        {"event": "budget_exhausted", "probability": 0.05},
    ]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
    "evidence_basis": ["retrieval_evidence.hits[0]"],
}


class StubProvider(WorldModelProvider):
    """A stub provider returning a fixed payload, recording requests."""

    name = "stub-strategy-outcome"

    def __init__(self, payload=GOOD_PAYLOAD, usage=None):
        self.payload = payload
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


class StrategyCase(HarnessTestCase):
    """A harness with an injected local embedding backend (hermetic)."""

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def solve(self, task, strategy="S04", objective=100.0,
              episode_id="ep1"):
        from pathlib import Path
        work = Path(self.home) / f"ws_{task['task_id']}_{strategy}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = execute_scored(self.h,task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record


class UnifiedProvider(WorldModelProvider):
    name='unified-solving-fixture'
    def __init__(self):
        self.requests=[];self.decision=0;self.g=.8;self.r=.2;self.cost=3
    def predict(self,request,timeout_s=None):
        import copy
        self.requests.append(copy.deepcopy(request))
        convention=request['benefit_convention']
        return {'payload':{'benefit':{**{k:convention[k] for k in ('kind','metric','unit')},'value':self.g,
            'baseline':{'kind':'unknown','note':'No comparable prior result.'}},
            'cost':{'tool_calls':request['candidate']['config'].get('seed',self.cost)},
            'risk':{'events':[] if self.r is None else [{'event':'implementation_failure','probability':self.r}]},
            'capability_gain':{'decision':self.decision,'claim':'A checked interval endpoint may extend the current rule.',
                'basis':['Actual candidate steps and frozen library'],
                'verification_conditions':[{'condition':'Check the delivered integer result.'}]}},'usage':None,'latency_s':0.001}


class TestUnifiedSolving(HarnessTestCase):
    """Use the public planner, executor and close-out with no test bypass."""
    def setUp(self):
        super().setUp()
        from pathlib import Path
        self.provider=UnifiedProvider()
        self.h=ORHarness(home=self.home,world_model=self.provider,embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        self.task={'task_id':'unified','family':'assignment','description':'Return the integer X=1.'}
        self.method={'name':'one integer','steps':['Set X=1 and return its objective.']}
        self.ws=Path(self.home)/'work';self.ws.mkdir()
        self.code=self.ws/'solve.py'
        self.code.write_text("import json\njson.dump({'status':'optimal','objective_value':1,'variables':{'X':1}},open('result.json','w'))\n")

    def candidate(self,name='one',**config):
        return {'action_type':'execute_strategy','strategy_id':name,'solver':'python','method':self.method,'config':config}

    def plan(self,candidates=None,**overrides):
        return self.h.plan_next(self.task,'E',candidates=candidates or [self.candidate()],limits={
            'cost_weights':{'tool_calls':1},'cost_scales':{'tool_calls':30},**overrides})

    def run_selected(self,plan):
        return self.h.execute(self.task,None,str(self.code),str(self.ws),prediction_id=plan['selected_prediction_id'])

    def assert_refused(self,prediction_id=None,**changes):
        from unittest.mock import patch
        args={'task':self.task,'code_path':str(self.code),'workspace':str(self.ws),'prediction_id':prediction_id,**changes}
        count=len(self.h.actions.query(action_type='execute_strategy'))
        with patch.object(self.h,'_run_executor') as run,self.assertRaises(ValueError):self.h.execute(**args)
        run.assert_not_called()
        self.assertEqual(len(self.h.actions.query(action_type='execute_strategy')),count)

    def test_single_candidate_U_is_saved_and_direct_execute_adopts_once(self):
        import math
        original_predict = self.provider.predict
        def without_usage(request, timeout_s=None):
            response = original_predict(request, timeout_s)
            return {**response, 'usage': None, 'latency_s': None}
        from unittest.mock import patch
        with patch.object(self.provider, 'predict', side_effect=without_usage):
            plan=self.plan()
        p=plan['selected_prediction_id']
        prediction=self.h.strategy_predictions.get(p)
        self.assertIsNone(prediction.trace.call_cost)
        self.assertEqual(prediction.trace.model_info['decision_action_id'],plan['decision_action_id'])
        self.assertNotIn('charged_to_parent_action',prediction.trace.model_info)
        self.assertIsNone(plan['planning_cost'])
        self.assertNotIn('selected_plan',self.h._episode_progress('unified','E'))
        self.assertIsNotNone(p)
        self.assertAlmostEqual(plan['selected_score']['utility'],.8-math.log1p(3)/math.log1p(30)-.2,places=6)
        decision=self.h.actions.get(plan['decision_action_id'])
        self.assertEqual(decision.outcome['selected_prediction_id'],p)
        self.assertEqual(decision.outcome['candidates'],plan['candidates'])
        self.assertEqual(decision.outcome['selected_score'],plan['selected_score'])
        self.assertEqual(self.h.bank.count(),0)
        self.assertEqual(self.h.bank.pending(task_id='unified'),[])
        self.assertIsNone(decision.outcome.get('adopted_by_action_id'))
        r=self.run_selected(plan)
        self.assertEqual(len(self.provider.requests),1)
        self.assertEqual(self.h.actions.get(r.action_id).params['scored_adoption']['prediction_id'],p)
        choices=[a for a in self.h.actions.query(action_type='select_strategy') if (a.outcome or {}).get('adopted_by_action_id')]
        self.assertEqual(len(choices),2)  # one decision pointer and one actual choice
        self.assertEqual(sum((a.outcome or {}).get('adoption_source')=='execute' for a in choices),1)
        self.assertTrue(r.execution_features['prediction_binding']['bound'])

    def test_shared_context_and_stable_candidate_independent_cost_scale(self):
        import math
        low=self.plan([self.candidate('low',seed=3)])
        high=self.plan([self.candidate('high',seed=30)])
        multi=self.plan([self.candidate('low',seed=3),self.candidate('high',seed=30)])
        self.assertLess(low['selected_score']['cost_normalized'],high['selected_score']['cost_normalized'])
        self.assertAlmostEqual(low['selected_score']['cost_normalized'],math.log1p(3)/math.log1p(30),places=6)
        self.assertEqual(multi['candidates'][0]['score']['cost_normalized'],low['selected_score']['cost_normalized'])
        self.assertEqual(multi['candidates'][1]['score']['cost_normalized'],high['selected_score']['cost_normalized'])
        self.assertEqual(multi['selected_prediction_id'],multi['candidates'][0]['prediction_id'])
        self.assertEqual(self.provider.requests[-1]['prediction_context']['context_id'],
                         self.provider.requests[-2]['prediction_context']['context_id'])
        self.assertEqual(len(self.provider.requests),4)

    def test_no_scale_omits_C_and_missing_R_is_partial_without_penalty(self):
        self.provider.r=None
        plan=self.plan(cost_scales={})
        score=plan['selected_score']
        self.assertEqual(score['utility'],.8)
        self.assertEqual(score['scoring_basis']['terms'],['G'])
        self.assertIn('scale',str(score['scoring_basis']['omitted_cost_dimensions']))
        self.run_selected(plan)

    def test_Hplus_never_changes_U_and_explicit_zero_is_valid(self):
        self.provider.g=0;self.provider.r=0;self.provider.cost=0
        self.provider.decision=0;a=self.plan()
        self.provider.decision=1;b=self.plan()
        self.assertEqual(a['selected_score']['utility'],0)
        self.assertEqual(a['selected_score']['utility'],b['selected_score']['utility'])
        self.run_selected(b)

    def test_resource_peak_comes_from_applicable_pre_decision_measured_history(self):
        import time
        self.h.execution_context={'host':'unit-runner','model':'executor/v1','phase':'initial_modelling'}
        first=self.plan(cost_scales={});record=self.run_selected(first)
        self.h.check_task_result(record.execution_id,{'reference_objective':1,'reference_source':'independent'})
        self.h.record(record)
        decision=self.h.actions.get(first['decision_action_id']);end=time.time()
        self.h.report_action('finish_task',self.task,'E',started_at=end,ended_at=end,host_usage={
            'schema':'or-host-usage/1','host':'unit-runner','model':'executor/v1','scope':'remaining_to_task_end',
            'report_id':'one-measured-window','window':{'started_at':decision.started_at,'ended_at':end,
                'anchor_action_id':decision.action_id,'complete':True,'end_event':'finish_task'},
            'tool_calls':30,'measured':['tool_calls']})
        self.h.close_episode(self.task['task_id'],'E')
        self.task={**self.task,'task_id':'next-original-task'}
        second=self.plan(cost_scales={})
        score=second['selected_score']
        self.assertEqual(score['scoring_basis']['cost_scales']['tool_calls'],30)
        source=score['scoring_basis']['cost_scale_sources']['tool_calls']
        self.assertEqual(source['source'],'pre_decision_measured_history')
        self.assertTrue(source['source_refs'])
        self.assertEqual(source['n'],1)

    def test_negative_U_still_uses_existing_permission_rules(self):
        self.provider.cost=30;plan=self.plan()
        self.assertLess(plan['selected_score']['utility'],0)
        self.run_selected(plan)

    def test_standalone_prediction_bare_execution_and_posthoc_bind_are_not_authorizations(self):
        p=self.h.predict_strategy_outcome(self.task,self.candidate(),'E')
        self.assert_refused(p.prediction_id)
        self.assert_refused(strategy_id='one',solver='python')
        action=self.h.actions.begin_action('execute_strategy',self.task['task_id'],'E',params={'strategy_id':'one','solver':'python'})
        with self.assertRaises(ValueError):self.h.bind_strategy_outcome(p.prediction_id,action.action_id)

    def test_unselected_manual_override_identity_and_config_conflicts_are_refused(self):
        plan=self.plan([self.candidate('low',seed=3),self.candidate('high',seed=30)])
        low,high=[r['prediction_id'] for r in plan['candidates']]
        self.assert_refused(high)
        with self.assertRaises(ValueError):self.h.choose_next(plan['decision_action_id'],prediction_id=high)
        with self.assertRaises(ValueError):self.h.choose_next(plan['decision_action_id'],chosen=ActionSpec.from_dict(self.candidate('other')))
        self.assert_refused(low,task={**self.task,'task_id':'other'})
        self.assert_refused(low,task={**self.task,'description':'A different original problem.'})
        self.assert_refused(low,episode_id='other')
        self.assert_refused(low,solver='other')
        self.assert_refused(low,method={'name':'other','steps':['Different implementation plan']})
        self.assert_refused(low,config={'seed':999})
        self.assertTrue(self.h.induction_status()['completed'])

    def test_missing_U_shadow_and_rejection_cannot_create_attempts_or_jobs(self):
        self.provider.g=None;plan=self.plan()
        self.assertIsNone(plan['selected_prediction_id'])
        self.assert_refused(plan['candidates'][0]['prediction_id'])
        self.provider.g=.8;self.h.plan_mode='shadow';plan=self.plan()
        self.assertIsNone(plan['selected_prediction_id']);self.assert_refused(plan['candidates'][0]['prediction_id'])
        self.h.plan_mode='advise';plan=self.plan()
        self.h.choose_next(plan['decision_action_id'],rejected=True)
        self.assert_refused(plan['selected_prediction_id'])
        self.assertEqual(self.h.induction_status()['jobs'],[])

    def test_compatibility_choice_reuses_one_adoption_and_duplicate_consumption_is_refused(self):
        from unittest.mock import patch
        plan=self.plan();p=plan['selected_prediction_id']
        chosen=self.h.choose_next(plan['decision_action_id'],prediction_id=p)
        r=self.run_selected(plan)
        self.assertEqual(self.h.actions.get(r.action_id).params['decision_choice_action_id'],chosen['action_id'])
        with patch.object(self.h,'_run_executor') as run,self.assertRaises(ValueError):self.run_selected(plan)
        run.assert_not_called()
        self.assertEqual(len(self.h.actions.query(action_type='execute_strategy')),1)

    def test_normal_loop_positive_receipt_zero_skip_and_existing_maintenance(self):
        self.provider.decision=1;plan=self.plan();r=self.run_selected(plan)
        self.h.check_task_result(r.execution_id,{'reference_objective':1,'reference_source':'independent'})
        self.h.record(r);closed=self.h.close_episode('unified','E')
        j=closed['induction_schedule']['pending_ids'][0]
        self.assertEqual(self.h.close_episode('unified','E')['induction_schedule']['pending_ids'],[j])
        self.h.induction_material(induction_id=j)
        arguments={'induction_id':j,'relations':[{'when':'The original task requests X=1.','do':'Set and report integer X=1.',
            'expect':'Meet the declared integer task.','evidence':[r.execution_id]}],
            'review':{'outcome':'new_rule','comparison':{'sufficient':True,'substantive':True,'basis':'derivation',
                'source':'Fixture reviewer inspected original input and captured integer script.',
                'assessment':'The direct construction meets the stated one-variable task.','evidence_refs':[r.execution_id]}}}
        result=self.h.induce(**arguments);again=self.h.induce(**arguments)
        self.assertEqual(result['induction_receipt']['receipt_ref'],again['induction_receipt']['receipt_ref'])
        self.assertEqual(result['induction_receipt']['prediction_id'],plan['selected_prediction_id'])
        entry=result['relations'][0]['saved']
        self.assertTrue(self.h.induce(relations=[{'entry_id':entry,'publication':{'state':'published','reason':'Checked the stated scope.'}}])['maintenance_only'])
        self.task={**self.task,'task_id':'zero'};self.provider.decision=0
        zero=self.plan();r=self.run_selected(zero);self.h.record(r)
        self.assertEqual(self.h.close_episode('zero','E')['induction_schedule']['pending_ids'],[])

    def test_failed_selected_positive_keeps_one_job_and_empty_receipt(self):
        self.provider.decision=1
        plan=self.plan()
        self.code.write_text("raise ValueError('implementation bug')\n")
        record=self.run_selected(plan)
        self.assertFalse(record.quality['feasible'])
        self.h.record(record)
        closed=self.h.close_episode('unified','E',terminal_state='failed')
        job=closed['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        result=self.h.induce(induction_id=job,review={'outcome':'insufficient_material',
            'conclusion':'The code exception supplies no evaluable result for the proposed endpoint check.'})
        self.assertEqual(result['induction_receipt']['prediction_id'],plan['selected_prediction_id'])
        self.assertEqual(self.h.close_episode('unified','E')['induction_schedule']['pending_ids'],[])
        with self.assertRaisesRegex(ValueError,'consumed'):
            self.run_selected(plan)

    def test_invalid_independent_peaks_omit_the_dimension(self):
        for peak in (0,-1,float('inf'),float('nan')):
            with self.subTest(peak=peak):
                plan=self.plan(cost_scales={'tool_calls':peak})
                score=plan['selected_score']
                self.assertIsNone(score['cost_normalized'])
                self.assertEqual(score['scoring_basis']['terms'],['G','R'])
                self.assertIn('tool_calls',score['scoring_basis']['omitted_cost_dimensions'])
                self.assertAlmostEqual(score['utility'],.6)

    def test_two_connections_claim_only_one_adoption(self):
        import concurrent.futures
        import threading
        plan=self.plan();pid=plan['selected_prediction_id']
        other=ORHarness(home=self.home,world_model=self.provider)
        self.addCleanup(other.close)
        actions=[h.actions.begin_action('execute_strategy','unified','E',params={})
                 for h in (self.h,other)]
        barrier=threading.Barrier(2)
        def claim(pair):
            h,action=pair
            barrier.wait(timeout=5)
            try:
                h._claim_prediction_for_action(pid,action.action_id)
                return action.action_id
            except ValueError:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            winners=list(pool.map(claim,zip((self.h,other),actions)))
        self.assertEqual(sum(bool(x) for x in winners),1)
        decision=self.h.actions.get(plan['decision_action_id'])
        self.assertIn(decision.outcome['adopted_by_action_id'],winners)
        choices=[a for a in self.h.actions.query(action_type='select_strategy')
                 if (a.outcome or {}).get('adoption_source')=='execute']
        self.assertEqual(len(choices),1)


# ---------------------------------------------------------------------------
# 1. the three pre-fixes hold end to end
# ---------------------------------------------------------------------------


class TestPreFixesHold(StrategyCase):

    def test_current_prediction_carries_real_retrieval_evidence(self):
        self.solve(_task("t1"))
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[0]
        block = request["prediction_context"]
        sem = (block["retrieval_evidence"]["semantic"])
        self.assertEqual(sem["status"], "ok",
                         "the semantic channel must really run")
        exec_hits = [x for x in block["retrieval_evidence"]["hits"]
                     if x["layer"] == "execution_evidence"]
        self.assertTrue(exec_hits,
                        "the recorded execution must reach the provider")
        self.assertEqual(prediction.status, "valid")

    def test_planning_carries_real_retrieval_evidence(self):
        self.solve(_task("t1"))
        self.h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        request = self.provider.requests[0]
        block = request["prediction_context"]
        self.assertEqual(block["retrieval_evidence"]["semantic"]["status"],
                         "ok")
        exec_hits = [x for x in block["retrieval_evidence"]["hits"]
                     if x["layer"] == "execution_evidence"]
        self.assertTrue(exec_hits)

    def test_explicit_cir_context_builds_and_is_reused(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1", cir=_cir())
        self.assertEqual(ctx.joint.sources["cir"], "caller_supplied")
        # Reuse through the SAME explicit CIR: the effective inputs match.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1",
            context=ctx, cir=_cir(), benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.trace.model_info["prediction_context_id"],
                         ctx.context_id)
        # One provider call, one context.
        self.assertEqual(len(self.provider.requests), 1)

    def test_explicit_cir_context_conflict_is_refused(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1", cir=_cir())
        # Reusing WITHOUT the CIR: the effective inputs differ -> refused.
        with self.assertRaises(ValueError) as caught:
            self.h.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S01"}, "ep1", context=ctx, benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertIn("effective problem input", str(caught.exception))
        self.assertEqual(self.provider.requests, [])

    def test_historical_path_refuses_external_recall(self):
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        self.solve(_task("t1"))
        current = self.h.recall(task)
        with self.assertRaises(ValueError) as caught:
            self.h.build_prediction_context(task, "ep1", snapshot=snap,
                                            recall_result=current)
        self.assertIn("cannot be proven to belong to the frozen moment",
                      str(caught.exception))

    def test_historical_path_does_not_rederive_knowledge_targets(self):
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        # Evidence arrives AFTER the snapshot; a re-derived proposal would
        # read it.
        self.solve(_task("t1"), strategy="S01")
        ctx = self.h.build_prediction_context(
            task, "ep1", snapshot=snap,
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")])
        self.assertEqual(ctx.knowledge_targets, [],
                         "no target is re-derived from today's records")
        self.assertTrue(any("knowledge-target proposal was not saved"
                            in m for m in ctx.missing))


# ---------------------------------------------------------------------------
# 2. one comparison, one context; frozen inputs
# ---------------------------------------------------------------------------


class TestOneContextPerComparison(StrategyCase):

    def test_one_context_shared_by_all_candidates(self):
        plan = self.h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        context_ids = {r["prediction_context"]["context_id"]
                       for r in self.provider.requests}
        self.assertEqual(len(context_ids), 1)
        self.assertEqual(plan["prediction_context_id"],
                         next(iter(context_ids)))

    def test_candidates_keep_their_own_config_and_identity(self):
        specs = [ActionSpec("execute_strategy", "t1", strategy_id="S01",
                            params={"time_limit": 60, "seed": 42}),
                 ActionSpec("execute_strategy", "t1", strategy_id="S01",
                            params={"time_limit": 120})]
        for spec in specs:
            self.h.predict_strategy_outcome(_task("t1"), spec, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        sent = [r["candidate"] for r in self.provider.requests]
        self.assertEqual(sent[0]["config"]["time_limit"], 60)
        self.assertEqual(sent[0]["config"]["seed"], 42)
        self.assertEqual(sent[1]["config"]["time_limit"], 120)
        # Same strategy_id, different config: different candidates.
        self.assertNotEqual(sent[0]["config"], sent[1]["config"])

    def test_later_bank_writes_do_not_change_the_saved_request(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1")
        frozen = json.dumps(ctx.provider_view(), sort_keys=True)
        self.solve(_task("t1"))          # the bank moves on
        again = json.dumps(ctx.provider_view(), sort_keys=True)
        self.assertEqual(frozen, again)


# ---------------------------------------------------------------------------
# 3. the protocol really reaches the provider; honest failure states
# ---------------------------------------------------------------------------


class TestProtocolReachesProvider(StrategyCase):

    def test_request_shape_and_fixed_identity(self):
        self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01",
             "solver": "highs", "config": {"time_limit": 60}}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[0]
        self.assertEqual(request["prediction_protocol"], "wm-so/1")
        self.assertEqual(request["request_kind"],
                         "strategy_outcome_prediction")
        self.assertEqual(request["candidate"]["strategy_id"], "S01")
        self.assertEqual(request["candidate"]["config"]["time_limit"], 60)
        self.assertIn("prediction_context", request)
        self.assertIn("output_contract", request)
        self.assertIn("candidate", request["output_contract"]
                      ["fixed_by_framework"])

    def test_provider_receives_content_not_ids(self):
        task = dict(_task("t1"), coupling=_cir())
        self.solve(_task("t1"))
        self.h.predict_strategy_outcome(
            task,
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        block = self.provider.requests[0]["prediction_context"]
        joint = block["joint_problem"]
        self.assertIn("distribution centre", joint["text"])
        # CIR is internal: the model reads the task text and math attributes,
        # never the entity/relation lists.
        self.assertNotIn("cir", joint)
        hits = block["retrieval_evidence"]["hits"]
        self.assertTrue(any(h["content"].get("task_text_excerpt")
                            for h in hits))
        self.assertIn("sources", block["harness_capability"])

    def test_payload_parses_into_the_new_contract(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.prediction_type, "strategy_outcome")
        self.assertEqual(prediction.benefit.kind, "solution_quality")
        self.assertEqual(prediction.benefit.value, 0.8)
        self.assertIsNotNone(prediction.benefit.baseline)
        self.assertEqual(sorted(prediction.cost.expected.measured_dims()),
                         ["llm_tokens", "solver_runtime_s"])
        self.assertEqual(len(prediction.risk.events), 2)
        self.assertEqual(prediction.uncertainty.source, "model_self_report")
        self.assertEqual(validate_strategy_outcome(prediction), [])
        # Round-trips through storage.
        stored = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertEqual(stored.to_dict(), prediction.to_dict())

    def test_call_cost_is_recorded_on_the_prediction(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertIsNotNone(prediction.trace.call_cost)
        self.assertIn("llm_tokens", prediction.trace.call_cost.measured_dims())

    def test_not_configured_is_honest(self):
        h = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "contract_only")
        self.assertFalse(prediction.provider_configured)
        self.assertFalse(prediction.service_available)
        self.assertFalse(prediction.prediction_made)

    def test_empty_payload_is_a_recorded_non_prediction(self):
        provider = StubProvider(payload={})
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertNotEqual(prediction.status, "valid")
        self.assertFalse(prediction.prediction_made)
        self.assertTrue(any("no predicted content" in n
                            for n in prediction.notes))

    def test_invalid_json_is_invalid_not_repaired(self):
        provider = StubProvider(payload="not-a-object")
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "invalid")
        # A non-object payload is classified by its TOP-LEVEL TYPE, so the
        # diagnosis says what the endpoint actually returned.
        self.assertEqual(
            prediction.trace.model_info["failure"]["kind"],
            "wrong_top_level")
        self.assertIn("not a prediction object",
                      prediction.trace.model_info["error"])

    def test_array_payload_is_wrong_top_level_not_truncation(self):
        """The field case: the endpoint answers ``[]``.

        ``[]`` is VALID JSON, so a plain "not an object" message would
        hide whether the answer was the wrong shape or was cut off. The
        diagnosis names the top-level type, which is what decides whether
        to fix the request or raise the output budget.
        """
        provider = StubProvider(payload=[])
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "invalid")
        failure = prediction.trace.model_info["failure"]
        self.assertEqual(failure["kind"], "wrong_top_level")
        self.assertEqual(failure["top_level"], "list")

    def test_truncation_empty_and_array_are_distinguished(self):
        import json as _json

        class Raw(WorldModelProvider):
            name = "raw"

            def __init__(self, envelope):
                self.envelope = envelope

            def predict(self, request, timeout_s=None):
                body = _json.dumps(self.envelope).encode()
                import urllib.request
                original = urllib.request.urlopen

                class Resp:
                    def read(self_inner):
                        return body

                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *a):
                        return False

                urllib.request.urlopen = lambda *a, **k: Resp()
                try:
                    from or_harness.world_model.provider import (
                        HttpChatProvider,
                    )
                    return HttpChatProvider(
                        "https://x/v1", "M", "k",
                        max_output_tokens=2048).predict(request)
                finally:
                    urllib.request.urlopen = original

        cases = {
            "[]": ("wrong_top_level", "length"),
            "```json\n[]\n```": ("wrong_top_level", "length"),
            '{"benefit": {"kind":': ("truncated", "length"),
            "": ("empty_response", "stop"),
        }
        kinds = {}
        for content, (expected, finish) in cases.items():
            envelope = {"choices": [{"message": {"content": content},
                                    "finish_reason": finish}],
                        "usage": {"prompt_tokens": 1,
                                  "completion_tokens": 2}}
            h = ORHarness(home=self.home, world_model=Raw(envelope),
                          embedding=self.backend)
            self.addCleanup(h.close)
            prediction = h.predict_strategy_outcome(
                _task("t1"),
                {"action_type": "execute_strategy", "strategy_id": "S01"},
                "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
            self.assertEqual(prediction.status, "invalid", content)
            kinds[content] = prediction.trace.model_info["failure"]["kind"]
        self.assertEqual(kinds["[]"], "wrong_top_level")
        self.assertEqual(kinds["```json\n[]\n```"], "wrong_top_level")
        self.assertEqual(kinds['{"benefit": {"kind":'], "truncated")
        self.assertEqual(kinds[""], "empty_response")

    def test_stop_by_length_wins_over_an_empty_body(self):
        """A cut-off answer is TRUNCATION even when nothing was produced.

        ``finish_reason == "length"`` is the authoritative evidence of a
        cut-off, so an empty body under it is reported as truncation (with
        the content shape kept as a separate fact), not as "the model
        chose to answer nothing".
        """
        import json as _json

        class Raw(WorldModelProvider):
            name = "raw"

            def predict(self, request, timeout_s=None):
                body = _json.dumps({
                    "choices": [{"message": {"content": ""},
                                 "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10875,
                              "completion_tokens": 6388}}).encode()
                import urllib.request
                original = urllib.request.urlopen

                class Resp:
                    def read(self_inner):
                        return body

                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *a):
                        return False

                urllib.request.urlopen = lambda *a, **k: Resp()
                try:
                    from or_harness.world_model.provider import (
                        HttpChatProvider,
                    )
                    return HttpChatProvider("https://x/v1", "M", "k",
                                            max_output_tokens=2048).predict(
                                                request)
                finally:
                    urllib.request.urlopen = original

        h = ORHarness(home=self.home, world_model=Raw(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        failure = prediction.trace.model_info["failure"]
        self.assertEqual(failure["kind"], "truncated")
        self.assertEqual(failure["content_kind"], "empty")
        self.assertEqual(failure["finish_reason"], "length")
        # The effective parameters that were actually sent are recorded.
        self.assertEqual(
            prediction.trace.model_info["effective_parameters"]["max_tokens"],
            2048)

    def test_missing_finish_reason_is_unknown_not_normal(self):
        """An absent ``finish_reason`` is not evidence of a clean ending."""
        class Raw(WorldModelProvider):
            name = "raw"

            def predict(self, request, timeout_s=None):
                return {"payload": None, "usage": None,
                        "error": "model content is not valid JSON: x",
                        "latency_s": 0.1,
                        "diagnostics": {"finish_reason": None,
                                        "content_kind": "text"}}

        h = ORHarness(home=self.home, world_model=Raw(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        failure = prediction.trace.model_info["failure"]
        self.assertIsNone(failure["finish_reason"])
        self.assertIn("UNKNOWN", failure["note"])

    def test_failure_record_reads_back_and_closes_an_episode(self):
        """A recorded provider failure must not block the read paths.

        This is the defect the field run hit: the failure was WRITTEN
        under a status the contract could not READ, so ``get``, ``query``,
        ``read_prediction`` and ``close-episode`` all raised.
        """
        class Boom(WorldModelProvider):
            name = "boom"

            def predict(self, request, timeout_s=None):
                raise TimeoutError("request timed out after 30s")

        h = ORHarness(home=self.home, world_model=Boom(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "invalid")
        self.assertEqual(
            prediction.trace.model_info["failure"]["kind"], "timeout")
        stored = h.strategy_predictions.get(prediction.prediction_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.status, "invalid")
        self.assertEqual(
            h.strategy_predictions.query(episode_id="ep1")[0].status,
            "invalid")
        record = h.read_prediction(prediction.prediction_id)
        self.assertEqual(record["prediction"]["status"], "invalid")
        h.snapshot(_task("t1"), episode_id="ep1")
        out = h.close_episode("t1", episode_id="ep1",
                              terminal_state="failed")
        self.assertIsNotNone(out)

    def test_legacy_provider_error_record_still_reads(self):
        """A record written BEFORE the vocabularies were unified."""
        from or_harness.world_model.contracts import (
            StrategyOutcomePrediction,
            normalize_stored_status,
        )
        self.assertEqual(normalize_stored_status("provider_error"),
                         ("invalid", "provider_error"))
        self.assertEqual(normalize_stored_status("bogus"),
                         ("bogus", None))
        payload = {
            "contract_version": "wm-contract/1",
            "prediction_type": "strategy_outcome",
            "prediction_id": "sp_legacy",
            "status": "provider_error",
            "candidate": {"action_type": "execute_strategy",
                          "strategy_id": "S01", "task_id": "t1"},
            "trace": {"prediction_kind": "strategy_outcome",
                      "model_info": {}},
            "error": "ProviderError: timed out after 129932 ms",
        }
        restored = StrategyOutcomePrediction.from_dict(payload)
        self.assertEqual(restored.status, "invalid")
        self.assertEqual(restored.trace.model_info["legacy_status"],
                         "provider_error")
        self.assertIn("timed out", restored.trace.model_info["error"])
        again = StrategyOutcomePrediction.from_dict(restored.to_dict())
        self.assertEqual(again.status, "invalid")

    def test_nan_and_out_of_range_values_are_rejected(self):
        payload = {
            "benefit": {"kind": "solution_quality", "metric": "q",
                        "value": 1.5,   # out of [0,1]
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": float("nan")},
            "risk": {"events": [{"event": "x", "probability": 1.5}]},
        }
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "invalid")
        joined = " ".join(prediction.notes)
        self.assertIn("NORMALIZED", joined)
        self.assertIn("cost.llm_tokens", joined)
        self.assertIn("probability", joined)

    def test_provider_error_keeps_its_cost(self):
        class ExplodingProvider(WorldModelProvider):
            name = "exploding"

            def predict(self, request, timeout_s=None):
                return {"payload": None, "usage": {
                    "prompt_tokens": 10, "completion_tokens": 5},
                    "error": "boom", "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=ExplodingProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "invalid")
        self.assertIsNotNone(prediction.trace.call_cost,
                             "a failed call's spend is real and kept")


# ---------------------------------------------------------------------------
# 4. scope, cost and uncertainty honesty
# ---------------------------------------------------------------------------


class TestScopeAndHonesty(StrategyCase):

    def test_unexecuted_window_predicts_validly_but_not_comparably(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01",
             "scope": "strategy_window",
             "window_id": "win::t1::ep1::S01"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertFalse(prediction.trace.comparable)
        self.assertTrue(prediction.trace.not_comparable_reasons)

    def test_missing_cost_omits_the_term_for_the_whole_comparison(self):
        from or_harness.world_model.planner import (
            PlanLimits,
            score_strategy_outcome_predictions,
        )
        # Two valid predictions: one predicts cost, one does not.
        payload_full = dict(GOOD_PAYLOAD)
        payload_full["cost"] = {"llm_tokens": 1000}
        payload_none = dict(GOOD_PAYLOAD)
        payload_none.pop("cost")
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        p_full = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S01"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        provider.payload = payload_none
        p_none = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S02"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        limits = PlanLimits(alpha=1.0, beta=1.0, gamma=1.0,
                            cost_weights={"llm_tokens": 1.0}, benefit_kind='solution_quality', benefit_metric='normalized_objective_gap', benefit_unit='1-gap')
        scores = score_strategy_outcome_predictions([p_full, p_none], limits)
        self.assertTrue(all("cost" in s.incomparable for s in scores))
        self.assertTrue(all(s.cost_normalized is None for s in scores))
        self.assertEqual(scores[1].utility, scores[0].utility)

    def test_business_objective_is_not_execution_cost(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        # The predicted cost dimensions are the RESOURCE dimensions only.
        if prediction.cost is not None:
            from or_harness.core.schema import COST_DIMENSIONS
            self.assertTrue(set(prediction.cost.expected.measured_dims())
                            <= set(COST_DIMENSIONS))

    def test_an_off_yardstick_benefit_does_not_enter_the_ranking(self):
        """Requirement 6: a benefit in a currency the comparison cannot read
        is reported, NOT scored as zero. The candidate keeps its cost/risk
        picture but produces no utility, so it can never silently win or
        lose the full ranking on an upside nobody could read.

        r12: the PRIMARY (default) yardstick is now task COMPLETION, so it
        is the SOLUTION-QUALITY claim that is off-yardstick here."""
        from or_harness.world_model.planner import (
            PlanLimits,
            score_strategy_outcome_predictions,
        )
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        # Candidate 1: a completion claim — the comparison's own yardstick.
        provider.payload = {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.9,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": 100},
        }
        p_completion = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S01"}, "ep1", benefit_convention={'kind': 'effective_completion', 'metric': 'task_result_check_passed', 'unit': 'boolean'})
        # Candidate 2: a solver-quality claim (a DIFFERENT currency from the
        # default completion yardstick).
        provider.payload = json.loads(json.dumps(GOOD_PAYLOAD))
        p_quality = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S02"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        limits = PlanLimits(alpha=1.0, beta=1.0, gamma=1.0,
                            cost_weights={"llm_tokens": 1.0}, benefit_kind='effective_completion', benefit_metric='task_result_check_passed', benefit_unit='boolean')
        scores = score_strategy_outcome_predictions(
            [p_completion, p_quality], limits)
        quality_score = scores[1]
        # Reported as incomparable on the benefit...
        self.assertIn("benefit", quality_score.incomparable)
        # ...and NO full-ranking utility was produced.
        self.assertIsNone(quality_score.utility,
                          "an off-yardstick benefit must not be ranked as "
                          "a zero benefit")
        # The on-yardstick (completion) candidate still gets a utility.
        self.assertIsNotNone(scores[0].utility)

    def test_a_quality_kind_with_a_different_metric_is_off_yardstick(self):
        """The KIND alone is not enough: two solution_quality predictions
        measuring DIFFERENT metrics are not one currency."""
        from or_harness.world_model.planner import (
            PlanLimits,
            score_strategy_outcome_predictions,
        )
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        provider.payload = {
            "benefit": {"kind": "solution_quality",
                        "metric": "business_cost_saving_ratio",
                        "unit": "ratio", "value": 0.8,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.5}},
            "cost": {"llm_tokens": 100},
        }
        prediction = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S01"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        scores = score_strategy_outcome_predictions(
            [prediction], PlanLimits(alpha=1.0, beta=1.0, gamma=1.0, benefit_kind='solution_quality', benefit_metric='normalized_objective_gap', benefit_unit='1-gap'))
        self.assertIn("benefit", scores[0].incomparable)
        self.assertIsNone(scores[0].utility)

    def test_model_self_report_is_never_relabelled(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.uncertainty.source, "model_self_report")
        self.assertTrue(any("UNCALIBRATED" in n
                            for n in prediction.uncertainty.notes))
        self.assertNotEqual(prediction.uncertainty.source, "measured")


# ---------------------------------------------------------------------------
# 5. predictions drive the suggestion; choices drive the selection
# ---------------------------------------------------------------------------


class TestPredictionDrivesDecision(StrategyCase):

    def _plan(self, payloads):
        """Plan with per-candidate payloads (S01 first, S02 second)."""
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        original = provider.predict

        def per_candidate(request, timeout_s=None):
            sid = request["candidate"]["strategy_id"]
            provider.payload = payloads[sid]
            return original(request, timeout_s=timeout_s)
        provider.predict = per_candidate
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        return h, provider, plan

    def test_prediction_changes_the_suggestion(self):
        # S02 predicts a much better benefit -> the suggestion must flip
        # from the catalog/Selector order to S02.
        better = json.loads(json.dumps(GOOD_PAYLOAD))
        better["benefit"]["value"] = 0.95
        worse = json.loads(json.dumps(GOOD_PAYLOAD))
        worse["benefit"]["value"] = 0.2
        _h, _provider, plan = self._plan({"S01": worse, "S02": better})
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["suggested"]["strategy_id"], "S02")
        self.assertIn("G=0.95", plan["suggestion_basis"])
        self.assertIn("R=omitted", plan["suggestion_basis"])


    def test_all_predictions_failed_reports_fallback(self):
        provider = StubProvider(payload=None)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        self.assertEqual(plan["status"], "no_valid_predictions")
        self.assertIn("call plan-next again", plan["truncation_reason"])
        self.assertIsNone(plan["suggested"])

    def test_horizon_two_is_refused_for_the_new_protocol(self):
        with self.assertRaises(ValueError) as caught:
            self.h.plan_next(
                _task("t1"), "ep1",
                candidates=[ActionSpec("execute_strategy", "t1",
                                       strategy_id="S01")],
                limits={"horizon": 2, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        self.assertIn("horizon=1 only", str(caught.exception))

    def test_budget_limit_stops_further_calls(self):
        h = ORHarness(home=self.home, world_model=StubProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        # A budget that real consumption has ALREADY exceeded stops the
        # decision before any model call.
        h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        report = h.report_action("model", _task("t1"), "ep1",
                                 outcome={"kind": "seed"},
                                 cost={"llm_tokens": 100})
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        self.assertEqual(plan["status"], "fallback")
        self.assertEqual(plan["model_calls_made"], 0)

    def test_max_calls_limit_binds(self):
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S04")],
            limits={"horizon": 1, "max_model_calls": 2, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'},
            )
        self.assertEqual(plan["model_calls_made"], 2)
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("model-call budget exhausted",
                      plan["truncation_reason"])


# ---------------------------------------------------------------------------
# 6. execution binding and cost accounting
# ---------------------------------------------------------------------------


class TestExecutionBinding(StrategyCase):

    def test_real_execution_binds_and_is_comparable(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        bound = bind_legacy(self.h,
            prediction.prediction_id, record.action_id)
        self.assertIsNone(bound.trace.model_info.get("binding_mismatch"))
        self.assertTrue(bound.trace.comparable)
        # Idempotent re-bind, no extra model call.
        calls = len(self.provider.requests)
        again = bind_legacy(self.h,
            prediction.prediction_id, record.action_id)
        self.assertEqual(len(self.provider.requests), calls)
        self.assertEqual(again.trace.comparable, bound.trace.comparable)

    def test_different_config_does_not_cross_bind(self):
        task = _task("t1")
        # Predict a candidate with a DIFFERENT solver than the one that ran.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "solver": "cbc"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")   # ran with highs
        bound = bind_legacy(self.h,
            prediction.prediction_id, record.action_id)
        mismatch = bound.trace.model_info.get("binding_mismatch") or {}
        self.assertIn("solver", mismatch)
        self.assertFalse(bound.trace.comparable)

    def test_unexecuted_candidate_generates_no_fact(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")   # a DIFFERENT strategy ran
        executions_before = self.h.bank.count()
        bound = bind_legacy(self.h,
            prediction.prediction_id, record.action_id)
        mismatch = bound.trace.model_info.get("binding_mismatch") or {}
        self.assertIn("strategy_id", mismatch)
        self.assertFalse(bound.trace.comparable)
        self.assertEqual(self.h.bank.count(), executions_before,
                         "no fact is fabricated for the unexecuted candidate")

    def test_planning_cost_counts_failed_calls_once(self):
        provider = StubProvider(payload=None)   # every call fails
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        self.assertEqual(plan["model_calls_made"], 2)
        cost = plan["planning_cost"]
        self.assertTrue(cost, "failed calls' spend is real and recorded")
        self.assertEqual(cost["cost"]["llm_tokens"], 300.0,
                         "two failed calls of 150 tokens each, counted once")
        # And charged ONCE to the decision action.
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 300.0)


class TestCliStrategyCommands(StrategyCase):
    """CLI surface with a real local HTTP stub server (line protocol)."""

    def _run(self, argv):
        from or_harness.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home] + argv)
        return code, json.loads(buf.getvalue())

    def _with_http_stub(self, argv):
        """Run a CLI command against a local OpenAI-compatible stub."""
        import threading
        import urllib.request
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen: List[Dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                seen.append(body)
                payload = {
                    "choices": [{"message": {"content": json.dumps(
                        GOOD_PAYLOAD)}}],
                    "usage": {"prompt_tokens": 100,
                              "completion_tokens": 50},
                }
                request=json.loads(body["messages"][1]["content"])
                content=json.loads(payload["choices"][0]["message"]["content"])
                content["benefit"].update({key:request["benefit_convention"][key]
                                           for key in ("kind","metric","unit")})
                payload["choices"][0]["message"]["content"]=json.dumps(content)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silence
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(thread.join, timeout=2)
        url = f"http://127.0.0.1:{server.server_port}"
        code, payload = self._run(["--world-model", f"{url}::stub-model"]
                                  + argv)
        return code, payload, seen

    def test_single_candidate_plan_calls_provider_once_and_can_be_inspected(self):
        code,payload,seen=self._with_http_stub([
            "plan-next","--task",json.dumps(_task("t1")),"--episode","ep1",
            "--candidates",json.dumps([{"strategy_id":"S01","solver":"python"}])])
        self.assertEqual(code,0)
        pid=payload["result"]["selected_prediction_id"]
        self.assertIsNotNone(pid)
        self.assertEqual(payload["result"]["selected_score"]["prediction_id"],pid)
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0]["model"],"stub-model")
        self.assertIn("wm-so/1",seen[0]["messages"][0]["content"])
        request=json.loads(seen[0]["messages"][1]["content"])
        self.assertEqual(request["prediction_protocol"],"wm-so/1")
        code,inspected=self._run(["inspect","--bank","predictions","--prediction",pid])
        self.assertEqual(code,0)
        self.assertEqual(inspected["result"]["prediction"]["kind"],"strategy_outcome")
        self.assertEqual(inspected["result"]["prediction"]["prediction"]["prediction_id"],pid)
        self.assertEqual(len(seen),1)

    def test_plan_candidates_share_the_same_frozen_context_over_http(self):
        code,payload,seen=self._with_http_stub([
            "plan-next","--task",json.dumps(_task("t1")),"--episode","ep1",
            "--candidates",json.dumps([{"strategy_id":sid,"solver":"python"}
                                       for sid in ("S01","S02")])])
        self.assertEqual(code,0)
        self.assertEqual(len(seen),2)
        requests=[json.loads(envelope["messages"][1]["content"]) for envelope in seen]
        plan=payload["result"]["plan"]
        self.assertEqual({request["prediction_context"]["context_id"] for request in requests},
                         {plan["prediction_context_id"]})
        self.assertEqual(requests[0]["prediction_context"],requests[1]["prediction_context"])
        self.assertEqual(len(plan["candidates"]),2)
        self.assertIsNotNone(payload["result"]["selected_prediction_id"])

    def test_bind_strategy_command(self):
        task = _task("t1")
        prediction = planned_prediction(self.h,
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "solver":"highs"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        work=Path(self.home)/"cli_bind";work.mkdir()
        script=work/"solve.py";script.write_text("import json\njson.dump({'status':'optimal','objective_value':100,'objective_bound':100},open('result.json','w'))\n")
        record=self.h.execute(task,None,str(script),str(work),prediction_id=prediction.prediction_id)
        code, payload = self._run([
            "bind-strategy", "--prediction", prediction.prediction_id,
            "--action", record.action_id])
        self.assertEqual(code, 0)
        # The execution ran under episode "ep1" (self.solve passes it), so
        # the identity is KNOWN and the binding is comparable.
        self.assertIn("COMPARABLE", payload["summary"])

    def test_plan_next_is_strategy_outcome_only(self):
        """Planning speaks ONE protocol. The old `--protocol` selector is
        gone, and a refusal (the removed flag, horizon=2) is a usage error
        rather than a silently different comparison."""
        code, payload, seen = self._with_http_stub([
            "plan-next", "--task", json.dumps(_task("t1")),
            "--episode", "ep1",
            "--candidates", json.dumps([
                {"action_type": "execute_strategy", "task_id": "t1",
                 "strategy_id": "S01"}])])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["protocol"], "strategy-outcome")
        self.assertTrue(seen, "the model was really called over HTTP")
        # The compared candidate's prediction id is handed over, so the
        # caller binds it instead of predicting again.
        candidate = payload["result"]["plan"]["candidates"][0]
        self.assertTrue(candidate["prediction_id"].startswith("sp_"))
        # The removed selector is refused (a usage error).
        with self.assertRaises(SystemExit):
            self._run(["plan-next", "--task", json.dumps(_task("t1")),
                       "--candidates", json.dumps([
                           {"action_type": "execute_strategy",
                            "strategy_id": "S01"}]),
                       "--protocol", "legacy"])
        # A two-step request is refused with the replacement path named —
        # never silently downgraded to a different comparison.
        code, rejected = self._run([
            "plan-next", "--task", json.dumps(_task("t1")),
            "--candidates", json.dumps([
                {"action_type": "execute_strategy",
                 "strategy_id": "S01"}]),
            "--horizon", "2"])
        self.assertEqual(code, 2)
        self.assertIn("horizon=1", rejected["summary"])
        self.assertIn("REAL observation", rejected["summary"])


from tests.harness.g_fixtures import GCase, G_NOTE, G_PAYLOAD
from or_harness.world_model.contracts import (
    BenefitEstimate, FAITHFUL_G_METRIC, FAITHFUL_G_OBSERVATION_VERSION,
)


class TestFaithfulGContract(GCase):
    def test_default_request_validation_persistence_and_utility_agree(self):
        from or_harness.world_model.planner import (
            normalize_benefit_convention, score_strategy_outcome_predictions, PlanLimits,
        )
        p = self.predict_g()
        required = self.provider.requests[-1]["benefit_convention"]
        self.assertEqual(required["metric"], FAITHFUL_G_METRIC)
        self.assertTrue(required["required"])
        self.assertEqual(required["scope"], "attempt")
        self.assertEqual(required["observation_rule_version"], FAITHFUL_G_OBSERVATION_VERSION)
        saved = self.h.get_strategy_outcome_prediction(p.prediction_id)
        self.assertEqual(saved.benefit.to_dict(), p.benefit.to_dict())
        self.assertEqual(normalize_benefit_convention(p.benefit.kind, p.benefit.metric),
                         ("effective_completion", FAITHFUL_G_METRIC))
        self.assertFalse(validate_strategy_outcome(p))
        score = score_strategy_outcome_predictions([p], PlanLimits(alpha=1, beta=0, gamma=0))[0]
        self.assertEqual(score.benefit_value, 0.7)
        self.assertEqual(score.utility, 0.7)

    def test_mismatched_metric_and_invalid_g_drop_only_benefit(self):
        import copy
        from or_harness.world_model.planner import score_strategy_outcome_predictions, PlanLimits
        variants = [
            {"kind": "solution_quality", "metric": "normalized_objective_gap", "unit": "1-gap", "value": 0.9},
            {"value": 1.2}, {"value": float("inf")}, {"value": float("nan")},
            {"value": True}, {"value": "0.7"}, {"unit": "boolean"}, {"scope": "strategy_window"},
            {"observation_rule_version": "unknown/9"},
        ]
        for index, patch in enumerate(variants):
            with self.subTest(patch=patch):
                self.provider.payload = copy.deepcopy(G_PAYLOAD)
                self.provider.payload["benefit"].update(patch)
                p = self.predict_g(episode=f"bad{index}")
                self.assertIsNone(p.benefit.value)
                self.assertEqual(p.benefit.notes, [G_NOTE])
                self.assertIn("benefit", p.trace.unsupported_fields)
                self.assertIsNotNone(p.cost)
                self.assertIsNotNone(p.risk)
                self.assertEqual(p.status, "valid")
                score = score_strategy_outcome_predictions([p], PlanLimits())[0]
                self.assertIsNone(score.benefit_value)
                self.assertIsNone(score.utility)

    def test_missing_numeric_g_retains_the_one_sentence(self):
        self.provider.payload["benefit"].pop("value")
        self.provider.payload["benefit"].pop("baseline")
        p = self.predict_g()
        self.assertEqual(p.status, "valid")
        self.assertIsNone(p.benefit.value)
        self.assertEqual(p.benefit.notes, [G_NOTE])

    def test_explicit_legacy_solver_gap_keeps_its_meaning(self):
        self.provider.payload = GOOD_PAYLOAD
        p = self.predict_g(benefit_convention={"kind": "solution_quality",
            "metric": "normalized_objective_gap", "unit": "1-gap"})
        self.assertEqual(p.benefit.value, 0.8)
        self.assertIsNone(p.benefit.observation_rule_version)
        r = self.run_g(p)
        self.check_g(r, evidence=False)
        b = self.close_g()["evaluations"][0]["benefit"]
        self.assertEqual(b["observed"], 1.0)
        old = BenefitEstimate.from_dict({"kind": "effective_completion",
            "metric": "task_result_check_passed", "unit": "boolean", "value": 0.6})
        self.assertEqual(old.metric, "task_result_check_passed")
        self.assertIsNone(old.observation_rule_version)


if __name__ == "__main__":
    unittest.main()
