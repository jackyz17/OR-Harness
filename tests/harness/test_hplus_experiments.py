"""Real integer attempts in separate training/control and frozen test homes."""
from pathlib import Path
import json
from contextlib import closing

from helpers import HarnessTestCase
from decision_fixtures import ExecutionFixtureProvider
from or_harness.api import ORHarness
from or_harness.experiments.hplus import integer_fixture, run_comparison
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.verification import verify_task_result


class TestHplusBuildFreeze(HarnessTestCase):
    def test_equal_initial_libraries_zero_review_random_budget_and_frozen_test(self):
        report=integer_fixture(str(Path(self.home)/'experiment'),seed=7)
        arms=report['arms']
        self.assertEqual(len({a['initial_library_digest'] for a in arms.values()}),1)
        self.assertEqual(arms['hplus']['triggers'],arms['random']['triggers'])
        self.assertEqual(arms['hplus']['triggers'],1)
        self.assertEqual(arms['hplus']['receipts'],1)
        self.assertEqual(arms['nohplus']['zero_signal_reviews'],4)
        self.assertEqual(arms['nohplus']['confirmed_zero_missed_opportunities'],1)
        self.assertEqual(arms['nohplus']['independently_validated_increments'],1)
        for arm in arms.values():
            self.assertEqual(arm['triggers'],arm['receipts'])
            self.assertGreater(arm['induction_wall_s'],0)
            self.assertIsNone(arm['induction_tokens'])
            self.assertIsNone(arm['induction_tool_calls'])
            self.assertEqual(arm['frozen_tests'][0]['objective'],283)
            self.assertTrue(arm['frozen_tests'][0]['library_unchanged'])
            self.assertEqual(arm['frozen_passes'],1)
            self.assertEqual(arm['frozen_tests'][0]['reference_acceptance']['state'],'passed')
            self.assertEqual(arm['frozen_tests'][0]['reference_acceptance']['reference_source']['source'],'bench_declared')
        for receipt in arms['nohplus']['receipt_details']:
            self.assertNotEqual(receipt['solve_action_id'],receipt['induction_action_id'])
            self.assertTrue(receipt['baseline_snapshot_ref'])


class TestFrozenSubmissionScoring(HarnessTestCase):
    def run_fixture(self, *, evaluator=True):
        initial=Path(self.home)/'initial'
        with closing(ORHarness(home=str(initial),embedding=LocalHashEmbeddingBackend())):
            pass
        task={'task_id':'frozen-question','family':'assignment',
              'text':'Choose an integer x from 0 through 5 to maximize x.'}
        calls=[];homes={};private_reference=5

        def propose(h,t):
            self.assertEqual(t,task)
            self.assertNotIn('reference',json.dumps(t))
            return {'name':'bounded integer candidate','steps':['Return a bounded integer and verify its reported objective.'],'solver':'python'}

        def attempt(h,t,prediction):
            self.assertEqual(t,task)
            work=Path(h.home)/'ws';work.mkdir()
            script=work/'solve.py'
            script.write_text("import json\njson.dump({'status':'optimal','objective_value':4,"
                              "'objective_bound':4,'variables':{'x':4}},open('result.json','w'))\n")
            record=h.execute(t,prediction_id=prediction.prediction_id,
                             code_path=str(script),workspace=str(work))
            checked=h.check_task_result(record.execution_id,{
                'integer':{'variables':['x']},'recompute_objective':{'coefficients':{'x':1}},
                'semantic_probe':{'path':'execution_features.solution_variables.x','min':0,'max':5}})
            self.assertEqual(checked['state'],'passed')
            self.assertEqual(checked['report']['phase'],'pre_submission')
            self.assertNotIn('reference_objective',checked['report']['scope']['basis'])
            calls.append(('attempt',record.execution_id))
            homes[record.execution_id]=str(h.home)
            return record

        def evaluate(t,submission):
            record=submission['execution'];execution_id=record['execution_id']
            self.assertEqual(t,task)
            with closing(ORHarness(home=homes[execution_id])) as reader:
                closed=reader.episode_closeout_record(task['task_id'],task['task_id'])
                self.assertEqual(closed['closeout_id'],submission['closeout']['closeout_id'])
            calls.append(('evaluate',execution_id))
            verdict=verify_task_result(record,{'reference_objective':private_reference,
                'reference_source':'bench_declared','reference_version':'evaluator-only/1'})
            # Even evaluator-side mutation must not change the locked answer.
            record['quality']['objective']=99999
            return verdict

        report=run_comparison(initial_home=str(initial),training_tasks=[],test_tasks=[task],
            output=str(Path(self.home)/'experiment'),provider_factory=ExecutionFixtureProvider,
            propose=propose,attempt=attempt,review=lambda *args:self.fail('No training review expected'),
            embedding_factory=LocalHashEmbeddingBackend,
            evaluate_submission=evaluate if evaluator else None)
        return report,calls,homes

    def test_failed_reference_scores_the_locked_answer_without_retry_or_leak(self):
        report,calls,homes=self.run_fixture()
        self.assertEqual([kind for kind,_ in calls],['attempt','evaluate']*3)
        for arm in report['arms'].values():
            self.assertEqual(len(arm['frozen_tests']),1)
            test=arm['frozen_tests'][0]
            self.assertEqual(test['task_check'],'passed')
            self.assertEqual(test['reference_acceptance']['state'],'failed')
            self.assertEqual(test['objective'],4)
            self.assertEqual(arm['frozen_passes'],0)
            self.assertTrue(test['submission']['closeout']['closeout_id'])
            self.assertTrue(test['submission']['answer_fingerprint'])
            with closing(ORHarness(home=homes[test['execution_id']])) as reader:
                stored=reader.bank.get(test['execution_id'])
                self.assertEqual(stored.quality['objective'],4)
                self.assertEqual(stored.execution_features['task_check']['phase'],'pre_submission')
                self.assertNotIn('reference_objective',stored.execution_features['task_check']['scope']['basis'])
                self.assertNotIn('evaluator-only/1',json.dumps(stored.to_dict()))
                with self.assertRaisesRegex(ValueError,'already closed'):
                    reader.plan_next({'task_id':'frozen-question','family':'assignment'},'frozen-question',
                                     candidates=[{'name':'retry','steps':['Retry after reference'],'solver':'python'}])

    def test_self_check_pass_without_an_evaluator_is_not_a_formal_pass(self):
        report,calls,_=self.run_fixture(evaluator=False)
        self.assertEqual([kind for kind,_ in calls],['attempt']*3)
        for arm in report['arms'].values():
            test=arm['frozen_tests'][0]
            self.assertEqual(test['task_check'],'passed')
            self.assertEqual(test['reference_acceptance']['state'],'insufficient')
            self.assertEqual(arm['frozen_passes'],0)
