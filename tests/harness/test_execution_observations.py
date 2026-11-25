"""Controlled failures verify observation plumbing, not forecasting ability."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from test_file_execution import FileCase
from or_harness.execution.executor import SafePythonExecutor
from or_harness.world_model.r_signal import technical_attempt_fact


class TestFailureStages(FileCase):
    def fact(self, record):
        # record() applies the framework's failure classification.
        self.h.record(record)
        saved = self.h.bank.get(record.execution_id)
        return technical_attempt_fact(saved)

    def test_submission_rejection_is_bound_without_fabricated_runtime(self):
        self.resource('data.txt','1')
        self.provider.payload['capability_gain']={'decision':1,
            'claim':'A proposed input-handling repair might support future file-driven executions.',
            'basis':['synthetic positive-signal fixture, not a claim of actual learning'],
            'applies_to':['file-driven attempts'],
            'verification_conditions':[{'condition':'Inspect the actual submitted code and file-access result.',
                'check_basis':'submitted source, input manifest and captured failure'}]}
        p=self.plan()
        r=self.execute('import subprocess\n',p)
        fact=self.fact(r)
        self.assertEqual(fact['failure_stage'],'submission')
        self.assertTrue(fact['submission_rejected'])
        self.assertFalse(fact['process_started'])
        self.assertFalse(fact['executed'])
        self.assertEqual(fact['r_events']['implementation_failure']['label'],1)
        self.assertEqual(fact['r_events']['environment_failure']['label'],0)
        self.assertEqual(fact['r_events']['timeout']['label'],0)
        self.assertFalse(fact['r_events']['timeout']['runtime_eligible'])
        self.assertIn('never ran',fact['r_events']['timeout']['basis'])
        self.assertEqual(r.cost.measured_dims(),{'latency_s'})
        self.assertGreater(r.cost.latency_s,0)
        self.assertIn('submitted_script',r.execution_features['artifacts'])
        self.assertEqual(r.execution_features['input_resources'][0]['path'],'data.txt')
        self.assertEqual(self.h.actions.get(r.action_id).params['prediction_id'],p)
        with self.assertRaises(ValueError): self.execute('pass',p)
        close=self.h.close_episode(self.task['task_id'],'ep')
        timeout=close['evaluations'][0]['risk']['r_events_observation']['events']['timeout']
        self.assertEqual(timeout['label'],0)
        self.assertFalse(timeout['runtime_eligible'])
        self.assertEqual(self.h.calibration_summary()['technical_r']['runtime_coverage']['script_not_started'],1)
        pending=close['induction_schedule']['pending_ids']
        self.assertEqual(len(pending),1)
        material=self.h.induction_material(induction_id=pending[0])
        self.assertIn(r.execution_id,json.dumps(material))
        from or_harness.world_model.hplus import jobs
        from or_harness.world_model.conditions import get_value
        held=get_value(self.h.store,jobs(self.h)[0]['source_fact_ref'])
        self.assertIn('data.txt',json.dumps(held))

    def test_input_missing_changed_or_artifact_corrupt_are_preserved_and_unknown(self):
        for variant in ('missing','changed','corrupt'):
            with self.subTest(variant=variant):
                # Distinct task/episode chain for each controlled submission.
                self.task['task_id']='input-'+variant
                self.task.pop('input_resources',None)
                origin=self.resource(variant+'.txt','original')
                p=self.plan()
                frozen=self.h.strategy_predictions.get(p).trace.model_info['input_resources']
                if variant=='missing': origin.unlink()
                elif variant=='changed': origin.write_text('different')
                else:
                    artifact=self.h.home/frozen[0]['artifact']['path']
                    artifact.chmod(0o644)  # controlled host-side corruption, outside the jail
                    artifact.write_text('tampered')
                r=self.execute("open('result.json','w').write('{}')",p)
                fact=self.fact(r)
                self.assertEqual(fact['failure_stage'],'submission')
                self.assertFalse(fact['executed'])
                self.assertIsNone(fact['label'])
                self.assertIsNone(fact['r_events']['environment_failure']['label'])
                self.assertEqual(r.execution_features['prediction_id'],p)
                self.assertEqual(r.execution_features['input_resources'],frozen)

    def test_process_launch_failure_is_environment_with_no_runtime(self):
        p=self.plan()
        with patch('or_harness.execution.executor.subprocess.Popen',side_effect=OSError('fixture spawn unavailable')):
            r=self.execute('pass',p)
        fact=self.fact(r)
        self.assertEqual(fact['failure_stage'],'launch')
        self.assertFalse(fact['process_started'])
        self.assertFalse(fact['executed'])
        self.assertIsNone(fact['label'])
        self.assertEqual(fact['r_events']['environment_failure']['label'],1)
        self.assertEqual(r.cost.measured_dims(),{'latency_s'})

    def test_bootstrap_failure_has_process_but_no_script_runtime(self):
        p=self.plan()
        with patch('or_harness.execution.sandbox.command',return_value=['/bin/false']):
            r=self.execute('pass',p)
        fact=self.fact(r)
        self.assertEqual(fact['failure_stage'],'sandbox_setup')
        self.assertTrue(fact['process_started'])
        self.assertFalse(fact['executed'])
        self.assertEqual(fact['r_events']['environment_failure']['label'],1)
        self.assertEqual(r.execution_features['tool_calls_lower_bound'],1)

    def test_python_and_solver_api_exceptions_are_runtime_implementation_failures(self):
        for source in ("raise ValueError('bad index')", "raise TypeError('solver API takes a list')"):
            with self.subTest(source=source):
                r=self.execute(source)
                fact=self.fact(r)
                self.assertEqual(fact['failure_stage'],'runtime')
                self.assertTrue(fact['executed'])
                self.assertTrue(fact['process_started'])
                self.assertEqual(fact['label'],1)
                self.assertEqual(fact['r_events']['environment_failure']['label'],0)
                self.assertIn('solver_runtime_s',r.cost.measured_dims())

    def test_setup_deadline_is_formal_timeout_without_algorithm_runtime(self):
        self.h.executor=SafePythonExecutor(timeout_seconds=1)
        p=self.plan()
        with patch('or_harness.execution.sandbox.command',return_value=[
                sys.executable,'-I','-c','import time; time.sleep(3)']):
            r=self.execute('pass',p)
        fact=self.fact(r)
        self.assertEqual(r.quality['status'],'timeout')
        self.assertEqual(fact['failure_stage'],'sandbox_setup')
        self.assertTrue(fact['process_started'])
        self.assertFalse(fact['runtime_eligible'])
        self.assertEqual(fact['r_events']['timeout']['label'],1)
        self.assertIsNone(fact['r_events']['environment_failure']['label'])
        self.assertNotIn('solver_runtime_s',r.cost.measured_dims())

    def test_missing_dependency_and_unavailable_license_are_environment_failures(self):
        for source in ("import nonexistent_build63_dependency", "raise RuntimeError('solver license unavailable')"):
            with self.subTest(source=source):
                fact=self.fact(self.execute(source))
                self.assertEqual(fact['failure_stage'],'runtime')
                self.assertTrue(fact['executed'])
                self.assertIsNone(fact['label'])
                self.assertEqual(fact['r_events']['environment_failure']['label'],1)

    def test_invalid_result_status_is_a_result_contract_failure(self):
        fact=self.fact(self.execute("import json\njson.dump({'status':'banana'},open('result.json','w'))"))
        self.assertEqual(fact['failure_stage'],'result')
        self.assertTrue(fact['result_received'])
        self.assertEqual(fact['label'],1)
        self.assertEqual(fact['r_events']['environment_failure']['label'],0)

    def test_formal_timeout_is_separate_from_solver_limit_return(self):
        self.h.executor=SafePythonExecutor(timeout_seconds=1)
        timed=self.execute('import time\ntime.sleep(3)')
        fact=self.fact(timed)
        self.assertEqual(timed.quality['status'],'timeout')
        self.assertEqual(fact['failure_stage'],'runtime')
        self.assertTrue(fact['runtime_eligible'])
        self.assertEqual(fact['r_events']['timeout']['label'],1)
        self.assertIsNone(fact['label'])
        returned=self.execute("import json\njson.dump({'status':'timeout','objective_value':3,'variables':{'x':3}},open('result.json','w'))")
        own=self.fact(returned)
        self.assertTrue(own['result_received'])
        self.assertEqual(own['r_events']['timeout']['label'],0)
        self.assertIsNone(own['failure_stage'])
        self.assertIn('workspace_result',returned.execution_features['artifacts'])

    def test_executor_exception_keeps_already_captured_script_and_inputs(self):
        self.resource('data.txt','1')
        p=self.plan()
        script=self.work/'solve.py'
        script.write_text('pass')
        with patch.object(self.h,'_run_executor',side_effect=RuntimeError('fixture executor failed')):
            with self.assertRaises(RuntimeError):
                self.h.execute(self.task,code_path=str(script),workspace=str(self.work),prediction_id=p)
        r=self.h.bank.pending()[0]
        self.assertEqual(r.execution_features['predicted_for'],p)
        self.assertIn('submitted_script',r.execution_features['artifacts'])
        self.assertEqual(r.execution_features['input_resources'][0]['path'],'data.txt')
        action=self.h.actions.get(self.h.strategy_predictions.get(p).trace.model_info['bound_action_id'])
        self.assertEqual(action.linked_execution_id,r.execution_id)
