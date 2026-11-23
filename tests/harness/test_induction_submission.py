"""Compact agent induction submission: one JSON, existing lifecycle underneath."""

import copy
import io
import json
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

from test_hplus_binary import BinaryCase
from test_rule_versions import RuleCase, SHORT_RULE
from or_harness.api import ORHarness
from or_harness.cli import main
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.rule_versions import event_records, inspect_versions
from or_harness.world_model.conditions import get_value
from or_harness.world_model.hplus import jobs, receipt_observation
from or_harness.world_model.induction_submission import (
    normalize_submission,
)


def body(harness, entry):
    return get_value(harness.store, entry.rule['body_ref'])


class CompactCase(BinaryCase):
    def job(self, induction_id):
        return next(j for j in jobs(self.h) if j['induction_id'] == induction_id)

    def material(self, induction_id):
        return self.h.induction_material(induction_id=induction_id)

    def compact(self, induction_id, record, *, result='new_rule', rule=None,
                refs=None, publication=None, analysis=None, **extra):
        job = self.job(induction_id)
        submission = {
            'result': result,
            'rule': copy.deepcopy(rule if rule is not None else
                                 (SHORT_RULE if result in ('new_rule', 'revision')
                                  else None)),
            'basis': {
                'kind': 'derivation',
                'refs': list(refs or [record.execution_id,
                                      job['source_fact_ref']]),
                'analysis': analysis or (
                    'The recorded actual method and the independent integer '
                    'check agree on the fixed-Y endpoint argument.'),
            },
        }
        if publication is not None:
            submission['publication'] = publication
        submission.update(extra)
        return submission


class TestCompactSubmission(CompactCase):
    def test_new_rule_success_uses_when_do_expect_and_candidate_by_default(self):
        p, record, closed = self.closed('compact-new')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission(
            self.compact(job_id, record), induction_id=job_id)
        entry_id = result['relations'][0]['saved']
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(body(self.h, entry)['when'], SHORT_RULE['when'])
        self.assertEqual(body(self.h, entry)['do'], SHORT_RULE['do'])
        self.assertEqual(body(self.h, entry)['expect'], SHORT_RULE['expect'])
        self.assertEqual(entry.predicates, {})
        self.assertEqual(body(self.h, entry)['effect_scope'], 'task_check')
        self.assertEqual(entry.rule['candidate_version'], 1)
        self.assertFalse(entry.is_published)
        receipt = result['induction_receipt']
        self.assertEqual(receipt['outcome'], 'new_rule')
        self.assertEqual(receipt['rule_updates'][0]['version'], 1)
        self.assertTrue(receipt['rule_updates'][0]['substantive'])
        self.assertEqual(receipt['comparison']['basis'], 'derivation')
        self.assertTrue(receipt['comparison']['sufficient'])
        self.assertIn(record.execution_id, receipt['comparison']['source'])
        self.assertEqual(receipt_observation(self.h, self.job(job_id))['label'], 1)
        self.assertEqual(self.job(job_id)['state'], 'completed')

    def test_outer_runner_can_return_a_compact_reviewer_submission(self):
        p, record, closed = self.closed('compact-runner')
        seen = []

        def reviewer(material, job):
            seen.append((material['induction_gate']['induction_id'],
                         job['induction_id']))
            return {'submission': {
                'result': 'new_rule',
                'rule': SHORT_RULE,
                'basis': {'kind': 'derivation',
                          'refs': [record.execution_id, job['source_fact_ref']],
                          'analysis': 'The runner received the actual material and frozen job.'},
            }}

        result = self.h.process_induction_jobs(reviewer)
        self.assertEqual(seen, [(closed['induction_schedule']['pending_ids'][0],
                                 closed['induction_schedule']['pending_ids'][0])])
        self.assertEqual(result['results'][0]['saved'], 1)
        self.assertEqual(self.h.sbank.count(), 1)

    def test_compact_revision_inherits_technical_effect_scope(self):
        _, first, closed = self.closed('compact-technical-a')
        job1 = closed['induction_schedule']['pending_ids'][0]
        self.material(job1)
        created = self.h.induce_submission(
            self.compact(job1, first,
                         rule={**SHORT_RULE, 'effect_scope': 'technical'}),
            induction_id=job1)
        entry_id = created['relations'][0]['saved']
        versions = inspect_versions(self.h, entry_id)['versions']
        self.assertEqual(versions[0]['body']['effect_scope'], 'technical')

        _, second, closed2 = self.closed('compact-technical-b')
        job2 = closed2['induction_schedule']['pending_ids'][0]
        self.material(job2)
        revised = self.h.induce_submission(
            self.compact(job2, second, result='revision',
                         rule={**SHORT_RULE,
                               'do': SHORT_RULE['do'] + ' Record the chosen endpoint.'},
                         entry_id=entry_id, version=1),
            induction_id=job2)
        versions = inspect_versions(self.h, entry_id)['versions']
        self.assertEqual(revised['relations'][0]['version'], 2)
        self.assertEqual(versions[1]['body']['effect_scope'], 'technical')

    def test_artifact_reference_binds_to_the_job_trigger_execution(self):
        p, record, closed = self.closed('compact-artifact')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission({
            'result': 'new_rule',
            'rule': SHORT_RULE,
            'basis': {'kind': 'artifact_inspection',
                      'refs': ['artifacts.submitted_script'],
                      'analysis': 'The captured script performs the same endpoint enumeration the check accepted.'},
        }, induction_id=job_id)
        relation = result['relations'][0]
        self.assertEqual(relation['execution_ids'], [record.execution_id])
        self.assertEqual(result['induction_receipt']['source_execution_ids'],
                         [record.execution_id])
        self.assertEqual(receipt_observation(self.h, self.job(job_id))['label'], 1)

    def test_revision_and_evidence_added_keep_version_history(self):
        _, first, closed = self.closed('compact-revision-a')
        job1 = closed['induction_schedule']['pending_ids'][0]
        self.material(job1)
        published = self.compact(job1, first, publication={
            'state': 'published',
            'reason': 'The recorded endpoint and independent integer check support this scope.',
        })
        first_result = self.h.induce_submission(published, induction_id=job1)
        entry_id = first_result['relations'][0]['saved']
        self.assertTrue(self.h.sbank.get(entry_id).is_published)

        _, second, closed2 = self.closed('compact-revision-b')
        job2 = closed2['induction_schedule']['pending_ids'][0]
        self.material(job2)
        revised_rule = {**SHORT_RULE,
                        'do': SHORT_RULE['do'] + ' Record the chosen endpoint in the result.'}
        revised = self.compact(job2, second, result='revision',
                               rule=revised_rule,
                               entry_id=entry_id, version=1)
        result = self.h.induce_submission(revised, induction_id=job2)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(result['relations'][0]['version'], 2)
        self.assertEqual(entry.rule['active_version'], 1)
        self.assertEqual(entry.rule['candidate_version'], 2)
        versions = inspect_versions(self.h, entry_id)['versions']
        self.assertEqual(versions[1]['body']['do'], revised_rule['do'])
        self.assertEqual(result['induction_receipt']['outcome'], 'revised')
        self.assertEqual(result['induction_receipt']['rule_updates'][0]['version'], 2)
        self.assertTrue(result['induction_receipt']['rule_updates'][0]['substantive'])
        self.assertEqual(len(versions), 2)

        _, third, closed3 = self.closed('compact-revision-c')
        job3 = closed3['induction_schedule']['pending_ids'][0]
        self.material(job3)
        evidence_added = self.compact(
            job3, third, result='evidence_added', entry_id=entry_id, version=2)
        third_result = self.h.induce_submission(evidence_added, induction_id=job3)
        self.assertEqual(third_result['induction_receipt']['outcome'], 'evidence_added')
        self.assertEqual(third_result['induction_receipt']['rule_updates'][0]['version'], 2)
        self.assertFalse(third_result['induction_receipt']['rule_updates'][0]['new_version'])
        self.assertEqual(receipt_observation(self.h, self.job(job3))['label'], 0)
        self.assertEqual(self.h.sbank.get(entry_id).rule['candidate_version'], 2)

    def test_already_covered_and_no_new_knowledge_are_supported_zeros(self):
        entry_id, _, _ = self.seed()
        self.activate(entry_id)
        _, record, closed = self.closed('compact-covered')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission(
            self.compact(job_id, record, result='already_covered',
                         rule=None, refs=[self.job(job_id)['source_fact_ref'],
                                          self.job(job_id)['signal']['source_snapshot_ref']]),
            induction_id=job_id)
        self.assertEqual(result['saved'], 0)
        self.assertEqual(result['induction_receipt']['outcome'], 'already_covered')
        self.assertFalse(result['induction_receipt']['rule_updates'])
        self.assertEqual(receipt_observation(self.h, self.job(job_id))['label'], 0)
        self.assertEqual(self.h.sbank.count(), 1)

    def test_no_new_knowledge_maps_to_no_increment(self):
        _, record, closed = self.closed('compact-no-new')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission(
            {'result': 'no_new_knowledge',
             'basis': {'kind': 'recorded_checks',
                       'refs': [self.job(job_id)['source_fact_ref']],
                       'analysis': 'The recorded check and the frozen scope show no reusable addition.'}},
            induction_id=job_id)
        self.assertEqual(result['induction_receipt']['outcome'], 'no_increment')
        self.assertEqual(receipt_observation(self.h, self.job(job_id))['label'], 0)
        self.assertEqual(self.h.sbank.count(), 0)

    def test_insufficient_evidence_records_unknown_without_a_rule(self):
        _, record, closed = self.closed('compact-insufficient')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission(
            {'result': 'insufficient_evidence',
             'hypothesis_outcome': 'unknown',
             'deviation_reason': 'No comparable checked case is available.'},
            induction_id=job_id)
        self.assertEqual(result['induction_receipt']['outcome'], 'insufficient_material')
        self.assertEqual(result['induction_receipt']['hypothesis_outcome'], 'unknown')
        self.assertIsNone(result['induction_receipt']['comparison'])
        self.assertEqual(self.h.sbank.count(), 0)
        self.assertIsNone(receipt_observation(self.h, self.job(job_id))['label'])

    def test_missing_or_bad_evidence_refuses_without_writing(self):
        _, record, closed = self.closed('compact-bad-evidence')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        with self.assertRaises(ValueError):
            self.h.induce_submission({'result': 'new_rule', 'rule': SHORT_RULE},
                                     induction_id=job_id)
        with self.assertRaises(ValueError):
            self.h.induce_submission(
                self.compact(job_id, record, refs=['execution:does-not-exist']),
                induction_id=job_id)
        with self.assertRaises(ValueError):
            self.h.induce_submission(
                self.compact(job_id, record, refs=['rule_fact:does-not-exist']),
                induction_id=job_id)
        self.assertEqual(self.h.sbank.count(), 0)
        self.assertEqual(self.job(job_id)['state'], 'pending')

    def test_dry_run_then_commit_and_repeat_is_idempotent(self):
        _, record, closed = self.closed('compact-idempotent')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        submission = self.compact(job_id, record)
        preview = self.h.induce_submission(submission, induction_id=job_id,
                                           dry_run=True)
        self.assertTrue(preview['dry_run'])
        self.assertEqual(preview['would_create'], 1)
        self.assertEqual(preview['saved'], 0)
        self.assertEqual(self.h.sbank.count(), 0)
        self.assertEqual(self.job(job_id)['state'], 'pending')

        result = self.h.induce_submission(submission, induction_id=job_id)
        repeated = self.h.induce_submission(submission, induction_id=job_id)
        self.assertTrue(repeated['already_completed'])
        self.assertEqual(repeated['induction_receipt'], result['induction_receipt'])
        self.assertEqual(self.h.sbank.count(), 1)
        self.assertEqual(len(self.h.actions.query(action_type='induce')), 1)

    def test_mid_failure_keeps_job_pending_and_retry_reuses_action(self):
        _, record, closed = self.closed('compact-recovery')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        submission = self.compact(job_id, record)
        with patch.object(self.h, '_end_induce_action',
                          side_effect=RuntimeError('interrupted before commit')):
            with self.assertRaises(RuntimeError):
                self.h.induce_submission(submission, induction_id=job_id)
        self.assertEqual(self.h.sbank.count(), 0)
        action_id = self.job(job_id)['action_id']
        self.assertIsNotNone(action_id)
        result = self.h.induce_submission(submission, induction_id=job_id)
        self.assertEqual(result['relations'][0]['saved'], self.h.sbank.list()[0].entry_id)
        self.assertEqual(self.job(job_id)['action_id'], action_id)
        self.assertEqual(self.job(job_id)['state'], 'completed')

    def test_hplus_zero_is_not_authorized_by_compact_submission(self):
        _, record, closed = self.closed('compact-zero', decision=0)
        jobs_here = closed['induction_schedule']['jobs']
        self.assertEqual(jobs_here[0]['state'], 'skipped')
        self.assertFalse(closed['induction_schedule']['pending_ids'])
        # A skipped (H+ = 0) job is not an authorization to write a rule.
        _, _, closed_positive = self.closed('compact-zero-positive')
        positive_job = closed_positive['induction_schedule']['pending_ids'][0]
        self.material(positive_job)
        zero_job = next(j for j in jobs(self.h)
                        if j['task_id'] == 'compact-zero')
        with self.assertRaises(ValueError):
            self.h.induce_submission(
                self.compact(positive_job, record), induction_id=zero_job['induction_id'])
        self.assertEqual(self.h.sbank.count(), 0)

    def test_compact_and_legacy_payloads_are_equivalent_on_the_same_job(self):
        p, record, closed = self.closed('compact-equivalence')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        job = self.job(job_id)
        submission = self.compact(
            job_id, record,
            publication={'state': 'published',
                         'reason': 'The recorded derivation and integer check support this scope.'})
        normalized = normalize_submission(self.h, submission, induction_id=job_id)

        # Build a *legacy* call from the same semantic input in an isolated
        # copy of the pre-submission memory.
        root = tempfile.mkdtemp(prefix='compact-legacy-',
                                dir=os.path.dirname(self.home))
        destination = os.path.join(root, 'home')
        self.addCleanup(shutil.rmtree, root, True)
        from or_harness.world_model.snapshot_maintenance import fork_memory
        fork_memory(self.home, destination)
        legacy = ORHarness(home=destination,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(legacy.close)
        legacy.induction_material(induction_id=job_id)
        compact_result = self.h.induce_submission(submission, induction_id=job_id)
        legacy_result = legacy.induce(
            induction_id=job_id,
            relations=normalized['relations'],
            review=normalized['review'])
        self.assertEqual(legacy_result['induction_receipt']['outcome'],
                         compact_result['induction_receipt']['outcome'])
        compact_entry = self.h.sbank.get(compact_result['relations'][0]['saved'])
        legacy_entry = legacy.sbank.get(legacy_result['relations'][0]['saved'])
        for field in ('when', 'do', 'expect'):
            self.assertEqual(body(self.h, compact_entry)[field],
                             body(legacy, legacy_entry)[field])
        self.assertEqual(compact_entry.predicates, legacy_entry.predicates)
        self.assertEqual(compact_entry.rule['active_version'],
                         legacy_entry.rule['active_version'])
        self.assertEqual(compact_entry.rule['candidate_version'],
                         legacy_entry.rule['candidate_version'])
        self.assertEqual(compact_entry.is_published, legacy_entry.is_published)
        self.assertEqual(compact_entry.verification_state,
                         legacy_entry.verification_state)
        cr = compact_result['induction_receipt']
        lr = legacy_result['induction_receipt']
        for field in ('outcome', 'hypothesis_outcome', 'verification',
                      'source_execution_ids', 'comparison'):
            self.assertEqual(cr[field], lr[field])
        self.assertEqual(cr['rule_updates'][0]['version'],
                         lr['rule_updates'][0]['version'])
        self.assertEqual(cr['rule_updates'][0]['substantive'],
                         lr['rule_updates'][0]['substantive'])
        self.assertEqual(cr['rule_updates'][0]['new_version'],
                         lr['rule_updates'][0]['new_version'])
        self.assertEqual(
            [(e['kind'], e.get('execution_id')) for e in
             event_records(self.h.store, compact_entry.entry_id)],
            [(e['kind'], e.get('execution_id')) for e in
             event_records(legacy.store, legacy_entry.entry_id)])

    def test_compact_schema_rejects_ambiguous_fields(self):
        from or_harness.world_model.induction_submission import normalize_submission
        base = {'result': 'new_rule', 'rule': SHORT_RULE,
                'basis': {'kind': 'derivation', 'refs': ['EX'],
                          'analysis': 'A supported argument over the recorded run.'}}
        with self.assertRaises(ValueError):
            normalize_submission(self.h, {**base, 'result': 'invented'})
        with self.assertRaises(ValueError):
            normalize_submission(self.h, {**base,
                'rule': {**SHORT_RULE, 'predicates': {'family': 'assignment'}}})
        with self.assertRaises(ValueError):
            normalize_submission(self.h, {'result': 'revision', 'rule': SHORT_RULE,
                'basis': base['basis']})
        with self.assertRaises(ValueError):
            normalize_submission(self.h, {'result': 'no_new_knowledge',
                'publication': {'state': 'published', 'reason': 'no write'},
                'basis': {'kind': 'derivation', 'refs': ['EX'],
                          'analysis': 'No reusable addition.'}})
        with self.assertRaises(ValueError):
            normalize_submission(self.h, {**base,
                'basis': {'kind': 'derivation', 'refs': [], 'analysis': 'empty'}})

    def test_cli_submission_and_mixed_interface_rejection(self):
        p, record, closed = self.closed('compact-cli')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        submission = self.compact(job_id, record)
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(['--home', self.home, 'induce',
                         '--induction-id', job_id,
                         '--submission', json.dumps(submission)])
        self.assertEqual(code, 0, out.getvalue())
        payload = json.loads(out.getvalue())
        self.assertEqual(payload['result']['induction_receipt']['outcome'], 'new_rule')
        self.assertEqual(self.h.sbank.count(), 1)

        _, record2, closed2 = self.closed('compact-cli-mixed')
        job2 = closed2['induction_schedule']['pending_ids'][0]
        self.material(job2)
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(['--home', self.home, 'induce',
                         '--induction-id', job2,
                         '--submission', json.dumps(self.compact(job2, record2)),
                         '--review', '{}'])
        self.assertEqual(code, 2, out.getvalue())
        self.assertEqual(self.h.sbank.count(), 1)


class TestCompactOutsideHplus(RuleCase):
    def test_new_rule_writes_without_an_induction_job(self):
        task, record = self.solve()
        self.h.close_episode(task['task_id'], task['task_id'])
        self.h.induction_material(task_id=task['task_id'])
        result = self.h.induce_submission({
            'result': 'new_rule',
            'rule': SHORT_RULE,
            'basis': {'kind': 'derivation', 'refs': [record.execution_id],
                      'analysis': 'The captured endpoint run and integer check support the stated scope.'},
        })
        self.assertEqual(result['saved'], 1)
        self.assertFalse(jobs(self.h))
        self.assertEqual(self.h.sbank.count(), 1)

    def test_cli_summary_reports_the_compact_rule_write(self):
        task, record = self.solve()
        self.h.close_episode(task['task_id'], task['task_id'])
        submission = {
            'result': 'new_rule',
            'rule': SHORT_RULE,
            'basis': {'kind': 'derivation', 'refs': [record.execution_id],
                      'analysis': 'The captured endpoint run and integer check support the stated scope.'},
        }
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(['--home', self.home, 'induce',
                         '--submission', json.dumps(submission)])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn('Saved 1 relation', out.getvalue())


class TestCompactInsufficientWithBasis(CompactCase):
    def test_supplied_but_insufficient_basis_stays_unknown(self):
        _, record, closed = self.closed('compact-insufficient-basis')
        job_id = closed['induction_schedule']['pending_ids'][0]
        self.material(job_id)
        result = self.h.induce_submission({
            'result': 'insufficient_evidence',
            'basis': {'kind': 'derivation',
                      'refs': [record.execution_id],
                      'analysis': 'The run does not establish the proposed boundary.'},
        }, induction_id=job_id)
        self.assertEqual(result['induction_receipt']['outcome'], 'insufficient_material')
        self.assertFalse(result['induction_receipt']['comparison']['sufficient'])
        self.assertIsNone(receipt_observation(self.h, self.job(job_id))['label'])
