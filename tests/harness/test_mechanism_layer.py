"""Tests for the snapshot-based cost feedback loop and failure
classification persistence. (Mechanism feature tests removed earlier — the
mechanism machinery was deleted as redundant with coupling features.)

Cost feedback semantics (Cost round): an execution is compared against the
PRE-EXECUTION prediction snapshot actually used (same strategy, same scope,
both sides measured), frozen on the record. Feedback is auxiliary Execution
Evidence (execution_features.cost_feedback); it never updates entry state.

Quality checks follow the same evidence-only rule: they are frozen onto the
fact (execution_features.quality_feedback) and replayed offline by the next
``induce`` (see test_induction.TestOfflineRevision)."""
import unittest

from helpers import HarnessTestCase, MEASURED_ALL

from or_harness.core.schema import CostVector, FailureRecord
from or_harness.api import ORHarness


class TestCostPredictionLoop(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def _entry_with_cost(self, tokens_hat=1500.0):
        recs = []
        for i in (1, 2):
            rec = self.make_record(
                execution_id=f"ex_c{i}", task_id=f"tc{i}", strategy_id="S01",
                cost=CostVector(llm_tokens=tokens_hat, tool_calls=2,
                                solver_runtime_s=1.0, retries=0, latency_s=1.0),
                cost_measured=MEASURED_ALL)
            self.h.bank.append(rec)
            recs.append(rec)
        # The entry under test is a STATISTICAL cost claim: built directly
        # (the statistical induction path is gone; a claim entry is
        # claim-only and carries no cost estimate). It is verified so it is
        # PUBLISHED knowledge before it may serve a prediction.
        from or_harness.core.schema import StrategicEntry, evidence_predicates
        predicates = evidence_predicates(recs, family="routing")
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id="S01",
            pattern={"predicates": predicates},
            expected_quality_hat=1.0,
            quality_interval=(0.5, 1.0),
            expected_cost_hat=CostVector(
                llm_tokens=tokens_hat, tool_calls=2, solver_runtime_s=1.0,
                retries=0, latency_s=1.0, measured=set(MEASURED_ALL)),
            cost_interval={d: (0.5, 2.0) for d in MEASURED_ALL},
            cost_support_n={d: 2 for d in MEASURED_ALL},
            support_n=2,
            verification={"state": "verified", "claim": "S01 holds",
                          "conclusion": "check passed"},
        )
        self.h.sbank.add(entry)
        return entry.entry_id

    def _task(self, task_id):
        return {"task_id": task_id, "family": "routing",
                "annotations": {"coupling": {
                    "semantic_coupling": 0.8, "resource_coupling": 0.9,
                    "temporal_coupling": 0.1, "route_complexity": 0.85}}}

    def test_cost_feedback_from_frozen_snapshot(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        entry = self.h.sbank.get(entry_id)
        self.assertIn("llm_tokens", entry.cost_interval)
        snapshot = self.h.predict_cost(self._task("tc3"), "S01")
        self.assertEqual(snapshot.source, "entry")
        rec = self.make_record(task_id="tc3", strategy_id="S01",
                               cost=CostVector(llm_tokens=1800, tool_calls=2,
                                               solver_runtime_s=1.0, retries=0,
                                               latency_s=1.0),
                               cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec, prediction=snapshot)
        feedback = outcome.get("cost_feedback")
        self.assertIsNotNone(feedback)
        self.assertEqual(feedback["source"], "entry")
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["predicted"], 1500.0, places=3)
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["actual"], 1800.0, places=3)
        # Persisted on the fact (Execution Evidence annotation).
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(stored.execution_features["cost_feedback"], feedback)
        self.assertEqual(stored.prediction_snapshot.source, "entry")

    def test_cost_feedback_never_mutates_entry(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        before = self.h.sbank.get(entry_id).prediction_track.to_dict()
        for i in range(3):
            snapshot = self.h.predict_cost(self._task(f"tc{i + 10}"), "S01")
            rec = self.make_record(task_id=f"tc{i + 10}", strategy_id="S01",
                                   cost=CostVector(llm_tokens=15000,
                                                   tool_calls=2,
                                                   solver_runtime_s=1.0,
                                                   retries=0, latency_s=1.0),
                                   cost_measured=MEASURED_ALL)
            self.h.record(rec, prediction=snapshot)
        after = self.h.sbank.get(entry_id)
        # Cost feedback never touches the lifecycle or the forward track.
        self.assertEqual(after.prediction_track.to_dict(), before)
        self.assertEqual(after.status, "candidate")
        self.assertEqual(after.prediction_track.consecutive_misses, 0)
        # The deviation is recorded per execution instead (evidence).
        stored = self.h.bank.query(task_id="tc10")[0]
        self.assertIn("cost_feedback", stored.execution_features)

    def test_a_execution_never_audits_b_prediction(self):
        self._entry_with_cost(tokens_hat=1500.0)  # entry for S01 only
        rec = self.make_record(task_id="tc_b", strategy_id="S06",
                               cost=CostVector(llm_tokens=300, tool_calls=1,
                                               solver_runtime_s=1.0, retries=0,
                                               latency_s=1.0),
                               cost_measured=MEASURED_ALL)
        # No usable evidence for S06 (no entry, no stats) -> snapshot is
        # unknown -> no feedback at all, and S01's track is untouched.
        snapshot = self.h.predict_cost(self._task("tc_b"), "S06")
        self.assertEqual(snapshot.source, "unknown")
        outcome = self.h.record(rec, prediction=snapshot)
        self.assertNotIn("cost_feedback", outcome)

    def test_quality_misses_demote_at_next_induce(self):
        """Quality misses accumulate as EVIDENCE online and demote the entry
        at the next offline knowledge write — never during record."""
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        for i in range(3):
            rec = self.make_record(task_id=f"tq{i}", strategy_id="S01", gap=0.9,
                                   cost=CostVector(llm_tokens=1500),
                                   cost_measured=MEASURED_ALL)
            outcome = self.h.record(rec)
            # Online: evidence written, knowledge untouched.
            self.assertTrue(outcome["prediction_checks"])
            self.assertEqual(self.h.sbank.get(entry_id).status, "candidate")
            self.assertEqual(
                self.h.sbank.get(entry_id).prediction_track.consecutive_misses, 0)
        # The lifecycle replay runs as part of the next knowledge write.
        result = self.h.induce(relations=[{
            "subject": "S01", "claim": "S01 holds in this cell",
            "evidence": [{"execution_id": "ex_c1", "role": "evidence"},
                         {"execution_id": "ex_c2", "role": "evidence"}]}])
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.status, "suspect")
        revisions = [r for r in result["revisions"] if r["entry_id"] == entry_id]
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]["forward"]["consecutive_misses"], 3)
        self.assertIn("demoted:->suspect", revisions[0]["transitions"])


class TestFailureClassificationPersistence(HarnessTestCase):
    def test_error_class_persisted_at_record(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fc", task_id="tfc", feasible=False, status="error",
            failures=[FailureRecord(1, "security policy: blocked import subprocess")])
        h.record(rec)
        stored = h.bank.get("ex_fc")
        self.assertEqual(stored.failures[0].error_class, "environment")
        h.close()

    def test_model_error_classified_as_model(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fm", task_id="tfm", feasible=False, status="error",
            failures=[FailureRecord(1, "ValueError: bad shape")])
        h.record(rec)
        stored = h.bank.get("ex_fm")
        self.assertEqual(stored.failures[0].error_class, "model")
        h.close()


if __name__ == "__main__":
    unittest.main()
