"""r21: knowledge UTILITY management — calibration is a diagnostic, maturity
moves only on the agent's explicit management request.

One behaviour per test class:

A. calibration no longer changes maturity: many hits or a miss streak leave
   the status untouched, and the forward track is still replayed/reported;
B. the agent's management request (`lifecycle_action`) promotes/demotes in
   the SAME `induce` call, never via adoption count or a hit rate; the
   administrative checks refuse a request with no real adopted evidence; a
   `refuting` verdict demotes and a later valid request lifts it (history kept);
C. order/consistency: the entry, the knowledge delta, the business result and
   the index agree; dry-run writes nothing; duplicate attribution is idempotent;
D. the induction material shows adoption counts, known-task spread, run
   outcomes and effect verdicts, with "no judgement" as its own state;
E. `induce --task` gets a targeted hint (no alias added).

No real world model; a local hashing embedding backend keeps it hermetic.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS, CostVector, ExecutionRecord, ProblemProfile,
    StrategicEntry)
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")


def _profile(task_id="t1", family="planning"):
    return ProblemProfile(problem_id=task_id, family=family,
                          resource_coupling=0.9, temporal_coupling=0.1,
                          route_complexity=0.8)


class UtilityCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in EMBEDDING_ENV_KEYS}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v)
                                 for k, v in saved.items() if v is not None])
        self.h = ORHarness(home=self.home,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def _qualitative_entry(self, *, strategy_id="method:bound",
                           family="planning"):
        return self.h.sbank.add(StrategicEntry(
            entry_id="", strategy_id=strategy_id,
            pattern={"predicates": {"family": family}},
            claim={"text": "derive a bound then enumerate", "kind": "rule",
                   "conditions": {"predicates": {}, "note": None},
                   "evidence": [],
                   "method": {"name": "bound-then-enumerate",
                              "steps": ["derive bound", "enumerate"]}},
            quality_estimated=False, verification={"state": "unverified"}))

    def _record_adopting(self, entry_id, *, task_id="t1", status="optimal"):
        rec = ExecutionRecord(
            execution_id=f"ex_{task_id}_{entry_id}",
            task_id=task_id, strategy_id="S1",
            profile_snapshot=_profile(task_id),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": status},
            cost=CostVector(llm_tokens=1, tool_calls=0, solver_runtime_s=0.1,
                            retries=0, latency_s=0.1,
                            measured=set(COST_DIMENSIONS)),
            solver={"name": "highs"}, source="executed")
        rec.method_actual = {"name": "bound-then-enumerate",
                             "steps": ["derive bound"]}
        self.h.record(rec, used_entry_ids=[entry_id])
        return rec


class TestCalibrationIsADiagnostic(UtilityCase):
    """A: calibration is replayed but never changes maturity."""

    def test_many_hits_do_not_promote(self):
        eid = self.h.sbank.add(StrategicEntry(
            entry_id="", strategy_id="S1",
            pattern={"predicates": {"family": "planning"}},
            quality_estimated=True, expected_quality_hat=0.9,
            quality_interval=(0.5, 1.0),
            verification={"state": "verified", "claim": "c"}))
        for i in range(6):
            rec = self._record_adopting(eid, task_id=f"t{i}")
            # pretend the declared prediction was hit by an execution check
            self.h.sbank.record_prediction(eid, hit=True)
        self.assertEqual(self.h.sbank.get(eid).status, "candidate")

    def test_forward_track_is_still_reported(self):
        eid = self._qualitative_entry()
        self._record_adopting(eid)
        report = self.h.induce(notes=["review"])["revisions"]
        item = [r for r in report if r["entry_id"] == eid][0]
        self.assertIn("forward", item)
        self.assertFalse(item["forward"]["calibratable"])
        self.assertEqual(item["forward"]["n_predictions"], 0)

    def test_adoption_visible_and_no_auto_promotion(self):
        eid = self._qualitative_entry()
        self._record_adopting(eid)
        report = self.h.induce(notes=["review"])["revisions"]
        item = [r for r in report if r["entry_id"] == eid][0]
        self.assertEqual(item["adoptions"]["n_adoptions"], 1)
        self.assertEqual(self.h.sbank.get(eid).status, "candidate")


class TestExplicitManagement(UtilityCase):
    """B: only the agent's request moves maturity."""

    def test_helped_without_request_does_not_promote(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped", "note": "worked here"}])
        self.assertEqual(self.h.sbank.get(eid).status, "candidate")

    def test_promote_lands_in_the_same_call(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        out = self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped",
             "note": "used it on this task and the effect held",
             "lifecycle_action": "promote",
             "evidence_execution_ids": [rec.execution_id]}])
        stored = self.h.sbank.get(eid)
        self.assertEqual(stored.status, "validated")
        # No numeric prediction was invented, and calibration stays zero.
        self.assertFalse(stored.quality_estimated)
        self.assertEqual(stored.prediction_track.n_predictions, 0)
        # The same call reports the transition, consistent with the store.
        self.assertEqual(out["business_result"], "revised")

    def test_promote_is_idempotent(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        payload = {"entry_id": eid, "execution_id": rec.execution_id,
                   "verdict": "helped", "note": "ok",
                   "lifecycle_action": "promote",
                   "evidence_execution_ids": [rec.execution_id]}
        self.h.induce(effect_attributions=[payload])
        self.h.induce(effect_attributions=[payload])
        self.assertEqual(self.h.sbank.get(eid).status, "validated")

    def test_demote_lands_in_the_same_call(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        self.h.sbank.apply_lifecycle_action(eid, "promote", reason="seed")
        self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "unrelated", "note": "effect not reproduced",
             "lifecycle_action": "demote",
             "evidence_execution_ids": [rec.execution_id]}])
        self.assertEqual(self.h.sbank.get(eid).status, "suspect")

    def test_ordinary_failure_without_attribution_does_not_demote(self):
        eid = self._qualitative_entry()
        rec = ExecutionRecord(
            execution_id="ex_fail", task_id="t1", strategy_id="S1",
            profile_snapshot=_profile("t1"),
            quality={"feasible": False, "objective": None, "status": "error"},
            cost=CostVector(llm_tokens=1, measured={"llm_tokens"}),
            solver={"name": "highs"}, source="executed")
        rec.method_actual = {"name": "bound", "steps": ["x"]}
        self.h.record(rec, used_entry_ids=[eid])
        self.h.induce(notes=["review"])
        self.assertEqual(self.h.sbank.get(eid).status, "candidate")

    def test_request_without_adopted_evidence_is_refused(self):
        eid = self._qualitative_entry()
        # No execution declared adoption of the entry -> refused.
        with self.assertRaises(ValueError):
            self.h.induce(effect_attributions=[
                {"entry_id": eid, "verdict": "helped", "note": "x",
                 "lifecycle_action": "promote"}])
        # No reason -> refused.
        self._record_adopting(eid)
        with self.assertRaises(ValueError):
            self.h.induce(effect_attributions=[
                {"entry_id": eid, "verdict": "helped",
                 "lifecycle_action": "promote",
                 "evidence_execution_ids": [f"ex_t1_{eid}"]}])

    def test_refuting_then_later_promote_lifts_and_history_stays(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "refuting", "note": "did not help"}])
        self.assertEqual(self.h.sbank.get(eid).status, "suspect")
        # A LATER valid management request lifts it; the negative event is
        # kept visible.
        self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped", "note": "re-examined, effect holds",
             "lifecycle_action": "promote",
             "evidence_execution_ids": [rec.execution_id]}])
        stored = self.h.sbank.get(eid)
        self.assertEqual(stored.status, "validated")
        verdicts = [a["verdict"] for a in stored.effect_attribution]
        self.assertIn("refuting", verdicts)


class TestConsistency(UtilityCase):
    """C: one call, one consistent state."""

    def test_delta_and_store_agree(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        out = self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped", "note": "ok", "lifecycle_action": "promote",
             "evidence_execution_ids": [rec.execution_id]}])
        changes = out["knowledge_delta"]["entry_changes"]
        status_change = [c for c in changes if c["entry_id"] == eid]
        self.assertTrue(status_change)
        self.assertEqual(status_change[0]["changed"]["status"]["after"],
                         "validated")
        # The reported after-state matches the store.
        after = {e["entry_id"]: e for e in out["knowledge_after"]["entries"]}
        self.assertEqual(after[eid]["status"], "validated")

    def test_dry_run_writes_nothing(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        self.h.induce(dry_run=True, effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped", "note": "ok", "lifecycle_action": "promote",
             "evidence_execution_ids": [rec.execution_id]}])
        stored = self.h.sbank.get(eid)
        self.assertEqual(stored.status, "candidate")
        self.assertEqual(stored.effect_attribution, [])

    def test_same_feedback_recorded_once(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        payload = {"entry_id": eid, "execution_id": rec.execution_id,
                   "verdict": "helped", "note": "same"}
        self.h.induce(effect_attributions=[payload])
        self.h.induce(effect_attributions=[payload])
        # Two attributions recorded (audit trail), but the adoption count is
        # one — a repeated feedback does not forge extra usage.
        report = self.h.induce(notes=["review"])["revisions"]
        item = [r for r in report if r["entry_id"] == eid][0]
        self.assertEqual(item["adoptions"]["n_adoptions"], 1)

    def test_unknown_entry_refused(self):
        with self.assertRaises(ValueError):
            self.h.induce(effect_attributions=[
                {"entry_id": "999", "verdict": "helped", "note": "x"}])


class TestMaterialShowsUsage(UtilityCase):
    """D: the material surfaces adoptions and effect verdicts."""

    def test_usage_block_with_unattributed_state(self):
        eid = self._qualitative_entry()
        self._record_adopting(eid, task_id="t1")
        self._record_adopting(eid, task_id="t2")
        out = self.h.induction_material()
        item = next(k for k in out["existing_knowledge"]
                    if k["entry_id"] == eid)
        usage = item["usage"]
        self.assertEqual(usage["n_adoptions"], 2)
        self.assertEqual(usage["n_known_tasks"], 2)
        self.assertTrue(usage["unattributed"])

    def test_attribution_appears(self):
        eid = self._qualitative_entry()
        rec = self._record_adopting(eid)
        self.h.induce(effect_attributions=[
            {"entry_id": eid, "execution_id": rec.execution_id,
             "verdict": "helped", "note": "ok"}])
        out = self.h.induction_material()
        item = next(k for k in out["existing_knowledge"]
                    if k["entry_id"] == eid)
        verdicts = [a["verdict"] for a in item["usage"]["attributed_effects"]]
        self.assertIn("helped", verdicts)
        self.assertFalse(item["usage"]["unattributed"])


class TestCliHint(HarnessTestCase):
    """E: `induce --task` gets a targeted hint, not a generic parse error."""

    def test_induce_task_is_explained(self):
        from or_harness import cli
        import io
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home, "induce", "--task", "t1"])
        finally:
            sys.stdout = old
        self.assertEqual(code, 2)
        self.assertIn("induction-material", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
