"""r7 joint-change semantics: verification, Q, cursor, episode binding.

Pins the behaviours this round introduced, each against a real defect the r6
ten-problem run exposed:

1. A probe PATH that does not resolve is ``insufficient``, NOT a refutation
   (r6 wrote ``outcome.gap`` for the record's ``quality.gap`` and was
   reported refuted).
2. ``code_unchanged`` backs a "the code was not changed" claim from the
   recorded code hash — an ``optimal`` status never does.
3. A task-check reference is tagged with its ``reference_source``; an
   unstated source is reported as self-derived.
4. A ``solution_quality`` of 0 whose own notes say the run is optimal is a
   gap/quality conflict (the value must be Q = 1 - gap), reported not
   silently converted.
5. ``review-material`` pages with a cursor, so a long history is walked in
   distinct batches instead of re-showing the newest records.
6. Binding a prediction to a real action whose episode it lacked backfills
   the episode, and a close-out reports bound predictions outside it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.strategy_prediction import (  # noqa: E402
    parse_strategy_outcome_payload,
)
from or_harness.world_model.contracts import CandidateRef  # noqa: E402
from or_harness.strategy.verification import (  # noqa: E402
    INSUFFICIENT,
    REFUTED,
    VERIFIED,
    verify_relation,
    verify_task_result,
)


def _rec(execution_id, *, gap=0.0, status="optimal", code_hash="abc123",
         task_id="t"):
    return {
        "execution_id": execution_id, "task_id": task_id,
        "strategy_id": "S", "measurement_scope": "attempt",
        "quality": {"status": status, "feasible": True, "gap": gap,
                    "objective": 10.0},
        "cost": {"measured": ["llm_tokens"], "llm_tokens": 10},
        "solver": {"name": "highs", "code_hash": code_hash},
    }


class TestProbePathHonesty(unittest.TestCase):
    def test_unresolved_path_is_insufficient_not_refuted(self):
        report = verify_relation(
            "c", evidence=[_rec("ex_1")],
            roles=[{"execution_id": "ex_1", "role": "a"}],
            assertions=[{"kind": "probe", "roles": ["a"],
                         "path": "outcome.gap", "equals": 0.0}])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("does not resolve", report["conclusion"])

    def test_resolved_path_still_decides(self):
        report = verify_relation(
            "c", evidence=[_rec("ex_1", gap=0.0)],
            roles=[{"execution_id": "ex_1", "role": "a"}],
            assertions=[{"kind": "probe", "roles": ["a"],
                         "path": "quality.gap", "equals": 0.0}])
        self.assertEqual(report["state"], VERIFIED)

    def test_task_check_wrong_path_is_insufficient(self):
        report = verify_task_result(
            _rec("ex_1"),
            {"semantic_probe": {"path": "outcome.gap", "equals": 0.0}})
        self.assertEqual(report["state"], "insufficient")


class TestCodeUnchangedAssertion(unittest.TestCase):
    def test_same_hash_verifies(self):
        report = verify_relation(
            "c", evidence=[_rec("ex_1"), _rec("ex_2")],
            roles=[{"execution_id": "ex_1", "role": "a"},
                   {"execution_id": "ex_2", "role": "a"}],
            assertions=[{"kind": "code_unchanged", "roles": ["a"]}])
        self.assertEqual(report["state"], VERIFIED)

    def test_different_hash_refutes(self):
        report = verify_relation(
            "c", evidence=[_rec("ex_1", code_hash="aaa"),
                           _rec("ex_2", code_hash="bbb")],
            roles=[{"execution_id": "ex_1", "role": "a"},
                   {"execution_id": "ex_2", "role": "a"}],
            assertions=[{"kind": "code_unchanged", "roles": ["a"]}])
        self.assertEqual(report["state"], REFUTED)

    def test_missing_hash_is_insufficient(self):
        record = _rec("ex_1")
        record["solver"] = {"name": "highs"}  # no code hash
        report = verify_relation(
            "c", evidence=[record],
            roles=[{"execution_id": "ex_1", "role": "a"}],
            assertions=[{"kind": "code_unchanged", "roles": ["a"]}])
        self.assertEqual(report["state"], INSUFFICIENT)


class TestReferenceSource(unittest.TestCase):
    def test_unstated_source_is_self_derived(self):
        report = verify_task_result(_rec("ex_1", gap=0.0),
                                    {"reference_objective": 10.0})
        self.assertEqual(report["state"], "passed")
        source = report["reference_source"]
        self.assertIsNone(source["source"])
        self.assertFalse(source["independent"])

    def test_bench_declared_is_independent(self):
        report = verify_task_result(
            _rec("ex_1"),
            {"reference_objective": 7.0, "reference_source": "bench_declared"})
        self.assertEqual(report["state"], "failed")
        self.assertTrue(report["reference_source"]["independent"])


class TestGapQualityConflict(unittest.TestCase):
    def _candidate(self):
        return CandidateRef.from_dict(
            {"task_id": "t", "strategy_id": "S",
             "action_type": "execute_strategy"})

    def test_zero_with_optimal_notes_is_a_problem(self):
        prediction = parse_strategy_outcome_payload(
            {"benefit": {"kind": "solution_quality",
                         "metric": "normalized_objective_gap", "value": 0.0,
                         "feasible": True,
                         "notes": ["HiGHS should solve to optimality "
                                   "(gap=0)"]}},
            self._candidate())
        self.assertEqual(prediction.status, "invalid")
        self.assertTrue(any("QUALITY Q = 1 - gap" in n
                            for n in prediction.notes))

    def test_zero_without_optimal_language_is_kept(self):
        prediction = parse_strategy_outcome_payload(
            {"benefit": {"kind": "solution_quality",
                         "metric": "normalized_objective_gap", "value": 0.0,
                         "baseline": {"kind": "no_knowledge", "value": 0.0}}},
            self._candidate())
        # No conflict language: the value is kept as written.
        self.assertFalse(any("QUALITY Q = 1 - gap" in n
                             for n in prediction.notes))


def _profile(problem_id="t1", family="routing", **coupling):
    from or_harness.core.schema import ProblemProfile
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values)


class TestReviewMaterialCursor(HarnessTestCase):
    def test_cursor_reads_the_next_older_batch(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(8):
            h.record(self.make_record(
                execution_id=f"ex_cursor_{i}", task_id=f"T_{i}",
                strategy_id="S", profile=_profile(f"T_{i}")))
        old = os.environ.get("OR_HARNESS_INDUCTION_MATERIAL_CHARS")
        os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = "2800"
        try:
            first = h.induction_material()
            self.assertTrue(first["budget"]["truncated_by_budget"])
            cursor = first["budget"]["next_cursor"]
            self.assertIsNotNone(cursor)
            second = h.induction_material(cursor=cursor)
        finally:
            if old is None:
                os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS", None)
            else:
                os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = old
        first_ids = {m["execution_id"] for m in first["material"]}
        second_ids = {m["execution_id"] for m in second["material"]}
        # The cursor advances to OLDER material — no overlap with the first.
        self.assertTrue(second_ids)
        self.assertFalse(first_ids & second_ids)

    def test_entry_reports_code_hash_and_changes(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.record(self.make_record(execution_id="ex_v1", task_id="T",
                                  strategy_id="S", profile=_profile("T")))
        later = self.make_record(execution_id="ex_v2", task_id="T",
                                 strategy_id="S", profile=_profile("T"))
        later.created_at = h.bank.get("ex_v1").created_at + 10
        h.record(later)
        result = h.induction_material(task_id="T")
        entry = next(m for m in result["material"]
                     if m["execution_id"] == "ex_v2")
        self.assertIn("code_hash", entry["outcome"])
        self.assertIn("changes", entry)
        self.assertIn("code_hash_changed", entry["changes"])

    def test_cross_task_hint_reports_the_task_span(self):
        """The batch tells the agent how many DISTINCT tasks it spans, so a
        single-task review is visibly single-task."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # One task only: the hint must warn against a one-task claim.
        h.record(self.make_record(execution_id="ex_solo", task_id="T_solo",
                                  strategy_id="S", profile=_profile("T_solo")))
        solo = h.induction_material()
        hint = solo["cross_task_hint"]
        self.assertEqual(hint["n_distinct_tasks_in_batch"], 1)
        self.assertIn("1 task only", hint["note"])
        self.assertIn("conditional_fact", hint["note"])
        # Two tasks: the hint points at the cross-task opportunity.
        for i in range(2):
            h.record(self.make_record(
                execution_id=f"ex_pair_{i}", task_id=f"T_pair_{i}",
                strategy_id="S", profile=_profile(f"T_pair_{i}")))
        pair = h.induction_material()
        hint = pair["cross_task_hint"]
        self.assertGreaterEqual(hint["n_distinct_tasks_in_batch"], 2)
        self.assertIn("look for a mechanism that recurs", hint["note"])


class TestEpisodeBackfill(HarnessTestCase):
    def _work(self):
        import tempfile
        from pathlib import Path
        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(self._tmpdir.cleanup)
        return Path(self._tmpdir.name)

    def test_bind_backfills_a_missing_prediction_episode(self):
        from or_harness.world_model.provider import WorldModelProvider

        class Stub(WorldModelProvider):
            name = "stub"

            def predict(self, request, timeout_s=None):
                return {"payload": {"benefit": {
                    "kind": "solution_quality",
                    "metric": "normalized_objective_gap", "value": 0.9,
                    "baseline": {"kind": "no_knowledge", "value": 0.0}}},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        h = ORHarness(home=self.home, world_model=Stub())
        self.addCleanup(h.close)
        work = self._work()
        task = {"task_id": "T", "family": "routing"}
        # A prediction made with NO episode.
        spec = ActionSpec("execute_strategy", "T", strategy_id="S01")
        prediction = h.predict_strategy_outcome(task, spec, None)
        self.assertIsNone(prediction.candidate.episode_id)
        # Its real action runs under an episode.
        script = self._mitm_script()
        record = h.execute(task, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        reloaded = h.get_strategy_outcome_prediction(prediction.prediction_id)
        self.assertEqual(reloaded.candidate.episode_id, "ep1")
        self.assertTrue(reloaded.trace.model_info.get(
            "episode_backfilled_from_action"))

    def _mitm_script(self):
        from pathlib import Path
        path = self._work() / "solve.py"
        path.write_text("import json\n"
                        "with open('result.json','w') as fh:\n"
                        "    json.dump({'status':'feasible',"
                        "'objective_value':900.0}, fh)\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
