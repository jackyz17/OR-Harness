"""Method evidence chain: planned vs actually-performed method.

The outer agent proposes a method (``{"name", "steps", ...}``) with a
candidate. This is a PLAN. The method that really ran is an OBSERVATION, and
it comes only from the solve script's own ``method_performed`` receipt
(stamped with the attempt's action id) or from an explicit harness
declaration. These tests pin the whole chain: proposal -> execution record ->
staging -> recording -> read-back -> induction material, and the honest
distinction between the two.
"""
import json
import os
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    ExecutionRecord,
    compare_methods,
    normalize_method,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "t1", "family": "routing", "description": "toy"}

PLAN = {"name": "rolling-horizon decomposition",
        "steps": ["relax the coupling constraint", "solve the master",
                  "solve the subproblem", "recombine"],
        "why": "the coupling is too strong to solve in one shot"}


def _spec(strategy_id, task_id="t1", episode_id="ep1", method=None):
    spec = {"action_type": "execute_strategy", "task_id": task_id,
            "episode_id": episode_id, "strategy_id": strategy_id,
            "solver": "highs"}
    if method is not None:
        spec["method"] = method
    return spec


class ScriptableProvider(WorldModelProvider):
    """Declared synthetic answers — a labelled stub, never a hidden law."""

    name = "scriptable"

    def __init__(self, quality=0.7):
        self.quality = quality
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap",
                        "unit": "1-gap", "value": self.quality,
                        "baseline": {"kind": "conditional_stats", "value": 0.0}},
            "uncertainty": {"execution_randomness": 0.2, "knowledge_gap": 0.3},
        }, "usage": {"completion_tokens": 10}, "error": None,
            "latency_s": 0.01}


class MethodChainCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider = ScriptableProvider()

    def harness(self):
        h = ORHarness(home=self.home, world_model=self.provider)
        self.addCleanup(h.close)
        return h

    def workspace(self, name="ws"):
        ws = Path(self.home) / name
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def write_script(self, ws, body):
        script = ws / "solve.py"
        script.write_text(textwrap.dedent(body), encoding="utf-8")
        return script


SOLVE_OK = """
    import json, os
    result = {"status": "optimal", "objective_value": 42.0,
              "objective_bound": 42.0, "runtime_seconds": 0.01,
              "solver": "highs"}
    aid = os.environ.get("OR_ACTION_ID")
    if aid:
        result["method_performed"] = {
            "action_id": aid,
            "name": "rolling-horizon decomposition",
            "steps": ["relax the coupling constraint", "solve the master",
                      "solve the subproblem", "recombine"],
        }
    with open("result.json", "w") as fh:
        json.dump(result, fh)
"""


class TestNormalizeMethod(unittest.TestCase):
    def test_empty_is_none(self):
        for value in (None, {}, [], "", {"name": "  "},
                      {"steps": ["", "  "]}):
            self.assertIsNone(normalize_method(value), msg=repr(value))

    def test_shape_is_canonical(self):
        got = normalize_method({"name": "Benders", "steps": "cut",
                                "why": " strong coupling ",
                                "fallback": "", "extra_key": 7})
        self.assertEqual(got["name"], "Benders")
        self.assertEqual(got["steps"], ["cut"])       # bare string is one step
        self.assertEqual(got["why"], "strong coupling")
        self.assertNotIn("fallback", got)             # blank is absent
        self.assertEqual(got["extra"], {"extra_key": 7})


class TestCompareMethods(unittest.TestCase):
    def test_unknown_when_a_side_is_missing(self):
        self.assertIsNone(compare_methods(None, {"name": "x", "steps": ["a"]}))
        self.assertIsNone(compare_methods({"name": "x", "steps": ["a"]}, None))

    def test_abbreviated_run_matches(self):
        planned = {"name": "Benders", "steps": ["master", "subproblem",
                                                "cleanup"]}
        actual = {"name": "Benders", "steps": ["master", "subproblem"]}
        observation = compare_methods(planned, actual)
        self.assertEqual(observation["verdict"], "match")

    def test_different_steps_mismatch(self):
        planned = {"name": "Benders", "steps": ["master", "subproblem"]}
        actual = {"name": "Benders", "steps": ["round the solution"]}
        self.assertEqual(compare_methods(planned, actual)["verdict"],
                         "mismatch")

    def test_a_reworded_step_is_not_a_different_method(self):
        # The receipt describes the same work with the plan's own words in a
        # different arrangement ("relax the coupling" for "relax the coupling
        # constraint"): the plan is prose and the receipt is prose, so this
        # is NOT a deviation (the reported regression: rewording blocked the
        # benefit).
        planned = {"name": "reroute",
                   "steps": ["relax the coupling constraint",
                             "solve the master"]}
        actual = {"name": "reroute",
                  "steps": ["relax the coupling", "solve the master"]}
        observation = compare_methods(planned, actual)
        self.assertEqual(observation["verdict"], "reworded")

    def test_new_method_vocabulary_still_mismatches(self):
        # A performed step names work the plan does not: reworded content is
        # tolerated, invented content is not.
        planned = {"name": "reroute", "steps": ["scan the depot space"]}
        actual = {"name": "reroute", "steps": ["scan the depots"]}
        self.assertEqual(compare_methods(planned, actual)["verdict"],
                         "mismatch")
    def test_a_merged_step_is_not_a_different_method(self):
        # Two planned steps performed as one: every performed word is covered
        # by the plan, so the plan was carried out and restated.
        planned = {"name": "Benders",
                   "steps": ["solve the master", "solve the subproblem"]}
        actual = {"name": "Benders",
                  "steps": ["solve the master and the subproblem"]}
        self.assertEqual(compare_methods(planned, actual)["verdict"],
                         "reworded")

    def test_a_renamed_but_step_identical_run_is_reworded(self):
        # Steps line up exactly; only the label differs. A name is a label,
        # not a method — the plan was carried out under another name.
        planned = {"name": "benders", "steps": ["master", "subproblem"]}
        actual = {"name": "cutting-plane benders",
                  "steps": ["master", "subproblem"]}
        observation = compare_methods(planned, actual)
        self.assertEqual(observation["verdict"], "reworded")
        self.assertTrue(observation["name_differs"])

    def test_a_genuinely_different_method_still_mismatches(self):
        # A performed step carries content the plan does not: a merge cannot
        # hide a different method.
        planned = {"name": "Benders", "steps": ["solve the master"]}
        actual = {"name": "Benders",
                  "steps": ["solve the master", "run a local search"]}
        self.assertEqual(compare_methods(planned, actual)["verdict"],
                         "mismatch")

    def test_names_only_is_a_label_not_a_method(self):
        # Both sides carry names but no steps: the verdict rests on the name
        # alone, and that is reported as such.
        planned = {"name": "greedy insertion", "steps": ["seed", "insert"]}
        actual = {"name": "greedy insertion"}
        observation = compare_methods(planned, actual)
        self.assertEqual(observation["verdict"], "match")
        self.assertIn("name", observation["reason"])


class TestMethodRoundTrip(MethodChainCase):
    def test_planned_and_actual_survive_to_dict_from_dict(self):
        record = self.make_record()
        record.method_planned = normalize_method(PLAN)
        record.method_actual = normalize_method(
            {"name": "rolling-horizon decomposition",
             "steps": ["solve the master"], "source": "executor_receipt"})
        restored = ExecutionRecord.from_dict(record.to_dict())
        self.assertEqual(restored.method_planned["name"], PLAN["name"])
        self.assertEqual(restored.method_planned["steps"], PLAN["steps"])
        self.assertEqual(restored.method_actual["steps"], ["solve the master"])
        self.assertEqual(restored.method_actual["source"], "executor_receipt")

    def test_legacy_payload_without_method_keys_loads_as_unknown(self):
        data = self.make_record().to_dict()
        data.pop("method_planned", None)
        data.pop("method_actual", None)
        restored = ExecutionRecord.from_dict(data)
        self.assertIsNone(restored.method_planned)
        self.assertIsNone(restored.method_actual)


class TestExecuteMethodChain(MethodChainCase):
    def test_candidate_method_is_planned_and_the_script_reports_actual(self):
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace()
        script = self.write_script(ws, SOLVE_OK)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        # The PLAN is on the record, marked as coming from the candidate.
        self.assertEqual(record.method_planned["name"], PLAN["name"])
        self.assertEqual(record.method_planned["source"], "candidate")
        # The ACTUAL method came from the script's stamped receipt.
        self.assertEqual(record.method_actual["name"], PLAN["name"])
        self.assertEqual(len(record.method_actual["steps"]), 4)
        self.assertIn("method_receipt", record.execution_features)
        # The real processing steps are in the trajectory, in order.
        actions = [s.action for s in record.trajectory]
        self.assertTrue(actions[0].startswith("execute:S01"))
        method_steps = [a for a in actions if a.startswith("method:")]
        self.assertEqual(len(method_steps), 4)
        self.assertIn("method:relax the coupling constraint", method_steps)

    def test_an_unstamped_receipt_is_refused(self):
        """A receipt with no matching action_id is a leftover from another
        run and must not be read as this attempt's method."""
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace("ws_stale")
        script = self.write_script(ws, """
            import json
            result = {"status": "optimal", "objective_value": 42.0,
                      "solver": "highs",
                      "method_performed": {"action_id": "not-this-run",
                                           "name": "something else",
                                           "steps": ["x"]}}
            with open("result.json", "w") as fh:
                json.dump(result, fh)
        """)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        # The plan is known; the performance is UNKNOWN, never a copy.
        self.assertEqual(record.method_planned["name"], PLAN["name"])
        self.assertIsNone(record.method_actual)
        self.assertNotIn("method_receipt", record.execution_features)
        self.assertFalse([s for s in record.trajectory
                          if s.action.startswith("method:")])

    def test_a_failure_keeps_the_receipt_and_the_record(self):
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace("ws_fail")
        script = self.write_script(ws, """
            import json, os, sys
            result = {"status": "error",
                      "method_performed": {
                          "action_id": os.environ.get("OR_ACTION_ID"),
                          "name": "rolling-horizon decomposition",
                          "steps": ["relax the coupling constraint",
                                    "solve the master"]}}
            with open("result.json", "w") as fh:
                json.dump(result, fh)
            sys.exit(1)
        """)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        self.assertFalse(record.quality["feasible"])
        self.assertTrue(record.failures)
        # A failed run still reports the method it actually got through.
        self.assertEqual(record.method_actual["steps"],
                         ["relax the coupling constraint", "solve the master"])
        # It survives staging and recording verbatim.
        h.record(record)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.method_actual["steps"],
                         ["relax the coupling constraint", "solve the master"])
        self.assertEqual(stored.method_planned["name"], PLAN["name"])

    def test_a_legacy_executor_still_gets_the_plan(self):
        """An injected executor without ``method_planned`` must not lose the
        plan: the caller attaches it to the record."""
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace("ws_legacy")

        class LegacyExecutor:
            def __init__(self, inner):
                self.inner = inner

            def execute(self, code_path, workspace, *, solver, task_id,
                        strategy_id, profile, code_hash=None, action_id=None):
                # Silently DROPS method_planned, like an executor written
                # before the field existed.
                return self.inner.execute(
                    code_path, workspace, solver=solver, task_id=task_id,
                    strategy_id=strategy_id, profile=profile,
                    code_hash=code_hash, action_id=action_id)

        script = self.write_script(ws, SOLVE_OK)
        h.executor = LegacyExecutor(h.executor)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        self.assertEqual(record.method_planned["name"], PLAN["name"])
        self.assertEqual(record.method_planned["source"], "candidate")


class TestBindMethodObservation(MethodChainCase):
    def test_binding_reports_a_matching_performed_method(self):
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace()
        script = self.write_script(ws, SOLVE_OK)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        h.record(record)
        stored = h.read_prediction(prediction.prediction_id)
        observed = stored["prediction"]["trace"]["model_info"]["method_observed"]
        self.assertIsNotNone(observed)
        self.assertEqual(observed["verdict"], "match")

    def test_binding_reports_a_mismatched_performed_method(self):
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace("ws_mismatch")
        script = self.write_script(ws, """
            import json, os
            result = {"status": "optimal", "objective_value": 42.0,
                      "solver": "highs",
                      "method_performed": {
                          "action_id": os.environ.get("OR_ACTION_ID"),
                          "name": "direct MIP solve",
                          "steps": ["build the full model",
                                    "solve in one shot"]}}
            with open("result.json", "w") as fh:
                json.dump(result, fh)
        """)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        h.record(record)
        stored = h.read_prediction(prediction.prediction_id)
        observed = stored["prediction"]["trace"]["model_info"]["method_observed"]
        self.assertEqual(observed["verdict"], "mismatch")

    def test_no_observed_method_leaves_the_comparison_unknown(self):
        h = self.harness()
        prediction = h.predict_strategy_outcome(TASK, _spec("S01", method=PLAN),
                                                "ep1")
        ws = self.workspace("ws_none")
        script = self.write_script(ws, """
            import json
            json.dump({"status": "optimal", "objective_value": 42.0,
                       "solver": "highs"}, open("result.json", "w"))
        """)
        record = h.execute(TASK, None, str(script), str(ws), solver=None,
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        h.record(record)
        stored = h.read_prediction(prediction.prediction_id)
        observed = stored["prediction"]["trace"]["model_info"]
        self.assertIsNone(observed["method_observed"])


class TestRecordDeclarations(MethodChainCase):
    def test_record_can_declare_a_method_without_execute(self):
        h = self.harness()
        record = self.make_record()
        result = h.record(record, method=PLAN,
                          method_actual={"name": "rolling-horizon decomposition",
                                         "steps": ["solve the master"]})
        stored = h.bank.get(result["execution_id"])
        self.assertEqual(stored.method_planned["name"], PLAN["name"])
        self.assertEqual(stored.method_planned["source"], "harness_declared")
        self.assertEqual(stored.method_actual["source"], "harness_declared")

    def test_a_declaration_never_overwrites_a_reported_performance(self):
        h = self.harness()
        record = self.make_record()
        record.method_actual = normalize_method(
            {"name": "observed method", "steps": ["observed step"]})
        result = h.record(record,
                          method_actual={"name": "declared method",
                                         "steps": ["declared step"]})
        stored = h.bank.get(result["execution_id"])
        self.assertEqual(stored.method_actual["name"], "observed method")


if __name__ == "__main__":
    unittest.main()
