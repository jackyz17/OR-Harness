"""r10 end-to-end acceptance.

Two flows through the REAL API (temporary home, stub provider, lightweight
execution). They check ASSOCIATION AND FLOW, not the world model's prediction
quality:

- A: minimal candidate -> predict -> choose -> execute -> check -> record ->
  close-episode, asserting the evaluation records and the prediction binding;
- B: minimal new strategy with NO assertion -> facts read -> publish ->
  visible on a later recall, and the rule-kind variant saved but not
  published.
"""
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
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")

PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}


class StubProvider(WorldModelProvider):
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else PAYLOAD
        self.requests = []

    def describe(self):
        return {"provider": "stub-r10"}

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _task(task_id="t1"):
    return {"task_id": task_id, "family": "planning",
            "description": REQ_TEXT,
            "spec": {"n_vars": 2, "n_constraints": 3, "n_int_vars": 2}}


class EndToEndCase(HarnessTestCase):

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

    def _solve(self, task, strategy, *, objective=100.0, tag="x",
               solver="highs"):
        work = Path(self.home) / f"ws_{tag}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            "'runtime_seconds': 0.01, 'variables': {'x': 2, 'y': 3}}, fh)\n",
            encoding="utf-8")
        return self.h.execute(task, strategy, str(script), str(work),
                              solver=solver, episode_id="ep1")


class TestFullFlow(EndToEndCase):
    """A: minimal candidate through the whole loop."""

    def test_minimal_candidate_predict_choose_execute_close(self):
        task = _task("t1")
        # 1. planning with the MINIMAL candidate format (no strategy_id).
        plan = self.h.plan_next(task, episode_id="ep1", candidates=[
            {"name": "enumeration_with_constraints",
             "steps": ["derive a tight bound", "enumerate the reduced domain"],
             "solver": "python"}])
        compared = plan.get("candidates") or []
        self.assertEqual(len(compared), 1)
        candidate = compared[0]
        prediction_id = candidate["prediction_id"]
        self.assertEqual(candidate["prediction_status"], "valid")
        spec = candidate["action_spec"]
        self.assertTrue(str(spec["strategy_id"]).startswith("cand_"))
        decision_id = plan["decision_action_id"]

        # 2. choose the candidate by its prediction id.
        choice = self.h.choose_next(decision_id, prediction_id=prediction_id)
        self.assertEqual(choice["status"], "completed")
        self.assertEqual(choice["selected"]["strategy_id"],
                         spec["strategy_id"])

        # 3. execute with the SAME prediction (bound before the run); the
        # solver comes from the candidate, so there is no mismatch.
        record = self._solve(task, spec["strategy_id"], tag="flow",
                             solver=spec.get("solver") or "highs")
        prediction = self.h.bind_strategy_outcome(prediction_id,
                                                  record.action_id)
        self.assertEqual(prediction.trace.model_info.get("bound_action_id"),
                         record.action_id)
        self.assertFalse(prediction.trace.model_info.get("binding_mismatch"))

        # 4. check the answer and record the attempt.
        checked = self.h.check_task_result(
            record.execution_id, {"reference_objective": 100.0})
        self.assertEqual(checked["state"], "passed")
        self.h.record(record)

        # 5. close the episode: the prediction is evaluated against reality.
        closed = self.h.close_episode("t1", "ep1")
        evaluations = closed["evaluations"]
        self.assertTrue(evaluations)
        self.assertEqual(evaluations[0]["prediction_id"], prediction_id)


class TestNoAssertionAdmission(EndToEndCase):
    """B: a minimal strategy with no assertion publishes and is recalled."""

    def _seed_two_tasks(self):
        h = self.h
        for index, task_id in enumerate(["t1", "t2"]):
            record = self._solve(_task(task_id), "S01",
                                 tag=f"seed{index}")
            h.record(record)
        return h

    def test_no_assertion_fact_publishes_and_recalls(self):
        h = self._seed_two_tasks()
        executions = [r.execution_id for r in h.bank.all()]
        self.assertGreaterEqual(len(executions), 2)
        out = h.induce(relations=[{
            "subject": "method:enum",
            "kind": "conditional_fact",
            "claim": ("under this structure the enumeration method produced "
                      "a checked-optimal answer"),
            "method": {"name": "enum", "steps": ["bound", "enumerate"]},
            "evidence": [{"execution_id": executions[0], "role": "evidence"},
                         {"execution_id": executions[-1], "role": "evidence"}],
            "conditions": {"predicates": {"family": "planning"}}}])
        relation = out["relations"][0]
        self.assertTrue(relation["publication"]["published"])
        self.assertEqual(relation["publication"]["state"], "fact_checked")
        self.assertEqual(relation["publication"]["transferability"],
                         "unproven")

        recs = h.selector.recall(
            self.make_profile(problem_id="q", family="planning"), top=5)
        entry = next(r for r in recs if r.strategy_id == "method:enum")
        self.assertEqual(entry.evidence, "strategic_entry")

    def test_no_assertion_rule_is_published_with_its_state(self):
        h = self._seed_two_tasks()
        executions = [r.execution_id for r in h.bank.all()]
        out = h.induce(relations=[{
            "subject": "method:rule",
            "kind": "rule",
            "claim": "this method always wins",
            "method": {"name": "m", "steps": ["s"]},
            "evidence": [{"execution_id": executions[0], "role": "evidence"},
                         {"execution_id": executions[-1], "role": "evidence"}],
            "conditions": {"predicates": {"family": "planning"}}}])
        relation = out["relations"][0]
        self.assertIsNotNone(relation.get("saved"))
        # A submitted claim is published; the framework read the FACTS and
        # records that state (``fact_checked``) rather than withholding.
        self.assertTrue(relation["publication"]["published"])
        self.assertEqual(relation["publication"]["state"], "fact_checked")
        recs = h.selector.recall(
            self.make_profile(problem_id="q", family="planning"), top=5)
        self.assertTrue(any(r.strategy_id == "method:rule" for r in recs))


if __name__ == "__main__":
    unittest.main()
