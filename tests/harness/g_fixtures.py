"""Hermetic G fixtures using real execute/check-task/close-episode paths."""
import copy
import json
import os
from pathlib import Path

from tests.harness.helpers import HarnessTestCase
from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.provider import WorldModelProvider
from or_harness.world_model.contracts import FAITHFUL_G_CONVENTION

COUPLED = {"name": "joint factory selection", "steps": [
    "Select the same factory for BE and CFC, preserving the coupling constraint."]}
SPLIT = {"name": "separate factory selection", "steps": [
    "Select a factory separately for BE and CFC then merge the assignments."]}
G_NOTE = "The task requires BE and CFC at one factory; separate selection then merging may produce a cross-factory answer."
G_PAYLOAD = {"benefit": {**FAITHFUL_G_CONVENTION, "value": 0.7,
    "baseline": {"kind": "unknown", "note": "no comparable prior cases"},
    "notes": [G_NOTE]}, "cost": {"solver_runtime_s": 0.1},
    "risk": {"events": [{"event": "implementation_failure", "probability": 0.2}]}}


class GProvider(WorldModelProvider):
    def __init__(self):
        self.payload = copy.deepcopy(G_PAYLOAD)
        self.requests = []
        self.usage = {}

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        return {"payload": copy.deepcopy(self.payload), "usage": self.usage, "latency_s": 0.001}


class GCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider = GProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        self.task = {"task_id": "factory", "family": "assignment",
            "description": "Assign BE and CFC to factories. BE and CFC must use the same factory.",
            "annotations": {"coupling": {"resource_coupling": 0.8}}}

    def predict_g(self, method=COUPLED, episode="ep1", **kwargs):
        return self.h.predict_strategy_outcome(self.task,
            {"action_type": "execute_strategy", "strategy_id": "S04", "method": method},
            episode, **kwargs)

    def run_g(self, prediction, *, split=False, crash=False, timeout=False, actual=None):
        work = Path(self.home) / prediction.prediction_id
        work.mkdir()
        script = work / "solve.py"
        actual = actual or prediction.candidate.method
        body = ("import json, os\n" + ("import time\ntime.sleep(2)\n" if timeout else "") + ("raise RuntimeError('script bug')\n" if crash else
            "be = 1\ncfc = " + ("2" if split else "be") + "\n"
            "result = {'status': 'optimal', 'objective_value': 100, 'objective_bound': 100, "
            "'runtime_seconds': 0.01, 'variables': {'BE': be, 'CFC': cfc}, "
            "'method_performed': " + repr(actual) + "}\n"
            "result['method_performed']['action_id'] = os.environ['OR_ACTION_ID']\n"
            "with open('result.json', 'w') as f: json.dump(result, f)\n"))
        script.write_text(body)
        record = self.h.execute(self.task, "S04", str(script), str(work), solver="highs",
            episode_id=prediction.candidate.episode_id, prediction_id=prediction.prediction_id)
        self.h.record(record)
        return record

    def check_g(self, record, *, fidelity="faithful", failure="unknown",
                coverage="original_task", acceptance="factory-coupling/v1", evidence=True):
        check = {"semantic_probe": [{
            "path": "execution_features.solution_variables.CFC", "equals": 1}]}
        if evidence:
            check["g_attribution"] = {
                "method_fidelity": fidelity, "failure_source": failure,
                "acceptance_scope": coverage, "acceptance_id": acceptance,
                "source": "test auditor reviewed captured script and result",
                "assessment": "Code follows the recorded factory-selection steps; compare the recorded BE=1 and CFC values against the same-factory requirement.",
                "evidence_refs": ["artifacts.submitted_script", "artifacts.workspace_result"]}
        return self.h.check_task_result(record.execution_id, check)

    def close_g(self, episode="ep1"):
        return self.h.close_episode(self.task["task_id"], episode)
