"""Isolated executable G demonstration; no remote provider or solver package.

PYTHONPATH=src python references/examples/g_signal.py > /tmp/g-request.json
The JSON is an excerpt of a request actually delivered to the stub provider.
"""
import copy
import json
import tempfile
from pathlib import Path

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.contracts import FAITHFUL_G_CONVENTION
from or_harness.world_model.provider import WorldModelProvider


class Provider(WorldModelProvider):
    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        steps = " ".join(request["candidate"]["method"]["steps"])
        reasoning = ("原题要求 BE 和 CFC 同厂，候选分别选厂后直接合并，预计可能产生跨厂解。"
            if "separately" in steps else
            "原题要求 BE 和 CFC 同厂，候选共同选厂并保留该约束，预计能满足同厂验收。")
        return {"payload": {
            "benefit": {**FAITHFUL_G_CONVENTION, "value": 0.7,
                "baseline": {"kind": "unknown", "note": "no comparable prior cases"},
                "notes": [reasoning]},
            "cost": {"solver_runtime_s": 0.1},
            "risk": {"events": [{"event": "implementation_failure", "probability": 0.2}]}},
            "usage": {}, "latency_s": 0.001}


def main():
    provider = Provider()
    task = {"task_id": "factory", "family": "assignment",
        "description": "Assign BE and CFC to factories; both must use the same factory.",
        "annotations": {"coupling": {"resource_coupling": 0.8}}}
    joint = {"name": "joint selection", "steps": ["Choose a common factory for BE and CFC."]}
    split = {"name": "separate selection", "steps": ["Select BE and CFC factories separately, then merge."]}
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="or-harness-g-demo-") as home:
        h = ORHarness(home=home, world_model=provider, embedding=LocalHashEmbeddingBackend())
        try:
            for ep, method, cfc, fidelity, failure in [
                ("method_omission", split, 2, "faithful", "candidate_method"),
                ("faithful_pass", joint, 1, "faithful", "unknown"),
                ("code_omission", joint, 2, "deviated", "implementation"),
            ]:
                p = h.predict_strategy_outcome(task, {"action_type": "execute_strategy",
                    "strategy_id": "S04", "method": method}, ep)
                work = Path(home) / ep
                work.mkdir()
                script = work / "solve.py"
                script.write_text("import json, os\nbe=1\ncfc=" + str(cfc) + "\n"
                    "result={'status':'optimal','objective_value':100,'objective_bound':100,"
                    "'runtime_seconds':0.01,'variables':{'BE':be,'CFC':cfc},'method_performed':"
                    + repr(method) + "}\n"
                    "result['method_performed']['action_id']=os.environ['OR_ACTION_ID']\n"
                    "result['config']={'action_id':os.environ['OR_ACTION_ID'],'time_limit':60}\n"
                    "with open('result.json','w') as f: json.dump(result,f)\n")
                record = h.execute(task, "S04", str(script), str(work), solver="highs",
                    episode_id=ep, prediction_id=p.prediction_id)
                h.record(record)
                h.check_task_result(record.execution_id, {
                    "semantic_probe": [{"path": "execution_features.solution_variables.CFC", "equals": 1}],
                    "g_attribution": {"method_fidelity": fidelity, "failure_source": failure,
                        "acceptance_scope": "original_task", "acceptance_id": "same-factory/v1",
                        "source": "example auditor inspected the captured script and BE/CFC result",
                        "assessment": "BE is fixed at 1; compare CFC against the original same-factory constraint and audit whether the code preserves the declared method.",
                        "evidence_refs": ["artifacts.submitted_script", "artifacts.workspace_result"]}})
                closed = h.close_episode(task["task_id"], ep)
                benefit = closed["evaluations"][0]["benefit"]
                outcomes.append({"episode": ep, "G_observed": benefit.get("observed"),
                    "label_eligibility": benefit["label_eligibility"], "reason": benefit.get("reason")})
            h.predict_strategy_outcome(task, {"action_type": "execute_strategy",
                "strategy_id": "S04", "method": joint}, "next")
            actual = provider.requests[-1]
            context = actual["prediction_context"]
            excerpt = {key: actual[key] for key in (
                "prediction_protocol", "request_kind", "candidate", "benefit_convention")}
            excerpt["prediction_context"] = {key: context[key] for key in (
                "task_id", "joint_problem", "prediction_reminders", "prediction_execution_pairs")}
            calibration = context["strategy_outcome_calibration"]
            excerpt["prediction_context"]["strategy_outcome_calibration"] = {
                key: calibration[key] for key in ("conditional_g", "reminders")}
            print(json.dumps({"capture": "excerpt of the actual next request sent to the stub; omitted fields are unchanged request fields",
                "constructed_outcomes": outcomes, "request_excerpt": excerpt}, ensure_ascii=False, indent=2))
        finally:
            h.close()


if __name__ == "__main__":
    main()
