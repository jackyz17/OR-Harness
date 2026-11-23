"""Real CLI workflow with an explicitly scripted, local world-model provider.

PYTHONPATH=src python3 references/examples/unified_solving.py --mode single
PYTHONPATH=src python3 references/examples/unified_solving.py --mode multi

Each run uses a new isolated home, scores before execution, checks the original
integer task, records the attempt and settles its frozen H+ job. The second task
uses the published rule with H+=0. This verifies wiring, not predictor quality.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHOD = {"name": "lower endpoint enumeration", "steps": [
    "Enumerate nonnegative integer Y; fixed Y has 2Y+130 <= X <= 641-Y.",
    "For this minimization choose the lower feasible integer X and compare all Y.",
    "Check integrality, both original inequalities and objective 2X-5Y."]}
REBUILD = {"name": "full integer enumeration", "steps": [
    "Rebuild the feasible set by enumerating every nonnegative integer X and Y.",
    "Check X+Y<=641 and X>=2Y+130, then compare 2X-5Y."]}


def run(mode="single", home=None):
    home = Path(home or tempfile.mkdtemp(prefix="orx-unified-"))
    home.mkdir(parents=True, exist_ok=True)
    work = home / "ws"
    work.mkdir(exist_ok=True)
    requests, commands = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            envelope = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request = json.loads(envelope["messages"][-1]["content"])
            requests.append(request)
            candidate = request["candidate"]
            reuse = candidate["method"]["name"] == METHOD["name"]
            covered = "covered" in json.dumps(request.get("prediction_context", {}).get("problem", request))
            payload = {"benefit": {
                **{k: request["benefit_convention"][k] for k in ("kind", "metric", "unit")},
                "value": .9, "baseline": {"kind": "unknown", "note": "Scripted wiring example."},
                "notes": ["The objective increases with X at fixed Y, so the lower feasible endpoint preserves each Y's minimum."]},
                "cost": {"tool_calls": 3 if reuse else 30},
                "risk": {"events": [{"event":"implementation_failure", "probability": .1}]},
                "capability_gain": {"decision": 0 if covered else 1,
                    "claim": "The lower endpoint complements a maximum-only endpoint rule.",
                    "basis": ["Compare the captured endpoint solution with a full enumeration."],
                    "verification_conditions": [{"condition": "Full enumeration gives the same objective 90."}]}}
            data = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}::scripted-example"

    def cli(*args):
        args = list(map(str, args))
        argv = [sys.executable, "-m", "or_harness.cli", "--home", str(home), *args]
        commands.append(shlex.join(argv))
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OR_EMBEDDING_BACKEND": "local-hashing"}
        proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"{commands[-1]}\n{proc.stdout}\n{proc.stderr}")
        output = json.loads(proc.stdout)
        return output["result"], output["summary"]

    script = work / "solve.py"
    script.write_text("import json,os\n"
        "rows=[(2*(2*y+130)-5*y,2*y+130,y) for y in range(171)]\n"
        "best=min(rows)\n"
        "full=min((2*x-5*y,x,y) for y in range(642) for x in range(642-y) if x>=2*y+130)\n"
        "assert best==full and best[0]==90\n"
        "r={'status':'optimal','objective_value':best[0],'objective_bound':best[0],"
        "'variables':{'X':best[1],'Y':best[2]},'runtime_seconds':0.01,'method_performed':"+repr(METHOD)+"}\n"
        "r['method_performed']['action_id']=os.environ['OR_ACTION_ID']\n"
        "json.dump(r,open('result.json','w'))\n")
    decisions, executions, closes, summaries = [], [], [], []
    try:
        for index, task_id in enumerate(("endpoint-source", "endpoint-covered")):
            task = {"task_id": task_id, "family": "assignment", "text":
                "Minimize 2X-5Y for nonnegative integers X,Y subject to X+Y<=641 and X>=2Y+130."}
            task_file = work / f"{task_id}.json"
            task_file.write_text(json.dumps(task))
            candidates = [{"strategy_id": "endpoint", "solver": "python", "method": METHOD}]
            if mode == "multi":
                candidates.append({"strategy_id": "full-enumeration", "solver": "python", "method": REBUILD})
            candidate_file = work / f"{task_id}-candidates.json"
            candidate_file.write_text(json.dumps(candidates))
            cli("profile", "--task", task_file)
            cli("recall", "--task", task_file)
            planned, summary = cli("--world-model", endpoint, "plan-next", "--task", task_file,
                "--episode", "E", "--candidates", candidate_file, "--cost-scales", '{"tool_calls":30}')
            pid = planned["selected_prediction_id"]
            assert pid and planned["selected_score"]["utility"] is not None, planned
            decisions.append(planned)
            summaries.append(summary)
            extra = ["--used-entry-ids", entry_ref] if index else []
            executed, summary = cli("execute", "--task", task_file, "--prediction", pid,
                "--code", script, "--workspace", work, *extra)
            ex = executed["execution_id"]
            actual_y = executed["execution"]["execution_features"]["solution_variables"]["Y"]
            executions.append(ex)
            summaries.append(summary)
            check = {"reference_objective": 90, "reference_source": "independent",
                "reference_version": "full-enumeration/1", "integer": {"variables": ["X", "Y"]},
                "recompute_objective": {"coefficients": {"X": 2, "Y": -5}},
                "semantic_probe": [
                    {"path": "execution_features.solution_variables.X",
                     "min": 2*actual_y+130, "max": 641-actual_y},
                    {"path": "execution_features.solution_variables.Y", "min": 0, "max": 170}],
                "hplus_verification": [{"prediction_id": pid, "condition_index":0, "state":"checked",
                    "finding":"Captured script asserts equality with full enumeration and objective 90.",
                    "source":"Example reviewer inspected the captured script and independent checker.",
                    "evidence_refs":["artifacts.submitted_script","task_check.checks"]}],
                "g_attribution": {"method_fidelity": "faithful", "failure_source": "unknown",
                    "acceptance_scope": "original_task", "acceptance_id": "endpoint-original/1",
                    "source": "Example reviewer inspected the captured script and its exhaustive feasibility comparison.",
                    "assessment": "The actual script enumerates the original constraints and confirms the minimum independently.",
                    "evidence_refs": ["artifacts.submitted_script", "artifacts.workspace_result"]}}
            checked, _ = cli("check-task", ex, "--check", json.dumps(check))
            assert checked["state"] == "passed", checked
            cli("record", "--from-staged", ex)
            closed, summary = cli("close-episode", "--task", task_id, "--episode", "E", "--terminal", "completed")
            summaries.append(summary)
            closes.append(closed)
            jobs = closed["induction_schedule"]["pending_ids"]
            if index:
                assert jobs == [], jobs
                continue
            assert len(jobs) == 1, jobs
            job = jobs[0]
            cli("induction-material", "--induction-id", job)
            submission = {"result": "new_rule",
                "rule": {"when": "Y can be finitely enumerated and each fixed-Y feasible integer X is a finite interval with monotone objective.",
                    "do": "Skip empty intervals; choose the lower X endpoint for increasing minimization and the opposite endpoint for maximization; enumerate X for nonmonotone objectives.",
                    "expect": "Preserve the optimum within this scope while reducing X enumeration."},
                "basis": {"kind": "derivation", "refs": [ex],
                    "analysis": "Coefficient 2 makes the fixed-Y objective increasing; the captured enumeration verifies objective 90."},
                "publication": {"state": "published", "reason": "Monotonicity derivation and captured full enumeration agree.",
                    "scope": ["finite integer intervals with fixed-Y monotone objective"]}}
            result, _ = cli("induce", "--induction-id", job, "--submission", json.dumps(submission))
            receipt = result["induction_receipt"]
            assert receipt["prediction_id"] == pid
            saved = result["relations"][0]
            entry_ref = f"{saved['saved']}@{saved['version']}"
            repeated, _ = cli("close-episode", "--task", task_id, "--episode", "E")
            assert repeated["induction_schedule"]["pending_ids"] == []
        assert len(requests) == 2 * (1 if mode == "single" else 2)
        status, _ = cli("induction-status")
        assert status["completed"], status
        return {"mode": mode, "home": str(home), "provider_kind": "scripted wiring fixture",
            "provider_calls": len(requests), "decisions": decisions, "execution_ids": executions,
            "induction_receipt": receipt, "induction_status": status, "commands": commands, "summaries": summaries,
            "request_fragment": {k: requests[0][k] for k in ("candidate", "benefit_convention", "prediction_context")}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--home", help="new isolated memory home")
    parser.add_argument("--output", help="optional complete audit JSON")
    args = parser.parse_args()
    result = run(args.mode, args.home)
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("mode", "home", "provider_calls", "execution_ids", "induction_receipt")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
