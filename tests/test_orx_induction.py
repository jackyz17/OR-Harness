"""Induction-chain tests for the orx CLI (artifact-as-state, per-cluster dirs).

Covers: cluster discovery over a seeded bank, the align -> induce -> refute ->
validate-pattern -> append-pattern chain, stamp enforcement, and the executor
verdict protocol (crashed refutation is NOT a counterexample).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORX = [sys.executable, str(REPO / "scripts" / "orx.py")]

PROBLEM_A = """A factory assigns 3 jobs to 2 machines.
Each job must go to exactly one machine; machine 1 has capacity 10, machine 2 has 8.
Job sizes: 4, 5, 6. Minimize total load imbalance."""

PROBLEM_B = """A hospital schedules 3 nurses into 2 wards.
Each nurse must staff exactly one ward; ward 1 needs at most 10 patients covered, ward 2 at most 8.
Nurse coverage capacities: 4, 5, 6. Minimize total uncovered patients."""

MODEL_TMPL = """<think>
Assignment structure with shared capacity.
</think>
<model>
SETS:
  j in Jobs = {{j1, j2, j3}}
  m in Machines = {{m1, m2}}
PARAMETERS:
  size[j]
  cap[m]
VARIABLES:
  y[j,m] binary
OBJECTIVE:
  minimize sum(j, sum(m, size[j] * y[j,m]))
CONSTRAINTS:
  C1: sum(m, y[j,m]) = 1
  C2: sum(j, size[j] * y[j,m]) <= cap[m]
</model>"""

SIGNATURE = {
    "objective": "linear",
    "decision": ["binary_assignment"],
    "constraint": ["assignment_exactly_once", "capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "machine_capacity"},
}

SOLVE_CODE = """
import json
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "minimize",
    "objective_value": 15,
    "objective_bound": 15,
    "mip_gap": 0.0,
    "runtime_seconds": 0.01,
    "variables": {{}},
    "diagnostics": {{}},
    "message": "stub",
}}
with open("result.json", "w") as handle:
    json.dump(result, handle)
"""


def run_orx(args, cwd, bank_home):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


def seed_run(bank: Path, tmp: Path, name: str, problem: str, family_note: str) -> None:
    """Drive one full solve run via the CLI to seed the bank with a realization."""
    run_dir = tmp / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "problem.txt").write_text(problem, encoding="utf-8")
    assert run_orx(["recall", "--problem-file", "problem.txt"], run_dir, bank)[0] == 0
    (run_dir / "model.txt").write_text(MODEL_TMPL, encoding="utf-8")
    code, out = run_orx(["validate"], run_dir, bank)
    assert out["passed"], out
    (run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
    code, out = run_orx(["signature"], run_dir, bank)
    assert out["passed"], out
    br = run_dir / "branches" / "highs"
    br.mkdir(parents=True, exist_ok=True)
    (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs"), encoding="utf-8")
    run_orx(["solve", "--solver", "highs"], run_dir, bank)
    run_orx(["gold", "--answer", "15"], run_dir, bank)
    exp = {
        "layer": "modeling",
        "title": family_note,
        "retrieval_text": family_note + ": exactly-once plus per-resource capacity constraints.",
        "modeling_aspect": "constraint",
        "action": "sum(m, y[j,m]) = 1; sum(j, size[j]*y[j,m]) <= cap[m]",
        "problem_family": "assignment",
    }
    exp_file = run_dir / "exp.json"
    exp_file.write_text(json.dumps(exp), encoding="utf-8")
    code, out = run_orx(["append", "--file", str(exp_file)], run_dir, bank)
    assert out["status"] == "appended", out
    code, out = run_orx(["episode"], run_dir, bank)
    assert out["recorded"], out


class TestInductionChain(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_induction_"))
        self.bank = self.tmp / "bank"
        self.work = self.tmp / "work"
        self.work.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def orx(self, *args):
        return run_orx(list(args), self.work, self.bank)

    def test_full_induction_chain(self):
        # Seed two structurally-isomorphic realizations from different families.
        seed_run(self.bank, self.tmp, "runA", PROBLEM_A, "Job-machine allocation capacity")
        seed_run(self.bank, self.tmp, "runB", PROBLEM_B, "Nurse-ward scheduling coverage")

        # clusters: the two realizations share a core_key -> one cluster
        code, out = self.orx("clusters")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(out["clusters"]), 1, out)
        cluster = out["clusters"][0]
        self.assertEqual(cluster["member_count"], 2)
        cluster_id = cluster["cluster_id"]
        member_ids = [m["realization_id"] for m in cluster["members"]]

        # align: first call writes the template
        code, out = self.orx("align", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertFalse(out["aligned"])
        cdir = Path(out["cluster_dir"])

        # fill alignment.json with grounded bindings
        alignment = {
            "bindings": [
                {"role": "competing_decisions", "realization_id": member_ids[0], "entity": "jobs"},
                {"role": "competing_decisions", "realization_id": member_ids[1], "entity": "parcels"},
                {"role": "resource_pool", "realization_id": member_ids[0], "entity": "machines"},
                {"role": "resource_pool", "realization_id": member_ids[1], "entity": "vehicles"},
                {"role": "capacity_limit", "realization_id": member_ids[0], "entity": "machine caps"},
                {"role": "capacity_limit", "realization_id": member_ids[1], "entity": "vehicle fits"},
            ]
        }
        (cdir / "alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
        code, out = self.orx("align", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["aligned"], out)
        self.assertEqual(out["bindings_accepted"], 6)

        # induce: template first, then real hypotheses
        code, out = self.orx("induce", "--cluster", cluster_id)
        self.assertFalse(out["induced"])
        hypotheses = {
            "hypotheses": [
                {
                    "statement": "When competing decisions are assigned to capacitated resources, pair exactly-once assignment constraints with per-resource capacity constraints.",
                    "structural_pattern": "assignment + capacity coupling",
                    "roles_used": ["competing_decisions", "resource_pool", "capacity_limit"],
                    "applicability_conditions": ["resources have hard capacities", "each decision uses exactly one resource"],
                    "complexity": 0.3,
                }
            ]
        }
        (cdir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
        code, out = self.orx("induce", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["induced"], out)
        hyp_id = out["hypotheses"][0]["hypothesis_id"]

        # refute: template first, then a refutation program that survives
        code, out = self.orx("refute", "--cluster", cluster_id)
        self.assertIn("instruction", out)
        refutations = {
            "refutations": [
                {
                    "hypothesis_id": hyp_id,
                    "failure_condition": "uncapacitated resources (capacity = infinity)",
                    "refutation_code": (
                        "import json\n"
                        "# principle still holds structurally even without binding capacities\n"
                        "print(json.dumps({'principle_failed': False, 'evidence': 'assignment structure unchanged'}))\n"
                    ),
                }
            ]
        }
        (cdir / "refutations.json").write_text(json.dumps(refutations), encoding="utf-8")
        code, out = self.orx("refute", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertEqual(out["results"][0]["verdict"], "survived", out)

        # validate-pattern: template first, then real transfer evidence
        code, out = self.orx("validate-pattern", "--cluster", cluster_id)
        self.assertIn("instruction", out)
        validation = {
            "unseen_tasks": ["Nurse shift assignment to wards with hourly coverage caps"],
            "transfer_results": [
                {"hypothesis_id": hyp_id, "task": "nurse assignment", "improved": True,
                 "with_objective": 12.0, "without_objective": 15.0}
            ],
        }
        (cdir / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
        code, out = self.orx("validate-pattern", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["verdicts"][0]["source_consistency"], out)

        # append-pattern: scoring + terminal
        code, out = self.orx("append-pattern", "--cluster", cluster_id)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(out["appended"]), 1, out)
        self.assertEqual(out["status"], "INDUCTION_FLOW_COMPLETE")
        pattern_id = out["appended"][0]["experience_id"]

        # the validated pattern is now retrievable as a prior
        code, out = self.orx("query", "--layer", "modeling", "--query", "assignment capacity")
        self.assertEqual(code, 0, out)
        ids = [h["experience_id"] for h in out["hits"]]
        self.assertIn(pattern_id, ids)

    def test_chain_enforcement(self):
        seed_run(self.bank, self.tmp, "runA", PROBLEM_A, "Job-machine allocation capacity")
        seed_run(self.bank, self.tmp, "runB", PROBLEM_B, "Nurse-ward scheduling coverage")
        code, out = self.orx("clusters")
        cluster_id = out["clusters"][0]["cluster_id"]

        # induce before align -> no cluster directory yet
        code, out = self.orx("induce", "--cluster", cluster_id)
        self.assertEqual(code, 2)
        self.assertIn("align", out["error"])

        # refute before induce -> no cluster dir
        code, out = self.orx("refute", "--cluster", cluster_id)
        self.assertEqual(code, 2)

        # align creates the dir; then stale-stamp check: edit alignment after stamping
        code, out = self.orx("align", "--cluster", cluster_id)
        cdir = Path(out["cluster_dir"])
        member_ids = [m["realization_id"] for m in json.loads(
            (cdir / "cluster.json").read_text(encoding="utf-8"))["members"]]
        alignment = {"bindings": [
            {"role": "resource_pool", "realization_id": member_ids[0], "entity": "machines"},
            {"role": "resource_pool", "realization_id": member_ids[1], "entity": "vehicles"},
        ]}
        (cdir / "alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
        code, out = self.orx("align", "--cluster", cluster_id)
        self.assertTrue(out["aligned"])
        # tamper with alignment.json after stamping
        (cdir / "alignment.json").write_text(json.dumps({"bindings": []}), encoding="utf-8")
        code, out = self.orx("induce", "--cluster", cluster_id)
        self.assertEqual(code, 2)
        self.assertIn("stale", out["error"])

    def test_crashed_refutation_is_not_counterexample(self):
        seed_run(self.bank, self.tmp, "runA", PROBLEM_A, "Job-machine allocation capacity")
        seed_run(self.bank, self.tmp, "runB", PROBLEM_B, "Nurse-ward scheduling coverage")
        code, out = self.orx("clusters")
        cluster_id = out["clusters"][0]["cluster_id"]

        self.orx("align", "--cluster", cluster_id)
        cdir = self.bank / "induction" / cluster_id
        member_ids = [m["realization_id"] for m in json.loads(
            (cdir / "cluster.json").read_text(encoding="utf-8"))["members"]]
        alignment = {"bindings": [
            {"role": "resource_pool", "realization_id": member_ids[0], "entity": "machines"},
            {"role": "resource_pool", "realization_id": member_ids[1], "entity": "vehicles"},
        ]}
        (cdir / "alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
        self.orx("align", "--cluster", cluster_id)

        self.orx("induce", "--cluster", cluster_id)
        hyp_id = "hyp_{}_1".format(cluster_id[:10])
        hypotheses = {"hypotheses": [
            {"statement": "Capacitated assignment pairs exactly-once with capacity constraints.",
             "structural_pattern": "assignment+capacity", "roles_used": ["resource_pool"],
             "applicability_conditions": [], "complexity": 0.3}
        ]}
        (cdir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
        code, out = self.orx("induce", "--cluster", cluster_id)
        self.assertTrue(out["induced"], out)
        hyp_id = out["hypotheses"][0]["hypothesis_id"]

        # crashing refutation program -> crashed_not_counterexample (anti self-judgment)
        refutations = {"refutations": [
            {"hypothesis_id": hyp_id, "failure_condition": "x",
             "refutation_code": "raise RuntimeError('cannot construct')"}
        ]}
        (cdir / "refutations.json").write_text(json.dumps(refutations), encoding="utf-8")
        code, out = self.orx("refute", "--cluster", cluster_id)
        self.assertEqual(out["results"][0]["verdict"], "crashed_not_counterexample", out)


if __name__ == "__main__":
    unittest.main()
