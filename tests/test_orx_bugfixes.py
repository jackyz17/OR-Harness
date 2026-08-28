"""Regression tests for the parallel-execution bug fixes (2026-08-25 round).

Covers:
  1. UtilityTracker concurrent _bump: no FileNotFoundError, no lost updates
     (the utility_stats.tmp race seen in parallel `orx solve`).
  2. LifecycleStore concurrent mark_deprecated: no state-file race.
  3. `orx solve` result record keeps variables/message from the solver output.
  4. `orx gold` / `orx episode` / `orx new-round` on a completed run give
     actionable guidance (fresh-run instruction) instead of a bare refusal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ORX = [sys.executable, str(REPO / "scripts" / "orx.py")]

PROBLEM = """A farmer raises cows and chickens.
Each cow yields 40 profit, each chicken 10 profit.
The farmer has at most 200 units of feed; a cow needs 8 units, a chicken 2.
At most 20 chickens may be raised.
How many of each should be raised to maximize profit?"""

MODEL = """<think>
Resource-allocation LP: two decisions share one feed budget.
</think>
<model>
SETS:
  i in Animals = {cow, chicken}
PARAMETERS:
  profit[i]
  feed_need[i]
  feed_limit
  max_chickens
VARIABLES:
  x[i] integer >= 0
OBJECTIVE:
  maximize sum(i, profit[i] * x[i])
CONSTRAINTS:
  C1: sum(i, feed_need[i] * x[i]) <= feed_limit
  C2: x[chicken] <= max_chickens
</model>"""

SIGNATURE = {
    "objective": "linear",
    "decision": ["integer_batch"],
    "constraint": ["capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "feed"},
}

SOLVE_CODE = """
import json
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "maximize",
    "objective_value": 900,
    "objective_bound": 900,
    "mip_gap": 0.0,
    "runtime_seconds": 0.01,
    "variables": {{"x[cow]": 20, "x[chicken]": 20}},
    "diagnostics": {{"nodes": 3}},
    "message": "stub solve message",
}}
with open("result.json", "w") as handle:
    json.dump(result, handle)
"""


class TestUtilityTrackerRace(unittest.TestCase):
    """Bug #1: concurrent _bump raced on utility_stats.tmp -> FileNotFoundError."""

    def test_concurrent_bumps_no_error_no_lost_updates(self):
        from or_experience_bank.core.utility_tracker import UtilityTracker

        tmp = Path(tempfile.mkdtemp(prefix="orx_race_"))
        tracker = UtilityTracker(tmp)
        errors = []

        def worker(n: int) -> None:
            try:
                for i in range(50):
                    tracker.record_retrieval("exp_{}".format(n % 4))
                    tracker.record_utility("exp_{}".format(n % 4))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], "concurrent bumps raised: {}".format(errors))
        # 8 threads x 50 iterations, spread over 4 ids -> 100 per id per counter.
        for eid in ("exp_0", "exp_1", "exp_2", "exp_3"):
            stats = tracker.stats_for(eid)
            self.assertEqual(stats["retrieval_count"], 100, "lost updates for {}".format(eid))
            self.assertEqual(stats["utility_count"], 100, "lost updates for {}".format(eid))

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class TestLifecycleRace(unittest.TestCase):
    def test_concurrent_mark_deprecated(self):
        from or_experience_bank.core.lifecycle import LifecycleStore

        tmp = Path(tempfile.mkdtemp(prefix="orx_lifecycle_"))
        store = LifecycleStore(tmp)
        errors = []

        def worker(n: int) -> None:
            try:
                store.mark_deprecated(
                    {"experience_id": "exp_{}".format(n), "title": "t", "retrieval_text": "r"},
                    reason="race test",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], "concurrent deprecations raised: {}".format(errors))
        for n in range(8):
            self.assertEqual(store.state_of("exp_{}".format(n)), "deprecated")

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def run_orx(args, cwd, bank_home):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    env["OR_EXPERIENCE_MIN_CV_BRANCHES"] = "2"  # legacy gate for these scenario tests
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=180
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


class TestResultRecordKeepsVariables(unittest.TestCase):
    """Bug #7: the solve record dropped the solver's variables/message payload."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="orx_vars_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        run_orx(["recall", "--problem-file", "problem.txt"], self.run_dir, self.bank)
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        run_orx(["validate"], self.run_dir, self.bank)
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        run_orx(["signature"], self.run_dir, self.bank)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_variables_and_message_preserved(self):
        br = self.run_dir / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs"), encoding="utf-8")
        code, out = run_orx(["solve", "--solver", "highs"], self.run_dir, self.bank)
        self.assertEqual(code, 0, out)
        record = json.loads((br / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["variables"], {"x[cow]": 20, "x[chicken]": 20})
        self.assertEqual(record["message"], "stub solve message")
        self.assertEqual(record["objective_bound"], 900)
        self.assertEqual(record["mip_gap"], 0.0)  # field preserved


class TestCompletedRunGuidance(unittest.TestCase):
    """Bugs #4/#5: completed-run refusals must tell the agent what to do."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="orx_done_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        run_orx(["recall", "--problem-file", "problem.txt"], self.run_dir, self.bank)
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        run_orx(["validate"], self.run_dir, self.bank)
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        run_orx(["signature"], self.run_dir, self.bank)
        for solver in ("highs", "pulp"):
            br = self.run_dir / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver), encoding="utf-8")
        run_orx(["solve", "--solver", "highs,pulp"], self.run_dir, self.bank)
        run_orx(["cross-validate"], self.run_dir, self.bank)
        run_orx(["gold", "--answer", "900"], self.run_dir, self.bank)
        run_orx(["episode"], self.run_dir, self.bank)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_fresh_run_guidance(self, args):
        code, out = run_orx(args, self.run_dir, self.bank)
        self.assertEqual(code, 2, out)
        self.assertIn("FRESH run", out["error"], out)
        self.assertIn("orx recall", out["error"], out)

    def test_gold_on_completed_run_gives_guidance(self):
        self._assert_fresh_run_guidance(["gold", "--answer", "999"])

    def test_episode_on_completed_run_gives_guidance(self):
        self._assert_fresh_run_guidance(["episode"])

    def test_new_round_on_completed_run_gives_guidance(self):
        self._assert_fresh_run_guidance(["new-round"])


if __name__ == "__main__":
    unittest.main()
