"""The runtime half of the documented-result-key contract.

The Skill tells an agent exactly which fields to read from a command's JSON
line. Those paths drift: a rename in the CLI leaves the docs citing a key
that no longer exists, and the agent reads ``None``. Nothing else in the
suite catches that, because a renamed key is not a crash.

This test runs each ONLINE command through the real CLI (harness injected,
stub provider, temp home) and asserts:

1. every key the docs cite for that command, and which
   ``_result_keys.py`` marks REQUIRED, really appears;
2. a key marked CONDITIONAL is never ``None`` when present (the docs say it
   is emitted only under a stated condition, so "present but null" would make
   the note false);
3. the map itself stays honest: every command it names exists, and every
   command the Skill cites result keys for is covered, so the static check
   cannot silently skip one.
"""
import io
import json
import os
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "references" / "examples"))
from _result_keys import DOCUMENTED_RESULT_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

TASK_TEXT = ("Ship 100 whole pallets from the depot to the store before the "
             "window closes.")

TASK = {"task_id": "t_docs", "family": "routing", "text": TASK_TEXT,
        "spec": {"n_vars": 6, "n_constraints": 4, "n_int_vars": 6}}

STUB_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "no_feasible_solution",
                         "probability": 0.2}]},
    "capability_gain": {
        "claim": "reusable warm start",
        "expected_changes": [
            {"metric": "solver_runtime_s", "direction": "decrease",
             "value": 20, "unit": "seconds", "value_kind": "relative",
             "beneficial_direction": "decrease",
             "baseline": {"kind": "conditional_stats", "value": 30}}],
        "verification_conditions": [{"condition": "a later runtime drop"}]},
}

SOLVE = """import json, os
result = {"status": "optimal", "objective_value": 42.0,
          "objective_bound": 42.0, "runtime_seconds": 0.01,
          "variables": {"x1": 3.0}}
aid = os.environ.get("OR_ACTION_ID")
if aid:
    result["method_performed"] = {
        "action_id": aid, "name": "direct MILP",
        "steps": ["write the MILP", "solve it"]}
with open("result.json", "w") as fh:
    json.dump(result, fh)
"""

CHECK = {"integer": {"variables": ["x1"]}}


class StubProvider(WorldModelProvider):
    name = "docs-keys-stub"

    def __init__(self, payload=None):
        self.payload = payload or STUB_PAYLOAD
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 900, "completion_tokens": 100},
                "error": None, "latency_s": 0.01}


class DocumentedKeysCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        self.workspace = Path(self.home) / "ws"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.script = self.workspace / "solve.py"
        self.script.write_text(SOLVE, encoding="utf-8")
        self.task_path = Path(self.home) / "task.json"
        self.task_path.write_text(json.dumps(TASK), encoding="utf-8")
        self.cand_path = Path(self.home) / "cand.json"
        self.cand_path.write_text(json.dumps(self._candidate()), encoding="utf-8")

    def _candidate(self, strategy_id="S01", **extra):
        return {"action_type": "execute_strategy", "task_id": "t_docs",
                "episode_id": "ep1", "strategy_id": strategy_id,
                "solver": "highs",
                "method": {"name": "direct MILP",
                           "steps": ["write the MILP", "solve it"]},
                **extra}

    def run_cli(self, *argv):
        """Run one command IN-PROCESS and return its parsed ``result``.

        The harness is injected so the command uses this test's temp home and
        stub provider rather than opening a second store — the same mechanism
        the other CLI-level tests use.
        """
        from or_harness import cli as cli_module
        original = cli_module._harness
        cli_module._harness = lambda args: self.h
        self.addCleanup(setattr, cli_module, "_harness", original)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli_module.main(["--home", self.home, *argv])
        output = buffer.getvalue().strip()
        self.assertNotEqual(code, 1,
                            f"`orx {' '.join(argv)}` crashed: {output[:400]}")
        self.assertEqual(code, 0,
                         f"`orx {' '.join(argv)}` exited {code}: {output[:400]}")
        return json.loads(output.splitlines()[-1])["result"]

    def assert_required(self, command, result):
        """Every REQUIRED declared key is present; a CONDITIONAL one present
        is not null."""
        entry = DOCUMENTED_RESULT_KEYS[command]
        missing = sorted(k for k in entry["required"] if k not in result)
        self.assertEqual(
            missing, [],
            f"`orx {command}` does not emit the required documented key(s) "
            f"{missing}. Either the command regressed or the docs (and "
            "_result_keys.py) are stale.")
        for key in sorted(entry["conditional"] & set(result)):
            self.assertIsNotNone(
                result[key],
                f"`orx {command}` emitted the conditional key {key!r} as "
                "None; the docs say it appears only when it has a value")


class TestDocumentedResultKeys(DocumentedKeysCase):

    def test_profile(self):
        self.assert_required("profile", self.run_cli(
            "profile", "--task", str(self.task_path)))

    def test_recall(self):
        self.assert_required("recall", self.run_cli(
            "recall", "--task", str(self.task_path), "--top", "3"))

    def test_predict_strategy(self):
        self.assert_required("predict-strategy", self.run_cli(
            "predict-strategy", "--task", str(self.task_path),
            "--candidate", str(self.cand_path), "--episode", "ep1"))

    def test_plan_next(self):
        self.assert_required("plan-next", self.run_cli(
            "plan-next", "--task", str(self.task_path), "--episode", "ep1",
            "--candidates", self._candidate_list()))

    def test_choose_next(self):
        plan = self.run_cli(
            "plan-next", "--task", str(self.task_path), "--episode", "ep1",
            "--candidates", self._candidate_list())
        chosen = plan["plan"]["candidates"][0]["prediction_id"]
        self.assert_required("choose-next", self.run_cli(
            "choose-next", "--decision", plan["decision_action_id"],
            "--prediction", chosen))

    def _candidate_list(self):
        path = Path(self.home) / "cands.json"
        path.write_text(json.dumps([self._candidate(),
                                    self._candidate("S02")]),
                        encoding="utf-8")
        return str(path)

    def _executed(self):
        prediction = self.run_cli(
            "predict-strategy", "--task", str(self.task_path),
            "--candidate", str(self.cand_path), "--episode", "ep1")
        executed = self.run_cli(
            "execute", "--task", str(self.task_path), "--episode", "ep1",
            "--prediction", prediction["prediction_id"],
            "--code", str(self.script), "--workspace", str(self.workspace))
        return executed["execution_id"]

    def test_execute(self):
        self.assertTrue(self._executed().startswith("ex_"))

    def test_check_task(self):
        execution_id = self._executed()
        self.assert_required("check-task", self.run_cli(
            "check-task", execution_id, "--check", json.dumps(CHECK)))

    def test_record(self):
        execution_id = self._executed()
        self.assert_required("record", self.run_cli(
            "record", "--from-staged", execution_id))

    def test_close_episode(self):
        execution_id = self._executed()
        self.run_cli("check-task", execution_id, "--check", json.dumps(CHECK))
        self.run_cli("record", "--from-staged", execution_id)
        self.assert_required("close-episode", self.run_cli(
            "close-episode", "--task", "t_docs", "--episode", "ep1",
            "--terminal", "completed"))

    def test_inspect_experience(self):
        self.assert_required("inspect", self.run_cli(
            "inspect", "--bank", "experience", "--task", "t_docs"))

    def test_inspect_capability(self):
        # The online-gain view is always present, even with no gain claimed.
        self.assert_required("inspect", self.run_cli(
            "inspect", "--bank", "capability"))

    def test_induction_candidates(self):
        self.assert_required("induction-candidates",
                             self.run_cli("induction-candidates"))


class TestTheMapStaysHonest(unittest.TestCase):
    """Guards the map against drifting from the CLI and from the Skill."""

    def test_every_declared_command_exists(self):
        from or_harness.cli import build_parser
        real = set(build_parser()._subparsers._group_actions[0].choices)
        unknown = sorted(set(DOCUMENTED_RESULT_KEYS) - real)
        self.assertEqual(unknown, [],
                         f"_result_keys.py names unknown command(s) {unknown}")

    def test_skill_table_commands_are_all_covered(self):
        """Every command the Skill cites a result key for is in the map, so
        the static check cannot silently skip one."""
        skill = (REPO / "SKILL.md").read_text(encoding="utf-8")
        row = re.compile(r"^\|\s*[^|]+\|\s*`orx\s+([a-z][a-z0-9-]+)[^`]*`\s*"
                         r"\|[^|]*\|([^|]*)\|", re.MULTILINE)
        uncovered = set()
        for match in row.finditer(skill):
            command, column = match.group(1), match.group(2)
            if "result." not in column:
                continue
            if command not in DOCUMENTED_RESULT_KEYS:
                uncovered.add(command)
        self.assertEqual(
            sorted(uncovered), [],
            f"SKILL.md cites result keys for {sorted(uncovered)}, which "
            "_result_keys.py does not cover — add them so they can be "
            "guarded")

    def test_required_keys_are_a_subset_of_declared(self):
        for command, entry in DOCUMENTED_RESULT_KEYS.items():
            self.assertFalse(
                entry["required"] & entry["conditional"],
                f"{command}: a key cannot be both required and conditional")


if __name__ == "__main__":
    unittest.main()
