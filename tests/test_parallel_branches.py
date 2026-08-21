from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from helpers import build_system
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter


class ParallelBranchTests(unittest.TestCase):
    def test_two_branches_run_concurrently_and_isolate_workspaces(self):
        with tempfile.TemporaryDirectory() as temp:
            adapters = [
                MockSolverAdapter("mock-a", [SolverExecutionResult(status="optimal", solver="mock-a", exit_code=0, objective_sense="minimize", objective_value=2, variables={})], delay=0.12),
                MockSolverAdapter("mock-b", [SolverExecutionResult(status="optimal", solver="mock-b", exit_code=0, objective_sense="minimize", objective_value=2, variables={})], delay=0.12),
            ]
            orchestrator, _, _, _ = build_system(temp, adapters, ["# branch a", "# branch b"])
            started = time.monotonic()
            result = asyncio.run(orchestrator.solve("Assign jobs and minimize cost", ["mock-a", "mock-b"], auto_append=False))
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.23)
            workspaces = [branch.workspace for branch in result.branches]
            self.assertEqual(len(set(workspaces)), 2)
            for branch in result.branches:
                trajectory = Path(temp) / "trajectories" / result.problem_id / branch.branch_id / "attempts.jsonl"
                self.assertTrue(trajectory.exists())
                self.assertNotIn(next(b.branch_id for b in result.branches if b.branch_id != branch.branch_id), trajectory.read_text())
            events = [item["event"] for item in result.timeline]
            self.assertGreater(events.index("all_branches_finished"), max(i for i, value in enumerate(events) if value == "branch_finished"))
            self.assertGreater(events.index("experience_extracted"), events.index("all_branches_finished"))

