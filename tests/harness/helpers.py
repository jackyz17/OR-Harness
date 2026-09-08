"""Shared helpers for or_harness tests."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from or_harness.core.schema import (  # noqa: E402
    CostVector,
    ExecutionRecord,
    FailureRecord,
    ProblemProfile,
)
from or_harness.core.storage import Store  # noqa: E402


class HarnessTestCase(unittest.TestCase):
    """Base class providing a temporary memory home + store."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = self._tmp.name
        self.store = Store(self.home)
        self.addCleanup(self.store.close)

    def make_profile(self, problem_id="t1", family="routing", **coupling) -> ProblemProfile:
        values = {"semantic_coupling": 0.8, "resource_coupling": 0.9,
                  "temporal_coupling": 0.1, "route_complexity": 0.85}
        values.update(coupling)
        return ProblemProfile(
            problem_id=problem_id,
            family=family,
            scale_features={"n_vars": 1000.0, "n_constraints": 500.0,
                            "n_int_vars": 1000.0, "density": 0.01},
            **values,
        )

    def make_record(self, execution_id=None, task_id="t1", strategy_id="S01",
                    profile=None, feasible=True, objective=100.0, gap=0.0,
                    status="optimal", cost=None, failures=None, solver=None,
                    source="executed", created_at=None) -> ExecutionRecord:
        rec = ExecutionRecord(
            execution_id=execution_id or ExecutionRecord.new_id(),
            task_id=task_id,
            strategy_id=strategy_id,
            profile_snapshot=profile or self.make_profile(problem_id=task_id),
            quality={"feasible": feasible, "objective": objective,
                     "gap": gap, "status": status},
            cost=cost or CostVector(llm_tokens=100, tool_calls=2,
                                    solver_runtime_s=1.0, retries=0, latency_s=1.5),
            failures=failures or [],
            solver=solver or {"name": "highs", "family": "milp", "code_hash": "abc123"},
            source=source,
        )
        if created_at is not None:
            rec.created_at = created_at
        return rec
