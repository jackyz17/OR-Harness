"""Append-only raw attempt trajectories, separate from reusable experience."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict

from ..core.schemas import AttemptRecord, BranchState

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


class TrajectoryStore:
    def __init__(self, bank_home: Path):
        self.root = Path(bank_home) / "trajectories"
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    def problem_dir(self, problem_id: str) -> Path:
        path = self.root / problem_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def initialize_problem(self, problem_id: str, problem: Dict) -> Path:
        path = self.problem_dir(problem_id) / "problem.json"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(problem, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        return path

    def branch_dir(self, problem_id: str, branch_id: str) -> Path:
        path = self.problem_dir(problem_id) / branch_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_attempt(self, record: AttemptRecord) -> Path:
        path = self.branch_dir(record.problem_id, record.branch_id) / "attempts.jsonl"
        line = (json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        with self._thread_lock:
            with path.open("ab", buffering=0) as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return path


class BranchStateSummarizer:
    """Keep only latest state and compact repair history for the next attempt."""

    def summarize(self, state: BranchState, max_chars: int = 8000) -> str:
        sections = [
            "Current formulation: " + state.current_formulation,
            "Current code:\n" + state.current_code[-5000:],
            "Latest feedback: " + state.latest_feedback[-1200:],
            "Resolved issues: " + "; ".join(state.resolved_issues[-5:]),
            "Unresolved issues: " + "; ".join(state.unresolved_issues[-5:]),
            "Ineffective repairs: " + "; ".join(state.ineffective_repairs[-5:]),
        ]
        return "\n".join(sections)[:max_chars]

