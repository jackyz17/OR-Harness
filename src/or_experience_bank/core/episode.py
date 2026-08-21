"""Episode store: problem-level scene snapshots (Phase 2 step 6, D8).

An Episode is the problem-level narrative record of one solve run — distinct from the
attempt-level trajectory evidence (trajectory.py). It carries the structural signature,
the verified model, per-branch outcome summaries, the gold answer and verdict, and the
produced realization ids. It is the raw material for offline induction and the provenance
target of realizations (evidence.source_episodes).

Two-phase append (Option A): solve() records a base Episode when solving finishes (gold
not yet known); evaluate_with_gold() appends a supplement record once gold arrives. Both
lines are append-only — the supplement links back via problem_id, never edits the base.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .modeling_schemas import BranchSummary, EpisodeRecord, StructuralSignature
from .schemas import utc_now

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


class EpisodeStore:
    """Append-only store of problem-level Episodes at bank_home/episodes/."""

    def __init__(self, bank_home: Path):
        self.root = Path(bank_home) / "episodes"
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "episodes.jsonl"
        self._path.touch(exist_ok=True)
        self._lock = threading.RLock()

    def _append_line(self, payload: Dict[str, Any]) -> None:
        line = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            with self._path.open("ab", buffering=0) as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def record_episode(self, episode: EpisodeRecord) -> str:
        """Phase (a): record the base Episode right after solving (gold pending)."""
        payload = episode.to_dict()
        payload["record_kind"] = "base"
        self._append_line(payload)
        return episode.episode_id

    def record_gold_supplement(
        self,
        problem_id: str,
        gold: Optional[float],
        matched: bool,
        produced_realization_ids: Optional[List[str]] = None,
    ) -> None:
        """Phase (b): append a supplement once gold arrives. Never edits the base line."""
        payload = {
            "record_kind": "gold_supplement",
            "problem_id": problem_id,
            "gold_answer": gold,
            "matched": matched,
            "produced_realization_ids": produced_realization_ids or [],
            "recorded_at": utc_now(),
        }
        self._append_line(payload)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        if not self._path.exists():
            return
        with self._path.open("rb") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    yield value

    def base_episodes(self) -> List[Dict[str, Any]]:
        return [r for r in self.iter_records() if r.get("record_kind") == "base"]

    def supplements_for(self, problem_id: str) -> List[Dict[str, Any]]:
        return [
            r for r in self.iter_records()
            if r.get("record_kind") == "gold_supplement" and r.get("problem_id") == problem_id
        ]

    def stats(self) -> Dict[str, int]:
        records = list(self.iter_records())
        return {
            "base": len([r for r in records if r.get("record_kind") == "base"]),
            "supplements": len([r for r in records if r.get("record_kind") == "gold_supplement"]),
        }


def build_episode(
    problem: str,
    problem_id: str,
    spec: Dict[str, Any],
    branches: List[Any],
    failure_count: int,
    status: str,
) -> EpisodeRecord:
    """Assemble an EpisodeRecord from a finished solve run."""
    signature = StructuralSignature.from_dict(spec.get("structural_signature", {}))
    branch_summaries = [
        BranchSummary(
            solver=b.solver,
            status=b.execution.status if b.execution else "unknown",
            attempts=len(b.attempts),
            objective_value=b.execution.objective_value if b.execution else None,
            termination_reason=b.termination_reason,
        )
        for b in branches
    ]
    objectives = [b.objective_value for b in branch_summaries if b.objective_value is not None]
    episode = EpisodeRecord(
        problem=problem,
        problem_id=problem_id,
        final_objective=objectives[0] if objectives else None,
        produced_realization_ids=[],
    )
    episode.normalized_spec = {
        "problem_family": spec.get("problem_family"),
        "objective": spec.get("objective"),
        "verified_model": spec.get("verified_model", ""),
        "status": status,
        "failure_count": failure_count,
    }
    episode.structural_signature = signature
    episode.branches = branch_summaries
    return episode


__all__ = ["EpisodeStore", "build_episode"]
