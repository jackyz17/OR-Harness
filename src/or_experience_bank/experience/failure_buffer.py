"""Failure buffer: stage-wise failure records kept for ONE solve run (Phase 2 step 1).

Failures are NOT written to any bank directly. They are buffered in memory for the
duration of a single solve() and consumed afterwards by comparative synthesis (success
vs failure contrast). Scope: per-problem, discarded after the run (cross-problem reuse
is the job of offline induction, not this buffer).

Three failure sources:
  - modeling    : StructuredModelingStage round issues (bad <model>)
  - branch_code : a solver branch attempt error (normalized_error)
  - reflection  : a model invalidated by outer reflection (wrong modeling direction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.schemas import utc_now


FAILURE_STAGES = ("modeling", "branch_code", "reflection")


@dataclass
class FailureRecord:
    """One buffered failure observation within a single solve run."""

    stage: str                                  # modeling | branch_code | reflection
    summary: str = ""                           # compact description of what failed
    normalized_error: Optional[str] = None      # branch_code: normalized error signature
    solver: Optional[str] = None                # branch_code: which solver branch
    attempt_id: Optional[str] = None            # pointer to the attempt evidence
    context: Dict[str, Any] = field(default_factory=dict)  # e.g. model snippet / issues
    failure_id: str = field(default_factory=lambda: "fail_" + uuid4().hex[:12])
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.stage not in FAILURE_STAGES:
            raise ValueError("stage {!r} not in {}".format(self.stage, FAILURE_STAGES))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "stage": self.stage,
            "summary": self.summary,
            "normalized_error": self.normalized_error,
            "solver": self.solver,
            "attempt_id": self.attempt_id,
            "context": self.context,
            "created_at": self.created_at,
        }


class FailureBuffer:
    """In-memory, per-solve buffer of failure records, grouped by stage."""

    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self._records: List[FailureRecord] = []

    def add(
        self,
        stage: str,
        summary: str,
        normalized_error: Optional[str] = None,
        solver: Optional[str] = None,
        attempt_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        record = FailureRecord(
            stage=stage,
            summary=summary,
            normalized_error=normalized_error,
            solver=solver,
            attempt_id=attempt_id,
            context=context or {},
        )
        self._records.append(record)
        return record

    def all(self) -> List[FailureRecord]:
        return list(self._records)

    def by_stage(self, stage: str) -> List[FailureRecord]:
        return [r for r in self._records if r.stage == stage]

    def is_empty(self) -> bool:
        return not self._records

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "count": self.count(),
            "records": [r.to_dict() for r in self._records],
        }


__all__ = ["FAILURE_STAGES", "FailureRecord", "FailureBuffer"]
