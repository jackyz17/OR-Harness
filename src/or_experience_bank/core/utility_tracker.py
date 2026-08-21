"""Utility statistics + soft-delete scoring (Phase 2 module 2.3).

Tracks how useful each experience actually is, in a mutable sidecar (bank/utility_stats.json)
— never in the append-only fact line.

Two counters per experience:
  retrieval_count : how many times it was returned by retrieve()  ("seen")
  utility_count   : how many times it contributed to a successful solve  ("helped")

Soft-delete (降权, not deletion): a record that has been seen enough (freq >= alpha) but
rarely helps (utility/freq < beta) is DEPRIORITIZED at retrieval by multiplying its score
by a penalty factor. The record stays in the hot bank (append-only); it just sinks.
The freq >= alpha guard protects NEW experiences: with zero/few retrievals they have no
evidence either way, so they must NOT be judged yet.

This module is the data source for LifecycleStore decisions (auto-deprecate candidates)
and for induction candidates' priority signal (retrieval_hits).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Default soft-delete thresholds (configurable).
DEFAULT_ALPHA = 5          # min retrievals before judging (protects new experiences)
DEFAULT_BETA = 0.1         # utility ratio below this -> low-utility
DEFAULT_PENALTY = 0.3      # retrieval score multiplier for low-utility records


class UtilityTracker:
    """Mutable utility counters (sidecar) + soft-delete scoring rules."""

    def __init__(
        self,
        bank_home: Path,
        alpha: int = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        penalty: float = DEFAULT_PENALTY,
    ):
        if alpha < 1:
            raise ValueError("alpha must be >= 1 (new experiences need a grace window)")
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        if not (0.0 < penalty < 1.0):
            raise ValueError("penalty must be in (0, 1)")
        self.bank_home = Path(bank_home)
        self.bank_dir = self.bank_home / "bank"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.bank_dir / "utility_stats.json"
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty

    # -- persistence ------------------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, int]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, stats: Dict[str, Dict[str, int]]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    def _bump(self, experience_id: str, field: str) -> None:
        if not experience_id:
            return
        stats = self._load()
        entry = stats.setdefault(experience_id, {"retrieval_count": 0, "utility_count": 0})
        entry[field] = int(entry.get(field, 0)) + 1
        self._save(stats)

    # -- counters ----------------------------------------------------------------

    def record_retrieval(self, experience_id: str) -> None:
        """+1 seen. Called for every hit returned by retrieve()."""
        self._bump(experience_id, "retrieval_count")

    def record_retrievals(self, experience_ids: List[str]) -> None:
        for eid in experience_ids:
            self.record_retrieval(eid)

    def record_utility(self, experience_id: str) -> None:
        """+1 helped. Called when an experience contributed to a successful solve."""
        self._bump(experience_id, "utility_count")

    def record_utilities(self, experience_ids: List[str]) -> None:
        for eid in experience_ids:
            self.record_utility(eid)

    # -- reads ----------------------------------------------------------------

    def stats_for(self, experience_id: str) -> Dict[str, int]:
        return dict(self._load().get(experience_id, {"retrieval_count": 0, "utility_count": 0}))

    def retrieval_count(self, experience_id: str) -> int:
        return self.stats_for(experience_id)["retrieval_count"]

    def utility_count(self, experience_id: str) -> int:
        return self.stats_for(experience_id)["utility_count"]

    def utility_ratio(self, experience_id: str) -> Optional[float]:
        """utility/retrieval, or None if never retrieved (no evidence)."""
        stats = self.stats_for(experience_id)
        freq = stats["retrieval_count"]
        if freq == 0:
            return None
        return stats["utility_count"] / float(freq)

    # -- soft-delete rules ------------------------------------------------------

    def is_low_utility(self, experience_id: str) -> bool:
        """True iff enough evidence (freq >= alpha) AND ratio < beta. New records -> False."""
        stats = self.stats_for(experience_id)
        freq = stats["retrieval_count"]
        if freq < self.alpha:
            return False  # grace window: not enough evidence to judge
        return (stats["utility_count"] / float(freq)) < self.beta

    def score_multiplier(self, experience_id: str) -> float:
        """Retrieval score multiplier: penalty for low-utility, 1.0 otherwise."""
        return self.penalty if self.is_low_utility(experience_id) else 1.0

    def apply_penalty(self, experience_id: str, score: float) -> float:
        return score * self.score_multiplier(experience_id)

    def stats(self) -> Dict[str, Any]:
        data = self._load()
        return {
            "tracked": len(data),
            "low_utility": len([e for e in data if self.is_low_utility(e)]),
            "path": str(self._path),
        }


__all__ = [
    "UtilityTracker",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_PENALTY",
]
