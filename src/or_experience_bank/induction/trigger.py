"""Induction trigger policy (v1, decision D4: manual / periodic batch).

Answers "should we run an induction pass now?" before the (expensive) pipeline burns
LLM + solver calls. v1 stays deliberately simple and robust:

  Gate 0 — candidate gate: there is at least one isomorphic + heterogeneous cluster
           worth inducing over (otherwise any run is wasted). Cheap: pure structural
           index, O(N), no LLM.
  Gate 1 — accumulation watermark (新增计数基线): at least `min_new_realizations` NEW
           realizations have been appended since the last induction run
           (periodic-by-volume, not by wall-clock — induction raw material is
           experience count, not calendar).
  Gate 2 — cooldown: a cluster whose membership has NOT changed since the last run
           is not re-induced (avoids re-burning LLM on an unchanged cluster).

State lives in an APPEND-ONLY sidecar JSONL (bank/induction_trigger_log.jsonl). A
"sidecar" (附属统计文件) is a small companion file next to the fact store that records
mutable stats; the fact layer itself is never modified (append-only red line, D2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.modeling_store import ModelingStore
from .candidates import CandidateCluster, SignatureClusterer


@dataclass
class TriggerDecision:
    """Whether to induce, and if so over which clusters."""

    should_induce: bool
    reason: str = ""
    realization_count: int = 0
    new_since_last: int = 0
    clusters_total: int = 0
    clusters_to_induce: List[CandidateCluster] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_induce": self.should_induce,
            "reason": self.reason,
            "realization_count": self.realization_count,
            "new_since_last": self.new_since_last,
            "clusters_total": self.clusters_total,
            "clusters_to_induce": [c.cluster_id for c in self.clusters_to_induce],
        }


class InductionTrigger:
    """v1 trigger: candidate gate + accumulation watermark + unchanged-cluster cooldown."""

    def __init__(
        self,
        store: ModelingStore,
        clusterer: SignatureClusterer,
        min_new_realizations: int = 3,
        log_path: Optional[Path] = None,
    ):
        if min_new_realizations < 1:
            raise ValueError("min_new_realizations must be >= 1")
        self.store = store
        self.clusterer = clusterer
        self.min_new_realizations = min_new_realizations
        self._log_path = Path(log_path) if log_path else (
            store.bank_dir / "induction_trigger_log.jsonl"
        )

    # -- public API -----------------------------------------------------------

    def decide(self) -> TriggerDecision:
        realizations = self.store.all_records()
        count = len(realizations)
        last = self._last_watermark()
        new_since = max(0, count - last)

        clusters = self.clusterer.discover(realizations)
        decision = TriggerDecision(
            should_induce=False,
            realization_count=count,
            new_since_last=new_since,
            clusters_total=len(clusters),
        )

        # Gate 0: candidate gate — nothing worth inducing over.
        if not clusters:
            decision.reason = "no heterogeneous isomorphic cluster (candidate gate)"
            return decision

        # Gate 1: cooldown — drop clusters whose membership is unchanged since last run.
        # Checked BEFORE the watermark: re-inducing an unchanged cluster is wasted work
        # regardless of how much new material accumulated elsewhere.
        seen = self._last_cluster_signatures()
        fresh = [c for c in clusters if self._cluster_signature(c) not in seen]
        if not fresh:
            decision.reason = "all candidate clusters unchanged since last run (cooldown)"
            return decision

        # Gate 2: accumulation watermark — not enough new material yet (skipped on the
        # very first run, when there is no prior watermark).
        if new_since < self.min_new_realizations and last > 0:
            decision.reason = "only {} new realizations (< {} watermark)".format(
                new_since, self.min_new_realizations
            )
            return decision

        decision.should_induce = True
        decision.clusters_to_induce = fresh
        decision.reason = "{} fresh cluster(s), {} new realization(s)".format(len(fresh), new_since)
        return decision

    def record_run(self, decision: TriggerDecision) -> None:
        """Append a watermark + cluster-signature snapshot after an induction run."""
        payload = {
            "realization_count": decision.realization_count,
            "cluster_signatures": sorted(
                self._cluster_signature(c) for c in decision.clusters_to_induce
            ),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # -- internal ---------------------------------------------------------------

    @staticmethod
    def _cluster_signature(cluster: CandidateCluster) -> str:
        """Membership fingerprint: changes iff the cluster's members change."""
        return cluster.core_key + "@" + ",".join(sorted(m.realization_id for m in cluster.members))

    def _read_log(self) -> List[Dict[str, Any]]:
        if not self._log_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self._log_path.open("rb") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def _last_watermark(self) -> int:
        rows = self._read_log()
        return int(rows[-1].get("realization_count", 0)) if rows else 0

    def _last_cluster_signatures(self) -> set:
        rows = self._read_log()
        return set(rows[-1].get("cluster_signatures", [])) if rows else set()


__all__ = ["InductionTrigger", "TriggerDecision"]
