"""Modeling Bank retriever: unified recall of modeling-bank records as planning priors.

The Modeling Bank (ModelingStore) lives in its OWN store separate from the flat four
layers, so it needs its own retrieval path. This retriever wraps a dedicated
EmbeddingIndex (in its own index sub-directory, so it never collides with the flat
layers' index files) over modeling-bank records.

Unified recall (all records are peers): one query returns a single list of records,
all labeled [E1], [E2], .... Records with status="validated" (induced via unseen
transfer) naturally rank alongside directly-solved records by similarity + utility.
Deprecated records are excluded from the rebuilt index entirely (they live only in
the cold archive). Optional utility tracker can be applied for score penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..core.lifecycle import DEPRECATED, LifecycleStore
from ..core.modeling_store import ModelingStore
from .index import EmbeddingIndex


@dataclass
class PlanningPriors:
    """Retrieved planning priors injected into the modeling stage prompt.

    All records are peers — no distinction between "pattern" and "realization".
    Labels map short prompt tags ([E1], [E2], ...) back to experience ids so the
    framework can parse explicit citations out of the LLM's <think> text.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)  # short tag -> experience_id

    def is_empty(self) -> bool:
        return not self.records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.get("experience_id") for r in self.records],
            "labels": dict(self.labels),
        }

    # Backward-compatible accessors for code that still references .patterns/.realizations
    @property
    def patterns(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("status") == "validated"]

    @property
    def realizations(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("status") != "validated"]


class ModelingRetriever:
    """Dedicated retrieval over the Modeling Bank (unified, all records are peers)."""

    def __init__(
        self,
        store: ModelingStore,
        index: Optional[EmbeddingIndex] = None,
        lifecycle: Optional[LifecycleStore] = None,
    ):
        self.store = store
        self.index = index or EmbeddingIndex(store.bank_home / "index" / "modeling_bank")
        self.lifecycle = lifecycle

    def _retrievable(self, record: Dict[str, Any]) -> bool:
        if self.lifecycle is not None and self.lifecycle.state_of(record.get("experience_id", "")) == DEPRECATED:
            return False
        return True

    def rebuild(self) -> Dict[str, Any]:
        """Rebuild the modeling-bank index: all non-deprecated records."""
        records: List[Dict[str, Any]] = []
        for row in self.store.all_records():
            if not self._retrievable(row):
                continue
            records.append(self._to_index_record(row))
        return self.index.rebuild("modeling", records)

    @staticmethod
    def _to_index_record(row: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a ModelingExperience into an indexable row."""
        record = dict(row)
        # Prefix retrieval_text with modeling_aspect for natural embedding capture
        aspect = row.get("modeling_aspect") or ""
        text = row.get("retrieval_text") or ""
        if aspect and text:
            record["retrieval_text"] = "[{}] {}".format(aspect, text)
        return record

    def retrieve_priors(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> PlanningPriors:
        hits = self.index.query("modeling", query_text)
        priors = PlanningPriors()
        for _, record in hits:
            priors.records.append(record)
        priors.records = priors.records[:max(0, top_k)]
        priors.labels = self._build_labels(priors)
        return priors

    @staticmethod
    def _build_labels(priors: PlanningPriors) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for index, record in enumerate(priors.records, start=1):
            labels["E{}".format(index)] = record.get("experience_id", "")
        return labels


__all__ = ["PlanningPriors", "ModelingRetriever"]
