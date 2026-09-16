"""Index synchronization: keeping the vector index current, never in the way.

Responsibility split (an explicit design decision, not an accident):

- WRITES keep the index fresh. Recording a fact, inducing an entry, and
  retiring one are the moments the underlying document changes, so the
  corresponding index item is refreshed right there — best effort.
- RECALL never writes. It re-reads each hit's current record by id and
  checks the document digest, so a stale index item can only ever cause a
  MISS (reported as stale), never a wrong answer.
- ``rebuild`` is the explicit maintenance operation: first build, repair
  after edits, or settle a model change.

Why best-effort: the fact must not be held hostage by a network call. An
embedding failure during ``record`` leaves the execution appended and
returns ``index_sync={"layer": "deferred", "reason": ...}`` so the harness
knows the memory is currently text-invisible and can re-run
``orx rebuild-index`` later. Nothing is rolled back and nothing is
fabricated.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from or_harness.strategy.embedding_index import (
    LAYER_EXECUTION,
    LAYER_STRATEGIC,
    LAYERS,
    document_digest,
    document_entry,
    document_execution,
)


class IndexSynchronizer:
    """Best-effort index maintenance for one harness instance."""

    def __init__(self, bank, sbank, catalog, store, index):
        self.bank = bank
        self.sbank = sbank
        self.catalog = catalog
        self.store = store
        self.index = index

    # -- documents -----------------------------------------------------------------

    def execution_document(self, record) -> Optional[str]:
        """The retrieval document of one fact, or None when its text is not
        recoverable (legacy record / never captured) — never a substitute."""
        if not record.task_text_digest:
            return None
        text = self.store.get_task_text(record.task_id,
                                       record.task_text_digest)
        if text is None:
            return None
        return document_execution(record, text)

    def entry_document(self, entry) -> str:
        return document_entry(entry, self.catalog.get(entry.strategy_id))

    # -- incremental syncs ----------------------------------------------------------

    def sync_execution(self, record) -> Dict[str, Any]:
        """Refresh one execution's index item. Never raises."""
        if self.index is None:
            return {"layer": LAYER_EXECUTION, "state": "skipped",
                    "reason": "no embedding backend configured"}
        document = self.execution_document(record)
        if document is None:
            return {"layer": LAYER_EXECUTION, "state": "skipped",
                    "reason": "task text unavailable (legacy record or never "
                              "captured); the fact is saved and stays visible "
                              "through profile retrieval and `orx inspect`"}
        try:
            out = self.index.upsert(LAYER_EXECUTION,
                                    [(record.execution_id, document)])
        except Exception as exc:  # noqa: BLE001 - save must not be blocked
            return {"layer": LAYER_EXECUTION, "state": "deferred",
                    "reason": f"embedding failed: {type(exc).__name__}: {exc}"}
        if out.get("skipped"):
            return {"layer": LAYER_EXECUTION, "state": "deferred",
                    "reason": out["skipped"]}
        return {"layer": LAYER_EXECUTION, "state": "synced",
                "items": out.get("items")}

    def sync_entries(self, entry_ids: Optional[Iterable[str]] = None
                     ) -> Dict[str, Any]:
        """Refresh the index items of the given entries (all when omitted).

        Entries whose document cannot be resolved (already retired) are
        REMOVED from the index instead — a vector that outlives its record
        would be pure noise.
        """
        if self.index is None:
            return {"layer": LAYER_STRATEGIC, "state": "skipped",
                    "reason": "no embedding backend configured"}
        wanted = (list(entry_ids) if entry_ids is not None
                  else [e.entry_id for e in self.sbank.list(include_dormant=True)])
        documents: List[Any] = []
        stale_ids: List[str] = []
        for entry_id in wanted:
            entry = self.sbank.get(entry_id)
            if entry is None:
                stale_ids.append(str(entry_id))
                continue
            documents.append((entry.entry_id, self.entry_document(entry)))
        try:
            if documents:
                out = self.index.upsert(LAYER_STRATEGIC, documents)
                if out.get("skipped"):
                    return {"layer": LAYER_STRATEGIC, "state": "deferred",
                            "reason": out["skipped"]}
            if stale_ids:
                self.index.remove(LAYER_STRATEGIC, stale_ids)
        except Exception as exc:  # noqa: BLE001 - never block the main path
            return {"layer": LAYER_STRATEGIC, "state": "deferred",
                    "reason": f"embedding failed: {type(exc).__name__}: {exc}"}
        return {"layer": LAYER_STRATEGIC, "state": "synced",
                "items": len(documents), "removed": len(stale_ids)}

    def forget_entry(self, entry_id: str) -> Dict[str, Any]:
        """Drop a retired entry's vector (delegates to ``sync_entries``)."""
        if self.index is None:
            return {"layer": LAYER_STRATEGIC, "state": "skipped",
                    "reason": "no embedding backend configured"}
        try:
            out = self.index.remove(LAYER_STRATEGIC, [entry_id])
        except Exception as exc:  # noqa: BLE001
            return {"layer": LAYER_STRATEGIC, "state": "deferred",
                    "reason": f"index write failed: {type(exc).__name__}: {exc}"}
        return {"layer": LAYER_STRATEGIC, "state": "synced",
                "removed": out.get("removed", 0)}

    # -- explicit rebuild ------------------------------------------------------------

    def rebuild(self, layer: str = "both", dry_run: bool = False) -> Dict[str, Any]:
        """Rebuild the index from current facts and entries.

        The explicit maintenance entry point (first build / repair / model
        change). ``--layer execution|strategic|both`` scopes it; the CLI's
        short names are accepted alongside the full layer ids. A dry run
        counts what WOULD be indexed and touches nothing — not even the
        index directory.
        """
        aliases = {"execution": LAYER_EXECUTION,
                   "strategic": LAYER_STRATEGIC}
        layer = aliases.get(layer, layer)
        if layer not in ("both",) + LAYERS:
            raise ValueError(f"layer must be 'both', 'execution', or "
                             f"'strategic' (got {layer!r})")
        targets = list(LAYERS) if layer == "both" else [layer]
        plan: Dict[str, Any] = {}
        for name in targets:
            if name == LAYER_EXECUTION:
                pairs, skipped = [], []
                for record in self.bank.all():
                    if record.source != "executed":
                        continue
                    document = self.execution_document(record)
                    if document is None:
                        skipped.append(record.execution_id)
                        continue
                    pairs.append((record.execution_id, document))
                plan[name] = {"documents": pairs,
                              "unindexable": skipped}
            else:
                pairs = [(entry.entry_id, self.entry_document(entry))
                         for entry in self.sbank.list(include_dormant=True)]
                plan[name] = {"documents": pairs, "unindexable": []}
        if dry_run:
            return {
                "dry_run": True,
                "layers": {name: {"would_index": len(plan[name]["documents"]),
                                  "unindexable": len(plan[name]["unindexable"])}
                           for name in targets},
                "backend": (None if self.index is None
                            else self.index.backend.model_id),
                "note": ("dry run: no embedding call was made and no index "
                         "file was written or modified"),
            }
        if self.index is None:
            raise ValueError(
                "no embedding backend configured: set OR_EMBEDDING_BASE_URL, "
                "OR_EMBEDDING_MODEL, and OR_EMBEDDING_API_KEY (or pass "
                "embedding=... to ORHarness) before rebuilding the index")
        results: Dict[str, Any] = {}
        for name in targets:
            results[name] = self.index.rebuild(name, plan[name]["documents"])
            results[name]["unindexable"] = len(plan[name]["unindexable"])
        return {"dry_run": False, "layers": results,
                "backend": self.index.backend.model_id}

    # -- health ----------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Read-only index health for ``doctor``.

        Fact/entry counts vs index counts, model id, and how many current
        documents are STALE (present in the index with an old digest) or
        missing (not indexed at all). Nothing is written and no embedding
        call is made.
        """
        if self.index is None:
            return {"configured": False,
                    "reason": "no embedding backend configured",
                    "layers": {}}
        layers: Dict[str, Any] = {}
        for name in LAYERS:
            status = self.index.status(name)
            indexed = {str(item.get("id")): item
                       for item in self.index.items(name)}
            if name == LAYER_EXECUTION:
                documents = []
                for record in self.bank.all():
                    if record.source != "executed":
                        continue
                    document = self.execution_document(record)
                    if document is not None:
                        documents.append((record.execution_id, document))
            else:
                documents = [(entry.entry_id, self.entry_document(entry))
                             for entry in self.sbank.list(include_dormant=True)]
            stale = sum(1 for doc_id, text in documents
                        if doc_id in indexed
                        and indexed[doc_id].get("doc_digest")
                        != document_digest(text))
            missing = sum(1 for doc_id, _ in documents if doc_id not in indexed)
            orphaned = sum(1 for doc_id in indexed
                           if doc_id not in {d for d, _ in documents})
            layers[name] = {
                **status,
                "documents": len(documents),
                "stale": stale,
                "missing": missing,
                "orphaned": orphaned,
            }
        return {"configured": True,
                "backend": self.index.backend.model_id,
                "layers": layers}

    def unindexed_counts(self, layers: Sequence[str] = LAYERS) -> Dict[str, int]:
        """Current memories with no index item (read-only)."""
        if self.index is None:
            return {LAYER_EXECUTION: 0, LAYER_STRATEGIC: 0}
        counts: Dict[str, int] = {}
        if LAYER_EXECUTION in layers:
            indexed = {str(i.get("id")) for i in self.index.items(LAYER_EXECUTION)}
            counts[LAYER_EXECUTION] = sum(
                1 for r in self.bank.all()
                if r.source == "executed"
                and (r.execution_id not in indexed
                     or self.execution_document(r) is None))
        if LAYER_STRATEGIC in layers:
            indexed = {str(i.get("id")) for i in self.index.items(LAYER_STRATEGIC)}
            counts[LAYER_STRATEGIC] = sum(
                1 for e in self.sbank.list(include_dormant=True)
                if e.entry_id not in indexed)
        return counts