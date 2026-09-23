"""Rebuildable embedding index for retrieval (discovery layer only).

Split of responsibilities (the design rule this module exists to serve):

- Embedding decides WHICH memories are SEEN. It is a discovery signal and
  nothing else: a cosine similarity is never a quality estimate, never a
  cost, never a failure probability, and never widens a claim's applicability.
- Profile predicates and applicability conditions decide whether a seen
  memory may be REUSED. That judgment lives in
  :mod:`or_harness.core.schema` (``group_key`` / ``profile_matches``) and is
  untouched by this module.

So this module is deliberately narrow: backend adapters, an atomic
sidecar-JSON vector index, and the two document builders that turn an
execution fact / a strategic entry into comparable text. It stores NO record
snapshot — only ``{id, doc_digest, vector}`` per item — because recall reads
the CURRENT state of a record by id and compares ``doc_digest`` to detect
staleness. An index is therefore pure derived data: rebuildable at any time,
safe to delete, and never a source of truth.

Ports the legacy project's retrieval index (``EmbeddingBackend`` protocol,
OpenAI-compatible adapter over stdlib urllib, thread + ``flock`` rebuild
lock, atomic ``.tmp`` → ``replace()`` publication, ``model_id``
invalidation on load) with two deliberate changes:

1. ``auto`` NO LONGER falls back to local hashing. A hashing backend is a
   deterministic bag-of-features vector — useful for offline tests, but it
   is NOT semantic retrieval, and silently presenting it as one would make
   text similarity look meaningful when it is not. Without a configured
   real model the vector channel is skipped and retrieval degrades to the
   profile path, honestly reported.
2. Credentials come from ``OR_EMBEDDING_BASE_URL`` / ``OR_EMBEDDING_MODEL``
   / ``OR_EMBEDDING_API_KEY``. These are NOT the ``OR_WM_*`` variables: the
   chat model used for outcome prediction and the embedding model may be
   different endpoints with different dimensions, and sharing one variable
   set would silently mix them.

Pure stdlib: ``json`` / ``hashlib`` / ``math`` / ``os`` / ``re`` /
``threading`` / ``urllib`` / ``fcntl``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from or_harness.core.schema import ExecutionRecord, StrategicEntry
#: ONE digest convention across the harness: the same truncation, the same
#: sort-keys/compact separators, the same ``default=str`` width rules. The
#: helper is imported rather than re-implemented so an index digest can
#: never disagree with a task/snapshot digest about the same text.
from or_harness.world_model.state import _stable_digest

#: The two indexed layers. Named after the layering of the memory itself —
#: facts (execution evidence) vs derived commitments (strategic knowledge).
LAYER_EXECUTION = "execution_evidence"
LAYER_STRATEGIC = "strategic_knowledge"
LAYERS: Tuple[str, ...] = (LAYER_EXECUTION, LAYER_STRATEGIC)

#: Index files live in a sidecar subdirectory of the harness home. They are
#: DERIVED data: deleting the whole directory loses nothing but work.
INDEX_SUBDIR = "index"

#: Payload schema token (versioned independently of the SQLite schema).
INDEX_SCHEMA_VERSION = "1.0"

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|[\u4e00-\u9fff]")

#: Readable dimension names for predicate transcription (the grouping
#: features are the only predicate dimensions that exist).
_DIMENSION_LABELS = {
    "resource_coupling": "resource coupling",
    "temporal_coupling": "temporal coupling",
    "route_complexity": "route complexity",
}

# Cross-process/thread lock guard for index writes: parallel ``orx``
# invocations are real (a harness may run several tasks at once), and two
# writers racing on the same ``.tmp`` file clobber each other's payload.
_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def _rebuild_lock(index_dir: Path):
    """Serialize index writes within a process (thread lock) and across
    processes (flock on a sidecar lock file)."""
    lock_path = Path(index_dir) / ".index_rebuild.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        with lock_path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def tokenize(text: str) -> List[str]:
    """Lower-cased word/character tokens (CJK falls back to single chars)."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class EmbeddingBackend(Protocol):
    """The whole backend contract: an identity, a width, and two calls."""

    model_id: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class LocalHashEmbeddingBackend:
    """Deterministic local embedding — EXPLICIT-USE ONLY (never ``auto``).

    Token and character n-gram features are signed-hashed into a fixed dense
    vector; no fitted vocabulary, no external service. This is a lexical
    feature hash, NOT a semantic model: two sentences with the same meaning
    and no shared tokens are unrelated to it. Kept for tests and for an
    explicitly requested offline run (``OR_EMBEDDING_BACKEND=local-hashing``),
    so a hermetic environment can exercise the vector path without pretending
    it has semantic understanding.
    """

    model_id = "local-hashing-embedding-v1"

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        normalized = (text or "").lower()
        features = tokenize(normalized)
        compact = re.sub(r"\s+", "", normalized)
        features.extend(compact[i:i + 3] for i in range(max(0, len(compact) - 2)))
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[position] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class OpenAICompatibleEmbeddingBackend:
    """Adapter for any OpenAI-compatible ``/embeddings`` endpoint.

    stdlib urllib only, credentials in a header and never persisted; the
    model name comes from the caller/environment, never from a built-in
    registry. ``dimension`` is 0 until the first call reports it.
    """

    def __init__(self, base_url: str, api_key: str, model_id: str,
                 timeout_seconds: float = 60.0):
        if not base_url or not api_key or not model_id:
            raise ValueError("base_url, api_key, and model_id are required")
        self.base_url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model_id = model_id
        self.dimension = 0
        self.timeout_seconds = float(timeout_seconds)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._request(list(texts))
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self._request([text])
        if vectors:
            self.dimension = len(vectors[0])
        return vectors[0]

    def _request(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({"model": self.model_id, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request,
                                    timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
        return [list(map(float, item["embedding"])) for item in ordered]

    def describe(self) -> Dict[str, Any]:
        """Non-sensitive description: model + host only, never credentials."""
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0]
        return {"source": "remote", "model_id": self.model_id,
                "endpoint_host": host, "dimension": self.dimension}


def create_embedding_backend(backend: Optional[str] = None
                             ) -> Optional[EmbeddingBackend]:
    """Build a backend from an explicit name, else from the environment.

    ``auto`` (the default) means: use the OpenAI-compatible endpoint when
    ``OR_EMBEDDING_BASE_URL`` / ``OR_EMBEDDING_MODEL`` / ``OR_EMBEDDING_API_KEY``
    are ALL present; otherwise return ``None`` so the caller degrades to the
    profile path. It never silently substitutes the hashing backend.

    ``local-hashing`` (or ``OR_EMBEDDING_BACKEND=local-hashing``) selects the
    deterministic hashing backend explicitly — tests and hermetic offline
    runs. ADMISSION NOTE: with the hashing backend the index's ``model_id``
    is ``local-hashing-embedding-v1``, so an index built that way is
    invisible to a real model and vice versa. The vector spaces never mix.
    """
    selected = (backend or os.environ.get("OR_EMBEDDING_BACKEND")
                or "auto").strip().lower()
    if selected in ("none", "off", "disabled"):
        return None
    if selected in ("local", "hashing", "local-hashing"):
        return LocalHashEmbeddingBackend()
    if selected not in ("auto", "openai-compatible", "openai", "remote"):
        raise ValueError(f"unsupported embedding backend {selected!r}")
    base_url = os.environ.get("OR_EMBEDDING_BASE_URL", "").strip()
    model_id = os.environ.get("OR_EMBEDDING_MODEL", "").strip()
    api_key = os.environ.get("OR_EMBEDDING_API_KEY", "").strip()
    if not (base_url and model_id and api_key):
        if selected == "auto":
            return None
        raise ValueError(
            "embedding backend 'openai-compatible' requires "
            "OR_EMBEDDING_BASE_URL, OR_EMBEDDING_MODEL, and "
            "OR_EMBEDDING_API_KEY")
    return OpenAICompatibleEmbeddingBackend(base_url, api_key, model_id)


# ---------------------------------------------------------------------------
# Document builders (already-existing fields only — no extra LLM processing)
# ---------------------------------------------------------------------------

def document_execution(record: ExecutionRecord, task_text: str) -> str:
    """The retrieval document of one execution fact.

    Task text FIRST and dominant (it is what a future similar task is
    matched against), then the strategy that was used and the outcome
    summary (status / feasibility / failure count) so "which strategy was
    tried on comparable problems, and how did it go" lives in the same
    vector space. Only fields already on the record are used.
    """
    status = record.quality.get("status")
    feasible = record.quality.get("feasible")
    parts = [
        (task_text or "").strip(),
        f"strategy {record.strategy_id}",
        f"family {record.profile_snapshot.family}",
        f"outcome {status}" if status else "",
        ("feasible" if feasible else "not feasible") if feasible is not None else "",
        f"failures {len(record.failures)}" if record.failures else "",
    ]
    return " ".join(part for part in parts if part)


def document_entry(entry: StrategicEntry) -> str:
    """The retrieval document of one strategic entry (knowledge text).

    Composed ENTIRELY of text that ALREADY EXISTS on the entry: the
    harness-written ``applicability`` notes, the entry's structured relation
    claims, its own recorded actions, and a readable transcription of its
    predicates. No new summarization step is invented — a generated
    paraphrase would be unverifiable content masquerading as the claim — and
    nothing is pulled from a built-in directory: a method's name and
    description are not evidence about what ran.
    """
    parts: List[str] = []
    parts.extend(str(note) for note in entry.applicability if str(note).strip())
    if entry.risk_conditions:
        parts.append("risks: " + "; ".join(str(r) for r in entry.risk_conditions))
    # Relation claims are part of what the knowledge SAYS: a reader searching
    # by text must be able to find "keep the cross-period state", which lives
    # in a relation, not in the entry's own notes. Editing a claim moves the
    # digest, so the existing stale-vector rule drops the old vector.
    for relation in entry.relations or []:
        claim = str((relation or {}).get("claim") or "").strip()
        if claim:
            parts.append("relation: " + claim)
    if entry.actions:
        parts.append("actions: " + " ".join(str(a) for a in entry.actions))
    parts.append(readable_predicates(entry.predicates))
    parts.append(f"strategy {entry.strategy_id}")
    return " ".join(part for part in parts if part and str(part).strip())


def readable_predicates(predicates: Dict[str, Any]) -> str:
    """Plain-text transcription of an applicability predicate set.

    ``family=vrp resource coupling in [0.25, 0.50] temporal coupling
    unknown`` — the same content as the structured predicates, written so
    it can participate in a text similarity comparison. Transcription only:
    it never changes what the predicates mean (the structured form stays the
    authority for matching).
    """
    parts: List[str] = []
    family = predicates.get("family")
    if family:
        parts.append(f"family {family}")
    for dim in ("resource_coupling", "temporal_coupling", "route_complexity"):
        if dim not in predicates:
            continue
        label = _DIMENSION_LABELS.get(dim, dim)
        value = predicates[dim]
        if isinstance(value, str) or value is None:
            parts.append(f"{label} unknown")
            continue
        try:
            lo, hi = (float(x) for x in value)
        except (TypeError, ValueError):
            continue
        parts.append(f"{label} in [{lo:.2f}, {hi:.2f}]")
    return " ".join(parts)


def document_digest(text: str) -> str:
    """Digest of one document's text (the staleness key of an index item)."""
    return _stable_digest(text)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for a length mismatch or a zero vector.

    The length guard matters: ``zip`` would silently compare the first N
    components of two vectors from different spaces and return a plausible
    number for it. An incomparable pair scores 0 instead.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class EmbeddingIndex:
    """Atomic sidecar-JSON index over one embedding backend."""

    def __init__(self, index_dir: os.PathLike[str] | str,
                 backend: EmbeddingBackend):
        self.index_dir = Path(index_dir)
        self.backend = backend

    # -- paths -------------------------------------------------------------------

    def path(self, layer: str) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}")
        return self.index_dir / f"{layer}.embedding.json"

    # -- reads (never write anything) ----------------------------------------------

    def status(self, layer: str) -> Dict[str, Any]:
        """Health of one layer WITHOUT building it: existence, model id,
        dimension, item count, and whether it is usable by this backend.

        Read-only by construction (no mkdir, no rebuild) so ``recall`` /
        ``doctor`` can report honestly instead of triggering writes.
        """
        path = self.path(layer)
        if not path.exists():
            return {"layer": layer, "exists": False, "usable": False,
                    "reason": "index missing", "model_id": None,
                    "dimension": None, "count": 0}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"layer": layer, "exists": True, "usable": False,
                    "reason": f"index unreadable: {exc}", "model_id": None,
                    "dimension": None, "count": 0}
        model_id = payload.get("model_id")
        items = payload.get("items") or []
        if model_id != self.backend.model_id:
            return {"layer": layer, "exists": True, "usable": False,
                    "reason": (f"embedding model changed (was {model_id}, now "
                               f"{self.backend.model_id})"),
                    "model_id": model_id,
                    "dimension": payload.get("dimension"), "count": len(items)}
        current = self.backend.dimension
        stored = payload.get("dimension")
        if current and stored and int(current) != int(stored):
            return {"layer": layer, "exists": True, "usable": False,
                    "reason": (f"embedding dimension changed (index {stored}, "
                               f"backend {current})"),
                    "model_id": model_id, "dimension": stored,
                    "count": len(items)}
        return {"layer": layer, "exists": True, "usable": True, "reason": None,
                "model_id": model_id, "dimension": stored, "count": len(items)}

    def load(self, layer: str) -> Dict[str, Any]:
        """The usable payload of one layer, or ``{}``.

        ``{}`` means "do not use this index", covering every refusal: absent
        file, unreadable file, or a ``model_id`` that is not this backend's.
        The model check is the vector-space isolation rule — vectors from
        two different models are not comparable, so a stale index is never
        partially used.
        """
        layer_status = self.status(layer)
        if not layer_status["usable"]:
            return {}
        try:
            return json.loads(self.path(layer).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - raced delete
            return {}

    def items(self, layer: str) -> List[Dict[str, Any]]:
        """Index items of a usable layer (``[]`` when unusable)."""
        return list(self.load(layer).get("items") or [])

    # -- writes --------------------------------------------------------------------

    def rebuild(self, layer: str, documents: Iterable[Tuple[str, str]]
                ) -> Dict[str, Any]:
        """Full rebuild from ``(id, document_text)`` pairs, atomically.

        Derives the whole index from current source documents and publishes
        it by writing a ``.tmp`` file and ``replace()``-ing it: a concurrent
        reader sees the old index or the new one, never a half-written file.
        """
        rows = [(str(doc_id), str(text)) for doc_id, text in documents]
        documents_only = [text for _, text in rows]
        vectors = (self.backend.embed_documents(documents_only)
                   if documents_only else [])
        if len(vectors) != len(rows):
            raise ValueError(
                f"backend returned {len(vectors)} vectors for {len(rows)} "
                "documents")
        dimension = (len(vectors[0]) if vectors else self.backend.dimension)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "model_id": self.backend.model_id,
            "dimension": dimension,
            "layer": layer,
            "built_at": time.time(),
            "items": [{"id": doc_id, "doc_digest": document_digest(text),
                       "vector": vector}
                      for (doc_id, text), vector in zip(rows, vectors)],
        }
        self._publish(layer, payload)
        return {"layer": layer, "items": len(rows),
                "model_id": self.backend.model_id, "dimension": dimension}

    def upsert(self, layer: str, documents: Iterable[Tuple[str, str]]
               ) -> Dict[str, Any]:
        """Incremental write of ``(id, document_text)`` pairs.

        Best-effort by design: callers treat a failure or a refusal as
        "deferred" and keep the fact they already saved. Refuses to touch an
        index built by a DIFFERENT model — merging vectors from two spaces
        would corrupt every future similarity, so the honest answer is to
        report the mismatch and let ``rebuild-index`` (an explicit write
        operation) settle it.
        """
        rows = [(str(doc_id), str(text)) for doc_id, text in documents]
        if not rows:
            return {"layer": layer, "upserted": 0, "items": None}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with _rebuild_lock(self.index_dir):
            path = self.path(layer)
            payload: Dict[str, Any] = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "model_id": self.backend.model_id,
                "dimension": self.backend.dimension,
                "layer": layer,
                "built_at": time.time(),
                "items": [],
            }
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing is not None:
                    if existing.get("model_id") != self.backend.model_id:
                        return {
                            "layer": layer, "upserted": 0, "items": None,
                            "skipped": (
                                "index was built by embedding model "
                                f"{existing.get('model_id')!r}, not "
                                f"{self.backend.model_id!r}; run "
                                "`orx rebuild-index` to rebuild it"),
                        }
                    payload = existing
                    payload["items"] = list(existing.get("items") or [])
            vectors = self.backend.embed_documents([text for _, text in rows])
            by_id = {item.get("id"): item for item in payload["items"]}
            for (doc_id, text), vector in zip(rows, vectors):
                by_id[doc_id] = {"id": doc_id,
                                 "doc_digest": document_digest(text),
                                 "vector": vector}
            payload["items"] = [by_id[k] for k in sorted(by_id)]
            payload["model_id"] = self.backend.model_id
            if vectors:
                payload["dimension"] = len(vectors[0])
            payload["built_at"] = time.time()
            self._publish_locked(layer, payload)
        return {"layer": layer, "upserted": len(rows),
                "items": len(payload["items"]),
                "model_id": self.backend.model_id}

    def remove(self, layer: str, ids: Iterable[str]) -> Dict[str, Any]:
        """Drop items by id (e.g. a retired entry). No-op when the index is
        absent or belongs to another model — an index that cannot be trusted
        is never partially edited."""
        wanted = {str(i) for i in ids}
        if not wanted:
            return {"layer": layer, "removed": 0}
        path = self.path(layer)
        if not path.exists():
            return {"layer": layer, "removed": 0, "skipped": "index missing"}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with _rebuild_lock(self.index_dir):
            if not path.exists():  # pragma: no cover - raced delete
                return {"layer": layer, "removed": 0, "skipped": "index missing"}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"layer": layer, "removed": 0,
                        "skipped": "index unreadable"}
            if payload.get("model_id") != self.backend.model_id:
                return {"layer": layer, "removed": 0,
                        "skipped": "index built by another embedding model"}
            before = list(payload.get("items") or [])
            payload["items"] = [item for item in before
                                if item.get("id") not in wanted]
            removed = len(before) - len(payload["items"])
            if removed:
                payload["built_at"] = time.time()
                self._publish_locked(layer, payload)
        return {"layer": layer, "removed": removed}

    def _publish(self, layer: str, payload: Dict[str, Any]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with _rebuild_lock(self.index_dir):
            self._publish_locked(layer, payload)

    def _publish_locked(self, layer: str, payload: Dict[str, Any]) -> None:
        """Atomic publication. Caller holds the rebuild lock."""
        target = self.path(layer)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        temporary.replace(target)


def index_dir_for(home: os.PathLike[str] | str) -> Path:
    """The sidecar index directory for one harness home."""
    return Path(home) / INDEX_SUBDIR


def describe_backend(backend: EmbeddingBackend) -> Dict[str, Any]:
    """Non-sensitive backend description for recall results / doctor."""
    describe = getattr(backend, "describe", None)
    if callable(describe):
        return dict(describe())
    source = ("local-hashing"
              if isinstance(backend, LocalHashEmbeddingBackend) else "injected")
    return {"source": source, "model_id": backend.model_id,
            "dimension": backend.dimension}