"""Rebuildable embedding-vector retrieval index."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Sequence

from ..core.schemas import ExperienceLayer


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


class EmbeddingBackend(Protocol):
    model_id: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class LocalHashEmbeddingBackend:
    """Deterministic local embedding for offline RAG and tests.

    Token and character n-gram features are signed-hashed into a fixed dense
    vector. It needs no fitted vocabulary and no external service.
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
        features.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[position] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class OpenAICompatibleEmbeddingBackend:
    """Optional adapter for any OpenAI-compatible embeddings endpoint."""

    def __init__(self, base_url: str, api_key: str, model_id: str, timeout_seconds: int = 60):
        if not base_url or not api_key or not model_id:
            raise ValueError("base_url, api_key, and model_id are required")
        self.base_url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model_id = model_id
        self.dimension = 0
        self.timeout_seconds = timeout_seconds

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
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
        return [list(map(float, item["embedding"])) for item in ordered]


def create_embedding_backend(backend: str = "auto", model_id: str = None) -> EmbeddingBackend:
    selected = (backend or "auto").lower()
    if selected in {"auto", "local", "hashing", "local-hashing"}:
        return LocalHashEmbeddingBackend()
    if selected in {"openai-compatible", "hermes"}:
        return OpenAICompatibleEmbeddingBackend(
            base_url=os.environ.get("OR_EXPERIENCE_EMBEDDING_BASE_URL", ""),
            api_key=os.environ.get("OR_EXPERIENCE_EMBEDDING_API_KEY", ""),
            model_id=model_id or os.environ.get("OR_EXPERIENCE_EMBEDDING_MODEL", ""),
        )
    raise ValueError("Unsupported embedding backend: " + backend)


class EmbeddingIndex:
    """Rebuildable dense-vector index backed by an EmbeddingBackend."""

    def __init__(self, index_dir: Path, backend: Optional[EmbeddingBackend] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend or LocalHashEmbeddingBackend()

    def path(self, layer: str) -> Path:
        ExperienceLayer(layer)
        return self.index_dir / (layer + ".embedding.json")

    def rebuild(self, layer: str, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(records)
        documents = [self._document(row) for row in rows]
        vectors = self.backend.embed_documents(documents)
        dimension = len(vectors[0]) if vectors else self.backend.dimension
        payload = {
            "schema_version": "1.0",
            "model_id": self.backend.model_id,
            "dimension": dimension,
            "layer": layer,
            "vectors": vectors,
            "records": rows,
        }
        target = self.path(layer)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(target)
        return {
            "layer": layer,
            "records": len(rows),
            "model_id": self.backend.model_id,
            "dimension": dimension,
        }

    def load(self, layer: str) -> Dict[str, Any]:
        path = self.path(layer)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_id") != self.backend.model_id:
            return {}
        return payload

    def query(self, layer: str, query: str) -> List[tuple]:
        payload = self.load(layer)
        if not payload:
            return []
        query_vector = self.backend.embed_query(query)
        scored = [
            (sum(a * b for a, b in zip(query_vector, vector)), record)
            for record, vector in zip(payload["records"], payload["vectors"])
        ]
        return sorted(scored, key=lambda item: (-item[0], item[1].get("experience_id", "")))

    @staticmethod
    def _document(row: Dict[str, Any]) -> str:
        context = row.get("problem_context", {})
        scope = row.get("scope", {})
        trigger = row.get("trigger", {})
        policy = row.get("policy", {})
        values = [
            row.get("title"), row.get("retrieval_text"), context.get("problem_family"),
            context.get("objective_type"), context.get("stage"),
            " ".join(context.get("keywords", [])), scope.get("solver"),
            scope.get("solver_family"), trigger.get("situation"),
            trigger.get("normalized_error"), trigger.get("solver_status"),
            trigger.get("performance_symptom"), policy.get("diagnosis"), policy.get("action"),
        ]
        return " ".join(str(value or "") for value in values)
