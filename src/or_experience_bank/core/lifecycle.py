"""Experience lifecycle states + compressed cold-storage archive (Phase 2 module 2.3).

Motivation: the bank must EVOLVE — new experiences arrive, induction distills new
patterns, and some experiences turn out to be useless or even harmful (using them makes
solving worse). They must be retired without breaking the audit chain.

Red line preserved: a record's CONTENT is never edited or physically deleted (content-hash
dedup, Episode provenance, and induction derived_from all depend on it). What changes is
the record's LIFECYCLE STATE, held in a mutable sidecar (bank/lifecycle.json), never in
the fact line itself.

State machine:
    active  ──(long-term low utility / observed harmful)──▶  deprecated  (moved OUT of the hot
      bank into cold archive)

Cold archive (archive/deprecated.jsonl): a COMPRESSED provenance card, not the full record.
The bulky retrieval_text is dropped, but its embedding VECTOR is kept (方案甲) so that both
exact-hash and approximate-similarity dedup still work against the archive — a harmful
experience cannot resurrect, whether verbatim or reworded.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Lifecycle states. Only two: active (default) and deprecated (retired).
# The "superseded" state has been removed: all modeling-bank records are peers,
# so a more general record does not retire a more specific one. Utility-based
# soft delete (low retrieval_count with low utility ratio) handles ranking
# demotion without a lifecycle state change.
ACTIVE = "active"
DEPRECATED = "deprecated"
LIFECYCLE_STATES = (ACTIVE, DEPRECATED)

# Default threshold for approximate (cosine) similarity dedup against the archive.
DEFAULT_SIMILARITY_THRESHOLD = 0.8


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _summarize(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class LifecycleStore:
    """Mutable lifecycle state (sidecar) + compressed cold archive.

    Owns two files next to the fact store:
      - bank/lifecycle.json         : {experience_id: {state, superseded_by, deprecated_at, reason}}
      - archive/deprecated.jsonl    : compressed provenance cards (append-only)
      - archive/deprecated_index.json : [{experience_id, content_hash, vector}] for dedup
    """

    def __init__(
        self,
        bank_home: Path,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.bank_home = Path(bank_home)
        self.bank_dir = self.bank_home / "bank"
        self.archive_dir = self.bank_home / "archive"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.bank_dir / "lifecycle.json"
        self._archive_path = self.archive_dir / "deprecated.jsonl"
        self._archive_index_path = self.archive_dir / "deprecated_index.json"
        self.similarity_threshold = similarity_threshold

    # -- state reads -----------------------------------------------------------

    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        if not self._state_path.exists():
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._state_path)

    def state_of(self, experience_id: str) -> str:
        return self._load_state().get(experience_id, {}).get("state", ACTIVE)

    def is_active(self, experience_id: str) -> bool:
        return self.state_of(experience_id) == ACTIVE

    def active_ids(self, candidate_ids: Optional[List[str]] = None) -> List[str]:
        """IDs whose state allows participation in retrieval/induction (active only)."""
        state = self._load_state()
        ids = candidate_ids if candidate_ids is not None else list(state.keys())
        return [i for i in ids if state.get(i, {}).get("state", ACTIVE) == ACTIVE]

    # -- state transitions -------------------------------------------------------

    def mark_deprecated(
        self,
        record: Dict[str, Any],
        reason: str,
        embed: Optional[Callable[[str], List[float]]] = None,
        deprecated_at: str = "",
    ) -> Dict[str, Any]:
        """Move a record OUT of the hot bank into the compressed cold archive.

        Returns the compressed archive card. The record's content is never modified;
        only its lifecycle state flips and a compressed copy is archived for provenance.
        """
        experience_id = record.get("experience_id", "")
        state = self._load_state()
        state[experience_id] = {
            "state": DEPRECATED,
            "deprecated_at": deprecated_at,
            "reason": reason,
        }
        self._save_state(state)
        card = self._build_archive_card(record, reason, deprecated_at, embed)
        self._append_archive(card)
        return card

    # -- archive construction ---------------------------------------------------

    def _build_archive_card(
        self,
        record: Dict[str, Any],
        reason: str,
        deprecated_at: str,
        embed: Optional[Callable[[str], List[float]]],
    ) -> Dict[str, Any]:
        """Compress a full record into a provenance card (方案甲: keep the vector, drop the text)."""
        retrieval_text = record.get("retrieval_text") or record.get("title") or ""
        signature = (record.get("math_scope") or {}).get("structural_signature") or {}
        vector: Optional[List[float]] = None
        if embed is not None and retrieval_text:
            try:
                vector = embed(retrieval_text)
            except Exception:
                vector = None
        card = {
            "experience_id": record.get("experience_id", ""),
            "content_hash": record.get("content_hash", ""),  # ORIGINAL full-record hash, for exact dedup
            "layer": record.get("layer", ""),
            "polarity": record.get("polarity", ""),
            "title": record.get("title", ""),
            "summary": _summarize(retrieval_text),
            "structural_signature": signature,
            "source_episodes": list((record.get("evidence") or {}).get("source_episodes", [])),
            "created_at": record.get("created_at", ""),
            "deprecated_at": deprecated_at,
            "deprecate_reason": reason,
            "retrieval_vector": vector,  # for approximate dedup; original text NOT stored
        }
        return card

    def _append_archive(self, card: Dict[str, Any]) -> None:
        with self._archive_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
        index = self._load_archive_index()
        index.append({
            "experience_id": card["experience_id"],
            "content_hash": card["content_hash"],
            "vector": card.get("retrieval_vector"),
        })
        self._save_archive_index(index)

    def _load_archive_index(self) -> List[Dict[str, Any]]:
        if not self._archive_index_path.exists():
            return []
        try:
            data = json.loads(self._archive_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save_archive_index(self, index: List[Dict[str, Any]]) -> None:
        tmp = self._archive_index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._archive_index_path)

    def iter_archive(self) -> List[Dict[str, Any]]:
        if not self._archive_path.exists():
            return []
        cards = []
        with self._archive_path.open("rb") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    cards.append(value)
        return cards

    # -- dedup against the archive (anti-resurrection) -----------------------------

    def archive_match(
        self,
        content_hash: str,
        retrieval_text: str = "",
        embed: Optional[Callable[[str], List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the matched archive entry if a candidate is a known-deprecated experience.

        Two layers (the user's correctness point: hash alone misses reworded duplicates):
          1. exact content-hash match (verbatim resurrection)
          2. approximate cosine-similarity match on the stored vector (reworded resurrection)
        """
        index = self._load_archive_index()
        for entry in index:
            if content_hash and entry.get("content_hash") == content_hash:
                return {"match": "exact", "experience_id": entry.get("experience_id")}
        if retrieval_text and embed is not None:
            try:
                vector = embed(retrieval_text)
            except Exception:
                vector = None
            if vector:
                best = 0.0
                best_id = None
                for entry in index:
                    ev = entry.get("vector")
                    if not ev:
                        continue
                    sim = _cosine(vector, ev)
                    if sim > best:
                        best = sim
                        best_id = entry.get("experience_id")
                if best >= self.similarity_threshold:
                    return {"match": "approximate", "experience_id": best_id, "similarity": best}
        return None


__all__ = [
    "ACTIVE",
    "ACTIVE",
    "DEPRECATED",
    "LIFECYCLE_STATES",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "LifecycleStore",
]
