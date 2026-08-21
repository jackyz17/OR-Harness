"""Independent append-only store for the Modeling Bank (Phase 2 step 4, Option 3, D10).

The modeling bank uses the ModelingExperience schema (all records are peers:
directly-solved with status=null, induced with status=validated). Per Option 3 it
gets its own store so the shared store/retrieval/validator for the flat layers
stays untouched.

Facts are append-only JSONL at bank/modeling_bank.jsonl. Mutable stats never touch a
written line. Exact-duplicate rejection uses the ModelingExperience content hash.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .modeling_schemas import ModelingExperience

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


class ModelingStoreError(RuntimeError):
    pass


_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_GUARD = threading.Lock()


class ModelingStore:
    """Append-only store for ModelingExperience records (realizations and patterns)."""

    def __init__(self, bank_home: Path):
        self.bank_home = Path(bank_home)
        self.bank_dir = self.bank_home / "bank"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.bank_dir / "modeling_bank.jsonl"
        self._path.touch(exist_ok=True)
        self._lock_path = self.bank_dir / ".modeling_append.lock"
        self._lock_path.touch(exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        key = str(self._lock_path.resolve())
        with _GUARD:
            lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with lock:
            with self._lock_path.open("a+b") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(self, record: ModelingExperience) -> Dict[str, Any]:
        """Validate, dedupe, and append a ModelingExperience. Never modifies old lines."""
        record.validate()
        if not record.content_hash:
            record.compute_content_hash()
        with self._locked():
            duplicate_id = self._find_hash(record.content_hash)
            if duplicate_id:
                return {
                    "status": "duplicate",
                    "experience_id": record.experience_id,
                    "content_hash": record.content_hash,
                    "duplicate_of": duplicate_id,
                }
            line = (json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            with self._path.open("ab", buffering=0) as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return {
            "status": "appended",
            "experience_id": record.experience_id,
            "content_hash": record.content_hash,
        }

    def _find_hash(self, content_hash: str) -> Optional[str]:
        for row in self.iter_records():
            if row.get("content_hash") == content_hash:
                return row.get("experience_id")
        return None

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

    def all_records(self) -> List[Dict[str, Any]]:
        """All modeling-bank records (peers, no depth hierarchy)."""
        return list(self.iter_records())

    def validated_records(self) -> List[Dict[str, Any]]:
        """Records with status='validated' (passed unseen transfer)."""
        return [r for r in self.iter_records() if r.get("status") == "validated"]

    def stats(self) -> Dict[str, Any]:
        records = list(self.iter_records())
        return {
            "total": len(records),
            "validated": len([r for r in records if r.get("status") == "validated"]),
            "path": str(self._path),
        }


__all__ = ["ModelingStore", "ModelingStoreError"]
