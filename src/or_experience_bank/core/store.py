"""Append-only JSONL source of truth with process-safe locking."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .schemas import ExperienceAppendResult, ExperienceLayer, ExperienceRecord

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the in-process lock
    fcntl = None  # type: ignore


class ExperienceStoreError(RuntimeError):
    pass


class CorruptExperienceBank(ExperienceStoreError):
    pass


_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def canonical_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return immutable semantic content used for exact duplicate detection."""
    ignored = {"experience_id", "created_at", "content_hash", "possible_duplicate_of"}
    return {key: value for key, value in record.items() if key not in ignored}


def compute_content_hash(record: Dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_payload(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_line(record: Dict[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


class AppendOnlyExperienceStore:
    """Store immutable ExperienceRecord lines; only append and read operations exist."""

    def __init__(self, bank_home: Path, on_append=None, lifecycle=None, embed=None):
        self.bank_home = Path(bank_home)
        self.bank_dir = self.bank_home / "bank"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.bank_dir / ".append.lock"
        self._lock_path.touch(exist_ok=True)
        self.on_append = on_append
        # Optional anti-resurrection: a LifecycleStore + embedding callable. When present,
        # append() rejects a candidate that matches a deprecated archive entry (exact hash
        # or approximate embedding similarity), so a retired harmful experience cannot
        # re-enter the bank verbatim or reworded.
        self.lifecycle = lifecycle
        self.embed = embed
        for layer in ExperienceLayer:
            self.layer_path(layer.value).touch(exist_ok=True)

    def layer_path(self, layer: str) -> Path:
        ExperienceLayer(layer)
        return self.bank_dir / (layer + ".jsonl")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        key = str(self._lock_path.resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with thread_lock:
            with self._lock_path.open("a+b") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def append(self, record: ExperienceRecord) -> ExperienceAppendResult:
        data = record.to_dict()
        data["content_hash"] = compute_content_hash(data)
        from ..solving.validator import ExperienceValidator

        report = ExperienceValidator().validate(data)
        if not report.valid:
            raise ExperienceStoreError("Invalid experience: " + "; ".join(report.errors))
        path = self.layer_path(data["layer"])
        with self._locked():
            duplicate_id = self._find_hash_unlocked(data["content_hash"])
            if duplicate_id:
                return ExperienceAppendResult(
                    status="duplicate",
                    experience_id=record.experience_id,
                    content_hash=data["content_hash"],
                    layer=data["layer"],
                    duplicate_of=duplicate_id,
                )
            # anti-resurrection: reject candidates matching a deprecated archive entry
            if self.lifecycle is not None:
                archived = self.lifecycle.archive_match(
                    content_hash=data["content_hash"],
                    retrieval_text=data.get("retrieval_text", ""),
                    embed=self.embed,
                )
                if archived:
                    return ExperienceAppendResult(
                        status="rejected_deprecated",
                        experience_id=record.experience_id,
                        content_hash=data["content_hash"],
                        layer=data["layer"],
                        duplicate_of=archived.get("experience_id"),
                    )
            with path.open("ab", buffering=0) as handle:
                handle.write(canonical_line(data))
                handle.flush()
                os.fsync(handle.fileno())
        record.content_hash = data["content_hash"]
        result = ExperienceAppendResult(
            status="appended",
            experience_id=record.experience_id,
            content_hash=record.content_hash,
            layer=record.layer,
        )
        if self.on_append:
            self.on_append(record.layer)
        return result

    def _find_hash_unlocked(self, content_hash: str) -> Optional[str]:
        for layer in ExperienceLayer:
            for row in self._iter_path(self.layer_path(layer.value), strict=False):
                if row.get("content_hash") == content_hash:
                    return str(row.get("experience_id"))
        return None

    def iter_records(self, layer: Optional[str] = None, strict: bool = True) -> Iterator[Dict[str, Any]]:
        layers = [ExperienceLayer(layer)] if layer else list(ExperienceLayer)
        for item in layers:
            yield from self._iter_path(self.layer_path(item.value), strict=strict)

    def _iter_path(self, path: Path, strict: bool) -> Iterator[Dict[str, Any]]:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if strict:
                        raise CorruptExperienceBank(
                            "Invalid JSONL at {}:{}: {}".format(path, line_number, exc)
                        ) from exc
                    continue
                if not isinstance(value, dict):
                    if strict:
                        raise CorruptExperienceBank("Non-object JSONL at {}:{}".format(path, line_number))
                    continue
                yield value

    def stats(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {"layers": {}, "total": 0}
        for layer in ExperienceLayer:
            rows = list(self.iter_records(layer.value))
            polarity: Dict[str, int] = {}
            generality: Dict[str, int] = {}
            solvers: Dict[str, int] = {}
            families: Dict[str, int] = {}
            for row in rows:
                polarity[row["polarity"]] = polarity.get(row["polarity"], 0) + 1
                gen = row["scope"]["generality"]
                generality[gen] = generality.get(gen, 0) + 1
                solver = row["scope"].get("solver") or "agnostic"
                solvers[solver] = solvers.get(solver, 0) + 1
                fam = row["problem_context"]["problem_family"]
                families[fam] = families.get(fam, 0) + 1
            output["layers"][layer.value] = {
                "count": len(rows),
                "polarity": polarity,
                "generality": generality,
                "solver": solvers,
                "problem_family": families,
            }
            output["total"] += len(rows)
        return output

    def validate_bank(self) -> Dict[str, Any]:
        from ..solving.validator import ExperienceValidator

        errors: List[str] = []
        ids = set()
        hashes = set()
        count = 0
        for layer in ExperienceLayer:
            path = self.layer_path(layer.value)
            try:
                rows = list(self._iter_path(path, strict=True))
            except CorruptExperienceBank as exc:
                errors.append(str(exc))
                continue
            for row in rows:
                count += 1
                report = ExperienceValidator().validate(row)
                errors.extend("{}: {}".format(path.name, err) for err in report.errors)
                if row.get("experience_id") in ids:
                    errors.append("duplicate experience_id: " + str(row.get("experience_id")))
                ids.add(row.get("experience_id"))
                expected = compute_content_hash(row)
                if row.get("content_hash") != expected:
                    errors.append("invalid content_hash: " + str(row.get("experience_id")))
                if row.get("content_hash") in hashes:
                    errors.append("duplicate content_hash: " + str(row.get("content_hash")))
                hashes.add(row.get("content_hash"))
        return {"valid": not errors, "record_count": count, "errors": errors}
