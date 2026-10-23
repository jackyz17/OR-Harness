"""Immutable local attempt artifacts, outside SQLite JSON payloads."""
import hashlib
from pathlib import Path

from or_harness.core.storage import StorageError


def capture(home, data, *, kind, source):
    if isinstance(data, str):
        data = data.encode("utf-8")
    version = hashlib.sha256(data).hexdigest()
    relative = Path("artifacts") / version[:2] / version
    target = Path(home) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as output:
            output.write(data)
    except FileExistsError:
        if hashlib.sha256(target.read_bytes()).hexdigest() != version:
            raise StorageError(f"immutable artifact integrity failure: {relative}")
    return {"path": relative.as_posix(), "sha256": version, "bytes": len(data),
            "kind": kind, "source": source}


def read(home, reference):
    version = reference["sha256"]
    if len(version) != 64 or any(char not in "0123456789abcdef" for char in version):
        raise StorageError("invalid artifact content version")
    target = Path(home) / "artifacts" / version[:2] / version
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != version:
        raise StorageError("immutable artifact integrity failure")
    return data
