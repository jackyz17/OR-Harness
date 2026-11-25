"""Explicit task inputs, versioned in the existing content-addressed store.

Declarations authorize files, never a directory scan. The runner is responsible
for supplying only this task's public inputs (not references or hidden checkers).
"""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath

from or_harness.core.storage import StorageError


class PreparedTask(dict):
    """Process-local marker; a serialized manifest is revalidated on entry."""


def declarations(task, *, base_dir=None):
    raw = task.get("input_resources", [])
    if not isinstance(raw, list):
        raise ValueError("task.input_resources must be a list of {source, path} files")
    out, names = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each input resource must be an object with source and path")
        name, source = item.get("path"), item.get("source")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("input path must be a nonempty POSIX relative file path")
        path = PurePosixPath(name)
        if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("input path must be relative without traversal or empty components")
        if name in names or any(name.startswith(n + "/") or n.startswith(name + "/") for n in names):
            raise ValueError("input paths must be unique files without parent/child collisions")
        if not isinstance(source, str) or not source or "\x00" in source or "\x00" in name:
            raise ValueError("input source must name one regular file")
        # abspath deliberately does NOT resolve symlinks: opening below rejects
        # symlinks in every component, including a symlinked parent directory.
        origin = os.path.abspath(os.path.join(str(base_dir or Path.cwd()), source))
        row = {"path": name, "source": origin}
        if item.get("sha256") is not None:
            version = item["sha256"]
            if not isinstance(version, str) or len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
                raise ValueError("input sha256 must be a lowercase 64-character content digest")
            row["sha256"] = version
        out.append(row)
        names.add(name)
    return out


def open_regular(path):
    """Pin a regular file without following symlinks, even in its parents."""
    path = Path(os.path.abspath(path))
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError("input source must be a regular file (no devices, pipes or directories)")
    return os.fdopen(fd, "rb")


def _digest(stream, output=None):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), size


def prepare_task_inputs(task, home):
    """Freeze sources once per preparation; no file bodies enter task JSON."""
    if isinstance(task, PreparedTask):
        return task
    if not task.get("input_resources"):
        declarations(task)  # reject malformed non-list declarations, including {}
        return task
    rows = []
    staging = Path(home) / "artifacts"
    staging.mkdir(parents=True, exist_ok=True)
    for row in declarations(task):
        with tempfile.NamedTemporaryFile(dir=staging, prefix="input-", delete=False) as copied:
            try:
                with open_regular(row["source"]) as source:
                    version, size = _digest(source, copied)
                if row.get("sha256") not in (None, version):
                    raise ValueError("input content does not match declared sha256: " + row["path"])
                relative = Path("artifacts") / version[:2] / version
                target = Path(home) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                copied.flush()
                os.fchmod(copied.fileno(), 0o444)
                try:
                    # Publish complete bytes atomically. Concurrent readers
                    # never see a half-written content-addressed input.
                    os.link(copied.name, target)
                except FileExistsError:
                    with open_regular(target) as stored:
                        if _digest(stored) != (version, size):
                            raise StorageError("immutable input artifact integrity failure")
            finally:
                Path(copied.name).unlink(missing_ok=True)
        rows.append({**row, "sha256": version, "bytes": size,
                     "artifact": {"path": relative.as_posix(), "sha256": version,
                                  "bytes": size, "kind": "task_input",
                                  "source": row["source"]}})
    return PreparedTask({**task, "input_resources": rows})


def execution_task_inputs(task, frozen):
    """Use the forecast's manifest for identity; verify bytes at submission.

    Missing/changed source files are submission outcomes after binding, not
    vanished attempts. A changed declaration is an ordinary identity refusal.
    """
    current = declarations(task)
    if len(current) != len(frozen):
        raise ValueError("task input declarations changed since prediction; plan again")
    for row, saved in zip(current, frozen):
        if any(row[k] != saved.get(k) for k in ("path", "source")) or row.get("sha256") not in (None, saved.get("sha256")):
            raise ValueError("task input declarations changed since prediction; plan again")
    return PreparedTask({**task, "input_resources": frozen}) if frozen else task


def verify_execution_inputs(resources, home):
    """Recheck origin and immutable store; return only exact files to mount."""
    declarations({"input_resources": resources})
    if resources and home is None:
        raise ValueError("declared inputs require their artifact home")
    mounts = []
    for item in resources:
        if not item.get("sha256") or not isinstance(item.get("bytes"), int) or isinstance(item.get("bytes"), bool) or item["bytes"] < 0:
            raise ValueError("execution inputs require a frozen sha256 and byte count")
        expected = (item["sha256"], item["bytes"])
        with open_regular(item["source"]) as source:
            if _digest(source) != expected:
                raise ValueError("input content changed since prediction: " + item["path"])
        target = Path(home) / "artifacts" / item["sha256"][:2] / item["sha256"]
        with open_regular(target) as stored:
            if _digest(stored) != expected:
                raise StorageError("immutable input artifact integrity failure: " + item["path"])
        mounts.append({"path": item["path"], "source": str(target.absolute())})
    return mounts
