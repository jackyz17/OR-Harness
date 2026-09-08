"""Sandboxed execution of harness-written solve scripts.

Migrated from the legacy SafePythonExecutor (AST sandbox + POSIX rlimits +
wall-clock timeout + result.json contract), now synchronous — the harness owns
all orchestration and concurrency. The executor also performs basic
verification (status legality, finite objective, gap recording) and produces
the execution half of the CostVector (solver_runtime_s, retries, latency_s,
tool_calls). ``llm_tokens`` is owned by the harness and backfilled via
``orx record --override``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    FailureRecord,
    ProblemProfile,
    TrajectoryStep,
)

ALLOWED_STATUSES = ("optimal", "feasible", "infeasible", "unbounded", "timeout", "error")


def _resource_limits(cpu_seconds: int, memory_bytes: int, file_bytes: int):
    def apply_limits() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        except (ImportError, OSError, ValueError):
            return

    return apply_limits


@dataclass
class ExecutionOutcome:
    """Raw result of one sandboxed attempt."""

    status: str
    solver: str
    exit_code: Optional[int] = None
    objective_sense: str = "unknown"
    objective_value: Optional[float] = None
    objective_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_seconds: float = 0.0
    wall_seconds: float = 0.0
    message: str = ""
    normalized_error: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


class SafePythonExecutor:
    """Best-effort local sandbox for harness-written Python code."""

    def __init__(self, timeout_seconds: int = 120, solver_timeout_seconds: int = 60,
                 max_stdout_chars: int = 20000, max_stderr_chars: int = 20000):
        self.timeout_seconds = timeout_seconds
        self.solver_timeout_seconds = solver_timeout_seconds
        self.max_stdout_chars = max_stdout_chars
        self.max_stderr_chars = max_stderr_chars

    # -- execution ------------------------------------------------------------

    def run(self, code_path: Path, workspace: Path, solver: str) -> ExecutionOutcome:
        code_path = Path(code_path).resolve()
        workspace = Path(workspace).resolve()
        if workspace not in code_path.parents and code_path != workspace:
            raise ValueError("Solve script must live inside its execution workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        security_error = self.validate_source(code_path.read_text(encoding="utf-8"))
        if security_error:
            return ExecutionOutcome(
                status="error", solver=solver,
                normalized_error="security policy: " + security_error,
                message="Solve script was rejected before execution")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OR_SOLVER_TIMEOUT_SECONDS": str(self.solver_timeout_seconds),
        }
        start = time.monotonic()
        kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _resource_limits(
                max(2, self.timeout_seconds), 2 * 1024 * 1024 * 1024, 64 * 1024 * 1024)
        try:
            proc = subprocess.run(
                [sys.executable, str(code_path)],
                cwd=str(workspace), env=env, capture_output=True,
                timeout=self.timeout_seconds, **kwargs)
        except subprocess.TimeoutExpired:
            return ExecutionOutcome(
                status="timeout", solver=solver,
                wall_seconds=time.monotonic() - start,
                normalized_error="execution timeout",
                message="Solve script exceeded the wall-clock timeout")
        wall = time.monotonic() - start
        stdout = _clip(proc.stdout.decode("utf-8", "replace"), self.max_stdout_chars)
        stderr = _clip(proc.stderr.decode("utf-8", "replace"), self.max_stderr_chars)
        result_path = workspace / "result.json"
        if proc.returncode != 0 or not result_path.exists():
            return ExecutionOutcome(
                status="error", solver=solver, exit_code=proc.returncode,
                wall_seconds=wall, stdout=stdout, stderr=stderr,
                normalized_error=_normalize_error(stderr or stdout or "missing result.json"),
                message="Process failed or did not write result.json")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionOutcome(
                status="error", solver=solver, exit_code=proc.returncode,
                wall_seconds=wall, stdout=stdout, stderr=stderr,
                normalized_error="invalid result.json: " + type(exc).__name__,
                message="result.json is invalid")
        return ExecutionOutcome(
            status=str(payload.get("status", "unknown")).lower(),
            solver=str(payload.get("solver", solver)),
            exit_code=proc.returncode,
            objective_sense=str(payload.get("objective_sense", "unknown")),
            objective_value=payload.get("objective_value"),
            objective_bound=payload.get("objective_bound"),
            mip_gap=payload.get("mip_gap"),
            runtime_seconds=float(payload.get("runtime_seconds") or wall),
            wall_seconds=wall,
            message=str(payload.get("message", "")),
            diagnostics=dict(payload.get("diagnostics") or {}),
            stdout=stdout, stderr=stderr)

    # -- verification (infrastructure, not a research contribution) -----------

    @staticmethod
    def verify(outcome: ExecutionOutcome) -> Dict[str, Any]:
        """Basic, cheap verification: legal status, finite objective when
        claimed, gap recorded when a bound exists. ``verification=strong`` is
        an optional outer-layer upgrade; this layer stays cheap by default."""
        problems: List[str] = []
        status = outcome.status
        if status not in ALLOWED_STATUSES:
            problems.append(f"illegal status {status!r}")
        feasible = status in ("optimal", "feasible")
        objective = outcome.objective_value
        if feasible:
            if objective is None:
                problems.append("feasible status without objective_value")
            elif isinstance(objective, (int, float)) and not _is_finite(objective):
                problems.append("objective_value is not finite")
        gap: Optional[float] = outcome.mip_gap
        if gap is None and feasible and _is_finite(outcome.objective_bound) and _is_finite(objective):
            bound, obj = float(outcome.objective_bound), float(objective)
            denom = max(abs(obj), 1e-9)
            gap = abs(bound - obj) / denom
        return {
            "feasible": feasible and not problems,
            "objective": objective,
            "bound": outcome.objective_bound,
            "gap": gap,
            "status": status if status in ALLOWED_STATUSES else "error",
            "problems": problems,
        }

    # -- record assembly ----------------------------------------------------------

    def execute(self, code_path: Path, workspace: Path, *, solver: str,
                task_id: str, strategy_id: str, profile: ProblemProfile,
                verification_level: str = "basic",
                code_hash: Optional[str] = None) -> ExecutionRecord:
        """Run once, verify, meter cost, and assemble an ExecutionRecord.

        The record is returned, not persisted — recording is the harness's
        explicit decision (``orx record``), keeping execute/record separable.
        """
        started = time.time()
        outcome = self.run(code_path, workspace, solver)
        check = self.verify(outcome)
        retries = 1.0 if outcome.status in ("error", "timeout") else 0.0
        latency = time.time() - started
        cost = CostVector(
            llm_tokens=0.0,  # harness backfills via record --override
            tool_calls=1.0,
            solver_runtime_s=outcome.runtime_seconds,
            retries=retries,
            latency_s=latency,
        )
        failures: List[FailureRecord] = []
        if outcome.status in ("error", "timeout"):
            failures.append(FailureRecord(
                attempt=1, error=outcome.normalized_error or outcome.message,
                recovery_action=None))
        trajectory = [TrajectoryStep(
            action=f"execute:{strategy_id} via {solver}",
            outcome=outcome.status, duration_s=outcome.wall_seconds)]
        if check["problems"]:
            trajectory.append(TrajectoryStep(
                action="verify:basic", outcome="; ".join(check["problems"])))
        digest = code_hash or _sha256_file(code_path)
        return ExecutionRecord(
            execution_id=ExecutionRecord.new_id(),
            task_id=task_id, strategy_id=strategy_id,
            profile_snapshot=profile, trajectory=trajectory,
            quality={"feasible": check["feasible"], "objective": check["objective"],
                     "gap": check["gap"], "status": check["status"],
                     "problems": check["problems"]},
            cost=cost, failures=failures,
            solver={"name": outcome.solver, "code_hash": digest},
            verification_level=verification_level)

    # -- static sandbox policy -------------------------------------------------------

    @staticmethod
    def validate_source(source: str) -> Optional[str]:
        """Reject network/shell/parent-path access before execution."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg}"
        blocked_roots = {"subprocess", "socket", "urllib", "http", "requests", "shutil"}
        os_blocked_attrs = {
            "system", "popen", "popen2", "popen3", "popen4",
            "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
            "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
            "fork", "kill", "killpg",
            "remove", "removedirs", "unlink", "rmdir",
            "listdir", "walk", "scandir",
            "chmod", "chown", "chroot", "chdir", "fchdir",
            "symlink", "link", "rename", "renames",
            "umask", "setuid", "setgid", "seteuid", "setegid",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in blocked_roots:
                        return "blocked import " + alias.name
                    if root == "os" and alias.name not in ("os", "os.path"):
                        return ("blocked import " + alias.name +
                                " (only os and os.path are allowed)")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root in blocked_roots:
                    return "blocked import " + str(module)
                if root == "os" and module not in ("os", "os.path"):
                    return ("blocked import " + str(module) +
                            " (only os and os.path are allowed)")
                if root == "pathlib":
                    return "blocked import " + str(module)
                if module == "os":
                    for alias in node.names:
                        if alias.name in os_blocked_attrs:
                            return "blocked from os import " + alias.name
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "open"):
                if (not node.args or not isinstance(node.args[0], ast.Constant)
                        or not isinstance(node.args[0].value, str)):
                    return ("open() requires a LITERAL string path (dynamic paths "
                            "are blocked); write your result with "
                            "open('result.json', 'w') in the execution workspace")
                target = node.args[0].value
                if (Path(target).is_absolute() or ".." in Path(target).parts
                        or Path(target).name != "result.json"):
                    return ("open() may only access the workspace-local file named "
                            "result.json (relative path, no directories)")
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "os" and node.attr in os_blocked_attrs:
                    return ("blocked call os.{}() — use open('result.json') for I/O"
                            .format(node.attr))
        return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    return text[:head] + "\n... [truncated] ...\n" + text[-(limit - head):]


def _normalize_error(text: str) -> str:
    clean = _clip(text, 2000)
    lines = [line for line in clean.splitlines() if line]
    exception_lines = [l for l in lines
                       if re.search(r"(?:Error|Exception|Traceback|infeasible|unbounded)", l, re.I)]
    return " | ".join(exception_lines[-3:] or lines[-2:])[:1000]


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value and abs(value) != float("inf")


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
