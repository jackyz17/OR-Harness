"""Best-effort local execution sandbox for untrusted generated Python code."""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from ..retrieval.query_builder import sanitize_feedback
from ..core.schemas import SolverExecutionResult


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


class SafePythonExecutor:
    def __init__(
        self,
        timeout_seconds: int = 120,
        solver_timeout_seconds: int = 60,
        max_stdout_chars: int = 20000,
        max_stderr_chars: int = 20000,
    ):
        self.timeout_seconds = timeout_seconds
        self.solver_timeout_seconds = solver_timeout_seconds
        self.max_stdout_chars = max_stdout_chars
        self.max_stderr_chars = max_stderr_chars

    async def execute(self, code_path: Path, workspace: Path, solver: str) -> SolverExecutionResult:
        code_path = Path(code_path).resolve()
        workspace = Path(workspace).resolve()
        if workspace not in code_path.parents:
            raise ValueError("Generated code must be inside its branch workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        security_error = self._validate_source(code_path.read_text(encoding="utf-8"))
        if security_error:
            return SolverExecutionResult(
                status="error", solver=solver, exit_code=None,
                normalized_error="security policy: " + security_error,
                message="Generated code was rejected before execution",
            )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OR_SOLVER_TIMEOUT_SECONDS": str(self.solver_timeout_seconds),
        }
        start = time.monotonic()
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _resource_limits(
                max(2, self.timeout_seconds), 2 * 1024 * 1024 * 1024, 64 * 1024 * 1024
            )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(code_path),
            cwd=str(workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return SolverExecutionResult(
                status="timeout",
                solver=solver,
                exit_code=process.returncode,
                runtime_seconds=time.monotonic() - start,
                normalized_error="execution timeout",
                message="Generated code exceeded wall-clock timeout",
            )
        stdout = sanitize_feedback(stdout_bytes.decode("utf-8", "replace"), self.max_stdout_chars)
        stderr = sanitize_feedback(stderr_bytes.decode("utf-8", "replace"), self.max_stderr_chars)
        runtime = time.monotonic() - start
        result_path = workspace / "result.json"
        if process.returncode != 0 or not result_path.exists():
            error = _normalize_error(stderr or stdout or "missing result.json")
            return SolverExecutionResult(
                status="error",
                solver=solver,
                exit_code=process.returncode,
                runtime_seconds=runtime,
                stdout=stdout,
                stderr=stderr,
                normalized_error=error,
                message="Process failed or did not write result.json",
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return SolverExecutionResult(
                status="error", solver=solver, exit_code=process.returncode,
                runtime_seconds=runtime, stdout=stdout, stderr=stderr,
                normalized_error="invalid result.json: " + type(exc).__name__,
                message="result.json is invalid",
            )
        # Normalize status to lowercase — solvers emit mixed-case (PuLP: "Optimal",
        # Gurobi adapter: integer→string). The contract is case-insensitive.
        return SolverExecutionResult(
            status=str(payload.get("status", "unknown")).lower(),
            solver=str(payload.get("solver", solver)),
            exit_code=process.returncode,
            objective_sense=str(payload.get("objective_sense", "unknown")),
            objective_value=payload.get("objective_value"),
            objective_bound=payload.get("objective_bound"),
            mip_gap=payload.get("mip_gap"),
            runtime_seconds=payload.get("runtime_seconds", runtime),
            variables=payload.get("variables", {}),
            diagnostics=payload.get("diagnostics", {}),
            message=str(payload.get("message", "")),
            stdout=stdout,
            stderr=stderr,
            result_path=str(result_path),
        )

    @staticmethod
    def _validate_source(source: str) -> Optional[str]:
        """Reject direct network/shell/parent-path access before execution."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return "SyntaxError: {}".format(exc.msg)
        # Modules that are always blocked (network, shell, process spawning).
        blocked_roots = {"subprocess", "socket", "urllib", "http", "requests", "shutil"}
        # `os` and `pathlib` are NOT blanket-blocked — solver code commonly needs
        # os.path.join / os.path.exists. Instead, we block dangerous os.* calls.
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
                    # Allow  IMDI  and os.path but block dangerous os.* submodules.
                    if root == "os" and alias.name not in ("os", "os.path"):
                        return "blocked import " + alias.name + " (only os and os.path are allowed)"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root in blocked_roots:
                    return "blocked import " + str(node.module)
                if root == "os" and module not in ("os", "os.path"):
                    return "blocked import " + str(module) + " (only os and os.path are allowed)"
                # Block pathlib entirely (use open() for result.json instead).
                if root == "pathlib":
                    return "blocked import " + str(module)
                # Block "from os import <dangerous>" — e.g. from os import system
                if module == "os":
                    for alias in node.names:
                        if alias.name in os_blocked_attrs:
                            return "blocked from os import " + alias.name
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    return "open() requires a literal branch-local result.json path"
                target = node.args[0].value
                if Path(target).is_absolute() or ".." in Path(target).parts or Path(target).name != "result.json":
                    return "open() may only access branch-local result.json"
            # Block dangerous os.* function calls (os.system, os.popen, etc.)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "os" and node.attr in os_blocked_attrs:
                    return "blocked call os.{}() — use open() for result.json I/O only".format(node.attr)
        return None


def _normalize_error(text: str) -> str:
    clean = sanitize_feedback(text, 2000)
    lines = [line for line in clean.splitlines() if line]
    exception_lines = [line for line in lines if re.search(r"(?:Error|Exception|Traceback|infeasible|unbounded)", line, re.I)]
    return " | ".join(exception_lines[-3:] or lines[-2:])[:1000]
