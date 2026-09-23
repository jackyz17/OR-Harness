"""Sandboxed execution of harness-written solve scripts.

Migrated from the legacy SafePythonExecutor (AST sandbox + POSIX rlimits +
wall-clock timeout + result.json contract), now synchronous — the harness owns
all orchestration and concurrency. The executor also performs basic
verification (status legality, finite objective, gap recording) and produces
the part of the CostVector it can actually observe.

Cost semantics — only TWO dimensions are genuinely measured here:

- ``latency_s``: the attempt's whole wall-clock span, from
  ``time.monotonic()``. It covers sandbox policy checking, environment
  construction, process spawn, interpreter start-up, imports, the solve
  itself and verification. It is an ATTEMPT overhead, not a solve latency;
  use ``solver_runtime_s`` for the solve.
- ``solver_runtime_s``: the script-reported ``runtime_seconds`` when it is
  a usable number (``solver_runtime_provenance="reported"``); otherwise the
  whole-subprocess wall clock as an EXPLICIT proxy (``"wall_proxy"``).
  A rejected or absent report is always downgraded to the proxy and noted —
  never silently zero, never pretending to be a precise solver runtime. A
  script that never ran (rejected by the sandbox policy) measures NOTHING.

Three dimensions are NOT measurable by the executor and stay UNKNOWN until
the harness declares them via ``orx record --override``:

- ``llm_tokens`` — the LLM is the outer harness's, invisible to the sandbox.
- ``tool_calls`` — ALL tool invocations within the record's declared scope
  (shell commands, file reads/writes, sandbox runs, solver calls). The
  executor sees exactly one of them (its own spawn) and records that as a
  provable LOWER BOUND (``execution_features.tool_calls_lower_bound``);
  it never claims to have measured the total.
- ``retries`` — whether THIS attempt is itself a retry is a harness
  declaration. The executor records only what it can prove: that no earlier
  attempt of the same (task, episode, strategy) exists, making retries=0 a
  fact rather than an assumption (see :meth:`api.ORHarness.execute`).

Consequently the measured-dimension mask produced here is exactly
``{latency_s, solver_runtime_s}``. A constant is never put in the mask: the
mask means "this value is a real observation", and a placeholder is not.
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

#: How many entries of a script-reported ``variables`` map are kept. A
#: solution vector is EVIDENCE a task-level check can read (integrality,
#: objective recomputation), so it is preserved rather than dropped — but a
#: model with 10^5 variables must not turn every record into a megabyte
#: blob. Truncation is always REPORTED (``solution_variables_truncated``),
#: never silent: a check that needed a dropped variable sees the key absent
#: and says so instead of passing.
MAX_SOLUTION_VARIABLES = 2000


def _solution_variables(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a script-reported ``variables`` map (or None).

    Accepts a JSON object of name -> number/bool/string; anything else (a
    list, a scalar, a missing key) is reported as NO solution rather than
    coerced into one — a check must never run against a fabricated vector.
    A nested structure is kept as-is: the check layer decides whether it
    can resolve a path into it.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    return {str(key): value for key, value in raw.items()}


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
    #: Provenance of runtime_seconds: "reported" (result.json) or
    #: "wall_proxy" (wall-clock of the whole script, explicit proxy).
    runtime_provenance: Optional[str] = None
    #: Why a script-reported runtime was not accepted (e.g. it was not a
    #: number, or was negative). None when nothing was rejected. Kept so a
    #: downgrade is always explainable, never a silent substitution.
    runtime_note: Optional[str] = None
    #: True when the script actually ran. A policy-rejected script never
    #: executed, so it measures NOTHING — not even a zero runtime.
    executed: bool = True
    wall_seconds: float = 0.0
    message: str = ""
    normalized_error: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    #: The solution vector the script reported (``result.json``'s optional
    #: ``variables`` map), or None when it reported none. This is what makes
    #: a TASK-level check possible (integer domains, objective
    #: recomputation): the solver's own feasibility verdict cannot answer
    #: "does this satisfy the original task", but the actual variable values
    #: can. Absent is reported as absent — never fabricated, never treated
    #: as an empty solution.
    variables: Optional[Dict[str, Any]] = None


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
                executed=False,
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
            # RLIMIT_CPU is set ABOVE the wall-clock timeout on purpose. When
            # the two are equal, a CPU-bound script is killed by SIGXCPU at
            # the same instant the wall deadline lands, and the race decides
            # the classification: the same "did not finish in time" condition
            # was reported as `error` (signal kill, no result.json) instead of
            # `timeout`. Giving the wall clock the first move makes both a
            # CPU-bound burn and a sleep-bound stall classify as `timeout`.
            kwargs["preexec_fn"] = _resource_limits(
                max(2, self.timeout_seconds + 5), 2 * 1024 * 1024 * 1024,
                64 * 1024 * 1024)
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
                normalized_error=_normalize_error(
                    stderr or stdout or _exit_note(proc.returncode)),
                message="Process failed or did not write result.json")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionOutcome(
                status="error", solver=solver, exit_code=proc.returncode,
                wall_seconds=wall, stdout=stdout, stderr=stderr,
                normalized_error="invalid result.json: " + type(exc).__name__,
                message="result.json is invalid")
        # Solver runtime: prefer the script-reported value; otherwise fall
        # back to wall-clock as an EXPLICIT proxy (never silently zero,
        # never masquerading as a precise solver runtime). A reported value
        # that is not a usable number (non-numeric, negative, NaN/inf) is
        # REJECTED and downgraded to the proxy with a note — the executor
        # must not crash on a malformed script and must not record a
        # nonsense measurement.
        reported = payload.get("runtime_seconds")
        runtime_note: Optional[str] = None
        runtime_seconds: Optional[float] = None
        runtime_provenance: Optional[str] = None
        if reported is not None:
            try:
                candidate = float(reported)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and _is_finite(candidate) and candidate >= 0.0:
                runtime_seconds = candidate
                runtime_provenance = "reported"
            else:
                runtime_note = (f"rejected script-reported runtime_seconds="
                                f"{reported!r}; using the wall-clock proxy")
        if runtime_seconds is None:
            runtime_seconds = wall
            runtime_provenance = "wall_proxy"
        return ExecutionOutcome(
            status=str(payload.get("status", "unknown")).lower(),
            solver=str(payload.get("solver", solver)),
            exit_code=proc.returncode,
            objective_sense=str(payload.get("objective_sense", "unknown")),
            objective_value=payload.get("objective_value"),
            objective_bound=payload.get("objective_bound"),
            mip_gap=payload.get("mip_gap"),
            runtime_seconds=runtime_seconds,
            runtime_provenance=runtime_provenance,
            runtime_note=runtime_note,
            wall_seconds=wall,
            message=str(payload.get("message", "")),
            diagnostics=dict(payload.get("diagnostics") or {}),
            variables=_solution_variables(payload.get("variables")),
            stdout=stdout, stderr=stderr)

    # -- verification (infrastructure, not a research contribution) -----------

    @staticmethod
    def verify(outcome: ExecutionOutcome) -> Dict[str, Any]:
        """Basic, cheap verification: legal status, finite objective when
        claimed, gap recorded when a bound exists. ``verification=strong`` is
        an optional outer-layer upgrade; this layer stays cheap by default.

        ``runtime_checks`` is reported SEPARATELY from ``problems`` on
        purpose: ``problems`` feeds ``quality.feasible``, and a suspicious
        runtime is a measurement-quality observation, not evidence that the
        solve itself was wrong. Folding it into ``problems`` would flip
        perfectly valid records to infeasible and poison every downstream
        consumer of ``quality``."""
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
        runtime_checks: List[str] = []
        if not outcome.executed:
            runtime_checks.append("the script never ran: no runtime was observed")
        else:
            runtime = outcome.runtime_seconds
            if not _is_finite(runtime):
                runtime_checks.append("solver_runtime_s is not finite")
            elif runtime < 0.0:
                runtime_checks.append("solver_runtime_s is negative")
            elif (outcome.runtime_provenance == "reported"
                  and runtime > outcome.wall_seconds):
                # A script cannot report more solve time than the whole
                # process it ran in. Only "reported" is checked: a
                # wall_proxy value IS the wall clock by construction.
                runtime_checks.append(
                    "script-reported runtime exceeds the whole process "
                    "wall clock")
        return {
            "feasible": feasible and not problems,
            "objective": objective,
            "bound": outcome.objective_bound,
            "gap": gap,
            "status": status if status in ALLOWED_STATUSES else "error",
            "problems": problems,
            "runtime_checks": runtime_checks,
        }

    # -- record assembly ----------------------------------------------------------

    def execute(self, code_path: Path, workspace: Path, *, solver: str,
                task_id: str, strategy_id: str, profile: ProblemProfile,
                verification_level: str = "basic",
                code_hash: Optional[str] = None) -> ExecutionRecord:
        """Run once, verify, meter cost, and assemble an ExecutionRecord.

        The record is returned, not persisted — recording is the harness's
        explicit decision (``orx record``), keeping execute/record separable.

        Cost semantics: this record covers ONE attempt. Only what the
        executor can OBSERVE enters the measured mask (``latency_s`` and, when
        the script really ran, ``solver_runtime_s``). ``tool_calls``,
        ``retries`` and ``llm_tokens`` are harness declarations — the
        executor records what it can prove about them instead of fabricating
        a measured constant:

        - ``tool_calls`` stays unmeasured and its provable lower bound (one
          sandbox invocation) goes to ``tool_calls_lower_bound``;
        - ``retries`` stays unmeasured (whether this attempt is itself a
          retry is not observable here — see ``api.ORHarness.execute`` for
          the provable-zero case);
        - ``llm_tokens`` stays unmeasured until ``record --override``.
        """
        started = time.monotonic()
        outcome = self.run(code_path, workspace, solver)
        check = self.verify(outcome)
        # monotonic, not time.time(): a wall clock can jump backwards (NTP),
        # which would record a negative latency for a perfectly normal run.
        latency = time.monotonic() - started
        if not outcome.executed:
            # A policy-rejected script never ran: there is no runtime to
            # measure and no call to count. Reporting 0.0 as a MEASURED
            # runtime would be a fabricated fact.
            runtime = 0.0
            runtime_provenance = None
            measured: set = set()
        else:
            if outcome.status in ("error", "timeout"):
                runtime = outcome.wall_seconds
                runtime_provenance = "wall_proxy"
            else:
                runtime = outcome.runtime_seconds
                runtime_provenance = outcome.runtime_provenance or "wall_proxy"
            measured = {"latency_s", "solver_runtime_s"}
        cost = CostVector(
            llm_tokens=0.0,   # harness declares via record --override
            tool_calls=0.0,   # harness declares; executor only proves a floor
            solver_runtime_s=runtime,
            retries=0.0,      # harness declares (see api.execute for the proof)
            latency_s=latency,
            measured=measured,
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
        execution_features: Dict[str, Any] = {}
        if outcome.diagnostics:
            execution_features["solver_diagnostics"] = dict(outcome.diagnostics)
        if outcome.variables is not None:
            # The solution vector is the ONLY evidence a task-level check
            # (integer domain, objective recomputation, per-value probes)
            # can read. Truncation is reported explicitly so a check that
            # needed a dropped variable stays insufficient instead of
            # silently passing.
            kept = dict(list(outcome.variables.items())[:MAX_SOLUTION_VARIABLES])
            execution_features["solution_variables"] = kept
            if len(outcome.variables) > len(kept):
                execution_features["solution_variables_truncated"] = {
                    "n_reported": len(outcome.variables),
                    "n_kept": len(kept),
                    "dropped": sorted(set(outcome.variables) - set(kept)),
                    "note": ("the solution vector exceeded the storage limit; "
                             "a check needing a dropped variable will report "
                             "it as unchecked"),
                }
        if outcome.executed:
            # The one call this executor can PROVE happened. The total
            # ``tool_calls`` (all tool invocations in the declared scope) is
            # the harness's to declare, and must never be below this floor.
            execution_features["tool_calls_lower_bound"] = 1
        cost_notes: List[str] = []
        if outcome.runtime_note:
            cost_notes.append(outcome.runtime_note)
        if check["runtime_checks"]:
            execution_features["runtime_checks"] = list(check["runtime_checks"])
            cost_notes.extend(check["runtime_checks"])
        if not outcome.executed:
            cost_notes.append(
                "the solve script was rejected before execution: no cost "
                "dimension was measured for this record")
        if cost_notes:
            execution_features["cost_notes"] = cost_notes
        return ExecutionRecord(
            execution_id=ExecutionRecord.new_id(),
            task_id=task_id, strategy_id=strategy_id,
            profile_snapshot=profile, trajectory=trajectory,
            quality={"feasible": check["feasible"], "objective": check["objective"],
                     "gap": check["gap"], "status": check["status"],
                     "problems": check["problems"]},
            cost=cost, failures=failures,
            solver={"name": outcome.solver, "code_hash": digest},
            execution_features=execution_features,
            verification_level=verification_level,
            measurement_scope="attempt",
            solver_runtime_provenance=runtime_provenance)

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


#: Signals worth naming explicitly: a bare "missing result.json" tells the
#: harness nothing about WHY the script produced no result.
_SIGNAL_NAMES = {9: "SIGKILL", 15: "SIGTERM", 24: "SIGXCPU", 25: "SIGXFSZ"}


def _exit_note(returncode: Optional[int]) -> str:
    """Explain a non-zero exit when the process wrote nothing at all."""
    if returncode is None:
        return "missing result.json"
    if returncode < 0:
        name = _SIGNAL_NAMES.get(-returncode)
        return (f"missing result.json: process killed by signal {-returncode}"
                + (f" ({name})" if name else ""))
    return f"missing result.json (exit code {returncode})"


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value and abs(value) != float("inf")


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
