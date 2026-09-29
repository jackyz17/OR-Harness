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
    normalize_method,
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


def _config_receipt(workspace: Path,
                    action_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read the script's CONFIG RECEIPT out of ``result.json``, or None.

    The receipt is the ``config`` object a script reports for the
    parameters that really took effect (a time limit it applied, a gap
    target it used, a seed it fixed). Two rules keep it from lying:

    * it must be stamped with THIS attempt's ``action_id`` under
      ``config.action_id``. The executor passes the id through
      ``OR_ACTION_ID``; a receipt whose stamp does not match is a
      leftover from an earlier run in the same workspace and is refused.
      A receipt with NO stamp is refused too — an unattributable config is
      not this attempt's configuration;
    * it is read on the FAILURE paths as well (timeout, error, invalid
      result.json) by the same function, so a run that set its parameters
      and then failed still reports them — "no result" is not "no config".

    Returns ``{"values": {...}, "source": {key: origin}}`` or None. The
    ``action_id`` stamp itself is removed from ``values`` (it is the
    receipt's identity, not a script parameter).
    """
    if not action_id:
        return None
    path = Path(workspace) / "result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        return None
    if str(config.get("action_id") or "") != str(action_id):
        return None
    values = {str(k): v for k, v in config.items() if k != "action_id"}
    if not values:
        return None
    return {"values": values}


def _method_receipt(workspace: Path,
                    action_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read the script's METHOD RECEIPT out of ``result.json``, or None.

    The receipt is the optional ``method_performed`` object: the method the
    solve script says it ACTUALLY carried out — ``{"name", "steps": [...],
    "why"?, "fallback"?}`` — which may differ from the candidate's plan (a
    fallback taken, a step dropped). It obeys the same two rules as the
    config receipt and for the same reason:

    * it must be stamped with THIS attempt's ``action_id``
      (``method_performed.action_id``); a receipt without the stamp, or
      with a different one, is a leftover from an earlier run in the same
      workspace and is REFUSED;
    * it is read on the FAILURE paths too, so a run that started its method
      and then failed still reports what it did — "no result" is not "no
      method".

    The method is normalized with :func:`normalize_method` so it shares the
    planned method's shape. Returns ``{"method": {...}, "action_id": str}``
    or None. An empty/blank method is reported as NO receipt (a method of
    nothing is not evidence of anything).
    """
    if not action_id:
        return None
    path = Path(workspace) / "result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("method_performed")
    if not isinstance(raw, dict) or not raw:
        return None
    if str(raw.get("action_id") or "") != str(action_id):
        return None
    method = normalize_method({k: v for k, v in raw.items()
                               if k != "action_id"})
    if method is None:
        return None
    return {"method": method, "action_id": str(action_id)}


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
    #: Solution variables the script reported (``result.json``'s optional
    #: ``variables`` map), or None when it reported none. This is what makes
    #: a TASK-level check possible (integer domains, objective
    #: recomputation): the solver's own feasibility verdict cannot answer
    #: "does this satisfy the original task", but the actual variable values
    #: can. Absent is reported as absent — never fabricated, never treated
    #: as an empty solution.
    variables: Optional[Dict[str, Any]] = None
    #: The script's CONFIG RECEIPT: the parameters that actually took
    #: effect, read back from ``result.json``'s optional ``config`` object
    #: when it is stamped with THIS attempt's ``action_id`` (the executor
    #: passes the id through the ``OR_ACTION_ID`` environment variable).
    #: None when the script reported no config, or the receipt's stamp does
    #: not match — a leftover ``result.json`` from an earlier run in the
    #: same workspace is stale and is never read as this attempt's
    #: configuration. Absent means UNKNOWN, never a copied prediction.
    config_report: Optional[Dict[str, Any]] = None
    #: True when a ``result.json`` left by an earlier run was removed before
    #: this attempt started. Recorded so a reader can see that the workspace
    #: was cleared (an old result could otherwise have been read as this
    #: run's body).
    stale_result_cleared: bool = False
    #: The script's METHOD RECEIPT: the method that was ACTUALLY carried
    #: out, read back from ``result.json``'s optional ``method_performed``
    #: object when it is stamped with THIS attempt's ``action_id``. None
    #: when the script reported no method (or its receipt carries no
    #: matching stamp) — an unobserved method stays UNKNOWN, never a copy of
    #: the plan. Same attribution discipline as ``config_report``.
    method_report: Optional[Dict[str, Any]] = None


class SafePythonExecutor:
    """Best-effort local sandbox for harness-written Python code."""

    def __init__(self, timeout_seconds: int = 120, solver_timeout_seconds: int = 60,
                 max_stdout_chars: int = 20000, max_stderr_chars: int = 20000):
        self.timeout_seconds = timeout_seconds
        self.solver_timeout_seconds = solver_timeout_seconds
        self.max_stdout_chars = max_stdout_chars
        self.max_stderr_chars = max_stderr_chars

    # -- execution ------------------------------------------------------------

    def run(self, code_path: Path, workspace: Path, solver: str,
            *, action_id: Optional[str] = None) -> ExecutionOutcome:
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
        # The action id travels to the script so its config receipt can be
        # stamped with it: a receipt is only THIS attempt's configuration
        # when the stamp matches. Without a stamp a leftover result.json
        # from an earlier run in the same workspace would be read as the
        # current attempt's configuration.
        if action_id:
            env["OR_ACTION_ID"] = str(action_id)
        start = time.monotonic()
        # ATTRIBUTION OF THE RESULT BODY. The workspace may still hold a
        # ``result.json`` from an earlier run (a re-run in the same
        # directory, an abandoned attempt). Verifying the CONFIG receipt's
        # action_id is not enough: a script that writes nothing at all would
        # leave the old file in place, and the framework would report the
        # PREVIOUS run's status/objective as this attempt's result — the
        # exact way an old sample enters the calibration. Two guards, in
        # order:
        #   1. remove any existing ``result.json`` BEFORE the script starts,
        #      so the only file that can be read afterwards is one this run
        #      actually wrote;
        #   2. refuse a file whose mtime predates this run (belt and braces
        #      against a script that restores a copy).
        result_path = workspace / "result.json"
        run_wall_start = time.time()
        stale_cleared = False
        try:
            if result_path.exists():
                result_path.unlink()
                stale_cleared = True
        except OSError as exc:
            return ExecutionOutcome(
                status="error", solver=solver, executed=False,
                normalized_error=(
                    f"execution workspace is not writable: could not clear a "
                    f"stale result.json ({type(exc).__name__}: {exc})"),
                message=("the workspace held a result.json this attempt could "
                         "not clear; refusing to run rather than risk reading "
                         "another run's result"))
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
            # A timeout still leaves whatever the script managed to write:
            # the config receipt is read back on the failure path too, so
            # a run that set its key parameters before hitting the wall
            # clock can still report the configuration that was in force.
            return ExecutionOutcome(
                status="timeout", solver=solver,
                wall_seconds=time.monotonic() - start,
                normalized_error="execution timeout",
                message="Solve script exceeded the wall-clock timeout",
                config_report=_config_receipt(workspace, action_id),
                method_report=_method_receipt(workspace, action_id),
                stale_result_cleared=stale_cleared)
        wall = time.monotonic() - start
        stdout = _clip(proc.stdout.decode("utf-8", "replace"), self.max_stdout_chars)
        stderr = _clip(proc.stderr.decode("utf-8", "replace"), self.max_stderr_chars)
        if proc.returncode != 0 or not result_path.exists():
            return ExecutionOutcome(
                status="error", solver=solver, exit_code=proc.returncode,
                wall_seconds=wall, stdout=stdout, stderr=stderr,
                normalized_error=_normalize_error(
                    stderr or stdout or _exit_note(proc.returncode)),
                message="Process failed or did not write result.json",
                config_report=_config_receipt(workspace, action_id),
                method_report=_method_receipt(workspace, action_id),
                stale_result_cleared=stale_cleared)
        # Guard 2: the file must have been written AFTER this run started. A
        # script that restores a pre-existing copy would otherwise hand back
        # an older attempt's body.
        try:
            if result_path.stat().st_mtime < run_wall_start - 1e-6:
                return ExecutionOutcome(
                    status="error", solver=solver, exit_code=proc.returncode,
                    wall_seconds=wall, stdout=stdout, stderr=stderr,
                    normalized_error=(
                        "result.json is older than this run: a stale result "
                        "cannot be attributed to this attempt"),
                    message=("result.json predates this attempt: refusing to "
                             "read another run's result as this one's"),
                    config_report=_config_receipt(workspace, action_id),
                    method_report=_method_receipt(workspace, action_id),
                    stale_result_cleared=stale_cleared)
        except OSError:
            pass
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionOutcome(
                status="error", solver=solver, exit_code=proc.returncode,
                wall_seconds=wall, stdout=stdout, stderr=stderr,
                normalized_error="invalid result.json: " + type(exc).__name__,
                message="result.json is invalid",
                config_report=_config_receipt(workspace, action_id),
                method_report=_method_receipt(workspace, action_id),
                stale_result_cleared=stale_cleared)
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
            config_report=_config_receipt(workspace, action_id),
            method_report=_method_receipt(workspace, action_id),
            stale_result_cleared=stale_cleared,
            stdout=stdout, stderr=stderr)

    # -- verification (infrastructure, not a research contribution) -----------

    @staticmethod
    def verify(outcome: ExecutionOutcome) -> Dict[str, Any]:
        """Basic, cheap verification: legal status, finite objective when
        claimed, gap recorded when a bound exists. This is the only depth
        this build performs — there is no second, stronger tier to select.

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
                code_hash: Optional[str] = None,
                action_id: Optional[str] = None,
                method_planned: Optional[Dict[str, Any]] = None
                ) -> ExecutionRecord:
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

        ``action_id`` (optional) is the unified action record this attempt
        belongs to. It is passed to the script through ``OR_ACTION_ID`` and
        used to accept ONLY a config receipt stamped with the same id, so a
        leftover ``result.json`` from an earlier run in the same workspace
        is never read as this attempt's configuration. The SAME stamp gates
        the optional ``method_performed`` receipt, for the same reason.

        ``method_planned`` (optional) is the outer agent's method for this
        candidate, copied onto the record as a PLAN (``method_planned``).
        The method that actually ran (``method_actual``) comes only from the
        script's own ``method_performed`` receipt — never from this argument,
        because a plan is not an observation.
        """
        started = time.monotonic()
        outcome = self.run(code_path, workspace, solver, action_id=action_id)
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
        # The method ACTUALLY performed, from the script's own receipt (if it
        # carried a matching stamp). An unobserved method stays None — the
        # plan below is never copied in as if it had been carried out.
        method_actual = None
        if isinstance(outcome.method_report, dict):
            method_actual = outcome.method_report.get("method")
        method_planned_norm = normalize_method(method_planned)
        trajectory = [TrajectoryStep(
            action=f"execute:{strategy_id} via {solver}",
            outcome=outcome.status, duration_s=outcome.wall_seconds)]
        # Real processing steps the script reported: each one is a step of
        # the method that ACTUALLY ran, in order, attributed to this attempt.
        # The literal 'execute:...' step above stays as the fallback so a
        # script that reports no method still has an honest trajectory.
        if method_actual:
            for step in method_actual.get("steps") or []:
                trajectory.append(TrajectoryStep(
                    action=f"method:{step}", outcome=outcome.status,
                    duration_s=0.0))
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
        # The configuration that ACTUALLY took effect, split by WHO could
        # observe it, and by HOW WELL the executor knows it. A key the
        # executor did not control and the script did not read back stays
        # ABSENT (unknown), so the binding can tell "the prediction matched"
        # from "nothing confirmed it". Three classes are kept apart:
        #   * ``executor`` — the sandbox's OWN policy (``script_timeout_s``
        #     is the wall clock over the whole script; ``solver`` is the
        #     solver the executor was told to use). These are FACTS about
        #     this run and a script may NOT overwrite them: a solve script
        #     that reports ``script_timeout_s=1`` while the sandbox really
        #     allowed 120s is either mistaken or trying to make itself match
        #     a prediction, and either way recording 1 would let a wrong
        #     sample into the calibration. The conflict is reported under
        #     ``conflicts`` instead.
        #   * ``executor_configured`` — the limit the executor HANDS to the
        #     script through the environment (``solver_timeout_s``). It is
        #     an instruction, not an observation: the framework cannot see
        #     whether the solver honoured it, so it is reported separately
        #     and is NEVER used as evidence that the solver used that value.
        #     What the solver really applied must be READ BACK by the script
        #     (``script_reported``).
        #   * ``script_reported`` — values the solve script read back from
        #     the solver. These are observations the executor could not make
        #     itself, and they fill in the keys the executor does not own.
        executor_values: Dict[str, Any] = {}
        executor_configured: Dict[str, Any] = {}
        if outcome.executed:
            executor_values["solver"] = outcome.solver or solver
            if self.timeout_seconds is not None:
                executor_values["script_timeout_s"] = float(self.timeout_seconds)
            if self.solver_timeout_seconds is not None:
                executor_configured["solver_timeout_s"] = float(
                    self.solver_timeout_seconds)
        receipt = outcome.config_report or {}
        reported_values = dict(receipt.get("values") or {})
        stale_cleared = bool(outcome.stale_result_cleared)
        conflicts: Dict[str, Any] = {}
        script_values: Dict[str, Any] = {}
        for key, value in reported_values.items():
            if key in executor_values:
                if value != executor_values[key]:
                    # The script CONTRADICTS something the executor really
                    # did. The executor's value stands; the disagreement is
                    # recorded, and the calibration reads the conflict rather
                    # than a script-overwritten "match".
                    conflicts[key] = {
                        "executor": executor_values[key],
                        "script_reported": value,
                        "note": ("the solve script reported a value for a "
                                 "key the executor controls; the executor's "
                                 "own value is kept and the disagreement is "
                                 "recorded — a script cannot overwrite what "
                                 "the sandbox actually did"),
                    }
                continue
            if key in executor_configured:
                # Not an observation either way: the executor INSTRUCTED the
                # script. Recorded as reported (what the script says it used)
                # with the instruction alongside, never merged.
                script_values[key] = value
                continue
            script_values[key] = value
        config_values: Dict[str, Any] = dict(executor_values)
        config_sources: Dict[str, str] = {k: "executor"
                                          for k in executor_values}
        for key, value in script_values.items():
            config_values[key] = value
            config_sources[key] = "script_reported"
        for key, value in executor_configured.items():
            # Kept in the block but NOT in ``values``: an instruction is not
            # an observation, and putting it in ``values`` is exactly how it
            # would be read as "the solver used this".
            config_sources.setdefault(key, "executor_configured")
        if config_values or executor_configured:
            block: Dict[str, Any] = {
                "values": config_values,
                "sources": config_sources,
                "action_id": str(action_id) if action_id else None,
                "note": ("the configuration that actually took effect, split "
                         "by observer: 'executor' keys are the ones the "
                         "sandbox set (a script may not overwrite them), "
                         "'script_reported' keys are the values the solve "
                         "script read back from the solver. A key absent "
                         "here is UNKNOWN — never a copy of the predicted "
                         "config"),
            }
            if executor_configured:
                block["executor_configured"] = executor_configured
                block["executor_configured_note"] = (
                    "these values were HANDED to the script by the executor "
                    "(through the environment). The framework cannot observe "
                    "whether the solver honoured them, so they are not "
                    "evidence that the solver used them: only a value the "
                    "script READ BACK appears under 'script_reported'")
            if conflicts:
                block["conflicts"] = conflicts
                block["conflicts_note"] = (
                    "the script reported a value for a key the executor "
                    "controls; the executor's own value was kept")
            if stale_cleared:
                block["stale_result_cleared"] = True
            if not reported_values:
                block["receipt"] = (
                    "the solve script reported no config (or its receipt "
                    "carried no matching action_id stamp): script-internal "
                    "parameters stay unknown")
            execution_features["execution_config"] = block
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
        # The method that REALLY ran, as a normalizer observation record (not
        # a copy of the plan). ``values`` is the performed method; a receipt
        # with no matching action_id stamp is refused by ``_method_receipt``,
        # so the block's presence means "this attempt reported this method".
        if isinstance(outcome.method_report, dict):
            execution_features["method_receipt"] = {
                "method": method_actual,
                "action_id": outcome.method_report.get("action_id"),
                "note": ("the method the solve script reports it ACTUALLY "
                         "performed, stamped with this attempt's action id. "
                         "It is an observation, kept separate from the "
                         "planned method — a plan is not a performed method"),
            }
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
            measurement_scope="attempt",
            solver_runtime_provenance=runtime_provenance,
            method_planned=method_planned_norm,
            method_actual=normalize_method(method_actual))

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
