"""Run-directory state: artifacts are the state, stamps enforce the chain.

ReAct-oriented redesign (2026-08): the harness agent is the orchestrator; the
framework is a set of stateless commands. All cross-call state lives in a run
directory on disk, so every tool invocation is an independent process:

    <run_dir>/
      problem.txt            the problem text (created by `orx recall`)
      priors.json            recall output: priors + En -> experience_id labels
      model.txt              agent-authored <think>/<model> response
      signature.json         agent-authored structural signature
      stamps/model.json      L1+L2 verdict + content hash of model.txt
      stamps/signature.json  vocabulary verdict + content hash of signature.json
      branches/<br>/solve.py agent-authored solver code (br = solver name)
      branches/<br>/result.json   sandbox execution outcome + hints
      gold.json              gold answer + match verdict (agent-declared)
      experiences.json       appended experience ids (one line per append)
      episode.json           terminal record + utility credit summary
      rounds/<n>/            archived artifacts of outer reflection round n
      journal.jsonl          append-only audit log of every orx command

The "unskippable chain" property is enforced by STAMP PRECONDITIONS instead of
one-time tokens: each command checks that the predecessor artifact exists AND
that its recorded content hash still matches the file on disk. A stale stamp
(model edited after validation) is rejected exactly like a missing one. This
keeps the ritual unskippable while making every step freely retryable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunError(RuntimeError):
    """Raised when a run directory is missing, malformed, or the chain is broken."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError("cannot read {}: {}".format(path.name, exc))
    if not isinstance(data, dict):
        raise RunError("{} does not contain a JSON object".format(path.name))
    return data


class RunStore:
    """Filesystem view of one solve run. Pure functions over the run directory."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        if not self.dir.is_dir():
            raise RunError(
                "not a run directory: {} (run `orx recall --problem <file>` inside your "
                "working directory first, or cd into the run directory)".format(self.dir)
            )
        if not (self.dir / "problem.txt").exists() and not (self.dir / "journal.jsonl").exists():
            # A run directory is created by `orx recall`; anything else is a random dir.
            raise RunError(
                "not a run directory: {} has no problem.txt/journal.jsonl (start a run "
                "with `orx recall --problem <file>`)".format(self.dir)
            )

    # -- artifact paths ------------------------------------------------------

    @property
    def problem_path(self) -> Path:
        return self.dir / "problem.txt"

    @property
    def priors_path(self) -> Path:
        return self.dir / "priors.json"

    @property
    def model_path(self) -> Path:
        return self.dir / "model.txt"

    @property
    def signature_path(self) -> Path:
        return self.dir / "signature.json"

    @property
    def gold_path(self) -> Path:
        return self.dir / "gold.json"

    @property
    def experiences_path(self) -> Path:
        return self.dir / "experiences.json"

    @property
    def episode_path(self) -> Path:
        return self.dir / "episode.json"

    @property
    def branches_dir(self) -> Path:
        return self.dir / "branches"

    @property
    def stamps_dir(self) -> Path:
        return self.dir / "stamps"

    @property
    def rounds_dir(self) -> Path:
        return self.dir / "rounds"

    # -- journal -------------------------------------------------------------

    def journal(self, command: str, payload: Dict[str, Any]) -> None:
        entry = {"ts": time.time(), "command": command, **payload}
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        with (self.dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)

    # -- stamps (chain enforcement) -------------------------------------------

    def stamp_path(self, name: str) -> Path:
        return self.stamps_dir / (name + ".json")

    def write_stamp(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.stamp_path(name)
        _atomic_write_json(path, payload)
        return path

    def read_stamp(self, name: str) -> Dict[str, Any]:
        path = self.stamp_path(name)
        if not path.exists():
            raise RunError("missing stamp '{}': run the predecessor step first".format(name))
        return _read_json(path)

    def require_stamp(self, name: str, source: Path) -> Dict[str, Any]:
        """Load a stamp and verify its source file is unchanged since stamping.

        This is the anti-skip / anti-stale mechanism: editing model.txt after
        validation invalidates the stamp, exactly like losing a token did.
        """
        stamp = self.read_stamp(name)
        recorded = stamp.get("source_sha256", "")
        if not source.exists():
            raise RunError(
                "stamp '{}' is stale: {} was deleted; redo the step".format(name, source.name)
            )
        current = _sha256_file(source)
        if recorded and current != recorded:
            raise RunError(
                "stamp '{}' is stale: {} changed after stamping; re-run the validation "
                "step for the new content".format(name, source.name)
            )
        return stamp

    # -- phase detection (for `orx status`) ------------------------------------

    def phase(self) -> Dict[str, Any]:
        """Scan artifacts and report where the run stands + what is next."""
        state: Dict[str, Any] = {"run_dir": str(self.dir)}
        state["has_problem"] = self.problem_path.exists()
        state["has_priors"] = self.priors_path.exists()
        state["model_stamped"] = self.stamp_path("model").exists()
        state["signature_stamped"] = self.stamp_path("signature").exists()
        branches = self.branch_results()
        state["branches"] = [
            {"branch": b, "solver": r.get("solver"), "status": r.get("status"),
             "objective_value": r.get("objective_value"), "valid": r.get("valid")}
            for b, r in sorted(branches.items())
        ]
        state["gold_recorded"] = self.gold_path.exists()
        state["appended_count"] = self.appended_count()
        state["episode_recorded"] = self.episode_path.exists()
        state["rounds_archived"] = len(self._round_dirs())

        if state["episode_recorded"]:
            state["phase"] = "complete"
            state["next"] = "run complete; start a new run with `orx recall` or run induction with `orx clusters`"
        elif state["appended_count"] > 0 or state["gold_recorded"]:
            state["phase"] = "appending"
            state["next"] = "finish appending experiences, then `orx episode`"
        elif state["signature_stamped"] and branches:
            state["phase"] = "solving"
            state["next"] = "branch executed: check gold (user-provided only), then `orx gold`"
        elif state["signature_stamped"]:
            state["phase"] = "signature_verified"
            state["next"] = "`orx hints --solver <name>`, write branches/<solver>/solve.py, then `orx solve --solver <name>`"
        elif state["model_stamped"]:
            state["phase"] = "model_verified"
            state["next"] = "write signature.json, then `orx signature`"
        elif state["has_priors"]:
            state["phase"] = "recalled"
            state["next"] = "write model.txt (<think>/<model>), then `orx validate`"
        elif state["has_problem"]:
            state["phase"] = "created"
            state["next"] = "`orx recall` to fetch planning priors"
        else:
            state["phase"] = "empty"
            state["next"] = "`orx recall --problem <file>` to start a run"
        return state

    # -- branch artifacts -------------------------------------------------------

    def branch_dir(self, solver: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in solver)
        return self.branches_dir / safe

    def branch_results(self) -> Dict[str, Dict[str, Any]]:
        """All executed branch outcomes (result.json per branch dir)."""
        results: Dict[str, Dict[str, Any]] = {}
        if not self.branches_dir.is_dir():
            return results
        for child in sorted(self.branches_dir.iterdir()):
            result_path = child / "result.json"
            if child.is_dir() and result_path.exists():
                try:
                    results[child.name] = _read_json(result_path)
                except RunError:
                    continue
        return results

    def valid_branches(self) -> List[Dict[str, Any]]:
        return [
            r for r in self.branch_results().values()
            if r.get("valid") and r.get("status") in {"optimal", "feasible"}
            and r.get("objective_value") is not None
        ]

    # -- experience / episode artifacts ------------------------------------------

    def appended_count(self) -> int:
        if not self.experiences_path.exists():
            return 0
        count = 0
        for line in self.experiences_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                count += 1
        return count

    def appended_ids(self) -> List[str]:
        ids: List[str] = []
        if not self.experiences_path.exists():
            return ids
        for line in self.experiences_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("experience_id"):
                ids.append(str(row["experience_id"]))
        return ids

    def record_append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self.experiences_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    # -- priors / citations --------------------------------------------------------

    def prior_labels(self) -> Dict[str, str]:
        if not self.priors_path.exists():
            return {}
        try:
            data = _read_json(self.priors_path)
        except RunError:
            return {}
        labels = data.get("labels", {})
        return {str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {}

    def problem_text(self) -> str:
        if not self.problem_path.exists():
            raise RunError("problem.txt missing: start the run with `orx recall --problem <file>`")
        return self.problem_path.read_text(encoding="utf-8")

    def problem_id(self) -> str:
        return "prob_" + hashlib.sha256(self.problem_text().encode("utf-8")).hexdigest()[:16]

    # -- outer reflection rounds ------------------------------------------------------

    def _round_dirs(self) -> List[Path]:
        if not self.rounds_dir.is_dir():
            return []
        return sorted(
            (c for c in self.rounds_dir.iterdir() if c.is_dir()),
            key=lambda p: p.name,
        )

    def archive_round(self) -> int:
        """Move current-round artifacts into rounds/<n>/ for a fresh reflection round.

        Keeps problem.txt and priors.json in place (the problem does not change;
        priors remain citable). Everything else is archived so the next round
        starts from a clean modeling slate.
        """
        round_index = len(self._round_dirs()) + 1
        target = self.rounds_dir / str(round_index)
        target.mkdir(parents=True, exist_ok=True)
        keep = {self.problem_path.name, self.priors_path.name}
        for child in self.dir.iterdir():
            if child.name in keep or child.name in {"journal.jsonl", "rounds"}:
                continue
            if child.is_file() or child.is_dir():
                os.replace(str(child), str(target / child.name))
        return round_index


def create_run(run_dir: Path, problem_text: str) -> RunStore:
    """Initialize a run directory with problem.txt (idempotent on same problem)."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    problem_path = run_dir / "problem.txt"
    if problem_path.exists():
        if problem_path.read_text(encoding="utf-8") != problem_text:
            raise RunError(
                "run directory already holds a DIFFERENT problem; use a fresh directory"
            )
    else:
        problem_path.write_text(problem_text, encoding="utf-8")
    return RunStore(run_dir)


__all__ = ["RunStore", "RunError", "create_run"]
