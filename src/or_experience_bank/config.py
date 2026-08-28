"""Configuration with CLI > environment > file > defaults precedence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by config.example.yaml without dependencies."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    last_key: Optional[str] = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("Invalid YAML list placement")
            parent.append(_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ValueError("Unsupported YAML line: " + stripped)
        key, value = stripped.split(":", 1)
        key, value = key.strip(), value.strip()
        if not value:
            next_is_list = False
            lines = text.splitlines()
            current_index = lines.index(raw)
            for future in lines[current_index + 1 :]:
                if future.strip() and not future.lstrip().startswith("#"):
                    next_is_list = future.strip().startswith("-")
                    break
            child: Any = [] if next_is_list else {}
            parent[key] = child
            stack.append((indent, child))
            last_key = key
        else:
            parent[key] = _scalar(value)
            last_key = key
    return root


def _scalar(value: str) -> Any:
    if value in {"null", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class ExperienceBankConfig:
    bank_home: Path = field(default_factory=lambda: _expand("~/.hermes/or-experience-bank"))
    retrieval_backend: str = "auto"
    embedding_model: Optional[str] = "local-hashing-embedding-v1"
    top_k: Dict[str, int] = field(
        default_factory=lambda: {"modeling": 5, "implementation": 5, "repair": 5, "solving": 5}
    )
    min_similarity: Optional[float] = None
    solvers: List[str] = field(
        default_factory=lambda: ["gurobi", "scip", "highs", "copt", "ortools", "pulp", "pyomo"]
    )
    max_parallel_branches: int = 3
    max_attempts_per_branch: int = 3
    min_cross_validation_branches: int = 3
    stop_on_repeated_error: bool = True
    stop_on_unchanged_code: bool = True
    python_timeout_seconds: int = 120
    solver_timeout_seconds: int = 60
    max_stdout_chars: int = 20000
    max_stderr_chars: int = 20000
    allow_network: bool = False
    auto_append: bool = True
    append_positive: bool = True
    append_negative: bool = True
    reject_exact_duplicates: bool = True
    detect_near_duplicates: bool = True
    minimum_positive_experience_level: str = "solver_feasible"

    @classmethod
    def load(
        cls, config_path: Optional[str] = None, cli_overrides: Optional[Dict[str, Any]] = None
    ) -> "ExperienceBankConfig":
        defaults = cls()
        values: Dict[str, Any] = {}
        if config_path:
            text = Path(config_path).read_text(encoding="utf-8")
            raw = json.loads(text) if config_path.endswith(".json") else _simple_yaml(text)
            values = _flatten_config(raw)
        env = _environment_overrides()
        values.update(env)
        values.update({k: v for k, v in (cli_overrides or {}).items() if v is not None})
        for key, value in values.items():
            if hasattr(defaults, key):
                if key == "bank_home":
                    value = _expand(str(value))
                setattr(defaults, key, value)
        defaults.max_attempts_per_branch = max(1, min(int(defaults.max_attempts_per_branch), 10))
        defaults.max_parallel_branches = max(1, min(int(defaults.max_parallel_branches), 16))
        defaults.min_cross_validation_branches = max(2, int(defaults.min_cross_validation_branches))
        return defaults

    def ensure_directories(self) -> None:
        for name in ("bank", "trajectories", "index", "runs", "logs"):
            (self.bank_home / name).mkdir(parents=True, exist_ok=True)


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = raw.get("retrieval", {})
    orchestration = raw.get("orchestration", {})
    execution = raw.get("execution", {})
    collection = raw.get("experience_collection", {})
    validation = raw.get("validation", {})
    result = {"bank_home": raw.get("bank_home", "~/.hermes/or-experience-bank")}
    mapping = {
        "retrieval_backend": retrieval.get("backend"),
        "embedding_model": retrieval.get("embedding_model"),
        "top_k": retrieval.get("top_k"),
        "min_similarity": retrieval.get("min_similarity"),
        "solvers": orchestration.get("solvers"),
        "max_parallel_branches": orchestration.get("max_parallel_branches"),
        "max_attempts_per_branch": orchestration.get("max_attempts_per_branch"),
        "min_cross_validation_branches": orchestration.get("min_cross_validation_branches"),
        "stop_on_repeated_error": orchestration.get("stop_on_repeated_error"),
        "stop_on_unchanged_code": orchestration.get("stop_on_unchanged_code"),
        "python_timeout_seconds": execution.get("python_timeout_seconds"),
        "solver_timeout_seconds": execution.get("solver_timeout_seconds"),
        "max_stdout_chars": execution.get("max_stdout_chars"),
        "max_stderr_chars": execution.get("max_stderr_chars"),
        "allow_network": execution.get("allow_network"),
        "auto_append": collection.get("auto_append"),
        "append_positive": collection.get("append_positive"),
        "append_negative": collection.get("append_negative"),
        "reject_exact_duplicates": collection.get("reject_exact_duplicates"),
        "detect_near_duplicates": collection.get("detect_near_duplicates"),
        "minimum_positive_experience_level": validation.get("minimum_positive_experience_level"),
    }
    result.update({k: v for k, v in mapping.items() if v is not None})
    return result


def _environment_overrides() -> Dict[str, Any]:
    env = os.environ
    mapping: Dict[str, Any] = {}
    if env.get("OR_EXPERIENCE_BANK_HOME"):
        mapping["bank_home"] = env["OR_EXPERIENCE_BANK_HOME"]
    if env.get("OR_EXPERIENCE_AUTO_APPEND"):
        mapping["auto_append"] = _bool(env["OR_EXPERIENCE_AUTO_APPEND"])
    if env.get("OR_EXPERIENCE_MAX_ATTEMPTS"):
        mapping["max_attempts_per_branch"] = int(env["OR_EXPERIENCE_MAX_ATTEMPTS"])
    if env.get("OR_EXPERIENCE_MIN_CV_BRANCHES"):
        mapping["min_cross_validation_branches"] = int(env["OR_EXPERIENCE_MIN_CV_BRANCHES"])
    if env.get("OR_EXPERIENCE_SOLVERS"):
        mapping["solvers"] = [x.strip() for x in env["OR_EXPERIENCE_SOLVERS"].split(",") if x.strip()]
    if env.get("OR_EXPERIENCE_EMBEDDING_BACKEND"):
        mapping["retrieval_backend"] = env["OR_EXPERIENCE_EMBEDDING_BACKEND"]
    if env.get("OR_EXPERIENCE_EMBEDDING_MODEL"):
        mapping["embedding_model"] = env["OR_EXPERIENCE_EMBEDDING_MODEL"]
    if env.get("OR_EXPERIENCE_EXECUTION_TIMEOUT"):
        mapping["python_timeout_seconds"] = int(env["OR_EXPERIENCE_EXECUTION_TIMEOUT"])
    if env.get("OR_EXPERIENCE_SOLVER_TIMEOUT"):
        mapping["solver_timeout_seconds"] = int(env["OR_EXPERIENCE_SOLVER_TIMEOUT"])
    return mapping
