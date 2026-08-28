"""Bank query commands: query / show / deprecate / stats / doctor / init.

All read-only except deprecate (lifecycle flip + cold archive) and init.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .components import Components, build_components
from .run_store import RunError


def cmd_query(
    comps: Components,
    layer: str,
    query: str,
    solver: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    if layer == "modeling":
        priors = comps.modeling_retriever.retrieve_priors(query, top_k=top_k)
        hits = [
            {
                "experience_id": rec.get("experience_id"),
                "title": rec.get("title"),
                "status": rec.get("status"),
                "modeling_aspect": rec.get("modeling_aspect"),
                "retrieval_text": rec.get("retrieval_text"),
            }
            for rec in priors.records
        ]
        return {"layer": "modeling", "hits": hits}

    filters: Dict[str, Any] = {}
    if solver:
        filters["solver"] = solver
    results = comps.retriever.retrieve(layer, query, metadata_filters=filters, top_k=top_k)
    return {
        "layer": layer,
        "hits": [
            {
                "experience_id": h.experience_id,
                "title": h.title,
                "polarity": h.polarity,
                "score": round(float(h.score), 4),
                "retrieval_text": h.retrieval_text,
            }
            for h in results
        ],
    }


def cmd_show(comps: Components, experience_id: str) -> Dict[str, Any]:
    for rec in comps.modeling_store.iter_records():
        if rec.get("experience_id") == experience_id:
            return {"layer": "modeling", "record": rec}
    for layer in ("implementation", "repair", "solving"):
        for rec in comps.store.iter_records(layer, strict=False):
            if rec.get("experience_id") == experience_id:
                return {"layer": layer, "record": rec}
    raise RunError("experience_id {!r} not found in any bank".format(experience_id))


def cmd_deprecate(comps: Components, experience_id: str, reason: str) -> Dict[str, Any]:
    record = None
    for rec in comps.modeling_store.iter_records():
        if rec.get("experience_id") == experience_id:
            record = rec
            break
    if record is None:
        for layer in ("implementation", "repair", "solving"):
            for rec in comps.store.iter_records(layer, strict=False):
                if rec.get("experience_id") == experience_id:
                    record = rec
                    break
            if record is not None:
                break
    if record is None:
        raise RunError("experience_id {!r} not found in any bank".format(experience_id))
    card = comps.lifecycle.mark_deprecated(record, reason, embed=comps.backend.embed_documents)
    return {
        "deprecated": True,
        "experience_id": experience_id,
        "reason": reason,
        "archive_card_summary": card.get("summary", ""),
    }


def cmd_stats(comps: Components) -> Dict[str, Any]:
    modeling = comps.modeling_store.stats()
    layers: Dict[str, int] = {}
    for layer in ("implementation", "repair", "solving"):
        layers[layer] = sum(1 for _ in comps.store.iter_records(layer, strict=False))
    episodes = comps.episode_store.stats()
    return {
        "bank_home": str(comps.config.bank_home),
        "modeling": {"total": modeling["total"], "validated": modeling["validated"]},
        "flat_layers": layers,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# doctor: deployment self-check
# ---------------------------------------------------------------------------

_SOLVER_PACKAGES = {
    "gurobi": "gurobipy",
    "scip": "pyscipopt",
    "highs": "highspy",
    "copt": "coptpy",
    "ortools": "ortools",
    "pulp": "pulp",
    "pyomo": "pyomo",
}


def cmd_doctor(comps: Components) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    checks["python"] = {"version": sys.version.split()[0], "ok": sys.version_info >= (3, 9)}

    bank = comps.config.bank_home
    writable = False
    try:
        probe = bank / ".orx_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    checks["bank"] = {"path": str(bank), "writable": writable, "ok": writable}

    solvers: Dict[str, Dict[str, Any]] = {}
    for name, package in _SOLVER_PACKAGES.items():
        try:
            importlib.import_module(package)
            solvers[name] = {"available": True, "package": package}
        except ImportError:
            solvers[name] = {"available": False, "package": package}
    checks["solvers"] = solvers
    checks["solvers_available"] = sorted(n for n, s in solvers.items() if s["available"])

    index_state = comps.retriever.validate_indexes()
    checks["indexes"] = {"valid": index_state.get("valid", False)}

    ok = (
        checks["python"]["ok"]
        and checks["bank"]["ok"]
        and bool(checks["solvers_available"])
        and checks["indexes"]["valid"]
    )
    return {
        "ok": ok,
        "checks": checks,
        "hint": (
            "all checks passed" if ok else
            "fix the failing checks above; at least one solver package is required "
            "for real solves (pip install highspy pulp ortools are free options)"
        ),
    }


def cmd_init(config_path: Optional[str] = None, bank_home: Optional[str] = None) -> Dict[str, Any]:
    from ..config import ExperienceBankConfig

    overrides: Dict[str, Any] = {}
    if bank_home:
        overrides["bank_home"] = bank_home
    config = ExperienceBankConfig.load(config_path=config_path, cli_overrides=overrides)
    config.ensure_directories()
    # Touch the fact files so the bank is immediately usable.
    from ..core.store import AppendOnlyExperienceStore
    from ..core.modeling_store import ModelingStore
    from ..core.episode import EpisodeStore
    AppendOnlyExperienceStore(config.bank_home)
    ModelingStore(config.bank_home)
    EpisodeStore(config.bank_home)
    return {
        "initialized": True,
        "bank_home": str(config.bank_home),
        "next": "run `orx doctor` to verify the environment, then `orx recall --problem <file>`",
    }


__all__ = [
    "cmd_query", "cmd_show", "cmd_deprecate", "cmd_stats", "cmd_doctor", "cmd_init",
]
