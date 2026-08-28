"""Offline induction-chain commands: trigger / clusters / align / induce / refute
/ validate-pattern / append-pattern.

Same artifact-as-state model as the solve chain, but the "run directory" is a
per-cluster directory under <bank_home>/induction/<cluster_id>/:

    <cluster_dir>/
      cluster.json       cluster snapshot (members, core_key, roles vocabulary)
      alignment.json     agent-authored role bindings -> stamped by `orx align`
      hypotheses.json    agent-authored hypotheses -> stamped by `orx induce`
      refutations.json   agent-authored refutation programs + executor verdicts
      validation.json    transfer evidence + scoring verdicts
      patterns.json      appended pattern ids (terminal)

Chain enforcement: each command stamps its input artifact with a content hash;
the next command requires the stamp and rejects changed content.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.modeling_schemas import (
    MathScope,
    MethodBody,
    ModelingEvidence,
    ModelingExperience,
    PatternScoring,
    PatternValidation,
    RoleMappingEntry,
    StructuralSignature,
)
from ..induction.candidates import SignatureClusterer
from ..induction.counterexample import CounterexampleSearcher
from .components import Components
from .run_store import RunError, _read_json, _sha256_file, _atomic_write_json

CANONICAL_ROLES = (
    "resource_pool", "capacity_limit", "competing_decisions", "objective_contribution",
    "demand_requirement", "coupling_constraint", "time_period", "flow_balance",
)


def _induction_root(comps: Components) -> Path:
    root = comps.config.bank_home / "induction"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cluster_dir(comps: Components, cluster_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cluster_id)
    path = _induction_root(comps) / safe
    if not path.is_dir():
        raise RunError(
            "no induction directory for cluster {!r}: run `orx clusters` and `orx align` "
            "first".format(cluster_id)
        )
    return path


def _resolve_cluster(comps: Components, cluster_id: str):
    clusterer = SignatureClusterer(
        lifecycle=comps.lifecycle, utility_tracker=comps.utility_tracker
    )
    for c in clusterer.discover(comps.modeling_store.all_records()):
        if c.cluster_id == cluster_id:
            return c
    raise RunError(
        "cluster {!r} not found in a fresh discovery pass; run `orx clusters` for "
        "current ids (cluster ids change when membership changes)".format(cluster_id)
    )


# ---------------------------------------------------------------------------
# trigger + clusters (discovery; no chain state)
# ---------------------------------------------------------------------------

def cmd_trigger(comps: Components) -> Dict[str, Any]:
    from ..induction.trigger import InductionTrigger
    clusterer = SignatureClusterer(lifecycle=comps.lifecycle, utility_tracker=comps.utility_tracker)
    trigger = InductionTrigger(store=comps.modeling_store, clusterer=clusterer)
    decision = trigger.decide()
    return decision.to_dict()


def cmd_clusters(comps: Components) -> Dict[str, Any]:
    clusterer = SignatureClusterer(lifecycle=comps.lifecycle, utility_tracker=comps.utility_tracker)
    clusters = clusterer.discover(comps.modeling_store.all_records())
    return {
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "core_key": c.core_key,
                "families": sorted({m.problem_family for m in c.members}),
                "member_count": c.size,
                "members": [
                    {
                        "realization_id": m.realization_id,
                        "problem_family": m.problem_family,
                        "title": m.title,
                        "retrieval_text": m.record.get("retrieval_text", ""),
                    }
                    for m in c.members
                ],
            }
            for c in clusters
        ]
    }


# ---------------------------------------------------------------------------
# align: agent writes alignment.json -> stamp
# ---------------------------------------------------------------------------

def cmd_align(comps: Components, cluster_id: str) -> Dict[str, Any]:
    cluster = _resolve_cluster(comps, cluster_id)
    cdir = _cluster_dir(comps, cluster_id) if (_induction_root(comps) / cluster_id).is_dir() else None
    # The align command CREATES the cluster dir; allow re-align (overwrite).
    cdir = _induction_root(comps) / "".join(c if c.isalnum() or c in "-_" else "_" for c in cluster_id)
    cdir.mkdir(parents=True, exist_ok=True)

    alignment_path = cdir / "alignment.json"
    if not alignment_path.exists():
        # Write the cluster snapshot + an alignment template for the agent to fill.
        _atomic_write_json(cdir / "cluster.json", {
            "cluster_id": cluster.cluster_id,
            "core_key": cluster.core_key,
            "members": [
                {"realization_id": m.realization_id, "problem_family": m.problem_family,
                 "title": m.title, "retrieval_text": m.record.get("retrieval_text", "")}
                for m in cluster.members
            ],
            "canonical_roles": list(CANONICAL_ROLES),
        })
        _atomic_write_json(alignment_path, {
            "_template": True,
            "bindings": [
                {"role": "resource_pool", "realization_id": "<member id>", "entity": "<concrete entity>"}
            ],
        })
        return {
            "aligned": False,
            "cluster_dir": str(cdir),
            "instruction": (
                "Fill alignment.json inside {} (replace the template): one binding per "
                "role/realization pair. Roles come from cluster.json canonical_roles; "
                "realization_id must be a cluster member. Then re-run `orx align --cluster {}`."
            ).format(cdir, cluster_id),
        }

    data = _read_json(alignment_path)
    if data.get("_template"):
        raise RunError("alignment.json is still the template: fill in real bindings first")

    member_ids = {m.realization_id for m in cluster.members}
    member_family = {m.realization_id: m.problem_family for m in cluster.members}
    roles: List[str] = []
    bindings: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for entry in data.get("bindings", []):
        role = str(entry.get("role", "")).strip()
        if not role:
            continue
        if role not in roles:
            roles.append(role)
        rid = entry.get("realization_id", "")
        if rid not in member_ids:
            dropped.append("{}:{}".format(role, rid))
            continue
        bindings.append({
            "role": role,
            "realization_id": rid,
            "entity": entry.get("entity", ""),
            "problem_family": member_family.get(rid, ""),
        })
    if not bindings:
        raise RunError(
            "no grounded bindings: every binding must cite a member realization_id; dropped: {}".format(dropped)
        )

    _atomic_write_json(cdir / "stamps" / "alignment.json", {
        "passed": True,
        "source": "alignment.json",
        "source_sha256": _sha256_file(alignment_path),
        "roles": roles,
        "bindings": bindings,
    })
    _journal(cdir, "align", {"roles": len(roles), "bindings": len(bindings), "dropped": len(dropped)})
    return {
        "aligned": True,
        "cluster_dir": str(cdir),
        "roles_mapped": len(roles),
        "bindings_accepted": len(bindings),
        "bindings_dropped": dropped,
        "next": "write hypotheses.json (1-3 principles grounded in the alignment), then `orx induce --cluster {}`".format(cluster_id),
    }


# ---------------------------------------------------------------------------
# induce: agent writes hypotheses.json -> stamp
# ---------------------------------------------------------------------------

def cmd_induce(comps: Components, cluster_id: str) -> Dict[str, Any]:
    cdir = _cluster_dir(comps, cluster_id)
    stamp = _require_stamp(cdir, "alignment", cdir / "alignment.json")

    hyp_path = cdir / "hypotheses.json"
    if not hyp_path.exists():
        _atomic_write_json(hyp_path, {
            "_template": True,
            "hypotheses": [
                {"statement": "...", "structural_pattern": "...", "roles_used": ["resource_pool"],
                 "applicability_conditions": ["..."], "complexity": 0.5}
            ],
        })
        return {
            "induced": False,
            "cluster_dir": str(cdir),
            "instruction": (
                "Fill hypotheses.json inside {}: 1-3 candidate principles grounded in the "
                "alignment roles (never a summary). Then re-run `orx induce --cluster {}`."
            ).format(cdir, cluster_id),
        }

    data = _read_json(hyp_path)
    if data.get("_template"):
        raise RunError("hypotheses.json is still the template: fill in real hypotheses first")

    parsed = []
    for idx, h in enumerate(data.get("hypotheses", [])):
        statement = str(h.get("statement", "")).strip()
        if not statement:
            continue
        parsed.append({
            "hypothesis_id": "hyp_{}_{}".format(cluster_id[:10], idx + 1),
            "statement": statement,
            "structural_pattern": str(h.get("structural_pattern", "")),
            "roles_used": list(h.get("roles_used", [])),
            "source_realization_ids": [b["realization_id"] for b in stamp.get("bindings", [])],
            "applicability_conditions": list(h.get("applicability_conditions", [])),
            "complexity": float(h.get("complexity", 0.5)),
            "status": "hypothesis",
        })
    if not parsed:
        raise RunError("no valid hypotheses: each needs a non-empty 'statement'")

    _atomic_write_json(cdir / "stamps" / "hypotheses.json", {
        "passed": True,
        "source": "hypotheses.json",
        "source_sha256": _sha256_file(hyp_path),
        "hypotheses": parsed,
    })
    _journal(cdir, "induce", {"hypotheses": len(parsed)})
    return {
        "induced": True,
        "hypotheses": parsed,
        "next": "write refutations.json (failure conditions + refutation programs), then `orx refute --cluster {}`".format(cluster_id),
    }


# ---------------------------------------------------------------------------
# refute: agent writes refutations.json; the EXECUTOR decides the verdict
# ---------------------------------------------------------------------------

def cmd_refute(comps: Components, cluster_id: str) -> Dict[str, Any]:
    cdir = _cluster_dir(comps, cluster_id)
    hyp_stamp = _require_stamp(cdir, "hypotheses", cdir / "hypotheses.json")
    hypotheses = hyp_stamp.get("hypotheses", [])

    ref_path = cdir / "refutations.json"
    if not ref_path.exists():
        _atomic_write_json(ref_path, {
            "_template": True,
            "refutations": [
                {"hypothesis_id": "<hyp id>", "failure_condition": "...",
                 "refutation_code": "print('{\"principle_failed\": false, \"evidence\": \"...\"}')"}
            ],
        })
        return {
            "refuted": False,
            "cluster_dir": str(cdir),
            "instruction": (
                "Fill refutations.json inside {}: for each hypothesis, a failure condition "
                "and a self-contained Python program that instantiates it. The program must "
                "print as its LAST stdout line: {{\"principle_failed\": true|false, \"evidence\": \"...\"}}. "
                "Then re-run `orx refute --cluster {}`."
            ).format(cdir, cluster_id),
        }

    data = _read_json(ref_path)
    if data.get("_template"):
        raise RunError("refutations.json is still the template: fill in real refutations first")

    searcher = CounterexampleSearcher(executor=comps.executor)
    results: List[Dict[str, Any]] = []
    for ce in data.get("refutations", []):
        hyp_id = ce.get("hypothesis_id", "")
        hyp = next((h for h in hypotheses if h["hypothesis_id"] == hyp_id), None)
        if hyp is None:
            results.append({"hypothesis_id": hyp_id, "verdict": "unknown_hypothesis"})
            continue
        ws = cdir / "refutation_workspaces" / hyp_id
        ws.mkdir(parents=True, exist_ok=True)
        code_path = ws / "refutation.py"
        code_path.write_text(ce.get("refutation_code", ""), encoding="utf-8")
        # Refutation programs are NOT solver scripts: they print their verdict as
        # the last stdout line instead of writing result.json. Run them with a
        # plain subprocess (same env hygiene as SafePythonExecutor) and judge by
        # exit code + parsed stdout — a crash is NOT a counterexample.
        verdict, principle_failed, exit_status = _run_refutation_program(code_path, ws, searcher)
        results.append({
            "hypothesis_id": hyp_id,
            "failure_condition": ce.get("failure_condition", ""),
            "executed": exit_status != "crashed",
            "principle_failed": principle_failed,
            "verdict": verdict,
            "exit_status": exit_status,
        })

    _atomic_write_json(cdir / "stamps" / "refutations.json", {
        "passed": True,
        "source": "refutations.json",
        "source_sha256": _sha256_file(ref_path),
        "results": results,
    })
    _journal(cdir, "refute", {"results": len(results)})
    return {
        "results": results,
        "next": "write validation.json (unseen tasks + with/without-principle transfer evidence), "
                "then `orx validate-pattern --cluster {}`".format(cluster_id),
    }


# ---------------------------------------------------------------------------
# validate-pattern: agent writes validation.json (unseen tasks + transfer evidence)
# ---------------------------------------------------------------------------

def cmd_validate_pattern(comps: Components, cluster_id: str) -> Dict[str, Any]:
    cdir = _cluster_dir(comps, cluster_id)
    ref_stamp = _require_stamp(cdir, "refutations", cdir / "refutations.json")
    hyp_stamp = _require_stamp(cdir, "hypotheses", cdir / "hypotheses.json")
    align_stamp = _require_stamp(cdir, "alignment", cdir / "alignment.json")
    hypotheses = hyp_stamp.get("hypotheses", [])

    val_path = cdir / "validation.json"
    if not val_path.exists():
        _atomic_write_json(val_path, {
            "_template": True,
            "unseen_tasks": ["<a REAL problem from past episodes, different family, same signature>"],
            "transfer_results": [
                {"hypothesis_id": "<hyp id>", "task": "...", "improved": True,
                 "with_objective": 0.0, "without_objective": 0.0}
            ],
        })
        return {
            "validated": False,
            "cluster_dir": str(cdir),
            "instruction": (
                "Fill validation.json inside {}: unseen_tasks must be REAL problems from past "
                "episodes (different family, same signature); transfer_results must come from "
                "solves you ACTUALLY ran with and without the principle. Then re-run "
                "`orx validate-pattern --cluster {}`."
            ).format(cdir, cluster_id),
        }

    data = _read_json(val_path)
    if data.get("_template"):
        raise RunError("validation.json is still the template: fill in real transfer evidence first")

    unseen = [str(t) for t in data.get("unseen_tasks", []) if str(t).strip()]
    if not unseen:
        raise RunError("unseen_tasks must be non-empty: transfer validation requires unseen problems")

    verdicts = []
    for hyp in hypotheses:
        consistent = bool(hyp.get("roles_used")) and all(
            r in align_stamp.get("roles", []) for r in hyp.get("roles_used", [])
        )
        verdicts.append({
            "hypothesis_id": hyp["hypothesis_id"],
            "source_consistency": consistent,
            "unseen_tasks_supplied": len(unseen),
        })

    _atomic_write_json(cdir / "stamps" / "validation.json", {
        "passed": True,
        "source": "validation.json",
        "source_sha256": _sha256_file(val_path),
        "unseen_tasks": unseen,
        "transfer_results": data.get("transfer_results", []),
        "verdicts": verdicts,
    })
    _journal(cdir, "validate-pattern", {"unseen": len(unseen)})
    return {
        "verdicts": verdicts,
        "next": "`orx append-pattern --cluster {}` to score and append validated patterns".format(cluster_id),
    }


# ---------------------------------------------------------------------------
# append-pattern: score + append validated patterns (terminal)
# ---------------------------------------------------------------------------

def cmd_append_pattern(comps: Components, cluster_id: str) -> Dict[str, Any]:
    cdir = _cluster_dir(comps, cluster_id)
    val_stamp = _require_stamp(cdir, "validation", cdir / "validation.json")
    ref_stamp = _require_stamp(cdir, "refutations", cdir / "refutations.json")
    hyp_stamp = _require_stamp(cdir, "hypotheses", cdir / "hypotheses.json")
    align_stamp = _require_stamp(cdir, "alignment", cdir / "alignment.json")
    cluster = _resolve_cluster(comps, cluster_id)

    hypotheses = hyp_stamp.get("hypotheses", [])
    transfer_results = val_stamp.get("transfer_results", [])
    cx_results = ref_stamp.get("results", [])
    roles = align_stamp.get("roles", [])
    bindings = align_stamp.get("bindings", [])

    _ALPHA = _BETA = _GAMMA = _DELTA = _LAM = _MU = 1.0
    _THRESHOLD = 0.5

    verdicts_map = {v.get("hypothesis_id"): v for v in val_stamp.get("verdicts", [])}
    source_texts = [m.record.get("retrieval_text") or m.title for m in cluster.members]

    appended = []
    refuted = []
    for hyp in hypotheses:
        hyp_id = hyp["hypothesis_id"]

        verdict = verdicts_map.get(hyp_id, {})
        coverage = 1.0 if verdict.get("source_consistency", True) else 0.0

        hyp_transfer = [t for t in transfer_results if t.get("hypothesis_id") == hyp_id]
        transferability = (
            len([t for t in hyp_transfer if t.get("improved")]) / len(hyp_transfer)
            if hyp_transfer else 0.0
        )

        hyp_cx = [c for c in cx_results if c.get("hypothesis_id") == hyp_id]
        if not hyp_cx:
            validation = 0.5
        elif any(c.get("verdict") == "counterexample" for c in hyp_cx):
            validation = 0.0
        elif all(c.get("verdict") == "crashed_not_counterexample" for c in hyp_cx):
            validation = 0.5
        else:
            validation = 1.0

        statement_lower = hyp.get("statement", "").strip().lower()
        novelty = 1.0 if statement_lower and not any(
            statement_lower in (t or "").lower() for t in source_texts
        ) else 0.0

        complexity = float(hyp.get("complexity", 0.5))
        cx_penalty = float(len([c for c in hyp_cx if c.get("verdict") == "counterexample"]))

        total = (_ALPHA * coverage + _BETA * transferability + _GAMMA * validation
                 + _DELTA * novelty - _LAM * complexity - _MU * cx_penalty)

        is_refuted = validation == 0.0
        transfer_improved = transferability > 0.0
        passes = (not is_refuted) and (total >= _THRESHOLD) and transfer_improved

        scoring = PatternScoring(
            coverage=coverage, transferability=transferability, validation=validation,
            novelty=novelty, complexity=complexity, counterexample_penalty=cx_penalty, total=total,
        )

        if not passes:
            reason = (
                "counterexample confirmed (refuted)" if is_refuted
                else "no unseen-task improvement" if not transfer_improved
                else "total={:.3f} < threshold {:.3f}".format(total, _THRESHOLD)
            )
            refuted.append({"hypothesis_id": hyp_id, "statement": hyp["statement"],
                            "reason": reason, "scoring": scoring.to_dict()})
            continue

        rep_sig = cluster.representative_signature
        rep_dict = rep_sig.to_dict() if hasattr(rep_sig, "to_dict") else (rep_sig or {})
        record = ModelingExperience(
            title="Induced: " + hyp["structural_pattern"][:60],
            polarity="positive",
            retrieval_text=hyp["statement"],
            layer="modeling",
            modeling_aspect="structure",
            math_scope=MathScope(
                structural_signature=StructuralSignature.from_dict(rep_dict),
                exclusions=list(hyp.get("applicability_conditions", [])),
            ),
            method=MethodBody(
                action_template=hyp["statement"],
                rationale="Induced via cross-memory structural induction over cluster " + cluster_id,
            ),
            evidence=ModelingEvidence(
                source_episodes=[],
                validation_level="solver_validated",
                causal_confidence="medium",
            ),
            derived_from_experience_ids=list(hyp.get("source_realization_ids", [])),
            status="validated",
            role_schema={r: "abstract structural role shared across the cluster" for r in roles},
            role_mappings=[
                RoleMappingEntry(realization_id=b["realization_id"],
                                 problem_family=b.get("problem_family", ""),
                                 mapping={b["role"]: b.get("entity", "")})
                for b in bindings
            ],
            applicability_conditions=list(hyp.get("applicability_conditions", [])),
            validation=PatternValidation(
                source_consistency="covered" if coverage > 0 else "not covered",
            ),
            scoring=scoring,
        )
        record.compute_content_hash()
        result = comps.modeling_store.append(record)
        appended.append({
            "hypothesis_id": hyp_id,
            "experience_id": record.experience_id,
            "title": record.title,
            "status": result.get("status", "appended"),
            "scoring": scoring.to_dict(),
        })

    if appended:
        comps.modeling_retriever.rebuild()

    # Record the induction run in the trigger watermark (cooldown bookkeeping).
    from ..induction.trigger import InductionTrigger
    from ..induction.candidates import SignatureClusterer
    clusterer = SignatureClusterer(lifecycle=comps.lifecycle, utility_tracker=comps.utility_tracker)
    trigger = InductionTrigger(store=comps.modeling_store, clusterer=clusterer)
    trigger.record_run(trigger.decide())

    _atomic_write_json(cdir / "patterns.json", {"appended": appended, "refuted": refuted})
    _journal(cdir, "append-pattern", {"appended": len(appended), "refuted": len(refuted)})
    return {
        "appended": appended,
        "refuted": refuted,
        "status": "INDUCTION_FLOW_COMPLETE",
        "next": "validated patterns are now retrievable via `orx recall` in future solves",
    }


# -- helpers -----------------------------------------------------------------

def _run_refutation_program(code_path: Path, ws: Path, searcher: CounterexampleSearcher):
    """Execute a refutation program with a plain subprocess and classify the verdict.

    Unlike solver scripts, refutation programs do not write result.json; their
    contract is: print {"principle_failed": true|false, "evidence": "..."} as the
    LAST stdout line. Exit != 0 or a missing verdict line => crashed/inconclusive,
    never a counterexample (anti self-judgment).
    """
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "crashed_not_counterexample", None, "timeout"
    if proc.returncode != 0:
        return "crashed_not_counterexample", None, "error"
    principle_failed = searcher.parse_verdict(proc.stdout or "")
    if principle_failed is True:
        return "counterexample", True, "ok"
    if principle_failed is False:
        return "survived", False, "ok"
    return "inconclusive", None, "ok"


def _require_stamp(cdir: Path, name: str, source: Path) -> Dict[str, Any]:
    stamp_path = cdir / "stamps" / (name + ".json")
    if not stamp_path.exists():
        raise RunError("missing stamp '{}': run the predecessor step first".format(name))
    stamp = _read_json(stamp_path)
    recorded = stamp.get("source_sha256", "")
    if not source.exists():
        raise RunError("stamp '{}' is stale: {} was deleted; redo the step".format(name, source.name))
    if recorded and _sha256_file(source) != recorded:
        raise RunError(
            "stamp '{}' is stale: {} changed after stamping; re-run the step for the new content".format(
                name, source.name)
        )
    return stamp


def _journal(cdir: Path, command: str, payload: Dict[str, Any]) -> None:
    import time
    entry = {"ts": time.time(), "command": command, **payload}
    with (cdir / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "cmd_trigger", "cmd_clusters", "cmd_align", "cmd_induce", "cmd_refute",
    "cmd_validate_pattern", "cmd_append_pattern",
]
