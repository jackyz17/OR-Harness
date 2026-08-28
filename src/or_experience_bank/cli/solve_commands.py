"""Online solve-chain commands: recall -> validate -> signature -> hints -> solve
-> cross-validate -> gold -> append -> episode (+ new-round).

Every command is a pure function: (run_dir files, bank) -> (files, stdout JSON).
Chain enforcement is via stamps (see run_store.py): a command checks that the
predecessor artifact exists and its content hash is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.modeling_schemas import (
    BranchSummary,
    EpisodeRecord,
    MathScope,
    MethodBody,
    ModelingEvidence,
    ModelingExperience,
    StructuralSignature,
)
from ..modeling.modeling_contract import FormatValidator, StructuralValidator, parse_modeling_output
from ..modeling.signature_extractor import SignatureExtractor
from .components import Components, SOLVER_FAMILY
from .run_store import RunError, RunStore, create_run

_USES_PAT = re.compile(r"\[uses\s+(E\d+(?:\s*,\s*E\d+)*)\]", re.IGNORECASE)


def _hit_to_dict(hit: Any) -> Dict[str, Any]:
    return {
        "experience_id": hit.experience_id,
        "title": hit.title,
        "polarity": hit.polarity,
        "score": round(float(hit.score), 4),
        "retrieval_text": hit.retrieval_text,
    }


def _truncate_variables(variables: Dict[str, Any], limit: int = 50) -> Dict[str, Any]:
    """Keep the solver's variable values for debugging, capped at `limit` entries."""
    if not isinstance(variables, dict) or not variables:
        return {}
    items = list(variables.items())[:limit]
    kept = dict(items)
    if len(variables) > limit:
        kept["_truncated"] = "{} more variables omitted".format(len(variables) - limit)
    return kept


# ---------------------------------------------------------------------------
# recall: create the run + fetch planning priors
# ---------------------------------------------------------------------------

def cmd_recall(
    comps: Components,
    run_dir: Path,
    problem_file: Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    problem_text = Path(problem_file).read_text(encoding="utf-8")
    run = create_run(run_dir, problem_text)

    priors = comps.modeling_retriever.retrieve_priors(problem_text, top_k=top_k)
    if not priors.records and list(comps.modeling_store.all_records()):
        comps.modeling_retriever.rebuild()
        priors = comps.modeling_retriever.retrieve_priors(problem_text, top_k=top_k)

    formatted: List[Dict[str, Any]] = []
    for idx, rec in enumerate(priors.records, start=1):
        formatted.append({
            "tag": "E{}".format(idx),
            "experience_id": rec.get("experience_id"),
            "title": rec.get("title"),
            "modeling_aspect": rec.get("modeling_aspect"),
            "status": rec.get("status"),
            "retrieval_text": rec.get("retrieval_text"),
        })
    labels = {"E{}".format(idx): rec.get("experience_id", "")
              for idx, rec in enumerate(priors.records, start=1)}

    payload = {
        "priors_count": len(formatted),
        "priors": formatted,
        "labels": labels,
        "instruction": (
            "Read the priors (cat priors.json). Compose model.txt with <think>/<model> "
            "tags; cite applied priors inside <think> as [uses En]. Then run `orx validate`."
        ),
    }
    _atomic_write(run.priors_path, payload)
    run.journal("recall", {"priors_count": len(formatted)})
    return {
        "priors_count": len(formatted),
        "priors_file": run.priors_path.name,
        "top_titles": [p.get("title") for p in formatted[:3]],
        "run_dir": str(run.dir),
        "next": "write model.txt, then `orx validate`",
    }


# ---------------------------------------------------------------------------
# validate: L1+L2 gate on model.txt -> stamp
# ---------------------------------------------------------------------------

def cmd_validate(comps: Components, run_dir: Path) -> Dict[str, Any]:
    run = RunStore(run_dir)
    if not run.model_path.exists():
        raise RunError("model.txt not found: write your <think>/<model> response there first")
    raw = run.model_path.read_text(encoding="utf-8")

    parsed = parse_modeling_output(raw)
    fmt = FormatValidator().validate(parsed["think"], parsed["model"])
    if not fmt.passed:
        run.journal("validate", {"passed": False, "layer": "L1"})
        return {"passed": False, "issues": [i.to_dict() for i in fmt.issues],
                "hint": "Both <think>...</think> and <model>...</model> must be closed; "
                        "SETS/PARAMETERS/VARIABLES/OBJECTIVE/CONSTRAINTS must all be present."}

    struct = StructuralValidator().validate(parsed["model"])
    if not struct.passed:
        run.journal("validate", {"passed": False, "layer": "L2"})
        return {"passed": False, "issues": [i.to_dict() for i in struct.issues],
                "hint": "Every symbol in OBJECTIVE/CONSTRAINTS must be declared in "
                        "SETS/PARAMETERS/VARIABLES; constraint labels must be C1, C2, ..."}

    # Citations: only tags present in priors.json labels count (no invented citations).
    labels = run.prior_labels()
    cited: List[str] = []
    for m in _USES_PAT.finditer(parsed["think"] or ""):
        for tag in re.split(r"\s*,\s*", m.group(1)):
            eid = labels.get(tag.upper())
            if eid and eid not in cited:
                cited.append(eid)

    stamp = {
        "passed": True,
        "source": "model.txt",
        "source_sha256": _sha256(run.model_path),
        "cited_prior_ids": cited,
        "think_chars": len(parsed["think"] or ""),
    }
    run.write_stamp("model", stamp)
    run.journal("validate", {"passed": True, "cited_priors": len(cited)})
    return {
        "passed": True,
        "cited_priors": len(cited),
        "next": "write signature.json (structural signature), then `orx signature`",
    }


# ---------------------------------------------------------------------------
# signature: vocabulary gate on signature.json -> stamp
# ---------------------------------------------------------------------------

def cmd_signature(comps: Components, run_dir: Path) -> Dict[str, Any]:
    run = RunStore(run_dir)
    run.require_stamp("model", run.model_path)  # chain: model must be validated & unchanged
    if not run.signature_path.exists():
        raise RunError("signature.json not found: write your structural signature there first")

    try:
        raw = json.loads(run.signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunError("signature.json is not valid JSON: {}".format(exc))

    result = SignatureExtractor().parse_and_validate(raw)
    if not result.valid:
        run.journal("signature", {"passed": False})
        return {"passed": False, "errors": result.errors, "retry_hint": result.retry_hint,
                "hint": "Fix signature.json and re-run `orx signature` (model stamp is unaffected)."}

    stamp = {
        "passed": True,
        "source": "signature.json",
        "source_sha256": _sha256(run.signature_path),
        "signature": result.signature.to_dict() if result.signature else None,
    }
    run.write_stamp("signature", stamp)
    run.journal("signature", {"passed": True})
    return {
        "passed": True,
        "signature": stamp["signature"],
        "next": "`orx hints --solver <name>`, write branches/<solver>/solve.py, then `orx solve --solver <name>`",
    }


# ---------------------------------------------------------------------------
# hints: pull bank hints BEFORE code generation
# ---------------------------------------------------------------------------

def cmd_hints(comps: Components, run_dir: Path, solver: str) -> Dict[str, Any]:
    run = RunStore(run_dir)
    run.require_stamp("signature", run.signature_path)  # chain: signature verified & unchanged
    solver_family = SOLVER_FAMILY.get(solver, "milp")

    impl_hits = comps.retriever.retrieve(
        "implementation",
        "Solver: {}\nSolver family: {}\nNeed: implement model, solve, write result.json".format(solver, solver_family),
        metadata_filters={"solver": solver, "solver_family": solver_family},
        top_k=3,
    )
    # Repair hints keyed to the latest error seen in this branch (if any).
    branch = run.branch_results().get(solver)
    repair_hits: List[Dict[str, Any]] = []
    repair_graph: Dict[str, Any] = {}
    if branch and branch.get("normalized_error"):
        error = branch["normalized_error"]
        repair_hits = comps.retriever.retrieve(
            "repair",
            "Solver: {}\nNormalized error: {}\nStatus: {}".format(solver, error, branch.get("status")),
            metadata_filters={"solver": solver, "solver_family": solver_family},
            top_k=3,
        )
        try:
            guidance = comps.retriever.repair_guidance(solver, error)
            repair_graph = {
                "actions": guidance.get("actions", [])[:3],
                "pitfalls": guidance.get("pitfalls", [])[:3],
                "path": guidance.get("repair_path"),
            }
        except (AttributeError, KeyError, TypeError):
            repair_graph = {}

    payload = {
        "solver": solver,
        "solver_family": solver_family,
        "implementation_hints": [_hit_to_dict(h) for h in impl_hits],
        "repair_hints": [_hit_to_dict(h) for h in repair_hits],
        "repair_graph_guidance": repair_graph,
        "result_contract": (
            "Write result.json in the branch directory with fields: status, solver, "
            "objective_sense, objective_value, objective_bound, mip_gap, runtime_seconds, "
            "variables, diagnostics, message."
        ),
    }
    hints_path = run.branch_dir(solver) / "hints.json"
    _atomic_write(hints_path, payload)
    run.journal("hints", {"solver": solver, "impl_hints": len(impl_hits), "repair_hints": len(repair_hits)})
    return {
        "solver": solver,
        "hints_file": str(hints_path.relative_to(run.dir)),
        "implementation_hint_count": len(impl_hits),
        "repair_hint_count": len(repair_hits),
        "next": "read hints (cat {}), write branches/{}/solve.py, then `orx solve --solver {}`".format(
            hints_path.relative_to(run.dir), solver, solver),
    }


# ---------------------------------------------------------------------------
# solve: sandbox-execute branches/<solver>/solve.py -> result.json + hints
# Single-solver form (one branch, typically a repair retry) and parallel form
# (multiple branches concurrently, mirroring the orchestrator's
# asyncio.gather + Semaphore semantics: branches run in PARALLEL, repair
# WITHIN a branch is the agent's sequential retry loop).
# ---------------------------------------------------------------------------

def cmd_solve(comps: Components, run_dir: Path, solver: str) -> Dict[str, Any]:
    """Execute ONE branch (or a repair retry of one branch)."""
    run = RunStore(run_dir)
    run.require_stamp("signature", run.signature_path)  # chain: signature verified & unchanged
    return _execute_branch(comps, run, solver)


def cmd_solve_parallel(comps: Components, run_dir: Path, solvers: List[str]) -> Dict[str, Any]:
    """Execute MULTIPLE branches concurrently (heterogeneous parallel exploration).

    Mirrors the orchestrator contract: all listed branches run via
    asyncio.gather bounded by max_parallel_branches; each branch's result.json
    is written independently. Repair within a branch remains the agent's
    sequential retry (fix solve.py, re-run solve for that solver).
    """
    run = RunStore(run_dir)
    run.require_stamp("signature", run.signature_path)

    solvers = [s.strip() for s in solvers if s.strip()]
    if not solvers:
        raise RunError("no solvers given: pass --solver a,b,c or --parallel --solver a --solver b")
    missing = [s for s in solvers if not (run.branch_dir(s) / "solve.py").exists()]
    if missing:
        raise RunError(
            "branches missing solve.py: {} (write each branch's code first; "
            "`orx hints --solver <name>` pulls bank hints before codegen)".format(", ".join(missing))
        )

    async def _gather() -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, comps.config.max_parallel_branches))

        async def guarded(solver: str) -> Dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(_execute_branch, comps, run, solver)

        return await asyncio.gather(*(guarded(s) for s in solvers))

    results = asyncio.run(_gather())
    valid = [r for r in results if r["valid"] and r["status"] in {"optimal", "feasible"}]
    minimum = max(2, comps.config.min_cross_validation_branches)
    return {
        "parallel": True,
        "branches": results,
        "branches_total": len(results),
        "branches_valid": len(valid),
        "next": (
            "all requested branches executed; run more branches or `orx cross-validate`"
            if len(valid) >= minimum else
            "fewer than {} valid branches: read repair_hints in the failing branches' "
            "result.json, fix each solve.py, re-run `orx solve --solver <failed>`".format(minimum)
        ),
    }


def _execute_branch(comps: Components, run: RunStore, solver: str) -> Dict[str, Any]:
    """Run one branch end-to-end: sandbox execute -> validate -> hints -> result.json."""
    branch_dir = run.branch_dir(solver)
    code_path = branch_dir / "solve.py"
    if not code_path.exists():
        raise RunError(
            "{} not found: write the complete solver script there first "
            "(`orx hints --solver {}` pulls bank hints before codegen)".format(code_path, solver)
        )

    exec_res = asyncio.run(comps.executor.execute(code_path, branch_dir, solver))
    val_res = comps.validator.validate(exec_res)

    solver_family = SOLVER_FAMILY.get(solver, "milp")
    impl_hits = comps.retriever.retrieve(
        "implementation",
        "Solver: {}\nSolver family: {}\nNeed: implement model, solve, write result.json".format(solver, solver_family),
        metadata_filters={"solver": solver, "solver_family": solver_family},
        top_k=3,
    )
    repair_hits: List[Dict[str, Any]] = []
    repair_graph: Dict[str, Any] = {}
    if exec_res.normalized_error:
        repair_hits = comps.retriever.retrieve(
            "repair",
            "Solver: {}\nNormalized error: {}\nStatus: {}".format(solver, exec_res.normalized_error, exec_res.status),
            metadata_filters={"solver": solver, "solver_family": solver_family},
            top_k=3,
        )
        try:
            guidance = comps.retriever.repair_guidance(solver, exec_res.normalized_error)
            repair_graph = {
                "actions": guidance.get("actions", [])[:3],
                "pitfalls": guidance.get("pitfalls", [])[:3],
                "path": guidance.get("repair_path"),
            }
        except (AttributeError, KeyError, TypeError):
            repair_graph = {}

    symptom = ""
    if exec_res.status == "timeout":
        symptom = "timeout"
    elif exec_res.mip_gap is not None and exec_res.mip_gap > 0.05:
        symptom = "large MIP gap"
    elif "numerical" in (exec_res.message or "").lower():
        symptom = "numerical warning"
    solving_hits: List[Dict[str, Any]] = []
    if symptom:
        solving_hits = comps.retriever.retrieve(
            "solving",
            "Solver: {}\nSymptom: {}\nStatus: {}\nMIP gap: {}".format(solver, symptom, exec_res.status, exec_res.mip_gap),
            metadata_filters={"solver": solver, "solver_family": solver_family},
            top_k=3,
        )

    record = {
        "branch": branch_dir.name,
        "solver": solver,
        "status": exec_res.status,
        "objective_value": exec_res.objective_value,
        "objective_bound": exec_res.objective_bound,
        "mip_gap": exec_res.mip_gap,
        "runtime_seconds": exec_res.runtime_seconds,
        "normalized_error": exec_res.normalized_error,
        "valid": val_res.valid,
        "validation_errors": val_res.errors,
        "performance_symptom": symptom or None,
        # Debug payload from the solver's own result.json (kept for inspection;
        # truncated to keep the record readable).
        "variables": _truncate_variables(exec_res.variables),
        "message": exec_res.message,
        "implementation_hints": [_hit_to_dict(h) for h in impl_hits],
        "repair_hints": [_hit_to_dict(h) for h in repair_hits],
        "repair_graph_guidance": repair_graph,
        "solving_hints": [_hit_to_dict(h) for h in solving_hits],
    }
    _atomic_write(branch_dir / "result.json", record)
    run.journal("solve", {"solver": solver, "status": exec_res.status, "valid": val_res.valid})

    if val_res.valid and exec_res.status in {"optimal", "feasible"}:
        next_step = "run another solver branch or `orx cross-validate`"
    else:
        next_step = "read repair_hints in branches/{}/result.json, fix solve.py, re-run `orx solve --solver {}`".format(
            branch_dir.name, solver)
    return {
        "branch": branch_dir.name,
        "solver": solver,
        "status": exec_res.status,
        "objective_value": exec_res.objective_value,
        "valid": val_res.valid,
        "normalized_error": exec_res.normalized_error,
        "performance_symptom": symptom or None,
        "result_file": str((branch_dir / "result.json").relative_to(run.dir)),
        "next": next_step,
    }


# ---------------------------------------------------------------------------
# cross-validate: >=min_cross_validation_branches (config, default 3) valid branches with matching objectives
# ---------------------------------------------------------------------------

def cmd_cross_validate(comps: Components, run_dir: Path, tolerance: float = 1e-4) -> Dict[str, Any]:
    run = RunStore(run_dir)
    run.require_stamp("signature", run.signature_path)

    minimum = max(2, comps.config.min_cross_validation_branches)
    valid = run.valid_branches()
    if len(valid) < minimum:
        run.journal("cross-validate", {"consistent": False, "reason": "insufficient branches"})
        return {
            "consistent": False,
            "reason": "need >={} valid branches with numeric objective, found {}".format(minimum, len(valid)),
            "branches": run.branch_results(),
            "next": "add more solver branches (`orx solve --solver <other>`); the minimum is "
                    "configurable via min_cross_validation_branches",
        }

    objs = [float(b["objective_value"]) for b in valid]
    max_diff = max(objs) - min(objs)
    rel_diff = max_diff / max(1.0, max(abs(x) for x in objs))
    consistent = rel_diff <= tolerance

    payload = {
        "consistent": consistent,
        "relative_diff": rel_diff,
        "tolerance": tolerance,
        "branches_compared": len(valid),
        "objectives": objs,
        "best_objective": objs[0],
    }
    _atomic_write(run.cross_validation_path, payload)
    run.journal("cross-validate", {"consistent": consistent, "rel_diff": rel_diff})
    if consistent:
        return {**payload, "next": "compare best_objective with the USER-PROVIDED gold, then `orx gold`"}
    return {**payload, "next": "branches disagree: add a third branch to triangulate, then re-run `orx cross-validate`"}


# ---------------------------------------------------------------------------
# gold: record the user-provided gold verdict (the gold gate)
# ---------------------------------------------------------------------------

def cmd_gold(
    comps: Components,
    run_dir: Path,
    gold: Optional[float],
    matched: Optional[bool] = None,
) -> Dict[str, Any]:
    run = RunStore(run_dir)
    if run.episode_path.exists():
        raise RunError(
            "this run is complete (episode.json exists): gold cannot be re-recorded "
            "on a finished run — episodes are append-only facts. If the gold was "
            "recorded incorrectly, start a FRESH run (`orx recall` in a new directory) "
            "and re-solve with the correct gold."
        )
    if not run.cross_validation_path.exists():
        raise RunError("cross_validation.json missing: run `orx cross-validate` first")
    cv = json.loads(run.cross_validation_path.read_text(encoding="utf-8"))
    if not cv.get("consistent"):
        raise RunError("branches are not cross-consistent; fix the model before recording gold")

    if gold is None:
        # No gold available: consistency-only validation (recorded explicitly).
        payload = {"gold_answer": None, "gold_matched": True, "basis": "consistency_only"}
        _atomic_write(run.gold_path, payload)
        run.journal("gold", {"basis": "consistency_only"})
        return {**payload,
                "warning": "cross-solver consistency does NOT prove correctness",
                "next": "`orx append` (consistency-only) then `orx episode`"}

    if matched is None:
        best = cv.get("best_objective")
        matched = best is not None and abs(float(gold) - float(best)) <= 1e-6 * max(1.0, abs(float(gold)))
    payload = {"gold_answer": gold, "gold_matched": bool(matched), "basis": "user_provided"}
    _atomic_write(run.gold_path, payload)
    run.journal("gold", {"gold": gold, "matched": bool(matched)})
    if matched:
        return {**payload, "next": "`orx append` experiences, then `orx episode`"}
    return {**payload,
            "next": "DO NOT append. Reflect on the modeling direction, revise model.txt, "
                    "`orx validate` again (or `orx new-round` to archive this round)"}


# ---------------------------------------------------------------------------
# append: admit one experience to the bank (gold gate enforced)
# ---------------------------------------------------------------------------

def cmd_append(comps: Components, run_dir: Path, experience_file: Path) -> Dict[str, Any]:
    run = RunStore(run_dir)
    if not run.gold_path.exists():
        raise RunError(
            "gold.json missing: the gold gate is enforced here. Run `orx gold` first "
            "(user-provided gold, or explicitly consistency-only)"
        )
    gold = json.loads(run.gold_path.read_text(encoding="utf-8"))
    if not gold.get("gold_matched"):
        raise RunError(
            "gold mismatch recorded: appending wrong-model experiences is forbidden. "
            "Reflect and re-model (`orx new-round`), or fix gold.json if it was mis-recorded"
        )

    try:
        experience = json.loads(Path(experience_file).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunError("experience file is not valid JSON: {}".format(exc))
    if not isinstance(experience, dict):
        raise RunError("experience file must contain a JSON object")

    problem_id = run.problem_id()
    branch_ids = sorted(run.branch_results().keys())
    layer = experience.get("layer", "modeling")

    if layer == "modeling":
        sig_stamp = run.read_stamp("signature")
        signature = StructuralSignature.from_dict(sig_stamp.get("signature") or {})
        record = ModelingExperience(
            title=experience.get("title", "Untitled modeling insight"),
            polarity=experience.get("polarity", "positive"),
            retrieval_text=experience.get("retrieval_text", ""),
            modeling_aspect=experience.get("modeling_aspect", "structure"),
            math_scope=MathScope(structural_signature=signature),
            method=MethodBody(
                action_template=experience.get("action", ""),
                wrong_form=experience.get("wrong_form"),
                rationale=experience.get("rationale", ""),
                derivation_ref=experience.get("derivation_ref"),
            ),
            evidence=ModelingEvidence(
                source_episodes=[problem_id],
                solver_feedback_summary=experience.get("diagnosis", ""),
                validation_level="solver_feasible",
                causal_confidence="medium",
            ),
        )
        result = comps.modeling_store.append(record)
        comps.modeling_retriever.rebuild()
        exp_id = result.get("experience_id") or record.experience_id
    else:
        from ..experience.admission import candidate_to_record
        solver_name = experience.get("solver")
        rec = candidate_to_record(
            experience,
            problem_id=problem_id,
            problem_family=experience.get("problem_family", "general_milp"),
            branch_ids=branch_ids,
            attempt_ids=[],
            solver=solver_name,
            solver_family=SOLVER_FAMILY.get(solver_name, "milp") if solver_name else None,
        )
        result = comps.store.append(rec)
        comps.retriever.rebuild(layer)
        exp_id = result.experience_id

    status = result.get("status", "appended") if isinstance(result, dict) else result.status
    run.record_append({"experience_id": exp_id, "layer": layer, "status": status, "title": experience.get("title", "")})
    run.journal("append", {"layer": layer, "status": status, "experience_id": exp_id})
    return {
        "status": status,
        "experience_id": exp_id,
        "layer": layer,
        "appended_total": run.appended_count(),
        "next": "append more lessons (one file each) or finish with `orx episode`",
    }


# ---------------------------------------------------------------------------
# episode: terminal record + utility attribution
# ---------------------------------------------------------------------------

def cmd_episode(
    comps: Components,
    run_dir: Path,
    gold_answer: Optional[float] = None,
) -> Dict[str, Any]:
    run = RunStore(run_dir)
    if not run.cross_validation_path.exists():
        raise RunError("cross_validation.json missing: run `orx cross-validate` first")
    if run.episode_path.exists():
        raise RunError(
            "episode.json already exists: this run is complete and its episode is an "
            "append-only fact (never amended). If the gold was recorded incorrectly, "
            "start a FRESH run (`orx recall` in a new directory) and re-solve — the "
            "corrected episode will be recorded there."
        )

    gold = {}
    if run.gold_path.exists():
        gold = json.loads(run.gold_path.read_text(encoding="utf-8"))
    gold_matched = bool(gold.get("gold_matched", True))
    if gold_answer is None:
        gold_answer = gold.get("gold_answer")

    sig_stamp = run.read_stamp("signature")
    model_stamp = run.read_stamp("model")
    branches = [
        BranchSummary(solver=b.get("solver", ""), status=b.get("status", "unknown"),
                      attempts=1, objective_value=b.get("objective_value"))
        for b in run.branch_results().values()
    ]
    appended_ids = run.appended_ids()

    episode = EpisodeRecord(
        problem=run.problem_text(),
        problem_id=run.problem_id(),
        final_objective=branches[0].objective_value if branches else None,
        produced_realization_ids=appended_ids,
    )
    episode.structural_signature = StructuralSignature.from_dict(sig_stamp.get("signature") or {})
    episode.branches = branches
    episode.normalized_spec = {
        "problem_family": (sig_stamp.get("signature") or {}).get("features", {}).get("domain", "general_milp"),
        "verified_model": _model_text(run),
        "status": "success" if gold_matched else "gold_mismatched",
    }
    episode_id = comps.episode_store.record_episode(episode)
    if gold_answer is not None:
        comps.episode_store.record_gold_supplement(
            problem_id=run.problem_id(),
            gold=gold_answer,
            matched=gold_matched,
            produced_realization_ids=appended_ids,
        )

    # Utility attribution: credit every cited prior on gold match (closes the loop).
    credited = 0
    cited = model_stamp.get("cited_prior_ids", [])
    if gold_matched and cited:
        for eid in cited:
            comps.utility_tracker.record_utility(eid)
            credited += 1

    payload = {
        "recorded": True,
        "episode_id": episode_id,
        "problem_id": run.problem_id(),
        "produced_realizations": len(appended_ids),
        "utility_credited": credited,
        "cited_priors": len(cited),
        "gold_answer": gold_answer,
        "gold_matched": gold_matched,
        "status": "SOLVE_FLOW_COMPLETE",
    }
    # Auto-check the induction trigger: the solve flow just appended new
    # realizations, so this is exactly when accumulation may have crossed the
    # watermark. Surfacing the decision here means the agent never needs an
    # external reminder — the observation itself carries the next obligation.
    payload["induction_check"] = _check_induction_trigger(comps)
    _atomic_write(run.episode_path, payload)
    run.journal("episode", {"episode_id": episode_id, "credited": credited})
    return payload


def _check_induction_trigger(comps: Components) -> Dict[str, Any]:
    """Run the 3-gate trigger decision and shape it as an agent-facing hint."""
    from ..induction.candidates import SignatureClusterer
    from ..induction.trigger import InductionTrigger

    clusterer = SignatureClusterer(lifecycle=comps.lifecycle, utility_tracker=comps.utility_tracker)
    trigger = InductionTrigger(store=comps.modeling_store, clusterer=clusterer)
    decision = trigger.decide()
    check = decision.to_dict()
    if check.get("should_induce"):
        check["instruction"] = (
            "Accumulation crossed the induction watermark: run `orx clusters` and process "
            "each candidate cluster (align -> induce -> refute -> validate-pattern -> "
            "append-pattern) BEFORE starting the next solve."
        )
    else:
        check["instruction"] = (
            "Induction not due yet ({}). No action needed; keep solving.".format(
                check.get("reason", "gates not satisfied")
            )
        )
    return check


# ---------------------------------------------------------------------------
# new-round: archive current artifacts for an outer reflection round
# ---------------------------------------------------------------------------

def cmd_new_round(comps: Components, run_dir: Path) -> Dict[str, Any]:
    run = RunStore(run_dir)
    if run.episode_path.exists():
        raise RunError(
            "run already complete (episode.json exists): reflection rounds happen "
            "BEFORE `orx episode`. If the gold was recorded incorrectly after "
            "completing this run, start a FRESH run (`orx recall` in a new directory) "
            "instead of re-rounding this one."
        )
    n = run.archive_round()
    run.journal("new-round", {"archived_to": "rounds/{}".format(n)})
    return {
        "archived_to": "rounds/{}".format(n),
        "kept": ["problem.txt", "priors.json"],
        "next": "revise your modeling direction, write model.txt, then `orx validate`",
    }


# ---------------------------------------------------------------------------
# status: where am I, what's next
# ---------------------------------------------------------------------------

def cmd_status(comps: Components, run_dir: Path) -> Dict[str, Any]:
    return RunStore(run_dir).phase()


# -- helpers -----------------------------------------------------------------

def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _model_text(run: RunStore) -> str:
    if not run.model_path.exists():
        return ""
    parsed = parse_modeling_output(run.model_path.read_text(encoding="utf-8"))
    return parsed.get("model") or ""


__all__ = [
    "cmd_recall", "cmd_validate", "cmd_signature", "cmd_hints", "cmd_solve",
    "cmd_solve_parallel", "cmd_cross_validate", "cmd_gold", "cmd_append",
    "cmd_episode", "cmd_new_round", "cmd_status",
]
