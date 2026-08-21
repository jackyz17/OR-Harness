#!/usr/bin/env python3
"""Phase 3 induction walkthrough: one concrete end-to-end example.

Scenario (straight from the discussion doc): three HETEROGENEOUS problems that share
the SAME underlying "shared scarce resource allocation" structure:
  - inventory     : allocate stock x[i] to a warehouse capacity
  - production    : assign production batches to machine capacity
  - workforce     : assign workers to a labor-hour pool

Expected induction flow:
  candidates  -> 1 isomorphic cross-family cluster
  encoding    -> signatures already present (reused)
  alignment   -> roles: resource_pool / capacity_limit / competing_decisions / objective_contribution
  inducer     -> P0 hypothesis: marginal-contribution-priority allocation
  counterexample -> LLM proposes "fixed setup cost"; solver EXECUTES -> principle survives
  validation  -> source consistency 3/3 + unseen transfer improves -> VALIDATED
  repository  -> depth=2 pattern appended to ModelingStore, sources untouched

Run:  PYTHONPATH=src python3 scripts/demo_induction_walkthrough.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.modeling_schemas import ModelingExperience
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.induction import (
    CounterexampleSearcher,
    InductionPipeline,
    LLMBackedAligner,
    LLMBackedCounterexampleSearcher,
    LLMBackedEncoder,
    LLMBackedInducer,
    PatternInducer,
    PatternValidator,
    SignatureClusterer,
    StructuralAligner,
    StructuralEncoder,
)
from or_experience_bank.llm_client import FakeLLMClient


def banner(title):
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# 1. Seed three heterogeneous records sharing one structure.
# ---------------------------------------------------------------------------
FAMILY = {"e_inv": "inventory", "e_prod": "production", "e_work": "workforce"}
TEXT = {
    "e_inv": "Allocate stock x[i] to warehouse capacity 5000m3 to minimize holding cost.",
    "e_prod": "Assign production batches x[j] to machine capacity 8h/day to maximize profit.",
    "e_work": "Assign workers x[k] to a 40h/week labor pool to minimize shortage cost.",
}
EPISODE = {"e_inv": "ep_inv", "e_prod": "ep_prod", "e_work": "ep_work"}


def make_realization(rid):
    rec = ModelingExperience(title=TEXT[rid], retrieval_text=TEXT[rid])
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict({
        "objective": "linear",
        "decision": ["binary_assignment"],
        "constraint": ["capacity"],
        "interaction": "shared_resource_coupled",
        "features": {"resource": "shared_scarce"},
    })
    rec.method.action_template = TEXT[rid]
    rec.evidence.source_episodes = [EPISODE[rid]]
    rec.experience_id = rid
    rec.compute_content_hash()
    return rec


def family_resolver(record):
    for ep in (record.get("evidence") or {}).get("source_episodes", []):
        rid = ep.replace("ep_", "e_")
        if rid in FAMILY:
            return FAMILY[rid]
    return record.get("experience_id", "")


# ---------------------------------------------------------------------------
# 2. A fake LLM whose queued answers drive the offline induction dialogue.
# ---------------------------------------------------------------------------
def build_llm():
    alignment = {
        "roles": ["resource_pool", "capacity_limit", "competing_decisions", "objective_contribution"],
        "bindings": [
            {"realization_id": "e_inv", "problem_family": "inventory",
             "mapping": {"resource_pool": "warehouse", "capacity_limit": "5000m3",
                         "competing_decisions": "stock x[i]", "objective_contribution": "holding cost"}},
            {"realization_id": "e_prod", "problem_family": "production",
             "mapping": {"resource_pool": "machine", "capacity_limit": "8h/day",
                         "competing_decisions": "batches x[j]", "objective_contribution": "unit profit"}},
            {"realization_id": "e_work", "problem_family": "workforce",
             "mapping": {"resource_pool": "labor pool", "capacity_limit": "40h/week",
                         "competing_decisions": "workers x[k]", "objective_contribution": "shortage cost"}},
        ],
        "confidence": 0.92,
        "notes": "all three share a shared-scarce-resource allocation structure",
    }
    hypothesis = [{
        "statement": ("When multiple decisions compete for a shared scarce resource with a "
                      "quantifiable marginal objective contribution, prioritize allocation to the "
                      "higher marginal-contribution decisions, subject to the coupling constraints."),
        "structural_pattern": "shared scarce resource + marginal contribution allocation",
        "roles_used": ["resource_pool", "capacity_limit", "competing_decisions", "objective_contribution"],
        "applicability_conditions": ["linear objective", "shared resource coupling"],
        "complexity": 0.3,
    }]
    failure_conditions = [["fixed setup cost", "nonconvex coupling"]]
    refutation_programs = [
        "print('instantiate fixed setup cost; check principle')",
        "print('instantiate nonconvex coupling; check principle')",
    ]
    # object consumption order: (1) alignment, (2) induced hypotheses, (3) failure conditions
    return FakeLLMClient(
        text_responses=refutation_programs,
        object_responses=[alignment, hypothesis] + failure_conditions,
    )


# ---------------------------------------------------------------------------
# 3. A stub solver/executor: refutation programs "run" and report survival.
# ---------------------------------------------------------------------------
class StubExecutor:
    async def execute(self, code_path, workspace, solver):
        code = Path(code_path).read_text()
        # the principle SURVIVES both proposed failure conditions in this demo
        return SolverExecutionResult(
            status="ok", solver=solver,
            stdout='{"principle_failed": false, "evidence": "principle holds under this condition"}',
        )


async def transfer_solver(task, principle):
    """Unseen task: WITH the principle we reach a better (lower) objective."""
    return 8.0 if principle is not None else 12.0


# ---------------------------------------------------------------------------
# 4. Walk each stage, printing real inputs/outputs.
# ---------------------------------------------------------------------------
async def main():
    tmp = tempfile.mkdtemp(prefix="induction_demo_")
    store = ModelingStore(Path(tmp))
    for rid in ("e_inv", "e_prod", "e_work"):
        store.append(make_realization(rid))

    banner("STEP 0  ·  Modeling Bank seeded with 3 heterogeneous records")
    for r in store.all_records():
        print("  [{fam:10}] {id}  ::  {title}".format(
            fam=family_resolver(r), id=r["experience_id"], title=r["title"]))

    # --- candidates ---
    banner("STEP 1  ·  candidates.py — isomorphic + heterogeneous cluster discovery")
    clusterer = SignatureClusterer(family_resolver=family_resolver)
    clusters = clusterer.discover(store.all_records())
    for c in clusters:
        print("  cluster {id}: families={fams}  size={n}  score={s:.0f}".format(
            id=c.cluster_id, fams=c.problem_families, n=c.size, s=c.score))
        print("    core_key = " + c.core_key)
        print("    shared_feature_keys = " + str(c.shared_feature_keys))
    cluster = clusters[0]

    # --- encoding ---
    banner("STEP 2  ·  encoding.py — batch structural encoding (signatures reused)")
    encoder = StructuralEncoder()
    for r in store.all_records():
        res = encoder.encode(r)
        sig = res.signature
        print("  {id}: status={st}  O={o} D={d} C={c} I={i} features={f}".format(
            id=r["experience_id"], st=res.status, o=sig.objective,
            d=sig.decision, c=sig.constraint, i=sig.interaction, f=sig.features))

    # --- alignment ---
    banner("STEP 3  ·  alignment.py — cross-memory role correspondence")
    llm = build_llm()
    alignment = await LLMBackedAligner(StructuralAligner(), llm).align(cluster)
    print("  shared roles: " + ", ".join(alignment.roles))
    for b in alignment.bindings:
        print("  - {fam:10} ({rid}):".format(fam=b.problem_family, rid=b.realization_id))
        for role, entity in b.mapping.items():
            print("        {role:22} <- {entity}".format(role=role, entity=entity))

    # --- induction (hypothesis) ---
    banner("STEP 4  ·  inducer.py — hypothesis generation (status=hypothesis, NOT knowledge)")
    hypotheses = await LLMBackedInducer(PatternInducer(), llm).induce(cluster, alignment)
    hyp = hypotheses[0]
    print("  P0 hypothesis: " + hyp.statement)
    print("  status = {st}   complexity = {cx}".format(st=hyp.status, cx=hyp.complexity))

    # --- counterexample search ---
    banner("STEP 5  ·  counterexample.py — solver-backed refutation (executor decides, not LLM)")
    cx_searcher = LLMBackedCounterexampleSearcher(
        CounterexampleSearcher(executor=StubExecutor()), llm)
    refutation = await cx_searcher.search(hyp, Path(tmp) / "ws" / hyp.hypothesis_id)
    for a in refutation.attempts:
        verdict = "REFUTED" if a.is_counterexample else "survived"
        print("  condition '{cond}'  -> executed={ex} principle_failed={pf}  [{v}]".format(
            cond=a.condition, ex=a.executed, pf=a.principle_failed, v=verdict))
    print("  refuted = " + str(refutation.refuted))

    # --- validation ---
    banner("STEP 6  ·  validation.py — source consistency + UNSEEN transfer + scoring")
    validator = PatternValidator(validation_threshold=0.5)
    transfer_tests = await validator.unseen_transfer(hyp, ["unseen: cloud-resource allocation"], transfer_solver)
    for t in transfer_tests:
        print("  unseen task '{t}': with_P={w}  without_P={wo}  -> {imp}".format(
            t=t.task, w=t.with_principle_objective, wo=t.without_principle_objective, imp=t.improvement))
    source_texts = [TEXT[r] for r in ("e_inv", "e_prod", "e_work")]
    from or_experience_bank.core.modeling_schemas import PatternValidation
    validation = PatternValidation(
        source_consistency="3/3 sources consistent", transfer_tests=transfer_tests)
    scoring = validator.score(hyp, ["e_inv", "e_prod", "e_work"], transfer_tests, refutation, source_texts)
    outcome = validator.decide(hyp, scoring, validation, refutation)
    print("  scoring: C={c:.2f} T={t:.2f} V={v:.2f} N={n:.2f} K={k:.2f} X={x:.2f}  total={tot:.2f}".format(
        c=scoring.coverage, t=scoring.transferability, v=scoring.validation,
        n=scoring.novelty, k=scoring.complexity, x=scoring.counterexample_penalty, tot=scoring.total))
    print("  decision: " + outcome.status.upper())

    # --- repository ---
    banner("STEP 7  ·  pipeline.py — validated pattern appended to ModelingStore (append-only)")
    # The pipeline drives align -> induce -> refute in sequence over ONE shared LLM,
    # so we give it a single FakeLLMClient whose object queue is ordered accordingly:
    #   (1) alignment, (2) induced hypotheses, (3) failure conditions.
    shared_llm = build_llm()
    pipeline = InductionPipeline(
        store=store, clusterer=clusterer,
        aligner=LLMBackedAligner(StructuralAligner(), shared_llm),
        inducer=LLMBackedInducer(PatternInducer(), shared_llm),
        counterexample=LLMBackedCounterexampleSearcher(
            CounterexampleSearcher(executor=StubExecutor()), shared_llm),
        validator=validator, transfer_solver=transfer_solver,
        workspace=Path(tmp) / "ws", unseen_tasks=["unseen: cloud-resource allocation"],
    )
    report = await pipeline.run()
    print("  report: clusters={c} hypotheses={h} validated={v} refuted={r}".format(
        c=report.clusters_found, h=report.hypotheses_generated,
        v=report.patterns_validated, r=report.patterns_refuted))
    for p in store.validated_records():
        print("  appended pattern {id}: depth={d} status={s}".format(
            id=p["experience_id"], s=p["status"]))
        print("    principle: " + p.get("method", {}).get("action_template", ""))
        print("    derived_from: " + str(sorted(p["derived_from_experience_ids"])))
        print("    role_schema roles: " + ", ".join(p["role_schema"].keys()))
    print("  source records still present (untouched): " + str(len(store.all_records())))

    banner("SUMMARY  ·  Induction != Summary")
    print("  - 3 heterogeneous records (inventory/production/workforce) shared ONE structure.")
    print("  - Alignment exposed the role correspondence (resource_pool <-> resource_pool ...).")
    print("  - The induced principle was a HYPOTHESIS until it survived solver refutation AND")
    print("    improved an UNSEEN task (8.0 < 12.0). Only then was it appended as a depth=2 pattern.")
    print("  - Sources remain untouched (append-only). P not in any single M_i, yet Transfer(P)>0.")


if __name__ == "__main__":
    asyncio.run(main())
