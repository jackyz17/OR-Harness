#!/usr/bin/env python3
"""Farmer problem walkthrough — I (the LLM) generate every modeling/code/synthesis
response, Mock solvers return pre-computed optimal results, no real solver needed.

Run:  PYTHONPATH=src python3 scripts/demo_farmer_walkthrough.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

# -- framework imports --
from or_experience_bank.config import ExperienceBankConfig
from or_experience_bank.llm_client import LLMClient
from or_experience_bank.core.store import AppendOnlyExperienceStore
from or_experience_bank.retrieval.retrieval import ExperienceRetriever
from or_experience_bank.retrieval.index import EmbeddingIndex
from or_experience_bank.solvers.mock import MockSolverAdapter
from or_experience_bank.solvers.registry import SolverRegistry
from or_experience_bank.solving.orchestrator import ORExperienceOrchestrator
from or_experience_bank.core.schemas import SolverExecutionResult

# =====================================================================
# 1. The LLM responses (I generate these as the agent's reasoning)
# =====================================================================

THINK_BLOCK = """\
<think>
This is a resource-constrained profit maximization problem with three integer
decision variables (cows, sheep, chickens). Profit per animal = sell price - feed cost:
  cow    = 500 - 100 = 400
  sheep  = 200 -  80 = 120
  chicken=   8 -   5 =   3

Constraints:
  1. Manure capacity:  10*cows + 5*sheep + 3*chickens <= 800
  2. Chicken upper:    chickens <= 50
  3. Min cows:         cows >= 10
  4. Min sheep:        sheep >= 20
  5. Total animals:    cows + sheep + chickens <= 100

All decision variables are non-negative integers (integer-bounded livestock counts).
The objective is linear, constraints are linear → this is a pure ILP.

Profit-per-manure-unit analysis:
  cow = 400/10 = 40,  sheep = 120/5 = 24,  chicken = 3/3 = 1
So cows dominate → maximize cows first, then sheep, chickens last.
</think>"""

MODEL_BLOCK = """\
<model>
SETS:
  a in Animals = {cow, sheep, chicken}

PARAMETERS:
  sell_price[a]
  feed_cost[a]
  manure_rate[a]
  manure_limit
  max_chickens
  min_cows
  min_sheep
  max_total

VARIABLES:
  x[a] integer >= 0

OBJECTIVE:
  maximize sum(a, (sell_price[a] - feed_cost[a]) * x[a])

CONSTRAINTS:
  C1: sum(a, manure_rate[a] * x[a]) <= manure_limit
  C2: x[chicken] <= max_chickens
  C3: x[cow] >= min_cows
  C4: x[sheep] >= min_sheep
  C5: sum(a, x[a]) <= max_total
</model>"""

SIGNATURE = {
    "objective": "linear",
    "decision": "integer_batch",
    "constraint": ["capacity", "covering"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "shared_scarce", "domain": "livestock_farm"}
}

# Pre-computed optimal: cows=70, sheep=20, chickens=0, profit=30400
OPTIMAL_RESULT_A = SolverExecutionResult(
    status="optimal", solver="mock-a", exit_code=0,
    objective_sense="maximize", objective_value=30400.0,
    runtime_seconds=0.01,
    variables={"cow": 70, "sheep": 20, "chicken": 0},
)
OPTIMAL_RESULT_B = SolverExecutionResult(
    status="optimal", solver="mock-b", exit_code=0,
    objective_sense="maximize", objective_value=30400.0,
    runtime_seconds=0.01,
    variables={"cow": 70, "sheep": 20, "chicken": 0},
)

# Comparative synthesis result (I write the experience lessons)
SYNTHESIS_CANDIDATES = [
    {
        "layer": "modeling",
        "title": "Profit-per-unit-of-scarce-resource ranking for integer allocation",
        "retrieval_text": "When allocating integer quantities under a shared scarce resource constraint, rank decision variables by profit-per-unit-of-resource-consumed and greedily favor the highest-ranked item until a constraint binds. cow=40/manure-unit >> sheep=24 >> chicken=1, so cows are maximized first, then sheep, then chickens.",
        "polarity": "positive",
        "diagnosis": "The key insight is that the manure constraint is the binding scarce resource, not the total-animal count. Computing profit density (profit / resource_consumption) reveals which animal gives the most profit per unit of limited resource.",
        "action": "Calculate profit_per_manure = (sell_price - feed_cost) / manure_rate for each animal. Sort descending. Allocate maximum feasible quantity of the highest-ranked animal first, respecting lower bounds on other variables, then fill remaining capacity with the next-ranked animal.",
        "rationale": "This is the greedy fractional-knapsack insight adapted to ILP: when one linear constraint is binding and variables have different profit density, the LP relaxation optimum aligns with the greedy ranking, and integrality holds when resource coefficients are favorable.",
    },
]

JUDGE_VERDICT = {"accept": True, "reason": "Actionable, accurate, and valuable for future resource-constrained allocation problems."}


# =====================================================================
# 2. Custom LLM client — I am the LLM
# =====================================================================

class FarmerLLMClient:
    """I act as the LLM. Route prompts by keyword to the right pre-generated answer."""

    def __init__(self):
        self.call_log = []  # track every call for display

    async def generate_text(self, prompt: str, timeout=None) -> str:
        self.call_log.append(("generate_text", prompt[:120]))
        p = prompt.lower()
        if "gams-style" in p or "gams" in p or "modeling expert" in p:
            # Modeling stage: think + model
            return THINK_BLOCK + "\n" + MODEL_BLOCK
        if "generate only executable python code" in p:
            # Solver code generation (mock solvers won't actually run it)
            return "# code generation for mock solver (not executed)\nimport json\njson.dump({}, open('result.json','w'))"
        if "extract reusable" in p or "contrasting" in p or "success" in p:
            return json.dumps(SYNTHESIS_CANDIDATES)
        return ""

    async def generate_object(self, prompt: str, timeout=None) -> object:
        self.call_log.append(("generate_object", prompt[:120]))
        p = prompt.lower()
        if "structural signature" in p or "extract a structural" in p:
            return SIGNATURE
        if "judge whether" in p or "admission" in p or "accept" in p:
            return JUDGE_VERDICT
        if "or modeling checker" in p or "semantic" in p or "defect" in p:
            return []  # no semantic defects
        if "extract reusable" in p or "contrasting" in p:
            return SYNTHESIS_CANDIDATES
        return []


# =====================================================================
# 3. Run the full flow
# =====================================================================

PROBLEM = (
    "A farmer needs to decide how many cows, sheep, and chickens to raise "
    "in order to achieve maximum profit. The farmer can sell cows, sheep, and "
    "chickens for $500, $200, and $8 each, respectively. The feed costs for each "
    "cow, sheep, and chicken are $100, $80, and $5, respectively. The profit is "
    "the difference between the selling price and the feed cost. Each cow, sheep, "
    "and chicken produces 10, 5, and 3 units of manure per day, respectively. "
    "Due to the limited time the farm staff has for cleaning the farm each day, "
    "they can handle up to 800 units of manure. Additionally, because of the "
    "limited farm size, the farmer can raise at most 50 chickens. Furthermore, "
    "the farmer must have at least 10 cows to meet customer demand. The farmer "
    "must also raise at least 20 sheep. Finally, the total number of animals "
    "cannot exceed 100."
)

GOLD = 30400


async def main():
    bank_home = Path(mkdtemp(prefix="or-farmer-demo-"))
    print(f"\n{'='*70}")
    print(f"  Farmer Problem Walkthrough — Framework Flow Demo")
    print(f"  Bank home: {bank_home}")
    print(f"{'='*70}\n")

    config = ExperienceBankConfig(bank_home=bank_home)
    config.ensure_directories()
    config.auto_append = True
    config.append_positive = True
    config.stop_on_repeated_error = True
    config.stop_on_unchanged_code = True

    store = AppendOnlyExperienceStore(config.bank_home / "bank")
    index = EmbeddingIndex(str(config.bank_home / "index"))
    retriever = ExperienceRetriever(store, index)
    llm = FarmerLLMClient()

    # Two mock solver branches — both return the optimal result
    adapter_a = MockSolverAdapter(name="mock-a", outcomes=[OPTIMAL_RESULT_A])
    adapter_b = MockSolverAdapter(name="mock-b", outcomes=[OPTIMAL_RESULT_B])
    registry = SolverRegistry()
    registry._adapters = {"mock-a": adapter_a, "mock-b": adapter_b}

    orchestrator = ORExperienceOrchestrator(
        config=config, store=store, retriever=retriever,
        registry=registry, llm_client=llm,
    )

    # --- Step 1: Solve (modeling gate → parallel branches → episode base) ---
    print("=" * 70)
    print("STEP 1: solve(problem, defer_extraction=True)")
    print("  → Modeling gate (think→model→verify L1/L2/L3)")
    print("  → 2 parallel mock solver branches")
    print("  → Episode base recorded (no experience appended yet)")
    print("=" * 70)

    result = await orchestrator.solve(
        PROBLEM, solvers=["mock-a", "mock-b"],
        max_attempts=1, defer_extraction=True,
    )

    print(f"\n  problem_id:        {result.problem_id}")
    print(f"  selected_branch:   {result.selected_branch_id}")
    print(f"  selection_reason:  {result.selection_reason}")
    print(f"  validation_level:  {result.validation_level}")
    print(f"  objective_comparable: {result.objective_comparable}")
    print(f"  retrieved_exp_ids: {result.retrieved_experience_ids}")
    print(f"  appended_exp_ids:  {result.appended_experience_ids}")

    for b in result.branches:
        print(f"\n  Branch {b.branch_id} ({b.solver}):")
        print(f"    termination: {b.termination_reason}")
        if b.execution:
            print(f"    status:      {b.execution.status}")
            print(f"    objective:   {b.execution.objective_value}")
            print(f"    variables:   {b.execution.variables}")

    print(f"\n  Timeline:")
    for event in result.timeline:
        print(f"    {event['event']}: {json.dumps({k:v for k,v in event.items() if k not in ('timestamp','event')}, ensure_ascii=False)}")

    if result.warnings:
        print(f"\n  Warnings: {result.warnings}")

    # --- Step 2: Gold evaluation (comparative synthesis → admission → bank) ---
    print("\n" + "=" * 70)
    print("STEP 2: evaluate_with_gold(gold)")
    print(f"  Gold: objective={GOLD} (cow=70, sheep=20, chicken=0)")
    print("  → Compare selected branch vs gold")
    print("  → Match → comparative synthesis (success vs no failures → plain success)")
    print("  → Admission judge → bank append → Episode gold supplement")
    print("=" * 70)

    verdict = await orchestrator.evaluate_with_gold(GOLD)
    print(f"\n  matched:           {verdict.matched}")
    print(f"  ready_for_extraction: {verdict.ready_for_extraction}")
    print(f"  reason:            {verdict.reason}")

    # --- Show what landed in the bank ---
    print("\n" + "=" * 70)
    print("STEP 3: Bank inspection")
    print("=" * 70)

    modeling_store = orchestrator.modeling_store
    records = modeling_store.all_records()
    validated = modeling_store.validated_records()
    flat_records = list(store.iter_records())
    episodes = list(orchestrator.episode_store.iter_records())

    print(f"\n  Modeling Bank: {len(records)} record(s), {len(validated)} validated")
    for r in records:
        title = r.get("title", "?")
        status = r.get("status") or "null"
        aspect = r.get("modeling_aspect", "?")
        print(f"    [{status}] [{aspect}] {title}")

    print(f"\n  Flat Bank: {len(flat_records)} record(s)")
    for r in flat_records:
        print(f"    [{r.get('layer')}] {r.get('title', '?')}")

    print(f"\n  Episodes: {len(episodes)}")
    for ep in episodes:
        status = ep.get("solve_status", "?")
        matched = ep.get("gold_matched", "—")
        app_ids = ep.get("produced_realization_ids", [])
        print(f"    {ep.get('episode_id','?')[:20]}  status={status}  gold_matched={matched}  produced={app_ids}")

    # --- LLM call log ---
    print("\n" + "=" * 70)
    print("LLM CALL LOG (what the framework asked me, and I answered)")
    print("=" * 70)
    for i, (call_type, prompt_snippet) in enumerate(llm.call_log, 1):
        print(f"  [{i}] {call_type}")
        print(f"      prompt: {prompt_snippet}...")

    # cleanup
    shutil.rmtree(bank_home)
    print(f"\n  (cleaned up {bank_home})")
    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
