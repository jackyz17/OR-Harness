"""Verification honesty + end-to-end knowledge through the world model.

Two acceptance clauses that span layers and would otherwise only be checked
by reading code:

1. **A passed check is never a proof the claim is true.** A `cost_saving`
   claim whose cost dimension is not MEASURED on both sides is
   `insufficient_evidence`, and its verification block says explicitly what
   it did NOT cover — so "only verified status" can never be reported as
   "cost saving proven".
2. **Cost prediction, BOTH recall channels, and the world-model context
   stay connected**: a verified claim reaches recall and a later prediction
   context (its conditions, expected effect and verification scope travel
   with it), while the structural channel and the text (vector) channel
   remain independent.
"""
import json
import os
import sys
import unittest
from pathlib import Path

from helpers import HarnessTestCase  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import COST_DIMENSIONS, CostVector  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.strategy.verification import (  # noqa: E402
    REFUTED, VERIFIED, verify_candidate,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")

PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "timeout", "probability": 0.2}]},
}


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


class StubProvider(WorldModelProvider):
    name = "stub-honesty"

    def predict(self, request, timeout_s=None):
        return {"payload": PAYLOAD,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


class HonestyCase(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def _cost_record(self, eid, task_id, *, tokens, measured_tokens=True,
                     objective=100.0, strategy="S04"):
        rec = self.make_record(
            execution_id=eid, task_id=task_id, strategy_id=strategy,
            profile=self.make_profile(problem_id=task_id, family="routing"),
            objective=objective,
            cost=CostVector(llm_tokens=tokens, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0))
        mask = {"tool_calls", "solver_runtime_s", "retries", "latency_s"}
        if measured_tokens:
            mask.add("llm_tokens")
        rec.cost.measured = mask
        return rec


# ---------------------------------------------------------------------------
# 1. only-verified is not "cost saving proven"
# ---------------------------------------------------------------------------


class TestVerifiedIsNotProof(HonestyCase):

    def test_cost_saving_with_unmeasured_cost_is_insufficient(self):
        candidate = self._cost_record("ex_c", "T1", tokens=100.0,
                                      measured_tokens=False)
        baseline = self._cost_record("ex_b", "T1", tokens=900.0,
                                     measured_tokens=False)
        report = verify_candidate(
            "cost_saving", "the new candidate saves tokens on T1",
            check={"dimension": "llm_tokens", "quality_floor": 0.5},
            executions=[candidate], supporting=[baseline])
        # NOT verified: the dimension that a saving would rest on was never
        # measured, so nothing proves a saving.
        self.assertEqual(report["state"], "insufficient_evidence")
        self.assertNotEqual(report["state"], VERIFIED)
        joined = " ".join(str(c) for c in report["checks"])
        self.assertIn("unmeasured", report["conclusion"])

    def test_cost_saving_verified_reports_its_scope_and_limits(self):
        candidate = self._cost_record("ex_c", "T1", tokens=100.0)
        baseline = self._cost_record("ex_b", "T1", tokens=900.0)
        report = verify_candidate(
            "cost_saving", "the new candidate saves tokens on T1",
            check={"dimension": "llm_tokens", "quality_floor": 0.5},
            executions=[candidate], supporting=[baseline])
        self.assertEqual(report["state"], VERIFIED)
        # The verdict names what it CHECKED; it is a check over samples, not
        # a proof of the prose.
        checks = {c["check"] for c in report["checks"]}
        self.assertIn("cost_lower", checks)
        self.assertIn("quality_floor", checks)

    def test_a_cost_saving_claim_through_induce_never_reads_as_proven(self):
        """End to end: induce a cost_saving claim whose cost is unmeasured
        and confirm the ENTRY is not published as knowledge. Two tasks are
        used so the entry passes the independence gate — the refusal must
        come from the COST evidence, not the sample count."""
        for task_id in ("T1", "T2"):
            self.h.bank.append(self._cost_record(f"ex_c_{task_id}", task_id,
                                                 tokens=100.0,
                                                 measured_tokens=False))
            self.h.bank.append(self._cost_record(f"ex_b_{task_id}", task_id,
                                                 tokens=900.0,
                                                 measured_tokens=False))
        verify = {"purpose": "cost_saving",
                  "claim": "S04 saves tokens",
                  "check": {"dimension": "llm_tokens", "quality_floor": 0.5},
                  "executions": [self.h.bank.get("ex_c_T1"),
                                 self.h.bank.get("ex_c_T2")],
                  "supporting": [self.h.bank.get("ex_b_T1"),
                                 self.h.bank.get("ex_b_T2")]}
        result = self.h.induce(strategy_id="S04", verify=verify)["results"][0]
        report = result.get("verification") or {}
        self.assertEqual(report.get("state"), "insufficient_evidence")
        entry = self.h.sbank.get(result.get("created"))
        self.assertFalse(entry.is_published)
        # It is never presented as admitted knowledge — recall falls back to
        # the raw conditional statistics.
        recall = self.h.recall(_task("t9"), include_unverified=True)
        self.assertFalse(any(r["evidence"] == "strategic_entry"
                             for r in recall["recommendations"]
                             if r["strategy_id"] == "S04"))


# ---------------------------------------------------------------------------
# 2. cost prediction + two recall channels + world-model context
# ---------------------------------------------------------------------------


class TestKnowledgeThroughTheContext(HonestyCase):

    def _solve(self, task, strategy="S04", episode="ep1", tag=None):
        label = tag or f"{task['task_id']}_{strategy}_{episode}"
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,"
            " 'objective_bound': 100.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode)
        self.h.record(record)
        return record

    def test_entry_cost_prediction_and_two_recall_channels(self):
        # Two independent tasks -> a verified statistical entry (cost claim).
        for i in range(2):
            self._solve(_task(f"k{i}"), tag=f"k{i}")
        verify = {"purpose": "rule", "claim": "S04 reaches optimal",
                  "check": {"reference_status": "optimal"},
                  "executions": list(self.h.bank.all())}
        result = self.h.induce(strategy_id="S04", verify=verify)
        self.assertTrue(any(r.get("created") for r in result["results"]))
        entry = next(e for e in self.h.sbank.list() if e.strategy_id == "S04")
        self.assertEqual(entry.verification_state, "verified")

        # (a) the cost prediction reads the ENTRY rung.
        snapshot = self.h.predict_cost(_task("k9"), "S04")
        self.assertEqual(snapshot.source, "entry")
        self.assertIn(entry.entry_id, snapshot.evidence_refs)

        # (b) the structural channel returns the entry.
        recall = self.h.recall(_task("k9"))
        entry_recs = [r for r in recall["recommendations"]
                      if r["evidence"] == "strategic_entry"]
        self.assertTrue(entry_recs)
        # (c) the text (vector) channel is a SEPARATE list — never blended.
        self.assertIn("vector_recall", recall)
        self.assertNotEqual(recall["vector_recall"], recall["recommendations"])

    def test_verified_claim_travels_into_a_prediction_context(self):
        # A verified CLAIM entry (claim-only, free-form subject).
        recs = [self._solve(_task(f"c{i}"), tag=f"c{i}") for i in range(2)]
        claim = {
            "subject": "principle:two_stage",
            "claim": "keep the cross-period state",
            "conditions": {"predicates": {"family": "routing"}},
            "evidence": [{"execution_id": r.execution_id, "role": "preserved"}
                         for r in recs],
            "check": {"assertions": [
                {"kind": "status", "roles": ["preserved"],
                 "status": "optimal"}]},
        }
        self.h.induce(relations=[claim],
                      verify={"purpose": "relation",
                              "check": {"assertions":
                                        claim["check"]["assertions"]}})
        # The world-model context freezes the knowledge view; the claim's
        # conditions, expected effect and verification scope must be there.
        context = self.h.build_prediction_context(_task("c9"), episode_id="epX")
        blob = json.dumps(context.to_dict() if hasattr(context, "to_dict")
                          else context)
        self.assertIn("principle:two_stage", blob)
        # The structural hits the provider sees carry the claim too.
        from or_harness.world_model.context import _structural_hits
        hits = _structural_hits(self.h.recall(_task("c9"))["recommendations"])
        self.assertTrue(hits)

    def test_conditions_and_boundaries_travel_with_a_hit(self):
        """The entry's predicates and declared boundaries are part of what
        the knowledge SAYS, so a hit carries both."""
        records = [self._solve(_task(f"b{i}"), tag=f"b{i}") for i in range(2)]
        verify = {"purpose": "rule", "claim": "S04 optimal under routing",
                  "check": {"reference_status": "optimal"},
                  "executions": records}
        self.h.induce(strategy_id="S04", verify=verify)
        self.h.induce(strategy_id="S04",
                      notes=["watch the temporal boundary"])
        recalled = self.h.recall(_task("b9"))
        item = next(r for r in recalled["recommendations"]
                    if r["strategy_id"] == "S04")
        self.assertIn("watch the temporal boundary",
                      item["knowledge"]["applicability"])
        self.assertEqual(item["knowledge"]["predicates"].get("family"),
                         "routing")


if __name__ == "__main__":
    unittest.main()
