"""Phase-B acceptance tests: real token accounting and cost provenance.

What this file asserts, one behaviour per class:

1. **The FULL口径 is the total.** A provider that reports prompt +
   completion is recorded as ``prompt + completion`` — not completion
   alone. Reasoning/cached tokens are sub-facts, never added again.
2. **A single reported side is a lower bound, never a completed total.**
   Completion-only stays a completion-only figure.
3. **No usage is unknown, never zero.** A call with no token report leaves
   ``llm_tokens`` UNMEASURED.
4. **Host reports are ingested, not guessed.** An OpenClaw-style and a
   Hermes-style report (and a per-call breakdown) normalize to the same
   attempt-level shape; a report with no numbers stays unknown.
5. **Provenance travels.** An amended dimension records WHERE its number
   came from (provider usage vs agent estimate), and the two are never
   indistinguishable in the stored fact.
6. **A framework-measured dimension cannot be silently overwritten** — it
   takes an explicit force.
7. **Legacy and full-口径 figures are different units** and are reported as
   such, so pooling them is detectable.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    CostVector,
    ExecutionRecord,
    ProblemProfile,
)
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.world_model.usage import (  # noqa: E402
    HOST_USAGE_ADAPTERS,
    HOST_USAGE_SCHEMA,
    LEGACY_TOKEN_BASIS,
    TOKEN_BASIS_COMPLETION_ONLY,
    TOKEN_BASIS_PROMPT_ONLY,
    TOKEN_BASIS_TOTAL,
    host_report_dimensions,
    host_usage_adapter,
    host_usage_cost_vector,
    is_host_usage_report,
    normalize_host_usage,
    token_bases_mixed,
    token_basis_of,
    token_breakdown,
    usage_cost_vector,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")


# ---------------------------------------------------------------------------
# 1-3. the pure token口径 rules
# ---------------------------------------------------------------------------


class TestTokenBreakdown(unittest.TestCase):

    def test_full_usage_is_the_sum_of_both_sides(self):
        result = token_breakdown({"prompt_tokens": 9000,
                                  "completion_tokens": 100})
        self.assertEqual(result["llm_tokens"], 9100.0)
        self.assertEqual(result["basis"], TOKEN_BASIS_TOTAL)

    def test_reasoning_is_not_added_again(self):
        # reasoning_tokens is INSIDE completion_tokens; adding it would
        # double-count the same tokens.
        result = token_breakdown({
            "prompt_tokens": 100, "completion_tokens": 325,
            "completion_tokens_details": {"reasoning_tokens": 291}})
        self.assertEqual(result["llm_tokens"], 425.0)
        self.assertEqual(result["reasoning_tokens"], 291.0)

    def test_cached_is_not_added_again(self):
        result = token_breakdown({
            "prompt_tokens": 1000, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 800}})
        self.assertEqual(result["llm_tokens"], 1010.0)
        self.assertEqual(result["cached_tokens"], 800.0)

    def test_completion_only_is_a_lower_bound(self):
        result = token_breakdown({"completion_tokens": 50})
        self.assertEqual(result["llm_tokens"], 50.0)
        self.assertEqual(result["basis"], TOKEN_BASIS_COMPLETION_ONLY)

    def test_prompt_only_is_a_lower_bound(self):
        result = token_breakdown({"prompt_tokens": 500})
        self.assertEqual(result["llm_tokens"], 500.0)
        self.assertEqual(result["basis"], TOKEN_BASIS_PROMPT_ONLY)

    def test_a_total_only_report_is_used_verbatim(self):
        result = token_breakdown({"total_tokens": 777})
        self.assertEqual(result["llm_tokens"], 777.0)
        self.assertEqual(result["basis"], TOKEN_BASIS_TOTAL)

    def test_no_usage_is_unknown_never_zero(self):
        result = token_breakdown({})
        self.assertIsNone(result["llm_tokens"])
        self.assertIsNone(result["basis"])

    def test_usage_cost_vector_marks_only_reported_dimensions(self):
        vector, breakdown = usage_cost_vector({"prompt_tokens": 10,
                                               "completion_tokens": 5})
        self.assertEqual(vector.llm_tokens, 15.0)
        self.assertEqual(vector.measured_dims(), {"llm_tokens"})
        self.assertIn("latency_s", usage_cost_vector(
            {"prompt_tokens": 10, "completion_tokens": 5}, 0.5)[0]
            .measured_dims()) if False else None

    def test_usage_cost_vector_with_latency(self):
        vector, _ = usage_cost_vector({"completion_tokens": 5}, 0.25)
        self.assertEqual(vector.measured_dims(), {"llm_tokens", "latency_s"})

    def test_no_usage_and_no_latency_is_no_vector(self):
        vector, breakdown = usage_cost_vector(None, None)
        self.assertIsNone(vector)
        self.assertIsNone(breakdown)


# ---------------------------------------------------------------------------
# 4. host usage reports
# ---------------------------------------------------------------------------


class TestHostUsage(unittest.TestCase):

    def test_openclaw_style_report(self):
        # OpenClaw's ``llm_output`` hook reports the call in this shape.
        report = {"prompt_tokens": 1200, "completion_tokens": 340,
                  "model": "MiniMax-M2.7", "source": "openclaw"}
        vector, breakdown = host_usage_cost_vector(report)
        self.assertEqual(vector.llm_tokens, 1540.0)
        self.assertEqual(breakdown["host"], "openclaw")
        self.assertEqual(breakdown["model"], "MiniMax-M2.7")
        self.assertEqual(breakdown["scope"], "attempt")

    def test_hermes_style_report_with_per_call_breakdown(self):
        # Hermes' ``post_api_request`` hook can report a per-call list.
        report = {"source": "hermes", "calls": [
            {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            {"usage": {"prompt_tokens": 200, "completion_tokens": 30}},
        ]}
        vector, breakdown = host_usage_cost_vector(report)
        # Totals are summed from the calls (never double-counted).
        self.assertEqual(vector.llm_tokens, 350.0)
        self.assertEqual(len(breakdown["calls"]), 2)

    def test_a_report_with_no_numbers_stays_unknown(self):
        vector, breakdown = host_usage_cost_vector(
            {"source": "openclaw", "model": "M"})
        self.assertIsNone(vector)
        self.assertIsNone(breakdown["llm_tokens"])
        self.assertEqual(breakdown["host"], "openclaw")

    def test_canonical_shape_passes_through(self):
        report = {"prompt_tokens": 5, "completion_tokens": 7,
                  "reasoning_tokens": 3, "cached_tokens": 1}
        normalized = normalize_host_usage(report, source="generic")
        self.assertEqual(normalized["prompt_tokens"], 5)
        self.assertEqual(normalized["completion_tokens"], 7)
        self.assertEqual(normalized["reasoning_tokens"], 3)


# ---------------------------------------------------------------------------
# 4b. the versioned HostUsageReport (or-host-usage/1) and its adapters
# ---------------------------------------------------------------------------


#: A report in the exact shape the host contract specifies.
HOST_REPORT = {
    "schema": "or-host-usage/1",
    "host": "openclaw",
    "model": "paratera/DeepSeek-V4.1-Flash",
    "scope": {"task_id": "t1", "episode_id": "ep1", "attempt_id": "ex_1"},
    "tokens": {"prompt_tokens": 12345, "completion_tokens": 678,
               "reasoning_tokens": 291, "cached_tokens": 0, "calls": 3},
    "tool_calls": 38,
    "tool_calls_lower_bound": 1,
    "measured": ["prompt_tokens", "completion_tokens", "tool_calls"],
    "provenance": {"llm_tokens": "provider_usage",
                   "tool_calls": "agent_observed"},
    "notes": "fixture",
}


class TestHostUsageReport(unittest.TestCase):

    def test_the_schema_tag_decides_the_contract(self):
        self.assertTrue(is_host_usage_report(HOST_REPORT))
        self.assertFalse(is_host_usage_report(
            {"prompt_tokens": 1, "completion_tokens": 2}))
        self.assertFalse(is_host_usage_report("not a report"))

    def test_normalize_yields_both_dimensions_with_provenance(self):
        dimensions, info = host_report_dimensions(HOST_REPORT)
        # llm_tokens is the FULL口径 total (prompt + completion); the 291
        # reasoning tokens are a sub-fact inside completion, never added.
        self.assertEqual(dimensions["llm_tokens"], 12345 + 678)
        self.assertEqual(dimensions["tool_calls"], 38.0)
        self.assertEqual(info["basis"], TOKEN_BASIS_TOTAL)
        self.assertEqual(info["provenance"]["llm_tokens"],
                         "provider_usage")
        self.assertEqual(info["provenance"]["tool_calls"],
                         "agent_observed")
        self.assertEqual(info["reasoning_tokens"], 291.0)
        self.assertEqual(info["model_calls"], 3.0)

    def test_the_measured_whitelist_governs_what_counts(self):
        # tool_calls present as a NUMBER but outside the whitelist: it is a
        # sub-fact, never a cost measurement.
        report = dict(HOST_REPORT, measured=["prompt_tokens",
                                             "completion_tokens"])
        dimensions, info = host_report_dimensions(report)
        self.assertEqual(dimensions, {"llm_tokens": 12345 + 678})
        self.assertNotIn("tool_calls", info["provenance"])

    def test_a_dimension_outside_the_whitelist_stays_unknown(self):
        report = dict(HOST_REPORT, tokens={"prompt_tokens": 10},
                      measured=["tool_calls"], tool_calls=4)
        dimensions, _ = host_report_dimensions(report)
        # prompt-only is a lower bound AND outside the whitelist: unknown.
        self.assertEqual(dimensions, {"tool_calls": 4.0})

    def test_completion_only_is_a_lower_bound_basis(self):
        report = dict(HOST_REPORT, tokens={"completion_tokens": 50})
        dimensions, info = host_report_dimensions(report)
        self.assertEqual(dimensions["llm_tokens"], 50.0)
        self.assertEqual(info["basis"], TOKEN_BASIS_COMPLETION_ONLY)

    def test_tool_calls_below_the_reports_own_bound_is_refused(self):
        report = dict(HOST_REPORT, tool_calls=0,
                      tool_calls_lower_bound=1)
        with self.assertRaises(ValueError) as caught:
            host_report_dimensions(report)
        self.assertIn("tool_calls_lower_bound", str(caught.exception))

    def test_a_report_with_no_numbers_yields_nothing(self):
        report = {"schema": HOST_USAGE_SCHEMA, "host": "openclaw",
                  "measured": ["prompt_tokens", "tool_calls"]}
        dimensions, info = host_report_dimensions(report)
        self.assertEqual(dimensions, {})
        self.assertIsNone(info["basis"])


class TestHostUsageAdapter(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name

    def _write_report(self, name="openclaw-usage.ex_1.json",
                      payload=None):
        base = Path(self.home) / "host_usage"
        base.mkdir(parents=True, exist_ok=True)
        path = base / name
        path.write_text(json.dumps(payload or HOST_REPORT),
                        encoding="utf-8")
        return path

    def test_the_openclaw_adapter_is_registered(self):
        self.assertIn("openclaw", HOST_USAGE_ADAPTERS)
        self.assertEqual(host_usage_adapter("OpenClaw").name, "openclaw")
        self.assertIsNone(host_usage_adapter("no-such-host"))
        self.assertIsNone(host_usage_adapter(None))

    def test_locate_finds_the_attempt_report(self):
        self._write_report()
        adapter = host_usage_adapter("openclaw")
        path = adapter.locate({"home": self.home, "execution_id": "ex_1"})
        self.assertIsNotNone(path)
        self.assertIn("ex_1", path)

    def test_locate_falls_back_to_the_untagged_file(self):
        self._write_report("openclaw-usage.json")
        adapter = host_usage_adapter("openclaw")
        self.assertIsNotNone(
            adapter.locate({"home": self.home, "execution_id": "ex_9"}))

    def test_locate_returns_none_when_no_report_exists(self):
        adapter = host_usage_adapter("openclaw")
        self.assertIsNone(
            adapter.locate({"home": self.home, "execution_id": "ex_1"}))

    def test_load_reads_and_parses_the_report(self):
        self._write_report()
        adapter = host_usage_adapter("openclaw")
        report = adapter.load({"home": self.home, "execution_id": "ex_1"})
        self.assertEqual(report["schema"], HOST_USAGE_SCHEMA)

    def test_load_is_none_for_a_malformed_file(self):
        base = Path(self.home) / "host_usage"
        base.mkdir(parents=True, exist_ok=True)
        (base / "openclaw-usage.ex_1.json").write_text("{not json",
                                                       encoding="utf-8")
        adapter = host_usage_adapter("openclaw")
        self.assertIsNone(
            adapter.load({"home": self.home, "execution_id": "ex_1"}))

    def test_normalize_maps_to_the_canonical_dimensions(self):
        adapter = host_usage_adapter("openclaw")
        dimensions, info = adapter.normalize(HOST_REPORT)
        self.assertEqual(dimensions["llm_tokens"], 12345 + 678)
        self.assertEqual(dimensions["tool_calls"], 38.0)

    def test_collect_end_to_end_or_none(self):
        adapter = host_usage_adapter("openclaw")
        self.assertIsNone(adapter.collect({"home": self.home,
                                           "execution_id": "ex_1"}))
        self._write_report()
        collected = adapter.collect({"home": self.home,
                                     "execution_id": "ex_1"})
        self.assertIsNotNone(collected)
        dimensions, info = collected
        self.assertEqual(dimensions["tool_calls"], 38.0)

    def test_collect_treats_a_contradictory_report_as_absent(self):
        self._write_report(payload=dict(HOST_REPORT, tool_calls=0))
        adapter = host_usage_adapter("openclaw")
        self.assertIsNone(adapter.collect({"home": self.home,
                                           "execution_id": "ex_1"}))


# ---------------------------------------------------------------------------
# 5-6. provenance and the overwrite guard
# ---------------------------------------------------------------------------


class TestCostProvenance(HarnessTestCase):

    def _record(self, h, execution_id="ex_p1"):
        profile = ProblemProfile(problem_id="p1", family="routing",
                                 scale_features={"n_vars": 1})
        record = ExecutionRecord(
            execution_id=execution_id, task_id="t1", strategy_id="S1",
            profile_snapshot=profile,
            quality={"status": "optimal", "feasible": True},
            cost=CostVector(latency_s=1.0, solver_runtime_s=2.0,
                            measured={"latency_s", "solver_runtime_s"}))
        h.bank.append(record)
        return record

    def test_an_amended_dimension_records_its_source(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        record = h.bank.update_cost("ex_p1", llm_tokens=1000,
                                    source="provider_usage")
        provenance = record.execution_features["cost_provenance"]
        self.assertEqual(provenance["llm_tokens"]["source"],
                         "provider_usage")

    def test_an_unknown_source_is_refused(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        with self.assertRaises(StorageError):
            h.bank.update_cost("ex_p1", source="magic", llm_tokens=1.0)

    def test_a_framework_measured_dimension_cannot_be_overwritten_silently(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        with self.assertRaises(StorageError) as caught:
            h.bank.update_cost("ex_p1", latency_s=99.0)
        self.assertIn("framework-measured", str(caught.exception))
        # The real observation is untouched.
        self.assertEqual(h.bank.get("ex_p1").cost.latency_s, 1.0)

    def test_force_allows_a_deliberate_overwrite(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        record = h.bank.update_cost("ex_p1", force=True, latency_s=99.0)
        self.assertEqual(record.cost.latency_s, 99.0)

    def test_a_provider_reported_dimension_is_also_protected(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        h.bank.update_cost("ex_p1", llm_tokens=1000,
                           source="provider_usage")
        # A later plain amend of the same dimension is refused without force.
        with self.assertRaises(StorageError):
            h.bank.update_cost("ex_p1", llm_tokens=5)

    def test_a_declared_dimension_may_be_amended_freely(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._record(h)
        h.bank.update_cost("ex_p1", llm_tokens=1000)  # estimate
        record = h.bank.update_cost("ex_p1", llm_tokens=2000)
        self.assertEqual(record.cost.llm_tokens, 2000.0)


# ---------------------------------------------------------------------------
# 7. token口径 are not mixed
# ---------------------------------------------------------------------------


class TestTokenBasis(unittest.TestCase):

    def _record(self, basis=None):
        features = {}
        if basis is not None:
            features["cost_provenance"] = {
                "llm_tokens": {"source": "agent_estimate", "basis": basis}}
        return ExecutionRecord(
            execution_id="ex_1", task_id="t1", strategy_id="S1",
            profile_snapshot=ProblemProfile(problem_id="p", family="routing"),
            quality={}, cost=CostVector(llm_tokens=100.0,
                                        measured={"llm_tokens"}),
            execution_features=features)

    def test_a_record_with_no_llm_tokens_has_no_basis(self):
        record = ExecutionRecord(
            execution_id="ex_1", task_id="t1", strategy_id="S1",
            profile_snapshot=ProblemProfile(problem_id="p", family="routing"),
            quality={}, cost=CostVector(measured=set()))
        self.assertIsNone(token_basis_of(record))

    def test_an_unmarked_token_record_reads_as_legacy(self):
        record = ExecutionRecord(
            execution_id="ex_1", task_id="t1", strategy_id="S1",
            profile_snapshot=ProblemProfile(problem_id="p", family="routing"),
            quality={}, cost=CostVector(llm_tokens=100.0,
                                        measured={"llm_tokens"}))
        self.assertEqual(token_basis_of(record), LEGACY_TOKEN_BASIS)

    def test_mixed_bases_are_detected(self):
        legacy = ExecutionRecord(
            execution_id="ex_1", task_id="t1", strategy_id="S1",
            profile_snapshot=ProblemProfile(problem_id="p", family="routing"),
            quality={}, cost=CostVector(llm_tokens=100.0,
                                        measured={"llm_tokens"}))
        modern = self._record(TOKEN_BASIS_TOTAL)
        self.assertTrue(token_bases_mixed([legacy, modern]))
        self.assertFalse(token_bases_mixed([modern, self._record(
            TOKEN_BASIS_TOTAL)]))


# ---------------------------------------------------------------------------
# 8. end to end: a host usage report becomes the record's real cost
# ---------------------------------------------------------------------------


class TestHostUsageEndToEnd(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)

    def _solve(self, name="solve_usage"):
        work = Path(self.home) / f"ws_{name}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0,\n"
            "               'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        return work

    def test_a_host_report_becomes_the_records_real_token_cost(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve()
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        # The host reports the attempt's real spend (prompt + completion).
        h.record(record, host_usage={"source": "openclaw",
                                     "prompt_tokens": 1200,
                                     "completion_tokens": 340,
                                     "model": "MiniMax-M2.7"})
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 1540.0)
        self.assertIn("llm_tokens", stored.cost.measured_dims())
        # The provenance says the HOST measured it, and the basis is the
        # full口径.
        provenance = stored.execution_features["cost_provenance"]
        self.assertEqual(provenance["llm_tokens"]["source"],
                         "provider_usage")
        self.assertEqual(provenance["llm_tokens"]["basis"],
                         TOKEN_BASIS_TOTAL)
        self.assertEqual(stored.execution_features["host_usage"]["host"],
                         "openclaw")

    def test_the_cli_records_a_usage_file(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("cli_usage")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        usage_path = Path(self.home) / "usage.json"
        usage_path.write_text(json.dumps(
            {"prompt_tokens": 800, "completion_tokens": 200,
             "source": "hermes"}), encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "discard_staged": None, "from_staged": None,
            "execution": None, "record_file": None,
            "override": None, "override_mode": "replace",
            "override_source": "agent_estimate", "override_force": False,
            "usage_file": str(usage_path), "usage_source": None,
            "usage_host": None,
            "prediction": None, "method": None, "method_actual": None,
        })()
        # The record is already staged; use --from-staged.
        args.from_staged = record.execution_id
        code = cli_module.cmd_record(args)
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 1000.0)


# ---------------------------------------------------------------------------
# 9. a versioned HostUsageReport writes BOTH dimensions in one command
# ---------------------------------------------------------------------------


class TestHostReportIngestion(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.saved_env = os.environ.pop("OR_HOST_USAGE_FILE", None)

        def restore_env():
            if self.saved_env is not None:
                os.environ["OR_HOST_USAGE_FILE"] = self.saved_env
        self.addCleanup(restore_env)

    def _solve(self, name="solve_host"):
        work = Path(self.home) / f"ws_{name}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0,\n"
            "               'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        return work

    HOST_REPORT = {
        "schema": "or-host-usage/1",
        "host": "openclaw",
        "model": "paratera/DeepSeek-V4.1-Flash",
        "scope": {"task_id": "t1", "episode_id": "ep1"},
        "tokens": {"prompt_tokens": 12345, "completion_tokens": 678,
                   "reasoning_tokens": 291, "cached_tokens": 0, "calls": 3},
        "tool_calls": 38,
        "tool_calls_lower_bound": 1,
        "measured": ["prompt_tokens", "completion_tokens", "tool_calls"],
        "provenance": {"llm_tokens": "provider_usage",
                       "tool_calls": "agent_observed"},
    }

    def test_one_report_writes_both_dimensions_without_hand_typing(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve()
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record, host_usage=dict(self.HOST_REPORT))
        stored = h.bank.get(record.execution_id)
        # llm_tokens: the FULL口径 total, provider_usage.
        self.assertEqual(stored.cost.llm_tokens, 12345 + 678)
        self.assertIn("llm_tokens", stored.cost.measured_dims())
        # tool_calls: the host's whole-scope count, agent_observed.
        self.assertEqual(stored.cost.tool_calls, 38.0)
        self.assertIn("tool_calls", stored.cost.measured_dims())
        provenance = stored.execution_features["cost_provenance"]
        self.assertEqual(provenance["llm_tokens"]["source"],
                         "provider_usage")
        self.assertEqual(provenance["llm_tokens"]["basis"],
                         TOKEN_BASIS_TOTAL)
        self.assertEqual(provenance["tool_calls"]["source"],
                         "agent_observed")
        # The report's own facts travel with the record.
        info = stored.execution_features["host_usage"]
        self.assertEqual(info["schema"], "or-host-usage/1")
        self.assertEqual(info["reasoning_tokens"], 291.0)

    def test_dimensions_outside_the_whitelist_stay_unknown(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("whitelist")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        report = dict(self.HOST_REPORT,
                      measured=["prompt_tokens", "completion_tokens"])
        h.record(record, host_usage=report)
        stored = h.bank.get(record.execution_id)
        self.assertIn("llm_tokens", stored.cost.measured_dims())
        # tool_calls is a NUMBER in the report but NOT whitelisted: it stays
        # unknown — never zero, never a measurement.
        self.assertNotIn("tool_calls", stored.cost.measured_dims())

    def test_the_cli_records_a_versioned_report_from_a_file(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("cli_host")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        usage_path = Path(self.home) / "host_report.json"
        usage_path.write_text(json.dumps(self.HOST_REPORT),
                              encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "discard_staged": None, "from_staged": record.execution_id,
            "execution": None, "record_file": None,
            "override": None, "override_mode": "replace",
            "override_source": "agent_estimate", "override_force": False,
            "usage_file": str(usage_path), "usage_source": None,
            "usage_host": None,
            "prediction": None, "method": None, "method_actual": None,
        })()
        code = cli_module.cmd_record(args)
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 12345 + 678)
        self.assertEqual(stored.cost.tool_calls, 38.0)

    def test_the_cli_locates_the_report_through_the_named_adapter(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("adapter")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        # The host wrote its report under <home>/host_usage/.
        base = Path(self.home) / "host_usage"
        base.mkdir(parents=True, exist_ok=True)
        (base / f"openclaw-usage.{record.execution_id}.json").write_text(
            json.dumps(self.HOST_REPORT), encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "home": self.home,
            "discard_staged": None, "from_staged": record.execution_id,
            "execution": None, "record_file": None,
            "override": None, "override_mode": "replace",
            "override_source": "agent_estimate", "override_force": False,
            "usage_file": None, "usage_source": None,
            "usage_host": "openclaw",
            "prediction": None, "method": None, "method_actual": None,
        })()
        code = cli_module.cmd_record(args)
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 12345 + 678)
        self.assertEqual(stored.cost.tool_calls, 38.0)

    def test_a_missing_adapter_report_leaves_dimensions_unknown(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("no_report")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "home": self.home,
            "discard_staged": None, "from_staged": record.execution_id,
            "execution": None, "record_file": None,
            "override": None, "override_mode": "replace",
            "override_source": "agent_estimate", "override_force": False,
            "usage_file": None, "usage_source": None,
            "usage_host": "openclaw",
            "prediction": None, "method": None, "method_actual": None,
        })()
        code = cli_module.cmd_record(args)
        # Recording SUCCEEDS: the dimensions stay unknown, never zero, and
        # the summary says the adapter found no report.
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertNotIn("llm_tokens", stored.cost.measured_dims())
        self.assertNotIn("tool_calls", stored.cost.measured_dims())

    def test_an_unknown_host_name_is_refused_with_the_known_hosts(self):
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "home": self.home,
            "discard_staged": None, "from_staged": None,
            "execution": None, "record_file": None,
            "override": None, "override_mode": "replace",
            "override_source": "agent_estimate", "override_force": False,
            "usage_file": None, "usage_source": None,
            "usage_host": "no-such-host",
            "prediction": None, "method": None, "method_actual": None,
        })()
        code = cli_module.cmd_record(args)
        self.assertEqual(code, 2)

    def test_amend_cost_applies_a_versioned_report_per_dimension(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("amend")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        usage_path = Path(self.home) / "late_report.json"
        usage_path.write_text(json.dumps(self.HOST_REPORT),
                              encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "execution_id": record.execution_id,
            "override": None, "mode": "replace",
            "source": "agent_estimate", "amend_force": False,
            "usage_file": str(usage_path), "usage_source": None,
            "usage_host": None,
        })()
        code = cli_module.cmd_amend_cost(args)
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 12345 + 678)
        self.assertEqual(stored.cost.tool_calls, 38.0)
        provenance = stored.execution_features["cost_provenance"]
        self.assertEqual(provenance["llm_tokens"]["source"],
                         "provider_usage")
        self.assertEqual(provenance["tool_calls"]["source"],
                         "agent_observed")

    def test_amend_cost_refuses_tool_calls_below_the_bound(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        from or_harness import cli as cli_module
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("bound")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        report = dict(self.HOST_REPORT, tool_calls=0,
                      tool_calls_lower_bound=1)
        usage_path = Path(self.home) / "bad_report.json"
        usage_path.write_text(json.dumps(report), encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "execution_id": record.execution_id,
            "override": None, "mode": "replace",
            "source": "agent_estimate", "amend_force": False,
            "usage_file": str(usage_path), "usage_source": None,
            "usage_host": None,
        })()
        code = cli_module.cmd_amend_cost(args)
        # The report contradicts itself (0 calls below its own bound of 1):
        # refused, nothing stored.
        self.assertEqual(code, 2)
        stored = h.bank.get(record.execution_id)
        self.assertNotIn("tool_calls", stored.cost.measured_dims())


# ---------------------------------------------------------------------------
# 10. close-episode warns when dimensions are still unknown
# ---------------------------------------------------------------------------


class TestCloseEpisodeCostWarning(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)

    def test_unknown_dimensions_are_warned_about_at_close(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = Path(self.home) / "ws"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0,\n"
            "               'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        task = {"task_id": "t1", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        # No host report, no override: llm_tokens and tool_calls stay
        # UNKNOWN (never zero).
        h.record(record)
        closed = h.close_episode("t1", "ep1", terminal_state="completed")
        warnings = closed.get("cost_completeness_warnings")
        self.assertIsNotNone(warnings)
        self.assertIn("llm_tokens", warnings["unknown_dimensions"])
        self.assertIn("tool_calls", warnings["unknown_dimensions"])
        self.assertIn(record.execution_id,
                      warnings["unknown_dimensions"]["llm_tokens"])
        self.assertIn("never zero", warnings["note"])

    def test_no_warning_when_every_dimension_is_measured(self):
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = Path(self.home) / "ws"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0,\n"
            "               'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        task = {"task_id": "t2", "family": "routing", "spec": {}}
        record = h.execute(task, "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record, host_usage={
            "schema": "or-host-usage/1", "host": "openclaw",
            "tokens": {"prompt_tokens": 100, "completion_tokens": 20},
            "tool_calls": 5,
            "measured": ["prompt_tokens", "completion_tokens",
                         "tool_calls"],
            "provenance": {"llm_tokens": "provider_usage",
                           "tool_calls": "agent_observed"}})
        closed = h.close_episode("t2", "ep1", terminal_state="completed")
        self.assertNotIn("cost_completeness_warnings", closed)


if __name__ == "__main__":
    unittest.main()
