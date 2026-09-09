"""ORHarness facade: the Python entry point for the outer harness agent.

The harness is the orchestrator. This layer advises and executes; the harness
may refuse recommendations, request alternatives, execute without recording,
override recorded costs, and decides when to induce and when to collect
garbage. No conversation loop, no runtime LLM calls, no hidden global state —
the memory location is always explicit (``home`` / ``OR_HARNESS_HOME``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from or_harness.adapters.solver import available_families, probe_all
from or_harness.core.schema import ExecutionRecord, ProblemProfile, profile_matches
from or_harness.core.storage import Store, resolve_home
from or_harness.execution.executor import SafePythonExecutor
from or_harness.profiling.profiler import derivation_report, profile_task
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import GarbageCollector
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.strategy.triggers import check_triggers, solver_advisories

#: Prediction hit tolerance: an observation counts as a miss when it falls
#: outside the entry's interval by more than this fraction of the interval
#: width (relative slack keeps wide honest intervals meaningful).
PREDICTION_HIT_SLACK = 0.15


class ORHarness:
    def __init__(self, home: Optional[str] = None, *,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 cost_weights: Optional[Dict[str, float]] = None,
                 catalog_path: Optional[str] = None,
                 executor: Optional[SafePythonExecutor] = None):
        self.home = resolve_home(home)
        self.store = Store(self.home)
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog(catalog_path)
        self.selector = Selector(self.catalog, self.sbank, self.stats,
                                 alpha=alpha, beta=beta, gamma=gamma,
                                 cost_weights=cost_weights)
        self.executor = executor or SafePythonExecutor()
        self.induction = InductionEngine(self.stats, self.sbank)
        self.gc = GarbageCollector(self.bank, self.sbank, self.stats)

    # -- capabilities ----------------------------------------------------------

    def profile(self, task: Dict[str, Any], code: Optional[str] = None) -> ProblemProfile:
        return profile_task(task, code)

    def derivation_report(self, task: Dict[str, Any],
                          code: Optional[str] = None) -> Dict[str, Any]:
        """Per-dimension coupling derivation report: value, origin
        (model/code/spec/supplied/null), notes, model verification issues,
        and cross-check warnings."""
        return derivation_report(self.profile(task, code))

    def recommend(self, task: Dict[str, Any], *, top: int = 3,
                  exclude: Optional[Sequence[str]] = None,
                  memory_mode: str = "cost-aware",
                  code: Optional[str] = None) -> Dict[str, Any]:
        profile = self.profile(task, code)
        recs = self.selector.recommend(profile, top=top, exclude=exclude,
                                       memory_mode=memory_mode)
        solvers = available_families()
        result = {
            "profile": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
            "available_solver_families": solvers,
            "solver_advisories": solver_advisories(self.bank),
        }
        profiling = profile.annotations.get("profiling") or {}
        if profiling.get("coupling_warnings"):
            result["coupling_warnings"] = profiling["coupling_warnings"]
        return result

    def execute(self, task: Dict[str, Any], strategy_id: str, code_path: str,
                workspace: str, *, solver: str,
                verification_level: str = "basic") -> ExecutionRecord:
        if strategy_id not in self.catalog:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        code_text = Path(code_path).read_text(encoding="utf-8")
        profile = self.profile(task, code_text)
        record = self.executor.execute(
            Path(code_path), Path(workspace), solver=solver,
            task_id=str(task["task_id"]), strategy_id=strategy_id,
            profile=profile, verification_level=verification_level)
        # Safety net: stage every execution — successes AND failures — so a
        # failed attempt is never silently lost when the harness immediately
        # retries. Staging is not recording; recording stays the harness's
        # explicit decision (`orx record`).
        self.bank.stage_pending(record)
        return record

    def record(self, record: ExecutionRecord,
               override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Append a fact, then run the automatic chain:
        cost backfill -> prediction checks -> dormancy wakeup -> C1-C6 hints.

        Also reports staged-but-unrecorded executions for the same task, so
        the harness notices a dropped failure (e.g. an abandoned first
        attempt) before it is forgotten."""
        # Persist failure classification once — a first-class fact, not a
        # re-derived view (environment vs model errors feed solver advisories
        # and future failure-pattern induction).
        from or_harness.strategy.triggers import classify_failure
        for failure in record.failures:
            if failure.error_class is None:
                failure.error_class = classify_failure(record)
        self.bank.append(record)
        if override:
            self.bank.update_cost(record.execution_id, **override)
            record = self.bank.get(record.execution_id)
        self.bank.clear_pending(record.execution_id)

        prediction_events = self._check_predictions(record)
        expected_map = {e.strategy_id: {"quality": e.expected_quality_hat}
                        for e in self.sbank.matching(record.profile_snapshot)}
        prior_failures = [r for r in self.bank.query(task_id=record.task_id)
                          if not r.quality.get("feasible", False)
                          and r.execution_id != record.execution_id]
        prior_failures += [p for p in self.bank.pending(task_id=record.task_id)
                           if p.execution_id != record.execution_id
                           and not p.quality.get("feasible", False)]
        hints = check_triggers(record, self.stats, self.catalog, expected_map,
                               prior_failures=prior_failures)
        unrecorded = [p.execution_id for p in
                      self.bank.pending(task_id=record.task_id)]
        result = {
            "execution_id": record.execution_id,
            "recorded": True,
            "prediction_checks": prediction_events,
            "induction_hints": [h.to_dict() for h in hints],
        }
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        return result

    def induce(self, *, strategy_id: Optional[str] = None, all_: bool = False,
               rebuild: bool = False, widen: Optional[str] = None,
               tighten: Optional[str] = None,
               dry_run: bool = False, force: bool = False,
               llm_conditions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if rebuild:
            return self.induction.rebuild(dry_run=dry_run)
        if widen:
            return self.induction.widen(widen)
        if tighten:
            return self.induction.tighten(tighten)
        targets = self._induction_targets(strategy_id, all_)
        results = []
        for profile, sid in targets:
            results.append(self.induction.induce(
                profile, sid, scope="L1", dry_run=dry_run, force=force,
                llm_conditions=llm_conditions))
        return {"results": results}

    def inspect(self, *, bank: str = "experience",
                task_id: Optional[str] = None,
                strategy_id: Optional[str] = None,
                status: Optional[str] = None) -> Dict[str, Any]:
        if bank == "experience":
            records = self.bank.query(task_id=task_id, strategy_id=strategy_id)
            return {"bank": "experience", "count": len(records),
                    "records": [r.to_dict() for r in records]}
        if bank == "strategic":
            entries = self.sbank.list(status=status, strategy_id=strategy_id)
            return {"bank": "strategic", "count": len(entries),
                    "entries": [e.to_dict() for e in entries]}
        if bank == "archive":
            cards = self.sbank.cold_archive()
            return {"bank": "cold_archive", "count": len(cards),
                    "cards": [c.to_dict() for c in cards]}
        raise ValueError("bank must be experience|strategic|archive")

    def collect_garbage(self, mode: str = "compact",
                        dry_run: bool = False) -> Dict[str, Any]:
        return self.gc.run(mode=mode, dry_run=dry_run)

    def retire(self, entry_id: str, reason: str) -> Dict[str, Any]:
        card = self.sbank.retire(entry_id, reason=reason)
        return {"retired": entry_id, "cold_archive_card": card.to_dict()}

    def doctor(self) -> Dict[str, Any]:
        reports = probe_all()
        pending = self.bank.pending()
        return {
            "home": str(self.home),
            "solvers": [r.to_dict() for r in reports],
            "available_families": available_families(),
            "memory": {"executions": self.bank.count(),
                       "entries": self.sbank.count(),
                       "cold_archive": len(self.sbank.cold_archive()),
                       "pending_staged": len(pending)},
            "pending_staged_executions": [
                {"execution_id": p.execution_id, "task_id": p.task_id,
                 "strategy_id": p.strategy_id,
                 "status": p.quality.get("status")} for p in pending],
        }

    def close(self) -> None:
        self.store.close()

    # -- automatic chain internals ------------------------------------------------

    def _check_predictions(self, record: ExecutionRecord) -> List[Dict[str, Any]]:
        """Forward validation: every matching entry's prediction vs this
        observation. Cross-family misses on wide entries tighten scope instead
        of demoting (the content may be right; the range was wrong).

        Cost predictions carry a parallel, WARNING-ONLY check: actual cost
        within the entry's multiplicative interval counts as a hit; misses
        feed cost calibration but never demote — quality and cost errors have
        different remedies (retire vs. re-weigh) and never share a verdict."""
        events: List[Dict[str, Any]] = []
        observed = quality_score(record)
        for entry in self.sbank.matching(record.profile_snapshot):
            lo, hi = entry.quality_interval
            width = max(hi - lo, 1e-6)
            slack = PREDICTION_HIT_SLACK * width
            hit = (lo - slack) <= observed <= (hi + slack)
            calibration_err = abs(observed - entry.expected_quality_hat)
            updated, transitions = self.sbank.record_prediction(
                entry.entry_id, hit, calibration_err)
            event: Dict[str, Any] = {
                "entry_id": entry.entry_id, "hit": hit,
                "observed_quality": round(observed, 4),
                "interval": [lo, hi], "transitions": transitions,
            }
            if not hit and entry.scope_level in ("L2", "L3"):
                outcome = self.induction.tighten(entry.entry_id)
                event["scope_tightened"] = outcome
            # Cost-side check (parallel track, warning-only).
            cost_event = self._check_cost_prediction(updated, record)
            if cost_event is not None:
                event["cost_check"] = cost_event
            events.append(event)
        return events

    def _check_cost_prediction(self, entry, record: ExecutionRecord
                               ) -> Optional[Dict[str, Any]]:
        """Check the entry's cost interval against the observed CostVector.

        Runs after the llm_tokens backfill (record applies overrides before
        calling this chain), so all five dimensions are present. Returns None
        when the entry carries no cost interval (pre-upgrade entries)."""
        if not entry.cost_interval:
            return None
        import math

        from or_harness.core.schema import COST_DIMENSIONS
        hits, log_errors = [], {}
        for dim in COST_DIMENSIONS:
            band = entry.cost_interval.get(dim)
            if band is None:
                continue
            lo_mult, hi_mult = band
            predicted = getattr(entry.expected_cost_hat, dim)
            actual = getattr(record.cost, dim)
            if predicted <= 0 and actual <= 0:
                hits.append(True)  # both zero: trivially consistent
                continue
            predicted = max(predicted, 1e-9)
            in_band = lo_mult * predicted <= actual <= hi_mult * predicted
            hits.append(in_band)
            log_errors[dim] = round(abs(math.log(max(actual, 1e-9) / predicted)), 4)
        if not hits:
            return None
        cost_hit = all(hits)
        entry.prediction_track.record_cost(cost_hit, log_errors)
        self.sbank.update(entry)
        return {
            "hit": cost_hit,
            "per_dimension_in_band": hits,
            "log_errors": log_errors,
            "cost_hit_rate": round(entry.prediction_track.cost_hit_rate, 4),
        }

    def _induction_targets(self, strategy_id: Optional[str], all_: bool):
        targets = []
        seen = set()
        for rec in self.bank.all():
            if rec.source != "executed":
                continue
            if strategy_id and rec.strategy_id != strategy_id:
                continue
            key = (rec.group_l1, rec.strategy_id)
            if key in seen:
                continue
            if not all_ and strategy_id is None:
                continue
            seen.add(key)
            targets.append((rec.profile_snapshot, rec.strategy_id))
        return targets
