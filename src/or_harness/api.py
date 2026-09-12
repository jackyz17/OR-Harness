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
from or_harness.core.coupling import understand as cir_understand
from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    PredictionSnapshot,
    ProblemProfile,
    compute_cost_feedback,
    profile_matches,
)
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

    def understand(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-model coupling-aware understanding.

        Validates the task's optional ``coupling`` field (a CIR), infers
        structural relations deterministically, derives coupling groups, and
        renders modeling guidance — all *before* the canonical model is
        written.  When no CIR is supplied, returns a prompt to submit one.
        """
        return cir_understand(task)

    def profile(self, task: Dict[str, Any], code: Optional[str] = None,
                cir: Optional[Any] = None) -> ProblemProfile:
        return profile_task(task, code, cir=cir)

    def derivation_report(self, task: Dict[str, Any],
                          code: Optional[str] = None,
                          cir: Optional[Any] = None) -> Dict[str, Any]:
        """Per-dimension coupling derivation report: value, origin
        (model/code/spec/supplied/null), notes, model verification issues,
        and cross-check warnings (including CIR ↔ model when a CIR is
        provided)."""
        return derivation_report(self.profile(task, code, cir=cir))

    def recall(self, task: Dict[str, Any], *, top: int = 3,
               exclude: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
               code: Optional[str] = None) -> Dict[str, Any]:
        profile = self.profile(task, code)
        recs = self.selector.recall(profile, top=top, exclude=exclude,
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

    def predict_cost(self, task: Dict[str, Any], strategy_id: str,
                     code: Optional[str] = None) -> PredictionSnapshot:
        """Pre-execution cost expectation for (problem conditions, strategy).

        Estimation scope: one execution ATTEMPT under the conditions visible
        BEFORE execution (task JSON / frozen profile — never post-modeling
        artifacts). Provenance ladder:
        1. a matching StrategicEntry's expected cost (entry);
        2. comparable conditional statistics over attempt-scope Evidence
           Bank records (stats — a recount, with per-dimension support and
           a scale-coverage check);
        3. unknown (no usable evidence — never a default zero presented as
           cheap, and never a task-scope total re-labelled as an attempt
           prediction).
        """
        if strategy_id not in self.catalog:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        profile = self.profile(task, code)
        entry = next((e for e in self.sbank.matching(profile)
                      if e.strategy_id == strategy_id), None)
        if entry is not None:
            return PredictionSnapshot(
                strategy_id=strategy_id,
                expected_cost=entry.expected_cost_hat,
                source="entry", measurement_scope="attempt",
                support_n=entry.support_n,
                support_per_dim=dict(entry.cost_support_n),
                evidence_refs=[entry.entry_id],
                note=f"strategic entry {entry.entry_id} ({entry.status})")
        cell = self.stats.for_profile(profile, "L1").get(strategy_id)
        if cell is not None and cell.n > 0:
            mismatch = self._scale_mismatch(profile, cell)
            if mismatch:
                return PredictionSnapshot(
                    strategy_id=strategy_id, expected_cost=None,
                    source="unknown", measurement_scope="attempt",
                    support_n=cell.n,
                    evidence_refs=list(cell.execution_ids),
                    note=("insufficient evidence: historical samples are not "
                          "comparable at this scale — " + mismatch))
            return PredictionSnapshot(
                strategy_id=strategy_id,
                expected_cost=cell.mean_cost,
                source="stats", measurement_scope="attempt",
                support_n=cell.n,
                support_per_dim=dict(cell.n_measured),
                evidence_refs=list(cell.execution_ids),
                note=f"conditional statistics over n={cell.n} attempt-scope "
                     "executions in this structural group")
        return PredictionSnapshot(strategy_id=strategy_id,
                                  expected_cost=None, source="unknown",
                                  measurement_scope="attempt",
                                  note="no attempt-scope cost evidence for "
                                       "this strategy×profile pair")

    @staticmethod
    def _scale_mismatch(profile, cell) -> Optional[str]:
        """Lightweight scale-comparability check: a target scale feature
        outside the historical sample coverage is flagged (no preset
        thresholds — sample coverage is the only yardstick)."""
        notes = []
        for feat, (lo, hi) in sorted(cell.scale_ranges.items()):
            value = profile.scale_features.get(feat)
            if value is None:
                continue
            if value < lo or value > hi:
                notes.append(f"{feat}={value} outside sample coverage "
                             f"[{lo}, {hi}]")
        return "; ".join(notes) if notes else None

    def execute(self, task: Dict[str, Any], strategy_id: str, code_path: str,
                workspace: str, *, solver: str,
                verification_level: str = "basic") -> ExecutionRecord:
        """Run one episode and assemble its Execution Evidence record.

        The returned record is an evidence unit: the strategy ACTUALLY used,
        the quality/cost ACTUALLY observed, the failures actually seen, and
        the implementation artifacts (solver output, diagnostics). When the
        task carries a CIR (``coupling`` field), a snapshot is preserved on
        the record so offline induction can re-bin this episode by structural
        context. The record makes no generalization claim.
        """
        if strategy_id not in self.catalog:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        # Frozen pre-strategy signature: profile is derived from the task's
        # coupling (CIR) / spec / annotations / model fields ONLY — never
        # from solve.py.  The generated solve script is a post-strategy
        # artifact; letting it redefine the problem's identity would create
        # a self-reinforcing loop (strategy → code → profile → grouping →
        # future strategy choice).
        profile = self.profile(task)
        record = self.executor.execute(
            Path(code_path), Path(workspace), solver=solver,
            task_id=str(task["task_id"]), strategy_id=strategy_id,
            profile=profile, verification_level=verification_level)
        # Evidence completeness: preserve the coupling-aware representation
        # snapshot (CIR) that was actually solved. Snapshot only — CIR
        # extraction and coupling understanding are untouched.
        if task.get("coupling"):
            record.cir_snapshot = dict(task["coupling"])
        # Safety net: stage every execution — successes AND failures — so a
        # failed attempt is never silently lost when the harness immediately
        # retries. Staging is not recording; recording stays the harness's
        # explicit decision (`orx record`).
        self.bank.stage_pending(record)
        return record

    def record(self, record: ExecutionRecord,
               override: Optional[Dict[str, float]] = None,
               retain_reason: Optional[str] = None, *,
               override_mode: str = "replace",
               prediction: Optional[PredictionSnapshot] = None) -> Dict[str, Any]:
        """Append a fact, then run the automatic chain:
        cost backfill -> prediction checks -> cost feedback -> dormancy
        wakeup -> C1-C6 hints.

        ``retain_reason`` is an EXPLICIT, optional representative-evidence
        mark (reserved for future compaction policies): a non-empty value
        wins; otherwise the mark the record already carries is preserved.
        No automatic retention marking is performed.

        ``prediction`` is the pre-execution cost prediction ACTUALLY used
        (from :meth:`predict_cost`); it is frozen onto the record as the
        snapshot that feedback is computed against — feedback never re-reads
        a post-hoc current estimate. ``override`` backfills harness-owned
        dimensions (llm_tokens, retries the harness declares, extra tool
        calls) with explicit accounting: replace (default, idempotent —
        re-applying the same measurement never double-counts) or increment.

        ``override_mode`` / ``prediction`` are keyword-only so the historical
        positional call ``record(record, override, retain_reason)`` keeps its
        original meaning.

        Cost feedback is computed AFTER the backfill, from the frozen
        snapshot and the amended actual value, and persisted with the fact.
        A later ``update_cost`` re-computes it, so the stored summary can
        never disagree with the stored cost.

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
        # Explicit retention mark wins; otherwise keep the record's value.
        if retain_reason and retain_reason.strip():
            record.retention_reason = retain_reason.strip()
        # Freeze the pre-execution prediction snapshot when the harness
        # supplies one and the record does not already carry it.
        if prediction is not None and record.prediction_snapshot is None:
            record.prediction_snapshot = prediction
        self.bank.append(record)
        if override:
            self.bank.update_cost(record.execution_id,
                                  mode=override_mode, **override)
            record = self.bank.get(record.execution_id)
        self.bank.clear_pending(record.execution_id)

        prediction_events = self._check_predictions(record)
        cost_feedback = compute_cost_feedback(
            record.prediction_snapshot, record.strategy_id,
            record.measurement_scope, record.cost)
        if cost_feedback is not None:
            self.bank.set_cost_feedback(record.execution_id, cost_feedback)
        expected_map = {e.strategy_id: {"quality": e.expected_quality_hat}
                        for e in self.sbank.matching(record.profile_snapshot)}
        # expected_map already uses {sid: {"quality": q}} format, matching
        # the new check_triggers signature.
        prior_failures = self._prior_failures(record)
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
        if cost_feedback is not None:
            result["cost_feedback"] = cost_feedback
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        return result

    def induce(self, *, strategy_id: Optional[str] = None, all_: bool = False,
               rebuild: bool = False, widen: Optional[str] = None,
               tighten: Optional[str] = None,
               dry_run: bool = False, force: bool = False,
               llm_conditions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Consolidate Execution Evidence into Strategic Knowledge.

        Input = facts (ExecutionRecord rows, source="executed"); output =
        derived StrategicEntry commitments (expected quality/cost/failure
        risk). TARGET semantics (next Induction migration round): admission
        validation completes BEFORE an entry enters the bank, in offline
        induction; online execution only records new evidence. CURRENT
        status: entries are born candidate and promoted/demoted online by
        forward quality checks; cost feedback never alters entry state. In
        either case, once admitted an entry's validity does not depend on
        the survival of the supporting evidence rows. New entries inherit
        the catalog vocabulary's strategy_type/actions — extension points
        for future induction — without ever overwriting harness-supplied
        values.
        """
        if rebuild:
            result = self.induction.rebuild(dry_run=dry_run)
            if not dry_run:
                for entry_id in result.get("entry_ids", []):
                    self._enrich_entry(entry_id)
            return result
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
        if not dry_run:
            for r in results:
                entry_id = r.get("created") or r.get("updated")
                if entry_id:
                    self._enrich_entry(entry_id)
        return {"results": results}

    def _enrich_entry(self, entry_id: str) -> None:
        """Inherit catalog vocabulary (strategy_type, actions) into an entry.

        Only fills EMPTY fields — never overwrites values the harness already
        supplied. Keeps InductionEngine decoupled from the catalog while
        letting new entries carry the vocabulary's structural knowledge.
        """
        entry = self.sbank.get(entry_id)
        if entry is None:
            return
        strat = self.catalog.get(entry.strategy_id)
        if strat is None:
            return
        changed = False
        if not entry.strategy_type and strat.strategy_type:
            entry.strategy_type = strat.strategy_type
            changed = True
        if not entry.actions and strat.actions:
            entry.actions = list(strat.actions)
            changed = True
        if changed:
            self.sbank.update(entry)

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

        Quality-side only. COST feedback is decoupled: it is computed solely
        against the record's frozen pre-execution snapshot (same strategy,
        same scope, both sides measured) via ``compute_cost_feedback`` —
        never against every matching entry here, so strategy A's execution
        can never audit strategy B's cost prediction, and cost feedback never
        mutates entry state or calibration online."""
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
            events.append(event)
        return events

    def task_cost_summary(self, task_id: str, *,
                          end_to_end_latency_s: Optional[float] = None
                          ) -> Dict[str, Any]:
        """Aggregate a task's full cost across its explicitly linked records.

        Only attempt-scope facts (the sole explicit association: shared
        ``task_id``) participate; a task-scope record can never be merged in.
        Every attempt is charged once, to the strategy that actually ran it —
        "A fails -> A retries -> B succeeds" charges all three, never all to
        B and never only the last success.

        Aggregation rules:
        - Cumulative dimensions (llm_tokens, tool_calls, solver_runtime_s)
          SUM over measured attempts.
        - retries sums too, because a record's retries expresses the NEW
          retries this attempt adds (a harness declaration, not a running
          total) — three attempts declared 0/1/1 total two retries.
        - Per-attempt latencies are reported as facts; end-to-end latency is
          NEVER inferred by taking max or summing them. It is only reported
          when the harness provides explicit task timing
          (``end_to_end_latency_s``); otherwise it is unknown.
        - A dimension any attempt did not measure is reported as incomplete
          ("known partial total"), never as a complete total.
        """
        records = [r for r in self.bank.query(task_id=task_id)
                   if r.source == "executed"
                   and r.measurement_scope == "attempt"]
        total: Dict[str, float] = {d: 0.0 for d in COST_DIMENSIONS}
        n_measured: Dict[str, int] = {d: 0 for d in COST_DIMENSIONS}
        per_attempt = []
        for rec in records:
            measured = rec.cost.measured_dims()
            for dim in COST_DIMENSIONS:
                if dim in measured:
                    n_measured[dim] += 1
                    if dim != "latency_s":
                        total[dim] += getattr(rec.cost, dim)
            per_attempt.append({
                "execution_id": rec.execution_id,
                "strategy_id": rec.strategy_id,
                "status": rec.quality.get("status"),
                "cost": rec.cost.to_dict(),
                "cost_measured": sorted(measured),
                "latency_s": rec.cost.latency_s,
            })
        complete = {d: (n_measured[d] == len(records) and len(records) > 0)
                    for d in COST_DIMENSIONS}
        # Latency never appears in total_cost: attempts may overlap, and
        # end-to-end wait is a separate harness-supplied fact (see below).
        total_cost = {d: (round(total[d], 4) if n_measured[d] > 0 else None)
                      for d in COST_DIMENSIONS if d != "latency_s"}
        return {
            "task_id": task_id,
            "n_attempts": len(records),
            "attempts": per_attempt,
            "total_cost": total_cost,
            "complete": complete,
            "n_measured": n_measured,
            "end_to_end_latency_s": (
                round(end_to_end_latency_s, 4)
                if end_to_end_latency_s is not None else None),
            "end_to_end_latency_source": (
                "harness-supplied" if end_to_end_latency_s is not None
                else "unknown"),
            "aggregation_note": (
                "cumulative dimensions summed over measured attempts; "
                "retries = sum of per-attempt NEW retries; per-attempt "
                "latencies reported separately and end-to-end latency is "
                "unknown unless the harness supplies explicit task timing "
                "(never inferred by max or sum); incomplete dimensions are "
                "marked complete=false — known partial totals are not "
                "complete totals"),
        }

    def _prior_failures(self, record: ExecutionRecord) -> List[ExecutionRecord]:
        """Failed executions (bank + staged) for the same task, excluding
        this record itself."""
        prior = [r for r in self.bank.query(task_id=record.task_id)
                 if r.execution_id != record.execution_id
                 and not r.quality.get("feasible", False)]
        prior += [p for p in self.bank.pending(task_id=record.task_id)
                  if p.execution_id != record.execution_id
                  and not p.quality.get("feasible", False)]
        return prior

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
