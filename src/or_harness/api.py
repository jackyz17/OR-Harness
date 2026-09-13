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
    group_key,
    profile_matches,
)
from or_harness.core.storage import Store, resolve_home
from or_harness.execution.executor import SafePythonExecutor
from or_harness.profiling.profiler import derivation_report, profile_task
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import GarbageCollector
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector, is_publishable
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
        # Published knowledge only: a candidate whose admission verification
        # is missing a verdict must not act as a verified entry prediction
        # either. Gating recall alone would leave this second door open —
        # the snapshot then falls back to the statistics ladder below, which
        # is exactly what "we have no verified knowledge yet" should look
        # like.
        entry = next((e for e in self.sbank.matching(profile)
                      if e.strategy_id == strategy_id
                      and is_publishable(e)), None)
        if entry is not None:
            return PredictionSnapshot(
                strategy_id=strategy_id,
                expected_cost=entry.expected_cost_hat,
                source="entry", measurement_scope="attempt",
                support_n=entry.support_n,
                support_per_dim=dict(entry.cost_support_n),
                evidence_refs=[entry.entry_id],
                note=f"strategic entry {entry.entry_id} ({entry.status})")
        cell = self.stats.for_profile(profile).get(strategy_id)
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
        frozen quality checks -> cost backfill -> cost feedback -> C1-C6
        hints.

        The chain is EVIDENCE-ONLY: it never promotes, demotes, or awakens a
        Strategic Knowledge entry. Quality checks are written onto the fact
        (``execution_features.quality_feedback``); the next explicit
        ``induce`` replays them offline (``InductionEngine.revise``).

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
        # Frozen quality checks are computed against the interval in force
        # RIGHT NOW and persisted with the fact — nothing downstream
        # re-scores a later execution against a post-hoc interval.
        prediction_checks = self._check_predictions(record)
        if prediction_checks:
            record.execution_features["quality_feedback"] = prediction_checks
        self.bank.append(record)
        if override:
            self.bank.update_cost(record.execution_id,
                                  mode=override_mode, **override)
            record = self.bank.get(record.execution_id)
        self.bank.clear_pending(record.execution_id)

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
            "prediction_checks": prediction_checks,
            "induction_hints": [h.to_dict() for h in hints],
        }
        if cost_feedback is not None:
            result["cost_feedback"] = cost_feedback
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        return result

    def induce(self, *, strategy_id: Optional[str] = None, all_: bool = False,
               rebuild: bool = False,
               dry_run: bool = False, force: bool = False,
               notes: Optional[List[str]] = None,
               verify: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Consolidate Execution Evidence into Strategic Knowledge.

        Input = facts (ExecutionRecord rows, source="executed"); output =
        derived StrategicEntry commitments (expected quality/cost/failure
        risk). This is where knowledge changes: recording only accumulates
        evidence, and `induce` (i) forms candidates from the statistics of
        each structural cell, (ii) creates/refreshes entries, and (iii)
        REVISES existing entries from the frozen forward checks recorded on
        the facts — promotion (n>=5, hit rate>=0.7), demotion (3 consecutive
        misses), dormancy wakeup — reported under ``revisions``. Cost feedback
        never alters entry state.
        Once an entry exists its validity does not depend on the survival of
        the supporting evidence rows. New entries inherit the catalog
        vocabulary's strategy_type/actions — extension points for future
        induction — without ever overwriting harness-supplied values.

        ``verify`` carries the harness's admission check for the candidate
        this call forms (see ``InductionEngine.induce``); the verdict is
        computed by the framework from real executions. Without it the entry
        is ``unverified`` and is not published as strategic knowledge —
        recall falls back to the raw conditional statistics.
        """
        if rebuild:
            result = self.induction.rebuild(dry_run=dry_run)
            if not dry_run:
                for entry_id in result.get("entry_ids", []):
                    self._enrich_entry(entry_id)
                result["revisions"] = self.induction.revise()
            return result
        targets = self._induction_targets(strategy_id, all_)
        results = []
        for profile, sid in targets:
            results.append(self.induction.induce(
                profile, sid, dry_run=dry_run, force=force,
                notes=notes, verify=verify))
        if not dry_run:
            for r in results:
                entry_id = r.get("created") or r.get("updated")
                if entry_id:
                    self._enrich_entry(entry_id)
        out: Dict[str, Any] = {"results": results}
        if targets:
            # Offline revision of the entries this call covers: lifecycle
            # state is re-derived from the frozen checks on the facts.
            out["revisions"] = self.induction.revise(strategy_id=strategy_id,
                                                     dry_run=dry_run)
        return out

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
        if not entry.fallback_strategy_id and strat.fallback:
            entry.fallback_strategy_id = strat.fallback
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
            # Echo the REQUESTED bank name ("archive"): the other branches do
            # the same, and callers switch on this field.
            return {"bank": "archive", "count": len(cards),
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
            # group_l1 is a derived index: report staleness (read-only — the
            # open path never rewrites a bank) so an upgraded database is
            # visibly diagnosed instead of silently mysterious.
            "index_health": self.bank.index_health(),
            "pending_staged_executions": [
                {"execution_id": p.execution_id, "task_id": p.task_id,
                 "strategy_id": p.strategy_id,
                 "status": p.quality.get("status")} for p in pending],
        }

    def close(self) -> None:
        self.store.close()

    # -- automatic chain internals ------------------------------------------------

    def _check_predictions(self, record: ExecutionRecord) -> List[Dict[str, Any]]:
        """Frozen forward checks: this execution against matching entries'
        intervals, as EVIDENCE.

        Online the harness only accumulates: each check is computed against
        the interval in force at this moment and written onto the fact
        (``execution_features.quality_feedback``). No entry is promoted,
        demoted, tightened, or awakened here — ``InductionEngine.revise``
        replays these checks at the next offline induction.

        Isolation rules (mirroring the cost-feedback contract):
        - the strategy that ACTUALLY ran owns the check (A never audits B);
        - only attempt-scope executions produce checks: a task-scope total
          never audits attempt-scope knowledge;
        - dormant entries are included — a matching execution is evidence
          about the pattern, and waking the entry is an offline decision.
        """
        if record.measurement_scope != "attempt":
            return []
        observed = quality_score(record)
        events: List[Dict[str, Any]] = []
        for entry in self.sbank.matching(record.profile_snapshot,
                                         include_dormant=True):
            if entry.strategy_id != record.strategy_id:
                continue
            lo, hi = entry.quality_interval
            width = max(hi - lo, 1e-6)
            slack = PREDICTION_HIT_SLACK * width
            events.append({
                "entry_id": entry.entry_id,
                "predicted": round(entry.expected_quality_hat, 4),
                "interval": [lo, hi],
                "observed": round(observed, 4),
                "hit": bool(lo - slack <= observed <= hi + slack),
            })
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
        """One induction target per (structural group, strategy).

        The group is DERIVED from each record's own profile snapshot rather
        than read from the stored index column: legacy rows carry the old
        index format, and letting that decide targets would make the facts
        invisible to induction."""
        targets = []
        seen = set()
        for rec in self.bank.all():
            if rec.source != "executed" or rec.measurement_scope != "attempt":
                continue
            if strategy_id and rec.strategy_id != strategy_id:
                continue
            key = (group_key(rec.profile_snapshot), rec.strategy_id)
            if key in seen:
                continue
            if not all_ and strategy_id is None:
                continue
            seen.add(key)
            targets.append((rec.profile_snapshot, rec.strategy_id))
        return targets
