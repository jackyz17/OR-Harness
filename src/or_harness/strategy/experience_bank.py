"""Execution Evidence Bank: the episodic fact layer, bounded by a window.

Records what actually happened — never what will happen. Facts are neutral:
the disposal ladder (suspect/dormant/retired/cold archive) applies only to
the derived Strategic Knowledge Bank. The Evidence Bank is the single source
of truth for what was observed.

Mutability contract (fact-preserving while retained, bounded by a window):
  - ``append``: the only way a new fact enters. Duplicate ids are rejected.
  - ``update_cost``: the sole backfill channel (e.g. llm_tokens becomes known
    later). Only cost dimensions may change; nothing else is ever rewritten.
    Backfill is EXPLICITLY REPLACEMENT by default (idempotent: re-applying
    the same measurement never double-counts); ``increment`` mode exists for
    harness-owned counters delivered in parts within the attempt's declared
    scope. Backfilled dimensions become measured. Any persisted cost
    feedback is re-computed against the amended value so the stored summary
    can never disagree with the fact (no stale actual=110 vs 1000).
  - ``set_cost_feedback``: the only channel that writes the record's cost
    feedback annotation (a computed summary over the frozen prediction
    snapshot and the actual cost). Narrow by design — it cannot rewrite any
    other execution feature.
  - ``set_task_check``: the only channel that writes the record's TASK-RESULT
    check annotation (``execution_features.task_check``). Narrow for the same
    reason: the solver's own verdict says the MODEL was solved, never that the
    answer satisfies the TASK, so the task-level verdict is a separate fact
    attached to the same record. It is written AFTER the fact may already be
    recorded (a late correction is a real event), and it never rewrites the
    observed quality, the cost, or the source — an answer confirmed wrong is
    still the answer that was produced, and its cost is still real.
  - ``stage_pending`` / ``clear_pending``: a no-lost-facts safety net between
    execution and the harness's explicit recording decision.
  - ``delete_episode_executions``: the ONE deletion channel, and it removes a
    WHOLE episode's executions — never a single row, so a contrast/repair
    chain is never split. Retention is bounded by the evidence window (see
    ``or_harness.world_model.episode_closeout.EvidenceWindowPolicy``): a
    fact retained inside the window is immutable under every path above, and
    outside it the whole episode leaves at once. What is NOT kept forever is
    the raw row; the derivation it supported (a knowledge entry's verification
    and range) is self-sufficient and survives. While a fact is retained it
    is a fact: the window bounds retention, it never licenses rewriting.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import (
    COST_DIMENSIONS,
    ExecutionRecord,
    compute_cost_feedback,
    group_key,
)
from or_harness.core.storage import Store, StorageError

#: Where a amended cost dimension's number CAME FROM. Kept per dimension so
#: a real measurement is never indistinguishable from a declared estimate:
#: - ``provider_usage``: a model provider reported it (a real observation);
#: - ``agent_observed``: a number READ OFF a real report (a provider
#:   dashboard, a host log line) — an OBSERVATION, so it may stand as a
#:   measured fact, and the caller must opt in explicitly;
#: - ``agent_estimate``: a value typed from memory or assumed, with no
#:   source behind it — the DEFAULT for a bare ``--override``.
#: The default is the honest one: a bare hand-typed number is a
#: DECLARATION. It is kept and shown, but it is never used as a calibration
#: actual, a measured cost claim or a learning evidence mean — the caller
#: says ``agent_observed``/``provider_usage`` when the number really came
#: from a report. See :mod:`or_harness.world_model.cost_eligibility` for the
#: one rule.
COST_SOURCES: tuple = ("provider_usage", "agent_observed", "agent_estimate")

#: Dimensions the FRAMEWORK itself measures during a run. Backfilling one of
#: these overwrites a real observation, so it takes an explicit ``force``.
FRAMEWORK_MEASURED_DIMS: tuple = ("latency_s", "solver_runtime_s")


class ExperienceBank:
    """Append-only store of :class:`ExecutionRecord` facts (the Execution
    Evidence layer). One fact = one episode of what actually happened:
    actual strategy, actual quality, actual cost, observed failures,
    implementation artifacts."""

    def __init__(self, store: Store):
        self.store = store

    @staticmethod
    def _framework_measured_dims(rec: ExecutionRecord) -> set:
        """The dimensions of one record the FRAMEWORK measured, not declared.

        ``latency_s`` / ``solver_runtime_s`` are measured by the executor;
        any dimension whose recorded provenance says ``provider_usage`` was
        measured by a provider. Everything else is a declaration (or
        unknown), so an amend may replace it freely.
        """
        dims = set(FRAMEWORK_MEASURED_DIMS)
        provenance = rec.execution_features.get("cost_provenance") or {}
        for dim, entry in provenance.items():
            if isinstance(entry, dict) \
                    and entry.get("source") == "provider_usage":
                dims.add(str(dim))
        return dims

    # -- writes -----------------------------------------------------------------

    def append(self, record: ExecutionRecord) -> str:
        """Append a fact. Raises on duplicate id or malformed payload."""
        # Validate by round-trip before persisting (malformed rejection).
        payload = ExecutionRecord.from_dict(record.to_dict())
        with self.store.transaction() as conn:
            try:
                conn.execute(
                    "INSERT INTO executions "
                    "(execution_id, task_id, strategy_id, family, group_l1, "
                    " source, created_at, payload) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        payload.execution_id,
                        payload.task_id,
                        payload.strategy_id,
                        payload.profile_snapshot.family,
                        payload.group_l1,
                        payload.source,
                        payload.created_at,
                        self.store.dumps(payload.to_dict()),
                    ),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise StorageError(
                        f"duplicate execution_id {payload.execution_id!r}: "
                        "the Experience Bank is append-only") from exc
                raise
        return payload.execution_id

    def update_cost(self, execution_id: str, *,
                    mode: str = "replace", source: str = "agent_estimate",
                    force: bool = False, basis: Optional[str] = None,
                    **dimensions: float) -> ExecutionRecord:
        """Backfill cost dimensions (harness-owned llm_tokens via --override).

        Appends nothing: this amends the fact's measured fields in place, which
        is the documented supplement channel. Only cost dimensions may change.

        ``mode`` makes the accounting explicit so nothing is double-counted:
        - ``replace`` (default): the value IS the measurement. Re-applying
          the same override is idempotent — the same tokens can never be
          added twice.
        - ``increment``: the value is an additional measured amount within
          the record's declared scope.

        ``source`` records WHERE the number came from (:data:`COST_SOURCES`):
        ``provider_usage`` (the provider reported it), ``agent_observed``
        (the operator read it off a real report), or ``agent_estimate`` —
        the DEFAULT for a bare backfill, because a hand-typed number with no
        stated source is a declaration. It is kept per dimension in
        ``execution_features.cost_provenance`` so a reader can tell a real
        measurement from a declaration — the two must never be
        indistinguishable in the stored fact, and an ``agent_estimate``
        stays out of every measured-truth consumer (calibration actuals,
        cost claims, learning means).

        ``force`` guards the ONE dangerous case: overwriting a dimension the
        FRAMEWORK itself measured (``latency_s`` / ``solver_runtime_s``, and
        any dimension recorded with ``source="provider_usage"``). A plain
        amend of such a dimension is refused (named) rather than silently
        rewriting a real observation.

        Backfilled dimensions are marked measured. Cost feedback is
        re-computed against the frozen prediction snapshot whenever a
        snapshot exists: the stored summary never disagrees with the stored
        fact, and a dimension that was unknown at record time produces
        feedback as soon as it becomes comparable.
        """
        if mode not in ("replace", "increment"):
            raise StorageError(f"unknown backfill mode {mode!r} "
                               "(expected 'replace' or 'increment')")
        if source not in COST_SOURCES:
            raise StorageError(f"unknown cost source {source!r} "
                               f"(expected one of {list(COST_SOURCES)})")
        unknown = set(dimensions) - set(COST_DIMENSIONS)
        if unknown:
            raise StorageError(f"unknown cost dimensions: {sorted(unknown)}")
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        # Overwrite guard: a framework-measured dimension is a real
        # observation. Rewriting it silently would destroy measured truth
        # (and re-derive feedback against a number that never happened), so
        # it takes an explicit ``force``.
        framework_measured = self._framework_measured_dims(rec)
        clobbered = sorted((set(dimensions) & framework_measured)
                           & (set(rec.cost.measured_dims())
                              | {"latency_s", "solver_runtime_s"}))
        if clobbered and not force:
            raise StorageError(
                f"refusing to overwrite framework-measured dimension(s) "
                f"{clobbered}: this record really measured them. Pass "
                "force=True (or --force) to replace them deliberately")
        # A declared tool_calls below the sandbox's provable floor is not a
        # measurement, it is a contradiction: the executor demonstrably made
        # at least that many calls. Refuse it rather than store a number the
        # record itself refutes.
        floor = rec.execution_features.get("tool_calls_lower_bound")
        if "tool_calls" in dimensions and floor is not None:
            declared = float(dimensions["tool_calls"])
            if declared < float(floor):
                raise StorageError(
                    f"tool_calls={declared} is below the provable lower bound "
                    f"({floor}): the executor itself ran the solve script at "
                    "least that many times. tool_calls counts ALL tool "
                    "invocations in the attempt's scope (shell commands, file "
                    "reads/writes, sandbox runs, solver calls)")
        for d, v in dimensions.items():
            if mode == "increment":
                setattr(rec.cost, d, float(getattr(rec.cost, d)) + float(v))
            else:
                setattr(rec.cost, d, float(v))
        rec.cost.mark_measured(*dimensions)
        # Provenance, per dimension: which dimensions were amended THIS
        # call and where the number came from. Kept on the fact so a reader
        # never has to guess whether ``llm_tokens`` is a real measurement or
        # a declared estimate.
        provenance = dict(rec.execution_features.get("cost_provenance") or {})
        for d in dimensions:
            entry = {"source": source, "mode": mode,
                     "amended_at": time.time()}
            # The token口径 (prompt+completion vs one side) is part of the
            # number's identity for ``llm_tokens``: a single-side figure is
            # a LOWER bound, and a later reader (and the eligibility rule)
            # must be able to tell it from a complete total.
            if basis is not None and d == "llm_tokens":
                entry["basis"] = str(basis)
            provenance[d] = entry
        rec.execution_features["cost_provenance"] = provenance
        # Recompute feedback whenever a prediction snapshot exists — not
        # only when feedback was already stored. A dimension that was
        # UNKNOWN at record time (so no feedback could be computed then)
        # becomes comparable once it is backfilled, and must produce
        # feedback now. No snapshot -> nothing to compare -> untouched.
        #
        # DECLARATION SCREENING: a DECLARED value (an ``agent_estimate``
        # amendment, the ``--override`` default) is not a measured truth, so
        # it must not manufacture cost feedback against the prediction. It
        # stays on the record for display, and the dimensions it would have
        # fed are reported as excluded with the reason instead.
        if rec.prediction_snapshot is not None:
            feedback = compute_cost_feedback(rec.prediction_snapshot,
                                             rec.strategy_id,
                                             rec.measurement_scope,
                                             rec.cost)
            if feedback is not None:
                from or_harness.world_model.cost_eligibility import (
                    MEASURED,
                    cost_eligibility,
                )
                # A DECLARED (or single-side) dimension is not a measured
                # truth, so it is dropped from the comparison rather than
                # manufacturing an error against the prediction.
                for key in ("per_dim", "per_dimension"):
                    block = dict(feedback.get(key) or {})
                    for dim in list(block):
                        if cost_eligibility(rec, dim) != MEASURED:
                            block.pop(dim)
                    if key in feedback:
                        feedback[key] = block
                if isinstance(feedback.get("n_dims"), int):
                    feedback["n_dims"] = len(feedback.get("per_dim") or {})
                if not feedback.get("per_dim") \
                        and not feedback.get("per_dimension"):
                    feedback = None
            if feedback is None:
                rec.execution_features.pop("cost_feedback", None)
            else:
                rec.execution_features["cost_feedback"] = feedback
        with self.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def annotate_features(self, execution_id: str,
                          patch: Dict[str, Any], *,
                          allow_staged: bool = True) -> ExecutionRecord:
        """Write a narrow ANNOTATION onto a fact's ``execution_features``.

        An annotation is a computed or observed ADDITION to a fact (frozen
        prediction checks, induction hints, a correction link) — never a
        rewrite of the observation itself: ``quality``, ``cost`` and
        ``source`` are untouched. This is the SAME narrow channel as
        :meth:`set_cost_feedback` / :meth:`set_task_check`, generalized so a
        new annotation does not need its own copy of the read-modify-write.

        ``patch`` maps feature keys to values; a value of ``None`` REMOVES
        the key (an annotation withdrawn is unknown, never a default).
        ``allow_staged`` also updates a staged-but-unrecorded execution in
        place, so an annotation attached by ``execute`` survives
        ``record --from-staged`` verbatim.
        """
        rec = self.get(execution_id)
        staged = False
        if rec is None and allow_staged:
            rec = self.get_pending(execution_id)
            staged = rec is not None
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        for key, value in patch.items():
            if value is None:
                rec.execution_features.pop(key, None)
            else:
                rec.execution_features[key] = value
        table = "pending_executions" if staged else "executions"
        with self.store.transaction() as conn:
            conn.execute(
                f"UPDATE {table} SET payload=? WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def set_cost_feedback(self, execution_id: str,
                          feedback: Optional[Dict[str, Any]]) -> ExecutionRecord:
        """Write (or remove) the record's cost feedback annotation.

        A narrow, single-purpose channel: it only ever touches the
        ``cost_feedback`` key inside ``execution_features`` (a computed
        summary over the frozen prediction snapshot and the actual cost).
        It cannot rewrite any other execution feature, historical fact, or
        derived-layer state.
        """
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if feedback is None:
            rec.execution_features.pop("cost_feedback", None)
        else:
            rec.execution_features["cost_feedback"] = dict(feedback)
        with self.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def set_task_check(self, execution_id: str,
                       report: Optional[Dict[str, Any]]) -> ExecutionRecord:
        """Write (or clear) the record's TASK-RESULT check annotation.

        The second narrow channel, same shape as :meth:`set_cost_feedback`:
        it touches ONLY ``execution_features.task_check`` and can rewrite no
        other feature, no observed quality, no cost, and no ``source``.

        Why this is a separate fact rather than an edit of ``quality``: the
        solver's ``quality`` says the MODEL was solved (a legal status, a
        finite objective, a gap). It cannot say the answer satisfies the
        TASK — a relaxed LP answered with fractional values is ``optimal``
        with ``gap=0`` and still wrong. Overwriting ``quality`` would also
        destroy the observation, and the observation is exactly what a
        correction has to be judged against.

        The report may arrive AFTER the fact was recorded (a correction
        learned late). That is the point: statistics and calibration re-read
        this annotation, so a corrected validity takes effect on the next
        read, while the original execution, its prediction and any stored
        historical evaluation stay untouched. Staged (unrecorded) executions
        are updated in place too, so the annotation survives
        ``record --from-staged`` verbatim.

        ``report=None`` removes the annotation (a check withdrawn — the
        answer's validity returns to UNKNOWN, never to a default verdict).
        """
        rec = self.get(execution_id)
        staged = False
        if rec is None:
            rec = self.get_pending(execution_id)
            staged = rec is not None
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if report is None:
            rec.execution_features.pop("task_check", None)
        else:
            rec.execution_features["task_check"] = dict(report)
        table = "pending_executions" if staged else "executions"
        with self.store.transaction() as conn:
            conn.execute(
                f"UPDATE {table} SET payload=? WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def exclude(self, execution_id: str, reason: str, *,
                superseded_by: Optional[str] = None) -> ExecutionRecord:
        """Withdraw a fact from the evidence set WITHOUT deleting it.

        The Evidence Bank is append-only, so a wrong observation cannot be
        erased — but it must stop counting. This is the explicit correction
        channel: the row is preserved for audit, its ``source`` becomes
        ``"excluded"`` (every statistics / induction / trigger / retrieval
        path requires ``source == "executed"``, so the fact drops out of
        all of them at once), and the reason is recorded on the fact under
        ``execution_features.correction``.

        ``superseded_by`` names the execution that replaces it (e.g. a
        re-run with a corrected answer) — a link, never an auto-inference:
        the correction is always the harness's explicit statement. Nothing
        is recomputed here; derived layers pick the change up at the next
        ``induce`` / ``rebuild-index``."""
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if rec.source == "excluded":
            raise StorageError(
                f"execution {execution_id!r} is already excluded")
        if superseded_by is not None and self.get(superseded_by) is None:
            raise StorageError(
                f"superseded_by execution {superseded_by!r} does not exist")
        rec.execution_features["correction"] = {
            "excluded": True,
            "reason": str(reason),
            "superseded_by": (str(superseded_by)
                              if superseded_by is not None else None),
            "excluded_at": time.time(),
        }
        rec.source = "excluded"
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE executions SET payload=?, source=? "
                "WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), rec.source, execution_id))
        return rec

    def restore(self, execution_id: str, reason: str) -> ExecutionRecord:
        """Reverse :meth:`exclude`: the fact counts as evidence again.

        A second explicit statement, because an exclusion can itself be
        wrong. The correction record is kept (``restored`` + reason) rather
        than erased, so the fact's history shows both decisions."""
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if rec.source != "excluded":
            raise StorageError(
                f"execution {execution_id!r} is not excluded (source="
                f"{rec.source!r})")
        correction = dict(rec.execution_features.get("correction") or {})
        correction["restored"] = True
        correction["restore_reason"] = str(reason)
        correction["restored_at"] = time.time()
        rec.execution_features["correction"] = correction
        rec.source = "executed"
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE executions SET payload=?, source=? "
                "WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), rec.source, execution_id))
        return rec

    # -- reads --------------------------------------------------------------------

    def get(self, execution_id: str) -> Optional[ExecutionRecord]:
        row = self.store.conn.execute(
            "SELECT payload FROM executions WHERE execution_id=?",
            (execution_id,)).fetchone()
        return self._decode(row) if row else None

    def query(self, *, task_id: Optional[str] = None,
              strategy_id: Optional[str] = None,
              family: Optional[str] = None,
              group_l1: Optional[str] = None,
              source: Optional[str] = None,
              scope: Optional[str] = None,
              limit: Optional[int] = None) -> List[ExecutionRecord]:
        """Query facts.

        ``family`` is the authoritative selector. ``group_l1`` is a DERIVED
        INDEX column whose FORMAT has changed over time — it has held at least
        ``family=routing|sc[..]|rc[..]|..`` (ladder era) and plain
        ``family=routing`` (the version that retired the ladder) — so it is
        never used to decide membership:

        - with ``group_l1`` given, the query first narrows to that family and
          then re-derives the structural key from each record's own profile
          snapshot, so facts written under any historical index format stay
          visible. Relying on the column directly is exactly how a whole
          generation of records dropped out of the statistics.
        """
        sql = "SELECT payload FROM executions"
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?"); params.append(task_id)
        if strategy_id is not None:
            clauses.append("strategy_id=?"); params.append(strategy_id)
        if group_l1 is not None and family is None:
            family = group_l1.split("|", 1)[0].replace("family=", "", 1) or None
        if family is not None:
            clauses.append("family=?"); params.append(family)
        if source is not None:
            clauses.append("source=?"); params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, execution_id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        records = [self._decode(r) for r in
                   self.store.conn.execute(sql, params).fetchall()]
        if group_l1 is not None:
            records = [r for r in records
                       if group_key(r.profile_snapshot) == group_l1]
        if scope is not None:
            records = [r for r in records if r.measurement_scope == scope]
        return records

    def index_health(self) -> Dict[str, Any]:
        """Read-only diagnostic: how many rows carry a stale ``group_l1``.

        ``group_l1`` is a DERIVED index (it must equal
        :func:`or_harness.core.schema.group_key` of the row's own profile),
        and it has held at least two historical formats. Reads never depend
        on it now, but a database upgraded from an older version is worth
        reporting honestly — including the previous release's plain
        ``family=routing`` form, which an earlier check mistook for healthy.
        Nothing is written here."""
        stale = 0
        for row in self.store.conn.execute("SELECT group_l1, payload FROM executions"):
            try:
                record = self._decode(row)
            except StorageError:
                stale += 1
                continue
            if str(row["group_l1"]) != record.group_l1:
                stale += 1
        total = self.count()
        return {"rows": total, "stale_group_index": stale,
                "note": ("group_l1 is a derived index; reads re-derive the key "
                         "from each record's profile snapshot, so stale rows "
                         "are still counted correctly")}

    def all(self) -> List[ExecutionRecord]:
        return self.query()

    def count(self) -> int:
        row = self.store.conn.execute("SELECT COUNT(*) AS n FROM executions").fetchone()
        return int(row["n"])

    # -- pending staging (the no-lost-facts safety net) --------------------------

    def stage_pending(self, record: ExecutionRecord) -> str:
        """Stage an execution produced by `orx execute` before the harness
        decides to record it. Every execution — successes AND failures — is
        staged automatically, so a failed attempt can never be silently lost
        when the harness immediately retries with a different approach.
        Staging is not recording: the fact enters the Experience Bank only
        via :meth:`append` (the harness's explicit decision)."""
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_executions "
                "(execution_id, task_id, created_at, payload) VALUES (?,?,?,?)",
                (record.execution_id, record.task_id, record.created_at,
                 self.store.dumps(record.to_dict())))
        return record.execution_id

    def pending(self, *, task_id: Optional[str] = None) -> List[ExecutionRecord]:
        """Staged-but-unrecorded executions, oldest first."""
        sql = "SELECT payload FROM pending_executions"
        params: List[Any] = []
        if task_id is not None:
            sql += " WHERE task_id=?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, execution_id ASC"
        return [self._decode(r) for r in
                self.store.conn.execute(sql, params).fetchall()]

    def get_pending(self, execution_id: str) -> Optional[ExecutionRecord]:
        row = self.store.conn.execute(
            "SELECT payload FROM pending_executions WHERE execution_id=?",
            (execution_id,)).fetchone()
        return self._decode(row) if row else None

    def clear_pending(self, execution_id: str) -> None:
        """Remove a staged execution after it was recorded (or explicitly
        discarded by the harness)."""
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM pending_executions WHERE execution_id=?",
                         (execution_id,))

    # -- bounded window (the ONLY deletion channel on facts) ----------------------

    def delete_episode_executions(self, execution_ids: Sequence[str]) -> int:
        """Remove a COMPLETE episode's executions (the evidence window).

        The single deletion channel on the Evidence Bank, and it is
        deliberately coarse: it takes a whole episode's execution ids, never
        one row, so a contrast/repair chain can never be split. Callers
        decide the window (see
        ``or_harness.world_model.episode_closeout.enforce_evidence_window``)
        and pass the episode's full execution set; the derived layers
        (vectors, orphaned task texts) are cleaned by the caller in the same
        maintenance step.

        While a fact is RETAINED it stays a fact — the window is the
        documented bound on retention, not a licence to rewrite. Idempotent:
        re-running with ids already gone removes nothing and does not raise.
        """
        ids = [str(e) for e in execution_ids]
        if not ids:
            return 0
        removed = 0
        with self.store.transaction() as conn:
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM executions WHERE execution_id "
                    f"IN ({placeholders})", chunk)
                removed += cur.rowcount
        return removed

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _decode(row: Any) -> ExecutionRecord:
        try:
            return ExecutionRecord.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt execution row: {exc}") from exc
