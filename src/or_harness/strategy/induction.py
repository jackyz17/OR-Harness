"""Induction: submit agent-formed strategies as structured knowledge.

The core question of induction is "what does the evidence entitle me to
claim?" — a strategy's applicability is read off the very executions that
support it: the family they came from, and the structural cell those
executions occupy. Nothing is guessed from a mean: the agent READS the
material (``ORHarness.induction_material`` — the methods actually performed,
the outcomes, the failures, the before/after changes, the same-task attempt
chains, the existing strategies) and submits the strategy in its own words.

ONE entry is ONE strategy: :meth:`InductionEngine.submit_relation` either
creates a new entry or revises an existing one under the same derived
identity, and two independent strategies never share an entry. There is NO
statistical path that turns a cell's means into a technique — the framework
will not derive a method from a strategy name and a number. This is the only
place knowledge changes.

A strategy is a reusable modelling, decomposition, search, checking or
repair technique — not necessarily the whole plan of one execution and not
necessarily one solver. It carries: the structure and conditions it applies
to; the concrete method; a grounded explanation and expected effect; its
cost, risks and boundary; and its supporting evidence, counterexamples and
anything still unverified.

Publication is a SEPARATE gate from saving. A strategy is saved and may be
verified as a fact about the tasks it cites, but a TRANSFERABLE strategy
needs the same mechanism observed on >= ``CLAIM_MIN_TASKS`` independent
tasks (distinct ``task_id``). Different tasks, strategy ids and cells may be
cited TOGETHER — there is no same-name / same-cell requirement. A single
observation that is verified publishes only as a ``conditional_fact`` stamped
``single_observation`` / ``transferability: unproven``; it never masquerades
as a rule. The distinct-task count is computed from the evidence the strategy
ACTUALLY cites, never padded.

The four observation angles — method contrast, recovery after an
intervention, structural reproduction, advantage reversal — are THINKING AIDS
in the guidance, not detectors the framework runs (the old detectors are
gone). The only LLM injection point is phrasing: the harness may attach
free-text applicability notes, which are kept for the reader and never
scored.

Verification is the CHECK block inside a submitted strategy (the embedded
``check`` — the same shape a separate ``--verify`` payload carries). The
framework evaluates the declared, computable checks; the agent is
responsible for whether the method explanation and the evidence agree. When
BOTH an embedded ``check`` and a separate ``--verify`` are supplied they are
a CONFLICT, reported, never silently merged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    CLAIM_MIN_TASKS,
    CostVector,
    ExecutionRecord,
    GROUPING_FEATURES,
    PredictionTrack,
    StrategicEntry,
    empty_verification,
    evidence_predicates,
    group_key,
    normalize_claim,
    validate_claim,
)
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank, apply_transitions
from or_harness.strategy.verification import verify_relation

#: A method description counts as substance when it names the steps actually
#: taken. The gate below refuses a claim that rests on evidence which reports
#: NO method content, because a strategy name plus a mean is a statistic, not
#: a technique.
def _has_method_content(record: ExecutionRecord) -> bool:
    """Whether a record reports anything about HOW the work was done.

    A planned method alone counts: the agent stated the method it intended,
    which is real (if partial) material for a claim. Neither side present
    means the record holds a name and numbers only.
    """
    planned = record.method_planned
    actual = record.method_actual
    return bool((planned or {}).get("steps") or (planned or {}).get("name")
                or (actual or {}).get("steps") or (actual or {}).get("name"))


def _resolve_relation_check(raw: Dict[str, Any],
                            verify: Optional[Dict[str, Any]]
                            ) -> Tuple[Optional[Dict[str, Any]],
                                       Optional[Dict[str, Any]]]:
    """Unify the two ways a relation's declared checks may be supplied.

    A ``--relation`` payload may carry its checks INSIDE the relation (a
    ``check`` block, as the documentation's example shows) or the caller may
    pass a separate ``--verify`` payload. Historically only the latter was
    read, so an embedded ``check`` was silently ignored and a claim that
    looked verified was published unverified. This normalizes both into ONE
    ``verify`` payload and reports where the checks came from, so the caller
    can never be surprised about which checks a verdict covered.

    Returns ``(verify, note)``. When both forms are present, ``verify`` wins
    and ``note`` records the override (``check_source: "verify_arg"`` plus
    the keys the embedded block declared) — a conflict is REPORTED, not
    swallowed. When only the embedded block is present it is promoted to a
    ``relation``-purpose verify payload. Nothing here changes what is
    checked; it only makes the two spellings equivalent.
    """
    embedded = raw.get("check")
    if not isinstance(embedded, dict) or not embedded:
        embedded = None
    if verify and embedded:
        return verify, {
            "check_source": "verify_arg",
            "note": ("both a standalone --verify payload and an embedded "
                     "relation 'check' were supplied; --verify WINS and the "
                     "embedded check was overridden and NOT evaluated"),
            "overridden_embedded_keys": sorted(embedded),
        }
    if verify:
        return verify, None
    if embedded:
        promoted = {
            "claim": raw.get("claim") or raw.get("text"),
            "check": embedded,
        }
        return promoted, {"check_source": "embedded_relation_check"}
    return verify, None


def relation_material_gate(relation: Dict[str, Any],
                           records: Sequence[ExecutionRecord]
                           ) -> Optional[Dict[str, Any]]:
    """Report that a relation's supporting evidence carries no method.

    The framework's rule for semantic induction: IT will not turn a strategy
    name and a mean into a technique. The agent, however, may state the
    method in its own claim text — that is precisely the abstraction the
    semantic-induction step asks for, and refusing it would be the framework
    second-guessing a better-informed author. So this is a WARNING, not a
    veto: the outcome records it, and the reader can see that the claim rests
    on numbers rather than on recorded processing.

    It fires only when NO cited execution reports a method AND the relation
    declares no ``method`` field of its own. The corresponding hard
    guarantee is elsewhere: the framework never writes method prose itself,
    and ``induction-material`` marks a batch whose evidence reports no method
    as ``missing: method_performed`` so the agent is told what to go read.
    """
    if any(_has_method_content(rec) for rec in records):
        return None
    declared = relation.get("method")
    if isinstance(declared, dict) and (declared.get("name")
                                       or declared.get("steps")):
        return None
    return {
        "reason": ("no supporting execution reports a method and the claim "
                   "declares none: the evidence holds a strategy name and "
                   "numbers only. The claim is saved as written, but the "
                   "framework did not and will not derive a technique from "
                   "those numbers — record how the work was actually done "
                   "(`orx execute --method`, or the solve script's "
                   "'method_performed' receipt) to make the claim's basis "
                   "reviewable"),
        "records_with_method": 0,
        "records": len(records),
    }


class InductionEngine:
    def __init__(self, stats: ConditionalStats, sbank: StrategicBank):
        self.stats = stats
        self.sbank = sbank

    # -- knowledge claims (ONE claim per entry) ---------------------------------

    def submit_relation(self, raw: Dict[str, Any], *,
                        dry_run: bool = False, force: bool = False,
                        verify: Optional[Dict[str, Any]] = None,
                        notes: Optional[List[str]] = None
                        ) -> Dict[str, Any]:
        """Create or refresh ONE knowledge CLAIM as a standalone entry.

        The unit of knowledge is the ENTRY, and one entry is one claim: a
        structured assertion (condition -> how -> consequence -> boundary)
        grounded in explicitly referenced evidence, with its OWN
        verification. There is no separate relation structure and no host
        lookup — a claim submitted here either creates a new entry or
        revises an existing one under the same identity (its
        ``strategy_id``), and two independent claims never share an entry.

        ``raw`` carries ``claim`` text plus ``evidence`` (execution ids and
        the role each plays). The framework DERIVES everything the evidence
        implies (tasks, family, structural cell, strategy ids) — the caller
        submits only ids and roles. Verification reuses
        :func:`verify_relation`; the cross-task independence requirement
        (:data:`CLAIM_MIN_TASKS`) is a PUBLICATION gate, not a save gate: a
        single-task claim is saved and may be verified as a fact about that
        task, but it is not published as transferable knowledge.

        The declared checks may be supplied in EITHER place: the standalone
        ``verify`` payload (``{"claim", "check", "executions"}``) OR a
        ``check`` block INSIDE ``raw`` (``{"assertions": [...]}``), which is
        what the documentation's ``--relation`` example shows. The two are
        unified here so an embedded ``check`` is never silently ignored. If
        BOTH are present, ``verify`` WINS and the outcome records
        ``check_source: "verify_arg"`` with a note that the embedded
        ``check`` was overridden — the conflict is reported, never swallowed.
        """
        # The stored claim is built on the schema's validate_claim so the
        # write path has ONE shape authority; ``claim`` text keeps the
        # caller's own key for backward compatibility.
        claim = validate_claim({
            "text": raw.get("claim") or raw.get("text"),
            "kind": raw.get("kind") or raw.get("source"),
            "subject": raw.get("subject"),
            "conditions": raw.get("conditions"),
            "method": raw.get("method"),
            "evidence": raw.get("evidence"),
        })
        # Unify the two check syntaxes BEFORE any verification runs, so the
        # embedded form is never dropped on the floor.
        verify, check_note = _resolve_relation_check(raw, verify)
        subject = claim.get("subject")
        # Resolve the referenced executions and derive the evidence identity.
        resolved = self._resolve_claim_evidence(claim["evidence"])
        if resolved.get("problem") is not None:
            return {"saved": None, "skipped": resolved["problem"]}
        records = resolved["records"]
        claim["evidence"] = resolved["evidence"]
        claim["strategy_ids"] = resolved["strategy_ids"]
        claim["tasks"] = resolved["tasks"]
        claim["family"] = resolved["family"]
        claim["cell"] = resolved["cell"]
        # MATERIAL REPORT (never a veto). A claim that restates a strategy
        # name and a mean is not a technique; when the cited evidence reports
        # no method at all, the outcome SAYS SO — and the claim the agent
        # wrote is still saved, because it may legitimately name the method
        # itself. What the framework refuses to do is derive a technique from
        # numbers alone.
        material_gate = relation_material_gate(
            {"claim": claim["text"], "method": claim.get("method")}, records)
        # Applicability: the claim's own conditions when declared, otherwise
        # the evidence's own structural cell.
        conditions = claim.get("conditions") or {}
        predicates = dict(conditions.get("predicates") or {})
        if not predicates:
            predicates = evidence_predicates(
                records, family=resolved["family"] or None)
        # Verification (optional): the entry's OWN verdict, covering the
        # declared checks over the cited evidence.
        if verify:
            report = verify_relation(
                str(verify.get("claim") or claim["text"]),
                evidence=verify.get("executions") or records,
                roles=claim["evidence"],
                assertions=(verify.get("check") or {}).get("assertions"))
            verification = report
        else:
            verification = empty_verification()
        verification["scope"] = dict(verification.get("scope") or {})
        verification["scope"].setdefault("distinct_tasks", len(claim["tasks"]))

        # Identity: the entry's ``strategy_id`` names the claim. A claim with
        # an explicit free-form ``subject`` uses it; otherwise the evidence's
        # single strategy names it. One entry === one claim.
        effective_subject = subject
        if not effective_subject and len(resolved["strategy_ids"]) == 1:
            effective_subject = resolved["strategy_ids"][0]
        # An EXPLICIT ``target_entry_id`` names the entry to revise
        # unambiguously: a substantive edit does not have to re-derive the
        # target from subject + cell + kind. When it does not match, the
        # submission REFUSES rather than silently revising a different entry
        # (or creating a new one under a name the caller did not intend).
        target_entry_id = raw.get("target_entry_id")
        entry = None
        if target_entry_id:
            entry = self.sbank.get(str(target_entry_id))
            if entry is None:
                return {"saved": None, "skipped": (
                    f"unknown target_entry_id {target_entry_id!r}: a revision "
                    "must name an existing entry (or omit it to create/refresh "
                    "by identity)")}
        else:
            entry = self._find_claim_entry(effective_subject or "claim",
                                           predicates, claim.get("kind"))
        if dry_run:
            return {"saved": None,
                    "would_" + ("update" if entry is not None else "create"):
                        entry.entry_id if entry is not None else
                        (effective_subject or "claim"),
                    "claim": claim,
                    "material": material_gate,
                    "check_note": check_note,
                    "publication": self._claim_publication_placeholder(claim,
                                                                       verification)}
        if entry is None:
            entry = self._new_claim_entry(
                effective_subject or "claim", predicates, records)
            veto = self._claim_veto(entry)
            if veto is not None:
                if not force:
                    return {"saved": None, "vetoed": veto,
                            "material": material_gate,
                            "check_note": check_note,
                            "skipped": ("cold-archive veto (use --force to "
                                        "override)")}
                self.sbank.revive(veto["pattern_hash"], force=True)
            entry.claim = claim
            entry.verification = verification
            entry.applicability = [str(n).strip() for n in (notes or [])
                                   if str(n).strip()]
            self.sbank.add(entry)
            return {"saved": entry.entry_id, "created_entry": entry.entry_id,
                    "entry": entry.to_dict(), "claim": claim,
                    "material": material_gate,
                    "check_note": check_note,
                    "publication": self._claim_publication(entry)}
        # Existing entry: revise the claim in place (dedup by content). A
        # SUBSTANTIVE change to an already-verified claim invalidates the
        # previous verdict: the old check no longer covers the new claim.
        merged, stale = self._merge_claim(
            entry.claim or {}, claim, entry.verification or {}, verification,
            fresh_verdict=bool(verify))
        entry.claim = merged
        entry.verification = verification if verify else stale
        entry.provenance = [r.execution_id for r in records][:50]
        new_notes = [str(n).strip() for n in (notes or []) if str(n).strip()]
        if new_notes:
            entry.applicability = list(entry.applicability or []) + new_notes
        self.sbank.update(entry)
        return {"saved": entry.entry_id, "updated_entry": entry.entry_id,
                "claim": merged,
                "material": material_gate,
                "check_note": check_note,
                "publication": self._claim_publication(entry)}

    def _resolve_claim_evidence(self, evidence: List[Dict[str, Any]]
                                ) -> Dict[str, Any]:
        """Resolve execution ids to records and derive the evidence identity.

        Nothing here is taken from the caller except the ids and roles: the
        tasks, family, structural cell and strategy ids are read from the
        recorded facts, so a caller cannot submit a second, contradictory
        identity for the same evidence."""
        bank = self.stats.bank
        records: List[ExecutionRecord] = []
        resolved: List[Dict[str, Any]] = []
        for item in evidence:
            record = bank.get(item["execution_id"])
            if record is None:
                return {"problem": (f"unknown execution {item['execution_id']!r}: "
                                    "a claim may only reference recorded "
                                    "facts")}
            if record.source != "executed":
                return {"problem": (f"execution {item['execution_id']!r} is "
                                    f"{record.source!r}, not 'executed': only "
                                    "real facts may support a claim")}
            records.append(record)
            resolved.append({
                "execution_id": record.execution_id,
                "role": item["role"],
                "task_id": record.task_id,
                "strategy_id": record.strategy_id,
                "family": record.profile_snapshot.family,
                "measurement_scope": record.measurement_scope,
            })
        tasks = sorted({r.task_id for r in records if r.task_id})
        families = {r.profile_snapshot.family for r in records}
        family = sorted(families)[0] if len(families) == 1 else None
        cells = {group_key(r.profile_snapshot) for r in records}
        cell = sorted(cells)[0] if len(cells) == 1 else None
        strategy_ids = sorted({r.strategy_id for r in records})
        return {"records": records, "evidence": resolved, "tasks": tasks,
                "family": family, "cell": cell, "strategy_ids": strategy_ids,
                "problem": None}

    def _new_claim_entry(self, subject: Optional[str],
                         predicates: Dict[str, Any],
                         records: Sequence[ExecutionRecord] = ()
                         ) -> StrategicEntry:
        """A knowledge entry for a claim with no strategy host.

        A claim is a METHOD explanation (condition -> how -> consequence ->
        boundary), NOT a statistical quality/cost claim: the entry is
        ``claim_only`` (``support_n == 0``) and makes no expected-value claim.
        The cited executions' statistics stay readable through
        ``ConditionalStats`` (which the world model still reads); they are not
        dressed up as an entry estimate here."""
        return StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=str(subject or "claim"),
            pattern={"predicates": dict(predicates)},
            expected_quality_hat=0.0,
            quality_interval=(0.0, 1.0),
            expected_cost_hat=CostVector(measured=set()),
            failure_prob=0.0,
            support_n=0,
            provenance=[r.execution_id for r in records][:50],
            verification=empty_verification(),
            claim=None,
        )

    def _find_claim_entry(self, subject: str,
                          predicates: Dict[str, Any],
                          kind: Optional[str]
                          ) -> Optional[StrategicEntry]:
        """The claim-bearing entry this submission REVISES, when one exists.

        Only CLAIM-bearing entries are candidates (an entry with a
        ``claim``). A statistical entry with the same identity is NOT a host:
        a claim and a statistical claim are separate knowledge objects and
        must not inherit each other's verdict — so this returns None when the
        only match is statistical, and a new claim-only entry is created.

        Matching is by ``strategy_id`` AND the claim's structural CELL AND
        its ``kind``, so two independent claims under one subject (a
        different cell, or a different kind) stay independent entries and
        never inherit each other's verification."""
        for entry in self.sbank.list(strategy_id=str(subject),
                                     include_dormant=True):
            claim = entry.claim
            if claim is None:
                continue
            if not self._same_cell(entry.predicates, predicates):
                continue
            if (claim.get("kind") or None) != (kind or None):
                continue
            return entry
        return None

    def _claim_veto(self, entry: StrategicEntry) -> Optional[Dict[str, Any]]:
        card = self.sbank.archive_vetoes(entry.strategy_id,
                                         entry.predicates)
        if card is None:
            return None
        return {"pattern_hash": card.pattern_hash, "reason": card.reason}

    @staticmethod
    def _merge_claim(current: Dict[str, Any],
                     incoming: Dict[str, Any], current_verification: Dict[str, Any],
                     incoming_verification: Dict[str, Any], *,
                     fresh_verdict: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Revise an existing claim in place, marking a stale verdict.

        Returns ``(merged_claim, verification)``. A SUBSTANTIVE change (claim
        text, conditions, evidence set, method) invalidates the previous
        verdict: the old check no longer covers the new claim. A pure
        re-submission with identical content is a no-op.

        ``fresh_verdict`` says the caller supplied a NEW verification. That
        verdict WINS outright — it was computed over the incoming evidence, so
        the previous one is neither kept nor marked stale."""
        substantive_keys = ("text", "conditions", "method", "kind")
        substantive = any(current.get(k) != incoming.get(k)
                          for k in substantive_keys)
        current_evidence = [(e.get("execution_id"), e.get("role"))
                            for e in current.get("evidence") or []]
        incoming_evidence = [(e.get("execution_id"), e.get("role"))
                             for e in incoming.get("evidence") or []]
        if current_evidence != incoming_evidence:
            substantive = True
        if fresh_verdict:
            # The incoming verdict already reflects the incoming evidence.
            return incoming, incoming_verification
        previous = dict(current_verification or {})
        if substantive and previous.get("state") == "verified":
            stale = dict(previous)
            stale["stale_after_revision"] = True
            stale["stale_reason"] = (
                "claim substantively revised (text/conditions/evidence/"
                "method) without a fresh verification; re-submit with a "
                "verification (a `--verify` payload or an embedded relation "
                "`check` block) to re-publish")
            return incoming, stale
        if not substantive and previous.get("state") == "verified":
            # Identical re-submission: keep the existing verdict.
            return incoming, previous
        return incoming, (previous or incoming_verification)

    @staticmethod
    def _claim_publication_placeholder(claim: Dict[str, Any],
                                       verification: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        state = str((verification or {}).get("state") or "unverified")
        scope = (verification or {}).get("scope") or {}
        tasks = {str(t) for t in (scope.get("tasks") or []) if str(t)} \
            or {str(t) for t in (claim.get("tasks") or []) if str(t)}
        kind = str(claim.get("kind") or "")
        single_fact = kind == "conditional_fact"
        reasons: List[str] = []
        if state != "verified":
            reasons.append(f"verification state is {state!r}")
        if not single_fact and len(tasks) < CLAIM_MIN_TASKS:
            reasons.append(
                f"verification covers {len(tasks)} task(s); a transferable "
                f"knowledge claim needs >= {CLAIM_MIN_TASKS} independent tasks")
        out = {"published": state == "verified"
                             and (single_fact or len(tasks) >= CLAIM_MIN_TASKS),
               "state": state, "distinct_tasks": len(tasks),
               "required_tasks": CLAIM_MIN_TASKS, "reasons": reasons}
        if single_fact:
            out["support_scope"] = "single_observation"
            out["transferability"] = "unproven"
        return out

    @staticmethod
    def _claim_publication(entry: StrategicEntry) -> Dict[str, Any]:
        """Why a claim-bearing entry is (or is not) publishable.

        Two publication rules, by claim KIND:

        * ``conditional_fact`` — a verified statement about the evidence it
          cites, INCLUDING a single observation ("under this structure, this
          method produced a checked-correct answer"). It publishes with ONE
          task, but it is stamped ``support_scope: single_observation`` and
          ``transferability: unproven`` so no reader mistakes it for a rule.
        * everything else — a TRANSFERABLE claim, which needs
          >= ``CLAIM_MIN_TASKS`` independent tasks: one task's observation is
          not transferable knowledge.
        """
        state = str((entry.verification or {}).get("state") or "unverified")
        block = entry.verification or {}
        scope = block.get("scope") or {}
        claim = entry.claim or {}
        tasks = {str(t) for t in (scope.get("tasks") or []) if str(t)} \
            or {str(t) for t in (claim.get("tasks") or []) if str(t)}
        single_fact = str(claim.get("kind") or "") == "conditional_fact"
        reasons: List[str] = []
        if block.get("stale_after_revision"):
            reasons.append("the claim was revised without a fresh verification")
        elif state != "verified":
            reasons.append(f"verification state is {state!r}")
        if not single_fact and len(tasks) < CLAIM_MIN_TASKS:
            reasons.append(
                f"verification covers {len(tasks)} task(s); a transferable "
                f"knowledge claim needs >= {CLAIM_MIN_TASKS} independent tasks "
                "(a single-task repair is a verified fact about that task, "
                "not yet knowledge)")
        out = {"published": (state == "verified"
                             and not block.get("stale_after_revision")
                             and (single_fact or len(tasks) >= CLAIM_MIN_TASKS)),
               "state": state, "distinct_tasks": len(tasks),
               "required_tasks": CLAIM_MIN_TASKS, "reasons": reasons}
        if single_fact:
            out["support_scope"] = "single_observation"
            out["transferability"] = "unproven"
            out["note"] = ("a conditional FACT about the cited evidence: it "
                           "publishes with one observation but makes NO "
                           "transfer claim — more evidence either widens it "
                           "into a rule or supersedes it")
        return out

    # -- offline revalidation --------------------------------------------------

    def revise(self, strategy_id: Optional[str] = None, *,
               dry_run: bool = False) -> List[Dict[str, Any]]:
        """Re-derive lifecycle state from the frozen evidence (offline).

        Online recording never changes knowledge: it writes one frozen check
        per matching entry onto the fact
        (``execution_features.quality_feedback``), taken against the interval
        that was in force at that moment. This pass replays those checks:

        - the forward track (n_predictions / hits / consecutive misses /
          calibration error) is REBUILT from the frozen checks — never
          re-scored against the entry's current interval;
        - a check that missed is a CONTENT miss: the claim covered that task
          when the execution ran, so the failure is evidence against the
          claim. Three consecutive content misses demote to ``suspect``;
        - promotion (n >= 5 checks, hit rate >= 0.7) and demotion are applied
          HERE, through the same rules the per-event API uses
          (:func:`apply_transitions`);
        - a dormant entry wakes when matching evidence is newer than its
          recency mark (``last_consulted_at`` or ``created_at``) — the same
          mark dormancy aging uses.

        This is lifecycle maintenance of EXISTING knowledge (a management
        ability), not statistical induction; it writes no new strategy. It
        runs as part of the ``induce`` knowledge write.

        ``dry_run`` returns the same report without writing anything.
        """
        checks_by_entry = self._frozen_checks()
        report: List[Dict[str, Any]] = []
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=True):
            checks = checks_by_entry.get(entry.entry_id)
            if not checks:
                continue  # no forward evidence for this entry yet
            rebuilt = self._replay(checks)
            probe = StrategicEntry.from_dict(entry.to_dict())
            track = probe.prediction_track
            track.n_predictions = rebuilt.n_predictions
            track.n_hits = rebuilt.n_hits
            track.consecutive_misses = rebuilt.consecutive_misses
            track.calibration_error = rebuilt.calibration_error
            newest = max(c["created_at"] for c in checks)
            recency = probe.last_consulted_at or probe.created_at
            transitions = (apply_transitions(probe)
                           if probe.status != "dormant" or newest > recency
                           else [])
            item: Dict[str, Any] = {
                "entry_id": entry.entry_id,
                "strategy_id": entry.strategy_id,
                "forward": {"n_predictions": track.n_predictions,
                            "n_hits": track.n_hits,
                            "hit_rate": round(track.hit_rate, 4),
                            "consecutive_misses": track.consecutive_misses,
                            "calibration_error": round(track.calibration_error,
                                                       4)},
                "misses": [c["execution_id"] for c in checks if not c["hit"]],
                "transitions": list(transitions),
            }
            if dry_run:
                report.append(item)
                continue
            live = entry.prediction_track
            live.n_predictions = track.n_predictions
            live.n_hits = track.n_hits
            live.consecutive_misses = track.consecutive_misses
            live.calibration_error = track.calibration_error
            entry.status = probe.status
            self.sbank.update(entry)
            report.append(item)
        return report

    def _frozen_checks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Frozen forward checks in the Evidence Bank, grouped by entry.

        Each check carries the fact it came from (execution id, creation
        time) so the replay can order the checks chronologically."""
        by_entry: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self.stats.bank.all():
            if rec.source != "executed":
                continue
            for raw in (rec.execution_features.get("quality_feedback") or []):
                entry_id = str(raw.get("entry_id", ""))
                if not entry_id:
                    continue
                by_entry.setdefault(entry_id, []).append({
                    "execution_id": rec.execution_id,
                    "created_at": rec.created_at,
                    "hit": bool(raw.get("hit", False)),
                    "observed": float(raw.get("observed", 0.0)),
                    "predicted": float(raw.get("predicted", 0.0)),
                })
        for checks in by_entry.values():
            checks.sort(key=lambda c: (c["created_at"], c["execution_id"]))
        return by_entry

    @staticmethod
    def _replay(checks: Sequence[Dict[str, Any]]) -> PredictionTrack:
        """Rebuild a forward track from frozen checks, chronologically."""
        track = PredictionTrack()
        for check in checks:
            track.record(check["hit"],
                         abs(check["observed"] - check["predicted"]))
        return track


    @staticmethod
    def _same_cell(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """True when two predicate sets describe the same structural cell."""
        if a.get("family") != b.get("family"):
            return False
        for f in GROUPING_FEATURES:
            if a.get(f) != b.get(f):
                return False
        return True

