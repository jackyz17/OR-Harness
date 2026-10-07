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

Publication is a SEPARATE gate from saving, and it is a VERIFICATION gate,
not a task-count gate: a strategy publishes when the claim's declared checks
hold over the evidence it cites. The number of independent tasks is reported
as a FACT (``distinct_tasks``) so a reader can weigh it, but two tasks are
not a proof and one task can reveal a conditional method with a derivation
behind it — the framework does not decide that question by counting. A
``conditional_fact`` is stamped ``single_observation`` /
``transferability: unproven`` when its evidence is one task; a transfer claim
carries its own scope and ``not_covered`` wording. The distinct-task count is
computed from the evidence the strategy ACTUALLY cites, never padded.

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
        """Submit ONE knowledge CLAIM as a NEW standalone entry.

        The unit of knowledge is the ENTRY, and one entry is one claim: a
        structured assertion (condition -> how -> consequence -> boundary)
        grounded in explicitly referenced evidence. Knowledge is ADDITIVE —
        this path only ever CREATES a new numbered entry. It never rewrites
        or merges an existing entry's core content: when new evidence
        revises or contradicts an existing claim, the revision is a SEPARATE
        entry, and the earlier one keeps its own text, conditions, evidence
        and verification (its use counts, feedback and lifecycle continue to
        update independently).

        ``raw`` carries ``claim`` text plus ``evidence`` (execution ids and
        the role each plays). The framework DERIVES everything the evidence
        implies (tasks, family, structural cell, strategy ids) — the caller
        submits only ids and roles.

        What the framework checks at submission is ADMINISTRATIVE ONLY:

        * the payload is well-formed and storable (``validate_claim``);
        * every cited execution really exists and is real evidence;
        * the assigned number is valid;
        * the write (and later the index sync) succeeds.

        There is NO publication gate on the CONTENT. A submitted claim is
        PUBLISHED as knowledge within its declared scope — the framework does
        not certify the conclusion. Any computable check (an embedded
        ``check`` block or a standalone ``verify`` payload) is EVALUATED and
        RECORDED as the agent's own audit trail, reported under
        ``verification`` — never required to publish. ``verification`` and
        the preserved ``not_covered`` wording state exactly what was checked;
        a claim with no check is published with ``verification.state ==
        "unverified"``, honestly labelled.
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
        # Verification is the agent's OWN audit trail: evaluate and record
        # whatever checks were declared, but never gate publication on them.
        # With no declared check the facts are still READ (``fact_checked``)
        # so a reader sees what was inspected rather than a bare claim.
        declared_assertions = ((verify or {}).get("check") or {}).get(
            "assertions")
        if verify:
            verification = verify_relation(
                str(verify.get("claim") or claim["text"]),
                evidence=verify.get("executions") or records,
                roles=claim["evidence"],
                assertions=declared_assertions)
        else:
            verification = verify_relation(
                claim["text"], evidence=records,
                roles=claim["evidence"], assertions=None)
        verification["scope"] = dict(verification.get("scope") or {})
        verification["scope"].setdefault("distinct_tasks", len(claim["tasks"]))

        # Identity: the entry's ``strategy_id`` NAMES the claim. A claim
        # with an explicit free-form ``subject`` uses it; otherwise the
        # evidence's single strategy names it. One entry === one claim.
        effective_subject = subject
        if not effective_subject and len(resolved["strategy_ids"]) == 1:
            effective_subject = resolved["strategy_ids"][0]
        if dry_run:
            return {"saved": None,
                    "would_create": (effective_subject or "claim"),
                    "claim": claim,
                    "material": material_gate,
                    "check_note": check_note,
                    "publication": self._claim_publication_placeholder(
                        claim, verification)}
        entry = self._new_claim_entry(
            effective_subject or "claim", predicates, records)
        # The cold archive still vetoes re-creating a RETIRED generalization
        # from the same (strategy, predicates) — an administrative
        # anti-resurrection guard, overridable with ``force``.
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
            # Identity is ASSIGNED by the bank on ``add``: the framework owns
            # the number, so no id is invented here (the placeholder is
            # overwritten).
            entry_id="",
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

    def _claim_veto(self, entry: StrategicEntry) -> Optional[Dict[str, Any]]:
        card = self.sbank.archive_vetoes(entry.strategy_id,
                                         entry.predicates)
        if card is None:
            return None
        return {"pattern_hash": card.pattern_hash, "reason": card.reason}

    @staticmethod
    def _claim_publication_placeholder(claim: Dict[str, Any],
                                       verification: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        state = str((verification or {}).get("state") or "unverified")
        scope = (verification or {}).get("scope") or {}
        tasks = {str(t) for t in (scope.get("tasks") or []) if str(t)} \
            or {str(t) for t in (claim.get("tasks") or []) if str(t)}
        single_fact = str(claim.get("kind") or "") == "conditional_fact"
        # Publication is the AGENT's decision to offer this knowledge within
        # its declared scope, not a framework verdict on the content.
        out = {"published": True,
               "state": state, "distinct_tasks": len(tasks),
               "note": ("publishing means the agent offers this knowledge "
                        "within its declared scope; it does not mean the "
                        "framework proved the conclusion"),
               "reasons": []}
        if single_fact or len(tasks) == 1:
            out["support_scope"] = "single_observation"
            out["transferability"] = "unproven"
        return out

    @staticmethod
    def _claim_publication(entry: StrategicEntry) -> Dict[str, Any]:
        """Why a claim-bearing entry is (or is not) offered as knowledge.

        Publication is the AGENT's call: a claim the agent submitted is
        offered within the scope it declared. The framework does NOT certify
        the conclusion, so this reports no content gate. What travels is the
        agent's own verification state (``verified`` / ``fact_checked`` /
        ``unverified``) and its scope, so a reader can see exactly what was
        checked — and ``not_covered`` on the verification block states that a
        passing check covers only the declared checks over the listed
        samples.
        """
        state = str((entry.verification or {}).get("state") or "unverified")
        block = entry.verification or {}
        scope = block.get("scope") or {}
        claim = entry.claim or {}
        tasks = {str(t) for t in (scope.get("tasks") or []) if str(t)} \
            or {str(t) for t in (claim.get("tasks") or []) if str(t)}
        single_fact = str(claim.get("kind") or "") == "conditional_fact"
        out = {"published": True,
               "state": state, "distinct_tasks": len(tasks),
               "note": ("publishing means the agent offers this knowledge "
                        "within its declared scope; it does not mean the "
                        "framework proved the conclusion"),
               "reasons": []}
        if single_fact or len(tasks) == 1:
            out["support_scope"] = "single_observation"
            out["transferability"] = "unproven"
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

