"""Induction: consolidate episodic facts into calibrated commitments.

The core question of induction is "what does the evidence entitle me to
claim?" — a claim's applicability is read off the very executions that
support it: the family they came from, and the structural cell those
executions occupy. Nothing is quantized into fixed bins by hand and nothing
has to be manually widened: a claim's cell is what its evidence demonstrated,
and in-scope failures are what pull it back down.

Entries are never born validated. Predictions are verified by future
executions (forward validation), not by self-test on training data. Those
verifications happen offline: recording freezes each check onto the fact, and
``revise`` replays them at induction time — promotion, demotion, and dormancy
wakeup all live here, never in the record chain. Quality checks are isolated
by strategy and scope, and cost deviations stay on the facts as evidence.

Admission is gated, revision is not: creating a claim takes two independent
tasks (three runs of one instance generalize about that instance), while an
entry that already exists is refreshed by any new matching evidence.

Induction is not limited to restating one cell's statistics. The patterns
worth generalizing are claims ACROSS evidence — how strategies compare
under one structural condition (``strategy_contrast``), what changed after an
intervention (``intervention_recovery``), whether a pattern recurs in an
independent family (``structural_reproduction``), and where a strategy's
advantage reverses (``advantage_reversal``). The detectors live in
:mod:`or_harness.strategy.triggers` and fire online; their persisted hints
carry the cross-execution evidence into the offline candidates
(``or_harness.world_model.maintenance``). A claim about such a pattern is
submitted as a STRUCTURED CLAIM (:meth:`InductionEngine.submit_relation`)
with the evidence that established it — the statistical path below never
phrases a contrast into free text. ONE entry is ONE claim: it either creates
a new entry or revises an existing one under the same identity, and two
independent claims never share an entry.

The only LLM injection point is phrasing: the harness may attach free-text
applicability notes, which are kept for the reader and never scored.

SEMANTIC INDUCTION is organized, not performed, here. The framework gathers
the material (``ORHarness.induction_material``: the methods actually used, the
two sides of a comparison, what followed, the outcome and check state), the
outer agent reads it and forms the "condition -> how -> consequence ->
boundary" claim in its own words, and
:meth:`InductionEngine.submit_relation` + :func:`verify_relation` check what
was submitted. Nothing invents a method: when the cited evidence reports no
method at all (and the claim declares none),
:func:`relation_material_gate` REPORTS that in the outcome — the claim is
saved as written, and the framework never derives a technique from a mean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import (
    CLAIM_MIN_TASKS,
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    GROUPING_FEATURES,
    PredictionTrack,
    ProblemProfile,
    StrategicEntry,
    empty_verification,
    evidence_predicates,
    group_key,
    min_interval_width,
    normalize_claim,
    predicates_cover,
    validate_claim,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats
from or_harness.strategy.strategic_bank import StrategicBank, apply_transitions
from or_harness.strategy.verification import (
    VERIFIED,
    verify_candidate,
    verify_relation,
)

#: v1 cost interval: multiplicative band around the point estimate. Actual
#: cost within [0.5x, 2.0x] of the prediction counts as a hit.
COST_INTERVAL_BAND = (0.5, 2.0)

#: Reported whenever a candidate is formed without an admission verdict: the
#: entry exists (and collects forward checks) but is not published knowledge.
UNVERIFIED_NOTE = ("recorded as an unverified candidate: not published as "
                   "strategic knowledge — recall falls back to conditional "
                   "statistics until an admission check passes")

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
    guarantee is elsewhere: the framework's own (statistical) induction
    writes no method prose at all, and ``induction_material`` marks such a
    candidate ``insufficient`` so the agent is told what is missing.
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

    # -- consolidation -----------------------------------------------------------

    def induce(self, profile: ProblemProfile, strategy_id: str, *,
               notes: Optional[List[str]] = None,
               dry_run: bool = False, force: bool = False,
               verify: Optional[Dict[str, Any]] = None,
               execution_ids: Optional[Sequence[str]] = None
               ) -> Dict[str, Any]:
        """Create or refresh the entry for (strategy, evidence set).

        Guard rails:
        - cold-archive veto (anti-resurrection): a retired pattern is refused
          unless ``force``, and ``force`` LIFTS the veto (removes the card)
          so the harness's judgment is made once, not repeated every round;
          a veto reports before the admission gate — it is the real blocker;
        - honest intervals: width floored by sample size;
        - no restatement-only entries: if an equally-wide or wider entry
          already covers the pattern with the same prediction, this is a no-op.

        Creating a claim also requires INDEPENDENT evidence: at least two
        distinct tasks. Repeating one task is repetition, not reproduction —
        it can still refresh a claim that already exists, but it cannot
        create one.

        The claim's predicates are read off the supporting records
        (:func:`evidence_predicates`), so refreshing with new evidence is what
        widens (or narrows) its applicability.

        ``notes`` are harness-written applicability notes (free text): kept on
        the entry for the reader, never scored. A CONTRAST against other
        evidence is not written here as prose any more: it is a relation
        claim, submitted through :meth:`submit_relation` with the evidence
        that established it (see ``orx induction-material`` for the material
        to read). The statistical path stays statistics.

        ``verify`` optionally carries the harness's offline admission check
        ``{"purpose", "claim", "check", "executions", "supporting"}``; the
        framework computes the verdict from real executions
        (:mod:`or_harness.strategy.verification`). Without it the entry is
        created ``unverified`` — a candidate that is NOT published as
        strategic knowledge.
        """
        if execution_ids is not None:
            # Explicit evidence scope (M4 bundle adoption): restrict to
            # the specified execution IDs — no silent scope widening.
            allowed = set(execution_ids)
            records = [r for r in self.stats.evidence(profile, strategy_id)
                       if r.execution_id in allowed]
        else:
            records = self.stats.evidence(profile, strategy_id)
        cell = self.stats.aggregate(group_key(profile), strategy_id, records)
        if cell.n < 2:
            return {"created": None, "skipped": "fewer than 2 supporting executions",
                    "cell": cell.to_dict()}
        predicates = evidence_predicates(
            records, family=profile.family)
        existing = self._find_existing(strategy_id, predicates,
                                       include_dormant=True)
        veto = self.sbank.archive_vetoes(strategy_id, predicates)
        if veto is not None:
            if not force:
                return {"created": None,
                        "vetoed": {"pattern_hash": veto.pattern_hash,
                                   "reason": veto.reason},
                        "skipped": "cold-archive veto (use --force to override)"}
            # The harness judged the environment drifted: lift the veto.
            # A rehearsal must not write — the lift happens only for real.
            if not dry_run:
                self.sbank.revive(veto.pattern_hash, force=True)
            else:
                veto = None
        tasks = sorted({r.task_id for r in records})
        if existing is None and len(tasks) < 2:
            # A veto (above) is the real blocker and reports first: telling the
            # harness to go collect a second task would send it down a path
            # that cannot succeed while the card stands.
            return {"created": None,
                    "verification": {"tasks": tasks, "required_tasks": 2},
                    "skipped": (f"needs independent evidence: all {cell.n} "
                                f"observations come from {len(tasks)} task "
                                f"{tasks} — a claim requires >=2 tasks"),
                    "cell": cell.to_dict()}

        quality_hat = cell.mean_quality
        lo, hi = self._honest_interval(cell)
        fail_prob = cell.fail_rate
        # Cost claims are COMPLETE-OR-SILENT. A dimension enters the entry's
        # cost claim only when EVERY supporting record measured it: a mean
        # over a subset ("3 of 4 records reported tokens") is a partial
        # observation dressed up as a full claim, and downstream that number
        # feeds strategy selection and world-model prediction as if it were
        # the whole truth. A dimension short of the full count is withheld —
        # the entry is still created, its quality claim and its other cost
        # dimensions are unaffected, and the withheld dimension simply reads
        # as unknown until the missing records are backfilled
        # (``orx record --override`` amends already-recorded facts in place).
        # The interval key set doubles as the entry's measured-dimension mask.
        complete_dims = cell.complete_dims()
        measured_cost_interval = {d: band for d, band in
                                  self._cost_interval().items()
                                  if d in complete_dims}
        withheld = {d: {"n_measured": int(cell.n_measured.get(d, 0)),
                        "n": int(cell.n)}
                    for d in COST_DIMENSIONS if d not in complete_dims}
        # A cost vector carrying ONLY the complete dimensions: an incomplete
        # dimension keeps a placeholder zero and stays out of the mask, so no
        # consumer can read a partial mean as a measured value.
        cost_hat = CostVector(
            **{d: (float(getattr(cell.mean_cost, d)) if d in complete_dims
                   else 0.0) for d in COST_DIMENSIONS},
            measured=set(complete_dims))
        note_texts = [str(n).strip() for n in (notes or []) if str(n).strip()]
        verification, verification_note = self._run_verification(
            verify, dry_run, strategy_id=strategy_id, profile=profile)

        if existing is not None:
            changed = (abs(existing.expected_quality_hat - quality_hat) > 0.02
                       or existing.support_n != cell.n
                       or existing.predicates != predicates
                       or abs(existing.failure_prob - fail_prob) > 0.02
                       or verification is not None
                       or self._cost_estimates_changed(existing, cost_hat,
                                                       measured_cost_interval,
                                                       cell.n_measured))
            if not changed and not note_texts:
                return {"created": None,
                        "skipped": f"entry {existing.entry_id} already encodes this "
                                   "evidence (restatement-only entries are forbidden)",
                        "entry_id": existing.entry_id}
            if dry_run:
                out = {"created": None, "would_update": existing.entry_id,
                       "cell": cell.to_dict()}
                self._note_withheld(out, withheld, cell)
                if verification is not None:
                    out["verification"] = verification
                return out
            # Substantive-revision check: when the CLAIM changes (predicates
            # or expected estimates beyond tolerance) without a fresh
            # admission verdict, the old verification no longer covers the
            # new claim — mark it stale rather than silently re-publishing
            # a revised claim under an old check. Pure support-count
            # growth does NOT trigger this: more evidence for the same
            # claim only sharpens it. Cost SUPPORT growth alone (same
            # point estimates, more measured samples) is likewise not a
            # claim change.
            cost_claim_changed = self._cost_estimates_changed(
                existing, cost_hat, measured_cost_interval, None)
            substantive = (existing.predicates != predicates
                           or abs(existing.expected_quality_hat
                                   - quality_hat) > 0.02
                           or abs(existing.failure_prob - fail_prob) > 0.02
                           or cost_claim_changed)
            existing.pattern = {"predicates": predicates}
            existing.expected_quality_hat = quality_hat
            existing.quality_interval = (lo, hi)
            existing.expected_cost_hat = cost_hat
            existing.cost_interval = measured_cost_interval
            existing.cost_support_n = dict(cell.n_measured)
            existing.failure_prob = fail_prob
            existing.support_n = cell.n
            existing.provenance = cell.execution_ids[:50]
            if verification is not None:
                existing.verification = verification
            elif substantive and existing.verification_state == "verified":
                existing.verification = dict(existing.verification)
                existing.verification["stale_after_revision"] = True
                existing.verification["stale_reason"] = (
                    "claim substantively revised (predicates or expected "
                    "estimates) without a fresh admission verdict; "
                    "re-verify with induce --verify to re-publish")
            if note_texts:
                existing.applicability.extend(note_texts)
            self.sbank.update(existing)
            out = {"updated": existing.entry_id, "cell": cell.to_dict(),
                   "predicates": predicates,
                   "notes_added": len(note_texts)}
            self._note_withheld(out, withheld, cell)
            if substantive and verification is None \
                    and existing.verification_state == "verified":
                out["verification_stale"] = True
            if verification is not None:
                out["verification"] = verification
            if verification_note is not None:
                out["skipped"] = verification_note
            return out

        if dry_run:
            out: Dict[str, Any] = {"would_create": {"strategy_id": strategy_id,
                                                    "predicates": predicates},
                                   "cell": cell.to_dict()}
            self._note_withheld(out, withheld, cell)
            if verification is not None:
                out["verification"] = verification
            return out
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=strategy_id,
            pattern={"predicates": predicates},
            expected_quality_hat=quality_hat,
            quality_interval=(lo, hi),
            expected_cost_hat=cost_hat,
            cost_interval=measured_cost_interval,
            cost_support_n=dict(cell.n_measured),
            failure_prob=fail_prob,
            applicability=note_texts,
            fallback_strategy_id=None,
            provenance=cell.execution_ids[:50],
            support_n=cell.n,
            verification=(verification if verification is not None
                          else empty_verification()),
        )
        self.sbank.add(entry)
        out = {"created": entry.entry_id, "entry": entry.to_dict(),
               "predicates": predicates, "cell": cell.to_dict()}
        self._note_withheld(out, withheld, cell)
        if verification is not None:
            out["verification"] = verification
        if verification_note is not None:
            out["skipped"] = verification_note
        elif verification is None:
            # No verdict supplied: the entry is recorded as a candidate but
            # is NOT published, and the caller is told exactly that instead
            # of having to infer it from an empty field.
            out["skipped"] = UNVERIFIED_NOTE
        return out

    # -- knowledge claims (ONE claim per entry) ---------------------------------

    def submit_relation(self, raw: Dict[str, Any], *,
                        dry_run: bool = False, force: bool = False,
                        verify: Optional[Dict[str, Any]] = None
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
                effective_subject or "claim", predicates)
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
                         predicates: Dict[str, Any]) -> StrategicEntry:
        """A knowledge entry for a claim with no strategy host."""
        return StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=str(subject or "claim"),
            pattern={"predicates": dict(predicates)},
            expected_quality_hat=0.0,
            quality_interval=(0.0, 1.0),
            expected_cost_hat=CostVector(measured=set()),
            failure_prob=0.0,
            support_n=0,
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


    @staticmethod
    def _note_withheld(out: Dict[str, Any], withheld: Dict[str, Dict[str, int]],
                       cell: GroupStats) -> None:
        """Attach the incomplete-dimension report to an induce outcome.

        Silence would leave the harness believing its entry carries a cost
        claim it does not. The report names the missing dimensions, how many
        records supported them, and how to close the gap — the fix is a
        backfill (``record --override`` amends an already-recorded fact), so
        nothing has to be re-run."""
        if not withheld:
            return
        out["cost_claim_withheld"] = {
            "dimensions": withheld,
            "supporting_executions": list(cell.execution_ids[:50]),
            "note": ("these dimensions were measured on only some supporting "
                     "records, so no cost claim was written for them (a mean "
                     "over a subset is a partial observation, not a claim). "
                     "Backfill the missing records with `orx amend-cost "
                     "<execution_id> --override <dim>=<value>` (amends in "
                     "place, idempotent) and re-run induce; the claim is "
                     "withheld, not refused"),
        }

    def _run_verification(self, verify: Optional[Dict[str, Any]],
                          dry_run: bool, *, strategy_id: str,
                          profile: ProblemProfile
                          ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Compute the admission verdict from real executions.

        Nothing is executed here and nothing is written: the harness hands in
        the executions it already produced (or intends to), and the framework
        applies the declared check. Returning ``None`` preserves whatever the
        entry already carries — re-inducing from the same evidence must not
        silently demote a verified claim back to unverified.

        The candidate's identity is passed in, so ONE payload can never be
        reused across induction targets: evidence for another strategy or
        family reports as not corresponding rather than verifying this
        candidate by accident."""
        if not verify:
            return None, None
        report = verify_candidate(
            verify.get("purpose"), str(verify.get("claim", "")),
            check=verify.get("check"),
            executions=verify.get("executions") or (),
            supporting=verify.get("supporting") or (),
            strategy_id=strategy_id, family=profile.family)
        state = report.get("state")
        note = None
        if state != VERIFIED:
            note = (f"candidate is {state}: not published as strategic "
                    "knowledge — " + str(report.get("conclusion", "")))
        return report, note

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
                            "calibration_error": round(track.calibration_error, 4)},
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

    def _cost_estimates_changed(self, existing: StrategicEntry,
                                new_hat: CostVector,
                                new_interval: Dict[str, Tuple[float, float]],
                                new_support: Optional[Dict[str, int]] = None
                                ) -> bool:
        """True when the entry's cost estimate needs refreshing: the measured
        dimension set changed, any measured dimension's point estimate moved
        materially (relative to its previous magnitude), or any dimension's
        effective sample size changed (identical means with more measured
        samples still raise the entry's — and its predictions' — support).
        This is the induction update-connection only — no induction
        refactoring."""
        if set(existing.cost_interval.keys()) != set(new_interval.keys()):
            return True
        if new_support is not None and dict(existing.cost_support_n) != dict(new_support):
            return True
        for dim in new_interval:
            old = getattr(existing.expected_cost_hat, dim)
            new = getattr(new_hat, dim)
            if abs(old - new) > 0.02 * max(abs(old), 1.0):
                return True
        return False

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _cost_interval() -> Dict[str, Tuple[float, float]]:
        """v1: fixed multiplicative band per dimension."""
        from or_harness.core.schema import COST_DIMENSIONS
        return {d: COST_INTERVAL_BAND for d in COST_DIMENSIONS}

    @staticmethod
    def _honest_interval(cell: GroupStats) -> Tuple[float, float]:
        """Interval honest to sample size: never narrower than the floor for
        n, centered on the observed mean, spread by the observed std.

        Full precision — this value is persisted (``StrategicEntry.to_dict``
        feeds the payload) and used for matching decisions, so rounding here
        would only move the defect to disk. Human-facing summaries round."""
        spread = max(cell.std_quality, min_interval_width(cell.n) / 2.0)
        lo = max(0.0, cell.mean_quality - spread)
        hi = min(1.0, cell.mean_quality + spread)
        if hi - lo < min_interval_width(cell.n):
            hi = min(1.0, lo + min_interval_width(cell.n))
            lo = max(0.0, hi - min_interval_width(cell.n))
        return (lo, hi)

    def _find_existing(self, strategy_id: str,
                       predicates: Dict[str, Any],
                       *, include_dormant: bool = True) -> Optional[StrategicEntry]:
        """The entry this evidence belongs to, if any.

        - an entry whose predicates already cover the new ones (the
          restatement case, including a family-free pattern a harness wrote);
        - otherwise the entry of the same (family, structural cell) evidence
          set: one evidence set owns exactly one claim, so growing evidence
          REFRESHES that claim instead of spawning a second one.

        Dormant entries are INCLUDED by default. Excluding them meant a
        dormant claim was invisible to the dedup pass, so induction created a
        fresh entry and then ``revise`` woke the old one — two entries for one
        knowledge object. Waking it and refreshing it are offline decisions
        made here, on the same entry id."""
        covering = self._find_covering(strategy_id, predicates,
                                       include_dormant=include_dormant)
        if covering is not None:
            return covering
        # Otherwise: the entry of the SAME CELL. Matching on the family alone
        # was the defect that merged structurally opposite regions (the fallback
        # below used to accept any entry naming the same family, so the second
        # cell "refreshed" the first cell's claim). The cell is fully described
        # by (family, the grouping-dimension predicates), so that is what is
        # compared — including the unknown bucket.
        if predicates.get("family") is None:
            return None
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=include_dormant):
            if self._same_cell(entry.predicates, predicates):
                return entry
        return None

    @staticmethod
    def _same_cell(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """True when two predicate sets describe the same structural cell."""
        if a.get("family") != b.get("family"):
            return False
        for f in GROUPING_FEATURES:
            if a.get(f) != b.get(f):
                return False
        return True

    def _find_covering(self, strategy_id: str,
                       predicates: Dict[str, Any],
                       *, include_dormant: bool = True) -> Optional[StrategicEntry]:
        for entry in self.sbank.list(strategy_id=strategy_id,
                                     include_dormant=include_dormant):
            if predicates_cover(entry.predicates, predicates):
                return entry
        return None

