"""Strategy recall: surface the REAL memory this cell has, in relevance order.

**Recall discovers material; it does not decide.** This module answers "what
has really been recorded about strategies in this structural cell?" — a
DISCOVERY channel. It deliberately produces NO utility score and applies NO
utility ranking: the old ``alpha*Q - beta*C - gamma*R`` ordering and its
``top`` cut silently evicted candidates, which is exactly the per-candidate
benefit/cost ranking that belongs to the world model's consequence
prediction and the agent's choice, not here.

What recall returns, per candidate, is a REPORT of the backing memory's
`expected`/`observed` values, its evidence basis, its support, and the
strategy's own recorded knowledge — for the agent to weigh. Ranked ordering
is by RELEVANCE (a strategic entry before a bare recount; then strategy id),
never by a utility scalar. ``top`` is a MATERIAL BUDGET: the caller may cap
how many rows it reads, and the cap is reported (``n_matched`` vs returned),
so a cap can never look like "nothing else matched".

**There is no candidate menu.** The candidate set is derived from memory
alone: a strategy is a candidate because something was really executed
(conditional statistics) or really induced (a strategic entry) for this
structural cell. When nothing matches, recall returns an EMPTY list — not a
placeholder row, not a zero quality. A caller-proposed list is filtered
against those same real memories; a proposed strategy with no memory is
reported as such.

Evidence precedence (per strategy):
  1. Strategic entries whose pattern matches the profile (commitments).
  2. Conditional statistics over the Experience Bank (recounts).

The method is named ``recall`` (not ``recommend``) because its purpose is to
*recall* accumulated experience — when there is none, it says so honestly
rather than fabricating priors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import (
    DEFAULT_COST_WEIGHTS,
    COST_DIMENSIONS,
    CostVector,
    ProblemProfile,
    StrategicEntry,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats
from or_harness.strategy.strategic_bank import StrategicBank

MEMORY_MODES = ("none", "cases", "strategic", "cost-aware")


@dataclass
class Recommendation:
    """One recalled memory about a strategy in this structural cell.

    ``strategy_id`` is an IDENTIFIER, not a menu entry: the framework never
    supplies a name, a description, a fallback chain or an action list for
    it. Whatever content travels with this recommendation came from the
    memory that backs it (see ``knowledge``) or from the harness that wrote
    the entry — never from a built-in directory.

    There is NO utility ``score``: the expected values below are REPORTS
    (a recount over real executions, or an entry's own estimate), and
    whether to act on them is the agent's decision after the world model
    predicts the candidate's consequences.
    """

    strategy_id: str
    expected_quality: float
    expected_cost: CostVector
    failure_prob: float
    evidence: str  # "strategic_entry" | "conditional_stats"
    evidence_refs: List[str] = field(default_factory=list)  # entry/execution ids
    confidence: float = 1.0
    cross_family: bool = False
    risk_warnings: List[str] = field(default_factory=list)
    basis: str = ""
    #: Dimensions of ``expected_cost`` that are actually measured (never
    #: treat an unmeasured placeholder zero as evidence of cheapness).
    cost_known_dims: List[str] = field(default_factory=list)
    #: Dimensions used for this recall's cost reporting — the common
    #: measured dimensions across cost-evidenced candidates. Empty = cost
    #: not comparable this recall (missing data never auto-benefits).
    cost_basis_dims: List[str] = field(default_factory=list)
    #: The entry's stated knowledge CLAIM (see ``core.schema.validate_claim``),
    #: when the backing entry states one. Its verdict is the entry's OWN
    #: ``verification`` block, reported under ``knowledge``.
    claim: Optional[Dict[str, Any]] = None
    #: What the backing ENTRY itself says about the method: its own
    #: applicability notes, its own recorded actions, its strategy_type and
    #: fallback, and its support. Present only on the ``strategic_entry``
    #: path, and only when the entry really carries the content — an absent
    #: field means the memory does not record it, which is stated rather
    #: than filled in from somewhere else.
    knowledge: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "expected": {
                "quality": round(self.expected_quality, 4),
                "cost": {d: round(v, 4) for d, v in self.expected_cost.to_dict().items()},
                "failure_prob": round(self.failure_prob, 4),
            },
            "evidence": self.evidence,
            "evidence_refs": list(self.evidence_refs),
            "confidence": round(self.confidence, 4),
            "cross_family": self.cross_family,
            "risk_warnings": list(self.risk_warnings),
            "basis": self.basis,
            "cost_known_dims": list(self.cost_known_dims),
            "cost_basis_dims": list(self.cost_basis_dims),
            "claim": (copy.deepcopy(self.claim)
                      if self.claim is not None else None),
            "knowledge": copy.deepcopy(self.knowledge),
        }


class Selector:
    def __init__(self, sbank: StrategicBank,
                 stats: ConditionalStats, *,
                 cost_weights: Optional[Dict[str, float]] = None):
        self.sbank = sbank
        self.stats = stats
        # Cost weights are retained ONLY for the experiments runner's cost
        # accounting and for reporting which dimensions are comparable;
        # recall applies NO cost scalarization to rank candidates.
        self.cost_weights = dict(cost_weights or DEFAULT_COST_WEIGHTS)

    # -- public ----------------------------------------------------------------

    def candidate_ids(self, profile: ProblemProfile, *,
                      memory_mode: str = "cost-aware",
                      include_unverified: bool = False) -> List[str]:
        """The strategies this cell has REAL memory about, sorted.

        The union of two evidence sources, both of which are facts about
        what really happened:

        - the strategy ids of this structural cell's aggregated evidence
          (executions that really ran and were recorded);
        - the strategy ids of matching strategic entries that are published
          (or of every matching entry, in the offline inspection view).

        A strategy nobody ever tried, and that no entry speaks about, is not
        a candidate — the framework has no menu to fall back on.
        """
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        if memory_mode == "none":
            return []
        ids = {sid for sid, cell in self.stats.for_profile(profile).items()
               if cell.n > 0}
        for entry in self._matching_entries(profile, memory_mode,
                                            include_unverified):
            ids.add(entry.strategy_id)
        return sorted(ids)

    def recall(self, profile: ProblemProfile, *, top: int = 3,
               exclude: Optional[Sequence[str]] = None,
               candidates: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
               include_unverified: bool = False,
               with_meta: bool = False
               ) -> Any:
        """Recall accumulated experience for this problem signature.

        Returns one row per strategy this cell has real memory about, in
        RELEVANCE order (a strategic entry before a bare recount; then
        strategy id). It does NOT rank by utility and computes no score:
        the ``expected`` values on each row are reports for the agent to
        weigh after the world model predicts a candidate's consequences.

        **No memory means an empty list** — there is no placeholder
        candidate and no zero-quality stand-in, because a fabricated row is
        indistinguishable downstream from a measured one.

        ``candidates`` is the CALLER's proposal set (the outer agent names
        the methods it is considering). Supplying it restricts the result to
        those ids and makes memory-less strategies simply absent (the caller
        can compare its own list against :meth:`candidate_ids`). It never
        creates evidence and never mutates the statistics.

        ``top`` is a MATERIAL BUDGET, not a utility cut: the number of rows
        the caller wants to READ. The cap is reported (``n_matched`` vs the
        rows returned) when ``with_meta=True``, so a small ``top`` can never
        look like "nothing else matched". The default ``with_meta=False``
        returns the plain list for existing callers.

        Only PUBLISHED knowledge is treated as strategic knowledge: an
        admission-verified entry, or a legacy entry from before admission
        verification existed. An entry whose verification explicitly failed
        (``refuted``) or could not be decided yet (``insufficient_evidence``)
        is NOT published: with ``include_unverified=False`` (the default) it
        contributes nothing and recall falls back to the raw conditional
        statistics. ``include_unverified=True`` is the offline/inspection
        view and returns it with an explicit warning.
        """
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        if memory_mode == "none":
            return ({"recall": [], "n_matched": 0, "returned": 0,
                     "material_budget": 0, "omitted": 0,
                     "note": "memory_mode 'none': no memory was consulted"}
                    if with_meta else [])
        excluded = set(exclude or [])
        proposed = (None if candidates is None
                    else {str(c) for c in candidates})
        entries = self._matching_entries(profile, memory_mode,
                                         include_unverified)
        cells = self.stats.for_profile(profile)
        ids = sorted(
            {sid for sid, cell in cells.items() if cell.n > 0}
            | {e.strategy_id for e in entries})
        if proposed is not None:
            ids = [sid for sid in ids if sid in proposed]
        ids = [sid for sid in ids if sid not in excluded]
        # Cost evidence is REPORTED (which dimensions are comparable), not
        # scalarized into a ranking. The basis is the common measured dims
        # across candidates that carry cost evidence; missing data never
        # auto-benefits — unshared dimensions are dropped for everyone.
        cost_basis = self._cost_basis_dims(ids, entries, cells)
        consulted: List[str] = []
        recs: List[Recommendation] = []
        for strategy_id in ids:
            for rec in self._collect(strategy_id, profile, entries, cells,
                                     memory_mode, cost_basis):
                consulted.extend(r for r in rec.evidence_refs
                                 if r.startswith("se_"))
                recs.append(rec)
        if consulted:
            self.sbank.mark_consulted(sorted(set(consulted)))
        # RELEVANCE order: a published entry (a commitment) before a bare
        # recount; ties by strategy id then first evidence ref. NEVER a
        # utility scalar — that lives in the world model's consequence
        # prediction and the agent's choice.
        rank = {"strategic_entry": 0, "conditional_stats": 1}
        recs.sort(key=lambda r: (rank.get(r.evidence, 2), r.strategy_id,
                                 r.evidence_refs[:1]))
        budget = max(1, int(top))
        shown = recs[:budget]
        if not with_meta:
            return shown
        return {
            "recall": shown,
            "n_matched": len(recs),
            "returned": len(shown),
            "material_budget": budget,
            "omitted": len(recs) - len(shown),
            "note": ("recall DISCOVERS material in relevance order; it does "
                     "NOT rank by utility. A `top` cap is a material budget, "
                     "and any omission is reported — it is not a decision "
                     "about which candidate is best"),
        }

    def _matching_entries(self, profile: ProblemProfile, memory_mode: str,
                          include_unverified: bool) -> List[StrategicEntry]:
        """The entries this recall may present, per mode and publication."""
        if memory_mode not in ("strategic", "cost-aware"):
            return []
        matched = self.sbank.matching(profile)
        if include_unverified:
            return matched
        return [e for e in matched if is_publishable(e)]

    # -- internals ---------------------------------------------------------------

    def _cost_basis_dims(self, candidates, entries, cells
                         ) -> Optional[List[str]]:
        """Common measured cost dimensions across cost-evidenced candidates.

        None = no candidate carries cost evidence at all (nothing to
        compare); a list (possibly empty) = the shared dimensions. Empty
        means cost is NOT comparable this recall.

        EVERY matching entry counts, not just the first per id: two claims
        that happen to share a strategy id are two pieces of evidence, and
        dropping one here would silently change what the recall compared.
        """
        known_sets = []
        for strategy_id in candidates:
            matched = [e for e in entries if e.strategy_id == strategy_id]
            if matched:
                known_sets.extend(
                    e.expected_cost_hat.measured_dims() for e in matched)
            cell = cells.get(strategy_id)
            if not matched and cell is not None and cell.n > 0:
                known_sets.append(cell.mean_cost.measured_dims())
        if not known_sets:
            return None
        common = set(known_sets[0])
        for dims in known_sets[1:]:
            common &= dims
        return sorted(common)

    def _collect(self, strategy_id: str, profile: ProblemProfile,
                 entries: List[StrategicEntry],
                 cells: Dict[str, GroupStats],
                 memory_mode: str,
                 cost_basis: Optional[List[str]] = None,
                 ) -> List[Recommendation]:
        """Collect the memory rows for one strategy: zero, one, or several.

        The ladder has exactly two rungs and no fallback: published entries
        for this cell, else this cell's conditional statistics. A strategy
        with neither yields NOTHING — an empty result is the honest answer,
        and a placeholder row would be indistinguishable from a measurement.

        MULTIPLE ENTRIES UNDER ONE ID produce one row EACH. Two claims that
        share a strategy id are still two claims (different predicates,
        different estimates); collapsing them into one row would silently
        discard one and present the other as "the" memory about that id.
        """
        matched = [e for e in entries if e.strategy_id == strategy_id]
        if matched and memory_mode in ("strategic", "cost-aware"):
            return [self._from_entry(strategy_id, profile, entry,
                                     cost_basis)
                    for entry in matched]
        cell = cells.get(strategy_id)
        if (cell is not None and cell.n > 0
                and memory_mode in ("cases", "strategic", "cost-aware")):
            return [self._from_stats(strategy_id, cell, cost_basis=cost_basis)]
        return []

    def _entry_confidence(self, entry: StrategicEntry, profile: ProblemProfile) -> float:
        """How much support this entry's estimate rests on, as a plain count.

        A PURE REPORT of support (n/5, capped at 1), NOT a discount applied
        to a score: there is no score. Cross-family application is LABELLED
        elsewhere and is not folded into this number.
        """
        return min(1.0, entry.support_n / PROMOTE_REFERENCE_N)

    @staticmethod
    def _is_cross_family(entry: StrategicEntry,
                         profile: ProblemProfile) -> bool:
        """True when the claim does not itself name this family — i.e. it is
        being applied outside the family it was induced from. Entries produced
        by induction carry a family predicate, so this only bites on
        harness-authored family-free patterns."""
        return "family" not in entry.predicates and bool(entry.predicates)

    def _from_entry(self, strategy_id: str, profile: ProblemProfile,
                    entry: StrategicEntry,
                    cost_basis: Optional[List[str]] = None
                    ) -> Recommendation:
        """A recalled row from a Strategic Knowledge entry.

        An entry that is not publishable is only reachable through the
        explicit offline view (``include_unverified=True``): the framework
        HOLDS that claim, it has not admitted it as knowledge. The row
        carries an explicit warning so the inspection view stays useful
        without letting the caller mistake an unchecked claim for a verified
        one.

        The content block is read off the ENTRY (its own notes, its own
        recorded actions, its own strategy_type/fallback, its own support) —
        the framework does not fill any of it in from a directory.
        """
        confidence = self._entry_confidence(entry, profile)
        cross_family = self._is_cross_family(entry, profile)
        cost = entry.expected_cost_hat
        warnings: List[str] = []
        if not is_publishable(entry):
            warnings.append(
                f"UNPUBLISHED candidate {entry.entry_id} "
                f"(verification={entry.verification_state}): the framework "
                "holds this claim, it has not been admitted as knowledge — "
                "its estimates support inspection, not a decision")
        if entry.status == "suspect":
            warnings.append(
                f"entry {entry.entry_id} is suspect (3 consecutive prediction "
                "misses); its estimate is unreliable until re-verified")
        if cross_family:
            warnings.append(
                f"cross-family generalization: entry {entry.entry_id} states no "
                "family predicate, so it is applied outside the family it "
                "was induced from")
        warnings.extend(self._cost_basis_warnings(cost_basis))
        warnings.extend(entry.risk_conditions)
        return Recommendation(
            strategy_id=strategy_id,
            expected_quality=entry.expected_quality_hat,
            expected_cost=cost, failure_prob=entry.failure_prob,
            evidence="strategic_entry", evidence_refs=[entry.entry_id],
            confidence=confidence, cross_family=cross_family,
            risk_warnings=warnings,
            basis=f"entry {entry.entry_id} ({entry.status}, "
                  f"hit_rate={entry.prediction_track.hit_rate:.2f}, "
                  f"n={entry.prediction_track.n_predictions})",
            cost_known_dims=sorted(cost.measured_dims()),
            cost_basis_dims=list(cost_basis or []),
            claim=(copy.deepcopy(entry.claim)
                   if entry.claim is not None else None),
            knowledge=self._entry_knowledge(entry))

    @staticmethod
    def _entry_knowledge(entry: StrategicEntry) -> Dict[str, Any]:
        """What the backing entry itself records about the method.

        Every field is read from the entry; a key is present only when the
        entry really carries content for it. Nothing is inferred from the
        strategy id — two different methods that happen to share an id are
        NOT merged here (see the entries' own predicates/notes for the
        distinction), and an entry that never recorded its actions reports
        none rather than being filled in from elsewhere.
        """
        from or_harness.core.schema import claim_evidence_refs, \
            claim_scope_tasks
        has_claim = entry.claim is not None
        evidence_refs = claim_evidence_refs(entry) if has_claim else []
        # ESTIMATE HONESTY: a claim-only entry (a claim with no statistical
        # support) carries the DEFAULT ``support_n=0`` / ``expected_quality_
        # hat=0.0``. Read without care those look like "measured zero", which
        # is exactly the misreading this block prevents: it says whether a
        # statistical estimate EXISTS at all, and separates the counts a
        # reader must not conflate — the executions a claim CITES, the
        # INDEPENDENT TASKS those cover, and the STATISTICAL support n. A
        # STATISTICAL entry (support_n>0, no claim) has no claim-cited
        # executions: its evidence is its support_n samples, so the claim-only
        # counts are reported as None (not zero, which would read as "no
        # evidence").
        out: Dict[str, Any] = {
            "entry_id": entry.entry_id,
            "strategy_type": entry.strategy_type,
            "support_n": entry.support_n,
            "verification_state": entry.verification_state,
            "status": entry.status,
            "verification": copy.deepcopy(entry.verification or {}),
            "claim": (copy.deepcopy(entry.claim)
                      if entry.claim is not None else None),
            # The entry's own applicability predicates: part of what the
            # knowledge SAYS (which conditions it claims to apply under).
            "predicates": copy.deepcopy(entry.predicates or {}),
            "applicability": [str(n) for n in (entry.applicability or [])],
            "actions": [str(a) for a in (entry.actions or [])],
            "fallback_strategy_id": entry.fallback_strategy_id,
            # The counts kept APART, each reported ONLY when it exists.
            # ``evidence_executions`` / ``distinct_tasks`` are claim-only
            # (None for a statistical entry — its evidence is support_n, not
            # cited executions). ``statistical_support_n`` is the sample the
            # quality/cost estimate rests on (0 = NO estimate).
            "estimate": {
                "estimated": entry.support_n > 0,
                "statistical_support_n": entry.support_n,
                "evidence_executions": (len(evidence_refs) if has_claim
                                        else None),
                "distinct_tasks": (len(claim_scope_tasks(entry)) if has_claim
                                   else None),
                "quality_hat": (entry.expected_quality_hat
                                if entry.support_n > 0 else None),
                "failure_prob": (entry.failure_prob
                                 if entry.support_n > 0 else None),
                "note": ("a statistical estimate backs this entry"
                         if entry.support_n > 0 else
                         "NO statistical estimate: this is a claim-only entry "
                         "(support_n=0 is 'not estimated', NOT 'measured 0'); "
                         "its quality/cost fields are absent, and only its "
                         "verification scope supports it"),
            },
            "note": ("content read from the entry that backs this "
                     "recommendation; empty lists mean the memory does not "
                     "record it"),
        }
        return out

    @staticmethod
    def _cost_basis_warnings(cost_basis: Optional[List[str]]) -> List[str]:
        if cost_basis == []:
            return ["cost not comparable across candidates: no common "
                    "measured cost dimension (missing data does not count "
                    "as cheap, and no cost ranking is made here)"]
        return []

    def _from_stats(self, strategy_id: str, cell: GroupStats,
                    cost_basis: Optional[List[str]] = None) -> Recommendation:
        """A recalled row from conditional statistics over the Evidence Bank.

        This path is a RECOUNT of observations (mean quality/cost actually
        observed in this structural group), not a Strategic Knowledge
        commitment: it carries no prediction interval, no calibration track,
        and no lifecycle. The Recommendation fields are named
        ``expected_*`` for API stability, but here they report observed
        means, and ``evidence`` is labelled ``conditional_stats`` so the two
        layers stay distinguishable downstream.
        """
        cost = cell.mean_cost
        warnings = []
        if cell.n < 2:
            warnings.append(f"single observation for {strategy_id} in "
                            "this structural group; treat as weak evidence")
        warnings.extend(self._cost_basis_warnings(cost_basis))
        return Recommendation(
            strategy_id=strategy_id,
            expected_quality=cell.mean_quality, expected_cost=cost,
            failure_prob=cell.fail_rate, evidence="conditional_stats",
            evidence_refs=list(cell.execution_ids),
            confidence=(min(1.0, cell.n / PROMOTE_REFERENCE_N)),
            cross_family=False,
            risk_warnings=warnings,
            basis=(f"conditional statistics over n={cell.n} executions in this group"),
            cost_known_dims=sorted(cost.measured_dims()),
            cost_basis_dims=list(cost_basis or []))


#: support_n at which an entry reaches full confidence.
PROMOTE_REFERENCE_N = 5.0


def is_publishable(entry: StrategicEntry) -> bool:
    """Whether an entry may be presented as published strategic knowledge.

    - ``verified``: yes — the claim passed its admission check.
    - NO verification block at all: yes — this is a legacy entry written
      before admission verification existed. Refusing to use it would
      silently discard accumulated knowledge; it stays usable and its missing
      verification stays visible in ``inspect``.
    - anything else (``unverified`` candidate, ``insufficient_evidence``,
      ``refuted``): no — the framework holds a candidate, not knowledge.
      Recall falls back to the raw conditional statistics, and the harness
      can still try the strategy.
    - ``stale_after_revision``: no — the entry's claim was SUBSTANTIVELY
      revised (predicates or expected estimates moved) without a fresh
      admission verdict, so the old verification no longer covers the new
      claim. Re-verify with ``induce --verify`` to re-publish.
    """
    block = entry.verification or {}
    if not block:
        return True
    if block.get("stale_after_revision"):
        return False
    return block.get("state") == "verified"
