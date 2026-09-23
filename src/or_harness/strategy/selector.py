"""Strategy selector: recall of REAL memory, transparent scoring.

Score = alpha * Q_hat - beta * C_scalar - gamma * R_hat

**There is no candidate menu.** The candidate set is derived from memory
alone: a strategy is a candidate because something was really executed
(conditional statistics) or really induced (a strategic entry) for this
structural cell. When nothing matches, recall returns an EMPTY list — not a
placeholder row, not a `-inf` score, not a zero quality. A candidate list the
caller proposes is scored against those same real memories; a proposed
strategy with no memory is reported as such and carries no score.

Evidence precedence (per strategy):
  1. Strategic entries whose pattern matches the profile (commitments).
  2. Conditional statistics over the Experience Bank (recounts) — this single
     path subsumes what older designs split into "case retrieval" and
     "statistics": both are the same data used two ways.

Ablation modes (--memory-mode), reused by the experiments runner:
  A none      — no memory consulted at all: nothing is recalled.
  B cases     — case-based evidence only (no entries, no cost weighting).
  C strategic — entries/stats condition the choice, WITHOUT cost weighting.
  D cost-aware— C + cost scalarization. C vs D only separates in
                "quality tied, cost divergent" scenarios — the experimental
                support for 'cost awareness is a necessary part of memory'.

Cross-family generalization (a family-free entry pattern matching a family it
has no provenance in) carries an explicit confidence discount and is
labelled.

Note: the method is named ``recall`` (not ``recommend``) because its purpose
is to *recall* accumulated experience — when there is none, it says so
honestly rather than fabricating priors.
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
from or_harness.strategy.strategic_bank import SUSPECT_SCORE_FACTOR, StrategicBank

MEMORY_MODES = ("none", "cases", "strategic", "cost-aware")
DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA = 1.0, 1.0, 1.0
CROSS_FAMILY_CONFIDENCE_DISCOUNT = 0.6
NEW_ENTRY_CONFIDENCE_FLOOR = 0.35  # confidence scales with support: n/5, floored


@dataclass
class Recommendation:
    """One recalled memory about a strategy in this structural cell.

    ``strategy_id`` is an IDENTIFIER, not a menu entry: the framework never
    supplies a name, a description, a fallback chain or an action list for
    it. Whatever content travels with this recommendation came from the
    memory that backs it (see ``knowledge``) or from the harness that wrote
    the entry — never from a built-in directory.
    """

    strategy_id: str
    score: float
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
    #: Dimensions used for this recall's cost scalarization — the common
    #: measured dimensions across cost-evidenced candidates. Empty = cost
    #: not comparable this recall (missing data never auto-benefits).
    cost_basis_dims: List[str] = field(default_factory=list)
    #: STRUCTURED relation claims carried by the backing entry (see
    #: ``core.schema.validate_relation``). Each carries its own
    #: ``verification`` state; a relation is published on its OWN verdict,
    #: independent of the entry's statistical admission.
    relations: List[Dict[str, Any]] = field(default_factory=list)
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
            "score": round(self.score, 4),
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
            "relations": [dict(r) for r in self.relations],
            "knowledge": copy.deepcopy(self.knowledge),
        }


class Selector:
    def __init__(self, sbank: StrategicBank,
                 stats: ConditionalStats, *,
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                 gamma: float = DEFAULT_GAMMA,
                 cost_weights: Optional[Dict[str, float]] = None):
        self.sbank = sbank
        self.stats = stats
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
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
               include_unverified: bool = False
               ) -> List[Recommendation]:
        """Recall accumulated experience for this problem signature.

        Returns one entry per strategy this cell has real memory about. Each
        carries an ``evidence`` field: ``strategic_entry`` (a commitment) or
        ``conditional_stats`` (a recount). **No memory means an empty
        list** — there is no placeholder candidate, no ``-inf`` score and no
        zero-quality stand-in, because a fabricated row is indistinguishable
        downstream from a measured one.

        ``candidates`` is the CALLER's proposal set (the outer agent names
        the methods it is considering). Supplying it does two things and
        nothing else: it restricts the result to those ids, and it makes the
        strategies that have no memory in this cell simply absent from the
        result (the caller can compare its own list against
        :meth:`candidate_ids`). It never creates evidence, never ranks a
        memory-less strategy, and never mutates the statistics.

        Only PUBLISHED knowledge is treated as strategic knowledge: an
        admission-verified entry, or a legacy entry from before admission
        verification existed (its provenance cannot be re-litigated, and
        silently discarding accumulated knowledge would be worse than
        labelling it). An entry whose verification explicitly failed
        (``refuted``) or could not be decided yet (``insufficient_evidence``)
        is NOT published: with ``include_unverified=False`` (the default) it
        contributes nothing and recall falls back to the raw conditional
        statistics. ``include_unverified=True`` is the offline/inspection
        view and returns it with an explicit warning.

        Unverified entries are still RECORDED as checks (a fact about the
        claim is worth keeping) — they are just not published.
        """
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        if memory_mode == "none":
            return []
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
        if memory_mode == "cost-aware":
            # Comparable cost dimensions: the common measured dims across
            # candidates that actually carry cost evidence. Missing data
            # never auto-benefits — unshared dimensions are dropped for
            # everyone, and no common dimension means cost is not
            # comparable (cost term set to zero with an explicit warning).
            cost_basis = self._cost_basis_dims(ids, entries, cells)
            norms = self._cost_norms(ids, entries, cells, cost_basis) \
                if cost_basis else {}
        else:
            cost_basis = None
            norms = {}
        consulted: List[str] = []
        recs: List[Recommendation] = []
        for strategy_id in ids:
            for rec in self._score(strategy_id, profile, entries, cells,
                                   memory_mode, norms, cost_basis):
                consulted.extend(r for r in rec.evidence_refs
                                 if r.startswith("se_"))
                recs.append(rec)
        if consulted:
            self.sbank.mark_consulted(sorted(set(consulted)))
        recs.sort(key=lambda r: (-r.score, r.strategy_id,
                                 r.evidence_refs[:1]))
        return recs[: max(1, top)]

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

    def _cost_norms(self, candidates, entries, cells,
                    dims: Optional[List[str]]) -> Dict[str, float]:
        """Per-dimension normalization divisors from the current candidate
        cost range, restricted to the comparable dimensions, so no raw unit
        (e.g. thousands of tokens) can swamp the quality term and no
        unmeasured dimension can distort normalization."""
        vectors: List[CostVector] = []
        for strategy_id in candidates:
            matched = [e for e in entries if e.strategy_id == strategy_id]
            vectors.extend(e.expected_cost_hat for e in matched)
            cell = cells.get(strategy_id)
            if not matched and cell is not None and cell.n > 0:
                vectors.append(cell.mean_cost)
            # No prior fallback — if no evidence, no vector contributes.
        norms: Dict[str, float] = {}
        for d in (dims or []):
            peak = max((getattr(v, d) for v in vectors), default=0.0) if vectors else 0.0
            norms[d] = float(peak) if peak > 0 else 1.0
        return norms

    def _score(self, strategy_id: str, profile: ProblemProfile,
               entries: List[StrategicEntry],
               cells: Dict[str, GroupStats],
               memory_mode: str,
               norms: Optional[Dict[str, float]] = None,
               cost_basis: Optional[List[str]] = None,
               ) -> List[Recommendation]:
        """Score one strategy from memory: zero, one, or several results.

        The ladder has exactly two rungs and no fallback: published entries
        for this cell, else this cell's conditional statistics. A strategy
        with neither yields NOTHING — an empty result is the honest answer,
        and a placeholder row would be indistinguishable from a measurement.

        MULTIPLE ENTRIES UNDER ONE ID produce one recommendation EACH. Two
        claims that share a strategy id are still two claims (different
        predicates, different estimates); collapsing them into one row would
        silently discard one and present the other as "the" memory about that
        id — which is exactly the mixing the design forbids.
        """
        matched = [e for e in entries if e.strategy_id == strategy_id]
        if matched and memory_mode in ("strategic", "cost-aware"):
            return [self._from_entry(strategy_id, profile, entry, memory_mode,
                                     norms, cost_basis)
                    for entry in matched]
        cell = cells.get(strategy_id)
        if (cell is not None and cell.n > 0
                and memory_mode in ("cases", "strategic", "cost-aware")):
            return [self._from_stats(strategy_id, cell, memory_mode,
                                     norms, cost_basis=cost_basis)]
        return []

    def _entry_confidence(self, entry: StrategicEntry, profile: ProblemProfile) -> float:
        # Confidence scales with support (new entries get a grace floor), and
        # cross-family generalization is explicitly discounted.
        conf = max(NEW_ENTRY_CONFIDENCE_FLOOR,
                   min(1.0, entry.support_n / PROMOTE_REFERENCE_N))
        if self._is_cross_family(entry, profile):
            conf *= CROSS_FAMILY_CONFIDENCE_DISCOUNT
        return conf

    @staticmethod
    def _is_cross_family(entry: StrategicEntry,
                         profile: ProblemProfile) -> bool:
        """True when the claim does not itself name this family — i.e. it is
        being applied outside the family it was induced from. Entries produced
        by induction carry a family predicate, so this only bites on
        harness-authored family-free patterns."""
        return "family" not in entry.predicates and bool(entry.predicates)

    def _from_entry(self, strategy_id: str, profile: ProblemProfile,
                    entry: StrategicEntry, memory_mode: str,
                    norms: Optional[Dict[str, float]] = None,
                    cost_basis: Optional[List[str]] = None
                    ) -> Recommendation:
        """Recommendation from a Strategic Knowledge entry.

        An entry that is not publishable is only reachable through the
        explicit offline view (``include_unverified=True``): the framework
        HOLDS that claim, it has not admitted it as knowledge. Scoring it the
        same way while attaching an explicit warning keeps the inspection
        view useful without letting the caller mistake an unchecked claim for
        a verified one.

        The content block is read off the ENTRY (its own notes, its own
        recorded actions, its own strategy_type/fallback, its own support) —
        the framework does not fill any of it in from a directory.
        """
        confidence = self._entry_confidence(entry, profile)
        cross_family = self._is_cross_family(entry, profile)
        cost = entry.expected_cost_hat
        cost_term = self._cost_term(cost, memory_mode, norms, cost_basis)
        score = (self.alpha * entry.expected_quality_hat
                 - self.beta * cost_term
                 - self.gamma * entry.failure_prob)
        warnings: List[str] = []
        if not is_publishable(entry):
            warnings.append(
                f"UNPUBLISHED candidate {entry.entry_id} "
                f"(verification={entry.verification_state}): the framework "
                "holds this claim, it has not been admitted as knowledge — "
                "its estimates support inspection, not a decision")
        if entry.status == "suspect":
            score *= SUSPECT_SCORE_FACTOR
            warnings.append(
                f"entry {entry.entry_id} is suspect (3 consecutive prediction "
                "misses); its estimate is downweighted x0.5")
        if cross_family:
            warnings.append(
                f"cross-family generalization: entry {entry.entry_id} states no "
                f"family predicate; confidence discounted "
                f"x{CROSS_FAMILY_CONFIDENCE_DISCOUNT}")
        warnings.extend(self._cost_basis_warnings(cost_basis))
        warnings.extend(entry.risk_conditions)
        return Recommendation(
            strategy_id=strategy_id, score=score,
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
            relations=[dict(r) for r in (entry.relations or [])],
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
        out: Dict[str, Any] = {
            "entry_id": entry.entry_id,
            "strategy_type": entry.strategy_type,
            "support_n": entry.support_n,
            "verification_state": entry.verification_state,
            "status": entry.status,
            "applicability": [str(n) for n in (entry.applicability or [])],
            "actions": [str(a) for a in (entry.actions or [])],
            "fallback_strategy_id": entry.fallback_strategy_id,
            "note": ("content read from the entry that backs this "
                     "recommendation; empty lists mean the memory does not "
                     "record it"),
        }
        return out

    def _cost_term(self, cost: CostVector, memory_mode: str,
                   norms: Optional[Dict[str, float]],
                   cost_basis: Optional[List[str]]) -> float:
        """Scalarized cost term restricted to the comparable dimensions.

        ``cost_basis=None``: no cost comparison performed this recall (or
        non-cost-aware mode) — term is zero. ``cost_basis=[]``: candidates
        carry cost evidence but share NO measured dimension — cost is not
        comparable, term is zero (missing data never auto-benefits).
        Otherwise the term uses only the shared dimensions' weights.
        """
        if memory_mode != "cost-aware" or cost_basis is None or not cost_basis:
            return 0.0
        weights = {d: self.cost_weights.get(d, 0.0) for d in cost_basis}
        return cost.scalarize(weights, norms)

    @staticmethod
    def _cost_basis_warnings(cost_basis: Optional[List[str]]) -> List[str]:
        if cost_basis == []:
            return ["cost not comparable across candidates: no common "
                    "measured cost dimension; cost term set to zero (missing "
                    "data does not count as cheap)"]
        return []

    def _from_stats(self, strategy_id: str, cell: GroupStats,
                    memory_mode: str,
                    norms: Optional[Dict[str, float]] = None,
                    basis: Optional[str] = None,
                    cross_family: bool = False,
                    cost_basis: Optional[List[str]] = None) -> Recommendation:
        """Recommendation from conditional statistics over the Evidence Bank.

        This path is a RECOUNT of observations (mean quality/cost actually
        observed in this structural group), not a Strategic Knowledge
        commitment: it carries no prediction interval, no calibration track,
        and no lifecycle. The Recommendation fields are named
        ``expected_*`` for API stability, but here they report observed
        means, and ``evidence`` is labelled ``conditional_stats`` so the two
        layers stay distinguishable downstream.
        """
        cost = cell.mean_cost
        cost_term = self._cost_term(cost, memory_mode, norms, cost_basis)
        score = (self.alpha * cell.mean_quality
                 - self.beta * cost_term
                 - self.gamma * cell.fail_rate)
        warnings = []
        if cell.n < 2:
            warnings.append(f"single observation for {strategy_id} in "
                            "this structural group; treat as weak evidence")
        warnings.extend(self._cost_basis_warnings(cost_basis))
        return Recommendation(
            strategy_id=strategy_id, score=score,
            expected_quality=cell.mean_quality, expected_cost=cost,
            failure_prob=cell.fail_rate, evidence="conditional_stats",
            evidence_refs=list(cell.execution_ids),
            confidence=(min(1.0, cell.n / PROMOTE_REFERENCE_N)
                        * (CROSS_FAMILY_CONFIDENCE_DISCOUNT if cross_family else 1.0)),
            cross_family=cross_family,
            risk_warnings=warnings,
            basis=basis or f"conditional statistics over n={cell.n} executions in this group",
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
