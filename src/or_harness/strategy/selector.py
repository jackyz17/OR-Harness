"""Strategy selector: two-layer fallback, transparent scoring, ablation modes.

Score = alpha * Q_hat - beta * C_scalar - gamma * R_hat

Evidence precedence (per strategy):
  1. Strategic entries whose pattern matches the profile (commitments).
  2. Conditional statistics over the Experience Bank (recounts) — this single
     path subsumes what older designs split into "case retrieval" and
     "statistics": both are the same data used two ways.
  3. No evidence — the catalog is a structural vocabulary (applicability,
     actions, fallback, solver_family) carrying NO prior quality/cost/risk
     scores. Without experience the selector honestly reports evidence=
     "no_memory" with score 0 and confidence 0.

Ablation modes (--memory-mode), reused by the experiments runner:
  A none      — default strategy, no memory.
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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import (
    DEFAULT_COST_WEIGHTS,
    COST_DIMENSIONS,
    CostVector,
    ProblemProfile,
    StrategicEntry,
    Strategy,
)
from or_harness.strategy.stats import ConditionalStats, GroupStats
from or_harness.strategy.strategic_bank import SUSPECT_SCORE_FACTOR, StrategicBank

MEMORY_MODES = ("none", "cases", "strategic", "cost-aware")
DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA = 1.0, 1.0, 1.0
CROSS_FAMILY_CONFIDENCE_DISCOUNT = 0.6
NEW_ENTRY_CONFIDENCE_FLOOR = 0.35  # confidence scales with support: n/5, floored


@dataclass
class Recommendation:
    strategy: Strategy
    score: float
    expected_quality: float
    expected_cost: CostVector
    failure_prob: float
    evidence: str  # "strategic_entry" | "conditional_stats" | "no_memory"
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy.strategy_id,
            "name": self.strategy.name,
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
        }


class Selector:
    def __init__(self, catalog: Dict[str, Strategy], sbank: StrategicBank,
                 stats: ConditionalStats, *,
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                 gamma: float = DEFAULT_GAMMA,
                 cost_weights: Optional[Dict[str, float]] = None):
        self.catalog = catalog
        self.sbank = sbank
        self.stats = stats
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.cost_weights = dict(cost_weights or DEFAULT_COST_WEIGHTS)

    # -- public ----------------------------------------------------------------

    def recall(self, profile: ProblemProfile, *, top: int = 3,
               exclude: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
               include_unverified: bool = False) -> List[Recommendation]:
        """Recall accumulated experience for this problem signature.

        Returns candidates filtered by applicability. Each candidate carries
        an ``evidence`` field: ``strategic_entry``, ``conditional_stats``, or
        ``no_memory``. When no experience exists the score is -inf and
        confidence is 0 — the catalog vocabulary is still returned as a
        candidate menu, but no quality/cost/risk claims are made.

        Only PUBLISHED knowledge is treated as strategic knowledge: an
        admission-verified entry, or a legacy entry from before admission
        verification existed (its provenance cannot be re-litigated, and
        silently discarding accumulated knowledge would be worse than
        labelling it). An entry whose verification explicitly failed
        (``refuted``) or could not be decided yet (``insufficient_evidence``)
        is NOT published: with ``include_unverified=False`` (the default) it
        contributes nothing and recall falls back to the raw conditional
        statistics or the catalog vocabulary. ``include_unverified=True`` is
        the offline/inspection view and returns it with an explicit warning.

        Unverified entries are still RECORDED as checks (a fact about the
        claim is worth keeping) — they are just not published.
        """
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        excluded = set(exclude or [])
        candidates = [s for s in self.catalog.values()
                      if s.strategy_id not in excluded
                      and self._applies(s, profile)]
        if memory_mode == "none":
            return [self._from_no_evidence(s) for s in candidates]

        all_entries = (self.sbank.matching(profile)
                       if memory_mode in ("strategic", "cost-aware") else [])
        if include_unverified:
            entries = all_entries
        else:
            entries = [e for e in all_entries if is_publishable(e)]
        cells = self.stats.for_profile(profile)
        if memory_mode == "cost-aware":
            # Comparable cost dimensions: the common measured dims across
            # candidates that actually carry cost evidence. Missing data
            # never auto-benefits — unshared dimensions are dropped for
            # everyone, and no common dimension means cost is not
            # comparable (cost term set to zero with an explicit warning).
            cost_basis = self._cost_basis_dims(candidates, entries, cells)
            norms = self._cost_norms(candidates, entries, cells, cost_basis) \
                if cost_basis else {}
        else:
            cost_basis = None
            norms = {}
        consulted: List[str] = []
        recs: List[Recommendation] = []
        for strategy in candidates:
            rec = self._score(strategy, profile, entries, cells, memory_mode,
                              norms, cost_basis)
            consulted.extend(r for r in rec.evidence_refs if r.startswith("se_"))
            recs.append(rec)
        if consulted:
            self.sbank.mark_consulted(sorted(set(consulted)))
        recs.sort(key=lambda r: (-r.score, r.strategy.strategy_id))
        return recs[: max(1, top)]

    # -- internals ---------------------------------------------------------------

    def _cost_basis_dims(self, candidates, entries, cells
                         ) -> Optional[List[str]]:
        """Common measured cost dimensions across cost-evidenced candidates.

        None = no candidate carries cost evidence at all (nothing to
        compare); a list (possibly empty) = the shared dimensions. Empty
        means cost is NOT comparable this recall.
        """
        known_sets = []
        for strategy in candidates:
            entry = next((e for e in entries
                          if e.strategy_id == strategy.strategy_id), None)
            cell = cells.get(strategy.strategy_id)
            if entry is not None:
                known_sets.append(entry.expected_cost_hat.measured_dims())
            elif cell is not None and cell.n > 0:
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
        for strategy in candidates:
            entry = next((e for e in entries if e.strategy_id == strategy.strategy_id), None)
            cell = cells.get(strategy.strategy_id)
            if entry is not None:
                vectors.append(entry.expected_cost_hat)
            elif cell is not None and cell.n > 0:
                vectors.append(cell.mean_cost)
            # No prior fallback — if no evidence, no vector contributes.
        norms: Dict[str, float] = {}
        for d in (dims or []):
            peak = max((getattr(v, d) for v in vectors), default=0.0) if vectors else 0.0
            norms[d] = float(peak) if peak > 0 else 1.0
        return norms

    @staticmethod
    def _applies(strategy: Strategy, profile: ProblemProfile) -> bool:
        from or_harness.core.schema import profile_matches
        return profile_matches(profile, strategy.applicability) if strategy.applicability else True

    def _score(self, strategy: Strategy, profile: ProblemProfile,
               entries: List[StrategicEntry],
               cells: Dict[str, GroupStats],
               memory_mode: str,
               norms: Optional[Dict[str, float]] = None,
               cost_basis: Optional[List[str]] = None,
               ) -> Recommendation:
        entry = next((e for e in entries if e.strategy_id == strategy.strategy_id), None)
        cell = cells.get(strategy.strategy_id)
        if entry is not None and memory_mode in ("strategic", "cost-aware"):
            return self._from_entry(strategy, profile, entry, memory_mode,
                                    norms, cost_basis)
        if cell is not None and cell.n > 0 and memory_mode in ("cases", "strategic", "cost-aware"):
            return self._from_stats(strategy, cell, memory_mode,
                                    norms, cost_basis=cost_basis)
        return self._from_no_evidence(strategy, norms=norms)
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

    def _from_entry(self, strategy: Strategy, profile: ProblemProfile,
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
            strategy=strategy, score=score,
            expected_quality=entry.expected_quality_hat,
            expected_cost=cost, failure_prob=entry.failure_prob,
            evidence="strategic_entry", evidence_refs=[entry.entry_id],
            confidence=confidence, cross_family=cross_family,
            risk_warnings=warnings,
            basis=f"entry {entry.entry_id} ({entry.status}, "
                  f"hit_rate={entry.prediction_track.hit_rate:.2f}, "
                  f"n={entry.prediction_track.n_predictions})",
            cost_known_dims=sorted(cost.measured_dims()),
            cost_basis_dims=list(cost_basis or []))

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

    def _from_stats(self, strategy: Strategy, cell: GroupStats,
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
            warnings.append(f"single observation for {strategy.strategy_id} in "
                            "this structural group; treat as weak evidence")
        warnings.extend(self._cost_basis_warnings(cost_basis))
        return Recommendation(
            strategy=strategy, score=score,
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

    def _from_no_evidence(self, strategy: Strategy,
                          norms: Optional[Dict[str, float]] = None
                          ) -> Recommendation:
        """No experience for this (strategy, profile) pair. The catalog is a
        structural vocabulary — it tells you the strategy *exists* and *is
        applicable*, but makes no quality/cost/risk claim. Score is
        negative-infinity so any strategy with real evidence (even a
        expensive one) ranks above it; confidence is zero. Expected cost is
        UNKNOWN (empty measured mask) — the placeholder zeros are never
        evidence of cheapness."""
        return Recommendation(
            strategy=strategy,
            score=float('-inf'),
            expected_quality=0.0,
            expected_cost=CostVector(measured=set()),
            failure_prob=0.0,
            evidence="no_memory", confidence=0.0,
            basis="no experience for this strategy×profile pair; "
                  "catalog vocabulary only — no quality claim, and cost is "
                  "UNKNOWN (not zero)",
            cost_known_dims=[], cost_basis_dims=[])


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
