"""Strategy selector: two-layer fallback, transparent scoring, ablation modes.

Score = alpha * Q_hat - beta * C_scalar - gamma * R_hat

Evidence precedence (per strategy):
  1. Strategic entries whose pattern matches the profile (commitments).
  2. Conditional statistics over the Experience Bank (recounts) — this single
     path subsumes what older designs split into "case retrieval" and
     "statistics": both are the same data used two ways.
  3. Built-in catalog priors (cold start).

Ablation modes (--memory-mode), reused by the experiments runner:
  A none      — default strategy, no memory.
  B cases     — case-based evidence only (no entries, no cost weighting).
  C strategic — entries/stats condition the choice, WITHOUT cost weighting.
  D cost-aware— C + cost scalarization. C vs D only separates in
                "quality tied, cost divergent" scenarios — the experimental
                support for 'cost awareness is a necessary part of memory'.

Cross-family generalization (an L2/L3 entry matching a family it has no
provenance in) carries an explicit confidence discount and is labelled.
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
    evidence: str  # "strategic_entry" | "conditional_stats" | "prior"
    evidence_refs: List[str] = field(default_factory=list)  # entry/execution ids
    confidence: float = 1.0
    cross_family: bool = False
    risk_warnings: List[str] = field(default_factory=list)
    basis: str = ""

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

    def recommend(self, profile: ProblemProfile, *, top: int = 3,
                  exclude: Optional[Sequence[str]] = None,
                  memory_mode: str = "cost-aware") -> List[Recommendation]:
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        excluded = set(exclude or [])
        candidates = [s for s in self.catalog.values()
                      if s.strategy_id not in excluded
                      and self._applies(s, profile)]
        if memory_mode == "none":
            default = next((s for s in candidates if s.strategy_id == "S01"),
                           candidates[0] if candidates else None)
            return ([self._from_prior(default, basis="memory off: default strategy")]
                    if default else [])

        entries = self.sbank.matching(profile) if memory_mode in ("strategic", "cost-aware") else []
        cells = self.stats.for_profile(profile, "L1")
        consulted: List[str] = []
        recs: List[Recommendation] = []
        for strategy in candidates:
            rec = self._score(strategy, profile, entries, cells, memory_mode)
            consulted.extend(r for r in rec.evidence_refs if r.startswith("se_"))
            recs.append(rec)
        if consulted:
            self.sbank.mark_consulted(sorted(set(consulted)))
        recs.sort(key=lambda r: (-r.score, r.strategy.strategy_id))
        return recs[: max(1, top)]

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _applies(strategy: Strategy, profile: ProblemProfile) -> bool:
        from or_harness.core.schema import profile_matches
        return profile_matches(profile, strategy.applicability) if strategy.applicability else True

    def _score(self, strategy: Strategy, profile: ProblemProfile,
               entries: List[StrategicEntry],
               cells: Dict[str, GroupStats],
               memory_mode: str) -> Recommendation:
        entry = next((e for e in entries if e.strategy_id == strategy.strategy_id), None)
        cell = cells.get(strategy.strategy_id)
        if entry is not None and memory_mode in ("strategic", "cost-aware"):
            return self._from_entry(strategy, profile, entry, memory_mode)
        if cell is not None and cell.n > 0 and memory_mode in ("cases", "strategic", "cost-aware"):
            return self._from_stats(strategy, cell, memory_mode)
        return self._from_prior(strategy, basis="no memory evidence; catalog prior")

    def _entry_confidence(self, entry: StrategicEntry, profile: ProblemProfile) -> float:
        # Confidence scales with support (new entries get a grace floor), and
        # cross-family generalization is explicitly discounted.
        conf = max(NEW_ENTRY_CONFIDENCE_FLOOR,
                   min(1.0, entry.support_n / PROMOTE_REFERENCE_N))
        cross_family = (entry.scope_level in ("L2", "L3")
                        and profile.family not in self._provenance_families(entry))
        if cross_family:
            conf *= CROSS_FAMILY_CONFIDENCE_DISCOUNT
        return conf

    def _provenance_families(self, entry: StrategicEntry) -> set:
        families = set()
        for ex_id in entry.provenance:
            rec = self.stats.bank.get(ex_id)
            if rec is not None:
                families.add(rec.profile_snapshot.family)
        return families

    def _from_entry(self, strategy: Strategy, profile: ProblemProfile,
                    entry: StrategicEntry, memory_mode: str) -> Recommendation:
        confidence = self._entry_confidence(entry, profile)
        cross_family = (entry.scope_level in ("L2", "L3")
                        and profile.family not in self._provenance_families(entry))
        cost = entry.expected_cost_hat
        cost_term = (cost.scalarize(self.cost_weights)
                     if memory_mode == "cost-aware" else 0.0)
        score = (self.alpha * entry.expected_quality_hat
                 - self.beta * cost_term
                 - self.gamma * entry.failure_prob)
        warnings: List[str] = []
        if entry.status == "suspect":
            score *= SUSPECT_SCORE_FACTOR
            warnings.append(
                f"entry {entry.entry_id} is suspect (3 consecutive prediction "
                "misses); its estimate is downweighted x0.5")
        if cross_family:
            warnings.append(
                f"cross-family generalization from {entry.scope_level} entry "
                f"{entry.entry_id}; confidence discounted x{CROSS_FAMILY_CONFIDENCE_DISCOUNT}")
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
                  f"n={entry.prediction_track.n_predictions})")

    def _from_stats(self, strategy: Strategy, cell: GroupStats,
                    memory_mode: str) -> Recommendation:
        cost = cell.mean_cost
        cost_term = (cost.scalarize(self.cost_weights)
                     if memory_mode == "cost-aware" else 0.0)
        score = (self.alpha * cell.mean_quality
                 - self.beta * cost_term
                 - self.gamma * cell.fail_rate)
        warnings = []
        if cell.n < 2:
            warnings.append(f"single observation for {strategy.strategy_id} in "
                            "this structural group; treat as weak evidence")
        return Recommendation(
            strategy=strategy, score=score,
            expected_quality=cell.mean_quality, expected_cost=cost,
            failure_prob=cell.fail_rate, evidence="conditional_stats",
            evidence_refs=list(cell.execution_ids),
            confidence=min(1.0, cell.n / PROMOTE_REFERENCE_N),
            risk_warnings=warnings,
            basis=f"conditional statistics over n={cell.n} executions in this group")

    def _from_prior(self, strategy: Strategy, basis: str) -> Recommendation:
        return Recommendation(
            strategy=strategy,
            score=(self.alpha * strategy.expected_quality
                   - self.beta * strategy.expected_cost.scalarize(self.cost_weights)
                   - self.gamma * strategy.expected_risk),
            expected_quality=strategy.expected_quality,
            expected_cost=strategy.expected_cost,
            failure_prob=strategy.expected_risk,
            evidence="prior", confidence=NEW_ENTRY_CONFIDENCE_FLOOR,
            basis=basis)


#: support_n at which an entry reaches full confidence.
PROMOTE_REFERENCE_N = 5.0
