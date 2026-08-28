"""Candidate retrieval: structurally-isomorphic cluster discovery (module 3.1).

This is the ENTRY POINT of the induction pipeline. It answers: "which groups of
realizations are worth inducing a cross-family pattern from?"

Red lines honoured here:
- Heterogeneous complementarity: a cluster must span >= 2 distinct problem families
  (A!=B!=C but structurally isomorphic). Same-family groups are redundancy, which is
  Auto-Dreamer's territory, not ours.
- Avoid O(N^2): we NEVER compare every realization pair. We build a two-level inverted
  index over the structural signature (core-4 fixed key + open feature dynamic keys),
  then induce within buckets. Pairwise work happens later (alignment), only inside a
  small candidate cluster.
- Provenance-grounded (borrowed from Auto-Dreamer): every cluster member carries its
  evidence.source_episodes handle so downstream stages can drill down to the raw
  Episode (the get_source_trace analogue) instead of trusting the abstraction alone.

problem_family resolution: ModelingExperience has no direct problem_family field (that
lives on the flat ExperienceRecord.problem_context, and on Episode.normalized_spec). We
resolve it, in order, from:
  1. an explicit per-realization override supplied by the caller (tests / known data);
  2. the linked Episode's normalized_spec["problem_family"] via evidence.source_episodes;
  3. a best-effort keyword parse of retrieval_text / title.
If none yields a family we fall back to the realization_id so that unresolved records
each count as their own "family" and never silently merge into fake heterogeneity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..core.modeling_schemas import StructuralSignature


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class ClusterMember:
    """One realization inside a candidate cluster, with provenance + priority signals."""

    realization_id: str
    title: str = ""
    problem_family: str = ""
    signature: Optional[StructuralSignature] = None
    # provenance handle: Episode ids this realization was distilled from (Auto-Dreamer's
    # provenance pointer; lets alignment drill down to raw evidence, get_source_trace-style).
    source_episodes: List[str] = field(default_factory=list)
    created_at: str = ""            # recency signal for trigger priority
    retrieval_hits: int = 0         # usage signal for trigger priority (Auto-Dreamer region hint)
    # the raw record, kept for downstream stages (alignment/inducer need method/roles)
    record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realization_id": self.realization_id,
            "title": self.title,
            "problem_family": self.problem_family,
            "signature": self.signature.to_dict() if self.signature else None,
            "source_episodes": list(self.source_episodes),
            "created_at": self.created_at,
            "retrieval_hits": self.retrieval_hits,
        }


@dataclass
class CandidateCluster:
    """A structurally-isomorphic, cross-family group of realizations worth inducing over."""

    cluster_id: str
    core_key: str                                   # shared core-4 signature key (isomorphism)
    shared_feature_keys: List[str] = field(default_factory=list)  # feature-key intersection
    members: List[ClusterMember] = field(default_factory=list)
    representative_signature: Optional[StructuralSignature] = None
    score: float = 0.0                              # trigger priority (higher = induce sooner)

    @property
    def problem_families(self) -> List[str]:
        return sorted({m.problem_family for m in self.members if m.problem_family})

    @property
    def size(self) -> int:
        return len(self.members)

    def is_heterogeneous(self, min_families: int = 2) -> bool:
        return len(self.problem_families) >= min_families

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "core_key": self.core_key,
            "shared_feature_keys": list(self.shared_feature_keys),
            "problem_families": self.problem_families,
            "size": self.size,
            "score": self.score,
            "members": [m.to_dict() for m in self.members],
        }


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class SignatureClusterer:
    """Two-level inverted index over realization signatures -> candidate clusters.

    Level 1 (fixed): bucket realizations by signature.core_key() -> structurally
    isomorphic groups. Level 2 (dynamic): within a bucket, sub-split by the intersection
    of open feature keys so that members also share problem-specific structure. This
    keeps candidate discovery O(N) index-building plus small in-bucket work, never O(N^2).
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_families: int = 2,
        family_resolver: Optional[Callable[[Mapping[str, Any]], str]] = None,
        episode_family_index: Optional[Mapping[str, str]] = None,
        lifecycle: Optional[Any] = None,
        utility_tracker: Optional[Any] = None,
    ):
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2 (a cluster needs at least a pair)")
        if min_families < 2:
            raise ValueError("min_families must be >= 2 (heterogeneous complementarity red line)")
        self.min_cluster_size = min_cluster_size
        self.min_families = min_families
        # optional caller override for problem_family (tests / known data)
        self._family_resolver = family_resolver
        # episode_id -> problem_family, used to resolve family via provenance links
        self._episode_family_index = dict(episode_family_index or {})
        # Phase 2.3 wiring: exclude deprecated records from clustering, and prefer live
        # retrieval-hit counts from the utility tracker over any static record field.
        self._lifecycle = lifecycle
        self._utility_tracker = utility_tracker

    # -- public API --------------------------------------------------------

    def discover(self, realizations: Iterable[Mapping[str, Any]]) -> List[CandidateCluster]:
        """Build the index and return heterogeneous candidate clusters, sorted by priority."""
        members = [self._to_member(r) for r in realizations]
        members = [m for m in members if m is not None]
        # exclude deprecated records: retired/harmful experiences must not seed induction
        if self._lifecycle is not None:
            members = [m for m in members if self._lifecycle.state_of(m.realization_id) != "deprecated"]
        core_buckets: Dict[str, List[ClusterMember]] = defaultdict(list)
        for m in members:
            core_buckets[m.signature.core_key()].append(m)

        clusters: List[CandidateCluster] = []
        for core_key, bucket in core_buckets.items():
            if len(bucket) < self.min_cluster_size:
                continue
            for sub in self._split_by_features(bucket):
                if len(sub) < self.min_cluster_size:
                    continue
                cluster = self._build_cluster(core_key, sub)
                if cluster.is_heterogeneous(self.min_families):
                    clusters.append(cluster)

        clusters.sort(key=lambda c: (c.score, c.size), reverse=True)
        return clusters

    # -- index building ----------------------------------------------------

    def _to_member(self, record: Mapping[str, Any]) -> Optional[ClusterMember]:
        try:
            signature = StructuralSignature.from_dict(
                (record.get("math_scope") or {}).get("structural_signature", {})
            )
        except Exception:
            return None  # no usable signature -> cannot participate in structural induction
        evidence = record.get("evidence") or {}
        rid = record.get("experience_id", "")
        # prefer the live utility-tracker count (Phase 2.3); fall back to a static field
        if self._utility_tracker is not None:
            retrieval_hits = self._utility_tracker.retrieval_count(rid)
        else:
            retrieval_hits = int(record.get("retrieval_count", 0) or 0)
        return ClusterMember(
            realization_id=rid,
            title=record.get("title", ""),
            problem_family=self._resolve_family(record),
            signature=signature,
            source_episodes=list(evidence.get("source_episodes", [])),
            created_at=record.get("created_at", ""),
            retrieval_hits=retrieval_hits,
            record=dict(record),
        )

    def _split_by_features(self, bucket: List[ClusterMember]) -> List[List[ClusterMember]]:
        """Sub-split a core bucket into feature-coherent groups (missing keys NOT penalized).

        Two members belong together when their shared feature KEYS (intersection) agree
        on VALUES for every common key. A member missing a key imposes no constraint on
        that key, so it stays compatible with everyone (alignment rule: missing != penalty,
        D9). This is a compatibility connected-components problem; we resolve it with a
        union-find so we never do full pairwise cross-bucket comparison.
        """
        n = len(bucket)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def compatible(a: ClusterMember, b: ClusterMember) -> bool:
            fa = a.signature.features if a.signature else {}
            fb = b.signature.features if b.signature else {}
            # Members that both carry open-feature structure must share the SAME key set:
            # alignment needs a common role vocabulary, so a temporal-structured record and
            # a network-structured record are NOT induced together. A member with NO open
            # features carries no structural prior, so it joins any group (missing != penalty).
            if fa and fb and set(fa) != set(fb):
                return False
            for key in set(fa) & set(fb):  # shared keys must agree on values
                if fa[key] != fb[key]:
                    return False
            return True

        for i in range(n):
            for j in range(i + 1, n):
                if compatible(bucket[i], bucket[j]):
                    union(i, j)

        groups: Dict[int, List[ClusterMember]] = defaultdict(list)
        for idx, member in enumerate(bucket):
            groups[find(idx)].append(member)
        return list(groups.values())

    def _build_cluster(self, core_key: str, members: List[ClusterMember]) -> CandidateCluster:
        shared = self._feature_key_intersection(members)
        # Stable id across processes: builtin hash() is randomized per process
        # (PYTHONHASHSEED), which breaks cross-command workflows like the orx CLI
        # where each command is a fresh process. Use sha256 over the membership.
        import hashlib
        digest = hashlib.sha256(
            (core_key + "|" + ",".join(sorted(m.realization_id for m in members))).encode("utf-8")
        ).hexdigest()
        cluster_id = "clu_" + digest[:10]
        return CandidateCluster(
            cluster_id=cluster_id,
            core_key=core_key,
            shared_feature_keys=shared,
            members=members,
            representative_signature=members[0].signature,
            score=self._priority(members),
        )

    @staticmethod
    def _feature_key_intersection(members: Sequence[ClusterMember]) -> List[str]:
        if not members:
            return []
        common: Optional[Set[str]] = None
        for m in members:
            keys = set((m.signature.features or {}).keys()) if m.signature else set()
            common = keys if common is None else (common & keys)
        return sorted(common or set())

    def _priority(self, members: Sequence[ClusterMember]) -> float:
        """Trigger priority: favour larger, more-heterogeneous, recently/active clusters.

        Borrows Auto-Dreamer's region-selection intuition (prefer freshly-written +
        recently-retrieved entries) but stays rule-based for v1.
        """
        size = len(members)
        families = len({m.problem_family for m in members if m.problem_family})
        hits = sum(m.retrieval_hits for m in members)
        recency = sum(1.0 for m in members if m.created_at)  # presence as a cheap proxy
        return float(size * 2 + families * 3 + hits + recency)

    # -- problem_family resolution ----------------------------------------

    def _resolve_family(self, record: Mapping[str, Any]) -> str:
        if self._family_resolver is not None:
            resolved = self._family_resolver(record)
            if resolved:
                return resolved
        # via provenance: linked Episode's normalized_spec.problem_family
        for ep in (record.get("evidence") or {}).get("source_episodes", []):
            family = self._episode_family_index.get(ep)
            if family:
                return family
        # best-effort keyword parse of retrieval_text / title
        text = (record.get("retrieval_text", "") + " " + record.get("title", "")).lower()
        for keyword in _KNOWN_FAMILY_KEYWORDS:
            if keyword in text:
                return keyword
        # fallback: isolate, never fake-merge
        return record.get("experience_id", "unknown")


_KNOWN_FAMILY_KEYWORDS = (
    "tsp", "vrp", "assignment", "scheduling", "inventory", "production",
    "facility", "allocation", "network", "knapsack", "covering", "routing",
    "blending", "portfolio", "workforce", "transportation", "lp",
)


__all__ = ["CandidateCluster", "ClusterMember", "SignatureClusterer"]
