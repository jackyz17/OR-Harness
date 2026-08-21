"""Layer-aware text retrieval with hard metadata and solver-scope filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .error_transition_graph import ErrorTransitionGraph
from .index import EmbeddingIndex
from ..core.lifecycle import DEPRECATED, LifecycleStore
from ..core.schemas import ExperienceGenerality, ExperienceLayer, RetrievalHit
from ..core.store import AppendOnlyExperienceStore
from ..core.utility_tracker import UtilityTracker


class ExperienceRetriever:
    def __init__(
        self,
        store: AppendOnlyExperienceStore,
        index: EmbeddingIndex,
        utility_tracker: Optional[UtilityTracker] = None,
        lifecycle: Optional[LifecycleStore] = None,
    ):
        self.store = store
        self.index = index
        # Optional Phase-2.3 add-ons. When present: retrieval hits are counted, low-utility
        # records are score-penalized (soft delete), and deprecated records are excluded
        # from the rebuilt index (they live only in the cold archive).
        self.utility_tracker = utility_tracker
        self.lifecycle = lifecycle

    def repair_guidance(self, solver: str, normalized_error: str, top_k: int = 5) -> Dict[str, Any]:
        """Derived error-transition-graph guidance for a (solver, error) — Option (b):
        the graph is rebuilt on demand from repair facts, never persisted as truth.

        Returns ranked repair actions (3-level generality migration), known pitfalls
        (errors a repair tends to surface), and a shortest repair path to success."""
        graph = ErrorTransitionGraph().rebuild(self.store.iter_records("repair", strict=False))
        return {
            "actions": graph.query(solver, normalized_error, top_k=top_k),
            "pitfalls": graph.known_pitfalls(solver, normalized_error),
            "repair_path": graph.shortest_repair_path(solver, normalized_error),
        }

    def rebuild(self, layer: Optional[str] = None) -> List[Dict[str, Any]]:
        layers = [ExperienceLayer(layer)] if layer else list(ExperienceLayer)
        return [
            self.index.rebuild(item.value, self._retrievable_records(item.value)) for item in layers
        ]

    def _retrievable_records(self, layer: str):
        """Fact records minus deprecated ones (deprecated live only in the cold archive)."""
        records = self.store.iter_records(layer)
        if self.lifecycle is None:
            return records
        return [r for r in records if self.lifecycle.state_of(r.get("experience_id", "")) != DEPRECATED]

    def retrieve(
        self,
        layer: str,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[RetrievalHit]:
        ExperienceLayer(layer)
        if not self.index.path(layer).exists():
            self.rebuild(layer)
        filters = dict(metadata_filters or {})
        filters.setdefault("layer", layer)
        scored: List[tuple] = []
        for score, record in self.index.query(layer, query):
            exp_id = record.get("experience_id", "")
            # belt-and-suspenders: skip deprecated even if a stale index still holds it
            if self.lifecycle is not None and self.lifecycle.state_of(exp_id) == DEPRECATED:
                continue
            if min_score is not None and score < min_score:
                continue
            if not self._matches(record, filters):
                continue
            # soft delete: penalize low-utility records (they sink, but are not deleted)
            if self.utility_tracker is not None:
                score = self.utility_tracker.apply_penalty(exp_id, score)
            scored.append((score, record))
        # re-rank after penalization, then take top_k
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: List[RetrievalHit] = []
        for score, record in scored[: max(0, top_k)]:
            hits.append(
                RetrievalHit(
                    experience_id=record["experience_id"],
                    layer=record["layer"],
                    title=record["title"],
                    score=score,
                    polarity=record["polarity"],
                    scope=record["scope"],
                    retrieval_text=record["retrieval_text"],
                    record=record,
                )
            )
        # count what was actually surfaced (utility statistics input)
        if self.utility_tracker is not None:
            self.utility_tracker.record_retrievals([h.experience_id for h in hits])
        return hits

    def validate_indexes(self) -> Dict[str, Any]:
        layers: Dict[str, Any] = {}
        valid = True
        for layer in ExperienceLayer:
            facts = list(self.store.iter_records(layer.value, strict=False))
            payload = self.index.load(layer.value)
            fact_pairs = [(row.get("experience_id"), row.get("content_hash")) for row in facts]
            index_pairs = [
                (row.get("experience_id"), row.get("content_hash"))
                for row in payload.get("records", [])
            ]
            layer_valid = bool(payload) and fact_pairs == index_pairs
            if not facts and not payload:
                layer_valid = True
            layers[layer.value] = {
                "valid": layer_valid,
                "fact_count": len(facts),
                "index_count": len(index_pairs),
                "model_id": payload.get("model_id"),
            }
            valid = valid and layer_valid
        return {"valid": valid, "layers": layers}

    def _matches(self, record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        context = record.get("problem_context", {})
        scope = record.get("scope", {})
        direct = {"layer": record.get("layer"), "polarity": record.get("polarity")}
        values = {
            **direct,
            "solver": scope.get("solver"),
            "solver_family": scope.get("solver_family"),
            "generality": scope.get("generality"),
            "problem_family": context.get("problem_family"),
            "stage": context.get("stage"),
        }
        for key, expected in filters.items():
            if expected is None:
                continue
            if key == "solver":
                generality = scope.get("generality")
                if generality == ExperienceGenerality.SOLVER_SPECIFIC.value and scope.get("solver") != expected:
                    return False
                if generality == ExperienceGenerality.SOLVER_FAMILY.value:
                    requested_family = filters.get("solver_family")
                    if requested_family and scope.get("solver_family") != requested_family:
                        return False
                continue
            if key == "solver_family" and scope.get("generality") == ExperienceGenerality.SOLVER_AGNOSTIC.value:
                continue
            if key in values and values[key] != expected:
                return False
        return True
