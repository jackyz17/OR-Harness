"""Derived error-transition graph for the Repair Bank (module 0.4, Decision D16).

Repair knowledge is inherently CHAINED: applying a repair to error A may succeed or
surface a NEW error B. Isolated JSONL facts lose this topology. This module derives a
single directed graph from the (append-only) repair facts:

    node = (solver, normalized_error)   -- composite key, logical isolation per solver
    edge = repair action {result, generality, success_rate, frequency}

Design (Option 2, finalized): ONE graph, not one physical graph per solver. Cross-solver
migration is governed by the repair record's own `generality` scope tag (we reuse the
existing 3-level scope instead of inventing a new isolation mechanism):

    solver_agnostic : Python-level errors (TypeError/IndexError/timeout) -> may cross solvers
    solver_family   : solver status/numeric errors (infeasible/unbounded)  -> migrate within family
    solver_specific : solver API errors (gurobipy/pyscipopt/ortools.cp)    -> never cross

The graph is a DERIVED index, never a fact: it can be rebuilt from the repair JSONL at
any time, preserving the append-only guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.schemas import ExperienceGenerality


# solver -> family, mirrors extractor._family; single source of truth for migration.
SOLVER_FAMILIES: Dict[str, str] = {
    "gurobi": "milp",
    "scip": "milp",
    "highs": "milp",
    "copt": "milp",
    "pulp": "milp",
    "pyomo": "milp",
    "ortools": "cp_sat",
}

_SUCCESS_STATUSES = {"optimal", "feasible"}
_NODE_KEY = Tuple[str, str]  # (solver, normalized_error)


def _family_of(solver: str) -> str:
    return SOLVER_FAMILIES.get(solver, "unknown")


class ErrorTransitionGraph:
    """Single directed error-transition graph with (solver, error) composite nodes."""

    def __init__(self) -> None:
        # node -> list of outgoing transition edges
        self._edges: Dict[_NODE_KEY, List[Dict[str, Any]]] = defaultdict(list)
        self._edge_stats: Dict[Tuple[_NODE_KEY, str], Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )

    # ------------------------------------------------------------------ build
    def add_transition(
        self,
        solver: str,
        normalized_error: str,
        action: str,
        result_status: str,
        generality: str = ExperienceGenerality.SOLVER_SPECIFIC.value,
        next_error: Optional[str] = None,
    ) -> None:
        """Record one observed repair transition (from a repair fact).

        result_status in {"optimal","feasible"} counts as success; anything else is a
        failure. When the repair surfaces a new error, pass it via next_error so the
        edge points to the chained node.
        """
        node = (solver, normalized_error)
        success = result_status in _SUCCESS_STATUSES
        edge = {
            "action": action,
            "generality": generality,
            "success": success,
            "next_node": (solver, next_error) if (next_error and not success) else None,
            "solver_family": _family_of(solver),
        }
        self._edges[node].append(edge)
        stats = self._edge_stats[(node, action)]
        stats["success" if success else "failure"] += 1

    def add_record(self, record: Dict[str, Any]) -> None:
        """Ingest one repair-layer JSONL fact (existing ExperienceRecord shape)."""
        scope = record.get("scope", {})
        trigger = record.get("trigger", {})
        policy = record.get("policy", {})
        solver = scope.get("solver")
        error = trigger.get("normalized_error")
        if not solver or not error:
            return
        # A successful repair record implies its action resolved the error.
        polarity = record.get("polarity", "positive")
        status = "feasible" if polarity == "positive" else "error"
        self.add_transition(
            solver=solver,
            normalized_error=error,
            action=policy.get("action", ""),
            result_status=status,
            generality=scope.get("generality", ExperienceGenerality.SOLVER_SPECIFIC.value),
        )

    def rebuild(self, records: Iterable[Dict[str, Any]]) -> "ErrorTransitionGraph":
        """Rebuild the whole graph from repair facts (derived index semantics)."""
        self._edges.clear()
        self._edge_stats.clear()
        for record in records:
            self.add_record(record)
        return self

    # ------------------------------------------------------------------ query
    def query(
        self, solver: str, normalized_error: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return ranked repair candidates for (solver, error) using 3-level migration.

        Order: (1) this solver's specific edges, (2) same-family edges, (3) agnostic
        edges. Within each tier, sort by observed success rate then frequency.
        """
        target_family = _family_of(solver)
        specific: List[Dict[str, Any]] = []
        family: List[Dict[str, Any]] = []
        agnostic: List[Dict[str, Any]] = []

        for (node_solver, node_error), edges in self._edges.items():
            for edge in edges:
                scored = self._score(node_solver, node_error, edge)
                if scored is None:
                    continue
                gen = edge["generality"]
                if node_solver == solver and node_error == normalized_error:
                    specific.append(scored)
                elif gen == ExperienceGenerality.SOLVER_FAMILY.value and edge["solver_family"] == target_family:
                    family.append(scored)
                elif gen == ExperienceGenerality.SOLVER_AGNOSTIC.value:
                    agnostic.append(scored)

        ranked = specific + family + agnostic
        # Stable tier ordering (specific < family < agnostic) then by score desc.
        return ranked[: max(0, top_k)]

    def _score(
        self, node_solver: str, node_error: str, edge: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        stats = self._edge_stats.get(((node_solver, node_error), edge["action"]))
        if not stats:
            return None
        total = stats["success"] + stats["failure"]
        rate = stats["success"] / total if total else 0.0
        return {
            "action": edge["action"],
            "generality": edge["generality"],
            "source_solver": node_solver,
            "success_rate": rate,
            "frequency": total,
            "next_error": (edge["next_node"][1] if edge["next_node"] else None),
        }

    # --------------------------------------------------------------- analytics
    def known_pitfalls(self, solver: str, normalized_error: str) -> List[str]:
        """New errors a repair on (solver, error) tends to surface ('fix A, beware B')."""
        pitfalls: List[str] = []
        for edge in self._edges.get((solver, normalized_error), []):
            if edge["next_node"] and not edge["success"]:
                pitfalls.append(edge["next_node"][1])
        return sorted(set(pitfalls))

    def shortest_repair_path(self, solver: str, normalized_error: str, max_depth: int = 4) -> List[str]:
        """BFS for a chain of repairs from error to success ('how to dig out of this hole')."""
        from collections import deque

        start = (solver, normalized_error)
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (node_solver, node_error), path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for edge in self._edges.get((node_solver, node_error), []):
                action = edge["action"]
                if edge["success"]:
                    return path + [action]
                nxt = edge["next_node"]
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [action]))
        return []

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": len(self._edges),
            "edges": sum(len(v) for v in self._edges.values()),
        }


__all__ = ["ErrorTransitionGraph", "SOLVER_FAMILIES"]
