"""Budget ledger: the B of the state, over ALL action costs (M1).

The consumption view deliberately covers MORE than the recorded
attempt-scope executions (``task_cost_summary`` keeps its legacy semantics
untouched):

1. recorded attempt-scope executions of the task;
2. staged-but-unrecorded executions (``bank.pending``) — a finished
   execution that has not been ``record``-ed yet has already spent real
   budget, and the view must not miss it. Deduplication is by
   ``execution_id``: once recorded, the same id moves from the staged set
   to the recorded set and is still counted exactly once;
3. non-execution actions (select / retrieve / verify / induce …) via their
   own ActionRecord costs;
4. macro actions with ``rollup="reference"`` contribute NOTHING here (their
   cost references child costs already counted); a macro's OWN additional
   spend is carried in its cost with ``rollup="own"``.

Status determination is honest about unknowns: with a declared budget,
``exceeded`` requires a MEASURED dimension over the limit; when every
measured dimension is within limits but some dimension is unknown, the
status is ``unconfirmed`` — "not yet known to exceed" is NOT "within
budget". Without a declared budget the consumption is still aggregated and
the status is ``no_budget_declared``.

Hypothetical actions (``source="hypothetical"``) are excluded from every
real consumption view.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.world_model.actions import ActionLog

#: Budget status values.
BUDGET_STATUSES = ("no_budget_declared", "ok", "unconfirmed", "exceeded")


class BudgetLedger:
    """Pure computation view over the Experience Bank + ActionLog."""

    def __init__(self, bank: ExperienceBank, actions: ActionLog):
        self.bank = bank
        self.actions = actions

    # -- consumption ----------------------------------------------------------

    def consumption(self, task_id: str,
                    episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Deduplicated real consumption for one task (optionally one
        episode of it).

        Aggregation rules mirror ``task_cost_summary``: cumulative
        dimensions sum over measured attempts; retries sum (per-attempt NEW
        retries); per-attempt latencies are reported as facts and never
        summed or maxed into an end-to-end figure. Staged executions count
        the same way, deduplicated by execution_id against recorded ones.
        Non-execution actions add their own measured costs (rollup="own"
        only)."""
        recorded = [r for r in self.bank.query(task_id=task_id)
                    if r.source == "executed"
                    and r.measurement_scope == "attempt"]
        if episode_id is not None:
            # Episode scoping: an execution belongs to an episode via its
            # execute_strategy action. Executions with NO action linkage
            # (recorded directly, or pre-M1 records) belong to NO episode:
            # deterministically charging them to every new episode would
            # make a fresh episode start "exceeded". They are reported
            # SEPARATELY (unattributed) instead of being folded in.
            recorded = [r for r in recorded
                        if self._episode_of_execution(r.execution_id)
                        == episode_id]
        recorded_ids = {r.execution_id for r in recorded}
        staged = [p for p in self.bank.pending(task_id=task_id)
                  if p.execution_id not in recorded_ids]
        if episode_id is not None:
            staged = [p for p in staged
                      if self._episode_of_execution(p.execution_id)
                      == episode_id]
        # Unattributed executions: same task, no episode linkage. Reported
        # as their own line so the caller sees the real history without it
        # being charged to this episode.
        if episode_id is not None:
            unattributed = [r for r in self.bank.query(task_id=task_id)
                            if r.source == "executed"
                            and r.measurement_scope == "attempt"
                            and self._episode_of_execution(r.execution_id)
                            is None]
        else:
            unattributed = []

        total: Dict[str, float] = {d: 0.0 for d in COST_DIMENSIONS}
        n_measured: Dict[str, int] = {d: 0 for d in COST_DIMENSIONS}
        per_attempt: List[Dict[str, Any]] = []
        for rec in recorded + staged:
            measured = rec.cost.measured_dims()
            for d in COST_DIMENSIONS:
                if d in measured:
                    n_measured[d] += 1
                    if d != "latency_s":
                        total[d] += getattr(rec.cost, d)
            per_attempt.append({
                "execution_id": rec.execution_id,
                "strategy_id": rec.strategy_id,
                "staged": rec.execution_id not in recorded_ids,
                "cost": rec.cost.to_dict(),
                "cost_measured": sorted(measured),
            })

        # Non-execution actions: their own costs. Hypothetical actions are
        # excluded everywhere. Cost attribution follows ROLLUP, not action
        # type: rollup="reference" means the cost lives on the child
        # (execution or child action) and is counted there — never here,
        # whatever the action type; rollup="own" (including an
        # execute_strategy's own additional spend beyond its child
        # execution) is counted here. A cost of None means UNKNOWN, not
        # zero: the action still participates in the completeness judgment.
        action_costs: List[Dict[str, Any]] = []
        actions = [a for a in self.actions.query(task_id=task_id)
                   if a.source != "hypothetical"]
        if episode_id is not None:
            actions = [a for a in actions if a.episode_id == episode_id]
        for act in actions:
            if act.rollup == "reference":
                continue  # reference costs are counted on the child
            if act.cost is None:
                # Unknown cost: no dimensions to add, but the action exists
                # and its cost is unmeasured — it must make the verdict
                # unconfirmed, never silently "ok".
                action_costs.append({
                    "action_id": act.action_id,
                    "action_type": act.action_type,
                    "cost": None,
                    "cost_measured": [],
                })
                continue
            measured = act.cost.measured_dims()
            for d in COST_DIMENSIONS:
                if d in measured:
                    n_measured[d] += 1
                    if d != "latency_s":
                        total[d] += getattr(act.cost, d)
            action_costs.append({
                "action_id": act.action_id,
                "action_type": act.action_type,
                "cost": act.cost.to_dict(),
                "cost_measured": sorted(measured),
            })

        # A dimension is unknown when at least one contributing item
        # (execution or action, measured or not) exists and any of them did
        # not measure it. latency_s is never summed but IS a declared budget
        # dimension, so its measurement state still matters for honesty.
        n_items = len(recorded) + len(staged) + len(action_costs)
        unknown_dims = [d for d in COST_DIMENSIONS
                        if n_items > 0 and n_measured[d] < n_items]
        unattributed_summary = None
        if unattributed:
            unattributed_summary = {
                "n": len(unattributed),
                "execution_ids": [r.execution_id for r in unattributed],
                "note": ("executions of this task with no episode linkage "
                         "(pre-M1 or directly recorded): reported for "
                         "visibility, NOT charged to this episode"),
            }
        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "n_attempts": len(recorded) + len(staged),
            "n_recorded": len(recorded),
            "n_staged": len(staged),
            "attempts": per_attempt,
            "action_costs": action_costs,
            "unattributed": unattributed_summary,
            "total_cost": {d: (round(total[d], 4) if n_measured[d] > 0
                               else None)
                           for d in COST_DIMENSIONS if d != "latency_s"},
            "n_measured": n_measured,
            "unknown_dims": unknown_dims,
            "aggregation_note": (
                "cumulative dimensions summed over measured recorded AND "
                "staged executions (deduplicated by execution_id) plus "
                "own-cost actions; macro reference costs are not "
                "re-counted; latency never summed (a declared latency "
                "budget is judged per-attempt, see view); unknown "
                "dimensions reported as null, never zero"),
        }

    def _episode_of_execution(self, execution_id: str) -> Optional[str]:
        """The episode a recorded/staged execution belongs to (via its
        execute_strategy action), or None when no action recorded it."""
        for act in self.actions.query():
            if (act.action_type == "execute_strategy"
                    and act.linked_execution_id == execution_id):
                return act.episode_id
        return None

    # -- status ----------------------------------------------------------------

    def view(self, task_id: str, episode_id: Optional[str] = None,
             budget: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Budget view: declaration + consumption + honest status.

        ``budget`` (optional, per CostVector dimension) may come from the
        caller or from a previously declared episode budget."""
        consumption = self.consumption(task_id, episode_id)
        if budget is None:
            budget = {}
        budget = {d: float(v) for d, v in budget.items()
                  if d in COST_DIMENSIONS}
        if not budget:
            status = "no_budget_declared"
        else:
            # Honesty is scoped to DECLARED dimensions: an undeclared
            # dimension is not part of this budget's promise. A declared
            # dimension that no contributing item measured (or that some
            # measured and others did not) makes the verdict unconfirmed.
            declared_unknown = [d for d in budget
                                if d in consumption["unknown_dims"]]
            exceeded = any(
                d in budget and consumption["n_measured"].get(d, 0) > 0
                and consumption["total_cost"].get(d) is not None
                and consumption["total_cost"][d] > budget[d]
                for d in COST_DIMENSIONS)
            # latency_s is never summed (attempts may overlap), so a
            # declared latency budget is judged PER ATTEMPT: any single
            # measured attempt latency over the limit exceeds the budget.
            if not exceeded and "latency_s" in budget:
                limit = budget["latency_s"]
                for attempt in consumption["attempts"]:
                    latency = (attempt.get("cost") or {}).get("latency_s")
                    if latency is not None and latency > limit:
                        exceeded = True
                        break
            if exceeded:
                status = "exceeded"
            elif declared_unknown:
                # Known spend within limits but a declared dimension is
                # unknown: NOT confirmed within budget.
                status = "unconfirmed"
            else:
                status = "ok"
        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "budget": budget or None,
            "status": status,
            "consumption": consumption,
            "status_note": {
                "no_budget_declared": "no budget declared: consumption is "
                                      "aggregated but not judged",
                "ok": "every dimension measured and within the declared "
                      "budget",
                "unconfirmed": "known spend is within the declared budget "
                               "but some cost dimensions are unknown — "
                               "actual remaining budget is NOT confirmed",
                "exceeded": "a measured dimension exceeds the declared "
                            "budget",
            }[status],
        }
