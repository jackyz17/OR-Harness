"""Attempt vs strategy execution window: the scope a prediction covers.

One ``execute_strategy`` call is ONE EXECUTION ATTEMPT — a single solve
invocation with its own staged/recorded ExecutionRecord. A STRATEGY
EXECUTION WINDOW is the larger thing a strategy actually is in practice:
writing the model, running the solver (possibly more than once), repairing,
verifying. Those are different units, and the difference is not cosmetic:
scoring a window-scope prediction against one attempt's numbers (or the
reverse) silently compares two different things.

This module derives a window from the REAL action log — never from an
intention. Its rules:

- **The predicted scope is declared, not assumed.** A window states which
  action types are IN scope (the ones the prediction speaks about) and
  which are AUXILIARY overhead (real spend, counted in the ledger, but not
  part of what was predicted). ``execute_strategy`` attempts are in scope by
  default because that is what the legacy prediction path actually
  predicted; ``model`` / ``verify`` / ``select_strategy`` are auxiliary
  unless the caller explicitly declares otherwise.
- **Only a COMPLETE real scope is comparable.** A window with no executed
  attempt, an attempt with no linked execution, or an attempt that has not
  ENDED is reported with ``comparable=False`` and the reasons. A prediction
  may not be marked comparable against a scope that does not exist, or that
  is still moving: a half-finished window has no final numbers to score
  against.
- **A window describes ONE identity.** Its task, episode and strategy are
  checked against the candidate it is used for
  (:func:`window_identity_problems`). A window of another task, another
  episode or another strategy is a mismatch, never a comparable scope —
  scoring across identities is the silent interchange this module exists to
  prevent.
- **Auxiliary cost is never hidden.** The auxiliary actions' own measured
  spend is reported separately (and is already counted by
  :class:`~or_harness.world_model.budget.BudgetLedger`), so "the strategy
  was cheap" cannot be claimed by omitting the modeling work that made it
  possible.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.schema import accumulate_measured_costs

#: Prefix of a deterministic window id (see :func:`window_id_for`).
WINDOW_ID_PREFIX = "win::"

#: Action types that may belong to a strategy execution window.
WINDOW_ACTION_TYPES = ("model", "select_strategy", "execute_strategy",
                       "verify", "finish_task")

#: The action types IN the predicted scope by default. This is exactly what
#: the existing single-step prediction path predicts: one solve attempt.
DEFAULT_IN_SCOPE_ACTION_TYPES = ("execute_strategy",)

#: Action types that are real work but auxiliary to the predicted scope by
#: default: they cost real budget and are reported, never predicted.
DEFAULT_AUXILIARY_ACTION_TYPES = ("model", "select_strategy", "verify")


def window_id_for(task_id: str, episode_id: Optional[str],
                  strategy_id: Optional[str],
                  round_index: Optional[int] = None) -> str:
    """A deterministic window id from the identity that defines it.

    Deterministic on purpose: the same real window asked for twice gets the
    same id, so a prediction's ``window_id`` reference stays resolvable and
    a duplicate window cannot appear as two.

    ``round_index`` (M4) separates DISTINCT SELECTION ROUNDS of the same
    (task, episode, strategy): a strategy chosen, abandoned, and chosen
    again under a different config is TWO windows, not one aggregated
    sample. ``None`` keeps the legacy three-part id (old references stay
    resolvable); an explicit ``0`` is the first round.
    """
    base = (f"{WINDOW_ID_PREFIX}{task_id}::{episode_id or '-'}::"
            f"{strategy_id or '-'}")
    if round_index is None:
        return base
    return f"{base}::r{int(round_index)}"


#: The identity that defines a window: (task, episode, strategy, round).
WindowIdentity = namedtuple("WindowIdentity",
                            ["task_id", "episode_id", "strategy_id",
                             "round_index"])


def parse_window_id(window_id: str) -> Optional[WindowIdentity]:
    """Split a window id back into the identity that defines it.

    Returns ``None`` when the string was not produced by
    :func:`window_id_for`. This is what lets a prediction REFUSE a
    ``window_id`` that belongs to another task / episode / strategy instead
    of trusting the caller's string. A fourth ``rN`` part (the selection
    round, M4) is optional: a three-part id is a legacy/round-less window
    and parses with ``round_index=None``.
    """
    if not isinstance(window_id, str) \
            or not window_id.startswith(WINDOW_ID_PREFIX):
        return None
    parts = window_id[len(WINDOW_ID_PREFIX):].split("::")
    if len(parts) not in (3, 4):
        return None
    task_id, episode, strategy = parts[:3]
    if not task_id:
        return None
    round_index: Optional[int] = None
    if len(parts) == 4:
        token = parts[3]
        if not token.startswith("r"):
            return None
        try:
            round_index = int(token[1:])
        except ValueError:
            return None
        if round_index < 0:
            return None
    return WindowIdentity(task_id,
                          None if episode == "-" else episode,
                          None if strategy == "-" else strategy,
                          round_index)


def window_identity_problems(window: Any, *,
                             task_id: Optional[str],
                             episode_id: Optional[str],
                             strategy_id: Optional[str],
                             round_index: Optional[int] = None
                             ) -> List[str]:
    """Reasons a window does NOT describe the identity it is used for.

    ``window`` may be a :class:`StrategyExecutionWindow`, a
    :class:`WindowIdentity` (from :func:`parse_window_id`), or a window-id
    STRING. An expected value of ``None`` means "no expectation recorded"
    and is not a mismatch; a window naming a DIFFERENT task / episode /
    strategy always is. Callers must treat a non-empty result as a refusal
    to use the window, never as a note to carry along.

    ``round_index`` (M4) participates only when BOTH sides record one: a
    round-less legacy window is not a round mismatch, but a window of round
    1 used for a round-0 candidate is.
    """
    problems: List[str] = []
    if isinstance(window, str):
        window = parse_window_id(window)
        if window is None:
            return [f"window id {window!r} is not a parseable window id"]
    window_task = getattr(window, "task_id", None)
    window_episode = getattr(window, "episode_id", None)
    window_strategy = getattr(window, "strategy_id", None)
    window_round = getattr(window, "round_index", None)
    label = getattr(window, "window_id", None) or (
        f"window for task {window_task!r}")
    if task_id is not None and window_task != task_id:
        problems.append(
            f"window {label!r} belongs to task {window_task!r}, "
            f"not {task_id!r}")
    if episode_id is not None and window_episode != episode_id:
        problems.append(
            f"window {label!r} belongs to episode {window_episode!r}, "
            f"not {episode_id!r}")
    if strategy_id is not None and window_strategy != strategy_id:
        problems.append(
            f"window {label!r} is about strategy {window_strategy!r}, "
            f"not {strategy_id!r}")
    if (round_index is not None and window_round is not None
            and window_round != round_index):
        problems.append(
            f"window {label!r} is selection round {window_round}, "
            f"not round {round_index}")
    return problems


@dataclass
class WindowAttempt:
    """One real execution attempt inside a window."""

    action_id: str
    execution_id: Optional[str] = None
    action_status: str = "running"
    measurement_scope: str = "attempt"
    cost: Optional[CostVector] = None
    cost_measured: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "execution_id": self.execution_id,
            "action_status": self.action_status,
            "measurement_scope": self.measurement_scope,
            "cost": (self.cost.to_dict() if self.cost is not None else None),
            "cost_measured": list(self.cost_measured),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class AuxiliaryAction:
    """A real action inside the window that the prediction does NOT cover."""

    action_id: str
    action_type: str
    action_status: str = "completed"
    rollup: str = "own"
    cost: Optional[CostVector] = None
    cost_measured: List[str] = field(default_factory=list)
    counted_in_ledger: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_status": self.action_status,
            "rollup": self.rollup,
            "cost": (self.cost.to_dict() if self.cost is not None else None),
            "cost_measured": list(self.cost_measured),
            "counted_in_ledger": bool(self.counted_in_ledger),
            "note": ("auxiliary overhead: real spend, counted in the budget "
                     "ledger, NOT part of the predicted scope"),
        }


@dataclass
class StrategyExecutionWindow:
    """A real strategy execution window, derived from the action log."""

    window_id: str
    task_id: str
    episode_id: Optional[str]
    strategy_id: Optional[str]
    #: Which SELECTION ROUND of this (task, episode, strategy) the window
    #: is (M4). ``None`` = a legacy/round-less window. The same strategy
    #: chosen again later (possibly under a different config) is a
    #: DIFFERENT round and a different evaluation sample — aggregating them
    #: would average a config that failed into one that succeeded.
    round_index: Optional[int] = None
    in_scope_action_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_IN_SCOPE_ACTION_TYPES))
    auxiliary_action_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_AUXILIARY_ACTION_TYPES))
    attempts: List[WindowAttempt] = field(default_factory=list)
    auxiliary_actions: List[AuxiliaryAction] = field(default_factory=list)
    scope_rationale: str = ""
    comparable: bool = False
    not_comparable_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def n_unfinished(self) -> int:
        """In-scope attempts that have not ENDED (status ``running``).

        A window with any unfinished attempt is not a completed scope, so it
        is never ``comparable``: there are no final numbers to score
        against.
        """
        return sum(1 for a in self.attempts if a.action_status == "running")

    @property
    def attempt_cost(self) -> Dict[str, Any]:
        """Attempt cost over the window's in-scope attempts (see
        :func:`_aggregate`)."""
        return _aggregate([a.cost for a in self.attempts])

    @property
    def auxiliary_cost(self) -> Dict[str, Any]:
        """Auxiliary overhead cost (``rollup="own"`` actions only).

        ``rollup="reference"`` actions are excluded: their cost references a
        child already counted, and summing them again would double-count.
        """
        return _aggregate([a.cost for a in self.auxiliary_actions
                           if a.rollup == "own"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "strategy_id": self.strategy_id,
            "round_index": self.round_index,
            "in_scope_action_types": list(self.in_scope_action_types),
            "auxiliary_action_types": list(self.auxiliary_action_types),
            "n_attempts": self.n_attempts,
            "n_unfinished": self.n_unfinished,
            "attempts": [a.to_dict() for a in self.attempts],
            "auxiliary_actions": [a.to_dict() for a in self.auxiliary_actions],
            "attempt_cost": self.attempt_cost,
            "auxiliary_cost": self.auxiliary_cost,
            "scope_rationale": self.scope_rationale,
            "comparable": bool(self.comparable),
            "not_comparable_reasons": list(self.not_comparable_reasons),
            "notes": list(self.notes),
        }


def _aggregate(costs: Sequence[Optional[CostVector]]) -> Dict[str, Any]:
    """Per-dimension total over costs, with the measurement state reported.

    For each dimension:

    - ``total``: the sum over the items that MEASURED it, or ``None`` when
      none did;
    - ``n_measured``: how many items measured it;
    - ``n_items``: how many items exist;
    - ``complete``: True only when EVERY item measured it.

    An incomplete total is reported as ``partial=True`` alongside the count,
    never silently presented as the whole picture — and never collapsed to
    ``None`` either, which would throw away real observations. A dimension
    no item measured stays ``None``: unknown, not a zero.
    """
    dims: Dict[str, Any] = {}
    total: Dict[str, float] = {dim: 0.0 for dim in COST_DIMENSIONS}
    n_measured: Dict[str, int] = {dim: 0 for dim in COST_DIMENSIONS}
    accumulate_measured_costs([c for c in costs if c is not None],
                              total, n_measured)
    for dim in COST_DIMENSIONS:
        dims[dim] = {
            "total": (round(total[dim], 6) if n_measured[dim] else None),
            "n_measured": n_measured[dim],
            "n_items": len(costs),
            "complete": bool(costs) and n_measured[dim] == len(costs),
            "partial": bool(n_measured[dim])
                       and n_measured[dim] != len(costs),
        }
    return dims


def build_execution_window(actions: Sequence[Any], *, task_id: str,
                           episode_id: Optional[str],
                           strategy_id: Optional[str] = None,
                           in_scope_action_types: Optional[Sequence[str]] = None,
                           auxiliary_action_types: Optional[Sequence[str]] = None,
                           round_index: Optional[int] = None,
                           round_boundaries: Optional[Sequence[Any]] = None,
                           ) -> StrategyExecutionWindow:
    """Derive a window from REAL action records (see module docstring).

    ``actions`` are :class:`~or_harness.world_model.actions.ActionRecord`
    objects (or anything with the same attributes). Hypothetical actions are
    excluded outright: an imagined action is not part of a real window.

    ``round_index`` (M4) selects ONE selection round of this (task,
    episode, strategy). Rounds are delimited by ``round_boundaries`` — the
    recorded ``select_strategy`` choice actions of this episode, in time
    order; a new choice of the SAME strategy starts a new round (the agent
    re-decided), and so does a choice of a different strategy in between
    (the strategy was switched away and back). ``round_index=None`` keeps
    the legacy whole-episode aggregation (all rounds in one window), which
    stays readable but is NOT a single evaluation sample when the strategy
    was chosen more than once.

    The window's ``comparable`` flag is the guard the contract needs: a
    window with no in-scope executed attempt, or whose in-scope actions were
    only reported rather than executed, is NOT comparable, and a
    window-scope prediction may then not be scored.
    """
    in_scope = list(in_scope_action_types or DEFAULT_IN_SCOPE_ACTION_TYPES)
    auxiliary = list(auxiliary_action_types
                     or DEFAULT_AUXILIARY_ACTION_TYPES)
    unknown_types = [t for t in in_scope + auxiliary
                     if t not in WINDOW_ACTION_TYPES]
    if unknown_types:
        raise ValueError(
            f"unknown window action type(s) {sorted(set(unknown_types))}; "
            f"expected a subset of {WINDOW_ACTION_TYPES}")
    overlap = set(in_scope) & set(auxiliary)
    if overlap:
        raise ValueError(
            f"action types {sorted(overlap)} cannot be both in scope and "
            "auxiliary: the scope of a prediction is one or the other")

    window = StrategyExecutionWindow(
        window_id=window_id_for(task_id, episode_id, strategy_id,
                                round_index),
        task_id=task_id,
        episode_id=episode_id,
        strategy_id=strategy_id,
        round_index=round_index,
        in_scope_action_types=in_scope,
        auxiliary_action_types=auxiliary,
    )
    window.scope_rationale = (
        "in-scope action types "
        f"{in_scope} are what this prediction speaks about; "
        f"{auxiliary} are real auxiliary overhead, reported separately and "
        "never folded into the predicted scope")

    candidates = []
    for action in actions or []:
        if getattr(action, "task_id", None) not in (None, task_id):
            continue
        if episode_id is not None \
                and getattr(action, "episode_id", None) != episode_id:
            continue
        if getattr(action, "source", "executed") == "hypothetical":
            continue
        candidates.append(action)
    candidates.sort(key=lambda a: (getattr(a, "started_at", 0.0) or 0.0,
                                   getattr(a, "action_id", "")))

    # Round delimitation (M4): the episode's recorded select_strategy
    # choices, in time order, cut the timeline into selection rounds. A
    # round N window covers the actions from the Nth choice of THIS
    # strategy up to (excluding) the next choice of any strategy.
    if round_index is not None:
        if round_boundaries is None:
            round_boundaries = [
                a for a in candidates
                if getattr(a, "action_type", "") == "select_strategy"]
        choices = sorted(
            [a for a in round_boundaries
             if getattr(a, "action_type", "") == "select_strategy"],
            key=lambda a: (getattr(a, "started_at", 0.0) or 0.0,
                           getattr(a, "action_id", "")))
        # The choices OF THIS STRATEGY number the rounds; a choice of
        # another strategy in between ENDS the current round (the strategy
        # was switched away).
        round_starts: List[Any] = []
        for choice in choices:
            chosen_strategy = ((choice.params or {}).get("strategy_id")
                               or (choice.outcome or {}).get(
                                   "selected", {}).get("strategy_id"))
            if chosen_strategy == strategy_id:
                round_starts.append(choice)
        if round_index >= len(round_starts):
            window.not_comparable_reasons.append(
                f"round {round_index} does not exist: this strategy was "
                f"chosen {len(round_starts)} time(s) in this episode")
            window.comparable = False
            return window
        start = round_starts[round_index]
        end = None
        for choice in choices:
            if (getattr(choice, "started_at", 0.0),
                    getattr(choice, "action_id", "")) > \
                    (getattr(start, "started_at", 0.0),
                     getattr(start, "action_id", "")):
                end = choice
                break
        def _in_round(action: Any) -> bool:
            key = (getattr(action, "started_at", 0.0) or 0.0,
                   getattr(action, "action_id", ""))
            start_key = (getattr(start, "started_at", 0.0) or 0.0,
                         getattr(start, "action_id", ""))
            if key < start_key:
                return False
            if end is None:
                return True
            end_key = (getattr(end, "started_at", 0.0) or 0.0,
                       getattr(end, "action_id", ""))
            return key < end_key
        candidates = [a for a in candidates if _in_round(a)]
        window.notes.append(
            f"selection round {round_index} of strategy {strategy_id}: "
            f"actions from choice action {getattr(start, 'action_id', '')} "
            f"up to the next recorded choice"
            + (f" ({getattr(end, 'action_id', '')})" if end is not None
               else " (end of episode)"))

    strategy_mismatch = 0
    for action in candidates:
        action_type = getattr(action, "action_type", "")
        params = dict(getattr(action, "params", None) or {})
        action_strategy = params.get("strategy_id")
        # Strategy filter: a window is about ONE strategy. An action naming
        # a different strategy belongs to that strategy's window.
        if strategy_id is not None and action_strategy is not None \
                and action_strategy != strategy_id:
            strategy_mismatch += 1
            continue
        cost = getattr(action, "cost", None)
        measured = (sorted(cost.measured_dims())
                    if isinstance(cost, CostVector) else [])
        if action_type in in_scope:
            window.attempts.append(WindowAttempt(
                action_id=getattr(action, "action_id", ""),
                execution_id=getattr(action, "linked_execution_id", None),
                action_status=getattr(action, "status", "running"),
                measurement_scope="attempt",
                cost=cost,
                cost_measured=measured,
                started_at=getattr(action, "started_at", None),
                ended_at=getattr(action, "ended_at", None),
            ))
        elif action_type in auxiliary:
            window.auxiliary_actions.append(AuxiliaryAction(
                action_id=getattr(action, "action_id", ""),
                action_type=action_type,
                action_status=getattr(action, "status", "completed"),
                rollup=getattr(action, "rollup", "own"),
                cost=cost,
                cost_measured=measured,
            ))

    # Comparability is decided from what was actually found, and every
    # refusal carries its reason. ALL of the checks below must pass: a
    # single unfinished attempt is enough to make the whole window
    # unscorable, because its total is not final yet.
    reasons: List[str] = []
    if not window.attempts:
        reasons.append(
            "no in-scope executed attempt in this window: a window-scope "
            "prediction cannot be compared against nothing")
    else:
        without_execution = [a for a in window.attempts
                             if not a.execution_id]
        if without_execution:
            reasons.append(
                f"{len(without_execution)} of {len(window.attempts)} "
                "in-scope attempt(s) have no linked execution: the scope is "
                "not backed by a real record")
        unfinished = [a for a in window.attempts
                      if a.action_status == "running"]
        if unfinished:
            reasons.append(
                f"{len(unfinished)} of {len(window.attempts)} in-scope "
                "attempt(s) have not ended (status 'running'): a window "
                "that is still running has no final numbers, so it cannot "
                "be compared against")
        if len(window.attempts) > 1:
            window.notes.append(
                f"{len(window.attempts)} attempts in this window: attempt "
                "costs are reported per attempt, and a retry count of the "
                "window is NOT the same as one attempt's retries")
    if not window.auxiliary_actions:
        window.notes.append(
            "no auxiliary action was recorded in this window; the window "
            "therefore covers the solve attempt(s) only — it is NOT "
            "evidence that modeling and repair cost nothing")
    window.comparable = not reasons
    window.not_comparable_reasons = reasons
    if strategy_mismatch:
        window.notes.append(
            f"{strategy_mismatch} action(s) naming another strategy were "
            "excluded from this window")
    return window


def window_from_records(harness, task_id: str, episode_id: Optional[str],
                        strategy_id: Optional[str] = None,
                        round_index: Optional[int] = None, **kwargs
                        ) -> StrategyExecutionWindow:
    """Convenience wrapper over ``harness.actions.query``.

    Kept here (not in the API layer) so the derivation rule has one home;
    the API exposes the same thing as ``ORHarness.strategy_execution_window``.
    """
    actions = harness.actions.query(task_id=task_id, episode_id=episode_id)
    return build_execution_window(actions, task_id=task_id,
                                  episode_id=episode_id,
                                  strategy_id=strategy_id,
                                  round_index=round_index, **kwargs)


def window_ref(window: StrategyExecutionWindow) -> Dict[str, Any]:
    """The minimal window reference a prediction carries.

    Deliberately small: the prediction references the window, it does not
    copy it — the action log remains the single source of truth.
    """
    return {
        "window_id": window.window_id,
        "task_id": window.task_id,
        "episode_id": window.episode_id,
        "strategy_id": window.strategy_id,
        "round_index": window.round_index,
        "n_attempts": window.n_attempts,
        "n_unfinished": window.n_unfinished,
        "in_scope_action_types": list(window.in_scope_action_types),
        "comparable": bool(window.comparable),
        "not_comparable_reasons": list(window.not_comparable_reasons),
    }

