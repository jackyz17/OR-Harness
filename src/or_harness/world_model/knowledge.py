"""M6 knowledge-evolution prediction: targets, structural need, and value.

PURE computation plus frozen contract constants. This module never calls a
model, never writes a bank, and never mutates an entry. Three
responsibilities:

1. :func:`learning_needs` — the framework-side, TRANSPARENT HEURISTIC that
   proposes which knowledge targets a candidate action could plausibly
   affect. These are candidate TARGETS (what could be learned), never
   value estimates: the model supplies the direction, the framework
   validates it. Honesty: the output is labelled
   ``heuristic_uncalibrated`` and must not be described as a calibrated
   expected knowledge value.
2. :func:`knowledge_value` — the ``delta * K`` term's K. Assembled from a
   structural need (framework), the predicted change direction (model), and
   a FRAMEWORK-side class reliability (never the model's own uncertainty or
   self-scored value). Returns ``None`` when no justified value exists.
3. :func:`validate_knowledge_changes` — structural validation of the
   model's raw payload plus the TIERED target check: an ``existing_entry``
   target must name an entry that really exists in the frozen knowledge
   view, while a ``hypothesis`` target is allowed but must carry an
   observable expectation and a check condition.

Design disciplines encoded here:

- **Unknown is not zero, and never "best".** A missing K yields NO positive
  reward (the decision degenerates to the solve utility), which is the
  opposite of the conservative treatment of unknown RISK. Unknown upside
  must never be rewarded — that would prefer the least-supported action.
- **The model cannot raise its own value.** ``uncertainty`` and any
  self-scored ``expected_knowledge_value`` are recorded only; K's magnitude
  comes from framework-side quantities.
- **Reuse basis counts INDEPENDENT TASKS, not executions.** Repeating one
  task inflates ``cell.n`` without adding cross-task reuse evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from or_harness.core.schema import min_interval_width

# ---------------------------------------------------------------------------
# frozen contract vocabulary (mirrored in the provider's system prompt)
# ---------------------------------------------------------------------------

#: What a predicted knowledge change DOES to its target.
CHANGE_KINDS = ("adds_evidence", "supports", "revises", "refutes",
                "candidate_forms", "narrows")

#: When the change is expected to become observable.
HORIZONS = ("after_execution", "after_consolidation")

#: Kinds of standing condition a predicted change depends on.
PRECONDITION_KINDS = ("independent_replication", "contrast_execution",
                      "verification_check")
#: ``additional_measurement:<dimension>`` is the fourth, open-ended form.
MEASUREMENT_PREFIX = "additional_measurement:"

#: Target tiers.
TARGET_KINDS = ("existing_entry", "hypothesis")

#: Identity label for the heuristic K (never present it as calibrated).
KNOWLEDGE_BASIS_HEURISTIC = "heuristic_uncalibrated"

#: Weights of the structural need's three components.
NEED_WEIGHTS = {"evidence_scarcity": 1.0 / 3.0,
                "interval_slack": 1.0 / 3.0,
                "reuse_demand": 1.0 / 3.0}

#: Divergence between an observed cell mean and a claim's expected quality
#: that makes a revision worth proposing (same 0.05 the maintenance layer
#: already uses for the same judgement).
REVISION_DIVERGENCE = 0.05

#: Per-precondition realizability discount.
PRECONDITION_DISCOUNT = 0.25


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def distinct_task_count(records: Iterable[Any]) -> int:
    """How many DISTINCT tasks the given records came from.

    Reuse basis must count independent tasks: five runs of one task are one
    replication, not five. Same expression the admission gate uses
    (``induction.py``: ``len({r.task_id for r in records})``)."""
    return len({getattr(r, "task_id", None) for r in records})


# ---------------------------------------------------------------------------
# structural target proposal (heuristic, framework-side)
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeTarget:
    """One candidate knowledge target for a CANDIDATE action.

    Carries what could be learned, never how valuable it is. The model
    picks the direction (``change``) from ``candidate_changes``; the
    structural need is computed separately by :func:`structural_need`."""

    kind: str  # existing_entry | hypothesis
    strategy_id: str
    cell_token: str = ""
    entry_id: Optional[str] = None
    candidate_changes: List[str] = field(default_factory=list)
    reason: str = ""
    #: Structural inputs frozen at proposal time (used by
    #: :func:`structural_need`); None where the quantity does not apply.
    support_n: Optional[int] = None
    interval_width: Optional[float] = None
    distinct_tasks: Optional[int] = None
    expected_quality_hat: Optional[float] = None
    #: The claim's own interval ``(lo, hi)``. A new observation inside it
    #: SUPPORTS the claim; one outside it REFUTES it — this is the
    #: judgement the promise/shortfall evaluation needs, and it reuses the
    #: claim's own stated range instead of inventing a tolerance.
    quality_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "strategy_id": self.strategy_id,
            "cell_token": self.cell_token,
            "entry_id": self.entry_id,
            "candidate_changes": list(self.candidate_changes),
            "reason": self.reason,
            "support_n": self.support_n,
            "interval_width": self.interval_width,
            "distinct_tasks": self.distinct_tasks,
            "expected_quality_hat": self.expected_quality_hat,
            "quality_interval": (list(self.quality_interval)
                                 if self.quality_interval is not None
                                 else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeTarget":
        raw_interval = data.get("quality_interval")
        return cls(
            kind=str(data.get("kind", "hypothesis")),
            strategy_id=str(data.get("strategy_id", "")),
            cell_token=str(data.get("cell_token", "")),
            entry_id=data.get("entry_id"),
            candidate_changes=[str(c) for c in
                               (data.get("candidate_changes") or [])],
            reason=str(data.get("reason", "")),
            support_n=data.get("support_n"),
            interval_width=data.get("interval_width"),
            distinct_tasks=data.get("distinct_tasks"),
            expected_quality_hat=data.get("expected_quality_hat"),
            quality_interval=((float(raw_interval[0]),
                               float(raw_interval[1]))
                              if isinstance(raw_interval, (list, tuple))
                              and len(raw_interval) == 2 else None),
        )


def _cell_token_from_snapshot(snapshot) -> str:
    """The structural cell of the snapshot's task, when derivable.

    ``problem_state.profile`` is a ``ProblemProfile`` dict; its group key is
    the cell identity the coverage view is keyed by."""
    from or_harness.core.schema import ProblemProfile, group_key
    profile = (getattr(snapshot, "problem_state", None) or {}).get("profile")
    if not isinstance(profile, dict):
        return ""
    try:
        return group_key(ProblemProfile.from_dict(profile))
    except Exception:
        return ""


def _knowledge_entries(snapshot) -> Dict[str, List[Dict[str, Any]]]:
    coverage = getattr(snapshot, "coverage", None) or {}
    layers = coverage.get("knowledge_layers") or {}
    return {
        "verified": list(layers.get("verified") or []),
        "legacy_unknown": list(layers.get("legacy_unknown") or []),
        "unverified": list(layers.get("unverified") or []),
    }


def learning_needs(snapshot, action_spec,
                   records: Optional[Sequence[Any]] = None
                   ) -> List[KnowledgeTarget]:
    """Candidate knowledge targets for ONE candidate action (heuristic).

    Framework-side and model-free: derived from the frozen snapshot's
    coverage view plus the cell's real evidence records. An empty list is a
    first-class answer — it means the framework found no structural reason
    to expect a knowledge change, and the model's knowledge predictions
    (if any) are then rejected as unresolved rather than trusted.

    ``records`` are the cell's supporting ExecutionRecords (obtainable via
    ``ConditionalStats.evidence(profile, strategy_id)``); they are needed
    because ``GroupStats`` carries no distinct-task count.
    """
    strategy_id = getattr(action_spec, "strategy_id", None) or ""
    targets: List[KnowledgeTarget] = []
    if not strategy_id:
        return targets
    cell_token = _cell_token_from_snapshot(snapshot)
    layers = _knowledge_entries(snapshot)
    coverage = getattr(snapshot, "coverage", None) or {}
    cells = coverage.get("cell_statistics") or {}
    cell = cells.get(strategy_id) or {}
    n_exec = int(cell.get("n") or 0)
    n_tasks = (distinct_task_count(records) if records is not None
               else None)
    mean_quality = cell.get("mean_quality")
    fail_rate = cell.get("fail_rate")

    def _entry_target(entry: Dict[str, Any], changes: List[str],
                      reason: str, width: Optional[float],
                      interval: Optional[Tuple[float, float]] = None
                      ) -> KnowledgeTarget:
        return KnowledgeTarget(
            kind="existing_entry",
            strategy_id=str(entry.get("strategy_id") or strategy_id),
            cell_token=cell_token,
            entry_id=entry.get("entry_id"),
            candidate_changes=list(changes),
            reason=reason,
            support_n=int(entry.get("support_n") or 0),
            interval_width=width,
            distinct_tasks=n_tasks,
            expected_quality_hat=entry.get("expected_quality_hat"),
            quality_interval=interval,
        )

    # (1) Unverified candidates: real evidence would support or refute them
    #     IMMEDIATELY (the execution itself is the evidence); whether the
    #     claim then survives induction is a separate, later question.
    for entry in layers["unverified"]:
        if str(entry.get("strategy_id")) != strategy_id:
            continue
        targets.append(_entry_target(
            entry, ["supports", "refutes"],
            "an unverified candidate covers this strategy: new evidence "
            "bears directly on whether it holds", None))

    # (2) Verified claims whose interval still has room to narrow: the floor
    #     drops as support grows (min_interval_width), so more independent
    #     evidence can legitimately tighten the claim.
    for entry in layers["verified"]:
        if str(entry.get("strategy_id")) != strategy_id:
            continue
        interval = entry.get("quality_interval") or [0.0, 1.0]
        width = float(interval[1]) - float(interval[0])
        support_n = int(entry.get("support_n") or 0)
        next_floor = min_interval_width(support_n + 1)
        as_range = (float(interval[0]), float(interval[1]))
        if width > next_floor + 1e-9:
            targets.append(_entry_target(
                entry, ["narrows", "revises", "refutes"],
                f"interval width {width:.3f} exceeds the floor allowed at "
                f"the next support level ({next_floor:.3f}): more evidence "
                "can legitimately tighten or move the claim", width,
                as_range))
        # (3) Observed quality has diverged from the claim.
        if (mean_quality is not None
                and entry.get("expected_quality_hat") is not None
                and abs(float(mean_quality)
                        - float(entry["expected_quality_hat"]))
                > REVISION_DIVERGENCE):
            if not any(t.entry_id == entry.get("entry_id")
                       for t in targets):
                targets.append(_entry_target(
                    entry, ["revises", "refutes"],
                    f"observed cell quality {float(mean_quality):.3f} "
                    f"diverges from the claim "
                    f"{float(entry['expected_quality_hat']):.3f} by more "
                    f"than {REVISION_DIVERGENCE}", width, as_range))

    # (4) An hypothesis for a strategy with SOME evidence but no claim
    #     covering it yet. Two sub-cases, both real:
    #
    #     - fewer than 2 INDEPENDENT tasks: the admission gate refuses, so
    #       more evidence (or a second task) is what would let a claim
    #       form;
    #     - 2+ independent tasks: the gate can now admit a claim, so the
    #       next induction is where one FORMS.
    #
    #     Both predict ``candidate_forms`` at the consolidation horizon,
    #     which is exactly the proposition an ordinary action should be
    #     able to anticipate. Distinct from the coverage gap
    #     (``strategies_without_evidence`` uses n == 0): here n >= 1, so
    #     something was really tried. Allowed with no existing entries at
    #     all, which keeps cold start workable once one task has run.
    covered = any(str(e.get("strategy_id")) == strategy_id
                  for layer in layers.values() for e in layer)
    if n_exec >= 1 and not covered:
        independent_enough = n_tasks is None or n_tasks >= 2
        if independent_enough:
            reason = (f"{n_exec} execution(s) from "
                      f"{'an unknown number of' if n_tasks is None else n_tasks} "
                      "independent task(s) with no claim covering them: the "
                      "admission gate's condition is met, so the next "
                      "induction is where a claim can FORM")
        else:
            reason = (f"{n_exec} execution(s) from {n_tasks} independent "
                      "task(s): below the admission gate's >=2-task "
                      "requirement, so a claim cannot form yet")
        targets.append(KnowledgeTarget(
            kind="hypothesis",
            strategy_id=strategy_id,
            cell_token=cell_token,
            candidate_changes=["candidate_forms", "adds_evidence"],
            reason=reason,
            support_n=n_exec,
            interval_width=None,
            distinct_tasks=n_tasks,
            expected_quality_hat=mean_quality,
        ))
    return targets


# ---------------------------------------------------------------------------
# structural need (framework-side, transparent)
# ---------------------------------------------------------------------------


def structural_need(*, support_n: Optional[int],
                    interval_width: Optional[float],
                    distinct_tasks: Optional[int]) -> Optional[float]:
    """How much the harness structurally NEEDS more evidence here, in [0,1].

    Three components, each independently defensible:

    - ``evidence_scarcity``: thinner support => larger marginal gain.
    - ``interval_slack``: how far the claim's interval sits above the
      HONEST floor for its support level (``min_interval_width``). At the
      floor the claim is already as tight as its evidence permits (0); a
      fully unknown interval has the most room (1). The floor is a
      function of n and DROPS as n grows, so this is monotonically
      meaningful rather than the sign-broken ratio it replaced.
    - ``reuse_demand``: how much INDEPENDENT cross-task evidence the cell
      already carries. Executions of a single task do not count.

    Returns ``None`` only when nothing structural is known at all.
    """
    if support_n is None and interval_width is None \
            and distinct_tasks is None:
        return None
    scarcity = 0.0
    if support_n is not None:
        n = max(0, int(support_n))
        scarcity = 1.0 - n / (n + 2.0)
    slack = 0.0
    if interval_width is None:
        slack = 1.0  # no claim yet: everything about it is unresolved
    else:
        floor = min_interval_width(int(support_n or 0))
        room = 1.0 - floor
        slack = (0.0 if room <= 1e-9
                 else _clamp((float(interval_width) - floor) / room))
    reuse = 0.0
    if distinct_tasks is not None:
        tasks = max(0, int(distinct_tasks))
        reuse = tasks / (tasks + 2.0)
    w = NEED_WEIGHTS
    return _clamp(w["evidence_scarcity"] * scarcity
                  + w["interval_slack"] * slack
                  + w["reuse_demand"] * reuse)


# ---------------------------------------------------------------------------
# target resolution and validation
# ---------------------------------------------------------------------------


def _target_index(targets: Sequence[KnowledgeTarget]
                  ) -> Tuple[set, set]:
    """The entry ids and strategy ids a decision actually proposed."""
    entries = {t.entry_id for t in targets if t.entry_id}
    strategies = {t.strategy_id for t in targets}
    return entries, strategies
def validate_knowledge_changes(payload: Any,
                               targets: Optional[Sequence[KnowledgeTarget]] = None,
                               action_strategy_id: Optional[str] = None,
                               ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Structural + tiered validation of the model's ``knowledge_changes``.

    Returns ``(accepted, problems)``. ``problems`` is non-empty when the
    payload is malformed (the caller then rejects the WHOLE prediction, the
    same reject-all semantics the rest of the contract uses). A target that
    is well-formed but not backed by this decision's proposal set is NOT a
    malformed payload — it is dropped and keeps a ``rejected`` note so a
    model cannot invent knowledge entries out of thin air.

    The two target kinds are checked DIFFERENTLY on purpose, because they
    make different claims:

    - **existing_entry** asserts something about a claim that already
      exists, so its ``entry_id`` must be one this decision actually
      surfaced. A model may not invent knowledge.
    - **hypothesis** asserts that a claim is worth TESTING. It is not a
      statement about existing knowledge, so requiring it to be in the
      proposal set would confine predictions to whatever the framework's
      heuristics happened to propose — and at cold start, when no structural
      target exists at all, that set is empty and every sound hypothesis
      would be discarded. It must instead name the strategy actually under
      test (``action_strategy_id``), and the structural layer still demands
      an expected observation and a check condition.

    Allowing a hypothesis is NOT allowing knowledge to be published: the
    prediction never enters the bank, and admission verification still
    gates everything downstream.

    ``targets=None`` performs STRUCTURAL validation only. That split is
    deliberate: the request contract can check the shape of
    ``knowledge_changes`` before any proposal set exists, while the tiered
    existence check needs the decision's proposals and therefore runs where
    they are known.
    """
    problems: List[str] = []
    if not isinstance(payload, list):
        return [], ["knowledge_changes must be a JSON array"]
    accepted: List[Dict[str, Any]] = []
    if targets is None:
        entry_ids: set = set()
        strategy_ids: set = set()
        tiered = False
    else:
        entry_ids, strategy_ids = _target_index(targets)
        tiered = True
    for index, item in enumerate(payload):
        where = f"knowledge_changes[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{where} must be a JSON object")
            continue
        change = item.get("change")
        if change not in CHANGE_KINDS:
            problems.append(f"{where}.change {change!r} not in "
                            f"{list(CHANGE_KINDS)}")
            continue
        horizon = item.get("horizon")
        if horizon not in HORIZONS:
            problems.append(f"{where}.horizon {horizon!r} not in "
                            f"{list(HORIZONS)}")
            continue
        target = item.get("target")
        if not isinstance(target, dict):
            problems.append(f"{where}.target must be a JSON object")
            continue
        kind = target.get("kind")
        if kind not in TARGET_KINDS:
            problems.append(f"{where}.target.kind {kind!r} not in "
                            f"{list(TARGET_KINDS)}")
            continue
        if not any(target.get(k) for k in
                   ("entry_id", "strategy_id", "cell_token")):
            problems.append(f"{where}.target needs at least one of "
                            "entry_id / strategy_id / cell_token")
            continue
        if kind == "hypothesis":
            if not isinstance(item.get("expected_observation"), dict) \
                    or not item["expected_observation"]:
                problems.append(
                    f"{where}: a hypothesis target requires a non-empty "
                    "expected_observation (what would be seen if it held)")
                continue
            if not isinstance(item.get("check_condition"), str) \
                    or not item["check_condition"].strip():
                problems.append(
                    f"{where}: a hypothesis target requires a "
                    "check_condition (how the observation would be judged)")
                continue
        preconditions = item.get("preconditions")
        if preconditions is not None:
            if not isinstance(preconditions, list) or not all(
                    isinstance(p, str) for p in preconditions):
                problems.append(f"{where}.preconditions must be a list of "
                                "strings")
                continue
            bad = [p for p in preconditions
                   if p not in PRECONDITION_KINDS
                   and not p.startswith(MEASUREMENT_PREFIX)]
            if bad:
                problems.append(f"{where}.preconditions has unknown "
                                f"entries {bad}")
                continue
        uncertainty = item.get("uncertainty")
        if uncertainty is not None:
            if isinstance(uncertainty, bool) \
                    or not isinstance(uncertainty, (int, float)):
                problems.append(f"{where}.uncertainty must be a number")
                continue
            if not (0.0 <= float(uncertainty) <= 1.0):
                problems.append(f"{where}.uncertainty must be in [0, 1]")
                continue
        extra = item.get("expected_extra_cost")
        if extra is not None and not isinstance(extra, dict):
            problems.append(f"{where}.expected_extra_cost must be a JSON "
                            "object of cost dimensions")
            continue
        basis = item.get("prediction_basis")
        if basis is not None and (not isinstance(basis, list)
                                  or not all(isinstance(b, str)
                                             for b in basis)):
            problems.append(f"{where}.prediction_basis must be a list of "
                            "strings")
            continue
        # Tiered target check (only when a proposal set was supplied). A
        # model may not invent EXISTING knowledge: an existing_entry target
        # must name an entry this decision actually surfaced. A hypothesis
        # is held to the different standard its claim deserves — see the
        # docstring — namely that it names the strategy under test.
        if not tiered:
            accepted.append(dict(item))
            continue
        if kind == "existing_entry":
            if not (target.get("entry_id") in entry_ids
                    and target.get("strategy_id") in strategy_ids):
                item = dict(item)
                item["rejected"] = ("target_unresolved: not among this "
                                    "decision's proposed targets")
                accepted.append(item)
                continue
        else:
            under_test = strategy_ids
            if action_strategy_id:
                # The strategy this action actually runs is always a
                # legitimate subject, whether or not a heuristic proposed
                # it — otherwise a cold start would discard every sound
                # hypothesis about the action's own strategy.
                under_test = set(under_test) | {action_strategy_id}
            if target.get("strategy_id") not in under_test:
                item = dict(item)
                item["rejected"] = ("target_unresolved: strategy is not "
                                    "under test in this decision")
                accepted.append(item)
                continue
        accepted.append(dict(item))
    return accepted, problems


# ---------------------------------------------------------------------------
# knowledge value K (framework-side magnitude, model-side direction)
# ---------------------------------------------------------------------------


def _targets_by_key(targets: Sequence[KnowledgeTarget]
                    ) -> Dict[Tuple[str, str], KnowledgeTarget]:
    out: Dict[Tuple[str, str], KnowledgeTarget] = {}
    for t in targets:
        out[("existing_entry", t.entry_id or t.strategy_id)] = t
        out[("hypothesis", t.strategy_id)] = t
    return out


def _lookup_target(targets: Sequence[KnowledgeTarget],
                   item: Dict[str, Any]) -> Optional[KnowledgeTarget]:
    target = item.get("target") or {}
    kind = target.get("kind")
    if kind == "existing_entry":
        for t in targets:
            if (t.kind == "existing_entry"
                    and t.entry_id
                    and t.entry_id == target.get("entry_id")
                    and t.strategy_id == target.get("strategy_id")):
                return t
    else:
        for t in targets:
            if t.kind == "hypothesis" \
                    and t.strategy_id == target.get("strategy_id"):
                return t
    return None


def precondition_realizability(item: Dict[str, Any]
                               ) -> Tuple[Optional[float], List[str]]:
    """How much of the predicted change is actually realizable, plus notes.

    ``None`` means "do not grant the value": a standing condition needs
    work that was never costed, or the change is not something this action
    can realize. Extra cost that IS declared (``expected_extra_cost``) is
    accepted because cost predictions are separately calibrated — unlike a
    value self-score, which is not.
    """
    notes: List[str] = []
    preconditions = [str(p) for p in (item.get("preconditions") or [])]
    if not preconditions:
        return 1.0, notes
    unsatisfied = []
    for precondition in preconditions:
        if precondition == "verification_check":
            unsatisfied.append(precondition)
        elif precondition == "contrast_execution":
            unsatisfied.append(precondition)
        elif precondition == "independent_replication":
            unsatisfied.append(precondition)
        elif precondition.startswith(MEASUREMENT_PREFIX):
            unsatisfied.append(precondition)
    if not unsatisfied:
        return 1.0, notes
    horizon = item.get("horizon")
    if horizon == "after_consolidation" \
            and not isinstance(item.get("expected_extra_cost"), dict):
        notes.append(
            "after-consolidation change depends on "
            f"{sorted(unsatisfied)} but no expected_extra_cost is declared: "
            "the extra work is uncosted, so no knowledge value is granted")
        return None, notes
    realizability = max(0.0, 1.0 - PRECONDITION_DISCOUNT
                        * len(unsatisfied))
    notes.append(
        f"realizability {realizability:.2f}: {len(unsatisfied)} standing "
        f"condition(s) still to satisfy {sorted(unsatisfied)}; the value is "
        "discounted, never granted in full")
    return realizability, notes


def knowledge_value(targets: Sequence[KnowledgeTarget],
                    predictions: Sequence[Any],
                    *, reliability=None
                    ) -> Tuple[Optional[float], Dict[str, Any]]:
    """K for one path: ``min(1, max over targets (need x fulfilment))``.

    ``targets`` are the structural proposals of the decision's ROOT
    candidate(s). ``predictions`` are the path's predictions. ``reliability``
    is a callable ``(change, horizon) -> Optional[float]`` supplying the
    FRAMEWORK-side class reliability; when it cannot supply a value the
    target contributes NOTHING (K is unknown), because the model's own
    ``uncertainty`` must never substitute for measured reliability.

    Returns ``(K, detail)``. ``K is None`` means "no justified value": the
    caller must then grant NO positive reward (not a full-weight reward),
    which is the opposite of the treatment of unknown risk.
    """
    detail: Dict[str, Any] = {
        "basis": KNOWLEDGE_BASIS_HEURISTIC,
        "targets": [],
        "value": None,
    }
    items: List[Tuple[Dict[str, Any], Any]] = []
    for prediction in predictions:
        if getattr(prediction, "status", None) != "valid":
            continue
        predicted = getattr(prediction, "predicted", None) or {}
        for item in predicted.get("knowledge_changes") or []:
            if isinstance(item, dict) and not item.get("rejected"):
                items.append((item, prediction))
    if not items:
        detail["note"] = ("no knowledge change predicted on this path; "
                          "no knowledge value is claimed")
        return None, detail

    best: Optional[float] = None
    seen: set = set()
    for item, prediction in items:
        target = _lookup_target(targets, item)
        record: Dict[str, Any] = {
            "change": item.get("change"),
            "horizon": item.get("horizon"),
            "prediction_id": getattr(prediction, "prediction_id", None),
            "target": dict(item.get("target") or {}),
        }
        if target is None:
            record["incomparable"] = ("target not among this decision's "
                                      "proposals")
            detail["targets"].append(record)
            continue
        # De-duplicate per (target identity, change, horizon): the same
        # proposition predicted twice in one path is ONE proposition.
        key = (target.entry_id or target.strategy_id,
               item.get("change"), item.get("horizon"))
        if key in seen:
            record["incomparable"] = "duplicate proposition on this path"
            detail["targets"].append(record)
            continue
        seen.add(key)
        need = structural_need(support_n=target.support_n,
                               interval_width=target.interval_width,
                               distinct_tasks=target.distinct_tasks)
        record["need"] = need
        if need is None:
            record["incomparable"] = ("no structural evidence about this "
                                      "target; no need is claimed")
            detail["targets"].append(record)
            continue
        effect = None
        if reliability is not None:
            effect = reliability(item.get("change"), item.get("horizon"))
        record["change_effect"] = effect
        if effect is None:
            record["incomparable"] = (
                "no measured reliability for this class of prediction; the "
                "model's own uncertainty is not a substitute — no "
                "knowledge value is granted")
            detail["targets"].append(record)
            continue
        realizability, notes = precondition_realizability(item)
        record["realizability"] = realizability
        if notes:
            record["notes"] = notes
        if realizability is None:
            record["incomparable"] = ("the predicted change is not "
                                      "realizable within this action's "
                                      "costed scope")
            detail["targets"].append(record)
            continue
        # The extra spend the predicted gain REQUIRES travels with the
        # value: a value whose enabling cost is never charged would let a
        # prediction buy influence it has not paid for. The caller charges
        # it (see planner.evaluate_path); this module only reports it.
        extra = item.get("expected_extra_cost")
        if isinstance(extra, dict) and extra:
            record["expected_extra_cost"] = {
                str(d): float(v) for d, v in extra.items()}
            detail.setdefault("expected_extra_cost", {})
            for dim, value in record["expected_extra_cost"].items():
                detail["expected_extra_cost"][dim] = (
                    detail["expected_extra_cost"].get(dim, 0.0) + value)
        contribution = _clamp(need * _clamp(float(effect))
                              * realizability)
        record["contribution"] = round(contribution, 6)
        detail["targets"].append(record)
        best = contribution if best is None else max(best, contribution)
    if best is not None:
        detail["value"] = round(best, 6)
        detail["note"] = ("max over targets of need x change_effect x "
                          "realizability; heuristic magnitude, not a "
                          "calibrated expected value")
    return best, detail
