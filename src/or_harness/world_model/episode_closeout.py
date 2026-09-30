"""Episode close-out: real outcomes, post-hoc evaluation, experience
calibration (world-model M4).

This module closes the loop the M3 service opened. A strategy-outcome
prediction (wm-so/1) was made BEFORE execution; M4 answers, AFTER the
episode really ended, three questions in order:

1. **What actually happened?** :func:`summarize_real_outcome` builds a
   traceable summary of one bound prediction's real execution window —
   the actual selection, the execution trajectory references, the
   validity/verification basis, the benefit-relevant observations, the
   real cost (scoped to what the prediction covered) and the observed
   risk events. Every field carries its own eligibility: evaluable,
   missing, unverified, scope-mismatched or identity-mismatched.
2. **How good was the prediction?** :func:`evaluate_strategy_prediction`
   compares the FROZEN prediction against that summary, field by field —
   benefit error only under the same metric/unit/baseline/scope, cost
   error per dimension only where both sides measured the same scope,
   Brier scores only for events with a known predicted probability AND a
   reliable binary label, interval coverage only where both were saved.
   The original prediction is never rewritten; the evaluation is a
   separate, append-only record.
3. **What has the model been worth historically?**
   :func:`build_calibration_summary` aggregates the evaluations of CLOSED
   episodes into a versioned summary — error statistics, interval
   coverage, event scores — grouped by protocol/metric/scope, with sample
   counts, distinct-episode counts and exclusion reasons. A group below
   the minimum sample threshold reports ``insufficient_evidence``, never
   a guessed reliability.

Design boundaries (the reasons this module is shaped the way it is):

- **Close-out is the single entry.** The summary evaluation and the
  calibration publication happen when the EPISODE ends, not per action:
  in-task feedback must not become in-task calibration. A window may
  terminate early (its facts are saved), but its aggregate evaluation is
  published only at close-out.
- **Unknown is never a truth.** A missing baseline, an unverified
  solution, an unobserved risk event, an identity field the action could
  not record — each is reported as such and EXCLUDED from the statistics
  it would pollute. No counterfactual label is fabricated for an
  unexecuted candidate.
- **The prediction stays frozen.** The evaluation references the
  prediction id; nothing on the prediction's predicted content is edited
  after the fact. A "better" prediction cannot be regenerated closer to
  the result.
- **Idempotent and recoverable.** Closing the same episode twice,
  re-reading after a restart, or re-binding produces the same facts once:
  the close-out record is keyed by (task, episode) and the evaluation
  records by prediction id.
- **No model calls, no solvers, no induction.** Close-out reads what was
  recorded. It never runs a strategy, never invokes the provider, never
  triggers knowledge induction — those are separate explicit actions.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.schema import is_finite_number as _finite
from or_harness.core.schema import (
    task_check_block,
    task_check_state,
    task_effective_quality,
)
from or_harness.world_model.attribution import (
    EVALUATION_DIMENSIONS,
    binding_attribution,
    blocked_dimensions,
    eligibility_for,
    method_deviation_note,
)

#: Version of the close-out record schema.
EPISODE_CLOSEOUT_VERSION = "wm-closeout/1"

#: Version of the calibration summary schema. ``wm-calib/2`` adds the
#: DIRECTED statistics (signed benefit error, log cost ratio), the two
#: counting units (observation units vs prediction-observation pairs), the
#: model-identity grouping key and the versioned risk-event vocabulary. A
#: v1 summary and a v2 summary count different things under the same event
#: names, so they are never pooled.
CALIBRATION_SUMMARY_VERSION = "wm-calib/2"

#: Version of the FRAMEWORK's risk-event vocabulary. Bumped whenever an
#: event's MEANING or its observation channel changes: a summary built
#: under one vocabulary must never be pooled with another, because the
#: same name would then count two different things. ``model_invalid`` used
#: to mean "the solver reported an error", which is not a modelling
#: verdict and was retired in v2.
EVENT_VOCABULARY_VERSION = "wm-events/2"

#: Terminal states an episode may be closed under. Only ``completed``
#: claims success; the others are honest endings, never dressed up.
EPISODE_TERMINAL_STATES = ("completed", "failed", "aborted",
                           "budget_exhausted")

#: Minimum resolved samples before a calibration group reports a
#: reliability figure (below it: ``insufficient_evidence``). Configurable
#: at build time; the effective value is recorded on the summary.
DEFAULT_MIN_CALIBRATION_SAMPLES = 5

#: Field eligibility values. ``evaluable`` is the only one that enters
#: statistics; every other value says WHY it does not. ``identity_mismatch``
#: is a KNOWN disagreement (the run was not the predicted run);
#: ``identity_unknown`` is an UNCONFIRMED field that still blocks the
#: dimension (the approach could not be confirmed). The two are different
#: facts and never share a label.
FIELD_ELIGIBILITY = ("evaluable", "missing", "unverified", "scope_mismatch",
                     "identity_mismatch", "identity_unknown",
                     "not_predicted", "unobserved", "unreliable_label")

#: The benefit metrics this build can actually OBSERVE from an execution.
#:
#: - ``normalized_objective_gap``: the solver's normalized gap (``1 -
#:   mip_gap``; an ``optimal`` status is a gap of 0).
#: - ``task_result_check_passed``: whether the ANSWER satisfied the TASK,
#:   taken from the execution's own ``check-task`` verdict (1.0 for
#:   ``passed``, 0.0 for a confirmed ``failed``, UNKNOWN for unchecked or
#:   ``insufficient``).
#:
#: They are DIFFERENT measurements of different things, so a prediction
#: declaring one is never scored against the other: "made progress" and
#: "passed the task check" are not the same claim, and the fact that both
#: land in [0,1] does not make them interchangeable.
OBSERVABLE_BENEFIT_METRIC = "normalized_objective_gap"
OBSERVABLE_COMPLETION_METRIC = "task_result_check_passed"

OBSERVABLE_BENEFIT_METRICS = (
    OBSERVABLE_BENEFIT_METRIC, OBSERVABLE_COMPLETION_METRIC)

#: Declared-metric spellings that map onto an observable metric. The
#: completion spellings are listed EXPLICITLY rather than folded into the
#: quality aliases: a caller that declares a completion rate must get the
#: completion observation, never the solver's gap silently relabelled.
_BENEFIT_METRIC_ALIASES = {
    "normalized_objective_gap": "normalized_objective_gap",
    "solution_quality": "normalized_objective_gap",
    "normalized_quality": "normalized_objective_gap",
    "normalized_gap": "normalized_objective_gap",
    "task_result_check_passed": "task_result_check_passed",
    "task_check_passed": "task_result_check_passed",
    "effective_completion": "task_result_check_passed",
    "completion_rate": "task_result_check_passed",
}

#: Risk-event names this build can actually OBSERVE, with the observation
#: UNIT each is counted in. Two units exist and they are never conflated:
#:
#: - ``execution``: the event is a property of ONE execution. The unit of
#:   observation is the execution itself, so a re-planning prediction bound
#:   to the same execution does NOT create a second observation.
#: - ``episode``: the event is a property of the whole episode (its budget
#:   ledger), so it is observed once per episode.
#:
#: An event outside this vocabulary has NO observation channel — its label
#: stays unknown and it is never scored, because "no failure log" is not
#: evidence that a business risk did not happen.
#:
#: Each name says WHAT IT OBSERVED, never WHY it happened. In particular
#: ``environment_failure`` / ``implementation_failure`` are the executor's
#: OWN recorded classification (``FailureRecord.error_class``); neither is
#: a statement about the mathematical model. ``solver_reported_infeasible``
#: is a pure observation of the solver's verdict, never by itself a failure
#: of the strategy: correctly diagnosing that the ORIGINAL problem is
#: infeasible is a valid result.
OBSERVABLE_RISK_EVENTS: Dict[str, Dict[str, Any]] = {
    "environment_failure": {
        "unit": "execution",
        "source": "ExecutionRecord.failures[].error_class == 'environment' "
                  "(sandbox policy, missing module, import error)",
        "applicability": "an in-scope execution of the bound prediction",
        "measured": "the executor recorded the failure class at run time",
        "note": "an execution whose failure carries NO error_class (a record "
                "written before the class existed) keeps the label unknown: "
                "the class is never re-derived from prose after the fact",
    },
    "implementation_failure": {
        "unit": "execution",
        "source": "ExecutionRecord.failures[].error_class == 'model' "
                  "(the harness's own script/stack failed)",
        "applicability": "an in-scope execution of the bound prediction",
        "measured": "the executor recorded the failure class at run time",
        "note": "this is an IMPLEMENTATION failure, NOT a mathematical "
                "modelling error; the two are never conflated",
    },
    "timeout": {
        "unit": "execution",
        "source": "ExecutionRecord.quality.status == 'timeout'",
        "applicability": "an in-scope execution of the bound prediction",
        "measured": "the executor's wall-clock/CPU limit fired",
        "note": "a timeout is not evidence that the strategy is wrong",
    },
    "solver_reported_infeasible": {
        "unit": "execution",
        "source": "ExecutionRecord.quality.status == 'infeasible'",
        "applicability": "an in-scope execution of the bound prediction",
        "measured": "the solver reported infeasibility",
        "note": "a REPORTED infeasibility is a fact about the solver's "
                "verdict, never automatically a strategy failure: correctly "
                "identifying that the original problem is infeasible is a "
                "VALID outcome. Whether the infeasibility was correct is a "
                "task-check question (task_check_failed), not this event's",
    },
    "task_check_failed": {
        "unit": "execution",
        "source": "execution_features.task_check.state",
        "applicability": "an in-scope execution carrying a task-result check",
        "measured": "the harness declared check bases and they ran on real "
                    "values",
        "note": "reflects the CHECK RESULT and nothing more: it says the "
                "declared bases did not hold, NOT that the model was wrong "
                "(the cause may be a misread task, an implementation bug, an "
                "unmet requirement, or a wrong reference). The check kind, "
                "its covered scope and its basis travel with the label",
    },
    "budget_exhausted": {
        "unit": "episode",
        "source": "the episode's declared budget view (measured real "
                  "consumption)",
        "applicability": "only a prediction whose DECLARED scope matches the "
                         "budget ledger's scope (the episode)",
        "measured": "every dimension the declared budget names is measured",
        "note": "the ledger is episode-scoped, so an attempt- or "
                "strategy-window-scope prediction has NO matching budget "
                "range: its label stays unknown with a scope_mismatch basis",
    },
}

#: Event names RETIRED in this vocabulary, and why. A prediction naming one
#: of these gets NO label and NO alias mapping: ``model_invalid`` used to be
#: derived from a solver ``error`` status, which is not a modelling verdict,
#: and ``no_feasible_solution`` conflated "reported infeasible" with "failed
#: to find a solution". Mapping them onto the new names would silently count
#: the old, wider meaning under a narrower label.
RETIRED_RISK_EVENTS: Dict[str, str] = {
    "model_invalid": "retired in wm-events/2: a solver error status or a "
                     "failed task check cannot establish that the MODEL was "
                     "wrong. Use task_check_failed for the check result; the "
                     "cause of a failure is the agent's diagnosis, not a "
                     "framework label",
    "no_feasible_solution": "retired in wm-events/2: ambiguous between a "
                            "REPORTED infeasibility (solver_reported_"
                            "infeasible) and a failure to find any feasible "
                            "solution. The two are different facts",
    "model_failure": "retired in wm-events/2: renamed implementation_failure, "
                     "which says what was observed (the harness's own code "
                     "failed) instead of implying a modelling error",
}


def _observable_benefit_metric(metric: Any) -> Optional[str]:
    """The canonical observable metric a declared metric maps to, or None."""
    name = str(metric or "").strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    return _BENEFIT_METRIC_ALIASES.get(name)


def _observe_completion(summary: Any, records: Sequence[Any]) -> None:
    """Observe ``task_result_check_passed`` from the executions themselves.

    The observation is the execution's OWN task-check verdict — the check
    the agent already ran. **No check is run here**, and no check is
    invented: a build that ran no check cannot report a completion, so an
    unchecked (or ``insufficient``) execution contributes NOTHING rather
    than a default zero.

    Mapping, and why it is this narrow:

    * ``passed`` -> 1.0 — a declared basis held on the recorded values.
    * ``failed`` -> 0.0 — the answer was CONFIRMED not to satisfy the task.
    * ``insufficient`` / no check -> UNKNOWN, excluded from the sample.
      "The check could not decide" is not a failure and not a success; the
      reviewer's rule is explicitly that it must not enter the error
      statistics.

    The window rule matches the quality channel: the LAST in-scope attempt
    that carries a verdict is the observation, declared before evaluation
    rather than chosen per result.
    """
    observed: List[float] = []
    unknown = 0
    for record in records:
        state = task_check_state(record)
        if state == "passed":
            observed.append(1.0)
        elif state == "failed":
            observed.append(0.0)
        else:
            unknown += 1
    if not observed:
        summary.eligibility["benefit"] = {
            "eligibility": "unobserved",
            "reason": ("no in-scope execution carries a task-check verdict: "
                       "completion is UNKNOWN (not 0.0) — run `check-task` "
                       "and it becomes observable"),
        }
        return
    summary.benefit = {
        "kind": "effective_completion",
        "metric": OBSERVABLE_COMPLETION_METRIC,
        "unit": "boolean",
        "observed": observed[-1],
        "rule": "last in-scope attempt carrying a task-check verdict",
        "n_observations": len(observed),
        "all_observations": observed,
        "source": ("the execution's own check-task verdict; no check is run "
                   "by the close-out"),
    }
    if unknown:
        summary.benefit["unknown_checks"] = unknown
        summary.benefit["unknown_note"] = (
            f"{unknown} in-scope execution(s) had no usable task-check "
            "verdict and contributed NOTHING to the sample: unchecked or "
            "insufficient validity is unknown, never a failure")
    if "benefit" not in summary.eligibility:
        summary.eligibility["benefit"] = {
            "eligibility": "evaluable",
            "reason": ("task-check verdict observed on the in-scope "
                       "execution(s)"),
        }


#: The scope the BUDGET LEDGER actually measures. ``budget.view`` aggregates
#: the whole episode (its own recorded attempts plus the prediction calls),
#: so its unit is the episode. A prediction can only be scored against
#: ``budget_exhausted`` when its declared scope equals this — and NO
#: prediction scope currently does (``attempt`` and ``strategy_window`` are
#: both narrower). This round deliberately does not build a per-attempt
#: budget system, so the event's FACT is still recorded (see
#: ``observe_episode_events``) while its Brier channel stays closed with an
#: explicit scope_mismatch basis rather than borrowing the episode verdict.
BUDGET_LEDGER_SCOPE = "episode"


def _budget_scope_matches(candidate: Any) -> bool:
    """Whether a prediction's declared scope matches the budget ledger."""
    return str(getattr(candidate, "scope", "attempt")) == BUDGET_LEDGER_SCOPE


def _normalize_event_name(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# retention / window / archive defaults (all configurable)
# ---------------------------------------------------------------------------

#: How many CLOSED task-episodes participate in the published calibration.
#: The window is the unit of the calibration sample set — it bounds both the
#: statistics and the work a close-out does, so prediction reads a small,
#: published summary instead of scanning the whole history.
DEFAULT_CALIBRATION_WINDOW = 50

#: Days a closed episode's ONLINE detail is kept so a LATE task check can
#: still land on it. This is a grace period for episodes that may still be
#: awaiting a check, NOT an unconditional extra retention for every episode:
#: an episode whose in-scope executions all carry a verdict leaves the
#: online set as soon as it drops out of the window.
DEFAULT_LATE_CHECK_GRACE_DAYS = 30.0

#: Archive directory name under the harness home.
ARCHIVE_DIRNAME = "calibration"

#: Archive capacity limits. All three are enforced (whichever bites first
#: evicts the OLDEST archive file), so the archive has a real, finite bound
#: rather than "whatever accumulates".
DEFAULT_ARCHIVE_MAX_FILE_BYTES = 64 * 1024 * 1024      # 64 MB per file
DEFAULT_ARCHIVE_MAX_TOTAL_BYTES = 1024 * 1024 * 1024   # 1 GB in total
DEFAULT_ARCHIVE_RETENTION_DAYS = 365.0

#: How many closed episodes may sit OUTSIDE the window (and past the grace
#: period) before an automatic archive pass runs after a close-out. Keeps
#: the online detail bounded without archiving on every single close.
DEFAULT_AUTO_ARCHIVE_THRESHOLD = 200

#: Environment variables that override the defaults above (the effective
#: values are recorded on every published summary).
ENV_CALIBRATION_WINDOW = "OR_CALIBRATION_WINDOW"
ENV_LATE_CHECK_GRACE_DAYS = "OR_CALIBRATION_LATE_CHECK_GRACE_DAYS"
ENV_ARCHIVE_MAX_FILE_BYTES = "OR_CALIBRATION_ARCHIVE_MAX_FILE_BYTES"
ENV_ARCHIVE_MAX_TOTAL_BYTES = "OR_CALIBRATION_ARCHIVE_MAX_TOTAL_BYTES"
ENV_ARCHIVE_RETENTION_DAYS = "OR_CALIBRATION_ARCHIVE_RETENTION_DAYS"
ENV_AUTO_ARCHIVE_THRESHOLD = "OR_CALIBRATION_AUTO_ARCHIVE_THRESHOLD"

# ---------------------------------------------------------------------------
# evidence-window defaults (the Execution Evidence Bank is BOUNDED)
# ---------------------------------------------------------------------------
#
# The Evidence Bank keeps a bounded RECENT window of complete episodes. This
# is NOT the calibration window (``policy.window``): the calibration window
# decides which closed episodes CALIBRATE (statistics), while this decides
# which raw execution rows are still retained. They are separate scopes for
# the same reason the calibration policy keeps three: collapsing them would
# make "how many samples calibrate" and "how much raw history is on disk" one
# number, which cannot answer both honestly.
#
# The bound is expressed in COMPLETE EPISODES (not bytes). A count is what
# can be enforced deterministically without measuring payload sizes on every
# maintenance run; there is deliberately NO byte-cap promise, because the
# only honest way to promise a byte bound would be to measure and enforce it,
# and a count that silently means "a few megabytes" would be a false promise.
# The eviction UNIT is the whole episode: contrast/repair chains are never
# split. Oldest = smallest ``closed_at`` in the close-out registry, so the
# ordering is the same one the calibration already uses.

#: Complete (closed) episodes retained in the Execution Evidence Bank.
DEFAULT_EVIDENCE_WINDOW_EPISODES = 800

#: Days an UNCLOSED episode's executions may stay before they are evicted as
#: a whole. Running / pending / abnormally-unclosed episodes are exempt from
#: the count bound (their outcome is not known), but they must still be
#: bounded — an episode that never closes cannot pin evidence forever.
DEFAULT_EVIDENCE_OPEN_GRACE_DAYS = 30.0

ENV_EVIDENCE_WINDOW_EPISODES = "OR_EVIDENCE_WINDOW_EPISODES"
ENV_EVIDENCE_OPEN_GRACE_DAYS = "OR_EVIDENCE_OPEN_GRACE_DAYS"


@dataclass
class EvidenceWindowPolicy:
    """The bound on retained Execution Evidence (a SEPARATE scope).

    - ``window_episodes``: how many COMPLETE episodes' executions are
      retained. Oldest first by the close-out registry's ``closed_at``.
    - ``open_grace_days``: how long an unclosed episode's executions may
      stay before the whole episode is evicted. It is a safeguard against an
      episode that NEVER closes, not an ordinary retention path.

    Three things are ALWAYS exempt, because deleting them would destroy an
    in-flight or still-being-used fact:
    - episodes inside the calibration window (they calibrate);
    - episodes awaiting a task verdict within ``late_check_grace_days``
      (a late check must still be able to land);
    - unclosed episodes younger than ``open_grace_days``.
    """

    window_episodes: int = DEFAULT_EVIDENCE_WINDOW_EPISODES
    open_grace_days: float = DEFAULT_EVIDENCE_OPEN_GRACE_DAYS

    @classmethod
    def from_env(cls, **overrides: Any) -> "EvidenceWindowPolicy":
        """Defaults <- environment <- explicit overrides (later wins)."""
        policy = cls(
            window_episodes=_env_int(ENV_EVIDENCE_WINDOW_EPISODES,
                                     DEFAULT_EVIDENCE_WINDOW_EPISODES),
            open_grace_days=_env_float(ENV_EVIDENCE_OPEN_GRACE_DAYS,
                                       DEFAULT_EVIDENCE_OPEN_GRACE_DAYS),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(policy, key):
                setattr(policy, key, value)
        return policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_episodes": int(self.window_episodes),
            "open_grace_days": float(self.open_grace_days),
            "unit": "complete episodes",
            "note": ("a bounded RECENT window of complete episodes (record "
                     "count, NOT a byte cap): episodes inside the calibration "
                     "window, episodes awaiting a task verdict within the "
                     "late-check grace period, and young unclosed episodes "
                     "are never evicted; the whole episode is evicted at "
                     "once so contrast/repair chains are never split. This "
                     "bounds the Evidence Bank, NOT the whole project "
                     "directory"),
        }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


@dataclass
class CalibrationPolicy:
    """The three retention scopes, kept EXPLICIT and separate.

    They answer three different questions and are never collapsed into one
    number, because a single "retention" setting cannot honestly bound the
    online database, decide which episodes calibrate, and cap the archive at
    the same time:

    - ``window``: which closed task-episodes CALIBRATE (the statistics).
    - ``late_check_grace_days``: how long an episode's ONLINE detail is kept
      so a late task check can still land on it. Applies to episodes that
      may still be awaiting a check; an episode whose executions all carry a
      verdict is not held by this.
    - ``archive_*``: what HISTORY is actually kept on disk — with a per-file
      size cap, a total-size cap and an age cap, so the archive cannot grow
      without bound (a total-size cap is what makes "the archive is bounded"
      a true statement).
    """

    window: int = DEFAULT_CALIBRATION_WINDOW
    late_check_grace_days: float = DEFAULT_LATE_CHECK_GRACE_DAYS
    archive_max_file_bytes: int = DEFAULT_ARCHIVE_MAX_FILE_BYTES
    archive_max_total_bytes: int = DEFAULT_ARCHIVE_MAX_TOTAL_BYTES
    archive_retention_days: float = DEFAULT_ARCHIVE_RETENTION_DAYS
    auto_archive_threshold: int = DEFAULT_AUTO_ARCHIVE_THRESHOLD

    @classmethod
    def from_env(cls, **overrides: Any) -> "CalibrationPolicy":
        """Defaults <- environment <- explicit overrides (later wins)."""
        policy = cls(
            window=_env_int(ENV_CALIBRATION_WINDOW,
                            DEFAULT_CALIBRATION_WINDOW),
            late_check_grace_days=_env_float(
                ENV_LATE_CHECK_GRACE_DAYS, DEFAULT_LATE_CHECK_GRACE_DAYS),
            archive_max_file_bytes=_env_int(
                ENV_ARCHIVE_MAX_FILE_BYTES, DEFAULT_ARCHIVE_MAX_FILE_BYTES),
            archive_max_total_bytes=_env_int(
                ENV_ARCHIVE_MAX_TOTAL_BYTES, DEFAULT_ARCHIVE_MAX_TOTAL_BYTES),
            archive_retention_days=_env_float(
                ENV_ARCHIVE_RETENTION_DAYS, DEFAULT_ARCHIVE_RETENTION_DAYS),
            auto_archive_threshold=_env_int(
                ENV_AUTO_ARCHIVE_THRESHOLD, DEFAULT_AUTO_ARCHIVE_THRESHOLD),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(policy, key):
                setattr(policy, key, value)
        return policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": int(self.window),
            "late_check_grace_days": float(self.late_check_grace_days),
            "archive_max_file_bytes": int(self.archive_max_file_bytes),
            "archive_max_total_bytes": int(self.archive_max_total_bytes),
            "archive_retention_days": float(self.archive_retention_days),
            "auto_archive_threshold": int(self.auto_archive_threshold),
            "note": ("three separate scopes: the WINDOW decides which closed "
                     "task-episodes calibrate; the GRACE period decides how "
                     "long online detail is kept for a possible late check; "
                     "the ARCHIVE caps (per-file, total, age) bound what "
                     "history is kept on disk"),
        }


# ---------------------------------------------------------------------------
# 1. the real outcome summary
# ---------------------------------------------------------------------------


@dataclass
class RealOutcomeSummary:
    """What actually happened in one bound prediction's real scope.

    Built from the action log and the execution records — never from the
    prediction. Each block carries its own eligibility so a later reader
    can tell "measured and comparable" from "absent", "unverified" or
    "belongs to another identity".
    """

    prediction_id: str
    task_id: str
    episode_id: Optional[str]
    strategy_id: Optional[str]
    action_id: Optional[str] = None
    execution_ids: List[str] = field(default_factory=list)
    #: The terminal state of the scope the prediction covered.
    scope_status: str = ""
    #: Benefit-relevant observations, with the metric/unit the PREDICTION
    # declared (the comparison must use the prediction's own yardstick).
    benefit: Dict[str, Any] = field(default_factory=dict)
    #: Real cost, scoped to the prediction's declared scope; auxiliary
    # overhead reported separately (it is real spend, but not what the
    # prediction spoke about).
    cost: Dict[str, Any] = field(default_factory=dict)
    auxiliary_cost: Dict[str, Any] = field(default_factory=dict)
    #: Observed risk events: occurred / not_occurred / unknown + basis.
    #: One row per event the PREDICTION named, plus the framework-observed
    #: events it did not (``predicted=False``) — the model never adjudicates
    #: its own prediction, and an event nobody predicted still counts as an
    #: observation.
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    #: The framework's OWN observation of every event in the vocabulary,
    #: keyed by event name, with its observation unit and unit ids. This is
    #: what the calibration layer counts OCCURRENCE RATES from: one
    #: execution observed once is one unit, however many predictions were
    #: bound to it.
    observed_events: Dict[str, Any] = field(default_factory=dict)
    #: Verification evidence for the solution the window produced.
    verification: Dict[str, Any] = field(default_factory=dict)
    #: Per-field eligibility: field -> (eligibility, reason).
    eligibility: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: Per-DIMENSION attribution of the binding's identity problems
    #: (``binding_attribution``): which comparison blocks are blocked and
    #: why. A known disagreement on the execution CONDITIONS blocks the
    #: outcome dimensions and preserves cost; an unconfirmed config key
    #: blocks nothing. The close-out reads THIS, never a blanket verdict.
    attribution: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closeout_version": EPISODE_CLOSEOUT_VERSION,
            "prediction_id": self.prediction_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "strategy_id": self.strategy_id,
            "action_id": self.action_id,
            "execution_ids": list(self.execution_ids),
            "scope_status": self.scope_status,
            "benefit": copy.deepcopy(self.benefit),
            "cost": copy.deepcopy(self.cost),
            "auxiliary_cost": copy.deepcopy(self.auxiliary_cost),
            "risk_events": copy.deepcopy(self.risk_events),
            "observed_events": copy.deepcopy(self.observed_events),
            "verification": copy.deepcopy(self.verification),
            "eligibility": copy.deepcopy(self.eligibility),
            "attribution": copy.deepcopy(self.attribution),
            "notes": list(self.notes),
        }


def _aggregate_costs(vectors: Sequence[Optional[CostVector]],
                     records: Optional[Sequence[Any]] = None
                     ) -> Dict[str, Any]:
    """Per-dimension totals over measured costs, with completeness.

    Same arithmetic as the execution-window aggregation: a dimension's
    total is the sum over the items that MEASURED it; ``complete`` is True
    only when every item measured it; a dimension nobody measured stays
    ``None`` (unknown, never zero). ``latency_s`` is never summed — it is
    reported per item elsewhere and never folded into an end-to-end figure.

    ``records`` (optional) enables DECLARATION SCREENING: a dimension whose
    contributing records include a declared estimate or a single-side token
    figure gets a ``screened`` block naming what was excluded and the total
    over only the ELIGIBLE records. The raw ``total`` is unchanged (it is
    the honest sum of what the mask says was measured); the reader decides
    which to use, and the calibration uses ``screened``.
    """
    dims: Dict[str, Any] = {}
    for dim in COST_DIMENSIONS:
        values: List[float] = []
        for cost in vectors:
            if cost is None or dim not in cost.measured_dims():
                continue
            values.append(float(getattr(cost, dim)))
        dims[dim] = {
            "total": round(sum(values), 6) if values else None,
            "n_measured": len(values),
            "n_items": len(vectors),
            "complete": bool(vectors) and len(values) == len(vectors),
        }
    if records:
        try:
            from or_harness.world_model.cost_eligibility import (
                screen_aggregate,
            )
            screened = screen_aggregate(dims, records)
            for dim, block in screened.items():
                dims[dim]["screened"] = block
        except Exception:  # noqa: BLE001 - a screening failure never
            # invalidates the raw aggregate.
            pass
    return dims


def summarize_real_outcome(harness, prediction) -> RealOutcomeSummary:
    """Build the real-outcome summary of ONE bound prediction.

    ``prediction`` is a :class:`~or_harness.world_model.contracts
    .StrategyOutcomePrediction` that has been bound to a real action
    (``trace.model_info.bound_action_id``). The summary is derived from
    the action log and the execution records; the prediction itself is
    read only for its DECLARED scope/metric (the yardstick the comparison
    must use), never for content.
    """
    info = prediction.trace.model_info
    action_id = info.get("bound_action_id")
    candidate = prediction.candidate
    summary = RealOutcomeSummary(
        prediction_id=prediction.prediction_id,
        task_id=candidate.task_id,
        episode_id=candidate.episode_id,
        strategy_id=candidate.strategy_id,
        action_id=action_id,
    )
    mismatch = info.get("binding_mismatch") or {}
    unknown = info.get("binding_unknown") or {}
    # Per-DIMENSION attribution. A binding problem blocks only the
    # comparison it really invalidates: a known disagreement on the
    # execution CONDITIONS (solver, strategy, effective input, a config
    # value) blocks the outcome dimensions and PRESERVES cost — a different
    # condition really did run and really did cost what it cost. An
    # unconfirmed (unknown) field blocks only the approach dimension it
    # names; an unreported ``time_limit`` blocks nothing at all. A
    # PERFORMED method that differs from the plan blocks the BENEFIT only.
    summary.attribution = binding_attribution(
        mismatch, unknown, info.get("method_observed"))
    blocked = summary.attribution.get("blocked") or {}
    for dim in EVALUATION_DIMENSIONS:
        entries = blocked.get(dim) or []
        if not entries:
            continue
        fields = ", ".join(sorted({str(e.get("field")) for e in entries}))
        summary.eligibility[f"identity.{dim}"] = {
            "eligibility": eligibility_for(entries),
            "reason": (f"the binding could not establish {fields} for this "
                       "comparison: this part of the sample is not scored "
                       "(the real spend is still kept wherever the "
                       "comparison does not depend on that field)"),
        }
    summary.notes.extend(summary.attribution.get("notes") or [])
    if action_id is None:
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the prediction was never bound to a real action: "
                      "an unexecuted candidate has no real outcome",
        }
        summary.notes.append(
            "unexecuted candidate: no counterfactual truth is fabricated")
        return summary

    action = harness.actions.get(action_id)
    if action is None:
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the bound action no longer exists in the log",
        }
        return summary

    # The real scope. An ATTEMPT-scope prediction is compared against the
    # bound action's own execution. A STRATEGY-WINDOW-scope prediction is
    # compared against the WHOLE window of its selection round: every
    # in-scope execution of that (task, episode, strategy, round) — a
    # failed first attempt and the repaired retry both count, so an
    # 11s-then-17s window reports 28s of real solve time, not just the
    # last attempt's 17s.
    records: List[Any] = []
    if candidate.scope == "strategy_window":
        # The DECLARED window identity (including its selection round, when
        # the window_id carries one) decides which window aggregates into
        # the real outcome: a round-1 prediction is scored against round
        # 1's executions only, never the whole-episode aggregation.
        declared_round = None
        if candidate.window_id:
            from or_harness.world_model.execution_window import (
                parse_window_id,
            )
            parsed = parse_window_id(candidate.window_id)
            if parsed is not None:
                declared_round = parsed.round_index
        window = harness.strategy_execution_window(
            action.task_id, action.episode_id,
            strategy_id=candidate.strategy_id,
            round_index=declared_round)
        window_actions = {a.action_id for a in window.attempts}
        if action.action_id not in window_actions:
            # The bound action is not in the derived window (it may be an
            # agent-reported action the derivation cannot see): fall back
            # to the bound action's own execution so the summary is never
            # empty, and say so.
            summary.notes.append(
                "the bound action is not among the derived window "
                "attempts; the summary covers the bound action's own "
                "execution only")
        for attempt in window.attempts:
            if attempt.execution_id is None:
                continue
            record = (harness.bank.get_pending(attempt.execution_id)
                      or harness.bank.get(attempt.execution_id))
            if record is not None:
                records.append(record)
            else:
                summary.eligibility["scope.execution"] = {
                    "eligibility": "missing",
                    "reason": (f"window attempt {attempt.action_id} has no "
                               "execution in the bank (staged or "
                               "recorded)"),
                }
        if not records and action.linked_execution_id:
            record = (harness.bank.get_pending(action.linked_execution_id)
                      or harness.bank.get(action.linked_execution_id))
            if record is not None:
                records.append(record)
        summary.notes.append(
            f"strategy-window scope (round "
            f"{declared_round if declared_round is not None else 'all'}): "
            f"{len(records)} in-scope execution(s) aggregate into the real "
            "outcome (failed retries included)")
    if action.linked_execution_id and not records:
        record = (harness.bank.get_pending(action.linked_execution_id)
                  or harness.bank.get(action.linked_execution_id))
        if record is not None:
            records.append(record)
        else:
            summary.eligibility["scope.execution"] = {
                "eligibility": "missing",
                "reason": "the bound action's linked execution is not in "
                          "the bank (staged or recorded)",
            }
    elif not records and not action.linked_execution_id:
        summary.eligibility["scope.execution"] = {
            "eligibility": "missing",
            "reason": "the bound action has no linked execution",
        }
    summary.execution_ids = [r.execution_id for r in records]

    # Scope status: the terminal state of what the prediction covered.
    if action.status == "running":
        summary.scope_status = "running"
        summary.eligibility["scope"] = {
            "eligibility": "missing",
            "reason": "the bound action has not ended: a running scope has "
                      "no final numbers",
        }
    else:
        summary.scope_status = str(action.status)
        if records:
            exec_status = records[-1].quality.get("status")
            summary.scope_status = f"{action.status}/{exec_status}"

    # -- benefit observations -------------------------------------------
    # The yardstick is the PREDICTION's own declared metric/unit/baseline:
    # the comparison must not invent a more convenient one afterwards, and
    # the declared metric decides WHICH observation channel applies.
    #
    # Two channels exist and they measure DIFFERENT things:
    #   * normalized_objective_gap — the solver's own gap (a QUALITY claim);
    #   * task_result_check_passed — the task-level check's verdict (a
    #     COMPLETION claim, taken from the check the execution already
    #     carries; no new check is run here).
    # A prediction declaring one is never scored against the other, even
    # though both land in [0,1].
    benefit = prediction.benefit
    if benefit is not None and benefit.kind == "solution_quality":
        observable = _observable_benefit_metric(benefit.metric)
        if observable is None:
            summary.eligibility["benefit"] = {
                "eligibility": "scope_mismatch",
                "reason": (f"no observation adapter exists for the declared "
                           f"metric {benefit.metric!r}: this build observes "
                           f"only {OBSERVABLE_BENEFIT_METRICS}. The "
                           "observed gap is never re-labelled as a metric "
                           "it did not measure"),
            }
        elif observable == OBSERVABLE_COMPLETION_METRIC:
            _observe_completion(summary, records)
        else:
            observed: List[float] = []
            task_check_gated = 0
            for record in records:
                quality = record.quality or {}
                gap = quality.get("gap")
                status = quality.get("status")
                if status == "optimal":
                    value = 1.0
                elif gap is not None and _finite(gap):
                    value = max(0.0, 1.0 - float(gap))
                elif quality.get("feasible"):
                    # A feasible solution with no gap/bound: the 0.5
                    # heuristic is NOT an observed quality truth.
                    summary.eligibility["benefit"] = {
                        "eligibility": "unverified",
                        "reason": "a feasible execution produced no "
                                  "gap/bound: the normalized quality is "
                                  "the 0.5 heuristic, not an observation",
                    }
                    continue
                else:
                    continue
                # TASK-CHECK GATE: an answer CONFIRMED not to satisfy the
                # task is a zero-quality observation. The calibration
                # channel compares the prediction against the TASK's real
                # outcome, so a relaxed answer's solver-side optimum must
                # not be scored as if the task had been solved.
                effective = task_effective_quality(record, value)
                if effective != value:
                    task_check_gated += 1
                observed.append(effective)
            if observed:
                # Window rule (declared BEFORE evaluation): the LAST
                # in-scope attempt's qualified solution is the window's
                # benefit observation — the rule the prediction's scope
                # statement implies, applied uniformly, never chosen per
                # result.
                summary.benefit = {
                    "kind": "solution_quality",
                    "metric": benefit.metric,
                    "unit": benefit.unit,
                    "observed": round(observed[-1], 6),
                    "rule": "last qualified in-scope attempt",
                    "n_observations": len(observed),
                    "all_observations": [round(v, 6) for v in observed],
                }
                if task_check_gated:
                    summary.benefit["task_check_gated"] = task_check_gated
                    summary.benefit["task_check_note"] = (
                        f"{task_check_gated} in-scope observation(s) were "
                        "set to 0.0 because a task-level check confirmed "
                        "the answer does not satisfy the task: the solver's "
                        "own optimality is not the task's outcome")
                if "benefit" not in summary.eligibility:
                    summary.eligibility["benefit"] = {
                        "eligibility": "evaluable",
                        "reason": "normalized quality observed on the "
                                  "in-scope execution(s)",
                    }
            elif "benefit" not in summary.eligibility:
                summary.eligibility["benefit"] = {
                    "eligibility": "unobserved",
                    "reason": "no in-scope execution produced a qualified "
                              "solution quality",
                }
    elif benefit is not None and benefit.kind == "effective_completion":
        observable = _observable_benefit_metric(benefit.metric)
        if observable == OBSERVABLE_COMPLETION_METRIC or observable is None \
                and str(benefit.metric or "") in (
                    "task_result_check_passed", "task_check_passed"):
            _observe_completion(summary, records)
        elif observable == OBSERVABLE_BENEFIT_METRIC:
            # Declared as a completion claim but measured by a QUALITY
            # metric: the two do not mean the same thing, so this is a
            # mismatch rather than a silent substitution.
            summary.eligibility["benefit"] = {
                "eligibility": "scope_mismatch",
                "reason": (f"benefit kind 'effective_completion' was "
                           f"declared with the quality metric "
                           f"{benefit.metric!r}: a solver-side gap is not a "
                           "task-completion observation, and the two are "
                           "never interchanged"),
            }
        else:
            _observe_completion(summary, records)
    elif benefit is not None:
        if benefit.kind == "valid_progress":
            # Progress is a claim about intermediate movement, which this
            # build has NO observation channel for. It is reported, never
            # scored — and never quietly replaced by a completion rate.
            summary.eligibility["benefit"] = {
                "eligibility": "scope_mismatch",
                "reason": ("no observation channel exists for benefit kind "
                           "'valid_progress': progress is not the task "
                           "check's outcome and not the solver's gap. "
                           "Declare kind='effective_completion' with "
                           "metric='task_result_check_passed' to have the "
                           "task check evaluate it"),
            }
        else:
            summary.eligibility["benefit"] = {
                "eligibility": "scope_mismatch",
                "reason": (f"no observation adapter exists for benefit kind "
                           f"{benefit.kind!r}: the benefit is reported, not "
                           "evaluated — this build evaluates normalized "
                           "solution quality and task-check completion"),
            }
    else:
        summary.eligibility["benefit"] = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no benefit to observe against",
        }

    # -- cost observations ------------------------------------------------
    in_scope_costs = [r.cost for r in records]
    summary.cost = _aggregate_costs(in_scope_costs, records)
    # Auxiliary overhead: the OTHER actions of the same episode (model /
    # verify / select_strategy / other executions) — real spend, reported,
    # never folded into the predicted scope's comparison.
    auxiliary: List[CostVector] = []
    for other in harness.actions.query(task_id=candidate.task_id,
                                        episode_id=candidate.episode_id):
        if other.action_id == action_id or other.rollup == "reference":
            continue
        if other.action_type == "execute_strategy" \
                and other.action_id != action_id:
            continue
        if other.cost is not None:
            auxiliary.append(other.cost)
    summary.auxiliary_cost = _aggregate_costs(auxiliary)
    summary.notes.append(
        "cost comparison uses the in-scope totals only; auxiliary overhead "
        "(modelling, verification, other attempts) is real spend reported "
        "separately and never charged to this prediction's scope")

    # -- risk event observations -------------------------------------------
    # The framework observes the events on its OWN, from the in-scope
    # executions and the episode's budget ledger — it does NOT start from
    # the model's predicted list. A prediction that never mentioned an event
    # the framework really observed is still recorded (it feeds the
    # occurrence-rate statistic); a prediction that mentioned an event with
    # no observation channel keeps the label unknown. The two lists are
    # merged by name below.
    observed = observe_episode_events(
        harness, records, task_id=candidate.task_id,
        episode_id=candidate.episode_id, scope_complete=(
            bool(records) and action.status != "running"))
    summary.observed_events = observed["events"]
    summary.risk_events = _pair_predicted_events(
        prediction, observed, candidate)
    # A per-prediction label is never the whole story: the framework's own
    # observation is kept so the calibration layer can count OCCURRENCE
    # RATES by observation unit (an execution observed once is one unit,
    # however many predictions were bound to it).
    summary.notes.append(
        "risk events are observed by the framework independently of the "
        "model's predictions; the observation unit is the execution (or the "
        "episode for the budget ledger), so several predictions bound to one "
        "execution produce ONE observation, not several")

    # -- verification -----------------------------------------------------
    verified_actions = [
        a for a in harness.actions.query(task_id=candidate.task_id,
                                         episode_id=candidate.episode_id)
        if a.action_type == "verify"]
    summary.verification = {
        "n_verify_actions": len(verified_actions),
        "solver_reported_status": (records[0].quality.get("status")
                                   if records else None),
        "note": ("solver 'optimal' does not by itself prove the business "
                 "requirements are satisfied; verification evidence is "
                 "whatever verify actions and the executor's checks "
                 "recorded"),
    }
    return summary


# ---------------------------------------------------------------------------
# 1b. framework-side risk observation (independent of any prediction)
# ---------------------------------------------------------------------------


def _execution_event_observations(records: Sequence[Any]
                                  ) -> Dict[str, Dict[str, Any]]:
    """Observe the execution-unit events, ONE LABEL PER EXECUTION.

    The label is stored per OBSERVATION UNIT (``units[execution_id]``), not
    once for the whole scope. A scope of two executions where one timed out
    and one succeeded is ONE timeout out of TWO units — collapsing it to a
    single scope-level label made the rate 100% instead of 50%, and made a
    single failed check look like every execution in the scope failed.

    An event that cannot be decided for a given execution (a record written
    before ``error_class`` existed) keeps ``label=None`` for THAT unit and
    says why — it is never inferred from prose, and it never borrows another
    execution's verdict.
    """
    events: Dict[str, Dict[str, Any]] = {}

    def _unit(name: str, execution_id: Optional[str], label: Optional[str],
              basis: str, detail: Optional[Dict[str, Any]] = None) -> None:
        entry = events.setdefault(name, {
            "event": name,
            "unit": OBSERVABLE_RISK_EVENTS[name]["unit"],
            "units": {},
        })
        entry["units"][str(execution_id)] = {
            "label": label, "label_basis": basis,
            **({"detail": detail} if detail else {}),
        }

    for record in records:
        quality = record.quality or {}
        status = quality.get("status")
        execution_id = record.execution_id

        if status == "timeout":
            _unit("timeout", execution_id, "occurred",
                  "the executor's time limit fired")
        else:
            _unit("timeout", execution_id, "not_occurred",
                  "the execution completed without hitting the time limit")

        if status == "infeasible":
            # A REPORTED infeasibility is a fact about the solver's
            # verdict. It is never by itself a strategy failure, so it is
            # recorded under its own name and never mapped onto a
            # failure/validity label.
            _unit("solver_reported_infeasible", execution_id, "occurred",
                  "the solver reported infeasibility (a verdict, not by "
                  "itself a strategy failure)")
        else:
            _unit("solver_reported_infeasible", execution_id, "not_occurred",
                  "the solver did not report infeasibility for this "
                  "execution")

        raw_classes = [getattr(f, "error_class", None)
                       for f in (record.failures or [])]
        classes = [c for c in raw_classes if c]
        if status == "error" and not classes:
            # A failure with NO recorded class (or a legacy record whose
            # class was never set): neither cause can be established, and
            # it is NOT inferred from the error text.
            for name in ("environment_failure", "implementation_failure"):
                _unit(name, execution_id, None,
                      "the failure carries no recorded error_class (a "
                      "record written before the class existed, or a "
                      "failure with no usable record): the cause is UNKNOWN "
                      "and is not inferred from the error text")
        else:
            for name, wanted in (("environment_failure", "environment"),
                                 ("implementation_failure", "model")):
                if wanted in classes:
                    _unit(name, execution_id, "occurred",
                          "the executor recorded an ENVIRONMENT failure "
                          "(sandbox policy / missing module)"
                          if wanted == "environment" else
                          "the executor recorded the harness's OWN code "
                          "failing (an implementation failure, not a "
                          "modelling error)")
                else:
                    # The failure class is KNOWN and is a different one (or
                    # there was no failure at all): this event did not
                    # occur. Silence here would drop a real not_occurred
                    # observation from the denominator.
                    _unit(name, execution_id, "not_occurred",
                          "the execution did not record this failure class")

        # The task-check channel: the CHECK RESULT, and nothing more.
        verdict = task_check_state(record)
        block = task_check_block(record) or {}
        if verdict == "failed":
            _unit("task_check_failed", execution_id, "occurred",
                  "a declared task check ran on real values and did not "
                  "hold",
                  detail={"checks": [c.get("check") for c in
                                     (block.get("checks") or [])],
                          "unchecked": list(
                              (block.get("scope") or {}).get("unchecked")
                              or []),
                          "intent": block.get("intent")})
        elif verdict == "passed":
            _unit("task_check_failed", execution_id, "not_occurred",
                  "a declared task check passed on its declared bases (the "
                  "covered scope is listed; a pass is not proof that the "
                  "whole model matches the task)")
        else:
            _unit("task_check_failed", execution_id, None,
                  "no task check is on record for this execution: whether "
                  "the answer satisfies the task is UNKNOWN")
    return events


def _scope_aggregate(units: Dict[str, Dict[str, Any]],
                     unit_ids: Sequence[str],
                     scope_complete: bool) -> Tuple[Optional[str], str]:
    """Aggregate per-unit labels into ONE label for a prediction's scope.

    - ``occurred``: at least one in-scope unit occurred (an occurrence
      anywhere in the scope is an occurrence of the scope).
    - ``not_occurred``: the scope is COMPLETE and every in-scope unit has a
      known, non-occurring label. An unknown unit blocks this direction —
      absence in an incompletely observed scope is not evidence.
    - ``None``: otherwise, with the reason.
    """
    relevant = [units.get(str(u)) for u in unit_ids]
    relevant = [entry for entry in relevant if entry is not None]
    if not relevant:
        return None, "the framework recorded no observation for this scope"
    occurred = [e for e in relevant if e.get("label") == "occurred"]
    if occurred:
        return "occurred", occurred[0].get("label_basis") or ""
    unknown = [e for e in relevant if e.get("label") is None]
    if unknown:
        return None, unknown[0].get("label_basis") or "unknown observation"
    if not scope_complete:
        return None, ("the scope did not fully complete, so the absence of "
                      "this event is not evidence")
    if len(relevant) < len(unit_ids):
        return None, ("some in-scope executions carry no observation for "
                      "this event, so absence is not evidence")
    return "not_occurred", (relevant[0].get("label_basis")
                            or "the completed scope produced no such "
                               "observation")


def observe_episode_events(harness, records: Sequence[Any], *,
                           task_id: str,
                           episode_id: Optional[str],
                           scope_complete: bool
                           ) -> Dict[str, Any]:
    """The FRAMEWORK's own observation of the risk events of one scope.

    Built from the in-scope executions and the episode's budget ledger —
    never from a prediction. This is the single source both the per-
    prediction labelling and the occurrence-rate statistic read, so the two
    can never disagree about what happened.

    ``scope_complete`` gates the "did not occur" direction for the
    execution-unit events: a scope that has not finished has no absence
    evidence, so those labels stay unknown. The budget event is decided by
    the ledger alone (its unit is the episode).
    """
    per_unit = _execution_event_observations(records)
    execution_ids = [str(r.execution_id) for r in records]
    events: Dict[str, Dict[str, Any]] = {}
    # Every observable execution-unit event is reported even when NOTHING
    # happened, so the occurrence-rate denominator is explicit rather than
    # an artefact of which events the model happened to predict. The SCOPE
    # label is derived from the per-unit labels; the units themselves are
    # kept so an aggregator counts each real execution exactly once.
    for name, definition in OBSERVABLE_RISK_EVENTS.items():
        if definition["unit"] != "execution":
            continue
        units = (per_unit.get(name) or {}).get("units") or {}
        label, basis = _scope_aggregate(units, execution_ids,
                                        scope_complete)
        events[name] = {
            "event": name,
            "unit": "execution",
            "label": label,
            "label_basis": basis,
            "unit_ids": list(execution_ids),
            # PER-UNIT labels: the aggregation of the occurrence rate reads
            # these, so two executions with different outcomes count as one
            # occurred and one not_occurred, not as two occurrences.
            "units": units,
        }
    # The budget ledger: unit = episode. THREE states, as before, but
    # recorded as a framework fact whether or not any prediction named it.
    declared_budget = (harness._load_budget(task_id, episode_id) or {})
    episode_unit = f"{task_id}|{episode_id or ''}"
    budget_entry: Dict[str, Any] = {
        "event": "budget_exhausted",
        "unit": "episode",
        "label": None,
        "label_basis": "",
        "unit_ids": [episode_unit],
        "units": {},
    }
    if declared_budget:
        budget_view = harness.budget.view(task_id, episode_id,
                                          budget=declared_budget)
        budget_status = budget_view.get("status")
        if budget_status == "exceeded":
            budget_entry["label"] = "occurred"
            budget_entry["label_basis"] = (
                "the declared budget was exceeded by measured real "
                "consumption")
        elif budget_status == "ok":
            budget_entry["label"] = "not_occurred"
            budget_entry["label_basis"] = (
                "the declared budget was confirmed within limits (every "
                "declared dimension measured)")
        else:
            budget_entry["label_basis"] = (
                f"budget status {budget_status!r}: consumption cannot be "
                "confirmed in either direction, so the label stays unknown")
    else:
        budget_entry["label_basis"] = (
            "no budget was declared for this episode, so no budget event "
            "can be observed")
    budget_entry["units"][episode_unit] = {
        "label": budget_entry["label"],
        "label_basis": budget_entry["label_basis"],
    }
    events["budget_exhausted"] = budget_entry
    return {
        "vocabulary_version": EVENT_VOCABULARY_VERSION,
        "task_id": task_id,
        "episode_id": episode_id,
        "scope_complete": bool(scope_complete),
        "events": events,
    }


def _pair_predicted_events(prediction, observed: Dict[str, Any],
                           candidate) -> List[Dict[str, Any]]:
    """Pair the model's predicted events with the framework's observations.

    One row per event the MODEL predicted, plus the framework-observed
    events it did not (marked ``predicted=False``, with no probability —
    they feed the occurrence-rate statistic and can never produce a Brier
    score). The framework never lets the model adjudicate its own
    prediction: the label comes from the observation, never from the
    probability.
    """
    events = observed["events"]
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    predicted_events = (prediction.risk.events
                        if prediction.risk is not None else [])
    for event in predicted_events:
        canonical = _normalize_event_name(event.event)
        seen.add(canonical)
        definition = OBSERVABLE_RISK_EVENTS.get(canonical)
        retired = RETIRED_RISK_EVENTS.get(canonical)
        observation = events.get(canonical)
        if definition is None:
            label = None
            basis = (f"event {event.event!r} is not in the framework's "
                     f"observation vocabulary ({EVENT_VOCABULARY_VERSION}): "
                     + (retired if retired else
                        "no observation channel exists for it, so its label "
                        "stays unknown and it is never scored"))
        else:
            # The budget event is decided by the LEDGER, whose scope is the
            # episode. A prediction whose DECLARED scope is narrower has no
            # matching budget range: the label stays unknown with an
            # explicit scope_mismatch basis instead of borrowing the
            # episode-wide verdict.
            if canonical == "budget_exhausted" \
                    and not _budget_scope_matches(candidate):
                label = None
                basis = (
                    f"scope_mismatch: the budget ledger is EPISODE-scoped, "
                    f"and this prediction declares scope "
                    f"{getattr(candidate, 'scope', 'attempt')!r}. The "
                    "episode-wide budget verdict is not this scope's "
                    "outcome, so the label stays unknown")
            elif observation is None:
                label = None
                basis = "the framework recorded no observation for this event"
            else:
                label = observation["label"]
                basis = observation["label_basis"]
        rows.append({
            "event": event.event,
            "canonical_event": canonical,
            "predicted": True,
            "observation_unit": (definition or {}).get("unit"),
            "predicted_probability": event.probability,
            "label": label,
            "label_basis": basis,
        })
    # Framework-observed events the model did NOT predict: recorded so the
    # occurrence rate has an honest denominator, never scored for accuracy.
    for name, observation in events.items():
        if name in seen:
            continue
        rows.append({
            "event": name,
            "canonical_event": name,
            "predicted": False,
            "observation_unit": observation.get("unit"),
            "predicted_probability": None,
            "label": observation.get("label"),
            "label_basis": observation.get("label_basis"),
            "unit_ids": list(observation.get("unit_ids") or []),
        })
    return rows


# ---------------------------------------------------------------------------
# 2. the post-hoc evaluation
# ---------------------------------------------------------------------------


@dataclass
class StrategyPredictionEvaluation:
    """The post-hoc comparison of ONE frozen prediction against its real
    outcome.

    Appended at close-out; the prediction itself is never modified. Every
    compared field names its own basis, and every excluded field names its
    exclusion reason — an evaluation with nothing comparable says so
    instead of scoring nothing as success.
    """

    evaluation_id: str = field(default_factory=lambda: _new_id("ev"))
    prediction_id: str = ""
    task_id: str = ""
    episode_id: Optional[str] = None
    #: The prediction's declared measurement scope (attempt /
    #: strategy_window): part of the calibration grouping key, so samples
    #: under different scopes never pool.
    scope: str = "attempt"
    #: The IDENTITY of the model that made the prediction, as the provider
    #: described itself (``provider_model`` / ``provider_version``).
    #: Different models are different predictors, so their errors never
    #: pool. ``"(unknown)"`` is the honest value for a prediction that
    #: recorded no identity (a legacy record) — it is never guessed from
    #: the provider NAME, which is not a version.
    model_identity: str = "(unknown)"
    created_at: float = field(default_factory=time.time)
    #: benefit / cost / risk / interval blocks, each with eligibility.
    benefit: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    interval: Dict[str, Any] = field(default_factory=dict)
    #: Overall state: evaluated (>=1 field compared), excluded (nothing
    #: comparable, with reasons), or pending (the scope is not finished).
    state: str = "evaluated"
    exclusion_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Per-field eligibility summary: ``{field: {eligibility, reason}}`` for
    #: benefit, cost (per dimension) and risk (per event). It answers the
    #: question "which parts of this prediction may be calibrated on?" ONE
    #: field at a time, so a reader never has to read a blanket verdict as
    #: "the whole prediction is scoreable". The individual blocks above
    #: remain the authority; this is the index over them.
    eligibility_summary: Dict[str, Any] = field(default_factory=dict)
    #: Which comparison blocks the binding's identity problems blocked, and
    #: by which fields: ``{dimension: [{field, kind}, ...]}``. Only the
    #: dimensions that really depend on an unresolved field appear; a
    #: dimension absent from it was evaluated normally. This is what makes
    #: a missing execution receipt a caveat instead of a discarded sample.
    attribution: Dict[str, Any] = field(default_factory=dict)
    #: A performed method that differs from the planned method
    #: (``compare_methods`` verdict ``mismatch``), or None. Reported
    #: separately so "the plan was not carried out" is visible without
    #: reading the identity bookkeeping. It is NOT an identity problem: the
    #: attempt really happened, it simply is not the planned method's answer.
    method_deviation: Optional[Dict[str, Any]] = None
    #: Whether ANY part of this prediction may enter the calibration sample
    #: (``state == "evaluated"``). Deliberately narrow: being BOUND (the
    #: prediction is linked to a real action) is a different, weaker fact
    #: and is NOT a promise that anything can be calibrated. ``False`` here
    #: is not a failure — it says which fields were unobservable.
    calibratable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_version": EPISODE_CLOSEOUT_VERSION,
            "evaluation_id": self.evaluation_id,
            "prediction_id": self.prediction_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "scope": self.scope,
            "model_identity": self.model_identity,
            "created_at": self.created_at,
            "state": self.state,
            "benefit": copy.deepcopy(self.benefit),
            "cost": copy.deepcopy(self.cost),
            "risk": copy.deepcopy(self.risk),
            "interval": copy.deepcopy(self.interval),
            # The per-field index. Derived on read for a legacy record that
            # never wrote it, so an old evaluation stays readable and never
            # gains a SECOND, drifting copy of the same state.
            "eligibility": (copy.deepcopy(self.eligibility_summary)
                            or self._derive_eligibility()),
            "attribution": copy.deepcopy(self.attribution),
            "method_deviation": copy.deepcopy(self.method_deviation),
            "calibratable": bool(self.calibratable or self._derive_calibratable()),
            "exclusion_reasons": list(self.exclusion_reasons),
            "notes": list(self.notes),
        }

    def _derive_eligibility(self) -> Dict[str, Any]:
        """The per-field index rebuilt from the blocks (legacy fallback).

        Used when an evaluation was written before the index existed: the
        field blocks are the authority, so deriving the index from them can
        never disagree with them — and it avoids storing two copies of the
        same verdict that could later drift apart.
        """
        out: Dict[str, Any] = {}
        benefit = self.benefit or {}
        if benefit:
            out["benefit"] = {
                "eligibility": benefit.get("eligibility"),
                "reason": benefit.get("reason"),
            }
        cost = self.cost or {}
        per_dim = {}
        for dim, entry in (cost.get("per_dim") or {}).items():
            per_dim[dim] = {"eligibility": "evaluable"}
        for dim, reason in (cost.get("excluded") or {}).items():
            per_dim[dim] = {"eligibility": "unobserved", "reason": reason}
        if per_dim:
            out["cost"] = {"dimensions": per_dim,
                           "eligibility": cost.get("eligibility")}
        risk = self.risk or {}
        events = {}
        for entry in risk.get("scored") or []:
            events[str(entry.get("event"))] = {"eligibility": "evaluable"}
        for entry in risk.get("unscored") or []:
            events.setdefault(str(entry.get("event")), {})["eligibility"] = (
                "unobserved")
            events[str(entry.get("event"))].setdefault(
                "reason", entry.get("reason"))
        if events:
            out["risk"] = {"events": events,
                           "eligibility": risk.get("eligibility")}
        interval = self.interval or {}
        if interval:
            out["interval"] = {"eligibility": interval.get("eligibility"),
                               "reason": interval.get("reason")}
        return out

    def _derive_calibratable(self) -> bool:
        return self.state == "evaluated"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyPredictionEvaluation":
        data = data or {}
        return cls(
            evaluation_id=str(data.get("evaluation_id")
                              or _new_id("ev")),
            prediction_id=str(data.get("prediction_id", "")),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            scope=str(data.get("scope", "attempt")),
            model_identity=str(data.get("model_identity") or "(unknown)"),
            created_at=float(data.get("created_at", time.time())),
            state=str(data.get("state", "evaluated")),
            benefit=copy.deepcopy(dict(data.get("benefit") or {})),
            cost=copy.deepcopy(dict(data.get("cost") or {})),
            risk=copy.deepcopy(dict(data.get("risk") or {})),
            interval=copy.deepcopy(dict(data.get("interval") or {})),
            eligibility_summary=copy.deepcopy(
                dict(data.get("eligibility") or {})),
            attribution=copy.deepcopy(dict(data.get("attribution") or {})),
            method_deviation=(copy.deepcopy(data.get("method_deviation"))
                              if data.get("method_deviation") else None),
            calibratable=bool(data.get("calibratable", False)),
            exclusion_reasons=[str(r) for r in
                               (data.get("exclusion_reasons") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def _prediction_model_identity(prediction) -> str:
    """The identity of the model that made a prediction, or ``(unknown)``.

    Read from ``trace.model_info`` (written by the prediction service from
    the provider's own ``describe()``). The label comes from the SAME
    helper the service uses for its own identity, so a prediction and its
    calibration group can never disagree about which model produced it.
    """
    info = (prediction.trace.model_info
            if prediction.trace is not None else {}) or {}
    from or_harness.world_model.strategy_prediction import (
        model_identity_label,
    )
    return model_identity_label(info)


def evaluate_strategy_prediction(prediction, summary: RealOutcomeSummary
                                 ) -> StrategyPredictionEvaluation:
    """Compare ONE frozen prediction against its real-outcome summary.

    Field-by-field eligibility, never a blanket verdict:

    - **benefit**: only when the prediction declared a value AND the
      summary observed the SAME metric (normalized solution quality) under
      the same scope. The error is the absolute difference; the baseline
      the prediction declared is carried along (a gain vs its baseline is
      the prediction's own claim, restated not recomputed).
    - **cost**: per dimension, only where the prediction predicted the
      dimension AND the real scope measured it completely. The error is
      the absolute log-ratio (the shared ``cost_error_per_dim``
      arithmetic); a zero/missing denominator produces no relative error.
    - **risk**: per event, a Brier score only when the prediction gave a
      probability AND the label is known. A single trajectory does not
      prove a probability; the score is recorded as one sample, never as
      an accuracy verdict.
    - **interval**: coverage only when the prediction saved an interval
      AND the observed value exists.

    The original prediction is read, never written.
    """
    evaluation = StrategyPredictionEvaluation(
        prediction_id=prediction.prediction_id,
        task_id=prediction.candidate.task_id,
        episode_id=prediction.candidate.episode_id,
        scope=prediction.candidate.scope,
        model_identity=_prediction_model_identity(prediction),
    )
    # Per-DIMENSION attribution, not a blanket verdict. ``blocked`` names
    # only the comparisons an identity problem really invalidates; a
    # dimension absent from it is free to be evaluated. This is what keeps
    # a missing execution receipt (an unconfirmed ``time_limit``) from
    # discarding a real, unambiguous observation.
    attribution = (summary.attribution
                   or binding_attribution(
                       (prediction.trace.model_info or {}).get(
                           "binding_mismatch") or {},
                       (prediction.trace.model_info or {}).get(
                           "binding_unknown") or {},
                       (prediction.trace.model_info or {}).get(
                           "method_observed")))
    blocked = blocked_dimensions(attribution)
    evaluation.attribution = copy.deepcopy(blocked)
    # A method deviation is REPORTED on the evaluation in two places: the
    # dimension map above (which blocks the benefit), and the flat
    # ``method_deviation`` block below, so a reader does not have to read
    # the identity bookkeeping to learn that the plan was not carried out.
    evaluation.method_deviation = copy.deepcopy(
        attribution.get("deviation"))
    if evaluation.method_deviation is not None:
        evaluation.notes.append(method_deviation_note(
            evaluation.method_deviation))

    def _blocked(dim: str) -> bool:
        return bool(blocked.get(dim))

    def _block_reason(dim: str) -> str:
        entries = blocked.get(dim) or []
        fields = ", ".join(sorted({str(e.get("field")) for e in entries}))
        return (f"the binding could not establish {fields} for this "
                "comparison")
    n_compared = 0

    # -- benefit -----------------------------------------------------------
    benefit = prediction.benefit
    # The DECLARED yardstick travels with the evaluation whatever the
    # eligibility: grouping is by what the prediction SAID it was
    # predicting, so an unobservable sample still lands in the group it
    # belongs to instead of a nameless bucket.
    declared_yardstick = ({
        "metric": benefit.metric, "unit": benefit.unit,
        "predicted": (round(float(benefit.value), 6)
                      if benefit.value is not None else None),
    } if benefit is not None else {})
    if benefit is None or benefit.value is None:
        evaluation.benefit = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no benefit value",
            **declared_yardstick,
        }
    elif "observed" not in summary.benefit:
        entry = summary.eligibility.get("benefit", {})
        evaluation.benefit = {
            "eligibility": entry.get("eligibility", "unobserved"),
            "reason": entry.get("reason", "no benefit observation"),
            **declared_yardstick,
        }
    elif _blocked("benefit"):
        evaluation.benefit = {
            "eligibility": eligibility_for(blocked["benefit"]),
            "reason": _block_reason("benefit") + ": the real outcome may "
                      "not be this prediction's truth",
            "identity_fields": sorted({str(e.get("field"))
                                       for e in blocked["benefit"]}),
            **declared_yardstick,
        }
    else:
        observed = summary.benefit["observed"]
        predicted = float(benefit.value)
        evaluation.benefit = {
            "eligibility": "evaluable",
            "metric": benefit.metric,
            "unit": benefit.unit,
            "predicted": round(predicted, 6),
            "observed": observed,
            "abs_error": round(abs(predicted - observed), 6),
            # DIRECTED error: positive = the prediction was too LOW (the
            # real outcome was better than predicted); negative = too HIGH.
            # A magnitude alone cannot tell an over-estimate from an
            # under-estimate, which is exactly what a later prediction needs
            # to correct for.
            "signed_error": round(observed - predicted, 6),
            "baseline": (benefit.baseline.to_dict()
                         if benefit.baseline is not None else None),
            "note": ("the metric/unit/baseline are the prediction's OWN "
                     "declared yardstick, restated — never re-chosen after "
                     "the result was seen; signed_error = observed - "
                     "predicted (positive = under-predicted)"),
        }
        # Carry the task-check gate through: a reader comparing "predicted
        # 0.8, observed 0.0" must be able to see that the zero came from a
        # CONFIRMED-wrong answer, not from a genuinely bad solve.
        if summary.benefit.get("task_check_gated"):
            evaluation.benefit["task_check_gated"] = \
                summary.benefit["task_check_gated"]
            evaluation.benefit["task_check_note"] = \
                summary.benefit.get("task_check_note")
        n_compared += 1

    # -- cost -----------------------------------------------------------------
    predicted_cost = prediction.cost
    if predicted_cost is None or predicted_cost.expected is None:
        evaluation.cost = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no cost",
        }
    elif _blocked("cost"):
        evaluation.cost = {
            "eligibility": eligibility_for(blocked["cost"]),
            "reason": _block_reason("cost") + ": the real spend is not "
                      "this prediction's cost",
            "identity_fields": sorted({str(e.get("field"))
                                       for e in blocked["cost"]}),
        }
    else:
        per_dim: Dict[str, Any] = {}
        excluded: Dict[str, str] = {}
        predicted_dims = predicted_cost.expected.measured_dims()
        for dim in sorted(predicted_dims):
            real = summary.cost.get(dim) or {}
            # DECLARATION SCREENING: a value that was DECLARED (an
            # ``agent_estimate``) or recorded from a single token side is
            # not a measured truth. It stays visible in the aggregate, but
            # it can never stand as the real total the prediction is scored
            # against.
            screened = real.get("screened")
            if screened is not None:
                eligible_n = screened.get("n_eligible_items", 0)
                total_n = screened.get("n_items", 0)
                if eligible_n < total_n:
                    kinds = ", ".join(
                        f"{k}={v}" for k, v in
                        sorted((screened.get("excluded") or {}).items()))
                    excluded[dim] = (
                        "partly declared on the real scope "
                        f"({eligible_n}/{total_n} records measured it from a "
                        f"trusted source; excluded: {kinds}): a "
                        "partly-declared total is not a measured truth to "
                        "score against (the declarations stay visible in the "
                        "aggregate)")
                    continue
            if real.get("total") is None:
                excluded[dim] = ("not measured on the real scope"
                                 if real.get("n_items") else
                                 "no real execution to measure it")
                continue
            if not real.get("complete"):
                excluded[dim] = ("partially measured on the real scope "
                                 f"({real.get('n_measured')}/"
                                 f"{real.get('n_items')}): an incomplete "
                                 "total is not a truth to score against")
                continue
            p = float(getattr(predicted_cost.expected, dim))
            a = float(real["total"])
            entry: Dict[str, Any] = {
                "predicted": round(p, 6),
                "actual": round(a, 6),
                "abs_error": round(abs(p - a), 6),
            }
            if p > 0 and a > 0:
                ratio = math.log(a / p)
                # DIRECTED error: positive = the real cost was HIGHER than
                # predicted (under-predicted), negative = lower. The
                # absolute log-error is kept alongside so existing readers
                # see no change.
                entry["log_ratio"] = round(ratio, 4)
                entry["log_error"] = round(abs(ratio), 4)
            else:
                entry["log_ratio"] = None
                entry["log_error"] = None
                entry["note"] = ("zero predicted or actual: no relative "
                                 "ratio is manufactured (a log-ratio would "
                                 "be infinite or undefined, and a "
                                 "substituted value would be a fabrication)")
            per_dim[dim] = entry
            n_compared += 1
        evaluation.cost = {
            "eligibility": "evaluable" if per_dim else "unobserved",
            "per_dim": per_dim,
            "excluded": excluded,
            "note": ("only dimensions the prediction predicted AND the "
                     "real scope measured completely participate; "
                     "auxiliary overhead is never charged to this scope; "
                     "log_ratio = log(actual/predicted) (positive = "
                     "under-predicted cost)"),
        }

    # -- risk -------------------------------------------------------------------
    # The framework's OWN observation of every vocabulary event travels with
    # EVERY evaluation, whatever the prediction carried: an event nobody
    # predicted is still an observation, and the occurrence rate is counted
    # from these units, not from how many predictions happened to name one.
    observed_units: Dict[str, Dict[str, Any]] = {}
    for name, observation in (summary.observed_events or {}).items():
        if not isinstance(observation, dict) or "label" not in observation:
            continue
        observed_units[name] = {
            "unit": observation.get("unit"),
            "label": observation.get("label"),
            "n_units": len(observation.get("unit_ids") or []),
            "label_basis": observation.get("label_basis"),
            "unit_ids": list(observation.get("unit_ids") or []),
            # PER-UNIT labels: the occurrence-rate aggregation needs the
            # label of EACH real execution, not the scope's single verdict.
            # Without them, one timeout among three executions counted as
            # three occurrences.
            "units": copy.deepcopy(observation.get("units") or {}),
        }
    risk = prediction.risk
    if risk is None or not risk.events:
        evaluation.risk = {
            "eligibility": "not_predicted",
            "reason": "the prediction carried no risk events",
            "scored": [],
            "unscored": [],
            "observed_units": observed_units,
        }
    elif _blocked("risk"):
        evaluation.risk = {
            "eligibility": eligibility_for(blocked["risk"]),
            "reason": _block_reason("risk") + ": the real outcome may "
                      "not be this prediction's truth",
            "identity_fields": sorted({str(e.get("field"))
                                       for e in blocked["risk"]}),
            "scored": [],
            "unscored": [],
            "observed_units": observed_units,
        }
    else:
        scored: List[Dict[str, Any]] = []
        unscored: List[Dict[str, Any]] = []
        for observed_event in summary.risk_events:
            probability = observed_event.get("predicted_probability")
            label = observed_event["label"]
            if probability is None:
                # Either the model gave no probability, or the event was
                # observed by the FRAMEWORK and never predicted. Either way
                # it is NOT a scored pair — but it IS an observation, and
                # the occurrence-rate statistic counts it.
                unscored.append({
                    "event": observed_event["event"],
                    "predicted": bool(observed_event.get("predicted")),
                    "observation_unit": observed_event.get(
                        "observation_unit"),
                    "label": label,
                    "reason": ("the model did not predict this event: it is "
                               "observed by the framework and counted in "
                               "the occurrence rate, but no Brier score can "
                               "be computed without a probability"
                               if not observed_event.get("predicted") else
                               "the prediction gave no probability"),
                })
                continue
            if label is None:
                unscored.append({
                    "event": observed_event["event"],
                    "predicted": True,
                    "observation_unit": observed_event.get(
                        "observation_unit"),
                    "label": None,
                    "reason": observed_event["label_basis"],
                })
                continue
            y = 1.0 if label == "occurred" else 0.0
            scored.append({
                "event": observed_event["event"],
                "canonical_event": observed_event.get("canonical_event"),
                "predicted_probability": round(float(probability), 6),
                "label": label,
                "label_basis": observed_event["label_basis"],
                "brier": round((float(probability) - y) ** 2, 6),
            })
            n_compared += 1
        evaluation.risk = {
            "eligibility": ("evaluable" if scored else "unobserved"),
            "scored": scored,
            "unscored": unscored,
            "observed_units": observed_units,
            "note": ("a Brier score is one sample of a probability's "
                     "quality, never a verdict from a single trajectory; "
                     "events are never averaged across different names; "
                     "`scored` holds prediction-observation PAIRS while "
                     "`observed_units` counts each real observation once, "
                     "however many predictions were bound to it"),
        }

    # -- interval ------------------------------------------------------------
    # An interval is the SAME observed value read as a range, so it shares
    # the benefit's eligibility: if the benefit may not be scored (an
    # identity problem or a method deviation), an interval "covering" that
    # value would be scoring the very comparison that was refused.
    if benefit is not None and benefit.interval is not None:
        if "observed" in summary.benefit and not _blocked("interval") \
                and not _blocked("benefit"):
            lo, hi = benefit.interval
            observed = summary.benefit["observed"]
            evaluation.interval = {
                "eligibility": "evaluable",
                "predicted_interval": [lo, hi],
                "observed": observed,
                "covered": bool(lo <= observed <= hi),
                # Width is reported alongside coverage: a "covered" verdict
                # from a mile-wide interval is not the same evidence as one
                # from a tight interval, and coverage alone cannot tell them
                # apart.
                "width": round(float(hi) - float(lo), 6),
            }
            n_compared += 1
        else:
            if "observed" not in summary.benefit:
                evaluation.interval = {
                    "eligibility": "unobserved",
                    "reason": "no comparable observed value for the "
                              "interval",
                }
            elif _blocked("interval"):
                evaluation.interval = {
                    "eligibility": eligibility_for(blocked["interval"]),
                    "reason": _block_reason("interval") + ": the real "
                              "outcome may not be this prediction's truth",
                }
            else:
                # Only the BENEFIT is blocked: the interval is drawn around
                # the SAME observed value, so it inherits the benefit's
                # eligibility rather than scoring a refused comparison.
                evaluation.interval = {
                    "eligibility": eligibility_for(blocked["benefit"]),
                    "reason": ("an interval drawn around the same observed "
                               "value shares the benefit's eligibility: "
                               + _block_reason("benefit")),
                }
    else:
        evaluation.interval = {
            "eligibility": "not_predicted",
            "reason": "the prediction saved no interval",
        }

    if summary.scope_status == "running":
        evaluation.state = "pending"
        evaluation.exclusion_reasons.append(
            "the bound scope has not ended: a running window has no final "
            "numbers to evaluate against")
    elif n_compared == 0:
        evaluation.state = "excluded"
        reasons = [f"{block.get('eligibility')}: {block.get('reason')}"
                   for block in (evaluation.benefit, evaluation.cost,
                                 evaluation.risk)
                   if isinstance(block, dict)
                   and block.get("eligibility") != "evaluable"]
        evaluation.exclusion_reasons.extend(
            reasons or ["no field was comparable"])
        evaluation.notes.append(
            "an excluded evaluation is neither a hit nor a miss: it never "
            "enters a denominator")
    else:
        evaluation.state = "evaluated"
    # The PER-FIELD index, built from the blocks so it can never disagree
    # with them. It answers "which parts may be calibrated?" one field at a
    # time, next to the single ``calibratable`` flag — being BOUND is the
    # weaker fact and is never read as a promise that all of this is
    # scoreable.
    evaluation.eligibility_summary = _evaluation_eligibility_index(evaluation)
    evaluation.calibratable = evaluation.state == "evaluated"
    return evaluation


def _evaluation_eligibility_index(evaluation
                                  ) -> Dict[str, Any]:
    """The per-field eligibility index of ONE evaluation.

    A convenience index over the blocks (benefit, each cost dimension, each
    risk event, the interval), each carrying its own eligibility and the
    reason when it is not ``evaluable``. It is DERIVED, never a second
    source of truth: the blocks remain the authority, so the two cannot
    drift.
    """
    out: Dict[str, Any] = {}
    benefit = evaluation.benefit or {}
    if benefit:
        entry: Dict[str, Any] = {"eligibility": benefit.get("eligibility")}
        if benefit.get("reason"):
            entry["reason"] = benefit["reason"]
        if benefit.get("metric"):
            entry["metric"] = benefit["metric"]
        out["benefit"] = entry
    cost = evaluation.cost or {}
    per_dim: Dict[str, Any] = {}
    for dim in (cost.get("per_dim") or {}):
        per_dim[dim] = {"eligibility": "evaluable"}
    for dim, reason in (cost.get("excluded") or {}).items():
        per_dim[dim] = {"eligibility": "unobserved", "reason": reason}
    if per_dim:
        out["cost"] = {"dimensions": per_dim,
                       "eligibility": cost.get("eligibility")}
    risk = evaluation.risk or {}
    events: Dict[str, Any] = {}
    for entry in (risk.get("scored") or []):
        events[str(entry.get("event"))] = {"eligibility": "evaluable"}
    for entry in (risk.get("unscored") or []):
        name = str(entry.get("event"))
        events[name] = {"eligibility": "unobserved"}
        if entry.get("reason"):
            events[name]["reason"] = entry["reason"]
    if events:
        out["risk"] = {"events": events,
                       "eligibility": risk.get("eligibility")}
    interval = evaluation.interval or {}
    if interval:
        out["interval"] = {"eligibility": interval.get("eligibility")}
        if interval.get("reason"):
            out["interval"]["reason"] = interval["reason"]
    return out


# ---------------------------------------------------------------------------
# 3. episode close-out
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCloseout:
    """The record that one episode ended, and what its predictions were
    worth.

    Created ONCE per (task, episode) by :func:`close_episode`; a repeated
    close is idempotent (the stored record is returned, nothing is
    re-counted). The close-out itself runs no solver, calls no model and
    triggers no induction.
    """

    closeout_id: str = field(default_factory=lambda: _new_id("cl"))
    task_id: str = ""
    episode_id: Optional[str] = None
    terminal_state: str = "completed"
    created_at: float = field(default_factory=time.time)
    #: The finish_task action that closed the episode, when one exists.
    finish_action_id: Optional[str] = None
    #: Evaluation records produced by this close-out (ids).
    evaluation_ids: List[str] = field(default_factory=list)
    #: Whether the calibration summary included this episode's samples.
    calibration_published: bool = False
    #: Unfinished actions at close time (reported, never fabricated away).
    unfinished_actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closeout_version": EPISODE_CLOSEOUT_VERSION,
            "closeout_id": self.closeout_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "terminal_state": self.terminal_state,
            "created_at": self.created_at,
            "finish_action_id": self.finish_action_id,
            "evaluation_ids": list(self.evaluation_ids),
            "calibration_published": bool(self.calibration_published),
            "unfinished_actions": list(self.unfinished_actions),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeCloseout":
        data = data or {}
        return cls(
            closeout_id=str(data.get("closeout_id") or _new_id("cl")),
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            terminal_state=str(data.get("terminal_state", "completed")),
            created_at=float(data.get("created_at", time.time())),
            finish_action_id=data.get("finish_action_id"),
            evaluation_ids=[str(e) for e in
                            (data.get("evaluation_ids") or [])],
            calibration_published=bool(
                data.get("calibration_published", False)),
            unfinished_actions=[str(a) for a in
                                (data.get("unfinished_actions") or [])],
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def close_episode(harness, task_id: str, episode_id: Optional[str], *,
                  terminal_state: str = "completed",
                  finish_action_id: Optional[str] = None,
                  min_calibration_samples: int =
                  DEFAULT_MIN_CALIBRATION_SAMPLES,
                  policy: Optional[CalibrationPolicy] = None,
                  ) -> Dict[str, Any]:
    """Close ONE episode and publish its experience calibration.

    The single M4 entry: it (1) refuses when actions are still running
    (pending, never a fabricated ending), (2) summarizes and evaluates
    every BOUND strategy-outcome prediction of the episode against its
    real outcome, (3) records the close-out once (idempotent), and (4)
    republishes the versioned calibration summary that later episodes'
    prediction contexts read.

    It does NOT run a solver, call the prediction model, or trigger
    induction. A failed/aborted/budget-exhausted episode closes honestly
    under its own terminal state — never dressed up as completed.

    **Publication is atomic and recoverable.** The evaluation records and
    the registry row are written FIRST (transaction 1); the summary is then
    rebuilt from the window and published in a SINGLE transaction that also
    flips the registry's ``published`` flag (transaction 2). A crash between
    the two leaves ``published=0``, which the next close (or an explicit
    ``rebuild``) detects and finishes — the episode is never "closed but
    unpublished" without a trace, and nothing is ever counted twice.
    """
    if terminal_state not in EPISODE_TERMINAL_STATES:
        raise ValueError(
            f"terminal_state must be one of {EPISODE_TERMINAL_STATES}")
    policy = policy or CalibrationPolicy.from_env()
    store = harness.store
    key = f"episode_closeout|{task_id}|{episode_id or ''}"
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is not None:
        stored = EpisodeCloseout.from_dict(store.loads(row["value"]))
        registry = store.get_closeout_registry(task_id, episode_id)
        result: Dict[str, Any] = {
            "closeout": stored.to_dict(),
            "already_closed": True,
            "note": ("this episode was already closed: the stored record "
                     "stands, nothing was re-counted or re-billed"),
        }
        if registry is None:
            # The close-out RECORD exists but the registry row does not: the
            # process died between the two writes (or the store predates the
            # table). Re-register it from the stored record — the registry
            # is derived index state — so the episode is not permanently
            # invisible to the window.
            backfilled = backfill_closeout_registry(harness)
            result["recovered_registry"] = backfilled > 0
            result["note"] += ("; the registry row was missing (an "
                               "interrupted close or a pre-registry store) "
                               "and has been rebuilt from the record")
        if registry is not None and not registry["published"]:
            # The previous close crashed between its two transactions: the
            # evaluations and the registry row exist, the summary does not.
            # Finish the publication now instead of leaving the episode
            # permanently unpublished (the alternative — reporting
            # already_closed and moving on — would silently starve every
            # later context of this episode's feedback).
            published = republish_calibration(
                harness, policy=policy,
                min_calibration_samples=min_calibration_samples)
            result["calibration_summary"] = published
            result["recovered_publication"] = True
            result["note"] += ("; the previous close had not published its "
                               "summary, so the publication was completed "
                               "now (nothing was re-counted)")
        elif registry is None or published_calibration_summary(
                harness) is None:
            # Either the registry was just rebuilt, or nothing has ever been
            # published: republish so the window is not left unreadable.
            result["calibration_summary"] = republish_calibration(
                harness, policy=policy,
                min_calibration_samples=min_calibration_samples)
        return result

    # Unfinished actions BLOCK the close-out: an episode with a running
    # action has no final state to evaluate against, and closing anyway
    # would freeze a "completed" record whose pending scopes can never be
    # back-filled (a re-close just returns the stored record). The
    # close-out stays PENDING — end the running actions first (or let them
    # finish), then close. This is a refusal to fabricate an ending, not
    # an error.
    unfinished = [a.action_id for a in harness.actions.query(
        task_id=task_id, episode_id=episode_id, status="running")]
    if unfinished:
        return {
            "closeout": None,
            "already_closed": False,
            "state": "pending",
            "unfinished_actions": unfinished,
            "note": ("the episode still has running action(s): the "
                     "close-out is REFUSED until they end. End them "
                     "(or let them finish), then close — a running scope "
                     "has no final numbers, and closing now would freeze "
                     "a record their results could never enter"),
        }

    closeout = EpisodeCloseout(
        task_id=task_id,
        episode_id=episode_id,
        terminal_state=terminal_state,
        finish_action_id=finish_action_id,
        unfinished_actions=[],
    )
    if terminal_state != "completed":
        closeout.notes.append(
            f"the episode ended in {terminal_state!r}: an honest terminal "
            "state, never reported as success")

    # Evaluate every BOUND strategy-outcome prediction of this episode.
    evaluations: List[StrategyPredictionEvaluation] = []
    for prediction in harness.strategy_predictions.query(
            task_id=task_id, episode_id=episode_id):
        if not prediction.trace.model_info.get("bound_action_id"):
            continue
        summary = summarize_real_outcome(harness, prediction)
        evaluation = evaluate_strategy_prediction(prediction, summary)
        evaluations.append(evaluation)

    # TRANSACTION 1 (ONE transaction): the evaluations, the close-out record
    # AND the registry row. Writing the record and the row separately left a
    # window in which a crash produced a close-out with no registry entry —
    # the episode then vanished from the window forever, because a re-close
    # saw the record and returned ``already_closed``. One transaction is
    # what makes the pair all-or-nothing; the registry row is written with
    # published=0 and flipped in transaction 2.
    for evaluation in evaluations:
        closeout.evaluation_ids.append(evaluation.evaluation_id)
    with store.transaction() as conn:
        for evaluation in evaluations:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (f"strategy_evaluation|{evaluation.evaluation_id}",
                 store.dumps(evaluation.to_dict())))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, store.dumps(closeout.to_dict())))
        conn.execute(
            "INSERT OR REPLACE INTO episode_closeouts "
            "(task_id, episode_id, closed_at, terminal_state, "
            " evaluation_ids, published, archived) VALUES (?,?,?,?,?,?,?)",
            (str(task_id), str(episode_id or ""), float(closeout.created_at),
             str(terminal_state), json.dumps(closeout.evaluation_ids), 0, 0))

    # TRANSACTION 2: rebuild the window's summary and publish it atomically.
    summary_block = republish_calibration(
        harness, policy=policy,
        min_calibration_samples=min_calibration_samples)
    closeout.calibration_published = True
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, store.dumps(closeout.to_dict())))

    # Automatic retention maintenance: a LIGHT check (one indexed COUNT of
    # registry rows outside the window and past the grace period). Only when
    # that count crosses the configured threshold does an archive pass run —
    # so retention is maintained without archiving on every single close.
    archive_result = maybe_auto_archive(harness, policy=policy)
    # Then bound the RAW evidence. Order matters and is deliberate: the
    # calibration publish above and the archive pass BEFORE it have already
    # taken every consolidation opportunity, so evicting the oldest episodes
    # now cannot race a step that still needs their raw facts. The pass is
    # GUARDED by one indexed COUNT, so an ordinary close-out walks nothing.
    window_result = maybe_enforce_evidence_window(harness,
                                                  calibration_policy=policy)
    return {
        "closeout": closeout.to_dict(),
        "already_closed": False,
        "evaluations": [e.to_dict() for e in evaluations],
        "calibration_summary": summary_block,
        "task_checks": _episode_task_check_summary(harness, task_id,
                                                   episode_id),
        "retention": archive_result,
        "evidence_window": window_result,
    }


def _episode_task_check_summary(harness, task_id: str,
                                episode_id: Optional[str]
                                ) -> Dict[str, Any]:
    """How many of the episode's executions carry a task-result verdict.

    Reported, never enforced: closing an episode whose answers were never
    checked is legitimate (the check may not be runnable yet), but the
    close-out must not let that read as "the answers were validated". A
    ``close-out`` is the end of the episode, NOT a statement that the answer
    was right.
    """
    counts: Dict[str, int] = {}
    unchecked = 0
    n = 0
    for record in harness.bank.query(task_id=task_id):
        n += 1
        verdict = task_check_state(record)
        if verdict is None:
            unchecked += 1
        else:
            counts[verdict] = counts.get(verdict, 0) + 1
    summary: Dict[str, Any] = {"n_executions": n, "verdicts": counts,
                              "unchecked": unchecked}
    if unchecked:
        summary["note"] = (
            f"{unchecked} of {n} execution(s) carry no task-result check: "
            "their answers' validity is UNKNOWN (feasibility is the solver's "
            "verdict on its own model, not on the task). The close-out ends "
            "the episode; it does not certify the answer")
    elif counts.get("failed"):
        summary["note"] = (
            f"{counts['failed']} execution(s) were confirmed NOT to satisfy "
            "the task: they remain recorded evidence with real cost, but they "
            "are not success samples")
    return summary


def _put_evaluation(store, evaluation: StrategyPredictionEvaluation
                    ) -> None:
    """Persist one evaluation record (idempotent by evaluation id)."""
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (f"strategy_evaluation|{evaluation.evaluation_id}",
             store.dumps(evaluation.to_dict())))


def get_evaluation(store, evaluation_id: str
                   ) -> Optional[StrategyPredictionEvaluation]:
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key=?",
        (f"strategy_evaluation|{evaluation_id}",)).fetchone()
    if row is None:
        return None
    return StrategyPredictionEvaluation.from_dict(store.loads(row["value"]))


def episode_closeout_record(harness, task_id: str,
                            episode_id: Optional[str]
                            ) -> Optional[EpisodeCloseout]:
    """The stored close-out of one episode, or None when it is open."""
    key = f"episode_closeout|{task_id}|{episode_id or ''}"
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    return EpisodeCloseout.from_dict(harness.store.loads(row["value"]))


# ---------------------------------------------------------------------------
# 4. the experience calibration summary
# ---------------------------------------------------------------------------


#: Meta key of the PUBLISHED calibration summary. The summary is derived
#: data, but it is published as ONE stored object so a normal prediction
#: read is a single row lookup instead of a full-history scan. It is
#: rewritten atomically whenever the window changes.
PUBLISHED_SUMMARY_KEY = "calibration_summary|published"


def calibration_window(harness, policy: Optional[CalibrationPolicy] = None,
                       *, readonly: bool = False
                       ) -> List[Dict[str, Any]]:
    """The CLOSED task-episodes that participate in the current calibration.

    Located through the registry's indexed ``closed_at`` ordering — the
    newest ``policy.window`` rows, and NOTHING else is read. This is the
    whole point of the registry: the window is "the first N registry rows",
    so a prediction read never scans or deserialises the evaluation history
    to discover which samples are recent.

    ``readonly=True`` skips the legacy-store backfill, so a caller that
    must not WRITE — a `--dry-run` preview, or any pure read — never
    leaves a migration behind. The trade-off is explicit: a pre-registry
    store reports an EMPTY window under ``readonly``, because registering
    its historical close-outs is a migration and a preview must not perform
    one. :func:`backfill_closeout_registry` is the explicit entry point for
    that migration.

    The default (``readonly=False``) keeps the historical behaviour: a
    store written BEFORE the registry existed has ``episode_closeout|...``
    meta records but no rows, and backfilling them here (once, idempotently)
    means an old database's episodes are not silently invisible to the
    window — ``orx calibration --rebuild`` on an unchanged legacy store used
    to report 0 window episodes and 0 evaluations.
    """
    if not readonly:
        backfill_closeout_registry(harness)
    policy = policy or CalibrationPolicy.from_env()
    return harness.store.closeout_registry(limit=policy.window)


def backfill_closeout_registry(harness, *, limit: Optional[int] = None
                               ) -> int:
    """Register any close-out that has a record but no registry row.

    The registry is DERIVED INDEX state, so rebuilding it from the records
    that already exist is always safe and idempotent. Two cases are covered,
    and they are different:

    - **a pre-registry store**: ``episode_closeout|...`` meta records exist,
      the table was created empty. Every record is registered with the
      close-out's OWN ``created_at`` as ``closed_at`` (so the window ordering
      is the historical one, not "whenever this migration ran");
    - **an interrupted close**: the close-out record and the evaluations
      were written, the registry row was not. The same scan registers it,
      which is what makes a crash between the two recoverable.
    """
    store = harness.store
    rows = store.conn.execute(
        "SELECT key, value FROM meta WHERE key LIKE 'episode_closeout|%'"
    ).fetchall()
    if not rows:
        return 0
    existing = {(r["task_id"], r["episode_id"] or "")
                for r in store.closeout_registry()}
    added = 0
    for row in rows:
        try:
            closeout = EpisodeCloseout.from_dict(store.loads(row["value"]))
        except Exception:
            continue
        identity = (str(closeout.task_id), str(closeout.episode_id or ""))
        if identity in existing:
            continue
        store.put_closeout_registry(
            closeout.task_id, closeout.episode_id,
            closed_at=float(closeout.created_at or time.time()),
            terminal_state=closeout.terminal_state,
            evaluation_ids=list(closeout.evaluation_ids or []),
            published=bool(closeout.calibration_published),
            archived=False)
        existing.add(identity)
        added += 1
        if limit is not None and added >= limit:
            break
    return added


def _evaluations_for_window(harness, window: Sequence[Dict[str, Any]]
                            ) -> List[StrategyPredictionEvaluation]:
    """Read ONLY the evaluations named by the window's registry rows.

    Each registry row carries its ``evaluation_ids``, so this reads exactly
    the window's payloads — no ``LIKE`` scan over the whole meta table, no
    deserialising rows that will be discarded. An id whose payload has been
    archived (or was never written) is skipped: the registry is the index,
    and a missing payload is a fact about the archive, not an error.
    """
    wanted: List[str] = []
    for row in window:
        wanted.extend(str(e) for e in (row.get("evaluation_ids") or []))
    if not wanted:
        return []
    out: List[StrategyPredictionEvaluation] = []
    for chunk_start in range(0, len(wanted), 400):
        chunk = wanted[chunk_start:chunk_start + 400]
        rows = harness.store.conn.execute(
            f"SELECT value FROM meta WHERE key IN "
            f"({','.join('?' for _ in chunk)})",
            [f"strategy_evaluation|{eid}" for eid in chunk]).fetchall()
        for row in rows:
            try:
                out.append(StrategyPredictionEvaluation.from_dict(
                    harness.store.loads(row["value"])))
            except Exception:
                continue
    return out


def _iter_closed_evaluations(harness, policy: Optional[CalibrationPolicy]
                             = None
                             ) -> List[StrategyPredictionEvaluation]:
    """Every evaluation of every closed episode IN THE CURRENT WINDOW.

    Kept under its original name because callers and tests use it, but its
    meaning is now bounded: the window is the sample set, so this is
    O(window) rather than O(history). An open episode's feedback still
    never calibrates anything (only registry rows exist for CLOSED
    episodes).
    """
    policy = policy or CalibrationPolicy.from_env()
    return _evaluations_for_window(harness,
                                   calibration_window(harness, policy,
                                                     readonly=True))


def _collect_observation_units(evaluation, unit_observations
                               ) -> None:
    """Add one evaluation's framework observations to the window's tally.

    Deduplicated by (event, unit id), so an execution observed by several
    bound predictions counts ONCE. The label counted is the PER-UNIT label
    (``units[unit_id]``) when the evaluation carries one: a scope whose two
    executions had different outcomes is ONE occurrence and ONE absence,
    not two occurrences. Only a legacy evaluation without per-unit labels
    falls back to the scope-level label (and, lacking unit ids, the episode
    identity).
    """
    risk = evaluation.risk or {}
    for event_name, observation in (risk.get("observed_units")
                                    or {}).items():
        if not isinstance(observation, dict):
            continue
        fallback_label = observation.get("label")
        fallback_basis = observation.get("label_basis")
        unit_ids = observation.get("unit_ids")
        per_unit = observation.get("units") or {}
        if not unit_ids:
            # Legacy evaluation without unit ids: fall back to the episode
            # identity so the unit is still counted once.
            unit_ids = [f"{evaluation.task_id}|"
                        f"{evaluation.episode_id or ''}"]
        bucket = unit_observations.setdefault(
            event_name, {"unit": observation.get("unit"),
                         "occurred": set(), "not_occurred": set(),
                         "unknown": set(), "seen": set()})
        for unit_id in unit_ids:
            key = str(unit_id)
            if key in bucket["seen"]:
                continue
            bucket["seen"].add(key)
            entry = per_unit.get(key)
            label = entry.get("label") if isinstance(entry, dict) \
                else fallback_label
            if label == "occurred":
                bucket["occurred"].add(key)
            elif label == "not_occurred":
                bucket["not_occurred"].add(key)
            else:
                bucket["unknown"].add(key)


def _observation_summary(harness, window: Sequence[Dict[str, Any]]
                         ) -> Dict[str, Any]:
    """The framework's occurrence tally over a window, from LIVE FACTS.

    Built from the executions the window's episodes really produced —
    NOT from stored evaluations. Two reasons this cannot read the frozen
    labels instead:

    - a stored evaluation's labels are a snapshot of what was known AT
      CLOSE-OUT. A late task check or an exclusion changes the FACTS, and
      a republished summary that kept serving the old labels would report a
      0% failure rate after every failure had been confirmed;
    - an episode with no BOUND prediction has no evaluation at all, yet its
      executions really happened. Reading only evaluations silently dropped
      those observations from the denominator.

    Counting is per OBSERVATION UNIT: every execution of every window
    episode contributes exactly one unit per event, whatever the number of
    predictions bound to it.

    **Bounded read.** The facts are read ONCE for the whole window, using
    the window's own ``evaluation_ids`` (the registry is the index), rather
    than rescanning the whole bank once per episode. The sliding window is
    what bounds the read; a per-episode full-bank scan made the read cost
    grow with TOTAL history even though the sample is window-limited.
    """
    facts = _window_executions(harness, window)
    unit_observations: Dict[str, Dict[str, Any]] = {}
    n_executions = 0
    n_episodes = 0
    for row in window:
        task_id = str(row["task_id"])
        episode_id = row.get("episode_id")
        n_episodes += 1
        records = facts.get((task_id, str(episode_id or "")), [])
        n_executions += len(records)
        if not records:
            continue
        # The scope of an episode's executions is complete once the episode
        # is closed (the close-out refuses while anything is running).
        observed = observe_episode_events(harness, records,
                                          task_id=task_id,
                                          episode_id=episode_id,
                                          scope_complete=True)
        for event_name, observation in observed["events"].items():
            units = observation.get("units") or {}
            unit_ids = observation.get("unit_ids") or []
            if not unit_ids:
                continue
            bucket = unit_observations.setdefault(
                event_name, {"unit": observation.get("unit"),
                             "occurred": set(), "not_occurred": set(),
                             "unknown": set(), "seen": set()})
            for unit_id in unit_ids:
                key = f"{task_id}|{episode_id or ''}|{unit_id}"
                if key in bucket["seen"]:
                    continue
                bucket["seen"].add(key)
                entry = units.get(str(unit_id)) or {}
                label = entry.get("label")
                if label == "occurred":
                    bucket["occurred"].add(key)
                elif label == "not_occurred":
                    bucket["not_occurred"].add(key)
                else:
                    bucket["unknown"].add(key)
    occurrence: Dict[str, Any] = {}
    for event_name, bucket in sorted(unit_observations.items()):
        labelled = len(bucket["occurred"]) + len(bucket["not_occurred"])
        occurrence[event_name] = {
            "observation_unit": bucket["unit"],
            "n_observation_units": len(bucket["seen"]),
            "n_labelled_units": labelled,
            "n_occurred": len(bucket["occurred"]),
            "n_not_occurred": len(bucket["not_occurred"]),
            "n_unknown": len(bucket["unknown"]),
            # Denominator is ALL labelled observation units — a DIFFERENT
            # sample set from mean_predicted_probability. Reported
            # separately and never subtracted from a mean probability.
            "unit_occurrence_rate": (
                round(len(bucket["occurred"]) / labelled, 6)
                if labelled else None),
            "rate_basis": ("labelled observation units (an execution or an "
                           "episode counted once, however many predictions "
                           "were bound to it)"),
            "source": ("derived from the current executions of the window's "
                       "episodes, not from stored evaluation labels"),
        }
    return {
        "occurrence": occurrence,
        "n_observation_units": n_executions,
        "n_observing_episodes": n_episodes,
    }


def _window_executions(harness, window: Sequence[Dict[str, Any]]
                       ) -> Dict[Tuple[str, str], List[Any]]:
    """The executions of every window episode, keyed ``(task, episode)``.

    **Bounded by construction.** Facts are read per TASK through the
    SQL-filtered ``bank.query(task_id=...)`` and the actions are read once
    per task and indexed by execution id — so neither the bank nor the
    action log is ever fully scanned, and the work is proportional to the
    window, not to the total history.

    Matching rules are exactly the ones the single-episode reader used:

    - an execution whose action records a DIFFERENT episode is another
      truth and is excluded;
    - an execution whose action cannot be resolved is KEPT when the
      episode filter is empty, because dropping it would silently shrink
      the occurrence denominator;
    - EXCLUDED facts are KEPT: ``exclude_execution`` withdraws a fact from
      the PREDICTION-comparison set, it does not claim the execution never
      ran, and the occurrence tally describes what was OBSERVED.
    - STAGED facts are KEPT too. An attempt that really ran but was never
      explicitly recorded (the harness abandoned it after an executor
      error, or has not yet called ``record``) is still an OBSERVATION:
      omitting it deleted a real failure from the occurrence tally and let
      a retry erase the attempt that preceded it. A staged fact that was
      later recorded has left the staging area, so the union never
      double-counts one attempt.
    """
    facts: Dict[Tuple[str, str], List[Any]] = {}
    for row in window:
        facts.setdefault(
            (str(row["task_id"]), str(row.get("episode_id") or "")), [])
    by_task: Dict[str, List[Any]] = {}
    actions_by_task: Dict[str, List[Any]] = {}
    for task_id, episode_id in facts:
        if task_id not in by_task:
            records = harness.bank.query(task_id=task_id)
            seen = {r.execution_id for r in records}
            # Union in the STAGED facts (never-recorded attempts), deduped
            # by id so an attempt is counted exactly once.
            for staged in harness.bank.pending(task_id=task_id):
                if staged.execution_id not in seen:
                    records.append(staged)
                    seen.add(staged.execution_id)
            by_task[task_id] = records
            actions_by_task[task_id] = harness.actions.query(task_id=task_id)
        actions = actions_by_task[task_id]
        for record in by_task[task_id]:
            action = next(
                (a for a in actions
                 if a.linked_execution_id == record.execution_id), None)
            if action is None:
                if episode_id == "":
                    facts[(task_id, episode_id)].append(record)
                continue
            if str(action.episode_id or "") == episode_id:
                facts[(task_id, episode_id)].append(record)
    return facts


def _episode_executions(harness, task_id: str,
                        episode_id: Optional[str]) -> List[Any]:
    """The executions that belong to ONE closed episode.

    Delegates to the same bounded, per-task read the window tally uses, so
    there is ONE matching rule in this module rather than two that can
    drift apart. Callers that already hold a window should use
    :func:`_window_executions` and read the facts once for the whole set.
    """
    window = [{"task_id": task_id, "episode_id": episode_id}]
    return _window_executions(harness, window)[
        (str(task_id), str(episode_id or ""))]


def build_calibration_summary(harness, *,
                              min_samples: int =
                              DEFAULT_MIN_CALIBRATION_SAMPLES,
                              policy: Optional[CalibrationPolicy] = None
                              ) -> Dict[str, Any]:
    """Aggregate the WINDOW's evaluations into a versioned summary.

    Grouped by (model identity, metric, unit, scope) so different
    definitions never mix — the same metric name under a different unit,
    scope or PREDICTING MODEL is a DIFFERENT group, never pooled. Risk
    events are scored PER EVENT NAME: different events are different random
    variables and their Brier scores are never averaged together. The
    sample threshold counts DISTINCT EPISODES (independent truths), not
    predictions: one truth bound to five re-planning predictions is ONE
    episode's evidence. An episode's identity is the FULL pair
    ``(task_id, episode_id)``.

    Two counting units are reported and NEVER conflated:

    - **prediction-observation pairs** (``n_scored`` / ``mean_brier`` /
      ``mean_predicted_probability`` / ``scored_occurrence_rate``): the
      sample for judging a probability, one per eligible prediction.
    - **observation units** (``n_observation_units`` /
      ``unit_occurrence_rate``): the framework's own count of how often an
      event really happened, once per execution (or per episode for the
      budget event). Several predictions bound to one execution are ONE
      unit — otherwise the occurrence rate would inflate with the number
      of predictions rather than the number of real events.

    ``mean_predicted_probability`` and ``scored_occurrence_rate`` share the
    SAME denominator (predictions with both a probability and a label), so
    their difference is a meaningful over/under-estimate signal.
    ``unit_occurrence_rate`` has its OWN denominator (all observed units)
    and is labelled separately — it must never be subtracted from a mean
    probability computed over a different sample set.

    Each group also reports DIRECTED statistics (mean signed benefit error,
    mean per-dimension cost log-ratio) so a reader can tell "historically
    too high" from "historically too low", not just "off by this much".
    A group below ``min_samples`` DISTINCT EPISODES reports
    ``insufficient_evidence`` — no reliability figure is invented.

    This is measured EXPERIENCE reliability, not a fitted calibrator and
    not a model weight: writing the summary does not promise future
    predictions improve. It is kept SEPARATE from the legacy knowledge
    prediction reliability (``prediction_class_reliability``).
    """
    policy = policy or CalibrationPolicy.from_env()
    window = calibration_window(harness, policy)
    evaluations = _evaluations_for_window(harness, window)
    groups: Dict[str, Dict[str, Any]] = {}
    exclusions: Dict[str, int] = {}
    corrected: List[Dict[str, Any]] = []
    for stored_evaluation in evaluations:
        # LIVE DERIVATION. The stored evaluation is history; what the
        # calibration reads is re-derived from the CURRENT facts, so a late
        # task check (in either direction) is reflected without rewriting
        # anything. The derived record is the one that counts; the stored
        # one is only the fallback when re-derivation is impossible.
        evaluation, live_changed = _live_evaluation(harness, stored_evaluation)
        if live_changed is not None \
                and live_changed.get("kind") in ("live_rederivation",
                                                 "rule_rebuild"):
            # A field moved (or the corrected rules re-scored a stored
            # `excluded` sample) but the sample still counts: reported, not
            # excluded. The corrected evaluation is what enters the means.
            corrected.append({**live_changed,
                              "counted": True,
                              "evaluation_id": stored_evaluation.evaluation_id})
        # A WITHDRAWN execution removes the sample entirely.
        is_corrected, correction = _live_validity_correction(
            harness, stored_evaluation)
        if is_corrected:
            exclusions["validity_corrected"] = exclusions.get(
                "validity_corrected", 0) + 1
            corrected.append({"evaluation_id":
                              stored_evaluation.evaluation_id,
                              **correction})
            continue
        if evaluation.state == "pending":
            exclusions["pending"] = exclusions.get("pending", 0) + 1
            continue
        benefit = evaluation.benefit or {}
        metric = str(benefit.get("metric") or "(none)")
        unit = str(benefit.get("unit") or "(none)")
        scope = str(evaluation.scope or "(none)")
        model_identity = str(evaluation.model_identity or "(unknown)")
        group_key = (f"strategy_outcome|{model_identity}|{metric}|{unit}"
                     f"|{scope}")
        group = groups.setdefault(
            group_key, {
                "n": 0, "all_episodes": set(),
                "benefit_episodes": set(), "benefit_abs_errors": [],
                "benefit_signed_errors": [],
                "cost_episodes": {}, "cost_log_errors": {},
                "cost_log_ratios": {},
                "interval_episodes": set(), "interval_covered": [],
                "interval_widths": [],
                "brier_episodes_by_event": {}, "brier_by_event": {},
                "probability_by_event": {}, "scored_label_by_event": {},
                "scored_episodes_by_event": {},
                "correlated_predictions": 0,
            })
        group["n"] += 1
        episode_key = (str(evaluation.task_id or ""),
                       str(evaluation.episode_id or ""))
        group["all_episodes"].add(episode_key)
        if benefit.get("eligibility") == "evaluable":
            group["benefit_episodes"].add(episode_key)
            if benefit.get("abs_error") is not None:
                group["benefit_abs_errors"].append(
                    float(benefit["abs_error"]))
            if benefit.get("signed_error") is not None:
                group["benefit_signed_errors"].append(
                    float(benefit["signed_error"]))
        cost = evaluation.cost or {}
        for dim, entry in (cost.get("per_dim") or {}).items():
            group["cost_episodes"].setdefault(dim, set()).add(episode_key)
            if entry.get("log_error") is not None:
                group["cost_log_errors"].setdefault(dim, []).append(
                    float(entry["log_error"]))
            if entry.get("log_ratio") is not None:
                group["cost_log_ratios"].setdefault(dim, []).append(
                    float(entry["log_ratio"]))
        interval = evaluation.interval or {}
        if interval.get("eligibility") == "evaluable":
            group["interval_episodes"].add(episode_key)
            group["interval_covered"].append(
                1.0 if interval.get("covered") else 0.0)
            if interval.get("width") is not None:
                group["interval_widths"].append(float(interval["width"]))
        risk = evaluation.risk or {}
        for entry in risk.get("scored") or []:
            # PER-EVENT accounting: different event names are different
            # variables; their scores are reported separately and never
            # averaged into one number. `scored` holds PREDICTION-
            # OBSERVATION PAIRS, so this counts pairs, not real events.
            event_name = str(entry.get("event") or "(unnamed)")
            group["brier_by_event"].setdefault(event_name, []).append(
                float(entry["brier"]))
            group["brier_episodes_by_event"].setdefault(
                event_name, set()).add(episode_key)
            group["scored_episodes_by_event"].setdefault(
                event_name, set()).add(episode_key)
            probability = entry.get("predicted_probability")
            if probability is not None:
                group["probability_by_event"].setdefault(
                    event_name, []).append(float(probability))
            label = entry.get("label")
            if label is not None:
                group["scored_label_by_event"].setdefault(
                    event_name, []).append(
                        1.0 if label == "occurred" else 0.0)

    def _mean(values: Sequence[float]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    def _evidence(basis: str, threshold: int, n_distinct: int,
                  n_samples: int) -> Optional[Dict[str, Any]]:
        """The per-statistic evidence verdict, or None when it has none.

        Every statistic carries its OWN basis and threshold: a group of 50
        episodes where only ONE predicted a timeout has ONE timeout sample,
        and pooling the group's count would let 49 unrelated samples "prove"
        a probability.

        ``None`` means the statistic was NEVER predicted or observed: that
        is an ABSENT statistic, not an under-sampled one, and reporting it
        as "insufficient evidence" would misread "nobody asked" as "too
        little data".
        """
        if n_samples == 0:
            return None
        if n_distinct >= threshold:
            return {"evidence": "measured", "n_distinct_episodes": n_distinct,
                    "min_distinct_episodes": int(threshold)}
        return {"evidence": "insufficient_evidence",
                "n_distinct_episodes": n_distinct,
                "min_distinct_episodes": int(threshold),
                "reason": (f"{n_distinct} distinct episode(s) < {threshold}: "
                           "no figure is claimed for this statistic "
                           "(correlated predictions of one truth are not "
                           "independent samples)")}

    out_groups: Dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        n = group["n"]
        distinct = len(group["all_episodes"])
        benefit_evidence = _evidence("benefit", min_samples,
                                     len(group["benefit_episodes"]),
                                     len(group["benefit_abs_errors"]))
        cost_evidence = {
            dim: verdict for dim, episodes in
            sorted(group["cost_episodes"].items())
            if (verdict := _evidence(
                f"cost.{dim}", min_samples, len(episodes),
                len(group["cost_log_errors"].get(dim) or []))) is not None}
        interval_evidence = _evidence("interval", min_samples,
                                      len(group["interval_episodes"]),
                                      len(group["interval_covered"]))
        brier_evidence = {
            event: verdict for event, episodes in
            sorted(group["brier_episodes_by_event"].items())
            if (verdict := _evidence(
                f"brier.{event}", min_samples, len(episodes),
                len(group["brier_by_event"].get(event) or []))) is not None}
        entry: Dict[str, Any] = {
            "n_samples": n,
            "n_distinct_episodes": distinct,
            "correlated_predictions": n - distinct,
            "mean_benefit_abs_error": _mean(group["benefit_abs_errors"]),
            # DIRECTED: positive = historically UNDER-predicted benefit.
            "mean_benefit_signed_error": _mean(
                group["benefit_signed_errors"]),
            "benefit_evidence": benefit_evidence,
            "mean_cost_log_error": {
                dim: _mean(values)
                for dim, values in sorted(
                    group["cost_log_errors"].items())},
            # DIRECTED: positive = historically UNDER-predicted cost.
            "mean_cost_log_ratio": {
                dim: _mean(values)
                for dim, values in sorted(
                    group["cost_log_ratios"].items())},
            "cost_evidence": cost_evidence,
            "interval_coverage": _mean(group["interval_covered"]),
            "n_interval_samples": len(group["interval_covered"]),
            "mean_interval_width": _mean(group["interval_widths"]),
            "interval_evidence": interval_evidence,
            "mean_brier_by_event": {
                event: _mean(values)
                for event, values in sorted(
                    group["brier_by_event"].items())},
            "n_brier_samples_by_event": {
                event: len(values)
                for event, values in sorted(
                    group["brier_by_event"].items())},
            "brier_evidence_by_event": brier_evidence,
            "mean_predicted_probability": {
                event: _mean(values)
                for event, values in sorted(
                    group["probability_by_event"].items())},
            # Same denominator as mean_predicted_probability: only
            # predictions that had BOTH a probability and a label.
            "scored_occurrence_rate": {
                event: _mean(values)
                for event, values in sorted(
                    group["scored_label_by_event"].items())},
        }
        # The GROUP basis summarises the episode count; the per-statistic
        # verdicts above name WHICH numbers are under-sampled.
        if distinct < min_samples:
            entry["reliability"] = None
            entry["basis"] = "insufficient_evidence"
            entry["note"] = (f"{distinct} distinct episode(s) < "
                             f"{min_samples}: no reliability figure is "
                             "claimed (correlated predictions of the same "
                             "truth do not count as independent samples)")
        else:
            entry["reliability"] = "measured_experience"
            entry["basis"] = "measured"
            entry["note"] = ("measured error/coverage statistics of past "
                             "closed-episode evaluations; a record of what "
                             "happened, never a promise that future "
                             "predictions improve")
            under_sampled = [
                key for key, verdict in
                ([("benefit", benefit_evidence)]
                 + [(f"cost.{d}", v) for d, v in cost_evidence.items()]
                 + [("interval", interval_evidence)]
                 + [(f"brier.{e}", v) for e, v in brier_evidence.items()])
                if verdict is not None
                and verdict["evidence"] != "measured"]
            if under_sampled:
                # The group has enough EPISODES, but not every statistic
                # does. `basis` stays the group-level verdict (the episode
                # count met the threshold), and `under_sampled_statistics`
                # names the numbers that must NOT be read as measured.
                entry["basis"] = "partially_measured"
                entry["under_sampled_statistics"] = sorted(under_sampled)
                entry["note"] += (
                    "; some statistics rest on fewer distinct episodes than "
                    "the threshold — see `*_evidence` and "
                    "`under_sampled_statistics`: " + ", ".join(
                        sorted(under_sampled)))
        out_groups[name] = entry

    # The framework's OWN occurrence statistics, per event name, counted by
    # observation unit (never by prediction count) from the CURRENT facts.
    live = _observation_summary(harness, window)
    occurrence = live["occurrence"]

    return {
        "calibration_version": CALIBRATION_SUMMARY_VERSION,
        "event_vocabulary_version": EVENT_VOCABULARY_VERSION,
        "protocol": "wm-so/1",
        "min_samples": int(min_samples),
        "min_samples_basis": ("distinct (task_id, episode_id) pairs, "
                              "applied PER STATISTIC (benefit, each cost "
                              "dimension, interval, each risk event)"),
        "window": policy.to_dict(),
        "n_window_episodes": len(window),
        "n_evaluations_total": len(evaluations),
        "n_evaluated": sum(1 for e in evaluations
                           if e.state == "evaluated"),
        "exclusions": exclusions,
        "validity_corrections": corrected,
        "groups": out_groups,
        "occurrence": occurrence,
        "applicability": "global_diagnostic",
        "applicability_note": (
            "these statistics carry NO strategy or problem-condition "
            "breakdown: they are a global diagnostic of the model's past "
            "error on this protocol, NOT a claim about the conditional "
            "bias of the candidate currently under consideration"),
        "note": ("experience calibration of the strategy-outcome "
                 "prediction service, from CLOSED episodes IN THE WINDOW "
                 "only; grouped by model identity/metric/unit/scope, risk "
                 "events scored per event name, and the sample threshold "
                 "counts DISTINCT EPISODES — kept separate from the legacy "
                 "knowledge-prediction reliability (a knowledge hit rate "
                 "never proves OR prediction accuracy), and never a model "
                 "weight or a fitted calibrator. `groups` counts "
                 "prediction-observation PAIRS; `occurrence` counts real "
                 "OBSERVATION UNITS. A sample whose answer was LATER "
                 "confirmed not to satisfy the task is counted under "
                 "exclusions.validity_corrected and leaves the means — the "
                 "stored evaluation itself is never rewritten"),
    }


def publish_calibration_summary(harness, summary: Dict[str, Any],
                                window: Sequence[Dict[str, Any]]
                                ) -> None:
    """Publish a summary and flip the window's registry rows atomically.

    ONE transaction writes the summary object AND marks every window
    episode ``published=1``. Splitting these would allow a state where the
    summary is visible but the registry still says "not published" (or the
    reverse), which the recovery path would then try to repair twice.
    """
    store = harness.store
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (PUBLISHED_SUMMARY_KEY, store.dumps(summary)))
        for row in window:
            conn.execute(
                "UPDATE episode_closeouts SET published=1 "
                "WHERE task_id=? AND episode_id=?",
                (str(row["task_id"]), str(row["episode_id"] or "")))


def republish_calibration(harness, *,
                          policy: Optional[CalibrationPolicy] = None,
                          min_calibration_samples: int =
                          DEFAULT_MIN_CALIBRATION_SAMPLES
                          ) -> Dict[str, Any]:
    """Rebuild the window summary from current state and publish it.

    The single place the published summary is written, so every trigger
    (a close-out, a late task check, an exclusion, an archive pass, an
    explicit rebuild) produces the same object by the same rules.
    """
    policy = policy or CalibrationPolicy.from_env()
    window = calibration_window(harness, policy)
    summary = build_calibration_summary(
        harness, min_samples=min_calibration_samples, policy=policy)
    summary["published_at"] = time.time()
    publish_calibration_summary(harness, summary, window)
    return summary


def published_calibration_summary(harness
                                  ) -> Optional[Dict[str, Any]]:
    """The stored published summary, or None when nothing was published yet.

    This is the read a NEW prediction context makes: ONE row lookup, no
    history scan. A store that has never published (an old database, or a
    project that never closed an episode) returns None and the caller
    reports the calibration as MISSING rather than scanning to build one.
    """
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?",
        (PUBLISHED_SUMMARY_KEY,)).fetchone()
    if row is None:
        return None
    try:
        return harness.store.loads(row["value"])
    except Exception:
        return None


def window_contains(harness, task_id: str, episode_id: Optional[str],
                    policy: Optional[CalibrationPolicy] = None) -> bool:
    """Whether one closed episode is currently IN the calibration window.

    Used by the re-publication triggers (a late check, an exclusion) to
    decide whether a change can affect the published summary at all: a
    correction on an episode that has already left the window cannot change
    the statistics, so no rebuild is needed.
    """
    policy = policy or CalibrationPolicy.from_env()
    # A pure READ: the legacy-store migration is not this call's job. A
    # store that has never been migrated reports an empty window here, and
    # the explicit entry points (`close_episode`, `calibration --rebuild`,
    # `archive-calibration`) perform the migration.
    for row in calibration_window(harness, policy, readonly=True):
        if str(row["task_id"]) == str(task_id) \
                and str(row["episode_id"] or "") == str(episode_id or ""):
            return True
    return False


def republish_if_in_window(harness, task_id: str,
                           episode_id: Optional[str], *,
                           policy: Optional[CalibrationPolicy] = None
                           ) -> Optional[Dict[str, Any]]:
    """Republish the summary IF the affected episode is in the window.

    The trigger every correction channel calls after writing a fact that
    could change a calibration sample (a late task check, an exclusion, a
    restore). It is deliberately cheap when it does nothing: locating the
    window is one indexed query, and an episode outside it returns None
    without rebuilding. Without this hook a stale label would keep being
    served to later predictions, which is exactly what the window is
    supposed to prevent.
    """
    policy = policy or CalibrationPolicy.from_env()
    if not window_contains(harness, task_id, episode_id, policy):
        return None
    return republish_calibration(harness, policy=policy)


def _evaluation_execution_ids(evaluation) -> List[str]:
    """The executions ONE evaluation actually compared against.

    Scoped precisely, because a correction must act on the execution it
    names: a late failure on execution A must not disqualify an evaluation
    of execution B in the same episode. The ids are read from the
    evaluation's own risk observations (which list the units it covered),
    falling back to the prediction's bound scope when absent.
    """
    ids: List[str] = []
    for observation in (evaluation.risk or {}).get("observed_units", {}
                                                   ).values():
        if not isinstance(observation, dict):
            continue
        for unit_id in observation.get("unit_ids") or []:
            unit_id = str(unit_id)
            # The budget event's "unit" is the episode, not an execution.
            if unit_id and unit_id not in ids and "|" not in unit_id:
                ids.append(unit_id)
    return ids


def _live_evaluation(harness, stored: StrategyPredictionEvaluation
                     ) -> Tuple[StrategyPredictionEvaluation,
                                Optional[Dict[str, Any]]]:
    """Re-derive ONE evaluation from the CURRENT facts.

    Returns ``(evaluation, correction_or_None)``. The STORED evaluation is
    never mutated — it stays the honest record of what was known at
    close-out, and it is what is returned when the live derivation is not
    possible (its prediction was archived, or the record is gone). What the
    calibration reads is the DERIVED one:

    - a task check that now FAILS turns the benefit observation into 0.0
      (the answer does not satisfy the task), while the measured COST stays
      exactly what it was: the answer was wrong, but it really did cost what
      it cost, so a correction must not take a valid measurement with it;
    - a task check that was WITHDRAWN restores the un-gated observation;
    - an execution WITHDRAWN from the evidence set (``exclude``) makes the
      evaluation uncountable — the fact itself is gone, so there is nothing
      left to calibrate against.

    A correction is reported with the FIELD that changed, so a reader can
    see exactly what the late fact moved.
    """
    prediction = harness.strategy_predictions.get(stored.prediction_id)
    if prediction is None:
        # The prediction itself is gone (archived, or from a build that
        # stored no prediction): the stored evaluation stands as history.
        return stored, None
    if not prediction.trace.model_info.get("bound_action_id"):
        return stored, None
    # A WINDOW-EVICTED fact is not a correction. When the raw executions a
    # stored evaluation compared have left the Evidence Bank through the
    # retention window (their source reference EXPIRED), the stored
    # evaluation is FINAL: the retention window is the documented cut-off,
    # and only genuinely PRESENT but non-counting facts (an ``exclude``)
    # are withdrawals. Conflating the two would read a retention policy as
    # a retraction of knowledge, which it is not.
    if any(harness.bank.get(eid) is None
           for eid in _evaluation_execution_ids(stored)):
        return stored, None
    try:
        summary = summarize_real_outcome(harness, prediction)
        derived = evaluate_strategy_prediction(prediction, summary)
    except Exception:      # a live derivation that cannot run changes nothing
        return stored, None

    # A WITHDRAWN fact: the executions this evaluation compared still exist
    # but no longer count as evidence at all (``exclude``).
    withdrawn = [eid for eid in _evaluation_execution_ids(stored)
                 if not _execution_counts_as_evidence(harness, eid)]
    if withdrawn:
        return derived, {
            "evaluation_id": stored.evaluation_id,
            "kind": "withdrawn_execution",
            "execution_ids": withdrawn,
            "field": "all",
            "reason": ("an execution this evaluation compared was withdrawn "
                       "from the evidence set: the stored sample no longer "
                       "counts"),
            "stored_state": stored.state,
            "derived_state": derived.state,
        }

    # Which fields moved between the stored record and the live facts?
    changed: Dict[str, Any] = {}
    stored_benefit = stored.benefit or {}
    derived_benefit = derived.benefit or {}
    if stored_benefit.get("observed") != derived_benefit.get("observed") \
            or bool(stored_benefit.get("task_check_gated")) \
            != bool(derived_benefit.get("task_check_gated")):
        changed["benefit"] = {
            "stored_observed": stored_benefit.get("observed"),
            "derived_observed": derived_benefit.get("observed"),
            "task_check_gated": derived_benefit.get("task_check_gated"),
            "cost_preserved": True,
        }
    if (stored.risk or {}).get("scored") != (derived.risk or {}).get("scored"):
        changed["risk"] = {"note": "risk labels moved with the live facts"}
    # A RULE REBUILD: the corrected attribution rules can move a stored
    # evaluation's STATE without any new fact arriving — an evaluation that
    # was `excluded` because a single unconfirmed config key used to
    # discard the whole sample now scores the cost dimension instead. This
    # is the "rebuild from retained facts and corrected rules" path: it is
    # reported with its BASIS, the prediction is never rewritten, and
    # repeated reads never add a second sample (the derivation is read-time
    # and keyed by evaluation id).
    stored_state = str(stored.state)
    derived_state = str(derived.state)
    if stored_state != derived_state:
        changed["state"] = {
            "stored_state": stored_state,
            "derived_state": derived_state,
            "stored_eligibility": copy.deepcopy(
                stored.eligibility_summary
                or stored._derive_eligibility()),
            "derived_eligibility": copy.deepcopy(
                derived.eligibility_summary
                or derived._derive_eligibility()),
            "basis": ("the identity-attribution rules changed: a field that "
                      "used to discard the WHOLE sample now blocks only the "
                      "dimension it really invalidates (an unconfirmed "
                      "config key no longer erases an unambiguous "
                      "observation)"),
        }
    if not changed:
        return derived, None
    return derived, {
        "evaluation_id": stored.evaluation_id,
        "kind": ("rule_rebuild" if "state" in changed
                 else "live_rederivation"),
        "fields": sorted(changed),
        "detail": changed,
        "reason": (
            "the sample was re-derived from the current facts and the "
            "corrected rules: the stored evaluation is kept as history, the "
            "derived record is what calibration counts, and the measured "
            "cost is preserved"
            + ("; the stored STATE changed because the attribution rules "
               "now block only the dimensions a problem really invalidates"
               if "state" in changed else
               ". A task-result verdict changed after the evaluation was "
               "written")),
        "stored_state": stored_state,
        "derived_state": derived_state,
    }


def _execution_counts_as_evidence(harness, execution_id: str) -> bool:
    """Whether an execution is still admissible evidence."""
    record = harness.bank.get(execution_id)
    if record is None:
        return False
    return str(record.source) == "executed"


def _live_validity_correction(harness, evaluation
                              ) -> Tuple[bool, Dict[str, Any]]:
    """Whether an evaluation must leave the means entirely.

    The narrow case: one of the executions it compared was WITHDRAWN from
    the evidence set (``exclude``), so the fact itself is gone and there is
    nothing left to calibrate against. A failed TASK CHECK is deliberately
    NOT this case — it re-derives the benefit observation to 0.0 and keeps
    the measured cost (see :func:`_live_evaluation`); excluding the whole
    evaluation would throw away a real measurement.

    A record that is ABSENT (evicted by the retention window) is deliberately
    NOT this case either: its source reference expired, the window is the
    documented cut-off, and the stored evaluation stands as the FINAL value.
    Only a fact still present but non-counting (``source != "executed"``) is
    a withdrawal.
    """
    task_id = str(evaluation.task_id or "")
    if not task_id:
        return False, {}
    execution_ids = _evaluation_execution_ids(evaluation)
    if not execution_ids:
        return False, {}
    for execution_id in execution_ids:
        record = harness.bank.get(execution_id)
        if record is None:
            # Window-evicted: the retention cut-off, not a withdrawal. The
            # stored evaluation is final (see _live_evaluation).
            continue
        if str(record.source) == "executed":
            continue
        if str(record.task_id) != task_id:
            continue
        correction = record.execution_features.get("correction") or {}
        return True, {
            "execution_id": execution_id,
            "task_id": task_id,
            "episode_id": evaluation.episode_id,
            "evaluated_execution_ids": execution_ids,
            "source": record.source,
            "evaluation_created_at": float(evaluation.created_at),
            "correction": (correction or None),
            "reason": ("the execution this evaluation compared was WITHDRAWN "
                       "from the evidence set (excluded): the stored "
                       "evaluation is kept as history but no longer counts "
                       "as a calibration sample"),
        }
    return False, {}


def calibration_summary_for_context(harness, *,
                                    policy: Optional[CalibrationPolicy] = None,
                                    model_identity: Optional[str] = None
                                    ) -> Dict[str, Any]:
    """The published calibration summary a NEW prediction context reads.

    This is a SINGLE-ROW read of the last published summary — no history
    scan, no rebuild. That is the point of publishing: a prediction read is
    O(1) however long the history is. When nothing has ever been published
    (an old database, or a project that never closed an episode) an EMPTY
    summary carrying the MISSING note is returned: the context records that
    no calibration was available rather than silently rebuilding one (which
    would make a read path write, and would let a prediction scan the whole
    history after all).

    ``model_identity`` is the ATTACHED provider's identity LABEL — the same
    string :func:`~or_harness.world_model.strategy_prediction
    .model_identity_label` produces for the grouping, never a caller guess
    and never the bare model name. ``None`` means the caller did not state
    an identity, and then NOTHING is filtered (the block is returned whole,
    marked ``filtered=False``). An identity that IS known — including the
    literal ``(unknown)`` — always filters, because "I do not know which
    model I am" must not be answered with every model's error statistics.
    """
    summary = published_calibration_summary(harness)
    if summary is None:
        policy = policy or CalibrationPolicy.from_env()
        return {
            "calibration_version": CALIBRATION_SUMMARY_VERSION,
            "event_vocabulary_version": EVENT_VOCABULARY_VERSION,
            "protocol": "wm-so/1",
            "window": policy.to_dict(),
            "n_window_episodes": 0,
            "n_evaluations_total": 0,
            "n_evaluated": 0,
            "exclusions": {},
            "validity_corrections": [],
            "groups": {},
            "occurrence": {},
            "applicability": "global_diagnostic",
            "missing": ("no calibration summary has been published yet: no "
                        "closed episode has been evaluated in this store, or "
                        "the summary predates this version. Run `orx "
                        "calibration --rebuild` to build one from the "
                        "current window"),
            "note": ("no published calibration is available; the context "
                     "records the gap rather than scanning history to fill "
                     "it"),
        }
    if model_identity is None:
        # No identity stated: the summary travels whole and says so, so a
        # reader is never left assuming a filter ran.
        summary = copy.deepcopy(summary)
        summary["filtered"] = False
        summary["filter_note"] = (
            "no model identity was supplied, so no filtering was applied: "
            "these groups may come from DIFFERENT models and must not be "
            "read as this predictor's own error statistics")
        return summary
    return _filter_calibration_by_model(summary, model_identity)


def _filter_calibration_by_model(summary: Dict[str, Any],
                                 model_identity: str
                                 ) -> Dict[str, Any]:
    """Keep only the groups produced by the SAME model identity.

    ``model_identity`` is the LABEL form (``model@version``), which is what
    the group keys carry. A different model is a different predictor, so its
    error statistics are not evidence about the attached one.

    ``(unknown)`` groups (predictions that recorded no identity) are kept
    ONLY when the current identity is itself ``(unknown)`` — otherwise a
    legacy group would silently stand in for a model it may not describe.
    The comparison is therefore exact on the LABEL, in both directions.
    """
    filtered = copy.deepcopy(summary)
    kept: Dict[str, Any] = {}
    withheld: List[str] = []
    for key, group in (summary.get("groups") or {}).items():
        parts = str(key).split("|")
        group_model = parts[1] if len(parts) > 1 else "(unknown)"
        if group_model == str(model_identity):
            kept[key] = copy.deepcopy(group)
        else:
            withheld.append(str(key))
    filtered["groups"] = kept
    filtered["filtered"] = True
    filtered["model_identity"] = str(model_identity)
    if withheld:
        filtered["withheld_groups"] = withheld
        filtered["filter_note"] = (
            f"{len(withheld)} group(s) from a different model identity were "
            f"withheld: they describe another model's errors and are not "
            f"evidence about {model_identity!r}")
    else:
        filtered["filter_note"] = (
            f"every group kept belongs to model identity "
            f"{model_identity!r}")
    return filtered


# ---------------------------------------------------------------------------
# 6. retention: archiving the detail (never the identity)
# ---------------------------------------------------------------------------


def archive_dir(harness) -> "os.PathLike[str]":
    """The archive directory under the harness home."""
    return harness.home / "archive" / ARCHIVE_DIRNAME


def _archive_files(harness) -> List["os.PathLike[str]"]:
    directory = archive_dir(harness)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.name.startswith("calibration-")
                  and p.name.endswith(".jsonl"))


def _archive_limits(harness, policy: CalibrationPolicy) -> Dict[str, Any]:
    """The archive's current size against its three caps (read-only)."""
    files = _archive_files(harness)
    total = sum(p.stat().st_size for p in files)
    return {
        "max_file_bytes": policy.archive_max_file_bytes,
        "max_total_bytes": policy.archive_max_total_bytes,
        "retention_days": policy.archive_retention_days,
        "n_files": len(files),
        "total_bytes": total,
    }


def _unregistered_closeouts_exist(harness) -> bool:
    """Whether the store has close-out records the registry does not know.

    Used ONLY to explain an empty preview on a pre-registry store. It is a
    COUNT against the meta table, not a migration: nothing is written, so a
    read path can report the state without changing it.
    """
    store = harness.store
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM meta WHERE key LIKE 'episode_closeout|%'"
    ).fetchone()
    if not row or not int(row["n"]):
        return False
    return len(harness.store.closeout_registry()) < int(row["n"])


def _archived_ids(harness) -> set:
    """Every record id already present in the archive.

    Read from the archive files' first token of each line (the record id),
    so a re-run of the archive pass is idempotent: an id already on disk is
    never written twice.
    """
    ids: set = set()
    for path in _archive_files(harness):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    archive_id = record.get("archive_id")
                    if archive_id:
                        ids.add(str(archive_id))
        except OSError:
            continue
    return ids


def _enforce_archive_limits(harness, policy: CalibrationPolicy
                            ) -> Dict[str, Any]:
    """Apply the archive's THREE caps: age, per-file size, total size.

    Whichever bites first evicts the OLDEST file. This is what makes "the
    archive is bounded" a true statement: a retention period alone would
    let a burst of activity store an unbounded volume, and a size cap alone
    would let stale data live forever. The TOTAL cap is honoured even when
    it means removing the last remaining file — a cap that silently yields
    to a single big file is not a cap.
    """
    removed: List[str] = []
    now = time.time()

    # 1. Age: delete files older than the retention period.
    if policy.archive_retention_days > 0:
        cutoff = now - policy.archive_retention_days * 86400.0
        for path in list(_archive_files(harness)):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed.append(path.name)
            except OSError:
                continue

    # 2. Per-file size: reported, not repaired. Records are rolled at write
    #    time so a file should not exceed the cap; a SINGLE record larger
    #    than the cap is the only way one can, and it is named here.
    oversized = [p.name for p in _archive_files(harness)
                 if p.stat().st_size > policy.archive_max_file_bytes]

    # 3. Total size: delete oldest files until the total fits. Unlike the
    #    per-file cap, this one is allowed to remove the LAST file: the
    #    total bound is the promise that the archive cannot grow forever.
    def _total() -> int:
        return sum(p.stat().st_size for p in _archive_files(harness)
                   if p.exists())
    while policy.archive_max_total_bytes > 0 \
            and _total() > policy.archive_max_total_bytes:
        current = _archive_files(harness)
        if not current:
            break
        oldest = current[0]
        try:
            oldest.unlink()
            removed.append(oldest.name)
        except OSError:
            break
    return {
        "removed_files": removed,
        "oversized_files": oversized,
        "remaining_files": len(_archive_files(harness)),
        "total_bytes": _total(),
    }


def _append_archive_records(harness, records: Sequence[Dict[str, Any]],
                            policy: CalibrationPolicy) -> Dict[str, Any]:
    """Append records, rolling to a NEW file as soon as the cap is reached.

    The roll is checked PER RECORD, not per batch: one batch of records can
    legitimately exceed the per-file cap, and checking only at batch start
    would leave a file far over the limit.
    """
    directory = archive_dir(harness)
    directory.mkdir(parents=True, exist_ok=True)
    files = _archive_files(harness)
    target = files[-1] if files else directory / "calibration-0001.jsonl"
    index = len(files) if files else 1
    written = 0
    written_to = target.name
    for record in records:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        try:
            size = target.stat().st_size
        except OSError:
            size = 0
        if policy.archive_max_file_bytes > 0 \
                and size and size + len(line.encode("utf-8")) \
                > policy.archive_max_file_bytes:
            index += 1
            target = directory / f"calibration-{index:04d}.jsonl"
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line)
        written += 1
        written_to = target.name
    return {"file": written_to, "written": written,
            "files": [p.name for p in _archive_files(harness)]}


def archive_calibration_detail(harness, *,
                               policy: Optional[CalibrationPolicy] = None,
                               dry_run: bool = False
                               ) -> Dict[str, Any]:
    """Move OUT-OF-WINDOW episode detail to the archive.

    Three scopes are respected, and they are NOT the same scope:

    1. **window** (``policy.window``): episodes still in it keep their
       detail online, because they calibrate.
    2. **grace** (``policy.late_check_grace_days``): a closed episode whose
       in-scope executions do NOT all carry a task verdict is kept online
       for this long, so a late check can still land on it. An episode
       whose executions are all checked is NOT held by the grace period —
       it leaves as soon as it is outside the window (no blanket extra
       retention).
    3. **archive caps**: per-file size, total size and age, so the archive
       itself is bounded.

    Only DETAIL is archived (evaluation payloads, prediction payloads,
    frozen context payloads). The registry tombstone stays ONLINE forever:
    it is what keeps a repeated close idempotent and the window locatable
    after the payloads are gone. Re-running is idempotent (an id already in
    the archive is skipped).

    Restoring an archived payload is possible for AUDIT, but a restored
    payload never re-enters the calibration automatically: the window is
    decided by the registry's ``closed_at`` ordering, not by whether a
    payload happens to be online.
    """
    policy = policy or CalibrationPolicy.from_env()
    store = harness.store
    # A dry run is a PREVIEW: it must not perform the legacy-store
    # migration, or "show me what would move" would silently register every
    # historical close-out. On a pre-registry store that means the preview
    # reports an empty window and SAYS SO, rather than writing rows.
    window = calibration_window(harness, policy, readonly=dry_run)
    if dry_run and not window and _unregistered_closeouts_exist(harness):
        return {
            "dry_run": True,
            "removed": {},
            "n_records": 0,
            "held_for_late_check": [],
            "archive_limits": _archive_limits(harness, policy),
            "pending_migration": {
                "reason": (
                    "this store has close-out records but no registry rows, "
                    "and a dry run does not perform the migration that would "
                    "register them"),
                "next": ("run `orx calibration --rebuild` (or one real "
                         "`orx archive-calibration`) once to register the "
                         "historical close-outs, then re-run the preview"),
            },
            "note": ("a preview reports; it never writes. The registry "
                     "migration is an explicit step"),
        }
    window_ids = {(str(r["task_id"]), str(r["episode_id"] or ""))
                  for r in window}
    already_archived = _archived_ids(harness)
    cutoff = time.time() - policy.late_check_grace_days * 86400.0

    # Candidate rows: closed, outside the window, not yet archived.
    candidates = [r for r in store.closeout_registry(archived=False)
                  if (str(r["task_id"]), str(r["episode_id"] or ""))
                  not in window_ids]

    to_write: List[Dict[str, Any]] = []
    to_clean: List[Dict[str, Any]] = []
    held_for_late_check: List[Dict[str, Any]] = []
    for row in candidates:
        task_id = str(row["task_id"])
        episode_id = row["episode_id"]
        if _episode_awaits_check(harness, task_id, episode_id):
            # An episode with an unchecked execution may still receive a
            # late verdict: keep it online until the grace period expires.
            if row["closed_at"] > cutoff:
                held_for_late_check.append({
                    "task_id": task_id, "episode_id": episode_id,
                    "reason": ("an in-scope execution carries no task-result "
                               "check and the grace period has not expired")})
                continue
        # Collect the detail payloads for this episode.
        payloads = _episode_detail_payloads(harness, task_id, episode_id)
        for payload in payloads:
            if payload["archive_id"] in already_archived:
                # ALREADY ON DISK: this record does not need writing again,
                # but its ONLINE copy must still be removed and its episode
                # marked. A retry after an interrupted pass ("file written,
                # delete not yet done") used to skip these entirely, so the
                # online copy and the archived flag were never cleaned up.
                to_clean.append(payload)
                continue
            payload["archived_at"] = time.time()
            to_write.append(payload)

    result: Dict[str, Any] = {
        "window_episodes": len(window),
        "candidate_episodes": len(candidates),
        "held_for_late_check": held_for_late_check,
        "n_records": len(to_write),
        "n_records_already_archived": len(to_clean),
        "dry_run": bool(dry_run),
        "policy": policy.to_dict(),
    }
    if dry_run:
        result["note"] = ("dry run: nothing was moved. The records listed "
                          "would be appended to the archive and removed "
                          "from the online store")
        return result

    write_info = _append_archive_records(harness, to_write, policy) \
        if to_write else {"file": None, "written": 0}
    # Only NOW remove the online payloads — after they are safely on disk.
    # The cleanup set is `to_write` (just confirmed on disk) UNION `to_clean`
    # (already on disk from an earlier, interrupted pass): a retry must be
    # able to finish the cleanup it did not complete.
    cleanup = to_write + to_clean
    evaluation_ids = [p["evaluation_id"] for p in cleanup
                      if p.get("record_type") == "evaluation"]
    prediction_ids = [p["prediction_id"] for p in cleanup
                      if p.get("record_type") == "contract_prediction"]
    context_ids = [p["context_id"] for p in cleanup
                   if p.get("record_type") == "prediction_context"]
    store.delete_evaluation_many(evaluation_ids)
    store.delete_contract_predictions(prediction_ids)
    store.delete_prediction_contexts(context_ids)
    # Mark the episodes whose detail is now fully archived (the tombstone
    # stays). An episode is marked only when nothing of it is still online,
    # so a partially-archived episode is not reported as done.
    for identity in {(p["task_id"], p["episode_id"] or "")
                     for p in cleanup}:
        task_id, episode_id = identity
        if _episode_detail_payloads(harness, task_id,
                                    episode_id or None):
            continue      # something is still online for this episode
        store.mark_closeout_archived(task_id, episode_id or None)

    limits = _enforce_archive_limits(harness, policy)
    result.update({
        "archive_write": write_info,
        "removed": {
            "evaluations": len(evaluation_ids),
            "contract_predictions": len(prediction_ids),
            "prediction_contexts": len(context_ids),
        },
        "archive_limits": limits,
        "note": ("only DETAIL was archived; the close-out registry rows "
                 "stay online so a repeated close remains idempotent and "
                 "the window remains locatable. A restored payload never "
                 "re-enters the calibration automatically. A re-run cleans "
                 "up records an earlier interrupted pass had written but "
                 "not yet removed"),
    })
    return result


def _episode_awaits_check(harness, task_id: str,
                          episode_id: Optional[str]) -> bool:
    """Whether any execution of the episode still lacks a task verdict.

    This is the DIRECTED exception the grace period applies to: an episode
    all of whose executions carry a verdict cannot benefit from a late
    check, so it is not held online. Only episodes genuinely awaiting a
    verdict are kept.
    """
    for record in harness.bank.query(task_id=task_id):
        action = harness.actions.by_execution(record.execution_id)
        if action is not None and episode_id is not None \
                and action.episode_id != episode_id:
            continue
        if task_check_state(record) is None:
            return True
    return False


def _episode_detail_payloads(harness, task_id: str,
                             episode_id: Optional[str]
                             ) -> List[Dict[str, Any]]:
    """Every archivable detail payload of one closed episode.

    Three record kinds are collected — evaluation payloads, strategy-outcome
    prediction payloads and frozen context payloads — so retention covers
    ALL the growing logs, not one query. Each carries an ``archive_id``
    that makes the archive idempotent.
    """
    store = harness.store
    out: List[Dict[str, Any]] = []
    registry = store.get_closeout_registry(task_id, episode_id) or {}
    for evaluation_id in registry.get("evaluation_ids") or []:
        row = store.conn.execute(
            "SELECT value FROM meta WHERE key=?",
            (f"strategy_evaluation|{evaluation_id}",)).fetchone()
        if row is None:
            continue
        out.append({
            "archive_id": f"evaluation|{evaluation_id}",
            "record_type": "evaluation",
            "evaluation_id": str(evaluation_id),
            "task_id": task_id,
            "episode_id": episode_id,
            "payload": row["value"],
        })
    for prediction_id in _prediction_ids_for(harness, task_id, episode_id):
        raw = store.get_contract_prediction(prediction_id)
        if raw is None:
            continue
        out.append({
            "archive_id": f"contract_prediction|{prediction_id}",
            "record_type": "contract_prediction",
            "prediction_id": str(prediction_id),
            "task_id": task_id,
            "episode_id": episode_id,
            "payload": raw,
        })
    for context_id in _context_ids_for(harness, task_id, episode_id):
        raw = store.get_prediction_context(context_id)
        if raw is None:
            continue
        out.append({
            "archive_id": f"prediction_context|{context_id}",
            "record_type": "prediction_context",
            "context_id": str(context_id),
            "task_id": task_id,
            "episode_id": episode_id,
            "payload": raw,
        })
    return out


def _prediction_ids_for(harness, task_id: str,
                        episode_id: Optional[str]) -> List[str]:
    rows = harness.store.conn.execute(
        "SELECT prediction_id FROM contract_predictions "
        "WHERE task_id=? AND episode_id IS ?",
        (str(task_id), episode_id)).fetchall()
    return [str(r["prediction_id"]) for r in rows]


def _context_ids_for(harness, task_id: str,
                     episode_id: Optional[str]) -> List[str]:
    rows = harness.store.conn.execute(
        "SELECT context_id FROM prediction_contexts "
        "WHERE task_id=? AND episode_id IS ?",
        (str(task_id), episode_id)).fetchall()
    return [str(r["context_id"]) for r in rows]


def maybe_auto_archive(harness, *,
                       policy: Optional[CalibrationPolicy] = None
                       ) -> Dict[str, Any]:
    """Light maintenance check after a close-out.

    ONE indexed count of registry rows outside the window and not yet
    archived. Only when that count crosses ``auto_archive_threshold`` does
    an archive pass run — so retention is maintained automatically without
    archiving on every close. Reported either way, so the caller can see
    that the check happened and what it decided.
    """
    policy = policy or CalibrationPolicy.from_env()
    store = harness.store
    window = calibration_window(harness, policy)
    window_ids = {(str(r["task_id"]), str(r["episode_id"] or ""))
                  for r in window}
    outside = [r for r in store.closeout_registry(archived=False)
               if (str(r["task_id"]), str(r["episode_id"] or ""))
               not in window_ids]
    if len(outside) < policy.auto_archive_threshold:
        return {
            "checked": True,
            "triggered": False,
            "n_outside_window": len(outside),
            "threshold": int(policy.auto_archive_threshold),
            "note": ("below the auto-archive threshold: nothing was "
                     "archived. The online detail is bounded by the window "
                     "plus the late-check grace period"),
        }
    result = archive_calibration_detail(harness, policy=policy)
    result.update({"checked": True, "triggered": True,
                   "n_outside_window": len(outside)})
    return result


# ---------------------------------------------------------------------------
# the bounded evidence window
# ---------------------------------------------------------------------------

def _episode_execution_ids(harness, task_id: str,
                           episode_id: Optional[str]) -> List[str]:
    """Every execution id that belongs to ONE episode (closed or not).

    Located the same way the rest of the module locates an episode's
    executions: per-task ``bank.query`` (SQL-filtered) plus the action log's
    ``linked_execution_id``. An execution whose action names a DIFFERENT
    episode is excluded; one whose action cannot be resolved belongs to the
    episode only when the episode filter is empty (the same rule
    ``_window_executions`` uses, so the occurrence denominator and the
    window's deletion set can never disagree)."""
    out: List[str] = []
    actions = harness.actions.query(task_id=task_id)
    for record in harness.bank.query(task_id=task_id):
        action = next((a for a in actions
                       if a.linked_execution_id == record.execution_id), None)
        if action is None:
            if not episode_id:
                out.append(record.execution_id)
            continue
        if str(action.episode_id or "") == str(episode_id or ""):
            out.append(record.execution_id)
    return out


def _open_episode_execution_ids(harness) -> Dict[Tuple[str, str], List[str]]:
    """Executions of every episode that has NO close-out registry row.

    An unclosed episode (running, pending, or abnormally unclosed) is exempt
    from the count bound — its outcome is not known, so evicting it would
    destroy live work. It is bounded instead by ``open_grace_days``: the
    whole episode is evicted once its newest execution is older than that.
    """
    registered = {(str(r["task_id"]), str(r["episode_id"] or ""))
                  for r in harness.store.closeout_registry()}
    by_task: Dict[str, List[str]] = {}
    for record in harness.bank.all():
        if record.task_id not in by_task:
            by_task[record.task_id] = []
        by_task[record.task_id].append(record.execution_id)
    out: Dict[Tuple[str, str], List[str]] = {}
    for task_id, exec_ids in by_task.items():
        actions = harness.actions.query(task_id=task_id)
        by_exec = {a.linked_execution_id: a for a in actions
                   if a.linked_execution_id}
        groups: Dict[str, List[str]] = {}
        for execution_id in exec_ids:
            action = by_exec.get(execution_id)
            episode_id = str(action.episode_id) if action is not None \
                and action.episode_id else ""
            groups.setdefault(episode_id, []).append(execution_id)
        for episode_id, ids in groups.items():
            if (str(task_id), episode_id) in registered:
                continue
            out[(str(task_id), episode_id)] = ids
    return out


def enforce_evidence_window(harness, *,
                            policy: Optional[EvidenceWindowPolicy] = None,
                            calibration_policy: Optional[CalibrationPolicy]
                            = None,
                            dry_run: bool = False) -> Dict[str, Any]:
    """Bound the Execution Evidence Bank to a recent window of complete episodes.

    The RUN ORDER is the whole design and is what makes this safe: the
    caller runs the existing offline maintenance FIRST (calibration publish /
    archive, induction), and this pass runs LAST. Eviction therefore never
    races a step that needs the raw facts; by the time it runs, every
    opportunity to consolidate has already been taken.

    Eviction is by COMPLETE EPISODE, oldest first (the close-out registry's
    ``closed_at``), and three things are never evicted:

    - episodes inside the calibration window (they calibrate);
    - episodes awaiting a task verdict within ``late_check_grace_days`` (a
      late check must still be able to land);
    - unclosed episodes younger than ``open_grace_days``.

    What the pass does NOT do is pretend the derivations depended on the raw
    rows. A knowledge entry keeps its own method content, verification
    range and evidence statement; a source execution id is a historical
    reference that MAY expire. Eviction is therefore reported as
    ``evicted`` — never as a refutation or a withdrawal (that is what
    ``exclude`` means, and the two must not be confused).

    ``dry_run`` writes NOTHING (no deletion, no vector removal, no text
    cleanup) and reports exactly what would be evicted. Re-running is
    idempotent. A crash mid-pass leaves some episodes evicted and some not;
    re-running finishes the job. User solve source files are NEVER touched —
    only rows in the banks' own store and derived index items.
    """
    policy = policy or EvidenceWindowPolicy.from_env()
    calibration = calibration_policy or CalibrationPolicy.from_env()
    window = calibration_window(harness, calibration, readonly=dry_run)
    window_ids = {(str(r["task_id"]), str(r["episode_id"] or ""))
                  for r in window}
    latency_cutoff = time.time() - calibration.late_check_grace_days * 86400.0
    open_cutoff = time.time() - policy.open_grace_days * 86400.0

    registry = harness.store.closeout_registry()   # newest first
    # Order oldest first for the "keep the newest N" selection. The window
    # holds the newest ``window_episodes`` rows; everything older that is not
    # protected is a candidate.
    kept: List[Dict[str, Any]] = []
    to_evict: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []
    for row in registry:
        identity = (str(row["task_id"]), str(row["episode_id"] or ""))
        if identity in window_ids:
            protected.append({"task_id": identity[0],
                              "episode_id": row["episode_id"],
                              "reason": "inside the calibration window"})
            continue
        if _episode_awaits_check(harness, identity[0],
                                 row["episode_id"]) \
                and row["closed_at"] > latency_cutoff:
            protected.append({"task_id": identity[0],
                              "episode_id": row["episode_id"],
                              "reason": ("an in-scope execution carries no "
                                         "task verdict and the late-check "
                                         "grace period has not expired")})
            continue
        # An episode whose executions are already gone has been evicted
        # before (the registry tombstone stays online for close idempotence
        # and window locatability). It is not a candidate again — otherwise
        # every later pass would keep re-reporting it.
        if not _episode_execution_ids(harness, identity[0],
                                      row["episode_id"]):
            continue
        eligible.append(row)
    # Newest eligible first: keep the first ``window_episodes``.
    eligible_sorted = sorted(eligible, key=lambda r: float(r["closed_at"]),
                             reverse=True)
    for index, row in enumerate(eligible_sorted):
        entry = {"task_id": str(row["task_id"]),
                 "episode_id": row["episode_id"],
                 "closed_at": float(row["closed_at"])}
        if index < policy.window_episodes:
            kept.append(entry)
        else:
            to_evict.append(entry)

    # Each evicted episode is removed WHOLE: a single execution's count never
    # splits a contrast/repair chain. Reported OLDEST FIRST (the order they
    # leave).
    to_evict.sort(key=lambda e: float(e["closed_at"]))
    evicted: List[Dict[str, Any]] = []
    for entry in to_evict:
        execution_ids = _episode_execution_ids(
            harness, entry["task_id"], entry["episode_id"])
        evicted.append({**entry, "n_executions": len(execution_ids),
                        "execution_ids": execution_ids})

    # Unclosed, aged-out episodes: bounded separately, reported separately.
    evicted_unclosed: List[Dict[str, Any]] = []
    for (task_id, episode_id), execution_ids in \
            _open_episode_execution_ids(harness).items():
        newest = max((harness.bank.get(e).created_at
                      for e in execution_ids if harness.bank.get(e) is not None),
                     default=None)
        if newest is None or newest > open_cutoff:
            continue
        evicted_unclosed.append({
            "task_id": task_id, "episode_id": episode_id or None,
            "newest_execution_at": newest,
            "n_executions": len(execution_ids),
            "execution_ids": execution_ids})

    # A single episode that is larger than the whole budget is KEPT whole
    # (never split) and reported: "bounded" is a count of episodes, and a
    # count is not a byte bound.
    total_retained_after = len(kept) + len(protected)
    over_budget = []
    if policy.window_episodes <= 0 and (evicted or evicted_unclosed):
        over_budget.append({
            "reason": ("window_episodes is 0: nothing may be retained, but a "
                       "single execution cannot be split from its episode — "
                       "the bound cannot be met without breaking a chain"),
        })

    result: Dict[str, Any] = {
        "policy": policy.to_dict(),
        "window_episodes": policy.window_episodes,
        "retained_episodes": total_retained_after,
        "protected": protected[:50],
        "n_protected": len(protected),
        "n_kept_recent": len(kept),
        "n_evicted": len(evicted),
        "evicted": evicted[:50],
        "n_evicted_unclosed": len(evicted_unclosed),
        "evicted_unclosed": evicted_unclosed[:50],
        "over_budget": over_budget,
        "dry_run": bool(dry_run),
        "note": ("a bounded WINDOW of complete episodes: an eviction is a "
                 "historical reference expiring, never a refutation or a "
                 "withdrawal. This bounds the Evidence Bank, not the whole "
                 "project directory"),
    }
    if dry_run:
        result["note"] = ("dry run: nothing was evicted and no index item "
                          "was touched. The episodes listed would leave the "
                          "bank, oldest first")
        return result

    # Apply: delete whole episodes, then clean ONLY what nothing references
    # any more (vectors; task-text versions no recorded/staged fact uses).
    removed_ids: List[str] = []
    removed_rows = 0
    for entry in evicted + evicted_unclosed:
        removed_rows += harness.bank.delete_episode_executions(
            entry["execution_ids"])
        removed_ids.extend(entry["execution_ids"])
    index_removed = None
    if removed_ids and harness.embedding_index is not None:
        from or_harness.strategy.embedding_index import LAYER_EXECUTION
        try:
            index_removed = harness.embedding_index.remove(
                LAYER_EXECUTION, sorted(set(removed_ids)))
        except Exception as exc:  # noqa: BLE001 - never block the window pass
            index_removed = {"removed": 0,
                             "deferred": f"{type(exc).__name__}: {exc}"}
    texts_removed = _cleanup_orphan_task_texts(harness)
    result.update({
        "removed_executions": removed_rows,
        "removed_ids": sorted(set(removed_ids))[:100],
        "index_removed": index_removed,
        "task_texts_removed": texts_removed,
        "note": ("whole episodes were evicted oldest first; derived vectors "
                 "and unreferenced task-text versions were cleaned in the "
                 "same pass. The knowledge entries those facts supported are "
                 "untouched — their claims and verification ranges are "
                 "self-contained"),
    })
    return result


def maybe_enforce_evidence_window(harness, *,
                                  policy: Optional[EvidenceWindowPolicy]
                                  = None,
                                  calibration_policy: Optional[CalibrationPolicy]
                                  = None) -> Dict[str, Any]:
    """Light check after a close-out; runs the full window pass only when
    the registry holds more CLOSED episodes than the window keeps.

    One indexed COUNT decides whether the expensive pass is worth running,
    so an ordinary close-out does not walk the bank. Reported either way, so
    the caller can see the check happened and what it decided."""
    policy = policy or EvidenceWindowPolicy.from_env()
    calibration = calibration_policy or CalibrationPolicy.from_env()
    window = calibration_window(harness, calibration, readonly=True)
    protected = {(str(r["task_id"]), str(r["episode_id"] or ""))
                 for r in window}
    total = harness.store.count_closeouts()
    if total <= policy.window_episodes:
        return {
            "checked": True,
            "triggered": False,
            "n_closed_episodes": total,
            "window_episodes": policy.window_episodes,
            "note": ("the bank holds no more closed episodes than the window "
                     "keeps: nothing was evicted"),
        }
    result = enforce_evidence_window(harness, policy=policy,
                                     calibration_policy=calibration)
    result.update({"checked": True, "triggered": True,
                   "n_closed_episodes": total,
                   "n_in_calibration_window": len(protected)})
    return result


def _cleanup_orphan_task_texts(harness) -> int:
    """Remove task-text versions no recorded or staged execution references.

    Idempotent and reference-counted: a version shared by several executions
    (or still staged) is kept. Only genuinely unreferenced text is removed —
    it exists to be a retrieval SOURCE DOCUMENT for executions, so a version
    nothing references is dead weight, not knowledge."""
    referenced = set()
    for record in harness.bank.all():
        if record.task_text_digest:
            referenced.add((str(record.task_id), str(record.task_text_digest)))
    for record in harness.bank.pending():
        if record.task_text_digest:
            referenced.add((str(record.task_id), str(record.task_text_digest)))
    removed = 0
    for key in harness.store.all_task_text_keys():
        identity = (str(key["task_id"]), str(key["text_digest"]))
        if identity in referenced:
            continue
        removed += harness.store.delete_task_text(*identity)
    return removed
