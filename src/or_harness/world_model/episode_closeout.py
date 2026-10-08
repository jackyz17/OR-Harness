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
from or_harness.core.schema import NON_CUMULATIVE_DIMENSIONS
from or_harness.core.schema import is_finite_number as _finite
from or_harness.core.schema import (
    task_check_block,
    task_check_state,
)
from or_harness.world_model.attribution import (
    EVALUATION_DIMENSIONS,
    binding_attribution,
    blocked_dimensions,
    eligibility_for,
    method_deviation,
    method_deviation_note,
)
from or_harness.world_model.cost_eligibility import (
    MEASURED,
    cost_eligibility,
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

#: Version of the OBSERVATION RULES: how a prediction's declared metric is
#: turned into a real observation. Bumped whenever a rule changes what an
#: observed value MEANS, so samples measured under one rule are never
#: pooled with another.
#:
#: ``wm-obs/2`` removes the task-check gate: a ``normalized_objective_gap``
#: observation is the solver's own figure, and a failed task check is a
#: SEPARATE fact carried alongside it rather than a rewrite of the quality
#: number (which mixed two different measurements into one).
#:
#: ``wm-obs/3`` (r12) makes TASK COMPLETION the PRIMARY benefit and records
#: the completion window's recovery structure (an intermediate confirmed
#: failure later recovered is reported as its own fact, never folded into
#: the final verdict) and the explicit cross-fact that a failed task check
#: does NOT represent effective completion. A ``wm-obs/2`` sample and a
#: ``wm-obs/3`` sample describe different primary benefits, so they are
#: never pooled.
OBSERVATION_RULE_VERSION = "wm-obs/3"

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


def _attempt_failed_without_result(record: Any) -> bool:
    """Whether an attempt RAN, failed, and produced no usable result.

    The completion OBSERVATION the task actually happened under: the script
    really executed (``execution_features.executed`` is not False — a
    sandbox-policy rejection never ran and is NOT an observation), the
    solver's own status is a failure (``error``/``timeout``), and the run
    is not feasible. Together these mean "this attempt did not complete the
    task" as a KNOWN fact, whatever the task check says (the check may be
    absent or ``insufficient`` — that is about the ANSWER's validity, not
    about whether the attempt completed).

    Deliberately NARROW: a crash that never ran (``executed=False``) is NOT
    this; a feasible-but-unchecked run is NOT this (the answer may be right,
    its validity is unknown).
    """
    features = getattr(record, "execution_features", None) or {}
    if features.get("executed") is False:
        return False  # the script never ran: not an observation of the task
    quality = getattr(record, "quality", None) or {}
    status = quality.get("status")
    if status not in ("error", "timeout"):
        return False
    if quality.get("feasible"):
        return False
    # A qualified solution with a gap would be a usable result; require the
    # absence of one (no objective, or an explicit infeasible/absent gap).
    return quality.get("objective") is None


def _observe_completion(summary: Any, records: Sequence[Any],
                        *, scope: str = "attempt") -> None:
    """Observe ``task_result_check_passed`` from the executions themselves.

    The observation is the execution's OWN task-check verdict — the check
    the agent already ran. **No check is run here**, and no check is
    invented.

    Mapping:

    * ``passed`` -> 1.0 — a declared basis held on the recorded values.
    * ``failed`` -> 0.0 — the answer was CONFIRMED not to satisfy the task.
    * ``insufficient`` / no check -> normally UNKNOWN, excluded from the
      sample. HOWEVER, when the predicted scope is ONE attempt and that
      attempt RAN, failed and produced no usable result, "this attempt did
      not complete the task" is a KNOWN fact regardless of the check — so it
      observes **0.0** (basis ``confirmed_failed_no_result``). The answer's
      correctness stays UNKNOWN (no verdict is fabricated). This is only for
      the ATTEMPT scope: a window/remaining scope must be judged by how the
      scope ENDED, never by its first failure alone.

    The window rule matches the quality channel: the LAST in-scope attempt
    that carries a verdict (or a known non-completion) is the observation,
    declared before evaluation rather than chosen per result.
    """
    observed: List[float] = []
    unknown = 0
    bases: List[str] = []
    # Per-attempt verdicts in attempt order, so a failure that a LATER
    # attempt recovered from is visible as its own fact. "The task failed at
    # attempt 2 and was completed at attempt 3" is a different story from
    # "the task never completed", and the primary benefit must not collapse
    # the two: the WINDOW rule below reads the LAST verdict, and the earlier
    # failure is reported beside it, never folded into the benefit.
    verdicts: List[Optional[str]] = []
    for record in records:
        state = task_check_state(record)
        verdicts.append(state)
        if state == "passed":
            observed.append(1.0)
            bases.append("task_check_passed")
        elif state == "failed":
            observed.append(0.0)
            bases.append("task_check_failed")
        elif state in (None, "insufficient") \
                and scope == "attempt" and _attempt_failed_without_result(
                    record):
            # The attempt RAN and failed with no usable result: a KNOWN
            # non-completion. The check may be absent or ``insufficient`` —
            # that concerns the ANSWER's validity, which stays UNKNOWN; the
            # ATTEMPT's non-completion is what is observed here. Only for
            # the attempt scope (a window is judged by how it ENDED).
            observed.append(0.0)
            bases.append("confirmed_failed_no_result")
            if state == "insufficient":
                summary.eligibility.setdefault("benefit_answer_validity", {
                    "eligibility": "unverified",
                    "reason": ("the attempt ran and failed without a usable "
                               "result (a KNOWN non-completion), but the "
                               "task check was INSUFFICIENT: the answer's "
                               "correctness is UNKNOWN and is not scored"),
                })
        else:
            unknown += 1
            bases.append("unknown")
    if not observed:
        summary.eligibility["benefit"] = {
            "eligibility": "unobserved",
            "reason": ("no in-scope execution carries a task-check verdict "
                       "and none is a known failed attempt: completion is "
                       "UNKNOWN (not 0.0) — run `check-task` and it becomes "
                       "observable"),
        }
        return
    final = observed[-1]
    # Recovered intermediate failures: an EARLY attempt confirmed failure
    # (verdict 0.0) followed by a LATER attempt that completed (final 1.0).
    # The main benefit is the FINAL verdict — a recovered failure is not a
    # failed task — but the recovery itself is carried as a fact so the two
    # cases are distinguishable.
    recovered = sum(1 for v in observed[:-1] if v == 0.0) \
        if final == 1.0 else 0
    summary.benefit = {
        "kind": "effective_completion",
        "metric": OBSERVABLE_COMPLETION_METRIC,
        "unit": "boolean",
        "observed": final,
        "rule": ("last in-scope attempt carrying a verdict or a known "
                 "non-completion"),
        "scope": scope,
        "completion_scope": scope,
        "basis": bases[-1],
        "n_observations": len(observed),
        "all_observations": observed,
        "verdicts": [v or "unknown" for v in verdicts],
        "source": ("the execution's own check-task verdict, or a confirmed "
                   "failed attempt with no usable result; no check is run "
                   "by the close-out"),
    }
    if recovered:
        summary.benefit["recovered_intermediate_failures"] = recovered
        summary.benefit["recovery_note"] = (
            f"{recovered} earlier in-scope attempt(s) were CONFIRMED not to "
            "satisfy the task, and a LATER attempt completed it. The primary "
            "benefit is the FINAL verdict (completed); the intermediate "
            "failures are reported as their own fact and are NOT counted as "
            "a failed task")
    if unknown:
        summary.benefit["unknown_checks"] = unknown
        summary.benefit["unknown_note"] = (
            f"{unknown} in-scope execution(s) had no usable task-check "
            "verdict and did not fail without a result: they contributed "
            "NOTHING to the sample — unchecked or insufficient validity is "
            "unknown, never a failure")
    if "benefit" not in summary.eligibility:
        summary.eligibility["benefit"] = {
            "eligibility": "evaluable",
            "reason": ("task-check verdict or a confirmed failed attempt "
                       "observed on the in-scope execution(s)"),
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
    #: Real cost, scoped to the prediction's DECLARED cost span (not the
    #: benefit scope); auxiliary overhead reported separately (it is real
    #: spend, but not what the prediction spoke about).
    cost: Dict[str, Any] = field(default_factory=dict)
    #: The executions whose spend the cost comparison covers (the observed
    #: span), separate from ``execution_ids`` (the BENEFIT scope's records).
    cost_records: List[str] = field(default_factory=list)
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
    #: The method the LAST in-scope record reported as PERFORMED (from the
    #: script's own receipt or a harness declaration), copied by value, or
    #: ``None`` when no record observed one (never a copy of the plan).
    method_actual: Optional[Dict[str, Any]] = None
    #: Failure error classes of the last in-scope record (``classify_failure``
    #: labels), so a failed pair carries its short failure type.
    failure_classes: List[str] = field(default_factory=list)
    #: The last in-scope record's solution status (``optimal``/``error``/
    #: ``timeout``...), a separate fact from the task-check verdict.
    execution_status: Optional[str] = None
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
            "cost_records": list(self.cost_records),
            "auxiliary_cost": copy.deepcopy(self.auxiliary_cost),
            "risk_events": copy.deepcopy(self.risk_events),
            "observed_events": copy.deepcopy(self.observed_events),
            "verification": copy.deepcopy(self.verification),
            "eligibility": copy.deepcopy(self.eligibility),
            "attribution": copy.deepcopy(self.attribution),
            "method_actual": copy.deepcopy(self.method_actual),
            "failure_classes": list(self.failure_classes),
            "execution_status": self.execution_status,
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

    ``records`` (optional) narrows each total to the records whose value is
    a MEASURED TRUTH: a DECLARED estimate (``agent_estimate``) or a
    single-side token figure is dropped from the total instead of standing
    in for a measurement. The dropped count is reported inline under
    ``excluded``, so a reader sees the number shrank and why.
    """
    dims: Dict[str, Any] = {}
    for dim in COST_DIMENSIONS:
        values: List[float] = []
        excluded = 0
        for index, cost in enumerate(vectors):
            if cost is None or dim not in cost.measured_dims():
                continue
            if records is not None and index < len(records):
                if cost_eligibility(records[index], dim) != MEASURED:
                    excluded += 1
                    continue
            values.append(float(getattr(cost, dim)))
        dims[dim] = _aggregated_dim(dim, values, len(vectors))
        if excluded:
            dims[dim]["excluded"] = excluded
    return dims


def _aggregated_dim(dim: str, values: Sequence[float],
                    n_items: int) -> Dict[str, Any]:
    """One dimension's aggregate, WITH an explicit comparable value.

    "Cannot be summed" is not "cannot be compared". For a CUMULATIVE
    dimension the comparable value IS the sum over the items that measured
    it. For a WALL-CLOCK span (``latency_s`` / ``remaining_latency_s``) the
    value is NEVER summed — overlapping prediction ranges would double-count
    the same span — but a SINGLE explicitly measured span (the common case:
    one real remaining-time measurement backfilled on the record) IS a
    comparable observation and participates in calibration. When several
    DISTINCT spans are present they cannot be combined, and no comparable
    value is produced (the reason is stated); identical spans reported more
    than once collapse to that one value.

    ``comparable`` is the ONE field a consumer should read for a
    comparison; ``total`` is kept for cumulative dimensions and stays
    ``None`` for a non-cumulative one (it is not a sum).
    """
    non_cumulative = dim in NON_CUMULATIVE_DIMENSIONS
    if non_cumulative:
        distinct = sorted({round(v, 6) for v in values})
        if len(distinct) == 1:
            comparable: Optional[float] = distinct[0]
            basis = ("one explicit wall-clock measurement; a span is "
                     "compared as itself and never summed across "
                     "overlapping ranges")
        elif len(distinct) > 1:
            comparable = None
            basis = (f"{len(distinct)} DISTINCT measured spans: overlapping "
                     "wall-clock ranges are never summed or combined, so no "
                     "single comparable value is produced")
        else:
            comparable = None
            basis = "not measured on the real scope"
        total = None
    else:
        comparable = round(sum(values), 6) if values else None
        basis = "sum over the items that measured it"
        total = comparable
    out: Dict[str, Any] = {
        "total": total,
        "comparable": {"value": comparable, "basis": basis,
                       "is_sum": not non_cumulative},
        "n_measured": len(values),
        "n_items": n_items,
        # For a cumulative dimension "complete" means every item measured
        # it; for a wall-clock span it means a single comparable value
        # exists (one explicit span, or consistent repeats).
        "complete": (bool(n_items) and len(values) == n_items
                     if not non_cumulative
                     else comparable is not None),
        "non_cumulative": non_cumulative,
        "per_item": ([round(v, 6) for v in values] if non_cumulative
                     else None),
    }
    if not non_cumulative and values:
        out["partial"] = len(values) != n_items
    return out


def _predicted_cost_dims(predicted_cost: Any) -> set:
    """Dimensions a cost prediction declared, with COMPONENT de-duplication.

    When the prediction declared BOTH a container (``remaining_latency_s``)
    and its component (``solver_runtime_s``), the component is not compared
    on top of the whole: the part and the whole are one spend, and scoring
    both would double-bill it. The component stays in the stored prediction
    (its diagnostic value is real) — it is only dropped from the COMPARISON.
    """
    from or_harness.core.schema import chargeable_dimensions
    if predicted_cost is None or predicted_cost.expected is None:
        return set()
    return chargeable_dimensions(predicted_cost.expected.measured_dims())


def _declared_cost_scope(prediction: Any) -> str:
    """The span a prediction's cost covers (``ExpectedCost.scope``).

    Defaults to ``remaining_to_task_end`` when the prediction carried no
    cost block: the build's cost contract is the remaining task, so an
    absent declaration is read as the default rather than silently narrowed
    to one attempt.
    """
    cost = getattr(prediction, "cost", None)
    scope = str(getattr(cost, "scope", "") or "")
    return scope or "remaining_to_task_end"


def _decision_anchor(harness, prediction,
                     anchor_action) -> Tuple[Optional[float], str]:
    """The decision ANCHOR and how it was established.

    r14: the anchor is the SHARED decision point, not a per-prediction
    timestamp. A strategy-outcome decision is recorded as ONE
    ``select_strategy`` action (``plan_next``), and every candidate predicted
    under it shares that decision's ``started_at`` — the predictions are
    created in a loop, so their own ``trace.created_at`` values differ and
    must NOT be used as "the" anchor. The decision id travels on each
    prediction's ``trace.model_info`` (written by ``plan_next``).

    Returns ``(anchor_seconds, basis)``:

    * ``decision_action`` — the recorded decision action's ``started_at``
      (the shared, correct anchor);
    * ``prediction_created_at_fallback`` — a LEGACY prediction with no
      decision id: its own ``trace.created_at`` is used but FLAGGED, so a
      reader knows the candidates were not anchored together;
    * ``bound_action_started_at_fallback`` — no trace timestamp either;
    * ``missing`` with ``None`` — nothing to anchor on.
    """
    info = getattr(getattr(prediction, "trace", None), "model_info", None) \
        or {}
    decision_id = info.get("decision_action_id")
    if decision_id:
        decision = harness.actions.get(str(decision_id))
        if decision is not None and decision.started_at is not None:
            return float(decision.started_at), "decision_action"
    created = getattr(getattr(prediction, "trace", None), "created_at", None)
    if created is not None:
        return float(created), "prediction_created_at_fallback"
    started = getattr(anchor_action, "started_at", None)
    if started is not None:
        return float(started), "bound_action_started_at_fallback"
    return None, "missing"


def _cost_scope_records(harness, prediction, action, bound_records,
                        candidate, cost_scope
                        ) -> Tuple[List[Any], str]:
    """The execution records whose spend the predicted cost is compared to.

    The span is chosen by the DECLARED cost scope, never by the benefit
    scope, so a ``remaining_to_task_end`` cost is observed against the whole
    remaining task (failed attempts and repairs included), not the first
    attempt. Overlap is avoided by keying on the execution itself: each
    in-episode execution contributes ONCE, whoever else also covers it.

    Returns ``(records, note)``. When the declared span cannot be observed
    the note says so and the records are the best available; the comparison
    itself is decided by completeness in the evaluation layer, so an
    unobservable span never fabricates a number.
    """
    if cost_scope == "attempt":
        return list(bound_records), (
            "cost scope 'attempt': the bound action's own execution only")
    if cost_scope == "strategy_window":
        return list(bound_records), (
            "cost scope 'strategy_window': the declared selection round's "
            "window")
    # remaining_to_task_end: the WHOLE episode's executions, from the
    # DECISION ANCHOR (when the prediction was made) to the end of the task.
    # Every execution that belongs to this task/episode contributes its
    # spend ONCE — a failed first attempt and the repaired retry both count,
    # so a predicted remaining total is not compared against the first
    # attempt.
    episode_records = _episode_executions(
        harness, candidate.task_id, candidate.episode_id)
    anchor, _basis = _decision_anchor(harness, prediction, action)
    if anchor is not None:
        # Keep executions that started at/after the anchor (the remaining
        # span). An execution with no resolvable action is KEPT when we
        # cannot place it in time, since dropping it would shrink the
        # remaining spend the prediction actually spoke about — the note
        # records that.
        kept: List[Any] = []
        unplaced = 0
        for record in episode_records:
            action_for = _action_for_execution(harness, record)
            if action_for is None or getattr(action_for, "started_at", None) \
                    is None:
                unplaced += 1
                kept.append(record)
                continue
            if float(action_for.started_at) >= float(anchor) - 1e-6:
                kept.append(record)
        note = (
            "cost scope 'remaining_to_task_end': every in-episode execution "
            "from the DECISION ANCHOR (the shared select_strategy decision "
            "action) to the end of the task aggregates ONCE (failed attempts "
            "and repairs included); overlapping ranges are not summed twice")
        if unplaced:
            note += (f"; {unplaced} execution(s) could not be placed in "
                     "time and were KEPT (dropping them would shrink the "
                     "real remaining spend)")
        return kept, note
    return episode_records, (
        "cost scope 'remaining_to_task_end': the decision anchor has no "
        "timestamp, so all in-episode executions aggregate (each counted "
        "once); the remaining span cannot be narrowed")


def _action_for_execution(harness, record) -> Optional[Any]:
    """The action that linked ONE execution, or None.

    Used only to place an execution relative to the decision anchor; a
    resolution failure is reported by the CALLER, never hidden.
    """
    try:
        actions = harness.actions.query(task_id=record.task_id)
    except Exception:  # noqa: BLE001 - a read failure must not crash close-out
        return None
    for action in actions:
        if action.linked_execution_id == record.execution_id:
            return action
    return None


def _closeout_created_at(harness, task_id: str,
                         episode_id: Optional[str]) -> Optional[float]:
    """The stored close-out record's ``created_at`` for this episode, or None.

    Used ONLY to bound the end-of-task search to the ORIGINAL solving phase:
    an action that ended AFTER the close-out was recorded belongs to a later
    offline pass (induction, audit, calibration re-evaluation) and must not
    push the original task's end forward.
    """
    try:
        key = f"episode_closeout|{task_id}|{episode_id or ''}"
        row = harness.store.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        payload = harness.store.loads(row["value"])
        created = payload.get("created_at")
        return float(created) if created is not None else None
    except Exception:  # noqa: BLE001 - a read failure just means "no bound"
        return None


def _task_end_boundary(harness, prediction, episode_id,
                       anchor: Optional[float],
                       closeout_created_at: Optional[float] = None
                       ) -> Tuple[Optional[float], str]:
    """The REAL end of the task, and how it was established.

    The remaining span runs from the decision anchor to the END OF THE TASK,
    covering modelling, coding, tool calls, solving, checking, repair and
    retries. The end is derived, in priority order, from the episode's OWN
    recorded actions:

    1. ``finish_task`` — the explicit task-termination action. Its
       ``ended_at`` is the ONLY event that states the task really ended
       (``basis="finish_task"``).
    2. otherwise the LATEST ``ended_at`` among the episode's relevant actions
       (``execute_strategy`` + ``verify``), which covers a check that ran
       after the last solve (``basis="last_action_not_task_end"``). This is
       NOT claimed to be the task end — it is the best available boundary and
       is FLAGGED as such.
    3. otherwise UNKNOWN (``None``, ``basis="no_end_boundary"``). The
       close-out's own ``created_at`` is NOT used as a substitute: "the
       close-out ran" is not "the task ended".

    **Original solving phase only.** Candidate actions are restricted to the
    prediction's OWN episode and to actions that ended at or before the
    close-out was recorded, so a LATER offline induction / audit / calibration
    re-evaluation cannot keep pushing the original task's end into the
    future.
    """
    try:
        actions = harness.actions.query(episode_id=episode_id)
    except Exception:  # noqa: BLE001 - a read failure is reported as unknown
        return None, "no_end_boundary"
    relevant = [a for a in actions
                if getattr(a, "action_type", "") in
                ("execute_strategy", "verify", "finish_task")]
    if closeout_created_at is not None:
        relevant = [a for a in relevant
                    if getattr(a, "ended_at", None) is not None
                    and float(a.ended_at) <= float(closeout_created_at) + 1e-6
                    or getattr(a, "ended_at", None) is None]
    # 1. finish_task is the only explicit termination event.
    finish_ends = [float(a.ended_at) for a in relevant
                   if getattr(a, "action_type", "") == "finish_task"
                   and getattr(a, "ended_at", None) is not None]
    if finish_ends:
        end = max(finish_ends)
        if anchor is None or end >= anchor:
            return end, "finish_task"
    # 2. latest relevant action end (NOT claimed to be the task end).
    ends = [float(a.ended_at) for a in relevant
            if getattr(a, "ended_at", None) is not None]
    if ends:
        end = max(ends)
        if anchor is None or end >= anchor:
            return end, "last_action_not_task_end"
    # 3. no real boundary.
    return None, "no_end_boundary"


def _observe_remaining_latency(summary, harness, prediction, anchor_action,
                               cost_records, cost_scope,
                               *, episode_id: Optional[str] = None,
                               closeout_created_at: Optional[float] = None
                               ) -> None:
    """Observe the REAL remaining wall-clock span (anchor -> task end).

    Measured from two REAL timestamps: the SHARED decision anchor and the
    task's REAL end (``_task_end_boundary``). When the end boundary is not
    established the dimension stays UNKNOWN with a reason — the framework
    never substitutes the script-execution cost for the remaining span, and
    never uses "the close-out ran" as the task end.

    Only meaningful for the ``remaining_to_task_end`` scope; an
    ``attempt``/``strategy_window`` cost does not claim the remaining span,
    so no ``remaining_latency_s`` is invented for it.
    """
    if cost_scope != "remaining_to_task_end":
        return
    anchor, anchor_basis = _decision_anchor(harness, prediction, anchor_action)
    end, end_basis = _task_end_boundary(harness, prediction, episode_id,
                                        anchor, closeout_created_at)
    # An EXPLICITLY MEASURED ``remaining_latency_s`` on the records is the
    # operator's own measurement of the whole span; it takes precedence over
    # the timestamp-derived span, which is a FALLBACK used only when no
    # record measured it. Overwriting a real backfilled measurement with a
    # narrower derived span would drop the observation the report needs.
    existing = summary.cost.get("remaining_latency_s") or {}
    if (existing.get("comparable") or {}).get("value") is not None:
        # Keep the record's own measurement; add the timestamp span as a
        # SEPARATE diagnostic when it can be derived, never as a rewrite.
        derived = (max(0.0, end - anchor)
                   if anchor is not None and end is not None else None)
        existing["observed_from"] = {
            "source": "record.remaining_latency_s",
            "anchor_basis": anchor_basis,
            "end_basis": end_basis,
        }
        if derived is not None:
            existing["timestamp_span"] = {
                "value": round(derived, 6),
                "note": ("anchor -> real task end: a diagnostic beside the "
                         "record's own measurement; it does NOT replace it")}
            existing["observed_from"]["timestamp_span"] = round(derived, 6)
        return
    if anchor is None or end is None:
        reason = ("the decision anchor has no start timestamp"
                  if anchor is None else
                  "no real task-end boundary (no finish_task, and no action "
                  "end that precedes the close-out): the remaining span "
                  "cannot be established, and the script-execution cost is "
                  "NOT substituted")
        summary.cost["remaining_latency_s"] = {
            "total": None,
            "comparable": {"value": None, "basis": reason, "is_sum": False},
            "n_measured": 0, "n_items": len(cost_records),
            "complete": False, "non_cumulative": True, "per_item": None,
            "eligibility": "missing",
            "observed_from": {"anchor_basis": anchor_basis,
                              "end_basis": end_basis},
        }
        return
    span = max(0.0, float(end) - float(anchor))
    complete = end_basis == "finish_task"
    summary.cost["remaining_latency_s"] = {
        "total": None,
        "comparable": {
            "value": round(span, 6),
            "basis": (f"measured from the decision anchor to the task end "
                      f"({end_basis}): a single real remaining span, never "
                      "a sum"),
            "is_sum": False},
        "n_measured": 1, "n_items": len(cost_records),
        "complete": complete, "non_cumulative": True,
        "per_item": [round(span, 6)],
        "eligibility": "evaluable",
        # A boundary that is not an explicit task-termination event is a
        # LOWER BOUND: the span may be longer. Reported, never dressed up.
        "boundary_is_task_end": complete,
        "observed_from": {"source": "action timestamps",
                          "anchor_basis": anchor_basis,
                          "end_basis": end_basis},
    }
    if not complete:
        summary.cost["remaining_latency_s"]["boundary_note"] = (
            "the end boundary is the latest relevant action, NOT an explicit "
            "task-termination event: the observed span is a LOWER BOUND on "
            "the remaining time")


def _derive_remaining_span(harness, anchor, cost_records,
                           *, prediction=None, episode_id=None,
                           closeout_created_at=None) -> Optional[float]:
    """The anchor -> real-task-end span from real timestamps, or None.

    A DIAGNOSTIC/FALLBACK: an explicitly measured ``remaining_latency_s`` on
    a record takes precedence. This derivation is used only when no record
    measured the dimension, and it needs BOTH the shared anchor and a real
    task-end boundary (see :func:`_task_end_boundary`).
    """
    if anchor is None:
        return None
    if prediction is not None:
        end, _basis = _task_end_boundary(harness, prediction, episode_id,
                                         anchor, closeout_created_at)
        return None if end is None else max(0.0, float(end) - float(anchor))
    end_candidates: List[float] = []
    for record in cost_records:
        action_for = _action_for_execution(harness, record)
        if action_for is not None and getattr(action_for, "ended_at", None) \
                is not None:
            end_candidates.append(float(action_for.ended_at))
    if not end_candidates:
        return None
    return max(0.0, max(end_candidates) - float(anchor))


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

    # The method that ACTUALLY ran, read from the real records (a script's
    # ``method_performed`` receipt, or a harness declaration) — never from
    # the prediction's plan. A partial execution or an encoding-stage
    # failure keeps whatever method was observed, and an unreported method
    # stays None (unknown), never a copy of the plan. The LAST in-scope
    # record is the attempt that produced the window's outcome.
    if records:
        actual = getattr(records[-1], "method_actual", None)
        if isinstance(actual, dict) and actual:
            summary.method_actual = copy.deepcopy(actual)
        summary.failure_classes = sorted(
            {str(f.error_class) for f in (records[-1].failures or [])
             if f.error_class})
        summary.execution_status = (records[-1].quality or {}).get("status")

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
            _observe_completion(summary, records, scope=candidate.scope)
        else:
            observed: List[float] = []
            # The TASK-check outcome is a SEPARATE fact, never a rewrite of
            # the quality observation. ``normalized_objective_gap`` measures
            # how well the SOLVER solved the model it was given; a failed
            # task check is a fact about the ANSWER satisfying the TASK and
            # is carried alongside (see ``task_check`` in the summary). The
            # two are different measurements and the same number is never
            # silently made to mean both.
            task_check_failed = 0
            task_check_states: List[Optional[str]] = []
            skipped_non_quality = 0
            for record in records:
                quality = record.quality or {}
                gap = quality.get("gap")
                status = quality.get("status")
                if status == "optimal":
                    value = 1.0
                elif status == "feasible" and gap is not None and _finite(gap):
                    value = max(0.0, 1.0 - float(gap))
                elif status == "feasible" and quality.get("feasible"):
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
                    # error / timeout / infeasible / unbounded: NO quality
                    # observation, even when a stray ``gap`` survived on the
                    # record. A failed run's gap is not a solution quality —
                    # reading it would let an error (gap=0 by default) score
                    # a full 1.0.
                    skipped_non_quality += 1
                    continue
                observed.append(value)
                state = task_check_state(record)
                task_check_states.append(state)
                if state == "failed":
                    task_check_failed += 1
            if skipped_non_quality and not observed:
                summary.eligibility.setdefault("benefit", {
                    "eligibility": "not_observable",
                    "reason": (f"{skipped_non_quality} in-scope execution(s) "
                               "carried no usable solution status "
                               "(error/timeout/infeasible/unbounded): a "
                               "failed run's gap is not a quality "
                               "observation"),
                })
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
                # The task check travels as its OWN fact next to the quality
                # observation, so a reader comparing "predicted 0.8,
                # observed 1.0" can also see the answer failed the task —
                # without the failure silently rewriting the quality number.
                if task_check_failed:
                    summary.benefit["task_check"] = {
                        "state": "failed",
                        "n_failed": task_check_failed,
                        "n_checked": len(task_check_states),
                        # r12: an explicit cross-fact. A high solver QUALITY
                        # beside a failed task check means "the solver solved
                        # the MODEL, but the ANSWER did not satisfy the TASK"
                        # — it is NOT an effective completion, and the
                        # quality number must never be read as one. This flag
                        # makes that reading impossible without changing the
                        # quality value itself.
                        "represents_effective_completion": False,
                        "note": (
                            f"{task_check_failed} in-scope observation(s) "
                            "carried a task check that confirmed the answer "
                            "does NOT satisfy the task. This is a SEPARATE "
                            "fact about task validity, not a rewrite of the "
                            "solver's own quality: the observed "
                            "normalized_objective_gap stays what the solver "
                            "achieved, and the task verdict is reported "
                            "alongside it"),
                    }
                elif any(s == "passed" for s in task_check_states):
                    summary.benefit["task_check"] = {
                        "state": "passed",
                        "n_checked": len(task_check_states),
                        "note": ("the in-scope answer passed the declared "
                                 "task check; the quality observation is "
                                 "still the solver's own figure"),
                    }
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
            _observe_completion(summary, records, scope=candidate.scope)
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
            _observe_completion(summary, records, scope=candidate.scope)
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
    # The observation span follows the prediction's DECLARED cost scope, so
    # a predicted span is compared against the SAME real span:
    #
    # - ``remaining_to_task_end`` (the default): the cost from the
    #   prediction's anchor to the END of the episode. A predicted total of
    #   300 tokens is compared against the WHOLE remaining spend (the failed
    #   first attempt PLUS the repaired retry), never the first attempt
    #   alone. Overlapping ranges are never summed: cumulative dimensions
    #   sum the in-episode executions ONCE each, and a wall-clock span uses
    #   a single explicit measurement.
    # - ``attempt``: the bound action's own execution.
    # - ``strategy_window``: the declared selection round's window.
    cost_scope = _declared_cost_scope(prediction)
    cost_records, scope_note = _cost_scope_records(
        harness, prediction, action, records, candidate, cost_scope)
    in_scope_costs = [r.cost for r in cost_records]
    summary.cost = _aggregate_costs(in_scope_costs, cost_records)
    summary.cost["scope"] = cost_scope
    summary.cost["scope_note"] = scope_note
    summary.cost_records = [r.execution_id for r in cost_records]
    # The REMAINING wall-clock span (anchor -> REAL task end), MEASURED from
    # real timestamps when they exist. This is the observation the primary
    # latency dimension needs: without it a model could predict ten minutes
    # as thirty seconds and never see an error. It is a SINGLE span, never
    # a sum; the end is an explicit task-termination event when one exists
    # (else a flagged lower bound), and it is left UNKNOWN (with a reason)
    # when a real boundary is missing — the framework never downgrades to
    # the script-execution cost, and never treats "the close-out ran" as the
    # task end.
    closeout_created_at = _closeout_created_at(harness, candidate.task_id,
                                               candidate.episode_id)
    _observe_remaining_latency(summary, harness, prediction, action,
                               cost_records, cost_scope,
                               episode_id=candidate.episode_id,
                               closeout_created_at=closeout_created_at)
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
        elif raw_classes and all(c == "unknown" for c in raw_classes):
            # r12: the executor CLASSIFIED the failure but could not place
            # it reliably (a stranger error matched no known pattern). The
            # cause is UNKNOWN, not "did not happen": marking these
            # not_occurred would assert an absence the framework cannot
            # establish, and would corrupt the risk calibration.
            for name in ("environment_failure", "implementation_failure"):
                _unit(name, execution_id, None,
                      "the failure was classified UNKNOWN (it matched no "
                      "reliable environment or implementation pattern): "
                      "neither cause can be established and neither is "
                      "recorded as NOT having happened")
        else:
            for name, wanted in (("environment_failure", "environment"),
                                 ("implementation_failure", "model")):
                if wanted in classes:
                    _unit(name, execution_id, "occurred",
                          "the executor recorded an ENVIRONMENT failure "
                          "(sandbox policy / missing module / unavailable "
                          "backend)"
                          if wanted == "environment" else
                          "the executor recorded the harness's OWN code "
                          "failing (an implementation failure, not a "
                          "modelling error)")
                elif "unknown" in classes and wanted not in classes:
                    # The ONLY recorded class is an unknown one: the
                    # absence of THIS cause is not established.
                    _unit(name, execution_id, None,
                          "the only recorded failure class is UNKNOWN: "
                          "the absence of this cause is not established")
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
    #: The OBSERVATION RULES version under which this evaluation's observed
    #: values were derived (see :data:`OBSERVATION_RULE_VERSION`). A reader
    #: compares it so samples measured under different rules are never
    #: silently pooled; a legacy record that wrote none reads as
    #: ``(unknown)`` rather than being assumed current.
    observation_rule_version: str = "(unknown)"
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
    #: The method the LAST in-scope record reported as PERFORMED, copied by
    #: value from the real record (never the plan). ``None`` = unobserved.
    method_actual: Optional[Dict[str, Any]] = None
    #: Failure error classes of the last in-scope record, so a failed pair
    #: carries its short failure type alongside the cost.
    failure_classes: List[str] = field(default_factory=list)
    #: The last in-scope record's own solution status
    #: (optimal/feasible/error/timeout/...), a separate fact from the
    #: task-check verdict.
    execution_status: Optional[str] = None
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
            "observation_rule_version": self.observation_rule_version,
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
            "method_actual": copy.deepcopy(self.method_actual),
            "failure_classes": list(self.failure_classes),
            "execution_status": self.execution_status,
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
            observation_rule_version=str(
                data.get("observation_rule_version") or "(unknown)"),
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
        observation_rule_version=OBSERVATION_RULE_VERSION,
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
    # The performed-vs-planned method comparison is REPORTED in the flat
    # ``method_deviation`` block so a reader can see the prose difference.
    # It blocks nothing: identity is decided by the structured fields, so a
    # re-worded receipt never invalidates a comparison whose strategy,
    # solver and config all matched. The plan is passed so a change can be
    # classified as the plan's DECLARED fallback versus a substantive
    # deviation — the two attribute differently (a declared fallback still
    # belongs to the original candidate; a substantive change belongs to
    # what actually ran).
    planned_method_obj = getattr(
        getattr(prediction, "candidate", None), "method", None)
    deviation = method_deviation(attribution.get("method_observed"),
                                 planned_method_obj)
    evaluation.method_deviation = copy.deepcopy(deviation)
    if deviation is not None:
        evaluation.notes.append(method_deviation_note(deviation))
    # The actual method and failure type are copied from the REAL summary
    # (records), never from the plan. Absent stays absent.
    if summary.method_actual:
        evaluation.method_actual = copy.deepcopy(summary.method_actual)
    evaluation.failure_classes = list(summary.failure_classes)
    evaluation.execution_status = summary.execution_status

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
        # The TASK-check outcome travels as its OWN fact next to the quality
        # observation. It does NOT rewrite the observed value: a reader
        # comparing "predicted 0.8, observed 1.0" must be able to see that
        # the answer nonetheless failed the task — the outcome-dimension
        # comparison is the solver's, the task verdict is separate.
        if summary.benefit.get("task_check"):
            evaluation.benefit["task_check"] = \
                copy.deepcopy(summary.benefit["task_check"])
        if benefit.interval is not None:
            # The interval's OWN semantics travel with it, so coverage is
            # read under the meaning the prediction declared.
            evaluation.benefit["interval_kind"] = benefit.interval_kind
            evaluation.benefit["interval_coverage"] = \
                benefit.interval_coverage
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
        predicted_dims = _predicted_cost_dims(predicted_cost)
        for dim in sorted(predicted_dims):
            real = summary.cost.get(dim) or {}
            # A value that was DECLARED (an ``agent_estimate``) or recorded
            # from a single token side is not a measured truth: the aggregate
            # already dropped it from the total, and the dropped count is
            # reported so the exclusion is visible rather than silent.
            dropped = int(real.get("excluded") or 0)
            if dropped:
                excluded[dim] = (
                    f"{dropped} of {real.get('n_items')} record(s) on the "
                    "real scope carried a DECLARED or single-side value, "
                    "which is not a measured truth: the total they would "
                    "have contributed to is not scored against")
                continue
            # The COMPARABLE value, not the (possibly withheld) sum: a
            # wall-clock span offers its single explicit measurement here
            # even though it is never summed. "Cannot be summed" is not
            # "cannot be compared".
            comparable = (real.get("comparable") or {})
            actual_value = comparable.get("value")
            if actual_value is None:
                reason = comparable.get("basis")
                if real.get("n_items") and not real.get("n_measured"):
                    excluded[dim] = ("not measured on the real scope"
                                     + (f" ({reason})" if reason else ""))
                elif real.get("n_items"):
                    excluded[dim] = (
                        "no single comparable value on the real scope: "
                        + (reason or "the measured values cannot be "
                                     "combined"))
                else:
                    excluded[dim] = "no real execution to measure it"
                continue
            if not real.get("complete"):
                excluded[dim] = ("partially measured on the real scope "
                                 f"({real.get('n_measured')}/"
                                 f"{real.get('n_items')}): an incomplete "
                                 "total is not a truth to score against")
                continue
            p = float(getattr(predicted_cost.expected, dim))
            a = float(actual_value)
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
            # SEMANTICS BY KIND. An ``outcome`` interval is a claim about
            # ONE run's value, so a single observation decides coverage. A
            # ``mean`` interval is a claim about the AVERAGE over repeated
            # runs — a single observation is NOT its claim (its own
            # definition says so), and this build has NO mean-observation
            # channel, so a mean interval is NOT covered/uncovered by one
            # value: it is reported UNOBSERVABLE with the reason. It never
            # enters a coverage numerator or denominator on a single value.
            if benefit.interval_kind == "mean":
                evaluation.interval = {
                    "eligibility": "unobservable",
                    "predicted_interval": [lo, hi],
                    "observed": observed,
                    "interval_kind": "mean",
                    "interval_coverage": benefit.interval_coverage,
                    "width": round(float(hi) - float(lo), 6),
                    "reason": ("a MEAN interval claims the average over "
                               "repeated runs; a single observation does "
                               "not decide its coverage, and this build has "
                               "no mean-observation channel. The interval is "
                               "recorded and grouped by kind, never scored "
                               "on one value"),
                }
            else:
                evaluation.interval = {
                    "eligibility": "evaluable",
                    "predicted_interval": [lo, hi],
                    "observed": observed,
                    "covered": bool(lo <= observed <= hi),
                    # The interval's OWN semantics travel with the block, so
                    # its coverage is read under the meaning the prediction
                    # declared (an outcome interval and a mean interval make
                    # different claims and are never averaged together).
                    "interval_kind": benefit.interval_kind,
                    "interval_coverage": benefit.interval_coverage,
                    # Width is reported alongside coverage: a "covered"
                    # verdict from a mile-wide interval is not the same
                    # evidence as one from a tight interval, and coverage
                    # alone cannot tell them apart.
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
    cost_warnings = _unmeasured_cost_warning(harness, task_id, episode_id)
    if cost_warnings:
        closeout.notes.append(cost_warnings["note"])
    return {
        "closeout": closeout.to_dict(),
        "already_closed": False,
        "evaluations": [e.to_dict() for e in evaluations],
        "calibration_summary": summary_block,
        "task_checks": _episode_task_check_summary(harness, task_id,
                                                   episode_id),
        "retention": archive_result,
        "evidence_window": window_result,
        **({"unclosed_bound_predictions": unclosed}
           if (unclosed := _unclosed_bound_predictions(harness, task_id,
                                                        episode_id)) else {}),
        **({"cost_completeness_warnings": cost_warnings}
           if cost_warnings else {}),
    }


def _unclosed_bound_predictions(harness, task_id: str,
                                episode_id: Optional[str]
                                ) -> List[Dict[str, Any]]:
    """Bound predictions of THIS task that live OUTSIDE the closed episode.

    A bound prediction under a DIFFERENT episode id (including the ``None``
    episode) is never evaluated by a close-out that matches (task, episode)
    exactly, so its real execution can silently miss calibration. Reported,
    never auto-collected: the episode is the caller's unit and the framework
    does not close an episode the caller did not name. The report names the
    other episodes and their bound predictions so they can be closed
    explicitly.
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[str, List[str]] = {}
    for prediction in harness.strategy_predictions.query(task_id=task_id):
        if not prediction.trace.model_info.get("bound_action_id"):
            continue
        other = prediction.candidate.episode_id
        if other == episode_id:
            continue
        seen.setdefault("" if other is None else str(other), []).append(
            prediction.prediction_id)
    for other, ids in seen.items():
        out.append({
            "episode_id": other or None,
            "prediction_ids": sorted(ids),
            "note": ("this episode's bound predictions are NOT evaluated by "
                     "this close-out (which matches the episode id exactly); "
                     "close it explicitly with `orx close-episode --task "
                     f"{task_id} --episode {other!r}` to bring them into "
                     "calibration"),
        })
    return out


def _unmeasured_cost_warning(harness, task_id: str,
                             episode_id: Optional[str]) -> Dict[str, Any]:
    """Which cost dimensions are still UNKNOWN across the task's records.

    Reported, never enforced: closing an episode with unmeasured dimensions
    is legitimate (the host may not have produced its usage report), but the
    close-out must not let that read as "the cost was zero". Calibration
    already accepts ONLY measured dimensions — this warning makes the gap
    VISIBLE at the moment it can still be repaired (`amend-cost`), instead
    of surfacing later as a silently shrunken sample.
    """
    records = list(harness.bank.query(task_id=task_id))
    if not records:
        return {}
    unknown: Dict[str, List[str]] = {}
    for record in records:
        measured = record.cost.measured_dims() if record.cost else set()
        for dim in ("llm_tokens", "tool_calls"):
            if dim not in measured:
                unknown.setdefault(dim, []).append(record.execution_id)
    if not unknown:
        return {}
    parts = []
    for dim in sorted(unknown):
        parts.append(f"{dim} on {len(unknown[dim])} of {len(records)} "
                     f"record(s)")
    return {
        "unknown_dimensions": {dim: sorted(ids)
                               for dim, ids in unknown.items()},
        "n_records": len(records),
        "note": (
            "cost dimensions still UNKNOWN at close-out ("
            + "; ".join(parts) + "): unknown is never zero, and these "
            "dimensions support no cost claim and no calibration until "
            "backfilled. Repair with `orx amend-cost <execution_id> "
            "--usage-file <host report>` (or --usage-host <host>) — the "
            "host's attempt-level usage report is the real observation; a "
            "hand-typed --override stays a declaration"),
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

#: Where the DERIVED paired-feedback block is stored at publish time, so the
#: prediction read path serves it in one row lookup instead of re-deriving it
#: from the window on every context build.
PAIRED_FEEDBACK_KEY = "paired_feedback|published"


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
        # The observation-rules version is part of the grouping key: samples
        # measured under different rules are DIFFERENT measurements and must
        # never pool. A legacy evaluation that wrote none groups under
        # ``(unknown)`` rather than being assumed current.
        obs_rule = str(getattr(evaluation, "observation_rule_version",
                               "") or "(unknown)")
        group_key = (f"strategy_outcome|{model_identity}|{metric}|{unit}"
                     f"|{scope}|{obs_rule}")
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
                # r12 COVERAGE: per event name, how many predictions (a)
                # predicted it AT ALL, (b) predicted a probability for it,
                # and (c) were answered with a reliable label. The gap
                # between (a) and (b) is "asked but no basis" (honest
                # unknown); between (b) and (c) is "predicted but not
                # scoreable". Coverage is reported so a model that skips a
                # risk event is visible, and a MISSING event is never
                # silently read as probability zero.
                "asked_by_event": {}, "probabilized_by_event": {},
                "labelled_by_event": {},
                "correlated_predictions": 0,
            })
        group["n"] += 1
        episode_key = (str(evaluation.task_id or ""),
                       str(evaluation.episode_id or ""))
        group["all_episodes"].add(episode_key)
        # Coverage: every prediction is ASKED about every observable event
        # (the prompt mandates considering implementation_failure and
        # task_check_failed); it may answer with a probability or with an
        # honest unknown. Counting the ask once per prediction keeps the
        # coverage denominator the same whether the model answered or not.
        for _event in OBSERVABLE_RISK_EVENTS:
            group["asked_by_event"][_event] = \
                group["asked_by_event"].get(_event, 0) + 1
        for _entry in (evaluation.risk or {}).get("scored") or []:
            _e = str(_entry.get("event") or "(unnamed)")
            group["probabilized_by_event"][_e] = \
                group["probabilized_by_event"].get(_e, 0) + 1
            group["labelled_by_event"][_e] = \
                group["labelled_by_event"].get(_e, 0) + 1
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
        # Coverage is split by WHAT the interval was about AND by the
        # NOMINAL level the prediction claimed: an outcome interval and a
        # mean interval make different claims, and a 0.5-nominal interval
        # and a 0.9-nominal interval are different promises — their hit
        # rates are never averaged into one number. The interval's own block
        # carries the declared kind and coverage (a bare range groups under
        # ``(unstated)``, never relabelled).
        ik = (evaluation.interval or {}).get("interval_kind")
        if interval.get("eligibility") == "evaluable":
            bucket = group.setdefault("interval_by_kind", {}).setdefault(
                str(ik or "(unstated)"), {"covered": [], "widths": [],
                                          "by_coverage": {}})
            bucket["covered"].append(
                1.0 if interval.get("covered") else 0.0)
            if interval.get("width") is not None:
                bucket["widths"].append(float(interval["width"]))
            nominal = interval.get("interval_coverage")
            level = (f"{float(nominal):.2f}" if nominal is not None
                     else "(unstated)")
            sub = bucket["by_coverage"].setdefault(
                level, {"covered": [], "widths": []})
            sub["covered"].append(
                1.0 if interval.get("covered") else 0.0)
            if interval.get("width") is not None:
                sub["widths"].append(float(interval["width"]))
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
            # Coverage split by interval KIND: an outcome interval and a
            # mean interval make different claims, so their hit rates are
            # reported separately and never pooled. Within a kind, coverage
            # is FURTHER split by the NOMINAL level claimed, because a 0.5
            # interval covering 50% and a 0.9 interval covering 50% are
            # different failures.
            "interval_by_kind": {
                kind: {
                    "n": len(bucket["covered"]),
                    "coverage": _mean(bucket["covered"]),
                    "mean_width": _mean(bucket["widths"]),
                    "by_nominal_coverage": {
                        level: {
                            "n": len(sub["covered"]),
                            "coverage": _mean(sub["covered"]),
                            "mean_width": _mean(sub["widths"]),
                        }
                        for level, sub in sorted(
                            (bucket.get("by_coverage") or {}).items())},
                }
                for kind, bucket in sorted(
                    (group.get("interval_by_kind") or {}).items())},
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
            # COVERAGE per observable event: how many of the group's
            # predictions were ASKED about it, gave it a probability, and
            # were answered with a reliable label. `n_asked` is the
            # denominator (every prediction is mandated to consider
            # implementation_failure and task_check_failed); `n_unknown`
            # is the honest-unknown count, which is NOT a zero and NOT a
            # failure. Reported separately from the Brier samples so a
            # model that omits an event is VISIBLE instead of silently
            # scoring as if it predicted zero.
            "risk_coverage": {
                event: {
                    "n_asked": group["asked_by_event"].get(event, 0),
                    "n_probabilized": group[
                        "probabilized_by_event"].get(event, 0),
                    "n_labelled": group["labelled_by_event"].get(event, 0),
                    "n_unknown": max(
                        0, n - group["probabilized_by_event"].get(event, 0)),
                    "coverage_rate": (
                        round(group["probabilized_by_event"].get(event, 0)
                              / group["asked_by_event"][event], 6)
                        if group["asked_by_event"].get(event) else None),
                }
                for event in sorted(OBSERVABLE_RISK_EVENTS)},
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
        "observation_rule_version": OBSERVATION_RULE_VERSION,
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
        # DETERMINISTIC rule reminders: the measured statistics turned into
        # short "what to watch next time" notes. No model call, no fitted
        # calibrator — a pure function of the groups above, so it is
        # recomputable and never a second source of truth.
        "reminders": _prediction_reminders(out_groups, min_samples),
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


def publish_paired_feedback(harness, block: Dict[str, Any]) -> None:
    """Store the DERIVED paired-feedback block (written at publish time)."""
    with harness.store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (PAIRED_FEEDBACK_KEY, harness.store.dumps(block)))


def paired_feedback_for_context(harness, *,
                                model_identity: Optional[str] = None
                                ) -> Dict[str, Any]:
    """The paired-feedback block a NEW prediction context reads.

    A SINGLE-ROW read of the last published block — no window scan, no
    history rebuild. This mirrors :func:`calibration_summary_for_context`:
    the DERIVATION is O(window) and happens at publish time, so a prediction
    read is O(1) however long the history is. When nothing has been
    published the block is EMPTY with a note recording the gap, rather than
    scanning the window to fill it (which would make a read path write).

    ``model_identity`` filters the pairs to the ATTACHED provider's model,
    exactly as the calibration summary is filtered: another model's past
    predictions are not evidence about this one. ``None`` means no filter
    was applied (and the block says so). The withheld count is reported, so
    a withheld pair is distinguishable from an absent one.
    """
    row = harness.store.conn.execute(
        "SELECT value FROM meta WHERE key=?",
        (PAIRED_FEEDBACK_KEY,)).fetchone()
    if row is None:
        return {
            "feedback_version": PAIRED_FEEDBACK_VERSION,
            "n_pairs_total": 0,
            "n_pairs_included": 0,
            "n_pairs_omitted": 0,
            "pairs": [],
            "missing": ("no paired feedback has been published yet: no "
                        "closed episode has been evaluated in this store, or "
                        "the block predates this version. Run `orx "
                        "calibration --rebuild` to publish one from the "
                        "current window"),
            "note": ("no published paired feedback is available; the context "
                     "records the gap rather than scanning history to fill "
                     "it"),
        }
    try:
        block = harness.store.loads(row["value"])
    except Exception:
        return {"feedback_version": PAIRED_FEEDBACK_VERSION,
                "pairs": [],
                "note": "the stored paired-feedback block could not be read"}
    if model_identity is None:
        block = copy.deepcopy(block)
        block["filtered"] = False
        block["filter_note"] = (
            "no model identity was supplied, so no filtering was applied: "
            "these pairs may come from DIFFERENT models and must not be "
            "read as this predictor's own record")
        return block
    return _filter_pairs_by_model(block, model_identity)


def _filter_pairs_by_model(block: Dict[str, Any],
                           model_identity: str) -> Dict[str, Any]:
    """Keep only the pairs produced by the SAME model identity.

    ``model_identity`` is the LABEL form, exactly as
    :func:`_filter_calibration_by_model` expects. ``(unknown)`` pairs (a
    legacy prediction with no recorded identity) are kept only when the
    current identity is itself unknown.
    """
    filtered = copy.deepcopy(block)
    # r12: the model identity is now a BLOCK-level field (shared by every
    # pair), so filtering checks the block first, then each pair for a
    # legacy block that still carries it per row.
    block_ident = block.get("model_identity")
    kept: List[Dict[str, Any]] = []
    withheld = 0
    if block_ident is not None:
        if str(block_ident) == str(model_identity):
            kept = list(block.get("pairs") or [])
        else:
            withheld = len(block.get("pairs") or [])
    else:
        for pair in (block.get("pairs") or []):
            if str(pair.get("model_identity") or "(unknown)") == \
                    str(model_identity):
                kept.append(pair)
            else:
                withheld += 1
    filtered["pairs"] = kept
    filtered["filtered"] = True
    filtered["model_identity"] = str(model_identity)
    filtered["n_pairs_included"] = len(kept)
    if withheld:
        filtered["n_pairs_withheld"] = withheld
        filtered["filter_note"] = (
            f"{withheld} pair(s) from a different model identity were "
            f"withheld: they describe another model's past predictions and "
            f"are not evidence about {model_identity!r}")
    else:
        filtered["filter_note"] = (
            f"every kept pair belongs to model identity {model_identity!r}")
    return filtered


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
    # The paired feedback is DERIVED here (the single publish point) and
    # stored, so a prediction read serves it in O(1) instead of re-deriving
    # it from the window on every context build.
    pairs = build_paired_feedback(harness, policy=policy)
    pairs["published_at"] = summary["published_at"]
    publish_paired_feedback(harness, pairs)
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
    # A late TASK-CHECK verdict no longer rewrites the quality observation
    # (the solver's own figure stands). It DOES move the separate
    # ``task_check`` fact carried alongside it, so that movement is what is
    # reported here — the sample keeps counting and the cost is preserved.
    stored_check = (stored_benefit.get("task_check") or {}).get("state")
    derived_check = (derived_benefit.get("task_check") or {}).get("state")
    if stored_benefit.get("observed") != derived_benefit.get("observed") \
            or stored_check != derived_check:
        changed["benefit"] = {
            "stored_observed": stored_benefit.get("observed"),
            "derived_observed": derived_benefit.get("observed"),
            "task_check": derived_benefit.get("task_check"),
            "cost_preserved": True,
        }
    if (stored.risk or {}).get("scored") != (derived.risk or {}).get("scored"):
        changed["risk"] = {"note": "risk labels moved with the live facts"}
    # A LATE COST MEASUREMENT. The host reports ``llm_tokens`` / ``tool_calls``
    # AFTER the run and they are backfilled by ``amend-cost`` — which can land
    # after the evaluation was frozen at close-out. The stored evaluation then
    # keeps ``excluded: "not measured on the real scope"`` for that dimension
    # while the LIVE facts now measure it, producing the self-contradiction
    # where the calibration reports a dimension measured but the stored
    # evaluation carries no per-dim entry for it. Re-derivation reads the
    # current facts, so the dimension appears; the change is reported and the
    # stored evaluation stays as history — nothing is rewritten.
    stored_cost = stored.cost or {}
    derived_cost = derived.cost or {}
    stored_dims = set((stored_cost.get("per_dim") or {}))
    derived_dims = set((derived_cost.get("per_dim") or {}))
    cost_moved = stored_dims != derived_dims
    if not cost_moved:
        for dim in stored_dims & derived_dims:
            s = (stored_cost.get("per_dim") or {}).get(dim) or {}
            d = (derived_cost.get("per_dim") or {}).get(dim) or {}
            if s.get("log_error") != d.get("log_error") \
                    or s.get("actual") != d.get("actual"):
                cost_moved = True
                break
    if cost_moved:
        changed["cost"] = {
            "stored_dims": sorted(stored_dims),
            "derived_dims": sorted(derived_dims),
            "newly_measured": sorted(derived_dims - stored_dims),
            "dropped": sorted(stored_dims - derived_dims),
            "eligibility": derived_cost.get("eligibility"),
            "note": ("a cost dimension's measurement moved after the "
                     "evaluation was frozen (typically a late host-usage "
                     "amend of llm_tokens / tool_calls): the live "
                     "re-derivation reflects it; the stored evaluation is "
                     "kept as history and nothing is rewritten"),
        }
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
               ("; a cost dimension's measurement moved (a late host-usage "
                "amend), so more of the real spend is now scored"
                if "cost" in changed else
                ". A task-result verdict changed after the evaluation was "
                "written"))),
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
            "observation_rule_version": OBSERVATION_RULE_VERSION,
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


# ---------------------------------------------------------------------------
# 5b. paired prediction-execution feedback (compact, per-episode)
# ---------------------------------------------------------------------------

#: Default character budget for the compact paired-feedback block a
#: prediction context carries. Overridden by
#: ``OR_HARNESS_PAIRED_FEEDBACK_CHARS``.
#:
#: r12 raises the default well above the old 8000: the deployed model has a
#: 256k context, and 8000 characters (a couple of pairs) starved the model of
#: its OWN past accuracy. This is a CEILING, not a target — the block is only
#: as large as the material it has, and it is reported with used/n-omitted.
#: A smaller deployment overrides it downward.
DEFAULT_PAIRED_FEEDBACK_CHARS = 64000


def _paired_feedback_budget() -> int:
    return _env_int("OR_HARNESS_PAIRED_FEEDBACK_CHARS",
                    DEFAULT_PAIRED_FEEDBACK_CHARS)


def _compact_problem_conditions(evaluation, prediction) -> Dict[str, Any]:
    """The KEY conditions of the problem a past pair was made under.

    Derived from what the closed episode already recorded, NEVER from a live
    profile read (which would let today's bank leak into a past pair). Kept
    deliberately small: the task id, the family and the structural cell the
    prediction was made for — a recognisable handle, not the whole joint
    representation. The family and cell come from the FROZEN prediction
    context/trace, so they describe the conditions of THAT decision.
    """
    conditions: Dict[str, Any] = {}
    if getattr(evaluation, "task_id", None):
        conditions["task_id"] = str(evaluation.task_id)
    candidate = getattr(prediction, "candidate", None)
    if candidate is not None and getattr(candidate, "strategy_id", None):
        conditions["strategy_id"] = str(candidate.strategy_id)
    info = (getattr(getattr(prediction, "trace", None), "model_info", None)
            or {})
    if isinstance(info, dict):
        if info.get("cell_token"):
            conditions["cell"] = str(info["cell_token"])
        if info.get("family"):
            conditions["family"] = str(info["family"])
    return conditions


def build_paired_feedback(
        harness, *,
        budget_chars: Optional[int] = None,
        policy: Optional[CalibrationPolicy] = None,
        ) -> Dict[str, Any]:
    """DERIVE the compact per-episode prediction-execution pairs.

    A DERIVED VIEW over the window's stored evaluations — no model call, no
    re-scoring, no new storage. Each row is the paired facts of ONE past
    decision: the conditions it was made under, the method planned, the
    ORIGINAL prediction (read-only), the REAL observation, the per-field
    difference, the task-check outcome, the scope, the missing or
    not-comparable reasons and a source reference.

    This is the EXPENSIVE, O(window) derivation. It is called at PUBLISH
    time (see :func:`republish_calibration`), never from the prediction read
    path: the read path serves the stored block so a prediction read stays
    O(1) however long the history is (see
    :func:`paired_feedback_for_context`).

    What it deliberately does NOT do:

    - it is NOT a top-k cut: every closed episode's pairs are candidates,
      and the included/omitted counts are REPORTED rather than silently
      dropped;
    - an UNEXECUTED candidate never appears (there is no real outcome to
      pair it with, and none is fabricated);
    - it does not filter by cell, strategy_id or verification state — a
      failure and a cross-cell case are exactly the material a prediction
      needs;
    - a single unknown or not-comparable FIELD does not drop the row: the
      other fields are still carried.

    r12 ORDERING and SHARING (so a small budget shows the most useful
    material, and the block does not repeat itself):

    - rows are ordered MOST-RELEVANT-FIRST: failures and repairs, then the
      most RECENT history, then the rest — never a raw oldest-first scan
      that lets one early long record hold the budget forever;
    - the model identity and the observation-rule version are stated ONCE
      at the block level instead of on every row (they are the same for the
      whole block), and each row carries only its ``evaluation_id``;
    - the risk label-basis sentences are SHORT CODES explained once in
      ``risk_reason_legend``.

    It returns a block with an empty ``pairs`` list and a note when there is
    nothing to show (no closed episodes).
    """
    budget = budget_chars if budget_chars is not None else \
        _paired_feedback_budget()
    policy = policy or CalibrationPolicy.from_env()
    window = calibration_window(harness, policy, readonly=True)
    evaluations = _evaluations_for_window(harness, window)
    rows: List[Dict[str, Any]] = []
    n_total = 0
    for stored_evaluation in evaluations:
        evaluation, _ = _live_evaluation(harness, stored_evaluation)
        if evaluation.state == "pending":
            # A running scope has no real outcome: not a pair yet.
            continue
        prediction = harness.strategy_predictions.get(
            stored_evaluation.prediction_id)
        row = _paired_feedback_row(evaluation, prediction)
        if row is None:
            continue
        n_total += 1
        rows.append((stored_evaluation, evaluation, row))

    # RELEVANCE-FIRST ordering. The plan's priority: cases relevant to the
    # current candidate, recent failures and repairs, then successes and
    # different-condition cases. Since the block is global (not per
    # candidate), the order is: failures/repairs first, then most RECENT.
    # This stops an early long record from holding the budget forever.
    def _is_failure(row: Dict[str, Any]) -> bool:
        return bool(row.get("failure_classes")) \
            or row.get("execution_status") in ("error", "timeout") \
            or row.get("risk_actual") is not None \
            and any((e.get("label") == "occurred")
                    for e in row.get("risk_actual") or [])

    def _recency(stored_evaluation) -> float:
        return float(getattr(stored_evaluation, "created_at", 0.0) or 0.0)

    indexed = list(enumerate(rows))

    def _sort_key(item):
        index, (stored_evaluation, _evaluation, row) = item
        return (0 if _is_failure(row) else 1,
                -_recency(stored_evaluation), index)

    indexed.sort(key=_sort_key)

    included: List[Dict[str, Any]] = []
    used = 0
    omitted = 0
    ident = None
    obs_rule = None
    for _index, (_stored, _evaluation, row) in indexed:
        # Hoist the block-invariant identity OFF the row: it is the same for
        # every pair, so repeating it per row is pure duplication.
        row_ident = row.pop("model_identity", None)
        row_rule = row.pop("observation_rule_version", None)
        if ident is None:
            ident = row_ident
        if obs_rule is None:
            obs_rule = row_rule
        size = len(json.dumps(row, ensure_ascii=False, default=str))
        if used + size > budget:
            omitted += 1
            # The identity was popped for a row we cannot include: put it
            # back is unnecessary (the row is dropped) — but its absence
            # must not silently change the block identity.
            continue
        used += size
        included.append(row)
    out: Dict[str, Any] = {
        "feedback_version": PAIRED_FEEDBACK_VERSION,
        "model_identity": ident,
        "observation_rule_version": obs_rule,
        "identity_note": ("these two fields are BLOCK-level: all pairs share "
                          "one predicting model and one observation rule, so "
                          "they are stated once rather than per row"),
        "n_pairs_total": n_total,
        "n_pairs_included": len(included),
        "n_pairs_omitted": omitted,
        "budget_chars": budget,
        "used_chars": used,
        "ordering": ("failures/repairs first, then most recent: the budget "
                     "shows the most useful material rather than an "
                     "oldest-first scan"),
        "risk_reason_legend": dict(RISK_REASON_LEGEND),
        "basis": ("derived from CLOSED episodes' stored evaluations in the "
                  "current window; no model call, no re-scoring, no new "
                  "storage"),
        "pairs": included,
    }
    if omitted:
        out["omission_note"] = (
            f"{omitted} pair(s) were omitted to stay within the character "
            "budget after {len(included)} were included (failures/recent "
            "first); they remain available through `orx calibration` and the "
            "evaluation store — the omission is reported, never silent")
    if not rows:
        out["note"] = ("no closed-episode prediction-execution pairs exist "
                       "in the window yet: a cold start carries no paired "
                       "feedback (this is absent, not 'nothing matched')")
    return out


#: Version of the paired-feedback block schema.
PAIRED_FEEDBACK_VERSION = "wm-pairs/1"

#: Version of the deterministic rule-reminder block (Q2-ii): the measured
#: statistics rendered as short "what to watch" notes. A pure function of
#: the groups, bumped when the rendering rules change.
REMINDER_VERSION = "wm-remind/1"


def _prediction_reminders(out_groups: Dict[str, Any],
                          min_samples: int) -> List[Dict[str, Any]]:
    """Deterministic "next-time" reminders derived from the group statistics.

    NOT a model-written lesson and NOT a fitted calibrator: a pure,
    recomputable function of the measured groups. Each reminder states the
    applicability it was derived under (the group's metric/unit/scope/
    observation-rule identity), the OBSERVED bias (with its direction and
    sample size), a concrete "watch this next time" instruction, and the
    group key as its support reference. A group below the episode threshold
    yields NO reminder (``insufficient_evidence`` is not a lesson), so a
    reminder is never a claim made from one observation.

    The DISTINCTION the plan requires is enforced here: the *observed bias*
    is a fact (the mean signed error), the *instruction* is a derived
    reminder — it carries ``kind="derived_reminder"`` and ``basis=
    "measured_statistics"`` so a reader never mistakes it for an observation
    or for the framework's own recommendation of a method.
    """
    reminders: List[Dict[str, Any]] = []
    rec_id = 0
    for key, group in sorted(out_groups.items()):
        parts = str(key).split("|")
        metric = parts[2] if len(parts) > 2 else "(none)"
        unit = parts[3] if len(parts) > 3 else "(none)"
        scope = parts[4] if len(parts) > 4 else "(none)"
        obs_rule = parts[5] if len(parts) > 5 else "(unknown)"
        applicability = {
            "metric": metric, "unit": unit, "scope": scope,
            "observation_rule_version": obs_rule,
            "model_identity": parts[1] if len(parts) > 1 else "(unknown)",
        }
        distinct = group.get("n_distinct_episodes") or 0
        if distinct < min_samples:
            continue
        # BENEFIT bias: directed signed error.
        signed = group.get("mean_benefit_signed_error")
        benefit_evidence = group.get("benefit_evidence") or {}
        if signed is not None \
                and benefit_evidence.get("evidence") == "measured":
            n = benefit_evidence.get("n_distinct_episodes")
            direction = ("UNDER-predicted" if signed > 0
                         else "OVER-predicted" if signed < 0 else "unbiased")
            if abs(float(signed)) > 1e-9:
                rec_id += 1
                reminders.append({
                    "reminder_id": f"rem_{rec_id:03d}",
                    "kind": "derived_reminder",
                    "basis": "measured_statistics",
                    "field": "benefit",
                    "applicability": copy.deepcopy(applicability),
                    "observed_bias": {
                        "mean_signed_error": signed,
                        "direction": direction,
                        "n_distinct_episodes": n,
                        "note": ("observed (fact): the mean directed error "
                                 "of past predictions under this "
                                 "applicability; positive = the real "
                                 "outcome was historically BETTER"),
                    },
                    "watch_next_time": (
                        f"under {metric}/{unit} at scope {scope}, past "
                        f"predictions were {direction} by "
                        f"{abs(float(signed)):.4f} on average over {n} "
                        "distinct episode(s): adjust the value in that "
                        "direction, or state "
                        "unsupported_fields['benefit'] if the evidence "
                        "does not justify a number"),
                    "support": [key],
                })
        # COST bias per dimension: directed log-ratio.
        cost_ratio = group.get("mean_cost_log_ratio") or {}
        cost_evidence = group.get("cost_evidence") or {}
        for dim, ratio in sorted(cost_ratio.items()):
            verdict = (cost_evidence.get(dim) or {})
            if ratio is None or verdict.get("evidence") != "measured":
                continue
            if abs(float(ratio)) <= 1e-9:
                continue
            n = verdict.get("n_distinct_episodes")
            rec_id += 1
            reminders.append({
                "reminder_id": f"rem_{rec_id:03d}",
                "kind": "derived_reminder",
                "basis": "measured_statistics",
                "field": f"cost.{dim}",
                "applicability": copy.deepcopy(applicability),
                "observed_bias": {
                    "mean_log_ratio": ratio,
                    "direction": ("UNDER-predicted" if ratio > 0
                                  else "OVER-predicted"),
                    "n_distinct_episodes": n,
                    "note": ("observed (fact): mean log(actual/predicted) "
                             "for this dimension; positive = real cost was "
                             "historically HIGHER"),
                },
                "watch_next_time": (
                    f"for cost dimension {dim}, past predictions were "
                    f"{'UNDER' if ratio > 0 else 'OVER'}-predicted by "
                    f"mean log-ratio {float(ratio):.4f} over {n} distinct "
                    "episode(s): bias the estimate accordingly"),
                "support": [key],
            })
        # INTERVAL coverage: a stated coverage that the data does not support
        # is worth a reminder; a covered interval that is very wide is a
        # different one.
        for kind, bucket in sorted(
                (group.get("interval_by_kind") or {}).items()):
            if not bucket or not bucket.get("n"):
                continue
            coverage = bucket.get("coverage")
            width = bucket.get("mean_width")
            if coverage is None:
                continue
            rec_id += 1
            reminders.append({
                "reminder_id": f"rem_{rec_id:03d}",
                "kind": "derived_reminder",
                "basis": "measured_statistics",
                "field": "interval",
                "applicability": {**copy.deepcopy(applicability),
                                  "interval_kind": kind},
                "observed_bias": {
                    "empirical_coverage": coverage,
                    "n": bucket.get("n"),
                    "mean_width": width,
                    "note": ("observed (fact): the empirical hit rate of "
                             "past intervals of this kind; compare it with "
                             "the nominal coverage the prediction claimed"),
                },
                "watch_next_time": (
                    f"past {kind} intervals covered the observed value "
                    f"{float(coverage):.3f} of the time (n="
                    f"{bucket.get('n')}, mean width {width}): check your "
                    "stated interval_coverage against this, and split "
                    "outcome vs mean intervals rather than mixing them"),
                "support": [key],
            })
    reminders.extend(_single_case_reminders(out_groups, min_samples))
    return reminders


#: A single-observation benefit error this large is worth a scoped "watch
#: this" note even without the multi-episode threshold: it is NOT a
#: statistical lesson, only a flagged case.
SINGLE_CASE_ERROR_THRESHOLD = 0.5


def _single_case_reminders(out_groups: Dict[str, Any],
                           min_samples: int) -> List[Dict[str, Any]]:
    """SCOPED reminders from a group that has too FEW episodes to generalise.

    A group below the multi-episode threshold yields no statistical reminder
    (one observation is not a lesson). But a LARGE single-observation error
    is still worth flagging as a case: the reminder is stamped
    ``kind="single_case_reminder"`` / ``basis="single_observation"`` with its
    applicability and the ONE episode id, so a reader sees a flagged case and
    never mistakes it for a measured bias. This is the range-limited
    counterpart the multi-episode reminders deliberately do not provide.
    """
    out: List[Dict[str, Any]] = []
    rec_id = 0
    for key, group in sorted(out_groups.items()):
        distinct = group.get("n_distinct_episodes") or 0
        if distinct == 0 or distinct >= min_samples:
            continue  # covered by the statistical reminders above
        evidence = group.get("benefit_evidence") or {}
        if evidence.get("evidence") != "measured":
            continue
        signed = group.get("mean_benefit_signed_error")
        if signed is None or abs(float(signed)) < SINGLE_CASE_ERROR_THRESHOLD:
            continue
        parts = str(key).split("|")
        rec_id += 1
        out.append({
            "reminder_id": f"rem_single_{rec_id:03d}",
            "kind": "single_case_reminder",
            "basis": "single_observation",
            "field": "benefit",
            "applicability": {
                "metric": parts[2] if len(parts) > 2 else "(none)",
                "unit": parts[3] if len(parts) > 3 else "(none)",
                "scope": parts[4] if len(parts) > 4 else "(none)",
                "model_identity": parts[1] if len(parts) > 1 else "(unknown)",
            },
            "observed": {
                "mean_signed_error": signed,
                "n_distinct_episodes": distinct,
                "note": ("observed (fact): a single episode's directed error; "
                         "NOT a measured bias — too few episodes to "
                         "generalise"),
            },
            "watch_next_time": (
                f"a single past prediction under this applicability missed by "
                f"{abs(float(signed)):.3f} (positive = the real outcome was "
                "better). Treat it as a flagged CASE, not a correction: it "
                "shows where your estimate can be far off, not how far it "
                "usually is"),
            "support": [key],
        })
    return out


def _compact_method_steps(steps: Sequence[Any], *,
                          max_steps: int = 12,
                          per_step_chars: int = 220) -> List[str]:
    """Compact a method's steps WITHOUT dropping the key constraints.

    The old rule kept the first THREE steps and clipped them, which silently
    deleted later steps that carry the binding constraints (an upper bound, a
    boundary condition, a final solve step) — the material then looked
    complete while the decisive relation was gone. r12 keeps EVERY step up to
    ``max_steps`` and only clips the individual text, so a normal six-step
    method travels whole. When a method really has more steps than the cap,
    the cap is spent on the FIRST and LAST steps (the setup and the final
    solve/bound usually live at the ends) and the middle is elided with an
    explicit marker rather than silently truncated.
    """
    cleaned = [(" ".join(str(s).split()))[:per_step_chars]
               for s in steps if str(s).strip()]
    if len(cleaned) <= max_steps:
        return cleaned
    head = max_steps // 2
    tail = max_steps - head - 1
    return (cleaned[:head]
            + [f"...[{len(cleaned) - head - tail} step(s) elided]..."]
            + (cleaned[-tail:] if tail else []))


def _risk_label_basis_code(basis: Any) -> Optional[str]:
    """Map a long risk label-basis sentence to a SHORT reason code.

    The framework's label reasons are drawn from a small fixed set of
    sentences (the vocabulary's ``measured``/``note`` text and the two
    "cannot compute a Brier score" cases). Repeating the whole sentence per
    event and per pair is pure duplication; the code carries the SAME
    meaning and is explained once in ``risk_reason_legend``. An
    unrecognised reason is passed through (clipped) rather than dropped, so
    no information is invented or lost.
    """
    if basis is None:
        return None
    text = str(basis)
    lowered = text.lower()
    if "no brier score can be computed" in lowered \
            or ("did not predict this event" in lowered):
        return "unpredicted_no_brier"
    if "gave no probability" in lowered:
        return "predicted_without_probability"
    if "unknown" in lowered and "class" in lowered:
        return "error_class_unknown"
    if "no task check" in lowered:
        return "no_task_check"
    if "time limit" in lowered:
        return "time_limit"
    if "did not report infeasibility" in lowered:
        return "no_infeasibility"
    if "infeasibility" in lowered:
        return "reported_infeasibility"
    if "did not record this failure class" in lowered:
        return "class_not_recorded"
    if "policy" in lowered or "module" in lowered:
        return "environment_failure"
    if "own code" in lowered or "implementation" in lowered:
        return "implementation_failure"
    if "check" in lowered and ("did not hold" in lowered
                               or "failed" in lowered):
        return "task_check_failed"
    # Unrecognised: carry a clipped form rather than drop the fact.
    return (" ".join(text.split()))[:80]


#: One-time explanation of the short ``risk_actual[*].label_basis`` codes,
#: so the block does not repeat the same rule text per event. Declared ONCE
#: per paired-feedback block (see :func:`build_paired_feedback`).
RISK_REASON_LEGEND: Dict[str, str] = {
    "unpredicted_no_brier": (
        "the model did not predict this event: it is observed by the "
        "framework and counted in the occurrence rate, but no Brier score "
        "can be computed without a probability"),
    "predicted_without_probability": (
        "the prediction named the event but gave no probability: it is "
        "observed, not scored"),
    "error_class_unknown": (
        "the failure's error_class is unknown/unrecorded: the cause is not "
        "inferred from the error text"),
    "no_task_check": (
        "no task check is on record for this execution: task validity is "
        "UNKNOWN, never a failure"),
    "time_limit": "the executor's time limit fired",
    "no_infeasibility": "the solver did not report infeasibility here",
    "reported_infeasibility": (
        "the solver reported infeasibility (a verdict, not by itself a "
        "strategy failure)"),
    "class_not_recorded": "the execution did not record this failure class",
    "environment_failure": (
        "an ENVIRONMENT failure (sandbox policy / missing module / "
        "unavailable backend)"),
    "implementation_failure": (
        "the harness's OWN code failed (an implementation failure, NOT a "
        "modelling error)"),
    "task_check_failed": (
        "a declared task check ran on real values and did not hold"),
}


def _paired_feedback_row(evaluation, prediction) -> Optional[Dict[str, Any]]:
    """ONE compact prediction-execution pair, or None when not a pair.

    ``None`` for a prediction that was never bound to a real execution: an
    unexecuted candidate has no real outcome, and none is fabricated.
    """
    if prediction is None:
        return None
    info = prediction.trace.model_info or {}
    if not info.get("bound_action_id"):
        return None
    benefit = evaluation.benefit or {}
    cost = evaluation.cost or {}
    row: Dict[str, Any] = {
        "evaluation_id": evaluation.evaluation_id,
        "prediction_id": evaluation.prediction_id,
        "task_id": evaluation.task_id,
        "episode_id": evaluation.episode_id,
        "scope": evaluation.scope,
        # The predicting MODEL's identity travels with the pair: another
        # model's past predictions are not evidence about THIS one, so the
        # read path filters on it exactly as the calibration summary does.
        "model_identity": getattr(evaluation, "model_identity", "(unknown)"),
        "observation_rule_version": getattr(
            evaluation, "observation_rule_version", None),
        "conditions": _compact_problem_conditions(evaluation, prediction),
        "task_check": copy.deepcopy(benefit.get("task_check")),
    }
    # The method that was planned, one line: its name, the number of steps
    # and the steps THEMSELVES. r12 keeps EVERY step (up to a cap that a
    # normal method never reaches) and clips only the per-step text, so a
    # method that carries a binding constraint in a later step travels
    # whole. The old "first three steps" rule silently deleted exactly the
    # constraints (an upper bound, a boundary, the final solve) a later
    # prediction needed. Never a paraphrase.
    method = getattr(getattr(prediction, "candidate", None), "method", None)
    if isinstance(method, dict) and method:
        steps = [str(s) for s in (method.get("steps") or [])]
        row["method_planned"] = {
            "name": method.get("name"),
            "n_steps": len(steps),
            "steps": _compact_method_steps(steps),
        }
        # The plan's own justification and declared fallback are KEPT: they
        # are the "why" the reader needs to judge applicability, and the
        # fallback is what makes a later deviation a DECLARED one.
        if method.get("why"):
            row["method_planned"]["why"] = (" ".join(
                str(method["why"]).split()))[:400]
        if method.get("fallback"):
            row["method_planned"]["fallback"] = (" ".join(
                str(method["fallback"]).split()))[:400]
    # The method ACTUALLY performed, when the run reported one (the script's
    # own receipt, or a harness declaration). It is a SEPARATE fact from the
    # plan: a plan is never copied in as if it had been carried out, and an
    # absent actual method stays absent (unknown), never equal to the plan.
    # When the actual steps are IDENTICAL to the plan's they are not repeated
    # (only the pointer is written) — the "same method content only once"
    # rule, which also makes a real change easier to spot.
    actual_method = getattr(evaluation, "method_actual", None)
    if isinstance(actual_method, dict) and actual_method:
        steps = [str(s) for s in (actual_method.get("steps") or [])]
        planned_steps = (row.get("method_planned") or {}).get("steps")
        compact = _compact_method_steps(steps)
        row["method_actual"] = {
            "name": actual_method.get("name"),
            "n_steps": len(steps),
        }
        if planned_steps is not None and compact == planned_steps:
            row["method_actual"]["steps"] = "same as method_planned.steps"
        else:
            row["method_actual"]["steps"] = compact
    # Whether the method that ran DEVIATED from the plan (a verdict, not the
    # plan copied over the performance). ``None`` when it could not be
    # compared (no plan, or no observed method).
    deviation = getattr(evaluation, "method_deviation", None)
    if isinstance(deviation, dict) and deviation:
        row["method_deviation"] = copy.deepcopy(deviation)
    # A failed pair carries its short failure TYPE and the solver's own
    # status, kept SEPARATE from the task-check verdict.
    if getattr(evaluation, "failure_classes", None):
        row["failure_classes"] = list(evaluation.failure_classes)
    if getattr(evaluation, "execution_status", None):
        row["execution_status"] = evaluation.execution_status
    # The predicted RISK events (their NAMES), so a pair shows what the model
    # was wary of beside what happened — a fact, never a recommendation.
    risk = getattr(prediction, "risk", None)
    if risk is not None and getattr(risk, "events", None):
        row["risk_predicted"] = [
            {"event": e.event, "probability": e.probability}
            for e in risk.events][:5]
    # The ORIGINAL prediction (read-only) and the real observation, with the
    # per-field difference. A field the prediction did not carry is simply
    # absent — never a zero.
    benefit_row: Dict[str, Any] = {}
    if benefit.get("predicted") is not None:
        benefit_row["predicted"] = benefit.get("predicted")
        benefit_row["metric"] = benefit.get("metric")
        # The interval's OWN semantics travel with the interval block, which
        # carries the declared kind/coverage; fall back to the benefit copy
        # for a legacy evaluation that has neither.
        interval_block = evaluation.interval or {}
        benefit_row["interval_kind"] = interval_block.get(
            "interval_kind", benefit.get("interval_kind"))
        benefit_row["interval_coverage"] = interval_block.get(
            "interval_coverage", benefit.get("interval_coverage"))
    if benefit.get("observed") is not None:
        benefit_row["observed"] = benefit.get("observed")
        benefit_row["signed_error"] = benefit.get("signed_error")
    if benefit.get("eligibility"):
        benefit_row["eligibility"] = benefit.get("eligibility")
        if benefit.get("reason"):
            benefit_row["reason"] = benefit.get("reason")
    if benefit_row:
        row["benefit"] = benefit_row
    per_dim = cost.get("per_dim") or {}
    if per_dim:
        row["cost_log_ratio"] = {
            dim: entry.get("log_ratio")
            for dim, entry in per_dim.items()
            if entry.get("log_ratio") is not None}
        # The REAL measured spend, per dimension — the absolute value, not
        # only the error RATIO. A ratio cannot substitute for the measurement
        # (it hides the scale), so both travel. ``predicted`` is the frozen
        # claim, ``actual`` the real measurement, and a dimension the
        # prediction did not carry is simply absent.
        row["cost_measured"] = {}
        for dim, entry in per_dim.items():
            if entry.get("actual") is None:
                continue
            row["cost_measured"][dim] = {
                "predicted": entry.get("predicted"),
                "actual": entry.get("actual"),
                "abs_error": entry.get("abs_error"),
            }
        if not row["cost_measured"]:
            row.pop("cost_measured")
    if cost.get("excluded"):
        row["cost_excluded"] = dict(cost.get("excluded"))
    # The real cost dimensions this evaluation observed but the prediction
    # did NOT predict (measured facts, not comparable to a claim): carried so
    # a reader sees the operation's true spend even where a ratio exists.
    if cost.get("observed_not_predicted"):
        row["cost_observed_unpredicted"] = copy.deepcopy(
            cost.get("observed_not_predicted"))
    # The REAL risk events: which happened, which did not, which are unknown.
    # A predicted probability is never read as an occurrence, and an
    # unknown label stays unknown (never defaulted to "did not happen").
    #
    # r12 COMPRESSION: the long explanatory prose (why an un-predicted event
    # has no Brier score, what a label means) is NOT re-inlined per event.
    # It is mapped to a SHORT reason code, explained ONCE in the block's
    # ``risk_reason_legend``. The old form repeated the same Brier sentence
    # in every pair and every event — hundreds of characters of duplicated
    # rule text that pushed real evidence out of the budget.
    risk_actual: List[Dict[str, Any]] = []
    for entry in (evaluation.risk or {}).get("scored") or []:
        risk_actual.append({
            "event": entry.get("event"),
            "label": entry.get("label"),
            "label_basis": _risk_label_basis_code(entry.get("label_basis")),
            "predicted_probability": entry.get("predicted_probability"),
            "brier": entry.get("brier"),
        })
    for entry in (evaluation.risk or {}).get("unscored") or []:
        risk_actual.append({
            "event": entry.get("event"),
            "label": entry.get("label"),
            "label_basis": _risk_label_basis_code(
                entry.get("reason") or entry.get("label_basis")),
            "predicted_probability": entry.get("predicted_probability"),
        })
    if risk_actual:
        row["risk_actual"] = risk_actual[:8]
    # The attribution's blocked dimensions: WHY a field is not comparable.
    # A row with a blocked field still carries every other field.
    blocked = evaluation.attribution or {}
    if blocked:
        row["blocked_dimensions"] = {
            dim: sorted({str(e.get("field")) for e in entries})
            for dim, entries in blocked.items()}
    if not row.get("benefit") and not row.get("cost_log_ratio") \
            and not row.get("task_check"):
        # Nothing at all to learn from: it is still a real pair (a failed
        # call, an unobserved outcome), so it is carried with its
        # exclusion reasons rather than dropped.
        row["note"] = ("this pair carries no comparable field; the "
                       "exclusion reasons above/below are the fact")
        if evaluation.exclusion_reasons:
            row["exclusion_reasons"] = list(evaluation.exclusion_reasons)
    return row


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
