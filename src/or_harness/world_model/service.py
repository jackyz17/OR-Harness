"""The M2 shadow loop: predict → (real action happens) → bind → compare.

One minimal but complete closed path:

1. ``predict_outcome`` freezes the input snapshot (BEFORE the action),
   assembles the input view from the snapshot's own memory views (no
   separate retrieval system), calls the provider EXPLICITLY, validates
   the payload, and persists the frozen prediction with its call cost.
2. The existing agent/strategy flow decides and executes as usual — the
   prediction never changes any recommendation (shadow mode).
3. ``bind_outcome`` attaches the prediction to the real action that ran,
   checking type/strategy/solver match; a mismatch is recorded, not
   silently compared.
4. ``compare_prediction`` produces the per-field feedback (status
   category, quality, per-dimension cost) — appended to the record, never
   overwriting the frozen prediction. Re-running the same comparison is
   idempotent and never re-invokes the model.

Evidence boundaries enforced here:
- predictions are frozen at generation time; later bank or config changes
  never rewrite them;
- an unbound prediction stays unbound (the candidate never ran — no
  counterfactual truth is fabricated from other strategies' results);
- the model call's own cost is recorded on the prediction (and optionally
  charged to a parent action as own cost) — it is never confused with the
  PREDICTED cost of the target action;
- predicted successor states stay inside the prediction record
  (hypothetical), never entering the real state chain.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.storage import Store, StorageError
from or_harness.world_model.prediction import (
    PROMPT_TEMPLATE_VERSION,
    SUPPORTED_ACTION_TYPES,
    ActionSpec,
    OutcomePrediction,
    validate_prediction_payload,
)
from or_harness.world_model.provider import WorldModelProvider

#: The snapshot fields the input view is assembled from. The SAME memory
#: views the snapshot already carries — no second retrieval system, no
# double-counted support.
INPUT_VIEW_KEYS = ("problem_state", "coverage", "budget_state",
                   "task_progress")


class PredictionService:
    """Predict / bind / compare over the frozen prediction log."""

    def __init__(self, store: Store, provider: WorldModelProvider):
        self.store = store
        self.provider = provider

    # -- predict ---------------------------------------------------------------

    def predict_outcome(self, task: Dict[str, Any],
                        action_spec: ActionSpec,
                        snapshot,  # BeliefSnapshot (already frozen by caller)
                        *, timeout_s: Optional[float] = None,
                        ) -> OutcomePrediction:
        """Assemble the input view, call the provider, validate, persist.

        ``snapshot`` must be the FROZEN pre-action state (the caller takes
        it before this method runs). The prediction is persisted with
        whatever the provider returned — including failures, which keep
        their known call cost. ``timeout_s`` (optional) is the caller's
        remaining time budget for this call, forwarded to the provider."""
        if action_spec.action_type not in SUPPORTED_ACTION_TYPES:
            prediction = OutcomePrediction(
                prediction_id=OutcomePrediction.new_id(),
                input_snapshot_id=snapshot.snapshot_id,
                action_spec=action_spec,
                status="unsupported_action",
                error=f"action type {action_spec.action_type!r} is not "
                      f"supported by the M2 single-step prediction path "
                      f"(supported: {SUPPORTED_ACTION_TYPES})",
                model_info=self.provider.describe(),
            )
            self._save(prediction)
            return prediction

        # Input view: the snapshot's own fields, recorded by key so the
        # stored prediction shows exactly what the model was given.
        view: Dict[str, Any] = {}
        for key in INPUT_VIEW_KEYS:
            value = getattr(snapshot, key, None)
            if value:
                view[key] = copy.deepcopy(value)
        request = {
            "action_spec": action_spec.to_dict(),
            "state": view,
        }
        try:
            # Backwards compatibility: existing / custom providers that
            # only take `predict(request)` are accepted without error.
            try:
                result = self.provider.predict(request, timeout_s=timeout_s)
            except TypeError:
                result = self.provider.predict(request)
        except Exception as exc:  # provider adapter failure
            prediction = OutcomePrediction(
                prediction_id=OutcomePrediction.new_id(),
                input_snapshot_id=snapshot.snapshot_id,
                action_spec=action_spec,
                status="provider_error",
                error=f"{type(exc).__name__}: {exc}",
                model_info=self.provider.describe(),
            )
            self._save(prediction)
            return prediction

        call_cost = self._call_cost(result)
        if result.get("not_configured"):
            prediction = OutcomePrediction(
                prediction_id=OutcomePrediction.new_id(),
                input_snapshot_id=snapshot.snapshot_id,
                action_spec=action_spec,
                status="not_configured",
                error=str(result.get("error") or "provider not configured"),
                model_info=self.provider.describe(),
            )
            self._save(prediction)
            return prediction

        payload = result.get("payload")
        if payload is None:
            # Model call happened (cost is real) but produced no usable
            # payload — an explicit invalid_output, never a fake success.
            prediction = OutcomePrediction(
                prediction_id=OutcomePrediction.new_id(),
                input_snapshot_id=snapshot.snapshot_id,
                action_spec=action_spec,
                status="invalid_output",
                error=str(result.get("error") or "no payload"),
                model_info=self._model_info(result),
                call_cost=call_cost,
                input_view_keys=list(view.keys()),
            )
            self._save(prediction)
            return prediction

        problems = validate_prediction_payload(payload, action_spec)
        if problems:
            prediction = OutcomePrediction(
                prediction_id=OutcomePrediction.new_id(),
                input_snapshot_id=snapshot.snapshot_id,
                action_spec=action_spec,
                status="invalid_output",
                error="; ".join(problems),
                model_info=self._model_info(result),
                call_cost=call_cost,
                input_view_keys=list(view.keys()),
            )
            self._save(prediction)
            return prediction

        predicted = {k: copy.deepcopy(v) for k, v in payload.items()
                     if k in ("outcome_status", "feasible", "quality",
                              "failure_prob", "expected_error_kinds",
                              "state_changes",
                              # M4 maintenance-assessment fields (induce
                              # action semantics).
                              "candidate_formation_prob",
                              "expected_reuse_benefit",
                              "generalization_risk")}
        if isinstance(payload.get("cost"), dict):
            dims = {d: float(v) for d, v in payload["cost"].items()
                    if d in COST_DIMENSIONS and v is not None}
            cost_vector = CostVector(**dims, measured=set(dims))
            predicted["cost"] = cost_vector.to_dict()
            # Persist the measured mask alongside: an explicitly predicted
            # zero is a prediction, a missing dimension is NOT — the mask
            # is the only way downstream consumers can tell them apart.
            predicted["cost_measured"] = sorted(dims)
        confidence = payload.get("confidence")
        prediction = OutcomePrediction(
            prediction_id=OutcomePrediction.new_id(),
            input_snapshot_id=snapshot.snapshot_id,
            action_spec=action_spec,
            status="valid",
            predicted=predicted,
            unsupported_fields=dict(payload.get("unsupported_fields") or {}),
            confidence=(float(confidence)
                        if isinstance(confidence, (int, float)) else None),
            evidence_basis=[str(e) for e in
                            (payload.get("evidence_basis") or [])],
            model_info=self._model_info(result),
            call_cost=call_cost,
            input_view_keys=list(view.keys()),
        )
        self._save(prediction)
        return prediction

    def _model_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(self.provider.describe())
        info["prompt_template_version"] = PROMPT_TEMPLATE_VERSION
        latency = result.get("latency_s")
        if latency is not None:
            info["call_latency_s"] = round(float(latency), 4)
        return info

    @staticmethod
    def _call_cost(result: Dict[str, Any]) -> Optional[CostVector]:
        """The model call's OWN cost from provider usage (measured when the
        provider reports it; unknown otherwise — never zero-as-cheap)."""
        usage = result.get("usage")
        latency = result.get("latency_s")
        if not usage and latency is None:
            return None
        vector = CostVector(measured=set())
        if usage:
            tokens = usage.get("completion_tokens")
            if isinstance(tokens, (int, float)) and tokens >= 0:
                vector.llm_tokens = float(tokens)
                vector.mark_measured("llm_tokens")
        if latency is not None:
            vector.latency_s = float(latency)
            vector.mark_measured("latency_s")
        return vector

    # -- bind -------------------------------------------------------------------

    def bind_outcome(self, prediction_id: str, action,
                     record_scope=None) -> OutcomePrediction:
        """Bind a prediction to the REAL action that ran.

        Match check (request identity): action type, task, episode,
        strategy, solver, prediction TIMING (the prediction must predate the
        action), and measurement scope (an attempt-scope prediction is
        never scored against a task-scope total). A mismatch is recorded
        (``binding_mismatch``) and the comparison will only cover the
        matching parts — a prediction for task A is never scored against
        task B's execution, and a post-hoc prediction is never scored at
        all."""
        prediction = self.get(prediction_id)
        if prediction is None:
            raise StorageError(f"unknown prediction_id {prediction_id!r}")
        if prediction.bound_action_id is not None:
            # Idempotent re-bind to the same action.
            if prediction.bound_action_id == action.action_id:
                return prediction
            raise StorageError(
                f"prediction {prediction_id!r} is already bound to action "
                f"{prediction.bound_action_id!r}")
        spec = prediction.action_spec
        # A prediction conditioned on a HYPOTHETICAL snapshot is a
        # conditional outlook (second step of a rollout), never a real
        # one-step feedback sample: it must not be bound to a real action.
        try:
            row = self.store.conn.execute(
                "SELECT payload FROM belief_snapshots WHERE snapshot_id=?",
                (prediction.input_snapshot_id,)).fetchone()
            if row is not None:
                from or_harness.world_model.state import BeliefSnapshot
                snap = BeliefSnapshot.from_dict(
                    self.store.loads(row["payload"]))
                if snap.hypothetical:
                    prediction.bound_action_id = action.action_id
                    prediction.binding_mismatch = {
                        "input_snapshot": {
                            "snapshot_id": snap.snapshot_id,
                            "hypothetical": True,
                            "reason": "prediction conditioned on a "
                                      "hypothetical successor state: a "
                                      "conditional outlook, not a real "
                                      "one-step prediction",
                        },
                    }
                    self._save(prediction)
                    return prediction
        except StorageError:
            raise
        except Exception:
            pass  # snapshot unreadable: fall through to normal checks
        mismatch: Dict[str, Any] = {}
        if action.action_type != spec.action_type:
            mismatch["action_type"] = {"predicted": spec.action_type,
                                      "actual": action.action_type}
        # Request identity: the prediction must be FOR THIS task and
        # episode. A prediction for task A never scores task B's execution.
        if spec.task_id and action.task_id != spec.task_id:
            mismatch["task_id"] = {"predicted": spec.task_id,
                                   "actual": action.task_id}
        if (spec.episode_id is not None
                and action.episode_id is not None
                and spec.episode_id != action.episode_id):
            mismatch["episode_id"] = {"predicted": spec.episode_id,
                                      "actual": action.episode_id}
        if (spec.strategy_id is not None
                and action.params.get("strategy_id") not in
                (None, spec.strategy_id)):
            mismatch["strategy_id"] = {"predicted": spec.strategy_id,
                                       "actual":
                                           action.params.get("strategy_id")}
        if (spec.solver is not None
                and action.params.get("solver") not in
                (None, spec.solver)):
            mismatch["solver"] = {"predicted": spec.solver,
                                  "actual": action.params.get("solver")}
        # Timing: the prediction must have been made BEFORE the action
        # started. A post-hoc "prediction" is hindsight, not evidence.
        if prediction.created_at > action.started_at:
            mismatch["timing"] = {
                "predicted_at": prediction.created_at,
                "action_started_at": action.started_at,
                "reason": "prediction was generated after the action began"}
        # Measurement scope (BOTH directions): an attempt-scope prediction
        # is never scored against a task-scope total, and a task-scope
        # prediction is never scored against an attempt.
        actual_scope = getattr(record_scope, "measurement_scope", "attempt")
        if actual_scope != spec.measurement_scope:
            mismatch["measurement_scope"] = {
                "predicted": spec.measurement_scope,
                "actual": actual_scope}
        # Source: only REAL actions produce real outcomes. A hypothetical
        # action's "outcome" is an inference, never feedback evidence.
        if getattr(action, "source", "executed") == "hypothetical":
            mismatch["action_source"] = {
                "predicted": "executed/agent_reported",
                "actual": "hypothetical",
                "reason": "a hypothetical action is not a real outcome"}
        # Unknown identity never counts as matching: when the prediction
        # names a task/episode the action must too.
        if spec.task_id and not action.task_id:
            mismatch["task_id"] = {"predicted": spec.task_id,
                                   "actual": None}
        prediction.bound_action_id = action.action_id
        prediction.binding_mismatch = mismatch or None
        self._save(prediction)
        return prediction

    # -- compare ------------------------------------------------------------------

    def compare_prediction(self, prediction_id: str, record) \
            -> OutcomePrediction:
        """Compare the frozen prediction against the real outcome.

        ``record`` is the ExecutionRecord of the bound action's linked
        execution. Comparison covers only fields BOTH sides define:
        - status category (predicted outcome_status vs actual quality
          status);
        - quality (when predicted AND the execution produced one);
        - cost per dimension (both sides measured — the same
          both-sides-measured discipline as compute_cost_feedback).
        Anything else is listed under ``not_compared`` with a reason.
        The result is APPENDED to ``feedback``; the frozen prediction is
        never modified. Re-running the same comparison is idempotent."""
        prediction = self.get(prediction_id)
        if prediction is None:
            raise StorageError(f"unknown prediction_id {prediction_id!r}")
        if prediction.bound_action_id is None:
            raise StorageError(
                f"prediction {prediction_id!r} is not bound to an action; "
                "bind_outcome first")
        if prediction.status != "valid":
            raise StorageError(
                f"prediction {prediction_id!r} has status "
                f"{prediction.status!r}: only a valid prediction is "
                "comparable")
        if prediction.feedback is not None:
            # Idempotent replay: the stored feedback stands; the model is
            # never re-invoked and nothing is re-counted.
            return prediction
        if prediction.binding_mismatch:
            # A mismatched binding is recorded as such — its comparable
            # scope is empty. No error metrics are fabricated from it.
            prediction.feedback = {
                "compared": False,
                "reason": "binding mismatch: the executed action differs "
                          "from the predicted candidate",
                "mismatch": prediction.binding_mismatch,
            }
            self._save(prediction)
            return prediction

        predicted = prediction.predicted
        actual_status = record.quality.get("status")
        actual_feasible = record.quality.get("feasible")
        compared: Dict[str, Any] = {}
        not_compared: Dict[str, str] = {}

        # Status category.
        pred_status = predicted.get("outcome_status")
        if pred_status in (None, "unknown"):
            not_compared["outcome_status"] = "not predicted"
        else:
            compared["outcome_status"] = {
                "predicted": pred_status, "actual": actual_status,
                "match": pred_status == actual_status,
            }
        # Feasibility.
        pred_feasible = predicted.get("feasible")
        if pred_feasible is None:
            not_compared["feasible"] = "not predicted"
        else:
            compared["feasible"] = {
                "predicted": pred_feasible,
                "actual": bool(actual_feasible),
                "match": pred_feasible == bool(actual_feasible),
            }
        # Quality: only when predicted AND actually observed (an
        # infeasible execution has no quality to compare).
        pred_quality = predicted.get("quality")
        if pred_quality is None:
            not_compared["quality"] = "not predicted"
        elif not actual_feasible:
            not_compared["quality"] = "no observed quality (execution " \
                                      "produced no usable solution)"
        else:
            from or_harness.strategy.stats import quality_score
            actual_q = quality_score(record)
            entry = {
                "predicted": round(float(pred_quality), 4),
                "actual": round(actual_q, 4),
                "abs_error": round(abs(float(pred_quality) - actual_q), 4),
            }
            # Honest quality semantics: a feasible execution with no
            # gap/bound yields the heuristic 0.5 placeholder — that is NOT
            # an observed quality truth, and an error measured against it
            # is not evidence about the model.
            if (record.quality.get("gap") is None
                    and record.quality.get("status") != "optimal"):
                entry["heuristic_placeholder"] = True
                entry["note"] = ("actual quality is the 0.5 heuristic "
                                 "(no gap/bound observed): not an "
                                 "observation; abs_error is informational "
                                 "only")
            compared["quality"] = entry
        # Cost per dimension: both sides measured. Dimensions predicted
        # but not observed (and vice versa) are listed individually — a
        # missing comparison is information, not silence.
        pred_cost_raw = predicted.get("cost")
        if not pred_cost_raw:
            not_compared["cost"] = "not predicted"
        else:
            pred_cost = CostVector.from_dict(pred_cost_raw)
            actual_measured = record.cost.measured_dims()
            for dim in COST_DIMENSIONS:
                if dim in pred_cost.measured_dims() \
                        and dim not in actual_measured:
                    not_compared[f"cost.{dim}"] = (
                        "predicted but not measured on the execution")
            # Shared arithmetic with the legacy record-chain feedback
            # (cost_error_per_dim): both-sides-measured dims only.
            from or_harness.core.schema import cost_error_per_dim
            per_dim = cost_error_per_dim(pred_cost, record.cost)
            if per_dim:
                compared["cost"] = per_dim
            elif "cost" not in not_compared and not any(
                    k.startswith("cost.") for k in not_compared):
                not_compared["cost"] = ("no dimension measured on both "
                                        "sides")

        if not compared:
            # compared=true must mean at least one field was ACTUALLY
            # compared; otherwise the honest answer is "nothing comparable"
            # with the per-field reasons preserved — never a vacuous
            # success sample.
            prediction.feedback = {
                "compared": False,
                "reason": "no comparable field: every predicted field was "
                          "either unobserved on the execution or not "
                          "predicted (see not_compared)",
                "execution_id": record.execution_id,
                "compared_fields": {},
                "not_compared": not_compared,
            }
        else:
            prediction.feedback = {
                "compared": True,
                "execution_id": record.execution_id,
                "compared_fields": compared,
                "not_compared": not_compared,
                "note": "online comparison records facts and errors only; "
                        "knowledge updates still go through explicit offline "
                        "induction with verification",
            }
        self._save(prediction)
        return prediction

    # -- persistence / query ---------------------------------------------------

    def _save(self, prediction: OutcomePrediction) -> None:
        payload = OutcomePrediction.from_dict(prediction.to_dict())
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO world_model_predictions "
                "(prediction_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (payload.prediction_id, payload.action_spec.task_id,
                 payload.action_spec.episode_id, payload.created_at,
                 self.store.dumps(payload.to_dict())))

    def get(self, prediction_id: str) -> Optional[OutcomePrediction]:
        row = self.store.conn.execute(
            "SELECT payload FROM world_model_predictions "
            "WHERE prediction_id=?", (prediction_id,)).fetchone()
        return (OutcomePrediction.from_dict(self.store.loads(row["payload"]))
                if row else None)

    def query(self, *, task_id: Optional[str] = None,
              episode_id: Optional[str] = None) -> List[OutcomePrediction]:
        sql = "SELECT payload FROM world_model_predictions"
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?"); params.append(task_id)
        if episode_id is not None:
            clauses.append("episode_id=?"); params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, prediction_id ASC"
        return [OutcomePrediction.from_dict(self.store.loads(r["payload"]))
                for r in self.store.conn.execute(sql, params).fetchall()]
