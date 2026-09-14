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
from or_harness.world_model.provider import (
    NotConfiguredProvider,
    WorldModelProvider,
)

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
                        ) -> OutcomePrediction:
        """Assemble the input view, call the provider, validate, persist.

        ``snapshot`` must be the FROZEN pre-action state (the caller takes
        it before this method runs). The prediction is persisted with
        whatever the provider returned — including failures, which keep
        their known call cost."""
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
                              "state_changes")}
        if isinstance(payload.get("cost"), dict):
            dims = {d: float(v) for d, v in payload["cost"].items()
                    if d in COST_DIMENSIONS and v is not None}
            predicted["cost"] = CostVector(
                **dims, measured=set(dims)).to_dict()
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

    def bind_outcome(self, prediction_id: str, action) -> OutcomePrediction:
        """Bind a prediction to the REAL action that ran.

        Match check: action type, strategy, solver. A mismatch is recorded
        (``binding_mismatch``) and the comparison will only cover the
        matching parts — a prediction for strategy A is never scored
        against strategy B's execution."""
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
        mismatch: Dict[str, Any] = {}
        if action.action_type != spec.action_type:
            mismatch["action_type"] = {"predicted": spec.action_type,
                                      "actual": action.action_type}
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
            compared["quality"] = {
                "predicted": round(float(pred_quality), 4),
                "actual": round(actual_q, 4),
                "abs_error": round(abs(float(pred_quality) - actual_q), 4),
            }
        # Cost per dimension: both sides measured. Dimensions predicted
        # but not observed (and vice versa) are listed individually — a
        # missing comparison is information, not silence.
        pred_cost_raw = predicted.get("cost")
        if not pred_cost_raw:
            not_compared["cost"] = "not predicted"
        else:
            pred_cost = CostVector.from_dict(pred_cost_raw)
            actual_measured = record.cost.measured_dims()
            per_dim: Dict[str, Dict[str, float]] = {}
            for dim in COST_DIMENSIONS:
                if dim not in pred_cost.measured_dims():
                    continue
                if dim not in actual_measured:
                    not_compared[f"cost.{dim}"] = (
                        "predicted but not measured on the execution")
                    continue
                p = getattr(pred_cost, dim)
                a = getattr(record.cost, dim)
                if p <= 0 and a <= 0:
                    continue  # both placeholders: nothing to learn
                per_dim[dim] = {
                    "predicted": round(p, 6),
                    "actual": round(a, 6),
                    "log_error": round(abs(math.log(
                        max(a, 1e-9) / max(p, 1e-9))), 4),
                }
            if per_dim:
                compared["cost"] = per_dim
            elif "cost" not in not_compared:
                not_compared["cost"] = ("no dimension measured on both "
                                        "sides")

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
