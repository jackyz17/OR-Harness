"""C plumbing fixtures: real harness lifecycle, synthetic host meter."""
import copy
import time

from tests.harness.g_fixtures import GCase, COUPLED

REUSE = {"name": "reuse joint model", "steps": [
    "Reuse the available joint supply-chain model and preserve the factory coupling.",
    "Change the coupling constraint, solve once, then run acceptance once."]}
REBUILD = {"name": "rebuild and decompose", "steps": [
    "Rebuild the supply-chain model from the original requirements.",
    "Solve three subproblems, merge their answers with the factory coupling preserved.",
    "Run acceptance on the merged answer and repair any violated coupling."]}


class CCase(GCase):
    def seed_host(self, episode, *, existing=True, model="executor-model/v1"):
        self.h.report_action("model" if existing else "select_strategy", self.task, episode,
            outcome={"artifact": "joint-model.lp", "status": "available"} if existing else {},
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": model,
                        "measured": []})
        if not existing:
            self.h.snapshot(self.task, episode, task_progress={"model_artifact": {
                "value": {"status": "not_created"}, "provenance": "agent_reported",
                "epistemic": "fact"}})

    def predict_c(self, episode="ep1", *, cost=None, method=REUSE, existing=True,
                  model="executor-model/v1"):
        self.seed_host(episode, existing=existing, model=model)
        self.provider.payload.pop("cost", None)
        if cost is not None:
            self.provider.payload["cost"] = copy.deepcopy(cost)
        return self.h.predict_strategy_outcome(self.task,
            {"action_type": "execute_strategy", "strategy_id": "S04", "method": method}, episode)

    def finish_c(self, prediction, *, tools=8, tokens=300, rounds=2, terminal="completed",
                 counters=False, calls=None, model="executor-model/v1", duplicate_record=None):
        end = time.time()
        decision_id = prediction.trace.model_info.get("decision_action_id")
        decision = self.h.actions.get(decision_id) if decision_id else None
        anchor = decision.started_at if decision else prediction.trace.created_at
        report = {"schema": "or-host-usage/1", "host": "test-host", "model": model,
            "scope": "remaining_to_task_end", "report_id": "report-" + prediction.prediction_id,
            "window": {"started_at": anchor, "ended_at": end,
                       "anchor_action_id": decision_id,
                       "complete": True, "end_event": "finish_task"},
            "tokens": {"prompt_tokens": tokens - 100 if tokens >= 100 else tokens,
                       "completion_tokens": 100 if tokens >= 100 else 0,
                       "calls": rounds}, "tool_calls": tools,
            "measured": ["llm_tokens", "tool_calls"]}
        if counters:
            report.pop("tokens")
            report.pop("tool_calls")
            report["counters"] = {"start": {"prompt_tokens": 900, "completion_tokens": 100,
                "tool_calls": 20, "calls": 7}, "end": {
                "prompt_tokens": 900 + tokens, "completion_tokens": 100,
                "tool_calls": 20 + tools, "calls": 7 + rounds}}
        if calls is not None:
            report.pop("tokens", None)
            report.pop("tool_calls", None)
            report["call_details"] = calls
        if duplicate_record is not None:
            # A repeated report reference must not re-charge the same call.
            self.h.record(duplicate_record, host_usage=report)
        finish = self.h.report_action("finish_task", self.task, prediction.candidate.episode_id,
            status="cancelled" if terminal == "aborted" else terminal,
            started_at=end, ended_at=end, host_usage=report)
        result = self.h.close_episode(self.task["task_id"], prediction.candidate.episode_id,
            terminal_state=terminal, finish_action_id=finish["action_id"], min_calibration_samples=1)
        return result, report
