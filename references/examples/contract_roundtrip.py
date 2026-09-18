"""Runnable contract example: build, round-trip, and read legacy payloads.

No model, no network, no solver, no credentials. Run from the repo root:

    PYTHONPATH=src python3 references/examples/contract_roundtrip.py

It demonstrates the three things an outer harness agent actually needs:

1. building a strategy-outcome contract for a task that has NO ``model``
   field and several UNKNOWN quantities — and seeing that the unknowns stay
   unknown instead of becoming measured zeros;
2. a lossless JSON round trip of both contracts;
3. reading a legacy unversioned prediction payload through the legacy view,
   and seeing which old fields have no honest new-contract equivalent.

And two things it must never get wrong:

4. a legacy ``ActionSpec`` is adapted WITHOUT losing its execution
   configuration, and an unmappable legacy scope is REFUSED rather than
   silently shrunk;
5. a window with an unfinished attempt, or one belonging to another task /
   episode / strategy, is NOT comparable.

It also asserts its own invariants, so a regression makes the script fail
rather than quietly printing a plausible-looking object.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import CostVector  # noqa: E402
from or_harness.world_model.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityEvolutionPrediction,
    EvidenceRef,
    ExpectedChange,
    ExpectedCost,
    ExperienceScope,
    LearningOperation,
    StrategyOutcomePrediction,
    TaskTargeting,
    UncertaintyStatement,
    VerificationCondition,
    detect_payload_version,
)
from or_harness.world_model.execution_window import (  # noqa: E402
    build_execution_window,
    window_identity_problems,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402

# A task with NO 'model' field: a normal state, not a defect.
TASK = {
    "task_id": "t_demo",
    "family": "routing",
    "description": "Multi-vehicle routing with a shared depot capacity.",
    # Note: no 'model', and no CIR — the contract is still buildable.
}

# A legacy payload written by the pre-contract prediction path.
LEGACY_PAYLOAD = {
    "prediction_id": "wp_demo_legacy",
    "input_snapshot_id": "bs_demo",
    "action_spec": {"action_type": "execute_strategy", "strategy_id": "S04",
                    "task_id": "t_demo", "measurement_scope": "attempt"},
    "status": "valid",
    "predicted": {
        "outcome_status": "feasible",
        "feasible": True,
        "quality": 0.82,
        "failure_prob": 0.15,
        "cost": {"llm_tokens": 1500.0},
        "cost_measured": ["llm_tokens"],
        "state_changes": {"current_solution": {"status": "feasible"}},
        "knowledge_changes": [{"change": "supports",
                               "horizon": "after_execution"}],
    },
    "confidence": 0.6,                       # self-reported, UNCALIBRATED
    "evidence_basis": ["coverage.cell_statistics.S04"],
    "call_cost": {"llm_tokens": 80.0},       # the CALL's own spend
}


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    home = tempfile.mkdtemp(prefix="orx-contract-demo-")
    harness = ORHarness(home=home)
    try:
        # ------------------------------------------------------------------
        banner("1. Strategy outcome contract — no model, partial unknowns")
        # ------------------------------------------------------------------
        prediction = harness.build_strategy_outcome_contract(
            TASK,
            CandidateRef(action_type="execute_strategy", strategy_id="S04",
                         solver="highs", scope="attempt",
                         preconditions=["the depot capacity is known"],
                         expected_scope=["write the model", "solve"],
                         stop_conditions=["objective gap <= 1%"]),
            "ep1",
            benefit=BenefitEstimate(
                kind="solution_quality",
                metric="normalized_objective_gap",
                unit="1 - mip_gap",
                value=0.82,
                baseline=BaselineStatement(
                    kind="conditional_stats", value=0.74,
                    ref=EvidenceRef("entry", "se_demo",
                                    "the claim currently in force")),
            ),
            cost=ExpectedCost(
                expected=CostVector(llm_tokens=1500.0,
                                    solver_runtime_s=4.0,
                                    measured={"llm_tokens",
                                              "solver_runtime_s"}),
                measured=["llm_tokens", "solver_runtime_s"],
                notes=["retries was never observed for this strategy"]),
            uncertainty=UncertaintyStatement(
                source="framework_heuristic",
                execution_randomness=0.2,
                knowledge_gap=0.7,
                missing={"calibrated_probability":
                         "no measured reliability for this class yet"}),
            unsupported_fields={
                "risk.events[loss].severity": "no loss model exists",
                "cost.retries": "never measured for this strategy"},
            evidence_basis=[EvidenceRef("entry", "se_demo"),
                            EvidenceRef("execution", "ex_demo_1")],
        )

        print(f"status              : {prediction.status}")
        print(f"provider_configured : {prediction.provider_configured}")
        print(f"service_available   : {prediction.service_available}")
        print(f"prediction_made     : {prediction.prediction_made}")
        print(f"scope               : {prediction.scope}")
        print(f"comparable          : {prediction.trace.comparable}")
        print(f"benefit             : {prediction.benefit.value} "
              f"({prediction.benefit.metric}, unit={prediction.benefit.unit})")
        print(f"benefit baseline    : "
              f"{prediction.benefit.baseline.kind}="
              f"{prediction.benefit.baseline.value}")
        print(f"cost expected_measured: {prediction.cost.measured}")
        print(f"risk                : {prediction.risk}")
        print(f"unsupported_fields  : {prediction.trace.unsupported_fields}")
        print(f"call cost           : {prediction.trace.call_cost}")
        print(f"input snapshot      : {prediction.trace.input_snapshot_id} "
              f"(version {prediction.trace.input_version})")
        for note in prediction.notes:
            print(f"note                : {note}")

        # Invariants this example asserts.
        assert prediction.status == "contract_only", (
            "no prediction service is attached: the contract must say so")
        # Even with a provider configured, merely CONSTRUCTING a contract
        # makes no model call — so it can never be 'valid'. (Here there is
        # not even a provider: provider_configured is False.)
        assert prediction.provider_configured is False
        assert prediction.prediction_made is False, (
            "a built contract with no provider is not a forecast")
        assert prediction.risk is None, "an unpredicted risk stays absent"
        assert prediction.trace.call_cost is None, (
            "building a contract makes no model call, so there is no call "
            "cost to report")
        assert "model" not in TASK, "this task really has no model field"
        assert prediction.benefit.value is not None
        assert prediction.benefit.baseline is not None

        # ------------------------------------------------------------------
        banner("2. Lossless JSON round trip")
        # ------------------------------------------------------------------
        dumped = prediction.to_dict()
        reloaded = StrategyOutcomePrediction.from_dict(dumped)
        assert reloaded.to_dict() == dumped, "round trip must be lossless"
        print(f"round trip OK, {len(json.dumps(dumped))} bytes of JSON")
        # Unknown stays unknown through the round trip.
        assert "solver_runtime_s" in reloaded.cost.measured
        assert "retries" not in reloaded.cost.expected.measured_dims()
        print("unknown dimensions stayed unknown (no fabricated zeros)")

        # ------------------------------------------------------------------
        banner("3. Capability evolution contract — contract only")
        # ------------------------------------------------------------------
        evolution = harness.build_capability_evolution_contract(
            TASK,
            LearningOperation(
                operation_type="induce", strategy_id="S04",
                description="induce a claim for S04 in this structural cell",
                scope=ExperienceScope(execution_ids=["ex_demo_1", "ex_demo_2"],
                                      task_ids=["t_demo", "t_demo_2"],
                                      family="routing",
                                      cell_token="rc[0.75,1.00]")),
            experience_scope=ExperienceScope(
                execution_ids=["ex_demo_1", "ex_demo_2"],
                task_ids=["t_demo", "t_demo_2"]),
            task_targeting=TaskTargeting(
                description="routing tasks with high resource coupling",
                family="routing", cell_token="rc[0.75,1.00]"),
            baseline=BaselineStatement(kind="no_knowledge", value=None,
                                       note="no claim covers S04 yet"),
            horizon="the next 10 matching routing tasks",
            horizon_tasks=10,
            expected_changes=[ExpectedChange(
                metric="mean_solution_quality", direction="increase",
                value=None,
                baseline=BaselineStatement(kind="no_knowledge"),
                notes=["direction only: no magnitude is claimable from two "
                       "executions"])],
            learning_cost=ExpectedCost(
                expected=CostVector(llm_tokens=4000.0,
                                    measured={"llm_tokens"}),
                measured=["llm_tokens"]),
            verification_conditions=[VerificationCondition(
                condition=("the claim passes admission verification and its "
                           "interval holds on the next 5 UNSEEN tasks"),
                evaluable=True,
                check_basis="induce --verify on executions outside the "
                            "inducing set",
                prediction_made=True, fact_bound=False,
                effect_verified=False)],
        )
        print(f"status              : {evolution.status}")
        print(f"provider_configured : {evolution.provider_configured}")
        print(f"service_implemented : {evolution.service_implemented}")
        print(f"service_available   : {evolution.service_available}")
        print(f"horizon             : {evolution.horizon} "
              f"({evolution.horizon_tasks} tasks)")
        print(f"capability evidence : "
              f"{ {k: v.status for k, v in evolution.current_evidence.sources.items()} }")
        print(f"score_scheme        : {evolution.current_evidence.score_scheme}")
        print(f"distinct tasks used : "
              f"{evolution.experience_scope.distinct_tasks}")
        print(f"verification        : prediction_made="
              f"{evolution.verification_conditions[0].prediction_made}, "
              f"fact_bound={evolution.verification_conditions[0].fact_bound}, "
              f"effect_verified="
              f"{evolution.verification_conditions[0].effect_verified}")
        for note in evolution.notes:
            print(f"note                : {note}")

        assert evolution.status == "contract_only", (
            "the capability prediction SERVICE is not attached — this is a "
            "schema-level object, not a capability forecast")
        assert evolution.service_available is False
        assert evolution.service_implemented is False, (
            "this build implements no capability-evolution service: a "
            "configured provider must not make it look available")
        assert evolution.prediction_made is False
        assert evolution.current_evidence.score_scheme == "no_composite_score"
        assert not hasattr(evolution.current_evidence, "composite_score"), (
            "no composite H score exists, by design")
        dumped_evo = evolution.to_dict()
        assert CapabilityEvolutionPrediction.from_dict(
            dumped_evo).to_dict() == dumped_evo
        print("round trip OK; no capability increment was invented")

        # ------------------------------------------------------------------
        banner("4. Reading a legacy payload (old semantics preserved)")
        # ------------------------------------------------------------------
        print(f"detected version    : {detect_payload_version(LEGACY_PAYLOAD)}")
        read = harness.read_prediction_payload(LEGACY_PAYLOAD)
        view = read["legacy_view"]
        print(f"legacy              : {read['legacy']}")
        print(f"mapped fields       : {sorted(view['mapped_fields'])}")
        print("gaps (recorded nothing, left absent):")
        for gap in view["gaps"]:
            print(f"  - {gap}")
        print("unmappable old content:")
        for key in sorted(view["unmappable"]):
            print(f"  - {key}")
        print(f"capability evidence : {view['capability_evidence']['sources']}")

        assert read["contract_version"] == LEGACY_CONTRACT_VERSION
        assert read["legacy"] is True
        assert view["mapped_fields"]["risk.events"][0]["severity"] is None, (
            "severity was never recorded: it must not be invented")
        assert "capability_increment" not in view["mapped_fields"]
        assert view["capability_evidence"]["sources"]["w_or"]["status"] == \
            "no_evidence", "nothing observed W_OR in a legacy payload"

        # An unknown version fails explicitly rather than being guessed at.
        unknown = harness.read_prediction_payload(
            {"contract_version": "wm-contract/99", "prediction_id": "x"})
        print(f"unknown version     : supported={unknown['supported']} "
              f"({unknown['contract_version']})")
        assert unknown["supported"] is False
        assert "contract" not in unknown and "legacy_view" not in unknown

        # ------------------------------------------------------------------
        banner("5. Legacy adaptation must not change the candidate")
        # ------------------------------------------------------------------
        legacy_spec = ActionSpec(
            action_type="execute_strategy", task_id="t_demo",
            strategy_id="S04", measurement_scope="attempt",
            params={"time_limit": 60, "mip_gap": 0.01, "seed": 42},
            budget_hint={"solver_runtime_s": 30.0})
        candidate = CandidateRef.from_action_spec(legacy_spec)
        print(f"config preserved    : {candidate.config}")
        print(f"scope / basis       : {candidate.scope} / "
              f"{candidate.scope_basis}")
        # The execution conditions survive: a time-limited, gap-targeted,
        # seeded run is NOT the same candidate as an unbounded one.
        assert candidate.config["time_limit"] == 60
        assert candidate.config["mip_gap"] == 0.01
        assert candidate.config["seed"] == 42
        assert candidate.config["budget_hint"] == {"solver_runtime_s": 30.0}
        assert candidate.scope_basis == "legacy_attempt"

        # A legacy 'task' scope covers the WHOLE task: it has no contract
        # equivalent, so it is refused instead of silently becoming "one
        # attempt".
        legacy_spec.measurement_scope = "task"
        try:
            CandidateRef.from_action_spec(legacy_spec)
            raise AssertionError("a legacy 'task' scope must be refused")
        except ValueError as exc:
            print(f"legacy 'task' scope : refused ({str(exc)[:64]}...)")

        # ------------------------------------------------------------------
        banner("6. An unfinished window is not comparable")
        # ------------------------------------------------------------------
        class _Action:
            def __init__(self, action_id, status, execution_id):
                self.action_id = action_id
                self.action_type = "execute_strategy"
                self.task_id = "t_demo"
                self.episode_id = "ep1"
                self.source = "executed"
                self.status = status
                self.rollup = "own"
                self.cost = None
                self.linked_execution_id = execution_id
                self.started_at = 0.0
                self.ended_at = 1.0
                self.params = {"strategy_id": "S04"}

        running = build_execution_window(
            [_Action("ac_1", "completed", "ex_1"),
             _Action("ac_2", "running", "ex_2")],
            task_id="t_demo", episode_id="ep1", strategy_id="S04")
        print(f"n_attempts          : {running.n_attempts}")
        print(f"n_unfinished        : {running.n_unfinished}")
        print(f"comparable          : {running.comparable}")
        for reason in running.not_comparable_reasons:
            print(f"  - {reason}")
        assert running.comparable is False, (
            "a window with an attempt still running has no final numbers")
        assert running.n_unfinished == 1

        # And a window of ANOTHER identity may not be used for this one.
        foreign = build_execution_window(
            [_Action("ac_9", "completed", "ex_9")],
            task_id="t_demo", episode_id="ep1", strategy_id="S09")
        problems = window_identity_problems(
            foreign, task_id="t_demo", episode_id="ep1", strategy_id="S04")
        print(f"foreign window      : {problems}")
        assert problems and "S09" in problems[0]

        banner("All assertions passed")
        print(f"contract version    : {CONTRACT_VERSION}")
        print("Nothing above called a model, wrote a fact, or changed "
              "knowledge.")
        return 0
    finally:
        harness.close()


if __name__ == "__main__":
    raise SystemExit(main())
