"""Doc consistency check: every CLI flag and Python symbol the docs promise.

Run from the repo root:

    PYTHONPATH=src python3 references/examples/_check_docs.py

It fails (non-zero exit) when a documented flag does not exist in the real
parser, or when a documented import is not importable. This is the guard
against documentation drifting ahead of the code.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.cli import build_parser  # noqa: E402

#: (command, [flags]) that the docs tell an agent to use.
DOCUMENTED_FLAGS = [
    ("contract", ["--kind", "--payload", "--task", "--spec", "--episode",
                  "--benefit", "--cost", "--operation", "--operation-desc",
                  "--scope", "--targeting", "--baseline", "--horizon",
                  "--horizon-tasks", "--expected-change", "--verification"]),
    ("profile", ["--task", "--code", "--cir"]),
    ("recall", ["--task", "--top", "--exclude", "--memory-mode",
                "--include-unverified"]),
    ("predict", ["--task", "--strategy"]),
    ("predict-outcome", ["--task", "--action-spec", "--episode",
                         "--parent-action"]),
    ("bind-outcome", ["--prediction", "--action"]),
    ("plan-next", ["--task", "--episode", "--candidates", "--horizon",
                   "--max-calls", "--delta", "--prediction-mode"]),
    ("assess-induction", ["--bundle", "--candidates-only", "--workload"]),
    ("bind-induction-outcome", ["--assessment"]),
    ("inspect", ["--bank", "--task"]),
    ("snapshot", ["--task", "--episode"]),
    ("action", ["--report", "--task", "--amend-cost", "--cost"]),
    ("budget", ["--task", "--episode", "--declare"]),
    ("gc", ["--mode", "--dry-run"]),
    ("retire", ["--entry", "--reason"]),
    ("rebuild-index", ["--layer", "--dry-run"]),
]

#: Global flags the docs promise.
DOCUMENTED_GLOBAL_FLAGS = ["--home", "--world-model", "--prediction-mode",
                           "--delta", "--alpha", "--beta", "--gamma",
                           "--cost-weights"]

#: Python imports the docs use.
DOCUMENTED_IMPORTS = [
    ("or_harness.api", ["ORHarness"]),
    ("or_harness.world_model", [
        "StrategyOutcomePrediction", "CapabilityEvolutionPrediction",
        "CandidateRef", "BenefitEstimate", "ExpectedCost", "RiskStatement",
        "UncertaintyStatement", "PredictionTrace", "EvidenceRef",
        "HarnessCapabilityEvidence", "LearningOperation", "ExperienceScope",
        "TaskTargeting", "BaselineStatement", "ExpectedChange",
        "VerificationCondition", "CONTRACT_VERSION",
        "LEGACY_CONTRACT_VERSION", "detect_payload_version",
        "legacy_prediction_view", "load_contract_payload",
        "prediction_kinds_for_mode", "validate_strategy_outcome",
        "validate_capability_evolution", "StrategyExecutionWindow",
        "build_execution_window", "window_id_for",
    ]),
    ("or_harness.world_model.contracts", ["LEGACY_UNMAPPABLE"]),
]


def _command_flags(parser, command: str) -> set:
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices \
                and command in action.choices:
            return {opt for sub in action.choices[command]._actions
                    for opt in sub.option_strings}
    raise AssertionError(f"command {command!r} not found in the parser")


def main() -> int:
    failures = []
    parser = build_parser()
    globals_present = {opt for action in parser._actions
                       for opt in action.option_strings}
    for flag in DOCUMENTED_GLOBAL_FLAGS:
        if flag not in globals_present:
            failures.append(f"global flag {flag} documented but missing")

    for command, flags in DOCUMENTED_FLAGS:
        present = _command_flags(parser, command)
        for flag in flags:
            if flag not in present:
                failures.append(
                    f"`orx {command} {flag}` documented but missing")
        print(f"orx {command:26s} {len(flags):2d} documented flags OK")

    for module_name, symbols in DOCUMENTED_IMPORTS:
        module = __import__(module_name, fromlist=symbols)
        for symbol in symbols:
            if not hasattr(module, symbol):
                failures.append(f"{module_name}.{symbol} documented but "
                                "not importable")
        print(f"{module_name:40s} {len(symbols):2d} symbols OK")

    # Every markdown link target in the agent-facing docs must exist.
    for doc in ["SKILL.md", "README.md", "README_zh.md",
                "references/world_model_contract.md",
                "references/commands.md", "references/concepts.md"]:
        path = ROOT / doc
        if not path.exists():
            failures.append(f"{doc} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#]+\.(?:md|py))\)", text):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{doc} links to missing {target}")
        print(f"{doc:44s} links OK")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll documented flags, symbols and links exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
