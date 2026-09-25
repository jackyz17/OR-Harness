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

#: The documents an outer agent loads at runtime. These must stay free of
#: development history and must only name commands that really exist.
AGENT_FACING_DOCS = [
    "SKILL.md", "README.md", "README_zh.md",
    "references/commands.md", "references/concepts.md",
    "references/induction.md", "references/modeling.md",
    "references/examples.md", "references/world_model_contract.md",
    "references/prediction_context.md", "references/strategy_outcome.md",
    "references/episode_closeout.md",
]

#: Runnable examples the docs may point at. They are scripts, not prose, so
#: they are checked for EXISTENCE and (by the test suite) for running clean.
DOCUMENTED_EXAMPLES = [
    "references/examples/no_catalog.py",
    "references/examples/task_check.py",
    "references/examples/episode_closeout.py",
    "references/examples/strategy_outcome.py",
]

#: (command, [flags]) that the docs tell an agent to use.
DOCUMENTED_FLAGS = [
    ("contract", ["--kind", "--payload", "--task", "--spec", "--episode",
                  "--benefit", "--cost", "--risk", "--operation",
                  "--operation-desc", "--scope", "--targeting", "--baseline",
                  "--horizon", "--horizon-tasks", "--expected-change",
                  "--verification"]),
    ("profile", ["--task", "--cir", "--allow-empty-cir"]),
    ("recall", ["--task", "--top", "--exclude", "--candidate",
                "--memory-mode", "--include-unverified"]),
    ("predict-cost", ["--task", "--strategy"]),
    ("predict-strategy", ["--task", "--candidate", "--episode",
                          "--context", "--cir"]),
    ("bind-strategy", ["--prediction", "--action"]),
    ("close-episode", ["--task", "--episode", "--terminal",
                       "--finish-action", "--min-samples"]),
    ("calibration", ["--min-samples", "--rebuild"]),
    ("archive-calibration", ["--dry-run"]),
    ("predict-capability", ["--operation", "--task", "--bundle", "--horizon",
                            "--horizon-tasks", "--budget", "--task-id",
                            "--episode", "--timeout"]),
    ("compare-capability", ["--predictions", "--horizon-tasks",
                            "--allow-quality-loss"]),
    ("accept-capability", ["--recommendation", "--prediction", "--verify",
                           "--note", "--force"]),
    ("reject-capability", ["--recommendation", "--prediction", "--reason"]),
    ("bind-capability", ["--prediction", "--adoption-action"]),
    ("evaluate-capability", ["--prediction", "--tasks", "--paired",
                             "--allow-descriptive"]),
    ("check-task", ["--check", "--episode"]),
    ("induce", ["--strategy", "--all", "--rebuild", "--dry-run", "--force",
                "--note", "--verify", "--family", "--cell", "--peer-strategy",
                "--peer-cell", "--relation"]),
    ("plan-next", ["--task", "--episode", "--candidates", "--horizon",
                   "--max-calls", "--delta", "--prediction-mode"]),
    ("induction-candidates", []),
    ("inspect", ["--bank", "--task", "--evaluation", "--prediction"]),
    ("snapshot", ["--task", "--episode"]),
    ("action", ["--report", "--task", "--amend-cost", "--cost"]),
    ("budget", ["--task", "--episode", "--declare"]),
    ("amend-cost", ["--override", "--mode"]),
    ("retire", ["--entry", "--reason"]),
    ("exclude-execution", ["--execution", "--reason", "--superseded-by"]),
    ("restore-execution", ["--execution", "--reason"]),
    ("rebuild-index", ["--layer", "--dry-run"]),
    ("context", ["--task", "--episode", "--top", "--cir",
                 "--math", "--include-unverified", "--context-id",
                 "--no-persist"]),
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
        "PredictionServiceStatus", "SERVICE_IMPLEMENTED_KINDS",
        "LEGACY_UNMAPPABLE_SCOPES", "contract_status_from_legacy_status",
        "parse_window_id", "window_identity_problems",
        "PredictionContext", "JointProblemRepresentation", "MathAttributes",
        "RetrievalView", "UnsupportedContextVersion",
        "PREDICTION_CONTEXT_VERSION", "JOINT_REPRESENTATION_VERSION",
        "MATH_ATTRIBUTE_ORIGINS", "EVIDENCE_CLASSES",
        "build_context", "build_joint_representation", "build_retrieval_view",
        "capability_evidence_with_sources", "capability_version",
        "cell_evidence_from_stats", "classify_evidence",
        "context_identity_problems", "dedupe_evidence",
        "evidence_identity", "frozen_knowledge_available",
        "frozen_knowledge_view", "knowledge_targets_from_context",
        "math_attributes", "memory_content_digest", "resolve_effective_cir",
        "retrieval_reuse_problems", "snapshot_conditions",
        "snapshot_from_context", "structure_problems",
        "task_with_effective_cir", "effective_input_version",
        "STRATEGY_OUTCOME_PROTOCOL_VERSION", "StrategyOutcomeService",
        "build_strategy_outcome_request", "parse_strategy_outcome_payload",
        # episode close-out and experience calibration (M4)
        "EPISODE_CLOSEOUT_VERSION", "CALIBRATION_SUMMARY_VERSION",
        "EPISODE_TERMINAL_STATES", "DEFAULT_MIN_CALIBRATION_SAMPLES",
        "EVENT_VOCABULARY_VERSION", "OBSERVABLE_RISK_EVENTS",
        "RETIRED_RISK_EVENTS", "PUBLISHED_SUMMARY_KEY",
        "CalibrationPolicy", "EpisodeCloseout", "RealOutcomeSummary",
        "StrategyPredictionEvaluation", "close_episode",
        "summarize_real_outcome", "evaluate_strategy_prediction",
        "build_calibration_summary", "calibration_summary_for_context",
        "calibration_window", "published_calibration_summary",
        "republish_calibration", "republish_if_in_window",
        "observe_episode_events", "archive_calibration_detail",
        "maybe_auto_archive", "episode_closeout_record",
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


def _all_commands(parser) -> set:
    out: set = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            out |= set(choices.keys() if hasattr(choices, "keys")
                       else choices)
    return out


#: `orx <command>` invocations the docs use, as (file, command) pairs. A
#: command named in prose but absent from the real parser is drift.
def _documented_orx_commands(text: str) -> set:
    return {m.group(1) for m in
            re.finditer(r"orx\s+([a-z][a-z0-9-]+)", text)}


def _check_frontmatter(path: Path) -> list:
    """The Skill frontmatter must be minimal and well-formed."""
    failures = []
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return [f"{path.name} does not start with YAML frontmatter"]
    end = text.find("\n---\n", 4)
    if end == -1:
        return [f"{path.name} frontmatter is not terminated"]
    block = text[4:end]
    keys = [line.split(":", 1)[0].strip()
            for line in block.splitlines()
            if line and not line.startswith((" ", "\t", "#"))
            and ":" in line]
    for required in ("name", "description"):
        if required not in keys:
            failures.append(f"{path.name} frontmatter is missing {required!r}")
    extra = [k for k in keys if k not in ("name", "description")]
    if extra:
        failures.append(
            f"{path.name} frontmatter has unnecessary keys {extra}: the "
            "skill spec needs only name + description")
    if "description" in keys:
        desc_start = block.find("description:")
        desc = block[desc_start:]
        if len(desc.strip()) < 200:
            failures.append(
                f"{path.name} description is too short to trigger reliably")
    return failures


def _check_not_hard_wrapped(path: Path) -> list:
    """Report a document whose prose is hard-wrapped.

    The unwrapper is idempotent and is the authority on what a wrapped
    paragraph is, so the exact test is "running it changes nothing" — a
    heuristic would misfire on legitimate lines that start with inline code.
    """
    if not path.exists():
        return []
    from _unwrap_prose import unwrap
    text = path.read_text(encoding="utf-8")
    if unwrap(text) == text:
        return []
    return [f"{path.name} is hard-wrapped: run "
            "`python3 references/examples/_unwrap_prose.py` to join the "
            "paragraphs into one line each"]


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

    # The Skill frontmatter must be minimal and well-formed.
    for problem in _check_frontmatter(ROOT / "SKILL.md"):
        failures.append(problem)
    print("SKILL.md frontmatter                       OK")

    # Every `orx <command>` the agent-facing docs mention must exist.
    real_commands = _all_commands(parser)
    for doc in AGENT_FACING_DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        named = _documented_orx_commands(path.read_text(encoding="utf-8"))
        unknown = sorted(c for c in named if c not in real_commands)
        for command in unknown:
            failures.append(f"{doc} mentions `orx {command}`, which does "
                            "not exist")
    print(f"orx command names across docs              OK")

    # Development history must not live in the agent-facing Skill. Only
    # unambiguous milestone markers are flagged — a domain name like a
    # production mode "M1" is not development history.
    history_pattern = re.compile(
        r"(?:world-model\s+M[0-9]\b|\(M[0-9]\)|\bPhase\s+[0-9]\b|"
        r"\bM[0-9]\s+note\b|\bdeferred to\s+M[0-9]\b|"
        r"\bwaits for\s+M[0-9]\b|\blands in\s+M[0-9]\b)")
    for doc in AGENT_FACING_DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in history_pattern.finditer(text):
            failures.append(
                f"{doc} carries development-history reference "
                f"{match.group(0)!r}: milestones belong in git history, not "
                "in the runtime Skill")
    print("no milestone/history references            OK")

    # Agent-facing docs must not be hard-wrapped: a paragraph is one logical
    # line, so a file's line count reflects structure rather than an editor's
    # width. A wrapped file is detected by finding a prose line whose
    # successor continues the same sentence.
    for doc in AGENT_FACING_DOCS:
        failures.extend(_check_not_hard_wrapped(ROOT / doc))
    print("docs are not hard-wrapped                 OK")

    # Every markdown link target in the agent-facing docs must exist.
    for doc in ["SKILL.md", "README.md", "README_zh.md",
                "references/world_model_contract.md",
                "references/prediction_context.md",
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

    # The runnable examples the docs promise must be present.
    for example in DOCUMENTED_EXAMPLES:
        if not (ROOT / example).exists():
            failures.append(f"documented example {example} is missing")
    print(f"runnable examples                         "
          f"{len(DOCUMENTED_EXAMPLES)} present")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll documented flags, symbols and links exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
