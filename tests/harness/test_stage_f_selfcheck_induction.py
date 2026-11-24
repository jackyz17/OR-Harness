"""Self-check, H+ and induction guidance contracts after the Stage-F review.

The solving agent's behaviour is driven by the Skill, its references and the
world-model prompt. These tests protect the parts that must not drift:

* the Skill keeps the full main workflow and routes low-frequency detail to
  references;
* reference links and heading anchors resolve;
* the documented CLI commands/options exist in the real parser;
* self-check guidance is sufficiency-driven (no one-check cap, no default full
  re-solve) and keeps model-before-solver ordering;
* H+ guidance keeps the binary contract and allows proportionate verification
  evidence instead of an absolute ban on a complete re-solve;
* induction guidance keeps the no-increment outcomes and the durable-knowledge
  boundary;
* the H+ receipt flows themselves still accept the documented outcomes.

Assertions use core terminology and semantic clusters, not exact sentences or
an arbitrary line budget.
"""

import re
import unittest
from pathlib import Path

from test_induction_submission import CompactCase

from or_harness.world_model.hplus import jobs, receipt_observation

REPO = Path(__file__).resolve().parents[2]
AGENT_DOCS = (
    "SKILL.md",
    "references/self_check.md",
    "references/induction.md",
    "references/commands.md",
    "references/episode_closeout.md",
    "references/concepts.md",
    "references/world_model_contract.md",
    "references/strategy_outcome.md",
    "references/modeling.md",
    "references/snapshot_lifecycle.md",
)


def _read(path):
    return (REPO / path).read_text(encoding="utf-8")


def _flat(text):
    return " ".join(text.split())


def _headings(text):
    return [line.lstrip("#").strip() for line in text.splitlines()
            if re.match(r"^#{2,6}\s+", line)]


def _slug(heading):
    """Approximate the GitHub heading anchor used by the markdown links."""
    value = heading.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    return value.strip().replace(" ", "-")


def _anchors(text):
    return {_slug(heading) for heading in _headings(text)}


def _links(text):
    return re.findall(r"\]\(([^)\s]+)\)", text)


def _command_flags(parser, command):
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices and command in choices:
            return {opt for sub in choices[command]._actions
                    for opt in sub.option_strings}
    return set()


class TestSkillWorkflow(unittest.TestCase):
    """The entry Skill must still carry the whole main solving flow."""

    @classmethod
    def setUpClass(cls):
        cls.skill_raw = _read("SKILL.md")
        cls.skill = _flat(cls.skill_raw)

    def test_main_workflow_stages_are_present(self):
        headings = _headings(self.skill_raw)
        expected = ("Workflow", "1. Prepare and recall", "2. Select a method",
                    "3a. Build and execute", "3b. Self-check",
                    "4. Commit the answer and close the episode",
                    "5. Run offline induction",
                    "6. Continue with the next task",
                    "NL4Opt example", "References")
        lowered = [heading.lower() for heading in headings]
        for stage in expected:
            self.assertTrue(any(stage.lower() in heading for heading in lowered),
                            f"SKILL.md lost the workflow stage {stage!r}")

    def test_workflow_diagram_and_core_commands_are_present(self):
        for token in ("1. Prepare the task and recall experience",
                      "2. Propose candidates and select by U",
                      "3b. Self-check the answer",
                      "4. Commit the answer and close the episode",
                      "5. Run offline induction"):
            self.assertIn(token, self.skill)
        for command in ("orx profile", "orx recall",
                        "plan-next", "orx execute", "orx check-task",
                        "orx record --from-staged", "orx close-episode",
                        "orx induction-material", "orx induce",
                        "orx induction-status"):
            self.assertIn(command, self.skill)

    def test_skill_reference_links_resolve(self):
        for link in _links(self.skill_raw):
            target = link.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://")):
                continue
            path = (REPO / target).resolve()
            self.assertTrue(path.exists(), f"SKILL.md links to missing {target}")

    def test_skill_routes_low_frequency_entry_management_to_reference(self):
        # Core action stays in the Skill; the detailed management report is in
        # the induction reference.
        self.assertIn("references/induction.md#existing-entry-use-and-utility",
                      self.skill)
        self.assertIn("entry_usage", self.skill)
        self.assertIn("--attribute-effect", self.skill)
        induction = _flat(_read("references/induction.md"))
        for detail in ("status_view", "forward_calibration",
                       "keep", "demote", "promote"):
            self.assertIn(detail, induction)
        # The low-frequency vocabulary must not be required reading here.
        for detail in ("status_view", "forward_calibration"):
            self.assertNotIn(detail, self.skill)

    def test_skill_size_is_reported_not_enforced(self):
        # Non-blocking quality signal: the Skill is a flow document, and a
        # reference file is the right place for detail. No hard line cap.
        lines = len(self.skill_raw.splitlines())
        if lines > 300:
            print(f"NOTE: SKILL.md is {lines} lines; move detail to references/")

    def test_documented_commands_and_options_match_the_cli(self):
        from or_harness.cli import build_parser
        parser = build_parser()
        expected = {
            "profile": ("--task",),
            "recall": ("--task", "--top"),
            "plan-next": ("--task", "--episode", "--candidates"),
            "execute": ("--task", "--prediction", "--code", "--workspace"),
            "check-task": ("--check", "--verification"),
            "record": ("--from-staged", "--usage-file"),
            "close-episode": ("--task", "--episode", "--terminal",
                              "--attribute-effect"),
            "induction-material": ("--induction-id",),
            "induce": ("--induction-id", "--submission", "--dry-run"),
            "induction-status": ("--task",),
        }
        for command, flags in expected.items():
            available = _command_flags(parser, command)
            self.assertTrue(available, f"Skill names unknown command {command}")
            for flag in flags:
                self.assertIn(flag, available,
                              f"SKILL cites `orx {command} {flag}` which the "
                              "parser does not provide")


class TestReferenceAnchors(unittest.TestCase):
    """Every internal markdown link in the agent-facing docs must resolve."""

    def test_all_reference_links_and_anchors_resolve(self):
        problems = []
        for document in AGENT_DOCS:
            path = REPO / document
            if not path.exists():
                problems.append(f"{document} is missing")
                continue
            for link in _links(path.read_text(encoding="utf-8")):
                if link.startswith(("http://", "https://")):
                    continue
                target, _, anchor = link.partition("#")
                resolved = (path.parent / target).resolve() if target \
                    else path.resolve()
                if not resolved.exists():
                    problems.append(f"{document} -> missing {link}")
                    continue
                if anchor and resolved.suffix == ".md":
                    if anchor not in _anchors(resolved.read_text(
                            encoding="utf-8")):
                        problems.append(f"{document} -> unknown anchor {link}")
        self.assertEqual(problems, [], "broken documentation links:\n  "
                         + "\n  ".join(problems))


class TestSelfCheckGuidance(unittest.TestCase):
    """The self-check stays sufficient-evidence driven, not count-limited."""

    @classmethod
    def setUpClass(cls):
        cls.skill = _flat(_read("SKILL.md"))
        cls.self_check = _flat(_read("references/self_check.md"))
        cls.commands = _flat(_read("references/commands.md"))
        cls.induction = _flat(_read("references/induction.md"))

    def test_model_correctness_comes_before_solver_verification(self):
        for text in (self.skill, self.self_check):
            self.assertIn("model", text.lower())
            self.assertIn("second solver", text.lower())
            self.assertTrue("wrong model" in text.lower()
                            or "model-to-statement" in text.lower())
        self.assertIn("model-to-statement", self.commands)

    def test_enhanced_verification_has_no_mechanical_count_cap(self):
        for text in (self.skill, self.self_check, self.commands):
            self.assertNotIn("ONE targeted independent check", text)
        self.assertIn("sufficient", self.skill.lower())
        self.assertIn("targeted", self.skill.lower())
        self.assertIn("sufficient", self.self_check.lower())
        self.assertIn("sufficient", self.commands.lower())

    def test_complete_resolve_is_not_the_default(self):
        for text in (self.skill, self.self_check):
            self.assertIn("complete independent re-solve", text)
            self.assertIn("default", text.lower())

    def test_failed_or_insufficient_checks_are_not_a_pass(self):
        self.assertIn("not a pass", self.commands.lower())
        self.assertTrue("not a pass" in self.self_check
                        or ("script_error" in self.self_check
                            and "assertion" in self.self_check))
        self.assertIn("neither a pass nor a failure", self.skill)

    def test_hplus_conditions_are_not_a_second_task_requirement(self):
        for text in (self.skill, self.self_check):
            self.assertIn("not a second task requirement", text)
        self.assertIn("not an extra task requirement", self.commands)
        self.assertIn("not_checked", self.self_check)
        self.assertIn("unavailable", self.self_check)

    def test_hplus_verification_evidence_is_proportionate(self):
        for text in (self.skill, self.self_check, self.induction):
            self.assertIn("complete re-solve", text)
            self.assertIn("default", text.lower())
        for text in (self.self_check, self.induction):
            lowered = text.lower()
            self.assertTrue(any(word in lowered for word in (
                "small-scale", "enumeration", "comparison",
                "targeted reproduction")),
                "proportionate H+ evidence options are missing")

    def test_self_check_and_induction_do_not_contradict_each_other(self):
        # Neither document caps enhanced verification at one check, and both
        # treat a complete re-solve as optional rather than forbidden.
        self.assertNotIn("ONE targeted independent check", self.self_check)
        self.assertIn("not the default", self.self_check.lower())
        self.assertIn("not the default", self.induction.lower())
        self.assertNotIn("is not a valid condition for this run",
                         self.induction)


class TestHplusPrompt(unittest.TestCase):
    """H+ predicts a review-worthy opportunity without weakening publication."""

    @classmethod
    def setUpClass(cls):
        from or_harness.world_model.strategy_prediction import (
            STRATEGY_OUTCOME_SYSTEM_PROMPT,
        )
        cls.prompt = _flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)

    def test_plain_translation_without_a_specific_learning_mechanism_is_zero(self):
        self.assertIn("Use 0 for mere task translation without a specific "
                      "transferable mechanism", self.prompt)
        self.assertIn("routine feasibility or objective recomputation without "
                      "a transferable finding", self.prompt)
        self.assertNotIn("STANDARD TASK TRANSLATION IS NOT AN INCREMENT", self.prompt)

    def test_hplus_is_not_a_difficulty_or_success_judgement(self):
        for term in ("task novelty or complexity alone", "a successful solver call alone",
                     "no concrete reasonable learning hypothesis"):
            self.assertIn(term, self.prompt)

    def test_classical_mechanisms_and_independent_validation_are_eligible(self):
        self.assertIn("Distinguish domain familiarity", self.prompt)
        self.assertIn("classical or mature method is not disqualified", self.prompt)
        self.assertIn("meaningful independent validation of an existing weakly "
                      "supported mechanism", self.prompt)

    def test_covered_zero_requires_no_meaningful_new_evidence(self):
        self.assertIn("fully covers the mechanism and this run offers no "
                      "meaningful new evidence", self.prompt)
        self.assertIn("not a guarantee of publishable knowledge", self.prompt)

    def test_cold_start_is_not_automatic(self):
        self.assertIn("COLD START", self.prompt)
        self.assertIn("NEVER sufficient", self.prompt)

    def test_verification_condition_allows_affordable_checks(self):
        for term in ("minimum feasible check",
                     "no default full re-solve",
                     "small-scale full enumeration",
                     "original-versus-reduced",
                     "targeted reproduction"):
            self.assertIn(term, self.prompt)
        self.assertNotIn("second complete solve of the task", self.prompt)

    def test_contract_terms_and_review_semantics_are_kept(self):
        for term in ("decision", "claim", "basis",
                     "verification_conditions", "A 1 requests review",
                     "ALWAYS causes one review"):
            self.assertIn(term, self.prompt)


class TestInductionGuidance(unittest.TestCase):
    """Induction reference: navigation, no-increment rule, boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.raw = _read("references/induction.md")
        cls.text = _flat(cls.raw)

    def test_contents_map_links_to_existing_anchors(self):
        self.assertIn("## Contents", self.raw)
        contents = self.raw.split("## Contents", 1)[1].split("\n## ", 1)[0]
        anchors = _anchors(self.raw)
        for link in re.findall(r"\]\(#([^)\s]+)\)", contents):
            self.assertIn(link, anchors,
                          f"contents entry points at missing anchor #{link}")

    def test_cold_start_is_explained_once(self):
        self.assertEqual(
            self.text.lower().count(
                "empty library or missing historical reviews"), 1)

    def test_standard_translation_is_not_a_knowledge_increment(self):
        for term in ("NECESSARY SOLVING WORK", "profit-maximisation",
                     "capacity constraint", "integer", "time index"):
            self.assertIn(term, self.text)
        self.assertIn("required solving work, not a reusable increment",
                      self.text.lower())

    def test_review_can_settle_without_writing_a_rule(self):
        for result in ("new_rule", "revision", "evidence_added",
                       "already_covered", "no_new_knowledge",
                       "insufficient_evidence"):
            self.assertIn(result, self.text)
        self.assertIn("never required to settle a positive job", self.text)
        self.assertIn("successful review", self.text)

    def test_hypothesis_outcome_is_separate_from_the_learning_result(self):
        for term in ("hypothesis_outcome", "deviation_reason", "refuted",
                     "unknown", "not sufficiently verified"):
            self.assertIn(term, self.text)
        self.assertIn("case 4", self.text.lower())

    def test_durable_knowledge_cannot_depend_on_post_hoc_answers(self):
        for term in ("benchmark scores", "reference answers",
                     "POST-HOC VERIFICATION EVIDENCE"):
            self.assertTrue(term.lower() in self.text.lower(), term)
        self.assertIn("modelling_translation", self.text)
        self.assertIn("real task input", self.text)

    def test_hplus_verification_evidence_is_proportionate(self):
        lowered = self.text.lower()
        self.assertIn("complete re-solve is not the default", lowered)
        self.assertTrue(any(term in lowered for term in (
            "small-scale", "enumeration", "comparison", "reproduction")))

    def test_worked_review_flows_are_available(self):
        self.assertIn("Worked review flows", self.text)
        for result in ("new_rule", "no_new_knowledge",
                       "insufficient_evidence"):
            self.assertIn(result, self.text)


class TestSelfCheckExampleBoundary(unittest.TestCase):
    """Pre-solve examples must not use hidden reference answers."""

    def test_0087_judges_the_domain_from_the_task_statement(self):
        text = _flat(_read("references/self_check.md"))
        self.assertIn("0087", text)
        self.assertNotIn("reference value happens to be integral", text)
        self.assertIn("task states continuous", text)
        self.assertIn("external answer", text)

    def test_reference_evidence_is_post_hoc_only(self):
        self_check = _flat(_read("references/self_check.md"))
        induction = _flat(_read("references/induction.md"))
        self.assertIn("modelling premise", self_check)
        self.assertIn("POST-HOC VERIFICATION EVIDENCE", induction)


class TestReviewFlows(CompactCase):
    """Receipt-level flows the tightened review standard must permit."""

    def test_positive_job_can_settle_as_no_new_knowledge(self):
        _, record, closed = self.closed("flow-no-new")
        job_id = closed["induction_schedule"]["pending_ids"][0]
        self.material(job_id)
        submission = self.compact(
            job_id, record, result="no_new_knowledge", rule=None,
            analysis="The run is ordinary model translation; no mechanism or "
                     "method boundary was tested.")
        submission["hypothesis_outcome"] = "refuted"
        submission["deviation_reason"] = (
            "The proposed trap did not exist; the statement was direct.")
        result = self.h.induce_submission(submission, induction_id=job_id)
        receipt = result["induction_receipt"]
        self.assertEqual(receipt["outcome"], "no_increment")
        self.assertEqual(receipt["hypothesis_outcome"], "refuted")
        self.assertFalse(receipt["rule_updates"])
        self.assertEqual(self.h.sbank.count(), 0)
        self.assertEqual(receipt_observation(self.h, self.job(job_id))["label"],
                         0)
        # The verification condition was NOT checked; it must stay unknown.
        self.assertEqual(receipt["verification"][0]["state"], "unknown")

    def test_positive_job_can_settle_as_insufficient_evidence(self):
        _, record, closed = self.closed("flow-insufficient")
        job_id = closed["induction_schedule"]["pending_ids"][0]
        self.material(job_id)
        result = self.h.induce_submission({
            "result": "insufficient_evidence",
            "hypothesis_outcome": "unknown",
            "deviation_reason": "The second solver was unavailable, so the "
                                "proposed boundary was not checked.",
        }, induction_id=job_id)
        receipt = result["induction_receipt"]
        self.assertEqual(receipt["outcome"], "insufficient_material")
        self.assertFalse(receipt["rule_updates"])
        self.assertEqual(self.h.sbank.count(), 0)
        self.assertIsNone(
            receipt_observation(self.h, self.job(job_id))["label"])
        self.assertEqual(receipt["verification"][0]["state"], "unknown")

    def test_refuted_hypothesis_can_coexist_with_a_separate_new_rule(self):
        _, record, closed = self.closed("flow-refuted-new")
        job_id = closed["induction_schedule"]["pending_ids"][0]
        self.material(job_id)
        submission = self.compact(
            job_id, record,
            analysis="The sign-aware endpoint derivation is supported by the "
                     "captured enumeration and the recorded integer check; the "
                     "original upper-only claim was wrong.")
        submission["hypothesis_outcome"] = "refuted"
        submission["deviation_reason"] = (
            "The proposed upper-only rule does not cover the minimization case.")
        result = self.h.induce_submission(submission, induction_id=job_id)
        receipt = result["induction_receipt"]
        self.assertEqual(receipt["hypothesis_outcome"], "refuted")
        self.assertEqual(receipt["outcome"], "new_rule")
        self.assertEqual(receipt["rule_updates"][0]["version"], 1)
        self.assertTrue(receipt["rule_updates"][0]["substantive"])
        self.assertEqual(receipt_observation(self.h, self.job(job_id))["label"],
                         1)

    def _attach_hplus_verification(self, record, prediction_id, *, sourced):
        old = record.execution_features["task_check"]["acceptance_rule"]["spec"]
        item = {"prediction_id": prediction_id, "condition_index": 0,
                "state": "checked",
                "finding": "The lower endpoint and the independent optimum "
                           "match."}
        if sourced:
            item.update({
                "source": "fixture auditor inspected the captured script and "
                          "the recorded integer check",
                "evidence_refs": ["artifacts.submitted_script",
                                  "task_check.checks"]})
        self.h.check_task_result(record.execution_id,
                                 {**old, "hplus_verification": [item]})

    def test_condition_is_checked_only_with_sourced_evidence(self):
        # A declaration that merely SAYS checked is not completion.
        p_unsourced, unsourced, closed = self.closed("flow-condition-claim")
        job_unsourced = closed["induction_schedule"]["pending_ids"][0]
        self._attach_hplus_verification(
            unsourced, p_unsourced.prediction_id, sourced=False)
        self.material(job_unsourced)
        claimed = self.h.induce_submission({
            "result": "no_new_knowledge",
            "hypothesis_outcome": "unknown",
            "basis": {"kind": "recorded_checks",
                      "refs": [self.job(job_unsourced)["source_fact_ref"]],
                      "analysis": "Ordinary translation; no increment."},
        }, induction_id=job_unsourced)
        self.assertEqual(
            claimed["induction_receipt"]["verification"][0]["state"],
            "unknown")

        # With a source and locatable evidence refs it becomes checked.
        p_sourced, sourced, closed2 = self.closed("flow-condition-evidence")
        job_sourced = closed2["induction_schedule"]["pending_ids"][0]
        self._attach_hplus_verification(
            sourced, p_sourced.prediction_id, sourced=True)
        self.material(job_sourced)
        checked = self.h.induce_submission({
            "result": "no_new_knowledge",
            "hypothesis_outcome": "supported",
            "basis": {"kind": "recorded_checks",
                      "refs": [self.job(job_sourced)["source_fact_ref"]],
                      "analysis": "Ordinary translation; no increment."},
        }, induction_id=job_sourced)
        self.assertEqual(
            checked["induction_receipt"]["verification"][0]["state"],
            "checked")

    def test_zero_signal_never_creates_a_review_job(self):
        _, record, closed = self.closed("flow-zero", 0)
        self.assertEqual(closed["induction_schedule"]["pending_ids"], [])
        job = next(j for j in jobs(self.h) if j["task_id"] == "flow-zero")
        self.assertEqual(job["state"], "skipped")
        self.assertFalse(job["selected"])
        self.assertEqual(self.h.sbank.count(), 0)


class TestInterfaceInvariance(unittest.TestCase):
    """No new H+ schema, lifecycle or classification is introduced."""

    def test_binary_hplus_version_and_review_outcomes_are_unchanged(self):
        from or_harness.world_model.hplus import HPLUS_VERSION, REVIEW_OUTCOMES
        self.assertEqual(HPLUS_VERSION, "wm-hplus-binary/1")
        self.assertEqual(set(REVIEW_OUTCOMES), {
            "knowledge_submitted", "new_rule", "revised", "evidence_added",
            "already_covered", "no_increment", "hypothesis_refuted",
            "insufficient_material"})

    def test_capability_gain_keeps_its_existing_fields(self):
        from or_harness.world_model.contracts import CapabilityGain
        gain = CapabilityGain.from_dict({
            "decision": 1,
            "claim": "A monotone endpoint reduction may extend a rule.",
            "basis": ["candidate steps"],
            "verification_conditions": [{"condition": "check the endpoint",
                                         "check_basis": "captured script"}],
        })
        payload = gain.to_dict()
        for key in ("decision", "claim", "basis",
                    "verification_conditions"):
            self.assertIn(key, payload)
        self.assertEqual(payload["decision"], 1)

    def test_compact_submission_results_are_unchanged(self):
        from or_harness.world_model.induction_submission import RESULTS
        self.assertEqual(set(RESULTS), {
            "new_rule", "revision", "evidence_added", "already_covered",
            "no_new_knowledge", "insufficient_evidence"})

    def test_hplus_stays_outside_the_solving_utility(self):
        from or_harness.world_model.strategy_prediction import (
            STRATEGY_OUTCOME_SYSTEM_PROMPT,
        )
        prompt = _flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)
        self.assertIn("separate from G/C/R solving utility", prompt)
        self.assertIn("frozen binary H+ induction switch", prompt)


if __name__ == "__main__":
    unittest.main()
