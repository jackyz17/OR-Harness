"""Self-check two-layer guidance and H+ increment/review tightening.

The behaviour of the solving agent and of the world-model reviewer is driven by
AGENT-FACING GUIDANCE plus the existing H+ protocol, not by a new framework
layer. These tests pin the guidance contract (docs and prompt) and exercise the
receipt-level review flows that the tightened standards must allow:

* a positive H+ job can settle as ``no_new_knowledge`` or
  ``insufficient_evidence`` without writing a rule;
* a refuted original hypothesis can coexist with a separate supported finding;
* an H+ verification condition is checked only with sourced evidence.

The framework interfaces themselves (binary H+, compact submission results,
capability_gain fields) stay unchanged; the last class checks that.
"""

import unittest
from pathlib import Path

from test_induction_submission import CompactCase
from test_rule_versions import SHORT_RULE

from or_harness.world_model.hplus import jobs, receipt_observation

REPO = Path(__file__).resolve().parents[2]


def _read(path):
    return (REPO / path).read_text(encoding="utf-8")


def _flat(text):
    return " ".join(text.split())


class TestSelfCheckGuidance(unittest.TestCase):
    """SKILL.md keeps the flow; the reference keeps the detail."""

    @classmethod
    def setUpClass(cls):
        cls.skill_raw = _read("SKILL.md")
        cls.skill = _flat(cls.skill_raw)
        cls.commands = _flat(_read("references/commands.md"))
        cls.closeout = _flat(_read("references/episode_closeout.md"))
        cls.self_check = _flat(_read("references/self_check.md"))
        cls.induction = _flat(_read("references/induction.md"))

    def test_skill_is_a_concise_flow_that_links_the_detail(self):
        self.assertLessEqual(len(self.skill_raw.splitlines()), 280)
        self.assertIn("references/self_check.md", self.skill)
        for phrase in ("does the model correctly express the task",
                       "base check",
                       "enhanced check",
                       "Do not run a complete independent re-solve as a "
                       "default step for every task",
                       "self-check done --> commit"):
            self.assertIn(phrase, self.skill)
        # Detailed lists, report fields and worked examples live in the
        # reference, not in the always-loaded entry doc.
        for detail in ("Layer 1 — necessary base check",
                       "report.scope.unchecked",
                       "two solvers agree on the same wrong model",
                       "B018"):
            self.assertNotIn(detail, self.skill)

    def test_reference_defines_the_two_layers(self):
        self.assertIn("Layer 1 — necessary base check", self.self_check)
        self.assertIn("Layer 2 — enhanced verification only when needed",
                      self.self_check)
        self.assertIn("These are check dimensions, not a fixed checklist",
                      self.self_check)
        self.assertIn("the objective and the quantity it actually optimises",
                      self.self_check)
        self.assertIn("feasibility of the returned solution against the "
                      "original constraints", self.self_check)

    def test_simple_task_does_not_default_to_a_full_resolve(self):
        self.assertIn("Do NOT run a complete independent re-solve as a default "
                      "step for every task", self.self_check)
        self.assertIn("This small task needs only the base check", self.skill)
        self.assertIn("do not default to a complete independent re-solve",
                      self.commands)

    def test_complex_task_may_still_add_independent_verification(self):
        self.assertIn("when a second algorithm is genuinely how a complex "
                      "problem gets a reliable answer, run it", self.self_check)
        self.assertIn("optimality cannot be confirmed from the available "
                      "evidence", self.self_check)
        self.assertIn("add an enhanced check only for a concrete reason",
                      self.commands)

    def test_model_verification_is_not_task_verification(self):
        self.assertIn("a second solver solving the same wrong model can "
                      "return the same wrong answer", self.self_check)
        self.assertIn("A second solver on the same wrong model still returns "
                      "the wrong answer", self.skill)
        self.assertIn("two solvers agree on the same wrong model",
                      self.self_check)
        self.assertIn("A solver's `optimal` proves only that the written "
                      "model was solved", self.commands)

    def test_failed_check_and_checker_error_are_not_a_pass(self):
        self.assertIn("Never mark the self-check complete or passed when a "
                      "verification script raised or an assertion failed",
                      self.self_check)
        self.assertIn("A bad check, a broken verification script or missing "
                      "evidence is corrected and re-run on the SAME execution",
                      self.self_check)
        self.assertIn("A failed or errored verification is evidence, not a "
                      "pass", self.skill)
        self.assertIn("A `script_error` or `assertion_violations` result is NOT "
                      "a pass", self.commands)
        self.assertIn("a failed check or a broken verification script is "
                      "analysed and corrected on the same execution",
                      self.closeout)

    def test_insufficient_is_neither_pass_nor_forced_resolve(self):
        self.assertIn("neither a pass nor a failure, and not by itself a "
                      "reason for a complete re-solve", self.skill)
        self.assertIn("an `insufficient` result is neither a pass nor a reason "
                      "to force a full re-solve", self.commands)

    def test_hplus_verification_is_not_a_second_task_requirement(self):
        self.assertIn("not a second task requirement", self.skill)
        self.assertIn("They are not a second task requirement", self.self_check)
        self.assertIn("never run a second complete solve merely to make it "
                      "show as checked", self.induction)
        self.assertIn("rather than running a second complete solve just to show "
                      "it as checked", self.commands)


class TestHplusJudgementPrompt(unittest.TestCase):
    """The capability_gain prompt must stop the standard-translation signal."""

    @classmethod
    def setUpClass(cls):
        from or_harness.world_model.strategy_prediction import (
            STRATEGY_OUTCOME_SYSTEM_PROMPT,
        )
        cls.prompt = _flat(STRATEGY_OUTCOME_SYSTEM_PROMPT)

    def test_the_old_translation_loophole_is_removed(self):
        self.assertNotIn("Reusable modelling knowledge also includes "
                         "translating", self.prompt)
        self.assertIn("STANDARD TASK TRANSLATION IS NOT AN INCREMENT",
                      self.prompt)

    def test_standard_modelling_steps_are_named_as_non_triggers(self):
        for example in (
                "writing a profit-maximisation objective",
                "writing a capacity constraint",
                "declaring variables integer",
                "adding a time index",
                "building an ordinary LP/MILP model",
                "running a routine constraint or result check"):
            self.assertIn(example, self.prompt)

    def test_hplus_is_not_difficulty_success_or_summarisability(self):
        self.assertIn("H+ is NOT a judgement that the task is difficult",
                      self.prompt)
        self.assertIn("that the candidate will succeed", self.prompt)
        self.assertIn("that you can summarise a trick", self.prompt)
        self.assertIn("that a reusable-sounding rule can be written",
                      self.prompt)

    def test_positive_requires_a_concrete_checkable_increment(self):
        self.assertIn("concrete reusable mechanism or boundary", self.prompt)
        self.assertIn("a structural reduction that preserves the feasible "
                      "region or the optimum", self.prompt)
        self.assertIn("can change how a future agent MODELS or EXECUTES a task",
                      self.prompt)
        self.assertIn("in ONE sentence in claim", self.prompt)
        self.assertIn("do not propose a condition that needs a later large "
                      "experiment or a second complete solve", self.prompt)
        self.assertIn("do not turn these four questions into separate fields",
                      self.prompt)

    def test_cold_start_is_neither_automatic_one_nor_automatic_zero(self):
        self.assertIn("an empty library or missing review history is NOT by "
                      "itself a reason for 0", self.prompt)
        self.assertIn("'no published rule covers it' is only necessary "
                      "background, NEVER sufficient", self.prompt)

    def test_zero_requires_a_specific_reason_and_no_invented_rule(self):
        self.assertIn("For 0 give the candidate-specific reason in basis and "
                      "OMIT claim", self.prompt)
        self.assertIn("never invent a rule ID, version or premise", self.prompt)
        self.assertIn("When the reason is that no concrete increment exists, "
                      "say so directly", self.prompt)

    def test_existing_contract_and_review_semantics_are_kept(self):
        for key in ("decision", "claim", "basis", "verification_conditions",
                    "applies_to", "expected_changes"):
            self.assertIn(key, self.prompt)
        self.assertIn("A 1 requests review, not publication", self.prompt)
        self.assertIn("A 1 ALWAYS causes one review after a bound real "
                      "attempt, even after failure or an unfinished check",
                      self.prompt)
        self.assertIn("frozen binary H+ induction switch", self.prompt)


class TestInductionReviewGuidance(unittest.TestCase):
    """The induction reference must allow and describe refusing an increment."""

    @classmethod
    def setUpClass(cls):
        cls.text = _flat(_read("references/induction.md"))

    def test_standard_translation_is_not_a_knowledge_increment(self):
        self.assertIn("NECESSARY SOLVING WORK, not a knowledge increment by "
                      "itself", self.text)
        for example in ("profit-maximisation objective",
                        "capacity constraint",
                        "declaring variables integer",
                        "time index"):
            self.assertIn(example, self.text)
        self.assertIn("That is required solving work, not a reusable "
                      "increment", self.text)

    def test_review_may_settle_without_writing_a_rule(self):
        for result in ("`new_rule`", "`revision`", "`evidence_added`",
                       "`already_covered`", "`no_new_knowledge`",
                       "`insufficient_evidence`"):
            self.assertIn(result, self.text)
        self.assertIn("`new_rule` and `revision` are never required to settle a "
                      "positive job", self.text)
        self.assertIn("it is not a preference order", self.text)
        self.assertIn("Writing no rule is a successful review", self.text)

    def test_hypothesis_outcome_stays_separate_from_the_learning_result(self):
        self.assertIn("Four cases are different", self.text)
        self.assertIn("Do not present case 4 as case 1", self.text)
        self.assertIn("never upgraded to checked to close the job", self.text)
        self.assertIn("do not invent a second hypothesis-status system",
                      self.text)

    def test_durable_rules_cannot_depend_on_hidden_benchmark_facts(self):
        for phrase in ("benchmark scores",
                       "a specific test set's reference answers",
                       "strategies that only work once the evaluation result "
                       "is known",
                       "if the benchmark reference uses domain X, use domain X",
                       "POST-HOC VERIFICATION EVIDENCE",
                       "real task input the future agent can see"):
            self.assertIn(phrase, self.text)
        self.assertIn("Do not add knowledge types such as "
                      "`benchmark_specific` or `modelling_translation`",
                      self.text)

    def test_verification_claim_matches_the_property(self):
        for phrase in ("exactly derived mathematical property",
                       "empirical success, cost or transfer claim",
                       "technical fact observed in a single run",
                       "unverified generalisation"):
            self.assertIn(phrase, self.text)
        self.assertIn("Do not demand a second complete solve for every entry",
                      self.text)

    def test_worked_flows_cover_all_three_outcomes(self):
        for marker in ("**new_rule — a specific mechanism survived review.**",
                       "**no_new_knowledge — the hypothesis dissolved.**",
                       "**insufficient_evidence — the condition was not "
                       "completed.**"):
            self.assertIn(marker, self.text)


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
