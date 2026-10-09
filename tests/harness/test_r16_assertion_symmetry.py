"""r16: assertion/claim decoupling, additive-only help, and the shape guard.

What this file asserts, one behaviour per test class:

A. the assertion vocabulary is SYMMETRIC and a refutation names the ASSERTION,
   not the natural-language claim: ``code_changed`` verifies a change and
   ``code_unchanged`` refutes the same evidence, each conclusion saying which
   of the two (assertion vs claim) the verdict is about;
B. ``orx induce --help`` describes what the code DOES: a submission ALWAYS
   creates a new numbered entry (no "REVISES"), and publication is on the
   claim's own verification rather than a >=2-independent-tasks count;
C. the agent's documents state the pre-publication value gate is the agent's,
   and that the empty review is the only way to record "no new knowledge";
E. the mechanical SHAPE guard refuses a placeholder claim or an empty method
   (without writing), while a real claim — or a claim with no method field at
   all — still saves.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import StrategicEntry  # noqa: E402
from or_harness.strategy.induction import InductionEngine  # noqa: E402
from or_harness.strategy.experience_bank import ExperienceBank  # noqa: E402
from or_harness.strategy.stats import ConditionalStats  # noqa: E402
from or_harness.strategy.strategic_bank import StrategicBank  # noqa: E402
from or_harness.strategy.verification import (  # noqa: E402
    ASSERTION_CODE_CHANGED,
    INSUFFICIENT,
    REFUTED,
    VERIFIED,
    verify_relation,
)

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _rec(execution_id, *, code_hash="abc123", status="optimal"):
    record = {
        "execution_id": execution_id, "task_id": "t",
        "strategy_id": "S", "measurement_scope": "attempt",
        "quality": {"status": status, "feasible": True, "gap": 0.0,
                    "objective": 10.0},
        "cost": {"measured": ["llm_tokens"], "llm_tokens": 10},
    }
    if code_hash is not None:
        record["solver"] = {"name": "cbc", "code_hash": code_hash}
    else:
        record["solver"] = {"name": "cbc"}
    return record


# ---------------------------------------------------------------------------
# A. the symmetric assertion and the assertion/claim decoupling
# ---------------------------------------------------------------------------


class TestAssertionSymmetric(unittest.TestCase):
    def _evidence(self):
        # The acceptance example: a continuous LP run and an integer MILP run
        # carry DIFFERENT code hashes.
        return [_rec("ex_cont", code_hash="lp_pulp_cbc"),
                _rec("ex_int", code_hash="milp_pulp_cbc")]

    def _roles(self):
        return [{"execution_id": "ex_cont", "role": "cont"},
                {"execution_id": "ex_int", "role": "int"}]

    def test_code_changed_verifies_a_change(self):
        report = verify_relation(
            "the formulation was changed from continuous to integer",
            evidence=self._evidence(), roles=self._roles(),
            assertions=[{"kind": ASSERTION_CODE_CHANGED,
                         "roles": ["cont", "int"]}])
        self.assertEqual(report["state"], VERIFIED)

    def test_code_unchanged_refutes_the_same_evidence_and_names_the_assertion(
            self):
        report = verify_relation(
            "the formulation was changed from continuous to integer",
            evidence=self._evidence(), roles=self._roles(),
            assertions=[{"kind": "code_unchanged", "roles": ["cont", "int"]}])
        self.assertEqual(report["state"], REFUTED)
        # The wording separates the two: the ASSERTION failed, the CLAIM may
        # still be true (the wrong assertion was chosen).
        self.assertIn("refutes the ASSERTION", report["conclusion"])
        self.assertIn("not necessarily the natural-language claim",
                      report["conclusion"])

    def test_code_changed_same_hash_refutes(self):
        report = verify_relation(
            "the code changed", evidence=[_rec("a", code_hash="same"),
                                          _rec("b", code_hash="same")],
            roles=[{"execution_id": "a", "role": "x"},
                   {"execution_id": "b", "role": "x"}],
            assertions=[{"kind": ASSERTION_CODE_CHANGED, "roles": ["x"]}])
        self.assertEqual(report["state"], REFUTED)
        self.assertIn("SAME code", report["conclusion"])

    def test_code_changed_missing_hash_is_insufficient(self):
        report = verify_relation(
            "the code changed",
            evidence=[_rec("a", code_hash=None), _rec("b", code_hash="h2")],
            roles=[{"execution_id": "a", "role": "x"},
                   {"execution_id": "b", "role": "y"}],
            assertions=[{"kind": ASSERTION_CODE_CHANGED, "roles": ["x", "y"]}])
        self.assertEqual(report["state"], INSUFFICIENT)


# ---------------------------------------------------------------------------
# B. the help text matches the implementation
# ---------------------------------------------------------------------------


class TestHelpMatchesImplementation(unittest.TestCase):
    def _induce_help(self):
        import io
        import contextlib
        from or_harness import cli
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit):
                cli.main(["induce", "--help"])
        # The help wraps at the terminal width: collapse the whitespace so a
        # phrase is matched across a line break.
        return " ".join(buffer.getvalue().split())

    def test_no_revises_claim(self):
        help_text = self._induce_help()
        self.assertNotIn("REVISES", help_text)
        # It says the truth instead: a submission ALWAYS creates a NEW entry.
        self.assertIn("ALWAYS creates a NEW numbered entry", help_text)

    def test_publication_is_not_a_task_count(self):
        help_text = self._induce_help()
        self.assertNotIn(">=2 independent tasks", help_text)
        self.assertIn("NOT on a task count", help_text)


# ---------------------------------------------------------------------------
# C. the documents state the agent-owned value gate
# ---------------------------------------------------------------------------


class TestDocsStateTheValueGate(unittest.TestCase):
    def test_skill_names_the_agent_value_gate(self):
        with open(os.path.join(REPO, "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Before publishing you must self-justify", text)
        self.assertIn("do **not** create a \"no new knowledge", text)
        self.assertIn("it is **not** a verdict on the knowledge's validity",
                      text)

    def test_induction_reference_separates_assertion_from_claim(self):
        with open(os.path.join(REPO, "references", "induction.md"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("describe **your DECLARED assertion only**", text)
        self.assertIn("first self-check that the assertion MATCHES the claim",
                      text)
        self.assertIn("`code_changed`", text)

    def test_empty_review_is_named_the_only_correct_form(self):
        with open(os.path.join(REPO, "references", "induction.md"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("this empty review is the **ONLY** correct form", text)


# ---------------------------------------------------------------------------
# E. the mechanical SHAPE guard
# ---------------------------------------------------------------------------


class ClaimCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.engine = InductionEngine(self.stats, self.sbank)
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _seed(self):
        for index, eid in enumerate(("ex_1", "ex_2"), start=1):
            record = self.make_record(
                execution_id=eid, task_id=f"T{index}",
                strategy_id="S01", status="optimal", gap=0.0,
                feasible=True)
            self.bank.append(record)


class TestShapeGuard(ClaimCase):
    def test_placeholder_claim_is_refused_without_writing(self):
        self._seed()
        out = self.engine.submit_relation({
            "subject": "adopt:existing",
            "claim": "no new knowledge; existing entries cover this task",
            "evidence": [{"execution_id": "ex_1", "role": "a"},
                         {"execution_id": "ex_2", "role": "b"}]})
        self.assertIsNone(out["saved"])
        self.assertIn("PLACEHOLDER", out["skipped"])
        self.assertEqual(self.sbank.count(), 0)
        self.assertEqual(out["shape_guard"], "placeholder_or_empty_method")

    def test_empty_method_steps_is_refused(self):
        self._seed()
        out = self.engine.submit_relation({
            "subject": "method:empty",
            "claim": "a claim that declares a method but states no steps",
            "method": {"name": "unnamed", "steps": []},
            "evidence": [{"execution_id": "ex_1", "role": "a"},
                         {"execution_id": "ex_2", "role": "b"}]})
        self.assertIsNone(out["saved"])
        self.assertIn("EMPTY method", out["skipped"])
        self.assertEqual(self.sbank.count(), 0)

    def test_a_real_claim_still_saves(self):
        self._seed()
        out = self.engine.submit_relation({
            "subject": "method:enumerate_bound",
            "claim": ("derive a finite Y bound, then enumerate integer Y and "
                      "solve the X subproblem on each feasible interval"),
            "method": {"name": "bound_then_enumerate",
                       "steps": ["derive the Y bound",
                                 "enumerate feasible Y"]},
            "evidence": [{"execution_id": "ex_1", "role": "a"},
                         {"execution_id": "ex_2", "role": "b"}]})
        self.assertIsNotNone(out["saved"])
        self.assertEqual(self.sbank.count(), 1)

    def test_a_claim_without_a_method_field_still_saves(self):
        """A conditional fact stated in prose declares no method field: that
        is not an empty method, so the SHAPE guard must not touch it."""
        self._seed()
        out = self.engine.submit_relation({
            "subject": "principle:keep_state",
            "claim": ("on temporally coupled scheduling, keep the cross-period "
                      "state; dropping it caused infeasibility"),
            "evidence": [{"execution_id": "ex_1", "role": "dropped"},
                         {"execution_id": "ex_2", "role": "preserved"}]})
        self.assertIsNotNone(out["saved"])

    def test_placeholder_phrase_in_subject_is_not_a_match(self):
        """The SUBJECT names the technique: a subject containing a marker word
        must not be mistaken for a placeholder claim."""
        self._seed()
        out = self.engine.submit_relation({
            "subject": "principle:keep_existing_state",
            "claim": ("retain the boundary-state variables so the inter-window "
                      "balance holds across periods"),
            "evidence": [{"execution_id": "ex_1", "role": "before"},
                         {"execution_id": "ex_2", "role": "after"}]})
        self.assertIsNotNone(out["saved"])


if __name__ == "__main__":
    unittest.main()
