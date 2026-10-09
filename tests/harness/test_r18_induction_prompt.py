"""r18: the induction prompt states the judgement principles, MINUS the bias.

The five-problem audit found three recurring mis-judgements:

- a derivable "lower bound + attains the bound" argument was read as a one-off
  because its numbers were task-specific (0148);
- the "an empirical advantage needs multiple independent tasks" rule was
  mis-applied to a method that rests on a mathematical argument (0150);
- "no new knowledge" was recorded as a placeholder entry named
  `no_new_knowledge` instead of an empty review (0128).

The fix is prompt/doc wording only: state that novelty is relative to the
strategy LIBRARY (not to operations research), that "coverage" must compare the
specific mechanism, and that a single task can back a conditional method while
only an ADVANTAGE needs comparable measurements. These guards pin that wording
so it cannot silently regress.
"""
import os
import unittest


REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _read(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return fh.read()


def _flat(text):
    """Collapse whitespace so a phrase is matched across a line wrap."""
    return " ".join(text.split())


class TestPortablePromptFile(unittest.TestCase):
    def setUp(self):
        self.prompt = _flat(_read(os.path.join("references",
                                               "induction_prompt.md")))

    def test_prompt_names_the_real_generating_entry(self):
        self.assertIn("bench_orarla/mkmsgs_acc_patched.py", self.prompt)
        self.assertIn("_induct_new_block.py", self.prompt)
        self.assertIn("INDUCT template", self.prompt)

    def test_prompt_is_novelty_relative_to_the_library(self):
        self.assertIn("library does not already cover", self.prompt)
        self.assertIn("not that it is new to operations", self.prompt)
        # A standard method is not rejected merely for being standard.
        self.assertIn("textbook method can be worth recording", self.prompt)

    def test_prompt_distinguishes_mechanisms(self):
        self.assertIn("does not cover", self.prompt)
        for mechanism in ("boundary selection", "variable elimination",
                          "feasibility check", "optimality argument"):
            self.assertIn(mechanism, self.prompt)

    def test_prompt_keeps_three_supports_apart(self):
        self.assertIn("MATHEMATICAL", self.prompt)
        self.assertIn("OBSERVED execution", self.prompt)
        self.assertIn("EMPIRICAL advantage", self.prompt)

    def test_prompt_has_the_short_abstract_example(self):
        self.assertIn("ax+by", self.prompt)
        self.assertIn("D/a", self.prompt)
        self.assertIn("not a built-in rule", self.prompt)

    def test_prompt_forbids_the_placeholder_entry(self):
        self.assertIn("empty review", self.prompt)
        self.assertIn("do NOT create a", self.prompt)
        self.assertIn("no new knowledge", self.prompt)

    def test_prompt_excludes_engineering_trivia(self):
        self.assertIn("API names", self.prompt)
        self.assertIn("status-string", self.prompt)


class TestDocsSynced(unittest.TestCase):
    def test_skill_says_novelty_is_relative_to_the_library(self):
        skill = _flat(_read("SKILL.md"))
        self.assertIn("does not already cover the mechanism", skill)
        self.assertIn("compare the SPECIFIC premise, operation and effect",
                      skill)

    def test_induction_reference_adds_the_abstraction_example(self):
        induction = _flat(_read(os.path.join("references", "induction.md")))
        self.assertIn("Abstract the numbers into conditions", induction)
        self.assertIn("ax+by", induction)
        self.assertIn("it is not a built-in rule", induction)

    def test_induction_reference_scopes_failures_and_engineering(self):
        induction = _flat(_read(os.path.join("references", "induction.md")))
        self.assertIn("engineering facts", induction.lower())
        self.assertIn("CLUE about a boundary", induction)


if __name__ == "__main__":
    unittest.main()
