"""r17: H+ follow-up timing/ownership is documented, not "optional".

The H+ (capability-gain) delayed-supervision loop had `bind-capability` /
`evaluate-capability` listed as OPTIONAL and never placed in the per-episode
workflow, so it was never triggered (observed: fact_bound=0,
effect_verified=0, every claim stuck pending). These guards pin the fix:

- the agent entry doc (SKILL.md) shows "H+ follow-up = close-out stage-1 +
  delayed stage-2" with its WHEN and WHO;
- commands.md no longer classes `bind-capability`/`evaluate-capability` under
  "Optional maintenance forecasts";
- the THREE states `state=bound` / `fact_bound` / `effect_verified` are
  distinguished, and "pending != failure" / "never re-predict" are stated.
"""
import os
import re
import unittest


REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _read(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return fh.read()


class TestSkillDocumentsTheHPlusFollowup(unittest.TestCase):
    def test_skill_has_a_dedicated_hplus_followup_section(self):
        skill = _read("SKILL.md")
        self.assertIn("### Frozen H+ and the building handoff", skill)
        self.assertIn("process_induction_jobs", skill)
        self.assertIn("same", skill.lower())
        self.assertIn("--induction-id", skill)

    def test_receipt_binding_follows_induction_not_solving(self):
        skill = _read("SKILL.md")
        self.assertIn("Solving alone is not a maintenance fact", skill)
        self.assertIn("Completed induction automatically binds the REAL induction action", skill)
        self.assertIn("Predicted-but-unexecuted candidates stay unobserved", skill)
        self.assertIn("Do not declare library building finished with pending jobs", skill)

    def test_skill_separates_hypothesis_increment_and_legacy_effect(self):
        skill = _read("SKILL.md")
        self.assertIn("Keep hypothesis truth separate", skill)
        self.assertIn("independently validated substantive increment", skill)
        self.assertIn("legacy `evaluate-capability`", skill)
        self.assertIn("It never changes G/C/R strategy ranking", skill)


class TestCommandsDoNotCallTheLoopOptional(unittest.TestCase):
    def test_bind_and_evaluate_are_out_of_the_optional_row(self):
        commands = _read("references/commands.md")
        row = re.compile(r"^\|\s*Optional maintenance forecasts\s*\|([^|]*)"
                         r"\|", re.MULTILINE)
        match = row.search(commands)
        self.assertIsNotNone(match,
                             "the optional-forecasts row must still exist")
        optional_commands = match.group(1)
        self.assertNotIn("bind-capability", optional_commands)
        self.assertNotIn("evaluate-capability", optional_commands)
        # The genuinely optional ones stay.
        for name in ("predict-capability", "compare-capability",
                     "accept-capability", "reject-capability"):
            self.assertIn(name, optional_commands)

    def test_a_dedicated_hplus_followup_row_exists(self):
        commands = _read("references/commands.md")
        self.assertIn("| H+ building handoff |", commands)
        self.assertIn("Consume every frozen positive job", commands)
        self.assertIn("induction-status", commands)

    def test_each_command_entry_names_its_trigger(self):
        commands = _read("references/commands.md")
        self.assertIn("**When (stage-1):** after the REAL induction completes", commands)
        self.assertIn("**When (stage-2):**", commands)
        self.assertIn("An executed solve alone is not an induction fact", commands)


if __name__ == "__main__":
    unittest.main()
