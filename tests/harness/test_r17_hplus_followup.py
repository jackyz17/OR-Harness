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
        self.assertIn("### H+ follow-up (after close-out)", skill)
        self.assertIn("delayed supervision", skill)
        # TWO stages at TWO times, never merged.
        self.assertIn("stage-1 `bind-capability", skill)
        self.assertIn("stage-2 `evaluate-capability", skill)

    def test_stage1_is_bound_at_close_out_and_stage2_is_delayed(self):
        skill = _read("SKILL.md")
        self.assertIn("do it right at close-out", skill)
        self.assertIn("delayed, bounded, rolling", skill)
        # An unexecuted candidate stays pending and is not bound.
        self.assertIn("not executed stays `pending`", skill)
        # Horizon-not-reached is pending, not a failure.
        self.assertIn("the evaluation stays **`pending` automatically**",
                      skill)
        self.assertIn("**NOT a failure**", skill)
        # Only observed_improvement verifies.
        self.assertIn("Only an `observed_improvement`", skill)

    def test_skill_separates_the_three_states(self):
        skill = _read("SKILL.md")
        self.assertIn("`state=bound`", skill)
        self.assertIn("`fact_bound`", skill)
        self.assertIn("`effect_verified`", skill)
        self.assertIn("Never re-predict", skill)
        self.assertIn("never ranks candidates", skill)


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
        self.assertIn("| H+ follow-up (delayed supervision) |", commands)
        self.assertIn("stage-1 `bind-capability` runs right after close-out",
                      commands)
        self.assertIn("stage-2 `evaluate-capability` runs later", commands)

    def test_each_command_entry_names_its_trigger(self):
        commands = _read("references/commands.md")
        self.assertIn("**When (stage-1):**", commands)
        self.assertIn("**When (stage-2):**", commands)
        self.assertIn("NOT inside `close-episode`", commands)


if __name__ == "__main__":
    unittest.main()
