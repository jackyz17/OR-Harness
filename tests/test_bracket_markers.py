"""Square-bracket marker tests ([THINK]/[MODEL]) — the harness-safe format.

Root cause (2026-08-26): Hermes-class harnesses reserve <think> as their own
reasoning-channel marker and strip/consume it before agent text reaches
model.txt, so angle-bracket tags could never survive the trip. The parser now
accepts square-bracket markers as the PREFERRED syntax (angle brackets remain
accepted for backward compatibility).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ORX = [sys.executable, str(REPO / "scripts" / "orx.py")]

PROBLEM = """A farmer raises cows and chickens.
Each cow yields 40 profit, each chicken 10 profit.
The farmer has at most 200 units of feed; a cow needs 8 units, a chicken 2.
At most 20 chickens may be raised.
How many of each should be raised to maximize profit?"""

MODEL_BODY = """SETS:
  i in Animals = {cow, chicken}
PARAMETERS:
  profit[i]
  feed_need[i]
  feed_limit
  max_chickens
VARIABLES:
  x[i] integer >= 0
OBJECTIVE:
  maximize sum(i, profit[i] * x[i])
CONSTRAINTS:
  C1: sum(i, feed_need[i] * x[i]) <= feed_limit
  C2: x[chicken] <= max_chickens"""

BRACKET_MODEL = "[THINK]\nResource-allocation LP: two decisions share one feed budget.\n[/THINK]\n[MODEL]\n" + MODEL_BODY + "\n[/MODEL]"
XML_MODEL = "<think>\nResource-allocation LP: two decisions share one feed budget.\n</think>\n<model>\n" + MODEL_BODY + "\n</model>"
LOWER_BRACKET_MODEL = BRACKET_MODEL.lower()


class TestParserMarkers(unittest.TestCase):
    def setUp(self):
        from or_experience_bank.modeling.modeling_contract import parse_modeling_output
        self.parse = parse_modeling_output

    def test_square_brackets_parsed(self):
        out = self.parse(BRACKET_MODEL)
        self.assertIsNotNone(out["think"])
        self.assertIsNotNone(out["model"])
        self.assertIn("feed budget", out["think"])
        self.assertIn("SETS:", out["model"])

    def test_angle_brackets_still_parsed(self):
        out = self.parse(XML_MODEL)
        self.assertIsNotNone(out["think"])
        self.assertIsNotNone(out["model"])

    def test_lowercase_brackets_parsed(self):
        out = self.parse(LOWER_BRACKET_MODEL)
        self.assertIsNotNone(out["think"])
        self.assertIsNotNone(out["model"])

    def test_bracket_precedence_over_xml(self):
        both = BRACKET_MODEL + "\n" + XML_MODEL
        out = self.parse(both)
        # square-bracket bodies win when both syntaxes appear
        self.assertIn("feed budget", out["think"])

    def test_no_markers_returns_none(self):
        out = self.parse("just plain text, no markers at all")
        self.assertIsNone(out["think"])
        self.assertIsNone(out["model"])


def run_orx(args, cwd, bank_home):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


class TestValidateAcceptsBracketFormat(unittest.TestCase):
    """The full `orx validate` gate must pass square-bracket model.txt."""

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_bracket_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        run_orx(["recall", "--problem-file", "problem.txt"], self.run_dir, self.bank)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bracket_model_passes_validate(self):
        (self.run_dir / "model.txt").write_text(BRACKET_MODEL, encoding="utf-8")
        code, out = run_orx(["validate"], self.run_dir, self.bank)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["passed"], out)

    def test_xml_model_still_passes_validate(self):
        (self.run_dir / "model.txt").write_text(XML_MODEL, encoding="utf-8")
        code, out = run_orx(["validate"], self.run_dir, self.bank)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["passed"], out)


if __name__ == "__main__":
    unittest.main()
