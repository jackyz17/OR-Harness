"""Tests for the harness-mode StdinLLMClient (the agent IS the LLM, D18)."""

from __future__ import annotations

import asyncio
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.llm_client import StdinLLMClient


def run_answer(client, prompt, response_format, answer_lines, end_marker="<<<END_LLM>>>"):
    """Drive one ask() with canned stdin lines; returns (answer, stderr_output)."""
    stdin = io.StringIO("\n".join(answer_lines) + "\n" + end_marker + "\n")
    stderr = io.StringIO()
    with mock.patch("sys.stdin", stdin), mock.patch("sys.stderr", stderr):
        if response_format == "text":
            answer = asyncio.run(client.generate_text(prompt))
        else:
            answer = asyncio.run(client.generate_object(prompt))
    return answer, stderr.getvalue()


class StdinLLMClientTest(unittest.TestCase):
    def test_text_prompt_roundtrip(self):
        client = StdinLLMClient()
        answer, stderr = run_answer(
            client, "model this problem", "text",
            ["<think>analysis</think>", "<model>SETS ...</model>"],
        )
        self.assertIn("<think>analysis</think>", answer)
        self.assertIn("<model>SETS", answer)
        # prompt goes to stderr so stdout stays clean for --json output
        self.assertIn("LLM PROMPT", stderr)
        self.assertIn("model this problem", stderr)

    def test_object_prompt_parses_json(self):
        client = StdinLLMClient()
        answer, _ = run_answer(
            client, "signature please", "json",
            ['{"objective": "linear", "decision": [], "constraint": [], "interaction": "independent"}'],
        )
        self.assertIsInstance(answer, dict)
        self.assertEqual(answer["objective"], "linear")

    def test_empty_answer_yields_defaults(self):
        client = StdinLLMClient()
        text_answer, _ = run_answer(client, "p", "text", [])
        self.assertEqual(text_answer, "")
        obj_answer, _ = run_answer(client, "p", "json", [])
        self.assertEqual(obj_answer, [])

    def test_eof_without_marker_still_returns(self):
        # a harness that answers without the end marker (EOF) still gets its lines back
        client = StdinLLMClient()
        stdin = io.StringIO("just one line\n")
        stderr = io.StringIO()
        with mock.patch("sys.stdin", stdin), mock.patch("sys.stderr", stderr):
            answer = asyncio.run(client.generate_text("p"))
        self.assertEqual(answer, "just one line")

    def test_end_marker_required(self):
        with self.assertRaises(ValueError):
            StdinLLMClient(end_marker="")


if __name__ == "__main__":
    unittest.main()
