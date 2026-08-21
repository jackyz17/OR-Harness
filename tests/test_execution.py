from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from helpers import SRC_DIR
from or_experience_bank.solving.execution import SafePythonExecutor


class ExecutionTests(unittest.TestCase):
    def test_result_contract_executes_and_parses(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            code = workspace / "solve.py"
            code.write_text(
                "import json\n"
                "result={'status':'optimal','solver':'test','objective_sense':'minimize','objective_value':4,'variables':{'x':1}}\n"
                "open('result.json','w').write(json.dumps(result))\n",
                encoding="utf-8",
            )
            result = asyncio.run(SafePythonExecutor(timeout_seconds=5).execute(code, workspace, "test"))
            self.assertEqual(result.status, "optimal")
            self.assertEqual(result.objective_value, 4)

    def test_network_shell_and_parent_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            code = workspace / "unsafe.py"
            code.write_text("import socket\nopen('../other.json','w').write('x')\n", encoding="utf-8")
            result = asyncio.run(SafePythonExecutor(timeout_seconds=5).execute(code, workspace, "test"))
            self.assertEqual(result.status, "error")
            self.assertIn("security policy", result.normalized_error)

