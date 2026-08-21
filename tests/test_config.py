from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import SKILL_DIR
from or_experience_bank.config import ExperienceBankConfig


class ConfigTests(unittest.TestCase):
    def test_cli_environment_file_default_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.yaml"
            config_path.write_text(
                "bank_home: /tmp/from-file\n"
                "orchestration:\n  max_attempts_per_branch: 2\n  solvers:\n    - scip\n"
                "retrieval:\n  backend: local\n  top_k:\n    modeling: 4\n    implementation: 3\n    repair: 2\n    solving: 1\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OR_EXPERIENCE_MAX_ATTEMPTS": "4", "OR_EXPERIENCE_BANK_HOME": str(Path(temp) / "env-home")}, clear=False):
                config = ExperienceBankConfig.load(str(config_path), {"max_attempts_per_branch": 5})
            self.assertEqual(config.max_attempts_per_branch, 5)
            self.assertEqual(config.bank_home, (Path(temp) / "env-home").resolve())
            self.assertEqual(config.solvers, ["scip"])
            self.assertEqual(config.top_k["modeling"], 4)

