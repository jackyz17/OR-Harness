#!/usr/bin/env python3
"""orx — OR Experience Bank CLI entry script for harness agents.

Usage:  python3 scripts/orx.py <command> [options]
Run `python3 scripts/orx.py --help` for the command list.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from or_experience_bank.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
