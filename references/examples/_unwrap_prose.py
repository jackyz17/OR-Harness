"""Unwrap hard-wrapped prose in Markdown documents.

Markdown does not need hard wrapping: a paragraph is one logical line. This
joins continuation lines back into their paragraph so a file's line count
reflects its structure (sections, tables, bullets, code) rather than the
author's editor width. Frontmatter, tables, headings, code fences, blockquotes
and blank lines are preserved verbatim.

    PYTHONPATH=src python3 references/examples/_unwrap_prose.py            # all docs
    PYTHONPATH=src python3 references/examples/_unwrap_prose.py SKILL.md   # one file

Content is never changed — only line breaks are removed — and the tool is
idempotent. `_check_docs.py` fails when a wrapped paragraph is reintroduced.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Documents an outer agent loads at runtime (kept in sync with the checker).
DOCS = [
    "SKILL.md", "README.md", "README_zh.md",
    "references/commands.md", "references/concepts.md",
    "references/induction.md", "references/modeling.md",
    "references/examples.md", "references/world_model_contract.md",
    "references/prediction_context.md", "references/strategy_outcome.md",
    "references/episode_closeout.md",
]


def unwrap(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    in_frontmatter = False
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append(" ".join(buffer))
            buffer.clear()

    for index, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()

        # YAML frontmatter is structural, never re-wrapped: it is delimited
        # by `---` at the very top and its indentation carries meaning.
        if index == 0 and stripped == "---":
            in_frontmatter = True
            out.append(line)
            continue
        if in_frontmatter:
            out.append(line)
            if stripped == "---":
                in_frontmatter = False
            continue

        if stripped.startswith(("```", "~~~")):
            flush()
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        if not stripped:
            flush()
            out.append("")
            continue

        if stripped.startswith(("#", "|", ">")):
            flush()
            out.append(line)
            continue

        # A list item needs its marker FOLLOWED BY SPACE: `**bold**` starts
        # with `*` but is a paragraph, not a bullet.
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s", line):
            flush()
            out.append(line)
            continue

        # A continuation line (no marker of its own) belongs to the block
        # above it: a paragraph still being accumulated, or the list item
        # directly above.
        if buffer:
            buffer.append(stripped)
        elif out and re.match(r"^\s*(?:[-*+]|\d+[.)])\s", out[-1]):
            out[-1] = out[-1] + " " + stripped
        else:
            buffer.append(stripped)

    flush()
    # Collapse 3+ blank lines to one and drop leading/trailing blanks.
    result: list[str] = []
    blanks = 0
    for line in out:
        if line == "":
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        result.append(line)
    while result and result[0] == "":
        result.pop(0)
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result) + "\n"


def main() -> int:
    targets = ([Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1
               else [ROOT / doc for doc in DOCS])
    changed = 0
    for path in targets:
        if not path.exists():
            print(f"{path}: MISSING")
            continue
        original = path.read_text(encoding="utf-8")
        updated = unwrap(original)
        if updated == original:
            print(f"{path}: already unwrapped "
                  f"({original.count(chr(10))} lines)")
            continue
        path.write_text(updated, encoding="utf-8")
        changed += 1
        print(f"{path}: {original.count(chr(10))} -> "
              f"{updated.count(chr(10))} lines")
    print(f"\n{changed} file(s) rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
