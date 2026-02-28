"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_plain_language_contracts.py

Docs language contracts for plain terminology in Cruncher markdown docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BANNED = re.compile(r"canonical", flags=re.IGNORECASE)


def test_cruncher_docs_and_workspaces_avoid_canonical_wording() -> None:
    markdown_files = sorted((ROOT / "docs").rglob("*.md"))
    markdown_files.extend(sorted((ROOT / "workspaces").rglob("*.md")))
    assert markdown_files, "Expected markdown docs under cruncher/docs and cruncher/workspaces"
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        match = BANNED.search(content)
        assert match is None, f"{path}: banned wording '{match.group(0)}'"
