"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/metadata.py

Shared system-of-record metadata contracts for documentation tooling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

SOR_MARKDOWN_FILES = (
    "ARCHITECTURE.md",
    "DESIGN.md",
    "SECURITY.md",
    "RELIABILITY.md",
    "PLANS.md",
    "QUALITY_SCORE.md",
)
OWNER_PATTERN = re.compile(r"^\*\*Owner:\*\*\s*(.+?)\s*$", re.MULTILINE)
LAST_VERIFIED_PATTERN = re.compile(r"^\*\*Last verified:\*\*\s*(.+?)\s*$", re.MULTILINE)

__all__ = ["LAST_VERIFIED_PATTERN", "OWNER_PATTERN", "SOR_MARKDOWN_FILES"]
