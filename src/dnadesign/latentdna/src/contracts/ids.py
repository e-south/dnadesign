"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/ids.py

Identifier helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from .errors import WorkspaceValidationError

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_identifier(value: str, *, label: str) -> str:
    if not IDENTIFIER_PATTERN.match(value):
        raise WorkspaceValidationError(f"{label} must use lowercase snake_case and match [a-z][a-z0-9_]*: {value!r}")
    return value
