"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli_output.py

Shared helpers for stable CLI-output assertions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def normalized_cli_output(output: str) -> str:
    """Strip terminal styling and collapse whitespace for stable CLI assertions."""

    return " ".join(_ANSI_ESCAPE_RE.sub("", output).split())


__all__ = ["normalized_cli_output"]
