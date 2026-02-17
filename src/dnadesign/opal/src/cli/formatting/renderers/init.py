"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/init.py

Renders init command output for OPAL CLI. Formats campaign initialization
summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...tui import kv_table, tui_enabled
from ..core import kv_block


def render_init_human(*, workdir: Path) -> str:
    if tui_enabled():
        table = kv_table(
            "[ok] Initialized campaign workspace",
            {
                "workdir": str(Path(workdir).resolve()),
                "directories": "outputs/",
            },
        )
        if table is not None:
            return table
    return kv_block(
        "[ok] Initialized campaign workspace",
        {
            "workdir": str(Path(workdir).resolve()),
            "directories": "outputs/",
        },
    )
