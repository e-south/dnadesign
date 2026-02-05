"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_ui_rich.py

Rich table rendering tests for USR CLI output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from dnadesign.usr.src import ui


def test_render_table_rich_formats_large_numbers(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "demo",
                "rows": 56517,
                "cols": 62,
                "size": "12MB",
                "updated": "2025-11-05T19:11:07",
            }
        ]
    )
    console = Console(record=True, width=60)
    monkeypatch.setattr(ui, "_require_rich", lambda: console)

    ui.render_table_rich(df, title="USR datasets")

    text = console.export_text()
    assert "56,517" in text
    assert "62" in text
