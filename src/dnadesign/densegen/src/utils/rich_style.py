"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/utils/rich_style.py

Shared Rich styling helpers for DenseGen CLI output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table

ACCENT = "cyan"
ACCENT_ALT = "green"
HEADER_STYLE = "bold cyan"
TITLE_STYLE = "bold green"
CAPTION_STYLE = "dim"
BOX_STYLE = box.ROUNDED


def make_table(*columns: str, **kwargs) -> Table:
    table = Table(*columns, box=BOX_STYLE, header_style=HEADER_STYLE, **kwargs)
    if getattr(table, "title_style", None) is None:
        table.title_style = TITLE_STYLE
    if getattr(table, "caption_style", None) is None:
        table.caption_style = CAPTION_STYLE
    return table


def make_panel(renderable, *, title: str, border_style: str = ACCENT) -> Panel:
    return Panel(renderable, title=title, border_style=border_style, box=BOX_STYLE)
