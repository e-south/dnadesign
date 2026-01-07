"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/tui.py

Rich-based TUI helpers for CLI output and progress bars.
Kept optional and gated by OPAL_CLI_RICH, OPAL_CLI_TUI, and TTY checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

from ..core.pretty import console_err, console_out
from ..core.progress import NullProgress, ProgressFactory, ProgressTracker

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val == "":
        return default
    return val in _TRUTHY


def tui_enabled() -> bool:
    if not _truthy_env("OPAL_CLI_TUI", True):
        return False
    if console_out() is None:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def progress_enabled() -> bool:
    if not _truthy_env("OPAL_CLI_TUI", True):
        return False
    if console_err() is None:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def kv_table(title: str, items: Mapping[str, Any]):
    if not tui_enabled():
        return None
    from rich import box
    from rich.table import Table

    table = Table(title=title, show_header=False, box=box.ASCII)
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for k, v in items.items():
        table.add_row(str(k), "" if v is None else str(v))
    return table


def list_table(title: str, rows: Sequence[str]):
    if not tui_enabled():
        return None
    from rich import box
    from rich.table import Table

    table = Table(title=title, show_header=False, box=box.ASCII)
    table.add_column("Item", overflow="fold")
    if not rows:
        table.add_row("(none)")
    else:
        for r in rows:
            table.add_row(str(r))
    return table


class RichProgress(ProgressTracker):
    def __init__(self, description: str, total: int, *, console):
        self._desc = description
        self._total = int(total)
        self._console = console
        self._progress = None
        self._task_id = None

    def __enter__(self) -> "RichProgress":
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
        )
        self._task_id = self._progress.add_task(self._desc, total=self._total)
        self._progress.start()
        return self

    def advance(self, n: int = 1) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.advance(self._task_id, int(n))

    def close(self) -> None:
        if self._progress is None:
            return
        try:
            self._progress.stop()
        finally:
            self._progress = None
            self._task_id = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def progress_factory() -> ProgressFactory | None:
    if not progress_enabled():
        return None
    console = console_err()

    def _factory(desc: str, total: int):
        if console is None:
            return NullProgress()
        return RichProgress(desc, total, console=console)

    return _factory
