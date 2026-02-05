"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/ui.py

Rendering helpers for CLI tables and diff views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .diff import DiffSummary


def _rstrip_block(s: str) -> str:
    # Ensure per-line rstrip to avoid ragged right margins in terminals
    return "\n".join(ln.rstrip() for ln in s.splitlines())


def print_df_plain(df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 200)
    block = df.to_string(index=False)
    print(_rstrip_block(block))


# ---------- Rich helpers ----------


def _require_rich() -> Any:
    try:
        from rich.console import Console

        return Console()
    except ImportError as e:
        raise RuntimeError("Rich requested but not available. Install 'rich' or pass --no-rich.") from e


def _align_for_dtype(dtype: Any) -> str:
    try:
        if pd.api.types.is_numeric_dtype(dtype):
            return "right"
    except (TypeError, ValueError):
        pass
    return "left"


def render_table_rich(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    max_colwidth: int = 80,
) -> None:
    console = _require_rich()
    import pandas as _pd
    from rich import box
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title=title,
        caption=caption,
        show_lines=False,
        expand=False,
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        caption_style="italic dim",
    )
    formatters: dict[str, Callable[[object], str]] = {}
    min_widths: dict[str, int] = {}
    numeric_cols: set[str] = set()

    for name in df.columns:
        dtype = df[name].dtype
        is_numeric = False
        try:
            is_numeric = _pd.api.types.is_numeric_dtype(dtype)
        except (TypeError, ValueError):
            is_numeric = False

        if is_numeric:
            numeric_cols.add(str(name))

            def _fmt(v: object) -> str:
                if v is None:
                    return ""
                if isinstance(v, bool):
                    return str(v)
                if isinstance(v, Integral):
                    return f"{int(v):,}"
                return str(v)

            fmt = _fmt
        else:

            def _fmt_text(v: object) -> str:
                return "" if v is None else str(v)

            fmt = _fmt_text

        formatters[str(name)] = fmt
        if is_numeric:
            formatted = [fmt(v) for v in df[name].tolist()]
            min_widths[str(name)] = max(len(str(name)), *(len(v) for v in formatted))

    for name in df.columns:
        dtype = df[name].dtype
        align = _align_for_dtype(dtype)
        # dtype-aware column coloring (subtle defaults)
        col_style = "magenta"
        try:
            if _pd.api.types.is_bool_dtype(dtype):
                col_style = "yellow"
            elif _pd.api.types.is_numeric_dtype(dtype):
                col_style = "bright_cyan"
            else:
                col_style = "magenta"
        except (TypeError, ValueError):
            pass
        col_name = str(name)
        overflow = "ellipsis" if col_name in numeric_cols else "fold"
        min_width = min_widths.get(col_name)
        table.add_column(
            col_name,
            justify=align,
            no_wrap=True,
            overflow=overflow,
            min_width=min_width,
            style=col_style,
        )
    for _, row in df.iterrows():
        cells = []
        for name, v in zip(df.columns, row.tolist()):
            s = formatters[str(name)](v)
            # cap visual noise
            if len(s) > max_colwidth:
                s = s[: max_colwidth - 1] + "…"
            cells.append(Text(s))
        table.add_row(*cells)
    console.print(table)


def render_schema_tree_rich(tree_lines: Iterable[str], title: Optional[str] = None) -> None:
    console = _require_rich()
    from rich.tree import Tree

    it = iter(tree_lines)
    try:
        first = next(it)
    except StopIteration:
        return
    root = Tree(title or "schema")
    # The render_schema_tree emits one field per line with indentation via spaces.
    # Convert into a Rich tree by inspecting indent level.
    stack: list[tuple[int, Any]] = [(0, root)]

    def level_of(s: str) -> int:
        # two spaces per indent in pretty.render_schema_tree
        return (len(s) - len(s.lstrip())) // 2

    def add_line(s: str) -> None:
        lvl = level_of(s)
        label = s.strip()
        # unwind stack to current level
        while stack and stack[-1][0] >= lvl + 1:
            stack.pop()
        parent = stack[-1][1]
        node = parent.add(label)
        stack.append((lvl + 1, node))

    add_line(first)
    for line in it:
        add_line(line)
    console.print(root)


def render_diff_rich(s: DiffSummary) -> None:
    console = _require_rich()
    from rich import box
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.text import Text

    def fmt_sz(n: Optional[int]) -> str:
        if n is None:
            return "?"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        x = float(n)
        while x >= 1024 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.0f}{units[i]}"

    pl, pr = s.primary_local, s.primary_remote
    left = f"sha: {pl.sha256 or '?'}\nsize: {fmt_sz(pl.size)}\nrows: {pl.rows or '?'}\ncols: {pl.cols or '?'}\n"
    right = f"sha: {pr.sha256 or '?'}\nsize: {fmt_sz(pr.size)}\nrows: {pr.rows or '?'}\ncols: {pr.cols or '?'}\n"
    status = Text("CHANGES" if s.has_change else "up-to-date")
    status.stylize("bold red" if s.has_change else "bold green")
    cols = Columns(
        [
            Panel(left, title="Local", box=box.ROUNDED, border_style="dim"),
            Panel(right, title="Remote", box=box.ROUNDED, border_style="dim"),
            Panel(status, title="Status", box=box.ROUNDED, border_style="cyan"),
        ],
        equal=True,
        expand=False,
    )
    console.print(Text(f"Dataset: {s.dataset}", style="bold"))
    console.print(cols)
    console.print(
        f"meta.md mtime: {s.meta_local_mtime or '-'} → {s.meta_remote_mtime or '-'}\n"
        f"verify: {s.verify_mode}\n"
        f".events.log: local={s.events_local_lines} remote={s.events_remote_lines} (+{max(0, s.events_remote_lines - s.events_local_lines)})\n"  # noqa
        f"_snapshots: remote_count={s.snapshots.count} newer_than_local={s.snapshots.newer_than_local}"
    )
