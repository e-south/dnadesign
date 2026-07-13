"""Render the campaign-level selection batch in the notebook viewport."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def render_notebook_selection_batch_panel(
    *,
    build_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    build_summary_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    controls: Any | None,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    selected_visual_choice: Mapping[str, Any],
) -> Any:
    """Render candidate memberships and the bounded batch contract."""

    rows = build_rows(selected_visual_choice)
    summary = build_summary_rows(selected_visual_choice)
    title = str(selected_visual_choice.get("title") or "Selection batch")
    items = [
        controls,
        mo.md(f"### {title}"),
        opal_table(pl.DataFrame(rows), page_size=18),
        mo.accordion(
            {"Batch contract": opal_table(pl.DataFrame(summary), page_size=6)},
            multiple=True,
        ),
    ]
    return mo.vstack([item for item in items if item is not None], gap=0.45)


__all__ = ["render_notebook_selection_batch_panel"]
