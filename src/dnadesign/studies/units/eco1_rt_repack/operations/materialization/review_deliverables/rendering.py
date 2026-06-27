"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/rendering.py

Shared SVG rendering helpers for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from matplotlib import rc_context


def save_accessible_svg(fig: Any, path: Path, *, title: str, description: str) -> None:
    """Save a Matplotlib figure as an SVG with a title and description node."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, format="svg", bbox_inches="tight")
    fig.clear()
    _inject_svg_accessibility(path, title=title, description=description)


def _inject_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    title_id = f"{path.stem}-title"
    desc_id = f"{path.stem}-desc"
    if "<title" not in text and "<svg " in text:
        text = text.replace("<svg ", f'<svg role="img" aria-labelledby="{title_id} {desc_id}" ', 1)
        svg_start = text.find("<svg ")
        svg_end = text.find(">", svg_start)
        if svg_start != -1 and svg_end != -1:
            accessible = (
                f'\n<title id="{escape(title_id)}">{escape(title)}</title>'
                f'\n<desc id="{escape(desc_id)}">{escape(description)}</desc>'
            )
            text = text[: svg_end + 1] + accessible + text[svg_end + 1 :]
    path.write_text(text, encoding="utf-8")


def shorten_label(label: str, *, max_length: int = 38) -> str:
    """Return a compact label for crowded visual axes."""

    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."
