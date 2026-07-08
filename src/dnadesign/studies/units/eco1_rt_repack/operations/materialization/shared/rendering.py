"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/shared/rendering.py

Shared SVG rendering helpers for Eco1 materializers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}
TITLE_SIZE = 16
LABEL_SIZE = 13.5
TICK_SIZE = 12
LEGEND_SIZE = 12


def save_accessible_svg(fig: Any, path: Path, *, title: str, description: str, dpi: int = 180) -> None:
    """Save a Matplotlib figure as an SVG with a title and description node."""

    import matplotlib.pyplot as plt
    from matplotlib import rc_context

    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, format="svg", bbox_inches="tight", dpi=dpi)
    _inject_svg_accessibility(path, title=title, description=description)
    plt.close(fig)


def _inject_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    title_id = f"{path.stem}-title"
    desc_id = f"{path.stem}-desc"
    if f'id="{title_id}"' not in text and "<svg " in text:
        svg_start = text.find("<svg ")
        svg_end = text.find(">", svg_start)
        if svg_start != -1 and svg_end != -1:
            svg_tag = text[svg_start : svg_end + 1]
            if "role=" not in svg_tag:
                svg_tag = svg_tag.replace("<svg ", '<svg role="img" ', 1)
            if "aria-labelledby=" not in svg_tag:
                svg_tag = svg_tag.replace("<svg ", f'<svg aria-labelledby="{title_id} {desc_id}" ', 1)
            accessible = (
                f'\n<title id="{escape(title_id)}">{escape(title)}</title>'
                f'\n<desc id="{escape(desc_id)}">{escape(description)}</desc>'
            )
            text = text[:svg_start] + svg_tag + accessible + text[svg_end + 1 :]
    path.write_text(text, encoding="utf-8")


def shorten_label(label: str, *, max_length: int = 38) -> str:
    """Return a compact label for crowded visual axes."""

    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def style_open_axes(ax: Any, *, grid: bool = True) -> None:
    """Apply the Eco1 review visual style for simple Matplotlib axes."""

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(color="#d0d7de", alpha=0.46, linewidth=0.7)
