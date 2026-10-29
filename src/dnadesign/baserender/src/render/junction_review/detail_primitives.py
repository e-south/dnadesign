"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/detail_primitives.py

Drawing marks used only by nucleotide-level Junction detail panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..sequence_preview import bounded_svg_gid
from .detail_geometry import COMPONENT_WIDTH
from .foundation import INK, PAIR
from .primitives import draw_molecular_path


def add_annotation(
    axis,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float,
    color: str,
    ha: str = "center",
    va: str = "baseline",
) -> None:
    axis.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va)


def add_pairs(axis, segments, *, gid: str) -> None:
    collection = LineCollection(segments, colors=PAIR, linewidths=0.8, zorder=1)
    collection.set_gid(bounded_svg_gid(gid))
    axis.add_collection(collection)


def add_break(axis, *, x: float, y: float, gid: str) -> None:
    """Mark a cropped strand with paired slashes, not a physical terminus."""

    for index, offset in enumerate((-0.20, 0.20)):
        xs = (x + offset - 0.18, x + offset + 0.18)
        ys = (y - 0.48, y + 0.48)
        axis.add_line(Line2D(xs, ys, color="white", linewidth=4.0, zorder=5))
        mark = Line2D(xs, ys, color=INK, linewidth=1.0, zorder=6)
        mark.set_gid(bounded_svg_gid(f"{gid}:{index}"))
        axis.add_line(mark)


def add_nick(axis, *, x: float, y: float, gid: str) -> None:
    """Draw the one physical nick as a single diagonal mark."""

    xs = (x - 0.22, x + 0.22)
    ys = (y - 0.52, y + 0.52)
    axis.add_line(Line2D(xs, ys, color="white", linewidth=4.2, zorder=5))
    mark = Line2D(xs, ys, color=INK, linewidth=1.2, zorder=6)
    mark.set_gid(bounded_svg_gid(gid))
    axis.add_line(mark)


def draw_component_path(axis, xs, ys, *, color: str, gid: str) -> None:
    draw_molecular_path(
        axis,
        xs,
        ys,
        color=color,
        gid=gid,
        linewidth=COMPONENT_WIDTH,
        zorder=0.5,
    )
