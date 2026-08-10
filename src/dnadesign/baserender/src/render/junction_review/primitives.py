"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/primitives.py

Deterministic molecular drawing primitives for Junction review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import cache
from typing import Sequence

from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, IdentityTransform

from ..sequence_preview import bounded_svg_gid
from .foundation import DOMAIN, INK, STRAND_EDGE

_DNA_BASES = "ACGT"


@cache
def _centered_base_glyph(base: str):
    if base not in _DNA_BASES:
        raise ValueError(f"unsupported DNA base {base!r}")
    raw = TextPath((0, 0), base, prop=FontProperties(family="monospace"), size=1.0)
    bounds = raw.get_extents()
    return raw.transformed(
        Affine2D().translate(
            -bounds.x0 - bounds.width / 2,
            -bounds.y0 - bounds.height / 2,
        )
    )


def draw_segmented_strand(
    axis,
    *,
    start_x: float,
    center_y: float,
    base_step: float,
    length: int,
    segments: Sequence[tuple[int, int, str]],
    height: float,
    gid_prefix: str,
) -> None:
    """Draw one contiguous oligo with clipped categorical component spans."""

    if length < 1:
        raise ValueError("segmented strand length must be positive")
    width = base_step * length
    body = Rectangle(
        (start_x, center_y - height / 2),
        width,
        height,
        facecolor=DOMAIN,
        edgecolor="none",
        zorder=0,
    )
    body.set_gid(bounded_svg_gid(f"{gid_prefix}:body"))
    axis.add_patch(body)
    for index, (start, end, color) in enumerate(segments):
        if not 0 <= start < end <= length:
            raise ValueError(f"invalid strand segment [{start}, {end}) for length {length}")
        segment = Rectangle(
            (start_x + start * base_step, center_y - height / 2),
            (end - start) * base_step,
            height,
            facecolor=color,
            edgecolor="none",
            zorder=0.2,
        )
        segment.set_clip_path(body)
        segment.set_gid(bounded_svg_gid(f"{gid_prefix}:segment:{index}"))
        axis.add_patch(segment)
    outline = Rectangle(
        (start_x, center_y - height / 2),
        width,
        height,
        facecolor="none",
        edgecolor=STRAND_EDGE,
        linewidth=0.55,
        zorder=0.4,
    )
    outline.set_gid(bounded_svg_gid(f"{gid_prefix}:outline"))
    axis.add_patch(outline)


def draw_molecular_path(
    axis,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    gid: str,
    linewidth: float = 9.0,
    zorder: float = 0.3,
) -> None:
    """Draw one contiguous molecular path with square termini and rounded bends."""

    line = Line2D(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder,
    )
    line.set_gid(bounded_svg_gid(gid))
    axis.add_line(line)


def draw_base_run(
    axis,
    sequence: str,
    *,
    start_x: float,
    start_y: float,
    delta_x: float,
    delta_y: float,
    gid_prefix: str,
    fontsize: float,
    color: str = INK,
) -> tuple[object, ...]:
    """Draw bases at explicit centers shared with pairing geometry."""

    safe_gid_prefix = bounded_svg_gid(gid_prefix)
    artists: list[object] = []
    for index, base in enumerate(sequence):
        artist = axis.text(
            start_x + (index + 0.5) * delta_x,
            start_y + (index + 0.5) * delta_y,
            base,
            family="monospace",
            fontsize=fontsize,
            color=color,
            ha="center",
            va="center",
            zorder=3,
        )
        artist.set_gid(f"{safe_gid_prefix}:base:{index}:{base}")
        artists.append(artist)
    return tuple(artists)


def draw_compact_base_run(
    axis,
    sequence: str,
    *,
    start_x: float,
    start_y: float,
    delta_x: float,
    delta_y: float,
    gid_prefix: str,
    fontsize: float,
    color: str = INK,
) -> tuple[PathCollection, ...]:
    """Draw a long exact sequence with four bounded vector collections.

    One collection per nucleotide avoids allocating a Matplotlib text artist
    for every product base. Collection offsets remain the authoritative base
    centers, so Watson-Crick guides use the same coordinates.
    """

    invalid = sorted(set(sequence) - set(_DNA_BASES))
    if invalid:
        raise ValueError(f"base run contains unsupported DNA symbols: {invalid}")
    safe_gid_prefix = bounded_svg_gid(gid_prefix)
    artists: list[PathCollection] = []
    for base in _DNA_BASES:
        offsets = [
            (
                start_x + (index + 0.5) * delta_x,
                start_y + (index + 0.5) * delta_y,
            )
            for index, symbol in enumerate(sequence)
            if symbol == base
        ]
        if not offsets:
            continue
        artist = PathCollection(
            [_centered_base_glyph(base)],
            sizes=[fontsize**2],
            offsets=offsets,
            transOffset=axis.transData,
            facecolors=color,
            edgecolors="none",
            zorder=3,
        )
        artist.set_transform(IdentityTransform())
        artist.set_gid(f"{safe_gid_prefix}:glyph:{base}")
        axis.add_collection(artist)
        artists.append(artist)
    return tuple(artists)
