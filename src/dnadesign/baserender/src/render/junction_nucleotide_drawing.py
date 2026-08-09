"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_nucleotide_drawing.py

Shared nucleotide drawing primitives for Junction review figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from ..core import RenderingError
from .junction_pairing_layout import BASES_PER_ROW, sequence_chunks

INK = "#172033"
MUTED = "#667085"
PAIR = "#CBD1D9"
FRAGMENT_A = "#EEF1F5"
FRAGMENT_B = "#E3E8EF"
TOEHOLD = "#5B8DEF"
BARCODE = "#2A9D8F"
PRIMER = "#D97706"

SEQUENCE_X = 0.105
SEQUENCE_WIDTH = 0.835
BASE_FONT_SIZE = 5.6


@dataclass(frozen=True, slots=True)
class ColoredSpan:
    """One zero-based half-open display span."""

    start: int
    end: int
    color: str


def base_x(index: int) -> float:
    return SEQUENCE_X + (index * SEQUENCE_WIDTH / BASES_PER_ROW)


def spaced(sequence: str) -> str:
    return " ".join(sequence)


def _pair_edges(axis, *, start: int, length: int, top_y: float, bottom_y: float) -> None:
    segments = [((base_x(index), top_y), (base_x(index), bottom_y)) for index in range(start, start + length)]
    axis.add_collection(LineCollection(segments, colors=PAIR, linewidths=0.42, zorder=1))


def _span_bars(axis, *, spans: tuple[ColoredSpan, ...], offset: int, y: float) -> None:
    for span in spans:
        if span.start < 0 or span.end <= span.start:
            raise RenderingError("three_way_junction_review received an invalid nucleotide display span")
        x = base_x(offset + span.start) - 0.003
        width = base_x(offset + span.end) - x
        axis.add_patch(Rectangle((x, y), width, 0.035, facecolor=span.color, edgecolor="none", alpha=0.9))


def draw_duplex(
    axis,
    *,
    top: str,
    bottom: str,
    y: float,
    coordinate_start: int | None = None,
    label: str | None = None,
) -> float:
    """Draw one fully paired, antiparallel duplex row."""

    if len(top) != len(bottom):
        raise RenderingError("three_way_junction_review cannot draw a duplex with unequal strand lengths")
    if len(top) > BASES_PER_ROW:
        raise RenderingError("three_way_junction_review duplex rows must be chunked before drawing")
    if label:
        axis.text(0.018, y, label, fontsize=6.2, color=MUTED, va="top")
    coordinate = "" if coordinate_start is None else f"{coordinate_start + 1}–{coordinate_start + len(top)}"
    axis.text(0.018, y - 0.12, coordinate, fontsize=5.2, family="monospace", color=MUTED, va="center")
    axis.text(0.076, y - 0.05, "5′", fontsize=5.5, color=MUTED, ha="right", va="center")
    axis.text(0.076, y - 0.24, "3′", fontsize=5.5, color=MUTED, ha="right", va="center")
    axis.text(
        SEQUENCE_X,
        y - 0.05,
        spaced(top),
        fontsize=BASE_FONT_SIZE,
        family="monospace",
        color=INK,
        va="center",
        zorder=3,
    )
    axis.text(
        SEQUENCE_X,
        y - 0.24,
        spaced(bottom),
        fontsize=BASE_FONT_SIZE,
        family="monospace",
        color=INK,
        va="center",
        zorder=3,
    )
    end_x = base_x(len(top) - 1) + 0.012
    axis.text(end_x, y - 0.05, "3′", fontsize=5.5, color=MUTED, va="center")
    axis.text(end_x, y - 0.24, "5′", fontsize=5.5, color=MUTED, va="center")
    _pair_edges(axis, start=0, length=len(top), top_y=y - 0.095, bottom_y=y - 0.195)
    return y - 0.36


def draw_aligned_fragment(
    axis,
    *,
    top: str,
    bottom_aligned: str,
    top_offset: int,
    bottom_offset: int,
    paired_start: int,
    paired_length: int,
    top_spans: tuple[ColoredSpan, ...],
    bottom_spans: tuple[ColoredSpan, ...],
    y: float,
) -> float:
    """Draw one annealed fragment with exact paired and unpaired bases."""

    width = max(top_offset + len(top), bottom_offset + len(bottom_aligned))
    if width > BASES_PER_ROW:
        raise RenderingError("expanded annealed-fragment rows exceed the nucleotide display width")
    _span_bars(axis, spans=top_spans, offset=top_offset, y=y + 0.015)
    _span_bars(axis, spans=bottom_spans, offset=bottom_offset, y=y - 0.31)
    axis.text(base_x(top_offset) - 0.012, y - 0.05, "5′", fontsize=5.5, color=MUTED, ha="right", va="center")
    axis.text(
        base_x(top_offset),
        y - 0.05,
        spaced(top),
        fontsize=BASE_FONT_SIZE,
        family="monospace",
        color=INK,
        va="center",
        zorder=3,
    )
    axis.text(
        base_x(bottom_offset),
        y - 0.24,
        spaced(bottom_aligned),
        fontsize=BASE_FONT_SIZE,
        family="monospace",
        color=INK,
        va="center",
        zorder=3,
    )
    axis.text(
        base_x(bottom_offset) - 0.012,
        y - 0.24,
        "3′",
        fontsize=5.5,
        color=MUTED,
        ha="right",
        va="center",
    )
    axis.text(base_x(top_offset + len(top)) + 0.004, y - 0.05, "3′", fontsize=5.5, color=MUTED, va="center")
    axis.text(
        base_x(bottom_offset + len(bottom_aligned)) + 0.004,
        y - 0.24,
        "5′",
        fontsize=5.5,
        color=MUTED,
        va="center",
    )
    _pair_edges(
        axis,
        start=paired_start,
        length=paired_length,
        top_y=y - 0.095,
        bottom_y=y - 0.195,
    )
    return y - 0.42


def draw_sequence_rows(axis, *, sequence: str, y: float, label: str, color: str = INK) -> float:
    """Draw one 5-prime-to-3-prime order sequence across bounded rows."""

    chunks = sequence_chunks(sequence)
    for index, chunk in enumerate(chunks):
        row_label = label if index == 0 else ""
        axis.text(0.018, y - 0.02, row_label, fontsize=5.6, color=MUTED, va="center")
        axis.text(0.085, y - 0.02, "5′", fontsize=5.2, color=MUTED, ha="right", va="center")
        axis.text(
            SEQUENCE_X,
            y - 0.02,
            spaced(chunk.sequence),
            fontsize=BASE_FONT_SIZE,
            family="monospace",
            color=color,
            va="center",
        )
        end_x = base_x(len(chunk.sequence) - 1) + 0.012
        axis.text(end_x, y - 0.02, "3′", fontsize=5.2, color=MUTED, va="center")
        y -= 0.19
    return y


__all__ = [
    "BARCODE",
    "FRAGMENT_A",
    "FRAGMENT_B",
    "INK",
    "MUTED",
    "PRIMER",
    "TOEHOLD",
    "ColoredSpan",
    "base_x",
    "draw_aligned_fragment",
    "draw_duplex",
    "draw_sequence_rows",
]
