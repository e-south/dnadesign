"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/three_way_junction_review.py

Nucleotide-level renderer for neutral three-way-junction review evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, RenderingError, SchemaError
from ..core.pydantic_validation import format_validation_error
from .junction_pairing_layout import BASES_PER_ROW, complement, review_content_height, sequence_chunks
from .palette import Palette
from .sequence_preview import bounded_text_preview

_INK = "#172033"
_MUTED = "#667085"
_PAIR = "#CBD1D9"
_FRAGMENT_A = "#EEF1F5"
_FRAGMENT_B = "#E3E8EF"
_TOEHOLD = "#5B8DEF"
_BARCODE = "#2A9D8F"
_PRIMER = "#D97706"

_AXIS_GID = "three-way-junction-review:base-pair-map"
_FIGURE_WIDTH_INCHES = 15.2
_MAX_REVIEW_DPI = 600
_MAX_REVIEW_FIGURE_SCALE = 4.0
_MAX_REVIEW_CANVAS_DIMENSION_PX = 16_384
_MAX_REVIEW_CANVAS_RGBA_BYTES = 64 * 1024 * 1024
_RGBA_BYTES_PER_PIXEL = 4

_SEQUENCE_X = 0.105
_SEQUENCE_WIDTH = 0.835
_BASE_FONT_SIZE = 5.6


def _review_figure_size(style: Style, review: ThreeWayJunctionReviewV1) -> tuple[float, float]:
    """Validate and return the content-aware figure size in inches."""

    try:
        dpi = float(style.dpi)
        figure_scale = float(style.figure_scale)
    except (TypeError, ValueError, OverflowError):
        raise SchemaError("three_way_junction_review style dimensions must be finite numbers") from None
    if not math.isfinite(dpi):
        raise SchemaError("three_way_junction_review style.dpi must be finite")
    if not math.isfinite(figure_scale):
        raise SchemaError("three_way_junction_review style.figure_scale must be finite")
    if dpi > _MAX_REVIEW_DPI:
        raise SchemaError("three_way_junction_review style.dpi exceeds the renderer limit")
    if figure_scale > _MAX_REVIEW_FIGURE_SCALE:
        raise SchemaError("three_way_junction_review style.figure_scale exceeds the renderer limit")

    figure_width = _FIGURE_WIDTH_INCHES * figure_scale
    figure_height = review_content_height(review) * figure_scale
    width_px = math.ceil(figure_width * dpi)
    height_px = math.ceil(figure_height * dpi)
    if max(width_px, height_px) > _MAX_REVIEW_CANVAS_DIMENSION_PX:
        raise SchemaError("three_way_junction_review canvas dimension exceeds the renderer limit")
    if width_px * height_px * _RGBA_BYTES_PER_PIXEL > _MAX_REVIEW_CANVAS_RGBA_BYTES:
        raise SchemaError("three_way_junction_review canvas exceeds the 64 MiB RGBA allocation limit")
    return figure_width, figure_height


def _review_from_record(record: Record) -> ThreeWayJunctionReviewV1:
    meta = record.meta if isinstance(record.meta, Mapping) else None
    if meta is None:
        raise RenderingError("three_way_junction_review requires record.meta.three_way_junction_review")
    payload = meta.get("three_way_junction_review")
    if not isinstance(payload, Mapping):
        raise RenderingError("three_way_junction_review requires record.meta.three_way_junction_review")
    try:
        return ThreeWayJunctionReviewV1.model_validate(payload)
    except ValidationError as exc:
        detail = format_validation_error(exc)
        raise RenderingError(f"three_way_junction_review received invalid review evidence: {detail}") from None


def _setup_axis(axis, *, height: float) -> None:
    axis.set_gid(_AXIS_GID)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, height)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _safe_identifier(value: str) -> str:
    preview = bounded_text_preview(value, visible_chars=36, exact_limit=64)
    if preview.abbreviated:
        return f"{preview.length_chars} chars · SHA-256[:12] {preview.sha256_prefix} · {preview.preview}"
    return preview.preview


def _base_x(index: int) -> float:
    return _SEQUENCE_X + (index * _SEQUENCE_WIDTH / BASES_PER_ROW)


def _spaced(sequence: str) -> str:
    return " ".join(sequence)


def _pair_edges(axis, *, length: int, top_y: float, bottom_y: float) -> None:
    segments = [((_base_x(index), top_y), (_base_x(index), bottom_y)) for index in range(length)]
    axis.add_collection(LineCollection(segments, colors=_PAIR, linewidths=0.42, zorder=1))


def _draw_duplex(
    axis,
    *,
    top: str,
    bottom: str,
    y: float,
    coordinate_start: int | None = None,
    label: str | None = None,
) -> float:
    if len(top) != len(bottom):
        raise RenderingError("three_way_junction_review cannot draw a duplex with unequal strand lengths")
    if label:
        axis.text(0.018, y, label, fontsize=6.2, color=_MUTED, va="top")
    coordinate = "" if coordinate_start is None else f"{coordinate_start + 1}–{coordinate_start + len(top)}"
    axis.text(0.018, y - 0.12, coordinate, fontsize=5.2, family="monospace", color=_MUTED, va="center")
    axis.text(0.076, y - 0.05, "5′", fontsize=5.5, color=_MUTED, ha="right", va="center")
    axis.text(0.076, y - 0.24, "3′", fontsize=5.5, color=_MUTED, ha="right", va="center")
    axis.text(
        _SEQUENCE_X,
        y - 0.05,
        _spaced(top),
        fontsize=_BASE_FONT_SIZE,
        family="monospace",
        color=_INK,
        va="center",
        zorder=3,
    )
    axis.text(
        _SEQUENCE_X,
        y - 0.24,
        _spaced(bottom),
        fontsize=_BASE_FONT_SIZE,
        family="monospace",
        color=_INK,
        va="center",
        zorder=3,
    )
    end_x = _base_x(len(top) - 1) + 0.012
    axis.text(end_x, y - 0.05, "3′", fontsize=5.5, color=_MUTED, va="center")
    axis.text(end_x, y - 0.24, "5′", fontsize=5.5, color=_MUTED, va="center")
    _pair_edges(axis, length=len(top), top_y=y - 0.095, bottom_y=y - 0.195)
    return y - 0.36


def _draw_sequence_rows(axis, *, sequence: str, y: float, label: str, color: str = _INK) -> float:
    chunks = sequence_chunks(sequence)
    for index, chunk in enumerate(chunks):
        row_label = label if index == 0 else ""
        axis.text(0.018, y - 0.02, row_label, fontsize=5.6, color=_MUTED, va="center")
        axis.text(0.085, y - 0.02, "5′", fontsize=5.2, color=_MUTED, ha="right", va="center")
        axis.text(
            _SEQUENCE_X,
            y - 0.02,
            _spaced(chunk.sequence),
            fontsize=_BASE_FONT_SIZE,
            family="monospace",
            color=color,
            va="center",
        )
        end_x = _base_x(len(chunk.sequence) - 1) + 0.012
        axis.text(end_x, y - 0.02, "3′", fontsize=5.2, color=_MUTED, va="center")
        y -= 0.19
    return y


def _intersections(start: int, end: int, review: ThreeWayJunctionReviewV1):
    spans = []
    for fragment in review.geometry.fragments:
        left = max(start, fragment.domain_span.start)
        right = min(end, fragment.domain_span.end)
        if left < right:
            spans.append(
                (left, right, _FRAGMENT_A if fragment.index % 2 == 0 else _FRAGMENT_B, f"F{fragment.index + 1}")
            )
    for junction in review.geometry.junctions:
        left = max(start, junction.toehold_span.start)
        right = min(end, junction.toehold_span.end)
        if left < right:
            spans.append((left, right, _TOEHOLD, "t"))
    return spans


def _draw_target(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    axis.text(0.018, y, "Target duplex", fontsize=8.5, fontweight="semibold", color=_INK, va="top")
    y -= 0.28
    target = review.target.sequence_5to3
    for chunk in sequence_chunks(target):
        for left, right, color, label in _intersections(chunk.start, chunk.end, review):
            x = _base_x(left - chunk.start) - 0.004
            width = max(0.008, _base_x(right - chunk.start) - x)
            axis.add_patch(Rectangle((x, y + 0.015), width, 0.035, facecolor=color, edgecolor="none", alpha=0.9))
            axis.text(x, y + 0.06, label, fontsize=4.7, color=color if label == "t" else _MUTED, va="bottom")
        y = _draw_duplex(
            axis,
            top=chunk.sequence,
            bottom=complement(chunk.sequence),
            y=y,
            coordinate_start=chunk.start,
        )
        y -= 0.10
    return y


def _draw_junctions(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    axis.text(0.018, y, "Junction duplexes", fontsize=8.5, fontweight="semibold", color=_INK, va="top")
    axis.text(
        0.16,
        y,
        "t/t* is the target-derived toehold; b/b* is the assigned barcode.",
        fontsize=5.8,
        color=_MUTED,
        va="top",
    )
    y -= 0.27
    for index, junction in enumerate(review.geometry.junctions, start=1):
        axis.text(
            0.018,
            y,
            (
                f"J{index:02d} · F{index:02d} → F{index + 1:02d} · "
                f"target bp {junction.toehold_span.start + 1}–{junction.toehold_span.end}"
            ),
            fontsize=5.8,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        y -= 0.16
        for kind, top, stored_complement, color in (
            ("toehold", junction.toehold, junction.toehold_complement, _TOEHOLD),
            ("barcode", junction.barcode, junction.barcode_complement, _BARCODE),
        ):
            top_chunks = sequence_chunks(top)
            aligned_bottom = stored_complement[::-1]
            bottom_chunks = sequence_chunks(aligned_bottom)
            for chunk_index, (top_chunk, bottom_chunk) in enumerate(zip(top_chunks, bottom_chunks, strict=True)):
                label = kind if chunk_index == 0 else f"{kind} cont."
                axis.text(0.018, y - 0.12, label, fontsize=5.4, color=color, va="center")
                y = _draw_duplex(axis, top=top_chunk.sequence, bottom=bottom_chunk.sequence, y=y)
        y -= 0.04
    return y


def _draw_oligo_orders(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    axis.text(0.018, y, "Fragment oligo orders", fontsize=8.5, fontweight="semibold", color=_INK, va="top")
    axis.text(0.16, y, "Every displayed strand is written 5′→3′.", fontsize=5.8, color=_MUTED, va="top")
    y -= 0.26
    for index, strand in enumerate(review.strands, start=1):
        axis.text(
            0.018,
            y,
            f"F{index:02d} · {strand.role}",
            fontsize=5.7,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        y -= 0.14
        y = _draw_sequence_rows(
            axis,
            sequence=strand.barcode_bearing_sequence_5to3,
            y=y,
            label="barcode",
            color=_BARCODE,
        )
        y = _draw_sequence_rows(
            axis,
            sequence=strand.complement_sequence_5to3,
            y=y,
            label="complement",
        )
    return y


def _draw_primers(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    axis.text(
        0.018,
        y,
        f"Recovery primers · {review.recovery.mode}",
        fontsize=8.5,
        fontweight="semibold",
        color=_INK,
        va="top",
    )
    y -= 0.25
    y = _draw_sequence_rows(
        axis,
        sequence=review.recovery.forward.order_sequence_5to3,
        y=y,
        label="forward",
        color=_PRIMER,
    )
    y = _draw_sequence_rows(
        axis,
        sequence=review.recovery.reverse.order_sequence_5to3,
        y=y,
        label="reverse",
        color=_PRIMER,
    )
    return y


@dataclass(frozen=True)
class ThreeWayJunctionReviewRenderer:
    def preflight(self, record: Record, style: Style, palette: Palette) -> None:
        _ = palette
        review = _review_from_record(record)
        _review_figure_size(style, review)

    def render(self, record: Record, style: Style, palette: Palette):
        _ = palette
        review = _review_from_record(record)
        figure_size = _review_figure_size(style, review)
        figure, axis = plt.subplots(1, 1, figsize=figure_size, dpi=style.dpi)
        height = figure_size[1] / style.figure_scale
        _setup_axis(axis, height=height)

        target_id = _safe_identifier(review.target.target_id)
        axis.text(
            0.018,
            height - 0.12,
            "Junction nucleotide audit",
            fontsize=12.5,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        axis.text(
            0.018,
            height - 0.43,
            f"{target_id} · {len(review.target.sequence_5to3)} bp · {len(review.strands) * 2} fragment oligos",
            fontsize=6.8,
            color=_MUTED,
            va="top",
        )
        y = height - 0.82
        y = _draw_target(axis, review, y=y)
        y = _draw_junctions(axis, review, y=y - 0.08)
        y = _draw_oligo_orders(axis, review, y=y - 0.08)
        _draw_primers(axis, review, y=y - 0.08)
        axis.text(
            0.018,
            0.12,
            "Sequence pairing map; not a thermodynamic, annealing, or PCR simulation.",
            fontsize=6.2,
            color=_MUTED,
            va="bottom",
        )
        figure.subplots_adjust(left=0.012, right=0.992, top=0.995, bottom=0.015)
        return figure


__all__ = ["ThreeWayJunctionReviewRenderer"]
