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
from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, RenderingError, SchemaError
from ..core.pydantic_validation import format_validation_error
from .junction_annealed_review import draw_annealed_fragments
from .junction_nucleotide_drawing import INK, MUTED
from .junction_pairing_layout import review_content_height
from .junction_review_sections import (
    draw_junctions,
    draw_oligo_orders,
    draw_primers,
    draw_recovered_duplex,
    draw_stage_path,
)
from .palette import Palette
from .sequence_preview import bounded_text_preview

_AXIS_GID = "three-way-junction-review:base-pair-map"
_FIGURE_WIDTH_INCHES = 15.2
_MAX_REVIEW_DPI = 600
_MAX_REVIEW_FIGURE_SCALE = 4.0
_MAX_REVIEW_CANVAS_DIMENSION_PX = 16_384
_MAX_REVIEW_CANVAS_RGBA_BYTES = 64 * 1024 * 1024
_RGBA_BYTES_PER_PIXEL = 4


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
            color=INK,
            va="top",
        )
        axis.text(
            0.018,
            height - 0.43,
            f"{target_id} · {len(review.target.sequence_5to3)} bp · {len(review.strands) * 2} fragment oligos",
            fontsize=6.8,
            color=MUTED,
            va="top",
        )
        y = draw_stage_path(axis, y=height - 0.70)
        y = draw_oligo_orders(axis, review, y=y)
        y = draw_annealed_fragments(axis, review, y=y - 0.08)
        y = draw_junctions(axis, review, y=y - 0.08)
        y = draw_recovered_duplex(axis, review, y=y - 0.08)
        draw_primers(axis, review, y=y - 0.08)
        axis.text(
            0.018,
            0.12,
            "Exact sequence-pairing audit; not a thermodynamic, annealing, ligation, or PCR simulation.",
            fontsize=6.2,
            color=MUTED,
            va="bottom",
        )
        figure.subplots_adjust(left=0.012, right=0.992, top=0.995, bottom=0.015)
        return figure


__all__ = ["ThreeWayJunctionReviewRenderer"]
