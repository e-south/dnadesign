"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review_common.py

Shared contracts and bounds for Junction review renderers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, RenderingError, SchemaError
from ..core.pydantic_validation import format_validation_error
from .sequence_preview import bounded_text_preview

INK = "#172033"
MUTED = "#667085"
PAIR = "#CDD3DB"
DOMAIN = "#E9EDF2"
BACKGROUND = "#FFFFFF"
JUNCTION_COLORS = (
    "#4E79A7",
    "#2A9D8F",
    "#7A9E4E",
    "#D4A72C",
    "#D96C5F",
    "#A06CD5",
    "#5C8D89",
    "#C76B98",
)

MAX_DPI = 600
MAX_FIGURE_SCALE = 4.0
MAX_CANVAS_DIMENSION_PX = 16_384
MAX_CANVAS_RGBA_BYTES = 64 * 1024 * 1024


def review_from_record(record: Record) -> ThreeWayJunctionReviewV1:
    """Load and revalidate the private Junction evidence on a BaseRender record."""

    meta = record.meta if isinstance(record.meta, Mapping) else None
    payload = None if meta is None else meta.get("three_way_junction_review")
    if not isinstance(payload, Mapping):
        raise RenderingError("Junction review renderers require record.meta.three_way_junction_review")
    try:
        return ThreeWayJunctionReviewV1.model_validate(payload)
    except ValidationError as exc:
        detail = format_validation_error(exc)
        raise RenderingError(f"Junction review renderer received invalid evidence: {detail}") from None


def safe_identifier(value: str) -> str:
    """Bound untrusted identifiers before adding them to a figure."""

    preview = bounded_text_preview(value, visible_chars=36, exact_limit=64)
    if preview.abbreviated:
        return f"{preview.length_chars} chars · SHA-256[:12] {preview.sha256_prefix} · {preview.preview}"
    return preview.preview


def junction_color(index: int) -> str:
    return JUNCTION_COLORS[index % len(JUNCTION_COLORS)]


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
        artist.set_gid(f"{gid_prefix}:base:{index}:{base}")
        artists.append(artist)
    return tuple(artists)


def validate_figure_size(
    style: Style,
    *,
    renderer: str,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Validate figure dimensions before Matplotlib allocates a canvas."""

    try:
        dpi = float(style.dpi)
        scale = float(style.figure_scale)
    except (TypeError, ValueError, OverflowError):
        raise SchemaError(f"{renderer} style dimensions must be finite numbers") from None
    if not math.isfinite(dpi) or dpi > MAX_DPI:
        raise SchemaError(f"{renderer} style.dpi exceeds the renderer limit")
    if not math.isfinite(scale) or scale > MAX_FIGURE_SCALE:
        raise SchemaError(f"{renderer} style.figure_scale exceeds the renderer limit")
    figure_width = width * scale
    figure_height = height * scale
    width_px = math.ceil(figure_width * dpi)
    height_px = math.ceil(figure_height * dpi)
    if max(width_px, height_px) > MAX_CANVAS_DIMENSION_PX:
        raise SchemaError(f"{renderer} canvas dimension exceeds the renderer limit")
    if width_px * height_px * 4 > MAX_CANVAS_RGBA_BYTES:
        raise SchemaError(f"{renderer} canvas exceeds the 64 MiB RGBA allocation limit")
    return figure_width, figure_height


def selected_ids(
    options: Mapping[str, object] | None,
    *,
    key: str,
    available: Sequence[str],
    maximum: int,
    required: bool,
    renderer: str,
) -> tuple[str, ...]:
    """Resolve one ordered, explicit identifier subset from renderer options."""

    raw = None if options is None else options.get(key)
    if raw is None:
        if required:
            raise SchemaError(f"{renderer} requires render.options.{key}")
        selected = tuple(available)
    else:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise SchemaError(f"{renderer} render.options.{key} must be a list of identifiers")
        selected = tuple(str(value).strip() for value in raw)
        if not selected or any(not value for value in selected):
            raise SchemaError(f"{renderer} render.options.{key} must contain non-empty identifiers")
        if len(selected) != len(set(selected)):
            raise SchemaError(f"{renderer} render.options.{key} must not contain duplicates")
    if len(selected) > maximum:
        raise SchemaError(f"{renderer} accepts at most {maximum} selected {key}")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SchemaError(f"{renderer} received unknown {key}: {unknown[:5]}")
    return selected


__all__ = [
    "BACKGROUND",
    "DOMAIN",
    "INK",
    "MUTED",
    "PAIR",
    "draw_base_run",
    "junction_color",
    "review_from_record",
    "safe_identifier",
    "selected_ids",
    "validate_figure_size",
]
