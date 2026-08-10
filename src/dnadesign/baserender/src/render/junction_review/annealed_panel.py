"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/annealed_panel.py

Selection, geometry, and composition for the fragment-annealing review panel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

from matplotlib.collections import LineCollection

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ...core import SchemaError
from ..sequence_preview import bounded_svg_gid
from .foundation import (
    BARCODE,
    BARCODE_DARK,
    DOMAIN_DARK,
    INK,
    MUTED,
    PAIR,
    TOEHOLD,
    TOEHOLD_DARK,
    safe_identifier,
    selected_ids,
)
from .fragment_geometry import FragmentPairGeometry, fragment_pair_geometry
from .primitives import draw_base_run, draw_segmented_strand

MAX_FRAGMENTS = 18
MAX_ROW_BASES = 140
ROW_STEP = 1.18
VERTICAL_MARGIN = 0.75
ANNOTATION_FONTSIZE = 12.5


def annealed_figure_height(fragment_count: int) -> float:
    """Return a canvas height with stable row and title clearance."""

    return VERTICAL_MARGIN + ROW_STEP * fragment_count


def fragment_selection(
    review: ThreeWayJunctionReviewV1,
    options: Mapping[str, object] | None,
    *,
    renderer: str,
) -> tuple[int, ...]:
    available = tuple(fragment.fragment_id for fragment in review.geometry.fragments)
    chosen = selected_ids(
        options,
        key="fragment_ids",
        available=available,
        maximum=MAX_FRAGMENTS,
        required=len(available) > MAX_FRAGMENTS,
        renderer=renderer,
    )
    by_id = {fragment_id: index for index, fragment_id in enumerate(available)}
    return tuple(by_id[fragment_id] for fragment_id in chosen)


def validate_fragment_rows(review: ThreeWayJunctionReviewV1, indices: tuple[int, ...], *, renderer: str) -> None:
    for index in indices:
        geometry = fragment_pair_geometry(review, index)
        if geometry.width > MAX_ROW_BASES:
            raise SchemaError(
                f"{renderer} fragment {geometry.strand.fragment_id!r} spans {geometry.width} displayed bases; "
                f"the limit is {MAX_ROW_BASES}"
            )


def _component_text_color(color: str) -> str:
    if color == BARCODE:
        return BARCODE_DARK
    if color == TOEHOLD:
        return TOEHOLD_DARK
    return DOMAIN_DARK


def _draw_span_labels(axis, *, spans, start_x: float, base_step: float, y: float, above: bool) -> None:
    for start, end, color, label in spans:
        left = start_x + start * base_step
        right = start_x + end * base_step
        axis.text(
            (left + right) / 2,
            y,
            label,
            ha="center",
            va="bottom" if above else "top",
            fontsize=ANNOTATION_FONTSIZE,
            color=_component_text_color(color),
        )


def _draw_fragment(
    axis,
    geometry: FragmentPairGeometry,
    *,
    index: int,
    y: float,
    base_step: float,
    maximum_width: int,
    fontsize: float,
) -> None:
    fragment = geometry.fragment
    strand = geometry.strand
    bottom = geometry.bottom_sequence_left_to_right
    row_start = 0.18 + (maximum_width - geometry.width) * base_step / 2
    top_start = row_start + geometry.top_offset * base_step
    bottom_start = row_start + geometry.bottom_offset * base_step
    top_y, bottom_y = y, y - 0.34
    strand_height = 0.15
    draw_segmented_strand(
        axis,
        start_x=top_start,
        center_y=top_y,
        base_step=base_step,
        length=len(strand.barcode_bearing_sequence_5to3),
        segments=tuple((start, end, color) for start, end, color, _label in geometry.top_spans),
        height=strand_height,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:top-strand",
    )
    draw_segmented_strand(
        axis,
        start_x=bottom_start,
        center_y=bottom_y,
        base_step=base_step,
        length=len(bottom),
        segments=tuple((start, end, color) for start, end, color, _label in geometry.bottom_spans),
        height=strand_height,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:bottom-strand",
    )
    _draw_span_labels(
        axis,
        spans=geometry.top_spans,
        start_x=top_start,
        base_step=base_step,
        y=top_y + 0.13,
        above=True,
    )
    _draw_span_labels(
        axis,
        spans=geometry.bottom_spans,
        start_x=bottom_start,
        base_step=base_step,
        y=bottom_y - 0.13,
        above=False,
    )
    pair_segments = [
        (
            (row_start + (base + 0.5) * base_step, top_y - 0.07),
            (row_start + (base + 0.5) * base_step, bottom_y + 0.07),
        )
        for base in range(geometry.paired_start, geometry.paired_start + geometry.paired_length)
    ]
    pairs = LineCollection(pair_segments, colors=PAIR, linewidths=0.6, zorder=1)
    pairs.set_gid(bounded_svg_gid(f"junction-annealed:{fragment.fragment_id}:watson-crick"))
    axis.add_collection(pairs)
    draw_base_run(
        axis,
        strand.barcode_bearing_sequence_5to3,
        start_x=top_start,
        start_y=top_y,
        delta_x=base_step,
        delta_y=0,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:top",
        fontsize=fontsize,
    )
    draw_base_run(
        axis,
        bottom,
        start_x=bottom_start,
        start_y=bottom_y,
        delta_x=base_step,
        delta_y=0,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:bottom",
        fontsize=fontsize,
    )
    axis.text(
        top_start - 0.008,
        top_y,
        "5′",
        ha="right",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=MUTED,
    )
    axis.text(
        top_start + len(strand.barcode_bearing_sequence_5to3) * base_step + 0.004,
        top_y,
        "3′",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=MUTED,
    )
    axis.text(
        bottom_start - 0.008,
        bottom_y,
        "3′",
        ha="right",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=MUTED,
    )
    axis.text(
        bottom_start + len(bottom) * base_step + 0.004,
        bottom_y,
        "5′",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=MUTED,
    )
    axis.text(
        0.015,
        y - 0.02,
        (
            f"F{index + 1:02d} · {fragment.role}\n"
            f"{len(strand.barcode_bearing_sequence_5to3)} nt / "
            f"{len(strand.complement_sequence_5to3)} nt"
        ),
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=INK,
    )


def draw_annealed_panel(axis, review: ThreeWayJunctionReviewV1, indices: tuple[int, ...], *, height: float) -> None:
    axis.set_gid("junction-annealed-fragments:map")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, height)
    axis.axis("off")
    rows = tuple(fragment_pair_geometry(review, index) for index in indices)
    maximum_width = max(row.width for row in rows)
    base_step = 0.76 / maximum_width
    base_fontsize = min(11.5, max(8.0, 1_100 / maximum_width))
    count = len(indices)
    axis.text(
        0.5,
        height - 0.10,
        (
            f"{count} {'fragment pair shows' if count == 1 else 'fragment pairs show'} "
            f"the expected annealing for {safe_identifier(review.target.target_id)}"
        ),
        fontsize=20.0,
        fontweight="semibold",
        color=INK,
        ha="center",
        va="top",
    )
    y = height - VERTICAL_MARGIN
    for index, geometry in zip(indices, rows, strict=True):
        _draw_fragment(
            axis,
            geometry,
            index=index,
            y=y,
            base_step=base_step,
            maximum_width=maximum_width,
            fontsize=base_fontsize,
        )
        y -= ROW_STEP
