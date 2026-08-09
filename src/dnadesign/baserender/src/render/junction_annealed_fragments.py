"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_annealed_fragments.py

Nucleotide-level maps of expected Junction fragment pairing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, SchemaError
from .junction_review_common import (
    DOMAIN,
    INK,
    MUTED,
    PAIR,
    draw_base_run,
    junction_color,
    review_from_record,
    safe_identifier,
    selected_ids,
    validate_figure_size,
)
from .palette import Palette
from .sequence_preview import bounded_svg_gid

_RENDERER = "junction_annealed_fragments"
_MAX_FRAGMENTS = 18
_MAX_ROW_BASES = 140
_FIGURE_WIDTH = 15.2


def _fragment_selection(
    review: ThreeWayJunctionReviewV1,
    options: Mapping[str, object] | None,
) -> tuple[int, ...]:
    available = tuple(fragment.fragment_id for fragment in review.geometry.fragments)
    chosen = selected_ids(
        options,
        key="fragment_ids",
        available=available,
        maximum=_MAX_FRAGMENTS,
        required=len(available) > _MAX_FRAGMENTS,
        renderer=_RENDERER,
    )
    by_id = {fragment_id: index for index, fragment_id in enumerate(available)}
    return tuple(by_id[fragment_id] for fragment_id in chosen)


def _row_geometry(review: ThreeWayJunctionReviewV1, index: int):
    previous = None if index == 0 else review.geometry.junctions[index - 1]
    following = None if index == len(review.strands) - 1 else review.geometry.junctions[index]
    fragment = review.geometry.fragments[index]
    strand = review.strands[index]
    domain_length = fragment.domain_span.end - fragment.domain_span.start
    top_domain_start = 0 if previous is None else len(previous.barcode)
    bottom_domain_start = 0 if previous is None else len(previous.toehold)
    aligned_domain_start = max(top_domain_start, bottom_domain_start)
    top_offset = aligned_domain_start - top_domain_start
    bottom_offset = aligned_domain_start - bottom_domain_start
    top_spans: list[tuple[int, int, str, str]] = []
    bottom_spans: list[tuple[int, int, str, str]] = []
    if previous is not None:
        color = junction_color(index - 1)
        top_spans.append((0, len(previous.barcode), color, f"b{index}*"))
        bottom_spans.append((0, len(previous.toehold), color, f"t{index}*"))
    top_spans.append((top_domain_start, top_domain_start + domain_length, DOMAIN, "target"))
    bottom_spans.append((bottom_domain_start, bottom_domain_start + domain_length, DOMAIN, "target"))
    if following is not None:
        color = junction_color(index)
        start = top_domain_start + domain_length
        top_spans.extend(
            (
                (start, start + len(following.toehold), color, f"t{index + 1}"),
                (
                    start + len(following.toehold),
                    start + len(following.toehold) + len(following.barcode),
                    color,
                    f"b{index + 1}",
                ),
            )
        )
    return (
        fragment,
        strand,
        strand.complement_sequence_5to3[::-1],
        top_offset,
        bottom_offset,
        aligned_domain_start,
        domain_length,
        tuple(top_spans),
        tuple(bottom_spans),
    )


def _validate_rows(review: ThreeWayJunctionReviewV1, indices: tuple[int, ...]) -> None:
    for index in indices:
        _, strand, bottom, top_offset, bottom_offset, *_ = _row_geometry(review, index)
        width = max(top_offset + len(strand.barcode_bearing_sequence_5to3), bottom_offset + len(bottom))
        if width > _MAX_ROW_BASES:
            raise SchemaError(
                f"{_RENDERER} fragment {strand.fragment_id!r} spans {width} displayed bases; "
                f"the limit is {_MAX_ROW_BASES}"
            )


def _draw_spans(axis, *, spans, offset: int, x_for, y: float, height: float) -> None:
    for start, end, color, label in spans:
        left = x_for(offset + start) - 0.003
        right = x_for(offset + end)
        axis.add_patch(
            Rectangle((left, y), right - left, height, facecolor=color, edgecolor="none", alpha=0.23, zorder=0)
        )
        axis.text(
            (left + right) / 2,
            y + height + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=color if color != DOMAIN else MUTED,
        )


def _draw_fragment(axis, review: ThreeWayJunctionReviewV1, *, index: int, y: float) -> None:
    (
        fragment,
        strand,
        bottom,
        top_offset,
        bottom_offset,
        paired_start,
        paired_length,
        top_spans,
        bottom_spans,
    ) = _row_geometry(review, index)
    width = max(top_offset + len(strand.barcode_bearing_sequence_5to3), bottom_offset + len(bottom))
    left_x, right_x = 0.13, 0.97
    step = (right_x - left_x) / max(width, 1)

    def x_for(base: float) -> float:
        return left_x + (base * step)

    top_y, bottom_y = y, y - 0.18
    _draw_spans(axis, spans=top_spans, offset=top_offset, x_for=x_for, y=top_y - 0.045, height=0.085)
    _draw_spans(axis, spans=bottom_spans, offset=bottom_offset, x_for=x_for, y=bottom_y - 0.045, height=0.085)
    pair_segments = [
        ((x_for(base + 0.5), top_y - 0.055), (x_for(base + 0.5), bottom_y + 0.055))
        for base in range(paired_start, paired_start + paired_length)
    ]
    pairs = LineCollection(pair_segments, colors=PAIR, linewidths=0.42, zorder=1)
    pairs.set_gid(bounded_svg_gid(f"junction-annealed:{fragment.fragment_id}:watson-crick"))
    axis.add_collection(pairs)
    draw_base_run(
        axis,
        strand.barcode_bearing_sequence_5to3,
        start_x=x_for(top_offset),
        start_y=top_y,
        delta_x=step,
        delta_y=0,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:top",
        fontsize=5.4,
    )
    draw_base_run(
        axis,
        bottom,
        start_x=x_for(bottom_offset),
        start_y=bottom_y,
        delta_x=step,
        delta_y=0,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:bottom",
        fontsize=5.4,
    )
    axis.text(x_for(top_offset) - 0.008, top_y, "5′", ha="right", va="center", fontsize=5.4, color=MUTED)
    axis.text(
        x_for(top_offset + len(strand.barcode_bearing_sequence_5to3)) + 0.004,
        top_y,
        "3′",
        va="center",
        fontsize=5.4,
        color=MUTED,
    )
    axis.text(x_for(bottom_offset) - 0.008, bottom_y, "3′", ha="right", va="center", fontsize=5.4, color=MUTED)
    axis.text(x_for(bottom_offset + len(bottom)) + 0.004, bottom_y, "5′", va="center", fontsize=5.4, color=MUTED)
    axis.text(
        0.015,
        y + 0.02,
        f"F{index + 1:02d}\n{fragment.role}",
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="semibold",
        color=INK,
    )


@dataclass(frozen=True)
class JunctionAnnealedFragmentsRenderer:
    """Render explicitly selected, sequence-derived fragment-pairing maps."""

    def preflight(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ) -> None:
        _ = palette
        review = review_from_record(record)
        indices = _fragment_selection(review, options)
        _validate_rows(review, indices)
        validate_figure_size(style, renderer=_RENDERER, width=_FIGURE_WIDTH, height=1.2 + 0.65 * len(indices))

    def render(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ):
        _ = palette
        review = review_from_record(record)
        indices = _fragment_selection(review, options)
        _validate_rows(review, indices)
        size = validate_figure_size(
            style,
            renderer=_RENDERER,
            width=_FIGURE_WIDTH,
            height=1.2 + 0.65 * len(indices),
        )
        figure, axis = plt.subplots(figsize=size, dpi=style.dpi)
        axis.set_gid("junction-annealed-fragments:map")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.2 + 0.65 * len(indices))
        axis.axis("off")
        height = 1.2 + 0.65 * len(indices)
        axis.text(
            0.015, height - 0.12, "Expected fragment pairing", fontsize=12.5, fontweight="semibold", color=INK, va="top"
        )
        axis.text(
            0.015,
            height - 0.42,
            (
                f"{safe_identifier(review.target.target_id)} · "
                f"{len(review.target.sequence_5to3)} bp target · {len(indices)} selected fragments"
            ),
            fontsize=6.7,
            color=MUTED,
            va="top",
        )
        axis.text(
            0.985,
            height - 0.12,
            "Bases and Watson–Crick edges come from the verified design record.",
            fontsize=6.0,
            color=MUTED,
            ha="right",
            va="top",
        )
        y = height - 0.92
        for index in indices:
            _draw_fragment(axis, review, index=index, y=y)
            y -= 0.65
        axis.text(
            0.015,
            0.08,
            "Sequence-derived pairing schematic; not a thermodynamic prediction or experimental result.",
            fontsize=6.0,
            color=MUTED,
            va="bottom",
        )
        figure.subplots_adjust(left=0.012, right=0.992, top=0.99, bottom=0.015)
        return figure


__all__ = ["JunctionAnnealedFragmentsRenderer"]
