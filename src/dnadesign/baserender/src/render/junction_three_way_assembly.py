"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_three_way_assembly.py

Overview and selected-detail views for Junction three-way assemblies.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, SchemaError
from .junction_review_common import (
    INK,
    MUTED,
    junction_color,
    review_from_record,
    safe_identifier,
    selected_ids,
    validate_figure_size,
)
from .junction_three_way_detail import draw_junction_detail, junction_detail_base_glyph_count
from .palette import Palette

_RENDERER = "junction_three_way_assembly"
_FIGURE_WIDTH = 15.2
_MAX_DETAIL_JUNCTIONS = 8
_MAX_DETAIL_BASE_GLYPHS_PER_JUNCTION = 512
_MAX_OVERVIEW_FRAGMENTS = 256


@dataclass(frozen=True, slots=True)
class AssemblyOptions:
    view: str
    junction_indices: tuple[int, ...]


def _resolve_options(
    review: ThreeWayJunctionReviewV1,
    options: Mapping[str, object] | None,
) -> AssemblyOptions:
    view = "overview" if options is None else str(options.get("view", "overview")).strip()
    if view not in {"overview", "junction_detail"}:
        raise SchemaError(f"{_RENDERER} render.options.view must be 'overview' or 'junction_detail'")
    available = tuple(junction.junction_id for junction in review.geometry.junctions)
    if view == "overview":
        if options is not None and options.get("junction_ids") is not None:
            raise SchemaError(f"{_RENDERER} render.options.junction_ids is only valid for junction_detail")
        return AssemblyOptions(view=view, junction_indices=())
    selected = selected_ids(
        options,
        key="junction_ids",
        available=available,
        maximum=_MAX_DETAIL_JUNCTIONS,
        required=True,
        renderer=_RENDERER,
    )
    by_id = {junction_id: index for index, junction_id in enumerate(available)}
    return AssemblyOptions(view=view, junction_indices=tuple(by_id[junction_id] for junction_id in selected))


def _overview_size(style: Style) -> tuple[float, float]:
    return validate_figure_size(style, renderer=_RENDERER, width=_FIGURE_WIDTH, height=4.2)


def _detail_size(style: Style, count: int) -> tuple[float, float]:
    return validate_figure_size(
        style,
        renderer=_RENDERER,
        width=_FIGURE_WIDTH,
        height=1.15 + math.ceil(count / 2) * 4.0,
    )


def _validate_detail_workload(review: ThreeWayJunctionReviewV1, indices: tuple[int, ...]) -> None:
    for index in indices:
        count = junction_detail_base_glyph_count(review, index)
        if count > _MAX_DETAIL_BASE_GLYPHS_PER_JUNCTION:
            junction_id = review.geometry.junctions[index].junction_id
            raise SchemaError(
                f"{_RENDERER} junction {junction_id!r} requires {count} base glyphs; "
                f"the per-junction limit is {_MAX_DETAIL_BASE_GLYPHS_PER_JUNCTION}"
            )


def _validate_overview_workload(review: ThreeWayJunctionReviewV1) -> None:
    count = len(review.geometry.fragments)
    if count > _MAX_OVERVIEW_FRAGMENTS:
        raise SchemaError(
            f"{_RENDERER} target {review.target.target_id!r} contains {count} fragments; "
            f"the overview limit is {_MAX_OVERVIEW_FRAGMENTS}"
        )


def _draw_overview(axis, review: ThreeWayJunctionReviewV1) -> None:
    axis.set_gid("junction-three-way-assembly:overview")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.025, 0.94, "Three-way assembly map", fontsize=12.5, fontweight="semibold", color=INK, va="top")
    fragment_count = len(review.geometry.fragments)
    junction_count = len(review.geometry.junctions)
    axis.text(
        0.025,
        0.86,
        (
            f"{safe_identifier(review.target.target_id)} · "
            f"{fragment_count} {'fragment' if fragment_count == 1 else 'fragments'} · "
            f"{junction_count} {'junction' if junction_count == 1 else 'junctions'}"
        ),
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    left, right = 0.055, 0.965
    top_y, bottom_y = 0.47, 0.39
    length = len(review.target.sequence_5to3)

    def x_for(coordinate: int) -> float:
        return left + ((right - left) * coordinate / length)

    axis.add_line(Line2D([left, right], [top_y, top_y], linewidth=4.2, color="#6B7280"))
    axis.add_line(Line2D([left, right], [bottom_y, bottom_y], linewidth=4.2, color="#9AA1AA"))
    for fragment in review.geometry.fragments:
        x0, x1 = x_for(fragment.domain_span.start), x_for(fragment.domain_span.end)
        fill = "#E2E6EB" if fragment.index % 2 == 0 else "#D5DBE2"
        axis.add_patch(Rectangle((x0, top_y - 0.024), x1 - x0, 0.048, facecolor=fill, edgecolor="none", zorder=2))
        axis.text(
            (x0 + x1) / 2,
            0.31,
            f"F{fragment.index + 1}",
            fontsize=5.6,
            color=INK,
            ha="center",
            va="top",
        )
    for index, junction in enumerate(review.geometry.junctions):
        x = x_for(junction.toehold_span.end)
        color = junction_color(index)
        axis.add_line(Line2D([x - 0.005, x - 0.005], [top_y, 0.73], linewidth=3.2, color=color))
        axis.add_line(Line2D([x + 0.005, x + 0.005], [top_y, 0.73], linewidth=3.2, color=color, alpha=0.68))
        axis.text(x, 0.77, f"J{index + 1}", fontsize=5.5, color=color, ha="center", va="bottom")
        axis.add_line(Line2D([x, x], [bottom_y - 0.025, bottom_y + 0.025], linewidth=1.0, color=INK))
    axis.text(
        0.025,
        0.10,
        "Each stem marks one barcode-mediated interface. Use view: junction_detail for exact bases.",
        fontsize=6.2,
        color=MUTED,
        va="bottom",
    )
    axis.text(
        0.975,
        0.10,
        "Sequence-derived topology; not a structure or assembly simulation.",
        fontsize=6.0,
        color=MUTED,
        ha="right",
        va="bottom",
    )


@dataclass(frozen=True)
class JunctionThreeWayAssemblyRenderer:
    """Render a target overview or explicitly selected nucleotide-level 3WJs."""

    def preflight(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ) -> None:
        _ = palette
        review = review_from_record(record)
        resolved = _resolve_options(review, options)
        if resolved.view == "overview":
            _validate_overview_workload(review)
        else:
            _validate_detail_workload(review, resolved.junction_indices)
        _overview_size(style) if resolved.view == "overview" else _detail_size(style, len(resolved.junction_indices))

    def render(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ):
        _ = palette
        review = review_from_record(record)
        resolved = _resolve_options(review, options)
        if resolved.view == "overview":
            _validate_overview_workload(review)
            figure, axis = plt.subplots(figsize=_overview_size(style), dpi=style.dpi)
            _draw_overview(axis, review)
            figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            return figure

        _validate_detail_workload(review, resolved.junction_indices)
        count = len(resolved.junction_indices)
        rows = math.ceil(count / 2)
        figure, axes = plt.subplots(rows, 2, figsize=_detail_size(style, count), dpi=style.dpi, squeeze=False)
        figure.suptitle(
            f"Three-way junction details · {safe_identifier(review.target.target_id)}",
            x=0.02,
            y=0.995,
            ha="left",
            va="top",
            fontsize=12.5,
            fontweight="semibold",
            color=INK,
        )
        for axis, index in zip(axes.flat, resolved.junction_indices, strict=False):
            draw_junction_detail(axis, review, index)
        for axis in tuple(axes.flat)[count:]:
            axis.axis("off")
        figure.text(
            0.02,
            0.012,
            "Exact local sequence mapping; topology is schematic, not a structure, ligation, or yield prediction.",
            fontsize=6.0,
            color=MUTED,
            va="bottom",
        )
        figure.subplots_adjust(left=0.015, right=0.99, top=0.94, bottom=0.045, wspace=0.04, hspace=0.12)
        return figure


__all__ = ["JunctionThreeWayAssemblyRenderer"]
