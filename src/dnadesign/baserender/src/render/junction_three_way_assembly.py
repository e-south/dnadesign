"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_three_way_assembly.py

Registered assembly-process and selected-detail renderer for Junction designs.

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

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, SchemaError
from .junction_review.assembly_geometry import (
    MAX_ASSEMBLY_FRAGMENTS,
    MAX_RECOVERED_DUPLEX_BASES,
    assembly_figure_height,
)
from .junction_review.assembly_panel import draw_assembly_process
from .junction_review.detail_geometry import junction_detail_base_glyph_count
from .junction_review.foundation import (
    INK,
    MUTED,
    fragment_order_lengths,
    length_summary,
    review_from_record,
    selected_ids,
    validate_figure_size,
)
from .junction_three_way_detail import draw_junction_detail
from .palette import Palette

_RENDERER = "junction_three_way_assembly"
_FIGURE_WIDTH = 15.2
_MAX_DETAIL_JUNCTIONS = 8
_MAX_DETAIL_BASE_GLYPHS_PER_JUNCTION = 512


@dataclass(frozen=True, slots=True)
class AssemblyOptions:
    view: str
    junction_indices: tuple[int, ...]


def _resolve_options(
    review: ThreeWayJunctionReviewV1,
    options: Mapping[str, object] | None,
) -> AssemblyOptions:
    view = "assembly" if options is None else str(options.get("view", "assembly")).strip()
    if view not in {"assembly", "junction_detail"}:
        raise SchemaError(f"{_RENDERER} render.options.view must be 'assembly' or 'junction_detail'")
    available = tuple(junction.junction_id for junction in review.geometry.junctions)
    if view == "assembly":
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


def _assembly_size(style: Style, review: ThreeWayJunctionReviewV1) -> tuple[float, float]:
    height = assembly_figure_height(len(review.recovery.extended_top_sequence_5to3))
    return validate_figure_size(style, renderer=_RENDERER, width=_FIGURE_WIDTH, height=height)


def _detail_size(style: Style, count: int) -> tuple[float, float]:
    columns = min(2, count)
    rows = math.ceil(count / columns)
    return validate_figure_size(
        style,
        renderer=_RENDERER,
        width=8.2 if columns == 1 else _FIGURE_WIDTH,
        height=1.65 + rows * 4.8,
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


def _validate_assembly_workload(review: ThreeWayJunctionReviewV1) -> None:
    product_length = len(review.recovery.extended_top_sequence_5to3)
    if product_length > MAX_RECOVERED_DUPLEX_BASES:
        raise SchemaError(
            f"{_RENDERER} target {review.target.target_id!r} has a {product_length} bp expected product; "
            f"the recovered-duplex limit is {MAX_RECOVERED_DUPLEX_BASES} bp"
        )
    fragment_count = len(review.geometry.fragments)
    if fragment_count > MAX_ASSEMBLY_FRAGMENTS:
        raise SchemaError(
            f"{_RENDERER} target {review.target.target_id!r} contains {fragment_count} fragments; "
            f"the assembly-view limit is {MAX_ASSEMBLY_FRAGMENTS}"
        )


@dataclass(frozen=True)
class JunctionThreeWayAssemblyRenderer:
    """Render an assembly process or explicitly selected nucleotide-level 3WJs."""

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
        if resolved.view == "assembly":
            _validate_assembly_workload(review)
            _assembly_size(style, review)
        else:
            _validate_detail_workload(review, resolved.junction_indices)
            _detail_size(style, len(resolved.junction_indices))

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
        if resolved.view == "assembly":
            _validate_assembly_workload(review)
            size = _assembly_size(style, review)
            coordinate_height = assembly_figure_height(len(review.recovery.extended_top_sequence_5to3))
            figure, axis = plt.subplots(figsize=size, dpi=style.dpi)
            draw_assembly_process(axis, review, height=coordinate_height)
            figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            return figure

        _validate_detail_workload(review, resolved.junction_indices)
        count = len(resolved.junction_indices)
        columns = min(2, count)
        rows = math.ceil(count / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=_detail_size(style, count),
            dpi=style.dpi,
            squeeze=False,
        )
        figure.suptitle(
            (
                f"{count} selected three-way "
                f"{'junction shows' if count == 1 else 'junctions show'} "
                "the expected local annealing geometry"
            ),
            x=0.5,
            y=0.995,
            ha="center",
            va="top",
            fontsize=17.0,
            fontweight="semibold",
            color=INK,
        )
        lengths = fragment_order_lengths(review)
        figure.text(
            0.5,
            0.95,
            (
                f"The {len(review.target.sequence_5to3)} bp target uses "
                f"{len(review.geometry.junctions[0].toehold)} nt toeholds and "
                f"{len(review.geometry.junctions[0].barcode)} nt barcodes; fragment oligos span "
                f"{length_summary(lengths)}"
            ),
            fontsize=10.5,
            color=MUTED,
            ha="center",
            va="top",
        )
        for axis, index in zip(axes.flat, resolved.junction_indices, strict=False):
            draw_junction_detail(axis, review, index)
        for axis in tuple(axes.flat)[count:]:
            axis.axis("off")
        figure.text(
            0.5,
            0.012,
            "Expected sequence geometry; no experimental measurements are shown",
            fontsize=9.0,
            color=MUTED,
            ha="center",
            va="bottom",
        )
        figure.subplots_adjust(left=0.012, right=0.992, top=0.91, bottom=0.045, wspace=0.02, hspace=0.08)
        return figure


__all__ = ["JunctionThreeWayAssemblyRenderer"]
