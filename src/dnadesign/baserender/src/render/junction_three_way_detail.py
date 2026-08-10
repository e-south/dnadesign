"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_three_way_detail.py

Nucleotide-level geometry for selected Junction three-way interfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from .junction_review.detail_geometry import (
    BOTTOM_Y,
    STEM_LEFT_X,
    STEM_RIGHT_X,
    STEM_START_Y,
    STRAND_WIDTH,
    TOP_Y,
    junction_detail_base_glyph_count,
    local_junction_geometry,
)
from .junction_review.detail_primitives import add_break, add_nick, add_pairs, draw_component_path
from .junction_review.foundation import (
    BARCODE,
    BARCODE_DARK,
    DOMAIN,
    INK,
    MUTED,
    STRAND_EDGE,
    TOEHOLD,
    TOEHOLD_DARK,
    display_junction_id,
)
from .junction_review.primitives import draw_base_run, draw_molecular_path
from .sequence_preview import bounded_svg_gid


def draw_junction_detail(axis, review: ThreeWayJunctionReviewV1, index: int) -> None:
    """Draw one exact, sequence-derived three-arm interface on a shared base scale."""

    geometry = local_junction_geometry(review, index)
    junction = geometry.junction
    left_fragment = geometry.left_fragment
    right_fragment = geometry.right_fragment
    left_context = geometry.left_context
    right_context = geometry.right_context
    toehold = junction.toehold
    toehold_bottom = junction.toehold_complement[::-1]
    complement = str.maketrans("ACGT", "TGCA")
    left_bottom = left_context.translate(complement)
    right_bottom = right_context.translate(complement)
    left_length = len(left_context) + len(toehold)
    horizontal_span = max(left_length, len(right_context), 1)
    stem_top = STEM_START_Y + len(junction.barcode)
    base_fontsize = max(5.0, min(8.4, 180 / max(horizontal_span, len(junction.barcode), 1)))

    axis.set_gid(bounded_svg_gid(f"junction-three-way-assembly:{junction.junction_id}:detail"))
    axis.set_xlim(-horizontal_span - 2.5, horizontal_span + 2.5)
    axis.set_ylim(BOTTOM_Y - 4.2, stem_top + 5.2)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    local_id = display_junction_id(junction.junction_id)
    axis.text(
        -horizontal_span - 2.1,
        stem_top + 4.2,
        (
            f"{local_id} joins F{left_fragment.index + 1:02d} to F{right_fragment.index + 1:02d} "
            f"at target bp {junction.toehold_span.start + 1}–{junction.toehold_span.end}"
        ),
        fontsize=9.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )

    left_x = -left_length
    toehold_x = -len(toehold)
    right_x = 0.0

    draw_molecular_path(
        axis,
        [left_x, STEM_LEFT_X, STEM_LEFT_X],
        [TOP_Y, TOP_Y, stem_top],
        color=STRAND_EDGE,
        gid=f"junction:{junction.junction_id}:left-and-barcode-arm",
        linewidth=STRAND_WIDTH,
        zorder=0.2,
    )
    draw_molecular_path(
        axis,
        [STEM_RIGHT_X, STEM_RIGHT_X, len(right_context)],
        [stem_top, TOP_Y, TOP_Y],
        color=STRAND_EDGE,
        gid=f"junction:{junction.junction_id}:barcode-and-right-arm",
        linewidth=STRAND_WIDTH,
        zorder=0.2,
    )
    draw_molecular_path(
        axis,
        [left_x, toehold_x - 0.34],
        [BOTTOM_Y, BOTTOM_Y],
        color=STRAND_EDGE,
        gid=f"junction:{junction.junction_id}:left-complement-arm",
        linewidth=STRAND_WIDTH,
        zorder=0.2,
    )
    draw_molecular_path(
        axis,
        [toehold_x + 0.34, len(right_context)],
        [BOTTOM_Y, BOTTOM_Y],
        color=STRAND_EDGE,
        gid=f"junction:{junction.junction_id}:right-complement-arm",
        linewidth=STRAND_WIDTH,
        zorder=0.2,
    )

    if left_context:
        draw_component_path(
            axis,
            [left_x, toehold_x],
            [TOP_Y, TOP_Y],
            color=DOMAIN,
            gid=f"junction:{junction.junction_id}:left-target-top",
        )
        draw_component_path(
            axis,
            [left_x, toehold_x - 0.34],
            [BOTTOM_Y, BOTTOM_Y],
            color=DOMAIN,
            gid=f"junction:{junction.junction_id}:left-target-bottom",
        )
    draw_component_path(
        axis,
        [toehold_x, STEM_LEFT_X],
        [TOP_Y, TOP_Y],
        color=TOEHOLD,
        gid=f"junction:{junction.junction_id}:toehold-top-path",
    )
    draw_component_path(
        axis,
        [STEM_LEFT_X, STEM_LEFT_X],
        [TOP_Y, stem_top],
        color=BARCODE,
        gid=f"junction:{junction.junction_id}:barcode-left-path",
    )
    draw_component_path(
        axis,
        [STEM_RIGHT_X, STEM_RIGHT_X],
        [stem_top, TOP_Y],
        color=BARCODE,
        gid=f"junction:{junction.junction_id}:barcode-right-path",
    )
    if right_context:
        draw_component_path(
            axis,
            [STEM_RIGHT_X, len(right_context)],
            [TOP_Y, TOP_Y],
            color=DOMAIN,
            gid=f"junction:{junction.junction_id}:right-target-top",
        )
        draw_component_path(
            axis,
            [0.0, len(right_context)],
            [BOTTOM_Y, BOTTOM_Y],
            color=DOMAIN,
            gid=f"junction:{junction.junction_id}:right-target-bottom",
        )
    draw_component_path(
        axis,
        [toehold_x + 0.34, 0.0],
        [BOTTOM_Y, BOTTOM_Y],
        color=TOEHOLD,
        gid=f"junction:{junction.junction_id}:toehold-bottom-path",
    )

    for sequence, start_x, y, role in (
        (left_context, left_x, TOP_Y, "left-top"),
        (toehold, toehold_x, TOP_Y, "toehold-top"),
        (right_context, right_x, TOP_Y, "right-top"),
        (left_bottom, left_x, BOTTOM_Y, "left-bottom"),
        (toehold_bottom, toehold_x, BOTTOM_Y, "toehold-bottom"),
        (right_bottom, right_x, BOTTOM_Y, "right-bottom"),
    ):
        draw_base_run(
            axis,
            sequence,
            start_x=start_x,
            start_y=y,
            delta_x=1.0,
            delta_y=0.0,
            gid_prefix=f"junction:{junction.junction_id}:{role}",
            fontsize=base_fontsize,
        )

    draw_base_run(
        axis,
        junction.barcode,
        start_x=STEM_LEFT_X,
        start_y=STEM_START_Y,
        delta_x=0.0,
        delta_y=1.0,
        gid_prefix=f"junction:{junction.junction_id}:barcode-b",
        fontsize=base_fontsize,
    )
    draw_base_run(
        axis,
        junction.barcode_complement,
        start_x=STEM_RIGHT_X,
        start_y=stem_top,
        delta_x=0.0,
        delta_y=-1.0,
        gid_prefix=f"junction:{junction.junction_id}:barcode-b-star",
        fontsize=base_fontsize,
    )

    target_pairs = [
        ((start + base + 0.5, TOP_Y - 0.48), (start + base + 0.5, BOTTOM_Y + 0.48))
        for start, length in (
            (left_x, len(left_context)),
            (toehold_x, len(toehold)),
            (right_x, len(right_context)),
        )
        for base in range(length)
    ]
    add_pairs(axis, target_pairs, gid=f"junction:{junction.junction_id}:target-pairs")
    barcode_pairs = [
        (
            ((STEM_LEFT_X + 0.32), STEM_START_Y + base + 0.5),
            ((STEM_RIGHT_X - 0.32), STEM_START_Y + base + 0.5),
        )
        for base in range(len(junction.barcode))
    ]
    add_pairs(axis, barcode_pairs, gid=f"junction:{junction.junction_id}:barcode-pairs")

    add_nick(axis, x=toehold_x, y=BOTTOM_Y, gid=f"junction:{junction.junction_id}:nick")
    axis.text(toehold_x, BOTTOM_Y - 1.0, "nick", fontsize=7.0, color=MUTED, ha="center", va="top")
    axis.text(
        (toehold_x + STEM_LEFT_X) / 2,
        TOP_Y + 1.0,
        f"t{index + 1}",
        fontsize=7.5,
        color=TOEHOLD_DARK,
        ha="center",
    )
    axis.text(
        toehold_x / 2,
        BOTTOM_Y - 0.9,
        f"t{index + 1}*",
        fontsize=7.5,
        color=TOEHOLD_DARK,
        ha="center",
        va="top",
    )
    axis.text(
        STEM_LEFT_X - 1.0,
        stem_top - 0.2,
        f"b{index + 1}",
        fontsize=7.5,
        color=BARCODE_DARK,
        ha="right",
    )
    axis.text(
        STEM_RIGHT_X + 1.0,
        stem_top - 0.2,
        f"b{index + 1}*",
        fontsize=7.5,
        color=BARCODE_DARK,
        ha="left",
    )
    axis.text(STEM_LEFT_X, stem_top + 0.8, "3′", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    axis.text(STEM_RIGHT_X, stem_top + 0.8, "5′", fontsize=7.5, color=MUTED, ha="center", va="bottom")

    if geometry.left_is_terminal:
        axis.text(left_x - 0.8, TOP_Y, "5′", fontsize=7.5, color=MUTED, ha="right", va="center")
        axis.text(left_x - 0.8, BOTTOM_Y, "3′", fontsize=7.5, color=MUTED, ha="right", va="center")
    else:
        add_break(axis, x=left_x, y=TOP_Y, gid=f"junction:{junction.junction_id}:left-top-break")
        add_break(axis, x=left_x, y=BOTTOM_Y, gid=f"junction:{junction.junction_id}:left-bottom-break")

    right_label_x = len(right_context) + 0.8
    if geometry.right_is_terminal:
        axis.text(right_label_x, TOP_Y, "3′", fontsize=7.5, color=MUTED, va="center")
        axis.text(right_label_x, BOTTOM_Y, "5′", fontsize=7.5, color=MUTED, va="center")
    else:
        add_break(
            axis,
            x=float(len(right_context)),
            y=TOP_Y,
            gid=f"junction:{junction.junction_id}:right-top-break",
        )
        add_break(
            axis,
            x=float(len(right_context)),
            y=BOTTOM_Y,
            gid=f"junction:{junction.junction_id}:right-bottom-break",
        )


__all__ = ["draw_junction_detail", "junction_detail_base_glyph_count"]
