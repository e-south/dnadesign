"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_three_way_detail.py

Nucleotide-level geometry for selected Junction three-way interfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from .junction_review_common import INK, MUTED, PAIR, draw_base_run, junction_color, safe_identifier

_CONTEXT_BASES = 6


def _add_backbone(axis, xs, ys, *, gid: str, color: str = INK) -> None:
    line = Line2D(xs, ys, color=color, linewidth=1.1, solid_capstyle="round", zorder=2)
    line.set_gid(gid)
    axis.add_line(line)


def _add_pairs(axis, segments, *, gid: str) -> None:
    collection = LineCollection(segments, colors=PAIR, linewidths=0.55, zorder=1)
    collection.set_gid(gid)
    axis.add_collection(collection)


def _add_break(axis, *, x: float, y: float, gid: str) -> None:
    """Mark a cropped strand without implying a physical terminus."""

    for index, offset in enumerate((-0.012, 0.012)):
        xs = (x + offset - 0.009, x + offset + 0.009)
        ys = (y - 0.045, y + 0.045)
        axis.add_line(Line2D(xs, ys, color="white", linewidth=3.2, zorder=4))
        mark = Line2D(xs, ys, color=INK, linewidth=0.9, zorder=5)
        mark.set_gid(f"{gid}:{index}")
        axis.add_line(mark)


def junction_detail_base_glyph_count(review: ThreeWayJunctionReviewV1, index: int) -> int:
    """Count the per-base text artists required by one local detail view."""

    junction = review.geometry.junctions[index]
    left_fragment = review.geometry.fragments[index]
    right_fragment = review.geometry.fragments[index + 1]
    left_bases = min(_CONTEXT_BASES, junction.toehold_span.start - left_fragment.domain_span.start)
    right_bases = min(_CONTEXT_BASES, right_fragment.domain_span.end - junction.toehold_span.end)
    return 2 * (left_bases + len(junction.toehold) + right_bases + len(junction.barcode))


def draw_junction_detail(axis, review: ThreeWayJunctionReviewV1, index: int) -> None:
    """Draw one exact, sequence-derived three-arm interface."""

    junction = review.geometry.junctions[index]
    left_fragment = review.geometry.fragments[index]
    right_fragment = review.geometry.fragments[index + 1]
    target = review.target.sequence_5to3
    left_start = max(left_fragment.domain_span.start, junction.toehold_span.start - _CONTEXT_BASES)
    right_end = min(right_fragment.domain_span.end, junction.toehold_span.end + _CONTEXT_BASES)
    left_is_terminal = left_start == 0
    right_is_terminal = right_end == len(target)
    left_context = target[left_start : junction.toehold_span.start]
    right_context = target[junction.toehold_span.end : right_end]
    toehold = junction.toehold
    toehold_bottom = junction.toehold_complement[::-1]
    left_bottom = left_context.translate(str.maketrans("ACGT", "TGCA"))
    right_bottom = right_context.translate(str.maketrans("ACGT", "TGCA"))
    color = junction_color(index)

    axis.set_gid(f"junction-three-way-assembly:{junction.junction_id}:detail")
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.axis("off")
    axis.text(
        -0.96,
        0.93,
        f"J{index + 1:02d} · F{index + 1:02d} → F{index + 2:02d}",
        fontsize=7.0,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    axis.text(
        0.96,
        0.93,
        f"target bp {junction.toehold_span.start + 1}–{junction.toehold_span.end}",
        fontsize=5.5,
        color=MUTED,
        ha="right",
        va="top",
    )

    total_left = len(left_context) + len(toehold)
    horizontal_step = min(0.075, 0.72 / max(total_left, len(right_context), 1))
    node_x = 0.0
    top_y, bottom_y = 0.16, -0.12
    left_x = node_x - total_left * horizontal_step
    toehold_x = node_x - len(toehold) * horizontal_step
    right_x = node_x + 0.08

    _add_backbone(
        axis,
        [left_x, node_x - 0.035, node_x - 0.035],
        [top_y, top_y, 0.74],
        gid=f"junction:{junction.junction_id}:left-and-barcode-arm",
    )
    _add_backbone(
        axis,
        [node_x + 0.115, node_x + 0.115, right_x + len(right_context) * horizontal_step],
        [0.74, top_y, top_y],
        gid=f"junction:{junction.junction_id}:barcode-and-right-arm",
    )
    _add_backbone(
        axis,
        [left_x, toehold_x - 0.012],
        [bottom_y, bottom_y],
        gid=f"junction:{junction.junction_id}:left-complement-arm",
        color="#4B5563",
    )
    _add_backbone(
        axis,
        [toehold_x + 0.012, right_x + len(right_context) * horizontal_step],
        [bottom_y, bottom_y],
        gid=f"junction:{junction.junction_id}:right-complement-arm",
        color="#4B5563",
    )

    for y in (top_y, bottom_y):
        axis.add_patch(
            Rectangle(
                (toehold_x - 0.01, y - 0.055),
                len(toehold) * horizontal_step + 0.02,
                0.11,
                facecolor=color,
                edgecolor="none",
                alpha=0.24,
                zorder=0,
            )
        )
    stem_height = 0.50
    for x in (node_x - 0.035, node_x + 0.055):
        axis.add_patch(Rectangle((x, top_y), 0.06, stem_height, facecolor=color, edgecolor="none", alpha=0.22))

    for sequence, start_x, y, role in (
        (left_context, left_x, top_y, "left-top"),
        (toehold, toehold_x, top_y, "toehold-top"),
        (right_context, right_x, top_y, "right-top"),
        (left_bottom, left_x, bottom_y, "left-bottom"),
        (toehold_bottom, toehold_x, bottom_y, "toehold-bottom"),
        (right_bottom, right_x, bottom_y, "right-bottom"),
    ):
        draw_base_run(
            axis,
            sequence,
            start_x=start_x,
            start_y=y,
            delta_x=horizontal_step,
            delta_y=0,
            gid_prefix=f"junction:{junction.junction_id}:{role}",
            fontsize=5.5,
        )

    barcode_y = top_y + 0.08
    barcode_step = stem_height / max(len(junction.barcode), 1)
    draw_base_run(
        axis,
        junction.barcode,
        start_x=node_x,
        start_y=barcode_y,
        delta_x=0,
        delta_y=barcode_step,
        gid_prefix=f"junction:{junction.junction_id}:barcode-b",
        fontsize=5.3,
    )
    draw_base_run(
        axis,
        junction.barcode_complement,
        start_x=node_x + 0.08,
        start_y=barcode_y + stem_height,
        delta_x=0,
        delta_y=-barcode_step,
        gid_prefix=f"junction:{junction.junction_id}:barcode-b-star",
        fontsize=5.3,
    )

    target_pairs = []
    for start, length in (
        (left_x, len(left_context)),
        (toehold_x, len(toehold)),
        (right_x, len(right_context)),
    ):
        target_pairs.extend(
            (
                (start + (base + 0.5) * horizontal_step, top_y - 0.035),
                (start + (base + 0.5) * horizontal_step, bottom_y + 0.035),
            )
            for base in range(length)
        )
    _add_pairs(axis, target_pairs, gid=f"junction:{junction.junction_id}:target-pairs")
    barcode_pairs = [
        (
            (node_x + 0.022, barcode_y + (base + 0.5) * barcode_step),
            (node_x + 0.058, barcode_y + (base + 0.5) * barcode_step),
        )
        for base in range(len(junction.barcode))
    ]
    _add_pairs(axis, barcode_pairs, gid=f"junction:{junction.junction_id}:barcode-pairs")

    nick = Line2D(
        [toehold_x - 0.012, toehold_x + 0.012],
        [bottom_y - 0.045, bottom_y + 0.045],
        color="#111827",
        linewidth=1.1,
    )
    nick.set_gid(f"junction:{junction.junction_id}:nick")
    axis.add_line(nick)
    axis.text(toehold_x, bottom_y - 0.14, "nick", fontsize=5.2, color=MUTED, ha="center", va="top")
    axis.text((toehold_x + node_x) / 2, top_y + 0.09, f"t{index + 1}", fontsize=5.5, color=color, ha="center")
    axis.text(
        (toehold_x + node_x) / 2,
        bottom_y - 0.09,
        f"t{index + 1}*",
        fontsize=5.5,
        color=color,
        ha="center",
        va="top",
    )
    axis.text(node_x - 0.05, 0.67, f"b{index + 1}", fontsize=5.5, color=color, ha="right")
    axis.text(node_x + 0.13, 0.67, f"b{index + 1}*", fontsize=5.5, color=color, ha="left")
    axis.text(node_x, 0.77, "3′", fontsize=5.3, color=MUTED, ha="center", va="bottom")
    axis.text(node_x + 0.08, 0.77, "5′", fontsize=5.3, color=MUTED, ha="center", va="bottom")
    if left_is_terminal:
        axis.text(left_x - 0.03, top_y, "5′", fontsize=5.3, color=MUTED, ha="right", va="center")
        axis.text(left_x - 0.03, bottom_y, "3′", fontsize=5.3, color=MUTED, ha="right", va="center")
    else:
        _add_break(axis, x=left_x, y=top_y, gid=f"junction:{junction.junction_id}:left-top-break")
        _add_break(axis, x=left_x, y=bottom_y, gid=f"junction:{junction.junction_id}:left-bottom-break")
    right_label_x = right_x + len(right_context) * horizontal_step + 0.03
    if right_is_terminal:
        axis.text(right_label_x, top_y, "3′", fontsize=5.3, color=MUTED, va="center")
        axis.text(right_label_x, bottom_y, "5′", fontsize=5.3, color=MUTED, va="center")
    else:
        right_boundary = right_x + len(right_context) * horizontal_step
        _add_break(axis, x=right_boundary, y=top_y, gid=f"junction:{junction.junction_id}:right-top-break")
        _add_break(axis, x=right_boundary, y=bottom_y, gid=f"junction:{junction.junction_id}:right-bottom-break")
    axis.text(-0.96, -0.86, safe_identifier(junction.junction_id), fontsize=5.0, color=MUTED, va="bottom")


__all__ = ["draw_junction_detail", "junction_detail_base_glyph_count"]
