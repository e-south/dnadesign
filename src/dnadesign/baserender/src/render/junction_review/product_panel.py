"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/product_panel.py

Exact recovered-duplex rows for Junction assembly review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_geometry import PRODUCT_BASES_PER_WINDOW, PRODUCT_ROW_STEP, product_windows
from .assembly_stages import target_spans
from .foundation import MUTED, PAIR, PRIMER, STRAND_EDGE
from .primitives import draw_compact_base_run, draw_segmented_strand


def _product_spans(review: ThreeWayJunctionReviewV1):
    prefix = len(review.recovery.forward.five_prime_extension_5to3)
    target_end = prefix + len(review.target.sequence_5to3)
    spans = list(target_spans(review, offset=prefix))
    if prefix:
        spans.append((0, prefix, PRIMER))
    product_length = len(review.recovery.extended_top_sequence_5to3)
    if target_end < product_length:
        spans.append((target_end, product_length, PRIMER))
    return tuple(sorted(spans, key=lambda item: item[0]))


def _window_spans(spans, *, start: int, end: int):
    return tuple(
        (max(span_start, start) - start, min(span_end, end) - start, color)
        for span_start, span_end, color in spans
        if max(span_start, start) < min(span_end, end)
    )


def _draw_continuation(axis, *, x: float, y: float, side: str, gid: str) -> None:
    direction = -1 if side == "left" else 1
    for index, offset in enumerate((-0.004, 0.004)):
        line = Line2D(
            (x + offset - direction * 0.0025, x + offset + direction * 0.0025),
            (y - 0.10, y + 0.10),
            color=STRAND_EDGE,
            linewidth=0.9,
            zorder=4,
        )
        line.set_gid(bounded_svg_gid(f"{gid}:{index}"))
        axis.add_line(line)


def draw_recovered_product(axis, review: ThreeWayJunctionReviewV1, *, first_y: float) -> None:
    """Draw every expected product base and declared Watson-Crick edge."""

    sequence = review.recovery.extended_top_sequence_5to3
    complement = sequence.translate(str.maketrans("ACGT", "TGCA"))
    spans = _product_spans(review)
    left, right = 0.105, 0.965
    base_step = (right - left) / PRODUCT_BASES_PER_WINDOW
    for window in product_windows(len(sequence)):
        top_y = first_y - window.index * PRODUCT_ROW_STEP
        bottom_y = top_y - 0.22
        top = sequence[window.start : window.end]
        bottom = complement[window.start : window.end]
        local_spans = _window_spans(spans, start=window.start, end=window.end)
        for strand_y, role in ((top_y, "top"), (bottom_y, "bottom")):
            draw_segmented_strand(
                axis,
                start_x=left,
                center_y=strand_y,
                base_step=base_step,
                length=len(top),
                segments=local_spans,
                height=0.145,
                gid_prefix=f"junction-three-way-assembly:product:{role}:window:{window.index}:strand",
            )
        for strand, strand_y, role in ((top, top_y, "top"), (bottom, bottom_y, "bottom")):
            draw_compact_base_run(
                axis,
                strand,
                start_x=left,
                start_y=strand_y,
                delta_x=base_step,
                delta_y=0.0,
                gid_prefix=f"junction-three-way-assembly:product:{role}:window:{window.index}",
                fontsize=10.5,
            )
        pairs = LineCollection(
            [
                (
                    (left + (base + 0.5) * base_step, top_y - 0.075),
                    (left + (base + 0.5) * base_step, bottom_y + 0.075),
                )
                for base in range(len(top))
            ],
            colors=PAIR,
            linewidths=0.45,
            zorder=1,
        )
        pairs.set_gid(bounded_svg_gid(f"junction-three-way-assembly:product:window:{window.index}:pairs"))
        axis.add_collection(pairs)
        axis.text(
            left - 0.012,
            (top_y + bottom_y) / 2,
            f"bp {window.start + 1}–{window.end}",
            fontsize=8.0,
            color=MUTED,
            ha="right",
            va="center",
        )
        if window.index == 0:
            axis.text(left - 0.008, top_y, "5′", fontsize=9.0, color=MUTED, ha="right", va="center")
            axis.text(left - 0.008, bottom_y, "3′", fontsize=9.0, color=MUTED, ha="right", va="center")
        else:
            _draw_continuation(
                axis,
                x=left,
                y=(top_y + bottom_y) / 2,
                side="left",
                gid=f"junction-three-way-assembly:product:window:{window.index}:left-continuation",
            )
        end_x = left + len(top) * base_step
        if window.end == len(sequence):
            axis.text(end_x + 0.008, top_y, "3′", fontsize=9.0, color=MUTED, va="center")
            axis.text(end_x + 0.008, bottom_y, "5′", fontsize=9.0, color=MUTED, va="center")
        else:
            _draw_continuation(
                axis,
                x=end_x,
                y=(top_y + bottom_y) / 2,
                side="right",
                gid=f"junction-three-way-assembly:product:window:{window.index}:right-continuation",
            )


__all__ = ["draw_recovered_product"]
