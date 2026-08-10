"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/product_panel.py

Exact expected-PCR-duplex rows for Junction assembly review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_geometry import PRODUCT_ROW_STEP, AssemblyLayout, product_windows
from .assembly_stages import target_spans
from .foundation import (
    MOLECULAR_ANNOTATION_FONTSIZE,
    MUTED,
    PAIR,
    PRIMER_BINDING_SITE_DARK,
    PRIMER_EXTENSION,
)
from .primitives import draw_compact_base_run, draw_continuation, draw_segmented_strand, draw_terminus


def _product_spans(review: ThreeWayJunctionReviewV1):
    prefix = len(review.recovery.forward.five_prime_extension_5to3)
    target_end = prefix + len(review.target.sequence_5to3)
    spans = list(target_spans(review, offset=prefix))
    if prefix:
        spans.append((0, prefix, PRIMER_EXTENSION))
    product_length = len(review.recovery.extended_top_sequence_5to3)
    if target_end < product_length:
        spans.append((target_end, product_length, PRIMER_EXTENSION))
    return tuple(sorted(spans, key=lambda item: item[0]))


def _window_spans(spans, *, start: int, end: int):
    return tuple(
        (max(span_start, start) - start, min(span_end, end) - start, color)
        for span_start, span_end, color in spans
        if max(span_start, start) < min(span_end, end)
    )


def _draw_primer_binding_site_labels(axis, review, *, window, left: float, base_step: float, y: float) -> None:
    prefix = len(review.recovery.forward.five_prime_extension_5to3)
    for primer in (review.recovery.forward, review.recovery.reverse):
        start = prefix + primer.target_binding_span.start
        end = prefix + primer.target_binding_span.end
        if window.start <= start and end <= window.end:
            artist = axis.text(
                left + ((start + end) / 2 - window.start) * base_step,
                y,
                f"{primer.direction} primer-binding site · {end - start} nt",
                fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
                color=PRIMER_BINDING_SITE_DARK,
                ha="center",
                va="bottom",
            )
            artist.set_gid(bounded_svg_gid(f"junction-three-way-assembly:product:primer-site:{primer.direction}"))


def draw_expected_pcr_product(
    axis,
    review: ThreeWayJunctionReviewV1,
    *,
    first_y: float,
    layout: AssemblyLayout,
) -> None:
    """Draw every expected product base and declared Watson-Crick edge."""

    sequence = review.recovery.extended_top_sequence_5to3
    complement = sequence.translate(str.maketrans("ACGT", "TGCA"))
    spans = _product_spans(review)
    left, right = layout.product_left, layout.product_right
    base_step = (right - left) / layout.product_bases_per_row
    windows = product_windows(len(sequence), bases_per_row=layout.product_bases_per_row)
    for window in windows:
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
                fontsize=11.0,
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
        _draw_primer_binding_site_labels(axis, review, window=window, left=left, base_step=base_step, y=top_y + 0.18)
        if len(windows) > 1:
            coordinate = axis.text(
                left + len(top) * base_step / 2,
                top_y + 0.18,
                f"bp {window.start + 1}–{window.end}",
                fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
                color=MUTED,
                ha="center",
                va="bottom",
            )
            coordinate.set_gid(bounded_svg_gid(f"junction-three-way-assembly:product:window:{window.index}:coordinate"))
        if window.index == 0:
            for strand_y, text, role in ((top_y, "5′", "top-left"), (bottom_y, "3′", "bottom-left")):
                draw_terminus(
                    axis,
                    x=left - 0.008,
                    y=strand_y,
                    text=text,
                    ha="right",
                    gid=f"junction-three-way-assembly:product:terminus:{role}",
                )
        else:
            draw_continuation(
                axis,
                x=left,
                y=(top_y + bottom_y) / 2,
                side="left",
                gid=f"junction-three-way-assembly:product:window:{window.index}:left-continuation",
            )
        end_x = left + len(top) * base_step
        if window.end == len(sequence):
            for strand_y, text, role in ((top_y, "3′", "top-right"), (bottom_y, "5′", "bottom-right")):
                draw_terminus(
                    axis,
                    x=end_x + 0.008,
                    y=strand_y,
                    text=text,
                    ha="left",
                    gid=f"junction-three-way-assembly:product:terminus:{role}",
                )
        else:
            draw_continuation(
                axis,
                x=end_x,
                y=(top_y + bottom_y) / 2,
                side="right",
                gid=f"junction-three-way-assembly:product:window:{window.index}:right-continuation",
            )


__all__ = ["draw_expected_pcr_product"]
