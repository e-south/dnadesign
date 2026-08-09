"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_annealed_review.py

Exact annealed-fragment section for Junction review figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from .junction_nucleotide_drawing import (
    BARCODE,
    FRAGMENT_A,
    FRAGMENT_B,
    INK,
    MUTED,
    TOEHOLD,
    ColoredSpan,
    draw_aligned_fragment,
)
from .junction_pairing_layout import expanded_annealed_fragments


def _annealed_spans(
    review: ThreeWayJunctionReviewV1,
    *,
    index: int,
    domain_length: int,
) -> tuple[tuple[ColoredSpan, ...], tuple[ColoredSpan, ...], int, int]:
    previous = None if index == 0 else review.geometry.junctions[index - 1]
    following = None if index == len(review.strands) - 1 else review.geometry.junctions[index]
    top_spans: list[ColoredSpan] = []
    bottom_spans: list[ColoredSpan] = []
    top_domain_start = 0
    bottom_domain_start = 0
    if previous is not None:
        top_spans.append(ColoredSpan(0, len(previous.barcode), BARCODE))
        bottom_spans.append(ColoredSpan(0, len(previous.toehold), TOEHOLD))
        top_domain_start = len(previous.barcode)
        bottom_domain_start = len(previous.toehold)
    fragment_color = FRAGMENT_A if index % 2 == 0 else FRAGMENT_B
    top_spans.append(ColoredSpan(top_domain_start, top_domain_start + domain_length, fragment_color))
    bottom_spans.append(ColoredSpan(bottom_domain_start, bottom_domain_start + domain_length, fragment_color))
    if following is not None:
        start = top_domain_start + domain_length
        top_spans.extend(
            (
                ColoredSpan(start, start + len(following.toehold), TOEHOLD),
                ColoredSpan(
                    start + len(following.toehold),
                    start + len(following.toehold) + len(following.barcode),
                    BARCODE,
                ),
            )
        )
    return tuple(top_spans), tuple(bottom_spans), top_domain_start, bottom_domain_start


def draw_annealed_fragments(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    """Draw all annealed fragment pairs when the exact view fits one row each."""

    axis.text(0.018, y, "2  Annealed fragment pairs", fontsize=8.5, fontweight="semibold", color=INK, va="top")
    if not expanded_annealed_fragments(review):
        axis.text(
            0.20,
            y,
            (
                f"Compact view for {len(review.strands)} fragments; exact order strands, "
                "junction pairs, and recovered duplex remain below."
            ),
            fontsize=5.8,
            color=MUTED,
            va="top",
        )
        return y - 0.34
    axis.text(
        0.20,
        y,
        "gray = target domain · blue = t/t* · green = b/b* · lines = Watson–Crick pairs",
        fontsize=5.8,
        color=MUTED,
        va="top",
    )
    y -= 0.27
    target = review.target.sequence_5to3
    for index, (fragment, strand) in enumerate(zip(review.geometry.fragments, review.strands, strict=True)):
        domain = target[fragment.domain_span.start : fragment.domain_span.end]
        top_spans, bottom_spans, top_domain_start, bottom_domain_start = _annealed_spans(
            review,
            index=index,
            domain_length=len(domain),
        )
        aligned_start = max(top_domain_start, bottom_domain_start)
        axis.text(
            0.018,
            y,
            (
                f"F{index + 1:02d} · {fragment.role} · paired target bp "
                f"{fragment.domain_span.start + 1}–{fragment.domain_span.end}"
            ),
            fontsize=5.8,
            fontweight="semibold",
            color=INK,
            va="top",
        )
        y -= 0.14
        y = draw_aligned_fragment(
            axis,
            top=strand.barcode_bearing_sequence_5to3,
            bottom_aligned=strand.complement_sequence_5to3[::-1],
            top_offset=aligned_start - top_domain_start,
            bottom_offset=aligned_start - bottom_domain_start,
            paired_start=aligned_start,
            paired_length=len(domain),
            top_spans=top_spans,
            bottom_spans=bottom_spans,
            y=y,
        )
        y -= 0.05
    return y


__all__ = ["draw_annealed_fragments"]
