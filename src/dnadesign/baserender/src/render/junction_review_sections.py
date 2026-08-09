"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review_sections.py

Method-stage sections for nucleotide-level Junction review figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from .junction_nucleotide_drawing import (
    BARCODE,
    FRAGMENT_A,
    FRAGMENT_B,
    INK,
    MUTED,
    PRIMER,
    TOEHOLD,
    base_x,
    draw_duplex,
    draw_sequence_rows,
)
from .junction_pairing_layout import sequence_chunks


def draw_stage_path(axis, *, y: float) -> float:
    """State the evidence flow without implying kinetic simulation."""

    labels = ("Oligos to order", "Annealed pairs", "3WJ interfaces", "Recovered duplex")
    positions = (0.018, 0.27, 0.52, 0.77)
    for index, (label, x) in enumerate(zip(labels, positions, strict=True), start=1):
        axis.text(x, y, f"{index}  {label}", fontsize=6.5, fontweight="semibold", color=INK, va="center")
        if index < len(labels):
            axis.text(x + 0.205, y, "→", fontsize=8.0, color=MUTED, va="center")
    return y - 0.28


def _product_intersections(start: int, end: int, review: ThreeWayJunctionReviewV1):
    spans = []
    target_offset = len(review.recovery.forward.five_prime_extension_5to3)
    target_end = target_offset + len(review.target.sequence_5to3)
    if start < target_offset:
        spans.append((start, min(end, target_offset), PRIMER, "5′ extension"))
    for fragment in review.geometry.fragments:
        left = max(start, target_offset + fragment.domain_span.start)
        right = min(end, target_offset + fragment.domain_span.end)
        if left < right:
            spans.append((left, right, FRAGMENT_A if fragment.index % 2 == 0 else FRAGMENT_B, f"F{fragment.index + 1}"))
    for junction in review.geometry.junctions:
        left = max(start, target_offset + junction.toehold_span.start)
        right = min(end, target_offset + junction.toehold_span.end)
        if left < right:
            spans.append((left, right, TOEHOLD, "t"))
    if end > target_end:
        spans.append((max(start, target_end), end, PRIMER, "3′ extension"))
    return spans


def draw_recovered_duplex(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    """Draw the exact recovered duplex, including declared primer extensions."""

    top = review.recovery.extended_top_sequence_5to3
    bottom_aligned = review.recovery.extended_bottom_sequence_5to3[::-1]
    axis.text(0.018, y, "4  Recovered duplex", fontsize=8.5, fontweight="semibold", color=INK, va="top")
    axis.text(
        0.18,
        y,
        f"{len(top)} bp product · {len(review.target.sequence_5to3)} bp target",
        fontsize=5.8,
        color=MUTED,
        va="top",
    )
    y -= 0.28
    for top_chunk, bottom_chunk in zip(sequence_chunks(top), sequence_chunks(bottom_aligned), strict=True):
        for left, right, color, label in _product_intersections(top_chunk.start, top_chunk.end, review):
            x = base_x(left - top_chunk.start) - 0.004
            width = max(0.008, base_x(right - top_chunk.start) - x)
            axis.add_patch(Rectangle((x, y + 0.015), width, 0.035, facecolor=color, edgecolor="none", alpha=0.9))
            axis.text(x, y + 0.06, label, fontsize=4.7, color=color if label == "t" else MUTED, va="bottom")
        y = draw_duplex(
            axis,
            top=top_chunk.sequence,
            bottom=bottom_chunk.sequence,
            y=y,
            coordinate_start=top_chunk.start,
        )
        y -= 0.10
    return y


def draw_junctions(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    """Draw every exact t/t-star and b/b-star interface."""

    axis.text(0.018, y, "3  Three-way-junction interfaces", fontsize=8.5, fontweight="semibold", color=INK, va="top")
    axis.text(
        0.24,
        y,
        "t/t* is target-derived; b/b* is the assigned barcode.",
        fontsize=5.8,
        color=MUTED,
        va="top",
    )
    y -= 0.27
    for index, junction in enumerate(review.geometry.junctions, start=1):
        axis.text(
            0.018,
            y,
            (
                f"J{index:02d} · F{index:02d} → F{index + 1:02d} · "
                f"target bp {junction.toehold_span.start + 1}–{junction.toehold_span.end}"
            ),
            fontsize=5.8,
            fontweight="semibold",
            color=INK,
            va="top",
        )
        y -= 0.16
        for kind, top, stored_complement, color in (
            ("toehold", junction.toehold, junction.toehold_complement, TOEHOLD),
            ("barcode", junction.barcode, junction.barcode_complement, BARCODE),
        ):
            top_chunks = sequence_chunks(top)
            aligned_bottom = stored_complement[::-1]
            bottom_chunks = sequence_chunks(aligned_bottom)
            for chunk_index, (top_chunk, bottom_chunk) in enumerate(zip(top_chunks, bottom_chunks, strict=True)):
                label = kind if chunk_index == 0 else f"{kind} cont."
                axis.text(0.018, y - 0.12, label, fontsize=5.4, color=color, va="center")
                y = draw_duplex(axis, top=top_chunk.sequence, bottom=bottom_chunk.sequence, y=y)
        y -= 0.04
    return y


def draw_oligo_orders(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    """Draw every exact fragment-oligo order sequence."""

    axis.text(0.018, y, "1  Oligos to order", fontsize=8.5, fontweight="semibold", color=INK, va="top")
    axis.text(0.16, y, "Every displayed strand is written 5′→3′.", fontsize=5.8, color=MUTED, va="top")
    y -= 0.26
    for index, strand in enumerate(review.strands, start=1):
        axis.text(
            0.018,
            y,
            f"F{index:02d} · {strand.role}",
            fontsize=5.7,
            fontweight="semibold",
            color=INK,
            va="top",
        )
        y -= 0.14
        y = draw_sequence_rows(
            axis,
            sequence=strand.barcode_bearing_sequence_5to3,
            y=y,
            label="barcode",
            color=BARCODE,
        )
        y = draw_sequence_rows(
            axis,
            sequence=strand.complement_sequence_5to3,
            y=y,
            label="complement",
        )
    return y


def draw_primers(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> float:
    """Draw the exact recovery-primer order sequences."""

    axis.text(
        0.018,
        y,
        f"Recovery primers · {review.recovery.mode}",
        fontsize=8.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    y -= 0.25
    y = draw_sequence_rows(
        axis,
        sequence=review.recovery.forward.order_sequence_5to3,
        y=y,
        label="forward",
        color=PRIMER,
    )
    y = draw_sequence_rows(
        axis,
        sequence=review.recovery.reverse.order_sequence_5to3,
        y=y,
        label="reverse",
        color=PRIMER,
    )
    return y


__all__ = [
    "draw_junctions",
    "draw_oligo_orders",
    "draw_primers",
    "draw_recovered_duplex",
    "draw_stage_path",
]
