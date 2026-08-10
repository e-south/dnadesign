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
    DOMAIN,
    DOMAIN_DARK,
    INK,
    MUTED,
    PAIR,
    TOEHOLD,
    TOEHOLD_DARK,
    fragment_order_lengths,
    length_summary,
    safe_identifier,
    selected_ids,
)
from .primitives import draw_base_run, draw_segmented_strand

MAX_FRAGMENTS = 18
MAX_ROW_BASES = 140


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
        top_spans.append((0, len(previous.barcode), BARCODE, f"b{index}*"))
        bottom_spans.append((0, len(previous.toehold), TOEHOLD, f"t{index}*"))
    top_spans.append((top_domain_start, top_domain_start + domain_length, DOMAIN, "target"))
    bottom_spans.append((bottom_domain_start, bottom_domain_start + domain_length, DOMAIN, "target"))
    if following is not None:
        start = top_domain_start + domain_length
        top_spans.extend(
            (
                (start, start + len(following.toehold), TOEHOLD, f"t{index + 1}"),
                (
                    start + len(following.toehold),
                    start + len(following.toehold) + len(following.barcode),
                    BARCODE,
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


def validate_fragment_rows(review: ThreeWayJunctionReviewV1, indices: tuple[int, ...], *, renderer: str) -> None:
    for index in indices:
        _, strand, bottom, top_offset, bottom_offset, *_ = _row_geometry(review, index)
        width = max(top_offset + len(strand.barcode_bearing_sequence_5to3), bottom_offset + len(bottom))
        if width > MAX_ROW_BASES:
            raise SchemaError(
                f"{renderer} fragment {strand.fragment_id!r} spans {width} displayed bases; "
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
            fontsize=6.8,
            color=_component_text_color(color),
        )


def _draw_fragment(
    axis, review, *, index: int, y: float, base_step: float, maximum_width: int, fontsize: float
) -> None:
    fragment, strand, bottom, top_offset, bottom_offset, paired_start, paired_length, top_spans, bottom_spans = (
        _row_geometry(review, index)
    )
    width = max(top_offset + len(strand.barcode_bearing_sequence_5to3), bottom_offset + len(bottom))
    row_start = 0.16 + (maximum_width - width) * base_step / 2
    top_start = row_start + top_offset * base_step
    bottom_start = row_start + bottom_offset * base_step
    top_y, bottom_y = y, y - 0.25
    strand_height = 0.13
    draw_segmented_strand(
        axis,
        start_x=top_start,
        center_y=top_y,
        base_step=base_step,
        length=len(strand.barcode_bearing_sequence_5to3),
        segments=tuple((start, end, color) for start, end, color, _label in top_spans),
        height=strand_height,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:top-strand",
    )
    draw_segmented_strand(
        axis,
        start_x=bottom_start,
        center_y=bottom_y,
        base_step=base_step,
        length=len(bottom),
        segments=tuple((start, end, color) for start, end, color, _label in bottom_spans),
        height=strand_height,
        gid_prefix=f"junction-annealed:{fragment.fragment_id}:bottom-strand",
    )
    _draw_span_labels(axis, spans=top_spans, start_x=top_start, base_step=base_step, y=top_y + 0.095, above=True)
    _draw_span_labels(
        axis,
        spans=bottom_spans,
        start_x=bottom_start,
        base_step=base_step,
        y=bottom_y - 0.095,
        above=False,
    )
    pair_segments = [
        (
            (row_start + (base + 0.5) * base_step, top_y - 0.07),
            (row_start + (base + 0.5) * base_step, bottom_y + 0.07),
        )
        for base in range(paired_start, paired_start + paired_length)
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
    axis.text(top_start - 0.008, top_y, "5′", ha="right", va="center", fontsize=7.2, color=MUTED)
    axis.text(
        top_start + len(strand.barcode_bearing_sequence_5to3) * base_step + 0.004,
        top_y,
        "3′",
        va="center",
        fontsize=7.2,
        color=MUTED,
    )
    axis.text(bottom_start - 0.008, bottom_y, "3′", ha="right", va="center", fontsize=7.2, color=MUTED)
    axis.text(bottom_start + len(bottom) * base_step + 0.004, bottom_y, "5′", va="center", fontsize=7.2, color=MUTED)
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
        fontsize=7.2,
        fontweight="semibold",
        color=INK,
    )


def draw_annealed_panel(axis, review: ThreeWayJunctionReviewV1, indices: tuple[int, ...], *, height: float) -> None:
    axis.set_gid("junction-annealed-fragments:map")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, height)
    axis.axis("off")
    rows = tuple(_row_geometry(review, index) for index in indices)
    maximum_width = max(max(row[3] + len(row[1].barcode_bearing_sequence_5to3), row[4] + len(row[2])) for row in rows)
    base_step = 0.80 / maximum_width
    base_fontsize = min(7.8, max(5.8, 820 / maximum_width))
    count = len(indices)
    axis.text(
        0.015,
        height - 0.10,
        (
            f"{count} selected fragment {'pair is' if count == 1 else 'pairs are'} "
            f"expected to anneal for {safe_identifier(review.target.target_id)}"
        ),
        fontsize=15.0,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    lengths = fragment_order_lengths(review, indices)
    axis.text(
        0.015,
        height - 0.43,
        (
            f"The {len(review.target.sequence_5to3)} bp target contributes "
            f"{len(lengths)} fragment oligos spanning {length_summary(lengths)}"
        ),
        fontsize=9.0,
        color=MUTED,
        va="top",
    )
    y = height - 1.02
    for index in indices:
        _draw_fragment(
            axis,
            review,
            index=index,
            y=y,
            base_step=base_step,
            maximum_width=maximum_width,
            fontsize=base_fontsize,
        )
        y -= 0.88
    axis.text(
        0.015,
        0.07,
        "Base pairing is sequence-derived and does not establish thermodynamic or experimental success",
        fontsize=8.0,
        color=MUTED,
        va="bottom",
    )
