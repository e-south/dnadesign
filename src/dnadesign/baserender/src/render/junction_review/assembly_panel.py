"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/assembly_panel.py

Composition for the Junction assembly-process review figure.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_stages import draw_orders_stage, draw_three_way_stage
from .foundation import INK, MUTED, fragment_order_lengths, length_summary, safe_identifier
from .product_panel import draw_recovered_product


def _stage_title(axis, y: float, text: str, *, gid: str) -> None:
    artist = axis.text(0.5, y, text, fontsize=13.5, fontweight="semibold", color=INK, ha="center", va="center")
    artist.set_gid(bounded_svg_gid(gid))


def _transition(axis, y: float, text: str, *, gid: str) -> None:
    artist = axis.text(0.5, y, text, fontsize=9.5, color=MUTED, ha="center", va="center")
    artist.set_gid(bounded_svg_gid(gid))


def draw_assembly_process(axis, review: ThreeWayJunctionReviewV1, *, height: float) -> None:
    """Draw separate oligos, the expected 3WJ state, and the exact recovered duplex."""

    axis.set_gid("junction-three-way-assembly:assembly")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, height)
    axis.axis("off")
    target_id = safe_identifier(review.target.target_id)
    lengths = fragment_order_lengths(review)
    axis.text(
        0.5,
        height - 0.10,
        f"The junction plan for {target_id} resolves each molecular state",
        fontsize=17.0,
        fontweight="semibold",
        color=INK,
        ha="center",
        va="top",
    )
    axis.text(
        0.5,
        height - 0.45,
        (
            f"The {len(review.target.sequence_5to3):,} bp target uses {len(review.geometry.fragments)} fragments and "
            f"{len(review.geometry.junctions)} junctions; its fragment oligos span {length_summary(lengths)}"
        ),
        fontsize=10.5,
        color=MUTED,
        ha="center",
        va="top",
    )
    axis.text(
        0.5,
        height - 0.72,
        (
            "Gray marks target sequence, amber marks toeholds, teal marks external barcodes, "
            "and lavender marks primer extensions"
        ),
        fontsize=9.0,
        color=MUTED,
        ha="center",
        va="top",
    )
    _stage_title(
        axis,
        height - 1.05,
        "The oligos remain separate before annealing",
        gid="junction-three-way-assembly:orders:title",
    )
    draw_orders_stage(axis, review, y=height - 1.47)
    _transition(
        axis,
        height - 2.10,
        "↓  modeled annealing",
        gid="junction-three-way-assembly:transition:annealing",
    )
    _stage_title(
        axis,
        height - 2.38,
        "The plan specifies an annealed pre-ligation state",
        gid="junction-three-way-assembly:three-way:title",
    )
    draw_three_way_stage(axis, review, y=height - 3.13)
    _transition(
        axis,
        height - 3.68,
        "↓  modeled ligation and PCR recovery",
        gid="junction-three-way-assembly:transition:recovery",
    )
    _stage_title(
        axis,
        height - 3.98,
        "The expected PCR product is a recovered duplex",
        gid="junction-three-way-assembly:product:title",
    )
    draw_recovered_product(axis, review, first_y=height - 4.43)
    axis.text(
        0.5,
        0.08,
        "This sequence-derived review does not establish annealing, ligation, amplification, or yield",
        fontsize=9.0,
        color=MUTED,
        ha="center",
        va="bottom",
    )


__all__ = ["draw_assembly_process"]
