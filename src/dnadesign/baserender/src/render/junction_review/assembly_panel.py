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
from .assembly_geometry import AssemblyLayout
from .assembly_stages import draw_orders_stage
from .foundation import INK, MUTED, STAGE_TITLE_FONTSIZE, safe_identifier
from .input_panel import draw_input_target
from .preligation_panel import draw_preligation_stage
from .product_panel import draw_expected_pcr_product


def _stage_title(axis, y: float, text: str, *, gid: str) -> None:
    artist = axis.text(0.5, y, text, fontsize=STAGE_TITLE_FONTSIZE, color=INK, ha="center", va="center")
    artist.set_gid(bounded_svg_gid(gid))


def _transition(axis, y: float, *, gid: str) -> None:
    artist = axis.text(0.5, y, "↓", fontsize=20.0, color=MUTED, ha="center", va="center")
    artist.set_gid(bounded_svg_gid(gid))


def draw_assembly_process(axis, review: ThreeWayJunctionReviewV1, *, layout: AssemblyLayout) -> None:
    """Draw the submitted target, oligos, expected 3WJ state, and PCR duplex."""

    axis.set_gid("junction-three-way-assembly:assembly")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, layout.height)
    axis.axis("off")
    target_id = safe_identifier(review.target.target_id)
    title = axis.text(
        0.5,
        layout.title_y,
        f"Oligo plan for {target_id}",
        fontsize=20.0,
        fontweight="semibold",
        color=INK,
        ha="center",
        va="top",
    )
    title.set_gid(bounded_svg_gid("junction-three-way-assembly:title"))
    _stage_title(
        axis,
        layout.input_title_y,
        "Input target sequence",
        gid="junction-three-way-assembly:input:title",
    )
    draw_input_target(axis, review, y=layout.input_first_y, layout=layout)
    _transition(
        axis,
        layout.fragmentation_transition_y,
        gid="junction-three-way-assembly:transition:fragmentation",
    )
    _stage_title(
        axis,
        layout.orders_title_y,
        "Fragment oligos encode the target",
        gid="junction-three-way-assembly:orders:title",
    )
    draw_orders_stage(axis, review, y=layout.orders_first_y, layout=layout)
    _transition(
        axis,
        layout.annealing_transition_y,
        gid="junction-three-way-assembly:transition:annealing",
    )
    _stage_title(
        axis,
        layout.preligation_title_y,
        "Annealing forms pre-ligation junctions",
        gid="junction-three-way-assembly:three-way:title",
    )
    draw_preligation_stage(
        axis,
        review,
        first_y=layout.preligation_first_y,
        layout=layout,
    )
    _transition(
        axis,
        layout.recovery_transition_y,
        gid="junction-three-way-assembly:transition:recovery",
    )
    _stage_title(
        axis,
        layout.product_title_y,
        "PCR yields the expected linear duplex",
        gid="junction-three-way-assembly:product:title",
    )
    draw_expected_pcr_product(axis, review, first_y=layout.product_first_y, layout=layout)


__all__ = ["draw_assembly_process"]
