"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/triptych.py

PWM trim triptych rendering for tetO trim rescue review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...compiler.exceptions import RetronMsdCompilerError
from ..contracts.plan import TetoReviewPlan
from .logo import PwmLogoColumn, PwmLogoLayer, write_pwm_logo_triptych
from .retention import PwmMotifOccurrence, load_meme_information_bits, load_meme_probability_matrix

UNIFORM_ZERO_INFORMATION_ROW = [0.25, 0.25, 0.25, 0.25]


def render_pwm_triptych(plan: TetoReviewPlan, *, out_dir: Path) -> tuple[Path, Path]:
    matrix = load_meme_probability_matrix(plan.meme_pwm_path)
    motif_bits = load_meme_information_bits(plan.meme_pwm_path)
    for occurrence in plan.motif_occurrences:
        if len(matrix) != occurrence.width:
            raise RetronMsdCompilerError(
                f"Expected {occurrence.width} tetR PWM positions for {occurrence.motif_instance_id}, "
                f"found {len(matrix)} in {plan.meme_pwm_path}"
            )
    if len(plan.parent_payload_sequence) != 19:
        raise RetronMsdCompilerError(
            "Expected the bidirectional TetR parent payload sequence to contain 19 nt for the trim triptych; "
            f"found {len(plan.parent_payload_sequence)}"
        )
    columns = _build_logo_columns(
        plan.parent_payload_sequence,
        motif_bits,
        motif_occurrences=plan.motif_occurrences,
    )
    logo_layers = tuple(
        _build_logo_layer(occurrence, matrix=matrix, parent_length=len(plan.parent_payload_sequence))
        for occurrence in plan.motif_occurrences
    )
    svg_path = out_dir / "reviews" / "pwm" / f"{plan.deliverable_plan_id}.pwm_trim_triptych.svg"
    png_path = svg_path.with_suffix(".png")
    write_pwm_logo_triptych(
        columns,
        parent_sequence=plan.parent_payload_sequence,
        logo_layers=logo_layers,
        panels=plan.pwm_panels,
        svg_path=svg_path,
        png_path=png_path,
        source_path=plan.meme_pwm_path,
    )
    return svg_path, png_path


def _build_logo_columns(
    parent_sequence: str,
    motif_bits: tuple[float, ...],
    *,
    motif_occurrences: tuple[PwmMotifOccurrence, ...],
) -> tuple[PwmLogoColumn, ...]:
    columns = []
    for position, parent_base in enumerate(parent_sequence):
        information_bits = sum(
            _information_bits_at_parent_position(position, occurrence, motif_bits) for occurrence in motif_occurrences
        )
        columns.append(
            PwmLogoColumn(
                parent_position_0=position,
                parent_base=parent_base,
                information_bits=information_bits,
                probabilities=dict(zip(("A", "C", "G", "T"), UNIFORM_ZERO_INFORMATION_ROW, strict=True)),
            )
        )
    return tuple(columns)


def _build_logo_layer(
    occurrence: PwmMotifOccurrence,
    *,
    matrix: tuple[tuple[float, float, float, float], ...],
    parent_length: int,
) -> PwmLogoLayer:
    padded = [tuple(float(value) for value in UNIFORM_ZERO_INFORMATION_ROW) for _index in range(parent_length)]
    visible_rows = matrix if occurrence.strand == "+" else tuple(reversed(matrix))
    for offset, row in enumerate(visible_rows):
        padded[occurrence.start_0 + offset] = tuple(float(value) for value in row)
    effect_matrix = tuple(padded) if occurrence.strand == "+" else tuple(reversed(padded))
    return PwmLogoLayer(
        motif_instance_id=occurrence.motif_instance_id,
        strand=occurrence.strand,
        start_0=occurrence.start_0,
        end_0=occurrence.end_0,
        occurrence_rank=occurrence.occurrence_rank,
        matrix=effect_matrix,
    )


def _information_bits_at_parent_position(
    position: int,
    occurrence: PwmMotifOccurrence,
    motif_bits: tuple[float, ...],
) -> float:
    motif_index = occurrence.end_0 - 1 - position if occurrence.strand == "-" else position - occurrence.start_0
    if motif_index < 0 or motif_index >= len(motif_bits):
        return 0.0
    return float(motif_bits[motif_index])


__all__ = ["render_pwm_triptych"]
