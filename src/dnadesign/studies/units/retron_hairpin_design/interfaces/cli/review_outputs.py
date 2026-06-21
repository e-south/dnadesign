"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/review_outputs.py

CLI command for Retron hairpin review-output generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...compiler.exceptions import RetronMsdCompilerError
from ...review_outputs.service import generate_teto_pwm_trim_rescue_review_outputs
from .io import emit, exit_with_error, format_option
from .messages import review_outputs_next_step


def review_outputs_command(
    deliverable_plan: Path | None = typer.Option(
        None,
        "--deliverable-plan",
        help="Retron workbench deliverable plan. Defaults to the tetO trim rescue v1 plan.",
    ),
    materialized_root: Path | None = typer.Option(
        None,
        "--materialized-root",
        help=(
            "Materialized single-unit sequence bundle root. Defaults to "
            "workbench/outputs/teto_pwm_trim_rescue_v1/materialized under --study-dir."
        ),
    ),
    study_dir: Path = typer.Option(
        Path("docs/studies/retron_hairpin_design"),
        "--study-dir",
        help="Retron hairpin study directory.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Review-output root. Defaults to workbench/outputs/teto_pwm_trim_rescue_v1 under --study-dir.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = format_option(output_format)
    try:
        resolved_plan = deliverable_plan or study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml"
        resolved_output_root = out_dir or study_dir / "workbench" / "outputs" / "teto_pwm_trim_rescue_v1"
        resolved_materialized_root = materialized_root or resolved_output_root / "materialized"
        result = generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=resolved_plan,
            materialized_root=resolved_materialized_root,
            out_dir=resolved_output_root,
        )
    except (RetronMsdCompilerError, OSError, ValueError) as exc:
        exit_with_error(exc, output_format=format_norm)
    emit(
        {
            "status": "ok",
            "output_dir": str(result.review_root),
            "pwm_triptych_svg": str(result.pwm_triptych_svg),
            "pwm_triptych_png": str(result.pwm_triptych_png),
            "sequence_montage_mp4": str(result.sequence_montage_mp4),
            "sequence_montage_manifest": str(result.sequence_montage_manifest),
            "handoff_tsv": str(result.handoff_tsv),
            "handoff_markdown": str(result.handoff_markdown),
            "review_manifest_path": str(result.review_manifest_path),
            "record_count": result.sequence_row_count,
            "handoff_verified_count": result.handoff_verified_count,
            "next_step": review_outputs_next_step(result.review_root),
        },
        output_format=format_norm,
    )


__all__ = ["review_outputs_command"]
