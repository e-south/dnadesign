"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/fixtures.py

CLI fixtures for tetO PWM trim review-output tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def fake_review_output_result(**kwargs: object) -> object:
    out_dir = Path(kwargs["out_dir"])

    class Result:
        pass

    result = Result()
    result.review_root = out_dir
    result.pwm_triptych_svg = out_dir / "reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.svg"
    result.pwm_triptych_png = out_dir / "reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png"
    result.sequence_montage_mp4 = out_dir / "reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4"
    result.sequence_montage_manifest = out_dir / "reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.manifest.json"
    result.handoff_tsv = out_dir / "reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv"
    result.handoff_markdown = out_dir / "reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md"
    result.benchling_genbank_dir = out_dir / "benchling_genbank"
    result.benchling_genbank_index = out_dir / "reviews/handoff/teto_pwm_trim_rescue_v1.benchling_genbank.tsv"
    result.benchling_genbank_count = 6
    result.review_manifest_path = out_dir / "reviews/review_manifest.json"
    result.sequence_row_count = 9
    result.handoff_verified_count = 9
    return result


def review_outputs_args(
    study_dir: Path,
    materialized_root: Path,
    *,
    output_format: str | None = None,
) -> list[str]:
    args = [
        "review-outputs",
        "--study-dir",
        study_dir.as_posix(),
        "--materialized-root",
        materialized_root.as_posix(),
    ]
    if output_format is not None:
        args.extend(["--format", output_format])
    return args
