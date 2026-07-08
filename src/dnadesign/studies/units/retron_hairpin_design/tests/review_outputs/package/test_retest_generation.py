"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_retest_generation.py

Tests for tetO-prior retest review-package generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_retron_hairpin_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import write_review_plan_fixture


def test_review_outputs_service_exposes_plan_driven_api_only() -> None:
    from dnadesign.studies.units.retron_hairpin_design import review_outputs
    from dnadesign.studies.units.retron_hairpin_design.review_outputs import service

    assert hasattr(review_outputs, "generate_retron_hairpin_review_outputs")
    assert not hasattr(review_outputs, "generate_teto_pwm_trim_rescue_review_outputs")
    assert not hasattr(service, "generate_teto_pwm_trim_rescue_review_outputs")


def test_teto_payload_prior_retest_review_outputs_generate_review_package(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    deliverable_plan_path = write_review_plan_fixture(
        tmp_path / "plan",
        repo_root=repo_root,
        deliverable_plan_id="teto_payload_trim_retest_v1",
    )
    materialized_root = write_fake_materialized_bundle(
        tmp_path / "materialized",
        repo_root=repo_root,
        design_set_id="teto_payload_trim_retest_v1",
    )
    out_dir = tmp_path / "workbench" / "outputs" / "teto_payload_trim_retest_v1"

    result = generate_retron_hairpin_review_outputs(
        deliverable_plan_path=deliverable_plan_path,
        materialized_root=materialized_root,
        out_dir=out_dir,
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert result.sequence_row_count == 4
    assert result.handoff_verified_count == 4
    assert result.pwm_triptych_svg == out_dir / "reviews" / "pwm" / "teto_payload_trim_retest_v1.pwm_trim_triptych.svg"
    assert result.handoff_tsv == out_dir / "reviews" / "handoff" / "teto_payload_trim_retest_v1.handoff.tsv"
    assert result.benchling_genbank_count == 4
    assert sorted(path.name for path in result.benchling_genbank_dir.iterdir() if not path.name.startswith(".")) == [
        "pES-retron-201-msd[tetO-retest]-r26-w02-17.gb",
        "pES-retron-202-msd[tetO-retest]-r26-w03-16.gb",
        "pES-retron-203-msd[tetO-retest]-r180-w02-17.gb",
        "pES-retron-204-msd[tetO-retest]-r180-w03-16.gb",
    ]

    review_manifest = json.loads(result.review_manifest_path.read_text(encoding="utf-8"))
    assert review_manifest["deliverable_plan_id"] == "teto_payload_trim_retest_v1"
    assert review_manifest["pwm_triptych"]["payload_trim_ids"] == [
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    ]
    assert review_manifest["benchling_genbank_import"]["assigned_retron_ids"] == {
        "r26-w02-17": "pES-retron-201",
        "r26-w03-16": "pES-retron-202",
        "r180-w02-17": "pES-retron-203",
        "r180-w03-16": "pES-retron-204",
    }
    assert review_manifest["benchling_genbank_import"]["files"] == [
        "benchling_genbank/pES-retron-201-msd[tetO-retest]-r26-w02-17.gb",
        "benchling_genbank/pES-retron-202-msd[tetO-retest]-r26-w03-16.gb",
        "benchling_genbank/pES-retron-203-msd[tetO-retest]-r180-w02-17.gb",
        "benchling_genbank/pES-retron-204-msd[tetO-retest]-r180-w03-16.gb",
    ]
    video_manifest = json.loads(result.sequence_montage_manifest.read_text(encoding="utf-8"))
    assert video_manifest["frames"][0]["variant_id"] == "r26-w02-17"
    assert video_manifest["frames"][0]["evidence_label"] == ("pES-retron-201 | tetO PWM [2,17) | r26 scaffold | 15 nt")
