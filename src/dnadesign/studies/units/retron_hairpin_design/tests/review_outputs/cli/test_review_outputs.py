"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/test_review_outputs.py

CLI tests for tetO PWM trim review-output generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.retron_hairpin_design.interfaces.cli import review_outputs as cli_review_outputs_module
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ...support.cli import RUNNER
from ...support.paths import repo_root_from
from .fixtures import fake_review_output_result, review_outputs_args


def test_review_outputs_cli_requires_explicit_plan(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = tmp_path / "materialized"

    result = RUNNER.invoke(
        app,
        review_outputs_args(study_dir, materialized_root=materialized_root, output_format="json"),
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Provide --deliverable-plan" in payload["error"]
    assert "workbench/deliverables/*.yaml" in payload["next_step"]


def test_review_outputs_cli_uses_explicit_plan_and_output_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    deliverable_plan = study_dir / "workbench" / "deliverables" / "teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
    materialized_root = tmp_path / "materialized"
    expected_root = tmp_path / "review-outputs"

    monkeypatch.setattr(
        cli_review_outputs_module,
        "generate_retron_hairpin_review_outputs",
        fake_review_output_result,
    )

    result = RUNNER.invoke(
        app,
        review_outputs_args(
            study_dir,
            deliverable_plan=deliverable_plan,
            materialized_root=materialized_root,
            out_dir=expected_root,
            output_format="json",
        ),
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["output_dir"] == expected_root.as_posix()
    assert payload["review_manifest_path"] == (expected_root / "reviews" / "review_manifest.json").as_posix()
    assert (
        payload["handoff_tsv"]
        == (expected_root / "reviews" / "handoff" / "teto_retained_span_trim_tetr_pwm_elite_v1.handoff.tsv").as_posix()
    )
    assert payload["benchling_genbank_dir"] == (expected_root / "benchling_genbank").as_posix()
    assert (
        payload["benchling_genbank_index"]
        == (
            expected_root / "reviews" / "handoff" / "teto_retained_span_trim_tetr_pwm_elite_v1.benchling_genbank.tsv"
        ).as_posix()
    )
    assert payload["benchling_genbank_count"] == 6
    assert payload["handoff_verified_count"] == 9
    assert payload["record_count"] == 9
    assert "teto_retained_span_trim_tetr_pwm_elite_v1.pwm_trim_triptych.png" in payload["next_step"]
