"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/test_review_outputs.py

CLI tests for tetO PWM trim rescue review-output generation.

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


def test_review_outputs_cli_defaults_to_workbench_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = tmp_path / "materialized"

    monkeypatch.setattr(
        cli_review_outputs_module,
        "generate_teto_pwm_trim_rescue_review_outputs",
        fake_review_output_result,
    )

    result = RUNNER.invoke(
        app,
        review_outputs_args(study_dir, materialized_root, output_format="json"),
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    expected_root = study_dir / "workbench" / "outputs" / "teto_pwm_trim_rescue_v1"
    assert payload["status"] == "ok"
    assert payload["output_dir"] == expected_root.as_posix()
    assert payload["review_manifest_path"] == (expected_root / "reviews" / "review_manifest.json").as_posix()
    assert (
        payload["handoff_tsv"]
        == (expected_root / "reviews" / "handoff" / "teto_pwm_trim_rescue_v1.handoff.tsv").as_posix()
    )
    assert payload["handoff_verified_count"] == 9
    assert "clone_handoff_index_tsv" not in payload
    assert "clone_handoff_verified_count" not in payload
    assert payload["record_count"] == 9
