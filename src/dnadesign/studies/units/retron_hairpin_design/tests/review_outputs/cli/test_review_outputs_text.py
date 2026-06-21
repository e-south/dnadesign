"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/test_review_outputs_text.py

Text-output CLI tests for tetO PWM trim rescue review-output generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.retron_hairpin_design.interfaces.cli import review_outputs as cli_review_outputs_module
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ...support.cli import RUNNER
from ...support.paths import repo_root_from
from .fixtures import fake_review_output_result, review_outputs_args


def test_review_outputs_cli_text_reports_sequence_handoff(
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

    result = RUNNER.invoke(app, review_outputs_args(study_dir, materialized_root))

    assert result.exit_code == 0, result.stdout
    assert "handoff_tsv:" in result.stdout
    assert "handoff_markdown:" in result.stdout
    assert "handoff_verified_count: 9" in result.stdout
    assert "clone_handoff" not in result.stdout
