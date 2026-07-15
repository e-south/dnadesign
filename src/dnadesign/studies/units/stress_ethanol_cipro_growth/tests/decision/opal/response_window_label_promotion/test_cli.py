"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_window_label_promotion/test_cli.py

Command-surface tests for immutable response-window label promotion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    cli,
    contracts,
)


def test_publish_command_reports_exact_immutable_artifact_paths(monkeypatch, capsys, tmp_path: Path) -> None:
    output = tmp_path / "dataset/_opal/response_window_labels_v1"
    monkeypatch.setattr(
        cli,
        "publish_response_window_labels",
        lambda **_: contracts.ResponseWindowLabelPromotionResult(
            output_directory=output,
            label_path=output / "observed_labels.parquet",
            study_provenance_path=output / "study_provenance.json",
            promotion_manifest_path=output / "promotion.manifest.json",
            candidate_count=35,
        ),
    )

    result = cli.main(
        [
            "publish",
            "--observation-bundle",
            str(tmp_path / "observations"),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 35
    assert payload["promotion_manifest_path"].endswith("promotion.manifest.json")
    assert payload["create_only"] is True


def test_publish_parser_has_no_overwrite_mode(tmp_path: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "publish",
            "--observation-bundle",
            str(tmp_path / "observations"),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
    )

    assert not hasattr(args, "overwrite")
