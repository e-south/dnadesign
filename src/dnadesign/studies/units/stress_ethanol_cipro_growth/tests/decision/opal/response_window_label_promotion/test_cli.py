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
    output = tmp_path / "dataset/_opal/response_window_labels_v4"
    monkeypatch.setattr(
        cli,
        "publish_response_window_labels",
        lambda **_: contracts.ResponseWindowLabelPromotionResult(
            output_directory=output,
            label_path=output / "observed_labels.parquet",
            study_provenance_path=output / "study_provenance.json",
            promotion_manifest_path=output / "promotion.manifest.json",
            label_event_count=35,
            unique_candidate_count=31,
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
    assert payload["label_event_count"] == 35
    assert payload["unique_candidate_count"] == 31
    assert "candidate_count" not in payload
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
    assert args.prior_promotion_manifest is None


def test_publish_parser_accepts_explicit_prior_promotion(tmp_path: Path) -> None:
    parser = cli.build_parser()
    prior = tmp_path / "dataset/_opal/response_window_labels_batch0_v4/promotion.manifest.json"

    args = parser.parse_args(
        [
            "publish",
            "--observation-bundle",
            str(tmp_path / "observations"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prior-promotion-manifest",
            str(prior),
        ]
    )

    assert args.prior_promotion_manifest == prior


def test_campaign_binding_verification_requires_an_explicit_config(tmp_path: Path) -> None:
    parser = cli.build_parser()

    try:
        parser.parse_args(
            [
                "verify-campaign-binding",
                "--dataset-root",
                str(tmp_path / "dataset"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - argparse must reject a missing required argument
        raise AssertionError("verify-campaign-binding accepted no campaign config")


def test_study_verify_has_no_implicit_campaign_config(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["verify", "--dataset-root", str(tmp_path / "dataset")])

    assert not hasattr(args, "campaign_config")
    assert args.output_relative_directory == "_opal/response_window_labels_v4"
