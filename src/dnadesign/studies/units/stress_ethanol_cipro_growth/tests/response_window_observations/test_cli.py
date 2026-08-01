"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_cli.py

Command-surface tests for response-window observation evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import cli


def test_preview_reports_label_truth_and_repeat_blockers(monkeypatch, capsys, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    monkeypatch.setattr(cli, "preview_response_window_observation_evidence", lambda **_: evidence)

    result = cli.main(
        [
            "preview",
            "--reader-root",
            str(tmp_path / "reader"),
            "--reader-experiment",
            str(tmp_path / "reader/experiment"),
            "--candidate-bindings",
            str(tmp_path / "bindings"),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["approval_status"] == "review_required"
    assert payload["candidate_count"] == 3
    assert payload["candidate_observation_preview_count"] == 1
    assert payload["repeated_candidate_count"] == 2
    assert payload["blocker_count"] == 3
    assert payload["ready_to_materialize"] is False


def test_parser_uses_study_owned_policy_and_reader_projection_defaults(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "preview",
            "--reader-root",
            str(tmp_path / "reader"),
            "--reader-experiment",
            str(tmp_path / "reader/experiment"),
            "--candidate-bindings",
            str(tmp_path / "bindings"),
        ]
    )

    assert args.policy.name == "observation_policy.yaml"
    assert args.reader_projection.name == "reader_response_projection.yaml"


def _evidence(tmp_path: Path):
    from dataclasses import replace

    import pandas as pd

    from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
        ResponseWindowObservationPreview,
    )
    from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.policy import (
        load_response_window_observation_policy,
    )
    from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.sources import (
        ResolvedReaderCandidateEvidence,
        ResponseWindowObservationEvidence,
    )

    policy = load_response_window_observation_policy(cli.DEFAULT_POLICY_PATH)
    primary = policy.aggregation.primary_reduction_id
    measurements = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "b", "c", "c"],
            "reader_experiment_id": ["e1", "e1", "e2", "e1", "e2"],
            "reduction_id": [primary] * 5,
        }
    )
    preview = ResponseWindowObservationPreview(
        observations=pd.DataFrame({"candidate_id": ["a"]}),
        contributions=pd.DataFrame(),
        bootstrap_draws=pd.DataFrame(),
        uncertainty=pd.DataFrame(),
        repeat_diagnostics=pd.DataFrame(
            {
                "candidate_id": [candidate_id for candidate_id in ("b", "c") for _ in range(8)],
                "component": [
                    component
                    for _ in ("b", "c")
                    for component in ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
                ],
                "range": [1.0] * 16,
                "status": ["review_required"] * 16,
            }
        ),
        reduction_sensitivity=pd.DataFrame(),
        event_time_sensitivity=pd.DataFrame(),
        blockers=(
            "b: repeated experiments require an explicit label-source decision",
            "c: repeated experiments require an explicit label-source decision",
            "response-window observation policy requires study approval",
        ),
    )
    return ResponseWindowObservationEvidence(
        policy=replace(policy, approval_status="review_required"),
        resolved=ResolvedReaderCandidateEvidence(
            measurements=measurements,
            bootstrap_draws=pd.DataFrame(),
            excluded_reader_designs=pd.DataFrame({"design_id": ["excluded"]}),
        ),
        preview=preview,
        reader_records=object(),  # type: ignore[arg-type]
        reader_catalog_path=tmp_path / "records.json",
        reader_catalog_sha256="a" * 64,
        reader_projection_path=tmp_path / "reader_response_projection.yaml",
        reader_projection_sha256="c" * 64,
        candidate_bindings_manifest_path=tmp_path / "bindings.json",
        candidate_bindings_manifest_sha256="b" * 64,
        candidate_bindings_path=tmp_path / "bindings.parquet",
    )
