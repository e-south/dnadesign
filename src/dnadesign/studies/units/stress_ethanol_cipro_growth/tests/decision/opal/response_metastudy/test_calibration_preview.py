"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_calibration_preview.py

Read-only RMF calibration preview tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy import cli
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressCampaignContract,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    calibration_preview,
)


def test_calibration_preview_binds_reader_identity_masks_and_campaign_parity() -> None:
    fields = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 0.4,
        "on_magnitude_scale": 0.3,
        "off_magnitude_scale": 0.2,
    }
    campaign = StressCampaignContract(
        slug="secg_rmf_greedy",
        config_path=Path("campaign.yaml"),
        target_views=(
            StressTargetView(id="ethanol", label="Ethanol", target_mask=(0.0, 1.0, 0.0, 1.0)),
            StressTargetView(id="ciprofloxacin", label="Ciprofloxacin", target_mask=(0.0, 0.0, 1.0, 1.0)),
            StressTargetView(id="and", label="AND", target_mask=(0.0, 0.0, 0.0, 1.0)),
        ),
        candidate_records_path=Path("records.parquet"),
        x_column_name="x",
        response_reduction_id="event_logmean_4_8h_post",
        rmf_calibration_by_view={view_id: fields for view_id in ("ethanol", "ciprofloxacin", "and")},
        rmf_calibration_cohort={
            "cohort_id": "exact_primary_reader_candidate_experiments_v1",
            "unit": "reader_candidate_experiment",
            "scale_quantile": 0.9,
            "reader_bundle_manifest_sha256": "a" * 64,
            "candidate_bindings_manifest_sha256": "c" * 64,
            "unit_count": 35,
            "excluded_nonexact_unit_count": 4,
        },
    )
    rows = []
    for view_id in ("ethanol", "ciprofloxacin", "and"):
        for component, scale in (
            ("response_separation", 0.4),
            ("on_magnitude_floor", 0.3),
            ("off_magnitude_ceiling", 0.2),
        ):
            rows.append(
                {
                    "selection_view_id": view_id,
                    "component": component,
                    "threshold": 0.0,
                    "scale": scale,
                    "scale_quantile": 0.9,
                    "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
                }
            )

    payload = calibration_preview.build_calibration_preview_payload(
        calibration=pd.DataFrame(rows),
        campaign=campaign,
        reader_catalog_sha256="a" * 64,
        reader_projection_sha256="sha256:" + "b" * 64,
        candidate_bindings_manifest_sha256="c" * 64,
        observation_policy_sha256="d" * 64,
        reader_record_receipt_sha256="e" * 64,
        approved_reader_record_receipt_sha256="e" * 64,
        approval_status="approved",
        source_blockers=(),
        primary_reduction_id="event_logmean_4_8h_post",
        calibration_unit_count=35,
        calibration_candidate_count=31,
        calibration_experiment_count=8,
        excluded_nonexact_unit_count=4,
        bootstrap_samples=500,
    )

    assert payload["schema_id"] == calibration_preview.SCHEMA_ID
    assert payload["mutation_posture"] == "preview_only"
    assert payload["campaign_matches_reader_calibration"] is True
    assert payload["source_ready"] is True
    assert payload["ready_for_campaign"] is True
    assert payload["reader_record_receipt_sha256"] == "e" * 64
    assert payload["blockers"] == []
    assert payload["campaign_matches_calibration_cohort"] is True
    assert payload["calibration_cohort"] == {
        "cohort_id": "exact_primary_reader_candidate_experiments_v1",
        "unit": "reader_candidate_experiment",
        "inclusion_rule": "study_bound_nonreference_primary_rows_with_all_eight_components_exact",
        "model_screen_selection_role": "none",
        "repeat_label_decision_role": "none",
        "unit_count": 35,
        "candidate_count": 31,
        "reader_experiment_count": 8,
        "excluded_nonexact_unit_count": 4,
    }
    assert [row["target_mask"] for row in payload["selection_views"]] == [
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert payload["selection_views"][0]["derived_calibration"] == fields


def test_calibration_preview_reports_drift_without_mutating_or_failing() -> None:
    campaign = StressCampaignContract(
        slug="secg_rmf_greedy",
        config_path=Path("campaign.yaml"),
        target_views=(StressTargetView(id="and", label="AND", target_mask=(0.0, 0.0, 0.0, 1.0)),),
        candidate_records_path=Path("records.parquet"),
        x_column_name="x",
        response_reduction_id="primary",
        rmf_calibration_by_view={
            "and": {
                "response_separation_min": 0.0,
                "on_magnitude_min": 0.0,
                "off_magnitude_max": 0.0,
                "response_separation_scale": 1.0,
                "on_magnitude_scale": 1.0,
                "off_magnitude_scale": 1.0,
            }
        },
        rmf_calibration_cohort={
            "cohort_id": "exact_primary_reader_candidate_experiments_v1",
            "unit": "reader_candidate_experiment",
            "scale_quantile": 0.9,
            "reader_bundle_manifest_sha256": "a" * 64,
            "candidate_bindings_manifest_sha256": "c" * 64,
            "unit_count": 1,
            "excluded_nonexact_unit_count": 0,
        },
    )
    calibration = pd.DataFrame(
        [
            {
                "selection_view_id": "and",
                "component": component,
                "threshold": 0.0,
                "scale": 0.5,
                "scale_quantile": 0.9,
                "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
            }
            for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")
        ]
    )

    payload = calibration_preview.build_calibration_preview_payload(
        calibration=calibration,
        campaign=campaign,
        reader_catalog_sha256="a" * 64,
        reader_projection_sha256="sha256:" + "b" * 64,
        candidate_bindings_manifest_sha256="c" * 64,
        observation_policy_sha256="d" * 64,
        reader_record_receipt_sha256="e" * 64,
        approved_reader_record_receipt_sha256="e" * 64,
        approval_status="approved",
        source_blockers=(),
        primary_reduction_id="primary",
        calibration_unit_count=1,
        calibration_candidate_count=1,
        calibration_experiment_count=1,
        excluded_nonexact_unit_count=0,
        bootstrap_samples=500,
    )

    assert payload["campaign_matches_reader_calibration"] is False
    assert payload["source_ready"] is True
    assert payload["ready_for_campaign"] is False
    assert payload["selection_views"][0]["matches_campaign_six_decimal_contract"] is False


def test_matching_calibration_stays_closed_when_policy_is_unapproved_and_unpinned() -> None:
    fields = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 0.4,
        "on_magnitude_scale": 0.3,
        "off_magnitude_scale": 0.2,
    }
    campaign = StressCampaignContract(
        slug="secg_rmf_greedy",
        config_path=Path("campaign.yaml"),
        target_views=(StressTargetView(id="and", label="AND", target_mask=(0.0, 0.0, 0.0, 1.0)),),
        candidate_records_path=Path("records.parquet"),
        x_column_name="x",
        response_reduction_id="primary",
        rmf_calibration_by_view={"and": fields},
        rmf_calibration_cohort={
            "cohort_id": "exact_primary_reader_candidate_experiments_v1",
            "unit": "reader_candidate_experiment",
            "scale_quantile": 0.9,
            "reader_bundle_manifest_sha256": "a" * 64,
            "candidate_bindings_manifest_sha256": "c" * 64,
            "unit_count": 1,
            "excluded_nonexact_unit_count": 0,
        },
    )
    calibration = pd.DataFrame(
        [
            {
                "selection_view_id": "and",
                "component": component,
                "threshold": 0.0,
                "scale": scale,
                "scale_quantile": 0.9,
                "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
            }
            for component, scale in (
                ("response_separation", 0.4),
                ("on_magnitude_floor", 0.3),
                ("off_magnitude_ceiling", 0.2),
            )
        ]
    )

    payload = calibration_preview.build_calibration_preview_payload(
        calibration=calibration,
        campaign=campaign,
        reader_catalog_sha256="a" * 64,
        reader_projection_sha256="b" * 64,
        candidate_bindings_manifest_sha256="c" * 64,
        observation_policy_sha256="d" * 64,
        reader_record_receipt_sha256="e" * 64,
        approved_reader_record_receipt_sha256=None,
        approval_status="review_required",
        source_blockers=(
            "canonical Reader record receipt requires study review and approval",
            "response-window observation policy requires study approval",
        ),
        primary_reduction_id="primary",
        calibration_unit_count=1,
        calibration_candidate_count=1,
        calibration_experiment_count=1,
        excluded_nonexact_unit_count=0,
        bootstrap_samples=500,
    )

    assert payload["campaign_matches_reader_calibration"] is True
    assert payload["source_ready"] is False
    assert payload["ready_for_campaign"] is False
    assert payload["approval_status"] == "review_required"
    assert payload["approved_reader_record_receipt_sha256"] is None
    assert payload["reader_record_receipt_sha256"] == "e" * 64
    assert payload["blocker_count"] == 2


def test_plain_calibration_preview_prints_readiness_receipt_and_blockers(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "preview_response_calibration",
        lambda **_kwargs: {
            "primary_reduction_id": "primary",
            "ready_for_campaign": False,
            "source_ready": False,
            "reader_record_receipt_sha256": "e" * 64,
            "campaign_matches_reader_calibration": True,
            "blockers": ["policy requires study approval"],
            "selection_views": [],
        },
    )

    result = cli.main(
        [
            "--repo-root",
            str(tmp_path),
            "--reader-root",
            str(tmp_path),
            "--reader-experiment",
            str(tmp_path),
            "--candidate-bindings",
            str(tmp_path),
            "--calibration-preview",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "ready_for_campaign=False" in output
    assert "source_ready=False" in output
    assert f"reader_record_receipt_sha256={'e' * 64}" in output
    assert "blocker=policy requires study approval" in output


def test_calibration_cohort_uses_every_exact_candidate_experiment_without_repeat_selection() -> None:
    components = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
    measurements = []
    draws = []
    for candidate_id, experiment_id, bound_kind in (
        ("a", "e1", "exact"),
        ("a", "e2", "exact"),
        ("b", "e2", "lower"),
    ):
        row = {
            "candidate_id": candidate_id,
            "design_id": f"design-{candidate_id}",
            "reader_experiment_id": experiment_id,
            "reduction_id": "primary",
        }
        for index, component in enumerate(components):
            row[component] = float(index)
            row[f"{component}_event_half_range"] = 0.1
            row[f"{component}_bound_kind"] = bound_kind if component == "r00" else "exact"
        measurements.append(row)
        for draw_index in range(100):
            draws.append(
                {
                    "candidate_id": candidate_id,
                    "design_id": f"design-{candidate_id}",
                    "reader_experiment_id": experiment_id,
                    "reduction_id": "primary",
                    "draw_index": draw_index,
                    **{component: float(index) + draw_index / 1_000 for index, component in enumerate(components)},
                }
            )

    cohort = calibration_preview.build_calibration_cohort(
        pd.DataFrame(measurements),
        pd.DataFrame(draws),
        primary_reduction_id="primary",
    )

    assert cohort.unit_count == 2
    assert cohort.candidate_count == 1
    assert cohort.reader_experiment_count == 2
    assert cohort.excluded_nonexact_unit_count == 1
    assert set(cohort.labels["id"]) == {"a::e1", "a::e2"}
    assert set(cohort.draws["id"]) == {"a::e1", "a::e2"}
    assert cohort.draws.groupby("id")["draw_index"].nunique().to_dict() == {"a::e1": 100, "a::e2": 100}
