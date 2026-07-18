"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/calibration_preview.py

Derive the stress-study RMF calibration without publishing metastudy output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.contracts import (
    EVENT_HALF_RANGE_COLUMNS,
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.sources import (
    preview_response_window_observation_evidence,
)

from ...source_evidence import rmf_round0_source_evidence_root
from ..core.contracts import MetastudyPaths, StressCampaignContract
from ..core.response_contracts import RESPONSE_REVIEW_SPEC
from ..evaluation.response_uncertainty import estimate_response_calibration_from_reader_draws
from .loading import assert_campaign_response_reduction, load_stress_campaign_contract

SCHEMA_ID = "stress_ethanol_cipro_growth.rmf_calibration_preview.v2"
CALIBRATION_COHORT_ID = "exact_primary_reader_candidate_experiments_v1"
_COMPONENT_FIELDS = {
    "response_separation": ("response_separation_min", "response_separation_scale"),
    "on_magnitude_floor": ("on_magnitude_min", "on_magnitude_scale"),
    "off_magnitude_ceiling": ("off_magnitude_max", "off_magnitude_scale"),
}


@dataclass(frozen=True)
class ResponseCalibrationCohort:
    """Decision-independent Reader candidate-experiment units used for assay scales."""

    labels: pd.DataFrame
    draws: pd.DataFrame
    unit_count: int
    candidate_count: int
    reader_experiment_count: int
    excluded_nonexact_unit_count: int


def preview_response_calibration(
    *,
    repo_root: Path,
    reader_bundle_root: Path,
    candidate_binding_bundle_root: Path,
) -> dict[str, object]:
    """Return Reader-derived campaign calibration without writing artifacts."""

    root = Path(repo_root).resolve()
    paths = MetastudyPaths(
        repo_root=root,
        reader_bundle_root=Path(reader_bundle_root).resolve(),
        out_dir=root / ".unused-calibration-preview",
        campaign_root=rmf_round0_source_evidence_root(root).resolve(),
    )
    campaign = load_stress_campaign_contract(paths)
    request_path = (
        root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth"
        / "response_window_observations/config/reader_response_window.yaml"
    )
    policy_path = (
        root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth"
        / "response_window_observations/config/observation_policy.yaml"
    )
    evidence = preview_response_window_observation_evidence(
        reader_bundle_root=paths.reader_bundle_root,
        reader_request_path=request_path,
        candidate_bindings_root=candidate_binding_bundle_root,
        policy_path=policy_path,
    )
    primary_reduction_id = evidence.policy.aggregation.primary_reduction_id
    assert_campaign_response_reduction(campaign, primary_reduction_id=primary_reduction_id)
    cohort = build_calibration_cohort(
        evidence.resolved.measurements,
        evidence.resolved.bootstrap_draws,
        primary_reduction_id=primary_reduction_id,
    )
    draw_counts = cohort.draws.groupby("id")["draw_index"].nunique()
    if draw_counts.empty or draw_counts.nunique() != 1:
        raise ValueError("Reader calibration-cohort bootstrap draw counts must be complete and identical.")
    bootstrap_samples = int(draw_counts.iloc[0])
    result = estimate_response_calibration_from_reader_draws(
        cohort.labels,
        cohort.draws,
        target_views=campaign.target_views,
        scale_quantile=RESPONSE_REVIEW_SPEC.scale_quantile,
        expected_bootstrap_samples=bootstrap_samples,
    )
    return build_calibration_preview_payload(
        calibration=result.calibration,
        campaign=campaign,
        reader_manifest_sha256=evidence.reader_manifest_sha256,
        reader_request_sha256="sha256:" + _file_sha256(request_path),
        candidate_bindings_manifest_sha256=evidence.candidate_bindings_manifest_sha256,
        observation_policy_sha256=evidence.policy.config_sha256,
        primary_reduction_id=primary_reduction_id,
        calibration_unit_count=cohort.unit_count,
        calibration_candidate_count=cohort.candidate_count,
        calibration_experiment_count=cohort.reader_experiment_count,
        excluded_nonexact_unit_count=cohort.excluded_nonexact_unit_count,
        bootstrap_samples=bootstrap_samples,
    )


def build_calibration_cohort(
    measurements: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    primary_reduction_id: str,
) -> ResponseCalibrationCohort:
    """Project every exact, study-bound primary candidate-experiment evidence unit."""

    identity = ["candidate_id", "design_id", "reader_experiment_id"]
    measurement_required = {
        *identity,
        "reduction_id",
        *VALUE_COLUMNS,
        *EVENT_HALF_RANGE_COLUMNS,
        *(f"{component}_bound_kind" for component in VALUE_COLUMNS),
    }
    draw_required = {*identity, "reduction_id", "draw_index", *VALUE_COLUMNS}
    if missing := sorted(measurement_required - set(measurements.columns)):
        raise ValueError(f"Reader calibration measurements lack fields: {missing}")
    if missing := sorted(draw_required - set(draws.columns)):
        raise ValueError(f"Reader calibration draws lack fields: {missing}")
    primary = measurements.loc[measurements["reduction_id"].astype(str).eq(primary_reduction_id)].copy()
    if primary.empty:
        raise ValueError(f"Reader calibration cohort lacks primary reduction {primary_reduction_id!r}.")
    duplicate = primary.duplicated(subset=["candidate_id", "reader_experiment_id"], keep=False)
    if duplicate.any():
        raise ValueError("Reader calibration cohort requires one design row per candidate experiment.")
    exact = primary.loc[
        primary[[f"{component}_bound_kind" for component in VALUE_COLUMNS]].astype(str).eq("exact").all(axis=1)
    ].copy()
    if exact.empty:
        raise ValueError("Reader calibration cohort contains no exact candidate-experiment units.")
    exact["id"] = exact["candidate_id"].astype(str) + "::" + exact["reader_experiment_id"].astype(str)
    if exact["id"].duplicated().any():
        raise ValueError("Reader calibration unit identities must be unique.")
    primary_draws = draws.loc[draws["reduction_id"].astype(str).eq(primary_reduction_id)].copy()
    keys = exact.loc[:, [*identity, "id"]]
    cohort_draws = primary_draws.merge(keys, on=identity, how="inner", validate="many_to_one")
    label_ids = set(exact["id"].astype(str))
    draw_ids = set(cohort_draws["id"].astype(str))
    if draw_ids != label_ids:
        raise ValueError(
            "Reader calibration draws disagree with exact cohort units: "
            f"missing={sorted(label_ids - draw_ids)}, extra={sorted(draw_ids - label_ids)}."
        )
    labels = exact.sort_values("id", kind="mergesort").reset_index(drop=True)
    cohort_draws = cohort_draws.sort_values(["id", "draw_index"], kind="mergesort").reset_index(drop=True)
    return ResponseCalibrationCohort(
        labels=labels,
        draws=cohort_draws,
        unit_count=len(labels),
        candidate_count=int(labels["candidate_id"].nunique()),
        reader_experiment_count=int(labels["reader_experiment_id"].nunique()),
        excluded_nonexact_unit_count=int(len(primary) - len(labels)),
    )


def build_calibration_preview_payload(
    *,
    calibration: pd.DataFrame,
    campaign: StressCampaignContract,
    reader_manifest_sha256: str,
    reader_request_sha256: str,
    candidate_bindings_manifest_sha256: str,
    observation_policy_sha256: str,
    primary_reduction_id: str,
    calibration_unit_count: int,
    calibration_candidate_count: int,
    calibration_experiment_count: int,
    excluded_nonexact_unit_count: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    """Build a stable JSON projection from one derived calibration table."""

    required = {"selection_view_id", "component", "threshold", "scale", "scale_quantile", "scale_basis"}
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"response calibration preview lacks fields: {missing}")
    view_rows: list[dict[str, object]] = []
    matches = True
    for view in campaign.target_views:
        configured = campaign.rmf_calibration_by_view[view.id]
        parameters: dict[str, float] = {}
        differences: dict[str, float] = {}
        rows = calibration.loc[calibration["selection_view_id"].astype(str).eq(view.id)]
        for component, (threshold_field, scale_field) in _COMPONENT_FIELDS.items():
            component_rows = rows.loc[rows["component"].astype(str).eq(component)]
            if len(component_rows) != 1:
                raise ValueError(f"response calibration preview expected one {view.id}/{component} row.")
            row = component_rows.iloc[0]
            parameters[threshold_field] = float(row["threshold"])
            parameters[scale_field] = float(row["scale"])
            differences[scale_field] = abs(float(configured[scale_field]) - float(row["scale"]))
        view_matches = all(value <= 5.0e-7 for value in differences.values())
        matches = matches and view_matches
        view_rows.append(
            {
                "selection_view_id": view.id,
                "target_mask": [int(value) for value in view.target_mask],
                "derived_calibration": parameters,
                "configured_calibration": {key: float(value) for key, value in sorted(configured.items())},
                "scale_absolute_difference": differences,
                "matches_campaign_six_decimal_contract": view_matches,
            }
        )
    calibration_cohort = {
        "cohort_id": CALIBRATION_COHORT_ID,
        "unit": "reader_candidate_experiment",
        "inclusion_rule": "study_bound_nonreference_primary_rows_with_all_eight_components_exact",
        "model_screen_selection_role": "none",
        "repeat_label_decision_role": "none",
        "unit_count": int(calibration_unit_count),
        "candidate_count": int(calibration_candidate_count),
        "reader_experiment_count": int(calibration_experiment_count),
        "excluded_nonexact_unit_count": int(excluded_nonexact_unit_count),
    }
    configured_cohort = dict(campaign.rmf_calibration_cohort)
    campaign_cohort_projection = {
        "cohort_id": calibration_cohort["cohort_id"],
        "unit": calibration_cohort["unit"],
        "scale_quantile": float(calibration["scale_quantile"].iloc[0]),
        "reader_bundle_manifest_sha256": reader_manifest_sha256,
        "candidate_bindings_manifest_sha256": candidate_bindings_manifest_sha256,
        "unit_count": calibration_cohort["unit_count"],
        "excluded_nonexact_unit_count": calibration_cohort["excluded_nonexact_unit_count"],
    }
    cohort_matches = configured_cohort == campaign_cohort_projection
    return {
        "schema_id": SCHEMA_ID,
        "study_id": "stress_ethanol_cipro_growth",
        "mutation_posture": "preview_only",
        "primary_reduction_id": primary_reduction_id,
        "reader_bundle_manifest_sha256": reader_manifest_sha256,
        "reader_request_sha256": reader_request_sha256,
        "candidate_bindings_manifest_sha256": candidate_bindings_manifest_sha256,
        "observation_policy_sha256": observation_policy_sha256,
        "calibration_cohort": calibration_cohort,
        "configured_campaign_calibration_cohort": configured_cohort,
        "campaign_matches_calibration_cohort": cohort_matches,
        "bootstrap_samples": int(bootstrap_samples),
        "scale_quantile": float(calibration["scale_quantile"].iloc[0]),
        "scale_basis": sorted(calibration["scale_basis"].astype(str).unique().tolist()),
        "campaign_matches_reader_calibration": matches and cohort_matches,
        "selection_views": view_rows,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CALIBRATION_COHORT_ID",
    "SCHEMA_ID",
    "ResponseCalibrationCohort",
    "build_calibration_cohort",
    "build_calibration_preview_payload",
    "preview_response_calibration",
]
