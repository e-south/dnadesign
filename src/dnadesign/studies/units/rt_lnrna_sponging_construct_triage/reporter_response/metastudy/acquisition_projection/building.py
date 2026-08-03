"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/acquisition_projection/building.py

Build selected-window projections from typed profile evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict

from ...measurement_profile import ReporterMeasurementProfile
from ...profile.measurement import TimeWindowReduction
from ..contracts._values import MetastudyContractError, canonical_digest
from ..contracts.profile import ProfileEvidence
from ..evidence_projection.contracts import ProfileEvidenceProjection
from ._values import window
from .contracts import (
    ACQUISITION_PROJECTION_CONTRACT_ID,
    AcquisitionContribution,
    AcquisitionCoordinate,
    AcquisitionMetricProjection,
    AcquisitionProjection,
    coordinate_key,
)


def build_acquisition_projection(
    evidence: Iterable[ProfileEvidence | ProfileEvidenceProjection],
    *,
    selected_reduction: tuple[float, float],
) -> AcquisitionProjection:
    """Summarize one window without treating acquisitions as biological replicates."""

    selected_window = window(selected_reduction)
    grouped: dict[tuple[str, float, str, str, str, str], list[AcquisitionContribution]] = defaultdict(list)
    for row in evidence:
        if not isinstance(row, (ProfileEvidence, ProfileEvidenceProjection)):
            raise MetastudyContractError("acquisition projection requires typed profile evidence")
        profile = row.profile
        reduction = profile.reduction
        if not isinstance(reduction, TimeWindowReduction):
            raise MetastudyContractError("acquisition projection accepts time-window profiles only")
        if (reduction.recorded_start_time_h, reduction.recorded_end_time_h) != selected_window:
            continue
        reduction_id = f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"
        reduction_digest = canonical_digest(asdict(reduction))
        acquisition_id = profile.provenance.reader_experiment_id
        measurements_by_dose: dict[float, list[object]] = defaultdict(list)
        for measurement in profile.measurements:
            if measurement.acquisition_id != acquisition_id:
                raise MetastudyContractError("profile measurement acquisition differs from Reader provenance")
            if measurement.role == "dose":
                assert measurement.dose_uM is not None
                measurements_by_dose[float(measurement.dose_uM)].append(measurement)
        normalization_available = not (
            isinstance(profile, ReporterMeasurementProfile) or getattr(profile, "reference_normalization", None)
        )
        responses_by_dose: dict[float, list[object]] = defaultdict(list)
        for response in getattr(profile, "dose_responses", ()):
            if response.acquisition_id != acquisition_id:
                raise MetastudyContractError("profile response acquisition differs from Reader provenance")
            responses_by_dose[float(response.dose_uM)].append(response)
        if normalization_available and set(responses_by_dose) != set(measurements_by_dose):
            raise MetastudyContractError("normalized responses differ from measured dose coordinates")
        for dose, measurements in measurements_by_dose.items():
            responses = responses_by_dose[dose]
            spaces = ("raw_measurement", "reference_normalized") if normalization_available else ("raw_measurement",)
            for metric_space in spaces:
                key = (
                    profile.subject_id,
                    dose,
                    reduction_id,
                    reduction_digest,
                    profile.observation_policy.digest,
                    metric_space,
                )
                grouped[key].append(
                    _contribution(
                        acquisition_id=acquisition_id,
                        profile_id=profile.profile_id,
                        profile_digest=row.audit.profile_digest,
                        measurements=measurements,
                        responses=responses,
                        metric_space=metric_space,
                    )
                )
    if not grouped:
        raise MetastudyContractError("acquisition projection requires evidence for the selected reduction")

    coordinates = tuple(_coordinate(key, values) for key, values in sorted(grouped.items()))
    return AcquisitionProjection(
        contract_id=ACQUISITION_PROJECTION_CONTRACT_ID,
        selected_reduction=selected_window,
        coordinates=tuple(sorted(coordinates, key=coordinate_key)),
    )


def _contribution(
    *,
    acquisition_id: str,
    profile_id: str,
    profile_digest: str,
    measurements: list[object],
    responses: list[object],
    metric_space: str,
) -> AcquisitionContribution:
    normalization_available = metric_space == "reference_normalized"
    replicate_ids = tuple(
        sorted(
            {
                measurement.biological_replicate_id
                for measurement in measurements
                if measurement.biological_replicate_id is not None
            }
        )
    )
    return AcquisitionContribution(
        acquisition_id=acquisition_id,
        profile_id=profile_id,
        profile_digest=profile_digest,
        declared_biological_replicate_ids=replicate_ids,
        rfp=None if normalization_available else _median(measurements, "rfp"),
        od600=None if normalization_available else _median(measurements, "od600"),
        rfp_over_od600=None if normalization_available else _median(measurements, "rfp_over_od600"),
        normalized_reporter_response=(
            _median(responses, "normalized_reporter_response") if normalization_available else None
        ),
        relative_od=_median(responses, "relative_od") if normalization_available else None,
    )


def _coordinate(
    key: tuple[str, float, str, str, str, str],
    values: list[AcquisitionContribution],
) -> AcquisitionCoordinate:
    subject_id, dose, reduction_id, reduction_digest, policy_digest, metric_space = key
    normalization_available = metric_space == "reference_normalized"
    contributions = tuple(sorted(values, key=lambda row: row.acquisition_id))
    acquisition_ids = tuple(row.acquisition_id for row in contributions)
    if len(acquisition_ids) != len(set(acquisition_ids)):
        raise MetastudyContractError(
            f"acquisition coordinate contains duplicate acquisition: {(subject_id, dose, reduction_id)!r}"
        )
    return AcquisitionCoordinate(
        subject_id=subject_id,
        condition_role="dose",
        metric_space=metric_space,
        dose_uM=dose,
        reduction_id=reduction_id,
        reduction_digest=reduction_digest,
        observation_policy_digest=policy_digest,
        acquisition_ids=acquisition_ids,
        contributions=contributions,
        rfp=None
        if normalization_available
        else _metric(tuple(row.rfp for row in contributions if row.rfp is not None)),
        od600=(
            None
            if normalization_available
            else _metric(tuple(row.od600 for row in contributions if row.od600 is not None))
        ),
        rfp_over_od600=(
            None
            if normalization_available
            else _metric(tuple(row.rfp_over_od600 for row in contributions if row.rfp_over_od600 is not None))
        ),
        normalized_reporter_response=(
            _metric(
                tuple(
                    row.normalized_reporter_response
                    for row in contributions
                    if row.normalized_reporter_response is not None
                )
            )
            if normalization_available
            else None
        ),
        relative_od=(
            _metric(tuple(row.relative_od for row in contributions if row.relative_od is not None))
            if normalization_available
            else None
        ),
    )


def _median(rows: list[object], field_name: str) -> float:
    return float(statistics.median(getattr(row, field_name) for row in rows))


def _metric(values: tuple[float, ...]) -> AcquisitionMetricProjection:
    estimate = float(statistics.median(values))
    leave_one_out = (
        tuple(float(statistics.median(values[:index] + values[index + 1 :])) for index in range(len(values)))
        if len(values) >= 2
        else ()
    )
    return AcquisitionMetricProjection(
        estimate=estimate,
        method="median_across_acquisitions",
        acquisition_count=len(values),
        leave_one_acquisition_out_estimates=leave_one_out,
    )


__all__ = ["build_acquisition_projection"]
