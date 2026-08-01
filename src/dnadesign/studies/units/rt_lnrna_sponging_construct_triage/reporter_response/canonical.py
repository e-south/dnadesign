"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/canonical.py

Canonical reporter-response derivation and cross-field validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Literal

from ._contract_values import ReporterResponseContractError
from ._contract_values import json_value as _json_value
from ._contract_values import ordered_dose_grid as _ordered_dose_grid
from .policy import ReporterResponseObservationPolicy
from .profile.measurement import (
    ConditionMeasurement,
    EndpointReduction,
    Reduction,
    TimeWindowReduction,
    validate_ratio_reduction_semantics,
)
from .profile.response import DoseResponse, PairingKind, PairingPolicy
from .profile.uncertainty import (
    BiologicalReplicateReductionStatistic,
    DoseUncertainty,
    EstimatedMetricUncertainty,
    NotEstimableMetricUncertainty,
    UncertaintyPolicy,
)


def derive_profile_rows(
    *,
    reduction: Reduction,
    dose_grid_uM: Iterable[float],
    measurements: Iterable[ConditionMeasurement],
    pairing_policy: PairingPolicy,
    observation_policy: ReporterResponseObservationPolicy,
    dose_uncertainties: Iterable[DoseUncertainty],
) -> tuple[
    tuple[float, ...],
    tuple[ConditionMeasurement, ...],
    tuple[DoseResponse, ...],
    tuple[DoseUncertainty, ...],
]:
    """Validate and canonicalize all cross-field profile data."""

    if not isinstance(reduction, (EndpointReduction, TimeWindowReduction)):
        raise ReporterResponseContractError("reduction must be endpoint or time_window")
    if not isinstance(pairing_policy, PairingPolicy):
        raise ReporterResponseContractError("pairing_policy must be PairingPolicy")
    if not isinstance(observation_policy, ReporterResponseObservationPolicy):
        raise ReporterResponseContractError("observation_policy must be ReporterResponseObservationPolicy")
    uncertainty_policy = observation_policy.biological_replicate_uncertainty_policy
    if pairing_policy.kind != observation_policy.pairing_kind:
        raise ReporterResponseContractError("pairing policy kind must equal the reporter-response observation policy")

    dose_grid = _ordered_dose_grid(dose_grid_uM)
    measurement_rows = tuple(measurements)
    if not measurement_rows:
        raise ReporterResponseContractError("measurements must not be empty")
    if not all(isinstance(row, ConditionMeasurement) for row in measurement_rows):
        raise ReporterResponseContractError("measurements must contain ConditionMeasurement values")
    validate_ratio_reduction_semantics(reduction, measurement_rows)
    observations = {row.observation_id: row for row in measurement_rows}
    if len(observations) != len(measurement_rows):
        raise ReporterResponseContractError("measurement observation_id values must be unique")

    dose_rows = tuple(row for row in measurement_rows if row.role == "dose")
    if not dose_rows:
        raise ReporterResponseContractError("measurements require at least one dose observation")
    observed_doses = tuple(sorted({float(row.dose_uM) for row in dose_rows if row.dose_uM is not None}))
    if observed_doses != dose_grid:
        raise ReporterResponseContractError(
            "dose observations must cover the declared ordered dose grid exactly; "
            f"expected {dose_grid}, observed {observed_doses}"
        )
    dose_unit_keys = tuple((float(row.dose_uM), row.acquisition_id, row.biological_replicate_id) for row in dose_rows)
    if len(dose_unit_keys) != len(set(dose_unit_keys)):
        raise ReporterResponseContractError(
            "duplicate dose rows for one scoped biological replicate and acquisition are not allowed"
        )
    within_acquisition_statistics = {row.within_acquisition_reduction_statistic for row in measurement_rows}
    if len(within_acquisition_statistics) != 1:
        raise ReporterResponseContractError(
            "all measurements in one profile must use one within_acquisition_reduction_statistic"
        )
    if next(iter(within_acquisition_statistics)) != observation_policy.within_acquisition_reduction_statistic:
        raise ReporterResponseContractError(
            "measurement within-acquisition reduction must equal the reporter-response observation policy"
        )

    assignments = {assignment.dose_observation_id: assignment for assignment in pairing_policy.assignments}
    expected_dose_ids = {row.observation_id for row in dose_rows}
    if set(assignments) != expected_dose_ids:
        raise ReporterResponseContractError(
            "pairing policy requires one explicit control assignment for every and only every dose observation"
        )

    used_baselines: set[str] = set()
    used_positive_controls: set[str] = set()
    responses: list[DoseResponse] = []
    grid_order = {dose: index for index, dose in enumerate(dose_grid)}
    for dose_row in sorted(dose_rows, key=lambda row: (grid_order[float(row.dose_uM)], row.observation_id)):
        assignment = assignments[dose_row.observation_id]
        baselines = _resolve_controls(
            observations,
            assignment.baseline_observation_ids,
            expected_role="baseline",
            dose_observation_id=dose_row.observation_id,
        )
        positive_controls = _resolve_controls(
            observations,
            assignment.positive_control_observation_ids,
            expected_role="positive_control",
            dose_observation_id=dose_row.observation_id,
        )
        if pairing_policy.kind == "paired_by_design":
            _validate_paired_control_strata(dose_row, (*baselines, *positive_controls))
        used_baselines.update(assignment.baseline_observation_ids)
        used_positive_controls.update(assignment.positive_control_observation_ids)

        baseline_ratio = statistics.median(row.rfp_over_od600 for row in baselines)
        positive_ratio = statistics.median(row.rfp_over_od600 for row in positive_controls)
        separation = positive_ratio - baseline_ratio
        if separation <= 0.0:
            raise ReporterResponseContractError(
                f"{dose_row.observation_id}: positive-control separation must be greater than zero"
            )
        baseline_od600 = statistics.median(row.od600 for row in baselines)
        if baseline_od600 <= 0.0:
            raise ReporterResponseContractError(f"{dose_row.observation_id}: baseline controls require positive OD600")
        responses.append(
            DoseResponse(
                dose_uM=float(dose_row.dose_uM),
                dose_observation_id=dose_row.observation_id,
                biological_replicate_id=dose_row.biological_replicate_id,
                acquisition_id=dose_row.acquisition_id,
                baseline_observation_ids=assignment.baseline_observation_ids,
                positive_control_observation_ids=assignment.positive_control_observation_ids,
                normalized_reporter_response=(dose_row.rfp_over_od600 - baseline_ratio) / separation,
                relative_od=dose_row.od600 / baseline_od600,
            )
        )

    declared_baselines = {row.observation_id for row in measurement_rows if row.role == "baseline"}
    declared_positive_controls = {row.observation_id for row in measurement_rows if row.role == "positive_control"}
    if used_baselines != declared_baselines or used_positive_controls != declared_positive_controls:
        raise ReporterResponseContractError("every declared control observation must participate in the pairing policy")

    response_rows = tuple(responses)
    uncertainty_rows = tuple(dose_uncertainties)
    if not all(isinstance(row, DoseUncertainty) for row in uncertainty_rows):
        raise ReporterResponseContractError("dose uncertainty rows must be DoseUncertainty values")
    uncertainty_doses = tuple(row.dose_uM for row in uncertainty_rows)
    if uncertainty_doses != dose_grid:
        raise ReporterResponseContractError(
            "dose uncertainty must cover the declared ordered dose grid exactly once and in order"
        )
    for uncertainty in uncertainty_rows:
        dose_measurements = tuple(row for row in dose_rows if row.dose_uM == uncertainty.dose_uM)
        biological_replicate_ids = {
            row.biological_replicate_id for row in dose_measurements if row.biological_replicate_id is not None
        }
        biological_replicate_count = len(biological_replicate_ids)
        identity_complete = all(row.biological_replicate_id is not None for row in dose_measurements)
        if uncertainty.biological_replicate_count != biological_replicate_count:
            raise ReporterResponseContractError(
                f"dose {uncertainty.dose_uM:g}: biological-replicate count must equal "
                f"the declared scoped identities ({biological_replicate_count})"
            )
        _validate_estimability(
            uncertainty,
            policy=uncertainty_policy,
            pairing_kind=pairing_policy.kind,
            identity_complete=identity_complete,
        )
        dose_responses = tuple(row for row in response_rows if row.dose_uM == uncertainty.dose_uM)
        for metric_name in ("normalized_reporter_response", "relative_od"):
            expected_estimate = _reduce_biological_replicate_values(
                (getattr(row, metric_name) for row in dose_responses),
                statistic=uncertainty_policy.biological_replicate_reduction_statistic,
            )
            supplied_estimate = getattr(uncertainty, metric_name).estimate
            if supplied_estimate != expected_estimate:
                raise ReporterResponseContractError(
                    f"dose {uncertainty.dose_uM:g}: {metric_name} estimate must equal the biological-replicate "
                    f"{uncertainty_policy.biological_replicate_reduction_statistic} of computed dose responses "
                    f"({expected_estimate!r})"
                )

    return dose_grid, measurement_rows, response_rows, uncertainty_rows


def _resolve_controls(
    observations: Mapping[str, ConditionMeasurement],
    observation_ids: tuple[str, ...],
    *,
    expected_role: Literal["baseline", "positive_control"],
    dose_observation_id: str,
) -> tuple[ConditionMeasurement, ...]:
    resolved: list[ConditionMeasurement] = []
    for observation_id in observation_ids:
        try:
            row = observations[observation_id]
        except KeyError as exc:
            raise ReporterResponseContractError(
                f"{dose_observation_id}: unknown {expected_role} observation {observation_id!r}"
            ) from exc
        if row.role != expected_role:
            raise ReporterResponseContractError(
                f"{dose_observation_id}: {observation_id!r} must have role {expected_role!r}"
            )
        resolved.append(row)
    return tuple(resolved)


def _validate_paired_control_strata(
    dose: ConditionMeasurement,
    controls: tuple[ConditionMeasurement, ...],
) -> None:
    for control in controls:
        if control.acquisition_id != dose.acquisition_id:
            raise ReporterResponseContractError(
                f"{dose.observation_id}: paired control {control.observation_id!r} must share "
                "acquisition_id with the dose observation"
            )


def comparability_key(
    *,
    observation_policy_digest: str,
    reduction: Reduction,
    dose_grid_uM: tuple[float, ...],
    dose_uncertainties: tuple[DoseUncertainty, ...],
) -> str:
    payload = {
        "observation_policy_digest": observation_policy_digest,
        "uncertainty": _uncertainty_comparability_payload(dose_uncertainties),
        "reduction": _json_value(asdict(reduction)),
        "dose_grid_uM": list(dose_grid_uM),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _uncertainty_comparability_payload(
    dose_uncertainties: tuple[DoseUncertainty, ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in dose_uncertainties:
        support_modes = {
            _biological_replicate_support_mode(row.normalized_reporter_response),
            _biological_replicate_support_mode(row.relative_od),
        }
        if len(support_modes) != 1:
            raise ReporterResponseContractError(f"dose {row.dose_uM:g}: metric identity-support modes must agree")
        dose_payload: dict[str, object] = {
            "dose_uM": row.dose_uM,
            "biological_replicate_support": {
                "mode": support_modes.pop(),
                "count": row.biological_replicate_count,
            },
        }
        for metric_name in ("normalized_reporter_response", "relative_od"):
            metric = getattr(row, metric_name)
            if isinstance(metric, EstimatedMetricUncertainty):
                dose_payload[metric_name] = {
                    "status": metric.status,
                    "method": metric.method,
                    "resampling_unit": metric.resampling_unit,
                    "confidence_level": metric.confidence_level,
                    "draws": metric.draws,
                }
            else:
                dose_payload[metric_name] = {
                    "status": metric.status,
                    "reason": metric.reason,
                }
        payload.append(dose_payload)
    return payload


def _biological_replicate_support_mode(
    metric: EstimatedMetricUncertainty | NotEstimableMetricUncertainty,
) -> Literal[
    "identity_unknown",
    "identity_declared_below_minimum",
    "identity_declared_sufficient",
]:
    if isinstance(metric, EstimatedMetricUncertainty) or metric.reason == "insufficient_valid_resamples":
        return "identity_declared_sufficient"
    if metric.reason == "biological_replicate_identity_unknown":
        return "identity_unknown"
    return "identity_declared_below_minimum"


def _reduce_biological_replicate_values(
    values: Iterable[float],
    *,
    statistic: BiologicalReplicateReductionStatistic,
) -> float:
    rows = tuple(values)
    if statistic == "median":
        return float(statistics.median(rows))
    return float(statistics.mean(rows))


def _validate_estimability(
    uncertainty: DoseUncertainty,
    *,
    policy: UncertaintyPolicy,
    pairing_kind: PairingKind,
    identity_complete: bool,
) -> None:
    required_reasons: set[str] = set()
    if not identity_complete:
        required_reasons.add("biological_replicate_identity_unknown")
    elif uncertainty.biological_replicate_count < policy.minimum_biological_replicates:
        required_reasons.add("below_minimum_biological_replicates")

    for metric_name in ("normalized_reporter_response", "relative_od"):
        metric = getattr(uncertainty, metric_name)
        if required_reasons:
            if isinstance(metric, EstimatedMetricUncertainty):
                raise ReporterResponseContractError(
                    f"dose {uncertainty.dose_uM:g}: {metric_name} uncertainty cannot be estimated "
                    "without complete declared biological-replicate support"
                )
            if metric.reason not in required_reasons:
                raise ReporterResponseContractError(
                    f"dose {uncertainty.dose_uM:g}: {metric_name} not-estimable reason must identify "
                    "the unmet biological-replicate identity requirement"
                )
            continue
        if isinstance(metric, NotEstimableMetricUncertainty):
            if metric.reason in {
                "biological_replicate_identity_unknown",
                "below_minimum_biological_replicates",
            }:
                raise ReporterResponseContractError(
                    f"dose {uncertainty.dose_uM:g}: {metric_name} not-estimable reason conflicts "
                    "with observed biological-replicate identities"
                )
            continue
        if metric.resampling_unit == "paired_biological_replicate" and pairing_kind != "paired_by_design":
            raise ReporterResponseContractError(
                f"dose {uncertainty.dose_uM:g}: paired biological-replicate resampling requires paired_by_design"
            )


__all__ = ["comparability_key", "derive_profile_rows"]
