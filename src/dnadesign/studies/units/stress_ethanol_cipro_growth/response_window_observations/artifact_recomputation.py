"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_recomputation.py

Recompute published observation semantics from experiment contributions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .artifact_contract import ResponseWindowObservationArtifactError
from .contracts import VALUE_COLUMNS

_RTOL = 1.0e-12
_ATOL = 1.0e-12


def validate_recomputed_observations(
    observations: pd.DataFrame,
    contributions: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    """Bind candidate observations and reported uncertainty points to source rows."""

    duplicate = contributions.duplicated(subset=["candidate_id", "reader_experiment_id"], keep=False)
    if duplicate.any():
        raise ResponseWindowObservationArtifactError(
            "published contributions must contain one evidence row per candidate and Reader experiment."
        )
    included = contributions.loc[contributions["included_in_label"].astype(bool)].copy()
    observation_by_id = observations.set_index("candidate_id")
    for candidate_id, frame in included.groupby("candidate_id", sort=True):
        candidate_id = str(candidate_id)
        row = observation_by_id.loc[candidate_id]
        experiment_count = int(frame["reader_experiment_id"].astype(str).nunique())
        if not _exact_int(row["experiment_count"], expected=experiment_count):
            raise ResponseWindowObservationArtifactError(
                f"observation experiment count disagrees with contributions for {candidate_id!r}."
            )
        expected_method = _aggregation_method(experiment_count)
        if row["aggregation_method"] != expected_method:
            raise ResponseWindowObservationArtifactError(
                f"observation aggregation method disagrees with contributions for {candidate_id!r}."
            )
        expected_design_ids = sorted(frame["design_id"].astype(str).unique().tolist())
        if _text_sequence(row["reader_design_ids"]) != expected_design_ids:
            raise ResponseWindowObservationArtifactError(
                f"observation Reader design IDs disagree with contributions for {candidate_id!r}."
            )
        weights = pd.to_numeric(frame["experiment_weight"], errors="coerce").to_numpy(dtype=float)
        expected_weight = 1.0 / experiment_count
        if not np.isfinite(weights).all() or not np.allclose(weights, expected_weight, rtol=_RTOL, atol=_ATOL):
            raise ResponseWindowObservationArtifactError(
                f"observation contributions do not use equal experiment weights for {candidate_id!r}."
            )
        expected_point = np.median(frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=float), axis=0)
        observed_point = row.loc[list(VALUE_COLUMNS)].to_numpy(dtype=float)
        if not np.allclose(observed_point, expected_point, rtol=_RTOL, atol=_ATOL):
            raise ResponseWindowObservationArtifactError(
                f"observation point estimate disagrees with contributions for {candidate_id!r}."
            )
    _validate_uncertainty_points(observation_by_id, uncertainty)


def _validate_uncertainty_points(observation_by_id: pd.DataFrame, uncertainty: pd.DataFrame) -> None:
    for row in uncertainty.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        component = str(row.component)
        try:
            observed = float(row.point_estimate)
            expected = float(observation_by_id.loc[candidate_id, component])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseWindowObservationArtifactError(
                "uncertainty point estimate identity or value is invalid."
            ) from exc
        if not np.isfinite(observed) or not np.isclose(observed, expected, rtol=_RTOL, atol=_ATOL):
            raise ResponseWindowObservationArtifactError(
                f"uncertainty point estimate disagrees with observation for {candidate_id!r} {component!r}."
            )


def _aggregation_method(experiment_count: int) -> str:
    if experiment_count == 1:
        return "single_experiment"
    if experiment_count == 2:
        return "two_experiment_midpoint"
    return "componentwise_experiment_median"


def _exact_int(value: object, *, expected: int) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and int(value) == expected


def _text_sequence(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


__all__ = ["validate_recomputed_observations"]
