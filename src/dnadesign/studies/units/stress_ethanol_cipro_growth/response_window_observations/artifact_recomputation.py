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
    observation_by_id = observations.set_index("candidate_id")
    included_ids = set(contributions.loc[contributions["included_in_label"].astype(bool), "candidate_id"].astype(str))
    if included_ids != set(observation_by_id.index.astype(str)):
        raise ResponseWindowObservationArtifactError(
            "published observations disagree with included contribution candidates."
        )
    for candidate_id, frame in contributions.groupby("candidate_id", sort=True):
        candidate_id = str(candidate_id)
        if candidate_id not in included_ids:
            continue
        row = observation_by_id.loc[candidate_id]
        reader_experiment_count = int(frame["reader_experiment_id"].astype(str).nunique())
        if not _exact_int(row["reader_experiment_count"], expected=reader_experiment_count):
            raise ResponseWindowObservationArtifactError(
                f"observation experiment count disagrees with contributions for {candidate_id!r}."
            )
        expected_method = "singleton_identity" if reader_experiment_count == 1 else "explicit_repeat_selection"
        if row["label_source_method"] != expected_method:
            raise ResponseWindowObservationArtifactError(
                f"observation label-source method disagrees with contributions for {candidate_id!r}."
            )
        expected_design_ids = sorted(frame["design_id"].astype(str).unique().tolist())
        if _text_sequence(row["reader_design_ids"]) != expected_design_ids:
            raise ResponseWindowObservationArtifactError(
                f"observation Reader design IDs disagree with contributions for {candidate_id!r}."
            )
        selected = frame.loc[frame["included_in_label"].astype(bool)]
        if len(selected) != 1:
            raise ResponseWindowObservationArtifactError(
                f"observation must have one included label source for {candidate_id!r}."
            )
        label_source_id = str(selected.iloc[0]["reader_experiment_id"])
        if str(row["label_source_reader_experiment_id"]) != label_source_id:
            raise ResponseWindowObservationArtifactError(
                f"observation label-source experiment disagrees for {candidate_id!r}."
            )
        expected_point = selected.loc[:, VALUE_COLUMNS].iloc[0].to_numpy(dtype=float)
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


def _exact_int(value: object, *, expected: int) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and int(value) == expected


def _text_sequence(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


__all__ = ["validate_recomputed_observations"]
