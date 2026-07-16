"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_label_source_validation.py

Validate published contribution and label-source semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifact_contract import ResponseWindowObservationArtifactError
from .censoring import bounded_label_blockers
from .contracts import VALUE_COLUMNS


def validate_label_source_contributions(frame: pd.DataFrame, *, candidate_ids: set[str]) -> None:
    required = {
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "repeat_decision",
        "repeat_decision_reason",
        "repeat_classification",
        "repeat_evidence_artifact",
        "repeat_evidence_sha256",
        "repeat_adjudicated_by",
        "repeat_adjudicated_at",
        "label_source_reader_experiment_id",
        "selected_as_label_source",
        "included_in_label",
        "label_exclusion_reason",
        *VALUE_COLUMNS,
    }
    if missing := sorted(required - set(frame.columns)):
        raise ResponseWindowObservationArtifactError(f"observation contributions disagree: missing={missing}")
    for column in ("selected_as_label_source", "included_in_label"):
        if not frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ResponseWindowObservationArtifactError(f"contribution {column!r} flags must be boolean.")
    selected = frame["selected_as_label_source"].astype(bool)
    included = frame["included_in_label"].astype(bool)
    if (included & ~selected).any():
        raise ResponseWindowObservationArtifactError("included contributions must be the selected label source.")
    selected_counts = frame.assign(_selected=selected).groupby("candidate_id")["_selected"].sum()
    included_counts = frame.assign(_included=included).groupby("candidate_id")["_included"].sum()
    if selected_counts.gt(1).any() or included_counts.gt(1).any():
        raise ResponseWindowObservationArtifactError("candidate contributions must select at most one label source.")
    for candidate_id, rows in frame.groupby("candidate_id", sort=True):
        _validate_candidate_source(str(candidate_id), rows, selected=selected.loc[rows.index])
    included_ids = set(frame.loc[included, "candidate_id"].astype(str))
    if included_ids != candidate_ids:
        raise ResponseWindowObservationArtifactError("included contribution candidates disagree with observations.")
    reasons = frame["label_exclusion_reason"]
    missing_exclusion = reasons.loc[~included].map(lambda value: value is None or not str(value).strip())
    if reasons.loc[included].notna().any() or missing_exclusion.any():
        raise ResponseWindowObservationArtifactError(
            "included contributions cannot carry exclusion reasons and excluded contributions require one."
        )
    if blockers := bounded_label_blockers(frame):
        raise ResponseWindowObservationArtifactError(f"included label sources contain bounded components: {blockers}")
    values = frame.loc[:, VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ResponseWindowObservationArtifactError("contributions contain non-finite response vectors.")


def _validate_candidate_source(candidate_id: str, rows: pd.DataFrame, *, selected: pd.Series) -> None:
    source_ids = rows["label_source_reader_experiment_id"].dropna().astype(str).unique().tolist()
    if len(source_ids) > 1:
        raise ResponseWindowObservationArtifactError(f"{candidate_id}: contribution label-source identities disagree.")
    expected = (
        rows["reader_experiment_id"].astype(str).eq(source_ids[0]) if source_ids else pd.Series(False, index=rows.index)
    )
    if not selected.eq(expected).all():
        raise ResponseWindowObservationArtifactError(
            f"{candidate_id}: selected-source flag disagrees with the declared Reader experiment."
        )


def validate_repeat_diagnostic_label_source(
    row: pd.Series,
    *,
    decision: pd.Series,
    contributions: pd.DataFrame,
    candidate_id: str,
    component: str,
) -> None:
    """Bind a repeat diagnostic's selected value to its declared contribution."""

    label_source_id = decision["label_source_reader_experiment_id"]
    if _missing(label_source_id):
        if not _missing(row["label_source_reader_experiment_id"]) or not _missing(row["label_source_value"]):
            raise ResponseWindowObservationArtifactError(
                f"repeat diagnostic label source must be empty for {candidate_id!r}."
            )
        return
    source_rows = contributions.loc[contributions["reader_experiment_id"].astype(str).eq(str(label_source_id))]
    if len(source_rows) != 1:
        raise ResponseWindowObservationArtifactError(
            f"repeat diagnostic label source is ambiguous for {candidate_id!r}."
        )
    if str(row["label_source_reader_experiment_id"]) != str(label_source_id) or not np.isclose(
        float(row["label_source_value"]),
        float(source_rows.iloc[0][component]),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ResponseWindowObservationArtifactError(
            f"repeat diagnostic label-source value disagrees for {candidate_id!r} {component!r}."
        )


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) or not str(value).strip()


__all__ = ["validate_label_source_contributions", "validate_repeat_diagnostic_label_source"]
