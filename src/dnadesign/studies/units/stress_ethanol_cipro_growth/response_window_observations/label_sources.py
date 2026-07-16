"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/label_sources.py

Select explicit Reader experiment sources for candidate observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import CANDIDATE_METADATA_COLUMNS, VALUE_COLUMNS, ResponseWindowAggregationError


def build_label_source_observations(
    primary: pd.DataFrame,
    *,
    decision_by_id: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Project one declared exact label source while retaining every evidence row."""

    blockers: list[str] = []
    contribution_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for candidate_id, frame in primary.groupby("candidate_id", sort=True):
        candidate_id = str(candidate_id)
        experiment_count = int(frame["reader_experiment_id"].nunique())
        decision = decision_by_id.get(candidate_id)
        status = "singleton" if experiment_count == 1 else str(decision["status"])
        label_source_id = _label_source_id(frame, status=status, decision=decision)
        if status == "review_required":
            blockers.append(f"{candidate_id}: repeated experiments require an explicit label-source decision")
        contribution = _contribution_rows(
            frame,
            status=status,
            decision=decision,
            label_source_id=label_source_id,
        )
        contribution_frames.append(contribution)
        included = contribution["included_in_label"].astype(bool)
        if not included.any():
            continue
        label_source = contribution.loc[included].iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "reader_design_ids": sorted(frame["design_id"].astype(str).unique().tolist()),
                "reader_experiment_count": experiment_count,
                "label_source_reader_experiment_id": str(label_source["reader_experiment_id"]),
                "label_source_method": ("singleton_identity" if experiment_count == 1 else "explicit_repeat_selection"),
                **_candidate_metadata(frame),
                **{component: float(label_source[component]) for component in VALUE_COLUMNS},
            }
        )
    observation_columns = [
        "candidate_id",
        "reader_design_ids",
        "reader_experiment_count",
        "label_source_reader_experiment_id",
        "label_source_method",
        *(column for column in CANDIDATE_METADATA_COLUMNS if column in primary.columns),
        *VALUE_COLUMNS,
    ]
    observations = (
        pd.DataFrame.from_records(rows, columns=observation_columns)
        .sort_values("candidate_id", kind="mergesort")
        .reset_index(drop=True)
    )
    contributions = (
        pd.concat(contribution_frames, ignore_index=True)
        .sort_values(["candidate_id", "reader_experiment_id", "design_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return observations, contributions, blockers


def validate_label_source_identity(
    *,
    candidate_id: str,
    status: str,
    value: object,
    observed_experiments: list[str],
) -> str | None:
    """Validate one repeat decision's explicit source against observed experiments."""

    if status == "label_source_selected":
        selected = str(value).strip()
        if selected not in observed_experiments:
            raise ResponseWindowAggregationError(
                f"{candidate_id}: selected Reader experiment is not one of the declared experiment identities."
            )
        return selected
    if not _missing(value):
        raise ResponseWindowAggregationError(
            f"{candidate_id}: repeat decision without selected status cannot name a label source."
        )
    return None


def _label_source_id(
    frame: pd.DataFrame,
    *,
    status: str,
    decision: dict[str, object] | None,
) -> str | None:
    if status == "singleton":
        return str(frame["reader_experiment_id"].iloc[0])
    if status == "label_source_selected":
        return str(decision["label_source_reader_experiment_id"])
    return None


def _contribution_rows(
    frame: pd.DataFrame,
    *,
    status: str,
    decision: dict[str, object] | None,
    label_source_id: str | None,
) -> pd.DataFrame:
    contribution = frame.copy()
    contribution["repeat_decision"] = status
    contribution["repeat_decision_reason"] = "single_experiment" if decision is None else str(decision["reason"])
    for column in ("classification", "evidence_artifact", "evidence_sha256", "adjudicated_by", "adjudicated_at"):
        contribution[f"repeat_{column}"] = None if decision is None else decision[column]
    contribution["label_source_reader_experiment_id"] = label_source_id
    selected = (
        contribution["reader_experiment_id"].astype(str).eq(label_source_id)
        if label_source_id is not None
        else pd.Series(False, index=contribution.index)
    )
    if selected.sum() > 1:
        candidate_id = str(contribution["candidate_id"].iloc[0])
        raise ResponseWindowAggregationError(
            f"{candidate_id}: selected Reader experiment resolves to multiple primary rows."
        )
    exact = _exact_primary_rows(contribution)
    included = selected & exact
    contribution["selected_as_label_source"] = selected.astype(bool)
    contribution["included_in_label"] = included.astype(bool)
    contribution["label_exclusion_reason"] = [
        _label_exclusion_reason(status=status, selected_as_source=bool(is_selected), exact=bool(is_exact))
        for is_selected, is_exact in zip(selected, exact, strict=True)
    ]
    return contribution


def _candidate_metadata(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in CANDIDATE_METADATA_COLUMNS:
        if column not in frame.columns:
            continue
        values = sorted(set(frame[column].dropna().astype(str)))
        if len(values) != 1 or not values[0].strip():
            raise ResponseWindowAggregationError(
                f"candidate {frame['candidate_id'].iloc[0]!r} has non-invariant {column!r} metadata."
            )
        result[column] = values[0]
    return result


def _exact_primary_rows(frame: pd.DataFrame) -> pd.Series:
    columns = [f"{component}_bound_kind" for component in VALUE_COLUMNS]
    return frame.loc[:, columns].astype(str).eq("exact").all(axis=1)


def _label_exclusion_reason(*, status: str, selected_as_source: bool, exact: bool) -> str | None:
    if selected_as_source:
        return None if exact else "nonexact_primary_component"
    return {
        "singleton": "not_selected_as_label_source",
        "label_source_selected": "not_selected_repeat_evidence",
        "review_required": "repeat_review_required",
        "label_source_excluded": "repeat_source_disagreement",
        "remeasure_required": "repeat_remeasurement_required",
    }[status]


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) or not str(value).strip()


__all__ = ["build_label_source_observations", "validate_label_source_identity"]
