"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_repeat_validation.py

Recompute repeated-experiment decisions and disagreement evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifact_contract import ResponseWindowObservationArtifactError
from .contracts import DECISION_COLUMNS, VALUE_COLUMNS, ResponseWindowAggregationError
from .repeat_adjudication import validate_repeat_adjudications
from .repeat_diagnostics import REPEAT_DIAGNOSTIC_COLUMNS

_DECISION_SOURCE_COLUMNS = {
    "status": "repeat_decision",
    "classification": "repeat_classification",
    "evidence_artifact": "repeat_evidence_artifact",
    "evidence_sha256": "repeat_evidence_sha256",
    "adjudicated_by": "repeat_adjudicated_by",
    "adjudicated_at": "repeat_adjudicated_at",
    "reason": "repeat_decision_reason",
}


def validate_repeat_records(diagnostics: pd.DataFrame, *, contributions: pd.DataFrame) -> None:
    """Verify final decisions and recompute every repeated-component range."""

    decisions = _decisions_from_contributions(contributions)
    try:
        validate_repeat_adjudications(decisions)
    except ResponseWindowAggregationError as exc:
        raise ResponseWindowObservationArtifactError(str(exc)) from exc
    if set(diagnostics.columns) != set(REPEAT_DIAGNOSTIC_COLUMNS):
        missing = sorted(set(REPEAT_DIAGNOSTIC_COLUMNS) - set(diagnostics.columns))
        extra = sorted(set(diagnostics.columns) - set(REPEAT_DIAGNOSTIC_COLUMNS))
        raise ResponseWindowObservationArtifactError(
            f"repeat diagnostic schema disagrees: missing={missing}, extra={extra}"
        )
    expected = {
        (str(candidate_id), component) for candidate_id in decisions["candidate_id"] for component in VALUE_COLUMNS
    }
    observed = set(diagnostics[["candidate_id", "component"]].astype(str).itertuples(index=False, name=None))
    if observed != expected or diagnostics.duplicated(subset=["candidate_id", "component"]).any():
        raise ResponseWindowObservationArtifactError("repeat diagnostic coverage disagrees with contributions.")
    _validate_diagnostic_values(diagnostics, contributions=contributions, decisions=decisions)


def _decisions_from_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, frame in contributions.groupby("candidate_id", sort=True):
        experiment_count = frame["reader_experiment_id"].astype(str).nunique()
        if experiment_count == 1:
            _validate_singleton_fields(candidate_id=str(candidate_id), frame=frame)
            continue
        values = {
            field: _invariant(frame[source], candidate_id=str(candidate_id), field=source)
            for field, source in _DECISION_SOURCE_COLUMNS.items()
        }
        if values["status"] == "review_required":
            raise ResponseWindowObservationArtifactError(
                f"{candidate_id}: unresolved repeat decisions cannot appear in a published observation bundle."
            )
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "reader_design_ids": sorted(frame["design_id"].astype(str).unique().tolist()),
                "reader_experiment_ids": sorted(frame["reader_experiment_id"].astype(str).unique().tolist()),
                **values,
            }
        )
    return pd.DataFrame.from_records(rows, columns=DECISION_COLUMNS)


def _validate_singleton_fields(*, candidate_id: str, frame: pd.DataFrame) -> None:
    if set(frame["repeat_decision"].astype(str)) != {"singleton"}:
        raise ResponseWindowObservationArtifactError(f"{candidate_id}: singleton repeat status is invalid.")
    for source in _DECISION_SOURCE_COLUMNS.values():
        if source in {"repeat_decision", "repeat_decision_reason"}:
            continue
        if frame[source].map(lambda value: not _missing(value)).any():
            raise ResponseWindowObservationArtifactError(
                f"{candidate_id}: singleton contribution carries repeat adjudication evidence."
            )


def _validate_diagnostic_values(
    diagnostics: pd.DataFrame,
    *,
    contributions: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    numeric = diagnostics[["experiment_count", "minimum", "median", "maximum", "range"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ResponseWindowObservationArtifactError("repeat diagnostics contain non-finite values.")
    diagnostic_index = diagnostics.set_index(["candidate_id", "component"])
    decision_index = decisions.set_index("candidate_id")
    for candidate_id, frame in contributions.groupby("candidate_id", sort=True):
        candidate_id = str(candidate_id)
        experiment_count = frame["reader_experiment_id"].astype(str).nunique()
        if experiment_count < 2:
            continue
        decision = decision_index.loc[candidate_id]
        for component in VALUE_COLUMNS:
            row = diagnostic_index.loc[(candidate_id, component)]
            values = frame[component].to_numpy(dtype=float)
            expected_values = np.asarray(
                [experiment_count, np.min(values), np.median(values), np.max(values), np.ptp(values)],
                dtype=float,
            )
            observed_values = row[["experiment_count", "minimum", "median", "maximum", "range"]].to_numpy(dtype=float)
            if not np.allclose(observed_values, expected_values, rtol=1.0e-12, atol=1.0e-12):
                raise ResponseWindowObservationArtifactError(
                    f"repeat diagnostics disagree with contributions for {candidate_id!r} {component!r}."
                )
            for field in _DECISION_SOURCE_COLUMNS:
                if not _same(row[field], decision[field]):
                    raise ResponseWindowObservationArtifactError(
                        f"repeat diagnostic {field!r} disagrees for {candidate_id!r}."
                    )


def _invariant(values: pd.Series, *, candidate_id: str, field: str) -> object:
    normalized = {_normalized(value) for value in values}
    if len(normalized) != 1:
        raise ResponseWindowObservationArtifactError(f"{candidate_id}: repeated contributions disagree on {field!r}.")
    return values.iloc[0]


def _normalized(value: object) -> tuple[str, str]:
    return ("missing", "") if _missing(value) else ("value", str(value))


def _same(left: object, right: object) -> bool:
    return _normalized(left) == _normalized(right)


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) or not str(value).strip()


__all__ = ["validate_repeat_records"]
