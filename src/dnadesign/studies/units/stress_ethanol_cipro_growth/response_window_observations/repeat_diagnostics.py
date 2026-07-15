"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/repeat_diagnostics.py

Component-level disagreement evidence for repeated Reader experiments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import VALUE_COLUMNS

REPEAT_DIAGNOSTIC_COLUMNS = (
    "candidate_id",
    "component",
    "experiment_count",
    "minimum",
    "median",
    "maximum",
    "range",
    "status",
    "classification",
    "evidence_artifact",
    "evidence_sha256",
    "adjudicated_by",
    "adjudicated_at",
    "reason",
)


def repeat_diagnostic_rows(
    primary_measurements: pd.DataFrame,
    *,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Describe source disagreement without turning it into an acceptance cutoff."""

    decision_by_id = decisions.set_index("candidate_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for candidate_id, frame in primary_measurements.groupby("candidate_id", sort=True):
        experiment_count = int(frame["reader_experiment_id"].nunique())
        if experiment_count < 2:
            continue
        decision = decision_by_id[str(candidate_id)]
        for component in VALUE_COLUMNS:
            values = frame[component].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "component": component,
                    "experiment_count": experiment_count,
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values)),
                    "status": str(decision["status"]),
                    "classification": str(decision["classification"]),
                    "evidence_artifact": decision["evidence_artifact"],
                    "evidence_sha256": decision["evidence_sha256"],
                    "adjudicated_by": decision["adjudicated_by"],
                    "adjudicated_at": decision["adjudicated_at"],
                    "reason": str(decision["reason"]),
                }
            )
    return pd.DataFrame.from_records(rows, columns=REPEAT_DIAGNOSTIC_COLUMNS)


__all__ = ["REPEAT_DIAGNOSTIC_COLUMNS", "repeat_diagnostic_rows"]
