"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/uncertainty.py

Selected Reader-experiment joint-bootstrap uncertainty propagation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .contracts import VALUE_COLUMNS, ResponseWindowAggregationError, ResponseWindowAggregationPolicy


def selected_source_candidate_draws(
    draws: pd.DataFrame,
    *,
    label_sources: pd.DataFrame,
    policy: ResponseWindowAggregationPolicy,
) -> pd.DataFrame:
    """Resample joint Reader draws from each candidate's declared label source."""

    records: list[dict[str, object]] = []
    required = {"candidate_id", "design_id", "reader_experiment_id"}
    if missing := sorted(required - set(label_sources.columns)):
        raise ResponseWindowAggregationError(f"selected label sources are missing identity columns: {missing}")
    if label_sources["candidate_id"].astype(str).duplicated().any():
        raise ResponseWindowAggregationError("selected label sources must contain one row per candidate.")
    for source in label_sources.sort_values("candidate_id", kind="mergesort").itertuples(index=False):
        candidate_id = str(source.candidate_id)
        frame = draws.loc[
            draws["candidate_id"].astype(str).eq(candidate_id)
            & draws["design_id"].astype(str).eq(str(source.design_id))
            & draws["reader_experiment_id"].astype(str).eq(str(source.reader_experiment_id))
        ].reset_index(drop=True)
        if frame.empty:
            raise ResponseWindowAggregationError(
                f"{candidate_id}: selected label source has no Reader bootstrap draws."
            )
        rng = np.random.default_rng(_candidate_seed(policy.random_seed, candidate_id))
        for draw_index in range(policy.bootstrap_samples):
            selected_index = int(rng.integers(0, len(frame)))
            vector = frame.loc[selected_index, VALUE_COLUMNS].to_numpy(dtype=float)
            records.append(
                {
                    "candidate_id": candidate_id,
                    "draw_index": draw_index,
                    **dict(zip(VALUE_COLUMNS, vector.tolist(), strict=True)),
                }
            )
    return pd.DataFrame.from_records(records, columns=["candidate_id", "draw_index", *VALUE_COLUMNS])


def uncertainty_rows(
    observations: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    policy: ResponseWindowAggregationPolicy,
) -> pd.DataFrame:
    alpha = (1.0 - policy.confidence_level) / 2.0
    point_by_id = observations.set_index("candidate_id") if not observations.empty else observations
    rows: list[dict[str, object]] = []
    for candidate_id, frame in draws.groupby("candidate_id", sort=True):
        for component in VALUE_COLUMNS:
            values = frame[component].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "component": component,
                    "label_source_reader_experiment_id": str(
                        point_by_id.loc[candidate_id, "label_source_reader_experiment_id"]
                    ),
                    "point_estimate": float(point_by_id.loc[candidate_id, component]),
                    "bootstrap_sd": float(np.std(values, ddof=1)),
                    "descriptive_interval_low": float(np.quantile(values, alpha)),
                    "descriptive_interval_high": float(np.quantile(values, 1.0 - alpha)),
                    "nominal_interval_mass": policy.confidence_level,
                    "interval_scope": "descriptive_selected_source_joint_bootstrap",
                    "population_coverage_claimed": False,
                    "bootstrap_samples": policy.bootstrap_samples,
                }
            )
    return pd.DataFrame.from_records(rows)


def _candidate_seed(base_seed: int, candidate_id: str) -> np.random.SeedSequence:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    candidate_word = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return np.random.SeedSequence([base_seed, candidate_word])


__all__ = ["selected_source_candidate_draws", "uncertainty_rows"]
