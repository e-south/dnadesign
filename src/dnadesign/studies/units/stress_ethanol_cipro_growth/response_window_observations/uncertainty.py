"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/uncertainty.py

Hierarchical experiment and Reader-draw uncertainty propagation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .contracts import VALUE_COLUMNS, ResponseWindowAggregationPolicy


def hierarchical_candidate_draws(
    draws: pd.DataFrame,
    *,
    candidate_ids: list[str],
    policy: ResponseWindowAggregationPolicy,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for candidate_id in sorted(candidate_ids):
        frame = draws.loc[draws["candidate_id"].eq(candidate_id)]
        experiments = sorted(frame["reader_experiment_id"].unique().tolist())
        rows_by_experiment = {
            experiment_id: frame.loc[frame["reader_experiment_id"].eq(experiment_id)].reset_index(drop=True)
            for experiment_id in experiments
        }
        rng = np.random.default_rng(_candidate_seed(policy.random_seed, candidate_id))
        for draw_index in range(policy.bootstrap_samples):
            sampled_experiments = rng.choice(experiments, size=len(experiments), replace=True)
            sampled_vectors = []
            for experiment_id in sampled_experiments:
                evidence = rows_by_experiment[str(experiment_id)]
                selected_index = int(rng.integers(0, len(evidence)))
                sampled_vectors.append(evidence.loc[selected_index, VALUE_COLUMNS].to_numpy(dtype=float))
            aggregate = np.median(np.vstack(sampled_vectors), axis=0)
            records.append(
                {
                    "candidate_id": candidate_id,
                    "draw_index": draw_index,
                    **dict(zip(VALUE_COLUMNS, aggregate.tolist(), strict=True)),
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
                    "experiment_count": int(point_by_id.loc[candidate_id, "experiment_count"]),
                    "point_estimate": float(point_by_id.loc[candidate_id, component]),
                    "hierarchical_bootstrap_sd": float(np.std(values, ddof=1)),
                    "descriptive_interval_low": float(np.quantile(values, alpha)),
                    "descriptive_interval_high": float(np.quantile(values, 1.0 - alpha)),
                    "nominal_interval_mass": policy.confidence_level,
                    "interval_scope": "descriptive_hierarchical_bootstrap",
                    "population_coverage_claimed": False,
                    "bootstrap_samples": policy.bootstrap_samples,
                }
            )
    return pd.DataFrame.from_records(rows)


def _candidate_seed(base_seed: int, candidate_id: str) -> np.random.SeedSequence:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    candidate_word = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return np.random.SeedSequence([base_seed, candidate_word])


__all__ = ["hierarchical_candidate_draws", "uncertainty_rows"]
