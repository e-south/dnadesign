"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/model_validation.py

Repeated held-out validation for the round-0 vec8 predictor.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.contracts import StressTargetView
from .model_validation_support import (
    cross_validated_predictions,
    group_cross_validated_predictions,
    target_metric_rows,
    target_view_metric_rows,
    validated_model_params,
    validated_xy,
)


def cross_validate_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_views: Sequence[StressTargetView],
    target_view_denoms: Mapping[str, float],
    model_params: Mapping[str, object],
    seeds: Sequence[int],
    n_splits: int,
    yops_eps: float,
    scaling_percentile: int,
    scaling_min_n: int,
    scaling_eps: float,
    intensity_log2_offset_delta: float,
) -> pd.DataFrame:
    x_values, y_values = validated_xy(x, y, n_splits=n_splits)
    params = validated_model_params(model_params)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        predicted = cross_validated_predictions(
            x_values,
            y_values,
            params=params,
            seed=int(seed),
            n_splits=int(n_splits),
            yops_eps=float(yops_eps),
        )
        rows.extend(
            target_metric_rows(
                y_values,
                predicted,
                seed=int(seed),
                split_strategy="shuffled_kfold",
                group_count=0,
            )
        )
        rows.extend(
            target_view_metric_rows(
                y_values,
                predicted,
                target_views=target_views,
                target_view_denoms=target_view_denoms,
                seed=int(seed),
                scaling_percentile=int(scaling_percentile),
                scaling_min_n=int(scaling_min_n),
                scaling_eps=float(scaling_eps),
                intensity_log2_offset_delta=float(intensity_log2_offset_delta),
                split_strategy="shuffled_kfold",
                group_count=0,
            )
        )
    result = pd.DataFrame(rows)
    result[["r2", "mae", "spearman"]] = result[["r2", "mae", "spearman"]].round(12)
    return result


def cross_validate_random_forest_by_group(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: Sequence[object],
    target_views: Sequence[StressTargetView],
    target_view_denoms: Mapping[str, float],
    model_params: Mapping[str, object],
    seeds: Sequence[int],
    yops_eps: float,
    scaling_percentile: int,
    scaling_min_n: int,
    scaling_eps: float,
    intensity_log2_offset_delta: float,
) -> pd.DataFrame:
    """Repeat RF fitting while holding out complete Reader experiment groups."""

    group_values = np.asarray(groups, dtype=object).ravel()
    unique_groups = np.unique(group_values.astype(str))
    if len(unique_groups) < 2:
        raise ValueError("grouped model validation requires at least two groups.")
    x_values, y_values = validated_xy(x, y, n_splits=len(unique_groups))
    if len(group_values) != len(x_values):
        raise ValueError("grouped model validation groups must align one-to-one with X and y rows.")
    if any(not str(group).strip() for group in group_values):
        raise ValueError("grouped model validation groups must be non-empty.")
    params = validated_model_params(model_params)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        predicted = group_cross_validated_predictions(
            x_values,
            y_values,
            groups=group_values,
            params=params,
            seed=int(seed),
            yops_eps=float(yops_eps),
        )
        rows.extend(
            target_metric_rows(
                y_values,
                predicted,
                seed=int(seed),
                split_strategy="leave_one_experiment_out",
                group_count=len(unique_groups),
            )
        )
        rows.extend(
            target_view_metric_rows(
                y_values,
                predicted,
                target_views=target_views,
                target_view_denoms=target_view_denoms,
                seed=int(seed),
                scaling_percentile=int(scaling_percentile),
                scaling_min_n=int(scaling_min_n),
                scaling_eps=float(scaling_eps),
                intensity_log2_offset_delta=float(intensity_log2_offset_delta),
                split_strategy="leave_one_experiment_out",
                group_count=len(unique_groups),
            )
        )
    result = pd.DataFrame(rows)
    result[["r2", "mae", "spearman"]] = result[["r2", "mae", "spearman"]].round(12)
    return result


def summarize_model_validation(frame: pd.DataFrame, *, split_strategy: str) -> dict[str, object]:
    selected = frame.loc[frame["split_strategy"].astype(str).eq(split_strategy)]
    if selected.empty:
        raise ValueError(f"model validation has no rows for split strategy {split_strategy!r}.")
    target_view_rows = selected[selected["scope"] == "selection_view_objective"]
    target = selected[selected["scope"] == "target"]
    target_view_medians = target_view_rows.groupby("metric_id", sort=True)["spearman"].median()
    return {
        "method": split_strategy,
        "seed_count": int(selected["seed"].nunique()),
        "group_count": int(selected["group_count"].max()),
        "weakest_target_view_median_score_spearman": float(target_view_medians.min()),
        "target_view_median_score_spearman": {str(key): float(value) for key, value in target_view_medians.items()},
        "median_target_spearman": float(target["spearman"].median()),
        "median_target_r2": float(target["r2"].median()),
    }
