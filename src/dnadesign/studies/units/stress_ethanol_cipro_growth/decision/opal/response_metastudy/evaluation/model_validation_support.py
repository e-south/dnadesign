"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/model_validation_support.py

Fit and metric primitives for SFXI model validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

from dnadesign.opal import SFXIScoringConfig, score_vec8_with_denom

from ..core.contracts import StressTargetView

TARGET_NAMES = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")


def validated_xy(x: np.ndarray, y: np.ndarray, *, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 2 or y_values.ndim != 2 or y_values.shape[1] != 8:
        raise ValueError("model validation requires X shape (n, d) and y shape (n, 8).")
    if len(x_values) != len(y_values):
        raise ValueError("model validation X and y row counts must match.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("model validation X and y must be finite.")
    if not (2 <= int(n_splits) <= len(x_values)):
        raise ValueError(f"n_splits must be in [2, {len(x_values)}]; got {n_splits}.")
    return x_values, y_values


def validated_model_params(
    model_params: Mapping[str, object],
    *,
    preserve_oob_score: bool = False,
) -> dict[str, object]:
    params = dict(model_params)
    params.pop("emit_feature_importance", None)
    if not preserve_oob_score:
        params["oob_score"] = False
    allowed = set(RandomForestRegressor().get_params(deep=False))
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unsupported random-forest validation params: {unknown}")
    return params


def cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: Mapping[str, object],
    seed: int,
    n_splits: int,
    yops_eps: float,
) -> np.ndarray:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _fit_folds(x, y, splitter.split(x), params=params, seed=seed, yops_eps=yops_eps)


def group_cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    params: Mapping[str, object],
    seed: int,
    yops_eps: float,
) -> np.ndarray:
    splitter = LeaveOneGroupOut()
    folds = splitter.split(x, y, groups=groups)
    return _fit_folds(x, y, folds, params=params, seed=seed, yops_eps=yops_eps)


def target_metric_rows(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    seed: int,
    split_strategy: str,
    group_count: int,
) -> list[dict[str, object]]:
    return [
        _metric_row(
            observed[:, index],
            predicted[:, index],
            seed=seed,
            scope="target",
            metric_id=target,
            split_strategy=split_strategy,
            group_count=group_count,
        )
        for index, target in enumerate(TARGET_NAMES)
    ]


def target_view_metric_rows(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    target_views: Sequence[StressTargetView],
    target_view_denoms: Mapping[str, float],
    seed: int,
    scaling_percentile: int,
    scaling_min_n: int,
    scaling_eps: float,
    intensity_log2_offset_delta: float,
    split_strategy: str,
    group_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_view in target_views:
        if target_view.id not in target_view_denoms:
            raise ValueError(f"missing persisted SFXI denominator for target view {target_view.id!r}.")
        config = SFXIScoringConfig(
            setpoint_vector=target_view.target_mask,
            scaling_percentile=scaling_percentile,
            scaling_min_n=scaling_min_n,
            scaling_eps=scaling_eps,
            intensity_log2_offset_delta=intensity_log2_offset_delta,
        )
        denom = float(target_view_denoms[target_view.id])
        observed_score = score_vec8_with_denom(observed, config, denom=denom).sfxi
        predicted_score = score_vec8_with_denom(predicted, config, denom=denom).sfxi
        rows.append(
            _metric_row(
                observed_score,
                predicted_score,
                seed=seed,
                scope="selection_view_objective",
                metric_id=target_view.id,
                split_strategy=split_strategy,
                group_count=group_count,
            )
        )
    return rows


def _fit_folds(
    x: np.ndarray,
    y: np.ndarray,
    folds,
    *,
    params: Mapping[str, object],
    seed: int,
    yops_eps: float,
) -> np.ndarray:
    predicted = np.empty_like(y, dtype=float)
    for train_index, test_index in folds:
        center, scale = _intensity_center_scale(y[train_index], eps=yops_eps)
        y_train = y[train_index].copy()
        y_train[:, 4:8] = (y_train[:, 4:8] - center[None, :]) / scale[None, :]
        model = RandomForestRegressor(**{**params, "random_state": seed, "n_jobs": 1})
        model.fit(x[train_index], y_train)
        fold_prediction = np.asarray(model.predict(x[test_index]), dtype=float)
        fold_prediction[:, 4:8] = fold_prediction[:, 4:8] * scale[None, :] + center[None, :]
        predicted[test_index] = fold_prediction
    return predicted


def _intensity_center_scale(y: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"yops_eps must be positive and finite; got {eps}.")
    intensity = y[:, 4:8]
    center = np.median(intensity, axis=0)
    scale = np.percentile(intensity, 75, axis=0) - np.percentile(intensity, 25, axis=0)
    return center, np.where(scale <= 0.0, float(eps), scale)


def _metric_row(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    seed: int,
    scope: str,
    metric_id: str,
    split_strategy: str,
    group_count: int,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "scope": scope,
        "metric_id": metric_id,
        "split_strategy": split_strategy,
        "group_count": int(group_count),
        "n": int(len(observed)),
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
    }
