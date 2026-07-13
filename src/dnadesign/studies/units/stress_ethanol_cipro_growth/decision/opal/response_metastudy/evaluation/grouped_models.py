"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/grouped_models.py

Fixed model contracts and grouped prediction fitting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from .model_validation_support import validated_model_params


@dataclass(frozen=True)
class ModelScreenSpec:
    """One fixed model and target-transform challenger."""

    id: str
    kind: Literal["mean", "random_forest", "pca_ridge", "pls"]
    components: int | None = None
    ridge_alpha: float | None = None
    target_transform: Literal["none", "robust_magnitude", "standard_all"] = "none"


DEFAULT_MODEL_SCREEN_SPECS: tuple[ModelScreenSpec, ...] = (
    ModelScreenSpec(id="mean_baseline", kind="mean"),
    ModelScreenSpec(id="robust_target_random_forest", kind="random_forest", target_transform="robust_magnitude"),
    ModelScreenSpec(
        id="pca4_ridge10",
        kind="pca_ridge",
        components=4,
        ridge_alpha=10.0,
        target_transform="standard_all",
    ),
    ModelScreenSpec(
        id="pca8_ridge10",
        kind="pca_ridge",
        components=8,
        ridge_alpha=10.0,
        target_transform="standard_all",
    ),
    ModelScreenSpec(
        id="pca12_ridge10",
        kind="pca_ridge",
        components=12,
        ridge_alpha=10.0,
        target_transform="standard_all",
    ),
    ModelScreenSpec(id="pls2", kind="pls", components=2),
    ModelScreenSpec(id="pls4", kind="pls", components=4),
    ModelScreenSpec(id="pls6", kind="pls", components=6),
)


def grouped_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    model_spec: ModelScreenSpec,
    random_forest_params: Mapping[str, object],
    pca_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    magnitude_start: int,
) -> np.ndarray:
    """Fit one fixed challenger under leave-one-group-out validation."""

    validated_x = np.asarray(x, dtype=float)
    validated_y = np.asarray(y, dtype=float)
    if (
        validated_x.ndim != 2
        or validated_y.ndim != 2
        or len(validated_x) != len(validated_y)
        or validated_y.shape[1] == 0
        or not np.isfinite(validated_x).all()
        or not np.isfinite(validated_y).all()
    ):
        raise ValueError("grouped model predictions require aligned finite two-dimensional X and y matrices.")
    predicted = np.empty_like(validated_y, dtype=float)
    for fold_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(validated_x, validated_y, groups)):
        if model_spec.kind == "pca_ridge":
            _fit_cached_pca_ridge(
                validated_x,
                validated_y,
                predicted,
                train_index=train_index,
                test_index=test_index,
                fold_index=fold_index,
                model_spec=model_spec,
                pca_cache=pca_cache,
            )
            continue
        model = _build_model(
            model_spec,
            train_rows=len(train_index),
            random_forest_params=random_forest_params,
        )
        predicted[test_index] = _fit_predict_target_transform(
            model,
            x_train=validated_x[train_index],
            y_train=validated_y[train_index],
            x_test=validated_x[test_index],
            transform=model_spec.target_transform,
            magnitude_start=magnitude_start,
        )
    return predicted


def _fit_cached_pca_ridge(
    x: np.ndarray,
    y: np.ndarray,
    predicted: np.ndarray,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    fold_index: int,
    model_spec: ModelScreenSpec,
    pca_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    if model_spec.target_transform != "standard_all":
        raise ValueError(f"{model_spec.id}: PCA-ridge requires target_transform='standard_all'.")
    if model_spec.components is None or model_spec.components <= 0 or model_spec.components >= len(train_index):
        raise ValueError(f"{model_spec.id}: components must be positive and smaller than every training fold.")
    if model_spec.ridge_alpha is None or model_spec.ridge_alpha <= 0.0:
        raise ValueError(f"{model_spec.id}: pca_ridge requires a positive ridge alpha.")
    cache_key = (model_spec.components, fold_index)
    if cache_key not in pca_cache:
        pca = PCA(n_components=model_spec.components, svd_solver="randomized", random_state=7)
        pca_cache[cache_key] = (
            train_index,
            test_index,
            pca.fit_transform(x[train_index]),
            pca.transform(x[test_index]),
        )
    cached_train, cached_test, x_train, x_test = pca_cache[cache_key]
    if not np.array_equal(cached_train, train_index) or not np.array_equal(cached_test, test_index):
        raise RuntimeError("PCA fold cache does not match the requested grouped split.")
    transformer = StandardScaler().fit(y[train_index])
    model = Ridge(alpha=model_spec.ridge_alpha).fit(x_train, transformer.transform(y[train_index]))
    predicted[test_index] = transformer.inverse_transform(model.predict(x_test))


def _fit_predict_target_transform(
    model,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    transform: str,
    magnitude_start: int,
) -> np.ndarray:
    if transform == "none":
        model.fit(x_train, y_train)
        return np.asarray(model.predict(x_test), dtype=float).reshape(len(x_test), y_train.shape[1])
    if transform != "robust_magnitude":
        raise ValueError(f"unsupported target transform outside PCA-ridge: {transform!r}.")
    if not 0 < magnitude_start < y_train.shape[1]:
        raise ValueError("robust magnitude transform requires a valid magnitude channel boundary.")
    center = np.zeros(y_train.shape[1], dtype=float)
    scale = np.ones(y_train.shape[1], dtype=float)
    magnitude = y_train[:, magnitude_start:]
    center[magnitude_start:] = np.median(magnitude, axis=0)
    iqr = np.percentile(magnitude, 75, axis=0) - np.percentile(magnitude, 25, axis=0)
    scale[magnitude_start:] = np.where(iqr <= 0.0, 1.0e-8, iqr)
    model.fit(x_train, (y_train - center[None, :]) / scale[None, :])
    predicted = np.asarray(model.predict(x_test), dtype=float).reshape(len(x_test), y_train.shape[1])
    return predicted * scale[None, :] + center[None, :]


def _build_model(
    spec: ModelScreenSpec,
    *,
    train_rows: int,
    random_forest_params: Mapping[str, object],
):
    if spec.kind == "mean":
        if spec.target_transform != "none":
            raise ValueError("mean baseline must use target_transform='none'.")
        return DummyRegressor(strategy="mean")
    if spec.kind == "random_forest":
        if spec.target_transform != "robust_magnitude":
            raise ValueError("random-forest challenger must use robust_magnitude target transform.")
        return RandomForestRegressor(**validated_model_params(random_forest_params))
    if spec.components is None or spec.components <= 0 or spec.components >= train_rows:
        raise ValueError(f"{spec.id}: components must be positive and smaller than every training fold.")
    if spec.kind == "pca_ridge":  # pragma: no cover - fitted through the cached path.
        raise RuntimeError("PCA-ridge must use the cached fold fitting path.")
    if spec.kind == "pls":
        if spec.target_transform != "none":
            raise ValueError("PLS handles target scaling internally and must use target_transform='none'.")
        return PLSRegression(n_components=spec.components, scale=True, max_iter=1000)
    raise ValueError(f"unsupported model screen kind: {spec.kind!r}.")


def validate_model_screen_specs(specs: Sequence[ModelScreenSpec]) -> tuple[ModelScreenSpec, ...]:
    """Require non-empty, uniquely named fixed challengers."""

    values = tuple(specs)
    if not values or len({spec.id for spec in values}) != len(values):
        raise ValueError("model screen specs must be non-empty with unique ids.")
    return values


__all__ = ["DEFAULT_MODEL_SCREEN_SPECS", "ModelScreenSpec", "grouped_predictions", "validate_model_screen_specs"]
