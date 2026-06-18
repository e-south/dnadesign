"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/evaluation/prediction_scoring.py

Prediction scoring helpers for DenseGen motif-QA probe metrics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.constants import AXIS_CLASS_TO_LOGIC4

_AXIS_CLASS_NAMES = tuple(AXIS_CLASS_TO_LOGIC4)
_AXIS_LOGIC4_MATRIX = np.asarray([AXIS_CLASS_TO_LOGIC4[name] for name in _AXIS_CLASS_NAMES], dtype=float)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    true = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    if true.size == 0:
        return float("nan")
    scores: list[float] = []
    for label in _AXIS_CLASS_NAMES:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else float((2 * tp) / denom))
    return float(np.mean(scores))


def label_lookup(labels: pd.DataFrame, *, class_column: str = "axis_class") -> pd.Series:
    return labels.set_index(labels["id"].astype(str))[class_column]


def validate_prediction_selection_contract(predictions: pd.DataFrame) -> None:
    selected_bool_mask(predictions)
    rank_competition(predictions)


def selected_bool_mask(frame: pd.DataFrame) -> pd.Series:
    if "sel__is_selected" not in frame.columns:
        raise RuntimeError("OPAL predictions missing required column: sel__is_selected")
    values = frame["sel__is_selected"]
    if values.isna().any():
        raise RuntimeError("OPAL predictions contain null sel__is_selected values")
    if not pd.api.types.is_bool_dtype(values):
        bad = values.loc[~values.map(lambda value: isinstance(value, (bool, np.bool_)))]
        if not bad.empty:
            preview = ", ".join(repr(value) for value in bad.head(5).tolist())
            raise RuntimeError(f"OPAL predictions sel__is_selected must be boolean; got {preview}")
    return values.astype(bool)


def rank_competition(frame: pd.DataFrame) -> pd.Series:
    if "sel__rank_competition" not in frame.columns:
        raise RuntimeError("OPAL predictions missing required column: sel__rank_competition")
    ranks = pd.to_numeric(frame["sel__rank_competition"], errors="coerce")
    if ranks.isna().any() or not np.isfinite(ranks.to_numpy(dtype=float)).all():
        raise RuntimeError("OPAL predictions contain non-finite sel__rank_competition values")
    if (ranks <= 0).any():
        raise RuntimeError("OPAL predictions sel__rank_competition must be positive")
    return ranks


def top_ids_from_prediction_frame(frame: pd.DataFrame, *, k: int) -> list[str]:
    score_col = "pred__score_selected"
    if score_col not in frame.columns:
        raise RuntimeError(f"OPAL predictions missing required column: {score_col}")
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    if not np.isfinite(frame[score_col].to_numpy(dtype=float)).all():
        raise RuntimeError(f"OPAL predictions contain non-finite {score_col} values")
    selected = frame.loc[selected_bool_mask(frame)].copy()
    if selected.empty:
        return []
    selected["sel__rank_competition"] = rank_competition(selected)
    selected = selected.sort_values(["sel__rank_competition", score_col, "id"], ascending=[True, False, True])
    return selected["id"].astype(str).head(int(k)).tolist()


def predicted_axis_classes(values: Sequence[Any]) -> list[str]:
    vectors = [np.asarray(value, dtype=float).ravel() for value in values]
    bad_dims = sorted({int(vector.size) for vector in vectors if vector.size != 4})
    if bad_dims:
        raise RuntimeError(f"OPAL prediction pred__y_hat_model must be logic4, got dimension(s): {bad_dims}")
    if not vectors:
        return []
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise RuntimeError("OPAL prediction pred__y_hat_model contains non-finite values")
    distances = np.linalg.norm(matrix[:, None, :] - _AXIS_LOGIC4_MATRIX[None, :, :], axis=2)
    best = np.argmin(distances, axis=1)
    return [_AXIS_CLASS_NAMES[int(index)] for index in best]
