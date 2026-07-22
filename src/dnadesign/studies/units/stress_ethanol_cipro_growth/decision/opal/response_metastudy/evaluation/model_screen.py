"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/model_screen.py

Grouped model and label-representation screen for RMF learning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dnadesign.opal import (
    ResponseMagnitudeFeasibilityComponents,
    calibrate_response_magnitude_feasibility,
    response_magnitude_feasibility_components,
)

from ..core.contracts import StressTargetView
from .grouped_models import DEFAULT_MODEL_SCREEN_SPECS, ModelScreenSpec, grouped_predictions
from .model_representations import (
    LabelRepresentation,
    decode_to_response_magnitude,
)
from .response_uncertainty import build_calibration_table


def screen_label_models(
    x: np.ndarray,
    *,
    groups: Sequence[object],
    candidate_ids: Sequence[str],
    representations: Sequence[LabelRepresentation],
    target_views: Sequence[StressTargetView],
    uncertainty_rows: pd.DataFrame,
    scale_quantile: float,
    bootstrap_samples: int,
    random_forest_params: Mapping[str, object],
    model_specs: Sequence[ModelScreenSpec] = DEFAULT_MODEL_SCREEN_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run leave-one-experiment-out comparisons and retrospective enrichment."""

    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 2 or x_values.shape[0] < 4 or not np.isfinite(x_values).all():
        raise ValueError("model screen X must be a finite two-dimensional matrix with at least four rows.")
    group_values = np.asarray(groups, dtype=object).ravel()
    if len(group_values) != len(x_values) or len(np.unique(group_values.astype(str))) < 3:
        raise ValueError("model screen requires aligned rows from at least three experiment groups.")
    ids = np.asarray([str(value) for value in candidate_ids], dtype=object)
    if len(ids) != len(x_values) or len(set(ids.tolist())) != len(ids):
        raise ValueError("model screen requires one unique candidate id per X row.")
    pca_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, object]] = []
    group_metric_frames: list[pd.DataFrame] = []
    enrichment_frames: list[pd.DataFrame] = []
    for representation in representations:
        if len(representation.target) != len(x_values) or len(representation.response_magnitude_truth) != len(x_values):
            raise ValueError(f"{representation.id}: label rows do not align with X.")
        for spec in model_specs:
            predicted_target = grouped_predictions(
                x_values,
                representation.target,
                groups=group_values,
                model_spec=spec,
                random_forest_params=random_forest_params,
                pca_cache=pca_cache,
                magnitude_start=4 if representation.decoder == "identity_response_magnitude" else 3,
            )
            predicted_response_magnitude = decode_to_response_magnitude(
                predicted_target,
                decoder=representation.decoder,
            )
            group_metrics = _group_metric_rows(
                representation,
                spec,
                predicted_response_magnitude=predicted_response_magnitude,
                groups=group_values,
                candidate_ids=ids,
                target_views=target_views,
                uncertainty_rows=uncertainty_rows,
                scale_quantile=scale_quantile,
                bootstrap_samples=bootstrap_samples,
            )
            group_metric_frames.append(group_metrics)
            summaries.append(
                _screen_summary(
                    representation,
                    spec,
                    predicted_response_magnitude=predicted_response_magnitude,
                    target_views=target_views,
                    group_metrics=group_metrics,
                )
            )
            enrichment_frames.append(
                _retrospective_enrichment(
                    representation,
                    spec,
                    predicted_response_magnitude=predicted_response_magnitude,
                    groups=group_values,
                    candidate_ids=ids,
                    target_views=target_views,
                    uncertainty_rows=uncertainty_rows,
                    scale_quantile=scale_quantile,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return (
        pd.DataFrame.from_records(summaries),
        pd.concat(group_metric_frames, ignore_index=True),
        pd.concat(enrichment_frames, ignore_index=True),
    )


def _group_metric_rows(
    representation: LabelRepresentation,
    model_spec: ModelScreenSpec,
    *,
    predicted_response_magnitude: np.ndarray,
    groups: np.ndarray,
    candidate_ids: np.ndarray,
    target_views: Sequence[StressTargetView],
    uncertainty_rows: pd.DataFrame,
    scale_quantile: float,
    bootstrap_samples: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group in sorted(np.unique(groups.astype(str))):
        indexes = np.flatnonzero(groups.astype(str) == group)
        fold_calibration = build_calibration_table(
            uncertainty_rows,
            scale_quantile=scale_quantile,
            bootstrap_samples=bootstrap_samples,
            exclude_experiment=group,
        )
        for target_view in target_views:
            scales = _target_view_scales(fold_calibration, selection_view_id=target_view.id)
            true_components = _components(representation.response_magnitude_truth[indexes], target_view)
            predicted_components = _components(predicted_response_magnitude[indexes], target_view)
            record: dict[str, object] = {
                "representation_id": representation.id,
                "promotion_eligible": representation.promotion_eligible,
                "model_id": model_spec.id,
                "model_role": _model_evidence_role(representation, model_spec),
                "target_transform": model_spec.target_transform,
                "selection_view_id": target_view.id,
                "reader_experiment_id": group,
                "held_out_candidate_count": int(len(indexes)),
                "held_out_candidate_ids": "|".join(candidate_ids[indexes].astype(str).tolist()),
            }
            for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling"):
                record[f"{component}_spearman"] = _spearman(
                    np.asarray(getattr(true_components, component), dtype=float),
                    np.asarray(getattr(predicted_components, component), dtype=float),
                )
            record["feasibility_spearman"] = _spearman(
                _feasibility(true_components, scales),
                _feasibility(predicted_components, scales),
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _screen_summary(
    representation: LabelRepresentation,
    model_spec: ModelScreenSpec,
    *,
    predicted_response_magnitude: np.ndarray,
    target_views: Sequence[StressTargetView],
    group_metrics: pd.DataFrame,
) -> dict[str, object]:
    truth = representation.response_magnitude_truth
    channel_correlations = [_spearman(truth[:, index], predicted_response_magnitude[:, index]) for index in range(8)]
    record: dict[str, object] = {
        "representation_id": representation.id,
        "promotion_eligible": representation.promotion_eligible,
        "model_id": model_spec.id,
        "model_role": _model_evidence_role(representation, model_spec),
        "target_transform": model_spec.target_transform,
        "validation": "leave_one_reader_experiment_out",
        "metric_scope": "median_within_held_out_experiment",
        "hyperparameter_posture": "fixed_screen_not_promoted",
        "median_channel_spearman": float(np.nanmedian(channel_correlations)),
        "minimum_channel_spearman": float(np.nanmin(channel_correlations)),
        "response_magnitude_mae": float(np.mean(np.abs(truth - predicted_response_magnitude))),
    }
    response_correlations: list[float] = []
    feasibility_correlations: list[float] = []
    defined_group_counts: list[int] = []
    for target_view in target_views:
        view_rows = group_metrics.loc[group_metrics["selection_view_id"].astype(str).eq(target_view.id)]
        response_separation = _finite_median(view_rows["response_separation_spearman"])
        feasibility = _finite_median(view_rows["feasibility_spearman"])
        defined_count = int(
            np.isfinite(view_rows[["response_separation_spearman", "feasibility_spearman"]].to_numpy(dtype=float))
            .all(axis=1)
            .sum()
        )
        record[f"{target_view.id}__response_separation_spearman"] = response_separation
        record[f"{target_view.id}__feasibility_spearman"] = feasibility
        record[f"{target_view.id}__defined_group_count"] = defined_count
        response_correlations.append(response_separation)
        feasibility_correlations.append(feasibility)
        defined_group_counts.append(defined_count)
    record["weakest_target_view_response_separation_spearman"] = _minimum_if_all_finite(response_correlations)
    record["median_target_view_response_separation_spearman"] = _median_if_all_finite(response_correlations)
    record["weakest_target_view_feasibility_spearman"] = _minimum_if_all_finite(feasibility_correlations)
    record["median_target_view_feasibility_spearman"] = _median_if_all_finite(feasibility_correlations)
    record["weakest_required_ordering_spearman"] = _minimum_if_all_finite(
        [
            float(record["weakest_target_view_response_separation_spearman"]),
            float(record["weakest_target_view_feasibility_spearman"]),
        ]
    )
    record["minimum_defined_group_count"] = int(min(defined_group_counts))
    record["all_target_view_metrics_finite"] = bool(
        np.isfinite(response_correlations).all() and np.isfinite(feasibility_correlations).all()
    )
    return record


def _retrospective_enrichment(
    representation: LabelRepresentation,
    model_spec: ModelScreenSpec,
    *,
    predicted_response_magnitude: np.ndarray,
    groups: np.ndarray,
    candidate_ids: np.ndarray,
    target_views: Sequence[StressTargetView],
    uncertainty_rows: pd.DataFrame,
    scale_quantile: float,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_view in target_views:
        for group in sorted(np.unique(groups.astype(str))):
            indexes = np.flatnonzero(groups.astype(str) == group)
            if len(indexes) < 2:
                continue
            fold_calibration = build_calibration_table(
                uncertainty_rows,
                scale_quantile=scale_quantile,
                bootstrap_samples=bootstrap_samples,
                exclude_experiment=group,
            )
            scales = _target_view_scales(fold_calibration, selection_view_id=target_view.id)
            truth = _feasibility(_components(representation.response_magnitude_truth[indexes], target_view), scales)
            predicted = _feasibility(_components(predicted_response_magnitude[indexes], target_view), scales)
            group_predictions = predicted
            top_mask = np.isclose(group_predictions, np.max(group_predictions), rtol=0.0, atol=1.0e-12)
            top_count = int(np.sum(top_mask))
            selection_defined = top_count == 1
            selected_local = int(np.argmax(group_predictions)) if selection_defined else None
            if selected_local is not None:
                selected_truth = truth[selected_local]
                lower_count = int(np.sum(truth < selected_truth))
                equal_count = int(np.sum(np.isclose(truth, selected_truth, rtol=0.0, atol=1.0e-12)))
                percentile = float((lower_count + 0.5 * equal_count) / len(truth))
            else:
                percentile = float("nan")
            rows.append(
                {
                    "representation_id": representation.id,
                    "promotion_eligible": representation.promotion_eligible,
                    "model_id": model_spec.id,
                    "model_role": _model_evidence_role(representation, model_spec),
                    "selection_view_id": target_view.id,
                    "reader_experiment_id": group,
                    "selected_candidate_id": (
                        str(candidate_ids[indexes[selected_local]]) if selected_local is not None else None
                    ),
                    "held_out_candidate_count": int(len(indexes)),
                    "top_prediction_tie_count": top_count,
                    "selection_defined": selection_defined,
                    "selected_true_feasibility": (
                        float(truth[selected_local]) if selected_local is not None else float("nan")
                    ),
                    "selected_true_percentile": percentile,
                    "percentile_definition": "within_group_midrank_random_expectation_0p5",
                    "beats_group_median": bool(percentile > 0.5) if selection_defined else float("nan"),
                }
            )
    return pd.DataFrame.from_records(rows)


def _model_evidence_role(representation: LabelRepresentation, model_spec: ModelScreenSpec) -> str:
    if model_spec.role != "campaign_model":
        return model_spec.role
    if representation.promotion_eligible and representation.decoder == "identity_response_magnitude":
        return "campaign_model"
    return "fixed_challenger"


def _components(values: np.ndarray, target_view: StressTargetView) -> ResponseMagnitudeFeasibilityComponents:
    return response_magnitude_feasibility_components(values, target_mask=target_view.target_mask)


def _feasibility(
    components: ResponseMagnitudeFeasibilityComponents,
    scales: Mapping[str, float],
) -> np.ndarray:
    return calibrate_response_magnitude_feasibility(
        components,
        calibration={
            "response_separation_min": 0.0,
            "on_magnitude_min": 0.0,
            "off_magnitude_max": 0.0,
            "response_separation_scale": float(scales["response_separation"]),
            "on_magnitude_scale": float(scales["on_magnitude_floor"]),
            "off_magnitude_scale": float(scales["off_magnitude_ceiling"]),
        },
    ).feasibility_margin


def _target_view_scales(calibration: pd.DataFrame, *, selection_view_id: str) -> dict[str, float]:
    rows = calibration.loc[calibration["selection_view_id"].astype(str).eq(str(selection_view_id))]
    scales = {str(row.component): float(row.scale) for row in rows.itertuples(index=False)}
    required = {"response_separation", "on_magnitude_floor", "off_magnitude_ceiling"}
    if set(scales) != required or any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError(f"selection view {selection_view_id!r} has invalid calibration scales: {scales}")
    return scales


def _finite_median(values: pd.Series) -> float:
    finite = values.loc[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    return float(np.median(finite)) if finite.size else float("nan")


def _minimum_if_all_finite(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.min(array)) if np.isfinite(array).all() else float("nan")


def _median_if_all_finite(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.median(array)) if np.isfinite(array).all() else float("nan")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


__all__ = [
    "screen_label_models",
]
