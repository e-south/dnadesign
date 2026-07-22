"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_prediction_surface_diagnostics.py

Derived diagnostics for the fixed MSRB prediction surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FAMILY_COLUMNS = (
    "response_family_score",
    "on_signal_family_score",
    "off_signal_suppression_family_score",
)
_PROVENANCE_COLUMNS = (
    "prediction_run_id",
    "prediction_source_sha256",
    "protocol_id",
    "protocol_source_sha256",
    "normalization_source_rows_sha256",
)


def build_prediction_surface_diagnostics(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize conformance prevalence and family-score geometry by view."""

    required = {
        "selection_view_id",
        "all_reference_directions_met",
        *_FAMILY_COLUMNS,
        *_PROVENANCE_COLUMNS,
    }
    missing = sorted(required - set(prediction_scores.columns))
    if missing:
        raise ValueError(f"prediction scores lack surface-diagnostic columns: {missing}.")
    if prediction_scores.empty:
        raise ValueError("prediction surface diagnostics require at least one prediction row.")
    values = prediction_scores.loc[:, _FAMILY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction family scores must be finite before surface diagnostics.")

    records: list[dict[str, object]] = []
    for view_id, rows in prediction_scores.groupby("selection_view_id", sort=True):
        family = rows.loc[:, _FAMILY_COLUMNS].to_numpy(dtype=float)
        pca_ratios, pca_defined = _standardized_pca_explained_variance(family)
        spearman = rows["on_signal_family_score"].corr(
            rows["off_signal_suppression_family_score"],
            method="spearman",
        )
        met_count = int(rows["all_reference_directions_met"].astype(bool).sum())
        record: dict[str, object] = {
            "selection_view_id": str(view_id),
            "predicted_row_count": len(rows),
            "all_reference_directions_met_count": met_count,
            "all_reference_directions_met_fraction": met_count / len(rows),
            "on_signal_off_suppression_spearman": float(spearman) if pd.notna(spearman) else np.nan,
            "on_signal_off_suppression_spearman_defined": bool(pd.notna(spearman)),
            "family_pca_explained_variance_ratio_pc1": pca_ratios[0],
            "family_pca_explained_variance_ratio_pc2": pca_ratios[1],
            "family_pca_explained_variance_ratio_pc3": pca_ratios[2],
            "family_pca_defined": pca_defined,
            "family_pca_standardization": "per_view_population_z_score_zero_variance_columns_set_to_zero",
            "source_table_id": "prediction_scores",
        }
        for column in _PROVENANCE_COLUMNS:
            record[column] = _single_value(rows, column=column, view_id=str(view_id))
        record["evidence_role"] = "prediction_surface_shape_diagnostic_no_selection_claim"
        records.append(record)
    return pd.DataFrame.from_records(records)


def _standardized_pca_explained_variance(values: np.ndarray) -> tuple[tuple[float, float, float], bool]:
    centered = values - values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    standardized = np.divide(centered, scales, out=np.zeros_like(centered), where=scales > 0.0)
    singular_values = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
    variances = singular_values**2
    total = float(variances.sum())
    if total <= 0.0:
        return (np.nan, np.nan, np.nan), False
    ratios = np.pad(variances / total, (0, max(0, 3 - len(variances))))[:3]
    return (float(ratios[0]), float(ratios[1]), float(ratios[2])), True


def _single_value(rows: pd.DataFrame, *, column: str, view_id: str) -> str:
    values = rows[column].astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"prediction surface view {view_id!r} has inconsistent {column!r} provenance.")
    return str(values[0])


__all__ = ["build_prediction_surface_diagnostics"]
