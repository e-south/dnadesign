"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/export.py

Deterministic OPAL-ready matrix export for Evo2 promoter feature bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..errors import CapabilityError
from .contracts import OpalMatrixExport, PromoterFeatureBundleConfig
from .execution import (
    _LOG_LIKELIHOOD_MEAN,
    _LOG_LIKELIHOOD_TOTAL,
    _OUTPUT_LAYER_ANCHOR_MEAN,
    _OUTPUT_LAYER_SEQ_MEAN,
)
from .selectors import resolve_intermediate_selector


def export_opal_matrix(
    *,
    row_ids: Sequence[str],
    columnar: Mapping[str, Sequence[object]],
    bundle: PromoterFeatureBundleConfig,
    model_id: str,
) -> OpalMatrixExport:
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    ordered_columns: list[tuple[str, str]] = []

    if bundle.collect_log_likelihood:
        ordered_columns.extend(
            [
                (_LOG_LIKELIHOOD_TOTAL, f"infer.evo2.{model_id}.{bundle.context.kind}.log_likelihood.total"),
                (
                    _LOG_LIKELIHOOD_MEAN,
                    f"infer.evo2.{model_id}.{bundle.context.kind}.log_likelihood.mean_per_token",
                ),
            ]
        )

    if bundle.collect_output_layer_mean:
        ordered_columns.append(
            (
                _OUTPUT_LAYER_SEQ_MEAN,
                f"infer.evo2.{model_id}.{bundle.context.kind}.output_layer_mean.seq_mean",
            )
        )
        if bundle.context.kind != "anchor_only" and bundle.pooling.anchor_mean_for_templated:
            ordered_columns.append(
                (
                    _OUTPUT_LAYER_ANCHOR_MEAN,
                    f"infer.evo2.{model_id}.{bundle.context.kind}.output_layer_mean.anchor_mean",
                )
            )

    if bundle.collect_intermediate_embedding:
        ordered_columns.append(
            (
                f"intermediate_embedding__{selector.intermediate_selector}__seq_mean",
                f"infer.evo2.{model_id}.{bundle.context.kind}.intermediate_embedding."
                f"{selector.intermediate_selector}.seq_mean",
            )
        )
        if bundle.context.kind != "anchor_only" and bundle.pooling.anchor_mean_for_templated:
            ordered_columns.append(
                (
                    f"intermediate_embedding__{selector.intermediate_selector}__anchor_mean",
                    f"infer.evo2.{model_id}.{bundle.context.kind}.intermediate_embedding."
                    f"{selector.intermediate_selector}.anchor_mean",
                )
            )

    total_rows = len(row_ids)
    feature_names: list[str] = []
    expanded_columns: list[tuple[str, bool]] = []
    for column_name, base_feature_name in ordered_columns:
        values = columnar.get(column_name)
        if values is None:
            raise CapabilityError(f"Missing feature column required for OPAL export: {column_name}")
        first_value = values[0] if len(values) > 0 else None
        if isinstance(first_value, list):
            width = len(first_value)
            for row_index, row_value in enumerate(values):
                if not isinstance(row_value, list):
                    raise CapabilityError(
                        f"OPAL export expected list values for vector column '{column_name}' at row {row_index}."
                    )
                if len(row_value) != width:
                    raise CapabilityError(
                        f"OPAL export requires stable vector width for '{column_name}'; "
                        f"row0={width} row{row_index}={len(row_value)}."
                    )
            feature_names.extend(f"{base_feature_name}[{index}]" for index in range(width))
            expanded_columns.append((column_name, True))
            continue
        feature_names.append(base_feature_name)
        expanded_columns.append((column_name, False))

    x_rows: list[list[float]] = []
    for row_index in range(total_rows):
        row: list[float] = []
        for column_name, is_vector in expanded_columns:
            values = columnar.get(column_name)
            if values is None:
                raise CapabilityError(f"Missing feature column required for OPAL export: {column_name}")
            value = values[row_index]
            if not is_vector and isinstance(value, (float, int)):
                row.append(float(value))
                continue
            if not isinstance(value, list):
                raise CapabilityError(
                    f"OPAL export expected float or list values for column '{column_name}', got {type(value).__name__}."
                )
            row.extend(float(item) for item in value)
        x_rows.append(row)

    return OpalMatrixExport(x=x_rows, feature_names=feature_names, row_ids=[str(value) for value in row_ids])
