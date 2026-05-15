"""Representation scorecard scalar builders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ...contracts.errors import ContractViolationError
from ...io.json_io import read_json
from ...io.matrix_io import read_matrix
from ...io.parquet_io import read_table, write_table
from ...metrics.definitions import resolve_metric_definition, validate_metric_registry
from ...workspaces.loader import WorkspaceContext
from ..classification_metrics import binary_metrics, dual_joint_margin
from ..common import BuiltScalarArtifact, ScalarInputRef, _effective_rank, _optional_param, _require_param


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path]:
    artifact_dir = context.output_root / "views" / view_id
    matrix_path = artifact_dir / "matrix.npy"
    rows_path = artifact_dir / "rows.parquet"
    if not matrix_path.is_file():
        raise ContractViolationError(f"view matrix is missing for {view_id!r}: {matrix_path}")
    if not rows_path.is_file():
        raise ContractViolationError(f"view rows are missing for {view_id!r}: {rows_path}")
    return matrix_path, rows_path


def _scalar_table_path(context: WorkspaceContext, scalar_id: str) -> Path:
    path = context.output_root / "scalars" / scalar_id / "table.parquet"
    if not path.is_file():
        raise ContractViolationError(f"scalar table is missing for {scalar_id!r}: {path}")
    return path


def _neighbor_paths(context: WorkspaceContext, neighbor_id: str) -> tuple[Path, Path, Path]:
    artifact_dir = context.output_root / "neighbors" / neighbor_id
    rows_path = artifact_dir / "rows.parquet"
    indices_path = artifact_dir / "indices.npy"
    distances_path = artifact_dir / "distances.npy"
    for path in [rows_path, indices_path, distances_path]:
        if not path.is_file():
            raise ContractViolationError(f"neighbor artifact is missing for {neighbor_id!r}: {path}")
    return rows_path, indices_path, distances_path


def _agreement_summary_path(context: WorkspaceContext, agreement_id: str) -> Path:
    path = context.output_root / "agreements" / agreement_id / "summary.json"
    if not path.is_file():
        raise ContractViolationError(f"agreement summary is missing for {agreement_id!r}: {path}")
    return path


def _reducer_summary_path(context: WorkspaceContext, reducer_id: str) -> Path:
    path = context.output_root / "reducers" / reducer_id / "summary.json"
    if not path.is_file():
        raise ContractViolationError(f"reducer summary is missing for {reducer_id!r}: {path}")
    return path


def _neighbor_label_enrichment(rows_table: pa.Table, indices: np.ndarray, *, label_column: str) -> float:
    if label_column not in rows_table.column_names:
        raise ContractViolationError(f"neighbor rows are missing label column {label_column!r}")
    labels = [str(value) for value in rows_table[label_column].combine_chunks().to_pylist()]
    global_counts: dict[str, int] = {}
    for label in labels:
        global_counts[label] = global_counts.get(label, 0) + 1
    total = len(labels)
    if total == 0:
        return float("nan")
    enrichments: list[float] = []
    k = int(indices.shape[1])
    for row_index, label in enumerate(labels):
        same_hits = sum(1 for index in indices[row_index] if labels[int(index)] == label)
        neighbor_fraction = same_hits / max(k, 1)
        background_fraction = global_counts[label] / total
        enrichments.append(neighbor_fraction - background_fraction)
    return float(np.mean(np.asarray(enrichments, dtype=np.float64)))


def _candidate_descriptor(candidate_id: str) -> dict[str, object]:
    family = "unknown"
    if "intermediate_embedding" in candidate_id:
        family = "intermediate_embedding"
    elif "output_layer_mean" in candidate_id:
        family = "output_layer_mean"
    elif candidate_id.startswith("log_likelihood_per_token_"):
        family = "log_likelihood"
    model = "20b" if "_20b_" in candidate_id else "7b" if "_7b_" in candidate_id else None
    scope = (
        "full_context_1kb"
        if "full_context_1kb" in candidate_id
        else "merged_anchor_insert_seq_mean"
        if "anchor_60bp" in candidate_id
        else None
    )
    label_parts = [part for part in [model.upper() if model is not None else None, family, scope] if part]
    return {
        "candidate_family": family,
        "candidate_model": model,
        "candidate_scope": scope,
        "candidate_label": " ".join(str(part).replace("_", " ") for part in label_parts) or candidate_id,
    }


def _task_reference_neighbor_metrics(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    label_column: str,
    positive_values: set[str],
    reference_labels: set[str],
) -> tuple[list[float], list[float]] | None:
    normalized_reference_labels = {label.lower() for label in reference_labels}
    row_labels = [str(row.get("usr_label__primary") or "").strip().lower() for row in rows]
    label_values = [str(row.get(label_column) or "") for row in rows]
    reference_indices = {
        index for index, reference_label in enumerate(row_labels) if reference_label in normalized_reference_labels
    }
    positive_indices = [index for index, value in enumerate(label_values) if value in positive_values]
    if not reference_indices or not positive_indices:
        return None
    hit_values: list[float] = []
    ranks: list[float] = []
    neighbor_count = int(indices.shape[1]) if indices.ndim == 2 else 0
    for row_index in positive_indices:
        neighbor_indices = [int(value) for value in indices[row_index].tolist()]
        hits = [rank + 1 for rank, neighbor_index in enumerate(neighbor_indices) if neighbor_index in reference_indices]
        hit_values.append(1.0 if hits else 0.0)
        ranks.append(float(hits[0] if hits else neighbor_count + 1))
    return hit_values, ranks


def _reference_neighbor_metrics(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    label_column: str,
    ethanol_values: set[str],
    cipro_values: set[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ethanol_summary = _task_reference_neighbor_metrics(
        rows,
        indices,
        label_column=label_column,
        positive_values=ethanol_values,
        reference_labels={"spyp"},
    )
    cipro_summary = _task_reference_neighbor_metrics(
        rows,
        indices,
        label_column=label_column,
        positive_values=cipro_values,
        reference_labels={"sulap"},
    )
    summaries = [summary for summary in [ethanol_summary, cipro_summary] if summary is not None]
    if summaries:
        hit_values = [value for summary in summaries for value in summary[0]]
        rank_values = [value for summary in summaries for value in summary[1]]
        metrics["reference_in_knn_rate"] = float(np.mean(np.asarray(hit_values, dtype=np.float64)))
        metrics["reference_neighbor_topk_censored_rank_median"] = float(
            np.median(np.asarray(rank_values, dtype=np.float64))
        )
    return metrics


def _candidate_metric_rows_by_candidate(source_table: pa.Table) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in source_table.to_pylist():
        candidate_id = str(row["candidate_id"])
        metric_id = str(row["metric_id"])
        if metric_id in grouped[candidate_id]:
            raise ContractViolationError(
                f"candidate metric source is non-unique on candidate_id={candidate_id!r}, metric_id={metric_id!r}"
            )
        grouped[candidate_id][metric_id] = row
    return grouped


def _candidate_metric_pairs_table(
    context: WorkspaceContext,
    *,
    source_scalar: str,
    x_metric_id: str,
    y_metric_id: str,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    source_path = _scalar_table_path(context, source_scalar)
    grouped = _candidate_metric_rows_by_candidate(read_table(source_path))
    rows: list[dict[str, object]] = []
    for candidate_id, metrics in sorted(grouped.items()):
        if x_metric_id not in metrics or y_metric_id not in metrics:
            continue
        descriptor = _candidate_descriptor(candidate_id)
        x_row = metrics[x_metric_id]
        y_row = metrics[y_metric_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                **descriptor,
                "x_metric_id": x_metric_id,
                "y_metric_id": y_metric_id,
                "x_display_name": x_row["display_name"],
                "y_display_name": y_row["display_name"],
                "x_metric_value": float(x_row["metric_value"]),
                "y_metric_value": float(y_row["metric_value"]),
                "x_direction": x_row["direction"],
                "y_direction": y_row["direction"],
                "x_unit": x_row["unit"],
                "y_unit": y_row["unit"],
            }
        )
    return (
        pa.Table.from_pylist(rows),
        [ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=source_path)],
        {"rows": len(rows), "source_scalar": source_scalar, "x_metric_id": x_metric_id, "y_metric_id": y_metric_id},
    )


def _candidate_metric_bars_table(
    context: WorkspaceContext,
    *,
    source_scalar: str,
    metric_ids: list[str],
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    source_path = _scalar_table_path(context, source_scalar)
    grouped = _candidate_metric_rows_by_candidate(read_table(source_path))
    rows: list[dict[str, object]] = []
    for candidate_id, metrics in sorted(grouped.items()):
        descriptor = _candidate_descriptor(candidate_id)
        for metric_id in metric_ids:
            metric_row = metrics.get(metric_id)
            if metric_row is None:
                continue
            rows.append(
                {
                    "category": metric_id,
                    "label": candidate_id,
                    "panel_id": metric_row["display_name"],
                    "metric_value": float(metric_row["metric_value"]),
                    "display_name": metric_row["display_name"],
                    "direction": metric_row["direction"],
                    "unit": metric_row["unit"],
                    **descriptor,
                }
            )
    return (
        pa.Table.from_pylist(rows),
        [ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=source_path)],
        {"rows": len(rows), "source_scalar": source_scalar, "metric_ids": metric_ids},
    )


def _representation_scorecard_table(
    context: WorkspaceContext,
    *,
    candidates: list[dict[str, Any]],
    label_column: str,
    ethanol_values: set[str],
    cipro_values: set[str],
    dual_values: set[str],
    neighbor_label_enrichments: list[dict[str, str]] | None = None,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    config = getattr(context, "config", None)
    validate_metric_registry(config)
    output_rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    seen_inputs: set[tuple[str, str, str]] = set()

    def add_input(kind: str, artifact_id: str, path: Path) -> None:
        key = (kind, artifact_id, path.as_posix())
        if key in seen_inputs:
            return
        seen_inputs.add(key)
        inputs.append(ScalarInputRef(kind=kind, artifact_id=artifact_id, path=path))

    for candidate in candidates:
        candidate_id = str(_require_param(candidate, "candidate_id"))
        candidate_metrics: dict[str, float] = {}

        wildtype_source = _optional_param(candidate, "wildtype_source", default=None)
        if wildtype_source is not None:
            path = _scalar_table_path(context, str(wildtype_source))
            add_input("scalar_table", str(wildtype_source), path)
            rows = read_table(path).to_pylist()
            ethanol_scores = np.asarray(
                [float(row["wildtype_margin_ethanol_vs_control"]) for row in rows], dtype=np.float64
            )
            cipro_scores = np.asarray(
                [float(row["wildtype_margin_cipro_vs_control"]) for row in rows], dtype=np.float64
            )
            candidate_metrics["wildtype_margin_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auroc"]
            candidate_metrics["wildtype_margin_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auprc"]
            candidate_metrics["wildtype_margin_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auroc"]
            candidate_metrics["wildtype_margin_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auprc"]
            if dual_values:
                dual_scores = dual_joint_margin(ethanol_scores, cipro_scores)
                candidate_metrics["wildtype_margin_dual_joint_auroc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auroc"]
                candidate_metrics["wildtype_margin_dual_joint_auprc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auprc"]

        synthetic_source = _optional_param(candidate, "synthetic_source", default=None)
        if synthetic_source is not None:
            path = _scalar_table_path(context, str(synthetic_source))
            add_input("scalar_table", str(synthetic_source), path)
            rows = read_table(path).to_pylist()
            ethanol_scores = np.asarray(
                [float(row["synthetic_margin_ethanol_vs_background"]) for row in rows], dtype=np.float64
            )
            cipro_scores = np.asarray(
                [float(row["synthetic_margin_cipro_vs_background"]) for row in rows], dtype=np.float64
            )
            candidate_metrics["synthetic_margin_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auroc"]
            candidate_metrics["synthetic_margin_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auprc"]
            candidate_metrics["synthetic_margin_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auroc"]
            candidate_metrics["synthetic_margin_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auprc"]
            if dual_values:
                dual_scores = dual_joint_margin(ethanol_scores, cipro_scores)
                candidate_metrics["synthetic_margin_dual_joint_auroc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auroc"]
                candidate_metrics["synthetic_margin_dual_joint_auprc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auprc"]

        scalar_view_id = _optional_param(candidate, "value_view_id", default=None)
        scalar_column = _optional_param(candidate, "value_column", default=None)
        if scalar_view_id is not None and scalar_column is not None:
            _, rows_path = _view_paths(context, str(scalar_view_id))
            add_input("view_rows", str(scalar_view_id), rows_path)
            table = read_table(rows_path)
            if scalar_column not in table.column_names:
                raise ContractViolationError(f"view {scalar_view_id} is missing score column {scalar_column!r}")
            rows = table.to_pylist()
            scores = np.asarray([float(row[scalar_column]) for row in rows], dtype=np.float64)
            candidate_metrics["scalar_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=scores,
            )["auroc"]
            candidate_metrics["scalar_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=scores,
            )["auprc"]
            candidate_metrics["scalar_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=scores,
            )["auroc"]
            candidate_metrics["scalar_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=scores,
            )["auprc"]

        neighbors_id = _optional_param(candidate, "neighbors_id", default=None)
        if neighbors_id is not None:
            rows_path, indices_path, _ = _neighbor_paths(context, str(neighbors_id))
            add_input("neighbor_rows", str(neighbors_id), rows_path)
            add_input("neighbor_set", str(neighbors_id), indices_path)
            rows_table = read_table(rows_path)
            indices = np.asarray(read_matrix(indices_path, mmap_mode=None), dtype=np.int64)
            if indices.shape[0] != rows_table.num_rows:
                raise ContractViolationError(
                    f"neighbor artifact {neighbors_id!r} row count does not match its neighbor index matrix"
                )
            enrichment_specs = neighbor_label_enrichments or [
                {
                    "label_column": "design_family",
                    "metric_name": "knn_design_family_enrichment_delta",
                }
            ]
            for enrichment_spec in enrichment_specs:
                enrichment_label_column = str(_require_param(enrichment_spec, "label_column"))
                enrichment_metric_name = str(_require_param(enrichment_spec, "metric_name", "metric_id"))
                if enrichment_label_column not in rows_table.column_names:
                    raise ContractViolationError(
                        f"neighbor rows are missing configured label enrichment column {enrichment_label_column!r}"
                    )
                candidate_metrics[enrichment_metric_name] = _neighbor_label_enrichment(
                    rows_table,
                    indices,
                    label_column=enrichment_label_column,
                )
            candidate_metrics.update(
                _reference_neighbor_metrics(
                    rows_table.to_pylist(),
                    indices,
                    label_column=label_column,
                    ethanol_values=ethanol_values,
                    cipro_values=cipro_values,
                )
            )

        context_source = _optional_param(candidate, "context_source", default=None)
        if context_source is not None:
            path = _scalar_table_path(context, str(context_source))
            add_input("scalar_table", str(context_source), path)
            table = read_table(path)
            if table.num_rows == 0:
                raise ContractViolationError(f"context source {context_source!r} is empty")
            context_rows = table.to_pylist()
            candidate_metrics["context_self_cosine_median"] = float(
                np.median(np.asarray(table["context_self_cosine"].to_pylist(), dtype=np.float64))
            )
            candidate_metrics["context_shift_l2_median"] = float(
                np.median(np.asarray(table["context_shift_l2"].to_pylist(), dtype=np.float64))
            )
            if "geometry_distance_correlation" in table.column_names:
                candidate_metrics["geometry_distance_correlation"] = float(
                    context_rows[0]["geometry_distance_correlation"]
                )

        agreement_id = _optional_param(candidate, "agreement_id", default=None)
        if agreement_id is not None:
            path = _agreement_summary_path(context, str(agreement_id))
            add_input("agreement_set", str(agreement_id), path)
            summary = read_json(path)
            knn_summary = summary.get("knn_overlap")
            if isinstance(knn_summary, dict):
                overlap_value = knn_summary.get(
                    "mean_neighbor_overlap_fraction",
                    knn_summary.get("mean_overlap_fraction"),
                )
                if isinstance(overlap_value, int | float):
                    candidate_metrics["neighbor_overlap_fraction"] = float(overlap_value)
            landmark_summary = summary.get("landmark_neighbor_overlap")
            if isinstance(landmark_summary, dict) and "mean_jaccard_overlap" in landmark_summary:
                candidate_metrics["landmark_neighbor_jaccard"] = float(landmark_summary["mean_jaccard_overlap"])

        reducer_id = _optional_param(candidate, "reducer_id", default=None)
        if reducer_id is not None:
            path = _reducer_summary_path(context, str(reducer_id))
            add_input("reducer", str(reducer_id), path)
            summary = read_json(path)
            explained_ratio = [float(value) for value in summary.get("explained_variance_ratio", [])]
            candidate_metrics["selected_rank"] = float(summary.get("output_dims", 0))
            candidate_metrics["explained_variance_captured"] = float(sum(explained_ratio))
            candidate_metrics["effective_rank"] = _effective_rank(explained_ratio)

        for metric_name, metric_value in sorted(candidate_metrics.items()):
            definition = resolve_metric_definition(metric_name, config=config)
            output_rows.append(
                {
                    "candidate_id": candidate_id,
                    "view_id": candidate_id,
                    "metric_name": metric_name,
                    "metric_id": definition.metric_id,
                    "metric_value": metric_value,
                    "value": metric_value,
                    "metric_family": definition.metric_family,
                    "evidence_tier": definition.evidence_tier,
                    "task_id": definition.task_id,
                    "mathematical_definition": definition.mathematical_definition,
                    "unit": definition.unit,
                    "direction": definition.direction,
                    "aggregation_level": definition.aggregation_level,
                    "higher_is_better": definition.higher_is_better,
                    "display_name": definition.display_name,
                    "definition_version": definition.definition_version,
                }
            )

    return pa.Table.from_pylist(output_rows), inputs, {"candidate_count": len(candidates), "metrics": len(output_rows)}


def build_representation_scorecard_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, Any],
) -> BuiltScalarArtifact:
    table, inputs, stats = _representation_scorecard_table(
        context,
        candidates=[dict(value) for value in _require_param(params, "candidates")],
        label_column=str(_optional_param(params, "label_column", default="design_family")),
        ethanol_values={
            str(value)
            for value in _optional_param(
                params,
                "ethanol_values",
                default=["ethanol", "ethanol_ciprofloxacin"],
            )
        },
        cipro_values={
            str(value)
            for value in _optional_param(
                params,
                "cipro_values",
                default=["ciprofloxacin", "ethanol_ciprofloxacin"],
            )
        },
        dual_values={str(value) for value in _optional_param(params, "dual_values", default=["ethanol_ciprofloxacin"])},
        neighbor_label_enrichments=[
            {str(key): str(value) for key, value in dict(item).items()}
            for item in _optional_param(params, "neighbor_label_enrichments", default=[])
        ],
    )
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )


def build_candidate_metric_pairs_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, Any],
) -> BuiltScalarArtifact:
    table, inputs, stats = _candidate_metric_pairs_table(
        context,
        source_scalar=str(_require_param(params, "source_scalar")),
        x_metric_id=str(_require_param(params, "x_metric_id")),
        y_metric_id=str(_require_param(params, "y_metric_id")),
    )
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )


def build_candidate_metric_bars_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, Any],
) -> BuiltScalarArtifact:
    table, inputs, stats = _candidate_metric_bars_table(
        context,
        source_scalar=str(_require_param(params, "source_scalar")),
        metric_ids=[str(value) for value in _require_param(params, "metric_ids")],
    )
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
