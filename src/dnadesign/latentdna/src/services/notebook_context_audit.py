"""
Context-audit assembly for workspace notebook controls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyarrow import Array

from ..contracts.notebook import WorkspaceNotebookContextAudit
from ..io.json_io import read_json
from ..io.parquet_io import read_schema, read_table


def _context_geometry_agreement_ids(context) -> list[str]:
    plot = context.config.plots.get("context_geometry_summary")
    if plot is None:
        return []
    if getattr(plot, "kind", None) == "agreement_summary_grid":
        return [str(agreement_id) for agreement_id in getattr(plot, "agreements", [])]
    agreement_id = getattr(plot, "agreement", None)
    if getattr(plot, "kind", None) == "agreement_summary" and agreement_id is not None:
        return [str(agreement_id)]
    return []


def _context_geometry_scalar_ids(context) -> list[str]:
    plot = context.config.plots.get("context_shift_reference_plane")
    if plot is None:
        return []
    if getattr(plot, "kind", None) in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        return [str(scalar_id) for scalar_id in getattr(plot, "scalars", [])]
    scalar_id = getattr(plot, "scalar", None)
    if getattr(plot, "kind", None) == "xy_scatter" and scalar_id is not None:
        return [str(scalar_id)]
    return []


def _agreement_summary_metrics(summary: object) -> dict[str, float]:
    if not isinstance(summary, dict):
        return {}
    metrics: dict[str, float] = {}
    mean_neighbor_overlap = summary.get("mean_neighbor_overlap_fraction", summary.get("mean_overlap_fraction"))
    if isinstance(mean_neighbor_overlap, int | float):
        metrics["mean_knn_overlap"] = float(mean_neighbor_overlap)
    landmark_summary = summary.get("landmark_neighbor_overlap")
    if isinstance(landmark_summary, dict):
        mean_jaccard = landmark_summary.get("mean_jaccard_overlap")
        if isinstance(mean_jaccard, int | float):
            metrics["mean_jaccard_overlap"] = float(mean_jaccard)
    return metrics


def _numpy_values(column: Array) -> np.ndarray:
    return column.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def _context_metric_arrays(table_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    schema = read_schema(table_path)
    required_columns = {"context_shift_l2", "context_self_cosine"}
    missing = required_columns.difference(schema.names)
    if missing:
        raise ValueError(", ".join(sorted(missing)))
    table = read_table(table_path, columns=sorted(required_columns))
    shift = _numpy_values(table["context_shift_l2"].combine_chunks())
    self_cosine = _numpy_values(table["context_self_cosine"].combine_chunks())
    return shift, self_cosine, int(table.num_rows)


def build_workspace_notebook_context_audit(context) -> WorkspaceNotebookContextAudit:
    scalar_ids = _context_geometry_scalar_ids(context)
    agreement_ids = _context_geometry_agreement_ids(context)
    min_signal_median = 1e-8
    payload = WorkspaceNotebookContextAudit(
        artifact_id="context_geometry_audit",
        status="missing",
        decision="not_evaluated",
        rule={
            "strategy": "paired_population_geometry",
            "min_signal_median": min_signal_median,
            "description": (
                "If median context_shift_l2 is below 1e-8, treat full-context replacement as numerically null; "
                "otherwise keep the anchor-versus-full-context comparison active."
            ),
        },
    )
    shift_arrays: list[np.ndarray] = []
    self_cosine_arrays: list[np.ndarray] = []
    available_scalar_ids: list[str] = []
    total_rows = 0
    missing_scalar_ids: list[str] = []
    invalid_scalar_ids: list[str] = []
    for scalar_id in scalar_ids:
        table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
        if not table_path.is_file():
            missing_scalar_ids.append(scalar_id)
            continue
        try:
            shift, self_cosine, row_count = _context_metric_arrays(table_path)
        except ValueError:
            invalid_scalar_ids.append(scalar_id)
            continue
        available_scalar_ids.append(scalar_id)
        shift_arrays.append(shift)
        self_cosine_arrays.append(self_cosine)
        total_rows += row_count
    if not available_scalar_ids:
        return payload
    if invalid_scalar_ids:
        payload.status = "error"
        payload.error = f"context audit table is missing required columns for: {sorted(invalid_scalar_ids)}"
        return payload
    shift = np.concatenate(shift_arrays)
    self_cosine = np.concatenate(self_cosine_arrays)
    if shift.size == 0 or self_cosine.size == 0:
        payload.status = "error"
        payload.error = "context audit table is empty"
        return payload
    shift_median = float(np.median(shift))
    self_cosine_median = float(np.median(self_cosine))
    payload.status = "ok"
    payload.decision = "no_context_signal" if shift_median < min_signal_median else "structured_context_shift"
    payload.rows = total_rows
    payload.table_path = (Path("scalars") / available_scalar_ids[0] / "table.parquet").as_posix()
    metrics: dict[str, object] = {
        "context_shift_l2_median": shift_median,
        "context_shift_l2_p95": float(np.percentile(shift, 95.0)),
        "context_self_cosine_median": self_cosine_median,
        "context_self_cosine_p05": float(np.percentile(self_cosine, 5.0)),
        "configured_scalar_panel_count": len(scalar_ids),
        "scalar_panel_count": len(available_scalar_ids),
        "scalar_panel_ids": scalar_ids,
        "table_paths": [
            (Path("scalars") / scalar_id / "table.parquet").as_posix() for scalar_id in available_scalar_ids
        ],
    }
    if missing_scalar_ids:
        metrics["missing_scalar_table_ids"] = missing_scalar_ids
    if agreement_ids:
        agreement_metrics: dict[str, dict[str, float]] = {}
        missing_agreement_summary_ids: list[str] = []
        for agreement_id in agreement_ids:
            summary_path = context.output_root / "agreements" / agreement_id / "summary.json"
            if not summary_path.is_file():
                missing_agreement_summary_ids.append(agreement_id)
                continue
            summary_metrics = _agreement_summary_metrics(read_json(summary_path))
            if summary_metrics:
                agreement_metrics[agreement_id] = summary_metrics
        metrics["configured_agreement_panel_count"] = len(agreement_ids)
        metrics["agreement_panel_count"] = len(agreement_metrics)
        metrics["agreement_panel_ids"] = agreement_ids
        if missing_agreement_summary_ids:
            metrics["missing_agreement_summary_ids"] = missing_agreement_summary_ids
        if agreement_metrics:
            metrics["agreement_panel_metrics"] = agreement_metrics
            knn_values = [
                panel_metrics["mean_knn_overlap"]
                for panel_metrics in agreement_metrics.values()
                if "mean_knn_overlap" in panel_metrics
            ]
            if knn_values:
                metrics["mean_knn_overlap"] = float(np.mean(knn_values, dtype=np.float64))
                metrics["neighbor_overlap_fraction"] = metrics["mean_knn_overlap"]
            jaccard_values = [
                panel_metrics["mean_jaccard_overlap"]
                for panel_metrics in agreement_metrics.values()
                if "mean_jaccard_overlap" in panel_metrics
            ]
            if jaccard_values:
                metrics["mean_jaccard_overlap"] = float(np.mean(jaccard_values, dtype=np.float64))
    payload.metrics = metrics
    return payload


__all__ = ["build_workspace_notebook_context_audit"]
