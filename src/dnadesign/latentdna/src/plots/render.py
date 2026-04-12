"""
Artifact-driven plotting helpers for latentdna.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.plot import SUPPORTED_PLOT_KINDS, ResolvedPlotSpec
from ..workspaces.loader import WorkspaceContext

_PLOT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#9467bd"]


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _color_series(rows: list[dict], column: str | None) -> tuple[list[str], list[str]]:
    if column is None:
        return [_PLOT_PALETTE[0]] * len(rows), []
    if rows and column not in rows[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    categories = sorted({str(row[column]) for row in rows})
    color_map = {name: _PLOT_PALETTE[index % len(_PLOT_PALETTE)] for index, name in enumerate(categories)}
    return [color_map[str(row[column])] for row in rows], categories


def _table_rows(table_path: Path) -> list[dict]:
    return pq.read_table(table_path).to_pylist()


def _numeric_columns(table: pa.Table) -> list[str]:
    numeric: list[str] = []
    for field in table.schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric.append(field.name)
    return numeric


def _secondary_numeric_column(table: pa.Table, *, primary: str) -> str:
    for candidate in _numeric_columns(table):
        if candidate != primary:
            return candidate
    raise ContractViolationError(
        f"plot rendering requires at least two numeric columns when {primary!r} is used as the first axis"
    )


def _table_artifact_path(context: WorkspaceContext, spec: ResolvedPlotSpec) -> tuple[str, str, Path]:
    candidates = [
        (
            "scalar_table",
            spec.scalar_id,
            context.output_root / "scalars" / spec.scalar_id / "table.parquet" if spec.scalar_id is not None else None,
        ),
        (
            "distance_set",
            spec.distance_id,
            context.output_root / "distances" / spec.distance_id / "table.parquet"
            if spec.distance_id is not None
            else None,
        ),
        (
            "enrichment_set",
            spec.enrichment_id,
            context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
            if spec.enrichment_id is not None
            else None,
        ),
        (
            "agreement_set",
            spec.agreement_id,
            context.output_root / "agreements" / spec.agreement_id / "table.parquet"
            if spec.agreement_id is not None
            else None,
        ),
    ]
    selected = [(kind, artifact_id, path) for kind, artifact_id, path in candidates if artifact_id is not None]
    if len(selected) != 1:
        raise ContractViolationError(
            "plot rendering requires exactly one table-backed artifact input for this plot kind"
        )
    artifact_kind, artifact_id, artifact_path = selected[0]
    assert artifact_path is not None
    if not artifact_path.exists():
        raise MissingArtifactError(f"{artifact_kind} artifact is missing for plot rendering: {artifact_id}")
    return artifact_kind, str(artifact_id), artifact_path


def _agreement_summary_metrics(summary: dict[str, object]) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    knn_summary = summary.get("knn_overlap")
    if isinstance(knn_summary, dict) and "mean_overlap_fraction" in knn_summary:
        metrics.append(("kNN overlap", float(knn_summary["mean_overlap_fraction"])))
    cluster_summary = summary.get("cluster_agreement")
    if isinstance(cluster_summary, dict):
        if "adjusted_rand_index" in cluster_summary:
            metrics.append(("ARI", float(cluster_summary["adjusted_rand_index"])))
        if "normalized_mutual_information" in cluster_summary:
            metrics.append(("NMI", float(cluster_summary["normalized_mutual_information"])))
    landmark_summary = summary.get("landmark_neighbor_overlap")
    if isinstance(landmark_summary, dict) and "mean_jaccard_overlap" in landmark_summary:
        metrics.append(("Landmark Jaccard", float(landmark_summary["mean_jaccard_overlap"])))
    return metrics


def _write_plot_outputs(fig: Any, artifact_dir: Path, *, formats: list[str]) -> list[str]:
    outputs: list[str] = []
    for format_name in formats:
        if format_name not in {"svg", "png"}:
            raise ContractViolationError(f"unsupported plot output format: {format_name!r}")
        output_path = artifact_dir / f"plot.{format_name}"
        if format_name == "svg":
            fig.savefig(output_path)
        else:
            fig.savefig(output_path, dpi=150)
        outputs.append(output_path.as_posix())
    return outputs


def render_plot_artifact(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    output_dir: Path,
) -> tuple[Path, list[str]]:
    if spec.kind not in SUPPORTED_PLOT_KINDS:
        raise ContractViolationError(f"unsupported plot kind: {spec.kind}")
    if spec.kind in {"projection_scatter", "projection_grid"} and not spec.projection_ids:
        raise ContractViolationError("plot rendering requires at least one projection artifact")
    if spec.kind == "heatmap" and spec.enrichment_id is None:
        raise ContractViolationError("heatmap rendering requires an enrichment artifact")
    if spec.kind == "distance_scatter" and spec.distance_id is None:
        raise ContractViolationError("distance_scatter rendering requires a distance artifact")
    if spec.kind == "agreement_summary" and spec.agreement_id is None:
        raise ContractViolationError("agreement_summary rendering requires an agreement artifact")

    plt = _pyplot()

    if spec.kind == "heatmap":
        table_path = context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"enrichment artifact is missing for heatmap rendering: {spec.enrichment_id}")
        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("heatmap rendering requires at least one enrichment row")
        metric_column = spec.value_column or "enrichment_delta"
        if metric_column not in rows[0]:
            raise ContractViolationError(f"heatmap value column is missing from enrichment table: {metric_column!r}")
        cohort_values = sorted({str(row["cohort_value"]) for row in rows})
        landmark_ids = list(dict.fromkeys(str(row["landmark_id"]) for row in rows))
        grid = np.zeros((len(landmark_ids), len(cohort_values)), dtype=np.float32)
        for row in rows:
            grid[landmark_ids.index(str(row["landmark_id"])), cohort_values.index(str(row["cohort_value"]))] = float(
                row[metric_column]
            )

        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(cohort_values), 1.5 + 1.2 * len(landmark_ids)))
        image = ax.imshow(grid, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(cohort_values)), cohort_values, rotation=30, ha="right")
        ax.set_yticks(range(len(landmark_ids)), landmark_ids)
        ax.set_xlabel("Cohort")
        ax.set_ylabel("Landmark")
        ax.set_title(spec.plot_id)
        for row_index in range(len(landmark_ids)):
            for column_index in range(len(cohort_values)):
                ax.text(
                    column_index,
                    row_index,
                    f"{grid[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, label=metric_column)
    elif spec.kind == "distance_scatter":
        _, _, table_path = _table_artifact_path(context, spec)
        table = pq.read_table(table_path)
        numeric_columns = _numeric_columns(table)
        if len(numeric_columns) < 2:
            raise ContractViolationError("distance_scatter rendering requires at least two numeric distance columns")
        resolved_x = spec.x_column or spec.value_column or numeric_columns[0]
        if resolved_x not in numeric_columns:
            raise ContractViolationError(f"distance_scatter x column is missing or non-numeric: {resolved_x!r}")
        resolved_y = spec.y_column or _secondary_numeric_column(table, primary=resolved_x)
        if resolved_y not in numeric_columns:
            raise ContractViolationError(f"distance_scatter y column is missing or non-numeric: {resolved_y!r}")

        rows = _table_rows(table_path)
        colors, categories = _color_series(rows, spec.color_column)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            [float(row[resolved_x]) for row in rows],
            [float(row[resolved_y]) for row in rows],
            c=colors,
            s=30,
            alpha=0.9,
        )
        ax.set_xlabel(resolved_x)
        ax.set_ylabel(resolved_y)
        ax.set_title(spec.plot_id)
        if categories:
            handles = [
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    color=_PLOT_PALETTE[index % len(_PLOT_PALETTE)],
                    label=category,
                )
                for index, category in enumerate(categories)
            ]
            ax.legend(handles=handles, title=spec.color_column)
    elif spec.kind == "distribution":
        artifact_kind, artifact_id, table_path = _table_artifact_path(context, spec)
        table = pq.read_table(table_path)
        numeric_columns = _numeric_columns(table)
        if not numeric_columns:
            raise ContractViolationError(
                f"distribution rendering requires at least one numeric column in {artifact_kind}"
            )
        metric_column = spec.value_column or numeric_columns[0]
        if metric_column not in numeric_columns:
            raise ContractViolationError(f"distribution value column is missing or non-numeric: {metric_column!r}")

        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("distribution rendering requires at least one row")
        values = np.asarray([float(row[metric_column]) for row in rows], dtype=np.float32)
        bin_count = max(5, min(30, int(np.sqrt(values.size)) + 1))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if spec.color_column is None:
            ax.hist(values, bins=bin_count, color=_PLOT_PALETTE[0], edgecolor="white", alpha=0.9)
        else:
            if rows and spec.color_column not in rows[0]:
                raise ContractViolationError(f"distribution color column is missing: {spec.color_column!r}")
            categories = sorted({str(row[spec.color_column]) for row in rows})
            for index, category in enumerate(categories):
                category_values = np.asarray(
                    [float(row[metric_column]) for row in rows if str(row[spec.color_column]) == category],
                    dtype=np.float32,
                )
                ax.hist(
                    category_values,
                    bins=bin_count,
                    alpha=0.55,
                    label=category,
                    color=_PLOT_PALETTE[index % len(_PLOT_PALETTE)],
                    edgecolor="white",
                )
            ax.legend(title=spec.color_column)
        ax.set_xlabel(metric_column)
        ax.set_ylabel("Count")
        ax.set_title(artifact_id)
    elif spec.kind == "agreement_summary":
        summary_path = context.output_root / "agreements" / spec.agreement_id / "summary.json"
        if not summary_path.exists():
            raise MissingArtifactError(f"agreement artifact is missing for plot rendering: {spec.agreement_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = _agreement_summary_metrics(summary)
        if not metrics:
            raise ContractViolationError(
                f"agreement_summary rendering found no plottable metrics for {spec.agreement_id}"
            )
        labels = [label for label, _ in metrics]
        values = [value for _, value in metrics]
        fig, ax = plt.subplots(figsize=(2 + 1.6 * len(labels), 4.5))
        bars = ax.bar(labels, values, color=[_PLOT_PALETTE[index % len(_PLOT_PALETTE)] for index in range(len(labels))])
        low = min(0.0, min(values))
        high = max(1.0, max(values))
        if low == high:
            high = low + 1.0
        padding = max((high - low) * 0.15, 0.05)
        ax.set_ylim(low - padding, high + padding)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Score")
        ax.set_title(spec.plot_id)
        for bar, value in zip(bars, values, strict=True):
            va = "bottom" if value >= 0 else "top"
            offset = 0.02 if value >= 0 else -0.02
            ax.text(bar.get_x() + (bar.get_width() / 2.0), value + offset, f"{value:.2f}", ha="center", va=va)
    else:
        projection_tables = []
        for projection_id in spec.projection_ids:
            projection_path = context.output_root / "projections" / projection_id / "coords.parquet"
            if not projection_path.exists():
                raise MissingArtifactError(f"projection artifact is missing for plot rendering: {projection_id}")
            projection_tables.append(_table_rows(projection_path))
        if spec.kind == "projection_scatter":
            rows = projection_tables[0]
            colors, categories = _color_series(rows, spec.color_column)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter([row["x"] for row in rows], [row["y"] for row in rows], c=colors, s=30, alpha=0.9)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_title(spec.projection_ids[0])
            if categories:
                handles = [
                    plt.Line2D(
                        [],
                        [],
                        linestyle="",
                        marker="o",
                        color=_PLOT_PALETTE[index % len(_PLOT_PALETTE)],
                        label=category,
                    )
                    for index, category in enumerate(categories)
                ]
                ax.legend(handles=handles, title=spec.color_column)
        else:
            columns = min(2, max(1, len(projection_tables)))
            rows_count = int(np.ceil(len(projection_tables) / columns))
            fig, axes = plt.subplots(rows_count, columns, figsize=(6 * columns, 5 * rows_count), squeeze=False)
            for axis in axes.ravel()[len(projection_tables) :]:
                axis.axis("off")
            for axis, projection_rows, projection_id in zip(
                axes.ravel(),
                projection_tables,
                spec.projection_ids,
                strict=False,
            ):
                colors, _ = _color_series(projection_rows, spec.color_column)
                axis.scatter(
                    [row["x"] for row in projection_rows],
                    [row["y"] for row in projection_rows],
                    c=colors,
                    s=20,
                )
                axis.set_title(projection_id)
                axis.set_xlabel("UMAP-1")
                axis.set_ylabel("UMAP-2")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs = _write_plot_outputs(fig, output_dir, formats=context.config.defaults.plot_formats)
    finally:
        plt.close(fig)
    return output_dir, outputs
