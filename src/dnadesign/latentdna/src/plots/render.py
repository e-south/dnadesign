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

_PUBLICATION_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
_TEXT_COLOR = "#16202A"
_GRID_COLOR = "#D5DCE4"


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _category_color_map(row_groups: list[list[dict]], column: str | None) -> tuple[dict[str, str], list[str]]:
    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    if flattened and column not in flattened[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    categories = sorted({str(row[column]) for row in flattened})
    color_map = {name: _PUBLICATION_PALETTE[index % len(_PUBLICATION_PALETTE)] for index, name in enumerate(categories)}
    return color_map, categories


def _color_series(
    rows: list[dict],
    column: str | None,
    *,
    color_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    if column is None:
        return [_PUBLICATION_PALETTE[0]] * len(rows), []
    if rows and column not in rows[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    resolved_map = color_map or _category_color_map([rows], column)[0]
    categories = sorted(resolved_map)
    return [resolved_map[str(row[column])] for row in rows], categories


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


def _scatter_style(row_count: int) -> tuple[float, float]:
    if row_count <= 250:
        return 34.0, 0.92
    if row_count <= 1_000:
        return 22.0, 0.84
    if row_count <= 5_000:
        return 12.0, 0.72
    return 7.0, 0.58


def _apply_axes_style(ax: Any, *, grid: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5C6874")
    ax.spines["bottom"].set_color("#5C6874")
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("semibold")
    if grid:
        ax.grid(True, color=_GRID_COLOR, linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)


def _legend_handles(plt: Any, categories: list[str], color_map: dict[str, str]) -> list[Any]:
    return [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7,
            color=color_map[category],
            label=category,
        )
        for category in categories
    ]


def _selected_label_rows(rows: list[dict], *, label_column: str | None, label_values: list[str]) -> list[dict]:
    if label_column is None or not label_values:
        return []
    if rows and label_column not in rows[0]:
        raise ContractViolationError(f"plot label column is missing: {label_column!r}")
    selected = {str(value) for value in label_values}
    return [row for row in rows if str(row[label_column]) in selected]


def _draw_label_overlay(
    ax: Any,
    rows: list[dict],
    *,
    label_column: str | None,
    label_values: list[str],
    color_column: str | None,
    color_map: dict[str, str],
) -> None:
    selected = _selected_label_rows(rows, label_column=label_column, label_values=label_values)
    if not selected or label_column is None:
        return
    highlight_colors, _ = _color_series(selected, color_column, color_map=color_map if color_map else None)
    ax.scatter(
        [float(row["x"]) for row in selected],
        [float(row["y"]) for row in selected],
        c=highlight_colors,
        s=135,
        marker="*",
        edgecolors="#111111",
        linewidths=0.8,
        zorder=5,
    )
    for row in selected:
        ax.annotate(
            str(row[label_column]),
            (float(row["x"]), float(row["y"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
            color=_TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=6,
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


def _ordered_numeric_axes(
    table: pa.Table,
    *,
    x_column: str | None,
    y_column: str | None,
    value_column: str | None,
) -> tuple[str, str]:
    numeric_columns = _numeric_columns(table)
    if len(numeric_columns) < 2:
        raise ContractViolationError("scatter rendering requires at least two numeric columns")
    resolved_x = x_column or value_column or numeric_columns[0]
    if resolved_x not in numeric_columns:
        raise ContractViolationError(f"scatter x column is missing or non-numeric: {resolved_x!r}")
    resolved_y = y_column or _secondary_numeric_column(table, primary=resolved_x)
    if resolved_y not in numeric_columns:
        raise ContractViolationError(f"scatter y column is missing or non-numeric: {resolved_y!r}")
    return resolved_x, resolved_y


def _shared_row_key_columns(left_rows: list[dict], right_rows: list[dict]) -> list[str]:
    if not left_rows or not right_rows:
        raise ContractViolationError("correspondence_heatmap requires non-empty cluster assignments")
    left_columns = set(left_rows[0]) - {"cluster_label"}
    right_columns = set(right_rows[0]) - {"cluster_label"}
    preferred_order = ["id", "subject_id", "record_key", "subject_key", "context_id", "context_key"]
    shared = [column for column in preferred_order if column in left_columns and column in right_columns]
    if shared:
        return shared
    shared = sorted(left_columns.intersection(right_columns))
    if not shared:
        raise ContractViolationError("correspondence_heatmap requires at least one shared row key column")
    return shared


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
    if spec.kind == "xy_scatter" and spec.scalar_id is None and spec.distance_id is None:
        raise ContractViolationError("xy_scatter rendering requires a scalar or distance artifact")
    if spec.kind == "curve" and spec.reducer_id is None:
        raise ContractViolationError("curve rendering requires a reducer artifact")
    if spec.kind == "correspondence_heatmap" and (spec.left_cluster_id is None or spec.right_cluster_id is None):
        raise ContractViolationError("correspondence_heatmap rendering requires two cluster artifacts")
    if spec.kind == "agreement_summary" and spec.agreement_id is None:
        raise ContractViolationError("agreement_summary rendering requires an agreement artifact")

    plt = _pyplot()

    if spec.kind == "heatmap":
        from matplotlib import colors as mcolors

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
        landmark_index = {landmark_id: index for index, landmark_id in enumerate(landmark_ids)}
        cohort_index = {cohort_value: index for index, cohort_value in enumerate(cohort_values)}
        grid = np.zeros((len(landmark_ids), len(cohort_values)), dtype=np.float32)
        for row in rows:
            grid[
                landmark_index[str(row["landmark_id"])],
                cohort_index[str(row["cohort_value"])],
            ] = float(row[metric_column])

        max_abs = max(float(np.max(np.abs(grid))), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(cohort_values), 1.5 + 1.2 * len(landmark_ids)))
        image = ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="auto")
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
                    color="white" if abs(grid[row_index, column_index]) > max_abs * 0.45 else _TEXT_COLOR,
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, label=metric_column)
        _apply_axes_style(ax, grid=False)
    elif spec.kind in {"distance_scatter", "xy_scatter"}:
        _, _, table_path = _table_artifact_path(context, spec)
        table = pq.read_table(table_path)
        resolved_x, resolved_y = _ordered_numeric_axes(
            table,
            x_column=spec.x_column,
            y_column=spec.y_column,
            value_column=spec.value_column,
        )

        rows = _table_rows(table_path)
        fig, ax = plt.subplots(figsize=(6, 5))
        x_values = [float(row[resolved_x]) for row in rows]
        y_values = [float(row[resolved_y]) for row in rows]
        render_mode = spec.render_mode or "points"
        colors, categories = _color_series(rows, spec.color_column)
        if render_mode == "hexbin":
            ax.hexbin(x_values, y_values, gridsize=max(10, min(35, int(np.sqrt(len(rows))) * 2)), cmap="Blues")
        elif render_mode == "density_contour":
            bins = max(10, min(30, int(np.sqrt(len(rows))) * 2))
            histogram, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            ax.contour(x_centers, y_centers, histogram.T, levels=4, cmap="Blues")
            ax.scatter(x_values, y_values, c=colors, s=8.0, alpha=0.35, edgecolors="none")
        else:
            point_size, alpha = _scatter_style(len(rows))
            ax.scatter(
                x_values,
                y_values,
                c=colors,
                s=point_size,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.25,
            )
        ax.set_xlabel(resolved_x)
        ax.set_ylabel(resolved_y)
        ax.set_title(spec.plot_id)
        _apply_axes_style(ax, grid=True)
        if categories and render_mode == "points":
            color_map, ordered_categories = _category_color_map([rows], spec.color_column)
            ax.legend(
                handles=_legend_handles(plt, ordered_categories, color_map),
                title=spec.color_column,
                frameon=False,
            )
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
        fig, ax = plt.subplots(figsize=(6, 4.5))
        render_mode = spec.render_mode or "histogram"
        bin_count = max(5, min(30, int(np.sqrt(values.size)) + 1))
        if render_mode == "ecdf":
            if spec.color_column is None:
                ordered = np.sort(values)
                cumulative = np.arange(1, len(ordered) + 1, dtype=np.float32) / float(len(ordered))
                ax.step(ordered, cumulative, where="post", color=_PUBLICATION_PALETTE[0], linewidth=2.0)
            else:
                if rows and spec.color_column not in rows[0]:
                    raise ContractViolationError(f"distribution color column is missing: {spec.color_column!r}")
                categories = sorted({str(row[spec.color_column]) for row in rows})
                for index, category in enumerate(categories):
                    category_values = np.sort(
                        np.asarray(
                            [float(row[metric_column]) for row in rows if str(row[spec.color_column]) == category],
                            dtype=np.float32,
                        )
                    )
                    cumulative = np.arange(1, len(category_values) + 1, dtype=np.float32) / float(len(category_values))
                    ax.step(
                        category_values,
                        cumulative,
                        where="post",
                        label=category,
                        color=_PUBLICATION_PALETTE[index % len(_PUBLICATION_PALETTE)],
                        linewidth=2.0,
                    )
                ax.legend(title=spec.color_column, frameon=False)
            ax.set_ylabel("ECDF")
        elif render_mode == "violin_box":
            if spec.color_column is None:
                violin = ax.violinplot([values], showmeans=False, showmedians=False)
                for body in violin["bodies"]:
                    body.set_facecolor(_PUBLICATION_PALETTE[0])
                    body.set_alpha(0.5)
                ax.boxplot([values], widths=0.18)
                ax.set_xticks([1], [metric_column])
            else:
                if rows and spec.color_column not in rows[0]:
                    raise ContractViolationError(f"distribution color column is missing: {spec.color_column!r}")
                categories = sorted({str(row[spec.color_column]) for row in rows})
                grouped_values = [
                    np.asarray(
                        [float(row[metric_column]) for row in rows if str(row[spec.color_column]) == category],
                        dtype=np.float32,
                    )
                    for category in categories
                ]
                violin = ax.violinplot(grouped_values, showmeans=False, showmedians=False)
                for index, body in enumerate(violin["bodies"]):
                    body.set_facecolor(_PUBLICATION_PALETTE[index % len(_PUBLICATION_PALETTE)])
                    body.set_alpha(0.45)
                ax.boxplot(grouped_values, widths=0.18)
                ax.set_xticks(range(1, len(categories) + 1), categories, rotation=25, ha="right")
            ax.set_ylabel(metric_column)
        else:
            if spec.color_column is None:
                ax.hist(values, bins=bin_count, color=_PUBLICATION_PALETTE[0], edgecolor="white", alpha=0.9)
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
                        color=_PUBLICATION_PALETTE[index % len(_PUBLICATION_PALETTE)],
                        edgecolor="white",
                    )
                ax.legend(title=spec.color_column, frameon=False)
            ax.set_ylabel("Count")
        ax.set_xlabel(metric_column)
        ax.set_title(artifact_id)
        _apply_axes_style(ax, grid=True)
    elif spec.kind == "curve":
        reducer_path = context.output_root / "reducers" / spec.reducer_id / "summary.json"
        if not reducer_path.exists():
            raise MissingArtifactError(f"reducer artifact is missing for curve rendering: {spec.reducer_id}")
        summary = json.loads(reducer_path.read_text(encoding="utf-8"))
        ratios = summary.get("explained_variance_ratio")
        if not isinstance(ratios, list) or not ratios:
            raise ContractViolationError(f"curve rendering requires explained_variance_ratio for {spec.reducer_id}")
        explained = np.asarray([float(value) for value in ratios], dtype=np.float32)
        cumulative = np.cumsum(explained)
        components = np.arange(1, len(explained) + 1, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(components, explained, marker="o", linewidth=1.8, color=_PUBLICATION_PALETTE[0], label="Explained")
        ax.plot(
            components,
            cumulative,
            marker="s",
            linewidth=1.8,
            color=_PUBLICATION_PALETTE[2],
            label="Cumulative",
        )
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance ratio")
        ax.set_ylim(0.0, max(1.0, float(cumulative.max()) * 1.05))
        ax.set_title(spec.plot_id)
        ax.legend(frameon=False)
        _apply_axes_style(ax, grid=True)
    elif spec.kind == "correspondence_heatmap":
        left_path = context.output_root / "clusters" / spec.left_cluster_id / "assignments.parquet"
        right_path = context.output_root / "clusters" / spec.right_cluster_id / "assignments.parquet"
        if not left_path.exists():
            raise MissingArtifactError(
                f"cluster artifact is missing for correspondence rendering: {spec.left_cluster_id}"
            )
        if not right_path.exists():
            raise MissingArtifactError(
                f"cluster artifact is missing for correspondence rendering: {spec.right_cluster_id}"
            )
        left_rows = _table_rows(left_path)
        right_rows = _table_rows(right_path)
        key_columns = _shared_row_key_columns(left_rows, right_rows)
        left_by_key: dict[tuple[object, ...], int] = {}
        right_by_key: dict[tuple[object, ...], int] = {}
        for row in left_rows:
            key = tuple(row[column] for column in key_columns)
            left_by_key[key] = int(row["cluster_label"])
        for row in right_rows:
            key = tuple(row[column] for column in key_columns)
            right_by_key[key] = int(row["cluster_label"])
        shared_keys = sorted(set(left_by_key).intersection(right_by_key))
        if not shared_keys:
            raise ContractViolationError("correspondence_heatmap found no aligned support between cluster assignments")
        left_labels = sorted({left_by_key[key] for key in shared_keys})
        right_labels = sorted({right_by_key[key] for key in shared_keys})
        left_index = {label: index for index, label in enumerate(left_labels)}
        right_index = {label: index for index, label in enumerate(right_labels)}
        grid = np.zeros((len(left_labels), len(right_labels)), dtype=np.float32)
        for key in shared_keys:
            grid[left_index[left_by_key[key]], right_index[right_by_key[key]]] += 1.0
        fig, ax = plt.subplots(figsize=(2 + 1.2 * len(right_labels), 1.8 + 1.1 * len(left_labels)))
        image = ax.imshow(grid, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(right_labels)), [str(label) for label in right_labels], rotation=25, ha="right")
        ax.set_yticks(range(len(left_labels)), [str(label) for label in left_labels])
        ax.set_xlabel(spec.right_cluster_id)
        ax.set_ylabel(spec.left_cluster_id)
        ax.set_title(spec.plot_id)
        for row_index in range(len(left_labels)):
            for column_index in range(len(right_labels)):
                ax.text(
                    column_index,
                    row_index,
                    f"{int(grid[row_index, column_index])}",
                    ha="center",
                    va="center",
                    color="white" if grid[row_index, column_index] > (grid.max() * 0.45) else _TEXT_COLOR,
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, label="Overlap count")
        _apply_axes_style(ax, grid=False)
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
        bars = ax.bar(
            labels,
            values,
            color=[_PUBLICATION_PALETTE[index % len(_PUBLICATION_PALETTE)] for index in range(len(labels))],
        )
        low = min(0.0, min(values))
        high = max(1.0, max(values))
        if low == high:
            high = low + 1.0
        padding = max((high - low) * 0.15, 0.05)
        ax.set_ylim(low - padding, high + padding)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Score")
        ax.set_title(spec.plot_id)
        _apply_axes_style(ax, grid=True)
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
            color_map, categories = _category_color_map([rows], spec.color_column)
            colors, _ = _color_series(rows, spec.color_column, color_map=color_map if color_map else None)
            fig, ax = plt.subplots(figsize=(6, 5))
            point_size, alpha = _scatter_style(len(rows))
            ax.scatter(
                [float(row["x"]) for row in rows],
                [float(row["y"]) for row in rows],
                c=colors,
                s=point_size,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.2,
                rasterized=len(rows) > 5_000,
            )
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_title(spec.projection_ids[0])
            _apply_axes_style(ax, grid=False)
            _draw_label_overlay(
                ax,
                rows,
                label_column=spec.label_column,
                label_values=spec.label_values,
                color_column=spec.color_column,
                color_map=color_map,
            )
            if categories:
                ax.legend(handles=_legend_handles(plt, categories, color_map), title=spec.color_column, frameon=False)
        else:
            columns = min(2, max(1, len(projection_tables)))
            rows_count = int(np.ceil(len(projection_tables) / columns))
            fig, axes = plt.subplots(rows_count, columns, figsize=(6 * columns, 5 * rows_count), squeeze=False)
            color_map, categories = _category_color_map(projection_tables, spec.color_column)
            titles = spec.panel_titles or list(spec.projection_ids)
            for axis in axes.ravel()[len(projection_tables) :]:
                axis.axis("off")
            for axis, projection_rows, projection_id, panel_title in zip(
                axes.ravel(),
                projection_tables,
                spec.projection_ids,
                titles,
                strict=False,
            ):
                colors, _ = _color_series(
                    projection_rows,
                    spec.color_column,
                    color_map=color_map if color_map else None,
                )
                point_size, alpha = _scatter_style(len(projection_rows))
                axis.scatter(
                    [float(row["x"]) for row in projection_rows],
                    [float(row["y"]) for row in projection_rows],
                    c=colors,
                    s=point_size,
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=0.2,
                    rasterized=len(projection_rows) > 5_000,
                )
                axis.set_title(panel_title)
                axis.set_xlabel("UMAP-1")
                axis.set_ylabel("UMAP-2")
                _apply_axes_style(axis, grid=False)
                _draw_label_overlay(
                    axis,
                    projection_rows,
                    label_column=spec.label_column,
                    label_values=spec.label_values,
                    color_column=spec.color_column,
                    color_map=color_map,
                )
            if categories:
                fig.legend(
                    handles=_legend_handles(plt, categories, color_map),
                    title=spec.color_column,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.995),
                    ncol=min(len(categories), 4),
                    frameon=False,
                )
    if spec.kind == "projection_grid" and spec.color_column:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs = _write_plot_outputs(fig, output_dir, formats=context.config.defaults.plot_formats)
    finally:
        plt.close(fig)
    return output_dir, outputs
