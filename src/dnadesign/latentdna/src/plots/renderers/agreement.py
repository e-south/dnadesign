"""Agreement and correspondence renderers for static plot artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...visual_style import PUBLICATION_PALETTE, SPINE_COLOR, TEXT_COLOR, humanize_display_text, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..axes import apply_axes_style
from ..layout import _panel_grid_dimensions
from ..tables import read_table_rows, require_row_columns


@dataclass(frozen=True, slots=True)
class AgreementRenderResult:
    """Rendered agreement/correspondence figure and metadata."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)


def shared_row_key_columns(left_rows: list[dict], right_rows: list[dict]) -> list[str]:
    """Resolve common row-key columns for two cluster-assignment tables."""

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


def _assignment_by_key(
    rows: list[dict[str, object]],
    *,
    key_columns: list[str],
    cluster_id: str,
) -> dict[tuple[object, ...], int]:
    require_row_columns(rows, [*key_columns, "cluster_label"], context=f"cluster assignments {cluster_id}")
    assignments: dict[tuple[object, ...], int] = {}
    for row_index, row in enumerate(rows):
        key = tuple(row[column] for column in key_columns)
        if key in assignments:
            raise ContractViolationError(
                f"correspondence_heatmap cluster {cluster_id!r} contains duplicate row key {key!r} "
                f"at row {row_index}; aggregate or deduplicate upstream before rendering"
            )
        try:
            assignments[key] = int(row["cluster_label"])
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(
                f"correspondence_heatmap cluster {cluster_id!r} has non-integer cluster_label "
                f"at row {row_index}: {row['cluster_label']!r}"
            ) from exc
    return assignments


def agreement_summary_metrics(summary: dict[str, object]) -> list[tuple[str, float]]:
    """Extract plottable scalar agreement metrics from an agreement summary."""

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


def render_agreement_summary_panel(
    axis: Any,
    *,
    metrics: list[tuple[str, float]],
    panel_title: str,
) -> None:
    """Render one agreement metric panel."""

    labels = [label for label, _ in metrics]
    values = [value for _, value in metrics]
    bars = axis.bar(
        labels,
        values,
        color=[PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)] for index in range(len(labels))],
    )
    low = min(0.0, min(values))
    high = max(1.0, max(values))
    if low == high:
        high = low + 1.0
    padding = max((high - low) * 0.15, 0.05)
    axis.set_ylim(low - padding, high + padding)
    axis.axhline(0.0, color=SPINE_COLOR, linewidth=0.8)
    axis.set_ylabel("Score")
    axis.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    apply_axes_style(axis, grid=True)
    for bar, value in zip(bars, values, strict=True):
        va = "bottom" if value >= 0 else "top"
        offset = 0.02 if value >= 0 else -0.02
        axis.text(bar.get_x() + (bar.get_width() / 2.0), value + offset, f"{value:.2f}", ha="center", va=va)


def render_correspondence_heatmap_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
) -> AgreementRenderResult:
    """Render overlap counts between two cluster-assignment artifacts."""

    assert spec.left_cluster_id is not None
    assert spec.right_cluster_id is not None
    left_path = context.output_root / "clusters" / spec.left_cluster_id / "assignments.parquet"
    right_path = context.output_root / "clusters" / spec.right_cluster_id / "assignments.parquet"
    if not left_path.exists():
        raise MissingArtifactError(f"cluster artifact is missing for correspondence rendering: {spec.left_cluster_id}")
    if not right_path.exists():
        raise MissingArtifactError(f"cluster artifact is missing for correspondence rendering: {spec.right_cluster_id}")
    left_rows = read_table_rows(left_path, required_columns=["cluster_label"], artifact_label=spec.left_cluster_id)
    right_rows = read_table_rows(right_path, required_columns=["cluster_label"], artifact_label=spec.right_cluster_id)
    key_columns = shared_row_key_columns(left_rows, right_rows)
    left_by_key = _assignment_by_key(left_rows, key_columns=key_columns, cluster_id=spec.left_cluster_id)
    right_by_key = _assignment_by_key(right_rows, key_columns=key_columns, cluster_id=spec.right_cluster_id)
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

    figure, axis = pyplot.subplots(figsize=(2 + 1.2 * len(right_labels), 1.8 + 1.1 * len(left_labels)))
    image = axis.imshow(grid, cmap="cividis", aspect="auto")
    axis.set_xticks(
        range(len(right_labels)),
        [humanize_display_text(str(label)) for label in right_labels],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(left_labels)), [humanize_display_text(str(label)) for label in left_labels])
    axis.set_xlabel(humanize_display_text(spec.right_cluster_id))
    axis.set_ylabel(humanize_display_text(spec.left_cluster_id))
    axis.set_title(wrap_plot_title(spec.plot_id, width=24), pad=8)
    for row_index in range(len(left_labels)):
        for column_index in range(len(right_labels)):
            axis.text(
                column_index,
                row_index,
                f"{int(grid[row_index, column_index])}",
                ha="center",
                va="center",
                color="white" if grid[row_index, column_index] > (grid.max() * 0.45) else TEXT_COLOR,
                fontsize=10,
            )
    colorbar = figure.colorbar(image, ax=axis, label="Overlap count")
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label("Overlap count", fontsize=11, color=TEXT_COLOR)
    apply_axes_style(axis, grid=False)
    return AgreementRenderResult(figure=figure)


def render_agreement_summary_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
) -> AgreementRenderResult:
    """Render one or more agreement-summary panels."""

    if spec.kind == "agreement_summary":
        assert spec.agreement_id is not None
        summary_path = context.output_root / "agreements" / spec.agreement_id / "summary.json"
        if not summary_path.exists():
            raise MissingArtifactError(f"agreement artifact is missing for plot rendering: {spec.agreement_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = agreement_summary_metrics(summary)
        if not metrics:
            raise ContractViolationError(
                f"agreement_summary rendering found no plottable metrics for {spec.agreement_id}"
            )
        figure, axis = pyplot.subplots(figsize=(2 + 1.6 * len(metrics), 4.5))
        render_agreement_summary_panel(axis, metrics=metrics, panel_title=spec.plot_id)
        return AgreementRenderResult(figure=figure)

    if spec.kind == "agreement_summary_grid":
        agreement_summaries: list[tuple[str, list[tuple[str, float]]]] = []
        for agreement_id in spec.agreement_ids:
            summary_path = context.output_root / "agreements" / agreement_id / "summary.json"
            if not summary_path.exists():
                raise MissingArtifactError(f"agreement artifact is missing for plot rendering: {agreement_id}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = agreement_summary_metrics(summary)
            if not metrics:
                raise ContractViolationError(
                    f"agreement_summary_grid rendering found no plottable metrics for {agreement_id}"
                )
            agreement_summaries.append((agreement_id, metrics))
        rows_count, columns = _panel_grid_dimensions(len(agreement_summaries))
        figure, axes = pyplot.subplots(rows_count, columns, figsize=(6 * columns, 4.5 * rows_count), squeeze=False)
        titles = spec.panel_titles or [agreement_id for agreement_id, _ in agreement_summaries]
        for axis in axes.ravel()[len(agreement_summaries) :]:
            axis.axis("off")
        for axis, (_, metrics), panel_title in zip(axes.ravel(), agreement_summaries, titles, strict=False):
            render_agreement_summary_panel(axis, metrics=metrics, panel_title=panel_title)
        return AgreementRenderResult(figure=figure)

    raise ContractViolationError(f"agreement renderer does not support plot kind: {spec.kind}")
