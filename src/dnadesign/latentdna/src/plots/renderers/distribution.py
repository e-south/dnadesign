"""Distribution renderers for static and notebook plot surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...labels import humanize_candidate
from ...metadata_axes import AxisStyle, normalize_axis_categories
from ...stats.rank import kendall_tau_b, linear_r2, spearman_correlation
from ...visual_style import PUBLICATION_PALETTE, humanize_display_text, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..axes import (
    apply_axes_style,
    axis_category_label,
    resolved_axis_label,
    style_compact_category_tick_labels,
)
from ..layout import _grid_figure_size, _panel_grid_dimensions, _prefer_single_row_panel_layout
from ..legends import style_legend
from ..tables import numeric_table_columns, read_table_rows, require_row_columns, table_artifact_path
from .scatter import axis_categories, axis_category_value, axis_style


@dataclass(frozen=True, slots=True)
class DistributionRenderResult:
    """Rendered distribution figure and metadata emitted by the renderer."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)


def derived_panel_label(identifier: str) -> str:
    """Return a compact panel label for derived scalar artifact identifiers."""

    candidate_key = str(identifier or "")
    matched_prefix = False
    for prefix in (
        "context_delta_distribution_",
        "context_geometry_metrics_",
        "wildtype_reference_margins_",
        "synthetic_centroid_margins_",
        "tradeoff_",
        "pca_",
    ):
        if candidate_key.startswith(prefix):
            candidate_key = candidate_key[len(prefix) :]
            matched_prefix = True
            break
    if not matched_prefix:
        return ""
    candidate_key = candidate_key.replace("_anchor_to_full_context", "")
    return humanize_candidate(candidate_key)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if np.isfinite(coerced) else None


def _finite_statistic(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _linear_fit_coefficients(x_values: list[float], y_values: list[float]) -> tuple[float, float] | None:
    if len(x_values) < 3 or len(y_values) < 3:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _ordinal_swarm_categories(
    rows: list[dict[str, object]],
    category_values: list[str],
    *,
    color_column: str,
    axis_styles: dict[str, AxisStyle] | None,
) -> list[str]:
    ordered_by_rank: dict[str, float] = {}
    for row, category in zip(rows, category_values, strict=True):
        order_value = _coerce_float(row.get("ordinal_plot_order"))
        if order_value is None:
            continue
        previous = ordered_by_rank.get(category)
        if previous is None or order_value < previous:
            ordered_by_rank[category] = order_value
    if ordered_by_rank:
        return sorted(
            set(category_values),
            key=lambda category: (ordered_by_rank.get(category, float("inf")), category),
        )
    return axis_categories(category_values, column=color_column, axis_styles=axis_styles)


def _render_ordinal_swarm(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    metric_column: str,
    color_column: str,
    category_values: list[str],
    axis_styles: dict[str, AxisStyle] | None,
) -> int:
    categories = _ordinal_swarm_categories(
        rows,
        category_values,
        color_column=color_column,
        axis_styles=axis_styles,
    )
    category_to_position = {category: index + 1 for index, category in enumerate(categories)}
    category_to_color = {
        category: PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)] for index, category in enumerate(categories)
    }
    rng = np.random.default_rng(17)
    stat_x: list[float] = []
    stat_y: list[float] = []
    median_points: list[tuple[float, float]] = []
    category_sizes: list[int] = []
    order_to_positions: dict[float, list[float]] = {}
    max_points_per_category = 700
    for category in categories:
        category_rows = [
            row for row, row_category in zip(rows, category_values, strict=True) if row_category == category
        ]
        y_values = np.asarray(
            [value for row in category_rows if (value := _coerce_float(row.get(metric_column))) is not None],
            dtype=np.float32,
        )
        rank_values = [
            order_value
            if (order_value := _coerce_float(row.get("ordinal_plot_order"))) is not None
            else float(category_to_position[category])
            for row in category_rows
            if _coerce_float(row.get(metric_column)) is not None
        ]
        stat_x.extend(rank_values)
        stat_y.extend(float(value) for value in y_values)
        if y_values.size == 0:
            continue
        category_sizes.append(int(y_values.size))
        if y_values.size > max_points_per_category:
            selected = np.sort(
                rng.choice(np.arange(y_values.size, dtype=np.int64), size=max_points_per_category, replace=False)
            )
            plotted_values = y_values[selected]
        else:
            plotted_values = y_values
        x_center = float(category_to_position[category])
        order_value = _coerce_float(category_rows[0].get("ordinal_plot_order"))
        if order_value is not None:
            order_to_positions.setdefault(order_value, []).append(x_center)
        jitter = rng.uniform(-0.18, 0.18, size=plotted_values.size)
        point_size = 36.0 if plotted_values.size == 1 else 24.0
        ax.scatter(
            np.full(plotted_values.size, x_center, dtype=np.float32) + jitter,
            plotted_values,
            s=point_size,
            color=category_to_color[category],
            edgecolors="none",
            alpha=0.82,
            rasterized=plotted_values.size > 350,
            zorder=2,
        )
        if y_values.size >= 2:
            median = float(np.median(y_values))
            quartiles = np.percentile(y_values, [25, 75])
            ax.plot(
                [x_center - 0.26, x_center + 0.26],
                [median, median],
                color="#111111",
                linewidth=1.5,
                zorder=3,
                label="_ordinal_class_median_tick",
            )
            if not np.isclose(float(quartiles[0]), float(quartiles[1])):
                ax.plot(
                    [x_center, x_center],
                    quartiles,
                    color="#111111",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=3,
                    label="_ordinal_class_iqr",
                )
        median_points.append((x_center, float(np.median(y_values))))
    should_draw_median_connector = len(categories) <= 12 or any(size > 1 for size in category_sizes)
    if len(median_points) >= 2 and should_draw_median_connector:
        median_points = sorted(median_points)
        ax.plot(
            [point[0] for point in median_points],
            [point[1] for point in median_points],
            color="#111111",
            linewidth=1.15,
            alpha=0.42,
            zorder=2.6,
            label="_ordinal_class_median_connector",
        )
    fit = _linear_fit_coefficients(stat_x, stat_y)
    if fit is not None and order_to_positions:
        slope, intercept = fit
        ordered_ranks = sorted(order_to_positions)
        if len(ordered_ranks) >= 2:
            display_positions = [float(np.mean(order_to_positions[rank])) for rank in ordered_ranks]
            fitted_values = [slope * rank + intercept for rank in ordered_ranks]
            ax.plot(
                display_positions,
                fitted_values,
                color="#5B6777",
                linewidth=1.65,
                linestyle=(0, (4, 3)),
                alpha=0.86,
                zorder=1.8,
                label="_ordinal_linear_fit",
            )
    ax.set_xticks(
        range(1, len(categories) + 1),
        [
            axis_category_label(category, column=color_column, axis_styles=axis_styles, compact=True)
            for category in categories
        ],
        rotation=0 if len(categories) <= 9 else 60 if len(categories) > 14 else 40,
        ha="center" if len(categories) <= 9 else "right",
    )
    ax.set_xlim(0.35, len(categories) + 0.65)
    rho = _finite_statistic(spearman_correlation(stat_x, stat_y, min_pairs=3))
    tau_b = _finite_statistic(kendall_tau_b(stat_x, stat_y, min_pairs=3))
    r2 = _finite_statistic(linear_r2(stat_x, stat_y, min_pairs=3))
    if rho is not None:
        stat_label = f"Ordinal-order rho={rho:.2f}"
        if tau_b is not None:
            stat_label = f"{stat_label}\nKendall tau-b={tau_b:.2f}"
        if r2 is not None:
            stat_label = f"{stat_label}\nlinear R^2={r2:.2f}"
        ax.text(
            0.03,
            0.97,
            stat_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "#D4D7DC", "alpha": 0.86, "pad": 2.5},
        )
    return len(categories)


def _style_ordinal_swarm_tick_labels(axis: Any, *, category_count: int) -> None:
    if category_count <= 9:
        font_size = 9.2
    elif category_count <= 14:
        font_size = 8.4
    else:
        font_size = 7.2
    for label in axis.get_xticklabels():
        label.set_fontsize(font_size)
        label.set_rotation_mode("anchor")
        label.set_ha("right" if category_count > 9 else "center")
        label.set_va("top")


def _default_static_filter(spec: ResolvedPlotSpec) -> tuple[str, str] | None:
    for option in spec.filter_options:
        if option.include_all or not option.values:
            continue
        return option.column, option.values[0].value
    return None


def _filter_rows(rows: list[dict[str, object]], *, column: str, value: str) -> list[dict[str, object]]:
    filtered = [row for row in rows if str(row.get(column) or "") == value]
    if not filtered:
        raise ContractViolationError(f"plot filter {column!r}={value!r} matched no rows")
    return filtered


def render_distribution_panel(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    metric_column: str,
    color_column: str | None,
    render_mode: str,
    panel_title: str,
    square: bool = False,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    """Render one distribution panel from already validated row dictionaries."""

    style = axis_style(axis_styles, color_column)
    category_values: list[str] | None = None
    if color_column is not None:
        require_row_columns(rows, [color_column], context="distribution color encoding")
        if style is None:
            category_values = [axis_category_value(row, color_column, axis_styles=axis_styles) for row in rows]
        else:
            category_values = normalize_axis_categories(
                style,
                [row[color_column] for row in rows],
                rows=rows,
            )
    if (
        render_mode in {"violin_box", "ordinal_swarm"}
        and color_column is not None
        and style is not None
        and style.ordinal_subset
    ):
        allowed = {str(value) for value in style.ordinal_subset}
        assert category_values is not None
        filtered_pairs = [
            (row, category) for row, category in zip(rows, category_values, strict=True) if category in allowed
        ]
        rows = [row for row, _ in filtered_pairs]
        category_values = [category for _, category in filtered_pairs]
        if not rows:
            raise ContractViolationError("ordinal distribution requires at least one row in the configured subset")
    if render_mode == "violin_box" and len(rows) > 24_000:
        rng = np.random.default_rng(17)
        if color_column is None:
            selected = rng.choice(np.arange(len(rows), dtype=np.int64), size=24_000, replace=False)
            rows = [rows[int(index)] for index in np.sort(selected)]
        else:
            sampled_rows: list[dict[str, object]] = []
            sampled_categories: list[str] = []
            grouped_indices: dict[str, list[int]] = {}
            assert category_values is not None
            for index, category in enumerate(category_values):
                grouped_indices.setdefault(category, []).append(index)
            max_per_group = max(1, 24_000 // max(len(grouped_indices), 1))
            for indices in grouped_indices.values():
                if len(indices) <= max_per_group:
                    sampled_rows.extend(rows[index] for index in indices)
                    sampled_categories.extend(category_values[index] for index in indices)
                    continue
                selected = rng.choice(np.asarray(indices, dtype=np.int64), size=max_per_group, replace=False)
                sampled_rows.extend(rows[int(index)] for index in np.sort(selected))
                sampled_categories.extend(category_values[int(index)] for index in np.sort(selected))
            rows = sampled_rows
            category_values = sampled_categories
    values = np.asarray([float(row[metric_column]) for row in rows], dtype=np.float32)
    bin_count = max(5, min(30, int(np.sqrt(values.size)) + 1))
    ordinal_category_count = 0
    boxplot_kwargs = {
        "widths": 0.18,
        "boxprops": {"color": "#111111", "linewidth": 1.2},
        "whiskerprops": {"color": "#111111", "linewidth": 1.2},
        "capprops": {"color": "#111111", "linewidth": 1.2},
        "medianprops": {"color": "#111111", "linewidth": 1.35},
        "flierprops": {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#111111",
            "markeredgewidth": 1.25,
            "markersize": 7.0,
            "linestyle": "none",
        },
    }
    if render_mode == "ecdf":
        x_axis_fallback = metric_column
        if color_column is None:
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1, dtype=np.float32) / float(len(ordered))
            ax.step(ordered, cumulative, where="post", color=PUBLICATION_PALETTE[0], linewidth=2.0)
        else:
            assert category_values is not None
            categories = sorted(set(category_values))
            for index, category in enumerate(categories):
                category_metric_values = np.sort(
                    np.asarray(
                        [
                            float(row[metric_column])
                            for row, row_category in zip(rows, category_values, strict=True)
                            if row_category == category
                        ],
                        dtype=np.float32,
                    )
                )
                cumulative = np.arange(1, len(category_metric_values) + 1, dtype=np.float32) / float(
                    len(category_metric_values)
                )
                ax.step(
                    category_metric_values,
                    cumulative,
                    where="post",
                    label=humanize_display_text(category),
                    color=PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)],
                    linewidth=2.0,
                )
            legend = ax.legend(frameon=False)
            style_legend(legend)
        ax.set_ylabel("ECDF")
    elif render_mode == "violin_box":
        x_axis_fallback = color_column or metric_column
        if color_column is None:
            violin = ax.violinplot([values], showmeans=False, showmedians=False)
            for body in violin["bodies"]:
                body.set_facecolor(PUBLICATION_PALETTE[0])
                body.set_alpha(0.5)
            ax.boxplot([values], **boxplot_kwargs)
            ax.set_xticks([1], [humanize_display_text(metric_column)])
        else:
            assert category_values is not None
            categories = axis_categories(
                category_values,
                column=color_column,
                axis_styles=axis_styles,
            )
            grouped_values = [
                np.asarray(
                    [
                        float(row[metric_column])
                        for row, row_category in zip(rows, category_values, strict=True)
                        if row_category == category
                    ],
                    dtype=np.float32,
                )
                for category in categories
            ]
            violin = ax.violinplot(grouped_values, showmeans=False, showmedians=False)
            for index, body in enumerate(violin["bodies"]):
                body.set_facecolor(PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)])
                body.set_alpha(0.45)
            ax.boxplot(grouped_values, **boxplot_kwargs)
            ax.set_xticks(
                range(1, len(categories) + 1),
                [
                    axis_category_label(category, column=color_column, axis_styles=axis_styles, compact=True)
                    for category in categories
                ],
                rotation=0 if style is not None and style.compact_display_labels else 25,
                ha="center" if style is not None and style.compact_display_labels else "right",
            )
        ax.set_ylabel(
            resolved_axis_label(
                explicit_label=y_axis_label,
                fallback_label=humanize_display_text(metric_column),
                width=18,
            )
        )
    elif render_mode == "ordinal_swarm":
        x_axis_fallback = color_column or metric_column
        if color_column is None or category_values is None:
            raise ContractViolationError("ordinal_swarm distribution rendering requires color_column")
        ordinal_category_count = _render_ordinal_swarm(
            ax,
            rows=rows,
            metric_column=metric_column,
            color_column=color_column,
            category_values=category_values,
            axis_styles=axis_styles,
        )
        ax.set_ylabel(
            resolved_axis_label(
                explicit_label=y_axis_label,
                fallback_label=humanize_display_text(metric_column),
                width=18,
            )
        )
    else:
        x_axis_fallback = metric_column
        if color_column is None:
            ax.hist(values, bins=bin_count, color=PUBLICATION_PALETTE[0], edgecolor="white", alpha=0.9)
        else:
            assert category_values is not None
            categories = sorted(set(category_values))
            for index, category in enumerate(categories):
                category_metric_values = np.asarray(
                    [
                        float(row[metric_column])
                        for row, row_category in zip(rows, category_values, strict=True)
                        if row_category == category
                    ],
                    dtype=np.float32,
                )
                ax.hist(
                    category_metric_values,
                    bins=bin_count,
                    alpha=0.55,
                    label=humanize_display_text(category),
                    color=PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)],
                    edgecolor="white",
                )
            legend = ax.legend(frameon=False)
            style_legend(legend)
        ax.set_ylabel("Count")
    ax.set_xlabel(
        resolved_axis_label(
            explicit_label=x_axis_label,
            fallback_label=x_axis_fallback,
            width=20,
        )
    )
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    apply_axes_style(ax, grid=True, square=square)
    if render_mode == "violin_box" and style is not None and style.compact_display_labels:
        style_compact_category_tick_labels(ax, axis_name="x")
    if render_mode == "ordinal_swarm":
        _style_ordinal_swarm_tick_labels(ax, category_count=ordinal_category_count)


def _render_distribution(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> DistributionRenderResult:
    artifact_kind, artifact_id, table_path = table_artifact_path(context, spec)
    table = pq.read_table(table_path)
    numeric_columns = numeric_table_columns(table)
    if not numeric_columns:
        raise ContractViolationError(f"distribution rendering requires at least one numeric column in {artifact_kind}")
    metric_column = spec.value_column or numeric_columns[0]
    if metric_column not in numeric_columns:
        raise ContractViolationError(f"distribution value column is missing or non-numeric: {metric_column!r}")

    rows = read_table_rows(table_path)
    if not rows:
        raise ContractViolationError("distribution rendering requires at least one row")
    figure, axis = pyplot.subplots(figsize=(5.4, 4.8))
    render_distribution_panel(
        axis,
        rows=rows,
        metric_column=metric_column,
        color_column=spec.color_column,
        render_mode=spec.render_mode or "histogram",
        panel_title=artifact_id,
        square=False,
        x_axis_label=spec.x_axis_label,
        y_axis_label=spec.y_axis_label,
        axis_styles=axis_styles,
    )
    return DistributionRenderResult(figure=figure)


def _render_distribution_grid(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> DistributionRenderResult:
    scalar_tables: list[tuple[str, list[dict[str, object]], str]] = []
    configured_metric_columns = list(spec.metric_columns or [])
    default_filter = _default_static_filter(spec)
    for index, scalar_id in enumerate(spec.scalar_ids):
        table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {scalar_id}")
        table = pq.read_table(table_path)
        numeric_columns = numeric_table_columns(table)
        if not numeric_columns:
            raise ContractViolationError(
                f"distribution_grid rendering requires at least one numeric column in scalar {scalar_id}"
            )
        rows = read_table_rows(table_path)
        if not rows:
            raise ContractViolationError("distribution_grid rendering requires at least one row per panel")
        if default_filter is not None:
            rows = _filter_rows(rows, column=default_filter[0], value=default_filter[1])
        if configured_metric_columns:
            for metric_column in configured_metric_columns:
                if metric_column not in numeric_columns:
                    raise ContractViolationError(
                        f"distribution_grid value column is missing or non-numeric: {metric_column!r}"
                    )
                scalar_tables.append((scalar_id, rows, metric_column))
            continue
        metric_column = (
            spec.value_columns[index] if index < len(spec.value_columns) else spec.value_column or numeric_columns[0]
        )
        if metric_column not in numeric_columns:
            raise ContractViolationError(f"distribution_grid value column is missing or non-numeric: {metric_column!r}")
        scalar_tables.append((scalar_id, rows, metric_column))
    prefer_single_row = _prefer_single_row_panel_layout(
        spec.plot_id,
        len(scalar_tables),
        configured=spec.single_row_panels,
    )
    rows_count, columns = _panel_grid_dimensions(len(scalar_tables), prefer_single_row=prefer_single_row)
    square_distribution_panels = bool(spec.square_panels)
    figure, axes = pyplot.subplots(
        rows_count,
        columns,
        figsize=_grid_figure_size(
            len(scalar_tables),
            square_panels=square_distribution_panels,
            prefer_single_row=prefer_single_row,
        ),
        squeeze=False,
    )
    titles = spec.panel_titles or [
        (
            f"{derived_panel_label(scalar_id)} · {humanize_display_text(metric_column)}"
            if derived_panel_label(scalar_id)
            else humanize_display_text(metric_column)
        )
        for scalar_id, _, metric_column in scalar_tables
    ]
    for axis in axes.ravel()[len(scalar_tables) :]:
        axis.axis("off")
    for panel_index, (axis, (_, rows, metric_column), panel_title) in enumerate(
        zip(axes.ravel(), scalar_tables, titles, strict=False)
    ):
        render_distribution_panel(
            axis,
            rows=rows,
            metric_column=metric_column,
            color_column=spec.color_column,
            render_mode=spec.render_mode or "histogram",
            panel_title=panel_title,
            square=square_distribution_panels,
            x_axis_label=spec.x_axis_label,
            y_axis_label=spec.y_axis_label,
            axis_styles=axis_styles,
        )
        if spec.hide_repeated_y_axis and spec.render_mode != "ordinal_swarm" and panel_index % columns != 0:
            axis.set_ylabel("")
            axis.set_yticklabels([])
    metadata: dict[str, object] = {}
    if configured_metric_columns:
        metadata["metric_columns"] = configured_metric_columns
    return DistributionRenderResult(figure=figure, metadata=metadata)


def render_distribution_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> DistributionRenderResult:
    """Render distribution plot kinds with schema-first data validation."""

    if spec.kind == "distribution":
        return _render_distribution(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    if spec.kind == "distribution_grid":
        return _render_distribution_grid(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    raise ContractViolationError(f"distribution renderer does not support plot kind: {spec.kind}")
