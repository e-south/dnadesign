"""Live plot-review rendering helpers for hue-switchable notebook surfaces."""

from __future__ import annotations

import json
import math
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..contracts.plot import ResolvedPlotSpec, metric_panel_uses_square_axes
from ..plots.render import (
    _category_color_map as static_category_color_map,
)
from ..plots.render import (
    _derived_panel_label,
    _render_curve_panel,
    _render_distribution_panel,
    _render_metric_panel,
)
from ..plots.render import (
    _grid_figure_size as static_grid_figure_size,
)
from ..plots.render import (
    _panel_grid_dimensions as static_panel_grid_dimensions,
)
from ..visual_style import TEXT_COLOR, humanize_display_text, wrap_plot_title
from ..visual_style import scatter_style as shared_scatter_style
from .browser_runtime_projection import enrich_projection_frame, render_projection_grid
from .browser_runtime_support import category_color_map as notebook_category_color_map
from .browser_runtime_support import (
    classify_hue_series,
    display_hue_label,
    draw_reference_labels,
    load_table,
    render_matplotlib_figure,
    style_notebook_axes,
    style_notebook_legend,
)


def load_plot_review_frames(
    plot_spec: dict[str, object],
    *,
    joinable_tables: list[dict[str, object]],
    output_root: Path,
) -> list[pd.DataFrame]:
    kind = str(plot_spec.get("kind") or "")
    if kind == "projection_grid":
        frames: list[pd.DataFrame] = []
        for projection_id in plot_spec.get("projection_ids", []):
            frame = load_table(output_root / "projections" / str(projection_id) / "coords.parquet")
            if not frame.empty:
                frame = enrich_projection_frame(frame, joinable_tables, output_root=output_root)
            frames.append(frame)
        return frames
    if kind in {"xy_scatter_grid", "paired_xy_scatter_grid", "distribution_grid"}:
        return [
            load_table(output_root / "scalars" / str(scalar_id) / "table.parquet")
            for scalar_id in plot_spec.get("scalar_ids", [])
        ]
    if kind in {"metric_panel_grid", "categorical_count"}:
        scalar_id = str(plot_spec.get("scalar_id") or "")
        if not scalar_id:
            return []
        return [load_table(output_root / "scalars" / scalar_id / "table.parquet")]
    return []


def _panel_grid_dimensions(panel_count: int) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if panel_count == 4:
        return 2, 2
    if panel_count == 8:
        return 2, 4
    columns = min(3, panel_count)
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def _panel_figure_size(panel_count: int) -> tuple[float, float]:
    rows, columns = _panel_grid_dimensions(panel_count)
    return ((4.15 * columns) + 0.35, (4.35 * rows) + 0.2)


def _configured_hue_kinds(plot_spec: dict[str, object]) -> dict[str, str]:
    options = plot_spec.get("hue_options", [])
    return {
        str(option.get("column")): str(option.get("type"))
        for option in options
        if isinstance(option, dict) and option.get("column") and option.get("type")
    }


def _scatter_axis_label(frame: pd.DataFrame, *, value_column: str, label_column: str) -> str:
    if label_column in frame.columns:
        labels = {
            str(value).strip() for value in frame[label_column].dropna().astype(str).tolist() if str(value).strip()
        }
        if len(labels) == 1:
            return humanize_display_text(next(iter(labels)))
    return humanize_display_text(value_column)


def _shared_numeric_bounds(frames: list[pd.DataFrame], hue_column: str) -> tuple[float | None, float | None]:
    values = [
        pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
        if hue_column in frame.columns
    ]
    if not values:
        return None, None
    combined = pd.concat(values, ignore_index=True)
    if combined.empty or combined.nunique() < 2:
        return None, None
    return float(combined.min()), float(combined.max())


def _categorical_hue_values(frames: list[pd.DataFrame], hue_column: str) -> list[str]:
    return sorted(
        {
            str(value)
            for frame in frames
            if hue_column in frame.columns
            for value in frame[hue_column].fillna("NA").astype(str).unique()
        }
    )


def _plot_panel_title(plot_spec: dict[str, object], index: int, fallback: str) -> str:
    titles = plot_spec.get("panel_titles", [])
    if isinstance(titles, list) and index < len(titles):
        return str(titles[index])
    return fallback


def render_plot_review_surface(
    plot_spec: dict[str, object],
    *,
    frames: list[pd.DataFrame],
    hue_column: str | None,
    reference_labels: list[str],
    joinable_tables: list[dict[str, object]],
    output_root: Path,
    workspace_dir: Path,
):
    kind = str(plot_spec.get("kind") or "")
    if kind == "projection_grid":
        panel_specs = [
            {
                "view_id": str(projection_id),
                "projection_id": str(projection_id),
                "title": _plot_panel_title(plot_spec, index, str(projection_id)),
            }
            for index, projection_id in enumerate(plot_spec.get("projection_ids", []))
        ]
        return render_projection_grid(
            panel_specs,
            frames=frames,
            hue_column=hue_column,
            hue_kinds=_configured_hue_kinds(plot_spec),
            joinable_tables=joinable_tables,
            reference_labels=reference_labels,
            output_root=output_root,
            workspace_dir=workspace_dir,
        )
    if kind in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        return _render_scatter_grid(
            plot_spec,
            frames=frames,
            hue_column=hue_column,
            reference_labels=reference_labels,
        )
    if kind == "categorical_count":
        return _render_categorical_count_grid(plot_spec, frames=frames)
    if kind == "metric_panel_grid":
        return _render_metric_grid(plot_spec, frames=frames)
    if kind == "distribution_grid":
        return _render_distribution_grid(plot_spec, frames=frames)
    if kind == "curve_grid":
        return _render_curve_grid(plot_spec, output_root=output_root)
    return mo.callout("The selected plot does not support live notebook rendering.", kind="warn")


def _render_scatter_grid(
    plot_spec: dict[str, object],
    *,
    frames: list[pd.DataFrame],
    hue_column: str | None,
    reference_labels: list[str],
):
    if not frames or not any(not frame.empty for frame in frames):
        return mo.callout("The selected plot has no persisted scalar data to render.", kind="warn")

    x_column = str(plot_spec.get("x_column") or "x")
    y_column = str(plot_spec.get("y_column") or "y")
    resolved_frames = [frame for frame in frames]
    hue_kinds = _configured_hue_kinds(plot_spec)

    effective_hue = hue_column if hue_column and any(hue_column in frame.columns for frame in resolved_frames) else None
    hue_kind = None
    if effective_hue is not None:
        hue_series = pd.concat(
            [frame[effective_hue] for frame in resolved_frames if effective_hue in frame.columns],
            ignore_index=True,
        )
        hue_kind = classify_hue_series(hue_series, configured_kind=hue_kinds.get(effective_hue))

    category_values = (
        _categorical_hue_values(resolved_frames, effective_hue) if hue_kind != "continuous" and effective_hue else []
    )
    category_map = notebook_category_color_map(category_values)
    numeric_vmin, numeric_vmax = (
        _shared_numeric_bounds(resolved_frames, effective_hue)
        if hue_kind == "continuous" and effective_hue
        else (None, None)
    )
    if hue_kind == "continuous" and (numeric_vmin is None or numeric_vmax is None):
        effective_hue = None
        hue_kind = None

    panel_count = len(resolved_frames)
    rows, columns = _panel_grid_dimensions(panel_count)
    fig, axes = plt.subplots(rows, columns, figsize=_panel_figure_size(panel_count), squeeze=False)
    axes_flat = axes.ravel()
    scatter_artist = None
    max_title_lines = 1

    for axis in axes_flat[panel_count:]:
        axis.set_axis_off()

    for index, (ax, frame) in enumerate(zip(axes_flat, resolved_frames, strict=False)):
        if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
            ax.text(0.5, 0.5, "Panel data missing", ha="center", va="center", fontsize=11, color="#5C6874")
            ax.set_axis_off()
            continue

        point_style = shared_scatter_style(len(frame))
        x_values = frame[x_column].to_numpy(dtype=float)
        y_values = frame[y_column].to_numpy(dtype=float)
        if effective_hue is None or effective_hue not in frame.columns:
            ax.scatter(
                x_values,
                y_values,
                c="#0072B2",
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        elif hue_kind == "continuous":
            hue_values = pd.to_numeric(frame[effective_hue], errors="coerce")
            valid = hue_values.notna()
            scatter_artist = ax.scatter(
                frame.loc[valid, x_column].to_numpy(dtype=float),
                frame.loc[valid, y_column].to_numpy(dtype=float),
                c=hue_values.loc[valid].to_numpy(dtype=float),
                cmap="cividis",
                vmin=numeric_vmin,
                vmax=numeric_vmax,
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        else:
            hue_values = frame[effective_hue].fillna("NA").astype(str)
            for category in category_values:
                mask = hue_values == category
                if not mask.any():
                    continue
                ax.scatter(
                    frame.loc[mask, x_column].to_numpy(dtype=float),
                    frame.loc[mask, y_column].to_numpy(dtype=float),
                    c=category_map[category],
                    s=point_style.point_size,
                    alpha=point_style.alpha,
                    linewidths=point_style.linewidths,
                    edgecolors=point_style.edgecolors,
                    rasterized=point_style.rasterized,
                )

        if x_values.size and float(x_values.min()) < 0.0 < float(x_values.max()):
            ax.axvline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)
        if y_values.size and float(y_values.min()) < 0.0 < float(y_values.max()):
            ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)

        wrapped_title = wrap_plot_title(_plot_panel_title(plot_spec, index, f"Panel {index + 1}"), width=30)
        max_title_lines = max(max_title_lines, wrapped_title.count("\n") + 1)
        ax.set_title(wrapped_title, pad=10 if "\n" in wrapped_title else 8)
        ax.set_xlabel(
            wrap_plot_title(_scatter_axis_label(frame, value_column=x_column, label_column="x_display_name"), width=20)
        )
        ax.set_ylabel(
            wrap_plot_title(_scatter_axis_label(frame, value_column=y_column, label_column="y_display_name"), width=20)
        )
        style_notebook_axes(ax, grid=True, square=True)

    legend_bottom = 0.0
    right_margin = 0.98
    label_right_padding_px = 12.0
    if category_values and effective_hue is not None:
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=category_map[category],
                    label=humanize_display_text(category),
                )
                for category in category_values
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(category_values)),
            frameon=False,
            borderaxespad=0.0,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        style_notebook_legend(legend)
        legend_bottom = 0.11
    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        right_margin = 0.84
        label_right_padding_px = 80.0
        colorbar = fig.colorbar(scatter_artist, ax=[axis for axis in axes_flat[:panel_count]], fraction=0.028, pad=0.02)
        colorbar.set_label(display_hue_label(effective_hue), fontsize=11.5, color=TEXT_COLOR)
        colorbar.ax.tick_params(labelsize=11.5, colors=TEXT_COLOR)

    top_margin = max(0.8, 0.96 - (0.042 * max(max_title_lines - 1, 0)))
    fig.subplots_adjust(
        left=0.09,
        right=right_margin,
        top=top_margin,
        bottom=max(legend_bottom, 0.08),
        wspace=0.24 if panel_count > 1 else 0.18,
        hspace=(0.62 + (0.04 * max(max_title_lines - 1, 0))) if panel_count > 1 else 0.3,
    )
    fig.canvas.draw()

    for ax, frame in zip(axes_flat, resolved_frames, strict=False):
        if ax.axison and not frame.empty:
            draw_reference_labels(
                ax,
                frame,
                reference_labels=reference_labels,
                x_column=x_column,
                y_column=y_column,
                right_padding_px=label_right_padding_px,
                left_padding_px=12.0,
            )
    return render_matplotlib_figure(fig, alt=str(plot_spec.get("plot_id") or "latentdna live plot"))


def _render_metric_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame]):
    if not frames or frames[0].empty:
        return mo.callout("The selected plot has no persisted scalar data to render.", kind="warn")

    frame = frames[0]
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    value_column = spec.value_column if spec.value_column and spec.value_column in frame.columns else None
    if value_column is None:
        value_column = "metric_value" if "metric_value" in frame.columns else "row_count"
    spec = spec.model_copy(update={"value_column": value_column})

    panel_values = list(dict.fromkeys(frame[spec.row_column].astype(str).tolist())) if spec.row_column else ["panel"]
    rows_count, columns = static_panel_grid_dimensions(len(panel_values))
    square_metric_panels = metric_panel_uses_square_axes(spec.plot_id)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=static_grid_figure_size(len(panel_values), square_panels=square_metric_panels),
        squeeze=False,
    )
    records = frame.to_dict(orient="records")
    color_map, categories = static_category_color_map([records], spec.color_column)
    for axis in axes.ravel()[len(panel_values) :]:
        axis.set_axis_off()
    for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
        panel_rows = [row for row in records if str(row.get(spec.row_column)) == panel_value]
        panel_title = str(panel_rows[0].get(spec.panel_column) or panel_value) if panel_rows else panel_value
        _render_metric_panel(
            axis,
            rows=panel_rows,
            spec=spec,
            panel_title=panel_title,
            color_map=color_map,
            square=square_metric_panels,
        )
    legend_bottom = 0.0
    if categories and spec.color_column is not None:
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=color_map[category],
                    label=humanize_display_text(category),
                )
                for category in categories
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(categories)),
            frameon=False,
            borderaxespad=0.0,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        style_notebook_legend(legend)
        legend_bottom = 0.11
    fig.tight_layout(rect=(0.0, legend_bottom, 1.0, 1.0), pad=0.95, h_pad=1.4, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=str(plot_spec.get("plot_id") or "latentdna live plot"))


def _render_distribution_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame]):
    resolved_frames = [frame for frame in frames if not frame.empty]
    if not resolved_frames:
        return mo.callout("The selected plot has no persisted scalar data to render.", kind="warn")

    spec = ResolvedPlotSpec.model_validate(plot_spec)
    metric_columns = list(spec.metric_columns or [])
    scalar_tables: list[tuple[str, pd.DataFrame, str]] = []
    for scalar_id, frame in zip(spec.scalar_ids, resolved_frames, strict=False):
        numeric_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column]) and column not in {"left_indices", "right_indices"}
        ]
        if metric_columns:
            for metric_column in metric_columns:
                if metric_column in numeric_columns:
                    scalar_tables.append((str(scalar_id), frame, metric_column))
        else:
            metric_column = (
                spec.value_column
                if spec.value_column in numeric_columns
                else (numeric_columns[0] if numeric_columns else None)
            )
            if metric_column is not None:
                scalar_tables.append((str(scalar_id), frame, metric_column))
    if not scalar_tables:
        return mo.callout("The selected plot has no numeric distributions to render.", kind="warn")

    rows_count, columns = static_panel_grid_dimensions(len(scalar_tables))
    square_distribution_panels = spec.plot_id == "context_delta_distributions"
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=static_grid_figure_size(len(scalar_tables), square_panels=square_distribution_panels),
        squeeze=False,
    )
    titles = spec.panel_titles or [
        (
            f"{_derived_panel_label(scalar_id)} · {humanize_display_text(metric_column)}"
            if _derived_panel_label(scalar_id)
            else humanize_display_text(metric_column)
        )
        for scalar_id, _, metric_column in scalar_tables
    ]
    for axis in axes.ravel()[len(scalar_tables) :]:
        axis.set_axis_off()
    for axis, (_, frame, metric_column), panel_title in zip(axes.ravel(), scalar_tables, titles, strict=False):
        _render_distribution_panel(
            axis,
            rows=frame.to_dict(orient="records"),
            metric_column=metric_column,
            color_column=spec.color_column,
            render_mode=spec.render_mode or "histogram",
            panel_title=panel_title,
            square=square_distribution_panels,
        )
    fig.tight_layout(pad=0.95, h_pad=1.4, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=str(plot_spec.get("plot_id") or "latentdna live plot"))


def _render_curve_grid(plot_spec: dict[str, object], *, output_root: Path):
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    reducer_summaries: list[tuple[str, dict[str, object]]] = []
    for reducer_id in spec.reducer_ids:
        summary_path = output_root / "reducers" / str(reducer_id) / "summary.json"
        if not summary_path.is_file():
            continue
        reducer_summaries.append((str(reducer_id), json.loads(summary_path.read_text(encoding="utf-8"))))
    if not reducer_summaries:
        return mo.callout("The selected plot has no persisted reducer summaries to render.", kind="warn")

    rows_count, columns = static_panel_grid_dimensions(len(reducer_summaries))
    square_curve_panels = spec.plot_id == "representation_scree_diagnostic"
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=static_grid_figure_size(len(reducer_summaries), square_panels=square_curve_panels),
        squeeze=False,
    )
    titles = spec.panel_titles or [humanize_display_text(reducer_id) for reducer_id, _ in reducer_summaries]
    for axis in axes.ravel()[len(reducer_summaries) :]:
        axis.set_axis_off()
    for axis, (reducer_id, summary), panel_title in zip(axes.ravel(), reducer_summaries, titles, strict=False):
        _render_curve_panel(
            axis,
            reducer_id=reducer_id,
            summary=summary,
            panel_title=panel_title,
            square=square_curve_panels,
            show_legend=False,
        )
    legend = fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", linewidth=1.8, color="#0072B2", label="Explained variance ratio"),
            plt.Line2D([], [], marker="s", linewidth=1.8, color="#009E73", label="Cumulative variance ratio"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    style_notebook_legend(legend)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), pad=0.95, h_pad=1.4, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=str(plot_spec.get("plot_id") or "latentdna live plot"))


def _render_categorical_count_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame]):
    if not frames or frames[0].empty:
        return mo.callout("The selected plot has no persisted scalar data to render.", kind="warn")

    frame = frames[0].copy()
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    required_columns = [spec.row_column, spec.column_column, spec.value_column]
    missing_columns = [column for column in required_columns if column is not None and column not in frame.columns]
    if missing_columns:
        return mo.callout(
            f"The selected plot is missing required categorical-count columns: {', '.join(missing_columns)}.",
            kind="warn",
        )

    if spec.panel_column and spec.panel_column in frame.columns:
        panel_values = list(dict.fromkeys(frame[spec.panel_column].astype(str).tolist()))
    else:
        panel_values = [str(plot_spec.get("plot_id") or "panel")]

    if len(panel_values) <= 3:
        rows_count, columns = len(panel_values), 1
    else:
        rows_count, columns = static_panel_grid_dimensions(len(panel_values))
    fig, axes = plt.subplots(rows_count, columns, figsize=(7.2 * columns, 3.85 * rows_count), squeeze=False)

    for axis in axes.ravel()[len(panel_values) :]:
        axis.set_axis_off()

    for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
        if spec.panel_column and spec.panel_column in frame.columns:
            panel_frame = frame.loc[frame[spec.panel_column].astype(str) == panel_value].copy()
        else:
            panel_frame = frame.copy()
        if panel_frame.empty:
            axis.set_axis_off()
            continue
        if "order" in panel_frame.columns:
            panel_frame = panel_frame.sort_values("order", kind="stable")

        labels = [humanize_display_text(value) for value in panel_frame[spec.column_column].tolist()]
        values = pd.to_numeric(panel_frame[spec.value_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_positions = np.arange(len(panel_frame), dtype=float)
        color_map = notebook_category_color_map(labels)
        bar_colors = [color_map[label] for label in labels]

        axis.barh(
            y_positions,
            values,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        axis.set_yticks(y_positions, labels)
        axis.invert_yaxis()
        axis.set_ylabel("")
        axis.set_xlabel("Percent of N")
        show_as_percent = all(0.0 <= value <= 1.0 for value in values)
        if show_as_percent:
            from matplotlib.ticker import PercentFormatter

            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        max_value = max(values, default=0.0)
        axis.set_xlim(0.0, max_value * 1.14 if max_value > 0 else 1.0)

        denominator_values = {
            int(float(value))
            for value in panel_frame.get("denominator", pd.Series(dtype=float)).tolist()
            if value is not None and not pd.isna(value)
        }
        panel_title = humanize_display_text(panel_value)
        if len(denominator_values) == 1:
            panel_title = f"{panel_title}\nN = {next(iter(denominator_values)):,}"
        axis.set_title(wrap_plot_title(panel_title, width=24), pad=10)
        style_notebook_axes(axis, grid=True)

        counts = pd.to_numeric(panel_frame.get("count", pd.Series([np.nan] * len(panel_frame))), errors="coerce")
        percents = pd.to_numeric(panel_frame.get("percent", pd.Series([np.nan] * len(panel_frame))), errors="coerce")
        for index, (y_position, value) in enumerate(zip(y_positions, values, strict=False)):
            label_parts: list[str] = []
            if index < len(counts) and not pd.isna(counts.iloc[index]):
                label_parts.append(f"{int(counts.iloc[index]):,}")
            if index < len(percents) and not pd.isna(percents.iloc[index]):
                label_parts.append(f"{float(percents.iloc[index]):.1f}%")
            axis.text(
                value + (max_value * 0.02 if max_value > 0 else 0.03),
                y_position,
                " | ".join(label_parts) if label_parts else f"{value:.1%}",
                va="center",
                ha="left",
                fontsize=11.0,
                color=TEXT_COLOR,
            )

    fig.tight_layout(pad=0.95, h_pad=1.5, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=str(plot_spec.get("plot_id") or "latentdna live plot"))
