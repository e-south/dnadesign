"""Metric-panel data contracts for static plot rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...metadata_axes import AxisStyle, ordered_categories_for_axis
from ...visual_style import (
    PUBLICATION_PALETTE,
    SPINE_COLOR,
    TEXT_COLOR,
    ZERO_LINE_COLOR,
    humanize_display_text,
    wrap_plot_title,
)
from ...workspaces.loader import WorkspaceContext
from ..axes import apply_axes_style, wrapped_axis_label
from ..layout import _HORIZONTAL_GROUPED_METRIC_PLOT_IDS
from ..panels import render_placeholder_panel
from ..tables import read_table_rows, require_row_columns
from .metric_labels import candidate_tick_label, metric_tick_labels_need_rotation, style_metric_tick_labels
from .scatter import axis_category_value, coerce_finite_float


@dataclass(frozen=True, slots=True)
class MetricPanelGroup:
    """Rows belonging to one semantic metric panel."""

    key: tuple[str, ...]
    title: str
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class MetricPanelGridInput:
    """Validated rows and resolved config for one metric-panel grid."""

    rows: list[dict[str, object]]
    resolved_spec: ResolvedPlotSpec
    groups: list[MetricPanelGroup]


def _add_metric_uncertainty_note(ax: Any, *, ci_enabled: bool) -> None:
    if not ci_enabled:
        return
    ax.text(
        0.012,
        0.02,
        "Whiskers: 95% bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=SPINE_COLOR,
    )


def _metric_panel_title(text: object, *, plot_id: str | None) -> str:
    del plot_id
    return wrap_plot_title(text, width=24, max_lines=None)


def metric_panel_label_column(spec: ResolvedPlotSpec) -> str:
    """Return the row-label column used by metric panels."""

    label_column = spec.label_column or spec.column_column
    if label_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires a label column")
    return label_column


def metric_panel_value_column(spec: ResolvedPlotSpec) -> str:
    """Return the configured metric value column used by metric panels."""

    if spec.value_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires value_column")
    return spec.value_column


def _metric_panel_group_columns(spec: ResolvedPlotSpec) -> list[str]:
    if spec.row_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires row_column")
    columns = [spec.row_column]
    if spec.panel_column is not None and spec.panel_column != spec.row_column:
        columns.append(spec.panel_column)
    return columns


def metric_panel_required_columns(spec: ResolvedPlotSpec) -> list[str | None]:
    """Return table columns required to render a metric-panel grid."""

    return [
        spec.row_column,
        spec.panel_column,
        spec.column_column,
        spec.label_column,
        metric_panel_value_column(spec),
        spec.color_column,
        spec.direction_column,
        spec.unit_column,
    ]


def metric_panel_groups(rows: list[dict[str, object]], spec: ResolvedPlotSpec) -> list[MetricPanelGroup]:
    """Group rows into semantic panels without collapsing distinct display metrics."""

    group_columns = _metric_panel_group_columns(spec)
    require_row_columns(rows, group_columns, context="metric_panel_grid panel grouping")
    groups_by_key: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[column]) for column in group_columns)
        groups_by_key.setdefault(key, []).append(row)
    groups: list[MetricPanelGroup] = []
    for key, group_rows in groups_by_key.items():
        first = group_rows[0]
        title = str(first[spec.panel_column]) if spec.panel_column is not None else str(first[spec.row_column])
        groups.append(MetricPanelGroup(key=key, title=title, rows=group_rows))
    return groups


def grouped_family_metric_keys_are_unique(rows: list[dict[str, object]]) -> bool:
    """Return whether grouped family bars can be drawn without row overwrites."""

    required_columns = ("candidate_family", "candidate_model", "candidate_scope")
    if any(any(column not in row for column in required_columns) for row in rows):
        return False
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = tuple(str(row.get(column) or "").strip() for column in required_columns)
        if not all(key):
            return False
        if key in seen:
            return False
        seen.add(key)
    return True


def metric_panel_uses_grouped_family_bars(rows: list[dict[str, object]], spec: ResolvedPlotSpec) -> bool:
    """Gate grouped family rendering on both config intent and unique semantic keys."""

    if spec.color_column != "candidate_family":
        return False
    return grouped_family_metric_keys_are_unique(rows)


def metric_panel_needs_candidate_label_ticks(rows: list[dict[str, object]], spec: ResolvedPlotSpec) -> bool:
    """Detect when compact candidate fields would hide distinct rows."""

    if spec.color_column != "candidate_family":
        return False
    required_columns = ("candidate_family", "candidate_model", "candidate_scope")
    if any(any(column not in row for column in required_columns) for row in rows):
        return False
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = tuple(str(row.get(column) or "").strip() for column in required_columns)
        if key in seen:
            return any(str(candidate_row.get("candidate_label") or "").strip() for candidate_row in rows)
        seen.add(key)
    return False


def load_metric_panel_grid_input(context: WorkspaceContext, spec: ResolvedPlotSpec) -> MetricPanelGridInput:
    """Load and validate metric-panel rows from a scalar table artifact."""

    if spec.scalar_id is None:
        raise ContractViolationError("metric_panel_grid rendering requires a scalar artifact")
    table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
    if not table_path.exists():
        raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {spec.scalar_id}")
    rows = read_table_rows(
        table_path,
        required_columns=metric_panel_required_columns(spec),
        artifact_label=f"metric_panel_grid scalar {spec.scalar_id}",
    )
    if not rows:
        raise ContractViolationError("metric_panel_grid rendering requires at least one row")
    groups = metric_panel_groups(rows, spec)
    return MetricPanelGridInput(rows=rows, resolved_spec=spec, groups=groups)


def _metric_axis_label(
    *,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
) -> str:
    base_source = str(spec.value_label or spec.value_column or "metric value")
    if base_source.strip().casefold() == "metric value" and spec.panel_column is not None:
        panel_labels = {
            str(row.get(spec.panel_column) or "").strip()
            for row in rows
            if str(row.get(spec.panel_column) or "").strip()
        }
        if len(panel_labels) == 1:
            base_source = next(iter(panel_labels))
    # Panel-specific qualifiers, such as "Reference set: W collection", belong
    # in the panel title. Repeating them in every axis label makes dense grids
    # hard to read and can force ellipsis truncation that hides the metric name.
    base_source = next((line.strip() for line in base_source.splitlines() if line.strip()), base_source)
    base_label = humanize_display_text(base_source)
    if spec.unit_column is None:
        return base_label
    units = {
        str(row.get(spec.unit_column) or "").strip() for row in rows if str(row.get(spec.unit_column) or "").strip()
    }
    if len(units) != 1:
        return base_label
    unit = next(iter(units))
    return f"{base_label} ({humanize_display_text(unit)})"


def _sorted_metric_rows(rows: list[dict[str, object]], *, spec: ResolvedPlotSpec) -> list[dict[str, object]]:
    label_column = metric_panel_label_column(spec)
    value_column = metric_panel_value_column(spec)
    sort_rule = spec.sort_rule or "panel_direction"
    candidate_scope_order = {
        "merged_anchor_insert_seq_mean": 0,
        "full_context_1kb": 1,
        "full_context_anchor_mean": 2,
        "context_anchor_mean_bidir_concat": 3,
        "reverse_complement_context_1kb": 4,
        "reverse_complement_context_anchor_mean": 5,
        "reference_core60": 6,
    }
    candidate_family_order = {
        "intermediate_embedding": 0,
        "output_layer_mean": 1,
    }

    def _candidate_order_key(row: dict[str, object]) -> tuple[float, int, int, str, str]:
        explicit_order = coerce_finite_float(row.get("candidate_order"))
        if explicit_order is not None:
            return (
                explicit_order,
                candidate_family_order.get(str(row.get("candidate_family") or ""), 99),
                candidate_scope_order.get(str(row.get("candidate_scope") or ""), 99),
                str(row.get("candidate_id") or "").casefold(),
                str(row.get(label_column) or "").casefold(),
            )
        return (
            1_000_000.0,
            candidate_family_order.get(str(row.get("candidate_family") or ""), 99),
            candidate_scope_order.get(str(row.get("candidate_scope") or ""), 99),
            str(row.get("candidate_id") or "").casefold(),
            str(row.get(label_column) or "").casefold(),
        )

    if sort_rule == "candidate_order":
        return sorted(rows, key=_candidate_order_key)
    if sort_rule == "label_asc":
        return sorted(rows, key=lambda row: str(row.get(label_column) or "").casefold())

    def _value_sort_key(row: dict[str, object], *, descending: bool) -> tuple[int, float, str]:
        value = coerce_finite_float(row.get(value_column))
        label = str(row.get(label_column) or "").casefold()
        if value is None:
            return (1, 0.0, label)
        sortable = -value if descending else value
        return (0, sortable, label)

    if sort_rule == "value_asc":
        return sorted(rows, key=lambda row: _value_sort_key(row, descending=False))
    if sort_rule == "value_desc":
        return sorted(rows, key=lambda row: _value_sort_key(row, descending=True))
    direction = ""
    if spec.direction_column is not None and rows:
        direction = str(rows[0].get(spec.direction_column) or "").strip().lower()
    descending = direction != "lower_is_better"
    return sorted(rows, key=lambda row: _value_sort_key(row, descending=descending))


def render_metric_panel(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    panel_title: str,
    color_map: dict[str, str],
    square: bool = False,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    """Render one metric-panel grid cell from validated metric rows."""

    value_column = metric_panel_value_column(spec)
    label_column = metric_panel_label_column(spec)
    ordered_rows = _sorted_metric_rows(rows, spec=spec)
    grouped_family_bars = metric_panel_uses_grouped_family_bars(ordered_rows, spec)
    horizontal_metric = spec.plot_id == "representation_health_summary" and not grouped_family_bars
    include_family = not (spec.color_column == "candidate_family")
    use_candidate_label_ticks = metric_panel_needs_candidate_label_ticks(ordered_rows, spec)
    tick_fallback_column = "candidate_label" if use_candidate_label_ticks else label_column
    labels = [
        candidate_tick_label(
            row,
            fallback_column=tick_fallback_column,
            plot_id=spec.plot_id,
            include_family=include_family,
            multiline=not horizontal_metric,
            force_fallback=use_candidate_label_ticks,
        )
        for row in ordered_rows
    ]
    if spec.color_column is not None:
        require_row_columns(ordered_rows, [spec.color_column], context="metric_panel_grid color encoding")
        bar_colors = [
            color_map[axis_category_value(row, spec.color_column, axis_styles=axis_styles)] for row in ordered_rows
        ]
    else:
        bar_colors = [PUBLICATION_PALETTE[0]] * len(ordered_rows)
    ci_enabled = spec.ci_lower_column is not None and spec.ci_upper_column is not None and ordered_rows
    horizontal_grouped_metric = grouped_family_bars and spec.plot_id in _HORIZONTAL_GROUPED_METRIC_PLOT_IDS

    if horizontal_grouped_metric:
        family_order = ordered_categories_for_axis(None, [str(row["candidate_family"]) for row in ordered_rows])
        group_keys = list(
            dict.fromkeys(
                (
                    str(row["candidate_model"]),
                    str(row["candidate_scope"]),
                )
                for row in ordered_rows
            )
        )
        shared_scope = {
            str(row.get("candidate_scope") or "").strip()
            for row in ordered_rows
            if str(row.get("candidate_scope") or "").strip()
        }
        include_scope = len(shared_scope) > 1
        group_labels = [
            candidate_tick_label(
                {
                    "candidate_model": model,
                    "candidate_scope": scope,
                },
                fallback_column=label_column,
                plot_id=spec.plot_id,
                include_family=False,
                include_scope=include_scope,
                multiline=False,
            )
            for model, scope in group_keys
        ]
        group_positions = np.arange(len(group_keys), dtype=float)
        group_height = min(0.78, 0.32 * max(len(family_order), 1))
        bar_height = group_height / max(len(family_order), 1)
        offsets = np.linspace(
            -(group_height / 2.0) + (bar_height / 2.0),
            (group_height / 2.0) - (bar_height / 2.0),
            max(len(family_order), 1),
        )
        bar_value_pairs: list[tuple[Any, float]] = []
        errorbar_specs: list[tuple[float, float, float, float]] = []
        missing_positions: list[float] = []
        for family, offset in zip(family_order, offsets, strict=False):
            family_rows = {
                (str(row["candidate_model"]), str(row["candidate_scope"])): row
                for row in ordered_rows
                if str(row["candidate_family"]) == family
            }
            family_positions: list[float] = []
            family_values: list[float] = []
            family_ci_rows: list[dict[str, object]] = []
            for group_position, group_key in zip(group_positions, group_keys, strict=False):
                row = family_rows.get(group_key)
                if row is None:
                    continue
                y_position = float(group_position + offset)
                value = coerce_finite_float(row.get(value_column))
                if value is None:
                    missing_positions.append(y_position)
                    continue
                family_positions.append(y_position)
                family_values.append(value)
                family_ci_rows.append(row)
            if not family_positions:
                continue
            family_bars = ax.barh(
                family_positions,
                family_values,
                height=bar_height * 0.9,
                color=color_map[family],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
            )
            bar_value_pairs.extend(zip(family_bars, family_values, strict=True))
            if ci_enabled:
                for bar, row in zip(family_bars, family_ci_rows, strict=False):
                    lower = coerce_finite_float(row.get(spec.ci_lower_column))
                    upper = coerce_finite_float(row.get(spec.ci_upper_column))
                    if lower is None or upper is None:
                        continue
                    errorbar_specs.append(
                        (
                            float(bar.get_y() + (bar.get_height() / 2.0)),
                            float(row[value_column]),
                            float(lower),
                            float(upper),
                        )
                    )
        ax.set_yticks(group_positions, group_labels)
        if group_positions.size:
            ax.set_ylim(float(group_positions.min()) - 0.55, float(group_positions.max()) + 0.55)
        ax.tick_params(axis="y", pad=6)
        style_metric_tick_labels(ax, label_count=max(len(group_labels), len(bar_value_pairs)), axis="y")
        finite_values = [value for _, value in bar_value_pairs]
        finite_value_array = np.asarray(finite_values, dtype=np.float64)
        if spec.reference_line is not None:
            ax.axvline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
            ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if errorbar_specs:
            ys = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
            xs = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
            lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
            uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
            ax.errorbar(
                xs,
                ys,
                xerr=np.vstack([lowers, uppers]),
                fmt="none",
                ecolor=SPINE_COLOR,
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.85,
            )
        ax.set_ylabel("")
        ax.set_xlabel(wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=28, max_lines=2))
        ax.set_title(_metric_panel_title(panel_title, plot_id=spec.plot_id), pad=8)
        apply_axes_style(ax, grid=True, square=square)
        _add_metric_uncertainty_note(ax, ci_enabled=ci_enabled)
        ax.margins(x=0.02, y=0.02)
        if not finite_value_array.size:
            render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Metric unavailable",
                detail="No finite values in this snapshot",
                square=square,
            )
            return
        span = float(finite_value_array.max() - finite_value_array.min())
        offset = max(span * 0.03, 0.018) if span > 0 else 0.018
        low = min(0.0, float(finite_value_array.min()))
        high = max(0.0, float(finite_value_array.max()))
        padding = max((high - low) * 0.1, 0.04)
        ax.set_xlim(low - padding, high + padding)
        missing_label_x = low + (padding * 0.6)
        ax.invert_yaxis()
        for bar, value in bar_value_pairs:
            x_text = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(
                x_text,
                bar.get_y() + (bar.get_height() / 2.0),
                f"{value:.3g}",
                va="center",
                ha=ha,
                fontsize=9,
                color=TEXT_COLOR,
            )
        for position in missing_positions:
            ax.text(
                missing_label_x,
                float(position),
                "NA",
                va="center",
                ha="left",
                fontsize=8.5,
                color=SPINE_COLOR,
            )
        return

    if horizontal_metric:
        positions = np.arange(len(ordered_rows), dtype=float)
        finite_positions: list[float] = []
        finite_values: list[float] = []
        finite_colors: list[str] = []
        finite_rows: list[dict[str, object]] = []
        missing_positions: list[float] = []
        for position, row, color in zip(positions, ordered_rows, bar_colors, strict=True):
            value = coerce_finite_float(row.get(value_column))
            if value is None:
                missing_positions.append(float(position))
                continue
            finite_positions.append(float(position))
            finite_values.append(value)
            finite_colors.append(color)
            finite_rows.append(row)

        bars = ax.barh(
            finite_positions,
            finite_values,
            color=finite_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        if ci_enabled:
            errorbar_specs: list[tuple[float, float, float, float]] = []
            for position, row in zip(finite_positions, finite_rows, strict=True):
                lower = coerce_finite_float(row.get(spec.ci_lower_column))
                upper = coerce_finite_float(row.get(spec.ci_upper_column))
                if lower is None or upper is None:
                    continue
                errorbar_specs.append((float(position), float(row[value_column]), lower, upper))
            if errorbar_specs:
                ys = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
                xs = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
                lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
                uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
                ax.errorbar(
                    xs,
                    ys,
                    xerr=np.vstack([lowers, uppers]),
                    fmt="none",
                    ecolor=SPINE_COLOR,
                    elinewidth=0.9,
                    capsize=2.0,
                    alpha=0.85,
                )
        ax.set_yticks(positions, labels)
        ax.tick_params(axis="y", pad=6)
        style_metric_tick_labels(ax, label_count=len(labels), axis="y")
        finite_value_array = np.asarray(finite_values, dtype=np.float64)
        if spec.reference_line is not None:
            ax.axvline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
            ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        ax.set_ylabel("")
        ax.set_xlabel(wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=28, max_lines=2))
        ax.set_title(_metric_panel_title(panel_title, plot_id=spec.plot_id), pad=8)
        apply_axes_style(ax, grid=True, square=square)
        _add_metric_uncertainty_note(ax, ci_enabled=ci_enabled)
        ax.margins(x=0.02, y=0.02)
        if not finite_value_array.size:
            render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Metric unavailable",
                detail="No finite values in this snapshot",
                square=square,
            )
            return
        if finite_value_array.size:
            span = float(finite_value_array.max() - finite_value_array.min())
            offset = max(span * 0.03, 0.018) if span > 0 else 0.018
            low = min(0.0, float(finite_value_array.min()))
            high = max(0.0, float(finite_value_array.max()))
            padding = max((high - low) * 0.1, 0.04)
            ax.set_xlim(low - padding, high + padding)
            missing_label_x = low + (padding * 0.6)
        else:
            offset = 0.018
            ax.set_xlim(-0.2, 0.2)
            missing_label_x = 0.04
        ax.invert_yaxis()
        for bar, value in zip(bars, finite_values, strict=True):
            x_text = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(
                x_text,
                bar.get_y() + (bar.get_height() / 2.0),
                f"{value:.3g}",
                va="center",
                ha=ha,
                fontsize=9,
                color=TEXT_COLOR,
            )
        for position in missing_positions:
            ax.text(
                missing_label_x,
                float(position),
                "NA",
                va="center",
                ha="left",
                fontsize=8.5,
                color=SPINE_COLOR,
            )
        return

    bar_value_pairs: list[tuple[Any, float]] = []
    errorbar_specs: list[tuple[float, float, float, float]] = []
    missing_positions: list[float] = []
    if grouped_family_bars:
        family_order = ordered_categories_for_axis(None, [str(row["candidate_family"]) for row in ordered_rows])
        group_keys = list(
            dict.fromkeys(
                (
                    str(row["candidate_model"]),
                    str(row["candidate_scope"]),
                )
                for row in ordered_rows
            )
        )
        shared_scope = {
            str(row.get("candidate_scope") or "").strip()
            for row in ordered_rows
            if str(row.get("candidate_scope") or "").strip()
        }
        include_scope = len(shared_scope) > 1
        group_labels = [
            candidate_tick_label(
                {
                    "candidate_model": model,
                    "candidate_scope": scope,
                },
                fallback_column=label_column,
                plot_id=spec.plot_id,
                include_family=False,
                include_scope=include_scope,
                multiline=False,
            )
            for model, scope in group_keys
        ]
        group_positions = np.arange(len(group_keys), dtype=float)
        group_width = min(0.78, 0.32 * max(len(family_order), 1))
        bar_width = group_width / max(len(family_order), 1)
        offsets = np.linspace(
            -(group_width / 2.0) + (bar_width / 2.0),
            (group_width / 2.0) - (bar_width / 2.0),
            max(len(family_order), 1),
        )
        for family, offset in zip(family_order, offsets, strict=False):
            family_rows = {
                (str(row["candidate_model"]), str(row["candidate_scope"])): row
                for row in ordered_rows
                if str(row["candidate_family"]) == family
            }
            family_positions: list[float] = []
            family_values: list[float] = []
            family_ci_rows: list[dict[str, object]] = []
            for group_position, group_key in zip(group_positions, group_keys, strict=False):
                row = family_rows.get(group_key)
                if row is None:
                    continue
                x_position = float(group_position + offset)
                value = coerce_finite_float(row.get(value_column))
                if value is None:
                    missing_positions.append(x_position)
                    continue
                family_positions.append(x_position)
                family_values.append(value)
                family_ci_rows.append(row)
            if not family_positions:
                continue
            family_bars = ax.bar(
                family_positions,
                family_values,
                width=bar_width * 0.9,
                color=color_map[family],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
            )
            bar_value_pairs.extend(zip(family_bars, family_values, strict=True))
            if ci_enabled:
                for bar, row in zip(family_bars, family_ci_rows, strict=False):
                    lower = coerce_finite_float(row.get(spec.ci_lower_column))
                    upper = coerce_finite_float(row.get(spec.ci_upper_column))
                    if lower is None or upper is None:
                        continue
                    errorbar_specs.append(
                        (
                            float(bar.get_x() + (bar.get_width() / 2.0)),
                            float(row[value_column]),
                            float(lower),
                            float(upper),
                        )
                    )
        ax.set_xticks(group_positions, group_labels)
        if group_positions.size:
            ax.set_xlim(float(group_positions.min()) - 0.55, float(group_positions.max()) + 0.55)
    else:
        positions = np.arange(len(ordered_rows), dtype=float)
        finite_positions: list[float] = []
        finite_values: list[float] = []
        finite_colors: list[str] = []
        finite_rows: list[dict[str, object]] = []
        for position, row, color in zip(positions, ordered_rows, bar_colors, strict=True):
            value = coerce_finite_float(row.get(value_column))
            if value is None:
                missing_positions.append(float(position))
                continue
            finite_positions.append(float(position))
            finite_values.append(value)
            finite_colors.append(color)
            finite_rows.append(row)
        bars = ax.bar(
            finite_positions,
            finite_values,
            color=finite_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        bar_value_pairs.extend(zip(bars, finite_values, strict=True))
        if ci_enabled:
            for position, row in zip(finite_positions, finite_rows, strict=True):
                lower = coerce_finite_float(row.get(spec.ci_lower_column))
                upper = coerce_finite_float(row.get(spec.ci_upper_column))
                if lower is None or upper is None:
                    continue
                errorbar_specs.append(
                    (
                        float(position),
                        float(row[value_column]),
                        float(lower),
                        float(upper),
                    )
                )
        ax.set_xticks(positions, labels)
        if positions.size:
            ax.set_xlim(float(positions.min()) - 0.55, float(positions.max()) + 0.55)
    ax.tick_params(axis="x", pad=6)
    tick_labels_for_count = group_labels if grouped_family_bars else labels
    tick_rotation = (
        32.0
        if metric_tick_labels_need_rotation(
            tick_labels_for_count,
            grouped_family_bars=grouped_family_bars,
            plot_id=spec.plot_id,
        )
        else 0.0
    )
    style_metric_tick_labels(
        ax,
        label_count=max(len(tick_labels_for_count), len(bar_value_pairs)),
        rotation=tick_rotation,
        ha="right" if tick_rotation else None,
    )
    finite_values = [value for _, value in bar_value_pairs]
    finite_value_array = np.asarray(finite_values, dtype=np.float64)
    if spec.reference_line is not None:
        ax.axhline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
    if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
    if errorbar_specs:
        xs = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
        ys = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
        lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
        uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([lowers, uppers]),
            fmt="none",
            ecolor=SPINE_COLOR,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.85,
        )
    ax.set_xlabel("")
    ax.set_ylabel(wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=20))
    ax.set_title(_metric_panel_title(panel_title, plot_id=spec.plot_id), pad=8)
    apply_axes_style(ax, grid=True, square=square)
    _add_metric_uncertainty_note(ax, ci_enabled=ci_enabled)
    ax.margins(x=0.02, y=0.02)
    if not finite_value_array.size:
        render_placeholder_panel(
            ax,
            panel_title=panel_title,
            message="Metric unavailable",
            detail="No finite values in this snapshot",
            square=square,
        )
        return
    if finite_value_array.size:
        span = float(finite_value_array.max() - finite_value_array.min())
        offset = max(span * 0.03, 0.018) if span > 0 else 0.018
        low = min(0.0, float(finite_value_array.min()))
        high = max(0.0, float(finite_value_array.max()))
        padding = max((high - low) * 0.1, 0.045)
        ax.set_ylim(low - padding, high + padding)
        missing_label_y = low + (padding * 0.55)
    else:
        offset = 0.018
        ax.set_ylim(-0.2, 0.2)
        missing_label_y = 0.03
    for bar, value in bar_value_pairs:
        y_text = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            y_text,
            f"{value:.3g}",
            va=va,
            ha="center",
            fontsize=9,
            color=TEXT_COLOR,
        )
    for position in missing_positions:
        ax.text(
            float(position),
            missing_label_y,
            "NA",
            va="bottom",
            ha="center",
            fontsize=8.5,
            color=SPINE_COLOR,
        )
