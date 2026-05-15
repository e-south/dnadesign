"""Static summaries for categorical-feature enrichment tables."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import PlotStaticFilterConfig, ResolvedPlotSpec
from ...visual_style import (
    PUBLICATION_PALETTE,
    SPINE_COLOR,
    TEXT_COLOR,
    ZERO_LINE_COLOR,
    humanize_display_text,
    wrap_plot_title,
)
from ...workspaces.loader import WorkspaceContext
from ..axes import apply_axes_style, resolved_axis_label
from ..layout import _panel_grid_dimensions
from ..panels import render_placeholder_panel
from ..tables import read_table_rows, require_row_columns
from .scatter import coerce_finite_float


@dataclass(frozen=True, slots=True)
class CategoricalEnrichmentSummaryResult:
    """Rendered enrichment summary figure and audit metadata."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)


def _required_columns(spec: ResolvedPlotSpec) -> list[str | None]:
    return [
        spec.row_column,
        spec.column_column,
        spec.value_column,
        spec.count_column,
        spec.total_column,
        spec.p_value_column,
        spec.q_value_column,
        spec.common_feature_column,
        *(item.column for item in spec.static_filters),
    ]


def _filter_value_matches(value: object, expected: str | int | float | bool) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool) and value is expected
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        actual = coerce_finite_float(value)
        return actual is not None and math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
    return str(value) == str(expected)


def _apply_static_filters(
    rows: list[dict[str, object]],
    filters: list[PlotStaticFilterConfig],
) -> list[dict[str, object]]:
    filtered_rows = rows
    for item in filters:
        filtered_rows = [row for row in filtered_rows if _filter_value_matches(row.get(item.column), item.equals)]
    return filtered_rows


def _ordered_group_values(rows: list[dict[str, object]], spec: ResolvedPlotSpec) -> list[str]:
    if spec.row_column is None:
        raise ContractViolationError("categorical_enrichment_summary requires row_column")
    observed = list(dict.fromkeys(str(row.get(spec.row_column)) for row in rows))
    if not spec.group_order:
        return sorted(observed, key=str.casefold)
    ordered = [group for group in spec.group_order if group in observed]
    extras = sorted((group for group in observed if group not in set(spec.group_order)), key=str.casefold)
    return [*ordered, *extras]


def _row_sort_key(row: dict[str, object], spec: ResolvedPlotSpec) -> tuple[int, float, float, str]:
    assert spec.value_column is not None
    value = coerce_finite_float(row.get(spec.value_column))
    p_value = coerce_finite_float(row.get(spec.p_value_column)) if spec.p_value_column is not None else None
    feature = str(row.get(spec.column_column) or "").casefold()
    if value is None:
        return (1, 0.0, p_value if p_value is not None else math.inf, feature)
    return (0, -value, p_value if p_value is not None else math.inf, feature)


def _selected_group_rows(
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    *,
    group_value: str,
) -> list[dict[str, object]]:
    if spec.row_column is None:
        raise ContractViolationError("categorical_enrichment_summary requires row_column")
    grouped = [row for row in rows if str(row.get(spec.row_column)) == group_value]
    ranked = sorted(grouped, key=lambda row: _row_sort_key(row, spec))
    limit = spec.max_features_per_group or 8
    return ranked[:limit]


def _feature_label(row: dict[str, object], spec: ResolvedPlotSpec) -> str:
    if spec.column_column is None:
        raise ContractViolationError("categorical_enrichment_summary requires column_column")
    label = str(row.get(spec.column_column) or "")
    count = row.get(spec.count_column) if spec.count_column is not None else None
    total = row.get(spec.total_column) if spec.total_column is not None else None
    if count is not None and total is not None:
        label = f"{label} ({count}/{total})"
    return wrap_plot_title(label, width=18, max_lines=2)


def _format_stat(value: object, *, prefix: str) -> str | None:
    numeric = coerce_finite_float(value)
    if numeric is None:
        return None
    if numeric < 0.001:
        return f"{prefix}={numeric:.1e}"
    return f"{prefix}={numeric:.3f}"


def _is_common_feature(row: dict[str, object], spec: ResolvedPlotSpec) -> bool:
    if spec.common_feature_column is None:
        return False
    return bool(row.get(spec.common_feature_column))


def _render_group_panel(
    axis: Any,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    *,
    group_value: str,
) -> dict[str, object]:
    selected = _selected_group_rows(rows, spec, group_value=group_value)
    selected = [row for row in selected if coerce_finite_float(row.get(spec.value_column)) is not None]
    if not selected:
        render_placeholder_panel(
            axis,
            panel_title=humanize_display_text(group_value),
            message="No enriched categories",
            detail="No rows pass the configured support filters",
            square=True,
        )
        return {"group": group_value, "rows_rendered": 0}

    display_rows = list(reversed(selected))
    values = [float(coerce_finite_float(row.get(spec.value_column)) or 0.0) for row in display_rows]
    y_positions = list(range(len(display_rows)))
    colors = [
        PUBLICATION_PALETTE[7] if _is_common_feature(row, spec) else PUBLICATION_PALETTE[0] for row in display_rows
    ]
    axis.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
    )
    if spec.reference_line is not None:
        axis.axvline(float(spec.reference_line), color=ZERO_LINE_COLOR, linewidth=1.0, linestyle="--", alpha=0.85)
    axis.set_yticks(y_positions, [_feature_label(row, spec) for row in display_rows])
    axis.tick_params(axis="y", labelsize=8.2, pad=4)
    axis.tick_params(axis="x", labelsize=8.4)
    axis.set_title(wrap_plot_title(humanize_display_text(group_value), width=18, max_lines=2), color=TEXT_COLOR)
    axis.set_xlabel(
        resolved_axis_label(
            explicit_label=spec.x_axis_label,
            fallback_label=humanize_display_text(spec.value_column or "metric value"),
            width=26,
            max_lines=2,
        )
    )
    max_value = max(values) if values else 1.0
    reference_value = float(spec.reference_line) if spec.reference_line is not None else 0.0
    axis.set_xlim(left=0.0, right=max(max_value, reference_value, 1.0) * 1.22)
    for y_position, value, row in zip(y_positions, values, display_rows, strict=True):
        stat = _format_stat(row.get(spec.q_value_column), prefix="q") if spec.q_value_column is not None else None
        if stat is None and spec.p_value_column is not None:
            stat = _format_stat(row.get(spec.p_value_column), prefix="p")
        if stat is not None:
            axis.text(
                value + max(max_value, 1.0) * 0.025,
                y_position,
                stat,
                va="center",
                ha="left",
                fontsize=7.4,
                color=SPINE_COLOR,
            )
    axis.set_box_aspect(1)
    apply_axes_style(axis, grid=True, square=False)
    return {
        "group": group_value,
        "rows_rendered": len(selected),
        "features": [str(row.get(spec.column_column) or "") for row in selected],
    }


def render_categorical_enrichment_summary_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
) -> CategoricalEnrichmentSummaryResult:
    """Render a compact static summary from a categorical enrichment table."""

    if spec.scalar_id is None:
        raise ContractViolationError("categorical_enrichment_summary rendering requires a scalar artifact")
    table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
    if not table_path.exists():
        raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {spec.scalar_id}")
    rows = read_table_rows(
        table_path,
        required_columns=_required_columns(spec),
        artifact_label=f"categorical_enrichment_summary scalar {spec.scalar_id}",
    )
    require_row_columns(rows, _required_columns(spec), context="categorical_enrichment_summary rows")
    filtered_rows = _apply_static_filters(rows, list(spec.static_filters))
    group_values = _ordered_group_values(filtered_rows, spec) if filtered_rows else []
    panel_count = max(len(group_values), 1)
    rows_count, columns = _panel_grid_dimensions(panel_count, prefer_single_row=False)
    panel_size = 4.6
    figure_width = max(5.2, columns * panel_size)
    figure_height = max(5.2, rows_count * panel_size)
    figure, axes = pyplot.subplots(rows_count, columns, figsize=(figure_width, figure_height), squeeze=False)
    panel_metadata: list[dict[str, object]] = []
    if not group_values:
        render_placeholder_panel(
            axes.ravel()[0],
            panel_title=spec.plot_id,
            message="No enrichment rows",
            detail="No rows pass the configured static filters",
            square=True,
        )
    for axis, group_value in zip(axes.ravel(), group_values, strict=False):
        panel_metadata.append(_render_group_panel(axis, filtered_rows, spec, group_value=group_value))
    for axis in axes.ravel()[len(group_values) :]:
        axis.axis("off")
    if spec.common_feature_column is not None and any(_is_common_feature(row, spec) for row in filtered_rows):
        handles = [
            pyplot.Line2D([0], [0], color=PUBLICATION_PALETTE[0], lw=7, label="feature"),
            pyplot.Line2D([0], [0], color=PUBLICATION_PALETTE[7], lw=7, label="common/global feature"),
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8.4,
            bbox_to_anchor=(0.5, -0.005),
        )
    metadata = {
        "source_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "rendered_groups": len(group_values),
        "static_filters": [item.model_dump(mode="json") for item in spec.static_filters],
        "max_features_per_group": spec.max_features_per_group,
        "panels": panel_metadata,
    }
    return CategoricalEnrichmentSummaryResult(figure=figure, metadata=metadata)
