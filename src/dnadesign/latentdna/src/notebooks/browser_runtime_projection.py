"""
Projection rendering helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import math
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..visual_style import (
    NONCANONICAL_SIG35_CATEGORY,
    PUBLICATION_PALETTE,
    TEXT_COLOR,
    compact_candidate_title,
    display_category_text,
    is_sig35_legend_category,
    legend_layout,
    normalize_sig35_hue_category_for_row,
    ordered_categories,
    wrap_plot_title,
)
from ..visual_style import (
    scatter_style as shared_scatter_style,
)
from .browser_runtime_support import (
    available_hues_for_frames,
    candidate_join_keys,
    category_color_map,
    classify_hue_series,
    continuous_hue_render_params,
    display_hue_label,
    draw_reference_labels,
    finite_non_null_hue_series,
    load_table,
    load_view_rows,
    normalize_categorical_hue_series,
    render_matplotlib_figure,
    resolve_join_keys,
    resolve_reference_annotation,
    style_notebook_axes,
    style_notebook_legend,
)


def _error_frame(message: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["load_error"] = message
    return frame


def _frame_load_error(frame: pd.DataFrame) -> str:
    if not isinstance(getattr(frame, "attrs", None), dict):
        return ""
    return str(frame.attrs.get("load_error") or "").strip()


def _frame_artifact_warning(frame: pd.DataFrame) -> str:
    if not isinstance(getattr(frame, "attrs", None), dict):
        return ""
    return str(frame.attrs.get("artifact_warning") or "").strip()


def _render_projection_placeholder(
    ax,
    *,
    panel_title: str,
    message: str,
    detail: str | None = None,
) -> None:
    ax.set_title(panel_title, fontweight="semibold", pad=10 if "\n" in panel_title else 8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    style_notebook_axes(ax, grid=False, square=True)
    ax.text(0.5, 0.58, message, ha="center", va="center", fontsize=11, color=TEXT_COLOR, transform=ax.transAxes)
    if detail:
        ax.text(
            0.5,
            0.41,
            wrap_plot_title(detail, width=28, max_lines=4),
            ha="center",
            va="center",
            fontsize=8.8,
            color="#5C6874",
            transform=ax.transAxes,
        )


def _render_projection_attention_badge(ax) -> None:
    ax.text(
        0.98,
        0.98,
        "Attention",
        ha="right",
        va="top",
        fontsize=7.8,
        color="#8C5A00",
        transform=ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFF1D6",
            "edgecolor": "#D9A441",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )


def _assert_unique_join_key(table: pd.DataFrame, join_key: str, *, artifact_id: str) -> None:
    duplicates = table[join_key][table[join_key].duplicated()].astype(str).tolist()
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate metadata join keys for `{join_key}` in `{artifact_id}`: {preview}")


def _table_view_ids(item: dict[str, object]) -> set[str]:
    return {str(view_id) for view_id in item.get("view_ids", []) if isinstance(view_id, str) and str(view_id).strip()}


def _table_matches_view(item: dict[str, object], view_id: str | None) -> bool:
    if not view_id:
        return True
    explicit_view_ids = _table_view_ids(item)
    if explicit_view_ids:
        return view_id in explicit_view_ids
    artifact_id = str(item.get("artifact_id") or "artifact")
    if artifact_id.startswith("design_centroid_margins_"):
        return artifact_id == f"design_centroid_margins_{view_id}"
    if artifact_id.startswith("context_delta_distribution_"):
        return artifact_id.removeprefix("context_delta_distribution_") in view_id
    return True


def _merge_view_row_columns(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    view_id: str | None,
    required_columns: set[str],
) -> tuple[pd.DataFrame, set[str]]:
    if not view_id or not required_columns:
        return frame, set()
    possible_view_join_columns = pd.DataFrame(columns=["construct__anchor_id", "context_id", "id", "subject_id"])
    view_row_columns = sorted(
        {
            *required_columns,
            *[right_key for _, right_key in candidate_join_keys(frame, possible_view_join_columns)],
        }
    )
    view_rows = load_view_rows(view_id, output_root=output_root, columns=view_row_columns)
    if view_rows.empty:
        return frame, set()
    join_keys = resolve_join_keys(frame, view_rows)
    if join_keys is None:
        return frame, set()
    left_key, right_key = join_keys
    selected_columns = [
        column for column in sorted(required_columns) if column in view_rows.columns and column != right_key
    ]
    if not selected_columns:
        return frame, set()
    _assert_unique_join_key(view_rows, right_key, artifact_id=f"view_rows:{view_id}")
    authoritative = view_rows[[right_key, *selected_columns]].copy()
    stale_columns = [column for column in selected_columns if column in frame.columns and column != left_key]
    merged = frame.drop(columns=stale_columns, errors="ignore")
    if left_key == right_key:
        return merged.merge(authoritative, on=left_key, how="left"), set(selected_columns)
    merged = merged.merge(authoritative, left_on=left_key, right_on=right_key, how="left")
    if right_key in merged.columns:
        merged = merged.drop(columns=[right_key])
    return merged, set(selected_columns)


def _required_column_sources(
    joinable_tables: list[dict[str, object]],
    *,
    requested_columns: set[str],
    view_id: str | None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    sources_by_column: dict[str, list[dict[str, object]]] = {column: [] for column in requested_columns}
    for item in joinable_tables:
        relative_path = item.get("relative_path")
        artifact_id = str(item.get("artifact_id") or "artifact")
        if not isinstance(relative_path, str):
            continue
        if not _table_matches_view(item, view_id):
            continue
        table_columns = {str(column) for column in item.get("columns", []) if isinstance(column, str)}
        for column in requested_columns.intersection(table_columns):
            sources_by_column[column].append(
                {
                    "artifact_id": artifact_id,
                    "relative_path": relative_path,
                }
            )

    resolved: dict[str, dict[str, object]] = {}
    ambiguous_sources: dict[str, list[dict[str, object]]] = {}
    for column, candidates in sources_by_column.items():
        if not candidates:
            continue
        if len(candidates) > 1:
            ambiguous_sources[column] = candidates
            continue
        resolved[column] = candidates[0]
    return resolved, ambiguous_sources


def _raise_ambiguous_required_columns(
    *,
    ambiguous_sources: dict[str, list[dict[str, object]]],
    view_id: str | None,
) -> None:
    for column, candidates in sorted(ambiguous_sources.items()):
        candidate_labels = ", ".join(str(candidate["artifact_id"]) for candidate in candidates)
        raise ValueError(f"ambiguous metadata source for `{column}` on `{view_id or 'projection'}`: {candidate_labels}")


def _merge_required_joinable_column(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    source: dict[str, object],
    view_id: str | None,
    column: str,
) -> pd.DataFrame:
    relative_path = str(source["relative_path"])
    artifact_id = str(source["artifact_id"])
    table = load_table(output_root / relative_path, require_fresh_manifest=True)
    if table.empty:
        raise ValueError(
            f"required metadata source `{artifact_id}` for `{column}` on `{view_id or 'projection'}` is empty"
        )
    join_keys = resolve_join_keys(frame, table)
    if join_keys is None:
        raise ValueError(
            f"required metadata source `{artifact_id}` for `{column}` cannot join onto `{view_id or 'projection'}`"
        )
    left_key, right_key = join_keys
    _assert_unique_join_key(table, right_key, artifact_id=artifact_id)
    merged_source = frame.drop(columns=[column], errors="ignore")
    selected = table[[right_key, *([column] if column != right_key else [])]].copy()
    if left_key == right_key:
        merged = merged_source.merge(selected, on=left_key, how="left")
    else:
        merged = merged_source.merge(selected, left_on=left_key, right_on=right_key, how="left")
        if right_key in merged.columns:
            merged = merged.drop(columns=[right_key])
    if column not in merged.columns:
        raise ValueError(
            "required metadata column "
            f"`{column}` from `{artifact_id}` was not materialized onto `{view_id or 'projection'}`"
        )
    return merged


def _column_has_required_values(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    series = finite_non_null_hue_series(frame, column)
    if series.empty:
        return False
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        series = series.map(lambda value: None if isinstance(value, str) and not value.strip() else value)
    return bool(series.notna().any())


def _categorical_hue_series(frame: pd.DataFrame, hue_column: str) -> pd.Series:
    hue_series = normalize_categorical_hue_series(hue_column, frame[hue_column])
    if hue_column != "sig35_variant":
        return hue_series
    discriminator_columns = [column for column in ("source_class", "source_family") if column in frame.columns]
    if not discriminator_columns:
        return hue_series
    discriminator_frame = frame[discriminator_columns]
    return pd.Series(
        [
            normalize_sig35_hue_category_for_row(row, value)
            for row, value in zip(discriminator_frame.to_dict("records"), frame[hue_column], strict=False)
        ],
        index=frame.index,
        dtype="object",
    )


def _assert_required_columns_materialized(
    frame: pd.DataFrame,
    *,
    required_columns: set[str],
    view_id: str | None,
) -> None:
    missing_columns = [column for column in sorted(required_columns) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"required metadata columns are missing on `{view_id or 'projection'}`: {missing_columns}")
    empty_columns = [column for column in sorted(required_columns) if not _column_has_required_values(frame, column)]
    if empty_columns:
        raise ValueError(f"required metadata columns are empty on `{view_id or 'projection'}`: {empty_columns}")


def enrich_projection_frame(
    frame: pd.DataFrame,
    joinable_tables: list[dict[str, object]],
    *,
    output_root: Path,
    view_id: str | None = None,
    required_columns: list[str] | set[str] | tuple[str, ...] | None = None,
    strict_required_columns: bool = True,
) -> pd.DataFrame:
    requested_columns = {str(column) for column in (required_columns or []) if str(column).strip()}
    enriched, authoritative_view_columns = _merge_view_row_columns(
        frame.copy(),
        output_root=output_root,
        view_id=view_id,
        required_columns=requested_columns,
    )
    effective_requested_columns = set(requested_columns)
    if requested_columns:
        required_sources, ambiguous_sources = _required_column_sources(
            joinable_tables,
            requested_columns=requested_columns.difference(authoritative_view_columns),
            view_id=view_id,
        )
        if ambiguous_sources:
            if strict_required_columns:
                _raise_ambiguous_required_columns(ambiguous_sources=ambiguous_sources, view_id=view_id)
            effective_requested_columns = requested_columns.difference(ambiguous_sources)
        for column in sorted(required_sources):
            enriched = _merge_required_joinable_column(
                enriched,
                output_root=output_root,
                source=required_sources[column],
                view_id=view_id,
                column=column,
            )
        if strict_required_columns:
            _assert_required_columns_materialized(
                enriched,
                required_columns=requested_columns,
                view_id=view_id,
            )
    for item in joinable_tables:
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str):
            continue
        if not _table_matches_view(item, view_id):
            continue
        artifact_id = str(item.get("artifact_id") or "artifact")
        table_columns = [str(column) for column in item.get("columns", []) if isinstance(column, str)]
        if requested_columns:
            needed_columns = effective_requested_columns.difference(enriched.columns)
            selected_columns = [column for column in table_columns if column in needed_columns]
            if not selected_columns:
                continue
        else:
            selected_columns = []
        table = load_table(output_root / relative_path, require_fresh_manifest=True)
        if table.empty:
            continue
        join_keys = resolve_join_keys(enriched, table)
        if join_keys is None:
            continue
        left_key, right_key = join_keys
        _assert_unique_join_key(table, right_key, artifact_id=artifact_id)
        keep_columns = (
            [right_key] + [column for column in selected_columns if column != right_key]
            if requested_columns
            else [column for column in table.columns if column not in {"x", "y"}]
        )
        table = table[keep_columns].copy()
        rename_map = {}
        for column in table.columns:
            if column == right_key:
                continue
            if column == "cluster_label":
                rename_map[column] = f"cluster_label__{artifact_id}"
            elif column in enriched.columns:
                rename_map[column] = f"{column}__{artifact_id}"
        if rename_map:
            table = table.rename(columns=rename_map)
        kept_data_columns = [column for column in table.columns if column != right_key]
        if not kept_data_columns:
            continue
        if left_key == right_key:
            enriched = enriched.merge(table, on=left_key, how="left")
            continue
        enriched = enriched.merge(table, left_on=left_key, right_on=right_key, how="left")
        if right_key in enriched.columns:
            enriched = enriched.drop(columns=[right_key])
    return enriched


def load_projection_frame(
    view_id: str | None,
    projection_id: str,
    joinable_tables: list[dict[str, object]],
    *,
    output_root: Path,
    required_columns: list[str] | set[str] | tuple[str, ...] | None = None,
    strict_required_columns: bool = True,
) -> pd.DataFrame:
    try:
        frame = load_table(
            output_root / "projections" / projection_id / "coords.parquet",
            require_fresh_manifest=True,
            allowed_statuses={"ok", "attention"},
        )
    except ValueError as exc:
        return _error_frame(str(exc))
    if frame.empty:
        return frame
    artifact_warning = _frame_artifact_warning(frame)
    artifact_status = str(frame.attrs.get("artifact_status") or "").strip()
    try:
        enriched = enrich_projection_frame(
            frame,
            joinable_tables,
            output_root=output_root,
            view_id=view_id,
            required_columns=required_columns,
            strict_required_columns=strict_required_columns,
        )
        if artifact_warning:
            enriched.attrs["artifact_warning"] = artifact_warning
        if artifact_status:
            enriched.attrs["artifact_status"] = artifact_status
        return enriched
    except ValueError as exc:
        return _error_frame(str(exc))


def _layout_frames(
    panel_specs: list[dict[str, object]],
    *,
    frames: list[pd.DataFrame] | None,
    joinable_tables: list[dict[str, object]],
    output_root: Path,
) -> list[pd.DataFrame]:
    if frames is not None:
        return list(frames)
    loaded: list[pd.DataFrame] = []
    for spec in panel_specs:
        projection_id = str(spec.get("projection_id") or "")
        view_id = str(spec.get("view_id") or "")
        if not projection_id or not view_id:
            loaded.append(pd.DataFrame())
            continue
        loaded.append(load_projection_frame(view_id, projection_id, joinable_tables, output_root=output_root))
    return loaded


def _panel_grid_dimensions(panel_count: int, *, prefer_single_row: bool = False) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if prefer_single_row and panel_count <= 4:
        return 1, panel_count
    if panel_count in {5, 6}:
        return 2, 3
    if panel_count in {7, 8}:
        return 2, 4
    if panel_count == 4:
        return 2, 2
    columns = min(4, panel_count)
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def render_projection_grid(
    panel_specs: list[dict[str, object]],
    *,
    frames: list[pd.DataFrame] | None = None,
    plot_id: str | None = None,
    hue_column: str | None,
    hue_kinds: dict[str, str] | None,
    joinable_tables: list[dict[str, object]],
    reference_labels: list[str],
    output_root: Path,
    workspace_dir: Path,
    reference_set_id: str | None = None,
    reference_match_column: str = "usr_label__primary",
    reference_display_labels: dict[str, str] | None = None,
    reference_label_limit: int | None = None,
    alt_text: str | None = None,
    prefer_single_row: bool = False,
):
    if not panel_specs:
        return mo.callout("No persisted projection coordinates are available for this geometry layout.", kind="warn")

    resolved_frames = _layout_frames(
        panel_specs,
        frames=frames,
        joinable_tables=joinable_tables,
        output_root=output_root,
    )
    load_errors = [_frame_load_error(frame) for frame in resolved_frames if _frame_load_error(frame)]
    if not any(not frame.empty for frame in resolved_frames) and load_errors:
        unique_errors = list(dict.fromkeys(load_errors))
        return mo.callout("Projection surface is unavailable: " + "; ".join(unique_errors), kind="warn")
    if not any(not frame.empty for frame in resolved_frames):
        return mo.callout(
            "The selected geometry layout is declared, but none of its projections are materialized yet.",
            kind="warn",
        )
    if reference_set_id is not None:
        reference_annotation = resolve_reference_annotation(
            reference_set_id,
            resolved_frames,
            workspace_dir=workspace_dir,
            fallback_labels=reference_labels,
        )
        reference_labels = [str(value) for value in reference_annotation.get("labels", []) if str(value).strip()]
        reference_match_column = str(reference_annotation.get("match_column") or "usr_label__primary")
        reference_display_labels = {
            str(key): str(value)
            for key, value in dict(reference_annotation.get("display_labels", {}) or {}).items()
            if str(key).strip() and str(value).strip()
        }
        resolved_label_limit = reference_annotation.get("label_limit")
        reference_label_limit = resolved_label_limit if isinstance(resolved_label_limit, int) else None

    effective_hue = hue_column
    if effective_hue:
        allowed = available_hues_for_frames(
            resolved_frames,
            preferred_hues=[effective_hue],
            hue_kinds=hue_kinds or {},
        )
        if effective_hue not in allowed:
            effective_hue = None

    n_panels = len(panel_specs)
    nrows, ncols = _panel_grid_dimensions(n_panels, prefer_single_row=prefer_single_row)
    if n_panels == 1:
        panel_width = 6.1
        panel_height = 6.2
    elif prefer_single_row and n_panels <= 4:
        panel_width = 3.55
        panel_height = 4.0
    else:
        panel_width = 3.65
        panel_height = 3.81
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=((panel_width * ncols) + 0.45, panel_height * nrows),
    )
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols).ravel()
    panel_axes = axes_array[:n_panels]

    hue_kind = None
    if effective_hue is not None:
        configured_hue_kind = (hue_kinds or {}).get(effective_hue)
        hue_kind = classify_hue_series(
            pd.concat(
                [frame[effective_hue] for frame in resolved_frames if effective_hue in frame.columns],
                ignore_index=True,
            ),
            configured_kind=configured_hue_kind,
        )
    treat_as_categorical = hue_kind in {"categorical", "binary", "ordinal"}

    numeric_frames = [
        pd.to_numeric(frame[effective_hue], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in resolved_frames
        if effective_hue is not None and hue_kind == "continuous" and effective_hue in frame.columns
    ]
    continuous_params = {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}
    if numeric_frames:
        combined_numeric = pd.concat(numeric_frames, ignore_index=True)
        if combined_numeric.nunique() >= 2:
            continuous_params = continuous_hue_render_params(effective_hue, combined_numeric)
        else:
            effective_hue = None
            hue_kind = None

    category_values = ordered_categories(
        {
            str(value)
            for frame in resolved_frames
            if effective_hue is not None and treat_as_categorical and effective_hue in frame.columns
            for value in _categorical_hue_series(frame, effective_hue).unique()
        },
        column=effective_hue,
    )
    if effective_hue == "sig35_variant":
        category_values = [category for category in category_values if is_sig35_legend_category(category)]
    category_map = category_color_map(category_values, column=effective_hue)

    scatter_artist = None
    max_title_lines = 1
    for axis_index, (ax, spec, frame) in enumerate(zip(panel_axes, panel_specs, resolved_frames, strict=True)):
        wrapped_title = wrap_plot_title(
            compact_candidate_title(str(spec.get("title") or spec.get("view_id") or f"Panel {axis_index + 1}")),
            width=32 if n_panels == 1 else 22,
            max_lines=3,
        )
        max_title_lines = max(max_title_lines, wrapped_title.count("\n") + 1)
        load_error = _frame_load_error(frame)
        artifact_warning = _frame_artifact_warning(frame)
        if load_error:
            _render_projection_placeholder(
                ax,
                panel_title=wrapped_title,
                message="Projection unavailable",
                detail=load_error,
            )
            continue
        if frame.empty or "x" not in frame.columns or "y" not in frame.columns:
            _render_projection_placeholder(ax, panel_title=wrapped_title, message="Projection missing")
            continue

        point_style = shared_scatter_style(len(frame))
        if effective_hue is None or effective_hue not in frame.columns:
            ax.scatter(
                frame["x"].to_numpy(dtype=float),
                frame["y"].to_numpy(dtype=float),
                c=PUBLICATION_PALETTE[0],
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        elif hue_kind == "continuous":
            hue_series = pd.to_numeric(frame[effective_hue], errors="coerce")
            valid = hue_series.notna()
            scatter_artist = ax.scatter(
                frame.loc[valid, "x"].to_numpy(dtype=float),
                frame.loc[valid, "y"].to_numpy(dtype=float),
                c=hue_series.loc[valid].to_numpy(dtype=float),
                cmap=str(continuous_params["cmap"]),
                norm=continuous_params["norm"],
                vmin=None if continuous_params["norm"] is not None else continuous_params["vmin"],
                vmax=None if continuous_params["norm"] is not None else continuous_params["vmax"],
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        else:
            hue_series = _categorical_hue_series(frame, effective_hue)
            plotted_mask = pd.Series(False, index=frame.index)
            for category in category_values:
                mask = hue_series == category
                if not mask.any():
                    continue
                plotted_mask |= mask
                ax.scatter(
                    frame.loc[mask, "x"].to_numpy(dtype=float),
                    frame.loc[mask, "y"].to_numpy(dtype=float),
                    c=category_map[category],
                    s=point_style.point_size,
                    alpha=point_style.alpha,
                    linewidths=point_style.linewidths,
                    edgecolors=point_style.edgecolors,
                    rasterized=point_style.rasterized,
                    label=category,
                )
            if effective_hue == "sig35_variant":
                noncanonical_mask = (~plotted_mask) & (hue_series == NONCANONICAL_SIG35_CATEGORY)
                if noncanonical_mask.any():
                    ax.scatter(
                        frame.loc[noncanonical_mask, "x"].to_numpy(dtype=float),
                        frame.loc[noncanonical_mask, "y"].to_numpy(dtype=float),
                        c="#9AA5B1",
                        s=point_style.point_size,
                        alpha=max(point_style.alpha * 0.55, 0.08),
                        linewidths=point_style.linewidths,
                        edgecolors=point_style.edgecolors,
                        rasterized=point_style.rasterized,
                    )

        ax.set_title(wrapped_title, fontweight="semibold", pad=10 if "\n" in wrapped_title else 8)
        if artifact_warning:
            _render_projection_attention_badge(ax)
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
        ax.margins(x=0.06, y=0.06)
        style_notebook_axes(ax, grid=True, square=True)

    for ax in axes_array[n_panels:]:
        ax.set_axis_off()

    bottom_margin = 0.085
    if category_values and effective_hue is not None:
        legend_labels = [display_category_text(category, column=effective_hue) for category in category_values]
        layout = legend_layout(
            legend_labels,
            plot_id=plot_id,
            default_anchor_y=0.006 if plot_id == "appendix_umap_gallery" else (0.012 if n_panels <= 1 else 0.008),
            default_base_margin=0.11,
            row_step=0.043,
        )
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=category_map[category],
                    label=label,
                )
                for category, label in zip(category_values, legend_labels, strict=True)
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            frameon=False,
            ncol=layout.columns,
            borderaxespad=0.0,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        style_notebook_legend(legend)
        bottom_margin = layout.bottom_margin
        if plot_id == "appendix_umap_gallery":
            bottom_margin = max(bottom_margin, 0.12)
        if n_panels == 2:
            bottom_margin = max(bottom_margin, 0.34)
        elif n_panels >= 8:
            bottom_margin = max(bottom_margin, 0.30)

    label_right_padding_px = 12.0
    colorbar_bottom = 0.0
    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        colorbar_bottom = 0.12
        label_right_padding_px = 28.0
    top_margin = max(0.8, 0.96 - (0.042 * max(max_title_lines - 1, 0)))
    if colorbar_bottom > 0.0:
        bottom_margin = max(bottom_margin, 0.2 if n_panels > 1 else 0.16)
    fig.subplots_adjust(
        left=0.1,
        right=0.97,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.26 if n_panels > 1 else 0.2,
        hspace=(0.62 + (0.04 * max(max_title_lines - 1, 0))) if n_panels > 1 else 0.3,
    )
    fig.canvas.draw()

    for ax, frame in zip(panel_axes, resolved_frames, strict=True):
        if ax.axison and not frame.empty:
            draw_reference_labels(
                ax,
                frame,
                reference_labels=reference_labels,
                reference_match_column=reference_match_column,
                reference_display_labels=reference_display_labels,
                reference_label_limit=reference_label_limit,
                right_padding_px=label_right_padding_px,
                left_padding_px=28.0,
            )

    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        colorbar_width = 0.66 if n_panels > 1 else 0.56
        colorbar_left = (1.0 - colorbar_width) / 2.0
        colorbar_height = 0.028
        colorbar = fig.colorbar(
            scatter_artist,
            cax=fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]),
            orientation="horizontal",
        )
        colorbar.ax.tick_params(labelsize=10.5, colors=TEXT_COLOR)
        colorbar.set_label(display_hue_label(effective_hue), fontsize=11.5, color=TEXT_COLOR)
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.xaxis.set_ticks_position("bottom")

    return render_matplotlib_figure(fig, alt=str(alt_text or "Latent geometry projection grid"))
