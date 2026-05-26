from __future__ import annotations

import io
from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from ._support import display_name, mapping
from .campaign_set_intervals import aggregate_center, interval_sentence
from .campaign_set_relationships import campaign_pair_contexts, metadata_fields, relationship_pair_membership
from .campaign_set_sources import campaign_plot_manifest, finite_number, manifest_tidy_csv_path, read_csv_dict_rows
from .campaign_set_vector_heatmap_stats import (
    common_text,
    heatmap_domain,
    heatmap_values,
    mse_formula,
    ordered_unique,
    plot_question,
    reference_mse_axis_fields,
    reference_vector,
    role_sort_key,
    sentence_text,
    target_mse_values,
    target_vector_label,
    vector_axis_scale,
    wrap_no_ellipsis,
)


def build_notebook_campaign_set_vector_heatmap_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    plot_name: str,
    group_key: str,
    relationship: Mapping[str, Any] | None = None,
    cohort: str = "selected",
) -> list[dict[str, Any]]:
    """Read selected-vector heatmap rows from vector_summary_heatmap tidy CSVs."""

    rows: list[dict[str, Any]] = []
    pair_membership = relationship_pair_membership(relationship)
    for campaign_model in campaigns:
        campaign = mapping(campaign_model.get("campaign"))
        slug = str(campaign.get("slug") or "unknown")
        pair_contexts = campaign_pair_contexts(campaign_model, pair_membership) if pair_membership else [None]
        if not pair_contexts:
            continue
        metadata = mapping(campaign.get("metadata"))
        manifest = campaign_plot_manifest(campaign_model, name=plot_name, kind="vector_summary_heatmap")
        if manifest is None:
            continue
        params = mapping(manifest.get("params"))
        tidy_path = manifest_tidy_csv_path(manifest)
        if tidy_path is None or not tidy_path.exists():
            continue
        group_value = str(metadata.get(str(group_key), "not recorded"))
        for raw in read_csv_dict_rows(tidy_path):
            row_type = str(raw.get("row_type") or "").strip()
            if row_type not in {"round", "reference_vector"}:
                continue
            if row_type == "round" and str(raw.get("cohort") or "") != str(cohort):
                continue
            value = finite_number(raw.get("value"))
            if value is None:
                continue
            round_value = finite_number(raw.get("round"))
            channel = str(raw.get("channel") or "").strip()
            if not channel:
                continue
            for pair_context in pair_contexts:
                rows.append(
                    {
                        **metadata_fields(metadata),
                        **(pair_context or {}),
                        "row_type": row_type,
                        "round": int(round_value) if round_value is not None else "",
                        "cohort": str(raw.get("cohort") or cohort),
                        "channel": channel,
                        "metric": "selected_predicted_vector",
                        "summary": "mean",
                        "value": float(value),
                        "cohort_count": finite_number(raw.get("n")),
                        "campaign": slug,
                        "campaign_label": display_name(slug),
                        "group_key": group_key,
                        "group": group_value,
                        "tidy_csv": str(tidy_path),
                        "metric_label": str(params.get("value_label") or "Mean predicted vector value"),
                        "legend_metric_label": str(params.get("reference_mse_legend_label") or "target-vector MSE"),
                        "mse_axis_label": str(
                            params.get("reference_mse_metric_label") or "MSE = mean((mean selected y_hat - target)^2)"
                        ),
                        "font_size": finite_number(params.get("font_size")),
                        "metric_expression": str(
                            params.get("reference_mse_expression")
                            or "MSE = mean((mean selected y_hat - reference)^2); lower is better"
                        ),
                        **reference_mse_axis_fields(params),
                    }
                )
    return rows


def render_notebook_campaign_set_vector_heatmap_comparison_image(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str,
    group_key: str,
    interval_kind: str = "iqr",
    interpretation_note: str = "",
    dpi: int = 180,
) -> dict[str, Any] | None:
    """Render side-by-side group heatmaps plus target-vector MSE trajectories."""

    data = [row for row in rows if isinstance(row, Mapping)]
    round_rows = [
        row
        for row in data
        if str(row.get("row_type") or "") == "round"
        and finite_number(row.get("round")) is not None
        and finite_number(row.get("value")) is not None
    ]
    reference_rows = [
        row
        for row in data
        if str(row.get("row_type") or "") == "reference_vector" and finite_number(row.get("value")) is not None
    ]
    if not round_rows or not reference_rows:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    from ...plots._mpl_utils import (
        apply_notebook_axes_style,
        apply_plot_style,
        apply_y_axis_scale,
        categorical_color,
        categorical_linestyle,
        categorical_marker,
        sequential_colormap,
    )

    requested_interval_kind = str(interval_kind or "none").strip()
    if requested_interval_kind not in {"none", "iqr"}:
        raise ValueError(f"Unsupported vector heatmap interval_kind: {requested_interval_kind!r}.")
    channels = ordered_unique(str(row.get("channel") or "") for row in round_rows)
    rounds = sorted({int(finite_number(row.get("round"))) for row in round_rows})
    groups = sorted({str(row.get("group") or "not recorded") for row in round_rows}, key=role_sort_key)
    if not channels or not rounds or not groups:
        return None
    reference = reference_vector(reference_rows, channels=channels)
    if set(reference) != set(channels):
        return None

    heat_values = heatmap_values(round_rows, groups=groups, rounds=rounds, channels=channels)
    mse_values = target_mse_values(round_rows, reference=reference, channels=channels)
    if not mse_values:
        return None
    axis_scale = vector_axis_scale(data)
    metric_label = common_text(data, "legend_metric_label") or "target-vector MSE"
    metric_expression = common_text(data, "metric_expression") or "MSE = mean_c((mean selected y_hat_c - target_c)^2)"
    mse_formula_text = mse_formula(metric_expression)
    target_vector_label_text = target_vector_label(reference, channels=channels)
    plot_question_text = plot_question(data, target_vector_label=target_vector_label_text)
    vmin, vmax = heatmap_domain(round_rows, reference.values())
    font_size = _common_float(data, "font_size") or 13.0
    title_font_size = font_size
    tick_font_size = font_size
    mse_axis_label = common_text(data, "mse_axis_label") or mse_formula_text

    apply_plot_style()
    heatmap_count = len(groups)
    fig_width = max(10.8, 1.6 * heatmap_count + 7.0)
    fig = plt.figure(figsize=(fig_width, 4.9))
    gs = fig.add_gridspec(
        1,
        heatmap_count + 3,
        width_ratios=[1.0] * heatmap_count + [0.06, 0.46, 2.15],
        wspace=0.08,
    )
    heat_axes = [fig.add_subplot(gs[0, index]) for index in range(heatmap_count)]
    cax = fig.add_subplot(gs[0, heatmap_count])
    spacer_ax = fig.add_subplot(gs[0, heatmap_count + 1])
    spacer_ax.axis("off")
    ax_mse = fig.add_subplot(gs[0, heatmap_count + 2])
    image = None
    for index, (axis, group) in enumerate(zip(heat_axes, groups, strict=True)):
        matrix = np.asarray(
            [
                [heat_values.get((group, round_index, channel), np.nan) for channel in channels]
                for round_index in rounds
            ],
            dtype=float,
        )
        image = axis.pcolormesh(
            np.arange(len(channels) + 1),
            np.arange(len(rounds) + 1),
            np.ma.masked_invalid(matrix),
            vmin=vmin,
            vmax=vmax,
            cmap=sequential_colormap("opal_seafoam"),
            edgecolors="white",
            linewidth=0.8,
            shading="flat",
        )
        axis.set_xlim(0, len(channels))
        axis.set_ylim(0, len(rounds))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{pretty_label(group)} oracle", fontsize=title_font_size)
        axis.set_xticks(
            np.arange(len(channels)) + 0.5,
            labels=_heatmap_channel_tick_labels(channels),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(np.arange(len(rounds)) + 0.5, labels=[str(round_index) for round_index in rounds])
        if index == 0:
            axis.set_ylabel("Round", fontsize=font_size)
        else:
            axis.set_yticklabels([])
        apply_notebook_axes_style(axis, grid=False, square=False)
        axis.tick_params(axis="both", length=0, labelsize=tick_font_size)
        for tick_label in axis.get_xticklabels():
            tick_label.set_horizontalalignment("right")
    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.set_title(r"Mean $\hat{y}$", fontsize=font_size, pad=8)
        cbar.ax.tick_params(labelsize=tick_font_size)

    rounds_with_interval = 0
    interval_unit_counts: list[int] = []
    for index, group in enumerate(groups):
        xs = []
        ys = []
        lows = []
        highs = []
        for round_index in rounds:
            values = mse_values.get((group, round_index), [])
            if not values:
                continue
            xs.append(round_index)
            ys.append(aggregate_center(values, center="mean"))
            if requested_interval_kind == "iqr" and len(values) >= 2:
                lows.append(float(np.quantile(values, 0.25)))
                highs.append(float(np.quantile(values, 0.75)))
                rounds_with_interval += 1
                interval_unit_counts.append(len(values))
            else:
                lows.append(float("nan"))
                highs.append(float("nan"))
        color = categorical_color(index)
        mask = np.isfinite(lows) & np.isfinite(highs)
        if bool(np.any(mask)):
            x_arr = np.asarray(xs, dtype=float)
            ax_mse.fill_between(
                x_arr[mask],
                np.asarray(lows, dtype=float)[mask],
                np.asarray(highs, dtype=float)[mask],
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        ax_mse.plot(
            xs,
            ys,
            color=color,
            marker=categorical_marker(index),
            linestyle=categorical_linestyle(index),
            linewidth=2.2,
            markersize=6,
            label=pretty_label(group),
            zorder=2,
        )
    apply_notebook_axes_style(ax_mse, square=True)
    ax_mse.set_xlabel("Round", fontsize=font_size)
    ax_mse.set_ylabel(mse_axis_label, fontsize=font_size, labelpad=18)
    ax_mse.set_title("Target-vector loss", fontsize=title_font_size)
    ax_mse.set_xticks(rounds)
    ax_mse.tick_params(axis="both", labelsize=tick_font_size)
    apply_y_axis_scale(
        ax_mse,
        limits=axis_scale.get("limits"),
        reference_lines=axis_scale.get("reference_lines"),
        include_zero_tick=bool(axis_scale.get("include_zero_tick")),
    )
    ax_mse.legend(loc="upper right", frameon=False, fontsize=font_size)
    fig.suptitle(
        wrap_no_ellipsis(plot_question_text, width=78),
        fontsize=title_font_size,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.82)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), facecolor="white")
    plt.close(fig)
    interval = {
        "kind": requested_interval_kind,
        "unit": "relationship pairs",
        "rounds_with_interval": rounds_with_interval,
        "min_unit_count": min(interval_unit_counts) if interval_unit_counts else 0,
        "max_unit_count": max(interval_unit_counts) if interval_unit_counts else 0,
        "is_confidence_interval": False,
    }
    interval_text = interval_sentence(
        interval_kind=requested_interval_kind,
        interval_unit="relationship pairs",
        rounds_with_interval=rounds_with_interval,
        confidence_level=0.95,
    )
    interpretation_text = sentence_text(interpretation_note)
    group_text = ", ".join(pretty_label(group) for group in groups)
    caption = (
        f"{plot_question_text}. Heatmaps show selected mean predicted vector values by round for each oracle role "
        f"using unit-square cells and one shared color scale. The right panel shows target-vector MSE across "
        f"matched positive/null pairs; {mse_formula_text}. Target: {target_vector_label_text}."
        f"{interval_text}{interpretation_text}"
    )
    alt_text = (
        f"{plot_question_text}. Heatmaps for {group_text} oracle roles share one color scale; the line panel "
        f"overlays target-vector MSE for the same roles. Target: {target_vector_label_text}."
        f"{interval_text}{interpretation_text}"
    )
    return {
        "image_bytes": buffer.getvalue(),
        "alt_text": alt_text,
        "caption": caption,
        "label": common_text(data, "collection_visual_label"),
        "title": plot_question_text,
        "plot_question": plot_question_text,
        "target_vector_label": target_vector_label_text,
        "mse_formula": mse_formula_text,
        "visual_contract": {
            "heatmap_cell_geometry": "unit_square_cells",
            "heatmap_cell_edges": "white_cell_edges_only",
            "heatmap_background_grid": "off",
            "mse_panel": "shared_axis_group_lines",
        },
        "metric_label": metric_label,
        "legend_metric_label": metric_label,
        "metric_expression": metric_expression,
        "axis_scale": axis_scale,
        "group_count": len(groups),
        "row_count": len(data),
        "interval": interval,
    }


def _common_float(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = {float(value) for row in rows if (value := finite_number(row.get(key))) is not None}
    return sorted(values)[0] if len(values) == 1 else None


def _heatmap_channel_tick_labels(labels: Iterable[str]) -> list[str]:
    return [str(label) for label in labels]
