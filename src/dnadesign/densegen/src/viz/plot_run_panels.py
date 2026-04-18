"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run_panels.py

TFBS usage and supplemental run-health plotting panels used by run diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_common import (
    _apply_style,
    _palette,
    _save_figure,
    _stage_b_plan_output_dir,
    _style,
    plan_group_from_name,
)
from .plot_run_helpers import (
    _ellipsize,
    _first_existing_column,
    _humanize_scope_label,
    _normalize_plan_name,
    _usage_available_unique,
    _usage_category_label,
    compact_plan_label,
    compact_regulator_label,
    order_regulators_for_display,
)


def _capitalize_first(text: str) -> str:
    token = str(text)
    for idx, char in enumerate(token):
        if char.isalpha():
            return token[:idx] + char.upper() + token[idx + 1 :]
    return token


def _stable_seed(token: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(token))) % (2**32)


def _stable_subsample(values: np.ndarray, *, token: str, max_points: int | None) -> np.ndarray:
    if max_points is None or int(max_points) <= 0 or values.size <= int(max_points):
        return values
    rng = np.random.default_rng(_stable_seed(token))
    indices = np.sort(rng.choice(values.size, size=int(max_points), replace=False))
    return values[indices]


def _mix_with_white(color: object, *, fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    weight = min(1.0, max(0.0, float(fraction)))
    mixed = rgb + (1.0 - rgb) * weight
    return tuple(float(value) for value in mixed)


def _progressive_regulator_shades(base_color: object, n_values: int) -> list[tuple[float, float, float]]:
    if int(n_values) <= 1:
        return [_mix_with_white(base_color, fraction=0.34)]
    fractions = np.linspace(0.62, 0.18, int(n_values))
    return [_mix_with_white(base_color, fraction=float(value)) for value in fractions.tolist()]


def _style_deemphasized_boxplot(boxplot: dict[str, list], *, colors: list[object]) -> None:
    for box, color in zip(boxplot.get("boxes", []), colors, strict=False):
        box.set(
            facecolor=_mix_with_white(color, fraction=0.68),
            edgecolor="#8d98a5",
            linewidth=0.95,
            alpha=0.64,
            zorder=1.0,
        )
    for median in boxplot.get("medians", []):
        median.set(color="#4a5564", linewidth=1.25, zorder=1.4)
    for whisker, color in zip(boxplot.get("whiskers", []), np.repeat(colors, 2), strict=False):
        whisker.set(color=_mix_with_white(color, fraction=0.5), linewidth=0.95, alpha=0.76, zorder=1.2)
    for cap, color in zip(boxplot.get("caps", []), np.repeat(colors, 2), strict=False):
        cap.set(color=_mix_with_white(color, fraction=0.5), linewidth=0.95, alpha=0.76, zorder=1.2)
    for flier in boxplot.get("fliers", []):
        flier.set(alpha=0.0, markersize=0.0)


def _draw_category_box_and_jitter(
    ax: plt.Axes,
    *,
    categories: list[str],
    values_by_category: dict[str, np.ndarray],
    colors_by_category: dict[str, object],
    positions: np.ndarray,
    orientation: str,
    jitter_width: float,
    max_points_per_category: int | None = None,
    rasterized: bool = False,
) -> None:
    non_empty_categories = [category for category in categories if values_by_category[category].size > 0]
    if not non_empty_categories:
        raise ValueError("box-and-jitter plot requires at least one non-empty category.")
    boxplot = ax.boxplot(
        [values_by_category[category].tolist() for category in non_empty_categories],
        positions=[float(positions[categories.index(category)]) for category in non_empty_categories],
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        orientation=orientation,
    )
    _style_deemphasized_boxplot(
        boxplot,
        colors=[colors_by_category[category] for category in non_empty_categories],
    )
    for category, position in zip(categories, positions.tolist(), strict=False):
        values = _stable_subsample(
            values_by_category[category],
            token=category,
            max_points=max_points_per_category,
        )
        if values.size == 0:
            continue
        jitter = np.random.default_rng(_stable_seed(category)).uniform(
            -float(jitter_width),
            float(jitter_width),
            size=int(values.size),
        )
        color = _mix_with_white(colors_by_category[category], fraction=0.35)
        if orientation == "vertical":
            ax.scatter(
                np.full(values.size, float(position)) + jitter,
                values,
                s=22.0,
                color=color,
                alpha=0.58,
                edgecolors="white",
                linewidths=0.28,
                zorder=3.0,
                rasterized=rasterized,
            )
        else:
            ax.scatter(
                values,
                np.full(values.size, float(position)) + jitter,
                s=22.0,
                color=color,
                alpha=0.58,
                edgecolors="white",
                linewidths=0.28,
                zorder=3.0,
                rasterized=rasterized,
            )


def _build_tfbs_usage_breakdown_figure(
    composition_df: pd.DataFrame,
    *,
    input_name: str,
    plan_name: str,
    style: Optional[dict] = None,
    pools: dict[str, pd.DataFrame] | None = None,
    library_members_df: pd.DataFrame | None = None,
    plot_label: str = "tfbs_concentration_profile",
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    style = _style(style)
    sub = composition_df[
        (composition_df["input_name"].astype(str) == str(input_name))
        & (composition_df["plan_name"].astype(str) == str(plan_name))
    ].copy()
    if sub.empty:
        raise ValueError(f"{plot_label} found no placements for {input_name}/{plan_name}.")
    sub["category_label"] = sub["tf"].map(_usage_category_label)
    sub = sub[sub["category_label"].astype(str).str.strip() != ""].copy()
    if sub.empty:
        raise ValueError(f"{plot_label} found no regulator TFBS counts for {input_name}/{plan_name}.")
    sub["tfbs"] = sub["tfbs"].astype(str)
    counts = (
        sub.groupby(["category_label", "tfbs"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "category_label", "tfbs"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    if counts.empty:
        raise ValueError(f"{plot_label} found no TFBS counts for {input_name}/{plan_name}.")

    total = float(counts["count"].sum())
    counts = counts.copy()
    counts["global_rank"] = np.arange(1, len(counts) + 1, dtype=int)
    all_values = counts["count"].astype(float).to_numpy()
    available_by_category, available_total = _usage_available_unique(
        input_name=input_name,
        plan_name=plan_name,
        pools=pools,
        library_members_df=library_members_df,
    )
    category_totals = counts.groupby("category_label")["count"].sum().sort_values(ascending=False)
    category_order = category_totals.index.astype(str).tolist()
    category_unique_used = counts.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_used"})
    top10 = all_values[: min(10, len(all_values))].sum() / total if total > 0 else 0.0
    top50 = all_values[: min(50, len(all_values))].sum() / total if total > 0 else 0.0

    fig_size = style.get("tfbs_usage_breakdown_figsize")
    if fig_size is None:
        fig_size = (10.8, 5.8)
    fig, (ax_usage, ax_cum) = plt.subplots(1, 2, figsize=(float(fig_size[0]), float(fig_size[1])), sharex=False)
    palette = _palette(style, max(1, len(category_order) + 1))
    category_colors = {label: palette[idx + 1] for idx, label in enumerate(category_order)}
    ax_usage.plot(
        counts["global_rank"].astype(float).to_numpy(),
        counts["count"].astype(float).to_numpy(),
        color=palette[0],
        linewidth=1.3,
        alpha=0.86,
        zorder=2,
    )
    for label in category_order:
        cat_points = counts[counts["category_label"] == label].sort_values(by=["global_rank"], ascending=[True])
        if cat_points.empty:
            continue
        x_vals = cat_points["global_rank"].astype(float).to_numpy()
        y_vals = cat_points["count"].astype(float).to_numpy()
        color = category_colors[label]
        ax_usage.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
    ax_usage.set_ylabel("Usage count")
    ax_usage.set_xlabel("Global TFBS rank (descending count)")
    input_label = _humanize_scope_label(input_name) or str(input_name)
    plan_label = _humanize_scope_label(plan_name) or str(plan_name)
    if input_label == plan_label:
        scope_label = plan_label
    else:
        scope_label = f"{plan_label} / {input_label}"
    ax_usage.set_title(f"TFBS usage distribution for {scope_label}")
    rank_share_rows: list[tuple[str, np.ndarray]] = []
    max_rank_within_regulator = 1
    max_rank_share = 0.0
    for label in category_order:
        cat_points = counts[counts["category_label"] == label].sort_values(
            by=["count", "tfbs"],
            ascending=[False, True],
        )
        if cat_points.empty:
            continue
        cat_values = cat_points["count"].astype(float).to_numpy()
        cat_total = float(cat_values.sum())
        cat_share = cat_values / cat_total if cat_total > 0 else np.zeros_like(cat_values)
        max_rank_within_regulator = max(max_rank_within_regulator, int(cat_share.shape[0]))
        if cat_share.size > 0:
            max_rank_share = max(max_rank_share, float(np.nanmax(cat_share)))
        rank_share_rows.append((label, cat_share))
    rank_heatmap = np.full((len(category_order), max_rank_within_regulator), np.nan, dtype=float)
    for row_idx, label in enumerate(category_order):
        share_values = next((shares for category, shares in rank_share_rows if category == label), np.array([]))
        if share_values.size > 0:
            rank_heatmap[row_idx, : share_values.size] = share_values
    vmax = max(0.01, min(1.0, max_rank_share if np.isfinite(max_rank_share) else 1.0))
    heatmap_image = ax_cum.imshow(
        rank_heatmap,
        cmap="magma",
        interpolation="nearest",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    if max_rank_within_regulator <= 10:
        tick_positions = np.arange(max_rank_within_regulator, dtype=float)
    else:
        tick_step = max(1, int(np.ceil(float(max_rank_within_regulator) / 8.0)))
        tick_positions = np.arange(0, max_rank_within_regulator, tick_step, dtype=float)
        if (max_rank_within_regulator - 1) not in tick_positions:
            tick_positions = np.append(tick_positions, float(max_rank_within_regulator - 1))
    ax_cum.set_xticks(tick_positions)
    ax_cum.set_xticklabels([str(int(pos) + 1) for pos in tick_positions.tolist()])
    ax_cum.set_yticks(np.arange(len(category_order), dtype=float))
    ax_cum.set_yticklabels([_capitalize_first(_ellipsize(label, max_len=16)) for label in category_order])
    ax_cum.set_xlabel("TFBS rank within regulator")
    ax_cum.set_ylabel("")
    ax_cum.set_title("Rank-share heatmap within regulator", pad=8.0)
    ax_usage.set_box_aspect(1.0)
    ax_cum.set_box_aspect(1.0)
    colorbar = fig.colorbar(
        heatmap_image,
        ax=ax_cum,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Share within regulator")
    colorbar.ax.tick_params(
        labelsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.72)),
    )

    if all_values.size > 0:
        y_max = float(np.nanmax(all_values)) * 1.08
        if y_max <= 0:
            y_max = 1.0
        ax_usage.set_ylim(0.0, y_max)

    summary_lines = [
        f"Placements in outputs: {int(total)}",
        f"Unique TFBS-pairs in outputs: {len(counts)}",
        f"Top10 share (specific TFBS rank): {top10:.1%}",
        f"Top50 share (specific TFBS rank): {top50:.1%}",
    ]
    if available_total > 0:
        summary_lines.append(
            f"Unique TFBS-pairs used / available: {len(counts)}/{available_total} ({len(counts) / available_total:.1%})"
        )
    summary_lines = [_capitalize_first(line) for line in summary_lines]
    summary_font_size = float(
        style.get(
            "tfbs_usage_summary_size",
            max(
                10.8,
                float(style.get("label_size", style.get("font_size", 13.0))) * 0.86,
            ),
        )
    )
    ax_usage.text(
        0.98,
        0.95,
        "\n".join(summary_lines),
        transform=ax_usage.transAxes,
        ha="right",
        va="top",
        fontsize=summary_font_size,
    )
    ax_usage.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    _apply_style(ax_usage, style)
    _apply_style(ax_cum, style)
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for label in category_order:
        cat_total = int(category_totals.loc[label])
        share = (float(cat_total) / total) if total > 0 else 0.0
        available_unique = int(available_by_category.get(label, 0))
        used_unique = int(category_unique_used.loc[label, "unique_used"] if label in category_unique_used.index else 0)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                markersize=6.0,
                color=category_colors[label],
            )
        )
        legend_labels.append(
            f"{_capitalize_first(label)}: placements {cat_total}/{int(total)} ({share:.1%}), "
            f"unique {used_unique}/{max(1, available_unique)}"
        )
    if legend_handles:
        legend_font_size = float(
            style.get(
                "tfbs_usage_legend_size",
                max(
                    float(style.get("label_size", style.get("font_size", 13.0))),
                    float(style.get("font_size", 13.0) * 0.95),
                ),
            )
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
            fontsize=legend_font_size,
        )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    return fig, {"usage": ax_usage, "cum": ax_cum}


def plot_tfbs_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    pools: dict[str, pd.DataFrame] | None = None,
    library_members_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
    plan_col: str = "plan_name",
    input_col: str = "input_name",
    plot_label: str = "tfbs_concentration_profile",
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError(f"{plot_label} requires composition.parquet with placements.")
    plan_col = str(plan_col or "").strip() or "plan_name"
    input_col = str(input_col or "").strip() or "input_name"
    required = {input_col, plan_col, "tf", "tfbs"}
    missing = required - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    normalized = composition_df.copy()
    if input_col != "input_name":
        normalized = normalized.rename(columns={input_col: "input_name"})
    if plan_col != "plan_name":
        normalized = normalized.rename(columns={plan_col: "plan_name"})
    paths: list[Path] = []
    for input_name, plan_name in normalized[["input_name", "plan_name"]].drop_duplicates().itertuples(index=False):
        fig, _axes = _build_tfbs_usage_breakdown_figure(
            normalized,
            input_name=str(input_name),
            plan_name=str(plan_name),
            style=style,
            pools=pools,
            library_members_df=library_members_df,
            plot_label=plot_label,
        )
        target_dir = _stage_b_plan_output_dir(out_path, input_name=str(input_name), plan_name=str(plan_name))
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"tfbs_usage{out_path.suffix}"
        _save_figure(fig, path, style=style)
        plt.close(fig)
        paths.append(path)
    return paths


def _build_run_health_compression_ratio_figure(
    dense_arrays_df: pd.DataFrame,
    *,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    if dense_arrays_df is None or dense_arrays_df.empty:
        raise ValueError("run_health compression_ratio_distribution requires dense-array outputs.")
    style = _style(style)
    ratio_col = _first_existing_column(
        dense_arrays_df,
        ["densegen__compression_ratio", "compression_ratio"],
        context="run_health compression_ratio_distribution",
    )
    plan_col = _first_existing_column(
        dense_arrays_df,
        ["densegen__plan", "plan_name"],
        context="run_health compression_ratio_distribution",
    )
    frame = dense_arrays_df[[ratio_col, plan_col]].copy()
    frame[ratio_col] = pd.to_numeric(frame[ratio_col], errors="coerce")
    plan_values = frame[plan_col].astype("object")
    normalized_plan_lookup = {
        raw_value: normalized_value
        for raw_value in pd.unique(plan_values)
        if (normalized_value := _normalize_plan_name(raw_value)) is not None
    }
    frame[plan_col] = plan_values.map(normalized_plan_lookup).fillna("all plans")
    frame = frame.dropna(subset=[ratio_col]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("run_health compression_ratio_distribution found no numeric compression_ratio values.")
    legend_max_raw = style.get("run_health_compression_legend_max", 14)
    try:
        legend_max = max(1, int(legend_max_raw))
    except Exception:
        legend_max = 14
    by_plan_group = False
    if int(frame[plan_col].nunique(dropna=True)) > int(legend_max):
        grouped_lookup = {
            str(plan_name): plan_group_from_name(plan_name) for plan_name in pd.unique(frame[plan_col].astype(str))
        }
        grouped = frame[plan_col].astype(str).map(grouped_lookup)
        if int(grouped.nunique(dropna=True)) < int(frame[plan_col].nunique(dropna=True)):
            frame = frame.assign(__plan_group=grouped)
            plan_col = "__plan_group"
            by_plan_group = True
    plan_counts = frame.groupby(plan_col)[ratio_col].size().sort_values(ascending=False)
    plan_names = [str(name) for name in plan_counts.index.tolist()]
    fig_size = style.get("run_health_compression_figsize")
    if fig_size is None:
        side = max(6.2, min(9.8, 4.8 + 0.24 * float(len(plan_names))))
        fig_size = (side, side)
    fig, ax = plt.subplots(
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )
    palette = _palette(style, max(1, len(plan_names)))
    colors_by_plan = {plan: palette[idx] for idx, plan in enumerate(plan_names)}
    values_by_plan = {
        plan: frame.loc[frame[plan_col].astype(str) == plan, ratio_col].to_numpy(dtype=float) for plan in plan_names
    }
    max_points_raw = style.get("run_health_compression_max_points_per_plan", 900)
    try:
        max_points_per_plan = None if max_points_raw is None else max(1, int(max_points_raw))
    except Exception:
        max_points_per_plan = 900
    positions = np.arange(len(plan_names), dtype=float)
    _draw_category_box_and_jitter(
        ax,
        categories=plan_names,
        values_by_category=values_by_plan,
        colors_by_category=colors_by_plan,
        positions=positions,
        orientation="vertical",
        jitter_width=0.16,
        max_points_per_category=max_points_per_plan,
        rasterized=True,
    )
    label_rotation = 45
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_ellipsize(compact_plan_label(plan), max_len=28) for plan in plan_names],
        rotation=label_rotation,
        ha="right",
    )
    ax.set_xlabel("Plan")
    ax.set_ylabel("Compression ratio")
    if by_plan_group:
        ax.set_title("Compression ratio by plan group")
    else:
        ax.set_title("Compression ratio by plan")
    ax.set_box_aspect(1.0)
    ax.set_xlim(-0.6, float(len(plan_names) - 1) + 0.6)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.28)
    ax.tick_params(axis="x", labelsize=float(style.get("label_size", style.get("font_size", 12.5))))
    ax.tick_params(axis="y", labelsize=float(style.get("label_size", style.get("font_size", 12.5))))
    bottom = 0.34
    fig.subplots_adjust(left=0.14, right=0.96, bottom=bottom, top=0.9)
    _apply_style(ax, style)
    return fig, {"compression": ax}


def _build_run_health_tfbs_length_by_regulator_figure(
    composition_df: pd.DataFrame,
    *,
    library_members_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    if composition_df is None or composition_df.empty:
        raise ValueError("run_health tfbs_length_by_regulator requires composition.parquet with placements.")
    required = {"tf", "tfbs"}
    missing = required - set(composition_df.columns)
    if missing:
        raise ValueError(
            "run_health tfbs_length_by_regulator requires composition columns: "
            f"{', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}."
        )
    style = _style(style)
    frame = composition_df.copy()
    frame["regulator"] = frame["tf"].map(_usage_category_label).astype(str)
    frame = frame[~frame["regulator"].str.startswith("fixed:")].copy()
    frame = frame[~frame["regulator"].str.lower().isin({"", "nan", "none"})].copy()
    if frame.empty:
        raise ValueError("run_health tfbs_length_by_regulator found no regulator TFBS placements.")
    if "length" in frame.columns:
        frame["tfbs_length"] = pd.to_numeric(frame["length"], errors="coerce")
    else:
        frame["tfbs_length"] = frame["tfbs"].astype(str).str.len().astype(float)
    frame = frame.dropna(subset=["tfbs_length"]).copy()
    if frame.empty:
        raise ValueError("run_health tfbs_length_by_regulator found no TFBS length values.")
    frame["tfbs_length"] = frame["tfbs_length"].astype(int)

    counts = frame.groupby("regulator")["tfbs_length"].size().sort_values(ascending=False)
    regulators = order_regulators_for_display(
        counts.index.astype(str).tolist(),
        counts_by_regulator=counts.astype(int).to_dict(),
    )
    fig_size = style.get("run_health_tfbs_length_figsize")
    if fig_size is None:
        side = max(5.6, min(9.4, 4.8 + 0.28 * float(len(regulators))))
        fig_size = (side, side)
    fig, ax = plt.subplots(
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )
    palette = _palette(style, max(1, len(regulators)))
    colors_by_regulator = {regulator: palette[idx] for idx, regulator in enumerate(regulators)}
    candidate_pool_sizes: dict[str, int] = {}
    if library_members_df is not None and not library_members_df.empty:
        lib = library_members_df.copy()
        tf_col = "tf" if "tf" in lib.columns else ("regulator_id" if "regulator_id" in lib.columns else None)
        tfbs_col = "tfbs" if "tfbs" in lib.columns else ("tfbs_sequence" if "tfbs_sequence" in lib.columns else None)
        if tf_col is not None and tfbs_col is not None:
            lib["regulator"] = lib[tf_col].map(_usage_category_label).astype(str)
            lib = lib[~lib["regulator"].str.startswith("fixed:")].copy()
            lib = lib[~lib["regulator"].str.lower().isin({"", "nan", "none"})].copy()
            if not lib.empty:
                candidate_pool_sizes = (
                    lib[["regulator", tfbs_col]]
                    .drop_duplicates()
                    .groupby("regulator")[tfbs_col]
                    .nunique()
                    .astype(int)
                    .to_dict()
                )
    if not candidate_pool_sizes:
        candidate_pool_sizes = (
            frame[["regulator", "tfbs"]].drop_duplicates().groupby("regulator")["tfbs"].nunique().astype(int).to_dict()
        )
    unique_lengths = sorted(frame["tfbs_length"].astype(int).unique().tolist())
    y_centers = np.arange(len(regulators), dtype=float)
    total_group_height = 0.82
    bar_height = total_group_height / float(max(1, len(unique_lengths)))
    max_count = 0
    for regulator_idx, regulator in enumerate(regulators):
        center = float(y_centers[regulator_idx])
        base_color = colors_by_regulator[regulator]
        shades = _progressive_regulator_shades(base_color, len(unique_lengths))
        regulator_mask = frame["regulator"].astype(str) == regulator
        for length_idx, length_value in enumerate(unique_lengths):
            position = center - (total_group_height / 2.0) + (bar_height * length_idx) + (bar_height / 2.0)
            count = int((regulator_mask & (frame["tfbs_length"].astype(int) == int(length_value))).sum())
            max_count = max(max_count, count)
            ax.barh(
                position,
                count,
                height=bar_height * 0.9,
                color=shades[length_idx],
                edgecolor=base_color,
                linewidth=0.95,
                alpha=0.94,
                zorder=2.0,
            )
    ax.set_yticks(y_centers)
    ax.set_yticklabels([compact_regulator_label(regulator) for regulator in regulators])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Regulator")
    ax.set_title("TFBS lengths by regulator across accepted outputs")
    ax.set_xlim(0.0, max(1.0, float(max_count) * 1.14))
    ax.set_box_aspect(1.0)
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.28)
    ax.tick_params(axis="x", labelsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.9)))
    ax.tick_params(axis="y", labelsize=float(style.get("label_size", style.get("font_size", 13.0))), pad=6.0)
    ax.margins(y=0.08)
    grayscale_shades = np.linspace(0.82, 0.38, len(unique_lengths))
    legend_handles = [
        Patch(
            facecolor=str(gray_value),
            edgecolor="#8f949b",
            linewidth=0.9,
            label=f"{int(length_value)} bp",
        )
        for gray_value, length_value in zip(grayscale_shades.tolist(), unique_lengths, strict=False)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        title="Length",
        fontsize=float(style.get("label_size", style.get("font_size", 13.0)) * 0.92),
        title_fontsize=float(style.get("label_size", style.get("font_size", 13.0))),
    )
    longest_regulator_label = max((len(compact_regulator_label(reg)) for reg in regulators), default=8)
    left_margin = max(0.22, min(0.32, 0.16 + 0.008 * longest_regulator_label))
    fig.subplots_adjust(left=left_margin, right=0.97, bottom=0.14, top=0.9)
    _apply_style(ax, style)
    return fig, {"length": ax}
