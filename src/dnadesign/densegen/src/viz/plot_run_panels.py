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
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

from .plot_common import _apply_style, _palette, _save_figure, _stage_b_plan_output_dir, _style, plan_group_from_name
from .plot_run_helpers import (
    _bin_attempts,
    _ellipsize,
    _first_existing_column,
    _humanize_scope_label,
    _normalize_plan_name,
    _usage_available_unique,
    _usage_category_label,
)


def _capitalize_first(text: str) -> str:
    token = str(text)
    for idx, char in enumerate(token):
        if char.isalpha():
            return token[:idx] + char.upper() + token[idx + 1 :]
    return token


def _build_tfbs_usage_breakdown_figure(
    composition_df: pd.DataFrame,
    *,
    input_name: str,
    plan_name: str,
    style: Optional[dict] = None,
    pools: dict[str, pd.DataFrame] | None = None,
    library_members_df: pd.DataFrame | None = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    style = _style(style)
    sub = composition_df[
        (composition_df["input_name"].astype(str) == str(input_name))
        & (composition_df["plan_name"].astype(str) == str(plan_name))
    ].copy()
    if sub.empty:
        raise ValueError(f"tfbs_usage found no placements for {input_name}/{plan_name}.")
    sub["category_label"] = sub["tf"].map(_usage_category_label)
    sub["tfbs"] = sub["tfbs"].astype(str)
    counts = (
        sub.groupby(["category_label", "tfbs"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "category_label", "tfbs"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    if counts.empty:
        raise ValueError(f"tfbs_usage found no TFBS counts for {input_name}/{plan_name}.")

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
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError("tfbs_usage requires composition.parquet with placements.")
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
    frame[plan_col] = frame[plan_col].map(_normalize_plan_name).fillna("all plans")
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
        grouped = frame[plan_col].astype(str).map(plan_group_from_name)
        if int(grouped.nunique(dropna=True)) < int(frame[plan_col].nunique(dropna=True)):
            frame = frame.assign(__plan_group=grouped)
            plan_col = "__plan_group"
            by_plan_group = True
    plan_counts = frame.groupby(plan_col)[ratio_col].size().sort_values(ascending=False)
    plan_names = [str(name) for name in plan_counts.index.tolist()]
    labeled_plans = set(plan_names[:legend_max])
    fig_size = style.get("run_health_compression_figsize")
    if fig_size is None:
        fig_size = (7.2, 4.0)
    fig, ax = plt.subplots(
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )
    values = frame[ratio_col].to_numpy(dtype=float)
    bins = max(8, min(42, int(np.ceil(np.sqrt(values.size) * 2.0))))
    edges, _centers = _bin_attempts(values, bins=bins)
    palette = _palette(style, max(1, len(plan_names)))
    for idx, plan in enumerate(plan_names):
        plan_values = frame.loc[frame[plan_col].astype(str) == plan, ratio_col].to_numpy(dtype=float)
        if plan_values.size == 0:
            continue
        ax.hist(
            plan_values,
            bins=edges,
            alpha=0.45,
            color=palette[idx],
            edgecolor="white",
            linewidth=0.55,
            label=(f"{_ellipsize(plan, max_len=24)} (n={plan_values.size})" if plan in labeled_plans else None),
        )
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Count")
    if by_plan_group:
        ax.set_title("Compression ratio distribution by plan group")
    else:
        ax.set_title("Compression ratio distribution by plan")
    legend_ncol = 1
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=legend_ncol,
        fontsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.88)),
    )
    hidden_plans = max(0, len(plan_names) - len(labeled_plans))
    if hidden_plans > 0:
        ax.text(
            0.995,
            0.01,
            f"Legend shows top {len(labeled_plans)} of {len(plan_names)} plans",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.82)),
            alpha=0.86,
        )
    fig.subplots_adjust(right=0.74)
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

    counts = (
        frame.groupby(["regulator", "tfbs_length"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["regulator", "tfbs_length"], ascending=[True, True])
    )
    if counts.empty:
        raise ValueError("run_health tfbs_length_by_regulator produced no regulator/length counts.")
    pivot = counts.pivot(index="regulator", columns="tfbs_length", values="count").fillna(0).astype(int)
    pivot["__total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total", ascending=False).drop(columns=["__total"])
    regulators = pivot.index.astype(str).tolist()
    lengths = [int(v) for v in pivot.columns.tolist()]
    fig_size = style.get("run_health_tfbs_length_figsize")
    if fig_size is None:
        if len(regulators) == 1:
            fig_size = (4.6, 4.6)
        else:
            fig_size = (7.2, 3.5)
    fig, ax = plt.subplots(
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )
    x = np.arange(len(regulators), dtype=float)
    width = 0.82 / max(1, len(lengths))
    palette = _palette(style, max(1, len(lengths)))
    candidate_pool_sizes: dict[str, int] = {}
    if library_members_df is not None and not library_members_df.empty:
        lib = library_members_df.copy()
        tf_col = "tf" if "tf" in lib.columns else ("regulator_id" if "regulator_id" in lib.columns else None)
        tfbs_col = "tfbs" if "tfbs" in lib.columns else ("tfbs_sequence" if "tfbs_sequence" in lib.columns else None)
        if tf_col is not None and tfbs_col is not None:
            lib["regulator"] = lib[tf_col].map(_usage_category_label).astype(str)
            lib = lib[~lib["regulator"].str.startswith("fixed:")].copy()
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
    if len(regulators) == 1:
        regulator = regulators[0]
        regulator_counts = (
            counts[counts["regulator"] == regulator]
            .sort_values(by=["tfbs_length"], ascending=[True])
            .reset_index(drop=True)
        )
        lengths_single = regulator_counts["tfbs_length"].astype(int).to_numpy()
        values_single = regulator_counts["count"].astype(float).to_numpy()
        single_color = "#7a7a7a"
        ax.bar(
            lengths_single.astype(float),
            values_single,
            width=0.8,
            color=single_color,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(lengths_single.astype(float))
        ax.set_xticklabels([str(item) for item in lengths_single.tolist()])
        ax.set_xlabel("TFBS length (bp)")
        ax.set_ylabel("Count in accepted outputs")
        ax.set_title("TFBS length distribution across accepted outputs")
        ax.text(
            0.99,
            0.96,
            f"Regulator: {_ellipsize(regulator, max_len=24)}\n"
            f"unique available: {int(candidate_pool_sizes.get(regulator, 0))}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(8.0, float(style.get("tick_size", style.get("font_size", 13.0) * 0.62))),
            color="#333333",
        )
    else:
        for idx, length in enumerate(lengths):
            offset = (float(idx) - (float(len(lengths) - 1) / 2.0)) * width
            y = pivot[length].to_numpy(dtype=float)
            ax.bar(
                x + offset,
                y,
                width=width * 0.92,
                color=palette[idx],
                edgecolor="white",
                linewidth=0.5,
                label=f"{length} bp",
            )
        x_labels = [
            f"{_ellipsize(label, max_len=20)}\n(n={int(candidate_pool_sizes.get(label, 0))})" for label in regulators
        ]
        rotate = 20 if len(regulators) > 4 else 0
        ax.set_xticks(x)
        ax.set_xticklabels(
            x_labels,
            rotation=rotate,
            ha="right" if rotate else "center",
        )
        ax.set_xlabel("Regulator")
        ax.set_ylabel("Count in accepted outputs")
        ax.set_title("TFBS length counts by regulator across accepted outputs")
        ax.legend(
            loc="upper right",
            title="TFBS length",
            frameon=False,
            fontsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.86)),
            title_fontsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.9)),
        )
    ax.margins(y=0.08)
    _apply_style(ax, style)
    return fig, {"length": ax}
