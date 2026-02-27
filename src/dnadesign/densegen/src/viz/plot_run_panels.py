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

    fig, (ax_usage, ax_cum) = plt.subplots(2, 1, figsize=(8.8, 6.0), sharex=False)
    palette = _palette(style, max(1, len(category_order) + 1))
    category_colors = {label: palette[idx + 1] for idx, label in enumerate(category_order)}
    ax_usage.axhline(0.0, color="#999999", linewidth=0.8, alpha=0.8, zorder=1)
    for label in category_order:
        cat_points = counts[counts["category_label"] == label]
        if cat_points.empty:
            continue
        x_vals = cat_points["global_rank"].astype(float).to_numpy()
        y_vals = cat_points["count"].astype(float).to_numpy()
        color = category_colors[label]
        ax_usage.vlines(
            x_vals,
            0.0,
            y_vals,
            color=color,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
        ax_usage.scatter(
            x_vals,
            y_vals,
            color=color,
            s=18,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.95,
            zorder=3,
        )
    ax_usage.set_ylabel("Usage count")
    ax_usage.set_xlabel("TFBS rank (specific sequence)")
    input_label = _humanize_scope_label(input_name) or str(input_name)
    plan_label = _humanize_scope_label(plan_name) or str(plan_name)
    ax_usage.set_title(f"TFBS usage breakdown for input {input_label} and plan {plan_label}")
    max_rank_within_regulator = 1
    for label in category_order:
        cat_points = counts[counts["category_label"] == label].sort_values(
            by=["count", "tfbs"],
            ascending=[False, True],
        )
        if cat_points.empty:
            continue
        cat_values = cat_points["count"].astype(float).to_numpy()
        cat_total = float(cat_values.sum())
        cat_ranks = np.arange(1, len(cat_values) + 1, dtype=float)
        cat_cum = np.cumsum(cat_values) / cat_total if cat_total > 0 else np.zeros_like(cat_values)
        max_rank_within_regulator = max(max_rank_within_regulator, int(cat_ranks[-1]))
        ax_cum.plot(
            cat_ranks,
            cat_cum,
            color=category_colors[label],
            linewidth=1.4,
            marker="o",
            markersize=2.8,
            alpha=0.9,
            zorder=3,
        )
    ax_cum.set_ylabel("Cumulative share within regulator")
    ax_cum.set_xlabel("TFBS rank within regulator")
    ax_cum.set_ylim(0.0, 1.03)
    ax_cum.set_xlim(0.7, float(max_rank_within_regulator) + 0.3)
    ax_cum.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

    if all_values.size > 0:
        y_max = float(np.nanmax(all_values)) * 1.08
        if y_max <= 0:
            y_max = 1.0
        ax_usage.set_ylim(0.0, y_max)

    summary_lines = [
        f"placements in outputs: {int(total)}",
        f"unique TFBS-pairs in outputs: {len(counts)}",
        f"top10 share (specific TFBS rank): {top10:.1%}",
        f"top50 share (specific TFBS rank): {top50:.1%}",
    ]
    if available_total > 0:
        summary_lines.append(
            f"unique TFBS-pairs used / available: {len(counts)}/{available_total} ({len(counts) / available_total:.1%})"
        )
    summary_lines = [_capitalize_first(line) for line in summary_lines]
    summary_font_size = float(
        style.get(
            "tfbs_usage_summary_size",
            max(
                10.0,
                float(style.get("label_size", style.get("font_size", 13.0))) * 0.9,
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
            f"{label}: placements {cat_total}/{int(total)} ({share:.1%}), "
            f"unique {used_unique}/{max(1, available_unique)}"
        )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
            fontsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.62)),
        )
    fig.tight_layout(rect=(0.0, 0.15, 1.0, 1.0))
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
        fig_size = (10.4, 4.9)
    fig, ax = plt.subplots(
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )
    x = np.arange(len(regulators), dtype=float)
    width = 0.82 / max(1, len(lengths))
    palette = _palette(style, max(1, len(lengths)))
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
