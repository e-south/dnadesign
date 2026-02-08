"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run.py

Run-level plotting for placements, TFBS usage, and run health diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from .plot_common import _apply_style, _palette, _stage_b_plan_output_dir, _style


def _bin_attempts(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def _axis_pixel_width(ax) -> float:
    fig = ax.figure
    width = fig.get_figwidth() * fig.dpi
    return max(1.0, width * ax.get_position().width)


def _resolution_bins(ax, n_points: int, *, min_bins: int = 25) -> int:
    if n_points <= 0:
        return int(min_bins)
    px = _axis_pixel_width(ax)
    return max(min_bins, min(int(px), int(n_points)))


def _usage_category_label(value: object) -> str:
    label = str(value or "").strip()
    if not label:
        return ""
    if label.startswith("fixed:"):
        return label
    if "_" in label:
        head, tail = label.split("_", 1)
        tail_upper = tail.upper()
        iupac = set("ACGTURYSWKMBDHVN")
        if len(tail_upper) >= 6 and set(tail_upper).issubset(iupac):
            return head
    return label


def _usage_available_unique(
    *,
    input_name: str,
    plan_name: str,
    pools: dict[str, pd.DataFrame] | None,
    library_members_df: pd.DataFrame | None,
) -> tuple[dict[str, int], int]:
    if library_members_df is not None and not library_members_df.empty:
        required = {"input_name", "plan_name", "tf", "tfbs"}
        missing = required - set(library_members_df.columns)
        if missing:
            raise ValueError(f"library_members.parquet missing required columns: {sorted(missing)}")
        offered = library_members_df[
            (library_members_df["input_name"].astype(str) == str(input_name))
            & (library_members_df["plan_name"].astype(str) == str(plan_name))
        ].copy()
        if offered.empty:
            return {}, 0
        offered["category_label"] = offered["tf"].map(_usage_category_label)
        offered["tfbs"] = offered["tfbs"].astype(str)
        unique_pairs = offered[["category_label", "tfbs"]].drop_duplicates()
        by_category = (
            unique_pairs.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_available"})
        )
        return by_category["unique_available"].to_dict(), int(len(unique_pairs))

    if pools and input_name in pools:
        pool_df = pools[input_name]
        if pool_df.empty or "tf" not in pool_df.columns:
            return {}, 0
        tfbs_col = "tfbs_sequence" if "tfbs_sequence" in pool_df.columns else "tfbs"
        if tfbs_col not in pool_df.columns:
            return {}, 0
        offered = pool_df.assign(
            category_label=pool_df["tf"].map(_usage_category_label),
            tfbs=pool_df[tfbs_col].astype(str),
        )[["category_label", "tfbs"]]
        unique_pairs = offered.drop_duplicates()
        by_category = (
            unique_pairs.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_available"})
        )
        return by_category["unique_available"].to_dict(), int(len(unique_pairs))
    return {}, 0


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
    all_values = counts["count"].astype(float).to_numpy()
    all_ranks = np.arange(1, len(all_values) + 1)
    all_cum = np.cumsum(all_values) / total if total > 0 else np.zeros_like(all_values)
    available_by_category, available_total = _usage_available_unique(
        input_name=input_name,
        plan_name=plan_name,
        pools=pools,
        library_members_df=library_members_df,
    )

    fig, (ax_usage, ax_cum) = plt.subplots(2, 1, figsize=(8.8, 6.0), sharex=False)
    palette = _palette(style, max(1, counts["category_label"].nunique() + 1))
    ax_usage.plot(
        all_ranks,
        all_values,
        color=palette[0],
        linewidth=1.5,
        marker="o",
        markersize=2.8,
        label="all TFBS-pairs",
        zorder=4,
    )
    ax_usage.set_yscale("log")
    ax_usage.set_ylabel("Usage count")
    ax_usage.set_title(f"TFBS usage breakdown - {input_name}/{plan_name}")

    ax_cum.plot(
        all_ranks,
        all_cum,
        color=palette[0],
        linewidth=1.5,
        marker="o",
        markersize=2.8,
        label="all TFBS-pairs",
        zorder=4,
    )
    ax_cum.set_ylabel("Cumulative share")
    ax_cum.set_xlabel("TFBS rank within category")
    ax_cum.set_ylim(0.0, 1.0)

    category_order = (
        counts.groupby("category_label")["count"].sum().sort_values(ascending=False).index.astype(str).tolist()
    )
    for idx, category in enumerate(category_order):
        cat = counts[counts["category_label"] == category].sort_values(by=["count", "tfbs"], ascending=[False, True])
        values = cat["count"].astype(float).to_numpy()
        cat_total = float(values.sum())
        ranks = np.arange(1, len(values) + 1)
        cat_cum = np.cumsum(values) / cat_total if cat_total > 0 else np.zeros_like(values)
        available_unique = int(available_by_category.get(category, 0))
        legend_label = (
            f"{category}: placements {int(cat_total)}/{int(total)} ({(cat_total / total) if total > 0 else 0.0:.1%}), "
            f"unique {len(cat)}/{max(1, available_unique)}"
        )
        color = palette[idx + 1]
        ax_usage.plot(ranks, values, color=color, linewidth=1.2, marker="o", markersize=2.4, label=legend_label)
        ax_cum.plot(ranks, cat_cum, color=color, linewidth=1.2, marker="o", markersize=2.4, label=legend_label)

    summary_lines = [f"placements in outputs: {int(total)}", f"unique TFBS-pairs in outputs: {len(counts)}"]
    if available_total > 0:
        summary_lines.append(
            f"unique TFBS-pairs used / available: {len(counts)}/{available_total} ({len(counts) / available_total:.1%})"
        )
    ax_usage.text(
        0.98,
        0.95,
        "\n".join(summary_lines),
        transform=ax_usage.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
    )

    _apply_style(ax_usage, style)
    _apply_style(ax_cum, style)
    handles, labels = ax_usage.get_legend_handles_labels()
    if handles:
        deduped: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            deduped[str(label)] = handle
        fig.legend(
            deduped.values(),
            deduped.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=bool(style.get("legend_frame", False)),
            fontsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.62)),
        )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    return fig, {"usage": ax_usage, "cum": ax_cum}


def plot_tfbs_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    pools: dict[str, pd.DataFrame] | None = None,
    library_members_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError("tfbs_usage requires composition.parquet with placements.")
    required = {"input_name", "plan_name", "tf", "tfbs"}
    missing = required - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    paths: list[Path] = []
    for input_name, plan_name in composition_df[["input_name", "plan_name"]].drop_duplicates().itertuples(index=False):
        fig, _axes = _build_tfbs_usage_breakdown_figure(
            composition_df,
            input_name=str(input_name),
            plan_name=str(plan_name),
            style=style,
            pools=pools,
            library_members_df=library_members_df,
        )
        target_dir = _stage_b_plan_output_dir(out_path, input_name=str(input_name), plan_name=str(plan_name))
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"tfbs_usage{out_path.suffix}"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def _prepare_run_health_inputs(
    attempts_df: pd.DataFrame,
    *,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
) -> tuple[dict, pd.DataFrame, list[str], dict[str, int], ProgressAxis, pd.Series, str, float]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError("run_health requires attempts.parquet.")
    required = {"status", "reason", "plan_name"}
    missing = required - set(attempts_df.columns)
    if missing:
        raise ValueError(f"attempts.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    attempts_df = _normalize_and_order_attempts(attempts_df)
    progress = _progress_axis(attempts_df, max_points=max(1, len(attempts_df) + 1))
    solver_x, solver_x_label = _solver_progress_x(attempts_df)
    legend_size = float(style.get("legend_size", style.get("font_size", 13) * 0.74))

    normalized_plan_series = attempts_df["plan_name"].map(_normalize_plan_name)
    if normalized_plan_series.isna().all():
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = "all plans"
    else:
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = normalized_plan_series.fillna("all plans")

    plan_names_unique = sorted(set(attempts_df["plan_name"].astype(str).tolist()))
    quota_map = dict(plan_quotas or {})
    if quota_map:
        plan_names = [name for name in quota_map.keys() if name in plan_names_unique]
        plan_names.extend([name for name in plan_names_unique if name not in set(plan_names)])
    else:
        plan_names = plan_names_unique
    if not plan_names:
        plan_names = ["all plans"]
        attempts_df["plan_name"] = "all plans"
    missing_quota = [plan for plan in plan_names if int(quota_map.get(plan, 0)) <= 0]
    if missing_quota:
        raise ValueError(
            "run_health requires generation.plan quotas for all plans in attempts. "
            f"Missing or invalid quota for: {', '.join(sorted(missing_quota))}"
        )
    return style, attempts_df, plan_names, quota_map, progress, solver_x, solver_x_label, legend_size


def _build_run_health_outcomes_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    _style_cfg, attempts_df, plan_names, _quota_map, _progress, solver_x, solver_x_label, legend_size = (
        _prepare_run_health_inputs(
            attempts_df,
            plan_quotas=plan_quotas,
            style=style,
        )
    )
    fig_size = _style_cfg.get("run_health_outcomes_figsize")
    if fig_size is None:
        fig_size = (12.8, 3.8)
    fig, ax = plt.subplots(figsize=(float(fig_size[0]), float(fig_size[1])), constrained_layout=True)

    plan_to_row = {name: i for i, name in enumerate(plan_names)}
    plot_df = attempts_df.copy()
    plot_df["_plan_row"] = plot_df["plan_name"].astype(str).map(plan_to_row).fillna(0).astype(float)
    plot_df["_solver_x"] = solver_x.to_numpy(dtype=float)

    accepted_or_duplicate = plot_df[plot_df["status"].isin(["ok", "duplicate"])]
    rejected = plot_df[plot_df["status"] == "rejected"]
    failed = plot_df[plot_df["status"] == "failed"]

    run_x = plot_df["_solver_x"].to_numpy(dtype=float)
    run_y = plot_df["_plan_row"].to_numpy(dtype=float)
    valid = np.isfinite(run_x) & np.isfinite(run_y)
    run_x = run_x[valid]
    run_y = run_y[valid]
    if run_x.size >= 2:
        changed = np.ones(run_x.size, dtype=bool)
        changed[1:] = (run_x[1:] != run_x[:-1]) | (run_y[1:] != run_y[:-1])
        path_x = run_x[changed]
        path_y = run_y[changed]
        if path_x.size >= 2:
            ax.plot(
                path_x,
                path_y,
                color="#c7c7c7",
                linewidth=2.5,
                alpha=0.45,
                zorder=1,
            )

    if not accepted_or_duplicate.empty:
        ax.scatter(
            accepted_or_duplicate["_solver_x"].to_numpy(dtype=float),
            accepted_or_duplicate["_plan_row"].to_numpy(dtype=float),
            s=34.0,
            marker="s",
            linewidths=0.35,
            edgecolors="#c4c4c4",
            color="#d9d9d9",
            zorder=2,
            label="_nolegend_",
        )
    if not rejected.empty:
        ax.scatter(
            rejected["_solver_x"].to_numpy(dtype=float),
            rejected["_plan_row"].to_numpy(dtype=float),
            s=40.0,
            marker="s",
            linewidths=0.35,
            edgecolors="#c4c4c4",
            color="#D55E00",
            zorder=3,
            label="_nolegend_",
        )
    if not failed.empty:
        ax.scatter(
            failed["_solver_x"].to_numpy(dtype=float),
            failed["_plan_row"].to_numpy(dtype=float),
            s=46.0,
            marker="x",
            linewidths=1.2,
            color="#D55E00",
            zorder=4,
            label="_nolegend_",
        )

    for row in range(len(plan_names) - 1):
        ax.axhline(row + 0.5, color="#d8d8d8", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_yticks(np.arange(len(plan_names), dtype=float))
    ax.set_yticklabels([_ellipsize(name, max_len=24) for name in plan_names])
    ax.set_ylim(-0.5, float(len(plan_names)) - 0.5)
    ticks = _solver_ticks(solver_x.to_numpy(dtype=float))
    if ticks.size > 0:
        ax.set_xticks(ticks)
    if solver_x.size > 0:
        ax.set_xlim(float(solver_x.min()) - 0.5, float(solver_x.max()) + 0.5)
    ax.set_xlabel(solver_x_label)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.set_title("Outcomes over time", pad=17.0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="#d9d9d9",
            markeredgecolor="#bcbcbc",
            markeredgewidth=0.5,
            markersize=6.0,
            label="accepted",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="#D55E00",
            markeredgecolor="#8f2a13",
            markeredgewidth=0.5,
            markersize=6.0,
            label="rejected",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="#D55E00",
            markersize=6.5,
            label="failed",
        ),
    ]
    label_size = float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13)))
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=label_size,
        ncol=1,
        borderaxespad=0.0,
    )
    _apply_style(ax, _style_cfg)
    return fig, {"outcome": ax}


def _build_run_health_detail_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    _style_cfg, attempts_df, plan_names, quota_map, progress, _solver_x, _solver_x_label, legend_size = (
        _prepare_run_health_inputs(
            attempts_df,
            plan_quotas=plan_quotas,
            style=style,
        )
    )
    fig_size = _style_cfg.get("run_health_detail_figsize")
    if fig_size is None:
        fig_size = (9.4, 4.9)
    fig, (ax_fail, ax_plan) = plt.subplots(
        1,
        2,
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )

    plan_palette = _palette(_style_cfg, max(3, len(plan_names)))
    plan_colors = {plan: plan_palette[idx] for idx, plan in enumerate(plan_names)}
    problem = attempts_df[attempts_df["status"].astype(str).isin(["rejected", "failed"])].copy()
    if problem.empty:
        ax_fail.text(
            0.5,
            0.5,
            "No rejected/failed reasons (only ok/duplicate)",
            ha="center",
            va="center",
            transform=ax_fail.transAxes,
        )
        ax_fail.set_axis_off()
    else:
        reason_plan = problem.copy()
        reason_plan["reason_family"] = reason_plan.apply(
            lambda row: _reason_family_label(str(row.get("status", "")), row.get("reason")),
            axis=1,
        )
        reason_counts = (
            reason_plan.groupby(["reason_family", "plan_name"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=plan_names, fill_value=0)
        )
        reason_counts["total"] = reason_counts.sum(axis=1)
        reason_counts = reason_counts.sort_values("total", ascending=False)
        proportions = reason_counts[plan_names].div(reason_counts["total"].replace(0, 1), axis=0)
        positions = np.arange(len(proportions), dtype=float)
        left = np.zeros(len(proportions), dtype=float)
        for plan in plan_names:
            vals = proportions[plan].to_numpy(dtype=float)
            ax_fail.barh(
                positions,
                vals,
                left=left,
                color=plan_colors[plan],
                edgecolor="white",
                linewidth=0.5,
                height=0.56,
            )
            left += vals
        ax_fail.set_yticks(positions)
        ax_fail.set_yticklabels([_ellipsize(item, max_len=30) for item in proportions.index.tolist()])
        ax_fail.invert_yaxis()
        ax_fail.set_xlim(0.0, 1.0)
        ax_fail.set_xlabel("Plan share of failed solves")
        ax_fail.set_title("Reason for failed solve")

    max_progress = 0.0
    for idx, plan in enumerate(plan_names):
        quota = int(quota_map[plan])
        plan_mask = attempts_df["plan_name"].astype(str) == plan
        accepted_mask = (plan_mask & (attempts_df["status"] == "ok")).astype(int).to_numpy(dtype=int)
        if progress.mode == "discrete":
            accepted_counts = accepted_mask.astype(float)
        else:
            if progress.bin_id is None:
                raise ValueError("run_health binned plan progress requires bin_id.")
            accepted_counts = (
                pd.DataFrame({"bin_id": progress.bin_id, "accepted": accepted_mask})
                .groupby("bin_id")["accepted"]
                .sum()
                .reindex(np.arange(len(progress.x), dtype=int), fill_value=0)
                .to_numpy(dtype=float)
            )
        cumulative = np.cumsum(accepted_counts)
        ratio = cumulative / float(max(1, quota))
        max_progress = max(max_progress, float(np.nanmax(ratio)) if ratio.size else 0.0)
        color = plan_colors[plan]
        accepted_final = int(cumulative[-1]) if cumulative.size > 0 else 0
        ax_plan.plot(
            progress.x,
            ratio,
            linewidth=1.6,
            color=color,
            label=f"{_ellipsize(plan, 20)} ({accepted_final}/{quota})",
        )
        hit = np.where(ratio >= 1.0)[0]
        if hit.size > 0:
            h = int(hit[0])
            ax_plan.scatter(
                float(progress.x[h]),
                float(ratio[h]),
                s=26.0,
                marker="o",
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
    ax_plan.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--")
    unique_quotas = sorted({int(value) for value in quota_map.values()})
    if len(unique_quotas) == 1:
        quota_text = f"Quota ({unique_quotas[0]})"
    else:
        quota_text = f"Quota ({unique_quotas[0]}-{unique_quotas[-1]})"
    ax_plan.text(
        0.01,
        1.0,
        quota_text,
        transform=ax_plan.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=max(10.0, legend_size * 1.2),
        color="#555555",
    )
    ax_plan.set_xlabel("Attempt index")
    ax_plan.set_ylabel("Accepted / quota")
    ax_plan.set_ylim(0.0, max(1.05, max_progress + 0.05))
    ax_plan.set_title("Quota progress")
    ax_plan.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax_fail.set_box_aspect(1.0)
    ax_plan.set_box_aspect(1.0)

    _apply_style(ax_fail, _style_cfg)
    _apply_style(ax_plan, _style_cfg)
    handles = [
        Line2D([0], [0], color=plan_colors[plan], linewidth=4.0, label=_ellipsize(plan, max_len=20))
        for plan in plan_names
    ]
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=max(1, len(plan_names)),
        frameon=False,
        fontsize=float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13))),
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    return fig, {"fail": ax_fail, "plan": ax_plan}


def plot_run_health(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    style = _style(style)
    plan_quotas = _extract_plan_quotas(cfg)
    fig_outcome, _axes_outcome = _build_run_health_outcomes_figure(
        attempts_df,
        events_df=events_df,
        plan_quotas=plan_quotas,
        style=style,
    )
    fig_detail, _axes_detail = _build_run_health_detail_figure(
        attempts_df,
        events_df=events_df,
        plan_quotas=plan_quotas,
        style=style,
    )
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = target_dir / f"outcomes_over_time{out_path.suffix}"
    run_health_path = target_dir / f"run_health{out_path.suffix}"
    legacy_detail_path = target_dir / f"run_health_detail{out_path.suffix}"
    legacy_detail_path.unlink(missing_ok=True)
    fig_outcome.savefig(outcomes_path)
    fig_detail.savefig(run_health_path)
    plt.close(fig_outcome)
    plt.close(fig_detail)
    summary_df = _run_health_summary_frame(_normalize_and_order_attempts(attempts_df), plan_quotas=plan_quotas)
    summary_df.to_csv(target_dir / "summary.csv", index=False)
    return [outcomes_path, run_health_path]


def _run_health_summary_frame(attempts_df: pd.DataFrame, *, plan_quotas: dict[str, int]) -> pd.DataFrame:
    n_attempts = int(len(attempts_df))
    n_ok = int((attempts_df["status"] == "ok").sum())
    n_rej = int((attempts_df["status"] == "rejected").sum())
    n_dup = int((attempts_df["status"] == "duplicate").sum())
    n_fail = int((attempts_df["status"] == "failed").sum())
    waste_rate = (n_rej + n_dup + n_fail) / float(max(1, n_attempts))
    rows: list[dict[str, object]] = [
        {"scope": "run", "name": "attempts", "value": n_attempts, "unit": "count"},
        {"scope": "run", "name": "ok", "value": n_ok, "unit": "count"},
        {"scope": "run", "name": "rejected", "value": n_rej, "unit": "count"},
        {"scope": "run", "name": "duplicate", "value": n_dup, "unit": "count"},
        {"scope": "run", "name": "failed", "value": n_fail, "unit": "count"},
        {"scope": "run", "name": "waste_rate", "value": waste_rate, "unit": "fraction"},
    ]
    by_plan = (
        attempts_df.groupby("plan_name")
        .agg(
            attempts=("status", "size"),
            ok=("status", lambda s: int((s == "ok").sum())),
            rejected=("status", lambda s: int((s == "rejected").sum())),
            duplicate=("status", lambda s: int((s == "duplicate").sum())),
            failed=("status", lambda s: int((s == "failed").sum())),
        )
        .reset_index()
    )
    for row in by_plan.to_dict(orient="records"):
        plan = str(row["plan_name"])
        quota = int(plan_quotas.get(plan, 0))
        ok_count = int(row["ok"])
        rows.append(
            {
                "scope": "plan",
                "name": f"{plan}:accepted_over_quota",
                "value": (ok_count / float(quota)) if quota > 0 else np.nan,
                "unit": "fraction",
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ProgressAxis:
    mode: str
    x: np.ndarray
    attempt_idx: np.ndarray
    bin_id: np.ndarray | None
    bin_size: int


def _progress_axis(
    df: pd.DataFrame,
    *,
    max_points: int = 500,
    target_bins: int = 160,
    min_bin_size: int = 10,
) -> ProgressAxis:
    if "run_order" not in df.columns:
        raise ValueError("run_health progress axis requires run_order.")
    n = int(len(df))
    if n <= 0:
        raise ValueError("run_health progress axis requires non-empty attempts.")
    attempt_idx = np.arange(1, n + 1, dtype=int)
    if n <= int(max_points):
        return ProgressAxis(
            mode="discrete",
            x=attempt_idx.astype(float),
            attempt_idx=attempt_idx,
            bin_id=None,
            bin_size=1,
        )

    target_bins = max(1, int(target_bins))
    min_bin_size = max(1, int(min_bin_size))
    bin_size = max(min_bin_size, int(np.ceil(float(n) / float(target_bins))))
    raw_bin = (attempt_idx - 1) // bin_size
    if raw_bin.size > 0:
        counts = np.bincount(raw_bin)
        if len(counts) > 1 and counts[-1] < min_bin_size:
            raw_bin[raw_bin == (len(counts) - 1)] = len(counts) - 2
    _, bin_id = np.unique(raw_bin, return_inverse=True)
    x = np.array([attempt_idx[bin_id == b].mean() for b in range(int(bin_id.max()) + 1)], dtype=float)
    return ProgressAxis(
        mode="binned",
        x=x,
        attempt_idx=attempt_idx,
        bin_id=bin_id.astype(int),
        bin_size=bin_size,
    )


def _solver_progress_x(attempts_df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "sampling_library_index" in attempts_df.columns:
        numeric = pd.to_numeric(attempts_df["sampling_library_index"], errors="coerce")
        if numeric.notna().sum() > 0 and int(numeric.nunique(dropna=True)) > 1:
            return numeric.ffill().bfill().astype(int), "Solver step"
    return attempts_df["run_order"].astype(int), "Attempt index"


def _aggregate_counts_for_progress(
    attempts_df: pd.DataFrame,
    *,
    statuses: list[str],
    progress: ProgressAxis,
) -> pd.DataFrame:
    status_series = attempts_df["status"].astype(str)
    if progress.mode == "discrete":
        return pd.DataFrame(
            {status: (status_series == status).astype(float).to_numpy() for status in statuses},
            index=np.arange(len(attempts_df), dtype=int),
        )
    if progress.bin_id is None:
        raise ValueError("Binned progress axis requires bin_id.")
    grouped = (
        attempts_df.assign(_bin_id=progress.bin_id)
        .groupby(["_bin_id", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=statuses, fill_value=0)
        .sort_index()
    )
    return grouped.astype(float).reset_index(drop=True)


def _rate_series_from_counts(counts: pd.DataFrame) -> dict[str, np.ndarray]:
    ok = counts.get("ok", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    rejected = counts.get("rejected", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    duplicate = counts.get("duplicate", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    failed = counts.get("failed", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    totals = ok + rejected + duplicate + failed
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    acceptance = ok / safe_totals
    waste = (rejected + duplicate + failed) / safe_totals
    duplicate_rate = duplicate / safe_totals
    return {
        "acceptance": acceptance,
        "waste": waste,
        "duplicate": duplicate_rate,
        "totals": totals,
    }


def _subtitle(
    ax: plt.Axes,
    text: str,
    *,
    fontsize: float,
    y: float = 1.02,
    color: str = "#444444",
) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=color,
    )


def _solver_ticks(values: np.ndarray, *, max_ticks: int = 10) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    unique = np.unique(values.astype(int))
    if unique.size <= max_ticks:
        return unique.astype(float)
    idx = np.linspace(0, unique.size - 1, num=max_ticks, dtype=int)
    return np.unique(unique[idx]).astype(float)


def _link_panels_by_ticks(fig: plt.Figure, ax_top: plt.Axes, ax_bottom: plt.Axes, ticks: np.ndarray) -> None:
    if ticks.size == 0:
        return
    y_top = float(ax_top.get_ylim()[0])
    y_bottom = float(ax_bottom.get_ylim()[1])
    for x in ticks.tolist():
        connector = ConnectionPatch(
            xyA=(float(x), y_top),
            coordsA=ax_top.transData,
            xyB=(float(x), y_bottom),
            coordsB=ax_bottom.transData,
            axesA=ax_top,
            axesB=ax_bottom,
            linestyle="--",
            linewidth=0.55,
            color="#9a9a9a",
            alpha=0.55,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(connector)


def _save_axes_subset(fig: plt.Figure, path: Path, axes: list[plt.Axes | None], *, pad: float = 0.05) -> None:
    selected = [ax for ax in axes if ax is not None]
    if not selected:
        raise ValueError(f"No axes provided for saving subset figure: {path}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = selected[0].get_tightbbox(renderer)
    for ax in selected[1:]:
        bbox = bbox.union([bbox, ax.get_tightbbox(renderer)])
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox_inches.expanded(1.0 + pad, 1.0 + pad))


def _aggregate_reason_pareto(problem_df: pd.DataFrame, *, top_k: int | None = 8) -> pd.DataFrame:
    if problem_df is None or problem_df.empty:
        return pd.DataFrame(columns=["rejected", "failed", "total"])
    required = {"status", "reason"}
    missing = required - set(problem_df.columns)
    if missing:
        raise ValueError(f"run_health reason analysis missing required columns: {sorted(missing)}")
    reasons = problem_df.copy()
    reasons["reason_family"] = reasons.apply(
        lambda row: _reason_family_label(str(row.get("status", "")), row.get("reason")),
        axis=1,
    )
    pivot = (
        reasons.groupby(["reason_family", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["rejected", "failed"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    if top_k is not None and len(pivot) > int(top_k):
        head = pivot.head(int(top_k)).copy()
        tail = pivot.iloc[int(top_k) :]
        other = pd.DataFrame(
            {
                "rejected": [float(tail["rejected"].sum())],
                "failed": [float(tail["failed"].sum())],
                "total": [float(tail["total"].sum())],
            },
            index=["other"],
        )
        pivot = pd.concat([head, other], axis=0)
    return pivot


def _build_run_health_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes | None]]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError("run_health requires attempts.parquet.")
    required = {"status", "reason", "plan_name"}
    missing = required - set(attempts_df.columns)
    if missing:
        raise ValueError(f"attempts.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    attempts_df = _normalize_and_order_attempts(attempts_df)
    statuses = ["ok", "rejected", "duplicate", "failed"]
    status_labels = {"ok": "accepted", "rejected": "rejected", "duplicate": "duplicate", "failed": "failed"}
    status_colors = {
        "ok": "#009E73",
        "rejected": "#E69F00",
        "duplicate": "#56B4E9",
        "failed": "#D55E00",
    }
    progress = _progress_axis(attempts_df, max_points=max(1, len(attempts_df) + 1))
    solver_x, solver_x_label = _solver_progress_x(attempts_df)
    legend_size = float(style.get("legend_size", style.get("font_size", 13) * 0.74))
    fig_size = style.get("run_health_figsize")
    if fig_size is None:
        fig_size = (13.5, 7.2)
    fig = plt.figure(figsize=(float(fig_size[0]), float(fig_size[1])), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[0.78, 0.78, 1.0])
    ax_outcome = fig.add_subplot(gs[0, :])
    ax_dup = fig.add_subplot(gs[1, :], sharex=ax_outcome)
    ax_fail = fig.add_subplot(gs[2, 0])
    ax_plan = fig.add_subplot(gs[2, 1])

    normalized_plan_series = attempts_df["plan_name"].map(_normalize_plan_name)
    if normalized_plan_series.isna().all():
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = "all plans"
    else:
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = normalized_plan_series.fillna("all plans")

    plan_names_unique = sorted(set(attempts_df["plan_name"].astype(str).tolist()))
    quota_map = dict(plan_quotas or {})
    if quota_map:
        plan_names = [name for name in quota_map.keys() if name in plan_names_unique]
        plan_names.extend([name for name in plan_names_unique if name not in set(plan_names)])
    else:
        plan_names = plan_names_unique
    if not plan_names:
        plan_names = ["all plans"]
        attempts_df["plan_name"] = "all plans"
    missing_quota = [plan for plan in plan_names if int(quota_map.get(plan, 0)) <= 0]
    if missing_quota:
        raise ValueError(
            "run_health requires generation.plan quotas for all plans in attempts. "
            f"Missing or invalid quota for: {', '.join(sorted(missing_quota))}"
        )

    show_statuses = statuses

    plan_to_row = {name: i for i, name in enumerate(plan_names)}
    attempts_df = attempts_df.copy()
    attempts_df["_plan_row"] = attempts_df["plan_name"].astype(str).map(plan_to_row).fillna(0).astype(float)
    attempts_df["_solver_x"] = solver_x.to_numpy(dtype=float)
    for status in show_statuses:
        sub = attempts_df[attempts_df["status"] == status]
        color = status_colors[status]
        label = status_labels[status] if status in {"ok", "rejected", "failed"} else "_nolegend_"
        if sub.empty:
            continue
        ax_outcome.scatter(
            sub["_solver_x"].to_numpy(dtype=float),
            sub["_plan_row"].to_numpy(dtype=float),
            s=10.0,
            marker="s",
            linewidths=0.32,
            edgecolors="#1f1f1f",
            color=color,
            label=label,
            zorder=3,
        )
    for row in range(len(plan_names) - 1):
        ax_outcome.axhline(row + 0.5, color="#d8d8d8", linewidth=0.7, alpha=0.6, zorder=1)
    ax_outcome.set_yticks(np.arange(len(plan_names), dtype=float))
    ax_outcome.set_yticklabels([_ellipsize(name, max_len=24) for name in plan_names])
    ax_outcome.set_ylim(-0.5, float(len(plan_names)) - 0.5)
    ax_outcome.set_title("Solver outcomes across plan rows")
    _subtitle(
        ax_outcome,
        "Solver outcomes by step for each subsampled plan.",
        fontsize=max(8.0, legend_size * 0.92),
    )
    ax_outcome.legend(
        loc="center right",
        frameon=True,
        fontsize=legend_size,
        ncol=1,
        borderaxespad=0.3,
    )
    ax_outcome.tick_params(axis="x", labelbottom=True, bottom=True)
    ax_outcome.set_xlabel(solver_x_label)
    ax_outcome.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)

    counts_by_step = (
        attempts_df.assign(_solver_x=solver_x.to_numpy(dtype=int))
        .groupby(["_solver_x", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=statuses, fill_value=0)
        .sort_index()
    )
    rates = _rate_series_from_counts(counts_by_step)
    rate_x = counts_by_step.index.to_numpy(dtype=float)
    ax_dup.plot(
        rate_x,
        rates["waste"],
        color=status_colors["failed"],
        linewidth=1.6,
        label="waste rate",
    )
    ax_dup.plot(
        rate_x,
        rates["duplicate"],
        color=status_colors["duplicate"],
        linewidth=1.1,
        linestyle="--",
        label="duplicate rate",
    )
    ax_dup.set_ylim(0.0, 1.0)
    ax_dup.set_ylabel("Rate")
    ax_dup.set_title("Waste prevalence over solver sequence")
    _subtitle(
        ax_dup,
        "waste = rejected + duplicate + failed per solver step. Dashed line shows duplicate share.",
        fontsize=max(8.0, legend_size * 0.92),
    )

    total_waste = int((attempts_df["status"].isin(["rejected", "duplicate", "failed"])).sum())
    if total_waste == 0:
        ax_dup.text(
            0.5,
            0.5,
            f"No waste observed (ok = {int((attempts_df['status'] == 'ok').sum())}/{len(attempts_df)})",
            transform=ax_dup.transAxes,
            ha="center",
            va="center",
            fontsize=max(8.0, legend_size),
            color="#333333",
        )

    ax_dup.tick_params(axis="x", labelbottom=True, bottom=True)
    ax_dup.set_xlabel(solver_x_label)
    ax_dup.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ticks = _solver_ticks(solver_x.to_numpy(dtype=float))
    if ticks.size > 0:
        ax_outcome.set_xticks(ticks)
        ax_dup.set_xticks(ticks)
    if rate_x.size > 0:
        ax_outcome.set_xlim(float(rate_x.min()) - 0.5, float(rate_x.max()) + 0.5)

    problem = attempts_df[attempts_df["status"].astype(str).isin(["rejected", "failed"])].copy()
    reason_pareto = _aggregate_reason_pareto(problem, top_k=None)
    if reason_pareto.empty:
        ax_fail.text(
            0.5,
            0.5,
            "No rejected/failed reasons (only ok/duplicate)",
            ha="center",
            va="center",
            transform=ax_fail.transAxes,
        )
        ax_fail.set_axis_off()
    else:
        positions = np.arange(len(reason_pareto), dtype=float)
        totals_reason = reason_pareto["total"].to_numpy(dtype=float)
        denominator = max(1.0, float(totals_reason.sum()))
        ax_fail.barh(positions, totals_reason, color="#4c78a8")
        ax_fail.set_yticks(positions)
        ax_fail.set_yticklabels([_ellipsize(item, max_len=28) for item in reason_pareto.index.tolist()])
        ax_fail.invert_yaxis()
        ax_fail.set_xlabel("Count")
        ax_fail.set_title("Rejected/failed reason composition")
        _subtitle(ax_fail, "Failure reasons", fontsize=max(8.0, legend_size * 0.9))
        for y, total in zip(positions, totals_reason.tolist()):
            share = float(total) / denominator
            ax_fail.text(
                float(total) + 0.3,
                float(y),
                f"{int(total)} ({share:.0%})",
                va="center",
                ha="left",
                fontsize=max(8.0, legend_size * 0.9),
                color="#333333",
            )

    max_progress = 0.0
    palette = _palette(style, max(3, len(plan_names)))
    for idx, plan in enumerate(plan_names):
        quota = int(quota_map[plan])
        plan_mask = attempts_df["plan_name"].astype(str) == plan
        accepted_mask = (plan_mask & (attempts_df["status"] == "ok")).astype(int).to_numpy(dtype=int)
        accepted_counts = accepted_mask.astype(float)
        cumulative = np.cumsum(accepted_counts)
        ratio = cumulative / float(max(1, quota))
        max_progress = max(max_progress, float(np.nanmax(ratio)) if ratio.size else 0.0)
        color = palette[idx]
        accepted_final = int(cumulative[-1]) if cumulative.size > 0 else 0
        ax_plan.plot(
            progress.x,
            ratio,
            linewidth=1.6,
            color=color,
            label=f"{_ellipsize(plan, 20)} {accepted_final}/{quota}",
        )
        hit = np.where(ratio >= 1.0)[0]
        if hit.size > 0:
            h = int(hit[0])
            ax_plan.scatter(
                float(progress.x[h]),
                float(ratio[h]),
                s=26.0,
                marker="o",
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
    ax_plan.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--")
    ax_plan.text(
        0.01,
        1.0,
        "quota",
        transform=ax_plan.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=max(8.0, legend_size * 0.88),
        color="#666666",
    )
    ax_plan.set_xlabel("Attempt index")
    ax_plan.set_ylabel("Accepted / quota")
    ax_plan.set_ylim(0.0, max(1.05, max_progress + 0.05))
    ax_plan.set_title("Quota attainment by plan")
    _subtitle(
        ax_plan,
        "Cumulative accepted libraries relative to each plan quota.",
        fontsize=max(8.0, legend_size * 0.9),
    )
    ax_plan.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=max(8.0, legend_size * 0.9),
        borderaxespad=0.0,
        ncol=1,
    )
    ax_plan.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax_fail.set_box_aspect(1.0)
    ax_plan.set_box_aspect(1.0)
    _link_panels_by_ticks(fig, ax_outcome, ax_dup, ticks)

    for ax in [ax_outcome, ax_dup, ax_fail, ax_plan]:
        if ax is not None:
            _apply_style(ax, style)

    fig.suptitle(
        "Run diagnostics: solver-step outcomes, waste prevalence, failure reasons, and quota progress",
        fontsize=float(style.get("title_size", style.get("font_size", 13) * 1.1)),
        y=1.01,
    )
    axes = {"outcome": ax_outcome, "dup": ax_dup, "fail": ax_fail, "plan": ax_plan}
    return fig, axes


def _extract_plan_quotas(cfg: dict | None) -> dict[str, int]:
    if not cfg:
        return {}
    if not isinstance(cfg, dict):
        return {}
    candidate_paths = [
        ("densegen", "generation", "plan"),
        ("config", "densegen", "generation", "plan"),
        ("generation", "plan"),
        ("config", "generation", "plan"),
    ]
    plan_items: list[dict] = []
    for path in candidate_paths:
        node: object = cfg
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, list):
            valid = [
                item
                for item in node
                if isinstance(item, dict)
                and _normalize_plan_name(item.get("name")) is not None
                and str(item.get("quota", "")).strip() != ""
            ]
            if valid:
                plan_items = valid
                break
    quotas: dict[str, int] = {}
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        name = _normalize_plan_name(item.get("name"))
        quota_raw = item.get("quota")
        if name is None:
            continue
        try:
            quota = int(quota_raw)
        except Exception:
            continue
        if quota > 0:
            quotas[name] = quota
    return quotas


def _normalize_plan_name(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    label = str(value).strip()
    if not label:
        return None
    if label.lower() in {"nan", "none"}:
        return None
    return label


def _ellipsize(label: object, max_len: int = 18) -> str:
    text = str(label or "")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _normalize_and_order_attempts(attempts_df: pd.DataFrame) -> pd.DataFrame:
    normalized = attempts_df.copy()
    normalized["status"] = normalized["status"].astype(str).str.strip().str.lower()
    allowed = {"ok", "rejected", "duplicate", "failed"}
    unknown = sorted({status for status in normalized["status"].tolist() if status not in allowed})
    if unknown:
        raise ValueError(
            f"Unknown attempt status values in attempts.parquet. Allowed statuses: {sorted(allowed)}. Found: {unknown}"
        )
    normalized["_row_order"] = np.arange(len(normalized), dtype=int)
    normalized["created_at"] = pd.to_datetime(normalized.get("created_at"), errors="coerce")
    if normalized["created_at"].notna().any():
        normalized = normalized.sort_values(["created_at", "_row_order"], kind="mergesort")
    else:
        normalized = normalized.sort_values(["_row_order"], kind="mergesort")
    normalized = normalized.reset_index(drop=True)
    normalized["plan_name"] = normalized["plan_name"].map(_normalize_plan_name).fillna("all plans")
    normalized["run_order"] = np.arange(1, len(normalized) + 1, dtype=int)
    return normalized


def _forbidden_kmer_tokens(value: str) -> list[str]:
    tokens: set[str] = set()
    json_match = re.search(r"\{.*\}", value)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                single = payload.get("forbidden_kmer")
                if isinstance(single, str) and single.strip():
                    tokens.add(single.strip().upper())
                multi = payload.get("forbidden_kmers")
                if isinstance(multi, list):
                    for item in multi:
                        if isinstance(item, str) and item.strip():
                            tokens.add(item.strip().upper())
        except Exception:
            pass
    for match in re.findall(r'"forbidden_kmer"\s*:\s*"([acgtun]+)"', value):
        tokens.add(match.upper())
    list_match = re.search(r'"forbidden_kmers"\s*:\s*\[([^\]]*)\]', value)
    if list_match:
        for match in re.findall(r'"([acgtun]+)"', list_match.group(1)):
            tokens.add(match.upper())
    for match in re.findall(r"forbidden_kmer(?:[:=]|_)?([acgtun]+)", value):
        tokens.add(match.upper())
    return sorted(tokens)


def _reason_family_label(status: str, reason: object) -> str:
    value = str(reason or "").strip().lower()
    if status == "duplicate" or value == "output_duplicate":
        return "duplicate output"
    if value in {"", "none", "nan"}:
        return "unknown"
    replacements = {
        "postprocess_forbidden_kmer": "forbidden kmer",
        "stall_no_solution": "no solution",
        "no_solution": "no solution",
        "failed_required_regulators": "required regulators",
        "failed_min_count_by_regulator": "min by regulator",
        "failed_min_count_per_tf": "min per TF",
        "failed_min_required_regulators": "min regulator groups",
    }
    if value in replacements:
        return replacements[value]
    if "forbidden_kmer" in value:
        tokens = _forbidden_kmer_tokens(value)
        if len(tokens) == 1:
            return f"forbidden kmer: {tokens[0]}"
        if len(tokens) > 1:
            return f"forbidden kmers: {', '.join(tokens)}"
        return "forbidden kmer"
    if "no_solution" in value:
        return "no solution"
    if "required_regulator" in value:
        return "required regulators"
    if "min_count_by_regulator" in value:
        return "min by regulator"
    if "min_count_per_tf" in value:
        return "min per TF"
    if "min_required_regulators" in value:
        return "min regulator groups"
    if "solver" in value:
        return "solver failure"
    return value.replace("_", " ")
