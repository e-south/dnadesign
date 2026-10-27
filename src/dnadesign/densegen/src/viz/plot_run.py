"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run.py

Run-level plotting for placements, TFBS usage, and run health diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from .plot_common import _apply_style, _palette, _save_figure, _style, plan_group_from_name
from .plot_run_health_utils import (
    aggregate_reason_pareto as _aggregate_reason_pareto,
)
from .plot_run_health_utils import (
    link_panels_by_ticks as _link_panels_by_ticks,
)
from .plot_run_health_utils import (
    rate_series_from_counts as _rate_series_from_counts,
)
from .plot_run_health_utils import (
    solver_ticks as _solver_ticks,
)
from .plot_run_health_utils import (
    subtitle as _subtitle,
)
from .plot_run_helpers import (
    _ellipsize,
    _normalize_plan_name,
    _reason_family_label,
    compact_failure_reason_label,
    compact_plan_label,
)
from .plot_run_helpers import (
    _usage_category_label as _usage_category_label_helper,
)
from .plot_run_helpers import (
    capitalize_first as _capitalize_first,
)
from .plot_run_helpers import (
    place_figure_legend_below_xlabel as _place_figure_legend_below_xlabel,
)
from .plot_run_helpers import (
    plan_markers as _plan_markers,
)
from .plot_run_helpers import (
    rename_output_paths as _rename_output_paths,
)
from .plot_run_inputs import (
    load_plan_quotas_from_effective_config as _extract_plan_quotas,
)
from .plot_run_inputs import (
    normalize_and_order_attempts as _normalize_and_order_attempts,
)
from .plot_run_panels import (
    _build_run_health_compression_ratio_figure as _build_run_health_compression_ratio_figure_panel,
)
from .plot_run_panels import (
    _build_run_health_tfbs_length_by_regulator_figure as _build_run_health_tfbs_length_by_regulator_figure_panel,
)
from .plot_run_panels import (
    _build_tfbs_usage_breakdown_figure as _build_tfbs_usage_breakdown_figure_panel,
)
from .plot_run_panels import (
    plot_tfbs_usage as plot_tfbs_usage_panel,
)
from .plot_run_summary import build_run_health_summary_frame as _run_health_summary_frame
from .plot_run_summary import render_run_health_summary_table_figure as _render_run_health_summary_table_figure_impl

_build_tfbs_usage_breakdown_figure = _build_tfbs_usage_breakdown_figure_panel
plot_tfbs_usage = plot_tfbs_usage_panel
_build_run_health_compression_ratio_figure = _build_run_health_compression_ratio_figure_panel
_build_run_health_tfbs_length_by_regulator_figure = _build_run_health_tfbs_length_by_regulator_figure_panel
_usage_category_label = _usage_category_label_helper


def _render_run_health_summary_table_figure(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> None:
    _render_run_health_summary_table_figure_impl(
        summary_df,
        out_path,
        style=style,
        save_figure=_save_figure,
    )


def _outcomes_attempts_per_row_for_workload(max_plan_attempts: int) -> int:
    attempts = max(1, int(max_plan_attempts))
    # Rows wrap within each plan, so pack by the largest per-plan workload instead
    # of the total run size; otherwise multi-plan runs collapse into a thin ribbon.
    if attempts < 1_000:
        return 10
    if attempts < 20_000:
        return 50
    if attempts < 100_000:
        return 100
    if attempts < 1_000_000:
        return 500
    return 1000


def _prepare_run_health_inputs(
    attempts_df: pd.DataFrame,
    *,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
    plot_label: str = "solve_pressure_and_progress",
) -> tuple[dict, pd.DataFrame, list[str], dict[str, int], ProgressAxis, pd.Series, str, float]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError(f"{plot_label} requires attempts.parquet.")
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
    plan_scope = str(style.get("run_health_plan_scope", "auto")).strip().lower()
    if plan_scope not in {"auto", "per_plan", "per_group"}:
        raise ValueError(f"run_health_plan_scope must be one of auto|per_plan|per_group, got {plan_scope!r}")
    try:
        max_labels = max(1, int(style.get("run_health_plan_max_labels", 14)))
    except Exception as exc:
        raise ValueError("run_health_plan_max_labels must be an integer > 0") from exc

    grouped_plan_series = attempts_df["plan_name"].astype(str).map(plan_group_from_name)
    grouped_unique = sorted({name for name in grouped_plan_series.astype(str).tolist() if str(name).strip()})
    should_group = False
    if plan_scope == "per_group":
        should_group = True
    elif plan_scope == "auto":
        should_group = len(plan_names_unique) > max_labels and len(grouped_unique) < len(plan_names_unique)
    if should_group:
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = grouped_plan_series
        if quota_map:
            grouped_quota: dict[str, int] = {}
            for plan_name, quota in quota_map.items():
                grouped_name = plan_group_from_name(str(plan_name))
                grouped_quota[grouped_name] = grouped_quota.get(grouped_name, 0) + int(quota)
            quota_map = grouped_quota
        plan_names_unique = sorted(set(attempts_df["plan_name"].astype(str).tolist()))

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
            f"{plot_label} requires generation.plan quotas for all plans in attempts. "
            f"Missing or invalid quota for: {', '.join(sorted(missing_quota))}"
        )
    return style, attempts_df, plan_names, quota_map, progress, solver_x, solver_x_label, legend_size


def _prepare_run_health_outcomes_inputs(
    attempts_df: pd.DataFrame,
    *,
    style: Optional[dict] = None,
    plot_label: str = "attempt_outcome_timeline",
) -> tuple[dict, pd.DataFrame, list[str], float]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError(f"{plot_label} requires attempts.parquet.")
    required = {"status", "plan_name"}
    missing = required - set(attempts_df.columns)
    if missing:
        raise ValueError(f"attempts.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    attempts_df = _normalize_and_order_attempts(attempts_df)
    normalized_plan_series = attempts_df["plan_name"].map(_normalize_plan_name)
    if normalized_plan_series.isna().all():
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = "all plans"
    else:
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = normalized_plan_series.fillna("all plans")
    plan_names_unique = sorted(set(attempts_df["plan_name"].astype(str).tolist()))
    plan_scope = str(style.get("run_health_outcomes_plan_scope", "per_group")).strip().lower()
    if plan_scope not in {"auto", "per_plan", "per_group"}:
        raise ValueError(f"run_health_outcomes_plan_scope must be one of auto|per_plan|per_group, got {plan_scope!r}")
    try:
        max_labels = max(1, int(style.get("run_health_outcomes_plan_max_labels", 48)))
    except Exception as exc:
        raise ValueError("run_health_outcomes_plan_max_labels must be an integer > 0") from exc
    grouped_plan_series = attempts_df["plan_name"].astype(str).map(plan_group_from_name)
    grouped_unique = sorted({name for name in grouped_plan_series.astype(str).tolist() if str(name).strip()})
    should_group = False
    if plan_scope == "per_group":
        should_group = True
    elif plan_scope == "auto":
        should_group = len(plan_names_unique) > max_labels and len(grouped_unique) < len(plan_names_unique)
    if should_group:
        attempts_df = attempts_df.copy()
        attempts_df["plan_name"] = grouped_plan_series
        plan_names_unique = sorted(set(attempts_df["plan_name"].astype(str).tolist()))
    plan_names = plan_names_unique or ["all plans"]
    if not plan_names:
        attempts_df["plan_name"] = "all plans"
        plan_names = ["all plans"]
    legend_size = float(style.get("legend_size", style.get("font_size", 13) * 0.74))
    return style, attempts_df, plan_names, legend_size


def _build_run_health_outcomes_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
    plot_label: str = "run_health",
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    del events_df
    del plan_quotas
    _style_cfg, attempts_df, plan_names, _legend_size = _prepare_run_health_outcomes_inputs(
        attempts_df,
        style=style,
        plot_label=plot_label,
    )
    plan_counts = attempts_df["plan_name"].astype(str).value_counts()
    max_plan_attempts = int(plan_counts.max()) if not plan_counts.empty else int(len(attempts_df))
    try:
        attempts_per_row_raw = _style_cfg.get("run_health_outcomes_attempts_per_row")
        if attempts_per_row_raw is None:
            attempts_per_row = _outcomes_attempts_per_row_for_workload(max_plan_attempts)
        else:
            attempts_per_row = max(1, int(attempts_per_row_raw))
    except Exception as exc:
        raise ValueError("run_health_outcomes_attempts_per_row must be an integer > 0") from exc

    fig_size = _style_cfg.get("run_health_outcomes_figsize")
    if fig_size is None:
        max_rows_estimate = max(1, int(np.ceil(float(max_plan_attempts) / float(attempts_per_row))))
        panel_width = max(2.3, min(3.6, 0.035 * float(attempts_per_row) + 1.1))
        fig_height = max(5.6, min(13.8, 0.028 * float(max_rows_estimate) + 3.0))
        fig_width = max(6.8, min(22.0, panel_width * float(len(plan_names)) + 2.5))
        fig_size = (fig_width, fig_height)
    fig, axes_grid = plt.subplots(
        1,
        len(plan_names),
        figsize=(float(fig_size[0]), float(fig_size[1])),
        squeeze=False,
        sharey=True,
        constrained_layout=False,
    )
    axes = [axis for axis in axes_grid[0]]
    fig_height = float(fig.get_size_inches()[1])
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.2, top=0.86, wspace=0.06)

    plot_df = attempts_df.copy()
    plot_df["_plan_attempt_rank"] = plot_df.groupby("plan_name", sort=False).cumcount().astype(int) + 1
    plot_df["_attempt_row"] = ((plot_df["_plan_attempt_rank"] - 1) // attempts_per_row + 1).astype(float)
    plot_df["_attempt_slot"] = ((plot_df["_plan_attempt_rank"] - 1) % attempts_per_row).astype(float)
    max_attempt_rank = int(plot_df["_plan_attempt_rank"].max()) if not plot_df.empty else 0
    max_rows = int(plot_df["_attempt_row"].max()) if not plot_df.empty else 1
    status_groups = {"accepted": {"ok", "duplicate"}, "rejected": {"rejected"}, "failed": {"failed"}}
    plan_values = np.asarray(plot_df["plan_name"].astype(str).to_numpy(), dtype=str)
    row_idx = plot_df["_attempt_row"].astype(int).to_numpy() - 1
    slot_idx = plot_df["_attempt_slot"].astype(int).to_numpy()
    status_values = np.asarray(plot_df["status"].astype(str).to_numpy(), dtype=str)
    valid = (row_idx >= 0) & (row_idx < max_rows) & (slot_idx >= 0) & (slot_idx < int(attempts_per_row))
    normalized_status = np.char.lower(np.char.strip(status_values))
    is_rejected = np.isin(normalized_status, list(status_groups["rejected"]))
    is_failed = np.isin(normalized_status, list(status_groups["failed"]))
    is_accepted = ~(is_rejected | is_failed)

    rejected_color = "#D55E00"
    failed_color = "#C62828"
    cmap = ListedColormap(["#ffffff", "#d9d9d9", rejected_color])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)
    try:
        rows_per_block = max(1, int(_style_cfg.get("run_health_outcomes_rows_per_block", 50)))
    except Exception as exc:
        raise ValueError("run_health_outcomes_rows_per_block must be an integer > 0") from exc

    if max_attempt_rank > 0:
        try:
            max_yticks = max(2, int(_style_cfg.get("run_health_outcomes_max_yticks", 12)))
        except Exception as exc:
            raise ValueError("run_health_outcomes_max_yticks must be an integer >= 2") from exc
        precomputed_tick_size = float(
            _style_cfg.get(
                "run_health_outcomes_tick_label_size",
                max(13.0, float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13))) * 1.02),
            )
        )
        max_visible_ticks = max(3, int((fig_height * 72.0) / max(18.0, precomputed_tick_size * 2.6)))
        max_yticks = min(max_yticks, max_visible_ticks)
        step_rows = max(1, int(np.ceil(float(max_rows) / float(max_yticks))))
        tick_rows = list(range(1, max_rows + 1, step_rows))
        if max_rows not in tick_rows:
            tick_rows.append(max_rows)
        tick_rows = sorted(set(tick_rows))
        tick_labels = [f"{((row - 1) * attempts_per_row + 1):,}" for row in tick_rows]
        y_tick_positions = [float(item) - 0.5 for item in tick_rows]
    else:
        y_tick_positions = []
        tick_labels = []

    label_size = float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13)))
    uniform_font_size = max(28.0, label_size * 1.85)
    panel_title_size = uniform_font_size
    tick_label_size = uniform_font_size
    axis_label_size = uniform_font_size
    legend_font_size = uniform_font_size
    local_style = dict(_style_cfg)
    local_style["tick_size"] = uniform_font_size
    local_style["label_size"] = uniform_font_size
    local_style["title_size"] = uniform_font_size
    try:
        x_pad = max(0.0, float(_style_cfg.get("run_health_outcomes_panel_xpad", 0.7)))
        y_pad = max(0.0, float(_style_cfg.get("run_health_outcomes_panel_ypad", 0.6)))
    except Exception as exc:
        raise ValueError("run_health_outcomes panel padding must be numeric >= 0") from exc

    for axis_idx, (ax, plan_name) in enumerate(zip(axes, plan_names)):
        panel_mask = plan_values == str(plan_name)
        panel_valid = valid & panel_mask
        panel_row_idx = row_idx[panel_valid]
        panel_slot_idx = slot_idx[panel_valid]
        panel_grid = np.zeros((max_rows, int(attempts_per_row)), dtype=np.uint8)
        panel_accepted = is_accepted[panel_valid]
        panel_rejected = is_rejected[panel_valid]
        panel_failed = is_failed[panel_valid]
        panel_grid[panel_row_idx[panel_accepted], panel_slot_idx[panel_accepted]] = 1
        panel_grid[panel_row_idx[panel_rejected], panel_slot_idx[panel_rejected]] = 2

        ax.imshow(
            panel_grid,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            extent=(0.0, float(attempts_per_row), float(max_rows), 0.0),
            zorder=2,
            aspect="auto",
        )

        if np.any(panel_failed):
            failed_x = panel_slot_idx[panel_failed].astype(float) + 0.5
            failed_y = panel_row_idx[panel_failed].astype(float) + 0.5
            ax.scatter(
                failed_x,
                failed_y,
                marker="X",
                s=56.0,
                linewidths=0.65,
                facecolors=failed_color,
                edgecolors="#ffffff",
                zorder=4,
                clip_on=False,
            )

        for block_start in range(rows_per_block, max_attempt_rank + 1, rows_per_block):
            boundary_row = int(np.ceil(float(block_start) / float(attempts_per_row)))
            if boundary_row <= int(max_rows):
                ax.axhline(float(boundary_row), color="#ececec", linewidth=0.65, alpha=0.9, zorder=3)

        ax.set_xlim(-x_pad, float(attempts_per_row) + x_pad)
        ax.set_ylim(float(max_rows) + y_pad, -y_pad)
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_title(
            textwrap.fill(compact_plan_label(plan_name), width=18, break_on_hyphens=False),
            pad=8.0,
            fontsize=panel_title_size,
        )
        _apply_style(ax, local_style)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.9)
        if axis_idx == 0:
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels(tick_labels)
            ax.tick_params(axis="y", labelsize=tick_label_size)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    title_size = uniform_font_size
    text_color = str(_style_cfg.get("text_color", "#111111"))
    fig.suptitle("Attempt outcomes by plan", y=0.94, fontsize=title_size, color=text_color)
    fig.supylabel("Attempt index", x=0.018, fontsize=axis_label_size, color=text_color)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="#d9d9d9",
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=8.0,
            label="Accepted",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=rejected_color,
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=8.0,
            label="Rejected",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markerfacecolor=failed_color,
            markeredgecolor="#ffffff",
            markeredgewidth=0.65,
            color=failed_color,
            markersize=9.0,
            label="Failed",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
        fontsize=legend_font_size,
        ncol=3,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(text_color)
    return fig, {"outcome": axes[0]}


def _build_run_health_detail_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
    plot_label: str = "run_health",
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    _style_cfg, attempts_df, plan_names, quota_map, progress, _solver_x, _solver_x_label, legend_size = (
        _prepare_run_health_inputs(
            attempts_df,
            plan_quotas=plan_quotas,
            style=style,
            plot_label=plot_label,
        )
    )
    fig_size = _style_cfg.get("run_health_detail_figsize")
    if fig_size is None:
        fig_height = max(7.8, min(14.2, 0.28 * float(len(plan_names)) + 6.8))
        fig_size = (12.4, fig_height)
    fig, (ax_fail, ax_plan) = plt.subplots(
        1,
        2,
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )

    plan_palette = _palette(_style_cfg, max(3, len(plan_names)))
    plan_colors = {plan: plan_palette[idx] for idx, plan in enumerate(plan_names)}
    plan_markers = _plan_markers(plan_names)
    base_font_size = float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13)))
    uniform_font_size = min(15.0, max(14.0, base_font_size * 1.1))
    problem = attempts_df[attempts_df["status"].astype(str).isin(["rejected", "failed"])].copy()
    reason_label_size: float | None = None
    reason_labels: list[str] = []
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
            lambda row: _reason_family_label(
                str(row.get("status", "")),
                row.get("reason"),
                row.get("detail_json"),
            ),
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
        fig_w, fig_h = fig.get_size_inches()
        fig.set_size_inches(fig_w, max(fig_h, 6.2 + 0.42 * float(len(reason_counts))))
        positions = np.arange(len(reason_counts), dtype=float)
        totals_reason = reason_counts["total"].to_numpy(dtype=float)
        counts_reason = totals_reason
        ax_fail.hlines(
            positions,
            0.0,
            counts_reason,
            color="#4c78a8",
            linewidth=2.0,
            alpha=0.85,
        )
        ax_fail.scatter(
            counts_reason,
            positions,
            s=36.0,
            color="#4c78a8",
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax_fail.set_yticks(positions)
        reason_labels = [compact_failure_reason_label(item) for item in reason_counts.index.tolist()]
        ax_fail.set_yticklabels(reason_labels)
        try:
            reason_label_size_raw = _style_cfg.get("run_health_reason_label_size")
            if reason_label_size_raw is None:
                reason_label_size = uniform_font_size
            else:
                reason_label_size = float(reason_label_size_raw)
            if reason_label_size <= 0:
                raise ValueError
        except Exception as exc:
            raise ValueError("run_health_reason_label_size must be a number > 0") from exc
        ax_fail.invert_yaxis()
        x_pad = 5.0
        y_pad = 0.2
        max_count = float(np.nanmax(counts_reason)) if counts_reason.size > 0 else 0.0
        ax_fail.set_xlim(0.0, max_count + x_pad)
        ax_fail.set_ylim(float(len(reason_counts)) - 0.5 + y_pad, -0.5 - y_pad)
        ax_fail.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))
        ax_fail.set_xlabel("Failed solve count", fontsize=uniform_font_size, labelpad=10.0)
        ax_fail.set_title("Reason for failed solve", pad=10.0, fontsize=uniform_font_size)
        ax_fail.tick_params(axis="y", pad=8.0)

    max_progress = 0.0
    for idx, plan in enumerate(plan_names):
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
        accepted_final = int(cumulative[-1]) if cumulative.size > 0 else 0
        ratio = cumulative / float(max(1, accepted_final))
        max_progress = max(max_progress, float(np.nanmax(ratio)) if ratio.size else 0.0)
        color = plan_colors[plan]
        marker = plan_markers[plan]
        plan_mask_values = plan_mask.to_numpy(dtype=bool)
        ax_plan.plot(
            progress.x,
            ratio,
            linewidth=1.6,
            color=color,
            label=compact_plan_label(plan),
        )
        if progress.mode == "discrete":
            final_indices = np.where(plan_mask_values)[0]
        else:
            if progress.bin_id is None:
                raise ValueError("run_health binned plan progress requires bin_id.")
            plan_presence = (
                pd.DataFrame({"bin_id": progress.bin_id, "has_plan": plan_mask_values.astype(int)})
                .groupby("bin_id")["has_plan"]
                .sum()
                .reindex(np.arange(len(progress.x), dtype=int), fill_value=0)
                .to_numpy(dtype=float)
            )
            final_indices = np.where(plan_presence > 0.0)[0]
        if final_indices.size > 0:
            h = int(final_indices[-1])
            ax_plan.scatter(
                float(progress.x[h]),
                float(ratio[h]),
                s=36.0,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
    ax_plan.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_plan.set_xlabel("Attempt index", fontsize=uniform_font_size, labelpad=10.0)
    ax_plan.set_ylabel("Cumulative accepted / final accepted", fontsize=uniform_font_size, labelpad=12.0)
    ax_plan.set_ylim(0.0, max(1.02, max_progress + 0.02))
    ax_plan.set_title("Accepted progress by plan", pad=10.0, fontsize=uniform_font_size)
    ax_plan.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax_fail.set_box_aspect(1.0)
    ax_plan.set_box_aspect(1.0)

    _apply_style(ax_fail, _style_cfg)
    if reason_label_size is not None:
        for tick in ax_fail.get_yticklabels():
            tick.set_fontsize(float(reason_label_size))
    _apply_style(ax_plan, _style_cfg)
    ax_fail.tick_params(axis="both", labelsize=uniform_font_size)
    ax_plan.tick_params(axis="both", labelsize=uniform_font_size)
    handles = []
    for plan in plan_names:
        marker = plan_markers[plan]
        color = plan_colors[plan]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.6,
                marker=marker,
                markersize=6.4,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=compact_plan_label(plan),
            )
        )
    legend_obj = fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=max(1, len(plan_names)),
        frameon=False,
        fontsize=uniform_font_size,
    )
    max_reason_chars = max((len(label) for label in reason_labels), default=12)
    left_margin = max(0.28, min(0.44, 0.18 + 0.0042 * float(max_reason_chars)))
    legend_rows = int(np.ceil(float(len(handles)) / float(max(1, len(plan_names)))))
    bottom_margin = max(0.22, 0.16 + 0.06 * float(legend_rows))
    fig.subplots_adjust(left=left_margin, right=0.985, bottom=bottom_margin, top=0.93, wspace=0.3)
    _place_figure_legend_below_xlabel(fig, ax_xlabel=ax_plan, legend=legend_obj, gap=0.03, min_bottom=0.024)
    return fig, {"fail": ax_fail, "plan": ax_plan}


def plot_run_health(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    composition_df: pd.DataFrame | None = None,
    library_members_df: pd.DataFrame | None = None,
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
    fig_compression, _axes_compression = _build_run_health_compression_ratio_figure(df, style=style)
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = target_dir / f"outcomes_over_time{out_path.suffix}"
    run_health_path = target_dir / f"run_health{out_path.suffix}"
    compression_path = target_dir / f"compression_ratio_distribution{out_path.suffix}"
    legacy_detail_path = target_dir / f"run_health_detail{out_path.suffix}"
    legacy_tfbs_length_path = target_dir / f"tfbs_length_by_regulator{out_path.suffix}"
    legacy_summary_table_path = target_dir / f"summary_table{out_path.suffix}"
    legacy_detail_path.unlink(missing_ok=True)
    legacy_tfbs_length_path.unlink(missing_ok=True)
    legacy_summary_table_path.unlink(missing_ok=True)
    _save_figure(fig_outcome, outcomes_path, style=style)
    _save_figure(fig_detail, run_health_path, style=style)
    _save_figure(fig_compression, compression_path, style=style)
    plt.close(fig_outcome)
    plt.close(fig_detail)
    plt.close(fig_compression)
    summary_df = _run_health_summary_frame(_normalize_and_order_attempts(attempts_df), plan_quotas=plan_quotas)
    summary_df.to_csv(target_dir / "summary.csv", index=False)
    return [outcomes_path, run_health_path, compression_path]


def plot_tfbs_concentration_profile(
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
    return _rename_output_paths(
        plot_tfbs_usage_panel(
            df,
            out_path,
            composition_df=composition_df,
            pools=pools,
            library_members_df=library_members_df,
            style=style,
            plan_col=plan_col,
            input_col=input_col,
            plot_label="tfbs_concentration_profile",
        ),
        stem="tfbs_concentration_profile",
    )


def plot_attempt_outcome_timeline(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del df
    style = _style(style)
    plan_quotas = _extract_plan_quotas(cfg)
    fig_outcome, _axes_outcome = _build_run_health_outcomes_figure(
        attempts_df,
        events_df=events_df,
        plan_quotas=plan_quotas,
        style=style,
        plot_label="attempt_outcome_timeline",
    )
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = target_dir / f"attempt_outcome_timeline{out_path.suffix}"
    _save_figure(fig_outcome, outcomes_path, style=style)
    plt.close(fig_outcome)
    return [outcomes_path]


def plot_solve_pressure_and_progress(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del df
    style = _style(style)
    plan_quotas = _extract_plan_quotas(cfg)
    fig_detail, _axes_detail = _build_run_health_detail_figure(
        attempts_df,
        events_df=events_df,
        plan_quotas=plan_quotas,
        style=style,
        plot_label="solve_pressure_and_progress",
    )
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = target_dir / f"solve_pressure_and_progress{out_path.suffix}"
    _save_figure(fig_detail, detail_path, style=style)
    plt.close(fig_detail)
    return [detail_path]


def plot_compression_ratio_by_plan(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> list[Path]:
    style = _style(style)
    fig_compression, _axes_compression = _build_run_health_compression_ratio_figure(df, style=style)
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    compression_path = target_dir / f"compression_ratio_by_plan{out_path.suffix}"
    _save_figure(fig_compression, compression_path, style=style)
    plt.close(fig_compression)
    return [compression_path]


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


def _build_run_health_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    plan_quotas: dict[str, int] | None = None,
    style: Optional[dict] = None,
    plot_label: str = "run_health",
) -> tuple[plt.Figure, dict[str, plt.Axes | None]]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError(f"{plot_label} requires attempts.parquet.")
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
            f"{plot_label} requires generation.plan quotas for all plans in attempts. "
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
        ax_fail.set_yticklabels(
            [_capitalize_first(_ellipsize(item, max_len=28)) for item in reason_pareto.index.tolist()]
        )
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
        plan_mask = attempts_df["plan_name"].astype(str) == plan
        accepted_mask = (plan_mask & (attempts_df["status"] == "ok")).astype(int).to_numpy(dtype=int)
        accepted_counts = accepted_mask.astype(float)
        cumulative = np.cumsum(accepted_counts)
        accepted_final = int(cumulative[-1]) if cumulative.size > 0 else 0
        ratio = cumulative / float(max(1, accepted_final))
        max_progress = max(max_progress, float(np.nanmax(ratio)) if ratio.size else 0.0)
        color = palette[idx]
        ax_plan.plot(
            progress.x,
            ratio,
            linewidth=1.6,
            color=color,
            label=f"{_ellipsize(plan, 20)} ({accepted_final} accepted)",
        )
        if ratio.size > 0:
            h = int(len(ratio) - 1)
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
    ax_plan.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_plan.set_xlabel("Attempt index")
    ax_plan.set_ylabel("Cumulative accepted / final accepted")
    ax_plan.set_ylim(0.0, max(1.02, max_progress + 0.02))
    ax_plan.set_title("Accepted progress by plan")
    _subtitle(
        ax_plan,
        "Each plan trace is normalized to its own final accepted count.",
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
