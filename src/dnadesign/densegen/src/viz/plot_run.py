"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run.py

Run-level plotting for placements, TFBS usage, and run health diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
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
)
from .plot_run_helpers import (
    _usage_category_label as _usage_category_label_helper,
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

_build_tfbs_usage_breakdown_figure = _build_tfbs_usage_breakdown_figure_panel
plot_tfbs_usage = plot_tfbs_usage_panel
_build_run_health_compression_ratio_figure = _build_run_health_compression_ratio_figure_panel
_build_run_health_tfbs_length_by_regulator_figure = _build_run_health_tfbs_length_by_regulator_figure_panel
_usage_category_label = _usage_category_label_helper

_PLAN_MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h")


def _capitalize_first(text: str) -> str:
    token = str(text)
    for idx, ch in enumerate(token):
        if ch.isalpha():
            return token[:idx] + ch.upper() + token[idx + 1 :]
    return token


def _plan_markers(plan_names: list[str]) -> dict[str, str]:
    return {plan: _PLAN_MARKER_CYCLE[idx % len(_PLAN_MARKER_CYCLE)] for idx, plan in enumerate(plan_names)}


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
        fig_height = max(4.2, min(12.0, 0.28 * float(len(plan_names)) + 2.8))
        fig_size = (13.6, fig_height)
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
    ax.set_yticklabels([_capitalize_first(_ellipsize(name, max_len=24)) for name in plan_names])
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
            label="Accepted",
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
            label="Rejected",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="#D55E00",
            markersize=6.5,
            label="Failed",
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
        fig_height = max(5.2, min(10.5, 0.18 * float(len(plan_names)) + 4.6))
        fig_size = (11.2, fig_height)
    fig, (ax_fail, ax_plan) = plt.subplots(
        1,
        2,
        figsize=(float(fig_size[0]), float(fig_size[1])),
        constrained_layout=False,
    )

    plan_palette = _palette(_style_cfg, max(3, len(plan_names)))
    plan_colors = {plan: plan_palette[idx] for idx, plan in enumerate(plan_names)}
    plan_markers = _plan_markers(plan_names)
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
        ax_fail.set_yticklabels(
            [_capitalize_first(_ellipsize(item, max_len=30)) for item in reason_counts.index.tolist()]
        )
        ax_fail.invert_yaxis()
        x_pad = 5.0
        y_pad = 0.2
        max_count = float(np.nanmax(counts_reason)) if counts_reason.size > 0 else 0.0
        ax_fail.set_xlim(0.0, max_count + x_pad)
        ax_fail.set_ylim(float(len(reason_counts)) - 0.5 + y_pad, -0.5 - y_pad)
        ax_fail.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))
        ax_fail.set_xlabel("Failed solve count")
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
        marker = plan_markers[plan]
        plan_mask_values = plan_mask.to_numpy(dtype=bool)
        accepted_final = int(cumulative[-1]) if cumulative.size > 0 else 0
        ax_plan.plot(
            progress.x,
            ratio,
            linewidth=1.6,
            color=color,
            label=f"{_ellipsize(plan, 20)} ({accepted_final}/{quota})",
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
                label=_ellipsize(plan, max_len=20),
            )
        )
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=max(1, min(3, len(plan_names))),
        frameon=False,
        fontsize=float(_style_cfg.get("label_size", _style_cfg.get("font_size", 13))),
    )
    legend_cols = max(1, min(3, len(plan_names)))
    legend_rows = max(1, int(np.ceil(float(len(plan_names)) / float(legend_cols))))
    bottom_pad = min(0.34, 0.08 + 0.045 * legend_rows)
    fig.tight_layout(rect=(0.0, bottom_pad, 1.0, 0.97))
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
    fig_tfbs_length, _axes_tfbs_length = _build_run_health_tfbs_length_by_regulator_figure(
        composition_df if composition_df is not None else pd.DataFrame(),
        library_members_df=library_members_df,
        style=style,
    )
    target_dir = out_path.parent / "run_health"
    target_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = target_dir / f"outcomes_over_time{out_path.suffix}"
    run_health_path = target_dir / f"run_health{out_path.suffix}"
    compression_path = target_dir / f"compression_ratio_distribution{out_path.suffix}"
    tfbs_length_path = target_dir / f"tfbs_length_by_regulator{out_path.suffix}"
    legacy_detail_path = target_dir / f"run_health_detail{out_path.suffix}"
    legacy_detail_path.unlink(missing_ok=True)
    _save_figure(fig_outcome, outcomes_path, style=style)
    _save_figure(fig_detail, run_health_path, style=style)
    _save_figure(fig_compression, compression_path, style=style)
    _save_figure(fig_tfbs_length, tfbs_length_path, style=style)
    plt.close(fig_outcome)
    plt.close(fig_detail)
    plt.close(fig_compression)
    plt.close(fig_tfbs_length)
    summary_df = _run_health_summary_frame(_normalize_and_order_attempts(attempts_df), plan_quotas=plan_quotas)
    summary_df.to_csv(target_dir / "summary.csv", index=False)
    summary_table_path = target_dir / f"summary_table{out_path.suffix}"
    _render_run_health_summary_table_figure(summary_df, summary_table_path, style=style)
    return [outcomes_path, run_health_path, compression_path, tfbs_length_path, summary_table_path]


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


def _render_run_health_summary_table_figure(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> None:
    _style_cfg = _style(style)
    display = summary_df.copy()
    if "value" in display.columns:
        display["value"] = display["value"].map(
            lambda value: f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    fig_width = min(22.0, max(10.0, 2.2 * len(display.columns) + 1.7))
    fig_height = min(22.0, max(3.0, 0.52 * max(1, len(display)) + 1.2))
    fig, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)), constrained_layout=False)
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=[str(col) for col in display.columns],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table_font = max(11.0, float(_style_cfg.get("tick_size", _style_cfg.get("font_size", 13.0) * 0.78)))
    table.set_fontsize(table_font)
    table.scale(1.03, 1.28)
    save_style = dict(_style_cfg)
    save_style["save_pad_inches"] = min(float(save_style.get("save_pad_inches", 0.08)), 0.02)
    _save_figure(fig, out_path, style=save_style)
    plt.close(fig)


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
    def _has_target_value(value: object | None) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        text = str(value).strip().lower()
        return text not in {"", "none", "nan"}

    def _plan_target(item: dict) -> object | None:
        quota_raw = item.get("quota")
        if _has_target_value(quota_raw):
            return quota_raw
        sequences_raw = item.get("sequences")
        if _has_target_value(sequences_raw):
            return sequences_raw
        return None

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
                and _plan_target(item) is not None
            ]
            if valid:
                plan_items = valid
                break
    quotas: dict[str, int] = {}
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        name = _normalize_plan_name(item.get("name"))
        quota_raw = _plan_target(item)
        if name is None:
            continue
        try:
            quota = int(quota_raw)
        except Exception:
            continue
        if quota > 0:
            quotas[name] = quota
    return quotas


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
