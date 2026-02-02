"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run.py

Run-level plotting for placements, TFBS usage, and run health diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_common import _apply_style, _safe_filename, _style


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


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    seq = str(seq).upper()
    gc = sum(1 for ch in seq if ch in {"G", "C"})
    return float(gc) / float(len(seq))


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
    for (input_name, plan_name), sub in composition_df.groupby(["input_name", "plan_name"]):
        counts = sub.groupby(["tf", "tfbs"]).size().sort_values(ascending=False)
        if counts.empty:
            raise ValueError(f"tfbs_usage found no TFBS counts for {input_name}/{plan_name}.")
        ranks = np.arange(1, len(counts) + 1)
        values = counts.to_numpy(dtype=float)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=style["figsize"])
        ax_left.plot(ranks, values, color="#4c78a8", linewidth=1.8)
        ax_left.set_xlabel("TFBS rank (by usage)")
        ax_left.set_ylabel("Usage count")
        ax_left.set_title("TFBS rank-frequency")

        ax_right.hist(values, bins="fd", color="#4c78a8", alpha=0.75)
        ax_right.set_xlabel("Usage count")
        ax_right.set_ylabel("Count")
        ax_right.set_title("TFBS usage distribution")
        ax_ecdf = ax_right.twinx()
        sorted_vals = np.sort(values)
        ax_ecdf.plot(sorted_vals, np.arange(1, len(sorted_vals) + 1) / len(sorted_vals), color="#f28e2b")
        ax_ecdf.set_ylabel("ECDF")

        annotations = []
        used_unique = int(len(counts))
        if pools and input_name in pools:
            pool_df = pools[input_name]
            if "tfbs_sequence" in pool_df.columns:
                tfbs_col = "tfbs_sequence"
            else:
                tfbs_col = "tfbs"
            pool_unique = pool_df.assign(tf=pool_df["tf"].astype(str), tfbs=pool_df[tfbs_col].astype(str))
            pool_unique = pool_unique.drop_duplicates(subset=["tf", "tfbs"])
            pool_count = int(len(pool_unique))
            if pool_count > 0:
                annotations.append(f"pool used >=1: {used_unique}/{pool_count} ({used_unique / pool_count:.2%})")
        if library_members_df is not None and not library_members_df.empty:
            offered = library_members_df[
                (library_members_df["input_name"].astype(str) == str(input_name))
                & (library_members_df["plan_name"].astype(str) == str(plan_name))
            ]
            if not offered.empty:
                offered_unique = offered.drop_duplicates(subset=["tf", "tfbs"])
                offered_count = int(len(offered_unique))
                if offered_count > 0:
                    annotations.append(
                        f"offered used >=1: {used_unique}/{offered_count} ({used_unique / offered_count:.2%})"
                    )
        if annotations:
            ax_right.text(0.98, 0.98, "\n".join(annotations), transform=ax_right.transAxes, ha="right", va="top")
        _apply_style(ax_left, style)
        _apply_style(ax_right, style)
        fig.tight_layout()
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_run_health(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> None:
    fig, _axes = _build_run_health_figure(attempts_df, events_df=events_df, style=style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_run_health_figure(
    attempts_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> tuple[plt.Figure, dict[str, plt.Axes | None]]:
    if attempts_df is None or attempts_df.empty:
        raise ValueError("run_health requires attempts.parquet.")
    required = {"attempt_index", "status", "reason", "plan_name"}
    missing = required - set(attempts_df.columns)
    if missing:
        raise ValueError(f"attempts.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    attempts_df = attempts_df.copy()
    attempts_df["attempt_index"] = pd.to_numeric(attempts_df["attempt_index"], errors="coerce").fillna(0).astype(int)
    attempts_df["created_at"] = pd.to_datetime(attempts_df.get("created_at"), errors="coerce")
    statuses = ["success", "duplicate", "failed"]
    plan_names = sorted({str(p) for p in attempts_df["plan_name"].astype(str).tolist() if str(p).strip()})
    show_plans = len(plan_names) > 1
    if show_plans:
        fig, axes = plt.subplots(2, 2, figsize=style["figsize"])
        ax_outcome, ax_dup, ax_fail, ax_plan = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=style["figsize"])
        ax_outcome, ax_dup, ax_fail = axes
        ax_plan = None

    bins = _resolution_bins(ax_outcome, len(attempts_df))
    edges, centers = _bin_attempts(attempts_df["attempt_index"].to_numpy(dtype=float), bins)
    if centers.size == 0:
        raise ValueError("run_health cannot bin attempts without attempt_index values.")
    counts_by_status: dict[str, np.ndarray] = {}
    for status in statuses:
        sub = attempts_df[attempts_df["status"].astype(str) == status]
        counts, _ = np.histogram(sub["attempt_index"].to_numpy(dtype=float), bins=edges)
        counts_by_status[status] = counts.astype(float)

    ax_outcome.stackplot(centers, [counts_by_status[s] for s in statuses], labels=statuses)
    ax_outcome.set_xlabel("Attempt index (binned)")
    ax_outcome.set_ylabel("Count")
    ax_outcome.set_title("Outcome mix")
    ax_outcome.legend(loc="upper right", frameon=bool(style.get("legend_frame", False)))

    totals = sum(counts_by_status.values())
    dup_counts = counts_by_status.get("duplicate", np.zeros_like(totals))
    dup_rate = np.divide(dup_counts, np.where(totals > 0, totals, 1.0))
    if dup_counts.sum() <= 0 or np.all(np.isnan(dup_rate)):
        ax_dup.text(0.5, 0.5, "No duplicates observed", ha="center", va="center", transform=ax_dup.transAxes)
        ax_dup.set_axis_off()
    else:
        ax_dup.plot(centers, dup_rate, color="#e15759")
        ax_dup.set_xlabel("Attempt index (binned)")
        ax_dup.set_ylabel("Duplicate rate")
        ax_dup.set_ylim(0.0, min(1.0, max(0.05, float(np.nanmax(dup_rate)) + 0.05)))
        ax_dup.set_title("Duplicate pressure")

    failed = attempts_df[attempts_df["status"].astype(str) == "failed"]
    if failed.empty:
        ax_fail.text(0.5, 0.5, "No failures", ha="center", va="center", transform=ax_fail.transAxes)
        ax_fail.set_axis_off()
    else:
        reason_counts = failed["reason"].astype(str).value_counts()
        if len(reason_counts) <= 8:
            positions = np.arange(len(reason_counts))
            ax_fail.bar(positions, reason_counts.values.tolist(), color="#4c78a8")
            ax_fail.set_xticks(positions)
            ax_fail.set_xticklabels(reason_counts.index.tolist(), rotation=45, ha="right")
            ax_fail.set_ylabel("Count")
            ax_fail.set_title("Failure reasons")
        else:
            ranks = np.arange(1, len(reason_counts) + 1)
            ax_fail.plot(ranks, reason_counts.values, color="#4c78a8", linewidth=1.6)
            ax_fail.set_xlabel("Reason rank")
            ax_fail.set_ylabel("Count")
            ax_fail.set_title("Failure rank-frequency")

    if show_plans and ax_plan is not None:
        for plan in plan_names:
            sub = attempts_df[(attempts_df["plan_name"].astype(str) == plan) & (attempts_df["status"] == "success")]
            counts, _ = np.histogram(sub["attempt_index"].to_numpy(dtype=float), bins=edges)
            cumulative = np.cumsum(counts)
            ax_plan.plot(centers, cumulative, label=plan)
        ax_plan.set_xlabel("Attempt index (binned)")
        ax_plan.set_ylabel("Cumulative successes")
        ax_plan.set_title("Plan progress")
        ax_plan.legend(loc="upper left", frameon=bool(style.get("legend_frame", False)))

    if events_df is not None and not events_df.empty and "created_at" in attempts_df.columns:
        event_times = pd.to_datetime(events_df.get("created_at"), errors="coerce").dropna()
        if not event_times.empty:
            attempt_times = attempts_df.dropna(subset=["created_at"]).sort_values("created_at")
            if not attempt_times.empty:
                idx_values = attempt_times["attempt_index"].to_numpy()
                time_values = attempt_times["created_at"].to_numpy()
                for evt_time in event_times.to_numpy():
                    insert = np.searchsorted(time_values, evt_time)
                    if insert <= 0:
                        event_idx = idx_values[0]
                    elif insert >= len(idx_values):
                        event_idx = idx_values[-1]
                    else:
                        before = time_values[insert - 1]
                        after = time_values[insert]
                        use_prev = (evt_time - before) <= (after - evt_time)
                        event_idx = idx_values[insert - 1] if use_prev else idx_values[insert]
                    ax_outcome.axvline(event_idx, color="#999999", linestyle="--", linewidth=1)

    for ax in [ax_outcome, ax_dup, ax_fail] + ([ax_plan] if ax_plan is not None else []):
        if ax is not None:
            _apply_style(ax, style)

    axes = {"outcome": ax_outcome, "dup": ax_dup, "fail": ax_fail, "plan": ax_plan}
    return fig, axes
