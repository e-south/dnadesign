"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/health_panel.py

Render optimizer move diagnostics with MH acceptance and move-mix panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.move_stats import move_stats_frame
from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style

_MH_KINDS = {"B", "M", "L", "W", "I"}
_MOVE_LABELS = {
    "S": "Single-site Gibbs",
    "B": "Block replace",
    "M": "Multi-site replace",
    "L": "Block slide",
    "W": "Block swap",
    "I": "Motif insertion",
}


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#555555")


def _cooling_markers(optimizer_stats: dict[str, object] | None) -> list[int]:
    if not isinstance(optimizer_stats, dict):
        return []
    payload = optimizer_stats.get("mcmc_cooling")
    if not isinstance(payload, dict):
        return []
    kind = str(payload.get("kind") or "").strip().lower()
    if kind != "piecewise":
        return []
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return []
    markers: list[int] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        sweeps = stage.get("sweeps")
        if isinstance(sweeps, (int, float)):
            markers.append(int(sweeps))
    return sorted(set(markers))


def _move_frame(optimizer_stats: dict[str, object] | None) -> pd.DataFrame:
    stats = optimizer_stats.get("move_stats") if isinstance(optimizer_stats, dict) else None
    frame = move_stats_frame(stats, phase=None)
    if frame.empty:
        return pd.DataFrame(columns=["sweep_idx", "chain", "move_kind", "attempted", "accepted"])
    return frame.loc[:, ["sweep_idx", "chain", "move_kind", "attempted", "accepted"]].reset_index(drop=True)


def _with_bins(move_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if move_df.empty:
        out = move_df.copy()
        if "_bin" not in out.columns:
            out["_bin"] = pd.Series(dtype=int)
        if "_bin_center" not in out.columns:
            out["_bin_center"] = pd.Series(dtype=float)
        return out, 1
    min_sweep = int(move_df["sweep_idx"].min())
    max_sweep = int(move_df["sweep_idx"].max())
    span = max_sweep - min_sweep + 1
    bin_width = max(1, int(round(span / 60.0)))
    out = move_df.copy()
    out["_bin"] = ((out["sweep_idx"] - min_sweep) // bin_width).astype(int)
    out["_bin_center"] = min_sweep + (out["_bin"] * bin_width) + (0.5 * bin_width)
    return out, bin_width


def _plot_mh_acceptance(ax: plt.Axes, binned_df: pd.DataFrame) -> bool:
    if "move_kind" not in binned_df.columns:
        _empty_panel(ax, "MH acceptance unavailable (no move stats).")
        return False
    mh_df = binned_df[binned_df["move_kind"].isin(_MH_KINDS)].copy()
    if mh_df.empty:
        _empty_panel(ax, "MH acceptance unavailable (no MH attempts in trace).")
        return False
    grouped = mh_df.groupby(["chain", "_bin", "_bin_center"], as_index=False)[["attempted", "accepted"]].sum().copy()
    grouped = grouped[grouped["attempted"] > 0].copy()
    if grouped.empty:
        _empty_panel(ax, "MH acceptance unavailable (no MH attempts in trace).")
        return False
    grouped["accept_rate"] = grouped["accepted"] / grouped["attempted"].astype(float)
    grouped = grouped.sort_values(["chain", "_bin"])
    for chain, chain_df in grouped.groupby("chain", sort=True):
        ax.plot(
            chain_df["_bin_center"].to_numpy(dtype=float),
            chain_df["accept_rate"].to_numpy(dtype=float),
            color="#a8a8a8",
            linewidth=0.8,
            alpha=0.5,
        )
    by_bin = grouped.groupby("_bin_center")["accept_rate"]
    x = np.array(sorted(by_bin.groups.keys()), dtype=float)
    median = np.array([float(by_bin.get_group(val).median()) for val in x], dtype=float)
    q25 = np.array([float(by_bin.get_group(val).quantile(0.25)) for val in x], dtype=float)
    q75 = np.array([float(by_bin.get_group(val).quantile(0.75)) for val in x], dtype=float)
    ax.fill_between(x, q25, q75, color="#4c78a8", alpha=0.20, linewidth=0)
    ax.plot(x, median, color="#4c78a8", linewidth=1.8, label="Median MH acceptance")
    ax.set_title("MH acceptance over sweeps (B/M/L/W/I moves only)")
    ax.set_ylabel("MH acceptance rate")
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.02,
        0.08,
        "MH excludes single-site Gibbs updates.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#5a5a5a",
    )
    apply_axes_style(ax, ygrid=True, xgrid=False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    return True


def _plot_move_mix(ax: plt.Axes, binned_df: pd.DataFrame) -> None:
    if binned_df.empty or "move_kind" not in binned_df.columns:
        _empty_panel(ax, "Move-mix unavailable")
        return
    mix = binned_df.groupby(["_bin_center", "move_kind"], as_index=False)["attempted"].sum()
    if mix.empty:
        _empty_panel(ax, "Move-mix unavailable")
        return
    pivot = mix.pivot(index="_bin_center", columns="move_kind", values="attempted").fillna(0.0)
    totals = pivot.sum(axis=1)
    valid = totals > 0
    pivot = pivot.loc[valid]
    totals = totals.loc[valid]
    if pivot.empty:
        _empty_panel(ax, "Move-mix unavailable")
        return
    fractions = pivot.div(totals, axis=0)
    x = fractions.index.to_numpy(dtype=float)
    kinds = [str(kind) for kind in fractions.columns]
    labels = [_MOVE_LABELS.get(kind, kind) for kind in kinds]
    y = [fractions[kind].to_numpy(dtype=float) for kind in kinds]
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(kinds))))
    ax.stackplot(x, y, labels=labels, colors=colors, alpha=0.85)
    ax.set_title("Move mix over sweeps")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Move fraction")
    ax.set_ylim(0.0, 1.0)
    apply_axes_style(ax, ygrid=True, xgrid=False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            frameon=False,
            fontsize=9,
            ncol=max(1, len(labels)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )


def _add_vertical_markers(axes: list[plt.Axes], markers: list[int]) -> None:
    if not markers:
        return
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_text = y_min + ((y_max - y_min) * 0.02)
        for idx, marker in enumerate(markers[:-1]):
            ax.axvline(marker, color="#909090", linestyle=":", linewidth=0.8, alpha=0.65)
            if idx == 0:
                ax.text(marker, y_text, "Cooling stage", ha="left", va="bottom", fontsize=8, color="#666666")


def plot_health_panel(
    optimizer_stats: dict[str, object] | None,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> dict[str, object]:
    move_df = _move_frame(optimizer_stats)
    binned_df, _ = _with_bins(move_df)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    ax_accept, ax_mix = axes
    has_mh_windows = _plot_mh_acceptance(ax_accept, binned_df)
    _plot_move_mix(ax_mix, binned_df)
    ax_accept.set_xlabel("")

    markers = _cooling_markers(optimizer_stats)
    _add_vertical_markers([ax_accept, ax_mix], markers)

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    move_kinds_present = sorted({str(kind) for kind in move_df["move_kind"].tolist()}) if not move_df.empty else []
    return {
        "has_mh_windows": bool(has_mh_windows),
        "move_kinds_present": move_kinds_present,
        "cooling_stage_count": len(markers),
        "rows": int(len(move_df)),
    }
