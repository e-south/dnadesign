"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/opt_trajectory.py

Plot causal particle lineage as raw LLR objective over sweep index.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig


def _require_df(df: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name} data is required for opt trajectory plot.")
    return df


def _require_numeric(df: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"{context} missing required column '{column}'.")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{context} column '{column}' must be numeric.")
    return values.astype(float)


def _sample_rows(df: pd.DataFrame, *, stride: int, group_col: str) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df
    if group_col not in df.columns:
        raise ValueError(f"Trajectory points missing required group column '{group_col}'.")
    sampled_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_col, sort=True, dropna=False):
        ordered = group.sort_values("sweep_idx").reset_index(drop=True)
        idx = list(range(0, len(ordered), stride))
        if len(ordered) - 1 not in idx:
            idx.append(len(ordered) - 1)
        sampled_parts.append(ordered.iloc[sorted(set(idx))])
    return pd.concat(sampled_parts).sort_values(["particle_id", "sweep_idx"]).reset_index(drop=True)


def _legend_labels(ax: plt.Axes) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="best")
    return labels


def plot_opt_trajectory(
    *,
    trajectory_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    png_compress_level: int,
    stride: int = 10,
    alpha_min: float = 0.15,
    alpha_max: float = 0.95,
    slot_overlay: bool = False,
) -> dict[str, object]:
    trajectory = _require_df(trajectory_df, name="Trajectory")
    if "particle_id" not in trajectory.columns:
        raise ValueError(
            "Trajectory particle_id not available; rerun with sample.output.save_trace=true and particle tracking."
        )
    particle_values = _require_numeric(trajectory, "particle_id", context="Trajectory")
    if "sweep_idx" in trajectory.columns:
        sweep_values = _require_numeric(trajectory, "sweep_idx", context="Trajectory")
    elif "sweep" in trajectory.columns:
        sweep_values = _require_numeric(trajectory, "sweep", context="Trajectory")
    else:
        raise ValueError("Trajectory points must include sweep_idx for trajectory plotting.")
    raw_llr_values = _require_numeric(trajectory, "raw_llr_objective", context="Trajectory")
    if not isinstance(alpha_min, (int, float)) or not isinstance(alpha_max, (int, float)):
        raise ValueError("Particle alpha bounds must be numeric.")
    alpha_lo = float(alpha_min)
    alpha_hi = float(alpha_max)
    if alpha_lo < 0 or alpha_hi > 1 or alpha_lo > alpha_hi:
        raise ValueError("Particle alpha bounds must satisfy 0 <= min <= max <= 1.")

    plot_df = trajectory.copy()
    plot_df["particle_id"] = particle_values.astype(int)
    plot_df["sweep_idx"] = sweep_values.astype(int)
    plot_df["raw_llr_objective"] = raw_llr_values.astype(float)
    if "slot_id" in plot_df.columns:
        slot_values = pd.to_numeric(plot_df["slot_id"], errors="coerce")
        if slot_values.notna().any() and slot_values.isna().any():
            raise ValueError("Trajectory slot_id must be numeric when provided.")
        plot_df["slot_id"] = slot_values
    else:
        plot_df["slot_id"] = np.nan

    sampled = _sample_rows(plot_df, stride=max(1, int(stride)), group_col="particle_id")
    particle_ids = sorted(int(v) for v in sampled["particle_id"].unique())
    if not particle_ids:
        raise ValueError("Trajectory plot requires at least one particle_id.")

    sweep_float = sampled["sweep_idx"].astype(float)
    sweep_min = float(sweep_float.min())
    sweep_span = float(sweep_float.max() - sweep_min)

    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    cmap = plt.get_cmap("tab20", max(1, len(particle_ids)))
    for idx, particle_id in enumerate(particle_ids):
        particle_df = sampled[sampled["particle_id"].astype(int) == particle_id].sort_values("sweep_idx")
        if particle_df.empty:
            continue
        x = particle_df["sweep_idx"].astype(float).to_numpy()
        y = particle_df["raw_llr_objective"].astype(float).to_numpy()
        sweeps = particle_df["sweep_idx"].astype(float).to_numpy()
        rgb = cmap(idx % cmap.N)
        if x.size >= 2:
            for seg_idx in range(1, x.size):
                if sweep_span <= 0:
                    alpha = 0.60
                else:
                    t = (float(sweeps[seg_idx]) - sweep_min) / sweep_span
                    alpha = alpha_lo + (alpha_hi - alpha_lo) * t
                ax.plot(
                    x[seg_idx - 1 : seg_idx + 1],
                    y[seg_idx - 1 : seg_idx + 1],
                    color=(rgb[0], rgb[1], rgb[2], alpha),
                    linewidth=1.2,
                    zorder=3,
                )
        if sweep_span <= 0:
            alphas = np.full(x.size, 0.60, dtype=float)
        else:
            alphas = alpha_lo + (alpha_hi - alpha_lo) * ((sweeps - sweep_min) / sweep_span)
        colors = np.column_stack(
            [
                np.full(x.size, rgb[0], dtype=float),
                np.full(x.size, rgb[1], dtype=float),
                np.full(x.size, rgb[2], dtype=float),
                np.asarray(alphas, dtype=float),
            ]
        )
        ax.scatter(x, y, s=16, c=colors, edgecolors="none", zorder=4)

    max_particle = max(particle_ids)
    ax.plot([], [], color="#333333", linewidth=1.2, label=f"particle lineage (id=0..{max_particle})")

    if slot_overlay:
        marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        slot_rows = sampled.dropna(subset=["slot_id"]).copy()
        if not slot_rows.empty:
            slot_rows["slot_id"] = slot_rows["slot_id"].astype(int)
            slot_ids = sorted(int(slot) for slot in slot_rows["slot_id"].unique())
            marker_map = {slot: marker_cycle[idx % len(marker_cycle)] for idx, slot in enumerate(slot_ids)}
            for slot_id in slot_ids:
                slot_df = slot_rows[slot_rows["slot_id"].astype(int) == slot_id]
                ax.scatter(
                    slot_df["sweep_idx"].astype(float),
                    slot_df["raw_llr_objective"].astype(float),
                    s=20,
                    marker=marker_map[slot_id],
                    facecolors="none",
                    edgecolors="#222222",
                    linewidth=0.45,
                    alpha=0.35,
                    zorder=5,
                )
            mapping = ", ".join(f"{slot}:{marker}" for slot, marker in marker_map.items())
            ax.text(
                0.02,
                0.02,
                f"slot marker map: {mapping}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#4a4a4a",
            )

    ax.set_xlabel("Sweep index")
    ax.set_ylabel("Raw LLR objective")
    ax.set_title("Optimization trajectory (particle lineage)")
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "particle_raw_llr",
        "legend_labels": legend_labels,
        "particle_count": len(particle_ids),
    }
