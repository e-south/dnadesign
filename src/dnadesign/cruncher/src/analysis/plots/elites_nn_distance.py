"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/elites_nn_distance.py

Plot nearest-neighbor distance distributions for elite sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig


def plot_elites_nn_distance(
    nn_df: pd.DataFrame,
    out_path: Path,
    *,
    baseline_nn: pd.Series | None = None,
    dpi: int,
    png_compress_level: int,
) -> None:
    if nn_df is None or nn_df.empty or "nn_dist" not in nn_df.columns:
        raise ValueError("Nearest-neighbor distance table is empty or missing nn_dist.")
    nn_vals = pd.to_numeric(nn_df["nn_dist"], errors="coerce").dropna()
    if nn_vals.empty:
        raise ValueError("Nearest-neighbor distance table has no finite nn_dist values.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(nn_vals, bins=20, color="#4c78a8", alpha=0.8, edgecolor="white", label="elites")
    baseline_vals = None
    if baseline_nn is not None:
        baseline_vals = pd.to_numeric(baseline_nn, errors="coerce").dropna()
        if not baseline_vals.empty:
            baseline_median = float(baseline_vals.median())
            ax.axvline(
                baseline_median,
                color="#9a9a9a",
                linestyle="--",
                linewidth=1.5,
                label="baseline median",
            )

    nn_min = float(nn_vals.min())
    nn_med = float(nn_vals.median())
    annotation = f"min={nn_min:.3f}\nmedian={nn_med:.3f}"
    ax.text(
        0.98,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.set_title("Elite nearest-neighbor distance")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    if baseline_vals is not None and not baseline_vals.empty:
        ax.legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
