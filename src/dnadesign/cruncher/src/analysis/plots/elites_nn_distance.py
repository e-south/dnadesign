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
import numpy as np
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
    n_elites = int(len(nn_df))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    baseline_vals = None
    if baseline_nn is not None:
        baseline_vals = pd.to_numeric(baseline_nn, errors="coerce").dropna()
    if n_elites < 3 or nn_vals.empty:
        ax.axis("off")
        lines = [
            "Elite NN-distance histogram is not informative for K < 3.",
            f"n_elites={n_elites}",
        ]
        if not nn_vals.empty:
            lines.append(f"min={float(nn_vals.min()):.3f}, median={float(nn_vals.median()):.3f}")
        else:
            lines.append("No finite NN distances available.")
        if baseline_vals is not None and not baseline_vals.empty:
            lines.append(f"random-baseline NN median={float(baseline_vals.median()):.3f}")
        ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=10, color="#444444")
    else:
        nn_min = float(nn_vals.min())
        nn_med = float(nn_vals.median())
        nn_max = float(nn_vals.max())
        if np.isclose(nn_min, nn_max, rtol=0.0, atol=1e-12):
            ax.bar([nn_min], [int(nn_vals.size)], width=0.8, color="#4c78a8", alpha=0.8, edgecolor="white")
            ax.set_xlim(nn_min - 1.0, nn_max + 1.0)
        else:
            bins = min(20, max(5, int(round(np.sqrt(float(nn_vals.size))))))
            ax.hist(nn_vals, bins=bins, color="#4c78a8", alpha=0.8, edgecolor="white", label="elites")
        if baseline_vals is not None and not baseline_vals.empty:
            baseline_median = float(baseline_vals.median())
            ax.axvline(
                baseline_median,
                color="#9a9a9a",
                linestyle="--",
                linewidth=1.5,
                label="random-baseline NN median",
            )

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
