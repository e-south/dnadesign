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
    dpi: int,
    png_compress_level: int,
) -> None:
    if nn_df is None or nn_df.empty or "nn_dist" not in nn_df.columns:
        raise ValueError("Nearest-neighbor distance table is empty or missing nn_dist.")
    nn_vals = pd.to_numeric(nn_df["nn_dist"], errors="coerce").dropna()
    if nn_vals.empty:
        raise ValueError("Nearest-neighbor distance table has no finite nn_dist values.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(nn_vals, bins=20, color="#4c78a8", alpha=0.8, edgecolor="white")
    ax.set_title("Elite nearest-neighbor distance")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
