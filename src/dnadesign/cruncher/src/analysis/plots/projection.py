"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/projection.py

Plot a score-space projection for sampled sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _compute_projection(df: pd.DataFrame, tf_names: list[str]) -> pd.DataFrame:
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns for projection: {missing}")
    scores = df[cols].to_numpy(dtype=float)
    min_norm = np.min(scores, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hmean = scores.shape[1] / np.sum(1.0 / scores, axis=1)
    return pd.DataFrame({"min_per_tf_norm": min_norm, "hmean_per_tf_norm": hmean})


def plot_scores_projection(
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    max_points: int,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()
    if df.empty:
        raise ValueError("No draw-phase sequences available for projection plot.")
    proj = _compute_projection(df, tf_list)
    proj = proj.replace([np.inf, -np.inf], np.nan).dropna()
    if proj.empty:
        raise ValueError("Projection plot has no finite score data.")
    idx = _subsample_indices(len(proj), max_points)
    proj_sub = proj.iloc[idx]

    elite_proj = pd.DataFrame()
    if elites_df is not None and not elites_df.empty:
        elite_proj = _compute_projection(elites_df, tf_list)
        elite_proj = elite_proj.replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        proj_sub["min_per_tf_norm"],
        proj_sub["hmean_per_tf_norm"],
        s=14,
        alpha=0.4,
        color="#4c78a8",
        label="draws",
    )
    if not elite_proj.empty:
        ax.scatter(
            elite_proj["min_per_tf_norm"],
            elite_proj["hmean_per_tf_norm"],
            s=40,
            color="#f58518",
            edgecolor="white",
            linewidth=0.6,
            label="elites",
        )
    ax.set_xlabel("Min per-TF norm")
    ax.set_ylabel("Harmonic mean per-TF norm")
    ax.set_title("Score projection")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
