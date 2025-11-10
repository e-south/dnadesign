"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/synergy_scatter.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # unused
    metric_id: Optional[str] = None,
    evaluators: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
) -> None:
    if not metric_id:
        raise RuntimeError("synergy_scatter: metric_id is required")
    observed_col = f"permuter__metric__{metric_id}"

    # Expected column name is determined by the singles metric used to build combos
    # i.e., CombineAA emitted expected__<singles_metric_id>
    expected_cols = [c for c in all_df.columns if c.startswith("permuter__expected__")]
    if not expected_cols:
        raise RuntimeError(
            "synergy_scatter: expected additive column not found "
            "(permuter__expected__<singles_metric_id>). "
            "Ensure variants were generated with the 'combine_aa' protocol v0.1+ and that "
            "you ran 'permuter evaluate' to populate the observed metric."
        )
    if observed_col not in all_df.columns:
        raise RuntimeError(
            f"synergy_scatter: observed metric column not found: {observed_col}"
        )

    # Prefer round-2 variants
    df = all_df.copy()
    if "permuter__round" in df.columns and (df["permuter__round"] == 2).any():
        df = df[df["permuter__round"] == 2]

    # Use the only expected column if multiple present
    expected_col = expected_cols[0]
    df = df[[observed_col, expected_col]].dropna().copy()
    if df.empty:
        raise RuntimeError(
            "synergy_scatter: no rows with both observed and expected values"
        )

    obs = df[observed_col].astype(float).to_numpy()
    exp = df[expected_col].astype(float).to_numpy()
    delta = obs - exp

    fs = float(font_scale) if font_scale else 1.0
    width, height = figsize if figsize else (6.6, 6.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    # y=x diagonal
    lim_min = float(min(np.min(obs), np.min(exp)))
    lim_max = float(max(np.max(obs), np.max(exp)))
    if lim_min == lim_max:
        lim_min, lim_max = lim_min - 1e-6, lim_max + 1e-6
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max], color="0.80", linewidth=1.2, zorder=0
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Scatter colored by Δ
    sc = ax.scatter(
        exp, obs, c=delta, s=22, alpha=0.70, cmap="coolwarm", edgecolors="none"
    )

    # Stats
    try:
        pr, _ = pearsonr(exp, obs)
    except Exception:
        pr = np.nan
    try:
        sr, _ = spearmanr(exp, obs)
    except Exception:
        sr = np.nan
    rmse = float(np.sqrt(np.mean((obs - exp) ** 2)))

    text = f"Pearson r={pr:.3f}\nSpearman ρ={sr:.3f}\nRMSE={rmse:.3f}"
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=int(round(9.5 * fs)),
        bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.8, pad=5.0),
    )

    # Labels
    ax.set_xlabel("Expected additive (from singles)", fontsize=int(round(11 * fs)))
    ax.set_ylabel(f"Observed {metric_id}", fontsize=int(round(11 * fs)))
    ax.tick_params(labelsize=int(round(10 * fs)))
    ax.grid(True, color="0.92", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title + subtitle
    title = f"{job_name}"
    fig.suptitle(title, fontsize=int(round(13 * fs)))
    if evaluators:
        fig.text(
            0.5,
            0.96,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs)),
            alpha=0.75,
        )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.84)
    cbar.set_label("Δ = observed − expected", rotation=90, fontsize=int(round(10 * fs)))
    cbar.ax.tick_params(labelsize=int(round(9 * fs)))

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
