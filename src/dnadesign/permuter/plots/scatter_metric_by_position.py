"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/scatter_metric_by_position.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def _extract_position(modifications: List[str] | str) -> int:
    mods = modifications if isinstance(modifications, list) else [modifications]
    for token in mods:
        digits = "".join(ch for ch in str(token) if ch.isdigit())
        if digits:
            return int(digits)
    return 0


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
) -> None:
    """
    Round-1 only: scatter of all scores per position (light gray),
    overlaid with mean±SD (dark gray), with no marker edges.
    """
    # 1) filter to round 1
    df = all_df[all_df["round"] == 1].copy()
    if df.empty:
        raise RuntimeError(f"{job_name}: no round-1 variants to plot")
    df["position"] = df["modifications"].apply(_extract_position)

    # 2) compute stats per position
    stats = (
        df.groupby("position")["score"]
        .agg(mean="mean", sd="std")
        .reset_index()
        .sort_values("position")
    )

    # 3) grid style
    mpl.rcParams.update(
        {
            "axes.axisbelow": True,
            "grid.color": "0.9",
            "grid.linestyle": "-",  # solid
            "grid.linewidth": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)

    # 4) all individual points (no edges)
    ax.scatter(
        df["position"],
        df["score"],
        color="lightgray",
        alpha=0.3,
        s=10,
        edgecolors="none",  # remove edges
        label="_nolegend_",
    )

    # 5) overlay mean ± SD (no marker edges)
    ax.errorbar(
        stats["position"],
        stats["mean"],
        yerr=stats["sd"].fillna(0),
        fmt="o-",
        color="gray",
        alpha=0.5,
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor="none",  # remove marker edge
        label="mean ± SD",
    )

    # 6) single compact title
    ref_name = df["ref_name"].iloc[0]
    ax.set_title(f"{job_name}_{ref_name}", fontsize=12, pad=6)

    # 7) labels
    ax.set_xlabel("Sequence position")
    ax.set_ylabel(df["score_type"].iloc[0].replace("_", " ").title())

    # 8) legend
    ax.legend(frameon=False, fontsize=8, loc="best")

    # 9) finalize
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
