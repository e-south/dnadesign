"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/bar_metric_by_mutation_count.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
) -> None:
    """
    Scatter of every variant's score (y) vs. variant rank (x),
    colored by number of mutations (categorical, no colorbar).
    Earlier rounds (fewer mutations) are drawn last, so they appear on top.
    """
    df = all_df.copy()
    # count mutations from the length of the modifications list
    df["mut_count"] = df["modifications"].apply(
        lambda m: len(m) if isinstance(m, list) else 1
    )
    # sort by descending score → assign rank
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # build a discrete colormap / legend
    mut_levels = sorted(df["mut_count"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {lvl: cmap(i % 10) for i, lvl in enumerate(mut_levels)}

    fig, ax = plt.subplots(figsize=(6, 4))
    # plot in reverse order: highest mut_count first, lowest last
    for lvl in sorted(mut_levels, reverse=True):
        sub = df[df["mut_count"] == lvl]
        ax.scatter(
            sub["rank"],
            sub["score"],
            color=color_map[lvl],
            s=20,
            alpha=0.5,  # increased opacity
            edgecolors="none",  # remove marker edges
            label=f"{lvl} mut{'s' if lvl != 1 else ''}",
        )

    # labels & title
    ax.set_xlabel("Variant rank (by score)", fontsize=9)
    ax.set_ylabel(all_df["score_type"].iloc[0].replace("_", " ").title(), fontsize=9)
    ax.set_title(f"{job_name} ({all_df['ref_name'].iloc[0]})", fontsize=10)

    # tidy up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=0.5)

    # legend as categorical
    ax.legend(frameon=False, title="Mutation count", fontsize=8, title_fontsize=9)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
